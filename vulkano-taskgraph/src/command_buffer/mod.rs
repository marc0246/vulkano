//! Recording commands to execute on the device.

#[allow(unused_imports)] // everything is exported for future-proofing
pub use self::commands::{clear::*, copy::*, dynamic_state::*, pipeline::*, sync::*};
use crate::{descriptor_set::GLOBAL_SET, graph::ResourceMap, resource::DeathRow, Id};
use ash::vk;
use std::{any::Any, sync::Arc};
use vulkano::{
    buffer::Buffer,
    command_buffer as raw,
    device::{Device, DeviceOwned},
    image::Image,
    pipeline::{PipelineBindPoint, PipelineLayout},
    VulkanObject,
};

mod commands;

/// A command buffer in the recording state.
///
/// Unlike [the raw `RecordingCommandBuffer`], this type has knowledge of the current task context
/// and can therefore validate resource accesses. (TODO)
///
/// [the raw `RecordingCommandBuffer`]: raw::RecordingCommandBuffer
pub struct RecordingCommandBuffer<'a> {
    inner: &'a mut raw::RecordingCommandBuffer,
    state: &'a mut CommandBufferState,
    accesses: ResourceAccesses<'a>,
    death_row: &'a mut DeathRow,
}

struct ResourceAccesses<'a> {
    resource_map: &'a ResourceMap<'a>,
}

#[derive(Default)]
pub(crate) struct CommandBufferState {
    descriptor_sets_graphics: Option<DescriptorSetState>,
    descriptor_sets_compute: Option<DescriptorSetState>,
}

struct DescriptorSetState {
    pipeline_layout: Arc<PipelineLayout>,
}

impl<'a> RecordingCommandBuffer<'a> {
    pub(crate) unsafe fn new(
        inner: &'a mut raw::RecordingCommandBuffer,
        state: &'a mut CommandBufferState,
        resource_map: &'a ResourceMap<'a>,
        death_row: &'a mut DeathRow,
    ) -> Self {
        RecordingCommandBuffer {
            inner,
            state,
            accesses: ResourceAccesses { resource_map },
            death_row,
        }
    }

    /// Returns the underlying raw command buffer.
    ///
    /// While this method is safe, using the command buffer isn't. You must guarantee that any
    /// subresources you use while recording commands are either accounted for in the [task's
    /// access set], or that those subresources don't require any synchronization (including layout
    /// transitions and queue family ownership transfers), or that no other task is accessing the
    /// subresources at the same time without appropriate synchronization.
    #[inline]
    pub fn as_raw(&mut self) -> &mut raw::RecordingCommandBuffer {
        self.inner
    }

    /// Queues the destruction of the given `object` after the destruction of the command buffer.
    #[inline]
    pub fn destroy_object(&mut self, object: Arc<impl Any + Send + Sync>) {
        self.death_row.push(object);
    }

    /// Queues the destruction of the given `objects` after the destruction of the command buffer.
    #[inline]
    pub fn destroy_objects(
        &mut self,
        objects: impl IntoIterator<Item = Arc<impl Any + Send + Sync>>,
    ) {
        self.death_row
            .extend(objects.into_iter().map(|object| object as _));
    }

    pub(crate) fn bind_global_set(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &Arc<PipelineLayout>,
    ) {
        let Some(bcx) = self.accesses.resource_map.resources().bindless_context() else {
            return;
        };

        if !pipeline_layout
            .set_layouts()
            .get(GLOBAL_SET as usize)
            .is_some_and(|layout| layout == bcx.global_set_layout())
        {
            return;
        }

        unsafe {
            self.as_raw().bind_descriptor_sets_unchecked(
                pipeline_bind_point,
                pipeline_layout,
                GLOBAL_SET,
                &[bcx.global_set().as_raw()],
                &[],
            )
        };
    }
}

unsafe impl DeviceOwned for RecordingCommandBuffer<'_> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

unsafe impl VulkanObject for RecordingCommandBuffer<'_> {
    type Handle = vk::CommandBuffer;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

impl<'a> ResourceAccesses<'a> {
    unsafe fn buffer_unchecked(&self, id: Id<Buffer>) -> &'a Arc<Buffer> {
        if id.is_virtual() {
            // SAFETY:
            // * The caller of `Task::execute` must ensure that `self.resource_map` maps the virtual
            //   IDs of the graph exhaustively.
            // * The caller must ensure that `id` is valid.
            unsafe { self.resource_map.buffer_unchecked(id) }.buffer()
        } else {
            let resources = self.resource_map.resources();

            // SAFETY:
            // * `ResourceMap` owns an `epoch::Guard`.
            // * The caller must ensure that `id` is valid.
            unsafe { resources.buffer_unchecked_unprotected(id) }.buffer()
        }
    }

    unsafe fn image_unchecked(&self, id: Id<Image>) -> &'a Arc<Image> {
        if id.is_virtual() {
            // SAFETY:
            // * The caller must ensure that `id` is valid.
            // * The caller of `Task::execute` must ensure that `self.resource_map` maps the virtual
            //   IDs of the graph exhaustively.
            unsafe { self.resource_map.image_unchecked(id) }.image()
        } else {
            let resources = self.resource_map.resources();

            // SAFETY:
            // * The caller must ensure that `id` is valid.
            // * `ResourceMap` owns an `epoch::Guard`.
            unsafe { resources.image_unchecked_unprotected(id) }.image()
        }
    }
}

impl CommandBufferState {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    fn invalidate_descriptor_sets(
        &mut self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_layout: &Arc<PipelineLayout>,
    ) -> Option<u32> {
        let state = match pipeline_bind_point {
            PipelineBindPoint::Graphics => &mut self.descriptor_sets_graphics,
            PipelineBindPoint::Compute => &mut self.descriptor_sets_compute,
            _ => unreachable!(),
        };

        match state {
            None => {
                *state = Some(DescriptorSetState {
                    pipeline_layout: pipeline_layout.clone(),
                });

                Some(0)
            }
            Some(state) => {
                if state.pipeline_layout == *pipeline_layout {
                    // If we're still using the exact same layout, then of course it's compatible.
                    None
                } else if state.pipeline_layout.push_constant_ranges()
                    != pipeline_layout.push_constant_ranges()
                {
                    state.pipeline_layout = pipeline_layout.clone();

                    // If the push constant ranges don't match, all bound descriptor sets are
                    // disturbed.
                    Some(0)
                } else {
                    state.pipeline_layout = pipeline_layout.clone();

                    let old_layouts = state.pipeline_layout.set_layouts();
                    let new_layouts = pipeline_layout.set_layouts();

                    // Find the first descriptor set layout in the current pipeline layout that
                    // isn't compatible with the corresponding set in the new pipeline layout.
                    // If an incompatible set was found, all bound sets from that slot onwards will
                    // be disturbed.
                    old_layouts
                        .iter()
                        .zip(new_layouts)
                        .position(|(old_layout, new_layout)| {
                            !old_layout.is_compatible_with(new_layout)
                        })
                        .map(|num| num as u32)
                }
            }
        }
    }
}

type Result<T = (), E = Box<vulkano::ValidationError>> = ::std::result::Result<T, E>;
