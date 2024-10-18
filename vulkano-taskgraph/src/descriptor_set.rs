use crate::{resource::ResourceStorage, Id, Ref};
use ahash::HashMap;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use concurrent_slotmap::{epoch, SlotId, SlotMap};
use std::{collections::BTreeMap, iter, sync::Arc};
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, Subbuffer},
    descriptor_set::{
        allocator::{AllocationHandle, DescriptorSetAlloc, DescriptorSetAllocator},
        layout::{
            DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType,
        },
        pool::{
            DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
            DescriptorSetAllocateInfo,
        },
        sys::RawDescriptorSet,
        DescriptorImageViewInfo, WriteDescriptorSet,
    },
    device::{Device, DeviceExtensions, DeviceFeatures, DeviceOwned},
    image::{
        sampler::{Sampler, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageLayout,
    },
    instance::Instance,
    pipeline::{
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::ShaderStages,
    DeviceSize, Validated, Version, VulkanError, VulkanObject,
};

// NOTE(Marc): The following constants must match the definitions in include/vulkano.glsl!

/// The set number of the [`GlobalDescriptorSet`].
pub const GLOBAL_SET: u32 = 0;
const SAMPLER_BINDING: u32 = 0;
const SAMPLED_IMAGE_BINDING: u32 = 1;
const STORAGE_IMAGE_BINDING: u32 = 2;
const STORAGE_BUFFER_BINDING: u32 = 3;
const ACCELERATION_STRUCTURE_BINDING: u32 = 4;

#[derive(Debug)]
pub struct BindlessContext {
    global_set: GlobalDescriptorSet,
}

impl BindlessContext {
    /// Returns the device extensions required to create a bindless context.
    pub fn required_extensions(instance: &Instance) -> DeviceExtensions {
        let mut extensions = DeviceExtensions::default();

        if instance.api_version() < Version::V1_2 {
            extensions.ext_descriptor_indexing = true;
        }

        extensions
    }

    /// Returns the device features required to create a bindless context.
    pub fn required_features(_instance: &Instance) -> DeviceFeatures {
        DeviceFeatures {
            shader_sampled_image_array_dynamic_indexing: true,
            shader_storage_image_array_dynamic_indexing: true,
            shader_storage_buffer_array_dynamic_indexing: true,
            descriptor_binding_sampled_image_update_after_bind: true,
            descriptor_binding_storage_image_update_after_bind: true,
            descriptor_binding_storage_buffer_update_after_bind: true,
            descriptor_binding_update_unused_while_pending: true,
            descriptor_binding_partially_bound: true,
            runtime_descriptor_array: true,
            ..DeviceFeatures::default()
        }
    }

    pub(crate) fn new(
        resources: &Arc<ResourceStorage>,
        create_info: &BindlessContextCreateInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        let global_set_layout =
            GlobalDescriptorSet::create_layout(resources, create_info.global_set)?;

        let global_set = GlobalDescriptorSet::new(resources, &global_set_layout)?;

        Ok(BindlessContext { global_set })
    }

    /// Returns the layout of the [`GlobalDescriptorSet`].
    #[inline]
    pub fn global_set_layout(&self) -> &Arc<DescriptorSetLayout> {
        self.global_set.inner.layout()
    }

    /// Creates a new bindless pipeline layout from the union of the push constant requirements of
    /// each stage in `stages` for push constant ranges and the [global descriptor set layout] for
    /// set layouts.
    ///
    /// All pipelines that you bind must have been created with a layout created like this or with
    /// a compatible layout for the bindless system to be able to bind its descriptor sets.
    ///
    /// It is recommended that you share the same pipeline layout object with as many pipelines as
    /// possible in order to reduce the amount of descriptor set (re)binding that is needed.
    ///
    /// See also [`pipeline_layout_create_info_from_stages`].
    ///
    /// [global descriptor set layout]: Self::global_set_layout
    /// [`pipeline_layout_create_info_from_stages`]: Self::pipeline_layout_create_info_from_stages
    pub fn pipeline_layout_from_stages<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> Result<Arc<PipelineLayout>, Validated<VulkanError>> {
        PipelineLayout::new(
            self.device().clone(),
            self.pipeline_layout_create_info_from_stages(stages),
        )
    }

    /// Creates a new bindless pipeline layout create info from the union of the push constant
    /// requirements of each stage in `stages` for push constant ranges and the [global descriptor
    /// set layout] for set layouts.
    ///
    /// All pipelines that you bind must have been created with a layout created like this or with
    /// a compatible layout for the bindless system to be able to bind its descriptor sets.
    ///
    /// It is recommended that you share the same pipeline layout object with as many pipelines as
    /// possible in order to reduce the amount of descriptor set (re)binding that is needed.
    ///
    /// See also [`pipeline_layout_from_stages`].
    ///
    /// [global descriptor set layout]: Self::global_set_layout
    /// [`pipeline_layout_from_stages`]: Self::pipeline_layout_from_stages
    pub fn pipeline_layout_create_info_from_stages<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> PipelineLayoutCreateInfo {
        let mut push_constant_ranges = Vec::<PushConstantRange>::new();

        for stage in stages {
            let entry_point_info = stage.entry_point.info();

            if let Some(range) = &entry_point_info.push_constant_requirements {
                if let Some(existing_range) =
                    push_constant_ranges.iter_mut().find(|existing_range| {
                        existing_range.offset == range.offset && existing_range.size == range.size
                    })
                {
                    // If this range was already used before, add our stage to it.
                    existing_range.stages |= range.stages;
                } else {
                    // If this range is new, insert it.
                    push_constant_ranges.push(*range);
                }
            }
        }

        PipelineLayoutCreateInfo {
            set_layouts: vec![self.global_set_layout().clone()],
            push_constant_ranges,
            ..Default::default()
        }
    }

    /// Returns the `GlobalDescriptorSet`.
    #[inline]
    pub fn global_set(&self) -> &GlobalDescriptorSet {
        &self.global_set
    }
}

unsafe impl DeviceOwned for BindlessContext {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.global_set.device()
    }
}

/// Parameters to create a new [`BindlessContext`].
#[derive(Clone, Debug)]
pub struct BindlessContextCreateInfo<'a> {
    /// Parameters to create the [`GlobalDescriptorSet`].
    ///
    /// The default value is `&GlobalDescriptorSetCreateInfo::new()`.
    pub global_set: &'a GlobalDescriptorSetCreateInfo<'a>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl BindlessContextCreateInfo<'_> {
    /// Creates a new `BindlessContextCreateInfo` with default values.
    #[inline]
    pub const fn new() -> Self {
        BindlessContextCreateInfo {
            global_set: {
                const DEFAULT: GlobalDescriptorSetCreateInfo<'_> =
                    GlobalDescriptorSetCreateInfo::new();

                &DEFAULT
            },
            _ne: crate::NE,
        }
    }
}

impl Default for BindlessContextCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct GlobalDescriptorSet {
    // DO NOT change the order of these fields! `ResourceStorage` must be dropped first because
    // that guarantees that all flights are waited on before the descriptor set is destroyed.
    resources: Arc<ResourceStorage>,
    inner: RawDescriptorSet,

    samplers: SlotMap<SamplerDescriptor>,
    sampled_images: SlotMap<SampledImageDescriptor>,
    storage_images: SlotMap<StorageImageDescriptor>,
    storage_buffers: SlotMap<StorageBufferDescriptor>,
    acceleration_structures: SlotMap<AccelerationStructureDescriptor>,
}

#[derive(Debug)]
pub struct SamplerDescriptor {
    sampler: Arc<Sampler>,
}

#[derive(Debug)]
pub struct SampledImageDescriptor {
    image_id: Id<Image>,
    image_view: Arc<ImageView>,
    image_layout: ImageLayout,
}

#[derive(Debug)]
pub struct StorageImageDescriptor {
    image_id: Id<Image>,
    image_view: Arc<ImageView>,
    image_layout: ImageLayout,
}

#[derive(Debug)]
pub struct StorageBufferDescriptor {
    buffer_id: Id<Buffer>,
    buffer: Arc<Buffer>,
    offset: DeviceSize,
    size: DeviceSize,
}

#[derive(Debug)]
pub struct AccelerationStructureDescriptor {
    acceleration_structure: Arc<AccelerationStructure>,
}

impl GlobalDescriptorSet {
    fn new(
        resources: &Arc<ResourceStorage>,
        layout: &Arc<DescriptorSetLayout>,
    ) -> Result<Self, Validated<VulkanError>> {
        let device = resources.device();

        let allocator = Arc::new(GlobalDescriptorSetAllocator::new(device));
        let inner = RawDescriptorSet::new(allocator, layout, 0).map_err(Validated::unwrap)?;

        let global = resources.global();

        let descriptor_count = |n| layout.bindings().get(&n).map_or(0, |b| b.descriptor_count);
        let max_samplers = descriptor_count(SAMPLER_BINDING);
        let max_sampled_images = descriptor_count(SAMPLED_IMAGE_BINDING);
        let max_storage_images = descriptor_count(STORAGE_IMAGE_BINDING);
        let max_storage_buffers = descriptor_count(STORAGE_BUFFER_BINDING);
        let max_acceleration_structures = descriptor_count(ACCELERATION_STRUCTURE_BINDING);

        Ok(GlobalDescriptorSet {
            resources: resources.clone(),
            inner,
            samplers: SlotMap::with_global(max_samplers, global.clone()),
            sampled_images: SlotMap::with_global(max_sampled_images, global.clone()),
            storage_images: SlotMap::with_global(max_storage_images, global.clone()),
            storage_buffers: SlotMap::with_global(max_storage_buffers, global.clone()),
            acceleration_structures: SlotMap::with_global(
                max_acceleration_structures,
                global.clone(),
            ),
        })
    }

    fn create_layout(
        resources: &Arc<ResourceStorage>,
        create_info: &GlobalDescriptorSetCreateInfo<'_>,
    ) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
        let device = resources.device();

        let binding_flags = DescriptorBindingFlags::UPDATE_AFTER_BIND
            | DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | DescriptorBindingFlags::PARTIALLY_BOUND;

        let stages = get_all_supported_shader_stages(device);

        let mut bindings = BTreeMap::from_iter([
            (
                SAMPLER_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_samplers,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
                },
            ),
            (
                SAMPLED_IMAGE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_sampled_images,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
                },
            ),
            (
                STORAGE_IMAGE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_storage_images,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                },
            ),
            (
                STORAGE_BUFFER_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_storage_buffers,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            ),
        ]);

        if device.enabled_features().acceleration_structure {
            bindings.insert(
                ACCELERATION_STRUCTURE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_acceleration_structures,
                    stages,
                    ..DescriptorSetLayoutBinding::descriptor_type(
                        DescriptorType::AccelerationStructure,
                    )
                },
            );
        }

        let layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                flags: DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                bindings,
                ..Default::default()
            },
        )?;

        Ok(layout)
    }

    /// Returns the underlying raw descriptor set.
    #[inline]
    pub fn as_raw(&self) -> &RawDescriptorSet {
        &self.inner
    }

    pub fn create_sampler(
        &self,
        create_info: SamplerCreateInfo,
    ) -> Result<SamplerId, Validated<VulkanError>> {
        let sampler = Sampler::new(self.device().clone(), create_info)?;

        Ok(self.add_sampler(sampler))
    }

    pub fn create_sampled_image(
        &self,
        image_id: Id<Image>,
        create_info: ImageViewCreateInfo,
        image_layout: ImageLayout,
    ) -> Result<SampledImageId, Validated<VulkanError>> {
        let image_state = self.resources.image(image_id).unwrap();
        let image_view = ImageView::new(image_state.image().clone(), create_info)?;

        Ok(self.add_sampled_image(image_id, image_view, image_layout))
    }

    pub fn create_storage_image(
        &self,
        image_id: Id<Image>,
        create_info: ImageViewCreateInfo,
        image_layout: ImageLayout,
    ) -> Result<StorageImageId, Validated<VulkanError>> {
        let image_state = self.resources.image(image_id).unwrap();
        let image_view = ImageView::new(image_state.image().clone(), create_info)?;

        Ok(self.add_storage_image(image_id, image_view, image_layout))
    }

    pub fn create_storage_buffer(
        &self,
        buffer_id: Id<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
    ) -> Result<StorageBufferId, Validated<VulkanError>> {
        let buffer_state = self.resources.buffer(buffer_id).unwrap();
        let buffer = buffer_state.buffer().clone();

        Ok(self.add_storage_buffer(buffer_id, buffer, offset, size))
    }

    fn add_sampler(&self, sampler: Arc<Sampler>) -> SamplerId {
        let descriptor = SamplerDescriptor {
            sampler: sampler.clone(),
        };
        let slot = self.samplers.insert(descriptor, self.resources.pin());

        let write =
            WriteDescriptorSet::sampler_array(SAMPLER_BINDING, slot.index(), iter::once(sampler));

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        SamplerId::new(slot)
    }

    fn add_sampled_image(
        &self,
        image_id: Id<Image>,
        image_view: Arc<ImageView>,
        image_layout: ImageLayout,
    ) -> SampledImageId {
        assert!(matches!(
            image_layout,
            ImageLayout::General
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
        ));

        let descriptor = SampledImageDescriptor {
            image_id,
            image_view: image_view.clone(),
            image_layout,
        };
        let slot = self.sampled_images.insert(descriptor, self.resources.pin());

        let write = WriteDescriptorSet::image_view_with_layout_array(
            SAMPLED_IMAGE_BINDING,
            slot.index(),
            iter::once(DescriptorImageViewInfo {
                image_view,
                image_layout,
            }),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        SampledImageId::new(slot)
    }

    fn add_storage_image(
        &self,
        image_id: Id<Image>,
        image_view: Arc<ImageView>,
        image_layout: ImageLayout,
    ) -> StorageImageId {
        assert_eq!(image_layout, ImageLayout::General);

        let descriptor = StorageImageDescriptor {
            image_id,
            image_view: image_view.clone(),
            image_layout,
        };
        let slot = self.storage_images.insert(descriptor, self.resources.pin());

        let write = WriteDescriptorSet::image_view_with_layout_array(
            STORAGE_IMAGE_BINDING,
            slot.index(),
            iter::once(DescriptorImageViewInfo {
                image_view,
                image_layout,
            }),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        StorageImageId::new(slot)
    }

    fn add_storage_buffer(
        &self,
        buffer_id: Id<Buffer>,
        buffer: Arc<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
    ) -> StorageBufferId {
        let subbuffer = Subbuffer::from(buffer.clone()).slice(offset..offset + size);

        let descriptor = StorageBufferDescriptor {
            buffer_id,
            buffer,
            offset,
            size,
        };
        let slot = self
            .storage_buffers
            .insert(descriptor, self.resources.pin());

        let write = WriteDescriptorSet::buffer_array(
            STORAGE_BUFFER_BINDING,
            slot.index(),
            iter::once(subbuffer),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        StorageBufferId::new(slot)
    }

    pub fn add_acceleration_structure(
        &self,
        acceleration_structure: Arc<AccelerationStructure>,
    ) -> AccelerationStructureId {
        let descriptor = AccelerationStructureDescriptor {
            acceleration_structure: acceleration_structure.clone(),
        };
        let slot = self
            .acceleration_structures
            .insert(descriptor, self.resources.pin());

        let write = WriteDescriptorSet::acceleration_structure_array(
            ACCELERATION_STRUCTURE_BINDING,
            slot.index(),
            iter::once(acceleration_structure),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        AccelerationStructureId::new(slot)
    }

    pub unsafe fn remove_sampler(&self, id: SamplerId) -> Option<Ref<'_, SamplerDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.samplers.remove(slot, self.resources.pin()).map(Ref)
    }

    pub unsafe fn remove_sampled_image(
        &self,
        id: SampledImageId,
    ) -> Option<Ref<'_, SampledImageDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.sampled_images
            .remove(slot, self.resources.pin())
            .map(Ref)
    }

    pub unsafe fn remove_storage_image(
        &self,
        id: StorageImageId,
    ) -> Option<Ref<'_, StorageImageDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.storage_images
            .remove(slot, self.resources.pin())
            .map(Ref)
    }

    pub unsafe fn remove_storage_buffer(
        &self,
        id: StorageBufferId,
    ) -> Option<Ref<'_, StorageBufferDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.storage_buffers
            .remove(slot, self.resources.pin())
            .map(Ref)
    }

    pub unsafe fn remove_acceleration_structure(
        &self,
        id: AccelerationStructureId,
    ) -> Option<Ref<'_, AccelerationStructureDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.acceleration_structures
            .remove(slot, self.resources.pin())
            .map(Ref)
    }

    #[inline]
    pub fn sampler(&self, id: SamplerId) -> Option<Ref<'_, SamplerDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.samplers.get(slot, self.resources.pin()).map(Ref)
    }

    #[inline]
    pub fn sampled_image(&self, id: SampledImageId) -> Option<Ref<'_, SampledImageDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.sampled_images.get(slot, self.resources.pin()).map(Ref)
    }

    #[inline]
    pub fn storage_image(&self, id: StorageImageId) -> Option<Ref<'_, StorageImageDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.storage_images.get(slot, self.resources.pin()).map(Ref)
    }

    #[inline]
    pub fn storage_buffer(&self, id: StorageBufferId) -> Option<Ref<'_, StorageBufferDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.storage_buffers
            .get(slot, self.resources.pin())
            .map(Ref)
    }

    #[inline]
    pub fn acceleration_structure(
        &self,
        id: AccelerationStructureId,
    ) -> Option<Ref<'_, AccelerationStructureDescriptor>> {
        let slot = SlotId::new(id.index, id.generation);

        self.acceleration_structures
            .get(slot, self.resources.pin())
            .map(Ref)
    }

    pub(crate) fn try_collect(&self, guard: &epoch::Guard<'_>) {
        self.samplers.try_collect(guard);
        self.sampled_images.try_collect(guard);
        self.storage_images.try_collect(guard);
        self.storage_buffers.try_collect(guard);
        self.acceleration_structures.try_collect(guard);
    }
}

unsafe impl VulkanObject for GlobalDescriptorSet {
    type Handle = vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for GlobalDescriptorSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl SamplerDescriptor {
    #[inline]
    pub fn sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }
}

impl SampledImageDescriptor {
    #[inline]
    pub fn image_id(&self) -> Id<Image> {
        self.image_id
    }

    #[inline]
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    #[inline]
    pub fn image_layout(&self) -> ImageLayout {
        self.image_layout
    }
}

impl StorageImageDescriptor {
    #[inline]
    pub fn image_id(&self) -> Id<Image> {
        self.image_id
    }

    #[inline]
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    #[inline]
    pub fn image_layout(&self) -> ImageLayout {
        self.image_layout
    }
}

impl StorageBufferDescriptor {
    #[inline]
    pub fn buffer_id(&self) -> Id<Buffer> {
        self.buffer_id
    }

    #[inline]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }
}

impl AccelerationStructureDescriptor {
    #[inline]
    pub fn acceleration_structure(&self) -> &Arc<AccelerationStructure> {
        &self.acceleration_structure
    }
}

/// Parameters to create a new [`GlobalDescriptorSet`].
#[derive(Clone, Debug)]
pub struct GlobalDescriptorSetCreateInfo<'a> {
    /// The maximum number of [`Sampler`] descriptors that the collection can hold at once.
    pub max_samplers: u32,

    /// The maximum number of sampled [`Image`] descriptors that the collection can hold at once.
    pub max_sampled_images: u32,

    /// The maximum number of storage [`Image`] descriptors that the collection can hold at once.
    pub max_storage_images: u32,

    /// The maximum number of storage [`Buffer`] descriptors that the collection can hold at once.
    pub max_storage_buffers: u32,

    /// The maximum number of [`AccelerationStructure`] descriptors that the collection can hold at
    /// once.
    pub max_acceleration_structures: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl GlobalDescriptorSetCreateInfo<'_> {
    /// Creates a new `GlobalDescriptorSetCreateInfo` with default values.
    #[inline]
    pub const fn new() -> Self {
        GlobalDescriptorSetCreateInfo {
            max_samplers: 1 << 8,
            max_sampled_images: 1 << 20,
            max_storage_images: 1 << 20,
            max_storage_buffers: 1 << 20,
            max_acceleration_structures: 0,
            _ne: crate::NE,
        }
    }
}

impl Default for GlobalDescriptorSetCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SamplerId {
    index: u32,
    generation: u32,
}

unsafe impl Pod for SamplerId {}
unsafe impl Zeroable for SamplerId {}

impl SamplerId {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = SamplerId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(slot: SlotId) -> Self {
        SamplerId {
            index: slot.index(),
            generation: slot.generation(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SampledImageId {
    index: u32,
    generation: u32,
}

unsafe impl Pod for SampledImageId {}
unsafe impl Zeroable for SampledImageId {}

impl SampledImageId {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = SampledImageId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(slot: SlotId) -> Self {
        SampledImageId {
            index: slot.index(),
            generation: slot.generation(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct StorageImageId {
    index: u32,
    generation: u32,
}

unsafe impl Pod for StorageImageId {}
unsafe impl Zeroable for StorageImageId {}

impl StorageImageId {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = StorageImageId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(slot: SlotId) -> Self {
        StorageImageId {
            index: slot.index(),
            generation: slot.generation(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct StorageBufferId {
    index: u32,
    generation: u32,
}

unsafe impl Pod for StorageBufferId {}
unsafe impl Zeroable for StorageBufferId {}

impl StorageBufferId {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = StorageBufferId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(slot: SlotId) -> Self {
        StorageBufferId {
            index: slot.index(),
            generation: slot.generation(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct AccelerationStructureId {
    index: u32,
    generation: u32,
}

unsafe impl Pod for AccelerationStructureId {}
unsafe impl Zeroable for AccelerationStructureId {}

impl AccelerationStructureId {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = AccelerationStructureId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(slot: SlotId) -> Self {
        AccelerationStructureId {
            index: slot.index(),
            generation: slot.generation(),
        }
    }
}

struct GlobalDescriptorSetAllocator {
    device: Arc<Device>,
}

impl GlobalDescriptorSetAllocator {
    fn new(device: &Arc<Device>) -> Self {
        GlobalDescriptorSetAllocator {
            device: device.clone(),
        }
    }
}

unsafe impl DescriptorSetAllocator for GlobalDescriptorSetAllocator {
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        _variable_count: u32,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        let mut pool_sizes = HashMap::default();

        for binding in layout.bindings().values() {
            *pool_sizes.entry(binding.descriptor_type).or_insert(0) += binding.descriptor_count;
        }

        let pool = Arc::new(DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                flags: DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
                max_sets: 1,
                pool_sizes,
                ..Default::default()
            },
        )?);

        let allocate_info = DescriptorSetAllocateInfo::new(layout.clone());

        let inner = unsafe { pool.allocate_descriptor_sets(iter::once(allocate_info)) }?
            .next()
            .unwrap();

        Ok(DescriptorSetAlloc {
            inner,
            pool,
            handle: AllocationHandle::null(),
        })
    }

    unsafe fn deallocate(&self, _allocation: DescriptorSetAlloc) {}
}

unsafe impl DeviceOwned for GlobalDescriptorSetAllocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

fn get_all_supported_shader_stages(device: &Arc<Device>) -> ShaderStages {
    let mut stages = ShaderStages::all_graphics() | ShaderStages::COMPUTE;

    if device.enabled_extensions().khr_ray_tracing_pipeline
        || device.enabled_extensions().nv_ray_tracing
    {
        stages |= ShaderStages::ANY_HIT
            | ShaderStages::CLOSEST_HIT
            | ShaderStages::MISS
            | ShaderStages::INTERSECTION
            | ShaderStages::CALLABLE;
    }

    if device.enabled_extensions().ext_mesh_shader || device.enabled_extensions().nv_mesh_shader {
        stages |= ShaderStages::TASK | ShaderStages::MESH;
    }

    if device.enabled_extensions().huawei_subpass_shading {
        stages |= ShaderStages::SUBPASS_SHADING;
    }

    stages
}
