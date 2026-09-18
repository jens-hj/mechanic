//! Small wgpu helpers shared by every device module.

use std::borrow::Cow;

use bytemuck::{Pod, bytes_of, cast_slice};
use wgpu::util::DeviceExt;

use super::{GpuPhysicsPipelines, TimestampResources};

pub(super) fn shader_module(
    pipelines: &GpuPhysicsPipelines,
    device: &wgpu::Device,
    label: &'static str,
    source: &str,
) -> wgpu::ShaderModule {
    if let Some(shader) = pipelines.shaders.lock().unwrap().get(label).cloned() {
        return shader;
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    pipelines
        .shaders
        .lock()
        .unwrap()
        .insert(label, shader.clone());
    shader
}

pub(super) fn compute_pipeline(
    pipelines: &GpuPhysicsPipelines,
    device: &wgpu::Device,
    label: &'static str,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let key = (label, entry_point);
    if let Some(pipeline) = pipelines.pipelines.lock().unwrap().get(&key).cloned() {
        return pipeline;
    }
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    pipelines
        .pipelines
        .lock()
        .unwrap()
        .insert(key, pipeline.clone());
    pipeline
}

pub(super) fn bind_group(
    device: &wgpu::Device,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    })
}

pub(super) fn timestamp_writes(
    timestamps: Option<&TimestampResources>,
    beginning: Option<u32>,
    end: Option<u32>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    if beginning.is_none() && end.is_none() {
        return None;
    }
    timestamps.map(|timestamps| wgpu::ComputePassTimestampWrites {
        query_set: &timestamps.query_set,
        beginning_of_pass_write_index: beginning,
        end_of_pass_write_index: end,
    })
}

pub(super) fn direct_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    workgroups: u32,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

pub(super) fn indirect_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    indirect_args: &wgpu::Buffer,
    indirect_offset: u64,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups_indirect(indirect_args, indirect_offset);
}

pub(super) fn indirect_dispatch_in_pass<'a>(
    pass: &mut wgpu::ComputePass<'a>,
    pipeline: &'a wgpu::ComputePipeline,
    bindings: &'a wgpu::BindGroup,
    indirect_args: &'a wgpu::Buffer,
    indirect_offset: u64,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups_indirect(indirect_args, indirect_offset);
}

pub(super) fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn create_uniform_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn create_storage_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        values,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

pub(super) fn create_state_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    create_buffer(
        device,
        label,
        values,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

pub(super) fn create_readonly_storage_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    create_buffer(device, label, values, wgpu::BufferUsages::STORAGE)
}

pub(super) fn create_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    if values.is_empty() {
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: u64::try_from(size_of::<T>().max(16)).unwrap_or(16),
            usage,
            mapped_at_creation: false,
        });
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: cast_slice(values),
        usage,
    })
}

pub(super) fn create_sized_buffer(
    device: &wgpu::Device,
    label: &str,
    size: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::try_from(size).unwrap_or(u64::MAX),
        usage,
        mapped_at_creation: false,
    })
}

pub(super) fn vec4(vector: bevy_math::Vec3, w: f32) -> [f32; 4] {
    [vector.x, vector.y, vector.z, w]
}

pub(super) fn wrapping_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[expect(clippy::cast_precision_loss)]
pub(super) fn diagnostic_units(value: u32) -> f32 {
    value as f32 / 1_000_000.0
}

pub(super) fn timestamp_milliseconds(start: u64, end: u64, period_nanoseconds: f64) -> f64 {
    let ticks = end.wrapping_sub(start);
    let bounded_ticks = u32::try_from(ticks).unwrap_or(u32::MAX);
    f64::from(bounded_ticks) * period_nanoseconds / 1_000_000.0
}
