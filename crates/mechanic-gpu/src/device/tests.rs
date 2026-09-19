mod contacts;
mod drives;
mod holds;
mod impulses;
mod linear;
mod pendulums;
mod pipelines;
mod piston;
mod steering;
mod suspension;
mod terrain_scenes;
mod tick_readback;
mod vehicles;

use contacts::{material_cube, shaped_wedge};
use linear::{
    assert_mixed_linear_constraints, linear_test_creation, linear_test_snapshot,
    mixed_linear_creation,
};
use pendulums::{pendulum_creation, relative_bearing_rotation};
use suspension::suspension_test_creation_with_anchor;
use tick_readback::copy_state_rows;
use vehicles::{articulated_car_fixture, snapshot_speed, transform_position};

use super::ASYNC_READBACK_RING_SIZE;
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ConstructionMaterial,
    CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveMode, EngineKind, FaceKind,
    FaceRef, GearboxConfig, GridRotation, PartId, PipeBendDimensions, PipeBendSpec, RigidLinkSpec,
    ServoSpec, WeldSpec,
};

use crate::GpuMechanismCoordinate;

use super::{
    EXTERNAL_IMPULSE_BATCH_CAPACITY, FULL_CYLINDER_GROUND_FIRST, GpuExternalImpulse,
    GpuGroundPlane, GpuGroundPlaneError, GpuImpulseError, GpuPhysics, GpuPhysicsConfig,
    GpuPhysicsPipelines, full_cylinder_ground_data, pipelines::collision::contact_pair_capacity,
    uses_fused_contact_schedule, uses_fused_velocity_schedule,
};

fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    eprintln!("GPU test adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("mechanic articulated test device"),
        ..Default::default()
    }))
    .ok()
}

fn spawned_part(outcome: BuildOutcome) -> PartId {
    let BuildOutcome::Spawned(part) = outcome else {
        unreachable!()
    };
    part
}

fn run_ticks(
    creation: &mechanic_core::CompiledCreation,
    ticks: u64,
    collisions_enabled: bool,
) -> Option<(Vec<crate::GpuTransform>, crate::GpuTickReadback)> {
    let (device, queue) = test_device()?;
    let gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        creation,
        GpuPhysicsConfig {
            collisions_enabled,
            ground_plane_enabled: true,
            mechanism_self_collisions: true,
            solver_iterations: 16,
        },
    )
    .ok()?;
    for tick in 1..=ticks {
        gpu.dispatch_tick(&device, &queue, tick);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let readback = gpu.read_last_tick(&device).ok()?;
    let snapshot = gpu
        .read_snapshot_transforms(&device, &queue, (ticks % 3) as u8)
        .ok()?;
    Some((snapshot, readback))
}
