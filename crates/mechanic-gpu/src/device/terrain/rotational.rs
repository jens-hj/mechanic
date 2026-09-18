//! Rotational sweeps whose start and end poses both miss the terrain.

use crate::{GpuPhysics, GpuPhysicsConfig, GpuTransform, GpuVelocity};
use bevy_math::{DVec3, Quat, Vec3};
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "keep the impact and its separation controls together"
)]
fn rotating_body_cannot_pass_through_terrain_between_clear_endpoint_poses() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("rotational terrain regression requires an adapter");
    eprintln!("Rotational terrain adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2, 1, 1], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let mut creation = graph.compile().unwrap();
    for collider in &mut creation.colliders {
        collider.material_properties.restitution = 0.0;
        collider.material_properties.youngs_modulus_pa = 200.0e9;
    }
    let mut unconstrained = None;
    for (axis, height, offset, hit, degenerate) in [
        (Vec3::Z, 0.27, DVec3::X * 4.0, false, false),
        (Vec3::Y, 0.14, DVec3::ZERO, false, false),
        (Vec3::Z, 0.27, DVec3::ZERO, false, true),
        (Vec3::Z, 0.27, DVec3::ZERO, true, false),
    ] {
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut terrain = super::tests::rigid_chunk();
        terrain.origin.0 = offset;
        if degenerate {
            terrain.vertices = vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]];
        }
        gpu.write_terrain_chunks(&device, &queue, [&terrain], DVec3::ZERO)
            .unwrap();
        gpu.enable_async_readback();
        let rotation = Quat::from_axis_angle(axis, std::f32::consts::FRAC_PI_4);
        gpu.write_body_states(
            &queue,
            &[GpuTransform {
                position: [0.0, height, 0.0, 0.0],
                rotation: rotation.to_array(),
            }],
            &[GpuVelocity {
                linear: [0.0; 4],
                angular: (axis * 50.0).extend(0.0).to_array(),
            }],
        )
        .unwrap();
        let extent = |rotation: Quat| {
            (rotation.inverse() * Vec3::Y)
                .abs()
                .dot(Vec3::new(0.25, 0.125, 0.125))
        };
        assert!(extent(rotation) < height);
        if hit {
            // Furthest vertices travel below 16 m/s. The intermediate pose
            // penetrates >5 mm despite clear integration endpoints.
            assert!(extent(Quat::from_rotation_z(63.435_f32.to_radians())) > 0.275);
        }
        gpu.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
        assert_eq!(state.diagnostics.error_flags, 0);
        let pose = state.transforms[0];
        if hit {
            let (full_pose, full_velocity): (GpuTransform, GpuVelocity) =
                unconstrained.expect("clear control runs first");
            let fraction = (pose.position[1] - height) / (full_pose.position[1] - height);
            assert!(fraction > 0.0 && fraction < 1.0);
            let spin = Quat::from_array(full_velocity.angular);
            let expected = (rotation
                + (spin * rotation) * (0.5 * mechanic_core::TICK_SECONDS_F32 * fraction))
                .normalize();
            assert!(
                (expected - Quat::from_array(pose.rotation)).length() < 1.0e-4,
                "translation and rotation used different sweep times"
            );
            assert!(
                state.diagnostics.contact_count > 0,
                "rotational crossing produced no contacts: {pose:?}"
            );
            assert!(extent(Quat::from_array(pose.rotation)) - pose.position[1] <= 0.005);
            assert!(
                state.velocities[0].angular[2] < 45.0,
                "impact did not slow rotation"
            );
            let velocity = state.velocities[0];
            let local_spin = Quat::from_array(pose.rotation).inverse()
                * Vec3::from_slice(&velocity.angular[..3]);
            let energy_per_half_mass = Vec3::from_slice(&velocity.linear[..3]).length_squared()
                + (local_spin * local_spin).dot(Vec3::new(0.125, 0.3125, 0.3125) / 12.0);
            let initial_energy = 50.0 * 50.0 * 0.3125 / 12.0;
            assert!(
                energy_per_half_mass <= initial_energy + 0.1,
                "rotational collision created kinetic energy: {energy_per_half_mass}"
            );
            eprintln!(
                "rotational impact: pose {pose:?}, velocity {velocity:?}, contacts {}, energy ratio {}",
                state.diagnostics.contact_count,
                energy_per_half_mass / initial_energy
            );
        } else {
            if axis == Vec3::Z && !degenerate {
                unconstrained = Some((pose, state.velocities[0]));
            }
            // A finite triangle with an overlapping AABB, and rotation parallel
            // to a nearby surface, must retain the complete integration step.
            assert_eq!(state.diagnostics.contact_count, 0);
            let angular = Quat::from_array(state.velocities[0].angular);
            let predicted = (rotation
                + (angular * rotation) * (0.5 * mechanic_core::TICK_SECONDS_F32))
                .normalize();
            assert!(
                predicted.dot(Quat::from_array(pose.rotation)).abs() > 1.0 - 1.0e-6,
                "clear sweep unexpectedly shortened motion: {pose:?}"
            );
            assert!(Vec3::from_slice(&state.velocities[0].angular[..3]).dot(axis) > 45.0);
        }
    }
}
