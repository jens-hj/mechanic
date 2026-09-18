//! Terrain triangles: contact, impact bounds, and recovery.

use super::*;

#[test]
pub(super) fn terrain_triangles_contact_cuboids_convex_parts_and_cylinders_on_slopes_and_walls() {
    let (device, queue) = test_device().expect("terrain collider regression requires an adapter");
    let mut cylinder = ConstructionGraph::new();
    cylinder
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
            BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
        )))
        .unwrap();
    let shapes = [
        material_cube(ConstructionMaterial::Steel, 3),
        shaped_wedge(3),
        cylinder.compile().unwrap(),
    ];
    for (shape, creation) in shapes.iter().enumerate() {
        for angle in [
            0.0,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
        ] {
            let mut terrain = super::super::terrain::tests::chunk();
            let rotation = bevy_math::Quat::from_rotation_z(angle);
            for vertex in &mut terrain.vertices {
                *vertex = (rotation * Vec3::from_array(*vertex)).to_array();
            }
            let mut gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
                .unwrap();
            gpu.dispatch_tick(&device, &queue, 1);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let readback = gpu.read_last_tick(&device).unwrap();
            assert_eq!(readback.error_flags, 0, "shape {shape}, angle {angle}");
            assert!(
                readback.active_contact_count > 0,
                "shape {shape}, angle {angle}: {readback:?}"
            );
        }
    }
}

#[test]
pub(super) fn terrain_contacts_enter_the_fused_articulated_solver() {
    let (device, queue) =
        test_device().expect("terrain articulated regression requires an adapter");
    let fixture = articulated_car_fixture();
    let mut gpu = GpuPhysics::new_with_config(
        &device,
        &queue,
        &fixture.creation,
        GpuPhysicsConfig {
            ground_plane_enabled: false,
            mechanism_self_collisions: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        gpu.solver_route(),
        super::super::GpuSolverRoute::FusedSmallMechanism
    );
    let mut terrain = super::super::terrain::tests::chunk();
    terrain.origin = mechanic_world::WorldPosition(bevy_math::DVec3::Y * 0.3);
    gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
        .unwrap();
    gpu.dispatch_tick(&device, &queue, 1);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let readback = gpu.read_last_tick(&device).unwrap();
    assert_eq!(readback.error_flags, 0, "{readback:?}");
    assert!(readback.active_contact_count > 0, "{readback:?}");
    assert!(readback.executed_solver_sweeps > 0, "{readback:?}");
}

#[test]
pub(super) fn rigid_terrain_impact_bounds_hold_for_convex_parts_and_cylinders() {
    let (device, queue) = test_device().expect("terrain impact regression requires an adapter");
    let mut cylinder = ConstructionGraph::new();
    cylinder
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
            BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
        )))
        .unwrap();
    for (shape, mut creation) in [shaped_wedge(5), cylinder.compile().unwrap()]
        .into_iter()
        .enumerate()
    {
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        for speed in [1.0, 5.0, 20.0] {
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
            gpu.write_terrain_chunks(
                &device,
                &queue,
                [&super::super::terrain::tests::rigid_chunk()],
                bevy_math::DVec3::ZERO,
            )
            .unwrap();
            gpu.enable_async_readback();
            gpu.apply_impulse(
                &device,
                &queue,
                0,
                creation.compounds[0].root_translation,
                Vec3::NEG_Y * speed * creation.compounds[0].mass_properties.mass,
            )
            .unwrap();
            for tick in 1..=120 {
                gpu.dispatch_tick(&device, &queue, tick);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
                assert_eq!(
                    sample.diagnostics.error_flags, 0,
                    "shape {shape}, speed {speed}, tick {tick}"
                );
                for collider in &creation.colliders {
                    let pose = sample.transforms[collider.compound_index as usize];
                    let rotation = bevy_math::Quat::from_array(pose.rotation);
                    let minimum = match &collider.shape {
                        mechanic_core::ColliderShape::Cuboid {
                            local_rotation,
                            half_extents,
                        } => {
                            let center = Vec3::from_slice(&pose.position[..3])
                                + rotation * collider.local_center;
                            center.y
                                - ((rotation * *local_rotation).inverse() * Vec3::Y)
                                    .abs()
                                    .dot(*half_extents)
                        }
                        mechanic_core::ColliderShape::Convex(convex) => convex
                            .vertices
                            .iter()
                            .map(|vertex| pose.position[1] + (rotation * *vertex).y)
                            .fold(f32::INFINITY, f32::min),
                    };
                    let limit = if tick > 100 { 0.002 } else { 0.005 };
                    assert!(
                        minimum >= -limit,
                        "shape {shape}, speed {speed}, tick {tick}: bottom {minimum}"
                    );
                }
            }
        }
    }
}

#[test]
pub(super) fn terrain_recovery_preserves_articulated_and_suspension_constraints() {
    use mechanic_core::{ShockSpec, SpringSpec, SuspensionSpec};
    let (device, queue) =
        test_device().expect("articulated terrain regression requires an adapter");
    let suspension = SuspensionSpec::new(
        Some(SpringSpec::default()),
        Some(ShockSpec::default()),
        None,
    )
    .unwrap();
    let scenes = [
        articulated_car_fixture().creation,
        suspension_test_creation_with_anchor(suspension, true, 1, false),
        suspension_test_creation_with_anchor(suspension, true, 65, false),
    ];
    for (scene, mut creation) in scenes.into_iter().enumerate() {
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                mechanism_self_collisions: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            gpu.solver_route(),
            if scene == 2 {
                super::super::GpuSolverRoute::General
            } else {
                super::super::GpuSolverRoute::FusedSmallMechanism
            }
        );
        let mut terrain = super::super::terrain::tests::rigid_chunk();
        for vertex in &mut terrain.vertices {
            vertex[0] *= 100.0;
            vertex[2] *= 100.0;
        }
        gpu.write_terrain_chunks(&device, &queue, [&terrain], bevy_math::DVec3::ZERO)
            .unwrap();
        gpu.enable_async_readback();
        for (body, compound) in creation.compounds.iter().enumerate() {
            if compound.is_static {
                continue;
            }
            gpu.apply_impulse(
                &device,
                &queue,
                u32::try_from(body).unwrap(),
                compound.root_translation,
                Vec3::NEG_Y * 20.0 * compound.mass_properties.mass,
            )
            .unwrap();
        }
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(
                sample.diagnostics.error_flags, 0,
                "scene {scene}, tick {tick}: {:?}",
                sample.diagnostics
            );
            let minimum = minimum_dynamic_collider_height(&creation, &sample.transforms);
            assert!(
                minimum >= -if tick > 100 { 0.002 } else { 0.005 },
                "scene {scene}, tick {tick}: bottom {minimum}"
            );
        }
    }
}

#[test]
pub(super) fn terrain_position_recovery_preserves_intentional_restitution() {
    let (device, queue) =
        test_device().expect("terrain restitution regression requires an adapter");
    let mut peaks = Vec::new();
    for restitution in [0.0, 0.8] {
        let mut creation = material_cube(ConstructionMaterial::Steel, 5);
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = restitution;
        }
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
        gpu.write_terrain_chunks(
            &device,
            &queue,
            [&super::super::terrain::tests::rigid_chunk()],
            bevy_math::DVec3::ZERO,
        )
        .unwrap();
        gpu.enable_async_readback();
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            creation.compounds[0].root_translation,
            Vec3::NEG_Y * 5.0 * creation.compounds[0].mass_properties.mass,
        )
        .unwrap();
        let mut peak = f32::NEG_INFINITY;
        for tick in 1..=30 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let sample = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(sample.diagnostics.error_flags, 0);
            peak = peak.max(sample.velocities[0].linear[1]);
        }
        peaks.push(peak);
    }
    assert!(
        peaks[0] < 0.001,
        "inelastic terrain contact rebounded: {peaks:?}"
    );
    assert!(
        peaks[1] > 1.0,
        "intentional restitution disappeared: {peaks:?}"
    );
}

pub(super) fn minimum_dynamic_collider_height(
    creation: &mechanic_core::CompiledCreation,
    transforms: &[crate::GpuTransform],
) -> f32 {
    creation
        .colliders
        .iter()
        .filter(|collider| !creation.compounds[collider.compound_index as usize].is_static)
        .map(|collider| {
            let pose = transforms[collider.compound_index as usize];
            let rotation = bevy_math::Quat::from_array(pose.rotation);
            match &collider.shape {
                mechanic_core::ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => {
                    let center =
                        Vec3::from_slice(&pose.position[..3]) + rotation * collider.local_center;
                    center.y
                        - ((rotation * *local_rotation).inverse() * Vec3::Y)
                            .abs()
                            .dot(*half_extents)
                }
                mechanic_core::ColliderShape::Convex(convex) => convex
                    .vertices
                    .iter()
                    .map(|vertex| pose.position[1] + (rotation * *vertex).y)
                    .fold(f32::INFINITY, f32::min),
            }
        })
        .fold(f32::INFINITY, f32::min)
}
