//! Impact checks using the production volumetric mesher and its BVHs.

use bevy_math::{DVec3, IVec3, Quat, Vec3};
use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};
use mechanic_world::{
    BrickCoord, TerrainField, TerrainMeshRequest, TerrainNodeId, TerrainOctree, TerrainScene,
    TerrainTransitionMask, WorldPosition, WorldSeed, mesh_chunk, raycast_density,
};

use crate::{GpuPhysics, GpuPhysicsConfig};

#[test]
fn generated_terrain_supports_fast_impacts_across_brick_boundaries() {
    check_generated_impacts(false);
}

#[test]
fn generated_caves_stop_wall_and_ceiling_impacts() {
    check_generated_impacts(true);
}

/// A point in a roomy underground void near the spawn, open for a metre
/// in every direction.
fn underground_void(field: &TerrainField) -> DVec3 {
    let spawn = field.safe_spawn().0;
    for ring in 0..200 {
        for (dx, dz) in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            let (x, z) = (
                spawn.x + dx * f64::from(ring) * 6.0,
                spawn.z + dz * f64::from(ring) * 6.0,
            );
            let surface = field.surface_height(x, z);
            let mut depth = 4.0;
            while depth < 70.0 {
                let point = DVec3::new(x, surface - depth, z);
                let roomy = [
                    DVec3::ZERO,
                    DVec3::X,
                    DVec3::NEG_X,
                    DVec3::Y,
                    DVec3::NEG_Y,
                    DVec3::Z,
                    DVec3::NEG_Z,
                ]
                .into_iter()
                .all(|offset| field.density(point + offset) < -1.0);
                if roomy && field.density(point) < -1.5 {
                    return point;
                }
                depth += 0.5;
            }
        }
    }
    panic!("no cave near spawn");
}

#[expect(
    clippy::too_many_lines,
    reason = "keep real-mesher setup and impact acceptance together"
)]
fn check_generated_impacts(caves: bool) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("generated terrain regression requires a real adapter");
    eprintln!("Generated terrain adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let field = TerrainField::new(WorldSeed(91));
    let terrain = TerrainOctree::default();
    let edits = terrain.snapshot();
    // The spawn is the world's guaranteed near-level ground.
    let spawn = field.safe_spawn().0;
    let mut fixtures: Vec<_> = [0.0, 1.6, -3.2]
        .into_iter()
        .filter(|_| !caves)
        .map(|offset| {
            let (x, z) = (spawn.x + offset, spawn.z + offset * 0.5);
            (
                format!("surface_{offset}"),
                DVec3::new(x, field.surface_height(x, z), z),
                Vec3::Y,
                120,
            )
        })
        .collect();
    let chamber = WorldPosition(underground_void(&field));
    for (name, direction) in [("cave_ceiling", DVec3::Y), ("cave_wall", DVec3::X)]
        .into_iter()
        .filter(|_| caves)
    {
        let hit = raycast_density(
            &TerrainScene {
                field: &field,
                edits: &terrain,
            },
            chamber,
            direction,
            64.0,
        )
        .expect("generated cave has a boundary");
        fixtures.push((name.to_owned(), hit.position.0, hit.normal, 10));
    }
    for (name, origin, normal, ticks) in fixtures {
        let brick = WorldPosition(origin).cell().unwrap().brick();
        let mut chunks = Vec::new();
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let node =
                        TerrainNodeId::leaf(BrickCoord::new(brick.x + x, brick.y + y, brick.z + z));
                    let mesh = mesh_chunk(
                        &field,
                        &edits,
                        TerrainMeshRequest {
                            node,
                            generation: 1,
                            transition_mask: TerrainTransitionMask::NONE,
                        },
                    );
                    chunks.push(mesh.collision_chunk());
                }
            }
        }
        let triangle_count: usize = chunks.iter().map(|chunk| chunk.indices.len() / 3).sum();
        assert!(triangle_count > 1000);
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut creation = graph.compile().unwrap();
        let orientation = Quat::from_rotation_arc(Vec3::Y, normal);
        for body in &mut creation.compounds {
            body.root_translation = orientation * body.root_translation;
            body.root_rotation = orientation * body.root_rotation;
        }
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
                ..Default::default()
            },
        )
        .unwrap();
        gpu.write_terrain_chunks(&device, &queue, &chunks, origin)
            .unwrap();
        gpu.enable_async_readback();
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            creation.compounds[0].root_translation,
            -normal * 20.0 * creation.compounds[0].mass_properties.mass,
        )
        .unwrap();
        let mut maximum_penetration = 0.0_f32;
        let mut maximum_contacts = 0;
        for tick in 1..=ticks {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let state = gpu.poll_tick_readback(&device).unwrap().unwrap();
            assert_eq!(state.diagnostics.error_flags, 0, "{name}, tick {tick}");
            maximum_contacts = maximum_contacts.max(state.diagnostics.contact_count);
            let pose = state.transforms[0];
            let penetration = super::tests::box_terrain_penetration(
                &chunks,
                origin,
                Vec3::from_slice(&pose.position[..3]),
                Quat::from_array(pose.rotation),
            );
            assert!(
                penetration.is_finite(),
                "body left the meshed fixture at tick {tick}"
            );
            maximum_penetration = maximum_penetration.max(penetration);
            assert!(
                penetration <= if tick > 100 { 0.002 } else { 0.005 },
                "{name}, tick {tick}: penetration {penetration}"
            );
            if normal == Vec3::Y {
                assert!(
                    state.velocities[0].linear[1] <= 0.001,
                    "{name}, tick {tick}: upward velocity {}",
                    state.velocities[0].linear[1]
                );
            } else if tick == 1 {
                // An off-centre contact can leave COM motion toward the wall
                // while its contact point pivots. Check dissipated kinetic energy
                // instead. For this uniform 1 m cube, I / mass = 1/6 on every axis.
                let velocity = state.velocities[0];
                let energy_per_half_mass = Vec3::from_slice(&velocity.linear[..3]).length_squared()
                    + Vec3::from_slice(&velocity.angular[..3]).length_squared() / 6.0;
                assert!(state.diagnostics.active_contact_count > 0);
                assert!(
                    energy_per_half_mass < 0.05 * 20.0 * 20.0,
                    "{name}: impact retained too much kinetic energy: {energy_per_half_mass}"
                );
            }
        }
        assert!(maximum_contacts > 0, "{name}: no terrain contacts");
        eprintln!(
            "{{\"scenario\":\"{name}\",\"speed_mps\":20,\"triangles\":{triangle_count},\"maximum_contacts\":{maximum_contacts},\"maximum_penetration_m\":{maximum_penetration},\"ticks\":{ticks},\"failure_flags\":0}}"
        );
    }
}
