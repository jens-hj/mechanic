//! Replays the preserved builder at matched scales without editing its save.
use super::{
    Arc, ConstructionGraph, CpuMachine, DVec3, Error, GRAVITY, IVec3, Instant,
    MachineCollisionGeometry, MachineState, SoftStepConfig, SoftStepTerrain, TerrainContactScene,
    TerrainNodeId, extent, ground, json, lowest_point,
};
use mechanic_core::{CreationDocument, PartDoc, RigidLinkDoc};
use mechanic_physics::{ExternalImpulse, MachineKinematics};
use mechanic_world::{
    BRICK_EDGE_METERS, BrickCoord, TerrainField, TerrainMeshRequest, TerrainOctree,
    TerrainTransitionMask, WorldDocument, mesh_chunk,
};

pub(super) struct Options {
    pub copies: usize,
    pub connected: bool,
    pub warmup: u64,
    pub ticks: u64,
    pub floor: bool,
    pub hold: bool,
}

#[allow(clippy::too_many_lines)] // Explicit replay protocol and raw evidence in one ordered pass.
pub(super) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    if options.floor && options.copies != 1 {
        return Err(
            "the diagnostic floor supports one copy; use saved terrain for scale runs".into(),
        );
    }
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../tests/fixtures/builder-world/generations/20/world.ron"
    ))?;
    let world: WorldDocument = ron::from_str(include_str!(
        "../../../tests/fixtures/builder-world/world.ron"
    ))?;
    let mut combined =
        CreationDocument::from_graph(&ConstructionGraph::new(), "Builder scale", &[]);
    let mut next_link = 1;
    let part_count = instance.creation.parts.len();
    // The first authored cuboid belongs to the builder's chassis. Rigid links
    // join copies of that same part; suspension bearing rows remain independent.
    if !matches!(
        instance.creation.parts.first(),
        Some(PartDoc::Cuboid { .. })
    ) {
        return Err("builder chassis identity changed".into());
    }
    for copy in 0..options.copies {
        let mut document = instance.creation.clone();
        document.remap_dimension_links(&mut next_link);
        document.transform_cardinal(0, IVec3::X * i32::try_from(copy)? * 64);
        combined.append(document)?;
        if options.connected && copy > 0 {
            combined.rigid_links.push(RigidLinkDoc {
                first: 0,
                second: u32::try_from(copy * part_count)?,
            });
        }
    }
    let loaded = combined.into_graph()?;
    let creation = loaded
        .graph
        .compile_with_suspension_sockets([], &loaded.sockets)?;
    let mut state = MachineState::at_rest(&creation);
    let origin = instance.root_pose.translation.0;
    if !instance.joint_coordinates.is_empty() {
        return Err("fixture joint-coordinate mapping requires review".into());
    }
    let geometry = MachineCollisionGeometry::new(&creation, 1)?;
    let mut scene = TerrainContactScene::default();
    let terrain_started = Instant::now();
    let mut region = None;
    if options.floor {
        let lowest = lowest_point(&creation, &state)?;
        for pose in &mut state.poses {
            pose.position.y -= lowest + 0.001;
        }
        scene.publish(1, &[ground(false)], &[])?;
    } else {
        let [minimum, maximum] = extent(&creation, &state)?;
        let minimum = minimum + origin - DVec3::splat(4.0);
        let maximum = maximum + origin + DVec3::splat(4.0);
        region = Some([minimum - origin, maximum - origin]);
        let field = TerrainField::new(world.seed);
        let edits = TerrainOctree::default().snapshot();
        let low = (minimum / BRICK_EDGE_METERS).floor().as_ivec3();
        let high = (maximum / BRICK_EDGE_METERS).floor().as_ivec3();
        let mut chunks = Vec::new();
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    let chunk = mesh_chunk(
                        &field,
                        &edits,
                        TerrainMeshRequest {
                            node: TerrainNodeId::leaf(BrickCoord::new(x, y, z)),
                            generation: 1,
                            transition_mask: TerrainTransitionMask::NONE,
                        },
                    );
                    if !chunk.vertices.is_empty() {
                        chunks.push(Arc::new(chunk.collision_chunk()));
                    }
                }
            }
        }
        scene.publish(1, &chunks, &[])?;
    }
    let terrain_ms = terrain_started.elapsed().as_secs_f64() * 1000.0;
    let roots = creation
        .dynamics
        .preorder
        .iter()
        .copied()
        .filter(|&body| creation.loop_topology.body_parents[body].is_root)
        .collect::<Vec<_>>();
    let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
    let settings = SoftStepConfig::default();
    println!(
        "{}",
        json!({"kind":"metadata", "scenario":"builder-scale", "copies":options.copies,"connected":options.connected,"hold":options.hold,"warmup_ticks":options.warmup,"measured_ticks":options.ticks,"terrain":if options.floor {"diagnostic-floor"} else {"saved-seed-leaf-mesh"},"terrain_generation_ms":terrain_ms,"fixture_generation":world.construction_generation,"instance_id":instance.id,"bodies":creation.compounds.len(),"velocities":creation.dynamics.elimination_parent.len(),"colliders":creation.colliders.len(),"coordinates":creation.dynamics.coordinate_velocities.len(),"route":"cpu","executable":std::env::current_exe()?})
    );
    let mut samples = Vec::new();
    let mut degraded = 0;
    let mut maximum_depth = 0.0_f64;
    let mut maximum_gap = 0.0_f64;
    let mut maximum_angle = 0.0_f64;
    let mut outside_region = false;
    for tick in 1..=options.warmup + options.ticks {
        let started = Instant::now();
        if options.hold && tick % 360 == 1 {
            let held = vec![true; creation.compounds.len()];
            let poses = machine.snapshot().state.poses.clone();
            machine.hold(&held, &poses)?;
        } else if options.hold && tick % 360 == 121 {
            machine.hold(
                &vec![false; creation.compounds.len()],
                &machine.snapshot().state.poses.clone(),
            )?;
        }
        let impulses = if tick % 120 == 60 {
            roots
                .iter()
                .map(|&body| ExternalImpulse {
                    tick,
                    topology_generation: 1,
                    body,
                    point: machine.snapshot().state.poses[body].position + DVec3::X * 0.2,
                    impulse: DVec3::NEG_Y
                        * f64::from(creation.compounds[body].mass_properties.mass)
                        * 0.5,
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        machine.step(
            GRAVITY,
            &settings,
            &impulses,
            &[],
            Some(SoftStepTerrain {
                scene: &scene,
                geometry: &geometry,
                topology_generation: 1,
                origin,
            }),
        )?;
        let conversion_started = Instant::now();
        let state = &machine.snapshot().state;
        let motions =
            MachineKinematics::published_motions(&creation, &state.poses, &state.velocities)?;
        let published = state
            .poses
            .iter()
            .zip(motions)
            .map(|(pose, motion)| {
                (
                    pose.position.as_vec3(),
                    pose.rotation.as_quat(),
                    motion.linear.as_vec3(),
                    motion.angular.as_vec3(),
                )
            })
            .collect::<Vec<_>>();
        std::hint::black_box(published);
        let conversion_ms = conversion_started.elapsed().as_secs_f64() * 1000.0;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        let d = machine.diagnostics();
        if let Some([minimum, maximum]) = region {
            outside_region |= state.poses.iter().any(|pose| {
                pose.position.cmplt(minimum).any() || pose.position.cmpgt(maximum).any()
            });
        }
        if tick > options.warmup {
            samples.push(elapsed);
            degraded += usize::from(d.degraded);
            maximum_depth = maximum_depth.max(d.maximum_penetration);
            maximum_gap = maximum_gap.max(d.closure_position_error);
            maximum_angle = maximum_angle.max(d.closure_angle_error);
        }
        println!(
            "{}",
            json!({"kind":"tick","tick":tick,"warmup":tick<=options.warmup,"duration_ms":elapsed,"conversion_ms":conversion_ms,"query_ms":d.query_ms,"dynamics_ms":d.dynamics_ms,"rows_ms":d.rows_ms,"constraints_ms":d.constraints_ms,"continuous_ms":d.continuous_ms,"continuous_shape_transformations":d.continuous_shape_transformations,"continuous_shape_cache_hits":d.continuous_shape_cache_hits,"continuous_hierarchy_node_pair_tests":d.continuous_hierarchy_node_pair_tests,"continuous_pose_evaluations":d.continuous_pose_evaluations,"continuous_velocity_evaluations":d.continuous_velocity_evaluations,"continuous_separation_evaluations":d.continuous_separation_evaluations,"continuous_collider_pair_candidates":d.continuous_collider_pair_candidates,"continuous_triangle_candidates":d.continuous_triangle_candidates,"contacts":d.contacts,"triangle_candidates":d.triangle_candidates,"collider_pair_candidates":d.collider_pair_candidates,"solver_scratch_bytes":d.solver_scratch_bytes,"solver_scratch_growth_bytes":d.solver_scratch_growth_bytes,"degraded":d.degraded,"degraded_reason":d.degraded_reason,"penetration_m":d.maximum_penetration,"closure_gap_m":d.closure_position_error,"closure_angle_rad":d.closure_angle_error,"state_hash":machine.snapshot().state_hash()})
        );
    }
    samples.sort_by(f64::total_cmp);
    let p95 = samples[(samples.len() * 95).div_ceil(100).saturating_sub(1)];
    println!(
        "{}",
        json!({"kind":"summary","p95_ms":p95,"degraded_ticks":degraded,"maximum_penetration_m":maximum_depth,"closure_gap_m":maximum_gap,"closure_angle_rad":maximum_angle,"outside_terrain_region":outside_region,"timing_gate":p95<=2.0 && degraded==0 && !outside_region,"rendering_measured":false})
    );
    Ok(())
}
