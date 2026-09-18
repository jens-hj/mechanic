use crate::pose::gpu_transform as pose;
use bevy::prelude::{IVec3, Mat3, Quat, Vec3};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionFrame, ConstructionGraph, CuboidSpec,
    GridRotation, PartId, RigidLinkSpec,
};
use mechanic_gpu::{GpuTransform, GpuVelocity};

use crate::simulation::publication::rebuilt_body_states;
use crate::simulation::state::{AppSimulation, LivePhysicsState};

fn spawn(graph: &mut ConstructionGraph, units: IVec3, size: [u8; 3]) -> PartId {
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(size, BuildPose::new(units, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    part
}

fn velocity(linear: Vec3, angular: Vec3) -> GpuVelocity {
    GpuVelocity {
        linear: linear.extend(0.0).to_array(),
        angular: angular.extend(0.0).to_array(),
    }
}

#[test]
fn failed_tick_keeps_the_last_completed_pose_even_between_visual_snapshots() {
    let airborne = pose(Vec3::Y * 5.0, Quat::IDENTITY);
    let landed = pose(Vec3::Y * 0.125, Quat::IDENTITY);
    let mut simulation = AppSimulation {
        transforms: vec![airborne],
        snapshot_tick: 10,
        live_state: Some(LivePhysicsState {
            tick: 11,
            transforms: vec![landed],
            velocities: vec![velocity(Vec3::ZERO, Vec3::ZERO)],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    let mut editor = crate::editor::state::EditorState::default();
    crate::simulation::state::stop_failed_simulation(
        &mut simulation,
        &mut editor,
        "tick failed".to_owned(),
    );
    assert_eq!(
        simulation.transforms[0].position.map(f32::to_bits),
        landed.position.map(f32::to_bits)
    );
    assert_eq!(
        simulation.previous_transforms[0].position.map(f32::to_bits),
        airborne.position.map(f32::to_bits)
    );
    assert_eq!(simulation.snapshot_tick, 11);
    assert!(simulation.render_dirty);
    assert_eq!(simulation.failure.as_deref(), Some("tick failed"));
    assert!(editor.feedback.unwrap().contains("tick failed"));
}

#[test]
fn reframed_replacement_chain_preserves_motion_and_keeps_local_pose_edits() {
    let mut graph = ConstructionGraph::new();
    let source = spawn(&mut graph, IVec3::new(4, 0, 0), [2, 4, 6]);
    let initial_frame =
        ConstructionFrame::new(Vec3::new(2.0, 3.0, -1.0), Quat::from_rotation_y(0.4)).unwrap();
    graph.reframe_parts([source], initial_frame).unwrap();
    let creation = graph.compile().unwrap();
    let old_root = creation.compounds[0].root_translation;
    let position = Vec3::new(7.0, 6.0, 5.0);
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);
    let linear = Vec3::new(2.0, 1.0, -3.0);
    let angular = Vec3::new(0.3, -0.2, 0.7);
    let previous = AppSimulation {
        creation: Some(creation),
        published_graph: graph.clone(),
        live_state: Some(LivePhysicsState {
            tick: 5,
            transforms: vec![pose(position, rotation)],
            velocities: vec![velocity(linear, angular)],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    graph
        .reframe_parts(
            [source],
            ConstructionFrame::new(Vec3::new(-3.0, 0.8, 2.0), Quat::from_rotation_x(-0.9)).unwrap(),
        )
        .unwrap();
    let frame = graph.part_frame_id(source).unwrap();
    graph.apply(BuildCommand::Remove(source)).unwrap();
    let intermediate = spawn(&mut graph, IVec3::new(4, 0, 0), [2, 4, 6]);
    graph.assign_part_frame(intermediate, frame).unwrap();
    graph.set_edit_source(intermediate, source).unwrap();
    graph.apply(BuildCommand::Remove(intermediate)).unwrap();
    let replacement = spawn(&mut graph, IVec3::new(5, 0, 0), [2, 4, 6]);
    graph.assign_part_frame(replacement, frame).unwrap();
    graph.set_edit_source(replacement, intermediate).unwrap();
    let rebuilt = graph.compile().unwrap();
    let (transforms, velocities) = rebuilt_body_states(&rebuilt, &graph, &previous);
    let displacement = rotation * (initial_frame.point(Vec3::new(1.25, 0.0, 0.0)) - old_root);
    assert!(
        Vec3::from_slice(&transforms[0].position[..3]).abs_diff_eq(position + displacement, 1.0e-5)
    );
    let visible_rotation = Quat::from_array(transforms[0].rotation)
        * graph.part_frame(replacement).unwrap().rotation();
    assert!(visible_rotation.abs_diff_eq(rotation * initial_frame.rotation(), 1.0e-5));
    assert!(
        Vec3::from_slice(&velocities[0].linear[..3])
            .abs_diff_eq(linear + angular.cross(displacement), 1.0e-5)
    );
    assert!(Vec3::from_slice(&velocities[0].angular[..3]).abs_diff_eq(angular, 1.0e-6));
}

#[test]
fn merging_reframed_bodies_preserves_visible_geometry_and_total_momentum() {
    let mut graph = ConstructionGraph::new();
    let a = spawn(&mut graph, IVec3::ZERO, [4, 4, 4]);
    let a_extra = spawn(&mut graph, IVec3::new(-4, 0, 0), [4, 4, 4]);
    let b = spawn(&mut graph, IVec3::new(8, 0, 0), [4, 2, 6]);
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: a,
            second: a_extra,
        }))
        .unwrap();
    let creation = graph.compile().unwrap();
    assert_eq!(creation.compounds.len(), 2);
    let relative =
        ConstructionFrame::new(Vec3::new(0.5, 0.2, 0.1), Quat::from_rotation_y(0.4)).unwrap();
    let world = ConstructionFrame::new(
        Vec3::new(10.0, 8.0, 6.0),
        Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.3),
    )
    .unwrap();
    let old_poses = creation
        .compounds
        .iter()
        .map(|compound| {
            let frame = if compound.source_parts.contains(&b) {
                world.compose(relative)
            } else {
                world
            };
            pose(frame.point(compound.root_translation), frame.rotation())
        })
        .collect::<Vec<_>>();
    let old_velocities = vec![
        velocity(Vec3::new(2.0, -1.0, 0.5), Vec3::new(0.1, 0.6, -0.4)),
        velocity(Vec3::new(-0.5, 2.0, 1.0), Vec3::new(-0.3, 0.2, 0.8)),
    ];
    let previous = AppSimulation {
        creation: Some(creation.clone()),
        published_graph: graph.clone(),
        live_state: Some(LivePhysicsState {
            tick: 8,
            transforms: old_poses.clone(),
            velocities: old_velocities.clone(),
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    graph.reframe_parts([b], relative).unwrap();
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: a,
            second: b,
        }))
        .unwrap();
    let rebuilt = graph.compile().unwrap();
    assert_eq!(rebuilt.compounds.len(), 1);
    crate::simulation::publication::validate_merged_body_poses(&rebuilt, &graph, &previous)
        .unwrap();
    let (transforms, velocities) = rebuilt_body_states(&rebuilt, &graph, &previous);
    let new_center = Vec3::from_slice(&transforms[0].position[..3]);
    let new_rotation = Quat::from_array(transforms[0].rotation);
    for part in [a, a_extra, b] {
        let old_body = creation
            .part_to_compound
            .iter()
            .find(|(id, _)| *id == part)
            .unwrap()
            .1 as usize;
        let old_pose = old_poses[old_body];
        let old_rotation = Quat::from_array(old_pose.rotation);
        let local_center = previous
            .published_graph
            .part(part)
            .unwrap()
            .pose()
            .translation();
        // Check the centre and an off-centre point, preserving orientation too.
        for point in [local_center, local_center + Vec3::new(0.25, -0.25, 0.25)] {
            let before = Vec3::from_slice(&old_pose.position[..3])
                + old_rotation * (point - creation.compounds[old_body].root_translation);
            let after = new_center
                + new_rotation
                    * (graph.part_frame(part).unwrap().point(point)
                        - rebuilt.compounds[0].root_translation);
            assert!(before.abs_diff_eq(after, 1.0e-5));
        }
    }
    let (momentum, spin) = total_momentum(&creation, &old_poses, &old_velocities, new_center);
    let mass = rebuilt.compounds[0].mass_properties;
    assert!(
        (mass.mass * Vec3::from_slice(&velocities[0].linear[..3]))
            .abs_diff_eq(momentum, momentum.length() * 1.0e-5)
    );
    let basis = Mat3::from_quat(new_rotation);
    let new_spin =
        basis * mass.inertia * basis.transpose() * Vec3::from_slice(&velocities[0].angular[..3]);
    assert!(new_spin.abs_diff_eq(spin, spin.length() * 1.0e-5));
}

fn total_momentum(
    creation: &mechanic_core::CompiledCreation,
    old_poses: &[GpuTransform],
    old_velocities: &[GpuVelocity],
    new_center: Vec3,
) -> (Vec3, Vec3) {
    let mut momentum = Vec3::ZERO;
    let mut spin = Vec3::ZERO;
    for (body, old) in creation.compounds.iter().enumerate() {
        let basis = Mat3::from_quat(Quat::from_array(old_poses[body].rotation));
        let p = old.mass_properties.mass * Vec3::from_slice(&old_velocities[body].linear[..3]);
        momentum += p;
        spin += basis
            * old.mass_properties.inertia
            * basis.transpose()
            * Vec3::from_slice(&old_velocities[body].angular[..3])
            + (Vec3::from_slice(&old_poses[body].position[..3]) - new_center).cross(p);
    }
    (momentum, spin)
}

fn moving_weld_fixture() -> (
    ConstructionGraph,
    mechanic_core::CompiledCreation,
    AppSimulation,
) {
    let mut graph = ConstructionGraph::new();
    let a = spawn(&mut graph, IVec3::ZERO, [4; 3]);
    let b = spawn(&mut graph, IVec3::new(4, 0, 0), [4; 3]);
    let old = graph.compile().unwrap();
    let at_click = old
        .compounds
        .iter()
        .map(|body| pose(body.root_translation, Quat::IDENTITY))
        .collect::<Vec<_>>();
    let world =
        ConstructionFrame::new(Vec3::new(3.0, 4.0, 5.0), Quat::from_rotation_z(0.6)).unwrap();
    let latest = old
        .compounds
        .iter()
        .map(|body| pose(world.point(body.root_translation), world.rotation()))
        .collect();
    let previous = AppSimulation {
        published_graph: graph.clone(),
        creation: Some(old),
        transforms: at_click,
        live_state: Some(LivePhysicsState {
            tick: 9,
            transforms: latest,
            velocities: vec![velocity(Vec3::X, Vec3::ZERO); 2],
            coordinates: Vec::new(),
        }),
        ..Default::default()
    };
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: a,
            second: b,
        }))
        .unwrap();
    let merged = graph.compile().unwrap();
    (graph, merged, previous)
}

#[test]
fn co_moving_sources_can_publish_a_weld_at_the_latest_tick() {
    let (graph, merged, mut previous) = moving_weld_fixture();
    crate::simulation::publication::validate_merged_body_poses(&merged, &graph, &previous).unwrap();
    // The quaternion double cover must not reject the same physical pose.
    let state = previous.live_state.as_mut().unwrap();
    state.transforms[1].rotation = (-Quat::from_array(state.transforms[1].rotation)).to_array();
    crate::simulation::publication::validate_merged_body_poses(&merged, &graph, &previous).unwrap();
}

#[test]
fn stale_moving_weld_rejects_translation_or_rotation_disagreement() {
    let (graph, merged, mut previous) = moving_weld_fixture();
    let compatible = previous.live_state.as_ref().unwrap().transforms.clone();
    previous.live_state.as_mut().unwrap().transforms[1].position[0] += 0.02;
    let error =
        crate::simulation::publication::validate_merged_body_poses(&merged, &graph, &previous)
            .unwrap_err();
    assert!(error.contains("moved apart"));
    // Rotate around the merged centre, so both mappings agree on position and
    // the orientation check alone must reject publication.
    let (poses, _) = rebuilt_body_states(&merged, &graph, &previous);
    let merged_center = Vec3::from_slice(&poses[0].position[..3]);
    let rotation = Quat::from_array(compatible[1].rotation) * Quat::from_rotation_y(0.02);
    let old_center = previous.creation.as_ref().unwrap().compounds[1].root_translation;
    previous.live_state.as_mut().unwrap().transforms[1] = pose(
        merged_center + rotation * (old_center - merged.compounds[0].root_translation),
        rotation,
    );
    let error =
        crate::simulation::publication::validate_merged_body_poses(&merged, &graph, &previous)
            .unwrap_err();
    assert!(error.contains("moved apart"));
    assert!(Quat::from_array(previous.transforms[1].rotation).abs_diff_eq(Quat::IDENTITY, 1.0e-6));
}
