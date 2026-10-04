//! Pairs a query remembers across poses answer exactly as fresh queries do.

use super::super::*;
use crate::MachineState;
use bevy_math::{DQuat, EulerRot};
use mechanic_core::CompiledCreation;

fn builder() -> CompiledCreation {
    let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(include_str!(
        "../../../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
    ))
    .unwrap();
    let loaded = instance.creation.into_graph().unwrap();
    loaded
        .graph
        .compile_with_sockets([], &loaded.sockets)
        .unwrap()
}

#[test]
fn remembered_separations_find_every_contact_a_fresh_query_finds() {
    let creation = builder();
    let remembered = MachineCollisionGeometry::new(&creation, 1).unwrap();
    let rest = MachineState::at_rest(&creation).poses;
    let scene = TerrainContactScene::default();
    let mut answered = 0;
    for step in 0..16 {
        let t = f64::from(step);
        // Each body drifts and turns its own way, so pairs close, part and
        // slide past one another, sometimes faster than a witness can cover.
        let poses = rest
            .iter()
            .enumerate()
            .map(|(body, pose)| {
                let phase = f64::from(u32::try_from(body).unwrap());
                let turn = DQuat::from_euler(
                    EulerRot::XYZ,
                    0.004 * t * phase.sin(),
                    0.006 * t * phase.cos(),
                    if step % 5 == 4 { 0.2 } else { 0.002 * t },
                );
                BodyPose {
                    position: pose.position
                        + DVec3::new(0.002 * t * phase.cos(), 0.001 * t, -0.001 * t),
                    rotation: turn * pose.rotation,
                }
            })
            .collect::<Vec<_>>();
        let fresh = MachineCollisionGeometry::new(&creation, 1).unwrap();
        for margin in [0.0, 0.005, 0.02] {
            let expected = scene.proximity(&fresh, &poses, DVec3::ZERO, margin);
            let actual = scene.proximity(&remembered, &poses, DVec3::ZERO, margin);
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual.contacts, expected.contacts, "step {step}");
                    assert_eq!(
                        actual.collider_pair_candidates,
                        expected.collider_pair_candidates
                    );
                    answered += 1;
                }
                // A wide margin over nearly parallel faces can refuse a query;
                // it must refuse it either way.
                (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                (actual, expected) => panic!("step {step}: {actual:?} against {expected:?}"),
            }
        }
    }
    assert!(answered >= 40, "only {answered} queries answered");
}

#[test]
fn radix_sorted_pairs_match_a_comparison_sort() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move |below: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        usize::try_from(seed % u64::try_from(below).unwrap()).unwrap()
    };
    let mut scratch = Vec::new();
    for (count, rows) in [(0, 2), (10, 3), (600, 2), (5000, 1744), (5000, 70_000)] {
        let mut pairs = (0..count)
            .map(|_| {
                let first = next(rows);
                [first, next(rows)]
            })
            .collect::<Vec<_>>();
        let mut expected = pairs.clone();
        expected.sort_unstable();
        super::super::geometry::sort_pairs(&mut pairs, &mut scratch, rows);
        assert_eq!(pairs, expected);
    }
}
