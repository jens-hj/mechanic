//! A tread changes only the contacts on the surface it is cut into: more grip
//! on ground that yields, a little less on rock, and harder pressure on both.

use super::terrain;
use crate::{BodyPose, MachineCollisionGeometry, TerrainContactQuery, TerrainContactScene};
use bevy_math::{DQuat, DVec3};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind, LayerFace,
    TreadPattern, TreadSpec,
};
use mechanic_world::TerrainMaterial;

const BLOCK: TreadSpec = match TreadSpec::new(TreadPattern::Block, 20) {
    Ok(tread) => tread,
    Err(_) => panic!("a valid tread"),
};

/// Contacts of a 1 m cube resting 1 mm into a floor of `material`, with
/// `tread` cut into `face`.
fn contacts(
    material: TerrainMaterial,
    face: FaceKind,
    tread: Option<TreadSpec>,
) -> TerrainContactQuery {
    let mut graph = ConstructionGraph::new();
    let spec = CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        unreachable!("spawning reports the part")
    };
    graph
        .apply(BuildCommand::SetTread {
            part,
            surface: LayerFace::Face(face),
            tread,
        })
        .unwrap();
    let creation = graph.compile().unwrap();
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    let poses = [BodyPose {
        position: DVec3::Y * 0.499,
        rotation: DQuat::IDENTITY,
    }];
    let mut scene = TerrainContactScene::default();
    scene.publish(1, &[terrain([material; 2])], &[]).unwrap();
    let query = scene.contacts(&geometry, &poses, DVec3::ZERO).unwrap();
    assert!(!query.contacts.is_empty());
    query
}

fn assert_scaled(treaded: f64, smooth: f64, factor: f32) {
    assert!(
        (treaded - smooth * f64::from(factor)).abs() < 1.0e-6,
        "{treaded} is not {smooth} × {factor}"
    );
}

#[test]
fn lugs_on_the_touching_face_bite_into_soil_and_press_harder() {
    let tread = BLOCK.response();
    let smooth = contacts(TerrainMaterial::Soil, FaceKind::NegativeY, None);
    let treaded = contacts(TerrainMaterial::Soil, FaceKind::NegativeY, Some(BLOCK));
    assert_eq!(smooth.contacts.len(), treaded.contacts.len());
    for (smooth, treaded) in smooth.contacts.iter().zip(&treaded.contacts) {
        assert!(smooth.yield_pa.is_finite());
        assert_scaled(
            treaded.response[0],
            smooth.response[0],
            tread.yielding_grip(),
        );
        assert_scaled(
            treaded.response[1],
            smooth.response[1],
            tread.yielding_grip(),
        );
        assert_scaled(treaded.failed_pa, smooth.failed_pa, tread.yielding_grip());
        assert!((treaded.yield_pa - smooth.yield_pa).abs() < 1.0e-6);
        assert!((treaded.pressure_factor - f64::from(tread.pressure_factor())).abs() < 1.0e-6);
        assert!((smooth.pressure_factor - 1.0).abs() < f64::EPSILON);
    }
}

#[test]
fn lugs_on_rock_give_up_a_little_grip() {
    let tread = BLOCK.response();
    let smooth = contacts(TerrainMaterial::Rock, FaceKind::NegativeY, None);
    let treaded = contacts(TerrainMaterial::Rock, FaceKind::NegativeY, Some(BLOCK));
    for (smooth, treaded) in smooth.contacts.iter().zip(&treaded.contacts) {
        assert!(!smooth.yield_pa.is_finite());
        assert_scaled(treaded.response[0], smooth.response[0], tread.firm_grip());
        assert!(treaded.response[0] < smooth.response[0]);
        assert!((treaded.response[2] - smooth.response[2]).abs() < f64::EPSILON);
    }
}

#[test]
fn a_tread_on_another_face_leaves_the_ground_contact_alone() {
    let smooth = contacts(TerrainMaterial::Soil, FaceKind::NegativeY, None);
    let topped = contacts(TerrainMaterial::Soil, FaceKind::PositiveY, Some(BLOCK));
    for (smooth, topped) in smooth.contacts.iter().zip(&topped.contacts) {
        for (smooth, topped) in smooth.response.iter().zip(topped.response) {
            assert!((smooth - topped).abs() < f64::EPSILON);
        }
        assert!((topped.pressure_factor - 1.0).abs() < f64::EPSILON);
    }
}
