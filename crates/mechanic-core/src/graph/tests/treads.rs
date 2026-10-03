use super::{cube_at, spawn, spawn_cylinder};
use crate::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CylinderDimensions, FaceKind,
    FaceRef, GraphError, GridRotation, LayerFace, PartId, TreadError, TreadMask, TreadPattern,
    TreadSpec, WeldSpec,
};
use bevy_math::{IVec3, Vec3};

fn tread(pattern: TreadPattern) -> TreadSpec {
    TreadSpec::new(pattern, 12).unwrap()
}

fn cut(
    graph: &mut ConstructionGraph,
    part: PartId,
    surface: LayerFace,
    tread: Option<TreadSpec>,
) -> Result<BuildOutcome, GraphError> {
    graph.apply(BuildCommand::SetTread {
        part,
        surface,
        tread,
    })
}

/// A 1 m by 0.5 m wheel standing on its tread with its axis along world X.
fn wheel(inner_diameter: f32) -> (ConstructionGraph, PartId) {
    let mut graph = ConstructionGraph::new();
    let wheel = spawn_cylinder(
        &mut graph,
        CylinderDimensions::new(1.0, inner_diameter, 0.5).unwrap(),
        BuildPose::from_position_ticks(IVec3::new(0, 200, 0), GridRotation::new(0, 0, 1)),
    );
    (graph, wheel)
}

#[test]
fn a_tread_is_cut_into_one_surface_without_touching_the_part() {
    let mut graph = ConstructionGraph::new();
    let block = spawn(&mut graph, cube_at(0));
    let before = graph.compile().unwrap().compounds[0].mass_properties;
    let bottom = LayerFace::Face(FaceKind::NegativeY);
    assert_eq!(
        cut(&mut graph, block, bottom, Some(tread(TreadPattern::Block))),
        Ok(BuildOutcome::TreadUpdated)
    );
    let treads = graph.part_treads(block);
    assert_eq!(treads.get(bottom), Some(tread(TreadPattern::Block)));
    assert_eq!(treads.iter().count(), 1);
    assert_eq!(graph.part(block).unwrap().size_meters(), Vec3::ONE);
    assert_eq!(
        graph.compile().unwrap().compounds[0].mass_properties,
        before
    );

    assert_eq!(
        cut(&mut graph, block, bottom, None),
        Ok(BuildOutcome::TreadUpdated)
    );
    assert!(graph.part_treads(block).is_empty());
    assert_eq!(graph.treaded_parts().count(), 0);
}

#[test]
fn a_surface_the_part_does_not_have_takes_no_tread() {
    let mut graph = ConstructionGraph::new();
    let block = spawn(&mut graph, cube_at(0));
    let ribs = Some(tread(TreadPattern::Ribbed));
    assert_eq!(
        cut(&mut graph, block, LayerFace::OuterWall, ribs),
        Err(GraphError::Tread(TreadError::UnsupportedSurface))
    );
    let (mut graph, wheel) = wheel(0.0);
    assert_eq!(
        cut(&mut graph, wheel, LayerFace::Bore, ribs),
        Err(GraphError::Tread(TreadError::BoreRequired))
    );
    assert_eq!(
        cut(
            &mut graph,
            wheel,
            LayerFace::Face(FaceKind::PositiveX),
            ribs
        ),
        Err(GraphError::Tread(TreadError::UnsupportedSurface))
    );
    assert!(graph.part_treads(wheel).is_empty());
}

#[test]
fn teeth_recut_the_surfaces_a_tread_was_cut_into() {
    let mut graph = ConstructionGraph::new();
    let block = spawn(&mut graph, cube_at(0));
    let top = LayerFace::Face(FaceKind::PositiveY);
    cut(&mut graph, block, top, Some(tread(TreadPattern::Lugged))).unwrap();
    let rack = crate::RackSpec::new(4, FaceKind::PositiveY, crate::Axis::X).unwrap();
    let racked = graph
        .part(block)
        .unwrap()
        .as_cuboid()
        .unwrap()
        .with_rack(rack)
        .unwrap();
    graph
        .apply(BuildCommand::SetRack {
            part: block,
            spec: racked,
        })
        .unwrap();
    assert!(graph.part_treads(block).is_empty());
    assert_eq!(
        cut(&mut graph, block, top, Some(tread(TreadPattern::Lugged))),
        Err(GraphError::Tread(TreadError::ToothedPart))
    );
}

#[test]
fn a_removed_part_takes_its_treads_with_it() {
    let mut graph = ConstructionGraph::new();
    let block = spawn(&mut graph, cube_at(0));
    cut(
        &mut graph,
        block,
        LayerFace::Face(FaceKind::NegativeY),
        Some(tread(TreadPattern::Studded)),
    )
    .unwrap();
    graph.apply(BuildCommand::Remove(block)).unwrap();
    assert_eq!(graph.treaded_parts().count(), 0);
}

#[test]
fn every_row_of_a_treaded_wheel_knows_its_wall_from_its_caps_and_bore() {
    let (mut graph, wheel) = wheel(0.5);
    cut(
        &mut graph,
        wheel,
        LayerFace::OuterWall,
        Some(tread(TreadPattern::Chevron)),
    )
    .unwrap();
    let compiled = graph.compile().unwrap();
    let rows = compiled
        .colliders
        .iter()
        .filter(|collider| collider.source_part == wheel)
        .collect::<Vec<_>>();
    assert!(!rows.is_empty());
    let row = rows[0].treads.unwrap();
    assert!(rows.iter().all(|collider| collider.treads == Some(row)));
    let treads = compiled.treads[row as usize];

    // The wheel's axis lies along world X.
    let centre = treads.local_center;
    let chevron = Some(tread(TreadPattern::Chevron));
    let under = centre - Vec3::Y * 0.5;
    assert_eq!(treads.tread_at(under, -Vec3::Y), chevron);
    assert_eq!(
        treads.surface_at(centre + Vec3::Z * 0.5, Vec3::Z),
        LayerFace::OuterWall
    );
    assert_eq!(
        treads.surface_at(centre - Vec3::Y * 0.25, Vec3::Y),
        LayerFace::Bore
    );
    assert_eq!(treads.tread_at(centre - Vec3::Y * 0.25, Vec3::Y), None);
    let cap = treads.surface_at(centre + Vec3::X * 0.25 + Vec3::Y * 0.4, Vec3::X);
    assert!(matches!(
        cap,
        LayerFace::Face(FaceKind::PositiveY | FaceKind::NegativeY)
    ));
    assert_eq!(treads.tread_at(centre + Vec3::X * 0.25, Vec3::X), None);
}

#[test]
fn a_treaded_block_keeps_its_own_collider_beside_a_welded_neighbour() {
    let plain = |treaded: bool| {
        let mut graph = ConstructionGraph::new();
        let first = spawn(&mut graph, cube_at(0));
        let second = spawn(&mut graph, cube_at(4));
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(first, FaceKind::PositiveX),
                second: FaceRef::part(second, FaceKind::NegativeX),
            }))
            .unwrap();
        if treaded {
            let studs = TreadPattern::Custom(TreadMask::new(0x0303_0000_0303).unwrap());
            cut(
                &mut graph,
                second,
                LayerFace::Face(FaceKind::NegativeY),
                Some(TreadSpec::new(studs, 20).unwrap()),
            )
            .unwrap();
        }
        let compiled = graph.compile().unwrap();
        (second, compiled)
    };
    let (_, merged) = plain(false);
    assert_eq!(merged.colliders.len(), 1);
    let (second, split) = plain(true);
    assert_eq!(split.colliders.len(), 2);
    let treaded = split
        .colliders
        .iter()
        .find(|collider| collider.treads.is_some())
        .unwrap();
    assert_eq!(treaded.source_part, second);
    let treads = split.treads[treaded.treads.unwrap() as usize];
    assert_eq!(
        treads.surface_at(treads.local_center - Vec3::Y * 0.5, -Vec3::Y),
        LayerFace::Face(FaceKind::NegativeY)
    );
}
