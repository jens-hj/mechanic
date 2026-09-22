//! Welds between touching faces and the rigid bodies they form.

use super::faces::{face_geometry_from_ref, overlap_center};
use super::{ALL_FACES, PlacementError, Result, ToString, Vec, vec};
use mechanic_core::{
    BuildCommand, ConstructionGraph, FaceKind, FaceOwner, FaceRef, PartId, PartSpec, WeldSpec,
};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
pub(crate) fn begin_weld(
    graph: &mut ConstructionGraph,
    face: FaceRef,
) -> Result<(), PlacementError> {
    graph
        .apply(BuildCommand::BeginPending(
            mechanic_core::PendingOperation::Weld(face),
        ))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(())
}

pub(crate) fn stage_weld_objects(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    if first == second || weld_body_owners(graph, first).contains(&second) {
        return Err(PlacementError::SameObject);
    }
    let Some((first_face, second_face)) = touching_weld_face_pair(graph, first, second) else {
        return Err(PlacementError::ObjectsDoNotTouch);
    };
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Weld(WeldSpec {
            first: first_face,
            second: second_face,
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

pub(super) fn touching_weld_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    let source = weld_body_owners(graph, first)
        .into_iter()
        .flat_map(|owner| owner_faces(graph, owner))
        .collect::<Vec<_>>();
    let destination = weld_body_owners(graph, second)
        .into_iter()
        .flat_map(|owner| owner_faces(graph, owner))
        .collect::<Vec<_>>();
    source.iter().find_map(|&first_face| {
        destination.iter().find_map(|&second_face| {
            overlap_center(
                &face_geometry_from_ref(first_face, Some(graph)),
                &face_geometry_from_ref(second_face, Some(graph)),
            )?;
            let mut mating = vec![second_face];
            mating.extend(
                destination
                    .iter()
                    .copied()
                    .filter(|face| *face != second_face),
            );
            graph.weld_contact_square(&source, &mating).ok()?;
            Some((first_face, second_face))
        })
    })
}

/// Every object that moves with `owner`, which is what the weld tool treats as
/// one selection.
pub(super) fn weld_body_owners(graph: &ConstructionGraph, owner: FaceOwner) -> Vec<FaceOwner> {
    match owner {
        FaceOwner::Ground => vec![FaceOwner::Ground],
        FaceOwner::Part(part) => rigid_body_parts(graph, part)
            .into_iter()
            .map(FaceOwner::Part)
            .collect(),
    }
}

pub(crate) fn rigid_body_parts(graph: &ConstructionGraph, seed: PartId) -> Vec<PartId> {
    if graph.part(seed).is_none() {
        return Vec::new();
    }

    let mut neighbours = HashMap::<PartId, Vec<PartId>>::new();
    for (_, weld) in graph.welds() {
        if let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        {
            neighbours.entry(first).or_default().push(second);
            neighbours.entry(second).or_default().push(first);
        }
    }
    for (_, link) in graph.rigid_links() {
        neighbours.entry(link.first).or_default().push(link.second);
        neighbours.entry(link.second).or_default().push(link.first);
    }

    let mut members = HashSet::from([seed]);
    let mut pending = vec![seed];
    while let Some(part) = pending.pop() {
        if let Some(connected) = neighbours.get(&part) {
            for &candidate in connected {
                if members.insert(candidate) {
                    pending.push(candidate);
                }
            }
        }
    }

    graph
        .parts()
        .filter_map(|(part, _)| members.contains(&part).then_some(part))
        .collect()
}

pub(super) fn touching_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    owner_faces(graph, first)
        .into_iter()
        .find_map(|first_face| {
            owner_faces(graph, second)
                .into_iter()
                .find_map(|second_face| {
                    overlap_center(
                        &face_geometry_from_ref(first_face, Some(graph)),
                        &face_geometry_from_ref(second_face, Some(graph)),
                    )
                    .map(|_| (first_face, second_face))
                })
        })
}

pub(super) fn owner_faces(graph: &ConstructionGraph, owner: FaceOwner) -> Vec<FaceRef> {
    match owner {
        FaceOwner::Ground => vec![FaceRef::ground()],
        FaceOwner::Part(part) => match graph.part(part).copied() {
            Some(
                PartSpec::Cuboid(_)
                | PartSpec::Controller(_)
                | PartSpec::Engine(_)
                | PartSpec::Transmission(_)
                | PartSpec::Servo(_)
                | PartSpec::Seat(_)
                | PartSpec::Dial(_)
                | PartSpec::Button(_)
                | PartSpec::Input(_)
                | PartSpec::DimensionLink(_),
            ) => ALL_FACES
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::Cylinder(_)) => [FaceKind::PositiveY, FaceKind::NegativeY]
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::PipeJunction(junction)) => junction
                .arms
                .faces()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::PipeBend(_)) => [FaceKind::NegativeX, FaceKind::PositiveY]
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            None => Vec::new(),
        },
    }
}
