//! Pipe runs, bends, junctions, and branches.

use super::bearings::{BearingAttachment, LinearAttachment, validate_linear_attachment};
use super::bounds::{parts_overlap, validate_part_in_bounds};
use super::faces::face_normal;
use super::grid::{snap_cardinal, snap_world_to_position_ticks};
use super::raycast::region_pieces;
use super::staging::{bearing_connected_weld_scope, connection_geometry_owner};
use super::welds::touching_face_pair;
use super::{
    ALL_FACES, CONTACT_EPSILON, CylinderPlacementCandidate, GRID_UNIT_METERS, PlacementBounds,
    PlacementError, PlacementSupport, Result, SurfaceHit, ToOwned, ToString, Vec, Vec3, format,
    vec,
};
use mechanic_core::{
    BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    CylinderDimensions, CylinderSpec, FaceKind, FaceOwner, FaceRef, GridRotation,
    POSITION_TICK_METERS, PartId, PartSpec, PipeArms, PipeBendDimensions, PipeBendSpec,
    PipeJunctionDimensions, PipeJunctionSpec, RigidLinkSpec, WeldSpec,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PipeRunAttachment<'a> {
    AutoWeld {
        source: FaceOwner,
    },
    Free,
    Linear(LinearAttachment<'a>),
    Bearing {
        source: FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
        rigid_targets: &'a [PartId],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PipeRunPiece {
    pub(crate) spec: PartSpec,
    pub(crate) inlet: FaceKind,
    pub(crate) outlet: FaceKind,
}

/// What joins two consecutive legs of a pipe run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipeNode {
    /// A 90-degree bend filling a square of `span` blocks.
    Bend { span: u8 },
}

impl PipeNode {
    /// Blocks of leg this node occupies, counted from its square's edge.
    pub(crate) const fn footprint_blocks(self, _outer_diameter: f32) -> u8 {
        match self {
            Self::Bend { span } => span,
        }
    }
}

/// Splits a pipe path into straight cylinders, block-span bends, and junctions.
///
/// `points` are the run start, each node's centre corner, and the run end.
/// Every straight left between node faces must be a whole number of blocks,
/// which holds when corners sit at the middle of their pipe channel.
pub(crate) fn pipe_run_pieces(
    points: &[Vec3],
    nodes: &[PipeNode],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
) -> Result<Vec<PipeRunPiece>, PlacementError> {
    if dimensions.sweep_angle_degrees() != 360 && !nodes.is_empty() {
        return Err(PlacementError::PipeRun(
            "partial-cylinder sectors support straight runs only".to_owned(),
        ));
    }
    let (directions, lengths) = pipe_path_segments(points, nodes)?;
    let fittings = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| {
            pipe_node_piece(
                node,
                points[index + 1],
                directions[index],
                directions[index + 1],
                dimensions,
                material,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut pieces = Vec::new();
    for segment in 0..directions.len() {
        append_pipe_segment(
            &mut pieces,
            points,
            &directions,
            &lengths,
            &fittings,
            dimensions,
            material,
            segment,
        )?;
    }
    if pieces.is_empty() {
        return Err(PlacementError::PipeRun(
            "pipe run contains no material".to_owned(),
        ));
    }
    for first in 0..pieces.len() {
        for second in first + 2..pieces.len() {
            if parts_overlap(pieces[first].spec, pieces[second].spec) {
                return Err(PlacementError::PipeRun(format!(
                    "pipe run intersects itself between pieces {} and {}",
                    first + 1,
                    second + 1
                )));
            }
        }
    }
    Ok(pieces)
}

/// Builds one node's fitting and the distance it trims from each adjacent leg.
pub(super) fn pipe_node_piece(
    node: PipeNode,
    corner: Vec3,
    incoming: Vec3,
    outgoing: Vec3,
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
) -> Result<(f32, PipeRunPiece), PlacementError> {
    let corner_ticks = snap_world_to_position_ticks(corner);
    match node {
        PipeNode::Bend { span } => {
            let bend = PipeBendDimensions::new(
                dimensions.outer_diameter(),
                dimensions.inner_diameter(),
                span,
            )
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
            let rotation = rotation_xy_to_directions(incoming, outgoing).ok_or_else(|| {
                PlacementError::PipeRun("pipe turn has no cardinal orientation".to_owned())
            })?;
            Ok((
                bend.radius(),
                PipeRunPiece {
                    spec: PartSpec::PipeBend(
                        PipeBendSpec::new(
                            bend,
                            BuildPose::from_position_ticks(corner_ticks, rotation),
                        )
                        .with_material(material),
                    ),
                    inlet: FaceKind::NegativeX,
                    outlet: FaceKind::PositiveY,
                },
            ))
        }
    }
}

/// Unrotated cube face whose outward normal points along a cardinal direction.
pub(crate) fn face_toward(direction: Vec3) -> FaceKind {
    let absolute = direction.abs();
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        if direction.x >= 0.0 {
            FaceKind::PositiveX
        } else {
            FaceKind::NegativeX
        }
    } else if absolute.y >= absolute.z {
        if direction.y >= 0.0 {
            FaceKind::PositiveY
        } else {
            FaceKind::NegativeY
        }
    } else if direction.z >= 0.0 {
        FaceKind::PositiveZ
    } else {
        FaceKind::NegativeZ
    }
}

pub(super) fn pipe_path_segments(
    points: &[Vec3],
    nodes: &[PipeNode],
) -> Result<(Vec<Vec3>, Vec<f32>), PlacementError> {
    if points.len() < 2 || nodes.len() + 2 != points.len() {
        return Err(PlacementError::PipeRun(
            "pipe run path and fitting counts do not match".to_owned(),
        ));
    }
    let mut directions = Vec::with_capacity(points.len() - 1);
    let mut lengths = Vec::with_capacity(points.len() - 1);
    for segment in points.windows(2) {
        let delta = segment[1] - segment[0];
        let length = delta.length();
        if length <= CONTACT_EPSILON {
            return Err(PlacementError::PipeRun(
                "pipe legs must have positive length".to_owned(),
            ));
        }
        let direction = delta / length;
        if !is_cardinal(direction) {
            return Err(PlacementError::PipeRun(
                "pipe legs must be grid-aligned".to_owned(),
            ));
        }
        directions.push(snap_cardinal(direction));
        lengths.push(length);
    }
    for (index, pair) in directions.windows(2).enumerate() {
        if pair[0].dot(pair[1]).abs() > CONTACT_EPSILON {
            return Err(PlacementError::PipeRun(format!(
                "bend {} must turn exactly 90°",
                index + 1
            )));
        }
    }
    Ok((directions, lengths))
}

#[expect(clippy::too_many_arguments)]
pub(super) fn append_pipe_segment(
    pieces: &mut Vec<PipeRunPiece>,
    points: &[Vec3],
    directions: &[Vec3],
    lengths: &[f32],
    fittings: &[(f32, PipeRunPiece)],
    dimensions: CylinderDimensions,
    material: mechanic_core::ConstructionMaterial,
    segment: usize,
) -> Result<(), PlacementError> {
    let start_trim = segment
        .checked_sub(1)
        .and_then(|node| fittings.get(node))
        .map_or(0.0, |fitting| fitting.0);
    let end_trim = fittings.get(segment).map_or(0.0, |fitting| fitting.0);
    let residual = lengths[segment] - start_trim - end_trim;
    if residual < -CONTACT_EPSILON {
        let required = start_trim + end_trim;
        return Err(PlacementError::PipeRun(format!(
            "leg {} needs {:.2} m clearance for adjacent fittings",
            segment + 1,
            required
        )));
    }
    let residual_blocks = residual / GRID_UNIT_METERS;
    if residual > CONTACT_EPSILON && (residual_blocks - residual_blocks.round()).abs() > 1.0e-3 {
        return Err(PlacementError::PipeRun(format!(
            "leg {} straight must be a whole number of blocks",
            segment + 1
        )));
    }
    if residual > CONTACT_EPSILON {
        let start = points[segment] + directions[segment] * start_trim;
        let end = points[segment + 1] - directions[segment] * end_trim;
        let pose = pose_for_axis_segment(start, end, directions[segment])?;
        let cylinder = CylinderSpec::new(
            CylinderDimensions::new(
                dimensions.outer_diameter(),
                dimensions.inner_diameter(),
                residual,
            )
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?
            .with_sweep_angle_degrees(dimensions.sweep_angle_degrees())
            .map_err(|error| PlacementError::PipeRun(error.to_string()))?,
            pose,
        )
        .with_material(material);
        pieces.push(PipeRunPiece {
            spec: PartSpec::Cylinder(cylinder),
            inlet: FaceKind::NegativeY,
            outlet: FaceKind::PositiveY,
        });
    }
    if let Some(&(_, fitting)) = fittings.get(segment) {
        pieces.push(fitting);
    }
    Ok(())
}

pub(crate) fn validate_pipe_run_in_bounds(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    for piece in pieces {
        validate_part_in_bounds(graph, piece.spec, bounds)?;
    }
    let existing = graph
        .parts()
        .filter(|(part, _)| graph.region_of(*part).is_none())
        .map(|(_, part)| match part {
            PartSpec::Cylinder(cylinder) => mechanic_core::cylinder_collider_count(*cylinder),
            PartSpec::PipeBend(_) => mechanic_core::PIPE_BEND_COLLIDER_COUNT,
            PartSpec::PipeJunction(junction) => junction.collider_count(),
            _ => 1,
        })
        .sum::<usize>()
        + graph
            .regions()
            .map(|(_, region)| region_pieces(region).len())
            .sum::<usize>();
    let required = existing
        + pieces
            .iter()
            .map(|piece| match piece.spec {
                PartSpec::Cylinder(_) => mechanic_core::CYLINDER_COLLIDER_COUNT,
                PartSpec::PipeBend(_) => mechanic_core::PIPE_BEND_COLLIDER_COUNT,
                PartSpec::PipeJunction(junction) => junction.collider_count(),
                _ => 0,
            })
            .sum::<usize>();
    if required > mechanic_core::MAX_COMPILED_COLLIDERS {
        return Err(PlacementError::PipeRun(format!(
            "pipe run needs {required} colliders; maximum is {}",
            mechanic_core::MAX_COMPILED_COLLIDERS
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn stage_pipe_run(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    attachment: PipeRunAttachment<'_>,
) -> Result<ConstructionGraph, PlacementError> {
    stage_pipe_run_in_bounds(graph, pieces, attachment, PlacementBounds::Garage)
}

#[expect(
    clippy::too_many_lines,
    reason = "validate and connect every pipe piece in one atomic transaction"
)]
pub(crate) fn stage_pipe_run_in_bounds(
    graph: &ConstructionGraph,
    pieces: &[PipeRunPiece],
    attachment: PipeRunAttachment<'_>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    let bearing = match attachment {
        PipeRunAttachment::Bearing {
            source,
            anchor,
            dimensions,
            rigid_targets,
        } => Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        PipeRunAttachment::Linear(linear) => {
            validate_linear_attachment(graph, linear)?;
            Some(BearingAttachment::from(linear))
        }
        PipeRunAttachment::Free | PipeRunAttachment::AutoWeld { .. } => None,
    };
    validate_pipe_run_in_bounds(graph, pieces, bounds)?;
    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope = match attachment {
        PipeRunAttachment::AutoWeld { source } => bearing_connected_weld_scope(graph, source),
        PipeRunAttachment::Free
        | PipeRunAttachment::Bearing { .. }
        | PipeRunAttachment::Linear(_) => None,
    };
    let mut staged = graph.begin_edit();
    let mut spawned = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let command = match piece.spec {
            PartSpec::Cylinder(spec) => BuildCommand::SpawnCylinder(spec),
            PartSpec::PipeBend(spec) => BuildCommand::SpawnPipeBend(spec),
            PartSpec::PipeJunction(spec) => BuildCommand::SpawnPipeJunction(spec),
            _ => unreachable!("pipe runs contain only straights, bends, and junctions"),
        };
        let BuildOutcome::Spawned(part) = staged
            .apply(command)
            .map_err(|error| PlacementError::Graph(error.to_string()))?
        else {
            unreachable!()
        };
        spawned.push(part);
    }
    let mut connections = pieces
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(spawned[index], pair[0].outlet),
                second: FaceRef::part(spawned[index + 1], pair[1].inlet),
            })
        })
        .collect::<Vec<_>>();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let target = FaceRef::part(spawned[0], pieces[0].inlet);
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(source, target, anchor, axis)
                .with_dimensions(dimensions)
                .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: spawned[0],
            })
        }));
    } else {
        for &part in &spawned {
            if weld_scope.is_none()
                && bounds == PlacementBounds::Garage
                && let Some((first, second)) =
                    touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
            let mut tested_owners = HashSet::new();
            for &other in &existing_parts {
                if weld_scope
                    .as_ref()
                    .is_some_and(|members| !members.contains(&other))
                    || !tested_owners.insert(connection_geometry_owner(&staged, other))
                {
                    continue;
                }
                if let Some((first, second)) =
                    touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
                {
                    connections.push(BuildCommand::Weld(WeldSpec { first, second }));
                }
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

/// Rebuilds a junction with one more open face, keeping every weld on it.
pub(super) fn open_pipe_junction_arm(
    graph: &ConstructionGraph,
    part: PartId,
    face: FaceKind,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    let Some(PartSpec::PipeJunction(junction)) = graph.part(part).copied() else {
        return Err(PlacementError::PipeRun(
            "pipe junction is no longer available".to_owned(),
        ));
    };
    ensure_pipe_part_replaceable(graph, part)?;
    let welds = welds_on_part(graph, part);
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Remove(part))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let BuildOutcome::Spawned(opened) = staged
        .apply(BuildCommand::SpawnPipeJunction(junction.with_arm(face)))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!("spawning a junction reports its part")
    };
    staged
        .apply_batch(
            welds
                .into_iter()
                .map(|(own, other)| {
                    BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(opened, own),
                        second: other,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok((staged.finish(), opened))
}

/// Each weld on `part` as its own face kind and the face it holds.
pub(super) fn welds_on_part(graph: &ConstructionGraph, part: PartId) -> Vec<(FaceKind, FaceRef)> {
    let owner = FaceOwner::Part(part);
    graph
        .welds()
        .filter_map(|(_, weld)| {
            if weld.first.owner == owner {
                Some((weld.first.face, weld.second))
            } else if weld.second.owner == owner {
                Some((weld.second.face, weld.first))
            } else {
                None
            }
        })
        .collect()
}

/// Pipe parts can only be swapped for new pieces when welds are all that hold them.
pub(super) fn ensure_pipe_part_replaceable(
    graph: &ConstructionGraph,
    part: PartId,
) -> Result<(), PlacementError> {
    let refuse = |why: &str| Err(PlacementError::PipeRun(why.to_owned()));
    if graph.region_of(part).is_some()
        || graph.part_frame(part) != Some(mechanic_core::ConstructionFrame::IDENTITY)
    {
        return refuse("pipes inside shaped regions or moving frames cannot branch");
    }
    if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
        return refuse("shaped pipes cannot branch");
    }
    let owner = FaceOwner::Part(part);
    if graph.bearings().any(|(_, bearing)| {
        bearing.source.owner == owner || bearing.target.is_some_and(|target| target.owner == owner)
    }) || graph
        .rigid_links()
        .any(|(_, link)| link.first == part || link.second == part)
    {
        return refuse("pipes with bearings or rigid links cannot branch");
    }
    Ok(())
}

/// Where a branch leaves an existing pipe part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipeBranchSite {
    /// Cut a tee into this straight pipe.
    Split(PartId),
    /// Open another arm on this junction.
    Extend(PartId),
}

impl PipeBranchSite {
    /// Part the branch replaces.
    pub(crate) const fn part(self) -> PartId {
        match self {
            Self::Split(part) | Self::Extend(part) => part,
        }
    }
}

/// A junction planned where a new pipe branches off an existing one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PipeBranch {
    pub(crate) site: PipeBranchSite,
    /// The junction once branched, with the new arm open.
    pub(crate) junction: PipeJunctionSpec,
}

/// Plans a branch where `hit` lands on the side of a straight pipe or a
/// junction, and the new pipe leaving it.
///
/// A straight pipe gets a tee on the channel cell nearest the hit. The new arm
/// takes the free direction facing `toward` most; each `turn` steps it on,
/// around the pipe for a tee, or to the next best facing free face.
pub(crate) fn pipe_branch_candidate(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
    toward: Vec3,
    turn: u8,
) -> Result<(CylinderPlacementCandidate, PipeBranch), PlacementError> {
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(PlacementError::CurvedSurface);
    };
    let (site, base, free) = match graph.part(part).copied() {
        Some(PartSpec::Cylinder(cylinder)) => {
            let (tee, axis) = tee_on_pipe(cylinder, hit.point)?;
            let best = [
                Vec3::X,
                Vec3::NEG_X,
                Vec3::Y,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
            ]
            .into_iter()
            .filter(|direction| direction.dot(axis).abs() < 0.5)
            .max_by(|left, right| left.dot(toward).total_cmp(&right.dot(toward)))
            .expect("three axes leave four perpendicular directions");
            let side = snap_cardinal(axis.cross(best));
            (
                PipeBranchSite::Split(part),
                tee,
                vec![best, side, -best, -side],
            )
        }
        Some(PartSpec::PipeJunction(junction)) => {
            let rotation = junction.pose.rotation.quaternion();
            let mut free = ALL_FACES
                .into_iter()
                .filter(|&face| !junction.arms.contains(face))
                .map(|face| snap_cardinal(rotation * face_normal(face)))
                .collect::<Vec<_>>();
            free.sort_by(|left, right| right.dot(toward).total_cmp(&left.dot(toward)));
            (PipeBranchSite::Extend(part), junction, free)
        }
        _ => return Err(PlacementError::CurvedSurface),
    };
    ensure_pipe_part_replaceable(graph, part)?;
    if free.is_empty() {
        return Err(PlacementError::PipeRun(
            "every face of this junction already has an arm".to_owned(),
        ));
    }
    let direction = free[usize::from(turn) % free.len()];
    let junction = base.with_arm(face_toward(
        base.pose.rotation.quaternion().inverse() * direction,
    ));
    let rotation = rotation_y_to_direction(direction).ok_or(PlacementError::CurvedSurface)?;
    // The branch keeps the junction's cross-section so it fits the new arm.
    let dimensions = CylinderDimensions::new(
        junction.dimensions.outer_diameter(),
        junction.dimensions.inner_diameter(),
        dimensions.axial_length(),
    )
    .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
    let arm_end = junction.pose.translation() + direction * junction.dimensions.half_side();
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(
            snap_world_to_position_ticks(arm_end + direction * dimensions.axial_length() * 0.5),
            rotation,
        ),
    );
    Ok((
        CylinderPlacementCandidate {
            spec,
            attached_face: FaceKind::NegativeY,
            anchor: Some(arm_end),
            support: PlacementSupport::Surface(FaceOwner::Part(part)),
        },
        PipeBranch { site, junction },
    ))
}

/// The tee a straight pipe takes on the channel cell nearest `point`, with
/// its two axial arms open, and the pipe's axis.
pub(super) fn tee_on_pipe(
    cylinder: CylinderSpec,
    point: Vec3,
) -> Result<(PipeJunctionSpec, Vec3), PlacementError> {
    if cylinder.dimensions.sweep_angle_degrees() != 360 {
        return Err(PlacementError::PipeRun(
            "only full pipes can branch".to_owned(),
        ));
    }
    let axis = snap_cardinal(cylinder.pose.rotation.quaternion() * Vec3::Y);
    let length = cylinder.dimensions.axial_length();
    let start = cylinder.pose.translation() - axis * length * 0.5;
    let offset = point - start;
    let radial = offset - axis * offset.dot(axis);
    if radial.length() < cylinder.dimensions.outer_diameter() * 0.25 {
        return Err(PlacementError::CurvedSurface);
    }
    let dimensions = PipeJunctionDimensions::new(
        cylinder.dimensions.outer_diameter(),
        cylinder.dimensions.inner_diameter(),
    )
    .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
    let half = dimensions.half_side();
    let free_cells = ((length - 2.0 * half) / GRID_UNIT_METERS + 1.0e-3).floor();
    if free_cells < 0.0 {
        return Err(PlacementError::PipeRun(
            "pipe is too short for a tee".to_owned(),
        ));
    }
    let cell = ((offset.dot(axis) - half) / GRID_UNIT_METERS)
        .round()
        .clamp(0.0, free_cells);
    let center = start + axis * (half + cell * GRID_UNIT_METERS);
    let tee = PipeJunctionSpec::new(
        dimensions,
        PipeArms::single(face_toward(-axis)).with(face_toward(axis)),
        BuildPose::from_position_ticks(
            snap_world_to_position_ticks(center),
            GridRotation::default(),
        ),
    )
    .with_material(cylinder.material)
    .with_appearance(cylinder.appearance);
    Ok((tee, axis))
}

/// Makes a planned branch's junction, keeping the welds already on the part
/// it replaces, and returns the graph with the junction's part.
pub(crate) fn apply_pipe_branch(
    graph: &ConstructionGraph,
    branch: PipeBranch,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    match branch.site {
        PipeBranchSite::Split(pipe) => split_pipe_for_branch(graph, pipe, branch.junction),
        PipeBranchSite::Extend(junction) => {
            let Some(PartSpec::PipeJunction(current)) = graph.part(junction).copied() else {
                return Err(PlacementError::PipeRun(
                    "pipe junction is no longer available".to_owned(),
                ));
            };
            let face = ALL_FACES
                .into_iter()
                .find(|&face| branch.junction.arms.contains(face) && !current.arms.contains(face))
                .ok_or_else(|| {
                    PlacementError::PipeRun("the junction already has that arm".to_owned())
                })?;
            open_pipe_junction_arm(graph, junction, face)
        }
    }
}

/// Replaces a straight pipe with up to two shorter straights around its tee,
/// moving each end weld onto whichever piece now carries that end.
pub(super) fn split_pipe_for_branch(
    graph: &ConstructionGraph,
    pipe: PartId,
    tee: PipeJunctionSpec,
) -> Result<(ConstructionGraph, PartId), PlacementError> {
    let Some(PartSpec::Cylinder(cylinder)) = graph.part(pipe).copied() else {
        return Err(PlacementError::PipeRun(
            "branched pipe is no longer available".to_owned(),
        ));
    };
    ensure_pipe_part_replaceable(graph, pipe)?;
    let axis = cylinder.pose.rotation.quaternion() * Vec3::Y;
    let length = cylinder.dimensions.axial_length();
    let start = cylinder.pose.translation() - axis * length * 0.5;
    let half = tee.dimensions.half_side();
    let along = (tee.pose.translation() - start).dot(axis);
    if along - half < -CONTACT_EPSILON || length - along - half < -CONTACT_EPSILON {
        return Err(PlacementError::PipeRun(
            "the tee no longer fits on this pipe".to_owned(),
        ));
    }
    let welds = welds_on_part(graph, pipe);
    if welds
        .iter()
        .any(|(face, _)| !matches!(face, FaceKind::NegativeY | FaceKind::PositiveY))
    {
        return Err(PlacementError::PipeRun(
            "pipes welded along their side cannot branch".to_owned(),
        ));
    }
    let graph_error = |error: mechanic_core::GraphError| PlacementError::Graph(error.to_string());
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::Remove(pipe))
        .map_err(graph_error)?;
    let mut spawn_straight = |from: f32, to: f32| -> Result<Option<PartId>, PlacementError> {
        if to - from <= CONTACT_EPSILON {
            return Ok(None);
        }
        let dimensions = CylinderDimensions::new(
            cylinder.dimensions.outer_diameter(),
            cylinder.dimensions.inner_diameter(),
            to - from,
        )
        .map_err(|error| PlacementError::PipeRun(error.to_string()))?;
        let spec = CylinderSpec::new(
            dimensions,
            BuildPose::from_position_ticks(
                snap_world_to_position_ticks(start + axis * ((from + to) * 0.5)),
                cylinder.pose.rotation,
            ),
        )
        .with_material(cylinder.material)
        .with_appearance(cylinder.appearance);
        match staged
            .apply(BuildCommand::SpawnCylinder(spec))
            .map_err(graph_error)?
        {
            BuildOutcome::Spawned(part) => Ok(Some(part)),
            _ => unreachable!("spawning a cylinder reports its part"),
        }
    };
    let before = spawn_straight(0.0, along - half)?;
    let after = spawn_straight(along + half, length)?;
    let BuildOutcome::Spawned(junction) = staged
        .apply(BuildCommand::SpawnPipeJunction(tee))
        .map_err(graph_error)?
    else {
        unreachable!("spawning a junction reports its part")
    };
    let (back, ahead) = (face_toward(-axis), face_toward(axis));
    let start_face = before.map_or(FaceRef::part(junction, back), |part| {
        FaceRef::part(part, FaceKind::NegativeY)
    });
    let end_face = after.map_or(FaceRef::part(junction, ahead), |part| {
        FaceRef::part(part, FaceKind::PositiveY)
    });
    let mut connections = welds
        .into_iter()
        .map(|(face, other)| {
            BuildCommand::Weld(WeldSpec {
                first: if face == FaceKind::NegativeY {
                    start_face
                } else {
                    end_face
                },
                second: other,
            })
        })
        .collect::<Vec<_>>();
    if let Some(before) = before {
        connections.push(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(before, FaceKind::PositiveY),
            second: FaceRef::part(junction, back),
        }));
    }
    if let Some(after) = after {
        connections.push(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(junction, ahead),
            second: FaceRef::part(after, FaceKind::NegativeY),
        }));
    }
    staged.apply_batch(connections).map_err(graph_error)?;
    Ok((staged.finish(), junction))
}

pub(super) fn is_cardinal(direction: Vec3) -> bool {
    let absolute = direction.abs();
    (absolute.max_element() - 1.0).abs() <= 1.0e-4
        && absolute.min_element() <= 1.0e-4
        && (absolute.x + absolute.y + absolute.z - 1.0).abs() <= 1.0e-4
}

pub(super) fn pose_for_axis_segment(
    start: Vec3,
    end: Vec3,
    direction: Vec3,
) -> Result<BuildPose, PlacementError> {
    let rotation = rotation_y_to_direction(direction).ok_or_else(|| {
        PlacementError::PipeRun("pipe leg has no cardinal orientation".to_owned())
    })?;
    let center_ticks = ((start + end) * 0.5 / POSITION_TICK_METERS)
        .round()
        .as_ivec3();
    Ok(BuildPose::from_position_ticks(center_ticks, rotation))
}

pub(super) fn rotation_y_to_direction(direction: Vec3) -> Option<GridRotation> {
    cardinal_rotations()
        .find(|rotation| (rotation.quaternion() * Vec3::Y).abs_diff_eq(direction, 1.0e-4))
}

pub(super) fn rotation_xy_to_directions(incoming: Vec3, outgoing: Vec3) -> Option<GridRotation> {
    cardinal_rotations().find(|rotation| {
        (rotation.quaternion() * Vec3::X).abs_diff_eq(incoming, 1.0e-4)
            && (rotation.quaternion() * Vec3::Y).abs_diff_eq(outgoing, 1.0e-4)
    })
}

pub(super) fn cardinal_rotations() -> impl Iterator<Item = GridRotation> {
    (0_u8..4).flat_map(|x| {
        (0_u8..4).flat_map(move |y| (0_u8..4).map(move |z| GridRotation::new(x, y, z)))
    })
}
