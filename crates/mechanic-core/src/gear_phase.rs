//! Tooth phases: how far each gear is turned so its teeth interleave with its
//! partners' at the rest pose. This is rendering only. The solver's slip bias
//! keeps meshed parts where they started, so teeth that interleave at rest
//! stay interleaved while the machine runs.

use std::collections::{BTreeMap, VecDeque};

use bevy_math::{Quat, Vec3};

use crate::{
    ConstructionGraph, GEAR_TOOTH_CENTER_FRACTION, GearLinkKind, GearLinkSpec, GearMesh,
    GearMeshSide, PartId, PartSpec,
};

/// The angle each gear is turned about its axis so its teeth interleave with
/// its partners', for every gear in the graph.
///
/// Racks keep their teeth where they are cut and the gears on them turn to
/// suit; among gears, the lowest part of each train stays and the rest follow
/// it mesh by mesh. A train that closes on itself, such as a planetary set,
/// is phased along one path through it. Worm and screw meshes do not phase
/// their partners.
pub fn gear_phases(graph: &ConstructionGraph) -> BTreeMap<PartId, f32> {
    let edges = graph
        .gear_links()
        .filter_map(|(_, link)| {
            let mesh = graph.gear_mesh(*link).ok()?;
            matches!(mesh.kind, GearLinkKind::Gears | GearLinkKind::Rack).then_some((*link, mesh))
        })
        .collect::<Vec<_>>();
    let is_rack = |part: PartId| matches!(graph.part(part), Some(PartSpec::Cuboid(cuboid)) if cuboid.rack().is_some());
    let is_gear = |part: PartId| matches!(graph.part(part), Some(PartSpec::Cylinder(cylinder)) if cylinder.gear().is_some());
    let racks = graph
        .parts()
        .map(|(part, _)| part)
        .filter(|&part| is_rack(part));
    let gears = graph
        .parts()
        .map(|(part, _)| part)
        .filter(|&part| is_gear(part));

    let mut phases = BTreeMap::new();
    let mut queue = VecDeque::new();
    for root in racks.chain(gears) {
        if phases.contains_key(&root) {
            continue;
        }
        phases.insert(root, 0.0);
        queue.push_back(root);
        while let Some(part) = queue.pop_front() {
            for (link, mesh) in &edges {
                let Some(other) = link.other(part) else {
                    continue;
                };
                if phases.contains_key(&other) {
                    continue;
                }
                let phase = if is_rack(other) {
                    // Two racks on one gear: the second keeps its teeth too.
                    0.0
                } else {
                    let (mine, theirs) = sides(*link, *mesh, part);
                    let fraction = tooth_fraction(graph, part, phases[&part], mine, theirs);
                    // Two parts whose teeth are counted opposite ways along
                    // the common tangent, as on external gears turning
                    // against each other, see each other's pitch mirrored.
                    let opposed = tangent(graph, part, mine, theirs)
                        .dot(tangent(graph, other, theirs, mine))
                        < 0.0;
                    let fraction = if opposed { -fraction } else { fraction };
                    following_phase(graph, other, fraction, theirs, mine)
                };
                phases.insert(other, phase);
                queue.push_back(other);
            }
        }
    }
    phases.retain(|&part, _| is_gear(part));
    phases
}

/// The mesh sides of `part` and of its partner, in that order.
fn sides(link: GearLinkSpec, mesh: GearMesh, part: PartId) -> (GearMeshSide, GearMeshSide) {
    let [first, second] = if mesh.swapped {
        [mesh.sides[1], mesh.sides[0]]
    } else {
        mesh.sides
    };
    if link.first == part {
        (first, second)
    } else {
        (second, first)
    }
}

/// Where along a tooth pitch `part`'s pitch point falls, in [0, 1): 0 is a
/// tooth centre and 0.5 the gap between two teeth.
fn tooth_fraction(
    graph: &ConstructionGraph,
    part: PartId,
    phase: f32,
    mine: GearMeshSide,
    theirs: GearMeshSide,
) -> f32 {
    match graph.part(part) {
        Some(PartSpec::Cylinder(cylinder)) => {
            let Some(gear) = cylinder.gear() else {
                return 0.0;
            };
            let pitch = core::f32::consts::TAU / f32::from(gear.teeth());
            let angle = pitch_angle(graph, part, mine, theirs);
            ((angle - phase) / pitch - GEAR_TOOTH_CENTER_FRACTION).rem_euclid(1.0)
        }
        Some(PartSpec::Cuboid(cuboid)) => {
            let Some(rack) = cuboid.rack() else {
                return 0.0;
            };
            // The pitch point along the rack: the gear's centre projected on it.
            let local = world_rotation(graph, part).inverse()
                * (theirs.center - graph.part_frame_point(part, cuboid.pose.translation()));
            let position = local[rack.along().index()];
            ((position - rack.tooth_line(cuboid.pose)) / rack.pitch()).rem_euclid(1.0)
        }
        _ => 0.0,
    }
}

/// The phase that puts a tooth centre of `gear` where its partner, whose
/// pitch point falls at `fraction` of a tooth, has a gap.
fn following_phase(
    graph: &ConstructionGraph,
    gear: PartId,
    fraction: f32,
    mine: GearMeshSide,
    theirs: GearMeshSide,
) -> f32 {
    let Some(PartSpec::Cylinder(cylinder)) = graph.part(gear) else {
        return 0.0;
    };
    let Some(spec) = cylinder.gear() else {
        return 0.0;
    };
    let pitch = core::f32::consts::TAU / f32::from(spec.teeth());
    let angle = pitch_angle(graph, gear, mine, theirs);
    (angle - pitch * (GEAR_TOOTH_CENTER_FRACTION + fraction + 0.5)).rem_euclid(pitch)
}

/// The direction in which a part counts its teeth at its pitch point with
/// the partner, in the world: the way a gear's tooth angle increases there,
/// or the axis a rack's teeth run along.
fn tangent(
    graph: &ConstructionGraph,
    part: PartId,
    mine: GearMeshSide,
    theirs: GearMeshSide,
) -> Vec3 {
    let local = match graph.part(part) {
        Some(PartSpec::Cylinder(_)) => {
            let angle = pitch_angle(graph, part, mine, theirs);
            Vec3::new(-angle.sin(), 0.0, angle.cos())
        }
        Some(PartSpec::Cuboid(cuboid)) => {
            cuboid.rack().map_or(Vec3::ZERO, |rack| rack.along().unit())
        }
        _ => Vec3::ZERO,
    };
    world_rotation(graph, part) * local
}

/// The angle in a gear's own XZ plane, from its X axis towards Z, at which
/// its pitch point with the partner lies.
fn pitch_angle(
    graph: &ConstructionGraph,
    gear: PartId,
    mine: GearMeshSide,
    theirs: GearMeshSide,
) -> f32 {
    let direction = if theirs.pitch_radius == 0.0 {
        // A rack's side axis is its outward normal; the gear sits on it.
        -theirs.axis
    } else {
        // Towards the partner, except that a pinion inside a ring meets it on
        // the far side; internal teeth carry a negative radius.
        let offset = theirs.center - mine.center;
        let toward = offset - mine.axis * offset.dot(mine.axis);
        toward.normalize_or_zero() * theirs.pitch_radius.signum()
    };
    let local = world_rotation(graph, gear).inverse() * direction;
    local.z.atan2(local.x)
}

/// The rotation taking a part's local axes to the world at the rest pose.
fn world_rotation(graph: &ConstructionGraph, part: PartId) -> Quat {
    let rotation = graph
        .part(part)
        .map_or(Quat::IDENTITY, |spec| spec.pose().rotation.quaternion());
    graph
        .part_frame(part)
        .map_or(rotation, |frame| frame.rotation() * rotation)
}

impl ConstructionGraph {
    /// A point authored in `part`'s construction frame, in the world.
    fn part_frame_point(&self, part: PartId, point: Vec3) -> Vec3 {
        self.part_frame(part)
            .map_or(point, |frame| frame.point(point))
    }
}

#[cfg(test)]
mod tests;
