//! The surface of a toothed part: a gear's teeth around its wall or bore, a
//! bevel gear's teeth on their cone, and a rack's teeth along a cuboid face.
//!
//! Teeth are trapezoids: a quarter pitch of tip, a quarter of root and two
//! sloped flanks. Everything else about the part is drawn as its envelope.

use bevy::prelude::{Quat, Vec2, Vec3};
use mechanic_core::{
    Axis, CuboidSpec, CylinderSpec, GEAR_TOOTH_CENTER_FRACTION, GearKind, GearSpec, PartId,
};
use std::collections::BTreeMap;
use std::f32::consts::TAU;

/// Smallest radius a bevel gear's small end shrinks to, as a share of the large end.
const BEVEL_SMALL_END: f32 = 0.35;

struct Builder<'a> {
    center: Vec3,
    rotation: Quat,
    scale: f32,
    positions: &'a mut Vec<[f32; 3]>,
    normals: &'a mut Vec<[f32; 3]>,
    indices: &'a mut Vec<u32>,
}

impl Builder<'_> {
    fn vertex(&mut self, local: Vec3, normal: Vec3) -> u32 {
        let world = self.center + self.rotation * (local * self.scale);
        let normal = self.rotation * normal;
        self.positions.push(world.to_array());
        self.normals.push(normal.normalize_or_zero().to_array());
        u32::try_from(self.positions.len() - 1).expect("mesh vertex count fits u32")
    }

    /// A flat triangle wound to face `normal`.
    fn triangle(&mut self, corners: [Vec3; 3], normal: Vec3) {
        let [a, b, c] = corners;
        let geometric = (b - a).cross(c - a);
        let (b, c) = if geometric.dot(normal) < 0.0 {
            (c, b)
        } else {
            (b, c)
        };
        let normal = if geometric == Vec3::ZERO {
            normal
        } else {
            geometric.normalize() * geometric.dot(normal).signum()
        };
        let first = self.vertex(a, normal);
        let second = self.vertex(b, normal);
        let third = self.vertex(c, normal);
        self.indices.extend([first, second, third]);
    }

    /// A flat quad, given in ring order, wound to face `normal`.
    fn quad(&mut self, corners: [Vec3; 4], normal: Vec3) {
        let [a, b, c, d] = corners;
        self.triangle([a, b, c], normal);
        self.triangle([a, c, d], normal);
    }
}

/// The tooth outline of a gear in its own XZ plane at the large end, four
/// points per tooth, as (angle, radius).
fn tooth_outline(gear: GearSpec) -> Vec<(f32, f32)> {
    let pitch = TAU / f32::from(gear.teeth());
    let (root, tip) = (gear.root_diameter() * 0.5, gear.tip_diameter() * 0.5);
    let mut outline = Vec::with_capacity(usize::from(gear.teeth()) * 4);
    for tooth in 0..gear.teeth() {
        let base = f32::from(tooth) * pitch;
        outline.extend([
            (base, root),
            (base + 0.25 * pitch, tip),
            (base + 0.5 * pitch, tip),
            (base + 0.75 * pitch, root),
        ]);
    }
    outline
}

fn radial(angle: f32, radius: f32) -> Vec3 {
    Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius)
}

/// The turn about its axis that interleaves a gear's teeth with its
/// partners', from [`mechanic_core::gear_phases`]; none for other parts. A
/// phase is measured from the gear's X axis towards its Z axis, as the tooth
/// outline is, which is a negative turn about Y.
pub(crate) fn tooth_phase(phases: &BTreeMap<PartId, f32>, part: PartId) -> Quat {
    phases
        .get(&part)
        .map_or(Quat::IDENTITY, |&phase| Quat::from_rotation_y(-phase))
}

/// Appends the whole boundary of a cylinder that carries teeth.
pub(crate) fn append_gear_cylinder(
    center: Vec3,
    rotation: Quat,
    spec: CylinderSpec,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let Some(gear) = spec.gear() else {
        return;
    };
    let mut builder = Builder {
        center,
        rotation,
        scale,
        positions,
        normals,
        indices,
    };
    let half = spec.dimensions.axial_length() * 0.5;
    let outline = tooth_outline(gear);
    let (outer, bore) = (
        spec.dimensions.outer_diameter() * 0.5,
        spec.dimensions.inner_diameter() * 0.5,
    );
    // The radius scale at each end: a bevel gear narrows from its large end.
    let ends = match gear.kind() {
        GearKind::Bevel {
            cone_angle_degrees,
            large_end,
        } => {
            let shrink =
                2.0 * half * f32::from(cone_angle_degrees).to_radians().tan() / gear.pitch_radius();
            let small = (1.0 - shrink).max(BEVEL_SMALL_END);
            let sign = large_end.sign();
            [(-sign * half, small), (sign * half, 1.0)]
        }
        GearKind::Spur | GearKind::Internal => [(-half, 1.0), (half, 1.0)],
    };
    let ring = |scale: f32| {
        outline
            .iter()
            .map(|&(angle, radius)| radial(angle, radius * scale))
            .collect::<Vec<_>>()
    };
    let [(y_low, scale_low), (y_high, scale_high)] = ends;
    let low = ring(scale_low);
    let high = ring(scale_high);
    let count = outline.len();
    let internal = gear.is_internal();
    // Tooth flanks, tips and roots between the two ends.
    for index in 0..count {
        let next = (index + 1) % count;
        let corners = [
            low[index].with_y(y_low),
            low[next].with_y(y_low),
            high[next].with_y(y_high),
            high[index].with_y(y_high),
        ];
        let outward = (corners[0] + corners[1] + corners[2] + corners[3]).with_y(0.0);
        let facing = if internal { -outward } else { outward };
        builder.quad(corners, facing);
    }
    // The wall the teeth are not on: the bore of an external gear, the outer
    // wall of a ring gear.
    let plain_radius = if internal { outer } else { bore };
    if plain_radius > 0.0 {
        for index in 0..count {
            let next = (index + 1) % count;
            let (a, b) = (
                radial(outline[index].0, plain_radius),
                radial(outline[next].0, plain_radius),
            );
            let corners = [
                a.with_y(y_low),
                b.with_y(y_low),
                b.with_y(y_high),
                a.with_y(y_high),
            ];
            let outward = (a + b).with_y(0.0);
            builder.quad(corners, if internal { outward } else { -outward });
        }
    }
    // End caps: a strip between the teeth and the plain wall, or a fan to the
    // axis of a solid gear.
    for (y, scale) in ends {
        let normal = Vec3::Y * y.signum();
        let teeth = ring(scale);
        for index in 0..count {
            let next = (index + 1) % count;
            if plain_radius > 0.0 {
                let plain = |index: usize| radial(outline[index].0, plain_radius).with_y(y);
                builder.quad(
                    [
                        teeth[index].with_y(y),
                        teeth[next].with_y(y),
                        plain(next),
                        plain(index),
                    ],
                    normal,
                );
            } else {
                builder.triangle(
                    [Vec3::Y * y, teeth[index].with_y(y), teeth[next].with_y(y)],
                    normal,
                );
            }
        }
    }
}

/// The local axis at right angles to a rack's face normal and tooth direction.
fn across_axis(face: Axis, along: Axis) -> Axis {
    match (face, along) {
        (Axis::X, Axis::Y) | (Axis::Y, Axis::X) => Axis::Z,
        (Axis::X, Axis::Z) | (Axis::Z, Axis::X) => Axis::Y,
        _ => Axis::X,
    }
}

/// Appends a cuboid with rack teeth cut into one face: the face sunk to the
/// tooth roots, and one prism per whole tooth standing on it.
pub(crate) fn append_rack_cuboid(
    center: Vec3,
    rotation: Quat,
    spec: CuboidSpec,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let Some(rack) = spec.rack() else {
        return;
    };
    let size = spec.size_meters();
    let face = rack.face();
    let normal = face.axis().unit() * face.sign();
    let along = rack.along().unit();
    let across = across_axis(face.axis(), rack.along()).unit();
    let depth = rack.root_depth();
    // The block, its toothed face sunk by the root depth.
    let mut sunk = size;
    sunk[face.axis().index()] -= depth;
    super::construction::append_transformed_cuboid(
        center - rotation * (normal * depth * 0.5 * scale),
        rotation,
        sunk * scale,
        positions,
        normals,
        indices,
    );
    let mut builder = Builder {
        center,
        rotation,
        scale,
        positions,
        normals,
        indices,
    };
    let half_along = size[rack.along().index()] * 0.5;
    let half_across = size[across_axis(face.axis(), rack.along()).index()] * 0.5;
    let top = size[face.axis().index()] * 0.5;
    let pitch = rack.pitch();
    let local = |u: f32, v: f32, height: f32| along * u + across * v + normal * (top - height);
    // Teeth continue the line shared with the block's neighbours, so the
    // ones at the ends may be cut by the block's edge.
    for tooth in rack.tooth_centers(spec.pose, 2.0 * half_along) {
        // Cross-section, base first, then up one flank, across the tip and down.
        let section = [
            Vec2::new(tooth - GEAR_TOOTH_CENTER_FRACTION * pitch, depth),
            Vec2::new(tooth - 0.125 * pitch, 0.0),
            Vec2::new(tooth + 0.125 * pitch, 0.0),
            Vec2::new(tooth + GEAR_TOOTH_CENTER_FRACTION * pitch, depth),
        ];
        let section = clipped_section(section, -half_along, half_along, depth);
        if section.len() < 3 {
            continue;
        }
        let at = |point: Vec2, v: f32| local(point.x, v, point.y);
        for pair in section.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let edge = Vec2::new(b.x - a.x, b.y - a.y);
            let facing = along * edge.y + normal * edge.x;
            builder.quad(
                [
                    at(a, -half_across),
                    at(b, -half_across),
                    at(b, half_across),
                    at(a, half_across),
                ],
                facing,
            );
        }
        for (v, facing) in [(-half_across, -across), (half_across, across)] {
            for corner in 1..section.len() - 1 {
                builder.triangle(
                    [
                        at(section[0], v),
                        at(section[corner], v),
                        at(section[corner + 1], v),
                    ],
                    facing,
                );
            }
        }
    }
}

/// A tooth's cross-section cut to the face between `low` and `high` along
/// it: the part of the profile inside, dropped to the root where the cut is.
/// The profile runs one way along the face.
fn clipped_section(section: [Vec2; 4], low: f32, high: f32, depth: f32) -> Vec<Vec2> {
    let crossing =
        |a: Vec2, b: Vec2, u: f32| Vec2::new(u, a.y + (b.y - a.y) * (u - a.x) / (b.x - a.x));
    let mut inside = Vec::with_capacity(6);
    if section[0].x < low {
        inside.push(Vec2::new(low, depth));
    }
    for (index, point) in section.into_iter().enumerate() {
        if let Some(&previous) = index.checked_sub(1).map(|index| &section[index]) {
            for edge in [low, high] {
                if previous.x < edge && point.x > edge {
                    inside.push(crossing(previous, point, edge));
                }
            }
        }
        if (low..=high).contains(&point.x) {
            inside.push(point);
        }
    }
    if section[3].x > high {
        inside.push(Vec2::new(high, depth));
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildPose, CylinderDimensions, FaceKind, RackSpec};

    fn bounds(positions: &[[f32; 3]]) -> (Vec3, Vec3) {
        positions.iter().fold(
            (Vec3::INFINITY, Vec3::NEG_INFINITY),
            |(low, high), &position| {
                let position = Vec3::from_array(position);
                (low.min(position), high.max(position))
            },
        )
    }

    #[test]
    fn a_tooth_phase_turns_the_outline_from_x_towards_z() {
        let mut graph = mechanic_core::ConstructionGraph::new();
        let mut spawn = || {
            let spec = CylinderSpec::new(
                CylinderDimensions::new(0.2, 0.0, 0.25).unwrap(),
                BuildPose::default(),
            );
            match graph
                .apply(mechanic_core::BuildCommand::SpawnCylinder(spec))
                .unwrap()
            {
                mechanic_core::BuildOutcome::Spawned(part) => part,
                outcome => panic!("{outcome:?}"),
            }
        };
        let (gear, plain) = (spawn(), spawn());
        let phases = BTreeMap::from([(gear, std::f32::consts::FRAC_PI_2)]);
        let turned = tooth_phase(&phases, gear) * radial(0.0, 1.0);
        assert!(turned.abs_diff_eq(Vec3::Z, 1.0e-6), "{turned}");
        assert!(radial(std::f32::consts::FRAC_PI_2, 1.0).abs_diff_eq(Vec3::Z, 1.0e-6));
        assert_eq!(tooth_phase(&phases, plain), Quat::IDENTITY);
    }

    #[test]
    fn a_spur_gear_fills_its_envelope_with_outward_facing_teeth() {
        let gear = GearSpec::new(4, 24, GearKind::Spur).unwrap();
        let spec = CylinderSpec::new(
            CylinderDimensions::new(gear.tip_diameter(), 0.1, 0.25).unwrap(),
            BuildPose::default(),
        )
        .with_gear(gear)
        .unwrap();
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        append_gear_cylinder(
            Vec3::ZERO,
            Quat::IDENTITY,
            spec,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        assert_eq!(indices.len() % 3, 0);
        let (low, high) = bounds(&positions);
        assert!((high.y - 0.125).abs() < 1.0e-6);
        let reach = positions
            .iter()
            .map(|position| position[0].hypot(position[2]))
            .fold(0.0_f32, f32::max);
        assert!(
            (reach - 0.13).abs() < 1.0e-5,
            "tips reach the outer wall: {reach}"
        );
        assert!(low.x < -0.129 && high.x > 0.129);
        // Every tooth flank faces away from the axis and every bore quad into it.
        for (position, normal) in positions.iter().zip(&normals) {
            let (position, normal) = (Vec3::from_array(*position), Vec3::from_array(*normal));
            if normal.y.abs() > 0.5 {
                continue;
            }
            let radial = position.with_y(0.0);
            let inward = radial.length() < 0.06;
            assert_eq!(normal.dot(radial) < 0.0, inward, "{position} {normal}");
        }
    }

    #[test]
    fn a_rack_cuts_teeth_along_its_face_and_the_end_ones_at_its_edge() {
        let spec = CuboidSpec::new([8, 1, 1], BuildPose::default())
            .unwrap()
            .with_rack(RackSpec::new(4, FaceKind::PositiveY, Axis::X).unwrap())
            .unwrap();
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        append_rack_cuboid(
            Vec3::ZERO,
            Quat::IDENTITY,
            spec,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let (low, high) = bounds(&positions);
        assert!((high.y - 0.125).abs() < 1.0e-6 && (low.y + 0.125).abs() < 1.0e-6);
        assert!(high.x <= 1.0 + 1.0e-6 && low.x >= -1.0 - 1.0e-6);
        // Teeth of 3.14 cm centred on the block's centre: 63 whole ones,
        // each three quads and two two-triangle caps, and one at each end
        // cut to a root, a flank and a triangle cap, on top of the sunk block.
        let (mut block, mut block_normals, mut block_indices) =
            (Vec::new(), Vec::new(), Vec::new());
        super::super::construction::append_transformed_cuboid(
            Vec3::ZERO,
            Quat::IDENTITY,
            spec.size_meters(),
            &mut block,
            &mut block_normals,
            &mut block_indices,
        );
        assert_eq!(
            indices.len() - block_indices.len(),
            63 * (3 * 6 + 2 * 6) + 2 * (2 * 6 + 2 * 3)
        );
    }

    #[test]
    fn a_cut_tooth_keeps_the_profile_inside_the_face_and_drops_to_its_root() {
        let section = [
            Vec2::new(-3.0, 1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(3.0, 1.0),
        ];
        assert_eq!(
            clipped_section(section, -10.0, 10.0, 1.0),
            section.to_vec(),
            "a whole tooth is untouched"
        );
        assert_eq!(
            clipped_section(section, -2.0, 2.0, 1.0),
            vec![
                Vec2::new(-2.0, 1.0),
                Vec2::new(-2.0, 0.5),
                Vec2::new(-1.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(2.0, 0.5),
                Vec2::new(2.0, 1.0),
            ],
            "both flanks cut halfway up"
        );
        assert_eq!(
            clipped_section(section, 0.0, 10.0, 1.0),
            vec![
                Vec2::new(0.0, 1.0),
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(3.0, 1.0),
            ],
            "cut through the tip"
        );
    }
}
