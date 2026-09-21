//! The surface of a spiral cylinder: its drawn walls swept along the helix, cut
//! off at the end planes, narrowed by the taper, and capped.

use bevy::prelude::{Quat, Vec3};
use mechanic_core::{CylinderSpec, POSITION_TICK_METERS, SpiralProfile, SpiralSpec};
use std::f32::consts::TAU;

/// Angular columns per turn. Twice the fifteen-degree facets of a plain
/// cylinder, because a ridge's rim shows its facets against the sky.
const COLUMNS_PER_TURN: u16 = 48;

// A surface point in the cylinder's own terms, before any taper.
#[derive(Clone, Copy)]
struct Point {
    angle: f32,
    y: f32,
    radius: f32,
}

impl Point {
    fn lerp(self, other: Self, along: f32) -> Self {
        Self {
            angle: self.angle + (other.angle - self.angle) * along,
            y: self.y + (other.y - self.y) * along,
            radius: self.radius + (other.radius - self.radius) * along,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Wall {
    Outer,
    Bore,
}

// How a wall's radius changes along one profile segment.
#[derive(Clone, Copy)]
enum Slope {
    /// Depth gained per metre along the axis.
    Along(f32),
    /// A square flank facing up the axis (positive) or down it.
    Flank(f32),
}

struct Surface<'a> {
    spiral: SpiralSpec,
    center: Vec3,
    rotation: Quat,
    scale: f32,
    length: f32,
    outer_diameter: f32,
    positions: &'a mut Vec<[f32; 3]>,
    normals: &'a mut Vec<[f32; 3]>,
    indices: &'a mut Vec<u32>,
}

/// Appends the whole boundary of a cylinder that carries a spiral.
pub(crate) fn append_spiral_cylinder(
    center: Vec3,
    rotation: Quat,
    spec: CylinderSpec,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let Some(spiral) = spec.spiral() else {
        return;
    };
    let mut surface = Surface {
        spiral,
        center,
        rotation,
        scale,
        length: spec.dimensions.axial_length(),
        outer_diameter: spec.dimensions.outer_diameter(),
        positions,
        normals,
        indices,
    };
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let bore = spec.dimensions.inner_diameter() * 0.5;
    surface.wall(spiral.outer(), Wall::Outer, outer);
    if bore > 0.0 {
        surface.wall(spiral.inner(), Wall::Bore, bore);
    }
    for end in [-1.0, 1.0] {
        surface.cap(end, outer, bore);
    }
}

// The profile's points over one pitch and on to the first point of the next, in
// metres of (position, depth). An undrawn profile is one level stretch.
fn outline(profile: SpiralProfile, pitch: f32) -> Vec<[f32; 2]> {
    let mut points = profile
        .points()
        .iter()
        .map(|point| {
            [
                f32::from(point.position_ticks) * POSITION_TICK_METERS,
                f32::from(point.depth_ticks) * POSITION_TICK_METERS,
            ]
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        points.push([0.0, 0.0]);
    }
    points.push([points[0][0] + pitch, points[0][1]]);
    points
}

impl Surface<'_> {
    fn taper(&self, y: f32) -> f32 {
        self.spiral.taper_scale(y, self.length, self.outer_diameter)
    }

    // Heights the surface is cut at: both ends, and where a taper begins.
    fn spans(&self) -> Vec<[f32; 2]> {
        let half = self.length * 0.5;
        match self.spiral.taper() {
            Some(taper) => {
                let begins = taper.end.sign()
                    * (half - f32::from(taper.length_ticks) * POSITION_TICK_METERS);
                vec![[-half, begins], [begins, half]]
                    .into_iter()
                    .filter(|span| span[1] - span[0] > 1.0e-5)
                    .collect()
            }
            None => vec![[-half, half]],
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "a few pitches along one part"
    )]
    fn wall(&mut self, profile: SpiralProfile, wall: Wall, surface_radius: f32) {
        let pitch = self.spiral.pitch_meters();
        let half = self.length * 0.5;
        let points = outline(profile, pitch);
        let step = TAU / f32::from(COLUMNS_PER_TURN);
        let radius = |depth: f32| match wall {
            Wall::Outer => surface_radius - depth,
            Wall::Bore => surface_radius + depth,
        };
        for column in 0..COLUMNS_PER_TURN {
            let angles = [step * f32::from(column), step * f32::from(column + 1)];
            let advance = angles.map(|angle| self.spiral.advance_meters(angle));
            let lowest = advance[0].min(advance[1]);
            let highest = advance[0].max(advance[1]);
            let first = (-(points[points.len() - 1][0] + highest) / pitch).floor() as i16;
            let last = ((self.length - points[0][0] - lowest) / pitch).ceil() as i16;
            for ridge in first..=last {
                let base = -half + f32::from(ridge) * pitch;
                for pair in points.windows(2) {
                    let (from, to) = (pair[0], pair[1]);
                    if base + to[0] + highest < -half || base + from[0] + lowest > half {
                        continue;
                    }
                    let slope = if to[0] - from[0] > 1.0e-6 {
                        Slope::Along((to[1] - from[1]) / (to[0] - from[0]))
                    } else if (to[1] - from[1]).abs() > 1.0e-6 {
                        Slope::Flank((to[1] - from[1]).signum())
                    } else {
                        continue;
                    };
                    let corner = |side: usize, point: [f32; 2]| Point {
                        angle: angles[side],
                        y: base + point[0] + advance[side],
                        radius: radius(point[1]),
                    };
                    let quad = [
                        corner(0, from),
                        corner(0, to),
                        corner(1, to),
                        corner(1, from),
                    ];
                    for span in self.spans() {
                        self.patch(&clipped(&quad, span), wall, slope);
                    }
                }
            }
        }
    }

    // Outward normal of a wall at a point on a segment of the given slope.
    fn wall_normal(&self, point: Point, wall: Wall, slope: Slope, taper_slope: f32) -> Vec3 {
        let radial = Vec3::new(point.angle.cos(), 0.0, point.angle.sin());
        let around = Vec3::new(-point.angle.sin(), 0.0, point.angle.cos());
        // Axial advance per radian: the wall is a function of `y - turn * angle`.
        let turn = self.spiral.advance_meters(1.0);
        let scale = if wall == Wall::Outer {
            self.taper(point.y)
        } else {
            1.0
        };
        let radius = (point.radius * scale).max(1.0e-4);
        let facing = if wall == Wall::Outer { 1.0 } else { -1.0 };
        let local = match slope {
            // The outer wall loses radius with depth and the bore gains it.
            Slope::Along(depth_slope) => {
                let rise = -facing * depth_slope * scale
                    + if wall == Wall::Outer {
                        point.radius * taper_slope
                    } else {
                        0.0
                    };
                (radial - Vec3::Y * rise
                    + around * (rise_around(depth_slope, facing, scale, turn) / radius))
                    * facing
            }
            Slope::Flank(sign) => (Vec3::Y - around * (turn / radius)) * sign,
        };
        self.rotation * local.normalize_or_zero()
    }

    fn place(&self, point: Point, wall: Wall) -> Vec3 {
        let scale = if wall == Wall::Outer {
            self.taper(point.y)
        } else {
            1.0
        };
        let radius = point.radius * scale;
        self.center
            + self.rotation
                * (Vec3::new(
                    radius * point.angle.cos(),
                    point.y,
                    radius * point.angle.sin(),
                ) * self.scale)
    }

    fn patch(&mut self, polygon: &[Point], wall: Wall, slope: Slope) {
        if polygon.len() < 3 {
            return;
        }
        let (heights, count) = polygon.iter().fold((0.0, 0.0), |(sum, count), point| {
            (sum + point.y, count + 1.0)
        });
        let middle: f32 = heights / count;
        let taper_slope = (self.taper(middle + 1.0e-3) - self.taper(middle - 1.0e-3)) / 2.0e-3;
        let placed = polygon
            .iter()
            .map(|&point| self.place(point, wall))
            .collect::<Vec<_>>();
        let normals = polygon
            .iter()
            .map(|&point| self.wall_normal(point, wall, slope, taper_slope))
            .collect::<Vec<_>>();
        self.emit(&placed, &normals);
    }

    // A fan over a convex polygon, wound to face the way its normals point.
    fn emit(&mut self, placed: &[Vec3], normals: &[Vec3]) {
        let facing = (1..placed.len() - 1)
            .map(|index| (placed[index] - placed[0]).cross(placed[index + 1] - placed[0]))
            .sum::<Vec3>();
        if facing.length_squared() < 1.0e-14 {
            return;
        }
        let forward = facing.dot(normals.iter().copied().sum::<Vec3>()) >= 0.0;
        let first = u32::try_from(self.positions.len()).expect("mesh vertices fit u32");
        for (position, normal) in placed.iter().zip(normals) {
            self.positions.push(position.to_array());
            self.normals.push(normal.to_array());
        }
        for index in 1..u32::try_from(placed.len()).expect("a small polygon") - 1 {
            if forward {
                self.indices
                    .extend([first, first + index, first + index + 1]);
            } else {
                self.indices
                    .extend([first, first + index + 1, first + index]);
            }
        }
    }

    // One end plane: the ring between the bore wall and the outer wall as the
    // profiles leave them at that height.
    fn cap(&mut self, end: f32, outer: f32, bore: f32) {
        let pitch = self.spiral.pitch_meters();
        let y = end * self.length * 0.5;
        let height = y + self.length * 0.5;
        let scale = self.taper(y);
        let step = TAU / f32::from(COLUMNS_PER_TURN);
        // Columns, and every angle at which a profile point crosses the plane,
        // taken from either side so a square flank stays square.
        let mut angles = (0..=COLUMNS_PER_TURN)
            .map(|column| step * f32::from(column))
            .collect::<Vec<_>>();
        let turn = self.spiral.advance_meters(1.0);
        for profile in [self.spiral.outer(), self.spiral.inner()] {
            for point in outline(profile, pitch) {
                let crossing = ((height - point[0]) / turn).rem_euclid(pitch / turn.abs());
                let mut angle = crossing;
                while angle < TAU {
                    angles.extend([(angle - 1.0e-4).max(0.0), (angle + 1.0e-4).min(TAU)]);
                    angle += pitch / turn.abs();
                }
            }
        }
        angles.sort_by(f32::total_cmp);
        let normal = self.rotation * (Vec3::Y * end);
        let rim = |surface: &Self, angle: f32| {
            let phase = surface.spiral.phase_meters(angle, height);
            [
                (outer - surface.spiral.outer().depth_at(phase, pitch)) * scale,
                if bore > 0.0 {
                    bore + surface.spiral.inner().depth_at(phase, pitch)
                } else {
                    0.0
                },
            ]
        };
        for pair in angles.windows(2) {
            if pair[1] - pair[0] < 1.0e-6 {
                continue;
            }
            let at = |angle: f32, radius: f32| {
                self.center
                    + self.rotation
                        * (Vec3::new(radius * angle.cos(), y, radius * angle.sin()) * self.scale)
            };
            let (from, to) = (rim(self, pair[0]), rim(self, pair[1]));
            let ring = [
                at(pair[0], from[1]),
                at(pair[0], from[0]),
                at(pair[1], to[0]),
                at(pair[1], to[1]),
            ];
            self.emit(&ring, &[normal; 4]);
        }
    }
}

// How much a sloped wall leans around the axis: the wall is a function of
// `y - turn * angle`, so what it gains along the axis it loses around it.
fn rise_around(depth_slope: f32, facing: f32, scale: f32, turn: f32) -> f32 {
    -facing * depth_slope * scale * turn
}

// What of a polygon lies between two heights.
fn clipped(polygon: &[Point], span: [f32; 2]) -> Vec<Point> {
    let mut kept = polygon.to_vec();
    for (bound, keep_above) in [(span[0], true), (span[1], false)] {
        let inside = |point: &Point| {
            if keep_above {
                point.y >= bound - 1.0e-6
            } else {
                point.y <= bound + 1.0e-6
            }
        };
        let mut next = Vec::with_capacity(kept.len() + 2);
        for index in 0..kept.len() {
            let (current, following) = (kept[index], kept[(index + 1) % kept.len()]);
            if inside(&current) {
                next.push(current);
            }
            if inside(&current) != inside(&following) {
                let along = (bound - current.y) / (following.y - current.y);
                next.push(current.lerp(following, along));
            }
        }
        kept = next;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildPose, CylinderDimensions, SpiralEnd, SpiralHand, SpiralTaper};
    use std::f32::consts::PI;

    // Volume the mesh encloses, and the least agreement between a triangle's
    // facing and its vertices' normals.
    fn enclosed(spec: CylinderSpec) -> (f32, f32) {
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        append_spiral_cylinder(
            Vec3::ZERO,
            Quat::IDENTITY,
            spec,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let mut volume = 0.0;
        let mut agreement = 1.0_f32;
        for triangle in indices.chunks_exact(3) {
            let [a, b, c] =
                [0, 1, 2].map(|corner| Vec3::from(positions[triangle[corner] as usize]));
            volume += a.dot(b.cross(c)) / 6.0;
            let facing = (b - a).cross(c - a);
            if facing.length() > 1.0e-7 {
                for corner in 0..3 {
                    let normal = Vec3::from(normals[triangle[corner] as usize]);
                    assert!((normal.length() - 1.0).abs() < 1.0e-3);
                    agreement = agreement.min(facing.normalize().dot(normal));
                }
            }
        }
        (volume, agreement)
    }

    fn cylinder(outer: f32, inner: f32, length: f32) -> CylinderSpec {
        CylinderSpec::new(
            CylinderDimensions::new(outer, inner, length).unwrap(),
            BuildPose::default(),
        )
    }

    #[test]
    fn an_auger_mesh_encloses_its_core_and_its_ridge_facing_outwards() {
        for hand in [SpiralHand::Right, SpiralHand::Left] {
            let spiral = SpiralSpec::new(
                100,
                1,
                hand,
                SpiralProfile::square(10, 60).unwrap(),
                SpiralProfile::PLAIN,
                None,
            )
            .unwrap();
            let (volume, agreement) =
                enclosed(cylinder(0.5, 0.0, 2.0).with_spiral(spiral).unwrap());
            let expected = PI * 0.1 * 0.1 * 2.0 + 8.0 * 0.025 * PI * (0.25 * 0.25 - 0.1 * 0.1);
            assert!(
                (volume - expected).abs() < expected * 0.02,
                "{volume} against {expected}"
            );
            assert!(agreement > 0.7, "a normal leans {agreement} off its face");
        }
    }

    #[test]
    fn a_threaded_bore_mesh_encloses_the_tube_less_its_groove() {
        let spiral = SpiralSpec::new(
            100,
            2,
            SpiralHand::Right,
            SpiralProfile::PLAIN,
            SpiralProfile::vee(20, 10).unwrap(),
            None,
        )
        .unwrap();
        let (volume, agreement) = enclosed(cylinder(0.5, 0.3, 1.0).with_spiral(spiral).unwrap());
        let ridge = 4.0 * (0.5 * 0.05 * 0.025) * 2.0 * PI * (0.175 - 0.025 / 3.0);
        let expected = PI * (0.25 * 0.25 - 0.175 * 0.175) + ridge;
        assert!(
            (volume - expected).abs() < expected * 0.02,
            "{volume} against {expected}"
        );
        assert!(agreement > 0.7, "a normal leans {agreement} off its face");
    }

    #[test]
    fn a_taper_alone_makes_a_cone_frustum() {
        let spiral = SpiralSpec::new(
            100,
            1,
            SpiralHand::Right,
            SpiralProfile::PLAIN,
            SpiralProfile::PLAIN,
            Some(SpiralTaper {
                end: SpiralEnd::NegativeY,
                length_ticks: 400,
                tip_diameter_ticks: 50,
            }),
        )
        .unwrap();
        let (volume, agreement) = enclosed(cylinder(0.5, 0.0, 1.0).with_spiral(spiral).unwrap());
        let expected = PI / 3.0 * (0.25 * 0.25 + 0.25 * 0.0625 + 0.0625 * 0.0625);
        assert!(
            (volume - expected).abs() < expected * 0.02,
            "{volume} against {expected}"
        );
        assert!(agreement > 0.7, "a normal leans {agreement} off its face");
    }
}
