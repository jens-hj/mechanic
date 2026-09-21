//! Convex pieces of a spiral cylinder: ridges swept along the helix, and the
//! narrowing core under a taper. The straight core stays an ordinary cylinder.

use super::model::{SurfacePatchKey, TopologySource};
use super::polygon::{ClipPlane, PolyCell, PolyFace, clip_cell, poly_cell_to_convex};
use crate::{ConvexPiece, CylinderSpec, POSITION_TICK_METERS, SpiralProfile, SpiralSpec};
use bevy_math::DVec3;
use core::f64::consts::TAU;

// Points nearer than this are one hull corner, and lie in one hull face.
const HULL_EPSILON: f64 = 1.0e-7;

const NO_PATCH: SurfacePatchKey = SurfacePatchKey {
    source: TopologySource::Base,
    local: 0,
};

/// The straight, constant part of a spiral cylinder: everything beneath the
/// deepest cut, short of any taper. Local to the cylinder, about its Y axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpiralCore {
    /// Radius beneath the deepest outer cut, in metres.
    pub outer_radius: f32,
    /// Radius beyond the deepest bore cut, in metres. Zero when solid.
    pub inner_radius: f32,
    /// Axial length in metres.
    pub length: f32,
    /// Centre along local Y in metres.
    pub center_y: f32,
}

/// The straight core of a spiral cylinder, or `None` for a plain cylinder or
/// one tapered over its whole length.
pub fn spiral_core(spec: CylinderSpec) -> Option<SpiralCore> {
    let spiral = spec.spiral()?;
    let ticks = |ticks: u16| f32::from(ticks) * POSITION_TICK_METERS;
    let (taper_length, end) = spiral.taper().map_or((0.0, 1.0), |taper| {
        (ticks(taper.length_ticks), taper.end.sign())
    });
    let length = spec.dimensions.axial_length() - taper_length;
    (length > 1.0e-4).then(|| SpiralCore {
        outer_radius: spec.dimensions.outer_diameter() * 0.5
            - ticks(spiral.outer().max_depth_ticks()),
        inner_radius: if spec.dimensions.inner_diameter() > 0.0 {
            spec.dimensions.inner_diameter() * 0.5 + ticks(spiral.inner().max_depth_ticks())
        } else {
            0.0
        },
        length,
        center_y: -end * taper_length * 0.5,
    })
}

/// Everything of a spiral cylinder outside its straight core, as convex pieces
/// in build space: each ridge cut into `steps_per_turn` runs per turn, and the
/// core where a taper narrows it. Each piece is the hull of its swept corners,
/// so neighbours overlap by the sagitta of one step and no more.
pub fn spiral_pieces(spec: CylinderSpec, steps_per_turn: u16) -> Vec<ConvexPiece> {
    let Some(spiral) = spec.spiral() else {
        return Vec::new();
    };
    let sweep = Sweep::new(spec, spiral);
    let step = TAU / f64::from(steps_per_turn);
    let mut cells = Vec::new();
    for index in 0..steps_per_turn {
        let angles = [step * f64::from(index), step * f64::from(index + 1)];
        sweep.ridges(spiral.outer(), Wall::Outer, angles, &mut cells);
        sweep.ridges(spiral.inner(), Wall::Bore, angles, &mut cells);
        sweep.tapered_core(angles, &mut cells);
    }
    let rotation = spec.pose.rotation.quaternion().as_dquat();
    let translation = spec.pose.translation().as_dvec3();
    cells
        .into_iter()
        .filter_map(|mut cell| {
            for face in &mut cell.faces {
                for vertex in &mut face.vertices {
                    *vertex = translation + rotation * *vertex;
                }
            }
            poly_cell_to_convex(&cell)
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Wall {
    Outer,
    Bore,
}

struct Sweep {
    spiral: SpiralSpec,
    length: f64,
    outer_diameter: f32,
    outer_radius: f64,
    bore_radius: f64,
}

impl Sweep {
    fn new(spec: CylinderSpec, spiral: SpiralSpec) -> Self {
        Self {
            spiral,
            length: f64::from(spec.dimensions.axial_length()),
            outer_diameter: spec.dimensions.outer_diameter(),
            outer_radius: f64::from(spec.dimensions.outer_diameter()) * 0.5,
            bore_radius: f64::from(spec.dimensions.inner_diameter()) * 0.5,
        }
    }

    fn scale(&self, y: f64) -> f64 {
        f64::from(
            self.spiral
                .taper_scale(y as f32, self.length as f32, self.outer_diameter),
        )
    }

    // A wall point at a depth, tapered when it is the outer wall.
    fn point(&self, wall: Wall, angle: f64, y: f64, depth: f64) -> DVec3 {
        let radius = match wall {
            Wall::Outer => (self.outer_radius - depth) * self.scale(y),
            Wall::Bore => self.bore_radius + depth,
        };
        DVec3::new(radius * angle.cos(), y, radius * angle.sin())
    }

    fn ridges(
        &self,
        profile: SpiralProfile,
        wall: Wall,
        angles: [f64; 2],
        cells: &mut Vec<PolyCell>,
    ) {
        let pitch = f64::from(self.spiral.pitch_meters());
        let floor = f64::from(profile.max_depth_ticks()) * f64::from(POSITION_TICK_METERS);
        let half = self.length * 0.5;
        let advance = angles.map(|angle| f64::from(self.spiral.advance_meters(angle as f32)));
        let (lowest, highest) = (advance[0].min(advance[1]), advance[0].max(advance[1]));
        for (from, to) in profile.segments(self.spiral.pitch_meters()) {
            let (from, to) = (from.map(f64::from), to.map(f64::from));
            if from[1] >= floor - 1.0e-6 && to[1] >= floor - 1.0e-6 {
                continue;
            }
            let first = (-(to[0] + highest) / pitch).floor() as i32;
            let last = ((self.length - from[0] - lowest) / pitch).ceil() as i32;
            for ridge in first..=last {
                let base = -half + f64::from(ridge) * pitch;
                if base + to[0] + highest <= -half + HULL_EPSILON
                    || base + from[0] + lowest >= half - HULL_EPSILON
                {
                    continue;
                }
                let corners = (0..2)
                    .flat_map(|side| {
                        let y = |position: f64| base + position + advance[side];
                        [
                            self.point(wall, angles[side], y(from[0]), floor),
                            self.point(wall, angles[side], y(to[0]), floor),
                            self.point(wall, angles[side], y(to[0]), to[1]),
                            self.point(wall, angles[side], y(from[0]), from[1]),
                        ]
                    })
                    .collect::<Vec<_>>();
                cells.extend(hull(&corners).and_then(|cell| self.within_ends(cell)));
            }
        }
    }

    // The core where a taper narrows it, which no straight cylinder covers.
    fn tapered_core(&self, angles: [f64; 2], cells: &mut Vec<PolyCell>) {
        let Some(taper) = self.spiral.taper() else {
            return;
        };
        let tick = f64::from(POSITION_TICK_METERS);
        let half = self.length * 0.5;
        let end = f64::from(taper.end.sign());
        let ends = [
            end * (half - f64::from(taper.length_ticks) * tick),
            end * half,
        ];
        let floor = f64::from(self.spiral.outer().max_depth_ticks()) * tick;
        // A solid cylinder's bore wall is its axis.
        let bore = f64::from(self.spiral.inner().max_depth_ticks()) * tick;
        let corners = angles
            .iter()
            .flat_map(|&angle| {
                ends.iter().flat_map(move |&y| {
                    [
                        self.point(Wall::Bore, angle, y, bore),
                        self.point(Wall::Outer, angle, y, floor),
                    ]
                })
            })
            .collect::<Vec<_>>();
        cells.extend(hull(&corners));
    }

    // What of a cell lies between the cylinder's end planes.
    fn within_ends(&self, mut cell: PolyCell) -> Option<PolyCell> {
        let half = self.length * 0.5;
        for sign in [1.0, -1.0] {
            let beyond = cell
                .faces
                .iter()
                .flat_map(|face| &face.vertices)
                .any(|vertex| vertex.y * sign > half + HULL_EPSILON);
            if beyond {
                cell = clip_cell(
                    &cell,
                    ClipPlane {
                        normal: DVec3::Y * sign,
                        offset: half,
                        patch: NO_PATCH,
                        family: NO_PATCH,
                        smoothing_group: 0,
                        smooth_with: Vec::new(),
                        uv_provenance: NO_PATCH,
                    },
                )?;
            }
        }
        Some(cell)
    }
}

// The convex hull of a handful of points as a cell with planar, outward-wound
// faces. `None` when the points span no volume.
fn hull(points: &[DVec3]) -> Option<PolyCell> {
    let mut corners = Vec::<DVec3>::new();
    for &point in points {
        if corners
            .iter()
            .all(|corner| corner.distance_squared(point) > HULL_EPSILON * HULL_EPSILON)
        {
            corners.push(point);
        }
    }
    let mut found = Vec::<Vec<usize>>::new();
    let mut faces = Vec::new();
    for first in 0..corners.len() {
        for second in first + 1..corners.len() {
            for third in second + 1..corners.len() {
                let normal = (corners[second] - corners[first])
                    .cross(corners[third] - corners[first])
                    .normalize_or_zero();
                if normal == DVec3::ZERO {
                    continue;
                }
                let heights = corners
                    .iter()
                    .map(|corner| normal.dot(*corner - corners[first]))
                    .collect::<Vec<_>>();
                let normal = if heights.iter().all(|height| *height <= HULL_EPSILON) {
                    normal
                } else if heights.iter().all(|height| *height >= -HULL_EPSILON) {
                    -normal
                } else {
                    continue;
                };
                let on_face = (0..corners.len())
                    .filter(|&index| heights[index].abs() <= HULL_EPSILON * 2.0)
                    .collect::<Vec<_>>();
                if found.contains(&on_face) {
                    continue;
                }
                let centre = on_face.iter().map(|&index| corners[index]).sum::<DVec3>()
                    / on_face.len() as f64;
                let tangent = (corners[first] - centre).normalize();
                let bitangent = normal.cross(tangent);
                let mut ordered = on_face.clone();
                ordered.sort_by(|&left, &right| {
                    let angle = |index: usize| {
                        let offset = corners[index] - centre;
                        offset.dot(bitangent).atan2(offset.dot(tangent))
                    };
                    angle(left).total_cmp(&angle(right))
                });
                found.push(on_face);
                faces.push(PolyFace {
                    vertices: ordered.into_iter().map(|index| corners[index]).collect(),
                    patch: NO_PATCH,
                    family: NO_PATCH,
                    smoothing_group: 0,
                    smooth_with: Vec::new(),
                    uv_provenance: NO_PATCH,
                });
            }
        }
    }
    (faces.len() >= 4).then_some(PolyCell { faces, band: 0 })
}
