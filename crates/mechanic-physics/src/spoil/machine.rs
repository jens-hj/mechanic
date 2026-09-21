//! The machine as loose material meets it: moving shapes that push and carry.

use bevy_math::{DQuat, DVec3};
use mechanic_core::{ColliderShape, CompiledCreation};

use crate::{BodyPose, SpatialMotion};

enum Shape {
    Cuboid(DVec3),
    /// Outward face planes: normal and offset, in the collider's frame.
    Convex(Vec<(DVec3, f64)>),
}

struct Collider {
    body: usize,
    centre: DVec3,
    rotation: DQuat,
    reach: f64,
    shape: Shape,
    friction: f64,
}

/// Where a sphere touches a collider.
pub(super) struct Touch {
    pub(super) body: usize,
    /// Out of the collider, towards the sphere.
    pub(super) normal: DVec3,
    pub(super) depth: f64,
    /// Velocity of the collider's surface at the touch.
    pub(super) velocity: DVec3,
    pub(super) friction: f64,
}

/// The colliders of one tick at global positions, with how their bodies move.
#[derive(Default)]
pub struct SpoilMachine {
    colliders: Vec<Collider>,
    bodies: Vec<(DVec3, SpatialMotion)>,
    low: DVec3,
    high: DVec3,
}

impl SpoilMachine {
    /// Places the creation's colliders. `origin` is the global position the
    /// poses are relative to. Rows beyond the shorter of `poses` and `motions`
    /// are left out.
    pub fn new(
        creation: &CompiledCreation,
        poses: &[BodyPose],
        motions: &[SpatialMotion],
        origin: DVec3,
    ) -> Self {
        let bodies = poses
            .iter()
            .zip(motions)
            .map(|(pose, motion)| (origin + pose.position, *motion))
            .collect::<Vec<_>>();
        let mut low = DVec3::INFINITY;
        let mut high = DVec3::NEG_INFINITY;
        let colliders = creation
            .colliders
            .iter()
            .filter_map(|collider| {
                let body = collider.compound_index as usize;
                let pose = poses.get(body).filter(|_| body < bodies.len())?;
                let centre =
                    origin + pose.position + pose.rotation * collider.local_center.as_dvec3();
                let (rotation, shape, reach) = match &collider.shape {
                    ColliderShape::Cuboid {
                        local_rotation,
                        half_extents,
                    } => (
                        pose.rotation * local_rotation.as_dquat(),
                        Shape::Cuboid(half_extents.as_dvec3()),
                        f64::from(half_extents.length()),
                    ),
                    ColliderShape::Convex(convex) => {
                        // Planes are given about the compound's centre of mass.
                        let shift = collider.local_center.as_dvec3();
                        (
                            pose.rotation,
                            Shape::Convex(
                                convex
                                    .face_planes
                                    .iter()
                                    .map(|plane| {
                                        let normal = plane.truncate().as_dvec3();
                                        (normal, f64::from(plane.w) - normal.dot(shift))
                                    })
                                    .collect(),
                            ),
                            convex
                                .vertices
                                .iter()
                                .map(|vertex| (vertex.as_dvec3() - shift).length())
                                .fold(0.0, f64::max),
                        )
                    }
                };
                low = low.min(centre - DVec3::splat(reach));
                high = high.max(centre + DVec3::splat(reach));
                Some(Collider {
                    body,
                    centre,
                    rotation,
                    reach,
                    shape,
                    friction: f64::from(collider.material_properties.dynamic_friction),
                })
            })
            .collect();
        Self {
            colliders,
            bodies,
            low,
            high,
        }
    }

    /// Global centre of a body.
    pub(super) fn body_centre(&self, body: usize) -> DVec3 {
        self.bodies[body].0
    }

    /// Whether anything of the machine moves within `margin` of a point.
    pub(super) fn stirs(&self, point: DVec3, margin: f64) -> bool {
        self.near(point, margin)
            && self.colliders.iter().any(|collider| {
                let motion = self.bodies[collider.body].1;
                motion.linear.length() + motion.angular.length() * collider.reach > 0.05
                    && collider.centre.distance_squared(point)
                        < (collider.reach + margin) * (collider.reach + margin)
            })
    }

    fn near(&self, point: DVec3, margin: f64) -> bool {
        point.cmpge(self.low - margin).all() && point.cmple(self.high + margin).all()
    }

    /// Whether the machine keeps spoil from being laid in the terrain cell
    /// centred here: it fills the cell, sits close above it, or works beside it.
    /// Ground laid under a press or in a tool's way is only pressed or dug out
    /// again; the spoil stays loose until the machine has moved on.
    pub fn keeps_clear(&self, cell_centre: DVec3) -> bool {
        /// Room a clod needs above the ground it is laid on, in metres.
        const HEADROOM_M: f64 = 0.2;
        /// How near a moving part ground is left alone, in metres.
        const WORKING_MARGIN_M: f64 = 0.15;
        self.overlaps(cell_centre, mechanic_world::TERRAIN_CELL_METERS * 0.5)
            || self.overlaps(cell_centre + DVec3::Y * HEADROOM_M * 0.5, HEADROOM_M * 0.5)
            || self.stirs(cell_centre, WORKING_MARGIN_M)
    }

    /// Whether a sphere overlaps any collider.
    pub fn overlaps(&self, centre: DVec3, radius: f64) -> bool {
        let mut found = Vec::new();
        self.touches(centre, radius, &mut found);
        !found.is_empty()
    }

    /// Every collider a sphere overlaps.
    pub(super) fn touches(&self, centre: DVec3, radius: f64, found: &mut Vec<Touch>) {
        found.clear();
        if !self.near(centre, radius) {
            return;
        }
        for collider in &self.colliders {
            let offset = centre - collider.centre;
            let reach = collider.reach + radius;
            if offset.length_squared() > reach * reach {
                continue;
            }
            let local = collider.rotation.inverse() * offset;
            let Some((normal, distance)) = (match &collider.shape {
                Shape::Cuboid(half) => Some(cuboid_distance(local, *half)),
                Shape::Convex(planes) => convex_distance(local, planes),
            }) else {
                continue;
            };
            if distance >= radius {
                continue;
            }
            let normal = collider.rotation * normal;
            let (body_centre, motion) = self.bodies[collider.body];
            let point = centre - normal * distance;
            found.push(Touch {
                body: collider.body,
                normal,
                depth: radius - distance,
                velocity: motion.linear + motion.angular.cross(point - body_centre),
                friction: collider.friction,
            });
        }
    }
}

// Direction out of a box and the signed distance to its surface.
fn cuboid_distance(local: DVec3, half: DVec3) -> (DVec3, f64) {
    let outside = (local.abs() - half).max(DVec3::ZERO);
    if outside != DVec3::ZERO {
        let distance = outside.length();
        return (outside * local.signum() / distance, distance);
    }
    let gap = half - local.abs();
    let axis = if gap.x <= gap.y && gap.x <= gap.z {
        DVec3::X
    } else if gap.y <= gap.z {
        DVec3::Y
    } else {
        DVec3::Z
    };
    (axis * local.signum(), -gap.dot(axis))
}

// Direction out of a convex shape and the signed distance to its surface.
// Inside, that is the nearest face. Outside, a point beside an edge or a corner
// is outside several faces at once: stepping back behind each in turn finds the
// offset to the edge, where the farthest face alone would read near and push
// along that face's normal, so a flight's rim passing a clod would lift it.
fn convex_distance(local: DVec3, planes: &[(DVec3, f64)]) -> Option<(DVec3, f64)> {
    let (normal, distance) = planes
        .iter()
        .map(|&(normal, offset)| (normal, normal.dot(local) - offset))
        .max_by(|first, second| first.1.total_cmp(&second.1))?;
    if distance <= 0.0 {
        return Some((normal, distance));
    }
    let mut out = normal * distance;
    for _ in 0..2 {
        for &(normal, offset) in planes {
            let short = normal.dot(local) - offset - normal.dot(out);
            if short > 0.0 {
                out += normal * short;
            }
        }
    }
    let length = out.length();
    Some((out / length, length))
}
