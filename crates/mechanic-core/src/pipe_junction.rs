//! Pipe junction geometry.
//!
//! A junction joins pipe arms at the centre of one channel cell. Each open
//! face carries an arm: a pipe of the junction's cross-section running from
//! the centre out to that face, so the fitting reads as crossing pipes whose
//! ends sit exactly on the cell boundary. A ball as wide as the pipe fills the
//! centre, rounding elbows and capping a lone arm without poking out of a
//! pipe running through.
//!
//! The ball and every arm contain the centre, so the fitting is star-shaped
//! about it and is sampled along rays. Each cube face's pyramid is cut into polar sectors whose
//! rings include the bore and pipe radii, so open ends are exact annuli.

use core::f64::consts::TAU;

use bevy_math::{DVec3, Mat3, Quat, Vec3};

use crate::{FaceKind, PipeArms, PipeJunctionSpec};

/// Polar sectors per cube face; 15° steps meet every square corner.
const SECTORS: u32 = 24;
/// Evenly spaced rings from each face centre to the square's edge.
const RINGS: u32 = 4;
/// Annular wall boxes per arm reaching past the hub.
const ARM_SEGMENTS: u16 = 16;
/// Arms shorter than this past the hub add no wall boxes.
const MIN_ARM_LENGTH: f32 = 1.0e-5;

/// Surface of an arm that bounds a junction along a ray from its centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipeJunctionSurface {
    /// Round wall of the arm.
    Wall,
    /// Flat end of the arm on the cell face.
    End,
    /// Ball filling the centre.
    Hub,
}

impl PipeJunctionSurface {
    pub(crate) const fn index(self) -> u32 {
        match self {
            Self::Wall => 0,
            Self::End => 1,
            Self::Hub => 2,
        }
    }
}

/// One triangle of a junction's boundary, sampled along three rays from its
/// centre, in part-local space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeJunctionTriangle {
    /// Outer boundary points, wound so the triangle faces away from the centre.
    pub outer: [Vec3; 3],
    /// Bore boundary points on the same rays, or the centre when solid.
    pub inner: [Vec3; 3],
    /// Arm and surface forming the outer points.
    pub outer_surface: (FaceKind, PipeJunctionSurface),
    /// Arm and surface forming the bore points, when hollow.
    pub inner_surface: Option<(FaceKind, PipeJunctionSurface)>,
}

impl PipeJunctionTriangle {
    /// Outward normal of the outer surface at `point`.
    pub fn outer_normal(&self, point: Vec3) -> Vec3 {
        surface_normal(self.outer_surface, point)
    }

    /// Normal of the bore surface at `point`, pointing into the bore.
    pub fn inner_normal(&self, point: Vec3) -> Option<Vec3> {
        self.inner_surface
            .map(|surface| -surface_normal(surface, point))
    }
}

/// Oriented box covering part of a junction, in part-local space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipeJunctionBox {
    /// Box centre.
    pub center: Vec3,
    /// Box orientation.
    pub rotation: Quat,
    /// Half extents along the box's own axes.
    pub half_extents: Vec3,
}

/// Junction boundary triangles in part-local space.
pub fn pipe_junction_triangles(spec: PipeJunctionSpec) -> Vec<PipeJunctionTriangle> {
    ray_triangles(spec)
        .into_iter()
        .map(|triangle| PipeJunctionTriangle {
            outer: triangle.outer.map(DVec3::as_vec3),
            inner: triangle.inner.map(DVec3::as_vec3),
            outer_surface: triangle.outer_surface,
            inner_surface: triangle.inner_surface,
        })
        .collect()
}

/// Boxes covering a junction's material for contact and placement.
///
/// A hub cube as wide as the pipe sits at the centre: solid, or six slabs
/// around a bore-wide chamber with a square passage on each open face. Arms
/// reaching past the hub add sixteen annular wall boxes each, like a pipe.
pub fn pipe_junction_wall_boxes(spec: PipeJunctionSpec) -> Vec<PipeJunctionBox> {
    let reach = spec.dimensions.half_side();
    let radius = spec.dimensions.outer_diameter() * 0.5;
    let bore = spec.dimensions.inner_diameter() * 0.5;
    let mut boxes = Vec::with_capacity(pipe_junction_box_count(spec));
    let axis_aligned = |center, half_extents| PipeJunctionBox {
        center,
        rotation: Quat::IDENTITY,
        half_extents,
    };
    if bore <= 0.0 {
        boxes.push(axis_aligned(Vec3::ZERO, Vec3::splat(radius)));
    } else {
        let wall = (radius - bore) * 0.5;
        let middle = (radius + bore) * 0.5;
        for face in PipeArms::ALL.faces() {
            let (u_axis, v_axis) = face.tangent_axes();
            let normal = arm_axis(face);
            let (u, v) = (u_axis.unit(), v_axis.unit());
            let along = face.axis().unit() * wall;
            if spec.arms.contains(face) {
                for side in [-1.0, 1.0] {
                    boxes.push(axis_aligned(
                        normal * middle + u * (side * middle),
                        along + u * wall + v * radius,
                    ));
                    boxes.push(axis_aligned(
                        normal * middle + v * (side * middle),
                        along + u * bore + v * wall,
                    ));
                }
            } else {
                boxes.push(axis_aligned(
                    normal * middle,
                    along + u * radius + v * radius,
                ));
            }
        }
    }
    let length = reach - radius;
    if length > MIN_ARM_LENGTH {
        let segment = core::f32::consts::TAU / f32::from(ARM_SEGMENTS);
        for face in spec.arms.faces() {
            let axis = arm_axis(face);
            let (u_axis, v_axis) = face.tangent_axes();
            for index in 0..ARM_SEGMENTS {
                let angle = segment * f32::from(index);
                let radial = u_axis.unit() * angle.cos() + v_axis.unit() * angle.sin();
                boxes.push(PipeJunctionBox {
                    center: axis * (radius + length * 0.5) + radial * ((radius + bore) * 0.5),
                    rotation: Quat::from_mat3(&Mat3::from_cols(radial, axis, radial.cross(axis))),
                    half_extents: Vec3::new(
                        (radius - bore) * 0.5,
                        length * 0.5,
                        radius * (segment * 0.5).tan(),
                    ),
                });
            }
        }
    }
    boxes
}

/// Number of boxes [`pipe_junction_wall_boxes`] produces.
pub(crate) fn pipe_junction_box_count(spec: PipeJunctionSpec) -> usize {
    let arms = spec.arms.count() as usize;
    let hub = if spec.dimensions.inner_diameter() <= 0.0 {
        1
    } else {
        6 + 3 * arms
    };
    let reach = spec.dimensions.half_side() - spec.dimensions.outer_diameter() * 0.5;
    hub + if reach > MIN_ARM_LENGTH {
        usize::from(ARM_SEGMENTS) * arms
    } else {
        0
    }
}

/// A boundary triangle along three rays, in part-local space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RayTriangle {
    pub(crate) outer: [DVec3; 3],
    pub(crate) inner: [DVec3; 3],
    pub(crate) outer_surface: (FaceKind, PipeJunctionSurface),
    pub(crate) inner_surface: Option<(FaceKind, PipeJunctionSurface)>,
}

/// Samples a junction's outside and bore along rays through every cube face.
pub(crate) fn ray_triangles(spec: PipeJunctionSpec) -> Vec<RayTriangle> {
    let reach = f64::from(spec.dimensions.half_side());
    let shape = Shape {
        arms: spec
            .arms
            .faces()
            .map(|face| (face, DVec3::from(arm_axis(face))))
            .collect(),
        reach,
        radius: f64::from(spec.dimensions.outer_diameter()) * 0.5,
        bore: f64::from(spec.dimensions.inner_diameter()) * 0.5,
    };
    let mut triangles = Vec::new();
    for face in PipeArms::ALL.faces() {
        let normal = DVec3::from(arm_axis(face));
        let (u_axis, v_axis) = face.tangent_axes();
        let (u, v) = (DVec3::from(u_axis.unit()), DVec3::from(v_axis.unit()));
        // Face-plane points along each sector ray, from the face centre to
        // the square's edge. Every ray has the same number of rings.
        let rays = (0..SECTORS)
            .map(|sector| {
                let angle = TAU * f64::from(sector) / f64::from(SECTORS);
                let (cos, sin) = (angle.cos(), angle.sin());
                let edge = reach / cos.abs().max(sin.abs());
                let mut rings = (0..=RINGS)
                    .map(|ring| edge * f64::from(ring) / f64::from(RINGS))
                    .collect::<Vec<_>>();
                rings.push(shape.radius.min(edge));
                rings.push(shape.bore.min(edge));
                rings.sort_by(f64::total_cmp);
                rings
                    .into_iter()
                    .map(|ring| normal * reach + (u * cos + v * sin) * ring)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for sector in 0..rays.len() {
            let (a, b) = (&rays[sector], &rays[(sector + 1) % rays.len()]);
            for ring in 0..a.len() - 1 {
                for corners in [
                    [a[ring], a[ring + 1], b[ring + 1]],
                    [a[ring], b[ring + 1], b[ring]],
                ] {
                    let area = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
                    if area.length_squared() <= 1.0e-18 {
                        continue;
                    }
                    if let Some(triangle) = shape.triangle(corners) {
                        triangles.push(triangle);
                    }
                }
            }
        }
    }
    triangles
}

struct Shape {
    arms: Vec<(FaceKind, DVec3)>,
    reach: f64,
    radius: f64,
    bore: f64,
}

impl Shape {
    fn hollow(&self) -> bool {
        self.bore > 1.0e-9
    }

    fn triangle(&self, corners: [DVec3; 3]) -> Option<RayTriangle> {
        let directions = corners.map(DVec3::normalize);
        let mut outer = directions.map(|direction| direction * self.exit(direction, self.radius).0);
        let mut inner = directions.map(|direction| {
            if self.hollow() {
                direction * self.exit(direction, self.bore).0
            } else {
                DVec3::ZERO
            }
        });
        if self.hollow()
            && outer
                .iter()
                .zip(&inner)
                .all(|(outer, inner)| outer.distance_squared(*inner) <= 1.0e-18)
        {
            // Rays leaving through the open bore of an arm end.
            return None;
        }
        let winding = (outer[1] - outer[0]).cross(outer[2] - outer[0]);
        if winding.dot(outer[0] + outer[1] + outer[2]) < 0.0 {
            outer.swap(1, 2);
            inner.swap(1, 2);
        }
        let middle = (corners[0] + corners[1] + corners[2]).normalize();
        Some(RayTriangle {
            outer,
            inner,
            outer_surface: self.exit(middle, self.radius).1,
            inner_surface: self.hollow().then(|| self.exit(middle, self.bore).1),
        })
    }

    /// Where the union of the ball and arms of `radius` ends along `direction`.
    fn exit(&self, direction: DVec3, radius: f64) -> (f64, (FaceKind, PipeJunctionSurface)) {
        self.arms.iter().fold(
            (radius, (self.arms[0].0, PipeJunctionSurface::Hub)),
            |best, &(face, axis)| {
                let (distance, surface) = arm_exit(direction, axis, radius, self.reach);
                if distance > best.0 + 1.0e-12 {
                    (distance, (face, surface))
                } else {
                    best
                }
            },
        )
    }
}

/// Distance along `direction` to where one arm ends, and the surface there.
/// The arm runs from the centre to `reach` in front of it.
fn arm_exit(direction: DVec3, axis: DVec3, radius: f64, reach: f64) -> (f64, PipeJunctionSurface) {
    let along = direction.dot(axis);
    if along < -1.0e-12 {
        return (0.0, PipeJunctionSurface::Wall);
    }
    let lateral = (1.0 - along * along).max(0.0).sqrt();
    let mut exit = (
        if lateral > 1.0e-12 {
            radius / lateral
        } else {
            f64::INFINITY
        },
        PipeJunctionSurface::Wall,
    );
    if along > 1.0e-12 && reach / along < exit.0 {
        exit = (reach / along, PipeJunctionSurface::End);
    }
    exit
}

fn arm_axis(face: FaceKind) -> Vec3 {
    face.axis().unit() * face.sign()
}

fn surface_normal((face, surface): (FaceKind, PipeJunctionSurface), point: Vec3) -> Vec3 {
    let axis = arm_axis(face);
    match surface {
        PipeJunctionSurface::Wall => (point - axis * point.dot(axis)).normalize_or_zero(),
        PipeJunctionSurface::End => axis,
        PipeJunctionSurface::Hub => point.normalize_or_zero(),
    }
}
