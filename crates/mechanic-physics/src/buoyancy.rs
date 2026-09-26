//! Buoyancy and drag on a machine in water.
//!
//! Every collider is cut once into small probes, each a volume at a point in
//! its body. A tick looks the water up once per collider; every substep then
//! pushes each probe up by the weight of the water it displaces and drags it
//! toward the current, as generalized forces beside gravity. A probe is
//! submerged gradually over its own height, so a body crossing the surface
//! feels a smooth force. Drag is capped so it can slow a probe to the
//! current's speed within a substep but never reverse it.

use bevy_math::{DVec3, Vec3, Vec4};
use mechanic_core::{ColliderShape, CompiledConvex, CompiledCreation, WATER_DENSITY_KG_M3};
use mechanic_world::{TerrainField, WaterSurface, WaterSurfaces};

use crate::{BodyPose, MachineKinematics, PhysicsError, SpatialMotion};

/// Longest edge of one probe, in metres: a 25 cm block holds eight.
const PROBE_EDGE_METRES: f64 = 0.125;

/// Most probes along one edge of a collider.
const MAX_PROBES_PER_EDGE: u32 = 6;

/// Probes along each edge of a convex collider's bounds.
const CONVEX_SAMPLES_PER_EDGE: u32 = 6;

/// Samples along each edge of a convex collider's bounds that measure its
/// volume, which its probes then share.
const CONVEX_VOLUME_SAMPLES_PER_EDGE: u32 = 24;

/// Pressure drag coefficient of a probe's face.
const DRAG_COEFFICIENT: f64 = 1.0;

/// Viscous damping, per second, of a submerged body's motion through the
/// water, so a floating body settles instead of bobbing for ever.
const LINEAR_DAMPING_PER_S: f64 = 1.5;

/// Water around a machine.
pub trait WaterSource: Sync {
    /// The water at a global point: the surface of the water it lies in or
    /// under, if any.
    fn surface(&self, point: DVec3) -> Option<WaterSurface>;
}

impl WaterSource for TerrainField {
    fn surface(&self, point: DVec3) -> Option<WaterSurface> {
        self.water_surface(point.x, point.z)
    }
}

impl WaterSource for WaterSurfaces {
    fn surface(&self, point: DVec3) -> Option<WaterSurface> {
        Self::surface(self, point)
    }
}

/// One volume of a body that water can lift.
#[derive(Clone, Copy, Debug)]
struct Probe {
    body: usize,
    /// Collider the probe came from; its water is looked up once per tick.
    collider: usize,
    /// Position in the body's frame.
    local: DVec3,
    volume: f64,
    /// Height over which the probe goes under.
    height: f64,
    /// Its collider's frontal area, shared out by volume.
    area: f64,
    /// The body's mass, shared out by volume.
    mass: f64,
}

/// Where a collider is and how far its probes reach from it.
#[derive(Clone, Copy, Debug)]
struct ColliderReach {
    body: usize,
    local: DVec3,
    reach: f64,
}

/// A creation's colliders cut into probes.
#[derive(Clone, Debug, Default)]
pub(crate) struct BuoyancyProbes {
    probes: Vec<Probe>,
    colliders: Vec<ColliderReach>,
}

/// The water one tick found around each collider.
#[derive(Clone, Debug, Default)]
pub(crate) struct WaterAround {
    surfaces: Vec<Option<WaterSurface>>,
    /// Floating-origin offset of the solver's positions.
    origin: DVec3,
}

impl WaterAround {
    /// Whether any collider has water around it.
    pub(crate) fn any(&self) -> bool {
        self.surfaces.iter().any(Option::is_some)
    }
}

impl BuoyancyProbes {
    /// Cuts every dynamic collider of a creation into probes. A cylinder's
    /// sixteen overlapping tangent boxes are replaced by its exact hull.
    pub(crate) fn new(creation: &CompiledCreation) -> Self {
        let mut probes = Self::default();
        let mut in_cylinder = vec![false; creation.colliders.len()];
        for cylinder in &creation.cylinders {
            let first = cylinder.first_collider as usize;
            let last = (first + mechanic_core::CYLINDER_COLLIDER_COUNT).min(in_cylinder.len());
            if first < last {
                in_cylinder[first..last].fill(true);
            }
            let body = cylinder.compound_index as usize;
            if !creation.compounds[body].is_static {
                probes.add_convex(body, &cylinder.hull());
            }
        }
        for (row, collider) in creation.colliders.iter().enumerate() {
            let body = collider.compound_index as usize;
            if in_cylinder[row] || creation.compounds[body].is_static {
                continue;
            }
            match &collider.shape {
                ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                } => probes.add_cuboid(body, collider.local_center, *local_rotation, *half_extents),
                ColliderShape::Convex(convex) => probes.add_convex(body, convex),
            }
        }
        let mut volumes = vec![0.0; creation.compounds.len()];
        let mut collider_volumes = vec![0.0; probes.colliders.len()];
        for probe in &probes.probes {
            volumes[probe.body] += probe.volume;
            collider_volumes[probe.collider] += probe.volume;
        }
        for probe in &mut probes.probes {
            let mass = f64::from(creation.compounds[probe.body].mass_properties.mass);
            probe.mass = mass * probe.volume / volumes[probe.body].max(f64::EPSILON);
            // A collider meets the water with about the face of a cube of
            // its volume, whichever way it moves.
            let collider = collider_volumes[probe.collider].max(f64::EPSILON);
            probe.area = collider.powf(2.0 / 3.0) * probe.volume / collider;
        }
        probes
    }

    fn begin_collider(&mut self, body: usize, local: DVec3, reach: f64) -> usize {
        self.colliders.push(ColliderReach { body, local, reach });
        self.colliders.len() - 1
    }

    fn add_cuboid(
        &mut self,
        body: usize,
        centre: Vec3,
        rotation: bevy_math::Quat,
        half_extents: Vec3,
    ) {
        let (centre, rotation, half) = (
            centre.as_dvec3(),
            rotation.as_dquat(),
            half_extents.as_dvec3(),
        );
        let collider = self.begin_collider(body, centre, half.length());
        let counts = half.to_array().map(|half| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a few probes per edge"
            )]
            let count = (2.0 * half / PROBE_EDGE_METRES).ceil().max(1.0) as u32;
            count.min(MAX_PROBES_PER_EDGE)
        });
        let step = DVec3::new(
            2.0 * half.x / f64::from(counts[0]),
            2.0 * half.y / f64::from(counts[1]),
            2.0 * half.z / f64::from(counts[2]),
        );
        let volume = step.x * step.y * step.z;
        for k in 0..counts[2] {
            for j in 0..counts[1] {
                for i in 0..counts[0] {
                    let offset = DVec3::new(
                        (f64::from(i) + 0.5).mul_add(step.x, -half.x),
                        (f64::from(j) + 0.5).mul_add(step.y, -half.y),
                        (f64::from(k) + 0.5).mul_add(step.z, -half.z),
                    );
                    self.probes.push(Probe {
                        body,
                        collider,
                        local: centre + rotation * offset,
                        volume,
                        height: volume.cbrt(),
                        area: 0.0,
                        mass: 0.0,
                    });
                }
            }
        }
    }

    fn add_convex(&mut self, body: usize, convex: &CompiledConvex) {
        let Some((lo, hi)) = convex.vertices.iter().map(|vertex| vertex.as_dvec3()).fold(
            None,
            |bounds: Option<(DVec3, DVec3)>, vertex| {
                Some(bounds.map_or((vertex, vertex), |(lo, hi)| {
                    (lo.min(vertex), hi.max(vertex))
                }))
            },
        ) else {
            return;
        };
        let inside = |point: DVec3| {
            convex.face_planes.iter().all(|plane: &Vec4| {
                plane.truncate().as_dvec3().dot(point) <= f64::from(plane.w) + 1.0e-9
            })
        };
        let centres = |samples: u32| {
            let step = (hi - lo) / f64::from(samples);
            (0..samples.pow(3)).map(move |index| {
                let [i, j, k] = [
                    index % samples,
                    index / samples % samples,
                    index / samples / samples,
                ];
                lo + DVec3::new(
                    (f64::from(i) + 0.5) * step.x,
                    (f64::from(j) + 0.5) * step.y,
                    (f64::from(k) + 0.5) * step.z,
                )
            })
        };
        let fine = CONVEX_VOLUME_SAMPLES_PER_EDGE;
        let cell = (hi - lo) / f64::from(fine);
        #[expect(clippy::cast_precision_loss, reason = "a few thousand samples")]
        let volume =
            centres(fine).filter(|&point| inside(point)).count() as f64 * cell.x * cell.y * cell.z;
        let points = centres(CONVEX_SAMPLES_PER_EDGE)
            .filter(|&point| inside(point))
            .collect::<Vec<_>>();
        if points.is_empty() || volume <= 0.0 {
            return;
        }
        let collider = self.begin_collider(body, (lo + hi) * 0.5, (hi - lo).length() * 0.5);
        #[expect(clippy::cast_precision_loss, reason = "a few hundred probes")]
        let share = volume / points.len() as f64;
        let height = ((hi - lo) / f64::from(CONVEX_SAMPLES_PER_EDGE)).min_element();
        for local in points {
            self.probes.push(Probe {
                body,
                collider,
                local,
                volume: share,
                height,
                area: 0.0,
                mass: 0.0,
            });
        }
    }

    /// Looks up the water around every collider at a tick's starting poses.
    pub(crate) fn water_around(
        &self,
        poses: &[BodyPose],
        water: &dyn WaterSource,
        origin: DVec3,
    ) -> WaterAround {
        let surfaces = self
            .colliders
            .iter()
            .map(|collider| {
                let pose = poses.get(collider.body)?;
                let centre = pose.position + pose.rotation * collider.local;
                let surface = water.surface(centre + origin)?;
                (centre.y + origin.y - collider.reach < surface.level).then_some(surface)
            })
            .collect();
        WaterAround { surfaces, origin }
    }

    /// Adds buoyancy and drag over one substep to a generalized force.
    ///
    /// # Errors
    /// Rejects velocities the kinematics cannot project.
    pub(crate) fn add_forces(
        &self,
        water: &WaterAround,
        model: &MachineKinematics<'_>,
        velocities: &[f64],
        gravity: DVec3,
        dt: f64,
        force: &mut [f64],
    ) -> Result<(), PhysicsError> {
        if !water.any() {
            return Ok(());
        }
        let motions = model.body_motions(velocities)?;
        let mut wrenches = vec![SpatialMotion::default(); motions.len()];
        for probe in &self.probes {
            let Some(surface) = water.surfaces[probe.collider] else {
                continue;
            };
            let pose = &model.poses[probe.body];
            let point = pose.position + pose.rotation * probe.local;
            let level = surface.level - water.origin.y;
            let submerged = ((level - point.y) / probe.height + 0.5).clamp(0.0, 1.0);
            if submerged <= 0.0 {
                continue;
            }
            let arm = point - model.centre(probe.body);
            let motion = motions[probe.body];
            let current = DVec3::new(surface.flow.x, 0.0, surface.flow.y);
            let relative = motion.linear + motion.angular.cross(arm) - current;
            let speed = relative.length();
            let displaced = WATER_DENSITY_KG_M3 * probe.volume * submerged;
            let mut push = -gravity * displaced;
            if speed > 0.0 {
                let area = probe.area * submerged;
                let resistance = 0.5 * WATER_DENSITY_KG_M3 * DRAG_COEFFICIENT * area * speed
                    + displaced * LINEAR_DAMPING_PER_S;
                // Never more than stops the probe relative to the water.
                let resistance = resistance.min(probe.mass / dt);
                push -= relative * resistance;
            }
            let wrench = &mut wrenches[probe.body];
            wrench.linear += push;
            wrench.angular += arm.cross(push);
        }
        for (value, add) in force.iter_mut().zip(model.project_wrenches(wrenches)) {
            *value += add;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
