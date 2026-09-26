//! Jittered-grid instancing of shapes: arches, spires, caps, and boulders.
//!
//! Space is cut into square cells in x and z. Each cell may own one instance
//! whose origin, rotation, and random vars derive from a hash of the cell, so
//! any point can find the instances that can reach it without global state.

use std::cell::RefCell;

use super::interval::Interval;
use super::tape::Tape;

/// Density far from every instance. Low enough that unions and fillets with
/// the ground never see it.
pub(crate) const SCATTER_FLOOR: f64 = -1.0e3;

const INSTANCE_CACHE_SLOTS: usize = 4_096;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instance {
    origin: [f64; 3],
    /// Rows of the world-to-local rotation.
    rotation: [[f64; 3]; 3],
    vars: [f64; MAX_VARS],
}

pub(crate) const MAX_VARS: usize = 8;

#[derive(Debug)]
pub(crate) struct ScatterGen {
    pub(crate) id: u64,
    pub(crate) cell: f64,
    pub(crate) reach: f64,
    pub(crate) jitter: f64,
    pub(crate) chance: f64,
    pub(crate) lift: (f64, f64),
    pub(crate) yaw: bool,
    pub(crate) tilt_radians: f64,
    pub(crate) vars: Vec<(f64, f64)>,
    pub(crate) seed: u64,
    pub(crate) mask: Option<Tape>,
    pub(crate) ground: Option<Tape>,
    pub(crate) shape: Tape,
}

type InstanceKey = (u64, i64, i64);

type InstanceSlot = Option<(InstanceKey, Option<Instance>)>;

thread_local! {
    /// Direct-mapped cache of recently used instances; a collision simply
    /// recomputes the instance, which is deterministic.
    static INSTANCES: RefCell<Vec<InstanceSlot>> =
        RefCell::new(vec![None; INSTANCE_CACHE_SLOTS]);
}

impl ScatterGen {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "cell indices are hashed bit for bit"
    )]
    fn instance(&self, cell_x: i64, cell_z: i64) -> Option<Instance> {
        let key = (self.id, cell_x, cell_z);
        let slot = (mix(self.id ^ (cell_x as u64).wrapping_mul(0x9e37_79b9) ^ (cell_z as u64) << 32)
            as usize)
            % INSTANCE_CACHE_SLOTS;
        INSTANCES.with(|cache| {
            if let Some((cached, instance)) = cache.borrow()[slot]
                && cached == key
            {
                return instance;
            }
            let created = self.create_instance(cell_x, cell_z);
            cache.borrow_mut()[slot] = Some((key, created));
            created
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "cell indices stay far below 2^52 inside the finite world"
    )]
    fn create_instance(&self, cell_x: i64, cell_z: i64) -> Option<Instance> {
        let mut random = Hash::new(self.seed, cell_x, cell_z);
        if random.unit() >= self.chance {
            return None;
        }
        let spread = self.jitter.clamp(0.0, 1.0);
        let x = (cell_x as f64 + 0.5 + (random.unit() - 0.5) * spread) * self.cell;
        let z = (cell_z as f64 + 0.5 + (random.unit() - 0.5) * spread) * self.cell;
        if let Some(mask) = &self.mask
            && mask.eval([x, 0.0, z], &[]) <= 0.0
        {
            return None;
        }
        let ground = self
            .ground
            .as_ref()
            .map_or(0.0, |ground| ground.eval([x, 0.0, z], &[]));
        let y = ground + random.between(self.lift.0, self.lift.1);
        let yaw = if self.yaw {
            random.unit() * core::f64::consts::TAU
        } else {
            0.0
        };
        let tilt = random.between(-self.tilt_radians, self.tilt_radians);
        let tilt_axis = random.unit() * core::f64::consts::TAU;
        let mut vars = [0.0; MAX_VARS];
        for (slot, &(lo, hi)) in vars.iter_mut().zip(&self.vars) {
            *slot = random.between(lo, hi);
        }
        Some(Instance {
            origin: [x, y, z],
            rotation: rotation(yaw, tilt, tilt_axis),
            vars,
        })
    }

    fn cells_near(&self, x: Interval, z: Interval) -> (i64, i64, i64, i64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates are finite and inside the world"
        )]
        let cell = |value: f64| (value / self.cell).floor() as i64;
        (
            cell(x.lo - self.reach),
            cell(x.hi + self.reach),
            cell(z.lo - self.reach),
            cell(z.hi + self.reach),
        )
    }

    pub(crate) fn sample(&self, point: [f64; 3], rock: f64) -> f64 {
        let (x0, x1, z0, z1) =
            self.cells_near(Interval::point(point[0]), Interval::point(point[2]));
        let mut best = SCATTER_FLOOR;
        for cell_z in z0..=z1 {
            for cell_x in x0..=x1 {
                if let Some(instance) = self.instance(cell_x, cell_z) {
                    best = self.sample_instance(&instance, point, rock, best);
                }
            }
        }
        best
    }

    /// Instances whose reach touches a box, so a block of points can be
    /// sampled against only them.
    pub(crate) fn instances_near(&self, domain: [Interval; 3], found: &mut Vec<Instance>) {
        found.clear();
        if domain
            .iter()
            .any(|axis| !axis.lo.is_finite() || !axis.hi.is_finite())
        {
            return;
        }
        let (x0, x1, z0, z1) = self.cells_near(domain[0], domain[2]);
        for cell_z in z0..=z1 {
            for cell_x in x0..=x1 {
                let Some(instance) = self.instance(cell_x, cell_z) else {
                    continue;
                };
                if gap_to_box(&instance, domain) <= self.reach {
                    found.push(instance);
                }
            }
        }
    }

    /// The scatter's value at a point among a box's nearby instances; equal
    /// to [`Self::sample`] for every point inside that box.
    pub(crate) fn sample_among(&self, instances: &[Instance], point: [f64; 3], rock: f64) -> f64 {
        instances.iter().fold(SCATTER_FLOOR, |best, instance| {
            self.sample_instance(instance, point, rock, best)
        })
    }

    fn sample_instance(&self, instance: &Instance, point: [f64; 3], rock: f64, best: f64) -> f64 {
        let offset = [
            point[0] - instance.origin[0],
            point[1] - instance.origin[1],
            point[2] - instance.origin[2],
        ];
        let distance = offset[0].hypot(offset[1]).hypot(offset[2]);
        if distance > self.reach || self.reach - distance <= best {
            return best;
        }
        let local = instance
            .rotation
            .map(|row| row[0].mul_add(offset[0], row[1].mul_add(offset[1], row[2] * offset[2])));
        let mut inputs = [0.0; MAX_VARS + 1];
        inputs[..MAX_VARS].copy_from_slice(&instance.vars);
        inputs[self.vars.len()] = rock;
        best.max(self.shape.eval(local, &inputs))
    }

    pub(crate) fn interval(&self, domain: [Interval; 3], rock: Interval) -> Interval {
        if domain
            .iter()
            .any(|axis| !axis.lo.is_finite() || !axis.hi.is_finite())
        {
            return Interval::new(SCATTER_FLOOR, f64::INFINITY);
        }
        let (x0, x1, z0, z1) = self.cells_near(domain[0], domain[2]);
        if (x1 - x0 + 1) * (z1 - z0 + 1) > 4_096 {
            return Interval::new(SCATTER_FLOOR, self.reach);
        }
        let mut bounds = Interval::point(SCATTER_FLOOR);
        for cell_z in z0..=z1 {
            for cell_x in x0..=x1 {
                let Some(instance) = self.instance(cell_x, cell_z) else {
                    continue;
                };
                if gap_to_box(&instance, domain) > self.reach {
                    continue;
                }
                let centre = domain.map(Interval::centre);
                let half = domain.map(Interval::radius);
                let offset = [
                    centre[0] - instance.origin[0],
                    centre[1] - instance.origin[1],
                    centre[2] - instance.origin[2],
                ];
                let local = core::array::from_fn(|axis| {
                    let row = instance.rotation[axis];
                    let middle =
                        row[0].mul_add(offset[0], row[1].mul_add(offset[1], row[2] * offset[2]));
                    let extent = row[0].abs().mul_add(
                        half[0],
                        row[1].abs().mul_add(half[1], row[2].abs() * half[2]),
                    );
                    Interval::new(middle - extent, middle + extent)
                });
                let mut inputs = [Interval::point(0.0); MAX_VARS + 1];
                for (input, var) in inputs.iter_mut().zip(instance.vars) {
                    *input = Interval::point(var);
                }
                inputs[self.vars.len()] = rock;
                bounds = bounds.hull(self.shape.interval(local, &inputs));
            }
        }
        bounds
    }
}

/// Distance from an instance's origin to the nearest point of a box, which
/// decides whether the instance's ball touches it at all.
fn gap_to_box(instance: &Instance, domain: [Interval; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let value = instance.origin[axis].clamp(domain[axis].lo, domain[axis].hi)
                - instance.origin[axis];
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

/// World-to-local rotation: undo the tilt about a horizontal axis, then yaw.
fn rotation(yaw: f64, tilt: f64, tilt_axis: f64) -> [[f64; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let yaw_matrix = [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]];
    let (ax, az) = (tilt_axis.cos(), tilt_axis.sin());
    let (s, c) = (-tilt).sin_cos();
    let t = 1.0 - c;
    // Rodrigues rotation about the unit axis (ax, 0, az).
    let tilt_matrix = [
        [t * ax * ax + c, -s * az, t * ax * az],
        [s * az, c, -s * ax],
        [t * ax * az, s * ax, t * az * az + c],
    ];
    let mut product = [[0.0; 3]; 3];
    for (row, output) in product.iter_mut().enumerate() {
        for (column, value) in output.iter_mut().enumerate() {
            *value = (0..3)
                .map(|index| yaw_matrix[row][index] * tilt_matrix[index][column])
                .sum();
        }
    }
    product
}

/// Deterministic per-cell random stream.
struct Hash(u64);

impl Hash {
    #[expect(clippy::cast_sign_loss, reason = "cell indices are hashed bit for bit")]
    fn new(seed: u64, x: i64, z: i64) -> Self {
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        state = mix(state ^ (x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
        state = mix(state ^ (z as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
        Self(state)
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits map exactly onto the unit interval"
    )]
    fn unit(&mut self) -> f64 {
        self.0 = mix(self.0.wrapping_add(0x9e37_79b9_7f4a_7c15));
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        (hi - lo).mul_add(self.unit(), lo)
    }
}

pub(crate) const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::rotation;

    #[test]
    fn instance_rotations_are_orthonormal() {
        for (yaw, tilt, axis) in [(0.3, 0.2, 1.0), (2.0, -0.4, 4.0), (5.0, 0.0, 0.0)] {
            let matrix = rotation(yaw, tilt, axis);
            for (row, first) in matrix.iter().enumerate() {
                for (other, second) in matrix.iter().enumerate() {
                    let dot: f64 = first.iter().zip(second).map(|(a, b)| a * b).sum();
                    let expected = if row == other { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1.0e-9);
                }
            }
        }
    }
}
