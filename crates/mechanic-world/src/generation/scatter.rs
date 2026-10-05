//! Jittered-grid instancing of shapes: arches, spires, caps, and boulders.
//!
//! Space is cut into square cells in x and z. Each cell may own one instance
//! whose origin, rotation, and random vars derive from a hash of the cell, so
//! any point can find the instances that can reach it without global state.

use std::cell::RefCell;

use super::grid::{JitterGrid, cache_slot};
use super::interval::Interval;
use super::tape::{Lane, ManyScratch, Tape};

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

/// Reusable buffers for [`ScatterGen::sample_among_many`].
#[derive(Debug, Default)]
pub(crate) struct ScatterScratch {
    selected: Vec<usize>,
    local: [Vec<f64>; 3],
    rocks: Vec<f64>,
    values: Vec<f64>,
    tape: ManyScratch,
}

#[derive(Debug)]
pub(crate) struct ScatterGen {
    pub(crate) id: u64,
    pub(crate) grid: JitterGrid,
    pub(crate) reach: f64,
    pub(crate) lift: (f64, f64),
    pub(crate) yaw: bool,
    pub(crate) tilt_radians: f64,
    pub(crate) vars: Vec<(f64, f64)>,
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
    fn instance(&self, cell_x: i64, cell_z: i64) -> Option<Instance> {
        let key = (self.id, cell_x, cell_z);
        let slot = cache_slot(self.id, cell_x, cell_z, INSTANCE_CACHE_SLOTS);
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

    fn create_instance(&self, cell_x: i64, cell_z: i64) -> Option<Instance> {
        let (x, z, mut random) = self.grid.place(cell_x, cell_z)?;
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
        self.grid.cells_near(x, z, self.reach)
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
    #[cfg(test)]
    pub(crate) fn sample_among(&self, instances: &[Instance], point: [f64; 3], rock: f64) -> f64 {
        instances.iter().fold(SCATTER_FLOOR, |best, instance| {
            self.sample_instance(instance, point, rock, best)
        })
    }

    /// The scatter's values at `count` points among a box's nearby
    /// instances, appended to `out` and equal bit for bit to
    /// [`Self::sample`] at each point inside that box. Instances are taken
    /// one at a time, in the same order, so every point folds the same
    /// values; each instance's shape runs once over all the points within
    /// its reach, its own vars shared by them.
    pub(crate) fn sample_among_many(
        &self,
        instances: &[Instance],
        point: [Lane<'_>; 3],
        rock: Lane<'_>,
        count: usize,
        scratch: &mut ScatterScratch,
        out: &mut Vec<f64>,
    ) {
        let start = out.len();
        out.resize(start + count, SCATTER_FLOOR);
        let best = &mut out[start..];
        // Squared offsets past this are certainly beyond reach, however the
        // exact distance below rounds; nearer points take the exact test.
        let beyond = (self.reach * (1.0 + 1.0e-9)).powi(2);
        let ScatterScratch {
            selected,
            local,
            rocks,
            values,
            tape,
        } = scratch;
        for instance in instances {
            selected.clear();
            local.iter_mut().for_each(Vec::clear);
            rocks.clear();
            for (index, best) in best.iter().enumerate() {
                let offset = [
                    point[0].at(index) - instance.origin[0],
                    point[1].at(index) - instance.origin[1],
                    point[2].at(index) - instance.origin[2],
                ];
                if offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2] > beyond {
                    continue;
                }
                let distance = offset[0].hypot(offset[1]).hypot(offset[2]);
                if distance > self.reach || self.reach - distance <= *best {
                    continue;
                }
                for (axis, row) in instance.rotation.iter().enumerate() {
                    local[axis].push(
                        row[0].mul_add(offset[0], row[1].mul_add(offset[1], row[2] * offset[2])),
                    );
                }
                if let Lane::Row(_) = rock {
                    rocks.push(rock.at(index));
                }
                selected.push(index);
            }
            if selected.is_empty() {
                continue;
            }
            let mut inputs = [Lane::Splat(0.0); MAX_VARS + 1];
            for (input, &var) in inputs.iter_mut().zip(&instance.vars) {
                *input = Lane::Splat(var);
            }
            inputs[self.vars.len()] = match rock {
                Lane::Row(_) => Lane::Row(rocks),
                shared @ Lane::Splat(_) => shared,
            };
            values.clear();
            self.shape.eval_many(
                [
                    Lane::Row(&local[0]),
                    Lane::Row(&local[1]),
                    Lane::Row(&local[2]),
                ],
                &inputs,
                selected.len(),
                tape,
                values,
            );
            for (&index, &value) in selected.iter().zip(values.iter()) {
                best[index] = best[index].max(value);
            }
        }
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

pub(crate) const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::compile::{Scope, compile, compile_varying};
    use super::super::interval::Interval;
    use super::super::load::parse;
    use super::super::spec::Expr;
    use super::super::tape::{Lane, Op};
    use super::{ScatterScratch, rotation};

    /// Sum of a jittered, masked, lifted, tilted scatter's densities over a
    /// fixed set of points, bit for bit.
    fn scatter_fingerprint() -> u64 {
        let expr: Expr = parse(
            "scatter",
            r#"Scatter(
                cell: 12, reach: 4, chance: 0.6, jitter: 0.9,
                mask: Noise(freq: 0.01, seed: 3, dims: Two),
                ground: Mul([Noise(freq: 0.02, dims: Two), C(5)]),
                lift: (-0.5, 0.5), tilt: 20,
                vars: {"r": (1.0, 3.0)},
                shape: Sphere(Ref("r")),
            )"#,
        )
        .expect("valid scatter");
        let empty = BTreeMap::new();
        let scope = Scope {
            local: &empty,
            library: &empty,
            fields: None,
        };
        let tape = compile(&expr, scope, &[], 99, "test").expect("compiles");
        let mut sum = 0.0;
        for index in 0..4_000_u32 {
            let t = f64::from(index);
            let point = [
                (t * 7.31).rem_euclid(160.0) - 80.0,
                (t * 0.37).rem_euclid(12.0) - 4.0,
                (t * 3.17).rem_euclid(160.0) - 80.0,
            ];
            sum += tape.eval(point, &[]).max(-10.0);
        }
        sum.to_bits()
    }

    #[test]
    fn scatter_output_is_unchanged() {
        // Ground and masks use the enclosing terrain seed; only shape
        // variation uses the scatter instance seed.
        assert_eq!(scatter_fingerprint(), 13_898_904_302_974_188_553);
    }

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

    #[test]
    fn many_points_sample_exactly_as_one_at_a_time() {
        let expr: Expr = parse(
            "scatter",
            r#"Scatter(
                cell: 5, reach: 4.5, chance: 0.8, jitter: 0.9,
                lift: (-0.5, 0.5), tilt: 25,
                vars: {"r": (1.0, 2.5), "w": (0.5, 1.5)},
                shape: Add([
                    Sphere(Mul([Ref("r"), Ref("w")])),
                    Mul([Noise(freq: 0.4, octaves: 2), C(0.6)]),
                    Mul([Ref("rock"), C(0.05)]),
                ]),
            )"#,
        )
        .expect("valid scatter");
        let empty = BTreeMap::new();
        let scope = Scope {
            local: &empty,
            library: &empty,
            fields: None,
        };
        let tape =
            compile_varying(&expr, scope, &["rock".to_owned()], 11, "test").expect("compiles");
        let scatter = tape
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Scatter(_, scatter) => Some(scatter),
                _ => None,
            })
            .expect("a scatter op");
        let dims = [9, 7, 8];
        let mut points: [Vec<f64>; 3] = Default::default();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    for (axis, index) in [i, j, k].into_iter().enumerate() {
                        let step = f64::from(u32::try_from(index).unwrap());
                        points[axis].push(
                            step.mul_add(1.37, -5.0) + 0.1 * f64::from(u8::try_from(axis).unwrap()),
                        );
                    }
                }
            }
        }
        let count = points[0].len();
        let rocks: Vec<f64> = (0..count)
            .map(|index| (f64::from(u32::try_from(index).unwrap()) * 0.37).sin() * 4.0)
            .collect();
        let span = |axis: usize| {
            let values = &points[axis];
            Interval::new(
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let mut nearby = Vec::new();
        scatter.instances_near([span(0), span(1), span(2)], &mut nearby);
        assert!(nearby.len() > 3, "the box should reach several instances");
        let mut scratch = ScatterScratch::default();
        for rock in [Lane::Row(&rocks), Lane::Splat(1.5)] {
            let mut many = vec![7.0];
            scatter.sample_among_many(
                &nearby,
                [
                    Lane::Row(&points[0]),
                    Lane::Row(&points[1]),
                    Lane::Row(&points[2]),
                ],
                rock,
                count,
                &mut scratch,
                &mut many,
            );
            assert_eq!(many.len(), count + 1, "appends after existing values");
            let mut reached = 0;
            for index in 0..count {
                let point = [points[0][index], points[1][index], points[2][index]];
                let one = scatter.sample_among(&nearby, point, rock.at(index));
                assert_eq!(many[index + 1].to_bits(), one.to_bits(), "point {index}");
                assert_eq!(
                    one.to_bits(),
                    scatter.sample(point, rock.at(index)).to_bits()
                );
                reached += usize::from(one > super::SCATTER_FLOOR);
            }
            assert!(
                reached > count / 4,
                "most points lie within some instance's reach"
            );
        }
    }
}
