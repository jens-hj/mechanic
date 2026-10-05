//! Flat SSA programs compiled from expressions, with point, interval, and
//! column-hoisted grid evaluation.
//!
//! Every op records which world axes its value depends on. Grid evaluation
//! stores an op only over those axes, so a heightfield's noise is evaluated
//! once per column however many cells lie beneath it.

use std::sync::Arc;

use super::fields::FieldSlot;
use super::interval::Interval;
use super::noise::NoiseGen;
use super::scatter::{SCATTER_FLOOR, ScatterGen, ScatterScratch};

/// Index of an op's result.
pub(crate) type Reg = u32;

/// Axis bit set: x = 1, y = 2, z = 4.
pub(crate) type Axes = u8;

pub(crate) const AXIS_X: Axes = 1;
pub(crate) const AXIS_Y: Axes = 2;
pub(crate) const AXIS_Z: Axes = 4;

/// Signed-distance primitives. Every one is 1-Lipschitz in position and in
/// each of its parameters, which is what their interval bound relies on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Primitive {
    Sphere,
    Box { half: [f64; 3], round: f64 },
    Torus,
    Capsule { a: [f64; 3], b: [f64; 3] },
    Cylinder,
    Cone,
}

#[derive(Clone, Debug)]
pub(crate) enum Op {
    X,
    Y,
    Z,
    Const(f64),
    /// Extra input supplied by the caller, such as a scatter var.
    Input(u32),
    /// Input that varies from point to point, such as the rock depth a
    /// carve reads; grid evaluation takes it as a whole buffer.
    Varying(u32),
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    Neg(Reg),
    Min(Reg, Reg),
    Max(Reg, Reg),
    Abs(Reg),
    Sqrt(Reg),
    Sin(Reg),
    Cos(Reg),
    Clamp(Reg, f64, f64),
    Smoothstep(Reg, f64, f64),
    Pow(Reg, f64),
    Spline(Reg, Arc<[(f64, f64)]>),
    Lerp(Reg, Reg, Reg),
    Terrace(Reg, f64, f64),
    /// `x - period * round(x / period)`: domain repetition.
    Repeat(Reg, f64),
    SmoothMax(Reg, Reg, f64),
    SmoothMin(Reg, Reg, f64),
    Noise([Reg; 3], Arc<NoiseGen>),
    /// Horizontal distance to the zero line of a planar noise at `(x, z)`.
    Fissure([Reg; 2], Arc<NoiseGen>),
    /// A world field at `(x, z)`.
    Field([Reg; 2], Arc<FieldSlot>),
    /// Negated signed distance of a primitive at the origin of `[x, y, z]`,
    /// with up to two expression parameters.
    Solid(Primitive, [Reg; 3], [Reg; 2]),
    /// Thickened gyroid with period in metres.
    Gyroid([Reg; 3], f64, Reg),
    /// Instances at `[x, y, z]`; the fourth register is the rock depth a
    /// shape may read, or an unused constant.
    Scatter([Reg; 4], Arc<ScatterGen>),
}

impl Op {
    pub(crate) fn inputs(&self) -> impl Iterator<Item = Reg> + '_ {
        let (fixed, extra): ([Option<Reg>; 3], &[Reg]) = match self {
            Self::X | Self::Y | Self::Z | Self::Const(_) | Self::Input(_) | Self::Varying(_) => {
                ([None; 3], &[])
            }
            Self::Neg(a)
            | Self::Abs(a)
            | Self::Sqrt(a)
            | Self::Sin(a)
            | Self::Cos(a)
            | Self::Clamp(a, ..)
            | Self::Smoothstep(a, ..)
            | Self::Pow(a, _)
            | Self::Spline(a, _)
            | Self::Terrace(a, ..)
            | Self::Repeat(a, _) => ([Some(*a), None, None], &[]),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b)
            | Self::SmoothMax(a, b, _)
            | Self::SmoothMin(a, b, _) => ([Some(*a), Some(*b), None], &[]),
            Self::Lerp(a, b, t) => ([Some(*a), Some(*b), Some(*t)], &[]),
            Self::Noise(domain, noise) => {
                if noise.is_planar() {
                    ([Some(domain[0]), Some(domain[2]), None], &[])
                } else {
                    ([Some(domain[0]), Some(domain[1]), Some(domain[2])], &[])
                }
            }
            Self::Fissure([x, z], _) | Self::Field([x, z], _) => ([Some(*x), Some(*z), None], &[]),
            Self::Solid(_, domain, params) => (domain.map(Some), &params[..]),
            Self::Gyroid(domain, _, thickness) => {
                (domain.map(Some), core::slice::from_ref(thickness))
            }
            Self::Scatter(domain, _) => (
                [Some(domain[0]), Some(domain[1]), Some(domain[2])],
                &domain[3..],
            ),
        };
        fixed.into_iter().flatten().chain(extra.iter().copied())
    }
}

/// Values of a tape's height-independent ops over one block's columns,
/// reused while grids keep to the same columns.
#[derive(Debug, Default)]
pub(crate) struct PlanarCache {
    xs: Vec<f64>,
    zs: Vec<f64>,
    values: Vec<Option<Vec<f64>>>,
}

/// Bounds of a tape's height-independent ops over one x and z extent, kept
/// for boxes stacked over the same columns.
#[derive(Debug, Default)]
pub(crate) struct PlanarIntervals {
    values: Vec<Option<Interval>>,
}

impl PlanarIntervals {
    /// Forgets every bound, for a box over other columns.
    pub(crate) fn clear(&mut self) {
        self.values.iter_mut().for_each(|value| *value = None);
    }
}

/// An op's input registers in [`Op::inputs`] order, gathered once so the
/// evaluators index them instead of re-deriving them at every point.
#[derive(Clone, Copy, Debug, Default)]
struct Arguments {
    registers: [Reg; 5],
    count: u8,
}

impl Arguments {
    fn of(op: &Op) -> Self {
        let mut arguments = Self::default();
        for input in op.inputs() {
            arguments.registers[usize::from(arguments.count)] = input;
            arguments.count += 1;
        }
        arguments
    }

    fn registers(&self) -> &[Reg] {
        &self.registers[..usize::from(self.count)]
    }
}

/// A compiled expression. The last op is the result.
#[derive(Clone, Debug)]
pub(crate) struct Tape {
    pub(crate) ops: Vec<Op>,
    pub(crate) axes: Vec<Axes>,
    last_use: Vec<u32>,
    arguments: Vec<Arguments>,
}

impl Tape {
    pub(crate) fn new(ops: Vec<Op>) -> Self {
        let mut axes = Vec::with_capacity(ops.len());
        for op in &ops {
            let own = match op {
                Op::X => AXIS_X,
                Op::Y => AXIS_Y,
                Op::Z => AXIS_Z,
                Op::Varying(_) => AXIS_X | AXIS_Y | AXIS_Z,
                _ => 0,
            };
            let mask = op
                .inputs()
                .fold(own, |mask, input| mask | axes[input as usize]);
            axes.push(mask);
        }
        let mut last_use =
            (0..u32::try_from(ops.len()).expect("tape fits u32")).collect::<Vec<_>>();
        for (index, op) in ops.iter().enumerate() {
            for input in op.inputs() {
                last_use[input as usize] = u32::try_from(index).expect("tape fits u32");
            }
        }
        let arguments = ops.iter().map(Arguments::of).collect();
        Self {
            ops,
            axes,
            last_use,
            arguments,
        }
    }

    pub(crate) fn result_axes(&self) -> Axes {
        *self.axes.last().expect("tapes are never empty")
    }

    /// Exact value at one point.
    pub(crate) fn eval(&self, point: [f64; 3], inputs: &[f64]) -> f64 {
        // Scattered shapes are evaluated per point and instance; small tapes
        // keep their values on the stack.
        const STACK_OPS: usize = 96;
        if self.ops.len() <= STACK_OPS {
            let mut values = [0.0; STACK_OPS];
            for (index, (op, arguments)) in self.ops.iter().zip(&self.arguments).enumerate() {
                values[index] = apply(op, arguments, &values[..index], point, inputs);
            }
            return values[self.ops.len() - 1];
        }
        let mut values = Vec::with_capacity(self.ops.len());
        for (op, arguments) in self.ops.iter().zip(&self.arguments) {
            let value = apply(op, arguments, &values, point, inputs);
            values.push(value);
        }
        *values.last().expect("tapes are never empty")
    }

    /// Bounds over an axis-aligned box.
    pub(crate) fn interval(&self, domain: [Interval; 3], inputs: &[Interval]) -> Interval {
        const STACK_OPS: usize = 96;
        if self.ops.len() <= STACK_OPS {
            let mut values = [Interval::point(0.0); STACK_OPS];
            for (index, (op, arguments)) in self.ops.iter().zip(&self.arguments).enumerate() {
                values[index] = apply_interval(op, arguments, &values[..index], domain, inputs);
            }
            return values[self.ops.len() - 1];
        }
        let mut values: Vec<Interval> = Vec::with_capacity(self.ops.len());
        for (op, arguments) in self.ops.iter().zip(&self.arguments) {
            let value = apply_interval(op, arguments, &values, domain, inputs);
            values.push(value);
        }
        *values.last().expect("tapes are never empty")
    }

    /// [`Self::interval`] without inputs, reusing the bounds of ops that do
    /// not depend on height from `cache`. The cache belongs to one x and z
    /// extent; clear it before a box over other columns.
    pub(crate) fn interval_columns(
        &self,
        domain: [Interval; 3],
        cache: &mut PlanarIntervals,
    ) -> Interval {
        const STACK_OPS: usize = 96;
        if cache.values.len() != self.ops.len() {
            cache.values.clear();
            cache.values.resize(self.ops.len(), None);
        }
        let mut stack = [Interval::point(0.0); STACK_OPS];
        let mut heap = Vec::new();
        let values: &mut [Interval] = if self.ops.len() <= STACK_OPS {
            &mut stack[..self.ops.len()]
        } else {
            heap.resize(self.ops.len(), Interval::point(0.0));
            &mut heap
        };
        for (index, (op, arguments)) in self.ops.iter().zip(&self.arguments).enumerate() {
            let planar = self.axes[index] & AXIS_Y == 0;
            values[index] = match cache.values[index] {
                Some(value) if planar => value,
                _ => {
                    let value = apply_interval(op, arguments, &values[..index], domain, &[]);
                    if planar {
                        cache.values[index] = Some(value);
                    }
                    value
                }
            };
        }
        values[self.ops.len() - 1]
    }

    /// [`Self::eval_grid_varying`] without inputs or a cache.
    #[cfg(test)]
    pub(crate) fn eval_grid(
        &self,
        dims: [usize; 3],
        coordinate: &dyn Fn(usize, usize) -> f64,
        out: &mut Vec<f64>,
    ) {
        self.eval_grid_varying(dims, coordinate, &[], None, out);
    }

    /// Values over a regular grid, x fastest. `coordinate(axis, index)`
    /// returns the world coordinate of a grid line, so callers own the
    /// rounding and point queries can reproduce it bit for bit. `varying`
    /// holds per-point inputs, each a full grid in the output's order. With
    /// a cache, values that do not vary with height are kept for the next
    /// grid over the same columns.
    ///
    /// Each op runs once over its whole extent: an input over fewer axes is
    /// spread over the op's extent first, so every op is one tight loop
    /// rather than one short loop per grid row.
    #[expect(
        clippy::too_many_lines,
        reason = "one extent loop per op kind keeps the hot path monomorphic"
    )]
    pub(crate) fn eval_grid_varying(
        &self,
        dims: [usize; 3],
        coordinate: &dyn Fn(usize, usize) -> f64,
        varying: &[&[f64]],
        mut cache: Option<&mut PlanarCache>,
        out: &mut Vec<f64>,
    ) {
        let mut buffers: Vec<Option<Vec<f64>>> = vec![None; self.ops.len()];
        let mut free: Vec<Vec<f64>> = Vec::new();
        let shape = |mask: Axes| {
            [
                if mask & AXIS_X != 0 { dims[0] } else { 1 },
                if mask & AXIS_Y != 0 { dims[1] } else { 1 },
                if mask & AXIS_Z != 0 { dims[2] } else { 1 },
            ]
        };
        let strides = |mask: Axes| {
            let extent = shape(mask);
            [
                usize::from(mask & AXIS_X != 0),
                if mask & AXIS_Y != 0 { extent[0] } else { 0 },
                if mask & AXIS_Z != 0 {
                    extent[0] * extent[1]
                } else {
                    0
                },
            ]
        };
        let axis_values = |axis: usize| {
            (0..dims[axis])
                .map(|index| coordinate(axis, index))
                .collect::<Vec<_>>()
        };
        let (xs, ys, zs) = (axis_values(0), axis_values(1), axis_values(2));
        if let Some(cache) = cache.as_deref_mut()
            && (cache.xs != xs || cache.zs != zs || cache.values.len() != self.ops.len())
        {
            cache.xs.clone_from(&xs);
            cache.zs.clone_from(&zs);
            cache.values.clear();
            cache.values.resize(self.ops.len(), None);
        }
        let planar = |index: usize| self.axes[index] & AXIS_Y == 0;
        let mut nearby = Vec::new();
        let mut spread: [Vec<f64>; 5] = Default::default();
        let mut scatter_scratch = ScatterScratch::default();
        for (index, op) in self.ops.iter().enumerate() {
            let mask = self.axes[index];
            if planar(index)
                && let Some(kept) = cache
                    .as_deref_mut()
                    .and_then(|cache| cache.values[index].take())
            {
                buffers[index] = Some(kept);
                continue;
            }
            let extent = shape(mask);
            let count = extent[0] * extent[1] * extent[2];
            let mut buffer = free.pop().unwrap_or_default();
            buffer.clear();
            buffer.reserve(count);
            let registers = self.arguments[index].registers();
            let input_values = |slot: usize| {
                buffers[registers[slot] as usize]
                    .as_deref()
                    .expect("inputs are evaluated before use")
            };
            // An input over some but not all of the op's axes is spread
            // over the op's extent, read where the old row loop read it.
            for (slot, &input) in registers.iter().enumerate() {
                let input_mask = self.axes[input as usize];
                if input_mask != mask && input_mask != 0 {
                    let values = input_values(slot);
                    let stride = strides(input_mask);
                    let target = &mut spread[slot];
                    target.clear();
                    for k in 0..extent[2] {
                        for j in 0..extent[1] {
                            let row = j * stride[1] + k * stride[2];
                            target.extend((0..extent[0]).map(|i| values[row + i * stride[0]]));
                        }
                    }
                }
            }
            let lanes: [Lane<'_>; 5] = core::array::from_fn(|slot| {
                let Some(&input) = registers.get(slot) else {
                    return Lane::Splat(0.0);
                };
                let input_mask = self.axes[input as usize];
                if input_mask == mask {
                    Lane::Row(input_values(slot))
                } else if input_mask == 0 {
                    Lane::Splat(input_values(slot)[0])
                } else {
                    Lane::Row(&spread[slot])
                }
            });
            match op {
                Op::X => buffer.extend_from_slice(&xs),
                Op::Y => buffer.extend_from_slice(&ys),
                Op::Z => buffer.extend_from_slice(&zs),
                Op::Const(value) => buffer.push(*value),
                Op::Input(_) => unreachable!("grids take per-point inputs as varying"),
                Op::Varying(input) => {
                    buffer.extend_from_slice(&varying[*input as usize][..count]);
                }
                Op::Scatter(_, scatter) => {
                    // The block's coordinates, possibly warped, span this
                    // box; only instances reaching it can matter to any
                    // point.
                    let span = |slot: usize| {
                        input_values(slot).iter().fold(
                            Interval::new(f64::INFINITY, f64::NEG_INFINITY),
                            |bounds, &value| {
                                Interval::new(bounds.lo.min(value), bounds.hi.max(value))
                            },
                        )
                    };
                    scatter.instances_near([span(0), span(1), span(2)], &mut nearby);
                    if nearby.is_empty() {
                        buffer.resize(count, SCATTER_FLOOR);
                    } else {
                        scatter.sample_among_many(
                            &nearby,
                            [lanes[0], lanes[1], lanes[2]],
                            lanes[3],
                            count,
                            &mut scatter_scratch,
                            &mut buffer,
                        );
                    }
                }
                _ => apply_lanes(op, &lanes[..registers.len()], count, &mut buffer),
            }
            buffers[index] = Some(buffer);
            for &input in registers {
                // Planar values are kept for the cache rather than reused.
                if self.last_use[input as usize] == u32::try_from(index).expect("tape fits u32")
                    && input as usize != self.ops.len() - 1
                    && !(cache.is_some() && planar(input as usize))
                    && let Some(released) = buffers[input as usize].take()
                {
                    free.push(released);
                }
            }
        }
        let result = buffers
            .pop()
            .flatten()
            .expect("the result is evaluated last");
        if let Some(cache) = cache {
            let last = self.ops.len() - 1;
            for (index, buffer) in buffers.into_iter().enumerate() {
                if planar(index) {
                    cache.values[index] = buffer;
                }
            }
            if planar(last) {
                cache.values[last] = Some(result.clone());
            }
        }
        let stride = strides(self.result_axes());
        out.clear();
        out.reserve(dims[0] * dims[1] * dims[2]);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    out.push(result[i * stride[0] + j * stride[1] + k * stride[2]]);
                }
            }
        }
    }

    /// Values at `count` points at once, appended to `out`, each equal bit
    /// for bit to [`Self::eval`] there. Coordinates and inputs are lanes, so
    /// a value shared by every point is passed once. Ops that depend only on
    /// shared values are evaluated once; the rest run op by op over chunks
    /// of points small enough to stay in cache.
    pub(crate) fn eval_many(
        &self,
        point: [Lane<'_>; 3],
        inputs: &[Lane<'_>],
        count: usize,
        scratch: &mut ManyScratch,
        out: &mut Vec<f64>,
    ) {
        let ManyScratch { shared, rows } = scratch;
        if rows.len() < self.ops.len() {
            rows.resize_with(self.ops.len(), Vec::new);
        }
        shared.clear();
        for (index, op) in self.ops.iter().enumerate() {
            let splat = |lane: Lane<'_>| match lane {
                Lane::Splat(value) => Some(value),
                Lane::Row(_) => None,
            };
            let value = match op {
                Op::X => splat(point[0]),
                Op::Y => splat(point[1]),
                Op::Z => splat(point[2]),
                Op::Const(value) => Some(*value),
                Op::Input(input) | Op::Varying(input) => splat(inputs[*input as usize]),
                _ => {
                    let registers = self.arguments[index].registers();
                    let mut arguments = [0.0; 5];
                    let mut every = true;
                    for (argument, &input) in arguments.iter_mut().zip(registers) {
                        match shared[input as usize] {
                            Some(value) => *argument = value,
                            None => every = false,
                        }
                    }
                    every.then(|| apply_gathered(op, &arguments[..registers.len()], [0.0; 3]))
                }
            };
            shared.push(value);
        }
        let last = self.ops.len() - 1;
        if let Some(value) = shared[last] {
            out.extend(std::iter::repeat_n(value, count));
            return;
        }
        let mut start = 0;
        while start < count {
            let end = (start + MANY_CHUNK).min(count);
            for (index, op) in self.ops.iter().enumerate() {
                if shared[index].is_some() {
                    continue;
                }
                let (done, rest) = rows.split_at_mut(index);
                let buffer = &mut rest[0];
                buffer.clear();
                match op {
                    Op::X => buffer.extend_from_slice(point[0].part(start..end)),
                    Op::Y => buffer.extend_from_slice(point[1].part(start..end)),
                    Op::Z => buffer.extend_from_slice(point[2].part(start..end)),
                    Op::Input(input) | Op::Varying(input) => {
                        buffer.extend_from_slice(inputs[*input as usize].part(start..end));
                    }
                    _ => {
                        let registers = self.arguments[index].registers();
                        let lanes: [Lane<'_>; 5] = core::array::from_fn(|slot| {
                            registers.get(slot).map_or(Lane::Splat(0.0), |&input| {
                                shared[input as usize]
                                    .map_or_else(|| Lane::Row(&done[input as usize]), Lane::Splat)
                            })
                        });
                        apply_lanes(op, &lanes[..registers.len()], end - start, buffer);
                    }
                }
            }
            out.extend_from_slice(&rows[last]);
            start = end;
        }
    }
}

/// Points per chunk of [`Tape::eval_many`]: a chunk of every op's values
/// stays within the first-level cache.
const MANY_CHUNK: usize = 128;

/// Reusable buffers for [`Tape::eval_many`].
#[derive(Debug, Default)]
pub(crate) struct ManyScratch {
    shared: Vec<Option<f64>>,
    rows: Vec<Vec<f64>>,
}

/// One argument's values over a run of points: a contiguous row, or one
/// value shared by every point.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Lane<'a> {
    Row(&'a [f64]),
    Splat(f64),
}

impl<'a> Lane<'a> {
    /// The value at point `index`.
    pub(crate) fn at(self, index: usize) -> f64 {
        match self {
            Self::Row(values) => values[index],
            Self::Splat(value) => value,
        }
    }

    /// A row's values over `range`; shared values are never read this way.
    fn part(self, range: core::ops::Range<usize>) -> &'a [f64] {
        match self {
            Self::Row(values) => &values[range],
            Self::Splat(_) => unreachable!("shared values are evaluated once"),
        }
    }
}

/// Appends `op`'s values at `count` points whose arguments are `lanes`.
/// Every point goes through exactly [`apply_gathered`]'s arithmetic, so the
/// results match point evaluation bit for bit. Coordinates, constants, and
/// inputs are the caller's.
fn apply_lanes(op: &Op, lanes: &[Lane<'_>], count: usize, out: &mut Vec<f64>) {
    macro_rules! unary {
        ($f:expr) => {
            match lanes[0] {
                Lane::Row(a) => out.extend(a[..count].iter().map(|&a| $f(a))),
                Lane::Splat(a) => out.extend(std::iter::repeat_n($f(a), count)),
            }
        };
    }
    macro_rules! binary {
        ($f:expr) => {
            match (lanes[0], lanes[1]) {
                (Lane::Row(a), Lane::Row(b)) => {
                    out.extend(a[..count].iter().zip(&b[..count]).map(|(&a, &b)| $f(a, b)));
                }
                (Lane::Row(a), Lane::Splat(b)) => {
                    out.extend(a[..count].iter().map(|&a| $f(a, b)));
                }
                (Lane::Splat(a), Lane::Row(b)) => {
                    out.extend(b[..count].iter().map(|&b| $f(a, b)));
                }
                (Lane::Splat(a), Lane::Splat(b)) => {
                    out.extend(std::iter::repeat_n($f(a, b), count));
                }
            }
        };
    }
    match op {
        Op::X | Op::Y | Op::Z | Op::Const(_) | Op::Input(_) | Op::Varying(_) => {
            unreachable!("coordinates, constants, and inputs are read directly")
        }
        Op::Add(..) => binary!(|a: f64, b: f64| a + b),
        Op::Sub(..) => binary!(|a: f64, b: f64| a - b),
        Op::Mul(..) => binary!(|a: f64, b: f64| a * b),
        Op::Div(..) => binary!(|a: f64, b: f64| a / b),
        Op::Neg(_) => unary!(|a: f64| -a),
        Op::Min(..) => binary!(f64::min),
        Op::Max(..) => binary!(f64::max),
        Op::Abs(_) => unary!(f64::abs),
        Op::Sqrt(_) => unary!(|a: f64| a.max(0.0).sqrt()),
        Op::Clamp(_, lo, hi) => unary!(|a: f64| a.clamp(*lo, *hi)),
        Op::Smoothstep(_, lo, hi) => unary!(|a: f64| smoothstep(*lo, *hi, a)),
        Op::Pow(_, exponent) => unary!(|a: f64| signed_pow(a, *exponent)),
        Op::Terrace(_, step, sharpness) => unary!(|a: f64| terrace(a, *step, *sharpness)),
        Op::SmoothMax(_, _, k) => binary!(|a: f64, b: f64| smooth_max(a, b, *k)),
        Op::SmoothMin(_, _, k) => binary!(|a: f64, b: f64| -smooth_max(-a, -b, *k)),
        Op::Solid(primitive, ..) => out.extend((0..count).map(|i| {
            solid(
                *primitive,
                [lanes[0].at(i), lanes[1].at(i), lanes[2].at(i)],
                [3, 4].map(|slot| lanes.get(slot).map_or(0.0, |lane| lane.at(i))),
            )
        })),
        Op::Noise(_, noise) => {
            if noise.is_planar() {
                noise.sample_many([lanes[0], Lane::Splat(0.0), lanes[1]], count, out);
            } else {
                noise.sample_many([lanes[0], lanes[1], lanes[2]], count, out);
            }
        }
        _ => out.extend((0..count).map(|i| {
            let mut arguments = [0.0; 5];
            for (argument, lane) in arguments.iter_mut().zip(lanes) {
                *argument = lane.at(i);
            }
            apply_gathered(op, &arguments[..lanes.len()], [0.0; 3])
        })),
    }
}

fn apply(op: &Op, registers: &Arguments, values: &[f64], point: [f64; 3], inputs: &[f64]) -> f64 {
    if let Op::Input(index) | Op::Varying(index) = op {
        return inputs[*index as usize];
    }
    let mut arguments = [0.0; 5];
    let registers = registers.registers();
    for (argument, &input) in arguments.iter_mut().zip(registers) {
        *argument = values[input as usize];
    }
    apply_gathered(op, &arguments[..registers.len()], point)
}

/// Evaluates one op from its gathered argument values. Point and grid
/// evaluation both call this, so they agree bit for bit.
fn apply_gathered(op: &Op, arguments: &[f64], point: [f64; 3]) -> f64 {
    let a = arguments.first().copied().unwrap_or(0.0);
    let b = arguments.get(1).copied().unwrap_or(0.0);
    match op {
        Op::X => point[0],
        Op::Y => point[1],
        Op::Z => point[2],
        Op::Const(value) => *value,
        Op::Input(_) | Op::Varying(_) => unreachable!("inputs are read directly"),
        Op::Add(..) => a + b,
        Op::Sub(..) => a - b,
        Op::Mul(..) => a * b,
        Op::Div(..) => a / b,
        Op::Neg(_) => -a,
        Op::Min(..) => a.min(b),
        Op::Max(..) => a.max(b),
        Op::Abs(_) => a.abs(),
        Op::Sqrt(_) => a.max(0.0).sqrt(),
        Op::Sin(_) => a.sin(),
        Op::Cos(_) => a.cos(),
        Op::Clamp(_, lo, hi) => a.clamp(*lo, *hi),
        Op::Smoothstep(_, lo, hi) => smoothstep(*lo, *hi, a),
        Op::Pow(_, exponent) => signed_pow(a, *exponent),
        Op::Spline(_, points) => spline(points, a),
        Op::Lerp(..) => {
            let t = arguments[2].clamp(0.0, 1.0);
            (b - a).mul_add(t, a)
        }
        Op::Terrace(_, step, sharpness) => terrace(a, *step, *sharpness),
        Op::Repeat(_, period) => repeat(a, *period),
        Op::SmoothMax(_, _, k) => smooth_max(a, b, *k),
        Op::SmoothMin(_, _, k) => -smooth_max(-a, -b, *k),
        Op::Noise(_, noise) => {
            if noise.is_planar() {
                noise.sample(a, 0.0, b)
            } else {
                noise.sample(a, b, arguments[2])
            }
        }
        Op::Fissure(_, noise) => noise.line_distance(a, b),
        Op::Field(_, field) => field.sample(a, b),
        Op::Solid(primitive, ..) => solid(
            *primitive,
            [a, b, arguments[2]],
            [
                arguments.get(3).copied().unwrap_or(0.0),
                arguments.get(4).copied().unwrap_or(0.0),
            ],
        ),
        Op::Gyroid(_, period, _) => gyroid([a, b, arguments[2]], *period, arguments[3]),
        Op::Scatter(_, scatter) => scatter.sample([a, b, arguments[2]], arguments[3]),
    }
}

fn apply_interval(
    op: &Op,
    registers: &Arguments,
    values: &[Interval],
    domain: [Interval; 3],
    inputs: &[Interval],
) -> Interval {
    let registers = registers.registers();
    let argument = |index: usize| {
        registers
            .get(index)
            .map_or(Interval::point(0.0), |&reg| values[reg as usize])
    };
    let a = argument(0);
    let b = argument(1);
    match op {
        Op::X => domain[0],
        Op::Y => domain[1],
        Op::Z => domain[2],
        Op::Const(value) => Interval::point(*value),
        Op::Input(index) | Op::Varying(index) => inputs[*index as usize],
        Op::Add(..) => a.add(b),
        Op::Sub(..) => a.sub(b),
        Op::Mul(..) => a.mul(b),
        Op::Div(..) => a.div(b),
        Op::Neg(_) => a.neg(),
        Op::Min(..) => a.min(b),
        Op::Max(..) => a.max(b),
        Op::Abs(_) => a.abs(),
        Op::Sqrt(_) => a.monotone(|value| value.max(0.0).sqrt()),
        Op::Sin(_) => a.sin(),
        Op::Cos(_) => a.cos(),
        Op::Clamp(_, lo, hi) => a.clamp(*lo, *hi),
        Op::Smoothstep(_, lo, hi) => a.monotone(|value| smoothstep(*lo, *hi, value)),
        Op::Pow(_, exponent) => a.monotone(|value| signed_pow(value, *exponent)),
        Op::Spline(_, points) => spline_interval(points, a),
        Op::Lerp(..) => {
            let t = argument(2).clamp(0.0, 1.0);
            a.add(b.sub(a).mul(t))
        }
        Op::Terrace(_, step, sharpness) => a.monotone(|value| terrace(value, *step, *sharpness)),
        Op::Repeat(_, period) => {
            let half = period * 0.5;
            let whole = Interval::new(-half, half);
            let turn = |value: f64| (value / period).round();
            if !(a.lo.is_finite() && a.hi.is_finite()) || turn(a.hi) > turn(a.lo) {
                whole
            } else {
                Interval::new(repeat(a.lo, *period), repeat(a.hi, *period))
            }
        }
        Op::SmoothMax(_, _, k) => {
            let exact = a.max(b);
            Interval::new(exact.lo, exact.hi + k * 0.25)
        }
        Op::SmoothMin(_, _, k) => {
            let exact = a.min(b);
            Interval::new(exact.lo - k * 0.25, exact.hi)
        }
        Op::Noise(_, noise) => {
            if noise.is_planar() {
                noise.interval(a, Interval::point(0.0), b)
            } else {
                noise.interval(a, b, argument(2))
            }
        }
        Op::Fissure(_, noise) => noise.line_distance_interval(a, b),
        Op::Field(_, field) => field.interval(a, b),
        Op::Solid(primitive, ..) => {
            solid_interval(*primitive, [a, b, argument(2)], [argument(3), argument(4)])
        }
        Op::Gyroid(_, period, _) => {
            let position = [a, b, argument(2)];
            let thickness = argument(3);
            let reach = position
                .iter()
                .map(|axis| axis.radius() * axis.radius())
                .sum::<f64>()
                .sqrt();
            if !reach.is_finite() || !thickness.lo.is_finite() || !thickness.hi.is_finite() {
                return Interval::EVERYTHING;
            }
            Interval::around(
                gyroid(position.map(Interval::centre), *period, thickness.centre()),
                GYROID_LIPSCHITZ,
                reach,
            )
            .add(Interval::new(-thickness.radius(), thickness.radius()))
        }
        Op::Scatter(_, scatter) => scatter.interval([a, b, argument(2)], argument(3)),
    }
}

/// Primitives are 1-Lipschitz in position and parameters, so the value at
/// the centre bounds a box to within its reach.
fn solid_interval(
    primitive: Primitive,
    position: [Interval; 3],
    parameters: [Interval; 2],
) -> Interval {
    let reach = position
        .iter()
        .map(|axis| axis.radius() * axis.radius())
        .sum::<f64>()
        .sqrt()
        + parameters.iter().map(|axis| axis.radius()).sum::<f64>();
    if !reach.is_finite() {
        return Interval::EVERYTHING;
    }
    Interval::around(
        solid(
            primitive,
            position.map(Interval::centre),
            parameters.map(Interval::centre),
        ),
        1.0,
        reach,
    )
}

pub(crate) fn smoothstep(lo: f64, hi: f64, value: f64) -> f64 {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * 2.0f64.mul_add(-t, 3.0)
}

fn repeat(value: f64, period: f64) -> f64 {
    value - period * (value / period).round()
}

fn signed_pow(value: f64, exponent: f64) -> f64 {
    value.signum() * value.abs().powf(exponent)
}

fn spline(points: &[(f64, f64)], value: f64) -> f64 {
    let Some(first) = points.first() else {
        return value;
    };
    if value <= first.0 {
        return first.1;
    }
    for pair in points.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        if value <= right.0 {
            let span = right.0 - left.0;
            if span <= 0.0 {
                return right.1;
            }
            return (right.1 - left.1).mul_add((value - left.0) / span, left.1);
        }
    }
    points.last().map_or(value, |last| last.1)
}

fn spline_interval(points: &[(f64, f64)], value: Interval) -> Interval {
    let mut bounds =
        Interval::point(spline(points, value.lo)).hull(Interval::point(spline(points, value.hi)));
    for &(input, output) in points {
        if value.lo < input && input < value.hi {
            bounds = bounds.hull(Interval::point(output));
        }
    }
    bounds
}

fn terrace(value: f64, step: f64, sharpness: f64) -> f64 {
    if step <= 0.0 {
        return value;
    }
    let scaled = value / step;
    let floor = scaled.floor();
    let fraction = scaled - floor;
    let riser = smoothstep(sharpness.clamp(0.0, 0.99), 1.0, fraction);
    (floor + riser) * step
}

/// Polynomial smooth maximum with fillet width `k`.
fn smooth_max(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Maximum slope of the gyroid expression divided by its distance scale.
const GYROID_LIPSCHITZ: f64 = 1.0;

fn gyroid(position: [f64; 3], period: f64, thickness: f64) -> f64 {
    let scale = core::f64::consts::TAU / period;
    let [x, y, z] = position.map(|value| value * scale);
    let field = x.sin().mul_add(y.cos(), y.sin() * z.cos()) + z.sin() * x.cos();
    // |∇field| ≤ 2√3 * scale; dividing by it keeps the result a distance bound.
    thickness * 0.5 - field.abs() / (2.0 * 3.0_f64.sqrt() * scale)
}

pub(crate) fn solid(primitive: Primitive, position: [f64; 3], parameters: [f64; 2]) -> f64 {
    let [x, y, z] = position;
    let length = |a: f64, b: f64, c: f64| a.hypot(b).hypot(c);
    let distance = match primitive {
        Primitive::Sphere => length(x, y, z) - parameters[0],
        Primitive::Box { half, round } => {
            let q = [
                x.abs() - (half[0] - round),
                y.abs() - (half[1] - round),
                z.abs() - (half[2] - round),
            ];
            length(q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)) + q[0].max(q[1]).max(q[2]).min(0.0)
                - round
        }
        Primitive::Torus => (x.hypot(z) - parameters[0]).hypot(y) - parameters[1],
        Primitive::Capsule { a, b } => {
            let pa = [x - a[0], y - a[1], z - a[2]];
            let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let along = ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2])
                / (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).max(f64::MIN_POSITIVE))
            .clamp(0.0, 1.0);
            length(
                pa[0] - ba[0] * along,
                pa[1] - ba[1] * along,
                pa[2] - ba[2] * along,
            ) - parameters[0]
        }
        Primitive::Cylinder => {
            let d = [x.hypot(z) - parameters[0], y.abs() - parameters[1]];
            d[0].max(d[1]).min(0.0) + d[0].max(0.0).hypot(d[1].max(0.0))
        }
        Primitive::Cone => {
            let (radius, height) = (parameters[0].max(1.0e-6), parameters[1].max(1.0e-6));
            let slant = radius.hypot(height);
            let side = (x.hypot(z) * height + y * radius - radius * height) / slant;
            side.max(-y).max(y - height)
        }
    };
    -distance
}

#[cfg(test)]
mod tests {
    use super::{Primitive, solid};

    #[test]
    fn primitives_are_solid_inside_and_empty_outside() {
        assert!(solid(Primitive::Sphere, [0.5, 0.0, 0.0], [1.0, 0.0]) > 0.0);
        assert!(solid(Primitive::Sphere, [1.5, 0.0, 0.0], [1.0, 0.0]) < 0.0);
        assert!(solid(Primitive::Torus, [3.0, 0.0, 0.0], [3.0, 0.5]) > 0.0);
        assert!(solid(Primitive::Torus, [0.0, 0.0, 0.0], [3.0, 0.5]) < 0.0);
        assert!(solid(Primitive::Cone, [0.0, 1.0, 0.0], [2.0, 4.0]) > 0.0);
        assert!(solid(Primitive::Cone, [0.0, 4.5, 0.0], [2.0, 4.0]) < 0.0);
        assert!(solid(Primitive::Cylinder, [0.0, 0.9, 0.0], [1.0, 1.0]) > 0.0);
        let boxed = Primitive::Box {
            half: [1.0, 2.0, 3.0],
            round: 0.2,
        };
        assert!(solid(boxed, [0.9, 1.9, 2.5], [0.0; 2]) > 0.0);
        assert!(solid(boxed, [1.1, 0.0, 0.0], [0.0; 2]) < 0.0);
    }
}
