//! Fractal noise with known output range and slope, so regions can be bounded.

mod kernels;

use std::cell::RefCell;

use fastnoise_lite::{
    CellularDistanceFunction, CellularReturnType, FastNoiseLite, NoiseType, RotationType3D,
};

use self::kernels::{Kernel, LANES};
use super::interval::Interval;
use super::scatter::mix;
use super::spec::{Dims, Fractal, NoiseDoc, NoiseKind};
use super::tape::Lane;

/// Upper bound on the gradient magnitude of one base octave at unit frequency,
/// after clamping to [-1, 1]. Measured with margin by
/// `base_noise_slope_stays_below_the_declared_bound`.
fn base_slope(kind: NoiseKind, dims: Dims) -> f64 {
    match (kind, dims) {
        (NoiseKind::Simplex, Dims::Two) => 8.5,
        (NoiseKind::SimplexSmooth, Dims::Two) => 5.5,
        (NoiseKind::Simplex | NoiseKind::SimplexSmooth | NoiseKind::Perlin, _) => 3.6,
        (NoiseKind::Value, _) => 2.0,
        (NoiseKind::Cells, _) => 1.5,
        (NoiseKind::CellEdges, _) => 2.8,
    }
}

/// Upper bound on the second directional derivative of one base octave at
/// unit frequency, for the smooth kinds; cellular noise has kinks and none.
/// Measured with margin by `base_noise_curvature_stays_below_the_declared_bound`.
fn base_curvature(kind: NoiseKind, dims: Dims) -> Option<f64> {
    match (kind, dims) {
        (NoiseKind::Simplex, Dims::Two) => Some(80.0),
        (NoiseKind::SimplexSmooth, Dims::Two) => Some(45.0),
        (NoiseKind::Perlin, Dims::Two) => Some(20.0),
        (NoiseKind::Simplex | NoiseKind::SimplexSmooth | NoiseKind::Perlin, Dims::Three) => {
            Some(17.0)
        }
        (NoiseKind::Value, _) => Some(7.5),
        (NoiseKind::Cells | NoiseKind::CellEdges, _) => None,
    }
}

thread_local! {
    /// One octave's raw values in [`NoiseGen::sample_many`].
    static SAMPLE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Slack for the tiny steps float rounding leaves in the base noise.
const STEP_SLACK: f64 = 0.01;

/// Half step of the central differences that estimate an octave's gradient,
/// in noise units.
const GRADIENT_STEP: f64 = 1.0e-3;

/// Bound on the error of those estimates: truncation plus `f32` rounding.
const GRADIENT_SLACK: f64 = 1.0e-3;

/// Beyond this reach in noise units a Taylor bound is no tighter than the
/// slope bound, so it is not computed.
const TAYLOR_REACH: f64 = 0.6;

/// Rounding in the base noise's single-precision lookups, in noise units.
const LINE_VALUE_SLACK: f64 = 1.0e-4;

/// Widest square, in wavelengths of the finest octave, over which a line
/// distance is bounded in one piece.
const LINE_PIECE_SPAN: f64 = 0.03;

/// A compiled noise lookup.
pub(crate) struct NoiseGen {
    octaves: Vec<FastNoiseLite>,
    /// Each octave's seed, for the lane kernel.
    seeds: Vec<i32>,
    /// The lane kernel equal to the octaves' generator, if it has one.
    kernel: Option<Kernel>,
    /// Per-octave lattice offsets, so no two noises share a zero at the origin.
    shifts: Vec<[f64; 3]>,
    frequencies: Vec<f64>,
    weights: Vec<f64>,
    fractal: Fractal,
    dims: Dims,
    amp: f64,
    offset: f64,
    base_slope: f64,
    base_curvature: Option<f64>,
}

impl core::fmt::Debug for NoiseGen {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("NoiseGen")
            .field("frequencies", &self.frequencies)
            .field("amp", &self.amp)
            .finish_non_exhaustive()
    }
}

impl NoiseGen {
    pub(crate) fn new(doc: &NoiseDoc, seed: i32) -> Self {
        let octave_count = usize::from(doc.octaves.max(1));
        let mut octaves = Vec::with_capacity(octave_count);
        let mut seeds = Vec::with_capacity(octave_count);
        let mut shifts = Vec::with_capacity(octave_count);
        let mut frequencies = Vec::with_capacity(octave_count);
        let mut weights = Vec::with_capacity(octave_count);
        let mut frequency = doc.freq;
        let mut weight = 1.0;
        for index in 0..octave_count {
            let octave_seed =
                seed.wrapping_add(i32::try_from(index).expect("octave count fits i32") * 7_919);
            let mut noise = FastNoiseLite::with_seed(octave_seed);
            noise.set_frequency(Some(1.0));
            match (doc.kind, doc.dims) {
                // FastNoiseLite's 3D simplex lattices leave steps of about
                // half a percent of the range, which would read as ledges on
                // tall terrain. Rotated gradient noise is continuous.
                (
                    NoiseKind::Simplex | NoiseKind::SimplexSmooth | NoiseKind::Perlin,
                    Dims::Three,
                ) => {
                    noise.set_noise_type(Some(NoiseType::Perlin));
                    noise.set_rotation_type_3d(Some(RotationType3D::ImproveXZPlanes));
                }
                (NoiseKind::Simplex, Dims::Two) => {
                    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
                }
                (NoiseKind::SimplexSmooth, Dims::Two) => {
                    noise.set_noise_type(Some(NoiseType::OpenSimplex2S));
                }
                (NoiseKind::Perlin, Dims::Two) => noise.set_noise_type(Some(NoiseType::Perlin)),
                (NoiseKind::Value, _) => noise.set_noise_type(Some(NoiseType::ValueCubic)),
                (NoiseKind::Cells | NoiseKind::CellEdges, _) => {
                    noise.set_noise_type(Some(NoiseType::Cellular));
                    noise.set_cellular_distance_function(Some(CellularDistanceFunction::Euclidean));
                    noise.set_cellular_return_type(Some(if doc.kind == NoiseKind::Cells {
                        CellularReturnType::Distance
                    } else {
                        CellularReturnType::Distance2Sub
                    }));
                }
            }
            octaves.push(noise);
            seeds.push(octave_seed);
            #[expect(clippy::cast_sign_loss, reason = "the seed is hashed bit for bit")]
            let mut state = mix(u64::from(octave_seed as u32) ^ 0x5bd1_e995);
            shifts.push(core::array::from_fn(|_| {
                state = mix(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
                #[expect(clippy::cast_precision_loss, reason = "20 random bits")]
                let unit = (state >> 44) as f64 / f64::from(1_u32 << 20);
                unit * 997.0
            }));
            frequencies.push(frequency);
            weights.push(weight);
            frequency *= doc.lacunarity;
            weight *= doc.gain;
        }
        let total: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        let kernel = match (doc.kind, doc.dims) {
            (NoiseKind::Simplex | NoiseKind::SimplexSmooth | NoiseKind::Perlin, Dims::Three) => {
                Some(Kernel::Perlin3)
            }
            (NoiseKind::Simplex, Dims::Two) => Some(Kernel::Simplex2),
            (NoiseKind::SimplexSmooth, Dims::Two) => Some(Kernel::SimplexSmooth2),
            _ => None,
        };
        Self {
            octaves,
            seeds,
            kernel,
            shifts,
            frequencies,
            weights,
            fractal: doc.fractal,
            dims: doc.dims,
            amp: doc.amp,
            offset: doc.offset,
            base_slope: base_slope(doc.kind, doc.dims),
            base_curvature: base_curvature(doc.kind, doc.dims),
        }
    }

    pub(crate) const fn is_planar(&self) -> bool {
        matches!(self.dims, Dims::Two)
    }

    pub(crate) fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut sum = 0.0;
        for (((noise, shift), frequency), weight) in self
            .octaves
            .iter()
            .zip(&self.shifts)
            .zip(&self.frequencies)
            .zip(&self.weights)
        {
            sum += weight * self.octave(noise, *shift, *frequency, x, y, z);
        }
        self.amp.mul_add(sum, self.offset)
    }

    /// [`Self::sample`] at `count` points, appended to `out` and equal to it
    /// bit for bit at each. Generators with a lane kernel look up a whole
    /// octave across the points at once; each point still sums its octaves
    /// in order.
    pub(crate) fn sample_many(&self, point: [Lane<'_>; 3], count: usize, out: &mut Vec<f64>) {
        let Some(kernel) = self.kernel else {
            out.extend((0..count).map(|index| {
                let y = if self.is_planar() {
                    0.0
                } else {
                    point[1].at(index)
                };
                self.sample(point[0].at(index), y, point[2].at(index))
            }));
            return;
        };
        let start = out.len();
        out.resize(start + count, 0.0);
        let sums = &mut out[start..];
        let planar = self.is_planar();
        SAMPLE_SCRATCH.with_borrow_mut(|raw| {
            for (((&seed, shift), frequency), weight) in self
                .seeds
                .iter()
                .zip(&self.shifts)
                .zip(&self.frequencies)
                .zip(&self.weights)
            {
                raw.clear();
                for first in (0..count).step_by(LANES) {
                    // The library's coordinates: 2D kernels take x and z.
                    let mut lanes = [[0.0; LANES]; 3];
                    for index in first..count.min(first + LANES) {
                        let lane = index - first;
                        let x = point[0].at(index).mul_add(*frequency, shift[0]);
                        let z = point[2].at(index).mul_add(*frequency, shift[2]);
                        lanes[0][lane] = x;
                        if planar {
                            lanes[1][lane] = z;
                        } else {
                            lanes[1][lane] = point[1].at(index).mul_add(*frequency, shift[1]);
                            lanes[2][lane] = z;
                        }
                    }
                    raw.extend(kernel.run(seed, &lanes[0], &lanes[1], &lanes[2]));
                }
                for (sum, &value) in sums.iter_mut().zip(raw.iter()) {
                    *sum += weight * self.shape(f64::from(value).clamp(-1.0, 1.0));
                }
            }
        });
        for sum in sums {
            *sum = self.amp.mul_add(*sum, self.offset);
        }
    }

    pub(crate) fn range(&self) -> Interval {
        Interval::new(self.offset - self.amp.abs(), self.offset + self.amp.abs())
    }

    /// Bounds over a box. Each octave takes the tighter of two sound bounds:
    /// its value at the centre plus its slope bound times the reach, and a
    /// second-order Taylor bound from its gradient at the centre and its
    /// curvature bound, which is far tighter for octaves much wider than the
    /// box.
    pub(crate) fn interval(&self, x: Interval, y: Interval, z: Interval) -> Interval {
        let radii = match self.dims {
            Dims::Two => [x.radius(), 0.0, z.radius()],
            Dims::Three => [x.radius(), y.radius(), z.radius()],
        };
        let reach = radii[0].hypot(radii[1]).hypot(radii[2]);
        if !reach.is_finite() {
            return self.range();
        }
        let centre = [x.centre(), y.centre(), z.centre()];
        let fractal_slope = match self.fractal {
            Fractal::Fbm => 1.0,
            Fractal::Ridged | Fractal::Billow => 2.0,
        };
        // Each octave saturates at its own amplitude, so fine octaves add
        // their small weight rather than their large slope.
        let mut lo = 0.0;
        let mut hi = 0.0;
        for (((noise, shift), frequency), weight) in self
            .octaves
            .iter()
            .zip(&self.shifts)
            .zip(&self.frequencies)
            .zip(&self.weights)
        {
            let raw = self.raw_octave(noise, *shift, *frequency, centre);
            let value = self.shape(raw);
            let spread = self.base_slope * frequency * fractal_slope * reach + STEP_SLACK;
            let mut bounds = Interval::new((value - spread).max(-1.0), (value + spread).min(1.0));
            if let Some(curvature) = self.base_curvature
                && frequency * reach < TAYLOR_REACH
            {
                let taylor =
                    self.taylor_octave((noise, *shift, *frequency), centre, radii, raw, curvature);
                let taylor = taylor.clamp(-1.0, 1.0);
                bounds = bounds.intersect(self.shape_interval(taylor));
            }
            lo += weight * bounds.lo;
            hi += weight * bounds.hi;
        }
        Interval::new(
            self.amp.abs().mul_add(lo, self.offset),
            self.amp.abs().mul_add(hi, self.offset),
        )
        .intersect(self.range())
    }

    /// Horizontal distance to the nearest zero of the noise, first-order:
    /// `|n| / |∇n|`, with the gradient from forward differences a small
    /// fraction of the finest octave's wavelength long.
    pub(crate) fn line_distance(&self, x: f64, z: f64) -> f64 {
        let value = self.sample(x, 0.0, z);
        let step = GRADIENT_STEP / self.frequencies.last().copied().unwrap_or(1.0);
        let dx = self.sample(x + step, 0.0, z) - value;
        let dz = self.sample(x, 0.0, z + step) - value;
        let slope = dx.hypot(dz) / step;
        value.abs() / slope.max(f64::MIN_POSITIVE)
    }

    /// Bounds on [`Self::line_distance`] over a box: `|n|` over it divided by
    /// the steepest gradient it can hold bounds the distance from below;
    /// above there is no useful bound.
    pub(crate) fn line_distance_interval(&self, x: Interval, z: Interval) -> Interval {
        if !(x.lo.is_finite() && x.hi.is_finite() && z.lo.is_finite() && z.hi.is_finite()) {
            return Interval::new(0.0, f64::INFINITY);
        }
        // Both bounds loosen with the square of a box's size against the
        // finest wavelength, so wide boxes are split into squares a small
        // fraction of it across; the distance bound is the smallest of theirs.
        let finest = self.frequencies.last().copied().unwrap_or(1.0);
        let span = (x.hi - x.lo).max(z.hi - z.lo) * finest;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to a few pieces"
        )]
        let pieces = (span / LINE_PIECE_SPAN).ceil().clamp(1.0, 6.0) as u32;
        let piece = |range: Interval, index: u32| {
            let width = (range.hi - range.lo) / f64::from(pieces);
            Interval::new(
                width.mul_add(f64::from(index), range.lo),
                width.mul_add(f64::from(index + 1), range.lo),
            )
        };
        let mut lowest = f64::INFINITY;
        for i in 0..pieces {
            for k in 0..pieces {
                let (x, z) = (piece(x, i), piece(z, k));
                let bound = self.taylor_line_bound(x, z).unwrap_or_else(|| {
                    let value = self.interval(x, Interval::point(0.0), z).abs();
                    value.lo / self.slope_bound(x, z)
                });
                if bound <= 0.0 {
                    return Interval::new(0.0, f64::INFINITY);
                }
                lowest = lowest.min(bound);
            }
        }
        Interval::new(lowest, f64::INFINITY)
    }

    /// Lower bound on the line distance over a box from the whole noise's
    /// value and gradient at the centre and its curvature bound: within the
    /// reach `r`, `|n| >= |n(c)| - |∇n(c)| r - K r²/2` and
    /// `|∇n| <= |∇n(c)| + K r`. Near a line this is the centre's distance
    /// less the reach. Only smooth fBm has a curvature bound.
    fn taylor_line_bound(&self, x: Interval, z: Interval) -> Option<f64> {
        let curvature = self.base_curvature?;
        if self.fractal != Fractal::Fbm {
            return None;
        }
        let reach = x.radius().hypot(z.radius());
        let centre = [x.centre(), 0.0, z.centre()];
        let mut value = 0.0;
        let mut gradient = [0.0; 2];
        let mut bend = 0.0;
        let mut slack = 0.0;
        for (((noise, shift), frequency), weight) in self
            .octaves
            .iter()
            .zip(&self.shifts)
            .zip(&self.frequencies)
            .zip(&self.weights)
        {
            let step = GRADIENT_STEP / frequency;
            let at = |dx: f64, dz: f64| {
                self.raw_octave(
                    noise,
                    *shift,
                    *frequency,
                    [centre[0] + dx, 0.0, centre[2] + dz],
                )
            };
            value += weight * at(0.0, 0.0);
            gradient[0] += weight * (at(step, 0.0) - at(-step, 0.0)) / (2.0 * step);
            gradient[1] += weight * (at(0.0, step) - at(0.0, -step)) / (2.0 * step);
            bend += weight * curvature * frequency * frequency;
            // Rounding, plus the error of the one-sided differences that
            // [`Self::line_distance`] takes.
            slack += weight * 0.5f64.mul_add(curvature * GRADIENT_STEP, GRADIENT_SLACK) * frequency;
        }
        let amp = self.amp.abs();
        let value = amp.mul_add(value, self.offset).abs() - amp * LINE_VALUE_SLACK;
        let slope = amp * (gradient[0].hypot(gradient[1]) + slack);
        let bend = amp * bend;
        let lowest = (0.5 * bend * reach).mul_add(-reach, slope.mul_add(-reach, value));
        Some((lowest / bend.mul_add(reach, slope)).max(0.0))
    }

    /// The steepest gradient the noise can have over a box of columns: each
    /// octave's gradient at the centre plus its curvature bound times the
    /// reach, or its slope bound when that is tighter or unknown.
    fn slope_bound(&self, x: Interval, z: Interval) -> f64 {
        let fractal_slope = match self.fractal {
            Fractal::Fbm => 1.0,
            Fractal::Ridged | Fractal::Billow => 2.0,
        };
        let reach = x.radius().hypot(z.radius());
        let centre = [x.centre(), 0.0, z.centre()];
        let mut slope = 0.0;
        for (((noise, shift), frequency), weight) in self
            .octaves
            .iter()
            .zip(&self.shifts)
            .zip(&self.frequencies)
            .zip(&self.weights)
        {
            let global = self.base_slope * frequency;
            let local = self
                .base_curvature
                .filter(|_| reach.is_finite())
                .map(|curvature| {
                    let step = GRADIENT_STEP / frequency;
                    let gradient = |axis: usize| {
                        let mut ahead = centre;
                        let mut behind = centre;
                        ahead[axis] += step;
                        behind[axis] -= step;
                        (self.raw_octave(noise, *shift, *frequency, ahead)
                            - self.raw_octave(noise, *shift, *frequency, behind))
                            / (2.0 * GRADIENT_STEP)
                    };
                    // Per unit of noise coordinate, then back to metres.
                    let at_centre = gradient(0).hypot(gradient(2)) + GRADIENT_SLACK;
                    (curvature * reach).mul_add(*frequency, at_centre) * frequency
                });
            let octave = local.map_or(global, |local| local.min(global));
            slope += weight * octave * fractal_slope;
        }
        slope * self.amp.abs()
    }

    /// Second-order bound on one octave's raw value over a box, in [-1, 1]
    /// before the fractal shaping.
    fn taylor_octave(
        &self,
        (noise, shift, frequency): (&FastNoiseLite, [f64; 3], f64),
        centre: [f64; 3],
        radii: [f64; 3],
        raw: f64,
        curvature: f64,
    ) -> Interval {
        let step = GRADIENT_STEP / frequency;
        let axes: &[usize] = match self.dims {
            Dims::Two => &[0, 2],
            Dims::Three => &[0, 1, 2],
        };
        let mut linear = 0.0;
        let mut squared = 0.0;
        for &axis in axes {
            let mut ahead = centre;
            let mut behind = centre;
            ahead[axis] += step;
            behind[axis] -= step;
            let gradient = (self.raw_octave(noise, shift, frequency, ahead)
                - self.raw_octave(noise, shift, frequency, behind))
                / (2.0 * GRADIENT_STEP);
            let radius = radii[axis] * frequency;
            linear += (gradient.abs() + GRADIENT_SLACK) * radius;
            squared += radius * radius;
        }
        let spread = 0.5f64.mul_add(curvature * squared, linear) + STEP_SLACK;
        Interval::new(raw - spread, raw + spread)
    }

    /// The fractal shaping applied to a raw octave interval.
    fn shape_interval(&self, raw: Interval) -> Interval {
        match self.fractal {
            Fractal::Fbm => raw,
            Fractal::Ridged => {
                let magnitude = raw.abs();
                Interval::new(1.0 - 2.0 * magnitude.hi, 1.0 - 2.0 * magnitude.lo)
            }
            Fractal::Billow => {
                let magnitude = raw.abs();
                Interval::new(2.0 * magnitude.lo - 1.0, 2.0 * magnitude.hi - 1.0)
            }
        }
    }

    /// One octave's fractal-shaped value in [-1, 1].
    fn octave(
        &self,
        noise: &FastNoiseLite,
        shift: [f64; 3],
        frequency: f64,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        self.shape(self.raw_octave(noise, shift, frequency, [x, y, z]))
    }

    /// One octave's base noise in [-1, 1], before fractal shaping.
    fn raw_octave(
        &self,
        noise: &FastNoiseLite,
        shift: [f64; 3],
        frequency: f64,
        [x, y, z]: [f64; 3],
    ) -> f64 {
        let value = match self.dims {
            Dims::Two => noise.get_noise_2d(
                x.mul_add(frequency, shift[0]),
                z.mul_add(frequency, shift[2]),
            ),
            Dims::Three => noise.get_noise_3d(
                x.mul_add(frequency, shift[0]),
                y.mul_add(frequency, shift[1]),
                z.mul_add(frequency, shift[2]),
            ),
        };
        f64::from(value).clamp(-1.0, 1.0)
    }

    fn shape(&self, value: f64) -> f64 {
        match self.fractal {
            Fractal::Fbm => value,
            Fractal::Ridged => 1.0 - 2.0 * value.abs(),
            Fractal::Billow => 2.0 * value.abs() - 1.0,
        }
    }
}

#[cfg(test)]
mod tests;
