use super::{NoiseGen, base_curvature, base_slope};
use crate::generation::interval::Interval;
use crate::generation::spec::{Dims, Fractal, NoiseDoc, NoiseKind};
use crate::generation::tape::Lane;

fn doc(kind: NoiseKind, dims: Dims) -> NoiseDoc {
    NoiseDoc {
        kind,
        fractal: Fractal::Fbm,
        octaves: 1,
        freq: 1.0,
        lacunarity: 2.0,
        gain: 0.5,
        amp: 1.0,
        offset: 0.0,
        dims,
        seed: 0,
    }
}

fn lines(freq: f64) -> NoiseGen {
    NoiseGen::new(
        &NoiseDoc {
            octaves: 2,
            freq,
            ..doc(NoiseKind::SimplexSmooth, Dims::Two)
        },
        5,
    )
}

/// A zero of the noise along x from `x`, by bisection.
fn zero_along_x(noise: &NoiseGen, x: f64, z: f64, reach: f64) -> Option<f64> {
    let steps = 400;
    let step = reach / f64::from(steps);
    let mut previous = (x, noise.sample(x, 0.0, z));
    for index in 1..=steps {
        let next = x + f64::from(index) * step;
        let value = noise.sample(next, 0.0, z);
        if (value > 0.0) != (previous.1 > 0.0) {
            let (mut lo, mut hi) = (previous.0, next);
            for _ in 0..60 {
                let middle = 0.5 * (lo + hi);
                if (noise.sample(middle, 0.0, z) > 0.0) == (previous.1 > 0.0) {
                    lo = middle;
                } else {
                    hi = middle;
                }
            }
            return Some(0.5 * (lo + hi));
        }
        previous = (next, value);
    }
    None
}

#[test]
fn line_distance_bounds_hold_every_sampled_distance() {
    for (freq, size) in [
        (0.011, 3.0),
        (0.0011, 14.0),
        (0.0011, 120.0),
        (0.000_09, 700.0),
    ] {
        let noise = lines(freq);
        for index in 0..200 {
            let t = f64::from(index);
            let (x, z) = ((t * 0.173) % 5.0 / freq, (t * 0.311) % 5.0 / freq);
            let bound = noise
                .line_distance_interval(Interval::new(x, x + size), Interval::new(z, z + size));
            for i in 0..=6 {
                for k in 0..=6 {
                    let distance = noise.line_distance(
                        x + size * f64::from(i) / 6.0,
                        z + size * f64::from(k) / 6.0,
                    );
                    assert!(
                        distance >= bound.lo,
                        "{distance} below {bound:?} at freq {freq}, box {size}"
                    );
                }
            }
        }
    }
}

#[test]
fn line_distance_measures_metres_to_the_line_at_every_scale() {
    // Half widths of 1 m, 30 m, and 500 m, each well inside its noise's
    // wavelength, as a trench, a ravine, and an ocean trench would be.
    for width in [1.0, 30.0, 500.0] {
        let freq = 0.02 / width;
        let noise = lines(freq);
        let mut measured = 0;
        for row in 0..20 {
            let z = f64::from(row) * 0.37 / freq;
            let Some(line) = zero_along_x(&noise, 0.0, z, 2.0 / freq) else {
                continue;
            };
            // Walk out across the line on both sides to where the
            // distance reaches the width: the walls of a cut
            // `width - distance`. Curvature shifts both walls the same
            // way, so their separation is what holds.
            let step = 1.0e-3 / freq;
            let across = [
                noise.sample(line + step, 0.0, z) - noise.sample(line - step, 0.0, z),
                noise.sample(line, 0.0, z + step) - noise.sample(line, 0.0, z - step),
            ];
            let length = across[0].hypot(across[1]);
            let across = [across[0] / length, across[1] / length];
            let wall = |direction: f64| {
                let mut offset = 0.0;
                while noise.line_distance(
                    line + direction * offset * across[0],
                    z + direction * offset * across[1],
                ) < width
                {
                    offset += width * 0.002;
                }
                offset
            };
            let separation = wall(1.0) + wall(-1.0);
            assert!(
                (separation / (2.0 * width) - 1.0).abs() < 0.1,
                "walls {separation} apart at width {width}"
            );
            measured += 1;
        }
        assert!(measured > 5, "found {measured} lines at width {width}");
    }
}

#[test]
fn base_noise_slope_stays_below_the_declared_bound() {
    let step = 1.0e-4;
    for kind in [
        NoiseKind::Simplex,
        NoiseKind::SimplexSmooth,
        NoiseKind::Perlin,
        NoiseKind::Value,
        NoiseKind::Cells,
        NoiseKind::CellEdges,
    ] {
        for dims in [Dims::Two, Dims::Three] {
            let noise = NoiseGen::new(&doc(kind, dims), 17);
            let mut steepest = 0.0_f64;
            for index in 0..40_000 {
                let t = f64::from(index);
                let (x, y, z) = (
                    (t * 0.137_1) % 400.0,
                    (t * 0.071_3).sin() * 40.0,
                    (t * 0.093_7) % 400.0,
                );
                let centre = noise.sample(x, y, z);
                let gradient = [
                    noise.sample(x + step, y, z) - centre,
                    noise.sample(x, y + step, z) - centre,
                    noise.sample(x, y, z + step) - centre,
                ];
                let slope = gradient
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
                    / step;
                // Cell borders are kinks; a finite difference straddling one
                // is not a slope.
                if slope < 50.0 {
                    steepest = steepest.max(slope);
                }
            }
            assert!(
                steepest < base_slope(kind, dims),
                "{kind:?} {dims:?} slope {steepest}"
            );
        }
    }
}

#[test]
fn base_noise_curvature_stays_below_the_declared_bound() {
    let step = 2.0e-3;
    for kind in [
        NoiseKind::Simplex,
        NoiseKind::SimplexSmooth,
        NoiseKind::Perlin,
        NoiseKind::Value,
    ] {
        for dims in [Dims::Two, Dims::Three] {
            let noise = NoiseGen::new(&doc(kind, dims), 17);
            let bound = base_curvature(kind, dims).expect("smooth noise has a curvature bound");
            let mut sharpest = 0.0_f64;
            for index in 0..100_000 {
                let t = f64::from(index);
                let (x, y, z) = (
                    (t * 0.137_1) % 400.0,
                    (t * 0.071_3).sin() * 40.0,
                    (t * 0.093_7) % 400.0,
                );
                let vertical = if matches!(dims, Dims::Two) {
                    0.0
                } else {
                    (t * 2.3).cos()
                };
                let direction = [(t * 1.7).sin(), vertical, (t * 0.9).cos()];
                let length = direction
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                let [dx, dy, dz] = direction.map(|value| value / length * step);
                let second = (noise.sample(x + dx, y + dy, z + dz) - 2.0 * noise.sample(x, y, z)
                    + noise.sample(x - dx, y - dy, z - dz))
                    / (step * step);
                // Float steps in the base noise are not curvature.
                if second.abs() < 1.0e3 {
                    sharpest = sharpest.max(second.abs());
                }
            }
            assert!(sharpest < bound, "{kind:?} {dims:?} curvature {sharpest}");
        }
    }
}

#[test]
fn taylor_bounds_contain_sampled_values_of_wide_octaves() {
    for kind in [
        NoiseKind::Simplex,
        NoiseKind::SimplexSmooth,
        NoiseKind::Value,
    ] {
        for dims in [Dims::Two, Dims::Three] {
            for fractal in [Fractal::Fbm, Fractal::Ridged, Fractal::Billow] {
                let mut wide = doc(kind, dims);
                wide.freq = 0.004;
                wide.octaves = 5;
                wide.fractal = fractal;
                wide.amp = 90.0;
                let noise = NoiseGen::new(&wide, 29);
                for (index, radius) in [2.0, 9.0, 30.0, 70.0].into_iter().enumerate() {
                    let centre = 173.0 * f64::from(u32::try_from(index).unwrap_or(0)) - 211.0;
                    let bounds = Interval::new(centre - radius, centre + radius);
                    let interval = noise.interval(bounds, bounds, bounds);
                    for a in 0..=12 {
                        for b in 0..=12 {
                            let at =
                                |step: i32| centre - radius + 2.0 * radius * f64::from(step) / 12.0;
                            let value = noise.sample(at(a), at(b), at(12 - a));
                            assert!(
                                interval.lo <= value && value <= interval.hi,
                                "{kind:?} {dims:?} {fractal:?} {value} outside {interval:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn noise_intervals_contain_sampled_values() {
    let mut fractal = doc(NoiseKind::Simplex, Dims::Three);
    fractal.octaves = 4;
    fractal.freq = 0.05;
    fractal.fractal = Fractal::Ridged;
    fractal.amp = 7.0;
    let noise = NoiseGen::new(&fractal, 3);
    for (centre, radius) in [(0.0, 1.0), (40.0, 3.0), (-120.0, 0.25), (9.0, 20.0)] {
        let bounds = Interval::new(centre - radius, centre + radius);
        let interval = noise.interval(bounds, bounds, bounds);
        for index in 0..=20 {
            let t = centre - radius + 2.0 * radius * f64::from(index) / 20.0;
            let value = noise.sample(t, centre + radius - (t - centre).abs(), t);
            assert!(interval.lo <= value && value <= interval.hi);
        }
    }
}

#[test]
fn many_points_sample_exactly_as_one_at_a_time() {
    let xs: Vec<f64> = (0..203)
        .map(|index| (f64::from(index) * 0.731).sin() * 400.0 - 17.0)
        .collect();
    let ys: Vec<f64> = (0..203)
        .map(|index| (f64::from(index) * 1.37).cos() * 60.0)
        .collect();
    let zs: Vec<f64> = (0..203)
        .map(|index| (f64::from(index) * 0.291).sin() * 900.0 + 3.0)
        .collect();
    for kind in [
        NoiseKind::Simplex,
        NoiseKind::SimplexSmooth,
        NoiseKind::Perlin,
        NoiseKind::Value,
        NoiseKind::Cells,
        NoiseKind::CellEdges,
    ] {
        for dims in [Dims::Two, Dims::Three] {
            for fractal in [Fractal::Fbm, Fractal::Ridged, Fractal::Billow] {
                let noise = NoiseGen::new(
                    &NoiseDoc {
                        fractal,
                        octaves: 3,
                        freq: 0.013,
                        amp: 7.5,
                        offset: -1.25,
                        ..doc(kind, dims)
                    },
                    41,
                );
                for (count, y) in [(203, Lane::Row(&ys)), (5, Lane::Splat(2.5))] {
                    let mut many = vec![1.0];
                    noise.sample_many([Lane::Row(&xs), y, Lane::Row(&zs)], count, &mut many);
                    assert_eq!(many.len(), count + 1);
                    for index in 0..count {
                        let y = if dims == Dims::Two { 0.0 } else { y.at(index) };
                        let one = noise.sample(xs[index], y, zs[index]);
                        assert_eq!(
                            many[index + 1].to_bits(),
                            one.to_bits(),
                            "{kind:?} {dims:?} {fractal:?} point {index}"
                        );
                    }
                }
            }
        }
    }
}
