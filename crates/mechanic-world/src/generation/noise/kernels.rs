//! `FastNoiseLite`'s hot generators over lanes of points.
//!
//! Each kernel repeats, statement for statement, the single-point code of
//! `fastnoise-lite` 1.1.1 with the `f64` feature, so every lane rounds
//! exactly as one lookup would; only the order in which independent points
//! are worked through changes. Points are taken [`LANES`] at a time with
//! each step written as a loop over the lanes, so the compiler keeps them in
//! vector registers. `kernels_match_fastnoise_lite_bit_for_bit` pins them
//! to the library.

#![expect(
    clippy::excessive_precision,
    clippy::many_single_char_names,
    reason = "the library's constants and names, kept verbatim to read against it"
)]

/// Points per kernel step.
pub(super) const LANES: usize = 8;

/// The kinds of generator with a lane kernel. Everything else is looked up
/// one point at a time through the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kernel {
    /// 3D Perlin with the `ImproveXZPlanes` rotation.
    Perlin3,
    /// 2D `OpenSimplex2`.
    Simplex2,
    /// 2D `OpenSimplex2S`.
    SimplexSmooth2,
}

const PRIME_X: i32 = 501_125_321;
const PRIME_Y: i32 = 1_136_930_381;
const PRIME_Z: i32 = 1_720_413_743;

type Lanes<T> = [T; LANES];

impl Kernel {
    /// Raw values at `LANES` points with frequency one; `z` is ignored by
    /// the 2D kernels.
    pub(super) fn run(
        self,
        seed: i32,
        x: &Lanes<f64>,
        y: &Lanes<f64>,
        z: &Lanes<f64>,
    ) -> Lanes<f32> {
        match self {
            Self::Perlin3 => perlin_3d(seed, *x, *y, *z),
            Self::Simplex2 => simplex_2d(seed, *x, *y),
            Self::SimplexSmooth2 => simplex_smooth_2d(seed, *x, *y),
        }
    }
}

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the library's own conversion"
)]
fn fast_floor(value: f64) -> i32 {
    if value >= 0. {
        value as i32
    } else {
        (value as i32).wrapping_sub(1)
    }
}

#[inline]
fn fraction(value: f64, floor: i32) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the library's own conversion"
    )]
    let fraction = (value - f64::from(floor)) as f32;
    fraction
}

#[inline]
fn interp_quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6. - 15.) + 10.)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline]
fn gradient_index(hash: i32, mask: i32) -> usize {
    let hash = hash.wrapping_mul(0x27d4_eb2d);
    let hash = hash ^ (hash >> 15);
    #[expect(clippy::cast_sign_loss, reason = "masked to a small index")]
    let index = (hash & mask) as usize;
    index
}

#[inline]
fn grad_3d(seed: i32, primed: [&Lanes<i32>; 3], offset: [&Lanes<f32>; 3]) -> Lanes<f32> {
    let mut index = [0; LANES];
    for lane in 0..LANES {
        index[lane] = gradient_index(
            seed ^ primed[0][lane] ^ primed[1][lane] ^ primed[2][lane],
            63 << 2,
        );
    }
    let mut value = [0.0; LANES];
    for lane in 0..LANES {
        let at = index[lane];
        value[lane] = offset[0][lane] * GRADIENTS_3D[at]
            + offset[1][lane] * GRADIENTS_3D[at | 1]
            + offset[2][lane] * GRADIENTS_3D[at | 2];
    }
    value
}

#[inline]
fn grad_2d(
    seed: i32,
    x_primed: &Lanes<i32>,
    y_primed: &Lanes<i32>,
    x: &Lanes<f32>,
    y: &Lanes<f32>,
) -> Lanes<f32> {
    let mut index = [0; LANES];
    for lane in 0..LANES {
        index[lane] = gradient_index(seed ^ x_primed[lane] ^ y_primed[lane], 127 << 1);
    }
    let mut value = [0.0; LANES];
    for lane in 0..LANES {
        let at = index[lane];
        value[lane] = x[lane] * GRADIENTS_2D[at] + y[lane] * GRADIENTS_2D[at | 1];
    }
    value
}

fn perlin_3d(seed: i32, mut x: Lanes<f64>, mut y: Lanes<f64>, mut z: Lanes<f64>) -> Lanes<f32> {
    for lane in 0..LANES {
        let xz = x[lane] + z[lane];
        let s2 = xz * -0.211_324_865_405_187;
        y[lane] *= 0.577_350_269_189_626;
        x[lane] += s2 - y[lane];
        z[lane] += s2 - y[lane];
        y[lane] += xz * 0.577_350_269_189_626;
    }
    let mut x0 = [0; LANES];
    let mut y0 = [0; LANES];
    let mut z0 = [0; LANES];
    let mut xd0 = [0.0; LANES];
    let mut yd0 = [0.0; LANES];
    let mut zd0 = [0.0; LANES];
    for lane in 0..LANES {
        x0[lane] = fast_floor(x[lane]);
        y0[lane] = fast_floor(y[lane]);
        z0[lane] = fast_floor(z[lane]);
        xd0[lane] = fraction(x[lane], x0[lane]);
        yd0[lane] = fraction(y[lane], y0[lane]);
        zd0[lane] = fraction(z[lane], z0[lane]);
    }
    let mut xd1 = [0.0; LANES];
    let mut yd1 = [0.0; LANES];
    let mut zd1 = [0.0; LANES];
    let mut xs = [0.0; LANES];
    let mut ys = [0.0; LANES];
    let mut zs = [0.0; LANES];
    let mut x1 = [0; LANES];
    let mut y1 = [0; LANES];
    let mut z1 = [0; LANES];
    for lane in 0..LANES {
        xd1[lane] = xd0[lane] - 1.;
        yd1[lane] = yd0[lane] - 1.;
        zd1[lane] = zd0[lane] - 1.;
        xs[lane] = interp_quintic(xd0[lane]);
        ys[lane] = interp_quintic(yd0[lane]);
        zs[lane] = interp_quintic(zd0[lane]);
        x0[lane] = x0[lane].wrapping_mul(PRIME_X);
        y0[lane] = y0[lane].wrapping_mul(PRIME_Y);
        z0[lane] = z0[lane].wrapping_mul(PRIME_Z);
        x1[lane] = x0[lane].wrapping_add(PRIME_X);
        y1[lane] = y0[lane].wrapping_add(PRIME_Y);
        z1[lane] = z0[lane].wrapping_add(PRIME_Z);
    }
    let g000 = grad_3d(seed, [&x0, &y0, &z0], [&xd0, &yd0, &zd0]);
    let g100 = grad_3d(seed, [&x1, &y0, &z0], [&xd1, &yd0, &zd0]);
    let g010 = grad_3d(seed, [&x0, &y1, &z0], [&xd0, &yd1, &zd0]);
    let g110 = grad_3d(seed, [&x1, &y1, &z0], [&xd1, &yd1, &zd0]);
    let g001 = grad_3d(seed, [&x0, &y0, &z1], [&xd0, &yd0, &zd1]);
    let g101 = grad_3d(seed, [&x1, &y0, &z1], [&xd1, &yd0, &zd1]);
    let g011 = grad_3d(seed, [&x0, &y1, &z1], [&xd0, &yd1, &zd1]);
    let g111 = grad_3d(seed, [&x1, &y1, &z1], [&xd1, &yd1, &zd1]);
    let mut value = [0.0; LANES];
    for lane in 0..LANES {
        let xf00 = lerp(g000[lane], g100[lane], xs[lane]);
        let xf10 = lerp(g010[lane], g110[lane], xs[lane]);
        let xf01 = lerp(g001[lane], g101[lane], xs[lane]);
        let xf11 = lerp(g011[lane], g111[lane], xs[lane]);
        let yf0 = lerp(xf00, xf10, ys[lane]);
        let yf1 = lerp(xf01, xf11, ys[lane]);
        value[lane] = lerp(yf0, yf1, zs[lane]) * 0.964_921_414_852_142_333_984_375;
    }
    value
}

/// The library's 2D skew for the simplex kinds, then the cell each point
/// falls in: primed corner, offsets within the cell.
#[inline]
fn simplex_cells(
    mut x: Lanes<f64>,
    mut y: Lanes<f64>,
) -> (Lanes<i32>, Lanes<i32>, Lanes<f32>, Lanes<f32>) {
    let sqrt3 = 1.732_050_807_568_877_293_527_446_341_505_9;
    let f2 = 0.5 * (sqrt3 - 1.);
    let mut i = [0; LANES];
    let mut j = [0; LANES];
    let mut xi = [0.0; LANES];
    let mut yi = [0.0; LANES];
    for lane in 0..LANES {
        let t = (x[lane] + y[lane]) * f2;
        x[lane] += t;
        y[lane] += t;
        i[lane] = fast_floor(x[lane]);
        j[lane] = fast_floor(y[lane]);
        xi[lane] = fraction(x[lane], i[lane]);
        yi[lane] = fraction(y[lane], j[lane]);
        i[lane] = i[lane].wrapping_mul(PRIME_X);
        j[lane] = j[lane].wrapping_mul(PRIME_Y);
    }
    (i, j, xi, yi)
}

#[expect(clippy::suboptimal_flops, reason = "the library's own arithmetic")]
fn simplex_2d(seed: i32, x: Lanes<f64>, y: Lanes<f64>) -> Lanes<f32> {
    let sqrt3: f32 = 1.732_050_807_568_877_293_527_446_341_505_9;
    let g2 = (3. - sqrt3) / 6.;
    let (i, j, xi, yi) = simplex_cells(x, y);
    let mut t = [0.0; LANES];
    let mut x0 = [0.0; LANES];
    let mut y0 = [0.0; LANES];
    let mut i2 = [0; LANES];
    let mut j2 = [0; LANES];
    let mut x2 = [0.0; LANES];
    let mut y2 = [0.0; LANES];
    let mut i1 = [0; LANES];
    let mut j1 = [0; LANES];
    let mut x1 = [0.0; LANES];
    let mut y1 = [0.0; LANES];
    for lane in 0..LANES {
        t[lane] = (xi[lane] + yi[lane]) * g2;
        x0[lane] = xi[lane] - t[lane];
        y0[lane] = yi[lane] - t[lane];
        i2[lane] = i[lane].wrapping_add(PRIME_X);
        j2[lane] = j[lane].wrapping_add(PRIME_Y);
        x2[lane] = x0[lane] + (2. * g2 - 1.);
        y2[lane] = y0[lane] + (2. * g2 - 1.);
        // The middle corner: the library's two branches, selected.
        if y0[lane] > x0[lane] {
            x1[lane] = x0[lane] + g2;
            y1[lane] = y0[lane] + (g2 - 1.);
            i1[lane] = i[lane];
            j1[lane] = j[lane].wrapping_add(PRIME_Y);
        } else {
            x1[lane] = x0[lane] + (g2 - 1.);
            y1[lane] = y0[lane] + g2;
            i1[lane] = i[lane].wrapping_add(PRIME_X);
            j1[lane] = j[lane];
        }
    }
    let g0 = grad_2d(seed, &i, &j, &x0, &y0);
    let g1 = grad_2d(seed, &i1, &j1, &x1, &y1);
    let g2_corner = grad_2d(seed, &i2, &j2, &x2, &y2);
    let mut value = [0.0; LANES];
    for lane in 0..LANES {
        let a = 0.5 - x0[lane] * x0[lane] - y0[lane] * y0[lane];
        let n0 = if a <= 0. {
            0.
        } else {
            (a * a) * (a * a) * g0[lane]
        };
        let c = (2. * (1. - 2. * g2) * (1. / g2 - 2.)) * t[lane]
            + ((-2. * (1. - 2. * g2) * (1. - 2. * g2)) + a);
        let n2 = if c <= 0. {
            0.
        } else {
            (c * c) * (c * c) * g2_corner[lane]
        };
        let b = 0.5 - x1[lane] * x1[lane] - y1[lane] * y1[lane];
        let n1 = if b <= 0. {
            0.
        } else {
            (b * b) * (b * b) * g1[lane]
        };
        value[lane] = (n0 + n1 + n2) * 99.836_854_463_036_47;
    }
    value
}

#[expect(clippy::suboptimal_flops, reason = "the library's own arithmetic")]
#[expect(clippy::too_many_lines, reason = "the library's corners, one by one")]
fn simplex_smooth_2d(seed: i32, x: Lanes<f64>, y: Lanes<f64>) -> Lanes<f32> {
    let sqrt3: f32 = 1.732_050_807_568_877_293_527_446_341_505_9;
    let g2 = (3. - sqrt3) / 6.;
    let (i, j, xi, yi) = simplex_cells(x, y);
    let mut i1 = [0; LANES];
    let mut j1 = [0; LANES];
    let mut t = [0.0; LANES];
    let mut x0 = [0.0; LANES];
    let mut y0 = [0.0; LANES];
    let mut x1 = [0.0; LANES];
    let mut y1 = [0.0; LANES];
    // The two corners the library picks by branching, as selected lanes.
    let mut ia = [0; LANES];
    let mut ja = [0; LANES];
    let mut xa = [0.0; LANES];
    let mut ya = [0.0; LANES];
    let mut ib = [0; LANES];
    let mut jb = [0; LANES];
    let mut xb = [0.0; LANES];
    let mut yb = [0.0; LANES];
    for lane in 0..LANES {
        let (i, j, xi, yi) = (i[lane], j[lane], xi[lane], yi[lane]);
        i1[lane] = i.wrapping_add(PRIME_X);
        j1[lane] = j.wrapping_add(PRIME_Y);
        let t_lane = (xi + yi) * g2;
        let x0_lane = xi - t_lane;
        let y0_lane = yi - t_lane;
        t[lane] = t_lane;
        x0[lane] = x0_lane;
        y0[lane] = y0_lane;
        x1[lane] = x0_lane - (1. - 2. * g2);
        y1[lane] = y0_lane - (1. - 2. * g2);
        let xmyi = xi - yi;
        let corner_a;
        let corner_b;
        if t_lane > g2 {
            corner_a = if xi + xmyi > 1. {
                (
                    x0_lane + (3. * g2 - 2.),
                    y0_lane + (3. * g2 - 1.),
                    i.wrapping_add(PRIME_X << 1),
                    j.wrapping_add(PRIME_Y),
                )
            } else {
                (
                    x0_lane + g2,
                    y0_lane + (g2 - 1.),
                    i,
                    j.wrapping_add(PRIME_Y),
                )
            };
            corner_b = if yi - xmyi > 1. {
                (
                    x0_lane + (3. * g2 - 1.),
                    y0_lane + (3. * g2 - 2.),
                    i.wrapping_add(PRIME_X),
                    j.wrapping_add(PRIME_Y << 1),
                )
            } else {
                (
                    x0_lane + (g2 - 1.),
                    y0_lane + g2,
                    i.wrapping_add(PRIME_X),
                    j,
                )
            };
        } else {
            corner_a = if xi + xmyi < 0. {
                (
                    x0_lane + (1. - g2),
                    y0_lane - g2,
                    i.wrapping_sub(PRIME_X),
                    j,
                )
            } else {
                (
                    x0_lane + (g2 - 1.),
                    y0_lane + g2,
                    i.wrapping_add(PRIME_X),
                    j,
                )
            };
            corner_b = if yi < xmyi {
                (
                    x0_lane - g2,
                    y0_lane - (g2 - 1.),
                    i,
                    j.wrapping_sub(PRIME_Y),
                )
            } else {
                (
                    x0_lane + g2,
                    y0_lane + (g2 - 1.),
                    i,
                    j.wrapping_add(PRIME_Y),
                )
            };
        }
        (xa[lane], ya[lane], ia[lane], ja[lane]) = corner_a;
        (xb[lane], yb[lane], ib[lane], jb[lane]) = corner_b;
    }
    let g0 = grad_2d(seed, &i, &j, &x0, &y0);
    let g1 = grad_2d(seed, &i1, &j1, &x1, &y1);
    let ga = grad_2d(seed, &ia, &ja, &xa, &ya);
    let gb = grad_2d(seed, &ib, &jb, &xb, &yb);
    let mut value = [0.0; LANES];
    for lane in 0..LANES {
        let a0 = (2. / 3.) - x0[lane] * x0[lane] - y0[lane] * y0[lane];
        let mut sum = (a0 * a0) * (a0 * a0) * g0[lane];
        let a1 = (2. * (1. - 2. * g2) * (1. / g2 - 2.)) * t[lane]
            + ((-2. * (1. - 2. * g2) * (1. - 2. * g2)) + a0);
        sum += (a1 * a1) * (a1 * a1) * g1[lane];
        let a2 = (2. / 3.) - xa[lane] * xa[lane] - ya[lane] * ya[lane];
        if a2 > 0. {
            sum += (a2 * a2) * (a2 * a2) * ga[lane];
        }
        let a3 = (2. / 3.) - xb[lane] * xb[lane] - yb[lane] * yb[lane];
        if a3 > 0. {
            sum += (a3 * a3) * (a3 * a3) * gb[lane];
        }
        value[lane] = sum * 18.241_961_944_860_65;
    }
    value
}

/// `FastNoiseLite::GRADIENTS_2D`.
#[expect(
    clippy::unreadable_literal,
    reason = "the library's table, digit for digit"
)]
static GRADIENTS_2D: [f32; 256] = [
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.923879532511287,
    0.38268343236509,
    0.923879532511287,
    -0.38268343236509,
    0.38268343236509,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    0.38268343236509,
    -0.38268343236509,
    0.923879532511287,
];

/// `FastNoiseLite::GRADIENTS_3D`.
static GRADIENTS_3D: [f32; 256] = [
    0., 1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 1., 0., 1., 0., -1., 0.,
    1., 0., 1., 0., -1., 0., -1., 0., -1., 0., 1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0.,
    -1., -1., 0., 0., 0., 1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 1., 0.,
    1., 0., -1., 0., 1., 0., 1., 0., -1., 0., -1., 0., -1., 0., 1., 1., 0., 0., -1., 1., 0., 0.,
    1., -1., 0., 0., -1., -1., 0., 0., 0., 1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0., -1.,
    -1., 0., 1., 0., 1., 0., -1., 0., 1., 0., 1., 0., -1., 0., -1., 0., -1., 0., 1., 1., 0., 0.,
    -1., 1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 0., 0., 1., 1., 0., 0., -1., 1., 0., 0., 1.,
    -1., 0., 0., -1., -1., 0., 1., 0., 1., 0., -1., 0., 1., 0., 1., 0., -1., 0., -1., 0., -1., 0.,
    1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 0., 0., 1., 1., 0., 0., -1.,
    1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 1., 0., 1., 0., -1., 0., 1., 0., 1., 0., -1., 0.,
    -1., 0., -1., 0., 1., 1., 0., 0., -1., 1., 0., 0., 1., -1., 0., 0., -1., -1., 0., 0., 1., 1.,
    0., 0., 0., -1., 1., 0., -1., 1., 0., 0., 0., -1., -1., 0.,
];

#[cfg(test)]
mod tests {
    use fastnoise_lite::{FastNoiseLite, NoiseType, RotationType3D};

    use super::{Kernel, LANES};

    /// Coordinates that probe the library's edge cases: lattice lines,
    /// negative cells, the skew's boundaries, and far-off positions.
    fn probes() -> Vec<[f64; 3]> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit
        };
        let mut points = Vec::new();
        for scale in [1.0, 37.0, 1.0e3, 2.0e5] {
            for _ in 0..4_000 {
                points.push([0, 1, 2].map(|_| (unit() * 2.0 - 1.0) * scale));
            }
        }
        for whole in -4..=4 {
            let whole = f64::from(whole);
            for nudge in [0.0, 1.0e-12, -1.0e-12, 0.5, -0.5, 1.0e-7] {
                points.push([whole + nudge, -whole, whole * 0.5 - nudge]);
                points.push([whole, whole + nudge, -whole - nudge]);
            }
        }
        points
    }

    fn library(kernel: Kernel, seed: i32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_frequency(Some(1.0));
        match kernel {
            Kernel::Perlin3 => {
                noise.set_noise_type(Some(NoiseType::Perlin));
                noise.set_rotation_type_3d(Some(RotationType3D::ImproveXZPlanes));
            }
            Kernel::Simplex2 => noise.set_noise_type(Some(NoiseType::OpenSimplex2)),
            Kernel::SimplexSmooth2 => noise.set_noise_type(Some(NoiseType::OpenSimplex2S)),
        }
        noise
    }

    #[test]
    fn kernels_match_fastnoise_lite_bit_for_bit() {
        let points = probes();
        for kernel in [Kernel::Perlin3, Kernel::Simplex2, Kernel::SimplexSmooth2] {
            for seed in [0, 1_337, -91_234_567, i32::MAX] {
                let noise = library(kernel, seed);
                for chunk in points.chunks(LANES) {
                    let mut lanes = [[0.0; LANES]; 3];
                    for (lane, point) in chunk.iter().enumerate() {
                        for axis in 0..3 {
                            lanes[axis][lane] = point[axis];
                        }
                    }
                    let values = kernel.run(seed, &lanes[0], &lanes[1], &lanes[2]);
                    for (lane, point) in chunk.iter().enumerate() {
                        let expected = match kernel {
                            Kernel::Perlin3 => noise.get_noise_3d(point[0], point[1], point[2]),
                            Kernel::Simplex2 | Kernel::SimplexSmooth2 => {
                                noise.get_noise_2d(point[0], point[1])
                            }
                        };
                        assert_eq!(
                            values[lane].to_bits(),
                            expected.to_bits(),
                            "{kernel:?} seed {seed} at {point:?}"
                        );
                    }
                }
            }
        }
    }
}
