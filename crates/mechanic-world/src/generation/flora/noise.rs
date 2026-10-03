//! Smooth noise with a uniform distribution, so a foliage density of 0.4
//! fills 40 % of a sleeve.

use std::sync::OnceLock;

use bevy_math::DVec3;

use super::super::scatter::mix;
use super::super::tape::smoothstep;
use super::FOLIAGE_NOISE_SCALE;

/// Quantiles of the raw lattice noise, measured once.
const QUANTILES: usize = 256;

/// Lattice noise at `point`, remapped through its own distribution so the
/// result is close to uniform on `[0, 1]`.
pub(super) fn foliage_noise(seed: u64, point: DVec3) -> f64 {
    let raw = lattice_noise(seed, point / FOLIAGE_NOISE_SCALE);
    let table = quantiles();
    let above = table.partition_point(|&quantile| quantile < raw);
    if above == 0 {
        return 0.0;
    }
    if above == table.len() {
        return 1.0;
    }
    let (low, high) = (table[above - 1], table[above]);
    let within = if high > low {
        (raw - low) / (high - low)
    } else {
        0.0
    };
    #[expect(
        clippy::cast_precision_loss,
        reason = "a few hundred quantiles are exact"
    )]
    let rank = (above as f64 - 1.0 + within) / (table.len() - 1) as f64;
    rank
}

fn quantiles() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        const SAMPLES: u64 = 1 << 16;
        let mut values = (0..SAMPLES)
            .map(|index| {
                let unit = |salt: u64| unit(mix(index.wrapping_mul(0x9e37_79b9) ^ salt));
                let point = DVec3::new(unit(1), unit(2), unit(3)) * 997.0;
                lattice_noise(0x5eed, point)
            })
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        (0..=QUANTILES)
            .map(|step| values[(step * (values.len() - 1)) / QUANTILES])
            .collect()
    })
}

/// Trilinear value noise with smoothstep weights on a unit lattice.
#[expect(
    clippy::cast_possible_truncation,
    reason = "lattice indices of finite coordinates"
)]
fn lattice_noise(seed: u64, point: DVec3) -> f64 {
    let base = point.floor();
    let fraction = point - base;
    let weights = [
        smoothstep(0.0, 1.0, fraction.x),
        smoothstep(0.0, 1.0, fraction.y),
        smoothstep(0.0, 1.0, fraction.z),
    ];
    let [x, y, z] = [base.x as i64, base.y as i64, base.z as i64];
    let mut total = 0.0;
    for corner in 0..8_i64 {
        let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let weight = [dx, dy, dz]
            .iter()
            .zip(weights)
            .map(|(&step, weight)| if step == 1 { weight } else { 1.0 - weight })
            .product::<f64>();
        total += weight * lattice_value(seed, x + dx, y + dy, z + dz);
    }
    total
}

#[expect(
    clippy::cast_sign_loss,
    reason = "lattice indices are hashed bit for bit"
)]
fn lattice_value(seed: u64, x: i64, y: i64, z: i64) -> f64 {
    let mut state = mix(seed ^ (x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    state = mix(state ^ (y as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    state = mix(state ^ (z as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    unit(state)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "53 random bits map exactly onto the unit interval"
)]
fn unit(bits: u64) -> f64 {
    (bits >> 11) as f64 / (1_u64 << 53) as f64
}
