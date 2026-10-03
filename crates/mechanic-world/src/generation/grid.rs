//! Jittered grids: space cut into square cells in x and z, each holding at
//! most one instance placed by a hash of the cell, so any point finds the
//! instances that can reach it without global state. Scatter shapes and trees
//! are both placed this way.

use super::interval::Interval;
use super::scatter::mix;

/// Placement of instances on a jittered grid.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JitterGrid {
    /// Grid spacing in metres.
    pub(crate) cell: f64,
    /// Fraction of the cell an origin may wander, in `[0, 1]`.
    pub(crate) jitter: f64,
    /// Probability that a cell holds an instance.
    pub(crate) chance: f64,
    /// Decorrelates grids.
    pub(crate) seed: u64,
}

impl JitterGrid {
    /// The instance origin `(x, z)` of a cell with the cell's random stream
    /// for drawing the rest of the instance, or `None` when the cell is empty.
    #[expect(
        clippy::cast_precision_loss,
        reason = "cell indices stay far below 2^52 inside the finite world"
    )]
    pub(crate) fn place(&self, cell_x: i64, cell_z: i64) -> Option<(f64, f64, Hash)> {
        let mut random = Hash::new(self.seed, cell_x, cell_z);
        if random.unit() >= self.chance {
            return None;
        }
        let spread = self.jitter.clamp(0.0, 1.0);
        let x = (cell_x as f64 + 0.5 + (random.unit() - 0.5) * spread) * self.cell;
        let z = (cell_z as f64 + 0.5 + (random.unit() - 0.5) * spread) * self.cell;
        Some((x, z, random))
    }

    /// Inclusive cell ranges `(x0, x1, z0, z1)` whose instances may reach a
    /// box when they extend `reach` from their origins.
    pub(crate) fn cells_near(&self, x: Interval, z: Interval, reach: f64) -> (i64, i64, i64, i64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "coordinates are finite and inside the world"
        )]
        let cell = |value: f64| (value / self.cell).floor() as i64;
        (
            cell(x.lo - reach),
            cell(x.hi + reach),
            cell(z.lo - reach),
            cell(z.hi + reach),
        )
    }
}

/// Slot of a cell in a direct-mapped cache of `slots` entries.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "cell indices are hashed bit for bit"
)]
pub(crate) const fn cache_slot(id: u64, cell_x: i64, cell_z: i64, slots: usize) -> usize {
    (mix(id ^ (cell_x as u64).wrapping_mul(0x9e37_79b9) ^ (cell_z as u64) << 32) as usize) % slots
}

/// Deterministic per-cell random stream.
pub(crate) struct Hash(u64);

impl Hash {
    #[expect(clippy::cast_sign_loss, reason = "cell indices are hashed bit for bit")]
    pub(crate) const fn new(seed: u64, x: i64, z: i64) -> Self {
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        state = mix(state ^ (x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
        state = mix(state ^ (z as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
        Self(state)
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits map exactly onto the unit interval"
    )]
    pub(crate) fn unit(&mut self) -> f64 {
        self.0 = mix(self.0.wrapping_add(0x9e37_79b9_7f4a_7c15));
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    pub(crate) fn between(&mut self, lo: f64, hi: f64) -> f64 {
        (hi - lo).mul_add(self.unit(), lo)
    }
}
