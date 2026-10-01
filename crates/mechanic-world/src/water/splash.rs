//! Where falling water lands, for drawing: the power of every fall landing
//! on a column, which churns the water there white. It is shown, never
//! saved, and fades soon after a fall stops.

use mechanic_core::WATER_DENSITY_KG_M3;

use super::{GRAVITY, WaterWorld};

/// Seconds over which a column's splash halves once its fall stops.
const HALF_LIFE_SECONDS: f64 = 1.5;

/// A splash weaker than this, in watts, is forgotten.
const FADED_WATTS: f64 = 0.1;

/// Power of a fall that churns the water it lands in about two-thirds
/// white, in watts: a breach of 10 L/s over a metre is 100 W.
const CHURNING_WATTS: f64 = 20.0;

impl WaterWorld {
    /// Books `volume` falling `drop` metres onto a column during this step.
    pub(super) fn splash(&mut self, column: (i32, i32), volume: f64, drop: f64) {
        if volume <= 0.0 || drop <= 0.0 {
            return;
        }
        *self.landed.entry(column).or_default() += WATER_DENSITY_KG_M3 * GRAVITY * volume * drop;
    }

    /// Folds the step's falls into each column's splash, a running mean of
    /// the power landing on it.
    pub(super) fn settle_splashes(&mut self, dt: f64) {
        let kept = 0.5_f64.powf(dt / HALF_LIFE_SECONDS);
        for power in self.splashes.values_mut() {
            *power *= kept;
        }
        for (column, energy) in self.landed.drain() {
            *self.splashes.entry(column).or_default() += energy / dt * (1.0 - kept);
        }
        self.splashes.retain(|_, power| *power >= FADED_WATTS);
    }

    /// How white the falls landing on a column churn its water, from 0 to
    /// 1.
    pub(super) fn churn_at(&self, column: (i32, i32)) -> f64 {
        self.splashes
            .get(&column)
            .map_or(0.0, |power| 1.0 - (-power / CHURNING_WATTS).exp())
    }
}
