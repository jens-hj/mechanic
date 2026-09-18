//! The fixed-step physics clock: turns frame time into whole ticks due.

use std::time::Duration;

use mechanic_core::TICK_RATE_HZ;

const PHASE_DENOMINATOR: u128 = 1_000_000_000;

/// Consecutive fixed ticks due after advancing the independent physics clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledTicks {
    first: u64,
    count: u64,
}

impl ScheduledTicks {
    /// Number of fixed ticks due. Backlog is never silently dropped.
    pub const fn count(self) -> u64 {
        self.count
    }
}

impl IntoIterator for ScheduledTicks {
    type Item = u64;
    type IntoIter = std::ops::Range<u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.first..self.first.saturating_add(self.count)
    }
}

/// Rational 60 Hz clock that accumulates independently of render cadence.
#[derive(Clone, Debug, Default)]
pub struct FixedStepScheduler {
    phase: u128,
    next_tick: u64,
}

impl FixedStepScheduler {
    /// Creates a scheduler whose first emitted tick is one.
    pub const fn new() -> Self {
        Self {
            phase: 0,
            next_tick: 1,
        }
    }

    /// Adds real elapsed time and returns every due fixed tick without a catch-up cap.
    pub fn advance(&mut self, elapsed: Duration) -> ScheduledTicks {
        self.phase = self
            .phase
            .saturating_add(elapsed.as_nanos().saturating_mul(u128::from(TICK_RATE_HZ)));
        let due = self.phase / PHASE_DENOMINATOR;
        self.phase %= PHASE_DENOMINATOR;
        let count = u64::try_from(due).unwrap_or(u64::MAX);
        let first = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(count);
        ScheduledTicks { first, count }
    }

    /// Next monotonic tick index.
    #[cfg(test)]
    pub const fn next_tick(&self) -> u64 {
        self.next_tick
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::FixedStepScheduler;

    #[test]
    fn one_second_always_produces_exactly_sixty_ticks() {
        let mut scheduler = FixedStepScheduler::new();
        let ticks = scheduler.advance(Duration::from_secs(1));
        assert_eq!(ticks.count(), 60);
        assert_eq!(
            ticks.into_iter().collect::<Vec<_>>(),
            (1..61).collect::<Vec<_>>()
        );
    }

    #[test]
    fn render_sized_fragments_do_not_drop_or_duplicate_ticks() {
        let mut scheduler = FixedStepScheduler::new();
        let mut emitted = Vec::new();
        for _ in 0..144 {
            emitted.extend(scheduler.advance(Duration::from_nanos(6_944_444)));
        }
        emitted.extend(scheduler.advance(Duration::from_nanos(64)));
        assert_eq!(emitted, (1..61).collect::<Vec<_>>());
    }
}
