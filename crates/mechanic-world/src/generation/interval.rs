//! Conservative interval arithmetic for classifying whole regions of the field.

/// A closed range `[lo, hi]` known to contain every value an expression takes
/// over some region. Infinite bounds are allowed; NaN never is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl Interval {
    pub(crate) const EVERYTHING: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    pub(crate) const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    pub(crate) const fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    pub(crate) fn centre(self) -> f64 {
        if self.lo.is_finite() && self.hi.is_finite() {
            0.5 * (self.lo + self.hi)
        } else if self.lo.is_finite() {
            self.lo
        } else if self.hi.is_finite() {
            self.hi
        } else {
            0.0
        }
    }

    pub(crate) fn radius(self) -> f64 {
        0.5 * (self.hi - self.lo)
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self::new(self.lo + other.lo, self.hi + other.hi).sanitised()
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self::new(self.lo - other.hi, self.hi - other.lo).sanitised()
    }

    pub(crate) fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }

    pub(crate) fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|value| value.is_nan()) {
            return Self::EVERYTHING;
        }
        Self::new(
            products.into_iter().fold(f64::INFINITY, f64::min),
            products.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    /// Everything when the divisor can be zero.
    pub(crate) fn div(self, other: Self) -> Self {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            Self::EVERYTHING
        } else {
            self.mul(Self::new(1.0 / other.hi, 1.0 / other.lo))
        }
    }

    pub(crate) fn min(self, other: Self) -> Self {
        Self::new(self.lo.min(other.lo), self.hi.min(other.hi))
    }

    pub(crate) fn max(self, other: Self) -> Self {
        Self::new(self.lo.max(other.lo), self.hi.max(other.hi))
    }

    pub(crate) fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Self::new(0.0, (-self.lo).max(self.hi))
        }
    }

    pub(crate) fn clamp(self, lo: f64, hi: f64) -> Self {
        Self::new(self.lo.clamp(lo, hi), self.hi.clamp(lo, hi))
    }

    /// Interval of a monotonically non-decreasing function.
    pub(crate) fn monotone(self, function: impl Fn(f64) -> f64) -> Self {
        Self::new(function(self.lo), function(self.hi)).sanitised()
    }

    pub(crate) fn sin(self) -> Self {
        periodic_bounds(self, f64::sin, core::f64::consts::FRAC_PI_2)
    }

    pub(crate) fn cos(self) -> Self {
        periodic_bounds(self, f64::cos, 0.0)
    }

    /// Interval of `centre ± lipschitz * reach`, used for smooth functions
    /// whose slope is bounded but whose shape interval arithmetic can't see.
    pub(crate) fn around(centre: f64, lipschitz: f64, reach: f64) -> Self {
        let spread = lipschitz * reach;
        Self::new(centre - spread, centre + spread).sanitised()
    }

    pub(crate) fn intersect(self, other: Self) -> Self {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Self::new(lo, hi) } else { self }
    }

    pub(crate) fn hull(self, other: Self) -> Self {
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    fn sanitised(self) -> Self {
        Self::new(
            if self.lo.is_nan() {
                f64::NEG_INFINITY
            } else {
                self.lo
            },
            if self.hi.is_nan() {
                f64::INFINITY
            } else {
                self.hi
            },
        )
    }
}

/// Bounds of a 2π-periodic function in [-1, 1] whose maximum lies at `peak`.
fn periodic_bounds(interval: Interval, function: fn(f64) -> f64, peak: f64) -> Interval {
    use core::f64::consts::{PI, TAU};
    if !interval.lo.is_finite() || !interval.hi.is_finite() || interval.hi - interval.lo >= TAU {
        return Interval::new(-1.0, 1.0);
    }
    let first = function(interval.lo);
    let second = function(interval.hi);
    let mut lo = first.min(second);
    let mut hi = first.max(second);
    let contains = |target: f64| {
        let turns = ((interval.lo - target) / TAU).ceil();
        target + turns * TAU <= interval.hi
    };
    if contains(peak) {
        hi = 1.0;
    }
    if contains(peak + PI) {
        lo = -1.0;
    }
    Interval::new(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::Interval;

    #[test]
    fn sine_bounds_contain_every_sampled_value() {
        for (lo, hi) in [
            (0.1, 0.4),
            (1.0, 2.0),
            (-4.0, -2.5),
            (3.0, 3.3),
            (-0.2, 7.0),
        ] {
            let bounds = Interval::new(lo, hi).sin();
            let cosine = Interval::new(lo, hi).cos();
            for step in 0..=100 {
                let value = lo + (hi - lo) * f64::from(step) / 100.0;
                assert!(bounds.lo <= value.sin() && value.sin() <= bounds.hi);
                assert!(cosine.lo <= value.cos() && value.cos() <= cosine.hi);
            }
        }
    }

    #[test]
    fn products_cover_mixed_signs() {
        let product = Interval::new(-2.0, 3.0).mul(Interval::new(-1.0, 4.0));
        assert_eq!(product, Interval::new(-8.0, 12.0));
    }
}
