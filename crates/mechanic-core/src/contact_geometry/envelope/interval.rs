//! Minimal outward-rounded intervals for geometric certificates. No fast-math.

use super::ContactGeometryError;
use bevy_math::DVec3;

#[derive(Clone, Copy, Debug)]
pub(in crate::contact_geometry) struct Interval {
    pub(in crate::contact_geometry) lo: f64,
    pub(in crate::contact_geometry) hi: f64,
}

impl Interval {
    pub(in crate::contact_geometry) fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    pub(in crate::contact_geometry) fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }
    pub(in crate::contact_geometry) fn sub(self, other: Self) -> Self {
        Self {
            lo: (self.lo - other.hi).next_down(),
            hi: (self.hi - other.lo).next_up(),
        }
    }
    pub(in crate::contact_geometry) fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|value| value.is_nan()) {
            return Self {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        Self {
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    pub(in crate::contact_geometry) fn scale(self, value: f64) -> Self {
        self.mul(Self::point(value))
    }
    pub(in crate::contact_geometry) fn abs(self) -> Self {
        Self {
            lo: if self.lo <= 0.0 && self.hi >= 0.0 {
                0.0
            } else {
                self.lo.abs().min(self.hi.abs())
            },
            hi: self.lo.abs().max(self.hi.abs()),
        }
    }
    pub(in crate::contact_geometry) fn square(self) -> Self {
        let positive = self.abs();
        Self {
            lo: (positive.lo * positive.lo).next_down().max(0.0),
            hi: (positive.hi * positive.hi).next_up(),
        }
    }
    pub(in crate::contact_geometry) fn sqrt(self) -> Result<Self, ContactGeometryError> {
        self.finite()?;
        if self.hi < 0.0 {
            return Err(ContactGeometryError);
        }
        Ok(Self {
            lo: self.lo.max(0.0).sqrt().next_down().max(0.0),
            hi: self.hi.sqrt().next_up(),
        })
    }
    pub(in crate::contact_geometry) fn div_positive(
        self,
        divisor: Self,
    ) -> Result<Self, ContactGeometryError> {
        if divisor.lo <= 0.0 {
            return Err(ContactGeometryError);
        }
        divisor.finite()?;
        self.mul(Self {
            lo: (1.0 / divisor.hi).next_down(),
            hi: (1.0 / divisor.lo).next_up(),
        })
        .finite()
    }
    pub(in crate::contact_geometry) fn finite(self) -> Result<Self, ContactGeometryError> {
        if self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi {
            Ok(self)
        } else {
            Err(ContactGeometryError)
        }
    }
}

pub(in crate::contact_geometry) fn dot(a: DVec3, b: DVec3) -> Interval {
    (0..3).fold(Interval::point(0.0), |sum, axis| {
        sum.add(Interval::point(a[axis]).scale(b[axis]))
    })
}
