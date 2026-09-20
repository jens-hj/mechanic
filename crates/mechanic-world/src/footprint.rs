//! The patch of ground a contact manifold presses on.

use bevy_math::DVec3;

use crate::TERRAIN_CELL_METERS;

/// Oriented rectangle of ground under one contact manifold. A knife edge is
/// long and half a cell wide; a flat foot is as wide as its corners spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadFootprint {
    /// Unit direction of the longer side, in the contact plane.
    pub axis: DVec3,
    /// Half the extent along `axis`, in metres.
    pub half_length: f64,
    /// Half the extent across `axis`, in metres.
    pub half_width: f64,
}

impl LoadFootprint {
    /// Smallest half extent: no footprint is narrower than half a terrain cell.
    pub const MINIMUM_HALF_EXTENT: f64 = TERRAIN_CELL_METERS * 0.5;
    /// Largest half extent a single manifold loads.
    pub const MAXIMUM_HALF_EXTENT: f64 = 1.0;

    /// A square footprint, for loads with no preferred direction.
    pub fn square(normal: DVec3, half_extent: f64) -> Self {
        Self {
            axis: normal.any_orthonormal_vector(),
            half_length: half_extent,
            half_width: half_extent,
        }
    }

    /// Centre and footprint of contact points sharing `normal`: the smallest
    /// rectangle, aligned with some pair of points, that holds them all.
    pub fn spanning(points: &[DVec3], normal: DVec3) -> (DVec3, Self) {
        let bound = |half: f64| half.clamp(Self::MINIMUM_HALF_EXTENT, Self::MAXIMUM_HALF_EXTENT);
        let mean = points.iter().sum::<DVec3>()
            / f64::from(u32::try_from(points.len()).unwrap_or(u32::MAX).max(1));
        let mut best = (mean, Self::square(normal, Self::MINIMUM_HALF_EXTENT));
        let mut smallest = f64::INFINITY;
        for (index, &a) in points.iter().enumerate() {
            for &b in &points[index + 1..] {
                let between = b - a;
                let Some(axis) = (between - normal * between.dot(normal)).try_normalize() else {
                    continue;
                };
                let across = normal.cross(axis);
                let mut low = [f64::INFINITY; 2];
                let mut high = [f64::NEG_INFINITY; 2];
                for &point in points {
                    for (side, direction) in [axis, across].into_iter().enumerate() {
                        let offset = (point - mean).dot(direction);
                        low[side] = low[side].min(offset);
                        high[side] = high[side].max(offset);
                    }
                }
                let half = [0.5 * (high[0] - low[0]), 0.5 * (high[1] - low[1])];
                let area = bound(half[0]) * bound(half[1]);
                if area < smallest {
                    smallest = area;
                    let centre = mean
                        + axis * (0.5 * (high[0] + low[0]))
                        + across * (0.5 * (high[1] + low[1]));
                    let (axis, half) = if half[0] >= half[1] {
                        (axis, half)
                    } else {
                        (across, [half[1], half[0]])
                    };
                    best = (
                        centre,
                        Self {
                            axis,
                            half_length: bound(half[0]),
                            half_width: bound(half[1]),
                        },
                    );
                }
            }
        }
        best
    }

    /// Loaded area in square metres.
    pub fn area(self) -> f64 {
        4.0 * self.half_length * self.half_width
    }

    /// Lowest and highest corner, relative to the centre, of the axis-aligned
    /// box around the footprint reaching `below` metres into the ground and
    /// `above` metres out of it along `normal`.
    pub fn bounds(self, normal: DVec3, below: f64, above: f64) -> [DVec3; 2] {
        let across = normal.cross(self.axis);
        let flat = self.axis.abs() * self.half_length + across.abs() * self.half_width;
        let (into, out) = (normal * -below, normal * above);
        [into.min(out) - flat, into.max(out) + flat]
    }

    /// Distance from the centre to a corner, in metres.
    pub fn reach(self) -> f64 {
        self.half_length.hypot(self.half_width)
    }

    /// Whether a point `offset` from the centre lies over the footprint.
    pub fn contains(self, normal: DVec3, offset: DVec3) -> bool {
        let [along, across] = self.planar(normal, offset);
        along <= self.half_length && across <= self.half_width
    }

    /// Whether a point `offset` from the centre lies closer than `margin`
    /// metres to the footprint. Half a cell of margin finds every cell the
    /// footprint overlaps, and not the neighbours it only touches.
    pub fn reaches(self, normal: DVec3, offset: DVec3, margin: f64) -> bool {
        // Cell centres are sums of rounded products; a neighbour one whole cell
        // away must not slip inside on rounding.
        let margin = margin - 1e-9;
        let [along, across] = self.planar(normal, offset);
        along < self.half_length + margin && across < self.half_width + margin
    }

    // Distance from the centre along and across the footprint.
    fn planar(self, normal: DVec3, offset: DVec3) -> [f64; 2] {
        let tangent = offset - normal * offset.dot(normal);
        let along = tangent.dot(self.axis);
        [along.abs(), (tangent - self.axis * along).length()]
    }

    pub(crate) fn is_valid(self, normal: DVec3) -> bool {
        let extents = Self::MINIMUM_HALF_EXTENT..=Self::MAXIMUM_HALF_EXTENT;
        self.axis.is_finite()
            && (self.axis.length_squared() - 1.0).abs() < 0.01
            && self.axis.dot(normal).abs() < 0.1
            && extents.contains(&self.half_length)
            && extents.contains(&self.half_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_knife_edge_is_long_and_half_a_cell_wide() {
        let (centre, footprint) = LoadFootprint::spanning(
            &[DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.25, 1.0, 0.0)],
            DVec3::Y,
        );
        assert!(centre.distance(DVec3::new(0.125, 1.0, 0.0)) < 1e-12);
        assert!(footprint.axis.x.abs() > 0.999);
        assert!((footprint.half_length - 0.125).abs() < 1e-12);
        assert!((footprint.half_width - LoadFootprint::MINIMUM_HALF_EXTENT).abs() < 1e-12);
        assert!(footprint.contains(DVec3::Y, DVec3::new(0.12, 0.3, 0.02)));
        assert!(!footprint.contains(DVec3::Y, DVec3::new(0.0, 0.0, 0.05)));
        assert!(!footprint.contains(DVec3::Y, DVec3::new(0.2, 0.0, 0.0)));
    }

    #[test]
    fn a_flat_foot_spreads_as_wide_as_its_corners() {
        let corners = [
            DVec3::new(-0.1, 0.0, -0.1),
            DVec3::new(0.1, 0.0, -0.1),
            DVec3::new(0.1, 0.0, 0.1),
            DVec3::new(-0.1, 0.0, 0.1),
        ];
        let (_, footprint) = LoadFootprint::spanning(&corners, DVec3::Y);
        assert!((footprint.area() - 0.04).abs() < 1e-9);
        assert!(footprint.is_valid(DVec3::Y));
    }

    #[test]
    fn a_tilted_footprint_is_bounded_by_a_box_that_holds_its_corners() {
        let normal = DVec3::new(0.6, 0.8, 0.0);
        let footprint = LoadFootprint {
            axis: DVec3::Z,
            half_length: 0.5,
            half_width: 0.1,
        };
        let [low, high] = footprint.bounds(normal, 0.05, 0.025);
        let across = normal.cross(footprint.axis);
        for corner in [
            DVec3::Z * 0.5 + across * 0.1 - normal * 0.05,
            DVec3::Z * -0.5 - across * 0.1 + normal * 0.025,
        ] {
            assert!(corner.cmpge(low - 1e-12).all() && corner.cmple(high + 1e-12).all());
        }
        assert!(high.y < 0.2 && low.y > -0.2);
    }

    #[test]
    fn a_single_point_loads_one_cell() {
        let (_, footprint) = LoadFootprint::spanning(&[DVec3::ZERO], DVec3::Y);
        assert!((footprint.area() - TERRAIN_CELL_METERS.powi(2)).abs() < 1e-12);
        assert!(footprint.is_valid(DVec3::Y));
    }
}
