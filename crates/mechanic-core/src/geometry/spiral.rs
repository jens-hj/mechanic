//! Spirals on cylinders: a drawn wall profile swept along a helix.
//!
//! A profile is depth below the wall's surface against position along the axis,
//! over one pitch, as a periodic polyline. The same record describes a groove
//! cut into a thick cylinder and a ridge added onto a thin one: both are an
//! envelope with depth drawn into it.

use super::grid::POSITION_TICK_METERS;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};
use thiserror::Error;

/// Most points one drawn profile holds.
pub const MAX_SPIRAL_PROFILE_POINTS: usize = 8;

/// Grid the profile and the pitch are drawn on, in position ticks (1.25 cm).
pub const SPIRAL_PROFILE_STEP_TICKS: u16 = 5;

/// Smallest distance between neighbouring ridges, in position ticks (5 cm).
pub const MIN_SPIRAL_PITCH_TICKS: u16 = 20;

/// Largest distance between neighbouring ridges, in position ticks (8 m).
pub const MAX_SPIRAL_PITCH_TICKS: u16 = 3200;

/// Most identical ridges winding around one cylinder.
pub const MAX_SPIRAL_STARTS: u8 = 8;

/// Smallest diameter a tapered end may come down to, in position ticks.
pub const MIN_SPIRAL_TIP_DIAMETER_TICKS: u16 = 5;

/// Most convex ridge colliders one spiral compiles to.
pub const MAX_SPIRAL_RIDGE_COLLIDERS: usize = 512;

/// Angular steps per turn ridge colliders are cut into when the budget allows.
pub const SPIRAL_COLLIDER_STEPS_PER_TURN: u16 = 12;

/// Fewest angular steps per turn a ridge collider run may be coarsened to.
pub const MIN_SPIRAL_COLLIDER_STEPS_PER_TURN: u16 = 6;

static INTERNED: LazyLock<Mutex<HashSet<&'static SpiralRecord>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Invalid spiral.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SpiralError {
    /// The pitch was outside the supported range or off the profile grid.
    #[error("spiral pitch must be between 5 cm and 8 m in 1.25 cm increments")]
    PitchOutOfRange,
    /// The number of starts was zero or too large.
    #[error("a spiral has between 1 and {MAX_SPIRAL_STARTS} starts")]
    StartsOutOfRange,
    /// A profile held more points than it has room for.
    #[error("a spiral profile holds at most {MAX_SPIRAL_PROFILE_POINTS} points")]
    TooManyPoints,
    /// A profile point lay off the grid, beyond the pitch, or before its predecessor.
    #[error("spiral profile points run along one pitch in order, on the 1.25 cm grid")]
    PointOutOfOrder,
    /// More than two points shared one position.
    #[error("a spiral profile has at most two points at one position")]
    StackedPoints,
    /// Neither profile is drawn and there is no taper.
    #[error("a spiral needs a drawn profile or a taper")]
    Empty,
    /// The cylinder is a partial sector.
    #[error("only a full cylinder takes a spiral")]
    PartialSector,
    /// The cylinder carries material layers.
    #[error("a layered cylinder does not take a spiral")]
    Layered,
    /// A bore profile was drawn on a solid cylinder.
    #[error("a bore spiral needs a hollow cylinder")]
    BoreRequired,
    /// The cuts leave less than the minimum wall.
    #[error("the spiral cuts too deep: 2.5 cm of wall must remain")]
    TooDeep,
    /// The taper is longer than the cylinder or its tip is out of range.
    #[error("the taper must fit the cylinder and end no wider than it began")]
    TaperOutOfRange,
    /// The ridges would need more colliders than one part may carry.
    #[error(
        "the pitch is too fine for this length: at most {MAX_SPIRAL_RIDGE_COLLIDERS} ridge segments"
    )]
    TooFine,
}

/// Which way a spiral winds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpiralHand {
    /// Advances along local positive Y when turned counter-clockwise seen from
    /// positive Y: an ordinary screw.
    #[default]
    Right,
    /// The mirror image.
    Left,
}

impl SpiralHand {
    /// Sign of the axial advance per radian about local Y.
    pub const fn sign(self) -> f32 {
        match self {
            Self::Right => 1.0,
            Self::Left => -1.0,
        }
    }
}

/// Which end of the cylinder a taper narrows towards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpiralEnd {
    /// The local positive-Y end.
    PositiveY,
    /// The local negative-Y end.
    NegativeY,
}

impl SpiralEnd {
    /// Sign of the end along local Y.
    pub const fn sign(self) -> f32 {
        match self {
            Self::PositiveY => 1.0,
            Self::NegativeY => -1.0,
        }
    }
}

/// One drawn profile point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpiralPoint {
    /// Position along the axis within one pitch, in position ticks.
    pub position_ticks: u16,
    /// Depth below the wall's surface, in position ticks.
    pub depth_ticks: u16,
}

impl SpiralPoint {
    /// Creates a point.
    pub const fn new(position_ticks: u16, depth_ticks: u16) -> Self {
        Self {
            position_ticks,
            depth_ticks,
        }
    }
}

/// A periodic polyline of depth against axial position over one pitch. After
/// the last point it runs on to the first point of the next pitch. Two points at
/// one position make a square flank. No points is a plain wall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpiralProfile {
    points: [SpiralPoint; MAX_SPIRAL_PROFILE_POINTS],
    count: u8,
}

impl Default for SpiralProfile {
    fn default() -> Self {
        Self::PLAIN
    }
}

impl SpiralProfile {
    /// An undrawn wall.
    pub const PLAIN: Self = Self {
        points: [SpiralPoint::new(0, 0); MAX_SPIRAL_PROFILE_POINTS],
        count: 0,
    };

    /// Creates a profile from points in drawing order.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when there are too many points, one lies off the
    /// grid, positions run backwards, or more than two share a position.
    pub fn new(points: &[SpiralPoint]) -> Result<Self, SpiralError> {
        if points.len() > MAX_SPIRAL_PROFILE_POINTS {
            return Err(SpiralError::TooManyPoints);
        }
        let mut profile = Self::PLAIN;
        for (index, point) in points.iter().enumerate() {
            if !point
                .position_ticks
                .is_multiple_of(SPIRAL_PROFILE_STEP_TICKS)
                || !point.depth_ticks.is_multiple_of(SPIRAL_PROFILE_STEP_TICKS)
                || (index > 0 && points[index - 1].position_ticks > point.position_ticks)
            {
                return Err(SpiralError::PointOutOfOrder);
            }
            if index > 1 && points[index - 2].position_ticks == point.position_ticks {
                return Err(SpiralError::StackedPoints);
            }
            profile.points[index] = *point;
            profile.count += 1;
        }
        Ok(profile)
    }

    /// A square-flanked ridge `width_ticks` wide standing `depth_ticks` above
    /// the groove floor.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when a value lies off the grid.
    pub fn square(width_ticks: u16, depth_ticks: u16) -> Result<Self, SpiralError> {
        Self::new(&[
            SpiralPoint::new(0, depth_ticks),
            SpiralPoint::new(0, 0),
            SpiralPoint::new(width_ticks, 0),
            SpiralPoint::new(width_ticks, depth_ticks),
        ])
    }

    /// A symmetric ridge `width_ticks` wide at its foot, coming to an edge.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when a value lies off the grid.
    pub fn vee(width_ticks: u16, depth_ticks: u16) -> Result<Self, SpiralError> {
        let half = width_ticks / 2 / SPIRAL_PROFILE_STEP_TICKS * SPIRAL_PROFILE_STEP_TICKS;
        Self::new(&[
            SpiralPoint::new(0, depth_ticks),
            SpiralPoint::new(half, 0),
            SpiralPoint::new(width_ticks, depth_ticks),
        ])
    }

    /// A ridge with one square flank and one sloped over `width_ticks`.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when a value lies off the grid.
    pub fn buttress(width_ticks: u16, depth_ticks: u16) -> Result<Self, SpiralError> {
        Self::new(&[
            SpiralPoint::new(0, depth_ticks),
            SpiralPoint::new(0, 0),
            SpiralPoint::new(width_ticks, depth_ticks),
        ])
    }

    /// The drawn points, in order.
    pub fn points(&self) -> &[SpiralPoint] {
        &self.points[..usize::from(self.count)]
    }

    /// Whether nothing is drawn, or everything drawn lies at one depth.
    pub fn is_plain(&self) -> bool {
        self.points()
            .iter()
            .all(|point| point.depth_ticks == self.points[0].depth_ticks)
    }

    /// Deepest drawn depth, in position ticks.
    pub fn max_depth_ticks(&self) -> u16 {
        self.points()
            .iter()
            .map(|point| point.depth_ticks)
            .max()
            .unwrap_or(0)
    }

    /// Shallowest drawn depth, in position ticks.
    pub fn min_depth_ticks(&self) -> u16 {
        self.points()
            .iter()
            .map(|point| point.depth_ticks)
            .min()
            .unwrap_or(0)
    }

    /// The polyline's segments over one pitch as `(from, to)` in metres of
    /// `(position, depth)`, the last running on into the next pitch. Square
    /// flanks, which have no length, are left out.
    pub fn segments(&self, pitch_meters: f32) -> Vec<([f32; 2], [f32; 2])> {
        let points = self.points();
        let Some(first) = points.first() else {
            return Vec::new();
        };
        let meters = |point: &SpiralPoint, shift: f32| {
            [
                f32::from(point.position_ticks) * POSITION_TICK_METERS + shift,
                f32::from(point.depth_ticks) * POSITION_TICK_METERS,
            ]
        };
        points
            .windows(2)
            .map(|pair| (meters(&pair[0], 0.0), meters(&pair[1], 0.0)))
            .chain(core::iter::once((
                meters(&points[points.len() - 1], 0.0),
                meters(first, pitch_meters),
            )))
            .filter(|(from, to)| to[0] - from[0] > 1.0e-6)
            .collect()
    }

    /// Depth in metres at an axial position, which may lie in any pitch. On a
    /// square flank the later point wins.
    pub fn depth_at(&self, position_meters: f32, pitch_meters: f32) -> f32 {
        let position = position_meters.rem_euclid(pitch_meters);
        let segments = self.segments(pitch_meters);
        let Some(&(first, _)) = segments.first() else {
            return f32::from(self.min_depth_ticks()) * POSITION_TICK_METERS;
        };
        // Before the first point the last segment of the pitch before runs in.
        let position = if position < first[0] {
            position + pitch_meters
        } else {
            position
        };
        segments
            .iter()
            .rev()
            .find(|(from, to)| position >= from[0] && position <= to[0])
            .map_or(first[1], |(from, to)| {
                let along = (position - from[0]) / (to[0] - from[0]);
                from[1] + (to[1] - from[1]) * along
            })
    }
}

/// An end narrowing to a tip. Radii scale linearly from full size where the
/// taper begins down to the tip at the end plane; a bore keeps its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpiralTaper {
    /// End the cylinder narrows towards.
    pub end: SpiralEnd,
    /// Length of the narrowing part, in position ticks.
    pub length_ticks: u16,
    /// Envelope diameter at the end plane, in position ticks.
    pub tip_diameter_ticks: u16,
}

/// A validated spiral: profiles for the outer wall and the bore, how they wind,
/// and an optional tapered end.
///
/// Parts are copied by value all over the builder, and a drawn spiral is far
/// larger than everything else a part carries. A spec is therefore a reference
/// to its one interned record. Records are never freed: each is about a hundred
/// bytes, and a player draws a few thousand distinct spirals at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpiralSpec(&'static SpiralRecord);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SpiralRecord {
    pitch_ticks: u16,
    starts: u8,
    hand: SpiralHand,
    outer: SpiralProfile,
    inner: SpiralProfile,
    taper: Option<SpiralTaper>,
}

impl SpiralSpec {
    /// Creates a spiral, checking everything that does not depend on the
    /// cylinder it goes on.
    ///
    /// # Errors
    ///
    /// Returns [`SpiralError`] when the pitch or starts are out of range, a
    /// profile point lies beyond the pitch, or nothing at all is drawn.
    pub fn new(
        pitch_ticks: u16,
        starts: u8,
        hand: SpiralHand,
        outer: SpiralProfile,
        inner: SpiralProfile,
        taper: Option<SpiralTaper>,
    ) -> Result<Self, SpiralError> {
        if !(MIN_SPIRAL_PITCH_TICKS..=MAX_SPIRAL_PITCH_TICKS).contains(&pitch_ticks)
            || !pitch_ticks.is_multiple_of(SPIRAL_PROFILE_STEP_TICKS)
        {
            return Err(SpiralError::PitchOutOfRange);
        }
        if !(1..=MAX_SPIRAL_STARTS).contains(&starts) {
            return Err(SpiralError::StartsOutOfRange);
        }
        for profile in [&outer, &inner] {
            if profile
                .points()
                .iter()
                .any(|point| point.position_ticks > pitch_ticks)
            {
                return Err(SpiralError::PointOutOfOrder);
            }
        }
        if outer.is_plain() && inner.is_plain() && taper.is_none() {
            return Err(SpiralError::Empty);
        }
        let record = SpiralRecord {
            pitch_ticks,
            starts,
            hand,
            outer,
            inner,
            taper,
        };
        let mut interned = INTERNED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&existing) = interned.get(&record) {
            return Ok(Self(existing));
        }
        let leaked: &'static SpiralRecord = Box::leak(Box::new(record));
        interned.insert(leaked);
        Ok(Self(leaked))
    }

    /// Distance between neighbouring ridges, in position ticks.
    pub const fn pitch_ticks(self) -> u16 {
        self.0.pitch_ticks
    }

    /// Distance between neighbouring ridges, in metres.
    pub fn pitch_meters(self) -> f32 {
        f32::from(self.0.pitch_ticks) * POSITION_TICK_METERS
    }

    /// Axial advance of one ridge over a whole turn, in metres.
    pub fn lead_meters(self) -> f32 {
        self.pitch_meters() * f32::from(self.0.starts)
    }

    /// Number of identical ridges.
    pub const fn starts(self) -> u8 {
        self.0.starts
    }

    /// Winding direction.
    pub const fn hand(self) -> SpiralHand {
        self.0.hand
    }

    /// Profile of the outer wall.
    pub const fn outer(self) -> SpiralProfile {
        self.0.outer
    }

    /// Profile of the bore wall.
    pub const fn inner(self) -> SpiralProfile {
        self.0.inner
    }

    /// Tapered end, if any.
    pub const fn taper(self) -> Option<SpiralTaper> {
        self.0.taper
    }

    /// Axial advance of a ridge at `angle`, measured from local X towards local
    /// Z as cylinder walls are, relative to where it stands at angle zero.
    pub fn advance_meters(self, angle: f32) -> f32 {
        -self.0.hand.sign() * self.lead_meters() * angle / core::f32::consts::TAU
    }

    /// Axial position within one pitch of the wall at `angle` and at height `y`
    /// above the cylinder's negative end.
    pub fn phase_meters(self, angle: f32, y: f32) -> f32 {
        (y - self.advance_meters(angle)).rem_euclid(self.pitch_meters())
    }

    /// Factor on outer radii at local height `y` of a cylinder `length` long
    /// and `outer_diameter` wide.
    pub fn taper_scale(self, y: f32, length: f32, outer_diameter: f32) -> f32 {
        let Some(taper) = self.0.taper else {
            return 1.0;
        };
        let taper_length = f32::from(taper.length_ticks) * POSITION_TICK_METERS;
        let tip = f32::from(taper.tip_diameter_ticks) * POSITION_TICK_METERS / outer_diameter;
        let into = (y * taper.end.sign() - (length * 0.5 - taper_length)) / taper_length;
        (1.0 + (tip - 1.0) * into.max(0.0)).max(0.01)
    }

    /// Whole ridge segments the collider run holds at `steps_per_turn`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive count far below usize::MAX"
    )]
    pub fn ridge_segments(self, length: f32, steps_per_turn: u16) -> usize {
        // Every angular step meets each ridge along the length once, and one
        // more at either end that the end planes cut short.
        let ridges = (length / self.pitch_meters()).ceil() + 2.0;
        let drawn = |profile: SpiralProfile| {
            let floor = f32::from(profile.max_depth_ticks()) * POSITION_TICK_METERS;
            profile
                .segments(self.pitch_meters())
                .iter()
                .filter(|(from, to)| from[1] < floor - 1.0e-6 || to[1] < floor - 1.0e-6)
                .count()
        };
        ridges as usize * usize::from(steps_per_turn) * (drawn(self.0.outer) + drawn(self.0.inner))
    }

    /// Angular steps per turn the ridge colliders use on a cylinder `length`
    /// long, or `None` when even the coarsest run is over budget.
    pub fn collider_steps_per_turn(self, length: f32) -> Option<u16> {
        (MIN_SPIRAL_COLLIDER_STEPS_PER_TURN..=SPIRAL_COLLIDER_STEPS_PER_TURN)
            .rev()
            .find(|&steps| self.ridge_segments(length, steps) <= MAX_SPIRAL_RIDGE_COLLIDERS)
    }
}
