//! Where everything in a lane sits.
//!
//! The panel's layout is not flex: a lane is a drawing, and a state card, a
//! transition wire, and the arrowhead on its end all have to agree on one
//! coordinate system. These are the functions that place them, kept apart from
//! the view so the numbers can be checked without a window.
//!
//! Angles follow the dial's convention rather than the renderer's: zero points
//! at twelve o'clock and grows clockwise, because that is how a joint's angle
//! reads on the dial. [`arc_span`] is the one place that converts.

use std::f32::consts::FRAC_PI_2;

/// Width of one state card.
pub(crate) const NODE_W: f32 = 204.0;

/// Height of one state card.
pub(crate) const NODE_H: f32 = 214.0;

/// Clear space between two state cards, which the wires route through.
pub(crate) const GAP: f32 = 78.0;

/// Clear space at each end of a lane.
pub(crate) const PADX: f32 = 34.0;

/// Vertical distance between two stacked wire lanes.
pub(crate) const RANK: f32 = 28.0;

/// Radius of the dial's track ring.
pub(crate) const DIAL_RADIUS: f32 = 54.0;

/// Side of the dial's square box.
pub(crate) const DIAL_BOX: f32 = 132.0;

/// Radius the limit grips sit at, outside the track.
pub(crate) const GRIP_RADIUS: f32 = 68.0;

/// How far a wire's arrowhead pushes past the card edge it points into.
const HEAD_REACH: f32 = 2.0;

/// Sweep below which an arc is not drawn at all, in degrees.
///
/// The renderer reads a zero span as a whole turn, so a dial resting at zero
/// would paint a full ring rather than nothing.
const MIN_SWEEP_DEGREES: f32 = 0.05;

/// Largest sweep an arc is allowed, in degrees. A whole turn still has to read
/// as a whole turn rather than closing back onto its own start.
const MAX_SWEEP_DEGREES: f32 = 359.9;

/// Centre of the state card at `index`, along the lane.
pub(crate) fn card_center_x(index: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)] // A lane holds at most eight cards.
    let index = index as f32;
    PADX + NODE_W / 2.0 + index * (NODE_W + GAP)
}

/// Left edge of the state card at `index`.
pub(crate) fn card_left(index: usize) -> f32 {
    card_center_x(index) - NODE_W / 2.0
}

/// Left edge of the "add a state" placeholder that follows `states` cards.
pub(crate) fn add_card_left(states: usize) -> f32 {
    card_center_x(states.saturating_sub(1)) + NODE_W / 2.0 + GAP
}

/// Full width of a lane holding `states` cards, plus room for the placeholder.
pub(crate) fn lane_width(states: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)] // A lane holds at most eight cards.
    let states = states as f32;
    PADX * 2.0 + (states + 1.0) * NODE_W + states * GAP
}

/// Height of the band above or below the cards.
///
/// Tall enough for every wire lane it must hold, plus `pad` of clearance past
/// the outermost label chip, so nothing is ever clipped.
pub(crate) fn band(wires: usize, pad: f32) -> f32 {
    if wires == 0 {
        return 34.0;
    }
    #[allow(clippy::cast_precision_loss)] // A lane holds at most eight wires.
    let wires = wires as f32;
    26.0 + (wires - 1.0) * RANK + pad
}

/// Height of the band above the cards, which the release wires run through.
pub(crate) fn top_band(release_wires: usize) -> f32 {
    band(release_wires, 18.0)
}

/// Height of the band below the cards, which the dwell wires run through.
///
/// Wider clearance than the top: the lane scrolls horizontally, so a scrollbar
/// can take a strip out of the bottom of the visible area.
pub(crate) fn bottom_band(dwell_wires: usize) -> f32 {
    band(dwell_wires, 30.0)
}

/// Full height of a lane.
pub(crate) fn lane_height(release_wires: usize, dwell_wires: usize) -> f32 {
    top_band(release_wires) + NODE_H + bottom_band(dwell_wires)
}

/// The lane line a wire of the given rank runs along, above the cards.
pub(crate) fn release_wire_lane(top: f32, rank: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)] // A lane holds at most eight wires.
    let rank = rank as f32;
    top - 26.0 - rank * RANK
}

/// The lane line a wire of the given rank runs along, below the cards.
pub(crate) fn dwell_wire_lane(top: f32, rank: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)] // A lane holds at most eight wires.
    let rank = rank as f32;
    top + NODE_H + 26.0 + rank * RANK
}

/// The turning points of one wire: out of the source, along a clear lane, into
/// the target.
///
/// Four points rather than a curve, because the renderer rounds a polyline's
/// corners itself. Routing every wire through a fixed lane height is what keeps
/// a long hop the same height as a short one.
pub(crate) fn route_points(from: (f32, f32), to: (f32, f32), lane: f32) -> [(f32, f32); 4] {
    [(from.0, from.1), (from.0, lane), (to.0, lane), (to.0, to.1)]
}

/// Where a release wire leaves its card, and where it arrives.
///
/// It leaves the source's top-right corner and arrives at the target's top
/// edge, far enough in for the arrowhead to sit on the edge rather than beside
/// it.
pub(crate) fn release_wire_ends(
    source: usize,
    target: usize,
    top: f32,
) -> ((f32, f32), (f32, f32)) {
    (
        (card_center_x(source) + NODE_W / 2.0, top),
        (card_center_x(target), top - HEAD_REACH),
    )
}

/// Where a dwell wire leaves its card, and where it arrives.
pub(crate) fn dwell_wire_ends(source: usize, target: usize, top: f32) -> ((f32, f32), (f32, f32)) {
    (
        (card_center_x(source) + NODE_W / 2.0, top + NODE_H),
        (card_center_x(target), top + NODE_H + HEAD_REACH),
    )
}

/// The turning points of a wire that is still being drawn.
///
/// It leaves its port and runs along the same lane an established wire would;
/// only the far end is loose, following the pointer until a card catches it.
/// Routing the draft exactly like the wire it will become is what makes the
/// gesture legible: what is drawn while dragging is what will be left behind.
pub(crate) fn draft_points(
    source: usize,
    pointer: (f32, f32),
    target: Option<usize>,
    top: f32,
    rank: usize,
    release: bool,
) -> [(f32, f32); 4] {
    let (from, caught, lane) = if release {
        let (from, _) = release_wire_ends(source, source, top);
        (
            from,
            target.map(|index| release_wire_ends(source, index, top).1),
            release_wire_lane(top, rank),
        )
    } else {
        let (from, _) = dwell_wire_ends(source, source, top);
        (
            from,
            target.map(|index| dwell_wire_ends(source, index, top).1),
            dwell_wire_lane(top, rank),
        )
    };
    route_points(from, caught.unwrap_or(pointer), lane)
}

/// Which card, if any, covers a point in the lane's coordinates.
///
/// The catch is a little taller than a card so a wire dropped just above or
/// below one still lands on it: the gesture is aimed at a card, not at a pixel.
pub(crate) fn card_at(point: (f32, f32), top: f32, cards: usize) -> Option<usize> {
    (0..cards).find(|&index| {
        let centre = card_center_x(index);
        point.0 > centre - NODE_W / 2.0
            && point.0 < centre + NODE_W / 2.0
            && point.1 > top - 12.0
            && point.1 < top + NODE_H + 12.0
    })
}

/// A point on the dial, in the dial box's own coordinates.
///
/// Zero degrees points at twelve o'clock and grows clockwise.
pub(crate) fn polar(radius: f32, degrees: f32) -> (f32, f32) {
    let radians = degrees.to_radians();
    let centre = DIAL_BOX / 2.0;
    (
        centre + radius * radians.sin(),
        centre - radius * radians.cos(),
    )
}

/// The renderer's `from`/`to` for an arc sweeping `degrees` from twelve
/// o'clock, or `None` when there is nothing to draw.
///
/// The renderer measures from the positive x axis, grows clockwise, and always
/// sweeps the positive way round — so a counter-clockwise dial reading is the
/// same arc written from its far end back to zero, not a negative sweep.
pub(crate) fn arc_span(degrees: f32) -> Option<(f32, f32)> {
    arc_span_between(0.0, degrees)
}

/// The renderer's `from`/`to` for the arc between two dial readings.
pub(crate) fn arc_span_between(start: f32, end: f32) -> Option<(f32, f32)> {
    let span = end - start;
    if span.abs() < MIN_SWEEP_DEGREES {
        return None;
    }
    let span = span.clamp(-MAX_SWEEP_DEGREES, MAX_SWEEP_DEGREES);
    let (low, high) = if span >= 0.0 {
        (start, start + span)
    } else {
        (start + span, start)
    };
    // The dial's zero is twelve o'clock; the renderer's is three o'clock.
    Some((low.to_radians() - FRAC_PI_2, high.to_radians() - FRAC_PI_2))
}

#[cfg(test)]
mod tests {
    use super::{
        add_card_left, arc_span, band, bottom_band, card_center_x, card_left, draft_points,
        lane_height, lane_width, polar, release_wire_ends, release_wire_lane, route_points,
        top_band,
    };

    /// The seeded three-state lane the design draws, measured off it.
    #[test]
    fn a_three_state_lane_measures_what_the_design_drew() {
        assert!((lane_width(3) - 1118.0).abs() < f32::EPSILON);
        for (index, expected) in [136.0, 418.0, 700.0].into_iter().enumerate() {
            assert!((card_center_x(index) - expected).abs() < f32::EPSILON);
        }
        for (index, expected) in [34.0, 316.0, 598.0].into_iter().enumerate() {
            assert!((card_left(index) - expected).abs() < f32::EPSILON);
        }
        assert!((add_card_left(3) - 880.0).abs() < f32::EPSILON);
    }

    /// A steering joint: two key-release wires above, none below.
    #[test]
    fn a_steering_lane_is_taller_above_than_below() {
        assert!((top_band(2) - 72.0).abs() < f32::EPSILON);
        assert!((bottom_band(0) - 34.0).abs() < f32::EPSILON);
        assert!((lane_height(2, 0) - 320.0).abs() < f32::EPSILON);
    }

    /// A timed sequence: nothing above, two dwell wires below.
    #[test]
    fn a_timed_lane_is_taller_below_than_above() {
        assert!((top_band(0) - 34.0).abs() < f32::EPSILON);
        assert!((bottom_band(2) - 84.0).abs() < f32::EPSILON);
        assert!((lane_height(0, 2) - 332.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_empty_band_still_leaves_room_for_a_label() {
        assert!((band(0, 18.0) - 34.0).abs() < f32::EPSILON);
        assert!((band(0, 30.0) - 34.0).abs() < f32::EPSILON);
    }

    /// Ranks stack away from the cards, so a longer hop can sit under a
    /// shorter one without crossing it.
    #[test]
    fn each_wire_rank_clears_the_one_before_it() {
        let top = top_band(3);
        assert!((release_wire_lane(top, 0) - (top - 26.0)).abs() < f32::EPSILON);
        assert!((release_wire_lane(top, 1) - (top - 54.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn a_wire_leaves_and_arrives_vertically_through_one_lane() {
        let points = route_points((100.0, 200.0), (400.0, 190.0), 150.0);
        assert_eq!(
            points,
            [
                (100.0, 200.0),
                (100.0, 150.0),
                (400.0, 150.0),
                (400.0, 190.0)
            ]
        );
    }

    #[test]
    fn the_dial_starts_at_twelve_oclock_and_grows_clockwise() {
        let (x, y) = polar(54.0, 0.0);
        assert!((x - 66.0).abs() < 1.0e-3, "zero is straight up: {x}");
        assert!((y - 12.0).abs() < 1.0e-3, "zero is straight up: {y}");
        let (x, y) = polar(54.0, 90.0);
        assert!((x - 120.0).abs() < 1.0e-3, "a quarter turn is right: {x}");
        assert!((y - 66.0).abs() < 1.0e-3, "a quarter turn is right: {y}");
    }

    /// The renderer only sweeps one way, so a negative reading has to be
    /// written from its far end back to zero or it paints the long way round.
    #[test]
    fn a_negative_reading_sweeps_back_to_zero_rather_than_the_long_way() {
        let (from, to) = arc_span(-90.0).expect("a quarter turn is worth drawing");
        assert!(to - from > 0.0, "the renderer never sweeps backwards");
        assert!(
            (to - from - std::f32::consts::FRAC_PI_2).abs() < 1.0e-4,
            "a quarter turn back is still a quarter turn"
        );
    }

    /// A zero span reads as a whole turn to the renderer, so it must not
    /// reach it at all.
    /// A draft leaves its port exactly where the wire it will become does, so
    /// nothing shifts at the moment it is let go.
    #[test]
    fn a_draft_wire_snaps_onto_the_card_it_is_dropped_on() {
        let top = top_band(1);
        let loose = draft_points(0, (500.0, 300.0), None, top, 0, true);
        let caught = draft_points(0, (500.0, 300.0), Some(1), top, 0, true);
        let settled = release_wire_ends(0, 1, top);
        assert_eq!(loose[0], caught[0], "a draft leaves its port either way");
        assert_eq!(loose[0], settled.0);
        assert_eq!(loose[3], (500.0, 300.0), "a loose end follows the pointer");
        assert_eq!(caught[3], settled.1, "a caught end is the wire's own end");
    }

    #[test]
    fn a_dial_resting_at_zero_draws_no_sweep() {
        assert!(arc_span(0.0).is_none());
        assert!(arc_span(0.01).is_none());
        assert!(arc_span(1.0).is_some());
    }
}
