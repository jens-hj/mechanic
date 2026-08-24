//! The control block, as a Mosaic tree.
//!
//! Built once and driven by reactive state thereafter. That is the reason
//! almost nothing here reads a value directly: a lane's contents are read
//! inside bindings, keyed off the wire the lane speaks for, so an edit
//! re-evaluates a binding instead of rebuilding the element. Elements
//! surviving an edit is what lets a drag keep the pointer it captured.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary is meant to be globbed.

use std::cell::RefCell;
use std::rc::Rc;

use bevy_mosaic::ui::*;
use mechanic_core::DriveLinkId;
use mosaic_core::theme::color;
use mosaic_macros::view;
use mosaic_widgets::input::EventCtx;

use super::geometry;
use super::model::{Intent, LaneModel, Mode, PanelEdit, PanelModel, Preset, StateModel};
use crate::control_panel::SpeedUnit;
use crate::ui::UiIntent;
#[allow(clippy::wildcard_imports)] // The design tokens are read as bare names.
use crate::ui::theme::*;

/// A wire being drawn, while the pointer still has hold of it.
///
/// Lane-local, not part of the model: nothing about it survives the gesture,
/// and the graph never hears about it until it is let go.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Draft {
    /// The state the wire leaves.
    source: usize,
    /// Whether it hands off on a key release rather than after a wait.
    release: bool,
    /// Where the pointer has got to, in the lane's coordinates.
    at: (f32, f32),
    /// The card it has caught, if it is over one.
    target: Option<usize>,
}

/// Everything the view reads and writes.
///
/// Handed around by value: every field is a `Copy` handle into the reactive
/// graph or a shared queue, so cloning one costs a pointer.
#[derive(Clone)]
pub(crate) struct Handles {
    /// What the graph says, refreshed whenever it changes.
    pub(crate) model: State<PanelModel>,
    /// The lane the pointer last landed in.
    pub(crate) selected: State<Option<DriveLinkId>>,
    /// The lane whose joint is being pointed out in the world.
    pub(crate) located: State<Option<DriveLinkId>>,
    /// The state waiting for a key to bind, if any.
    pub(crate) capturing: State<Option<(DriveLinkId, u8)>>,
    /// What the whole overlay is asking the world to change, which a drive
    /// edit joins rather than queues beside: one queue is one order.
    pub(crate) intents: Rc<RefCell<Vec<UiIntent>>>,
}

impl Handles {
    /// Asks the world to put this panel away.
    fn close(&self) {
        self.intents.borrow_mut().push(UiIntent::CloseControlPanel);
    }

    /// Queues an edit against one joint.
    pub(crate) fn edit(&self, joint: DriveLinkId, edit: PanelEdit) {
        self.intents.borrow_mut().push(UiIntent::Drive(Intent {
            lane: joint,
            edit,
            transient: false,
        }));
    }

    /// Queues an edit made part-way through a drag, which must not land in
    /// history on its own.
    pub(crate) fn dragging(&self, joint: DriveLinkId, edit: PanelEdit) {
        self.intents.borrow_mut().push(UiIntent::Drive(Intent {
            lane: joint,
            edit,
            transient: true,
        }));
    }

    /// Reads something out of one lane, or falls back when the joint has gone.
    pub(crate) fn read<T: 'static>(
        &self,
        joint: DriveLinkId,
        fallback: T,
        of: impl Fn(&LaneModel) -> T,
    ) -> T {
        self.model
            .with(|model| model.lane(joint).map(&of))
            .unwrap_or(fallback)
    }
}

/// Reads something out of one lane.
///
/// Free rather than a method so a binding's closure captures only `Copy`
/// handles — a binding is read from several places, and a closure holding an
/// `Rc` could only be used once.
fn lane_read<T: 'static>(
    model: State<PanelModel>,
    joint: DriveLinkId,
    fallback: T,
    of: impl Fn(&LaneModel) -> T,
) -> T {
    model
        .with(|panel| panel.lane(joint).map(&of))
        .unwrap_or(fallback)
}

/// The panel's own frame.
///
/// Inset from the window by its own margin rather than by a padded wrapper: a
/// full-bleed wrapper would take the pointer everywhere it covers, and the
/// overlay's root is the one element allowed to do that.
pub(crate) fn panel(handles: &Handles) -> Element {
    let header = header(handles);
    let lanes = lanes(handles);
    view! {
        col margin:22px fill:shell radius:14px stroke:(width:1px color:shell-edge)
            shadow:(offset:(x:0px y:30px) blur:90px color:#00000099)
            font-color:ink.fg {
            (header)
            (lanes)
        }
    }
}

/// The title bar: what this is, how much of it there is, and what its colours
/// mean.
fn header(handles: &Handles) -> Element {
    let model = handles.model;
    let close = handles.clone();
    view! {
        row height:min-content align:center justify:between
            pad:(horizontal:22px vertical:16px)
            stroke:(width:1px color:shell-rule edges:bottom) {
            row height:min-content align:center gap:14px {
                col width:34px height:34px align:center justify:center radius:8px
                    fill:#0F2A2A stroke:(width:1px color:accent.key) font-color:accent.key {
                    icon size:18px mark
                }
                col height:min-content gap:1px {
                    text font-size:19px font-weight:700 letter-spacing:2.7px "CONTROL BLOCK"
                    text font-size:12px letter-spacing:0.6px font-color:ink.dim
                        { $model.subtitle() }
                }
            }
            row height:min-content align:center gap:18px {
                (legend())
                col width:32px height:32px align:center justify:center radius:8px
                    stroke:(width:1px color:shell-rule) font-color:ink.muted
                    hover { fill:reticle.fill-over stroke:(width:1px color:accent.key) }
                    @click:{ close.close() } {
                    canvas width:18px height:18px nohit {
                        line from:(x:3px y:3px) to:(x:15px y:15px)
                            stroke:(width:4px cap:square color:ink.muted)
                        line from:(x:15px y:3px) to:(x:3px y:15px)
                            stroke:(width:4px cap:square color:ink.muted)
                    }
                }
            }
        }
    }
}

/// What each colour means, so the lanes below need no captions.
fn legend() -> Element {
    view! {
        row height:min-content align:center gap:18px {
            (legend_item("hold angle"))
            (legend_item("spin"))
            (legend_item("key"))
            (legend_item("time"))
        }
    }
}

/// One entry of the legend.
fn legend_item(label: &'static str) -> Element {
    let tile = match label {
        "hold angle" => view! {
            col width:22px height:22px align:center justify:center radius:6px
                fill:wash.angle font-color:accent.angle { icon size:14px legend-angle }
        },
        "spin" => view! {
            col width:22px height:22px align:center justify:center radius:6px
                fill:wash.speed font-color:accent.speed { icon size:14px legend-spin }
        },
        "key" => view! {
            col width:22px height:22px align:center justify:center radius:6px
                fill:wash.key font-color:accent.key { icon size:14px legend-key }
        },
        _ => view! {
            col width:22px height:22px align:center justify:center radius:6px
                fill:wash.time font-color:accent.time { icon size:14px legend-time }
        },
    };
    view! {
        row height:min-content align:center gap:7px {
            (tile)
            text font-size:12px letter-spacing:0.5px font-color:ink.legend (label)
        }
    }
}

/// One lane per joint, scrolling vertically.
fn lanes(handles: &Handles) -> Element {
    let model = handles.model;
    let inner = handles.clone();
    view! {
        col height:1fr {
            scroll pad:(left:14px right:14px top:8px bottom:18px) {
                for (id, ()) in { $model.keys() } {
                    (lane_row(&inner, *id))
                }
            } as body
            {
                body.thumb_color(|| color(lane.edge_on));
            }
        }
    }
}

/// One joint: what it may do, and the states it moves through.
fn lane_row(handles: &Handles, id: DriveLinkId) -> Element {
    let handles = handles.clone();
    let selected = handles.selected;
    let located = handles.located;
    let number = {
        let handles = handles.clone();
        move || handles.read(id, 0, |joint| joint.number).to_string()
    };
    view! {
        row height:min-content margin:(top:10px) radius:12px
            fill:if $selected == Some(id) { { color(lane.fill_on) } } else { { color(lane.fill) } }
            stroke:(width:1px color:{
                if located.get() == Some(id) {
                    color(accent.key)
                } else if selected.get() == Some(id) {
                    color(lane.edge_on)
                } else {
                    color(lane.edge)
                }
            })
            @pointer-down:{ selected.set(Some(id)) } {
            col width:262px shrink:0 height:fill gap:10px
                pad:(left:14px right:14px top:14px bottom:12px)
                stroke:(width:1px color:lane.edge edges:right) {
                row height:min-content align:center gap:9px {
                    col width:30px height:30px shrink:0 align:center justify:center radius:7px
                        fill:if $selected == Some(id) { { color(wash.badge) } }
                            else { { color(badge.fill) } }
                        stroke:(width:1px color:{
                            if selected.get() == Some(id) { color(accent.key) } else { color(badge.edge) }
                        })
                        font-color:{
                            if selected.get() == Some(id) { color(accent.key) } else { color(badge.ink) }
                        } {
                        text font-size:15px font-weight:700 { number() }
                    }
                    (name_field(&handles, id))
                    col width:28px height:28px shrink:0 align:center justify:center radius:7px
                        stroke:(width:1px color:{
                            if located.get() == Some(id) { color(accent.key) } else { color(reticle.edge) }
                        })
                        font-color:{
                            if located.get() == Some(id) { color(accent.key) } else { color(ink.faint) }
                        }
                        hover { fill:reticle.fill_over }
                        @pointer:{ move |event: &PointerEvent, _: &mut EventCtx| {
                            match event.kind {
                                PointerEventKind::Enter => located.set(Some(id)),
                                PointerEventKind::Leave => located.set(None),
                                _ => {}
                            }
                        } } {
                        icon size:16px locate
                    }
                }
                (chips(&handles, id))
                (presets(&handles, id))
            }
            (canvas_for(&handles, id))
        }
    }
}

/// The lane's drawing: one card per state, wired together.
///
/// A stack, so every child is placed from the same origin and `translate:`
/// means the same thing in all of them — the shared frame a wire needs in
/// order to leave one card and arrive at another. The wires themselves are a
/// canvas behind the cards, which is where the shapes live.
fn canvas_for(handles: &Handles, id: DriveLinkId) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let cards = handles.clone();
    let releases = handles.clone();
    let dwells = handles.clone();
    let adder = handles.clone();
    let wiring = handles.clone();
    let labels = handles.clone();
    let dwell_label_handles = handles.clone();
    let release_labels = move || lane_read(model, id, 0, |joint| joint.release_wires.len());
    let dwell_labels = move || lane_read(model, id, 0, |joint| joint.dwell_wires.len());
    let count = move || lane_read(model, id, 0, |joint| joint.states.len());
    // Where the lane landed, so a wire dropped on a card can be read in the
    // same coordinates the cards are placed in.
    let frame: State<Vector2> = State::new(Vector2::ZERO);
    // The wire currently being pulled out of a port, if any. It lives here
    // rather than in the port that started it because everything that has to
    // draw it — the wire itself and the card it catches — is a sibling.
    let draft: State<Option<Draft>> = State::new(None);
    // The lane carries its whole drawing, bands included, so it is measured
    // rather than left to shrink onto its children.
    let width = move || Dimension::Px(geometry::lane_width(count()));
    let height = move || {
        Dimension::Px(lane_read(model, id, 282.0, |joint| {
            geometry::lane_height(joint.release_wires.len(), joint.dwell_wires.len())
        }))
    };
    view! {
        col width:1fr height:{ height() } min-width:0px {
            scroll {
                stack width:{ width() } height:{ height() } align:start justify:start
                    @layout:{ move |bounds: Rect| frame.set(bounds.origin) } {
                    (wires(&wiring, id, draft))
                    for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                        (state_card(&cards, id, *index, draft))
                    }
                    (add_card(&adder, id))
                    // The ports hang off the cards' corners and the labels sit
                    // on the wires, but each is its own child of the lane: a
                    // grouping wrapper would have to be full-bleed, and a
                    // full-bleed box either swallows the pointer or, told not
                    // to, takes its whole subtree out of reach with it.
                    for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                        (release_port(&releases, id, *index, frame, draft))
                    }
                    for (index, ()) in { (0..count()).map(|index| (index, ())) } {
                        (dwell_port(&dwells, id, *index, frame, draft))
                    }
                    for (rank, ()) in { (0..release_labels()).map(|rank| (rank, ())) } {
                        (wire_label(&labels, id, *rank, true))
                    }
                    for (rank, ()) in { (0..dwell_labels()).map(|rank| (rank, ())) } {
                        (wire_label(&dwell_label_handles, id, *rank, false))
                    }
                }
            }
        }
    }
}

/// Every wire in the lane, drawn behind the cards.
///
/// A canvas, because a wire is the one thing here that has to know where two
/// separate cards are: inside it every point resolves against the same rect.
/// The lines are written here rather than built by a helper because a line
/// carrying an arrowhead is two elements, and a target-less `view!` can only
/// return one.
fn wires(handles: &Handles, id: DriveLinkId, draft: State<Option<Draft>>) -> Element {
    let model = handles.model;
    let releases = move || lane_read(model, id, 0, |joint| joint.release_wires.len());
    let dwells = move || lane_read(model, id, 0, |joint| joint.dwell_wires.len());
    let drawing = move || draft.get().is_some();
    let point = move |which: usize| {
        let points = draft_points(model, id, draft.get());
        (Length::px(points[which].0), Length::px(points[which].1))
    };
    view! {
        canvas nohit {
            for (rank, ()) in { (0..releases()).map(|rank| (rank, ())) } {
                let at = *rank;
                line through:(
                    (x:{ Length::px(wire_points(model, id, at, true)[0].0) }
                     y:{ Length::px(wire_points(model, id, at, true)[0].1) })
                    (x:{ Length::px(wire_points(model, id, at, true)[1].0) }
                     y:{ Length::px(wire_points(model, id, at, true)[1].1) })
                    (x:{ Length::px(wire_points(model, id, at, true)[2].0) }
                     y:{ Length::px(wire_points(model, id, at, true)[2].1) })
                    (x:{ Length::px(wire_points(model, id, at, true)[3].0) }
                     y:{ Length::px(wire_points(model, id, at, true)[3].1) })
                )
                    corner:12px head:triangle
                    stroke:(width:2px color:accent.key)
            }
            for (rank, ()) in { (0..dwells()).map(|rank| (rank, ())) } {
                let at = *rank;
                line through:(
                    (x:{ Length::px(wire_points(model, id, at, false)[0].0) }
                     y:{ Length::px(wire_points(model, id, at, false)[0].1) })
                    (x:{ Length::px(wire_points(model, id, at, false)[1].0) }
                     y:{ Length::px(wire_points(model, id, at, false)[1].1) })
                    (x:{ Length::px(wire_points(model, id, at, false)[2].0) }
                     y:{ Length::px(wire_points(model, id, at, false)[2].1) })
                    (x:{ Length::px(wire_points(model, id, at, false)[3].0) }
                     y:{ Length::px(wire_points(model, id, at, false)[3].1) })
                )
                    corner:12px head:triangle
                    stroke:(width:2px color:accent.time)
            }
            // The wire being pulled out of a port, routed like the one it will
            // become: it leaves the same port, runs along a lane rather than
            // straight at the pointer, and ends on the card it has caught.
            if drawing() {
                line through:(
                    (x:{ point(0).0 } y:{ point(0).1 })
                    (x:{ point(1).0 } y:{ point(1).1 })
                    (x:{ point(2).0 } y:{ point(2).1 })
                    (x:{ point(3).0 } y:{ point(3).1 })
                )
                    corner:12px head:triangle
                    stroke:(width:2px color:draft_color(draft.get()))
            }
        }
    }
}

/// Where the wire being drawn runs, in the lane's own coordinates.
fn draft_points(
    model: State<PanelModel>,
    id: DriveLinkId,
    draft: Option<Draft>,
) -> [(f32, f32); 4] {
    let Some(draft) = draft else {
        return [(0.0, 0.0); 4];
    };
    let top = band_top(model, id);
    let rank = draft_rank(model, id, draft.source, draft.release);
    geometry::draft_points(
        draft.source,
        draft.at,
        draft.target,
        top,
        rank,
        draft.release,
    )
}

/// Which lane the wire being drawn runs in.
///
/// The one its source already uses, where it has one. A wire that does not
/// exist yet has no lane of its own — the band only grows to hold it once it
/// lands — so it borrows the outermost lane there is rather than routing itself
/// above the band and out of sight.
fn draft_rank(model: State<PanelModel>, id: DriveLinkId, source: usize, release: bool) -> usize {
    lane_read(model, id, 0, |joint| {
        let family = if release {
            &joint.release_wires
        } else {
            &joint.dwell_wires
        };
        family
            .iter()
            .position(|wire| wire.source == source)
            .unwrap_or_else(|| family.len().saturating_sub(1))
    })
}

/// What colour the wire being drawn reads in: its own once it has caught a
/// card, and a plain white while it is still loose.
fn draft_color(draft: Option<Draft>) -> Color {
    let Some(draft) = draft else {
        return color(ink.fg);
    };
    if draft.target.is_none() {
        return color(ink.fg);
    }
    if draft.release {
        color(accent.key)
    } else {
        color(accent.time)
    }
}

/// Where one wire turns, in the lane's own coordinates.
fn wire_points(
    model: State<PanelModel>,
    id: DriveLinkId,
    rank: usize,
    release: bool,
) -> [(f32, f32); 4] {
    lane_read(model, id, [(0.0, 0.0); 4], |joint| {
        let family = if release {
            &joint.release_wires
        } else {
            &joint.dwell_wires
        };
        family.get(rank).map_or([(0.0, 0.0); 4], |found| {
            found.route(rank, geometry::top_band(joint.release_wires.len()), release)
        })
    })
}

/// What one wire's pill reads.
fn wire_text(model: State<PanelModel>, id: DriveLinkId, rank: usize, release: bool) -> String {
    lane_read(model, id, String::new(), |joint| {
        let family = if release {
            &joint.release_wires
        } else {
            &joint.dwell_wires
        };
        family
            .get(rank)
            .map(|found| found.label.clone())
            .unwrap_or_default()
    })
}

/// Which state one wire leaves.
fn wire_source(model: State<PanelModel>, id: DriveLinkId, rank: usize, release: bool) -> u8 {
    lane_read(model, id, 0u8, |joint| {
        let family = if release {
            &joint.release_wires
        } else {
            &joint.dwell_wires
        };
        family
            .get(rank)
            .and_then(|found| u8::try_from(found.source).ok())
            .unwrap_or(0)
    })
}

/// Where one wire's pill sits: the midpoint of its run along the lane.
fn wire_label_at(
    model: State<PanelModel>,
    id: DriveLinkId,
    rank: usize,
    release: bool,
) -> (f32, f32) {
    lane_read(model, id, (0.0, 0.0), |joint| {
        let family = if release {
            &joint.release_wires
        } else {
            &joint.dwell_wires
        };
        family.get(rank).map_or((0.0, 0.0), |found| {
            found.label_at(rank, geometry::top_band(joint.release_wires.len()), release)
        })
    })
}

/// How long one wire's dwell is, as it would be typed.
fn wire_dwell_text(model: State<PanelModel>, id: DriveLinkId, rank: usize) -> String {
    lane_read(model, id, String::new(), |joint| {
        joint
            .dwell_wires
            .get(rank)
            .and_then(|found| joint.states.get(found.source))
            .and_then(|state| state.dwell)
            .map_or_else(String::new, |(seconds, _)| format!("{seconds:.1}"))
    })
}

/// One wire's pill.
///
/// A release pill names a state and only steps through them; a dwell pill
/// carries a number, so clicking it opens a field.
fn wire_label(handles: &Handles, id: DriveLinkId, rank: usize, release: bool) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let typing: State<bool> = State::new(false);
    let buffer: State<String> = State::new(String::new());
    let commit: Rc<dyn Fn()> = Rc::new({
        let handles = handles.clone();
        move || {
            if let Ok(seconds) = buffer.get_untracked().trim().parse::<f32>() {
                handles.edit(
                    id,
                    PanelEdit::SetDwell {
                        state: wire_source(model, id, rank, release),
                        seconds,
                    },
                );
            }
            typing.set(false);
        }
    });
    // Centred on the wire by measuring: a pill's width depends on its text, so
    // half of it is only known once it has been laid out.
    let width: State<f32> = State::new(0.0);
    let at = move || wire_label_at(model, id, rank, release);
    view! {
        row width:min-content height:24px align:center gap:5px
            // Wider on the right than the design's even 9px: the icon carries
            // its own bearing inside its box and a glyph does not, so equal
            // padding reads as text crowding the cap it sits against.
            pad:(left:9px right:12px top:0px bottom:0px) radius:12px
            translate:(x:{ Length::px(at().0 - width.get() / 2.0) }
                       y:{ Length::px(at().1 - 12.0) })
            @layout:{ move |bounds: Rect| {
                if (width.get_untracked() - bounds.size.width).abs() > 0.5 {
                    width.set(bounds.size.width);
                }
            } }
            fill:port.fill
            stroke:(width:1px color:{
                if release { color(accent.key) } else { color(accent.time) }
            })
            font-color:{ if release { color(accent.key) } else { color(accent.time) } }
            hover { fill:port.fill-over }
            @click:{
                if !release && !typing.get_untracked() {
                    buffer.set(wire_dwell_text(model, id, rank));
                    typing.set(true);
                }
            } {
            if release { icon size:13px label-release } else { icon size:13px label-dwell }
            if $typing {
                (editable_field(buffer, Rc::clone(&commit), typing))
            } else {
                text font-size:12px text-wrap:none { wire_text(model, id, rank, release) }
            }
        }
    }
}

/// One state: what the joint does, what starts it, and what ends it.
fn state_card(
    handles: &Handles,
    id: DriveLinkId,
    index: usize,
    draft: State<Option<Draft>>,
) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let selected = handles.selected;
    // A wire being dragged onto this card says so on the card itself, not only
    // by where its far end happens to point.
    let caught = move || draft.get().is_some_and(|draft| draft.target == Some(index));
    let angled = move || angle_mode(model, id, index);
    let top = move || band_top(model, id);
    let left = geometry::card_left(index);
    let face = dial_face(&handles, id, index);
    let cap = keycap(&handles, id, index);
    let header = card_header(&handles, id, index);
    view! {
        col width:204px height:214px align:center
            pad:(left:12px right:12px top:10px bottom:4px)
            translate:(x:(Length::px(left)) y:{ Length::px(top()) })
            radius:14px fill:card.fill
            stroke:(width:{ if caught() { 2.0 } else { 1.0 } } color:{
                if caught() { draft_color(draft.get()) }
                else if selected.get() == Some(id) { color(card.edge_on) }
                else { color(card.edge) }
            })
            shadow:(offset:(x:0px y:12px) blur:26px color:#00000073)
            font-color:{ if angled() { color(accent.angle) } else { color(accent.speed) } } {
            (header)
            (face)
            (cap)
        }
    }
}

/// Whether a state holds an angle rather than spinning.
fn angle_mode(model: State<PanelModel>, id: DriveLinkId, index: usize) -> bool {
    lane_read(model, id, false, |joint| {
        joint
            .states
            .get(index)
            .is_some_and(|state| state.mode == Mode::Angle)
    })
}

/// Where the cards sit, once the wires above them have their lanes.
fn band_top(model: State<PanelModel>, id: DriveLinkId) -> f32 {
    lane_read(model, id, 34.0, |joint| {
        geometry::top_band(joint.release_wires.len())
    })
}

/// Reads one state, or a resting one when the joint has gone.
fn state_of(model: State<PanelModel>, id: DriveLinkId, index: usize) -> StateModel {
    lane_read(model, id, StateModel::resting(), |joint| {
        joint
            .states
            .get(index)
            .cloned()
            .unwrap_or_else(StateModel::resting)
    })
}

/// The card's top bar: which state this is, and the switches that change what
/// kind of state it is.
fn card_header(handles: &Handles, id: DriveLinkId, index: usize) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let slot = u8::try_from(index).unwrap_or(u8::MAX);
    let angled = move || angle_mode(model, id, index);
    let to_angle = handles.clone();
    let to_speed = handles.clone();
    let remove = handles.clone();
    view! {
        row width:fill height:20px align:center justify:between {
            row width:min-content height:20px align:center gap:6px {
                row width:min-content height:20px align:center
                    pad:(horizontal:7px vertical:0px) radius:5px
                    fill:{ if angled() { color(wash.pill_angle) } else { color(wash.pill_speed) } } {
                    text font-size:12px font-weight:700 { format!("S{}", index + 1) }
                }
                if index == 0 {
                    col width:14px height:14px font-color:ink.dim { icon size:14px home }
                }
            }
            row width:min-content height:20px align:center gap:4px {
                col width:26px height:20px align:center justify:center radius:5px
                    fill:{ if angled() { color(wash.mode_angle) } else { Color::TRANSPARENT } }
                    font-color:{ if angled() { color(accent.angle) } else { color(mode.off) } }
                    @click:{ to_angle.edit(id, PanelEdit::SetMode { state: slot, mode: Mode::Angle }) } {
                    icon size:15px mode-angle
                }
                col width:26px height:20px align:center justify:center radius:5px
                    fill:{ if angled() { Color::TRANSPARENT } else { color(wash.mode_speed) } }
                    font-color:{ if angled() { color(mode.off) } else { color(accent.speed) } }
                    @click:{ to_speed.edit(id, PanelEdit::SetMode { state: slot, mode: Mode::Speed }) } {
                    icon size:15px mode-speed
                }
                col width:20px height:20px align:center justify:center radius:5px
                    font-color:accent.danger opacity:0.5
                    hover { opacity:1.0 fill:wash.delete }
                    @click:{ remove.edit(id, PanelEdit::RemoveState { state: slot }) } {
                    icon size:12px delete
                }
            }
        }
    }
}

/// What one dial reads, gathered so the view can ask for it by name.
///
/// `Copy`, because a binding's closure has to be — every field is a handle or
/// an index.
#[derive(Clone, Copy)]
struct Dial {
    model: State<PanelModel>,
    id: DriveLinkId,
    index: usize,
}

impl Dial {
    /// Whether this state holds an angle rather than spinning.
    fn angled(self) -> bool {
        angle_mode(self.model, self.id, self.index)
    }

    /// The colour this dial reads in.
    fn accent(self) -> Color {
        if self.angled() {
            color(accent.angle)
        } else {
            color(accent.speed)
        }
    }

    /// Fastest the joint may turn, in degrees a second.
    fn ceiling(self) -> f32 {
        lane_read(self.model, self.id, 1.0, |joint| joint.speed)
    }

    /// Converts the displayed speed back to the degrees-per-second value the
    /// graph-facing edit seam accepts.
    fn stored_speed(self, displayed: f32) -> f32 {
        match lane_read(self.model, self.id, SpeedUnit::Rpm, |joint| {
            joint.speed_unit
        }) {
            SpeedUnit::Rpm => displayed * 6.0,
            SpeedUnit::DegreesPerSecond => displayed,
        }
    }

    /// How far round the dial the reading sits.
    fn sweep(self) -> f32 {
        state_of(self.model, self.id, self.index).sweep(self.ceiling())
    }

    /// The joint's travel limits, in degrees.
    fn travel(self) -> Option<(f32, f32)> {
        lane_read(self.model, self.id, None, |joint| joint.travel)
    }

    /// The arc from zero to the reading.
    fn span(self) -> Option<(f32, f32)> {
        geometry::arc_span(self.sweep())
    }

    /// The arc covering everywhere the joint may go.
    fn travel_span(self) -> Option<(f32, f32)> {
        self.travel()
            .and_then(|(low, high)| geometry::arc_span_between(low, high))
    }

    /// The arc marking a reading the joint cannot reach.
    fn over_span(self) -> Option<(f32, f32)> {
        geometry::arc_span(if self.sweep() >= 0.0 { 180.0 } else { -180.0 })
    }

    /// Whether the reading asks for more than the joint has.
    fn over(self) -> bool {
        state_of(self.model, self.id, self.index).overspeed(self.ceiling())
    }

    /// Where the handle sits.
    fn head(self) -> (f32, f32) {
        geometry::polar(geometry::DIAL_RADIUS, self.sweep())
    }

    /// Whether the travel handles are shown. Travel belongs to the joint, so
    /// they only appear on a dial that is reading an angle.
    fn grips(self) -> bool {
        self.angled() && self.travel().is_some()
    }

    /// One end of the joint's travel, in degrees.
    fn limit(self, low: bool) -> f32 {
        self.travel()
            .map_or(0.0, |(min, max)| if low { min } else { max })
    }

    /// Where one travel handle sits.
    fn grip(self, low: bool) -> (f32, f32) {
        geometry::polar(geometry::GRIP_RADIUS, self.limit(low))
    }

    /// One end of a travel tick, inside or outside the track.
    fn tick(self, low: bool, outer: bool) -> (f32, f32) {
        geometry::polar(if outer { 62.0 } else { 46.0 }, self.limit(low))
    }
}

/// The dial: where the joint is asked to be, and how far it may go.
///
/// Zero is twelve o'clock and the sweep grows clockwise, so a reading is where
/// the joint would point. A speed reading is a fraction of the joint's own
/// ceiling drawn as half a turn either way, which keeps a fast joint and a slow
/// one legible at the same size.
fn dial_face(handles: &Handles, id: DriveLinkId, index: usize) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let slot = u8::try_from(index).unwrap_or(u8::MAX);
    let reading = Dial { model, id, index };
    // Where the dial landed, so a drag can be read as an angle about its
    // centre rather than as a distance.
    let centre: State<Vector2> = State::new(Vector2::ZERO);
    let readout = dial_readout(&handles, id, index);
    // The grips are built inside the branch that shows them, not adopted into
    // it: travel switches off and on, and closing the branch frees whatever is
    // in it — an element built once out here would be gone the second time
    // round.
    let low = handles.clone();
    let high = handles.clone();
    view! {
        stack width:132px height:132px margin:(top:4px)
            @layout:{ move |bounds: Rect| centre.set(bounds.center()) }
            @drag:{ move |event: &DragEvent, _: &mut EventCtx| {
                let middle = centre.get_untracked();
                let across = event.position.x - middle.x;
                let down = event.position.y - middle.y;
                // A dead zone at the middle: near the centre the angle is all
                // but undefined, and the dial would snap wildly.
                if across.hypot(down) < 14.0 {
                    return;
                }
                let degrees = across.atan2(-down).to_degrees();
                let value = if reading.angled() {
                    let step = if event.modifiers.shift { 1.0 } else { 5.0 };
                    (degrees / step).round() * step
                } else {
                    // Half a turn either way covers the joint's whole range,
                    // in twenty detents, so a fast joint and a slow one read
                    // the same.
                    let top = reading.ceiling();
                    let detent = top / 20.0;
                    let reading = (degrees / 180.0).clamp(-1.0, 1.0) * top;
                    if detent > 0.0 { (reading / detent).round() * detent } else { 0.0 }
                };
                let value = if reading.angled() {
                    value
                } else {
                    reading.stored_speed(value)
                };
                let edit = PanelEdit::SetValue { state: slot, value };
                // Only the move that ends the gesture belongs in history.
                if event.phase == DragPhase::End {
                    handles.edit(id, edit);
                } else {
                    handles.dragging(id, edit);
                }
            } } {
            // Sized, because a canvas left to itself shrinks onto the union
            // of what it draws and pulls that drawing flush with its own
            // corner — which would slide the dial off the number at its centre
            // by however much the sweep happened to reach.
            canvas width:132px height:132px {
                circle at:(x:66px y:66px) radius:54px stroke:(width:13px color:dial.track)
                if reading.travel_span().is_some() && reading.angled() {
                    circle at:(x:66px y:66px) radius:54px
                        arc:(from:{ reading.travel_span().map_or(0.0, |span| span.0) }
                             to:{ reading.travel_span().map_or(0.0, |span| span.1) })
                        stroke:(width:13px color:wash.travel-arc)
                }
                if reading.span().is_some() {
                    circle at:(x:66px y:66px) radius:54px
                        arc:(from:{ reading.span().map_or(0.0, |span| span.0) }
                             to:{ reading.span().map_or(0.0, |span| span.1) })
                        stroke:(width:13px cap:round color:{ reading.accent() })
                }
                if reading.over() {
                    circle at:(x:66px y:66px) radius:63px
                        arc:(from:{ reading.over_span().map_or(0.0, |span| span.0) }
                             to:{ reading.over_span().map_or(0.0, |span| span.1) })
                        stroke:(width:3px color:accent.danger)
                }
                line from:(x:66px y:3px) to:(x:66px y:16px)
                    stroke:(width:2px cap:round color:dial.tick)
                circle at:(x:{ Length::px(reading.head().0) } y:{ Length::px(reading.head().1) })
                    radius:9px fill:dial.knob
                    stroke:(width:3px color:{ reading.accent() })
                if reading.grips() {
                    line from:(x:{ Length::px(reading.tick(true, false).0) }
                               y:{ Length::px(reading.tick(true, false).1) })
                         to:(x:{ Length::px(reading.tick(true, true).0) }
                             y:{ Length::px(reading.tick(true, true).1) })
                        stroke:(width:2px cap:round color:dial.limit)
                    line from:(x:{ Length::px(reading.tick(false, false).0) }
                               y:{ Length::px(reading.tick(false, false).1) })
                         to:(x:{ Length::px(reading.tick(false, true).0) }
                             y:{ Length::px(reading.tick(false, true).1) })
                        stroke:(width:2px cap:round color:dial.limit)
                    circle at:(x:{ Length::px(reading.grip(true).0) }
                              y:{ Length::px(reading.grip(true).1) })
                        radius:6.5px fill:dial.knob stroke:(width:2px color:dial.grip)
                    circle at:(x:{ Length::px(reading.grip(false).0) }
                              y:{ Length::px(reading.grip(false).1) })
                        radius:6.5px fill:dial.knob stroke:(width:2px color:dial.grip)
                }
            }
            (readout)
            if reading.grips() {
                (limit_grip(&low, id, true, centre))
                (limit_grip(&high, id, false, centre))
            }
        }
    }
}

/// A handle for one end of the joint's travel.
///
/// Invisible: the grip itself is drawn on the dial, and this is the box that
/// takes the drag. Keeping the two apart is what lets the drawn grip sit
/// outside the dial's track without the dial having to grow.
fn limit_grip(handles: &Handles, id: DriveLinkId, low: bool, pivot: State<Vector2>) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let travel = move || lane_read(model, id, None, |joint| joint.travel);
    let at = move || {
        let degrees = travel().map_or(0.0, |(min, max)| if low { min } else { max });
        geometry::polar(geometry::GRIP_RADIUS, degrees)
    };
    view! {
        col width:18px height:18px
            translate:(x:{ Length::px(at().0 - 9.0) } y:{ Length::px(at().1 - 9.0) })
            @drag:{ move |event: &DragEvent, _: &mut EventCtx| {
                // The dial underneath reads the same gesture as a change of
                // value, and does not get it: a recognizer that has taken a
                // gesture stops it reaching an ancestor's.
                let middle = pivot.get_untracked();
                let across = event.position.x - middle.x;
                let down = event.position.y - middle.y;
                if across.hypot(down) < 14.0 {
                    return;
                }
                let step = if event.modifiers.shift { 1.0 } else { 5.0 };
                let degrees = (across.atan2(-down).to_degrees() / step).round() * step;
                let Some((min, max)) = travel() else { return };
                let edit = if low {
                    PanelEdit::SetTravel { min: degrees, max }
                } else {
                    PanelEdit::SetTravel { min, max: degrees }
                };
                if event.phase == DragPhase::End {
                    handles.edit(id, edit);
                } else {
                    handles.dragging(id, edit);
                }
            } } {}
    }
}

/// The number at the centre of the dial, and what it is measured in.
fn dial_readout(handles: &Handles, id: DriveLinkId, index: usize) -> Element {
    let model = handles.model;
    let angled = move || angle_mode(model, id, index);
    let value = move || state_of(model, id, index).value;
    let unit = move || lane_read(model, id, "RPM", LaneModel::speed_unit_text);
    view! {
        col width:fill height:fill align:center justify:center nohit {
            text font-size:21px font-weight:700
                font-color:{ if angled() { color(accent.angle) } else { color(accent.speed) } } {
                if angled() { format!("{:.0}°", value()) } else { format!("{:.0}", value()) }
            }
            text font-size:11px font-color:ink.faint margin:(top:2px) {
                if angled() { "degrees" } else { unit() }
            }
        }
    }
}

/// The key that puts the joint in this state.
fn keycap(handles: &Handles, id: DriveLinkId, index: usize) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let capturing = handles.capturing;
    let slot = u8::try_from(index).unwrap_or(u8::MAX);
    let bound = move || state_of(model, id, index).key;
    let arming = move || capturing.get() == Some((id, slot));
    let clear = handles.clone();
    view! {
        row width:min-content height:34px min-width:46px align:center justify:center gap:6px
            pad:(horizontal:10px vertical:0px) radius:8px margin:(top:6px)
            fill:{
                if arming() { color(wash.capturing) }
                else if bound().is_some() { color(accent.key) }
                else { Color::TRANSPARENT }
            }
            stroke:(width:1px color:{
                if arming() || bound().is_some() { color(accent.key) } else { color(key.edge_off) }
            })
            font-color:{
                if arming() { color(accent.key) }
                else if bound().is_some() { color(key.ink_on) }
                else { color(key.ink_off) }
            }
            hover { stroke:(width:1px color:accent.key) }
            @click:{
                if bound().is_some() {
                    clear.edit(id, PanelEdit::ClearKey { state: slot });
                    capturing.set(None);
                } else if arming() {
                    capturing.set(None);
                } else {
                    capturing.set(Some((id, slot)));
                }
            } {
            if bound().is_none() && !arming() {
                icon size:18px keyboard
            } else {
                text font-size:15px font-weight:700 {
                    if arming() {
                        "…".to_owned()
                    } else {
                        bound().map(String::from).unwrap_or_default()
                    }
                }
            }
        }
    }
}

/// What happens when the key goes up: stay put, or hand off to another state.
fn release_port(
    handles: &Handles,
    id: DriveLinkId,
    index: usize,
    frame: State<Vector2>,
    draft: State<Option<Draft>>,
) -> Element {
    let handles = handles.clone();
    let dragging = handles.clone();
    let model = handles.model;
    let slot = u8::try_from(index).unwrap_or(u8::MAX);
    let keyed = move || state_of(model, id, index).key.is_some();
    let latched = move || state_of(model, id, index).release.is_none();
    let left = geometry::card_center_x(index) + geometry::NODE_W / 2.0 - 11.0;
    let top = move || band_top(model, id) - 11.0;
    // A drag retargets the wire; a click without one steps through the
    // choices. The flag is what keeps a drag from doing both.
    let dragged: State<bool> = State::new(false);
    view! {
        col width:22px height:22px align:center justify:center
            translate:(x:(Length::px(left)) y:{ Length::px(top()) })
            radius:11px fill:port.fill
            stroke:(width:1px color:{
                if !keyed() { color(port.off) }
                else if latched() { color(port.idle) }
                else { color(accent.key) }
            })
            font-color:{
                if !keyed() { color(port.off) }
                else if latched() { color(port.idle) }
                else { color(accent.key) }
            }
            opacity:{ if keyed() { 1.0 } else { 0.35 } }
            hover { fill:port.fill-over }
            @drag:{ move |event: &DragEvent, _: &mut EventCtx| {
                dragged.set(true);
                let at = event.position - frame.get_untracked();
                let cards = lane_read(model, id, 0, |joint| joint.states.len());
                let target = geometry::card_at((at.x, at.y), band_top(model, id), cards);
                if event.phase != DragPhase::End {
                    // The wire follows the pointer from the first move, and
                    // catches on whatever card it is over.
                    draft.set(Some(Draft {
                        source: index,
                        release: true,
                        at: (at.x, at.y),
                        target,
                    }));
                    return;
                }
                draft.set(None);
                // Dropped on nothing, the state simply stays where it is.
                let target = target.and_then(|found| u8::try_from(found).ok());
                dragging.edit(id, PanelEdit::SetRelease { state: slot, target });
            } }
            @click:{
                if dragged.get_untracked() {
                    dragged.set(false);
                } else {
                    handles.edit(id, PanelEdit::CycleRelease { state: slot });
                }
            } {
            if latched() { icon size:12px port-latch } else { icon size:12px port-linked }
        }
    }
}

/// How long the state waits before handing off by itself.
fn dwell_port(
    handles: &Handles,
    id: DriveLinkId,
    index: usize,
    frame: State<Vector2>,
    draft: State<Option<Draft>>,
) -> Element {
    let handles = handles.clone();
    let dragging = handles.clone();
    let model = handles.model;
    let slot = u8::try_from(index).unwrap_or(u8::MAX);
    let waiting = move || state_of(model, id, index).dwell.is_some();
    let dragged: State<bool> = State::new(false);
    let left = geometry::card_center_x(index) + geometry::NODE_W / 2.0 - 11.0;
    let top = move || band_top(model, id) + geometry::NODE_H - 11.0;
    view! {
        col width:22px height:22px align:center justify:center
            translate:(x:(Length::px(left)) y:{ Length::px(top()) })
            radius:11px fill:port.fill
            stroke:(width:1px color:{
                if waiting() { color(accent.time) } else { color(port.idle) }
            })
            font-color:{ if waiting() { color(accent.time) } else { color(port.idle) } }
            opacity:{ if waiting() { 1.0 } else { 0.5 } }
            hover { fill:port.fill-over }
            @drag:{ move |event: &DragEvent, _: &mut EventCtx| {
                dragged.set(true);
                let at = event.position - frame.get_untracked();
                let cards = lane_read(model, id, 0, |joint| joint.states.len());
                let target = geometry::card_at((at.x, at.y), band_top(model, id), cards);
                if event.phase != DragPhase::End {
                    draft.set(Some(Draft {
                        source: index,
                        release: false,
                        at: (at.x, at.y),
                        target,
                    }));
                    return;
                }
                draft.set(None);
                // Dropped on nothing, a timed hand-off has nowhere to go, so
                // the state keeps whatever it had.
                if let Some(target) = target.and_then(|found| u8::try_from(found).ok()) {
                    dragging.edit(id, PanelEdit::SetDwellTarget { state: slot, target });
                }
            } }
            @click:{
                if dragged.get_untracked() {
                    dragged.set(false);
                } else {
                    handles.edit(id, PanelEdit::ToggleDwell { state: slot });
                }
            } {
            icon size:12px port-dwell
        }
    }
}

/// The empty slot after the last card, which adds a state.
fn add_card(handles: &Handles, id: DriveLinkId) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let count = move || lane_read(model, id, 0, |joint| joint.states.len());
    let top = move || band_top(model, id);
    let left = move || Length::px(geometry::add_card_left(count()));
    view! {
        col width:204px height:214px align:center justify:center
            translate:(x:{ left() } y:{ Length::px(top()) })
            radius:14px stroke:(width:1px color:add.edge) opacity:0.6
            hover { stroke:(width:1px color:accent.key) fill:add.fill-over opacity:1.0 }
            @click:{ handles.edit(id, PanelEdit::AddState) } {
            canvas width:132px height:132px {
                circle at:(x:66px y:66px) radius:54px stroke:(width:13px color:add.ring)
                line from:(x:66px y:53px) to:(x:66px y:79px)
                    stroke:(width:2.4px cap:round color:ink.dim)
                line from:(x:53px y:66px) to:(x:79px y:66px)
                    stroke:(width:2.4px cap:round color:ink.dim)
            }
        }
    }
}

/// The joint's name, edited where it is read.
///
/// Committed on Enter rather than on every keystroke: a commit rewrites the
/// model the field is bound to, and doing that mid-word would fight the person
/// typing.
fn name_field(handles: &Handles, id: DriveLinkId) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let editing: State<bool> = State::new(false);
    let buffer: State<String> = State::new(String::new());
    let commit: Rc<dyn Fn()> = Rc::new({
        let handles = handles.clone();
        move || {
            handles.edit(id, PanelEdit::SetName(buffer.get_untracked()));
            editing.set(false);
        }
    });
    view! {
        row width:1fr height:min-content align:center
            @click:{
                if !editing.get_untracked() {
                    buffer.set(lane_read(model, id, String::new(), |joint| joint.name.clone()));
                    editing.set(true);
                }
            } {
            if $editing {
                (editable_field(buffer, Rc::clone(&commit), editing))
            } else {
                text width:1fr font-size:15px font-weight:600 text-wrap:none {
                    lane_read(model, id, String::new(), LaneModel::title)
                }
            }
        }
    }
}

/// What the attached hardware lets the joint do, plus its program switches.
fn chips(handles: &Handles, id: DriveLinkId) -> Element {
    let speed = capability_chip(handles, id, Chip::Speed);
    let actuator = capability_chip(handles, id, Chip::Actuator);
    let electric = capability_chip(handles, id, Chip::Electric);
    let gas = capability_chip(handles, id, Chip::Gas);
    let travel = switch_chip(handles, id, Chip::Travel);
    let repeat = switch_chip(handles, id, Chip::Repeat);
    view! {
        grid height:min-content cols:(repeat(2 minmax(0px 1fr))) col-gap:6px row-gap:6px {
            (speed)
            (actuator)
            (electric)
            (gas)
            (travel)
            (repeat)
        }
    }
}

/// Which property a chip stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Chip {
    /// Fastest the joint may turn.
    Speed,
    /// Assigned actuator family and available torque.
    Actuator,
    /// Electric motor contribution.
    Electric,
    /// Gas motor contribution.
    Gas,
    /// How far it may turn.
    Travel,
    /// Whether the sequence repeats.
    Repeat,
}

/// A hardware capability chip. Clicking speed toggles its unit; the remaining
/// chips cycle assignments that can actually be supplied by the machine.
fn capability_chip(handles: &Handles, id: DriveLinkId, which: Chip) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let edit = match which {
        Chip::Speed => PanelEdit::ToggleSpeedUnit,
        Chip::Actuator => PanelEdit::CycleActuator,
        Chip::Electric => PanelEdit::CycleElectric,
        Chip::Gas => PanelEdit::CycleGas,
        Chip::Travel | Chip::Repeat => unreachable!("switches use switch_chip"),
    };

    view! {
        row height:44px align:center gap:8px pad:(horizontal:9px vertical:0px) radius:8px
            fill:chip.fill stroke:(width:1px color:chip.edge)
            font-color:{ if which == Chip::Speed { color(chip.speed) } else { color(chip.torque) } }
            hover { stroke:(width:1px color:chip.edge-over) }
            @click:{ handles.edit(id, edit.clone()) } {
            if which == Chip::Speed { icon size:20px chip-speed } else { icon size:20px chip-torque }
            text width:1fr font-size:12px letter-spacing:-0.12px text-wrap:none {
                if which == Chip::Speed {
                    lane_read(model, id, String::new(), LaneModel::speed_text)
                } else if which == Chip::Actuator {
                    lane_read(model, id, String::new(), LaneModel::torque_text)
                } else if which == Chip::Electric {
                    lane_read(model, id, String::new(), LaneModel::electric_text)
                } else {
                    lane_read(model, id, String::new(), LaneModel::gas_text)
                }
            }
        }
    }
}

/// A chip that flips between two settled states.
fn switch_chip(handles: &Handles, id: DriveLinkId, which: Chip) -> Element {
    let handles = handles.clone();
    let model = handles.model;
    let travel = which == Chip::Travel;
    let on = move || {
        if travel {
            lane_read(model, id, false, |joint| joint.travel.is_some())
        } else {
            lane_read(model, id, false, |joint| joint.loops)
        }
    };

    view! {
        row height:44px align:center gap:8px pad:(horizontal:9px vertical:0px) radius:8px
            fill:{
                if !on() { color(chip.fill) }
                else if travel { color(wash.chip_travel) }
                else { color(wash.chip_loop) }
            }
            stroke:(width:1px color:{
                if !on() { color(chip.edge) }
                else if travel { color(chip.travel_edge) }
                else { color(chip.loop_edge) }
            })
            font-color:{
                if !on() { color(ink.muted) }
                else if travel { color(accent.angle) }
                else { color(accent.time) }
            }
            hover { stroke:(width:1px color:chip.edge-over) }
            @click:{
                handles.edit(id, if travel { PanelEdit::ToggleTravel } else { PanelEdit::ToggleLoop });
            } {
            if travel {
                if on() {
                    icon size:20px chip-travel-limited
                } else {
                    icon size:20px chip-travel-free
                }
            } else {
                if on() { icon size:20px chip-loop } else { icon size:20px chip-once }
            }
            text width:1fr font-size:12px letter-spacing:-0.12px text-wrap:none {
                if travel {
                    lane_read(model, id, String::new(), LaneModel::travel_text)
                } else {
                    lane_read(model, id, String::new(), |joint| joint.loop_text().to_owned())
                }
            }
        }
    }
}

/// One-click starting points, so a joint does not have to be programmed from
/// nothing to do the obvious thing.
fn presets(handles: &Handles, id: DriveLinkId) -> Element {
    let steer = preset_button(handles, id, Preset::Steer);
    let drive = preset_button(handles, id, Preset::Drive);
    let spin = preset_button(handles, id, Preset::Spin);
    view! {
        row height:min-content gap:6px margin:(top:2px) {
            (steer)
            (drive)
            (spin)
        }
    }
}

/// One ready-made program.
fn preset_button(handles: &Handles, id: DriveLinkId, which: Preset) -> Element {
    let handles = handles.clone();
    let glyph = match which {
        Preset::Steer => view! { icon size:22px preset-steer },
        Preset::Drive => view! { icon size:22px preset-drive },
        Preset::Spin => view! { icon size:22px preset-spin },
    };
    view! {
        col width:1fr height:34px align:center justify:center radius:8px
            stroke:(width:1px color:preset.edge) font-color:ink.muted
            hover { stroke:(width:1px color:accent.key) fill:preset.fill-over }
            @click:{ handles.edit(id, PanelEdit::ApplyPreset(which)) } {
            (glyph)
        }
    }
}

/// A number being typed, which commits on Enter and backs out on Escape.
fn editable_field(buffer: State<String>, commit: Rc<dyn Fn()>, typing: State<bool>) -> Element {
    view! {
        row width:1fr height:24px align:center
            @key:{ move |event: &KeyEvent, ctx: &mut EventCtx| {
                if !matches!(event.kind, KeyEventKind::Down { .. }) { return; }
                match event.key {
                    Key::Enter => { commit(); ctx.stop_propagation(); }
                    Key::Escape => { typing.set(false); ctx.stop_propagation(); }
                    _ => {}
                }
            } } {
            input width:1fr font-size:12px fill:#00000000 pad:(horizontal:0px vertical:0px)
                stroke:(width:0px color:#00000000) buffer
        }
    }
}
