//! The Gear tool: teeth on cylinders, racks along blocks, new gears on bare
//! bearings, and dragging between two toothed parts to mesh them.

use crate::builder::gears::{
    GearDimension, GearSeat, GearSettings, GearTarget, RackTarget, ReachChoice, choose_reach,
    face_plane_hit, gear_target_from_hit, is_meshable, mesh_label, mesh_summary, rack_cut,
    rack_run, rack_target_from_hit, reach_line, reaches, stage_gear, stage_mesh,
    stage_meshes_admitted, stage_rack, stage_rack_run, stage_unmesh, toothed, unracked, untoothed,
    validate_gear,
};
use crate::builder::{
    CylinderPlacementCandidate, PlacementBounds, PlacementError, SurfaceHit,
    center_cylinder_candidate_on_bearing, cylinder_candidate_from_hit_with_grid,
    stage_bearing_cylinder_in_bounds, validate_cylinder_candidate_in_bounds,
};
use crate::controls::GameAction;
use crate::editor::build_actions::bearing_socket_targets;
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::hover::clear_hover;
use crate::editor::raycast::hovered_part;
use crate::editor::state::EditorState;
use bevy::prelude::ButtonInput;
use mechanic_core::{
    ConstructionFrame, ConstructionGraph, ConstructionMaterial, CylinderDimensions, CylinderSpec,
    GRID_UNIT_METERS, GearKind, GearLinkSpec, GearMesh, JointKind, MaterialAppearance, PartId,
    PartSpec,
};

/// What the tool is pointed at and what a click would make of it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GearPreview {
    /// Teeth on the cylinder under the cursor.
    Teeth {
        target: GearTarget,
        spec: CylinderSpec,
    },
    /// Rack teeth along the block face under the cursor.
    Rack { target: RackTarget },
    /// A new toothed cylinder on the bare bearing under the cursor.
    Socket {
        index: usize,
        candidate: CylinderPlacementCandidate,
    },
}

/// A mesh being dragged out from one toothed part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MeshDrag {
    pub(crate) from: PartId,
    /// The pointer was released back on `from`, so the mesh waits for a
    /// second click instead of a drag.
    pub(crate) armed: bool,
}

/// A rack being dragged out across the blocks of one face.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RackDrag {
    pub(crate) start: RackTarget,
    /// The blocks under the drag so far, the start block among them.
    pub(crate) run: Vec<RackTarget>,
}

/// Where a new gear's partner is being chosen: on a bearing, or on a plain
/// cylinder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReachAnchor {
    Socket(usize),
    Cylinder(PartId),
}

/// The Gear tool's state.
#[derive(Clone, Debug, Default)]
pub(crate) struct GearTool {
    pub(crate) settings: GearSettings,
    pub(crate) preview: Option<GearPreview>,
    /// What the player was last told, so it is said once.
    pub(crate) announced: Option<String>,
    pub(crate) drag: Option<MeshDrag>,
    pub(crate) rack_drag: Option<RackDrag>,
    /// Which of the partners in reach the new gear fits, stepped by Rotate,
    /// and where that was chosen; it starts over on another seat.
    pub(crate) reach_choice: usize,
    pub(crate) reach_anchor: Option<ReachAnchor>,
}

/// The partner choice for `anchor`: the one made there, or the first on a
/// new seat.
fn reach_choice(state: &mut EditorState, anchor: ReachAnchor) -> usize {
    if state.gears.reach_anchor != Some(anchor) {
        state.gears.reach_anchor = Some(anchor);
        state.gears.reach_choice = 0;
    }
    state.gears.reach_choice
}

/// What the status line adds when more than one partner is in reach.
fn choice_suffix(chosen: Option<ReachChoice>) -> String {
    match chosen {
        Some(choice) if choice.count > 1 => {
            format!(
                " ({} of {}; Rotate for the next)",
                choice.index + 1,
                choice.count
            )
        }
        _ => String::new(),
    }
}

fn announce(state: &mut EditorState, line: String) {
    if state.gears.announced.as_deref() != Some(line.as_str()) {
        state.gears.announced = Some(line.clone());
        state.feedback = Some(line);
    }
}

/// Works out what the settings make of whatever is under the cursor: a new
/// gear on a bare bearing, teeth on a cylinder, or a rack along a block face.
/// Returns why they cannot, if they cannot.
pub(crate) fn hover(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) -> Option<PlacementError> {
    if let Some(start) = state.gears.rack_drag.as_ref().map(|drag| drag.start) {
        // The drag covers whatever the pointer stands over on the start
        // face's plane, block under it or not.
        if let Some((origin, direction)) = state.pointer_ray
            && let Some(far) = face_plane_hit(&start, origin, direction)
        {
            let run = rack_run(graph, &start, far);
            if let Some(drag) = state.gears.rack_drag.as_mut() {
                drag.run = run;
            }
        }
        let run = state
            .gears
            .rack_drag
            .as_ref()
            .map(|drag| drag.run.clone())
            .unwrap_or_default();
        state.gears.preview = Some(GearPreview::Rack { target: start });
        return match rack_cut(state.gears.settings, &run) {
            Ok(cuts) => {
                announce(
                    state,
                    format!("Release: {}", state.gears.settings.rack_summary(&cuts)),
                );
                None
            }
            Err(error) => Some(error),
        };
    }
    if let Some(index) = state.hovered_bearing
        && let Some(socket) = state.placed_bearings.get(index).copied()
        && socket.kind == JointKind::Rotational
    {
        let choice = reach_choice(state, ReachAnchor::Socket(index));
        return match gear_on_socket(graph, state, index, choice, material, appearance) {
            Ok((candidate, chosen)) => {
                state.attachment_bearing = Some(index);
                state.cylinder_preview = Some(candidate);
                state.gears.preview = Some(GearPreview::Socket { index, candidate });
                let line = socket_line(graph, state.gears.settings, candidate, chosen);
                announce(state, line);
                validate_cylinder_candidate_in_bounds(graph, candidate, state.placement_bounds)
                    .err()
            }
            Err(error) => Some(error),
        };
    }
    let hit = state.hovered?;
    if let Ok(target) = gear_target_from_hit(graph, hit) {
        if target.spec.spiral().is_some() {
            // A worm keeps its thread; it can still be dragged into a mesh.
            announce(state, format!("Worm: {}", mesh_summary(graph, target.part)));
            return None;
        }
        let choice = reach_choice(state, ReachAnchor::Cylinder(target.part));
        let (settings, chosen) = settings_for(
            graph,
            state.gears.settings,
            &target,
            choice,
            state.placement_bounds,
        );
        return match toothed(settings, &target) {
            Ok(spec) => {
                let error = validate_gear(graph, &target, spec, state.placement_bounds).err();
                state.gears.preview = Some(GearPreview::Teeth { target, spec });
                if error.is_none() {
                    announce(state, teeth_line(graph, &target, settings, chosen));
                }
                error
            }
            Err(error) => Some(error),
        };
    }
    match rack_target_from_hit(graph, hit) {
        Ok(target) => match rack_cut(state.gears.settings, &[target]) {
            Ok(cuts) => {
                state.gears.preview = Some(GearPreview::Rack { target });
                announce(
                    state,
                    if target.spec.rack().is_some() {
                        format!("Rack: {}", mesh_summary(graph, target.part))
                    } else {
                        format!(
                            "Click, or drag across blocks: {}",
                            state.gears.settings.rack_summary(&cuts)
                        )
                    },
                );
                None
            }
            Err(error) => Some(error),
        },
        Err(error) => Some(error),
    }
}

// What pointing at a bare bearing says: the gear a click puts there and what
// it meshes, or just misses.
fn socket_line(
    graph: &ConstructionGraph,
    settings: GearSettings,
    candidate: CylinderPlacementCandidate,
    chosen: Option<ReachChoice>,
) -> String {
    let placed = placed_settings(settings, candidate.spec);
    let seat = GearSeat::of_cylinder(candidate.spec, ConstructionFrame::IDENTITY);
    let line = format!("Click: {} on this bearing", placed.summary());
    match chosen {
        Some(choice) => format!(
            "{line}; {}{}",
            reach_line(graph, placed, seat, choice.reach),
            choice_suffix(chosen)
        ),
        None => line,
    }
}

// The settings a click cuts into the target: as set by hand, or as the part
// already has them, or else reaching the `choice`th toothed neighbour whose
// teeth fit on the cylinder, or else as many teeth as the cylinder holds.
fn settings_for(
    graph: &ConstructionGraph,
    settings: GearSettings,
    target: &GearTarget,
    choice: usize,
    bounds: PlacementBounds,
) -> (GearSettings, Option<ReachChoice>) {
    if settings.fitted || target.spec.gear().is_some() {
        return (settings.fitted_to(target.spec), None);
    }
    let seat = GearSeat::of_cylinder(target.spec, target.frame);
    let reaching = |reach: &crate::builder::gears::Reach| GearSettings {
        kind: settings.kind,
        ..reach.fitted
    };
    let chosen = choose_reach(
        reaches(
            graph,
            settings.suited_to(target.spec),
            seat,
            Some(target.part),
        ),
        choice,
        |reach| {
            toothed(reaching(reach), target)
                .is_ok_and(|spec| validate_gear(graph, target, spec, bounds).is_ok())
        },
    );
    (
        chosen.map_or_else(
            || settings.fitted_to(target.spec),
            |choice| reaching(&choice.reach),
        ),
        chosen,
    )
}

// What pointing at a cylinder says: its teeth and meshes, or what a click
// cuts, and either way what its teeth reach or just miss.
fn teeth_line(
    graph: &ConstructionGraph,
    target: &GearTarget,
    settings: GearSettings,
    chosen: Option<ReachChoice>,
) -> String {
    let seat = GearSeat::of_cylinder(target.spec, target.frame);
    let (line, cut) = if let Some(gear) = target.spec.gear() {
        let line = format!(
            "{}-tooth {} gear: {}",
            gear.teeth(),
            crate::builder::gears::kind_label(gear.kind()),
            mesh_summary(graph, target.part)
        );
        if graph.part_gear_links(target.part).next().is_some() {
            return line;
        }
        (line, placed_settings(settings, target.spec))
    } else {
        let cut = settings.suited_to(target.spec);
        (format!("Click: {}", cut.summary()), cut)
    };
    let reach = chosen.map(|choice| choice.reach).or_else(|| {
        reaches(graph, cut, seat, Some(target.part))
            .into_iter()
            .next()
    });
    reach.map_or_else(
        || line.clone(),
        |reach| {
            format!(
                "{line}; {}{}",
                reach_line(graph, cut, seat, reach),
                choice_suffix(chosen)
            )
        },
    )
}

// The settings as a toothed cylinder carries them, keeping the tool's choices
// the teeth do not record.
fn placed_settings(settings: GearSettings, spec: CylinderSpec) -> GearSettings {
    settings
        .picked_from(PartSpec::Cylinder(spec))
        .map_or(settings, |placed| GearSettings {
            fitted: settings.fitted,
            across: settings.across,
            ..placed
        })
}

// A one-block-long gear centred on a bare bearing, its outer wall at the
// teeth's tips. Unless the tooth count is set by hand, the teeth reach the
// `choice`th toothed neighbour the gear can be placed against.
fn gear_on_socket(
    graph: &ConstructionGraph,
    state: &EditorState,
    index: usize,
    choice: usize,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) -> Result<(CylinderPlacementCandidate, Option<ReachChoice>), PlacementError> {
    let socket = state.placed_bearings[index];
    let mut settings = state.gears.settings;
    // A new gear is solid, so it takes external teeth.
    if settings.kind == GearKind::Internal {
        settings.kind = GearKind::Spur;
    }
    let candidate_for = |settings: GearSettings| {
        let gear = settings.gear()?;
        let dimensions = CylinderDimensions::new(gear.tip_diameter(), 0.0, GRID_UNIT_METERS)
            .map_err(|error| PlacementError::Graph(error.to_string()))?;
        let hit = SurfaceHit {
            distance: 0.0,
            point: socket.anchor,
            face: socket.source,
        };
        let candidate = cylinder_candidate_from_hit_with_grid(
            graph,
            hit,
            dimensions,
            state.placement_grid,
            state.placement_bounds,
        )?;
        let mut candidate = center_cylinder_candidate_on_bearing(candidate, socket.anchor);
        candidate.spec = candidate
            .spec
            .with_material(material)
            .with_appearance(appearance)
            .with_gear(gear)
            .map_err(|error| PlacementError::Graph(error.to_string()))?;
        Ok::<_, PlacementError>(candidate)
    };
    let candidate = candidate_for(settings)?;
    // The seat does not move with the diameter, so one pass finds the teeth.
    let seat = GearSeat::of_cylinder(candidate.spec, ConstructionFrame::IDENTITY);
    if settings.fitted {
        let chosen = choose_reach(reaches(graph, settings, seat, None), choice, |_| true);
        return Ok((candidate, chosen));
    }
    let reaching = |reach: &crate::builder::gears::Reach| GearSettings {
        kind: settings.kind,
        ..reach.fitted
    };
    let chosen = choose_reach(reaches(graph, settings, seat, None), choice, |reach| {
        candidate_for(reaching(reach)).is_ok_and(|candidate| {
            validate_cylinder_candidate_in_bounds(graph, candidate, state.placement_bounds).is_ok()
        })
    });
    match chosen {
        Some(choice) => Ok((candidate_for(reaching(&choice.reach))?, chosen)),
        None => Ok((candidate, None)),
    }
}

// Keys that change the settings. Rotate turns a rack across, unless it is
// stepping the partner choice.
fn adjusted(
    actions: &ButtonInput<GameAction>,
    mut settings: GearSettings,
    rotate_chooses: bool,
) -> GearSettings {
    let coarse = !actions.pressed(GameAction::FinePlacement);
    for (decrease, increase, dimension) in [
        (
            GameAction::CylinderOuterDecrease,
            GameAction::CylinderOuterIncrease,
            GearDimension::Module,
        ),
        (
            GameAction::CylinderLengthDecrease,
            GameAction::CylinderLengthIncrease,
            GearDimension::Teeth,
        ),
    ] {
        let direction =
            i8::from(actions.just_pressed(increase)) - i8::from(actions.just_pressed(decrease));
        if direction != 0 {
            settings = settings.adjusted(dimension, direction, coarse);
        }
    }
    if actions.just_pressed(GameAction::PipeTurn) {
        settings = settings.cycled();
    }
    if actions.just_pressed(GameAction::Rotate) && !rotate_chooses {
        settings.across = !settings.across;
    }
    settings
}

/// The meshable part under the cursor: one with teeth, a rack, or a thread.
fn meshable_under_cursor(graph: &ConstructionGraph, state: &EditorState) -> Option<PartId> {
    let part =
        hovered_part(state.hovered).or_else(|| state.hovered_simulation.map(|hit| hit.part))?;
    graph
        .part(part)
        .is_some_and(|spec| is_meshable(*spec))
        .then_some(part)
}

#[expect(
    clippy::too_many_lines,
    reason = "one ordered pass over keys, secondary, the mesh drag and the click"
)]
/// Click a cylinder to cut teeth into it, press a block face and release
/// across the blocks a rack should run over, or click a bare bearing to put
/// a gear on it. Changing a setting while pointing at a toothed part reshapes
/// it at once. Press one toothed part and release on another to mesh them; a
/// click on a toothed part arms the mesh for a second click. Secondary takes
/// teeth off, or breaks a part's meshes.
pub(crate) fn handle_gear_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let under = meshable_under_cursor(graph, state);
    if actions.just_pressed(GameAction::Interact)
        && let Some(picked) = hovered_part(state.hovered)
            .and_then(|part| graph.part(part).copied())
            .and_then(|spec| state.gears.settings.picked_from(spec))
    {
        state.gears.settings = picked;
        state.feedback = Some(format!("Picked up: {}", picked.summary()));
        return;
    }
    // Not Clear Pipette, which puts the held tool away.
    if actions.just_pressed(GameAction::ShapeSnap) {
        state.gears.settings = GearSettings::default();
        state.feedback = Some("Gear settings reset; the next cylinder sizes them".to_owned());
        return;
    }
    // On a bearing or a plain cylinder, Rotate steps to the next partner in
    // reach; the hover then reads the choice and says what it fits.
    let choosing = match state.gears.preview {
        Some(GearPreview::Socket { .. }) => true,
        Some(GearPreview::Teeth { target, .. }) => target.spec.gear().is_none(),
        _ => false,
    } && !state.gears.settings.fitted;
    if choosing && actions.just_pressed(GameAction::Rotate) {
        state.gears.reach_choice = state.gears.reach_choice.wrapping_add(1);
        state.gears.announced = None;
    }
    let before = state.gears.settings;
    let settings = adjusted(actions, before, choosing);
    let changed = settings != before;
    state.gears.settings = settings;
    if changed {
        state.feedback = Some(settings.summary());
    }

    if actions.just_pressed(GameAction::Secondary) {
        if state.gears.drag.take().is_some() {
            state.feedback = Some("Mesh cancelled".to_owned());
            return;
        }
        if state.gears.rack_drag.take().is_some() {
            state.feedback = Some("Rack cancelled".to_owned());
            return;
        }
        match state.gears.preview {
            Some(GearPreview::Teeth { target, .. }) if target.spec.gear().is_some() => {
                let plain = untoothed(&target);
                let staged = validate_gear(graph, &target, plain, state.placement_bounds)
                    .and_then(|()| stage_gear(graph, &target, plain, state.placement_bounds));
                commit(graph, state, history, staged, "Teeth removed");
            }
            Some(GearPreview::Rack { target, .. }) if target.spec.rack().is_some() => {
                let staged = stage_rack(graph, &target, unracked(&target));
                commit(graph, state, history, staged, "Rack removed");
            }
            _ => match under {
                Some(part) if graph.part_gear_links(part).next().is_some() => {
                    let staged = stage_unmesh(graph, part).map(|(graph, count)| {
                        (
                            graph,
                            format!(
                                "Unmeshed {count} mesh{}",
                                if count == 1 { "" } else { "es" }
                            ),
                        )
                    });
                    commit_named(graph, state, history, staged);
                }
                _ => {
                    state.feedback = Some(
                        "Point at teeth to take them off, or at a meshed part to unmesh it"
                            .to_owned(),
                    );
                }
            },
        }
        return;
    }

    let pressed = actions.just_pressed(GameAction::Primary);
    let released = actions.just_released(GameAction::Primary);
    if released && let Some(drag) = state.gears.rack_drag.take() {
        let cuts = rack_cut(settings, &drag.run);
        let done = cuts
            .as_ref()
            .map(|cuts| settings.rack_summary(cuts))
            .unwrap_or_default();
        let staged = cuts.and_then(|cuts| stage_rack_run(graph, &cuts));
        commit(graph, state, history, staged, &done);
        return;
    }
    if let Some(drag) = state.gears.drag {
        match under {
            Some(other) if other != drag.from && (pressed || released) => {
                state.gears.drag = None;
                let staged = stage_mesh(graph, drag.from, other).map(|(graph, mesh)| {
                    let link = GearLinkSpec {
                        first: drag.from,
                        second: other,
                    };
                    let done = format!("Meshed with {}", mesh_label(&graph, link, drag.from, mesh));
                    (graph, done)
                });
                commit_named(graph, state, history, staged);
                return;
            }
            Some(_) if released && !drag.armed => {
                state.gears.drag = Some(MeshDrag {
                    armed: true,
                    ..drag
                });
                state.feedback = Some(
                    "Now click another toothed part to mesh, or click again to re-cut".to_owned(),
                );
                return;
            }
            Some(_) if pressed => {
                // A second click on the same part: re-cut it with the settings.
                state.gears.drag = None;
            }
            None if pressed => {
                state.gears.drag = None;
            }
            None if released && !drag.armed => {
                state.gears.drag = None;
                return;
            }
            _ => {}
        }
    }
    let carries_teeth = matches!(
        state.gears.preview,
        Some(GearPreview::Teeth { target, .. }) if target.spec.gear().is_some()
    ) || matches!(
        state.gears.preview,
        Some(GearPreview::Rack { target, .. }) if target.spec.rack().is_some()
    );
    if pressed
        && state.gears.drag.is_none()
        && carries_teeth
        && !changed
        && let Some(part) = under
    {
        state.gears.drag = Some(MeshDrag {
            from: part,
            armed: false,
        });
        state.feedback = Some("Drag to another toothed part to mesh them".to_owned());
        return;
    }
    if pressed
        && state.gears.drag.is_none()
        && let Some(worm) = under.filter(|part| {
            graph
                .part(*part)
                .and_then(|spec| spec.as_cylinder())
                .is_some_and(|cylinder| cylinder.spiral().is_some())
        })
    {
        state.gears.drag = Some(MeshDrag {
            from: worm,
            armed: false,
        });
        state.feedback =
            Some("Drag to a gear to mesh it as a worm, or to any part for a nut".to_owned());
        return;
    }
    if !(pressed || (changed && carries_teeth)) {
        return;
    }
    let Some(preview) = state.gears.preview else {
        if pressed {
            state.feedback = Some(state.preview_error.as_ref().map_or_else(
                || "Point at a cylinder, a block face, or a bare bearing".to_owned(),
                ToString::to_string,
            ));
        }
        return;
    };
    match preview {
        GearPreview::Teeth { target, .. } => {
            let (fitted, _) = settings_for(
                graph,
                settings,
                &target,
                state.gears.reach_choice,
                state.placement_bounds,
            );
            state.gears.settings = fitted;
            let done = fitted.suited_to(target.spec).summary();
            let staged = toothed(fitted, &target)
                .and_then(|spec| stage_gear(graph, &target, spec, state.placement_bounds));
            commit(graph, state, history, staged, &done);
        }
        GearPreview::Rack { target, .. } => {
            if pressed && target.spec.rack().is_none() {
                // The rack is cut on release, across every block dragged over.
                state.gears.rack_drag = Some(RackDrag {
                    start: target,
                    run: vec![target],
                });
                state.feedback = Some("Drag across the blocks to rack; release to cut".to_owned());
                return;
            }
            let cuts = rack_cut(settings, &[target]);
            let done = cuts
                .as_ref()
                .map(|cuts| settings.rack_summary(cuts))
                .unwrap_or_default();
            let staged = cuts.and_then(|cuts| stage_rack_run(graph, &cuts));
            commit(graph, state, history, staged, &done);
        }
        GearPreview::Socket { index, candidate } => {
            let Some(socket) = state.placed_bearings.get(index).copied() else {
                state.feedback = Some("Bearing is no longer available".to_owned());
                return;
            };
            let staged = stage_bearing_cylinder_in_bounds(
                graph,
                candidate,
                socket.source,
                socket.anchor,
                socket.dimensions,
                &bearing_socket_targets(graph, socket),
                state.placement_bounds,
            )
            .and_then(|staged| {
                let new_part = staged
                    .parts()
                    .map(|(part, _)| part)
                    .find(|part| graph.part(*part).is_none())
                    .ok_or_else(|| PlacementError::Graph("the gear was not placed".to_owned()))?;
                stage_meshes_admitted(&staged, new_part)
            });
            let placed = placed_settings(settings, candidate.spec);
            state.gears.settings = placed;
            let done = format!("Placed {} on the bearing", placed.summary());
            commit(graph, state, history, staged, &done);
        }
    }
}

fn commit(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    staged: Result<(ConstructionGraph, Vec<(PartId, GearMesh)>), PlacementError>,
    done: &str,
) {
    let named = staged.map(|(staged, partners)| {
        let mut line = done.to_owned();
        if !partners.is_empty() {
            let names = partners
                .iter()
                .map(|(other, mesh)| {
                    let part = staged
                        .gear_links()
                        .find_map(|(_, link)| link.other(*other))
                        .unwrap_or(*other);
                    let pair = GearLinkSpec {
                        first: part,
                        second: *other,
                    };
                    mesh_label(&staged, pair, part, *mesh)
                })
                .collect::<Vec<_>>();
            line = format!("{line}; meshed with {}", names.join(", "));
        }
        (staged, line)
    });
    commit_named(graph, state, history, named);
}

fn commit_named(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    staged: Result<(ConstructionGraph, String), PlacementError>,
) {
    match staged {
        Ok((staged, done)) => {
            let previous = EditorSnapshot::capture(graph, state);
            *graph = staged;
            history.commit(previous);
            state.construction_mesh_dirty = true;
            clear_hover(state);
            state.gears.preview = None;
            state.gears.announced = None;
            state.feedback = Some(done);
        }
        Err(error) => state.feedback = Some(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::build_actions::PlacedBearing;
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef,
        GearSpec, GridRotation,
    };

    fn spawned(outcome: BuildOutcome) -> PartId {
        let BuildOutcome::Spawned(part) = outcome else {
            panic!("spawn expected")
        };
        part
    }

    // A plain cylinder `outer` across, one block long, standing 2 m up.
    fn shaft(graph: &mut ConstructionGraph, outer: f32, x: i32) -> PartId {
        spawned(
            graph
                .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                    CylinderDimensions::new(outer, 0.0, 0.25).unwrap(),
                    BuildPose::from_position_ticks(IVec3::new(x, 800, 0), GridRotation::default()),
                )))
                .unwrap(),
        )
    }

    fn tap(action: GameAction) -> ButtonInput<GameAction> {
        let mut actions = ButtonInput::default();
        actions.press(action);
        actions
    }

    fn release(action: GameAction) -> ButtonInput<GameAction> {
        let mut actions = ButtonInput::default();
        actions.press(action);
        actions.clear();
        actions.release(action);
        actions
    }

    // Aims a real ray at the part from above and lets the tool's hover run as in play.
    fn look(graph: &ConstructionGraph, state: &mut EditorState, x: f32) {
        let origin = Vec3::new(x, 4.0, 0.02);
        state.hovered = crate::builder::raycast_construction(graph, origin, Vec3::NEG_Y);
        state.hovered_bearing = None;
        state.pointer_ray = Some((origin, Vec3::NEG_Y));
        crate::editor::hover::refresh_tool_preview(graph, state, crate::hotbar::Tool::Gear);
    }

    fn gear_of(graph: &ConstructionGraph, part: PartId) -> Option<GearSpec> {
        graph
            .part(part)
            .and_then(|spec| spec.as_cylinder())
            .and_then(CylinderSpec::gear)
    }

    #[test]
    fn a_click_tooths_a_cylinder_and_a_second_gear_snaps_into_mesh() {
        let mut graph = ConstructionGraph::new();
        let pinion = shaft(&mut graph, 0.26, 0);
        let wheel = shaft(&mut graph, 0.38, 120);
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        look(&graph, &mut state, 0.0);
        assert_eq!(state.preview_error, None, "{:?}", state.feedback);
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(
            gear_of(&graph, pinion).unwrap().teeth(),
            24,
            "{:?}",
            state.feedback
        );
        assert_eq!(graph.gear_links().count(), 0);

        look(&graph, &mut state, 0.3);
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(
            gear_of(&graph, wheel).unwrap().teeth(),
            36,
            "{:?}",
            state.feedback
        );
        assert_eq!(graph.gear_links().count(), 1);
        assert!(
            state
                .feedback
                .as_deref()
                .unwrap()
                .contains("meshed with 24-tooth spur"),
            "{:?}",
            state.feedback
        );
        assert_eq!(history.undo.len(), 2);

        // Pointing at the wheel, one tooth fewer reshapes it at once and keeps
        // the mesh: the pitch circles are now 5 mm apart, within a module.
        look(&graph, &mut state, 0.3);
        let mut fine = tap(GameAction::CylinderLengthDecrease);
        fine.press(GameAction::FinePlacement);
        handle_gear_actions(&fine, &mut graph, &mut state, &mut history);
        assert_eq!(
            gear_of(&graph, wheel).unwrap().teeth(),
            35,
            "{:?}",
            state.feedback
        );
        assert_eq!(graph.gear_links().count(), 1);
        // Six fewer leave a 3.5 cm gap, and the mesh goes with it.
        look(&graph, &mut state, 0.3);
        handle_gear_actions(
            &tap(GameAction::CylinderLengthDecrease),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(gear_of(&graph, wheel).unwrap().teeth(), 29);
        assert_eq!(graph.gear_links().count(), 0);
    }

    #[test]
    fn dragging_between_two_gears_meshes_them_and_right_click_unmeshes() {
        let mut graph = ConstructionGraph::new();
        let pinion = shaft(&mut graph, 0.26, 0);
        let wheel = shaft(&mut graph, 0.38, 120);
        for (part, teeth) in [(pinion, 24), (wheel, 36)] {
            let spec = graph.part(part).unwrap().as_cylinder().unwrap();
            let spec = spec
                .with_gear(GearSpec::new(4, teeth, GearKind::Spur).unwrap())
                .unwrap();
            graph.apply(BuildCommand::SetGear { part, spec }).unwrap();
        }
        assert_eq!(graph.gear_links().count(), 0);
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        look(&graph, &mut state, 0.0);
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(
            state.gears.drag,
            Some(MeshDrag {
                from: pinion,
                armed: false
            })
        );
        look(&graph, &mut state, 0.3);
        handle_gear_actions(
            &release(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(graph.gear_links().count(), 1, "{:?}", state.feedback);
        assert_eq!(state.gears.drag, None);

        look(&graph, &mut state, 0.3);
        handle_gear_actions(
            &tap(GameAction::Secondary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert!(
            gear_of(&graph, wheel).is_none(),
            "right-click takes the teeth off"
        );
        assert_eq!(graph.gear_links().count(), 0, "and the mesh with them");
    }

    #[test]
    fn placing_a_gear_on_a_bare_bearing_commits_cylinder_and_teeth_together() {
        let mut graph = ConstructionGraph::new();
        let plate = spawned(
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 1, 4],
                        BuildPose::from_position_ticks(
                            IVec3::new(0, 800, 0),
                            GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap(),
        );
        let mut state = EditorState::default();
        state.placed_bearings.push(PlacedBearing {
            kind: JointKind::Rotational,
            axis: Vec3::Y,
            source: FaceRef::part(plate, FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 2.125, 0.0),
            dimensions: BearingDimensions::default(),
        });
        state.hovered_bearing = Some(0);
        state.hovered = Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 2.125, 0.0),
            face: FaceRef::part(plate, FaceKind::PositiveY),
        });
        let mut history = EditorHistory::default();
        let error = hover(
            &graph,
            &mut state,
            ConstructionMaterial::Steel,
            MaterialAppearance::BAKED,
        );
        assert_eq!(error, None);
        assert!(matches!(
            state.gears.preview,
            Some(GearPreview::Socket { .. })
        ));
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(graph.part_count(), 2, "{:?}", state.feedback);
        assert_eq!(graph.bearing_count(), 1);
        let gear = graph
            .parts()
            .find_map(|(_, spec)| spec.as_cylinder().and_then(CylinderSpec::gear))
            .expect("the new cylinder carries teeth");
        assert_eq!(gear.teeth(), 24);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the reaching gear and the hand-set miss share one build"
    )]
    fn a_gear_on_a_bearing_grows_to_reach_the_gear_beside_it() {
        let mut graph = ConstructionGraph::new();
        let plate = spawned(
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 1, 4],
                        BuildPose::from_position_ticks(
                            IVec3::new(0, 800, 0),
                            GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap(),
        );
        // A 24-tooth pinion, 26 cm across its tips, centred at `ticks`.
        let pinion_at = |graph: &mut ConstructionGraph, ticks: IVec3| {
            let spec = CylinderSpec::new(
                CylinderDimensions::new(0.26, 0.0, 0.25).unwrap(),
                BuildPose::from_position_ticks(ticks, GridRotation::default()),
            )
            .with_gear(GearSpec::new(4, 24, GearKind::Spur).unwrap())
            .unwrap();
            spawned(graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap())
        };
        // 31 cm from the bearing, at the height the new gear will stand: 62
        // modules of centre distance leave it 38 teeth.
        let pinion = pinion_at(&mut graph, IVec3::new(124, 900, 0));
        let mut state = EditorState::default();
        state.placed_bearings.push(PlacedBearing {
            kind: JointKind::Rotational,
            axis: Vec3::Y,
            source: FaceRef::part(plate, FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 2.125, 0.0),
            dimensions: BearingDimensions::default(),
        });
        state.hovered_bearing = Some(0);
        state.hovered = Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 2.125, 0.0),
            face: FaceRef::part(plate, FaceKind::PositiveY),
        });
        let mut history = EditorHistory::default();
        let error = hover(
            &graph,
            &mut state,
            ConstructionMaterial::Steel,
            MaterialAppearance::BAKED,
        );
        assert_eq!(error, None);
        assert!(
            state
                .feedback
                .as_deref()
                .unwrap()
                .contains("38-tooth spur gear")
                && state
                    .feedback
                    .as_deref()
                    .unwrap()
                    .contains("meshes 24-tooth spur"),
            "{:?}",
            state.feedback
        );
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        // The tips reach two modules into the pinion's teeth; placement
        // lets meshing parts do that.
        assert_eq!(graph.part_count(), 3, "{:?}", state.feedback);
        let wheel = graph
            .parts()
            .map(|(part, _)| part)
            .find(|part| *part != plate && *part != pinion)
            .unwrap();
        assert_eq!(gear_of(&graph, wheel).unwrap().teeth(), 38);
        assert_eq!(graph.gear_links().count(), 1);
        // The placed gear is drawn where the pinion is, reaching to x = -0.2.
        let mesh = crate::render::mesh::construction::combined_material_construction_mesh(
            &graph,
            None,
            ConstructionMaterial::Steel,
        );
        let low_x = mesh
            .attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION)
            .and_then(|positions| positions.as_float3())
            .unwrap()
            .iter()
            .map(|position| position[0])
            .fold(f32::INFINITY, f32::min);
        assert!(low_x < -0.19, "the wheel is drawn: lowest x {low_x}");
        assert!(
            state
                .feedback
                .as_deref()
                .unwrap()
                .contains("meshed with 24-tooth spur"),
            "{:?}",
            state.feedback
        );

        // Set by hand, the count stays and the line says what it misses.
        let mut state = EditorState::default();
        state.gears.settings = GearSettings::default().adjusted(GearDimension::Teeth, 1, true);
        assert_eq!(state.gears.settings.teeth, 30);
        state.placed_bearings.push(PlacedBearing {
            kind: JointKind::Rotational,
            axis: Vec3::Y,
            source: FaceRef::part(plate, FaceKind::NegativeY),
            anchor: Vec3::new(0.0, 1.875, 0.0),
            dimensions: BearingDimensions::default(),
        });
        state.hovered_bearing = Some(0);
        state.hovered = Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 1.875, 0.0),
            face: FaceRef::part(plate, FaceKind::NegativeY),
        });
        pinion_at(&mut graph, IVec3::new(-124, 700, 0));
        let error = hover(
            &graph,
            &mut state,
            ConstructionMaterial::Steel,
            MaterialAppearance::BAKED,
        );
        assert_eq!(error, None);
        assert!(
            state
                .feedback
                .as_deref()
                .unwrap()
                .contains("30-tooth spur gear")
                && state
                    .feedback
                    .as_deref()
                    .unwrap()
                    .contains("pitch circle 40 mm short of 24-tooth spur"),
            "{:?}",
            state.feedback
        );
    }

    #[test]
    fn a_gear_on_a_bearing_skips_a_partner_it_cannot_stand_against_and_rotate_takes_the_next() {
        fn hovered(graph: &ConstructionGraph, state: &mut EditorState) -> String {
            let error = hover(
                graph,
                state,
                ConstructionMaterial::Steel,
                MaterialAppearance::BAKED,
            );
            assert_eq!(error, None);
            state.feedback.clone().unwrap()
        }
        let mut graph = ConstructionGraph::new();
        let plate = spawned(
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 1, 4],
                        BuildPose::from_position_ticks(
                            IVec3::new(0, 800, 0),
                            GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap(),
        );
        let gear_at = |graph: &mut ConstructionGraph, module: u8, teeth: u16, ticks: IVec3| {
            let gear = GearSpec::new(module, teeth, GearKind::Spur).unwrap();
            let spec = CylinderSpec::new(
                CylinderDimensions::new(gear.tip_diameter(), 0.0, 0.25).unwrap(),
                BuildPose::from_position_ticks(ticks, GridRotation::default()),
            )
            .with_gear(gear)
            .unwrap();
            spawned(graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap())
        };
        // A 24-tooth pinion 31 cm from the bearing, which a 38-tooth gear
        // reaches; a 36-tooth wheel of module 1.25 cm 42 cm the other way,
        // which a 31-tooth gear reaches; and a 72-tooth wheel 70 cm off, whose
        // 68-tooth gear would swallow the pinion.
        let pinion = gear_at(&mut graph, 4, 24, IVec3::new(124, 900, 0));
        let wheel = gear_at(&mut graph, 5, 36, IVec3::new(0, 900, 168));
        let big = gear_at(&mut graph, 4, 72, IVec3::new(-280, 900, 0));
        let mut state = EditorState::default();
        // The count the last placed gear left behind suits the big wheel.
        state.gears.settings.teeth = 68;
        state.placed_bearings.push(PlacedBearing {
            kind: JointKind::Rotational,
            axis: Vec3::Y,
            source: FaceRef::part(plate, FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 2.125, 0.0),
            dimensions: BearingDimensions::default(),
        });
        state.hovered_bearing = Some(0);
        state.hovered = Some(SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 2.125, 0.0),
            face: FaceRef::part(plate, FaceKind::PositiveY),
        });
        let mut history = EditorHistory::default();
        let rotate = |graph: &mut ConstructionGraph, state: &mut EditorState, history: &mut _| {
            handle_gear_actions(&tap(GameAction::Rotate), graph, state, history);
        };

        let line = hovered(&graph, &mut state);
        assert!(
            line.contains("38-tooth spur gear")
                && line.contains("meshes 24-tooth spur")
                && line.contains("(1 of 2; Rotate for the next)"),
            "{line}"
        );
        rotate(&mut graph, &mut state, &mut history);
        assert!(
            !state.gears.settings.across,
            "Rotate chose a partner, not a rack side"
        );
        let line = hovered(&graph, &mut state);
        assert!(
            line.contains("31-tooth spur gear")
                && line.contains("meshes 36-tooth spur")
                && line.contains("(2 of 2"),
            "{line}"
        );
        rotate(&mut graph, &mut state, &mut history);
        let line = hovered(&graph, &mut state);
        assert!(line.contains("(1 of 2"), "the choice wraps round: {line}");
        rotate(&mut graph, &mut state, &mut history);
        hovered(&graph, &mut state);
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(graph.part_count(), 5, "{:?}", state.feedback);
        let placed = graph
            .parts()
            .map(|(part, _)| part)
            .find(|part| ![plate, pinion, wheel, big].contains(part))
            .unwrap();
        assert_eq!(gear_of(&graph, placed).unwrap().teeth(), 31);
        let partners = graph
            .part_gear_links(placed)
            .filter_map(|(_, link)| link.other(placed))
            .collect::<Vec<_>>();
        assert_eq!(partners, vec![wheel]);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "complete rack drag and meshing integration fixture"
    )]
    fn a_drag_across_welded_blocks_cuts_one_rack_that_a_pinion_over_a_joint_meshes_once() {
        let mut graph = ConstructionGraph::new();
        // Four cubes in a row 2 m up, the first three welded, and a 24-tooth
        // pinion on a Z axis standing on the joint between the first two,
        // its pitch circle on their top faces' pitch line.
        let block = |graph: &mut ConstructionGraph, x: i32| {
            spawned(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [1, 1, 1],
                            BuildPose::from_position_ticks(
                                IVec3::new(x, 800, 0),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    ))
                    .unwrap(),
            )
        };
        let blocks = [0, 100, 200, 300].map(|x| block(&mut graph, x));
        for pair in blocks[..3].windows(2) {
            graph
                .apply(BuildCommand::RigidLink(mechanic_core::RigidLinkSpec {
                    first: pair[0],
                    second: pair[1],
                }))
                .unwrap();
        }
        let pinion = spawned(
            graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(0.26, 0.0, 0.25).unwrap(),
                        BuildPose::from_position_ticks(
                            IVec3::new(50, 894, 0),
                            GridRotation::new(1, 0, 0),
                        ),
                    )
                    .with_gear(GearSpec::new(4, 24, GearKind::Spur).unwrap())
                    .unwrap(),
                ))
                .unwrap(),
        );
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        look(&graph, &mut state, -0.1);
        assert!(
            state
                .feedback
                .as_deref()
                .is_some_and(|line| line.contains("drag across blocks")),
            "{:?}",
            state.feedback
        );
        handle_gear_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert!(state.gears.rack_drag.is_some(), "{:?}", state.feedback);
        assert_eq!(
            graph
                .parts()
                .filter(|(_, spec)| spec
                    .as_cuboid()
                    .is_some_and(|cuboid| cuboid.rack().is_some()))
                .count(),
            0
        );

        // Over the unwelded fourth block: the run stops at the third.
        look(&graph, &mut state, 0.7);
        let run = state.gears.rack_drag.as_ref().unwrap().run.clone();
        assert_eq!(
            run.iter().map(|target| target.part).collect::<Vec<_>>(),
            blocks[..3].to_vec(),
            "{:?}",
            state.feedback
        );
        assert!(
            state
                .feedback
                .as_deref()
                .is_some_and(|line| line.contains("across 3 blocks")),
            "{:?}",
            state.feedback
        );
        handle_gear_actions(
            &release(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(state.gears.rack_drag, None);
        assert_eq!(history.undo.len(), 1, "one step for the whole run");
        let racks = blocks.map(|part| graph.part(part).unwrap().as_cuboid().unwrap().rack());
        assert!(
            racks[..3].iter().all(Option::is_some),
            "{:?}",
            state.feedback
        );
        assert!(racks[3].is_none(), "the loose block stays plain");
        assert!(
            racks[..3]
                .iter()
                .all(|rack| rack.unwrap().along() == mechanic_core::Axis::X)
        );
        // The pinion stands on two of the blocks and meshes the rack once.
        assert_eq!(graph.part_gear_links(pinion).count(), 1);
        assert!(
            state
                .feedback
                .as_deref()
                .is_some_and(|line| line.contains("meshed with 24-tooth spur")
                    && line.matches("24-tooth spur").count() == 1),
            "{:?}",
            state.feedback
        );
    }
}
