//! Keeping construction foundations in step with the terrain beneath them.

use super::list::{WorldListPhase, WorldListState};
use super::streaming::TerrainAcknowledgements;
use super::transfer::{bounds_foundation_support, framed_part_bounds};
use super::{Res, ResMut, ToOwned, Vec, WorldDiagnostics, WorldRuntime, format};
use crate::editor::history::EditorHistory;
use crate::editor::state::{EditorGraph, EditorState};
use mechanic_core::{ConstructionEditDelta, PartId, PartSpec};
use mechanic_world::{FoundationSpatialIndex, FoundationSupport, TerrainEditBatch, TerrainScene};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

#[derive(Clone, Debug)]
pub(super) struct TerrainFoundation {
    pub(super) part: PartId,
    pub(super) support: FoundationSupport,
}

pub(super) struct PendingFoundationSync {
    pub(super) editor_revision: u64,
    pub(super) parts: BTreeMap<PartId, PartSpec>,
    pub(super) frames: BTreeMap<PartId, mechanic_core::ConstructionFrame>,
    pub(super) replaced_parts: BTreeSet<PartId>,
    pub(super) new_parts: Vec<PartId>,
    pub(super) next_part: usize,
    pub(super) foundations: Vec<TerrainFoundation>,
    pub(super) index: FoundationSpatialIndex,
}

pub(super) fn foundation_edit_is_ready(
    acknowledgements: TerrainAcknowledgements,
    pending: &TerrainEditBatch,
    foundation_acknowledgement: u64,
    stroke_idle: bool,
) -> bool {
    !pending.is_empty()
        && foundation_acknowledgement != pending.generation
        && acknowledgements.completed(pending.generation)
        && stroke_idle
}

pub(super) const FOUNDATION_SYNC_FRAME_BUDGET: Duration = Duration::from_millis(2);

pub(super) const FOUNDATION_SYNC_MAX_PARTS_PER_FRAME: usize = 32;

#[expect(
    clippy::too_many_lines,
    reason = "revision staging and bounded support sampling form one cutover"
)]
pub(crate) fn sync_world_foundations(
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    mut runtime: ResMut<WorldRuntime>,
    list: Res<WorldListState>,
    mut editor: ResMut<EditorState>,
    mut diagnostics: ResMut<WorldDiagnostics>,
) {
    if list.phase() != WorldListPhase::Playing {
        return;
    }
    let frame_started = std::time::Instant::now();
    diagnostics.foundation_candidate_count = 0;
    diagnostics.foundation_sample_count = 0;
    diagnostics.foundation_refresh_ms = 0.0;

    let terrain_changed = foundation_edit_is_ready(
        runtime.terrain_acknowledgements,
        &runtime.pending_foundation_edit,
        runtime.foundation_edit_acknowledgement,
        runtime.terrain_edit_task.is_none()
            && runtime.pending_terrain_edits.is_empty()
            && runtime.last_brush_edit.is_none(),
    );
    if terrain_changed {
        // Partially sampled construction belongs to the previous terrain cut.
        // Discard it and restart from the newly published terrain below.
        runtime.pending_foundation_sync = None;
        refresh_foundations_after_terrain_edit(&mut runtime, &mut diagnostics, &mut editor);
    }

    let editor_changed = runtime.synced_editor_revision != history.current_revision;
    let needs_initial_sync = runtime.known_world_parts.is_empty()
        && graph.0.parts().next().is_some()
        && runtime.pending_foundation_sync.is_none();
    let pending_matches = runtime
        .pending_foundation_sync
        .as_ref()
        .is_some_and(|pending| pending.editor_revision == history.current_revision);
    if (editor_changed || needs_initial_sync) && !pending_matches {
        let current_parts = graph
            .0
            .parts()
            .map(|(part, spec)| (part, *spec))
            .collect::<BTreeMap<_, _>>();
        let delta =
            ConstructionEditDelta::between_parts(&runtime.known_world_parts, &current_parts);
        let current_frames = graph
            .0
            .parts()
            .map(|(part, _)| {
                (
                    part,
                    graph.0.part_frame(part).expect("world part has frame"),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let reframed_parts = current_frames
            .iter()
            .filter_map(|(part, frame)| {
                (runtime.known_world_frames.get(part) != Some(frame)).then_some(*part)
            })
            .collect::<BTreeSet<_>>();
        let replaced_parts = delta
            .removed
            .iter()
            .chain(&delta.modified)
            .copied()
            .chain(reframed_parts.iter().copied())
            .collect();
        let new_parts = delta
            .added
            .iter()
            .chain(&delta.modified)
            .copied()
            .chain(reframed_parts.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        runtime.pending_foundation_sync = Some(PendingFoundationSync {
            editor_revision: history.current_revision,
            parts: current_parts,
            frames: current_frames,
            replaced_parts,
            new_parts,
            next_part: 0,
            foundations: Vec::new(),
            index: FoundationSpatialIndex::default(),
        });
        let now = runtime.clock;
        runtime.autosave.mutate(now);
    }

    let Some(mut pending) = runtime.pending_foundation_sync.take() else {
        diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
        return;
    };
    diagnostics.foundation_candidate_count =
        u64::try_from(pending.new_parts.len()).unwrap_or(u64::MAX);
    // Ground welds belong to the saved terrain, independently of mesh streaming.
    let scene = TerrainScene {
        field: &runtime.field,
        edits: &runtime.edits,
    };
    let mut processed = 0_usize;
    while pending.next_part < pending.new_parts.len()
        && processed < FOUNDATION_SYNC_MAX_PARTS_PER_FRAME
        && (processed == 0 || frame_started.elapsed() < FOUNDATION_SYNC_FRAME_BUDGET)
    {
        let part = pending.new_parts[pending.next_part];
        pending.next_part += 1;
        processed += 1;
        let Some(&spec) = pending.parts.get(&part) else {
            continue;
        };
        let frame = *pending.frames.get(&part).expect("pending part has frame");
        let support = bounds_foundation_support(
            &scene,
            framed_part_bounds(spec, frame),
            runtime.floating_origin,
        );
        diagnostics.foundation_sample_count = diagnostics
            .foundation_sample_count
            .saturating_add(u64::try_from(support.sample_count()).unwrap_or(u64::MAX));
        if support.has_valid_anchor() {
            pending.index.insert(part, &support);
            pending
                .foundations
                .push(TerrainFoundation { part, support });
        }
    }

    if pending.next_part < pending.new_parts.len() {
        runtime.pending_foundation_sync = Some(pending);
        diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
        return;
    }

    let released_provisional_parts = !pending.new_parts.is_empty();
    let removed_foundation = runtime
        .foundations
        .iter()
        .any(|foundation| pending.replaced_parts.contains(&foundation.part));
    for &part in &pending.replaced_parts {
        runtime.foundation_index.remove(part);
    }
    runtime
        .foundations
        .retain(|foundation| !pending.replaced_parts.contains(&foundation.part));
    let added = !pending.foundations.is_empty();
    runtime.foundation_index.append(pending.index);
    runtime.foundations.append(&mut pending.foundations);
    runtime.known_world_parts = pending.parts;
    runtime.known_world_frames = pending.frames;
    runtime.synced_editor_revision = pending.editor_revision;
    if added || removed_foundation || released_provisional_parts {
        runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
    }
    diagnostics.foundation_refresh_ms = frame_started.elapsed().as_secs_f64() * 1_000.0;
}

pub(super) fn refresh_foundations_after_terrain_edit(
    runtime: &mut WorldRuntime,
    diagnostics: &mut WorldDiagnostics,
    editor: &mut EditorState,
) {
    let candidates = runtime
        .foundation_index
        .candidates(&runtime.pending_foundation_edit.changed_bricks);
    let changed_bricks = runtime.pending_foundation_edit.changed_bricks.clone();
    diagnostics.foundation_candidate_count = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
    // Ground welds belong to the saved terrain, independently of mesh streaming.
    let scene = TerrainScene {
        field: &runtime.field,
        edits: &runtime.edits,
    };
    let mut detached = 0_u64;
    let mut anchors_changed = 0_u64;
    runtime.foundations.retain_mut(|foundation| {
        if !candidates.contains(&foundation.part) {
            return true;
        }
        let refresh = foundation.support.refresh_changed(&scene, &changed_bricks);
        diagnostics.foundation_sample_count = diagnostics
            .foundation_sample_count
            .saturating_add(u64::try_from(refresh.sampled).unwrap_or(u64::MAX));
        anchors_changed = anchors_changed
            .saturating_add(u64::try_from(refresh.anchors_changed).unwrap_or(u64::MAX));
        if refresh.detached {
            detached = detached.saturating_add(1);
            runtime.foundation_index.remove(foundation.part);
            false
        } else {
            true
        }
    });
    runtime.foundation_edit_acknowledgement = runtime.pending_foundation_edit.generation;
    runtime.pending_foundation_edit = TerrainEditBatch::default();
    if anchors_changed > 0 {
        runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
    }
    if detached > 0 {
        editor.feedback = Some(if detached == 1 {
            "Foundation lost its last terrain anchor — construction released".to_owned()
        } else {
            format!(
                "{detached} foundations lost their last terrain anchors — constructions released"
            )
        });
    }
}
