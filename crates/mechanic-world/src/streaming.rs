//! Balanced adaptive-octree selection and generation-aware streaming state.

#![expect(clippy::cast_precision_loss)]

mod bounds_cache;
mod faces;
mod publication;
mod selection;

pub use bounds_cache::TerrainBoundsCache;
pub use faces::{TerrainFace, TerrainTransitionMask};
pub use publication::{
    TerrainPublicationDelta, TerrainPublicationUpsert, TerrainReadiness, publication_face_mask,
};
use publication::{neighbour_candidates, publication_face_mask_maps};
pub use selection::{
    ActiveTerrainNode, CAVE_STREAMED_LEVEL, MAX_STREAMED_LEVEL, MIN_TERRAIN_DETAIL_SCALE,
    STREAMED_LEVELS, TerrainSelection, TerrainSelectionStats, select_active_nodes,
    select_active_nodes_cached, select_active_nodes_with_interests,
};
use selection::{adjacent_leaf, owner_of_leaf};

use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

use bevy_math::DVec3;

use crate::{TerrainNodeId, WorldPosition};

/// Generation-aware node streaming state shared by render and collision owners.
#[derive(Clone, Debug, Default)]
pub struct TerrainStreamer {
    cut_generation: u64,
    desired: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    pending: BTreeMap<TerrainNodeId, PendingTerrainNode>,
    staged: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    active: BTreeMap<TerrainNodeId, ActiveTerrainNode>,
    pinned: BTreeSet<TerrainNodeId>,
    critical: BTreeSet<TerrainNodeId>,
    seam_dependencies: BTreeSet<TerrainNodeId>,
    dirty_publication: BTreeSet<TerrainNodeId>,
    replacement_needs: BTreeMap<TerrainNodeId, BTreeSet<TerrainNodeId>>,
    replacement_owners: BTreeMap<TerrainNodeId, BTreeSet<TerrainNodeId>>,
    view: Option<TerrainView>,
    queue: RequestQueue,
    last_retired: Vec<TerrainNodeId>,
}

/// Nodes one activation made current, and the nodes it replaced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainActivation {
    /// Newly active nodes.
    pub activated: Vec<ActiveTerrainNode>,
    /// Previously active nodes the activation removed; a node whose
    /// generation changed appears in both lists.
    pub retired: Vec<TerrainNodeId>,
}

/// Where the camera looks, so work in view streams before work behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainView {
    /// Horizontal look direction; need not be normalised.
    pub forward: DVec3,
    /// Half the horizontal field of view in radians, before any margin.
    pub half_angle: f64,
}

/// Pending work sorted by priority, rebuilt when the cut, the focus, or the
/// view changes enough to reorder it.
#[derive(Clone, Debug, Default)]
struct RequestQueue {
    /// Worst priority first, so the best is popped from the end.
    order: Vec<(RequestPriority, TerrainNodeId)>,
    focus: Option<DVec3>,
    view: Option<TerrainView>,
    stale: bool,
}

/// Rings, in metres of horizontal distance, that order streaming work: all
/// work in a ring precedes any farther ring, whatever its seams or edits.
const PRIORITY_RINGS: [f64; 6] = [16.0, 40.0, 64.0, 160.0, 400.0, 640.0];

/// Focus movement that reorders queued work.
const REQUEUE_METRES: f64 = 4.0;

/// Extra half angle beyond the field of view still counted as in view, so
/// turning a little finds ground ready.
const VIEW_MARGIN_RADIANS: f64 = 0.35;

/// Ordering of one pending request; smaller is sooner.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct RequestPriority {
    /// Neither pinned nor needed before entering the world.
    routine: bool,
    ring: usize,
    newest_edit: std::cmp::Reverse<u64>,
    out_of_view: bool,
    not_seam: bool,
    distance: f64,
}

#[derive(Clone, Debug)]
struct PendingTerrainNode {
    node: ActiveTerrainNode,
    queued_at: Instant,
}

impl TerrainStreamer {
    /// Reconciles a newly selected cut and queues missing or stale generations.
    pub fn set_desired(&mut self, cut: impl IntoIterator<Item = ActiveTerrainNode>) {
        let desired = cut.into_iter().map(|node| (node.id, node)).collect();
        if self.desired != desired {
            self.cut_generation = self.cut_generation.wrapping_add(1).max(1);
        }
        self.desired = desired;
        self.pending
            .retain(|id, pending| self.desired.get(id) == Some(&pending.node));
        self.staged
            .retain(|id, staged| self.desired.get(id) == Some(staged));
        for (&id, &node) in &self.desired {
            if self.active.get(&id) != Some(&node) && self.staged.get(&id) != Some(&node) {
                self.pending
                    .entry(id)
                    .or_insert_with(|| PendingTerrainNode {
                        node,
                        queued_at: Instant::now(),
                    });
            }
        }
        let obsolete = self
            .active
            .iter()
            .filter_map(|(&id, active)| (self.desired.get(&id) != Some(active)).then_some(id))
            .collect::<BTreeSet<_>>();
        self.replacement_needs.clear();
        self.replacement_owners.clear();
        for &replacement in self.desired.keys() {
            let mut ancestor = Some(replacement);
            while let Some(candidate) = ancestor {
                if obsolete.contains(&candidate) {
                    self.replacement_needs
                        .entry(candidate)
                        .or_default()
                        .insert(replacement);
                    self.replacement_owners
                        .entry(replacement)
                        .or_default()
                        .insert(candidate);
                }
                ancestor = candidate.parent();
            }
        }
        for &old in &obsolete {
            let mut ancestor = old.parent();
            while let Some(candidate) = ancestor {
                if self.desired.contains_key(&candidate) {
                    self.replacement_needs
                        .entry(old)
                        .or_default()
                        .insert(candidate);
                    self.replacement_owners
                        .entry(candidate)
                        .or_default()
                        .insert(old);
                    break;
                }
                ancestor = candidate.parent();
            }
        }
        for id in obsolete {
            if !self.replacement_needs.contains_key(&id) && !self.pinned.contains(&id) {
                self.active.remove(&id);
                self.mark_publication_dirty(id);
            }
        }
        self.rebuild_seam_dependencies();
        self.queue.stale = true;
    }

    /// Pins nodes overlapped by construction bodies.
    pub fn set_pinned(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.pinned.clear();
        self.pinned.extend(nodes);
        self.queue.stale = true;
    }

    /// Sets nodes that must resolve before local world entry.
    pub fn set_critical_nodes(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.critical.clear();
        self.critical.extend(nodes);
        self.queue.stale = true;
    }

    /// Sets the camera's look direction; work in view streams first within
    /// each distance ring.
    pub fn set_view(&mut self, view: Option<TerrainView>) {
        self.view = view;
    }

    /// Whether a job for `node` still produces something the cut wants.
    pub fn wants(&self, node: &ActiveTerrainNode) -> bool {
        self.desired.get(&node.id) == Some(node)
    }

    /// Highest-priority pending request not already in flight.
    ///
    /// Nearby ground comes first: pinned and startup-critical nodes, then
    /// distance rings, and within a ring edited nodes, nodes in view, and
    /// nodes seams depend on, nearest first.
    pub fn next_request(
        &mut self,
        in_flight: &BTreeSet<TerrainNodeId>,
        focus: WorldPosition,
    ) -> Option<ActiveTerrainNode> {
        let turned = match (self.queue.view, self.view) {
            (Some(queued), Some(current)) => {
                queued
                    .forward
                    .normalize_or_zero()
                    .dot(current.forward.normalize_or_zero())
                    < (VIEW_MARGIN_RADIANS * 0.5).cos()
            }
            (queued, current) => queued.is_some() != current.is_some(),
        };
        if self.queue.stale
            || turned
            || self.queue.focus.is_none_or(|queued| {
                queued.distance_squared(focus.0) > REQUEUE_METRES * REQUEUE_METRES
            })
        {
            self.rebuild_queue(focus.0);
        }
        let mut skipped = Vec::new();
        let found = loop {
            let Some((priority, id)) = self.queue.order.pop() else {
                break None;
            };
            let Some(pending) = self.pending.get(&id) else {
                continue;
            };
            if in_flight.contains(&id) {
                skipped.push((priority, id));
                continue;
            }
            break Some(pending.node);
        };
        self.queue.order.extend(skipped.into_iter().rev());
        found
    }

    fn rebuild_queue(&mut self, focus: DVec3) {
        let view = self.view.map(|view| {
            let forward = DVec3::new(view.forward.x, 0.0, view.forward.z).normalize_or_zero();
            (
                forward,
                (view.half_angle + VIEW_MARGIN_RADIANS)
                    .min(core::f64::consts::PI)
                    .cos(),
            )
        });
        let mut order = self
            .pending
            .values()
            .map(|pending| {
                let node = pending.node;
                let (minimum, maximum) = selection::node_bounds(node.id);
                let distance =
                    selection::horizontal_distance_squared_to_bounds(focus, minimum, maximum)
                        .sqrt();
                let ring = PRIORITY_RINGS
                    .iter()
                    .position(|&reach| distance <= reach)
                    .unwrap_or(PRIORITY_RINGS.len());
                let out_of_view = view.is_some_and(|(forward, cos_limit)| {
                    let centre = (minimum + maximum) * 0.5 - focus;
                    let direction = DVec3::new(centre.x, 0.0, centre.z);
                    let half_width = (maximum.x - minimum.x) * 0.5;
                    direction.length() > half_width * 1.5
                        && forward != DVec3::ZERO
                        && direction.normalize_or_zero().dot(forward) < cos_limit
                });
                let priority = RequestPriority {
                    routine: !(self.pinned.contains(&node.id) || self.critical.contains(&node.id)),
                    ring,
                    newest_edit: std::cmp::Reverse(node.generation),
                    out_of_view,
                    not_seam: !self.seam_dependencies.contains(&node.id),
                    distance,
                };
                (priority, node.id)
            })
            .collect::<Vec<_>>();
        order.sort_by(|first, second| {
            second
                .0
                .partial_cmp(&first.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| second.1.cmp(&first.1))
        });
        self.queue = RequestQueue {
            order,
            focus: Some(focus),
            view: self.view,
            stale: false,
        };
    }

    /// Removes a request from the pending queue once a worker accepts it.
    pub fn mark_started(&mut self, node: ActiveTerrainNode) {
        if self.pending.get(&node.id).map(|pending| pending.node) == Some(node) {
            self.pending.remove(&node.id);
        }
    }

    /// Stages a result if it still matches the desired generation and seam mask.
    pub fn stage(&mut self, node: ActiveTerrainNode) -> bool {
        if self.desired.get(&node.id) != Some(&node) {
            return false;
        }
        self.staged.insert(node.id, node);
        true
    }

    /// Atomically activates a complete local replacement at a safe
    /// render/physics boundary.
    pub fn activate(&mut self, id: TerrainNodeId) -> Vec<ActiveTerrainNode> {
        self.activate_replacing(id).activated
    }

    /// Whether a node is in the active cut, in any generation.
    pub fn is_active(&self, id: TerrainNodeId) -> bool {
        self.active.contains_key(&id)
    }

    /// Like [`Self::activate`], also naming the previously active nodes the
    /// activation replaced, so a renderer can show the new nodes and drop the
    /// old ones in one step.
    pub fn activate_replacing(&mut self, id: TerrainNodeId) -> TerrainActivation {
        let activated = self.activate_group(id);
        let retired = core::mem::take(&mut self.last_retired);
        TerrainActivation { activated, retired }
    }

    fn activate_group(&mut self, id: TerrainNodeId) -> Vec<ActiveTerrainNode> {
        self.last_retired.clear();
        if !self.staged.contains_key(&id) {
            return Vec::new();
        }
        let mut old_nodes = self
            .replacement_owners
            .get(&id)
            .cloned()
            .unwrap_or_default();
        if old_nodes.is_empty() {
            let Some(node) = self.staged.remove(&id) else {
                return Vec::new();
            };
            self.active.insert(id, node);
            self.mark_publication_dirty(id);
            return vec![node];
        }

        let mut replacements = BTreeSet::new();
        loop {
            let previous_old_count = old_nodes.len();
            for old in old_nodes.clone() {
                replacements.extend(
                    self.replacement_needs
                        .get(&old)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
            }
            for replacement in replacements.clone() {
                old_nodes.extend(
                    self.replacement_owners
                        .get(&replacement)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
            }
            if old_nodes.len() == previous_old_count {
                break;
            }
        }
        let complete = replacements.iter().all(|replacement| {
            self.desired.get(replacement) == self.active.get(replacement)
                || self.staged.get(replacement) == self.desired.get(replacement)
        });
        if !complete {
            return Vec::new();
        }

        for old in &old_nodes {
            self.active.remove(old);
            self.last_retired.push(*old);
            self.mark_publication_dirty(*old);
            if let Some(needs) = self.replacement_needs.remove(old) {
                for replacement in needs {
                    if let Some(owners) = self.replacement_owners.get_mut(&replacement) {
                        owners.remove(old);
                        if owners.is_empty() {
                            self.replacement_owners.remove(&replacement);
                        }
                    }
                }
            }
        }
        let mut activated = Vec::new();
        for replacement in replacements {
            if let Some(node) = self.staged.remove(&replacement) {
                self.active.insert(replacement, node);
                self.mark_publication_dirty(replacement);
                activated.push(node);
            }
        }
        activated
    }

    /// Pending plus staged work count.
    pub fn backlog(&self) -> usize {
        self.pending.len() + self.staged.len()
    }

    /// Current active cut.
    pub fn active(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.active.values().copied()
    }

    /// Active nodes whose generation and transition dependency exactly match
    /// the current desired cut. Faces touching any other active generation
    /// must remain capped.
    pub fn current_active(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.active
            .values()
            .filter(|node| self.desired.get(&node.id) == Some(*node))
            .copied()
    }

    /// Current desired cut, including work not active yet.
    pub fn desired(&self) -> impl Iterator<Item = ActiveTerrainNode> + '_ {
        self.desired.values().copied()
    }

    /// Whether every desired node `within` accepts is active exactly as desired.
    /// Work elsewhere, such as distant detail still streaming, does not count.
    pub fn settled_where(&self, mut within: impl FnMut(TerrainNodeId) -> bool) -> bool {
        self.desired
            .iter()
            .all(|(id, node)| !within(*id) || self.active.get(id) == Some(node))
    }

    /// Current startup-region completion. Empty active chunks count as resolved.
    pub fn local_readiness(&self) -> TerrainReadiness {
        let mut readiness = TerrainReadiness::default();
        for id in &self.critical {
            let Some(desired) = self.desired.get(id) else {
                continue;
            };
            readiness.total += 1;
            readiness.resolved += usize::from(self.active.get(id) == Some(desired));
        }
        readiness
    }

    /// Age of the oldest request still waiting for a worker.
    pub fn oldest_queue_age(&self) -> Duration {
        self.pending
            .values()
            .map(|pending| pending.queued_at.elapsed())
            .max()
            .unwrap_or_default()
    }

    /// Nodes whose mesh or face readiness may need republishing.
    pub fn take_dirty_publication(&mut self) -> BTreeSet<TerrainNodeId> {
        core::mem::take(&mut self.dirty_publication)
    }

    /// Takes the incremental render/collision publication delta.
    ///
    /// Face readiness is computed directly against streamer-owned maps, so the
    /// app does not need to rebuild complete active and desired sets merely to
    /// publish a handful of changed chunks.
    pub fn take_publication_delta(&mut self) -> TerrainPublicationDelta {
        let dirty = core::mem::take(&mut self.dirty_publication);
        let mut delta = TerrainPublicationDelta {
            generation: self.cut_generation,
            ..TerrainPublicationDelta::default()
        };
        for id in dirty {
            if let Some(&node) = self.active.get(&id)
                && self.desired.get(&id) == Some(&node)
            {
                delta.upserts.push(TerrainPublicationUpsert {
                    node,
                    ready_faces: publication_face_mask_maps(id, &self.active, &self.desired),
                });
            } else {
                delta.removals.push(id);
            }
        }
        delta
    }

    /// True when publication bookkeeping has work for the main thread.
    pub fn has_dirty_publication(&self) -> bool {
        !self.dirty_publication.is_empty()
    }

    /// Returns publication work that did not fit the current main-thread budget.
    pub fn defer_publication(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.dirty_publication.extend(nodes);
    }

    fn rebuild_seam_dependencies(&mut self) {
        self.seam_dependencies.clear();
        let selected = self.desired.keys().copied().collect::<BTreeSet<_>>();
        for node in self.desired.values() {
            if node.transition_mask == TerrainTransitionMask::NONE {
                continue;
            }
            self.seam_dependencies.insert(node.id);
            for face in TerrainFace::ALL {
                if !node.transition_mask.contains(face) {
                    continue;
                }
                if let Some(neighbour) = adjacent_leaf(node.id, face)
                    && let Some(owner) = owner_of_leaf(&selected, neighbour)
                {
                    self.seam_dependencies.insert(owner);
                }
            }
        }
    }

    fn mark_publication_dirty(&mut self, id: TerrainNodeId) {
        self.dirty_publication.insert(id);
        for face in TerrainFace::ALL {
            for neighbour in neighbour_candidates(id, face) {
                if self.desired.contains_key(&neighbour) || self.active.contains_key(&neighbour) {
                    self.dirty_publication.insert(neighbour);
                }
            }
        }
    }
}

/// Terrain worker count during interactive play.
///
/// Four logical cores remain available for Bevy's main, render, IO, and async
/// work. This yields six terrain workers on the reference 10-core M1 Pro.
pub fn terrain_worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(4, usize::from)
        .saturating_sub(4)
        .clamp(1, 8)
}

/// Terrain worker count while the opaque world-loading screen is active.
pub fn terrain_loading_worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(4, usize::from)
        .saturating_sub(2)
        .clamp(1, 8)
}

#[cfg(test)]
mod tests;
