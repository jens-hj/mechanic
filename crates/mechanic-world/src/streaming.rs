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
    ActiveTerrainNode, TerrainSelection, TerrainSelectionStats, select_active_nodes,
    select_active_nodes_cached, select_active_nodes_with_interests,
};
use selection::{adjacent_leaf, owner_of_leaf};

use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

use bevy_math::DVec3;

use crate::{BRICK_EDGE_METERS, TerrainNodeId, WorldPosition};

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
    }

    /// Pins nodes overlapped by construction bodies.
    pub fn set_pinned(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.pinned.clear();
        self.pinned.extend(nodes);
    }

    /// Sets nodes that must resolve before local world entry.
    pub fn set_critical_nodes(&mut self, nodes: impl IntoIterator<Item = TerrainNodeId>) {
        self.critical.clear();
        self.critical.extend(nodes);
    }

    /// Highest-priority pending request not already in flight.
    pub fn next_request(
        &self,
        in_flight: &BTreeSet<TerrainNodeId>,
        focus: WorldPosition,
    ) -> Option<ActiveTerrainNode> {
        self.pending
            .values()
            .map(|pending| pending.node)
            .filter(|node| !in_flight.contains(&node.id))
            .min_by(|first, second| {
                let priority = |node: &ActiveTerrainNode| {
                    let centre = DVec3::new(
                        (f64::from(node.id.coordinates.x) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                        (f64::from(node.id.coordinates.y) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                        (f64::from(node.id.coordinates.z) + node.id.edge_bricks() as f64 * 0.5)
                            * BRICK_EDGE_METERS,
                    );
                    (
                        !(self.pinned.contains(&node.id) || self.critical.contains(&node.id)),
                        std::cmp::Reverse(node.generation),
                        !self.seam_dependencies.contains(&node.id),
                        centre.distance_squared(focus.0),
                    )
                };
                let first_priority = priority(first);
                let second_priority = priority(second);
                first_priority
                    .0
                    .cmp(&second_priority.0)
                    .then_with(|| first_priority.1.cmp(&second_priority.1))
                    .then_with(|| first_priority.2.cmp(&second_priority.2))
                    .then_with(|| first_priority.3.total_cmp(&second_priority.3))
                    .then_with(|| first.id.cmp(&second.id))
            })
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
