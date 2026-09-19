mod apply;
mod commands;
mod error;
mod machine;
mod predicates;
mod queries;
mod region_checks;
mod solid_cache;
mod specs;
mod validate;
mod weld_geometry;

pub use commands::{AppearanceTarget, BuildCommand, BuildOutcome, PendingOperation};
pub use error::GraphError;
pub use specs::{
    ActuatorInventory, BearingDimensionError, BearingDimensions, BearingSpec, DriveLinkSpec,
    InputSeatLinkSpec, MAX_BEARING_OUTER_DIAMETER, MIN_BEARING_DIAMETER_GAP,
    MIN_BEARING_OUTER_DIAMETER, RigidLinkSpec, SeatControllerLinkSpec, WeldSpec,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    BearingId, CuboidSpec, DriveLinkId, EngineKind, FaceOwner, GearboxConfig, InputSeatLinkId,
    PartId, PartSpec, RegionId, RigidLinkId, SeatControllerLinkId, ShapeFeature, ShapeFeatureId,
    ShapeRegion, WeldId, id::Arena,
};

/// Editable, CPU-owned construction topology.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraphData {
    solid_cache: solid_cache::SolidCache,
    pub(crate) construction_frames: crate::frame::ConstructionFrames,
    pub(crate) parts: Arena<PartSpec, PartId>,
    pub(crate) welds: Arena<WeldSpec, WeldId>,
    pub(crate) rigid_links: Arena<RigidLinkSpec, RigidLinkId>,
    pub(crate) bearings: Arena<BearingSpec, BearingId>,
    pub(crate) drive_links: Arena<DriveLinkSpec, DriveLinkId>,
    pub(crate) input_seat_links: Arena<InputSeatLinkSpec, InputSeatLinkId>,
    pub(crate) seat_controller_links: Arena<SeatControllerLinkSpec, SeatControllerLinkId>,
    /// Transmission part to its engine-or-transmission parent.
    pub(crate) transmission_parents: BTreeMap<PartId, PartId>,
    /// Transmission part to the weld created atomically with it.
    pub(crate) transmission_welds: BTreeMap<PartId, WeldId>,
    /// Persistent per-controller, per-engine-family gearbox overrides.
    pub(crate) gearbox_configs: BTreeMap<(PartId, EngineKind), GearboxConfig>,
    /// Editable shape regions. A region owns the geometry of the blocks it
    /// covers, so those blocks stop emitting boxes of their own.
    pub(crate) regions: Arena<ShapeRegion, RegionId>,
    /// Shape features in stable-handle storage.
    pub(crate) shape_features: Arena<ShapeFeature, ShapeFeatureId>,
    /// Explicit global replay order, independent of arena slot reuse.
    pub(crate) shape_feature_order: Vec<ShapeFeatureId>,
    pending: Option<PendingOperation>,
}

/// Copy-on-write editable construction graph.
///
/// Clones are immutable shared revisions. The first mutation of a staged edit
/// copies graph storage once, which keeps history capture constant-time.
#[derive(Clone, Debug, Default)]
pub struct ConstructionGraph {
    data: Arc<ConstructionGraphData>,
}

impl core::ops::Deref for ConstructionGraph {
    type Target = ConstructionGraphData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl core::ops::DerefMut for ConstructionGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.data)
    }
}

/// One privately owned graph revision being staged with a single deep clone.
///
/// Dropping an edit discards every mutation. Call [`Self::finish`] only after
/// every command succeeds.
#[derive(Clone, Debug)]
pub struct ConstructionGraphEdit {
    graph: ConstructionGraph,
}

/// Complete structural assembly reached from one seed part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralComponent {
    parts: BTreeSet<PartId>,
    touches_authored_ground: bool,
}

impl StructuralComponent {
    /// Whether this assembly contains a part.
    pub fn contains(&self, part: PartId) -> bool {
        self.parts.contains(&part)
    }

    /// Parts in stable graph order.
    pub fn parts(&self) -> impl Iterator<Item = PartId> + '_ {
        self.parts.iter().copied()
    }

    /// Whether a graph weld or caller-provided terrain anchor grounds the assembly.
    pub const fn touches_authored_ground(&self) -> bool {
        self.touches_authored_ground
    }
}

/// A construction separated into the selected assembly and everything left behind.
#[derive(Clone, Debug)]
pub struct GraphPartition {
    /// Graph with the selected structural assembly removed.
    pub remainder: ConstructionGraph,
    /// Graph containing only the selected structural assembly.
    pub component: ConstructionGraph,
}

impl ConstructionGraphEdit {
    /// Reserves storage for a known bulk edit without changing graph contents.
    pub fn reserve_parts_and_welds(&mut self, parts: usize, welds: usize) {
        self.graph.parts.reserve(parts);
        self.graph.welds.reserve(welds);
    }

    /// Inserts an already placement-validated cuboid batch and returns stable IDs.
    pub fn spawn_cuboids(&mut self, specs: impl IntoIterator<Item = CuboidSpec>) -> Vec<PartId> {
        specs
            .into_iter()
            .map(|spec| self.graph.insert_edit_part(spec.into()))
            .collect()
    }

    /// Applies one validated command without cloning the staged graph again.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when the command violates graph or geometry invariants.
    pub fn apply(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        self.graph.apply_validated(command)
    }

    /// Applies ordered validated commands without cloning the staged graph again.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`]. The caller must discard this edit on failure.
    pub fn apply_batch(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<Vec<BuildOutcome>, GraphError> {
        commands
            .into_iter()
            .map(|command| self.graph.apply_validated(command))
            .collect()
    }

    /// Applies validated commands while discarding outcomes the caller does not need.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`]. The caller must discard this edit on failure.
    pub fn apply_batch_discarding_outcomes(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<(), GraphError> {
        for command in commands {
            self.graph.apply_validated(command)?;
        }
        Ok(())
    }

    /// Commits the staged revision to its caller.
    pub fn finish(self) -> ConstructionGraph {
        self.graph
    }
}

impl core::ops::Deref for ConstructionGraphEdit {
    type Target = ConstructionGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl ConstructionGraph {
    /// Whether two handles still refer to the same immutable graph revision.
    pub fn shares_revision(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    pub(crate) fn transform_bearing_space(&mut self, transform: crate::ConstructionFrame) {
        let bearings = self
            .bearings
            .iter()
            .map(|(id, bearing)| (id, *bearing))
            .collect::<Vec<_>>();
        for (id, mut bearing) in bearings {
            bearing.shared_anchor = transform.point(bearing.shared_anchor);
            bearing.axis = transform.vector(bearing.axis);
            bearing.kind.rotate(|vector| transform.vector(vector));
            if let Some(destination) = self.bearings.get_mut(id) {
                *destination = bearing;
            }
        }
        if let Some(PendingOperation::Bearing { anchor, .. }) = &mut self.pending {
            *anchor = transform.point(*anchor);
        }
    }

    /// Creates an empty graph in paused build mode.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a transaction by deep-cloning this revision exactly once.
    pub fn begin_edit(&self) -> ConstructionGraphEdit {
        ConstructionGraphEdit {
            graph: self.clone(),
        }
    }

    /// Traverses every structural connection from `seed`.
    ///
    /// Welds, rigid links, bearings, and shared shape regions all join the
    /// component. `authored_ground_parts` supplies terrain-foundation anchors
    /// owned by the application in addition to explicit graph ground welds.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::MissingPart`] when `seed` is not live in this graph.
    pub fn structural_component(
        &self,
        seed: PartId,
        authored_ground_parts: impl IntoIterator<Item = PartId>,
    ) -> Result<StructuralComponent, GraphError> {
        self.parts.get(seed).ok_or(GraphError::MissingPart(seed))?;
        let authored_ground_parts = authored_ground_parts.into_iter().collect::<BTreeSet<_>>();
        let mut parts = BTreeSet::from([seed]);
        let mut frontier = vec![seed];
        let mut touches_authored_ground = false;
        while let Some(part) = frontier.pop() {
            touches_authored_ground |= authored_ground_parts.contains(&part);
            let mut visit = |other: PartId| {
                if parts.insert(other) {
                    frontier.push(other);
                }
            };
            for (_, weld) in self.welds.iter() {
                match (weld.first.owner, weld.second.owner) {
                    (FaceOwner::Part(first), FaceOwner::Part(second)) if first == part => {
                        visit(second);
                    }
                    (FaceOwner::Part(first), FaceOwner::Part(second)) if second == part => {
                        visit(first);
                    }
                    (FaceOwner::Part(candidate), FaceOwner::Ground)
                    | (FaceOwner::Ground, FaceOwner::Part(candidate))
                        if candidate == part =>
                    {
                        touches_authored_ground = true;
                    }
                    _ => {}
                }
            }
            for (_, link) in self.rigid_links.iter() {
                if link.first == part {
                    visit(link.second);
                } else if link.second == part {
                    visit(link.first);
                }
            }
            for (_, bearing) in self.bearings.iter() {
                let endpoints = match (bearing.source.owner, bearing.target.map(|face| face.owner))
                {
                    (FaceOwner::Part(first), Some(FaceOwner::Part(second))) => {
                        Some((first, second))
                    }
                    _ => None,
                };
                if let Some((first, second)) = endpoints {
                    if first == part {
                        visit(second);
                    } else if second == part {
                        visit(first);
                    }
                }
            }
            if let Some(region) = self.region_of(part) {
                let members = self
                    .parts()
                    .filter_map(|(candidate, _)| {
                        (self.region_of(candidate) == Some(region)).then_some(candidate)
                    })
                    .collect::<Vec<_>>();
                for member in members {
                    visit(member);
                }
            }
        }
        Ok(StructuralComponent {
            parts,
            touches_authored_ground,
        })
    }

    /// Splits one complete structural assembly from this graph while retaining
    /// every connection, program, region, and graph-owned relation on its side.
    ///
    /// # Panics
    ///
    /// Panics only if removing a live part from an internally valid cloned graph fails.
    pub fn partition(&self, component: &StructuralComponent) -> GraphPartition {
        let all_parts = self.parts().map(|(part, _)| part).collect::<Vec<_>>();
        let mut remainder = self.clone();
        let mut extracted = self.clone();
        for part in &all_parts {
            if component.contains(*part) {
                if remainder.part(*part).is_some() {
                    remainder
                        .apply(BuildCommand::Remove(*part))
                        .expect("partition removes live parts");
                }
            } else if extracted.part(*part).is_some() {
                extracted
                    .apply(BuildCommand::Remove(*part))
                    .expect("partition removes live parts");
            }
        }
        GraphPartition {
            remainder,
            component: extracted,
        }
    }

    /// Applies an edit transactionally. On failure, no mutation is retained.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when a handle is stale or connection geometry is invalid.
    pub fn apply(&mut self, command: BuildCommand) -> Result<BuildOutcome, GraphError> {
        let mut staged = self.clone();
        let outcome = staged.apply_validated(command)?;
        *self = staged;
        Ok(outcome)
    }

    /// Applies a batch atomically while cloning the graph only once. This is
    /// intended for benchmark scene generation and bulk UI paste operations.
    ///
    /// # Errors
    ///
    /// Returns the first [`GraphError`] and retains none of the batch mutations.
    pub fn apply_batch(
        &mut self,
        commands: impl IntoIterator<Item = BuildCommand>,
    ) -> Result<Vec<BuildOutcome>, GraphError> {
        let mut staged = self.clone();
        let outcomes = commands
            .into_iter()
            .map(|command| staged.apply_validated(command))
            .collect::<Result<Vec<_>, _>>()?;
        *self = staged;
        Ok(outcomes)
    }
}

#[cfg(test)]
mod tests;
