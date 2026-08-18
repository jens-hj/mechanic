use mechanic_core::{BearingId, CompiledCreation};

use crate::{MAX_BEARINGS, MAX_BODIES};

/// Fixed-capacity resource whose limit was exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityKind {
    /// Compound-body rows.
    Bodies,
    /// Passive-bearing rows.
    Bearings,
    /// Broadphase candidate pairs.
    ContactPairs,
}

/// Terminal failure details for the currently loaded simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum FailureStatus {
    /// Topology failed CPU validation or exact-coordinate compilation.
    InvalidTopology(String),
    /// Closure solver exhausted its convergence budget.
    ConstraintNonConvergence {
        /// Bearing nearest the largest closure residual.
        bearing: Option<BearingId>,
        /// Largest derived anchor residual in metres.
        anchor_residual_meters: f32,
        /// Largest derived axis residual in degrees.
        axis_residual_degrees: f32,
    },
    /// A fixed GPU buffer was too small. No rows were dropped.
    CapacityOverflow {
        /// Overflowed resource.
        kind: CapacityKind,
        /// Rows required by the scene or tick.
        required: usize,
        /// Rows provisioned by the runtime.
        capacity: usize,
    },
    /// A physics kernel observed invalid numeric state.
    InvalidNumericState,
    /// Shared render/compute device was lost or rejected work.
    DeviceFailure(String),
}

/// High-level state visible to application UI.
#[derive(Clone, Debug, PartialEq)]
pub enum SimulationStatus {
    /// CPU graph may be edited.
    Building,
    /// A valid compiled scene is paused and ready to run.
    Paused,
    /// GPU fixed scheduler is accepting ticks.
    Running,
    /// Current load or tick failed; editing remains disabled until pause/reload.
    Failed(FailureStatus),
}

/// Opaque reference to the newest complete pair of GPU snapshot slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublishedGpuState {
    /// Changes whenever a new scene is loaded.
    pub generation: u64,
    /// Older complete snapshot slot.
    pub previous_slot: u8,
    /// Newest complete snapshot slot.
    pub current_slot: u8,
    /// Tick sequence in `previous_slot`.
    pub previous_tick: u64,
    /// Tick sequence in `current_slot`.
    pub current_tick: u64,
    /// Compound rows available to GPU-driven rendering.
    pub body_count: u32,
}

/// Last completed tick diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickStatistics {
    /// Monotonic completed tick count.
    pub tick_index: u64,
    /// Rolling completed ticks per wall-clock second.
    pub physics_tps: f32,
    /// GPU timestamp duration for all physics passes.
    pub gpu_tick_ms: f32,
    /// Active compound count.
    pub body_count: u32,
    /// Active persistent contact count.
    pub contact_count: u32,
    /// Largest bearing anchor residual in metres.
    pub anchor_residual_meters: f32,
    /// Largest bearing axis residual in degrees.
    pub axis_residual_degrees: f32,
}

/// CPU-facing lifecycle and publication state for the GPU physics scheduler.
#[derive(Clone, Debug)]
pub struct PhysicsRuntime {
    status: SimulationStatus,
    creation: Option<CompiledCreation>,
    statistics: TickStatistics,
    published: Option<PublishedGpuState>,
    generation: u64,
}

impl Default for PhysicsRuntime {
    fn default() -> Self {
        Self {
            status: SimulationStatus::Building,
            creation: None,
            statistics: TickStatistics::default(),
            published: None,
            generation: 0,
        }
    }
}

impl PhysicsRuntime {
    /// Creates an unloaded runtime in build mode.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates fixed capacities and stages an immutable creation for upload.
    ///
    /// # Errors
    ///
    /// Returns [`FailureStatus::CapacityOverflow`] if body or bearing capacity is exceeded.
    pub fn load(&mut self, creation: CompiledCreation) -> Result<(), FailureStatus> {
        if creation.compounds.len() > MAX_BODIES {
            return self.fail_capacity(CapacityKind::Bodies, creation.compounds.len(), MAX_BODIES);
        }
        if creation.bearings.len() > MAX_BEARINGS {
            return self.fail_capacity(
                CapacityKind::Bearings,
                creation.bearings.len(),
                MAX_BEARINGS,
            );
        }

        self.generation = self.generation.wrapping_add(1);
        let body_count = u32::try_from(creation.compounds.len()).unwrap_or(u32::MAX);
        self.creation = Some(creation);
        self.statistics = TickStatistics {
            body_count,
            ..TickStatistics::default()
        };
        self.published = Some(PublishedGpuState {
            generation: self.generation,
            previous_slot: 0,
            current_slot: 0,
            previous_tick: 0,
            current_tick: 0,
            body_count,
        });
        self.status = SimulationStatus::Paused;
        Ok(())
    }

    /// Enters fixed-step simulation after a successful load.
    ///
    /// # Errors
    ///
    /// Returns the current failure or an invalid-topology status when no paused
    /// compiled scene is ready.
    pub fn start(&mut self) -> Result<(), FailureStatus> {
        match self.status {
            SimulationStatus::Paused => {
                self.status = SimulationStatus::Running;
                Ok(())
            }
            SimulationStatus::Failed(ref failure) => Err(failure.clone()),
            SimulationStatus::Building | SimulationStatus::Running => Err(
                FailureStatus::InvalidTopology("no paused compiled scene is ready".to_owned()),
            ),
        }
    }

    /// Pauses simulation. A failed scene returns to editable build mode.
    pub fn pause(&mut self) {
        if matches!(self.status, SimulationStatus::Failed(_)) {
            self.creation = None;
            self.published = None;
            self.status = SimulationStatus::Building;
        } else {
            self.status = if self.creation.is_some() {
                SimulationStatus::Paused
            } else {
                SimulationStatus::Building
            };
        }
    }

    /// Current lifecycle state.
    pub const fn status(&self) -> &SimulationStatus {
        &self.status
    }

    /// Current terminal failure, if any.
    pub fn failure(&self) -> Option<&FailureStatus> {
        match &self.status {
            SimulationStatus::Failed(failure) => Some(failure),
            _ => None,
        }
    }

    /// Last completed tick statistics.
    pub const fn tick_statistics(&self) -> TickStatistics {
        self.statistics
    }

    /// Latest complete GPU snapshot pair.
    pub const fn published_state(&self) -> Option<PublishedGpuState> {
        self.published
    }

    /// Immutable compiled creation currently staged on the GPU.
    pub const fn creation(&self) -> Option<&CompiledCreation> {
        self.creation.as_ref()
    }

    /// Publishes a successfully validated GPU tick.
    pub fn publish_tick(&mut self, mut statistics: TickStatistics) {
        if self.status != SimulationStatus::Running {
            return;
        }
        let next_tick = self.statistics.tick_index.wrapping_add(1);
        statistics.tick_index = next_tick;
        statistics.body_count = self.statistics.body_count;
        self.statistics = statistics;
        let next_slot = u8::try_from(next_tick % 3).unwrap_or(0);
        if let Some(published) = &mut self.published {
            published.previous_slot = published.current_slot;
            published.previous_tick = published.current_tick;
            published.current_slot = next_slot;
            published.current_tick = next_tick;
        }
    }

    /// Blocks future publication after a terminal GPU failure.
    pub fn fail(&mut self, failure: FailureStatus) {
        self.status = SimulationStatus::Failed(failure);
    }

    fn fail_capacity(
        &mut self,
        kind: CapacityKind,
        required: usize,
        capacity: usize,
    ) -> Result<(), FailureStatus> {
        let failure = FailureStatus::CapacityOverflow {
            kind,
            required,
            capacity,
        };
        self.status = SimulationStatus::Failed(failure.clone());
        Err(failure)
    }
}

#[cfg(test)]
mod tests {
    use bevy_math::IVec3;
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, GridRotation,
    };

    use super::{PhysicsRuntime, SimulationStatus, TickStatistics};

    fn one_body() -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        assert!(matches!(
            graph.apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::ZERO, GridRotation::default())
                )
                .unwrap()
            )),
            Ok(BuildOutcome::Spawned(_))
        ));
        graph.compile().unwrap()
    }

    #[test]
    fn snapshots_advance_only_for_running_successful_ticks() {
        let mut runtime = PhysicsRuntime::new();
        runtime.load(one_body()).unwrap();
        runtime.publish_tick(TickStatistics::default());
        assert_eq!(runtime.published_state().unwrap().current_tick, 0);

        runtime.start().unwrap();
        runtime.publish_tick(TickStatistics::default());
        runtime.publish_tick(TickStatistics::default());
        let published = runtime.published_state().unwrap();
        assert_eq!((published.previous_tick, published.current_tick), (1, 2));
        assert_eq!((published.previous_slot, published.current_slot), (1, 2));
        assert_eq!(runtime.status(), &SimulationStatus::Running);
    }
}
