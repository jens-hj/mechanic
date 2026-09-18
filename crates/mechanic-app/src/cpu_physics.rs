//! Default CPU solver route; `MECHANIC_PHYSICS=gpu` selects the GPU runtime.
//!
//! The GPU runtime stays resident and keeps owning drive resolution and every
//! buffer the renderer reads; this route replaces the tick itself, stepping
//! `mechanic_physics::CpuMachine` against its own terrain scene, kept current
//! with the chunks around the bodies, and publishing the same body/joint state
//! the readback would. See `docs/physics-cpu.md`.

use bevy::math::{DQuat, DVec3};
use mechanic_core::{CompiledCreation, CoordinateDrive};
use mechanic_gpu::{
    GpuExternalImpulse, GpuMechanismCoordinate, GpuMechanismDrive, GpuTransform, GpuVelocity,
};
use mechanic_physics::{
    BodyPose, CpuMachine, DriveCommand, ExternalImpulse, MachineKinematics, MachineState,
    PhysicsError, SoftStepConfig, SoftStepTerrain, TerrainContactScene,
};
use mechanic_world::{TerrainMeshChunk, TerrainNodeId};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// Which solver advances published ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Route {
    /// The GPU runtime.
    Gpu,
    /// The CPU solver.
    #[default]
    Cpu,
}

/// Reads the route from an explicit setting, defaulting to the CPU solver.
pub(crate) fn route_from(value: Option<&str>) -> Route {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("gpu") => Route::Gpu,
        _ => Route::Cpu,
    }
}

/// The route this process runs, read once from `MECHANIC_PHYSICS`.
pub(crate) fn route() -> Route {
    static ROUTE: OnceLock<Route> = OnceLock::new();
    *ROUTE.get_or_init(|| {
        route_from(
            std::env::var("MECHANIC_PHYSICS")
                .ok()
                .filter(|value| !value.is_empty())
                .as_deref(),
        )
    })
}

/// Whether published ticks come from the CPU solver.
pub(crate) fn selected() -> bool {
    route() == Route::Cpu
}

/// One completed CPU tick, in exactly the form a GPU readback publishes.
pub(crate) struct Completed {
    pub(crate) sequence: u64,
    /// Body poses for the renderer and world walking.
    pub(crate) transforms: Vec<GpuTransform>,
    /// Body linear and angular velocity.
    pub(crate) velocities: Vec<GpuVelocity>,
    /// Joint coordinate positions and rates.
    pub(crate) coordinates: Vec<GpuMechanismCoordinate>,
}

/// A CPU solver bound to one published creation and terrain cut.
pub(crate) struct CpuRoute {
    machine: CpuMachine,
    creation: CompiledCreation,
    base_creation: CompiledCreation,
    clump_ids: Vec<u64>,
    clump_shapes: Vec<(u64, u32, bevy::math::DVec3)>,
    clump_origin: DVec3,
    geometry: mechanic_physics::MachineCollisionGeometry,
    scene: TerrainContactScene,
    /// Mesh generation of every chunk in `scene`.
    chunks: BTreeMap<TerrainNodeId, u64>,
    generation: u64,
    /// Terrain publications applied to `scene`, which must keep increasing.
    publication: u64,
    origin: DVec3,
    /// Whether terrain has been published at least once.
    published: bool,
    /// App tick at publication; earlier commands belong to a retired scene.
    base_tick: u64,
    settings: SoftStepConfig,
    /// Ticks since this route was built that hit a numerical fallback.
    degraded_ticks: u64,
}

/// Immutable construction collision data prepared alongside graph compilation.
/// Live poses are supplied only when this revision is installed.
pub(crate) struct PreparedRoute {
    creation: CompiledCreation,
    geometry: mechanic_physics::MachineCollisionGeometry,
    generation: u64,
}

impl PreparedRoute {
    pub(crate) fn new(creation: &CompiledCreation, generation: u64) -> Result<Self, String> {
        let geometry = mechanic_physics::MachineCollisionGeometry::new(creation, generation)
            .map_err(|error| unsupported(&error))?;
        Ok(Self {
            creation: creation.clone(),
            geometry,
            generation,
        })
    }

    pub(crate) fn install(
        self,
        generation: u64,
        base_tick: u64,
        transforms: &[GpuTransform],
        velocities: &[GpuVelocity],
        coordinates: &[GpuMechanismCoordinate],
    ) -> Result<CpuRoute, String> {
        if self.generation != generation {
            return Err(
                "CPU collision preparation belongs to a different construction revision".to_owned(),
            );
        }
        let state = machine_state(&self.creation, transforms, velocities, coordinates)?;
        let machine = CpuMachine::new(self.creation.clone(), generation, state)
            .map_err(|error| unsupported(&error))?;
        Ok(CpuRoute {
            machine,
            base_creation: self.creation.clone(),
            clump_ids: Vec::new(),
            clump_shapes: Vec::new(),
            clump_origin: DVec3::ZERO,
            creation: self.creation,
            geometry: self.geometry,
            scene: TerrainContactScene::default(),
            chunks: BTreeMap::new(),
            generation,
            publication: 0,
            origin: DVec3::ZERO,
            published: false,
            base_tick,
            settings: SoftStepConfig::default(),
            degraded_ticks: 0,
        })
    }
}

impl CpuRoute {
    pub(crate) fn has_clumps(&self) -> bool {
        !self.clump_ids.is_empty()
    }
    pub(crate) fn publish_material<'a>(
        &mut self,
        chunks: impl Iterator<Item = &'a TerrainMeshChunk>,
        clumps: &mechanic_world::ClumpCollection,
        origin: DVec3,
    ) -> Result<(), String> {
        let chunks = chunks.collect::<Vec<_>>();
        let publication = self.publication + 1;
        let mut scene = TerrainContactScene::default();
        let upserts = chunks
            .iter()
            .map(|chunk| Arc::new(chunk.collision_chunk()))
            .collect::<Vec<_>>();
        scene
            .publish(publication, &upserts, &[])
            .map_err(|error| error.to_string())?;
        let prepared = mechanic_physics::PreparedClumpBodies::new(
            &self.base_creation,
            &self.machine.snapshot().state,
            clumps,
            origin,
            self.generation,
        )
        .map_err(|error| error.to_string())?;
        self.machine
            .replace_bodies(
                prepared.creation.clone(),
                prepared.state,
                self.base_creation.compounds.len(),
            )
            .map_err(|error| error.to_string())?;
        self.creation = prepared.creation;
        self.geometry = prepared.geometry;
        self.clump_ids = prepared.ids;
        self.clump_origin = origin;
        self.clump_shapes = clumps
            .bodies
            .values()
            .map(|body| (body.id, body.quanta, body.half_extents))
            .collect();
        self.scene = scene;
        self.publication = publication;
        self.chunks = chunks
            .iter()
            .map(|chunk| (chunk.node, chunk.generation))
            .collect();
        self.origin = origin;
        self.published = true;
        Ok(())
    }

    /// Validates runtime body replacement before installing any changed rows.
    pub(crate) fn publish_clumps(
        &mut self,
        clumps: &mechanic_world::ClumpCollection,
    ) -> Result<(), String> {
        let shapes = clumps
            .bodies
            .values()
            .map(|body| (body.id, body.quanta, body.half_extents))
            .collect::<Vec<_>>();
        if shapes == self.clump_shapes && self.clump_origin == self.origin {
            return Ok(());
        }
        let prepared = mechanic_physics::PreparedClumpBodies::new(
            &self.base_creation,
            &self.machine.snapshot().state,
            clumps,
            self.origin,
            self.generation,
        )
        .map_err(|error| format!("cannot prepare material bodies: {error}"))?;
        self.machine
            .replace_bodies(
                prepared.creation.clone(),
                prepared.state,
                self.base_creation.compounds.len(),
            )
            .map_err(|error| format!("cannot publish material bodies: {error}"))?;
        self.creation = prepared.creation;
        self.geometry = prepared.geometry;
        self.clump_ids = prepared.ids;
        self.clump_shapes = shapes;
        self.clump_origin = self.origin;
        Ok(())
    }

    /// Copies accepted clump poses into persistent world ownership.
    pub(crate) fn update_clumps(&self, world: &mut crate::world::WorldRuntime) {
        if self.clump_ids.iter().any(|id| {
            world
                .clumps
                .bodies
                .get(id)
                .is_some_and(|body| !body.sleeping)
        }) {
            world.material_motion();
        }
        let state = &self.machine.snapshot().state;
        let Ok(motions) =
            MachineKinematics::published_motions(&self.creation, &state.poses, &state.velocities)
        else {
            return;
        };
        for (offset, id) in self.clump_ids.iter().enumerate() {
            let row = self.base_creation.compounds.len() + offset;
            if let Some(body) = world.clumps.bodies.get_mut(id) {
                body.position =
                    mechanic_world::WorldPosition(self.origin + state.poses[row].position);
                body.rotation = state.poses[row].rotation;
                body.linear_velocity = motions[row].linear;
                body.angular_velocity = motions[row].angular;
                let supported = self.machine.terrain_loads().iter().any(|load| {
                    load.body == row && load.normal.y > 0.25 && load.normal_impulse > 0.0
                });
                if !body.sleeping {
                    let soft =
                        mechanic_world::BreakageResponse::for_material(body.material).deposits;
                    body.update_settling(
                        if soft {
                            supported
                        } else {
                            self.machine.body_supported(row)
                        },
                        mechanic_core::TICK_SECONDS,
                    );
                    if !mechanic_world::BreakageResponse::for_material(body.material).deposits
                        && body.settled_seconds >= 1.0
                    {
                        body.sleeping = true;
                        body.linear_velocity = DVec3::ZERO;
                        body.angular_velocity = DVec3::ZERO;
                    }
                }
            }
        }
    }

    pub(crate) fn prepare_clump_tick(
        &mut self,
        clumps: &mut mechanic_world::ClumpCollection,
    ) -> Result<(), String> {
        let state = &self.machine.snapshot().state;
        let motions =
            MachineKinematics::published_motions(&self.creation, &state.poses, &state.velocities)
                .map_err(|error| error.to_string())?;
        let mut available = clumps.available();
        for (offset, id) in self.clump_ids.iter().enumerate() {
            let Some(body) = clumps.bodies.get_mut(id) else {
                continue;
            };
            if !body.sleeping || available == 0 {
                continue;
            }
            let own_row = self.base_creation.compounds.len() + offset;
            let wake = self.creation.colliders.iter().any(|collider| {
                let row = collider.compound_index as usize;
                if row == own_row
                    || motions[row].linear.length() + motions[row].angular.length() < 0.05
                {
                    return false;
                }
                let radius = match &collider.shape {
                    mechanic_core::ColliderShape::Cuboid { half_extents, .. } => {
                        f64::from(half_extents.length())
                    }
                    mechanic_core::ColliderShape::Convex(convex) => convex
                        .vertices
                        .iter()
                        .map(|v| f64::from(v.length()))
                        .fold(0.0, f64::max),
                };
                let centre = state.poses[row].position
                    + state.poses[row].rotation * collider.local_center.as_dvec3();
                centre.distance(body.position.0 - self.origin)
                    < radius + body.half_extents.length() + 0.05
            });
            if wake {
                body.sleeping = false;
                body.settled_seconds = 0.0;
                available -= 1;
            }
        }
        let sleeping = self
            .clump_ids
            .iter()
            .map(|id| clumps.bodies.get(id).is_some_and(|body| body.sleeping))
            .collect::<Vec<_>>();
        self.machine
            .hold_runtime(self.base_creation.compounds.len(), &sleeping)
            .map_err(|error| error.to_string())
    }

    /// Feed accepted CPU loads to world-owned compaction using this query's origin.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "world density and pressure use f32"
    )]
    pub(crate) fn accumulate_soil(&self, world: &mut crate::world::WorldRuntime) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| {
            !std::env::var("MECHANIC_SOIL")
                .is_ok_and(|value| value.trim().eq_ignore_ascii_case("off"))
        }) {
            return;
        }
        world.accumulate_soil(self.machine.terrain_loads().iter().map(|load| {
            let area = std::f64::consts::PI * load.patch_radius.powi(2);
            mechanic_world::SoilPatch {
                centre: mechanic_world::WorldPosition(self.origin + load.point),
                normal: load.normal,
                radius: load.patch_radius,
                pressure_pa: (load.normal_impulse / (mechanic_core::TICK_SECONDS * area)) as f32,
                seconds: mechanic_core::TICK_SECONDS as f32,
            }
        }));
        world.accumulate_breakage(
            self.machine
                .terrain_loads()
                .iter()
                .filter(|load| load.body < self.base_creation.compounds.len())
                .map(|load| {
                    let area = std::f64::consts::PI * load.patch_radius.powi(2);
                    mechanic_world::BreakagePatch {
                        centre: mechanic_world::WorldPosition(self.origin + load.point),
                        normal: load.normal,
                        radius: load.patch_radius,
                        stress_pa: load.footprint_impulse / (mechanic_core::TICK_SECONDS * area),
                        work_j: load.work_j,
                    }
                }),
        );
    }

    /// Transfers the world-owned terrain cut across a construction publication.
    /// Body state, contacts and construction collision geometry remain those of
    /// the new route. The next publication still reconciles remeshed/removed chunks.
    pub(crate) fn inherit_terrain(&mut self, previous: &mut Self) {
        self.scene = std::mem::take(&mut previous.scene);
        self.chunks = std::mem::take(&mut previous.chunks);
        self.publication = previous.publication;
        self.origin = previous.origin;
        self.published = std::mem::take(&mut previous.published);
    }

    /// Binds the CPU solver to a published creation, starting from the live body
    /// and joint state the GPU route would have uploaded.
    ///
    /// # Errors
    /// Returns a message naming what the CPU solver cannot run, so the caller can
    /// refuse the publication instead of silently falling back to the GPU.
    #[cfg(test)]
    pub(crate) fn new(
        creation: &CompiledCreation,
        generation: u64,
        base_tick: u64,
        transforms: &[GpuTransform],
        velocities: &[GpuVelocity],
        coordinates: &[GpuMechanismCoordinate],
    ) -> Result<Self, String> {
        PreparedRoute::new(creation, generation)?.install(
            generation,
            base_tick,
            transforms,
            velocities,
            coordinates,
        )
    }

    /// Brings the collision scene up to the terrain around the bodies. Called every
    /// frame, independent of GPU terrain preparation, so CPU ticks never wait on
    /// it; only chunks that appeared, changed generation or left are published.
    ///
    /// # Errors
    /// Returns a message when the chunk geometry or the frame is invalid.
    pub(crate) fn publish_terrain<'a>(
        &mut self,
        chunks: impl IntoIterator<Item = &'a TerrainMeshChunk>,
        origin: DVec3,
    ) -> Result<(), String> {
        let started = std::time::Instant::now();
        let chunks = chunks.into_iter().collect::<Vec<_>>();
        let current = chunks
            .iter()
            .map(|mesh| (mesh.node, mesh.generation))
            .collect::<BTreeMap<_, _>>();
        self.origin = origin;
        if self.published && current == self.chunks {
            crate::performance_capture::record(
                "cpu_terrain_update",
                || serde_json::json!({"duration_ms":started.elapsed().as_secs_f64()*1000.0,"changed_chunks":0}),
            );
            return Ok(());
        }
        let upserts = chunks
            .iter()
            .filter(|mesh| self.chunks.get(&mesh.node) != Some(&mesh.generation))
            .map(|mesh| Arc::new(mesh.collision_chunk()))
            .collect::<Vec<_>>();
        let removed = self
            .chunks
            .keys()
            .filter(|node| !current.contains_key(node))
            .copied()
            .collect::<Vec<_>>();
        self.publication += 1;
        if self
            .scene
            .publish(self.publication, &upserts, &removed)
            .is_err()
        {
            // A node whose generation went backwards cannot update in place, so
            // the whole cut is rebuilt.
            let upserts = chunks
                .iter()
                .map(|mesh| Arc::new(mesh.collision_chunk()))
                .collect::<Vec<_>>();
            let mut scene = TerrainContactScene::default();
            scene
                .publish(self.publication, &upserts, &[])
                .map_err(|error| format!("cannot publish terrain to the CPU solver: {error}"))?;
            self.scene = scene;
        }
        crate::performance_capture::record(
            "cpu_terrain_update",
            || serde_json::json!({"duration_ms":started.elapsed().as_secs_f64()*1000.0,"changed_chunks":upserts.len()+removed.len()}),
        );
        self.chunks = current;
        self.published = true;
        Ok(())
    }

    /// Whether a tick can run: the CPU solver has no ground plane of its own, so
    /// it must never step before a cut is published.
    pub(crate) fn is_ready(&self) -> bool {
        self.published
    }

    /// Advances one published tick.
    ///
    /// # Errors
    /// Returns a diagnostic message. A failed tick leaves the previous snapshot
    /// untouched, so the caller must stop the simulation and show it rather than
    /// publish anything.
    pub(crate) fn step(
        &mut self,
        tick: u64,
        gravity: DVec3,
        drives: &[GpuMechanismDrive],
        impulses: &[GpuExternalImpulse],
    ) -> Result<Completed, String> {
        let started = std::time::Instant::now();
        tick.checked_sub(self.base_tick)
            .ok_or_else(|| format!("tick {tick} precedes this CPU publication"))?;
        // The app drops overdue ticks by skipping their labels. The CPU machine
        // counts only completed steps, so commands must use its next local tick.
        let machine_tick = self
            .machine
            .snapshot()
            .tick
            .checked_add(1)
            .ok_or_else(|| "CPU tick counter exhausted".to_owned())?;
        let commands = drives
            .iter()
            .enumerate()
            .map(|(coordinate, &row)| DriveCommand {
                tick: machine_tick,
                topology_generation: self.generation,
                coordinate,
                drive: CoordinateDrive::from(row),
            })
            .collect::<Vec<_>>();
        let impulses = impulses
            .iter()
            .map(|row| ExternalImpulse {
                tick: machine_tick,
                topology_generation: self.generation,
                body: row.metadata[0] as usize,
                point: DVec3::new(
                    f64::from(row.world_point[0]),
                    f64::from(row.world_point[1]),
                    f64::from(row.world_point[2]),
                ),
                impulse: DVec3::new(
                    f64::from(row.impulse[0]),
                    f64::from(row.impulse[1]),
                    f64::from(row.impulse[2]),
                ),
            })
            .collect::<Vec<_>>();
        let terrain = SoftStepTerrain {
            scene: &self.scene,
            geometry: &self.geometry,
            topology_generation: self.generation,
            origin: self.origin,
        };
        let outcome =
            self.machine
                .step(gravity, &self.settings, &impulses, &commands, Some(terrain));
        if let Err(error) = outcome {
            return Err(self.failure(tick, &error));
        }
        if self.machine.diagnostics().degraded {
            self.degraded_ticks += 1;
        }
        let publication_started = std::time::Instant::now();
        let completed = self.published_state()?;
        let publication_ms = publication_started.elapsed().as_secs_f64() * 1000.0;
        crate::performance_capture::record("physics_cpu_tick", || {
            let d = self.machine.diagnostics();
            serde_json::json!({"tick": tick, "sequence": machine_tick, "route": "cpu", "duration_ms": started.elapsed().as_secs_f64()*1000.0, "conversion_ms": publication_ms, "query_ms": d.query_ms, "solve_ms": d.solve_ms, "dynamics_ms": d.dynamics_ms, "rows_ms": d.rows_ms, "constraints_ms": d.constraints_ms, "continuous_ms": d.continuous_ms, "requeries": d.requeries, "empty_contact_reuses": d.empty_contact_reuses, "refreshed_contact_groups": d.refreshed_contact_groups, "reused_contact_groups": d.reused_contact_groups, "detailed_sweep_preparations": d.detailed_sweep_preparations, "continuous_cached_supports":d.continuous_cached_supports, "clearance_certificate_failures": d.clearance_certificate_failures, "continuous_sweeps": d.continuous_sweeps, "continuous_hits": d.continuous_hits,"continuous_shape_transformations":d.continuous_shape_transformations,"continuous_shape_cache_hits":d.continuous_shape_cache_hits,"continuous_hierarchy_node_pair_tests":d.continuous_hierarchy_node_pair_tests,"continuous_pose_evaluations":d.continuous_pose_evaluations,"continuous_velocity_evaluations":d.continuous_velocity_evaluations,"continuous_separation_evaluations":d.continuous_separation_evaluations,"continuous_collider_pair_candidates":d.continuous_collider_pair_candidates,"continuous_triangle_candidates":d.continuous_triangle_candidates, "degraded": d.degraded, "degraded_reason": d.degraded_reason, "contacts": d.contacts, "rows": d.rows, "triangle_candidates":d.triangle_candidates, "collider_pair_candidates":d.collider_pair_candidates,"solver_scratch_bytes":d.solver_scratch_bytes,"solver_scratch_growth_bytes":d.solver_scratch_growth_bytes})
        });
        Ok(completed)
    }

    /// Why a tick was refused. The soft-step solver only refuses invalid input.
    fn failure(&self, tick: u64, error: &PhysicsError) -> String {
        format!(
            "the CPU solver refused tick {tick}: {error} ({} earlier ticks degraded). \
             Set MECHANIC_PHYSICS=gpu to run the GPU solver.",
            self.degraded_ticks
        )
    }

    /// Holds whole mechanisms at prescribed poses, like the GPU runtime's body
    /// holds; poses of bodies not held are ignored.
    ///
    /// # Errors
    /// Returns a message when the solver refuses the mask or a held pose.
    pub(crate) fn hold(&mut self, held: &[bool], poses: &[GpuTransform]) -> Result<(), String> {
        let mut poses = poses.iter().map(body_pose).collect::<Vec<_>>();
        poses.extend_from_slice(
            &self.machine.snapshot().state.poses[self.base_creation.compounds.len()..],
        );
        let mut held = held.to_vec();
        held.resize(poses.len(), false);
        self.machine
            .hold(&held, &poses)
            .map_err(|error| format!("the CPU solver refused a body hold: {error}"))
    }

    /// Ticks since this route was built that fell back after numerical trouble.
    pub(crate) fn degraded_ticks(&self) -> u64 {
        self.degraded_ticks
    }

    /// Converts the committed snapshot into the publication the renderer reads.
    fn published_state(&self) -> Result<Completed, String> {
        let state = &self.machine.snapshot().state;
        let motions =
            MachineKinematics::published_motions(&self.creation, &state.poses, &state.velocities)
                .map_err(|error| format!("cannot reconstruct CPU body velocities: {error}"))?;
        let transforms = state
            .poses
            .iter()
            .take(self.base_creation.compounds.len())
            .map(|pose| {
                let position = pose.position.as_vec3();
                GpuTransform {
                    position: [position.x, position.y, position.z, 0.0],
                    rotation: pose.rotation.as_quat().normalize().to_array(),
                }
            })
            .collect();
        let velocities = motions
            .iter()
            .take(self.base_creation.compounds.len())
            .map(|motion| {
                let linear = motion.linear.as_vec3();
                let angular = motion.angular.as_vec3();
                GpuVelocity {
                    linear: [linear.x, linear.y, linear.z, 0.0],
                    angular: [angular.x, angular.y, angular.z, 0.0],
                }
            })
            .collect();
        let coordinates = self
            .creation
            .dynamics
            .coordinate_velocities
            .iter()
            .enumerate()
            .map(|(coordinate, &row)| GpuMechanismCoordinate {
                position: narrow(state.coordinates[coordinate]),
                velocity: narrow(state.velocities[row]),
            })
            .collect();
        Ok(Completed {
            sequence: self.machine.snapshot().tick,
            transforms,
            velocities,
            coordinates,
        })
    }
}

/// Names what the CPU solver cannot run, in the user's terms.
fn unsupported(error: &PhysicsError) -> String {
    format!(
        "the CPU solver cannot run this creation: {error}. \
         Set MECHANIC_PHYSICS=gpu to run the GPU solver."
    )
}

/// Seeds generalized state from the body and joint state the app carries across
/// publications. Roots own six velocity rows; every other row is a joint rate.
fn machine_state(
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    velocities: &[GpuVelocity],
    coordinates: &[GpuMechanismCoordinate],
) -> Result<MachineState, String> {
    let bodies = creation.compounds.len();
    if transforms.len() != bodies
        || velocities.len() != bodies
        || coordinates.len() != creation.dynamics.coordinate_velocities.len()
    {
        return Err(format!(
            "published state does not match the compiled creation: {} transforms, \
             {} velocities and {} coordinates for {bodies} bodies",
            transforms.len(),
            velocities.len(),
            coordinates.len()
        ));
    }
    let poses = transforms.iter().map(body_pose).collect();
    // Rows are assigned in preorder, not body order, so only the elimination tree
    // states how many generalized velocities the creation has.
    let mut rates = vec![0.0; creation.dynamics.elimination_parent.len()];
    for (body, rows) in creation.dynamics.body_velocities.iter().enumerate() {
        if rows.len() != 6 {
            continue;
        }
        let velocity = velocities[body];
        for (offset, value) in [
            velocity.linear[0],
            velocity.linear[1],
            velocity.linear[2],
            velocity.angular[0],
            velocity.angular[1],
            velocity.angular[2],
        ]
        .into_iter()
        .enumerate()
        {
            rates[rows.start + offset] = f64::from(value);
        }
    }
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        rates[row] = f64::from(coordinates[coordinate].velocity);
    }
    Ok(MachineState {
        poses,
        coordinates: coordinates
            .iter()
            .map(|coordinate| f64::from(coordinate.position))
            .collect(),
        velocities: rates,
    })
}

fn body_pose(transform: &GpuTransform) -> BodyPose {
    BodyPose {
        position: DVec3::new(
            f64::from(transform.position[0]),
            f64::from(transform.position[1]),
            f64::from(transform.position[2]),
        ),
        rotation: DQuat::from_xyzw(
            f64::from(transform.rotation[0]),
            f64::from(transform.rotation[1]),
            f64::from(transform.rotation[2]),
            f64::from(transform.rotation[3]),
        )
        .normalize(),
    }
}

/// Gravity the GPU runtime applies, in metres per second squared.
pub(crate) fn gravity() -> DVec3 {
    mechanic_core::GRAVITY
}

/// The publication format is `f32` by design; the solver's own state stays `f64`.
#[expect(clippy::cast_possible_truncation)]
fn narrow(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests;
