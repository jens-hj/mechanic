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
    PhysicsError, SoftStepSettings, SoftStepTerrain, TerrainContactScene,
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
    settings: SoftStepSettings,
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
            settings: SoftStepSettings::default(),
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
    #[allow(clippy::cast_possible_truncation)] // World density and pressure use f32.
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
#[allow(clippy::cast_possible_truncation)]
fn narrow(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    #[test]
    fn clumps_keep_simulating_without_authored_bodies_and_rebase_globally() {
        use mechanic_world::{ClumpCollection, MaterialClump, TerrainMaterial, WorldPosition};
        let base = mechanic_core::CompiledCreation::default();
        let mut route = super::CpuRoute::new(&base, 1, 0, &[], &[], &[]).unwrap();
        let origin = super::DVec3::new(100.0, 0.0, 100.0);
        route.publish_terrain([], origin).unwrap();
        let body = MaterialClump {
            id: 1,
            material: TerrainMaterial::Rock,
            quanta: 510,
            half_extents: super::DVec3::splat(0.025),
            position: WorldPosition(origin + super::DVec3::Y * 2.0),
            rotation: bevy::math::DQuat::IDENTITY,
            linear_velocity: super::DVec3::ZERO,
            angular_velocity: super::DVec3::ZERO,
            settled_seconds: 0.0,
            sleeping: false,
        };
        let mut clumps = ClumpCollection {
            next_id: 2,
            bodies: [(1, body)].into(),
        };
        route.publish_clumps(&clumps).unwrap();
        let tick = route.step(1, super::gravity(), &[], &[]).unwrap();
        assert!(tick.transforms.is_empty());
        let local = route.machine.snapshot().state.poses[0].position;
        assert!(local.y < 2.0);
        let global = origin + local;
        clumps.bodies.get_mut(&1).unwrap().position = WorldPosition(global);
        let next_origin = origin + super::DVec3::X * 100.0;
        route.publish_terrain([], next_origin).unwrap();
        route.publish_clumps(&clumps).unwrap();
        assert!(
            route.machine.snapshot().state.poses[0]
                .position
                .abs_diff_eq(global - next_origin, 1e-9)
        );
    }

    use super::*;
    use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};

    fn cube() -> CompiledCreation {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        graph.compile().unwrap()
    }

    #[test]
    fn only_an_explicit_gpu_setting_leaves_the_cpu_route() {
        assert_eq!(route_from(None), Route::Cpu);
        assert_eq!(route_from(Some("")), Route::Cpu);
        assert_eq!(route_from(Some("cpu")), Route::Cpu);
        assert_eq!(route_from(Some("nonsense")), Route::Cpu);
        assert_eq!(route_from(Some("gpu")), Route::Gpu);
        assert_eq!(route_from(Some(" GPU ")), Route::Gpu);
    }

    #[test]
    fn a_drive_row_survives_the_round_trip_through_the_uploaded_form() {
        let creation = cube();
        for drive in creation.coordinate_drives.iter().copied().chain([
            CoordinateDrive {
                mode: mechanic_core::DriveMode::Speed,
                target_speed: 2.5,
                target_angle: 0.0,
                max_speed: 7.0,
                max_acceleration: 12.0,
                source_a_max_acceleration: 12.0,
                source_a_no_load_speed: 7.0,
                source_b_max_acceleration: 3.0,
                source_b_no_load_speed: 1.5,
                min_angle: f32::NEG_INFINITY,
                max_angle: f32::INFINITY,
            },
            CoordinateDrive {
                mode: mechanic_core::DriveMode::Angle,
                target_speed: 0.0,
                target_angle: -0.75,
                max_speed: 3.0,
                max_acceleration: f32::INFINITY,
                source_a_max_acceleration: f32::INFINITY,
                source_a_no_load_speed: 3.0,
                source_b_max_acceleration: 0.0,
                source_b_no_load_speed: 0.0,
                min_angle: -1.5,
                max_angle: 1.5,
            },
        ]) {
            let row = GpuMechanismDrive::from(drive);
            assert_eq!(CoordinateDrive::from(row), drive);
        }
    }

    // A hinged pair: one free root body plus one joint coordinate.
    fn hinge() -> CompiledCreation {
        use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation, PartId};
        let mut graph = ConstructionGraph::new();
        let mut spawn = |ticks: bevy::math::IVec3| {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                panic!("spawn expected");
            };
            part as PartId
        };
        let root = spawn(bevy::math::IVec3::ZERO);
        let tip = spawn(bevy::math::IVec3::X * 400);
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(root, FaceKind::PositiveX),
                FaceRef::part(tip, FaceKind::NegativeX),
                bevy::math::Vec3::X * 0.5,
                bevy::math::Vec3::X,
            )))
            .unwrap();
        graph.compile().unwrap()
    }

    #[test]
    fn seeding_covers_every_generalized_row_whatever_order_bodies_are_compiled_in() {
        // Velocity rows are assigned in preorder, so the last body's range is not
        // the row count: a root compiled after its child sizes it six rows short.
        let mut creation = hinge();
        creation.dynamics.body_velocities.reverse();
        let transforms = vec![
            GpuTransform {
                position: [0.0, 1.0, 0.0, 0.0],
                rotation: bevy::math::Quat::IDENTITY.to_array(),
            };
            creation.compounds.len()
        ];
        let velocities = vec![
            GpuVelocity {
                linear: [0.5, -1.5, 0.25, 0.0],
                angular: [0.1, 0.2, 0.3, 0.0],
            };
            creation.compounds.len()
        ];
        let coordinates = vec![
            GpuMechanismCoordinate {
                position: 0.25,
                velocity: -0.75,
            };
            creation.dynamics.coordinate_velocities.len()
        ];
        assert_eq!(coordinates.len(), 1);

        let state = machine_state(&creation, &transforms, &velocities, &coordinates).unwrap();

        assert_eq!(
            state.velocities.len(),
            creation.dynamics.elimination_parent.len()
        );
        let row = creation.dynamics.coordinate_velocities[0];
        assert!((state.velocities[row] + 0.75).abs() < 1e-6);
        let root = creation
            .dynamics
            .body_velocities
            .iter()
            .find(|rows| rows.len() == 6)
            .unwrap()
            .clone();
        assert!((state.velocities[root.start + 1] + 1.5).abs() < 1e-6);
        assert!((state.velocities[root.start + 5] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn seeding_reads_root_velocity_rows_and_joint_rates() {
        let creation = cube();
        let transforms = vec![GpuTransform {
            position: [1.0, 2.0, 3.0, 0.0],
            rotation: bevy::math::Quat::IDENTITY.to_array(),
        }];
        let velocities = vec![GpuVelocity {
            linear: [0.5, -1.5, 0.25, 0.0],
            angular: [0.1, 0.2, 0.3, 0.0],
        }];
        let state = machine_state(&creation, &transforms, &velocities, &[]).unwrap();

        assert_eq!(state.poses.len(), 1);
        assert!((state.poses[0].position.y - 2.0).abs() < 1e-6);
        assert_eq!(state.velocities.len(), 6);
        assert!((state.velocities[1] + 1.5).abs() < 1e-6);
        assert!((state.velocities[5] - 0.3).abs() < 1e-6);

        // A state from another creation is refused, never silently padded.
        assert!(machine_state(&creation, &[], &velocities, &[]).is_err());
    }

    // A flat two-triangle floor, in the mesh form the world streams.
    fn floor() -> mechanic_world::TerrainMeshChunk {
        use mechanic_world::{
            TerrainIndexGroups, TerrainMaterial, TerrainMeshChunk, TerrainNodeId,
            TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle,
            WorldBounds, WorldPosition,
        };
        let reach = 8.0_f32;
        let bounds = WorldBounds {
            minimum: WorldPosition(DVec3::new(-8.0, 0.0, -8.0)),
            maximum: WorldPosition(DVec3::new(8.0, 0.0, 8.0)),
        };
        let mut rock = [0.0; TerrainMaterial::COUNT];
        rock[usize::from(TerrainMaterial::Rock.code())] = 1.0;
        TerrainMeshChunk {
            node: TerrainNodeId::ROOT,
            origin: WorldPosition(DVec3::ZERO),
            vertices: vec![
                [-reach, 0.0, -reach],
                [-reach, 0.0, reach],
                [reach, 0.0, reach],
                [reach, 0.0, -reach],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            index_groups: TerrainIndexGroups {
                regular: vec![0, 1, 2, 0, 2, 3],
                ..Default::default()
            },
            material_weights: vec![rock; 4],
            bounds,
            triangle_bvh: TriangleBvh {
                bounds,
                triangles: vec![
                    TriangleBvhTriangle {
                        indices: [0, 1, 2],
                        group_mask: TerrainTriangleGroupMask::REGULAR,
                    },
                    TriangleBvhTriangle {
                        indices: [0, 2, 3],
                        group_mask: TerrainTriangleGroupMask::REGULAR,
                    },
                ],
                nodes: vec![TriangleBvhNode {
                    bounds,
                    first_triangle: 0,
                    triangle_count: 2,
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                    ..Default::default()
                }],
            },
            generation: 1,
            ..Default::default()
        }
    }

    fn dropped_cube(height: f32) -> (CompiledCreation, Vec<GpuTransform>, Vec<GpuVelocity>) {
        let creation = cube();
        let transforms = vec![GpuTransform {
            position: [0.0, height, 0.0, 0.0],
            rotation: bevy::math::Quat::IDENTITY.to_array(),
        }];
        let velocities = vec![GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4],
        }];
        (creation, transforms, velocities)
    }

    #[test]
    fn prepared_collision_installs_latest_body_state_and_rejects_another_revision() {
        let (creation, mut transforms, mut velocities) = dropped_cube(0.5);
        let prepared = PreparedRoute::new(&creation, 7).unwrap();
        // The old simulation continues moving while geometry is prepared.
        transforms[0].position = [3.0, 2.0, -1.0, 0.0];
        velocities[0].linear = [0.6, 0.0, 0.0, 0.0];
        let mut route = prepared
            .install(7, 40, &transforms, &velocities, &[])
            .unwrap();
        let published = route.published_state().unwrap();
        assert!(
            bevy::math::Vec4::from_array(published.transforms[0].position)
                .abs_diff_eq(bevy::math::Vec4::from_array(transforms[0].position), 1.0e-6)
        );
        assert!(
            bevy::math::Vec4::from_array(published.velocities[0].linear)
                .abs_diff_eq(bevy::math::Vec4::from_array(velocities[0].linear), 1.0e-6)
        );
        route.publish_terrain([], DVec3::ZERO).unwrap();
        let next = route.step(41, DVec3::ZERO, &[], &[]).unwrap();
        assert!((next.transforms[0].position[0] - 3.01).abs() < 1.0e-5);
        assert!(
            PreparedRoute::new(&creation, 7)
                .unwrap()
                .install(8, 40, &transforms, &velocities, &[])
                .is_err()
        );
        assert!(
            PreparedRoute::new(&creation, 7)
                .unwrap()
                .install(7, 40, &[], &velocities, &[])
                .is_err()
        );
    }

    #[test]
    fn a_cube_published_on_the_floor_settles_and_keeps_publishing_ticks() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();

        // Ticking before a cut would drop the cube through the world.
        assert!(!route.is_ready());
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        assert!(route.is_ready());

        let mut published = 0;
        for tick in 1..=30 {
            let completed = route
                .step(tick, gravity(), &[], &[])
                .unwrap_or_else(|message| panic!("tick {tick}: {message}"));
            assert_eq!(completed.transforms.len(), 1);
            assert_eq!(completed.velocities.len(), 1);
            let height = completed.transforms[0].position[1];
            assert!(
                (0.495..=0.503).contains(&height),
                "tick {tick} settled at {height}"
            );
            published += 1;
        }
        assert_eq!(published, 30);
        let resting = route.step(31, gravity(), &[], &[]).unwrap();
        // Settled in velocity too, not merely held in place.
        assert!(
            resting.velocities[0].linear[1].abs() < 1e-2,
            "resting vertical velocity {}",
            resting.velocities[0].linear[1]
        );
        assert!((resting.transforms[0].position[1] - 0.5).abs() < 0.005);
        assert_eq!(route.degraded_ticks(), 0);
    }

    #[test]
    fn construction_publication_keeps_terrain_and_still_applies_remeshes_and_removals() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let mut previous = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
        previous.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        let publication = previous.publication;
        let mut replacement =
            CpuRoute::new(&creation, 8, 10, &transforms, &velocities, &[]).unwrap();
        replacement.inherit_terrain(&mut previous);
        assert!(replacement.is_ready());
        assert!(!previous.is_ready());
        replacement
            .publish_terrain([&floor()], DVec3::ZERO)
            .unwrap();
        assert_eq!(
            replacement.publication, publication,
            "unchanged terrain must not be rebuilt after an edit"
        );
        let resting = replacement.step(11, gravity(), &[], &[]).unwrap();
        assert!(resting.transforms[0].position[1] > 0.49);

        let mut remeshed = floor();
        remeshed.generation += 1;
        replacement
            .publish_terrain([&remeshed], DVec3::ZERO)
            .unwrap();
        assert_eq!(replacement.publication, publication + 1);
        replacement.publish_terrain([], DVec3::ZERO).unwrap();
        for tick in 12..=31 {
            let state = replacement.step(tick, gravity(), &[], &[]).unwrap();
            if tick == 31 {
                assert!(state.transforms[0].position[1] < 0.4);
            }
        }
    }

    #[test]
    fn terrain_updates_follow_the_chunks_around_the_bodies_without_republishing_unchanged_ones() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        let publication = route.publication;

        // The same chunks every frame cost nothing.
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        assert_eq!(route.publication, publication);
        let resting = route.step(1, gravity(), &[], &[]).unwrap();
        assert!(resting.transforms[0].position[1] > 0.49);

        // A chunk that left the cut stops supporting the cube.
        route.publish_terrain([], DVec3::ZERO).unwrap();
        let mut fallen = resting;
        for tick in 2..=20 {
            fallen = route.step(tick, gravity(), &[], &[]).unwrap();
        }
        assert!(fallen.transforms[0].position[1] < 0.4);

        // A remeshed chunk comes back as a new generation.
        let mut remeshed = floor();
        remeshed.generation = 2;
        route.publish_terrain([&remeshed], DVec3::ZERO).unwrap();
        assert!(route.publication > publication + 1);
        assert_eq!(route.chunks.get(&remeshed.node), Some(&2));
    }

    #[test]
    fn a_cube_dropped_on_another_cube_rests_on_it_instead_of_falling_through() {
        use mechanic_core::GridRotation;
        let mut graph = ConstructionGraph::new();
        for ticks in [bevy::math::IVec3::ZERO, bevy::math::IVec3::X * 800] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        let transforms = [0.5, 1.52]
            .map(|height| GpuTransform {
                position: [0.0, height, 0.0, 0.0],
                rotation: bevy::math::Quat::IDENTITY.to_array(),
            })
            .to_vec();
        let velocities = vec![
            GpuVelocity {
                linear: [0.0; 4],
                angular: [0.0; 4],
            };
            2
        ];
        let mut route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        let mut last = None;
        for tick in 1..=60 {
            let completed = route
                .step(tick, gravity(), &[], &[])
                .unwrap_or_else(|message| panic!("tick {tick}: {message}"));
            let [lower, upper] = [0, 1].map(|body| completed.transforms[body].position[1]);
            assert!(
                upper - lower >= 0.99,
                "tick {tick}: upper cube sank to {upper} over {lower}"
            );
            last = Some(upper);
        }
        assert!((last.unwrap() - 1.5).abs() < 0.01);
    }

    #[test]
    fn dropped_app_ticks_keep_joint_commands_and_impulses_on_the_next_cpu_step() {
        let creation = hinge();
        let initial = MachineState::at_rest(&creation);
        let transforms = initial
            .poses
            .iter()
            .map(|pose| GpuTransform {
                position: [
                    narrow(pose.position.x),
                    narrow(pose.position.y) + 5.0,
                    narrow(pose.position.z),
                    0.0,
                ],
                rotation: pose.rotation.to_array().map(narrow),
            })
            .collect::<Vec<_>>();
        let velocities = vec![
            GpuVelocity {
                linear: [0.0; 4],
                angular: [0.0; 4]
            };
            transforms.len()
        ];
        let coordinates = vec![
            GpuMechanismCoordinate {
                position: 0.0,
                velocity: 0.0
            };
            creation.dynamics.coordinate_velocities.len()
        ];
        let drives = creation
            .coordinate_drives
            .iter()
            .copied()
            .map(GpuMechanismDrive::from)
            .collect::<Vec<_>>();
        assert_eq!(drives.len(), 1);
        let run = |ticks: [u64; 3]| {
            let mut route =
                CpuRoute::new(&creation, 7, 10, &transforms, &velocities, &coordinates).unwrap();
            route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
            for tick in ticks {
                route
                    .step(
                        tick,
                        gravity(),
                        &drives,
                        &[GpuExternalImpulse::new(
                            0,
                            bevy::math::Vec3::new(0.0, 5.0, 0.0),
                            bevy::math::Vec3::X * 0.01,
                        )],
                    )
                    .unwrap();
            }
            assert_eq!(route.machine.snapshot().tick, 3);
            let completed = route.published_state().unwrap();
            assert!(completed.transforms[0].position[1] < transforms[0].position[1]);
            route.machine.snapshot().state_hash()
        };
        assert_eq!(run([11, 12, 13]), run([11, 23, 41]));
    }

    #[test]
    fn a_tick_before_the_publication_is_refused_rather_than_renumbered() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let mut route = CpuRoute::new(&creation, 7, 10, &transforms, &velocities, &[]).unwrap();
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();

        let Err(message) = route.step(9, gravity(), &[], &[]) else {
            panic!("a tick before the publication must be refused");
        };
        assert!(
            message.contains("precedes this CPU publication"),
            "{message}"
        );
        route.step(11, gravity(), &[], &[]).unwrap();
    }

    #[test]
    fn an_unsupported_creation_is_refused_with_a_message_naming_the_route() {
        let message = unsupported(&PhysicsError::InvalidCollision);
        assert!(
            message.contains("the CPU solver cannot run this creation"),
            "{message}"
        );
        assert!(message.contains("MECHANIC_PHYSICS=gpu"), "{message}");
    }

    #[test]
    fn a_closed_loop_creation_publishes_on_the_cpu_route() {
        use bevy::math::{IVec3, Vec3};
        use mechanic_core::{BearingSpec, BuildOutcome, FaceKind, FaceRef, GridRotation};

        // A parallelogram whose coupler's second bearing closes a loop.
        let mut graph = ConstructionGraph::new();
        let mut spawn = |ticks: IVec3, dimensions: [u8; 3]| {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        dimensions,
                        BuildPose::from_position_ticks(ticks, GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                panic!("spawn expected");
            };
            part
        };
        let height = IVec3::Y * 800;
        let ground = spawn(height, [8, 1, 1]);
        let left = spawn(height + IVec3::new(-350, 250, 100), [1, 6, 1]);
        let right = spawn(height + IVec3::new(350, 250, 100), [1, 6, 1]);
        let coupler = spawn(height + IVec3::new(0, 500, 200), [8, 1, 1]);
        for (source, target, anchor) in [
            (ground, left, Vec3::new(-0.875, 2.0, 0.125)),
            (ground, right, Vec3::new(0.875, 2.0, 0.125)),
            (left, coupler, Vec3::new(-0.875, 3.25, 0.375)),
            (right, coupler, Vec3::new(0.875, 3.25, 0.375)),
        ] {
            graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(source, FaceKind::PositiveZ),
                    FaceRef::part(target, FaceKind::NegativeZ),
                    anchor,
                    Vec3::Z,
                )))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        assert_eq!(creation.dynamics.loops.len(), 1);
        let transforms = creation
            .compounds
            .iter()
            .map(|body| GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let velocities = vec![
            GpuVelocity {
                linear: [0.0; 4],
                angular: [0.0; 4],
            };
            transforms.len()
        ];
        let coordinates = vec![
            GpuMechanismCoordinate {
                position: 0.0,
                velocity: 0.0,
            };
            creation.dynamics.coordinate_velocities.len()
        ];

        let mut route =
            CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &coordinates).unwrap();
        route.publish_terrain([&floor()], DVec3::ZERO).unwrap();
        let mut completed = None;
        for tick in 1..=60 {
            completed = Some(route.step(tick, gravity(), &[], &[]).unwrap());
        }
        let completed = completed.unwrap();
        assert!(
            completed.transforms[0].position[1] < transforms[0].position[1],
            "the linkage should fall"
        );
    }

    #[test]
    fn a_refused_tick_names_the_tick_and_the_way_back_to_the_gpu() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
        let message = route.failure(12, &PhysicsError::InvalidCommand);

        assert!(message.contains("refused tick 12"), "{message}");
        assert!(message.contains("MECHANIC_PHYSICS=gpu"), "{message}");
        assert_eq!(route.degraded_ticks(), 0);
    }
}
