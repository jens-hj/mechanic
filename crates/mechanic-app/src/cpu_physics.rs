//! Optional CPU solver route, selected with `MECHANIC_PHYSICS=cpu`.
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
    BodyPose, CpuMachine, DriveCommand, ExternalImpulse, MachineDynamics, MachineState,
    PhysicsError, SoftStepSettings, SoftStepTerrain, TerrainContactScene,
};
use mechanic_world::{TerrainMeshChunk, TerrainNodeId};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// Which solver advances published ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Route {
    /// The shipping GPU runtime.
    #[default]
    Gpu,
    /// The experimental CPU solver.
    Cpu,
}

/// Reads the route from an explicit setting, defaulting to the GPU runtime.
pub(crate) fn route_from(value: Option<&str>) -> Route {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("cpu") => Route::Cpu,
        _ => Route::Gpu,
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

impl CpuRoute {
    /// Binds the CPU solver to a published creation, starting from the live body
    /// and joint state the GPU route would have uploaded.
    ///
    /// # Errors
    /// Returns a message naming what the CPU solver cannot run, so the caller can
    /// refuse the publication instead of silently falling back to the GPU.
    pub(crate) fn new(
        creation: &CompiledCreation,
        generation: u64,
        base_tick: u64,
        transforms: &[GpuTransform],
        velocities: &[GpuVelocity],
        coordinates: &[GpuMechanismCoordinate],
    ) -> Result<Self, String> {
        let state = machine_state(creation, transforms, velocities, coordinates)?;
        let geometry = mechanic_physics::MachineCollisionGeometry::new(creation, generation)
            .map_err(|error| unsupported(creation, &error))?;
        let machine = CpuMachine::new(creation.clone(), generation, state)
            .map_err(|error| unsupported(creation, &error))?;
        Ok(Self {
            machine,
            creation: creation.clone(),
            geometry,
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
        let chunks = chunks.into_iter().collect::<Vec<_>>();
        let current = chunks
            .iter()
            .map(|mesh| (mesh.node, mesh.generation))
            .collect::<BTreeMap<_, _>>();
        self.origin = origin;
        if self.published && current == self.chunks {
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
        self.published_state()
    }

    /// Why a tick was refused. The soft-step solver only refuses invalid input.
    fn failure(&self, tick: u64, error: &PhysicsError) -> String {
        format!(
            "the CPU solver refused tick {tick}: {error} ({} earlier ticks degraded). \
             This is the experimental route; MECHANIC_PHYSICS=gpu runs the shipping solver.",
            self.degraded_ticks
        )
    }

    /// Ticks since this route was built that fell back after numerical trouble.
    pub(crate) fn degraded_ticks(&self) -> u64 {
        self.degraded_ticks
    }

    /// Converts the committed snapshot into the publication the renderer reads.
    fn published_state(&self) -> Result<Completed, String> {
        let state = &self.machine.snapshot().state;
        let model = MachineDynamics::assemble(&self.creation, &state.poses, &state.coordinates)
            .map_err(|error| format!("cannot reconstruct CPU body poses: {error}"))?;
        let motions = model
            .body_motions(&state.velocities)
            .map_err(|error| format!("cannot reconstruct CPU body velocities: {error}"))?;
        let transforms = state
            .poses
            .iter()
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
            transforms,
            velocities,
            coordinates,
        })
    }
}

/// Names what the CPU solver cannot run, in the user's terms.
fn unsupported(creation: &CompiledCreation, error: &PhysicsError) -> String {
    let detail = match error {
        PhysicsError::UnsupportedJointLoops => format!(
            "it closes {} mechanism loop(s). The CPU solver runs tree mechanisms only",
            creation.dynamics.loops.len()
        ),
        other => format!("the CPU solver rejected it: {other}"),
    };
    format!(
        "MECHANIC_PHYSICS=cpu cannot run this creation because {detail}. \
         Unset the variable, or set MECHANIC_PHYSICS=gpu, to run the shipping solver."
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
    let poses = transforms
        .iter()
        .map(|transform| BodyPose {
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
        })
        .collect();
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

/// Gravity the GPU runtime applies, in metres per second squared.
pub(crate) fn gravity() -> DVec3 {
    -DVec3::Y * 9.81
}

/// The publication format is `f32` by design; the solver's own state stays `f64`.
#[allow(clippy::cast_possible_truncation)]
fn narrow(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
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
    fn only_an_explicit_cpu_setting_leaves_the_gpu_route() {
        assert_eq!(route_from(None), Route::Gpu);
        assert_eq!(route_from(Some("")), Route::Gpu);
        assert_eq!(route_from(Some("gpu")), Route::Gpu);
        assert_eq!(route_from(Some("nonsense")), Route::Gpu);
        assert_eq!(route_from(Some("cpu")), Route::Cpu);
        assert_eq!(route_from(Some(" CPU ")), Route::Cpu);
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
        let creation = cube();
        let loops = unsupported(&creation, &PhysicsError::UnsupportedJointLoops);
        assert!(loops.contains("MECHANIC_PHYSICS=cpu cannot run"), "{loops}");
        assert!(loops.contains("tree mechanisms only"), "{loops}");
        assert!(loops.contains("MECHANIC_PHYSICS=gpu"), "{loops}");

        // Anything else still names the route and the way back to the GPU.
        let other = unsupported(&creation, &PhysicsError::InvalidCollision);
        assert!(other.contains("the CPU solver rejected it"), "{other}");
        assert!(other.contains("MECHANIC_PHYSICS=gpu"), "{other}");
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
