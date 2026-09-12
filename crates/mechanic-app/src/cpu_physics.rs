//! Optional CPU solver route, selected with `MECHANIC_PHYSICS=cpu`.
//!
//! The GPU runtime stays resident and keeps owning terrain preparation, drive
//! resolution and every buffer the renderer reads; this route only replaces the
//! tick itself, stepping `mechanic_physics::CpuJointMachine` against the same
//! published terrain cut and publishing the same body/joint state the readback
//! would. It exists to run the CPU solver against real creations, not to retire
//! the GPU route: `docs/cpu-solver-repair.md` lists the states that still fail.

use bevy::math::{DQuat, DVec3};
use mechanic_core::{CompiledCreation, CoordinateDrive};
use mechanic_gpu::{
    GpuExternalImpulse, GpuMechanismCoordinate, GpuMechanismDrive, GpuTransform, GpuVelocity,
};
use mechanic_physics::{
    BodyPose, CpuJointMachine, DriveCommand, ExternalImpulse, JointTickSettings, MachineDynamics,
    MachineState, PhysicsError, TerrainContactScene, TerrainIntegration, TerrainSubstep,
};
use mechanic_world::TerrainMeshChunk;
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

/// Policy bounds for one CPU tick. These match the terrain-tick experiments.
const MAXIMUM_DEPTH: f64 = 0.005;
const MAXIMUM_EVALUATIONS: usize = 128;
const MAXIMUM_EVENT_TRIALS: usize = 128;
const RESTITUTION_THRESHOLD: f64 = 1.0;
const STICTION_THRESHOLD: f64 = 1e-7;

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
    machine: CpuJointMachine,
    creation: CompiledCreation,
    geometry: mechanic_physics::MachineCollisionGeometry,
    scene: TerrainContactScene,
    generation: u64,
    /// Terrain publications applied to `scene`, which must keep increasing.
    publication: u64,
    origin: DVec3,
    /// Whether terrain has been published at least once.
    published: bool,
    /// App tick that the machine's own tick zero corresponds to.
    base_tick: u64,
    integration: TerrainIntegration,
    settings: JointTickSettings,
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
        let machine = CpuJointMachine::new(creation.clone(), generation, state)
            .map_err(|error| unsupported(creation, &error))?;
        Ok(Self {
            machine,
            creation: creation.clone(),
            geometry,
            scene: TerrainContactScene::default(),
            generation,
            publication: 0,
            origin: DVec3::ZERO,
            published: false,
            base_tick,
            integration: TerrainIntegration::EventResolved,
            settings: JointTickSettings::default(),
        })
    }

    /// Replaces the collision scene with the accepted terrain cut. Called from the
    /// same place that publishes the cut to the GPU, so both routes see one cut.
    ///
    /// # Errors
    /// Returns a message when the chunk geometry or the frame is invalid.
    pub(crate) fn publish_terrain<'a>(
        &mut self,
        chunks: impl IntoIterator<Item = &'a TerrainMeshChunk>,
        origin: DVec3,
    ) -> Result<(), String> {
        let upserts = chunks
            .into_iter()
            .map(|mesh| Arc::new(mesh.collision_chunk()))
            .collect::<Vec<_>>();
        // A fresh scene keeps publication generations monotonic without tracking
        // which nodes the world dropped between cuts.
        let mut scene = TerrainContactScene::default();
        self.publication += 1;
        scene
            .publish(self.publication, &upserts, &[])
            .map_err(|error| format!("cannot publish terrain to the CPU solver: {error}"))?;
        self.scene = scene;
        self.origin = origin;
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
        let machine_tick = tick
            .checked_sub(self.base_tick)
            .ok_or_else(|| format!("tick {tick} precedes this CPU publication"))?;
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
        let terrain = TerrainSubstep {
            integration: self.integration,
            scene: &self.scene,
            geometry: &self.geometry,
            topology_generation: self.generation,
            origin: self.origin,
            maximum_depth: MAXIMUM_DEPTH,
            maximum_evaluations: MAXIMUM_EVALUATIONS,
            maximum_event_trials: MAXIMUM_EVENT_TRIALS,
            restitution_threshold: RESTITUTION_THRESHOLD,
            stiction_threshold: STICTION_THRESHOLD,
        };
        let outcome =
            self.machine
                .step_with_terrain(gravity, self.settings, &impulses, &commands, &terrain);
        if let Err(error) = outcome {
            return Err(self.failure(tick, &error));
        }
        self.published_state()
    }

    /// What the last attempted tick did, for the pause message.
    fn failure(&self, tick: u64, error: &PhysicsError) -> String {
        let diagnostics = self.machine.diagnostics();
        let stage = diagnostics
            .failure_stage
            .map_or_else(|| "none".to_owned(), |stage| format!("{stage:?}"));
        let attempts = diagnostics
            .attempt_failures
            .iter()
            .map(|failure| {
                format!(
                    "{} substeps {:?} at {:.3} ms",
                    failure.substeps,
                    failure.stage,
                    failure.accepted_seconds * 1000.0
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "the CPU solver could not complete tick {tick}: {error}. \
             Stage {stage}, {} substeps over {} attempts, residual {:e}, \
             contacts {} over {} manifolds, event trials {}, impact holds {}. \
             Attempts: [{attempts}]. \
             This is the experimental route; MECHANIC_PHYSICS=gpu runs the shipping solver.",
            diagnostics.substeps,
            diagnostics.attempts,
            diagnostics.residual,
            diagnostics.surface_points,
            diagnostics.surface_manifolds,
            diagnostics.event_trials,
            diagnostics.terrain_impact_holds,
        )
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
    let mut rates = vec![
        0.0;
        creation
            .dynamics
            .body_velocities
            .last()
            .map_or(0, |rows| rows.end)
    ];
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
                (0.498..=0.503).contains(&height),
                "tick {tick} settled at {height}"
            );
            published += 1;
        }
        assert_eq!(published, 30);
        let resting = route.step(31, gravity(), &[], &[]).unwrap();
        // A settled body must be at rest in velocity too, not merely held in place
        // by position recovery: the endpoint policy holds the pose while gravity
        // keeps accumulating, which would launch the creation when it releases.
        assert!(
            resting.velocities[0].linear[1].abs() < 1e-6,
            "resting vertical velocity {}",
            resting.velocities[0].linear[1]
        );
        assert!((resting.transforms[0].position[1] - 0.5).abs() < 1e-6);
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
    fn a_failed_tick_reports_the_stage_and_the_way_back_to_the_gpu() {
        let (creation, transforms, velocities) = dropped_cube(0.502);
        let route = CpuRoute::new(&creation, 7, 0, &transforms, &velocities, &[]).unwrap();
        let message = route.failure(12, &PhysicsError::NotConverged);

        assert!(message.contains("could not complete tick 12"), "{message}");
        assert!(message.contains("Stage"), "{message}");
        assert!(message.contains("event trials"), "{message}");
        assert!(message.contains("MECHANIC_PHYSICS=gpu"), "{message}");
    }
}
