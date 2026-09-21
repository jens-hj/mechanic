//! Publishes the world's accepted terrain cut before submitting dependent ticks.

use bevy::{
    math::{DVec3, Vec3},
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};
use mechanic_gpu::{PreparedTerrainUpdate, TerrainPreparationCache};
use mechanic_world::{
    TerrainMeshChunk, TerrainNodeId, TerrainTriangleGroupMask, WorldBounds, WorldPosition,
};
use std::time::Instant;

use crate::simulation::state::AppSimulation;
use crate::world::WorldRuntime;

/// Exact terrain generations, seam selection, and frame represented by a GPU scene.
#[derive(Debug, PartialEq)]
pub(crate) struct TerrainPublicationKey {
    origin: DVec3,
    chunks: Vec<(TerrainNodeId, u64, TerrainTriangleGroupMask)>,
}

impl TerrainPublicationKey {
    fn new<'a>(origin: DVec3, chunks: impl Iterator<Item = &'a TerrainMeshChunk>) -> Self {
        Self {
            origin,
            chunks: chunks
                .map(|chunk| {
                    (
                        chunk.node,
                        chunk.generation,
                        TerrainTriangleGroupMask::REGULAR,
                    )
                })
                .collect(),
        }
    }

    /// Cheap identity for detecting a cut that stopped changing.
    ///
    /// Comparing fingerprints avoids retaining a second copy of the chunk list
    /// purely to notice that this frame requested the same cut as the last.
    fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.origin.to_array().map(f64::to_bits).hash(&mut hasher);
        for (node, generation, mask) in &self.chunks {
            node.hash(&mut hasher);
            generation.hash(&mut hasher);
            mask.hash(&mut hasher);
        }
        hasher.finish()
    }
}

struct Pending {
    key: TerrainPublicationKey,
    started: Instant,
    snapshot_ms: f64,
    task: Task<(
        TerrainPreparationCache,
        Result<PreparedTerrainUpdate, String>,
        f64,
    )>,
}

#[derive(Default)]
pub(crate) struct TerrainPublication {
    accepted: Option<TerrainPublicationKey>,
    cache: TerrainPreparationCache,
    pending: Option<Pending>,
    revision: u64,
    /// Fingerprint of the cut requested on the previous frame.
    ///
    /// While terrain streams, the desired cut gains chunks every frame, so a
    /// preparation started immediately is always stale on arrival. Waiting for
    /// one unchanged frame turns that livelock into one preparation per settled
    /// cut.
    observed: Option<u64>,
    /// Body poses the accepted cut was selected around.
    accepted_positions: Vec<Vec3>,
    geometry_fingerprint: Option<u64>,
    layout_fingerprint: Option<u64>,
    /// Floating origin at which local terrain streaming has finished at least
    /// once. Ticks never run at an origin that has not settled.
    settled_origin: Option<DVec3>,
}

impl TerrainPublication {
    pub(crate) const fn geometry_fingerprint(&self) -> Option<u64> {
        self.geometry_fingerprint
    }
    pub(crate) const fn layout_fingerprint(&self) -> Option<u64> {
        self.layout_fingerprint
    }
    /// Inherits the terrain cut of the scene this publication replaces.
    ///
    /// `resident` states whether the replacement scene adopted and rebound the
    /// previous device geometry. Only then is the accepted cut still published,
    /// which is what lets a construction edit skip terrain work entirely; a
    /// scene without that geometry keeps the packed chunks and republishes.
    pub(crate) fn inherit(&mut self, previous: &mut Self, resident: bool) {
        self.cache = std::mem::take(&mut previous.cache);
        self.revision = previous.revision;
        self.settled_origin = previous.settled_origin;
        if resident {
            self.accepted = previous.accepted.take();
            self.geometry_fingerprint = previous.geometry_fingerprint.take();
            self.layout_fingerprint = previous.layout_fingerprint.take();
        }
    }
}

/// Half-extent, in metres, of the terrain each simulated body publishes around
/// itself. Sized so a body has to travel a long way before the cut it stands on
/// stops covering it, not so a contact can reach that far.
const INTEREST_MARGIN_METRES: f64 = 96.0;

/// How far a body may travel from the pose its cut was accepted at before ticks
/// wait for a wider cut. The remaining margin covers body extent and contact
/// reach, so the published terrain always extends well past every collider.
const INTEREST_TRAVEL_METRES: f32 = 48.0;

/// Interest boxes snap to this lattice so ordinary motion does not reselect
/// chunks, and therefore does not republish, on almost every frame.
const INTEREST_SNAP_METRES: f64 = 32.0;

/// Local-space positions of the bodies the terrain cut must cover.
///
/// Readback poses lag the scene by a tick and are empty before the first one, so
/// a freshly compiled scene falls back to its compiled roots. Publishing an
/// empty cut would drop every body through the world.
fn physics_body_positions(simulation: &AppSimulation) -> Vec<Vec3> {
    let compounds = simulation
        .creation
        .as_ref()
        .map_or(0, |creation| creation.compounds.len());
    if compounds > 0 && simulation.transforms.len() == compounds {
        return simulation
            .transforms
            .iter()
            .map(|transform| Vec3::from_slice(&transform.position[..3]))
            .collect();
    }
    simulation
        .creation
        .as_ref()
        .map_or_else(Vec::new, |creation| {
            creation
                .compounds
                .iter()
                .map(|compound| compound.root_translation)
                .collect()
        })
}

/// One lattice-snapped global box per body, in the order of `positions`.
fn interest_regions(world: &WorldRuntime, positions: &[Vec3]) -> Vec<WorldBounds> {
    positions
        .iter()
        .map(|position| {
            let centre = world.local_to_global(*position).0;
            let snapped = (centre / INTEREST_SNAP_METRES).round() * INTEREST_SNAP_METRES;
            let margin = DVec3::splat(INTEREST_MARGIN_METRES);
            WorldBounds {
                minimum: WorldPosition(snapped - margin),
                maximum: WorldPosition(snapped + margin),
            }
        })
        .collect()
}

/// Records why dependent ticks are not running against the current cut.
fn record_gate(reason: &'static str, accepted: bool, chunks: usize, waiting_ms: Option<f64>) {
    crate::performance_capture::record("terrain_gate", || {
        serde_json::json!({
            "reason": reason,
            "accepted": accepted,
            "chunks": chunks,
            "waiting_ms": waiting_ms,
        })
    });
}

/// Starts one worker preparation of the requested cut.
fn begin_preparation(
    publication: &mut TerrainPublication,
    world: &WorldRuntime,
    interest: &[WorldBounds],
    origin: DVec3,
    key: TerrainPublicationKey,
) {
    let started = Instant::now();
    let mut cache = std::mem::take(&mut publication.cache);
    publication.revision += 1;
    let revision = publication.revision;
    let request = cache.request_meshes(world.physics_terrain_near(interest), origin, revision);
    let snapshot_ms = started.elapsed().as_secs_f64() * 1000.0;
    publication.pending = Some(Pending {
        key,
        started,
        snapshot_ms,
        task: AsyncComputeTaskPool::get().spawn(async move {
            let started = Instant::now();
            let result = cache
                .prepare_request(request)
                .map_err(|error| error.to_string());
            (cache, result, started.elapsed().as_secs_f64() * 1000.0)
        }),
    });
}

/// Whether an accepted cut can carry ticks while a replacement prepares.
///
/// Bodies are stored relative to the floating origin, so a rebase moves them
/// without moving published terrain: a cut from another origin is unusable and
/// must block. A cut that merely predates newly streamed chunks is exactly as
/// correct as it was on the frame it was accepted, and every frame between
/// publications already runs on it.
fn accepted_cut_may_tick(
    accepted: Option<&TerrainPublicationKey>,
    accepted_positions: &[Vec3],
    origin: DVec3,
    positions: &[Vec3],
) -> bool {
    accepted.is_some_and(|accepted| accepted.origin == origin)
        && accepted_positions.len() == positions.len()
        && accepted_positions
            .iter()
            .zip(positions)
            .all(|(accepted, current)| {
                accepted.distance_squared(*current)
                    <= INTEREST_TRAVEL_METRES * INTEREST_TRAVEL_METRES
            })
}

/// Returns true when dependent ticks may run against the published terrain.
///
/// Ticking waits for local streaming to finish once per floating origin, which
/// covers world entry. After that, terrain streaming ahead of a moving body never
/// holds ticks: the critical region follows the focus, so waiting on it froze
/// physics for as long as a vehicle kept driving into new terrain.
pub(crate) fn publish(
    simulation: &mut AppSimulation,
    world: &WorldRuntime,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<bool, String> {
    let origin = world.local_to_global(Vec3::ZERO).0;
    let positions = physics_body_positions(simulation);
    let interest = interest_regions(world, &positions);
    let key = TerrainPublicationKey::new(origin, world.physics_terrain_near(&interest));
    let publication = &mut simulation.terrain_publication;
    if world.physics_terrain_ready() {
        publication.settled_origin = Some(origin);
    }
    let settled = publication.settled_origin == Some(origin);
    let current = publication.accepted.as_ref() == Some(&key);
    let gpu_may_tick = if simulation.gpu.is_some() {
        publish_gpu(simulation, world, device, queue, &interest, key, settled)?
    } else {
        simulation.terrain_publication.accepted = Some(key);
        simulation.terrain_publication.accepted_positions = positions;
        false
    };
    let Some(cpu) = simulation.cpu.as_mut() else {
        return Ok(gpu_may_tick);
    };
    // The CPU scene follows the chunks around the bodies every frame, so it never
    // waits for a GPU preparation, which only matters if the GPU takes over.
    if settled || current {
        cpu.publish_terrain(world.physics_terrain_near(&interest), origin)?;
    }
    Ok(settled && cpu.is_ready())
}

/// Publishes the cut to the GPU scene and returns whether GPU ticks may run.
fn publish_gpu(
    simulation: &mut AppSimulation,
    world: &WorldRuntime,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    interest: &[WorldBounds],
    key: TerrainPublicationKey,
    settled: bool,
) -> Result<bool, String> {
    let origin = key.origin;
    let positions = physics_body_positions(simulation);
    let publication = &mut simulation.terrain_publication;
    if publication.accepted.as_ref() == Some(&key) {
        publication.accepted_positions = positions;
        publication.observed = None;
        if !settled {
            record_gate("streaming_pending", true, key.chunks.len(), None);
        }
        return Ok(settled);
    }
    let may_tick = |publication: &TerrainPublication| {
        settled
            && accepted_cut_may_tick(
                publication.accepted.as_ref(),
                &publication.accepted_positions,
                origin,
                &positions,
            )
    };
    if let Some(pending) = &mut publication.pending {
        let Some((cache, result, preparation_ms)) = check_ready(&mut pending.task) else {
            record_gate(
                "preparation_pending",
                publication.accepted.is_some(),
                key.chunks.len(),
                Some(pending.started.elapsed().as_secs_f64() * 1000.0),
            );
            return Ok(may_tick(publication));
        };
        let pending = publication.pending.take().unwrap();
        publication.cache = cache;
        if pending.key == key {
            let prepared = result?;
            let started = Instant::now();
            let stats = simulation
                .gpu
                .as_mut()
                .ok_or("terrain publication requires GPU physics")?
                .publish_prepared_terrain(device, queue, &prepared, publication.revision, origin)
                .map_err(|error| format!("cannot publish terrain collision: {error}"))?;
            publication.geometry_fingerprint = Some(prepared.geometry_fingerprint());
            publication.layout_fingerprint = Some(stats.layout_fingerprint);
            crate::performance_capture::record("terrain_publication", || {
                serde_json::json!({
                    "source_revision": publication.revision,
                    "geometry_fingerprint": publication.geometry_fingerprint.map(|hash| format!("{hash:016x}")),
                    "layout_fingerprint": publication.layout_fingerprint.map(|hash| format!("{hash:016x}")),
                    "preparation_ms": preparation_ms + pending.snapshot_ms,
                "snapshot_ms": pending.snapshot_ms,
                "worker_preparation_ms": preparation_ms,
                    "publication_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "latency_ms": pending.started.elapsed().as_secs_f64() * 1000.0,
                    "uploaded_bytes": stats.uploaded_bytes,
                "copied_bytes": stats.copied_bytes,
                    "reused_chunks": stats.reused_chunks,
                    "uploaded_chunks": stats.uploaded_chunks,
                    "cut_chunks": pending.key.chunks.len(),
                    "interest_regions": interest.len(),
                })
            });
            publication.accepted = Some(key);
            publication.accepted_positions = positions;
            return Ok(settled);
        }
        // A stale result never writes the GPU, consumes an impulse, or advances a tick.
        record_gate(
            "stale_result",
            publication.accepted.is_some(),
            key.chunks.len(),
            None,
        );
    } else {
        record_gate(
            "key_changed",
            publication.accepted.is_some(),
            key.chunks.len(),
            None,
        );
    }
    // One unchanged frame before preparing. Without it a streaming cut spawns a
    // preparation every frame and discards every result, and each attempt pays a
    // full main-thread chunk snapshot.
    let fingerprint = key.fingerprint();
    if publication.observed.replace(fingerprint) != Some(fingerprint) {
        return Ok(may_tick(publication));
    }
    begin_preparation(publication, world, interest, origin, key);
    Ok(may_tick(publication))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_inherited_cut_initializes_cpu_terrain_after_a_body_split() {
        use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};
        use mechanic_gpu::{GpuTransform, GpuVelocity};

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("real GPU adapter required");
        eprintln!("Terrain publication adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let world = <WorldRuntime as bevy::prelude::FromWorld>::from_world(
            &mut bevy::prelude::World::new(),
        );
        let origin = world.local_to_global(Vec3::ZERO).0;
        let mut graph = ConstructionGraph::new();
        for height in [0, 800] {
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::from_position_ticks(
                            bevy::math::IVec3::Y * height,
                            GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let creation = graph.compile().unwrap();
        assert_eq!(creation.compounds.len(), 2);
        let transforms: Vec<_> = creation
            .compounds
            .iter()
            .map(|body| GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect();
        let cpu = crate::cpu_physics::CpuRoute::new(
            &creation,
            2,
            0,
            &transforms,
            &[GpuVelocity {
                linear: [0.0; 4],
                angular: [0.0; 4],
            }; 2],
            &[],
        )
        .unwrap();
        let mut simulation = AppSimulation {
            cpu: Some(Box::new(cpu)),
            creation: Some(creation),
            transforms,
            ..Default::default()
        };
        let positions = physics_body_positions(&simulation);
        let interest = interest_regions(&world, &positions);
        let mut retired = TerrainPublication {
            accepted: Some(TerrainPublicationKey::new(
                origin,
                world.physics_terrain_near(&interest),
            )),
            accepted_positions: vec![Vec3::ZERO],
            ..Default::default()
        };
        simulation.terrain_publication.inherit(&mut retired, true);
        assert!(!simulation.cpu.as_ref().unwrap().is_ready());

        let may_tick = publish(&mut simulation, &world, &device, &queue).unwrap();
        assert!(simulation.cpu.as_ref().unwrap().is_ready());
        assert_eq!(may_tick, world.physics_terrain_ready());
        assert!(accepted_cut_may_tick(
            simulation.terrain_publication.accepted.as_ref(),
            &simulation.terrain_publication.accepted_positions,
            origin,
            &positions,
        ));
        // The replacement can advance both detached bodies, even though its
        // device cut was inherited and required no GPU publication.
        let completed = simulation
            .cpu
            .as_mut()
            .unwrap()
            .step(1, mechanic_core::GRAVITY, &[], &[])
            .unwrap();
        for (before, after) in simulation.transforms.iter().zip(&completed.transforms) {
            assert!(after.position[1] < before.position[1]);
        }

        // Driving into terrain that is still streaming must not freeze physics:
        // once streaming finished at this origin, later incomplete streaming
        // around a moved focus keeps the CPU route ticking.
        assert!(!world.physics_terrain_ready());
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
        simulation.terrain_publication.settled_origin = Some(origin);
        simulation.terrain_publication.accepted = None;
        assert!(publish(&mut simulation, &world, &device, &queue).unwrap());

        // A floating-origin rebase still waits for streaming at the new origin.
        simulation.terrain_publication.settled_origin = Some(origin + DVec3::X);
        assert!(!publish(&mut simulation, &world, &device, &queue).unwrap());
    }

    #[test]
    fn ticks_continue_on_an_accepted_cut_but_never_before_one_or_across_a_rebase() {
        let cut =
            |origin| TerrainPublicationKey::new(origin, [&TerrainMeshChunk::default()].into_iter());
        let here = [Vec3::ZERO];

        // Nothing published yet: bodies would fall through the world.
        assert!(!accepted_cut_may_tick(None, &here, DVec3::ZERO, &here));

        // A cut that only predates newly streamed chunks keeps ticking.
        assert!(accepted_cut_may_tick(
            Some(&cut(DVec3::ZERO)),
            &here,
            DVec3::ZERO,
            &here
        ));

        // A rebase moves bodies without moving published terrain.
        assert!(!accepted_cut_may_tick(
            Some(&cut(DVec3::ZERO)),
            &here,
            DVec3::X,
            &here
        ));
    }

    #[test]
    fn a_body_leaving_the_cut_it_was_published_for_waits_for_a_wider_one() {
        let cut =
            TerrainPublicationKey::new(DVec3::ZERO, [&TerrainMeshChunk::default()].into_iter());
        let accepted = [Vec3::ZERO];
        let inside = [Vec3::X * (INTEREST_TRAVEL_METRES - 1.0)];
        let outside = [Vec3::X * (INTEREST_TRAVEL_METRES + 1.0)];

        assert!(accepted_cut_may_tick(
            Some(&cut),
            &accepted,
            DVec3::ZERO,
            &inside
        ));
        assert!(!accepted_cut_may_tick(
            Some(&cut),
            &accepted,
            DVec3::ZERO,
            &outside
        ));

        // A republished scene with a different body count has no correspondence.
        assert!(!accepted_cut_may_tick(
            Some(&cut),
            &accepted,
            DVec3::ZERO,
            &[Vec3::ZERO, Vec3::ZERO]
        ));
    }

    #[test]
    fn fingerprints_separate_cuts_that_gained_a_chunk_or_shifted_origin() {
        let chunk = |generation| TerrainMeshChunk {
            generation,
            ..Default::default()
        };
        let one = chunk(1);
        let key = TerrainPublicationKey::new(DVec3::ZERO, [&one].into_iter());
        assert_eq!(
            key.fingerprint(),
            TerrainPublicationKey::new(DVec3::ZERO, [&one].into_iter()).fingerprint(),
        );
        assert_ne!(
            key.fingerprint(),
            TerrainPublicationKey::new(DVec3::X, [&one].into_iter()).fingerprint(),
        );
        let two = chunk(2);
        assert_ne!(
            key.fingerprint(),
            TerrainPublicationKey::new(DVec3::ZERO, [&one, &two].into_iter()).fingerprint(),
        );
    }

    #[test]
    fn inheriting_keeps_the_accepted_cut_only_when_the_replacement_owns_the_geometry() {
        let key =
            || TerrainPublicationKey::new(DVec3::ZERO, [&TerrainMeshChunk::default()].into_iter());
        let previous = || TerrainPublication {
            accepted: Some(key()),
            revision: 7,
            ..TerrainPublication::default()
        };

        // A replacement scene that adopted the device geometry is already
        // publishing the accepted cut, so it needs no terrain work at all.
        let mut retired = previous();
        let mut resident = TerrainPublication::default();
        resident.inherit(&mut retired, true);
        assert_eq!(resident.accepted, Some(key()));
        assert_eq!(resident.revision, 7);
        assert!(retired.accepted.is_none(), "one scene owns the cut");

        // Without that geometry the same cut must be published again.
        let mut retired = previous();
        let mut fresh = TerrainPublication::default();
        fresh.inherit(&mut retired, false);
        assert_eq!(fresh.accepted, None);
        assert_eq!(fresh.revision, 7, "revisions never repeat across scenes");
    }

    #[test]
    fn terrain_publication_key_changes_for_remesh_retirement_and_origin_shift() {
        let mut chunk = TerrainMeshChunk {
            generation: 1,
            ..Default::default()
        };
        let initial = TerrainPublicationKey::new(DVec3::ZERO, [&chunk].into_iter());
        assert_eq!(
            initial,
            TerrainPublicationKey::new(DVec3::ZERO, [&chunk].into_iter())
        );
        assert_ne!(
            initial,
            TerrainPublicationKey::new(DVec3::X, [&chunk].into_iter())
        );
        chunk.generation += 1;
        assert_ne!(
            initial,
            TerrainPublicationKey::new(DVec3::ZERO, [&chunk].into_iter())
        );
        assert_ne!(
            initial,
            TerrainPublicationKey::new(DVec3::ZERO, [].into_iter())
        );
    }
}
