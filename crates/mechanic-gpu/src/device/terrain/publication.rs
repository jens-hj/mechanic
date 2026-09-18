//! Worker-owned packing and stable GPU chunk allocations.
use std::{
    collections::BTreeMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use bevy_math::{DVec3, Vec3};
use bytemuck::{Zeroable, cast_slice};
use mechanic_world::{
    TerrainCollisionChunk, TerrainMeshChunk, TerrainNodeId, TerrainTriangleGroupMask,
};

use super::{GpuTerrainError, TerrainRow, pack_chunks};

#[derive(Clone, Debug, PartialEq)]
struct Identity {
    node: TerrainNodeId,
    generation: u64,
    origin: DVec3,
    groups: TerrainTriangleGroupMask,
}

#[derive(Debug)]
struct PackedChunk {
    identity: Identity,
    rows: Vec<TerrainRow>,
    minimum: Vec3,
    maximum: Vec3,
    triangles: usize,
    content_hash: u64,
}

/// CPU-only cache that can be moved to the terrain preparation worker.
/// A generation must change whenever a node's geometry or materials change.
/// Cloning shares the packed chunks, so a replacement scene inherits them.
#[derive(Clone, Debug, Default)]
pub struct TerrainPreparationCache {
    chunks: Vec<Arc<PackedChunk>>,
}

/// Validated CPU geometry awaiting a caller-owned tick-boundary publication.
#[derive(Debug)]
pub struct PreparedTerrainUpdate {
    /// Revision of the requested terrain cut; callers must reject stale results.
    pub source_revision: u64,
    /// Double-precision coordinate frame in which placement will be published.
    pub physics_origin: DVec3,
    chunks: Vec<Arc<PackedChunk>>,
}

impl PreparedTerrainUpdate {
    /// Same-build identity of the ordered packed geometry, materials, generations,
    /// and placement frame. Cached chunk digests avoid rescanning resident rows.
    pub fn geometry_fingerprint(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.physics_origin
            .to_array()
            .map(f64::to_bits)
            .hash(&mut hash);
        for chunk in &self.chunks {
            chunk.identity.node.hash(&mut hash);
            chunk.identity.generation.hash(&mut hash);
            chunk
                .identity
                .origin
                .to_array()
                .map(f64::to_bits)
                .hash(&mut hash);
            chunk.identity.groups.hash(&mut hash);
            chunk.content_hash.hash(&mut hash);
        }
        hash.finish()
    }
}

/// Cheap main-thread snapshot containing owned geometry only for changed chunks.
#[derive(Debug)]
pub struct TerrainPreparationRequest {
    inputs: Vec<ChunkInput>,
    origin: DVec3,
    revision: u64,
}

#[derive(Debug)]
enum ChunkInput {
    Cached(Arc<PackedChunk>),
    Changed(Box<TerrainCollisionChunk>),
}

impl TerrainPreparationCache {
    /// Snapshots the current mesh cut without cloning unchanged resident geometry.
    /// The expensive packing step is deferred to [`Self::prepare_request`].
    pub fn request_meshes<'a>(
        &self,
        meshes: impl IntoIterator<Item = &'a TerrainMeshChunk>,
        origin: DVec3,
        revision: u64,
    ) -> TerrainPreparationRequest {
        let cached: BTreeMap<_, _> = self
            .chunks
            .iter()
            .map(|chunk| (chunk.identity.node, chunk))
            .collect();
        let inputs = meshes
            .into_iter()
            .map(|mesh| {
                let identity = Identity {
                    node: mesh.node,
                    generation: mesh.generation,
                    origin: mesh.origin.0,
                    groups: TerrainTriangleGroupMask::REGULAR,
                };
                if let Some(chunk) = cached
                    .get(&mesh.node)
                    .filter(|chunk| chunk.identity == identity)
                {
                    ChunkInput::Cached(Arc::clone(chunk))
                } else {
                    ChunkInput::Changed(Box::new(mesh.collision_chunk()))
                }
            })
            .collect();
        TerrainPreparationRequest {
            inputs,
            origin,
            revision,
        }
    }

    /// Completes a mesh snapshot on a worker, retaining the previous cache on error.
    ///
    /// # Errors
    /// Rejects invalid geometry or a non-finite coordinate frame.
    pub fn prepare_request(
        &mut self,
        request: TerrainPreparationRequest,
    ) -> Result<PreparedTerrainUpdate, GpuTerrainError> {
        if !request.origin.is_finite() {
            return Err(GpuTerrainError::InvalidGeometry);
        }
        let mut chunks = Vec::with_capacity(request.inputs.len());
        for input in request.inputs {
            match input {
                ChunkInput::Cached(chunk) => chunks.push(chunk),
                ChunkInput::Changed(chunk) => {
                    let prepared = Self::default().prepare(
                        [chunk.as_ref()],
                        request.origin,
                        request.revision,
                    )?;
                    chunks.extend(prepared.chunks);
                }
            }
        }
        self.chunks.clone_from(&chunks);
        Ok(PreparedTerrainUpdate {
            source_revision: request.revision,
            physics_origin: request.origin,
            chunks,
        })
    }

    /// Packs changed generations and reuses unchanged chunk-local geometry.
    /// Seam selection participates in the cache identity.
    ///
    /// # Errors
    /// Rejects invalid geometry without replacing the accepted cache.
    pub fn prepare<'a>(
        &mut self,
        chunks: impl IntoIterator<Item = &'a TerrainCollisionChunk>,
        physics_origin: DVec3,
        source_revision: u64,
    ) -> Result<PreparedTerrainUpdate, GpuTerrainError> {
        if !physics_origin.is_finite() {
            return Err(GpuTerrainError::InvalidGeometry);
        }
        let mut packed = Vec::new();
        for chunk in chunks {
            let identity = Identity {
                node: chunk.node,
                generation: chunk.generation,
                origin: chunk.origin.0,
                groups: chunk.active_groups,
            };
            if let Some(existing) = self.chunks.iter().find(|item| item.identity == identity) {
                packed.push(Arc::clone(existing));
                continue;
            }
            let rows = pack_chunks([chunk], chunk.origin.0)?;
            if rows.is_empty() {
                continue;
            }
            let minimum = rows.iter().fold(Vec3::splat(f32::INFINITY), |a, row| {
                a.min(Vec3::from_slice(&row.minimum[..3]))
            });
            let maximum = rows.iter().fold(Vec3::splat(f32::NEG_INFINITY), |a, row| {
                a.max(Vec3::from_slice(&row.maximum[..3]))
            });
            let triangles = rows.iter().filter(|row| row.metadata[1] == 1).count();
            let mut hash = DefaultHasher::new();
            cast_slice::<_, u8>(&rows).hash(&mut hash);
            packed.push(Arc::new(PackedChunk {
                identity,
                rows,
                minimum,
                maximum,
                triangles,
                content_hash: hash.finish(),
            }));
        }
        self.chunks.clone_from(&packed);
        Ok(PreparedTerrainUpdate {
            source_revision,
            physics_origin,
            chunks: packed,
        })
    }
}

/// Byte counts for a successful terrain publication, excluding cache clears.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainUploadStats {
    /// Same-build identity of reachable packed rows and their device addresses.
    pub layout_fingerprint: u64,
    /// Bytes sent to the GPU, including placement and top-level BVH rows.
    pub uploaded_bytes: u64,
    /// Bytes copied on-device when growing the reusable geometry buffer.
    pub copied_bytes: u64,
    /// Chunk allocations reused without geometry upload.
    pub reused_chunks: usize,
    /// Changed chunk allocations whose geometry was uploaded.
    pub uploaded_chunks: usize,
}

#[derive(Clone, Debug)]
struct Allocation {
    chunk: Arc<PackedChunk>,
    offset: usize,
    capacity: usize,
}

/// Terrain collision geometry already resident on the device.
///
/// A construction edit rebuilds the whole physics scene, but the terrain under
/// it is unchanged. Moving this residency to the replacement scene keeps the
/// uploaded geometry and its chunk allocations, so the next publication only
/// rebinds them instead of re-uploading every chunk.
#[derive(Debug, Default)]
pub struct TerrainResidency(pub(in crate::device) TerrainGpuScene);

#[derive(Debug, Default)]
pub(in crate::device) struct TerrainGpuScene {
    pub(super) buffer: Option<wgpu::Buffer>,
    capacity: usize,
    allocations: Vec<Allocation>,
}

impl TerrainGpuScene {
    /// Buffer and triangle count of the last accepted publication, if any.
    pub(in crate::device) fn resident(&self) -> Option<(wgpu::Buffer, usize)> {
        let buffer = self.buffer.clone()?;
        Some((
            buffer,
            self.allocations.iter().map(|a| a.chunk.triangles).sum(),
        ))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "validate the complete allocation plan before the first GPU write"
    )]
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTerrainUpdate,
    ) -> Result<(wgpu::Buffer, usize, TerrainUploadStats), GpuTerrainError> {
        let mut remaining = self.allocations.clone();
        let mut allocations = Vec::new();
        let mut changed = Vec::new();
        // Reserve existing chunks first, so replacement allocations cannot evict them.
        for chunk in &prepared.chunks {
            if let Some(index) = remaining.iter().position(|a| Arc::ptr_eq(&a.chunk, chunk)) {
                allocations.push(remaining.remove(index));
            } else {
                changed.push(Arc::clone(chunk));
            }
        }
        let reused = allocations.len();
        let mut end = self
            .allocations
            .iter()
            .map(|a| a.offset + a.capacity)
            .max()
            .unwrap_or(1);
        for chunk in changed {
            let required = chunk.rows.len() + 1;
            let (offset, capacity) =
                if let Some(index) = remaining.iter().position(|a| a.capacity >= required) {
                    let free = remaining.remove(index);
                    (free.offset, free.capacity)
                } else {
                    let offset = end;
                    end = end.checked_add(required).ok_or(GpuTerrainError::Capacity)?;
                    (offset, required)
                };
            allocations.push(Allocation {
                chunk,
                offset,
                capacity,
            });
        }
        // Inactive tail allocations are released. Holes may be reused next publication.
        let top_start = allocations
            .iter()
            .map(|a| a.offset + a.capacity)
            .max()
            .unwrap_or(1);
        let mut leaves = Vec::new();
        for allocation in &allocations {
            let shift = (allocation.chunk.identity.origin - prepared.physics_origin).as_vec3();
            let minimum = allocation.chunk.minimum + shift;
            let maximum = allocation.chunk.maximum + shift;
            if !shift.is_finite() || !minimum.is_finite() || !maximum.is_finite() {
                return Err(GpuTerrainError::InvalidGeometry);
            }
            leaves.push(TerrainRow {
                minimum: minimum.extend(0.0).to_array(),
                maximum: maximum.extend(0.0).to_array(),
                metadata: [
                    0,
                    2,
                    u32::try_from(allocation.offset + 1).map_err(|_| GpuTerrainError::Capacity)?,
                    u32::try_from(allocation.offset + 1 + allocation.chunk.rows.len())
                        .map_err(|_| GpuTerrainError::Capacity)?,
                ],
                ..TerrainRow::zeroed()
            });
        }
        let mut top = Vec::new();
        append_top(&mut leaves, &mut top, top_start)?;
        let required = top_start
            .checked_add(top.len())
            .ok_or(GpuTerrainError::Capacity)?;
        let limit = usize::try_from(device.limits().max_storage_buffer_binding_size)
            .unwrap_or(usize::MAX)
            / size_of::<TerrainRow>();
        if required > limit || required >= 0x2000_0000 {
            return Err(GpuTerrainError::Capacity);
        }
        let grow = self.buffer.is_none() || required > self.capacity;
        let capacity = if grow {
            required.next_power_of_two().min(limit)
        } else {
            self.capacity
        };
        // Validate everything before any queue writes or accepted-state mutation.
        let buffer = if grow {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mechanic incremental terrain BVH"),
                size: (capacity * size_of::<TerrainRow>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        } else {
            self.buffer.as_ref().unwrap().clone()
        };
        let mut layout_hash = DefaultHasher::new();
        prepared.geometry_fingerprint().hash(&mut layout_hash);
        top_start.hash(&mut layout_hash);
        required.hash(&mut layout_hash);
        cast_slice::<_, u8>(&top).hash(&mut layout_hash);
        for allocation in &allocations {
            allocation.offset.hash(&mut layout_hash);
            allocation.chunk.content_hash.hash(&mut layout_hash);
        }
        let mut stats = TerrainUploadStats {
            layout_fingerprint: layout_hash.finish(),
            ..Default::default()
        };
        if grow && let Some(previous) = &self.buffer {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mechanic grow terrain allocation"),
            });
            stats.copied_bytes = (self.capacity * size_of::<TerrainRow>()) as u64;
            encoder.copy_buffer_to_buffer(previous, 0, &buffer, 0, stats.copied_bytes);
            // Submit the old contents before queue writes patch the new buffer.
            queue.submit([encoder.finish()]);
        }
        let mut write = |offset: usize, rows: &[TerrainRow]| {
            if rows.is_empty() {
                return;
            }
            let bytes = cast_slice(rows);
            queue.write_buffer(&buffer, (offset * size_of::<TerrainRow>()) as u64, bytes);
            stats.uploaded_bytes += bytes.len() as u64;
        };
        for (index, allocation) in allocations.iter().enumerate() {
            let placement = TerrainRow {
                first: (allocation.chunk.identity.origin - prepared.physics_origin)
                    .as_vec3()
                    .extend(0.0)
                    .to_array(),
                ..TerrainRow::zeroed()
            };
            write(allocation.offset, &[placement]);
            if index >= reused {
                let mut rows = allocation.chunk.rows.clone();
                for row in &mut rows {
                    row.metadata[0] += u32::try_from(allocation.offset + 1).unwrap();
                    row.metadata[2] = u32::try_from(allocation.offset).unwrap();
                }
                write(allocation.offset + 1, &rows);
            }
        }
        write(top_start, &top);
        write(
            0,
            &[TerrainRow {
                metadata: [
                    u32::try_from(top_start).unwrap(),
                    u32::try_from(required).unwrap(),
                    0,
                    0,
                ],
                ..TerrainRow::zeroed()
            }],
        );
        stats.reused_chunks = reused;
        stats.uploaded_chunks = allocations.len() - stats.reused_chunks;
        let triangles = allocations.iter().map(|a| a.chunk.triangles).sum();
        self.allocations = allocations;
        self.buffer = Some(buffer.clone());
        self.capacity = capacity;
        Ok((buffer, triangles, stats))
    }
}

fn append_top(
    leaves: &mut [TerrainRow],
    rows: &mut Vec<TerrainRow>,
    base: usize,
) -> Result<(), GpuTerrainError> {
    if leaves.is_empty() {
        return Ok(());
    }
    let index = rows.len();
    if leaves.len() == 1 {
        rows.push(leaves[0]);
    } else {
        let minimum = leaves.iter().fold(Vec3::splat(f32::INFINITY), |a, row| {
            a.min(Vec3::from_slice(&row.minimum[..3]))
        });
        let maximum = leaves
            .iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), |a, row| {
                a.max(Vec3::from_slice(&row.maximum[..3]))
            });
        let extent = maximum - minimum;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        leaves.sort_by(|a, b| {
            (a.minimum[axis] + a.maximum[axis]).total_cmp(&(b.minimum[axis] + b.maximum[axis]))
        });
        rows.push(TerrainRow {
            minimum: minimum.extend(0.0).to_array(),
            maximum: maximum.extend(0.0).to_array(),
            ..TerrainRow::zeroed()
        });
        let middle = leaves.len() / 2;
        let (left, right) = leaves.split_at_mut(middle);
        append_top(left, rows, base)?;
        append_top(right, rows, base)?;
    }
    rows[index].metadata[0] =
        u32::try_from(base + rows.len()).map_err(|_| GpuTerrainError::Capacity)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_identity_tracks_packed_content_order_and_origin() {
        let chunk = super::super::tests::chunk();
        let mut changed = chunk.clone();
        changed.generation += 1;
        let fingerprint = |chunks: Vec<&TerrainCollisionChunk>, origin| {
            TerrainPreparationCache::default()
                .prepare(chunks, origin, 1)
                .unwrap()
                .geometry_fingerprint()
        };
        let initial = fingerprint(vec![&chunk], DVec3::ZERO);
        assert_eq!(initial, fingerprint(vec![&chunk], DVec3::ZERO));
        assert_ne!(initial, fingerprint(vec![&chunk], DVec3::X));
        assert_ne!(
            fingerprint(vec![&chunk, &changed], DVec3::ZERO),
            fingerprint(vec![&changed, &chunk], DVec3::ZERO)
        );
        changed.generation = chunk.generation;
        changed.vertices[0][0] += 0.01;
        assert_ne!(initial, fingerprint(vec![&changed], DVec3::ZERO));
    }

    #[test]
    fn preparation_reuses_geometry_across_origin_shifts_and_replaces_changed_generations() {
        let mut cache = TerrainPreparationCache::default();
        let mut chunk = super::super::tests::chunk();
        let first = cache.prepare([&chunk], DVec3::ZERO, 1).unwrap();
        let shifted = cache.prepare([&chunk], DVec3::X * 1.0e10, 2).unwrap();
        assert!(Arc::ptr_eq(&first.chunks[0], &shifted.chunks[0]));
        chunk.generation += 1;
        let changed = cache.prepare([&chunk], DVec3::ZERO, 3).unwrap();
        assert!(!Arc::ptr_eq(&first.chunks[0], &changed.chunks[0]));
        chunk.active_groups = TerrainTriangleGroupMask::default();
        assert!(
            cache
                .prepare([&chunk], DVec3::ZERO, 4)
                .unwrap()
                .chunks
                .is_empty()
        );
    }

    #[test]
    fn mesh_handoff_copies_only_changed_geometry() {
        let chunk = super::super::tests::chunk();
        let mut cache = TerrainPreparationCache::default();
        let initial = cache.prepare([&chunk], DVec3::ZERO, 1).unwrap();
        let mut mesh = TerrainMeshChunk {
            node: chunk.node,
            origin: chunk.origin,
            generation: chunk.generation,
            vertices: chunk.vertices,
            material_weights: chunk.material_weights,
            triangle_bvh: chunk.triangle_bvh,
            ..Default::default()
        };
        let request = cache.request_meshes([&mesh], DVec3::X, 2);
        assert!(matches!(&request.inputs[0], ChunkInput::Cached(_)));
        let prepared = cache.prepare_request(request).unwrap();
        assert!(Arc::ptr_eq(&initial.chunks[0], &prepared.chunks[0]));
        mesh.generation += 1;
        let request = cache.request_meshes([&mesh], DVec3::X, 3);
        assert!(matches!(&request.inputs[0], ChunkInput::Changed(_)));
        let prepared = cache.prepare_request(request).unwrap();
        assert!(!Arc::ptr_eq(&initial.chunks[0], &prepared.chunks[0]));
        mesh.generation += 1;
        mesh.vertices[0][0] = f32::NAN;
        let request = cache.request_meshes([&mesh], DVec3::ZERO, 4);
        assert!(cache.prepare_request(request).is_err());
        assert!(Arc::ptr_eq(&cache.chunks[0], &prepared.chunks[0]));
    }

    #[test]
    fn failed_preparation_preserves_cached_geometry() {
        let mut cache = TerrainPreparationCache::default();
        let chunk = super::super::tests::chunk();
        let first = cache.prepare([&chunk], DVec3::ZERO, 1).unwrap();
        let mut invalid = chunk.clone();
        invalid.generation += 1;
        invalid.vertices[0][0] = f32::NAN;
        assert!(cache.prepare([&invalid], DVec3::ZERO, 2).is_err());
        let next = cache.prepare([&chunk], DVec3::ZERO, 3).unwrap();
        assert!(Arc::ptr_eq(&first.chunks[0], &next.chunks[0]));
    }

    #[test]
    fn top_level_bvh_miss_skips_all_distant_chunks() {
        let mut leaves: Vec<_> = (0_u16..32)
            .map(|x| TerrainRow {
                minimum: [f32::from(x) * 10.0, 0.0, 0.0, 0.0],
                maximum: [f32::from(x) * 10.0 + 1.0, 1.0, 1.0, 0.0],
                metadata: [0, 2, u32::from(x), u32::from(x + 1)],
                ..TerrainRow::zeroed()
            })
            .collect();
        let mut top = Vec::new();
        append_top(&mut leaves, &mut top, 10).unwrap();
        assert_eq!(top.len(), 63);
        assert_eq!(top[0].metadata[0], 73);
        let mut index = 0;
        let mut visited = 0;
        let mut found = Vec::new();
        while index < top.len() {
            visited += 1;
            let row = top[index];
            if row.minimum[0] > 0.5 || row.maximum[0] < 0.5 {
                index = row.metadata[0] as usize - 10;
            } else {
                if row.metadata[1] == 2 {
                    found.push(row.metadata[2]);
                }
                index += 1;
            }
        }
        assert_eq!(found, [0]);
        assert!(visited <= 11, "visited {visited} nodes");
    }

    #[test]
    fn incremental_publication_reuses_chunks_and_rejects_stale_output_before_writes() {
        use crate::{GpuPhysics, GpuPhysicsConfig};
        use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec};
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        eprintln!("Incremental terrain adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(
                        bevy_math::IVec3::new(0, 3, 0),
                        mechanic_core::GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &graph.compile().unwrap(),
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut cache = TerrainPreparationCache::default();
        let chunk = super::super::tests::rigid_chunk();
        let prepared = cache.prepare([&chunk], DVec3::ZERO, 1).unwrap();
        let first = gpu
            .publish_prepared_terrain(&device, &queue, &prepared, 1, DVec3::ZERO)
            .unwrap();
        let reused = gpu
            .publish_prepared_terrain(&device, &queue, &prepared, 1, DVec3::ZERO)
            .unwrap();
        assert_eq!(reused.reused_chunks, 1);
        assert_eq!(reused.uploaded_chunks, 0);
        assert!(reused.uploaded_bytes < first.uploaded_bytes);
        assert!(matches!(
            gpu.publish_prepared_terrain(&device, &queue, &prepared, 2, DVec3::ZERO),
            Err(GpuTerrainError::StalePreparation)
        ));
        assert!(matches!(
            gpu.publish_prepared_terrain(&device, &queue, &prepared, 1, DVec3::X),
            Err(GpuTerrainError::StalePreparation)
        ));
        gpu.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().contact_count, 4);
        let invalid_origin = DVec3::splat(f64::MAX);
        let invalid = cache.prepare([&chunk], invalid_origin, 2).unwrap();
        assert!(matches!(
            gpu.publish_prepared_terrain(&device, &queue, &invalid, 2, invalid_origin),
            Err(GpuTerrainError::InvalidGeometry)
        ));
        gpu.dispatch_tick(&device, &queue, 2);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().contact_count, 4);
        let mut distant = chunk.clone();
        distant.origin.0 += DVec3::X * 50.0;
        let expanded = cache.prepare([&chunk, &distant], DVec3::ZERO, 3).unwrap();
        gpu.publish_prepared_terrain(&device, &queue, &expanded, 3, DVec3::ZERO)
            .unwrap();
        distant.generation += 1;
        let replacement = cache.prepare([&chunk, &distant], DVec3::ZERO, 4).unwrap();
        let stats = gpu
            .publish_prepared_terrain(&device, &queue, &replacement, 4, DVec3::ZERO)
            .unwrap();
        assert_eq!(stats.reused_chunks, 1);
        assert_eq!(stats.uploaded_chunks, 1);
        let shifted = cache
            .prepare([&chunk, &distant], DVec3::X * 10.0, 5)
            .unwrap();
        let stats = gpu
            .publish_prepared_terrain(&device, &queue, &shifted, 5, DVec3::X * 10.0)
            .unwrap();
        assert_eq!(stats.reused_chunks, 2);
        assert_eq!(stats.uploaded_chunks, 0);
        gpu.dispatch_tick(&device, &queue, 3);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().contact_count, 0);
        let empty = cache.prepare([], DVec3::ZERO, 2).unwrap();
        gpu.publish_prepared_terrain(&device, &queue, &empty, 2, DVec3::ZERO)
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 2);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().contact_count, 0);
    }
}
