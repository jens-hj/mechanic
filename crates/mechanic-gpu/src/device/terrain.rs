//! Tick-boundary uploads for the terrain contact kernel.

#[cfg(test)]
mod generated;
mod publication;
mod recovery;
pub(super) use publication::TerrainGpuScene;
pub use publication::{
    PreparedTerrainUpdate, TerrainPreparationCache, TerrainPreparationRequest, TerrainResidency,
    TerrainUploadStats,
};
#[cfg(test)]
mod rotational;
mod sweep;
pub(super) use recovery::PositionRecovery;

use bevy_math::{DVec3, Vec3};
use bytemuck::{Pod, Zeroable};
use mechanic_world::TerrainCollisionChunk;
#[cfg(test)]
use mechanic_world::TerrainMaterial;

use super::{GpuPhysics, bind_group, entry};

/// Stackless preorder BVH row. Internal rows skip to `metadata.x` on a miss;
/// triangle rows have `metadata.y == 1`. All coordinates are physics-local.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct TerrainRow {
    minimum: [f32; 4],
    maximum: [f32; 4],
    first: [f32; 4],
    second: [f32; 4],
    third: [f32; 4],
    response: [f32; 4],
    metadata: [u32; 4],
}

/// A terrain publication could not be represented by the GPU buffers.
#[derive(Debug, thiserror::Error)]
pub enum GpuTerrainError {
    /// Prepared geometry belongs to an obsolete source revision or origin.
    #[error("terrain preparation is stale")]
    StalePreparation,
    /// The complete replacement exceeds the device storage-buffer limit.
    #[error("terrain collision upload exceeds device capacity")]
    Capacity,
    /// Geometry contains invalid indices or non-finite coordinates.
    #[error("terrain collision upload contains invalid geometry")]
    InvalidGeometry,
}

impl GpuPhysics {
    /// Replaces the complete terrain contact scene at a caller-owned tick boundary.
    /// Coordinates are rebased in double precision before upload. On failure the
    /// previous scene remains active. Replacement invalidates contact warm starts.
    ///
    /// # Errors
    /// Returns an error for invalid geometry or a scene exceeding device limits.
    ///
    /// # Panics
    /// Panics if device validation rejects a terrain pipeline or binding layout.
    pub fn write_terrain_chunks<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        chunks: impl IntoIterator<Item = &'a TerrainCollisionChunk>,
        physics_origin: DVec3,
    ) -> Result<(), GpuTerrainError> {
        let prepared = TerrainPreparationCache::default().prepare(chunks, physics_origin, 0)?;
        self.publish_prepared_terrain(device, queue, &prepared, 0, physics_origin)?;
        Ok(())
    }

    /// Detaches terrain geometry already resident on the device.
    ///
    /// The next publication of this scene re-uploads everything, so only call
    /// this on a scene that is being replaced.
    pub fn take_terrain_residency(&mut self) -> TerrainResidency {
        TerrainResidency(std::mem::take(&mut self.terrain_scene))
    }

    /// Adopts terrain geometry uploaded by the scene this one replaces.
    ///
    /// Rebinding the resident buffer makes the same terrain cut collidable
    /// immediately, so a construction edit needs no terrain publication at all.
    ///
    /// # Panics
    /// Panics if device validation rejects the terrain pipeline bindings.
    pub fn adopt_terrain_residency(&mut self, device: &wgpu::Device, residency: TerrainResidency) {
        self.terrain_scene = residency.0;
        self.collision.terrain_bind_group = None;
        self.collision.terrain_geometry_bind_group = None;
        self.collision.terrain_triangle_count = 0;
        let Some((buffer, triangles)) = self.terrain_scene.resident() else {
            return;
        };
        self.bind_terrain(device, &buffer, triangles);
    }

    /// Binds resident terrain geometry to the contact and recovery pipelines.
    fn bind_terrain(&mut self, device: &wgpu::Device, buffer: &wgpu::Buffer, triangles: usize) {
        if self.terrain_recovery.is_none() {
            self.terrain_recovery = Some(PositionRecovery::new(self, device));
        }
        let bindings = bind_group(
            device,
            "mechanic terrain contacts",
            &self.collision.terrain_pipeline,
            &[
                entry(0, &self.config),
                entry(1, &self.positions),
                entry(2, &self.rotations),
                entry(5, &self.diagnostics),
                entry(6, &self.colliders),
                entry(10, &self.collision.contacts),
                entry(28, &self.convex_shapes),
                entry(31, buffer),
                entry(
                    32,
                    &self
                        .terrain_recovery
                        .as_ref()
                        .expect("terrain recovery initialized")
                        .previous_poses,
                ),
            ],
        );
        let geometry_bindings = bind_group(
            device,
            "mechanic terrain correction geometry bindings",
            &self
                .terrain_recovery
                .as_ref()
                .expect("terrain recovery initialized")
                .geometry,
            &[
                entry(0, &self.config),
                entry(1, &self.positions),
                entry(2, &self.rotations),
                entry(5, &self.diagnostics),
                entry(6, &self.colliders),
                entry(10, &self.collision.contacts),
                entry(15, &self.collision.persistent_manifolds),
                entry(28, &self.convex_shapes),
                entry(31, buffer),
            ],
        );
        let sweep_bindings = self
            .terrain_recovery
            .as_ref()
            .expect("terrain recovery initialized")
            .sweep
            .bind(self, device, buffer);
        self.terrain_recovery
            .as_mut()
            .expect("terrain recovery initialized")
            .sweep
            .bindings = Some(sweep_bindings);
        self.collision.terrain_bind_group = Some(bindings);
        self.collision.terrain_geometry_bind_group = Some(geometry_bindings);
        self.collision.terrain_triangle_count = triangles;
    }

    /// Publishes validated worker output before dependent ticks. Stale output is
    /// rejected before writing any buffers; unchanged chunk geometry is reused.
    ///
    /// # Errors
    /// Returns an error for stale output or a scene exceeding device limits.
    ///
    /// # Panics
    /// Panics if device validation rejects the terrain pipeline bindings.
    pub fn publish_prepared_terrain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTerrainUpdate,
        expected_revision: u64,
        physics_origin: DVec3,
    ) -> Result<TerrainUploadStats, GpuTerrainError> {
        if prepared.source_revision != expected_revision
            || prepared.physics_origin != physics_origin
        {
            return Err(GpuTerrainError::StalePreparation);
        }
        let (buffer, triangles, stats) = self.terrain_scene.upload(device, queue, prepared)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mechanic invalidate terrain contact generations"),
        });
        encoder.clear_buffer(&self.collision.manifold_keys, 0, None);
        encoder.clear_buffer(&self.collision.persistent_manifolds, 0, None);
        queue.submit([encoder.finish()]);
        self.bind_terrain(device, &buffer, triangles);
        Ok(stats)
    }
}

fn pack_chunks<'a>(
    chunks: impl IntoIterator<Item = &'a TerrainCollisionChunk>,
    physics_origin: DVec3,
) -> Result<Vec<TerrainRow>, GpuTerrainError> {
    if !physics_origin.is_finite() {
        return Err(GpuTerrainError::InvalidGeometry);
    }
    let mut rows = Vec::new();
    for chunk in chunks {
        // Preserve the chunk's existing BVH hierarchy with explicit escape rows.
        let mut work = vec![(0_usize, false)];
        let mut parents = Vec::new();
        if chunk.triangle_bvh.nodes.is_empty() {
            for triangle in &chunk.triangle_bvh.triangles {
                append_triangle(&mut rows, chunk, triangle, physics_origin)?;
            }
            continue;
        }
        while let Some((index, close)) = work.pop() {
            if close {
                let row: usize = parents.pop().ok_or(GpuTerrainError::InvalidGeometry)?;
                rows[row].metadata[0] =
                    u32::try_from(rows.len()).map_err(|_| GpuTerrainError::Capacity)?;
                continue;
            }
            let node = chunk
                .triangle_bvh
                .nodes
                .get(index)
                .ok_or(GpuTerrainError::InvalidGeometry)?;
            if !node.group_mask.intersects(chunk.active_groups) {
                continue;
            }
            let minimum = (node.bounds.minimum.0 - physics_origin).as_vec3();
            let maximum = (node.bounds.maximum.0 - physics_origin).as_vec3();
            if !minimum.is_finite() || !maximum.is_finite() {
                return Err(GpuTerrainError::InvalidGeometry);
            }
            parents.push(rows.len());
            rows.push(TerrainRow {
                minimum: minimum.extend(0.0).to_array(),
                maximum: maximum.extend(0.0).to_array(),
                ..TerrainRow::zeroed()
            });
            work.push((index, true));
            if node.triangle_count > 0 {
                let first = node.first_triangle as usize;
                let end = first
                    .checked_add(node.triangle_count as usize)
                    .ok_or(GpuTerrainError::InvalidGeometry)?;
                for triangle in chunk
                    .triangle_bvh
                    .triangles
                    .get(first..end)
                    .ok_or(GpuTerrainError::InvalidGeometry)?
                {
                    append_triangle(&mut rows, chunk, triangle, physics_origin)?;
                }
            } else {
                for child in [node.right_child, node.left_child].into_iter().flatten() {
                    if child as usize <= index {
                        return Err(GpuTerrainError::InvalidGeometry);
                    }
                    work.push((child as usize, false));
                }
            }
        }
    }
    Ok(rows)
}

fn append_triangle(
    rows: &mut Vec<TerrainRow>,
    chunk: &TerrainCollisionChunk,
    triangle: &mechanic_world::TriangleBvhTriangle,
    origin: DVec3,
) -> Result<(), GpuTerrainError> {
    if !triangle.group_mask.intersects(chunk.active_groups) {
        return Ok(());
    }
    let mut points = [Vec3::ZERO; 3];
    for (point, index) in points.iter_mut().zip(triangle.indices) {
        let vertex = chunk
            .vertices
            .get(index as usize)
            .ok_or(GpuTerrainError::InvalidGeometry)?;
        *point = (chunk.origin.0 - origin + Vec3::from_array(*vertex).as_dvec3()).as_vec3();
        if !point.is_finite() {
            return Err(GpuTerrainError::InvalidGeometry);
        }
    }
    let response = chunk
        .triangle_surface_response(triangle.indices)
        .map_err(|_| GpuTerrainError::InvalidGeometry)?
        .to_array();
    let [first, second, third] = points;
    rows.push(TerrainRow {
        minimum: first.min(second).min(third).extend(0.0).to_array(),
        maximum: first.max(second).max(third).extend(0.0).to_array(),
        first: first.extend(0.0).to_array(),
        second: second.extend(0.0).to_array(),
        third: third.extend(0.0).to_array(),
        response,
        metadata: [
            u32::try_from(rows.len() + 1).map_err(|_| GpuTerrainError::Capacity)?,
            1,
            0,
            0,
        ],
    });
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::GpuPhysicsConfig;
    use bevy_math::IVec3;
    use mechanic_core::{BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, GridRotation};
    use mechanic_world::{
        TerrainTriangleGroupMask, TriangleBvh, TriangleBvhTriangle, WorldPosition,
    };

    pub(in crate::device) fn chunk() -> TerrainCollisionChunk {
        let mut weights = [0.0; TerrainMaterial::COUNT];
        weights[usize::from(TerrainMaterial::Soil.code())] = 1.0;
        TerrainCollisionChunk {
            vertices: vec![[-5.0, 0.0, -5.0], [0.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
            indices: vec![0, 1, 2],
            material_weights: vec![weights; 3],
            triangle_bvh: TriangleBvh {
                triangles: vec![TriangleBvhTriangle {
                    indices: [0, 1, 2],
                    group_mask: TerrainTriangleGroupMask::REGULAR,
                }],
                ..Default::default()
            },
            active_groups: TerrainTriangleGroupMask::REGULAR,
            generation: 1,
            ..Default::default()
        }
    }

    pub(in crate::device) fn rigid_chunk() -> TerrainCollisionChunk {
        let mut terrain = chunk();
        for weights in &mut terrain.material_weights {
            weights.fill(0.0);
            weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
        }
        terrain
    }

    #[test]
    fn terrain_upload_rebases_before_narrowing_and_uses_material_response() {
        let mut chunk = chunk();
        let origin = DVec3::splat(1.0e10);
        chunk.origin = WorldPosition(origin);
        let rows = pack_chunks([&chunk], origin).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].first.map(f32::to_bits),
            [-5.0_f32, 0.0, -5.0, 0.0].map(f32::to_bits)
        );
        assert!((rows[0].response[0] - 0.7).abs() < 1.0e-6);
        assert_eq!(rows[0].metadata, [1, 1, 0, 0]);
        chunk.active_groups = TerrainTriangleGroupMask::default();
        assert!(pack_chunks([&chunk], origin).unwrap().is_empty());
    }

    #[test]
    fn terrain_upload_rejects_invalid_indices_and_non_finite_materials() {
        let mut chunk = chunk();
        chunk.material_weights[0][0] = f32::NAN;
        assert!(pack_chunks([&chunk], DVec3::ZERO).is_err());
        chunk.material_weights[0][0] = 0.0;
        chunk.triangle_bvh.triangles[0].indices[0] = 999;
        assert!(pack_chunks([&chunk], DVec3::ZERO).is_err());
    }

    #[test]
    fn terrain_upload_preserves_bvh_escape_ranges() {
        use mechanic_world::{TriangleBvhNode, WorldBounds};
        let mut chunk = chunk();
        let bounds = WorldBounds {
            minimum: WorldPosition(DVec3::new(-5.0, 0.0, -5.0)),
            maximum: WorldPosition(DVec3::new(5.0, 0.0, 5.0)),
        };
        chunk.triangle_bvh.nodes = vec![
            TriangleBvhNode {
                bounds,
                left_child: Some(1),
                right_child: Some(2),
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            },
            TriangleBvhNode {
                bounds,
                triangle_count: 1,
                group_mask: TerrainTriangleGroupMask::REGULAR,
                ..Default::default()
            },
            // Inactive seam geometry must not enter the uploaded traversal.
            TriangleBvhNode {
                bounds,
                triangle_count: 1,
                ..Default::default()
            },
        ];
        let rows = pack_chunks([&chunk], DVec3::ZERO).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows.iter().map(|row| row.metadata[0]).collect::<Vec<_>>(),
            vec![3, 3, 3]
        );
        assert_eq!(
            rows.iter().map(|row| row.metadata[1]).collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        chunk.triangle_bvh.nodes[0].left_child = Some(0);
        assert!(pack_chunks([&chunk], DVec3::ZERO).is_err());
    }

    #[test]
    #[ignore = "diagnostic GPU impact trace; run explicitly with --ignored --nocapture"]
    fn terrain_downward_impact_trace() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain impact trace requires an adapter");
        eprintln!("Terrain impact adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        for speed in [1.0, 5.0, 20.0] {
            let mut graph = ConstructionGraph::new();
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
            let mut creation = graph.compile().unwrap();
            for collider in &mut creation.colliders {
                collider.material_properties.restitution = 0.0;
                collider.material_properties.youngs_modulus_pa = 200.0e9;
            }
            let mut gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.write_terrain_chunks(&device, &queue, [&chunk()], DVec3::ZERO)
                .unwrap();
            gpu.enable_async_readback();
            gpu.apply_impulse(
                &device,
                &queue,
                0,
                creation.compounds[0].root_translation,
                Vec3::NEG_Y * speed * creation.compounds[0].mass_properties.mass,
            )
            .unwrap();
            let mut previous_velocity = -speed;
            let mut previous_height = creation.compounds[0].root_translation.y;
            for tick in 1..=60 {
                gpu.dispatch_tick(&device, &queue, tick);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let sample = gpu
                    .poll_tick_readback(&device)
                    .unwrap()
                    .expect("completed impact tick");
                let pose = sample.transforms[0];
                let rotation = bevy_math::Quat::from_array(pose.rotation);
                let vertical_radius = (rotation.inverse() * Vec3::Y).abs().element_sum() * 0.5;
                let penetration = (vertical_radius - pose.position[1]).max(0.0);
                let incoming = (previous_velocity
                    - mechanic_core::STANDARD_GRAVITY_M_S2_F32 * mechanic_core::TICK_SECONDS_F32)
                    * 0.999;
                let outgoing = sample.velocities[0].linear[1];
                let position_correction = pose.position[1]
                    - (previous_height + incoming * mechanic_core::TICK_SECONDS_F32);
                println!(
                    "{{\"scenario\":\"terrain_downward_impact\",\"speed_mps\":{speed},\"tick\":{tick},\"penetration_m\":{penetration},\"incoming_y_mps\":{incoming},\"outgoing_y_mps\":{outgoing},\"restitution\":0,\"recovery_velocity_target_mps\":0,\"position_correction_y_m\":{position_correction},\"contacts\":{},\"failure_flags\":{}}}",
                    sample.diagnostics.contact_count, sample.diagnostics.error_flags
                );
                assert_eq!(sample.diagnostics.error_flags, 0);
                previous_velocity = outgoing;
                previous_height = pose.position[1];
            }
        }
    }

    #[test]
    fn rigid_terrain_impacts_do_not_tunnel_or_rebound_from_recovery() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain impact regression requires an adapter");
        eprintln!(
            "Terrain impact regression adapter: {:?}",
            adapter.get_info()
        );
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        for blocks in [1, 4] {
            for speed in [0.0, 1.0, 5.0, 20.0] {
                let mut graph = ConstructionGraph::new();
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [blocks; 3],
                            BuildPose::from_half_grid(
                                IVec3::new(0, if blocks == 1 { 1 } else { 5 }, 0),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    ))
                    .unwrap();
                let mut creation = graph.compile().unwrap();
                for collider in &mut creation.colliders {
                    collider.material_properties.restitution = 0.0;
                    collider.material_properties.youngs_modulus_pa = 200.0e9;
                }
                let mut gpu = GpuPhysics::new_with_config(
                    &device,
                    &queue,
                    &creation,
                    GpuPhysicsConfig {
                        ground_plane_enabled: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                gpu.write_terrain_chunks(&device, &queue, [&rigid_chunk()], DVec3::ZERO)
                    .unwrap();
                gpu.enable_async_readback();
                gpu.apply_impulse(
                    &device,
                    &queue,
                    0,
                    creation.compounds[0].root_translation,
                    Vec3::NEG_Y * speed * creation.compounds[0].mass_properties.mass,
                )
                .unwrap();
                for tick in 1..=120 {
                    gpu.dispatch_tick(&device, &queue, tick);
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    let sample = gpu
                        .poll_tick_readback(&device)
                        .unwrap()
                        .expect("completed tick");
                    let pose = sample.transforms[0];
                    let rotation = bevy_math::Quat::from_array(pose.rotation);
                    let radius = (rotation.inverse() * Vec3::Y).abs().element_sum()
                        * f32::from(blocks)
                        * 0.125;
                    let penetration = radius - pose.position[1];
                    assert_eq!(sample.diagnostics.error_flags, 0);
                    assert!(
                        penetration <= if tick > 100 { 0.002 } else { 0.005 },
                        "{blocks} blocks at {speed} m/s, tick {tick}: penetration {penetration}"
                    );
                    assert!(
                        sample.velocities[0].linear[1] <= 0.001,
                        "{blocks} blocks at {speed} m/s, tick {tick}: rebound {:?}",
                        sample.velocities[0]
                    );
                }
            }
        }
    }

    #[test]
    fn dense_terrain_triangles_share_support_without_overflow_or_rebound() {
        check_dense_terrain_impact(0.0, 0.0);
    }

    #[test]
    fn curved_terrain_triangles_support_impacts_without_overflow_or_rebound() {
        for curvature in [0.02, -0.02] {
            for tilt in [0.0_f32, 20.0] {
                check_dense_terrain_impact(curvature, tilt.to_radians());
            }
        }
    }

    // Clip to the four footprint faces in box space. The maximum height over
    // these polygons measures the actual triangulated surface, including edge
    // intersections, rather than an infinite plane through its highest vertex.
    pub(super) fn box_terrain_penetration(
        chunks: &[TerrainCollisionChunk],
        origin: DVec3,
        center: Vec3,
        rotation: bevy_math::Quat,
    ) -> f32 {
        let mut penetration = f32::NEG_INFINITY;
        for chunk in chunks {
            for triangle in &chunk.triangle_bvh.triangles {
                if !triangle.group_mask.intersects(chunk.active_groups) {
                    continue;
                }
                let mut polygon: Vec<_> = triangle
                    .indices
                    .iter()
                    .map(|&index| {
                        rotation.inverse()
                            * ((chunk.origin.0 - origin
                                + Vec3::from_array(chunk.vertices[index as usize]).as_dvec3())
                            .as_vec3()
                                - center)
                    })
                    .collect();
                for axis in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
                    let mut clipped = Vec::new();
                    for index in 0..polygon.len() {
                        let first = polygon[index];
                        let second = polygon[(index + 1) % polygon.len()];
                        let a = first.dot(axis) - 0.5;
                        let b = second.dot(axis) - 0.5;
                        if a <= 0.0 {
                            clipped.push(first);
                        }
                        if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
                            clipped.push(first.lerp(second, a / (a - b)));
                        }
                    }
                    polygon = clipped;
                }
                for vertex in polygon {
                    penetration = penetration.max(vertex.y + 0.5);
                }
            }
        }
        penetration
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keep fixture setup and tick acceptance together"
    )]
    fn check_dense_terrain_impact(curvature: f32, tilt: f32) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("dense terrain regression requires an adapter");
        eprintln!("Dense terrain adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mut creation = graph.compile().unwrap();
        let orientation = bevy_math::Quat::from_rotation_z(tilt);
        let normal = orientation * Vec3::Y;
        for body in &mut creation.compounds {
            body.root_translation = orientation * body.root_translation;
            body.root_rotation = orientation * body.root_rotation;
        }
        for collider in &mut creation.colliders {
            collider.material_properties.restitution = 0.0;
            collider.material_properties.youngs_modulus_pa = 200.0e9;
        }
        // Split the 5 cm mesh into two chunks through the footprint. Reduction
        // must span chunk seams, while preserving finite triangle identities.
        let mut chunks = [rigid_chunk(), rigid_chunk()];
        for chunk in &mut chunks {
            chunk.vertices.clear();
            chunk.indices.clear();
            chunk.material_weights.clear();
            chunk.triangle_bvh.triangles.clear();
        }
        for x in -12_i16..12 {
            for z in -12_i16..12 {
                let chunk = &mut chunks[usize::from(x >= 0)];
                let base = u32::try_from(chunk.vertices.len()).unwrap();
                let x = f32::from(x) * 0.05;
                let z = f32::from(z) * 0.05;
                chunk.vertices.extend([
                    [x, 0.0, z],
                    [x, 0.0, z + 0.05],
                    [x + 0.05, 0.0, z + 0.05],
                    [x + 0.05, 0.0, z],
                ]);
                let mut weights = [0.0; TerrainMaterial::COUNT];
                weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
                chunk.material_weights.extend([weights; 4]);
                for indices in [[base, base + 1, base + 2], [base, base + 2, base + 3]] {
                    chunk.indices.extend(indices);
                    chunk.triangle_bvh.triangles.push(TriangleBvhTriangle {
                        indices,
                        group_mask: TerrainTriangleGroupMask::REGULAR,
                    });
                }
            }
        }
        for chunk in &mut chunks {
            for vertex in &mut chunk.vertices {
                vertex[1] = -curvature * (vertex[0] * vertex[0] + vertex[2] * vertex[2]);
                *vertex = (orientation * Vec3::from_array(*vertex)).to_array();
            }
        }
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        gpu.write_terrain_chunks(&device, &queue, chunks.iter(), DVec3::ZERO)
            .unwrap();
        gpu.enable_async_readback();
        gpu.apply_impulse(
            &device,
            &queue,
            0,
            creation.compounds[0].root_translation,
            -normal * 20.0 * creation.compounds[0].mass_properties.mass,
        )
        .unwrap();
        let mut supported_ticks = 0;
        let mut maximum_penetration = 0.0_f32;
        let mut settling_penetration = 0.0_f32;
        let mut maximum_upward_velocity = 0.0_f32;
        let mut maximum_contacts = 0;
        for tick in 1..=120 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let sample = gpu
                .poll_tick_readback(&device)
                .unwrap()
                .expect("completed tick");
            assert_eq!(sample.diagnostics.error_flags, 0);
            assert!(sample.diagnostics.contact_count <= if curvature == 0.0 { 4 } else { 5 });
            supported_ticks += usize::from(sample.diagnostics.contact_count > 0);
            let pose = sample.transforms[0];
            let rotation = bevy_math::Quat::from_array(pose.rotation);
            let center = Vec3::from_slice(&pose.position[..3]);
            let penetration = box_terrain_penetration(&chunks, DVec3::ZERO, center, rotation);
            maximum_penetration = maximum_penetration.max(penetration);
            if tick > 100 {
                settling_penetration = settling_penetration.max(penetration);
            }
            maximum_upward_velocity = maximum_upward_velocity.max(sample.velocities[0].linear[1]);
            maximum_contacts = maximum_contacts.max(sample.diagnostics.contact_count);
            assert!(
                penetration <= if tick > 100 { 0.002 } else { 0.005 },
                "curvature {curvature}, tilt {tilt}, tick {tick}: penetration {penetration}"
            );
            assert!(
                sample.velocities[0].linear[1] <= 0.001,
                "tick {tick}: rebound {:?}",
                sample.velocities[0]
            );
        }
        assert!(
            supported_ticks > 100,
            "only {supported_ticks} supported ticks"
        );
        eprintln!(
            "{{\"scenario\":\"dense_terrain_impact\",\"speed_mps\":20,\"curvature\":{curvature},\"tilt_radians\":{tilt},\"triangles\":1152,\"ticks\":120,\"supported_ticks\":{supported_ticks},\"maximum_penetration_m\":{maximum_penetration},\"settling_penetration_m\":{settling_penetration},\"maximum_upward_velocity_mps\":{maximum_upward_velocity},\"maximum_contacts\":{maximum_contacts},\"failure_flags\":0}}"
        );
    }

    /// A construction edit replaces the whole physics scene. Adopting the retired
    /// scene's terrain must keep the same geometry collidable without uploading
    /// or repacking a single chunk.
    #[test]
    fn adopted_terrain_residency_collides_without_uploading_chunks_again() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain residency regression requires an adapter");
        eprintln!("Terrain residency adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let scene = || {
            let mut graph = ConstructionGraph::new();
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap();
            let creation = graph.compile().unwrap();
            GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let mut retired = scene();
        let mut cache = TerrainPreparationCache::default();
        let prepared = cache.prepare([&chunk()], DVec3::ZERO, 1).unwrap();
        let uploaded = retired
            .publish_prepared_terrain(&device, &queue, &prepared, 1, DVec3::ZERO)
            .unwrap();
        assert_eq!(uploaded.uploaded_chunks, 1);
        retired.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let before = retired.read_last_tick(&device).unwrap();
        assert_eq!(before.error_flags, 0);
        assert_eq!(before.contact_count, 4);

        // The replacement scene stands in for the one a placement compiles.
        let mut replacement = scene();
        replacement.adopt_terrain_residency(&device, retired.take_terrain_residency());
        replacement.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let adopted = replacement.read_last_tick(&device).unwrap();
        assert_eq!(adopted.error_flags, 0);
        assert_eq!(
            adopted.contact_count, before.contact_count,
            "adopted terrain must support the same contacts"
        );
        assert_eq!(adopted.active_contact_count, before.active_contact_count);

        // Republishing the same cut reuses every allocation instead of uploading it.
        let republished = cache
            .prepare([&chunk()], DVec3::ZERO, 2)
            .and_then(|prepared| {
                replacement.publish_prepared_terrain(&device, &queue, &prepared, 2, DVec3::ZERO)
            })
            .unwrap();
        assert_eq!(republished.uploaded_chunks, 0);
        assert_eq!(republished.reused_chunks, 1);
        assert_eq!(republished.copied_bytes, 0);
    }

    #[test]
    fn an_immovable_body_generates_no_terrain_contacts() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain GPU regression requires an adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let contacts = |anchored: bool| {
            let mut graph = ConstructionGraph::new();
            let mechanic_core::BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                panic!("cuboid spawns");
            };
            let creation = graph
                .compile_with_static_parts(anchored.then_some(part))
                .unwrap();
            let mut gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    ..Default::default()
                },
            )
            .unwrap();
            gpu.write_terrain_chunks(&device, &queue, [&chunk()], DVec3::ZERO)
                .unwrap();
            gpu.dispatch_tick(&device, &queue, 1);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let readback = gpu.read_last_tick(&device).unwrap();
            assert_eq!(readback.error_flags, 0);
            readback.contact_count
        };

        // The solver discards a terrain contact whose only body cannot move, so
        // generating one is pure cost. The dynamic case proves the same geometry
        // still contacts.
        assert_eq!(contacts(false), 4);
        assert_eq!(contacts(true), 0);
    }

    #[test]
    fn gpu_terrain_contacts_use_finite_triangles_without_ground_planes() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("terrain GPU regression requires an adapter");
        eprintln!("Terrain GPU test adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let mut gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                ground_plane_enabled: false,
                ..Default::default()
            },
        )
        .unwrap();
        let terrain = chunk();
        gpu.write_terrain_chunks(&device, &queue, [&terrain], DVec3::ZERO)
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 1);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let readback = gpu.read_last_tick(&device).unwrap();
        assert_eq!(readback.error_flags, 0);
        assert_eq!(readback.contact_count, 4);
        assert_eq!(readback.active_contact_count, 4);
        let mut invalid = terrain.clone();
        invalid.triangle_bvh.triangles[0].indices[0] = 999;
        assert!(
            gpu.write_terrain_chunks(&device, &queue, [&invalid], DVec3::ZERO)
                .is_err()
        );
        gpu.dispatch_tick(&device, &queue, 2);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(gpu.read_last_tick(&device).unwrap().contact_count, 4);
        // Moving the finite triangle away must remove support rather than leave
        // the infinite plane through it behind.
        let mut away = terrain;
        // Its AABB still overlaps the box; only finite polygon clipping rejects it.
        away.origin = WorldPosition(DVec3::X * 4.0);
        gpu.write_terrain_chunks(&device, &queue, [&away], DVec3::ZERO)
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 3);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let readback = gpu.read_last_tick(&device).unwrap();
        assert_eq!(readback.error_flags, 0);
        assert_eq!(readback.contact_count, 0);
        assert_eq!(readback.active_contact_count, 0);

        // Coplanar surfaces with different responses must remain distinct.
        let mut soil = chunk();
        soil.origin = WorldPosition(DVec3::Y * 0.25);
        let mut rock = rigid_chunk();
        rock.origin = soil.origin;
        gpu.write_terrain_chunks(&device, &queue, [&soil, &rock], DVec3::ZERO)
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 4);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let readback = gpu.read_last_tick(&device).unwrap();
        assert_eq!(readback.error_flags, 0);
        assert_eq!(readback.contact_count, 8);

        // Exceed the local reduction table with distinct planes. Every plane
        // still emits its support; the table limit is not a geometry limit.
        let layers: Vec<_> = (0_i16..17)
            .map(|layer| {
                let mut terrain = rigid_chunk();
                terrain.origin = WorldPosition(DVec3::Y * (0.3 + f64::from(layer) * 0.001));
                terrain
            })
            .collect();
        gpu.write_terrain_chunks(&device, &queue, layers.iter(), DVec3::ZERO)
            .unwrap();
        gpu.dispatch_tick(&device, &queue, 5);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let readback = gpu.read_last_tick(&device).unwrap();
        assert_eq!(readback.error_flags, 0);
        assert_eq!(readback.contact_count, 68);
    }
}
