//! Deterministic finite-world generation, sparse edits, meshing, queries, and persistence.
//!
//! Rendering remains owned by `mechanic-app`, and the GPU runtime consumes only
//! [`TerrainCollisionChunk`]. This crate deliberately has no Bevy ECS dependency.

mod coordinates;
mod edits;
mod generation;
mod mesh;
mod persistence;
mod query;
mod streaming;
mod transvoxel;

pub use coordinates::{
    BRICK_EDGE_CELLS, BRICK_EDGE_METERS, BUILD_POSITION_TICK_METERS, BrickCoord, CoordinateError,
    FloatingOrigin, TERRAIN_CELL_METERS, WORLD_HALF_EXTENT_CELLS, WORLD_HALF_EXTENT_METERS,
    WorldCell, WorldGeneratorVersion, WorldPosition, WorldSeed,
};
pub use edits::{
    BrickDecodeError, REMOVED_CELL_CUBIC_METERS, REMOVED_CELL_LITRES, TerrainBrick,
    TerrainDensityClass, TerrainEditBatch, TerrainEditError, TerrainEditOutcome, TerrainNodeId,
    TerrainNodeSummary, TerrainOctree, TerrainOctreeSnapshot, TerrainSource, decode_brick,
    encode_brick,
};
pub use generation::{CaveEdge, CaveGraph, CaveNode, TerrainField, TerrainMaterial, TerrainSample};
pub use mesh::{
    LatticeEdgeVertexCache, PreparedTerrainRegion, TerrainCollisionChunk, TerrainIndexGroups,
    TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest, TerrainRayHit,
    TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds,
    mesh_chunk, mesh_chunk_profiled, mesh_chunk_profiled_prepared,
};
pub use persistence::{
    AUTOSAVE_DEBOUNCE, AUTOSAVE_DIRTY_INTERVAL, AutosaveState, OpenWorldResult, SavedWorld,
    SavedWorldStatus, WORLD_FORMAT_VERSION, WorldCreationInstanceDoc, WorldDocument,
    WorldInstanceIndexDoc, WorldPoseDoc, WorldSaveError, WorldStore,
};
pub use query::{
    ActiveTerrainScene, FoundationRefresh, FoundationSample, FoundationSpatialIndex,
    FoundationSupport, KinematicCapsule, KinematicCapsuleConfig, KinematicInput, TerrainDensity,
    TerrainScene, TerrainSpatialIndex, WorldConstructionEditability, raycast_density,
};
pub use streaming::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainCoordinatorResult, TerrainFace,
    TerrainPublicationDelta, TerrainPublicationUpsert, TerrainReadiness, TerrainSelection,
    TerrainSelectionDelta, TerrainSelectionStats, TerrainStreamer, TerrainTransitionMask,
    active_face_mask, publication_face_mask, select_active_nodes, select_active_nodes_cached,
    terrain_loading_worker_count, terrain_worker_count,
};
