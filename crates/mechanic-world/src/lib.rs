//! Deterministic finite-world generation, sparse edits, meshing, queries, and persistence.
//!
//! Rendering remains owned by `mechanic-app`, and the GPU runtime consumes only
//! [`TerrainCollisionChunk`]. This crate deliberately has no Bevy ECS dependency.

mod breakage;
mod clumps;
mod construction_collision;
mod coordinates;
mod edits;
mod generation;
mod material_response;
mod mesh;
mod persistence;
mod query;
mod soil;
pub use breakage::{
    BreakageAccumulator, BreakagePatch, BreakageResponse, ExtractionCell, MATERIAL_QUANTUM_M3,
};
pub use clumps::{ClumpCollection, MAX_ACTIVE_CLUMPS, MaterialClump, MaterialTransfer};
mod streaming;
pub use soil::{SoilAccumulator, SoilCompression, SoilPatch, SoilResponse};
mod transvoxel;

pub use construction_collision::{
    ConstructionBodyPose, ConstructionCollisionIndex, ConstructionCollisionMetrics,
    ConstructionContact, KinematicCollisionScene,
};
pub use coordinates::{
    BRICK_EDGE_CELLS, BRICK_EDGE_METERS, BrickCoord, CoordinateError, FloatingOrigin,
    TERRAIN_CELL_METERS, WORLD_HALF_EXTENT_CELLS, WORLD_HALF_EXTENT_METERS, WorldCell,
    WorldGeneratorVersion, WorldPosition, WorldSeed,
};
pub use edits::{
    BrickDecodeError, REMOVED_CELL_CUBIC_METERS, REMOVED_CELL_LITRES, TerrainBrick,
    TerrainDensityClass, TerrainEditBatch, TerrainEditError, TerrainEditOutcome, TerrainNodeId,
    TerrainNodeSummary, TerrainOctree, TerrainOctreeSnapshot, TerrainSource, decode_brick,
    encode_brick,
};
pub use generation::{CaveEdge, CaveGraph, CaveNode, TerrainField, TerrainMaterial, TerrainSample};
pub use material_response::{TerrainMaterialError, TerrainSurfaceResponse};
pub use mesh::{
    LatticeEdgeVertexCache, PreparedTerrainRegion, TerrainCollisionChunk, TerrainIndexGroups,
    TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest, TerrainRayHit,
    TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds,
    mesh_chunk, mesh_chunk_profiled, mesh_chunk_profiled_prepared,
};
pub use persistence::{
    AUTOSAVE_DEBOUNCE, AUTOSAVE_DIRTY_INTERVAL, AutosaveState, FrozenCreationDoc, OpenWorldResult,
    SavedWorld, SavedWorldStatus, WORLD_FORMAT_VERSION, WorldCreationInstanceDoc, WorldDocument,
    WorldInstanceIndexDoc, WorldPoseDoc, WorldSaveError, WorldStore,
};
pub use query::{
    ActiveTerrainScene, FoundationRefresh, FoundationSample, FoundationSpatialIndex,
    FoundationSupport, KinematicCapsule, KinematicCapsuleConfig, KinematicContactReaction,
    KinematicInput, KinematicSupport, KinematicTickResult, TerrainDensity, TerrainScene,
    TerrainSpatialIndex, WorldConstructionEditability, raycast_density,
};
pub use streaming::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainFace, TerrainPublicationDelta,
    TerrainPublicationUpsert, TerrainReadiness, TerrainSelection, TerrainSelectionDelta,
    TerrainSelectionStats, TerrainStreamer, TerrainTransitionMask, publication_face_mask,
    select_active_nodes, select_active_nodes_cached, select_active_nodes_with_interests,
    terrain_loading_worker_count, terrain_worker_count,
};
