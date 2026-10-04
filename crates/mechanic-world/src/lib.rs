//! Deterministic finite-world generation, sparse edits, meshing, queries, and persistence.
//!
//! Rendering remains owned by `mechanic-app`, and the GPU runtime consumes only
//! [`TerrainCollisionChunk`]. This crate deliberately has no Bevy ECS dependency.

mod breakage;
mod celestial;
mod clumps;
mod construction_collision;
mod coordinates;
mod edits;
mod footprint;
mod generation;
mod material_response;
mod mesh;
mod persistence;
mod query;
mod soil;
mod streaming;
#[cfg(test)]
mod testing;
mod transvoxel;
mod water;

pub use breakage::{
    BreakageAccumulator, BreakagePatch, BreakageResponse, CELL_QUANTA, ExtractionCell,
    MATERIAL_QUANTUM_M3,
};
pub use celestial::{
    CelestialMoon, CelestialSky, CelestialStar, CelestialSystem, MAX_SYSTEM_MOONS,
    MAX_SYSTEM_STARS, SkyMoon, SkyStar, blackbody_colour,
};
pub use clumps::{
    ClumpCollection, MAX_ACTIVE_CLUMPS, MaterialClump, SETTLE_SECONDS, TransferLimits,
};
pub use construction_collision::{
    ConstructionBodyState, ConstructionCollisionIndex, ConstructionCollisionMetrics,
    ConstructionContact, KinematicCollisionScene,
};
pub use coordinates::{
    BRICK_EDGE_CELLS, BRICK_EDGE_METERS, BrickCoord, CoordinateError, FloatingOrigin,
    TERRAIN_CELL_METERS, WORLD_HALF_EXTENT_CELLS, WORLD_HALF_EXTENT_METERS, WorldCell,
    WorldGeneratorVersion, WorldPosition, WorldSeed,
};
pub use edits::{
    BrickDecodeError, REMOVED_CELL_CUBIC_METERS, REMOVED_CELL_LITRES, Repose, SPOIL_LOOSENESS,
    SedimentApplied, SedimentChange, SpoilSlump, TerrainBrick, TerrainDensityClass,
    TerrainEditBatch, TerrainEditError, TerrainEditOutcome, TerrainNodeId, TerrainNodeSummary,
    TerrainOctree, TerrainOctreeSnapshot, TerrainSource, decode_brick, encode_brick,
};
pub use footprint::LoadFootprint;
pub use generation::{
    Axis, BarkTraits, FoliageBlob, FoliageSpec, GenomeSweep, LakeBasin, LeafTraits, Outflow, Part,
    RiverReach, RootsSpec, Segment, SpeciesSpec, SurfaceId, SurfaceLook, SurfacePalette,
    TREE_TEXTURE_LUMA, TREE_TEXTURE_METRES, TerrainField, TerrainMaterial, TerrainSample,
    TextureSet, TreeLod, TreeMetrics, TreeModel, TreeSurface, TreeTexture, TreeTextureMaps,
    WaterBody, WaterSurface, WorldgenError, WorldgenSpec, grow_tree,
};
pub use material_response::TerrainMaterialError;
pub use mesh::{
    LatticeEdgeVertexCache, PreparedTerrainRegion, TerrainCollisionChunk, TerrainIndexGroups,
    TerrainMeshChunk, TerrainMeshMetrics, TerrainMeshRequest, TerrainRayHit,
    TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WaterSheet,
    WaterTile, WorldBounds, joined_water_sheet, mesh_chunk, mesh_chunk_profiled,
    mesh_chunk_profiled_prepared, water_sheet,
};
pub use persistence::{
    AUTOSAVE_DEBOUNCE, AUTOSAVE_DIRTY_INTERVAL, AutosaveState, FrozenCreationDoc, OpenWorldOutcome,
    SavedWorld, SavedWorldStatus, WORLD_FORMAT_VERSION, WorldCreationInstanceDoc, WorldDocument,
    WorldInstanceIndexDoc, WorldPoseDoc, WorldSaveError, WorldStore,
};
pub use query::{
    ActiveTerrainScene, FoundationRefresh, FoundationSample, FoundationSpatialIndex,
    FoundationSupport, KinematicCapsule, KinematicCapsuleConfig, KinematicContactReaction,
    KinematicInput, KinematicSupport, KinematicTickOutcome, TerrainDensity, TerrainScene,
    TerrainSpatialIndex, raycast_density,
};
pub use soil::{
    GROUND_NORMAL_MIN_Y, SoilAccumulator, SoilCompression, SoilPatch, SoilResponse, loose_strength,
};
pub use streaming::{
    ActiveTerrainNode, CAVE_STREAMED_LEVEL, MAX_STREAMED_LEVEL, MIN_TERRAIN_DETAIL_SCALE,
    STREAMED_LEVELS, TERRAIN_HORIZON_METRES, TerrainActivation, TerrainBoundsCache, TerrainFace,
    TerrainPublicationDelta, TerrainPublicationUpsert, TerrainReadiness, TerrainSelection,
    TerrainSelectionStats, TerrainStreamer, TerrainTransitionMask, TerrainView,
    publication_face_mask, select_active_nodes, select_active_nodes_cached,
    select_active_nodes_with_interests, terrain_loading_worker_count, terrain_worker_count,
};
pub use water::{
    BedDoc, ErosionConfig, GrassDoc, JoinedCellDoc, NativeGrass, PoolDoc, PoolView, RunningView,
    SURFACE_TILE_COLUMNS, SedimentDiagnosticColumn, SedimentDiagnostics, SedimentDoc,
    SedimentLedger, SedimentLoad, SheetDoc, SoilDoc, StoredSurface, StoredWaterDoc, SurfaceTile,
    SurplusDoc, TerrainWater, WATER_CELL_EDGE_CELLS, WATER_CELL_METRES, WaterCell, WaterGround,
    WaterLedger, WaterNetwork, WaterPhases, WaterShift, WaterStep, WaterSurfaces, WaterWorld,
    WetGround,
};
