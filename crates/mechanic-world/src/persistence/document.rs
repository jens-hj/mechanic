//! The serialized world: header, player pose, and creation instances.

use super::error::WorldSaveError;
use super::files::unix_now;
use crate::{WorldGeneratorVersion, WorldPosition, WorldSeed};
use mechanic_core::{CreationDocument, DimensionLinkId};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// World document version written by this build.
pub const WORLD_FORMAT_VERSION: u32 = 6;

/// Serializable global orientation and position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldPoseDoc {
    /// Global translation in metres.
    pub translation: WorldPosition,
    /// Quaternion in x/y/z/w order.
    pub rotation: [f32; 4],
}

impl Default for WorldPoseDoc {
    fn default() -> Self {
        Self {
            translation: WorldPosition::default(),
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Lightweight index row kept in `world.ron`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldInstanceIndexDoc {
    /// Stable instance identity.
    pub id: u64,
    /// Display label for menus and recovery diagnostics.
    pub name: String,
}

/// One placed creation stored independently from the world index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldCreationInstanceDoc {
    /// Stable instance identity.
    pub id: u64,
    /// Embedded authored creation graph.
    pub creation: CreationDocument,
    /// Latest global root pose.
    pub root_pose: WorldPoseDoc,
    /// Latest articulated coordinates in stable compiled-joint order.
    pub joint_coordinates: Vec<f32>,
}

/// Frozen creation target, restored before physics begins.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrozenCreationDoc {
    /// Active Dimension Link identifying the held structural creation.
    pub link: DimensionLinkId,
    /// Validated global link-center target in metres, independent of animation.
    pub target: WorldPosition,
    /// Cardinal heading in quarter turns, from zero through three.
    pub heading: u8,
    /// Published construction generation containing this creation.
    pub construction_generation: u64,
}

/// Top-level metadata and player state for one finite world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldDocument {
    /// Persistence format version.
    pub version: u32,
    /// Display name.
    pub name: String,
    /// Deterministic generation recipe.
    pub generator_version: WorldGeneratorVersion,
    /// Actual numeric world seed.
    pub seed: WorldSeed,
    /// Unix timestamp of most recent play.
    pub last_played_unix_seconds: u64,
    /// Latest player pose.
    pub player_pose: WorldPoseDoc,
    /// Position from which this Garage visit began.
    pub return_anchor: Option<WorldPosition>,
    /// Published paired World/Garage construction generation. Zero is empty.
    pub construction_generation: u64,
    /// Sole active Dimension Link across this world's paired spaces.
    pub active_dimension_link: Option<DimensionLinkId>,
    /// Optional held creation and its final alignment target.
    pub frozen_creation: Option<FrozenCreationDoc>,
    /// Next stable Dimension Link identity allocated in this world.
    pub next_dimension_link_id: u64,
    /// Independently saved placed creations.
    pub instances: Vec<WorldInstanceIndexDoc>,
}

impl WorldDocument {
    /// Creates a new named world at its deterministic safe spawn.
    pub fn new(name: impl Into<String>, seed: WorldSeed, safe_spawn: WorldPosition) -> Self {
        Self {
            version: WORLD_FORMAT_VERSION,
            name: name.into(),
            generator_version: WorldGeneratorVersion::CURRENT,
            seed,
            last_played_unix_seconds: unix_now(),
            player_pose: WorldPoseDoc {
                translation: safe_spawn,
                ..WorldPoseDoc::default()
            },
            return_anchor: None,
            construction_generation: 0,
            active_dimension_link: None,
            frozen_creation: None,
            next_dimension_link_id: 1,
            instances: Vec::new(),
        }
    }
}

pub(super) fn validate_frozen_creation(
    world: &WorldDocument,
    path: &Path,
) -> Result<(), WorldSaveError> {
    let Some(frozen) = world.frozen_creation else {
        return Ok(());
    };
    let message = if !frozen.target.0.is_finite() {
        Some("target must be finite")
    } else if frozen.heading >= 4 {
        Some("heading must be a cardinal quarter turn")
    } else if world.active_dimension_link != Some(frozen.link) {
        Some("link must match the active Dimension Link")
    } else if world.construction_generation == 0
        || frozen.construction_generation != world.construction_generation
    {
        Some("generation must match the nonzero published construction generation")
    } else {
        None
    };
    message.map_or(Ok(()), |message| {
        Err(WorldSaveError::InvalidFrozenCreation {
            path: path.to_owned(),
            message,
        })
    })
}
