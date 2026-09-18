//! Atomic world-document, instance, and edited-brick persistence.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use mechanic_core::{CreationDocument, DimensionLinkId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    TerrainBrick, TerrainOctree, WorldGeneratorVersion, WorldPosition, WorldSeed, decode_brick,
    encode_brick,
};

/// World document version written by this build.
pub const WORLD_FORMAT_VERSION: u32 = 6;
/// Delay after the last mutation before an ordinary autosave.
pub const AUTOSAVE_DEBOUNCE: Duration = Duration::from_secs(2);
/// Maximum time dirty data waits even while mutations continue.
pub const AUTOSAVE_DIRTY_INTERVAL: Duration = Duration::from_secs(30);

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

/// World row used by the most-recent-first world list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedWorld {
    /// Display name.
    pub name: Option<String>,
    /// Actual numeric seed.
    pub seed: Option<WorldSeed>,
    /// Last played timestamp.
    pub last_played_unix_seconds: Option<u64>,
    /// World directory.
    pub path: PathBuf,
    /// Compatibility/corruption state from minimal manifest inspection.
    pub status: SavedWorldStatus,
}

/// World-list compatibility state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SavedWorldStatus {
    /// Current world and generator format; full manifest decoding succeeded.
    Current,
    /// Recognizable manifest from an incompatible format or generator.
    Outdated,
    /// Current-looking or unreadable data that must be preserved for recovery.
    Corrupt {
        /// Exact file that failed inspection.
        file: PathBuf,
        /// User-facing failure detail.
        message: String,
    },
}

/// Result of opening an entry from the world list.
#[derive(Clone, Debug, PartialEq)]
pub enum OpenWorldOutcome {
    /// Current world ready to play.
    Opened(Box<WorldDocument>),
    /// Incompatible direct-child directory was removed as requested by policy.
    OutdatedRemoved {
        /// Exact removed directory.
        path: PathBuf,
    },
}

/// Filesystem owner for all worlds below an application-data root.
#[derive(Clone, Debug)]
pub struct WorldStore {
    root: PathBuf,
}

impl WorldStore {
    /// Atomically saves terrain and loose material in one ownership snapshot.
    ///
    /// # Errors
    /// Reports malformed clumps, encoding failures, or the exact I/O path.
    pub fn save_material_state(
        &self,
        world_name: &str,
        terrain: &TerrainOctree,
        clumps: &crate::ClumpCollection,
    ) -> Result<(), WorldSaveError> {
        let path = self.directory_for(world_name).join("material.bin");
        let corrupt = |message: &str| WorldSaveError::CorruptCurrent {
            path: path.clone(),
            message: message.to_owned(),
        };
        if !clumps.is_valid() {
            return Err(corrupt("invalid clump ownership"));
        }
        let text = ron::to_string(clumps).map_err(|_| corrupt("cannot encode clumps"))?;
        let mut bytes = b"MECS\x01\x00".to_vec();
        bytes.extend_from_slice(
            &u64::try_from(text.len())
                .map_err(|_| corrupt("clump payload too large"))?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(text.as_bytes());
        let snapshot = terrain.snapshot();
        let bricks = snapshot.bricks().collect::<Vec<_>>();
        bytes.extend_from_slice(
            &u64::try_from(bricks.len())
                .map_err(|_| corrupt("too many bricks"))?
                .to_le_bytes(),
        );
        for brick in bricks {
            let payload = encode_brick(brick);
            bytes.extend_from_slice(
                &u64::try_from(payload.len())
                    .map_err(|_| corrupt("brick too large"))?
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(&payload);
        }
        atomic_write(&path, &bytes)
    }

    /// Loads the matching terrain and loose-material ownership snapshot.
    /// A world with no snapshot has no material edits yet.
    ///
    /// # Errors
    /// Rejects corrupt, duplicate, truncated, trailing or unsupported data.
    pub fn load_material_state(
        &self,
        world_name: &str,
    ) -> Result<(TerrainOctree, crate::ClumpCollection), WorldSaveError> {
        let path = self.directory_for(world_name).join("material.bin");
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok((TerrainOctree::default(), crate::ClumpCollection::default()));
            }
            Err(source) => return Err(WorldSaveError::Io { path, source }),
        };
        let corrupt = |message: &str| WorldSaveError::CorruptCurrent {
            path: path.clone(),
            message: message.to_owned(),
        };
        if !bytes.starts_with(b"MECS\x01\x00") {
            return Err(corrupt("unsupported material snapshot"));
        }
        let mut remaining = &bytes[6..];
        let read_size = |input: &mut &[u8]| -> Result<usize, WorldSaveError> {
            let (size, tail) = input
                .split_at_checked(8)
                .ok_or_else(|| corrupt("truncated material snapshot"))?;
            *input = tail;
            usize::try_from(u64::from_le_bytes(
                size.try_into().map_err(|_| corrupt("invalid size"))?,
            ))
            .map_err(|_| corrupt("oversized payload"))
        };
        let size = read_size(&mut remaining)?;
        let (text, rest) = remaining
            .split_at_checked(size)
            .ok_or_else(|| corrupt("truncated clumps"))?;
        remaining = rest;
        let clumps: crate::ClumpCollection =
            ron::de::from_bytes(text).map_err(|_| corrupt("invalid clump payload"))?;
        if !clumps.is_valid() {
            return Err(corrupt("invalid clump ownership"));
        }
        let count = read_size(&mut remaining)?;
        if count > remaining.len() / 8 {
            return Err(corrupt("invalid brick count"));
        }
        let mut terrain = TerrainOctree::default();
        for _ in 0..count {
            let size = read_size(&mut remaining)?;
            let (payload, rest) = remaining
                .split_at_checked(size)
                .ok_or_else(|| corrupt("truncated brick"))?;
            remaining = rest;
            let brick = decode_brick(payload).map_err(|_| corrupt("invalid brick payload"))?;
            if terrain.brick(brick.coordinate()).is_some() {
                return Err(corrupt("duplicate brick"));
            }
            terrain.insert_saved_brick(brick);
        }
        if !remaining.is_empty() {
            return Err(corrupt("trailing material data"));
        }
        Ok((terrain, clumps))
    }

    /// Creates a store rooted at an explicit directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Creates the platform-default Mechanic world store.
    pub fn platform_default() -> Option<Self> {
        data_root().map(|root| Self::new(root.join("Mechanic/worlds")))
    }

    /// Directory for a display name.
    pub fn directory_for(&self, name: &str) -> PathBuf {
        self.root.join(slug(name))
    }

    /// Creates and persists a current-format world. A blank seed is filled by
    /// the operating system's cryptographic random source.
    ///
    /// # Errors
    ///
    /// Reports OS randomness or the exact manifest path that could not be saved.
    pub fn create_world(
        &self,
        name: impl Into<String>,
        seed: Option<u64>,
    ) -> Result<WorldDocument, WorldSaveError> {
        let name = name.into();
        let seed = if let Some(seed) = seed {
            WorldSeed(seed)
        } else {
            let mut bytes = [0_u8; 8];
            getrandom::fill(&mut bytes).map_err(WorldSaveError::Random)?;
            WorldSeed(u64::from_le_bytes(bytes))
        };
        let field = crate::TerrainField::new(seed);
        let world = WorldDocument::new(name, seed, field.safe_spawn());
        self.save_world(&world)?;
        Ok(world)
    }

    /// Creates or updates `world.ron` atomically.
    ///
    /// # Errors
    ///
    /// Reports the exact target file on encoding or I/O failure.
    pub fn save_world(&self, world: &WorldDocument) -> Result<PathBuf, WorldSaveError> {
        let path = self.directory_for(&world.name).join("world.ron");
        validate_frozen_creation(world, &path)?;
        let text = ron::ser::to_string_pretty(world, ron::ser::PrettyConfig::default()).map_err(
            |source| WorldSaveError::Encode {
                path: path.clone(),
                source,
            },
        )?;
        atomic_write(&path, text.as_bytes())?;
        Ok(path)
    }

    /// Atomically writes one placed creation.
    ///
    /// # Errors
    ///
    /// Reports the exact instance path on encoding or I/O failure.
    pub fn save_instance(
        &self,
        world_name: &str,
        instance: &WorldCreationInstanceDoc,
    ) -> Result<PathBuf, WorldSaveError> {
        let path = self
            .directory_for(world_name)
            .join("instances")
            .join(format!("{}.ron", instance.id));
        let text = ron::ser::to_string_pretty(instance, ron::ser::PrettyConfig::default())
            .map_err(|source| WorldSaveError::Encode {
                path: path.clone(),
                source,
            })?;
        atomic_write(&path, text.as_bytes())?;
        Ok(path)
    }

    /// Writes both construction spaces, then atomically publishes their generation.
    ///
    /// The caller's document is updated only after both generation files exist.
    ///
    /// # Errors
    ///
    /// Returns the exact encode or I/O failure without publishing the new generation.
    pub fn save_space_pair(
        &self,
        world: &mut WorldDocument,
        world_space: &WorldCreationInstanceDoc,
        garage_space: &WorldCreationInstanceDoc,
    ) -> Result<(), WorldSaveError> {
        let generation = world.construction_generation.saturating_add(1);
        let directory = self
            .directory_for(&world.name)
            .join("generations")
            .join(generation.to_string());
        Self::save_generation_space(&directory.join("world.ron"), world_space)?;
        Self::save_generation_space(&directory.join("garage.ron"), garage_space)?;
        let previous = world.construction_generation;
        let previous_frozen = world.frozen_creation;
        world.construction_generation = generation;
        if let Some(frozen) = &mut world.frozen_creation {
            frozen.construction_generation = generation;
        }
        if let Err(error) = self.save_world(world) {
            world.construction_generation = previous;
            world.frozen_creation = previous_frozen;
            return Err(error);
        }
        Ok(())
    }

    fn save_generation_space(
        path: &Path,
        instance: &WorldCreationInstanceDoc,
    ) -> Result<(), WorldSaveError> {
        let text = ron::ser::to_string_pretty(instance, ron::ser::PrettyConfig::default())
            .map_err(|source| WorldSaveError::Encode {
                path: path.to_owned(),
                source,
            })?;
        atomic_write(path, text.as_bytes())
    }

    /// Loads the published paired construction generation.
    ///
    /// # Errors
    ///
    /// Returns the exact I/O or parse failure for either published space document.
    pub fn load_space_pair(
        &self,
        world: &WorldDocument,
    ) -> Result<Option<(WorldCreationInstanceDoc, WorldCreationInstanceDoc)>, WorldSaveError> {
        if world.construction_generation == 0 {
            return Ok(None);
        }
        let directory = self
            .directory_for(&world.name)
            .join("generations")
            .join(world.construction_generation.to_string());
        Ok(Some((
            self.load_instance(&directory.join("world.ron"))?,
            self.load_instance(&directory.join("garage.ron"))?,
        )))
    }

    /// Atomically writes one versioned RLE edited brick.
    ///
    /// # Errors
    ///
    /// Reports the exact brick path on I/O failure.
    pub fn save_brick(
        &self,
        world_name: &str,
        brick: &TerrainBrick,
    ) -> Result<PathBuf, WorldSaveError> {
        let coordinate = brick.coordinate();
        let path = self.directory_for(world_name).join("terrain").join(format!(
            "{}_{}_{}.bin",
            coordinate.x, coordinate.y, coordinate.z
        ));
        atomic_write(&path, &encode_brick(brick))?;
        Ok(path)
    }

    /// Loads `world.ron` without modifying any recovery data.
    ///
    /// # Errors
    ///
    /// Reports the exact file on I/O, decoding, or unsupported-version failure.
    pub fn load_world(&self, directory: &Path) -> Result<WorldDocument, WorldSaveError> {
        let path = directory.join("world.ron");
        let text = read_exact(&path)?;
        let world: WorldDocument =
            ron::from_str(&text).map_err(|source| WorldSaveError::Decode {
                path: path.clone(),
                source: Box::new(source),
            })?;
        if world.version != WORLD_FORMAT_VERSION
            || world.generator_version != WorldGeneratorVersion::CURRENT
        {
            return Err(WorldSaveError::UnsupportedVersion { path });
        }
        validate_frozen_creation(&world, &path)?;
        Ok(world)
    }

    /// Loads an instance and names its exact corrupt file on failure.
    ///
    /// # Errors
    ///
    /// Reports the exact instance path on I/O or decoding failure.
    pub fn load_instance(&self, path: &Path) -> Result<WorldCreationInstanceDoc, WorldSaveError> {
        let text = read_exact(path)?;
        ron::from_str(&text).map_err(|source| WorldSaveError::Decode {
            path: path.to_owned(),
            source: Box::new(source),
        })
    }

    /// Loads an edited brick and never silently regenerates corruption.
    ///
    /// # Errors
    ///
    /// Reports the exact brick path on I/O or binary decoding failure.
    pub fn load_brick(&self, path: &Path) -> Result<TerrainBrick, WorldSaveError> {
        let bytes = fs::read(path).map_err(|source| WorldSaveError::Io {
            path: path.to_owned(),
            source,
        })?;
        decode_brick(&bytes).map_err(|source| WorldSaveError::Brick {
            path: path.to_owned(),
            source,
        })
    }

    /// Loads every edited brick for a world in stable filename order.
    ///
    /// # Errors
    ///
    /// Stops at the first corrupt file and reports its exact path. No caller
    /// should replace that file with regenerated procedural terrain.
    pub fn load_bricks(&self, world_name: &str) -> Result<Vec<TerrainBrick>, WorldSaveError> {
        let directory = self.directory_for(world_name).join("terrain");
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(WorldSaveError::Io {
                    path: directory,
                    source,
                });
            }
        };
        let mut paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "bin"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
            .into_iter()
            .map(|path| self.load_brick(&path))
            .collect()
    }

    /// Loads leaves in stable order and rebuilds every parent deterministically.
    ///
    /// # Errors
    ///
    /// Stops at the exact corrupt leaf without modifying any save data.
    pub fn load_octree(&self, world_name: &str) -> Result<TerrainOctree, WorldSaveError> {
        let mut terrain = TerrainOctree::default();
        for brick in self.load_bricks(world_name)? {
            terrain.insert_saved_brick(brick);
        }
        Ok(terrain)
    }

    /// Atomically saves every dirty authoritative leaf and marks only successful
    /// writes clean.
    ///
    /// # Errors
    ///
    /// Stops at the first exact leaf path that fails; later leaves stay dirty.
    pub fn save_dirty_leaves(
        &self,
        world_name: &str,
        terrain: &mut TerrainOctree,
    ) -> Result<Vec<PathBuf>, WorldSaveError> {
        let leaves = terrain.dirty_leaves().collect::<Vec<_>>();
        let mut paths = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let brick =
                terrain
                    .brick(leaf.coordinates)
                    .ok_or_else(|| WorldSaveError::CorruptCurrent {
                        path: self.directory_for(world_name).join("terrain"),
                        message: format!("dirty octree leaf has no payload: {leaf:?}"),
                    })?;
            paths.push(self.save_brick(world_name, brick)?);
            terrain.mark_saved(leaf.coordinates);
        }
        Ok(paths)
    }

    /// Lists current, outdated, and corrupt worlds most-recent-first.
    ///
    /// Only a minimal header is inspected before compatibility classification.
    /// Corrupt current-format data is always retained and reported with its
    /// exact manifest path.
    pub fn list(&self) -> Vec<SavedWorld> {
        let Ok(entries) = fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut worlds = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .map(|entry| {
                let directory = entry.path();
                let manifest = directory.join("world.ron");
                match inspect_manifest(&manifest) {
                    Ok(header)
                        if header.version != WORLD_FORMAT_VERSION
                            || header.generator_version != WorldGeneratorVersion::CURRENT =>
                    {
                        SavedWorld {
                            name: Some(header.name),
                            seed: Some(header.seed),
                            last_played_unix_seconds: Some(header.last_played_unix_seconds),
                            path: directory,
                            status: SavedWorldStatus::Outdated,
                        }
                    }
                    Ok(header) => match self.load_world(&directory) {
                        Ok(world) => SavedWorld {
                            name: Some(world.name),
                            seed: Some(world.seed),
                            last_played_unix_seconds: Some(world.last_played_unix_seconds),
                            path: directory,
                            status: SavedWorldStatus::Current,
                        },
                        Err(error) => SavedWorld {
                            name: Some(header.name),
                            seed: Some(header.seed),
                            last_played_unix_seconds: Some(header.last_played_unix_seconds),
                            path: directory,
                            status: SavedWorldStatus::Corrupt {
                                file: manifest,
                                message: error.to_string(),
                            },
                        },
                    },
                    Err(error) => SavedWorld {
                        name: None,
                        seed: None,
                        last_played_unix_seconds: None,
                        path: directory,
                        status: SavedWorldStatus::Corrupt {
                            file: manifest,
                            message: error.to_string(),
                        },
                    },
                }
            })
            .collect::<Vec<_>>();
        worlds.sort_by(|first, second| {
            second
                .last_played_unix_seconds
                .cmp(&first.last_played_unix_seconds)
                .then_with(|| first.name.cmp(&second.name))
                .then_with(|| first.path.cmp(&second.path))
        });
        worlds
    }

    /// Opens a current list entry or removes an incompatible direct child.
    ///
    /// # Errors
    ///
    /// Current corrupt worlds report their exact failing file and remain
    /// untouched. Outdated deletion validates the direct-child target again.
    pub fn open_entry(&self, entry: &SavedWorld) -> Result<OpenWorldOutcome, WorldSaveError> {
        match &entry.status {
            SavedWorldStatus::Current => self
                .load_world(&entry.path)
                .map(Box::new)
                .map(OpenWorldOutcome::Opened),
            SavedWorldStatus::Outdated => {
                self.delete_world(&entry.path)?;
                Ok(OpenWorldOutcome::OutdatedRemoved {
                    path: entry.path.clone(),
                })
            }
            SavedWorldStatus::Corrupt { file, message } => Err(WorldSaveError::CorruptCurrent {
                path: file.clone(),
                message: message.clone(),
            }),
        }
    }

    /// Removes exactly one resolved world directory.
    ///
    /// # Errors
    ///
    /// Returns an exact path on failure. Callers are responsible for explicit
    /// user confirmation before invoking this destructive action.
    pub fn delete_world(&self, directory: &Path) -> Result<(), WorldSaveError> {
        if directory.parent() != Some(self.root.as_path()) {
            return Err(WorldSaveError::OutsideStore {
                path: directory.to_owned(),
            });
        }
        fs::remove_dir_all(directory).map_err(|source| WorldSaveError::Io {
            path: directory.to_owned(),
            source,
        })
    }
}

/// Autosave timing state independent of Bevy's scheduling layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutosaveState {
    dirty_since: Option<Duration>,
    last_mutation: Option<Duration>,
}

impl AutosaveState {
    /// Marks persistent world state dirty at monotonic `now`.
    pub fn mutate(&mut self, now: Duration) {
        self.dirty_since.get_or_insert(now);
        self.last_mutation = Some(now);
    }

    /// Whether the debounce or maximum dirty interval requires a save.
    pub fn due(self, now: Duration) -> bool {
        self.dirty_since
            .is_some_and(|dirty| now.saturating_sub(dirty) >= AUTOSAVE_DIRTY_INTERVAL)
            || self
                .last_mutation
                .is_some_and(|mutation| now.saturating_sub(mutation) >= AUTOSAVE_DEBOUNCE)
    }

    /// Clears dirty timing only after every atomic write succeeds.
    pub fn saved(&mut self) {
        self.dirty_since = None;
        self.last_mutation = None;
    }

    /// True when any persistent mutation remains unsaved.
    pub const fn is_dirty(self) -> bool {
        self.dirty_since.is_some()
    }
}

/// Exact persistence failure with the original file path.
#[derive(Debug, Error)]
pub enum WorldSaveError {
    /// Operating-system random seed generation failed.
    #[error("operating-system random seed generation failed: {0}")]
    Random(getrandom::Error),
    /// Filesystem operation failed.
    #[error("world file {path} could not be read or written: {source}", path = path.display())]
    Io {
        /// Exact affected path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// RON encoding failed.
    #[error("world file {path} could not be encoded: {source}", path = path.display())]
    Encode {
        /// Exact affected path.
        path: PathBuf,
        /// Encoding error.
        #[source]
        source: ron::Error,
    },
    /// RON decoding failed.
    #[error("world file {path} is corrupt: {source}", path = path.display())]
    Decode {
        /// Exact corrupt path.
        path: PathBuf,
        /// Decoding error.
        #[source]
        source: Box<ron::error::SpannedError>,
    },
    /// Edited-brick binary payload is corrupt.
    #[error("edited terrain file {path} is corrupt: {source}", path = path.display())]
    Brick {
        /// Exact corrupt path.
        path: PathBuf,
        /// Binary decoding failure.
        #[source]
        source: crate::BrickDecodeError,
    },
    /// World or generator version is not understood.
    #[error("world file {path} has an unsupported format or generator version", path = path.display())]
    UnsupportedVersion {
        /// Exact unsupported path.
        path: PathBuf,
    },
    /// Frozen state does not match the published creation or a valid target.
    #[error("world file {path} has invalid frozen creation state: {message}", path = path.display())]
    InvalidFrozenCreation {
        /// Exact affected manifest.
        path: PathBuf,
        /// Invalid frozen-state invariant.
        message: &'static str,
    },
    /// Deletion target was not a direct child of this store.
    #[error("refusing world path outside the store: {path}", path = path.display())]
    OutsideStore {
        /// Refused path.
        path: PathBuf,
    },
    /// Current-format corrupt data is preserved for explicit recovery.
    #[error("current world file {path} is corrupt and was left untouched: {message}", path = path.display())]
    CorruptCurrent {
        /// Exact corrupt file.
        path: PathBuf,
        /// Original inspection detail.
        message: String,
    },
}

fn validate_frozen_creation(world: &WorldDocument, path: &Path) -> Result<(), WorldSaveError> {
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

#[derive(Deserialize)]
struct ManifestHeader {
    version: u32,
    name: String,
    generator_version: WorldGeneratorVersion,
    seed: WorldSeed,
    last_played_unix_seconds: u64,
}

fn inspect_manifest(path: &Path) -> Result<ManifestHeader, WorldSaveError> {
    let text = read_exact(path)?;
    ron::from_str(&text).map_err(|source| WorldSaveError::Decode {
        path: path.to_owned(),
        source: Box::new(source),
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), WorldSaveError> {
    let parent = path
        .parent()
        .expect("every world file has a parent directory");
    fs::create_dir_all(parent).map_err(|source| WorldSaveError::Io {
        path: parent.to_owned(),
        source,
    })?;
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("save")
    ));
    fs::write(&temporary, bytes).map_err(|source| WorldSaveError::Io {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| WorldSaveError::Io {
        path: path.to_owned(),
        source,
    })
}

fn read_exact(path: &Path) -> Result<String, WorldSaveError> {
    fs::read_to_string(path).map_err(|source| WorldSaveError::Io {
        path: path.to_owned(),
        source,
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn data_root() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|home| Path::new(&home).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/share")))
    }
}

fn slug(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "world".to_owned()
    } else {
        slug.to_owned()
    }
}

#[cfg(test)]
mod tests;
