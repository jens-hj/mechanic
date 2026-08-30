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
pub const WORLD_FORMAT_VERSION: u32 = 3;
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
pub enum OpenWorldResult {
    /// Current world ready to play.
    Opened(WorldDocument),
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
        world.construction_generation = generation;
        if let Err(error) = self.save_world(world) {
            world.construction_generation = previous;
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
    pub fn open_entry(&self, entry: &SavedWorld) -> Result<OpenWorldResult, WorldSaveError> {
        match &entry.status {
            SavedWorldStatus::Current => self.load_world(&entry.path).map(OpenWorldResult::Opened),
            SavedWorldStatus::Outdated => {
                self.delete_world(&entry.path)?;
                Ok(OpenWorldResult::OutdatedRemoved {
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
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU32, Ordering},
        time::Duration,
    };

    use mechanic_core::{CREATION_FORMAT_VERSION, CreationDocument};

    use super::{
        AutosaveState, OpenWorldResult, SavedWorldStatus, WORLD_FORMAT_VERSION,
        WorldCreationInstanceDoc, WorldDocument, WorldPoseDoc, WorldSaveError, WorldStore,
    };
    use crate::{TerrainField, TerrainOctree, WorldPosition, WorldSeed};

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            Self(std::env::temp_dir().join(format!(
                "mechanic-world-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn empty_creation() -> CreationDocument {
        CreationDocument {
            version: CREATION_FORMAT_VERSION,
            name: "Anchor".to_owned(),
            parts: Vec::new(),
            welds: Vec::new(),
            rigid_links: Vec::new(),
            bearings: Vec::new(),
            drive_links: Vec::new(),
            input_seat_links: Vec::new(),
            seat_controller_links: Vec::new(),
            gearbox_configs: Vec::new(),
            regions: Vec::new(),
            sockets: Vec::new(),
        }
    }

    #[test]
    fn world_instance_and_brick_round_trip() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let field = TerrainField::new(WorldSeed(84));
        let world = WorldDocument::new("Violet Reach", field.seed(), field.safe_spawn());
        let world_path = store.save_world(&world).unwrap();
        assert_eq!(
            store.load_world(world_path.parent().unwrap()).unwrap(),
            world
        );

        let instance = WorldCreationInstanceDoc {
            id: 17,
            creation: empty_creation(),
            root_pose: WorldPoseDoc::default(),
            joint_coordinates: vec![0.25, -0.5],
        };
        let instance_path = store.save_instance(&world.name, &instance).unwrap();
        assert_eq!(store.load_instance(&instance_path).unwrap(), instance);

        let mut edits = TerrainOctree::default();
        edits.promote(&field, crate::BrickCoord::new(0, 0, 0));
        let brick = edits.brick(crate::BrickCoord::new(0, 0, 0)).unwrap();
        let brick_path = store.save_brick(&world.name, brick).unwrap();
        assert_eq!(store.load_brick(&brick_path).unwrap(), *brick);
    }

    #[test]
    fn paired_spaces_publish_and_reload_one_generation() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let field = TerrainField::new(WorldSeed(91));
        let mut world = WorldDocument::new("Paired", field.seed(), field.safe_spawn());
        let world_space = WorldCreationInstanceDoc {
            id: 1,
            creation: empty_creation(),
            root_pose: WorldPoseDoc::default(),
            joint_coordinates: Vec::new(),
        };
        let garage_space = WorldCreationInstanceDoc {
            id: 2,
            creation: CreationDocument {
                name: "Garage".to_owned(),
                ..empty_creation()
            },
            root_pose: WorldPoseDoc::default(),
            joint_coordinates: Vec::new(),
        };

        store
            .save_space_pair(&mut world, &world_space, &garage_space)
            .unwrap();
        assert_eq!(world.construction_generation, 1);
        let published = store.load_world(&store.directory_for("Paired")).unwrap();
        assert_eq!(published.construction_generation, 1);
        let interrupted = store.directory_for("Paired").join("generations").join("2");
        fs::create_dir_all(&interrupted).unwrap();
        fs::write(interrupted.join("world.ron"), b"incomplete generation").unwrap();
        assert_eq!(
            store.load_space_pair(&published).unwrap(),
            Some((world_space, garage_space))
        );
    }

    #[test]
    fn corrupt_brick_reports_exact_file_and_preserves_it() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let path = temporary.0.join("broken.bin");
        fs::create_dir_all(&temporary.0).unwrap();
        fs::write(&path, b"original recovery data").unwrap();
        let error = store.load_brick(&path).unwrap_err();
        assert!(matches!(error, WorldSaveError::Brick { path: ref failed, .. } if failed == &path));
        assert_eq!(fs::read(&path).unwrap(), b"original recovery data");
    }

    #[test]
    fn autosave_debounces_but_never_waits_more_than_thirty_seconds() {
        let mut autosave = AutosaveState::default();
        autosave.mutate(Duration::ZERO);
        autosave.mutate(Duration::from_secs(1));
        assert!(!autosave.due(Duration::from_millis(2_999)));
        assert!(autosave.due(Duration::from_secs(3)));
        autosave.mutate(Duration::from_secs(29));
        assert!(autosave.due(Duration::from_secs(30)));
        autosave.saved();
        assert!(!autosave.is_dirty());
    }

    #[test]
    fn list_classifies_version_two_outdated_and_corrupt_manifests() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let field = TerrainField::new(WorldSeed(11));
        let current = WorldDocument::new("Current", field.seed(), field.safe_spawn());
        store.save_world(&current).unwrap();

        let outdated_path = store.directory_for("Outdated").join("world.ron");
        fs::create_dir_all(outdated_path.parent().unwrap()).unwrap();
        let mut outdated = WorldDocument::new("Outdated", field.seed(), field.safe_spawn());
        outdated.version = WORLD_FORMAT_VERSION - 1;
        fs::write(&outdated_path, ron::to_string(&outdated).unwrap()).unwrap();

        let corrupt_path = store.directory_for("Corrupt").join("world.ron");
        fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
        fs::write(
            &corrupt_path,
            format!(
                "(version:{WORLD_FORMAT_VERSION},name:\"Corrupt\",generator_version:1,seed:11,last_played_unix_seconds:3,player_pose:broken)"
            ),
        )
        .unwrap();

        let listed = store.list();
        assert_eq!(listed.len(), 3);
        assert!(
            listed
                .iter()
                .any(|world| world.status == SavedWorldStatus::Current)
        );
        assert!(
            listed
                .iter()
                .any(|world| world.status == SavedWorldStatus::Outdated)
        );
        assert!(listed.iter().any(|world| matches!(
            world.status,
            SavedWorldStatus::Corrupt { ref file, .. } if file == &corrupt_path
        )));
    }

    #[test]
    fn opening_outdated_deletes_only_exact_child_and_corrupt_current_is_preserved() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let field = TerrainField::new(WorldSeed(12));
        let mut outdated = WorldDocument::new("Old", field.seed(), field.safe_spawn());
        outdated.version = WORLD_FORMAT_VERSION - 1;
        let directory = store.directory_for(&outdated.name);
        let manifest = directory.join("world.ron");
        fs::create_dir_all(&directory).unwrap();
        fs::write(&manifest, ron::to_string(&outdated).unwrap()).unwrap();
        let entry = store
            .list()
            .into_iter()
            .find(|entry| entry.path == directory)
            .unwrap();
        assert_eq!(
            store.open_entry(&entry).unwrap(),
            OpenWorldResult::OutdatedRemoved {
                path: directory.clone()
            }
        );
        assert!(!directory.exists());

        let corrupt_directory = store.directory_for("Current corrupt");
        let corrupt_manifest = corrupt_directory.join("world.ron");
        fs::create_dir_all(&corrupt_directory).unwrap();
        let bytes = format!(
            "(version:{WORLD_FORMAT_VERSION},name:\"Current corrupt\",generator_version:1,seed:12,last_played_unix_seconds:4,player_pose:broken)"
        );
        fs::write(&corrupt_manifest, &bytes).unwrap();
        let corrupt = store
            .list()
            .into_iter()
            .find(|entry| entry.path == corrupt_directory)
            .unwrap();
        assert!(matches!(
            store.open_entry(&corrupt),
            Err(WorldSaveError::CorruptCurrent { ref path, .. }) if path == &corrupt_manifest
        ));
        assert_eq!(fs::read_to_string(corrupt_manifest).unwrap(), bytes);
    }

    #[test]
    fn dirty_leaf_save_and_hierarchy_reconstruction_are_deterministic() {
        let temporary = TempDir::new();
        let store = WorldStore::new(&temporary.0);
        let field = TerrainField::new(WorldSeed(13));
        let world = WorldDocument::new("Octree", field.seed(), field.safe_spawn());
        store.save_world(&world).unwrap();
        let mut terrain = TerrainOctree::default();
        terrain
            .excavate_sphere(
                &field,
                WorldPosition(bevy_math::DVec3::new(
                    0.0,
                    field.surface_height(0.0, 0.0),
                    0.0,
                )),
                0.3,
            )
            .unwrap();
        let saved = store.save_dirty_leaves(&world.name, &mut terrain).unwrap();
        assert!(!saved.is_empty());
        assert_eq!(terrain.dirty_leaves().count(), 0);
        let rebuilt = store.load_octree(&world.name).unwrap();
        assert_eq!(
            rebuilt.node(crate::TerrainNodeId::ROOT),
            terrain.node(crate::TerrainNodeId::ROOT)
        );
        assert_eq!(
            rebuilt.brick_coordinates().collect::<Vec<_>>(),
            terrain.brick_coordinates().collect::<Vec<_>>()
        );
    }
}
