//! Reads world-generation documents from the embedded set or a directory.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use thiserror::Error;

use super::spec::{BiomeDoc, LibraryDoc, WorldDoc};

/// Why a world-generation definition could not be used.
#[derive(Debug, Error)]
pub enum WorldgenError {
    /// A file could not be read.
    #[error("cannot read worldgen file `{file}`: {message}")]
    Missing {
        /// Relative path.
        file: String,
        /// Underlying reason.
        message: String,
    },
    /// A file is not valid RON for its document type.
    #[error("{file}:{message}")]
    Parse {
        /// Relative path.
        file: String,
        /// RON's position and message.
        message: String,
    },
    /// The documents parse but do not describe a usable world.
    #[error("{context}: {message}")]
    Invalid {
        /// Which biome, rule, or field.
        context: String,
        /// What is wrong.
        message: String,
    },
}

/// Parsed, cross-checked world-generation input.
#[derive(Clone, Debug)]
pub struct WorldgenSpec {
    pub(crate) world: WorldDoc,
    pub(crate) library: LibraryDoc,
    pub(crate) biomes: Vec<BiomeDoc>,
    hash: u64,
}

const EMBEDDED: &[(&str, &str)] = &[
    ("world.ron", include_str!("../../worldgen/world.ron")),
    ("library.ron", include_str!("../../worldgen/library.ron")),
    (
        "biomes/verdant_hills.ron",
        include_str!("../../worldgen/biomes/verdant_hills.ron"),
    ),
    (
        "biomes/dune_sea.ron",
        include_str!("../../worldgen/biomes/dune_sea.ron"),
    ),
    (
        "biomes/arch_steppe.ron",
        include_str!("../../worldgen/biomes/arch_steppe.ron"),
    ),
    (
        "biomes/titan_crags.ron",
        include_str!("../../worldgen/biomes/titan_crags.ron"),
    ),
    (
        "biomes/karst_needles.ron",
        include_str!("../../worldgen/biomes/karst_needles.ron"),
    ),
    (
        "biomes/gyroid_reef.ron",
        include_str!("../../worldgen/biomes/gyroid_reef.ron"),
    ),
    (
        "biomes/drift_isles.ron",
        include_str!("../../worldgen/biomes/drift_isles.ron"),
    ),
    (
        "biomes/shelf_mire.ron",
        include_str!("../../worldgen/biomes/shelf_mire.ron"),
    ),
    (
        "biomes/sunken_coast.ron",
        include_str!("../../worldgen/biomes/sunken_coast.ron"),
    ),
];

impl WorldgenSpec {
    /// The definition compiled into this build. Parsed once per process.
    ///
    /// # Panics
    ///
    /// Panics if the embedded files are invalid; tests keep them valid.
    pub fn embedded() -> Arc<Self> {
        static SPEC: OnceLock<Arc<WorldgenSpec>> = OnceLock::new();
        Arc::clone(SPEC.get_or_init(|| {
            Arc::new(
                Self::load(|file| {
                    EMBEDDED
                        .iter()
                        .find(|(name, _)| *name == file)
                        .map(|(_, text)| (*text).to_owned())
                        .ok_or_else(|| format!("`{file}` is not embedded"))
                })
                .unwrap_or_else(|error| panic!("embedded worldgen is invalid: {error}")),
            )
        }))
    }

    /// Reads `world.ron`, `library.ron`, and `biomes/<name>.ron` from `root`.
    ///
    /// # Errors
    ///
    /// Returns the first unreadable, malformed, or inconsistent document.
    pub fn from_dir(root: &Path) -> Result<Self, WorldgenError> {
        Self::load(|file| {
            std::fs::read_to_string(root.join(file)).map_err(|error| error.to_string())
        })
    }

    fn load(read: impl Fn(&str) -> Result<String, String>) -> Result<Self, WorldgenError> {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        let mut text = |file: &str| {
            let contents = read(file).map_err(|message| WorldgenError::Missing {
                file: file.to_owned(),
                message,
            })?;
            for byte in file.bytes().chain(contents.bytes()) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
            Ok::<_, WorldgenError>(contents)
        };
        let world: WorldDoc = parse("world.ron", &text("world.ron")?)?;
        let library: LibraryDoc = parse("library.ron", &text("library.ron")?)?;
        let mut biomes = Vec::with_capacity(world.biomes.len());
        for name in &world.biomes {
            let file = format!("biomes/{name}.ron");
            let biome: BiomeDoc = parse(&file, &text(&file)?)?;
            if &biome.name != name {
                return Err(WorldgenError::Invalid {
                    context: file,
                    message: format!("declares name `{}`, expected `{name}`", biome.name),
                });
            }
            biomes.push(biome);
        }
        if biomes.is_empty() {
            return Err(WorldgenError::Invalid {
                context: "world.ron".to_owned(),
                message: "lists no biomes".to_owned(),
            });
        }
        if !biomes.iter().any(|biome| biome.name == world.spawn.biome) {
            return Err(WorldgenError::Invalid {
                context: "world.ron spawn".to_owned(),
                message: format!("unknown biome `{}`", world.spawn.biome),
            });
        }
        if world.vertical.0 >= world.vertical.1 {
            return Err(WorldgenError::Invalid {
                context: "world.ron vertical".to_owned(),
                message: "minimum must lie below maximum".to_owned(),
            });
        }
        Ok(Self {
            world,
            library,
            biomes,
            hash,
        })
    }

    /// Stable digest of every source file; a changed definition makes older
    /// saves outdated.
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Biome names in declaration order.
    pub fn biome_names(&self) -> impl Iterator<Item = &str> {
        self.biomes.iter().map(|biome| biome.name.as_str())
    }
}

fn parse<T: serde::de::DeserializeOwned>(file: &str, text: &str) -> Result<T, WorldgenError> {
    ron::Options::default()
        .with_default_extension(
            ron::extensions::Extensions::IMPLICIT_SOME
                | ron::extensions::Extensions::UNWRAP_VARIANT_NEWTYPES,
        )
        .from_str(text)
        .map_err(|error| WorldgenError::Parse {
            file: file.to_owned(),
            message: error.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::{EMBEDDED, WorldgenSpec};

    #[test]
    fn every_listed_biome_is_embedded() {
        let spec = WorldgenSpec::embedded();
        for name in spec.biome_names() {
            let file = format!("biomes/{name}.ron");
            assert!(
                EMBEDDED.iter().any(|(embedded, _)| *embedded == file),
                "{file}"
            );
        }
        assert_eq!(EMBEDDED.len(), spec.biomes.len() + 2);
    }

    #[test]
    fn parse_errors_name_the_file_and_position() {
        let error = WorldgenSpec::load(|file| {
            Ok(if file == "world.ron" {
                "World(vertical: (0, 1), sea_level: nope)".to_owned()
            } else {
                String::new()
            })
        })
        .unwrap_err()
        .to_string();
        assert!(error.starts_with("world.ron:1:"), "{error}");
    }
}
