//! Species genomes as authored in `flora.ron`, and their validation.

use serde::Deserialize;

use super::super::load::parse;
use crate::WorldgenError;

/// The flora library file: every species a world may grow.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename = "Flora")]
struct FloraLibraryDoc {
    species: Vec<SpeciesDoc>,
}

/// One species as written in `flora.ron`. See [`SpeciesSpec`] for meanings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename = "Species")]
struct SpeciesDoc {
    name: String,
    height: (f64, f64),
    width: f64,
    girth: f64,
    stems: (u32, u32),
    crown_base: f64,
    dominance: f64,
    split_chance: f64,
    split_count: (u32, u32),
    /// Degrees.
    split_angle: f64,
    tropism: f64,
    wobble: f64,
    foliage: FoliageDoc,
    bark: String,
    roots: RootsDoc,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct FoliageDoc {
    look: String,
    size: f64,
    density: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
struct RootsDoc {
    spread: f64,
    depth: f64,
}

/// A validated tree genome. Every tree of a species is grown from these
/// numbers and a seed.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesSpec {
    /// Unique identifier.
    pub name: String,
    /// Range of mature heights in metres; each tree draws one.
    pub height: (f64, f64),
    /// Crown diameter as a fraction of height.
    pub width: f64,
    /// Trunk base diameter as a fraction of height, shared among the stems.
    pub girth: f64,
    /// Range of stem counts rising from the base.
    pub stems: (u32, u32),
    /// Fraction of a stem's height that is bare before the first split.
    pub crown_base: f64,
    /// Apical dominance in `[0, 1]`: the chance the leader survives a split,
    /// and the crown envelope's blend from dome (0) to cone (1).
    pub dominance: f64,
    /// Chance that a node splits.
    pub split_chance: f64,
    /// Range of children per split.
    pub split_count: (u32, u32),
    /// Angle of a child from its parent, in radians.
    pub split_angle: f64,
    /// Bend per segment in `[-1, 1]`: positive seeks the sun, negative hangs.
    /// Thin, flexible branches respond most.
    pub tropism: f64,
    /// Random kink per segment in `[0, 1]`.
    pub wobble: f64,
    /// How foliage clothes the terminal twigs.
    pub foliage: FoliageSpec,
    /// Surface look of the wood.
    pub bark: String,
    /// How the roots spread.
    pub roots: RootsSpec,
}

/// Foliage around terminal twigs.
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageSpec {
    /// Surface look of the leaves or needles.
    pub look: String,
    /// Radius of the sleeve around each terminal segment, in metres.
    pub size: f64,
    /// Fraction of the sleeve that is filled, in `[0, 1]`.
    pub density: f64,
}

/// The root system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootsSpec {
    /// Root radius as a fraction of crown radius.
    pub spread: f64,
    /// Depth of the main roots in metres.
    pub depth: f64,
}

impl SpeciesSpec {
    /// Parses and validates a flora library.
    ///
    /// # Errors
    ///
    /// Returns a parse error, or the first species whose genome is out of range
    /// or whose name repeats.
    pub fn library(text: &str) -> Result<Vec<Self>, WorldgenError> {
        let doc: FloraLibraryDoc = parse("flora.ron", text)?;
        let mut species = Vec::with_capacity(doc.species.len());
        for entry in doc.species {
            let spec = Self::validate(entry)?;
            if species.iter().any(|known: &Self| known.name == spec.name) {
                return Err(invalid(&spec.name, "declared twice"));
            }
            species.push(spec);
        }
        Ok(species)
    }

    /// The flora library compiled into this build.
    pub fn embedded() -> Vec<Self> {
        crate::WorldgenSpec::embedded().species().to_vec()
    }

    fn validate(doc: SpeciesDoc) -> Result<Self, WorldgenError> {
        let name = doc.name;
        let check = |ok: bool, message: &str| {
            if ok {
                Ok(())
            } else {
                Err(invalid(&name, message))
            }
        };
        check(!name.is_empty(), "needs a name")?;
        check(
            doc.height.0 > 0.0 && doc.height.0 <= doc.height.1,
            "height must be positive and ordered",
        )?;
        check(doc.width > 0.0, "width must be positive")?;
        check(doc.girth > 0.0, "girth must be positive")?;
        check(
            doc.stems.0 >= 1 && doc.stems.0 <= doc.stems.1,
            "stems must be at least 1 and ordered",
        )?;
        check(
            doc.split_count.0 >= 1 && doc.split_count.0 <= doc.split_count.1,
            "split_count must be at least 1 and ordered",
        )?;
        for (value, field) in [
            (doc.crown_base, "crown_base"),
            (doc.dominance, "dominance"),
            (doc.split_chance, "split_chance"),
            (doc.wobble, "wobble"),
            (doc.foliage.density, "foliage density"),
        ] {
            check(
                (0.0..=1.0).contains(&value),
                &format!("{field} must lie in [0, 1]"),
            )?;
        }
        check(
            (-1.0..=1.0).contains(&doc.tropism),
            "tropism must lie in [-1, 1]",
        )?;
        check(
            (0.0..=180.0).contains(&doc.split_angle),
            "split_angle must lie in [0, 180] degrees",
        )?;
        check(doc.foliage.size >= 0.0, "foliage size must not be negative")?;
        check(
            doc.roots.spread >= 0.0 && doc.roots.depth >= 0.0,
            "roots must not be negative",
        )?;
        check(
            !doc.bark.is_empty() && !doc.foliage.look.is_empty(),
            "bark and foliage need looks",
        )?;
        Ok(Self {
            name,
            height: doc.height,
            width: doc.width,
            girth: doc.girth,
            stems: doc.stems,
            crown_base: doc.crown_base,
            dominance: doc.dominance,
            split_chance: doc.split_chance,
            split_count: doc.split_count,
            split_angle: doc.split_angle.to_radians(),
            tropism: doc.tropism,
            wobble: doc.wobble,
            foliage: FoliageSpec {
                look: doc.foliage.look,
                size: doc.foliage.size,
                density: doc.foliage.density,
            },
            bark: doc.bark,
            roots: RootsSpec {
                spread: doc.roots.spread,
                depth: doc.roots.depth,
            },
        })
    }
}

fn invalid(name: &str, message: &str) -> WorldgenError {
    WorldgenError::Invalid {
        context: format!("flora.ron species `{name}`"),
        message: message.to_owned(),
    }
}
