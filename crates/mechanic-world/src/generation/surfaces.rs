//! Surface palette and the per-biome rules that paint it onto the ground.

use serde::{Deserialize, Serialize};

use super::WorldgenError;
use super::compile::{Scope, compile};
use super::spec::{Cond, SurfaceDoc, SurfaceRuleDoc, TextureSet};
use super::tape::Tape;
use crate::TerrainMaterial;

/// Index into the world's surface palette: what a cell looks like.
///
/// The first [`TerrainMaterial::COUNT`] ids are the plain, untinted look of
/// each material in `code()` order, so edits and spoil can name a surface
/// without consulting the world.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SurfaceId(pub u16);

impl SurfaceId {
    /// The plain look of a material.
    pub const fn plain(material: TerrainMaterial) -> Self {
        Self(material.code() as u16)
    }
}

/// How one palette entry renders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceLook {
    /// Physical behaviour.
    pub material: TerrainMaterial,
    /// Texture family.
    pub texture: TextureSet,
    /// Linear-light RGB tint.
    pub tint: [f32; 3],
    /// Tint only through the texture's tint mask.
    pub masked: bool,
    /// Replace hue instead of multiplying.
    pub recolor: bool,
    /// Roughness multiplier.
    pub roughness: f32,
    /// Texture repeat multiplier.
    pub scale: f32,
}

/// Every surface a world can show, indexed by [`SurfaceId`].
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePalette {
    names: Vec<String>,
    looks: Vec<SurfaceLook>,
}

impl SurfacePalette {
    pub(crate) fn new(docs: &[SurfaceDoc]) -> Result<Self, WorldgenError> {
        let mut names = Vec::new();
        let mut looks = Vec::new();
        for material in TerrainMaterial::BY_CODE {
            names.push(material.name().to_owned());
            looks.push(SurfaceLook {
                material,
                texture: material.plain_texture(),
                tint: [1.0; 3],
                masked: true,
                recolor: false,
                roughness: 1.0,
                scale: 1.0,
            });
        }
        for doc in docs {
            if names.contains(&doc.name) {
                return Err(WorldgenError::Invalid {
                    context: format!("palette `{}`", doc.name),
                    message: "names a surface twice or shadows a material".to_owned(),
                });
            }
            names.push(doc.name.clone());
            looks.push(SurfaceLook {
                material: doc.material,
                texture: doc.texture,
                tint: parse_tint(&doc.tint).ok_or_else(|| WorldgenError::Invalid {
                    context: format!("palette `{}`", doc.name),
                    message: format!("tint `{}` is not #rrggbb", doc.tint),
                })?,
                masked: doc.masked,
                recolor: doc.recolor,
                #[expect(clippy::cast_possible_truncation, reason = "shader parameters are f32")]
                roughness: doc.roughness as f32,
                #[expect(clippy::cast_possible_truncation, reason = "shader parameters are f32")]
                scale: doc.scale as f32,
            });
        }
        if looks.len() > usize::from(u16::MAX) {
            return Err(WorldgenError::Invalid {
                context: "palette".to_owned(),
                message: "too many surfaces".to_owned(),
            });
        }
        Ok(Self { names, looks })
    }

    /// Look of a surface; unknown ids fall back to plain rock.
    pub fn look(&self, surface: SurfaceId) -> SurfaceLook {
        self.looks
            .get(usize::from(surface.0))
            .copied()
            .unwrap_or(self.looks[usize::from(SurfaceId::plain(TerrainMaterial::Rock).0)])
    }

    /// Every look in id order.
    pub fn looks(&self) -> &[SurfaceLook] {
        &self.looks
    }

    /// Id of a named surface.
    ///
    /// # Panics
    ///
    /// Never: the palette's size is checked when it is built.
    pub fn id(&self, name: &str) -> Option<SurfaceId> {
        self.names
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| SurfaceId(u16::try_from(index).expect("palette size is checked")))
    }
}

/// sRGB `#rrggbb` to linear RGB.
fn parse_tint(text: &str) -> Option<[f32; 3]> {
    let hex = text.strip_prefix('#')?;
    if hex.len() != 6 {
        return None;
    }
    let channel = |index: usize| {
        let value = f32::from(u8::from_str_radix(&hex[index..index + 2], 16).ok()?) / 255.0;
        Some(if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        })
    };
    Some([channel(0)?, channel(2)?, channel(4)?])
}

/// What a rule can see about the point being painted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceProbe {
    pub(crate) position: [f64; 3],
    /// Metres below the surface, zero at and above it.
    pub(crate) depth: f64,
    /// Outward normal's y component.
    pub(crate) up: f64,
    /// Horizontal distance to the nearest river centre line.
    pub(crate) river_distance: f64,
    /// Carve layer whose void shapes the point, plus one; zero for none.
    pub(crate) carved: u8,
}

#[derive(Debug)]
enum CompiledCond {
    Always,
    Depth(f64, f64),
    Up(f64, f64),
    Altitude(f64, f64),
    Field(Tape, f64, f64),
    NearRiver(f64),
    /// Carve layer plus one, or zero for any layer.
    Carved(u8),
    All(Vec<Self>),
    Any(Vec<Self>),
    Not(Box<Self>),
}

impl CompiledCond {
    fn holds(&self, probe: &SurfaceProbe) -> bool {
        let within = |value: f64, lo: f64, hi: f64| lo <= value && value <= hi;
        match self {
            Self::Always => true,
            Self::Depth(lo, hi) => within(probe.depth, *lo, *hi),
            Self::Up(lo, hi) => within(probe.up, *lo, *hi),
            Self::Altitude(lo, hi) => within(probe.position[1], *lo, *hi),
            Self::Field(tape, lo, hi) => within(tape.eval(probe.position, &[]), *lo, *hi),
            Self::NearRiver(distance) => probe.river_distance <= *distance,
            Self::Carved(0) => probe.carved != 0,
            Self::Carved(layer) => probe.carved == *layer,
            Self::All(conds) => conds.iter().all(|cond| cond.holds(probe)),
            Self::Any(conds) => conds.iter().any(|cond| cond.holds(probe)),
            Self::Not(cond) => !cond.holds(probe),
        }
    }
}

/// A biome's ordered surface rules.
#[derive(Debug)]
pub(crate) struct SurfaceRules {
    rules: Vec<(CompiledCond, SurfaceId, TerrainMaterial)>,
}

impl SurfaceRules {
    pub(crate) fn new(
        docs: &[SurfaceRuleDoc],
        palette: &SurfacePalette,
        carves: &[String],
        scope: Scope<'_>,
        seed: u64,
        context: &str,
    ) -> Result<Self, WorldgenError> {
        let mut rules = Vec::with_capacity(docs.len());
        for (index, doc) in docs.iter().enumerate() {
            let context = format!("{context} surface rule {index}");
            let surface = palette
                .id(&doc.surface)
                .ok_or_else(|| WorldgenError::Invalid {
                    context: context.clone(),
                    message: format!("unknown surface `{}`", doc.surface),
                })?;
            let cond = compile_cond(&doc.when, carves, scope, seed, &context)?;
            rules.push((cond, surface, palette.look(surface).material));
        }
        if !matches!(rules.last(), Some((CompiledCond::Always, ..))) {
            return Err(WorldgenError::Invalid {
                context: context.to_owned(),
                message: "the last surface rule must be `when: Always`".to_owned(),
            });
        }
        Ok(Self { rules })
    }

    pub(crate) fn paint(&self, probe: &SurfaceProbe) -> (TerrainMaterial, SurfaceId) {
        self.rules
            .iter()
            .find(|(cond, ..)| cond.holds(probe))
            .map(|(_, surface, material)| (*material, *surface))
            .expect("the last rule always holds")
    }
}

fn compile_cond(
    cond: &Cond,
    carves: &[String],
    scope: Scope<'_>,
    seed: u64,
    context: &str,
) -> Result<CompiledCond, WorldgenError> {
    let many = |conds: &[Cond]| {
        conds
            .iter()
            .map(|cond| compile_cond(cond, carves, scope, seed, context))
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(match cond {
        Cond::Always => CompiledCond::Always,
        Cond::Depth(lo, hi) => CompiledCond::Depth(*lo, *hi),
        Cond::Up(lo, hi) => CompiledCond::Up(*lo, *hi),
        Cond::Ceiling => CompiledCond::Up(-1.0, -0.2),
        Cond::Altitude(lo, hi) => CompiledCond::Altitude(*lo, *hi),
        Cond::Field(expr, lo, hi) => {
            CompiledCond::Field(compile(expr, scope, &[], seed, context)?, *lo, *hi)
        }
        Cond::NearRiver(distance) => CompiledCond::NearRiver(*distance),
        Cond::Carved(None) => CompiledCond::Carved(0),
        Cond::Carved(Some(name)) => {
            let layer = carves
                .iter()
                .position(|carve| carve == name)
                .ok_or_else(|| WorldgenError::Invalid {
                    context: context.to_owned(),
                    message: format!("unknown carve layer `{name}`"),
                })?;
            CompiledCond::Carved(u8::try_from(layer + 1).expect("carve layers are capped"))
        }
        Cond::All(conds) => CompiledCond::All(many(conds)?),
        Cond::Any(conds) => CompiledCond::Any(many(conds)?),
        Cond::Not(cond) => {
            CompiledCond::Not(Box::new(compile_cond(cond, carves, scope, seed, context)?))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{SurfaceId, SurfacePalette, parse_tint};
    use crate::TerrainMaterial;

    #[test]
    fn plain_surfaces_follow_material_codes() {
        let palette = SurfacePalette::new(&[]).unwrap();
        for material in TerrainMaterial::ALL {
            assert_eq!(palette.look(SurfaceId::plain(material)).material, material);
            assert_eq!(
                palette.id(material.name()),
                Some(SurfaceId::plain(material))
            );
        }
    }

    #[test]
    fn tints_are_linearised_srgb() {
        assert_eq!(parse_tint("#ffffff"), Some([1.0; 3]));
        let mid = parse_tint("#808080").unwrap()[0];
        assert!((mid - 0.2158).abs() < 1.0e-3);
        assert_eq!(parse_tint("fff"), None);
    }
}
