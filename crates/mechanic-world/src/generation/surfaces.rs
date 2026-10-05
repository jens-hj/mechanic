//! Surface palette and the per-biome rules that paint it onto the ground.

use std::cell::Cell;

use serde::{Deserialize, Serialize};

use super::WorldgenError;
use super::compile::{Scope, compile};
use super::flora::TreeTexture;
use super::spec::{Cond, Expr, SurfaceDoc, SurfaceRuleDoc, TextureSet};
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
    /// Index into [`SurfacePalette::tree_textures`] of the procedural
    /// texture drawn instead of `texture`, if any.
    pub tree_texture: Option<u16>,
}

/// Every surface a world can show, indexed by [`SurfaceId`].
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePalette {
    names: Vec<String>,
    looks: Vec<SurfaceLook>,
    tree_textures: Vec<TreeTexture>,
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
                tree_texture: None,
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
                tree_texture: None,
            });
        }
        if looks.len() > usize::from(u16::MAX) {
            return Err(WorldgenError::Invalid {
                context: "palette".to_owned(),
                message: "too many surfaces".to_owned(),
            });
        }
        Ok(Self {
            names,
            looks,
            tree_textures: Vec::new(),
        })
    }

    /// Adds `name`, a copy of the `base` look that draws a procedural tree
    /// texture instead of its texture set.
    pub(crate) fn add_tree_look(
        &mut self,
        name: String,
        base: SurfaceId,
        texture: TreeTexture,
    ) -> Result<SurfaceId, WorldgenError> {
        let invalid = |message: &str| WorldgenError::Invalid {
            context: format!("palette `{name}`"),
            message: message.to_owned(),
        };
        if self.names.contains(&name) {
            return Err(invalid("names a surface twice or shadows a material"));
        }
        let id = u16::try_from(self.looks.len())
            .ok()
            .filter(|&id| id < u16::MAX)
            .ok_or_else(|| invalid("too many surfaces"))?;
        let index = u16::try_from(self.tree_textures.len())
            .map_err(|_| invalid("too many tree textures"))?;
        let mut look = self.look(base);
        look.tree_texture = Some(index);
        self.names.push(name);
        self.looks.push(look);
        self.tree_textures.push(texture);
        Ok(SurfaceId(id))
    }

    /// Procedural tree textures the palette's looks draw, by index.
    pub fn tree_textures(&self) -> &[TreeTexture] {
        &self.tree_textures
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
    /// Signed depth of the estimated surface below connected established water.
    pub(crate) water_depth: Option<f64>,
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
    WaterDepth(f64, f64),
    Up(f64, f64),
    Altitude(f64, f64),
    /// A field, by index into the rules' distinct field tapes, within bounds.
    Field(usize, f64, f64),
    NearRiver(f64),
    /// Carve layer plus one, or zero for any layer.
    Carved(u8),
    All(Vec<Self>),
    Any(Vec<Self>),
    Not(Box<Self>),
}

/// Fields read while painting one point. Rules that read the same field
/// share one evaluation, which gives each the value it would compute.
struct FieldValues<'a> {
    tapes: &'a [Tape],
    values: [Cell<Option<f64>>; REMEMBERED_FIELDS],
}

/// Distinct fields per biome whose values a probe remembers; any beyond are
/// evaluated each time they are read.
const REMEMBERED_FIELDS: usize = 8;

impl<'a> FieldValues<'a> {
    fn new(tapes: &'a [Tape]) -> Self {
        Self {
            tapes,
            values: Default::default(),
        }
    }

    fn get(&self, index: usize, position: [f64; 3]) -> f64 {
        let Some(value) = self.values.get(index) else {
            return self.tapes[index].eval(position, &[]);
        };
        value.get().unwrap_or_else(|| {
            let fresh = self.tapes[index].eval(position, &[]);
            value.set(Some(fresh));
            fresh
        })
    }
}

impl CompiledCond {
    fn holds(&self, probe: &SurfaceProbe, fields: &FieldValues<'_>) -> bool {
        let within = |value: f64, lo: f64, hi: f64| lo <= value && value <= hi;
        match self {
            Self::Always => true,
            Self::Depth(lo, hi) => within(probe.depth, *lo, *hi),
            Self::WaterDepth(lo, hi) => probe
                .water_depth
                .is_some_and(|depth| within(depth, *lo, *hi)),
            Self::Up(lo, hi) => within(probe.up, *lo, *hi),
            Self::Altitude(lo, hi) => within(probe.position[1], *lo, *hi),
            Self::Field(index, lo, hi) => within(fields.get(*index, probe.position), *lo, *hi),
            Self::NearRiver(distance) => probe.river_distance <= *distance,
            Self::Carved(0) => probe.carved != 0,
            Self::Carved(layer) => probe.carved == *layer,
            Self::All(conds) => conds.iter().all(|cond| cond.holds(probe, fields)),
            Self::Any(conds) => conds.iter().any(|cond| cond.holds(probe, fields)),
            Self::Not(cond) => !cond.holds(probe, fields),
        }
    }
}

/// A biome's ordered surface rules.
#[derive(Debug)]
pub(crate) struct SurfaceRules {
    rules: Vec<(CompiledCond, SurfaceId, TerrainMaterial)>,
    /// Each distinct field the rules read, compiled once.
    fields: Vec<Tape>,
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
        let mut fields = Vec::new();
        for (index, doc) in docs.iter().enumerate() {
            let context = format!("{context} surface rule {index}");
            let surface = palette
                .id(&doc.surface)
                .ok_or_else(|| WorldgenError::Invalid {
                    context: context.clone(),
                    message: format!("unknown surface `{}`", doc.surface),
                })?;
            let cond = compile_cond(&doc.when, carves, scope, seed, &context, &mut fields)?;
            rules.push((cond, surface, palette.look(surface).material));
        }
        if !matches!(rules.last(), Some((CompiledCond::Always, ..))) {
            return Err(WorldgenError::Invalid {
                context: context.to_owned(),
                message: "the last surface rule must be `when: Always`".to_owned(),
            });
        }
        Ok(Self {
            rules,
            fields: fields.into_iter().map(|(_, tape)| tape).collect(),
        })
    }

    pub(crate) fn paint(&self, probe: &SurfaceProbe) -> (TerrainMaterial, SurfaceId) {
        let fields = FieldValues::new(&self.fields);
        self.rules
            .iter()
            .find(|(cond, ..)| cond.holds(probe, &fields))
            .map(|(_, surface, material)| (*material, *surface))
            .expect("the last rule always holds")
    }
}

/// Compiles a condition. Fields are compiled into `fields`, once for each
/// distinct expression.
fn compile_cond(
    cond: &Cond,
    carves: &[String],
    scope: Scope<'_>,
    seed: u64,
    context: &str,
    fields: &mut Vec<(Expr, Tape)>,
) -> Result<CompiledCond, WorldgenError> {
    let many = |conds: &[Cond], fields: &mut Vec<(Expr, Tape)>| {
        conds
            .iter()
            .map(|cond| compile_cond(cond, carves, scope, seed, context, fields))
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(match cond {
        Cond::Always => CompiledCond::Always,
        Cond::Depth(lo, hi) => CompiledCond::Depth(*lo, *hi),
        Cond::WaterDepth(lo, hi) => CompiledCond::WaterDepth(*lo, *hi),
        Cond::Up(lo, hi) => CompiledCond::Up(*lo, *hi),
        Cond::Ceiling => CompiledCond::Up(-1.0, -0.2),
        Cond::Altitude(lo, hi) => CompiledCond::Altitude(*lo, *hi),
        Cond::Field(expr, lo, hi) => {
            let index = if let Some(index) = fields.iter().position(|(seen, _)| seen == expr) {
                index
            } else {
                fields.push((expr.clone(), compile(expr, scope, &[], seed, context)?));
                fields.len() - 1
            };
            CompiledCond::Field(index, *lo, *hi)
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
        Cond::All(conds) => CompiledCond::All(many(conds, fields)?),
        Cond::Any(conds) => CompiledCond::Any(many(conds, fields)?),
        Cond::Not(cond) => CompiledCond::Not(Box::new(compile_cond(
            cond, carves, scope, seed, context, fields,
        )?)),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        CompiledCond, FieldValues, SurfaceId, SurfacePalette, SurfaceProbe, SurfaceRules,
        parse_tint,
    };
    use crate::TerrainMaterial;
    use crate::generation::compile::{Scope, compile};
    use crate::generation::load::parse;
    use crate::generation::spec::{Expr, SurfaceRuleDoc};

    #[test]
    fn water_depth_matches_beds_and_shores_but_not_dry_or_absent_water() {
        let mut probe = SurfaceProbe {
            position: [0.0; 3],
            depth: 0.0,
            water_depth: None,
            up: 1.0,
            river_distance: f64::INFINITY,
            carved: 0,
        };
        let condition = CompiledCond::WaterDepth(-0.35, 3.0);
        for (depth, expected) in [
            (None, false),
            (Some(-0.36), false),
            (Some(-0.35), true),
            (Some(0.0), true),
            (Some(2.0), true),
            (Some(3.0), true),
            (Some(3.01), false),
        ] {
            probe.water_depth = depth;
            assert_eq!(
                condition.holds(&probe, &FieldValues::new(&[])),
                expected,
                "{depth:?}"
            );
        }
    }

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

    #[test]
    fn rules_reading_one_field_share_it_and_paint_as_if_each_read_it() {
        let docs: Vec<SurfaceRuleDoc> = parse(
            "rules",
            r#"[
                (when: All([Field(Noise(freq: 0.1, seed: 4), 0.2, 1), Up(0.9, 1)]), use: "rock"),
                (when: Not(Field(Noise(freq: 0.1, seed: 4), -1, 0.2)), use: "soil"),
                (when: Field(Noise(freq: 0.1, seed: 5), 0, 1), use: "sand"),
                (when: Always, use: "iron"),
            ]"#,
        )
        .expect("valid rules");
        let empty = BTreeMap::new();
        let scope = Scope {
            local: &empty,
            library: &empty,
            fields: None,
        };
        let palette = SurfacePalette::new(&[]).unwrap();
        let rules = SurfaceRules::new(&docs, &palette, &[], scope, 3, "test").unwrap();
        assert_eq!(rules.fields.len(), 2, "the repeated field is compiled once");
        let field = |text: &str| {
            let expr: Expr = parse("field", text).unwrap();
            compile(&expr, scope, &[], 3, "test").unwrap()
        };
        let (first, second) = (
            field("Noise(freq: 0.1, seed: 4)"),
            field("Noise(freq: 0.1, seed: 5)"),
        );
        let mut painted = [0; 4];
        for step in 0..400_u32 {
            let t = f64::from(step);
            let probe = SurfaceProbe {
                position: [(t * 0.37).sin() * 40.0, t * 0.05, (t * 0.11).cos() * 40.0],
                depth: 0.0,
                water_depth: None,
                up: if step % 3 == 0 { 0.5 } else { 1.0 },
                river_distance: f64::INFINITY,
                carved: 0,
            };
            let a = first.eval(probe.position, &[]);
            let b = second.eval(probe.position, &[]);
            let expected = if (0.2..=1.0).contains(&a) && probe.up >= 0.9 {
                0
            } else if !(-1.0..=0.2).contains(&a) {
                1
            } else if (0.0..=1.0).contains(&b) {
                2
            } else {
                3
            };
            let names = ["rock", "soil", "sand", "iron"];
            assert_eq!(rules.paint(&probe).1, palette.id(names[expected]).unwrap());
            painted[expected] += 1;
        }
        assert!(painted.iter().all(|&count| count > 0), "{painted:?}");
    }
}
