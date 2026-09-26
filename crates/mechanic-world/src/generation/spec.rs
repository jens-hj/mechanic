//! Declarative world-generation documents as authored in RON.
//!
//! Density is positive inside solid ground and roughly measures metres to the
//! surface, so a heightfield is `Height(h)` = `h - y` and an SDF primitive is
//! its negated distance. Union is therefore `Max`, intersection `Min`.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::TerrainMaterial;

/// One scalar expression over the current domain `(x, y, z)` in metres.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub enum Expr {
    /// Current domain x.
    X,
    /// Current domain y.
    Y,
    /// Current domain z.
    Z,
    /// A constant.
    C(f64),
    /// A named definition from the biome's `let`, the library, or a scatter var.
    Ref(String),
    /// Sum of all terms.
    Add(Vec<Expr>),
    /// Product of all factors.
    Mul(Vec<Expr>),
    /// First minus second.
    Sub(Box<Expr>, Box<Expr>),
    /// Negation.
    Neg(Box<Expr>),
    /// Smallest term.
    Min(Vec<Expr>),
    /// Largest term.
    Max(Vec<Expr>),
    /// First operand divided by the second; the second must stay away from
    /// zero.
    Div(Box<Expr>, Box<Expr>),
    /// Euclidean length of the operands taken as a vector.
    Length(Vec<Expr>),
    /// Absolute value.
    Abs(Box<Expr>),
    /// Sine of radians.
    Sin(Box<Expr>),
    /// Cosine of radians.
    Cos(Box<Expr>),
    /// Value clamped to `[min, max]`.
    Clamp(Box<Expr>, f64, f64),
    /// Linear map of `from` onto `to`, unclamped.
    Remap(Box<Expr>, (f64, f64), (f64, f64)),
    /// Hermite step from 0 at `from.0` to 1 at `from.1`.
    Smoothstep(Box<Expr>, f64, f64),
    /// Sign-preserving power `sign(v)|v|^exp`.
    Pow(Box<Expr>, f64),
    /// Piecewise-linear curve through sorted `(input, output)` points.
    Spline(Box<Expr>, Vec<(f64, f64)>),
    /// `a + (b - a) * clamp(t, 0, 1)`.
    Lerp(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Steps of `step` metres; `sharpness` in `[0, 1)` flattens each tread.
    Terrace {
        /// Value to terrace.
        of: Box<Expr>,
        /// Riser spacing.
        step: f64,
        /// Fraction of each step that is flat tread.
        sharpness: f64,
    },
    /// Solid wherever any operand is (maximum).
    Union(Vec<Expr>),
    /// Solid only where every operand is (minimum).
    Intersect(Vec<Expr>),
    /// First operand with the second carved away.
    Subtract(Box<Expr>, Box<Expr>),
    /// Union with a fillet of radius about `k`.
    SmoothUnion(f64, Vec<Expr>),
    /// Intersection with a fillet of radius about `k`.
    SmoothIntersect(f64, Vec<Expr>),
    /// Subtraction with a fillet of radius about `k`.
    SmoothSubtract(f64, Box<Expr>, Box<Expr>),
    /// Coherent noise sampled in the current domain.
    Noise(NoiseDoc),
    /// `height(x, z) - y`: ground below a heightfield.
    Height(Box<Expr>),
    /// Offsets the domain before evaluating `of`.
    Translate((f64, f64, f64), Box<Expr>),
    /// Uniform scale that keeps distances in metres.
    Scale(f64, Box<Expr>),
    /// Per-axis stretch of the shape; distances shrink by the smallest factor.
    Stretch((f64, f64, f64), Box<Expr>),
    /// Rotation of the shape about x by degrees.
    RotateX(f64, Box<Expr>),
    /// Rotation of the shape about y by degrees.
    RotateY(f64, Box<Expr>),
    /// Rotation of the shape about z by degrees.
    RotateZ(f64, Box<Expr>),
    /// Twist about the y axis by `rate` degrees per metre of height.
    Twist(f64, Box<Expr>),
    /// Infinite repetition with the given period per axis; 0 disables an axis.
    Repeat((f64, f64, f64), Box<Expr>),
    /// Displaces the domain by vector noise before evaluating `of`.
    Warp {
        /// Noise whose `amp` is the displacement in metres.
        by: NoiseDoc,
        /// Multiplier on the vertical displacement; 0 warps horizontally only.
        #[serde(default = "one")]
        vertical: f64,
        /// Warped expression.
        of: Box<Expr>,
    },
    /// Replaces domain axes with arbitrary expressions.
    Domain {
        /// New x, or the current one.
        #[serde(default)]
        x: Option<Box<Expr>>,
        /// New y, or the current one.
        #[serde(default)]
        y: Option<Box<Expr>>,
        /// New z, or the current one.
        #[serde(default)]
        z: Option<Box<Expr>>,
        /// Expression in the new domain.
        of: Box<Expr>,
    },
    /// Solid ball of `radius` at the origin.
    Sphere(Box<Expr>),
    /// Solid box with half-extents and edge rounding.
    Box {
        /// Half extent per axis.
        half: (f64, f64, f64),
        /// Rounding radius taken from the extents.
        #[serde(default)]
        round: f64,
    },
    /// Torus lying in the xz plane: a ring of `major` radius and `minor` thickness.
    Torus {
        /// Ring radius.
        major: Box<Expr>,
        /// Tube radius.
        minor: Box<Expr>,
    },
    /// Segment from `a` to `b` swept by `radius`.
    Capsule {
        /// First end.
        a: (f64, f64, f64),
        /// Second end.
        b: (f64, f64, f64),
        /// Sweep radius.
        radius: Box<Expr>,
    },
    /// Vertical cylinder centred on the origin.
    Cylinder {
        /// Radius.
        radius: Box<Expr>,
        /// Half height.
        half_height: Box<Expr>,
    },
    /// Cone with its base disc on y = 0 and its apex at `height`.
    Cone {
        /// Base radius.
        radius: Box<Expr>,
        /// Apex height.
        height: Box<Expr>,
    },
    /// Half-space below the plane `dot(normal, p) = offset`.
    Plane {
        /// Outward normal, normalised on compile.
        normal: (f64, f64, f64),
        /// Offset along the normal.
        #[serde(default)]
        offset: f64,
    },
    /// A thickened gyroid sheet, a triply periodic minimal surface.
    Gyroid {
        /// Cell period in metres.
        period: f64,
        /// Sheet thickness in metres.
        thickness: Box<Expr>,
    },
    /// Jittered-grid instances of a shape, merged by union.
    Scatter(Box<ScatterDoc>),
    /// Horizontal distance in metres to the nearest zero line of a 2D noise:
    /// `|n| / |∇n|`, exact near the line whatever the noise's amplitude or
    /// frequency. `Sub(C(w), Fissure(..))` is a wall-to-wall cut `2w` wide.
    /// Keep `w` well below the noise's wavelength; the estimate only holds
    /// near a line. `offset` moves the line to another level of the noise.
    Fissure(NoiseDoc),
}

const fn one() -> f64 {
    1.0
}

const fn two() -> f64 {
    2.0
}

const fn half() -> f64 {
    0.5
}

const fn one_octave() -> u8 {
    1
}

/// Base noise function.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Hash)]
pub enum NoiseKind {
    /// Smooth `OpenSimplex2` gradient noise.
    #[default]
    Simplex,
    /// Smoother, rounder `OpenSimplex2S`.
    SimplexSmooth,
    /// Classic Perlin gradient noise.
    Perlin,
    /// Interpolated value noise; blocky.
    Value,
    /// Distance to the nearest cell point; pits and domes.
    Cells,
    /// Second minus first cell distance; thin ridges on cell borders.
    CellEdges,
}

/// How octaves combine.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Hash)]
pub enum Fractal {
    /// Plain fractal Brownian motion.
    #[default]
    Fbm,
    /// Sharp crests where the base noise crosses zero.
    Ridged,
    /// Rounded billows; ridged upside down.
    Billow,
}

/// Dimensionality of a noise lookup.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Hash)]
pub enum Dims {
    /// Varies in x and z only; cheap and column-constant.
    Two,
    /// Varies in every axis.
    #[default]
    Three,
}

/// Parameters of one noise lookup. Output lies in `offset ± amp`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NoiseDoc {
    /// Base function.
    #[serde(default)]
    pub kind: NoiseKind,
    /// Octave combination.
    #[serde(default)]
    pub fractal: Fractal,
    /// Octave count.
    #[serde(default = "one_octave")]
    pub octaves: u8,
    /// Cycles per metre of the first octave.
    pub freq: f64,
    /// Frequency multiplier per octave.
    #[serde(default = "two")]
    pub lacunarity: f64,
    /// Amplitude multiplier per octave.
    #[serde(default = "half")]
    pub gain: f64,
    /// Output half range.
    #[serde(default = "one")]
    pub amp: f64,
    /// Output centre.
    #[serde(default)]
    pub offset: f64,
    /// Sampled dimensions.
    #[serde(default)]
    pub dims: Dims,
    /// Decorrelates otherwise identical noises.
    #[serde(default)]
    pub seed: u32,
}

/// Jittered-grid instancing. Each grid cell may hold one instance whose origin
/// sits at `ground` plus `lift`, optionally yawed and tilted at random.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ScatterDoc {
    /// Grid spacing in metres.
    pub cell: f64,
    /// Largest distance from an instance origin its shape can reach, fillets
    /// included. Instances are only evaluated inside this ball.
    pub reach: f64,
    /// Fraction of the cell an origin may wander, in `[0, 1]`.
    #[serde(default = "jitter")]
    pub jitter: f64,
    /// Probability that a cell holds an instance.
    #[serde(default = "one")]
    pub chance: f64,
    /// Instance exists only where this `(x, z)` expression is positive.
    #[serde(default)]
    pub mask: Option<Expr>,
    /// Origin height as an `(x, z)` expression; 0 when absent.
    #[serde(default)]
    pub ground: Option<Expr>,
    /// Random vertical offset range added to `ground`.
    #[serde(default)]
    pub lift: (f64, f64),
    /// Random rotation about y.
    #[serde(default = "yes")]
    pub yaw: bool,
    /// Largest random tilt in degrees.
    #[serde(default)]
    pub tilt: f64,
    /// Per-instance random values, usable as `Ref(name)` in `shape`.
    #[serde(default)]
    pub vars: BTreeMap<String, (f64, f64)>,
    /// Decorrelates otherwise identical scatters.
    #[serde(default)]
    pub seed: u32,
    /// Shape in instance-local coordinates.
    pub shape: Expr,
}

const fn jitter() -> f64 {
    0.8
}

const fn yes() -> bool {
    true
}

/// Texture family a surface draws its maps from.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash)]
pub enum TextureSet {
    /// Grass blades.
    Grass,
    /// Loose dirt.
    Dirt,
    /// Rough stone.
    Stone,
    /// Fine sand.
    Sand,
    /// Iron ore.
    Iron,
    /// Graphite.
    Graphite,
    /// Copper ore.
    Copper,
}

impl TextureSet {
    /// Every texture set in array-layer order.
    pub const ALL: [Self; 7] = [
        Self::Grass,
        Self::Dirt,
        Self::Stone,
        Self::Sand,
        Self::Iron,
        Self::Graphite,
        Self::Copper,
    ];

    /// Texture-array layer.
    pub const fn layer(self) -> u32 {
        match self {
            Self::Grass => 0,
            Self::Dirt => 1,
            Self::Stone => 2,
            Self::Sand => 3,
            Self::Iron => 4,
            Self::Graphite => 5,
            Self::Copper => 6,
        }
    }
}

/// One named ground appearance with its physical behaviour.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct SurfaceDoc {
    /// Name referenced by surface rules.
    pub name: String,
    /// How the ground responds to loads and digging.
    pub material: TerrainMaterial,
    /// Texture family.
    pub texture: TextureSet,
    /// sRGB tint as `#rrggbb`.
    #[serde(default = "white")]
    pub tint: String,
    /// Tint only where the texture's tint mask allows, when it has one.
    #[serde(default = "yes")]
    pub masked: bool,
    /// Replace the texture's hue with the tint instead of multiplying by it,
    /// keeping only its light and shade.
    #[serde(default)]
    pub recolor: bool,
    /// Roughness multiplier.
    #[serde(default = "one")]
    pub roughness: f64,
    /// Texture repeat multiplier; larger is coarser.
    #[serde(default = "one")]
    pub scale: f64,
}

fn white() -> String {
    "#ffffff".to_owned()
}

/// Condition on a point just below or at the surface.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub enum Cond {
    /// Always true.
    Always,
    /// Metres below the surface in `[min, max]`.
    Depth(f64, f64),
    /// Surface normal's y component in `[min, max]`; 1 is flat ground.
    Up(f64, f64),
    /// Surfaces facing downward: overhangs, arch undersides, cave roofs.
    Ceiling,
    /// World height in `[min, max]`.
    Altitude(f64, f64),
    /// Expression value in `[min, max]`.
    Field(Expr, f64, f64),
    /// Within this many metres of a river's centre line.
    NearRiver(f64),
    /// The face was opened by a carve layer: the named one, or any.
    Carved(Option<String>),
    /// Every condition holds.
    All(Vec<Cond>),
    /// Any condition holds.
    Any(Vec<Cond>),
    /// The condition does not hold.
    Not(Box<Cond>),
}

/// First matching rule wins.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct SurfaceRuleDoc {
    /// When the rule applies.
    pub when: Cond,
    /// Surface name.
    #[serde(rename = "use")]
    pub surface: String,
}

/// Target point in climate space. Unset axes match anything.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
pub struct ClimateDoc {
    /// Warm is positive.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Wet is positive.
    #[serde(default)]
    pub humidity: Option<f64>,
    /// Inland is positive, open sea strongly negative.
    #[serde(default)]
    pub continentalness: Option<f64>,
    /// How strange the land is allowed to become.
    #[serde(default)]
    pub weirdness: Option<f64>,
}

/// One biome: where it occurs and what shape language it speaks.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename = "Biome")]
pub struct BiomeDoc {
    /// Unique identifier.
    pub name: String,
    /// Preferred climate.
    pub climate: ClimateDoc,
    /// Added to the climate distance; positive makes the biome rarer.
    #[serde(default)]
    pub rarity: f64,
    /// Ground height as an `(x, z)` expression, used by rivers and spawn and
    /// available to `density` as `Ref("height")`.
    pub height: Expr,
    /// Full 3D density; `Height(Ref("height"))` when absent.
    #[serde(default)]
    pub density: Option<Expr>,
    /// Local named definitions.
    #[serde(default, rename = "let")]
    pub definitions: BTreeMap<String, Expr>,
    /// River depth multiplier; 0 keeps rivers out.
    #[serde(default = "one")]
    pub rivers: f64,
    /// Multiplier per carve layer, 1 when absent: 0 keeps a layer out and
    /// values above 1 widen its voids.
    #[serde(default)]
    pub carves: BTreeMap<String, f64>,
    /// Ordered surface rules.
    pub surface: Vec<SurfaceRuleDoc>,
}

/// Climate channels, each an `(x, z)` expression roughly in `[-1, 1]`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ClimateFieldsDoc {
    /// Temperature.
    pub temperature: Expr,
    /// Humidity.
    pub humidity: Expr,
    /// Continentalness.
    pub continentalness: Expr,
    /// Weirdness.
    pub weirdness: Expr,
}

/// Rivers traced over the blended biome heights at world creation.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RiversDoc {
    /// Upstream area in square kilometres at which a stream appears.
    pub source_area: f64,
    /// Half width at the source and at the largest catchment, in metres.
    pub half_width: (f64, f64),
    /// Channel depth at the source and at the largest catchment.
    pub depth: (f64, f64),
    /// Rise of the valley walls per metre from the bank.
    pub bank_slope: f64,
    /// Horizontal meander displacement in metres.
    pub meander: f64,
}

/// Where the world starts.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct SpawnDoc {
    /// Biome forced around the origin.
    pub biome: String,
    /// Radius of the forced area in metres.
    pub radius: f64,
}

/// `world.ron`: the frame every biome is placed into.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename = "World")]
pub struct WorldDoc {
    /// Lowest and highest generated height; solid below, empty above.
    pub vertical: (f64, f64),
    /// Height rivers drain to.
    pub sea_level: f64,
    /// Climate fields.
    pub climate: ClimateFieldsDoc,
    /// Climate-distance band over which neighbouring biomes blend.
    pub blend: f64,
    /// Spawn area.
    pub spawn: SpawnDoc,
    /// River network.
    pub rivers: RiversDoc,
    /// Layers that open voids in the ground: caves, entrances, ravines.
    pub carves: Vec<CarveDoc>,
    /// Biome file stems under `biomes/`, in order.
    pub biomes: Vec<String>,
    /// Surface palette.
    pub palette: Vec<SurfaceDoc>,
}

/// One layer of voids cut into the blended biome ground.
///
/// Every expression may read the world's fields: `Ref("ground")`, the
/// blended biome height before 3D features, and `Ref("base")`, the regional
/// low ground under it, both interpolated from the 32 m drainage grid, plus
/// the four climate channels by name.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CarveDoc {
    /// Name biomes use to scale the layer.
    pub name: String,
    /// Where the layer is open: positive is void, roughly metres. Besides
    /// the fields it may read `Ref("rock")`, the biome density at the point:
    /// how deep into rock it lies, so a cut can taper with depth.
    pub void: Expr,
    /// Rock, in metres of biome density, kept between a void and any face of
    /// the ground as an `(x, z)` expression. At or below zero the layer breaks
    /// through the surface.
    pub roof: Expr,
    /// Height below which the layer closes, as an `(x, z)` expression.
    pub floor: Expr,
    /// Height above which the layer closes; unbounded when absent.
    #[serde(default)]
    pub top: Option<Expr>,
    /// Beyond cave distance only voids within this much rock of the surface
    /// are cut, and only where the roof lets them breach, since enclosed
    /// voids are invisible there. Set it to the layer's deepest open cut.
    #[serde(default = "visible_depth")]
    pub visible: f64,
    /// Multiplier for biomes that do not list the layer.
    #[serde(default = "one")]
    pub unlisted: f64,
}

const fn visible_depth() -> f64 {
    12.0
}

/// `library.ron`: definitions shared by every biome.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename = "Library")]
pub struct LibraryDoc {
    /// Named expressions.
    pub definitions: BTreeMap<String, Expr>,
}
