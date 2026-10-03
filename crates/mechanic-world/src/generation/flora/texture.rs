//! Bark and leaf maps drawn from a species' genome.
//!
//! The same numbers that grow a tree decide how its surfaces look. A trunk
//! thick for its height is old: deep fissures between wide plates. Splitting
//! many ways at once breaks those plates into scales, and a wandering habit
//! makes the fissures wander. Smooth, young bark on a strong leader shows
//! horizontal lenticels instead, and a clump of many slender stems shows the
//! rings of its nodes. Leaves are as long as the foliage is deep, broad on a
//! deep crown and needle-thin on a shallow one; they hang or rise with the
//! twigs' tropism, fan out with the split angle, and cover as densely as the
//! foliage is filled.
//!
//! Every map tiles across [`TREE_TEXTURE_METRES`] and holds light and shade
//! only: the look a species names supplies the colour.

use bevy_math::{DVec2, DVec3};

use super::super::scatter::mix;
use super::super::tape::smoothstep;
use super::{INTERNODES, SpeciesSpec};

/// Metres one repeat of a tree texture spans: the terrain shader repeats
/// every texture this often.
pub const TREE_TEXTURE_METRES: f64 = 1.5;

/// Mean linear luminance of every tree base colour, so a recoloured look
/// keeps its tint's brightness on average.
pub const TREE_TEXTURE_LUMA: f32 = 0.2;

/// Which surface of a tree a texture draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeSurface {
    /// Trunk, branch, and root bark.
    Bark,
    /// Leaves or needles.
    Foliage,
}

/// One procedural tree texture: a species' bark or foliage.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTexture {
    /// The genome the texture is read from.
    pub species: SpeciesSpec,
    /// Which surface it draws.
    pub surface: TreeSurface,
}

/// RGBA8 maps of one tree texture, rows from the bottom of the repeat up.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTextureMaps {
    /// Pixels per edge.
    pub edge: u32,
    /// sRGB grey: light and shade for the look's colour.
    pub base_color: Vec<u8>,
    /// Tangent-space normal, x along the row and y up the repeat.
    pub normal: Vec<u8>,
    /// Occlusion, roughness, and metalness.
    pub orm: Vec<u8>,
}

/// How a species' bark looks, read from its genome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarkTraits {
    /// Depth of the fissures between plates, in metres.
    pub fissure_depth: f64,
    /// Width of a plate across the trunk, in metres.
    pub plate_width: f64,
    /// Plate length over width.
    pub plate_aspect: f64,
    /// How plainly plates show, 0 to 1: on old bark, or as scales on bark
    /// whose plates are short.
    pub plates: f64,
    /// How far fissures wander sideways, in plate widths.
    pub meander: f64,
    /// Strength of horizontal lenticels, 0 to 1.
    pub lenticels: f64,
    /// Strength of node rings, 0 to 1.
    pub rings: f64,
    /// Spacing of node rings, in metres.
    pub node_spacing: f64,
}

impl BarkTraits {
    /// The bark a genome grows.
    pub fn of(species: &SpeciesSpec) -> Self {
        // Trunks thick for their height are old and furrowed.
        let age = smoothstep(0.02, 0.075, species.girth);
        let splits = f64::midpoint(
            f64::from(species.split_count.0),
            f64::from(species.split_count.1),
        );
        let smooth = 1.0 - age;
        let plate_aspect = 1.0 + 6.0 / splits;
        // Splitting many ways at once breaks plates into short scales.
        let scales = 1.0 - smoothstep(2.0, 3.0, plate_aspect);
        Self {
            fissure_depth: 0.002 + 0.03 * age,
            plate_width: 0.03 + 0.12 * age,
            plate_aspect,
            plates: age.max(scales),
            meander: 0.15 + species.wobble,
            lenticels: smooth * species.dominance * (2.0 / splits).min(1.0),
            rings: smooth * smoothstep(1.0, 10.0, f64::from(species.stems.1)),
            node_spacing: f64::midpoint(species.height.0, species.height.1) / INTERNODES,
        }
    }
}

/// How a species' leaves look, read from its genome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafTraits {
    /// Leaf length, in metres.
    pub length: f64,
    /// Length over width: needles are long and thin.
    pub aspect: f64,
    /// Mean direction from the stalk: −1 hangs straight down, 1 points up.
    pub lift: f64,
    /// Spread of directions about the mean, in radians either side.
    pub spread: f64,
    /// Fraction of the surface under leaves.
    pub cover: f64,
    /// Depth of the lobes along a leaf's edge, as a fraction of its width.
    pub lobes: f64,
}

impl LeafTraits {
    /// The leaves a genome grows.
    pub fn of(species: &SpeciesSpec) -> Self {
        let size = species.foliage.size;
        let broad = smoothstep(0.2, 0.7, size);
        let alignment = species.tropism.abs();
        Self {
            length: 0.025 + 0.1 * size,
            aspect: 7.0 - 5.5 * broad,
            lift: species.tropism,
            spread: (1.0 - alignment).mul_add(2.0 * species.split_angle, 0.25),
            cover: 0.55 + 0.45 * species.foliage.density,
            lobes: species.wobble * broad,
        }
    }
}

impl TreeTexture {
    /// Draws the texture at `edge` pixels per repeat.
    ///
    /// # Panics
    ///
    /// If `edge` is zero.
    pub fn maps(&self, edge: u32) -> TreeTextureMaps {
        assert!(edge > 0, "a texture needs pixels");
        let seed = name_seed(&self.species.name)
            ^ match self.surface {
                TreeSurface::Bark => 0xba4c,
                TreeSurface::Foliage => 0xf011,
            };
        let relief = match self.surface {
            TreeSurface::Bark => bark(&BarkTraits::of(&self.species), seed, edge),
            TreeSurface::Foliage => leaves(&LeafTraits::of(&self.species), seed, edge),
        };
        relief.into_maps(edge)
    }
}

/// Height, light, occlusion and roughness per pixel, before encoding.
struct Relief {
    height: Vec<f64>,
    light: Vec<f64>,
    occlusion: Vec<f64>,
    roughness: Vec<f64>,
}

impl Relief {
    fn new(pixels: usize) -> Self {
        Self {
            height: vec![0.0; pixels],
            light: vec![0.0; pixels],
            occlusion: vec![1.0; pixels],
            roughness: vec![0.0; pixels],
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "unit values scaled to bytes"
    )]
    fn into_maps(self, edge: u32) -> TreeTextureMaps {
        let side = edge as usize;
        let pixel = TREE_TEXTURE_METRES / f64::from(edge);
        let byte = |value: f64| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        // Light is linear; scale it to the shared mean before encoding.
        #[expect(clippy::cast_precision_loss, reason = "a few hundred thousand pixels")]
        let mean = self.light.iter().sum::<f64>() / self.light.len() as f64;
        let gain = f64::from(TREE_TEXTURE_LUMA) / mean.max(1.0e-6);
        let mut base_color = Vec::with_capacity(side * side * 4);
        let mut normal = Vec::with_capacity(side * side * 4);
        let mut orm = Vec::with_capacity(side * side * 4);
        for row in 0..side {
            for column in 0..side {
                let index = row * side + column;
                let grey = byte(linear_to_srgb(self.light[index] * gain));
                base_color.extend_from_slice(&[grey, grey, grey, 255]);
                let at = |dx: usize, dy: usize| {
                    self.height[((row + dy) % side) * side + (column + dx) % side]
                };
                let slope_x = (at(1, 0) - at(side - 1, 0)) / (2.0 * pixel);
                let slope_y = (at(0, 1) - at(0, side - 1)) / (2.0 * pixel);
                let n = DVec3::new(-slope_x, -slope_y, 1.0).normalize();
                normal.extend_from_slice(&[
                    byte(n.x.mul_add(0.5, 0.5)),
                    byte(n.y.mul_add(0.5, 0.5)),
                    byte(n.z.mul_add(0.5, 0.5)),
                    255,
                ]);
                orm.extend_from_slice(&[
                    byte(self.occlusion[index]),
                    byte(self.roughness[index]),
                    0,
                    255,
                ]);
            }
        }
        TreeTextureMaps {
            edge,
            base_color,
            normal,
            orm,
        }
    }
}

fn bark(traits: &BarkTraits, seed: u64, edge: u32) -> Relief {
    let side = edge as usize;
    let mut relief = Relief::new(side * side);
    let across = |metres: f64| TREE_TEXTURE_METRES / metres;
    // Plates as cells of a stretched lattice: their borders are fissures.
    let plates = Cells::new(
        seed,
        across(traits.plate_width),
        across(traits.plate_width * traits.plate_aspect),
    );
    let lenticels = Cells::new(seed ^ 0x1e47, across(0.06), across(0.025));
    let wander = Noise::new(seed ^ 0x3a7d, plates.across * 0.5, plates.up * 0.5, 3);
    let grain = Noise::new(seed ^ 0x96a1, across(0.01), across(0.06), 2);
    let blotch = Noise::new(seed ^ 0x51c3, across(0.3), across(0.3), 3);
    let rings = across(traits.node_spacing).round().max(1.0);
    // Relief deep enough to see even on shallow scales.
    let depth = traits.fissure_depth.max(0.004 * traits.plates);
    for row in 0..side {
        for column in 0..side {
            let index = row * side + column;
            let point = pixel_point(column, row, edge);
            // Fissures wander sideways along their length.
            let shift = (wander.fbm(point) - 0.5) * 2.0 * traits.meander * traits.plate_width;
            let (near, gap) = plates.border(DVec2::new(point.x + shift, point.y));
            // 0 on a plate's crown, 1 in the bottom of a fissure, as plainly
            // as plates show.
            let fissure = (1.0 - smoothstep(0.0, 0.35, gap)) * traits.plates;
            let crown = near * 0.2 * traits.plates;
            let fine = grain.fbm(point) - 0.5;
            let mut height = depth * (1.0 - fissure - crown) + 0.0015 * fine;
            let mut light = (1.0 - 0.65 * fissure) * (0.9 + 0.2 * (blotch.fbm(point) - 0.5));
            light *= 1.0 + 0.15 * fine * (0.3 + 0.7 * traits.plates);
            // Lenticels: thin dark dashes across the stem.
            let (offset, chance) = lenticels.nearest(point);
            let half_length = 0.006 + 0.02 * chance;
            let lenticel = traits.lenticels
                * (1.0 - smoothstep(0.6, 1.0, offset.x.abs() / half_length))
                * (1.0 - smoothstep(0.4, 1.0, offset.y.abs() / 0.0025));
            light *= 1.0 - 0.75 * lenticel;
            height -= 0.0008 * lenticel;
            // Node rings: a raised, darker band every node.
            let phase = (point.y / TREE_TEXTURE_METRES * rings).fract();
            let ring_distance = phase.min(1.0 - phase) / rings * TREE_TEXTURE_METRES;
            let ring = traits.rings * (-(ring_distance / 0.008).powi(2)).exp();
            light *= 1.0 - 0.5 * ring;
            height += 0.003 * ring;
            relief.height[index] = height;
            relief.light[index] = light.max(0.02);
            relief.occlusion[index] = 1.0 - 0.6 * fissure * smoothstep(0.0, 0.02, depth);
            relief.roughness[index] =
                0.1f64.mul_add(fissure, 0.3f64.mul_add(smoothstep(0.0, 0.03, depth), 0.6));
        }
    }
    relief
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel counts of a few hundred"
)]
fn leaves(traits: &LeafTraits, seed: u64, edge: u32) -> Relief {
    let side = edge as usize;
    let pixel = TREE_TEXTURE_METRES / f64::from(edge);
    let mut relief = Relief::new(side * side);
    // Shadowed depth of the crown, seen through gaps.
    let depth = Noise::new(seed ^ 0x0dd5, 6.0, 6.0, 2);
    for row in 0..side {
        for column in 0..side {
            let index = row * side + column;
            let shade = depth.fbm(pixel_point(column, row, edge));
            relief.height[index] = -0.03;
            relief.light[index] = 0.12 + 0.1 * shade;
            relief.occlusion[index] = 0.35;
            relief.roughness[index] = 0.9;
        }
    }
    let length = traits.length;
    let width = length / traits.aspect;
    // Enough leaves, overlapping, to cover the asked fraction of the repeat.
    let area = std::f64::consts::FRAC_PI_4 * length * width;
    let count = (-(1.0 - traits.cover).ln() * TREE_TEXTURE_METRES * TREE_TEXTURE_METRES / area)
        .ceil() as u64;
    let mut state = mix(seed);
    let mut unit = || {
        state = mix(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mean = traits.lift * std::f64::consts::FRAC_PI_2;
    for leaf in 0..count {
        let base = DVec2::new(unit(), unit()) * TREE_TEXTURE_METRES;
        // Mirrored left and right about the vertical; hung or raised by lift.
        let side_sign = if unit() < 0.5 { -1.0 } else { 1.0 };
        let angle = (unit() - 0.5).mul_add(2.0 * traits.spread, mean);
        let along = DVec2::new(side_sign * angle.cos(), angle.sin());
        let across = along.perp();
        let leaf_length = length * (0.75 + 0.5 * unit());
        let tone = 0.75 + 0.5 * unit();
        let layer = leaf as f64 / count as f64 * 0.02;
        // The leaf's own box, turned with it.
        let reach = (along.abs() * leaf_length * 0.5 + across.abs() * width * 0.5) / pixel + 1.0;
        let centre = (base + along * leaf_length * 0.5) / pixel;
        let (low_x, high_x) = ((centre.x - reach.x) as i64, (centre.x + reach.x) as i64 + 1);
        let (low_y, high_y) = ((centre.y - reach.y) as i64, (centre.y + reach.y) as i64 + 1);
        for y in low_y..=high_y {
            for x in low_x..=high_x {
                let local = DVec2::new(x as f64, y as f64) * pixel - base;
                let s = local.dot(along) / leaf_length;
                if !(0.0..=1.0).contains(&s) {
                    continue;
                }
                let t = local.dot(across);
                let lobed = 1.0 - traits.lobes * (s * 3.0 * std::f64::consts::PI).sin().powi(2);
                let half = 0.5 * width * (s * std::f64::consts::PI).sin().powf(0.6) * lobed;
                if half <= 0.0 || t.abs() > half {
                    continue;
                }
                let index =
                    (y.rem_euclid(side as i64) * side as i64 + x.rem_euclid(side as i64)) as usize;
                let rise = 1.0 - (t / half).powi(2);
                let midrib = (-(t / (0.08 * width + pixel)).powi(2)).exp();
                relief.height[index] = layer + 0.15 * half * rise.sqrt() - 0.0008 * midrib;
                relief.light[index] = tone * (0.85 + 0.15 * rise) * (1.0 - 0.2 * midrib);
                relief.occlusion[index] = 0.8 + 0.2 * (leaf as f64 / count as f64);
                relief.roughness[index] = 0.55;
            }
        }
    }
    relief
}

/// Repeat coordinates of a pixel's centre, in metres.
fn pixel_point(column: usize, row: usize, edge: u32) -> DVec2 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel indices of a few thousand"
    )]
    let (x, y) = (column as f64 + 0.5, row as f64 + 0.5);
    DVec2::new(x, y) * (TREE_TEXTURE_METRES / f64::from(edge))
}

/// A seed from a species name, so each species keeps its own pattern.
fn name_seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |state, byte| {
        mix(state ^ u64::from(byte))
    })
}

fn hash_unit(seed: u64, x: i64, y: i64) -> f64 {
    #[expect(
        clippy::cast_sign_loss,
        reason = "lattice indices are hashed bit for bit"
    )]
    let state = mix(mix(seed ^ (x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        ^ (y as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
    let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
    unit
}

/// A jittered lattice of cells that tiles the repeat: `across` by `up`
/// cells, rounded to whole cells. Each cell's point and random value are
/// drawn once.
struct Cells {
    across: f64,
    up: f64,
    columns: i64,
    rows: i64,
    /// Each cell's point, within the cell.
    points: Vec<DVec2>,
    /// A random value per cell.
    chances: Vec<f64>,
}

impl Cells {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "cell counts of a few hundred"
    )]
    fn new(seed: u64, across: f64, up: f64) -> Self {
        let (across, up) = (across.round().max(1.0), up.round().max(1.0));
        let (columns, rows) = (across as i64, up as i64);
        let cells = (columns * rows) as usize;
        let (mut points, mut chances) = (Vec::with_capacity(cells), Vec::with_capacity(cells));
        for y in 0..rows {
            for x in 0..columns {
                let jitter = DVec2::new(hash_unit(seed, x, y), hash_unit(seed ^ 1, x, y));
                points.push(jitter * 0.7 + 0.15);
                chances.push(hash_unit(seed ^ 2, x, y));
            }
        }
        Self {
            across,
            up,
            columns,
            rows,
            points,
            chances,
        }
    }

    fn scaled(&self, point: DVec2) -> DVec2 {
        DVec2::new(point.x * self.across, point.y * self.up) / TREE_TEXTURE_METRES
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "wrapped cell indices"
    )]
    fn index(&self, x: i64, y: i64) -> usize {
        (x.rem_euclid(self.columns) + y.rem_euclid(self.rows) * self.columns) as usize
    }

    /// Each of the nine cells around a point: its point in cell units, and
    /// its index.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "cell indices of a few hundred"
    )]
    fn around(&self, local: DVec2) -> impl Iterator<Item = (DVec2, usize)> + '_ {
        let (cx, cy) = (local.x.floor() as i64, local.y.floor() as i64);
        (-1..=1).flat_map(move |dy| {
            (-1..=1).map(move |dx| {
                let (x, y) = (cx + dx, cy + dy);
                let index = self.index(x, y);
                (DVec2::new(x as f64, y as f64) + self.points[index], index)
            })
        })
    }

    /// Distance to the nearest cell point and how far the point lies from
    /// the border with the next nearest, both in cells.
    fn border(&self, point: DVec2) -> (f64, f64) {
        let local = self.scaled(point);
        let (mut first, mut second) = (f64::INFINITY, f64::INFINITY);
        for (feature, _) in self.around(local) {
            let distance = local.distance(feature);
            if distance < first {
                second = first;
                first = distance;
            } else if distance < second {
                second = distance;
            }
        }
        (first, (second - first) * 0.5)
    }

    /// Offset in metres from the nearest cell point, and a random value
    /// for that cell.
    fn nearest(&self, point: DVec2) -> (DVec2, f64) {
        let local = self.scaled(point);
        let mut best = (DVec2::splat(f64::INFINITY), 0.0);
        for (feature, index) in self.around(local) {
            let offset = DVec2::new(
                (local.x - feature.x) / self.across,
                (local.y - feature.y) / self.up,
            ) * TREE_TEXTURE_METRES;
            if offset.length_squared() < best.0.length_squared() {
                best = (offset, self.chances[index]);
            }
        }
        best
    }
}

/// Smooth value noise that tiles the repeat, in octaves of halving
/// wavelength and amplitude. Each octave's lattice is drawn once.
struct Noise {
    across: f64,
    up: f64,
    /// Lattice values per octave, with their column and row counts.
    octaves: Vec<(i64, i64, Vec<f64>)>,
}

impl Noise {
    fn new(seed: u64, across: f64, up: f64, octaves: u32) -> Self {
        let (across, up) = (across.round().max(1.0), up.round().max(1.0));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cell counts of a few hundred"
        )]
        let octaves = (0..octaves)
            .map(|octave| {
                let scale = f64::from(1_u32 << octave);
                let (columns, rows) = ((across * scale) as i64, (up * scale) as i64);
                let values = (0..rows)
                    .flat_map(|y| (0..columns).map(move |x| (x, y)))
                    .map(|(x, y)| hash_unit(seed ^ u64::from(octave), x, y))
                    .collect();
                (columns, rows, values)
            })
            .collect();
        Self {
            across,
            up,
            octaves,
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "wrapped lattice indices"
    )]
    fn value(&self, point: DVec2, octave: usize) -> f64 {
        let (columns, rows, values) = &self.octaves[octave];
        let scale = f64::from(1_u32 << octave);
        let local =
            DVec2::new(point.x * self.across, point.y * self.up) / TREE_TEXTURE_METRES * scale;
        let base = local.floor();
        let fraction = local - base;
        let weight = |value: f64| smoothstep(0.0, 1.0, value);
        let (wx, wy) = (weight(fraction.x), weight(fraction.y));
        let corner = |dx: i64, dy: i64| {
            let x = (base.x as i64 + dx).rem_euclid(*columns);
            let y = (base.y as i64 + dy).rem_euclid(*rows);
            values[(x + y * columns) as usize]
        };
        let low = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * wx;
        let high = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * wx;
        low + (high - low) * wy
    }

    /// Every octave, in `[0, 1]`.
    fn fbm(&self, point: DVec2) -> f64 {
        let (mut total, mut weight, mut amplitude) = (0.0, 0.0, 1.0);
        for octave in 0..self.octaves.len() {
            total += self.value(point, octave) * amplitude;
            weight += amplitude;
            amplitude *= 0.5;
        }
        total / weight
    }
}

fn linear_to_srgb(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055f64.mul_add(value.powf(1.0 / 2.4), -0.055)
    }
}
