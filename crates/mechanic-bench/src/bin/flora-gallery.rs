//! Pictures of grown trees for tuning the species genome, and their measures.
//!
//! Writes `sheet.png`, one row per species in `flora.ron` order: side views of
//! several seeds, then a top view of the first. `species-<name>.png` shows
//! two seeds larger. Unless `--no-voxels`, `voxels-<name>.png` shows the first
//! seed as the terrain field holds it: the front-most solid cells at 5 cm, the
//! same at the 20 cm stride of a coarser level of detail, and a slice through
//! the trunk. With `--sweep <field>` or `--sweep all`, `sweep-<field>.png`
//! grows the base species (`--species`, oak by default) at seven values of
//! that field from its smallest to its largest, two seeds each. Emits one
//! JSONL line per tree and per sweep step, then a summary.
//!
//! `cargo run -p mechanic-bench --release --bin flora-gallery -- --out <dir>`
//! with optional `--flora <file>`, `--seeds <n>`, `--sweep <field|all>`,
//! `--species <name>`, and `--no-voxels`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "pixel grids of a few thousand and colour bytes"
)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy_math::DVec3;
use mechanic_bench::images::write_png;
use mechanic_world::{
    GenomeSweep, Part, SpeciesSpec, TERRAIN_CELL_METERS, TreeMetrics, TreeModel, grow_tree,
};

const PANEL_WIDTH: usize = 300;
const PANEL_HEIGHT: usize = 420;
const SWEEP_STEPS: usize = 7;
const SWEEP_SEEDS: u64 = 2;
const FILL_SAMPLES: u32 = 4_000;

const SKY: [f32; 3] = [0.87, 0.90, 0.93];
const SOIL: [f32; 3] = [0.66, 0.56, 0.44];
const LAWN: [f32; 3] = [0.82, 0.86, 0.78];
const ROOT: [f32; 3] = [0.36, 0.24, 0.15];

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    let out = PathBuf::from(value("--out").unwrap_or_else(|| "flora-gallery".to_owned()));
    let seeds: u64 = value("--seeds").map_or(Ok(4), |value| value.parse())?;
    let voxels = !args.iter().any(|arg| arg == "--no-voxels");
    std::fs::create_dir_all(&out)?;
    let library = match value("--flora") {
        Some(path) => SpeciesSpec::library(&std::fs::read_to_string(path)?)?,
        None => SpeciesSpec::embedded(),
    };

    let mut rows = Vec::new();
    for species in &library {
        let mut trees = Vec::new();
        for seed in 0..seeds {
            let started = Instant::now();
            let tree = grow_tree(species, tree_seed(seed), DVec3::ZERO);
            let grow_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let metrics = TreeMetrics::measure(&tree);
            let filled = TreeMetrics::filled_fraction(&tree, FILL_SAMPLES);
            let sample_ns = sample_cost(&tree);
            println!(
                "{}",
                serde_json::json!({
                    "kind": "tree",
                    "species": species.name,
                    "seed": seed,
                    "grow_ms": grow_ms,
                    "sample_ns_per_cell": sample_ns,
                    "filled_fraction": filled,
                    "metrics": metrics_json(&metrics),
                })
            );
            trees.push(tree);
        }
        rows.push((species, trees));
    }

    write_sheet(&out, &rows)?;
    for (species, trees) in &rows {
        write_close_up(&out, species, trees)?;
        if voxels && let Some(tree) = trees.first() {
            write_voxels(&out, species, tree)?;
        }
    }

    if let Some(sweep) = value("--sweep") {
        let base_name = value("--species").unwrap_or_else(|| "oak".to_owned());
        let base = library
            .iter()
            .find(|species| species.name == base_name)
            .ok_or_else(|| format!("no species `{base_name}`"))?;
        for field in GenomeSweep::ALL
            .iter()
            .filter(|field| sweep == "all" || field.field == sweep)
        {
            write_sweep(&out, base, field)?;
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "summary",
            "species": library.iter().map(|species| species.name.clone()).collect::<Vec<_>>(),
            "out": out,
        })
    );
    Ok(())
}

const fn tree_seed(seed: u64) -> u64 {
    seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0x7ee5
}

fn metrics_json(metrics: &TreeMetrics) -> serde_json::Value {
    serde_json::json!({
        "height": metrics.height,
        "crown_width": metrics.crown_width,
        "width_ratio": metrics.width_ratio,
        "girth_ratio": metrics.girth_ratio,
        "stems": metrics.stems,
        "lowest_foliage": metrics.lowest_foliage,
        "widest_at": metrics.widest_at,
        "segments": metrics.segments,
        "foliage_blobs": metrics.foliage_blobs,
        "children_per_split": metrics.children_per_split,
        "branch_angle": metrics.branch_angle,
        "terminal_rise": metrics.terminal_rise,
        "lateral_rise": metrics.lateral_rise,
        "tortuosity": metrics.tortuosity,
        "split_density": metrics.split_density,
        "foliage_volume": metrics.foliage_volume,
        "root_radius": metrics.root_radius,
        "root_depth": metrics.root_depth,
        "trunk_share": metrics.trunk_share,
        "max_stem_lean": metrics.max_stem_lean,
        "leader_reach": metrics.leader_reach,
    })
}

/// Mean cost of one density sample over a 25 cm grid through the tree.
fn sample_cost(tree: &TreeModel) -> f64 {
    let step = 0.25;
    let span = tree.max - tree.min;
    let counts = (span / step).ceil();
    let started = Instant::now();
    let mut solid = 0_u64;
    let mut samples = 0_u64;
    for z in 0..counts.z as usize {
        for y in 0..counts.y as usize {
            for x in 0..counts.x as usize {
                let point = tree.min + DVec3::new(x as f64, y as f64, z as f64) * step;
                if tree.sample(point).is_some_and(|(density, _)| density > 0.0) {
                    solid += 1;
                }
                samples += 1;
            }
        }
    }
    std::hint::black_box(solid);
    started.elapsed().as_secs_f64() * 1.0e9 / samples.max(1) as f64
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Side,
    Top,
}

/// A panel's mapping from world metres to pixels.
#[derive(Clone, Copy)]
struct Frame {
    view: View,
    metres_per_pixel: f64,
    /// World coordinate at the left edge.
    left: f64,
    /// World coordinate at the top edge: height for side views, z for top.
    top: f64,
}

impl Frame {
    /// Pixel position and nearness (larger is nearer the viewer).
    fn project(&self, point: DVec3) -> (f64, f64, f64) {
        let x = (point.x - self.left) / self.metres_per_pixel;
        match self.view {
            View::Side => (x, (self.top - point.y) / self.metres_per_pixel, point.z),
            View::Top => (x, (point.z - self.top) / self.metres_per_pixel, point.y),
        }
    }

    fn unproject(&self, x: f64, y: f64, nearness: f64) -> DVec3 {
        let world_x = x.mul_add(self.metres_per_pixel, self.left);
        match self.view {
            View::Side => DVec3::new(
                world_x,
                y.mul_add(-self.metres_per_pixel, self.top),
                nearness,
            ),
            View::Top => DVec3::new(
                world_x,
                nearness,
                y.mul_add(self.metres_per_pixel, self.top),
            ),
        }
    }
}

struct Canvas {
    width: usize,
    height: usize,
    rgb: Vec<[f32; 3]>,
    nearness: Vec<f64>,
}

impl Canvas {
    fn new(width: usize, height: usize, colour: [f32; 3]) -> Self {
        Self {
            width,
            height,
            rgb: vec![colour; width * height],
            nearness: vec![f64::NEG_INFINITY; width * height],
        }
    }

    fn paint(&mut self, x: usize, y: usize, colour: [f32; 3]) {
        if x < self.width && y < self.height {
            self.rgb[y * self.width + x] = colour;
        }
    }

    fn blit(&mut self, panel: &Self, left: usize, top: usize) {
        for y in 0..panel.height {
            for x in 0..panel.width {
                self.paint(left + x, top + y, panel.rgb[y * panel.width + x]);
            }
        }
    }

    fn write(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let bytes = self
            .rgb
            .iter()
            .flat_map(|colour| {
                colour.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect::<Vec<_>>();
        write_png(path, self.width, self.height, &bytes)
    }

    /// Ground and sky (side) or lawn (top), with a 1 m and 5 m grid.
    fn backdrop(&mut self, frame: &Frame) {
        for y in 0..self.height {
            for x in 0..self.width {
                let world = frame.unproject(x as f64 + 0.5, y as f64 + 0.5, 0.0);
                let (along, up) = match frame.view {
                    View::Side => (world.x, world.y),
                    View::Top => (world.x, world.z),
                };
                let mut colour = match frame.view {
                    View::Side if world.y < 0.0 => SOIL,
                    View::Side => SKY,
                    View::Top => LAWN,
                };
                let line = |value: f64, every: f64| {
                    (value / every).round().mul_add(-every, value).abs()
                        < frame.metres_per_pixel * 0.5
                };
                let shade = if line(along, 5.0) || line(up, 5.0) {
                    0.88
                } else if frame.metres_per_pixel < 0.2 && (line(along, 1.0) || line(up, 1.0)) {
                    0.95
                } else {
                    1.0
                };
                colour = colour.map(|channel| channel * shade);
                self.paint(x, y, colour);
            }
        }
    }

    /// Draws a tapered capsule, nearest surface wins. Foliage is drawn only
    /// where the tree is solid just inside its surface, so its holes show.
    fn capsule(
        &mut self,
        frame: &Frame,
        (a, b): (DVec3, DVec3),
        (ra, rb): (f64, f64),
        colour: [f32; 3],
        foliage_of: Option<&TreeModel>,
    ) {
        let (ax, ay, az) = frame.project(a);
        let (bx, by, bz) = frame.project(b);
        let (pa, pb) = (ra / frame.metres_per_pixel, rb / frame.metres_per_pixel);
        let reach = pa.max(pb).max(0.5);
        let x0 = (ax.min(bx) - reach).floor().max(0.0) as usize;
        let y0 = (ay.min(by) - reach).floor().max(0.0) as usize;
        let x1 = ((ax.max(bx) + reach).ceil().max(0.0) as usize).min(self.width);
        let y1 = ((ay.max(by) + reach).ceil().max(0.0) as usize).min(self.height);
        let (dx, dy) = (bx - ax, by - ay);
        let length_squared = dx.mul_add(dx, dy * dy);
        for y in y0..y1 {
            for x in x0..x1 {
                let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                let t = if length_squared > 0.0 {
                    ((cx - ax).mul_add(dx, (cy - ay) * dy) / length_squared).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let distance = (cx - t.mul_add(dx, ax)).hypot(cy - t.mul_add(dy, ay));
                let radius = (pb - pa).mul_add(t, pa).max(0.5);
                if distance > radius {
                    continue;
                }
                let bulge = (1.0 - (distance / radius).powi(2)).max(0.0).sqrt();
                let world_radius = radius * frame.metres_per_pixel;
                let nearness = (bz - az).mul_add(t, az) + bulge * world_radius;
                let index = y * self.width + x;
                if nearness <= self.nearness[index] {
                    continue;
                }
                if let Some(tree) = foliage_of {
                    let inset = (world_radius * 0.5).min(0.1);
                    let point = frame.unproject(cx, cy, nearness - inset);
                    if !tree
                        .sample(point)
                        .is_some_and(|(density, part)| density > 0.0 && part == Part::Foliage)
                    {
                        continue;
                    }
                }
                self.nearness[index] = nearness;
                let shade = 0.5f32.mul_add(bulge as f32, 0.5);
                self.rgb[index] = colour.map(|channel| channel * shade);
            }
        }
    }
}

fn bark_colour(species: &SpeciesSpec) -> [f32; 3] {
    if species.bark.contains("birch") {
        [0.88, 0.87, 0.82]
    } else if species.bark.contains("bamboo") {
        [0.66, 0.70, 0.32]
    } else if species.bark.contains("spruce") {
        [0.40, 0.27, 0.20]
    } else {
        [0.44, 0.33, 0.22]
    }
}

fn foliage_colour(species: &SpeciesSpec) -> [f32; 3] {
    let look = &species.foliage.look;
    if look.contains("needle") {
        [0.14, 0.33, 0.22]
    } else if look.contains("willow") {
        [0.50, 0.64, 0.30]
    } else if look.contains("bamboo") {
        [0.42, 0.62, 0.24]
    } else if look.contains("birch") {
        [0.44, 0.66, 0.26]
    } else {
        [0.26, 0.48, 0.18]
    }
}

/// Room a set of trees needs: metres above and below the origin, and half
/// width.
fn extent(trees: &[&TreeModel]) -> (f64, f64, f64) {
    trees
        .iter()
        .fold((1.0, 0.5, 1.0), |(above, below, half), tree| {
            let span = (tree.max - tree.origin)
                .abs()
                .max((tree.min - tree.origin).abs());
            (
                above.max(tree.max.y - tree.origin.y),
                below.max(tree.origin.y - tree.min.y),
                half.max(span.x).max(span.z),
            )
        })
}

fn side_frame(tree: &TreeModel, metres_per_pixel: f64, above: f64, height: usize) -> Frame {
    let width_m = PANEL_WIDTH as f64 * metres_per_pixel;
    let room = height as f64 * metres_per_pixel;
    Frame {
        view: View::Side,
        metres_per_pixel,
        left: tree.origin.x - width_m * 0.5,
        top: tree.origin.y + above + 0.02 * room,
    }
}

fn render(
    species: &SpeciesSpec,
    tree: &TreeModel,
    frame: &Frame,
    width: usize,
    height: usize,
) -> Canvas {
    let mut canvas = Canvas::new(width, height, SKY);
    canvas.backdrop(frame);
    let bark = bark_colour(species);
    for segment in &tree.segments {
        let colour = if segment.part == Part::Root {
            ROOT
        } else {
            bark
        };
        canvas.capsule(
            frame,
            (segment.a, segment.b),
            (segment.ra, segment.rb),
            colour,
            None,
        );
    }
    let leaves = foliage_colour(species);
    for blob in &tree.foliage {
        canvas.capsule(
            frame,
            (blob.a, blob.b),
            (blob.radius, blob.radius),
            leaves,
            Some(tree),
        );
    }
    canvas
}

/// Fits side views of `trees` into panels of the given size.
fn side_scale(trees: &[&TreeModel], width: usize, height: usize) -> (f64, f64) {
    let (above, below, half) = extent(trees);
    let metres_per_pixel =
        ((above + below) / (height as f64 * 0.96)).max(2.0 * half / (width as f64 * 0.96));
    (metres_per_pixel, above)
}

fn write_sheet(out: &Path, rows: &[(&SpeciesSpec, Vec<TreeModel>)]) -> Result<(), Box<dyn Error>> {
    let every = rows.iter().flat_map(|(_, trees)| trees).collect::<Vec<_>>();
    let (metres_per_pixel, above) = side_scale(&every, PANEL_WIDTH, PANEL_HEIGHT);
    let columns = rows.iter().map(|(_, trees)| trees.len()).max().unwrap_or(0) + 1;
    let mut sheet = Canvas::new(PANEL_WIDTH * columns, PANEL_HEIGHT * rows.len(), [1.0; 3]);
    for (row, (species, trees)) in rows.iter().enumerate() {
        for (column, tree) in trees.iter().enumerate() {
            let frame = side_frame(tree, metres_per_pixel, above, PANEL_HEIGHT);
            let panel = render(species, tree, &frame, PANEL_WIDTH - 2, PANEL_HEIGHT - 2);
            sheet.blit(&panel, column * PANEL_WIDTH, row * PANEL_HEIGHT);
        }
        if let Some(tree) = trees.first() {
            let width_m = PANEL_WIDTH as f64 * metres_per_pixel;
            let frame = Frame {
                view: View::Top,
                metres_per_pixel,
                left: tree.origin.x - width_m * 0.5,
                top: tree.origin.z - width_m * 0.5,
            };
            let panel = render(species, tree, &frame, PANEL_WIDTH - 2, PANEL_WIDTH - 2);
            sheet.blit(&panel, trees.len() * PANEL_WIDTH, row * PANEL_HEIGHT);
        }
    }
    sheet.write(&out.join("sheet.png"))
}

fn write_close_up(
    out: &Path,
    species: &SpeciesSpec,
    trees: &[TreeModel],
) -> Result<(), Box<dyn Error>> {
    let shown = trees.iter().take(2).collect::<Vec<_>>();
    let (width, height) = (PANEL_WIDTH * 2, PANEL_HEIGHT * 2);
    let (metres_per_pixel, above) = side_scale(&shown, width, height);
    let mut canvas = Canvas::new(width * shown.len(), height, [1.0; 3]);
    for (column, tree) in shown.iter().enumerate() {
        let mut frame = side_frame(tree, metres_per_pixel, above, height);
        frame.left = tree.origin.x - width as f64 * metres_per_pixel * 0.5;
        let panel = render(species, tree, &frame, width - 2, height - 2);
        canvas.blit(&panel, column * width, 0);
    }
    canvas.write(&out.join(format!("species-{}.png", species.name)))
}

/// The tree as terrain cells: front-most solid cells at the true cell size,
/// the same at a 20 cm stride, and a slice through the base.
fn write_voxels(out: &Path, species: &SpeciesSpec, tree: &TreeModel) -> Result<(), Box<dyn Error>> {
    let cell = TERRAIN_CELL_METERS;
    let columns = ((tree.max.x - tree.min.x) / cell).ceil() as usize + 1;
    let rows = ((tree.max.y - tree.min.y) / cell).ceil() as usize + 1;
    let mut canvas = Canvas::new(columns * 3 + 8, rows, [1.0; 3]);
    let bark = bark_colour(species);
    let leaves = foliage_colour(species);
    let colour = |part: Part| match part {
        Part::Wood => bark,
        Part::Root => ROOT,
        Part::Foliage => leaves,
    };
    let depth_span = (tree.max.z - tree.min.z).max(cell);
    for (panel, stride) in [(0_usize, 1_usize), (1, 4)] {
        let step = cell * stride as f64;
        let lattice_stride = i32::try_from(stride)?;
        for row in (0..rows).step_by(stride) {
            for column in (0..columns).step_by(stride) {
                let x = (column as f64).mul_add(cell, tree.min.x);
                let y = (row as f64).mul_add(-cell, tree.max.y);
                let mut z = tree.max.z;
                let mut found = None;
                while z >= tree.min.z {
                    if let Some((density, part)) =
                        tree.sample_at_stride(DVec3::new(x, y, z), lattice_stride)
                        && density > 0.0
                    {
                        found = Some((part, (z - tree.min.z) / depth_span));
                        break;
                    }
                    z -= step;
                }
                let shade = |nearness: f64| 0.45 + 0.55 * nearness as f32;
                let paint = found.map_or(
                    if y < tree.origin.y { SOIL } else { SKY },
                    |(part, nearness)| colour(part).map(|channel| channel * shade(nearness)),
                );
                for dy in 0..stride {
                    for dx in 0..stride {
                        canvas.paint(panel * (columns + 4) + column + dx, row + dy, paint);
                    }
                }
            }
        }
    }
    for row in 0..rows {
        for column in 0..columns {
            let point = DVec3::new(
                (column as f64).mul_add(cell, tree.min.x),
                (row as f64).mul_add(-cell, tree.max.y),
                tree.origin.z,
            );
            let shown = match tree.sample(point) {
                Some((density, part)) if density > 0.0 => colour(part),
                _ if point.y < tree.origin.y => SOIL,
                _ => SKY,
            };
            canvas.paint(2 * (columns + 4) + column, row, shown);
        }
    }
    canvas.write(&out.join(format!("voxels-{}.png", species.name)))
}

fn write_sweep(out: &Path, base: &SpeciesSpec, field: &GenomeSweep) -> Result<(), Box<dyn Error>> {
    let mut grown = Vec::new();
    for step in 0..SWEEP_STEPS {
        let t = step as f64 / (SWEEP_STEPS - 1) as f64;
        let mut species = base.clone();
        (field.apply)(&mut species, t);
        let mut measures = Vec::new();
        for seed in 0..SWEEP_SEEDS {
            let tree = grow_tree(&species, tree_seed(seed), DVec3::ZERO);
            let metrics = TreeMetrics::measure(&tree);
            measures.push((field.metric)(&tree, &metrics));
            grown.push((step, seed, species.clone(), tree));
        }
        println!(
            "{}",
            serde_json::json!({
                "kind": "sweep",
                "field": field.field,
                "base": base.name,
                "t": t,
                "measure": measures.iter().sum::<f64>() / measures.len() as f64,
            })
        );
    }
    let (width, height) = (220, 320);
    let trees = grown.iter().map(|(.., tree)| tree).collect::<Vec<_>>();
    let (metres_per_pixel, above) = side_scale(&trees, width, height);
    let mut canvas = Canvas::new(width * SWEEP_STEPS, height * SWEEP_SEEDS as usize, [1.0; 3]);
    for (step, seed, species, tree) in &grown {
        let mut frame = side_frame(tree, metres_per_pixel, above, height);
        frame.left = tree.origin.x - width as f64 * metres_per_pixel * 0.5;
        let panel = render(species, tree, &frame, width - 2, height - 2);
        canvas.blit(&panel, step * width, *seed as usize * height);
    }
    canvas.write(&out.join(format!("sweep-{}.png", field.field.replace('.', "-"))))
}
