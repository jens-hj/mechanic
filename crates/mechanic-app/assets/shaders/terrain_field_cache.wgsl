// Generates the procedural terrain fields into the cache around the camera;
// see world/terrain_cache.rs. One invocation fills one texel of one layer
// with every field the terrain shader composes grass, dirt and stone from:
// distances to cell borders, cell ids and noise. Grain finer than a
// centimetre is not cached; the terrain shader reads it from a small tile. Features too small for the
// layer's texels fade to their mean, as a texture's mips would.

// One rectangle to fill. Mirrors `FieldRegion` in world/terrain_cache.rs.
struct Region {
    // Lower corner and size, in world texels of the layer.
    origin: vec2<i32>,
    size: vec2<u32>,
    layer: u32,
    // Projection plane: x (yz), y (xz) or z (xy).
    plane: u32,
    // Texel edge in metres.
    texel: f32,
}

@group(0) @binding(0) var stone_fields: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(1) var soil_fields: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var grass_fields: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> region: Region;

// Mirrors `FIELD_EDGE` in world/terrain_cache.rs.
const FIELD_EDGE: i32 = 512;

// Field encodings. Mirror the same constants in terrain_material.wgsl.
// Slab border distance spans [0, SLAB_RANGE] metres; chip and pebble
// distances inward from their outline span [-BORDER_RANGE, BORDER_RANGE].
const SLAB_RANGE: f32 = 0.05;
const BORDER_RANGE: f32 = 0.03;

// Feature sizes, in metres. Mirror the same constants in
// terrain_material.wgsl.
const STONE_SLAB_CELL: f32 = 0.75;
const STONE_CHIP_CELL: f32 = 0.24;
const DIRT_PEBBLE_CELL: f32 = 0.08;
const GRASS_STROKE_WIDTH: f32 = 0.08;

// Shapes only the generator needs.
const STONE_CLUSTER_METRES: f32 = 1.2;
const DIRT_PEBBLE_SHARE: f32 = 0.09;
const DIRT_BAND_LENGTH: f32 = 1.6;
const DIRT_BAND_WIDTH: f32 = 0.3;
const GRASS_STROKE_LENGTH: f32 = 0.45;
const TONE_METRES: f32 = 3.5;

fn hash(cell: vec2<i32>, seed: u32) -> u32 {
    var state = bitcast<u32>(cell.x) * 0x8da6b343u
        + bitcast<u32>(cell.y) * 0xd8163841u
        + seed * 0xcb1ab31fu;
    state ^= state >> 16u;
    state *= 0x7feb352du;
    state ^= state >> 15u;
    state *= 0x846ca68bu;
    state ^= state >> 16u;
    return state;
}

fn unit(bits: u32) -> f32 {
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

// The share of a feature `size` metres across that texels `texel` metres
// wide still hold: all of it from four texels, none below one and a half.
fn resolved(size: f32, texel: f32) -> f32 {
    return smoothstep(1.5, 4.0, size / texel);
}

// Value noise in [0, 1].
fn value_noise(point: vec2<f32>, seed: u32) -> f32 {
    let base = floor(point);
    let f = point - base;
    let u = f * f * (3.0 - 2.0 * f);
    let cell = vec2<i32>(base);
    let a = unit(hash(cell, seed));
    let b = unit(hash(cell + vec2<i32>(1, 0), seed));
    let c = unit(hash(cell + vec2<i32>(0, 1), seed));
    let d = unit(hash(cell + vec2<i32>(1, 1), seed));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal value noise around zero, from `wavelength` down by halves, leaving
// out octaves the texels cannot hold.
fn fractal(point: vec2<f32>, wavelength: f32, octaves: i32, texel: f32, seed: u32) -> f32 {
    var sum = 0.0;
    var size = wavelength;
    var amplitude = 0.5;
    for (var octave = 0; octave < octaves; octave += 1) {
        let shown = resolved(size, texel);
        if shown <= 0.0 {
            break;
        }
        sum += (value_noise(point / size, seed + u32(octave) * 7919u) - 0.5) * amplitude * shown;
        size *= 0.5;
        amplitude *= 0.5;
    }
    return sum;
}

struct Cell {
    // Distance in to the cell's nearest border, in cell units.
    border: f32,
    // Four random numbers that belong to the cell.
    id: vec4<f32>,
}

fn seed_offset(lattice: vec2<f32>, bits: u32, jitter: f32) -> vec2<f32> {
    let random = vec2<f32>(f32(bits & 0xffffu), f32(bits >> 16u)) * (1.0 / 65535.0);
    return lattice + 0.5 + (random - 0.5) * jitter;
}

// Voronoi cells with one jittered seed per unit lattice cell, and the exact
// distance to the nearest border: the closest bisector to any neighbour.
fn voronoi(point: vec2<f32>, jitter: f32, seed: u32) -> Cell {
    let base = floor(point);
    let cell = vec2<i32>(base);
    let local = point - base;
    var best = 1.0e9;
    var nearest = vec2<f32>(0.0);
    var nearest_lattice = vec2<i32>(0);
    var nearest_bits = 0u;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let lattice = vec2<i32>(x, y);
            let bits = hash(cell + lattice, seed);
            let offset = seed_offset(vec2<f32>(lattice), bits, jitter) - local;
            let distance = dot(offset, offset);
            if distance < best {
                best = distance;
                nearest = offset;
                nearest_lattice = lattice;
                nearest_bits = bits;
            }
        }
    }
    var border = 1.0e9;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let lattice = nearest_lattice + vec2<i32>(x, y);
            if all(lattice == nearest_lattice) {
                continue;
            }
            let bits = hash(cell + lattice, seed);
            let offset = seed_offset(vec2<f32>(lattice), bits, jitter) - local;
            let across = offset - nearest;
            border = min(border, dot(0.5 * (nearest + offset), normalize(across)));
        }
    }
    var out: Cell;
    out.border = border;
    out.id = vec4<f32>(
        unit(nearest_bits * 0x9e3779b9u),
        unit(nearest_bits * 0x85ebca6bu),
        unit(nearest_bits * 0xc2b2ae35u),
        unit(nearest_bits * 0x27d4eb2fu),
    );
    return out;
}

// Signed distance inward from an outline, as a field byte.
fn signed_field(inward: f32) -> f32 {
    return 0.5 + 0.5 * clamp(inward / BORDER_RANGE, -1.0, 1.0);
}

fn stone(point: vec2<f32>, texel: f32, seed: u32) -> vec4<f32> {
    let slab = voronoi(point / STONE_SLAB_CELL, 0.95, seed + 1u);
    var chip_inward = -BORDER_RANGE;
    var chip_id = 0.0;
    if resolved(STONE_CHIP_CELL * 0.5, texel) > 0.0 {
        let cluster = value_noise(point / STONE_CLUSTER_METRES, seed + 2u);
        let share = smoothstep(0.25, 0.75, cluster);
        if share > 0.0 {
            let chip = voronoi(point / STONE_CHIP_CELL, 0.9, seed + 3u);
            if chip.id.x < share * 0.85 {
                chip_inward = chip.border * STONE_CHIP_CELL
                    - STONE_CHIP_CELL * (0.04 + 0.08 * chip.id.y);
            }
            chip_id = chip.id.z;
        }
    }
    return vec4<f32>(
        clamp(slab.border * STONE_SLAB_CELL / SLAB_RANGE, 0.0, 1.0),
        signed_field(chip_inward),
        slab.id.w,
        chip_id,
    );
}

fn soil(point: vec2<f32>, texel: f32, seed: u32) -> vec4<f32> {
    // Soft bands run diagonally, wandering as they go.
    let along = vec2<f32>(0.7071, -0.7071);
    let warped = point + vec2<f32>(0.25 * fractal(point, 1.0, 2, texel, seed + 4u));
    let squeezed = warped / DIRT_BAND_WIDTH
        + along * dot(warped, along) * (1.0 / DIRT_BAND_LENGTH - 1.0 / DIRT_BAND_WIDTH);
    let band = 0.5 + fractal(squeezed, 1.0, 3, texel / DIRT_BAND_WIDTH, seed + 5u);
    var pebble_inward = -BORDER_RANGE;
    var pebble_id = 0.0;
    if resolved(DIRT_PEBBLE_CELL * 0.5, texel) > 0.0 {
        let pebble = voronoi(point / DIRT_PEBBLE_CELL, 0.9, seed + 6u);
        if pebble.id.x < DIRT_PEBBLE_SHARE {
            pebble_inward = pebble.border * DIRT_PEBBLE_CELL
                - DIRT_PEBBLE_CELL * (0.1 + 0.1 * pebble.id.y);
        }
        pebble_id = pebble.id.z;
    }
    return vec4<f32>(clamp(band, 0.0, 1.0), signed_field(pebble_inward), pebble_id, 0.0);
}

fn grass(point: vec2<f32>, texel: f32, seed: u32) -> vec4<f32> {
    // Strokes run along the plane's first axis, waving as slow noise bends
    // them.
    let bend = value_noise(point / 0.9, seed + 8u);
    let combed = point + vec2<f32>(0.0, 0.35 * bend);
    // Smooth noise survives coarser texels than a hard feature would, so
    // strokes stay until two texels span one.
    let stroke = mix(
        0.5,
        value_noise(combed / vec2<f32>(GRASS_STROKE_LENGTH, GRASS_STROKE_WIDTH), seed + 9u),
        smoothstep(1.0, 2.0, GRASS_STROKE_WIDTH / texel),
    );
    let tone = 0.5 + fractal(point, TONE_METRES, 3, texel, seed + 11u);
    return vec4<f32>(stroke, clamp(tone, 0.0, 1.0), 0.0, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn generate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if any(invocation.xy >= region.size) {
        return;
    }
    let texel = region.origin + vec2<i32>(invocation.xy);
    let stored = ((texel % FIELD_EDGE) + FIELD_EDGE) % FIELD_EDGE;
    let point = (vec2<f32>(texel) + 0.5) * region.texel;
    // Each plane draws its own pattern, so a cliff does not echo the ground.
    let seed = region.plane * 101u;
    let layer = i32(region.layer);
    textureStore(stone_fields, stored, layer, stone(point, region.texel, seed));
    textureStore(soil_fields, stored, layer, soil(point, region.texel, seed));
    textureStore(grass_fields, stored, layer, grass(point, region.texel, seed));
}
