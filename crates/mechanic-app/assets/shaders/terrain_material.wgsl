#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_functions,
    mesh_view_bindings::view,
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
    view_transformations::position_world_to_clip,
}

// One palette surface: how a texture layer is tinted and shaded. Mirrors
// `TerrainSurfaceGpu` in world/terrain_render.rs.
struct Surface {
    // Linear tint; w > 0.5 replaces the texture's hue instead of multiplying.
    tint: vec4<f32>,
    // Texture layer, tint-masked flag, roughness multiplier, repeat multiplier.
    params: vec4<f32>,
    // Mean luminance of the layer's base colour, to keep recolours' brightness,
    // and the procedural recipe drawing it, or zero for its textures.
    shade: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var base_color_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var normal_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var orm_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var tint_masks: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var terrain_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var<storage, read> surfaces: array<Surface>;
// How wet the ground is around the camera: fill, the height of the ground
// it was measured at over `wet_window.w`, and how far its grass has wilted.
// Mirrors `TerrainRenderMaterial::wetness` in world/terrain_render.rs.
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var wetness_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var wetness_sampler: sampler;
// Lower x and z corner and edge of the wetness map, and its base height.
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var<uniform> wet_window: vec4<f32>;
// Independent removal, deposition, pending amount, and relative bed height.
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var erosion_map: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var<uniform> erosion_window: vec4<f32>;
// Procedural bark and leaf maps, drawn from each species' genome. Mirrors
// `TerrainRenderMaterial::tree_base_color` and its neighbours.
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var tree_base_color_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var tree_normal_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var tree_orm_maps: texture_2d_array<f32>;
#ifdef PROCEDURAL_GROUND
// The procedural field cache around the camera, a layer per plane and level,
// and each layer's window: lower corner in texels and whether it is filled.
// Mirrors `TerrainRenderMaterial::stone_fields` and its neighbours.
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var stone_fields: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(15) var soil_fields: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(16) var grass_fields: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(17) var<storage, read> field_windows: array<vec4<i32>>;
// Fine grain under the fields: fleck coverage, grit and speckle. Mirrors
// `TerrainRenderMaterial::grain`.
@group(#{MATERIAL_BIND_GROUP}) @binding(18) var grain_maps: texture_2d_array<f32>;
#endif

fn sediment_heatmap(position: vec3<f32>, lit: vec3<f32>) -> vec3<f32> {
    if erosion_window.z <= 0.0 { return lit; }
    let uv = (position.xz - erosion_window.xy) / erosion_window.z;
    if any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) { return lit; }
    let cell = uv * vec2<f32>(textureDimensions(erosion_map));
    let sample = textureLoad(erosion_map, vec2<i32>(cell), 0);
    let neutral = vec3<f32>(clamp(dot(lit, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.08, 0.6) * 0.45);
    let near = 1.0 - smoothstep(0.15, 0.4, abs(position.y - (sample.a + erosion_window.w)));
    let activity = sample.rgb * near;
    let total = activity.r + activity.g;
    let blue_share = activity.g / max(total, 0.00001);
    let mixed = 4.0 * blue_share * (1.0 - blue_share);
    let hue = mix(mix(vec3<f32>(1.0, 0.24, 0.015), vec3<f32>(0.02, 0.32, 1.0), blue_share),
        vec3<f32>(0.65, 0.08, 0.95), mixed);
    var color = mix(neutral, hue, max(activity.r, activity.g));
    // Hatching remains spatially anchored as the map recentres by whole cells.
    if fract((cell.x + cell.y) * 2.0) < 0.4 {
        color = mix(color, vec3<f32>(1.0, 0.9, 0.015), activity.b);
    }
    return color;
}


// Every chunk names up to eight palette surfaces; each vertex is one-hot over
// those slots. The slot table is the same for all of a chunk's vertices, so
// flat interpolation carries it exactly.
struct TerrainVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_b: vec2<f32>,
    @location(8) weights_low: vec4<f32>,
    @location(9) weights_high: vec4<f32>,
    @location(10) slots: vec4<u32>,
}

struct TerrainVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) uv_b: vec2<f32>,
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    @location(8) weights_low: vec4<f32>,
    @location(9) weights_high: vec4<f32>,
    @location(10) @interpolate(flat) slots: vec4<u32>,
}

@vertex
fn vertex(vertex: TerrainVertex) -> TerrainVaryings {
    var out: TerrainVaryings;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index,
    );
    out.uv = vertex.uv;
    out.uv_b = vertex.uv_b;
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    out.weights_low = vertex.weights_low;
    out.weights_high = vertex.weights_high;
    out.slots = vertex.slots;
    return out;
}

fn projection_weights(normal: vec3<f32>) -> vec3<f32> {
    // The fourth power is exact for signed components after two squares and
    // avoids the substantially more expensive generic pow implementation.
    let squared = normal * normal;
    let sharpened = squared * squared;
    return sharpened / max(dot(sharpened, vec3<f32>(1.0)), 0.0001);
}

fn dominant_projection(projection: vec3<f32>) -> vec3<f32> {
    if projection.x >= projection.y && projection.x >= projection.z {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if projection.y >= projection.z {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn footprint_adjusted_projection(
    footprint: f32,
    projection: vec3<f32>,
) -> vec3<f32> {
    // Triplanar blending is important up close. Once a fragment covers a
    // sizeable part of a material repeat, the secondary projections are no
    // longer resolvable and only multiply texture bandwidth. Fade to the
    // dominant projection so distant terrain retains every PBR map and its
    // authored mip detail while using one third as many texture samples.
    let dominant_weight = smoothstep(0.04, 0.08, footprint);
    return mix(projection, dominant_projection(projection), dominant_weight);
}

// Material textures repeat every 1.5 m, the figure the mesh divides world
// metres by; see terrain_chunk_mesh in world/terrain_render.rs.
const TEXTURE_REPEAT_METRES: f32 = 1.5;
// Boundary shaping, in world metres and in weight units. The feature size is
// the scale of the crumbs a surface edge breaks into, the strength how far the
// edge wanders, and the depth how wide the remaining transition is.
const BOUNDARY_FEATURE_METRES: f32 = 0.08;
const BOUNDARY_NOISE_STRENGTH: f32 = 0.3;
const BOUNDARY_BLEND_DEPTH: f32 = 0.1;
const SLOTS: u32 = 8u;
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// The grass texture's layer: `TextureSet::Grass.layer()`.
const GRASS_LAYER: i32 = 0;

// Layers from here on are tree texture layers. Mirrors `TREE_LAYER_BASE` in
// world/terrain_render.rs.
const TREE_LAYER_BASE: i32 = 64;

// Hue of dead grass, straw, as a multiple of its luminance.
const STRAW: vec3<f32> = vec3<f32>(1.3, 1.0, 0.45);

fn hash_corner(corner: vec3<f32>) -> f32 {
    // The odd multipliers are the collision kernel's cell hash, mixing three
    // lattice coordinates into one well-distributed word.
    var state = bitcast<u32>(i32(corner.x)) * 0x8da6b343u
        + bitcast<u32>(i32(corner.y)) * 0xd8163841u
        + bitcast<u32>(i32(corner.z)) * 0xcb1ab31fu;
    state ^= state >> 16u;
    state *= 0x7feb352du;
    state ^= state >> 15u;
    return f32(state) * (1.0 / 4294967296.0);
}

// Value and analytic derivative, so mineral-domain warping keeps explicit
// texture gradients even inside the material-selection branch.
fn value_noise_gradient(point: vec3<f32>) -> vec4<f32> {
    let base = floor(point);
    let fraction = point - base;
    let weight = fraction * fraction * (3.0 - 2.0 * fraction);
    let slope = 6.0 * fraction * (1.0 - fraction);
    let a = hash_corner(base);
    let b = hash_corner(base + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash_corner(base + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash_corner(base + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash_corner(base + vec3<f32>(0.0, 0.0, 1.0));
    let f = hash_corner(base + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash_corner(base + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash_corner(base + vec3<f32>(1.0, 1.0, 1.0));
    let low = mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
    let high = mix(mix(e, f, weight.x), mix(g, h, weight.x), weight.y);
    let dx = mix(mix(b - a, d - c, weight.y), mix(f - e, h - g, weight.y), weight.z);
    let dy = mix(mix(c - a, d - b, weight.x), mix(g - e, h - f, weight.x), weight.z);
    return vec4<f32>(mix(low, high, weight.z), slope * vec3<f32>(dx, dy, high - low));
}

fn value_noise(point: vec3<f32>) -> f32 {
    return value_noise_gradient(point).x;
}

// Wet ground's ragged edge: noise this coarse and this strong moves where the
// damp fringe fades out, never ground wet or dry through.
const WET_EDGE_FEATURE_METRES: f32 = 0.3;
const WET_EDGE_NOISE_STRENGTH: f32 = 0.6;

// The wetness map at `uv`, filtered by a cubic B-spline from four bilinear
// taps: smooth across its 20 cm texels, so wet ground shows no grid of
// columns.
fn sample_wetness(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(wetness_map));
    let texel = uv * size - 0.5;
    let base = floor(texel);
    let f = texel - base;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let low = (base - 0.5 + w1 / g0) / size;
    let high = (base + 1.5 + w3 / g1) / size;
    return g0.y * (g0.x * textureSampleLevel(wetness_map, wetness_sampler, low, 0.0)
            + g1.x * textureSampleLevel(wetness_map, wetness_sampler, vec2<f32>(high.x, low.y), 0.0))
        + g1.y * (g0.x * textureSampleLevel(wetness_map, wetness_sampler, vec2<f32>(low.x, high.y), 0.0)
            + g1.x * textureSampleLevel(wetness_map, wetness_sampler, high, 0.0));
}

// A surface's standing height at this fragment: its interpolated weight, moved
// by noise this surface alone sees. A surface with no weight here stands well
// below every present one and can never win the comparison that follows.
fn boundary_height(weight: f32, point: vec3<f32>, offset: f32) -> f32 {
    if weight <= 0.0 {
        return -1.0;
    }
    return weight + BOUNDARY_NOISE_STRENGTH * (value_noise(point + vec3<f32>(offset)) - 0.5);
}

fn slot_surface(slots: vec4<u32>, slot: u32) -> u32 {
    let word = slots[slot / 2u];
    return (word >> ((slot % 2u) * 16u)) & 0xffffu;
}

struct TerrainSample {
    color: vec4<f32>,
    surface: vec3<f32>,
    normal: vec3<f32>,
    mask: f32,
}

// Samples every map of a layer at `uv`, which moves by `du` and `dv` per
// pixel. Gradients are taken once in uniform control flow: lookups sit
// inside per-fragment branches, where implicit derivatives are undefined.
fn sample_projection(
    layer: i32,
    uv: vec2<f32>,
    du: vec2<f32>,
    dv: vec2<f32>,
    masked: bool,
) -> array<vec4<f32>, 4> {
    if layer >= TREE_LAYER_BASE {
        let tree = layer - TREE_LAYER_BASE;
        return array<vec4<f32>, 4>(
            textureSampleGrad(tree_base_color_maps, terrain_sampler, uv, tree, du, dv),
            textureSampleGrad(tree_orm_maps, terrain_sampler, uv, tree, du, dv),
            textureSampleGrad(tree_normal_maps, terrain_sampler, uv, tree, du, dv),
            vec4<f32>(1.0),
        );
    }
    // Untinted and unmasked surfaces skip the mask lookup entirely.
    var mask = vec4<f32>(1.0);
    if masked {
        mask = textureSampleGrad(tint_masks, terrain_sampler, uv, layer, du, dv);
    }
    return array<vec4<f32>, 4>(
        textureSampleGrad(base_color_maps, terrain_sampler, uv, layer, du, dv),
        textureSampleGrad(orm_maps, terrain_sampler, uv, layer, du, dv),
        textureSampleGrad(normal_maps, terrain_sampler, uv, layer, du, dv),
        mask,
    );
}

// Each projection decision serves all four maps. Keep the per-projection
// normal normalization and per-surface normalization used by the full blend.
fn sample_layer(
    layer: i32,
    masked: bool,
    coordinates: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
    projection: vec3<f32>,
    geometric_normal: vec3<f32>,
) -> TerrainSample {
    // Whiteout blend: every planar normal carries the surface it perturbs, so a
    // slope keeps its shading even where the projection collapses to one axis.
    let geometry = normalize(geometric_normal);
    let direction = select(vec3<f32>(-1.0), vec3<f32>(1.0), geometry >= vec3<f32>(0.0));
    var sampled: TerrainSample;
    // Bark fissures and hanging leaves run up their maps, so tree textures
    // keep up on the x-facing sides too, as they do on the z-facing ones.
    let upright = layer >= TREE_LAYER_BASE;
    if projection.x > 0.001 {
        let maps = sample_projection(
            layer,
            select(coordinates.yz, coordinates.zy, upright),
            select(ddx.yz, ddx.zy, upright),
            select(ddy.yz, ddy.zy, upright),
            masked,
        );
        sampled.color += maps[0] * projection.x;
        sampled.surface += maps[1].rgb * projection.x;
        sampled.mask += maps[3].r * projection.x;
        var tangent = normalize(maps[2].rgb * 2.0 - 1.0);
        if upright {
            tangent = vec3<f32>(tangent.y, tangent.x, tangent.z);
        }
        sampled.normal += vec3<f32>(
            abs(tangent.z) * geometry.x,
            tangent.x + geometry.y,
            tangent.y * direction.x + geometry.z,
        ) * projection.x;
    }
    if projection.y > 0.001 {
        let maps = sample_projection(layer, coordinates.xz, ddx.xz, ddy.xz, masked);
        sampled.color += maps[0] * projection.y;
        sampled.surface += maps[1].rgb * projection.y;
        sampled.mask += maps[3].r * projection.y;
        let tangent = normalize(maps[2].rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x + geometry.x,
            abs(tangent.z) * geometry.y,
            -tangent.y * direction.y + geometry.z,
        ) * projection.y;
    }
    if projection.z > 0.001 {
        let maps = sample_projection(layer, coordinates.xy, ddx.xy, ddy.xy, masked);
        sampled.color += maps[0] * projection.z;
        sampled.surface += maps[1].rgb * projection.z;
        sampled.mask += maps[3].r * projection.z;
        let tangent = normalize(maps[2].rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x * direction.z + geometry.x,
            tangent.y + geometry.y,
            abs(tangent.z) * geometry.z,
        ) * projection.z;
    }
    sampled.normal = normalize(sampled.normal);
    return sampled;
}

#ifdef PROCEDURAL_GROUND
// Procedural materials. Grass, dirt and stone are composed from fields that a
// compute shader keeps in world space around the camera: distances to cell
// borders, cell ids and noise (world/terrain_cache.rs). Nothing repeats.
// Each projection reads the cache level whose texels suit the pixel, and the
// distance fields keep outlines crisp however far a texel is magnified.

// Recipe numbers. Mirrors `procedural_recipe` in world/terrain_render.rs.
const RECIPE_GRASS: u32 = 1u;
const RECIPE_DIRT: u32 = 2u;
const RECIPE_STONE: u32 = 3u;

// Linear palettes, each built around its material's mean colour.
const GRASS_BASE: vec3<f32> = vec3<f32>(0.0976, 0.1683, 0.0513);
const GRASS_DARK: vec3<f32> = vec3<f32>(0.0545, 0.1119, 0.0296);
const GRASS_LIGHT: vec3<f32> = vec3<f32>(0.1620, 0.2623, 0.0762);
const GRASS_FLECK: vec3<f32> = vec3<f32>(0.2623, 0.3813, 0.1170);
const GRASS_DRY: vec3<f32> = vec3<f32>(0.1812, 0.2016, 0.0545);
const DIRT_BASE: vec3<f32> = vec3<f32>(0.0782, 0.0423, 0.0262);
const DIRT_DARK: vec3<f32> = vec3<f32>(0.0513, 0.0284, 0.0185);
const DIRT_LIGHT: vec3<f32> = vec3<f32>(0.1070, 0.0595, 0.0356);
const PEBBLE: vec3<f32> = vec3<f32>(0.1500, 0.1274, 0.1070);
const PEBBLE_LINE: vec3<f32> = vec3<f32>(0.0232, 0.0144, 0.0103);
const STONE_BASE: vec3<f32> = vec3<f32>(0.1329, 0.1529, 0.1812);

// The cache's shape. Mirrors world/terrain_cache.rs.
const FIELD_EDGE: f32 = 512.0;
const FIELD_LEVELS: u32 = 10u;
const FIELD_TEXEL: f32 = 0.003;
// Texels kept clear of a window's edge.
const FIELD_MARGIN: f32 = 2.0;
// Texels from a window's centre it can be trusted to reach: half its edge,
// less the snap the window may lag by and the margin.
const FIELD_REACH: f32 = 0.5 * FIELD_EDGE - 32.0 - FIELD_MARGIN;
// The last share of a level's reach, in octaves, over which it hands over
// to the next coarser level.
const FIELD_HANDOVER: f32 = 0.15;
// Field encodings and feature sizes. Mirror terrain_field_cache.wgsl.
const BORDER_RANGE: f32 = 0.03;
const DIRT_PEBBLE_CELL: f32 = 0.08;
const GRASS_STROKE_WIDTH: f32 = 0.08;
// The grain tile's span. Mirrors the cell counts in world/terrain_cache.rs.
const GRAIN_METRES: f32 = 0.75;
// Texels along each edge of the grain tile. Mirrors `GRAIN_EDGE` in
// world/terrain_cache.rs.
const GRAIN_EDGE: f32 = 512.0;
// Dark lines along cell borders, in metres.
const OUTLINE_METRES: f32 = 0.006;

// What a recipe yields at one point of one projection: colour, occlusion and
// roughness, the tint mask, and the slope of its relief along the plane.
struct Procedural {
    color: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    mask: f32,
    gradient: vec2<f32>,
}

// The share of a feature `size` metres across that a pixel `footprint`
// metres wide still shows: all of it from four pixels, none below one and a
// half.
fn resolved(size: f32, footprint: f32) -> f32 {
    return smoothstep(1.5, 4.0, size / max(footprint, 1.0e-6));
}

// Whether a layer is filled and holds `point` at least `FIELD_MARGIN`
// texels inside its window.
fn field_holds(layer: u32, point: vec2<f32>, texel: f32) -> bool {
    let window = field_windows[layer];
    let local = point * (1.0 / texel) - vec2<f32>(window.xy);
    return window.z != 0 && all(local > vec2<f32>(FIELD_MARGIN))
        && all(local < vec2<f32>(FIELD_EDGE - FIELD_MARGIN));
}

// The cache layers one projection reads: the finest whose texels are no
// finer than the pixel and whose window holds the point, and how much of
// the next coarser layer to blend in.
struct FieldPick {
    found: bool,
    layer: i32,
    texel: f32,
    blend: f32,
}

// `offset` is how far the point lies from the camera along the plane, which
// every window is centred on, and `wanted` the level whose texels suit the
// pixel, as a fraction.
fn pick_fields(plane: u32, point: vec2<f32>, offset: vec2<f32>, wanted: f32) -> FieldPick {
    var pick: FieldPick;
    pick.found = false;
    // The level a window must be to reach this far, as a fraction.
    let distance = max(abs(offset.x), abs(offset.y));
    let needed = log2(max(distance, 1.0e-6) * (1.0 / (FIELD_TEXEL * FIELD_REACH)));
    let reach_level = max(ceil(needed), 0.0);
    var level = u32(max(floor(wanted), reach_level));
    // Blend toward the coarser level late between texel sizes, and before
    // the window's edge. Coarse levels are always generated first, so the
    // coarser window is as current as this one.
    var blend = smoothstep(reach_level - FIELD_HANDOVER, reach_level, needed);
    if f32(level) == floor(wanted) {
        blend = max(blend, smoothstep(0.7, 1.0, fract(wanted)));
    }
    if level >= FIELD_LEVELS {
        return pick;
    }
    var texel = FIELD_TEXEL * f32(1u << level);
    // A window that has not caught up with the camera hands over whole.
    if !field_holds(plane * FIELD_LEVELS + level, point, texel) {
        level += 1u;
        texel *= 2.0;
        blend = 0.0;
        if level >= FIELD_LEVELS || !field_holds(plane * FIELD_LEVELS + level, point, texel) {
            return pick;
        }
    }
    pick.found = true;
    pick.layer = i32(plane * FIELD_LEVELS + level);
    pick.texel = texel;
    if level + 1u < FIELD_LEVELS {
        pick.blend = blend;
    }
    return pick;
}

fn sample_fields(fields: texture_2d_array<f32>, pick: FieldPick, point: vec2<f32>) -> vec4<f32> {
    let uv = point * (1.0 / (pick.texel * FIELD_EDGE));
    var value = textureSampleLevel(fields, terrain_sampler, uv, pick.layer, 0.0);
    if pick.blend > 0.0 {
        let coarser = textureSampleLevel(fields, terrain_sampler, uv * 0.5, pick.layer + 1, 0.0);
        value = mix(value, coarser, pick.blend);
    }
    return value;
}

// The texel holding `point` in the finer layer, unfiltered: for cell ids.
// Windows wrap at the edge, a power of two, so a mask finds the texel.
fn load_fields(fields: texture_2d_array<f32>, pick: FieldPick, point: vec2<f32>) -> vec4<f32> {
    let texel = vec2<i32>(floor(point * (1.0 / pick.texel)));
    return textureLoad(fields, texel & vec2<i32>(i32(FIELD_EDGE) - 1), pick.layer, 0);
}

// A fragment's body and outline from its signed distance field: how far in
// it is and how much of its outline shows.
fn fragment_shape(field: f32, footprint: f32) -> vec2<f32> {
    let inward = (field - 0.5) * 2.0 * BORDER_RANGE;
    let width = max(OUTLINE_METRES, footprint);
    let body = smoothstep(0.0, 0.5 * width, inward);
    let line = smoothstep(-width, -0.5 * width, inward) - smoothstep(0.5 * width, width, inward);
    return vec2<f32>(body, line);
}

// A cell's facet slope from one of its random numbers.
fn facet(id: f32, steepness: f32) -> vec2<f32> {
    return (fract(id * vec2<f32>(13.1, 71.7)) - 0.5) * steepness;
}

fn grass_look(fields: vec4<f32>, grain: vec4<f32>, footprint: f32) -> Procedural {
    var out: Procedural;
    out.occlusion = 0.98;
    out.roughness = 0.88;
    out.mask = 1.0;
    // Broad patches, some greener and some drier.
    let tone = fields.g - 0.5;
    out.color = mix(GRASS_BASE, GRASS_DRY, clamp(tone * 1.2, 0.0, 0.6)) * (1.0 + 0.5 * tone);
    // Wind-combed strokes, dark and light.
    let shown = resolved(GRASS_STROKE_WIDTH, footprint);
    let dark = smoothstep(0.7, 0.75, fields.r) * 0.75 * shown;
    let light = (1.0 - smoothstep(0.24, 0.29, fields.r)) * 0.6 * shown;
    out.color = mix(out.color, out.color * (GRASS_DARK / GRASS_BASE), dark);
    out.color = mix(out.color, out.color * (GRASS_LIGHT / GRASS_BASE), light);
    out.occlusion -= 0.08 * dark;
    // Bright flecks, like light catching single blades.
    out.color = mix(out.color, GRASS_FLECK, grain.r * 0.45);
    return out;
}

fn dirt_look(
    soil: vec4<f32>,
    pebble_id: f32,
    tone: f32,
    grain: vec4<f32>,
    footprint: f32,
) -> Procedural {
    var out: Procedural;
    out.occlusion = 1.0;
    out.roughness = 0.92;
    out.mask = 1.0;
    // Soft diagonal bands.
    out.color = mix(DIRT_LIGHT, DIRT_DARK, smoothstep(0.3, 0.7, soil.r));
    out.color *= 1.0 + 0.4 * (tone - 0.5);
    // Grit: specks a little lighter or darker than the soil around them.
    out.color *= 1.0 + 0.7 * (grain.g - 0.5);
    // Pebbles, outlined and left untinted.
    let shape = fragment_shape(soil.g, footprint) * resolved(DIRT_PEBBLE_CELL * 0.5, footprint);
    out.color = mix(out.color, PEBBLE * (0.5 + 0.6 * pebble_id), shape.x);
    out.color = mix(out.color, PEBBLE_LINE, shape.y);
    out.mask = 1.0 - max(shape.x, shape.y);
    out.roughness = mix(out.roughness, 0.7, shape.x);
    out.occlusion -= 0.3 * shape.y;
    out.gradient = facet(pebble_id, 0.3) * shape.x;
    return out;
}

// Broad weathering varies continuously across the world. Fractures belong to
// the rock geometry; mineral detail comes from the measured stone maps below.
fn stone_look(tone: f32, grain: vec4<f32>) -> Procedural {
    var out: Procedural;
    out.color = STONE_BASE * (0.85 + 0.3 * tone) * (0.96 + 0.08 * grain.b);
    out.occlusion = 1.0;
    out.roughness = 1.0;
    out.mask = 1.0;
    out.gradient = vec2<f32>(0.0);
    return out;
}

// A recipe with no fields to read: its mean.
fn plain_look(recipe: u32) -> Procedural {
    var out: Procedural;
    out.occlusion = 0.97;
    out.roughness = 0.85;
    out.mask = 1.0;
    switch recipe {
        case RECIPE_GRASS: { out.color = GRASS_BASE; }
        case RECIPE_DIRT: { out.color = DIRT_BASE; }
        default: { out.color = STONE_BASE; }
    }
    return out;
}

// One projection of a recipe at `point`, metres along its plane, which moves
// by `ddx` and `ddy` per pixel.
fn procedural_plane(
    recipe: u32,
    plane: u32,
    point: vec2<f32>,
    offset: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
) -> Procedural {
    // Fields are chosen for the pixel's geometric-mean width, no less than a
    // quarter of its long side: grazing ground keeps its detail across the
    // view and aliases little along it.
    let major = max(length(ddx), length(ddy));
    let footprint = max(sqrt(major * min(length(ddx), length(ddy))), major * 0.25);
    let pick = pick_fields(plane, point, offset, log2(max(footprint, 1.0e-6) * (1.0 / FIELD_TEXEL)));
    if !pick.found {
        return plain_look(recipe);
    }
    let grass = sample_fields(grass_fields, pick, point);
    // The grain tile, shifted by one of four offsets in regions the unique
    // stroke noise draws: no stretch of grain repeats, and a seam in noise
    // this fine does not show.
    // One trilinear lookup at the mip for the pixel's mean width:
    // anisotropic filtering costs more than this grain is worth.
    let shift = floor(grass.r * 4.0) * vec2<f32>(0.37, 0.61);
    let grain = textureSampleLevel(
        grain_maps,
        terrain_sampler,
        point * (1.0 / GRAIN_METRES) + shift,
        0,
        log2(max(footprint * (GRAIN_EDGE / GRAIN_METRES), 1.0)),
    );
    switch recipe {
        case RECIPE_GRASS: { return grass_look(grass, grain, footprint); }
        case RECIPE_DIRT: {
            let soil = sample_fields(soil_fields, pick, point);
            let ids = load_fields(soil_fields, pick, point);
            return dirt_look(soil, ids.b, grass.g, grain, footprint);
        }
        default: {
            return stone_look(grass.g, grain);
        }
    }
}

// Mean luminance of a recipe's colour, for recolours: its palette's base.
fn recipe_luma(recipe: u32) -> f32 {
    switch recipe {
        case RECIPE_GRASS: { return dot(GRASS_BASE, LUMA); }
        case RECIPE_DIRT: { return dot(DIRT_BASE, LUMA); }
        default: { return dot(STONE_BASE, LUMA); }
    }
}

// Draws a recipe at `point` metres, which moves by `ddx` and `ddy` metres
// per pixel, blending the projections as the textured layers do.
fn procedural_layer(
    recipe: u32,
    point: vec3<f32>,
    offset: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
    projection: vec3<f32>,
    geometric_normal: vec3<f32>,
) -> TerrainSample {
    let geometry = normalize(geometric_normal);
    var sampled: TerrainSample;
    var slope = vec3<f32>(0.0);
    for (var plane = 0u; plane < 3u; plane += 1u) {
        let weight = projection[plane];
        if weight <= 0.001 {
            continue;
        }
        var along: vec2<f32>;
        var away: vec2<f32>;
        var across: vec2<f32>;
        var down: vec2<f32>;
        switch plane {
            case 0u: { along = point.yz; away = offset.yz; across = ddx.yz; down = ddy.yz; }
            case 1u: { along = point.xz; away = offset.xz; across = ddx.xz; down = ddy.xz; }
            default: { along = point.xy; away = offset.xy; across = ddx.xy; down = ddy.xy; }
        }
        let drawn = procedural_plane(recipe, plane, along, away, across, down);
        sampled.color += vec4<f32>(max(drawn.color, vec3<f32>(0.0)), 1.0) * weight;
        sampled.surface += vec3<f32>(drawn.occlusion, drawn.roughness, 0.0) * weight;
        sampled.mask += drawn.mask * weight;
        switch plane {
            case 0u: { slope += vec3<f32>(0.0, drawn.gradient) * weight; }
            case 1u: { slope += vec3<f32>(drawn.gradient.x, 0.0, drawn.gradient.y) * weight; }
            default: { slope += vec3<f32>(drawn.gradient, 0.0) * weight; }
        }
    }
    sampled.surface = vec3<f32>(
        clamp(sampled.surface.x, 0.0, 1.0),
        clamp(sampled.surface.y, 0.02, 1.0),
        0.0,
    );
    // Relief tilts the normal against the part of its slope along the ground.
    sampled.normal = normalize(geometry - (slope - geometry * dot(slope, geometry)));
    return sampled;
}
#endif

// Applies a palette surface's look to a sampled texture layer, or to the
// procedural recipe that drew it.
fn shade_surface(sampled: TerrainSample, surface: Surface, recipe: u32) -> TerrainSample {
    var shaded = sampled;
    let strength = select(1.0, sampled.mask, surface.params.y > 0.5);
    var tinted: vec3<f32>;
    if surface.tint.w > 0.5 {
        // Recolour: keep the texture's light and shade around its mean, take
        // the hue and brightness from the tint.
        let luma = dot(sampled.color.rgb, LUMA);
#ifdef PROCEDURAL_GROUND
        let mean = select(surface.shade.x, recipe_luma(recipe), recipe != 0u);
#else
        let mean = surface.shade.x;
#endif
        tinted = surface.tint.rgb * (luma / max(mean, 0.02));
    } else {
        tinted = sampled.color.rgb * surface.tint.rgb;
    }
    shaded.color = vec4<f32>(mix(sampled.color.rgb, tinted, strength), sampled.color.a);
    shaded.surface.g = clamp(sampled.surface.g * surface.params.z, 0.02, 1.0);
    return shaded;
}

@fragment
fn fragment(
    varyings: TerrainVaryings,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var in: VertexOutput;
    in.position = varyings.position;
    in.world_position = varyings.world_position;
    in.world_normal = varyings.world_normal;
#ifdef VERTEX_UVS_A
    in.uv = varyings.uv;
#endif
#ifdef VERTEX_UVS_B
    in.uv_b = varyings.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    in.instance_index = varyings.instance_index;
#endif
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    let coordinates = vec3<f32>(varyings.uv.x, varyings.uv_b.x, varyings.uv.y);
    // How much ground one pixel spans, in material repeats. Taken here, in
    // uniform control flow, for the projections and procedural surfaces alike.
    let ddx = dpdx(coordinates);
    let ddy = dpdy(coordinates);
    let footprint = max(length(ddx), length(ddy));
    let projection = footprint_adjusted_projection(
        footprint,
        projection_weights(normalize(pbr_input.world_normal)),
    );

    var weights = array<f32, 8>(
        varyings.weights_low.x, varyings.weights_low.y,
        varyings.weights_low.z, varyings.weights_low.w,
        varyings.weights_high.x, varyings.weights_high.y,
        varyings.weights_high.z, varyings.weights_high.w,
    );
    var sum = 0.0;
    var dominant = 0.0;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        weights[slot] = max(weights[slot], 0.0);
        sum += weights[slot];
    }
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        weights[slot] /= max(sum, 0.0001);
        dominant = max(dominant, weights[slot]);
    }

    // Interpolated weights ramp linearly across whichever triangles straddle a
    // surface boundary, so the transition is as wide as those triangles happen
    // to look. Take the tallest surface and keep only what stands within a
    // fixed depth of it, so the transition is the same width whatever the
    // ground does, and let the noise decide the crumbs along it. Vertices hold
    // one-hot weights, so a fragment standing wholly on one surface is already
    // its own answer and pays for none of this.
    if dominant < 1.0 {
        let point = coordinates * (TEXTURE_REPEAT_METRES / BOUNDARY_FEATURE_METRES);
        var heights: array<f32, 8>;
        var tallest = -1.0;
        for (var slot = 0u; slot < SLOTS; slot += 1u) {
            heights[slot] = boundary_height(weights[slot], point, f32(slot) * 13.7);
            tallest = max(tallest, heights[slot]);
        }
        let standing = tallest - BOUNDARY_BLEND_DEPTH;
        var shares = 0.0;
        for (var slot = 0u; slot < SLOTS; slot += 1u) {
            weights[slot] = max(heights[slot] - standing, 0.0);
            shares += weights[slot];
        }
        for (var slot = 0u; slot < SLOTS; slot += 1u) {
            weights[slot] /= shares;
        }
    }

    // The wetness map where the fragment lies on it, near the ground it was
    // measured at: a cave under a wet field stays dry.
    let wet_uv = (varyings.world_position.xz - wet_window.xy) / max(wet_window.z, 1.0e-3);
    var wet = vec4<f32>(0.0);
    var near = 0.0;
    if all(wet_uv > vec2<f32>(0.0)) && all(wet_uv < vec2<f32>(1.0)) {
        wet = sample_wetness(wet_uv);
        near = 1.0 - smoothstep(0.15, 0.4, abs(varyings.world_position.y - (wet.g + wet_window.w)));
    }
    let wilt = clamp(wet.b, 0.0, 1.0) * near;

    var base_color = vec4<f32>(0.0);
    var surface = vec3<f32>(0.0);
    var mapped_normal = vec3<f32>(0.0);
    // The shaping above returns an exact zero for every surface the fragment
    // does not stand on, so those skip all their triplanar lookups.
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        let weight = weights[slot];
        if weight <= 0.0 {
            continue;
        }
        let look = surfaces[slot_surface(varyings.slots, slot)];
        let repeat = max(look.params.w, 0.05);
        var drawn: TerrainSample;
#ifdef PROCEDURAL_GROUND
        let recipe = u32(look.shade.y);
        if recipe != 0u {
            // The cache holds one field space, so procedural layers ignore
            // the surface's repeat.
            drawn = procedural_layer(
                recipe,
                coordinates * TEXTURE_REPEAT_METRES,
                varyings.world_position.xyz - view.world_position,
                ddx * TEXTURE_REPEAT_METRES,
                ddy * TEXTURE_REPEAT_METRES,
                projection,
                pbr_input.world_normal,
            );
            if recipe == RECIPE_STONE {
                // Smooth, nonperiodic distortion breaks the mineral tile's
                // visible grid without discontinuities or extra texture taps.
                let point = coordinates / repeat;
                let drift = value_noise_gradient(point * 0.24 + vec3<f32>(13.7, 4.1, 91.3));
                let direction = vec3<f32>(0.8, 0.584, -0.488);
                let across = ddx / repeat;
                let down = ddy / repeat;
                let warped = point + direction * (drift.x - 0.5);
                let warped_x = across + direction * (0.24 * dot(drift.yzw, across));
                let warped_y = down + direction * (0.24 * dot(drift.yzw, down));
                let mineral = sample_layer(
                    i32(look.params.x),
                    false,
                    warped,
                    warped_x,
                    warped_y,
                    projection,
                    pbr_input.world_normal,
                );
                // Keep the procedural mean used by palette recolouring while
                // preserving mineral colour, roughness and small-scale relief.
                drawn.color = vec4<f32>(
                    drawn.color.rgb * mineral.color.rgb / max(look.shade.x, 0.02),
                    mineral.color.a,
                );
                drawn.surface = mineral.surface;
                drawn.normal = mineral.normal;
            }
        } else {
#else
        let recipe = 0u;
        {
#endif
            drawn = sample_layer(
                i32(look.params.x),
                look.params.y > 0.5 && any(look.tint.rgb != vec3<f32>(1.0)),
                coordinates / repeat,
                ddx / repeat,
                ddy / repeat,
                projection,
                pbr_input.world_normal,
            );
        }
        var sampled = shade_surface(drawn, look, recipe);
        if i32(look.params.x) == GRASS_LAYER && wilt > 0.0 {
            // Wilting grass yellows to straw, keeping its light and shade.
            let straw = dot(sampled.color.rgb, LUMA) * STRAW;
            sampled.color = vec4<f32>(mix(sampled.color.rgb, straw, wilt), sampled.color.a);
        }
        base_color += sampled.color * weight;
        surface += sampled.surface * weight;
        mapped_normal += sampled.normal * weight;
    }
    // Wet ground is darker and glossier, where water has soaked in or runs
    // over it.
    var roughness = surface.g;
    if near > 0.0 {
        // Noise moves the fringe in and out, most where it is half wet.
        let shown = clamp(wet.r, 0.0, 1.0);
        let point = coordinates * (TEXTURE_REPEAT_METRES / WET_EDGE_FEATURE_METRES);
        let ragged = WET_EDGE_NOISE_STRENGTH * (value_noise(point + vec3<f32>(71.3)) - 0.5);
        let wetness = clamp(shown + ragged * 4.0 * shown * (1.0 - shown), 0.0, 1.0) * near;
        base_color = vec4<f32>(base_color.rgb * mix(1.0, 0.5, wetness), base_color.a);
        roughness = mix(roughness, 0.25, wetness);
    }
    pbr_input.material.base_color = base_color;
    pbr_input.diffuse_occlusion = vec3<f32>(surface.r);
    pbr_input.specular_occlusion = surface.r;
    pbr_input.material.perceptual_roughness = roughness;
    pbr_input.material.metallic = surface.b;

    pbr_input.N = normalize(mapped_normal);

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material,
        pbr_input.material.base_color,
    );
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = vec4<f32>(sediment_heatmap(varyings.world_position.xyz, out.color.rgb), out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
