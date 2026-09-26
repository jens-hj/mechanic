#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_functions,
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
    // Mean luminance of the layer's base colour, to keep recolours' brightness.
    shade: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var base_color_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var normal_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var orm_maps: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var tint_masks: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var terrain_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var<storage, read> surfaces: array<Surface>;

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
    coordinates: vec3<f32>,
    projection: vec3<f32>,
) -> vec3<f32> {
    // Triplanar blending is important up close. Once a fragment covers a
    // sizeable part of a material repeat, the secondary projections are no
    // longer resolvable and only multiply texture bandwidth. Fade to the
    // dominant projection so distant terrain retains every PBR map and its
    // authored mip detail while using one third as many texture samples.
    let footprint = max(length(dpdx(coordinates)), length(dpdy(coordinates)));
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

fn value_noise(point: vec3<f32>) -> f32 {
    let base = floor(point);
    let fraction = point - base;
    let weight = fraction * fraction * (3.0 - 2.0 * fraction);
    let low = mix(
        mix(
            hash_corner(base),
            hash_corner(base + vec3<f32>(1.0, 0.0, 0.0)),
            weight.x,
        ),
        mix(
            hash_corner(base + vec3<f32>(0.0, 1.0, 0.0)),
            hash_corner(base + vec3<f32>(1.0, 1.0, 0.0)),
            weight.x,
        ),
        weight.y,
    );
    let high = mix(
        mix(
            hash_corner(base + vec3<f32>(0.0, 0.0, 1.0)),
            hash_corner(base + vec3<f32>(1.0, 0.0, 1.0)),
            weight.x,
        ),
        mix(
            hash_corner(base + vec3<f32>(0.0, 1.0, 1.0)),
            hash_corner(base + vec3<f32>(1.0, 1.0, 1.0)),
            weight.x,
        ),
        weight.y,
    );
    return mix(low, high, weight.z);
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

fn sample_projection(layer: i32, uv: vec2<f32>, masked: bool) -> array<vec4<f32>, 4> {
    // Untinted and unmasked surfaces skip the mask lookup entirely.
    var mask = vec4<f32>(1.0);
    if masked {
        mask = textureSample(tint_masks, terrain_sampler, uv, layer);
    }
    return array<vec4<f32>, 4>(
        textureSample(base_color_maps, terrain_sampler, uv, layer),
        textureSample(orm_maps, terrain_sampler, uv, layer),
        textureSample(normal_maps, terrain_sampler, uv, layer),
        mask,
    );
}

// Each projection decision serves all four maps. Keep the per-projection
// normal normalization and per-surface normalization used by the full blend.
fn sample_layer(
    layer: i32,
    masked: bool,
    coordinates: vec3<f32>,
    projection: vec3<f32>,
    geometric_normal: vec3<f32>,
) -> TerrainSample {
    // Whiteout blend: every planar normal carries the surface it perturbs, so a
    // slope keeps its shading even where the projection collapses to one axis.
    let geometry = normalize(geometric_normal);
    let direction = select(vec3<f32>(-1.0), vec3<f32>(1.0), geometry >= vec3<f32>(0.0));
    var sampled: TerrainSample;
    if projection.x > 0.001 {
        let maps = sample_projection(layer, coordinates.yz, masked);
        sampled.color += maps[0] * projection.x;
        sampled.surface += maps[1].rgb * projection.x;
        sampled.mask += maps[3].r * projection.x;
        let tangent = normalize(maps[2].rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            abs(tangent.z) * geometry.x,
            tangent.x + geometry.y,
            tangent.y * direction.x + geometry.z,
        ) * projection.x;
    }
    if projection.y > 0.001 {
        let maps = sample_projection(layer, coordinates.xz, masked);
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
        let maps = sample_projection(layer, coordinates.xy, masked);
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

// Applies a palette surface's look to a sampled texture layer.
fn shade_surface(sampled: TerrainSample, surface: Surface) -> TerrainSample {
    var shaded = sampled;
    let strength = select(1.0, sampled.mask, surface.params.y > 0.5);
    var tinted: vec3<f32>;
    if surface.tint.w > 0.5 {
        // Recolour: keep the texture's light and shade around its mean, take
        // the hue and brightness from the tint.
        let luma = dot(sampled.color.rgb, LUMA);
        tinted = surface.tint.rgb * (luma / max(surface.shade.x, 0.02));
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
    let projection = footprint_adjusted_projection(
        coordinates,
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
        let sampled = shade_surface(
            sample_layer(
                i32(look.params.x),
                look.params.y > 0.5 && any(look.tint.rgb != vec3<f32>(1.0)),
                coordinates / max(look.params.w, 0.05),
                projection,
                pbr_input.world_normal,
            ),
            look,
        );
        base_color += sampled.color * weight;
        surface += sampled.surface * weight;
        mapped_normal += sampled.normal * weight;
    }
    pbr_input.material.base_color = base_color;
    pbr_input.diffuse_occlusion = vec3<f32>(surface.r);
    pbr_input.specular_occlusion = surface.r;
    pbr_input.material.perceptual_roughness = surface.g;
    pbr_input.material.metallic = surface.b;

    pbr_input.N = normalize(mapped_normal);

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material,
        pbr_input.material.base_color,
    );
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
