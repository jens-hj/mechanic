#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var grass_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var dirt_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var stone_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var sand_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var iron_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var graphite_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var grass_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var dirt_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var stone_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var grass_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var dirt_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var stone_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var sand_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var iron_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var graphite_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(15) var terrain_sampler: sampler;

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
// the scale of the crumbs a material edge breaks into, the strength how far the
// edge wanders, and the depth how wide the remaining transition is.
const BOUNDARY_FEATURE_METRES: f32 = 0.08;
const BOUNDARY_NOISE_STRENGTH: f32 = 0.3;
const BOUNDARY_BLEND_DEPTH: f32 = 0.1;

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

// A material's standing height at this fragment: its interpolated weight, moved
// by noise this material alone sees. A material with no weight here stands well
// below every present one and can never win the comparison that follows.
fn boundary_height(weight: f32, point: vec3<f32>, offset: f32) -> f32 {
    if weight <= 0.0 {
        return -1.0;
    }
    return weight + BOUNDARY_NOISE_STRENGTH * (value_noise(point + vec3<f32>(offset)) - 0.5);
}

struct TerrainSample {
    color: vec4<f32>,
    surface: vec3<f32>,
    normal: vec3<f32>,
}

// Each projection decision serves all three maps. Keep the per-projection
// normal normalization and per-material normalization used by the full blend.
fn sample_material(
    color_map: texture_2d<f32>,
    surface_map: texture_2d<f32>,
    normal_map: texture_2d<f32>,
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
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.yz) * projection.x;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.yz).rgb * projection.x;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.yz).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            abs(tangent.z) * geometry.x,
            tangent.x + geometry.y,
            tangent.y * direction.x + geometry.z,
        ) * projection.x;
    }
    if projection.y > 0.001 {
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.xz) * projection.y;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.xz).rgb * projection.y;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.xz).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x + geometry.x,
            abs(tangent.z) * geometry.y,
            -tangent.y * direction.y + geometry.z,
        ) * projection.y;
    }
    if projection.z > 0.001 {
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.xy) * projection.z;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.xy).rgb * projection.z;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.xy).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x * direction.z + geometry.x,
            tangent.y + geometry.y,
            abs(tangent.z) * geometry.z,
        ) * projection.z;
    }
    sampled.normal = normalize(sampled.normal);
    return sampled;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    let coordinates = vec3<f32>(in.uv.x, in.uv_b.x, in.uv.y);
    let projection = footprint_adjusted_projection(
        coordinates,
        projection_weights(normalize(pbr_input.world_normal)),
    );
    var material_weights_a = max(in.color, vec4<f32>(0.0));
    var material_weights_b = vec2<f32>(
        max(in.uv_b.y, 0.0),
        max(1.0 - dot(material_weights_a, vec4<f32>(1.0)) - max(in.uv_b.y, 0.0), 0.0),
    );
    let material_weight_sum = dot(material_weights_a, vec4<f32>(1.0))
        + dot(material_weights_b, vec2<f32>(1.0));
    material_weights_a /= max(material_weight_sum, 0.0001);
    material_weights_b /= max(material_weight_sum, 0.0001);

    // Interpolated weights ramp linearly across whichever triangles straddle a
    // material boundary, so the transition is as wide as those triangles happen
    // to look: a hard line on a deposit's steep skirt, a smear where the ground
    // is shallow. Take the tallest material and keep only what stands within a
    // fixed depth of it, so the transition is the same width whatever the
    // surface does, and let the noise decide the crumbs along it. Vertices hold
    // one-hot weights, so a fragment standing wholly on one material is already
    // its own answer and pays for none of this.
    let dominant = max(
        max(max(material_weights_a.x, material_weights_a.y),
            max(material_weights_a.z, material_weights_a.w)),
        max(material_weights_b.x, material_weights_b.y),
    );
    if dominant < 1.0 {
        let point = coordinates * (TEXTURE_REPEAT_METRES / BOUNDARY_FEATURE_METRES);
        let heights_a = vec4<f32>(
            boundary_height(material_weights_a.x, point, 0.0),
            boundary_height(material_weights_a.y, point, 13.0),
            boundary_height(material_weights_a.z, point, 27.0),
            boundary_height(material_weights_a.w, point, 41.0),
        );
        let heights_b = vec2<f32>(
            boundary_height(material_weights_b.x, point, 55.0),
            boundary_height(material_weights_b.y, point, 69.0),
        );
        let tallest = max(
            max(max(heights_a.x, heights_a.y), max(heights_a.z, heights_a.w)),
            max(heights_b.x, heights_b.y),
        );
        let standing = tallest - BOUNDARY_BLEND_DEPTH;
        let shares_a = max(heights_a - vec4<f32>(standing), vec4<f32>(0.0));
        let shares_b = max(heights_b - vec2<f32>(standing), vec2<f32>(0.0));
        let share_sum = dot(shares_a, vec4<f32>(1.0)) + dot(shares_b, vec2<f32>(1.0));
        material_weights_a = shares_a / share_sum;
        material_weights_b = shares_b / share_sum;
    }

    var base_color = vec4<f32>(0.0);
    var surface = vec3<f32>(0.0);
    var mapped_normal = vec3<f32>(0.0);
    // Mesh vertices carry one-hot weights, and the shaping above returns an
    // exact zero for every material the fragment does not stand on, so avoid
    // those three triplanar map lookups.
    if material_weights_a.x > 0.0 {
        let sampled = sample_material(
            grass_base_color, grass_orm, grass_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.x;
        surface += sampled.surface * material_weights_a.x;
        mapped_normal += sampled.normal * material_weights_a.x;
    }
    if material_weights_a.y > 0.0 {
        let sampled = sample_material(
            dirt_base_color, dirt_orm, dirt_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.y;
        surface += sampled.surface * material_weights_a.y;
        mapped_normal += sampled.normal * material_weights_a.y;
    }
    if material_weights_a.z > 0.0 {
        let sampled = sample_material(
            stone_base_color, stone_orm, stone_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.z;
        surface += sampled.surface * material_weights_a.z;
        mapped_normal += sampled.normal * material_weights_a.z;
    }
    if material_weights_a.w > 0.0 {
        let sampled = sample_material(
            sand_base_color, sand_orm, dirt_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.w;
        surface += sampled.surface * material_weights_a.w;
        mapped_normal += sampled.normal * material_weights_a.w;
    }
    if material_weights_b.x > 0.0 {
        let sampled = sample_material(
            iron_base_color, iron_orm, stone_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_b.x;
        surface += sampled.surface * material_weights_b.x;
        mapped_normal += sampled.normal * material_weights_b.x;
    }
    if material_weights_b.y > 0.0 {
        let sampled = sample_material(
            graphite_base_color, graphite_orm, stone_normal,
            coordinates, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_b.y;
        surface += sampled.surface * material_weights_b.y;
        mapped_normal += sampled.normal * material_weights_b.y;
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
