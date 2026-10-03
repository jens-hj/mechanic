//! Authored rock geometry follows the same sampling, editing and meshing path as ground.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_math::{DVec3, IVec3};

use super::super::compile::{Scope, compile};
use super::super::spec::Expr;
use super::super::tape::Tape;
use super::super::{CompiledWorld, Lattice, TerrainField, WorldgenSpec};
use crate::{
    TerrainDensityClass, TerrainMaterial, TerrainMeshRequest, TerrainNodeId, TerrainOctree,
    TerrainTransitionMask, WorldPosition, WorldSeed, mesh_chunk, select_active_nodes,
};

// Edited in-memory definitions retain their source digest; compile them directly
// so fixtures neither reuse nor pollute the production seed/digest cache.
fn fixture_field(seed: WorldSeed, spec: WorldgenSpec) -> TerrainField {
    TerrainField {
        seed,
        version: crate::WorldGeneratorVersion::CURRENT,
        world: Arc::new(CompiledWorld::new(seed, &spec).unwrap()),
        spec: Arc::new(spec),
    }
}

fn expression(spec: &WorldgenSpec, biome: usize, expr: &Expr, seed: u64) -> Tape {
    let mut local = spec.biomes[biome].definitions.clone();
    local.insert("height".into(), spec.biomes[biome].height.clone());
    compile(
        expr,
        Scope {
            local: &local,
            library: &spec.library.definitions,
            fields: None,
        },
        &[],
        seed,
        "rock test",
    )
    .unwrap()
}

#[test]
fn weathered_rocks_have_bounded_varied_surfaces() {
    let spec = WorldgenSpec::embedded();
    let local = BTreeMap::new();
    let shape = compile(
        &Expr::Ref("weathered_rock".into()),
        Scope {
            local: &local,
            library: &spec.library.definitions,
            fields: None,
        },
        &["r", "cut", "phase", "width", "rise", "depth", "lean"].map(str::to_owned),
        42,
        "weathered rock",
    )
    .unwrap();
    let mut changed = 0;
    for phase in [0.0, 19.0, 137.0] {
        let vars = [2.0, 0.8, phase, 1.2, 0.9, 1.0, 0.1];
        assert!(shape.eval([0.0; 3], &vars) > 0.0);
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut outside = [0.0; 3];
                outside[axis] = sign * 4.0;
                assert!(shape.eval(outside, &vars) < 0.0);
            }
        }
        for y in -4..4 {
            let point = [2.0, f64::from(y) * 0.2, 0.0];
            let reference = [2.0, 0.8, 0.0, 1.2, 0.9, 1.0, 0.1];
            changed += usize::from(
                (shape.eval(point, &vars) - shape.eval(point, &reference)).abs() > 0.01,
            );
        }
    }
    assert!(changed > 5, "rocks must not repeat the same relief");
    // Authored reach culls samples and streamed nodes. The largest stretched
    // variants must fit inside the declared sphere.
    for (reach, vars) in [
        (6.0, [2.7, 0.96, 137.0, 1.4, 1.1, 1.2, 0.2]),
        (13.0, [5.0, 0.96, 137.0, 1.3, 1.9, 1.25, 0.18]),
    ] {
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let direction = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    if direction.length_squared() > 0.0 {
                        assert!(
                            shape.eval((direction.normalize() * reach).to_array(), &vars) < 0.0
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn scatter_ground_and_mask_use_the_surrounding_terrain_seed() {
    let mut spec = (*WorldgenSpec::embedded()).clone();
    let hills = spec
        .biomes
        .iter()
        .position(|b| b.name == "verdant_hills")
        .unwrap();
    let Expr::Scatter(scatter) = spec.biomes[hills].definitions.get_mut("boulders").unwrap() else {
        panic!()
    };
    scatter.chance = 1.0;
    scatter.jitter = 0.0;
    scatter.lift = (0.0, 0.0);
    scatter.tilt = 0.0;
    scatter.shape = Expr::Sphere(Box::new(Expr::C(0.5)));
    scatter.mask = Some(Expr::Sub(
        Box::new(Expr::Ref("height".into())),
        Box::new(Expr::C(14.0)),
    ));
    for seed in [7, 42, 91] {
        let height = expression(&spec, hills, &Expr::Ref("height".into()), seed);
        let rocks = expression(&spec, hills, &Expr::Ref("boulders".into()), seed);
        let mut present = 0;
        let mut absent = 0;
        for i in -8..8 {
            let x = f64::from(i) * 36.0 + 18.0;
            let y = height.eval([x, 0.0, 18.0], &[]);
            let density = rocks.eval([x, y, 18.0], &[]);
            if y > 14.0 {
                present += 1;
                assert!(
                    density > 0.49,
                    "seed {seed}: rock did not follow its ground"
                );
                assert!(rocks.eval([x, y + 1.0, 18.0], &[]) < 0.0);
            } else {
                absent += 1;
                assert!(density < 0.0, "mask must read the same hills");
            }
        }
        assert!(present > 0 && absent > 0);
    }
}

#[test]
fn cliff_blocks_grow_from_edges_and_leave_flat_treads_open() {
    let mut spec = (*WorldgenSpec::embedded()).clone();
    let crags = spec
        .biomes
        .iter()
        .position(|b| b.name == "titan_crags")
        .unwrap();
    // A predictable ramp isolates the shipped terrace and placement rules.
    spec.biomes[crags].height =
        Expr::Add(vec![Expr::C(100.0), Expr::Mul(vec![Expr::X, Expr::C(0.8)])]);
    let edge = expression(&spec, crags, &Expr::Ref("cliff_edge".into()), 42);
    let ledge = expression(&spec, crags, &Expr::Ref("ledge".into()), 42);
    let rocks = expression(&spec, crags, &Expr::Ref("edge_blocks".into()), 42);
    let mut protrusions = 0;
    let mut flats = 0;
    for x in -100..100 {
        let x = f64::from(x);
        let y = ledge.eval([x, 0.0, 0.0], &[]);
        flats += usize::from(edge.eval([x, 0.0, 0.0], &[]) < 0.0);
        for z in -20..20 {
            if rocks.eval([x, y + 2.0, f64::from(z)], &[]) > 0.0 {
                protrusions += 1;
            }
        }
    }
    assert!(flats > 20);
    assert!(
        protrusions > 20,
        "no exposed rocky formations at cliff edges"
    );
    spec.biomes[crags].height = Expr::C(104.0);
    let flat_rocks = expression(&spec, crags, &Expr::Ref("edge_blocks".into()), 42);
    for x in -30..30 {
        assert!(flat_rocks.eval([f64::from(x), 105.0, 0.0], &[]) < 0.0);
    }
}

#[test]
fn boulders_are_rock_that_meshes_and_excavates_through_the_terrain_octree() {
    let mut spec = (*WorldgenSpec::embedded()).clone();
    spec.biomes.retain(|b| b.name == "verdant_hills");
    spec.biomes[0].height = Expr::C(20.0);
    spec.biomes[0].flora.clear();
    spec.biomes[0].rivers = 0.0;
    spec.biomes[0].carves = spec
        .world
        .carves
        .iter()
        .map(|c| (c.name.clone(), 0.0))
        .collect();
    let field = fixture_field(WorldSeed(42), spec);
    let point = (-100..100)
        .find_map(|x| {
            (-100..100).find_map(|z| {
                let point = WorldPosition(DVec3::new(f64::from(x), 20.6, f64::from(z)));
                (field.sample_position(point).density > 0.25).then_some(point)
            })
        })
        .expect("the authored scatter must produce exposed boulders");
    let cell = point.cell().unwrap();
    assert_eq!(field.sample_cell(cell).material, TerrainMaterial::Rock);
    let top = field.topmost_surface(point.0.x, point.0.z).unwrap();
    let surface = WorldPosition(DVec3::new(point.0.x, top - 0.02, point.0.z));
    assert_eq!(
        field.sample_position(surface).material,
        TerrainMaterial::Rock
    );
    assert_eq!(
        field.classify(surface.0 - DVec3::splat(0.3), surface.0 + DVec3::splat(0.3)),
        TerrainDensityClass::Mixed
    );
    let surface_cell = surface.cell().unwrap();
    let lattice = Lattice {
        origin: IVec3::new(surface_cell.x, surface_cell.y, surface_cell.z),
        stride: 2,
        dims: [4; 3],
        centred: false,
    };
    let densities = field.density_lattice(&lattice);
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..4 {
                let p = DVec3::new(
                    lattice.coordinate(0, i),
                    lattice.coordinate(1, j),
                    lattice.coordinate(2, k),
                );
                assert!((densities[i + 4 * (j + 4 * k)] - field.density(p)).abs() < 1.0e-9);
            }
        }
    }
    let mut terrain = TerrainOctree::default();
    let before = terrain.snapshot();
    let request = TerrainMeshRequest {
        node: TerrainNodeId::leaf(surface_cell.brick()),
        generation: 0,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let before_cut = select_active_nodes(&field, &before, point);
    let owner = |node: &&crate::ActiveTerrainNode| {
        TerrainNodeId::containing(cell.brick(), node.id.level) == Some(node.id)
    };
    let old = before_cut
        .iter()
        .find(owner)
        .expect("rock must be streamed");
    let mesh = mesh_chunk(&field, &before, request);
    assert!(!mesh.vertices.is_empty());
    let outcome = terrain.excavate_sphere(&field, point, 0.25).unwrap();
    assert!(!outcome.changed_brick_coordinates().is_empty());
    let after_cut = select_active_nodes(&field, &terrain.snapshot(), point);
    let new = after_cut
        .iter()
        .find(owner)
        .expect("edited rock must stay streamed");
    assert!(
        new.generation > old.generation,
        "excavation must invalidate the streamed mesh"
    );
    assert!(!terrain.sample_cell(&field, cell).is_solid());
    assert!(before.sample_cell(&field, cell).is_solid());
    assert!(
        terrain
            .node(TerrainNodeId::leaf(cell.brick()))
            .unwrap()
            .latest_revision
            > 0
    );
}

#[test]
fn shipped_boulders_and_cliff_formations_survive_world_composition() {
    for seed in [42, 7] {
        let spec = WorldgenSpec::embedded();
        let field = TerrainField::from_spec(WorldSeed(seed), Arc::clone(&spec)).unwrap();
        let mut bare_spec = (*spec).clone();
        for biome in &mut bare_spec.biomes {
            for name in ["boulders", "edge_blocks"] {
                if biome.definitions.contains_key(name) {
                    biome.definitions.insert(name.into(), Expr::C(-1_000.0));
                }
            }
        }
        let bare = fixture_field(WorldSeed(seed), bare_spec);
        for name in ["verdant_hills", "titan_crags"] {
            let (cx, cz) = super::heart_of(&field, name);
            let example = (-60..60)
                .find_map(|i| {
                    (-60..60).find_map(|k| {
                        let (x, z) = (cx + f64::from(i) * 2.0, cz + f64::from(k) * 2.0);
                        if field.biome_at(x, z) != name {
                            return None;
                        }
                        let ground = bare.surface_height(x, z);
                        (1..20).find_map(|dy| {
                            let point = DVec3::new(x, ground + f64::from(dy) * 0.5, z);
                            (field.density(point) > 0.3
                                && bare.density(point) < -0.3
                                && field.sample_position(WorldPosition(point)).material
                                    == TerrainMaterial::Rock)
                                .then_some(point)
                        })
                    })
                })
                .unwrap_or_else(|| panic!("seed {seed}: no exposed {name} rocks"));
            eprintln!("rock example seed={seed} biome={name} point={example}");
        }
    }
}
