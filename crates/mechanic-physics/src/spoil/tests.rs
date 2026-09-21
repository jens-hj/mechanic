use bevy_math::{DQuat, DVec3, Vec3};
use mechanic_core::{CompiledCreation, MaterialProperties, RuntimeBox, TICK_SECONDS};
use mechanic_world::{
    ClumpCollection, MaterialClump, TerrainField, TerrainMaterial, TerrainOctree, WorldPosition,
    WorldSeed,
};

use super::{SpoilMachine, SpoilSolver, spoil_radius};
use crate::{BodyPose, SpatialMotion};

const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

fn clump(id: u64, material: TerrainMaterial, cells: u32, position: DVec3) -> MaterialClump {
    let quanta = 510 * cells;
    MaterialClump {
        id,
        material,
        quanta,
        half_extents: DVec3::splat(
            (f64::from(quanta) * mechanic_world::MATERIAL_QUANTUM_M3).cbrt() * 0.5,
        ),
        position: WorldPosition(position),
        rotation: DQuat::IDENTITY,
        linear_velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
        settled_seconds: 0.0,
        sleeping: false,
    }
}

fn collection(bodies: impl IntoIterator<Item = MaterialClump>) -> ClumpCollection {
    let bodies = bodies
        .into_iter()
        .map(|body| (body.id, body))
        .collect::<std::collections::BTreeMap<_, _>>();
    ClumpCollection {
        next_id: bodies.keys().max().map_or(1, |id| id + 1),
        bodies,
    }
}

// A spot on generated ground and the height of its surface.
fn ground() -> (TerrainField, TerrainOctree, DVec3) {
    let field = TerrainField::new(WorldSeed(84));
    let spawn = field.safe_spawn().0;
    let surface = field.surface_height(spawn.x, spawn.z);
    (
        field,
        TerrainOctree::default(),
        DVec3::new(spawn.x, surface, spawn.z),
    )
}

// One 1 m × 10 cm × 1 m steel plate, centred at `centre` and moving at `velocity`.
fn plate(centre: DVec3, velocity: DVec3) -> SpoilMachine {
    let creation = CompiledCreation::default()
        .with_runtime_boxes(&[RuntimeBox {
            half_extents: Vec3::new(0.5, 0.05, 0.5),
            mass: 100.0,
            material: MaterialProperties {
                density_kg_m3: 7_800.0,
                static_friction: 0.8,
                dynamic_friction: 0.6,
                restitution: 0.0,
                rolling_resistance: 0.0,
                youngs_modulus_pa: 1e8,
            },
        }])
        .unwrap();
    SpoilMachine::new(
        &creation,
        &[BodyPose {
            position: centre,
            rotation: DQuat::IDENTITY,
        }],
        &[SpatialMotion {
            linear: velocity,
            angular: DVec3::ZERO,
        }],
        DVec3::ZERO,
    )
}

#[test]
fn spoil_dropped_on_ground_comes_to_rest_on_it() {
    let (field, terrain, spot) = ground();
    let mut clumps = collection([clump(1, TerrainMaterial::Soil, 1, spot + DVec3::Y * 0.5)]);
    let mut solver = SpoilSolver::default();
    for _ in 0..180 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &SpoilMachine::default(),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let body = &clumps.bodies[&1];
    let surface = field.surface_height(body.position.0.x, body.position.0.z);
    let bottom = body.position.0.y - spoil_radius(body.quanta);
    assert!(
        (bottom - surface).abs() < 0.04,
        "bottom {bottom} surface {surface}"
    );
    assert!(body.linear_velocity.length() < 0.05);
    assert!(body.settled_seconds > 0.5);
}

#[test]
fn spoil_inside_solid_ground_is_pushed_out_not_lost() {
    let (field, terrain, spot) = ground();
    let mut clumps = collection([clump(1, TerrainMaterial::Rock, 1, spot - DVec3::Y * 0.4)]);
    let mut solver = SpoilSolver::default();
    for _ in 0..240 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &SpoilMachine::default(),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let body = &clumps.bodies[&1];
    let surface = field.surface_height(body.position.0.x, body.position.0.z);
    assert!(
        body.position.0.y > surface - 0.02,
        "{} under {surface}",
        body.position.0.y
    );
    assert!(body.position.0.y < surface + 0.2);
}

#[test]
fn a_moving_blade_carries_spoil_with_it() {
    let (field, terrain, spot) = ground();
    let start = spot + DVec3::Y * 5.0;
    let velocity = DVec3::X * 0.5;
    let mut clumps = collection([clump(1, TerrainMaterial::Soil, 1, start + DVec3::Y * 0.09)]);
    let mut solver = SpoilSolver::default();
    for tick in 0..60 {
        let machine = plate(start + velocity * f64::from(tick) * TICK_SECONDS, velocity);
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &machine,
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let body = &clumps.bodies[&1];
    assert!(
        (body.position.0.x - start.x - 0.5).abs() < 0.1,
        "{}",
        body.position.0.x - start.x
    );
    assert!((body.position.0.y - start.y - 0.05 - spoil_radius(510)).abs() < 0.01);
}

#[test]
fn spoil_in_a_bucket_weighs_on_the_machine() {
    let (field, terrain, spot) = ground();
    let centre = spot + DVec3::Y * 5.0;
    let resting = centre + DVec3::new(0.2, 0.05 + spoil_radius(510 * 8), 0.0);
    let mut clumps = collection([clump(1, TerrainMaterial::Soil, 8, resting)]);
    let mass = clumps.bodies[&1].mass_kg();
    let mut solver = SpoilSolver::default();
    let mut step = super::SpoilStep::default();
    for _ in 0..30 {
        step = solver.step(
            &mut clumps,
            &terrain,
            &field,
            &plate(centre, DVec3::ZERO),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    assert_eq!(step.reactions.len(), 1);
    let reaction = step.reactions[0];
    let weight = mass * 9.81 * TICK_SECONDS;
    assert!(
        (reaction.impulse.y + weight).abs() < weight * 0.05,
        "{reaction:?} against {weight}"
    );
    // The load sits where the spoil does, not at the plate's centre.
    assert!((reaction.point.x - resting.x).abs() < 0.02);
}

#[test]
fn spoil_heaps_instead_of_stacking_in_a_column() {
    let (field, terrain, spot) = ground();
    let mut clumps = collection((0..24_u32).map(|index| {
        clump(
            u64::from(index) + 1,
            TerrainMaterial::Sand,
            1,
            // A hair off the vertical, as anything poured is.
            spot + DVec3::new(
                f64::from(index % 3) * 0.004,
                0.3 + f64::from(index) * 0.08,
                f64::from(index % 5) * 0.003,
            ),
        )
    }));
    let mut solver = SpoilSolver::default();
    for _ in 0..600 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &SpoilMachine::default(),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let radius = spoil_radius(510);
    let top = clumps
        .bodies
        .values()
        .map(|body| body.position.0.y)
        .fold(f64::MIN, f64::max);
    let surface = field.surface_height(spot.x, spot.z);
    assert!(
        top - surface < radius * 2.0 * 6.0,
        "a column {} m tall",
        top - surface
    );
    for (id, body) in &clumps.bodies {
        let ground = field.surface_height(body.position.0.x, body.position.0.z);
        assert!(body.position.0.y > ground - 0.02, "clump {id} sank");
    }
}

#[test]
fn spoil_thrown_onto_a_deck_lies_still_and_falls_asleep() {
    let (field, terrain, spot) = ground();
    let centre = spot + DVec3::Y * 5.0;
    let mut thrown = clump(1, TerrainMaterial::Soil, 2, centre + DVec3::Y * 0.4);
    thrown.linear_velocity = DVec3::new(0.6, 0.0, 0.2);
    thrown.angular_velocity = DVec3::new(3.0, 1.0, -2.0);
    let mut clumps = collection([thrown]);
    let mut solver = SpoilSolver::default();
    for _ in 0..240 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &plate(centre, DVec3::ZERO),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let body = &clumps.bodies[&1];
    assert!(
        body.sleeping,
        "still awake, turning at {:?}",
        body.angular_velocity
    );
    assert_eq!(body.angular_velocity, DVec3::ZERO);
    // A deck holds spoil; only the ground takes it back.
    assert!(!body.can_deposit());
    // It wakes when the deck moves under it.
    solver.step(
        &mut clumps,
        &terrain,
        &field,
        &plate(centre, DVec3::X),
        GRAVITY,
        TICK_SECONDS,
    );
    assert!(!clumps.bodies[&1].sleeping);
}

#[test]
fn spoil_on_rough_ground_stops_creeping_and_settles() {
    let (field, terrain, spot) = ground();
    let mut clumps = collection((0..12_u32).map(|index| {
        let mut body = clump(
            u64::from(index) + 1,
            TerrainMaterial::Soil,
            1 + index % 4,
            spot + DVec3::new(
                f64::from(index % 4) * 0.37,
                0.4,
                f64::from(index / 4) * 0.41,
            ),
        );
        body.linear_velocity = DVec3::new(0.8, 0.0, -0.5);
        body
    }));
    let mut solver = SpoilSolver::default();
    let machine = SpoilMachine::default();
    for _ in 0..150 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &machine,
            GRAVITY,
            TICK_SECONDS,
        );
    }
    let before = clumps
        .bodies
        .values()
        .map(|body| body.position.0)
        .collect::<Vec<_>>();
    for _ in 0..30 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &machine,
            GRAVITY,
            TICK_SECONDS,
        );
    }
    for (body, before) in clumps.bodies.values().zip(before) {
        assert!(
            body.position.0.distance(before) < 1e-6,
            "clump {} crept {} m",
            body.id,
            body.position.0.distance(before)
        );
        assert_eq!(body.angular_velocity, DVec3::ZERO);
        assert!(body.can_deposit(), "clump {} never settled", body.id);
    }
}

#[test]
fn soft_clods_lying_together_gather_into_one_and_keep_their_material() {
    let (field, terrain, spot) = ground();
    let mut clumps = collection(
        (0..6_u32)
            .map(|index| {
                clump(
                    u64::from(index) + 1,
                    TerrainMaterial::Soil,
                    1,
                    spot + DVec3::new(f64::from(index) * 0.03, 0.1, 0.0),
                )
            })
            .chain([clump(
                7,
                TerrainMaterial::Rock,
                1,
                spot + DVec3::new(0.09, 0.2, 0.0),
            )]),
    );
    let quanta = |clumps: &ClumpCollection| {
        clumps
            .bodies
            .values()
            .map(|body| u64::from(body.quanta))
            .sum::<u64>()
    };
    let before = quanta(&clumps);
    let mut solver = SpoilSolver::default();
    for _ in 0..120 {
        solver.step(
            &mut clumps,
            &terrain,
            &field,
            &SpoilMachine::default(),
            GRAVITY,
            TICK_SECONDS,
        );
    }
    assert_eq!(
        quanta(&clumps),
        before,
        "nothing is made or lost by gathering"
    );
    let soil = clumps
        .bodies
        .values()
        .filter(|body| body.material == TerrainMaterial::Soil)
        .count();
    assert!(soil < 6, "{soil} clods never gathered");
    assert!(clumps.bodies.values().all(MaterialClump::is_valid));
    // Hard fragments stay what they broke into.
    assert_eq!(clumps.bodies[&7].quanta, 510);
}

#[test]
fn a_clod_wedged_between_a_dirt_wall_and_a_block_settles() {
    use mechanic_world::{ExtractionCell, WorldCell};
    let (field, mut terrain, spot) = ground();
    // A trench with upright walls of undisturbed ground.
    let corner = WorldPosition(spot).cell().unwrap();
    let trench = (-12..4)
        .flat_map(|y| (0..8).flat_map(move |x| (-10..10).map(move |z| (x, y, z))))
        .map(|(x, y, z)| WorldCell::new(corner.x + x, corner.y + y, corner.z + z))
        .filter(|&cell| terrain.sample_cell(&field, cell).is_solid())
        .map(|cell| ExtractionCell {
            cell,
            sample: terrain.sample_cell(&field, cell),
            throw: DVec3::ZERO,
        })
        .collect::<Vec<_>>();
    terrain.extract_cells(&field, &trench).unwrap();
    // A steel slab leaning towards the trench wall, the gap closing downwards:
    // 16 cm at the top, nothing 80 cm down. Neither face is under the clod.
    let wall = f64::from(corner.x) * mechanic_world::TERRAIN_CELL_METERS;
    let lean = (0.16_f64 / 0.8).atan();
    let normal = DVec3::new(-lean.cos(), lean.sin(), 0.0);
    let face = DVec3::new(wall + 0.08, spot.y - 0.4, spot.z);
    let creation = CompiledCreation::default()
        .with_runtime_boxes(&[RuntimeBox {
            half_extents: Vec3::new(0.5, 0.05, 0.5),
            mass: 100.0,
            material: MaterialProperties {
                density_kg_m3: 7_800.0,
                static_friction: 0.8,
                dynamic_friction: 0.6,
                restitution: 0.0,
                rolling_resistance: 0.0,
                youngs_modulus_pa: 1e8,
            },
        }])
        .unwrap();
    let slab = SpoilMachine::new(
        &creation,
        &[BodyPose {
            position: face - normal * 0.05,
            rotation: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2 - lean),
        }],
        &[SpatialMotion {
            linear: DVec3::ZERO,
            angular: DVec3::ZERO,
        }],
        DVec3::ZERO,
    );
    let mut clumps = collection([clump(
        1,
        TerrainMaterial::Soil,
        8,
        DVec3::new(wall + 0.08, spot.y + 0.1, spot.z),
    )]);
    let mut solver = SpoilSolver::default();
    for _ in 0..240 {
        solver.step(&mut clumps, &terrain, &field, &slab, GRAVITY, TICK_SECONDS);
    }
    let body = &clumps.bodies[&1];
    let depth = spot.y - body.position.0.y;
    assert!(
        (0.0..0.5).contains(&depth),
        "not caught in the gap: {depth} m down"
    );
    assert!(body.linear_velocity.length() < 0.05, "still moving");
    assert!(body.can_deposit(), "caught, and never settled");
}
