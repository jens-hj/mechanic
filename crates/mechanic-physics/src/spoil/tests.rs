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
