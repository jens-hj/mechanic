//! Grass: it drowns under standing water, wears under a current, is
//! smothered by silt, and grows back where the seed grows grass.

use bevy_math::DVec3;

use super::{Ground, room};
use crate::water::ground::CELL_TERRAIN_CELLS;
use crate::water::{
    ErosionConfig, StoredWaterDoc, WaterCell, WaterGround, WaterNetwork, WaterWorld,
};
use crate::{
    BrickCoord, LakeBasin, RiverReach, SedimentApplied, SedimentChange, SurfaceId, TerrainMaterial,
    WaterSurface,
};

/// A 1 m box open at the top, its floor at -1 m a meadow: grass 18 cm deep
/// over soil, under `silt` metres of silt.
struct Meadow {
    ground: Ground,
    silt: f64,
}

/// The meadow's floor, the top of its grass.
const FLOOR: f64 = -1.0;

fn meadow(silt: f64) -> Meadow {
    Meadow {
        ground: Ground {
            rooms: vec![room([0.0, FLOOR, 0.0], [1.0, 1.0, 1.0])],
            lake: None,
            river: None,
            rock: false,
        },
        silt,
    }
}

impl WaterNetwork for Meadow {
    fn lake(&self, lake: u32) -> Option<LakeBasin> {
        self.ground.lake(lake)
    }

    fn reach(&self, reach: u32) -> Option<RiverReach> {
        self.ground.reach(reach)
    }
}

impl WaterGround for Meadow {
    fn open_cells(&self, cell: WaterCell) -> [bool; CELL_TERRAIN_CELLS] {
        self.ground.open_cells(cell)
    }

    fn implicit(&self, point: DVec3) -> Option<WaterSurface> {
        self.ground.implicit(point)
    }

    fn surface(&self, x: f64, z: f64) -> Option<WaterSurface> {
        self.ground.surface(x, z)
    }

    fn may_hold_water(&self, brick: BrickCoord) -> bool {
        self.ground.may_hold_water(brick)
    }

    fn edited(&self, brick: BrickCoord) -> bool {
        self.ground.edited(brick)
    }

    fn dug(&self, cell: WaterCell) -> bool {
        self.ground.dug(cell)
    }

    fn ground_top(&self, x: f64, z: f64, from: f64, reach: f64) -> Option<f64> {
        self.ground.ground_top(x, z, from, reach)
    }

    fn material(&self, point: DVec3) -> Option<TerrainMaterial> {
        self.ground.material(point)?;
        Some(if point.y > FLOOR - self.silt {
            TerrainMaterial::Soil
        } else if point.y > FLOOR - self.silt - 0.18 {
            TerrainMaterial::SurfaceCover
        } else {
            TerrainMaterial::Soil
        })
    }

    fn native_grass(&self, x: f64, z: f64, _top: f64) -> crate::NativeGrass {
        // The seed grew grass at the floor, whatever lies on it since.
        let grows = self.material(DVec3::new(x, FLOOR - self.silt - 0.02, z))
            == Some(TerrainMaterial::SurfaceCover);
        crate::NativeGrass::plain(grows)
    }
}

/// The meadow's middle column.
const MIDDLE: (i32, i32) = (2, 2);

/// Answers the water's asks as the ground would, and returns them: every
/// take gives all it asks, every lay lays all it is given, and every change
/// of what grows turns the square's 16 columns.
fn answer(water: &mut WaterWorld) -> Vec<SedimentChange> {
    let asks = water.sediment_requests();
    let applied = asks
        .iter()
        .map(|change| {
            let mut done = SedimentApplied::default();
            match change.quanta {
                0 => done.relabelled = 16,
                ..0 => {
                    done.taken[TerrainMaterial::Soil.code() as usize] =
                        change.quanta.unsigned_abs();
                }
                _ => done.laid = change.quanta.unsigned_abs(),
            }
            done
        })
        .collect::<Vec<_>>();
    water.sediment_applied(&applied);
    asks
}

/// Runs the water `seconds` over `ground`, answering its asks, and returns
/// every change of what grows it asked for.
fn run(water: &mut WaterWorld, ground: &impl WaterGround, seconds: u32) -> Vec<SedimentChange> {
    let mut turned = Vec::new();
    for _ in 0..seconds * 20 {
        water.step(ground, 0.05);
        turned.extend(
            answer(water)
                .into_iter()
                .filter(|change| change.quanta == 0),
        );
    }
    turned
}

/// The asks that turned the meadow's middle column.
fn middle(turned: &[SedimentChange]) -> Vec<SedimentChange> {
    turned
        .iter()
        .filter(|change| (change.x / 4, change.z / 4) == MIDDLE)
        .copied()
        .collect()
}

/// Weeks of grass's life in seconds.
const SPEED: f64 = 100_000.0;

#[test]
fn grass_under_standing_water_dies_and_turns_to_its_soil() {
    let ground = meadow(0.0);
    let mut water = WaterWorld::new();
    water.set_erosion(ErosionConfig { speed: SPEED });
    water.terrain_changed(&ground, ground.ground.bricks());
    // Enough to soak the ground and stand 30 cm deep over it.
    water.deposit(&ground, DVec3::new(0.5, 0.5, 0.5), 0.6);
    let turned = run(&mut water, &ground, 5);
    let health = water.grass_health(MIDDLE);
    assert!(health > 0.0 && health < 1.0, "grass wilting: {health}");
    // Two weeks under water, in about 12 s.
    let turned = [turned, run(&mut water, &ground, 20)].concat();
    let died = middle(&turned);
    assert_eq!(died.len(), 1, "the grass died once: {died:?}");
    assert_eq!(died[0].material, TerrainMaterial::Soil);
    assert_eq!(died[0].look, Some(SurfaceId::plain(TerrainMaterial::Soil)));
    assert!(water.grass_health(MIDDLE) <= 0.0);
    // Still under water, the dead ground grows nothing back.
    assert!(middle(&run(&mut water, &ground, 60)).is_empty());
}

#[test]
fn a_flood_wears_turf_through_in_hours_while_a_gentle_sheet_spares_it() {
    let ground = meadow(0.0);
    let cell = WaterCell::new(MIDDLE.0, -5, MIDDLE.1);
    let hour = 3_600.0;
    // A sheet 2 cm deep dragging at 6 Pa for a day.
    let mut gentle = WaterWorld::new();
    gentle.wet_grass(&ground, cell, FLOOR, 0.02, 6.0, 24.0 * hour);
    let spared = gentle.grass_health(MIDDLE);
    assert!(spared > 0.9, "a gentle sheet left {spared}");
    // A flood half a metre deep dragging at 130 Pa for twelve hours.
    let mut flood = WaterWorld::new();
    flood.wet_grass(&ground, cell, FLOOR, 0.5, 130.0, 12.0 * hour);
    let worn = flood.grass_health(MIDDLE);
    assert!(worn <= 0.0, "a flood left {worn}");
}

#[test]
fn silt_laid_over_living_grass_is_grown_through() {
    let mut water = WaterWorld::new();
    water.set_erosion(ErosionConfig { speed: SPEED });
    let ground = meadow(0.0);
    let cell = WaterCell::new(MIDDLE.0, -5, MIDDLE.1);
    // Water once reached it, and five centimetres of silt came to lie on
    // it: grass a column wide smothered halfway.
    water.wet_grass(&ground, cell, FLOOR, 0.0, 0.0, 0.0);
    let quanta = 0.05 * 0.04 / crate::MATERIAL_QUANTUM_M3;
    water.grass_buried(MIDDLE, quanta);
    let health = water.grass_health(MIDDLE);
    assert!((health - 0.5).abs() < 0.01, "silt left {health}");
    // A week, in about six seconds, and it is grass on top again once the
    // ground is next asked.
    let silted = meadow(0.05);
    let turned = middle(&run(&mut water, &silted, 15));
    assert_eq!(turned.len(), 1, "grew through once: {turned:?}");
    assert_eq!(turned[0].material, TerrainMaterial::SurfaceCover);
    assert!(water.grass_health(MIDDLE) > health, "and it recovers");
}

/// Grass torn from the meadow's middle column of `ground`, and six weeks
/// after, in about 36 s: the changes of what grows it asked for.
fn after_tearing(water: &mut WaterWorld, ground: &impl WaterGround) -> Vec<SedimentChange> {
    water.set_erosion(ErosionConfig { speed: SPEED });
    water.wet_grass(
        ground,
        WaterCell::new(MIDDLE.0, -5, MIDDLE.1),
        FLOOR,
        0.0,
        0.0,
        0.0,
    );
    water.grass_stripped(MIDDLE);
    middle(&run(water, ground, 45))
}

#[test]
fn dead_ground_regrows_only_where_the_world_grows_grass() {
    let mut water = WaterWorld::new();
    let turned = after_tearing(&mut water, &meadow(0.0));
    assert_eq!(turned.len(), 1, "grass grew back once: {turned:?}");
    assert_eq!(turned[0].material, TerrainMaterial::SurfaceCover);
    assert!(water.grass_health(MIDDLE) >= 1.0, "fresh grass");
    // Where the seed laid soil, nothing grows back.
    let mut water = WaterWorld::new();
    let turned = after_tearing(&mut water, &meadow(0.0).ground);
    assert!(turned.is_empty(), "nothing grows here: {turned:?}");
    assert!(water.grass_health(MIDDLE) >= 1.0, "nothing tracked");
}

#[test]
fn wilted_grass_recovers_once_the_water_leaves() {
    let ground = meadow(0.0);
    let cell = WaterCell::new(MIDDLE.0, -5, MIDDLE.1);
    let mut water = WaterWorld::new();
    water.set_erosion(ErosionConfig { speed: SPEED });
    water.wet_grass(&ground, cell, FLOOR, 0.1, 0.0, 7.0 * 86_400.0 / SPEED);
    let wilted = water.grass_health(MIDDLE);
    assert!(wilted < 0.6, "a week under water left {wilted}");
    // Its ground dry, two weeks bring it back, in about 14 s.
    let turned = run(&mut water, &ground, 20);
    assert!(middle(&turned).is_empty(), "living grass is not turned");
    assert!(water.grass_health(MIDDLE) >= 1.0);
}

#[test]
fn grass_state_survives_a_save_and_old_saves_load_with_healthy_grass() {
    let ground = meadow(0.0);
    let cell = WaterCell::new(MIDDLE.0, -5, MIDDLE.1);
    let mut water = WaterWorld::new();
    water.wet_grass(&ground, cell, FLOOR, 0.1, 0.0, 7.0 * 86_400.0);
    let health = water.grass_health(MIDDLE);
    let loaded = WaterWorld::from_doc(&ground, &water.to_doc());
    assert!((loaded.grass_health(MIDDLE) - health).abs() < 1.0e-12);
    // A save from before grass was kept.
    let old: StoredWaterDoc = ron::from_str("(sheets: [], soil: [])").expect("an old save");
    assert!(old.grass.is_empty());
    let loaded = WaterWorld::from_doc(&ground, &old);
    assert!(loaded.grass_health(MIDDLE) >= 1.0);
}

#[test]
fn the_seed_grows_grass_on_meadows_over_its_loam_and_none_in_a_deep_cut() {
    let field = crate::TerrainField::new(crate::WorldSeed(7));
    let terrain = crate::TerrainOctree::default();
    let ground = crate::TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let spawn = field.safe_spawn().0;
    let meadow = (1..400)
        .flat_map(|ring| {
            let radius = f64::from(ring) * 0.5;
            (0..16).map(move |step| {
                let angle = f64::from(step) / 16.0 * std::f64::consts::TAU;
                (
                    spawn.x + radius * angle.cos(),
                    spawn.z + radius * angle.sin(),
                )
            })
        })
        .find_map(|(x, z)| {
            let top = field.topmost_surface(x, z)?;
            (ground.material(DVec3::new(x, top - 0.02, z)) == Some(TerrainMaterial::SurfaceCover))
                .then_some((x, z, top))
        })
        .expect("a meadow near spawn");
    let (x, z, top) = meadow;
    let native = ground.native_grass(x, z, top);
    let grass = native.grass.expect("grass grows on a meadow");
    assert_eq!(
        field.palette().look(grass).material,
        TerrainMaterial::SurfaceCover
    );
    assert_eq!(native.under.0, TerrainMaterial::Soil, "loam under it");
    assert!(ground.native_grass(x, z, top - 0.5).grass.is_none());
}
