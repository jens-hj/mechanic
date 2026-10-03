//! Grass: turf drowns under standing water, wears under a current and is
//! smothered by silt, and grows back where the world grows grass once the
//! water leaves. It changes at the world's own pace, as erosion does, and
//! the erosion speed hurries both alike.
//!
//! Each column of grass water has harmed keeps how alive its grass is. Grass
//! on top with no life left dies: the water asks the ground to turn it, and
//! the roots under it, into the soil the seed laid under it. Grass buried
//! alive under silt grows up through it, and bare ground where the seed
//! grows grass grows it again, in time. Weakened turf holds the less against
//! a current, down to what soil holds, so turf a flood has weakened is cut
//! like soil: grass dies first and the gully follows.

use serde::{Deserialize, Serialize};

use super::cells::CellMap;
use super::ground::NativeGrass;
use super::{WATER_CELL_EDGE_CELLS, WATER_CELL_METRES, WaterCell, WaterGround, WaterWorld};
use crate::{MATERIAL_QUANTUM_M3, SedimentChange, TerrainMaterial};

/// Seconds in a day.
const DAY_SECONDS: f64 = 86_400.0;

/// Seconds grass lives under water as deep as it is tall: turf grasses
/// survive one to three weeks wholly under water.
const DROWN_SECONDS: f64 = 14.0 * DAY_SECONDS;

/// Depth of water that covers grass, in metres.
const GRASS_METRES: f64 = 0.05;

/// How fast grass dies under a film of water, or in waterlogged ground with
/// none standing on it, against grass covered: grasses stand waterlogging
/// for weeks to months.
const WATERLOGGED_SHARE: f64 = 0.25;

/// Drag at which a current wears turf through in [`WORN_SECONDS`], in Pa:
/// good grass holds 2 m/s for about 50 hours and 3 m/s for about 10
/// (Hewlett, Boorman and Bramley, CIRIA 116, 1987), so the time it holds
/// falls with the square of the drag.
const WORN_PA: f64 = 60.0;

/// Seconds a current dragging at [`WORN_PA`] takes to wear turf through.
const WORN_SECONDS: f64 = 50.0 * 3_600.0;

/// Drag under which a current does grass no harm, in Pa.
const HARMLESS_PA: f64 = 2.0;

/// Silt laid over grass that smothers it, in metres.
const SMOTHERING_METRES: f64 = 0.1;

/// Seconds harmed grass takes to recover wholly once the water leaves.
const RECOVER_SECONDS: f64 = 14.0 * DAY_SECONDS;

/// Seconds grass buried alive under silt takes to grow up through it.
const GROW_THROUGH_SECONDS: f64 = 7.0 * DAY_SECONDS;

/// Seconds bare ground where the seed grows grass takes to grow it again:
/// turf recovers over four to eight weeks.
const REGROW_SECONDS: f64 = 42.0 * DAY_SECONDS;

/// Ground more than this full of water is waterlogged.
const WATERLOGGED_FILL: f64 = 0.9;

/// Columns of grass grow in turn, one of this many sets each step: each
/// about once a second.
const GROW_SETS: u32 = 20;

/// Horizontal area of one water cell, in square metres.
const CELL_AREA_M2: f64 = WATER_CELL_METRES * WATER_CELL_METRES;

/// The grass of one column water has harmed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Grass {
    /// Height of the ground's top, in metres.
    top: f64,
    /// What the ground's top is made of, once measured.
    on_top: Option<TerrainMaterial>,
    /// How alive it is, from 0 to 1: on the column's top, or buried alive
    /// under silt.
    health: f64,
    /// How far grass has grown back over the top, from 0 to 1: up through
    /// silt over living grass, or over bare ground.
    regrowth: f64,
    /// Whether the column's top is bare: its grass died or was torn away.
    bare: bool,
    /// Whether water stood on or ran over it since it last grew.
    wet: bool,
    /// What grows on the column by nature, once found.
    native: Option<NativeGrass>,
    /// How the ground's top is to turn, once asked.
    turn: Option<Turn>,
    /// Whether the ground was asked to turn it and has not answered.
    asked: bool,
}

impl Grass {
    const fn alive(top: f64, health: f64) -> Self {
        Self {
            top,
            on_top: None,
            health,
            regrowth: 0.0,
            bare: false,
            wet: false,
            native: None,
            turn: None,
            asked: false,
        }
    }
}

/// Harmed grass by column.
pub(super) type GrassMap = CellMap<(i32, i32), Grass>;

/// How the ground's top turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Turn {
    /// Grass dies to the soil under it.
    Wither,
    /// Grass grows over it again.
    Regrow,
}

/// One column's harmed grass, in a saved world.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrassDoc {
    /// The column, in water cells along x and z.
    pub column: (i32, i32),
    /// Height of the ground's top, in metres.
    pub top: f64,
    /// How alive its grass is, from 0 to 1.
    pub health: f64,
    /// How far grass has grown back over its top, from 0 to 1.
    pub regrowth: f64,
    /// Whether its top is bare.
    pub bare: bool,
}

/// How fast grass under water `depth` metres deep, dragging at it with
/// `drag` Pa, dies, as a share of its life each second.
fn harm(depth: f64, drag: f64) -> f64 {
    let covered = (depth / GRASS_METRES).clamp(0.0, 1.0);
    let drowning = (WATERLOGGED_SHARE + (1.0 - WATERLOGGED_SHARE) * covered) / DROWN_SECONDS;
    let wearing = if drag > HARMLESS_PA {
        (drag / WORN_PA).powi(2) / WORN_SECONDS
    } else {
        0.0
    };
    drowning + wearing
}

impl WaterWorld {
    /// How alive the grass of a column is, from 0 to 1: grass water has
    /// not harmed is wholly alive.
    pub(super) fn grass_health(&self, column: (i32, i32)) -> f64 {
        self.grass
            .get(&column)
            .map_or(1.0, |grass| if grass.bare { 0.0 } else { grass.health })
    }

    /// Water `depth` metres deep, dragging at its bed with `drag` Pa, over
    /// `dt` seconds on the ground of `cell`, whose top is at `floor`: it
    /// harms grass there, and keeps it from growing.
    pub(super) fn wet_grass(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        floor: f64,
        depth: f64,
        drag: f64,
        dt: f64,
    ) {
        let column = (cell.x, cell.z);
        let speed = self.erosion_speed();
        if let Some(grass) = self.grass.get_mut(&column) {
            grass.wet = true;
            grass.top = floor;
            if !grass.bare {
                grass.health = (grass.health - speed * dt * harm(depth, drag)).max(0.0);
            }
            return;
        }
        let material = self.soil_material(ground, cell, floor);
        if material != TerrainMaterial::SurfaceCover {
            return;
        }
        let mut grass = Grass::alive(floor, (1.0_f64 - speed * dt * harm(depth, drag)).max(0.0));
        grass.wet = true;
        grass.on_top = Some(material);
        self.grass.insert(column, grass);
    }

    /// Stored water standing on grass, over `dt`: each pool's bed.
    fn wet_grass_under_pools(&mut self, ground: &impl WaterGround, dt: f64) {
        let beds = self
            .pools
            .values()
            .flat_map(|pool| {
                let level = pool.level;
                pool.beds()
                    .map(move |(cell, floor)| (cell, floor, level - floor))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (cell, floor, depth) in beds {
            self.wet_grass(ground, cell, floor, depth.max(0.0), 0.0, dt);
        }
    }

    /// Grass grows, recovers and dies over `dt`: one set of columns each
    /// step, over that set's turn, and still water's beds once a round.
    pub(super) fn grow_grass(&mut self, ground: &impl WaterGround, dt: f64) {
        let set = self.grass_set;
        self.grass_set = (set + 1) % GROW_SETS;
        let turn = dt * f64::from(GROW_SETS);
        if set == 0 {
            self.wet_grass_under_pools(ground, turn);
        }
        let speed = self.erosion_speed();
        let ours = |&(x, z): &(i32, i32)| {
            (i64::from(x) + 2 * i64::from(z)).rem_euclid(i64::from(GROW_SETS)) == i64::from(set)
        };
        let columns = self.grass.keys().copied().filter(ours).collect::<Vec<_>>();
        for column in columns {
            let waterlogged = self.soil_fill(column.0, column.1) > WATERLOGGED_FILL;
            let Some(grass) = self.grass.get_mut(&column) else {
                continue;
            };
            let centre = WaterCell::new(column.0, 0, column.1).centre();
            let top = grass.top;
            let on_top = *grass.on_top.get_or_insert_with(|| {
                ground
                    .material(bevy_math::DVec3::new(centre.x, top - 0.02, centre.z))
                    .unwrap_or(TerrainMaterial::Rock)
            });
            if grass.native.is_none() {
                grass.native = Some(ground.native_grass(centre.x, centre.z, top));
            }
            let wet = std::mem::take(&mut grass.wet);
            if grass.asked || grass.turn.is_some() {
                continue;
            }
            let grows = grass.native.and_then(|native| native.grass).is_some();
            if grass.bare {
                if !grows {
                    // Nothing grows here by nature: bare it stays.
                    self.grass.remove(&column);
                } else if !wet && !waterlogged {
                    grass.regrowth += speed * turn / REGROW_SECONDS;
                    if grass.regrowth >= 1.0 {
                        grass.turn = Some(Turn::Regrow);
                    }
                }
                continue;
            }
            if waterlogged && !wet {
                grass.health = (grass.health - speed * turn * harm(0.0, 0.0)).max(0.0);
            }
            if grass.health <= 0.0 {
                grass.turn = Some(Turn::Wither);
                continue;
            }
            if wet || waterlogged {
                continue;
            }
            grass.health = (grass.health + speed * turn / RECOVER_SECONDS).min(1.0);
            if on_top == TerrainMaterial::SurfaceCover {
                if grass.health >= 1.0 {
                    self.grass.remove(&column);
                }
            } else if grows {
                // Buried alive under silt: it grows up through it, or over
                // whatever else came to lie on it where grass grows.
                grass.regrowth += speed * turn / GROW_THROUGH_SECONDS;
                if grass.regrowth >= 1.0 {
                    grass.turn = Some(Turn::Regrow);
                }
            }
        }
    }

    /// What the ground's top is made of under a cell whose ground's top is
    /// at `floor`, measured the first time it is asked.
    pub(super) fn soil_material(
        &mut self,
        ground: &impl WaterGround,
        cell: WaterCell,
        floor: f64,
    ) -> TerrainMaterial {
        let soil = self.soil_at(ground, cell, floor);
        *soil.material.get_or_insert_with(|| {
            let centre = cell.centre();
            ground
                .material(bevy_math::DVec3::new(centre.x, soil.top - 0.02, centre.z))
                .unwrap_or(TerrainMaterial::Rock)
        })
    }

    /// The changes to the ground's top grass asks for, after `changes`, up
    /// to `most` in all: grass that died, and grass grown back.
    pub(super) fn grass_requests(
        &mut self,
        changes: &mut Vec<SedimentChange>,
        asked: &mut Vec<((i32, i32), Turn)>,
        most: usize,
    ) {
        let mut columns = self
            .grass
            .iter()
            .filter(|(_, grass)| grass.turn.is_some() && !grass.asked)
            .map(|(&column, _)| column)
            .collect::<Vec<_>>();
        columns.sort_unstable();
        let edge = WATER_CELL_EDGE_CELLS;
        for column in columns {
            if changes.len() >= most {
                break;
            }
            let Some(grass) = self.grass.get_mut(&column) else {
                continue;
            };
            let height = grass.top;
            let (Some(turn), Some(native)) = (grass.turn, grass.native) else {
                continue;
            };
            let (material, look) = match turn {
                Turn::Wither => native.under,
                Turn::Regrow => {
                    let Some(look) = native.grass else {
                        grass.turn = None;
                        continue;
                    };
                    (TerrainMaterial::SurfaceCover, look)
                }
            };
            changes.push(SedimentChange {
                x: column.0 * edge,
                z: column.1 * edge,
                edge,
                height,
                quanta: 0,
                material,
                look: Some(look),
            });
            grass.asked = true;
            asked.push((column, turn));
        }
    }

    /// What grass a take lays bare on a column turns to: what the seed laid
    /// under its grass, where known.
    pub(super) fn under_grass(&self, column: (i32, i32)) -> Option<NativeGrass> {
        self.grass.get(&column).and_then(|grass| grass.native)
    }

    /// The ground turned a column's top as grass asked, relabelling
    /// `relabelled` cells.
    pub(super) fn grass_turned(&mut self, column: (i32, i32), turn: Turn, relabelled: u64) {
        let Some(grass) = self.grass.get_mut(&column) else {
            return;
        };
        grass.asked = false;
        grass.turn = None;
        if relabelled == 0 {
            // The top was not what the water thought: measured again, the
            // grass is followed afresh.
            self.grass.remove(&column);
            self.top_changed(column, None);
            return;
        }
        let on_top = match turn {
            Turn::Wither => {
                grass.bare = true;
                grass.health = 0.0;
                grass.regrowth = 0.0;
                grass.native.map(|native| native.under.0)
            }
            Turn::Regrow => {
                // Fresh grass, and as alive as the grass it grew from.
                if grass.bare {
                    self.grass.remove(&column);
                } else {
                    grass.regrowth = 0.0;
                }
                Some(TerrainMaterial::SurfaceCover)
            }
        };
        self.top_changed(column, on_top);
    }

    /// The ground could not turn a column's top: it is asked again.
    pub(super) fn grass_refused(&mut self, column: (i32, i32)) {
        if let Some(grass) = self.grass.get_mut(&column) {
            grass.asked = false;
        }
    }

    /// A take tore a column's turf away: the ground under it is bare.
    pub(super) fn grass_stripped(&mut self, column: (i32, i32)) {
        let top = self.bed_top(column);
        let grass = self.grass.entry(column).or_insert(Grass::alive(top, 0.0));
        grass.bare = true;
        grass.health = 0.0;
        grass.regrowth = 0.0;
        self.top_changed(column, None);
    }

    /// Height of a column's ground as the water last knew it.
    fn bed_top(&self, column: (i32, i32)) -> f64 {
        self.grass
            .get(&column)
            .map(|grass| grass.top)
            .or_else(|| self.soil.get(&column).map(|soil| soil.top))
            .unwrap_or(0.0)
    }

    /// What a column's top is made of changed: `on_top` where known, or to
    /// be measured again.
    fn top_changed(&mut self, column: (i32, i32), on_top: Option<TerrainMaterial>) {
        if let Some(grass) = self.grass.get_mut(&column) {
            grass.on_top = on_top;
        }
        if let Some(soil) = self.soil.get_mut(&column) {
            soil.material = on_top;
        }
    }

    /// The ground's top over a column is at `top` now, and to be measured
    /// again.
    pub(super) fn grass_ground_changed(&mut self, column: (i32, i32), top: f64) {
        if let Some(grass) = self.grass.get_mut(&column) {
            grass.top = top;
            grass.on_top = None;
        }
    }

    /// Silt `laid` quanta deep over a column: grass under it is smothered
    /// the more the deeper it lies.
    pub(super) fn grass_buried(&mut self, column: (i32, i32), laid: f64) {
        let on_grass = self
            .soil
            .get(&column)
            .and_then(|soil| soil.material)
            .is_some_and(|material| material == TerrainMaterial::SurfaceCover);
        let thickness = laid * MATERIAL_QUANTUM_M3 / CELL_AREA_M2;
        if !on_grass && !self.grass.contains_key(&column) {
            return;
        }
        let top = self.bed_top(column);
        let grass = self.grass.entry(column).or_insert(Grass::alive(top, 1.0));
        if !grass.bare {
            grass.health = (grass.health - thickness / SMOTHERING_METRES).max(0.0);
            grass.regrowth = 0.0;
        }
        self.top_changed(column, None);
    }

    /// Wilting grass, by column, to draw: how far from green each is.
    pub(super) fn wilting(&self) -> impl Iterator<Item = ((i32, i32), f64, f64)> + '_ {
        self.grass
            .iter()
            .filter(|(_, grass)| !grass.bare && grass.health < 1.0)
            .map(|(&column, grass)| (column, grass.top, 1.0 - grass.health))
    }

    pub(super) fn grass_docs(&self) -> Vec<GrassDoc> {
        let mut docs = self
            .grass
            .iter()
            .map(|(&column, grass)| GrassDoc {
                column,
                top: grass.top,
                health: grass.health,
                regrowth: grass.regrowth,
                bare: grass.bare,
            })
            .collect::<Vec<_>>();
        docs.sort_by_key(|doc| doc.column);
        docs
    }

    pub(super) fn load_grass(&mut self, docs: &[GrassDoc]) {
        for doc in docs {
            let mut grass = Grass::alive(doc.top, doc.health.clamp(0.0, 1.0));
            grass.regrowth = doc.regrowth.clamp(0.0, 1.0);
            grass.bare = doc.bare;
            self.grass.insert(doc.column, grass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DAY_SECONDS, HARMLESS_PA, harm};

    #[test]
    fn grass_drowns_in_weeks_and_a_flood_wears_it_through_in_hours() {
        let days = |rate: f64| 1.0 / rate / DAY_SECONDS;
        let covered = days(harm(0.1, 0.0));
        assert!(
            (10.0..20.0).contains(&covered),
            "drowns in {covered:.1} days"
        );
        // A film, or waterlogged ground, takes several times as long.
        assert!(days(harm(0.002, 0.0)) > 3.0 * covered);
        // A gentle sheet over a meadow does it next to no harm beyond
        // drowning; a flood wears it through within a day.
        assert!(harm(0.05, HARMLESS_PA) <= harm(0.05, 0.0));
        assert!(days(harm(0.05, 6.0)) > 0.9 * covered);
        let flood = 1.0 / harm(0.5, 130.0) / 3_600.0;
        assert!(
            (5.0..15.0).contains(&flood),
            "a flood wears it through in {flood:.1} h"
        );
    }
}
