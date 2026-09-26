//! One body of still water: its cells, volume and level, and the border it
//! floods into.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};

use super::ground::Openings;
use super::{FINE_LAYER_METRES, FINE_VOLUME_M3, WaterCell};

/// Horizontal area of one terrain cell, in square metres.
const FINE_AREA_M2: f64 = FINE_LAYER_METRES * FINE_LAYER_METRES;

/// A border cell's floor, ordered for the priority flood.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Floor(pub(super) f64);

impl Eq for Floor {}

impl PartialOrd for Floor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Floor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// One body of still water.
#[derive(Clone, Debug)]
pub(super) struct Pool {
    /// The cell the pool began in; it rebuilds from here.
    pub(super) seed: WaterCell,
    pub(super) volume: f64,
    pub(super) level: f64,
    /// Member cells and their openings.
    pub(super) members: HashMap<WaterCell, Openings>,
    /// Open terrain cells of all members, by global terrain-cell layer.
    layers: BTreeMap<i32, u32>,
    /// Neighbours not yet taken in, lowest floor first.
    pub(super) border: BinaryHeap<Reverse<(Floor, WaterCell)>>,
    pub(super) queued: HashSet<WaterCell>,
    /// Neighbours it exchanges water with rather than takes in: other pools'
    /// cells, seed-derived water, and drops it spills over.
    pub(super) contacts: BTreeSet<WaterCell>,
}

impl Pool {
    pub(super) fn new(seed: WaterCell, volume: f64) -> Self {
        Self {
            seed,
            volume,
            level: f64::NEG_INFINITY,
            members: HashMap::new(),
            layers: BTreeMap::new(),
            border: BinaryHeap::new(),
            queued: HashSet::new(),
            contacts: BTreeSet::new(),
        }
    }

    pub(super) fn add_member(&mut self, cell: WaterCell, openings: Openings) {
        if let Some(previous) = self.members.insert(cell, openings) {
            self.remove_layers(cell, previous);
        }
        for (layer, &open) in openings.iter().enumerate() {
            if open > 0 {
                *self.layers.entry(cell.fine_layer(layer)).or_default() += u32::from(open);
            }
        }
        self.queued.remove(&cell);
        self.contacts.remove(&cell);
    }

    pub(super) fn remove_member(&mut self, cell: WaterCell) {
        if let Some(openings) = self.members.remove(&cell) {
            self.remove_layers(cell, openings);
        }
    }

    fn remove_layers(&mut self, cell: WaterCell, openings: Openings) {
        for (layer, &open) in openings.iter().enumerate() {
            let key = cell.fine_layer(layer);
            if let Some(count) = self.layers.get_mut(&key) {
                *count -= u32::from(open).min(*count);
                if *count == 0 {
                    self.layers.remove(&key);
                }
            }
        }
    }

    /// Queues a neighbour for the flood.
    pub(super) fn queue(&mut self, cell: WaterCell, floor: f64) {
        if !self.members.contains_key(&cell) && self.queued.insert(cell) {
            self.border.push(Reverse((Floor(floor), cell)));
        }
    }

    /// The lowest queued neighbour, if its floor lies below `height`.
    pub(super) fn next_below(&mut self, height: f64) -> Option<WaterCell> {
        let Reverse((Floor(floor), cell)) = *self.border.peek()?;
        if floor >= height {
            return None;
        }
        self.border.pop();
        self.queued.remove(&cell);
        Some(cell)
    }

    /// Sets the level the pool's volume reaches over its members. Past the
    /// top of its members it rises as through the top layer.
    pub(super) fn settle(&mut self) {
        let mut remaining = self.volume;
        let mut last = None;
        for (&layer, &count) in &self.layers {
            let bottom = f64::from(layer) * FINE_LAYER_METRES;
            let capacity = f64::from(count) * FINE_VOLUME_M3;
            if remaining <= capacity {
                self.level = bottom + remaining / (f64::from(count) * FINE_AREA_M2);
                return;
            }
            remaining -= capacity;
            last = Some((bottom + FINE_LAYER_METRES, count));
        }
        self.level = last.map_or(f64::NEG_INFINITY, |(top, count)| {
            top + remaining / (f64::from(count) * FINE_AREA_M2)
        });
    }

    /// Water its members hold below a height, in m³.
    pub(super) fn held_below(&self, height: f64) -> f64 {
        self.layers
            .iter()
            .map(|(&layer, &count)| {
                let bottom = f64::from(layer) * FINE_LAYER_METRES;
                let fill = ((height - bottom) / FINE_LAYER_METRES).clamp(0.0, 1.0);
                f64::from(count) * FINE_VOLUME_M3 * fill
            })
            .sum()
    }

    /// Open area at the pool's surface, in square metres.
    pub(super) fn surface_area(&self) -> f64 {
        let layer = (self.level / FINE_LAYER_METRES).floor();
        #[expect(clippy::cast_possible_truncation, reason = "layer index of a height")]
        let layer = layer as i32;
        let count = self
            .layers
            .range(..=layer)
            .next_back()
            .map_or(0, |(_, &count)| count);
        f64::from(count) * FINE_AREA_M2
    }

    /// Takes in another pool's cells and water.
    pub(super) fn absorb_pool(&mut self, other: Self) {
        self.volume += other.volume;
        for (cell, openings) in other.members {
            self.add_member(cell, openings);
        }
        for Reverse((floor, cell)) in other.border {
            if !self.members.contains_key(&cell) {
                self.queue(cell, floor.0);
            }
        }
        for cell in other.contacts {
            if !self.members.contains_key(&cell) {
                self.contacts.insert(cell);
            }
        }
    }
}
