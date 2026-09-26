//! Water in flight: streams pouring over lips.
//!
//! Water leaving a pool or a sheet over a drop is launched with its speed
//! over the lip, in one parcel a step, and each parcel falls along its arc
//! until it meets a pool, seed-derived water, running water or the ground.
//! The water in the air is counted, so a tall fall holds water and a stream
//! that stops keeps falling until its last parcel lands.

use std::collections::{BTreeSet, VecDeque};

use bevy_math::DVec3;
use serde::{Deserialize, Serialize};

use super::{End, GRAVITY, WaterCell, WaterFall, WaterGround, WaterWorld, floor_of};

/// Longest stretch of a parcel's path taken in one test, in metres.
const SWEEP_METRES: f64 = 0.1;

/// Longest a parcel falls before it is set down where it is, in seconds.
const FLIGHT_SECONDS: f64 = 30.0;

/// Water on its way down.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parcel {
    /// Where it is.
    pub position: DVec3,
    /// How fast it moves, in m/s.
    pub velocity: DVec3,
    /// Water it carries, in m³.
    pub volume_m3: f64,
    /// Seconds it has flown.
    pub age: f64,
}

/// One stream pouring over a lip.
#[derive(Clone, Debug, Default)]
pub(super) struct Jet {
    /// Where it leaves the lip.
    launch: DVec3,
    /// Parcels in flight, the oldest first.
    parcels: VecDeque<Parcel>,
    /// Water launched this step, in m³.
    poured: f64,
}

/// One stream in a saved world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JetDoc {
    /// The cell it pours over.
    pub lip: WaterCell,
    /// Where it leaves the lip.
    pub launch: DVec3,
    /// Parcels in flight.
    pub parcels: Vec<Parcel>,
}

/// Water launched over a lip.
#[derive(Clone, Copy, Debug)]
pub(super) struct Launch {
    /// The cell it pours over.
    pub(super) lip: WaterCell,
    /// Where it leaves the lip.
    pub(super) from: DVec3,
    /// Its speed as it leaves, in m/s.
    pub(super) velocity: DVec3,
}

impl WaterWorld {
    /// Water in flight, in m³.
    pub fn falling_m3(&self) -> f64 {
        self.jets
            .values()
            .flat_map(|jet| &jet.parcels)
            .map(|parcel| parcel.volume_m3)
            .sum()
    }

    pub(super) fn jet_docs(&self) -> Vec<JetDoc> {
        self.jets
            .iter()
            .map(|(&lip, jet)| JetDoc {
                lip,
                launch: jet.launch,
                parcels: jet.parcels.iter().copied().collect(),
            })
            .collect()
    }

    pub(super) fn load_jets(&mut self, docs: &[JetDoc]) {
        for doc in docs {
            self.jets.insert(
                doc.lip,
                Jet {
                    launch: doc.launch,
                    parcels: doc.parcels.iter().copied().collect(),
                    poured: 0.0,
                },
            );
        }
    }

    /// Launches water over a lip.
    pub(super) fn pour(&mut self, launch: Launch, volume: f64) {
        if volume <= 0.0 {
            return;
        }
        let jet = self.jets.entry(launch.lip).or_default();
        jet.launch = launch.from;
        jet.poured += volume;
        jet.parcels.push_back(Parcel {
            position: launch.from,
            velocity: launch.velocity,
            volume_m3: volume,
            age: 0.0,
        });
    }

    /// Flies every parcel for `dt` seconds and sets down those that land.
    /// Returns the water set down and the pools it fed.
    pub(super) fn step_jets(&mut self, ground: &impl WaterGround, dt: f64) -> (f64, BTreeSet<u32>) {
        let mut landed = 0.0;
        let mut fed = BTreeSet::new();
        let lips = self.jets.keys().copied().collect::<Vec<_>>();
        for lip in lips {
            let Some(mut jet) = self.jets.remove(&lip) else {
                continue;
            };
            let mut flying = VecDeque::with_capacity(jet.parcels.len());
            for mut parcel in jet.parcels.drain(..) {
                let start = parcel.position;
                parcel.velocity.y -= GRAVITY * dt;
                parcel.age += dt;
                let end = start + parcel.velocity * dt;
                match self.sweep(ground, start, end) {
                    Some(to) => {
                        if let End::Pool(id) = to {
                            fed.insert(id);
                        }
                        self.deposit_end(ground, to, parcel.volume_m3);
                        landed += parcel.volume_m3;
                    }
                    _ if parcel.age > FLIGHT_SECONDS => {
                        let to = self.landing(ground, WaterCell::containing(end));
                        self.deposit_end(ground, to, parcel.volume_m3);
                        landed += parcel.volume_m3;
                    }
                    _ => {
                        parcel.position = end;
                        flying.push_back(parcel);
                    }
                }
            }
            jet.parcels = flying;
            if !jet.parcels.is_empty() {
                self.jets.insert(lip, jet);
            }
        }
        (landed, fed)
    }

    /// Where water moving from `start` to `end` first meets something: a
    /// pool, seed-derived water, running water, or the ground, where it
    /// lands as water resting in the open cell before it.
    fn sweep(&mut self, ground: &impl WaterGround, start: DVec3, end: DVec3) -> Option<End> {
        let length = start.distance(end);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a step's flight is a few metres"
        )]
        let samples = ((length / SWEEP_METRES).ceil() as usize).max(1);
        let mut open = WaterCell::containing(start);
        for sample in 1..=samples {
            #[expect(clippy::cast_precision_loss, reason = "a few dozen samples")]
            let point = start.lerp(end, sample as f64 / samples as f64);
            let cell = WaterCell::containing(point);
            if let Some(&id) = self.owner.get(&cell)
                && self.pools.get(&id).is_some_and(|pool| pool.level > point.y)
            {
                return Some(End::Pool(id));
            }
            if let Some(surface) = self.implicit(ground, cell)
                && surface.level > point.y
            {
                return Some(End::Body(surface.body));
            }
            if let Some(surface) = self.running(cell)
                && surface.level > point.y
            {
                return Some(End::Sheet(cell));
            }
            let floor = floor_of(cell, self.openings(ground, cell));
            if floor.is_none_or(|floor| point.y < floor) {
                // Ground: the water comes to rest in the open cell it left,
                // or below it when it struck the ground beneath that cell.
                return Some(self.landing(ground, if floor.is_some() { cell } else { open }));
            }
            open = cell;
        }
        None
    }

    /// Every stream in flight, to draw: its launch point and parcels, and
    /// the water poured into it this step. Clears what was poured.
    pub(super) fn take_falls(&mut self, dt: f64) -> Vec<WaterFall> {
        let mut falls = Vec::with_capacity(self.jets.len());
        for jet in self.jets.values_mut() {
            let mut points = Vec::with_capacity(jet.parcels.len() + 1);
            if jet.poured > 0.0 {
                points.push(jet.launch);
            }
            points.extend(jet.parcels.iter().rev().map(|parcel| parcel.position));
            falls.push(WaterFall {
                points,
                rate_m3_s: jet.poured / dt,
            });
            jet.poured = 0.0;
        }
        falls
    }
}
