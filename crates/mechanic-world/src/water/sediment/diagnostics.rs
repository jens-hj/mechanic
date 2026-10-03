//! Optional session history of completed sediment transfers, never persisted.

use std::collections::BTreeMap;

use crate::{MATERIAL_QUANTUM_M3, SedimentApplied, WATER_CELL_METRES, WaterWorld};

/// Simulation seconds for recent sediment activity to lose half its strength.
const HALF_LIFE_SECONDS: f64 = 10.0;

/// Material transfers in one water-cell column, in material quanta.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SedimentDiagnosticColumn {
    /// Water-cell coordinates along x and z.
    pub column: (i32, i32),
    /// Bed height in world metres.
    pub height: f64,
    /// Completed removal and deposition since reset, independently.
    pub accumulated: [f64; 2],
    /// Completed removal and deposition with a ten-second simulation half-life.
    pub recent: [f64; 2],
    /// Settled sediment, including requests awaiting a terrain response.
    pub pending: f64,
}

/// A consistent, read-only snapshot of optional sediment diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SedimentDiagnostics {
    /// Water simulation seconds recorded since reset.
    pub seconds: f64,
    /// Columns with recorded transfers or pending sediment.
    pub columns: Vec<SedimentDiagnosticColumn>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Recorder {
    seconds: f64,
    columns: BTreeMap<(i32, i32), (SedimentDiagnosticColumn, f64)>,
}

impl WaterWorld {
    /// Enables optional session diagnostics. Disabling releases the history.
    pub fn set_sediment_diagnostics(&mut self, enabled: bool) {
        if enabled {
            self.sediment.diagnostics.get_or_insert_default();
        } else {
            self.sediment.diagnostics = None;
        }
    }

    /// Clears completed-transfer history without changing sediment or pending edits.
    pub fn reset_sediment_diagnostics(&mut self) {
        if let Some(recorder) = &mut self.sediment.diagnostics {
            *recorder = Recorder::default();
        }
    }

    /// Captures completed activity and current pending sediment, when enabled.
    pub fn sediment_diagnostics(&self) -> Option<SedimentDiagnostics> {
        let recorder = self.sediment.diagnostics.as_ref()?;
        let mut columns = recorder
            .columns
            .iter()
            .map(|(&key, &(mut column, updated))| {
                column.recent = column
                    .recent
                    .map(|value| value * decay(recorder.seconds - updated));
                (key, column)
            })
            .collect::<BTreeMap<_, _>>();
        for (&key, bed) in &self.sediment.beds {
            let pending = bed.settled.total() + bed.laying.total();
            if pending > 0.0 {
                let column = columns.entry(key).or_insert(SedimentDiagnosticColumn {
                    column: key,
                    ..Default::default()
                });
                column.height = bed.height;
                column.pending = pending;
            }
        }
        Some(SedimentDiagnostics {
            seconds: recorder.seconds,
            columns: columns.into_values().collect(),
        })
    }

    pub(in crate::water) fn advance_sediment_diagnostics(&mut self, dt: f64) {
        if let Some(recorder) = &mut self.sediment.diagnostics {
            recorder.seconds += dt.max(0.0);
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "terrain transfer quantities fit f64 precision"
    )]
    pub(super) fn record_sediment(&mut self, key: (i32, i32), applied: SedimentApplied) {
        let Some(recorder) = &mut self.sediment.diagnostics else {
            return;
        };
        let amounts = [applied.total_taken() as f64, applied.laid as f64];
        if applied.total_taken() == 0 && applied.laid == 0 {
            return;
        }
        let (column, updated) = recorder.columns.entry(key).or_default();
        column.column = key;
        if let Some(bed) = self.sediment.beds.get(&key) {
            column.height = bed.height;
        }
        column.height +=
            (amounts[1] - amounts[0]) * MATERIAL_QUANTUM_M3 / WATER_CELL_METRES.powi(2);
        for (index, amount) in amounts.into_iter().enumerate() {
            column.accumulated[index] += amount;
            column.recent[index] =
                column.recent[index] * decay(recorder.seconds - *updated) + amount;
        }
        *updated = recorder.seconds;
    }
}

fn decay(seconds: f64) -> f64 {
    2.0_f64.powf(-seconds / HALF_LIFE_SECONDS)
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "small integral quanta and exact binary half-life values"
)]
mod tests {
    use super::super::{Ask, Bed, SedimentLoad};
    use super::*;

    #[test]
    fn completed_transfers_decay_independently_and_pending_is_a_current_balance() {
        let mut water = WaterWorld::default();
        water.set_sediment_diagnostics(true);
        water.sediment.beds.insert(
            (2, -3),
            Bed {
                height: 4.0,
                settled: SedimentLoad {
                    sand: 40.0,
                    fines: 0.0,
                },
                laying: SedimentLoad {
                    sand: 60.0,
                    fines: 0.0,
                },
                ..Default::default()
            },
        );
        water
            .sediment
            .asked
            .push(((2, -3), Ask::Lay { sand: true }));
        assert_eq!(
            water.sediment_diagnostics().unwrap().columns[0].pending,
            100.0
        );
        water.sediment_applied(&[SedimentApplied {
            laid: 50,
            ..Default::default()
        }]);
        water.advance_sediment_diagnostics(10.0);
        let snapshot = water.sediment_diagnostics().unwrap();
        assert_eq!(snapshot.columns[0].accumulated, [0.0, 50.0]);
        assert_eq!(snapshot.columns[0].recent, [0.0, 25.0]);
        assert_eq!(snapshot.columns[0].pending, 50.0);
        water.reset_sediment_diagnostics();
        let reset = water.sediment_diagnostics().unwrap();
        assert_eq!(reset.seconds, 0.0);
        assert_eq!(reset.columns[0].accumulated, [0.0; 2]);
        assert_eq!(reset.columns[0].pending, 50.0);
        water.set_sediment_diagnostics(false);
        assert!(water.sediment_diagnostics().is_none());
    }

    #[test]
    fn refused_edits_do_not_record_completed_activity_or_lose_pending_sediment() {
        let mut water = WaterWorld::default();
        water.set_sediment_diagnostics(true);
        water.sediment.beds.insert(
            (0, 0),
            Bed {
                laying: SedimentLoad {
                    sand: 30.0,
                    fines: 0.0,
                },
                ..Default::default()
            },
        );
        water.sediment.asked.push(((0, 0), Ask::Lay { sand: true }));
        water.sediment_refused();
        let column = water.sediment_diagnostics().unwrap().columns[0];
        assert_eq!(column.accumulated, [0.0; 2]);
        assert_eq!(column.pending, 30.0);
    }
}
