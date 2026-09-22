//! Transient controller numbers shared by feedback, sequencing and both solvers.

use mechanic_core::{ConstructionGraph, GraphError, NumericParameter, PartId};

use super::AppSimulation;

impl AppSimulation {
    pub(crate) fn effective_graph(&self) -> &ConstructionGraph {
        self.controller_values
            .as_ref()
            .unwrap_or(&self.published_graph)
    }

    pub(crate) fn operate_dial(&mut self, dial: PartId, position: f32) -> Result<bool, GraphError> {
        let mut candidate = self.effective_graph().clone();
        let constrained = candidate.operate_dial(dial, position)?;
        self.controller_overrides = self
            .controller_overrides
            .keys()
            .copied()
            .chain(
                candidate
                    .input_configuration(dial)
                    .into_iter()
                    .flat_map(|configuration| {
                        configuration.analog.iter().map(|mapping| mapping.target)
                    }),
            )
            .filter_map(|target| candidate.numeric_value(target).map(|value| (target, value)))
            .collect();
        self.controller_values = Some(candidate);
        Ok(constrained)
    }

    /// Advances only the authoring revision when controller edits leave topology intact.
    pub(crate) fn accept_controller_edit(
        &mut self,
        previous_revision: u64,
        revision: u64,
        authored: &ConstructionGraph,
    ) {
        if let Some((published, foundation)) = self.world_revision
            && published == previous_revision
        {
            self.published_graph = authored.clone();
            self.world_revision = Some((revision, foundation));
        }
    }

    /// Reconciles configuration changes without declaring a numeric field edit.
    pub(crate) fn reconcile_controller_values(
        &mut self,
        before: &ConstructionGraph,
        after: &ConstructionGraph,
    ) {
        if self.reconcile_controller_edit(before, after, &[]).is_err() {
            // Configuration changes can invalidate the old runtime envelope.
            // Reset the entire layer; never preserve only part of a coupled set.
            self.controller_overrides.clear();
            self.controller_values = Some(after.clone());
        }
    }

    /// Explicit edits replace their targets even when the authored value is unchanged.
    /// Other runtime targets survive configuration changes, including unlinking.
    pub(crate) fn reconcile_controller_edit(
        &mut self,
        before: &ConstructionGraph,
        after: &ConstructionGraph,
        edited: &[NumericParameter],
    ) -> Result<(), GraphError> {
        let retained = self
            .controller_overrides
            .iter()
            .filter_map(|(&target, &value)| {
                let authored = before.numeric_value(target)?;
                let next = after.numeric_value(target)?;
                (!edited.contains(&target)
                    && authored.to_bits() == next.to_bits()
                    && value.to_bits() != next.to_bits())
                .then_some((target, value))
            })
            .collect::<Vec<_>>();
        let mut candidate = after.clone();
        // Validate on a candidate before changing either runtime layer.
        candidate.apply_numeric_values(&retained)?;
        self.controller_overrides = retained
            .iter()
            .filter_map(|(target, _)| {
                candidate
                    .numeric_value(*target)
                    .map(|value| (*target, value))
            })
            .collect();
        self.controller_values = Some(candidate);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
