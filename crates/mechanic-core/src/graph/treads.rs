//! Treads cut into part surfaces, kept beside the parts they belong to.
//!
//! A tread is surface detail rather than geometry: it leaves the envelope,
//! mass and connections alone. Keeping it out of [`crate::PartSpec`] keeps
//! every part small; the graph drops a tread whenever its surface stops being
//! able to carry one.

use super::ConstructionGraph;
use super::error::GraphError;
use crate::{LayerFace, PartId, SurfaceTreads, TreadSpec};

impl ConstructionGraph {
    /// The treads cut into one part's surfaces; none for a missing part.
    pub fn part_treads(&self, part: PartId) -> SurfaceTreads {
        self.treads.get(&part).copied().unwrap_or_default()
    }

    /// Every part carrying at least one tread, in stable order.
    pub fn treaded_parts(&self) -> impl Iterator<Item = (PartId, SurfaceTreads)> + '_ {
        self.treads.iter().map(|(&part, &treads)| (part, treads))
    }

    pub(super) fn set_tread(
        &mut self,
        part: PartId,
        surface: LayerFace,
        tread: Option<TreadSpec>,
    ) -> Result<(), GraphError> {
        let spec = self
            .parts
            .get(part)
            .copied()
            .ok_or(GraphError::MissingPart(part))?;
        if tread.is_some() {
            if self.region_of(part).is_some() {
                return Err(GraphError::TreadInRegion(part));
            }
            spec.tread_fits(surface)?;
        }
        let treads = self.part_treads(part).with(surface, tread);
        if treads.is_empty() {
            self.treads.remove(&part);
        } else {
            self.treads.insert(part, treads);
        }
        Ok(())
    }

    /// Drops every tread whose part is gone or whose surface can no longer
    /// carry one: teeth or a spiral recut it, a layer filled its bore, or a
    /// shaped region took the part's surfaces over.
    pub(super) fn prune_treads(&mut self) {
        let mut treads = self.treads.clone();
        treads.retain(|&part, treads| {
            let Some(&spec) = self.parts.get(part) else {
                return false;
            };
            if self.region_of(part).is_some() {
                return false;
            }
            for (surface, _) in treads.iter() {
                if spec.tread_fits(surface).is_err() {
                    *treads = treads.with(surface, None);
                }
            }
            !treads.is_empty()
        });
        if treads != self.treads {
            self.treads = treads;
        }
    }
}
