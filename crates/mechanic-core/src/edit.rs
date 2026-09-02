//! Incremental construction edit descriptions shared by simulation and rendering.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ConstructionGraph, ConstructionMaterial, FaceOwner, PartId, PartSpec, RegionId};

/// Maximum part ownership per asynchronously replaceable geometry page.
pub const CONSTRUCTION_PAGE_MAX_PARTS: usize = 256;
/// Maximum vertex ownership per asynchronously replaceable geometry page.
pub const CONSTRUCTION_PAGE_MAX_VERTICES: usize = 65_000;

/// Stable owner of one construction geometry page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstructionGeometryOwner {
    /// Ordinary construction material geometry.
    Material(ConstructionMaterial),
    /// Authored machine appearance, using its stable application index.
    Authored(u8),
    /// Bearing and joint-ring geometry.
    Bearings,
    /// Shape-region-owned surface geometry.
    Region(RegionId),
}

/// Stable identity of one paged construction mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstructionPageKey {
    /// Geometry family owning the page.
    pub owner: ConstructionGeometryOwner,
    /// Zero-based page within that family.
    pub page: u32,
}

/// One complete replacement page ready for an atomic render cutover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstructionRenderPage {
    /// Stable page identity.
    pub key: ConstructionPageKey,
    /// Parts whose geometry is contained in the page.
    pub parts: Vec<PartId>,
    /// Built vertex count, bounded by [`CONSTRUCTION_PAGE_MAX_VERTICES`].
    pub vertex_count: usize,
}

/// Generation-exact construction render publication.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstructionRenderDelta {
    /// Exact committed graph revision represented by every page.
    pub generation: u64,
    /// Complete page replacements; old pages remain visible until all are ready.
    pub replacements: Vec<ConstructionRenderPage>,
    /// Pages no longer owned by the replacement revision.
    pub removals: Vec<ConstructionPageKey>,
}

impl ConstructionRenderDelta {
    /// Validates the fixed page contract before a render owner accepts work.
    pub fn validate(&self) -> bool {
        self.replacements.iter().all(|page| {
            page.parts.len() <= CONSTRUCTION_PAGE_MAX_PARTS
                && page.vertex_count <= CONSTRUCTION_PAGE_MAX_VERTICES
        })
    }
}

/// Exact graph change owned by one committed construction revision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstructionEditDelta {
    /// Newly allocated parts.
    pub added: BTreeSet<PartId>,
    /// Parts absent from the replacement graph.
    pub removed: BTreeSet<PartId>,
    /// Stable part identities whose authored geometry changed.
    pub modified: BTreeSet<PartId>,
    /// Parts whose visible geometry depends on changed topology, such as pipe ends.
    pub topology_dependent: BTreeSet<PartId>,
    /// Shape regions whose owned geometry changed or changed ownership.
    pub region_owned_geometry: BTreeSet<RegionId>,
}

impl ConstructionEditDelta {
    /// Computes the stable change between two immutable graph revisions.
    pub fn between(previous: &ConstructionGraph, current: &ConstructionGraph) -> Self {
        let previous_parts = previous.parts().map(|(id, spec)| (id, *spec)).collect();
        let current_parts = current.parts().map(|(id, spec)| (id, *spec)).collect();
        let mut delta = Self::between_parts(&previous_parts, &current_parts);

        let previous_features = previous
            .shape_features()
            .map(|(id, feature)| (id, feature.clone()))
            .collect::<Vec<_>>();
        let current_features = current
            .shape_features()
            .map(|(id, feature)| (id, feature.clone()))
            .collect::<Vec<_>>();
        if previous_features != current_features {
            for owner in previous_features
                .iter()
                .chain(&current_features)
                .flat_map(|(_, feature)| feature.targets.iter().map(|target| target.owner))
            {
                match owner {
                    crate::SolidOwner::Part(part) => {
                        delta.modified.insert(part);
                    }
                    crate::SolidOwner::Region(region) => {
                        delta.region_owned_geometry.insert(region);
                    }
                }
            }
        }

        let previous_regions = previous.regions().collect::<BTreeMap<_, _>>();
        let current_regions = current.regions().collect::<BTreeMap<_, _>>();
        delta.region_owned_geometry.extend(
            previous_regions
                .iter()
                .filter_map(|(&id, region)| {
                    (current_regions.get(&id).copied() != Some(*region)).then_some(id)
                })
                .chain(
                    current_regions
                        .keys()
                        .filter(|id| !previous_regions.contains_key(id))
                        .copied(),
                ),
        );

        let previous_welds = previous.welds().collect::<BTreeMap<_, _>>();
        let current_welds = current.welds().collect::<BTreeMap<_, _>>();
        for face in previous_welds
            .iter()
            .filter(|(id, weld)| current_welds.get(id).copied() != Some(*weld))
            .flat_map(|(_, weld)| [weld.first.owner, weld.second.owner])
            .chain(
                current_welds
                    .iter()
                    .filter(|(id, _)| !previous_welds.contains_key(id))
                    .flat_map(|(_, weld)| [weld.first.owner, weld.second.owner]),
            )
        {
            if let FaceOwner::Part(part) = face {
                delta.topology_dependent.insert(part);
            }
        }

        // A weld can change cap visibility and texture phase throughout one
        // connected pipe run. Rebuilding every pipe page remains bounded and
        // prevents a local delta from leaving distant dependent UVs stale.
        if delta.topology_dependent.iter().any(|&part| {
            previous_parts
                .get(&part)
                .or_else(|| current_parts.get(&part))
                .is_some_and(is_pipe)
        }) {
            delta.topology_dependent.extend(
                current_parts
                    .iter()
                    .filter_map(|(&part, spec)| is_pipe(spec).then_some(part)),
            );
        }
        delta
    }

    /// Computes the geometry portion from compact part snapshots.
    pub fn between_parts(
        previous: &BTreeMap<PartId, PartSpec>,
        current: &BTreeMap<PartId, PartSpec>,
    ) -> Self {
        Self {
            added: current
                .keys()
                .filter(|id| !previous.contains_key(id))
                .copied()
                .collect(),
            removed: previous
                .keys()
                .filter(|id| !current.contains_key(id))
                .copied()
                .collect(),
            modified: current
                .iter()
                .filter_map(|(&id, spec)| {
                    previous
                        .get(&id)
                        .is_some_and(|previous| previous != spec)
                        .then_some(id)
                })
                .collect(),
            ..Self::default()
        }
    }

    /// Every part whose foundation, geometry page, or dependent page may change.
    pub fn affected_parts(&self) -> BTreeSet<PartId> {
        self.added
            .iter()
            .chain(&self.removed)
            .chain(&self.modified)
            .chain(&self.topology_dependent)
            .copied()
            .collect()
    }

    /// True when no observable construction ownership changed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.modified.is_empty()
            && self.topology_dependent.is_empty()
            && self.region_owned_geometry.is_empty()
    }
}

const fn is_pipe(spec: &PartSpec) -> bool {
    matches!(spec, PartSpec::Cylinder(_) | PartSpec::PipeBend(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildCommand, BuildPose, CuboidSpec};

    #[test]
    fn part_delta_distinguishes_added_removed_and_modified() {
        let mut previous = ConstructionGraph::default();
        let crate::BuildOutcome::Spawned(first) = previous
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
            ))
            .expect("part is added")
        else {
            panic!("spawn reports its part");
        };
        let mut current = previous.clone();
        current
            .apply(BuildCommand::Remove(first))
            .expect("part is removed");
        current
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        bevy_math::IVec3::X,
                        crate::GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .expect("replacement is added");

        let delta = ConstructionEditDelta::between(&previous, &current);
        assert_eq!(delta.removed, BTreeSet::from([first]));
        assert_eq!(delta.added.len(), 1);
        assert!(delta.modified.is_empty());
    }
}
