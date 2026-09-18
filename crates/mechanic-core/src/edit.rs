//! Incremental construction edit descriptions shared by simulation and rendering.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ConstructionGraph, FaceOwner, PartId, PartSpec, RegionId};

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
        for (part, _) in current.parts() {
            if previous.part(part).is_some()
                && previous.part_frame(part) != current.part_frame(part)
            {
                delta.modified.insert(part);
                if let Some(region) = current.region_of(part) {
                    delta.region_owned_geometry.insert(region);
                }
            }
        }

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

        let has_pipes = previous_parts
            .values()
            .chain(current_parts.values())
            .any(is_pipe);
        if has_pipes {
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
    matches!(
        spec,
        PartSpec::Cylinder(_) | PartSpec::PipeBend(_) | PartSpec::PipeJunction(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildCommand, BuildPose, ConstructionMaterial, CuboidSpec};

    #[test]
    fn reframing_invalidates_only_changed_parts_and_their_owned_regions() {
        use bevy_math::{IVec3, Quat, Vec3};
        let mut previous = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in [0, 4, 8] {
            let crate::BuildOutcome::Spawned(part) = previous
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::from_half_grid(
                            IVec3::new(2 * x + 1, 1, 1),
                            crate::GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        let mut regions = Vec::new();
        for x in [0, 4] {
            let region = crate::ShapeRegion::new(
                IVec3::new(2 * x, 0, 0),
                IVec3::ONE,
                ConstructionMaterial::Steel,
            )
            .unwrap();
            let crate::BuildOutcome::RegionAdded(region) =
                previous.apply(BuildCommand::AddRegion(region)).unwrap()
            else {
                unreachable!()
            };
            regions.push(region);
        }
        // Both changed and unchanged geometry already use nondefault frames.
        for (part, angle) in [(parts[0], 0.3), (parts[1], -0.6), (parts[2], 0.8)] {
            previous
                .reframe_parts(
                    [part],
                    crate::ConstructionFrame::new(Vec3::ZERO, Quat::from_rotation_y(angle))
                        .unwrap(),
                )
                .unwrap();
        }
        let mut current = previous.clone();
        current
            .reframe_parts(
                [parts[0], parts[2]],
                crate::ConstructionFrame::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_x(0.7))
                    .unwrap(),
            )
            .unwrap();
        let delta = ConstructionEditDelta::between(&previous, &current);
        assert_eq!(delta.modified, BTreeSet::from([parts[0], parts[2]]));
        assert_eq!(delta.affected_parts(), BTreeSet::from([parts[0], parts[2]]));
        assert_eq!(delta.region_owned_geometry, BTreeSet::from([regions[0]]));
        assert!(delta.added.is_empty());
        assert!(delta.removed.is_empty());
        assert!(delta.topology_dependent.is_empty());
        assert_eq!(previous.part_frame(parts[1]), current.part_frame(parts[1]));
    }

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
