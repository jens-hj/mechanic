//! Input-keyed geometry reuse across immutable graph revisions.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use crate::{
    ConstructionFrame, EvaluatedSolid, GraphError, PartSpec, ShapeFeature, ShapeFeatureId,
    ShapeRegion, SolidOwner,
};

#[derive(Clone, Copy)]
pub(super) enum SolidBase<'a> {
    Part(&'a PartSpec),
    Region(&'a ShapeRegion),
}

#[derive(Debug)]
enum StoredBase {
    Part(PartSpec),
    Region(ShapeRegion),
}

impl SolidBase<'_> {
    fn matches(&self, stored: &StoredBase) -> bool {
        match (self, stored) {
            (Self::Part(part), StoredBase::Part(old)) => *part == old,
            (Self::Region(region), StoredBase::Region(old)) => *region == old,
            _ => false,
        }
    }

    fn to_owned(self) -> StoredBase {
        match self {
            Self::Part(part) => StoredBase::Part(*part),
            Self::Region(region) => StoredBase::Region(region.clone()),
        }
    }
}

#[derive(Debug)]
struct Entry {
    base: StoredBase,
    features: Vec<(ShapeFeatureId, ShapeFeature)>,
    frame: ConstructionFrame,
    solid: Result<Arc<EvaluatedSolid>, GraphError>,
}

/// Keeps only the most recent input for each owner slot. Using the arena index
/// bounds storage even when deletion and placement repeatedly reuse a slot.
/// Cloned graphs share this memo, but compare all geometry inputs before reuse;
/// edits, rejected previews, undo and concurrent compilation cannot stale it.
#[derive(Clone, Debug, Default)]
pub(super) struct SolidCache(Arc<Mutex<Entries>>);

type Entries = BTreeMap<(bool, u32), Arc<Entry>>;

impl SolidCache {
    pub(super) fn evaluate(
        &self,
        owner: SolidOwner,
        base: SolidBase<'_>,
        features: &[(ShapeFeatureId, ShapeFeature)],
        frame: ConstructionFrame,
    ) -> Result<Arc<EvaluatedSolid>, GraphError> {
        let key = match owner {
            SolidOwner::Part(part) => (false, part.index()),
            SolidOwner::Region(region) => (true, region.index()),
        };
        let cached = self.0.lock().expect("solid cache lock").get(&key).cloned();
        if let Some(entry) = cached
            && base.matches(&entry.base)
            && features == entry.features
            && frame == entry.frame
        {
            return entry.solid.clone();
        }
        // Geometry evaluation never holds the lock: rendering must not wait for
        // an asynchronous compiler evaluating another owner or revision.
        let solid = match base {
            SolidBase::Part(spec) => crate::evaluate_part_solid(*spec, features.iter().cloned()),
            SolidBase::Region(region) => {
                crate::evaluate_region_solid(region, features.iter().cloned())
            }
        }
        .map(|mut solid| {
            frame.transform_solid(&mut solid);
            Arc::new(solid)
        })
        .map_err(GraphError::from);
        let entry = Arc::new(Entry {
            base: base.to_owned(),
            features: features.to_vec(),
            frame,
            solid: solid.clone(),
        });
        let retired = self.0.lock().expect("solid cache lock").insert(key, entry);
        drop(retired);
        solid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, EdgeChainRef,
        EdgeTreatment,
    };
    use bevy_math::{Quat, Vec3};

    fn block(graph: &mut ConstructionGraph, size: u8) -> crate::PartId {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([size; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        part
    }

    #[test]
    fn unrelated_placement_reuses_geometry_and_reframing_updates_it() {
        let mut graph = ConstructionGraph::new();
        let part = block(&mut graph, 2);
        let owner = SolidOwner::Part(part);
        let original = graph.evaluated_solid_shared(owner).unwrap();
        block(&mut graph, 1);
        assert!(Arc::ptr_eq(
            &original,
            &graph.evaluated_solid_shared(owner).unwrap()
        ));
        let snapshot = graph.clone();
        let frame =
            ConstructionFrame::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.4)).unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let mut expected = (*original).clone();
        frame.transform_solid(&mut expected);
        assert_eq!(*graph.evaluated_solid_shared(owner).unwrap(), expected);
        assert_eq!(*snapshot.evaluated_solid_shared(owner).unwrap(), *original);
    }

    #[test]
    fn feature_amount_prefix_and_rejected_edits_preserve_each_revision() {
        let mut graph = ConstructionGraph::new();
        let part = block(&mut graph, 2);
        let owner = SolidOwner::Part(part);
        let base = graph.evaluated_solid(owner).unwrap();
        let target = EdgeChainRef {
            owner,
            edge: base.logical_edges[0].key,
        };
        let BuildOutcome::ShapeFeatureAdded(feature) = graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [target],
                EdgeTreatment::Fillet,
                10,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let first = graph.clone();
        let first_solid = first.evaluated_solid(owner).unwrap();
        graph
            .apply(BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: 20,
            })
            .unwrap();
        let second_solid = graph.evaluated_solid(owner).unwrap();
        assert!(second_solid.volume() < first_solid.volume());
        assert_eq!(graph.evaluated_solid_before(owner, feature).unwrap(), base);
        assert_eq!(graph.evaluated_solid(owner).unwrap(), second_solid);
        assert!(
            graph
                .apply(BuildCommand::SetShapeFeatureAmount {
                    feature,
                    amount_ticks: u32::MAX
                })
                .is_err()
        );
        assert_eq!(graph.evaluated_solid(owner).unwrap(), second_solid);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..20 {
                    assert_eq!(first.evaluated_solid(owner).unwrap(), first_solid);
                }
            });
            for _ in 0..20 {
                assert_eq!(graph.evaluated_solid(owner).unwrap(), second_solid);
            }
        });
    }

    #[test]
    fn changing_base_geometry_in_a_reused_slot_updates_the_boundary() {
        let mut graph = ConstructionGraph::new();
        let part = block(&mut graph, 1);
        let owner = SolidOwner::Part(part);
        let original = graph.evaluated_solid(owner).unwrap();
        graph.apply(BuildCommand::Remove(part)).unwrap();
        let replacement = CuboidSpec::new([2; 3], BuildPose::default()).unwrap();
        let replacement_part = block(&mut graph, 2);
        assert_eq!(part.index(), replacement_part.index());
        assert!(graph.evaluated_solid(owner).is_err());
        let expected = crate::evaluate_part_solid(PartSpec::Cuboid(replacement), []).unwrap();
        assert_eq!(
            graph
                .evaluated_solid(SolidOwner::Part(replacement_part))
                .unwrap(),
            expected
        );
        assert!(expected.volume() > original.volume());
    }

    #[test]
    fn region_cage_changes_update_cached_geometry() {
        let mut graph = ConstructionGraph::new();
        let region = graph.regions.insert(
            ShapeRegion::new(
                bevy_math::IVec3::ZERO,
                bevy_math::IVec3::ONE,
                crate::ConstructionMaterial::default(),
            )
            .unwrap(),
        );
        let owner = SolidOwner::Region(region);
        let original = graph.evaluated_solid(owner).unwrap();
        graph
            .regions
            .get_mut(region)
            .unwrap()
            .set_offset([0; 3], [10; 3])
            .unwrap();
        let expected = crate::evaluate_region_solid(graph.region(region).unwrap(), []).unwrap();
        assert_eq!(graph.evaluated_solid(owner).unwrap(), expected);
        assert!(expected.volume() < original.volume());
    }
}
