//! Half-block bayonet couplers for joining moving creations.

use crate::{BuildPose, GRID_UNIT_METERS};
use bevy_math::Vec3;

/// One half of a rigid bayonet connection, facing local positive Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplerSpec {
    /// Envelope centre and cardinal orientation.
    pub pose: BuildPose,
}

impl CouplerSpec {
    /// Creates an unconnected coupler.
    pub const fn new(pose: BuildPose) -> Self {
        Self { pose }
    }

    /// Half-block envelope; two mounted face to face occupy one block.
    pub const fn size_meters(self) -> Vec3 {
        Vec3::new(GRID_UNIT_METERS, GRID_UNIT_METERS * 0.5, GRID_UNIT_METERS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuildCommand, BuildOutcome, ConstructionGraph, ControllerSpec, CreationDocument, DriveKey,
        InputConfiguration, PartSpec,
    };

    #[test]
    fn half_block_geometry_and_controller_binding_survive_save_load() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(coupler) = graph
            .apply(BuildCommand::SpawnCoupler(CouplerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("coupler")
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            panic!("controller")
        };
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input: coupler,
                configuration: InputConfiguration {
                    controller: Some(controller),
                    key: DriveKey::new('C'),
                    ..Default::default()
                },
            })
            .unwrap();
        let doc = CreationDocument::from_graph(&graph, "Coupler", &[]);
        let loaded = ron::from_str::<CreationDocument>(&ron::to_string(&doc).unwrap())
            .unwrap()
            .into_graph()
            .unwrap();
        assert_eq!(
            CreationDocument::from_graph(&loaded.graph, "Coupler", &[]),
            doc
        );
        let (id, spec) = loaded
            .graph
            .parts()
            .find(|(_, spec)| matches!(spec, PartSpec::Coupler(_)))
            .unwrap();
        assert_eq!(spec.size_meters(), Vec3::new(0.25, 0.125, 0.25));
        let controller = loaded
            .graph
            .input_configuration(id)
            .unwrap()
            .controller
            .unwrap();
        let mut graph = loaded.graph;
        graph.apply(BuildCommand::Remove(controller)).unwrap();
        assert!(graph.input_configuration(id).unwrap().controller.is_none());
        assert_eq!(graph.compile().unwrap().compounds.len(), 1);
    }
}
