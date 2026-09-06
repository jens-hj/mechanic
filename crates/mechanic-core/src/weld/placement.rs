//! Construction staging for source-default placement.

use crate::{
    BuildCommand, CompiledCreation, ConstructionFrame, ConstructionGraph, FaceOwner, FaceRef,
    PartId, WeldSpec,
};

/// A staged source relocation. Runtime publication must revalidate its world endpoint
/// and initialize source bodies from these defaults, bypassing ordinary pose transfer.
#[derive(Clone, Debug)]
pub struct WeldPlacement {
    /// Graph with the source frames relocated and the selected faces welded.
    pub graph: ConstructionGraph,
    /// Every source part, including bodies connected through joints.
    pub source_parts: Vec<PartId>,
    /// Stable destination part used to resolve the latest destination motion.
    pub destination: PartId,
    /// One transform from source authored geometry into destination build space.
    pub source_transform: ConstructionFrame,
    /// Compiled default assembly after relocation and welding.
    pub creation: CompiledCreation,
}

impl WeldPlacement {
    /// Stages separate-creation placement without reading or modifying live poses.
    /// `transform` maps authored source points into the destination's authored frame.
    /// Terrain anchors must be supplied by the runtime. Connected bodies need the
    /// separate in-place weld path; relocating one side would change joint defaults.
    ///
    /// # Errors
    /// Rejects anchored or connected sources, insufficient contact, intersecting defaults,
    /// and invalid construction graphs. The input graph is unchanged on failure.
    pub fn stage(
        graph: &ConstructionGraph,
        source: FaceRef,
        destination: FaceRef,
        transform: ConstructionFrame,
        terrain_anchors: impl IntoIterator<Item = PartId>,
    ) -> Result<Self, String> {
        let (FaceOwner::Part(source_part), FaceOwner::Part(destination_part)) =
            (source.owner, destination.owner)
        else {
            return Err("Placement requires two construction features".to_owned());
        };
        let source_component = graph
            .structural_component(source_part, terrain_anchors)
            .map_err(|error| error.to_string())?;
        if source_component.contains(destination_part) {
            return Err("Connected bodies may only weld in place".to_owned());
        }
        if source_component.touches_authored_ground() {
            return Err("Terrain-anchored sources cannot relocate".to_owned());
        }
        let source_parts = source_component.parts().collect::<Vec<_>>();
        let destination_component = graph
            .structural_component(destination_part, [])
            .map_err(|error| error.to_string())?;
        let mut staged = graph.clone();
        staged
            .reframe_parts(source_parts.iter().copied(), transform)
            .map_err(|error| error.to_string())?;
        staged
            .weld_contact_square(
                &staged
                    .weld_mating_faces(source)
                    .map_err(|e| e.to_string())?,
                &staged
                    .weld_mating_faces(destination)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        let defaults = staged.compile().map_err(|error| error.to_string())?;
        let geometry = defaults
            .colliders
            .iter()
            .map(|collider| {
                let body = &defaults.compounds[collider.compound_index as usize];
                Ok(crate::WeldCollider::new(
                    collider,
                    ConstructionFrame::new(body.root_translation, body.root_rotation)
                        .map_err(|error| error.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (i, a) in defaults
            .colliders
            .iter()
            .enumerate()
            .filter(|(_, c)| source_parts.contains(&c.source_part))
        {
            for (j, _) in defaults
                .colliders
                .iter()
                .enumerate()
                .filter(|(_, c)| destination_component.contains(c.source_part))
            {
                if geometry[i].penetration(&geometry[j]) > 0.001 {
                    return Err(format!(
                        "The combined default pose intersects part {:?}",
                        a.source_part
                    ));
                }
            }
        }
        staged
            .apply(BuildCommand::Weld(WeldSpec {
                first: source,
                second: destination,
            }))
            .map_err(|error| error.to_string())?;
        staged
            .apply(BuildCommand::CancelPending)
            .map_err(|error| error.to_string())?;
        let creation = staged.compile().map_err(|error| error.to_string())?;
        Ok(Self {
            graph: staged,
            source_parts,
            destination: destination_part,
            source_transform: transform,
            creation,
        })
    }

    /// Every default source body receives the same placement, including articulated limbs.
    ///
    /// # Panics
    /// Panics if the compiled creation has been modified to contain a nonrigid transform.
    pub fn default_source_frames(
        &self,
        world_from_destination_build: ConstructionFrame,
    ) -> impl Iterator<Item = (usize, ConstructionFrame)> + '_ {
        self.creation
            .compounds
            .iter()
            .enumerate()
            .filter(|(_, body)| {
                body.source_parts
                    .iter()
                    .any(|part| self.source_parts.contains(part))
            })
            .map(move |(index, body)| {
                let authored = ConstructionFrame::new(body.root_translation, body.root_rotation)
                    .expect("compiled default body transform is rigid");
                (index, world_from_destination_build.compose(authored))
            })
    }
}
