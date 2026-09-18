//! Shared exact fixtures for the GPU runner and compiled CPU inertia experiment.

use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation, ConstructionGraph,
    CuboidSpec, FaceKind, FaceRef, GridRotation,
};

pub(crate) fn build_bearing_chain(bearing_count: usize) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch((0..=bearing_count).map(|index| {
            let x = i32::try_from(index.saturating_mul(4)).expect("chain coordinate fits i32");
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                )
                .expect("one-metre cube is in range"),
            )
        }))
        .map_err(|error| format!("bearing-chain part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    graph
        .apply_batch((0..bearing_count).map(|index| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "exact existing bounded benchmark lattice"
            )]
            let anchor = Vec3::new(index as f32 + 0.5, 0.5, 0.0);
            BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[index], FaceKind::PositiveX),
                FaceRef::part(parts[index + 1], FaceKind::NegativeX),
                anchor,
                Vec3::X,
            ))
        }))
        .map_err(|error| format!("bearing-chain joint generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}
