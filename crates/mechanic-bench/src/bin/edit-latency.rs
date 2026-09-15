//! Reproducible CPU geometry timings for interactive construction edits.

use std::{hint::black_box, time::Instant};

use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, CylinderDimensions,
    CylinderSpec, EdgeChainRef, EdgeTreatment, PipeArms, PipeJunctionDimensions, PipeJunctionSpec,
    ShapeFeature, SolidOwner,
};

fn main() {
    for (name, command) in [
        (
            "block",
            BuildCommand::Spawn(CuboidSpec::new([2; 3], BuildPose::default()).unwrap()),
        ),
        (
            "pipe",
            BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(0.20, 0.10, 0.5).unwrap(),
                BuildPose::default(),
            )),
        ),
        (
            "junction",
            BuildCommand::SpawnPipeJunction(PipeJunctionSpec::new(
                PipeJunctionDimensions::new(0.20, 0.10).unwrap(),
                PipeArms::from_bits(0b01_0011).unwrap(),
                BuildPose::default(),
            )),
        ),
    ] {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let started = Instant::now();
        for _ in 0..100 {
            black_box(graph.evaluated_solid(owner).unwrap());
        }
        println!(
            "{name}: 100 boundary queries {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        let started = Instant::now();
        for _ in 0..100 {
            black_box(graph.evaluated_solid_shared(owner).unwrap());
        }
        println!(
            "{name}: 100 shared cached queries {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        if name == "block" {
            let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
            for treatment in [EdgeTreatment::Fillet, EdgeTreatment::Chamfer] {
                let started = Instant::now();
                for amount in 1..=20 {
                    let mut preview = graph.clone();
                    preview
                        .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                            [EdgeChainRef { owner, edge }],
                            treatment,
                            amount,
                        )))
                        .unwrap();
                    for _ in 0..10 {
                        black_box(preview.evaluated_solid(owner).unwrap());
                    }
                }
                println!(
                    "{treatment:?}: 20 preview steps with 10 queries {:.3} ms",
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }
}
