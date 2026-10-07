//! `energy-drift`: frictionless joints and free bodies left to run for thirty
//! seconds without contacts, at 1, 2, 4 and 8 substeps. Reports how far their
//! kinetic plus potential energy wanders and whether it trends.

use std::error::Error;

use bevy_math::{DVec3, IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation, ConstructionGraph,
    CuboidSpec, CylinderDimensions, CylinderSpec, FaceKind, FaceRef, GridRotation, PartId,
};
use mechanic_physics::{CpuMachine, MachineKinematics, MachineState, SoftStepConfig};
use serde_json::json;

use super::GRAVITY;

const TICKS: usize = 30 * 60;

pub(super) fn run() -> Result<(), Box<dyn Error>> {
    type Case = (
        &'static str,
        fn() -> Result<(CompiledCreation, MachineState), Box<dyn Error>>,
        bool,
    );
    let cases: [Case; 6] = [
        ("anchored-balanced-spin", || balanced(true, false), true),
        (
            "anchored-unbalanced-pendulum",
            || unbalanced(true, 0.0),
            true,
        ),
        ("anchored-unbalanced-spin", || unbalanced(true, 10.0), true),
        ("free-balanced-spin-wobble", || balanced(false, true), false),
        ("free-unbalanced-spin", || unbalanced(false, 20.0), false),
        ("free-wheel-150-rad-s-wobble", wheel, false),
    ];
    for (name, build, gravity) in cases {
        for substeps in [1, 2, 4, 8] {
            let (creation, state) = build()?;
            measure(
                name,
                &creation,
                state,
                if gravity { GRAVITY } else { DVec3::ZERO },
                substeps,
            )?;
        }
    }
    Ok(())
}

#[expect(clippy::cast_precision_loss, reason = "tick counts are far below 2^52")]
fn measure(
    name: &str,
    creation: &CompiledCreation,
    state: MachineState,
    gravity: DVec3,
    substeps: u32,
) -> Result<(), Box<dyn Error>> {
    let energy =
        |creation: &CompiledCreation, state: &MachineState| {
            MachineKinematics::assemble(creation, &state.poses, &state.coordinates)?
                .mechanical_energy(creation, &state.velocities, gravity)
        };
    let initial = energy(creation, &state)?;
    let settings = SoftStepConfig {
        substeps,
        ..SoftStepConfig::default()
    };
    let mut machine = CpuMachine::new(creation.clone(), 1, state)?;
    let (mut lowest, mut highest) = (0.0_f64, 0.0_f64);
    let (mut first_half, mut second_half) = (0.0, 0.0);
    let mut degraded = None;
    for tick in 1..=TICKS {
        machine.step(gravity, &settings, &[], &[], None)?;
        if machine.diagnostics().degraded {
            degraded = Some(tick);
            break;
        }
        let change = energy(creation, &machine.snapshot().state)? - initial;
        lowest = lowest.min(change);
        highest = highest.max(change);
        if tick <= TICKS / 2 {
            first_half += change;
        } else {
            second_half += change;
        }
    }
    let state = &machine.snapshot().state;
    let half = (TICKS / 2) as f64;
    let record = json!({
        "scenario": "energy-drift",
        "case": name,
        "substeps": substeps,
        "seconds": TICKS / 60,
        "initial_energy_j": initial,
        "lowest_change_j": lowest,
        "highest_change_j": highest,
        // Mean change over the second fifteen seconds less the first: a
        // bounded wobble gives about zero, a drift its rate times 15 s.
        "trend_j": (second_half - first_half) / half,
        "fastest_final": state.velocities.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
        "degraded_tick": degraded,
    });
    println!("{record}");
    Ok(())
}

fn spawn(
    graph: &mut ConstructionGraph,
    ticks: IVec3,
    dimensions: [u8; 3],
) -> Result<PartId, Box<dyn Error>> {
    match graph.apply(BuildCommand::Spawn(CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(ticks, GridRotation::default()),
    )?))? {
        BuildOutcome::Spawned(part) => Ok(part),
        _ => Err("spawn expected".into()),
    }
}

// A 1 m steel block 5 m up with a bearing along its +X face, carrying either a
// centred 0.5 m block or a 1.5 m arm whose centre of mass sits 0.625 m off the
// axis.
fn rotor(anchored: bool, unbalanced: bool) -> Result<CompiledCreation, Box<dyn Error>> {
    let mut graph = ConstructionGraph::new();
    let root = spawn(&mut graph, IVec3::Y * 2000, [4, 4, 4])?;
    let (tip, anchor) = if unbalanced {
        (
            spawn(&mut graph, IVec3::new(300, 2000, 400), [2, 2, 6])?,
            0.375,
        )
    } else {
        (spawn(&mut graph, IVec3::new(300, 2000, 0), [2, 2, 2])?, 0.0)
    };
    graph.apply(BuildCommand::AddBearing(BearingSpec::new(
        FaceRef::part(root, FaceKind::PositiveX),
        FaceRef::part(tip, FaceKind::NegativeX),
        Vec3::new(0.5, 5.0, anchor),
        Vec3::X,
    )))?;
    Ok(graph.compile_with_static_parts(if anchored { vec![root] } else { vec![] })?)
}

fn root(creation: &CompiledCreation) -> Result<usize, Box<dyn Error>> {
    creation
        .loop_topology
        .body_parents
        .iter()
        .position(|parents| parents.is_root)
        .ok_or_else(|| "no root body".into())
}

// The tip spins at 20 rad/s; a free body also tumbles 0.5 rad/s about Y and
// 0.3 rad/s about Z.
fn balanced(
    anchored: bool,
    wobble: bool,
) -> Result<(CompiledCreation, MachineState), Box<dyn Error>> {
    let creation = rotor(anchored, false)?;
    let mut state = MachineState::at_rest(&creation);
    state.velocities[creation.dynamics.coordinate_velocities[0]] =
        if anchored { 10.0 } else { 20.0 };
    if wobble {
        let rows = creation.dynamics.body_velocities[root(&creation)?].clone();
        state.velocities[rows][3..6].copy_from_slice(&[0.0, 0.5, 0.3]);
    }
    Ok((creation, state))
}

// The arm spins at `speed`, or at zero is released horizontal. A free pair has
// its momentum cancelled so it whirls in place.
fn unbalanced(
    anchored: bool,
    speed: f64,
) -> Result<(CompiledCreation, MachineState), Box<dyn Error>> {
    let creation = rotor(anchored, true)?;
    let mut state = MachineState::at_rest(&creation);
    state.velocities[creation.dynamics.coordinate_velocities[0]] = speed;
    if !anchored {
        let model = MachineKinematics::assemble(&creation, &state.poses, &state.coordinates)?;
        let (momentum, mass) = model
            .body_motions(&state.velocities)?
            .iter()
            .zip(&creation.dynamics.inertias)
            .fold((DVec3::ZERO, 0.0), |(momentum, mass), (motion, inertia)| {
                let body = f64::from(inertia.mass);
                (momentum + motion.linear * body, mass + body)
            });
        let rows = creation.dynamics.body_velocities[root(&creation)?].clone();
        let velocity = -momentum / mass;
        state.velocities[rows][0..3].copy_from_slice(&velocity.to_array());
    }
    Ok((creation, state))
}

// A 0.95 m steel wheel spinning 150 rad/s about its axle with a 3 rad/s wobble.
fn wheel() -> Result<(CompiledCreation, MachineState), Box<dyn Error>> {
    let mut graph = ConstructionGraph::new();
    graph.apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
        CylinderDimensions::new(0.95, 0.0, 0.25)?,
        BuildPose::from_position_ticks(IVec3::Y * 2000, GridRotation::new(1, 0, 0)),
    )))?;
    let creation = graph.compile()?;
    let mut state = MachineState::at_rest(&creation);
    state.velocities[3..6].copy_from_slice(&[3.0, 0.0, -150.0]);
    Ok((creation, state))
}
