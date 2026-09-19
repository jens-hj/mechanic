//! Generate and exercise the saved suspension comparison platform.
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, BumpStopSpec, ConstructionGraph,
    ConstructionMaterial, CreationDocument, CuboidSpec, FaceKind, FaceRef, GridRotation, JointKind,
    MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, PartId, ShockBodyEnd,
    ShockSpec, SpringSpec, SuspensionSpec, WeldSpec,
};
use mechanic_gpu::GpuPhysics;
use std::{error::Error, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn block(
    graph: &mut ConstructionGraph,
    size: [u8; 3],
    ticks: [i32; 3],
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) -> Result<PartId> {
    let spec = CuboidSpec::new(
        size,
        BuildPose::from_position_ticks(IVec3::from_array(ticks), GridRotation::default()),
    )?
    .with_material(material)
    .with_appearance(appearance);
    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec))? else {
        unreachable!()
    };
    Ok(id)
}

fn paint(rgb: [u8; 3]) -> MaterialAppearance {
    MaterialAppearance::new(
        MaterialColor::Dye(MaterialDye::new(rgb, 0.7).unwrap()),
        MaterialFinish::Painted,
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "keep the six declarative stations beside their shared assembly recipe"
)]
fn main() -> Result<()> {
    let destination = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("creations/suspension-playground.mech"),
        PathBuf::from,
    );
    let mut graph = ConstructionGraph::new();
    let base = block(
        &mut graph,
        [24, 1, 20],
        [0, 50, 0],
        ConstructionMaterial::Concrete,
        paint([62, 70, 80]),
    )?;
    graph.apply(BuildCommand::Weld(WeldSpec {
        first: FaceRef::ground(),
        second: FaceRef::part(base, FaceKind::NegativeY),
    }))?;
    let spring = SpringSpec::new(0.75, 0.16, 0.1375, 16, 0.0)?;
    let preloaded = SpringSpec::new(0.75, 0.16, 0.1375, 16, 0.075)?;
    let shock =
        |end, compression, rebound| ShockSpec::new(0.75, 0.1, end, 0.0, compression, rebound);
    let stations = [
        (
            "Free spring",
            [-700, -500],
            [246, 177, 55],
            SuspensionSpec::new(Some(spring), None, None)?,
        ),
        (
            "Preloaded spring",
            [0, -500],
            [242, 112, 68],
            SuspensionSpec::new(Some(preloaded), None, None)?,
        ),
        (
            "Light damping",
            [700, -500],
            [64, 194, 180],
            SuspensionSpec::new(
                Some(spring),
                Some(shock(ShockBodyEnd::Source, 1.0, 1.6)?),
                None,
            )?,
        ),
        (
            "Heavy damping",
            [-700, 500],
            [79, 149, 235],
            SuspensionSpec::new(
                Some(spring),
                Some(shock(ShockBodyEnd::Source, 30.0, 48.0)?),
                None,
            )?,
        ),
        (
            "Shock and bump stop",
            [0, 500],
            [186, 128, 231],
            SuspensionSpec::new(
                None,
                Some(shock(ShockBodyEnd::Source, 4.0, 6.4)?),
                Some(BumpStopSpec::new(0.1, 0.065)?),
            )?,
        ),
        (
            "Inverted with bump stop",
            [700, 500],
            [224, 91, 145],
            SuspensionSpec::new(
                Some(spring),
                Some(shock(ShockBodyEnd::Opposite, 2.0, 3.2)?),
                Some(BumpStopSpec::new(0.1, 0.065)?),
            )?,
        ),
    ];
    for (label, [x, z], rgb, spec) in stations {
        let appearance = paint(rgb);
        let pedestal = block(
            &mut graph,
            [3, 1, 3],
            [x, 150, z],
            ConstructionMaterial::Aluminium,
            appearance,
        )?;
        graph.apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::PositiveY),
            second: FaceRef::part(pedestal, FaceKind::NegativeY),
        }))?;
        let load = block(
            &mut graph,
            [2, 1, 2],
            [x, 550, z],
            ConstructionMaterial::Plastic,
            appearance,
        )?;
        let mut appearances = spec.appearances();
        appearances[0] = appearance;
        graph.apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(pedestal, FaceKind::PositiveY),
                FaceRef::part(load, FaceKind::NegativeY),
                Vec3::new(
                    f32::from(i16::try_from(x)?) * 0.0025,
                    0.5,
                    f32::from(i16::try_from(z)?) * 0.0025,
                ),
                Vec3::Y,
            )
            .with_kind(JointKind::Suspension(spec.with_appearances(appearances))),
        ))?;
        println!(
            "{label}: rate {:.1} N/mm, travel {:.1} mm",
            spec.spring().map_or(0.0, |s| s.rate() / 1000.0),
            spec.compression_limit().0 * 1000.0
        );
    }
    let document = CreationDocument::from_graph(&graph, "Suspension - Motion Playground", &[]);
    let encoded = ron::ser::to_string_pretty(&document, ron::ser::PrettyConfig::default())?;
    let loaded: CreationDocument = ron::from_str(&encoded)?;
    let compiled = loaded.into_graph()?.graph.compile()?;
    assert_eq!(compiled.bearings.len(), 6);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    println!("Adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let gpu = GpuPhysics::new(&device, &queue, &compiled)?;
    for tick in 1..=600 {
        gpu.dispatch_tick(&device, &queue, tick);
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let diagnostics = gpu.read_last_tick(&device)?;
        assert_eq!(diagnostics.error_flags, 0, "tick {tick}: {diagnostics:?}");
    }
    let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0)?;
    assert!(
        snapshot
            .iter()
            .all(|body| body.position.iter().all(|v| v.is_finite()))
    );
    // Saved creations belong in the Garage's 5–10 m editable band, matching
    // the linear-bearing playground. Ground welds belong only to this GPU rig.
    let mut document = document;
    document.welds.retain(|weld| {
        !matches!(weld.first.owner, mechanic_core::FaceOwnerDoc::Ground)
            && !matches!(weld.second.owner, mechanic_core::FaceOwnerDoc::Ground)
    });
    document.transform_cardinal(0, IVec3::new(0, 40, 0));
    let encoded = ron::ser::to_string_pretty(&document, ron::ser::PrettyConfig::default())?;
    ron::from_str::<CreationDocument>(&encoded)?
        .into_graph()?
        .graph
        .compile()?;
    std::fs::write(&destination, encoded)?;
    println!(
        "Saved {}: {} parts, six working suspension stations; 600 GPU ticks, zero failure flags.",
        destination.display(),
        graph.part_count()
    );
    Ok(())
}
