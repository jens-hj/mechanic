//! Generate a wide-track, four-wheel-drive suspension test car and exercise it on the GPU.
use bevy_math::{Quat, Vec3};
use mechanic_core::{
    BearingDoc, BumpStopSpec, ConstructionGraph, ConstructionMaterial, CreationDocument,
    DriveTarget, FaceKind, FaceOwnerDoc, FaceRefDoc, InputSeatLinkDoc, JointKind,
    MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, PartDoc, PoseDoc, RigidLinkDoc,
    SeatControllerLinkDoc, ShockBodyEnd, ShockSpec, SpringSpec, SuspensionSpec,
};
use mechanic_gpu::{GpuMechanismDrive, GpuPhysics};
use std::{error::Error, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn pose(x: i32, y: i32, z: i32) -> PoseDoc {
    PoseDoc {
        translation_ticks: [x, y, z],
        rotation: [0; 3],
    }
}

fn block(doc: &mut CreationDocument, dimensions: [u8; 3], pose: PoseDoc) -> u32 {
    let id = u32::try_from(doc.parts.len()).unwrap();
    doc.parts.push(PartDoc::Cuboid {
        dimensions,
        pose,
        material: ConstructionMaterial::Wood,
        appearance: MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([36, 126, 218], 0.8).unwrap()),
            MaterialFinish::Painted,
        ),
        layers: Vec::new(),
    });
    doc.part_frames.push(0);
    id
}

fn face(part: u32, face: FaceKind) -> FaceRefDoc {
    FaceRefDoc {
        owner: FaceOwnerDoc::Part(part),
        face,
        patch: None,
    }
}

fn bearing(
    doc: &mut CreationDocument,
    source: FaceRefDoc,
    target: FaceRefDoc,
    anchor: [f32; 3],
    axis: [f32; 3],
    kind: JointKind,
) -> u32 {
    let id = u32::try_from(doc.bearings.len()).unwrap();
    doc.bearings.push(BearingDoc {
        source,
        target: Some(target),
        anchor,
        axis,
        kind,
        outer_diameter: 0.15,
        inner_diameter: 0.0,
    });
    id
}

#[expect(clippy::too_many_lines)]
fn main() -> Result<()> {
    let destination = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("creations/suspension-car.mech"),
        PathBuf::from,
    );
    let template: CreationDocument =
        ron::from_str(include_str!("../tests/fixtures/front_steered_car.mech"))?;
    let mut doc =
        CreationDocument::from_graph(&ConstructionGraph::new(), "Suspension - Sprint AWD", &[]);
    let chassis = block(&mut doc, [12, 1, 2], pose(0, 500, 0));
    let mut crossmembers = Vec::new();
    for x in [-500, 500] {
        let crossmember = block(&mut doc, [1, 1, 10], pose(x, 500, 0));
        doc.rigid_links.push(RigidLinkDoc {
            first: chassis,
            second: crossmember,
        });
        crossmembers.push(crossmember);
    }
    let spring = SpringSpec::new(0.25, 0.16, 0.13, 3, 0.005)?;
    let suspension = SuspensionSpec::new(
        Some(spring),
        Some(ShockSpec::new(
            0.25,
            0.1,
            ShockBodyEnd::Source,
            0.0,
            8.0,
            12.0,
        )?),
        Some(BumpStopSpec::new(0.025, 0.065)?),
    )?;
    let mut steering = Vec::new();
    let mut axles = Vec::new();
    for x in [-500_i16, 500] {
        for z in [500_i16, -500] {
            let xf = f32::from(x) * 0.0025;
            let zf = f32::from(z) * 0.0025;
            let sign = z.signum();
            let carrier = block(&mut doc, [1, 1, 1], pose(i32::from(x), 300, i32::from(z)));
            bearing(
                &mut doc,
                face(crossmembers[usize::from(x > 0)], FaceKind::NegativeY),
                face(carrier, FaceKind::PositiveY),
                [xf, 1.125, zf],
                [0.0, -1.0, 0.0],
                JointKind::Suspension(suspension),
            );
            let knuckle = block(&mut doc, [1, 1, 1], pose(i32::from(x), 200, i32::from(z)));
            if x < 0 {
                steering.push(bearing(
                    &mut doc,
                    face(carrier, FaceKind::NegativeY),
                    face(knuckle, FaceKind::PositiveY),
                    [xf, 0.625, zf],
                    [0.0, -1.0, 0.0],
                    JointKind::Rotational,
                ));
            } else {
                doc.rigid_links.push(RigidLinkDoc {
                    first: carrier,
                    second: knuckle,
                });
            }
            let wheel = u32::try_from(doc.parts.len())?;
            doc.parts.push(PartDoc::Cylinder {
                outer_diameter: 0.9,
                inner_diameter: 0.0,
                length_units: 1,
                sweep_degrees: 360,
                pose: PoseDoc {
                    translation_ticks: [i32::from(x), 200, i32::from(sign) * 600],
                    rotation: if sign > 0 { [1, 0, 0] } else { [1, 2, 2] },
                },
                material: ConstructionMaterial::Rubber,
                appearance: MaterialAppearance::default(),
                layers: Vec::new(),
            });
            doc.part_frames.push(0);
            axles.push(bearing(
                &mut doc,
                face(
                    knuckle,
                    if sign > 0 {
                        FaceKind::PositiveZ
                    } else {
                        FaceKind::NegativeZ
                    },
                ),
                face(wheel, FaceKind::NegativeY),
                [xf, 0.5, zf + f32::from(sign) * 0.125],
                [0.0, 0.0, f32::from(sign)],
                JointKind::Rotational,
            ));
        }
    }
    // Keep the proven seat/input/controller wiring and engine allocation.
    let electronics = u32::try_from(doc.parts.len())?;
    for (offset, original) in template.parts[62..=70].iter().enumerate() {
        let mut part = original.clone();
        let (PartDoc::Input { pose: p }
        | PartDoc::Seat { pose: p }
        | PartDoc::Servo { pose: p }
        | PartDoc::Engine { pose: p, .. }
        | PartDoc::Controller { pose: p }) = &mut part
        else {
            unreachable!()
        };
        // Preserve physical engine/controller port adjacency on the low deck.
        p.translation_ticks[1] -= 2550;
        p.translation_ticks[2] -= 150;
        doc.parts.push(part);
        doc.part_frames.push(0);
        doc.rigid_links.push(RigidLinkDoc {
            first: chassis,
            second: electronics + u32::try_from(offset)?,
        });
    }
    for weld in &template.welds {
        if let (FaceOwnerDoc::Part(a), FaceOwnerDoc::Part(b)) =
            (weld.first.owner, weld.second.owner)
            && (62..=70).contains(&a)
            && (62..=70).contains(&b)
        {
            let mut weld = *weld;
            weld.first.owner = FaceOwnerDoc::Part(electronics + a - 62);
            weld.second.owner = FaceOwnerDoc::Part(electronics + b - 62);
            doc.welds.push(weld);
        }
    }
    doc.input_seat_links.push(InputSeatLinkDoc {
        input: electronics,
        seat: electronics + 1,
    });
    doc.seat_controller_links.push(SeatControllerLinkDoc {
        seat: electronics + 1,
        controller: electronics + 8,
    });
    for mut link in template.drive_links {
        link.controller = electronics + 8;
        if link.bearing < 2 {
            link.bearing = steering[usize::try_from(link.bearing)?];
            link.limits.angle_limits = Some((-0.21, 0.21));
            for state in &mut link.program.states {
                if let DriveTarget::Angle(angle) = &mut state.target
                    && *angle != 0.0
                {
                    *angle = angle.signum() * 0.20;
                }
            }
            link.name = "Steering".into();
        } else {
            // Match the fixture axle ordering to the new corner ordering.
            let old = link.bearing;
            link.bearing = axles[match old {
                2 => 0,
                3 => 1,
                4 => 3,
                5 => 2,
                _ => unreachable!(),
            }];
            link.limits.max_speed_rad_s = 40.0;
            for state in &mut link.program.states {
                if let DriveTarget::Speed(speed) = &mut state.target {
                    *speed = if *speed > 0.0 {
                        40.0
                    } else if *speed < 0.0 {
                        -12.0
                    } else {
                        0.0
                    };
                }
            }
            link.name = "AWD".into();
        }
        doc.drive_links.push(link);
    }
    let encoded = ron::ser::to_string_pretty(&doc, ron::ser::PrettyConfig::default())?;
    let creation = ron::from_str::<CreationDocument>(&encoded)?
        .into_graph()?
        .graph
        .compile()?;
    assert_eq!(creation.bearings.len(), 10);
    println!(
        "Mass {:.1} kg; spring {:.1} N/mm; travel {:.1} mm",
        creation
            .compounds
            .iter()
            .map(|c| c.mass_properties.mass)
            .sum::<f32>(),
        spring.rate() / 1000.0,
        suspension.compression_limit().0 * 1000.0
    );
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    println!("Adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let gpu = GpuPhysics::new(&device, &queue, &creation)?;
    let mut drives: Vec<_> = creation
        .coordinate_drives
        .iter()
        .copied()
        .map(GpuMechanismDrive::from)
        .collect();
    let chassis_body = usize::try_from(
        creation
            .part_to_compound
            .iter()
            .find(|(part, _)| part.index() == chassis)
            .unwrap()
            .1,
    )?;
    let mut max_speed = 0.0_f32;
    let mut previous = Vec3::ZERO;
    let mut start = Vec3::ZERO;
    for tick in 1..=1800 {
        if tick == 181 {
            for link in &doc.drive_links {
                let coordinate = usize::try_from(
                    creation.bearings[usize::try_from(link.bearing)?]
                        .coordinate_index
                        .unwrap(),
                )?;
                if link.name == "AWD" {
                    drives[coordinate].target_speed = if link.reversed { -40.0 } else { 40.0 };
                }
            }
            gpu.write_mechanism_drives(&queue, &drives)?;
        }
        if tick == 1201 || tick == 1381 || tick == 1501 || tick == 1681 {
            for link in &doc.drive_links {
                if link.name == "Steering" {
                    let coordinate = usize::try_from(
                        creation.bearings[usize::try_from(link.bearing)?]
                            .coordinate_index
                            .unwrap(),
                    )?;
                    drives[coordinate].target_angle = match tick {
                        1201 => -0.20,
                        1501 => 0.20,
                        _ => 0.0,
                    };
                }
            }
            gpu.write_mechanism_drives(&queue, &drives)?;
        }
        gpu.dispatch_tick(&device, &queue, tick);
        if tick % 30 == 0 {
            let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0)?;
            let chassis = &snapshot[chassis_body];
            let p = Vec3::from_slice(&chassis.position[..3]);
            let up = Quat::from_array(chassis.rotation) * Vec3::Y;
            assert!(
                p.is_finite() && up.y > 0.95,
                "unstable tick {tick}: {p:?}, up {up:?}"
            );
            if tick == 180 {
                start = p;
            }
            if tick > 180 {
                max_speed = max_speed.max((p - previous).length() * 2.0);
            }
            if tick % 180 == 0 {
                println!(
                    "tick {tick}: speed {:.1}, position {p:?}",
                    (p - previous).length() * 7.2
                );
            }
            previous = p;
            assert_eq!(gpu.read_last_tick(&device)?.error_flags, 0);
        }
    }
    assert!(max_speed > 10.0, "car must exceed 36 km/h: {max_speed}");
    println!(
        "1800 ticks, zero failure flags, peak {:.1} km/h, travel {:.1} m",
        max_speed * 3.6,
        (previous - start).length()
    );
    doc.transform_cardinal(0, bevy_math::IVec3::new(0, 24, 0));
    let encoded = ron::ser::to_string_pretty(&doc, ron::ser::PrettyConfig::default())?;
    ron::from_str::<CreationDocument>(&encoded)?
        .into_graph()?
        .graph
        .compile()?;
    std::fs::write(&destination, encoded)?;
    println!("Saved {}", destination.display());
    Ok(())
}
