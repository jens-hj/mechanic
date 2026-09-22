//! Graph values written out as document rows.

use super::doc::{
    DriveDwellDoc, DriveLimitsDoc, DriveProgramDoc, DriveStateDoc, DriveTriggerDoc,
    EdgeChainRefDoc, FaceOwnerDoc, FaceRefDoc, MaterialLayerDoc, PartDoc, SolidOwnerDoc,
    TopologyKeyDoc, TopologySourceDoc,
};
use crate::{
    DriveLimits, DriveProgram, EdgeChainRef, FaceOwner, FaceRef, GridDimension, PartId, PartSpec,
    ShapeFeatureId, SolidOwner, TopologySource,
};
use std::collections::HashMap;

pub(super) fn face_doc(
    face: FaceRef,
    parts: &HashMap<PartId, u32>,
    features: &HashMap<ShapeFeatureId, u32>,
) -> FaceRefDoc {
    FaceRefDoc {
        owner: match face.owner {
            FaceOwner::Part(part) => FaceOwnerDoc::Part(
                *parts
                    .get(&part)
                    .expect("every referenced part is live in the graph it came from"),
            ),
            FaceOwner::Ground => FaceOwnerDoc::Ground,
        },
        face: face.face,
        patch: face.patch.map(|patch| TopologyKeyDoc {
            source: match patch.source {
                TopologySource::Base => TopologySourceDoc::Base,
                TopologySource::Feature(feature) => TopologySourceDoc::Feature(
                    *features
                        .get(&feature)
                        .expect("a referenced generated patch has a live feature"),
                ),
            },
            local: patch.local,
        }),
    }
}

pub(super) fn edge_chain_doc(
    target: EdgeChainRef,
    parts: &HashMap<PartId, u32>,
    regions: &HashMap<crate::RegionId, u32>,
    features: &HashMap<ShapeFeatureId, u32>,
) -> EdgeChainRefDoc {
    EdgeChainRefDoc {
        owner: match target.owner {
            SolidOwner::Part(part) => SolidOwnerDoc::Part(
                *parts
                    .get(&part)
                    .expect("every feature part owner is live in its graph"),
            ),
            SolidOwner::Region(region) => SolidOwnerDoc::Region(
                *regions
                    .get(&region)
                    .expect("every feature region owner is live in its graph"),
            ),
        },
        edge: TopologyKeyDoc {
            source: match target.edge.source {
                TopologySource::Base => TopologySourceDoc::Base,
                TopologySource::Feature(feature) => TopologySourceDoc::Feature(
                    *features
                        .get(&feature)
                        .expect("generated topology references a live earlier feature"),
                ),
            },
            local: target.edge.local,
        },
    }
}

pub(super) fn layer_docs(layers: crate::MaterialLayers) -> Vec<MaterialLayerDoc> {
    layers
        .iter()
        .map(|layer| MaterialLayerDoc {
            face: layer.face,
            thickness: layer.thickness,
            material: layer.material,
            appearance: layer.appearance,
        })
        .collect()
}

pub(super) fn part_doc(spec: PartSpec, transmission_parent: Option<u32>) -> PartDoc {
    match spec {
        PartSpec::Cuboid(cuboid) => {
            let core = cuboid.without_layers();
            PartDoc::Cuboid {
                dimensions: core.dimensions.map(GridDimension::units),
                pose: core.pose.into(),
                material: core.material,
                appearance: core.appearance,
                layers: layer_docs(cuboid.layers()),
                rack: cuboid.rack().map(|rack| super::doc::RackDoc {
                    module_ticks: rack.module_ticks(),
                    face: rack.face(),
                    along: rack.along(),
                }),
            }
        }
        PartSpec::Cylinder(cylinder) => {
            let core = cylinder.without_layers();
            PartDoc::Cylinder {
                outer_diameter: core.dimensions.outer_diameter(),
                inner_diameter: core.dimensions.inner_diameter(),
                length_units: core.dimensions.axial_length_units(),
                sweep_degrees: core.dimensions.sweep_angle_degrees(),
                pose: core.pose.into(),
                material: core.material,
                appearance: core.appearance,
                layers: layer_docs(cylinder.layers()),
                spiral: cylinder.spiral().map(spiral_doc),
                gear: cylinder.gear().map(|gear| super::doc::GearDoc {
                    module_ticks: gear.module_ticks(),
                    teeth: gear.teeth(),
                    kind: gear.kind(),
                }),
            }
        }
        PartSpec::PipeBend(bend) => PartDoc::PipeBend {
            outer_diameter: bend.dimensions.outer_diameter(),
            inner_diameter: bend.dimensions.inner_diameter(),
            span_blocks: bend.dimensions.span_blocks(),
            pose: bend.pose.into(),
            material: bend.material,
            appearance: bend.appearance,
        },
        PartSpec::PipeJunction(junction) => PartDoc::PipeJunction {
            outer_diameter: junction.dimensions.outer_diameter(),
            inner_diameter: junction.dimensions.inner_diameter(),
            arms: junction.arms.bits(),
            pose: junction.pose.into(),
            material: junction.material,
            appearance: junction.appearance,
        },
        PartSpec::Controller(controller) => PartDoc::Controller {
            pose: controller.pose.into(),
        },
        PartSpec::Engine(engine) => PartDoc::Engine {
            kind: engine.kind,
            pose: engine.pose.into(),
        },
        PartSpec::Transmission(transmission) => PartDoc::Transmission {
            parent: transmission_parent.expect("every transmission has a live graph parent"),
            pose: transmission.pose.into(),
        },
        PartSpec::Servo(servo) => PartDoc::Servo {
            pose: servo.pose.into(),
        },
        PartSpec::Seat(seat) => PartDoc::Seat {
            pose: seat.pose.into(),
        },
        PartSpec::Dial(spec) => PartDoc::Dial {
            size: spec.size,
            pose: spec.pose.into(),
        },
        PartSpec::Button(spec) => PartDoc::Button {
            size: spec.size,
            pose: spec.pose.into(),
        },
        PartSpec::Input(input) => PartDoc::Input {
            pose: input.pose.into(),
        },
        PartSpec::DimensionLink(link) => PartDoc::DimensionLink {
            id: link.id,
            pose: link.pose.into(),
        },
    }
}

pub(super) fn limits_doc(limits: DriveLimits) -> DriveLimitsDoc {
    let torque = limits.max_torque_newton_meters();
    DriveLimitsDoc {
        max_speed_rad_s: limits.max_speed_rad_s(),
        max_torque_newton_meters: torque.is_finite().then_some(torque),
        angle_limits: limits.angle_limits(),
    }
}

pub(super) fn program_doc(program: &DriveProgram) -> DriveProgramDoc {
    DriveProgramDoc {
        loops: program.loops(),
        states: program
            .states()
            .iter()
            .map(|state| DriveStateDoc {
                target: state.target(),
                dwell: state.dwell().map(|dwell| DriveDwellDoc {
                    seconds: dwell.seconds(),
                    next: dwell.next(),
                }),
                trigger: state.trigger().map(|trigger| DriveTriggerDoc {
                    key: trigger.key().symbol(),
                    release: trigger.release(),
                }),
            })
            .collect(),
    }
}

fn spiral_doc(spiral: crate::SpiralSpec) -> super::doc::SpiralDoc {
    let points = |profile: crate::SpiralProfile| {
        profile
            .points()
            .iter()
            .map(|point| [point.position_ticks, point.depth_ticks])
            .collect()
    };
    super::doc::SpiralDoc {
        pitch_ticks: spiral.pitch_ticks(),
        starts: spiral.starts(),
        hand: spiral.hand(),
        outer: points(spiral.outer()),
        inner: points(spiral.inner()),
        taper: spiral.taper().map(|taper| super::doc::SpiralTaperDoc {
            end: taper.end,
            length_ticks: taper.length_ticks,
            tip_diameter_ticks: taper.tip_diameter_ticks,
        }),
    }
}
