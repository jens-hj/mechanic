//! Read-only instruments for the vehicle occupied by the player.

#![allow(clippy::wildcard_imports)] // Mosaic's authoring vocabulary.

use bevy::prelude::Vec3;
use bevy_mosaic::ui::*;
use mechanic_core::{EngineKind, GearboxConfig, PartId, ShiftMode};
use mosaic_core::Rect;
use mosaic_macros::{component, view};

use super::components::{PanelSurface, PanelSurfaceProps};
#[allow(unused_imports)] // Consumed by view! expansion.
use super::styles::*;
use super::theme::{accent, ink};
use crate::sequencer::{DriveSequencer, GearboxRuntime};
use crate::{AppSimulation, measured_engine_speeds};

const PANEL_WIDTH: f32 = 236.0;
const INSET: f32 = 18.0;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Model {
    pub(crate) open: bool,
    speed: String,
    engines: Vec<EngineLane>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EngineLane {
    title: String,
    rpm: String,
    gear: String,
}

pub(crate) fn capture(
    seat: Option<PartId>,
    simulation: &AppSimulation,
    sequencer: &DriveSequencer,
    gearboxes: &GearboxRuntime,
) -> Model {
    let graph = &simulation.published_graph;
    let Some(seat) = seat.filter(|seat| graph.is_seat(*seat)) else {
        return Model::default();
    };
    let mut model = Model {
        open: true,
        speed: vehicle_speed_kmh(simulation, seat)
            .map_or_else(|| "—".to_owned(), |speed| format!("{speed:.1}")),
        engines: Vec::new(),
    };
    if let Some(controller) = graph.seat_controller(seat)
        && let Some(inventory) = graph.actuator_inventory(controller)
    {
        let speeds = measured_engine_speeds(graph, simulation, sequencer);
        for (kind, count) in [
            (EngineKind::Gas, inventory.gas_engines),
            (EngineKind::Electric, inventory.electric_engines),
        ] {
            if count == 0 {
                continue;
            }
            let config = graph.gearbox_config(controller, kind).ok();
            let output_speed = speeds.iter().find_map(|(candidate, family, speed)| {
                (*candidate == controller && *family == kind).then_some(*speed)
            });
            model.engines.push(engine_lane(
                kind,
                count,
                config.as_ref(),
                gearboxes.active_gear(controller, kind),
                output_speed,
            ));
        }
    }
    model
}

/// Seat-body translation, not wheel spin or camera motion, per simulated second.
#[allow(clippy::cast_precision_loss)]
fn vehicle_speed_kmh(simulation: &AppSimulation, seat: PartId) -> Option<f32> {
    let creation = simulation.creation.as_ref()?;
    let body = creation
        .part_to_compound
        .iter()
        .find_map(|(part, body)| (*part == seat).then_some(*body as usize))?;
    let ticks = simulation
        .snapshot_tick
        .saturating_sub(simulation.previous_snapshot_tick);
    if ticks == 0 {
        return None;
    }
    let previous = simulation.previous_transforms.get(body)?;
    let current = simulation.transforms.get(body)?;
    let displacement =
        Vec3::from_slice(&current.position[..3]) - Vec3::from_slice(&previous.position[..3]);
    Some(displacement.length() / (ticks as f32 * mechanic_core::TICK_SECONDS_F32) * 3.6)
}

fn engine_lane(
    kind: EngineKind,
    count: u32,
    config: Option<&GearboxConfig>,
    gear: Option<usize>,
    output_speed: Option<f32>,
) -> EngineLane {
    let family = match kind {
        EngineKind::Gas => "GAS",
        EngineKind::Electric => "ELECTRIC",
    };
    let mode = config.map_or("—", |config| match config.mode() {
        ShiftMode::Auto => "AUTO",
        ShiftMode::Manual => "MANUAL",
    });
    let engaged = config.zip(gear).and_then(|(config, gear)| {
        config
            .ratios()
            .get(gear)
            .map(|ratio| (config, gear, *ratio))
    });
    let gear = engaged.map_or_else(
        || if config.is_some() { "N" } else { "—" }.to_owned(),
        |(config, index, _)| {
            let reverse = usize::from(config.reverse_gears());
            if kind == EngineKind::Gas && index < reverse {
                format!("R{}", index + 1)
            } else {
                format!("{}", index.saturating_sub(reverse) + 1)
            }
        },
    );
    // This engine model has no independent crankshaft state in neutral.
    let rpm = engaged.zip(output_speed).map_or_else(
        || "— RPM".to_owned(),
        |((_, _, ratio), speed)| {
            let rpm = speed.abs() * ratio * 60.0 / core::f32::consts::TAU;
            format!("{rpm:.0} RPM")
        },
    );
    EngineLane {
        title: format!("{family} ×{count} · {mode}"),
        rpm,
        gear: format!("GEAR {gear}"),
    }
}

#[component]
pub(crate) fn DrivingOverlay(model: State<Model>, viewport: State<Size>) -> Element {
    let size = State::new(Size::ZERO);
    let at = move || {
        let window = viewport.get();
        (
            Length::px((window.width - PANEL_WIDTH - INSET).max(INSET)),
            Length::px((window.height - size.get().height - INSET).max(INSET)),
        )
    };
    view! {
        col width:min-content height:min-content nohit
            translate:(x:{ at().0 } y:{ at().1 })
            @layout:{ move |bounds: Rect| {
                if size.get_untracked() != bounds.size {
                    size.set(bounds.size);
                }
            } } {
            PanelSurface elevated:false width:(Length::px(PANEL_WIDTH)) height:min-content
                gap:10px pad:14px {
                row width:fill height:44px align:center justify:between {
                    text font-size:36px font-weight:700 font-color:accent.speed
                        text-wrap:none { model.get().speed }
                    text #mechanic.caption "km/h"
                }
                for (lane, ()) in { model.get().engines.into_iter().map(|lane| (lane, ())) } {
                    (engine_readout(lane.clone()))
                }
            }
        }
    }
}

fn engine_readout(lane: EngineLane) -> Element {
    view! {
        col width:fill height:min-content gap:5px {
            text #mechanic.caption font-color:ink.muted (lane.title)
            row width:fill height:22px align:center justify:between {
                text font-size:16px font-weight:600 text-wrap:none (lane.rpm)
                text font-size:16px font-weight:600 font-color:accent.speed
                    text-wrap:none (lane.gear)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mechanic_core::{BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, SeatSpec};
    use mechanic_gpu::GpuTransform;
    use mosaic_core::Vector2;

    use super::*;
    use crate::ui::testing::{Overlay, VIEWPORT};

    fn seated_simulation() -> (AppSimulation, PartId) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
            .unwrap()
        else {
            panic!("spawn seat");
        };
        let creation = graph.compile().unwrap();
        let transforms = vec![
            GpuTransform {
                position: [0.0, 0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            };
            creation.compounds.len()
        ];
        (
            AppSimulation {
                published_graph: graph,
                creation: Some(creation),
                previous_transforms: transforms.clone(),
                transforms,
                previous_snapshot_tick: 10,
                snapshot_tick: 16,
                ..AppSimulation::default()
            },
            seat,
        )
    }

    #[test]
    fn speed_uses_vehicle_translation_and_simulation_time() {
        let (mut simulation, seat) = seated_simulation();
        // One metre in six 60-Hz ticks is 36 km/h, regardless of render FPS.
        simulation.transforms[0].position[0] = 1.0;
        assert!((vehicle_speed_kmh(&simulation, seat).unwrap() - 36.0).abs() < 0.001);
        simulation.transforms[0].position[0] = -1.0;
        assert!((vehicle_speed_kmh(&simulation, seat).unwrap() - 36.0).abs() < 0.001);
        simulation.transforms[0].position[0] = 0.0;
        simulation.transforms[0].rotation = bevy::prelude::Quat::from_rotation_y(0.5).to_array();
        assert_eq!(vehicle_speed_kmh(&simulation, seat), Some(0.0));
        simulation.snapshot_tick = simulation.previous_snapshot_tick;
        assert_eq!(vehicle_speed_kmh(&simulation, seat), None);
        simulation.snapshot_tick += 1;
        simulation.previous_transforms.clear();
        assert_eq!(vehicle_speed_kmh(&simulation, seat), None);
    }

    #[test]
    fn rpm_uses_the_active_ratio_without_multiplying_by_engine_count() {
        let config = GearboxConfig::for_depth(2, true);
        let speed = core::f32::consts::TAU * 10.0;
        let first = engine_lane(EngineKind::Gas, 2, Some(&config), Some(1), Some(speed));
        assert_eq!(first.title, "GAS ×2 · AUTO");
        assert_eq!(first.gear, "GEAR 1");
        assert_eq!(first.rpm, "1800 RPM");
        let second = engine_lane(EngineKind::Gas, 2, Some(&config), Some(2), Some(speed));
        assert_eq!(second.gear, "GEAR 2");
        assert_eq!(second.rpm, "600 RPM");
        let reverse = engine_lane(EngineKind::Gas, 2, Some(&config), Some(0), Some(-speed));
        assert_eq!(reverse.gear, "GEAR R1");
        assert_eq!(reverse.rpm, "2400 RPM");
    }

    #[test]
    fn neutral_and_missing_measurements_do_not_invent_rpm() {
        let config = GearboxConfig::direct();
        let neutral = engine_lane(EngineKind::Gas, 1, Some(&config), None, Some(10.0));
        assert_eq!(neutral.gear, "GEAR N");
        assert_eq!(neutral.rpm, "— RPM");
        let missing = engine_lane(EngineKind::Gas, 1, Some(&config), Some(0), None);
        assert_eq!(missing.rpm, "— RPM");
        let electric = engine_lane(EngineKind::Electric, 1, Some(&config), Some(0), Some(0.0));
        assert_eq!(electric.gear, "GEAR 1");
        assert_eq!(electric.rpm, "0 RPM");
    }

    #[test]
    fn leaving_the_seat_hides_the_instruments() {
        let (simulation, seat) = seated_simulation();
        let sequencer = DriveSequencer::default();
        let gearboxes = GearboxRuntime::default();
        assert!(capture(Some(seat), &simulation, &sequencer, &gearboxes).open);
        assert_eq!(
            capture(None, &simulation, &sequencer, &gearboxes),
            Model::default()
        );
        assert!(
            !capture(
                Some(seat),
                &AppSimulation::default(),
                &sequencer,
                &gearboxes
            )
            .open
        );
    }

    #[test]
    fn occupied_seat_selects_its_controller_and_combines_the_two_gas_engines() {
        use bevy::prelude::IVec3;
        use mechanic_core::{
            ControllerSpec, EngineSpec, FaceKind, FaceRef, GridRotation, SeatControllerLinkSpec,
            WeldSpec,
        };

        fn spawn(graph: &mut ConstructionGraph, command: BuildCommand) -> PartId {
            let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
                panic!("expected a part");
            };
            part
        }

        let (mut simulation, seat) = seated_simulation();
        let graph = &mut simulation.published_graph;
        let pose = |y| BuildPose::new(IVec3::new(0, y, 0), GridRotation::default());
        let controller = spawn(
            graph,
            BuildCommand::SpawnController(ControllerSpec::new(pose(40))),
        );
        for (y, first_face, second_face) in [
            (42, FaceKind::PositiveY, FaceKind::NegativeY),
            (38, FaceKind::NegativeY, FaceKind::PositiveY),
        ] {
            let engine = spawn(
                graph,
                BuildCommand::SpawnEngine(EngineSpec::new(EngineKind::Gas, pose(y))),
            );
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(controller, first_face),
                    second: FaceRef::part(engine, second_face),
                }))
                .unwrap();
        }
        graph
            .apply(BuildCommand::AddSeatControllerLink(
                SeatControllerLinkSpec { seat, controller },
            ))
            .unwrap();
        let other_seat = spawn(graph, BuildCommand::SpawnSeat(SeatSpec::new(pose(80))));
        simulation.creation = Some(graph.compile().unwrap());
        let sequencer = DriveSequencer::default();
        let mut gearboxes = GearboxRuntime::default();
        gearboxes.start(&simulation.published_graph, &sequencer);
        let occupied = capture(Some(seat), &simulation, &sequencer, &gearboxes);
        assert_eq!(occupied.engines.len(), 1);
        assert_eq!(occupied.engines[0].title, "GAS ×2 · AUTO");
        assert!(
            capture(Some(other_seat), &simulation, &sequencer, &gearboxes)
                .engines
                .is_empty()
        );
    }

    #[test]
    fn instruments_stay_bottom_right_and_do_not_capture_driving_input() {
        let overlay = Overlay::mount();
        let config = GearboxConfig::direct();
        overlay.handles.driving.set(Model {
            open: true,
            speed: "36.0".to_owned(),
            engines: vec![engine_lane(
                EngineKind::Gas,
                2,
                Some(&config),
                Some(0),
                Some(0.0),
            )],
        });
        overlay.settle();
        let panel = overlay
            .shapes()
            .into_iter()
            .find(|shape| (shape.rect.size.width - PANEL_WIDTH).abs() < 0.5)
            .expect("driving panel");
        assert!(
            (panel.rect.origin.x + panel.rect.size.width - (VIEWPORT.width - INSET)).abs() < 1.0
        );
        assert!(
            (panel.rect.origin.y + panel.rect.size.height - (VIEWPORT.height - INSET)).abs() < 1.0
        );
        assert!(
            !overlay.wants_pointer_at(Vector2::new(VIEWPORT.width - 30.0, VIEWPORT.height - 30.0))
        );
        assert!(overlay.labels().contains(&"km/h".to_owned()));
        assert!(overlay.labels().contains(&"GEAR 1".to_owned()));
        overlay.handles.driving.set(Model::default());
        overlay.settle();
        assert!(!overlay.labels().contains(&"km/h".to_owned()));
    }
}
