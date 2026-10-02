//! Read-only developer navigation status, independent of the help panel.
#![allow(clippy::wildcard_imports, reason = "Mosaic authoring vocabulary")]

use super::components::{PanelSurface, PanelSurfaceProps};
use super::styles::*;
use super::theme::*;
use crate::controls::{Controls, GameAction};
use crate::dev_tools::{DevMode, DevTools};
use bevy::prelude::{NonSendMut, Res};
use bevy_mosaic::ui::*;
use mosaic_core::Rect;
use mosaic_macros::{component, view};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Model {
    pub(crate) enabled: bool,
    pub(crate) spectator: bool,
    title: String,
    modes: String,
    speed: String,
    erosion: String,
    heatmap: String,
    heatmap_visible: bool,
    history: String,
    notice: String,
    time: String,
}

fn capture(dev: &DevTools, controls: &Controls) -> Model {
    if !dev.enabled {
        return Model::default();
    }
    let mode_label = match dev.mode {
        DevMode::Normal => "Normal",
        DevMode::Noclip => "Noclip · returns to start",
        DevMode::Spectator => "Detached · live simulation",
    };
    Model {
        enabled: true,
        spectator: dev.spectator(),
        title: format!("DEV TOOLS · {mode_label} · {}×", dev.speed),
        modes: format!(
            "{} Noclip   {} Detached",
            controls.label(GameAction::DevNoclip),
            controls.label(GameAction::DevSpectator)
        ),
        speed: format!(
            "{} Slower   {} Faster   {} Reset",
            controls.label(GameAction::DevSpeedDecrease),
            controls.label(GameAction::DevSpeedIncrease),
            controls.label(GameAction::DevSpeedReset)
        ),
        erosion: format!(
            "Ground {}× · erosion and grass · {} cycle",
            dev.erosion,
            controls.label(GameAction::DevErosion)
        ),
        heatmap: format!(
            "Heatmap {} · {} cycle · {} reset",
            dev.erosion_map.label(),
            controls.label(GameAction::DevErosionMap),
            controls.label(GameAction::DevErosionReset)
        ),
        heatmap_visible: dev.erosion_map != crate::dev_tools::ErosionMap::Off,
        history: String::new(),
        notice: dev.notice.to_owned(),
        time: String::new(),
    }
}

pub(crate) fn push(
    ui: Option<NonSendMut<super::AppUi>>,
    dev: Res<DevTools>,
    settings: Res<crate::settings::AppSettings>,
    runtime: Res<crate::world::WorldRuntime>,
    sky: Option<Res<crate::sky::SkyState>>,
) {
    let Some(ui) = ui else {
        return;
    };
    let mut next = capture(&dev, settings.controls());
    if dev.enabled {
        let recorded = runtime
            .sediment_diagnostics(dev.erosion_generation)
            .map_or(0.0, |snapshot| snapshot.seconds);
        next.history = format!("Recorded {recorded:.0} s · water hidden · 102.4 m map");
        let seconds = sky.as_ref().map_or_else(
            || runtime.time_of_day_seconds(),
            |sky| sky.displayed_seconds(&runtime),
        );
        let minutes = (seconds / 60.0).floor();
        next.time = format!(
            "{:02.0}:{:02.0} · Cycle {} · {} / {} · {} pause",
            (minutes / 60.0).floor(),
            minutes % 60.0,
            sky.as_ref().map_or("paused", |sky| sky.status()),
            settings.controls().label(GameAction::DevTimeEarlier),
            settings.controls().label(GameAction::DevTimeLater),
            settings.controls().label(GameAction::DevTimePause)
        );
    }
    if ui.handles.dev_tools.get_untracked() != next {
        ui.handles.dev_tools.set(next);
    }
}

#[component]
pub(crate) fn DevOverlay(model: State<Model>, viewport: State<Size>) -> Element {
    let size = State::new(Size::ZERO);
    view! {
        col width:min-content height:min-content nohit
            translate:(x:16px y:{ Length::px((viewport.get().height - size.get().height - 120.0).max(16.0)) })
            @layout:{ move |bounds: Rect| {
                if size.get_untracked() != bounds.size { size.set(bounds.size); }
            } } {
            PanelSurface elevated:false width:390px height:min-content pad:12px gap:5px {
                text #mechanic.caption font-color:accent.speed { model.get().title }
                text #mechanic.caption { model.get().modes }
                text #mechanic.caption { model.get().speed }
                text #mechanic.caption { model.get().time }
                text #mechanic.caption { model.get().erosion }
                text #mechanic.caption { model.get().heatmap }
                if model.with(|model| model.heatmap_visible) {
                    text #mechanic.caption { "Orange removed · Blue deposited · Purple both" }
                    text #mechanic.caption { "Yellow hatching: pending sediment (current)" }
                    text #mechanic.caption { "Log scale: 1 / 10 / 100+ mm equivalent depth" }
                    text #mechanic.caption { model.get().history }
                }
                if model.with(|model| !model.notice.is_empty()) {
                    text #mechanic.caption { model.get().notice }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::testing::{Overlay, VIEWPORT};
    use mosaic_core::Vector2;

    #[test]
    fn dev_status_is_hidden_in_world_selection_and_returns_in_world() {
        let overlay = Overlay::mount();
        let mut dev = DevTools::default();
        dev.enabled = true;
        overlay
            .handles
            .dev_tools
            .set(capture(&dev, &Controls::default()));
        for world_selection_open in [true, false, true, false] {
            overlay
                .handles
                .worlds
                .update(|model| model.open = world_selection_open);
            overlay.settle();
            assert_eq!(
                overlay
                    .labels()
                    .iter()
                    .any(|label| label.contains("DEV TOOLS")),
                !world_selection_open,
            );
        }
    }

    #[test]
    fn dev_status_uses_rebound_labels_and_is_pointer_transparent_without_help() {
        let overlay = Overlay::mount();
        let hidden = overlay.element_count();
        let mut dev = DevTools::default();
        dev.enabled = true;
        dev.mode = DevMode::Normal;
        dev.speed = 8.0;
        let mut controls = Controls::default();
        controls.set(
            GameAction::DevNoclip,
            0,
            Some(crate::controls::InputChord::key(
                bevy::prelude::KeyCode::F10,
            )),
        );
        let model = capture(&dev, &controls);
        assert!(model.title.contains("DEV TOOLS"));
        assert!(model.title.contains("Normal"));
        assert!(model.title.contains("8×"));
        assert!(model.modes.contains("F10"));
        overlay.handles.dev_tools.set(model);
        overlay.settle();
        assert!(overlay.element_count() > hidden);
        assert!(!overlay.handles.help_open.get_untracked());
        assert!(!overlay.wants_pointer_at(Vector2::new(100.0, VIEWPORT.height - 140.0)));
        dev.enabled = false;
        overlay.handles.dev_tools.set(capture(&dev, &controls));
        overlay.settle();
        assert_eq!(overlay.element_count(), hidden);
    }
    #[test]
    fn heatmap_legend_and_controls_follow_the_selected_mode_and_bindings() {
        let overlay = Overlay::mount();
        let mut dev = DevTools::default();
        dev.enabled = true;
        dev.erosion_map = crate::dev_tools::ErosionMap::Recent;
        let mut controls = Controls::default();
        controls.set(
            GameAction::DevErosionMap,
            0,
            Some(crate::controls::InputChord::key(
                bevy::prelude::KeyCode::KeyH,
            )),
        );
        overlay.handles.dev_tools.set(capture(&dev, &controls));
        overlay.settle();
        let labels = overlay.labels();
        assert!(
            labels
                .iter()
                .any(|label| label.contains("Recent") && label.contains("H cycle"))
        );
        assert!(labels.iter().any(|label| label.contains("Yellow hatching")));
        dev.erosion_map = crate::dev_tools::ErosionMap::Off;
        overlay.handles.dev_tools.set(capture(&dev, &controls));
        overlay.settle();
        assert!(
            !overlay
                .labels()
                .iter()
                .any(|label| label.contains("Yellow hatching"))
        );
    }
}
