#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

//! A Mosaic panel composited over a live Bevy scene.
//!
//! What this is here to prove, in one screen: the overlay blends over 3D
//! geometry instead of erasing it, the pointer reaches the right element, a
//! text field takes focus and keystrokes, and the tree repaints from reactive
//! state rather than from anything Bevy pushes at it.
//!
//! Run it with `cargo run -p bevy_mosaic --example overlay`.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_mosaic::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MosaicPlugin))
        .add_systems(Startup, (setup_scene, setup_overlay))
        .add_systems(Update, (spin, report_arbitration, screenshot_once))
        .run();
}

/// A cube worth occluding, so it is obvious whether the overlay composited or
/// clobbered.
#[derive(Component)]
struct Spin;

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        // Matches mechanic's own cameras, and sidesteps the tonemapping LUTs
        // this slim feature set does not carry.
        Tonemapping::None,
        Transform::from_xyz(0.0, 1.8, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        // Without this the overlay has no view to paint into.
        MosaicCamera,
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.6, 1.6, 1.6))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.72, 0.85),
            ..default()
        })),
        Spin,
    ));
}

fn spin(time: Res<Time>, mut cubes: Query<&mut Transform, With<Spin>>) {
    for mut transform in &mut cubes {
        transform.rotate_y(time.delta_secs() * 0.8);
        transform.rotate_x(time.delta_secs() * 0.3);
    }
}

/// Build the tree once. Everything after this is reactive state changing.
fn setup_overlay(mosaic: NonSend<MosaicContext>) {
    let ui = mosaic.ui();
    let view = panel::build(ui);
    ui.mount(&view);
}

/// The UI lives in its own module so it can glob Mosaic's authoring vocabulary
/// without fighting Bevy's prelude — both define `State`, `Children` and
/// `Interaction`. This is the pattern to copy.
mod panel {
    #[allow(clippy::wildcard_imports)] // The authoring vocabulary is meant to be globbed.
    use bevy_mosaic::ui::*;

    pub fn build(ui: &Ui) -> Element {
        let count: State<i64> = State::new(0);
        let name: State<String> = State::new("front left".to_string());

        let _ambient = ui.enter();
        view! {
            col pad:28px gap:16px align:start justify:start {
                col width:320px height:min-content pad:20px gap:14px
                    fill:mocha.mantle radius:12px {
                    text font-size:18px font-color:mocha.text "Mosaic in Bevy"
                    text font-size:13px font-color:mocha.subtext0
                        "the cube keeps spinning behind this panel"

                    row gap:10px align:center height:min-content {
                        button @click:{ $count -= 1 } "-1"
                        text font-size:20px font-color:mocha.blue { $count.to_string() }
                        button @click:{ $count += 1 } "+1"
                    }

                    col gap:6px {
                        text font-size:12px font-color:mocha.subtext0 "joint name"
                        input placeholder:"name this joint" name
                    }

                    text font-size:12px font-color:mocha.subtext0 { format!("hello, {}", $name) }
                }
            }
        }
    }
}

/// Capture one frame to a file when `OVERLAY_SCREENSHOT` names a path, then
/// quit. How this example proves it composited rather than merely ran — a
/// window that never crashed still tells you nothing about what is in it.
fn screenshot_once(
    mut commands: Commands,
    frames: Res<bevy::diagnostic::FrameCount>,
    mut done: Local<bool>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(path) = std::env::var("OVERLAY_SCREENSHOT") else {
        return;
    };
    // A few frames in, so the first-frame surface dance is over.
    if *done || frames.0 < 30 {
        if *done && frames.0 > 90 {
            exit.write(AppExit::Success);
        }
        return;
    }
    *done = true;
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}

/// The contract an app gates its own input on: while the pointer is over the
/// overlay, or a field has the keyboard, the world should not also react.
fn report_arbitration(mosaic: NonSend<MosaicContext>, mut was: Local<(bool, bool)>) {
    let now = (mosaic.wants_pointer(), mosaic.wants_keyboard());
    if now != *was {
        info!("wants_pointer={} wants_keyboard={}", now.0, now.1);
        *was = now;
    }
}
