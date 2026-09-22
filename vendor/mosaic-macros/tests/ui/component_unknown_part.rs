// A typo'd part name at a component call site is caught by method resolution
// on the generated handle — "no method `knbo` on `LampHandle`".
use mosaic_core::Color;
use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_widgets::{ComponentHandle, Element, ElementPatch, Fx, Transition, Ui};

#[component]
fn Lamp() -> Element {
    view! {
        col {
            el width:10px {} as pub knob
        }
    }
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();
    parent.adopt(&view! {
        col {
            Lamp {
                knbo fill:(Color::WHITE)
            }
        }
    });
}
