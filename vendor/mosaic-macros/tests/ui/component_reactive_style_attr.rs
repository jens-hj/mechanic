// Reactive component syntax constructs a `Derived<T>` and therefore remains
// invalid for root-patching style attributes, whose setters take plain values.
use mosaic_core::Derived;
use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_widgets::{ComponentHandle, Element, ElementPatch, Fx, Transition, Ui};

#[component]
fn Card() -> Element {
    Element::orphan(Style::column())
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        Card fill:{ PaintSpec::default() }
    });
}
