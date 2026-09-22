use mosaic_core::{Derived, State};
use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_widgets::{ComponentHandle, Element, ElementPatch, Fx, Transition, Ui};

#[component]
fn Control(value: Derived<usize>, change: impl Fn(usize) + 'static) -> Element {
    change(value.get());
    Element::orphan(Style::default())
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();
    let value = State::new(1usize);
    let selected = Derived::new(move || value.get());

    parent.adopt(&view! {
        Control value:{ value.get() + 1 } change:(move |next| value.set(next))
    });
    parent.adopt(&view! {
        Control value:(selected) change:(|_| {})
    });
}
