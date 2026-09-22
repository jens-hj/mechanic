use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_text::TextStyle;
use mosaic_widgets::{ComponentHandle, Element, ElementPatch, Fx, Transition, Ui, text};

#[component]
fn Badge(#[prop(optional)] item_count: usize) -> Element {
    let _ = item_count;
    Element::orphan(Style::default())
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        row min_width:10px translate_children:(mosaic_core::Vector2::ZERO)
            enter_exit:(Fx::default()) hover { max_width:20px } {
            text font_size:14 "legacy"
            Badge item_count:2
        }
    });
}
