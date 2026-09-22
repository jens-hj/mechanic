use mosaic_layout::*;
use mosaic_macros::component;
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_widgets::{Element, ElementPatch, Fx, Transition};

/// A component with documented props.
#[component]
fn Card(
    /// The heading displayed at the top.
    title: String,
    /// Optional supporting text.
    #[prop(optional)]
    subtitle: String,
) -> Element {
    let _ = (title, subtitle);
    Element::orphan(Style::default())
}

fn main() {}
