use mosaic_layout::*;
use mosaic_macros::component;
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_widgets::{Element, ElementPatch, Fx, Transition, Ui};

#[component]
fn Control(
    size: f32,
    #[prop(default = size * 0.25)] radius: f32,
    #[prop(default = || {})] action: impl Fn() + 'static,
    #[prop(optional)] quiet: bool,
) -> Element {
    action();
    let _ = (size, radius, quiet);
    Element::orphan(Style::default())
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    Control(ControlProps::builder().size(40.0).build());
    Control(
        ControlProps::builder()
            .size(0.0)
            .radius(0.0)
            .action(|| {})
            .quiet(true)
            .build(),
    );
}
