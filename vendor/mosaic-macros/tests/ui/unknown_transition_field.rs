use mosaic_widgets::Element;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::Ui;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        row transition:(width: ease(240.0) glow: 1.0) {}
    });
}
