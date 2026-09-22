use mosaic_core::State;
use mosaic_layout::Style;
use mosaic_macros::{style, view};
use mosaic_widgets::{Element, StyleCtx, StyleSet, Ui, Visual};

style! {
    #interactive focusable
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let enabled = State::new(true);
    let _view: Element = view! {
        el if $enabled { #interactive } {}
    };
}
