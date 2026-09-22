use mosaic_layout::{Edges, Length, Style};
use mosaic_macros::{style, view};
use mosaic_widgets::{Element, StyleCtx, StyleSet, Ui, Visual};

style! {
    #card radius:3px
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! { el pad:8px #card {} });
}
