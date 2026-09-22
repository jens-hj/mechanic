use mosaic_widgets::Element;
use mosaic_macros::view;
use mosaic_layout::Style;
use mosaic_widgets::{ReorderOptions, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();
    parent.adopt(&view! {
        row reorderable:(controlled) {}
    });
}
