use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, ImageSource, ImgStyle, Ui, img};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! {
        el {
            img "tests/assets/definitely-missing.png"
        }
    });
}
