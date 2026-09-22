use mosaic_widgets::Element;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::Ui;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        row {
            // A directional light comes from infinity; placing it with `at:`
            // contradicts `angle:`.
            row light:(angle: 45deg at: (x:30% y:20%)) {}
            // `elevation:` alone has no heading to rake along.
            row light:(elevation: 30deg) {}
            // Parallel rays have nothing to fall off over.
            row light:(angle: 45deg reach: (x:40px y:40px)) {}
        }
    });
}
