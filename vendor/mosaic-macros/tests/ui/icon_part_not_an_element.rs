// An icon part addresses a shape in the SVG, so it takes paints and nothing
// that only an element could have.
use mosaic_core::theme::SvgToken;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, IconPaints, IconStroke, IconStyle, Ui, icon};

const MARK: SvgToken = SvgToken::new(0x1D0, 0);

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el {
            icon MARK {
                blade-a opacity:0.5
            }
        }
    });
}
