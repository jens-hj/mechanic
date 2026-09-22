use mosaic_core::mocha;
use mosaic_layout::*;
use mosaic_macros::view;
use mosaic_render::{StrokeEdges, StrokeSpec};
use mosaic_text::TextStyle;
use mosaic_widgets::*;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();

    parent.adopt(&view! {
        stack {
            col searchable {
                text "prose to search"
            }
            row justify:end align:start pad:12px {
                find
            }
        }
    });

    parent.adopt(&view! {
        row {
            find {
                bar fill:mocha.crust radius:14px
                field width:240px
                counter font-color:mocha.overlay1
                prev radius:4px
                next radius:4px
                close radius:4px hover { fill:mocha.red }
            }
            empty {
                bar stroke:(width:1.0 color:mocha.red)
            }
            open {
                bar stroke:(width:1.0 color:mocha.lavender)
            } as found
            // The `as` binding is the widget's handle, not its root element.
            { let _: &mosaic_widgets::FindBar = &found; }
        }
    });

    let _: Element = parent.clone();
}
