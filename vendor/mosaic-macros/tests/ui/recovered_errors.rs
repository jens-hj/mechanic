use mosaic_widgets::Element;
use mosaic_layout::{Length, Style};
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::{Ui, text};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        col gap:8px {
            text font-size:14px "before a broken sibling"
            row wat:1 {
                text siize:2 "child of a broken row still expands"
            }
            button "no click"
        }
    });
}
