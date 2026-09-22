use mosaic_layout::{Grid, GridTrack, GridTracks, Length, Style};
use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let _view: Element = view! {
        grid cols:(auto 1fr minmax(120px 2fr)) rows:(repeat(2 auto 1fr)) backfill gap:12px {
            el col-span:2 {}
            el grid-col:2 row-span:2 {}
        }
    };
    let _responsive: Element = view! {
        grid cols:(repeat(auto-fit minmax(240px 1fr))) {}
    };
}
