use mosaic_core::{Color, Size, State};
use mosaic_layout::{Align, AxisPair, Edges, Grid, GridTrack, GridTracks, Justify, Length, Style};
use mosaic_macros::view;
use mosaic_widgets::{Element, StyleCtx, StyleSet, Ui, Visual};

mosaic_macros::style! {
    #compact-grid radius:6px fill:#556677
}

#[test]
fn canonical_grid_syntax_builds() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let grid: Element = view! {
        grid cols:(auto 1fr minmax(120px 2fr)) rows:(repeat(2 auto 1fr)) backfill gap:12px
            content-align:(x:between y:center) items-align:(x:center y:start) {
            el col-span:2 self-align:(x:end y:center) {}
            el grid-col:2 row-span:2 {}
        }
    };
    drop(grid);

    let responsive: Element = view! {
        grid cols:(repeat(auto-fit minmax(240px 1fr))) {}
    };
    drop(responsive);

    let shown = State::new(true);
    let rows = State::new(vec![(1u32, 1u32), (2, 2)]);
    let reactive: Element = view! {
        grid cols:if $shown { 2 } else { 3 } gap:if $shown { 4px } else { 12px }
            pad:if $shown {
                (horizontal:if $shown { 8px } else { 10px } vertical:4px)
            } else {
                (horizontal:16px vertical:12px)
            }
            if $shown {
                #compact-grid
                radius:4px
                fill:#112233
            } else {
                radius:8px
                fill:#334455
            }
            backfill:{ shown.get() } {
            el {}
            if $shown {
                el {}
                el {}
            } else if $rows.len() == 1 {
                el {}
            } else {
                el {}
                el {}
                el {}
            }
            for (_key, _value) in { rows.get() } {
                el grid-col:1 {}
                el grid-row:1 {}
            }
        }
    };
    ui.root().adopt(&reactive);
    let _ = ui.frame(Size::new(400.0, 300.0), 1.0);
    shown.set(false);
    rows.set(vec![(2, 2), (3, 3)]);
    let _ = ui.frame(Size::new(400.0, 300.0), 1.0);
}
