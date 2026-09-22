use mosaic_macros::{scheme, theme, view};
use mosaic_core::{Theme, ThemeToken, resolve_theme_value};
use mosaic_layout::Style;
use mosaic_render::{
    BackdropFilter, BoxPoint, DirectionalLight, FilterChain, FilterLength, LightSpec, PointLight,
    Refraction,
};
use mosaic_widgets::{Element, Ui, Visual};

scheme! {
    Bases {
        general-filter: Filter,
        point: PointLight,
    }
}

fn main() {
    let values = theme! {
        Bases {
            general-filter: blur(8px),
            point: (at:center intensity:80%),
        }
    };
    mosaic_core::theme::install(&values);

    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! {
        col {
            row filter:refract(general-filter bezel:12px) {}
            row light:(point angle:45deg) {}
        }
    });
}
