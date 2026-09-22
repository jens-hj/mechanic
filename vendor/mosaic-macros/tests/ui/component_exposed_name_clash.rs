// Parts, exposed props, and exposed states share the generated handle's
// accessor namespace — one name cannot be exposed twice.
use mosaic_macros::component;

struct Element;

#[component]
fn Foo(#[prop(exposed)] spot: State<bool>) -> Element {
    let _ = spot;
    mosaic_macros::view! {
        col {
            el width:10px {} as pub spot
        }
    }
}

fn main() {}
