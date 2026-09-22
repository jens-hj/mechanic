// `#[exposed]` must sit on a top-level `let` of the body: the handle is built
// from the body's tail, where nested bindings are out of scope.
use mosaic_macros::component;

struct Element;

#[component]
fn Foo() -> Element {
    {
        #[exposed]
        let hidden: bool = false;
        let _ = hidden;
    }
    loop {}
}

fn main() {}
