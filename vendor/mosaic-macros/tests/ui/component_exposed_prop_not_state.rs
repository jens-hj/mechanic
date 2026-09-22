// `#[prop(exposed)]` requires a literal `State<T>`/`Derived<T>` type — the
// handle accessor is `ReadState<T>`, so the macro must see `T` in the source.
use mosaic_macros::component;

struct Element;

#[component]
fn Foo(#[prop(exposed)] count: i32) -> Element {
    let _ = count;
    loop {}
}

fn main() {}
