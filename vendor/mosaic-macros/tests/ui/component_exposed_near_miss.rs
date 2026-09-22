// An exposed state named a near-miss of an interaction state would be
// reported as a typo at every call site — rejected at the definition instead.
use mosaic_macros::component;

struct Element;
struct State<T>(T);

#[component]
fn Foo() -> Element {
    #[exposed]
    let hovered: State<bool> = State(false);
    let _ = hovered;
    loop {}
}

fn main() {}
