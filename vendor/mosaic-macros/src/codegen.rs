//! Lowering the parsed `view!` tree to builder-API calls.
//!
//! Every node becomes exactly the statements a caller would write by hand;
//! errors surface as inline `compile_error!`s so the rest of the tree — and
//! the IDE's view of it — keeps expanding.

use mosaic_syntax::vocabulary::{self, FLAGS, FONT_ATTRS, HANDLERS, SEMANTIC_ATTRS};
use mosaic_syntax::{
    AsBinding, Attr, ButtonContent, ButtonNode, CallNode, ComponentNode, CondAlternative, CondNode,
    ConditionalRun, ConditionalRunAlternative, Container, Content, ForKind, ForNode, Handler,
    IconNode, IconPartRun, ImgNode, InputNode, InteractionState, LetKind, LetNode, Node, PartRun,
    ScrollNode, ShapeNode, ShapeTag, StateBlock, StateWidgetNode, StyleUse, Tag, TextNode,
    ThemeClause, TooltipAlign, TooltipCollision, TooltipNode, TooltipSide, TooltipTrigger, Value,
    View, WidgetTag, did_you_mean,
};
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, quote_spanned};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use syn::spanned::Spanned;
use syn::{Expr, Result};

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// The internal name for an element built without an explicit `as`. Macro
/// hygiene keeps it from colliding with anything the caller wrote.
fn node_ident() -> Ident {
    Ident::new("__view_node", Span::mixed_site())
}

/// The name a child block sees its enclosing element under. Call-site hygiene
/// so `{ … }` escape blocks can reference it.
fn parent_ident() -> Ident {
    Ident::new("parent", Span::call_site())
}

thread_local! {
    /// Addresses of the `AsBinding`s selected by the syntax layer as stable
    /// for the invocation currently being lowered.
    static STABLE_BINDINGS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    static IMMEDIATE_DEPS: RefCell<HashMap<usize, Vec<String>>> = RefCell::new(HashMap::new());
}

fn is_stable_binding(binding: &AsBinding) -> bool {
    let key = binding as *const AsBinding as usize;
    STABLE_BINDINGS.with(|bindings| bindings.borrow().contains(&key))
}

fn core_crate_path() -> TokenStream2 {
    if let Ok(found) = crate_name("mosaic") {
        return match found {
            FoundCrate::Itself => quote!(crate::core),
            FoundCrate::Name(name) => {
                let name = Ident::new(&name, Span::call_site());
                quote!(::#name::core)
            }
        };
    }
    match crate_name("mosaic-core") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let name = Ident::new(&name, Span::call_site());
            quote!(::#name)
        }
        Err(_) => quote!(::mosaic_core),
    }
}

fn widgets_crate_path() -> TokenStream2 {
    if let Ok(found) = crate_name("mosaic") {
        return match found {
            FoundCrate::Itself => quote!(crate::widgets),
            FoundCrate::Name(name) => {
                let name = Ident::new(&name, Span::call_site());
                quote!(::#name::widgets)
            }
        };
    }
    match crate_name("mosaic-widgets") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let name = Ident::new(&name, Span::call_site());
            quote!(::#name)
        }
        Err(_) => quote!(::mosaic_widgets),
    }
}

pub fn expand(view: &View, source: &str) -> TokenStream2 {
    let errors = view.errors.iter().map(syn::Error::to_compile_error);
    let stable_nodes = view.stable_bound_nodes();
    let stable = stable_nodes
        .iter()
        .map(|(binding, _)| *binding)
        .collect::<Vec<_>>();
    STABLE_BINDINGS.with(|bindings| {
        bindings.borrow_mut().extend(
            stable
                .iter()
                .map(|binding| *binding as *const AsBinding as usize),
        );
    });
    IMMEDIATE_DEPS.with(|dependencies| {
        dependencies.borrow_mut().extend(
            view.stable_immediate_dependencies()
                .into_iter()
                .map(|(binding, names)| (binding as *const AsBinding as usize, names)),
        );
    });
    // `view!` builds its single top-level node as a detached orphan and returns
    // it: `view! { col { … } }` is an expression the caller mounts
    // (`ui.root().adopt(&…)`) or returns from a `fn(..) -> Element`.
    let main = emit_orphan_root(&view.body);
    STABLE_BINDINGS.with(|bindings| bindings.borrow_mut().clear());
    IMMEDIATE_DEPS.with(|dependencies| dependencies.borrow_mut().clear());
    // Keep recovered diagnostics and the returned element inside one block so
    // the expansion remains a valid expression even when errors were found.
    let compiled = if stable.is_empty() {
        quote!({ #(#errors;)* #main })
    } else {
        let core = core_crate_path();
        let widgets = widgets_crate_path();
        let slots = stable_nodes.iter().map(|(binding, node)| {
            let name = &binding.name;
            let ty = stable_binding_type(node, &widgets);
            quote!(let #name = #core::ViewBinding::<#ty>::new();)
        });
        let owners = stable
            .iter()
            .map(|binding| {
                let name = &binding.name;
                quote!(__view_result.own_view_binding(#name);)
            })
            .collect::<Vec<_>>();
        quote!({
            #(#errors;)*
            let __view_build_guard = #core::ViewBuildGuard::new();
            #(#slots)*
            let __view_result = #main;
            #(#owners)*
            ::core::mem::drop(__view_build_guard);
            __view_result
        })
    };
    let widgets = widgets_crate_path();
    quote!({
        let __view_hot_result = #compiled;
        #[allow(unexpected_cfgs)]
        {
            #[cfg(mosaic_adapter_hot_image)]
            #widgets::__register_hot_template(
                &__view_hot_result,
                file!(),
                line!(),
                column!(),
                env!("CARGO_MANIFEST_DIR"),
                #source,
            );
        }
        __view_hot_result
    })
}

fn stable_binding_type(node: &Node, widgets: &TokenStream2) -> TokenStream2 {
    match node {
        Node::Container(_)
        | Node::Text(_)
        | Node::Button(_)
        | Node::Input(_)
        | Node::Img(_)
        | Node::Icon(_)
        | Node::Shape(_) => quote!(#widgets::Element),
        Node::Scroll(_) => quote!(#widgets::Scroll),
        Node::StateWidget(widget) => {
            let ty = Ident::new(
                match widget.tag {
                    WidgetTag::Toggle => "Toggle",
                    WidgetTag::Checkbox => "Checkbox",
                    WidgetTag::Radio => "Radio",
                    WidgetTag::Slider => "Slider",
                    WidgetTag::Progress => "Progress",
                    WidgetTag::Stepper => "Stepper",
                    WidgetTag::Select => "Select",
                    WidgetTag::Find => "FindBar",
                },
                widget.span,
            );
            quote!(#widgets::#ty)
        }
        Node::Component(component) => {
            let ty = Ident::new(&format!("{}Handle", component.func), component.func.span());
            quote!(#ty)
        }
        // Ordinary call nodes are detached element-producing helpers. Typed
        // component handles use the PascalCase component-node path above.
        Node::Call(_) => quote!(#widgets::Element),
        Node::Tooltip(_)
        | Node::If(_)
        | Node::For(_)
        | Node::Children(_)
        | Node::Let(_)
        | Node::Escape(_) => {
            unreachable!("dynamic nodes have no stable handle")
        }
    }
}

/// The handle identifier an element-producing node binds (its `as` name, else
/// the hygienic default). `None` for nodes that build no single handle
/// (`if`/`for`/`children`/escape).
fn node_handle(node: &Node) -> Option<Ident> {
    let name = match node {
        Node::Container(c) => &c.name,
        Node::Scroll(s) => &s.name,
        Node::Text(t) => &t.name,
        Node::Button(b) => &b.name,
        Node::Input(i) => &i.name,
        Node::Img(i) => &i.name,
        Node::Icon(i) => &i.name,
        Node::Shape(shape) => &shape.name,
        Node::StateWidget(w) => &w.name,
        Node::Component(c) => &c.name,
        Node::Call(c) => &c.name,
        Node::Tooltip(_)
        | Node::If(_)
        | Node::For(_)
        | Node::Children(_)
        | Node::Let(_)
        | Node::Escape(_) => {
            return None;
        }
    };
    Some(binding_ident(name))
}

fn node_binding(node: &Node) -> Option<&AsBinding> {
    match node {
        Node::Container(node) => node.name.as_ref(),
        Node::Scroll(node) => node.name.as_ref(),
        Node::Text(node) => node.name.as_ref(),
        Node::Button(node) => node.name.as_ref(),
        Node::Input(node) => node.name.as_ref(),
        Node::Img(node) => node.name.as_ref(),
        Node::Icon(node) => node.name.as_ref(),
        Node::Shape(node) => node.name.as_ref(),
        Node::StateWidget(node) => node.name.as_ref(),
        Node::Component(node) => node.name.as_ref(),
        Node::Call(node) => node.name.as_ref(),
        Node::Tooltip(_)
        | Node::If(_)
        | Node::For(_)
        | Node::Children(_)
        | Node::Let(_)
        | Node::Escape(_) => None,
    }
}

/// The element ident a node's `as` binding provides, else the hygienic
/// default.
fn binding_ident(name: &Option<AsBinding>) -> Ident {
    name.as_ref()
        .map(|binding| {
            if is_stable_binding(binding) {
                Ident::new(
                    &format!("__view_bound_{}", binding.name),
                    Span::mixed_site(),
                )
            } else {
                binding.name.clone()
            }
        })
        .unwrap_or_else(node_ident)
}

/// Populates a stable binding after its concrete handle has been created.
fn fill_binding(name: &Option<AsBinding>, value: &Ident) -> Option<TokenStream2> {
    let binding = name.as_ref().filter(|binding| is_stable_binding(binding))?;
    let slot = &binding.name;
    Some(quote!(#slot.set(#value.clone());))
}

fn fill_call_binding(name: &Option<AsBinding>, value: &Ident) -> Option<TokenStream2> {
    let binding = name.as_ref().filter(|binding| is_stable_binding(binding))?;
    let slot = &binding.name;
    Some(quote!(#slot.set(ComponentHandle::root(&#value).clone());))
}

/// The well-known alias orphan-root lowering binds the view's root element
/// to, referenced by `expose_part` registrations anywhere in the body. Like
/// [`parent_ident`], a `call_site` ident so it resolves across the emitters'
/// separate `quote!` invocations.
fn view_root_ident() -> Ident {
    Ident::new("__view_root", Span::call_site())
}

/// Inspection metadata every `as` binding asks for, followed by the
/// `expose_part` registration for `as pub`. Both go through
/// `ComponentHandle::root`, so call nodes yielding handles work too.
fn expose_stmt(name: &Option<AsBinding>, elem: &Ident) -> Option<TokenStream2> {
    let binding = name.as_ref()?;
    let part = binding.name.to_string();
    let root = view_root_ident();
    let expose = binding
        .is_exposed()
        .then(|| quote!(#root.expose_part(#part, ComponentHandle::root(&#elem));));
    Some(quote! {
        ComponentHandle::root(&#elem).inspection_name(#part);
        #expose
    })
}

/// The statement a `with theme` clause emits: the built element scopes the
/// source theme over itself and the counted neighborhood. `target` is an
/// expression evaluating to something with `scoped_theme` (an `Element`, or
/// a handle's root).
fn theme_scope_stmt(theme: &Option<ThemeClause>, target: TokenStream2) -> Option<TokenStream2> {
    let clause = theme.as_ref()?;
    let expr = &clause.expr;
    let ancestors = clause.ancestors.unwrap_or(0);
    let children = clause.children.unwrap_or(0);
    Some(quote!(#target.scoped_theme(move || #expr, #ancestors, #children);))
}

/// The source span a node anchors to, for diagnostics.
fn node_span(node: &Node) -> Span {
    match node {
        Node::Container(c) => c.span,
        Node::Scroll(s) => s.span,
        Node::Text(t) => t.span,
        Node::Button(b) => b.span,
        Node::Input(i) => i.span,
        Node::Img(i) => i.span,
        Node::Icon(i) => i.span,
        Node::Shape(shape) => shape.span,
        Node::StateWidget(w) => w.span,
        Node::Tooltip(t) => t.span,
        Node::Component(c) => c.func.span(),
        Node::Call(c) => c.span,
        Node::If(c) => c.span,
        Node::For(f) => f.span,
        Node::Let(l) => l.span,
        Node::Children(tag) => tag.span(),
        Node::Escape(block) => block.span(),
    }
}

/// A `{ let parent = <ref>; <children> }` block.
fn emit_block(parent_ref: TokenStream2, nodes: &[Node]) -> TokenStream2 {
    let parent = parent_ident();
    // `show`/`keyed` take over *all* of an element's children, so an `if`/`for`
    // can only bind the parent when it is the block's single node. Beside any
    // sibling it binds its own wrapper child instead — one node among the rest.
    // Bindings are not siblings: they build nothing to take over.
    let owns_parent = nodes.iter().filter(|node| node.is_structural()).count() == 1;
    let stable_prefix = nodes
        .iter()
        .position(|node| node_binding(node).is_none_or(|binding| !is_stable_binding(binding)))
        .unwrap_or(nodes.len());
    // A `let` is deliberately absent from this set: reordering siblings around
    // one could move a use above its binding, so a block that contains one
    // simply keeps its source order.
    let schedulable = nodes[stable_prefix..]
        .iter()
        .all(|node| matches!(node, Node::Escape(_)));
    let order = schedulable
        .then(|| scheduled_sibling_order(&nodes[..stable_prefix]))
        .flatten()
        .map(|mut order| {
            order.extend(stable_prefix..nodes.len());
            order
        })
        .unwrap_or_else(|| (0..nodes.len()).collect());
    let reordered = order.iter().copied().ne(0..nodes.len());
    let stmts = order
        .iter()
        .map(|&index| emit_node(&nodes[index], owns_parent));
    let restore = reordered.then(|| {
        let roots = nodes[..stable_prefix].iter().map(stable_node_root);
        quote!(#parent.reorder_view_children(&[#(#roots),*]);)
    });
    quote!({
        let #parent = #parent_ref;
        #(#stmts)*
        #restore
    })
}

fn stable_descendant_names(node: &Node, names: &mut HashSet<String>) {
    if let Some(binding) = node_binding(node).filter(|binding| is_stable_binding(binding)) {
        names.insert(binding.name.to_string());
    }
    match node {
        Node::Container(node) => {
            for child in &node.body {
                stable_descendant_names(child, names);
            }
        }
        Node::Scroll(node) => {
            for child in &node.body {
                stable_descendant_names(child, names);
            }
        }
        Node::Button(node) => {
            if let ButtonContent::Children { body, .. } = &node.content {
                for child in body {
                    stable_descendant_names(child, names);
                }
            }
        }
        Node::Component(_)
        | Node::If(_)
        | Node::For(_)
        | Node::Text(_)
        | Node::Input(_)
        | Node::Img(_)
        | Node::Icon(_)
        | Node::Shape(_)
        | Node::StateWidget(_)
        | Node::Tooltip(_)
        | Node::Call(_)
        | Node::Children(_)
        | Node::Let(_)
        | Node::Escape(_) => {}
    }
}

fn stable_descendant_bindings<'a>(node: &'a Node, bindings: &mut Vec<&'a AsBinding>) {
    if let Some(binding) = node_binding(node).filter(|binding| is_stable_binding(binding)) {
        bindings.push(binding);
    }
    match node {
        Node::Container(node) => {
            for child in &node.body {
                stable_descendant_bindings(child, bindings);
            }
        }
        Node::Scroll(node) => {
            for child in &node.body {
                stable_descendant_bindings(child, bindings);
            }
        }
        Node::Button(node) => {
            if let ButtonContent::Children { body, .. } = &node.content {
                for child in body {
                    stable_descendant_bindings(child, bindings);
                }
            }
        }
        _ => {}
    }
}

/// Topologically schedules an all-stable sibling run. Attachment order is
/// restored afterward, so only handle construction order changes.
fn scheduled_sibling_order(nodes: &[Node]) -> Option<Vec<usize>> {
    if nodes
        .iter()
        .any(|node| node_binding(node).is_none_or(|binding| !is_stable_binding(binding)))
    {
        return None;
    }
    let mut owners = HashMap::<String, usize>::new();
    for (index, node) in nodes.iter().enumerate() {
        let mut names = HashSet::new();
        stable_descendant_names(node, &mut names);
        owners.extend(names.into_iter().map(|name| (name, index)));
    }
    let mut remaining = (0..nodes.len()).collect::<Vec<_>>();
    let mut order = Vec::with_capacity(nodes.len());
    while !remaining.is_empty() {
        let ready = remaining.iter().position(|&index| {
            let mut bindings = Vec::new();
            stable_descendant_bindings(&nodes[index], &mut bindings);
            bindings.into_iter().all(|binding| {
                let dependencies = IMMEDIATE_DEPS
                    .with(|deps| {
                        deps.borrow()
                            .get(&(binding as *const AsBinding as usize))
                            .cloned()
                    })
                    .unwrap_or_default();
                dependencies.into_iter().all(|dependency| {
                    owners
                        .get(&dependency)
                        .is_none_or(|owner| *owner == index || order.contains(owner))
                })
            })
        });
        let ready = ready?;
        order.push(remaining.remove(ready));
    }
    Some(order)
}

fn stable_node_root(node: &Node) -> TokenStream2 {
    let handle = node_handle(node).expect("scheduled stable nodes have handles");
    match node {
        Node::Container(_)
        | Node::Text(_)
        | Node::Button(_)
        | Node::Input(_)
        | Node::Img(_)
        | Node::Icon(_)
        | Node::Shape(_) => quote!(&#handle),
        Node::Scroll(_) | Node::StateWidget(_) => quote!(#handle.root()),
        Node::Component(_) | Node::Call(_) => quote!(ComponentHandle::root(&#handle)),
        Node::Tooltip(_)
        | Node::If(_)
        | Node::For(_)
        | Node::Children(_)
        | Node::Let(_)
        | Node::Escape(_) => {
            unreachable!("scheduled stable nodes have roots")
        }
    }
}

fn emit_node(node: &Node, owns_parent: bool) -> TokenStream2 {
    match node {
        Node::Container(c) => emit_container(c),
        Node::Scroll(s) => emit_scroll(s),
        Node::Text(t) => emit_text(t),
        Node::Button(b) => emit_button(b),
        Node::Input(i) => emit_input(i),
        Node::Img(i) => emit_img(i),
        Node::Icon(i) => emit_icon(i),
        Node::Shape(shape) => emit_shape(shape),
        Node::StateWidget(w) => emit_state_widget(w),
        Node::Tooltip(t) => emit_tooltip(t),
        Node::Component(c) => emit_component(c),
        Node::Call(c) => emit_call(c),
        // Invokes the enclosing component's `children` closure at this point in
        // the tree, building the caller's nested nodes into the current parent.
        Node::Children(tag) => {
            let parent = parent_ident();
            quote!((#tag)(#parent);)
        }
        Node::If(c) => emit_if(c, owns_parent),
        Node::For(f) => emit_for(f, owns_parent),
        Node::Let(l) => emit_let(l),
        // An escape inlines its statements into the surrounding children
        // block (rather than opening a nested scope), so a `let` in it is
        // visible to the sibling nodes that follow — the shared-scope
        // behaviour the hatch exists for.
        Node::Escape(block) => {
            let stmts = &block.stmts;
            quote!(#(#stmts)*)
        }
    }
}

/// A `let` binding, emitted straight into the surrounding children block so it
/// scopes over the siblings that follow — the same inlining an escape block
/// gets, and for the same reason.
///
/// The memo form wraps its body in a [`Derived`], whose recomputation is what
/// makes `$name` reads of it re-run. The closure is `move` so the binding
/// outlives the build, and the whole statement is spanned at the name the
/// author wrote so a type error blames the memo rather than the macro.
fn emit_let(binding: &LetNode) -> TokenStream2 {
    match &binding.kind {
        LetKind::Value(local) => quote!(#local),
        LetKind::Memo {
            name,
            name_span,
            expr,
        } => {
            let core = core_crate_path();
            quote_spanned!(*name_span=> let #name = #core::Derived::new(move || #expr);)
        }
    }
}

/// A statement's tokens, or its error as an inline `compile_error!` — so one
/// bad attribute reports itself without suppressing the rest of the node.
pub(crate) fn stmt_or_error(stmt: Result<TokenStream2>) -> TokenStream2 {
    stmt.unwrap_or_else(|err| err.to_compile_error())
}

/// The `__interaction` binding the state folds read, bound only when there
/// are any: a style's own conditions ask its context instead, so a node that
/// names one but declares no state block of its own must not allocate the
/// element's interaction state.
fn interaction_binding(elem: &Ident, folds: &[TokenStream2]) -> TokenStream2 {
    if folds.is_empty() {
        quote!()
    } else {
        quote!(let __interaction = #elem.interaction();)
    }
}

/// The ident a node's style context is bound to. `mixed_site`, so every
/// emitter's separate `quote!` resolves to the same binding.
fn style_ctx_ident() -> Ident {
    Ident::new("__style_cx", Span::mixed_site())
}

/// The context for a `|`-applied style, whose layer channels reset before
/// they push.
fn style_ctx_replace_ident() -> Ident {
    Ident::new("__style_cx_replace", Span::mixed_site())
}

/// One `#style` reference, lowered.
pub(crate) struct StyleRef {
    path: TokenStream2,
    replace: bool,
    guard: Option<TokenStream2>,
}

/// A node's valued attributes, split by what they lower to.
#[derive(Default)]
pub(crate) struct SplitAttrs {
    /// The node's `#style` references, in source order. Their folds seed
    /// every channel below, so the node's own attributes land on top.
    pub(crate) styles: Vec<StyleRef>,
    /// The part these fragments style, when they are a part run's rather than
    /// the node's own: the styles then fold through their part channels.
    pub(crate) part: Option<String>,
    pub(crate) style_frags: Vec<TokenStream2>,
    style_reactive: bool,
    pub(crate) visual_frags: Vec<TokenStream2>,
    visual_reactive: bool,
    /// Replacement groups already opened by this authored run. The first
    /// ordinary declaration resets its inherited channel; later declarations
    /// join that replacement group. `name:+value` never opens or resets one.
    visual_layers: VisualLayerRuns,
    /// The `opacity:` value and whether it is reactive.
    opacity: Option<(TokenStream2, bool)>,
    /// Ordered positional backdrop-filter slots.
    filters: Vec<TokenStream2>,
    /// Whether any base filter argument was reactive, so the chain rebuilds
    /// per frame even without state-scoped filters.
    filters_reactive: bool,
    /// Ordered visual-mask slots.
    masks: Vec<TokenStream2>,
    masks_reactive: bool,
    /// Resize edge declarations, accumulated as a set in source order.
    resize: Vec<(TokenStream2, bool)>,
    /// The `translate:` value and whether it is reactive.
    translate: Option<(TokenStream2, bool)>,
    /// The `translate-children:` value and whether it is reactive.
    translate_children: Option<(TokenStream2, bool)>,
    /// Text-style fields authored on this element.
    /// Every `font-color:` on the node, in source order — repeating it stacks
    /// the paint layers, the same rule `fill:` and `filter:` follow.
    font_color: Vec<(TokenStream2, bool)>,
    font_family: Option<(TokenStream2, bool)>,
    font_size: Option<(TokenStream2, bool)>,
    font_weight: Option<(TokenStream2, bool)>,
    line_height: Option<(TokenStream2, bool)>,
    font_style: Option<(TokenStream2, bool)>,
    font_stretch: Option<(TokenStream2, bool)>,
    letter_spacing: Option<(TokenStream2, bool)>,
    text_wrap: Option<(TokenStream2, bool)>,
    tab_size: Option<(TokenStream2, bool)>,
    text_transform: Option<(TokenStream2, bool)>,
    underline: Option<(TokenStream2, bool)>,
    strikethrough: Option<(TokenStream2, bool)>,
    overline: Option<(TokenStream2, bool)>,
    /// Per-state overrides, indexed in precedence order (focused, hover,
    /// pressed, disabled) so the emitted folds apply lowest-precedence first.
    states: [Option<StateFrags>; 4],
    /// Source-ordered guarded runs. These fold after the preceding base run
    /// and before the node's final interaction-state layers.
    authored_states: Vec<(TokenStream2, StateFrags)>,
    custom_states: Vec<(TokenStream2, StateFrags)>,
    /// Flags, handlers, lifecycle declarations, and inline errors, in
    /// declaration order.
    pub(crate) extras: Vec<TokenStream2>,
}

/// One state block's routed fragments — the subset of the attribute surface
/// that can vary by interaction state.
#[derive(Default)]
pub(crate) struct StateFrags {
    pub(crate) style_frags: Vec<TokenStream2>,
    pub(crate) visual_frags: Vec<TokenStream2>,
    /// See [`SplitAttrs::visual_layers`]. Each state is its own authored layer.
    visual_layers: VisualLayerRuns,
    opacity: Option<TokenStream2>,
    filters: Vec<TokenStream2>,
    masks: Vec<TokenStream2>,
    translate: Option<TokenStream2>,
    translate_children: Option<TokenStream2>,
}

#[derive(Default)]
struct VisualLayerRuns {
    fill: bool,
    stroke: bool,
    shadow: bool,
    light: bool,
}

impl VisualLayerRuns {
    /// Whether a component builder call must use its additive delegating
    /// setter. The first ordinary value opens the replacement group; later
    /// ordinary values join it through the same setter as an explicit `:+`.
    fn component_concat(&mut self, name: &str, explicit: bool) -> bool {
        if explicit {
            return true;
        }
        let seen = match name {
            "fill" => &mut self.fill,
            "stroke" => &mut self.stroke,
            "shadow" => &mut self.shadow,
            "light" => &mut self.light,
            _ => return false,
        };
        let concat = *seen;
        *seen = true;
        concat
    }
}

/// The precedence slot a state folds into: base < focused < hover < pressed < disabled.
pub(crate) fn state_index(state: InteractionState) -> usize {
    match state {
        InteractionState::Focused => 0,
        InteractionState::Hover => 1,
        InteractionState::Pressed => 2,
        InteractionState::Disabled => 3,
    }
}

/// The reactive condition guarding a precedence slot's fold, read from the
/// element's `__interaction` handle.
fn state_cond(index: usize) -> TokenStream2 {
    match index {
        0 => quote!(__interaction.focus_visible()),
        1 => quote!(__interaction.hovered()),
        2 => quote!(__interaction.pressed()),
        _ => quote!(__interaction.disabled()),
    }
}

fn view_name(name: &str) -> String {
    name.replace('_', "-")
}

fn unknown_attr_message(name: &str, vocab: &[&str]) -> String {
    let mut msg = format!("unknown attribute `{}`", view_name(name));
    let legacy_text_name = match name {
        "size" => Some("font_size"),
        "color" => Some("font_color"),
        "family" => Some("font_family"),
        "weight" => Some("font_weight"),
        "font_line_height" => Some("line_height"),
        _ => None,
    };
    if let Some(meant) = legacy_text_name.or_else(|| did_you_mean(name, vocab)) {
        msg.push_str(&format!(" — did you mean `{}`?", view_name(meant)));
    } else {
        msg.push_str(&format!(
            " — expected one of: {}",
            vocab
                .iter()
                .map(|name| view_name(name))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    msg
}

impl SplitAttrs {
    pub(crate) fn visual_base_is_reactive(&self) -> bool {
        self.visual_reactive
    }

    /// Routes one `name:value` attribute — element attribute, lifecycle
    /// declaration, style fragment, visual fragment, or an unknown-attribute
    /// `compile_error!` suggesting from the node's vocabulary (`extra_vocab`
    /// plus the shared attr set).
    pub(crate) fn route_valued(
        &mut self,
        elem: &Ident,
        name: &Ident,
        value: &Value,
        extra_vocab: &[&'static str],
    ) {
        if let Value::Grouped { values, .. } = value {
            for value in values {
                self.route_valued(elem, name, value, extra_vocab);
            }
            return;
        }
        if matches!(value, Value::Inherit { .. }) {
            let key = name.to_string();
            if key == "gap" {
                self.style_frags.push(quote!(.inherit(|__style| {
                    __style.row_gap.mark_inherit();
                    __style.column_gap.mark_inherit();
                })));
                return;
            }
            let field = match key.as_str() {
                "row_gap" | "col_gap" | "grow" | "shrink" | "basis" | "width" | "height"
                | "min_width" | "max_width" | "min_height" | "max_height" | "margin"
                | "justify" => Some((
                    true,
                    match key.as_str() {
                        "col_gap" => "column_gap",
                        other => other,
                    },
                )),
                "pad" => Some((true, "padding")),
                "align" => Some((true, "align_items")),
                "place" => Some((true, "align_self")),
                "content_align" | "items_align" | "self_align" => Some((true, key.as_str())),
                "fill" | "radius" | "exponent" | "stroke" | "shadow" | "light" => {
                    let field = match key.as_str() {
                        "radius" => "radii",
                        "exponent" => "exponents",
                        "stroke" => "strokes",
                        "shadow" => "shadows",
                        "light" => "lights",
                        other => other,
                    };
                    Some((false, field))
                }
                "rotate" => Some((false, "rotation")),
                _ => None,
            };
            if let Some((style, field)) = field {
                let field = Ident::new(field, name.span());
                if style {
                    self.style_frags
                        .push(quote!(.inherit(|__style| __style.#field.mark_inherit())));
                } else {
                    self.visual_frags
                        .push(quote!(.inherit(|__visual| __visual.#field.mark_inherit())));
                }
            } else {
                let method = match key.as_str() {
                    "opacity" => Some("inherit_opacity"),
                    "translate" => Some("inherit_translate"),
                    "translate_children" => Some("inherit_translate_children"),
                    "font_color" => Some("inherit_font_color"),
                    "font_family" => Some("inherit_font_family"),
                    "font_size" => Some("inherit_font_size"),
                    "font_weight" => Some("inherit_font_weight"),
                    "line_height" => Some("inherit_font_line_height"),
                    "font_style" => Some("inherit_font_style"),
                    "font_stretch" => Some("inherit_font_stretch"),
                    "letter_spacing" => Some("inherit_letter_spacing"),
                    "text_wrap" => Some("inherit_text_wrap"),
                    "tab_size" => Some("inherit_tab_size"),
                    "text_transform" => Some("inherit_text_transform"),
                    "underline" => Some("inherit_underline"),
                    "strikethrough" => Some("inherit_strikethrough"),
                    "overline" => Some("inherit_overline"),
                    _ => None,
                };
                if let Some(method) = method {
                    let method = Ident::new(method, name.span());
                    self.extras.push(quote!(#elem.#method();));
                } else {
                    self.extras.push(
                        syn::Error::new(
                            name.span(),
                            format!(
                                "attribute `{}` is not inheritable",
                                view_name(&name.to_string())
                            ),
                        )
                        .to_compile_error(),
                    );
                }
            }
            return;
        }
        match name.to_string().as_str() {
            "filter" => {
                self.filters_reactive |= value.is_reactive();
                self.filters.push(resolve_theme_if_token(
                    value,
                    value.tokens(),
                    quote!(BackdropFilter::None),
                ));
            }
            "mask" => {
                self.masks_reactive |= value.is_reactive();
                self.masks.push(resolve_theme_if_token(
                    value,
                    value.tokens(),
                    quote!(MaskSpec::paint(Color::TRANSPARENT)),
                ));
            }
            "opacity" => self.opacity = Some((value.f32(), value.is_reactive())),
            "sample" => self
                .extras
                .push(stmt_or_error(backdrop_sample_stmt(elem, name, value))),
            "translate" => {
                let translated =
                    resolve_theme_if_token(value, value.translate(), quote!(Translate::ZERO));
                self.translate = Some((translated, value.is_reactive()));
            }
            "translate_children" => {
                let translated =
                    resolve_theme_if_token(value, value.translate(), quote!(Translate::ZERO));
                self.translate_children = Some((translated, value.is_reactive()));
            }
            "font_color" => self.font_color.push((value.tokens(), value.is_reactive())),
            "font_family" => {
                let reactive = value.is_reactive();
                self.font_family = Some((
                    resolve_theme_if_token(
                        value,
                        value.font_family(),
                        quote!(FontFamily::default()),
                    ),
                    reactive,
                ));
            }
            // A font length is written with a unit and lowered to the logical
            // pixels the text layer takes; `px_part` is what makes a `Length`
            // scheme token usable here too.
            "font_size" => self.font_size = Some((font_px(value), value.is_reactive())),
            "font_weight" => self.font_weight = Some((value.tokens(), value.is_reactive())),
            // Leading is written either as a length (absolute) or as a bare
            // number (a multiple of the font size); `IntoLineHeight` is what
            // lets one attribute take both.
            "line_height" => {
                self.line_height = Some((value.line_height(), value.is_reactive()));
            }
            "font_style" => {
                let reactive = value.is_reactive();
                self.font_style = Some((
                    resolve_theme_if_token(value, value.font_style(), quote!(FontStyle::default())),
                    reactive,
                ));
            }
            "font_stretch" => {
                let reactive = value.is_reactive();
                self.font_stretch = Some((
                    resolve_theme_if_token(
                        value,
                        value.font_stretch(),
                        quote!(FontStretch::default()),
                    ),
                    reactive,
                ));
            }
            "letter_spacing" => {
                self.letter_spacing = Some((font_px(value), value.is_reactive()));
            }
            "text_wrap" => {
                let reactive = value.is_reactive();
                self.text_wrap = Some((
                    resolve_theme_if_token(value, value.text_wrap(), quote!(TextWrap::default())),
                    reactive,
                ));
            }
            "tab_size" => self.tab_size = Some((value.tokens(), value.is_reactive())),
            "text_transform" => {
                let reactive = value.is_reactive();
                self.text_transform = Some((
                    resolve_theme_if_token(
                        value,
                        value.text_transform(),
                        quote!(TextTransform::default()),
                    ),
                    reactive,
                ));
            }
            "underline" => {
                let reactive = value.is_reactive();
                self.underline = Some((
                    resolve_theme_if_token(value, value.tokens(), quote!(false)),
                    reactive,
                ));
            }
            "strikethrough" => {
                let reactive = value.is_reactive();
                self.strikethrough = Some((
                    resolve_theme_if_token(value, value.tokens(), quote!(false)),
                    reactive,
                ));
            }
            "overline" => {
                let reactive = value.is_reactive();
                self.overline = Some((
                    resolve_theme_if_token(value, value.tokens(), quote!(false)),
                    reactive,
                ));
            }
            "focus_style" => self
                .extras
                .push(stmt_or_error(focus_style_stmt(elem, name, value))),
            "disabled" => self.extras.push(participation_stmt(elem, name, value)),
            "role" | "label" | "description" => self
                .extras
                .push(stmt_or_error(semantic_stmt(elem, name, value))),
            "selectable" | "searchable" => self.extras.push(participation_stmt(elem, name, value)),
            "resizable" => match resize_value_tokens(value) {
                Ok(tokens) => self.resize.push((tokens, value.is_reactive())),
                Err(error) => self.extras.push(error.to_compile_error()),
            },
            "transition" | "enter" | "exit" | "enter_exit" | "draggable" | "reorderable" => self
                .extras
                .push(stmt_or_error(lifecycle_stmt(elem, name, value))),
            _ => {
                if let Some(frag) = style_frag(name, value) {
                    self.style_reactive |= value.is_reactive();
                    self.style_frags.push(frag);
                } else if let Some(frag) = visual_frag(name, value, &mut self.visual_layers) {
                    self.visual_reactive |= value.is_reactive();
                    self.visual_frags.push(frag);
                } else {
                    let mut vocab: Vec<&str> = Vec::new();
                    vocab.extend_from_slice(extra_vocab);
                    vocab.extend(vocabulary::container_attr_names());
                    let msg = unknown_attr_message(&name.to_string(), &vocab);
                    self.extras
                        .push(syn::Error::new(name.span(), msg).to_compile_error());
                }
            }
        }
    }

    /// Routes a node's state blocks into their precedence slots. `unscopable`
    /// names attributes the node accepts only at base level (the `text`
    /// widget's text-style set — the style closure predates the element).
    fn route_states(
        &mut self,
        states: &[StateBlock],
        extra_vocab: &[&'static str],
        unscopable: &[&str],
    ) {
        for block in states {
            for (name, value) in &block.attrs {
                let values = match value {
                    Value::Grouped { values, .. } => values.as_slice(),
                    value => core::slice::from_ref(value),
                };
                for value in values {
                    let key = name.to_string();
                    if matches!(value, Value::Inherit { .. }) {
                        self.extras.push(
                            syn::Error::new(
                                name.span(),
                                "`inherit` cannot be used inside an interaction state block",
                            )
                            .to_compile_error(),
                        );
                        continue;
                    }
                    if SEMANTIC_ATTRS.contains(&key.as_str()) {
                        self.extras.push(
                            syn::Error::new(
                                name.span(),
                                format!(
                                    "semantic attribute `{}` cannot be state-scoped — bind its value directly with `{{ ... }}`",
                                    view_name(&key)
                                ),
                            )
                            .to_compile_error(),
                        );
                        continue;
                    }
                    if unscopable.contains(&key.as_str()) {
                        self.extras.push(
                            syn::Error::new(
                                name.span(),
                                format!(
                                    "`{}` is inherited text styling and cannot be state-scoped",
                                    view_name(&name.to_string())
                                ),
                            )
                            .to_compile_error(),
                        );
                        continue;
                    }
                    let frags = self.states[state_index(block.state)]
                        .get_or_insert_with(StateFrags::default);
                    match key.as_str() {
                        "transition" | "enter" | "exit" | "enter_exit" => self.extras.push(
                            syn::Error::new(
                                name.span(),
                                format!(
                                    "`{}` is a one-time declaration and cannot be state-scoped",
                                    view_name(&name.to_string())
                                ),
                            )
                            .to_compile_error(),
                        ),
                        "opacity" => frags.opacity = Some(value.f32()),
                        "filter" => frags.filters.push(value.tokens()),
                        "mask" => frags.masks.push(value.tokens()),
                        "translate" => frags.translate = Some(value.translate()),
                        "translate_children" => frags.translate_children = Some(value.translate()),
                        _ => {
                            if let Some(frag) = style_frag(name, value) {
                                frags.style_frags.push(frag);
                            } else if let Some(frag) =
                                visual_frag(name, value, &mut frags.visual_layers)
                            {
                                frags.visual_frags.push(frag);
                            } else {
                                let mut vocab: Vec<&str> = Vec::new();
                                vocab.extend_from_slice(extra_vocab);
                                vocab.extend(vocabulary::container_attr_names());
                                let msg = unknown_attr_message(&key, &vocab);
                                self.extras
                                    .push(syn::Error::new(name.span(), msg).to_compile_error());
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn route_conditional<'a>(
        &mut self,
        attrs: impl IntoIterator<Item = (&'a Ident, &'a Value)>,
        condition: TokenStream2,
        extra_vocab: &[&'static str],
        unscopable: &[&str],
    ) {
        let mut frags = StateFrags::default();
        for (name, value) in attrs {
            let values = match value {
                Value::Grouped { values, .. } => values.as_slice(),
                value => core::slice::from_ref(value),
            };
            for value in values {
                let key = name.to_string();
                if matches!(value, Value::Inherit { .. }) {
                    self.extras.push(
                        syn::Error::new(
                            name.span(),
                            "`inherit` cannot be used inside a state block",
                        )
                        .to_compile_error(),
                    );
                    continue;
                }
                if unscopable.contains(&key.as_str()) {
                    self.extras.push(
                        syn::Error::new(
                            name.span(),
                            format!(
                                "`{}` is inherited text styling and cannot be state-scoped",
                                view_name(&key)
                            ),
                        )
                        .to_compile_error(),
                    );
                    continue;
                }
                if SEMANTIC_ATTRS.contains(&key.as_str()) {
                    self.extras.push(
                        syn::Error::new(
                            name.span(),
                            format!(
                                "semantic attribute `{}` cannot be state-scoped — bind its value directly with `{{ ... }}`",
                                view_name(&key)
                            ),
                        )
                        .to_compile_error(),
                    );
                    continue;
                }
                match key.as_str() {
                    "transition" | "enter" | "exit" | "enter_exit" => self.extras.push(
                        syn::Error::new(
                            name.span(),
                            format!("`{}` cannot be state-scoped", view_name(&key)),
                        )
                        .to_compile_error(),
                    ),
                    "opacity" => frags.opacity = Some(value.f32()),
                    "filter" => frags.filters.push(value.tokens()),
                    "mask" => frags.masks.push(value.tokens()),
                    "translate" => frags.translate = Some(value.translate()),
                    "translate_children" => frags.translate_children = Some(value.translate()),
                    _ => {
                        if let Some(frag) = style_frag(name, value) {
                            frags.style_frags.push(frag);
                        } else if let Some(frag) =
                            visual_frag(name, value, &mut frags.visual_layers)
                        {
                            frags.visual_frags.push(frag);
                        } else {
                            let mut vocab: Vec<&str> = Vec::new();
                            vocab.extend_from_slice(extra_vocab);
                            vocab.extend(vocabulary::container_attr_names());
                            self.extras.push(
                                syn::Error::new(name.span(), unknown_attr_message(&key, &vocab))
                                    .to_compile_error(),
                            );
                        }
                    }
                }
            }
        }
        self.custom_states.push((condition, frags));
    }

    fn route_authored<'a>(
        &mut self,
        attrs: impl IntoIterator<Item = (&'a Ident, &'a Value)>,
        condition: TokenStream2,
        extra_vocab: &[&'static str],
        unscopable: &[&str],
    ) {
        let before = self.custom_states.len();
        self.route_conditional(attrs, condition, extra_vocab, unscopable);
        let layer = self.custom_states.remove(before);
        self.authored_states.push(layer);
    }

    /// Whether the node names any style, which is what turns every channel
    /// below into a seeded reactive binding.
    fn has_styles(&self) -> bool {
        !self.styles.is_empty()
    }

    /// The context each reference folds against.
    fn ctx_of(&self, style: &StyleRef) -> Ident {
        if style.replace {
            style_ctx_replace_ident()
        } else {
            style_ctx_ident()
        }
    }

    /// Binds the node's style contexts. Emitted once, right after the element
    /// exists, so every channel and every part run below shares them.
    fn style_ctx_stmt(&self, elem: &Ident) -> TokenStream2 {
        if !self.has_styles() {
            return quote!();
        }
        let cx = style_ctx_ident();
        let replacing = self.styles.iter().any(|style| style.replace).then(|| {
            let replace = style_ctx_replace_ident();
            quote!(let #replace = #cx.replacing();)
        });
        quote!(
            let #cx = StyleCtx::new(&#elem);
            #replacing
        )
    }

    /// Clones the contexts into a `move` closure. Each binding owns one, so a
    /// node with several reactive channels does not fight over the original.
    fn capture_ctx(&self) -> TokenStream2 {
        if !self.has_styles() {
            return quote!();
        }
        let cx = style_ctx_ident();
        let replacing = self.styles.iter().any(|style| style.replace).then(|| {
            let replace = style_ctx_replace_ident();
            quote!(let #replace = #replace.clone();)
        });
        quote!(
            let #cx = #cx.clone();
            #replacing
        )
    }

    /// The runtime test for whether any named style writes the part `name`.
    ///
    /// A widget's parts are reached through its handle's accessors, not the
    /// element part registry, so codegen synthesizes a run for each of them
    /// when a style is named. This guard keeps that from installing an
    /// identity binding over the widget's own appearance for the parts the
    /// style says nothing about.
    fn style_declares_part(&self, name: &str) -> Option<TokenStream2> {
        if !self.has_styles() {
            return None;
        }
        let tests = self.styles.iter().map(|style| {
            let path = &style.path;
            quote!(#path.declares_part(#name))
        });
        Some(quote!(#(#tests)||*))
    }

    /// Re-targets the node's contexts at one of its parts, shadowing them for
    /// the block a part run emits. A part run's own `hover { … }` then follows
    /// the part, while the styles it seeds still resolve named states on the
    /// node — the split `view!` already makes with `__root_interaction`.
    fn style_part_ctx_stmt(&self, elem: &Ident) -> TokenStream2 {
        if !self.has_styles() {
            return quote!();
        }
        let cx = style_ctx_ident();
        let replacing = self.styles.iter().any(|style| style.replace).then(|| {
            let replace = style_ctx_replace_ident();
            quote!(let #replace = #cx.replacing();)
        });
        quote!(
            let #cx = #cx.for_part(&#elem);
            #replacing
        )
    }

    /// The style folds seeding `#var`, in reference order — always ahead of
    /// the node's own fragments, which is what makes an inline attribute
    /// override the style that declared it.
    fn seeds(&self, var: &TokenStream2, channel: &str) -> Vec<TokenStream2> {
        self.styles
            .iter()
            .map(|style| {
                let path = &style.path;
                let cx = self.ctx_of(style);
                match &self.part {
                    Some(part) => {
                        let method = Ident::new(&format!("fold_part_{channel}"), Span::call_site());
                        let fold = quote!(#path.#method(#part, #var, &#cx));
                        match &style.guard {
                            Some(guard) => quote!(let #var = if #guard { #fold } else { #var };),
                            None => quote!(let #var = #fold;),
                        }
                    }
                    None => {
                        let method = Ident::new(&format!("fold_{channel}"), Span::call_site());
                        let fold = quote!(#path.#method(#var, &#cx));
                        match &style.guard {
                            Some(guard) => quote!(let #var = if #guard { #fold } else { #var };),
                            None => quote!(let #var = #fold;),
                        }
                    }
                }
            })
            .collect()
    }

    fn style_seeds(&self) -> Vec<TokenStream2> {
        self.seeds(&quote!(__style), "style")
    }

    fn visual_seeds(&self) -> Vec<TokenStream2> {
        self.seeds(&quote!(__visual), "visual")
    }

    /// The state-independent half of each named style's visual fold. Widget
    /// bindings insert their own state selection between these folds and the
    /// matching state folds below.
    fn visual_base_seeds(&self) -> Vec<TokenStream2> {
        self.seeds(&quote!(__visual), "visual_base")
    }

    /// The active-state half of each named style's visual fold.
    fn visual_state_seeds(&self) -> Vec<TokenStream2> {
        self.seeds(&quote!(__visual), "visual_states")
    }

    /// Explicit reactive base values intentionally land after the widget has
    /// selected its state, both inline and in a named style.
    fn visual_override_seeds(&self) -> Vec<TokenStream2> {
        self.seeds(&quote!(__visual), "visual_override")
    }

    /// The element-setter channels every named style writes, plus the parts
    /// it declares that this node does not style itself — those the node does
    /// style are seeded into the node's own part runs instead, so the two
    /// never race for one binding.
    fn style_element_stmts(&self, elem: &Ident, declared: &[String]) -> TokenStream2 {
        let stmts = self.styles.iter().map(|style| {
            let path = &style.path;
            let cx = self.ctx_of(style);
            if style.guard.is_some() {
                return quote!();
            }
            match &self.part {
                Some(part) => quote!(#path.apply_part_element(#part, &#elem, &#cx);),
                None => quote!(
                    #path.apply_element(&#elem, &#cx);
                    #path.apply_parts_except(&#cx, &[#(#declared),*]);
                ),
            }
        });
        quote!(#(#stmts)*)
    }

    /// Whether any state block carries fragments selected by `pick`.
    fn any_state(&self, pick: impl Fn(&StateFrags) -> bool) -> bool {
        self.authored_states.iter().any(|(_, state)| pick(state))
            || self.states.iter().flatten().any(&pick)
            || self.custom_states.iter().any(|(_, state)| pick(state))
    }

    /// The per-state conditional refolds of `#var` (a `Style` or `Visual`
    /// under construction), in precedence order.
    pub(crate) fn chain_folds(
        &self,
        var: &TokenStream2,
        pick: impl Fn(&StateFrags) -> &[TokenStream2],
    ) -> Vec<TokenStream2> {
        let mut folds = Vec::new();
        for (cond, state) in &self.authored_states {
            let frags = pick(state);
            if !frags.is_empty() {
                folds.push(quote!(let #var = if #cond { #var #(#frags)* } else { #var };));
            }
        }
        for (index, slot) in self.states[..3].iter().enumerate() {
            let Some(frags) = slot else { continue };
            let frags = pick(frags);
            if frags.is_empty() {
                continue;
            }
            let cond = state_cond(index);
            folds.push(quote!(let #var = if #cond { #var #(#frags)* } else { #var };));
        }
        for (cond, state) in &self.custom_states {
            let frags = pick(state);
            if !frags.is_empty() {
                folds.push(quote!(let #var = if #cond { #var #(#frags)* } else { #var };));
            }
        }
        if let Some(frags) = &self.states[3] {
            let frags = pick(frags);
            if !frags.is_empty() {
                folds.push(quote!(let #var = if __interaction.disabled() { #var #(#frags)* } else { #var };));
            }
        }
        folds
    }

    /// The per-state replacements of the scalar `__value`, in precedence
    /// order; `coerce` wraps each branch (identity for `f32`, `.into()` for
    /// `Translate`).
    fn value_folds(
        &self,
        pick: impl Fn(&StateFrags) -> Option<&TokenStream2>,
        coerce: impl Fn(&TokenStream2) -> TokenStream2,
    ) -> Vec<TokenStream2> {
        let mut folds = Vec::new();
        for (cond, state) in &self.authored_states {
            let Some(value) = pick(state) else { continue };
            let value = coerce(value);
            folds.push(quote!(let __value = if #cond { #value } else { __value };));
        }
        for (index, slot) in self.states[..3].iter().enumerate() {
            let Some(value) = slot.as_ref().and_then(&pick) else {
                continue;
            };
            let cond = state_cond(index);
            let value = coerce(value);
            folds.push(quote!(let __value = if #cond { #value } else { __value };));
        }
        for (cond, state) in &self.custom_states {
            let Some(value) = pick(state) else { continue };
            let value = coerce(value);
            folds.push(quote!(let __value = if #cond { #value } else { __value };));
        }
        if let Some(value) = self.states[3].as_ref().and_then(&pick) {
            let value = coerce(value);
            folds.push(
                quote!(let __value = if __interaction.disabled() { #value } else { __value };),
            );
        }
        folds
    }

    /// The style application for the element a container node creates:
    /// `base` is the tag's `Style` constructor, user fragments chain onto it,
    /// and state folds — when present — turn the whole thing into one
    /// reactive binding.
    fn container_style_stmt(
        &self,
        elem: &Ident,
        parent: &Ident,
        base: TokenStream2,
    ) -> TokenStream2 {
        let frags = &self.style_frags;
        let style_expr = quote!(#base #(#frags)*);
        if self.has_styles() {
            // The style folds need the element they resolve against, so the
            // node is built from its bare tag base and everything — styles
            // first, then its own attributes, then its states — lands in the
            // one binding the element holds.
            let ctx = self.style_ctx_stmt(elem);
            let capture = self.capture_ctx();
            let seeds = self.style_seeds();
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            let interaction = interaction_binding(elem, &folds);
            return quote!(
                let #elem = #parent.child(#base);
                #ctx
                {
                    #capture
                    #interaction
                    #elem.style_dyn(move || {
                        let __style = #base;
                        #(#seeds)*
                        let __style = __style #(#frags)*;
                        #(#folds)*
                        __style
                    });
                }
            );
        }
        if self.any_state(|s| !s.style_frags.is_empty()) {
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            quote!(
                let #elem = #parent.child(#base);
                {
                    let __interaction = #elem.interaction();
                    #elem.style_dyn(move || {
                        let __style = #style_expr;
                        #(#folds)*
                        __style
                    });
                }
            )
        } else if self.style_reactive {
            quote!(let #elem = #parent.child(#base); #elem.style_dyn(move || #style_expr);)
        } else {
            quote!(let #elem = #parent.child(#style_expr);)
        }
    }

    /// Like [`container_style_stmt`](Self::container_style_stmt) but roots the
    /// element at a detached [`Element::orphan`] — no enclosing `parent` — for
    /// the header-less `view!{}` top node.
    fn orphan_style_stmt(&self, elem: &Ident, base: TokenStream2) -> TokenStream2 {
        let frags = &self.style_frags;
        let style_expr = quote!(#base #(#frags)*);
        if self.has_styles() {
            let ctx = self.style_ctx_stmt(elem);
            let capture = self.capture_ctx();
            let seeds = self.style_seeds();
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            let interaction = interaction_binding(elem, &folds);
            return quote!(
                let #elem = Element::orphan(#base);
                #ctx
                {
                    #capture
                    #interaction
                    #elem.style_dyn(move || {
                        let __style = #base;
                        #(#seeds)*
                        let __style = __style #(#frags)*;
                        #(#folds)*
                        __style
                    });
                }
            );
        }
        if self.any_state(|s| !s.style_frags.is_empty()) {
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            quote!(
                let #elem = Element::orphan(#base);
                {
                    let __interaction = #elem.interaction();
                    #elem.style_dyn(move || {
                        let __style = #style_expr;
                        #(#folds)*
                        __style
                    });
                }
            )
        } else if self.style_reactive {
            quote!(let #elem = Element::orphan(#base); #elem.style_dyn(move || #style_expr);)
        } else {
            quote!(let #elem = Element::orphan(#style_expr);)
        }
    }

    /// The visual application over a fresh `Visual::new()`, for nodes whose
    /// appearance is entirely the author's (containers, `text`).
    fn visual_stmt(&self, elem: &Ident) -> TokenStream2 {
        let has_states = self.any_state(|s| !s.visual_frags.is_empty());
        if self.has_styles() {
            let capture = self.capture_ctx();
            let seeds = self.visual_seeds();
            let frags = &self.visual_frags;
            let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
            let interaction = interaction_binding(elem, &folds);
            return quote!({
                #capture
                #interaction
                #elem.visual_dyn(move || {
                    let __visual = Visual::new();
                    #(#seeds)*
                    let __visual = __visual #(#frags)*;
                    #(#folds)*
                    __visual
                });
            });
        }
        if self.visual_frags.is_empty() && !has_states {
            return quote!();
        }
        let frags = &self.visual_frags;
        let visual_expr = quote!(Visual::new() #(#frags)*);
        if has_states {
            let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
            quote!({
                let __interaction = #elem.interaction();
                #elem.visual_dyn(move || {
                    let __visual = #visual_expr;
                    #(#folds)*
                    __visual
                });
            })
        } else if self.visual_reactive {
            quote!(#elem.visual_dyn(move || #visual_expr);)
        } else {
            quote!(#elem.visual(#visual_expr);)
        }
    }

    /// Visual patches for an already-built widget part. Unlike
    /// [`visual_stmt`](Self::visual_stmt), this folds over the widget's current
    /// visual instead of replacing it.
    fn revisual_stmt(&self, elem: &Ident) -> TokenStream2 {
        let has_states = self.any_state(|s| !s.visual_frags.is_empty());
        if self.has_styles() {
            let capture = self.capture_ctx();
            let seeds = self.visual_seeds();
            let frags = &self.visual_frags;
            let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
            let interaction = interaction_binding(elem, &folds);
            return quote!({
                #capture
                #interaction
                #elem.revisual_dyn(move |__visual| {
                    #(#seeds)*
                    let __visual = __visual #(#frags)*;
                    #(#folds)*
                    __visual
                });
            });
        }
        if self.visual_frags.is_empty() && !has_states {
            return quote!();
        }
        let frags = &self.visual_frags;
        let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
        if has_states {
            quote!({
                let __interaction = #elem.interaction();
                #elem.revisual_dyn(move |__visual| {
                    let __visual = __visual #(#frags)*;
                    #(#folds)*
                    __visual
                });
            })
        } else if self.visual_reactive {
            quote!(#elem.revisual_dyn(move |__visual| __visual #(#frags)*);)
        } else {
            quote!(#elem.revisual(|__visual| __visual #(#frags)*);)
        }
    }

    /// The style application for a node built by a widget (or returned by a
    /// component): user fragments fold over the root style the widget set —
    /// its layout contract — via the restyle seam, instead of replacing it;
    /// state folds make the binding reactive.
    fn restyle_stmt(&self, elem: &Ident) -> TokenStream2 {
        let has_states = self.any_state(|s| !s.style_frags.is_empty());
        if self.has_styles() {
            let capture = self.capture_ctx();
            let seeds = self.style_seeds();
            let frags = &self.style_frags;
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            let interaction = interaction_binding(elem, &folds);
            return quote!({
                #capture
                #interaction
                #elem.restyle_dyn(move |__style| {
                    #(#seeds)*
                    let __style = __style #(#frags)*;
                    #(#folds)*
                    __style
                });
            });
        }
        if self.style_frags.is_empty() && !has_states {
            return quote!();
        }
        let frags = &self.style_frags;
        if has_states {
            let folds = self.chain_folds(&quote!(__style), |s| &s.style_frags);
            let rebase = (!frags.is_empty()).then(|| quote!(let __style = __style #(#frags)*;));
            quote!({
                let __interaction = #elem.interaction();
                #elem.restyle_dyn(move |__style| {
                    #rebase
                    #(#folds)*
                    __style
                });
            })
        } else if self.style_reactive {
            quote!(#elem.restyle_dyn(move |__style| __style #(#frags)*);)
        } else {
            quote!(#elem.restyle(|__style| __style #(#frags)*);)
        }
    }

    /// Rebuilds a widget-owned visual in the one binding an element holds.
    /// Static authored values customize the resting visual before the widget
    /// selects hover/pressed/checked/etc.; authored state blocks then override
    /// that selected state. An explicitly reactive base value is the escape
    /// hatch and lands after widget selection, so it owns the property it
    /// computes without discarding the rest of the widget visual.
    fn widget_visual_stmt(
        &self,
        elem: &Ident,
        style_ty: &str,
        base: TokenStream2,
        widget_state: TokenStream2,
    ) -> TokenStream2 {
        let has_states = self.any_state(|s| !s.visual_frags.is_empty());
        if self.visual_frags.is_empty() && !has_states && !self.has_styles() {
            return quote!();
        }
        let frags = &self.visual_frags;
        let style_ty = Ident::new(style_ty, Span::call_site());
        let widgets = crate::component::widgets_path();
        let base_seeds = self.visual_base_seeds();
        let override_seeds = self.visual_override_seeds();
        let state_seeds = self.visual_state_seeds();
        let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
        let authored_base =
            (!self.visual_reactive).then(|| quote!(let __visual = __visual #(#frags)*;));
        let reactive_override = self
            .visual_reactive
            .then(|| quote!(let __visual = __visual #(#frags)*;));
        let body = if has_states || self.has_styles() || self.visual_reactive {
            quote!({
                let __visual = #base;
                #(#base_seeds)*
                #authored_base
                let __visual = #widget_state;
                #(#override_seeds)*
                #reactive_override
                #(#state_seeds)*
                #(#folds)*
                __visual
            })
        } else {
            quote!({
                let __visual = #base #(#frags)*;
                #widget_state
            })
        };
        let capture = self.capture_ctx();
        quote!({
            #capture
            let __widget_style = #widgets::#style_ty::default();
            let __interaction = #elem.interaction();
            #elem.visual_dyn(move || #body);
        })
    }

    /// The visual application for a component's returned root: state
    /// fragments fold over whatever visual the component itself declared,
    /// captured at bind time by the revisual seam.
    fn patch_visual_stmt(&self, elem: &Ident) -> TokenStream2 {
        if !self.any_state(|s| !s.visual_frags.is_empty()) && !self.has_styles() {
            return quote!();
        }
        let folds = self.chain_folds(&quote!(__visual), |s| &s.visual_frags);
        let capture = self.capture_ctx();
        let seeds = self.visual_seeds();
        quote!({
            #capture
            let __interaction = #elem.interaction();
            #elem.revisual_dyn(move |__visual| {
                #(#seeds)*
                #(#folds)*
                __visual
            });
        })
    }

    /// The `opacity` application: plain (static or reactive) without state
    /// overrides, one folding reactive binding with them (base 1.0).
    fn opacity_stmt(&self, elem: &Ident) -> TokenStream2 {
        let has_states = self.any_state(|s| s.opacity.is_some());
        let marks_disabled = self.states[state_index(InteractionState::Disabled)]
            .as_ref()
            .is_some_and(|state| state.opacity.is_some())
            .then(|| quote!(#elem.authored_disabled_opacity();));
        match (&self.opacity, has_states) {
            (None, false) => quote!(#marks_disabled),
            (Some((value, reactive)), false) => {
                if *reactive {
                    quote!(#marks_disabled #elem.opacity_dyn(move || #value);)
                } else {
                    quote!(#marks_disabled #elem.opacity(#value);)
                }
            }
            (base, true) => {
                let base = base
                    .as_ref()
                    .map(|(value, _)| value.clone())
                    .unwrap_or_else(|| quote!(1.0f32));
                let folds = self.value_folds(|s| s.opacity.as_ref(), |v| quote!(#v));
                quote!({
                    #marks_disabled
                    let __interaction = #elem.interaction();
                    #elem.opacity_dyn(move || {
                        let __value = #base;
                        #(#folds)*
                        __value
                    });
                })
            }
        }
    }

    fn filters_stmt(&self, elem: &Ident) -> TokenStream2 {
        let base = &self.filters;
        let has_states = self.any_state(|s| !s.filters.is_empty());
        if !has_states {
            return if base.is_empty() {
                quote!()
            } else if self.filters_reactive {
                quote!(#elem.filters_dyn(move || FilterChain::new() #(.push(#base))*);)
            } else {
                quote!(#elem.filters(FilterChain::new() #(.push(#base))*);)
            };
        }
        let mut folds = Vec::new();
        for (index, slot) in self.states.iter().enumerate() {
            let Some(frags) = slot else { continue };
            if frags.filters.is_empty() {
                continue;
            }
            let condition = state_cond(index);
            let filters = &frags.filters;
            folds.push(quote!(if #condition {
                __value = __value.patch(&(FilterChain::new() #(.push(#filters))*));
            }));
        }
        quote!({
            let __interaction = #elem.interaction();
            #elem.filters_dyn(move || {
                let mut __value = FilterChain::new() #(.push(#base))*;
                #(#folds)*
                __value
            });
        })
    }

    fn masks_stmt(&self, elem: &Ident) -> TokenStream2 {
        let base = &self.masks;
        let has_states = self.any_state(|state| !state.masks.is_empty());
        if !has_states {
            return if base.is_empty() {
                quote!()
            } else if self.masks_reactive {
                quote!(#elem.masks_dyn(move || vec![#(#base),*]);)
            } else {
                quote!(#elem.masks(vec![#(#base),*]);)
            };
        }
        let mut folds = Vec::new();
        for (index, slot) in self.states.iter().enumerate() {
            let Some(frags) = slot else { continue };
            if frags.masks.is_empty() {
                continue;
            }
            let condition = state_cond(index);
            let masks = &frags.masks;
            folds.push(quote!(if #condition { __value = vec![#(#masks),*]; }));
        }
        quote!({
            let __interaction = #elem.interaction();
            #elem.masks_dyn(move || {
                let mut __value = vec![#(#base),*];
                #(#folds)*
                __value
            });
        })
    }

    /// A `translate`/`translate-children` application, mirroring
    /// [`opacity_stmt`] with `Translate::ZERO` as the fold base.
    fn translate_stmt(
        &self,
        elem: &Ident,
        method: &str,
        base: &Option<(TokenStream2, bool)>,
        pick: impl Fn(&StateFrags) -> Option<&TokenStream2>,
    ) -> TokenStream2 {
        let has_states = self.any_state(|s| pick(s).is_some());
        let plain = Ident::new(method, Span::call_site());
        match (base, has_states) {
            (None, false) => quote!(),
            (Some((value, reactive)), false) => {
                if *reactive {
                    let method = Ident::new(&format!("{method}_dyn"), Span::call_site());
                    quote!(#elem.#method(move || #value);)
                } else {
                    quote!(#elem.#plain(#value);)
                }
            }
            (base, true) => {
                let method = Ident::new(&format!("{method}_dyn"), Span::call_site());
                let base = base
                    .as_ref()
                    .map(|(value, _)| value.clone())
                    .unwrap_or_else(|| quote!(Translate::ZERO));
                let folds = self.value_folds(pick, |v| quote!((#v).into()));
                quote!({
                    let __interaction = #elem.interaction();
                    #elem.#method(move || {
                        let __value: Translate = (#base).into();
                        #(#folds)*
                        __value
                    });
                })
            }
        }
    }

    /// Every element-level application beyond style/visual: opacity and the
    /// two translates, each with its state folds when present.
    pub(crate) fn element_stmts(&self, elem: &Ident) -> TokenStream2 {
        let opacity = self.opacity_stmt(elem);
        let filters = self.filters_stmt(elem);
        let masks = self.masks_stmt(elem);
        let resize = if self.resize.is_empty() {
            quote!()
        } else {
            let values = self.resize.iter().map(|(value, _)| value);
            if self.resize.iter().any(|(_, reactive)| *reactive) {
                quote!(#elem.resizable_with_dyn(move || {
                    let mut __edges = ResizeEdges::NONE;
                    #(let __value: ResizeOptions = (#values).into();
                      __edges |= __value.edges;)*
                    ResizeOptions::new(__edges)
                });)
            } else {
                quote!({
                    let mut __edges = ResizeEdges::NONE;
                    #(let __value: ResizeOptions = (#values).into();
                      __edges |= __value.edges;)*
                    #elem.resizable_with(ResizeOptions::new(__edges));
                })
            }
        };
        let translate =
            self.translate_stmt(elem, "translate", &self.translate, |s| s.translate.as_ref());
        let translate_children =
            self.translate_stmt(elem, "translate_children", &self.translate_children, |s| {
                s.translate_children.as_ref()
            });
        let font_stmt = |field: &Option<(TokenStream2, bool)>, method: &str| {
            let method = Ident::new(method, Span::call_site());
            match field {
                Some((value, true)) => {
                    let dynamic = Ident::new(&format!("{method}_dyn"), Span::call_site());
                    quote!(#elem.#dynamic(move || #value);)
                }
                Some((value, false)) => quote!(#elem.#method(#value);),
                None => quote!(),
            }
        };
        let font_color = paint_stack_stmt(elem, &self.font_color, "font_color");
        let font_family = font_stmt(&self.font_family, "font_family");
        let font_size = font_stmt(&self.font_size, "font_size");
        let font_weight = font_stmt(&self.font_weight, "font_weight");
        let line_height = font_stmt(&self.line_height, "font_line_height");
        let font_style = font_stmt(&self.font_style, "font_style");
        let font_stretch = font_stmt(&self.font_stretch, "font_stretch");
        let letter_spacing = font_stmt(&self.letter_spacing, "letter_spacing");
        let text_wrap = font_stmt(&self.text_wrap, "text_wrap");
        let tab_size = font_stmt(&self.tab_size, "tab_size");
        let text_transform = font_stmt(&self.text_transform, "text_transform");
        let underline = font_stmt(&self.underline, "underline");
        let strikethrough = font_stmt(&self.strikethrough, "strikethrough");
        let overline = font_stmt(&self.overline, "overline");
        quote!(
            #opacity #filters #masks #resize #translate #translate_children
            #font_color #font_family #font_size #font_weight #line_height
            #font_style #font_stretch #letter_spacing #text_wrap #tab_size
            #text_transform #underline #strikethrough #overline
        )
    }
}

/// Splits a container-like node's attributes into style/visual fragments and
/// extras (flags, handlers, element attributes) targeting `elem`. An
/// attribute that fails to lower becomes an inline `compile_error!` among the
/// extras; the rest of the node is unaffected.
/// Lowers a node's `#style` references. The path is emitted with the `#`'s
/// span so a name that does not resolve reports against what was written.
fn lower_styles(styles: &[StyleUse]) -> Vec<StyleRef> {
    styles
        .iter()
        .map(|style| {
            let path = &style.path;
            StyleRef {
                path: quote_spanned!(style.span=> #path),
                replace: style.replace,
                guard: None,
            }
        })
        .collect()
}

fn split_attrs(elem: &Ident, styles: &[StyleUse], attrs: &[Attr]) -> SplitAttrs {
    let mut split = SplitAttrs {
        styles: lower_styles(styles),
        ..SplitAttrs::default()
    };
    let mut after_conditional = false;
    let mut guarded = Vec::new();
    for attr in attrs {
        match attr {
            Attr::Valued(name, value) if is_grid_template_attr(name) => {
                flush_guarded(&mut split, &mut guarded);
                split.style_reactive |= value.is_reactive();
            }
            Attr::Valued(name, value) if after_conditional => guarded.push((name, value)),
            Attr::Valued(name, value) => split.route_valued(elem, name, value, &[]),
            Attr::Flag(name) if name == "backfill" => flush_guarded(&mut split, &mut guarded),
            Attr::Flag(name) => {
                flush_guarded(&mut split, &mut guarded);
                split.extras.push(stmt_or_error(flag_stmt(elem, name)));
            }
            Attr::Handler(handler) => {
                flush_guarded(&mut split, &mut guarded);
                split
                    .extras
                    .push(stmt_or_error(handler_stmt(elem, handler)));
            }
            Attr::Conditional(run) => {
                flush_guarded(&mut split, &mut guarded);
                route_conditional_run(&mut split, run, quote!(true));
                after_conditional = true;
            }
        }
    }
    flush_guarded(&mut split, &mut guarded);
    split
}

fn flush_guarded(split: &mut SplitAttrs, attrs: &mut Vec<(&Ident, &Value)>) {
    if attrs.is_empty() {
        return;
    }
    split.route_authored(attrs.drain(..), quote!(true), &[], FONT_ATTRS);
}

fn route_conditional_run(split: &mut SplitAttrs, run: &ConditionalRun, parent: TokenStream2) {
    let cond = &run.cond;
    let selected = quote!((#parent) && (#cond));
    route_conditional_branch(
        split,
        &run.styles,
        &run.attrs,
        &run.states,
        selected.clone(),
    );
    let remaining = quote!((#parent) && !(#cond));
    match run.alternative.as_deref() {
        Some(ConditionalRunAlternative::ElseIf(next)) => {
            route_conditional_run(split, next, remaining)
        }
        Some(ConditionalRunAlternative::Else {
            styles,
            attrs,
            states,
            ..
        }) => route_conditional_branch(split, styles, attrs, states, remaining),
        None => {}
    }
}

fn route_conditional_branch(
    split: &mut SplitAttrs,
    styles: &[StyleUse],
    attrs: &[Attr],
    states: &[StateBlock],
    guard: TokenStream2,
) {
    for style in styles {
        let path = &style.path;
        split
            .extras
            .push(quote_spanned!(style.span=> const { #path.assert_conditional_safe(); };));
        split.styles.push(StyleRef {
            path: quote_spanned!(style.span=> #path),
            replace: style.replace,
            guard: Some(guard.clone()),
        });
    }
    // Keep nested runs in their authored position. Each uninterrupted run of
    // direct attributes is one replacement layer; a nested conditional then
    // folds on top before later attributes resume with a fresh layer.
    let mut direct = Vec::new();
    for attr in attrs {
        match attr {
            Attr::Valued(name, value) if !is_grid_template_attr(name) => {
                direct.push((name, value));
            }
            Attr::Conditional(nested) => {
                flush_conditional_attrs(split, &mut direct, guard.clone());
                route_conditional_run(split, nested, guard.clone());
            }
            Attr::Flag(name) | Attr::Handler(Handler { name, .. }) => {
                flush_conditional_attrs(split, &mut direct, guard.clone());
                split.extras.push(
                    syn::Error::new(
                        name.span(),
                        "handlers and behavior flags are not reversible inside conditional attribute runs",
                    )
                    .to_compile_error(),
                );
            }
            Attr::Valued(name, _) => {
                flush_conditional_attrs(split, &mut direct, guard.clone());
                split.extras.push(
                    syn::Error::new(
                        name.span(),
                        "grid configuration in a conditional run must be written as a conditional value",
                    )
                    .to_compile_error(),
                );
            }
        }
    }
    flush_conditional_attrs(split, &mut direct, guard.clone());
    for block in states {
        let interaction = state_cond(state_index(block.state));
        let condition = quote!((#guard) && (#interaction));
        split.route_conditional(
            block.attrs.iter().map(|(name, value)| (name, value)),
            condition,
            &[],
            FONT_ATTRS,
        );
    }
}

fn flush_conditional_attrs(
    split: &mut SplitAttrs,
    attrs: &mut Vec<(&Ident, &Value)>,
    guard: TokenStream2,
) {
    if attrs.is_empty() {
        return;
    }
    split.route_authored(attrs.drain(..), guard, &[], FONT_ATTRS);
}

fn is_grid_template_attr(name: &Ident) -> bool {
    matches!(
        name.to_string().as_str(),
        "cols" | "rows" | "auto_cols" | "auto_rows" | "backfill"
    )
}

/// [`split_attrs`] for a leaf widget's named attributes (leaves take no bare
/// flags; handlers are collected separately by the parser).
fn split_leaf_attrs(
    elem: &Ident,
    styles: &[StyleUse],
    attrs: &[(Ident, Value)],
    extra_vocab: &[&'static str],
) -> SplitAttrs {
    let mut split = SplitAttrs {
        styles: lower_styles(styles),
        ..SplitAttrs::default()
    };
    for (name, value) in attrs {
        split.route_valued(elem, name, value, extra_vocab);
    }
    split
}

/// Stamps the built element with the node's source location, for inspection
/// tooling. The tokens carry the node's span, so `file!()`/`line!()` resolve
/// to the line the author wrote; the branch folds away in release builds.
fn stamp_source(elem: &Ident, span: Span) -> TokenStream2 {
    quote_spanned!(span=> if cfg!(debug_assertions) {
        #elem.debug_source(file!(), line!(), column!());
    })
}

fn emit_container(c: &Container) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&c.name);
    let mut split = split_attrs(&elem, &c.styles, &c.attrs);
    split.route_states(&c.states, &[], FONT_ATTRS);

    let base = base_style(c);
    let style_stmt = split.container_style_stmt(&elem, &parent, base);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;

    // A canvas builds its connectors first so they paint under the shapes they
    // attach to; the authored order is untouched, which is what the formatter
    // writes back.
    let children = match matches!(c.tag, Tag::Canvas)
        .then(|| mosaic_syntax::canvas_paint_order(&c.body))
        .flatten()
    {
        Some(order) => {
            let reordered: Vec<Node> = order
                .into_iter()
                .map(|index| c.body[index].clone())
                .collect();
            emit_block(quote!(&#elem), &reordered)
        }
        None => emit_block(quote!(&#elem), &c.body),
    };

    let stamp = stamp_source(&elem, c.span);
    let boundary_name = container_name(&c.tag);
    let boundary = quote!(#elem.inspection_container(#boundary_name););
    let expose = expose_stmt(&c.name, &elem);
    let theme = theme_scope_stmt(&c.theme, quote!(#elem));
    let fill = fill_binding(&c.name, &elem);
    let fit = canvas_fit_stmt(c, &elem);
    quote!(
        #style_stmt
        #fill
        #stamp
        #boundary
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#extras)*
        #expose
        #children;
        #fit
        #theme
    )
}

/// Makes a `canvas` shrink to its drawing when the author gave it no size.
///
/// The geometries are re-emitted here rather than read back off the built
/// elements: they are the same expressions the children were built from, so
/// what the canvas measures and what it paints cannot disagree.
fn canvas_fit_stmt(c: &Container, elem: &Ident) -> Option<TokenStream2> {
    if !matches!(c.tag, Tag::Canvas) {
        return None;
    }
    let sized = c.attrs.iter().any(|attr| {
        matches!(attr, Attr::Valued(name, _) if matches!(name.to_string().as_str(), "width" | "height"))
    });
    if sized {
        return None;
    }
    let mut geometries: Vec<TokenStream2> = Vec::new();
    for node in &c.body {
        let Node::Shape(shape) = node else { continue };
        let outline = shape_geometry_expr(shape);
        geometries.extend(marker_geometries(shape, &outline));
        geometries.push(outline);
    }
    if geometries.is_empty() {
        return None;
    }
    Some(quote!(canvas_fit(&#elem, move || ::std::vec![#(#geometries),*]);))
}

/// The header-less form: exactly one top-level element-producing node, built
/// detached and returned, with any `let` bindings hoisted ahead of it.
///
/// The bindings are split off first so the single-node rule counts only
/// structural nodes — that is what lets a component body open with a binding
/// (`view! { let title = …; col { … } }`) instead of having to hoist it out of
/// the macro. A binding written *after* the root has nothing left to scope
/// over, so it is reported rather than silently dropped.
fn emit_orphan_root(body: &[Node]) -> TokenStream2 {
    let root = body.iter().position(Node::is_structural);
    let bindings = body
        .iter()
        .take(root.unwrap_or(body.len()))
        .map(emit_node_binding)
        .collect::<Vec<_>>();
    if let Some(stray) = body
        .iter()
        .skip(root.map_or(body.len(), |index| index + 1))
        .find(|node| !node.is_structural())
    {
        return syn::Error::new(
            node_span(stray),
            "a `let` after the view's only node has nothing to scope over — \
             move it above the node",
        )
        .to_compile_error();
    }
    if bindings.is_empty() {
        return emit_orphan_structural_root(body);
    }
    let structural = body.iter().filter(|node| node.is_structural());
    let main = emit_orphan_structural_root(&structural.cloned().collect::<Vec<_>>());
    quote!({ #(#bindings)* #main })
}

/// A `let` node's statement, for the callers that hoist bindings out of a node
/// list rather than emitting the list in place.
fn emit_node_binding(node: &Node) -> TokenStream2 {
    match node {
        Node::Let(binding) => emit_let(binding),
        _ => unreachable!("only `let` nodes are hoisted as bindings"),
    }
}

fn emit_orphan_structural_root(body: &[Node]) -> TokenStream2 {
    match body {
        [Node::Container(c)] => emit_orphan_container(c),
        [
            node @ (Node::Scroll(_)
            | Node::Text(_)
            | Node::Button(_)
            | Node::Input(_)
            | Node::Img(_)
            | Node::Icon(_)
            | Node::Shape(_)
            | Node::StateWidget(_)),
        ] => emit_orphan_widget(node),
        // A single call/expression node already yields a detached `Element` —
        // return it directly (the caller adopts it), so `view! { card(x) }`
        // and `fn wrap() -> Element { view! { inner() } }` work.
        [Node::Call(c)] => {
            let call = &c.call;
            match &c.name {
                Some(_) => {
                    let elem = binding_ident(&c.name);
                    let fill = fill_call_binding(&c.name, &elem);
                    quote!({ let #elem = #call; #fill ComponentHandle::root(&#elem).clone() })
                }
                None => quote!(#call),
            }
        }
        // A single component builds a detached element — return its root
        // element (the caller adopts it), so a component body / helper can
        // return one whether or not the component exposes a handle.
        [Node::Component(c)] => {
            let (build, _, root) = component_build(c);
            quote!({ #build #root })
        }
        [other] => syn::Error::new(
            node_span(other),
            "a target-less `view! { … }` builds and returns one top-level element; \
             wrap an `if` or `for` in a container",
        )
        .to_compile_error(),
        [] => syn::Error::new(
            Span::call_site(),
            "a target-less `view! { … }` needs one top-level element-producing node",
        )
        .to_compile_error(),
        [first, ..] => syn::Error::new(
            node_span(first),
            "a target-less `view! { … }` returns one top-level element; wrap these siblings \
             in one `col`/`row`/`stack`/`el`",
        )
        .to_compile_error(),
    }
}

/// Builds a widget through its ordinary parent-taking constructor, then
/// removes the construction parent without leaving it in the returned tree.
fn emit_orphan_widget(node: &Node) -> TokenStream2 {
    let parent = parent_ident();
    let view_root = view_root_ident();
    let build = emit_node(node, false);
    let handle = node_handle(node).expect("widget nodes bind one handle");
    let root = match node {
        Node::Scroll(_) => quote!(#handle.root().clone()),
        Node::StateWidget(_) => {
            let elem = Ident::new("__widget_root", Span::mixed_site());
            quote!(#elem.clone())
        }
        Node::Text(_)
        | Node::Button(_)
        | Node::Input(_)
        | Node::Img(_)
        | Node::Icon(_)
        | Node::Shape(_) => quote!(#handle.clone()),
        _ => unreachable!("emit_orphan_widget only receives widget nodes"),
    };

    quote!({
        let #view_root = Element::orphan(Style::default());
        let #parent = &#view_root;
        #build
        let __view_widget_root = #root;
        #view_root.promote_orphan_child(&__view_widget_root)
    })
}

/// A call/expression node: the value (a detached [`Element`] or a component
/// handle, e.g. from a header-less `view!{}` or a `#[component]` fn) is
/// adopted by the structural parent through `ComponentHandle::root`. `as
/// name` binds the returned value.
fn emit_call(c: &CallNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&c.name);
    let call = &c.call;
    let expose = expose_stmt(&c.name, &elem);
    let theme = theme_scope_stmt(&c.theme, quote!(ComponentHandle::root(&#elem)));
    let fill = fill_call_binding(&c.name, &elem);
    quote!(let #elem = #call; #fill #parent.adopt(ComponentHandle::root(&#elem)); #expose #theme)
}

/// Like [`emit_container`] but roots the subtree at a detached
/// [`Element::orphan`] instead of a child of an enclosing `parent`, and
/// evaluates to the container's handle.
fn emit_orphan_container(c: &Container) -> TokenStream2 {
    let elem = binding_ident(&c.name);
    let mut split = split_attrs(&elem, &c.styles, &c.attrs);
    split.route_states(&c.states, &[], FONT_ATTRS);

    let base = base_style(c);
    let style_stmt = split.orphan_style_stmt(&elem, base);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;

    // The same two canvas rules the nested path applies: connectors build
    // first so they paint under what they attach to, and a canvas with no
    // authored size takes the size of its drawing. A `view!` whose only node
    // is the canvas comes through here instead.
    let children = match matches!(c.tag, Tag::Canvas)
        .then(|| mosaic_syntax::canvas_paint_order(&c.body))
        .flatten()
    {
        Some(order) => {
            let reordered: Vec<Node> = order
                .into_iter()
                .map(|index| c.body[index].clone())
                .collect();
            emit_block(quote!(&#elem), &reordered)
        }
        None => emit_block(quote!(&#elem), &c.body),
    };
    let fit = canvas_fit_stmt(c, &elem);

    let stamp = stamp_source(&elem, c.span);
    let boundary_name = container_name(&c.tag);
    let boundary = quote!(#elem.inspection_container(#boundary_name););
    // The root alias `expose_part` registrations reference: bound before the
    // body, so it is in scope in every nested children block. The root's own
    // `as pub` registers it as a part of itself, which is harmless.
    let root = view_root_ident();
    let expose = expose_stmt(&c.name, &elem);
    let theme = theme_scope_stmt(&c.theme, quote!(#elem));
    let fill = fill_binding(&c.name, &elem);
    quote!({
        #style_stmt
        #fill
        #stamp
        #boundary
        let #root = #elem.clone();
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#extras)*
        #expose
        #children;
        #fit
        #theme
        #elem
    })
}

/// A `scroll` node: the widget call plus its content. Attributes style the
/// *scrolling content* — the column being declared — over the widget's
/// documented contract (a non-shrinking, width-filling column); the viewport
/// is the box the scroll sits in, since the widget fills its parent. `as`
/// binds the `Scroll` handle.
fn emit_scroll(s: &ScrollNode) -> TokenStream2 {
    let parent = parent_ident();
    let handle = binding_ident(&s.name);
    let elem = Ident::new("__scroll_content", Span::mixed_site());
    let mut split = split_attrs(&elem, &s.styles, &s.attrs);
    split.route_states(&s.states, &[], FONT_ATTRS);

    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;

    let children = emit_block(quote!(&#elem), &s.body);

    let widget = Ident::new("scroll", s.span);
    let stamp = stamp_source(&elem, s.span);
    let theme = theme_scope_stmt(&s.theme, quote!(#handle.root()));
    let fill = fill_binding(&s.name, &handle);
    quote!(
        let #handle = #widget(#parent);
        #fill
        let #elem = #handle.content().clone();
        #stamp
        #style_ctx
        #style_stmt
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#extras)*
        #children;
        #theme
    )
}

fn emit_text(t: &TextNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&t.name);
    let style = quote!(TextStyle::inherited());
    // Method calls, not the free `text`/`text_dyn` functions: a scheme
    // const named `text` (a nested token group, say) would shadow those.
    let build = match &t.content {
        Content::Lit(lit) => {
            let widget = Ident::new("text_leaf", t.span);
            quote!(let #elem = #parent.#widget(#lit, #style);)
        }
        Content::Static(expr) => {
            let widget = Ident::new("text_leaf", t.span);
            quote!(let #elem = #parent.#widget(#expr, #style);)
        }
        Content::ReactiveExpr(expr) => {
            let widget = Ident::new("text_leaf_dyn", t.span);
            quote!(let #elem = #parent.#widget(move || #expr, #style);)
        }
        Content::Reactive(block) => {
            let widget = Ident::new("text_leaf_dyn", t.span);
            quote!(let #elem = #parent.#widget(move || #block, #style);)
        }
    };
    let handlers = t
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = t.flags.iter().map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let mut split = split_leaf_attrs(&elem, &t.styles, &t.attrs, &[]);
    split.route_states(&t.states, &[], FONT_ATTRS);
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let stamp = stamp_source(&elem, t.span);
    let expose = expose_stmt(&t.name, &elem);
    let theme = theme_scope_stmt(&t.theme, quote!(#elem));
    let fill = fill_binding(&t.name, &elem);
    let tooltip = t.tooltip.as_ref().map(|tooltip| {
        let attachment = emit_tooltip(tooltip);
        let parent = parent_ident();
        quote!({ let #parent = &#elem; #attachment })
    });
    quote!(#build #fill #stamp #style_ctx #style_stmt #visual_stmt #style_element_stmts #element_stmts #(#flags)* #(#handlers)* #(#extras)* #expose #tooltip #theme)
}

fn emit_tooltip(t: &TooltipNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = Ident::new("__tooltip_root", Span::mixed_site());
    let trigger = Ident::new(
        match t.trigger {
            TooltipTrigger::HoverFocus => "HoverFocus",
            TooltipTrigger::Hover => "Hover",
            TooltipTrigger::Focus => "Focus",
            TooltipTrigger::Always => "Always",
            TooltipTrigger::Manual => "Manual",
        },
        t.span,
    );
    let side = Ident::new(
        match t.side {
            TooltipSide::Top => "Top",
            TooltipSide::Bottom => "Bottom",
            TooltipSide::Left => "Left",
            TooltipSide::Right => "Right",
        },
        t.span,
    );
    let align = Ident::new(
        match t.align {
            TooltipAlign::Start => "Start",
            TooltipAlign::Center => "Center",
            TooltipAlign::End => "End",
        },
        t.span,
    );
    let collision = Ident::new(
        match t.collision {
            TooltipCollision::None => "None",
            TooltipCollision::Flip => "Flip",
            TooltipCollision::Shift => "Shift",
            TooltipCollision::FlipShift => "FlipShift",
        },
        t.span,
    );
    let gap = &t.gap;
    let viewport_pad = &t.viewport_pad;
    let placement = if let Some(position) = &t.position {
        let ax = &position.anchor_x;
        let ay = &position.anchor_y;
        let cx = &position.content_x;
        let cy = &position.content_y;
        let ox = &position.offset_x;
        let oy = &position.offset_y;
        quote!(OverlayPlacement::points(
            OverlayPoint::new(#ax, #ay),
            OverlayPoint::new(#cx, #cy),
            Vector2::new(#ox, #oy),
        )
        .viewport_pad(#viewport_pad)
        .collision(OverlayCollision::#collision))
    } else {
        quote!(OverlayPlacement::default()
            .side(OverlaySide::#side)
            .align(OverlayAlign::#align)
            .gap(#gap)
            .viewport_pad(#viewport_pad)
            .collision(OverlayCollision::#collision))
    };
    let show_delay = &t.show_delay_ms;
    let hide_delay = &t.hide_delay_ms;
    let open = t
        .open
        .as_ref()
        .map(|open| quote!(__tooltip_options = __tooltip_options.open(#open);));
    let summary = &t.summary;

    let handlers = t
        .handlers
        .iter()
        .map(|handler| stmt_or_error(handler_stmt(&elem, handler)));
    let flags = t
        .flags
        .iter()
        .map(|flag| stmt_or_error(flag_stmt(&elem, flag)));
    let mut split = split_leaf_attrs(&elem, &t.styles, &t.attrs, &[]);
    split.route_states(&t.states, &[], FONT_ATTRS);
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let default_content = t
        .text
        .as_ref()
        .map(|text| quote!(style_text_tooltip_root(&#elem, #text);));
    let children = (!t.body.is_empty()).then(|| emit_block(quote!(&#elem), &t.body));
    let widget = Ident::new("tooltip", t.span);
    quote!({
        let mut __tooltip_options = TooltipOptions::default()
            .trigger(TooltipTrigger::#trigger)
            .show_delay(::core::time::Duration::from_secs_f32((#show_delay) / 1000.0_f32))
            .hide_delay(::core::time::Duration::from_secs_f32((#hide_delay) / 1000.0_f32))
            .placement_dyn(move || #placement);
        #open
        #widget(#parent, #summary, __tooltip_options, move |__tooltip_parent| {
            let #elem = __tooltip_parent.clone();
            #default_content
            #style_ctx
            #style_stmt
            #visual_stmt
            #style_element_stmts
            #element_stmts
            #(#flags)*
            #(#handlers)*
            #(#extras)*
            #children
        });
    })
}

fn emit_attached_tooltip(tooltip: Option<&TooltipNode>, anchor: &Ident) -> TokenStream2 {
    tooltip.map_or_else(TokenStream2::new, |tooltip| {
        let attachment = emit_tooltip(tooltip);
        let parent = parent_ident();
        quote!({ let #parent = &#anchor; #attachment })
    })
}

fn emit_icon(i: &IconNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&i.name);
    // `size`, `fill`, and `stroke` are the icon's own attributes; everything
    // else routes through the shared surface. They are passed as extra
    // vocabulary so unknown-attribute suggestions still mention them.
    const OWN: &[&str] = &["size", "fill", "stroke"];
    let mut size = None;
    let mut fills = Vec::<TokenStream2>::new();
    let mut stroke: Option<TokenStream2> = None;
    let mut errors: Vec<TokenStream2> = Vec::new();
    let mut split = SplitAttrs::default();
    for (name, value) in &i.attrs {
        match name.to_string().as_str() {
            "size" => size = Some(value.dimension()),
            "fill" => match value {
                Value::Grouped { values, .. } => {
                    fills.extend(values.iter().map(Value::tokens));
                }
                value => fills.push(value.tokens()),
            },
            "stroke" => match value {
                Value::Record { group, .. } | Value::RecordPlus { group, .. } => {
                    match mosaic_syntax::assemble_icon_stroke(group) {
                        Ok((value, _)) => stroke = Some(quote!(#value)),
                        Err(error) => errors.push(error.to_compile_error()),
                    }
                }
                // A bare `stroke:accent` is the color, as it reads.
                other => {
                    let value = other.tokens();
                    stroke = Some(quote!(resolve_icon_stroke(#value)));
                }
            },
            _ => split.route_valued(&elem, name, value, OWN),
        }
    }
    split.route_states(&i.states, &[], FONT_ATTRS);

    // A scheme token resolves to the one solid color a shape paints with,
    // rather than becoming a layered paint as a container's `fill:` would.
    let paint_field = |value: Option<TokenStream2>| match value {
        Some(expr) => quote!(::core::option::Option::Some(resolve_icon_color(#expr))),
        None => quote!(::core::option::Option::None),
    };
    let fill = fills.split_first().map(|(first, rest)| {
        if rest.is_empty() {
            quote!(#first)
        } else {
            quote!(PaintSpec::from(#first) #(.push(#rest))*)
        }
    });
    let fill_field = paint_field(fill);
    let stroke = stroke.unwrap_or_else(|| quote!(IconStroke::default()));
    let parts: Vec<TokenStream2> = i.parts.iter().map(emit_icon_part).collect();
    let style = quote!({
        let mut __icon_style = IconStyle::default();
        let __icon_stroke = #stroke;
        __icon_style.paints = IconPaints {
            fill: #fill_field,
            stroke: __icon_stroke.color,
            stroke_width: __icon_stroke.width,
        };
        #(#parts)*
        __icon_style
    });

    let build = match &i.token {
        Content::Static(expr) => {
            let widget = Ident::new("icon", i.span);
            quote!(let #elem = #widget(#parent, #expr, #style);)
        }
        Content::ReactiveExpr(expr) => {
            let widget = Ident::new("icon_dyn", i.span);
            quote!(let #elem = #widget(#parent, move || #expr, #style);)
        }
        Content::Reactive(block) => {
            let widget = Ident::new("icon_dyn", i.span);
            quote!(let #elem = #widget(#parent, move || #block, #style);)
        }
        // The parser rejects a literal operand, so this is unreachable in
        // practice; emit an error rather than a panic if it ever is not.
        Content::Lit(lit) => {
            syn::Error::new(lit.span(), "an `icon` takes a scheme token").to_compile_error()
        }
    };

    // `size:` is square by definition: one length, both axes.
    let size_stmt = size.map(|len| quote!(#elem.restyle(|__s| __s.width(#len).height(#len));));
    let handlers = i
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = i.flags.iter().map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let stamp = stamp_source(&elem, i.span);
    let expose = expose_stmt(&i.name, &elem);
    let theme = theme_scope_stmt(&i.theme, quote!(#elem));
    let fill_binding = fill_binding(&i.name, &elem);
    let tooltip = emit_attached_tooltip(i.tooltip.as_deref(), &elem);
    quote!(#build #fill_binding #stamp #size_stmt #style_ctx #style_stmt #visual_stmt #style_element_stmts #element_stmts #(#flags)* #(#handlers)* #(#extras)* #(#errors)* #expose #tooltip #theme)
}

/// Lowers one icon part run to a rule on the style.
///
/// A rule rather than a value: it is re-run whenever the theme moves or the
/// icon's interaction state changes, which is what makes a part's
/// `hover { … }` block work without the icon knowing anything about it.
fn emit_icon_part(part: &IconPartRun) -> TokenStream2 {
    let id = &part.id;
    let base = icon_paint_stmts(&part.attrs);
    // Same precedence as everywhere else: base < focused < hover < pressed
    // < disabled.
    let mut states: Vec<&StateBlock> = part.states.iter().collect();
    states.sort_by_key(|block| state_index(block.state));
    let states = states.into_iter().map(|block| {
        let cond = state_cond(state_index(block.state));
        let stmts = icon_paint_stmts(&block.attrs);
        quote!(if #cond { #(#stmts)* })
    });
    quote!(__icon_style = __icon_style.part_dyn(#id, move |__interaction| {
        let mut __paints = IconPaints::default();
        #(#base)*
        #(#states)*
        __paints
    });)
}

/// The assignments one run's attributes make to a part's paints.
///
/// A stroke writes only the halves it names, so `hover { stroke:(color:…) }`
/// keeps the width the run outside it set.
fn icon_paint_stmts(attrs: &[(Ident, Value)]) -> Vec<TokenStream2> {
    let mut stmts = Vec::new();
    for (name, value) in attrs {
        match name.to_string().as_str() {
            "fill" => {
                let expr = value.tokens();
                stmts.push(quote!(
                    __paints.fill =
                        ::core::option::Option::Some(resolve_icon_color(#expr));
                ));
            }
            "stroke" => {
                let stroke = match value {
                    Value::Record { group, .. } | Value::RecordPlus { group, .. } => {
                        match mosaic_syntax::assemble_icon_stroke(group) {
                            Ok((value, _)) => quote!(#value),
                            Err(error) => {
                                stmts.push(error.to_compile_error());
                                continue;
                            }
                        }
                    }
                    other => {
                        let value = other.tokens();
                        quote!(resolve_icon_stroke(#value))
                    }
                };
                stmts.push(quote!({
                    let __stroke = #stroke;
                    if __stroke.color.is_some() {
                        __paints.stroke = __stroke.color;
                    }
                    if __stroke.width.is_some() {
                        __paints.stroke_width = __stroke.width;
                    }
                }));
            }
            // The parser already rejected every other name, with the message
            // that explains why; repeating it here would only double it.
            _ => {}
        }
    }
    stmts
}

fn emit_img(i: &ImgNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&i.name);
    // `fit` is the widget's own attribute; everything else routes through the
    // shared style/visual/element surface. Passing "fit" as extra vocabulary
    // keeps it in the unknown-attribute suggestions.
    let mut fit = None;
    let mut split = SplitAttrs::default();
    for (name, value) in &i.attrs {
        if name == "fit" {
            fit = Some(resolve_theme_if_token(
                value,
                value.fit(),
                quote!(ObjectFit::Fill),
            ));
        } else {
            split.route_valued(&elem, name, value, &["fit"]);
        }
    }
    split.route_states(&i.states, &[], FONT_ATTRS);
    let style = match fit {
        // A block rather than `..Default::default()` update syntax: `fit` is
        // currently the style's only field, and clippy flags a needless
        // update in that case.
        Some(fit) => quote!({
            let mut __img_style = ImgStyle::default();
            __img_style.fit = #fit;
            __img_style
        }),
        None => quote!(ImgStyle::default()),
    };
    let build = match &i.content {
        Content::Lit(lit) => {
            let widget = Ident::new("img", i.span);
            let path = if std::path::Path::new(&lit.value()).is_absolute() {
                quote!(#lit)
            } else {
                quote!(::core::concat!(
                    ::core::env!("CARGO_MANIFEST_DIR"),
                    "/",
                    #lit
                ))
            };
            quote!(let #elem = #widget(
                #parent,
                ImageSource::__hot_embedded(#path, ::core::include_bytes!(#path)),
                #style,
            );)
        }
        Content::Static(expr) => {
            let widget = Ident::new("img", i.span);
            quote!(let #elem = #widget(#parent, #expr, #style);)
        }
        Content::ReactiveExpr(expr) => {
            let widget = Ident::new("img_dyn", i.span);
            quote!(let #elem = #widget(#parent, move || #expr, #style);)
        }
        Content::Reactive(block) => {
            let widget = Ident::new("img_dyn", i.span);
            quote!(let #elem = #widget(#parent, move || #block, #style);)
        }
    };
    let handlers = i
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = i.flags.iter().map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let stamp = stamp_source(&elem, i.span);
    let expose = expose_stmt(&i.name, &elem);
    let theme = theme_scope_stmt(&i.theme, quote!(#elem));
    let fill = fill_binding(&i.name, &elem);
    let tooltip = emit_attached_tooltip(i.tooltip.as_deref(), &elem);
    quote!(#build #fill #stamp #style_ctx #style_stmt #visual_stmt #style_element_stmts #element_stmts #(#flags)* #(#handlers)* #(#extras)* #expose #tooltip #theme)
}

/// Builds the `GeometrySpec` expression a shape node's geometry attributes
/// describe.
///
/// Which attributes are present is what picks the construction — the parser
/// has already rejected a mix of two — so this only has to read the set it is
/// given. A shape with no geometry attributes at all keeps the default box,
/// which is what makes `circle` inside a `hover { radius:… }` state work: the
/// state's own run supplies the outline.
fn shape_geometry_expr(node: &ShapeNode) -> TokenStream2 {
    let mut at = None;
    let mut size = None;
    let mut radius = None;
    let mut through: Vec<TokenStream2> = Vec::new();
    let mut arc = None;
    let mut from = None;
    let mut to = None;
    let mut angle = None;
    let mut length = None;
    let mut sides = None;
    let mut placement: Option<(String, TokenStream2)> = None;
    let mut corner = None;
    let mut rounding = None;
    for (name, value) in &node.attrs {
        let key = name.to_string();
        match key.as_str() {
            "at" => at = Some(shape_point_tokens(&key, value)),
            "size" => size = Some(attr_value_tokens(&key, value)),
            // Mirrored into the outline as well as painted by the visual, so
            // a diagonal anchor lands on the rounded corner rather than on the
            // box corner the rounding cut away.
            "radius" if matches!(node.tag, ShapeTag::Rect) => {
                corner = Some(attr_value_tokens(&key, value));
            }
            "radius" => radius = Some(shape_radius_tokens(value)),
            "through" => collect_points(value, &mut through),
            "arc" => arc = Some(attr_value_tokens(&key, value)),
            "from" => from = Some(shape_point_tokens(&key, value)),
            "to" => to = Some(shape_point_tokens(&key, value)),
            "angle" => angle = Some(attr_value_tokens(&key, value)),
            "length" => length = Some(attr_value_tokens(&key, value)),
            "sides" => sides = Some(attr_value_tokens(&key, value)),
            // The corner rounds whatever construction produced the points, so
            // it wraps the outline rather than taking part in building one.
            // A bare length is the radius; `Into` gives it the circular
            // exponent, which is the same conversion the record spelling
            // bypasses by naming both.
            "corner" => {
                let tokens = match value {
                    Value::Record { .. } => value.tokens(),
                    _ => value.length(),
                };
                rounding = Some(quote!(::core::convert::Into::<CornerSpec>::into(#tokens)));
            }
            other if mosaic_syntax::vocabulary::COMPASS_NAMES.contains(&other) => {
                placement = Some((key.clone(), shape_point_tokens(&key, value)));
            }
            _ => {}
        }
    }
    let at = at.unwrap_or(quote!(PointSpec::Box(BoxPoint::CENTER)));
    let arc = match arc {
        Some(arc) => quote!(Some(#arc)),
        None => quote!(None),
    };
    let inner = shape_geometry_inner(
        node, at, size, radius, corner, through, arc, from, to, angle, length, sides,
    );
    // Rounding sits under placement: a placed outline is moved as a whole, and
    // moving it cannot change the shape of its corners.
    let inner = match rounding {
        Some(corner) => quote!(GeometrySpec::cornered(&#inner, #corner)),
        None => inner,
    };
    match placement {
        Some((word, at)) => {
            let variant = Ident::new(&pascal_case(&word.replace('-', "_")), node.span);
            quote!(GeometrySpec::placed(&#inner, Anchor::#variant, #at))
        }
        None => inner,
    }
}

#[allow(clippy::too_many_arguments)]
fn shape_geometry_inner(
    node: &ShapeNode,
    at: TokenStream2,
    size: Option<TokenStream2>,
    radius: Option<TokenStream2>,
    corner: Option<TokenStream2>,
    through: Vec<TokenStream2>,
    arc: TokenStream2,
    from: Option<TokenStream2>,
    to: Option<TokenStream2>,
    angle: Option<TokenStream2>,
    length: Option<TokenStream2>,
    sides: Option<TokenStream2>,
) -> TokenStream2 {
    match node.tag {
        ShapeTag::Circle if !through.is_empty() => {
            quote!(GeometrySpec::circumcircle([#(#through),*], #arc))
        }
        ShapeTag::Circle => {
            let radius = radius.unwrap_or(quote!(Radius::Uniform(Length::percent(0.5))));
            quote!(GeometrySpec::ellipse(#at, #radius, #arc))
        }
        // Two opposite corners, folded into the centered spelling by the
        // constructor so both constructions produce one variant.
        ShapeTag::Rect if through.len() >= 2 => {
            let a = &through[0];
            let b = &through[1];
            quote!(GeometrySpec::rect_between(#a, #b))
        }
        ShapeTag::Rect => {
            let size = size.unwrap_or(quote!(BoxSize::new(
                Length::percent(1.0),
                Length::percent(1.0)
            )));
            let corner = corner.unwrap_or(quote!(Length::ZERO));
            quote!(GeometrySpec::rect_rounded(#at, #size, #corner))
        }
        ShapeTag::Line if !through.is_empty() => {
            quote!(GeometrySpec::path(::std::vec![#(#through),*], false))
        }
        ShapeTag::Line => {
            let from = from.unwrap_or(quote!(PointSpec::Box(BoxPoint::CENTER)));
            match (to, angle, length) {
                (Some(to), _, _) => {
                    quote!(GeometrySpec::path(::std::vec![#from, #to], false))
                }
                (None, angle, length) => {
                    let angle = angle.unwrap_or(quote!(0.0f32));
                    let length = length.unwrap_or(quote!(Length::px(0.0)));
                    quote!(GeometrySpec::ray(#from, #angle, #length))
                }
            }
        }
        ShapeTag::Polygon if !through.is_empty() => {
            quote!(GeometrySpec::path(::std::vec![#(#through),*], true))
        }
        ShapeTag::Polygon => {
            // A regular polygon has one reach, so a per-axis radius collapses
            // to its x — the same length either spelling names for a circle.
            let radius = radius
                .map(|radius| {
                    quote!(match #radius {
                        Radius::Uniform(length) => length,
                        Radius::Axes(x, _) => x,
                    })
                })
                .unwrap_or(quote!(Length::percent(0.5)));
            let sides = sides.unwrap_or(quote!(3u32));
            quote!(GeometrySpec::regular(#at, #radius, (#sides) as u32))
        }
    }
}

/// The sibling elements a line's `head:`/`tail:` markers become.
///
/// A marker is its own element because it is its own distance field: the line
/// is an open stroked path and the arrowhead is a filled polygon, and one
/// element paints one shape. It is a *sibling* rather than a child so it
/// resolves against the same box the line does — inside a canvas that is the
/// canvas, which is what puts the arrowhead exactly on the line's end.
/// How far each end of a line is pulled back by the marker sitting on it.
///
/// The markers themselves are built from the *untrimmed* line, so their
/// reference point stays the real endpoint — an arrowhead still points exactly
/// where the line ended, it just no longer has the line showing through its
/// tip.
fn marker_trim(node: &ShapeNode) -> Option<(TokenStream2, TokenStream2)> {
    let reach = |wanted: &str| {
        node.attrs.iter().find_map(|(name, value)| {
            if name != wanted {
                return None;
            }
            let Value::Marker(marker) = value else {
                return None;
            };
            let size = match &marker.size {
                Some(size) => {
                    let tokens = &size.tokens;
                    quote!(#tokens)
                }
                None => quote!(Length::px(MarkerSpec::DEFAULT_SIZE)),
            };
            Some(match marker.kind {
                mosaic_syntax::MarkerKind::Triangle => size,
                // A tick crosses the line, so it trims nothing.
                mosaic_syntax::MarkerKind::Bar => quote!(Length::ZERO),
                _ => quote!((#size) * 0.5),
            })
        })
    };
    let head = reach("head");
    let tail = reach("tail");
    if head.is_none() && tail.is_none() {
        return None;
    }
    Some((
        tail.unwrap_or(quote!(Length::ZERO)),
        head.unwrap_or(quote!(Length::ZERO)),
    ))
}

/// The geometry of each marker on a shape, paired with the fill it takes.
///
/// Shared by the elements the markers become and by the canvas that sizes
/// itself to the drawing: an arrowhead is ink like any other, so a canvas that
/// left it out would cut it off at the edge.
fn marker_geometries(node: &ShapeNode, line: &TokenStream2) -> Vec<TokenStream2> {
    node.attrs
        .iter()
        .filter_map(|(name, value)| {
            let end = match name.to_string().as_str() {
                "head" => quote!(MarkerEnd::Head),
                "tail" => quote!(MarkerEnd::Tail),
                _ => return None,
            };
            let Value::Marker(marker) = value else {
                return None;
            };
            let shape = match &marker.kind {
                mosaic_syntax::MarkerKind::Triangle => quote!(MarkerShape::Triangle),
                mosaic_syntax::MarkerKind::Circle => quote!(MarkerShape::Circle),
                mosaic_syntax::MarkerKind::Square => quote!(MarkerShape::Square),
                mosaic_syntax::MarkerKind::Bar => quote!(MarkerShape::Bar),
                mosaic_syntax::MarkerKind::Named(_) => match &marker.target {
                    Some(target) => {
                        let outline = shape_geometry_expr(target);
                        quote!(match (#outline).outline_arc() {
                            Some(outline) => MarkerShape::Custom(outline),
                            None => MarkerShape::Triangle,
                        })
                    }
                    None => return None,
                },
            };
            let size = match &marker.size {
                Some(size) => {
                    let tokens = &size.tokens;
                    quote!(#tokens)
                }
                None => quote!(Length::px(MarkerSpec::DEFAULT_SIZE)),
            };
            let rotate = marker.rotate;
            Some(quote!(GeometrySpec::marker(
                &#line,
                #end,
                MarkerSpec {
                    shape: #shape,
                    size: #size,
                    rotate: #rotate,
                }
            )))
        })
        .collect()
}

fn emit_markers(node: &ShapeNode, line: &TokenStream2) -> TokenStream2 {
    let parent = parent_ident();
    // The line's stroke colour is the marker's fill unless the marker says
    // otherwise, so `head:triangle` needs no styling of its own.
    let stroke_color = node
        .attrs
        .iter()
        .find(|(name, _)| name == "stroke")
        .and_then(|(_, value)| record_field_tokens(value, "color"));
    let markers = node.attrs.iter().filter_map(|(name, value)| {
        let end = match name.to_string().as_str() {
            "head" => quote!(MarkerEnd::Head),
            "tail" => quote!(MarkerEnd::Tail),
            _ => return None,
        };
        let Value::Marker(marker) = value else {
            return None;
        };
        let shape = match &marker.kind {
            mosaic_syntax::MarkerKind::Triangle => quote!(MarkerShape::Triangle),
            mosaic_syntax::MarkerKind::Circle => quote!(MarkerShape::Circle),
            mosaic_syntax::MarkerKind::Square => quote!(MarkerShape::Square),
            mosaic_syntax::MarkerKind::Bar => quote!(MarkerShape::Bar),
            mosaic_syntax::MarkerKind::Named(name) => match &marker.target {
                Some(target) => {
                    let outline = shape_geometry_expr(target);
                    quote!(match (#outline).outline_arc() {
                        Some(outline) => MarkerShape::Custom(outline),
                        None => MarkerShape::Triangle,
                    })
                }
                None => {
                    return Some(
                        syn::Error::new(
                            name.span(),
                            format!(
                                "no shape named `{name}` in this canvas — a custom marker names                                  a sibling shape's `as` binding"
                            ),
                        )
                        .to_compile_error(),
                    );
                }
            },
        };
        let size = match &marker.size {
            Some(size) => {
                let tokens = &size.tokens;
                quote!(#tokens)
            }
            None => quote!(Length::px(MarkerSpec::DEFAULT_SIZE)),
        };
        let rotate = marker.rotate;
        let fill = match (&marker.fill, &stroke_color) {
            (Some(fill), _) => {
                let tokens = fill.tokens();
                Some(quote!(.fill(#tokens)))
            }
            (None, Some(color)) => Some(quote!(.fill(#color))),
            (None, None) => None,
        };
        let geometry = quote!(GeometrySpec::marker(
            &#line,
            #end,
            MarkerSpec {
                shape: #shape,
                size: #size,
                rotate: #rotate,
            }
        ));
        Some(quote!({
            let __marker = shape_filled(#parent, move || #geometry);
            __marker.visual(Visual::new().geometry(#geometry) #fill);
        }))
    });
    quote!(#(#markers)*)
}

/// The tokens of one field inside a record value's group.
///
/// A crude scan rather than a parse: the record grammars live in
/// `mosaic-syntax` and their field types are private to it, while all this
/// needs is "the tokens the author wrote after `color:`". It stops at the next
/// `name:` pair at the top level, which is exactly where the field ends.
fn record_field_tokens(value: &Value, wanted: &str) -> Option<TokenStream2> {
    let (Value::Record { group, .. } | Value::RecordPlus { group, .. }) = value else {
        return None;
    };
    let trees: Vec<proc_macro2::TokenTree> = group.stream().into_iter().collect();
    let mut index = 0;
    while index < trees.len() {
        let proc_macro2::TokenTree::Ident(name) = &trees[index] else {
            index += 1;
            continue;
        };
        let is_colon = matches!(
            trees.get(index + 1),
            Some(proc_macro2::TokenTree::Punct(punct))
                if punct.as_char() == ':' && punct.spacing() == proc_macro2::Spacing::Alone
        );
        if !is_colon {
            index += 1;
            continue;
        }
        let start = index + 2;
        let mut end = start;
        while end < trees.len() {
            // `Color::WHITE` is not a new field: a path separator's first
            // colon is joint, a field's colon is alone.
            let next_is_field = matches!(&trees[end], proc_macro2::TokenTree::Ident(_))
                && matches!(
                    trees.get(end + 1),
                    Some(proc_macro2::TokenTree::Punct(punct))
                        if punct.as_char() == ':' && punct.spacing() == proc_macro2::Spacing::Alone
                );
            if next_is_field {
                break;
            }
            end += 1;
        }
        if name == wanted {
            return Some(trees[start..end].iter().cloned().collect());
        }
        index = end;
    }
    None
}

/// A shape's coordinate as the [`PointSpec`] the geometry takes.
///
/// An anchored point inlines the outline it names: the referenced shape's
/// geometry expression is emitted again, here, inside this one. That is what
/// makes the reference exact and order-independent — there is no handle to read
/// and nothing to be filled in later.
///
/// [`PointSpec`]: mosaic_render::PointSpec
fn shape_point_tokens(key: &str, value: &Value) -> TokenStream2 {
    let Value::Anchor(anchor) = value else {
        let point = attr_value_tokens(key, value);
        return quote!(PointSpec::Box(#point));
    };
    let Some(target) = &anchor.target else {
        // The resolver already reported why; emit something that type-checks so
        // the rest of the expansion still produces useful diagnostics.
        return quote!(PointSpec::Box(BoxPoint::CENTER));
    };
    let outline = shape_geometry_expr(target);
    let variant = Ident::new(
        &pascal_case(&anchor.anchor.to_string()),
        anchor.anchor.span(),
    );
    let offset = |offset: &Option<mosaic_syntax::AnchorOffset>| match offset {
        Some(offset) => {
            let tokens = &offset.tokens;
            if offset.negative {
                quote!((#tokens) * -1.0)
            } else {
                quote!(#tokens)
            }
        }
        None => quote!(Length::ZERO),
    };
    let dx = offset(&anchor.dx);
    let dy = offset(&anchor.dy);
    quote!(PointSpec::anchored(&#outline, Anchor::#variant, #dx, #dy))
}

/// `north_east` as `NorthEast` — the anchor word's variant.
fn pascal_case(name: &str) -> String {
    name.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// A shape's `radius:` as the [`Radius`] the geometry takes.
///
/// One length is uniform — a circle, and a relative one stays round in any box.
/// An `(x:… y:…)` record is per-axis, and it arrives here already lowered to a
/// `BoxPoint`, whose two lengths are exactly the pair wanted.
///
/// [`Radius`]: mosaic_render::Radius
fn shape_radius_tokens(value: &Value) -> TokenStream2 {
    match value {
        Value::Record { .. } => {
            let point = attr_value_tokens("radius", value);
            quote!({
                let __radius: BoxPoint = #point;
                Radius::Axes(__radius.x, __radius.y)
            })
        }
        _ => {
            let length = attr_value_tokens("radius", value);
            quote!(Radius::Uniform(#length))
        }
    }
}

/// Flattens a `through:` value — one point, or a group of them — into the
/// point list in source order.
fn collect_points(value: &Value, out: &mut Vec<TokenStream2>) {
    match value {
        Value::Grouped { values, .. } => {
            for value in values {
                collect_points(value, out);
            }
        }
        other => out.push(shape_point_tokens("through", other)),
    }
}

fn emit_shape(node: &ShapeNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&node.name);
    // Everything that is not geometry routes through the shared
    // style/visual/element surface, and passing the geometry names as extra
    // vocabulary keeps them in the unknown-attribute suggestions.
    let mut geometry_names = mosaic_syntax::shape_geometry_attrs(node.tag);
    // Markers build their own elements below, so they are geometry as far as
    // attribute routing is concerned.
    geometry_names.extend(["head", "tail"]);
    let geometry = shape_geometry_expr(node);

    let mut split = SplitAttrs::default();
    for (name, value) in &node.attrs {
        if geometry_names.contains(&name.to_string().as_str()) {
            split.visual_reactive |= value.is_reactive();
            continue;
        }
        split.route_valued(&elem, name, value, &geometry_names);
    }
    split.route_states(&node.states, &[], FONT_ATTRS);
    // The geometry rides the visual, so every other appearance attribute —
    // and every state block that changes one — composes with it the way it
    // composes with a radius. It is emitted through the same channel rather
    // than set on the element separately, or a `hover { … }` visual would
    // replace the outline with the default box.
    // The line the marker sits on is trimmed so the marker alone reaches the
    // end; the markers below are built from the untrimmed geometry, so their
    // reference point stays the real endpoint.
    let painted = match marker_trim(node) {
        Some((tail, head)) => quote!(GeometrySpec::trimmed(&#geometry, #tail, #head)),
        None => geometry.clone(),
    };
    split.visual_frags.insert(0, quote!(.geometry(#painted)));

    // A canvas child fills the canvas, so every shape in one resolves its
    // coordinates against the same box.
    let widget = Ident::new(
        if node.in_canvas {
            "shape_filled"
        } else {
            "shape"
        },
        node.span,
    );
    // Evaluated again here, on its own, because measuring reads it every
    // layout pass while the visual reads it every paint — one closure each,
    // rather than one shared value that could not follow state.
    let build = quote!(let #elem = #widget(#parent, move || #painted););

    let handlers = node
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = node
        .flags
        .iter()
        .map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.visual_stmt(&elem);
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let stamp = stamp_source(&elem, node.span);
    let expose = expose_stmt(&node.name, &elem);
    let theme = theme_scope_stmt(&node.theme, quote!(#elem));
    let fill = fill_binding(&node.name, &elem);
    let tooltip = emit_attached_tooltip(node.tooltip.as_deref(), &elem);
    // Children lay out in the outline's bounds rather than the element's box,
    // which for a canvas child is the whole canvas.
    let body = (!node.body.is_empty()).then(|| {
        let children = emit_block(quote!(&__shape_content), &node.body);
        quote!({
            let __shape_content = shape_content(&#elem, move || #geometry);
            #children
        })
    });
    // Markers come after the line so they paint over it.
    let markers = emit_markers(node, &geometry);
    quote!(#build #fill #stamp #style_ctx #style_stmt #visual_stmt #style_element_stmts #element_stmts #(#flags)* #(#handlers)* #(#extras)* #expose #tooltip #theme #body #markers)
}

fn emit_button(b: &ButtonNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&b.name);
    let click = match &b.click.value {
        Value::Reactive(block) => quote!(move || #block),
        Value::ReactiveExpr(expr) => quote!(move || #expr),
        Value::Static(expr) => quote!(#expr),
        Value::Inherit { .. } => {
            syn::Error::new(b.span, "handler `@click` is not inheritable").to_compile_error()
        }
        Value::Marker(_)
        | Value::Anchor(_)
        | Value::Size(_)
        | Value::Concat { .. }
        | Value::Conditional { .. }
        | Value::Length(_)
        | Value::ReactiveUnit(_)
        | Value::MaybeToken { .. }
        | Value::Record { .. }
        | Value::RecordPlus { .. }
        | Value::Filter { .. }
        | Value::Gradient { .. }
        | Value::Color { .. }
        | Value::Grouped { .. } => {
            syn::Error::new(b.span, "@click:{ … } must be a closure or expression")
                .to_compile_error()
        }
    };
    let click_stop = match stop_modifier(&b.click, true) {
        Ok(true) => quote!(
            #elem.on_pointer(move |__event, __ctx| {
                if matches!(__event.kind, PointerEventKind::Click(PointerButton::Primary)) {
                    __ctx.stop_propagation();
                }
            });
        ),
        Ok(false) => quote!(),
        Err(error) => error.to_compile_error(),
    };
    // A string / expression label builds through `button` (which renders the
    // label into the widget's own styled text child); children — or a reactive
    // label — build the bare `button_container` shell and fill it, the label
    // case adding a `text_dyn` in the button's own label style.
    let build = match &b.content {
        ButtonContent::Label(content) => match content.as_ref() {
            Content::Lit(lit) => {
                let widget = Ident::new("button", b.span);
                quote!(let #elem = #widget(#parent, #lit, #click);)
            }
            Content::Static(expr) => {
                let widget = Ident::new("button", b.span);
                quote!(let #elem = #widget(#parent, #expr, #click);)
            }
            Content::ReactiveExpr(expr) => {
                let widget = Ident::new("button_container", b.span);
                quote!(
                    let #elem = #widget(#parent, #click);
                    #elem.text_leaf_dyn(move || #expr, ButtonStyle::default().label);
                )
            }
            Content::Reactive(block) => {
                let widget = Ident::new("button_container", b.span);
                quote!(
                    let #elem = #widget(#parent, #click);
                    #elem.text_leaf_dyn(move || #block, ButtonStyle::default().label);
                )
            }
        },
        ButtonContent::Children { body, .. } => {
            let widget = Ident::new("button_container", b.span);
            let children = emit_block(quote!(&#elem), body);
            quote!(let #elem = #widget(#parent, #click); #children;)
        }
    };
    let handlers = b
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = b.flags.iter().map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let mut split = split_leaf_attrs(&elem, &b.styles, &b.attrs, &[]);
    split.route_states(&b.states, &[], FONT_ATTRS);
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.widget_visual_stmt(
        &elem,
        "ButtonStyle",
        quote!(__widget_style.base_visual()),
        quote!(__widget_style.apply_visual_state(
            __visual,
            __interaction.hovered(),
            __interaction.pressed(),
            __interaction.focus_visible(),
        )),
    );
    let declared_parts: Vec<String> = Vec::new();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let stamp = stamp_source(&elem, b.span);
    let expose = expose_stmt(&b.name, &elem);
    let theme = theme_scope_stmt(&b.theme, quote!(#elem));
    let fill = fill_binding(&b.name, &elem);
    let tooltip = emit_attached_tooltip(b.tooltip.as_deref(), &elem);
    quote!(
        #build
        #fill
        #click_stop
        #stamp
        #style_ctx
        #style_stmt
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#flags)*
        #(#handlers)*
        #(#extras)*
        #expose
        #tooltip
        #theme
    )
}

fn emit_input(i: &InputNode) -> TokenStream2 {
    let parent = parent_ident();
    let elem = binding_ident(&i.name);
    let state = &i.state;
    let handlers = i
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&elem, h)));
    let flags = i.flags.iter().map(|f| stmt_or_error(flag_stmt(&elem, f)));
    let placeholder = i
        .attrs
        .iter()
        .find(|(name, _)| name == "placeholder")
        .map(|(_, value)| attr_value_tokens("placeholder", value));
    let mut split = SplitAttrs {
        styles: lower_styles(&i.styles),
        ..SplitAttrs::default()
    };
    for (name, value) in &i.attrs {
        if name != "placeholder" {
            split.route_valued(&elem, name, value, &["placeholder"]);
        }
    }
    split.route_states(&i.states, &[], FONT_ATTRS);
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = split.widget_visual_stmt(
        &elem,
        "TextInputStyle",
        quote!(__widget_style.base_visual()),
        quote!(__widget_style.apply_visual_state(__visual, __interaction.focus_visible(),)),
    );
    // A field registers its parts on the element itself, so a named style
    // resolves them by name the way it does a component's — every part except
    // the ones this node styles, which seed their own bindings below.
    let declared_parts: Vec<String> = i.parts.iter().map(|part| part.name.to_string()).collect();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let parts = i.parts.iter().map(|part| emit_input_part(&elem, i, part));
    let widget = Ident::new(
        if i.multiline {
            "text_area"
        } else {
            "text_input"
        },
        i.span,
    );
    let hidden = i.hidden.then(|| quote!(.hidden(true)));
    let placeholder = placeholder.map(|value| quote!(.placeholder(#value)));
    let build = if hidden.is_some() || placeholder.is_some() {
        let widget = Ident::new(
            if i.multiline {
                "text_area_with_options"
            } else {
                "text_input_with_options"
            },
            i.span,
        );
        quote!(let #elem = #widget(
            #parent,
            #state,
            TextInputOptions::default() #hidden #placeholder,
        );)
    } else {
        quote!(let #elem = #widget(#parent, #state);)
    };
    let stamp = stamp_source(&elem, i.span);
    let expose = expose_stmt(&i.name, &elem);
    let theme = theme_scope_stmt(&i.theme, quote!(#elem));
    let fill = fill_binding(&i.name, &elem);
    let tooltip = emit_attached_tooltip(i.tooltip.as_deref(), &elem);
    quote!(
        #build
        #fill
        #stamp
        #style_ctx
        #style_stmt
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#flags)*
        #(#handlers)*
        #(#extras)*
        #(#parts)*
        #expose
        #tooltip
        #theme
    )
}

/// Styling for one exposed part of an `input`.
///
/// The field puts its parts in the element part registry rather than on a
/// handle, so the part is fetched by name — a name the parser already checked,
/// which is why the miss is unreachable rather than diagnosed.
///
/// The `placeholder` part is a stand-in for a glyph run and takes no pointer
/// input, so an interaction state block on it could never fire; it is refused
/// rather than silently ignored.
fn emit_input_part(root: &Ident, i: &InputNode, part: &PartRun) -> TokenStream2 {
    let elem = Ident::new("__input_part", Span::mixed_site());
    let mut split = split_leaf_attrs(&elem, &i.styles, &part.attrs, &[]);
    split.part = Some(part.name.to_string());
    for state in &part.states {
        split.extras.push(
            syn::Error::new(
                state.span,
                format!(
                    "`{}` takes no interaction states — it is hint text, not a target for the \
                     pointer. Put the state block on the `input` itself; the placeholder follows \
                     the field's text color.",
                    part.name
                ),
            )
            .to_compile_error(),
        );
    }
    let style_ctx = split.style_part_ctx_stmt(&elem);
    let style = split.restyle_stmt(&elem);
    let visual = split.revisual_stmt(&elem);
    let style_element = split.style_element_stmts(&elem, &[]);
    let element = split.element_stmts(&elem);
    let extras = &split.extras;
    let name = part.name.to_string();
    quote!({
        if let Some(#elem) = #root.part(#name) {
            #style_ctx
            #style #visual #style_element #element
        }
        #(#extras)*
    })
}

fn emit_state_widget(widget: &StateWidgetNode) -> TokenStream2 {
    let parent = parent_ident();
    let handle = binding_ident(&widget.name);
    let elem = Ident::new("__widget_root", Span::mixed_site());
    let state = widget_state(widget);
    let own_names: &[&str] = match widget.tag {
        WidgetTag::Checkbox => &["label"],
        WidgetTag::Radio => &["value", "label"],
        WidgetTag::Slider | WidgetTag::Stepper => &["min", "max", "step"],
        WidgetTag::Select => &["options"],
        WidgetTag::Toggle | WidgetTag::Progress | WidgetTag::Find => &[],
    };
    let own = |name: &str| {
        widget
            .attrs
            .iter()
            .find(|(attr, _)| attr == name)
            .map(|(_, value)| attr_value_tokens(name, value))
    };
    let own_f32 = |name: &str| {
        widget
            .attrs
            .iter()
            .find(|(attr, _)| attr == name)
            .map(|(_, value)| value.f32())
    };
    let build = match widget.tag {
        WidgetTag::Toggle => {
            let call = Ident::new("toggle", widget.span);
            quote!(let #handle = #call(#parent, #state);)
        }
        WidgetTag::Checkbox => match own("label") {
            Some(label) => {
                let call = Ident::new("checkbox_labeled", widget.span);
                quote!(let #handle = #call(#parent, #state, #label);)
            }
            None => {
                let call = Ident::new("checkbox", widget.span);
                quote!(let #handle = #call(#parent, #state);)
            }
        },
        WidgetTag::Radio => {
            let Some(value) = own("value") else {
                return syn::Error::new(widget.span, "radio needs a `value:(expr)` attribute")
                    .to_compile_error();
            };
            match own("label") {
                Some(label) => {
                    let call = Ident::new("radio_labeled", widget.span);
                    quote!(
                        let __radio_value = #value;
                        let #handle = #call(#parent, #state, __radio_value.clone(), #label);
                    )
                }
                None => {
                    let call = Ident::new("radio", widget.span);
                    quote!(
                        let __radio_value = #value;
                        let #handle = #call(#parent, #state, __radio_value.clone());
                    )
                }
            }
        }
        WidgetTag::Slider => {
            let min = own_f32("min").unwrap_or_else(|| quote!(0.0f32));
            let max = own_f32("max").unwrap_or_else(|| quote!(1.0f32));
            let step = own_f32("step").map_or_else(|| quote!(None), |value| quote!(Some(#value)));
            let call = Ident::new("slider_styled", widget.span);
            quote!(let #handle = #call(#parent, #state, (#min)..=(#max), #step, SliderStyle::default());)
        }
        WidgetTag::Progress => {
            if matches!(widget.state, Some(Expr::Block(_))) {
                let call = Ident::new("progress_dyn", widget.span);
                quote!(let #handle = #call(#parent, move || #state);)
            } else {
                let call = Ident::new("progress", widget.span);
                quote!(let #handle = #call(#parent, #state);)
            }
        }
        WidgetTag::Stepper => {
            let min = own_f32("min").unwrap_or_else(|| quote!(0.0f32));
            let max = own_f32("max").unwrap_or_else(|| quote!(10.0f32));
            let step = own_f32("step").unwrap_or_else(|| quote!(1.0f32));
            let call = Ident::new("stepper_styled", widget.span);
            quote!(let #handle = #call(#parent, #state, (#min)..=(#max), #step, StepperStyle::default());)
        }
        WidgetTag::Select => {
            let Some(options) = own("options") else {
                return syn::Error::new(widget.span, "select needs an `options:(expr)` attribute")
                    .to_compile_error();
            };
            let call = Ident::new("select", widget.span);
            quote!(let #handle = #call(#parent, #state, (#options).into_iter().map(Into::into).collect());)
        }
        WidgetTag::Find => {
            let call = Ident::new("find_bar", widget.span);
            quote!(let #handle = #call(#parent);)
        }
    };

    let widget_states = widget_state_locals(widget, &handle);
    let expose_states = widget_state_flags(widget, &elem);
    let mut split = SplitAttrs {
        styles: lower_styles(&widget.styles),
        ..SplitAttrs::default()
    };
    for (name, value) in &widget.attrs {
        if !own_names.contains(&name.to_string().as_str()) {
            split.route_valued(&elem, name, value, own_names);
        }
    }
    split.route_states(&widget.states, &[], FONT_ATTRS);
    for state_block in disabled_last(&widget.tag_states) {
        let word = state_block.keyword.to_string();
        if word == "disabled" && state_block.attrs.iter().any(|(name, _)| name == "opacity") {
            split
                .extras
                .push(quote!(#elem.authored_disabled_opacity();));
        }
        let condition = match word.as_str() {
            "hover" => quote!(__interaction.hovered()),
            "pressed" => quote!(__interaction.pressed()),
            "focused" => quote!(__interaction.focus_visible()),
            "disabled" => quote!(__interaction.disabled()),
            "on" | "checked" => quote!((#state).get()),
            // A select's `selected` belongs to a menu row, not to the widget
            // as a whole — the trigger's state is `open`.
            "selected" if widget.tag == WidgetTag::Select => {
                if !state_block.attrs.is_empty() {
                    split.extras.push(
                        syn::Error::new(
                            state_block.keyword.span(),
                            "`selected` styles a select's `item`, so its block holds an \
                             `item … ` run rather than attributes of its own",
                        )
                        .to_compile_error(),
                    );
                }
                quote!(false)
            }
            "selected" => quote!((#state).get() == __radio_value),
            "open" => quote!(__widget_open.get()),
            "empty" => quote!(__widget_empty()),
            _ => quote!(false),
        };
        split.route_conditional(
            state_block.attrs.iter().map(|(n, v)| (n, v)),
            condition,
            &[],
            FONT_ATTRS,
        );
    }
    let style_stmt = split.restyle_stmt(&elem);
    let visual_stmt = match widget.tag {
        WidgetTag::Toggle => split.widget_visual_stmt(
            &elem,
            "ToggleStyle",
            quote!(__widget_style.track_base_visual()),
            quote!(__widget_style.apply_track_state(
                __visual,
                (#state).get(),
                __interaction.hovered(),
                __interaction.focus_visible(),
            )),
        ),
        WidgetTag::Stepper => split.widget_visual_stmt(
            &elem,
            "StepperStyle",
            quote!(__widget_style.base_visual()),
            quote!(__widget_style.apply_visual_state(__visual, __interaction.focus_visible(),)),
        ),
        WidgetTag::Select => split.widget_visual_stmt(
            &elem,
            "SelectStyle",
            quote!(__widget_style.trigger_base_visual()),
            quote!(__widget_style.apply_trigger_state(
                __visual,
                __widget_open.get(),
                __interaction.hovered(),
                __interaction.pressed(),
                __interaction.focus_visible(),
            )),
        ),
        WidgetTag::Find => split.widget_visual_stmt(
            &elem,
            "FindBarStyle",
            quote!(__widget_style.bar_base_visual()),
            quote!(__visual),
        ),
        _ => split.revisual_stmt(&elem),
    };
    let declared_parts: Vec<String> = widget
        .tag
        .parts()
        .iter()
        .map(|name| name.to_string())
        .collect();
    let style_ctx = split.style_ctx_stmt(&elem);
    let style_element_stmts = split.style_element_stmts(&elem, &declared_parts);
    let element_stmts = split.element_stmts(&elem);
    let extras = &split.extras;
    let flags = widget
        .flags
        .iter()
        .map(|flag| stmt_or_error(flag_stmt(&elem, flag)));
    let handlers = widget
        .handlers
        .iter()
        .map(|handler| stmt_or_error(handler_stmt(&elem, handler)));
    // A widget's parts hang off its handle's accessors rather than the
    // element part registry, so a named style cannot find them at run time
    // the way it finds a component's. Every part the widget has but the node
    // does not style itself gets a synthesized run instead, guarded by
    // whether a style actually writes it.
    let style_only_parts: Vec<PartRun> = if widget.styles.is_empty() {
        Vec::new()
    } else {
        widget
            .tag
            .parts()
            .iter()
            .filter(|name| !widget.parts.iter().any(|part| part.name == **name))
            .map(|name| PartRun {
                name: Ident::new(name, widget.span),
                span: widget.span,
                attrs: Vec::new(),
                states: Vec::new(),
            })
            .collect()
    };
    let parts = widget
        .parts
        .iter()
        .chain(style_only_parts.iter())
        .map(|part| emit_widget_part(&handle, widget, part));
    let stamp = stamp_source(&elem, widget.span);
    let theme = theme_scope_stmt(&widget.theme, quote!(#elem));
    let fill = fill_binding(&widget.name, &handle);
    let tooltip = emit_attached_tooltip(widget.tooltip.as_deref(), &elem);
    quote!(
        #build
        #fill
        #widget_states
        let #elem = #handle.root().clone();
        #stamp
        #expose_states
        #style_ctx
        #style_stmt
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#flags)*
        #(#handlers)*
        #(#extras)*
        #(#parts)*
        #tooltip
        #theme
    )
}

/// The widget's controlled state, for the arms that only run on tags that have
/// one. A stateless tag never reaches them — the parser rejects the keywords
/// that read it — so the placeholder is never evaluated.
fn widget_state(widget: &StateWidgetNode) -> TokenStream2 {
    widget
        .state
        .as_ref()
        .map_or_else(|| quote!(()), |state| quote!(#state))
}

/// The locals a widget's own state conditions read, bound once beneath the
/// build so every style and visual binding captures cheap `Copy` handles
/// rather than the widget handle itself.
fn widget_state_locals(widget: &StateWidgetNode, handle: &Ident) -> TokenStream2 {
    match widget.tag {
        WidgetTag::Select => quote!(let __widget_open = #handle.open();),
        WidgetTag::Find => quote!(
            let __widget_open = #handle.open();
            let __widget_query = #handle.query();
            let __widget_matches = #handle.matches();
            // Typed, and nothing came back — the bar's one piece of news.
            let __widget_empty =
                move || !__widget_query.get().is_empty() && __widget_matches.get() == 0;
        ),
        _ => quote!(),
    }
}

fn disabled_last(
    blocks: &[mosaic_syntax::WidgetStateBlock],
) -> Vec<&mosaic_syntax::WidgetStateBlock> {
    blocks
        .iter()
        .filter(|block| block.keyword != "disabled")
        .chain(blocks.iter().filter(|block| block.keyword == "disabled"))
        .collect()
}

/// Registers the widget's own states as style conditions, so a `style!`
/// block scoped to `on`/`checked`/`selected`/`open`/`empty` resolves against
/// it.
///
/// Only a node that actually names a style pays for this: the registration
/// exists for `StyleCtx::active`, and nothing else reads it.
fn widget_state_flags(widget: &StateWidgetNode, elem: &Ident) -> TokenStream2 {
    if widget.styles.is_empty() {
        return quote!();
    }
    let state = widget_state(widget);
    let flags: &[(&str, &str)] = match widget.tag {
        WidgetTag::Toggle | WidgetTag::Checkbox => &[("on", "state"), ("checked", "state")],
        WidgetTag::Radio => &[("selected", "radio")],
        WidgetTag::Select => &[("open", "open")],
        WidgetTag::Find => &[("open", "open"), ("empty", "empty")],
        // The remaining widgets carry a value, not a condition.
        WidgetTag::Slider | WidgetTag::Progress | WidgetTag::Stepper => return quote!(),
    };
    let bind = widget
        .state
        .is_some()
        .then(|| quote!(let __style_state = #state;));
    let registrations = flags.iter().map(|(name, source)| {
        let flag = match *source {
            "radio" => quote!(move || __style_state.get() == __radio_value.clone()),
            "open" => quote!(move || __widget_open.get()),
            "empty" => quote!(move || __widget_empty()),
            _ => quote!(move || __style_state.get()),
        };
        quote!(#elem.expose_state_flag(#name, #flag);)
    });
    quote!(#bind #(#registrations)*)
}

fn emit_widget_part(handle: &Ident, widget: &StateWidgetNode, part: &PartRun) -> TokenStream2 {
    let tag = widget.tag;
    let widget_state = widget_state(widget);
    let name = part.name.to_string();
    let elem = Ident::new("__widget_part", Span::mixed_site());
    let mut split = split_leaf_attrs(&elem, &widget.styles, &part.attrs, &[]);
    split.part = Some(part.name.to_string());
    split.route_states(&part.states, &[], FONT_ATTRS);
    for state_block in disabled_last(&widget.tag_states) {
        for state_part in &state_block.parts {
            if state_part.name != part.name {
                continue;
            }
            let word = state_block.keyword.to_string();
            let state = &widget_state;
            let condition = match word.as_str() {
                "hover" => quote!(__root_interaction.hovered()),
                "pressed" => quote!(__root_interaction.pressed()),
                "focused" => quote!(__root_interaction.focus_visible()),
                "disabled" => quote!(__root_interaction.disabled()),
                "on" | "checked" => quote!((#state).get()),
                // A select's `selected` names the menu row holding the current
                // value; a radio's names the option holding the group.
                "selected" if tag == WidgetTag::Select => {
                    if part.name == "item" {
                        quote!((#state).get() == __item_index)
                    } else {
                        split.extras.push(
                            syn::Error::new(
                                state_part.name.span(),
                                "`selected` styles a select's `item` — the row holding the \
                                 current value",
                            )
                            .to_compile_error(),
                        );
                        quote!(false)
                    }
                }
                "selected" => quote!((#state).get() == __radio_value_part),
                "open" => quote!(__widget_open.get()),
                "empty" => quote!(__widget_empty()),
                _ => quote!(false),
            };
            split.route_conditional(
                state_part.attrs.iter().map(|(n, v)| (n, v)),
                condition,
                &[],
                FONT_ATTRS,
            );
        }
    }
    let style_ctx = split.style_part_ctx_stmt(&elem);
    let scoped_style_state =
        (!widget.styles.is_empty() && tag == WidgetTag::Select && name == "item").then(|| {
            let cx = style_ctx_ident();
            let state = &widget.state;
            quote!(
                let __style_selected = #state;
                let #cx = #cx.with_state_override(
                    "selected",
                    move || __style_selected.get() == __item_index,
                );
            )
        });
    let style = split.restyle_stmt(&elem);
    let visual = match (tag, name.as_str()) {
        (WidgetTag::Toggle, "track") => split.widget_visual_stmt(
            &elem,
            "ToggleStyle",
            quote!(__widget_style.track_base_visual()),
            quote!(__widget_style.apply_track_state(
                __visual,
                (#widget_state).get(),
                __root_interaction.hovered(),
                __root_interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Checkbox, "box") => split.widget_visual_stmt(
            &elem,
            "CheckboxStyle",
            quote!(__widget_style.indicator_base_visual()),
            quote!(__widget_style.apply_indicator_state(
                __visual,
                (#widget_state).get(),
                __root_interaction.hovered(),
                __root_interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Radio, "ring") => split.widget_visual_stmt(
            &elem,
            "RadioStyle",
            quote!(__widget_style.indicator_base_visual()),
            quote!(__widget_style.apply_indicator_state(
                __visual,
                (#widget_state).get() == __radio_value_part,
                __root_interaction.hovered(),
                __root_interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Radio, "dot") => split.widget_visual_stmt(
            &elem,
            "RadioStyle",
            quote!(__widget_style.dot_visual()),
            quote!(__visual),
        ),
        (WidgetTag::Slider, "thumb") => split.widget_visual_stmt(
            &elem,
            "SliderStyle",
            quote!(__widget_style.thumb_base_visual()),
            quote!(__widget_style.apply_thumb_state(
                __visual,
                __root_interaction.hovered(),
                __root_interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Stepper, "decrement" | "increment") => split.widget_visual_stmt(
            &elem,
            "ButtonStyle",
            quote!(__widget_style.base_visual()),
            quote!(__widget_style.apply_visual_state(
                __visual,
                __interaction.hovered(),
                __interaction.pressed(),
                __interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Select, "trigger") => split.widget_visual_stmt(
            &elem,
            "SelectStyle",
            quote!(__widget_style.trigger_base_visual()),
            quote!(__widget_style.apply_trigger_state(
                __visual,
                __widget_open.get(),
                __root_interaction.hovered(),
                __root_interaction.pressed(),
                __root_interaction.focus_visible(),
            )),
        ),
        (WidgetTag::Select, "item") => split.widget_visual_stmt(
            &elem,
            "SelectStyle",
            quote!(__widget_style.item_base_visual()),
            quote!(__widget_style.apply_item_state(
                __visual,
                __interaction.hovered(),
                __widget_highlighted.get() == __item_index,
            )),
        ),
        (WidgetTag::Find, "bar") => split.widget_visual_stmt(
            &elem,
            "FindBarStyle",
            quote!(__widget_style.bar_base_visual()),
            quote!(__visual),
        ),
        (WidgetTag::Find, "prev" | "next" | "close") => split.widget_visual_stmt(
            &elem,
            "ButtonStyle",
            quote!(__widget_style.base_visual()),
            quote!(__widget_style.apply_visual_state(
                __visual,
                __interaction.hovered(),
                __interaction.pressed(),
                __interaction.focus_visible(),
            )),
        ),
        _ => split.revisual_stmt(&elem),
    };
    let style_element = split.style_element_stmts(&elem, &[]);
    let element = split.element_stmts(&elem);
    let extras = &split.extras;
    let accessor = match (tag, name.as_str()) {
        (WidgetTag::Toggle, "track")
        | (WidgetTag::Progress, "track")
        | (WidgetTag::Select, "trigger")
        | (WidgetTag::Find, "bar") => Ident::new("root", part.span),
        (WidgetTag::Checkbox, "box") | (WidgetTag::Radio, "ring") => {
            Ident::new("indicator", part.span)
        }
        (_, other) => Ident::new(other, part.span),
    };
    let radio_value =
        (tag == WidgetTag::Radio).then(|| quote!(let __radio_value_part = __radio_value.clone();));
    let select_highlighted = (tag == WidgetTag::Select && name == "item")
        .then(|| quote!(let __widget_highlighted = #handle.highlighted();));
    let declares = part
        .attrs
        .is_empty()
        .then(|| split.style_declares_part(&name))
        .flatten();
    let apply = quote!(
        #radio_value
        #select_highlighted
        let __root_interaction = #handle.root().interaction();
        #style_ctx
        #scoped_style_state
        #style #visual #style_element #element #(#extras)*
    );
    let apply = match declares {
        Some(condition) => quote!(if #condition { #apply }),
        None => apply,
    };
    if tag == WidgetTag::Select && matches!(name.as_str(), "menu" | "item") {
        let setter = if name == "menu" {
            Ident::new("set_menu_style", part.span)
        } else {
            Ident::new("set_item_style", part.span)
        };
        // The item styler is handed the row's index, so `selected` can ask
        // whether this row is the one the select's state points at.
        let params = if name == "item" {
            quote!(|__part, __item_index: usize|)
        } else {
            quote!(|__part|)
        };
        let capture = split.capture_ctx();
        return quote!({
            let __select_handle = #handle.clone();
            #capture
            #handle.#setter(move #params {
                let #handle = __select_handle.clone();
                let #elem = __part.clone();
                #apply
            });
        });
    }
    if matches!(
        (tag, name.as_str()),
        (WidgetTag::Checkbox | WidgetTag::Radio, "label") | (WidgetTag::Select, "menu")
    ) {
        quote!(if let Some(#elem) = #handle.#accessor().cloned() { #apply })
    } else {
        quote!({ let #elem = #handle.#accessor().clone(); #apply })
    }
}

/// A component invocation: `Name(parent, NameProps::builder().attr(v)….build())`,
/// every named attribute chained onto the builder — a prop's own setter when
/// the component declares one by that name, else the builder's delegating
/// style/visual/element setter — with a `children` closure when a `{ … }`
/// block is present, and handlers / state blocks / `as` applied to the
/// returned element.
fn emit_component(c: &ComponentNode) -> TokenStream2 {
    let parent = parent_ident();
    let (build, _, root) = component_build(c);
    // A component builds a detached element; the structural parent adopts it.
    let theme = theme_scope_stmt(&c.theme, quote!(#root));
    quote!(#build #parent.adopt(&#root); #theme)
}

/// The condition tokens a component state block folds under: the root's
/// interaction for `hover`/`pressed`/`focused`/`disabled` (through
/// `interaction`, the
/// caller-chosen binding ident), else a read of the exposed state's
/// pre-bound `ReadState` local.
fn component_condition(
    keyword: &Ident,
    interaction: &TokenStream2,
    state_locals: &[(String, Ident)],
) -> TokenStream2 {
    let word = keyword.to_string();
    match word.as_str() {
        "hover" => quote!(#interaction.hovered()),
        "pressed" => quote!(#interaction.pressed()),
        "focused" => quote!(#interaction.focus_visible()),
        "disabled" => quote!(#interaction.disabled()),
        _ => {
            let local = state_locals
                .iter()
                .find(|(name, _)| *name == word)
                .map(|(_, local)| local)
                .expect("a local is bound for every non-interaction state block");
            quote!(#local.get())
        }
    }
}

/// The build + attribute/handler/state application for a component node, minus
/// the parent adoption — shared by [`emit_component`] (adopts into the parent)
/// and the header-less top-node path (returns the root). Returns the
/// statements, the ident holding the component's return value (an `Element`
/// or a generated handle — what `as` binds), and the ident holding its root
/// `Element` (what structural statements target).
fn component_build(c: &ComponentNode) -> (TokenStream2, Ident, Ident) {
    let elem = binding_ident(&c.name);
    let fill = fill_binding(&c.name, &elem);
    let root = Ident::new("__component_root", Span::mixed_site());
    let func = &c.func;
    let props_ty = Ident::new(&format!("{func}Props"), func.span());

    let mut chain = Vec::new();
    let mut visual_layers = VisualLayerRuns::default();
    for (name, value) in &c.props {
        if matches!(
            name.to_string().as_str(),
            "draggable" | "resizable" | "reorderable"
        ) {
            continue;
        }
        let key = name.to_string();
        let v = match value {
            Value::Reactive(block) => quote!(Derived::new(move || #block)),
            Value::ReactiveExpr(expr) => quote!(Derived::new(move || #expr)),
            _ => attr_value_tokens(&key, value),
        };
        let method = if visual_layers.component_concat(&key, value.is_concat()) {
            Ident::new(&format!("{name}_concat"), name.span())
        } else {
            name.clone()
        };
        chain.push(quote!(.#method(#v)));
    }
    // A braces block holding only part runs sets no children — the closure
    // would demand a `children` slot the component may not have.
    if let Some(body) = &c.children
        && !body.is_empty()
    {
        let cp = Ident::new("__children_parent", Span::mixed_site());
        let block = emit_block(quote!(#cp), body);
        chain.push(quote!(.children(move |#cp: &Element| {
            #cp.inspection_authored_children(|| #block)
        })));
    }

    // Exposed-state blocks read through `Copy` `ReadState` locals bound once
    // up front, so the reactive fold closures capture those instead of the
    // handle itself. The accessor call carries the keyword's span: a typo
    // becomes "no method" right at the state block.
    let mut state_locals: Vec<(String, Ident)> = Vec::new();
    let mut state_lets = Vec::new();
    for block in disabled_last(&c.tag_states) {
        let word = block.keyword.to_string();
        if matches!(word.as_str(), "hover" | "pressed" | "focused" | "disabled")
            || state_locals.iter().any(|(name, _)| *name == word)
        {
            continue;
        }
        let local = Ident::new(&format!("__component_state_{word}"), Span::mixed_site());
        let accessor = &block.keyword;
        state_lets.push(quote!(let #local = #elem.#accessor();));
        state_locals.push((word, local));
    }

    let handlers = c
        .handlers
        .iter()
        .map(|h| stmt_or_error(handler_stmt(&root, h)));
    let flags = c.flags.iter().map(|f| stmt_or_error(flag_stmt(&root, f)));
    let behaviors = c
        .props
        .iter()
        .filter(|(name, _)| {
            matches!(
                name.to_string().as_str(),
                "draggable" | "resizable" | "reorderable"
            )
        })
        .map(|(name, value)| stmt_or_error(lifecycle_stmt(&root, name, value)));
    let mut split = SplitAttrs {
        styles: lower_styles(&c.styles),
        ..SplitAttrs::default()
    };
    split.route_states(&c.states, &[], FONT_ATTRS);
    for block in disabled_last(&c.tag_states) {
        if block.keyword == "disabled" && block.attrs.iter().any(|(name, _)| name == "opacity") {
            split
                .extras
                .push(quote!(#root.authored_disabled_opacity();));
        }
        let condition = component_condition(&block.keyword, &quote!(__interaction), &state_locals);
        split.route_conditional(
            block.attrs.iter().map(|(n, v)| (n, v)),
            condition,
            &[],
            FONT_ATTRS,
        );
    }
    let style_stmt = split.restyle_stmt(&root);
    let visual_stmt = split.patch_visual_stmt(&root);
    let declared_parts: Vec<String> = c.parts.iter().map(|part| part.name.to_string()).collect();
    let style_ctx = split.style_ctx_stmt(&root);
    let style_element_stmts = split.style_element_stmts(&root, &declared_parts);
    let element_stmts = split.element_stmts(&root);
    let extras = &split.extras;

    // A part mentioned only inside a state block still gets a (bare) run, so
    // its conditional styling is not silently dropped.
    let mut state_only_parts: Vec<PartRun> = Vec::new();
    for block in disabled_last(&c.tag_states) {
        for state_part in &block.parts {
            if !c.parts.iter().any(|part| part.name == state_part.name)
                && !state_only_parts
                    .iter()
                    .any(|part| part.name == state_part.name)
            {
                state_only_parts.push(PartRun {
                    name: state_part.name.clone(),
                    span: state_part.span,
                    attrs: Vec::new(),
                    states: Vec::new(),
                });
            }
        }
    }
    let parts = c
        .parts
        .iter()
        .chain(state_only_parts.iter())
        .map(|part| emit_component_part(&elem, &root, c, part, &state_locals));
    let tooltip = emit_attached_tooltip(c.tooltip.as_deref(), &root);

    // The component body already stamps the root node it builds. Restamping it
    // here would attach the component tag's authored span to `debug_source`
    // and the location macros, making rust-analyzer offer those generated APIs
    // as definitions for a Ctrl+click on the component name.
    let build = quote!(
        let #elem = #func(#props_ty::builder() #(#chain)* .build());
        #fill
        let #root = ComponentHandle::root(&#elem).clone();
        #(#state_lets)*
        #style_ctx
        #style_stmt
        #visual_stmt
        #style_element_stmts
        #element_stmts
        #(#flags)*
        #(#behaviors)*
        #(#handlers)*
        #(#extras)*
        #(#parts)*
        #tooltip
    );
    (build, elem, root)
}

/// Styling for one exposed part of a component, mirroring
/// [`emit_widget_part`]: the part element comes from the handle's accessor
/// (a typo is "no method" on the handle), its own state blocks use the
/// part's interaction, and re-targeting runs inside the component's state
/// blocks fold under the root's interaction or the exposed state's local.
fn emit_component_part(
    handle: &Ident,
    root: &Ident,
    c: &ComponentNode,
    part: &PartRun,
    state_locals: &[(String, Ident)],
) -> TokenStream2 {
    let elem = Ident::new("__component_part", Span::mixed_site());
    let mut split = split_leaf_attrs(&elem, &c.styles, &part.attrs, &[]);
    split.part = Some(part.name.to_string());
    split.route_states(&part.states, &[], FONT_ATTRS);
    for block in disabled_last(&c.tag_states) {
        for state_part in &block.parts {
            if state_part.name != part.name {
                continue;
            }
            let condition =
                component_condition(&block.keyword, &quote!(__root_interaction), state_locals);
            split.route_conditional(
                state_part.attrs.iter().map(|(n, v)| (n, v)),
                condition,
                &[],
                FONT_ATTRS,
            );
        }
    }
    let style_ctx = split.style_part_ctx_stmt(&elem);
    let style = split.restyle_stmt(&elem);
    let visual = split.revisual_stmt(&elem);
    let style_element = split.style_element_stmts(&elem, &[]);
    let element = split.element_stmts(&elem);
    let extras = &split.extras;
    let accessor = &part.name;
    quote!({
        let #elem = #handle.#accessor().clone();
        let __root_interaction = #root.interaction();
        #style_ctx
        #style #visual #style_element #element #(#extras)*
    })
}

fn emit_if(c: &CondNode, owns_parent: bool) -> TokenStream2 {
    let host = binding_host(owns_parent);
    let el = node_ident();
    let (select, branches) = emit_switch_branches(c, &el, 0);
    quote!(#host.switch(move || #select, move |#el, __branch| match *__branch {
        #(#branches)*
        _ => {}
    });)
}

fn emit_switch_branches(
    cond: &CondNode,
    el: &Ident,
    index: usize,
) -> (TokenStream2, Vec<TokenStream2>) {
    let test = &cond.cond;
    let body = emit_block(quote!(#el), &cond.body);
    let mut branches = vec![quote!(#index => #body,)];
    let fallback = match cond.alternative.as_deref() {
        Some(CondAlternative::ElseIf(next)) => {
            let (select, mut rest) = emit_switch_branches(next, el, index + 1);
            branches.append(&mut rest);
            select
        }
        Some(CondAlternative::Else { body, .. }) => {
            let else_index = index + 1;
            let body = emit_block(quote!(#el), body);
            branches.push(quote!(#else_index => #body,));
            quote!(#else_index)
        }
        None => quote!(usize::MAX),
    };
    (quote!(if #test { #index } else { #fallback }), branches)
}

fn emit_for(f: &ForNode, owns_parent: bool) -> TokenStream2 {
    match &f.kind {
        ForKind::Keyed { key, value, each } => {
            let host = binding_host(owns_parent);
            // Bindings do not count toward the single-node shape, so
            // `for … { let v = *v; row { … } }` still takes the wrapper-free
            // `keyed_view` path — `emit_orphan_root` hoists them into the
            // per-item closure ahead of the element it builds. Counting them
            // would silently drop every item behind a pass-through wrapper and
            // change how the items lay out.
            let structural = f
                .body
                .iter()
                .filter(|node| node.is_structural())
                .collect::<Vec<_>>();
            if matches!(
                structural.as_slice(),
                [Node::Container(_) | Node::Component(_) | Node::Call(_)]
            ) {
                let body = emit_orphan_root(&f.body);
                quote!(#host.keyed_view(move || #each, move |#key, #value| #body);)
            } else {
                let el = node_ident();
                let body = emit_block(quote!(#el), &f.body);
                quote!(#host.keyed(move || #each, move |#el, #key, #value| #body);)
            }
        }
        // A build-time loop: the body builds into the ambient `parent` once per
        // iteration (methods borrow it, so it survives across iterations). No
        // wrapper and no reconciliation, so it needs no host and coexists with
        // siblings regardless of `owns_parent`.
        ForKind::Static { pat, iter } => {
            let stmts = f.body.iter().map(|node| emit_node(node, false));
            quote!(for #pat in #iter { #(#stmts)* })
        }
    }
}

/// The element a `show`/`keyed` binding takes over: the enclosing `parent` when
/// the `if`/`for` is a block's only node, else a layout pass-through wrapper
/// child ([`Element::binding_child`]) so it owns its own subtree and can sit
/// among siblings while the body's children stack and space like siblings of
/// the enclosing container. Emitted as a block expression so it drops straight
/// into the `.switch(…)`/`.keyed(…)` receiver position.
fn binding_host(owns_parent: bool) -> TokenStream2 {
    let parent = parent_ident();
    if owns_parent {
        quote!(#parent)
    } else {
        quote!((&#parent.binding_child()))
    }
}

/// The tag's base `Style` constructor, spanned to the tag token so IDE
/// features on `row`/`col`/`stack`/`el` resolve to it.
fn base_style(container: &Container) -> TokenStream2 {
    let tag = &container.tag;
    let span = container.span;
    if matches!(tag, Tag::Grid) {
        let value = |key: &str| {
            container.attrs.iter().find_map(|attr| match attr {
                Attr::Valued(name, value) if name == key => Some(value.tokens()),
                _ => None,
            })
        };
        let columns = value("cols");
        let rows = value("rows");
        let mut grid = match (columns, rows) {
            (Some(columns), Some(rows)) => quote!(Grid::columns(#columns).with_rows(#rows)),
            (Some(columns), None) => quote!(Grid::columns(#columns)),
            (None, Some(rows)) => quote!(Grid::rows(#rows)),
            (None, None) => quote!(Grid::columns(1usize)),
        };
        if let Some(track) = value("auto_cols") {
            grid = quote!(#grid.auto_columns(#track));
        }
        if let Some(track) = value("auto_rows") {
            grid = quote!(#grid.auto_rows(#track));
        }
        let backfill = container.attrs.iter().find_map(|attr| match attr {
            Attr::Valued(name, value) if name == "backfill" => Some(value.tokens()),
            Attr::Flag(name) if name == "backfill" => Some(quote!(true)),
            _ => None,
        });
        if let Some(enabled) = backfill {
            grid = quote!(#grid.backfill(#enabled));
        }
        return quote_spanned!(span=> Style::grid(#grid).view_fill());
    }
    let ctor = match tag {
        Tag::Row => "row",
        Tag::Col => "column",
        Tag::Stack | Tag::Canvas => "stack",
        Tag::Grid => unreachable!(),
        Tag::El => {
            let ctor = Ident::new("default", span);
            return quote!(Style::#ctor());
        }
    };
    let ctor = Ident::new(ctor, span);
    quote!(Style::#ctor().view_fill())
}

fn container_name(tag: &Tag) -> &'static str {
    match tag {
        Tag::Row => "row",
        Tag::Col => "col",
        Tag::Stack => "stack",
        Tag::Canvas => "canvas",
        Tag::Grid => "grid",
        Tag::El => "el",
    }
}

/// A builder-method ident carrying the source attribute's span, so IDE
/// features (hover, go-to-def) and diagnostics land on the token the user
/// wrote. Spans from the input resolve in the caller's scope, so hygiene is
/// unchanged.
fn spanned_method(method: &str, name: &Ident) -> Ident {
    Ident::new(method, name.span())
}

/// The value tokens an attribute name implies — the coercion (length/size
/// literals, keywords, field records) every consumer of that name applies —
/// or the plain tokens for a name outside the shared vocabulary (a component
/// prop).
/// Lowers a font length (`font-size`, `letter-spacing`) to the `f32` logical
/// pixels the text layer stores.
///
/// Going through `Length` rather than emitting the number directly is what
/// lets a `Length` scheme token stand in for a literal — `px_part` reads the
/// pixels off either.
fn font_px(value: &Value) -> TokenStream2 {
    let length = value.length();
    quote!(IntoFontLength::font_px(#length))
}

fn resolve_theme(value: TokenStream2, fallback: TokenStream2) -> TokenStream2 {
    quote!(resolve_theme_value(#value, || #fallback))
}

fn resolve_theme_if_token(
    authored: &Value,
    value: TokenStream2,
    fallback: TokenStream2,
) -> TokenStream2 {
    if authored.is_maybe_token() {
        resolve_theme(value, fallback)
    } else {
        value
    }
}

fn attr_value_tokens(name: &str, value: &Value) -> TokenStream2 {
    match name {
        // A shape's `length:` — how far a `line` runs — is a length, and it is
        // spelled with a unit like every other one. `radius:` is shared with
        // the container's corner radius, which is the same shape of value.
        "gap" | "row_gap" | "col_gap" | "radius" | "length" => value.length(),
        // A shape's coordinates. `arc:` and `angle:` are already lowered to
        // their expression by the parser, so they pass through.
        "at" | "from" | "to" | "through" | "arc" | "angle" | "sides" => value.tokens(),
        "grid_column" | "grid_col" | "grid_row" | "column_span" | "col_span" | "row_span"
        | "content_align" | "items_align" | "self_align" => value.tokens(),
        "grow" | "shrink" | "exponent" | "rotate" | "opacity" => value.f32(),
        "basis" | "width" | "height" => value.dimension(),
        "min_width" | "max_width" | "min_height" | "max_height" => value.size_bound(),
        "pad" | "margin" => value.edges(),
        "align" | "place" => resolve_theme_if_token(value, value.align(), quote!(Align::Start)),
        "justify" => resolve_theme_if_token(value, value.justify(), quote!(Justify::Start)),
        "role" => value.role(),
        "translate" | "translate_children" => {
            resolve_theme_if_token(value, value.translate(), quote!(Translate::ZERO))
        }
        "stroke" => resolve_theme_if_token(
            value,
            value.tokens(),
            quote!(StrokeSpec::new(0.0, Color::TRANSPARENT)),
        ),
        "shadow" => resolve_theme_if_token(
            value,
            value.tokens(),
            quote!(ShadowSpec::new(
                Vector2::new(0.0, 0.0),
                0.0,
                0.0,
                Color::TRANSPARENT
            )),
        ),
        "light" => resolve_theme_if_token(
            value,
            value.tokens(),
            quote!(LightSpec::point(BoxPoint::CENTER).intensity(0.0)),
        ),
        _ => value.tokens(),
    }
}

/// Lowers a style attribute to a `.method(arg)` fragment, or `None` if the
/// name is not a style attribute (it may still be a visual one).
fn style_frag(name: &Ident, value: &Value) -> Option<TokenStream2> {
    let key = name.to_string();
    let method = match key.as_str() {
        "gap" | "row_gap" | "grow" | "shrink" | "basis" | "width" | "height" | "min_width"
        | "max_width" | "min_height" | "max_height" | "margin" | "justify" | "grid_column"
        | "grid_row" | "column_span" | "row_span" | "content_align" | "items_align"
        | "self_align" => key.as_str(),
        "col_gap" => "column_gap",
        "grid_col" => "grid_column",
        "col_span" => "column_span",
        "pad" => "padding",
        "align" => "align_items",
        "place" => "align_self",
        _ => return None,
    };
    let v = attr_value_tokens(&key, value);
    let method = spanned_method(method, name);
    Some(quote!(.#method(#v)))
}

/// Lowers a visual attribute to a `.method(arg)` fragment, or `None`.
fn visual_frag(name: &Ident, value: &Value, seen: &mut VisualLayerRuns) -> Option<TokenStream2> {
    let key = name.to_string();
    let v = attr_value_tokens(&key, value);
    let concat = value.is_concat();
    let fragment = match key.as_str() {
        "fill" => {
            let method = if concat || seen.fill { "paint" } else { "fill" };
            if !concat {
                seen.fill = true;
            }
            let method = spanned_method(method, name);
            quote!(.#method(#v))
        }
        "stroke" => {
            let reset = !concat && !seen.stroke;
            if !concat {
                seen.stroke = true;
            }
            let method = spanned_method("stroke", name);
            if reset {
                quote!(.reset_strokes().#method(#v))
            } else {
                quote!(.#method(#v))
            }
        }
        "shadow" => {
            let reset = !concat && !seen.shadow;
            if !concat {
                seen.shadow = true;
            }
            let method = spanned_method("shadow", name);
            if reset {
                quote!(.reset_shadows().#method(#v))
            } else {
                quote!(.#method(#v))
            }
        }
        "light" => {
            let reset = !concat && !seen.light;
            if !concat {
                seen.light = true;
            }
            let method = spanned_method("light", name);
            if reset {
                quote!(.reset_lights().#method(#v))
            } else {
                quote!(.#method(#v))
            }
        }
        "radius" | "exponent" => {
            let method = spanned_method(&key, name);
            quote!(.#method(#v))
        }
        "rotate" => {
            let method = spanned_method("rotation", name);
            quote!(.#method(#v))
        }
        _ => return None,
    };
    Some(fragment)
}

/// Lowers `focus-style:auto|always|keyboard|never` to [`Element::focus_style`].
/// The policy is a one-time declaration on the element; an app that flips it
/// while running does so through the tree (`ui.set_focus_style(…)`), which is
/// reactive on its own.
fn focus_style_stmt(elem: &Ident, name: &Ident, value: &Value) -> Result<TokenStream2> {
    if value.is_reactive() && !matches!(value, Value::MaybeToken { .. }) {
        return Err(syn::Error::new(
            name.span(),
            "`focus-style` is a fixed policy, so it cannot bind to state — write \
             `focus-style:always` (or set the app's with `ui.set_focus_style(…)`)",
        ));
    }
    let v = resolve_theme_if_token(value, value.focus_style(), quote!(FocusStyle::Auto));
    let method = spanned_method("focus_style", name);
    Ok(quote!(#elem.#method(#v);))
}

/// Lowers authored semantic metadata into its override channel. Widgets keep
/// writing their base semantics through `Element::role`/`label`/etc.; these
/// setters never take ownership of widget-maintained value state.
fn semantic_stmt(elem: &Ident, name: &Ident, value: &Value) -> Result<TokenStream2> {
    let key = name.to_string();
    let reactive = value.is_reactive();
    let (method, dynamic, value) = match key.as_str() {
        "role" => {
            if let Some(word) = value.keyword() {
                let authored = word.to_string();
                let roles = vocabulary::ValueGrammar::Role.keywords();
                if !roles.iter().any(|role| role.replace('-', "_") == authored) {
                    let (last, leading) = roles.split_last().expect("semantic roles are non-empty");
                    return Err(syn::Error::new(
                        word.span(),
                        format!(
                            "unknown semantic role; expected {}, or {last}",
                            leading.join(", ")
                        ),
                    ));
                }
            }
            ("semantic_role", "semantic_role_dyn", value.role())
        }
        "label" => ("semantic_label", "semantic_label_dyn", value.tokens()),
        "description" => (
            "semantic_description",
            "semantic_description_dyn",
            value.tokens(),
        ),
        _ => unreachable!("semantic attributes are closed"),
    };
    let method = Ident::new(if reactive { dynamic } else { method }, name.span());
    if reactive {
        Ok(quote!(#elem.#method(move || #value);))
    } else {
        Ok(quote!(#elem.#method(#value);))
    }
}

/// `selectable:` / `searchable:` — the valued form of the two participation
/// flags, which a subtree uses to opt out of an app that granted them to
/// everything, or to follow a binding.
fn participation_stmt(elem: &Ident, name: &Ident, value: &Value) -> TokenStream2 {
    let reactive = value.is_reactive();
    let method = format!("{name}{}", if reactive { "_dyn" } else { "" });
    let method = spanned_method(&method, name);
    let tokens = value.tokens();
    if reactive {
        quote!(#elem.#method(move || #tokens);)
    } else {
        quote!(#elem.#method(#tokens);)
    }
}

fn resize_value_tokens(value: &Value) -> Result<TokenStream2> {
    let Value::Static(expr) = value else {
        let tokens = value.tokens();
        return Ok(tokens);
    };
    let expr = if let Expr::Paren(paren) = expr {
        paren.expr.as_ref()
    } else {
        expr
    };
    let edge = |expr: &Expr| -> Result<TokenStream2> {
        let Expr::Path(path) = expr else {
            return Err(syn::Error::new(
                expr.span(),
                "resize edge must be `left`, `right`, `top`, `bottom`, `all`, or `none`",
            ));
        };
        let Some(word) = path.path.get_ident().map(ToString::to_string) else {
            return Ok(quote!((#expr).into()));
        };
        match word.as_str() {
            "left" => Ok(quote!(ResizeEdges::LEFT)),
            "right" => Ok(quote!(ResizeEdges::RIGHT)),
            "top" => Ok(quote!(ResizeEdges::TOP)),
            "bottom" => Ok(quote!(ResizeEdges::BOTTOM)),
            "all" => Ok(quote!(ResizeEdges::ALL)),
            "none" => Ok(quote!(ResizeEdges::NONE)),
            _ => Err(syn::Error::new(
                path.span(),
                format!(
                    "unknown resize edge `{word}`; expected left, right, top, bottom, all, or none"
                ),
            )),
        }
    };
    let edges = match expr {
        Expr::Tuple(tuple) => {
            if tuple.elems.is_empty() {
                return Err(syn::Error::new(
                    tuple.span(),
                    "expected at least one resize edge",
                ));
            }
            tuple.elems.iter().map(edge).collect::<Result<Vec<_>>>()?
        }
        other => vec![edge(other)?],
    };
    Ok(quote!(ResizeOptions::new(#(#edges)|*)))
}

/// Lowers a lifecycle declaration — `transition:(spec)` / `enter:` / `exit:`
/// / `enter-exit:` — to the `Element` method of the same name. Lifecycle
/// attributes are one-time declarations, so their values must be static.
fn lifecycle_stmt(elem: &Ident, name: &Ident, value: &Value) -> Result<TokenStream2> {
    if value.is_reactive() && !matches!(value, Value::MaybeToken { .. }) {
        let v = value.tokens();
        let stmt = match name.to_string().as_str() {
            "transition" => quote!(#elem.transition_dyn(move || #v);),
            "draggable" => quote!(#elem.draggable_with_dyn(move || (#v).into());),
            "resizable" => quote!(#elem.resizable_with_dyn(move || (#v).into());),
            "reorderable" => quote!(#elem.reorderable_with_dyn(move || #v);),
            _ => {
                return Err(syn::Error::new(
                    name.span(),
                    format!(
                        "`{0}` fires once at mount/unmount, so it cannot bind to \
                         state — use `{0}:(…)`",
                        view_name(&name.to_string())
                    ),
                ));
            }
        };
        return Ok(stmt);
    }
    let mut v = value.tokens();
    let method = match name.to_string().as_str() {
        "draggable" => {
            if let Value::Static(expr) = value {
                let expr = if let Expr::Paren(paren) = expr {
                    paren.expr.as_ref()
                } else {
                    expr
                };
                if let Expr::Path(path) = expr {
                    let word = path.path.get_ident().map(ToString::to_string);
                    v = match word.as_deref() {
                        Some("stay") => quote!(DragOptions::from(DragRelease::Stay)),
                        Some("snap") => quote!(DragOptions::from(DragRelease::Snap)),
                        _ => v,
                    };
                }
            }
            "draggable_with"
        }
        "reorderable" => {
            if let Value::Static(expr) = value {
                let expr = if let Expr::Paren(paren) = expr {
                    paren.expr.as_ref()
                } else {
                    expr
                };
                if let Expr::Path(path) = expr
                    && path.path.is_ident("controlled")
                {
                    v = quote!(ReorderOptions::default().controlled());
                }
            }
            "reorderable_with"
        }
        "resizable" => {
            v = resize_value_tokens(value)?;
            "resizable_with"
        }
        _ => name.to_string().leak(),
    };
    v = match name.to_string().as_str() {
        "transition" => resolve_theme_if_token(value, v, quote!(Transition::default())),
        "draggable" => resolve_theme_if_token(value, v, quote!(DragOptions::default())),
        "resizable" => resolve_theme_if_token(value, v, quote!(ResizeOptions::default())),
        "reorderable" => resolve_theme_if_token(value, v, quote!(ReorderOptions::default())),
        "enter" | "exit" | "enter_exit" => resolve_theme_if_token(value, v, quote!(fade())),
        _ => v,
    };
    let method = Ident::new(method, name.span());
    Ok(quote!(#elem.#method(#v);))
}

/// Lowers `sample:parent` / `sample:behind` to the backdrop sampling scope.
/// The bare keywords are the whole vocabulary here, so an unknown word is a
/// typo worth naming rather than an expression to pass through.
fn backdrop_sample_stmt(elem: &Ident, name: &Ident, value: &Value) -> Result<TokenStream2> {
    if value.is_reactive() && !matches!(value, Value::MaybeToken { .. }) {
        return Err(syn::Error::new(
            name.span(),
            "`sample` picks what a backdrop reads and cannot bind to state",
        ));
    }
    let mut expr = value.tokens();
    if let Value::Static(inner) = value {
        let inner = match inner {
            Expr::Paren(paren) => paren.expr.as_ref(),
            other => other,
        };
        if let Expr::Path(path) = inner {
            let word = path.path.get_ident().map(ToString::to_string);
            expr = match word.as_deref() {
                Some("parent") => quote!(BackdropSample::Parent),
                Some("behind") => quote!(BackdropSample::Behind),
                Some(_) => expr,
                None => expr,
            };
        }
    }
    let expr = resolve_theme_if_token(value, expr, quote!(BackdropSample::Parent));
    Ok(quote!(#elem.backdrop_sample(#expr);))
}

pub(crate) fn flag_stmt(elem: &Ident, name: &Ident) -> Result<TokenStream2> {
    let (method, arg) = match name.to_string().as_str() {
        "clip" => ("clips", true),
        "disabled" => ("disabled", true),
        "focus" | "focusable" => ("focusable", true),
        "nohit" => ("hit_testable", false),
        "draggable" => ("draggable", true),
        "resizable" => ("resizable", true),
        "reorderable" => ("reorderable", true),
        // Bare sugar for `name:true`; the valued form goes through
        // `participation_stmt`.
        "selectable" => ("selectable", true),
        "searchable" => ("searchable", true),
        // The decoration flags are bare sugar for `name:true`; the valued form
        // goes through `route_valued` instead.
        "underline" => ("underline", true),
        "strikethrough" => ("strikethrough", true),
        "overline" => ("overline", true),
        other => {
            // A bare valued-attribute name is a missing value, not a typo.
            let msg = if vocabulary::container_attr(other).is_some() {
                format!("attribute `{other}` needs a value — write `{other}:(…)`")
            } else if let Some(meant) = did_you_mean(other, FLAGS) {
                format!("unknown flag `{other}` — did you mean `{meant}`?")
            } else {
                format!("unknown flag `{other}` (expected clip/focus/nohit)")
            };
            return Err(syn::Error::new(name.span(), msg));
        }
    };
    let method = spanned_method(method, name);
    if name == "draggable" || name == "resizable" || name == "reorderable" {
        Ok(quote!(#elem.#method();))
    } else {
        Ok(quote!(#elem.#method(#arg);))
    }
}

fn handler_stmt(elem: &Ident, handler: &Handler) -> Result<TokenStream2> {
    if matches!(&handler.value, Value::Inherit { .. }) {
        return Err(syn::Error::new(
            handler.name.span(),
            format!("handler `@{}` is not inheritable", handler.name),
        ));
    }
    if handler.name == "layout_size" {
        stop_modifier(handler, false)?;
        let Value::Static(Expr::Path(path)) = &handler.value else {
            return Err(syn::Error::new(
                handler.name.span(),
                "`@layout-size` needs a local binding name, for example `@layout-size:size`",
            ));
        };
        if path.qself.is_some()
            || path.path.leading_colon.is_some()
            || path.path.segments.len() != 1
        {
            return Err(syn::Error::new(
                path.span(),
                "`@layout-size` needs a single local binding name",
            ));
        }
        let binding = &path.path.segments[0].ident;
        let method = spanned_method("layout_size", &handler.name);
        return Ok(quote!(let #binding = #elem.#method();));
    }
    let value = match (&*handler.name.to_string(), &handler.value) {
        ("click" | "pointer_down", Value::Reactive(block)) => quote!(move || #block),
        _ => handler.value.closure_arg(),
    };
    let name = handler.name.to_string();
    let supports_stop = matches!(
        name.as_str(),
        "pointer" | "pointer_down" | "click" | "drag" | "resize" | "key"
    );
    let stop = stop_modifier(handler, supports_stop)?;
    if name == "click" && stop {
        return Ok(quote!({
            let mut __handler = #value;
            #elem.on_pointer(move |__event, __ctx| {
                if matches!(__event.kind, PointerEventKind::Click(PointerButton::Primary)) {
                    __handler();
                    __ctx.stop_propagation();
                }
            });
        }));
    }
    if name == "pointer_down" {
        return Ok(quote!({
            let mut __handler = #value;
            #elem.on_pointer(move |__event, __ctx| {
                if matches!(__event.kind, PointerEventKind::Down(PointerButton::Primary)) {
                    __handler();
                    if #stop {
                        __ctx.stop_propagation();
                    }
                }
            });
        }));
    }
    let method = match handler.name.to_string().as_str() {
        "pointer" => "on_pointer",
        "click" => "on_click",
        "drag" => "on_drag",
        "resize" => "on_resize",
        "reorder" => "on_reorder",
        "key" => "on_key",
        "focus" => "on_focus_change",
        "layout" => "on_layout",
        other => {
            let source_name = other.replace('_', "-");
            let msg = match did_you_mean(&source_name, HANDLERS) {
                Some(meant) => {
                    format!("unknown handler `@{source_name}` — did you mean `@{meant}`?")
                }
                None => format!(
                    "unknown handler `@{source_name}` (expected {})",
                    HANDLERS.join(", ")
                ),
            };
            return Err(syn::Error::new(handler.name.span(), msg));
        }
    };
    let method = spanned_method(method, &handler.name);
    if stop {
        Ok(quote!({
            let mut __handler = #value;
            #elem.#method(move |__event, __ctx| {
                __handler(__event, __ctx);
                __ctx.stop_propagation();
            });
        }))
    } else {
        Ok(quote!(#elem.#method(#value);))
    }
}

fn stop_modifier(handler: &Handler, supported: bool) -> Result<bool> {
    let mut stop = None;
    for modifier in &handler.modifiers {
        if modifier != "stop" {
            return Err(syn::Error::new(
                modifier.span(),
                format!("unknown handler modifier `.{modifier}` (expected `.stop`)"),
            ));
        }
        if stop.is_some() {
            return Err(syn::Error::new(
                modifier.span(),
                "handler modifier `.stop` may only be written once",
            ));
        }
        stop = Some(modifier.span());
    }
    if let Some(span) = stop
        && !supported
    {
        return Err(syn::Error::new(
            span,
            format!(
                "handler `@{}` does not bubble and cannot use `.stop`",
                handler.name.to_string().replace('_', "-")
            ),
        ));
    }
    Ok(stop.is_some())
}

/// Applies a stack of paint layers to `elem` via `method` (`font_color`).
///
/// Repeating the attribute layers rather than replacing: the first entry seeds
/// the stack and each later one paints over it, in source order. One layer is
/// the ordinary case and lowers to the same call it always did.
fn paint_stack_stmt(elem: &Ident, layers: &[(TokenStream2, bool)], method: &str) -> TokenStream2 {
    let (first, rest) = match layers.split_first() {
        Some(split) => split,
        None => return quote!(),
    };
    let method_ident = Ident::new(method, Span::call_site());
    let pushes = rest.iter().map(|(value, _)| quote!(.push(#value)));
    let (first_value, _) = first;
    let stack = if rest.is_empty() {
        quote!(#first_value)
    } else {
        quote!(PaintSpec::from(#first_value) #(#pushes)*)
    };
    // Reactive if any layer is: the whole stack rebuilds when one of them moves.
    if layers.iter().any(|(_, reactive)| *reactive) {
        let dynamic = Ident::new(&format!("{method}_dyn"), Span::call_site());
        quote!(#elem.#dynamic(move || #stack);)
    } else {
        quote!(#elem.#method_ident(#stack);)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosaic_syntax::vocabulary::CONTAINER_ATTRS;
    use syn::parse_quote;

    /// Every attribute the shared vocabulary advertises must actually route to a
    /// builder call or the grid-template lowering path — never fall through to
    /// the "unknown attribute" branch. This pins the vocabulary list to codegen,
    /// so a name added to one without the other is caught here rather than
    /// shipping a completion that expands to a compile error.
    #[test]
    fn every_vocabulary_attr_routes() {
        let elem = node_ident();
        for spec in CONTAINER_ATTRS {
            let name = Ident::new(spec.name, Span::call_site());
            let value = Value::Static(parse_quote!(x));
            let mut split = SplitAttrs::default();
            split.route_valued(&elem, &name, &value, &[]);
            let emitted = split
                .extras
                .iter()
                .map(|t| t.to_string())
                .collect::<String>();
            assert!(
                is_grid_template_attr(&name) || !emitted.contains("unknown attribute"),
                "`{}` is listed in the vocabulary but not routed by codegen",
                spec.name
            );
        }
    }

    #[test]
    fn img_literals_embed_relative_and_absolute_files() {
        let relative: View = syn::parse_str("el { img \"assets/picture.png\" }").unwrap();
        let emitted = expand(&relative, "el { img \"assets/picture.png\" }").to_string();
        assert!(emitted.contains("ImageSource :: __hot_embedded"));
        assert!(emitted.contains("include_bytes ! (:: core :: concat !"));
        assert!(emitted.contains("env ! (\"CARGO_MANIFEST_DIR\")"));
        assert!(emitted.contains("\"assets/picture.png\""));

        let absolute: View = syn::parse_str("el { img \"/assets/picture.png\" }").unwrap();
        let emitted = expand(&absolute, "el { img \"/assets/picture.png\" }").to_string();
        assert!(emitted.contains("include_bytes ! (\"/assets/picture.png\")"));
        assert!(emitted.contains("__hot_embedded (\"/assets/picture.png\""));
    }
}
