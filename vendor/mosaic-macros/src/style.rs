//! Lowering `style!{}` to `StyleSet` constants.
//!
//! Each declaration becomes one `const` plus the private functions its fields
//! point at — a fold per channel the attribute run can write. The bodies come
//! from the same [`SplitAttrs`] routing `view!` uses, so a style accepts
//! exactly the attribute surface a node does and cannot drift from it.
//!
//! The one substitution is the state condition: `view!` tests a node's
//! `Interaction` directly, while a style is written before it knows what it
//! will be applied to, so it asks its context by name. That is what lets a
//! style scope a run to `on`, `checked`, or any state the host exposes.

use mosaic_syntax::vocabulary::FONT_ATTRS;
use mosaic_syntax::{PartRun, StateBlock, StyleDecl, StyleSheet, Value, WidgetStateBlock};
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};

use crate::codegen::{SplitAttrs, flag_stmt, state_index, stmt_or_error};

/// The layer channels `|` resets before the replacing style pushes its own.
const LAYER_RESETS: &[(&str, &str)] = &[
    ("fill", "reset_fill"),
    ("stroke", "reset_strokes"),
    ("shadow", "reset_shadows"),
    ("light", "reset_lights"),
];

pub fn expand(sheet: &StyleSheet) -> TokenStream2 {
    let errors = sheet.errors.iter().map(syn::Error::to_compile_error);
    // A name declared twice is already reported; emitting both would bury
    // that under a pile of redefinition errors for the generated items.
    let decls = sheet
        .decls
        .iter()
        .enumerate()
        .filter(|(index, decl)| {
            !sheet.decls[..*index]
                .iter()
                .any(|prior| prior.path == decl.path)
        })
        .map(|(_, decl)| emit_decl(decl));
    let groups = emit_groups(sheet);
    quote!(#(#errors)* #(#decls)* #groups)
}

/// The context every generated fold reads its states and mode from.
fn cx() -> Ident {
    Ident::new("__style_cx", Span::call_site())
}

/// The element the element-setter channels target.
fn host() -> Ident {
    Ident::new("__style_el", Span::call_site())
}

/// The part name a part channel dispatches on.
fn part_arg() -> Ident {
    Ident::new("__style_part", Span::call_site())
}

/// A declaration's flat name, used for its generated functions and — for a
/// plain declaration — its constant.
fn flat(decl: &StyleDecl) -> String {
    decl.path
        .iter()
        .map(Ident::to_string)
        .collect::<Vec<_>>()
        .join("_")
}

fn fn_name(decl: &StyleDecl, channel: &str) -> Ident {
    format_ident!("__style_{}_{}", flat(decl), channel)
}

/// Routes the declaration's own attribute run, its interaction states, and
/// its exposed-state blocks into one [`SplitAttrs`].
///
/// State blocks route as *conditional* fragments — the seam `view!` already
/// uses for a widget's `on`/`checked` — with the condition asking the context
/// by name instead of reading an `Interaction`. Precedence follows `view!`:
/// base, then focused, hover, pressed, the named states in written order, and
/// disabled last.
fn route_root(decl: &StyleDecl) -> SplitAttrs {
    let elem = host();
    let mut split = SplitAttrs::default();
    for (name, value) in &decl.attrs_run {
        split.route_valued(&elem, name, value, &[]);
    }
    for flag in &decl.flags {
        split.extras.push(stmt_or_error(flag_stmt(&elem, flag)));
    }
    route_states(&mut split, &decl.states, &decl.tag_states, false);
    split
}

/// The shared state routing for a root run and a part run. `scoped` marks a
/// part run, whose own interaction states follow the part while the
/// declaration's named states still resolve on the node.
fn route_states(
    split: &mut SplitAttrs,
    states: &[StateBlock],
    tag_states: &[WidgetStateBlock],
    scoped: bool,
) {
    let mut ordered: Vec<&StateBlock> = states.iter().collect();
    ordered.sort_by_key(|block| state_index(block.state));
    let disabled = ordered
        .iter()
        .position(|block| block.state == mosaic_syntax::InteractionState::Disabled)
        .map(|index| ordered.remove(index));
    for block in ordered {
        let name = block.state.name();
        let cx = cx();
        split.route_conditional(
            block.attrs.iter().map(|(n, v)| (n, v)),
            quote!(#cx.active(#name)),
            &[],
            FONT_ATTRS,
        );
    }
    for block in tag_states {
        let name = block.keyword.to_string();
        let cx = cx();
        // A named state on a part run belongs to the node, never the part —
        // `on { thumb … }` is the widget being on, not the thumb.
        let condition = if scoped {
            quote!(#cx.root_active(#name))
        } else {
            quote!(#cx.active(#name))
        };
        split.route_conditional(
            block.attrs.iter().map(|(n, v)| (n, v)),
            condition,
            &[],
            FONT_ATTRS,
        );
    }
    if let Some(block) = disabled {
        if block.attrs.iter().any(|(name, _)| name == "opacity") {
            let elem = host();
            split
                .extras
                .push(quote::quote!(#elem.authored_disabled_opacity();));
        }
        let cx = cx();
        split.route_conditional(
            block.attrs.iter().map(|(n, v)| (n, v)),
            quote!(#cx.active("disabled")),
            &[],
            FONT_ATTRS,
        );
    }
}

/// Every attribute name the declaration writes anywhere, which is what the
/// `|` operator resets before this style pushes its own layers.
fn declared_names(decl: &StyleDecl) -> Vec<String> {
    let mut names = Vec::new();
    let mut push = |attrs: &[(Ident, Value)]| {
        names.extend(attrs.iter().map(|(name, _)| name.to_string()));
    };
    push(&decl.attrs_run);
    for block in &decl.states {
        push(&block.attrs);
    }
    for block in &decl.tag_states {
        push(&block.attrs);
        for part in &block.parts {
            push(&part.attrs);
        }
    }
    for part in &decl.parts {
        push(&part.attrs);
        for block in &part.states {
            push(&block.attrs);
        }
    }
    names
}

/// The `reset_*` chain a replacing style applies first. Only the channels it
/// actually writes are reset: `#a | #b` supersedes what `b` declares and
/// leaves the rest of `a` standing.
fn layer_resets(names: &[String]) -> TokenStream2 {
    let resets = LAYER_RESETS
        .iter()
        .filter(|(attr, _)| names.iter().any(|name| name == attr))
        .map(|(_, reset)| Ident::new(reset, Span::call_site()));
    let chain: Vec<_> = resets.collect();
    if chain.is_empty() {
        return quote!();
    }
    let cx = cx();
    quote!(let __visual = if #cx.replaces() { __visual #(.#chain())* } else { __visual };)
}

fn style_body(split: &SplitAttrs) -> TokenStream2 {
    let frags = &split.style_frags;
    let folds = split.chain_folds(&quote!(__style), |state| &state.style_frags);
    quote!(
        let __style = __style #(#frags)*;
        #(#folds)*
        __style
    )
}

fn visual_base_body(split: &SplitAttrs, resets: &TokenStream2) -> TokenStream2 {
    let frags = &split.visual_frags;
    quote!(
        #resets
        let __visual = __visual #(#frags)*;
        __visual
    )
}

fn visual_states_body(split: &SplitAttrs) -> TokenStream2 {
    let folds = split.chain_folds(&quote!(__visual), |state| &state.visual_frags);
    quote!(
        #(#folds)*
        __visual
    )
}

fn element_body(split: &SplitAttrs) -> TokenStream2 {
    let stmts = split.element_stmts(&host());
    let extras = &split.extras;
    let cx = cx();
    quote!(let #cx = (*#cx).clone(); #stmts #(#extras)*)
}

/// Whether a routed run writes anything on a channel, so a part that styles
/// only its layout never supersedes the widget's own appearance binding.
fn writes_style(split: &SplitAttrs) -> bool {
    !split.style_frags.is_empty()
        || !split
            .chain_folds(&quote!(__style), |state| &state.style_frags)
            .is_empty()
}

fn writes_visual(split: &SplitAttrs) -> bool {
    !split.visual_frags.is_empty()
        || !split
            .chain_folds(&quote!(__visual), |state| &state.visual_frags)
            .is_empty()
}

fn writes_element(split: &SplitAttrs) -> bool {
    !split.element_stmts(&host()).is_empty() || !split.extras.is_empty()
}

/// The part runs a declaration carries, merged across the body and the named
/// state blocks: a part mentioned only inside `on { … }` still gets a run, so
/// its conditional styling is not dropped.
fn part_runs(decl: &StyleDecl) -> Vec<(String, SplitAttrs)> {
    let mut names: Vec<String> = Vec::new();
    for part in &decl.parts {
        let name = part.name.to_string();
        if !names.contains(&name) {
            names.push(name);
        }
    }
    for block in &decl.tag_states {
        for part in &block.parts {
            let name = part.name.to_string();
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }

    names
        .into_iter()
        .map(|name| {
            let elem = host();
            let mut split = SplitAttrs::default();
            let own: Vec<&PartRun> = decl.parts.iter().filter(|part| part.name == name).collect();
            for part in &own {
                for (attr, value) in &part.attrs {
                    split.route_valued(&elem, attr, value, &[]);
                }
            }
            for part in &own {
                route_states(&mut split, &part.states, &[], true);
            }
            for block in &decl.tag_states {
                for part in block.parts.iter().filter(|part| part.name == name) {
                    let keyword = block.keyword.to_string();
                    let cx = cx();
                    split.route_conditional(
                        part.attrs.iter().map(|(n, v)| (n, v)),
                        quote!(#cx.root_active(#keyword)),
                        &[],
                        FONT_ATTRS,
                    );
                }
            }
            (name, split)
        })
        .collect()
}

/// One declaration's private channel functions, plus its constant when it is
/// a plain (undotted) name. A dotted declaration's value becomes a field of
/// its group's constant instead, emitted by [`emit_groups`].
fn emit_decl(decl: &StyleDecl) -> TokenStream2 {
    let split = route_root(decl);
    let resets = layer_resets(&declared_names(decl));
    let parts = part_runs(decl);

    let cx = cx();
    let elem = host();
    let arg = part_arg();

    let style_fn = fn_name(decl, "style");
    let visual_fn = fn_name(decl, "visual");
    let visual_states_fn = fn_name(decl, "visual_states");
    let element_fn = fn_name(decl, "element");
    let part_style_fn = fn_name(decl, "part_style");
    let part_visual_fn = fn_name(decl, "part_visual");
    let part_visual_states_fn = fn_name(decl, "part_visual_states");
    let part_element_fn = fn_name(decl, "part_element");

    let style = style_body(&split);
    let visual = visual_base_body(&split, &resets);
    let visual_states = visual_states_body(&split);
    let element = element_body(&split);

    let part_style_arms =
        parts
            .iter()
            .filter(|(_, split)| writes_style(split))
            .map(|(name, split)| {
                let body = style_body(split);
                quote!(#name => { #body })
            });
    let part_visual_arms =
        parts
            .iter()
            .filter(|(_, split)| writes_visual(split))
            .map(|(name, split)| {
                let body = visual_base_body(split, &resets);
                quote!(#name => { #body })
            });
    let part_visual_state_arms =
        parts
            .iter()
            .filter(|(_, split)| writes_visual(split))
            .map(|(name, split)| {
                let body = visual_states_body(split);
                quote!(#name => { #body })
            });
    let part_element_arms =
        parts
            .iter()
            .filter(|(_, split)| writes_element(split))
            .map(|(name, split)| {
                let body = element_body(split);
                quote!(#name => { #body })
            });

    // `__interaction` is bound by the element channel whenever a state scopes
    // one of its attributes, and a style's conditions read the context
    // instead — so the binding can go unused in code the author never sees.
    let channels = quote!(
        #[doc(hidden)]
        #[allow(unused_variables)]
        fn #style_fn(__style: Style, #cx: &StyleCtx) -> Style { #style }

        #[doc(hidden)]
        #[allow(unused_variables)]
        fn #visual_fn(__visual: Visual, #cx: &StyleCtx) -> Visual { #visual }

        #[doc(hidden)]
        #[allow(unused_variables)]
        fn #visual_states_fn(__visual: Visual, #cx: &StyleCtx) -> Visual { #visual_states }

        #[doc(hidden)]
        #[allow(unused_variables)]
        fn #element_fn(#elem: &Element, #cx: &StyleCtx) { #element }
    );

    let part_channels = (!parts.is_empty()).then(|| {
        quote!(
            #[doc(hidden)]
            #[allow(unused_variables)]
            fn #part_style_fn(#arg: &str, __style: Style, #cx: &StyleCtx) -> Style {
                match #arg { #(#part_style_arms)* _ => __style }
            }

            #[doc(hidden)]
            #[allow(unused_variables)]
            fn #part_visual_fn(#arg: &str, __visual: Visual, #cx: &StyleCtx) -> Visual {
                match #arg { #(#part_visual_arms)* _ => __visual }
            }

            #[doc(hidden)]
            #[allow(unused_variables)]
            fn #part_visual_states_fn(#arg: &str, __visual: Visual, #cx: &StyleCtx) -> Visual {
                match #arg { #(#part_visual_state_arms)* _ => __visual }
            }

            #[doc(hidden)]
            #[allow(unused_variables)]
            fn #part_element_fn(#arg: &str, #elem: &Element, #cx: &StyleCtx) {
                match #arg { #(#part_element_arms)* _ => {} }
            }
        )
    });

    let item = (decl.path.len() == 1).then(|| {
        let attrs = &decl.attrs;
        let vis = &decl.vis;
        let name = &decl.path[0];
        let set = set_expr(decl, &parts, split.visual_base_is_reactive());
        quote!(
            #(#attrs)*
            #[allow(non_upper_case_globals)]
            #vis const #name: StyleSet = #set;
        )
    });

    quote!(#channels #part_channels #item)
}

/// The `StyleSet` value a declaration builds, shared by a plain constant and
/// a group's field.
fn set_expr(
    decl: &StyleDecl,
    parts: &[(String, SplitAttrs)],
    root_visual_reactive: bool,
) -> TokenStream2 {
    let root = route_root(decl);
    // Part style/visual folds are reversible in exactly the same way as the
    // root folds. Only the mount-only element setter channel makes a style
    // unsafe to select conditionally.
    let conditional_safe =
        !writes_element(&root) && parts.iter().all(|(_, split)| !writes_element(split));
    let name = decl.name();
    let style_fn = fn_name(decl, "style");
    let visual_fn = fn_name(decl, "visual");
    let visual_states_fn = fn_name(decl, "visual_states");
    let element_fn = fn_name(decl, "element");
    let part_style_fn = fn_name(decl, "part_style");
    let part_visual_fn = fn_name(decl, "part_visual");
    let part_visual_states_fn = fn_name(decl, "part_visual_states");
    let part_element_fn = fn_name(decl, "part_element");
    let specs: Vec<_> = parts
        .iter()
        .map(|(part, split)| {
            let style = writes_style(split);
            let visual = writes_visual(split);
            let visual_reactive = split.visual_base_is_reactive();
            let element = writes_element(split);
            quote!(StylePart::new(
                #part,
                #style,
                #visual,
                #visual_reactive,
                #element,
            ))
        })
        .collect();
    let with_parts = (!specs.is_empty()).then(|| {
        quote!(.with_parts(
            &[#(#specs),*],
            #part_style_fn,
            #part_visual_fn,
            #part_visual_states_fn,
            #part_element_fn,
        ))
    });
    quote!(
        StyleSet::new(#name)
            .with_style(#style_fn)
            .with_visual(#visual_fn, #visual_states_fn, #root_visual_reactive)
            .with_element(#element_fn)
            #with_parts
            .with_conditional_safe(#conditional_safe)
    )
}

/// The struct-and-constant pair a dotted declaration group expands to, one
/// field per member — the shape `scheme!` gives a nested token group, so a
/// dotted reference is plain Rust field access.
fn emit_groups(sheet: &StyleSheet) -> TokenStream2 {
    let mut groups: Vec<(Ident, Vec<&StyleDecl>)> = Vec::new();
    for decl in sheet.decls.iter().filter(|decl| decl.path.len() > 1) {
        let head = decl.path[0].clone();
        match groups.iter_mut().find(|(name, _)| *name == head) {
            Some((_, members)) => members.push(decl),
            None => groups.push((head, vec![decl])),
        }
    }

    let items = groups.into_iter().map(|(name, members)| {
        let ty = format_ident!("{}Styles", pascal(&name.to_string()));
        let vis = &members[0].vis;
        let fields = members.iter().map(|decl| {
            let field = &decl.path[decl.path.len() - 1];
            let attrs = &decl.attrs;
            quote!(#(#attrs)* pub #field: StyleSet)
        });
        let values = members.iter().map(|decl| {
            let field = &decl.path[decl.path.len() - 1];
            let split = route_root(decl);
            let parts = part_runs(decl);
            let set = set_expr(decl, &parts, split.visual_base_is_reactive());
            quote!(#field: #set)
        });
        quote!(
            #[doc = "Styles declared in this group by `style!`."]
            #[derive(Clone, Copy)]
            #vis struct #ty { #(#fields),* }

            #[allow(non_upper_case_globals)]
            #vis const #name: #ty = #ty { #(#values),* };
        )
    });
    quote!(#(#items)*)
}

fn pascal(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}
