//! The `scheme!` macro: design-token names as compile-time citizens.
//!
//! A scheme declares token *names* and kinds — flat, or nested in named
//! groups — and the macro turns each leaf into a typed token reachable by
//! the declared path: a root-level token is a plain `const`, a group becomes
//! a `const` holding a struct of handles, so `fill:bg.base` in `view!` is
//! nothing special, just Rust field access rustc resolves, and a typo is an
//! ordinary resolution error. A token may carry a default value
//! (`title:Scalar = 36`); a theme then only states what it overrides.
//!
//! Alongside the value struct and `Theme` impl, the macro generates a
//! companion `macro_rules!` (`__mosaic_theme_{Name}`) that `theme!`
//! delegates to: only the scheme knows its full token list and defaults, so
//! only scheme-generated code can fill omitted fields while keeping a
//! missing required token a compile error.

use mosaic_syntax::vocabulary::{SCHEME_KINDS, SchemeKind as Kind};
use proc_macro2::{Punct, Spacing, TokenStream};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Ident, Result, Token, Visibility};

/// One `name: Kind [= default]` declaration.
struct TokenDecl {
    attrs: Vec<Attribute>,
    name: Ident,
    kind: Kind,
    /// The default value, already lowered to an `Into::into(…)` expression.
    default: Option<TokenStream>,
    /// The name was written snake_case rather than kebab — legal, but nudged.
    legacy: bool,
}

/// One `name { … }` group of nested declarations.
struct GroupDecl {
    attrs: Vec<Attribute>,
    name: Ident,
    entries: Vec<Entry>,
    /// See [`TokenDecl::legacy`].
    legacy: bool,
}

enum Entry {
    Token(TokenDecl),
    Group(GroupDecl),
}

impl Entry {
    fn name(&self) -> &Ident {
        match self {
            Entry::Token(token) => &token.name,
            Entry::Group(group) => &group.name,
        }
    }

    /// The kebab-case nudges this entry and everything under it earn.
    fn nudges(&self) -> TokenStream {
        match self {
            Entry::Token(token) => crate::nudge::kebab_case_token(&token.name, token.legacy),
            Entry::Group(group) => {
                let own = crate::nudge::kebab_case_token(&group.name, group.legacy);
                let inner = group.entries.iter().map(Entry::nudges);
                quote!(#own #(#inner)*)
            }
        }
    }
}

/// How many independent index spaces the store keeps. Kinds that share a
/// slot family share one counter — see [`Kind::slot`].
const SLOT_FAMILIES: usize = 6;

trait KindExt {
    fn slot(self) -> usize;
    fn stores_owned_value(self) -> bool;
    fn handle(self) -> TokenStream;
    fn value(self) -> TokenStream;
}

fn parse_kind(input: ParseStream) -> Result<Kind> {
    let ident: Ident = input.parse()?;
    let mut argument: Option<Ident> = None;
    if input.peek(Token![<]) {
        input.parse::<Token![<]>()?;
        argument = Some(input.parse()?);
        input.parse::<Token![>]>()?;
    }
    let name = ident.to_string();
    let argument = argument.map(|a| a.to_string());
    match (Kind::from_name(&name), argument.as_deref()) {
        (Some(kind), None) => Ok(kind),
        // Reserved: designed for, not yet built. A "not yet" beats an
        // "unknown", which would send someone looking for a typo.
        (None, None) if name == "Drawing" => Err(syn::Error::new(
            ident.span(),
            "token kind `Drawing` is reserved for a future drawing pipeline \
                 and is not yet supported",
        )),
        (None, Some("Svg")) if name == "Animatable" => Err(syn::Error::new(
            ident.span(),
            "animatable icon tokens are not yet supported — declare `Svg` for now",
        )),
        _ => {
            let written = match &argument {
                Some(arg) => format!("{name}<{arg}>"),
                None => name.clone(),
            };
            Err(syn::Error::new(
                ident.span(),
                format!(
                    "unknown token kind `{written}` — expected one of {}",
                    SCHEME_KINDS
                        .iter()
                        .map(|spec| format!("`{}`", spec.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ))
        }
    }
}

impl KindExt for Kind {
    /// Which store slot family this kind's index counts within.
    ///
    /// Icon kinds share one family because they share one erased slot vector
    /// in the store, so adding a further icon kind later does not renumber
    /// the tokens a scheme has already declared.
    fn slot(self) -> usize {
        match self {
            Kind::Paint => 0,
            Kind::Color => 1,
            Kind::Length => 2,
            Kind::Scalar => 3,
            Kind::Svg | Kind::Element => 4,
            _ => 5,
        }
    }

    /// Whether the value type is `Clone` but not `Copy`, so `apply` must
    /// clone it into the store.
    fn stores_owned_value(self) -> bool {
        !matches!(self, Kind::Color | Kind::Length | Kind::Scalar)
    }

    /// The token handle type a const of this kind holds.
    fn handle(self) -> TokenStream {
        match self {
            Kind::Paint => quote!(PaintToken),
            Kind::Color => quote!(ColorToken),
            Kind::Length => quote!(LengthToken),
            Kind::Scalar => quote!(ScalarToken),
            Kind::Svg => quote!(SvgToken),
            Kind::Element => quote!(ElementToken),
            _ => {
                let value = self.value();
                quote!(ThemeToken<#value>)
            }
        }
    }

    /// The value type the theme struct's field holds.
    fn value(self) -> TokenStream {
        self.spec()
            .value_type
            .parse()
            .expect("scheme kind value type is Rust tokens")
    }
}

fn parse_entries(input: ParseStream) -> Result<Vec<Entry>> {
    let mut entries: Vec<Entry> = Vec::new();
    while !input.is_empty() {
        let attrs = input.call(Attribute::parse_outer)?;
        // Token names are kebab-case in source and snake_case as generated
        // Rust — the same rule `view!` applies to attribute names, sharing
        // the same parser so the two cannot drift. A snake_case name is a
        // scheme's to declare, so it compiles; the author is nudged toward
        // kebab rather than stopped (see `crate::nudge`).
        let (name, legacy) = mosaic_syntax::parse_kebab_ident(input, "token")?;
        let legacy = legacy.is_some();
        if let Some(first) = entries.iter().find(|e| e.name() == &name) {
            return Err(syn::Error::new(
                name.span(),
                format!("duplicate name `{}` in this group", first.name()),
            ));
        }
        if input.peek(syn::token::Brace) {
            let body;
            syn::braced!(body in input);
            let inner = parse_entries(&body)?;
            if inner.is_empty() {
                return Err(syn::Error::new(
                    name.span(),
                    "a group needs at least one token",
                ));
            }
            entries.push(Entry::Group(GroupDecl {
                attrs,
                name,
                entries: inner,
                legacy,
            }));
            // A comma after a group's closing brace is optional, like after
            // an item.
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        } else {
            input.parse::<Token![:]>()?;
            let kind = parse_kind(input)?;
            let default = if input.peek(Token![=]) {
                input.parse::<Token![=]>()?;
                Some(mosaic_syntax::parse_scheme_value(input, kind)?)
            } else {
                None
            };
            entries.push(Entry::Token(TokenDecl {
                attrs,
                name,
                kind,
                default,
                legacy,
            }));
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
    }
    Ok(entries)
}

struct Scheme {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    entries: Vec<Entry>,
}

impl Parse for Scheme {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        let body;
        syn::braced!(body in input);
        let entries = parse_entries(&body)?;
        if entries.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "a scheme needs at least one token",
            ));
        }
        Ok(Scheme {
            attrs,
            vis,
            name,
            entries,
        })
    }
}

/// FNV-1a over the declaring crate and the scheme's name: a stable tag
/// telling one scheme's tokens from another's at runtime, cheap enough to
/// bake into every const.
///
/// The crate is in the hash because the name alone is not unique — two crates
/// linked into one binary both calling their scheme `Palette` is the ordinary
/// case, not a strange one, and a shared tag means each install silently
/// overwrites the other's values. The tag is computed where the scheme is
/// *declared*, so a scheme keeps its identity wherever it is used, and
/// `CARGO_CRATE_NAME` is what the compiler was told to build (falling back to
/// the package, then to nothing, for rustc invocations without cargo).
fn scheme_tag(name: &Ident) -> u32 {
    let krate = std::env::var("CARGO_CRATE_NAME")
        .or_else(|_| std::env::var("CARGO_PKG_NAME"))
        .unwrap_or_default();
    tag_of(&krate, &name.to_string())
}

/// The tag itself, apart from the environment it reads.
fn tag_of(krate: &str, name: &str) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for byte in krate.bytes().chain(*b"::").chain(name.bytes()) {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// One leaf token with its full path and assigned per-kind store index.
struct Leaf {
    path: Vec<Ident>,
    kind: Kind,
    has_default: bool,
    /// Path segments joined with `_`: the field name `theme!` flattens to.
    flat: Ident,
    /// Path segments joined with `.`: the name diagnostics print.
    dotted: String,
    index: u16,
}

/// Walks the tree in declaration order, assigning per-kind indices — each
/// kind has its own slots in the store — and catching two nested tokens
/// whose paths flatten to the same `theme!` field name.
fn flatten(entries: &[Entry]) -> Result<Vec<Leaf>> {
    fn walk(
        entries: &[Entry],
        prefix: &mut Vec<Ident>,
        counters: &mut [u16; SLOT_FAMILIES],
        leaves: &mut Vec<Leaf>,
    ) -> Result<()> {
        for entry in entries {
            match entry {
                Entry::Token(token) => {
                    let mut path = prefix.clone();
                    path.push(token.name.clone());
                    let strings: Vec<String> = path.iter().map(Ident::to_string).collect();
                    let flat = Ident::new(&strings.join("_"), token.name.span());
                    let dotted = strings.join(".");
                    if let Some(other) = leaves.iter().find(|leaf| leaf.flat == flat) {
                        return Err(syn::Error::new(
                            token.name.span(),
                            format!(
                                "tokens `{}` and `{dotted}` flatten to the same \
                                 `theme!` field name `{flat}`; rename one",
                                other.dotted
                            ),
                        ));
                    }
                    let index = &mut counters[token.kind.slot()];
                    let assigned = *index;
                    *index += 1;
                    leaves.push(Leaf {
                        path,
                        kind: token.kind,
                        has_default: token.default.is_some(),
                        flat,
                        dotted,
                        index: assigned,
                    });
                }
                Entry::Group(group) => {
                    prefix.push(group.name.clone());
                    walk(&group.entries, prefix, counters, leaves)?;
                    prefix.pop();
                }
            }
        }
        Ok(())
    }
    let mut leaves = Vec::new();
    walk(
        entries,
        &mut Vec::new(),
        &mut [0u16; SLOT_FAMILIES],
        &mut leaves,
    )?;
    Ok(leaves)
}

/// `text_col` → `TextCol`, for struct names built from group paths.
fn camel(ident: &Ident) -> String {
    ident
        .to_string()
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// The type name for a group's handle struct (`ThemeTextSizeTokens`) or
/// value struct (`ThemeTextSize`).
fn group_ty(scheme: &Ident, path: &[&Ident], tokens: bool) -> Ident {
    let camel_path: String = path.iter().map(|seg| camel(seg)).collect();
    if tokens {
        format_ident!("{scheme}{camel_path}Tokens")
    } else {
        format_ident!("{scheme}{camel_path}")
    }
}

pub fn expand(input: TokenStream) -> Result<TokenStream> {
    let scheme: Scheme = syn::parse2(input)?;
    let Scheme {
        attrs,
        vis,
        name,
        entries,
    } = &scheme;
    let tag = scheme_tag(name);
    let leaves = flatten(entries)?;
    let index_of = |path: &[&Ident]| -> u16 {
        let dotted = path
            .iter()
            .map(|seg| seg.to_string())
            .collect::<Vec<_>>()
            .join(".");
        leaves
            .iter()
            .find(|leaf| leaf.dotted == dotted)
            .expect("every declared token was flattened")
            .index
    };

    // --- Token consts and per-group handle structs -------------------------

    let mut token_structs: Vec<TokenStream> = Vec::new();
    // Post-order walk: nested groups' structs are defined before the structs
    // whose fields name them (order doesn't matter to rustc, but reads well).
    fn emit_token_structs<'a>(
        scheme: &Scheme,
        entries: &'a [Entry],
        path: &mut Vec<&'a Ident>,
        out: &mut Vec<TokenStream>,
    ) {
        for entry in entries {
            if let Entry::Group(group) = entry {
                path.push(&group.name);
                emit_token_structs(scheme, &group.entries, path, out);
                let ty = group_ty(&scheme.name, path, true);
                let vis = &scheme.vis;
                let fields = group.entries.iter().map(|entry| match entry {
                    Entry::Token(token) => {
                        let name = &token.name;
                        let handle = token.kind.handle();
                        quote!(pub #name: #handle,)
                    }
                    Entry::Group(inner) => {
                        let name = &inner.name;
                        path.push(&inner.name);
                        let inner_ty = group_ty(&scheme.name, path, true);
                        path.pop();
                        quote!(pub #name: #inner_ty,)
                    }
                });
                let doc = format!(
                    "Token handles for the `{}` group of [`{}`].",
                    group.name, scheme.name
                );
                out.push(quote! {
                    #[doc = #doc]
                    #[derive(Clone, Copy)]
                    #[allow(non_camel_case_types)]
                    #vis struct #ty {
                        #(#fields)*
                    }
                });
                path.pop();
            }
        }
    }
    emit_token_structs(&scheme, entries, &mut Vec::new(), &mut token_structs);

    /// The const initializer for a group's handle struct.
    fn group_init<'a>(
        scheme: &Scheme,
        tag: u32,
        group: &'a GroupDecl,
        path: &mut Vec<&'a Ident>,
        index_of: &dyn Fn(&[&Ident]) -> u16,
    ) -> TokenStream {
        let ty = group_ty(&scheme.name, path, true);
        let fields = group
            .entries
            .iter()
            .map(|entry| match entry {
                Entry::Token(token) => {
                    let name = &token.name;
                    let handle = token.kind.handle();
                    path.push(&token.name);
                    let index = index_of(path);
                    path.pop();
                    quote!(#name: <#handle>::new(#tag, #index),)
                }
                Entry::Group(inner) => {
                    let name = &inner.name;
                    path.push(&inner.name);
                    let init = group_init(scheme, tag, inner, path, index_of);
                    path.pop();
                    quote!(#name: #init,)
                }
            })
            .collect::<Vec<_>>();
        quote!(#ty { #(#fields)* })
    }

    let consts = entries.iter().map(|entry| match entry {
        Entry::Token(token) => {
            let token_attrs = &token.attrs;
            let token_name = &token.name;
            let handle = token.kind.handle();
            let index = index_of(&[&token.name]);
            quote! {
                #(#token_attrs)*
                #[allow(non_upper_case_globals)]
                #vis const #token_name: #handle = <#handle>::new(#tag, #index);
            }
        }
        Entry::Group(group) => {
            let group_attrs = &group.attrs;
            let group_name = &group.name;
            let mut path = vec![&group.name];
            let ty = group_ty(name, &path, true);
            let init = group_init(&scheme, tag, group, &mut path, &index_of);
            quote! {
                #(#group_attrs)*
                #[allow(non_upper_case_globals)]
                #vis const #group_name: #ty = #init;
            }
        }
    });

    // --- Value structs -----------------------------------------------------

    let mut value_structs: Vec<TokenStream> = Vec::new();
    fn emit_value_structs<'a>(
        scheme: &Scheme,
        entries: &'a [Entry],
        path: &mut Vec<&'a Ident>,
        out: &mut Vec<TokenStream>,
    ) {
        for entry in entries {
            if let Entry::Group(group) = entry {
                path.push(&group.name);
                emit_value_structs(scheme, &group.entries, path, out);
                let ty = group_ty(&scheme.name, path, false);
                let vis = &scheme.vis;
                let fields = group.entries.iter().map(|entry| match entry {
                    Entry::Token(token) => {
                        let attrs = &token.attrs;
                        let name = &token.name;
                        let value = token.kind.value();
                        quote!(#(#attrs)* pub #name: #value,)
                    }
                    Entry::Group(inner) => {
                        let attrs = &inner.attrs;
                        let name = &inner.name;
                        path.push(&inner.name);
                        let inner_ty = group_ty(&scheme.name, path, false);
                        path.pop();
                        quote!(#(#attrs)* pub #name: #inner_ty,)
                    }
                });
                let doc = format!(
                    "Values for the `{}` group of [`{}`].",
                    group.name, scheme.name
                );
                out.push(quote! {
                    #[doc = #doc]
                    #[derive(Clone, Debug, PartialEq)]
                    #vis struct #ty {
                        #(#fields)*
                    }
                });
                path.pop();
            }
        }
    }
    emit_value_structs(&scheme, entries, &mut Vec::new(), &mut value_structs);

    let root_fields = entries.iter().map(|entry| match entry {
        Entry::Token(token) => {
            let attrs = &token.attrs;
            let field_name = &token.name;
            let value = token.kind.value();
            quote!(#(#attrs)* pub #field_name: #value,)
        }
        Entry::Group(group) => {
            let attrs = &group.attrs;
            let field_name = &group.name;
            let ty = group_ty(name, &[&group.name], false);
            quote!(#(#attrs)* pub #field_name: #ty,)
        }
    });

    // --- Theme impl --------------------------------------------------------

    let writes = leaves.iter().map(|leaf| {
        let path = &leaf.path;
        match leaf.kind {
            // Icons go through the widget crate's seam, which tags the value
            // with its kind so a token can never read back another kind's.
            Kind::Svg => quote!(set_svg(#(#path).*, self.#(#path).*.clone());),
            Kind::Element => quote!(set_icon_element(#(#path).*, self.#(#path).*.clone());),
            // The paint value is not `Copy`; the store takes it owned.
            kind if kind.stores_owned_value() => quote!(#(#path).*.set(self.#(#path).*.clone());),
            _ => quote!(#(#path).*.set(self.#(#path).*);),
        }
    });

    // --- Default value fns -------------------------------------------------

    // Defaults live in hidden inherent fns so the companion macro can reach
    // them through the scheme's path — the values then resolve here, where
    // the scheme's types are in scope, not at the `theme!` call site.
    let mut default_fns: Vec<TokenStream> = Vec::new();
    fn emit_default_fns(entries: &[Entry], prefix: &mut Vec<Ident>, out: &mut Vec<TokenStream>) {
        for entry in entries {
            match entry {
                Entry::Token(token) => {
                    if let Some(default) = &token.default {
                        let mut segs: Vec<String> = prefix.iter().map(Ident::to_string).collect();
                        segs.push(token.name.to_string());
                        let fn_name = format_ident!("__default_{}", segs.join("_"));
                        let value = token.kind.value();
                        out.push(quote! {
                            #[doc(hidden)]
                            pub fn #fn_name() -> #value { #default }
                        });
                    }
                }
                Entry::Group(group) => {
                    prefix.push(group.name.clone());
                    emit_default_fns(&group.entries, prefix, out);
                    prefix.pop();
                }
            }
        }
    }
    emit_default_fns(entries, &mut Vec::new(), &mut default_fns);
    let default_impl = (!default_fns.is_empty()).then(|| {
        quote! {
            impl #name {
                #(#default_fns)*
            }
        }
    });

    // --- Companion macro for `theme!` --------------------------------------

    let companion = companion_macro(&scheme, &leaves);

    // Kebab is the canonical spelling; a name that isn't earns a warning, not
    // a rejection.
    let nudges = entries.iter().map(Entry::nudges);

    Ok(quote! {
        #(#nudges)*

        #(#attrs)*
        #[derive(Clone, Debug, PartialEq)]
        #vis struct #name {
            #(#root_fields)*
        }

        #(#value_structs)*

        #(#token_structs)*

        #(#consts)*

        impl Theme for #name {
            fn scheme_tag(&self) -> u32 {
                #tag
            }

            fn apply(&self) {
                #(#writes)*
            }
        }

        #default_impl

        #companion
    })
}

/// The nested struct literal the companion macro's `@build` rules expand to,
/// with every field filled by an `@pick` call over the caller's field list.
/// `base` is the base slot the picks thread through: `[]` (fall back to
/// defaults) or `[__b]` (fall back to the `..base` value).
fn build_literal(
    scheme: &Scheme,
    entries: &[Entry],
    path: &mut Vec<Ident>,
    base: &TokenStream,
    d: &Punct,
    companion: &Ident,
) -> TokenStream {
    let scheme_name = &scheme.name;
    let fields = entries
        .iter()
        .map(|entry| match entry {
            Entry::Token(token) => {
                let field_name = &token.name;
                let mut segs: Vec<String> = path.iter().map(Ident::to_string).collect();
                segs.push(token.name.to_string());
                let flat = format_ident!("{}", segs.join("_"));
                quote! {
                    #field_name: #d(#d p)* #companion!(
                        @pick (#d(#d p)*) (#d(#d h)*) #base #flat #d([#d n #d v #d s])*
                    ),
                }
            }
            Entry::Group(group) => {
                let field_name = &group.name;
                path.push(group.name.clone());
                let refs: Vec<&Ident> = path.iter().collect();
                let ty = group_ty(scheme_name, &refs, false);
                let inner = build_literal(scheme, &group.entries, path, base, d, companion);
                path.pop();
                quote!(#field_name: #d(#d p)* #ty #inner,)
            }
        })
        .collect::<Vec<_>>();
    if path.is_empty() {
        quote!(#d(#d p)* #scheme_name { #(#fields)* })
    } else {
        quote!({ #(#fields)* })
    }
}

/// The `macro_rules!` companion `theme!` delegates to. Only the scheme knows
/// its full token list and which tokens have defaults, so the companion —
/// not `theme!` — reconstructs the complete struct literal: each field is
/// picked from the caller's `[flat (value) "dotted"]` list, falling back to
/// the `..base` value, then the declared default, then a `compile_error!`.
///
/// Every rule carries the caller's path prefix `($($p)*)` (e.g. `style ::`)
/// because macro-generated paths and recursive invocations resolve at the
/// `theme!` call site, which need not have the scheme's items in scope.
fn companion_macro(scheme: &Scheme, leaves: &[Leaf]) -> TokenStream {
    let d = Punct::new('$', Spacing::Alone);
    let d = &d;
    let name = &scheme.name;
    let vis = &scheme.vis;
    let companion = format_ident!("__mosaic_theme_{}", name);

    let literal_defaults = build_literal(
        scheme,
        &scheme.entries,
        &mut Vec::new(),
        &quote!([]),
        d,
        &companion,
    );
    let literal_base = build_literal(
        scheme,
        &scheme.entries,
        &mut Vec::new(),
        &quote!([__b]),
        d,
        &companion,
    );

    let pick_arms = leaves.iter().map(|leaf| {
        let flat = &leaf.flat;
        let path = &leaf.path;
        let kind = Ident::new(leaf.kind.spec().name, proc_macro2::Span::call_site());
        let exhausted_base = match leaf.kind {
            kind if kind.stores_owned_value() => quote!(#d b.#(#path).*.clone()),
            _ => quote!(#d b.#(#path).*),
        };
        let exhausted_none = if leaf.has_default {
            let fn_name = format_ident!("__default_{}", flat);
            quote!(#d(#d p)* #name::#fn_name())
        } else {
            let message = format!(
                "theme for `{name}` omits required token `{}`, which has no default",
                leaf.dotted
            );
            quote!(compile_error!(#message))
        };
        quote! {
            (@pick (#d(#d p:tt)*) (#d(#d h:tt)*) #d base:tt #flat
                [#flat #d v:tt #d s:literal] #d(#d rest:tt)*) => {
                #d(#d h)*!(#kind #d v)
            };
            (@pick (#d(#d p:tt)*) (#d(#d h:tt)*) #d base:tt #flat
                [#d o:ident #d v:tt #d s:literal] #d(#d rest:tt)*) => {
                #d(#d p)* #companion!(@pick (#d(#d p)*) (#d(#d h)*) #d base #flat #d(#d rest)*)
            };
            (@pick (#d(#d p:tt)*) (#d(#d h:tt)*) [#d b:ident] #flat) => { #exhausted_base };
            (@pick (#d(#d p:tt)*) (#d(#d h:tt)*) [] #flat) => { #exhausted_none };
        }
    });

    let check_known_arms = leaves.iter().map(|leaf| {
        let flat = &leaf.flat;
        quote! {
            (@check (#d(#d p:tt)*) [#flat #d v:tt #d s:literal] #d(#d rest:tt)*) => {
                #d(#d p)* #companion!(@check (#d(#d p)*) #d(#d rest)*);
            };
        }
    });
    let unknown_message = format!("` is not a token of scheme `{name}`");

    // `use` lifts the textual macro into the module's item namespace so
    // `style::__mosaic_theme_S!` works. A macro_rules item is at most
    // crate-visible without `#[macro_export]`, so a `pub` scheme's companion
    // is capped at `pub(crate)` — `theme!` reaches it from anywhere in the
    // defining crate; other crates write the struct literal.
    let reexport = match vis {
        Visibility::Inherited => quote!(pub(self) use #companion;),
        Visibility::Public(_) => quote!(pub(crate) use #companion;),
        vis => quote!(#vis use #companion;),
    };

    quote! {
        #[doc(hidden)]
        #[allow(unused_macros)]
        macro_rules! #companion {
            (@build (#d(#d p:tt)*) (#d(#d h:tt)*) #d([#d n:ident #d v:tt #d s:literal])*) => {{
                #d(#d p)* #companion!(@check (#d(#d p)*) #d([#d n #d v #d s])*);
                #literal_defaults
            }};
            (@build (#d(#d p:tt)*) (#d(#d h:tt)*) #d([#d n:ident #d v:tt #d s:literal])* .. #d base:expr) => {{
                #d(#d p)* #companion!(@check (#d(#d p)*) #d([#d n #d v #d s])*);
                let __b = &(#d base);
                #literal_base
            }};
            (@check (#d(#d p:tt)*)) => {};
            #(#check_known_arms)*
            (@check (#d(#d p:tt)*) [#d bad:ident #d v:tt #d s:literal] #d(#d rest:tt)*) => {
                compile_error!(concat!("`", #d s, #unknown_message));
            };
            #(#pick_arms)*
        }
        #[allow(unused_imports)]
        #reexport
    }
}

#[cfg(test)]
mod tests {
    use super::tag_of;

    #[test]
    fn the_same_scheme_name_in_two_crates_is_two_schemes() {
        assert_ne!(
            tag_of("mosaic_example_mail_app", "Palette"),
            tag_of("mosaic_example_music_app", "Palette"),
            "a shared tag would make one app's theme overwrite the other's"
        );
    }

    #[test]
    fn a_scheme_keeps_its_tag() {
        assert_eq!(tag_of("app", "Palette"), tag_of("app", "Palette"));
        assert_ne!(tag_of("app", "Palette"), tag_of("app", "Geometry"));
    }
}
