//! The `theme!` macro: a theme written in the DSL's value syntax.
//!
//! A theme fills the struct [`scheme!`](crate::scheme) makes, but writing
//! one in plain Rust means spelling out `Length::px(10.0)` and
//! `GradientSpec::new(KindSpec::Angle { … }, …)` for values the `view!`
//! grammar writes as `10px` and `linear(angle:90deg, stops:(…))`. This
//! macro parses those field values with the same grammar — nested group
//! braces mirror the scheme's nesting — so a theme reads like the
//! attributes it will end up filling.
//!
//! Only the scheme knows its full token list and defaults, so after
//! lowering the values this macro flattens the fields and delegates to the
//! scheme-generated companion macro (`__mosaic_theme_{Name}`), which
//! rebuilds the complete struct literal: given fields win, then `..base`,
//! then declared defaults; a missing required token or an unknown name is a
//! compile error.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Path, Result, Token};

/// One `token:value` pair, flattened: `text { size { title:40 } }`
/// becomes flat name `text_size_title`, dotted name `text.size.title`.
struct Field {
    flat: Ident,
    dotted: String,
    /// Raw authored tokens. The scheme companion supplies the declared kind
    /// before the contextual parser lowers them.
    value: TokenStream,
}

struct ThemeLit {
    path: Path,
    fields: Vec<Field>,
    /// A `..other` base, letting a theme state only what it changes.
    rest: Option<Expr>,
}

fn parse_fields(
    input: ParseStream,
    prefix: &mut Vec<String>,
    fields: &mut Vec<Field>,
) -> Result<Option<Expr>> {
    let mut rest = None;
    while !input.is_empty() {
        if input.peek(Token![..]) {
            if !prefix.is_empty() {
                return Err(input.error("`..base` goes at the theme's top level"));
            }
            input.parse::<Token![..]>()?;
            rest = Some(input.parse()?);
            break;
        }
        // Kebab in source, snake_case as the generated field — the same rule
        // and the same parser `scheme!` uses to declare the token. A theme
        // spells its tokens however the scheme declared them, so an odd
        // spelling passes silently here: the nudge toward kebab belongs at
        // the declaration, and repeating it per theme would only be noise.
        let (name, _legacy) = mosaic_syntax::parse_kebab_ident(input, "token")?;
        if input.peek(syn::token::Brace) {
            let body;
            syn::braced!(body in input);
            prefix.push(name.to_string());
            parse_fields(&body, prefix, fields)?;
            prefix.pop();
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        } else {
            input.parse::<Token![:]>()?;
            let mut value = TokenStream::new();
            while !input.is_empty() && !input.peek(Token![,]) {
                value.extend(std::iter::once(input.parse::<proc_macro2::TokenTree>()?));
            }
            if value.is_empty() {
                return Err(input.error("expected a token value"));
            }
            let mut segs = prefix.clone();
            segs.push(name.to_string());
            let flat = Ident::new(&segs.join("_"), name.span());
            let dotted = segs.join(".");
            if fields.iter().any(|field| field.flat == flat) {
                return Err(syn::Error::new(
                    name.span(),
                    format!("duplicate token `{dotted}`"),
                ));
            }
            fields.push(Field {
                flat,
                dotted,
                value,
            });
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
    }
    if !input.is_empty() {
        return Err(input.error("`..base` must come last"));
    }
    Ok(rest)
}

impl Parse for ThemeLit {
    fn parse(input: ParseStream) -> Result<Self> {
        let path: Path = input.parse()?;
        let body;
        syn::braced!(body in input);
        let mut fields = Vec::new();
        let rest = parse_fields(&body, &mut Vec::new(), &mut fields)?;
        Ok(ThemeLit { path, fields, rest })
    }
}

pub fn expand(input: TokenStream) -> Result<TokenStream> {
    let ThemeLit { path, fields, rest } = syn::parse2(input)?;

    // The scheme's companion macro lives next to the scheme struct: rewrite
    // the last path segment and keep the prefix, passing the prefix along —
    // paths inside the companion's expansion resolve at this call site.
    let scheme = &path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(&path, "expected a scheme name"))?
        .ident;
    let companion = format_ident!("__mosaic_theme_{}", scheme);
    let mut prefix = TokenStream::new();
    if let Some(colons) = path.leading_colon {
        prefix.extend(quote!(#colons));
    }
    for pair in path.segments.pairs() {
        if pair.punct().is_some() {
            let seg = pair.value();
            prefix.extend(quote!(#seg ::));
        }
    }

    let helper = helper_path();
    let fields = fields.iter().map(
        |Field {
             flat,
             dotted,
             value,
         }| { quote!([#flat [#value] #dotted]) },
    );
    let rest = rest.map(|base| quote!(.. #base));
    Ok(quote!(#prefix #companion!(@build (#prefix) (#helper) #(#fields)* #rest)))
}

fn helper_path() -> TokenStream {
    use proc_macro_crate::{FoundCrate, crate_name};
    let found = crate_name("mosaic").or_else(|_| crate_name("mosaic-macros"));
    match found {
        Ok(FoundCrate::Itself) => quote!(crate::__mosaic_scheme_value),
        Ok(FoundCrate::Name(name)) => {
            let name = Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#name::__mosaic_scheme_value)
        }
        Err(_) => quote!(::__mosaic_scheme_value),
    }
}
