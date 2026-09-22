//! `mosaic-text` — text shaping and glyph rasterization.
//!
//! Correct international text (shaping, bidi, font fallback) is an enormous
//! problem, so Mosaic borrows [`cosmic-text`] (+ `swash`) for exactly that layer
//! and nothing above it. This crate wraps cosmic-text in a small, Mosaic-shaped
//! API:
//!
//! - [`FontContext`] owns the font database and the rasterization cache.
//! - [`FontContext::shape`] turns a string + [`TextStyle`] into a [`ShapedText`],
//!   which exposes [`TextMetrics`] (measured size, line height, ascent/descent)
//!   for the future layout engine (M4) to consume.
//! - [`ShapedText::place`] rasterizes the shaped glyphs to 8-bit coverage and
//!   emits a [`GlyphRun`] — the render seam's text paint command. `mosaic-render`
//!   never names cosmic-text; only this crate does.
//!
//! A permissively-licensed fallback font ([DejaVu Sans]) is embedded so shaping
//! and metrics are deterministic without system fonts (hermetic tests, headless
//! CI, and Android where `/system/fonts` availability varies). System fonts are
//! still loaded for real coverage when present (see [`FontContext::new`]).
//!
//! Coordinates are **logical pixels**, but glyphs are rasterized at the
//! display's pixel density (the `scale` passed to [`ShapedText::place`]) so text
//! stays crisp on HiDPI — the coverage bitmaps are physical-resolution while
//! placement stays logical.
//!
//! [`cosmic-text`]: https://docs.rs/cosmic-text
//! [DejaVu Sans]: https://dejavu-fonts.github.io/

mod edit;

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cosmic_text::{
    Attrs, Buffer, CacheKey, Cursor, Family, FontSystem, Metrics, Shaping, Stretch, Style,
    SwashCache, SwashContent, UnderlineStyle, Weight, Wrap, fontdb,
};
use mosaic_core::{Color, Inherit, Length, Rect, Size, Vector2};
use mosaic_render::{GlyphImage, GlyphRun, Paint, PaintSpec, PlacedGlyph};
use unicode_segmentation::UnicodeSegmentation;

pub use edit::{CaretMotion, ColorSpan, EditBuffer, LineMetadata};

/// The embedded fallback font (DejaVu Sans). See `fonts/LICENSE`.
static EMBEDDED_FONT: &[u8] = include_bytes!("../fonts/DejaVuSans.ttf");

/// Family name of the embedded fallback font. Shape with this
/// family (or [`FontFamily::Named`] of it) to force deterministic output.
pub const EMBEDDED_FONT_FAMILY: &str = "DejaVu Sans";

/// The smallest font size handed to the shaper. Below this it is not text
/// anyone can read, and at exactly zero cosmic-text aborts the process.
const MIN_FONT_SIZE: f32 = 1.0;

/// Which font family to shape with. Maps onto cosmic-text's generic families
/// plus named lookup; resolution (and fallback) is cosmic-text's job.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FontFamily {
    #[default]
    SansSerif,
    Serif,
    Monospace,
    /// A specific family by name, e.g. `"DejaVu Sans"`.
    Named(String),
}

impl FontFamily {
    pub(crate) fn as_cosmic(&self) -> Family<'_> {
        match self {
            FontFamily::SansSerif => Family::SansSerif,
            FontFamily::Serif => Family::Serif,
            FontFamily::Monospace => Family::Monospace,
            FontFamily::Named(name) => Family::Name(name),
        }
    }
}

/// Upright, italic, or slanted. Maps onto cosmic-text's `Style`; the font's own
/// italic face is used when it has one, and synthesized otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

impl FontStyle {
    pub(crate) fn as_cosmic(self) -> Style {
        match self {
            FontStyle::Normal => Style::Normal,
            FontStyle::Italic => Style::Italic,
            FontStyle::Oblique => Style::Oblique,
        }
    }
}

/// Width of the font face, for families shipping condensed/expanded cuts. Has
/// no effect on families that do not — there is nothing to synthesize from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontStretch {
    UltraCondensed,
    ExtraCondensed,
    Condensed,
    SemiCondensed,
    #[default]
    Normal,
    SemiExpanded,
    Expanded,
    ExtraExpanded,
    UltraExpanded,
}

impl FontStretch {
    pub(crate) fn as_cosmic(self) -> Stretch {
        match self {
            FontStretch::UltraCondensed => Stretch::UltraCondensed,
            FontStretch::ExtraCondensed => Stretch::ExtraCondensed,
            FontStretch::Condensed => Stretch::Condensed,
            FontStretch::SemiCondensed => Stretch::SemiCondensed,
            FontStretch::Normal => Stretch::Normal,
            FontStretch::SemiExpanded => Stretch::SemiExpanded,
            FontStretch::Expanded => Stretch::Expanded,
            FontStretch::ExtraExpanded => Stretch::ExtraExpanded,
            FontStretch::UltraExpanded => Stretch::UltraExpanded,
        }
    }
}

/// The distance between the baselines of consecutive lines.
///
/// Written either as an absolute length or as a multiple of the resolved font
/// size. The multiple is what keeps a paragraph's rhythm when the size it
/// inherits changes; the absolute form pins the leading to a layout grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineHeight {
    /// Leading in logical pixels, independent of the font size.
    Px(f32),
    /// Leading as a multiple of the resolved font size.
    Scale(f32),
}

impl LineHeight {
    /// The leading in logical pixels for text shaped at `size_px`.
    pub fn resolve(self, size_px: f32) -> f32 {
        match self {
            LineHeight::Px(px) => px,
            LineHeight::Scale(scale) => size_px * scale,
        }
    }
}

impl From<Length> for LineHeight {
    fn from(length: Length) -> Self {
        LineHeight::Px(length.px_part())
    }
}

/// The `view!` spelling of the value, so tooling reports what an author writes.
impl core::fmt::Display for LineHeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LineHeight::Px(px) => write!(f, "{px}px"),
            LineHeight::Scale(scale) => write!(f, "{scale}"),
        }
    }
}

/// How a `view!` line-height value becomes a [`LineHeight`]: a bare number is a
/// multiple of the font size, a length is absolute. The `view!` macro emits
/// through this trait, which is why both spellings type-check at one attribute.
pub trait IntoLineHeight {
    fn line_height(self) -> LineHeight;
}

impl IntoLineHeight for LineHeight {
    fn line_height(self) -> LineHeight {
        self
    }
}

impl IntoLineHeight for f32 {
    fn line_height(self) -> LineHeight {
        LineHeight::Scale(self)
    }
}

impl IntoLineHeight for Length {
    fn line_height(self) -> LineHeight {
        LineHeight::Px(self.px_part())
    }
}

/// Where a line may break when the text is wider than its box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextWrap {
    /// Never wrap; the text overflows its box on one line per `\n`.
    None,
    /// Break between words only, so a long word overflows rather than splitting.
    Word,
    /// Break anywhere, mid-word.
    Glyph,
    /// Break between words, falling back to mid-word for a word that cannot fit
    /// a line by itself.
    #[default]
    WordOrGlyph,
}

impl TextWrap {
    pub(crate) fn as_cosmic(self) -> Wrap {
        match self {
            TextWrap::None => Wrap::None,
            TextWrap::Word => Wrap::Word,
            TextWrap::Glyph => Wrap::Glyph,
            TextWrap::WordOrGlyph => Wrap::WordOrGlyph,
        }
    }
}

/// A case transform applied to the string before it is shaped. Presentational
/// only — it does not change the text the app holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
    /// Uppercase the first letter of each whitespace-separated word.
    Capitalize,
}

impl TextTransform {
    /// Borrows when there is nothing to do, so the common case does not allocate.
    pub(crate) fn apply(self, text: &str) -> Cow<'_, str> {
        match self {
            TextTransform::None => Cow::Borrowed(text),
            TextTransform::Uppercase => Cow::Owned(text.to_uppercase()),
            TextTransform::Lowercase => Cow::Owned(text.to_lowercase()),
            // Split on the character class rather than `split_whitespace` so the
            // original spacing survives the round trip.
            TextTransform::Capitalize => {
                let mut out = String::with_capacity(text.len());
                let mut at_word_start = true;
                for ch in text.chars() {
                    if at_word_start && !ch.is_whitespace() {
                        out.extend(ch.to_uppercase());
                    } else {
                        out.push(ch);
                    }
                    at_word_start = ch.is_whitespace();
                }
                Cow::Owned(out)
            }
        }
    }
}

/// How to shape a run of text: family, size, weight, and line height.
///
/// Line height defaults to `1.3 × size_px` when [`line_height`](Self::line_height)
/// is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub family: Inherit<FontFamily>,
    /// Font size in logical pixels.
    pub size_px: Inherit<f32>,
    /// Leading, absolute or relative to the size; `None` derives it from `size_px`.
    pub line_height: Inherit<Option<LineHeight>>,
    /// OpenType weight (100–900); 400 is regular, 700 bold.
    pub weight: Inherit<u16>,
    /// Upright, italic, or oblique.
    pub style: Inherit<FontStyle>,
    /// Condensed/expanded face selection.
    pub stretch: Inherit<FontStretch>,
    /// Extra tracking between glyphs, in logical pixels. Converted to the EM
    /// fraction cosmic-text wants at shape time, against the resolved size.
    pub letter_spacing_px: Inherit<f32>,
    /// Where lines may break.
    pub wrap: Inherit<TextWrap>,
    /// Width of a tab stop, in space characters.
    pub tab_size: Inherit<u16>,
    /// A case transform applied before shaping.
    pub transform: Inherit<TextTransform>,
    /// Rule under the baseline, positioned from the font's own metrics.
    pub underline: Inherit<bool>,
    /// Rule through the middle of the glyphs.
    pub strikethrough: Inherit<bool>,
    /// Rule above the ascent.
    pub overline: Inherit<bool>,
    /// What fills the glyphs: a flat color, or a gradient written against the
    /// text element's box. Inherits like every other font attribute, so a
    /// gradient set on a container flows down to the text inside it.
    pub color: Inherit<PaintSpec>,
}

impl TextStyle {
    /// A style with a locally-authored size; every other *inheritable* field
    /// inherits. See [`inherited`](Self::inherited) for which those are.
    pub fn new(size_px: f32) -> Self {
        TextStyle {
            size_px: Inherit::set(size_px),
            ..TextStyle::inherited()
        }
    }

    /// The tree-root fallback style.
    ///
    /// All `font-*` properties cascade — color, family, size, stretch, style,
    /// and weight — plus `line_height` and `tab_size`. The rest are authored
    /// per element and deliberately do *not* flow down. `inherit` remains
    /// available to opt in explicitly.
    pub fn inherited() -> Self {
        TextStyle {
            family: Inherit::from_parent(FontFamily::SansSerif),
            size_px: Inherit::from_parent(16.0),
            line_height: Inherit::from_parent(None),
            weight: Inherit::from_parent(400),
            style: Inherit::from_parent(FontStyle::Normal),
            stretch: Inherit::from_parent(FontStretch::Normal),
            letter_spacing_px: Inherit::set(0.0),
            wrap: Inherit::set(TextWrap::WordOrGlyph),
            tab_size: Inherit::from_parent(8),
            transform: Inherit::set(TextTransform::None),
            underline: Inherit::set(false),
            strikethrough: Inherit::set(false),
            overline: Inherit::set(false),
            color: Inherit::from_parent(PaintSpec::solid(Color::WHITE)),
        }
    }

    pub fn family(mut self, family: FontFamily) -> Self {
        self.family = family.into();
        self
    }

    pub fn weight(mut self, weight: u16) -> Self {
        self.weight = weight.into();
        self
    }

    pub fn line_height(mut self, height: impl Into<LineHeight>) -> Self {
        self.line_height = Some(height.into()).into();
        self
    }

    pub fn style(mut self, style: FontStyle) -> Self {
        self.style = style.into();
        self
    }

    pub fn stretch(mut self, stretch: FontStretch) -> Self {
        self.stretch = stretch.into();
        self
    }

    pub fn letter_spacing(mut self, px: f32) -> Self {
        self.letter_spacing_px = px.into();
        self
    }

    pub fn wrap(mut self, wrap: TextWrap) -> Self {
        self.wrap = wrap.into();
        self
    }

    pub fn tab_size(mut self, spaces: u16) -> Self {
        self.tab_size = spaces.into();
        self
    }

    pub fn transform(mut self, transform: TextTransform) -> Self {
        self.transform = transform.into();
        self
    }

    pub fn underline(mut self, on: bool) -> Self {
        self.underline = on.into();
        self
    }

    pub fn strikethrough(mut self, on: bool) -> Self {
        self.strikethrough = on.into();
        self
    }

    pub fn overline(mut self, on: bool) -> Self {
        self.overline = on.into();
        self
    }

    /// Whether any decoration rule is on, so callers can skip the geometry walk.
    pub fn has_decoration(&self) -> bool {
        *self.underline || *self.strikethrough || *self.overline
    }

    /// Fill the glyphs. Takes a color, a gradient spec, or a whole
    /// [`PaintSpec`] stack.
    pub fn color(mut self, color: impl Into<PaintSpec>) -> Self {
        self.color = color.into().into();
        self
    }

    pub fn resolve_from(&mut self, parent: &TextStyle) {
        self.family.resolve_from(&parent.family);
        self.size_px.resolve_from(&parent.size_px);
        self.line_height.resolve_from(&parent.line_height);
        self.weight.resolve_from(&parent.weight);
        self.style.resolve_from(&parent.style);
        self.stretch.resolve_from(&parent.stretch);
        self.letter_spacing_px
            .resolve_from(&parent.letter_spacing_px);
        self.wrap.resolve_from(&parent.wrap);
        self.tab_size.resolve_from(&parent.tab_size);
        self.transform.resolve_from(&parent.transform);
        self.underline.resolve_from(&parent.underline);
        self.strikethrough.resolve_from(&parent.strikethrough);
        self.overline.resolve_from(&parent.overline);
        self.color.resolve_from(&parent.color);
    }

    pub fn has_inherit(&self) -> bool {
        self.family.is_inherit()
            || self.size_px.is_inherit()
            || self.line_height.is_inherit()
            || self.weight.is_inherit()
            || self.style.is_inherit()
            || self.stretch.is_inherit()
            || self.letter_spacing_px.is_inherit()
            || self.wrap.is_inherit()
            || self.tab_size.is_inherit()
            || self.transform.is_inherit()
            || self.underline.is_inherit()
            || self.strikethrough.is_inherit()
            || self.overline.is_inherit()
            || self.color.is_inherit()
    }

    pub fn same_shaping(&self, other: &TextStyle) -> bool {
        *self.family == *other.family
            && *self.size_px == *other.size_px
            && *self.line_height == *other.line_height
            && *self.weight == *other.weight
            && *self.style == *other.style
            && *self.stretch == *other.stretch
            && *self.letter_spacing_px == *other.letter_spacing_px
            && *self.wrap == *other.wrap
            && *self.tab_size == *other.tab_size
            && *self.transform == *other.transform
            // Decoration spans are computed during layout, not at paint time,
            // so toggling one has to re-shape even though metrics are unchanged.
            && *self.underline == *other.underline
            && *self.strikethrough == *other.strikethrough
            && *self.overline == *other.overline
    }

    pub fn same_authored(&self, other: &TextStyle) -> bool {
        self.same_authored_shaping(other) && self.color.same_authored(&other.color)
    }

    pub fn same_authored_shaping(&self, other: &TextStyle) -> bool {
        self.family.same_authored(&other.family)
            && self.size_px.same_authored(&other.size_px)
            && self.line_height.same_authored(&other.line_height)
            && self.weight.same_authored(&other.weight)
            && self.style.same_authored(&other.style)
            && self.stretch.same_authored(&other.stretch)
            && self
                .letter_spacing_px
                .same_authored(&other.letter_spacing_px)
            && self.wrap.same_authored(&other.wrap)
            && self.tab_size.same_authored(&other.tab_size)
            && self.transform.same_authored(&other.transform)
            && self.underline.same_authored(&other.underline)
            && self.strikethrough.same_authored(&other.strikethrough)
            && self.overline.same_authored(&other.overline)
    }

    pub fn resolved_line_height(&self) -> f32 {
        match *self.line_height {
            Some(height) => height.resolve(*self.size_px),
            None => *self.size_px * 1.3,
        }
    }

    /// The size actually handed to the shaper, floored at [`MIN_FONT_SIZE`].
    pub(crate) fn shaped_size(&self) -> f32 {
        self.size_px.max(MIN_FONT_SIZE)
    }

    /// The cosmic-text attributes for this style. Shared by the static and
    /// editable paths so the two cannot drift.
    ///
    /// Note `transform` is deliberately absent: it rewrites the string, and the
    /// editable path maps caret offsets back onto the app's own text, which a
    /// rewrite would desynchronize.
    pub(crate) fn attrs(&self) -> Attrs<'_> {
        let mut attrs = Attrs::new()
            .family(self.family.as_cosmic())
            .weight(Weight(*self.weight))
            .style(self.style.as_cosmic())
            .stretch(self.stretch.as_cosmic());
        // cosmic-text tracks in EM; authors write logical pixels.
        if *self.letter_spacing_px != 0.0 {
            attrs = attrs.letter_spacing(*self.letter_spacing_px / self.shaped_size());
        }
        // Setting these makes the layout pass emit decoration spans carrying the
        // font's own offset/thickness metrics, which `decoration_rects` reads.
        if *self.underline {
            attrs = attrs.underline(UnderlineStyle::Single);
        }
        if *self.strikethrough {
            attrs = attrs.strikethrough();
        }
        if *self.overline {
            attrs = attrs.overline();
        }
        attrs
    }

    pub(crate) fn metrics(&self) -> Metrics {
        // The shaper aborts the process (a non-unwinding assert) on a zero
        // line height, so a degenerate size never reaches it. Zero is
        // reachable from ordinary app code — an uninstalled scheme token
        // falls back to `0.0` — and a hard crash is the wrong way to report
        // that; tiny-but-visible text lets the fallback warning be read. The
        // floor is only that guard: leading tighter than the size is legitimate
        // typography, so it is passed through.
        let size = self.shaped_size();
        Metrics::new(size, self.resolved_line_height().max(MIN_FONT_SIZE))
    }
}

/// Measured extent of a [`ShapedText`], in logical pixels. This is what the
/// layout engine (M4) reads to place a text box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMetrics {
    /// Tight bounding size of the laid-out text.
    pub size: Size,
    /// Distance between baselines.
    pub line_height: f32,
    /// Top of the first line to its baseline.
    pub ascent: f32,
    /// First baseline to the bottom of the line.
    pub descent: f32,
}

/// The font database plus the glyph rasterization cache. Create one per
/// application (it is `!Send` in spirit — the UI thread owns it).
pub struct FontContext {
    pub(crate) font_system: FontSystem,
    swash_cache: SwashCache,
    /// Rasterized coverage bitmaps, by the same key the atlas files them
    /// under.
    ///
    /// Placement runs for every visible glyph on every painted frame, while
    /// the bitmap for a given key never changes — so without this each frame
    /// re-copied every glyph's pixels into a fresh allocation that the atlas,
    /// which already holds that key, then ignored. Holding them here makes a
    /// repeat glyph an `Arc` clone, which is what [`PlacedGlyph::image`]
    /// always claimed it was.
    glyphs: HashMap<CacheKey, Option<RasterGlyph>>,
}

/// One glyph as the rasterizer left it: the coverage bitmap, where it sits
/// relative to the pen, and the atlas key it files under.
#[derive(Clone)]
struct RasterGlyph {
    /// Atlas key, hashed once here rather than per placement.
    key: u64,
    image: Arc<GlyphImage>,
    /// Bitmap offset from the pen position, in physical pixels.
    left: i32,
    top: i32,
}

impl core::fmt::Debug for FontContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FontContext").finish_non_exhaustive()
    }
}

impl FontContext {
    /// System fonts plus the embedded fallback. Use this in real apps so text
    /// gets full platform coverage while still rendering where system fonts are
    /// missing.
    pub fn new() -> Self {
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(EMBEDDED_FONT.to_vec());
        FontContext {
            font_system,
            swash_cache: SwashCache::new(),
            glyphs: HashMap::new(),
        }
    }

    /// Only the embedded fallback font — no system fonts. Deterministic, so it
    /// is what tests (and hermetic/headless environments) should use.
    pub fn embedded_only() -> Self {
        let mut db = fontdb::Database::new();
        db.load_font_data(EMBEDDED_FONT.to_vec());
        let font_system = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
        FontContext {
            font_system,
            swash_cache: SwashCache::new(),
            glyphs: HashMap::new(),
        }
    }

    /// Add one bundled TTF, OTF, TTC, or OpenType collection to this context.
    ///
    /// Load application fonts before creating text that names their family;
    /// existing shaped buffers keep the faces they were built with. Use
    /// [`has_family`](Self::has_family) when setup can run repeatedly, such as
    /// across hot reloads, to avoid registering the same bytes more than once.
    pub fn load_font_data(&mut self, data: impl Into<Vec<u8>>) {
        self.font_system.db_mut().load_font_data(data.into());
    }

    /// Point [`FontFamily::SansSerif`] at a concrete family.
    ///
    /// The generic families resolve through names that only exist because a
    /// platform installs them, so on a host without system fonts — a browser,
    /// a hermetic test — every generic silently falls back to the same face.
    /// An application that bundles its own fonts says which is which here,
    /// after loading them.
    pub fn set_sans_serif_family(&mut self, family: impl Into<String>) {
        self.font_system.db_mut().set_sans_serif_family(family);
    }

    /// Point [`FontFamily::Serif`] at a concrete family. See
    /// [`set_sans_serif_family`](Self::set_sans_serif_family).
    pub fn set_serif_family(&mut self, family: impl Into<String>) {
        self.font_system.db_mut().set_serif_family(family);
    }

    /// Point [`FontFamily::Monospace`] at a concrete family. See
    /// [`set_sans_serif_family`](Self::set_sans_serif_family).
    pub fn set_monospace_family(&mut self, family: impl Into<String>) {
        self.font_system.db_mut().set_monospace_family(family);
    }

    /// Whether any registered face advertises `family` by this exact name.
    pub fn has_family(&self, family: &str) -> bool {
        self.font_system
            .db()
            .faces()
            .any(|face| face.families.iter().any(|(name, _)| name == family))
    }

    /// Whether any registered face of `family` carries this width class.
    ///
    /// A condensed or expanded cut joins the family it belongs to rather than
    /// forming one of its own, so [`has_family`](Self::has_family) cannot see
    /// it. This is the guard for loading such a cut only once.
    pub fn has_width_class(&self, family: &str, stretch: FontStretch) -> bool {
        let stretch = stretch.as_cosmic();
        self.font_system.db().faces().any(|face| {
            face.stretch == stretch && face.families.iter().any(|(name, _)| name == family)
        })
    }

    /// The rasterization for `key`, rasterizing on first sight.
    ///
    /// `None` means the key has nothing to draw — whitespace, a color glyph
    /// (unsupported), or a zero-area box. That verdict is cached too, so a run
    /// of spaces stops asking the rasterizer every frame.
    fn raster(&mut self, key: CacheKey) -> Option<RasterGlyph> {
        // Returned by value so the hit path — every glyph of every frame after
        // the first — costs exactly one lookup. Borrowing would force a
        // `contains_key` before the index (the borrow checker cannot see that
        // the miss arm has released the map), and at the sizes real text
        // rasterizes to, a second hash of the key costs more than the bitmap
        // copy this cache exists to avoid. Cloning is an `Arc` bump and three
        // scalars.
        if let Some(entry) = self.glyphs.get(&key) {
            return entry.clone();
        }
        // Not `entry`: rasterizing needs `&mut self.font_system` alongside the
        // cache, which an occupied entry would still be holding.
        let entry = self
            .swash_cache
            .get_image(&mut self.font_system, key)
            .as_ref()
            .filter(|image| image.content == SwashContent::Mask)
            .filter(|image| image.placement.width > 0 && image.placement.height > 0)
            .map(|image| RasterGlyph {
                key: glyph_key(&key),
                image: Arc::new(GlyphImage {
                    width: image.placement.width,
                    height: image.placement.height,
                    data: image.data.clone().into_boxed_slice(),
                }),
                left: image.placement.left,
                top: image.placement.top,
            });
        self.glyphs.insert(key, entry.clone());
        entry
    }

    /// Shape and lay out `text` in `style`, wrapping at `wrap_px` logical pixels
    /// if given (`None` = no wrapping, one visual line per `\n`-separated line).
    ///
    /// A `wrap_px` of `0.0` is the min-content probe: it wraps at word
    /// boundaries only, so the measured width is the longest word (an
    /// unbreakable run) rather than the widest single glyph.
    pub fn shape(&mut self, text: &str, style: &TextStyle, wrap_px: Option<f32>) -> ShapedText {
        let mut buffer = Buffer::new(&mut self.font_system, style.metrics());
        configure_buffer(&mut buffer, style, wrap_px);
        let attrs = style.attrs();
        let text = style.transform.apply(text);
        buffer.set_text(&text, &attrs, Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut self.font_system, false);

        let metrics = measure(&buffer, style.metrics().line_height);
        ShapedText {
            buffer,
            metrics,
            runs: Vec::new(),
        }
    }

    /// Shape and lay out several differently-styled runs as one continuous
    /// text — a paragraph with a code chip or a bold clause in it.
    ///
    /// This is what makes styled text *one* element: the runs share a single
    /// buffer, so they wrap into each other the way words in a sentence do,
    /// and every byte offset the result reports indexes the concatenation.
    /// Building the same paragraph out of one element per run cannot wrap at
    /// all without the caller re-implementing line breaking.
    ///
    /// Each run's style is resolved against `base` first, so a run states only
    /// what it changes. `wrap_px` works exactly as in [`shape`](Self::shape).
    pub fn shape_runs(
        &mut self,
        runs: &[TextRun<'_>],
        base: &TextStyle,
        wrap_px: Option<f32>,
    ) -> ShapedText {
        let mut buffer = Buffer::new(&mut self.font_system, base.metrics());
        configure_buffer(&mut buffer, base, wrap_px);
        // A transform rewrites its run's text, so it has to be applied before
        // the runs are concatenated — the ranges below index the transformed
        // string, which is also what `ShapedText::text` returns and therefore
        // what selection, find and accessibility all index.
        let resolved: Vec<TextStyle> = runs
            .iter()
            .map(|run| {
                let mut style = run.style.clone();
                style.resolve_from(base);
                style
            })
            .collect();
        let texts: Vec<Cow<'_, str>> = runs
            .iter()
            .zip(&resolved)
            .map(|(run, style)| style.transform.apply(run.text))
            .collect();
        let mut ranges = Vec::with_capacity(runs.len());
        let mut offset = 0;
        for text in &texts {
            ranges.push(offset..offset + text.len());
            offset += text.len();
        }
        let base_attrs = base.attrs();
        let spans: Vec<(&str, Attrs<'_>)> = texts
            .iter()
            .zip(&resolved)
            .enumerate()
            .map(|(index, (text, style))| {
                // The `index + 1` tagging is the convention `place_glyphs_styled`
                // decodes, shared with the editable path's color spans so the
                // two cannot drift. Metadata `0` means "the element's own paint".
                let mut attrs = style.attrs().metadata(index + 1);
                // Only pay for per-run metrics where a run actually resizes
                // the line; uniform-size runs keep the buffer's own metrics.
                if style.metrics() != base.metrics() {
                    attrs = attrs.metrics(style.metrics());
                }
                (text.as_ref(), attrs)
            })
            .collect();
        buffer.set_rich_text(spans, &base_attrs, Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut self.font_system, false);

        let metrics = measure(&buffer, base.metrics().line_height);
        ShapedText {
            buffer,
            metrics,
            runs: ranges,
        }
    }
}

/// The buffer setup shared by [`FontContext::shape`] and
/// [`FontContext::shape_runs`], so the single-run fast path and the styled-run
/// path cannot drift in how they wrap.
fn configure_buffer(buffer: &mut Buffer, style: &TextStyle, wrap_px: Option<f32>) {
    // The min-content probe's mode wins over the authored one: it is asking
    // "how narrow can this get while words stay whole", which `Wrap::None`
    // would answer with the full single-line width.
    if wrap_px == Some(0.0) {
        buffer.set_wrap(Wrap::Word);
    } else {
        buffer.set_wrap(style.wrap.as_cosmic());
    }
    buffer.set_tab_width(*style.tab_size);
    buffer.set_size(wrap_px, None);
}

/// One styled run of a multi-style text: the text, and the style it differs
/// from the surrounding text in.
///
/// The style is resolved against the base style at shaping time, so a run
/// states only what it changes — a code chip sets a family, not a whole font.
#[derive(Debug, Clone, Copy)]
pub struct TextRun<'a> {
    pub text: &'a str,
    pub style: &'a TextStyle,
}

impl Default for FontContext {
    fn default() -> Self {
        FontContext::new()
    }
}

/// Shaped, laid-out text ready to measure or rasterize. Holds the cosmic-text
/// buffer so [`place`](Self::place) can rasterize on demand.
pub struct ShapedText {
    buffer: Buffer,
    metrics: TextMetrics,
    /// Byte range of each styled run, indexing [`text`](Self::text). Empty for
    /// text shaped from a single style, which is the common case and pays
    /// nothing for this.
    runs: Vec<core::ops::Range<usize>>,
}

impl core::fmt::Debug for ShapedText {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShapedText")
            .field("metrics", &self.metrics)
            .finish_non_exhaustive()
    }
}

/// One segment of a positioned glyph outline, in logical pixels with Y downward.
///
/// A glyph can contain several closed subpaths, including counters (holes).
/// Preserve their winding when tessellating the outline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutlineCommand {
    /// Begin a subpath.
    MoveTo(Vector2),
    /// Continue with a straight segment.
    LineTo(Vector2),
    /// Continue with a quadratic Bézier: control point, then endpoint.
    QuadTo(Vector2, Vector2),
    /// Continue with a cubic Bézier: two control points, then endpoint.
    CurveTo(Vector2, Vector2, Vector2),
    /// Close the current subpath.
    Close,
}

/// The positioned scalable outline of one shaped glyph.
///
/// Commands retain curves and separate subpaths; a renderer chooses its own
/// flattening tolerance, tessellation, or extrusion depth.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphOutline {
    /// Positioned path commands, in the same logical coordinate system as text.
    pub commands: Vec<OutlineCommand>,
}

impl ShapedText {
    /// Extract scalable outlines using this text's already-shaped glyphs.
    ///
    /// Positions are logical pixels with the text's top-left at `origin` and Y
    /// increasing downward. Font fallback, ligatures, weight, and glyph offsets
    /// are the same as for [`place`](Self::place). Outlines are unhinted and
    /// positions are not pixel-snapped, so they remain suitable for scaling and
    /// extrusion. Whitespace and glyphs without scalable outlines are skipped.
    /// Color glyph outlines, when available, are returned as geometry only.
    ///
    /// Use the same [`FontContext`] that shaped the text. No second shaping pass
    /// or independent font lookup is performed.
    pub fn outlines(&self, ctx: &mut FontContext, origin: Vector2) -> Vec<GlyphOutline> {
        use cosmic_text::{CacheKeyFlags, Command};

        let mut outlines = Vec::new();
        for line in self.buffer.layout_runs() {
            for glyph in line.glyphs {
                let mut key = glyph.physical((0.0, 0.0), 1.0).cache_key;
                key.flags.insert(CacheKeyFlags::DISABLE_HINTING);
                let Some(commands) = ctx
                    .swash_cache
                    .get_outline_commands(&mut ctx.font_system, key)
                else {
                    continue;
                };
                if commands.is_empty() {
                    continue;
                }
                let x = origin.x + glyph.x + glyph.font_size * glyph.x_offset;
                let y = origin.y + line.line_y + glyph.y - glyph.font_size * glyph.y_offset;
                let point = |px: f32, py: f32| Vector2::new(x + px, y - py);
                let commands = commands
                    .iter()
                    .map(|command| match *command {
                        Command::MoveTo(p) => OutlineCommand::MoveTo(point(p.x, p.y)),
                        Command::LineTo(p) => OutlineCommand::LineTo(point(p.x, p.y)),
                        Command::QuadTo(c, p) => {
                            OutlineCommand::QuadTo(point(c.x, c.y), point(p.x, p.y))
                        }
                        Command::CurveTo(a, b, p) => OutlineCommand::CurveTo(
                            point(a.x, a.y),
                            point(b.x, b.y),
                            point(p.x, p.y),
                        ),
                        Command::Close => OutlineCommand::Close,
                    })
                    .collect();
                outlines.push(GlyphOutline { commands });
            }
        }
        outlines
    }

    /// Measured extent, for layout.
    pub fn metrics(&self) -> TextMetrics {
        self.metrics
    }

    /// Rasterize every glyph to 8-bit coverage and emit a [`GlyphRun`] with the
    /// text's top-left at `origin` (logical pixels). `scale` is the display's
    /// pixel density (physical pixels per logical pixel): glyphs are rasterized
    /// at that density so text is crisp on HiDPI, while placement stays logical.
    /// Whitespace and empty glyphs are skipped. Needs `ctx` because rasterization
    /// draws from its cache.
    ///
    /// The same character at the same size/font/weight/`scale` yields a stable
    /// [`PlacedGlyph::key`], so the backend caches each glyph in its atlas once.
    pub fn place(
        &self,
        ctx: &mut FontContext,
        color: impl Into<Paint>,
        origin: Vector2,
        scale: f32,
    ) -> GlyphRun {
        place_glyphs(ctx, &self.buffer, color.into(), origin, scale)
    }

    /// [`place`](Self::place), with each styled run painted in its own color.
    ///
    /// `run_paints` is indexed by run, parallel to
    /// [`run_ranges`](Self::run_ranges); a run with no entry falls back to
    /// `color`, which is also what unrun text uses.
    pub fn place_runs(
        &self,
        ctx: &mut FontContext,
        color: impl Into<Paint>,
        run_paints: &[Paint],
        origin: Vector2,
        scale: f32,
    ) -> GlyphRun {
        place_glyphs_styled(ctx, &self.buffer, color.into(), run_paints, origin, scale)
    }

    /// Byte range of each styled run, indexing [`text`](Self::text). Empty
    /// when the text was shaped from a single style.
    pub fn run_ranges(&self) -> &[core::ops::Range<usize>] {
        &self.runs
    }

    /// The shaped string, lines joined with `\n`. Every byte offset this type
    /// takes or returns indexes into it.
    pub fn text(&self) -> String {
        buffer_text(&self.buffer)
    }

    /// The byte offset nearest `point`, in buffer-local logical pixels (the
    /// text's top-left is the origin). `None` when the buffer has no lines to
    /// land on.
    ///
    /// This is the click half of selecting static text: the caller stores the
    /// offset as a selection anchor, then extends to a later `hit` as the
    /// pointer drags.
    pub fn hit(&self, point: Vector2) -> Option<usize> {
        self.buffer
            .hit(point.x, point.y)
            .map(|cursor| cursor_offset(&self.buffer, cursor))
    }

    /// The caret rect at `offset`, in buffer-local logical pixels.
    ///
    /// The offset is clamped to the shaped string and snapped to a character
    /// boundary, like the offsets accepted by [`selection_rects`](Self::selection_rects).
    pub fn caret_rect(&self, offset: usize) -> Rect {
        let cursor = offset_cursor(&self.buffer, offset);
        let (x, y) = self.buffer.cursor_position(&cursor).unwrap_or((0.0, 0.0));
        Rect::from_xywh(x.trunc(), y.trunc(), 1.0, self.metrics.line_height)
    }

    /// The rects covering the text between two byte offsets, one per visual row
    /// the selection touches, in buffer-local logical pixels. Callers translate
    /// by the text's origin, as they do for [`place`](Self::place).
    ///
    /// Takes the offsets in either order, like
    /// [`EditBuffer::set_selection_bytes`] — a selection is an anchor and a
    /// focus, and a drag that runs backwards needs no normalizing at the call
    /// site. Offsets are clamped to the text and snapped to character
    /// boundaries; equal offsets are a caret and cover nothing.
    pub fn selection_rects(&self, start: usize, end: usize) -> Vec<Rect> {
        let (start, end) = (start.min(end), start.max(end));
        if start == end {
            return Vec::new();
        }
        selection_rects(
            &self.buffer,
            offset_cursor(&self.buffer, start),
            offset_cursor(&self.buffer, end),
        )
    }

    /// The word enclosing `offset` — what a double-click selects. See
    /// `word_range` for how a between-words offset resolves.
    pub fn word_at(&self, offset: usize) -> core::ops::Range<usize> {
        word_range(&self.text(), offset)
    }

    /// The logical line enclosing `offset`, excluding its newline — what a
    /// triple-click selects. A wrapped paragraph counts as one line.
    pub fn line_at(&self, offset: usize) -> core::ops::Range<usize> {
        line_range(&self.text(), offset)
    }

    /// The decoration rules to fill for this text, in logical pixels with the
    /// text's top-left at `origin`. Empty unless the style turned one on.
    ///
    /// These are plain rects rather than part of the [`GlyphRun`]: a rule is a
    /// solid fill, so the caller draws it through the ordinary shape path
    /// instead of the glyph atlas. Draw them *after* the run — a strikethrough
    /// belongs over its glyphs.
    pub fn decorations(&self, origin: Vector2, scale: f32) -> Vec<Rect> {
        decoration_rects(&self.buffer, origin, scale)
    }
}

/// The filled rules for a laid-out buffer's text decorations (underline,
/// strikethrough, overline), in logical pixels with the text's top-left at
/// `origin`. Shared by static ([`ShapedText`]) and editable ([`EditBuffer`])
/// text, like [`place_glyphs`].
///
/// Geometry comes from cosmic-text: the layout pass emits a decoration span per
/// run carrying the font's own offset and thickness (in EM), so a rule sits
/// where the type designer put it rather than at a guessed fraction of the size.
/// The arithmetic mirrors cosmic-text's own `render_decoration` so Mosaic's
/// rules land exactly where its reference renderer would put them.
pub(crate) fn decoration_rects(buffer: &Buffer, origin: Vector2, scale: f32) -> Vec<Rect> {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let inv = 1.0 / scale;
    // Snap like `place_glyphs` does, and for the same reason: a rule that
    // rounds independently of the glyphs it belongs to shimmers against them.
    let origin_x = (origin.x * scale).round();
    let origin_y = (origin.y * scale).round();
    let mut rects = Vec::new();
    let mut push = |x: f32, y: f32, w: f32, thickness: f32| {
        let px = origin_x + (x * scale).round();
        let py = origin_y + (y * scale).round();
        // A rule thinner than one device pixel would vanish entirely.
        let pw = (w * scale).round();
        let ph = (thickness * scale).round().max(1.0);
        if pw >= 1.0 {
            rects.push(Rect::from_xywh(px * inv, py * inv, pw * inv, ph * inv));
        }
    };
    for run in buffer.layout_runs() {
        for span in run.decorations {
            let glyphs = &run.glyphs[span.glyph_range.clone()];
            if glyphs.is_empty() {
                continue;
            }
            // Min/max rather than first/last: an RTL run stores its glyphs in
            // right-to-left order, so the ends are not the extremes.
            let mut x_min = f32::INFINITY;
            let mut x_max = f32::NEG_INFINITY;
            for glyph in glyphs {
                x_min = x_min.min(glyph.x);
                x_max = x_max.max(glyph.x + glyph.w);
            }
            let width = x_max - x_min;
            if width <= 0.0 {
                continue;
            }
            let deco = &span.data;
            let size = span.font_size;
            let rule = |metrics: &cosmic_text::DecorationMetrics| {
                (metrics.thickness * size).max(1.0).ceil()
            };
            if deco.text_decoration.underline != UnderlineStyle::None {
                let thickness = rule(&deco.underline_metrics);
                let y = run.line_y - deco.underline_metrics.offset * size;
                push(x_min, y, width, thickness);
                if deco.text_decoration.underline == UnderlineStyle::Double {
                    push(x_min, y + thickness * 2.0, width, thickness);
                }
            }
            if deco.text_decoration.strikethrough {
                let thickness = rule(&deco.strikethrough_metrics);
                let y = run.line_y - deco.strikethrough_metrics.offset * size;
                push(x_min, y, width, thickness);
            }
            if deco.text_decoration.overline {
                // Overline reuses the underline thickness, clamped so a font
                // with a tall ascent cannot push the rule out of its line box.
                let thickness = rule(&deco.underline_metrics);
                let y = (run.line_y - deco.ascent * size).max(run.line_top);
                push(x_min, y, width, thickness);
            }
        }
    }
    rects
}

/// Rasterizes a laid-out buffer's glyphs into a [`GlyphRun`] — the placement
/// path shared by static ([`ShapedText`]) and editable ([`EditBuffer`]) text.
pub(crate) fn place_glyphs(
    ctx: &mut FontContext,
    buffer: &Buffer,
    color: Paint,
    origin: Vector2,
    scale: f32,
) -> GlyphRun {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let inv = 1.0 / scale;
    // Snap the run's origin to the physical pixel grid *once*. Each glyph's
    // physical offset is already integer (cosmic-text rounds pen positions),
    // so a whole-run integer origin puts every glyph on the device grid while
    // preserving the glyphs' exact relative positions. Snapping each glyph
    // independently instead would let neighbors round opposite ways and jitter.
    let origin_x = (origin.x * scale).round();
    let origin_y = (origin.y * scale).round();
    let mut run = GlyphRun::new();
    for line in buffer.layout_runs() {
        for glyph in line.glyphs {
            // Rasterize in physical pixels: cosmic-text scales the glyph and
            // bins the fractional pen position into the cache key.
            let physical = glyph.physical((0.0, line.line_y * scale), scale);
            // Cloned out of the cache so the borrow ends here: an `Arc` bump
            // and three copies, not a bitmap.
            let Some(RasterGlyph {
                key,
                image,
                left,
                top,
            }) = ctx.raster(physical.cache_key)
            else {
                continue;
            };
            let (w, h) = (image.width, image.height);
            // Glyph position in physical px (integer), relative to the snapped
            // origin, then back to logical for the destination quad. The
            // quad's edges land on whole device pixels, so atlas texels map
            // 1:1 to device pixels.
            let px_left = origin_x + (physical.x + left) as f32;
            let px_top = origin_y + (physical.y - top) as f32;
            let dest = Rect::from_xywh(px_left * inv, px_top * inv, w as f32 * inv, h as f32 * inv);
            run.glyphs.push(PlacedGlyph {
                key,
                image,
                dest,
                color: color.clone(),
            });
        }
    }
    run
}

pub(crate) fn place_glyphs_styled(
    ctx: &mut FontContext,
    buffer: &Buffer,
    color: Paint,
    colors: &[Paint],
    origin: Vector2,
    scale: f32,
) -> GlyphRun {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let mut metadata = Vec::new();
    for line in buffer.layout_runs() {
        for glyph in line.glyphs {
            let physical = glyph.physical((0.0, line.line_y * scale), scale);
            // Exactly the predicate `place_glyphs` uses to decide whether a
            // glyph yields a quad — shared rather than restated, so the
            // metadata stays aligned with the run it is zipped against below.
            // It reads the rasterization cache instead of asking the
            // rasterizer a second time for every glyph.
            if ctx.raster(physical.cache_key).is_some() {
                metadata.push(glyph.metadata);
            }
        }
    }
    let mut run = place_glyphs(ctx, buffer, color.clone(), origin, scale);
    for (glyph, metadata) in run.glyphs.iter_mut().zip(metadata) {
        if let Some(span) = metadata.checked_sub(1).and_then(|index| colors.get(index)) {
            glyph.color = span.clone();
        }
    }
    run
}

/// Measure a shaped buffer: widest line, total height, and the first line's
/// ascent/descent about its baseline.
pub(crate) fn measure(buffer: &Buffer, fallback_line_height: f32) -> TextMetrics {
    let mut width = 0.0f32;
    let mut bottom = 0.0f32;
    let mut line_height = fallback_line_height;
    let mut ascent = 0.0f32;
    let mut descent = 0.0f32;
    let mut first = true;
    for line in buffer.layout_runs() {
        width = width.max(line.line_w);
        bottom = bottom.max(line.line_top + line.line_height);
        if first {
            line_height = line.line_height;
            ascent = line.line_y - line.line_top;
            descent = (line.line_top + line.line_height) - line.line_y;
            first = false;
        }
    }
    TextMetrics {
        size: Size::new(width, bottom),
        line_height,
        ascent,
        descent,
    }
}

/// A buffer's lines joined with `\n` — the string byte offsets index into.
///
/// Shared by static ([`ShapedText`]) and editable ([`EditBuffer`]) text so both
/// answer offsets against the same string.
pub(crate) fn buffer_text(buffer: &Buffer) -> String {
    let mut out = String::new();
    for (i, line) in buffer.lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line.text());
    }
    out
}

/// The byte offset into [`buffer_text`] that `cursor` sits at.
pub(crate) fn cursor_offset(buffer: &Buffer, cursor: Cursor) -> usize {
    buffer
        .lines
        .iter()
        .take(cursor.line)
        .map(|line| line.text().len() + 1)
        .sum::<usize>()
        + cursor.index
}

/// The cursor at a byte offset into [`buffer_text`], clamped to the end and
/// snapped down to a character boundary. Snapping here rather than at the call
/// sites keeps a cursor from ever landing mid-character, which panics the
/// moment cosmic-text slices the line.
pub(crate) fn offset_cursor(buffer: &Buffer, offset: usize) -> Cursor {
    let mut remaining = offset;
    for (i, line) in buffer.lines.iter().enumerate() {
        let text = line.text();
        if remaining <= text.len() {
            while !text.is_char_boundary(remaining) {
                remaining -= 1;
            }
            return Cursor::new(i, remaining);
        }
        remaining -= text.len() + 1;
    }
    let last = buffer.lines.len().saturating_sub(1);
    Cursor::new(last, buffer.lines[last].text().len())
}

/// The rects covering `start..end`, one per visual row the range touches, in
/// buffer-local logical pixels. Callers translate by the text's origin.
///
/// Geometry comes from cosmic-text's own `highlight`, which walks the run's
/// grapheme clusters and accumulates by min/max rather than by end glyphs: a
/// bidirectional run stores its glyphs in visual order, so the selected span of
/// a mixed run is not the interval between its first and last selected glyph.
/// A run that interleaves directions yields several disjoint rects.
///
/// Shared by static ([`ShapedText`]) and editable ([`EditBuffer`]) text, like
/// [`decoration_rects`].
pub(crate) fn selection_rects(buffer: &Buffer, start: Cursor, end: Cursor) -> Vec<Rect> {
    let mut rects = Vec::new();
    for run in buffer.layout_runs() {
        // cosmic-text constrains byte bounds on the two endpoint lines, but a
        // run outside their logical line range differs from both endpoints
        // and would otherwise be highlighted in full.
        if run.line_i < start.line || run.line_i > end.line {
            continue;
        }
        for (x, width) in run.highlight(start, end) {
            rects.push(Rect::from_xywh(x, run.line_top, width, run.line_height));
        }
    }
    rects
}

/// The word enclosing `offset`, as a byte range into `text`.
///
/// Words are the Unicode ones (UAX #29), which is what cosmic-text resolves its
/// own word motions with, so selecting a word agrees with walking over it. An
/// offset between words expands over the whole run separating them instead, so
/// a double-click always resolves to something.
pub(crate) fn word_range(text: &str, offset: usize) -> core::ops::Range<usize> {
    let offset = offset.min(text.len());
    let mut previous_end = 0;
    for (start, word) in text.unicode_word_indices() {
        let end = start + word.len();
        if offset < start {
            // Landed in the gap that ended at this word.
            return previous_end..start;
        }
        if offset < end || (offset == end && end == text.len()) {
            return start..end;
        }
        previous_end = end;
    }
    previous_end..text.len()
}

/// The logical line enclosing `offset`, as a byte range into `text`, excluding
/// the terminating newline.
///
/// Logical, not visual: a wrapped paragraph is one line, which is what
/// selecting by line is expected to mean.
pub(crate) fn line_range(text: &str, offset: usize) -> core::ops::Range<usize> {
    let offset = offset.min(text.len());
    let start = text[..offset].rfind('\n').map_or(0, |i| i + 1);
    let end = text[offset..].find('\n').map_or(text.len(), |i| offset + i);
    start..end
}

/// Collapse a cosmic-text [`CacheKey`] to a stable 64-bit atlas key. The hasher
/// has fixed keys, so the same glyph maps to the same value across runs.
fn glyph_key(cache_key: &CacheKey) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cache_key.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn black() -> Color {
        Color::from_srgb8(0, 0, 0, 255)
    }

    /// Static and editable text must answer the same geometry for the same
    /// selection: a caret dragged out of an input and across a paragraph draws
    /// one continuous highlight only if both halves agree.
    fn assert_selection_agrees(text: &str, wrap: Option<f32>, start: usize, end: usize) {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(16.0);

        let shaped = ctx.shape(text, &style, wrap);
        let from_shaped = shaped.selection_rects(start, end);

        let mut editor = EditBuffer::new(style);
        editor.set_text(&mut ctx, text);
        editor.set_wrap_width(&mut ctx, wrap);
        editor.set_selection_bytes(&mut ctx, start, end);
        let from_editor = editor.selection_rects(&mut ctx);

        assert_eq!(
            from_shaped, from_editor,
            "selection {start}..{end} of {text:?} (wrap {wrap:?}) disagrees"
        );
        assert!(
            !from_shaped.is_empty(),
            "selection {start}..{end} of {text:?} produced no rects, so agreement proves nothing"
        );
    }

    #[test]
    fn a_line_height_multiple_follows_the_font_size() {
        let scaled = TextStyle::new(20.0).line_height(LineHeight::Scale(1.5));
        assert_eq!(scaled.resolved_line_height(), 30.0);
        let inherited_larger = TextStyle {
            size_px: Inherit::set(40.0),
            ..scaled
        };
        assert_eq!(inherited_larger.resolved_line_height(), 60.0);
    }

    #[test]
    fn an_absolute_line_height_ignores_the_font_size() {
        let absolute = TextStyle::new(20.0).line_height(LineHeight::Px(24.0));
        assert_eq!(absolute.resolved_line_height(), 24.0);
        let inherited_larger = TextStyle {
            size_px: Inherit::set(40.0),
            ..absolute
        };
        assert_eq!(inherited_larger.resolved_line_height(), 24.0);
        assert_eq!(Length::px(24.0).line_height(), LineHeight::Px(24.0));
        assert_eq!(1.5f32.line_height(), LineHeight::Scale(1.5));
    }

    /// Leading tighter than the size is legitimate typography, and only the
    /// shaper's zero-height abort is guarded against.
    #[test]
    fn leading_tighter_than_the_size_reaches_the_shaper() {
        let tight = TextStyle::new(20.0).line_height(LineHeight::Scale(0.75));
        assert_eq!(tight.metrics().line_height, 15.0);
        let degenerate = TextStyle::new(20.0).line_height(LineHeight::Px(0.0));
        assert_eq!(degenerate.metrics().line_height, MIN_FONT_SIZE);
    }

    #[test]
    fn static_and_editable_selection_geometry_agree() {
        assert_selection_agrees("Hello, world", None, 0, 5);
        assert_selection_agrees("Hello, world", None, 7, 12);
        // Across a hard line break: two rows, so two rects.
        assert_selection_agrees("first line\nsecond line", None, 3, 15);
    }

    #[test]
    fn static_and_editable_caret_geometry_agree() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(16.0);
        let text = "one\ntwo";
        let shaped = ctx.shape(text, &style, None);
        let mut editor = EditBuffer::new(style);
        editor.set_text(&mut ctx, text);

        for offset in [0, 2, 4, text.len()] {
            editor.set_selection_bytes(&mut ctx, offset, offset);
            assert_eq!(shaped.caret_rect(offset), editor.caret_rect(&mut ctx));
        }
    }

    #[test]
    fn static_and_editable_selection_geometry_agree_when_wrapped() {
        // Narrow enough that the run breaks into several visual rows, which is
        // where a per-row rect walk can drift from a per-line one.
        let text = "the quick brown fox jumps over the lazy dog";
        assert_selection_agrees(text, Some(80.0), 0, text.len());
        assert_selection_agrees(text, Some(80.0), 4, 30);
    }

    #[test]
    fn static_and_editable_selection_geometry_agree_for_a_right_to_left_run() {
        // Hebrew: DejaVu Sans covers it, and bidi puts the run in visual order
        // reversed from its bytes, so the selected span is not the interval
        // between the first and last selected glyph.
        let hebrew = "שלום עולם";
        assert_selection_agrees(hebrew, None, 0, hebrew.len());
        // A mixed run, where a single row can need several disjoint rects.
        let mixed = "start שלום end";
        assert_selection_agrees(mixed, None, 0, mixed.len());
        assert_selection_agrees(mixed, None, 3, 12);
    }

    #[test]
    fn a_reversed_selection_range_reads_the_same_as_a_forward_one() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("Hello, world", &TextStyle::new(16.0), None);
        assert_eq!(shaped.selection_rects(2, 9), shaped.selection_rects(9, 2));
    }

    #[test]
    fn an_empty_selection_covers_nothing() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("Hello, world", &TextStyle::new(16.0), None);
        assert!(shaped.selection_rects(4, 4).is_empty());
    }

    #[test]
    fn a_multiline_selection_does_not_cover_lines_outside_its_endpoints() {
        let mut ctx = FontContext::embedded_only();
        let text = "before\nfirst selected\nsecond selected\nafter";
        let shaped = ctx.shape(text, &TextStyle::new(16.0), None);
        let start = text.find("selected").unwrap() + 2;
        let end = text.rfind("selected").unwrap() + 4;

        let rects = shaped.selection_rects(start, end);
        assert_eq!(rects.len(), 2, "only the two endpoint lines are selected");
        assert!(rects[0].origin.x > 0.0, "the first line starts partway in");
        assert_eq!(rects[1].origin.x, 0.0, "the second line starts at its edge");
        assert_ne!(rects[0].origin.y, rects[1].origin.y);
    }

    /// Offsets reaching the shaper must never land mid-character; the range is
    /// clamped and snapped rather than trusted.
    #[test]
    fn selection_offsets_are_clamped_and_snapped_to_character_boundaries() {
        let mut ctx = FontContext::embedded_only();
        // Each of these characters is multiple bytes wide.
        let text = "héllo wörld";
        let shaped = ctx.shape(text, &TextStyle::new(16.0), None);
        assert!(!shaped.selection_rects(0, usize::MAX).is_empty());
        // Byte 2 is inside the 'é'.
        assert!(!shaped.selection_rects(2, 6).is_empty());
    }

    #[test]
    fn hitting_the_start_and_end_of_a_run_yields_the_end_offsets() {
        let mut ctx = FontContext::embedded_only();
        let text = "Hello";
        let shaped = ctx.shape(text, &TextStyle::new(16.0), None);
        let height = shaped.metrics().line_height;
        let width = shaped.metrics().size.width;
        assert_eq!(shaped.hit(Vector2::new(0.0, height / 2.0)), Some(0));
        assert_eq!(
            shaped.hit(Vector2::new(width + 20.0, height / 2.0)),
            Some(text.len())
        );
    }

    #[test]
    fn a_hit_round_trips_through_the_selection_it_anchors() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("Hello, world", &TextStyle::new(16.0), None);
        let height = shaped.metrics().line_height;
        let offset = shaped
            .hit(Vector2::new(
                shaped.metrics().size.width / 2.0,
                height / 2.0,
            ))
            .expect("a point inside the run hits");
        assert!(offset > 0 && offset < 12, "unexpected offset {offset}");
        assert!(!shaped.selection_rects(0, offset).is_empty());
    }

    #[test]
    fn text_joins_the_buffer_lines_that_offsets_index_into() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("first\nsecond", &TextStyle::new(16.0), None);
        assert_eq!(shaped.text(), "first\nsecond");
    }

    #[test]
    fn a_word_selection_covers_the_word_under_the_offset() {
        let text = "the quick brown fox";
        assert_eq!(word_range(text, 0), 0..3);
        assert_eq!(word_range(text, 2), 0..3);
        // The offset at a word's trailing edge belongs to that word.
        assert_eq!(word_range(text, 6), 4..9);
        assert_eq!(word_range(text, 16), 16..19);
        // Past the last word.
        assert_eq!(word_range(text, text.len()), 16..19);
    }

    /// A double-click has to resolve to something wherever it lands, so an
    /// offset between words takes the run separating them.
    #[test]
    fn a_word_selection_between_words_covers_the_separator_run() {
        let text = "one   two";
        assert_eq!(word_range(text, 4), 3..6);
        // The whole separator, punctuation and the spaces flanking it alike.
        let punctuated = "a -- b";
        assert_eq!(word_range(punctuated, 2), 1..5);
    }

    #[test]
    fn a_line_selection_covers_the_logical_line_without_its_newline() {
        let text = "first\nsecond\nthird";
        assert_eq!(line_range(text, 0), 0..5);
        assert_eq!(line_range(text, 5), 0..5);
        assert_eq!(line_range(text, 6), 6..12);
        assert_eq!(line_range(text, text.len()), 13..18);
    }

    /// Selecting by line means the paragraph, not the visual row it wrapped
    /// onto — the offsets are logical throughout.
    #[test]
    fn a_line_selection_spans_a_wrapped_paragraph() {
        let mut ctx = FontContext::embedded_only();
        let text = "the quick brown fox jumps over the lazy dog";
        let shaped = ctx.shape(text, &TextStyle::new(16.0), Some(80.0));
        let line = shaped.line_at(20);
        assert_eq!(line, 0..text.len());
        assert!(
            shaped.selection_rects(line.start, line.end).len() > 1,
            "a wrapped paragraph should highlight across several rows"
        );
    }

    #[test]
    fn every_font_prefixed_property_inherits() {
        let red = Color::from_srgb8(255, 0, 0, 255);
        let defaults = TextStyle::inherited();
        assert!(defaults.color.is_inherit());
        assert!(defaults.family.is_inherit());
        assert!(defaults.size_px.is_inherit());
        assert!(defaults.stretch.is_inherit());
        assert!(defaults.style.is_inherit());
        assert!(defaults.weight.is_inherit());

        // Typography properties outside the literal `font-*` surface retain
        // their existing local defaults.
        assert!(!defaults.letter_spacing_px.is_inherit());
        assert!(!defaults.wrap.is_inherit());
        assert!(!defaults.transform.is_inherit());
        assert!(!defaults.underline.is_inherit());
        assert!(!defaults.strikethrough.is_inherit());
        assert!(!defaults.overline.is_inherit());

        let parent = TextStyle::new(24.0)
            .color(red)
            .family(FontFamily::Monospace)
            .stretch(FontStretch::Condensed)
            .style(FontStyle::Italic)
            .weight(700);
        let mut child = TextStyle::inherited();
        child.resolve_from(&parent);

        assert_eq!(*child.color, PaintSpec::solid(red));
        assert_eq!(*child.family, FontFamily::Monospace);
        assert_eq!(*child.size_px, 24.0);
        assert_eq!(*child.stretch, FontStretch::Condensed);
        assert_eq!(*child.style, FontStyle::Italic);
        assert_eq!(*child.weight, 700);
    }

    /// A zero font size is reachable from app code — an uninstalled scheme
    /// token falls back to `0.0` — and cosmic-text answers that with a
    /// non-unwinding abort. Shaping must survive it.
    #[test]
    fn a_zero_font_size_shapes_instead_of_aborting() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("Hello", &TextStyle::new(0.0), None);
        assert!(shaped.metrics().line_height > 0.0);
    }

    #[test]
    fn letter_spacing_widens_the_run_and_is_measured_in_pixels() {
        let mut ctx = FontContext::embedded_only();
        let plain = ctx.shape("Hello", &TextStyle::new(20.0), None).metrics();
        let tracked = ctx
            .shape("Hello", &TextStyle::new(20.0).letter_spacing(4.0), None)
            .metrics();
        // Five glyphs, four (or five, depending on trailing advance) gaps of
        // 4px — assert the direction and rough magnitude rather than an exact
        // figure, which depends on where the shaper applies the last advance.
        let grew = tracked.size.width - plain.size.width;
        assert!(
            (12.0..=20.0).contains(&grew),
            "expected ~4px per gap, grew by {grew}"
        );
        let tightened = ctx
            .shape("Hello", &TextStyle::new(20.0).letter_spacing(-1.0), None)
            .metrics();
        assert!(tightened.size.width < plain.size.width);
    }

    #[test]
    fn text_transform_changes_the_shaped_string() {
        let mut ctx = FontContext::embedded_only();
        let lower = ctx.shape("hi there", &TextStyle::new(16.0), None).metrics();
        let upper = ctx
            .shape(
                "hi there",
                &TextStyle::new(16.0).transform(TextTransform::Uppercase),
                None,
            )
            .metrics();
        // Capitals are wider than lowercase in DejaVu Sans.
        assert!(
            upper.size.width > lower.size.width,
            "{} vs {}",
            upper.size.width,
            lower.size.width
        );
    }

    #[test]
    fn capitalize_preserves_original_spacing() {
        assert_eq!(
            TextTransform::Capitalize.apply("hello  wide world"),
            "Hello  Wide World"
        );
        assert_eq!(TextTransform::Capitalize.apply(""), "");
        assert_eq!(TextTransform::Capitalize.apply("  x"), "  X");
    }

    #[test]
    fn wrap_none_keeps_one_line_past_the_wrap_width() {
        let mut ctx = FontContext::embedded_only();
        let text = "the quick brown fox jumps over the lazy dog";
        let wrapped = ctx.shape(text, &TextStyle::new(16.0), Some(80.0)).metrics();
        let unwrapped = ctx
            .shape(text, &TextStyle::new(16.0).wrap(TextWrap::None), Some(80.0))
            .metrics();
        assert!(
            wrapped.size.height > unwrapped.size.height,
            "wrapping should add lines: {} vs {}",
            wrapped.size.height,
            unwrapped.size.height
        );
        // `None` overflows its box rather than breaking.
        assert!(unwrapped.size.width > 80.0);
    }

    /// The min-content probe asks how narrow the text can get with words
    /// intact. An authored `wrap:none` must not answer that with the full
    /// single-line width, or a `text` node would never shrink.
    #[test]
    fn wrap_none_does_not_defeat_the_min_content_probe() {
        let mut ctx = FontContext::embedded_only();
        let text = "the quick brown fox";
        let probe = ctx
            .shape(text, &TextStyle::new(16.0).wrap(TextWrap::None), Some(0.0))
            .metrics();
        let full = ctx
            .shape(text, &TextStyle::new(16.0).wrap(TextWrap::None), None)
            .metrics();
        assert!(
            probe.size.width < full.size.width,
            "probe {} should be the longest word, not the whole line {}",
            probe.size.width,
            full.size.width
        );
    }

    #[test]
    fn tab_size_scales_the_tab_stop() {
        let mut ctx = FontContext::embedded_only();
        let narrow = ctx
            .shape("a\tb", &TextStyle::new(16.0).tab_size(2), None)
            .metrics();
        let wide = ctx
            .shape("a\tb", &TextStyle::new(16.0).tab_size(16), None)
            .metrics();
        assert!(
            wide.size.width > narrow.size.width,
            "{} vs {}",
            wide.size.width,
            narrow.size.width
        );
    }

    /// Every shaping-relevant field must be visible to `same_shaping`, or the
    /// paint walk will reuse a stale measurement when only that field changed.
    #[test]
    fn shaping_comparison_covers_every_shaping_field() {
        let base = TextStyle::new(16.0);
        for (label, changed) in [
            ("style", base.clone().style(FontStyle::Italic)),
            ("stretch", base.clone().stretch(FontStretch::Condensed)),
            ("letter_spacing", base.clone().letter_spacing(2.0)),
            ("wrap", base.clone().wrap(TextWrap::None)),
            ("tab_size", base.clone().tab_size(4)),
            (
                "transform",
                base.clone().transform(TextTransform::Uppercase),
            ),
        ] {
            assert!(
                !base.same_shaping(&changed),
                "`{label}` must count as a shaping change"
            );
            assert!(
                !base.same_authored_shaping(&changed),
                "`{label}` must count as an authored shaping change"
            );
        }
    }

    #[test]
    fn embedded_font_shapes_and_measures() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(16.0);
        let shaped = ctx.shape("Hello", &style, None);
        let m = shaped.metrics();
        // "Hello" at 16px must have positive extent and a sane line height.
        assert!(m.size.width > 0.0, "width was {}", m.size.width);
        assert!(m.size.height > 0.0, "height was {}", m.size.height);
        assert!((m.line_height - 16.0 * 1.3).abs() < 0.001);
        assert!(m.ascent > 0.0 && m.descent > 0.0);
    }

    #[test]
    fn place_emits_one_glyph_per_visible_char() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(32.0);
        // Three visible glyphs, one space (space rasterizes empty -> skipped).
        let shaped = ctx.shape("a b", &style, None);
        let run = shaped.place(&mut ctx, black(), Vector2::new(0.0, 0.0), 1.0);
        assert_eq!(run.glyphs.len(), 2, "expected 'a' and 'b', got {:?}", run);
        for g in &run.glyphs {
            assert!(g.image.width > 0 && g.image.height > 0);
            assert_eq!(
                g.image.data.len(),
                (g.image.width * g.image.height) as usize
            );
        }
    }

    #[test]
    fn hidpi_rasterizes_larger_bitmap_at_same_logical_size() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(32.0);
        let g1 = {
            let r = ctx
                .shape("a", &style, None)
                .place(&mut ctx, black(), Vector2::ZERO, 1.0);
            r.glyphs[0].clone()
        };
        let g2 = {
            let r = ctx
                .shape("a", &style, None)
                .place(&mut ctx, black(), Vector2::ZERO, 2.0);
            r.glyphs[0].clone()
        };
        // At 2x the coverage bitmap is ~twice as tall (physical texels)...
        assert!(
            g2.image.height >= g1.image.height * 2 - 2,
            "2x bitmap {} vs 1x {}",
            g2.image.height,
            g1.image.height
        );
        // ...but the logical destination stays about the same size.
        assert!((g2.dest.size.height - g1.dest.size.height).abs() < 2.0);
    }

    #[test]
    fn keys_are_stable_across_placements() {
        // The atlas caches by key, so the same glyph at the same size and
        // subpixel position must yield the same key every time it is placed —
        // otherwise the atlas would re-upload identical bitmaps forever.
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(24.0);
        let at = Vector2::new(0.0, 0.0);
        let a = ctx
            .shape("mosaic", &style, None)
            .place(&mut ctx, black(), at, 1.0);
        let b = ctx
            .shape("mosaic", &style, None)
            .place(&mut ctx, black(), at, 1.0);
        let ka: Vec<u64> = a.glyphs.iter().map(|g| g.key).collect();
        let kb: Vec<u64> = b.glyphs.iter().map(|g| g.key).collect();
        assert!(!ka.is_empty());
        assert_eq!(ka, kb);
    }

    #[test]
    fn empty_text_is_empty_run() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(16.0);
        let run =
            ctx.shape("   ", &style, None)
                .place(&mut ctx, black(), Vector2::new(0.0, 0.0), 1.0);
        assert!(run.is_empty());
    }

    #[test]
    fn wrapping_increases_height() {
        let mut ctx = FontContext::embedded_only();
        let style = TextStyle::new(16.0);
        let text = "the quick brown fox jumps over the lazy dog";
        let one_line = ctx.shape(text, &style, None).metrics().size.height;
        let wrapped = ctx.shape(text, &style, Some(80.0)).metrics().size.height;
        assert!(wrapped > one_line, "wrapped {wrapped} vs {one_line}");
    }
    #[test]
    fn outlines_preserve_curved_outer_and_inner_contours() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("O", &TextStyle::new(32.0), None);
        let outlines = shaped.outlines(&mut ctx, Vector2::ZERO);
        assert_eq!(outlines.len(), 1);
        let commands = &outlines[0].commands;
        assert_eq!(
            commands
                .iter()
                .filter(|c| matches!(c, OutlineCommand::MoveTo(_)))
                .count(),
            2
        );
        assert_eq!(
            commands
                .iter()
                .filter(|c| matches!(c, OutlineCommand::Close))
                .count(),
            2
        );
        assert!(
            commands
                .iter()
                .any(|c| matches!(c, OutlineCommand::QuadTo(..) | OutlineCommand::CurveTo(..)))
        );
    }

    #[test]
    fn outlines_skip_whitespace_and_follow_shaped_spacing() {
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("O O", &TextStyle::new(32.0), None);
        let outlines = shaped.outlines(&mut ctx, Vector2::ZERO);
        assert_eq!(outlines.len(), 2);
        let glyphs = shaped.buffer.layout_runs().next().unwrap().glyphs;
        let expected = glyphs.last().unwrap().x - glyphs.first().unwrap().x;
        let OutlineCommand::MoveTo(a) = outlines[0].commands[0] else {
            panic!("missing first contour")
        };
        let OutlineCommand::MoveTo(b) = outlines[1].commands[0] else {
            panic!("missing second contour")
        };
        assert!((b.x - a.x - expected).abs() < 0.001);
        assert!((b.y - a.y).abs() < 0.001);
        assert!(
            ctx.shape("   ", &TextStyle::new(32.0), None)
                .outlines(&mut ctx, Vector2::ZERO)
                .is_empty()
        );
    }

    #[test]
    fn outlines_translate_all_curve_points_without_reshaping() {
        fn points(command: &OutlineCommand) -> Vec<Vector2> {
            match *command {
                OutlineCommand::MoveTo(p) | OutlineCommand::LineTo(p) => vec![p],
                OutlineCommand::QuadTo(c, p) => vec![c, p],
                OutlineCommand::CurveTo(a, b, p) => vec![a, b, p],
                OutlineCommand::Close => Vec::new(),
            }
        }
        let mut ctx = FontContext::embedded_only();
        let shaped = ctx.shape("B8", &TextStyle::new(32.0), None);
        let initial = shaped.outlines(&mut ctx, Vector2::ZERO);
        let offset = Vector2::new(23.25, -17.5);
        let moved = shaped.outlines(&mut ctx, offset);
        assert_eq!(initial.len(), moved.len());
        assert!(!initial.is_empty());
        for (a, b) in initial.iter().zip(&moved) {
            assert_eq!(a.commands.len(), b.commands.len());
            for (a, b) in a.commands.iter().zip(&b.commands) {
                let a = points(a);
                let b = points(b);
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(&b) {
                    assert!((b.x - a.x - offset.x).abs() < 0.001);
                    assert!((b.y - a.y - offset.y).abs() < 0.001);
                }
            }
        }
    }
}
