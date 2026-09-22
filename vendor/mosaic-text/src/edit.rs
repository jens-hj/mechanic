//! [`EditBuffer`] — editable text: caret, selection, hit-testing, and
//! placement, on top of cosmic-text's editor. Nothing outside this crate
//! names cosmic-text; this module translates Mosaic's editing vocabulary
//! ([`CaretMotion`], logical-pixel points and rects) into it.
//!
//! The buffer is line-based and fully multi-line: inserting `"\n"` makes a
//! line, [`CaretMotion::Up`]/[`Down`](CaretMotion::Down) walk visual lines,
//! and geometry (caret rect, selection rects) is reported per line. A
//! single-line widget is a policy choice layered above — strip newlines and
//! ignore the vertical motions.
//!
//! Coordinates are logical pixels relative to the text's top-left corner;
//! widgets translate to and from their own rects.

use cosmic_text::{
    Action, Attrs, AttrsList, Buffer, Cursor, Edit, Editor, Motion, Selection, Shaping,
};
use mosaic_core::{Color, Rect, Vector2};
use mosaic_render::{GlyphRun, Paint};

use crate::{FontContext, TextMetrics, TextStyle, measure, place_glyphs, place_glyphs_styled};

/// A foreground color applied to a byte range in an [`EditBuffer`].
///
/// Ranges index the committed UTF-8 text, are snapped to character boundaries,
/// and are repaired as text is inserted or removed. Overlapping spans use the
/// later span, matching painter order.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorSpan {
    pub range: core::ops::Range<usize>,
    pub color: Color,
}

impl ColorSpan {
    pub fn new(range: core::ops::Range<usize>, color: Color) -> Self {
        Self { range, color }
    }
}

/// Cached logical-line information for editor chrome and status displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineMetadata {
    pub index: usize,
    pub start: usize,
    pub end: usize,
}

/// A caret movement, applied with or without extending the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaretMotion {
    /// One grapheme left.
    Left,
    /// One grapheme right.
    Right,
    /// To the previous word boundary.
    WordLeft,
    /// To the next word boundary.
    WordRight,
    /// To the start of the current line.
    LineStart,
    /// To the end of the current line.
    LineEnd,
    /// One visual line up.
    Up,
    /// One visual line down.
    Down,
    /// To the start of the text.
    TextStart,
    /// To the end of the text.
    TextEnd,
}

impl CaretMotion {
    fn as_cosmic(self) -> Motion {
        match self {
            CaretMotion::Left => Motion::Left,
            CaretMotion::Right => Motion::Right,
            CaretMotion::WordLeft => Motion::LeftWord,
            CaretMotion::WordRight => Motion::RightWord,
            CaretMotion::LineStart => Motion::Home,
            CaretMotion::LineEnd => Motion::End,
            CaretMotion::Up => Motion::Up,
            CaretMotion::Down => Motion::Down,
            CaretMotion::TextStart => Motion::BufferStart,
            CaretMotion::TextEnd => Motion::BufferEnd,
        }
    }
}

/// Editable, shaped text with a caret and a selection.
///
/// Mutations take the [`FontContext`] because editing re-shapes the changed
/// lines; geometry queries take it for the same reason (they answer from
/// the current shaping). Create one per editable field and keep it — the
/// shaping cache lives in the buffer across edits.
///
/// Composition (IME preedit) lives in the buffer too: the in-flight text is
/// inserted so it shapes and measures like anything else, but it is tracked
/// as a range and excluded from [`text`](Self::text) — the logical value
/// never contains half-composed input. See [`set_preedit`](Self::set_preedit).
pub struct EditBuffer {
    editor: Editor<'static>,
    style: TextStyle,
    wrap_width: Option<f32>,
    preedit: Option<PreeditRange>,
    color_spans: Vec<ColorSpan>,
}

/// Where the composition text sits in the buffer. Valid only while no other
/// mutation intervenes — the composition methods and [`EditBuffer::insert`]
/// maintain it; widgets hold other edits off while composing.
#[derive(Clone, Copy)]
struct PreeditRange {
    start: Cursor,
    end: Cursor,
}

impl core::fmt::Debug for EditBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EditBuffer")
            .field("style", &self.style)
            .finish_non_exhaustive()
    }
}

impl EditBuffer {
    /// An empty buffer shaping with `style`, unwrapped (lines run as wide
    /// as their content; see [`set_wrap_width`](Self::set_wrap_width)).
    pub fn new(style: TextStyle) -> Self {
        let mut buffer = EditBuffer {
            editor: Editor::new(Buffer::new_empty(style.metrics())),
            style,
            wrap_width: None,
            preedit: None,
            color_spans: Vec::new(),
        };
        // The editor's line operations index into the line list assuming at
        // least one line exists, which a raw empty buffer violates;
        // `set_text` establishes the empty line, and no edit removes it.
        let attrs = attrs_for(&buffer.style);
        buffer
            .editor
            .with_buffer_mut(|inner| inner.set_text("", &attrs, Shaping::Advanced, None));
        buffer
    }

    /// Applies a new effective shaping style without disturbing text, caret,
    /// selection, or composition state.
    pub fn set_style(&mut self, ctx: &mut FontContext, style: TextStyle) {
        if self.style.same_shaping(&style) {
            self.style = style;
            return;
        }
        self.style = style;
        let metrics = self.style.metrics();
        let attrs = attrs_for(&self.style);
        self.editor.with_buffer_mut(|buffer| {
            buffer.set_metrics(metrics);
            for line in &mut buffer.lines {
                line.set_attrs_list(AttrsList::new(&attrs));
            }
        });
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Replaces the whole text; the caret moves to the end, unselected.
    /// How a widget applies an external (programmatic) value change.
    pub fn set_text(&mut self, ctx: &mut FontContext, text: &str) {
        let before = self.text();
        self.preedit = None;
        let attrs = attrs_for(&self.style);
        self.editor
            .with_buffer_mut(|buffer| buffer.set_text(text, &attrs, Shaping::Advanced, None));
        self.editor.set_selection(Selection::None);
        self.editor
            .action(&mut ctx.font_system, Action::Motion(Motion::BufferEnd));
        self.repair_color_spans(&before, text);
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Replaces syntax-color spans without changing text or selection.
    pub fn set_color_spans(&mut self, ctx: &mut FontContext, spans: Vec<ColorSpan>) {
        let text = self.text();
        let spans: Vec<_> = spans
            .into_iter()
            .filter_map(|mut span| {
                span.range.start = snap_down(&text, span.range.start.min(text.len()));
                span.range.end = snap_down(&text, span.range.end.min(text.len()));
                (span.range.start < span.range.end).then_some(span)
            })
            .collect();
        if self.color_spans == spans {
            return;
        }
        self.color_spans = spans;
        self.apply_color_spans();
        self.sync(ctx);
    }

    pub fn color_spans(&self) -> &[ColorSpan] {
        &self.color_spans
    }

    /// One entry per logical line, including the final empty line.
    pub fn lines(&self) -> Vec<LineMetadata> {
        let text = self.text();
        let mut start = 0;
        let mut lines = Vec::new();
        for (index, line) in text.split('\n').enumerate() {
            let end = start + line.len();
            lines.push(LineMetadata { index, start, end });
            start = end + 1;
        }
        lines
    }

    /// One-based logical line and Unicode-scalar column at the caret.
    pub fn caret_line_column(&self) -> (usize, usize) {
        let text = self.display_text();
        let caret = self.selection_bytes().1.min(text.len());
        let before = &text[..caret];
        let line = before.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let column = before
            .rsplit_once('\n')
            .map_or(before, |(_, tail)| tail)
            .chars()
            .count()
            + 1;
        (line, column)
    }

    /// The current text, lines joined with `\n`. Composition text is
    /// excluded — it only becomes part of the value when committed
    /// ([`insert`](Self::insert)).
    pub fn text(&self) -> String {
        let mut out = self.display_text();
        if let Some(range) = self.preedit {
            out.replace_range(
                self.cursor_offset(range.start)..self.cursor_offset(range.end),
                "",
            );
        }
        out
    }

    /// The buffer's full contents as displayed — [`text`](Self::text) with
    /// any in-flight composition still in place. What a buffer-mirroring
    /// input method sees; the byte offsets of
    /// [`selection_bytes`](Self::selection_bytes),
    /// [`preedit_bytes`](Self::preedit_bytes), and
    /// [`set_selection_bytes`](Self::set_selection_bytes) index into this
    /// string.
    pub fn display_text(&self) -> String {
        self.editor.with_buffer(crate::buffer_text)
    }

    /// The selection as byte offsets into [`display_text`](Self::display_text)
    /// (`start <= end`; the caret when they are equal).
    pub fn selection_bytes(&self) -> (usize, usize) {
        match self.editor.selection_bounds() {
            Some((start, end)) => (self.cursor_offset(start), self.cursor_offset(end)),
            None => {
                let caret = self.cursor_offset(self.editor.cursor());
                (caret, caret)
            }
        }
    }

    /// The caret's byte offset into [`display_text`](Self::display_text).
    ///
    /// Unlike the second value returned by [`selection_bytes`](Self::selection_bytes),
    /// this preserves which edge is the moving end of a backwards selection.
    pub fn caret_offset(&self) -> usize {
        self.cursor_offset(self.editor.cursor())
    }

    /// The composition's byte range into
    /// [`display_text`](Self::display_text), while one is active.
    pub fn preedit_bytes(&self) -> Option<(usize, usize)> {
        self.preedit.map(|range| {
            (
                self.cursor_offset(range.start),
                self.cursor_offset(range.end),
            )
        })
    }

    /// Places the caret and selection at byte offsets into
    /// [`display_text`](Self::display_text), clamped to the text and snapped
    /// to character boundaries; equal offsets collapse to a caret. How a
    /// buffer-mirroring input method moves the caret.
    pub fn set_selection_bytes(&mut self, ctx: &mut FontContext, start: usize, end: usize) {
        let text = self.display_text();
        let snap = |offset: usize| {
            let mut offset = offset.min(text.len());
            while !text.is_char_boundary(offset) {
                offset -= 1;
            }
            offset
        };
        let (start, end) = (snap(start.min(end)), snap(start.max(end)));
        if start == end {
            self.editor.set_selection(Selection::None);
        } else {
            self.editor
                .set_selection(Selection::Normal(self.offset_cursor(start)));
        }
        self.editor.set_cursor(self.offset_cursor(end));
        self.sync(ctx);
    }

    /// Wrap width in logical pixels; `None` (the default) never wraps.
    pub fn set_wrap_width(&mut self, ctx: &mut FontContext, width: Option<f32>) {
        if self.wrap_width == width {
            return;
        }
        self.wrap_width = width;
        self.editor.with_buffer_mut(|buffer| {
            buffer.set_size(width, None);
        });
        self.sync(ctx);
    }

    /// Inserts at the caret, replacing the selection if there is one. Any
    /// in-flight composition is discarded first — an insert *is* the commit,
    /// so a platform that skips the empty closing preedit still ends clean.
    pub fn insert(&mut self, ctx: &mut FontContext, text: &str) {
        let before = self.text();
        self.remove_preedit_text();
        self.editor.insert_string(text, None);
        self.repair_color_spans(&before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Replaces the composition text at the caret. The first composition of
    /// a sequence replaces the selection, like an insert; each following
    /// call replaces the previous composition text. `cursor` is the caret's
    /// byte range within `text` (`None` when the input method hides the
    /// caret); the composition ends with an empty `text` or
    /// [`clear_preedit`](Self::clear_preedit), and its text never enters
    /// [`text`](Self::text).
    pub fn set_preedit(
        &mut self,
        ctx: &mut FontContext,
        text: &str,
        cursor: Option<(usize, usize)>,
    ) {
        if !self.remove_preedit_text() && self.has_selection() {
            self.editor.delete_selection();
        }
        if text.is_empty() {
            self.sync(ctx);
            return;
        }
        let start = self.editor.cursor();
        self.editor.insert_string(text, None);
        let end = self.editor.cursor();
        self.preedit = Some(PreeditRange { start, end });
        if let Some((caret, _)) = cursor {
            let offset = self.cursor_offset(start) + caret.min(text.len());
            let caret = self.offset_cursor(offset);
            self.editor.set_cursor(caret);
        }
        self.sync(ctx);
    }

    /// Ends the composition, removing its text; the caret returns to where
    /// the composition began. A no-op when none is active.
    pub fn clear_preedit(&mut self, ctx: &mut FontContext) {
        if self.remove_preedit_text() {
            self.sync(ctx);
        }
    }

    /// Whether a composition is in flight.
    pub fn has_preedit(&self) -> bool {
        self.preedit.is_some()
    }

    /// The composition text as per-line rects (empty without one) — what a
    /// widget underlines.
    pub fn preedit_rects(&mut self, ctx: &mut FontContext) -> Vec<Rect> {
        self.sync(ctx);
        let Some(range) = self.preedit else {
            return Vec::new();
        };
        self.editor.with_buffer(|buffer| {
            let mut rects = Vec::new();
            for run in buffer.layout_runs() {
                for (x, width) in run.highlight(range.start, range.end) {
                    rects.push(Rect::from_xywh(x, run.line_top, width, run.line_height));
                }
            }
            rects
        })
    }

    /// Deletes the active composition's text, leaving the caret at its
    /// start. Returns whether one was active. Callers re-shape.
    fn remove_preedit_text(&mut self) -> bool {
        let Some(range) = self.preedit.take() else {
            return false;
        };
        self.editor.set_selection(Selection::Normal(range.start));
        self.editor.set_cursor(range.end);
        self.editor.delete_selection();
        true
    }

    /// The cursor's byte offset into [`text`](Self::text)'s joined form.
    fn cursor_offset(&self, cursor: Cursor) -> usize {
        self.editor
            .with_buffer(|buffer| crate::cursor_offset(buffer, cursor))
    }

    /// The cursor at a byte offset into the joined text (clamped to the end).
    fn offset_cursor(&self, offset: usize) -> Cursor {
        self.editor
            .with_buffer(|buffer| crate::offset_cursor(buffer, offset))
    }

    /// Deletes committed text around the caret (or selection): `before`
    /// bytes leftward of its start and `after` bytes rightward of its end,
    /// snapped to character boundaries. The caret lands where the deleted
    /// ranges closed, unselected. An active composition is preserved — it
    /// keeps floating at the caret while the text around it changes, which
    /// is how buffer-mirroring input methods revise earlier words
    /// mid-composition.
    pub fn delete_surrounding(&mut self, ctx: &mut FontContext, before: usize, after: usize) {
        let committed_before = self.text();
        // Lift the composition out so the deletion can't invalidate its
        // cursors, then restore it at the caret afterwards.
        let preedit = self.preedit.map(|range| {
            let (start, end) = (
                self.cursor_offset(range.start),
                self.cursor_offset(range.end),
            );
            let caret = self.cursor_offset(self.editor.cursor()).clamp(start, end) - start;
            (self.display_text()[start..end].to_string(), caret)
        });
        self.remove_preedit_text();

        let text = self.display_text();
        let (sel_start, sel_end) = self.selection_bytes();
        let snap_down = |mut offset: usize| {
            while !text.is_char_boundary(offset) {
                offset -= 1;
            }
            offset
        };
        let snap_up = |mut offset: usize| {
            while offset < text.len() && !text.is_char_boundary(offset) {
                offset += 1;
            }
            offset
        };
        let del_start = snap_down(sel_start.saturating_sub(before));
        let del_end = snap_up((sel_end.saturating_add(after)).min(text.len()));
        // The later range first, so the earlier offsets stay valid.
        if del_end > sel_end {
            self.editor
                .set_selection(Selection::Normal(self.offset_cursor(sel_end)));
            self.editor.set_cursor(self.offset_cursor(del_end));
            self.editor.delete_selection();
        }
        if del_start < sel_start {
            self.editor
                .set_selection(Selection::Normal(self.offset_cursor(del_start)));
            self.editor.set_cursor(self.offset_cursor(sel_start));
            self.editor.delete_selection();
        }
        self.editor.set_selection(Selection::None);
        self.editor.set_cursor(self.offset_cursor(del_start));

        if let Some((text, caret)) = preedit {
            let start = self.editor.cursor();
            self.editor.insert_string(&text, None);
            let end = self.editor.cursor();
            self.preedit = Some(PreeditRange { start, end });
            let caret = self.offset_cursor(self.cursor_offset(start) + caret);
            self.editor.set_cursor(caret);
        }
        self.repair_color_spans(&committed_before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Deletes the selection, or the grapheme before the caret.
    pub fn delete_backward(&mut self, ctx: &mut FontContext) {
        let before = self.text();
        self.editor.action(&mut ctx.font_system, Action::Backspace);
        self.repair_color_spans(&before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Deletes the selection, or the grapheme after the caret.
    pub fn delete_forward(&mut self, ctx: &mut FontContext) {
        let before = self.text();
        self.editor.action(&mut ctx.font_system, Action::Delete);
        self.repair_color_spans(&before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Deletes the selection, or back to the previous word boundary.
    pub fn delete_word_backward(&mut self, ctx: &mut FontContext) {
        let before = self.text();
        if !self.has_selection() {
            self.move_caret(ctx, CaretMotion::WordLeft, true);
        }
        self.editor.delete_selection();
        self.repair_color_spans(&before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Deletes the selection, or forward to the next word boundary.
    pub fn delete_word_forward(&mut self, ctx: &mut FontContext) {
        let before = self.text();
        if !self.has_selection() {
            self.move_caret(ctx, CaretMotion::WordRight, true);
        }
        self.editor.delete_selection();
        self.repair_color_spans(&before, &self.text());
        self.apply_color_spans();
        self.sync(ctx);
    }

    /// Moves the caret. With `select` the selection extends from its anchor
    /// (anchored at the caret if nothing was selected); without it the
    /// selection clears — a plain Left/Right collapses to the selection's
    /// edge instead of moving, as editors conventionally do.
    pub fn move_caret(&mut self, ctx: &mut FontContext, motion: CaretMotion, select: bool) {
        if select {
            if matches!(self.editor.selection(), Selection::None) {
                self.editor
                    .set_selection(Selection::Normal(self.editor.cursor()));
            }
        } else if let Some((start, end)) = self.editor.selection_bounds()
            && matches!(motion, CaretMotion::Left | CaretMotion::Right)
        {
            self.editor.set_selection(Selection::None);
            self.editor.set_cursor(match motion {
                CaretMotion::Left => start,
                _ => end,
            });
            self.sync(ctx);
            return;
        } else {
            self.editor.set_selection(Selection::None);
        }
        self.editor
            .action(&mut ctx.font_system, Action::Motion(motion.as_cosmic()));
        self.sync(ctx);
    }

    /// Places the caret at `position` (a click), or extends the selection
    /// to it (a drag).
    pub fn caret_to(&mut self, ctx: &mut FontContext, position: Vector2, select: bool) {
        if !select {
            self.editor.set_selection(Selection::None);
        }
        let (x, y) = (position.x.round() as i32, position.y.round() as i32);
        let action = if select {
            Action::Drag { x, y }
        } else {
            Action::Click { x, y }
        };
        self.editor.action(&mut ctx.font_system, action);
        self.sync(ctx);
    }

    /// Places the caret at a byte offset, or extends the selection to it.
    ///
    /// This is the offset counterpart of [`caret_to`](Self::caret_to), for a
    /// caller that hit-tests a visual representation of the edited text.
    pub fn caret_to_offset(&mut self, ctx: &mut FontContext, offset: usize, select: bool) {
        if select {
            if matches!(self.editor.selection(), Selection::None) {
                self.editor
                    .set_selection(Selection::Normal(self.editor.cursor()));
            }
        } else {
            self.editor.set_selection(Selection::None);
        }
        self.editor.set_cursor(self.offset_cursor(offset));
        self.sync(ctx);
    }

    /// Selects the word the caret sits in — what a double click takes.
    /// Leaves an empty selection where there is no word to take.
    pub fn select_word(&mut self, ctx: &mut FontContext) {
        let (_, caret) = self.selection_bytes();
        let range = crate::word_range(&self.display_text(), caret);
        self.set_selection_bytes(ctx, range.start, range.end);
    }

    /// Selects the line the caret sits in, excluding its newline — what a
    /// triple click takes. Logical, so a wrapped paragraph goes whole.
    pub fn select_line(&mut self, ctx: &mut FontContext) {
        let (_, caret) = self.selection_bytes();
        let range = crate::line_range(&self.display_text(), caret);
        self.set_selection_bytes(ctx, range.start, range.end);
    }

    /// Selects everything, caret at the end.
    pub fn select_all(&mut self, ctx: &mut FontContext) {
        self.editor.set_cursor(Cursor::new(0, 0));
        self.editor
            .set_selection(Selection::Normal(Cursor::new(0, 0)));
        self.editor
            .action(&mut ctx.font_system, Action::Motion(Motion::BufferEnd));
        self.sync(ctx);
    }

    /// Turns an empty drag selection back into an ordinary caret.
    ///
    /// `cosmic-text` can retain a normal selection whose anchor and cursor
    /// are equal after a pointer move. It is semantically a caret, and leaving
    /// it armed can make later editing and selection geometry treat it as a
    /// range.
    pub fn finish_selection(&mut self) {
        if self
            .editor
            .selection_bounds()
            .is_some_and(|(start, end)| start == end)
        {
            self.editor.set_selection(Selection::None);
        }
    }

    pub fn has_selection(&self) -> bool {
        self.editor
            .selection_bounds()
            .is_some_and(|(start, end)| start != end)
    }

    /// The selected text, if any.
    pub fn selected_text(&self) -> Option<String> {
        self.has_selection()
            .then(|| self.editor.copy_selection())
            .flatten()
    }

    /// The caret's rect: a hairline of the line's height whose x is the
    /// caret position. Widgets widen and color it.
    pub fn caret_rect(&mut self, ctx: &mut FontContext) -> Rect {
        self.sync(ctx);
        let (x, y) = self.editor.cursor_position().unwrap_or((0, 0));
        Rect::from_xywh(x as f32, y as f32, 1.0, self.style.metrics().line_height)
    }

    /// The selection as per-line rects (empty without a selection).
    pub fn selection_rects(&mut self, ctx: &mut FontContext) -> Vec<Rect> {
        self.sync(ctx);
        let Some((start, end)) = self.editor.selection_bounds() else {
            return Vec::new();
        };
        if start == end {
            return Vec::new();
        }
        self.editor
            .with_buffer(|buffer| crate::selection_rects(buffer, start, end))
    }

    /// Measured extent of the current text, for layout.
    pub fn metrics(&mut self, ctx: &mut FontContext) -> TextMetrics {
        self.sync(ctx);
        let line_height = self.style.metrics().line_height;
        self.editor
            .with_buffer(|buffer| measure(buffer, line_height))
    }

    /// Rasterizes the current text. Same contract as
    /// [`ShapedText::place`](crate::ShapedText::place).
    pub fn place(
        &mut self,
        ctx: &mut FontContext,
        color: impl Into<Paint>,
        origin: Vector2,
        scale: f32,
    ) -> GlyphRun {
        self.sync(ctx);
        let color = color.into();
        self.editor
            .with_buffer(|buffer| place_glyphs(ctx, buffer, color, origin, scale))
    }

    /// Rasterizes with the buffer's byte-ranged foreground colors.
    pub fn place_styled(
        &mut self,
        ctx: &mut FontContext,
        color: impl Into<Paint>,
        origin: Vector2,
        scale: f32,
    ) -> GlyphRun {
        self.sync(ctx);
        let colors: Vec<_> = self
            .color_spans
            .iter()
            .map(|span| Paint::solid(span.color))
            .collect();
        self.editor.with_buffer(|buffer| {
            place_glyphs_styled(ctx, buffer, color.into(), &colors, origin, scale)
        })
    }

    fn apply_color_spans(&mut self) {
        let base = attrs_for(&self.style);
        let spans = &self.color_spans;
        self.editor.with_buffer_mut(|buffer| {
            let mut line_start = 0;
            for line in &mut buffer.lines {
                let line_end = line_start + line.text().len();
                let mut attrs = AttrsList::new(&base);
                for (index, span) in spans.iter().enumerate() {
                    let start = span.range.start.max(line_start);
                    let end = span.range.end.min(line_end);
                    if start < end {
                        attrs.add_span(
                            start - line_start..end - line_start,
                            &base.clone().metadata(index + 1),
                        );
                    }
                }
                line.set_attrs_list(attrs);
                line_start = line_end + 1;
            }
        });
    }

    fn repair_color_spans(&mut self, before: &str, after: &str) {
        let prefix = common_prefix(before, after);
        let before_suffix = common_suffix(&before[prefix..], &after[prefix..]);
        let old_end = before.len() - before_suffix;
        let new_end = after.len() - before_suffix;
        let delta = new_end as isize - old_end as isize;
        for span in &mut self.color_spans {
            if old_end == prefix {
                if span.range.start >= prefix {
                    span.range.start = span.range.start.saturating_add_signed(delta);
                }
                if span.range.end > prefix {
                    span.range.end = span.range.end.saturating_add_signed(delta);
                }
            } else {
                span.range.start = repair_offset(span.range.start, prefix, old_end, new_end, delta);
                span.range.end = repair_offset(span.range.end, prefix, old_end, prefix, delta);
            }
        }
        self.color_spans
            .retain(|span| span.range.start < span.range.end);
    }

    /// Re-shapes whatever the last mutation touched.
    fn sync(&mut self, ctx: &mut FontContext) {
        self.editor.shape_as_needed(&mut ctx.font_system, false);
    }
}

fn snap_down(text: &str, mut offset: usize) -> usize {
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn common_prefix(a: &str, b: &str) -> usize {
    let limit = a.len().min(b.len());
    let mut bytes = 0;
    for ((ai, ac), (bi, bc)) in a.char_indices().zip(b.char_indices()) {
        if ai >= limit || bi >= limit || ac != bc {
            break;
        }
        bytes = ai + ac.len_utf8();
    }
    bytes
}

fn common_suffix(a: &str, b: &str) -> usize {
    let mut bytes = 0;
    for (ac, bc) in a.chars().rev().zip(b.chars().rev()) {
        if ac != bc {
            break;
        }
        bytes += ac.len_utf8();
    }
    bytes
}

fn repair_offset(
    offset: usize,
    start: usize,
    old_end: usize,
    replacement_edge: usize,
    delta: isize,
) -> usize {
    if offset <= start {
        offset
    } else if offset >= old_end {
        offset.saturating_add_signed(delta)
    } else {
        replacement_edge
    }
}

fn attrs_for(style: &TextStyle) -> Attrs<'_> {
    style.attrs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FontContext {
        FontContext::embedded_only()
    }

    fn style() -> TextStyle {
        TextStyle::new(16.0)
    }

    fn buffer(ctx: &mut FontContext, text: &str) -> EditBuffer {
        let mut buffer = EditBuffer::new(style());
        buffer.set_text(ctx, text);
        buffer
    }

    #[test]
    fn set_text_round_trips_and_puts_the_caret_at_the_end() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "hello world");
        assert_eq!(buf.text(), "hello world");
        buf.insert(&mut ctx, "!");
        assert_eq!(buf.text(), "hello world!");
    }

    #[test]
    fn insert_and_delete_edit_at_the_caret() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "helo");
        buf.move_caret(&mut ctx, CaretMotion::Left, false);
        buf.move_caret(&mut ctx, CaretMotion::Left, false);
        buf.insert(&mut ctx, "l");
        assert_eq!(buf.text(), "hello");

        buf.delete_backward(&mut ctx);
        assert_eq!(buf.text(), "helo");
        buf.delete_forward(&mut ctx);
        assert_eq!(buf.text(), "heo");
    }

    #[test]
    fn word_motion_selects_words() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "alpha beta gamma");
        buf.move_caret(&mut ctx, CaretMotion::WordLeft, true);
        assert_eq!(buf.selected_text().as_deref(), Some("gamma"));

        buf.move_caret(&mut ctx, CaretMotion::WordLeft, true);
        assert_eq!(buf.selected_text().as_deref(), Some("beta gamma"));
    }

    #[test]
    fn selection_replace_and_select_all() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "alpha beta");
        buf.move_caret(&mut ctx, CaretMotion::WordLeft, true);
        buf.insert(&mut ctx, "gamma");
        assert_eq!(buf.text(), "alpha gamma");

        buf.select_all(&mut ctx);
        assert_eq!(buf.selected_text().as_deref(), Some("alpha gamma"));
        buf.delete_backward(&mut ctx);
        assert_eq!(buf.text(), "");
    }

    #[test]
    fn color_spans_repair_across_edits_and_color_glyphs() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "let answer");
        let keyword = Color::from_srgb8(120, 90, 255, 255);
        buf.set_color_spans(&mut ctx, vec![ColorSpan::new(0..3, keyword)]);
        buf.move_caret(&mut ctx, CaretMotion::TextStart, false);
        buf.insert(&mut ctx, "// ");
        assert_eq!(buf.color_spans()[0].range, 3..6);

        let run = buf.place_styled(
            &mut ctx,
            Color::from_srgb8(220, 220, 220, 255),
            Vector2::ZERO,
            1.0,
        );
        assert!(
            run.glyphs
                .iter()
                .any(|glyph| glyph.color == Paint::solid(keyword))
        );
    }

    #[test]
    fn logical_lines_and_caret_position_include_empty_lines() {
        let mut ctx = ctx();
        let buf = buffer(&mut ctx, "alpha\nβeta\n");
        assert_eq!(
            buf.lines(),
            vec![
                LineMetadata {
                    index: 0,
                    start: 0,
                    end: 5
                },
                LineMetadata {
                    index: 1,
                    start: 6,
                    end: 11
                },
                LineMetadata {
                    index: 2,
                    start: 12,
                    end: 12
                },
            ]
        );
        assert_eq!(buf.caret_line_column(), (3, 1));
    }

    #[test]
    fn plain_motion_collapses_the_selection_to_its_edge() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "abc");
        buf.select_all(&mut ctx);
        buf.move_caret(&mut ctx, CaretMotion::Left, false);
        assert!(!buf.has_selection());
        buf.insert(&mut ctx, "x");
        assert_eq!(buf.text(), "xabc", "caret collapsed to the start");
    }

    #[test]
    fn caret_rect_advances_rightward() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "mm");
        let end = buf.caret_rect(&mut ctx);
        assert!(end.origin.x > 0.0);
        assert_eq!(end.size.height, style().metrics().line_height);

        buf.move_caret(&mut ctx, CaretMotion::TextStart, false);
        let start = buf.caret_rect(&mut ctx);
        assert_eq!(start.origin.x, 0.0);
        assert!(end.origin.x > start.origin.x);
    }

    #[test]
    fn clicks_place_the_caret_and_drags_select() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "hello world");
        let width = buf.metrics(&mut ctx).size.width;

        buf.caret_to(&mut ctx, Vector2::new(0.0, 4.0), false);
        buf.insert(&mut ctx, ">");
        assert_eq!(buf.text(), ">hello world");

        buf.caret_to(&mut ctx, Vector2::new(0.0, 4.0), false);
        buf.caret_to(&mut ctx, Vector2::new(width + 10.0, 4.0), true);
        assert_eq!(buf.selected_text().as_deref(), Some(">hello world"));
        assert!(!buf.selection_rects(&mut ctx).is_empty());

        buf.caret_to(&mut ctx, Vector2::new(0.0, 4.0), false);
        assert!(!buf.has_selection(), "a plain click collapses a selection");

        let len = buf.text().len();
        buf.set_selection_bytes(&mut ctx, 0, len);
        buf.caret_to(&mut ctx, Vector2::new(0.0, 4.0), false);
        assert!(
            !buf.has_selection(),
            "a click collapses an external selection"
        );
    }

    #[test]
    fn an_equal_drag_selection_is_only_a_caret() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "first line\nsecond line\nthird line");

        buf.caret_to_offset(&mut ctx, 6, false);
        buf.caret_to_offset(&mut ctx, 6, true);
        assert!(!buf.has_selection());
        assert_eq!(buf.selected_text(), None);
        assert!(buf.selection_rects(&mut ctx).is_empty());

        buf.caret_to_offset(&mut ctx, 20, true);
        assert!(buf.has_selection(), "moving away creates a real selection");
        buf.caret_to_offset(&mut ctx, 6, true);
        assert!(!buf.has_selection(), "returning to the anchor is a caret");
        assert!(buf.selection_rects(&mut ctx).is_empty());

        buf.finish_selection();
        assert!(matches!(buf.editor.selection(), Selection::None));
    }

    #[test]
    fn offset_hit_testing_preserves_a_backwards_selection_focus() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "hello");

        buf.caret_to_offset(&mut ctx, 2, true);
        assert_eq!(buf.selection_bytes(), (2, 5));
        assert_eq!(buf.caret_offset(), 2);

        buf.caret_to_offset(&mut ctx, 1, true);
        assert_eq!(buf.selection_bytes(), (1, 5));
        assert_eq!(buf.caret_offset(), 1);
    }

    #[test]
    fn editing_a_never_set_buffer_works() {
        let mut ctx = ctx();
        let mut buf = EditBuffer::new(style());
        // A drag on the empty field selects nothing but arms a selection;
        // the insert then goes through the selection-replace path.
        buf.caret_to(&mut ctx, Vector2::new(0.0, 4.0), false);
        buf.caret_to(&mut ctx, Vector2::new(8.0, 4.0), true);
        buf.insert(&mut ctx, "hi");
        assert_eq!(buf.text(), "hi");

        let mut buf = EditBuffer::new(style());
        buf.delete_backward(&mut ctx);
        buf.delete_forward(&mut ctx);
        assert_eq!(buf.text(), "");
    }

    #[test]
    fn preedit_shapes_but_stays_out_of_the_text() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "ab");
        buf.move_caret(&mut ctx, CaretMotion::Left, false);

        buf.set_preedit(&mut ctx, "ni", Some((2, 2)));
        assert!(buf.has_preedit());
        assert_eq!(buf.text(), "ab", "composition is not part of the value");
        assert!(!buf.preedit_rects(&mut ctx).is_empty());
        let with = buf.metrics(&mut ctx).size.width;

        buf.set_preedit(&mut ctx, "nih", Some((3, 3)));
        assert_eq!(buf.text(), "ab", "replacement swaps the composition");
        assert!(buf.metrics(&mut ctx).size.width > with);

        buf.clear_preedit(&mut ctx);
        assert!(!buf.has_preedit());
        assert_eq!(buf.text(), "ab");
        assert!(buf.preedit_rects(&mut ctx).is_empty());
        buf.insert(&mut ctx, "-");
        assert_eq!(buf.text(), "a-b", "caret returned to the composition start");
    }

    #[test]
    fn commit_replaces_the_composition() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "");

        // The winit-shaped sequence: grow, shrink to empty, then commit.
        buf.set_preedit(&mut ctx, "n", Some((1, 1)));
        buf.set_preedit(&mut ctx, "ni", Some((2, 2)));
        buf.set_preedit(&mut ctx, "", None);
        buf.insert(&mut ctx, "你");
        assert_eq!(buf.text(), "你");

        // A commit that skips the empty closing preedit still ends clean.
        buf.set_preedit(&mut ctx, "ha", Some((2, 2)));
        buf.insert(&mut ctx, "哈");
        assert_eq!(buf.text(), "你哈");
        assert!(!buf.has_preedit());
    }

    #[test]
    fn preedit_replaces_the_selection_once() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "abc");
        buf.select_all(&mut ctx);

        buf.set_preedit(&mut ctx, "x", Some((1, 1)));
        assert_eq!(buf.text(), "");
        buf.set_preedit(&mut ctx, "xy", Some((2, 2)));
        buf.insert(&mut ctx, "xyz");
        assert_eq!(buf.text(), "xyz");
    }

    #[test]
    fn preedit_cursor_places_the_caret_inside_the_composition() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "");
        buf.set_preedit(&mut ctx, "mm", Some((2, 2)));
        let at_end = buf.caret_rect(&mut ctx);
        buf.set_preedit(&mut ctx, "mm", Some((0, 0)));
        let at_start = buf.caret_rect(&mut ctx);
        assert!(at_start.origin.x < at_end.origin.x);
    }

    #[test]
    fn multi_line_text_reports_per_line_geometry() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "one\ntwo");
        assert_eq!(buf.text(), "one\ntwo");

        let end = buf.caret_rect(&mut ctx);
        buf.move_caret(&mut ctx, CaretMotion::Up, false);
        let up = buf.caret_rect(&mut ctx);
        assert!(up.origin.y < end.origin.y, "Up moved a visual line");

        buf.select_all(&mut ctx);
        assert_eq!(buf.selection_rects(&mut ctx).len(), 2, "one rect per line");
    }

    #[test]
    fn display_text_and_byte_offsets_expose_the_composition() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "ab");
        assert_eq!(buf.display_text(), "ab");
        assert_eq!(buf.selection_bytes(), (2, 2), "caret at the end");
        assert_eq!(buf.preedit_bytes(), None);

        buf.move_caret(&mut ctx, CaretMotion::Left, false);
        buf.set_preedit(&mut ctx, "ni", Some((2, 2)));
        assert_eq!(buf.display_text(), "anib", "composition is displayed");
        assert_eq!(buf.text(), "ab", "…but stays out of the value");
        assert_eq!(buf.preedit_bytes(), Some((1, 3)));
        assert_eq!(buf.selection_bytes(), (3, 3), "caret after the composition");
    }

    #[test]
    fn set_selection_bytes_places_caret_and_selection() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "hello");

        buf.set_selection_bytes(&mut ctx, 1, 1);
        assert!(!buf.has_selection());
        buf.insert(&mut ctx, "-");
        assert_eq!(buf.text(), "h-ello");

        buf.set_selection_bytes(&mut ctx, 2, 5);
        assert_eq!(buf.selected_text().as_deref(), Some("ell"));
        assert_eq!(buf.selection_bytes(), (2, 5));

        // Reversed and out-of-range offsets normalize; mid-character
        // offsets snap to a boundary.
        buf.set_text(&mut ctx, "你好");
        buf.set_selection_bytes(&mut ctx, 100, 1);
        assert_eq!(buf.selected_text().as_deref(), Some("你好"), "0..len");
    }

    #[test]
    fn delete_surrounding_removes_text_around_the_caret() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "alpha beta gamma");
        buf.set_selection_bytes(&mut ctx, 10, 10); // after "beta"

        buf.delete_surrounding(&mut ctx, 4, 6);
        assert_eq!(buf.text(), "alpha ", "removed \"beta\" and \" gamma\"");
        assert_eq!(buf.selection_bytes(), (6, 6), "caret where the gap closed");

        // Over-long ranges clamp to the text.
        buf.delete_surrounding(&mut ctx, 100, 100);
        assert_eq!(buf.text(), "");
    }

    #[test]
    fn delete_surrounding_preserves_an_active_composition() {
        let mut ctx = ctx();
        let mut buf = buffer(&mut ctx, "wrod ");
        buf.set_preedit(&mut ctx, "ne", Some((2, 2)));
        assert_eq!(buf.display_text(), "wrod ne");

        // The input method corrects the committed word behind the caret
        // without dropping the in-flight composition.
        buf.delete_surrounding(&mut ctx, 5, 0);
        assert_eq!(buf.display_text(), "ne");
        assert_eq!(buf.text(), "", "only the composition remains");
        assert_eq!(buf.preedit_bytes(), Some((0, 2)));

        buf.insert(&mut ctx, "new ");
        assert_eq!(buf.text(), "new ");
        assert!(!buf.has_preedit());
    }
}
