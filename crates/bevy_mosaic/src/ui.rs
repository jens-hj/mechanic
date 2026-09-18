//! Mosaic's authoring vocabulary: widgets, layout, paint, reactive state, the
//! theme palettes, and the `view!` macro.
//!
//! This mirrors `mosaic::prelude` with the windowing items removed — `App`,
//! `WindowConfig`, `AppContext` and friends belong to Mosaic's own runtime,
//! which a Bevy app replaces. Skipping them is also what keeps `winit`,
//! `arboard` and `accesskit_winit` out of the dependency graph.
//!
//! Glob import it in modules that build UI — under clippy's pedantic lints
//! that needs an `#[allow(clippy::wildcard_imports)]` on the module, which is
//! the trade for `view!` reading the way it does everywhere else. It is kept
//! separate from
//! [`prelude`](crate::prelude) because Mosaic and Bevy both define `State`, `Children` and
//! `Interaction`: importing both preludes into one module is ambiguous on all
//! three, which is a decision only the calling module can make.

pub use mosaic_core::{
    Catppuccin, CatppuccinFlavor, Color, ColorToken, Derived, DisplayP3, Easing, Effect,
    ElementToken, FRAPPE, Hsl, Hsv, Hwb, IntoFontLength, LATTE, Lab, Lch, LengthToken, MACCHIATO,
    MOCHA, Motion, Oklab, Oklch, PaintToken, ReadState, Rect, ScalarToken, Scope, Size, Spring,
    Srgb, State, StateSender, SvgToken, SystemScheme, ThemeToken, Vector2, ViewBinding,
    ViewBuildGuard, batch, col, ease, frappe, install_theme, latte, macchiato, mocha, on_cleanup,
    resolve_theme_value, spring, state_channel, untracked,
};
pub use mosaic_layout::{
    Align, AxisPair, Dimension, Direction, Edges, Grid, GridPacking, GridTrack, GridTracks,
    Inherit, Justify, LayoutMode, Length, SizeBound, Style, Translate,
};
pub use mosaic_macros::{component, displacement, preview, scheme, style, surface, theme, view};
pub use mosaic_render::{
    Anchor, ArcSpec, BackdropFilter, BackdropSample, BoxPoint, BoxSize, CIRCULAR_EXPONENT,
    ColorEdit, CornerSpec, DirectionalLight, DisplacementField, Extend, FieldProgram, FieldUniform,
    FilterChain, FilterLength, FilterScale, GaussianPlan, GeometrySpec, GradientSpec, GradientStop,
    InterpSpace, KindSpec, LayerSpec, LightKindSpec, LightPosition, LightRecordBase, LightSpec,
    LightSpill, LightTargets, LineCap, LineJoin, MarkerEnd, MarkerShape, MarkerSpec, MaskComposite,
    MaskMode, MeshSpec, PaintCmd, PaintSpec, PlannedFilter, PointLight, PointSpec, Radius, Reach,
    Refraction, Scene, ShadowSpec, Shape, SpotLight, StopSpec, StrokeEdges, StrokeSpec,
    SurfaceProfile, Theme, ThemedColor, ThemedF32, ThemedVec2, even_stops, plan_gaussian,
};
pub use mosaic_text::{
    FontFamily, FontStretch, FontStyle, IntoLineHeight, LineHeight, TextStyle, TextTransform,
    TextWrap,
};
pub use mosaic_widgets::input::{
    Clipboard, ImeEvent, Key, KeyEvent, KeyEventKind, Modifiers, PointerButton, PointerEvent,
    PointerEventKind, PointerType, TextInputContents,
};
pub use mosaic_widgets::{
    Animated, ButtonStyle, Checkbox, CheckboxStyle, Children, ColorPicker, ColorPickerStyle,
    ComponentHandle, DragAxis, DragEvent, DragOptions, DragPhase, DragRelease, EditorState,
    Element, ElementPatch, ElementSpec, FindBar, FindBarStyle, FocusStyle, Fx, IconPaints,
    IconPart, IconPartStyle, IconSource, IconStroke, IconStyle, IconValue, ImageSource, ImgStyle,
    InspectionAttribute, InspectionAttributeOrigin, InspectionBoundary, InspectionBoundaryKind,
    InspectionDetails, InspectionDetailsSeed, InspectionNode, InspectionSnapshot, InspectionTreeId,
    Interaction, IntoIconColor, IntoIconStroke, IntoSemanticRole, IntoSemanticText,
    IntoTextContent, LayoutSize, MaskSpec, ObjectFit, OverlayAlign, OverlayCollision,
    OverlayPlacement, OverlayPoint, OverlayPosition, OverlaySide, Progress, ProgressStyle, Prop,
    Radii, Radio, RadioStyle, ReorderAxis, ReorderEvent, ReorderLocation, ReorderMode,
    ReorderOptions, ResizeEdges, ResizeEvent, ResizeOptions, ResizePhase, Role, Scroll, Select,
    SelectStyle, Semantics, Slider, SliderStyle, SpanBackground, Stepper, StepperStyle, StyleCtx,
    StylePart, StyleSet, SvgSpec, TextAreaStyle, TextContent, TextEditor, TextEditorStyle,
    TextInputOptions, TextInputStyle, TextSpan, ThemeAsset, Toggle, ToggleStyle, Tooltip,
    TooltipOptions, TooltipTrigger, Transition, Ui, VirtualList, Visual, bind_anchored_overlay,
    button, button_container, button_container_styled, button_styled, canvas_fit, checkbox,
    checkbox_labeled, checkbox_styled, color_picker, color_picker_styled, fade, find_bar,
    find_bar_styled, fly, icon, icon_dyn, icon_element, img, img_dyn, peek_icon_element, peek_svg,
    progress, progress_dyn, progress_dyn_styled, progress_styled, radio, radio_labeled,
    radio_styled, resolve_icon_color, resolve_icon_stroke, scroll, select, select_styled,
    set_icon_element, set_svg, shape, shape_content, shape_filled, slide, slide_x, slider,
    slider_styled, stepper, stepper_styled, style_text_tooltip_root, text, text_area,
    text_area_styled, text_area_styled_with_options, text_area_with_options, text_dyn,
    text_dyn_styled, text_editor, text_editor_styled, text_input, text_input_styled,
    text_input_styled_with_options, text_input_with_options, text_tooltip, toggle, toggle_styled,
    tooltip, virtual_list, virtual_list_measured,
};
