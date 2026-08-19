use bevy::{
    prelude::*,
    ui::{FocusPolicy, RelativeCursorPosition},
};

const SLOT_SIZE: f32 = 64.0;
const ICON_COLOR: Color = Color::srgb(0.82, 0.90, 0.97);
const BEARING_COLOR: Color = Color::srgb(0.95, 0.58, 0.08);
const SLOT_BACKGROUND: Color = Color::srgba(0.025, 0.035, 0.05, 0.92);
const SLOT_HOVER_BACKGROUND: Color = Color::srgba(0.08, 0.12, 0.16, 0.96);
const SLOT_SELECTED_BACKGROUND: Color = Color::srgba(0.08, 0.20, 0.27, 0.98);
const SLOT_BORDER: Color = Color::srgba(0.35, 0.43, 0.50, 0.9);
const SLOT_HOVER_BORDER: Color = Color::srgb(0.72, 0.82, 0.90);
const SLOT_SELECTED_BORDER: Color = Color::srgb(0.25, 0.85, 1.0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Tool {
    #[default]
    Block,
    Bearing,
    Weld,
    Hammer,
}

impl Tool {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Block => "Block",
            Self::Bearing => "Bearing",
            Self::Weld => "Weld",
            Self::Hammer => "Hammer",
        }
    }

    pub(crate) const fn shortcut(self) -> &'static str {
        match self {
            Self::Block => "1",
            Self::Bearing => "2",
            Self::Weld => "3",
            Self::Hammer => "4",
        }
    }

    pub(crate) const fn works_while_simulating(self) -> bool {
        matches!(self, Self::Hammer)
    }

    pub(crate) const fn works_in_mode(self, simulating: bool) -> bool {
        self.works_while_simulating() == simulating
    }
}

pub(crate) const fn shortcut_tool(key: KeyCode) -> Option<Tool> {
    match key {
        KeyCode::Digit1 => Some(Tool::Block),
        KeyCode::Digit2 => Some(Tool::Bearing),
        KeyCode::Digit3 => Some(Tool::Weld),
        KeyCode::Digit4 => Some(Tool::Hammer),
        _ => None,
    }
}

#[derive(Resource, Debug, Default)]
pub(crate) struct SelectedTool(pub(crate) Tool);

#[derive(Resource, Debug, Default)]
pub(crate) struct HotbarPointerCapture(bool);

impl HotbarPointerCapture {
    pub(crate) const fn active(&self) -> bool {
        self.0
    }
}

#[derive(Component)]
pub(crate) struct HotbarSlot(Tool);

#[derive(Component)]
pub(crate) struct HotbarTooltip;

#[derive(Component)]
pub(crate) struct HotbarSurface;

pub(crate) fn spawn(commands: &mut Commands) {
    commands
        .spawn((
            Name::new("Tool hotbar"),
            Node {
                position_type: PositionType::Absolute,
                bottom: px(18),
                left: px(0),
                width: percent(100),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: px(6),
                ..default()
            },
            FocusPolicy::Pass,
        ))
        .with_children(|root| {
            root.spawn((
                HotbarTooltip,
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    min_width: px(72),
                    height: px(26),
                    padding: UiRect::axes(px(9), px(3)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(4)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.94)),
                Visibility::Hidden,
                FocusPolicy::Pass,
            ));

            root.spawn((
                Name::new("Tool slots"),
                HotbarSurface,
                RelativeCursorPosition::default(),
                Node {
                    padding: UiRect::all(px(8)),
                    column_gap: px(8),
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(8)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.015, 0.02, 0.03, 0.80)),
                FocusPolicy::Pass,
            ))
            .with_children(|bar| {
                for tool in [Tool::Block, Tool::Bearing, Tool::Weld, Tool::Hammer] {
                    spawn_slot(bar, tool);
                }
            });
        });
}

fn spawn_slot(parent: &mut ChildSpawnerCommands<'_>, tool: Tool) {
    parent
        .spawn((
            Name::new(format!("{} tool", tool.label())),
            Button,
            HotbarSlot(tool),
            Node {
                width: px(SLOT_SIZE),
                height: px(SLOT_SIZE),
                border: UiRect::all(px(3)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(SLOT_BACKGROUND),
            BorderColor::all(SLOT_BORDER),
        ))
        .with_children(|slot| {
            spawn_icon(slot, tool);
            slot.spawn((
                Text::new(tool.shortcut()),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    position_type: PositionType::Absolute,
                    top: px(2),
                    left: px(5),
                    ..default()
                },
            ));
        });
}

fn spawn_icon(parent: &mut ChildSpawnerCommands<'_>, tool: Tool) {
    parent
        .spawn(Node {
            width: px(40),
            height: px(40),
            position_type: PositionType::Relative,
            ..default()
        })
        .with_children(|icon| match tool {
            Tool::Block => spawn_block_icon(icon),
            Tool::Bearing => spawn_bearing_icon(icon),
            Tool::Weld => spawn_weld_icon(icon),
            Tool::Hammer => spawn_hammer_icon(icon),
        });
}

fn spawn_block_icon(icon: &mut ChildSpawnerCommands<'_>) {
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(5),
            width: px(22),
            height: px(22),
            border: UiRect::all(px(2)),
            ..default()
        },
        BorderColor::all(Color::srgba(0.48, 0.68, 0.84, 0.9)),
    ));
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(6),
            top: px(12),
            width: px(25),
            height: px(24),
            border: UiRect::all(px(2)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.22, 0.55, 0.86)),
        BorderColor::all(ICON_COLOR),
    ));
}

fn spawn_bearing_icon(icon: &mut ChildSpawnerCommands<'_>) {
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(4),
            top: px(4),
            width: px(32),
            height: px(32),
            border: UiRect::all(px(7)),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        BorderColor::all(BEARING_COLOR),
    ));
}

fn spawn_weld_icon(icon: &mut ChildSpawnerCommands<'_>) {
    for rotation in [-35.0, 35.0] {
        icon.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(5),
                top: px(16),
                width: px(30),
                height: px(7),
                border_radius: BorderRadius::all(px(2)),
                ..default()
            },
            BackgroundColor(ICON_COLOR),
            UiTransform::from_rotation(Rot2::degrees(rotation)),
        ));
    }
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(16),
            top: px(16),
            width: px(8),
            height: px(8),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        BackgroundColor(BEARING_COLOR),
    ));
}

fn spawn_hammer_icon(icon: &mut ChildSpawnerCommands<'_>) {
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(17),
            top: px(13),
            width: px(7),
            height: px(27),
            border_radius: BorderRadius::all(px(2)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.68, 0.43, 0.22)),
        UiTransform::from_rotation(Rot2::degrees(-28.0)),
    ));
    icon.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(5),
            top: px(5),
            width: px(30),
            height: px(11),
            border_radius: BorderRadius::all(px(2)),
            ..default()
        },
        BackgroundColor(ICON_COLOR),
        UiTransform::from_rotation(Rot2::degrees(-28.0)),
    ));
}

#[allow(clippy::type_complexity)]
pub(crate) fn update(
    mut selection: ResMut<SelectedTool>,
    mut capture: ResMut<HotbarPointerCapture>,
    mut slots: Query<
        (
            &Interaction,
            &HotbarSlot,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
    surface: Single<&RelativeCursorPosition, With<HotbarSurface>>,
    mut tooltip: Single<(&mut Text, &mut Visibility), With<HotbarTooltip>>,
) {
    capture.0 = surface.cursor_over();
    let mut hovered = None;
    let mut requested = None;
    for (interaction, slot, _, _) in &mut slots {
        if *interaction != Interaction::None {
            capture.0 = true;
            hovered = Some(slot.0);
        }
        if *interaction == Interaction::Pressed {
            requested = Some(slot.0);
        }
    }
    if let Some(tool) = requested {
        selection.0 = tool;
    }

    for (interaction, slot, mut background, mut border) in &mut slots {
        let selected = selection.0 == slot.0;
        background.0 = if selected {
            SLOT_SELECTED_BACKGROUND
        } else if *interaction != Interaction::None {
            SLOT_HOVER_BACKGROUND
        } else {
            SLOT_BACKGROUND
        };
        *border = BorderColor::all(if selected {
            SLOT_SELECTED_BORDER
        } else if *interaction != Interaction::None {
            SLOT_HOVER_BORDER
        } else {
            SLOT_BORDER
        });
    }

    if let Some(tool) = hovered {
        tool.label().clone_into(&mut tooltip.0.0);
        *tooltip.1 = Visibility::Visible;
    } else {
        tooltip.0.0.clear();
        *tooltip.1 = Visibility::Hidden;
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use bevy::ui::RelativeCursorPosition;

    use super::{
        HotbarPointerCapture, HotbarSlot, HotbarSurface, HotbarTooltip, SelectedTool, Tool,
        shortcut_tool, update,
    };

    #[test]
    fn numbered_shortcuts_follow_hotbar_order() {
        assert_eq!(shortcut_tool(KeyCode::Digit1), Some(Tool::Block));
        assert_eq!(shortcut_tool(KeyCode::Digit2), Some(Tool::Bearing));
        assert_eq!(shortcut_tool(KeyCode::Digit3), Some(Tool::Weld));
        assert_eq!(shortcut_tool(KeyCode::Digit4), Some(Tool::Hammer));
    }

    #[test]
    fn tools_act_only_in_their_supported_mode() {
        for tool in [Tool::Block, Tool::Bearing, Tool::Weld] {
            assert!(tool.works_in_mode(false));
            assert!(!tool.works_in_mode(true));
        }
        assert!(!Tool::Hammer.works_in_mode(false));
        assert!(Tool::Hammer.works_in_mode(true));
    }

    #[test]
    fn pressed_slot_selects_tool_and_captures_pointer() {
        let mut app = App::new();
        app.init_resource::<SelectedTool>()
            .init_resource::<HotbarPointerCapture>()
            .add_systems(Update, update);
        app.world_mut().spawn((
            Button,
            HotbarSlot(Tool::Weld),
            Interaction::Pressed,
            BackgroundColor::default(),
            BorderColor::default(),
        ));
        app.world_mut()
            .spawn((HotbarSurface, RelativeCursorPosition::default()));
        let tooltip = app
            .world_mut()
            .spawn((HotbarTooltip, Text::new(""), Visibility::Hidden))
            .id();

        app.update();

        assert_eq!(app.world().resource::<SelectedTool>().0, Tool::Weld);
        assert!(app.world().resource::<HotbarPointerCapture>().active());
        assert_eq!(app.world().entity(tooltip).get::<Text>().unwrap().0, "Weld");
        assert_eq!(
            *app.world().entity(tooltip).get::<Visibility>().unwrap(),
            Visibility::Visible
        );
    }

    #[test]
    fn padding_between_slots_also_captures_pointer() {
        let mut app = App::new();
        app.init_resource::<SelectedTool>()
            .init_resource::<HotbarPointerCapture>()
            .add_systems(Update, update);
        app.world_mut().spawn((
            Button,
            HotbarSlot(Tool::Block),
            Interaction::None,
            BackgroundColor::default(),
            BorderColor::default(),
        ));
        app.world_mut().spawn((
            HotbarSurface,
            RelativeCursorPosition {
                cursor_over: true,
                normalized: Some(Vec2::ZERO),
            },
        ));
        app.world_mut()
            .spawn((HotbarTooltip, Text::new(""), Visibility::Hidden));

        app.update();

        assert!(app.world().resource::<HotbarPointerCapture>().active());
        assert_eq!(app.world().resource::<SelectedTool>().0, Tool::Block);
    }
}
