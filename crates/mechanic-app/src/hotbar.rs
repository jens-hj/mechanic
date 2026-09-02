//! The tools, and which one is in hand.
//!
//! What the bar looks like lives in [`crate::ui::hotbar`]; this is what a tool
//! is, which key picks it, and what it may do in which mode.

use bevy::prelude::*;
use mechanic_core::ConstructionMaterial;
use mechanic_world::TerrainMaterial;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum Tool {
    #[default]
    Block,
    Cylinder,
    Bearing,
    Weld,
    Hammer,
    Controller,
    Connector,
    GasEngine,
    ElectricEngine,
    Transmission,
    Servo,
    Seat,
    Input,
    DimensionLink,
    Shape,
    Chroma,
}

/// The four tools exposed by the primary hotbar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum MainTool {
    #[default]
    MatterManipulator,
    Welder,
    Connector,
    Hammer,
}

impl MainTool {
    pub(crate) const ALL: [Self; 4] = [
        Self::MatterManipulator,
        Self::Welder,
        Self::Connector,
        Self::Hammer,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::MatterManipulator => "Matter Manipulator",
            Self::Welder => "Welder",
            Self::Connector => "Connector",
            Self::Hammer => "Hammer",
        }
    }
}

/// The operation exposed by the Matter Manipulator's secondary hotbar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum MatterMode {
    #[default]
    Block,
    Cylinder,
    Item,
    Terrain,
    Manipulate,
    Chroma,
}

impl MatterMode {
    pub(crate) const ALL: [Self; 6] = [
        Self::Block,
        Self::Cylinder,
        Self::Item,
        Self::Terrain,
        Self::Manipulate,
        Self::Chroma,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Block => "Block",
            Self::Cylinder => "Cylinder",
            Self::Item => "Item Placer",
            Self::Terrain => "Terrain",
            Self::Manipulate => "Manipulate",
            Self::Chroma => "Chroma",
        }
    }

    pub(crate) const fn short_label(self) -> &'static str {
        match self {
            Self::Block => "BLOCK",
            Self::Cylinder => "PIPE",
            Self::Item => "ITEMS",
            Self::Terrain => "TERRAIN",
            Self::Manipulate => "SHAPE",
            Self::Chroma => "CHROMA",
        }
    }
}

/// Placeable selected inside Matter Manipulator → Item Placer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum PlaceableItem {
    #[default]
    Bearing,
    ControlBlock,
    GasEngine,
    ElectricEngine,
    Transmission,
    Servo,
    Seat,
    Input,
    DimensionLink,
}

/// One contextual choice shown by the hold-Tab selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WheelChoice {
    ConstructionMaterial(ConstructionMaterial),
    Item(PlaceableItem),
    TerrainMaterial(TerrainMaterial),
    ShapeMode(crate::shape_tool::ShapeEditMode),
}

/// A radial selector's data model, shared by input and rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WheelContext {
    ConstructionMaterial,
    Item,
    TerrainMaterial,
    Shape,
}

impl WheelChoice {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ConstructionMaterial(material) => material.label(),
            Self::Item(item) => item.label(),
            Self::TerrainMaterial(TerrainMaterial::SurfaceCover) => "Grass",
            Self::TerrainMaterial(TerrainMaterial::Soil) => "Dirt",
            Self::TerrainMaterial(TerrainMaterial::Rock) => "Stone",
            Self::TerrainMaterial(TerrainMaterial::Sand) => "Sand",
            Self::TerrainMaterial(TerrainMaterial::Iron) => "Iron",
            Self::TerrainMaterial(TerrainMaterial::Graphite) => "Graphite",
            Self::ShapeMode(mode) => mode.label(),
        }
    }

    pub(crate) const fn context(self) -> WheelContext {
        match self {
            Self::ConstructionMaterial(_) => WheelContext::ConstructionMaterial,
            Self::Item(_) => WheelContext::Item,
            Self::TerrainMaterial(_) => WheelContext::TerrainMaterial,
            Self::ShapeMode(_) => WheelContext::Shape,
        }
    }
}

impl WheelContext {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ConstructionMaterial => "MATERIALS",
            Self::Item => "ITEMS",
            Self::TerrainMaterial => "TERRAIN",
            Self::Shape => "SHAPE",
        }
    }

    pub(crate) const fn count(self) -> usize {
        match self {
            Self::ConstructionMaterial => ConstructionMaterial::ALL.len(),
            Self::Item => PlaceableItem::ALL.len(),
            Self::TerrainMaterial => TerrainMaterial::ALL.len(),
            Self::Shape => crate::shape_tool::ShapeEditMode::ALL.len(),
        }
    }

    pub(crate) fn choice(self, index: usize) -> Option<WheelChoice> {
        match self {
            Self::ConstructionMaterial => ConstructionMaterial::ALL
                .get(index)
                .copied()
                .map(WheelChoice::ConstructionMaterial),
            Self::Item => PlaceableItem::ALL
                .get(index)
                .copied()
                .map(WheelChoice::Item),
            Self::TerrainMaterial => TerrainMaterial::ALL
                .get(index)
                .copied()
                .map(WheelChoice::TerrainMaterial),
            Self::Shape => crate::shape_tool::ShapeEditMode::ALL
                .get(index)
                .copied()
                .map(WheelChoice::ShapeMode),
        }
    }

    pub(crate) fn choices(self) -> impl Iterator<Item = WheelChoice> {
        (0..self.count()).filter_map(move |index| self.choice(index))
    }
}

impl PlaceableItem {
    pub(crate) const ALL: [Self; 9] = [
        Self::Bearing,
        Self::ControlBlock,
        Self::GasEngine,
        Self::ElectricEngine,
        Self::Transmission,
        Self::Servo,
        Self::Seat,
        Self::Input,
        Self::DimensionLink,
    ];

    pub(crate) const fn label(self) -> &'static str {
        self.editor_tool().label()
    }

    pub(crate) const fn editor_tool(self) -> Tool {
        match self {
            Self::Bearing => Tool::Bearing,
            Self::ControlBlock => Tool::Controller,
            Self::GasEngine => Tool::GasEngine,
            Self::ElectricEngine => Tool::ElectricEngine,
            Self::Transmission => Tool::Transmission,
            Self::Servo => Tool::Servo,
            Self::Seat => Tool::Seat,
            Self::Input => Tool::Input,
            Self::DimensionLink => Tool::DimensionLink,
        }
    }

    pub(crate) const fn from_editor_tool(tool: Tool) -> Option<Self> {
        match tool {
            Tool::Bearing => Some(Self::Bearing),
            Tool::Controller => Some(Self::ControlBlock),
            Tool::GasEngine => Some(Self::GasEngine),
            Tool::ElectricEngine => Some(Self::ElectricEngine),
            Tool::Transmission => Some(Self::Transmission),
            Tool::Servo => Some(Self::Servo),
            Tool::Seat => Some(Self::Seat),
            Tool::Input => Some(Self::Input),
            Tool::DimensionLink => Some(Self::DimensionLink),
            _ => None,
        }
    }
}

impl Tool {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Block => "Blocker Placer",
            Self::Cylinder => "Pipe / Cylinder",
            Self::Bearing => "Bearing",
            Self::Weld => "Weld",
            Self::Hammer => "Hammer",
            Self::Controller => "Control Block",
            Self::Connector => "Connector",
            Self::GasEngine => "Gas Engine",
            Self::ElectricEngine => "Electric Engine",
            Self::Transmission => "Transmission",
            Self::Servo => "Servo",
            Self::Seat => "Seat",
            Self::Input => "Input",
            Self::DimensionLink => "Dimension Link",
            Self::Shape => "Shape",
            Self::Chroma => "Chroma",
        }
    }

    /// Whether this tool works with control blocks and their wires.
    pub(crate) const fn edits_drives(self) -> bool {
        matches!(self, Self::Controller | Self::Connector)
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectedTool {
    pub(crate) tool: Option<MainTool>,
    pub(crate) matter_mode: MatterMode,
    pub(crate) item: PlaceableItem,
}

impl Default for SelectedTool {
    fn default() -> Self {
        Self {
            tool: Some(MainTool::MatterManipulator),
            matter_mode: MatterMode::Block,
            item: PlaceableItem::Bearing,
        }
    }
}

impl SelectedTool {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_editor_tool(tool: Tool) -> Self {
        let mut selection = Self::default();
        selection.select_editor_tool(tool);
        selection
    }

    pub(crate) const fn active_editor_tool(self) -> Option<Tool> {
        let Some(tool) = self.tool else {
            return None;
        };
        match tool {
            MainTool::MatterManipulator => match self.matter_mode {
                MatterMode::Block => Some(Tool::Block),
                MatterMode::Cylinder => Some(Tool::Cylinder),
                MatterMode::Item => Some(self.item.editor_tool()),
                MatterMode::Terrain => None,
                MatterMode::Manipulate => Some(Tool::Shape),
                MatterMode::Chroma => Some(Tool::Chroma),
            },
            MainTool::Welder => Some(Tool::Weld),
            MainTool::Connector => Some(Tool::Connector),
            MainTool::Hammer => Some(Tool::Hammer),
        }
    }

    pub(crate) fn select_tool(&mut self, tool: MainTool) {
        self.tool = Some(tool);
    }

    pub(crate) fn select_mode(&mut self, mode: MatterMode) {
        self.tool = Some(MainTool::MatterManipulator);
        self.matter_mode = mode;
    }

    pub(crate) fn select_item(&mut self, item: PlaceableItem) {
        self.item = item;
        self.select_mode(MatterMode::Item);
    }

    pub(crate) fn select_editor_tool(&mut self, tool: Tool) {
        match tool {
            Tool::Block => self.select_mode(MatterMode::Block),
            Tool::Cylinder => self.select_mode(MatterMode::Cylinder),
            Tool::Shape => self.select_mode(MatterMode::Manipulate),
            Tool::Chroma => self.select_mode(MatterMode::Chroma),
            Tool::Weld => self.select_tool(MainTool::Welder),
            Tool::Connector => self.select_tool(MainTool::Connector),
            Tool::Hammer => self.select_tool(MainTool::Hammer),
            item => self.select_item(
                PlaceableItem::from_editor_tool(item)
                    .expect("every remaining editor tool is a placeable item"),
            ),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tool = None;
    }
}

/// Material shared by the Blocker Placer and Cylinder for this process.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SelectedMaterial(pub(crate) ConstructionMaterial);

/// Terrain material remembered by Matter Manipulator → Terrain.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectedTerrainMaterial(pub(crate) TerrainMaterial);

impl Default for SelectedTerrainMaterial {
    fn default() -> Self {
        Self(TerrainMaterial::Soil)
    }
}

#[cfg(test)]
mod tests {
    use super::{MainTool, MatterMode, PlaceableItem, SelectedTool, Tool};

    #[test]
    fn defaults_to_matter_block_and_remembers_context() {
        let mut selected = SelectedTool::default();
        assert_eq!(selected.tool, Some(MainTool::MatterManipulator));
        assert_eq!(selected.active_editor_tool(), Some(Tool::Block));

        selected.select_item(PlaceableItem::Servo);
        selected.select_tool(MainTool::Welder);
        selected.select_mode(MatterMode::Item);
        assert_eq!(selected.item, PlaceableItem::Servo);
        assert_eq!(selected.active_editor_tool(), Some(Tool::Servo));
    }

    #[test]
    fn dimension_link_is_the_ninth_item_choice() {
        assert_eq!(
            PlaceableItem::ALL,
            [
                PlaceableItem::Bearing,
                PlaceableItem::ControlBlock,
                PlaceableItem::GasEngine,
                PlaceableItem::ElectricEngine,
                PlaceableItem::Transmission,
                PlaceableItem::Servo,
                PlaceableItem::Seat,
                PlaceableItem::Input,
                PlaceableItem::DimensionLink,
            ]
        );
        assert_eq!(
            PlaceableItem::DimensionLink.editor_tool(),
            Tool::DimensionLink
        );
    }

    #[test]
    fn item_mode_is_named_as_a_placer_in_the_ui() {
        assert_eq!(MatterMode::Item.label(), "Item Placer");
        assert_eq!(MatterMode::Item.short_label(), "ITEMS");
    }

    #[test]
    fn every_legacy_placement_path_maps_to_a_matter_mode() {
        for item in PlaceableItem::ALL {
            let mut selected = SelectedTool::default();
            selected.select_editor_tool(item.editor_tool());
            assert_eq!(selected.tool, Some(MainTool::MatterManipulator));
            assert_eq!(selected.matter_mode, MatterMode::Item);
            assert_eq!(selected.item, item);
        }
        let mut selected = SelectedTool::default();
        selected.select_editor_tool(Tool::Shape);
        assert_eq!(selected.matter_mode, MatterMode::Manipulate);
    }
}
