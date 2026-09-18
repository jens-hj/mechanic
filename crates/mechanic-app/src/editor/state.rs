//! The authored graph and the editor's transient interaction state.

use crate::builder::{
    CylinderPlacementCandidate, PlacementBounds, PlacementCandidate, PlacementError, PlacementGrid,
    PlacementSnapIndex, SmartGuide, SurfaceHit,
};
use crate::editor::build_actions::PlacedBearing;
use crate::editor::history::ChromaStroke;
use crate::editor::hover::{BlockDrag, DeleteDrag, DeleteTarget, clear_hover};
use crate::editor::pipe::{PipeDrag, PipeValidation};
use crate::editor::placement::{FreePlacementSettings, SmartSnapSettings};
use crate::editor::raycast::SimulationHit;
use crate::editor::shape_actions::{DEFAULT_LAYER_THICKNESS_METERS, LayerPreview, RegionDrag};
use crate::editor::wiring::WireDrag;
use crate::simulation::publication::WorldPhysicsRevision;
use crate::{linear_editor, live_edit, shape_tool, suspension_editor, weld_publication, weld_tool};
use bevy::prelude::{Resource, Vec2, Vec3};
use mechanic_core::{CageIndex, ConstructionGraph, PartId, RegionId};

#[derive(Resource, Default)]
pub(crate) struct EditorGraph(pub(crate) ConstructionGraph);

/// Display name of the creation currently open, when one was saved or loaded.
/// It prefills the modal's name field so re-saving keeps the same file.
#[derive(Resource, Default)]
pub(crate) struct CurrentCreation(pub(crate) Option<String>);

#[derive(Resource, Default)]
pub(crate) struct EditorState {
    pub(crate) weld: weld_tool::WeldTool,
    pub(crate) weld_restore: Option<weld_publication::Restore>,
    pub(crate) history_capture: Option<weld_publication::Restore>,
    pub(crate) edit_context: Option<live_edit::EditContext>,
    pub(crate) world_hovered_part: Option<PartId>,
    pub(crate) linear: linear_editor::LinearToolState,
    pub(crate) suspension: suspension_editor::SuspensionToolState,
    pub(crate) linear_attachment: Option<PlacedBearing>,
    pub(crate) placement_bounds: PlacementBounds,
    pub(crate) hovered: Option<SurfaceHit>,
    pub(crate) hovered_simulation: Option<SimulationHit>,
    /// Unattached bearing surface directly hit by the pointer ray.
    pub(crate) hovered_bearing: Option<usize>,
    /// Unattached bearing that would claim the current block preview.
    pub(crate) attachment_bearing: Option<usize>,
    pub(crate) preview: Option<PlacementCandidate>,
    pub(crate) cylinder_preview: Option<CylinderPlacementCandidate>,
    /// Junction planned under the cylinder preview when it branches off a pipe's side.
    pub(crate) pipe_branch_preview: Option<crate::builder::PipeBranch>,
    /// Part the branch preview last planned on, and the player's turns of its new arm.
    pub(crate) pipe_branch_turn: (Option<mechanic_core::PartId>, u8),
    /// Empty-space point offered when an eligible Garage tool misses construction.
    pub(crate) free_placement_point: Option<Vec3>,
    pub(crate) bearing_preview_anchor: Option<Vec3>,
    pub(crate) preview_error: Option<PlacementError>,
    /// A staged action that is allowed but costs something the player should
    /// see first — currently a weld that locks a bearing solid.
    pub(crate) preview_warning: Option<String>,
    pub(crate) placement_grid: PlacementGrid,
    pub(crate) smart_guides: Vec<SmartGuide>,
    pub(crate) smart_snap: SmartSnapSettings,
    pub(crate) free_placement: FreePlacementSettings,
    pub(crate) snap_index: PlacementSnapIndex,
    /// One of the 24 grid-aligned orientations used by authored parts.
    pub(crate) authored_orientation: u8,
    pub(crate) feedback: Option<String>,
    pub(crate) construction_mesh_dirty: bool,
    /// Monotonic identity of the latest synchronously accepted block volume.
    pub(crate) construction_publication_generation: u64,
    /// Last immutable graph revision atomically published to construction meshes.
    pub(crate) rendered_graph: ConstructionGraph,
    pub(crate) rendered_world_revision: Option<WorldPhysicsRevision>,
    pub(crate) delete_target: Option<DeleteTarget>,
    pub(crate) block_drag: Option<BlockDrag>,
    pub(crate) block_preview_revision: u64,
    pub(crate) pipe_drag: Option<PipeDrag>,
    pub(crate) pipe_validation: Option<PipeValidation>,
    /// Wall the Layer tool is pointed at and the layer it would add.
    pub(crate) layer_preview: Option<LayerPreview>,
    /// Radial layer drag in progress.
    pub(crate) layer_drag: Option<crate::builder::LayerDrag>,
    /// Last committed layer thickness in metres; zero until the first layer.
    pub(crate) layer_thickness: f32,
    pub(crate) delete_drag: Option<DeleteDrag>,
    pub(crate) delete_preview_revision: u64,
    pub(crate) placed_bearings: Vec<PlacedBearing>,
    /// Control block the panel edits, and the one a new wire starts from.
    pub(crate) selected_controller: Option<PartId>,
    /// Drive rows changed and a running simulation still holds the old ones.
    pub(crate) drive_rows_dirty: bool,
    /// Drive wire the pointer is dragging out.
    pub(crate) wire_drag: Option<WireDrag>,
    /// Latest pointer ray, so a dragged wire can follow the cursor.
    pub(crate) pointer_ray: Option<(Vec3, Vec3)>,
    /// Latest pointer position paired with [`Self::pointer_ray`].
    pub(crate) pointer_position: Option<Vec2>,
    /// The region the Shape tool is editing. Nothing can be shaped until one is
    /// chosen, and while one is, everything else fades back.
    pub(crate) active_region: Option<RegionId>,
    /// Area being dragged out to become a region.
    pub(crate) region_drag: Option<RegionDrag>,
    /// Cage vertex the pointer is over.
    pub(crate) hovered_vertex: Option<CageIndex>,
    /// Vertex the pointer is dragging.
    pub(crate) vertex_drag: Option<shape_tool::VertexDrag>,
    /// Vertices painted by Shift+left, moved together by one drag.
    pub(crate) selected_vertices: Vec<CageIndex>,
    /// Whether Shift+left is sweeping across cage vertices.
    pub(crate) paint_selecting: bool,
    /// The new cage vertex the pointer is currently being offered.
    pub(crate) edge_offer: Option<shape_tool::EdgeInsertion>,
    /// Construction solid currently focused by Chamfer or Fillet mode.
    pub(crate) feature_focus: Option<mechanic_core::SolidOwner>,
    /// Logical feature edge under the pointer.
    pub(crate) hovered_feature_edge: Option<shape_tool::FeatureEdgeHit>,
    /// Separate logical chains sharing the next feature amount.
    pub(crate) selected_feature_edges: Vec<mechanic_core::EdgeChainRef>,
    /// Chamfer/fillet amount drag in progress.
    pub(crate) feature_drag: Option<shape_tool::FeatureDrag>,
    /// Earlier feature selected through its virtual source overlay.
    pub(crate) selected_shape_feature: Option<mechanic_core::ShapeFeatureId>,
    /// Earlier feature whose dashed source chain is under the pointer.
    pub(crate) hovered_source_feature: Option<mechanic_core::ShapeFeatureId>,
    /// Active paint/remove drag, committed to history as one edit on release.
    pub(crate) chroma_stroke: Option<ChromaStroke>,
}

impl EditorState {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn world_drag_active(&self) -> bool {
        self.contextual_selector_blocked() || self.active_region.is_some()
    }

    /// Whether a transient gesture owns the pointer strongly enough to block
    /// the contextual hold-Tab selector. A Shape focus by itself is retained
    /// across mode changes and therefore does not block the selector.
    pub(crate) fn contextual_selector_blocked(&self) -> bool {
        self.suspension.drag.is_some()
            || self.block_drag.is_some()
            || self.pipe_drag.is_some()
            || self.layer_drag.is_some()
            || self.delete_drag.is_some()
            || self.delete_target.is_some()
            || self.region_drag.is_some()
            || self.vertex_drag.is_some()
            || self.feature_drag.is_some()
            || self.wire_drag.is_some()
            || self.paint_selecting
            || self.chroma_stroke.is_some()
    }

    pub(crate) fn next_layer_thickness(&self) -> f32 {
        if self.layer_thickness > 0.0 {
            self.layer_thickness
        } else {
            DEFAULT_LAYER_THICKNESS_METERS
        }
    }

    pub(crate) fn pipe_bend_active(&self) -> bool {
        self.pipe_drag
            .as_ref()
            .is_some_and(|drag| drag.choosing_direction || !drag.nodes.is_empty())
    }

    pub(crate) fn cancel_delete_gesture(&mut self) -> bool {
        let cancelled_drag = self.delete_drag.take().is_some();
        let cancelled_target = self.delete_target.take().is_some();
        let cancelled = cancelled_drag || cancelled_target;
        if cancelled {
            clear_hover(self);
        }
        cancelled
    }
}
