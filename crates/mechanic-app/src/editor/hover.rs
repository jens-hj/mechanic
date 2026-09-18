//! What the cursor is over, and the block, bearing-offset, and delete drags that follow it.

use crate::editor::raycast::{raycast_placed_bearings, raycast_rotational_bearings};
use crate::*;

/// Roughly the old five-pixel threshold at a typical desktop field of view.
pub(crate) const DRAG_DEAD_ZONE_RADIANS: f32 = 0.004;

#[derive(Clone, Debug)]
pub(crate) struct BlockDrag {
    pub(crate) start: PlacementCandidate,
    pub(crate) attachment: BlockAttachment,
    /// Smart guides acquired by the starting block before the gesture began.
    pub(crate) start_guides: Vec<SmartGuide>,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    pub(crate) press: PointerSample,
    pub(crate) plane: PlacementPlane,
    /// Span at the last press or plane rotation. The plane's own two axes grow
    /// from here; the third keeps what it already had.
    pub(crate) anchor_span: IVec3,
    /// Blocks beyond the start block along each axis, signed.
    pub(crate) span: IVec3,
    pub(crate) last_span: Option<IVec3>,
    pub(crate) volume: BlockVolume,
    pub(crate) error: Option<PlacementError>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PointerSample {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) cursor: Vec2,
    pub(crate) ray_origin: Vec3,
    pub(crate) ray_direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BlockAttachment {
    Linear {
        socket: PlacedBearing,
    },
    AutoWeld {
        source: FaceOwner,
    },
    Free,
    Bearing {
        source: mechanic_core::FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct DeleteDrag {
    pub(crate) start: CuboidSpec,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    pub(crate) press: PointerSample,
    pub(crate) plane: PlacementPlane,
    /// Span at the last press or plane rotation.
    pub(crate) anchor_span: IVec3,
    /// Blocks beyond the start block along each axis, signed.
    pub(crate) span: IVec3,
    pub(crate) last_span: Option<IVec3>,
    pub(crate) parts: Vec<PartId>,
    pub(crate) error: Option<PlacementError>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BearingOffsetDrag {
    pub(crate) start: Vec3,
    pub(crate) endpoint: Vec3,
    pub(crate) normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DeleteTarget {
    PlacedBearing(usize),
    Part(PartId),
}

pub(crate) fn handle_tool_change(
    selection: Res<SelectedTool>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
) {
    if !selection.is_changed() {
        return;
    }
    cancel_transient_editor_state(&mut graph.0, &mut state);
    if !selection
        .active_editor_tool()
        .is_some_and(Tool::edits_drives)
    {
        state.selected_controller = None;
    }
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn update_hover(
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    actions: Res<ButtonInput<GameAction>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    selected_material: Option<Res<SelectedMaterial>>,
    chroma_brush: Res<ChromaBrush>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    space: Res<State<world::AppSpace>>,
    world_runtime: Res<world::WorldRuntime>,
) {
    suspension_editor::sync_sockets(&graph.0, &mut state);
    let placement_bounds = match space.get() {
        world::AppSpace::Garage => PlacementBounds::GarageBuild,
        world::AppSpace::World => PlacementBounds::World {
            origin: world_runtime.horizontal_origin(),
        },
    };
    state.placement_bounds = placement_bounds;
    if state.block_drag.is_none() && state.pipe_drag.is_none() {
        state.placement_grid = active_placement_grid(&actions);
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        clear_hover(&mut state);
        return;
    }
    let cursor = camera::viewport_center(Vec2::new(window.width(), window.height()));
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        state.pointer_position = None;
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.pipe_drag.is_some() {
            invalidate_pipe_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        if let Some(tool) = selection.active_editor_tool() {
            refresh_tool_preview_with_cylinder(
                &graph.0,
                &mut state,
                tool,
                cylinder_settings.dimensions,
                bearing_settings.dimensions,
                selected_material
                    .as_deref()
                    .map_or(ConstructionMaterial::Steel, |value| value.0),
                chroma_brush.appearance,
            );
        }
        return;
    };
    if selection.active_editor_tool() == Some(Tool::Weld) {
        weld_tool::sync_mode(&mut state, selection.weld_mode);
        match selection.weld_mode {
            hotbar::WeldMode::Join => weld_tool::join_hover(&graph.0, &simulation, &mut state, ray),
            hotbar::WeldMode::Place => weld_tool::hover(
                &graph.0,
                &simulation,
                &mut state,
                ray,
                &actions,
                &world_runtime,
            ),
        }
        return;
    }
    state.weld.cancel();
    state.pointer_position = Some(cursor);
    let suspension_pick = suspension_render::raycast_scene_component(
        &graph.0,
        Some(&simulation),
        &state.placed_bearings,
        ray.origin,
        ray.direction.as_vec3(),
    );
    state.suspension.picked_component = suspension_pick.and_then(|(index, _, owner)| {
        let mechanic_core::BearingKind::Suspension(spec) = state.placed_bearings[index].kind else {
            return None;
        };
        Some((index, suspension_editor::component_index(spec, owner)))
    });
    let ray_direction = ray.direction.as_vec3();
    let terrain_ground = placement_bounds
        .is_world()
        .then(|| world_runtime.raycast_terrain(ray.origin, ray_direction, 64.0))
        .flatten()
        .map(|(point, distance)| SurfaceHit {
            distance,
            point,
            face: FaceRef::ground(),
        });
    let moving_hit = (*space.get() == world::AppSpace::World)
        .then(|| {
            simulation
                .creation
                .as_ref()
                .and_then(|creation| {
                    raycast_simulation(
                        &simulation.published_graph,
                        creation,
                        &simulation.transforms,
                        ray.origin,
                        ray_direction,
                    )
                })
                .filter(|hit| {
                    !simulation.creation.as_ref().is_some_and(|creation| {
                        creation.compounds[hit.body_index as usize].is_static
                    })
                })
        })
        .flatten();
    state.hovered_simulation = moving_hit;
    let nearest_editable = builder::raycast_construction_filtered_with_ground(
        &graph.0,
        ray.origin,
        ray_direction,
        terrain_ground,
        |part| editor_part_is_static_or_pending(&graph.0, &simulation, part),
    );
    state.world_hovered_part = if moving_hit
        .is_some_and(|moving| nearest_editable.is_none_or(|hit| moving.distance <= hit.distance))
    {
        moving_hit.map(|hit| hit.part)
    } else {
        hovered_part(nearest_editable)
    };
    let attached_gesture = state.suspension.drag.is_some()
        || state.block_drag.is_some()
        || state.pipe_drag.is_some()
        || state.delete_drag.is_some()
        || state.region_drag.is_some()
        || state.vertex_drag.is_some()
        || state.feature_drag.is_some()
        || state.wire_drag.is_some()
        || graph.0.pending().is_some();
    let prior = state
        .edit_context
        .map(|context| (context.anchor, context.frame));
    let anchor = if attached_gesture && state.edit_context.is_some() {
        state.edit_context.map(|context| context.anchor)
    } else if let Some((index, distance, _)) = suspension_pick
        && nearest_editable.is_none_or(|hit| distance <= hit.distance)
        && moving_hit.is_none_or(|hit| distance <= hit.distance)
    {
        let socket = state.placed_bearings[index];
        let rubber_stop = selection.active_editor_tool() == Some(Tool::Cylinder)
            && selected_material
                .as_deref()
                .is_some_and(|m| m.0 == ConstructionMaterial::Rubber)
            && matches!(socket.kind, mechanic_core::BearingKind::Suspension(spec) if spec.shock().is_some());
        let opposite = matches!(
            selection.active_editor_tool(),
            Some(Tool::Block | Tool::Cylinder)
        ) && !rubber_stop;
        let source = match socket.source.owner {
            FaceOwner::Part(part) => Some(part),
            FaceOwner::Ground => None,
        };
        if opposite {
            bearing_socket_targets(&graph.0, socket)
                .first()
                .copied()
                .or(source)
        } else {
            source
        }
    } else if moving_hit
        .is_some_and(|moving| nearest_editable.is_none_or(|hit| moving.distance <= hit.distance))
    {
        moving_hit.map(|hit| hit.part)
    } else {
        hovered_part(nearest_editable).filter(|&part| {
            graph.0.part_frame(part) != Some(mechanic_core::ConstructionFrame::IDENTITY)
        })
    };
    state.edit_context =
        anchor.and_then(|part| live_edit::EditContext::resolve(&graph.0, &simulation, part));
    if attached_gesture && prior.is_some() && state.edit_context.is_none() {
        cancel_transient_editor_state(&mut graph.0, &mut state);
        state.feedback = Some("The gesture's construction was removed".to_owned());
        return;
    }
    let context = state.edit_context;
    let mut view = live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    if prior != context.map(|context| (context.anchor, context.frame)) {
        state.snap_index.rebuild(&graph.0);
    }
    let world_ray = ray;
    let ray = context.map_or(ray, |context| context.ray(ray));
    let ray_direction = ray.direction.as_vec3();
    state.pointer_ray = Some((ray.origin, ray_direction));
    let terrain_ground = if context.is_some() {
        None
    } else {
        terrain_ground
    };
    let placement_bounds = context.map_or(placement_bounds, |context| {
        placement_bounds.in_edit_frame(context.frame_to_world)
    });
    state.placement_bounds = placement_bounds;
    let accepts_part = |part| {
        context.map_or_else(
            || {
                !placement_bounds.is_world()
                    || editor_part_is_static_or_pending(&graph.0, &simulation, part)
            },
            |context| context.accepts_part(&graph.0, &simulation, part),
        )
    };
    let nearest_editable = builder::raycast_construction_filtered_with_ground(
        &graph.0,
        ray.origin,
        ray_direction,
        terrain_ground,
        accepts_part,
    );
    let raycast_surface = |annulus: Option<(f32, f32)>| {
        let hit = match annulus {
            Some((inner, outer)) if context.is_some() => {
                builder::raycast_construction_for_annulus_filtered_with_ground(
                    &graph.0,
                    ray.origin,
                    ray_direction,
                    inner,
                    outer,
                    terrain_ground,
                    accepts_part,
                )
            }
            Some((inner, outer)) if placement_bounds.is_world() => {
                raycast_construction_for_annulus_with_ground(
                    &graph.0,
                    ray.origin,
                    ray_direction,
                    inner,
                    outer,
                    terrain_ground,
                )
            }
            Some((inner, outer)) if placement_bounds == PlacementBounds::GarageBuild => {
                raycast_construction_for_annulus_with_ground(
                    &graph.0,
                    ray.origin,
                    ray_direction,
                    inner,
                    outer,
                    None,
                )
            }
            Some((inner, outer)) => {
                raycast_construction_for_annulus(&graph.0, ray.origin, ray_direction, inner, outer)
            }
            None => nearest_editable,
        };
        let hit = hit.filter(|hit| match hit.face.owner {
            FaceOwner::Ground => true,
            FaceOwner::Part(part) => accepts_part(part),
        });
        if context.is_none()
            && moving_hit.is_some_and(|moving| {
                hit.is_none_or(|editable| moving.distance <= editable.distance)
            })
        {
            None
        } else {
            hit
        }
    };
    if state.block_drag.is_some() {
        refresh_block_drag(&graph.0, state, cursor, ray.origin, ray.direction.as_vec3());
        return;
    }
    if state.pipe_drag.is_some() {
        refresh_pipe_drag(&graph.0, state, cursor, ray.origin, ray.direction.as_vec3());
        return;
    }
    if state.delete_drag.is_some() {
        refresh_delete_drag(
            &graph.0,
            state,
            &simulation,
            cursor,
            ray.origin,
            ray.direction.as_vec3(),
        );
        return;
    }
    let geometric_bearing_hit = raycast_rotational_bearings(
        &state.placed_bearings,
        ray.origin,
        ray_direction,
        crate::editor::raycast::BearingPick::Ring,
        |s| Some((s.anchor, s.axis)),
    )
    .into_iter()
    .chain(linear_editor::raycast_scene(
        &graph.0,
        None,
        &state.placed_bearings,
        ray.origin,
        ray_direction,
    ))
    .chain(suspension_pick.map(|(index, distance, _)| (index, distance)))
    .min_by(|a, b| a.1.total_cmp(&b.1));
    let Some(tool) = selection.active_editor_tool() else {
        let construction_hit = raycast_surface(None);
        let bearing_hit = geometric_bearing_hit;
        if let Some((bearing, distance)) = bearing_hit
            && construction_hit.is_none_or(|hit| distance <= hit.distance)
        {
            state.hovered = construction_hit;
            state.hovered_bearing = Some(bearing);
        } else {
            state.hovered = construction_hit;
            state.hovered_bearing = None;
        }
        state.preview = None;
        state.cylinder_preview = None;
        state.pipe_branch_preview = None;
        return;
    };
    let construction_hit = if actions.pressed(GameAction::Secondary) {
        raycast_surface(None)
    } else {
        match tool {
            Tool::Bearing => raycast_surface(Some((
                bearing_settings.dimensions.inner_diameter(),
                bearing_settings.dimensions.outer_diameter(),
            ))),
            Tool::Cylinder => raycast_surface(Some((
                cylinder_settings.dimensions.inner_diameter(),
                cylinder_settings.dimensions.outer_diameter(),
            ))),
            Tool::LinearBearing
            | Tool::Spring
            | Tool::Shock
            | Tool::Block
            | Tool::Weld
            | Tool::Hammer
            | Tool::Controller
            | Tool::Connector
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Transmission
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink
            | Tool::Shape
            | Tool::Layer
            | Tool::Chroma => raycast_surface(None),
        }
    };
    // Wiring aims at the whole joint, hole and pin included, so a wire can be
    // dropped on a bearing without having to hit its thin ring.
    let wiring = tool == Tool::Connector;
    let canonical_wiring_graph = wiring.then(|| graph.0.canonicalized());
    let canonical_wiring_bearings = wiring.then(|| {
        state
            .placed_bearings
            .iter()
            .map(|&bearing| live_edit::transform_bearing(bearing, graph.0.view_to_build()))
            .collect::<Vec<_>>()
    });
    let bearing_hit = if wiring {
        raycast_placed_bearings(
            canonical_wiring_graph
                .as_ref()
                .expect("wiring graph exists"),
            Some(&simulation),
            canonical_wiring_bearings
                .as_ref()
                .expect("wiring sockets exist"),
            world_ray.origin,
            world_ray.direction.as_vec3(),
            crate::editor::raycast::BearingPick::Disc,
        )
        .or_else(|| {
            raycast_placed_bearings(
                canonical_wiring_graph
                    .as_ref()
                    .expect("wiring graph exists"),
                Some(&simulation),
                canonical_wiring_bearings
                    .as_ref()
                    .expect("wiring sockets exist"),
                world_ray.origin,
                world_ray.direction.as_vec3(),
                crate::editor::raycast::BearingPick::Ring,
            )
        })
    } else if matches!(
        tool,
        Tool::Block | Tool::Cylinder | Tool::Spring | Tool::Shock | Tool::Chroma
    ) || actions.pressed(GameAction::Secondary)
    {
        geometric_bearing_hit
    } else {
        None
    };
    // A joint is usually buried under the parts it carries. The overlay draws it
    // through them, so wiring picks it through them too -- otherwise a bearing
    // is only clickable from the one angle where nothing covers it.
    if let Some((bearing, distance)) = bearing_hit
        && (wiring || construction_hit.is_none_or(|hit| distance <= hit.distance))
    {
        state.free_placement_point = None;
        state.hovered = construction_hit;
        state.hovered_bearing = Some(bearing);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            state,
            tool,
            cylinder_settings.dimensions,
            bearing_settings.dimensions,
            selected_material
                .as_deref()
                .map_or(ConstructionMaterial::Steel, |value| value.0),
            chroma_brush.appearance,
        );
        return;
    }
    let Some(hit) = construction_hit else {
        let free_point = free_placement_point_on_miss(
            tool,
            placement_bounds,
            ray.origin,
            ray_direction,
            state.free_placement.range,
            actions.pressed(GameAction::Secondary),
        );
        clear_editor_hover(state);
        state.free_placement_point = free_point;
        refresh_tool_preview_with_cylinder(
            &graph.0,
            state,
            tool,
            cylinder_settings.dimensions,
            bearing_settings.dimensions,
            selected_material
                .as_deref()
                .map_or(ConstructionMaterial::Steel, |value| value.0),
            chroma_brush.appearance,
        );
        return;
    };
    state.free_placement_point = None;
    state.hovered_bearing = None;
    state.hovered = Some(hit);
    refresh_tool_preview_with_cylinder(
        &graph.0,
        state,
        tool,
        cylinder_settings.dimensions,
        bearing_settings.dimensions,
        selected_material
            .as_deref()
            .map_or(ConstructionMaterial::Steel, |value| value.0),
        chroma_brush.appearance,
    );
}

#[expect(clippy::too_many_lines)]
pub(crate) fn refresh_block_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, start_guides, press, plane, anchor_span, last_span) = {
        let drag = state
            .block_drag
            .as_ref()
            .expect("block drag was checked by caller");
        (
            drag.start,
            drag.start_guides.clone(),
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let gridded_span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        let Some(span) = block_span_from_rays(
            start.spec,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            invalidate_block_drag(state, PlacementError::DragPlaneUnavailable);
            return;
        };
        span
    } else {
        anchor_span
    };
    let (span, endpoint_guides) = if state.smart_snap.enabled {
        raycast_placement_plane_point(ray_origin, ray_direction, start.spec, plane).map_or(
            (gridded_span, Vec::new()),
            |pointer| {
                let bounds = state.placement_bounds;
                smart_snap_block_span(
                    &state.snap_index,
                    start.spec,
                    plane,
                    gridded_span,
                    pointer,
                    state.smart_snap.range,
                    |guided_span| {
                        BlockVolume::new(start.spec, guided_span).is_ok_and(|volume| {
                            validate_block_volume_in_bounds(
                                graph,
                                &state.snap_index,
                                start,
                                volume,
                                bounds,
                            )
                            .is_ok()
                        })
                    },
                )
            },
        )
    } else {
        (gridded_span, Vec::new())
    };
    let mut combined_guides = if state.smart_snap.enabled {
        start_guides
    } else {
        Vec::new()
    };
    for guide in endpoint_guides {
        if !combined_guides.contains(&guide) {
            combined_guides.push(guide);
        }
    }
    if last_span == Some(span) && state.smart_guides == combined_guides {
        return;
    }
    let result = BlockVolume::new(start.spec, span).and_then(|volume| {
        validate_block_volume_in_bounds(
            graph,
            &state.snap_index,
            start,
            volume,
            state.placement_bounds,
        )?;
        Ok(volume)
    });
    let drag = state
        .block_drag
        .as_mut()
        .expect("block drag remains active while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
    state.smart_guides = combined_guides;
    match result {
        Ok(volume) => {
            drag.volume = volume;
            drag.error = None;
            state.preview_error = None;
        }
        Err(error) => {
            drag.error = Some(error.clone());
            state.preview_error = Some(error);
        }
    }
    state.block_preview_revision = state.block_preview_revision.wrapping_add(1);
}

pub(crate) fn invalidate_block_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .block_drag
        .as_mut()
        .expect("block drag was checked by caller");
    if drag.error.as_ref() != Some(&error) {
        drag.error = Some(error.clone());
        state.preview_error = Some(error);
    }
}

pub(crate) fn refresh_bearing_offset_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> bool {
    let Some((press, offset)) = state
        .pipe_drag
        .as_ref()
        .and_then(|drag| drag.bearing_offset.map(|offset| (drag.press, offset)))
    else {
        return false;
    };
    if !camera::ray_drag_started(press.ray_direction, ray_direction) {
        return true;
    }
    let Some(delta) = bearing_offset_from_rays(
        offset.start,
        offset.normal,
        state.placement_grid,
        press.ray_origin,
        press.ray_direction,
        ray_origin,
        ray_direction,
    ) else {
        invalidate_pipe_drag(state, PlacementError::DragPlaneUnavailable);
        return true;
    };
    let drag = state.pipe_drag.as_mut().expect("pipe drag remains active");
    drag.start = offset.start + delta;
    drag.endpoint = offset.endpoint + delta;
    rebuild_pipe_drag(graph, state);
    true
}

pub(crate) fn bearing_offset_from_rays(
    plane_origin: Vec3,
    plane_normal: Vec3,
    grid: PlacementGrid,
    press_origin: Vec3,
    press_direction: Vec3,
    current_origin: Vec3,
    current_direction: Vec3,
) -> Option<Vec3> {
    fn intersection(origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
        let denominator = direction.dot(normal);
        if !origin.is_finite() || !direction.is_finite() || denominator.abs() <= f32::EPSILON {
            return None;
        }
        let distance = (point - origin).dot(normal) / denominator;
        (distance >= 0.0 && distance.is_finite()).then_some(origin + direction * distance)
    }

    let press = intersection(press_origin, press_direction, plane_origin, plane_normal)?;
    let current = intersection(
        current_origin,
        current_direction,
        plane_origin,
        plane_normal,
    )?;
    let step = grid.step_meters();
    let mut offset = ((current - press) / step).round() * step;
    offset -= plane_normal * offset.dot(plane_normal);
    Some(offset)
}

pub(crate) fn refresh_delete_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    simulation: &AppSimulation,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, press, plane, anchor_span, last_span) = {
        let drag = state
            .delete_drag
            .as_ref()
            .expect("delete drag was checked by caller");
        (
            drag.start,
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        let Some(span) = block_span_from_rays(
            start,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            invalidate_delete_drag(state, PlacementError::DragPlaneUnavailable);
            return;
        };
        span
    } else {
        anchor_span
    };
    if last_span == Some(span) {
        return;
    }
    let result = delete_box_parts(graph, start, span).map(|mut parts| {
        if let Some(context) = state.edit_context {
            parts.retain(|&part| context.accepts_part(graph, simulation, part));
        }
        parts
    });
    let drag = state
        .delete_drag
        .as_mut()
        .expect("delete drag remains active while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
    match result {
        Ok(parts) => {
            drag.parts = parts;
            drag.error = None;
        }
        Err(error) => drag.error = Some(error),
    }
    state.delete_preview_revision = state.delete_preview_revision.wrapping_add(1);
}

pub(crate) fn invalidate_delete_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .delete_drag
        .as_mut()
        .expect("delete drag was checked by caller");
    if drag.error.as_ref() != Some(&error) {
        drag.error = Some(error);
    }
}

/// Every block whose centre falls inside the cuboid a delete drag spans.
pub(crate) fn delete_box_parts(
    graph: &ConstructionGraph,
    start: CuboidSpec,
    span: IVec3,
) -> Result<Vec<PartId>, PlacementError> {
    let centers = block_box_specs(start, span)?
        .into_iter()
        .map(|spec| spec.pose.translation_position_ticks())
        .collect::<HashSet<_>>();
    Ok(graph
        .parts()
        .filter_map(|(part, spec)| {
            matches!(spec, PartSpec::Cuboid(_))
                .then(|| {
                    graph.part_position(part).is_some_and(|point| {
                        let ticks = point / mechanic_core::POSITION_TICK_METERS;
                        let rounded = ticks.round();
                        ticks.abs_diff_eq(rounded, 1.0e-3) && centers.contains(&rounded.as_ivec3())
                    })
                })
                .unwrap_or(false)
                .then_some(part)
        })
        .collect())
}

/// Describes what a staged weld costs, or nothing when it costs nothing.
pub(crate) fn weld_lockup_warning(
    before: &ConstructionGraph,
    after: &ConstructionGraph,
) -> Option<String> {
    match builder::newly_locked_bearings(before, after) {
        0 => None,
        1 => Some("This weld locks 1 bearing solid".to_owned()),
        count => Some(format!("This weld locks {count} bearings solid")),
    }
}

pub(crate) fn clear_hover(state: &mut EditorState) {
    state.suspension.preview = None;
    state.suspension.picked_component = None;
    state.suspension.attachment = None;
    state.suspension.insertion = None;
    state.hovered = None;
    state.hovered_simulation = None;
    state.world_hovered_part = None;
    state.hovered_bearing = None;
    state.attachment_bearing = None;
    state.linear_attachment = None;
    state.preview = None;
    state.cylinder_preview = None;
    state.pipe_branch_preview = None;
    state.free_placement_point = None;
    state.bearing_preview_anchor = None;
    state.preview_error = None;
    state.preview_warning = None;
    state.smart_guides.clear();
}

/// Clears editor-only targeting without discarding a live simulation hit.
pub(crate) fn clear_editor_hover(state: &mut EditorState) {
    let hovered_simulation = state.hovered_simulation;
    let world_hovered_part = state.world_hovered_part;
    clear_hover(state);
    state.hovered_simulation = hovered_simulation;
    state.world_hovered_part = world_hovered_part;
}

#[expect(clippy::too_many_lines)]
pub(crate) fn refresh_tool_preview_with_cylinder(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    tool: Tool,
    cylinder_dimensions: CylinderDimensions,
    bearing_dimensions: BearingDimensions,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) {
    let placement_grid = state.placement_grid;
    state.preview = None;
    state.cylinder_preview = None;
    state.pipe_branch_preview = None;
    state.layer_preview = None;
    state.bearing_preview_anchor = None;
    state.attachment_bearing = None;
    state.linear_attachment = None;
    state.preview_warning = None;
    state.smart_guides.clear();
    if suspension_editor::refresh(graph, state, tool, material, cylinder_dimensions) {
        return;
    }
    // A shaped face is no longer an axis-aligned rectangle, so nothing can sit
    // flush on it until it is flattened back onto the grid.
    if let Some(hit) = state.hovered
        && !state
            .hovered_bearing
            .and_then(|index| state.placed_bearings.get(index))
            .is_some_and(|socket| matches!(socket.kind, mechanic_core::BearingKind::Linear(_)))
        && !builder::face_is_flat(graph, hit.face)
        && !matches!(tool, Tool::Shape | Tool::Hammer)
    {
        state.preview_error = Some(PlacementError::SurfaceNotFlat);
        return;
    }
    if matches!(tool, Tool::Block | Tool::Cylinder)
        && let Some(index) = state.hovered_bearing
        && state
            .placed_bearings
            .get(index)
            .is_some_and(|socket| matches!(socket.kind, mechanic_core::BearingKind::Linear(_)))
    {
        state.preview = None;
        state.cylinder_preview = None;
        state.pipe_branch_preview = None;
        state.attachment_bearing = None;
        let occupied = graph
            .bearings()
            .find_map(|(_, bearing)| {
                (bearing_uses_socket(bearing, state.placed_bearings[index])).then_some(bearing.kind)
            })
            .and_then(|kind| {
                if let mechanic_core::BearingKind::Linear(rail) = kind {
                    Some(rail.face)
                } else {
                    None
                }
            });
        if let Some((socket, point)) =
            linear_editor::selected_socket_on_face(state, index, occupied)
        {
            let mechanic_core::BearingKind::Linear(rail) = socket.kind else {
                unreachable!()
            };
            state.attachment_bearing = Some(index);
            state.linear_attachment = Some(socket);
            let targets = bearing_socket_targets(graph, socket);
            state.preview_error = if tool == Tool::Block {
                match builder::linear_block_candidate(
                    socket.anchor,
                    rail,
                    socket.axis,
                    point,
                    [1; 3],
                    GridRotation::default(),
                ) {
                    Ok(mut candidate) => {
                        candidate.spec = candidate
                            .spec
                            .with_material(material)
                            .with_appearance(appearance);
                        state.preview = Some(candidate);
                        builder::stage_linear_block_batch_in_bounds(
                            graph,
                            candidate,
                            &[candidate.spec],
                            linear_editor::attachment(socket, &targets),
                            state.placement_bounds,
                        )
                        .err()
                    }
                    Err(error) => Some(error),
                }
            } else {
                match builder::linear_cylinder_candidate(
                    socket.anchor,
                    rail,
                    socket.axis,
                    point,
                    cylinder_dimensions,
                    0,
                ) {
                    Ok(mut candidate) => {
                        candidate.spec = candidate
                            .spec
                            .with_material(material)
                            .with_appearance(appearance);
                        state.cylinder_preview = Some(candidate);
                        builder::stage_linear_cylinder_in_bounds(
                            graph,
                            candidate,
                            linear_editor::attachment(socket, &targets),
                            state.placement_bounds,
                        )
                        .err()
                    }
                    Err(error) => Some(error),
                }
            };
        }
        return;
    }
    state.preview_error = match (tool, graph.pending()) {
        (Tool::Block, _) => {
            let surface_candidate = state.hovered.and_then(|hit| {
                (hit.face.patch.is_some()
                    || try_face_geometry_from_ref(hit.face, Some(graph)).is_some())
                .then(|| {
                    candidate_from_hit_with_grid_and_supports(
                        graph,
                        hit,
                        placement_grid,
                        state.placement_bounds,
                    )
                })
                .map(|(mut candidate, supports)| {
                    candidate.spec = candidate
                        .spec
                        .with_material(material)
                        .with_appearance(appearance);
                    let smart_snap = state.smart_snap;
                    if smart_snap.enabled {
                        let bounds = state.placement_bounds;
                        let (snapped_candidate, active_guides) =
                            smart_snap_cuboid_candidate_with_supports(
                                &state.snap_index,
                                hit,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                &supports,
                                |guided| {
                                    validate_indexed_block_batch_in_bounds(
                                        &state.snap_index,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            );
                        state.smart_guides = active_guides;
                        candidate = snapped_candidate;
                    }
                    candidate
                })
            });
            let free_candidate = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                let mut candidate = free_cuboid_candidate(
                    point,
                    direction,
                    [1; 3],
                    GridRotation::default(),
                    placement_grid,
                    state.placement_bounds,
                );
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_free_cuboid_candidate(
                        &state.snap_index,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_indexed_block_batch_in_bounds(
                                &state.snap_index,
                                guided,
                                &[guided.spec],
                                bounds,
                            )
                            .is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            let placement_candidate = surface_candidate.or(free_candidate);
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    placement_candidate.is_none_or(|candidate| {
                        bearing_overlaps_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            let bearing_index = direct_bearing.or_else(|| {
                placement_candidate.and_then(|candidate| {
                    state.placed_bearings.iter().position(|bearing| {
                        bearing_overlaps_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            state.attachment_bearing = bearing_index;
            if let Some(bearing) =
                bearing_index.and_then(|index| state.placed_bearings.get(index).copied())
            {
                let mut candidate = placement_candidate.unwrap_or_else(|| {
                    let mut candidate =
                        bearing_attachment_candidate(graph, bearing.source, bearing.anchor);
                    candidate.spec = candidate
                        .spec
                        .with_material(material)
                        .with_appearance(appearance);
                    candidate
                });
                candidate.support = PlacementSupport::Bearing;
                let error = stage_bearing_attachment_in_bounds(
                    graph,
                    candidate,
                    bearing.source,
                    bearing.anchor,
                    bearing.dimensions,
                    state.placement_bounds,
                )
                .err();
                state.preview = Some(candidate);
                error
            } else {
                placement_candidate.and_then(|candidate| {
                    let error = validate_indexed_block_batch_in_bounds(
                        &state.snap_index,
                        candidate,
                        &[candidate.spec],
                        state.placement_bounds,
                    )
                    .err();
                    state.preview = Some(candidate);
                    error
                })
            }
        }
        (Tool::Weld, Some(PendingOperation::Weld(first))) => {
            state.hovered.and_then(|hit| {
                match stage_weld_objects(graph, first.owner, hit.face.owner) {
                    // A weld that closes a loop is allowed; if it also leaves a
                    // bearing with both sides in one body, say so before the
                    // click rather than leaving the player to wonder why
                    // nothing turns.
                    Ok(staged) => {
                        state.preview_warning = weld_lockup_warning(graph, &staged);
                        None
                    }
                    Err(error) => Some(error),
                }
            })
        }
        (Tool::Cylinder, _) => {
            let mut branch_preview = None;
            let surface_candidate = state.hovered.and_then(|hit| {
                let surface = cylinder_candidate_from_hit_with_grid(
                    graph,
                    hit,
                    cylinder_dimensions,
                    placement_grid,
                    state.placement_bounds,
                );
                let mut candidate = if let Ok(candidate) = surface {
                    candidate
                } else {
                    // The side of a pipe or junction branches off through a
                    // junction whose new arm faces the player; R turns it.
                    let FaceOwner::Part(part) = hit.face.owner else {
                        return None;
                    };
                    if state.pipe_branch_turn.0 != Some(part) {
                        state.pipe_branch_turn = (Some(part), 0);
                    }
                    let toward = state
                        .pointer_ray
                        .map_or(Vec3::Y, |(_, direction)| -direction);
                    let (candidate, branch) = crate::builder::pipe_branch_candidate(
                        graph,
                        hit,
                        cylinder_dimensions,
                        toward,
                        state.pipe_branch_turn.1,
                    )
                    .ok()?;
                    branch_preview = Some(branch);
                    candidate
                };
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled && branch_preview.is_none() {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_cylinder_candidate(
                        graph,
                        &state.snap_index,
                        hit,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_cylinder_candidate_in_bounds(graph, guided, bounds).is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            state.pipe_branch_preview = branch_preview;
            let free_candidate = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                let mut candidate = free_cylinder_candidate(
                    point,
                    direction,
                    cylinder_dimensions,
                    placement_grid,
                    state.placement_bounds,
                );
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_free_cylinder_candidate(
                        &state.snap_index,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_cylinder_candidate_in_bounds(graph, guided, bounds).is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            let placement_candidate = surface_candidate.or(free_candidate);
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    placement_candidate.is_none_or(|candidate| {
                        bearing_overlaps_cylinder_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            let bearing_index = direct_bearing.or_else(|| {
                placement_candidate.and_then(|candidate| {
                    state.placed_bearings.iter().position(|bearing| {
                        bearing_overlaps_cylinder_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            state.attachment_bearing = bearing_index;
            let candidate = if let Some(bearing) =
                bearing_index.and_then(|index| state.placed_bearings.get(index).copied())
            {
                placement_candidate
                    .or_else(|| {
                        let hit = SurfaceHit {
                            distance: 0.0,
                            point: bearing.anchor,
                            face: bearing.source,
                        };
                        cylinder_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            cylinder_dimensions,
                            placement_grid,
                            state.placement_bounds,
                        )
                        .ok()
                        .map(|mut candidate| {
                            candidate.spec = candidate
                                .spec
                                .with_material(material)
                                .with_appearance(appearance);
                            candidate
                        })
                    })
                    .map(|candidate| {
                        center_cylinder_candidate_on_bearing(candidate, bearing.anchor)
                    })
            } else {
                placement_candidate
            };
            candidate.and_then(|candidate| {
                let error =
                    validate_cylinder_candidate_in_bounds(graph, candidate, state.placement_bounds)
                        .err();
                state.cylinder_preview = Some(candidate);
                error
            })
        }
        (Tool::Layer, _) => {
            let (thickness, target) = match &state.layer_drag {
                Some(drag) => (drag.thickness, Some(Ok(drag.target.clone()))),
                None => (
                    state.next_layer_thickness(),
                    state
                        .hovered
                        .map(|hit| crate::builder::layer_target_from_hit(graph, hit)),
                ),
            };
            match target {
                None => None,
                Some(Err(error)) => Some(error),
                Some(Ok(target)) => {
                    match crate::builder::layered_parts(&target, thickness, material, appearance) {
                        Ok(layered) => {
                            let error = crate::builder::validate_layered_parts(
                                graph,
                                &target,
                                &layered,
                                state.placement_bounds,
                            )
                            .err();
                            state.layer_preview = Some(LayerPreview {
                                target,
                                layered,
                                material,
                                appearance,
                            });
                            error
                        }
                        Err(error) => Some(error),
                    }
                }
            }
        }
        (Tool::Transmission, _) => state.hovered.and_then(|hit| {
            match transmission_candidate_from_hit_in_bounds(graph, hit, state.placement_bounds) {
                Ok((_, candidate)) => {
                    state.preview = Some(candidate);
                    None
                }
                Err(error) => {
                    if try_face_geometry_from_ref(hit.face, Some(graph)).is_some() {
                        state.preview = Some(oriented_cuboid_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            TransmissionSpec::GRID_UNITS,
                            GridRotation::default(),
                            placement_grid,
                            state.placement_bounds,
                        ));
                    }
                    Some(error)
                }
            }
        }),
        (
            tool @ (Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink),
            _,
        ) => {
            let dimensions = match tool {
                Tool::Controller => ControllerSpec::GRID_UNITS,
                Tool::GasEngine => EngineKind::Gas.grid_units(),
                Tool::ElectricEngine => EngineKind::Electric.grid_units(),
                Tool::Servo => ServoSpec::GRID_UNITS,
                Tool::Seat => SeatSpec::GRID_UNITS,
                Tool::Input => InputSpec::GRID_UNITS,
                Tool::DimensionLink => DimensionLinkSpec::GRID_UNITS,
                _ => unreachable!(),
            };
            let rotation = authored_orientation(state.authored_orientation);
            let surface = state
                .hovered
                .filter(|hit| try_face_geometry_from_ref(hit.face, Some(graph)).is_some())
                .map(|hit| {
                    (
                        oriented_cuboid_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            dimensions,
                            rotation,
                            placement_grid,
                            state.placement_bounds,
                        ),
                        Some(hit),
                    )
                });
            let free = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                Some((
                    free_cuboid_candidate(
                        point,
                        direction,
                        dimensions,
                        rotation,
                        placement_grid,
                        state.placement_bounds,
                    ),
                    None,
                ))
            });
            surface.or(free).and_then(|(mut candidate, hit)| {
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = hit.map_or_else(
                        || {
                            smart_snap_free_cuboid_candidate(
                                &state.snap_index,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    validate_block_batch_in_bounds(
                                        graph,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            )
                        },
                        |hit| {
                            smart_snap_cuboid_candidate(
                                graph,
                                &state.snap_index,
                                hit,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    validate_block_batch_in_bounds(
                                        graph,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            )
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                let error = validate_block_batch_in_bounds(
                    graph,
                    candidate,
                    &[candidate.spec],
                    state.placement_bounds,
                )
                .err();
                state.preview = Some(candidate);
                error
            })
        }
        // Shaping edits the grid rather than placing anything, so like these
        // it has no placement ghost of its own.
        (
            Tool::Weld
            | Tool::Hammer
            | Tool::Connector
            | Tool::Shape
            | Tool::Chroma
            | Tool::Spring
            | Tool::Shock,
            _,
        ) => None,
        (Tool::LinearBearing, _) => {
            linear_editor::refresh(graph, state);
            state.preview_error.clone()
        }
        (Tool::Bearing, _) => state.hovered.and_then(|hit| {
            if try_face_geometry_from_ref(hit.face, Some(graph)).is_none() {
                Some(PlacementError::CurvedSurface)
            } else {
                match bearing_anchor_from_hit_with_grid(
                    graph,
                    hit,
                    placement_grid,
                    state.placement_bounds,
                ) {
                    Ok(mut anchor) => {
                        let smart_snap = state.smart_snap;
                        if smart_snap.enabled {
                            let normal_axis = PlacementPlane::from_normal(
                                face_geometry_from_ref(hit.face, Some(graph)).normal,
                            )
                            .normal_axis();
                            let (snapped_anchor, active_guides) = smart_snap_anchor(
                                &state.snap_index,
                                anchor,
                                normal_axis,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    bearing_support_face(
                                        graph,
                                        hit.face,
                                        guided,
                                        bearing_dimensions,
                                    )
                                    .is_some()
                                },
                            );
                            anchor = snapped_anchor;
                            state.smart_guides = active_guides;
                        }
                        state.bearing_preview_anchor = Some(anchor);
                        None
                    }
                    Err(error) => Some(error),
                }
            }
        }),
    };
}

#[cfg(test)]
pub(crate) fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState, tool: Tool) {
    state.snap_index.rebuild(graph);
    refresh_tool_preview_with_cylinder(
        graph,
        state,
        tool,
        CylinderDimensions::default(),
        BearingDimensions::default(),
        ConstructionMaterial::Steel,
        MaterialAppearance::BAKED,
    );
}
