//! Shape regions, edge features, and material layers: selection, dragging, and commits.

use crate::*;

/// The overlay's cyan, matching the `accent.speed` the panels use. Selected
/// shape corners take it so a selection reads by colour and not only by size.
pub(crate) const SHAPE_SELECTION_COLOR: Color = Color::srgb(0.247, 0.796, 0.878);

/// A drag that claims an area of existing blocks for the Shape tool, using the
/// same gesture the Block tool places with.
#[derive(Clone, Debug)]
pub(crate) struct RegionDrag {
    /// The block first clicked, which anchors the area.
    pub(crate) start: CuboidSpec,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    pub(crate) press: PointerSample,
    pub(crate) plane: PlacementPlane,
    /// Span at the last press or plane rotation.
    pub(crate) anchor_span: IVec3,
    /// Cells beyond the start block along each axis, signed.
    pub(crate) span: IVec3,
    pub(crate) last_span: Option<IVec3>,
    /// The area as it currently stands, whether or not it is claimable.
    pub(crate) region: ShapeRegion,
    /// Why the area cannot be claimed, if it cannot.
    pub(crate) error: Option<String>,
}

/// Thickness of the first layer before a drag has chosen one, in metres.
pub(crate) const DEFAULT_LAYER_THICKNESS_METERS: f32 = 0.25;

/// Surface the Layer tool points at and the parts it would become.
#[derive(Clone, Debug)]
pub(crate) struct LayerPreview {
    pub(crate) target: crate::builder::LayerTarget,
    pub(crate) layered: Vec<(PartId, PartSpec)>,
    pub(crate) material: ConstructionMaterial,
    pub(crate) appearance: MaterialAppearance,
}

#[derive(Component)]
pub(crate) struct ShapeNodeVisual;

#[derive(Component)]
pub(crate) struct ShapeSelectedVisual;

/// The plane a drag is currently sliding along.
#[derive(Component)]
pub(crate) struct ShapePlaneVisual;

/// The arrows naming that plane's two axes.
#[derive(Component)]
pub(crate) struct ShapeArrowVisual;

/// One of the Shape tool's three overlay batches, each excluding the others so
/// the three `Single` parameters can be held at once.
pub(crate) type ShapeOverlay<'w, 's, Own, First, Second> =
    Single<'w, 's, &'static mut Visibility, (With<Own>, Without<First>, Without<Second>)>;

/// Whether the Shape tool has something of its own for `Escape` to unwind.
pub(crate) fn shape_tool_is_busy(tool: Option<Tool>, state: &EditorState) -> bool {
    tool == Some(Tool::Shape)
        && (state.region_drag.is_some()
            || state.active_region.is_some()
            || state.vertex_drag.is_some()
            || !state.selected_vertices.is_empty())
}

/// Selecting an editable area, then hovering, dragging, and mirroring its cage.
///
/// Shaping edits a region rather than a part, so it runs beside the placement
/// tools rather than through them. A drag previews live by rebuilding the
/// construction mesh each frame and commits one batched command on release,
/// which keeps a whole symmetric edit to a single undo entry.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn handle_shape_actions(
    actions: Res<ButtonInput<GameAction>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    mut mirror: ResMut<shape_tool::ShapeMirror>,
    mut snap: ResMut<shape_tool::ShapeSnap>,
    mode: Res<shape_tool::ShapeEditMode>,
    _simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    camera_transform: Single<&GlobalTransform, With<MainCamera>>,
) {
    let mut view = live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    if selection.active_editor_tool() != Some(Tool::Shape) {
        if state.vertex_drag.take().is_some() {
            state.construction_mesh_dirty = true;
        }
        leave_region(state);
        leave_feature_shape(state);
        return;
    }
    if mode.is_changed() {
        if state.vertex_drag.take().is_some() || state.feature_drag.take().is_some() {
            state.construction_mesh_dirty = true;
        }
        state.hovered_vertex = None;
        state.edge_offer = None;
        state.paint_selecting = false;
        state.selected_vertices.clear();
        state.hovered_feature_edge = None;
        state.selected_feature_edges.clear();
        state.selected_shape_feature = None;
        if *mode != shape_tool::ShapeEditMode::Vertex {
            *snap = shape_tool::ShapeSnap::feature_default();
        }
        state.feedback = Some(format!("Shape mode: {} — {}", mode.label(), snap.label()));
    }
    // Regions can vanish under the tool when their blocks are deleted.
    if state
        .active_region
        .is_some_and(|id| graph.0.region(id).is_none())
    {
        leave_region(state);
    }
    if *mode != shape_tool::ShapeEditMode::Vertex {
        handle_feature_shape_actions(
            &actions,
            &keys,
            &mut graph.0,
            state,
            &mut history,
            *snap,
            *mode,
            *overlay,
            &player,
            &wheel,
        );
        return;
    }
    if state
        .feature_focus
        .is_some_and(|owner| matches!(owner, mechanic_core::SolidOwner::Part(_)))
        && state.active_region.is_none()
    {
        state.feedback = Some(
            "Vertex editing is unavailable for this solid; choose Chamfer or Fillet".to_owned(),
        );
        return;
    }
    let camera_transform = state.edit_context.map_or_else(
        || **camera_transform,
        |context| {
            let inverse = context.frame_to_world.inverse();
            let mut local = camera_transform.compute_transform();
            local.translation = inverse.point(local.translation);
            local.rotation = inverse.rotation() * local.rotation;
            GlobalTransform::from(local)
        },
    );
    if handle_shape_keyboard(
        &actions,
        &camera_transform,
        &mut graph.0,
        state,
        &mut history,
        &mut mirror,
        &mut snap,
    ) {
        return;
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        return;
    }
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        return;
    };
    let pointer_position = state.pointer_position;

    if actions.just_pressed(GameAction::Secondary) {
        if state.region_drag.take().is_some() {
            state.feedback = Some("Area selection cancelled".to_owned());
        } else if state.vertex_drag.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Shape drag cancelled".to_owned());
        } else if state.paint_selecting || !state.selected_vertices.is_empty() {
            state.paint_selecting = false;
            state.selected_vertices.clear();
            state.feedback = Some("Selection cleared".to_owned());
        } else if state.active_region.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Left the region".to_owned());
        }
        return;
    }

    // Without a region in hand the tool is a chooser: the same drag the Block
    // tool uses, claiming an area instead of filling one.
    let Some(region_id) = state.active_region else {
        choose_region(
            &actions,
            &mut graph.0,
            state,
            &mut history,
            pointer_position,
            ray_origin,
            ray_direction,
        );
        return;
    };
    let Some(region) = graph.0.region(region_id).cloned() else {
        return;
    };

    if let Some(drag) = state.vertex_drag.as_mut() {
        let offset = shape_tool::drag_offset(&region, drag, *snap, ray_origin, ray_direction);
        if offset != drag.offset {
            drag.offset = offset;
            state.construction_mesh_dirty = true;
        }
        if actions.just_released(GameAction::Primary) {
            let drag = state.vertex_drag.take().expect("a drag is in progress");
            if drag.offset == drag.start_offset {
                select_clicked_vertex(state, drag.index, shift_held(&actions));
            } else {
                commit_vertex_drag(
                    &mut graph.0,
                    state,
                    &mut history,
                    region_id,
                    &region,
                    &drag,
                    *mirror,
                );
            }
        }
        return;
    }

    state.hovered_vertex = shape_tool::hovered_vertex(&region, ray_origin, ray_direction);
    state.edge_offer = state
        .hovered_vertex
        .is_none()
        .then(|| shape_tool::edge_insertion(&region, ray_origin, ray_direction))
        .flatten();

    if state.paint_selecting {
        if actions.just_released(GameAction::Primary)
            || !actions.pressed(GameAction::Primary)
            || !shift_held(&actions)
        {
            state.paint_selecting = false;
            state.feedback = Some(format!(
                "Selected {} corners",
                state.selected_vertices.len()
            ));
        } else if let Some(index) = state.hovered_vertex
            && !state.selected_vertices.contains(&index)
        {
            state.selected_vertices.push(index);
        }
        return;
    }

    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    if shift_held(&actions) {
        state.paint_selecting = true;
        if let Some(index) = state.hovered_vertex
            && !state.selected_vertices.contains(&index)
        {
            state.selected_vertices.push(index);
        }
        return;
    }
    if let Some(index) = state.hovered_vertex {
        let drag = shape_tool::begin_group_drag(
            &region,
            index,
            &state.selected_vertices,
            ray_origin,
            ray_direction,
        );
        // Grabbing a vertex outside the selection abandons it, which is what
        // makes starting over cost nothing.
        if drag.group.is_empty() {
            state.selected_vertices.clear();
        }
        state.feedback = Some(format!(
            "Moving on {} axis — Rotate changes axis",
            drag.axis_label()
        ));
        state.vertex_drag = Some(drag);
    } else if let Some(offer) = state.edge_offer {
        subdivide_region(&mut graph.0, state, &mut history, region_id, offer);
    }
}

/// Drops everything that only makes sense while a region is being edited.
pub(crate) fn leave_region(state: &mut EditorState) {
    if state.active_region.take().is_some() {
        state.construction_mesh_dirty = true;
    }
    state.hovered_vertex = None;
    state.paint_selecting = false;
    state.edge_offer = None;
    state.region_drag = None;
    state.selected_vertices.clear();
}

pub(crate) fn leave_feature_shape(state: &mut EditorState) {
    state.feature_focus = None;
    state.hovered_feature_edge = None;
    state.selected_feature_edges.clear();
    state.feature_drag = None;
    state.selected_shape_feature = None;
    state.hovered_source_feature = None;
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn handle_feature_shape_actions(
    actions: &ButtonInput<GameAction>,
    keys: &ButtonInput<KeyCode>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    snap: shape_tool::ShapeSnap,
    mode: shape_tool::ShapeEditMode,
    overlay: ui::UiInput,
    player: &PlayerState,
    wheel: &MaterialWheelState,
) {
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        return;
    }
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        return;
    };

    if let Some(feature) = state.selected_shape_feature {
        if keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace) {
            let snapshot = EditorSnapshot::capture(graph, state);
            match graph.apply(BuildCommand::RemoveShapeFeature(feature)) {
                Ok(_) => {
                    history.commit(snapshot);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some("Removed feature".to_owned());
                }
                Err(error) => {
                    state.feedback = Some(format!("Cannot remove feature: {error}"));
                }
            }
            return;
        }
        let direction = if actions.just_pressed(GameAction::NudgeRight)
            || actions.just_pressed(GameAction::NudgeUp)
        {
            1_i64
        } else if actions.just_pressed(GameAction::NudgeLeft)
            || actions.just_pressed(GameAction::NudgeDown)
        {
            -1_i64
        } else {
            0
        };
        if direction != 0
            && let Some(existing) = graph.shape_feature(feature).cloned()
        {
            let amount = (i64::from(existing.amount_ticks) + direction * i64::from(snap.steps))
                .max(i64::from(snap.steps));
            let amount = u32::try_from(amount).unwrap_or(u32::MAX);
            let snapshot = EditorSnapshot::capture(graph, state);
            match graph.apply(BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: amount,
            }) {
                Ok(_) => {
                    history.commit(snapshot);
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(feature_amount_label(existing.treatment, amount));
                }
                Err(error) => state.feedback = Some(format!("Cannot adjust feature: {error}")),
            }
            return;
        }
    }

    if actions.just_pressed(GameAction::Secondary) {
        if state.feature_drag.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Feature drag cancelled".to_owned());
        } else if !state.selected_feature_edges.is_empty()
            || state.selected_shape_feature.take().is_some()
        {
            state.selected_feature_edges.clear();
            state.feedback = Some("Edge selection cleared".to_owned());
        } else if state.feature_focus.take().is_some() {
            state.feedback = Some("Left the solid".to_owned());
        }
        return;
    }

    if let Some(drag) = state.feature_drag.as_mut() {
        let proposed = drag.proposed_amount(snap, ray_origin, ray_direction);
        let clamped = if proposed == drag.amount_ticks {
            drag.amount_ticks
        } else {
            clamp_feature_amount(graph, drag, proposed, snap.steps.cast_unsigned())
        };
        if clamped != proposed {
            drag.discard_rejected_excess(clamped);
        }
        if clamped != drag.amount_ticks {
            drag.amount_ticks = clamped;
            state.construction_mesh_dirty = true;
            state.feedback = Some(feature_amount_label(drag.treatment, clamped));
        }
        if actions.just_released(GameAction::Primary) {
            let drag = state.feature_drag.take().expect("feature drag is active");
            if drag.amount_ticks == 0 {
                state.feedback = Some("Edges selected — drag inward to add a feature".to_owned());
                return;
            }
            let snapshot = EditorSnapshot::capture(graph, state);
            let command = if let Some(feature) = drag.feature {
                BuildCommand::SetShapeFeatureAmount {
                    feature,
                    amount_ticks: drag.amount_ticks,
                }
            } else {
                BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
                    drag.targets.clone(),
                    drag.treatment,
                    drag.amount_ticks,
                ))
            };
            match graph.apply(command) {
                Ok(BuildOutcome::ShapeFeatureAdded(feature)) => {
                    history.commit(snapshot);
                    state.feature_focus = graph
                        .shape_feature(feature)
                        .and_then(|feature| feature.targets.first())
                        .map(|target| target.owner);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(format!(
                        "Added {}",
                        feature_amount_label(drag.treatment, drag.amount_ticks)
                    ));
                }
                Ok(BuildOutcome::ShapeFeatureUpdated) => {
                    history.commit(snapshot);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(format!(
                        "Updated {}",
                        feature_amount_label(drag.treatment, drag.amount_ticks)
                    ));
                }
                Ok(_) => unreachable!("feature edits report a feature outcome"),
                Err(error) => state.feedback = Some(format!("Cannot apply feature: {error}")),
            }
        }
        return;
    }

    let pointed_owner = state.hovered.and_then(|hit| match hit.face.owner {
        FaceOwner::Part(part) => Some(graph.region_of(part).map_or(
            mechanic_core::SolidOwner::Part(part),
            mechanic_core::SolidOwner::Region,
        )),
        FaceOwner::Ground => None,
    });
    let focused_owner = pointed_owner.or(state.feature_focus);
    let evaluated_hit = focused_owner
        .and_then(|owner| {
            let solid = graph.evaluated_solid_shared(owner).ok()?;
            shape_tool::hovered_feature_edge(&solid, owner, ray_origin, ray_direction)
        })
        .or_else(|| {
            if focused_owner.is_none() {
                hovered_feature_edge_without_surface(graph, ray_origin, ray_direction)
            } else {
                None
            }
        });
    let owner = focused_owner.or_else(|| evaluated_hit.map(|hit| hit.target.owner));
    state.hovered_source_feature = None;
    let treatment = match mode {
        shape_tool::ShapeEditMode::Chamfer => mechanic_core::EdgeTreatment::Chamfer,
        shape_tool::ShapeEditMode::Fillet => mechanic_core::EdgeTreatment::Fillet,
        shape_tool::ShapeEditMode::Vertex => unreachable!(),
    };
    let virtual_hit = owner.and_then(|owner| {
        graph
            .shape_features()
            .filter(|(_, feature)| feature.treatment == treatment)
            .filter_map(|(feature_id, feature)| {
                let solid = graph.evaluated_solid_before(owner, feature_id).ok()?;
                feature
                    .targets
                    .iter()
                    .copied()
                    .filter(|target| target.owner == owner)
                    .filter_map(|target| {
                        shape_tool::hovered_source_edge(&solid, target, ray_origin, ray_direction)
                    })
                    .min_by(|left, right| left.distance.total_cmp(&right.distance))
                    .map(|hit| (feature_id, hit))
            })
            .min_by(|left, right| left.1.distance.total_cmp(&right.1.distance))
    });
    let (source_feature, hovered) = closer_feature_hit(evaluated_hit, virtual_hit);
    state.hovered_source_feature = source_feature;
    state.hovered_feature_edge = hovered;

    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    let Some(hit) = state.hovered_feature_edge else {
        if pointed_owner.is_some() {
            state.selected_feature_edges.clear();
            state.feature_focus = pointed_owner;
            state.feedback = Some("Aim at a highlighted logical edge".to_owned());
        } else {
            state.feedback = Some("Aim at a construction solid".to_owned());
        }
        return;
    };

    if let Some(feature_id) = state.hovered_source_feature {
        let Some(feature) = graph.shape_feature(feature_id).cloned() else {
            return;
        };
        state.selected_shape_feature = Some(feature_id);
        state.selected_feature_edges.clone_from(&feature.targets);
        state.feature_focus = Some(hit.target.owner);
        state.feature_drag = Some(shape_tool::FeatureDrag::begin(
            hit,
            feature.targets,
            feature.treatment,
            Some(feature_id),
            feature.amount_ticks,
            ray_origin,
            ray_direction,
        ));
        state.feedback = Some(format!(
            "Adjusting {} — drag or use arrows; Delete removes",
            feature_amount_label(feature.treatment, feature.amount_ticks)
        ));
        return;
    }

    if shift_held(actions) {
        if !state.selected_feature_edges.is_empty()
            && !shape_owners_connected(
                graph,
                state.selected_feature_edges[0].owner,
                hit.target.owner,
            )
        {
            state.selected_feature_edges.clear();
        }
        let chain = tangent_feature_chain(graph, hit.target);
        if chain
            .iter()
            .all(|target| state.selected_feature_edges.contains(target))
        {
            state
                .selected_feature_edges
                .retain(|target| !chain.contains(target));
            state.feedback = Some(format!(
                "Selected {} edge chain(s)",
                state.selected_feature_edges.len()
            ));
            return;
        }
        for target in chain {
            if !state.selected_feature_edges.contains(&target) {
                state.selected_feature_edges.push(target);
            }
        }
    } else if !state.selected_feature_edges.contains(&hit.target) {
        state.selected_feature_edges = tangent_feature_chain(graph, hit.target);
    }
    state.feature_focus = Some(hit.target.owner);
    state.selected_shape_feature = None;
    state.feature_drag = Some(shape_tool::FeatureDrag::begin(
        hit,
        state.selected_feature_edges.clone(),
        treatment,
        None,
        0,
        ray_origin,
        ray_direction,
    ));
    state.feedback = Some(format!(
        "Selected {} edge chain(s) — drag inward",
        state.selected_feature_edges.len()
    ));
}

pub(crate) fn clamp_feature_amount(
    graph: &ConstructionGraph,
    drag: &mut shape_tool::FeatureDrag,
    proposed: u32,
    increment: u32,
) -> u32 {
    let mut amount = proposed;
    while amount > 0 {
        let mut preview = graph.clone();
        let command = if let Some(feature) = drag.feature {
            BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: amount,
            }
        } else {
            BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
                drag.targets.clone(),
                drag.treatment,
                amount,
            ))
        };
        if preview.apply(command).is_ok() {
            drag.validated_preview = Some(shape_tool::ValidatedFeaturePreview {
                source: graph.clone(),
                graph: preview,
                key: (drag.feature, drag.targets.clone(), drag.treatment, amount),
            });
            return amount;
        }
        amount = amount.saturating_sub(increment.max(1));
    }
    0
}

pub(crate) fn hovered_feature_edge_without_surface(
    graph: &ConstructionGraph,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<shape_tool::FeatureEdgeHit> {
    let mut owners = graph
        .parts()
        .filter_map(|(part, spec)| {
            ordinary_material(*spec)?;
            Some(graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            ))
        })
        .collect::<Vec<_>>();
    owners.sort_unstable();
    owners.dedup();

    owners
        .into_iter()
        .filter_map(|owner| {
            let (minimum, maximum) = match owner {
                mechanic_core::SolidOwner::Part(part) => {
                    builder::part_world_bounds(*graph.part(part)?)
                }
                mechanic_core::SolidOwner::Region(region) => {
                    region_world_bounds(graph.region(region)?)
                }
            };
            shape_tool::inflated_aabb_ray_distance(minimum, maximum, ray_origin, ray_direction)?;
            let solid = graph.evaluated_solid_shared(owner).ok()?;
            shape_tool::hovered_feature_edge(&solid, owner, ray_origin, ray_direction)
        })
        .min_by(|left, right| {
            let left_along = (left.point - ray_origin).dot(ray_direction);
            let right_along = (right.point - ray_origin).dot(ray_direction);
            left_along
                .total_cmp(&right_along)
                .then_with(|| left.distance.total_cmp(&right.distance))
        })
}

pub(crate) fn closer_feature_hit(
    evaluated: Option<shape_tool::FeatureEdgeHit>,
    source: Option<(mechanic_core::ShapeFeatureId, shape_tool::FeatureEdgeHit)>,
) -> (
    Option<mechanic_core::ShapeFeatureId>,
    Option<shape_tool::FeatureEdgeHit>,
) {
    match (evaluated, source) {
        (Some(real), Some((feature, dashed))) if dashed.distance < real.distance => {
            (Some(feature), Some(dashed))
        }
        (Some(real), _) => (None, Some(real)),
        (None, Some((feature, dashed))) => (Some(feature), Some(dashed)),
        (None, None) => (None, None),
    }
}

pub(crate) fn feature_amount_label(
    treatment: mechanic_core::EdgeTreatment,
    amount_ticks: u32,
) -> String {
    let metres = f64::from(amount_ticks) * f64::from(POSITION_TICK_METERS);
    let name = match treatment {
        mechanic_core::EdgeTreatment::Chamfer => "setback",
        mechanic_core::EdgeTreatment::Fillet => "radius",
    };
    if metres < 0.1 {
        format!("{name}: {:.1} mm", metres * 1000.0)
    } else {
        format!("{name}: {metres:.3} m")
    }
}

pub(crate) fn shape_owners_connected(
    graph: &ConstructionGraph,
    first: mechanic_core::SolidOwner,
    second: mechanic_core::SolidOwner,
) -> bool {
    weld_connected_shape_owners(graph, first).contains(&second)
}

/// Finds the complete weld component with one indexed traversal.
///
/// Edge selection used to repeat a full weld scan for every construction
/// owner. Dense creations made that cubic in practice: selecting one edge in
/// the 504-part fillet test creation performed hundreds of millions of weld
/// checks before tangent matching even began.
pub(crate) fn weld_connected_shape_owners(
    graph: &ConstructionGraph,
    initial: mechanic_core::SolidOwner,
) -> HashSet<mechanic_core::SolidOwner> {
    let starts = match initial {
        mechanic_core::SolidOwner::Part(part) => vec![part],
        mechanic_core::SolidOwner::Region(region) => graph
            .parts()
            .filter_map(|(part, _)| (graph.region_of(part) == Some(region)).then_some(part))
            .collect(),
    };
    if starts.is_empty() {
        return HashSet::new();
    }

    let mut neighbours = HashMap::<PartId, Vec<PartId>>::new();
    for (_, weld) in graph.welds() {
        let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        else {
            continue;
        };
        neighbours.entry(first).or_default().push(second);
        neighbours.entry(second).or_default().push(first);
    }

    let mut reached = starts.iter().copied().collect::<HashSet<_>>();
    let mut pending = starts;
    while let Some(part) = pending.pop() {
        if let Some(adjacent) = neighbours.get(&part) {
            for &next in adjacent {
                if reached.insert(next) {
                    pending.push(next);
                }
            }
        }
    }
    reached
        .into_iter()
        .map(|part| {
            graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            )
        })
        .collect()
}

pub(crate) fn tangent_feature_chain(
    graph: &ConstructionGraph,
    initial: mechanic_core::EdgeChainRef,
) -> Vec<mechanic_core::EdgeChainRef> {
    let mut candidates = Vec::<(mechanic_core::EdgeChainRef, Vec<(Vec3, Vec3)>)>::new();
    let connected = weld_connected_shape_owners(graph, initial.owner);
    let mut owners = graph
        .parts()
        .filter_map(|(part, spec)| {
            ordinary_material(*spec)?;
            let owner = graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            );
            connected.contains(&owner).then_some(owner)
        })
        .collect::<Vec<_>>();
    owners.sort_unstable();
    owners.dedup();
    for owner in owners {
        let Ok(solid) = graph.evaluated_solid_shared(owner) else {
            continue;
        };
        for logical in &solid.logical_edges {
            if !logical.convex {
                continue;
            }
            let target = mechanic_core::EdgeChainRef {
                owner,
                edge: logical.key,
            };
            let endpoints = logical_chain_endpoints(&solid, logical);
            candidates.push((target, endpoints));
        }
    }
    let mut selected = vec![initial];
    loop {
        let mut additions = Vec::new();
        for selected_target in selected.clone() {
            let Some((_, endpoints)) = candidates
                .iter()
                .find(|(target, _)| *target == selected_target)
            else {
                continue;
            };
            for &(point, tangent) in endpoints {
                let matches = candidates
                    .iter()
                    .filter(|(target, _)| !selected.contains(target))
                    .filter(|(_, candidate_endpoints)| {
                        candidate_endpoints
                            .iter()
                            .any(|(candidate, candidate_tangent)| {
                                point.distance(*candidate) <= mechanic_core::ANCHOR_TOLERANCE_METERS
                                    && tangent.dot(*candidate_tangent).abs() >= 1.0 - 1.0e-4
                            })
                    })
                    .map(|(target, _)| *target)
                    .collect::<Vec<_>>();
                if matches.len() == 1 && !additions.contains(&matches[0]) {
                    additions.push(matches[0]);
                }
            }
        }
        if additions.is_empty() {
            break;
        }
        selected.extend(additions);
    }
    selected
}

pub(crate) fn logical_chain_endpoints(
    solid: &mechanic_core::EvaluatedSolid,
    logical: &mechanic_core::LogicalEdge,
) -> Vec<(Vec3, Vec3)> {
    let mut occurrences = Vec::<(Vec3, Vec3)>::new();
    for &edge_index in &logical.half_edges {
        let edge = solid.half_edges[edge_index as usize];
        let next = solid.half_edges[edge.next as usize];
        let start = solid.vertices[edge.origin as usize].position;
        let end = solid.vertices[next.origin as usize].position;
        let tangent = (end - start).normalize_or_zero();
        occurrences.push((start, tangent));
        occurrences.push((end, tangent));
    }
    occurrences
        .iter()
        .copied()
        .filter(|(point, _)| {
            occurrences
                .iter()
                .filter(|(candidate, _)| {
                    point.distance(*candidate) <= mechanic_core::ANCHOR_TOLERANCE_METERS
                })
                .count()
                == 1
        })
        .collect()
}

/// Presses on a part surface and drags a new material layer out along its
/// normal: out thickens, back thins, in the active placement-grid step.
pub(crate) fn handle_layer_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let grid = active_placement_grid(actions);
    if let Some(mut drag) = state.layer_drag.take() {
        if let Some((origin, direction)) = state.pointer_ray {
            let previous = drag.thickness;
            drag.update(grid, origin, direction);
            if drag.thickness > previous
                && checked_layer(graph, &drag, drag.thickness, state.placement_bounds).is_err()
                && checked_layer(graph, &drag, previous, state.placement_bounds).is_ok()
            {
                drag.discard_rejected_excess(previous);
            }
        }
        state.feedback = Some(
            match checked_layer(graph, &drag, drag.thickness, state.placement_bounds) {
                Ok(layered) => layer_feedback(&drag, &layered, grid),
                Err(error) => error.to_string(),
            },
        );
        if !actions.just_released(GameAction::Primary) {
            state.layer_drag = Some(drag);
            return;
        }
        let previous = EditorSnapshot::capture(graph, state);
        match crate::builder::stage_layer(
            graph,
            &drag.target,
            drag.thickness,
            drag.material,
            drag.appearance,
            state.placement_bounds,
        ) {
            Ok((staged, layered)) => {
                *graph = staged;
                history.commit(previous);
                state.layer_thickness = drag.thickness;
                state.construction_mesh_dirty = true;
                clear_hover(state);
                state.feedback = Some(layer_feedback(&drag, &layered, grid));
            }
            Err(error) => state.feedback = Some(error.to_string()),
        }
        return;
    }
    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    let Some(preview) = state.layer_preview.clone() else {
        state.feedback = Some(state.preview_error.as_ref().map_or_else(
            || "Point at a flat face, or a full cylinder's wall or bore".to_owned(),
            ToString::to_string,
        ));
        return;
    };
    let Some((origin, direction)) = state.pointer_ray else {
        state.feedback = Some("Pointer ray is unavailable".to_owned());
        return;
    };
    state.layer_drag = Some(crate::builder::LayerDrag::begin(
        preview.target,
        preview.material,
        preview.appearance,
        state.next_layer_thickness(),
        origin,
        direction,
    ));
}

/// The dragged parts with a layer `thickness` thick, if they fit.
pub(crate) fn checked_layer(
    graph: &ConstructionGraph,
    drag: &crate::builder::LayerDrag,
    thickness: f32,
    bounds: crate::builder::PlacementBounds,
) -> Result<Vec<(PartId, PartSpec)>, crate::builder::PlacementError> {
    let layered =
        crate::builder::layered_parts(&drag.target, thickness, drag.material, drag.appearance)?;
    crate::builder::validate_layered_parts(graph, &drag.target, &layered, bounds).map(|()| layered)
}

pub(crate) fn layer_feedback(
    drag: &crate::builder::LayerDrag,
    layered: &[(PartId, PartSpec)],
    grid: crate::builder::PlacementGrid,
) -> String {
    let Some(&(_, spec)) = layered.first() else {
        return String::new();
    };
    let centimetres = drag.thickness * 100.0;
    let material = drag.material.label();
    let step = if layered.len() > 1 {
        format!("{} blocks, {} steps", layered.len(), grid.label())
    } else {
        format!("{} steps", grid.label())
    };
    match (drag.target.face, spec) {
        (mechanic_core::LayerFace::OuterWall, PartSpec::Cylinder(cylinder)) => format!(
            "{centimetres:.0} cm {material} layer → outer diameter {:.2} m ({step})",
            cylinder.dimensions.outer_diameter()
        ),
        (mechanic_core::LayerFace::Bore, PartSpec::Cylinder(cylinder)) => format!(
            "{centimetres:.0} cm {material} bore layer → inner diameter {:.2} m ({step})",
            cylinder.dimensions.inner_diameter()
        ),
        (mechanic_core::LayerFace::Face(face), _) => {
            use mechanic_core::FaceKind;
            let (axis, label) = match face {
                FaceKind::PositiveX => (0, "+X"),
                FaceKind::NegativeX => (0, "-X"),
                FaceKind::PositiveY => (1, "+Y"),
                FaceKind::NegativeY => (1, "-Y"),
                FaceKind::PositiveZ => (2, "+Z"),
                FaceKind::NegativeZ => (2, "-Z"),
            };
            let depth = match spec {
                PartSpec::Cylinder(cylinder) => cylinder.dimensions.axial_length(),
                _ => spec
                    .as_cuboid()
                    .map_or(0.0, |cuboid| cuboid.size_meters()[axis]),
            };
            format!(
                "{centimetres:.0} cm {material} layer on {label} face → {depth:.2} m deep ({step})"
            )
        }
        _ => format!("{centimetres:.0} cm {material} layer ({step})"),
    }
}

/// The area a drag covers: the block it started on, grown by `span` cells.
pub(crate) fn region_area(start: CuboidSpec, span: IVec3) -> ShapeRegion {
    let cells = part_cells(start);
    ShapeRegion::from_origin_steps(
        cells.corner_steps(IVec3::ZERO, 0) + span.min(IVec3::ZERO) * POSITION_TICKS_PER_GRID_UNIT,
        cells.counts() + span.abs(),
        start.material,
    )
    .expect("a drag area is at least the block it started on")
}

/// Drags an area of blocks out and claims it as an editable region, using the
/// same gesture the Block tool places with — Rotate included.
pub(crate) fn choose_region(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    cursor: Option<Vec2>,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    if state.region_drag.is_some() {
        if let Some(cursor) = cursor {
            refresh_region_drag(graph, state, cursor, ray_origin, ray_direction);
        }
        if actions.just_released(GameAction::Primary) {
            commit_region_drag(graph, state, history);
        }
        return;
    }
    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    let Some(hit) = state.hovered else {
        state.feedback = Some("Aim at a block to choose an area".to_owned());
        return;
    };
    let FaceOwner::Part(part) = hit.face.owner else {
        state.feedback = Some("The ground cannot be shaped".to_owned());
        return;
    };
    // Clicking a block already inside a region reopens it rather than refusing.
    if let Some(existing) = graph.region_of(part) {
        state.active_region = Some(existing);
        state.construction_mesh_dirty = true;
        state.feedback = Some("Editing region — drag its corners".to_owned());
        return;
    }
    let Some(start) = graph.part(part).and_then(|spec| spec.as_cuboid()) else {
        state.feedback = Some("Only blocks can be shaped".to_owned());
        return;
    };
    let Some(cursor) = cursor else {
        state.feedback = Some("Pointer position is unavailable".to_owned());
        return;
    };
    let plane =
        PlacementPlane::from_normal(builder::face_geometry_from_ref(hit.face, Some(graph)).normal);
    let region = region_area(start, IVec3::ZERO);
    let error = graph
        .check_region_area(&region)
        .err()
        .map(|error| error.to_string());
    state.region_drag = Some(RegionDrag {
        start,
        press: PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        },
        plane,
        anchor_span: IVec3::ZERO,
        span: IVec3::ZERO,
        last_span: Some(IVec3::ZERO),
        region,
        error,
    });
    state.feedback = Some(format!(
        "Choosing an area on {} plane — release to shape it, Rotate changes plane",
        plane.label()
    ));
}

/// Re-measures the dragged area against the pointer and re-checks the rules.
pub(crate) fn refresh_region_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, press, plane, anchor_span, last_span) = {
        let drag = state
            .region_drag
            .as_ref()
            .expect("a region drag was checked by the caller");
        (
            drag.start,
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        // A plane the pointer cannot reach leaves the last good area standing
        // rather than collapsing the drag.
        let Some(span) = builder::block_span_from_rays(
            start,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            return;
        };
        span
    } else {
        anchor_span
    };
    if last_span == Some(span) {
        return;
    }
    let region = region_area(start, span);
    let cells = region.size_cells().element_product();
    let error = if cells > i32::try_from(builder::MAX_DRAG_BLOCKS).expect("the cap fits in i32") {
        Some(format!(
            "an area is limited to {} blocks",
            builder::MAX_DRAG_BLOCKS
        ))
    } else {
        graph
            .check_region_area(&region)
            .err()
            .map(|error| error.to_string())
    };
    let drag = state
        .region_drag
        .as_mut()
        .expect("the region drag stays open while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
    drag.region = region;
    drag.error = error;
}

/// Claims the dragged area, if it broke no rule.
pub(crate) fn commit_region_drag(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let drag = state
        .region_drag
        .take()
        .expect("a region drag was checked by the caller");
    if let Some(error) = drag.error {
        state.feedback = Some(format!("Cannot shape: {error}"));
        return;
    }
    let cells = drag.region.size_cells();
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::AddRegion(drag.region)) {
        Ok(BuildOutcome::RegionAdded(id)) => {
            history.commit(snapshot);
            state.active_region = Some(id);
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Editing {}x{}x{} region — drag its corners",
                cells.x, cells.y, cells.z
            ));
        }
        Ok(_) => unreachable!("adding a region reports the region it added"),
        Err(error) => state.feedback = Some(format!("Cannot shape: {error}")),
    }
}

/// Inserts a cage plane where the pointer offered one.
pub(crate) fn subdivide_region(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    region: RegionId,
    offer: shape_tool::EdgeInsertion,
) {
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::SubdivideRegion {
        region,
        axis: offer.axis,
        position: offer.position,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.construction_mesh_dirty = true;
            state.edge_offer = None;
            state.feedback = Some("Added a cage vertex".to_owned());
        }
        Err(error) => state.feedback = Some(format!("Cannot subdivide: {error}")),
    }
}

/// The Shape tool's keyboard: mirror planes, the step size, and nudging.
///
/// Returns whether it consumed the frame, which a nudge does so the pointer
/// does not also act on the same input.
pub(crate) fn handle_shape_keyboard(
    actions: &ButtonInput<GameAction>,
    camera_transform: &GlobalTransform,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    mirror: &mut shape_tool::ShapeMirror,
    snap: &mut shape_tool::ShapeSnap,
) -> bool {
    if actions.just_pressed(GameAction::ShapeMirrorX) {
        mirror.x = !mirror.x;
        state.feedback = Some(mirror.label());
    }
    if actions.just_pressed(GameAction::ShapeMirrorZ) {
        mirror.z = !mirror.z;
        state.feedback = Some(mirror.label());
    }
    if actions.just_pressed(GameAction::ShapeSnap) {
        snap.cycle();
        state.feedback = Some(snap.label());
    }
    let Some((axis, direction)) = nudge_request(actions, camera_transform) else {
        return false;
    };
    nudge_selection(graph, state, history, axis, direction, *snap, *mirror);
    true
}

/// Which way the arrow keys are asking the selection to move.
///
/// The keys read as screen directions and resolve to whichever world axis lies
/// nearest, so a nudge goes where it looks like it should while still landing
/// on the grid. Depth is deliberately absent: orbiting the camera is how the
/// third axis is reached.
pub(crate) fn nudge_request(
    actions: &ButtonInput<GameAction>,
    camera_transform: &GlobalTransform,
) -> Option<(usize, i32)> {
    let (right, up) = (
        camera_transform.right().as_vec3(),
        camera_transform.up().as_vec3(),
    );
    let (basis, sign) = if actions.just_pressed(GameAction::NudgeRight) {
        (right, 1)
    } else if actions.just_pressed(GameAction::NudgeLeft) {
        (right, -1)
    } else if actions.just_pressed(GameAction::NudgeUp) {
        (up, 1)
    } else if actions.just_pressed(GameAction::NudgeDown) {
        (up, -1)
    } else {
        return None;
    };
    let (axis, axis_sign) = shape_tool::screen_axis(basis);
    Some((axis, axis_sign * sign))
}

/// Moves every selected cage vertex one increment, as one undo entry.
pub(crate) fn nudge_selection(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    axis: usize,
    direction: i32,
    snap: shape_tool::ShapeSnap,
    mirror: shape_tool::ShapeMirror,
) {
    let Some(region_id) = state.active_region else {
        return;
    };
    if state.selected_vertices.is_empty() {
        state.feedback = Some("Select a corner first — click one or drag a box".to_owned());
        return;
    }
    let Some(region) = graph.region(region_id).cloned() else {
        return;
    };
    let edits = shape_tool::nudge_edits(
        &region,
        &state.selected_vertices,
        axis,
        direction,
        snap,
        mirror,
    );
    if edits.is_empty() {
        state.feedback = Some("Corner is already as far as it goes".to_owned());
        return;
    }
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::SetRegionVertices {
        region: region_id,
        vertices: edits,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Nudged {} corner(s) — {}",
                state.selected_vertices.len(),
                snap.label()
            ));
        }
        Err(error) => state.feedback = Some(format!("Cannot shape: {error}")),
    }
}

/// A click on a vertex picks it for the keyboard; holding shift builds a set up
/// one corner at a time.
pub(crate) fn select_clicked_vertex(state: &mut EditorState, index: CageIndex, extend: bool) {
    if extend {
        if let Some(at) = state
            .selected_vertices
            .iter()
            .position(|&other| other == index)
        {
            state.selected_vertices.remove(at);
        } else {
            state.selected_vertices.push(index);
        }
    } else {
        state.selected_vertices = vec![index];
    }
    state.feedback = Some(match state.selected_vertices.len() {
        0 => "Selection cleared".to_owned(),
        1 => "Corner selected — arrows nudge it".to_owned(),
        count => format!("Selected {count} corners"),
    });
}

pub(crate) fn shift_held(actions: &ButtonInput<GameAction>) -> bool {
    actions.pressed(GameAction::SelectionModifier)
}

/// Commits a finished drag, expanded across the active mirror planes.
pub(crate) fn commit_vertex_drag(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    region_id: RegionId,
    region: &ShapeRegion,
    drag: &shape_tool::VertexDrag,
    mirror: shape_tool::ShapeMirror,
) {
    state.construction_mesh_dirty = true;
    if drag.offset == drag.start_offset {
        return;
    }
    let snapshot = EditorSnapshot::capture(graph, state);
    let edits = shape_tool::drag_edits(region, drag, mirror);
    let count = edits.len();
    match graph.apply(BuildCommand::SetRegionVertices {
        region: region_id,
        vertices: edits,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.feedback = Some(if count > 1 {
                format!("Shaped {count} corners")
            } else {
                "Shaped a corner".to_owned()
            });
        }
        Err(error) => {
            state.feedback = Some(format!("Cannot shape: {error}"));
        }
    }
}

/// Fades everything outside the region being edited.
///
/// With a region in hand the rest of the build drops back to a ghost so the
/// area under the cursor is the only thing reading as solid. Leaving the region
/// puts every material back.
pub(crate) fn sync_region_focus(
    state: Res<EditorState>,
    mode: Res<shape_tool::ShapeEditMode>,
    selection: Res<SelectedTool>,
    visuals: Res<EditorVisuals>,
    mut construction_visuals: Query<(
        &ConstructionVisual,
        &mut MeshMaterial3d<ConstructionRenderMaterial>,
    )>,
) {
    let editing = region_focus_is_active(
        selection.active_editor_tool(),
        *mode,
        state.active_region.is_some(),
    );
    for (visual, mut material) in &mut construction_visuals {
        let index = material_index(visual.0);
        let wanted = if editing {
            &visuals.ghost_materials[index]
        } else {
            &visuals.construction_materials[index]
        };
        if material.0.id() != wanted.id() {
            material.0 = wanted.clone();
        }
    }
}

pub(crate) fn region_focus_is_active(
    tool: Option<Tool>,
    mode: shape_tool::ShapeEditMode,
    has_active_region: bool,
) -> bool {
    tool == Some(Tool::Shape)
        && matches!(mode, shape_tool::ShapeEditMode::Vertex)
        && has_active_region
}

/// Draws the active region's cage: its vertices, the edges between them, and
/// the new vertex the pointer is being offered.
///
/// Only vertices near the pointer appear, so choosing the tool does not bury the
/// build in handles.
#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
pub(crate) fn sync_shape_nodes(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    mirror: Res<shape_tool::ShapeMirror>,
    mode: Res<shape_tool::ShapeEditMode>,
    selection: Res<SelectedTool>,
    _simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut markers: ShapeOverlay<ShapeNodeVisual, ShapeSelectedVisual, ShapePlaneVisual>,
    mut selected_markers: ShapeOverlay<ShapeSelectedVisual, ShapeNodeVisual, ShapePlaneVisual>,
) {
    let local_graph = state
        .edit_context
        .and_then(|context| graph.0.in_edit_frame(context.frame).ok())
        .map(EditorGraph);
    let graph = local_graph.as_ref().unwrap_or(&graph);
    let hide = selection.active_editor_tool() != Some(Tool::Shape);

    // Two batches rather than one: a selected corner reads by colour as well as
    // by size, which size alone was not carrying.
    let mut plain = OverlayGeometry::default();
    let mut chosen = OverlayGeometry::default();

    if !hide && *mode != shape_tool::ShapeEditMode::Vertex {
        let owner = state
            .hovered_feature_edge
            .map(|hit| hit.target.owner)
            .or(state.feature_focus);
        if let Some(owner) = owner
            && let Ok(solid) = graph.0.evaluated_solid_shared(owner)
        {
            for logical in &solid.logical_edges {
                if !logical.convex {
                    continue;
                }
                let target = mechanic_core::EdgeChainRef {
                    owner,
                    edge: logical.key,
                };
                let selected = state.selected_feature_edges.contains(&target);
                let hovered = state
                    .hovered_feature_edge
                    .is_some_and(|hit| hit.target == target);
                let geometry = if selected { &mut chosen } else { &mut plain };
                let thickness = if hovered {
                    0.022
                } else if selected {
                    0.017
                } else {
                    0.010
                };
                for &edge_index in &logical.half_edges {
                    let edge = solid.half_edges[edge_index as usize];
                    let next = solid.half_edges[edge.next as usize];
                    let start = solid.vertices[edge.origin as usize].position;
                    let end = solid.vertices[next.origin as usize].position;
                    append_overlay_bar(start, end, thickness, geometry);
                }
            }
            let treatment = match *mode {
                shape_tool::ShapeEditMode::Chamfer => mechanic_core::EdgeTreatment::Chamfer,
                shape_tool::ShapeEditMode::Fillet => mechanic_core::EdgeTreatment::Fillet,
                shape_tool::ShapeEditMode::Vertex => unreachable!(),
            };
            for (feature_id, feature) in graph
                .0
                .shape_features()
                .filter(|(_, feature)| feature.treatment == treatment)
            {
                let Ok(source) = graph.0.evaluated_solid_before(owner, feature_id) else {
                    continue;
                };
                let selected = state.selected_shape_feature == Some(feature_id);
                let geometry = if selected { &mut chosen } else { &mut plain };
                for target in feature
                    .targets
                    .iter()
                    .filter(|target| target.owner == owner)
                {
                    let Some(logical) = source.logical_edge(target.edge) else {
                        continue;
                    };
                    for &edge_index in &logical.half_edges {
                        let edge = source.half_edges[edge_index as usize];
                        let next = source.half_edges[edge.next as usize];
                        append_dashed_overlay_bar(
                            source.vertices[edge.origin as usize].position,
                            source.vertices[next.origin as usize].position,
                            if selected { 0.014 } else { 0.008 },
                            geometry,
                        );
                    }
                }
            }
        }
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    }

    // The area being dragged out, cyan while it is claimable and plain while a
    // rule refuses it, so the outline itself carries the verdict.
    if !hide && let Some(drag) = state.region_drag.as_ref() {
        let target = if drag.error.is_some() {
            &mut plain
        } else {
            &mut chosen
        };
        append_region_outline(&drag.region, target);
    }

    let Some((_, region)) = (if hide {
        None
    } else {
        preview_region(&graph.0, &state, *mirror)
    }) else {
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    };
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    };
    let dragged = state.vertex_drag.as_ref().map(|drag| drag.index);
    for (index, position, distance) in
        shape_tool::revealed_vertices(&region, ray_origin, ray_direction)
    {
        let selected = state.selected_vertices.contains(&index);
        let mut size = shape_tool::vertex_marker_size(distance);
        if selected {
            size *= 1.5;
        }
        if state.hovered_vertex == Some(index) || dragged == Some(index) {
            size *= 1.8;
        }
        let target = if selected { &mut chosen } else { &mut plain };
        append_transformed_cuboid(
            position,
            Quat::IDENTITY,
            Vec3::splat(size),
            &mut target.positions,
            &mut target.normals,
            &mut target.indices,
        );
    }
    // The vertex the pointer is being offered on an edge, shown in the same
    // cyan as a selection because taking it is what it becomes.
    if let Some(offer) = state.edge_offer {
        append_transformed_cuboid(
            offer.at,
            Quat::IDENTITY,
            Vec3::splat(0.024),
            &mut chosen.positions,
            &mut chosen.normals,
            &mut chosen.indices,
        );
    }
    **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
    **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
}

/// Draws the guide for an open drag: a placement plane or one vertex axis.
///
/// Shared by placing blocks and choosing a shape area, because both measure the
/// pointer against a plane and both rotate it with Rotate.
pub(crate) fn sync_drag_plane(
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut plane_marker: ShapeOverlay<ShapePlaneVisual, ShapeArrowVisual, ShapeSelectedVisual>,
    mut arrow_marker: ShapeOverlay<ShapeArrowVisual, ShapePlaneVisual, ShapeSelectedVisual>,
) {
    let mut sheet = OverlayGeometry::default();
    let mut arrows = OverlayGeometry::default();
    if let Some(hit) = state.hovered_feature_edge {
        append_feature_pull_arrow(hit.point, hit.bisector, &mut arrows);
    } else if let Some(drag) = state.pipe_drag.as_ref() {
        let direction = *drag
            .directions
            .last()
            .expect("a pipe run has one direction");
        let absolute = direction.abs();
        let incoming_axis = if absolute.x >= absolute.y && absolute.x >= absolute.z {
            0
        } else if absolute.y >= absolute.z {
            1
        } else {
            2
        };
        let arrow_origin = if drag.bearing_offset.is_some() {
            drag.start
        } else {
            drag.endpoint
        };
        if drag.bearing_offset.is_some()
            || drag.choosing_direction
            || drag.mode != PipeEditMode::Length
        {
            for axis in 0..3 {
                if axis != incoming_axis {
                    append_axis_arrows(arrow_origin, axis, &mut arrows);
                }
            }
        } else {
            append_axis_arrows(arrow_origin, incoming_axis, &mut arrows);
        }
    } else if let Some((low, high, plane)) = active_drag_plane(&state, &simulation) {
        append_drag_plane(low, high, plane, &mut sheet);
        append_plane_arrows(low, high, plane, &mut arrows);
    } else if let Some(drag) = state.vertex_drag.as_ref() {
        append_axis_arrows(drag.position(), drag.axis, &mut arrows);
    }
    **plane_marker = write_overlay(&mut meshes, &visuals.shape_plane_mesh, sheet);
    **arrow_marker = write_overlay(&mut meshes, &visuals.shape_arrow_mesh, arrows);
}

/// Draws the increasing-amount direction for an edge treatment. Most of the
/// shaft stays outside the solid while the head lands at the edge, so the
/// inward 45-degree direction remains visible instead of disappearing behind
/// the preview mesh.
pub(crate) fn append_feature_pull_arrow(at: Vec3, direction: Vec3, geometry: &mut OverlayGeometry) {
    const OUTSIDE_REACH: f32 = 0.16;
    const TIP_INSET: f32 = 0.015;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;

    let Some(along) = direction.try_normalize() else {
        return;
    };
    let first_across = along.any_orthonormal_vector();
    let second_across = along.cross(first_across).normalize_or_zero();
    let base = at - along * OUTSIDE_REACH;
    let tip = at + along * TIP_INSET;
    let neck = tip - along * HEAD_LENGTH;
    for across in [first_across, second_across] {
        let normal = along.cross(across);
        append_mesh_quad(
            [
                base - across * SHAFT_HALF_WIDTH,
                base + across * SHAFT_HALF_WIDTH,
                neck + across * SHAFT_HALF_WIDTH,
                neck - across * SHAFT_HALF_WIDTH,
            ],
            normal,
            &mut geometry.positions,
            &mut geometry.normals,
            &mut geometry.indices,
        );
        append_mesh_triangle(
            [
                neck - across * HEAD_HALF_WIDTH,
                neck + across * HEAD_HALF_WIDTH,
                tip,
            ],
            normal,
            &mut geometry.positions,
            &mut geometry.normals,
            &mut geometry.indices,
        );
    }
}

/// The bounds and plane of whichever drag is open, if one is.
pub(crate) fn active_drag_plane(
    state: &EditorState,
    _simulation: &AppSimulation,
) -> Option<(Vec3, Vec3, PlacementPlane)> {
    if let Some(drag) = state.region_drag.as_ref() {
        let (low, high) = region_world_bounds(&drag.region);
        return Some((low, high, drag.plane));
    }
    if let Some(drag) = state.delete_drag.as_ref() {
        let (low, high) = block_box_bounds(drag.start, drag.span);
        return Some((low, high, drag.plane));
    }
    let drag = state.block_drag.as_ref()?;
    let (low, high) = drag.volume.bounds();
    Some((low, high, drag.plane))
}
