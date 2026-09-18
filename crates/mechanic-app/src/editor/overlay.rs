//! Placement lattice, smart-snap guides, and shape-edit overlay geometry.

use crate::builder::{
    PlacementBounds, PlacementGrid, PlacementPlane, SmartGuide, part_world_bounds,
};
use crate::controls::GameAction;
use crate::editor::placement::{
    PlacementLatticeKey, PlacementLatticeVisual, SmartGuideVisual, SmartSnapRangeKey,
    SmartSnapRangeVisual,
};
use crate::editor::shape_actions::{
    ShapeArrowVisual, ShapeNodeVisual, ShapePlaneVisual, ShapeSelectedVisual,
};
use crate::editor::state::EditorState;
use crate::hotbar::{SelectedTool, Tool};
use crate::render::mesh::construction::append_transformed_cuboid;
use crate::render::mesh::primitives::{append_mesh_quad, append_mesh_triangle, renderable_mesh};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::{
    Assets, ButtonInput, Handle, IVec3, Mesh, Mesh3d, Or, Quat, Query, Res, ResMut, Single,
    Transform, Vec3, Visibility, With, Without,
};
use bevy::render::render_resource::PrimitiveTopology;
use mechanic_core::{
    GRID_UNIT_METERS, POSITION_TICK_METERS, POSITION_TICKS_PER_GRID_UNIT,
    POSITION_TICKS_PER_HALF_GRID_UNIT, PartSpec, ShapeRegion,
};

/// One overlay batch being assembled.
#[derive(Default)]
pub(crate) struct OverlayGeometry {
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) indices: Vec<u32>,
}

/// Writes one overlay batch into its mesh, reporting whether it has anything to
/// draw.
pub(crate) fn write_overlay(
    meshes: &mut Assets<Mesh>,
    handle: &Handle<Mesh>,
    geometry: OverlayGeometry,
) -> Visibility {
    if geometry.positions.is_empty() {
        return Visibility::Hidden;
    }
    if let Some(mut mesh) = meshes.get_mut(handle) {
        *mesh = renderable_mesh(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals)
            .with_inserted_indices(Indices::U32(geometry.indices)),
        );
    }
    Visibility::Visible
}

#[expect(clippy::type_complexity, clippy::too_many_lines)]
pub(crate) fn sync_placement_overlays(
    state: Res<EditorState>,
    selection: Res<SelectedTool>,
    actions: Res<ButtonInput<GameAction>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut lattice: Single<
        (&Mesh3d, &mut Visibility, &mut PlacementLatticeVisual),
        (
            With<PlacementLatticeVisual>,
            Without<SmartGuideVisual>,
            Without<SmartSnapRangeVisual>,
        ),
    >,
    mut guides: Single<
        (&Mesh3d, &mut Visibility, &mut SmartGuideVisual),
        (
            With<SmartGuideVisual>,
            Without<PlacementLatticeVisual>,
            Without<SmartSnapRangeVisual>,
        ),
    >,
    mut range: Single<
        (&Mesh3d, &mut Visibility, &mut SmartSnapRangeVisual),
        (
            With<SmartSnapRangeVisual>,
            Without<PlacementLatticeVisual>,
            Without<SmartGuideVisual>,
        ),
    >,
) {
    let tool = selection.active_editor_tool();
    let placing = matches!(
        tool,
        Some(
            Tool::Block
                | Tool::Cylinder
                | Tool::Bearing
                | Tool::Controller
                | Tool::GasEngine
                | Tool::ElectricEngine
                | Tool::Transmission
                | Tool::Servo
                | Tool::Seat
                | Tool::Input
                | Tool::DimensionLink
        )
    );
    let target = state
        .block_drag
        .as_ref()
        .map(|drag| {
            let (low, high) = drag.volume.bounds();
            (low, high, Some(drag.plane))
        })
        .or_else(|| {
            state
                .pipe_drag
                .as_ref()
                .map(|drag| (drag.endpoint, drag.endpoint, None))
        })
        .or_else(|| {
            state.preview.map(|candidate| {
                let (low, high) = part_world_bounds(PartSpec::Cuboid(candidate.spec));
                (low, high, None)
            })
        })
        .or_else(|| {
            state.cylinder_preview.map(|candidate| {
                let (low, high) = part_world_bounds(PartSpec::Cylinder(candidate.spec));
                (low, high, None)
            })
        })
        .or_else(|| {
            (tool == Some(Tool::Bearing))
                .then_some(state.bearing_preview_anchor?)
                .map(|anchor| (anchor, anchor, None))
        });
    let Some((low, high, plane)) = target.filter(|_| placing) else {
        *lattice.1 = Visibility::Hidden;
        *guides.1 = Visibility::Hidden;
        *range.1 = Visibility::Hidden;
        return;
    };

    let origin = placement_origin_meters(state.placement_bounds);
    let low_ticks = position_ticks(low + origin);
    let high_ticks = position_ticks(high + origin);
    let key = PlacementLatticeKey {
        grid: state.placement_grid,
        low_ticks,
        high_ticks,
        plane,
    };
    if lattice.2.key == Some(key) {
        *lattice.1 = Visibility::Visible;
    } else {
        let geometry = placement_lattice_geometry(
            low_ticks.as_vec3() * POSITION_TICK_METERS - origin,
            high_ticks.as_vec3() * POSITION_TICK_METERS - origin,
            origin,
            state.placement_grid,
            plane,
        );
        *lattice.1 = write_overlay(&mut meshes, &lattice.0.0, geometry);
        lattice.2.key = Some(key);
    }

    if guides.2.guides == state.smart_guides {
        *guides.1 = if state.smart_guides.is_empty() {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    } else {
        let geometry = smart_guide_geometry(&state.smart_guides);
        *guides.1 = write_overlay(&mut meshes, &guides.0.0, geometry);
        guides.2.guides.clone_from(&state.smart_guides);
    }

    if actions.pressed(GameAction::ToggleObjectSnap) {
        let range_ticks = position_tick(state.smart_snap.range);
        let key = SmartSnapRangeKey {
            low_ticks,
            high_ticks,
            range_ticks,
            plane,
        };
        if range.2.key == Some(key) {
            *range.1 = Visibility::Visible;
        } else {
            let geometry = smart_snap_range_geometry(
                low_ticks.as_vec3() * POSITION_TICK_METERS - origin,
                high_ticks.as_vec3() * POSITION_TICK_METERS - origin,
                state.smart_snap.range,
                plane,
            );
            *range.1 = write_overlay(&mut meshes, &range.0.0, geometry);
            range.2.key = Some(key);
        }
    } else {
        *range.1 = Visibility::Hidden;
    }
}

/// Overlay meshes live in the gesture's grid; moving its frame requires no mesh rebuild.
#[expect(clippy::type_complexity)]
pub(crate) fn sync_edit_overlay_transforms(
    state: Res<EditorState>,
    mut overlays: Query<
        &mut Transform,
        Or<(
            With<ShapeNodeVisual>,
            With<ShapeSelectedVisual>,
            With<ShapePlaneVisual>,
            With<ShapeArrowVisual>,
            With<PlacementLatticeVisual>,
            With<SmartGuideVisual>,
            With<SmartSnapRangeVisual>,
        )>,
    >,
) {
    let frame = state
        .edit_context
        .map_or(mechanic_core::ConstructionFrame::IDENTITY, |context| {
            context.frame_to_world
        });
    for mut transform in &mut overlays {
        *transform =
            Transform::from_translation(frame.translation()).with_rotation(frame.rotation());
    }
}

#[expect(clippy::cast_possible_truncation)]
pub(crate) fn placement_origin_meters(bounds: PlacementBounds) -> Vec3 {
    match bounds {
        PlacementBounds::Garage
        | PlacementBounds::GarageBuild
        | PlacementBounds::GarageBuildFrame { .. } => Vec3::ZERO,
        PlacementBounds::World { origin } => Vec3::new(origin.x as f32, 0.0, origin.y as f32),
    }
}

pub(crate) fn position_ticks(position: Vec3) -> IVec3 {
    (position / POSITION_TICK_METERS).round().as_ivec3()
}

#[expect(clippy::cast_possible_truncation)]
pub(crate) fn position_tick(position: f32) -> i32 {
    (position / POSITION_TICK_METERS).round() as i32
}

#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn placement_lattice_geometry(
    selection_low: Vec3,
    selection_high: Vec3,
    origin: Vec3,
    grid: PlacementGrid,
    plane: Option<PlacementPlane>,
) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    let step = grid.step_ticks() as f32 * POSITION_TICK_METERS;
    let mut low = selection_low - Vec3::splat(step);
    let mut high = selection_high + Vec3::splat(step);
    if let Some(plane) = plane {
        let normal = plane.normal_axis();
        low[normal] = (selection_low[normal] + selection_high[normal]) * 0.5;
        high[normal] = low[normal];
        append_planar_lattice(
            selection_low,
            selection_high,
            low,
            high,
            origin,
            grid,
            plane,
            &mut geometry,
        );
        return geometry;
    }

    let coordinates: [Vec<f32>; 3] = core::array::from_fn(|axis| {
        lattice_coordinates(
            low[axis] + origin[axis],
            high[axis] + origin[axis],
            axis,
            grid,
        )
        .into_iter()
        .map(|global| global - origin[axis])
        .collect::<Vec<_>>()
    });
    for direction in 0..3 {
        let first = (direction + 1) % 3;
        let second = (direction + 2) % 3;
        for &a in &coordinates[first] {
            for &b in &coordinates[second] {
                if coordinate_inside(a, selection_low[first], selection_high[first])
                    && coordinate_inside(b, selection_low[second], selection_high[second])
                {
                    continue;
                }
                let mut at = (low + high) * 0.5;
                at[first] = a;
                at[second] = b;
                let first_tick = ((a + origin[first]) / POSITION_TICK_METERS).round() as i32;
                let second_tick = ((b + origin[second]) / POSITION_TICK_METERS).round() as i32;
                let thickness = lattice_thickness(first, first_tick)
                    .max(lattice_thickness(second, second_tick));
                append_lattice_line(
                    low[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    &mut geometry,
                );
            }
        }
    }
    geometry
}

#[expect(clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub(crate) fn append_planar_lattice(
    selection_low: Vec3,
    selection_high: Vec3,
    low: Vec3,
    high: Vec3,
    origin: Vec3,
    grid: PlacementGrid,
    plane: PlacementPlane,
    geometry: &mut OverlayGeometry,
) {
    let [first, second] = plane.tangent_axes();
    for (direction, cross) in [(first, second), (second, first)] {
        let coordinates = lattice_coordinates(
            low[cross] + origin[cross],
            high[cross] + origin[cross],
            cross,
            grid,
        );
        for global_coordinate in coordinates {
            let coordinate = global_coordinate - origin[cross];
            let tick = (global_coordinate / POSITION_TICK_METERS).round() as i32;
            let thickness = lattice_thickness(cross, tick);
            let mut at = (low + high) * 0.5;
            at[cross] = coordinate;
            if coordinate_inside(coordinate, selection_low[cross], selection_high[cross]) {
                append_lattice_line(
                    low[direction],
                    selection_low[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
                append_lattice_line(
                    selection_high[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
            } else {
                append_lattice_line(
                    low[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
            }
        }
    }
}

pub(crate) fn coordinate_inside(coordinate: f32, low: f32, high: f32) -> bool {
    const TOLERANCE: f32 = POSITION_TICK_METERS * 0.25;
    coordinate >= low - TOLERANCE && coordinate <= high + TOLERANCE
}

pub(crate) fn append_lattice_line(
    low: f32,
    high: f32,
    direction: usize,
    mut at: Vec3,
    thickness: f32,
    geometry: &mut OverlayGeometry,
) {
    if high - low <= f32::EPSILON {
        return;
    }
    at[direction] = (low + high) * 0.5;
    let mut half = Vec3::splat(thickness * 0.5);
    half[direction] = (high - low) * 0.5;
    append_transformed_cuboid(
        at,
        Quat::IDENTITY,
        half,
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn lattice_coordinates(
    low: f32,
    high: f32,
    axis: usize,
    grid: PlacementGrid,
) -> Vec<f32> {
    let step = grid.step_ticks();
    let phase = if axis == 1 {
        0
    } else {
        POSITION_TICKS_PER_HALF_GRID_UNIT.rem_euclid(step)
    };
    let low_tick = (low / POSITION_TICK_METERS).ceil() as i32;
    let high_tick = (high / POSITION_TICK_METERS).floor() as i32;
    let mut tick = low_tick + (phase - low_tick).rem_euclid(step);
    let mut coordinates = Vec::new();
    while tick <= high_tick {
        coordinates.push(tick as f32 * POSITION_TICK_METERS);
        tick = tick.saturating_add(step);
    }
    coordinates
}

pub(crate) fn lattice_thickness(axis: usize, tick: i32) -> f32 {
    let major_phase = if axis == 1 {
        0
    } else {
        POSITION_TICKS_PER_HALF_GRID_UNIT
    };
    if (tick - major_phase).rem_euclid(POSITION_TICKS_PER_GRID_UNIT) == 0 {
        0.004
    } else if (tick - major_phase).rem_euclid(20) == 0 {
        0.002
    } else {
        0.0008
    }
}

pub(crate) fn smart_guide_geometry(guides: &[SmartGuide]) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    for guide in guides {
        append_overlay_bar(guide.from, guide.to, 0.007, &mut geometry);
        for point in [guide.from, guide.to] {
            append_transformed_cuboid(
                point,
                Quat::IDENTITY,
                Vec3::splat(0.012),
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
        }
    }
    geometry
}

pub(crate) fn smart_snap_range_geometry(
    selection_low: Vec3,
    selection_high: Vec3,
    range: f32,
    plane: Option<PlacementPlane>,
) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    if let Some(plane) = plane {
        let [first, second] = plane.tangent_axes();
        append_snap_range_outline(
            selection_low,
            selection_high,
            range,
            first,
            second,
            plane.normal_axis(),
            &mut geometry,
        );
    } else {
        for (first, second, normal) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            append_snap_range_outline(
                selection_low,
                selection_high,
                range,
                first,
                second,
                normal,
                &mut geometry,
            );
        }
    }
    geometry
}

pub(crate) fn append_snap_range_outline(
    selection_low: Vec3,
    selection_high: Vec3,
    range: f32,
    first: usize,
    second: usize,
    normal: usize,
    geometry: &mut OverlayGeometry,
) {
    const CORNER_SEGMENTS: u8 = 8;
    const THICKNESS: f32 = 0.004;

    let normal_coordinate = (selection_low[normal] + selection_high[normal]) * 0.5;
    let point = |first_coordinate: f32, second_coordinate: f32| {
        let mut point = Vec3::ZERO;
        point[first] = first_coordinate;
        point[second] = second_coordinate;
        point[normal] = normal_coordinate;
        point
    };

    for second_coordinate in [
        selection_low[second] - range,
        selection_high[second] + range,
    ] {
        append_overlay_bar(
            point(selection_low[first], second_coordinate),
            point(selection_high[first], second_coordinate),
            THICKNESS,
            geometry,
        );
    }
    for first_coordinate in [selection_low[first] - range, selection_high[first] + range] {
        append_overlay_bar(
            point(first_coordinate, selection_low[second]),
            point(first_coordinate, selection_high[second]),
            THICKNESS,
            geometry,
        );
    }

    for (center_first, center_second, start_angle) in [
        (selection_high[first], selection_high[second], 0.0),
        (
            selection_low[first],
            selection_high[second],
            core::f32::consts::FRAC_PI_2,
        ),
        (
            selection_low[first],
            selection_low[second],
            core::f32::consts::PI,
        ),
        (
            selection_high[first],
            selection_low[second],
            3.0 * core::f32::consts::FRAC_PI_2,
        ),
    ] {
        let mut previous = point(
            center_first + range * start_angle.cos(),
            center_second + range * start_angle.sin(),
        );
        for segment in 1..=CORNER_SEGMENTS {
            let angle = start_angle
                + core::f32::consts::FRAC_PI_2 * f32::from(segment) / f32::from(CORNER_SEGMENTS);
            let next = point(
                center_first + range * angle.cos(),
                center_second + range * angle.sin(),
            );
            append_overlay_bar(previous, next, THICKNESS, geometry);
            previous = next;
        }
    }
}

pub(crate) fn append_overlay_bar(
    from: Vec3,
    to: Vec3,
    thickness: f32,
    geometry: &mut OverlayGeometry,
) {
    let delta = to - from;
    let length = delta.length().max(thickness);
    let rotation = if delta.length_squared() <= f32::EPSILON {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(Vec3::X, delta.normalize())
    };
    append_transformed_cuboid(
        (from + to) * 0.5,
        rotation,
        Vec3::new(length * 0.5, thickness * 0.5, thickness * 0.5),
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

pub(crate) fn append_dashed_overlay_bar(
    from: Vec3,
    to: Vec3,
    thickness: f32,
    geometry: &mut OverlayGeometry,
) {
    let delta = to - from;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let direction = delta / length;
    let dash = (thickness * 4.0).max(0.018);
    let gap = dash * 0.7;
    let mut start = 0.0;
    while start < length {
        let end = (start + dash).min(length);
        append_overlay_bar(
            from + direction * start,
            from + direction * end,
            thickness,
            geometry,
        );
        start += dash + gap;
    }
}

/// Draws a region's bounding box as twelve thin bars, so a dragged area reads
/// as a volume rather than a face.
pub(crate) fn append_region_outline(region: &ShapeRegion, geometry: &mut OverlayGeometry) {
    const THICKNESS: f32 = 0.012;
    let (low_steps, high_steps) = region.bounds_steps();
    let low = low_steps.as_vec3() * POSITION_TICK_METERS;
    let high = high_steps.as_vec3() * POSITION_TICK_METERS;
    let centre = (low + high) * 0.5;
    let extent = high - low;
    for axis in 0..3 {
        let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut size = Vec3::splat(THICKNESS);
        // The bar runs the full length of its axis and overshoots at the ends
        // by its own width, which is what closes the corners.
        size[axis] = extent[axis] + THICKNESS;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let mut at = centre;
            at[first] += a * extent[first] * 0.5;
            at[second] += b * extent[second] * 0.5;
            append_transformed_cuboid(
                at,
                Quat::IDENTITY,
                size * 0.5,
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
        }
    }
}

/// Draws the plane an area drag is sliding along, as a translucent sheet through
/// the block the drag started on — the same plane the pointer is measured
/// against, so Rotate visibly rotates it.
pub(crate) fn append_drag_plane(
    low: Vec3,
    high: Vec3,
    plane: PlacementPlane,
    geometry: &mut OverlayGeometry,
) {
    const THICKNESS: f32 = 0.004;
    /// Overhang past the area, so the sheet reads as a plane rather than a lid.
    const MARGIN: f32 = GRID_UNIT_METERS;
    let normal_axis = plane.normal_axis();
    let mut size = (high - low) + Vec3::splat(MARGIN * 2.0);
    size[normal_axis] = THICKNESS;
    append_transformed_cuboid(
        (low + high) * 0.5,
        Quat::IDENTITY,
        size * 0.5,
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

/// Draws an arrow along each of the drag plane's four cardinal directions, so
/// the plane says which two axes the pointer is driving.
pub(crate) fn append_plane_arrows(
    low: Vec3,
    high: Vec3,
    plane: PlacementPlane,
    geometry: &mut OverlayGeometry,
) {
    /// Clear of the sheet's own slab, so the arrows never fight it for depth.
    const LIFT: f32 = 0.005;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;
    /// How far an arrow reaches, kept between these so it reads as a gizmo on a
    /// single block and does not span the whole sheet on a large area.
    const MIN_REACH: f32 = 0.14;
    const MAX_REACH: f32 = 0.55;

    let centre = (low + high) * 0.5;
    let extents = (high - low) * 0.5;
    let normal_axis = plane.normal_axis();
    let normal = Vec3::AXES[normal_axis];
    for (index, axis) in plane.tangent_axes().into_iter().enumerate() {
        let along = Vec3::AXES[axis];
        let across = Vec3::AXES[plane.tangent_axes()[1 - index]];
        let reach = (extents[axis] + GRID_UNIT_METERS * 0.5).clamp(MIN_REACH, MAX_REACH);
        let shaft = (reach - HEAD_LENGTH).max(HEAD_LENGTH * 0.5);
        for direction in [1.0_f32, -1.0] {
            let tip = along * (direction * reach);
            let neck = along * (direction * shaft);
            // One copy either side of the sheet, so the arrow reads whichever
            // face of the plane the camera is looking at.
            for side in [1.0_f32, -1.0] {
                let base = centre + normal * (side * LIFT);
                let facing = normal * side;
                append_mesh_quad(
                    [
                        base - across * SHAFT_HALF_WIDTH,
                        base + across * SHAFT_HALF_WIDTH,
                        base + neck + across * SHAFT_HALF_WIDTH,
                        base + neck - across * SHAFT_HALF_WIDTH,
                    ],
                    facing,
                    &mut geometry.positions,
                    &mut geometry.normals,
                    &mut geometry.indices,
                );
                append_mesh_triangle(
                    [
                        base + neck - across * HEAD_HALF_WIDTH,
                        base + neck + across * HEAD_HALF_WIDTH,
                        base + tip,
                    ],
                    facing,
                    &mut geometry.positions,
                    &mut geometry.normals,
                    &mut geometry.indices,
                );
            }
        }
    }
}

/// Draws a two-headed arrow along the one axis a cage vertex may currently
/// move. Two crossed profiles keep it readable from any camera angle.
pub(crate) fn append_axis_arrows(at: Vec3, axis: usize, geometry: &mut OverlayGeometry) {
    const GAP: f32 = 0.035;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;
    const REACH: f32 = 0.18;

    let along = Vec3::AXES[axis];
    let perpendicular = [(axis + 1) % 3, (axis + 2) % 3];
    for across_axis in perpendicular {
        let across = Vec3::AXES[across_axis];
        let normal = along.cross(across);
        for direction in [1.0_f32, -1.0] {
            let base = at + along * (direction * GAP);
            let neck = at + along * (direction * (REACH - HEAD_LENGTH));
            let tip = at + along * (direction * REACH);
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
}

/// A region's bounding box in world metres.
pub(crate) fn region_world_bounds(region: &ShapeRegion) -> (Vec3, Vec3) {
    let (low, high) = region.bounds_steps();
    (
        low.as_vec3() * POSITION_TICK_METERS,
        high.as_vec3() * POSITION_TICK_METERS,
    )
}
