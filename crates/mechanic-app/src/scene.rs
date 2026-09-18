//! Startup: the garage scene, cameras, lights, and the shared meshes and materials every editor visual uses.

use crate::avatar::spawn_player_avatar;
use crate::camera::{FovCamera, MainCamera, PlayerCamera, PlayerState};
use crate::chroma::ConstructionRenderMaterial;
use crate::editor::placement::{PlacementLatticeVisual, SmartGuideVisual, SmartSnapRangeVisual};
use crate::editor::preview::{
    ActionPreview, BearingVisual, ConstructionVisual, DeletePreview, DriveXrayVisual,
    EditorVisuals, JointXrayVisual, SelectionPreview,
};
use crate::editor::shape_actions::{
    SHAPE_SELECTION_COLOR, ShapeArrowVisual, ShapeNodeVisual, ShapePlaneVisual, ShapeSelectedVisual,
};
use crate::editor::wiring::{WireDragVisual, WireHoverVisual};
use crate::render::authored::{AuthoredPart, AuthoredPartVisual, CONTROLLER_SURFACE_COLOR};
use crate::render::environment::{SKY_CUBEMAP_SIZE, SKY_ENVIRONMENT_INTENSITY, sky_cubemap};
use crate::render::materials::{
    BearingTextureMipsPending, PREVIEW_RENDER_DEPTH_BIAS, authored_part_material,
    authored_preview_material, bearing_surface_material, configure_repeating_texture,
    construction_material, construction_tint_mask_path, material_index, preview_material,
};
use crate::render::mesh::bearing::single_bearing_mesh;
use crate::render::mesh::construction::{single_authored_part_mesh, single_cylinder_mesh};
use crate::render::mesh::drive::wire_drag_preview_mesh;
use crate::render::mesh::primitives::degenerate_overlay_mesh;
use crate::settings::AppSettings;
use crate::{camera, garage, render_diagnostics, render_experiments, tool_fx};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::ImageLoaderSettings;
use bevy::prelude::{
    Alpha, AlphaMode, AssetServer, Assets, Camera, Camera3d, ClearColorConfig, Color, Commands,
    Cuboid, GeneratedEnvironmentMapLight, Image, Mesh, Mesh3d, MeshMaterial3d, Name,
    PerspectiveProjection, Projection, Res, ResMut, StandardMaterial, Transform, Vec3, Visibility,
    default, format, vec,
};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use mechanic_core::{BearingDimensions, ConstructionMaterial, CylinderDimensions};

#[expect(
    clippy::too_many_lines,
    reason = "one-time Bevy scene composition is clearest in declaration order"
)]
pub(crate) fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    settings: Res<AppSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut construction_render_materials: ResMut<Assets<ConstructionRenderMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let construction_meshes = ConstructionMaterial::ALL.map(|_| meshes.add(Cuboid::default()));
    let bearing_mesh = meshes.add(Cuboid::default());
    let joint_xray_mesh = meshes.add(Cuboid::default());
    let shape_node_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_selected_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_plane_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_arrow_mesh = meshes.add(degenerate_overlay_mesh());
    let placement_lattice_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_guide_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_snap_range_mesh = meshes.add(degenerate_overlay_mesh());
    let controller_mesh = meshes.add(Cuboid::default());
    let gas_engine_mesh = meshes.add(Cuboid::default());
    let electric_engine_mesh = meshes.add(Cuboid::default());
    let gas_transmission_mesh = meshes.add(Cuboid::default());
    let electric_transmission_mesh = meshes.add(Cuboid::default());
    let servo_mesh = meshes.add(Cuboid::default());
    let seat_mesh = meshes.add(Cuboid::default());
    let input_mesh = meshes.add(Cuboid::default());
    let dimension_link_disabled_mesh = meshes.add(Cuboid::default());
    let dimension_link_enabled_mesh = meshes.add(Cuboid::default());
    let authored_preview_meshes =
        AuthoredPart::ALL.map(|appearance| meshes.add(single_authored_part_mesh(appearance)));
    let drive_xray_mesh = meshes.add(Cuboid::default());
    let wire_drag_mesh = meshes.add(wire_drag_preview_mesh(Vec3::ZERO, Vec3::ZERO));
    let wire_hover_mesh = meshes.add(degenerate_overlay_mesh());
    let cube_preview_mesh = meshes.add(Cuboid::default());
    let cylinder_preview_mesh = meshes.add(single_cylinder_mesh(CylinderDimensions::default()));
    let bearing_preview_mesh = meshes.add(single_bearing_mesh(BearingDimensions::default()));
    let block_drag_preview_mesh = meshes.add(Cuboid::default());
    let delete_drag_preview_mesh = meshes.add(Cuboid::default());
    let weld_hover_preview_mesh = meshes.add(Cuboid::default());
    let weld_selection_preview_mesh = meshes.add(Cuboid::default());
    let white_tint_mask = images.add(Image::new_fill(
        Extent3d::default(),
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    ));
    let tint_mask = |material| match construction_tint_mask_path(material) {
        Some(path) => asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                configure_repeating_texture(settings, false);
            })
            .load(path),
        None => white_tint_mask.clone(),
    };
    let construction_materials = ConstructionMaterial::ALL.map(|material| {
        construction_render_materials.add(construction_material(
            &asset_server,
            material,
            tint_mask(material),
        ))
    });
    // Faded copies, swapped in while a region is being edited so the area under
    // the cursor is the only thing that reads as solid.
    let ghost_materials = ConstructionMaterial::ALL.map(|material| {
        let mut ghost = construction_material(&asset_server, material, tint_mask(material));
        ghost.base.base_color = ghost.base.base_color.with_alpha(0.16);
        ghost.base.alpha_mode = AlphaMode::Blend;
        construction_render_materials.add(ghost)
    });
    let bearing_material = bearing_surface_material(&asset_server);
    commands.insert_resource(BearingTextureMipsPending(vec![
        bearing_material
            .base_color_texture
            .clone()
            .expect("the bearing has a base-color map"),
        bearing_material
            .normal_map_texture
            .clone()
            .expect("the bearing has a normal map"),
        bearing_material
            .metallic_roughness_texture
            .clone()
            .expect("the bearing has an ORM map"),
    ]));
    let bearing_material = materials.add(bearing_material);
    let authored_materials = [
        authored_part_material(&asset_server, "machines/controller/controller"),
        authored_part_material(&asset_server, "machines/gas_engine/gas_engine"),
        authored_part_material(&asset_server, "machines/electric_engine/electric_engine"),
        authored_part_material(&asset_server, "machines/transmission_gas/transmission_gas"),
        authored_part_material(
            &asset_server,
            "machines/transmission_electric/transmission_electric",
        ),
        authored_part_material(&asset_server, "machines/servo/servo"),
        authored_part_material(&asset_server, "machines/seat/seat"),
        authored_part_material(&asset_server, "machines/input/input"),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/disabled/dimension_link",
        ),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/enabled/dimension_link",
        ),
    ];
    let authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 1.0, 1.0, 0.46),
        ))
    });
    let invalid_authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 0.18, 0.16, 0.52),
        ))
    });
    let authored_materials = authored_materials.map(|material| materials.add(material));
    let drive_xray_material = materials.add(StandardMaterial {
        base_color: CONTROLLER_SURFACE_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let wire_drag_material = drive_xray_material.clone();
    let wire_hover_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.86, 0.99, 1.0),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let joint_xray_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let white_preview_material = materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.34)));
    let chroma_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.46)));
    let red_preview_material = materials.add(preview_material(Color::srgba(1.0, 0.06, 0.04, 0.46)));
    let amber_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 0.60, 0.06, 0.46)));
    let green_preview_material =
        materials.add(preview_material(Color::srgba(0.12, 1.0, 0.28, 0.52)));
    let placement_lattice_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.24),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_guide_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.86, 0.18),
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_snap_range_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.86, 0.18, 0.34),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });

    spawn_player_avatar(&mut commands, &mut meshes, &mut materials);

    commands.insert_resource(EditorVisuals {
        construction_meshes: construction_meshes.clone(),
        construction_materials: construction_materials.clone(),
        ghost_materials: ghost_materials.clone(),
        authored_materials: authored_materials.clone(),
        bearing_material: bearing_material.clone(),
        bearing_mesh: bearing_mesh.clone(),
        joint_xray_mesh: joint_xray_mesh.clone(),
        shape_node_mesh: shape_node_mesh.clone(),
        shape_selected_mesh: shape_selected_mesh.clone(),
        shape_plane_mesh: shape_plane_mesh.clone(),
        shape_arrow_mesh: shape_arrow_mesh.clone(),
        controller_mesh: controller_mesh.clone(),
        gas_engine_mesh: gas_engine_mesh.clone(),
        electric_engine_mesh: electric_engine_mesh.clone(),
        gas_transmission_mesh: gas_transmission_mesh.clone(),
        electric_transmission_mesh: electric_transmission_mesh.clone(),
        servo_mesh: servo_mesh.clone(),
        seat_mesh: seat_mesh.clone(),
        input_mesh: input_mesh.clone(),
        dimension_link_disabled_mesh: dimension_link_disabled_mesh.clone(),
        dimension_link_enabled_mesh: dimension_link_enabled_mesh.clone(),
        authored_preview_meshes,
        authored_preview_materials,
        invalid_authored_preview_materials,
        drive_xray_mesh: drive_xray_mesh.clone(),
        wire_drag_mesh: wire_drag_mesh.clone(),
        wire_hover_mesh: wire_hover_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
        cylinder_preview_mesh,
        bearing_preview_mesh,
        white_preview_material: white_preview_material.clone(),
        chroma_preview_material,
        green_preview_material,
        red_preview_material: red_preview_material.clone(),
        amber_preview_material,
        block_drag_preview_mesh,
        delete_drag_preview_mesh,
        weld_hover_preview_mesh,
        weld_selection_preview_mesh,
    });

    garage::spawn(&mut commands, &asset_server, &mut meshes, &mut materials);
    for material in ConstructionMaterial::ALL {
        let index = material_index(material);
        commands.spawn((
            Name::new(format!("{} construction mesh", material.label())),
            Mesh3d(construction_meshes[index].clone()),
            MeshMaterial3d(construction_materials[index].clone()),
            NoFrustumCulling,
            Visibility::Hidden,
            ConstructionVisual(material),
        ));
    }
    commands.spawn((
        Name::new("Bearing mesh"),
        Mesh3d(bearing_mesh.clone()),
        MeshMaterial3d(bearing_material),
        NoFrustumCulling,
        Visibility::Hidden,
        BearingVisual,
    ));
    commands.spawn((
        Name::new("Control block mesh"),
        Mesh3d(controller_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Controller.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Controller),
    ));
    commands.spawn((
        Name::new("Gas engine mesh"),
        Mesh3d(gas_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasEngine),
    ));
    commands.spawn((
        Name::new("Electric engine mesh"),
        Mesh3d(electric_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricEngine),
    ));
    commands.spawn((
        Name::new("Gas transmission mesh"),
        Mesh3d(gas_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasTransmission),
    ));
    commands.spawn((
        Name::new("Electric transmission mesh"),
        Mesh3d(electric_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricTransmission),
    ));
    commands.spawn((
        Name::new("Servo mesh"),
        Mesh3d(servo_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Servo.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Servo),
    ));
    commands.spawn((
        Name::new("Seat mesh"),
        Mesh3d(seat_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Seat.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Seat),
    ));
    commands.spawn((
        Name::new("Input mesh"),
        Mesh3d(input_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Input.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Input),
    ));
    commands.spawn((
        Name::new("Disabled Dimension Link mesh"),
        Mesh3d(dimension_link_disabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkDisabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkDisabled),
    ));
    commands.spawn((
        Name::new("Enabled Dimension Link mesh"),
        Mesh3d(dimension_link_enabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkEnabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkEnabled),
    ));
    commands.spawn((
        Name::new("Joint x-ray mesh"),
        Mesh3d(joint_xray_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        JointXrayVisual,
    ));
    commands.spawn((
        Name::new("Placement lattice"),
        Mesh3d(placement_lattice_mesh),
        MeshMaterial3d(placement_lattice_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        PlacementLatticeVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart placement guides"),
        Mesh3d(smart_guide_mesh),
        MeshMaterial3d(smart_guide_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartGuideVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart snap range"),
        Mesh3d(smart_snap_range_mesh),
        MeshMaterial3d(smart_snap_range_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartSnapRangeVisual::default(),
    ));
    let shape_selected_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Selected shape node markers"),
        Mesh3d(shape_selected_mesh),
        MeshMaterial3d(shape_selected_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeSelectedVisual,
    ));
    let shape_plane_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.14),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Drag plane"),
        Mesh3d(shape_plane_mesh),
        MeshMaterial3d(shape_plane_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapePlaneVisual,
    ));
    commands.spawn((
        Name::new("Drag plane arrows"),
        Mesh3d(shape_arrow_mesh),
        MeshMaterial3d(shape_selected_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeArrowVisual,
    ));
    commands.spawn((
        Name::new("Shape node markers"),
        Mesh3d(shape_node_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeNodeVisual,
    ));
    commands.spawn((
        Name::new("Drive x-ray mesh"),
        Mesh3d(drive_xray_mesh),
        MeshMaterial3d(drive_xray_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        DriveXrayVisual,
    ));
    // Kept visible with a degenerate mesh while idle: a hidden mesh has no slab
    // allocation, so writing the first frame of a drag into it would log a
    // use-after-free.
    commands.spawn((
        Name::new("Drive wire drag"),
        Mesh3d(wire_drag_mesh),
        MeshMaterial3d(wire_drag_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireDragVisual,
    ));
    commands.spawn((
        Name::new("Drive wire hover"),
        Mesh3d(wire_hover_mesh),
        MeshMaterial3d(wire_hover_material),
        Transform::default(),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireHoverVisual,
    ));
    commands.spawn((
        Name::new("Action preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material.clone()),
        Transform::default(),
        Visibility::Hidden,
        ActionPreview,
    ));
    commands.spawn((
        Name::new("Selection preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material),
        Transform::default(),
        Visibility::Hidden,
        SelectionPreview,
    ));
    commands.spawn((
        Name::new("Delete preview"),
        Mesh3d(cube_preview_mesh),
        MeshMaterial3d(red_preview_material),
        Transform::default(),
        Visibility::Hidden,
        DeletePreview,
    ));

    // Filter the authored sky once, then retain the resulting diffuse and
    // roughness-aware specular maps without regenerating them every frame.
    let environment_map = images.add(sky_cubemap(SKY_CUBEMAP_SIZE));

    let player_camera = PlayerCamera::default();
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: settings.camera_fov_degrees().to_radians(),
        ..default()
    });
    let camera_transform = player_camera.apply_pullback(
        PlayerState::default().position + Vec3::Y * camera::EYE_HEIGHT,
        player_camera.look_rotation(),
    );
    commands
        .spawn((
            Name::new("Player camera"),
            tool_fx::bloom(),
            tool_fx::FxCamera,
            Camera3d::default(),
            render_experiments::current().msaa(),
            projection.clone(),
            garage::EXPOSURE,
            Tonemapping::SomewhatBoringDisplayTransform,
            garage::fog(),
            GeneratedEnvironmentMapLight {
                environment_map,
                intensity: SKY_ENVIRONMENT_INTENSITY,
                ..default()
            },
            camera_transform,
            player_camera,
            MainCamera,
            render_diagnostics::ProfiledCamera,
            FovCamera,
        ))
        .with_children(|camera| {
            camera.spawn((
                Name::new("Joint x-ray camera"),
                Camera3d::default(),
                // Both cameras share the world target and must use the same MSAA.
                render_experiments::current().msaa(),
                projection,
                // The overlay rides the camera that draws last. This pass loads
                // rather than clears, so an overlay painted before it is drawn
                // over by the joints showing through.
                bevy_mosaic::MosaicCamera,
                Camera {
                    order: 2,
                    clear_color: ClearColorConfig::None,
                    output_mode: bevy::camera::CameraOutputMode::Write {
                        blend_state: Some(
                            bevy::render::render_resource::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
                        ),
                        clear_color: ClearColorConfig::None,
                    },
                    ..default()
                },
                Tonemapping::None,
                RenderLayers::layer(1),
                FovCamera,
                Transform::default(),
            ));
        });
}
