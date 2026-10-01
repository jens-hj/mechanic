//! Compare the same lit surface through world and transparent tool cameras.
use super::*;
use bevy::{
    camera::RenderTarget,
    light::{Atmosphere, SunDisk, atmosphere::ScatteringMedium},
    pbr::AtmosphereSettings,
    render::{
        RenderPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::TextureFormat,
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    window::ExitCondition,
};

#[derive(Resource, Default)]
struct Pixels([Vec<u8>; 2]);

#[test]
#[ignore = "requires a real GPU"]
#[expect(
    clippy::too_many_lines,
    reason = "self-contained two-camera GPU regression fixture"
)]
fn tool_and_world_receive_matching_atmospheric_light() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins(ViewmodelLightingPlugin)
    .insert_resource(State::new(AppSpace::World))
    .init_resource::<Pixels>();
    app.finish();
    app.cleanup();
    eprintln!(
        "Tool lighting adapter: {:?}",
        app.sub_app(RenderApp)
            .world()
            .resource::<RenderAdapterInfo>()
            .0
    );
    let medium = app
        .world_mut()
        .resource_mut::<Assets<ScatteringMedium>>()
        .add(ScatteringMedium::earth(256, 256));
    app.world_mut().spawn((
        Atmosphere::earth(medium),
        GlobalTransform::from_translation(Vec3::new(0.0, -6_360_000.0, 0.0)),
    ));
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Sphere::new(1.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            perceptual_roughness: 0.6,
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 2.0, 0.0),
        RenderLayers::from_layers(&[0, VIEWMODEL_LAYER]),
    ));
    let sun = app
        .world_mut()
        .spawn((
            crate::sky::Sun,
            SunDisk::EARTH,
            DirectionalLight {
                illuminance: bevy::light::light_consts::lux::RAW_SUNLIGHT,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::default(),
        ))
        .id();
    let mut cameras = Vec::new();
    for index in 0..2 {
        let mut image = Image::new_target_texture(64, 64, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        app.world_mut()
            .spawn(Readback::texture(target.clone()))
            .observe(
                move |event: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
                    pixels.0[index].clone_from(&event.data);
                },
            );
        let mut camera = app.world_mut().spawn((
            Camera3d::default(),
            bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
            Camera {
                order: isize::from(index != 0),
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            RenderTarget::Image(target.into()),
            Exposure { ev100: 13.0 },
            Msaa::Off,
            Transform::from_xyz(0.0, 2.0, 4.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
        ));
        if index == 0 {
            camera.insert((
                MainCamera,
                AtmosphereSettings::default(),
                AmbientLight {
                    brightness: 0.0,
                    ..default()
                },
            ));
        } else {
            camera.insert((ViewmodelCamera, RenderLayers::layer(VIEWMODEL_LAYER)));
        }
        cameras.push(camera.id());
    }
    for (elevation, illuminance, ev100) in [
        (1.0_f32, bevy::light::light_consts::lux::RAW_SUNLIGHT, 13.0),
        (0.03, bevy::light::light_consts::lux::RAW_SUNLIGHT, 10.5),
        (1.0, 4.0, 1.0),
        (-0.2, bevy::light::light_consts::lux::RAW_SUNLIGHT, 1.0),
    ] {
        *app.world_mut().get_mut::<Transform>(sun).unwrap() = Transform::default()
            .looking_to(-Vec3::new(0.0, elevation.sin(), elevation.cos()), Vec3::Y);
        app.world_mut()
            .get_mut::<DirectionalLight>(sun)
            .unwrap()
            .illuminance = illuminance;
        app.world_mut()
            .get_mut::<Exposure>(cameras[0])
            .unwrap()
            .ev100 = ev100;
        for _ in 0..24 {
            app.update();
            app.sub_app(RenderApp)
                .world()
                .resource::<RenderDevice>()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
        let pixels = &app.world().resource::<Pixels>().0;
        assert_eq!(pixels[0].len(), 64 * 64 * 4);
        assert_eq!(pixels[1].len(), 64 * 64 * 4);
        let center = (32 * 64 + 32) * 4;
        eprintln!(
            "ev100={ev100}: world={:?}, tool={:?}",
            &pixels[0][center..center + 4],
            &pixels[1][center..center + 4]
        );
        for channel in 0..3 {
            assert!(
                pixels[0][center + channel].abs_diff(pixels[1][center + channel]) <= 3,
                "world/tool lighting differs at ev100={ev100}"
            );
        }
        if elevation > 0.0 {
            assert!(pixels[1][center] > 10, "lit surface must be visible");
        } else {
            assert!(
                pixels[1][center] < 5,
                "sun below the horizon must be occluded"
            );
        }
        assert_eq!(pixels[1][3], 0, "tool background must stay transparent");
        assert_eq!(pixels[1][center + 3], 255);
    }
    // A world-layer roof outside the camera frustum must still shadow the tool.
    *app.world_mut().get_mut::<Transform>(sun).unwrap() =
        Transform::default().looking_to(-Vec3::new(0.0, 1.0_f32.sin(), 1.0_f32.cos()), Vec3::Y);
    app.world_mut()
        .get_mut::<DirectionalLight>(sun)
        .unwrap()
        .illuminance = bevy::light::light_consts::lux::RAW_SUNLIGHT;
    app.world_mut()
        .get_mut::<Exposure>(cameras[0])
        .unwrap()
        .ev100 = 13.0;
    let roof_mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(2.0, 0.2, 2.0));
    let roof_material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let roof = app
        .world_mut()
        .spawn((
            Mesh3d(roof_mesh),
            MeshMaterial3d(roof_material),
            Transform::from_xyz(0.0, 4.0, 2.3),
        ))
        .id();
    let mut center_red = Vec::new();
    for covered in [true, false] {
        if !covered {
            app.world_mut().despawn(roof);
        }
        for _ in 0..12 {
            app.update();
            app.sub_app(RenderApp)
                .world()
                .resource::<RenderDevice>()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
        let center = (32 * 64 + 32) * 4;
        let pixels = &app.world().resource::<Pixels>().0;
        for channel in 0..3 {
            assert!(
                pixels[0][center + channel].abs_diff(pixels[1][center + channel]) <= 3,
                "world shadow occlusion must match in both cameras"
            );
        }
        center_red.push(pixels[1][center]);
    }
    eprintln!(
        "World roof shadow: covered={}, uncovered={}",
        center_red[0], center_red[1]
    );
    assert!(
        center_red[0] < center_red[1] / 2,
        "the tool must receive the world's roof shadow"
    );
    let render_tool = app
        .world()
        .entity(cameras[1])
        .get::<bevy::render::sync_world::RenderEntity>()
        .unwrap()
        .id();
    assert!(
        app.sub_app(RenderApp)
            .world()
            .get::<bevy::pbr::ViewLightEntities>(render_tool)
            .unwrap()
            .lights
            .is_empty()
    );
    app.world_mut()
        .get_mut::<Transform>(cameras[1])
        .unwrap()
        .translation
        .x += 0.1;
    app.update();
    assert!(
        !app.sub_app(RenderApp)
            .world()
            .get::<bevy::pbr::ViewLightEntities>(render_tool)
            .unwrap()
            .lights
            .is_empty(),
        "displaced cameras must use independent shadows"
    );
}
