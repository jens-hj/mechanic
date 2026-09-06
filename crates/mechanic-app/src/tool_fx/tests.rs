use super::*;
use particles::{edges, pulse};
fn frame(tool: Kind) -> EmitterFrame {
    EmitterFrame {
        tool,
        origin: Transform::IDENTITY,
        target: Vec3::Z,
        normal: Vec3::Y,
        connector_phase: 0.0,
        connector_plates: [None; 6],
    }
}
fn shards(p: &Particles) -> usize {
    p.shards.slots.iter().flatten().count()
}
fn traces(p: &Particles) -> usize {
    p.traces.slots.iter().flatten().count()
}
#[test]
fn fixed_pools_drop_excess_and_expire() {
    let mut p = Particles::default();
    for _ in 0..100 {
        p.request(
            Request::Sledge {
                hit: Vec3::ZERO,
                normal: Vec3::Y,
            },
            Vec3::ZERO,
        );
    }
    assert_eq!(shards(&p), SHARDS);
    assert_eq!(traces(&p), TRACES);
    for _ in 0..20 {
        p.advance(0.05, None);
    }
    assert_eq!(shards(&p), 0);
    assert_eq!(traces(&p), 0);
    assert_eq!(p.shards.slots.len(), SHARDS);
    assert_eq!(p.traces.slots.len(), TRACES);
}
#[test]
fn shard_tones_step_without_alpha_fading() {
    let mut p = Particles::default();
    p.request(
        Request::Freeze {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Vec3::ZERO,
    );
    let mut s = *p.shards.slots.iter().flatten().next().unwrap();
    for (age, tone) in [(0.0, 2), (0.4, 1), (0.8, 0)] {
        s.age = age * s.life;
        assert_eq!(s.tone(), tone);
        assert!((color(s.kind, tone)[3] - 1.0).abs() < f32::EPSILON);
    }
}
#[test]
fn sustained_density_is_independent_of_frame_rate() {
    for kind in [Kind::Matter, Kind::Welder] {
        let mut counts = Vec::new();
        for hz in [30_u16, 60, 120] {
            let mut p = Particles::default();
            for _ in 0..hz / 10 {
                p.advance(1.0 / f32::from(hz), Some(frame(kind)));
            }
            counts.push(shards(&p));
        }
        assert_eq!(counts[0], counts[1]);
        assert_eq!(counts[1], counts[2]);
    }
}
#[test]
fn hitches_integrate_at_most_fifty_milliseconds() {
    let mut hitch = Particles::default();
    let mut normal = Particles::default();
    hitch.advance(10.0, Some(frame(Kind::Matter)));
    normal.advance(0.05, Some(frame(Kind::Matter)));
    assert!((hitch.clock - normal.clock).abs() < f32::EPSILON);
    assert_eq!(shards(&hitch), shards(&normal));
}
#[test]
fn connector_rebuilds_at_moving_endpoint_and_cancels_immediately() {
    let mut p = Particles::default();
    for i in 0_u16..100 {
        let mut f = frame(Kind::Connector);
        f.target.x = f32::from(i);
        p.advance(0.001, Some(f));
        assert_eq!(traces(&p), 15);
    }
    assert!(p.traces.slots.iter().flatten().any(|s| s.b.x > 90.0));
    p.advance(0.001, None);
    assert_eq!(traces(&p), 0);
}
#[test]
fn every_large_box_edge_receives_dashes_within_budget() {
    for size in [
        Vec3::splat(0.25),
        Vec3::splat(1000.0),
        Vec3::new(1000.0, 0.25, 0.25),
    ] {
        for clock in [0.0, 0.1, 18.3] {
            let mut segments = Vec::new();
            halo(Vec3::ZERO, size, clock, |a, b, inner| {
                segments.push((a, b, inner));
            });
            assert!(segments.len() <= DASHES + 12);
            assert_eq!(segments.iter().filter(|s| s.2).count(), 12);
            for (a, b) in edges(-Vec3::splat(0.075), size + Vec3::splat(0.075)) {
                let edge = b - a;
                assert!(segments.iter().any(|&(x, y, inner)| !inner
                    && (x - a).cross(edge).length() < 0.1
                    && (y - a).cross(edge).length() < 0.1
                    && (x - a).dot(edge) >= 0.0
                    && (y - a).dot(edge) <= edge.length_squared() + 0.1));
            }
        }
    }
}
#[test]
fn aperture_pulse_uses_shared_clock_and_exact_range() {
    assert!((pulse(0.0) - 0.75).abs() < 1e-6);
    assert!((pulse(0.5 / 0.55) - 2.30).abs() < 1e-6);
    assert!((pulse(1.0 / 0.55) - 0.75).abs() < 1e-6);
}
#[test]
fn atlas_mask_selects_all_six_plates_and_excludes_cracks() {
    for center in [
        Vec2::new(0.25, 0.125),
        Vec2::new(0.75, 0.125),
        Vec2::new(0.25, 0.375),
        Vec2::new(0.75, 0.375),
        Vec2::new(0.125, 0.625),
        Vec2::new(0.375, 0.625),
    ] {
        assert!(aperture::aperture_mask(center));
        assert!(aperture::aperture_mask(center + Vec2::splat(0.02)));
        assert!(!aperture::aperture_mask(center + Vec2::X * 0.07));
    }
    assert!(!aperture::aperture_mask(Vec2::new(0.75, 0.625)));
}
#[test]
fn world_cleanup_clears_pending_and_live_feedback() {
    let mut fx = ToolFx::default();
    fx.push(Request::Matter { hit: Vec3::Z });
    fx.particles.advance(0.05, Some(frame(Kind::Matter)));
    fx.aperture_multiplier = 2.0;
    fx.clear();
    assert_eq!(shards(&fx.particles), 0);
    assert!(fx.requests.iter().all(Option::is_none));
    assert!((fx.aperture_multiplier - 1.0).abs() < f32::EPSILON);
}
#[test]
fn only_a_changed_gesture_emits_and_is_consumed_once() {
    let mut app = App::new();
    app.init_resource::<ToolFx>()
        .init_resource::<EditorHistory>()
        .add_systems(Update, finish_gesture);
    let revision = app.world().resource::<EditorHistory>().current_revision;
    app.world_mut().resource_mut::<ToolFx>().gesture = Some((revision, Vec3::Z));
    app.update();
    assert!(
        app.world()
            .resource::<ToolFx>()
            .requests
            .iter()
            .all(Option::is_none)
    );
    app.world_mut().resource_mut::<ToolFx>().gesture = Some((revision, Vec3::Z));
    app.world_mut()
        .resource_mut::<EditorHistory>()
        .current_revision += 1;
    app.update();
    app.update();
    assert_eq!(
        app.world()
            .resource::<ToolFx>()
            .requests
            .iter()
            .flatten()
            .count(),
        1
    );
}
#[test]
fn stationary_lifts_emit_nothing_and_upward_lifts_wake_downward() {
    let mut p = Particles::default();
    p.wake(Vec3::ZERO, Vec3::ZERO, 1.0, 0.05);
    assert_eq!(traces(&p), 0);
    p.wake(Vec3::ZERO, Vec3::Y * 0.25, 1.0, 0.05);
    assert_eq!(traces(&p), 6);
    assert!(p.traces.slots.iter().flatten().all(|s| s.b.y < s.a.y));
}

#[test]
fn releasing_a_hold_clears_violet_feedback_but_preserves_editing_feedback() {
    let mut particles = Particles::default();
    particles.request(
        Request::Freeze {
            center: Vec3::ZERO,
            radius: 1.0,
        },
        Vec3::ZERO,
    );
    particles.request(Request::Matter { hit: Vec3::Z }, Vec3::ZERO);
    particles.clear_freeze();
    assert_eq!(shards(&particles), 22);
    assert_eq!(traces(&particles), 10);
    assert!(
        particles
            .shards
            .slots
            .iter()
            .flatten()
            .all(|s| s.kind == Kind::Matter)
    );
}

fn place_blocks(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
) {
    crate::handle_block_actions(&actions, &mut graph.0, &mut state, &mut history);
}
#[test]
fn accepted_block_gesture_emits_once_and_rejected_placement_emits_nothing() {
    for valid in [false, true] {
        let graph = mechanic_core::ConstructionGraph::new();
        let hit = crate::SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = crate::candidate_from_hit(&graph, hit);
        let mut app = App::new();
        app.insert_resource(EditorGraph(graph))
            .insert_resource(EditorState {
                hovered: Some(hit),
                preview: valid.then_some(candidate),
                pointer_position: Some(Vec2::new(320.0, 240.0)),
                pointer_ray: Some((
                    Vec3::new(-0.3, 2.0, -0.2),
                    Vec3::new(0.2, -1.0, 0.3).normalize(),
                )),
                ..default()
            })
            .init_resource::<EditorHistory>()
            .init_resource::<ToolFx>()
            .init_resource::<ButtonInput<GameAction>>()
            .init_resource::<MaterialWheelState>()
            .init_resource::<crate::ui::UiInput>()
            .insert_resource(SelectedTool::from_editor_tool(crate::Tool::Block))
            .insert_resource(PlayerState {
                input_captured: true,
                ..default()
            })
            .init_resource::<DimensionFreeze>()
            .add_systems(
                Update,
                (capture_gesture, place_blocks, finish_gesture).chain(),
            );
        app.world_mut().spawn(Window {
            focused: true,
            ..default()
        });
        app.world_mut()
            .resource_mut::<ButtonInput<GameAction>>()
            .press(GameAction::Primary);
        app.update();
        assert!(
            app.world()
                .resource::<ToolFx>()
                .requests
                .iter()
                .all(Option::is_none)
        );
        {
            let mut actions = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            actions.clear();
            actions.release(GameAction::Primary);
        }
        app.update();
        app.update();
        assert_eq!(
            app.world()
                .resource::<ToolFx>()
                .requests
                .iter()
                .flatten()
                .count(),
            usize::from(valid)
        );
        assert_eq!(
            app.world().resource::<EditorGraph>().0.part_count(),
            usize::from(valid)
        );
    }
}

#[test]
fn connector_streams_follow_all_six_plate_tips_without_accumulating() {
    let mut p = Particles::default();
    for step in 0_u16..100 {
        let mut f = frame(Kind::Connector);
        f.target = Vec3::new(f32::from(step), 2.0, -3.0);
        f.connector_plates = std::array::from_fn(|index| {
            let angle = f32::from(step) * 0.1 + [0.0, 1.0, 2.0, 3.0, 4.0, 5.0][index];
            Some(Vec3::new(angle.cos(), angle.sin(), 0.5))
        });
        p.advance(1.0 / 60.0, Some(f));
        assert_eq!(traces(&p), 81);
        for origin in f.connector_plates.into_iter().flatten() {
            assert!(p.traces.slots.iter().flatten().any(|s| s.a == origin));
        }
        assert_eq!(
            p.traces
                .slots
                .iter()
                .flatten()
                .filter(|s| s.b.distance(f.target) < 1.0e-5)
                .count(),
            7
        );
    }
    p.advance(1.0 / 60.0, None);
    assert_eq!(traces(&p), 0);
}

#[test]
fn plate_emitters_use_live_transforms_of_only_the_active_end() {
    let mut app = App::new();
    app.init_resource::<ToolEmitter>()
        .add_systems(Update, collect_plate_emitters);
    let mut tips = Vec::new();
    for end in 0_u8..2 {
        for panel in 0_u8..6 {
            tips.push(
                app.world_mut()
                    .spawn((
                        ConnectorPlateEmitter {
                            end: usize::from(end),
                            panel: usize::from(panel),
                        },
                        GlobalTransform::from_translation(Vec3::new(
                            f32::from(panel),
                            f32::from(end),
                            0.0,
                        )),
                    ))
                    .id(),
            );
        }
    }
    app.update();
    assert!(
        app.world()
            .resource::<ToolEmitter>()
            .connector_plates
            .iter()
            .flatten()
            .all(|p| p.y == 0.0)
    );
    app.world_mut().resource_mut::<ToolEmitter>().active_end = 1;
    app.world_mut()
        .entity_mut(tips[6])
        .insert(GlobalTransform::from_translation(Vec3::splat(9.0)));
    app.update();
    assert_eq!(
        app.world().resource::<ToolEmitter>().connector_plates[0],
        Some(Vec3::splat(9.0))
    );
    app.world_mut().despawn(tips[6]);
    app.update();
    assert_eq!(
        app.world().resource::<ToolEmitter>().connector_plates[0],
        None
    );
}
