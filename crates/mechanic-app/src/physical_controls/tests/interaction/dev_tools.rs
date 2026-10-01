//! Spectator input isolation while vehicle simulation stays live.

use super::*;

#[test]
fn dev_spectator_releases_momentary_input_but_preserves_toggle_latches() {
    for toggle in [false, true] {
        let (mut app, controller, _) = fixture(false, false);
        if toggle {
            let mut graph = app.world_mut().resource_mut::<EditorGraph>();
            let button = graph
                .0
                .parts()
                .find_map(|(id, part)| {
                    matches!(part, mechanic_core::PartSpec::Button(_)).then_some(id)
                })
                .unwrap();
            let mut configuration = graph.0.input_configuration(button).unwrap().clone();
            configuration.button_mode = mechanic_core::ButtonMode::Toggle;
            graph
                .0
                .apply(BuildCommand::SetInputConfiguration {
                    input: button,
                    configuration,
                })
                .unwrap();
        }
        press(&mut app);
        assert!(held(&app, controller));
        let mut dev = crate::dev_tools::DevTools::default();
        dev.enabled = true;
        dev.mode = crate::dev_tools::DevMode::Spectator;
        app.insert_resource(dev);
        app.update();
        assert_eq!(held(&app, controller), toggle);
        assert!(app.world().resource::<AppSimulation>().is_running());
    }
}

#[test]
fn dev_spectator_keeps_seat_updates_from_overwriting_the_detached_camera() {
    let (mut app, _, _) = fixture(false, true);
    let seat = app.world().resource::<PlayerState>().seat;
    let mut dev = crate::dev_tools::DevTools::default();
    dev.enabled = true;
    dev.mode = crate::dev_tools::DevMode::Spectator;
    app.insert_resource(dev);
    let detached = Transform::from_xyz(100.0, 30.0, -20.0);
    let entity = app
        .world_mut()
        .query_filtered::<Entity, With<MainCamera>>()
        .single(app.world())
        .unwrap();
    app.world_mut()
        .entity_mut(entity)
        .insert((detached, GlobalTransform::from(detached)));
    press(&mut app);
    assert_eq!(
        *app.world().entity(entity).get::<Transform>().unwrap(),
        detached
    );
    assert_eq!(app.world().resource::<PlayerState>().seat, seat);
    app.world_mut()
        .resource_mut::<crate::dev_tools::DevTools>()
        .mode = crate::dev_tools::DevMode::Normal;
    app.world_mut()
        .resource_mut::<ButtonInput<GameAction>>()
        .reset_all();
    app.update();
    assert_ne!(
        *app.world().entity(entity).get::<Transform>().unwrap(),
        detached
    );
    assert_eq!(app.world().resource::<PlayerState>().seat, seat);
}
