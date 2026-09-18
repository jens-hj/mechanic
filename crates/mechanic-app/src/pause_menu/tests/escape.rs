use crate::camera::set_projection_fov;
use crate::pause_menu::EscapeTarget;
use crate::pause_menu::ExitDisposition;
use crate::pause_menu::escape_target;
use crate::pause_menu::exit_disposition;
use crate::*;

#[test]
fn existing_escape_owners_take_priority_before_pause() {
    assert_eq!(
        escape_target(false, false, true, true, true),
        EscapeTarget::ExistingUi
    );
    assert_eq!(
        escape_target(false, false, false, true, true),
        EscapeTarget::ControlPanel
    );
    assert_eq!(
        escape_target(false, false, false, false, true),
        EscapeTarget::WorldState
    );
    assert_eq!(
        escape_target(false, false, false, false, false),
        EscapeTarget::OpenPause
    );
}

#[test]
fn pause_confirmation_cancels_before_pause_continues() {
    assert_eq!(
        escape_target(true, true, true, true, true),
        EscapeTarget::PauseSubmenu
    );
    assert_eq!(
        escape_target(true, false, true, true, true),
        EscapeTarget::PauseMenu
    );
}

#[test]
fn revisions_restore_clean_identity_through_undo_and_redo() {
    let graph = ConstructionGraph::default();
    let state = EditorState::default();
    let mut history = EditorHistory::default();
    assert!(
        !history.is_dirty(),
        "the initial blank construction is clean"
    );

    history.commit(EditorSnapshot::capture(&graph, &state));
    assert!(history.is_dirty(), "a successful edit creates a revision");
    history.mark_clean();
    assert!(
        !history.is_dirty(),
        "a successful save marks that revision clean"
    );

    history.commit(EditorSnapshot::capture(&graph, &state));
    assert!(history.is_dirty(), "a later edit is dirty");
    history
        .undo(EditorSnapshot::capture(&graph, &state))
        .expect("undo reaches the saved revision");
    assert!(!history.is_dirty());
    history
        .redo(EditorSnapshot::capture(&graph, &state))
        .expect("redo reaches the edited revision");
    assert!(history.is_dirty());
}

#[test]
fn clean_exit_is_immediate_and_dirty_exit_requires_confirmation() {
    assert_eq!(exit_disposition(false), ExitDisposition::Exit);
    assert_eq!(exit_disposition(true), ExitDisposition::ConfirmUnsaved);
}

#[test]
fn both_camera_projections_receive_the_same_fov() {
    let mut projections = [
        Projection::Perspective(PerspectiveProjection::default()),
        Projection::Perspective(PerspectiveProjection::default()),
    ];
    let wanted = 80.0_f32.to_radians();
    for projection in &mut projections {
        set_projection_fov(projection, wanted);
    }
    for projection in projections {
        let Projection::Perspective(perspective) = projection else {
            panic!("test projection remains perspective");
        };
        assert!((perspective.fov - wanted).abs() < f32::EPSILON);
    }
}
