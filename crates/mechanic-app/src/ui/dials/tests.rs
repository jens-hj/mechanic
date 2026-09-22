use super::*;
use bevy::math::{IVec3, Vec3};
use mechanic_core::{
    AnalogMapping, AnalogRange, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
    ConstructionGraph, ControllerSpec, CuboidSpec, DialSpec, DriveLinkId, DriveLinkSpec,
    DriveProgram, DriveState, DriveTarget, FaceKind, FaceRef, GridRotation, InputConfiguration,
    InputSize,
};

fn spawn(graph: &mut ConstructionGraph, command: BuildCommand) -> PartId {
    let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
        panic!("expected a part");
    };
    part
}

fn wired() -> (
    ConstructionGraph,
    crate::control_panel::ControlPanelState,
    DriveLinkId,
) {
    let mut graph = ConstructionGraph::new();
    let mut block = |y| {
        spawn(
            &mut graph,
            BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::new(0, y, 0), GridRotation::default()),
                )
                .unwrap(),
            ),
        )
    };
    let base = block(1);
    let rotor = block(3);
    let controller = spawn(
        &mut graph,
        BuildCommand::SpawnController(ControllerSpec::new(BuildPose::new(
            IVec3::new(20, 0, 0),
            GridRotation::default(),
        ))),
    );
    let BuildOutcome::BearingAdded(bearing) = graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(rotor, FaceKind::NegativeY),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::Y,
        )))
        .unwrap()
    else {
        panic!("expected bearing")
    };
    let mut spec = DriveLinkSpec::new(controller, bearing);
    spec.program = DriveProgram::new(
        &[
            DriveState::new(DriveTarget::Speed(1.0)).unwrap(),
            DriveState::new(DriveTarget::Angle(0.5)).unwrap(),
        ],
        false,
    )
    .unwrap();
    let BuildOutcome::DriveLinked(link) = graph.apply(BuildCommand::AddDriveLink(spec)).unwrap()
    else {
        panic!("expected link")
    };
    let mut panel = crate::control_panel::ControlPanelState::default();
    panel.open(controller);
    (graph, panel, link)
}

#[test]
fn snapshot_uses_selected_angular_units_and_legal_native_bounds() {
    let (graph, mut panel, link) = wired();
    let speed = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let angle = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularPosition(1),
    };
    let rpm = capture(&graph, &panel, None);
    let field = rpm
        .fields
        .iter()
        .find(|field| field.draft.targets.contains(&speed))
        .unwrap();
    assert_eq!(field.draft.unit, "RPM");
    assert!((field.draft.current - mechanic_core::rad_s_to_rpm(1.0)).abs() < 0.001);
    let bounds = graph.numeric_metadata(speed).unwrap();
    assert!((field.draft.minimum - bounds.minimum * field.draft.factor).abs() < 0.001);
    assert!((field.draft.maximum - bounds.maximum * field.draft.factor).abs() < 0.001);
    assert!(rpm.fields.iter().all(|field| !field.draft.targets.contains(
        &NumericParameter::Drive {
            link,
            parameter: DriveParameter::Dwell(0),
        }
    )));
    panel.toggle_speed_unit();
    let degrees = capture(&graph, &panel, None);
    let field = degrees
        .fields
        .iter()
        .find(|field| field.draft.targets.contains(&speed))
        .unwrap();
    assert_eq!(field.draft.unit, "°/s");
    assert!((field.draft.current - 1.0_f32.to_degrees()).abs() < 0.001);
    let field = degrees
        .fields
        .iter()
        .find(|field| field.draft.targets.contains(&angle))
        .unwrap();
    assert_eq!(field.draft.unit, "°");
    assert!((field.draft.current - 0.5_f32.to_degrees()).abs() < 0.001);
    assert_eq!(graph.numeric_value(speed), Some(1.0));
}

#[test]
fn assigned_names_and_mixed_feedback_follow_current_values_without_writes() {
    let (mut graph, panel, link) = wired();
    let dial = spawn(
        &mut graph,
        BuildCommand::SpawnDial(DialSpec::new(InputSize::Panel, BuildPose::default())),
    );
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                name: "Throttle".into(),
                controller: panel.controller(),
                ..Default::default()
            },
        })
        .unwrap();
    let speed = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let angle = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularPosition(1),
    };
    graph
        .assign_dial(
            dial,
            &[
                AnalogMapping {
                    target: speed,
                    range: AnalogRange::new([0.0, 2.0], false).unwrap(),
                },
                AnalogMapping {
                    target: angle,
                    range: AnalogRange::new([0.0, 2.0], false).unwrap(),
                },
            ],
        )
        .unwrap();
    let model = capture(&graph, &panel, None);
    assert_eq!(model.dials.len(), 1);
    assert_eq!(model.dials[0].name, "Throttle");
    assert_eq!(model.dials[0].position, "Mixed");
    assert_eq!(model.dials[0].count, 2);
    for target in [speed, angle] {
        let field = model
            .fields
            .iter()
            .find(|field| field.draft.targets.contains(&target))
            .unwrap();
        assert_eq!(field.badge, "Throttle");
        assert_eq!(field.draft.dial, Some(dial));
    }
    assert_eq!(graph.numeric_value(speed), Some(1.0));
    assert_eq!(graph.numeric_value(angle), Some(0.5));
    graph.unlink_dial_targets(&[angle]);
    let model = capture(&graph, &panel, None);
    assert_eq!(model.dials[0].position, "50%");
    assert_eq!(model.dials[0].count, 1);
    assert_eq!(graph.numeric_value(angle), Some(0.5));
}

fn draft() -> Draft {
    Draft {
        targets: Vec::new(),
        name: "State 1 target".into(),
        current: 12.0,
        dial: None,
        minimum: 0.0,
        maximum: 100.0,
        reverse: false,
        factor: 1.0,
        unit: "RPM".into(),
        integer: false,
        error: None,
    }
}

#[test]
fn endpoints_reject_nonfinite_and_fractional_integer_values() {
    let mut draft = draft();
    draft.integer = true;
    for value in ["NaN", "inf", "1.5", ""] {
        draft.error = None;
        edit_endpoint(&mut draft, value, true);
        assert!(draft.error.is_some());
        assert!(draft.minimum.abs() < f32::EPSILON);
    }
    draft.error = None;
    edit_endpoint(&mut draft, "2", true);
    edit_endpoint(&mut draft, "8", false);
    assert!(draft.error.is_none());
    assert_eq!((draft.minimum, draft.maximum), (2.0, 8.0));
}

#[test]
fn assignment_editor_is_compact_and_its_rows_do_not_overlap() {
    let (graph, panel, _) = wired();
    let overlay = crate::ui::testing::Overlay::mount();
    overlay
        .handles
        .block
        .model
        .set(crate::ui::control_block::capture(
            &panel,
            &graph,
            &crate::sequencer::GearboxRuntime::default(),
            false,
        ));
    let mut model = capture(&graph, &panel, None);
    model.fields[0].badge = "Dial 1".into();
    model.draft = Some(model.fields[0].draft.clone());
    overlay.handles.block.dials.set(model);
    overlay.settle();
    let boxes = overlay.rects();
    let header = boxes
        .iter()
        .find(|(_, rect)| {
            (rect.size.width - 100.0).abs() < 0.5 && (rect.size.height - 28.0).abs() < 0.5
        })
        .expect("compact Dials button")
        .1;
    let (index, (depth, editor)) = boxes
        .iter()
        .enumerate()
        .find(|(_, (_, rect))| (rect.size.width - 640.0).abs() < 0.5 && rect.size.height > 100.0)
        .expect("expanded assignment editor");
    assert!(
        editor.size.height < 320.0,
        "editor height: {}",
        editor.size.height
    );
    assert!(editor.origin.y >= header.origin.y + header.size.height);
    let children = boxes[index + 1..]
        .iter()
        .take_while(|(level, _)| level > depth)
        .filter(|(level, rect)| *level == depth + 1 && rect.size.height > 0.5)
        .map(|(_, rect)| rect)
        .collect::<Vec<_>>();
    assert!(
        children.len() >= 4,
        "editor must expose separate content rows"
    );
    assert!(
        children.iter().any(|rect| rect.size.width > 150.0
            && rect.size.width < 300.0
            && (rect.size.height - 28.0).abs() < 0.5),
        "Reverse direction must fit on one line"
    );
    let (badge_index, (badge_depth, badge)) = boxes
        .iter()
        .enumerate()
        .find(|(_, (_, rect))| {
            (rect.size.height - 20.0).abs() < 0.5
                && rect.size.width > 35.0
                && rect.size.width < 70.0
        })
        .expect("compact Dial 1 badge");
    let label = boxes[badge_index + 1..]
        .iter()
        .take_while(|(depth, _)| depth > badge_depth)
        .find(|(_, rect)| (rect.size.height - 14.0).abs() < 0.5)
        .expect("badge label")
        .1;
    assert!(
        label.origin.y >= badge.origin.y
            && label.origin.y + label.size.height <= badge.origin.y + badge.size.height + 0.5,
        "badge label must stay vertically inside its outline"
    );
    for pair in children.windows(2) {
        assert!(
            pair[1].origin.y + 0.5 >= pair[0].origin.y + pair[0].size.height,
            "assignment rows overlap: {pair:?}"
        );
    }
    for child in children {
        assert!(
            child.origin.y + child.size.height <= editor.origin.y + editor.size.height + 0.5,
            "editor content exceeds its panel"
        );
    }
}

#[test]
fn empty_editor_mounts_without_dials_or_graph_references() {
    let overlay = crate::ui::testing::Overlay::mount();
    overlay
        .handles
        .block
        .model
        .set(crate::ui::control_block::PanelModel {
            open: true,
            ..Default::default()
        });
    overlay.handles.block.dials.set(Model {
        draft: Some(draft()),
        ..Model::default()
    });
    overlay.settle();
    assert!(overlay.handles.block.dials.get_untracked().dials.is_empty());
    assert!(overlay.handles.block.dial_intents.borrow().is_empty());
}

#[test]
fn opening_a_draft_populates_endpoints_without_replacing_an_active_edit() {
    let overlay = crate::ui::testing::Overlay::mount();
    let handles = &overlay.handles.block;
    let draft = draft();
    sync_opened_draft(handles, Some(&draft));
    assert_eq!(
        handles.dial_minimum.get_untracked(),
        draft.minimum.to_string()
    );
    assert_eq!(
        handles.dial_maximum.get_untracked(),
        draft.maximum.to_string()
    );
    handles.dials.set(Model {
        draft: Some(draft.clone()),
        ..Model::default()
    });
    handles.dial_minimum.set("-1.".into());
    sync_opened_draft(handles, Some(&draft));
    assert_eq!(handles.dial_minimum.get_untracked(), "-1.");
}

#[test]
fn opening_an_unbound_field_selects_the_only_connected_dial_without_binding_it() {
    let (mut graph, panel, link) = wired();
    let dial = spawn(
        &mut graph,
        BuildCommand::SpawnDial(DialSpec::new(InputSize::Panel, BuildPose::default())),
    );
    graph
        .apply(BuildCommand::SetInputConfiguration {
            input: dial,
            configuration: InputConfiguration {
                controller: panel.controller(),
                name: "motor".into(),
                ..InputConfiguration::default()
            },
        })
        .unwrap();
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let model = capture(&graph, &panel, None);
    let draft = draft_for(&model, target).unwrap();
    assert_eq!(draft.dial, Some(dial));
    assert!(graph.input_configuration(dial).unwrap().analog.is_empty());
    assert_eq!(graph.numeric_value(target), Some(1.0));

    let mut multiple = model;
    multiple.dials.push(Dial {
        id: panel.controller().unwrap(),
        name: "Another dial".into(),
        position: "Mixed".into(),
        count: 0,
    });
    assert_eq!(draft_for(&multiple, target).unwrap().dial, None);
    multiple.fields[0].draft.dial = Some(dial);
    assert_eq!(draft_for(&multiple, target).unwrap().dial, Some(dial));

    let mut assignments = crate::dial_assignment::DialAssignments {
        draft: Some(draft),
        ..crate::dial_assignment::DialAssignments::default()
    };
    let mut graph = crate::editor::state::EditorGraph(graph);
    crate::dial_assignment::apply(
        &mut assignments,
        &mut graph,
        &mut crate::editor::state::EditorState::default(),
        &mut crate::editor::history::EditorHistory::default(),
        false,
    );
    assert!(assignments.draft.is_none(), "{:?}", assignments.draft);
    assert_eq!(graph.0.input_configuration(dial).unwrap().analog.len(), 1);
    assert_eq!(graph.0.numeric_value(target), Some(1.0));
}

#[test]
fn applying_without_a_selected_dial_explains_selection_instead_of_a_missing_target() {
    let (graph, panel, link) = wired();
    let target = NumericParameter::Drive {
        link,
        parameter: DriveParameter::AngularSpeed(0),
    };
    let mut assignments = crate::dial_assignment::DialAssignments {
        draft: draft_for(&capture(&graph, &panel, None), target),
        ..crate::dial_assignment::DialAssignments::default()
    };
    let mut graph = crate::editor::state::EditorGraph(graph);
    crate::dial_assignment::apply(
        &mut assignments,
        &mut graph,
        &mut crate::editor::state::EditorState::default(),
        &mut crate::editor::history::EditorHistory::default(),
        false,
    );
    assert_eq!(
        assignments.draft.unwrap().error.as_deref(),
        Some("Select a connected dial before applying.")
    );
    assert_eq!(graph.0.numeric_value(target), Some(1.0));
}

#[test]
fn assignment_editor_explains_selection_and_blocks_apply_until_a_dial_is_chosen() {
    let (graph, panel, _) = wired();
    let motor = panel.controller().unwrap();
    let other = graph.parts().find(|(id, _)| *id != motor).unwrap().0;
    let overlay = crate::ui::testing::Overlay::mount();
    overlay
        .handles
        .block
        .model
        .set(crate::ui::control_block::PanelModel {
            open: true,
            ..crate::ui::control_block::PanelModel::default()
        });
    let mut model = Model {
        draft: Some(draft()),
        dials: [(motor, "motor"), (other, "steering")]
            .into_iter()
            .map(|(id, name)| Dial {
                id,
                name: name.into(),
                position: "Mixed".into(),
                count: 0,
            })
            .collect(),
        ..Model::default()
    };
    overlay.handles.block.dials.set(model.clone());
    overlay.settle();
    assert!(overlay.labels().contains(&"Dial to assign".into()));
    assert!(
        overlay
            .labels()
            .contains(&"Choose a dial below to enable Apply.".into())
    );
    let apply = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| {
            (rect.size.width - 76.0).abs() < 0.5 && (rect.size.height - 28.0).abs() < 0.5
        })
        .min_by(|a, b| a.origin.x.total_cmp(&b.origin.x))
        .unwrap();
    overlay.click(Vector2::new(
        apply.origin.x + apply.size.width * 0.5,
        apply.origin.y + apply.size.height * 0.5,
    ));
    assert!(overlay.handles.block.dial_intents.borrow().is_empty());
    model.draft.as_mut().unwrap().dial = Some(motor);
    overlay.handles.block.dials.set(model);
    overlay.settle();
    assert!(overlay.labels().contains(&"● motor · Selected".into()));
    assert!(
        !overlay
            .labels()
            .contains(&"Choose a dial below to enable Apply.".into())
    );
    let apply = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| {
            (rect.size.width - 76.0).abs() < 0.5 && (rect.size.height - 28.0).abs() < 0.5
        })
        .min_by(|a, b| a.origin.x.total_cmp(&b.origin.x))
        .unwrap();
    overlay.click(Vector2::new(
        apply.origin.x + apply.size.width * 0.5,
        apply.origin.y + apply.size.height * 0.5,
    ));
    assert!(matches!(
        overlay.handles.block.dial_intents.borrow().as_slice(),
        [Intent::Apply(_, _)]
    ));
}
