use bevy_math::{IVec3, Vec3};

use super::{
    BuildPose, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensionError,
    CylinderDimensions, CylinderSpec, DimensionLinkId, DimensionLinkSpec, EngineKind, EngineSpec,
    FaceKind, GridRotation, InputSpec, LayerError, LayerFace, PartSpec, PipeBendDimensionError,
    PipeBendDimensions, PipeBendSpec, SeatSpec, ServoSpec, cuboid_face, pipe_bend_face,
    snap_world_to_grid,
};

#[test]
fn snaps_world_coordinates_to_quarter_metre_grid() {
    assert_eq!(
        snap_world_to_grid(Vec3::new(0.37, -0.13, 1.99)),
        IVec3::new(1, -1, 8)
    );
}

#[test]
fn dimension_link_has_a_fixed_two_by_one_by_one_block_envelope() {
    let pose = BuildPose::from_position_ticks(IVec3::new(7, 200, -3), GridRotation::new(0, 1, 0));
    let link = DimensionLinkSpec::new(DimensionLinkId(42), pose);
    assert_eq!(link.id, DimensionLinkId(42));
    assert_eq!(
        link.cuboid().dimensions.map(super::GridDimension::units),
        [2, 1, 1]
    );
    assert_eq!(link.cuboid().pose, pose);
    assert!(
        link.cuboid()
            .size_meters()
            .abs_diff_eq(Vec3::new(0.5, 0.25, 0.25), 1.0e-6)
    );
}

#[test]
fn half_grid_pose_preserves_eighth_metre_centres_in_both_directions() {
    let pose = BuildPose::from_half_grid(IVec3::new(-3, 1, 4), GridRotation::default());

    assert_eq!(pose.translation_half_units(), IVec3::new(-3, 1, 4));
    assert!(
        pose.translation()
            .abs_diff_eq(Vec3::new(-0.375, 0.125, 0.5), 1.0e-6)
    );
    assert_eq!(
        BuildPose::new(IVec3::new(-3, 1, 4), GridRotation::default()).translation_half_units(),
        IVec3::new(-6, 2, 8)
    );
}

#[test]
fn pose_preserves_fine_position_ticks_in_both_directions() {
    let pose =
        BuildPose::from_position_ticks(IVec3::new(-130, 20, 190), GridRotation::new(1, 2, 3));
    assert_eq!(pose.translation_position_ticks(), IVec3::new(-130, 20, 190));
    assert!(
        pose.translation()
            .abs_diff_eq(Vec3::new(-0.325, 0.05, 0.475), 1.0e-6)
    );
}

#[test]
fn one_centimetre_positions_round_trip_exactly_across_zero() {
    for ticks in [IVec3::new(4, -4, 8), IVec3::new(-4, 4, -8)] {
        let pose = BuildPose::from_position_ticks(ticks, GridRotation::default());
        assert_eq!(pose.translation_position_ticks(), ticks);
        assert!(
            pose.translation()
                .abs_diff_eq(ticks.as_vec3() * 0.0025, 1.0e-7)
        );
    }
}

#[test]
fn rejects_dimensions_outside_builder_range() {
    assert!(CuboidSpec::new([0, 4, 4], BuildPose::default()).is_err());
    assert!(CuboidSpec::new([33, 4, 4], BuildPose::default()).is_err());
    assert!(CuboidSpec::new([1, 32, 1], BuildPose::default()).is_ok());
}

#[test]
fn construction_material_property_rows_are_exact() {
    let expected: [(ConstructionMaterial, [f32; 6]); 12] = [
        (
            ConstructionMaterial::Aluminium,
            [2_700.0, 0.61, 0.47, 0.25, 0.004, 69.0e9],
        ),
        (
            ConstructionMaterial::CarbonFiber,
            [1_600.0, 0.40, 0.30, 0.20, 0.008, 70.0e9],
        ),
        (
            ConstructionMaterial::Concrete,
            [2_400.0, 0.80, 0.65, 0.05, 0.020, 30.0e9],
        ),
        (
            ConstructionMaterial::Dirt,
            [1_600.0, 0.72, 0.55, 0.05, 0.030, 0.05e9],
        ),
        (
            ConstructionMaterial::Graphite,
            [1_900.0, 0.25, 0.15, 0.10, 0.010, 12.0e9],
        ),
        (
            ConstructionMaterial::Iron,
            [7_870.0, 0.70, 0.55, 0.15, 0.003, 170.0e9],
        ),
        (
            ConstructionMaterial::Plastic,
            [950.0, 0.40, 0.30, 0.40, 0.020, 1.0e9],
        ),
        (
            ConstructionMaterial::Rubber,
            [1_100.0, 1.00, 0.80, 0.70, 0.010, 0.01e9],
        ),
        (
            ConstructionMaterial::Sand,
            [1_700.0, 0.65, 0.50, 0.05, 0.035, 0.03e9],
        ),
        (
            ConstructionMaterial::Steel,
            [7_850.0, 0.74, 0.57, 0.20, 0.002, 200.0e9],
        ),
        (
            ConstructionMaterial::Stone,
            [2_700.0, 0.60, 0.48, 0.05, 0.015, 50.0e9],
        ),
        (
            ConstructionMaterial::Wood,
            [700.0, 0.48, 0.30, 0.15, 0.025, 10.0e9],
        ),
    ];
    for (material, expected) in expected {
        let properties = material.properties();
        let actual = [
            properties.density_kg_m3,
            properties.static_friction,
            properties.dynamic_friction,
            properties.restitution,
            properties.rolling_resistance,
            properties.youngs_modulus_pa,
        ];
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        assert!(properties.density_kg_m3 > 0.0);
        assert!(properties.static_friction >= properties.dynamic_friction);
        assert!((0.0..=1.0).contains(&properties.dynamic_friction));
        assert!((0.0..=1.0).contains(&properties.restitution));
        assert!((0.0..=1.0).contains(&properties.rolling_resistance));
        assert!(properties.youngs_modulus_pa > 0.0);
    }
}

#[test]
fn nominal_block_compliance_uses_the_construction_scale() {
    let steel = ConstructionMaterial::Steel.properties();
    assert_eq!(
        steel.nominal_block_compliance().to_bits(),
        (1.0 / (200.0e9 * super::GRID_UNIT_METERS)).to_bits(),
    );
    assert!(
        ConstructionMaterial::Rubber
            .properties()
            .nominal_block_compliance()
            > steel.nominal_block_compliance()
    );
}

#[test]
fn ordinary_part_specs_default_to_steel_and_accept_an_explicit_material() {
    let cuboid = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
    let cylinder = CylinderSpec::new(CylinderDimensions::default(), BuildPose::default());
    assert_eq!(cuboid.material, ConstructionMaterial::Steel);
    assert_eq!(cylinder.material, ConstructionMaterial::Steel);
    assert_eq!(
        cuboid.with_material(ConstructionMaterial::Wood).material,
        ConstructionMaterial::Wood,
    );
    assert_eq!(
        cylinder
            .with_material(ConstructionMaterial::Plastic)
            .material,
        ConstructionMaterial::Plastic,
    );
}

#[test]
fn authored_parts_keep_their_fixed_grid_envelopes() {
    assert_eq!(
        ControllerSpec::new(BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [2, 2, 1]
    );
    assert_eq!(
        EngineSpec::new(EngineKind::Gas, BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [2, 2, 3]
    );
    assert_eq!(
        EngineSpec::new(EngineKind::Electric, BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [2, 2, 2]
    );
    assert_eq!(
        ServoSpec::new(BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [1, 1, 1]
    );
    assert_eq!(
        SeatSpec::new(BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [2, 1, 2]
    );
    assert_eq!(
        InputSpec::new(BuildPose::default())
            .cuboid()
            .dimensions
            .map(super::GridDimension::units),
        [2, 1, 1]
    );
    for (actual, expected) in [
        (EngineKind::Electric.stall_torque_newton_meters(), 500.0),
        (EngineKind::Electric.no_load_rpm(), 120.0),
        (EngineKind::Gas.stall_torque_newton_meters(), 6_000.0),
        (EngineKind::Gas.no_load_rpm(), 360.0),
        (ServoSpec::STALL_TORQUE_NEWTON_METERS, 12_000.0),
        (ServoSpec::NO_LOAD_RPM, 30.0),
    ] {
        assert!((actual - expected).abs() < f32::EPSILON);
    }
}

#[test]
fn cylinder_dimensions_validate_defaults_bounds_wall_and_length_grid() {
    let dimensions = CylinderDimensions::default();
    assert!((dimensions.outer_diameter() - 0.25).abs() < f32::EPSILON);
    assert!(dimensions.inner_diameter().abs() < f32::EPSILON);
    assert!((dimensions.axial_length() - 0.25).abs() < f32::EPSILON);
    assert_eq!(dimensions.sweep_angle_degrees(), 360);
    assert!(CylinderDimensions::new(0.05, 0.0, 0.25).is_ok());
    assert!(CylinderDimensions::new(8.0, 7.95, 8.0).is_ok());
    assert_eq!(
        CylinderDimensions::new(0.049, 0.0, 0.25),
        Err(CylinderDimensionError::OuterDiameterOutOfRange)
    );
    assert_eq!(
        CylinderDimensions::new(0.25, 0.201, 0.25),
        Err(CylinderDimensionError::InnerDiameterOutOfRange)
    );
    assert_eq!(
        CylinderDimensions::new(0.25, 0.0, 0.30),
        Err(CylinderDimensionError::AxialLengthOutOfRange)
    );
    assert_eq!(
        CylinderDimensions::new(0.25, 0.0, 8.25),
        Err(CylinderDimensionError::AxialLengthOutOfRange)
    );
    assert!(
        dimensions
            .with_sweep_angle_degrees(15)
            .is_ok_and(|dimensions| dimensions.sweep_angle_degrees() == 15)
    );
    assert_eq!(
        dimensions.with_sweep_angle_degrees(14),
        Err(CylinderDimensionError::SweepAngleOutOfRange)
    );
    assert_eq!(
        dimensions.with_sweep_angle_degrees(361),
        Err(CylinderDimensionError::SweepAngleOutOfRange)
    );
}

#[test]
fn rotates_face_orientation_in_quarter_turns() {
    let spec = CuboidSpec::new(
        [4, 8, 4],
        BuildPose::new(IVec3::ZERO, GridRotation::new(0, 0, 1)),
    )
    .unwrap();
    let face = cuboid_face(spec, FaceKind::PositiveX);

    assert!(face.normal.abs_diff_eq(Vec3::Y, 1.0e-6));
    assert!(face.center.abs_diff_eq(Vec3::new(0.0, 0.5, 0.0), 1.0e-6));
}

#[test]
fn pipe_bend_one_block_pipe_bends_inside_one_block() {
    let bend = PipeBendDimensions::new(0.25, 0.10, 1).unwrap();
    assert!((bend.radius() - 0.125).abs() < 1.0e-6);
    assert!((bend.radius() + bend.outer_diameter() * 0.5 - 0.25).abs() < 1.0e-6);
    let larger = PipeBendDimensions::new(0.20, 0.0, 3).unwrap();
    assert!((larger.radius() - 0.625).abs() < 1.0e-6);
}

#[test]
fn wide_pipe_minimum_span_steps_up_by_channel_width() {
    for (outer, span) in [(0.05, 1), (0.25, 1), (0.30, 2), (0.50, 2), (0.60, 3)] {
        assert_eq!(PipeBendDimensions::minimum_span(outer), span, "OD {outer}");
    }
    let wide = PipeBendDimensions::new(0.50, 0.0, 2).unwrap();
    assert!((wide.radius() - 0.25).abs() < 1.0e-6);
}

#[test]
fn span_below_channel_width_or_out_of_range_is_rejected() {
    assert_eq!(
        PipeBendDimensions::new(0.30, 0.10, 1),
        Err(PipeBendDimensionError::SpanTooSmallForDiameter)
    );
    assert_eq!(
        PipeBendDimensions::new(0.25, 0.10, 0),
        Err(PipeBendDimensionError::SpanOutOfRange)
    );
    assert_eq!(
        PipeBendDimensions::new(0.25, 0.21, 1),
        Err(PipeBendDimensionError::InnerDiameterOutOfRange)
    );
}

#[test]
fn pipe_bend_exposes_only_its_two_tangent_annular_ends() {
    let spec = PipeBendSpec::new(
        PipeBendDimensions::new(0.25, 0.10, 2).unwrap(),
        BuildPose::new(IVec3::new(4, 8, 0), GridRotation::new(0, 0, 1)),
    );
    let inlet = pipe_bend_face(spec, FaceKind::NegativeX).unwrap();
    let outlet = pipe_bend_face(spec, FaceKind::PositiveY).unwrap();
    assert!(inlet.normal.abs_diff_eq(Vec3::NEG_Y, 1.0e-6));
    assert!(outlet.normal.abs_diff_eq(Vec3::NEG_X, 1.0e-6));
    assert!(pipe_bend_face(spec, FaceKind::PositiveZ).is_none());
    assert!(((inlet.center - spec.pose.translation()).length() - 0.375).abs() < 1.0e-5);
    assert!(((outlet.center - spec.pose.translation()).length() - 0.375).abs() < 1.0e-5);
}

fn layer_core(outer: f32, inner: f32) -> PartSpec {
    PartSpec::Cylinder(CylinderSpec::new(
        CylinderDimensions::new(outer, inner, 0.5).unwrap(),
        BuildPose::default(),
    ))
}

fn rubber_layer(spec: PartSpec, face: LayerFace, thickness: f32) -> Result<PartSpec, LayerError> {
    spec.with_layer(
        face,
        thickness,
        ConstructionMaterial::Rubber,
        crate::MaterialAppearance::BAKED,
    )
}

#[test]
fn outer_layer_grows_the_envelope_and_keeps_the_core_band() {
    let layered = rubber_layer(layer_core(1.0, 0.0), LayerFace::OuterWall, 0.25).unwrap();
    let cylinder = layered.as_cylinder().unwrap();
    assert!((cylinder.dimensions.outer_diameter() - 1.5).abs() < 1.0e-5);
    assert_eq!(cylinder.material, ConstructionMaterial::Steel);
    assert_eq!(
        cylinder.outer_contact_material(),
        ConstructionMaterial::Rubber
    );
    assert_eq!(layered.band_at_local_point(Vec3::new(0.4, 0.0, 0.0)), 0);
    assert_eq!(layered.band_at_local_point(Vec3::new(0.6, 0.0, 0.0)), 1);
    assert!(layered.shares_core_with(layer_core(1.0, 0.0)));
}

#[test]
fn bore_layer_can_fill_to_a_solid_core() {
    let pipe = layer_core(1.0, 0.5);
    let lined = rubber_layer(pipe, LayerFace::Bore, 0.1).unwrap();
    let dimensions = lined.as_cylinder().unwrap().dimensions;
    assert!((dimensions.inner_diameter() - 0.3).abs() < 1.0e-5);
    assert_eq!(lined.band_at_local_point(Vec3::new(0.2, 0.0, 0.0)), 1);
    assert_eq!(lined.band_at_local_point(Vec3::new(0.3, 0.0, 0.0)), 0);
    let cored = rubber_layer(pipe, LayerFace::Bore, 0.25).unwrap();
    assert!(
        cored
            .as_cylinder()
            .unwrap()
            .dimensions
            .inner_diameter()
            .abs()
            < f32::EPSILON
    );
    assert!(cored.shares_core_with(pipe));
    assert_eq!(
        rubber_layer(layer_core(1.0, 0.0), LayerFace::Bore, 0.1),
        Err(LayerError::BoreRequired)
    );
}

#[test]
fn face_layer_grows_the_cuboid_and_shifts_its_centre_outward() {
    let block = PartSpec::Cuboid(
        CuboidSpec::new(
            [2, 2, 2],
            BuildPose::from_position_ticks(IVec3::new(0, 100, 0), GridRotation::default()),
        )
        .unwrap(),
    );
    let layered = rubber_layer(block, LayerFace::Face(FaceKind::PositiveY), 0.05).unwrap();
    let cuboid = layered.as_cuboid().unwrap();
    assert!(
        cuboid
            .size_meters()
            .abs_diff_eq(Vec3::new(0.5, 0.55, 0.5), 1.0e-5)
    );
    assert_eq!(
        cuboid.pose.translation_position_ticks(),
        IVec3::new(0, 110, 0)
    );
    assert_eq!(cuboid.without_layers(), block.as_cuboid().unwrap());
    assert_eq!(layered.band_at_local_point(Vec3::new(0.0, 0.26, 0.0)), 1);
    assert_eq!(layered.band_at_local_point(Vec3::new(0.0, 0.2, 0.0)), 0);
    assert_eq!(
        rubber_layer(block, LayerFace::Face(FaceKind::PositiveY), 0.0125),
        Err(LayerError::ThicknessOutOfRange),
        "a flat layer keeps the centre on whole ticks"
    );
    assert_eq!(
        rubber_layer(block, LayerFace::OuterWall, 0.25),
        Err(LayerError::UnsupportedFace)
    );
}

#[test]
fn cap_layer_lengthens_the_cylinder_off_grid() {
    let layered = rubber_layer(
        layer_core(1.0, 0.0),
        LayerFace::Face(FaceKind::NegativeY),
        0.01,
    )
    .unwrap();
    let cylinder = layered.as_cylinder().unwrap();
    assert_eq!(cylinder.dimensions.axial_length_ticks(), 204);
    assert_eq!(
        cylinder.pose.translation_position_ticks(),
        IVec3::new(0, -2, 0)
    );
    assert_eq!(layered.band_at_local_point(Vec3::new(0.0, -0.253, 0.0)), 1);
    assert!(layered.shares_core_with(layer_core(1.0, 0.0)));
    assert_eq!(
        rubber_layer(
            layer_core(1.0, 0.0),
            LayerFace::Face(FaceKind::PositiveX),
            0.25
        ),
        Err(LayerError::UnsupportedFace)
    );
}

#[test]
fn later_layers_own_the_corners_they_cover() {
    let cap = rubber_layer(
        layer_core(1.0, 0.0),
        LayerFace::Face(FaceKind::PositiveY),
        0.25,
    )
    .unwrap();
    let tyre = cap
        .with_layer(
            LayerFace::OuterWall,
            0.25,
            ConstructionMaterial::Concrete,
            crate::MaterialAppearance::BAKED,
        )
        .unwrap();
    // The wall layer spans the lengthened cylinder, cap zone included.
    let corner = Vec3::new(0.6, 0.3, 0.0);
    assert_eq!(tyre.band_at_local_point(corner - Vec3::Y * 0.125), 2);
    assert_eq!(
        tyre.band_at_local_point(Vec3::new(0.2, 0.3, 0.0) - Vec3::Y * 0.125),
        1
    );
    let mut many = layer_core(0.5, 0.0);
    for _ in 0..super::MAX_PART_LAYERS {
        many = rubber_layer(many, LayerFace::OuterWall, 0.05).unwrap();
    }
    assert_eq!(
        rubber_layer(many, LayerFace::OuterWall, 0.05),
        Err(LayerError::TooManyLayers)
    );
}

fn auger_spiral() -> super::SpiralSpec {
    super::SpiralSpec::new(
        100,
        1,
        super::SpiralHand::Right,
        super::SpiralProfile::square(10, 60).unwrap(),
        super::SpiralProfile::PLAIN,
        None,
    )
    .unwrap()
}

#[test]
fn a_square_profile_reads_as_ridge_then_groove_in_every_pitch() {
    let profile = super::SpiralProfile::square(10, 60).unwrap();
    let pitch = 0.25;
    for turn in [-1.0_f32, 0.0, 3.0] {
        assert!(profile.depth_at(turn * pitch + 0.0125, pitch).abs() < 1.0e-6);
        assert!((profile.depth_at(turn * pitch + 0.1, pitch) - 0.15).abs() < 1.0e-6);
    }
    let vee = super::SpiralProfile::vee(40, 20).unwrap();
    assert!((vee.depth_at(0.025, pitch) - 0.025).abs() < 1.0e-6);
    assert!((vee.depth_at(0.2, pitch) - 0.05).abs() < 1.0e-6);
}

#[test]
fn a_spiral_is_refused_where_it_cannot_be_cut() {
    let solid = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 2.0).unwrap(),
        BuildPose::default(),
    );
    assert!(solid.with_spiral(auger_spiral()).is_ok());

    let sector = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 2.0)
            .unwrap()
            .with_sweep_angle_degrees(180)
            .unwrap(),
        BuildPose::default(),
    );
    assert_eq!(
        sector.with_spiral(auger_spiral()),
        Err(super::SpiralError::PartialSector)
    );

    let thin = CylinderSpec::new(
        CylinderDimensions::new(0.3, 0.0, 2.0).unwrap(),
        BuildPose::default(),
    );
    assert_eq!(
        thin.with_spiral(auger_spiral()),
        Err(super::SpiralError::TooDeep)
    );

    let bore_thread = super::SpiralSpec::new(
        100,
        1,
        super::SpiralHand::Left,
        super::SpiralProfile::PLAIN,
        super::SpiralProfile::vee(20, 10).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(
        solid.with_spiral(bore_thread),
        Err(super::SpiralError::BoreRequired)
    );
    let tube = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.3, 2.0).unwrap(),
        BuildPose::default(),
    );
    assert!(tube.with_spiral(bore_thread).is_ok());

    let fine_thread = super::SpiralSpec::new(
        20,
        1,
        super::SpiralHand::Right,
        super::SpiralProfile::vee(20, 5).unwrap(),
        super::SpiralProfile::PLAIN,
        None,
    )
    .unwrap();
    let long = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 8.0).unwrap(),
        BuildPose::default(),
    );
    assert_eq!(
        long.with_spiral(fine_thread),
        Err(super::SpiralError::TooFine)
    );
}

#[test]
fn spirals_and_material_layers_refuse_each_other() {
    let plain = CylinderSpec::new(
        CylinderDimensions::new(0.5, 0.0, 2.0).unwrap(),
        BuildPose::default(),
    );
    let layer = |cylinder: CylinderSpec| {
        cylinder.with_layer(
            LayerFace::OuterWall,
            0.05,
            ConstructionMaterial::Rubber,
            crate::MaterialAppearance::BAKED,
        )
    };
    assert_eq!(
        layer(plain).unwrap().with_spiral(auger_spiral()),
        Err(super::SpiralError::Layered)
    );
    assert_eq!(
        layer(plain.with_spiral(auger_spiral()).unwrap()),
        Err(LayerError::SpiralPart)
    );
}

#[test]
fn a_profile_must_run_forwards_on_its_grid_within_one_pitch() {
    use super::{SpiralError, SpiralPoint, SpiralProfile};
    assert_eq!(
        SpiralProfile::new(&[SpiralPoint::new(20, 0), SpiralPoint::new(10, 5)]),
        Err(SpiralError::PointOutOfOrder)
    );
    assert_eq!(
        SpiralProfile::new(&[SpiralPoint::new(0, 3)]),
        Err(SpiralError::PointOutOfOrder)
    );
    assert_eq!(
        SpiralProfile::new(&[SpiralPoint::new(10, 0); 3]),
        Err(SpiralError::StackedPoints)
    );
    assert_eq!(
        super::SpiralSpec::new(
            20,
            1,
            super::SpiralHand::Right,
            SpiralProfile::square(40, 10).unwrap(),
            SpiralProfile::PLAIN,
            None,
        ),
        Err(SpiralError::PointOutOfOrder)
    );
    assert_eq!(
        super::SpiralSpec::new(
            100,
            1,
            super::SpiralHand::Right,
            SpiralProfile::PLAIN,
            SpiralProfile::PLAIN,
            None,
        ),
        Err(SpiralError::Empty)
    );
}
