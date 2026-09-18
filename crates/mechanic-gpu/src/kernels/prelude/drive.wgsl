struct Drive {
    mode: u32,
    max_acceleration: f32,
    max_speed: f32,
    target_speed: f32,
    target_angle: f32,
    min_angle: f32,
    max_angle: f32,
    source_a_max_acceleration: f32,
    source_a_no_load_speed: f32,
    source_b_max_acceleration: f32,
    source_b_no_load_speed: f32,
    padding: f32,
};

struct DriveConstraint {
    bearing: Bearing,
    drive: Drive,
    // Axis inertia, accumulated impulse, current angle, reserved.
    state: vec4<f32>,
    // Child body, parent body, tree direction, coordinate.
    metadata: vec4<u32>,
};

const FIXED_VELOCITY_SCALE: f32 = 1048576.0;
// The outer position loop converts error to a tapered velocity request. The
// existing torque-limited velocity loop supplies derivative damping, matching
// the cascaded position/velocity control used by physical servos.
const DRIVE_ANGLE_POSITION_GAIN: f32 = 6.0;
const DRIVE_ANGLE_BRAKE_MARGIN: f32 = 0.8;
const DRIVE_ANGLE_DEADBAND: f32 = 0.0005;
