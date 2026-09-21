//! Spirals: fitting the tool's settings to the cylinder under the cursor and staging the cut.

use super::bounds::{part_world_bounds, parts_overlap_with_frame, validate_world_bounds};
use super::{CONTACT_EPSILON, PlacementBounds, PlacementError, SurfaceHit, ToString, Vec3};
use mechanic_core::{
    BuildCommand, ConstructionFrame, ConstructionGraph, CylinderDimensions, CylinderSpec,
    FaceOwner, MAX_SPIRAL_PITCH_TICKS, MAX_SPIRAL_STARTS, MIN_SPIRAL_PITCH_TICKS,
    MIN_SPIRAL_TIP_DIAMETER_TICKS, POSITION_TICK_METERS, PartId, PartSpec,
    SPIRAL_PROFILE_STEP_TICKS, SpiralEnd, SpiralHand, SpiralProfile, SpiralSpec, SpiralTaper,
};

/// Shape of the ridge between its foot and its crest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SpiralPreset {
    /// Square flanks: auger flights and screw conveyors.
    #[default]
    Square,
    /// Both flanks sloped to an edge: threads.
    Vee,
    /// One square flank and one sloped.
    Buttress,
}

impl SpiralPreset {
    pub(crate) const fn next(self) -> Self {
        match self {
            Self::Square => Self::Vee,
            Self::Vee => Self::Buttress,
            Self::Buttress => Self::Square,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::Vee => "V",
            Self::Buttress => "buttress",
        }
    }

    fn profile(self, width_ticks: u16, depth_ticks: u16) -> Result<SpiralProfile, PlacementError> {
        match self {
            Self::Square => SpiralProfile::square(width_ticks, depth_ticks),
            Self::Vee => SpiralProfile::vee(width_ticks, depth_ticks),
            Self::Buttress => SpiralProfile::buttress(width_ticks, depth_ticks),
        }
        .map_err(spiral_error)
    }

    fn of(profile: SpiralProfile) -> Self {
        match profile.points() {
            [_, _, _, _] => Self::Square,
            [first, second, _] if first.position_ticks == second.position_ticks => Self::Buttress,
            _ => Self::Vee,
        }
    }
}

/// Whether the ridge is what a cut leaves standing, or stands on the cylinder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SpiralMode {
    /// The cylinder keeps its size and the groove is cut into it.
    #[default]
    Cut,
    /// The cylinder becomes the core and the ridge grows out of it.
    Add,
}

/// Which wall of the cylinder the spiral goes on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SpiralWall {
    #[default]
    Outer,
    Bore,
}

/// What the Spiral tool applies. All lengths are position ticks of 2.5 mm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpiralSettings {
    pub(crate) preset: SpiralPreset,
    pub(crate) mode: SpiralMode,
    pub(crate) wall: SpiralWall,
    pub(crate) pitch_ticks: u16,
    pub(crate) width_ticks: u16,
    pub(crate) depth_ticks: u16,
    pub(crate) starts: u8,
    pub(crate) hand: SpiralHand,
    /// Length of the tapered end nearest the cursor; zero for none.
    pub(crate) taper_ticks: u16,
    pub(crate) tip_ticks: u16,
    /// Whether pitch and depth have been sized to a cylinder yet.
    pub(crate) fitted: bool,
}

impl Default for SpiralSettings {
    fn default() -> Self {
        Self {
            preset: SpiralPreset::Square,
            mode: SpiralMode::Cut,
            wall: SpiralWall::Outer,
            pitch_ticks: 100,
            width_ticks: 10,
            depth_ticks: 20,
            starts: 1,
            hand: SpiralHand::Right,
            taper_ticks: 0,
            tip_ticks: 20,
            fitted: false,
        }
    }
}

/// One adjustable number of the tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpiralDimension {
    Pitch,
    Width,
    Depth,
    Starts,
    Taper,
    Tip,
}

impl SpiralSettings {
    /// Steps one number up or down on the profile grid, within its range.
    pub(crate) fn adjusted(mut self, dimension: SpiralDimension, direction: i8) -> Self {
        let step = i32::from(SPIRAL_PROFILE_STEP_TICKS) * i32::from(direction);
        let stepped = |value: u16, step: i32, low: u16, high: u16| {
            u16::try_from((i32::from(value) + step).clamp(i32::from(low), i32::from(high)))
                .expect("clamped into u16 range")
        };
        match dimension {
            SpiralDimension::Pitch => {
                self.pitch_ticks = stepped(
                    self.pitch_ticks,
                    step * 2,
                    MIN_SPIRAL_PITCH_TICKS,
                    MAX_SPIRAL_PITCH_TICKS,
                );
            }
            SpiralDimension::Width => {
                self.width_ticks = stepped(self.width_ticks, step, SPIRAL_PROFILE_STEP_TICKS, 3200);
            }
            SpiralDimension::Depth => {
                self.depth_ticks = stepped(self.depth_ticks, step, SPIRAL_PROFILE_STEP_TICKS, 1600);
            }
            SpiralDimension::Starts => {
                self.starts = u8::try_from(
                    (i32::from(self.starts) + i32::from(direction))
                        .clamp(1, i32::from(MAX_SPIRAL_STARTS)),
                )
                .expect("clamped into u8 range");
            }
            SpiralDimension::Taper => {
                self.taper_ticks = stepped(self.taper_ticks, step * 4, 0, 3200);
            }
            SpiralDimension::Tip => {
                self.tip_ticks = stepped(self.tip_ticks, step, MIN_SPIRAL_TIP_DIAMETER_TICKS, 3200);
            }
        }
        self.width_ticks = self.width_ticks.min(self.pitch_ticks);
        self
    }

    /// Sizes pitch and depth to a cylinder the first time the tool meets one:
    /// a pitch of one diameter, and a groove down to a third of the radius, or
    /// a ridge as high as the radius.
    pub(crate) fn fitted_to(mut self, cylinder: CylinderSpec) -> Self {
        if self.fitted {
            return self;
        }
        let grid = f32::from(SPIRAL_PROFILE_STEP_TICKS) * POSITION_TICK_METERS;
        let on_grid = |meters: f32, low: u16, high: u16| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a positive step count within one part"
            )]
            let steps = (meters / grid).round().max(1.0) as u16;
            (steps * SPIRAL_PROFILE_STEP_TICKS).clamp(low, high)
        };
        let outer = cylinder.dimensions.outer_diameter();
        let wall = (outer - cylinder.dimensions.inner_diameter()) * 0.5;
        self.pitch_ticks = on_grid(outer, MIN_SPIRAL_PITCH_TICKS, MAX_SPIRAL_PITCH_TICKS);
        self.depth_ticks = match self.mode {
            SpiralMode::Cut => on_grid(wall * 0.65, SPIRAL_PROFILE_STEP_TICKS, 1600),
            SpiralMode::Add => on_grid(outer * 0.5, SPIRAL_PROFILE_STEP_TICKS, 1600),
        };
        self.width_ticks = self
            .width_ticks
            .min(self.pitch_ticks / 2 / SPIRAL_PROFILE_STEP_TICKS * SPIRAL_PROFILE_STEP_TICKS)
            .max(SPIRAL_PROFILE_STEP_TICKS);
        self.fitted = true;
        self
    }

    /// The settings as they go onto one cylinder. They are the player's
    /// wishes, set with no cylinder in mind, so they bend to fit instead of
    /// being refused: a solid cylinder takes the spiral on its outer wall, a cut
    /// goes no deeper than the wall allows, and a tip is no wider than the part.
    pub(crate) fn suited_to(self, cylinder: CylinderSpec) -> Self {
        let mut suited = self;
        let dimensions = cylinder.dimensions;
        let (outer, inner) = (dimensions.outer_diameter(), dimensions.inner_diameter());
        if inner <= 0.0 {
            suited.wall = SpiralWall::Outer;
        }
        let on_grid = |meters: f32| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a length within one part, floored at zero"
            )]
            let ticks = (meters / POSITION_TICK_METERS + 1.0e-3).floor().max(0.0) as u16;
            ticks / SPIRAL_PROFILE_STEP_TICKS * SPIRAL_PROFILE_STEP_TICKS
        };
        let least_wall = mechanic_core::MIN_CYLINDER_DIAMETER_GAP * 0.5;
        let room = match (suited.mode, suited.wall) {
            (SpiralMode::Cut, wall) => {
                let across = cylinder.spiral().map_or(0, |spiral| match wall {
                    SpiralWall::Outer => spiral.inner().max_depth_ticks(),
                    SpiralWall::Bore => spiral.outer().max_depth_ticks(),
                });
                (outer - inner) * 0.5 - f32::from(across) * POSITION_TICK_METERS - least_wall
            }
            // A ridge grown into the bore must leave the bore open.
            (SpiralMode::Add, SpiralWall::Bore) => blank(suited, cylinder).1 * 0.5 - least_wall,
            (SpiralMode::Add, SpiralWall::Outer) => f32::INFINITY,
        };
        if room.is_finite() {
            suited.depth_ticks = suited
                .depth_ticks
                .min(on_grid(room))
                .max(SPIRAL_PROFILE_STEP_TICKS);
        }
        let envelope = if (suited.mode, suited.wall) == (SpiralMode::Add, SpiralWall::Outer) {
            blank(suited, cylinder).0 + f32::from(suited.depth_ticks) * POSITION_TICK_METERS * 2.0
        } else {
            outer
        };
        suited.tip_ticks = suited
            .tip_ticks
            .min(on_grid(envelope))
            .max(MIN_SPIRAL_TIP_DIAMETER_TICKS);
        suited
    }

    /// Reads the settings back off a cylinder that carries a spiral.
    pub(crate) fn picked_from(self, cylinder: CylinderSpec) -> Option<Self> {
        let spiral = cylinder.spiral()?;
        let (wall, profile) = if spiral.outer().is_plain() && !spiral.inner().is_plain() {
            (SpiralWall::Bore, spiral.inner())
        } else {
            (SpiralWall::Outer, spiral.outer())
        };
        Some(Self {
            preset: SpiralPreset::of(profile),
            wall,
            pitch_ticks: spiral.pitch_ticks(),
            width_ticks: profile
                .points()
                .last()
                .map_or(self.width_ticks, |point| point.position_ticks)
                .max(SPIRAL_PROFILE_STEP_TICKS),
            depth_ticks: profile.max_depth_ticks().max(SPIRAL_PROFILE_STEP_TICKS),
            starts: spiral.starts(),
            hand: spiral.hand(),
            taper_ticks: spiral.taper().map_or(0, |taper| taper.length_ticks),
            tip_ticks: spiral
                .taper()
                .map_or(self.tip_ticks, |taper| taper.tip_diameter_ticks),
            fitted: true,
            ..self
        })
    }

    pub(crate) fn summary(self) -> String {
        let centimetres = |ticks: u16| f32::from(ticks) * POSITION_TICK_METERS * 100.0;
        let taper = if self.taper_ticks == 0 {
            String::new()
        } else {
            format!(
                ", taper {:.0} cm to {:.1} cm",
                centimetres(self.taper_ticks),
                centimetres(self.tip_ticks)
            )
        };
        format!(
            "{} {} spiral on the {}: pitch {:.1} cm, ridge {:.1} cm wide, {:.1} cm {}, {} start{}, {}-hand{taper}",
            match self.mode {
                SpiralMode::Cut => "Cut",
                SpiralMode::Add => "Added",
            },
            self.preset.label(),
            match self.wall {
                SpiralWall::Outer => "outer wall",
                SpiralWall::Bore => "bore",
            },
            centimetres(self.pitch_ticks),
            centimetres(self.width_ticks),
            centimetres(self.depth_ticks),
            match self.mode {
                SpiralMode::Cut => "deep",
                SpiralMode::Add => "high",
            },
            self.starts,
            if self.starts == 1 { "" } else { "s" },
            match self.hand {
                SpiralHand::Right => "right",
                SpiralHand::Left => "left",
            },
        )
    }
}

/// The cylinder a spiral goes on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpiralTarget {
    pub(crate) part: PartId,
    pub(crate) spec: CylinderSpec,
    /// Construction frame the part is authored in.
    pub(crate) frame: ConstructionFrame,
    /// End of the cylinder nearest the picked point, which a taper narrows.
    pub(crate) near_end: SpiralEnd,
}

fn spiral_error(error: mechanic_core::SpiralError) -> PlacementError {
    PlacementError::Graph(error.to_string())
}

pub(crate) fn spiral_target_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<SpiralTarget, PlacementError> {
    let not_a_cylinder = || PlacementError::Graph("point at a full cylinder".to_owned());
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(not_a_cylinder());
    };
    let Some(PartSpec::Cylinder(spec)) = graph.part(part).copied() else {
        return Err(not_a_cylinder());
    };
    let frame = graph.part_frame(part).ok_or_else(not_a_cylinder)?;
    let local = spec.pose.rotation.quaternion().inverse()
        * (frame.inverse().point(hit.point) - spec.pose.translation());
    Ok(SpiralTarget {
        part,
        spec,
        frame,
        near_end: if local.y < 0.0 {
            SpiralEnd::NegativeY
        } else {
            SpiralEnd::PositiveY
        },
    })
}

// The cylinder a new spiral starts from. A cut starts from the full envelope; an
// added ridge starts from what stands beneath the deepest cut already there.
fn blank(settings: SpiralSettings, cylinder: CylinderSpec) -> (f32, f32) {
    let dimensions = cylinder.dimensions;
    let (mut outer, mut inner) = (dimensions.outer_diameter(), dimensions.inner_diameter());
    if let (SpiralMode::Add, Some(spiral)) = (settings.mode, cylinder.spiral()) {
        let ticks = |ticks: u16| f32::from(ticks) * POSITION_TICK_METERS * 2.0;
        match settings.wall {
            SpiralWall::Outer => outer -= ticks(spiral.outer().max_depth_ticks()),
            SpiralWall::Bore => inner += ticks(spiral.inner().max_depth_ticks()),
        }
    }
    (outer, inner)
}

/// The target with the settings' spiral on the chosen wall. Whatever is drawn on
/// the other wall stays.
pub(crate) fn spiralled(
    settings: SpiralSettings,
    target: &SpiralTarget,
) -> Result<CylinderSpec, PlacementError> {
    let cylinder = target.spec;
    if !cylinder.layers().is_empty() {
        return Err(spiral_error(mechanic_core::SpiralError::Layered));
    }
    if cylinder.dimensions.sweep_angle_degrees() != mechanic_core::MAX_CYLINDER_SWEEP_DEGREES {
        return Err(spiral_error(mechanic_core::SpiralError::PartialSector));
    }
    let settings = settings.suited_to(cylinder);
    let (mut outer, mut inner) = blank(settings, cylinder);
    let growth = f32::from(settings.depth_ticks) * POSITION_TICK_METERS * 2.0;
    if settings.mode == SpiralMode::Add {
        match settings.wall {
            SpiralWall::Outer => outer += growth,
            SpiralWall::Bore => inner -= growth,
        }
    }
    if settings.wall == SpiralWall::Bore && inner <= 0.0 {
        return Err(PlacementError::Graph(
            "a bore spiral needs a hollow cylinder with room for the ridge".to_owned(),
        ));
    }
    let profile = settings
        .preset
        .profile(settings.width_ticks, settings.depth_ticks)?;
    let kept = cylinder.spiral();
    let (outer_profile, inner_profile) = match settings.wall {
        SpiralWall::Outer => (
            profile,
            kept.map_or(SpiralProfile::PLAIN, SpiralSpec::inner),
        ),
        SpiralWall::Bore => (
            kept.map_or(SpiralProfile::PLAIN, SpiralSpec::outer),
            profile,
        ),
    };
    let taper = (settings.taper_ticks > 0).then_some(SpiralTaper {
        end: target.near_end,
        length_ticks: settings
            .taper_ticks
            .min(cylinder.dimensions.axial_length_ticks()),
        tip_diameter_ticks: settings.tip_ticks,
    });
    let spiral = SpiralSpec::new(
        settings.pitch_ticks,
        settings.starts,
        settings.hand,
        outer_profile,
        inner_profile,
        taper,
    )
    .map_err(spiral_error)?;
    resized(cylinder, outer, inner)?
        .with_spiral(spiral)
        .map_err(spiral_error)
}

/// The target without its spiral: grooves filled in when the tool cuts, ridges
/// stripped to the core when it adds.
pub(crate) fn unspiralled(
    settings: SpiralSettings,
    target: &SpiralTarget,
) -> Result<CylinderSpec, PlacementError> {
    let ticks = |ticks: u16| f32::from(ticks) * POSITION_TICK_METERS * 2.0;
    let dimensions = target.spec.dimensions;
    let (mut outer, mut inner) = (dimensions.outer_diameter(), dimensions.inner_diameter());
    if let (SpiralMode::Add, Some(spiral)) = (settings.mode, target.spec.spiral()) {
        outer -= ticks(spiral.outer().max_depth_ticks());
        if inner > 0.0 {
            inner += ticks(spiral.inner().max_depth_ticks());
        }
    }
    resized(target.spec, outer, inner)
}

// The same plain cylinder with other diameters.
fn resized(cylinder: CylinderSpec, outer: f32, inner: f32) -> Result<CylinderSpec, PlacementError> {
    let dimensions =
        CylinderDimensions::new(outer, inner.max(0.0), cylinder.dimensions.axial_length())
            .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(CylinderSpec::new(dimensions, cylinder.pose)
        .with_material(cylinder.material)
        .with_appearance(cylinder.appearance))
}

/// Whether the target may become `spec`: a grown envelope must stay inside the
/// build space and clear of every other part.
pub(crate) fn validate_spiral(
    graph: &ConstructionGraph,
    target: &SpiralTarget,
    spec: CylinderSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let grew = spec.dimensions.outer_diameter() > target.spec.dimensions.outer_diameter() + 1.0e-4
        || spec.dimensions.inner_diameter() < target.spec.dimensions.inner_diameter() - 1.0e-4;
    if !grew {
        return Ok(());
    }
    let (minimum, maximum) = part_world_bounds(PartSpec::Cylinder(spec));
    if target.frame == ConstructionFrame::IDENTITY {
        validate_world_bounds(minimum, maximum, bounds)?;
    }
    let into_target = target.frame.inverse();
    for (part, existing) in graph.parts() {
        if part == target.part {
            continue;
        }
        let existing_frame = graph
            .part_frame(part)
            .expect("validated parts have construction frames");
        if existing_frame == target.frame {
            let (low, high) = part_world_bounds(*existing);
            if (low - maximum).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                || (minimum - high).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
            {
                continue;
            }
        }
        if parts_overlap_with_frame(
            PartSpec::Cylinder(spec),
            *existing,
            into_target.compose(existing_frame),
        ) {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

/// Replaces the target's walls in place in one edit, keeping its connections.
pub(crate) fn stage_spiral(
    graph: &ConstructionGraph,
    target: &SpiralTarget,
    spec: CylinderSpec,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_spiral(graph, target, spec, bounds)?;
    let mut staged = graph.begin_edit();
    staged
        .apply_batch([BuildCommand::SetSpiral {
            part: target.part,
            spec,
        }])
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BuildOutcome, BuildPose, CuboidSpec, GridRotation};

    // A cylinder standing in an empty build space, 50 cm across and 2 m long.
    fn shaft(outer: f32, inner: f32) -> (ConstructionGraph, SpiralTarget) {
        let mut graph = ConstructionGraph::new();
        let spec = CylinderSpec::new(
            CylinderDimensions::new(outer, inner, 2.0).unwrap(),
            BuildPose::from_position_ticks(bevy::math::IVec3::Y * 800, GridRotation::default()),
        );
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            unreachable!()
        };
        let frame = graph.part_frame(part).unwrap();
        (
            graph,
            SpiralTarget {
                part,
                spec,
                frame,
                near_end: SpiralEnd::NegativeY,
            },
        )
    }

    #[test]
    fn settings_size_themselves_to_the_first_cylinder_they_meet() {
        let (_, target) = shaft(0.5, 0.0);
        let fitted = SpiralSettings::default().fitted_to(target.spec);
        assert_eq!(fitted.pitch_ticks, 200, "a pitch of one diameter");
        assert_eq!(
            fitted.depth_ticks, 65,
            "a groove down to a third of the radius"
        );
        assert!(spiralled(fitted, &target).is_ok());
        // Once sized they stay as the player left them.
        let (_, thin) = shaft(0.25, 0.0);
        assert_eq!(fitted.fitted_to(thin.spec), fitted);
    }

    #[test]
    fn a_cut_keeps_the_cylinders_size_and_an_added_ridge_widens_it() {
        let (_, target) = shaft(0.5, 0.0);
        let cut = SpiralSettings {
            depth_ticks: 60,
            ..SpiralSettings::default()
        };
        let spec = spiralled(cut, &target).unwrap();
        assert!((spec.dimensions.outer_diameter() - 0.5).abs() < 1.0e-6);
        assert_eq!(spec.spiral().unwrap().outer().max_depth_ticks(), 60);

        let added = SpiralSettings {
            mode: SpiralMode::Add,
            ..cut
        };
        let spec = spiralled(added, &target).unwrap();
        assert!((spec.dimensions.outer_diameter() - 0.8).abs() < 1.0e-6);
        assert_eq!(spec.pose, target.spec.pose);
    }

    #[test]
    fn reshaping_an_added_spiral_starts_again_from_the_shaft() {
        let (_, mut target) = shaft(0.5, 0.0);
        let added = SpiralSettings {
            mode: SpiralMode::Add,
            depth_ticks: 60,
            ..SpiralSettings::default()
        };
        target.spec = spiralled(added, &target).unwrap();
        let higher = added.adjusted(SpiralDimension::Depth, 4);
        let spec = spiralled(higher, &target).unwrap();
        assert!((spec.dimensions.outer_diameter() - 0.9).abs() < 1.0e-6);
        let plain = unspiralled(added, &target).unwrap();
        assert!((plain.dimensions.outer_diameter() - 0.5).abs() < 1.0e-6);
        assert!(plain.spiral().is_none());
    }

    #[test]
    fn a_bore_spiral_leaves_the_outer_walls_spiral_alone() {
        let (_, mut target) = shaft(0.75, 0.25);
        let outside = SpiralSettings {
            depth_ticks: 20,
            ..SpiralSettings::default()
        };
        target.spec = spiralled(outside, &target).unwrap();
        let inside = SpiralSettings {
            wall: SpiralWall::Bore,
            preset: SpiralPreset::Vee,
            depth_ticks: 10,
            ..outside
        };
        let spiral = spiralled(inside, &target).unwrap().spiral().unwrap();
        assert_eq!(spiral.outer().max_depth_ticks(), 20);
        assert_eq!(spiral.inner().max_depth_ticks(), 10);
        // A solid cylinder has no bore, so the same settings go on its wall.
        let (_, solid) = shaft(0.5, 0.0);
        let spiral = spiralled(inside, &solid).unwrap().spiral().unwrap();
        assert_eq!(spiral.outer().max_depth_ticks(), 10);
        assert!(spiral.inner().is_plain());
    }

    #[test]
    fn settings_that_do_not_fit_a_cylinder_bend_to_it_instead_of_being_refused() {
        // Sized on a thick shaft, with a wide tip and the bore chosen.
        let wishes = SpiralSettings {
            wall: SpiralWall::Bore,
            depth_ticks: 300,
            taper_ticks: 200,
            tip_ticks: 400,
            fitted: true,
            ..SpiralSettings::default()
        };
        let (_, thin) = shaft(0.25, 0.0);
        let suited = wishes.suited_to(thin.spec);
        assert_eq!(
            suited.wall,
            SpiralWall::Outer,
            "a solid cylinder has no bore"
        );
        assert_eq!(
            suited.depth_ticks, 40,
            "10 cm: all but 2.5 cm of the radius"
        );
        assert_eq!(suited.tip_ticks, 100, "no wider than the cylinder");
        let spec = spiralled(wishes, &thin).expect("the wishes bend to fit");
        assert_eq!(spec.spiral().unwrap().outer().max_depth_ticks(), 40);

        // A ridge grown into a bore leaves the bore open.
        let (_, tube) = shaft(0.5, 0.25);
        let inward = SpiralSettings {
            mode: SpiralMode::Add,
            ..wishes
        };
        let spec = spiralled(inward, &tube).expect("the ridge is kept short of the axis");
        assert!(spec.dimensions.inner_diameter() >= 0.05 - 1.0e-6);
    }

    #[test]
    fn a_taper_narrows_the_end_the_player_points_at() {
        let (_, target) = shaft(0.5, 0.0);
        let pointed = SpiralSettings {
            depth_ticks: 60,
            taper_ticks: 120,
            tip_ticks: 20,
            ..SpiralSettings::default()
        };
        let taper = spiralled(pointed, &target)
            .unwrap()
            .spiral()
            .unwrap()
            .taper();
        assert_eq!(taper.map(|taper| taper.end), Some(SpiralEnd::NegativeY));
    }

    #[test]
    fn settings_picked_up_from_a_spiral_put_the_same_spiral_back() {
        let (_, mut target) = shaft(0.5, 0.0);
        let settings = SpiralSettings {
            preset: SpiralPreset::Buttress,
            pitch_ticks: 120,
            width_ticks: 40,
            depth_ticks: 50,
            starts: 2,
            hand: SpiralHand::Left,
            fitted: true,
            ..SpiralSettings::default()
        };
        let spec = spiralled(settings, &target).unwrap();
        target.spec = spec;
        let picked = SpiralSettings::default().picked_from(spec).unwrap();
        assert_eq!(picked, settings);
        assert_eq!(spiralled(picked, &target).unwrap(), spec);
    }

    #[test]
    fn an_added_ridge_is_refused_where_a_neighbour_stands_in_its_way() {
        let (mut graph, target) = shaft(0.5, 0.0);
        // A block whose near face is 10 cm off the shaft's wall.
        let BuildOutcome::Spawned(block) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        bevy::math::IVec3::new(190, 800, 0),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let added = SpiralSettings {
            mode: SpiralMode::Add,
            depth_ticks: 60,
            ..SpiralSettings::default()
        };
        let spec = spiralled(added, &target).unwrap();
        assert_eq!(
            validate_spiral(&graph, &target, spec, PlacementBounds::Garage),
            Err(PlacementError::OverlapsPart(block))
        );
        assert!(stage_spiral(&graph, &target, spec, PlacementBounds::Garage).is_err());
        // Cutting into the shaft needs no room at all.
        let cut = spiralled(
            SpiralSettings {
                depth_ticks: 60,
                ..SpiralSettings::default()
            },
            &target,
        )
        .unwrap();
        let staged = stage_spiral(&graph, &target, cut, PlacementBounds::Garage).unwrap();
        assert_eq!(staged.part(target.part), Some(&PartSpec::Cylinder(cut)));
    }
}
