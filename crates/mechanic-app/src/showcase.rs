use std::{error::Error, fmt};

use bevy::prelude::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
    DimensionError, FaceKind, FaceRef, GRID_UNIT_METERS, GraphError, GridRotation, PartId,
    TopologyError, WeldSpec,
};

pub(crate) const PART_COUNT: usize = 20_000;
pub(crate) const WELD_COUNT: usize = 14_704;
pub(crate) const BEARING_COUNT: usize = 3_712;
pub(crate) const COMPOUND_COUNT: usize = 5_297;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CreationPreset {
    PendulumGarden256,
    MobileWorkshop1024,
    ClosureLab4096,
    KineticShowcase20000,
}

impl CreationPreset {
    pub(crate) const ALL: [Self; 4] = [
        Self::PendulumGarden256,
        Self::MobileWorkshop1024,
        Self::ClosureLab4096,
        Self::KineticShowcase20000,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::PendulumGarden256 => "Pendulum Garden — 256 parts",
            Self::MobileWorkshop1024 => "Mobile Workshop — 1,024 parts",
            Self::ClosureLab4096 => "Closure Lab — 4,096 parts",
            Self::KineticShowcase20000 => "Kinetic Showcase — 20,000 parts",
        }
    }

    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::PendulumGarden256 => "Branched pendulums with arms and counterweights",
            Self::MobileWorkshop1024 => "Welded crossbars, branching links, and payloads",
            Self::ClosureLab4096 => "512 closed loops under falling contact stacks",
            Self::KineticShowcase20000 => "The full towers, ropes, mobiles, and obstacles",
        }
    }

    pub(crate) const fn part_count(self) -> usize {
        match self {
            Self::PendulumGarden256 => 256,
            Self::MobileWorkshop1024 => 1_024,
            Self::ClosureLab4096 => 4_096,
            Self::KineticShowcase20000 => PART_COUNT,
        }
    }

    pub(crate) const fn body_count(self) -> usize {
        match self {
            Self::KineticShowcase20000 => COMPOUND_COUNT,
            Self::MobileWorkshop1024 => 640,
            _ => self.part_count(),
        }
    }

    pub(crate) const fn weld_count(self) -> usize {
        match self {
            Self::PendulumGarden256 => 64,
            Self::MobileWorkshop1024 | Self::ClosureLab4096 => 512,
            Self::KineticShowcase20000 => WELD_COUNT,
        }
    }

    pub(crate) const fn bearing_count(self) -> usize {
        match self {
            Self::PendulumGarden256 => 192,
            Self::MobileWorkshop1024 => 512,
            Self::ClosureLab4096 => 2_048,
            Self::KineticShowcase20000 => BEARING_COUNT,
        }
    }

    pub(crate) fn matches(self, graph: &ConstructionGraph) -> bool {
        graph.part_count() == self.part_count()
            && graph.weld_count() == self.weld_count()
            && graph.bearing_count() == self.bearing_count()
    }
}

const TOWER_SIDE: usize = 12;
const TOWER_HEIGHT: usize = 24;
const OVERHEAD_SITES: usize = 144;
const CORE_STRUCTURE_PARTS: usize = 14_000;
// These hanger and lattice blocks join the named 14,000-block structure in the
// grounded support compound and reconcile the aggregate body/weld totals.
const LATTICE_BRACE_PARTS: usize = 416;
const SUPPORT_Z: i32 = 14;

#[derive(Debug)]
pub(crate) enum ShowcaseError {
    Dimension(DimensionError),
    Graph(GraphError),
    Topology(TopologyError),
    InternalCounts {
        parts: usize,
        welds: usize,
        bearings: usize,
    },
}

impl fmt::Display for ShowcaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension(error) => error.fmt(formatter),
            Self::Graph(error) => error.fmt(formatter),
            Self::Topology(error) => error.fmt(formatter),
            Self::InternalCounts {
                parts,
                welds,
                bearings,
            } => write!(
                formatter,
                "showcase generated {parts} parts, {welds} welds, and {bearings} bearings"
            ),
        }
    }
}

impl Error for ShowcaseError {}

impl From<DimensionError> for ShowcaseError {
    fn from(error: DimensionError) -> Self {
        Self::Dimension(error)
    }
}

impl From<GraphError> for ShowcaseError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<TopologyError> for ShowcaseError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

#[derive(Clone, Copy)]
enum PlannedFace {
    Part(usize, FaceKind),
    Ground,
}

#[derive(Clone, Copy)]
struct PlannedWeld {
    first: PlannedFace,
    second: PlannedFace,
}

#[derive(Clone, Copy)]
struct PlannedBearing {
    source: usize,
    source_face: FaceKind,
    target: usize,
    target_face: FaceKind,
    anchor: Vec3,
    axis: Vec3,
}

#[derive(Default)]
struct ShowcasePlan {
    parts: Vec<CuboidSpec>,
    welds: Vec<PlannedWeld>,
    bearings: Vec<PlannedBearing>,
}

impl ShowcasePlan {
    fn part(&mut self, dimensions: [u8; 3], position: [i32; 3]) -> Result<usize, DimensionError> {
        let index = self.parts.len();
        self.parts.push(CuboidSpec::new(
            dimensions,
            BuildPose::new(IVec3::from_array(position), GridRotation::default()),
        )?);
        Ok(index)
    }

    fn weld_parts(
        &mut self,
        first: usize,
        first_face: FaceKind,
        second: usize,
        second_face: FaceKind,
    ) {
        self.welds.push(PlannedWeld {
            first: PlannedFace::Part(first, first_face),
            second: PlannedFace::Part(second, second_face),
        });
    }

    fn weld_ground(&mut self, part: usize) {
        self.welds.push(PlannedWeld {
            first: PlannedFace::Ground,
            second: PlannedFace::Part(part, FaceKind::NegativeY),
        });
    }

    fn bearing(
        &mut self,
        source: usize,
        source_face: FaceKind,
        target: usize,
        target_face: FaceKind,
        anchor_units: Vec3,
        axis: Vec3,
    ) {
        self.bearings.push(PlannedBearing {
            source,
            source_face,
            target,
            target_face,
            anchor: anchor_units * GRID_UNIT_METERS,
            axis,
        });
    }
}

/// Builds the deterministic app-local showcase through validated graph batches.
pub(crate) fn build() -> Result<ConstructionGraph, ShowcaseError> {
    let mut plan = ShowcasePlan::default();
    let supports = add_grounded_structure(&mut plan)?;
    let rope_sites = add_mechanisms(&mut plan, &supports)?;
    add_loose_obstacles(&mut plan, &rope_sites)?;

    if plan.parts.len() != PART_COUNT
        || plan.welds.len() != WELD_COUNT
        || plan.bearings.len() != BEARING_COUNT
    {
        return Err(ShowcaseError::InternalCounts {
            parts: plan.parts.len(),
            welds: plan.welds.len(),
            bearings: plan.bearings.len(),
        });
    }

    instantiate(plan)
}

pub(crate) fn build_preset(preset: CreationPreset) -> Result<ConstructionGraph, ShowcaseError> {
    match preset {
        CreationPreset::PendulumGarden256 => build_pendulum_garden(),
        CreationPreset::MobileWorkshop1024 => build_mobile_workshop(),
        CreationPreset::ClosureLab4096 => build_closure_lab(),
        CreationPreset::KineticShowcase20000 => build(),
    }
}

pub(crate) fn uses_reduced_collision_mode(graph: &ConstructionGraph) -> bool {
    CreationPreset::KineticShowcase20000.matches(graph)
}

fn grid_origin(index: usize, item_count: usize, x_spacing: i32, z_spacing: i32) -> (i32, i32) {
    let mut columns = 1_usize;
    while columns * columns < item_count {
        columns += 1;
    }
    let row_count = item_count.div_ceil(columns);
    let column = i32::try_from(index % columns).unwrap();
    let row = i32::try_from(index / columns).unwrap();
    let x_span = i32::try_from(columns.saturating_sub(1)).unwrap() * x_spacing;
    let z_span = i32::try_from(row_count.saturating_sub(1)).unwrap() * z_spacing;
    (
        column * x_spacing - x_span / 2,
        row * z_spacing - z_span / 2,
    )
}

fn build_pendulum_garden() -> Result<ConstructionGraph, ShowcaseError> {
    let mut plan = ShowcasePlan::default();
    for index in 0..64 {
        let (x, z) = grid_origin(index, 64, 18, 10);
        let support = plan.part([2, 32, 2], [x, 16, z])?;
        let arm = plan.part([12, 2, 2], [x + 5, 30, z + 2])?;
        let pendulum = plan.part([2, 12, 6], [x + 12, 25, z + 4])?;
        let counterweight = plan.part([2, 4, 6], [x - 2, 30, z])?;
        plan.weld_ground(support);
        plan.bearing(
            support,
            FaceKind::PositiveZ,
            arm,
            FaceKind::NegativeZ,
            Vec3::new(units(x), 30.0, units(z + 1)),
            Vec3::Z,
        );
        plan.bearing(
            arm,
            FaceKind::PositiveX,
            pendulum,
            FaceKind::NegativeX,
            Vec3::new(units(x + 11), 30.0, units(z + 2)),
            Vec3::X,
        );
        plan.bearing(
            arm,
            FaceKind::NegativeX,
            counterweight,
            FaceKind::PositiveX,
            Vec3::new(units(x - 1), 30.0, units(z + 2)),
            Vec3::NEG_X,
        );
    }
    instantiate(plan)
}

fn build_mobile_workshop() -> Result<ConstructionGraph, ShowcaseError> {
    let mut plan = ShowcasePlan::default();
    for index in 0..128 {
        let (x, z) = grid_origin(index, 128, 24, 18);
        let support = plan.part([2, 32, 2], [x, 16, z])?;
        let root = plan.part([10, 2, 2], [x + 4, 30, z + 2])?;
        let crossbar = plan.part([2, 2, 10], [x + 10, 30, z + 2])?;
        let first_branch = plan.part([6, 10, 2], [x + 12, 26, z - 4])?;
        let second_branch = plan.part([6, 10, 2], [x + 12, 26, z + 8])?;
        let first_payload = plan.part([4, 4, 4], [x + 12, 19, z - 4])?;
        let second_payload = plan.part([4, 4, 4], [x + 12, 19, z + 8])?;
        let rotor = plan.part([8, 2, 2], [x + 13, 32, z + 2])?;

        plan.weld_ground(support);
        plan.weld_parts(root, FaceKind::PositiveX, crossbar, FaceKind::NegativeX);
        plan.weld_parts(
            first_branch,
            FaceKind::NegativeY,
            first_payload,
            FaceKind::PositiveY,
        );
        plan.weld_parts(
            second_branch,
            FaceKind::NegativeY,
            second_payload,
            FaceKind::PositiveY,
        );
        plan.bearing(
            support,
            FaceKind::PositiveZ,
            root,
            FaceKind::NegativeZ,
            Vec3::new(units(x), 30.0, units(z + 1)),
            Vec3::Z,
        );
        plan.bearing(
            crossbar,
            FaceKind::NegativeZ,
            first_branch,
            FaceKind::PositiveZ,
            Vec3::new(units(x + 10), 30.0, units(z - 3)),
            Vec3::NEG_Z,
        );
        plan.bearing(
            crossbar,
            FaceKind::PositiveZ,
            second_branch,
            FaceKind::NegativeZ,
            Vec3::new(units(x + 10), 30.0, units(z + 7)),
            Vec3::Z,
        );
        plan.bearing(
            crossbar,
            FaceKind::PositiveY,
            rotor,
            FaceKind::NegativeY,
            Vec3::new(units(x + 10), 31.0, units(z + 2)),
            Vec3::Y,
        );
    }
    instantiate(plan)
}

fn build_closure_lab() -> Result<ConstructionGraph, ShowcaseError> {
    let mut plan = ShowcasePlan::default();
    for index in 0..512 {
        let (origin_x, origin_z) = grid_origin(index, 512, 8, 6);
        let lower_left = plan.part([2, 2, 2], [origin_x, 1, origin_z])?;
        let lower_right = plan.part([2, 2, 2], [origin_x + 2, 1, origin_z])?;
        let upper_right = plan.part([2, 2, 2], [origin_x + 2, 3, origin_z])?;
        let upper_left = plan.part([2, 2, 2], [origin_x, 3, origin_z])?;
        for y in [7, 9, 11, 13] {
            plan.part([2, 2, 2], [origin_x + (y % 4), y, origin_z])?;
        }
        plan.weld_ground(lower_left);
        plan.bearing(
            lower_left,
            FaceKind::PositiveX,
            lower_right,
            FaceKind::NegativeX,
            Vec3::new(units(origin_x + 1), 1.0, units(origin_z)),
            Vec3::X,
        );
        plan.bearing(
            lower_right,
            FaceKind::PositiveY,
            upper_right,
            FaceKind::NegativeY,
            Vec3::new(units(origin_x + 2), 2.0, units(origin_z)),
            Vec3::Y,
        );
        plan.bearing(
            upper_right,
            FaceKind::NegativeX,
            upper_left,
            FaceKind::PositiveX,
            Vec3::new(units(origin_x + 1), 3.0, units(origin_z)),
            Vec3::NEG_X,
        );
        plan.bearing(
            upper_left,
            FaceKind::NegativeY,
            lower_left,
            FaceKind::PositiveY,
            Vec3::new(units(origin_x), 2.0, units(origin_z)),
            Vec3::NEG_Y,
        );
    }
    instantiate(plan)
}

fn instantiate(plan: ShowcasePlan) -> Result<ConstructionGraph, ShowcaseError> {
    let ShowcasePlan {
        parts,
        welds,
        bearings,
    } = plan;
    let mut graph = ConstructionGraph::new();
    let outcomes = graph.apply_batch(parts.into_iter().map(BuildCommand::Spawn))?;
    let part_ids = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("spawn batch only contains spawn commands"),
        })
        .collect::<Vec<_>>();

    let connection_commands = welds
        .into_iter()
        .map(|weld| {
            BuildCommand::Weld(WeldSpec {
                first: resolve_face(weld.first, &part_ids),
                second: resolve_face(weld.second, &part_ids),
            })
        })
        .chain(bearings.into_iter().map(|bearing| {
            BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(part_ids[bearing.source], bearing.source_face),
                FaceRef::part(part_ids[bearing.target], bearing.target_face),
                bearing.anchor,
                bearing.axis,
            ))
        }));
    graph.apply_batch(connection_commands)?;
    Ok(graph)
}

fn resolve_face(face: PlannedFace, parts: &[PartId]) -> FaceRef {
    match face {
        PlannedFace::Part(index, kind) => FaceRef::part(parts[index], kind),
        PlannedFace::Ground => FaceRef::ground(),
    }
}

#[allow(clippy::too_many_lines)] // The structural spanning tree is clearest in build order.
fn add_grounded_structure(
    plan: &mut ShowcasePlan,
) -> Result<[[usize; OVERHEAD_SITES]; 2], DimensionError> {
    let tower_centres = [-108, -36, 36, 108];
    let mut tower_tops = Vec::with_capacity(tower_centres.len());
    let mut ground_part = None;

    for &tower_x in &tower_centres {
        let base = plan.parts.len();
        for height in 0..TOWER_HEIGHT {
            for z_index in 0..TOWER_SIDE {
                for x_index in 0..TOWER_SIDE {
                    let x = tower_x - 10 + i32::try_from(x_index).unwrap() * 2;
                    let y = 1 + i32::try_from(height).unwrap() * 2;
                    let z = -10 + i32::try_from(z_index).unwrap() * 2;
                    let part = plan.part([2, 2, 2], [x, y, z])?;
                    if ground_part.is_none() {
                        ground_part = Some(part);
                    }
                    let parent = if x_index > 0 {
                        Some((part - 1, FaceKind::NegativeX, FaceKind::PositiveX))
                    } else if z_index > 0 {
                        Some((part - TOWER_SIDE, FaceKind::NegativeZ, FaceKind::PositiveZ))
                    } else if height > 0 {
                        Some((
                            part - TOWER_SIDE * TOWER_SIDE,
                            FaceKind::NegativeY,
                            FaceKind::PositiveY,
                        ))
                    } else {
                        None
                    };
                    if let Some((parent, part_face, parent_face)) = parent {
                        plan.weld_parts(part, part_face, parent, parent_face);
                    }
                }
            }
        }
        let top_centre = base + (TOWER_HEIGHT - 1) * TOWER_SIDE * TOWER_SIDE + 5 * TOWER_SIDE + 5;
        tower_tops.push(top_centre);
    }
    plan.weld_ground(ground_part.expect("the structural towers are non-empty"));

    let mut overhead = [0_usize; OVERHEAD_SITES];
    for index in 0..OVERHEAD_SITES {
        let x = overhead_x(index);
        let part = plan.part([2, 2, 2], [x, 49, 0])?;
        overhead[index] = part;
        if index > 0 {
            plan.weld_parts(
                overhead[index - 1],
                FaceKind::PositiveX,
                part,
                FaceKind::NegativeX,
            );
        }
    }
    for (&tower_x, &tower_top) in tower_centres.iter().zip(&tower_tops) {
        let beam_index = usize::try_from(tower_x / 2 + 71).unwrap();
        plan.weld_parts(
            tower_top,
            FaceKind::PositiveY,
            overhead[beam_index],
            FaceKind::NegativeY,
        );
        for direction in [-1, 1] {
            let mut parent = overhead[beam_index];
            for offset in 1..=4 {
                let crossbar = plan.part([2, 2, 2], [tower_x, 49, direction * offset * 2])?;
                let (parent_face, child_face) = z_faces(direction);
                plan.weld_parts(parent, parent_face, crossbar, child_face);
                parent = crossbar;
            }
        }
    }
    debug_assert_eq!(plan.parts.len(), CORE_STRUCTURE_PARTS);

    let connector = plan.part([2, 2, 2], [overhead_x(0), 51, 0])?;
    plan.weld_parts(
        overhead[0],
        FaceKind::PositiveY,
        connector,
        FaceKind::NegativeY,
    );
    let mut supports = [[0_usize; OVERHEAD_SITES]; 2];
    for (side_index, direction) in [-1, 1].into_iter().enumerate() {
        let mut bridge_parent = connector;
        for offset in 1..=6 {
            let bridge = plan.part([2, 2, 2], [overhead_x(0), 51, direction * offset * 2])?;
            let (parent_face, child_face) = z_faces(direction);
            plan.weld_parts(bridge_parent, parent_face, bridge, child_face);
            bridge_parent = bridge;
        }
        for index in 0..OVERHEAD_SITES {
            let support = plan.part([2, 2, 2], [overhead_x(index), 51, direction * SUPPORT_Z])?;
            supports[side_index][index] = support;
            if index == 0 {
                let (parent_face, support_face) = z_faces(direction);
                plan.weld_parts(bridge_parent, parent_face, support, support_face);
            } else {
                plan.weld_parts(
                    supports[side_index][index - 1],
                    FaceKind::PositiveX,
                    support,
                    FaceKind::NegativeX,
                );
            }
        }
    }
    let mut parent = connector;
    for index in 0..115 {
        let brace = plan.part([2, 2, 2], [overhead_x(index), 53, 0])?;
        if index == 0 {
            plan.weld_parts(parent, FaceKind::PositiveY, brace, FaceKind::NegativeY);
        } else {
            plan.weld_parts(parent, FaceKind::PositiveX, brace, FaceKind::NegativeX);
        }
        parent = brace;
    }
    debug_assert_eq!(plan.parts.len(), CORE_STRUCTURE_PARTS + LATTICE_BRACE_PARTS);
    Ok(supports)
}

fn add_mechanisms(
    plan: &mut ShowcasePlan,
    supports: &[[usize; OVERHEAD_SITES]; 2],
) -> Result<Vec<(i32, i32)>, DimensionError> {
    let mut rope_sites = Vec::with_capacity(256);
    for (side_index, direction) in [-1, 1].into_iter().enumerate() {
        for (index, &support) in supports[side_index].iter().enumerate() {
            let x = overhead_x(index);
            if index % 9 == 0 {
                add_mobile(plan, support, x, direction, index / 9)?;
            } else {
                add_rope(plan, support, x, direction)?;
                rope_sites.push((x, direction));
            }
        }
    }
    Ok(rope_sites)
}

fn add_rope(
    plan: &mut ShowcasePlan,
    support: usize,
    x: i32,
    direction: i32,
) -> Result<(), DimensionError> {
    let mut links = [0_usize; 12];
    for index in 0..links.len() {
        let y = 50 - i32::try_from(index).unwrap() * 3;
        let lane = SUPPORT_Z + if index.is_multiple_of(2) { 2 } else { 4 };
        let link = plan.part([1, 4, 2], [x, y, direction * lane])?;
        links[index] = link;
        if index == 0 {
            let (source_face, target_face, axis) = z_bearing(direction);
            plan.bearing(
                support,
                source_face,
                link,
                target_face,
                Vec3::new(units(x), 51.0, units(direction * (SUPPORT_Z + 1))),
                axis,
            );
        } else {
            bearing_between_z_links(
                plan,
                links[index - 1],
                link,
                x,
                52 - small_index(index) * 3,
                direction * (SUPPORT_Z + 3),
                if index.is_multiple_of(2) {
                    -direction
                } else {
                    direction
                },
            );
        }
    }
    let last = links[11];
    let payload = plan.part([2, 4, 4], [x + direction, 13, direction * (SUPPORT_Z + 4)])?;
    plan.weld_parts(last, FaceKind::NegativeY, payload, FaceKind::PositiveY);
    Ok(())
}

fn add_mobile(
    plan: &mut ShowcasePlan,
    support: usize,
    x: i32,
    direction: i32,
    row: usize,
) -> Result<(), DimensionError> {
    let width = if row.is_multiple_of(2) { 4 } else { 6 };
    let first_z = direction * (SUPPORT_Z + 1 + width / 2);
    let mut stem = [0_usize; 8];
    for index in 0..stem.len() {
        let y = 50 - i32::try_from(index).unwrap() * 3;
        let z = first_z + direction * width * i32::try_from(index).unwrap();
        let link = plan.part([1, 4, u8::try_from(width).unwrap()], [x, y, z])?;
        stem[index] = link;
        if index == 0 {
            let (source_face, target_face, axis) = z_bearing(direction);
            plan.bearing(
                support,
                source_face,
                link,
                target_face,
                Vec3::new(units(x), 51.0, units(direction * (SUPPORT_Z + 1))),
                axis,
            );
        } else {
            let previous_z = z - direction * width;
            let plane_z = i32::midpoint(previous_z, z);
            bearing_between_z_links(
                plan,
                stem[index - 1],
                link,
                x,
                52 - small_index(index) * 3,
                plane_z,
                direction,
            );
        }
    }
    let crossbar_z = first_z + direction * width * 7;
    let crossbar_x = x + if row.is_multiple_of(2) { 2 } else { -2 };
    let crossbar = plan.part([12, 2, 2], [crossbar_x, 26, crossbar_z])?;
    plan.weld_parts(stem[7], FaceKind::NegativeY, crossbar, FaceKind::PositiveY);
    for child_direction in [-1, 1] {
        add_mobile_child(
            plan,
            crossbar,
            crossbar_x,
            crossbar_z,
            child_direction,
            direction,
        )?;
    }
    Ok(())
}

fn add_mobile_child(
    plan: &mut ShowcasePlan,
    crossbar: usize,
    x: i32,
    z: i32,
    direction: i32,
    row_direction: i32,
) -> Result<(), DimensionError> {
    let child_z = z + row_direction;
    let mut links = [0_usize; 6];
    for index in 0..links.len() {
        let lane = if index.is_multiple_of(2) { 7 } else { 9 };
        let child_x = x + direction * lane;
        let y = 25 - i32::try_from(index).unwrap() * 3;
        let link = plan.part([2, 4, 1], [child_x, y, child_z])?;
        links[index] = link;
        if index == 0 {
            let (source_face, target_face, axis) = x_bearing(direction);
            plan.bearing(
                crossbar,
                source_face,
                link,
                target_face,
                Vec3::new(
                    units(x + direction * 6),
                    26.0,
                    units(z) + units(row_direction) * 0.75,
                ),
                axis,
            );
        } else {
            let previous_x = x + direction * if index.is_multiple_of(2) { 9 } else { 7 };
            let movement = (child_x - previous_x).signum();
            let (source_face, target_face, axis) = x_bearing(movement);
            plan.bearing(
                links[index - 1],
                source_face,
                link,
                target_face,
                Vec3::new(
                    units(previous_x + child_x) * 0.5,
                    units(27 - i32::try_from(index).unwrap() * 3) - 0.5,
                    units(child_z),
                ),
                axis,
            );
        }
    }
    Ok(())
}

fn bearing_between_z_links(
    plan: &mut ShowcasePlan,
    previous: usize,
    current: usize,
    x: i32,
    anchor_y_ceiling: i32,
    plane_z: i32,
    movement: i32,
) {
    let (source_face, target_face, axis) = z_bearing(movement);
    plan.bearing(
        previous,
        source_face,
        current,
        target_face,
        Vec3::new(units(x), units(anchor_y_ceiling) - 0.5, units(plane_z)),
        axis,
    );
}

fn add_loose_obstacles(
    plan: &mut ShowcasePlan,
    rope_sites: &[(i32, i32)],
) -> Result<(), DimensionError> {
    for &(x, direction) in rope_sites {
        for depth in 0..6 {
            plan.part([1, 4, 2], [x, 2, direction * (SUPPORT_Z + 8 + depth * 3)])?;
        }
    }
    for direction in [-1, 1] {
        let sites = rope_sites
            .iter()
            .filter(|(_, site_direction)| *site_direction == direction)
            .take(6);
        for (column, &(x, _)) in sites.enumerate() {
            let height = [2, 3, 4, 5, 6, 4][column];
            for level in 0..height {
                plan.part([2, 2, 2], [x, 1 + level * 2, direction * (SUPPORT_Z + 26)])?;
            }
        }
    }
    Ok(())
}

fn overhead_x(index: usize) -> i32 {
    -142 + small_index(index) * 2
}

fn small_index(index: usize) -> i32 {
    i32::try_from(index).expect("showcase loop bounds fit i32")
}

fn units(value: i32) -> f32 {
    f32::from(i16::try_from(value).expect("showcase grid coordinates fit i16"))
}

const fn z_faces(direction: i32) -> (FaceKind, FaceKind) {
    if direction > 0 {
        (FaceKind::PositiveZ, FaceKind::NegativeZ)
    } else {
        (FaceKind::NegativeZ, FaceKind::PositiveZ)
    }
}

fn z_bearing(direction: i32) -> (FaceKind, FaceKind, Vec3) {
    let (source, target) = z_faces(direction);
    (source, target, Vec3::Z * units(direction))
}

fn x_bearing(direction: i32) -> (FaceKind, FaceKind, Vec3) {
    if direction > 0 {
        (FaceKind::PositiveX, FaceKind::NegativeX, Vec3::X)
    } else {
        (FaceKind::NegativeX, FaceKind::PositiveX, Vec3::NEG_X)
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::{Quat, Vec3};

    use super::{
        BEARING_COUNT, COMPOUND_COUNT, CreationPreset, PART_COUNT, WELD_COUNT, build, build_preset,
        uses_reduced_collision_mode,
    };
    use mechanic_gpu::{GpuPhysics, GpuPhysicsConfig, MAX_BEARINGS, MAX_BODIES, MAX_COLLIDERS};

    #[test]
    fn showcase_has_exact_deterministic_topology() {
        let graph = build().unwrap();
        let creation = graph.compile().unwrap();

        assert_eq!(graph.part_count(), PART_COUNT);
        assert_eq!(graph.weld_count(), WELD_COUNT);
        assert_eq!(graph.bearing_count(), BEARING_COUNT);
        assert_eq!(creation.colliders.len(), PART_COUNT);
        assert_eq!(creation.compounds.len(), COMPOUND_COUNT);
        assert_eq!(creation.bearings.len(), BEARING_COUNT);
        assert_eq!(creation.loop_topology.tree_bearings.len(), BEARING_COUNT);
        assert!(creation.loop_topology.closure_bearings.is_empty());
        assert!(graph.pending().is_none());
        assert_eq!(
            creation
                .loop_topology
                .mechanism_components
                .iter()
                .filter(|component| component.len() > 1)
                .count(),
            1
        );
        assert!(
            creation
                .compounds
                .iter()
                .any(|compound| { compound.is_static && compound.source_parts.len() == 14_416 })
        );
        assert!(creation.compounds.iter().all(|compound| {
            let mass = compound.mass_properties;
            mass.mass.is_finite()
                && mass.center_of_mass.is_finite()
                && mass.inertia.is_finite()
                && mass.inverse_inertia.is_finite()
        }));
        assert!(creation.compounds.len() <= MAX_BODIES);
        assert!(creation.colliders.len() <= MAX_COLLIDERS);
        assert!(creation.bearings.len() <= MAX_BEARINGS);
        assert!(uses_reduced_collision_mode(&graph));
        assert_no_initial_intersections(&graph);
    }

    #[test]
    fn smaller_creation_presets_have_exact_valid_topology() {
        for preset in CreationPreset::ALL[..3].iter().copied() {
            let graph = build_preset(preset).unwrap();
            let creation = graph.compile().unwrap();
            let (component_count, component_size, closure_count) = match preset {
                CreationPreset::PendulumGarden256 => (64, 4, 0),
                CreationPreset::MobileWorkshop1024 => (128, 5, 0),
                CreationPreset::ClosureLab4096 => (512, 4, 512),
                CreationPreset::KineticShowcase20000 => unreachable!(),
            };

            assert_eq!(graph.part_count(), preset.part_count());
            assert_eq!(graph.weld_count(), preset.weld_count());
            assert_eq!(graph.bearing_count(), preset.bearing_count());
            assert_eq!(creation.compounds.len(), preset.body_count());
            assert_eq!(creation.bearings.len(), preset.bearing_count());
            assert_eq!(
                creation.loop_topology.tree_bearings.len(),
                preset.bearing_count() - closure_count
            );
            assert_eq!(creation.loop_topology.closure_bearings.len(), closure_count);
            assert_eq!(
                creation
                    .loop_topology
                    .mechanism_components
                    .iter()
                    .filter(|component| component.len() == component_size)
                    .count(),
                component_count
            );
            assert_bearings_are_centered_on_face_overlaps(&graph);
            assert!(!uses_reduced_collision_mode(&graph));
            assert_no_initial_intersections(&graph);
        }
    }

    #[test]
    fn smaller_creation_presets_run_120_gpu_ticks_without_failure() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()
        else {
            return;
        };
        let Some((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("mechanic creation preset test device"),
                ..Default::default()
            }))
            .ok()
        else {
            return;
        };

        for preset in CreationPreset::ALL[..3].iter().copied() {
            let creation = build_preset(preset).unwrap().compile().unwrap();
            let gpu = GpuPhysics::new_with_config(
                &device,
                &queue,
                &creation,
                GpuPhysicsConfig {
                    mechanism_self_collisions: true,
                    ..GpuPhysicsConfig::default()
                },
            )
            .unwrap();
            for tick in 1..=120 {
                gpu.dispatch_tick(&device, &queue, tick);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let diagnostics = gpu.read_last_tick(&device).unwrap();
                assert_eq!(
                    diagnostics.error_flags,
                    0,
                    "{} tick {tick} diagnostics: {diagnostics:?}",
                    preset.label()
                );
            }
            let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
            let moved_body_count = snapshot
                .iter()
                .zip(&creation.compounds)
                .filter(|(transform, compound)| {
                    let position = Vec3::from_slice(&transform.position[..3]);
                    let rotation = Quat::from_array(transform.rotation);
                    position.distance(compound.root_translation) > 1.0e-4
                        || rotation.angle_between(compound.root_rotation) > 1.0e-4
                })
                .count();
            assert!(
                moved_body_count > 0,
                "{} did not produce any simulated motion",
                preset.label()
            );
        }
    }

    #[test]
    fn dynamic_presets_do_not_gain_unbounded_spin() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()
        else {
            return;
        };
        let Some((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
        else {
            return;
        };

        for preset in [
            CreationPreset::MobileWorkshop1024,
            CreationPreset::ClosureLab4096,
        ] {
            let creation = build_preset(preset).unwrap().compile().unwrap();
            let gpu = GpuPhysics::new(&device, &queue, &creation).unwrap();
            for tick in 1..=600 {
                gpu.dispatch_tick(&device, &queue, tick);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let current = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
            let previous = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
            let maximum_angular_speed = current
                .iter()
                .zip(previous)
                .map(|(current, previous)| {
                    Quat::from_array(current.rotation)
                        .angle_between(Quat::from_array(previous.rotation))
                        * 60.0
                })
                .fold(0.0_f32, f32::max);
            let diagnostics = gpu.read_last_tick(&device).unwrap();
            assert_eq!(diagnostics.error_flags, 0, "{}", preset.label());
            assert!(
                maximum_angular_speed < 3.0,
                "{} reached {maximum_angular_speed:.3} rad/s",
                preset.label()
            );
        }
    }

    fn assert_bearings_are_centered_on_face_overlaps(graph: &mechanic_core::ConstructionGraph) {
        for (_, bearing) in graph.bearings() {
            let source = match bearing.source.owner {
                mechanic_core::FaceOwner::Part(part) => graph.part(part).unwrap(),
                mechanic_core::FaceOwner::Ground => unreachable!(),
            };
            let target = match bearing.target.owner {
                mechanic_core::FaceOwner::Part(part) => graph.part(part).unwrap(),
                mechanic_core::FaceOwner::Ground => unreachable!(),
            };
            let source_center = source.pose.translation();
            let target_center = target.pose.translation();
            let source_half = source.size_meters() * 0.5;
            let target_half = target.size_meters() * 0.5;
            let source_minimum = source_center - source_half;
            let source_maximum = source_center + source_half;
            let target_minimum = target_center - target_half;
            let target_maximum = target_center + target_half;
            let normal_axis = (0..3).find(|&axis| bearing.axis[axis].abs() > 0.5).unwrap();
            let mut expected = source_center + bearing.axis * source_half[normal_axis];
            for axis in 0..3 {
                if axis != normal_axis {
                    expected[axis] = (source_minimum[axis].max(target_minimum[axis])
                        + source_maximum[axis].min(target_maximum[axis]))
                        * 0.5;
                }
            }
            assert!(
                bearing.shared_anchor.abs_diff_eq(expected, 1.0e-5),
                "bearing anchor {:?} is not centered at {:?}",
                bearing.shared_anchor,
                expected
            );
        }
    }

    #[test]
    fn showcase_runs_1_200_gpu_ticks_without_failure_or_blowup() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()
        else {
            return;
        };
        let Some((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("mechanic showcase test device"),
                ..Default::default()
            }))
            .ok()
        else {
            return;
        };
        let creation = build().unwrap().compile().unwrap();
        let gpu = GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            GpuPhysicsConfig {
                mechanism_self_collisions: false,
                ..GpuPhysicsConfig::default()
            },
        )
        .unwrap();

        for tick in 1..=1_200 {
            gpu.dispatch_tick(&device, &queue, tick);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let diagnostics = gpu.read_last_tick(&device).unwrap();
            assert_eq!(
                diagnostics.error_flags, 0,
                "showcase tick {tick} diagnostics: {diagnostics:?}"
            );
        }
        let snapshot = gpu.read_snapshot_transforms(&device, &queue, 0).unwrap();
        let previous_snapshot = gpu.read_snapshot_transforms(&device, &queue, 2).unwrap();
        let mut maximum_displacement = 0.0_f32;
        let mut maximum_displacement_body = 0_usize;
        let mut maximum_speed = 0.0_f32;
        let mut maximum_speed_body = 0_usize;
        for (body, (transform, compound)) in snapshot.iter().zip(&creation.compounds).enumerate() {
            if compound.is_static || creation.loop_topology.body_parents[body].is_root {
                continue;
            }
            let position = Vec3::from_slice(&transform.position[..3]);
            let displacement = position.distance(compound.root_translation);
            if displacement > maximum_displacement {
                maximum_displacement = displacement;
                maximum_displacement_body = body;
            }
            let previous_position = Vec3::from_slice(&previous_snapshot[body].position[..3]);
            let speed = position.distance(previous_position) * 60.0;
            if speed > maximum_speed {
                maximum_speed = speed;
                maximum_speed_body = body;
            }
            assert!(
                position.is_finite(),
                "showcase body {body} has invalid position {position:?}"
            );
        }
        assert!(
            maximum_displacement < 5.0,
            "showcase body {maximum_displacement_body} moved {maximum_displacement} metres"
        );
        assert!(
            maximum_speed < 2.0,
            "showcase body {maximum_speed_body} reached {maximum_speed} m/s"
        );
    }

    fn assert_no_initial_intersections(graph: &mechanic_core::ConstructionGraph) {
        const EPSILON: f32 = 1.0e-5;

        let mut bounds = graph
            .parts()
            .map(|(part, spec)| {
                let half = spec.size_meters() * 0.5;
                (
                    part,
                    spec.pose.translation() - half,
                    spec.pose.translation() + half,
                )
            })
            .collect::<Vec<_>>();
        bounds.sort_unstable_by(|left, right| left.1.x.total_cmp(&right.1.x));

        for (index, &(part, minimum, maximum)) in bounds.iter().enumerate() {
            for &(other_part, other_minimum, other_maximum) in &bounds[index + 1..] {
                if other_minimum.x >= maximum.x - EPSILON {
                    break;
                }
                let overlap = minimum.cmplt(other_maximum - Vec3::splat(EPSILON)).all()
                    && other_minimum.cmplt(maximum - Vec3::splat(EPSILON)).all();
                assert!(
                    !overlap,
                    "showcase parts {part:?} and {other_part:?} initially intersect"
                );
            }
        }
    }
}
