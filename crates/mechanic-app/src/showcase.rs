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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[expect(
    clippy::too_many_lines,
    reason = "the structural spanning tree is clearest in build order"
)]
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
mod tests;
