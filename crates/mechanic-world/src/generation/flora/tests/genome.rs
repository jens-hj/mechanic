use bevy_math::DVec3;

use super::super::{Part, TreeMetrics, grow_tree};
use super::{SEEDS, grove, mean, preset};

#[test]
fn same_seed_grows_the_same_tree() {
    let oak = preset("oak");
    let origin = DVec3::new(1.0, 2.0, 3.0);
    assert_eq!(grow_tree(&oak, 42, origin), grow_tree(&oak, 42, origin));
}

#[test]
fn different_seeds_grow_different_trees() {
    let oak = preset("oak");
    let first = grow_tree(&oak, 1, DVec3::ZERO);
    let second = grow_tree(&oak, 2, DVec3::ZERO);
    assert!(
        first.segments.len() != second.segments.len()
            || (first.height - second.height).abs() > 1.0e-6
    );
}

#[test]
fn every_preset_matches_its_genome() {
    for species in super::super::SpeciesSpec::embedded() {
        let trees = grove(&species, SEEDS);
        for (_, metrics) in &trees {
            assert!(
                metrics.height >= species.height.0 * 0.8
                    && metrics.height <= species.height.1 * 1.2,
                "{} height {:.1} outside {:?}",
                species.name,
                metrics.height,
                species.height
            );
            assert!(
                (species.stems.0..=species.stems.1).contains(&metrics.stems),
                "{} grew {} stems",
                species.name,
                metrics.stems
            );
        }
        let width = mean(trees.iter().map(|(_, metrics)| metrics.width_ratio));
        assert!(
            (width - species.width).abs() <= species.width * 0.3,
            "{} crown width ratio {width:.2}, genome {}",
            species.name,
            species.width
        );
    }
}

#[test]
fn spruce_is_widest_low_and_oak_widest_high() {
    let widest = |name| mean(grove(&preset(name), SEEDS).iter().map(|(_, m)| m.widest_at));
    let (spruce, oak, birch) = (widest("spruce"), widest("oak"), widest("birch"));
    assert!(spruce < 0.35, "spruce widest at {spruce:.2}");
    assert!(oak > 0.5, "oak widest at {oak:.2}");
    assert!((0.3..=0.7).contains(&birch), "birch widest at {birch:.2}");
}

#[test]
fn willow_hangs_and_poplar_reaches_up() {
    let rise = |name| {
        mean(
            grove(&preset(name), SEEDS)
                .iter()
                .map(|(_, m)| m.terminal_rise),
        )
    };
    let (willow, poplar) = (rise("willow"), rise("poplar"));
    assert!(willow < -0.3, "willow twigs rise {willow:.2}");
    assert!(poplar > 0.3, "poplar twigs rise {poplar:.2}");
    let spruce = mean(
        grove(&preset("spruce"), SEEDS)
            .iter()
            .map(|(_, m)| m.lateral_rise),
    );
    assert!(spruce.abs() < 0.4, "spruce branches rise {spruce:.2}");
}

#[test]
fn bamboo_grows_a_clump_of_upright_stems() {
    for (_, metrics) in grove(&preset("bamboo"), SEEDS) {
        assert!(
            metrics.trunk_share >= 0.8,
            "trunk share {:.2}",
            metrics.trunk_share
        );
        assert!(
            metrics.max_stem_lean < 15.0,
            "lean {:.1}",
            metrics.max_stem_lean
        );
    }
}

#[test]
fn foliage_density_sets_filled_fraction() {
    let mut oak = preset("oak");
    for density in [0.2, 0.5, 0.8] {
        oak.foliage.density = density;
        let filled = mean(
            grove(&oak, 2)
                .iter()
                .map(|(tree, _)| TreeMetrics::filled_fraction(tree, 4_000)),
        );
        assert!(
            (filled - density).abs() <= 0.12,
            "density {density} filled {filled:.2}"
        );
    }
}

#[test]
fn oak_roots_go_deeper_and_spruce_roots_wider() {
    let roots = |name| {
        let trees = grove(&preset(name), SEEDS);
        (
            mean(trees.iter().map(|(_, m)| m.root_depth)),
            mean(trees.iter().map(|(_, m)| m.root_radius)),
        )
    };
    let (oak_depth, _) = roots("oak");
    let (spruce_depth, spruce_radius) = roots("spruce");
    assert!(
        oak_depth > spruce_depth,
        "oak {oak_depth:.2} spruce {spruce_depth:.2}"
    );
    assert!(
        spruce_radius > 2.0 * spruce_depth,
        "spruce roots {spruce_radius:.2} wide, {spruce_depth:.2} deep"
    );
}

#[test]
fn trees_are_connected_and_bounded() {
    for species in super::super::SpeciesSpec::embedded() {
        for (tree, _) in grove(&species, SEEDS) {
            for segment in &tree.segments {
                for value in [segment.a, segment.b] {
                    assert!(
                        value.is_finite(),
                        "{} has a non-finite segment",
                        species.name
                    );
                }
            }
            for axis in &tree.axes {
                let first = &tree.segments[axis.segments.0 as usize];
                match axis.parent_segment {
                    Some(parent) => {
                        let parent = &tree.segments[parent as usize];
                        assert!(
                            first.a.distance(parent.b) < 1.0e-9,
                            "{} axis starts off its parent",
                            species.name
                        );
                    }
                    None if axis.part == Part::Root => {
                        assert!(first.a.distance(tree.origin) < 1.0e-9);
                    }
                    None => {
                        // Flares sit at the base.
                        assert!((first.a - tree.origin).with_y(0.0).length() < tree.height);
                    }
                }
            }
            let span = tree.max - tree.min;
            assert!(
                span.x.max(span.z) <= 1.6 * tree.height,
                "{} spans {:.1} m across at height {:.1}",
                species.name,
                span.x.max(span.z),
                tree.height
            );
            assert!(
                tree.max.y - tree.origin.y <= 1.2 * tree.height,
                "{} rises {:.1} m at height {:.1}",
                species.name,
                tree.max.y - tree.origin.y,
                tree.height
            );
        }
    }
}

#[test]
fn foliage_and_wood_are_solid_and_far_air_is_not() {
    let tree = grow_tree(&preset("oak"), 5, DVec3::ZERO);
    let trunk = tree.segments[1];
    let inside = trunk.a.lerp(trunk.b, 0.5);
    assert!(matches!(tree.sample(inside), Some((density, Part::Wood)) if density > 0.0));
    assert!(
        tree.sample(DVec3::new(0.0, tree.max.y + 5.0, 0.0))
            .is_none()
    );
    // A root segment well away from the trunk's flare.
    let root = tree
        .segments
        .iter()
        .filter(|segment| segment.part == Part::Root && segment.ra > 0.05)
        .find(|segment| (segment.a - tree.origin).with_y(0.0).length() > 2.0)
        .map(|segment| segment.a.lerp(segment.b, 0.5));
    assert!(
        root.and_then(|point| tree.sample(point))
            .is_some_and(|(density, part)| density > 0.0 && part == Part::Root)
    );
}
