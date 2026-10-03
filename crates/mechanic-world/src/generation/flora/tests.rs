mod field;
mod genome;
mod library;
mod sweeps;

use bevy_math::DVec3;

use super::{SpeciesSpec, TreeMetrics, TreeModel, grow_tree};

const SEEDS: u64 = 16;

fn preset(name: &str) -> SpeciesSpec {
    SpeciesSpec::embedded()
        .into_iter()
        .find(|species| species.name == name)
        .unwrap_or_else(|| panic!("no preset `{name}`"))
}

fn grove(species: &SpeciesSpec, seeds: u64) -> Vec<(TreeModel, TreeMetrics)> {
    (0..seeds)
        .map(|seed| {
            let tree = grow_tree(species, seed * 7_919 + 3, DVec3::new(5.0, 12.0, -3.0));
            let metrics = TreeMetrics::measure(&tree);
            (tree, metrics)
        })
        .collect()
}

fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .into_iter()
        .fold((0.0, 0.0), |(sum, count), value| (sum + value, count + 1.0));
    sum / count
}
