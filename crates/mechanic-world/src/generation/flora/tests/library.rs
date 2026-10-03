use super::super::SpeciesSpec;

const VALID: &str = r#"Flora(species: [Species(
    name: "test", height: (5, 8), width: 0.5, girth: 0.03, stems: (1, 2), crown_base: 0.2,
    dominance: 0.5, split_chance: 0.5, split_count: (1, 3), split_angle: 40,
    tropism: 0.0, wobble: 0.2,
    foliage: (look: "leaves", size: 0.4, density: 0.6),
    bark: "bark",
    roots: (spread: 1.0, depth: 1.0),
)])"#;

#[test]
fn embedded_library_holds_every_reference_species() {
    let names = SpeciesSpec::embedded()
        .into_iter()
        .map(|species| species.name)
        .collect::<Vec<_>>();
    for name in ["spruce", "oak", "birch", "willow", "bamboo", "poplar"] {
        assert!(names.iter().any(|known| known == name), "missing {name}");
    }
}

#[test]
fn a_valid_species_parses_with_its_angle_in_radians() {
    let species = SpeciesSpec::library(VALID).expect("valid");
    assert!((species[0].split_angle - 40.0_f64.to_radians()).abs() < 1.0e-12);
}

#[test]
fn out_of_range_genomes_are_rejected() {
    for (from, to) in [
        ("height: (5, 8)", "height: (8, 5)"),
        ("stems: (1, 2)", "stems: (0, 2)"),
        ("split_count: (1, 3)", "split_count: (3, 1)"),
        ("dominance: 0.5", "dominance: 1.5"),
        ("tropism: 0.0", "tropism: -2.0"),
        ("density: 0.6", "density: 1.2"),
        ("split_angle: 40", "split_angle: 200"),
    ] {
        let text = VALID.replace(from, to);
        assert!(SpeciesSpec::library(&text).is_err(), "accepted {to}");
    }
}

#[test]
fn a_repeated_species_name_is_rejected() {
    let body = VALID
        .trim_start_matches("Flora(species: [")
        .trim_end_matches("])");
    let text = format!("Flora(species: [{body}, {body}])");
    assert!(SpeciesSpec::library(&text).is_err());
}
