//! Repository conventions that no compiler or linter enforces.
//!
//! Each check returns one line per violation. `AGENTS.md` states the rule;
//! this module keeps the tree from drifting away from it.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::run::Failure;

/// One convention: its name in the report and the function that finds violations.
struct Check {
    name: &'static str,
    run: fn(&Path) -> Result<Vec<String>, String>,
}

const CHECKS: &[Check] = &[
    Check {
        name: "Mosaic crates share one revision",
        run: mosaic_revisions_match,
    },
    Check {
        name: "crate manifests take every dependency from the workspace",
        run: dependencies_come_from_workspace,
    },
    Check {
        name: "modules are foo.rs beside foo/, never mod.rs",
        run: no_mod_rs,
    },
    Check {
        name: "modules live where their name says, without #[path]",
        run: no_path_attributes,
    },
];

/// Runs every check and reports all violations together.
pub fn check(root: &Path) -> Result<(), Failure> {
    let mut violations = 0_usize;
    for check in CHECKS {
        match (check.run)(root) {
            Ok(found) if found.is_empty() => println!("ok    {}", check.name),
            Ok(found) => {
                println!("FAIL  {}", check.name);
                for line in &found {
                    println!("        {line}");
                }
                violations += found.len();
            }
            Err(error) => {
                println!("FAIL  {}: {error}", check.name);
                violations += 1;
            }
        }
    }
    if violations == 0 {
        Ok(())
    } else {
        Err(Failure::new(format!(
            "{violations} consistency violation(s)"
        )))
    }
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

/// Paths of every `crates/*/Cargo.toml`, sorted.
fn crate_manifests(root: &Path) -> Result<Vec<PathBuf>, String> {
    let crates = root.join("crates");
    let mut manifests: Vec<_> = fs::read_dir(&crates)
        .map_err(|error| format!("cannot list {}: {error}", crates.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("Cargo.toml"))
        .filter(|manifest| manifest.is_file())
        .collect();
    manifests.sort();
    Ok(manifests)
}

/// Every `.rs` file under `crates/`, sorted, skipping build output.
fn rust_sources(root: &Path) -> Result<Vec<PathBuf>, String> {
    fn walk(directory: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
        let entries = fs::read_dir(directory)
            .map_err(|error| format!("cannot list {}: {error}", directory.display()))?;
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "target") {
                    walk(&path, found)?;
                }
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                found.push(path);
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(&root.join("crates"), &mut found)?;
    found.sort();
    Ok(found)
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn no_mod_rs(root: &Path) -> Result<Vec<String>, String> {
    Ok(rust_sources(root)?
        .iter()
        .filter(|path| path.file_name().is_some_and(|name| name == "mod.rs"))
        .map(|path| relative(root, path))
        .collect())
}

fn no_path_attributes(root: &Path) -> Result<Vec<String>, String> {
    // Cargo discovers `src/bin/<name>.rs` only as a single file, so a binary
    // with submodules has to point at them. Everything else follows its name.
    let binaries = root.join("crates/mechanic-bench/src/bin");
    let mut violations = Vec::new();
    for path in rust_sources(root)? {
        if path.starts_with(&binaries) {
            continue;
        }
        for (index, line) in read(&path)?.lines().enumerate() {
            if line.trim_start().starts_with("#[path") {
                violations.push(format!("{}:{}", relative(root, &path), index + 1));
            }
        }
    }
    Ok(violations)
}

fn mosaic_revisions_match(root: &Path) -> Result<Vec<String>, String> {
    let manifest = read(&root.join("Cargo.toml"))?;
    let revisions: Vec<(&str, &str)> = manifest
        .lines()
        .filter(|line| line.starts_with("mosaic"))
        .filter_map(|line| {
            let (name, rest) = line.split_once('=')?;
            let revision = rest.split_once("rev = \"")?.1.split('"').next()?;
            Some((name.trim(), revision))
        })
        .collect();
    let Some(&(_, expected)) = revisions.first() else {
        return Err("no pinned Mosaic crates found in the workspace manifest".to_owned());
    };
    Ok(revisions
        .iter()
        .filter(|(_, revision)| *revision != expected)
        .map(|(name, revision)| format!("{name} is pinned to {revision}, expected {expected}"))
        .collect())
}

fn dependencies_come_from_workspace(root: &Path) -> Result<Vec<String>, String> {
    let mut violations = Vec::new();
    for manifest in crate_manifests(root)? {
        let text = read(&manifest)?;
        let mut in_dependencies = false;
        for (index, line) in text.lines().enumerate() {
            if line.starts_with('[') {
                in_dependencies = line.trim_end_matches(']').ends_with("dependencies");
                continue;
            }
            // Continuation lines of a multi-line table are indented or closing
            // brackets; only a line opening a dependency names its source.
            let opens_dependency = line
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_alphanumeric() || first == '_');
            if in_dependencies && opens_dependency && !line.contains("workspace = true") {
                violations.push(format!(
                    "{}:{}: {line}",
                    relative(root, &manifest),
                    index + 1
                ));
            }
        }
    }
    Ok(violations)
}
