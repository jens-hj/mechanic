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
    Check {
        name: "library roots hold only docs, mod lines, and re-exports",
        run: library_roots_define_no_items,
    },
    Check {
        name: "tests live in a tests module, never in *_tests.rs",
        run: no_tests_suffix_files,
    },
    Check {
        name: "lint suppressions are #[expect], or #[allow] with a reason",
        run: no_bare_allow,
    },
    Check {
        name: "docs/environment.md lists every MECHANIC_* variable the code reads",
        run: environment_variables_are_documented,
    },
    Check {
        name: "the app names its environment variables only in env.rs",
        run: app_environment_names_live_in_the_registry,
    },
    Check {
        name: "README.md lists every benchmark scenario",
        run: scenarios_are_documented,
    },
    Check {
        name: "every vendored crate is described in vendor/README.md",
        run: vendored_crates_are_described,
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

fn library_roots_define_no_items(root: &Path) -> Result<Vec<String>, String> {
    const ITEMS: [&str; 9] = [
        "fn ", "struct ", "enum ", "const ", "static ", "type ", "trait ", "impl ", "impl<",
    ];
    let mut violations = Vec::new();
    for manifest in crate_manifests(root)? {
        let library = manifest.with_file_name("src").join("lib.rs");
        if !library.is_file() {
            continue;
        }
        for (index, line) in read(&library)?.lines().enumerate() {
            let declaration = line
                .trim_start_matches("pub(crate) ")
                .trim_start_matches("pub ");
            let opens_inline_module = declaration.starts_with("mod ") && line.ends_with('{');
            if ITEMS.iter().any(|item| declaration.starts_with(item)) || opens_inline_module {
                violations.push(format!(
                    "{}:{}: {line}",
                    relative(root, &library),
                    index + 1
                ));
            }
        }
    }
    Ok(violations)
}

fn no_tests_suffix_files(root: &Path) -> Result<Vec<String>, String> {
    Ok(rust_sources(root)?
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_tests.rs"))
        })
        .map(|path| relative(root, path))
        .collect())
}

fn no_path_attributes(root: &Path) -> Result<Vec<String>, String> {
    let mut violations = Vec::new();
    for path in rust_sources(root)? {
        for (index, line) in read(&path)?.lines().enumerate() {
            if line.trim_start().starts_with("#[path") {
                violations.push(format!("{}:{}", relative(root, &path), index + 1));
            }
        }
    }
    Ok(violations)
}

fn no_bare_allow(root: &Path) -> Result<Vec<String>, String> {
    let mut violations = Vec::new();
    for path in rust_sources(root)? {
        let text = read(&path)?;
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let attribute = line.trim_start();
            if !(attribute.starts_with("#[allow(") || attribute.starts_with("#![allow(")) {
                continue;
            }
            // The attribute may wrap; it ends at the first line closing it.
            let end = lines[index..]
                .iter()
                .position(|candidate| candidate.contains(")]"))
                .map_or(index, |offset| index + offset);
            if !lines[index..=end]
                .iter()
                .any(|part| part.contains("reason ="))
            {
                violations.push(format!("{}:{}", relative(root, &path), index + 1));
            }
        }
    }
    Ok(violations)
}

/// Every `"MECHANIC_…"` string literal in the crates' sources.
fn environment_variables_are_documented(root: &Path) -> Result<Vec<String>, String> {
    let documented = read(&root.join("docs/environment.md"))?;
    let mut missing = std::collections::BTreeSet::new();
    for path in rust_sources(root)? {
        if path.starts_with(root.join("crates/xtask")) {
            continue;
        }
        for literal in read(&path)?.split('"').skip(1).step_by(2) {
            let is_variable = literal.starts_with("MECHANIC_")
                && literal
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            if is_variable && !documented.contains(&format!("`{literal}`")) {
                missing.insert(format!("{literal} ({})", relative(root, &path)));
            }
        }
    }
    Ok(missing.into_iter().collect())
}

fn app_environment_names_live_in_the_registry(root: &Path) -> Result<Vec<String>, String> {
    let app = root.join("crates/mechanic-app/src");
    let registry = app.join("env.rs");
    let mut violations = Vec::new();
    for path in rust_sources(root)? {
        if !path.starts_with(&app) || path == registry {
            continue;
        }
        for (index, line) in read(&path)?.lines().enumerate() {
            let names_a_variable = line.split('"').skip(1).step_by(2).any(|literal| {
                literal.starts_with("MECHANIC_")
                    && literal
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            });
            if names_a_variable {
                violations.push(format!("{}:{}", relative(root, &path), index + 1));
            }
        }
    }
    Ok(violations)
}

/// Scenario names are the string patterns of `Scenario::parse`.
fn scenarios_are_documented(root: &Path) -> Result<Vec<String>, String> {
    let source = read(&root.join("crates/mechanic-bench/src/main.rs"))?;
    let readme = read(&root.join("README.md"))?;
    let names: Vec<&str> = source
        .lines()
        .filter(|line| line.contains("=> Some(Self::"))
        .filter_map(|line| line.split('"').nth(1))
        .collect();
    if names.is_empty() {
        return Err("found no scenario names in mechanic-bench".to_owned());
    }
    Ok(names
        .into_iter()
        .filter(|name| !readme.contains(&format!("--scenario {name}")))
        .map(|name| format!("--scenario {name}"))
        .collect())
}

fn vendored_crates_are_described(root: &Path) -> Result<Vec<String>, String> {
    let vendor = root.join("vendor");
    let described = read(&vendor.join("README.md"))?;
    let entries = fs::read_dir(&vendor)
        .map_err(|error| format!("cannot list {}: {error}", vendor.display()))?;
    let mut missing: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| !described.contains(&format!("`{name}`")))
        .collect();
    missing.sort();
    Ok(missing)
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
