//! Workspace task runner.
//!
//! `cargo xtask <task>` is the single entry point for every check a change has
//! to pass. CI runs `cargo xtask ci`, so a green local run means a green
//! pipeline. Run `cargo xtask help` for the task list.

mod consistency;
mod run;

use std::{path::PathBuf, process::ExitCode};

use run::{Failure, cargo, python};

/// One runnable task: its command-line name, a one-line summary, and its body.
struct Task {
    name: &'static str,
    summary: &'static str,
    run: fn(&[String]) -> Result<(), Failure>,
}

/// Every task, in the order `ci` runs them.
const TASKS: &[Task] = &[
    Task {
        name: "fmt",
        summary: "verify formatting (pass `--fix` to apply it)",
        run: fmt,
    },
    Task {
        name: "consistency",
        summary: "verify the repository conventions that no compiler checks",
        run: |_| consistency::check(&workspace_root()),
    },
    Task {
        name: "lint",
        summary: "clippy on every target with warnings denied",
        run: |extra| {
            cargo(
                &["clippy", "--workspace", "--all-targets"],
                extra,
                &["--", "-D", "warnings"],
            )
        },
    },
    Task {
        name: "test",
        summary: "every workspace test; never stops at the first failure",
        run: |extra| cargo(&["test", "--workspace", "--no-fail-fast"], extra, &[]),
    },
    Task {
        name: "doc",
        summary: "build the API docs with rustdoc warnings denied",
        run: doc,
    },
    Task {
        name: "wgsl",
        summary: "parse and validate the compute kernels without a GPU",
        run: |extra| cargo(&["test", "-p", "mechanic-gpu", "--lib", "wgsl"], extra, &[]),
    },
    Task {
        name: "scripts-test",
        summary: "run the Python regression tests under scripts/",
        run: scripts_test,
    },
    Task {
        name: "budgets",
        summary: "enforce the app's wall-clock test budgets, run alone and serially",
        run: budgets,
    },
    Task {
        name: "bench-smoke",
        summary: "run the quick headless benchmark scenario",
        run: |extra| {
            cargo(
                &["run", "-p", "mechanic-bench", "--", "--scenario", "smoke"],
                extra,
                &[],
            )
        },
    },
];

/// Tasks `ci` skips: `wgsl` is a subset of `test`, `budgets` is meaningless on a
/// shared runner, and `bench-smoke` needs a real adapter.
const NOT_IN_CI: &[&str] = &["wgsl", "budgets", "bench-smoke"];

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "help".to_owned());
    let extra: Vec<String> = arguments.collect();
    let outcome = match name.as_str() {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "ci" => ci(),
        _ => match TASKS.iter().find(|task| task.name == name) {
            Some(task) => (task.run)(&extra),
            None => Err(Failure::new(format!(
                "unknown task `{name}`; run `cargo xtask help`"
            ))),
        },
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("xtask: {failure}");
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!("usage: cargo xtask <task> [extra arguments passed to the underlying command]\n");
    for task in TASKS {
        println!("  {:<13} {}", task.name, task.summary);
    }
    println!(
        "  {:<13} every task above except {}",
        "ci",
        NOT_IN_CI.join(", ")
    );
}

/// Runs every CI task and reports all failures rather than only the first.
fn ci() -> Result<(), Failure> {
    let mut failed = Vec::new();
    for task in TASKS.iter().filter(|task| !NOT_IN_CI.contains(&task.name)) {
        println!("\n=== xtask {} ===", task.name);
        if let Err(failure) = (task.run)(&[]) {
            eprintln!("xtask: {failure}");
            failed.push(task.name);
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(Failure::new(format!("failed tasks: {}", failed.join(", "))))
    }
}

fn fmt(extra: &[String]) -> Result<(), Failure> {
    if extra.iter().any(|argument| argument == "--fix") {
        cargo(&["fmt", "--all"], &[], &[])
    } else {
        cargo(&["fmt", "--all", "--", "--check"], extra, &[])
    }
}

fn doc(extra: &[String]) -> Result<(), Failure> {
    run::cargo_with_env(
        &[
            "doc",
            "--workspace",
            "--no-deps",
            "--document-private-items",
        ],
        extra,
        &[("RUSTDOCFLAGS", "-D warnings")],
    )
}

/// The tests that carry an interactive budget, by name filter.
const BUDGET_TESTS: &[&str] = &["fast_volume_path", "stays_within_one_frame"];

fn budgets(extra: &[String]) -> Result<(), Failure> {
    let mut arguments = vec!["test", "-p", "mechanic-app", "--"];
    arguments.extend(BUDGET_TESTS);
    arguments.push("--test-threads=1");
    run::cargo_with_env(&arguments, extra, &[("MECHANIC_TIMING_TESTS", "1")])
}

fn scripts_test(extra: &[String]) -> Result<(), Failure> {
    let scripts = workspace_root().join("scripts");
    let mut tests: Vec<PathBuf> = std::fs::read_dir(&scripts)
        .map_err(|error| Failure::new(format!("cannot list {}: {error}", scripts.display())))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let is_python = path.extension().is_some_and(|extension| extension == "py");
            let is_test = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("test-"));
            is_python && is_test
        })
        .collect();
    tests.sort();
    if tests.is_empty() {
        return Err(Failure::new("no scripts/test-*.py found".to_owned()));
    }
    for test in tests {
        python(&test, extra)?;
    }
    Ok(())
}

/// The workspace root: two levels above this crate's manifest.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("xtask lives at <workspace>/crates/xtask")
        .to_path_buf()
}
