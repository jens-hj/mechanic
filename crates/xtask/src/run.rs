//! Child-process helpers shared by every task.

use std::{
    fmt,
    path::Path,
    process::{Command, ExitStatus},
};

/// Why a task failed, already phrased for the terminal.
#[derive(Debug)]
pub struct Failure(String);

impl Failure {
    /// Wraps a ready-to-print message.
    pub fn new(message: String) -> Self {
        Self(message)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Runs `cargo <leading> <extra> <trailing>` with the toolchain that built xtask.
pub fn cargo(leading: &[&str], extra: &[String], trailing: &[&str]) -> Result<(), Failure> {
    let mut command = cargo_command();
    command.args(leading).args(extra).args(trailing);
    finish(command)
}

/// Runs `cargo <leading> <extra>` with additional environment variables.
pub fn cargo_with_env(
    leading: &[&str],
    extra: &[String],
    environment: &[(&str, &str)],
) -> Result<(), Failure> {
    let mut command = cargo_command();
    command.args(leading).args(extra);
    command.envs(environment.iter().copied());
    finish(command)
}

/// Runs one Python script with the platform's interpreter name.
pub fn python(script: &Path, extra: &[String]) -> Result<(), Failure> {
    let interpreter = if cfg!(windows) { "python" } else { "python3" };
    let mut command = Command::new(interpreter);
    command.arg(script).args(extra);
    finish(command)
}

fn cargo_command() -> Command {
    // Cargo exports its own path to subcommands; fall back for direct runs.
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn finish(mut command: Command) -> Result<(), Failure> {
    let rendered = render(&command);
    println!("$ {rendered}");
    let status = command
        .status()
        .map_err(|error| Failure::new(format!("cannot start `{rendered}`: {error}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(Failure::new(format!("`{rendered}` {}", describe(status))))
    }
}

fn render(command: &Command) -> String {
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|part| part.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}

fn describe(status: ExitStatus) -> String {
    status.code().map_or_else(
        || "was terminated by a signal".to_owned(),
        |code| format!("exited with status {code}"),
    )
}
