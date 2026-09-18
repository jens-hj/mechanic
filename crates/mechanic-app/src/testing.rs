//! Fixtures shared by tests across the crate.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

/// A directory of its own per test, removed on drop, so tests that touch the
/// filesystem stay parallel-safe without mutating the process environment.
pub(crate) struct TempDir(pub(crate) PathBuf);

impl TempDir {
    /// Reserves a unique path under the system temporary directory. Nothing
    /// exists there yet, for code that must create its own directory.
    pub(crate) fn new(label: &str) -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "mechanic-{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    /// [`TempDir::new`], with the directory already created.
    pub(crate) fn created(label: &str) -> Self {
        let directory = Self::new(label);
        std::fs::create_dir_all(&directory.0).expect("temporary directory");
        directory
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
