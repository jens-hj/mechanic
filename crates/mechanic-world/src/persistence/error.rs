//! Failures opening, reading, or writing a world.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Exact persistence failure with the original file path.
#[derive(Debug, Error)]
pub enum WorldSaveError {
    /// Operating-system random seed generation failed.
    #[error("operating-system random seed generation failed: {0}")]
    Random(getrandom::Error),
    /// Filesystem operation failed.
    #[error("world file {path} could not be read or written: {source}", path = path.display())]
    Io {
        /// Exact affected path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// RON encoding failed.
    #[error("world file {path} could not be encoded: {source}", path = path.display())]
    Encode {
        /// Exact affected path.
        path: PathBuf,
        /// Encoding error.
        #[source]
        source: ron::Error,
    },
    /// RON decoding failed.
    #[error("world file {path} is corrupt: {source}", path = path.display())]
    Decode {
        /// Exact corrupt path.
        path: PathBuf,
        /// Decoding error.
        #[source]
        source: Box<ron::error::SpannedError>,
    },
    /// Edited-brick binary payload is corrupt.
    #[error("edited terrain file {path} is corrupt: {source}", path = path.display())]
    Brick {
        /// Exact corrupt path.
        path: PathBuf,
        /// Binary decoding failure.
        #[source]
        source: crate::BrickDecodeError,
    },
    /// World or generator version is not understood.
    #[error("world file {path} has an unsupported format or generator version", path = path.display())]
    UnsupportedVersion {
        /// Exact unsupported path.
        path: PathBuf,
    },
    /// Frozen state does not match the published creation or a valid target.
    #[error("world file {path} has invalid frozen creation state: {message}", path = path.display())]
    InvalidFrozenCreation {
        /// Exact affected manifest.
        path: PathBuf,
        /// Invalid frozen-state invariant.
        message: &'static str,
    },
    /// Deletion target was not a direct child of this store.
    #[error("refusing world path outside the store: {path}", path = path.display())]
    OutsideStore {
        /// Refused path.
        path: PathBuf,
    },
    /// Current-format corrupt data is preserved for explicit recovery.
    #[error("current world file {path} is corrupt and was left untouched: {message}", path = path.display())]
    CorruptCurrent {
        /// Exact corrupt file.
        path: PathBuf,
        /// Original inspection detail.
        message: String,
    },
}
