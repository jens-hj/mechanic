//! A fast hash for maps keyed by cells: their keys are a few small
//! integers, which a keyed hash guarding against collision attacks only
//! slows down.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Maps keyed by cells or bricks.
pub(super) type CellMap<K, V> = HashMap<K, V, BuildHasherDefault<CellHasher>>;

/// Multiply-and-rotate hashing of whole words (the `FxHash` of rustc).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CellHasher(u64);

impl CellHasher {
    const fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for CellHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[expect(clippy::cast_sign_loss, reason = "hashing keeps the bits")]
    fn write_i32(&mut self, value: i32) {
        self.add(u64::from(value as u32));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}
