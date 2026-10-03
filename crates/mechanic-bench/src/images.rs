//! Writing the pictures that preview binaries emit.

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Writes 8-bit RGB pixels, row by row, as a PNG.
///
/// # Errors
///
/// Returns any file or encoding error, or dimensions that do not fit a PNG.
pub fn write_png(
    path: &Path,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(path)?),
        u32::try_from(width)?,
        u32::try_from(height)?,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgb)?;
    Ok(())
}
