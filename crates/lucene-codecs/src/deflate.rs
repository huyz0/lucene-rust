//! Port of the decode half of `org.apache.lucene.codecs.lucene90.
//! DeflateWithPresetDictCompressionMode` (the encode half is
//! [`crate::deflater`], a port of the zlib `deflate` Java's `Deflater` runs) -- raw DEFLATE (no zlib header, no
//! Adler-32 trailer, matching Java's `Inflater(true)`), used by
//! [`crate::stored_fields`]'s `Mode.BEST_COMPRESSION`. Structurally the same
//! preset-dictionary scheme as [`crate::lz4`] (a dictionary unit, then
//! fixed-size sub-blocks that may back-reference into it).
//!
//! Unlike LZ4 (self-terminating from a plain output-length count alone),
//! DEFLATE needs to be told exactly how many *compressed* bytes to feed the
//! inflator, so [`decompress`] takes `compressed_len` explicitly rather than
//! reading it from `input` itself. In this format that length sits
//! immediately before its own unit's bytes (`doDecompress` reads it there),
//! unlike LZ4's, which is batched with every other unit's up front -- see
//! `stored_fields::decompress_unit`'s doc comment for both shapes.
//!
//! Decoded by [`lucene_util::inflate`], which writes straight into
//! `dest[d_off..]` and lets back-references reach the dictionary bytes
//! already sitting before `d_off` -- the same trick [`crate::lz4::decompress`]
//! uses.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::{Error, Result};

use lucene_util::inflate::Inflater;

thread_local! {
    static INFLATER: std::cell::RefCell<Inflater> = std::cell::RefCell::new(Inflater::default());
}

/// Decompresses `compressed_len` raw-DEFLATE bytes from `input` into
/// `dest[d_off..d_off+decompressed_len]`. Back-references may reach earlier
/// into `dest` than `d_off` (a preset dictionary).
pub(crate) fn decompress(
    input: &mut SliceInput<'_>,
    compressed_len: usize,
    decompressed_len: usize,
    dest: &mut [u8],
    d_off: usize,
) -> Result<usize> {
    let dest_end = d_off
        .checked_add(decompressed_len)
        .ok_or(Error::Eof { offset: d_off })?;
    if dest_end > dest.len() {
        return Err(Error::Eof { offset: dest_end });
    }
    if compressed_len == 0 {
        if decompressed_len != 0 {
            return Err(Error::Corrupted(
                "deflate unit has zero compressed length but nonzero decompressed length"
                    .to_string(),
            ));
        }
        return Ok(d_off);
    }

    // Even when `decompressed_len == 0`, `compressed_len` bytes were still
    // written to `input` by the writer (a real DEFLATE stream always emits
    // at least a final-block marker, even for zero-length content -- see
    // `crate::deflater`'s `deflate_slow`) and MUST still be consumed here:
    // `input` is shared across every unit in a chunk ([`crate::stored_fields
    // ::decompress_unit`] decodes a dictionary then several sub-blocks off
    // the same reader), so skipping these bytes would desync every unit
    // that follows.
    if compressed_len > input.remaining() {
        return Err(Error::Corrupted(format!(
            "deflate compressed length {compressed_len} exceeds {} remaining bytes",
            input.remaining()
        )));
    }
    // Borrowed straight out of the input -- no per-unit copy. `remaining`
    // was checked above, so the end cannot pass the input's end.
    let at = input.position();
    // ARITH: `compressed_len <= remaining() == len - at`.
    #[allow(clippy::arithmetic_side_effects)]
    let compressed = input.slice(at, at + compressed_len)?;
    input.skip(compressed_len)?;
    if decompressed_len == 0 {
        return Ok(d_off);
    }

    // One decoder per thread: its tables are rebuilt per block, never
    // reallocated.
    INFLATER
        .with(|cell| {
            cell.borrow_mut()
                .inflate(compressed, dest, d_off, decompressed_len)
        })
        .map_err(|e| Error::Corrupted(e.to_string()))?;
    Ok(dest_end)
}

#[cfg(test)]
mod tests;
