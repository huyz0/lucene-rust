//! `SimpleTextLiveDocsFormat`: a segment's live documents as one `.liv`
//! text file per deletion generation (`_N_<gen base 36>.liv`).
//!
//! ```text
//! size <maxDoc>
//!   doc <docID>          every *live* document, ascending
//! END
//! checksum <20 digits>
//! ```
//!
//! What differs from Java: Java's reader sets each listed doc in a
//! `java.util.BitSet`, which grows past `size` (and throws
//! `IndexOutOfBoundsException` for a negative one); here the live docs are a
//! [`FixedBitSet`] of `size` bits and a doc outside `0..size` is
//! `Corrupted`. `size` itself must not exceed the caller's `maxDoc` (Java
//! does not check it), which bounds the allocation by the segment rather
//! than by a number read off the file.

use lucene_util::fixed_bit_set::FixedBitSet;

use super::util::{self, parse_int, read_line, read_value, tail};
use super::{corrupt, Result};

/// `SimpleTextLiveDocsFormat.LIVEDOCS_EXTENSION`.
pub const LIVEDOCS_EXTENSION: &str = "liv";
/// `SIZE`.
pub const SIZE: &[u8] = b"size ";
/// `DOC`.
pub const DOC: &[u8] = b"  doc ";
/// `END`.
pub const END: &[u8] = b"END";

/// `IndexFileNames.fileNameFromGeneration(segment, "liv", delGen)`.
pub fn live_docs_file_name(segment: &str, del_gen: i64) -> Option<String> {
    super::file_name_from_generation(segment, LIVEDOCS_EXTENSION, del_gen)
}

/// `SimpleTextLiveDocsFormat.readLiveDocs`: the `.liv` file's live docs
/// (`SimpleTextBits`: `length()` is the file's `size`).
///
/// # Errors
/// `Corrupted` for a line without its prefix, a number that does not parse,
/// a `size` outside `0..=max_doc`, a doc outside `0..size` or a bad
/// checksum; `Eof` for a short file.
pub fn read(data: &[u8], max_doc: i32) -> Result<FixedBitSet> {
    let mut pos = 0usize;
    let mut scratch = Vec::new();
    let size = parse_int(read_value(data, &mut pos, &mut scratch, SIZE)?)?;
    let size = usize::try_from(size)
        .ok()
        .filter(|_| size <= max_doc)
        .ok_or_else(|| corrupt(format!("invalid live docs size {size} (maxDoc={max_doc})")))?;
    let mut bits = FixedBitSet::new(size);
    read_line(data, &mut pos, &mut scratch)?;
    while scratch != END {
        let v = tail(&scratch, DOC).ok_or_else(|| {
            corrupt(format!(
                "SimpleText failure: expected doc line but got {}",
                String::from_utf8_lossy(&scratch)
            ))
        })?;
        let docid = parse_int(v)?;
        let index = usize::try_from(docid)
            .ok()
            .filter(|&d| d < bits.len())
            .ok_or_else(|| corrupt(format!("live doc {docid} outside 0..{size}")))?;
        bits.set(index);
        read_line(data, &mut pos, &mut scratch)?;
    }
    util::check_footer(data, 0, &mut pos)?;
    Ok(bits)
}

/// `SimpleTextLiveDocsFormat.writeLiveDocs`: the `.liv` file for `bits`
/// (a set bit is a live document).
pub fn write(bits: &FixedBitSet) -> Vec<u8> {
    let mut out = Vec::new();
    util::write_line(&mut out, SIZE, bits.len().to_string().as_bytes());
    for i in 0..bits.len() {
        if bits.get(i) {
            util::write_line(&mut out, DOC, i.to_string().as_bytes());
        }
    }
    util::write(&mut out, END);
    util::write_newline(&mut out);
    util::write_checksum(&mut out);
    out
}
