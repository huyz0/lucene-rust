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
    // Java: SimpleTextLiveDocsFormat.readLiveDocs
    let mut pos = 0usize;
    let mut scratch = Vec::new();
    let size = parse_int(read_value(data, &mut pos, &mut scratch, SIZE)?)?;
    let size = usize::try_from(size)
        .ok()
        .filter(|_| size <= max_doc)
        .ok_or_else(|| corrupt(format!("invalid live docs size {size} (maxDoc={max_doc})")))?;
    // ALLOC: `size` was just checked against the caller's `max_doc`, the
    // segment's document count, not a number the file chose.
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
    // Java: SimpleTextLiveDocsFormat.writeLiveDocs
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bits() -> FixedBitSet {
        let mut b = FixedBitSet::new(5);
        b.set(0);
        b.set(3);
        b.set(4);
        b
    }

    fn reseal(body: &str) -> Vec<u8> {
        let mut out = body.as_bytes().to_vec();
        util::write_checksum(&mut out);
        out
    }

    #[test]
    fn round_trip_and_text() {
        let out = write(&bits());
        let text = String::from_utf8(out.clone()).unwrap();
        assert!(text.starts_with("size 5\n  doc 0\n  doc 3\n  doc 4\nEND\nchecksum "));
        let back = read(&out, 5).unwrap();
        assert_eq!(back.len(), 5);
        assert!((0..5).all(|i| back.get(i) == bits().get(i)));
        assert_eq!(write(&back), out);
        assert_eq!(live_docs_file_name("_0", 1).as_deref(), Some("_0_1.liv"));
        assert_eq!(live_docs_file_name("_0", 36).as_deref(), Some("_0_10.liv"));
        assert_eq!(live_docs_file_name("_0", 0).as_deref(), Some("_0.liv"));
        assert_eq!(live_docs_file_name("_0", -1), None);
    }

    #[test]
    fn damaged_files_are_refused() {
        for body in [
            "size 6\n  doc 0\nEND\n",
            "size -1\nEND\n",
            "size x\nEND\n",
            "size 5\n  doc 5\nEND\n",
            "size 5\n  doc -1\nEND\n",
            "size 5\n  dok 1\nEND\n",
            "size 5\n  doc +1\nEND\n",
            "sise 5\nEND\n",
        ] {
            assert!(read(&reseal(body), 5).is_err(), "{body}");
        }
        let out = write(&bits());
        assert!(read(&out[..out.len() - 3], 5).is_err());
        let mut flipped = out.clone();
        flipped[8] = b'1';
        assert!(read(&flipped, 5).is_err());
        assert!(read(&reseal("size 5\n  doc 1\n"), 5).is_err());
    }
}
