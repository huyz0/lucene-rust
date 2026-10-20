//! The SimpleText codec (`org.apache.lucene.codecs.simpletext`): every
//! format as human-readable text, one value per line, for debugging and
//! teaching -- never for production.
//!
//! Ported so far:
//! - the postings format (`SimpleTextPostingsFormat`, file `.pst`):
//!   [`fields_writer::SimpleTextFieldsWriter`] (with the line constants
//!   and [`fields_writer::write`]) and
//!   [`fields_reader::SimpleTextFieldsReader`];
//! - its skip lists with impacts ([`skip`]);
//! - the segment info (`.si`, [`segment_info`]), field infos (`.inf`,
//!   [`field_infos`]), live docs (`.liv`, [`live_docs`]), compound file
//!   (`.scf`, [`compound`]) and stored fields (`.fld`,
//!   [`stored_fields_writer`] and [`stored_fields_reader`]) formats, each
//!   with its own line constants and a note on what it does differently;
//! - doc values (`.dat`, [`doc_values_writer`], [`doc_values_reader`], with
//!   `DecimalFormat`'s parsing in [`decimal_format`]), norms (`.len`,
//!   [`norms`]) and term vectors (`.tvc`, [`term_vectors_writer`] and
//!   [`term_vectors_reader`]), each module noting what it does differently;
//! - points (`.dim`/`.dii`, [`points_writer`] with the line constants,
//!   [`bkd_writer`], [`points_reader`], [`bkd_reader`]);
//! - the line, escape and checksum primitives ([`util`]).
//!
//! # The `.pst` file
//!
//! ```text
//! field <name>
//!   term <bytes, \n and \ escaped>
//!     doc <docID>
//!       freq <freq>                   (fields with freqs)
//!       pos <position>                (fields with positions)
//!       startOffset <n>               (fields with offsets)
//!       endOffset <n>
//!         payload <bytes>             (non-empty payloads only)
//!     skipList                        (terms with >= 8 docs; see [`skip`])
//!     ...
//! END
//! checksum <CRC32 of every byte above, 20 digits>
//! ```
//!
//! The reader keeps the whole file in memory and, on a field's first use,
//! walks its lines once into an FST from each term to `((docsStart,
//! skipPointer), (docFreq, totalTermFreq))`; the enums then read the docs
//! lines in place.
//!
//! # What differs from Java
//!
//! - Writer and reader work on bytes in memory, not on `IndexOutput` /
//!   `IndexInput`: [`fields_writer::write`] returns the file,
//!   [`fields_reader::SimpleTextFieldsReader::open`] opens it from its bytes.
//!   The writer takes the flush's in-memory postings
//!   ([`crate::postings_writer::FieldPostingsInput`]) plus dense norms columns,
//!   not a `Fields` + `NormsProducer`.
//! - Every Java `assert` on a line's prefix, and every `NumberFormatException`
//!   or out-of-range doc, is a `Corrupted` error; arithmetic on parsed counts
//!   is checked. Numbers parse from ASCII digits only (Java's
//!   `Character.digit` also takes other Unicode decimal digits). Java's skip
//!   reader's `skipTo` returns the skipped count (or -1);
//!   [`skip::SimpleTextSkipReader::skip_to`] returns nothing (no caller reads
//!   it).
//! - Field names are decoded with `String::from_utf8_lossy`, whose count of
//!   U+FFFD for malformed UTF-8 can differ from Java's decoder; they are
//!   ordered by UTF-16 code unit, as Java's `TreeMap<String, Long>`.
//! - `Fields.size()` (Java: -1), enum reuse (`canReuse`) and
//!   `ramBytesUsed`/`toString` details are not ported.
//! - The `.si`/`.inf`/`.liv`/`.scf`/`.fld` readers and writers take and
//!   return bytes and this crate's models; the `.si` model is
//!   [`segment_info::SimpleTextSegmentInfo`] (`lucene-index`'s
//!   `SegmentInfo` sits above this crate), whose index-sort fields stay
//!   undecoded provider/type/bytes triples. `HashMap`/`HashSet` contents
//!   (diagnostics, attributes, files) keep file order; a repeated key keeps
//!   its first slot and its last value. Java's `assert`s on line prefixes,
//!   counts and sort order, its `NumberFormatException`s,
//!   `NegativeArraySizeException`s, array index errors and the
//!   `IllegalArgumentException` of an out-of-range version are `Corrupted`.
//! - `postings`/`impacts` hand out the closed enums
//!   [`fields_reader::SimpleTextPostings`] and
//!   [`fields_reader::SimpleTextImpactsEnum`] (Java's `SlowImpactsEnum` for
//!   `docFreq <= 8` is the `Slow` variant) instead of `PostingsEnum`
//!   subclasses; on a field with no terms (`TermsEnum.EMPTY`) they are
//!   [`Error::IllegalState`], as Java's.
//! - Points: [`points_writer::SimpleTextPointsWriter::write_field`] takes a
//!   field's `(packedValue, docID)`s in `visitDocValues` order and its
//!   `size()`, and builds the tree only through `SimpleTextBKDWriter.add` and
//!   `finish`, the one path Java's `writeField` takes (its
//!   `writeField(MutablePointTree)`, `OneDimensionBKDWriter` and
//!   `rotateToTree` are unreachable from this codec and not ported). The tree
//!   writer reuses [`crate::bkd_writer`]'s point stores, radix selector and
//!   sorter; a `too many nodes` failure is [`Error::Runtime`]. The readers'
//!   refusals of corrupt input are in [`points_reader`] and [`bkd_reader`].

pub mod bkd_reader;
pub mod bkd_writer;
pub mod compound;
pub mod decimal_format;
pub mod doc_values_reader;
pub mod doc_values_writer;
pub mod field_infos;
pub mod fields_reader;
pub mod fields_writer;
pub mod live_docs;
pub mod norms;
pub mod points_reader;
pub mod points_writer;
pub mod segment_info;
pub mod skip;
pub mod stored_fields_reader;
pub mod stored_fields_writer;
pub mod term_vectors_reader;
pub mod term_vectors_writer;
pub mod util;

#[cfg(test)]
mod tests;

pub use crate::blocktree::SeekStatus;

use crate::field_infos::IndexOptions;

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// `PostingsEnum.NONE`.
pub const NONE: i32 = 0;
/// `PostingsEnum.FREQS`.
pub const FREQS: i32 = 1 << 3;
/// `PostingsEnum.POSITIONS`.
pub const POSITIONS: i32 = FREQS | (1 << 4);
/// `PostingsEnum.OFFSETS`.
pub const OFFSETS: i32 = POSITIONS | (1 << 5);
/// `PostingsEnum.PAYLOADS`.
pub const PAYLOADS: i32 = POSITIONS | (1 << 6);
/// `PostingsEnum.ALL`.
pub const ALL: i32 = OFFSETS | PAYLOADS;

/// Why a SimpleText file could not be written or read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A malformed line, number, footer or checksum; a short file.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// Building a field's terms FST.
    #[error(transparent)]
    Compile(#[from] crate::fst_compiler::CompileError),
    /// Walking a field's terms FST.
    #[error(transparent)]
    Fst(#[from] crate::fst_compiler::FstReadError),
    /// What `new FieldInfo(...)` / `new FieldInfos(infos)` refuse.
    #[error(transparent)]
    FieldInfos(#[from] crate::field_infos::Error),
    /// What a [`crate::stored_fields::StoredFieldVisitor`] returns.
    #[error(transparent)]
    StoredFields(#[from] crate::stored_fields::Error),
    /// Java's `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// Java's `IllegalStateException`.
    #[error("{0}")]
    IllegalState(&'static str),
    /// Java's `UnsupportedOperationException`.
    #[error("{0}")]
    Unsupported(&'static str),
    /// Java's bare `RuntimeException` (a writer's bookkeeping check).
    #[error("{0}")]
    Runtime(String),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `indexOptions.compareTo(DOCS_AND_FREQS) >= 0`.
pub fn subsumes_freqs(options: IndexOptions) -> bool {
    matches!(
        options,
        IndexOptions::DocsAndFreqs | IndexOptions::DocsAndCustomFreqs
    ) || options.subsumes_positions()
}

/// `PostingsEnum.featureRequested(flags, feature)`.
pub fn feature_requested(flags: i32, feature: i32) -> bool {
    (flags & feature) == feature
}

/// `IndexFileNames.segmentFileName(segment, suffix, ext)`.
pub fn segment_file_name(segment: &str, suffix: &str, ext: &str) -> String {
    let mut name = String::from(segment);
    if !suffix.is_empty() {
        name.push('_');
        name.push_str(suffix);
    }
    if !ext.is_empty() {
        name.push('.');
        name.push_str(ext);
    }
    name
}

/// `IndexFileNames.fileNameFromGeneration(segment, ext, gen)`: `None` for
/// `gen == -1` (Java's `null`) and for any other negative `gen` (Java's
/// `assert gen > 0`); no generation suffix for 0; else `_<gen base 36>`.
pub fn file_name_from_generation(segment: &str, ext: &str, generation: i64) -> Option<String> {
    match generation {
        ..0 => None,
        0 => Some(segment_file_name(segment, "", ext)),
        g => Some(segment_file_name(
            segment,
            &lucene_util::base36::to_base36(g),
            ext,
        )),
    }
}

/// A file pointer read off the file (or built by the reader), as an index
/// into its bytes.
fn file_pointer(data: &[u8], fp: i64, what: &str) -> Result<usize> {
    usize::try_from(fp)
        .ok()
        .filter(|&p| p <= data.len())
        .ok_or_else(|| {
            corrupt(format!(
                "{what} {fp} outside the file ({} bytes)",
                data.len()
            ))
        })
}
