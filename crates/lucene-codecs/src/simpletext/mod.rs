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
//! - `postings`/`impacts` hand out the closed enums
//!   [`fields_reader::SimpleTextPostings`] and
//!   [`fields_reader::SimpleTextImpactsEnum`] (Java's `SlowImpactsEnum` for
//!   `docFreq <= 8` is the `Slow` variant) instead of `PostingsEnum`
//!   subclasses; on a field with no terms (`TermsEnum.EMPTY`) they are
//!   [`Error::IllegalState`], as Java's.

pub mod fields_reader;
pub mod fields_writer;
pub mod skip;
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
    /// Java's `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// Java's `IllegalStateException`.
    #[error("{0}")]
    IllegalState(&'static str),
    /// Java's `UnsupportedOperationException`.
    #[error("{0}")]
    Unsupported(&'static str),
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
