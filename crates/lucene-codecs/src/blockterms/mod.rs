//! Port of lucene-codecs' `org.apache.lucene.codecs.blockterms`: a terms
//! dictionary that writes a field's terms in blocks, one block between two
//! *index terms*, and a pluggable terms index that finds the block a term
//! would be in -- every Nth term held in packed arrays
//! ([`FixedGapTermsIndexWriter`]/[`FixedGapTermsIndexReader`], with `ord()`
//! and `seekExact(ord)`), or the terms an [`IndexTermSelector`] picks, held
//! in an FST ([`VariableGapTermsIndexWriter`]/[`VariableGapTermsIndexReader`],
//! no ords). Over `Lucene104PostingsWriter`'s postings.
//!
//! Lucene 10.5.0 ships no postings format over this package (the old
//! `LuceneFixedGap`/`LuceneVarGap*` formats are in the test framework); a
//! codec assembles one from a [`BlockTermsWriter`] and a terms index writer,
//! as [`write()`] does.
//!
//! # The files
//!
//! All under the group's codec suffix (`_0_BlockTermsFixedGap_0.tib`),
//! beside the postings' `.doc`/`.pos`/`.pay`/`.psm`:
//!
//! ```text
//! .tib  IndexHeader("BlockTermsWriter", 4)
//!       IndexHeader("Lucene104PostingsWriterTerms", 0), vInt(BLOCK_SIZE)
//!       per field: blocks, then vInt(0)
//!         block: vInt(termCount) vInt(commonPrefix)
//!                vInt(n) suffixes: per term vInt(len - commonPrefix), bytes
//!                vInt(n) stats: per term vInt(docFreq)
//!                               [vLong(totalTermFreq - docFreq)]
//!                vInt(n) metadata: per term Lucene104 encodeTerm, the first
//!                                  absolute, the others against the one before
//!         - commonPrefix: the shortest prefix any term of the block shares
//!           with the last term of the block before it (0 for the first)
//!       vInt(numFields), per field with terms: vInt(number) vLong(numTerms)
//!         vLong(termsStartPointer) [vLong(sumTotalTermFreq)] vLong(sumDocFreq)
//!         vInt(docCount)
//!       long(dirStart), Footer
//! .tii  IndexHeader("FixedGapTermsIndex", 4)        (fixed gap)
//!       vInt(interval) vInt(PackedInts.VERSION_CURRENT) vInt(4096)
//!       per field: each index term's sort-key prefix against the term before
//!         it (none for the first), the terms' .tib offsets from the field's
//!         start (MonotonicBlockPacked), the prefixes' running end offsets,
//!         0 first (MonotonicBlockPacked)
//!       vInt(numFields), per field: vInt(number) vInt(numIndexTerms)
//!         vLong(termsStart) vLong(indexStart) vLong(packedIndexStart)
//!         vLong(packedOffsetsStart)
//!       long(dirStart), Footer
//! .tmv  IndexHeader("VariableGapTermsMeta", 4)      (variable gap)
//!       per field: int(number) vLong(.tiv offset) FST metadata
//!       int(-1), Footer
//! .tiv  IndexHeader("VariableGapTermsIndex", 4)
//!       per field: an FST (PositiveIntOutputs) body from each index term's
//!         prefix that tells it from the term before it -- and the empty
//!         term -- to its block's .tib file pointer
//!       Footer
//! ```
//!
//! A seek asks the terms index for the last index term `<=` the target,
//! loads that block and scans it, comparing the target against the
//! block's common prefix first and then each term's suffix in place.
//!
//! # What differs from Java
//!
//! - Module-level, like the `memory`, `blocktreeords` and `uniformsplit`
//!   formats: the files go in and come out as bytes, this port's
//!   `IndexWriter` routes no field here and `DirectoryReader` does not open
//!   these files. Fields are written in the order they are given (Java:
//!   field name order, which is what a flush or a merge passes).
//! - `TermsIndexWriterBase.addField` returns a `FieldWriter` in Java; here the
//!   index writer holds the field being written ([`TermsIndexWriterBase`]'s
//!   `check_index_term`/`add`/`finish` act on it), as fields are written one
//!   at a time. The index readers are one enum ([`TermsIndexReader`]) and
//!   `FieldIndexEnum.next()` returns `None` where Java returns `-1`.
//! - A block's three blobs are borrowed from the `.tib` in place (Java copies
//!   each into a reused `byte[]`).
//! - Where Java asserts, reads past a blob or misbehaves on bytes it would not
//!   write (a block whose common prefix is longer than the term it extends, a
//!   suffix past its blob, an index offset outside the index's bytes, a
//!   seek on a field the index lacks), this port returns
//!   [`Error::Store`] (`Corrupted`) or, for the missing index,
//!   [`Error::IllegalState`] with Java's message.
//! - `ord()`/`seekExact(ord)` without ord support (the variable-gap index)
//!   return [`Error::Unsupported`]; so does `postings` on an enum that is not
//!   on a term (Java hands out an enum over garbage). `next()` after `END`
//!   reads past the last block's suffixes in Java (or, from
//!   `termBlockOrd == 0`, past the end marker); here `next_block` resets
//!   the block ord at the end marker and `next()` returns `None` again. A `seekExact(ord)` past `size()` is
//!   [`Error::IllegalArgument`] (Java: an `assert`).
//! - Stage-3 optimisations (`cx_bterms_seek`): `FixedGapTermsIndexReader`
//!   decodes its two packed arrays at open into plain arrays of prefix
//!   ranges and block pointers (Java reads the packed bits per probe), so a
//!   bad prefix offset, or more index terms than prefix bytes, is refused at
//!   open, at 24 bytes per index term of RAM where Java's monotonic packed
//!   arrays take a few bits; a block's one-byte `vInt`s are read without an input.
//! - `.psm` is neither read nor validated; the postings enums do not take a
//!   `reuse` instance and `impacts` is not ported (as `blocktreeords`).
//!   `Terms.getMin`/`getMax`, `intersect` (`Terms`' defaults, not this
//!   package's) and the `Accountable` sizes are not ported.

pub mod fixed_gap;
mod reader;
#[cfg(test)]
mod tests;
pub mod variable_gap;
mod writer;

pub use fixed_gap::{FixedGapIndexEnum, FixedGapTermsIndexReader, FixedGapTermsIndexWriter};
pub use reader::{BlockTermsFieldReader, BlockTermsReader, SegmentTermsEnum, TermsIndexReader};
pub use variable_gap::{
    EveryNOrDocFreqTermSelector, EveryNTermSelector, IndexTermSelector, VariableGapFiles,
    VariableGapIndexEnum, VariableGapTermsIndexReader, VariableGapTermsIndexWriter,
};
pub use writer::BlockTermsWriter;

pub use crate::blocktree::SeekStatus;
use crate::blocktree::{POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT, POSTINGS_VERSION_START};
use crate::field_infos::FieldInfo;
use crate::postings::TermMetadata;
use crate::postings_writer::{
    self, Error as PostingsWriterError, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource,
    WriteOptions,
};
use crate::segment_state::SegmentWriteState;

/// `BlockTermsWriter.CODEC_NAME`.
pub const CODEC_NAME: &str = "BlockTermsWriter";
/// `BlockTermsWriter.VERSION_START`.
pub const VERSION_START: i32 = 4;
/// `BlockTermsWriter.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_START;
/// `BlockTermsWriter.TERMS_EXTENSION`.
pub const TERMS_EXTENSION: &str = "tib";

/// What reading or writing this package fails with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Headers, footers, checksums, field metadata and malformed blocks.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A variable-gap index FST that does not decode.
    #[error(transparent)]
    Fst(#[from] crate::fst_compiler::FstReadError),
    /// Building a variable-gap index FST.
    #[error(transparent)]
    Compile(#[from] crate::fst_compiler::CompileError),
    /// The postings writer's input checks.
    #[error(transparent)]
    Write(#[from] PostingsWriterError),
    /// Decoding a term's postings.
    #[error(transparent)]
    Postings(#[from] crate::postings::Error),
    /// Java's `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// Java's `IllegalStateException`.
    #[error("{0}")]
    IllegalState(&'static str),
    /// Java's `UnsupportedOperationException`, and the states Java leaves to
    /// garbage or a `NullPointerException`.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `TermStats`: what a terms index's selector sees of a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermStats {
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq` (`-1` for a field without frequencies, as
    /// `Lucene104PostingsWriter` leaves it).
    pub total_term_freq: i64,
}

/// `BlockTermState` as this dictionary reads it: the statistics, the
/// postings pointers, the term's ord, and where its block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTermState {
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq` (`docFreq` for a field without frequencies).
    pub total_term_freq: i64,
    /// `docStartFP`, `singletonDocID`, `posStartFP`, `payStartFP`,
    /// `lastPosBlockOffset`.
    pub meta: TermMetadata,
    /// `ord` (meaningless without ord support).
    pub ord: i64,
    /// `termBlockOrd`: the term's 1-based index in its block.
    pub term_block_ord: i32,
    /// `blockFilePointer`: the start of the term's block in the `.tib`.
    pub block_file_pointer: usize,
}

impl Default for BlockTermState {
    /// `postingsReader.newTermState()` with `totalTermFreq = ord = -1`, as
    /// `SegmentTermsEnum`'s constructor leaves it.
    fn default() -> Self {
        BlockTermState {
            doc_freq: 0,
            total_term_freq: -1,
            meta: TermMetadata::EMPTY,
            ord: -1,
            term_block_ord: 0,
            block_file_pointer: 0,
        }
    }
}

/// `TermsIndexWriterBase` and its `FieldWriter`: a terms index the
/// [`BlockTermsWriter`] fills field by field, the field being written held
/// by the index writer.
pub trait TermsIndexWriterBase {
    /// The finished index files.
    type Files;

    /// `addField(fieldInfo, termsFilePointer)`: starts a field whose blocks
    /// begin at `terms_file_pointer` in the `.tib`.
    ///
    /// # Errors
    /// What the index fails with.
    fn add_field(&mut self, field: &FieldInfo, terms_file_pointer: i64) -> Result<()>;

    /// `FieldWriter.checkIndexTerm(text, stats)`: whether `text` starts a
    /// block.
    fn check_index_term(&mut self, text: &[u8], stats: TermStats) -> bool;

    /// `FieldWriter.add(text, stats, termsFilePointer)`: records the index
    /// term `text`, whose block starts at `terms_file_pointer`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for terms out of order.
    fn add(&mut self, text: &[u8], stats: TermStats, terms_file_pointer: i64) -> Result<()>;

    /// `FieldWriter.finish(termsFilePointer)`.
    ///
    /// # Errors
    /// What building the field's index fails with.
    fn finish(&mut self, terms_file_pointer: i64) -> Result<()>;

    /// `close()`: the finished files.
    ///
    /// # Errors
    /// What the index fails with.
    fn close(self) -> Result<Self::Files>;
}

/// What [`write()`] produces.
#[derive(Debug, Clone, Default)]
pub struct BlockTermsOutput<F> {
    /// `.doc`, `.psm`, `.pos`, `.pay`.
    pub postings: PostingsFiles,
    /// `.tib`: the blocks and the fields' summaries.
    pub tib: Vec<u8>,
    /// The terms index's files.
    pub index: F,
}

/// A postings format's `fieldsConsumer(state).write(fields, norms)` and
/// `close()`: `new BlockTermsWriter(termsIndexWriter, state, new
/// Lucene104PostingsWriter(state))` over `inputs`.
///
/// # Errors
/// What the postings writer rejects in `inputs`, a field number the
/// segment's `FieldInfos` lacks, and what the terms index fails with.
pub fn write<T: TermSource, I: TermsIndexWriterBase>(
    terms_index_writer: I,
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    state: &SegmentWriteState<'_>,
) -> Result<BlockTermsOutput<I::Files>> {
    let options = state.postings_options(WriteOptions::default());
    let mut terms = BlockTermsWriter::new(terms_index_writer, state);
    let postings = postings_writer::write_postings(
        inputs,
        norms,
        options.has_prox,
        options.has_payloads_or_offsets,
        state.segment_id,
        state.segment_suffix,
        |field| terms.write_field(field),
    )?;
    let (tib, index) = terms.close()?;
    Ok(BlockTermsOutput {
        postings,
        tib,
        index,
    })
}

/// `seekDir(input)`: the directory offset in the long before the footer.
fn seek_dir(input: &mut lucene_store::data_input::SliceInput<'_>, len: usize) -> Result<()> {
    use lucene_store::data_input::DataInput;
    let at = len
        .checked_sub(lucene_store::codec_util::FOOTER_LENGTH + 8)
        .ok_or_else(|| corrupt("file too short for its directory offset"))?;
    input.seek(at)?;
    let dir = input.read_i64()?;
    let dir = usize::try_from(dir)
        .ok()
        .filter(|&d| d <= at)
        .ok_or_else(|| corrupt(format!("invalid directory offset {dir}")))?;
    Ok(input.seek(dir)?)
}

/// A `.tib`/`.tii` file pointer off disk as a `usize` inside `len`.
fn file_pointer(fp: i64, len: usize, what: &str) -> Result<usize> {
    usize::try_from(fp)
        .ok()
        .filter(|&p| p <= len)
        .ok_or_else(|| corrupt(format!("invalid {what}: {fp}")))
}
