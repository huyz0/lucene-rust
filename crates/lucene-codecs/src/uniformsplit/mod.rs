//! Port of lucene-codecs' `org.apache.lucene.codecs.uniformsplit`:
//! `UniformSplitPostingsFormat`, a terms dictionary that cuts a field's
//! sorted terms into blocks of about the same number of lines and indexes
//! each block by the shortest prefix that tells its first term from the
//! term before it, over `Lucene104PostingsWriter`'s postings. The
//! [`sharedterms`] subpackage shares one dictionary among a group's fields;
//! the block writer is generic over a line's term states ([`writer`]'s
//! `BlockSink`) and the block reader reads a line's term state from its
//! field list when the terms are shared, for it.
//!
//! # The files
//!
//! Two, under the group's codec suffix (`_0_UniformSplit_0.ustb`/`.ustd`),
//! beside the postings' `.doc`/`.pos`/`.pay`/`.psm`:
//!
//! ```text
//! .ustb  IndexHeader("UniformSplit", 1)
//!        IndexHeader("Lucene104PostingsWriterTerms", 0), vInt(BLOCK_SIZE)
//!        blocks:  vInt(numBytes), the block (encoded: the encoder's bytes)
//!          block: vInt(linesCount) vLong(baseDocsFP) vLong(basePosFP)
//!                 vLong(basePayFP) vInt(termStatesBaseOffset)
//!                 vInt(middleLineOffset)
//!                 lines: vInt(termStateRelativeOffset), then the term --
//!                   first and middle line: vLong(length), bytes;
//!                   others: vLong(suffixLength << bits(prevLength) |
//!                     mdpLength - 1), suffix bytes (0 for the empty term)
//!                 term states: vInt(docFreq) [vLong(ttf - docFreq)]
//!                   vInt(singletonDocID) | vLong(docStartFP - baseDocsFP)
//!                   [vLong(posStartFP - base) [vLong(payStartFP - base)]
//!                    [vLong(lastPosBlockOffset) when ttf > 256]]
//!          - the singleton form iff docFreq == 1; payStartFP iff the field
//!            has offsets or payloads;
//!          - each base*FP is the block's first non-zero pointer (0: unset);
//!          - termStatesBaseOffset and middleLineOffset count from the first
//!            line's start, not the block's; middleLineIndex = linesCount >> 1
//!        vInt(numFields), the fields' metadata (encoded: vLong(len), bytes):
//!          vInt(number) vLong(numTerms) vLong(sumDocFreq)
//!          [vLong(sumTotalTermFreq - sumDocFreq)] vInt(docCount)
//!          vLong(dictionaryStartFP) vLong(firstBlockStartFP)
//!          vLong(lastBlockStartFP) vInt(lastTerm.length), lastTerm
//!        long(fieldsStartFP), Footer
//! .ustd  IndexHeader("UniformSplit", 1)
//!        per field: an FST (PositiveIntOutputs) from each block's key -- the
//!          first term's minimal distinguishing prefix -- to the block's file
//!          pointer (encoded: vLong(len), the encoder's bytes)
//!        Footer
//! ```
//!
//! A reader finds a term's block with the dictionary's floor seek, then
//! compares the term to the block's middle line before scanning, each line
//! compared from the prefix it shares with the line before. See [`writer`]
//! for how the blocks are cut.
//!
//! # What differs from Java
//!
//! - Module-level, like the bloom filter and the `memory` and
//!   `blocktreeords` formats: the files go in and come out as bytes, this
//!   port's `IndexWriter` routes no field here and `DirectoryReader` does not
//!   open these files. Fields are written in the order they are given (Java:
//!   field name order, which is what a flush or a merge passes).
//! - [`BlockEncoder`]/[`BlockDecoder`] take and return whole byte slices
//!   (Java: a `DataInput` and a length in, `WritableBytes`/`BytesRef` out).
//! - Each field's dictionary FST is decoded when the reader opens (Java:
//!   lazily, on a field's first seek), so a corrupt dictionary fails `open`;
//!   `dictionaryOnHeap` therefore changes nothing and is only recorded.
//! - Where Java asserts, reads garbage or loops (a line whose shared prefix
//!   is longer than the line before it, an offset past its block, a block
//!   file pointer the dictionary gives that is behind the block an
//!   intersection is in), this port returns [`Error::Store`] (`Corrupted`).
//!   On the write side, settings below `target >= 3 * delta + 1` can leave a
//!   field's last lines with nowhere to cut but before the first line (Java:
//!   an `AssertionError`, or without assertions an exception on the empty
//!   block); the writer returns [`Error::IllegalArgument`].
//! - `docFreq`/`totalTermFreq`/`termState`/`postings` on an unpositioned
//!   enum return [`Error::Unsupported`] (Java: a `NullPointerException`), as
//!   do `ord()` and `seekExact(ord)` (`UnsupportedOperationException`).
//! - `.psm` is neither read nor validated; the postings enums do not take a
//!   `reuse` instance and `impacts` is not ported (as `blocktreeords`).
//! - `RamUsageUtil` and the `Accountable` sizes are not ported.

mod block_header;
mod block_line;
mod block_reader;
mod dictionary;
mod field_metadata;
mod intersect;
mod reader;
pub mod sharedterms;
mod term_state;
#[cfg(test)]
mod tests;
mod writer;

use std::fmt;
use std::sync::Arc;

pub(crate) use block_header::BlockHeader;
pub(crate) use block_line::{BlockLineReader, TermBytes};
pub use block_reader::BlockReader;
pub use dictionary::FstDictionary;
pub use field_metadata::FieldMetadata;
pub use intersect::IntersectBlockReader;
pub use reader::{UniformSplitTerms, UniformSplitTermsReader};
pub(crate) use term_state::DeltaBaseTermStateSerializer;
pub use term_state::UniformSplitTermState;
pub use writer::UniformSplitTermsWriter;

pub use crate::blocktree::SeekStatus;
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource, WriteOptions,
};
use crate::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::data_input::{DataInput, SliceInput};

/// `UniformSplitPostingsFormat.TERMS_DICTIONARY_EXTENSION`.
pub const TERMS_DICTIONARY_EXTENSION: &str = "ustd";
/// `UniformSplitPostingsFormat.TERMS_BLOCKS_EXTENSION`.
pub const TERMS_BLOCKS_EXTENSION: &str = "ustb";
/// `UniformSplitPostingsFormat.NAME`: the format's and both headers' name.
pub const NAME: &str = "UniformSplit";
/// `UniformSplitPostingsFormat.VERSION_START`.
pub const VERSION_START: i32 = 0;
/// `UniformSplitPostingsFormat.VERSION_ENCODABLE_FIELDS_METADATA`: the
/// version from which the fields' metadata goes through the encoder.
pub const VERSION_ENCODABLE_FIELDS_METADATA: i32 = 1;
/// `UniformSplitPostingsFormat.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_ENCODABLE_FIELDS_METADATA;
/// `UniformSplitTermsWriter.DEFAULT_TARGET_NUM_BLOCK_LINES`.
pub const DEFAULT_TARGET_NUM_BLOCK_LINES: i32 = 32;
/// `UniformSplitTermsWriter.DEFAULT_DELTA_NUM_LINES`: `(int) (32 * 0.1)`.
pub const DEFAULT_DELTA_NUM_LINES: i32 = 3;
/// `UniformSplitTermsWriter.MAX_NUM_BLOCK_LINES`.
pub const MAX_NUM_BLOCK_LINES: i32 = 1_000;

/// What reading or writing this format fails with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Headers, footers, checksums, field metadata and malformed blocks.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A dictionary FST that does not decode.
    #[error(transparent)]
    Fst(#[from] crate::fst_compiler::FstReadError),
    /// Building a field's dictionary.
    #[error(transparent)]
    Compile(#[from] crate::fst_compiler::CompileError),
    /// The postings writer's input checks.
    #[error(transparent)]
    Write(#[from] postings_writer::Error),
    /// Decoding a term's postings.
    #[error(transparent)]
    Postings(#[from] crate::postings::Error),
    /// Java's `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// Java's `UnsupportedOperationException`, and the states Java leaves to
    /// a `NullPointerException`.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `BlockEncoder`: encodes one block, a field's dictionary, or the fields'
/// metadata, whole.
pub trait BlockEncoder: Send + Sync {
    /// `encode(blockBytes, length)`.
    ///
    /// # Errors
    /// What the encoding fails with.
    fn encode(&self, bytes: &[u8]) -> Result<Vec<u8>>;

    /// Encode a dictionary. Defaults to [`Self::encode`]; a test format can
    /// observe Java's `writeDictionary` override around this call.
    ///
    /// # Errors
    /// What the encoding fails with.
    fn encode_dictionary(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        self.encode(bytes)
    }

    /// Encode field metadata. Defaults to [`Self::encode`]; a test format
    /// can observe Java's `writeEncodedFieldsMetadata` override here.
    ///
    /// # Errors
    /// What the encoding fails with.
    fn encode_fields_metadata(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        self.encode(bytes)
    }
}

/// `BlockDecoder`: [`BlockEncoder`]'s inverse.
pub trait BlockDecoder: Send + Sync {
    /// `decode(blockBytes, length)`.
    ///
    /// # Errors
    /// What the decoding fails with (Java: an `IOException`).
    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>>;
}

/// `UniformSplitTermsWriter.validateSettings(targetNumBlockLines,
/// deltaNumLines)`.
///
/// # Errors
/// [`Error::IllegalArgument`] for the settings Java refuses.
pub fn validate_settings(target_num_block_lines: i32, delta_num_lines: i32) -> Result<()> {
    if target_num_block_lines <= 0 {
        return Err(Error::IllegalArgument(format!(
            "Invalid negative or nul targetNumBlockLines={target_num_block_lines}"
        )));
    }
    if delta_num_lines < 0 {
        return Err(Error::IllegalArgument(format!(
            "Invalid negative deltaNumLines={delta_num_lines}"
        )));
    }
    if delta_num_lines >= target_num_block_lines {
        return Err(Error::IllegalArgument(format!(
            "Invalid too large deltaNumLines={delta_num_lines}, it must be < \
             targetNumBlockLines={target_num_block_lines}"
        )));
    }
    // Both are in 0..i32::MAX here, so the i64 sum cannot overflow.
    let sum = i64::from(target_num_block_lines).saturating_add(i64::from(delta_num_lines));
    if sum > i64::from(MAX_NUM_BLOCK_LINES) {
        return Err(Error::IllegalArgument(format!(
            "Invalid (targetNumBlockLines + deltaNumLines)={sum}, it must be <= \
             MAX_NUM_BLOCK_LINES={MAX_NUM_BLOCK_LINES}"
        )));
    }
    Ok(())
}

/// `UniformSplitPostingsFormat(targetNumBlockLines, deltaNumLines,
/// blockEncoder, blockDecoder, dictionaryOnHeap)`.
#[derive(Clone)]
pub struct UniformSplitPostingsFormat {
    target_num_block_lines: i32,
    delta_num_lines: i32,
    block_encoder: Option<Arc<dyn BlockEncoder>>,
    block_decoder: Option<Arc<dyn BlockDecoder>>,
    dictionary_on_heap: bool,
}

impl fmt::Debug for UniformSplitPostingsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniformSplitPostingsFormat")
            .field("target_num_block_lines", &self.target_num_block_lines)
            .field("delta_num_lines", &self.delta_num_lines)
            .field("encoded", &self.block_encoder.is_some())
            .field("dictionary_on_heap", &self.dictionary_on_heap)
            .finish()
    }
}

impl Default for UniformSplitPostingsFormat {
    /// `new UniformSplitPostingsFormat()`: 32 +- 3 lines, no encoding.
    fn default() -> Self {
        UniformSplitPostingsFormat {
            target_num_block_lines: DEFAULT_TARGET_NUM_BLOCK_LINES,
            delta_num_lines: DEFAULT_DELTA_NUM_LINES,
            block_encoder: None,
            block_decoder: None,
            dictionary_on_heap: false,
        }
    }
}

/// What [`UniformSplitPostingsFormat::write`] produces.
#[derive(Debug, Clone, Default)]
pub struct UniformSplitOutput {
    /// `.doc`, `.psm`, `.pos`, `.pay`.
    pub postings: PostingsFiles,
    /// `.ustb`: the blocks and the fields' metadata.
    pub ustb: Vec<u8>,
    /// `.ustd`: each field's dictionary.
    pub ustd: Vec<u8>,
}

impl UniformSplitPostingsFormat {
    /// The constructor, with `validateSettings`' and `validateBlockEncoder`'s
    /// checks.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for block sizes Java refuses, or an encoder
    /// without a decoder (or the reverse).
    pub fn new(
        target_num_block_lines: i32,
        delta_num_lines: i32,
        block_encoder: Option<Arc<dyn BlockEncoder>>,
        block_decoder: Option<Arc<dyn BlockDecoder>>,
        dictionary_on_heap: bool,
    ) -> Result<Self> {
        validate_settings(target_num_block_lines, delta_num_lines)?;
        if block_encoder.is_some() != block_decoder.is_some() {
            return Err(Error::IllegalArgument(
                "Invalid blockEncoder and blockDecoder, both must be null or both must be non-null"
                    .into(),
            ));
        }
        Ok(UniformSplitPostingsFormat {
            target_num_block_lines,
            delta_num_lines,
            block_encoder,
            block_decoder,
            dictionary_on_heap,
        })
    }

    /// `targetNumBlockLines`.
    pub fn target_num_block_lines(&self) -> i32 {
        self.target_num_block_lines
    }

    /// `deltaNumLines`.
    pub fn delta_num_lines(&self) -> i32 {
        self.delta_num_lines
    }

    /// `dictionaryOnHeap` (no effect here: dictionaries are always decoded
    /// into memory).
    pub fn dictionary_on_heap(&self) -> bool {
        self.dictionary_on_heap
    }

    /// `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// What the postings writer rejects in `inputs`, a dictionary that cannot
    /// be built, and what the encoder fails with.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<UniformSplitOutput> {
        let options = state.postings_options(WriteOptions::default());
        let mut terms = UniformSplitTermsWriter::new(
            state,
            self.target_num_block_lines,
            self.delta_num_lines,
            self.block_encoder.clone(),
        )?;
        let postings = postings_writer::write_postings(
            inputs,
            norms,
            options.has_prox,
            options.has_payloads_or_offsets,
            state.segment_id,
            state.segment_suffix,
            |field| terms.write_field(field),
        )?;
        let (ustb, ustd) = terms.finish()?;
        Ok(UniformSplitOutput {
            postings,
            ustb,
            ustd,
        })
    }

    /// `fieldsProducer(state)`: the group's fields, from its `.ustb`/`.ustd`.
    ///
    /// # Errors
    /// See [`UniformSplitTermsReader::open`].
    pub fn open(
        &self,
        ustb: Vec<u8>,
        ustd: &[u8],
        state: &SegmentReadState<'_>,
    ) -> Result<UniformSplitTermsReader> {
        UniformSplitTermsReader::open(ustb, ustd, state, self.block_decoder.clone())
    }
}

/// `ByteArrayDataInput.readVInt` at `*pos`.
fn read_vint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vint()?;
    *pos = input.position();
    Ok(v)
}

/// `ByteArrayDataInput.readVLong` at `*pos`.
fn read_vlong(buf: &[u8], pos: &mut usize) -> Result<i64> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vlong()?;
    *pos = input.position();
    Ok(v)
}

/// `n` bytes at `*pos`, advancing it.
fn read_slice<'b>(buf: &'b [u8], pos: &mut usize, n: usize) -> Result<&'b [u8]> {
    let end = pos
        .checked_add(n)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| corrupt(format!("{n} bytes past the end of the input")))?;
    let bytes = &buf[*pos..end];
    *pos = end;
    Ok(bytes)
}

/// A length off disk as `usize`.
fn length(n: i64) -> Result<usize> {
    usize::try_from(n).map_err(|_| corrupt(format!("negative length {n}")))
}
