//! Port of lucene-sandbox's `org.apache.lucene.sandbox.codecs.idversion`:
//! `IDVersionPostingsFormat`, a postings format for a primary-key field
//! whose every term occurs in exactly one document, with a 64-bit version
//! (the occurrence's 8-byte payload) recorded beside it, so an optimistic
//! concurrency check can ask "does this id exist with a version >= v?" and
//! be refused by the terms index without reading a block.
//!
//! # The files
//!
//! Two, under the group's codec suffix (`_0_IDVersion_0.tiv`/`.tipv`):
//!
//! ```text
//! .tiv   IndexHeader("VersionBlockTreeTermsDict", 1)
//!        IndexHeader("IDVersionPostingsWriterTerms", 1)   -- IDVersionPostingsWriter.init
//!        blocks:  vInt(entCount << 1 | isLastInFloor)
//!                 vInt(suffixBytes << 1 | isLeaf), suffix bytes:
//!                   leaf:      vInt(suffix), bytes
//!                   non-leaf:  vInt(suffix << 1 | isBlock), bytes[, vLong(blockFP - subFP)]
//!                 vInt(metaBytes), meta bytes: per term vInt(docID), then
//!                   vLong(version) for the block's first term, zLong(delta) after
//!        field directory: vInt(numFields), per field vInt(number), vLong(numTerms),
//!          vInt(rootCode.len), rootCode, vLong(Long.MAX_VALUE - maxVersion),
//!          vInt+bytes minTerm, vInt+bytes maxTerm
//!        long(dirStart), Footer
//! .tipv  IndexHeader("VersionBlockTreeTermsIndex", 1)
//!        per field: the FST (PairOutputs<ByteSequenceOutputs, PositiveIntOutputs>),
//!          metadata and body
//!        per field vLong(indexStartFP), long(dirStart), Footer
//! ```
//!
//! It is the Lucene 4 block tree: no suffix compression, no statistics block
//! (every term has `docFreq` = `totalTermFreq` = 1), and an FST terms index
//! whose second output is `Long.MAX_VALUE - maxVersion` of everything below
//! the prefix (positive, so the FST can share it as a minimum). See
//! [`writer`] for how the blocks are cut and [`reader`] for the seek.

mod reader;
mod terms_enum;
mod writer;

pub use reader::{VersionBlockTreeTermsReader, VersionFieldReader};
pub use terms_enum::IdVersionSegmentTermsEnum;
pub use writer::{write_fields, IdVersionFieldInput, IdVersionOutput};

use lucene_store::error::Error;

/// `IDVersionPostingsFormat`'s `getName()`: the per-field format name and
/// the stem of its codec suffix.
pub const FORMAT_NAME: &str = "IDVersion";
/// `IDVersionPostingsFormat.MIN_VERSION`.
pub const MIN_VERSION: i64 = 0;
/// `IDVersionPostingsFormat.MAX_VERSION`: the top two bits are reserved.
pub const MAX_VERSION: i64 = 0x3fff_ffff_ffff_ffff;

/// `VersionBlockTreeTermsWriter.TERMS_EXTENSION`.
pub const TERMS_EXTENSION: &str = "tiv";
/// `VersionBlockTreeTermsWriter.TERMS_INDEX_EXTENSION`.
pub const TERMS_INDEX_EXTENSION: &str = "tipv";
const TERMS_CODEC_NAME: &str = "VersionBlockTreeTermsDict";
const TERMS_INDEX_CODEC_NAME: &str = "VersionBlockTreeTermsIndex";
/// `IDVersionPostingsWriter.TERMS_CODEC`.
const POSTINGS_TERMS_CODEC: &str = "IDVersionPostingsWriterTerms";
/// `VERSION_START` = `VERSION_CURRENT` of all three headers.
const VERSION_CURRENT: i32 = 1;

const OUTPUT_FLAGS_NUM_BITS: u32 = 2;
const OUTPUT_FLAG_IS_FLOOR: i64 = 0x1;
const OUTPUT_FLAG_HAS_TERMS: i64 = 0x2;

/// `VersionBlockTreeTermsWriter.DEFAULT_MIN_BLOCK_SIZE`.
pub const DEFAULT_MIN_BLOCK_SIZE: usize = 25;
/// `VersionBlockTreeTermsWriter.DEFAULT_MAX_BLOCK_SIZE`.
pub const DEFAULT_MAX_BLOCK_SIZE: usize = 48;

/// The FST outputs of the terms index: the block's output bytes and
/// `Long.MAX_VALUE - maxVersion`.
type IndexOutputs =
    crate::fst::PairOutputs<crate::fst::ByteSequenceOutputs, crate::fst::PositiveIntOutputs>;
/// One [`IndexOutputs`] value.
type IndexOutput = crate::fst::Pair<Vec<u8>, i64>;

/// `IDVersionPostingsFormat(minTermsInBlock, maxTermsInBlock)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdVersionPostingsFormat {
    min_terms_in_block: usize,
    max_terms_in_block: usize,
}

impl Default for IdVersionPostingsFormat {
    /// `new IDVersionPostingsFormat()`: 25 and 48.
    fn default() -> Self {
        IdVersionPostingsFormat {
            min_terms_in_block: DEFAULT_MIN_BLOCK_SIZE,
            max_terms_in_block: DEFAULT_MAX_BLOCK_SIZE,
        }
    }
}

impl IdVersionPostingsFormat {
    /// The constructor, with `Lucene103BlockTreeTermsWriter.validateSettings`'
    /// checks.
    ///
    /// # Errors
    /// `IllegalArgument` for block sizes Java refuses.
    pub fn new(min_terms_in_block: usize, max_terms_in_block: usize) -> Result<Self, Error> {
        crate::postings_writer::WriteOptions {
            min_items_in_block: min_terms_in_block,
            max_items_in_block: max_terms_in_block,
            ..Default::default()
        }
        .validate()
        .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        Ok(IdVersionPostingsFormat {
            min_terms_in_block,
            max_terms_in_block,
        })
    }

    /// `minTermsInBlock`.
    pub fn min_terms_in_block(&self) -> usize {
        self.min_terms_in_block
    }

    /// `maxTermsInBlock`.
    pub fn max_terms_in_block(&self) -> usize {
        self.max_terms_in_block
    }
}

/// `IDVersionPostingsFormat.bytesToLong`: the big-endian long of an 8-byte
/// payload.
pub fn bytes_to_long(bytes: &[u8; 8]) -> i64 {
    i64::from_be_bytes(*bytes)
}

/// `IDVersionPostingsFormat.longToBytes`: the 8-byte payload of a version.
///
/// # Errors
/// `IllegalArgument` outside [`MIN_VERSION`]`..=`[`MAX_VERSION`], with
/// Java's message.
pub fn long_to_bytes(version: i64) -> Result<[u8; 8], Error> {
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(Error::IllegalArgument(format!(
            "version must be >= MIN_VERSION={MIN_VERSION} and <= MAX_VERSION={MAX_VERSION} (got: {version})"
        )));
    }
    Ok(version.to_be_bytes())
}

/// `IDVersionTermState`: a `BlockTermState` plus the term's single document
/// and its version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IdVersionTermState {
    /// `BlockTermState.docFreq` (1 once decoded).
    pub doc_freq: i32,
    /// `BlockTermState.totalTermFreq` (1 once decoded, -1 before).
    pub total_term_freq: i64,
    /// `BlockTermState.termBlockOrd`.
    pub term_block_ord: i32,
    /// `docID`.
    pub doc_id: i32,
    /// `idVersion`.
    pub id_version: i64,
}

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// `SingleDocsEnum`: the one document of a term, without positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleDocsEnum {
    doc: i32,
    single_doc_id: i32,
}

impl SingleDocsEnum {
    /// `new SingleDocsEnum()` + `reset(singleDocID)`.
    pub fn new(single_doc_id: i32) -> Self {
        SingleDocsEnum {
            doc: -1,
            single_doc_id,
        }
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        self.doc = if self.doc == -1 {
            self.single_doc_id
        } else {
            NO_MORE_DOCS
        };
        self.doc
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// `advance(target)`: the document only when it is not yet passed and
    /// not below `target`.
    pub fn advance(&mut self, target: i32) -> i32 {
        self.doc = if self.doc == -1 && target <= self.single_doc_id {
            self.single_doc_id
        } else {
            NO_MORE_DOCS
        };
        self.doc
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        1
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        1
    }

    /// `nextPosition()`: no positions.
    //
    // SENTINEL: `-1` = "no position/offsets indexed", as Java; no Rust
    // caller decodes with it.
    pub fn next_position(&self) -> i32 {
        -1
    }
}

/// `SinglePostingsEnum`: the one document of a term, at position 0, its
/// version as the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinglePostingsEnum {
    doc: i32,
    pos: i32,
    single_doc_id: i32,
    version: i64,
}

impl SinglePostingsEnum {
    /// `new SinglePostingsEnum()` + `reset(singleDocID, version)`.
    pub fn new(single_doc_id: i32, version: i64) -> Self {
        SinglePostingsEnum {
            doc: -1,
            pos: -1,
            single_doc_id,
            version,
        }
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        self.doc = if self.doc == -1 {
            self.single_doc_id
        } else {
            NO_MORE_DOCS
        };
        self.pos = -1;
        self.doc
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// `advance(target)`.
    pub fn advance(&mut self, target: i32) -> i32 {
        if self.doc == -1 && target <= self.single_doc_id {
            self.doc = self.single_doc_id;
            self.pos = -1;
        } else {
            self.doc = NO_MORE_DOCS;
        }
        self.doc
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        1
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        1
    }

    /// `nextPosition()`: position 0, after which [`Self::payload`] is the
    /// version.
    pub fn next_position(&mut self) -> i32 {
        self.pos = 0;
        self.pos
    }

    /// `getPayload()`: the version's 8 bytes. Java's `longToBytes` would
    /// throw for a version out of range; a decoded one never is (it is a
    /// `vLong`/`zLong` sum the writer checked), and this returns its bytes
    /// regardless.
    pub fn payload(&self) -> [u8; 8] {
        self.version.to_be_bytes()
    }

    /// `startOffset()`.
    //
    // SENTINEL: `-1` = "no position/offsets indexed", as Java; no Rust
    // caller decodes with it.
    pub fn start_offset(&self) -> i32 {
        -1
    }

    /// `endOffset()`.
    //
    // SENTINEL: `-1` = "no position/offsets indexed", as Java; no Rust
    // caller decodes with it.
    pub fn end_offset(&self) -> i32 {
        -1
    }
}

/// What `IDVersionPostingsReader.postings` hands back, by the flags asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdVersionPostings {
    /// Positions (or payloads) were asked for.
    Positions(SinglePostingsEnum),
    /// Documents (and frequencies) only.
    Docs(SingleDocsEnum),
}

#[cfg(test)]
mod tests;
