//! Port of lucene-codecs' `org.apache.lucene.codecs.blocktreeords`:
//! `BlockTreeOrdsPostingsFormat`, the Lucene 4 block-tree terms dictionary
//! extended with term ords, so a terms enum answers `ord()` and
//! `seekExact(long ord)`, over `Lucene104PostingsWriter`'s postings.
//!
//! # The files
//!
//! Two, under the group's codec suffix (`_0_BlockTreeOrds_0.tio`/`.tipo`),
//! beside the postings' `.doc`/`.pos`/`.pay`/`.psm`:
//!
//! ```text
//! .tio   IndexHeader("OrdsBlockTreeTerms", 1)
//!        IndexHeader("Lucene104PostingsWriterTerms", 0), vInt(BLOCK_SIZE)
//!        blocks:  vInt(entCount << 1 | isLastInFloor)
//!                 vInt(suffixBytes << 1 | isLeaf), suffix bytes:
//!                   leaf:      vInt(suffix), bytes
//!                   non-leaf:  vInt(suffix << 1), bytes                  (a term)
//!                              vInt(suffix << 1 | 1), bytes,
//!                              vLong(blockFP - subFP), vLong(subTermCount) (a sub-block)
//!                 vInt(statsBytes), per term vInt(docFreq)
//!                   [, vLong(totalTermFreq - docFreq) unless DOCS]
//!                 vInt(metaBytes), per term encodeTerm (the block's first absolute)
//!        field directory: vInt(numFields), per field vInt(number), vLong(numTerms),
//!          vInt(rootCode.len), rootCode, [vLong(sumTotalTermFreq) unless DOCS,]
//!          vLong(sumDocFreq), vInt(docCount), vInt+bytes minTerm, vInt+bytes maxTerm
//!        long(dirStart), Footer
//! .tipo  IndexHeader("OrdsBlockTreeIndex", 1)
//!        per field: the FST (FSTOrdsOutputs), metadata and body
//!        per field vLong(indexStartFP), long(dirStart), Footer
//! ```
//!
//! Each terms-index output ([`Output`]) is a block's frame data -- `vLong(fp
//! << 2 | hasTerms << 1 | isFloor)`, then for a floor block `vInt(followers)`
//! and per follower its lead byte, `vLong` of the terms before it and
//! `vLong(fpDelta << 1 | hasTerms)` -- plus the first and last ord of the
//! terms below the prefix. `seekExact(ord)` walks the FST by those ranges
//! to the deepest block holding the ord, then scans it. See [`writer`] for
//! how the blocks are cut.
//!
//! # What differs from Java
//!
//! - Module-level, like the bloom filter and the `memory` formats: the
//!   files go in and come out as bytes, this port's `IndexWriter` routes no
//!   field here and `DirectoryReader` does not open these files.
//! - Fields are written in the order they are given; Java's are in field
//!   name order, which is what a caller passing a flush's or a merge's
//!   fields in that order gets.
//! - The writer compiles only the root block's FST, from the entry lists
//!   every block passes up; Java compiles one per block and re-reads it into
//!   its parent's. The bytes are the same.
//! - Where Java asserts or would read garbage (a frame past the block's
//!   entries, an intersect start term whose prefix the automaton rejects,
//!   a sub-block missing from its parent), this port returns
//!   [`Error::Store`] (`Corrupted`) or [`Error::Unsupported`].
//! - `next()` after it returned `None` re-seeks the empty term as Java
//!   does, and where Java asserts that term exists and then reads past a
//!   block, this port goes on from where the seek stopped or returns
//!   `Corrupted`; `ord()` before a term is positioned is `-1`.
//! - `seekExact(ord)`'s `getByOutput` sums only the ords of each candidate
//!   arc (Java sums whole outputs; only the ords decide) and binary-searches
//!   direct-addressing nodes by presence index, as binary-search nodes are
//!   (Java scans them): the Java-faithful walk measured `cx_ords_seek`
//!   0.89x, this one 1.58x. An arc target not below its node (a cycle in a
//!   corrupt `.tipo`, where Java would loop forever) is `Corrupted`.
//! - `.psm` is neither read nor validated (the whole-file checksums of
//!   `.doc`/`.pos`/`.pay` stand in for its recorded lengths); the postings
//!   enums do not take a `reuse` instance and `impacts` is not ported.

mod intersect;
mod outputs;
mod reader;
mod segment_enum;
#[cfg(test)]
mod tests;
mod writer;

pub use intersect::OrdsIntersectTermsEnum;
pub use outputs::{FstOrdsOutputs, Output};
pub use reader::{OrdsBlockTreeTermsReader, OrdsFieldReader};
pub use segment_enum::{OrdsSegmentTermsEnum, OrdsTermState};
pub use writer::OrdsBlockTreeTermsWriter;

pub use crate::blocktree::SeekStatus;
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource, WriteOptions,
};
use crate::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::data_input::{DataInput, SliceInput};

/// `OrdsBlockTreeTermsWriter.TERMS_EXTENSION`.
pub const TERMS_EXTENSION: &str = "tio";
/// `OrdsBlockTreeTermsWriter.TERMS_INDEX_EXTENSION`.
pub const TERMS_INDEX_EXTENSION: &str = "tipo";
/// `OrdsBlockTreeTermsWriter.TERMS_CODEC_NAME`.
pub const TERMS_CODEC_NAME: &str = "OrdsBlockTreeTerms";
/// `OrdsBlockTreeTermsWriter.TERMS_INDEX_CODEC_NAME`.
pub const TERMS_INDEX_CODEC_NAME: &str = "OrdsBlockTreeIndex";
/// `VERSION_START` = `VERSION_CURRENT` of both headers.
pub const VERSION_CURRENT: i32 = 1;
/// `OrdsBlockTreeTermsWriter.DEFAULT_MIN_BLOCK_SIZE`.
pub const DEFAULT_MIN_BLOCK_SIZE: usize = 25;
/// `OrdsBlockTreeTermsWriter.DEFAULT_MAX_BLOCK_SIZE`.
pub const DEFAULT_MAX_BLOCK_SIZE: usize = 48;

const OUTPUT_FLAGS_NUM_BITS: u32 = 2;
const OUTPUT_FLAG_IS_FLOOR: i64 = 0x1;
const OUTPUT_FLAG_HAS_TERMS: i64 = 0x2;

/// What reading or writing this format fails with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Headers, footers, checksums, field summaries and malformed blocks.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A terms index FST that does not decode.
    #[error(transparent)]
    Fst(#[from] crate::fst_compiler::FstReadError),
    /// Building a field's terms index.
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
    /// Java's `UnsupportedOperationException`, and the states Java asserts
    /// away.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `BlockTreeOrdsPostingsFormat(minTermsInBlock, maxTermsInBlock)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTreeOrdsPostingsFormat {
    min_terms_in_block: usize,
    max_terms_in_block: usize,
}

impl Default for BlockTreeOrdsPostingsFormat {
    /// `new BlockTreeOrdsPostingsFormat()`: 25 and 48.
    fn default() -> Self {
        BlockTreeOrdsPostingsFormat {
            min_terms_in_block: DEFAULT_MIN_BLOCK_SIZE,
            max_terms_in_block: DEFAULT_MAX_BLOCK_SIZE,
        }
    }
}

/// `new BlockTreeOrdsPostingsFormat()`, the instance
/// [`crate::terms_dictionary::lookup`] reads `BlockTreeOrds` groups with.
pub static DEFAULT_FORMAT: BlockTreeOrdsPostingsFormat = BlockTreeOrdsPostingsFormat {
    min_terms_in_block: DEFAULT_MIN_BLOCK_SIZE,
    max_terms_in_block: DEFAULT_MAX_BLOCK_SIZE,
};

impl crate::terms_dictionary::TermsDictionaryFormat for BlockTreeOrdsPostingsFormat {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn extensions(&self) -> &'static [&'static str] {
        &[TERMS_EXTENSION, TERMS_INDEX_EXTENSION]
    }

    fn write(
        &self,
        inputs: &[crate::terms_dictionary::ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)> {
        let mut terms =
            OrdsBlockTreeTermsWriter::new(state, self.min_terms_in_block, self.max_terms_in_block);
        let postings = crate::terms_dictionary::write_field_by_field(
            inputs,
            norms,
            options,
            state,
            |field| terms.write_field(field).map_err(to_writer_error),
        )?;
        let (tio, tipo) = terms.finish();
        Ok((postings, vec![tio, tipo]))
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> crate::blocktree::Result<Vec<postings_writer::DictionaryField>> {
        let err = to_blocktree_error;
        let [tio, tipo] = files else {
            return Err(crate::blocktree::Error::Unsupported(
                "a BlockTreeOrds group needs its .tio and .tipo",
            ));
        };
        let reader = OrdsBlockTreeTermsReader::open(tio.to_vec(), tipo, state).map_err(err)?;
        reader.dictionary_fields().map_err(err)
    }
}

impl OrdsBlockTreeTermsReader {
    /// Every field's terms with their statistics and postings metadata
    /// ([`crate::terms_dictionary::TermsDictionaryFormat::read`]).
    ///
    /// # Errors
    /// A block that does not decode.
    pub fn dictionary_fields(&self) -> Result<Vec<postings_writer::DictionaryField>> {
        let mut fields = Vec::with_capacity(self.size());
        for name in self.field_names() {
            let Some(terms) = self.terms(name) else {
                continue;
            };
            let info = terms.field_info();
            let mut field = postings_writer::DictionaryField {
                field_number: info.number,
                index_options: info.index_options,
                has_payloads: info.store_payloads,
                doc_count: terms.doc_count(),
                terms: Vec::new(),
            };
            let mut it = terms.iter();
            while let Some(bytes) = it.next()? {
                let bytes = bytes.to_vec();
                field.terms.push(postings_writer::DictionaryTerm {
                    bytes,
                    doc_freq: it.doc_freq()?,
                    total_term_freq: it.total_term_freq()?,
                    meta: it.term_metadata()?,
                });
            }
            fields.push(field);
        }
        Ok(fields)
    }
}

/// This module's [`Error`] as the block tree reader's.
fn to_blocktree_error(e: Error) -> crate::blocktree::Error {
    match e {
        Error::Store(e) => crate::blocktree::Error::Store(e),
        Error::Postings(e) => crate::blocktree::Error::Postings(e),
        e => crate::blocktree::Error::ForeignDictionary {
            format: BlockTreeOrdsPostingsFormat::NAME,
            message: e.to_string(),
        },
    }
}

/// This module's [`Error`] as the postings writer's.
fn to_writer_error(e: Error) -> postings_writer::Error {
    match e {
        Error::Write(e) => e,
        Error::Store(e) => postings_writer::Error::Store(e),
        e => postings_writer::Error::Dictionary(e.to_string()),
    }
}

/// What [`BlockTreeOrdsPostingsFormat::write`] produces.
#[derive(Debug, Clone, Default)]
pub struct BlockTreeOrdsOutput {
    /// `.doc`, `.psm`, `.pos`, `.pay`.
    pub postings: PostingsFiles,
    /// `.tio`: the blocks and the field directory.
    pub tio: Vec<u8>,
    /// `.tipo`: each field's FST terms index.
    pub tipo: Vec<u8>,
}

impl BlockTreeOrdsPostingsFormat {
    /// `getName()`.
    pub const NAME: &'static str = "BlockTreeOrds";

    /// The constructor, with `Lucene103BlockTreeTermsWriter.validateSettings`'
    /// checks.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for block sizes Java refuses.
    pub fn new(min_terms_in_block: usize, max_terms_in_block: usize) -> Result<Self> {
        WriteOptions {
            min_items_in_block: min_terms_in_block,
            max_items_in_block: max_terms_in_block,
            ..Default::default()
        }
        .validate()
        .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        Ok(BlockTreeOrdsPostingsFormat {
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

    /// `BLOCK_SIZE`: the format's own constant, left at Lucene 4.1's 128
    /// (the postings it writes use `Lucene104PostingsFormat`'s 256).
    pub const BLOCK_SIZE: i32 = 128;

    /// `toString()`: the name and [`Self::BLOCK_SIZE`].
    pub fn name(&self) -> String {
        format!("{}(blocksize={})", Self::NAME, Self::BLOCK_SIZE)
    }

    /// `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// What the postings writer rejects in `inputs`, and a terms index that
    /// cannot be built.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BlockTreeOrdsOutput> {
        let options = state.postings_options(WriteOptions::default());
        let mut terms =
            OrdsBlockTreeTermsWriter::new(state, self.min_terms_in_block, self.max_terms_in_block);
        let postings = postings_writer::write_postings(
            inputs,
            norms,
            options.has_prox,
            options.has_payloads_or_offsets,
            state.segment_id,
            state.segment_suffix,
            |field| terms.write_field(field),
        )?;
        let (tio, tipo) = terms.finish();
        Ok(BlockTreeOrdsOutput {
            postings,
            tio,
            tipo,
        })
    }

    /// `fieldsProducer(state)`: the group's fields, from its `.tio`/`.tipo`.
    ///
    /// # Errors
    /// See [`OrdsBlockTreeTermsReader::open`].
    pub fn open(
        &self,
        tio: Vec<u8>,
        tipo: &[u8],
        state: &SegmentReadState<'_>,
    ) -> Result<OrdsBlockTreeTermsReader> {
        OrdsBlockTreeTermsReader::open(tio, tipo, state)
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

/// `ByteArrayDataInput.readByte() & 0xff` at `*pos`.
fn read_byte(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let b = *buf
        .get(*pos)
        .ok_or_else(|| corrupt("read past the end of a block"))?;
    *pos = pos.saturating_add(1);
    Ok(i32::from(b))
}

/// A length off disk as `usize`.
fn length(n: i32) -> Result<usize> {
    usize::try_from(n).map_err(|_| corrupt(format!("negative length {n}")))
}

/// `code >>> k` of a Java `int`.
fn ushr32(code: i32, k: u32) -> i32 {
    ((code as u32) >> k) as i32
}

/// `code >>> k` of a Java `long`.
fn ushr64(code: i64, k: u32) -> i64 {
    ((code as u64) >> k) as i64
}

/// `in.readBytes(dst, 0, n)` into a buffer reused across blocks.
fn read_exact(input: &mut SliceInput<'_>, dst: &mut Vec<u8>, n: usize) -> Result<()> {
    if n > input.len().saturating_sub(input.position()) {
        return Err(corrupt(format!("{n} bytes past the end of the terms file")));
    }
    let at = input.position();
    // ARITH: `n` is within the remaining input, checked above.
    #[allow(clippy::arithmetic_side_effects)]
    let end = at + n;
    dst.clear();
    dst.extend_from_slice(input.slice(at, end)?);
    input.seek(end)?;
    Ok(())
}

/// One block as `loadBlock` reads it: the header, the suffix, stats and
/// metadata blobs, and the file pointer after it.
#[derive(Debug, Default, Clone)]
struct BlockData {
    ent_count: i32,
    is_last_in_floor: bool,
    is_leaf_block: bool,
    fp_end: i64,
}

/// `loadBlock`'s reads at `fp`: fills the three blobs, returns the header.
fn read_block(
    tio: &[u8],
    fp: i64,
    suffix_bytes: &mut Vec<u8>,
    stat_bytes: &mut Vec<u8>,
    bytes: &mut Vec<u8>,
) -> Result<BlockData> {
    let mut input = SliceInput::new(tio);
    let at = usize::try_from(fp).map_err(|_| corrupt(format!("invalid block fp {fp}")))?;
    input.seek(at)?;
    let code = input.read_vint()?;
    let ent_count = ushr32(code, 1);
    if ent_count <= 0 {
        // Java asserts `entCount > 0`.
        return Err(corrupt(format!("block at {fp} has {ent_count} entries")));
    }
    let is_last_in_floor = code & 1 != 0;
    let code = input.read_vint()?;
    let is_leaf_block = code & 1 != 0;
    read_exact(&mut input, suffix_bytes, length(ushr32(code, 1))?)?;
    let n = length(input.read_vint()?)?;
    read_exact(&mut input, stat_bytes, n)?;
    let n = length(input.read_vint()?)?;
    read_exact(&mut input, bytes, n)?;
    Ok(BlockData {
        ent_count,
        is_last_in_floor,
        is_leaf_block,
        fp_end: input.position() as i64,
    })
}
