//! Port of `BlockTermsReader`, its `FieldReader` and `SegmentTermsEnum`:
//! opens a group's `.tib` over a loaded terms index, and walks a field's
//! blocks -- a seek asks the index for a block and scans it, a `next()`
//! reads the following term's suffix, and a term's statistics and postings
//! metadata are decoded only when asked for.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use super::{
    corrupt, file_pointer, seek_dir, BlockTermState, Error, FixedGapIndexEnum,
    FixedGapTermsIndexReader, Result, SeekStatus, VariableGapIndexEnum,
    VariableGapTermsIndexReader, CODEC_NAME, POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT,
    POSTINGS_VERSION_START, VERSION_CURRENT, VERSION_START,
};
use crate::field_infos::{FieldInfo, IndexOptions};
use crate::postings::{self, DocInput, Postings, PostingsFlags};
use crate::segment_state::SegmentReadState;

/// `TermsIndexReaderBase`: the terms index a [`BlockTermsReader`] seeks
/// with, one of the two this package has.
#[derive(Debug)]
pub enum TermsIndexReader {
    /// `FixedGapTermsIndexReader`.
    FixedGap(FixedGapTermsIndexReader),
    /// `VariableGapTermsIndexReader`.
    VariableGap(VariableGapTermsIndexReader),
}

impl From<FixedGapTermsIndexReader> for TermsIndexReader {
    fn from(r: FixedGapTermsIndexReader) -> Self {
        TermsIndexReader::FixedGap(r)
    }
}

impl From<VariableGapTermsIndexReader> for TermsIndexReader {
    fn from(r: VariableGapTermsIndexReader) -> Self {
        TermsIndexReader::VariableGap(r)
    }
}

impl TermsIndexReader {
    /// `supportsOrd()`.
    pub fn supports_ord(&self) -> bool {
        match self {
            TermsIndexReader::FixedGap(r) => r.supports_ord(),
            TermsIndexReader::VariableGap(r) => r.supports_ord(),
        }
    }

    /// `getFieldEnum(fieldInfo)`.
    fn field_enum(&self, field: &str) -> Option<FieldIndexEnum<'_>> {
        match self {
            TermsIndexReader::FixedGap(r) => r.field_enum(field).map(FieldIndexEnum::FixedGap),
            TermsIndexReader::VariableGap(r) => {
                r.field_enum(field).map(FieldIndexEnum::VariableGap)
            }
        }
    }
}

/// `TermsIndexReaderBase.FieldIndexEnum`.
#[derive(Debug)]
enum FieldIndexEnum<'a> {
    FixedGap(FixedGapIndexEnum<'a>),
    VariableGap(VariableGapIndexEnum<'a>),
}

impl FieldIndexEnum<'_> {
    /// `term()` (empty before the first seek).
    fn term(&self) -> &[u8] {
        match self {
            FieldIndexEnum::FixedGap(e) => e.term(),
            FieldIndexEnum::VariableGap(e) => e.term().unwrap_or(&[]),
        }
    }

    /// `seek(target)`.
    fn seek(&mut self, target: &[u8]) -> Result<i64> {
        match self {
            FieldIndexEnum::FixedGap(e) => e.seek(target),
            FieldIndexEnum::VariableGap(e) => e.seek(target),
        }
    }

    /// `next()`, `None` for Java's `-1`.
    fn next(&mut self) -> Result<Option<i64>> {
        match self {
            FieldIndexEnum::FixedGap(e) => e.next(),
            FieldIndexEnum::VariableGap(e) => e.next(),
        }
    }

    /// `ord()`.
    fn ord(&self) -> Result<i64> {
        match self {
            FieldIndexEnum::FixedGap(e) => Ok(e.ord()),
            FieldIndexEnum::VariableGap(_) => Err(Error::Unsupported("ord")),
        }
    }

    /// `seek(ord)`.
    fn seek_ord(&mut self, ord: i64) -> Result<i64> {
        match self {
            FieldIndexEnum::FixedGap(e) => e.seek_ord(ord),
            FieldIndexEnum::VariableGap(_) => Err(Error::Unsupported("seek(ord)")),
        }
    }
}

/// `BlockTermsReader`: the fields of one group, by name.
#[derive(Debug)]
pub struct BlockTermsReader {
    /// `in`: the whole `.tib`.
    pub(super) tib: Arc<[u8]>,
    pub(super) fields: BTreeMap<String, BlockTermsFieldReader>,
}

/// `BlockTermsReader.FieldReader`: one field's statistics.
pub struct BlockTermsFieldReader {
    info: FieldInfo,
    tib: Arc<[u8]>,
    index: Arc<TermsIndexReader>,
    num_terms: i64,
    terms_start_pointer: usize,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
}

impl fmt::Debug for BlockTermsFieldReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockTermsFieldReader")
            .field("field", &self.info.name)
            .field("num_terms", &self.num_terms)
            .field("terms_start_pointer", &self.terms_start_pointer)
            .finish()
    }
}

impl BlockTermsReader {
    /// `new BlockTermsReader(indexReader, postingsReader, state)`: the
    /// header, the postings writer's header, the footer's shape, and each
    /// field's summary.
    ///
    /// # Errors
    /// `Corrupted` for a bad header or footer, a field number
    /// `state.field_infos` lacks, a field summary Java refuses (or one with
    /// no terms or a start past the file, which Java asserts against), a
    /// duplicate field; [`Error::Unsupported`] for a postings block size
    /// other than `Lucene104PostingsFormat`'s.
    pub fn open(
        tib: Vec<u8>,
        index: impl Into<TermsIndexReader>,
        state: &SegmentReadState<'_>,
    ) -> Result<Self> {
        let tib: Arc<[u8]> = tib.into();
        let index = Arc::new(index.into());
        let mut input = SliceInput::new(&tib);
        codec_util::check_index_header(
            &mut input,
            CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        // `postingsReader.init(in, state)`.
        codec_util::check_index_header(
            &mut input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        if input.read_vint()? != postings::BLOCK_SIZE {
            return Err(Error::Unsupported(
                "index-time BLOCK_SIZE differs from read-time BLOCK_SIZE",
            ));
        }
        codec_util::retrieve_checksum(&tib)?;
        seek_dir(&mut input, tib.len())?;

        let num_fields = input.read_vint()?;
        if num_fields < 0 {
            return Err(corrupt(format!("invalid number of fields: {num_fields}")));
        }
        let mut fields = BTreeMap::new();
        for _ in 0..num_fields {
            let field = input.read_vint()?;
            let num_terms = input.read_vlong()?;
            if num_terms <= 0 {
                return Err(corrupt(format!("invalid numTerms: {num_terms}")));
            }
            let terms_start_pointer = input.read_vlong()?;
            let info = state
                .field_infos
                .field_by_number(field)
                .ok_or_else(|| corrupt(format!("invalid field number: {field}")))?;
            let sum_total_term_freq = input.read_vlong()?;
            let sum_doc_freq = if info.index_options == IndexOptions::Docs {
                sum_total_term_freq
            } else {
                input.read_vlong()?
            };
            let doc_count = input.read_vint()?;
            if doc_count < 0 || doc_count > state.max_doc {
                // #docs with field must be <= #docs
                return Err(corrupt(format!(
                    "invalid docCount: {doc_count} maxDoc: {}",
                    state.max_doc
                )));
            }
            if sum_doc_freq < i64::from(doc_count) {
                // #postings must be >= #docs with field
                return Err(corrupt(format!(
                    "invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}"
                )));
            }
            if sum_total_term_freq < sum_doc_freq {
                // #positions must be >= #postings
                return Err(corrupt(format!(
                    "invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}"
                )));
            }
            let terms_start_pointer =
                file_pointer(terms_start_pointer, tib.len(), "termsStartPointer")?;
            let reader = BlockTermsFieldReader {
                info: info.clone(),
                tib: Arc::clone(&tib),
                index: Arc::clone(&index),
                num_terms,
                terms_start_pointer,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
            };
            if fields.insert(info.name.clone(), reader).is_some() {
                return Err(corrupt(format!("duplicate fields: {}", info.name)));
            }
        }
        Ok(BlockTermsReader { tib, fields })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&BlockTermsFieldReader> {
        self.fields.get(field)
    }

    /// `iterator()`: the field names, in order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.fields.len()
    }

    /// `checkIntegrity()`'s terms dictionary half: the whole `.tib`'s
    /// checksum (the postings' are the caller's, with the `.doc`).
    ///
    /// # Errors
    /// `Corrupted` when the checksum does not match.
    pub fn check_integrity(&self) -> Result<()> {
        let end = self
            .tib
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(&self.tib, end)?;
        Ok(())
    }
}

impl fmt::Display for BlockTermsReader {
    /// `toString()`, the readers by their class names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index = self
            .fields
            .values()
            .next()
            .map_or("?", |fr| match *fr.index {
                TermsIndexReader::FixedGap(_) => "FixedGapTermsIndexReader",
                TermsIndexReader::VariableGap(_) => "VariableGapTermsIndexReader",
            });
        write!(
            f,
            "BlockTermsReader(index={index},delegate=Lucene104PostingsReader)"
        )
    }
}

impl BlockTermsFieldReader {
    /// The field.
    pub fn info(&self) -> &FieldInfo {
        &self.info
    }

    /// `iterator()`: a [`SegmentTermsEnum`] at the field's first block.
    pub fn iterator(&self) -> SegmentTermsEnum<'_> {
        SegmentTermsEnum {
            fr: self,
            fp: self.terms_start_pointer,
            index_enum: self.index.field_enum(&self.info.name),
            do_ord: self.index.supports_ord(),
            state: BlockTermState::default(),
            term: Vec::new(),
            index_is_current: false,
            did_index_next: false,
            next_index_term: None,
            seek_pending: false,
            suffixes: Blob::default(),
            term_block_prefix: 0,
            block_term_count: 0,
            freqs: Blob::default(),
            meta: Blob::default(),
            meta_data_upto: 0,
        }
    }

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        self.info.index_options != IndexOptions::Docs
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.num_terms
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }
}

/// One of a block's three blobs and the `ByteArrayDataInput` over it.
#[derive(Debug, Default, Clone, Copy)]
struct Blob<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Blob<'a> {
    fn input(&self) -> Result<SliceInput<'a>> {
        let mut input = SliceInput::new(self.bytes);
        input.seek(self.pos)?;
        Ok(input)
    }

    fn read_vint(&mut self) -> Result<i32> {
        // Not in Java (stage-3 optimisation, `cx_bterms_seek`): a one-byte
        // `vInt` -- nearly every suffix length and many a `docFreq` -- is
        // read without building a `SliceInput`.
        if let Some(&b) = self.bytes.get(self.pos) {
            if b < 0x80 {
                self.pos = self.pos.saturating_add(1);
                return Ok(i32::from(b));
            }
        }
        let mut input = self.input()?;
        let v = input.read_vint()?;
        self.pos = input.position();
        Ok(v)
    }

    fn read_vlong(&mut self) -> Result<i64> {
        let mut input = self.input()?;
        let v = input.read_vlong()?;
        self.pos = input.position();
        Ok(v)
    }

    /// A `vInt` length.
    fn read_len(&mut self) -> Result<usize> {
        let n = self.read_vint()?;
        usize::try_from(n).map_err(|_| corrupt(format!("invalid suffix length: {n}")))
    }

    /// The next `n` bytes, without moving past them.
    fn peek(&self, n: usize) -> Result<&'a [u8]> {
        self.pos
            .checked_add(n)
            .and_then(|end| self.bytes.get(self.pos..end))
            .ok_or_else(|| corrupt(format!("suffix of {n} bytes past its block")))
    }

    /// `readBytes`: the next `n` bytes.
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self.peek(n)?;
        self.pos = self.pos.saturating_add(n);
        Ok(bytes)
    }
}

/// `FieldReader.SegmentTermsEnum`.
#[derive(Debug)]
pub struct SegmentTermsEnum<'r> {
    fr: &'r BlockTermsFieldReader,
    /// `in.getFilePointer()`.
    fp: usize,
    state: BlockTermState,
    do_ord: bool,
    index_enum: Option<FieldIndexEnum<'r>>,
    term: Vec<u8>,
    /// `indexIsCurrent`: the index enum is still on the current block's
    /// index term.
    index_is_current: bool,
    /// `didIndexNext`: `next_index_term` brackets the current block.
    did_index_next: bool,
    /// `nextIndexTerm` (Java aliases the index enum's term; a copy here).
    next_index_term: Option<Vec<u8>>,
    /// `seekPending`: after `seekExact(TermState)`, the real seek waits
    /// for a `next()`.
    seek_pending: bool,
    /// `termSuffixesReader`.
    suffixes: Blob<'r>,
    /// `termBlockPrefix`: the prefix every term of the block shares.
    term_block_prefix: usize,
    /// `blockTermCount`.
    block_term_count: i32,
    /// `freqReader`.
    freqs: Blob<'r>,
    /// `bytesReader`.
    meta: Blob<'r>,
    /// `metaDataUpto`.
    meta_data_upto: i32,
}

/// The field's index enum, or Java's `IllegalStateException`.
fn loaded<'e, 'r>(
    index_enum: &'e mut Option<FieldIndexEnum<'r>>,
) -> Result<&'e mut FieldIndexEnum<'r>> {
    index_enum
        .as_mut()
        .ok_or(Error::IllegalState("terms index was not loaded"))
}

impl<'r> SegmentTermsEnum<'r> {
    /// `term.setLength(termBlockPrefix + suffix); readBytes(...)`: the next
    /// `suffix` bytes after the block's common prefix.
    fn read_term(&mut self, suffix: usize) -> Result<()> {
        let bytes = self.suffixes.read_bytes(suffix)?;
        if self.term.len() < self.term_block_prefix {
            return Err(corrupt(format!(
                "block prefix {} longer than the term before it",
                self.term_block_prefix
            )));
        }
        self.term.truncate(self.term_block_prefix);
        self.term.extend_from_slice(bytes);
        Ok(())
    }

    /// `seekCeil(target)`: the index's block for `target`, scanned
    /// against the block's common prefix and then each term's suffix in
    /// place; a target past the current term but before the next index term
    /// scans on from where the enum is.
    ///
    /// # Errors
    /// [`Error::IllegalState`] without a terms index for the field;
    /// `Corrupted` for a block or index that does not decode.
    // ARITH: `ord` and `termBlockOrd` count terms of a field and of a block
    // (`int`s Java increments alike); `blockTermCount >= 1` once a block is
    // loaded; `common < term_block_prefix`; `term_block_prefix + suffix`
    // is checked.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        loaded(&mut self.index_enum)?;
        // Check cache
        let mut do_seek = true;
        if self.index_is_current {
            match self.term.as_slice().cmp(target) {
                Ordering::Equal => return Ok(SeekStatus::Found),
                Ordering::Less => {
                    // Target term is after current term
                    if !self.did_index_next {
                        let index_enum = loaded(&mut self.index_enum)?;
                        let next = index_enum.next()?;
                        let next_term = next.map(|_| index_enum.term().to_vec());
                        self.next_index_term = next_term;
                        self.did_index_next = true;
                    }
                    // Target's in current index block (or the last block):
                    if self
                        .next_index_term
                        .as_deref()
                        .is_none_or(|next| target < next)
                    {
                        do_seek = false;
                    }
                }
                Ordering::Greater => {}
            }
        }
        if do_seek {
            // Ask terms index to find biggest indexed term (= first term in a
            // block) that's <= our text:
            let len = self.fr.tib.len();
            let index_enum = loaded(&mut self.index_enum)?;
            let fp = index_enum.seek(target)?;
            self.fp = file_pointer(fp, len, "block file pointer")?;
            if !self.next_block()? {
                return Err(corrupt("terms index points at the end of the field"));
            }
            // `nextBlock` cleared `indexIsCurrent`.
            self.index_is_current = true;
            self.did_index_next = false;
            let index_enum = loaded(&mut self.index_enum)?;
            let ord = if self.do_ord {
                Some(index_enum.ord()?)
            } else {
                None
            };
            self.term.clear();
            self.term.extend_from_slice(index_enum.term());
            if let Some(ord) = ord {
                self.state.ord = ord - 1;
            }
        } else if self.state.term_block_ord == self.block_term_count && !self.next_block()? {
            // Target is beyond current block, but the current block is the
            // last
            self.index_is_current = false;
            return Ok(SeekStatus::End);
        }
        self.seek_pending = false;

        let mut common = 0;
        // Scan within block. We could do this by calling _next() and testing
        // the resulting term, but this is wasteful. Instead, we first confirm
        // the target matches the common prefix of this block, and then we
        // scan the term bytes directly from the termSuffixesreader's byte[],
        // saving a copy into the BytesRef term per term. Only when we return
        // do we then copy the bytes into the term.
        loop {
            // First, see if target term matches common prefix in this block:
            if common < self.term_block_prefix {
                let (Some(&have), Some(&want)) = (self.term.get(common), target.get(common)) else {
                    return Err(corrupt(format!(
                        "block prefix {} longer than the term or the target",
                        self.term_block_prefix
                    )));
                };
                match have.cmp(&want) {
                    Ordering::Less => {
                        // TODO: maybe we should store common prefix in block
                        // header? (instead of relying on last term of previous
                        // block)

                        // Target's prefix is after the common block prefix,
                        // so term cannot be in this block but it could be in
                        // next block. We must scan to end-of-block to set
                        // common prefix for next block:
                        if self.state.term_block_ord < self.block_term_count {
                            while self.state.term_block_ord < self.block_term_count - 1 {
                                self.state.term_block_ord += 1;
                                self.state.ord += 1;
                                let n = self.suffixes.read_len()?;
                                self.suffixes.read_bytes(n)?;
                            }
                            let suffix = self.suffixes.read_len()?;
                            self.read_term(suffix)?;
                        }
                        self.state.ord += 1;
                        if !self.next_block()? {
                            self.index_is_current = false;
                            return Ok(SeekStatus::End);
                        }
                        common = 0;
                    }
                    Ordering::Greater => {
                        // Target's prefix is before the common prefix of this
                        // block, so we position to start of block and return
                        // NOT_FOUND:
                        let suffix = self.suffixes.read_len()?;
                        self.read_term(suffix)?;
                        return Ok(SeekStatus::NotFound);
                    }
                    Ordering::Equal => common += 1,
                }
                continue;
            }

            // Test every term in this block
            loop {
                self.state.term_block_ord += 1;
                self.state.ord += 1;
                let suffix = self.suffixes.read_len()?;
                // We know the prefix matches, so just compare the new suffix:
                let term_len = self
                    .term_block_prefix
                    .checked_add(suffix)
                    .ok_or_else(|| corrupt("term length overflows"))?;
                let bytes = self.suffixes.peek(suffix)?;
                let limit = term_len.min(target.len());
                let mut next = false;
                let tail = target.get(self.term_block_prefix..limit).unwrap_or(&[]);
                for (&have, &want) in bytes.iter().zip(tail) {
                    match have.cmp(&want) {
                        Ordering::Less => {
                            // Current term is still before the target; keep
                            // scanning
                            next = true;
                            break;
                        }
                        Ordering::Greater => {
                            // Done! Current term is after target. Stop here,
                            // fill in real term, return NOT_FOUND.
                            self.read_term(suffix)?;
                            return Ok(SeekStatus::NotFound);
                        }
                        Ordering::Equal => {}
                    }
                }
                if !next && target.len() <= term_len {
                    self.read_term(suffix)?;
                    return Ok(if target.len() == term_len {
                        // Done! Exact match. Stop here, fill in real term,
                        // return FOUND.
                        SeekStatus::Found
                    } else {
                        SeekStatus::NotFound
                    });
                }
                if self.state.term_block_ord == self.block_term_count {
                    // Must pre-fill term for next block's common prefix
                    self.read_term(suffix)?;
                    break;
                }
                self.suffixes.read_bytes(suffix)?;
            }

            // The purpose of the terms dict index is to seek the enum to the
            // closest index term before the term we are looking for. So we
            // should never cross another index term (besides the first one)
            // while we are scanning:
            if !self.next_block()? {
                self.index_is_current = false;
                return Ok(SeekStatus::End);
            }
            common = 0;
        }
    }

    /// `seekExact(target)`: `seekCeil(target) == FOUND`.
    ///
    /// # Errors
    /// As [`seek_ceil`](Self::seek_ceil).
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        Ok(self.seek_ceil(target)? == SeekStatus::Found)
    }

    /// `next()`: the next term, or `None` past the field's last (and on
    /// every call after it). After `seekExact(TermState)` it first loads the
    /// state's block and steps up to the state's term.
    ///
    /// # Errors
    /// `Corrupted` for a block that does not decode.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        // If seek was previously called and the term was cached, usually
        // caller is just going to pull a D/&PEnum or get docFreq, etc. But,
        // if they then call next(), this method catches up all internal state
        // so next() works properly:
        if self.seek_pending {
            self.fp = self.state.block_file_pointer;
            let pending_seek_count = self.state.term_block_ord;
            if !self.next_block()? {
                return Err(corrupt("term state points at the end of the field"));
            }
            let sav_ord = self.state.ord;
            // Rescan all terms up to the pending term
            while self.state.term_block_ord < pending_seek_count {
                if !self.next_term()? {
                    return Err(corrupt("term state past its block"));
                }
            }
            self.seek_pending = false;
            self.state.ord = sav_ord;
        }
        Ok(if self.next_term()? {
            Some(&self.term)
        } else {
            None
        })
    }

    /// `_next()`: decodes only the next term's bytes; `false` past the
    /// field's last term.
    // ARITH: `termBlockOrd < blockTermCount` (an `int`) before it grows;
    // `ord` counts the field's terms.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_term(&mut self) -> Result<bool> {
        if self.state.term_block_ord == self.block_term_count && !self.next_block()? {
            self.index_is_current = false;
            return Ok(false);
        }
        let suffix = self.suffixes.read_len()?;
        self.read_term(suffix)?;
        self.state.term_block_ord += 1;
        // NOTE: meaningless in the non-ord case
        self.state.ord += 1;
        Ok(true)
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        &self.term
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn doc_freq(&mut self) -> Result<i32> {
        self.decode_meta_data()?;
        Ok(self.state.doc_freq)
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn total_term_freq(&mut self) -> Result<i64> {
        self.decode_meta_data()?;
        Ok(self.state.total_term_freq)
    }

    /// `postings(reuse, flags)` over the group's `.doc`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on an enum that is not on a term; what
    /// decoding the metadata or the `.doc` fails with.
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.decode_meta_data()?;
        let state = self.state;
        if state.doc_freq <= 0 {
            return Err(Error::Unsupported("the enum is not positioned on a term"));
        }
        let info = &self.fr.info;
        if state.doc_freq == 1 {
            return Ok(postings::singleton_postings(
                state.meta,
                state.total_term_freq,
            )?);
        }
        Ok(doc_in.read_postings_with_flags(
            state.meta,
            state.doc_freq,
            info.index_options,
            info.store_payloads,
            flags,
        )?)
    }

    /// `seekExact(target, otherState)`: positions on `target` without
    /// reading; a `next()` catches up.
    pub fn seek_exact_state(&mut self, target: &[u8], other_state: &BlockTermState) {
        self.state = *other_state;
        self.seek_pending = true;
        self.index_is_current = false;
        self.term.clear();
        self.term.extend_from_slice(target);
    }

    /// `termState()`: the current term's decoded state.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn term_state(&mut self) -> Result<BlockTermState> {
        self.decode_meta_data()?;
        Ok(self.state)
    }

    /// `seekExact(ord)`: the index term at or before `ord`, then `_next()`
    /// up to it.
    ///
    /// # Errors
    /// [`Error::IllegalState`] without a terms index for the field;
    /// [`Error::Unsupported`] over the variable-gap index;
    /// [`Error::IllegalArgument`] for an ord outside `0..size()` (Java: an
    /// `assert`); `Corrupted` for a block or index that does not decode.
    // ARITH: `0 <= ord < numTerms` and the index term's ord is `<= ord`,
    // so `ord - state.ord` is in `1..=numTerms`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        loaded(&mut self.index_enum)?;
        if ord < 0 || ord >= self.fr.num_terms {
            return Err(Error::IllegalArgument(format!("ord out of bounds: {ord}")));
        }
        let len = self.fr.tib.len();
        let fp = loaded(&mut self.index_enum)?.seek_ord(ord)?;
        self.fp = file_pointer(fp, len, "block file pointer")?;
        if !self.next_block()? {
            return Err(corrupt("terms index points at the end of the field"));
        }
        self.index_is_current = true;
        self.did_index_next = false;
        self.seek_pending = false;
        let index_enum = loaded(&mut self.index_enum)?;
        let index_ord = index_enum.ord()?;
        self.term.clear();
        self.term.extend_from_slice(index_enum.term());
        self.state.ord = index_ord - 1;
        // Now, scan:
        let mut left = ord - self.state.ord;
        while left > 0 {
            if !self.next_term()? {
                return Err(corrupt(format!("ord {ord} past the field's terms")));
            }
            left -= 1;
        }
        Ok(())
    }

    /// `ord()`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when the terms index has no ords.
    pub fn ord(&self) -> Result<i64> {
        if !self.do_ord {
            return Err(Error::Unsupported("ord"));
        }
        Ok(self.state.ord)
    }

    /// `nextBlock()`: the next block's term count, common prefix and three
    /// blobs -- the statistics and metadata decoded only when asked for.
    /// `false` at the field's end marker, where the enum then stays (Java
    /// reads stale suffix bytes, or past the marker, on the next call).
    fn next_block(&mut self) -> Result<bool> {
        let tib: &'r [u8] = &self.fr.tib;
        self.state.block_file_pointer = self.fp;
        let mut input = SliceInput::new(tib);
        input.seek(self.fp)?;
        let count = input.read_vint()?;
        if count == 0 {
            self.block_term_count = 0;
            self.state.term_block_ord = 0;
            return Ok(false);
        }
        if count < 0 {
            return Err(corrupt(format!("invalid block term count: {count}")));
        }
        let prefix = input.read_vint()?;
        let prefix = usize::try_from(prefix)
            .map_err(|_| corrupt(format!("invalid block prefix: {prefix}")))?;
        let mut blob = || -> Result<Blob<'r>> {
            let len = input.read_vint()?;
            let len =
                usize::try_from(len).map_err(|_| corrupt(format!("invalid blob length: {len}")))?;
            let start = input.position();
            let end = start
                .checked_add(len)
                .ok_or_else(|| corrupt("blob past the end"))?;
            let bytes = input.slice(start, end)?;
            input.seek(end)?;
            Ok(Blob { bytes, pos: 0 })
        };
        let suffixes = blob()?;
        let freqs = blob()?;
        let meta = blob()?;
        self.fp = input.position();
        self.block_term_count = count;
        self.term_block_prefix = prefix;
        self.suffixes = suffixes;
        self.freqs = freqs;
        self.meta = meta;
        self.meta_data_upto = 0;
        self.state.term_block_ord = 0;
        self.index_is_current = false;
        Ok(true)
    }

    /// `decodeMetaData()`: every term's statistics and metadata up to the
    /// current one not yet decoded; nothing while a seek is pending (the
    /// state is the one handed in).
    // ARITH: `meta_data_upto < limit`, an `int`; `docFreq + (ttf - docFreq)`
    // is checked.
    #[allow(clippy::arithmetic_side_effects)]
    fn decode_meta_data(&mut self) -> Result<()> {
        if self.seek_pending {
            return Ok(());
        }
        let info = &self.fr.info;
        let limit = self.state.term_block_ord;
        let mut absolute = self.meta_data_upto == 0;
        while self.meta_data_upto < limit {
            // TODO: we could make "tiers" of metadata, ie, decode docFreq/totalTF
            // but not the rest.
            self.state.doc_freq = self.freqs.read_vint()?;
            self.state.total_term_freq = if info.index_options == IndexOptions::Docs {
                // all postings have tf=1
                i64::from(self.state.doc_freq)
            } else {
                i64::from(self.state.doc_freq)
                    .checked_add(self.freqs.read_vlong()?)
                    .ok_or_else(|| corrupt("totalTermFreq overflows"))?
            };
            let mut input = self.meta.input()?;
            self.state.meta = postings::decode_term_metadata(
                &mut input,
                self.state.doc_freq,
                absolute,
                self.state.meta,
                info.index_options,
                info.store_payloads,
                self.state.total_term_freq,
            )?;
            self.meta.pos = input.position();
            self.meta_data_upto += 1;
            absolute = false;
        }
        Ok(())
    }
}

impl BlockTermsReader {
    /// Every field's terms with their statistics and postings metadata
    /// ([`crate::terms_dictionary::TermsDictionaryFormat::read`]).
    ///
    /// # Errors
    /// A block or term metadata that does not decode.
    pub fn dictionary_fields(&self) -> Result<Vec<crate::postings_writer::DictionaryField>> {
        let mut fields = Vec::with_capacity(self.size());
        for name in self.field_names() {
            let Some(terms) = self.terms(name) else {
                continue;
            };
            let info = terms.info();
            let mut field = crate::postings_writer::DictionaryField {
                field_number: info.number,
                index_options: info.index_options,
                has_payloads: info.store_payloads,
                doc_count: terms.doc_count(),
                terms: Vec::new(),
            };
            let mut it = terms.iterator();
            while let Some(bytes) = it.next()? {
                let bytes = bytes.to_vec();
                let state = it.term_state()?;
                field.terms.push(crate::postings_writer::DictionaryTerm {
                    bytes,
                    doc_freq: state.doc_freq,
                    total_term_freq: state.total_term_freq,
                    meta: state.meta,
                });
            }
            fields.push(field);
        }
        Ok(fields)
    }
}
