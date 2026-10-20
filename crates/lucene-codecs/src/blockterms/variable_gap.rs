//! Port of `VariableGapTermsIndexWriter` and `VariableGapTermsIndexReader`:
//! the index terms an [`IndexTermSelector`] picks, each cut to the prefix
//! that tells it from the term before it, in one FST per field mapping the
//! prefix to its block's `.tib` file pointer. The FSTs' metadata go in the
//! `.tmv`, their bodies in the `.tiv`. No ords.

use std::collections::BTreeMap;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use super::{corrupt, Error, Result, TermStats, TermsIndexWriterBase};
use crate::field_infos::FieldInfo;
use crate::fst::{InputType, PositiveIntOutputs};
use crate::fst_compiler::{FstCompiler, FstCompilerBuilder};
use crate::fst_util::{BytesRefFstEnum, TypedFst};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `VariableGapTermsIndexWriter.TERMS_INDEX_EXTENSION`.
pub const TERMS_INDEX_EXTENSION: &str = "tiv";
/// `VariableGapTermsIndexWriter.TERMS_META_EXTENSION`.
pub const TERMS_META_EXTENSION: &str = "tmv";
/// `VariableGapTermsIndexWriter.CODEC_NAME`.
pub const CODEC_NAME: &str = "VariableGapTermsIndex";
/// `VariableGapTermsIndexWriter.META_CODEC_NAME`.
pub const META_CODEC_NAME: &str = "VariableGapTermsMeta";
/// `VariableGapTermsIndexWriter.VERSION_START`.
pub const VERSION_START: i32 = 4;
/// `VariableGapTermsIndexWriter.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_START;

/// `VariableGapTermsIndexWriter.IndexTermSelector`: which terms are index
/// terms (a field's first term always is).
pub trait IndexTermSelector {
    /// `isIndexTerm(term, stats)`: called on every term, in order.
    fn is_index_term(&mut self, term: &[u8], stats: TermStats) -> bool;
    /// `newField(fieldInfo)`: called when a field starts.
    fn new_field(&mut self, field: &FieldInfo);
}

/// `EveryNTermSelector`: every `interval`th term.
#[derive(Debug, Clone)]
pub struct EveryNTermSelector {
    count: i32,
    interval: i32,
}

impl EveryNTermSelector {
    /// `new EveryNTermSelector(interval)`.
    pub fn new(interval: i32) -> Self {
        EveryNTermSelector {
            count: interval,
            interval,
        }
    }
}

impl IndexTermSelector for EveryNTermSelector {
    // ARITH: `count` stays below `interval`, an `int`, before it grows.
    #[allow(clippy::arithmetic_side_effects)]
    fn is_index_term(&mut self, _term: &[u8], _stats: TermStats) -> bool {
        if self.count >= self.interval {
            self.count = 1;
            true
        } else {
            self.count += 1;
            false
        }
    }

    fn new_field(&mut self, _field: &FieldInfo) {
        self.count = self.interval;
    }
}

/// `EveryNOrDocFreqTermSelector`: a term whose `docFreq` reaches
/// `doc_freq_thresh`, or every `interval`th term.
#[derive(Debug, Clone)]
pub struct EveryNOrDocFreqTermSelector {
    count: i32,
    doc_freq_thresh: i32,
    interval: i32,
}

impl EveryNOrDocFreqTermSelector {
    /// `new EveryNOrDocFreqTermSelector(docFreqThresh, interval)`.
    pub fn new(doc_freq_thresh: i32, interval: i32) -> Self {
        EveryNOrDocFreqTermSelector {
            count: interval,
            doc_freq_thresh,
            interval,
        }
    }
}

impl IndexTermSelector for EveryNOrDocFreqTermSelector {
    // ARITH: `count` stays below `interval`, an `int`, before it grows.
    #[allow(clippy::arithmetic_side_effects)]
    fn is_index_term(&mut self, _term: &[u8], stats: TermStats) -> bool {
        if stats.doc_freq >= self.doc_freq_thresh || self.count >= self.interval {
            self.count = 1;
            true
        } else {
            self.count += 1;
            false
        }
    }

    fn new_field(&mut self, _field: &FieldInfo) {
        self.count = self.interval;
    }
}

/// `indexedTermPrefixLength(priorTerm, indexedTerm)`: the first differing
/// byte and one more, or one byte past the prior term (no more than the
/// indexed term).
// ARITH: `byte_idx < limit` and `prior_term.len()` is a slice length.
#[allow(clippy::arithmetic_side_effects)]
pub(super) fn indexed_term_prefix_length(prior_term: &[u8], indexed_term: &[u8]) -> usize {
    match prior_term
        .iter()
        .zip(indexed_term)
        .position(|(a, b)| a != b)
    {
        Some(byte_idx) => byte_idx + 1,
        None => (1 + prior_term.len()).min(indexed_term.len()),
    }
}

/// `VariableGapTermsIndexWriter.FSTFieldWriter`.
struct FstFieldWriter {
    compiler: FstCompiler<PositiveIntOutputs>,
    number: i32,
    last_term: Vec<u8>,
    first: bool,
}

/// The `.tiv` and `.tmv` a [`VariableGapTermsIndexWriter`] writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VariableGapFiles {
    /// `.tiv`: the FSTs' bodies.
    pub tiv: Vec<u8>,
    /// `.tmv`: the fields and their FSTs' metadata.
    pub tmv: Vec<u8>,
}

/// `VariableGapTermsIndexWriter`, over a selector.
pub struct VariableGapTermsIndexWriter<S: IndexTermSelector> {
    /// `out` and `metaOut`.
    files: VariableGapFiles,
    policy: S,
    current: Option<FstFieldWriter>,
}

impl<S: IndexTermSelector> std::fmt::Debug for VariableGapTermsIndexWriter<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariableGapTermsIndexWriter")
            .field("tiv_len", &self.files.tiv.len())
            .field("tmv_len", &self.files.tmv.len())
            .finish()
    }
}

impl<S: IndexTermSelector> VariableGapTermsIndexWriter<S> {
    /// `new VariableGapTermsIndexWriter(state, policy)`: both headers.
    pub fn new(state: &SegmentWriteState<'_>, policy: S) -> Self {
        let mut files = VariableGapFiles::default();
        codec_util::write_index_header(
            &mut files.tmv,
            META_CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        codec_util::write_index_header(
            &mut files.tiv,
            CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        VariableGapTermsIndexWriter {
            files,
            policy,
            current: None,
        }
    }
}

impl<S: IndexTermSelector> TermsIndexWriterBase for VariableGapTermsIndexWriter<S> {
    type Files = VariableGapFiles;

    /// `addField(field, termsFilePointer)`: `policy.newField`, and an FST
    /// whose empty input maps to the field's first block.
    fn add_field(&mut self, field: &FieldInfo, terms_file_pointer: i64) -> Result<()> {
        self.policy.new_field(field);
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        compiler.add_bytes(&[], terms_file_pointer)?;
        self.current = Some(FstFieldWriter {
            compiler,
            number: field.number,
            last_term: Vec::new(),
            first: true,
        });
        Ok(())
    }

    /// `FSTFieldWriter.checkIndexTerm`: what the policy says, and the first
    /// term; a term that is not one is kept as `lastTerm`.
    fn check_index_term(&mut self, text: &[u8], stats: TermStats) -> bool {
        let Some(f) = self.current.as_mut() else {
            return false;
        };
        if self.policy.is_index_term(text, stats) || f.first {
            f.first = false;
            true
        } else {
            f.last_term.clear();
            f.last_term.extend_from_slice(text);
            false
        }
    }

    /// `FSTFieldWriter.add`: the term's distinguishing prefix into the FST
    /// (the empty term is already there, as the field's start).
    fn add(&mut self, text: &[u8], _stats: TermStats, terms_file_pointer: i64) -> Result<()> {
        let f = self
            .current
            .as_mut()
            .ok_or(Error::IllegalState("no field was added"))?;
        if text.is_empty() {
            return Ok(());
        }
        let length = indexed_term_prefix_length(&f.last_term, text);
        f.compiler.add_bytes(&text[..length], terms_file_pointer)?;
        f.last_term.clear();
        f.last_term.extend_from_slice(text);
        Ok(())
    }

    /// `FSTFieldWriter.finish`: the field's number, the FST body's offset
    /// and its metadata into the `.tmv`, its body into the `.tiv`.
    fn finish(&mut self, _terms_file_pointer: i64) -> Result<()> {
        let mut f = self
            .current
            .take()
            .ok_or(Error::IllegalState("no field was added"))?;
        if let Some(compiled) = f.compiler.compile() {
            let fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiled);
            self.files.tmv.write_i32(f.number);
            self.files.tmv.write_vlong(self.files.tiv.len() as i64);
            let (meta, body) = fst.save_split();
            self.files.tmv.write_bytes(&meta);
            self.files.tiv.write_bytes(&body);
        }
        Ok(())
    }

    /// `close()`: the `.tmv`'s end marker, and both footers.
    fn close(mut self) -> Result<VariableGapFiles> {
        self.files.tmv.write_i32(-1);
        codec_util::write_footer(&mut self.files.tmv);
        codec_util::write_footer(&mut self.files.tiv);
        Ok(self.files)
    }
}

/// `VariableGapTermsIndexReader`: every field's FST, loaded at open.
#[derive(Debug)]
pub struct VariableGapTermsIndexReader {
    fields: BTreeMap<String, TypedFst<PositiveIntOutputs>>,
}

impl VariableGapTermsIndexReader {
    /// `new VariableGapTermsIndexReader(state)` over the `.tmv` and `.tiv`:
    /// both headers, the fields until `-1`, each FST's metadata and body,
    /// and both footers (the checksums first, so that a damaged byte is
    /// reported as one).
    ///
    /// # Errors
    /// `Corrupted` for a bad header, checksum or footer, a gap between two
    /// FST bodies, a field number `state.field_infos` lacks, a duplicate
    /// field, an FST that does not decode, or bytes left before a footer.
    pub fn open(tmv: &[u8], tiv: &[u8], state: &SegmentReadState<'_>) -> Result<Self> {
        let meta_end = tmv
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        let index_end = tiv
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(tmv, meta_end)?;
        codec_util::check_whole_file_footer(tiv, index_end)?;

        let mut meta_in = SliceInput::new(tmv);
        let mut index_in = SliceInput::new(tiv);
        codec_util::check_index_header(
            &mut meta_in,
            META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        codec_util::check_index_header(
            &mut index_in,
            CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        let mut fields = BTreeMap::new();
        // Read directory; every entry takes at least four bytes, so the loop
        // ends with the `.tmv`.
        loop {
            let field = meta_in.read_i32()?;
            if field == -1 {
                break;
            }
            let index_start = meta_in.read_vlong()?;
            let info = state
                .field_infos
                .field_by_number(field)
                .ok_or_else(|| corrupt(format!("invalid field number: {field}")))?;
            if index_in.position() as i64 != index_start {
                return Err(corrupt(format!(
                    "Gap in FST, expected position {}, got {index_start}",
                    index_in.position()
                )));
            }
            let meta_at = meta_in.position();
            let index_at = index_in.position();
            let (fst, meta_n, index_n) = TypedFst::<PositiveIntOutputs>::read_split_prefix(
                tmv.get(meta_at..meta_end).unwrap_or(&[]),
                tiv.get(index_at..index_end).unwrap_or(&[]),
            )?;
            meta_in.seek(meta_at.saturating_add(meta_n))?;
            index_in.seek(index_at.saturating_add(index_n))?;
            if fields.insert(info.name.clone(), fst).is_some() {
                return Err(corrupt(format!("duplicate field: {}", info.name)));
            }
        }
        codec_util::check_footer(&mut meta_in, tmv.len())?;
        codec_util::check_footer(&mut index_in, tiv.len())?;
        Ok(VariableGapTermsIndexReader { fields })
    }

    /// `supportsOrd()`.
    pub fn supports_ord(&self) -> bool {
        false
    }

    /// `getFieldEnum(fieldInfo)`: `None` when the field has no FST.
    pub fn field_enum(&self, field: &str) -> Option<VariableGapIndexEnum<'_>> {
        self.fields.get(field).map(|fst| VariableGapIndexEnum {
            fst_enum: BytesRefFstEnum::new(fst),
        })
    }

    /// The fields with an FST, in name order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }
}

/// `VariableGapTermsIndexReader.IndexEnum`.
pub struct VariableGapIndexEnum<'a> {
    fst_enum: BytesRefFstEnum<'a, PositiveIntOutputs>,
}

impl std::fmt::Debug for VariableGapIndexEnum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariableGapIndexEnum")
            .field("term", &self.term())
            .finish()
    }
}

impl VariableGapIndexEnum<'_> {
    /// `term()`: the index term (prefix) last landed on.
    pub fn term(&self) -> Option<&[u8]> {
        self.fst_enum.current().map(|(t, _)| t)
    }

    /// `seek(target)`: the last index term `<= target`, and its block's
    /// file pointer.
    ///
    /// # Errors
    /// `Corrupted` when nothing is `<= target` (the empty term always is in
    /// an FST this package writes; Java throws a `NullPointerException`),
    /// or the FST does not decode.
    pub fn seek(&mut self, target: &[u8]) -> Result<i64> {
        self.fst_enum
            .seek_floor(target)?
            .map(|(_, &fp)| fp)
            .ok_or_else(|| corrupt("terms index has no term <= the target"))
    }

    /// `next()`: the next index term's block, or `None` past the last
    /// (Java: `-1`).
    ///
    /// # Errors
    /// `Corrupted` for an FST that does not decode.
    // A cursor that also seeks, as Java's `FieldIndexEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<i64>> {
        Ok(self.fst_enum.next()?.map(|(_, &fp)| fp))
    }
}
