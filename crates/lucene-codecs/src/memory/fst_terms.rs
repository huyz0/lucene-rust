//! Port of `FSTTermsWriter` and `FSTTermsReader` (its `TermsReader`,
//! `SegmentTermsEnum` and `IntersectTermsEnum`).

use std::collections::BTreeMap;
use std::fmt;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::automaton::{AutomatonType, ByteRunnable, CompiledAutomaton};

use super::fst_term_outputs::{FstTermOutputs, TermData};
use super::{
    corrupt, Error, Result, SeekStatus, POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT,
    POSTINGS_VERSION_START,
};
use crate::field_infos::{FieldInfo, IndexOptions};
use crate::fst::InputType;
use crate::fst_compiler::{FstCompilerBuilder, FstOutputs};
use crate::fst_util::{self, target_has_arcs, Arc, BytesRefFstEnum, TypedFst};
use crate::postings::{self, DocInput, Postings, PostingsFlags, TermMetadata};
use crate::postings_writer::{self, FieldTermStates};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `FSTTermsWriter.TERMS_EXTENSION`.
pub const TERMS_EXTENSION: &str = "tfp";
/// `FSTTermsWriter.TERMS_CODEC_NAME`.
pub const TERMS_CODEC_NAME: &str = "FSTTerms";
/// `FSTTermsWriter.TERMS_VERSION_START`.
pub const TERMS_VERSION_START: i32 = 2;
/// `FSTTermsWriter.TERMS_VERSION_CURRENT`.
pub const TERMS_VERSION_CURRENT: i32 = TERMS_VERSION_START;

/// `FSTTermsWriter.FieldMetaData`.
struct FieldMetaData {
    number: i32,
    index_options: IndexOptions,
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    /// `dict.save(out, out)`: the FST's metadata and body.
    dict: Vec<u8>,
}

/// `FSTTermsWriter`: the `.tfp` of one postings format instance, fed one
/// field at a time by the postings writer
/// ([`postings_writer::write_postings`]).
pub struct FstTermsWriter {
    out: Vec<u8>,
    fields: Vec<FieldMetaData>,
}

impl fmt::Debug for FstTermsWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FstTermsWriter")
            .field("bytes", &self.out.len())
            .field("fields", &self.fields.len())
            .finish()
    }
}

impl FstTermsWriter {
    /// The constructor: the `.tfp` header and `postingsWriter.init(out)`.
    pub fn new(state: &SegmentWriteState<'_>) -> Self {
        let mut out = Vec::new();
        codec_util::write_index_header(
            &mut out,
            TERMS_CODEC_NAME,
            TERMS_VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        postings_writer::write_postings_terms_header(
            &mut out,
            state.segment_id,
            state.segment_suffix,
        );
        FstTermsWriter {
            out,
            fields: Vec::new(),
        }
    }

    /// `write(fields, norms)` for one field: every term's [`TermData`]
    /// (`TermsWriter.finishTerm`) into the field's FST, kept when the field
    /// has a term (`TermsWriter.finish`).
    ///
    /// # Errors
    /// An FST the compiler refuses (terms out of order).
    pub fn write_field(&mut self, field: &FieldTermStates<'_>) -> Result<()> {
        let dict = if field.index_options == IndexOptions::Docs {
            build_dict::<FstTermOutputs<false>>(field)?
        } else {
            build_dict::<FstTermOutputs<true>>(field)?
        };
        let Some(dict) = dict else {
            return Ok(());
        };
        let has_freq = field.index_options != IndexOptions::Docs;
        let mut sum_total_term_freq = 0i64;
        let mut sum_doc_freq = 0i64;
        for i in 0..field.len() {
            sum_total_term_freq = sum_total_term_freq.wrapping_add(field.total_term_freq(i));
            sum_doc_freq = sum_doc_freq.wrapping_add(i64::from(field.doc_freq(i)));
        }
        self.fields.push(FieldMetaData {
            number: field.field_number,
            index_options: field.index_options,
            num_terms: field.len() as i64,
            sum_total_term_freq: if has_freq { sum_total_term_freq } else { -1 },
            sum_doc_freq,
            doc_count: field.doc_count,
            dict,
        });
        Ok(())
    }

    /// `close()`: the field summary, `dirStart` and the footer.
    pub fn finish(mut self) -> Vec<u8> {
        let out = &mut self.out;
        let dir_start = out.len() as i64;
        // At most one entry per field of a segment.
        out.write_vint(self.fields.len() as i32);
        for field in &self.fields {
            out.write_vint(field.number);
            out.write_vlong(field.num_terms);
            if field.index_options != IndexOptions::Docs {
                out.write_vlong(field.sum_total_term_freq);
            }
            out.write_vlong(field.sum_doc_freq);
            out.write_vint(field.doc_count);
            out.write_bytes(&field.dict);
        }
        out.write_i64(dir_start);
        codec_util::write_footer(out);
        self.out
    }
}

/// One field's FST, saved; `None` for a field without terms.
fn build_dict<O: FstOutputs<Value = TermData>>(
    field: &FieldTermStates<'_>,
) -> Result<Option<Vec<u8>>> {
    if field.is_empty() {
        return Ok(None);
    }
    let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<O>();
    let mut meta = Vec::new();
    for i in 0..field.len() {
        meta.clear();
        field.encode_term(&mut meta, i);
        let data = TermData {
            bytes: (!meta.is_empty()).then(|| meta.clone()),
            doc_freq: field.doc_freq(i),
            total_term_freq: field.total_term_freq(i),
        };
        compiler.add_bytes(field.term(i), data)?;
    }
    Ok(compiler.compile().map(|fst| fst.save::<O>()))
}

/// A field's dictionary, by whether its outputs carry `totalTermFreq`.
#[derive(Debug)]
enum Dict {
    /// `IndexOptions.DOCS`.
    Docs(TypedFst<FstTermOutputs<false>>),
    /// Frequencies and up.
    Freqs(TypedFst<FstTermOutputs<true>>),
}

/// `FSTTermsReader`: every field of a `.tfp`, in name order.
#[derive(Debug)]
pub struct FstTermsReader {
    fields: BTreeMap<String, FstFieldReader>,
}

impl FstTermsReader {
    /// `new FSTTermsReader(state, postingsReader)`: checks the header and
    /// the whole file's checksum, the postings writer's header, and reads
    /// each field's summary and FST.
    ///
    /// # Errors
    /// `Corrupted` for a bad header, footer or checksum, a field number
    /// `state.field_infos` lacks, a field summary `checkFieldSummary`
    /// refuses, a duplicate field; [`Error::Fst`] for an FST that does not
    /// decode; [`Error::Unsupported`] for a postings block size other than
    /// `Lucene104PostingsFormat`'s.
    pub fn open(tfp: &[u8], state: &SegmentReadState<'_>) -> Result<Self> {
        let mut input = SliceInput::new(tfp);
        codec_util::check_index_header(
            &mut input,
            TERMS_CODEC_NAME,
            TERMS_VERSION_START,
            TERMS_VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        // `CodecUtil.checksumEntireFile(in)`.
        let footer_start = tfp
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt(format!("file too short for a footer: {}", tfp.len())))?;
        codec_util::check_whole_file_footer(tfp, footer_start)?;
        // `Lucene104PostingsReader.init(in, state)`.
        codec_util::check_index_header(
            &mut input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        let index_block_size = input.read_vint()?;
        if index_block_size != postings::BLOCK_SIZE {
            return Err(Error::Unsupported(
                "index-time BLOCK_SIZE differs from read-time BLOCK_SIZE",
            ));
        }
        // `seekDir(in)`.
        let dir_at = footer_start
            .checked_sub(8)
            .ok_or_else(|| corrupt("file too short for dirStart"))?;
        input.seek(dir_at)?;
        let dir_start = input.read_i64()?;
        let dir_start = usize::try_from(dir_start)
            .ok()
            .filter(|&d| d <= dir_at)
            .ok_or_else(|| corrupt(format!("invalid dirStart: {dir_start}")))?;
        let mut input = SliceInput::new(&tfp[..dir_at]);
        input.seek(dir_start)?;

        let num_fields = input.read_vint()?;
        let mut fields = BTreeMap::new();
        for _ in 0..num_fields {
            let field_number = input.read_vint()?;
            let info = state
                .field_infos
                .field_by_number(field_number)
                .ok_or_else(|| corrupt(format!("invalid field number: {field_number}")))?;
            let num_terms = input.read_vlong()?;
            let sum_total_term_freq = input.read_vlong()?;
            // Without frequencies sumTotalTermFreq == sumDocFreq, one value.
            let sum_doc_freq = if info.index_options == IndexOptions::Docs {
                sum_total_term_freq
            } else {
                input.read_vlong()?
            };
            let doc_count = input.read_vint()?;
            let rest = input.as_slice();
            let (dict, used) = if info.index_options == IndexOptions::Docs {
                let (fst, used) = TypedFst::read_prefix(rest)?;
                (Dict::Docs(fst), used)
            } else {
                let (fst, used) = TypedFst::read_prefix(rest)?;
                (Dict::Freqs(fst), used)
            };
            input.skip(used)?;
            let field = FstFieldReader {
                info: info.clone(),
                num_terms,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
                dict,
            };
            check_field_summary(state.max_doc, &field)?;
            if fields.insert(info.name.clone(), field).is_some() {
                return Err(corrupt(format!("duplicate fields: {}", info.name)));
            }
        }
        Ok(FstTermsReader { fields })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&FstFieldReader> {
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
}

impl fmt::Display for FstTermsReader {
    /// `toString()`, the postings reader by its class name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FSTTermsReader(fields={},delegate=Lucene104PostingsReader)",
            self.fields.len()
        )
    }
}

/// `checkFieldSummary`.
fn check_field_summary(max_doc: i32, field: &FstFieldReader) -> Result<()> {
    if field.doc_count < 0 || field.doc_count > max_doc {
        return Err(corrupt(format!(
            "invalid docCount: {} maxDoc: {max_doc}",
            field.doc_count
        )));
    }
    if field.sum_doc_freq < i64::from(field.doc_count) {
        return Err(corrupt(format!(
            "invalid sumDocFreq: {} docCount: {}",
            field.sum_doc_freq, field.doc_count
        )));
    }
    if field.sum_total_term_freq < field.sum_doc_freq {
        return Err(corrupt(format!(
            "invalid sumTotalTermFreq: {} sumDocFreq: {}",
            field.sum_total_term_freq, field.sum_doc_freq
        )));
    }
    Ok(())
}

/// `FSTTermsReader.TermsReader`: one field's statistics and dictionary.
#[derive(Debug)]
pub struct FstFieldReader {
    info: FieldInfo,
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    dict: Dict,
}

impl fmt::Display for FstFieldReader {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FSTTerms(terms={},postings={},positions={},docs={})",
            self.num_terms, self.sum_doc_freq, self.sum_total_term_freq, self.doc_count
        )
    }
}

impl FstFieldReader {
    /// The field's `FieldInfo`.
    pub fn field_info(&self) -> &FieldInfo {
        &self.info
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

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        !matches!(
            self.info.index_options,
            IndexOptions::None | IndexOptions::Docs
        )
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }

    /// `iterator()`: a `SegmentTermsEnum`, unpositioned.
    pub fn iter(&self) -> FstSegmentTermsEnum<'_> {
        FstSegmentTermsEnum {
            field: self,
            inner: match &self.dict {
                Dict::Docs(fst) => SegmentEnum::Docs(BytesRefFstEnum::new(fst)),
                Dict::Freqs(fst) => SegmentEnum::Freqs(BytesRefFstEnum::new(fst)),
            },
            term: None,
            meta: TermData::NO_OUTPUT,
        }
    }

    /// `intersect(compiled, startTerm)`: the terms `compiled` accepts, from
    /// the first one after `start_term` (Java's `startTerm`, exclusive).
    ///
    /// # Errors
    /// [`Error::Unsupported`] for an automaton that is not `NORMAL`
    /// (`IllegalArgumentException`: use `CompiledAutomaton.getTermsEnum`);
    /// [`Error::Fst`] for a dictionary that does not decode.
    pub fn intersect<'a>(
        &'a self,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<FstIntersectTermsEnum<'a>> {
        let fsa = match (compiled.automaton_type, compiled.get_byte_runnable()) {
            (AutomatonType::NORMAL, Some(run)) => run,
            _ => {
                return Err(Error::Unsupported(
                    "please use CompiledAutomaton.getTermsEnum instead",
                ))
            }
        };
        Ok(FstIntersectTermsEnum {
            field: self,
            inner: match &self.dict {
                Dict::Docs(fst) => IntersectEnum::Docs(Intersect::new(fst, fsa, start_term)?),
                Dict::Freqs(fst) => IntersectEnum::Freqs(Intersect::new(fst, fsa, start_term)?),
            },
            decoded: None,
        })
    }

    /// `postingsReader.decodeTerm(bytes, fieldInfo, state, absolute=true)`
    /// over a term's output.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn term_metadata(&self, data: &TermData) -> Result<TermMetadata> {
        let bytes = data.bytes.as_deref().unwrap_or_default();
        let mut input = SliceInput::new(bytes);
        Ok(postings::decode_term_metadata(
            &mut input,
            data.doc_freq,
            true,
            TermMetadata::EMPTY,
            self.info.index_options,
            self.info.store_payloads,
            data.total_term_freq,
        )?)
    }

    /// `postings(reuse, flags)` for a term's output: its documents and
    /// frequencies, read from the group's `.doc` (a `docFreq == 1` term's
    /// one document is in its metadata).
    ///
    /// # Errors
    /// What decoding the metadata or the `.doc` fails with.
    pub fn postings(
        &self,
        data: &TermData,
        doc_in: &DocInput<'_>,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        let meta = self.term_metadata(data)?;
        let total_term_freq = term_total_term_freq(data);
        if data.doc_freq == 1 {
            return Ok(postings::singleton_postings(meta, total_term_freq)?);
        }
        Ok(doc_in.read_postings_with_flags(
            meta,
            data.doc_freq,
            self.info.index_options,
            self.info.store_payloads,
            flags,
        )?)
    }
}

/// `BaseTermsEnum.totalTermFreq()`: `docFreq` where the field has no
/// frequencies.
fn term_total_term_freq(data: &TermData) -> i64 {
    if data.total_term_freq == -1 {
        i64::from(data.doc_freq)
    } else {
        data.total_term_freq
    }
}

#[derive(Debug)]
enum SegmentEnum<'a> {
    Docs(BytesRefFstEnum<'a, FstTermOutputs<false>>),
    Freqs(BytesRefFstEnum<'a, FstTermOutputs<true>>),
}

/// `FSTTermsReader.TermsReader.SegmentTermsEnum`: every term of a field,
/// through `BytesRefFSTEnum`.
#[derive(Debug)]
pub struct FstSegmentTermsEnum<'a> {
    field: &'a FstFieldReader,
    inner: SegmentEnum<'a>,
    /// `term`, `None` when unpositioned or exhausted.
    term: Option<Vec<u8>>,
    /// `meta`: the current term's output.
    meta: TermData,
}

/// `updateEnum(pair)`.
fn update_enum(term: &mut Option<Vec<u8>>, meta: &mut TermData, pair: Option<(&[u8], &TermData)>) {
    match pair {
        None => *term = None,
        Some((t, m)) => {
            let buf = term.get_or_insert_with(Vec::new);
            buf.clear();
            buf.extend_from_slice(t);
            meta.clone_from(m);
        }
    }
}

impl FstSegmentTermsEnum<'_> {
    /// `next()`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        let pair = match &mut self.inner {
            SegmentEnum::Docs(e) => e.next()?,
            SegmentEnum::Freqs(e) => e.next()?,
        };
        update_enum(&mut self.term, &mut self.meta, pair);
        Ok(self.term.as_deref())
    }

    /// `seekExact(target)`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        let pair = match &mut self.inner {
            SegmentEnum::Docs(e) => e.seek_exact(target)?,
            SegmentEnum::Freqs(e) => e.seek_exact(target)?,
        };
        update_enum(&mut self.term, &mut self.meta, pair);
        Ok(self.term.is_some())
    }

    /// `seekCeil(target)`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let pair = match &mut self.inner {
            SegmentEnum::Docs(e) => e.seek_ceil(target)?,
            SegmentEnum::Freqs(e) => e.seek_ceil(target)?,
        };
        update_enum(&mut self.term, &mut self.meta, pair);
        Ok(match self.term.as_deref() {
            None => SeekStatus::End,
            Some(t) if t == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        self.term.as_deref()
    }

    /// The current term's output.
    pub fn term_data(&self) -> &TermData {
        &self.meta
    }

    /// `docFreq()`.
    pub fn doc_freq(&self) -> i32 {
        self.meta.doc_freq
    }

    /// `totalTermFreq()`.
    pub fn total_term_freq(&self) -> i64 {
        term_total_term_freq(&self.meta)
    }

    /// `termState()`'s postings half: the current term's decoded metadata.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn term_metadata(&self) -> Result<TermMetadata> {
        self.field.term_metadata(&self.meta)
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// What decoding the metadata or the `.doc` fails with.
    pub fn postings(&self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.field.postings(&self.meta, doc_in, flags)
    }

    /// `seekExact(ord)` / `ord()`: unsupported, as in Java.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn ord(&self) -> Result<i64> {
        Err(Error::Unsupported("FSTTermsReader terms have no ords"))
    }
}

/// `IntersectTermsEnum.Frame`.
#[derive(Debug, Clone)]
struct Frame {
    fst_arc: Arc<TermData>,
    output: TermData,
    fsa_state: i32,
}

/// `IntersectTermsEnum` over one of the dictionary's two output types.
struct Intersect<'a, O: FstOutputs<Value = TermData>> {
    fst: &'a TypedFst<O>,
    fsa: &'a dyn ByteRunnable,
    /// `term`: `None` when the enum is exhausted; `level == len + 1`.
    term: Option<Vec<u8>>,
    pending: bool,
    stack: Vec<Frame>,
    level: usize,
    /// To which level the metadata has been accumulated.
    meta_upto: usize,
    meta: TermData,
}

impl<O: FstOutputs<Value = TermData>> fmt::Debug for Intersect<'_, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Intersect")
            .field("term", &self.term)
            .field("level", &self.level)
            .finish()
    }
}

impl<'a, O: FstOutputs<Value = TermData>> Intersect<'a, O> {
    fn new(
        fst: &'a TypedFst<O>,
        fsa: &'a dyn ByteRunnable,
        start_term: Option<&[u8]>,
    ) -> Result<Self> {
        let first = fst.first_arc();
        // `loadVirtualFrame`: never popped. Its arc is only ever asked
        // `isFinal()` by `loadMetaData` at level 0, which the code below
        // answers without it.
        let virtual_frame = Frame {
            fst_arc: first.clone(),
            output: TermData::NO_OUTPUT,
            fsa_state: -1,
        };
        // `loadFirstFrame`.
        let first_frame = Frame {
            output: first.output().clone(),
            fst_arc: first,
            fsa_state: 0,
        };
        let mut e = Intersect {
            fst,
            fsa,
            term: None,
            pending: false,
            stack: vec![virtual_frame, first_frame],
            level: 0,
            meta_upto: 1,
            meta: TermData::NO_OUTPUT,
        };
        e.push_frame();
        match start_term {
            None => e.pending = e.is_accept(e.level),
            Some(start) => {
                e.do_seek_ceil(start)?;
                e.pending =
                    e.term.as_deref() != Some(start) && e.is_valid(e.level) && e.is_accept(e.level);
            }
        }
        Ok(e)
    }

    /// `fsa.step(state, label)`, dead once dead.
    // SENTINEL: `-1`, the automaton's dead state, as `ByteRunnable.step`.
    fn fsa_step(&self, state: i32, label: i32) -> i32 {
        if state < 0 {
            -1
        } else {
            self.fsa.step(state, label)
        }
    }

    /// `loadMetaData()`: accumulates the outputs down to `level`, lazily.
    // ARITH: `meta_upto < level` inside the loop.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_meta_data(&mut self) {
        while self.meta_upto != self.level {
            let last = self.stack[self.meta_upto].output.clone();
            self.meta_upto += 1;
            let next = &mut self.stack[self.meta_upto];
            next.output = O::add(&next.output, &last);
        }
        if self.level == 0 {
            // The virtual frame's arc is not final.
            self.meta = self.stack[0].output.clone();
            return;
        }
        let last = &self.stack[self.meta_upto];
        self.meta = if last.fst_arc.is_final() {
            O::add(&last.output, last.fst_arc.next_final_output())
        } else {
            last.output.clone()
        };
    }

    fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.do_seek_ceil(target)?;
        self.load_meta_data();
        Ok(match self.term.as_deref() {
            None => SeekStatus::End,
            Some(t) if t == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }

    // ARITH: `level >= 1` wherever a frame is popped or a top frame read.
    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Result<bool> {
        if self.pending {
            self.pending = false;
            self.load_meta_data();
            return Ok(self.term.is_some());
        }
        'dfs: while self.level > 0 {
            self.new_frame();
            if self.load_expand_frame()? {
                // Has a valid target.
                self.push_frame();
                if self.is_accept(self.level) {
                    break;
                }
                continue;
            }
            self.pop_frame();
            while self.level > 0 {
                if self.load_next_frame()? {
                    // Has a valid sibling.
                    self.push_frame();
                    if self.is_accept(self.level) {
                        break 'dfs;
                    }
                    continue 'dfs;
                }
                self.pop_frame();
            }
            return Ok(false);
        }
        self.load_meta_data();
        Ok(self.term.is_some())
    }

    /// `doSeekCeil(target)`; `true` when positioned on a term.
    // ARITH: `upto < target.len()`, and `level >= 1` where frames are read.
    #[allow(clippy::arithmetic_side_effects)]
    fn do_seek_ceil(&mut self, target: &[u8]) -> Result<bool> {
        let mut frame: Option<()> = None;
        let mut upto = 0;
        while upto < target.len() {
            // To the target's prefix, or its ceiling label.
            self.new_frame();
            let label = i32::from(target[upto]);
            frame = self.load_ceil_frame(label)?.then_some(());
            if frame.is_none() || self.stack[self.level + 1].fst_arc.label() != label {
                break;
            }
            self.push_frame();
            upto += 1;
        }
        if upto == target.len() {
            // Got the target.
            return Ok(self.term.is_some());
        }
        if frame.is_some() {
            // Got a larger term's prefix.
            self.push_frame();
            return if self.is_accept(self.level) {
                Ok(self.term.is_some())
            } else {
                self.next()
            };
        }
        while self.level > 0 {
            // Got the target's prefix: advance to a larger term.
            self.pop_frame();
            while self.level > 0 && !self.can_rewind(self.level + 1) {
                self.pop_frame();
            }
            if self.load_next_frame()? {
                self.push_frame();
                return if self.is_accept(self.level) {
                    Ok(self.term.is_some())
                } else {
                    self.next()
                };
            }
        }
        Ok(false)
    }

    /// `loadExpandFrame(topFrame(), newFrame())`.
    // ARITH: `level + 1` is a frame `new_frame` made.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_expand_frame(&mut self) -> Result<bool> {
        if !self.can_grow(self.level) {
            return Ok(false);
        }
        let (top_state, target) = {
            let top = &self.stack[self.level];
            (top.fsa_state, top.fst_arc.target())
        };
        let mut input = self.fst.bytes_reader();
        let fst = self.fst;
        let frame = &mut self.stack[self.level + 1];
        fst.read_first_real_target_arc(target, &mut frame.fst_arc, &mut input)?;
        let label = frame.fst_arc.label();
        let state = self.fsa_step(top_state, label);
        self.stack[self.level + 1].fsa_state = state;
        if state == -1 {
            return self.load_next_frame();
        }
        let frame = &mut self.stack[self.level + 1];
        frame.output = frame.fst_arc.output().clone();
        Ok(true)
    }

    /// `loadNextFrame(topFrame(), frame)`.
    // ARITH: `level + 1` is a frame `new_frame` made.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_next_frame(&mut self) -> Result<bool> {
        let at = self.level + 1;
        if !self.can_rewind(at) {
            return Ok(false);
        }
        let top_state = self.stack[self.level].fsa_state;
        let mut input = self.fst.bytes_reader();
        while !self.stack[at].fst_arc.is_last() {
            self.fst
                .read_next_real_arc(&mut self.stack[at].fst_arc, &mut input)?;
            let state = self.fsa_step(top_state, self.stack[at].fst_arc.label());
            self.stack[at].fsa_state = state;
            if state != -1 {
                break;
            }
        }
        let frame = &mut self.stack[at];
        if frame.fsa_state == -1 {
            return Ok(false);
        }
        frame.output = frame.fst_arc.output().clone();
        Ok(true)
    }

    /// `loadCeilFrame(label, topFrame(), frame)`: the arc with the least
    /// label `>= label` the automaton does not reject.
    // ARITH: `level + 1` is a frame `new_frame` made.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_ceil_frame(&mut self, label: i32) -> Result<bool> {
        let mut input = self.fst.bytes_reader();
        let top = &self.stack[self.level];
        let top_state = top.fsa_state;
        let Some(arc) = fst_util::read_ceil_arc(label, self.fst, &top.fst_arc, &mut input)? else {
            return Ok(false);
        };
        let state = self.fsa_step(top_state, arc.label());
        let frame = &mut self.stack[self.level + 1];
        frame.fst_arc = arc;
        frame.fsa_state = state;
        if state == -1 {
            return self.load_next_frame();
        }
        let frame = &mut self.stack[self.level + 1];
        frame.output = frame.fst_arc.output().clone();
        Ok(true)
    }

    /// `isAccept(frame)`: a term both the FST and the automaton accept.
    fn is_accept(&self, at: usize) -> bool {
        let frame = &self.stack[at];
        frame.fsa_state >= 0 && self.fsa.is_accept(frame.fsa_state) && frame.fst_arc.is_final()
    }

    /// `isValid(frame)`.
    fn is_valid(&self, at: usize) -> bool {
        self.stack[at].fsa_state != -1
    }

    /// `canGrow(frame)`.
    fn can_grow(&self, at: usize) -> bool {
        let frame = &self.stack[at];
        frame.fsa_state != -1 && target_has_arcs(&frame.fst_arc)
    }

    /// `canRewind(frame)`.
    fn can_rewind(&self, at: usize) -> bool {
        !self.stack[at].fst_arc.is_last()
    }

    /// `pushFrame(frame)`: the frame at `level + 1` becomes the top.
    // ARITH: the depth of a term.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_frame(&mut self) {
        let label = self.stack[self.level + 1].fst_arc.label();
        match &mut self.term {
            None => self.term = Some(Vec::new()),
            Some(t) => t.push(label as u8),
        }
        self.level += 1;
    }

    /// `popFrame()`: the top frame stays at `level + 1`.
    // ARITH: called with `level >= 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn pop_frame(&mut self) {
        match &mut self.term {
            Some(t) if !t.is_empty() => {
                t.pop();
            }
            _ => self.term = None,
        }
        self.level -= 1;
        self.meta_upto = self.meta_upto.min(self.level);
    }

    /// `newFrame()`: makes sure a frame exists at `level + 1`.
    // ARITH: the depth of a term.
    #[allow(clippy::arithmetic_side_effects)]
    fn new_frame(&mut self) {
        if self.stack.len() <= self.level + 1 {
            let template = self.stack[self.level].clone();
            self.stack.push(template);
        }
    }
}

#[derive(Debug)]
enum IntersectEnum<'a> {
    Docs(Intersect<'a, FstTermOutputs<false>>),
    Freqs(Intersect<'a, FstTermOutputs<true>>),
}

macro_rules! with_intersect {
    ($e:expr, $i:ident => $body:expr) => {
        match $e {
            IntersectEnum::Docs($i) => $body,
            IntersectEnum::Freqs($i) => $body,
        }
    };
}

/// `FSTTermsReader.TermsReader.IntersectTermsEnum`: the terms of a field a
/// compiled automaton accepts, by a depth-first walk of the FST and the
/// automaton together.
#[derive(Debug)]
pub struct FstIntersectTermsEnum<'a> {
    field: &'a FstFieldReader,
    inner: IntersectEnum<'a>,
    /// `decoded`: the current term's metadata, once decoded.
    decoded: Option<TermMetadata>,
}

impl FstIntersectTermsEnum<'_> {
    /// `next()`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        self.decoded = None;
        let found = with_intersect!(&mut self.inner, e => e.next()?);
        Ok(if found { self.term() } else { None })
    }

    /// `seekCeil(target)`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.decoded = None;
        with_intersect!(&mut self.inner, e => e.seek_ceil(target))
    }

    /// `seekExact(target)`: `BaseTermsEnum`'s, through `seekCeil`.
    ///
    /// # Errors
    /// [`Error::Fst`] for a dictionary that does not decode.
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        Ok(self.seek_ceil(target)? == SeekStatus::Found)
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        with_intersect!(&self.inner, e => e.term.as_deref())
    }

    /// The current term's output (`meta`).
    pub fn term_data(&self) -> &TermData {
        with_intersect!(&self.inner, e => &e.meta)
    }

    /// `docFreq()`.
    pub fn doc_freq(&self) -> i32 {
        self.term_data().doc_freq
    }

    /// `totalTermFreq()`.
    pub fn total_term_freq(&self) -> i64 {
        term_total_term_freq(self.term_data())
    }

    /// `decodeMetaData()`: the current term's postings metadata.
    ///
    /// # Errors
    /// `Corrupted` for metadata that does not decode.
    pub fn term_metadata(&mut self) -> Result<TermMetadata> {
        if let Some(meta) = self.decoded {
            return Ok(meta);
        }
        let meta = self.field.term_metadata(self.term_data())?;
        self.decoded = Some(meta);
        Ok(meta)
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// What decoding the metadata or the `.doc` fails with.
    pub fn postings(&self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.field.postings(self.term_data(), doc_in, flags)
    }
}
