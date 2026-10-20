//! `SimpleTextFieldsReader`: the `.pst`'s fields, each field's terms loaded
//! into an FST on first use, and the docs/positions enums that read the text
//! in place.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

use lucene_util::fixed_bit_set::FixedBitSet;

use super::fields_writer::{DOC, END, END_OFFSET, FIELD, FREQ, PAYLOAD, POS, START_OFFSET, TERM};
use super::skip::{Impacts, SimpleTextSkipReader, BLOCK_SIZE, SKIP_LIST};
use super::util::{self, parse_int, tail};
use super::{
    corrupt, feature_requested, file_pointer, subsumes_freqs, Error, Result, SeekStatus,
    NO_MORE_DOCS, POSITIONS,
};
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::fst::{InputType, Pair, PairOutputs, PositiveIntOutputs};
use crate::fst_compiler::FstCompilerBuilder;
use crate::fst_util::{BytesRefFstEnum, TypedFst};
use crate::segment_state::SegmentReadState;

/// `PairOutputs<Pair<Long, Long>, Pair<Long, Long>>`: `((docsStart,
/// skipPointer), (docFreq, totalTermFreq))` per term.
pub type TermOutputs = PairOutputs<
    PairOutputs<PositiveIntOutputs, PositiveIntOutputs>,
    PairOutputs<PositiveIntOutputs, PositiveIntOutputs>,
>;
type TermOutput = Pair<Pair<i64, i64>, Pair<i64, i64>>;

/// `SimpleTextFieldsReader`.
pub struct SimpleTextFieldsReader {
    data: Arc<[u8]>,
    /// `fields`: each field's start, just past its `field` line.
    fields: BTreeMap<String, usize>,
    /// The keys of `fields` in Java's `TreeMap<String, ...>` order: by UTF-16
    /// code unit, not by the UTF-8 bytes `BTreeMap` sorts on.
    order: Vec<String>,
    field_infos: FieldInfos,
    max_doc: i32,
    /// `termsCache`.
    terms_cache: BTreeMap<String, OnceLock<SimpleTextTerms>>,
}

impl fmt::Debug for SimpleTextFieldsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleTextFieldsReader")
            .field("fields", &self.fields)
            .finish()
    }
}

impl fmt::Display for SimpleTextFieldsReader {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SimpleTextFieldsReader(fields={})", self.fields.len())
    }
}

impl SimpleTextFieldsReader {
    /// `new SimpleTextFieldsReader(state)` over the `.pst`'s bytes: every
    /// `field` line up to `END`, then the checksum footer.
    ///
    /// # Errors
    /// `Eof` for a file without `END`; `Corrupted` for a bad footer.
    pub fn open(pst: Vec<u8>, state: &SegmentReadState<'_>) -> Result<Self> {
        let data: Arc<[u8]> = pst.into();
        let fields = read_fields(&data)?;
        let terms_cache = fields
            .keys()
            .map(|k| (k.clone(), OnceLock::new()))
            .collect();
        let mut order: Vec<String> = fields.keys().cloned().collect();
        order.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        Ok(SimpleTextFieldsReader {
            data,
            fields,
            order,
            field_infos: state.field_infos.clone(),
            max_doc: state.max_doc,
            terms_cache,
        })
    }

    /// `iterator()`: the field names, in Java's (UTF-16) order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }

    /// `terms(field)`: the field's terms, loaded on first use; `None` for a
    /// field the file does not hold.
    ///
    /// # Errors
    /// `Corrupted` for a field `FieldInfos` lacks (Java: a
    /// `NullPointerException` later) and for what loading the terms finds.
    pub fn terms(&self, field: &str) -> Result<Option<&SimpleTextTerms>> {
        let (Some(&start), Some(cell)) = (self.fields.get(field), self.terms_cache.get(field))
        else {
            return Ok(None);
        };
        if let Some(terms) = cell.get() {
            return Ok(Some(terms));
        }
        let info = self
            .field_infos
            .field_by_name(field)
            .ok_or_else(|| corrupt(format!("field {field} is not in FieldInfos")))?;
        let terms =
            SimpleTextTerms::new(Arc::clone(&self.data), info.clone(), start, self.max_doc)?;
        // A racing load built the same terms; either copy is the answer.
        let _ = cell.set(terms);
        Ok(cell.get())
    }

    /// `checkIntegrity()`: nothing (Java's is empty too; the footer is
    /// checked at open).
    ///
    /// # Errors
    /// Never.
    pub fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
}

/// `readFields(in)`.
fn read_fields(data: &[u8]) -> Result<BTreeMap<String, usize>> {
    let mut scratch = Vec::new();
    let mut fields = BTreeMap::new();
    let mut pos = 0usize;
    loop {
        util::read_line(data, &mut pos, &mut scratch)?;
        if scratch == END {
            util::check_footer(data, 0, &mut pos)?;
            return Ok(fields);
        } else if let Some(name) = tail(&scratch, FIELD) {
            fields.insert(String::from_utf8_lossy(name).into_owned(), pos);
        }
    }
}

/// `SimpleTextTerms`: one field's statistics and its terms' FST.
pub struct SimpleTextTerms {
    data: Arc<[u8]>,
    info: FieldInfo,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    fst: Option<TypedFst<TermOutputs>>,
    term_count: i32,
}

impl fmt::Debug for SimpleTextTerms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for SimpleTextTerms {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimpleTextTerms(terms={},postings={},positions={},docs={})",
            self.term_count, self.sum_doc_freq, self.sum_total_term_freq, self.doc_count
        )
    }
}

fn overflow(what: &str) -> Error {
    corrupt(format!("{what} overflows"))
}

fn output(docs_start: i64, skip_pointer: i64, doc_freq: i32, total_term_freq: i64) -> TermOutput {
    Pair {
        first: Pair {
            first: docs_start,
            second: skip_pointer,
        },
        second: Pair {
            first: i64::from(doc_freq),
            second: total_term_freq,
        },
    }
}

impl SimpleTextTerms {
    /// `new SimpleTextTerms(field, termsStart, maxDoc)`: `loadTerms()`.
    fn new(data: Arc<[u8]>, info: FieldInfo, terms_start: usize, max_doc: i32) -> Result<Self> {
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<TermOutputs>();
        let mut pos = terms_start;
        let mut scratch = Vec::new();
        let mut last_term: Vec<u8> = Vec::new();
        let mut last_docs_start: i64 = -1;
        let mut doc_freq: i32 = 0;
        let mut total_term_freq: i64 = 0;
        let mut skip_pointer: i64 = 0;
        let mut sum_total_term_freq: i64 = 0;
        let mut sum_doc_freq: i64 = 0;
        let mut term_count: i32 = 0;
        let mut visited_docs = FixedBitSet::new(usize::try_from(max_doc).unwrap_or(0));
        let add = |compiler: &mut crate::fst_compiler::FstCompiler<TermOutputs>,
                   term: &[u8],
                   docs_start: i64,
                   skip_pointer: i64,
                   doc_freq: i32,
                   total_term_freq: i64|
         -> Result<()> {
            // `PositiveIntOutputs` asserts its outputs are not negative.
            if total_term_freq < 0 {
                return Err(corrupt(format!("negative totalTermFreq {total_term_freq}")));
            }
            compiler.add_bytes(
                term,
                output(docs_start, skip_pointer, doc_freq, total_term_freq),
            )?;
            Ok(())
        };
        loop {
            util::read_line(&data, &mut pos, &mut scratch)?;
            let line = scratch.as_slice();
            if line == END || tail(line, FIELD).is_some() {
                if last_docs_start != -1 {
                    add(
                        &mut compiler,
                        &last_term,
                        last_docs_start,
                        skip_pointer,
                        doc_freq,
                        total_term_freq,
                    )?;
                    sum_total_term_freq = sum_total_term_freq
                        .checked_add(total_term_freq)
                        .ok_or_else(|| overflow("sumTotalTermFreq"))?;
                }
                break;
            } else if let Some(v) = tail(line, DOC) {
                doc_freq = doc_freq.checked_add(1).ok_or_else(|| overflow("docFreq"))?;
                sum_doc_freq = sum_doc_freq
                    .checked_add(1)
                    .ok_or_else(|| overflow("sumDocFreq"))?;
                total_term_freq = total_term_freq
                    .checked_add(1)
                    .ok_or_else(|| overflow("totalTermFreq"))?;
                let doc_id = parse_int(v)?;
                let doc = usize::try_from(doc_id)
                    .ok()
                    .filter(|&d| d < visited_docs.len())
                    .ok_or_else(|| {
                        corrupt(format!("doc {doc_id} out of bounds (maxDoc {max_doc})"))
                    })?;
                visited_docs.set(doc);
            } else if let Some(v) = tail(line, FREQ) {
                // Java's `parseInt(...) - 1` is `int`; widened first here.
                let extra = i64::from(parse_int(v)?)
                    .checked_sub(1)
                    .ok_or_else(|| overflow("totalTermFreq"))?;
                total_term_freq = total_term_freq
                    .checked_add(extra)
                    .ok_or_else(|| overflow("totalTermFreq"))?;
            } else if tail(line, SKIP_LIST).is_some() {
                skip_pointer = i64::try_from(pos).map_err(|_| overflow("skipPointer"))?;
            } else if let Some(v) = tail(line, TERM) {
                if last_docs_start != -1 {
                    add(
                        &mut compiler,
                        &last_term,
                        last_docs_start,
                        skip_pointer,
                        doc_freq,
                        total_term_freq,
                    )?;
                }
                last_docs_start = i64::try_from(pos).map_err(|_| overflow("docsStart"))?;
                last_term.clear();
                last_term.extend_from_slice(v);
                doc_freq = 0;
                sum_total_term_freq = sum_total_term_freq
                    .checked_add(total_term_freq)
                    .ok_or_else(|| overflow("sumTotalTermFreq"))?;
                total_term_freq = 0;
                term_count = term_count
                    .checked_add(1)
                    .ok_or_else(|| overflow("termCount"))?;
                skip_pointer = 0;
            }
        }
        let doc_count =
            i32::try_from(visited_docs.cardinality()).map_err(|_| overflow("docCount"))?;
        let fst = compiler.compile().map(TypedFst::from_compiled);
        Ok(SimpleTextTerms {
            data,
            info,
            sum_total_term_freq,
            sum_doc_freq,
            doc_count,
            fst,
            term_count,
        })
    }

    /// The field.
    pub fn info(&self) -> &FieldInfo {
        &self.info
    }

    /// `iterator()`: `TermsEnum.EMPTY` for a field with no terms.
    pub fn iterator(&self) -> SimpleTextTermsEnum<'_> {
        SimpleTextTermsEnum {
            data: &self.data,
            index_options: self.info.index_options,
            fst_enum: self.fst.as_ref().map(BytesRefFstEnum::new),
            doc_freq: 0,
            total_term_freq: 0,
            docs_start: 0,
            skip_pointer: 0,
        }
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        i64::from(self.term_count)
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
        subsumes_freqs(self.info.index_options)
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
}

/// `(docsStart, skipPointer, docFreq, totalTermFreq)` out of a term's
/// output; `docFreq` is Java's `intValue()`.
fn decode(o: &TermOutput) -> (i64, i64, i32, i64) {
    (
        o.first.first,
        o.first.second,
        o.second.first as i32,
        o.second.second,
    )
}

/// `SimpleTextTermsEnum`.
pub struct SimpleTextTermsEnum<'r> {
    data: &'r [u8],
    index_options: IndexOptions,
    /// `None`: `TermsEnum.EMPTY`.
    fst_enum: Option<BytesRefFstEnum<'r, TermOutputs>>,
    doc_freq: i32,
    total_term_freq: i64,
    docs_start: i64,
    skip_pointer: i64,
}

impl fmt::Debug for SimpleTextTermsEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleTextTermsEnum")
            .field("term", &self.term())
            .field("doc_freq", &self.doc_freq)
            .finish()
    }
}

impl<'r> SimpleTextTermsEnum<'r> {
    fn apply(&mut self, (docs_start, skip_pointer, doc_freq, ttf): (i64, i64, i32, i64)) {
        self.docs_start = docs_start;
        self.skip_pointer = skip_pointer;
        self.doc_freq = doc_freq;
        self.total_term_freq = ttf;
    }

    /// `seekExact(text)`.
    ///
    /// # Errors
    /// What walking the FST fails with.
    pub fn seek_exact(&mut self, text: &[u8]) -> Result<bool> {
        let Some(e) = self.fst_enum.as_mut() else {
            return Ok(false);
        };
        match e.seek_exact(text)?.map(|(_, o)| decode(o)) {
            Some(state) => {
                self.apply(state);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// `seekCeil(text)`.
    ///
    /// # Errors
    /// What walking the FST fails with.
    pub fn seek_ceil(&mut self, text: &[u8]) -> Result<SeekStatus> {
        let Some(e) = self.fst_enum.as_mut() else {
            return Ok(SeekStatus::End);
        };
        match e.seek_ceil(text)?.map(|(t, o)| (t == text, decode(o))) {
            None => Ok(SeekStatus::End),
            Some((found, state)) => {
                self.apply(state);
                Ok(if found {
                    SeekStatus::Found
                } else {
                    SeekStatus::NotFound
                })
            }
        }
    }

    /// `next()`.
    ///
    /// # Errors
    /// What walking the FST fails with.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        let Some(e) = self.fst_enum.as_mut() else {
            return Ok(None);
        };
        match e.next()?.map(|(_, o)| decode(o)) {
            Some(state) => {
                self.apply(state);
                Ok(Some(self.term()))
            }
            None => Ok(None),
        }
    }

    /// `term()`: `fstEnum.current().input`.
    pub fn term(&self) -> &[u8] {
        self.fst_enum
            .as_ref()
            .and_then(|e| e.current())
            .map_or(&[], |(t, _)| t)
    }

    /// `ord()`.
    ///
    /// # Errors
    /// Always: Java's `UnsupportedOperationException`.
    pub fn ord(&self) -> Result<i64> {
        Err(Error::Unsupported("ord"))
    }

    /// `seekExact(long ord)`.
    ///
    /// # Errors
    /// Always: Java's `UnsupportedOperationException`.
    pub fn seek_exact_ord(&mut self, _ord: i64) -> Result<()> {
        Err(Error::Unsupported("seekExact(ord)"))
    }

    /// `docFreq()`.
    pub fn doc_freq(&self) -> i32 {
        self.doc_freq
    }

    /// `totalTermFreq()`: `docFreq` for a DOCS field.
    pub fn total_term_freq(&self) -> i64 {
        if self.index_options == IndexOptions::Docs {
            i64::from(self.doc_freq)
        } else {
            self.total_term_freq
        }
    }

    /// `postings(null, flags)`: a positions enum when the field has positions
    /// and `flags` asks for them, a docs enum otherwise.
    ///
    /// # Errors
    /// [`Error::IllegalState`] on a field with no terms (Java's
    /// `TermsEnum.EMPTY`); `Corrupted` for a term start or skip list outside
    /// the file.
    pub fn postings(&self, flags: i32) -> Result<SimpleTextPostings<'r>> {
        if self.fst_enum.is_none() {
            return Err(Error::IllegalState("this method should never be called"));
        }
        let has_positions = self.index_options.subsumes_positions();
        if has_positions && feature_requested(flags, POSITIONS) {
            return Ok(SimpleTextPostings::Positions(SimpleTextPostingsEnum::new(
                self.data,
                self.docs_start,
                self.index_options,
                self.doc_freq,
                self.skip_pointer,
            )?));
        }
        Ok(SimpleTextPostings::Docs(SimpleTextDocsEnum::new(
            self.data,
            self.docs_start,
            self.index_options == IndexOptions::Docs,
            self.doc_freq,
            self.skip_pointer,
        )?))
    }

    /// `impacts(flags)`: a `SlowImpactsEnum` for a term without skip data.
    ///
    /// # Errors
    /// What [`Self::postings`] fails with.
    pub fn impacts(&self, flags: i32) -> Result<SimpleTextImpactsEnum<'r>> {
        let postings = self.postings(flags)?;
        Ok(if self.doc_freq <= BLOCK_SIZE {
            // no skip data
            SimpleTextImpactsEnum::Slow(postings)
        } else {
            SimpleTextImpactsEnum::Skip(postings)
        })
    }
}

/// `SimpleTextDocsEnum`: docs (and freqs), read line by line.
#[derive(Debug, Clone)]
pub struct SimpleTextDocsEnum<'a> {
    data: &'a [u8],
    /// `in`'s file pointer.
    pos: usize,
    omit_tf: bool,
    doc_id: i32,
    tf: i32,
    scratch: Vec<u8>,
    cost: i32,
    skip_reader: SimpleTextSkipReader<'a>,
    next_skip_doc: i32,
    seek_to: i64,
}

impl<'a> SimpleTextDocsEnum<'a> {
    /// `new SimpleTextDocsEnum().reset(fp, omitTF, docFreq, skipPointer)`.
    fn new(
        data: &'a [u8],
        fp: i64,
        omit_tf: bool,
        doc_freq: i32,
        skip_pointer: i64,
    ) -> Result<Self> {
        let mut skip_reader = SimpleTextSkipReader::new(data);
        skip_reader.reset(skip_pointer, doc_freq)?;
        Ok(SimpleTextDocsEnum {
            data,
            pos: file_pointer(data, fp, "docs start")?,
            omit_tf,
            doc_id: -1,
            tf: 1,
            scratch: Vec::new(),
            cost: doc_freq,
            skip_reader,
            next_skip_doc: 0,
            seek_to: -1,
        })
    }

    /// `readDoc()`.
    fn read_doc(&mut self) -> Result<i32> {
        if self.doc_id == NO_MORE_DOCS {
            return Ok(self.doc_id);
        }
        let mut first = true;
        let mut term_freq = 0;
        loop {
            let line_start = self.pos;
            util::read_line(self.data, &mut self.pos, &mut self.scratch)?;
            let line = self.scratch.as_slice();
            if let Some(v) = tail(line, DOC) {
                if !first {
                    self.pos = line_start;
                    if !self.omit_tf {
                        self.tf = term_freq;
                    }
                    return Ok(self.doc_id);
                }
                self.doc_id = parse_int(v)?;
                term_freq = 0;
                first = false;
            } else if let Some(v) = tail(line, FREQ) {
                term_freq = parse_int(v)?;
            } else if is_occurrence_line(line) {
                // skip
            } else {
                if !first {
                    self.pos = line_start;
                    if !self.omit_tf {
                        self.tf = term_freq;
                    }
                    return Ok(self.doc_id);
                }
                self.doc_id = NO_MORE_DOCS;
                return Ok(self.doc_id);
            }
        }
    }

    /// `advanceTarget(target)`.
    fn advance_target(&mut self, target: i32) -> Result<i32> {
        if self.seek_to > 0 {
            self.pos = file_pointer(self.data, self.seek_to, "skipDocFP")?;
            self.seek_to = -1;
        }
        loop {
            let doc = self.read_doc()?;
            if doc >= target {
                return Ok(doc);
            }
        }
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// What reading a line, a number or the skip list fails with.
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        self.advance_shallow(target)?;
        self.advance_target(target)
    }

    /// `advanceShallow(target)`.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn advance_shallow(&mut self, target: i32) -> Result<()> {
        if target > self.next_skip_doc {
            self.skip_reader.skip_to(target)?;
            if self.skip_reader.next_skip_doc() != NO_MORE_DOCS {
                self.seek_to = self.skip_reader.next_skip_doc_fp();
            }
            self.next_skip_doc = self.skip_reader.next_skip_doc();
        }
        Ok(())
    }
}

/// `SimpleTextPostingsEnum`: docs, freqs and each document's positions,
/// offsets and payloads.
#[derive(Debug, Clone)]
pub struct SimpleTextPostingsEnum<'a> {
    data: &'a [u8],
    /// `in`'s file pointer: the next position line.
    pos: usize,
    doc_id: i32,
    tf: i32,
    scratch: Vec<u8>,
    /// `scratch2`: the current payload.
    payload: Vec<u8>,
    has_payload: bool,
    position: i32,
    next_doc_start: usize,
    read_offsets: bool,
    read_positions: bool,
    start_offset: i32,
    end_offset: i32,
    cost: i32,
    skip_reader: SimpleTextSkipReader<'a>,
    next_skip_doc: i32,
    seek_to: i64,
}

/// A `pos`, `startOffset`, `endOffset` or `payload` line, which a docs scan
/// steps over.
fn is_occurrence_line(line: &[u8]) -> bool {
    tail(line, POS).is_some()
        || tail(line, START_OFFSET).is_some()
        || tail(line, END_OFFSET).is_some()
        || tail(line, PAYLOAD).is_some()
}

impl<'a> SimpleTextPostingsEnum<'a> {
    /// `new SimpleTextPostingsEnum().reset(fp, indexOptions, docFreq,
    /// skipPointer)`.
    fn new(
        data: &'a [u8],
        fp: i64,
        index_options: IndexOptions,
        doc_freq: i32,
        skip_pointer: i64,
    ) -> Result<Self> {
        let mut skip_reader = SimpleTextSkipReader::new(data);
        skip_reader.reset(skip_pointer, doc_freq)?;
        Ok(SimpleTextPostingsEnum {
            data,
            pos: 0,
            doc_id: -1,
            tf: 0,
            scratch: Vec::new(),
            payload: Vec::new(),
            has_payload: false,
            position: 0,
            next_doc_start: file_pointer(data, fp, "docs start")?,
            read_offsets: index_options.subsumes_offsets(),
            read_positions: index_options.subsumes_positions(),
            start_offset: -1,
            end_offset: -1,
            cost: doc_freq,
            skip_reader,
            next_skip_doc: 0,
            seek_to: -1,
        })
    }

    /// `readDoc()`.
    fn read_doc(&mut self) -> Result<i32> {
        let mut first = true;
        let mut pos = self.next_doc_start;
        let mut pos_start = 0usize;
        loop {
            let line_start = pos;
            util::read_line(self.data, &mut pos, &mut self.scratch)?;
            let line = self.scratch.as_slice();
            if let Some(v) = tail(line, DOC) {
                if !first {
                    self.next_doc_start = line_start;
                    self.pos = pos_start;
                    return Ok(self.doc_id);
                }
                self.doc_id = parse_int(v)?;
                self.tf = 0;
                first = false;
            } else if let Some(v) = tail(line, FREQ) {
                self.tf = parse_int(v)?;
                pos_start = pos;
            } else if is_occurrence_line(line) {
                // skip
            } else {
                if !first {
                    self.next_doc_start = line_start;
                    self.pos = pos_start;
                    return Ok(self.doc_id);
                }
                self.pos = pos;
                self.doc_id = NO_MORE_DOCS;
                return Ok(self.doc_id);
            }
        }
    }

    /// `advanceTarget(target)`.
    fn advance_target(&mut self, target: i32) -> Result<i32> {
        if self.seek_to > 0 {
            self.next_doc_start = file_pointer(self.data, self.seek_to, "skipDocFP")?;
            self.seek_to = -1;
        }
        loop {
            let doc = self.read_doc()?;
            if doc >= target {
                return Ok(doc);
            }
        }
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// What reading a line, a number or the skip list fails with.
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        self.advance_shallow(target)?;
        self.advance_target(target)
    }

    /// `advanceShallow(target)`.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn advance_shallow(&mut self, target: i32) -> Result<()> {
        if target > self.next_skip_doc {
            self.skip_reader.skip_to(target)?;
            if self.skip_reader.next_skip_doc() != NO_MORE_DOCS {
                self.seek_to = self.skip_reader.next_skip_doc_fp();
            }
        }
        self.next_skip_doc = self.skip_reader.next_skip_doc();
        Ok(())
    }

    /// The line at `in`, past `prefix` (Java asserts the prefix).
    fn read_value(&mut self, prefix: &[u8]) -> Result<i32> {
        util::read_line(self.data, &mut self.pos, &mut self.scratch)?;
        let v = tail(&self.scratch, prefix).ok_or_else(|| {
            corrupt(format!(
                "expected {} but got line={}",
                String::from_utf8_lossy(prefix).trim(),
                String::from_utf8_lossy(&self.scratch)
            ))
        })?;
        parse_int(v)
    }

    /// `nextPosition()`: the next position, its offsets and its payload.
    ///
    /// # Errors
    /// `Corrupted` for a line that is not the position, offset Java expects
    /// (Java asserts) or a number that does not parse; `Eof`.
    pub fn next_position(&mut self) -> Result<i32> {
        self.position = if self.read_positions {
            self.read_value(POS)?
        } else {
            NO_POSITION
        };
        if self.read_offsets {
            self.start_offset = self.read_value(START_OFFSET)?;
            self.end_offset = self.read_value(END_OFFSET)?;
        }
        let fp = self.pos;
        util::read_line(self.data, &mut self.pos, &mut self.scratch)?;
        if let Some(v) = tail(&self.scratch, PAYLOAD) {
            self.payload.clear();
            self.payload.extend_from_slice(v);
            self.has_payload = true;
        } else {
            self.has_payload = false;
            self.pos = fp;
        }
        Ok(self.position)
    }
}

/// `-1`: `nextPosition()`/`startOffset()`/`endOffset()` where the enum has
/// none.
const NO_POSITION: i32 = -1;

/// What `TermsEnum.postings` hands out.
#[derive(Debug, Clone)]
pub enum SimpleTextPostings<'a> {
    /// `SimpleTextDocsEnum`.
    Docs(SimpleTextDocsEnum<'a>),
    /// `SimpleTextPostingsEnum`.
    Positions(SimpleTextPostingsEnum<'a>),
}

impl SimpleTextPostings<'_> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        match self {
            SimpleTextPostings::Docs(e) => e.doc_id,
            SimpleTextPostings::Positions(e) => e.doc_id,
        }
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        match self {
            SimpleTextPostings::Docs(e) => e.tf,
            SimpleTextPostings::Positions(e) => e.tf,
        }
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        match self {
            SimpleTextPostings::Docs(e) => i64::from(e.cost),
            SimpleTextPostings::Positions(e) => i64::from(e.cost),
        }
    }

    /// `nextDoc()`: `advance(docID + 1)` (Java's `int` wrap past
    /// `NO_MORE_DOCS`).
    ///
    /// # Errors
    /// What [`Self::advance`] fails with.
    pub fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc_id().wrapping_add(1);
        self.advance(target)
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// What reading a line, a number or the skip list fails with.
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        match self {
            SimpleTextPostings::Docs(e) => e.advance(target),
            SimpleTextPostings::Positions(e) => e.advance(target),
        }
    }

    /// `nextPosition()`: [`NO_POSITION`] on a docs enum.
    ///
    /// # Errors
    /// What [`SimpleTextPostingsEnum::next_position`] fails with.
    // SENTINEL: `-1` = "this enum reads no positions", Java's value.
    pub fn next_position(&mut self) -> Result<i32> {
        match self {
            SimpleTextPostings::Docs(_) => Ok(NO_POSITION),
            SimpleTextPostings::Positions(e) => e.next_position(),
        }
    }

    /// `startOffset()`: [`NO_POSITION`] without offsets.
    pub fn start_offset(&self) -> i32 {
        match self {
            SimpleTextPostings::Docs(_) => NO_POSITION,
            SimpleTextPostings::Positions(e) => e.start_offset,
        }
    }

    /// `endOffset()`: [`NO_POSITION`] without offsets.
    pub fn end_offset(&self) -> i32 {
        match self {
            SimpleTextPostings::Docs(_) => NO_POSITION,
            SimpleTextPostings::Positions(e) => e.end_offset,
        }
    }

    /// `getPayload()`: the current position's payload, `None` without one.
    pub fn payload(&self) -> Option<&[u8]> {
        match self {
            SimpleTextPostings::Positions(e) if e.has_payload => Some(&e.payload),
            _ => None,
        }
    }

    /// `advanceShallow(target)`.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn advance_shallow(&mut self, target: i32) -> Result<()> {
        match self {
            SimpleTextPostings::Docs(e) => e.advance_shallow(target),
            SimpleTextPostings::Positions(e) => e.advance_shallow(target),
        }
    }

    /// `getImpacts()`: `advanceShallow(docID)`, then the skip reader's.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn impacts(&mut self) -> Result<Impacts<'_>> {
        let doc = self.doc_id();
        self.advance_shallow(doc)?;
        Ok(match self {
            SimpleTextPostings::Docs(e) => e.skip_reader.impacts(),
            SimpleTextPostings::Positions(e) => e.skip_reader.impacts(),
        })
    }
}

/// What `TermsEnum.impacts` hands out.
#[derive(Debug, Clone)]
pub enum SimpleTextImpactsEnum<'a> {
    /// `new SlowImpactsEnum(postings(null, flags))`: `docFreq <= BLOCK_SIZE`.
    Slow(SimpleTextPostings<'a>),
    /// The postings enum itself, over its skip list.
    Skip(SimpleTextPostings<'a>),
}

impl<'a> SimpleTextImpactsEnum<'a> {
    /// The postings underneath.
    pub fn postings(&mut self) -> &mut SimpleTextPostings<'a> {
        match self {
            SimpleTextImpactsEnum::Slow(p) | SimpleTextImpactsEnum::Skip(p) => p,
        }
    }

    /// `advanceShallow(target)`: nothing on a `SlowImpactsEnum`.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn advance_shallow(&mut self, target: i32) -> Result<()> {
        match self {
            SimpleTextImpactsEnum::Slow(_) => Ok(()),
            SimpleTextImpactsEnum::Skip(p) => p.advance_shallow(target),
        }
    }

    /// `getImpacts()`.
    ///
    /// # Errors
    /// What reading the skip list fails with.
    pub fn impacts(&mut self) -> Result<Impacts<'_>> {
        match self {
            SimpleTextImpactsEnum::Slow(_) => Ok(Impacts::Slow),
            SimpleTextImpactsEnum::Skip(p) => p.impacts(),
        }
    }
}
