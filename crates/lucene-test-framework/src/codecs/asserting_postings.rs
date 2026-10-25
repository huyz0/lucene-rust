//! Java's asserting postings consumer validates by re-enumerating only after
//! the delegate's successful write. It preserves Java's UTF-16 field ordering,
//! unsigned term ordering, requested posting flags and reuse argument, and checks
//! increasing document IDs, positive frequencies, positions and offsets. There
//! is deliberately no maxDoc check. Reader terms use the existing
//! `AssertingLeafReader.AssertingTerms`; both sides double-close their delegate,
//! stopping if the first close fails. Merge instances stay wrapped.
//!
//! The write-side enum seam is local: it represents lazy errors, exact integer
//! flags and reuse that the public search reader's owned enums cannot express.
//! Typed references exclude Java's null iterator/enum/delegate failures. Names
//! and next-term values are owned copies. Read-side term statistics and posting
//! frequency/offset access inherit the public reader's infallible accessors and
//! its unsupported ordinal/state/reuse and per-call thread-check limitations.
//! The explicit Lucene104 adapter reuses `cranky_postings`' seven-file codec,
//! eager input, buffered output, field-level norms snapshot and publication/error
//! timing limits. Its merge bridge defers mapped materialization until fields
//! are requested, then rebuilds owned views on repeated enumeration. Native
//! merge readers alias one close state, while acquired terms retain snapshots.
//! Each native handle drops its decoded snapshot on close; closing an alias
//! invalidates its siblings, whose snapshots live until their own close/drop.
//! A merge scans every source and validates every map on each names/terms
//! request, but allocates postings only for the requested field. Source faults
//! and invalid maps in unrelated fields keep their original callback order.
//! Native Display is a stable label. IOContext, Directory failures and
//! Codec/IndexWriter routing are absent. No random decisions are added.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings::Position;
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_index::indexing_chain::MAX_POSITION;
use lucene_search::reader::{
    self as read, MaterializedPostings, PostingsFlags, Terms as ReadTerms,
    TermsEnum as ReadTermsEnum, NO_MORE_DOCS,
};

use super::cranky_postings::{self as byte, Fields as ByteFields, FieldsConsumer as ByteConsumer};
use crate::index::asserting_leaf_reader::AssertingTerms;
pub use byte::{output_files, Error, Files, NormsProducer, OwnedNorms, ReadState, Result};

pub const FREQS: u32 = 8;
pub const POSITIONS: u32 = 24;
pub const OFFSETS: u32 = 56;
pub const PAYLOADS: u32 = 88;

fn failure(message: impl Into<String>) -> Error {
    Error::Message(message.into())
}

/// The write API's Fields, kept separate from the search reader's enum API.
pub trait WriteFields {
    fn field_names(&self) -> Result<Vec<String>>;
    fn terms(&self, name: &str) -> Result<Option<Box<dyn WriteTerms + '_>>>;
}
pub trait WriteTerms {
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum + '_>>;
    fn has_positions(&self) -> bool;
    fn has_offsets(&self) -> bool;
    fn has_payloads(&self) -> bool;
}
pub trait WriteTermsEnum {
    fn next(&mut self) -> Result<Option<Vec<u8>>>;
    fn postings(
        &mut self,
        reuse: Option<Box<dyn WritePostings>>,
        flags: u32,
    ) -> Result<Box<dyn WritePostings>>;
}
pub trait WritePostings {
    fn next_doc(&mut self) -> Result<i32>;
    fn freq(&mut self) -> Result<i32>;
    fn next_position(&mut self) -> Result<i32>;
    fn start_offset(&mut self) -> Result<i32>;
    fn end_offset(&mut self) -> Result<i32>;
    fn payload(&mut self) -> Result<Option<Vec<u8>>>;
}
pub struct MergeState<'a> {
    pub bytes: byte::MergeState<'a>,
    pub field_infos: &'a FieldInfos,
}
pub trait FieldsConsumer {
    fn write(&mut self, fields: &dyn WriteFields, norms: Option<&dyn NormsProducer>) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    fn merge(&mut self, state: &MergeState<'_>, norms: Option<&dyn NormsProducer>) -> Result<()> {
        for source in state.bytes.sources {
            if let Some(reader) = source.reader {
                (state.bytes.check_aborted)()?;
                byte::FieldsProducer::check_integrity(reader)?;
            }
        }
        self.write(&MappedWriteFields(state), norms)
    }
}

/// The measured merge hotspot rebuilt all fields for every single-field lookup.
/// Keep the full scan/error ordering, but only allocate the requested result.
struct MappedWriteFields<'a, 'b>(&'a MergeState<'b>);
impl MappedWriteFields<'_, '_> {
    fn scan(&self, selected: Option<&str>) -> Result<(Vec<String>, Option<byte::FieldPostings>)> {
        let mut names = std::collections::BTreeSet::new();
        let mut terms: std::collections::BTreeMap<Vec<u8>, Vec<byte::Posting>> =
            std::collections::BTreeMap::new();
        let mut present = false;
        for source in self.0.bytes.sources {
            let Some(reader) = source.reader else {
                continue;
            };
            for field in reader.materialize()?.iter() {
                let collect = selected.is_some_and(|name| name == field.name);
                if selected.is_none() {
                    names.insert(field.name.clone());
                }
                present |= collect;
                for term in &field.terms {
                    let mut selected_postings = if collect {
                        Some(terms.entry(term.bytes.clone()).or_default())
                    } else {
                        None
                    };
                    for posting in &term.postings {
                        if posting.doc < 0 || posting.doc >= source.max_doc {
                            return Err(failure("source document out of bounds"));
                        }
                        let index = usize::try_from(posting.doc)
                            .map_err(|_| failure("negative source document"))?;
                        let doc = *source
                            .doc_map
                            .get(index)
                            .ok_or_else(|| failure("document map out of bounds"))?;
                        // SENTINEL: Java MergeState.DocMap returns -1 for deleted documents.
                        if doc == -1 {
                            continue;
                        }
                        if doc < 0 {
                            return Err(failure("invalid mapped document"));
                        }
                        if let Some(postings) = selected_postings.as_mut() {
                            let mut posting = posting.clone();
                            posting.doc = doc;
                            postings.push(posting);
                        }
                    }
                }
            }
        }
        let mut names: Vec<_> = names.into_iter().collect();
        names.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        let field = present.then(|| byte::FieldPostings {
            name: selected.expect("selected field is present").into(),
            terms: terms
                .into_iter()
                .filter_map(|(bytes, mut postings)| {
                    if postings.is_empty() {
                        return None;
                    }
                    postings.sort_by_key(|posting| posting.doc);
                    Some(byte::Term { bytes, postings })
                })
                .collect(),
        });
        Ok((names, field))
    }
}
impl WriteFields for MappedWriteFields<'_, '_> {
    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self.scan(None)?.0)
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn WriteTerms + '_>>> {
        let Some(field) = self.scan(Some(name))?.1 else {
            return Ok(None);
        };
        let info = self
            .0
            .field_infos
            .field_by_name(name)
            .ok_or_else(|| failure("unknown postings field"))?;
        Ok(Some(Box::new(ModelWriteTerms {
            field: Cow::Owned(field),
            info: Cow::Borrowed(info),
        })))
    }
}
pub trait FieldsProducer: fmt::Display {
    fn field_names(&self) -> Result<Vec<String>>;
    fn terms(&self, field: &str) -> Result<Option<Box<dyn ReadTerms + '_>>>;
    fn size(&self) -> i32;
    fn check_integrity(&self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    fn get_merge_instance(&self) -> Self
    where
        Self: Sized;
}
pub trait PostingsFormat {
    type Consumer<'a>: FieldsConsumer;
    type Producer: FieldsProducer;
    fn name(&self) -> &str;
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>>;
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer>;
}
pub struct AssertingPostingsFormat<F> {
    delegate: F,
}
impl<F> AssertingPostingsFormat<F> {
    pub fn new(delegate: F) -> Self {
        Self { delegate }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: PostingsFormat> PostingsFormat for AssertingPostingsFormat<F> {
    type Consumer<'a> = AssertingFieldsConsumer<'a, F::Consumer<'a>>;
    type Producer = AssertingFieldsProducer<F::Producer>;
    fn name(&self) -> &str {
        "Asserting"
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        Ok(AssertingFieldsConsumer::new(
            self.delegate.fields_consumer(state)?,
            state.field_infos,
        ))
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        Ok(AssertingFieldsProducer::new(
            self.delegate.fields_producer(state)?,
        ))
    }
}
pub struct AssertingFieldsConsumer<'a, W> {
    delegate: W,
    fields: &'a FieldInfos,
}
impl<'a, W> AssertingFieldsConsumer<'a, W> {
    pub fn new(delegate: W, fields: &'a FieldInfos) -> Self {
        Self { delegate, fields }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: FieldsConsumer> FieldsConsumer for AssertingFieldsConsumer<'_, W> {
    /// Java: AssertingFieldsConsumer.write. Delegate success precedes validation.
    fn write(&mut self, fields: &dyn WriteFields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        self.delegate.write(fields, norms)?;
        let mut last_field: Option<String> = None;
        for name in fields.field_names()? {
            let info = self
                .fields
                .field_by_name(&name)
                .expect("unknown postings field");
            assert!(last_field
                .as_ref()
                .is_none_or(|last| last.encode_utf16().cmp(name.encode_utf16()).is_lt()));
            last_field = Some(name.clone());
            let Some(terms) = fields.terms(&name)? else {
                continue;
            };
            let mut terms_enum = terms.iterator()?;
            let mut last_term: Option<Vec<u8>> = None;
            let mut postings: Option<Box<dyn WritePostings>> = None;
            let has_freqs = has_freqs(info.index_options);
            let has_positions = info.index_options.subsumes_positions();
            let has_offsets = info.index_options.subsumes_offsets();
            let has_payloads = terms.has_payloads();
            assert_eq!(has_positions, terms.has_positions());
            assert_eq!(has_offsets, terms.has_offsets());
            while let Some(term) = terms_enum.next()? {
                assert!(last_term.as_ref().is_none_or(|last| last < &term));
                last_term = Some(term);
                let flags = flags(has_freqs, has_positions, has_offsets, has_payloads);
                postings = Some(terms_enum.postings(postings.take(), flags)?);
                let docs = postings.as_mut().expect("postings assigned above");
                let mut last_doc = -1;
                loop {
                    let doc = docs.next_doc()?;
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    assert!(doc > last_doc);
                    last_doc = doc;
                    if has_freqs {
                        let freq = docs.freq()?;
                        assert!(freq > 0);
                        if has_positions {
                            let mut last_pos = -1;
                            let mut last_start = -1;
                            for _ in 0..freq {
                                let pos = docs.next_position()?;
                                assert!(pos >= last_pos);
                                assert!(pos <= MAX_POSITION);
                                last_pos = pos;
                                if has_offsets {
                                    let start = docs.start_offset()?;
                                    let end = docs.end_offset()?;
                                    assert!(end >= start);
                                    assert!(start >= last_start);
                                    last_start = start;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}
fn has_freqs(options: IndexOptions) -> bool {
    !matches!(options, IndexOptions::None | IndexOptions::Docs)
}
fn flags(freqs: bool, positions: bool, offsets: bool, payloads: bool) -> u32 {
    if !positions {
        if freqs {
            FREQS
        } else {
            0
        }
    } else {
        POSITIONS | if offsets { OFFSETS } else { 0 } | if payloads { PAYLOADS } else { 0 }
    }
}
pub struct AssertingFieldsProducer<R> {
    delegate: R,
}
impl<R: FieldsProducer> AssertingFieldsProducer<R> {
    pub fn new(delegate: R) -> Self {
        let wrapped = Self { delegate };
        let _ = wrapped.to_string();
        wrapped
    }
    pub fn delegate(&self) -> &R {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut R {
        &mut self.delegate
    }
}
impl<R: FieldsProducer> FieldsProducer for AssertingFieldsProducer<R> {
    fn field_names(&self) -> Result<Vec<String>> {
        self.delegate.field_names()
    }
    fn terms(&self, field: &str) -> Result<Option<Box<dyn ReadTerms + '_>>> {
        Ok(self
            .delegate
            .terms(field)?
            .map(|terms| Box::new(AssertingTerms::new(terms)) as Box<dyn ReadTerms + '_>))
    }
    fn size(&self) -> i32 {
        self.delegate.size()
    }
    fn check_integrity(&self) -> Result<()> {
        self.delegate.check_integrity()
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    fn get_merge_instance(&self) -> Self {
        Self::new(self.delegate.get_merge_instance())
    }
}
impl<R: FieldsProducer> fmt::Display for AssertingFieldsProducer<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssertingFieldsProducer({})", self.delegate)
    }
}

/// Lazy bridge from the already-differential-tested native mapped/owned Fields.
pub struct NativeFieldsView<'a> {
    fields: &'a dyn ByteFields,
    infos: &'a FieldInfos,
}
impl<'a> NativeFieldsView<'a> {
    pub fn new(fields: &'a dyn ByteFields, infos: &'a FieldInfos) -> Self {
        Self { fields, infos }
    }
}
impl WriteFields for NativeFieldsView<'_> {
    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self
            .fields
            .materialize()?
            .iter()
            .map(|field| field.name.clone())
            .collect())
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn WriteTerms + '_>>> {
        let fields = self.fields.materialize()?;
        let field = match fields {
            Cow::Borrowed(fields) => fields
                .iter()
                .find(|field| field.name == name)
                .map(Cow::Borrowed),
            Cow::Owned(fields) => fields
                .into_iter()
                .find(|field| field.name == name)
                .map(Cow::Owned),
        };
        let Some(field) = field else {
            return Ok(None);
        };
        let info = self
            .infos
            .field_by_name(name)
            .ok_or_else(|| failure("unknown postings field"))?;
        Ok(Some(Box::new(ModelWriteTerms {
            field,
            info: Cow::Borrowed(info),
        })))
    }
}
struct ModelWriteTerms<'a> {
    field: Cow<'a, byte::FieldPostings>,
    info: Cow<'a, FieldInfo>,
}
impl WriteTerms for ModelWriteTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum + '_>> {
        Ok(Box::new(ModelWriteEnum {
            field: &self.field,
            next: 0,
            current: None,
        }))
    }
    fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }
    fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }
    fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }
}
struct ModelWriteEnum<'a> {
    field: &'a byte::FieldPostings,
    next: usize,
    current: Option<usize>,
}
impl WriteTermsEnum for ModelWriteEnum<'_> {
    fn next(&mut self) -> Result<Option<Vec<u8>>> {
        let Some(term) = self.field.terms.get(self.next) else {
            self.current = None;
            return Ok(None);
        };
        self.current = Some(self.next);
        self.next += 1;
        Ok(Some(term.bytes.clone()))
    }
    fn postings(
        &mut self,
        _reuse: Option<Box<dyn WritePostings>>,
        _flags: u32,
    ) -> Result<Box<dyn WritePostings>> {
        let current = self
            .current
            .ok_or_else(|| failure("unpositioned terms enum"))?;
        Ok(Box::new(ModelWritePostings {
            postings: self.field.terms[current].postings.clone(),
            next: 0,
            current: None,
            position: 0,
        }))
    }
}
struct ModelWritePostings {
    postings: Vec<byte::Posting>,
    next: usize,
    current: Option<usize>,
    position: usize,
}
impl ModelWritePostings {
    fn posting(&self) -> Result<&byte::Posting> {
        self.current
            .and_then(|i| self.postings.get(i))
            .ok_or_else(|| failure("unpositioned postings"))
    }
    fn occurrence(&self) -> Result<&Position> {
        self.posting()?
            .positions
            .get(
                self.position
                    .checked_sub(1)
                    .ok_or_else(|| failure("no position"))?,
            )
            .ok_or_else(|| failure("missing position"))
    }
}
impl WritePostings for ModelWritePostings {
    fn next_doc(&mut self) -> Result<i32> {
        self.position = 0;
        let Some(posting) = self.postings.get(self.next) else {
            self.current = None;
            return Ok(NO_MORE_DOCS);
        };
        self.current = Some(self.next);
        self.next += 1;
        Ok(posting.doc)
    }
    fn freq(&mut self) -> Result<i32> {
        Ok(self.posting()?.freq)
    }
    fn next_position(&mut self) -> Result<i32> {
        let pos = self
            .posting()?
            .positions
            .get(self.position)
            .ok_or_else(|| failure("missing position"))?
            .position;
        self.position += 1;
        Ok(pos)
    }
    fn start_offset(&mut self) -> Result<i32> {
        Ok(self.occurrence()?.start_offset)
    }
    fn end_offset(&mut self) -> Result<i32> {
        Ok(self.occurrence()?.end_offset)
    }
    fn payload(&mut self) -> Result<Option<Vec<u8>>> {
        let bytes = &self.occurrence()?.payload;
        Ok((!bytes.is_empty()).then(|| bytes.clone()))
    }
}

/// Native seven-file format; format selection remains explicit.
#[derive(Clone, Copy, Default)]
pub struct Lucene104PostingsFormat;
pub struct Lucene104FieldsConsumer<'a> {
    delegate: byte::Lucene104FieldsConsumer<'a>,
    fields: &'a FieldInfos,
}
impl Lucene104FieldsConsumer<'_> {
    pub fn output(&self) -> Option<&lucene_codecs::postings_writer::Output> {
        self.delegate.output()
    }
}
struct EncodingFields<'a> {
    source: &'a dyn WriteFields,
    infos: &'a FieldInfos,
}
impl ByteFields for EncodingFields<'_> {
    fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
        let mut result = Vec::new();
        for name in self.source.field_names()? {
            let Some(terms) = self.source.terms(&name)? else {
                continue;
            };
            let info = self
                .infos
                .field_by_name(&name)
                .ok_or_else(|| failure("unknown postings field"))?;
            let has_positions = info.index_options.subsumes_positions();
            let has_offsets = info.index_options.subsumes_offsets();
            let has_payloads = terms.has_payloads();
            let flags = flags(
                has_freqs(info.index_options),
                has_positions,
                has_offsets,
                has_payloads,
            );
            let mut e = terms.iterator()?;
            let mut encoded = Vec::new();
            let mut reuse = None;
            while let Some(bytes) = e.next()? {
                let mut docs = e.postings(reuse.take(), flags)?;
                let mut postings = Vec::new();
                loop {
                    let doc = docs.next_doc()?;
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    let freq = if has_freqs(info.index_options) {
                        docs.freq()?
                    } else {
                        1
                    };
                    let mut positions = Vec::new();
                    if has_positions {
                        for _ in 0..freq {
                            let position = docs.next_position()?;
                            let (start_offset, end_offset) = if has_offsets {
                                (docs.start_offset()?, docs.end_offset()?)
                            } else {
                                (-1, -1)
                            };
                            let payload = if has_payloads {
                                docs.payload()?.unwrap_or_default()
                            } else {
                                Vec::new()
                            };
                            positions.push(Position {
                                position,
                                start_offset,
                                end_offset,
                                payload,
                            });
                        }
                    }
                    postings.push(byte::Posting {
                        doc,
                        freq,
                        positions,
                    });
                }
                reuse = Some(docs);
                encoded.push(byte::Term { bytes, postings });
            }
            result.push(byte::FieldPostings {
                name,
                terms: encoded,
            });
        }
        Ok(Cow::Owned(result))
    }
}
impl FieldsConsumer for Lucene104FieldsConsumer<'_> {
    fn write(&mut self, fields: &dyn WriteFields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        self.delegate.write(
            &EncodingFields {
                source: fields,
                infos: self.fields,
            },
            norms,
        )
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()
    }
}
#[derive(Clone)]
pub struct Lucene104FieldsProducer {
    fields: Option<Rc<Vec<byte::FieldPostings>>>,
    shared: Rc<NativeReaderState>,
}
struct NativeReaderState {
    files: RefCell<Files>,
    infos: FieldInfos,
    closed: Cell<bool>,
}
impl fmt::Display for Lucene104FieldsProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Lucene104FieldsProducer")
    }
}
impl ByteFields for Lucene104FieldsProducer {
    fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
        if self.shared.closed.get() {
            return Err(failure("closed postings reader"));
        }
        Ok(Cow::Borrowed(
            self.fields
                .as_deref()
                .expect("open native reader")
                .as_slice(),
        ))
    }
}
impl byte::FieldsProducer for Lucene104FieldsProducer {
    fn check_integrity(&self) -> Result<()> {
        FieldsProducer::check_integrity(self)
    }
    fn close(&mut self) -> Result<()> {
        FieldsProducer::close(self)
    }
}
impl FieldsProducer for Lucene104FieldsProducer {
    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self
            .materialize()?
            .iter()
            .map(|field| field.name.clone())
            .collect())
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn ReadTerms + '_>>> {
        if self.shared.closed.get() {
            return Ok(None);
        }
        let fields = self.fields.as_deref().expect("open native reader");
        let Some(field) = fields.iter().find(|field| field.name == name) else {
            return Ok(None);
        };
        let info = self
            .shared
            .infos
            .field_by_name(name)
            .ok_or_else(|| failure("unknown postings field"))?;
        Ok(Some(Box::new(ModelReadTerms {
            field: Cow::Borrowed(field),
            info: Cow::Borrowed(info),
        })))
    }
    fn size(&self) -> i32 {
        if self.shared.closed.get() {
            0
        } else {
            i32::try_from(self.fields.as_deref().expect("open native reader").len())
                .expect("field count fits Java int")
        }
    }
    fn check_integrity(&self) -> Result<()> {
        self.materialize()?;
        for bytes in self.shared.files.borrow().values() {
            let footer = bytes
                .len()
                .checked_sub(lucene_store::codec_util::FOOTER_LENGTH)
                .ok_or_else(|| failure("missing postings footer"))?;
            lucene_store::codec_util::check_whole_file_footer(bytes, footer)?;
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.fields = None;
        self.shared.files.borrow_mut().clear();
        self.shared.closed.set(true);
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        self.clone()
    }
}
impl PostingsFormat for Lucene104PostingsFormat {
    type Consumer<'a> = Lucene104FieldsConsumer<'a>;
    type Producer = Lucene104FieldsProducer;
    fn name(&self) -> &str {
        "Lucene104"
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        Ok(Lucene104FieldsConsumer {
            delegate: byte::PostingsFormat::fields_consumer(
                &mut byte::Lucene104PostingsFormat,
                state,
            )?,
            fields: state.field_infos,
        })
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        let mut decoded =
            byte::PostingsFormat::fields_producer(&mut byte::Lucene104PostingsFormat, state)?;
        let fields = Rc::new(decoded.materialize()?.into_owned());
        byte::FieldsProducer::close(&mut decoded)?;
        Ok(Lucene104FieldsProducer {
            fields: Some(fields),
            shared: Rc::new(NativeReaderState {
                files: RefCell::new(state.files.clone()),
                infos: state.segment.field_infos.clone(),
                closed: Cell::new(false),
            }),
        })
    }
}
struct ModelReadTerms<'a> {
    field: Cow<'a, byte::FieldPostings>,
    info: Cow<'a, FieldInfo>,
}
impl ReadTerms for ModelReadTerms<'_> {
    fn iterator(&self) -> lucene_search::Result<Box<dyn ReadTermsEnum + '_>> {
        Ok(Box::new(ModelReadEnum {
            model: self,
            next: 0,
            current: None,
        }))
    }
    fn size(&self) -> i64 {
        self.field.terms.len() as i64
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.field
            .terms
            .iter()
            .flat_map(|term| &term.postings)
            .map(|posting| {
                if self.has_freqs() {
                    i64::from(posting.freq)
                } else {
                    1
                }
            })
            .sum()
    }
    fn sum_doc_freq(&self) -> i64 {
        self.field
            .terms
            .iter()
            .map(|term| term.postings.len() as i64)
            .sum()
    }
    fn doc_count(&self) -> i32 {
        // The measured small-field read visits this statistic three times.
        // Bound the contiguous sort scratch to 128 ids (512 bytes); larger
        // fields retain the original O(unique documents) tree memory bound.
        let small = self.field.terms.iter().try_fold(0usize, |count, term| {
            count
                .checked_add(term.postings.len())
                .filter(|&count| count <= 128)
        });
        let Some(count) = small else {
            let mut docs = std::collections::BTreeSet::new();
            for term in &self.field.terms {
                for posting in &term.postings {
                    docs.insert(posting.doc);
                }
            }
            return docs.len() as i32;
        };
        let mut docs = Vec::with_capacity(count);
        docs.extend(
            self.field
                .terms
                .iter()
                .flat_map(|term| term.postings.iter().map(|posting| posting.doc)),
        );
        docs.sort_unstable();
        docs.dedup();
        docs.len() as i32
    }
    fn has_freqs(&self) -> bool {
        has_freqs(self.info.index_options)
    }
    fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }
    fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }
    fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }
}
struct ModelReadEnum<'a, 'b> {
    model: &'a ModelReadTerms<'b>,
    next: usize,
    current: Option<usize>,
}
impl ModelReadEnum<'_, '_> {
    fn positioned_term(&self) -> lucene_search::Result<&byte::Term> {
        self.current
            .and_then(|i| self.model.field.terms.get(i))
            .ok_or_else(|| lucene_search::Error::IllegalArgument("unpositioned terms enum".into()))
    }
}
impl ReadTermsEnum for ModelReadEnum<'_, '_> {
    fn next(&mut self) -> lucene_search::Result<Option<&[u8]>> {
        let Some(term) = self.model.field.terms.get(self.next) else {
            self.current = None;
            return Ok(None);
        };
        self.current = Some(self.next);
        self.next += 1;
        Ok(Some(&term.bytes))
    }
    fn term(&self) -> Option<&[u8]> {
        self.current
            .and_then(|i| self.model.field.terms.get(i))
            .map(|term| term.bytes.as_slice())
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> lucene_search::Result<SeekStatus> {
        let at = self
            .model
            .field
            .terms
            .partition_point(|term| term.bytes.as_slice() < target);
        self.next = at.saturating_add(1);
        self.current = (at < self.model.field.terms.len()).then_some(at);
        Ok(match self.term() {
            None => SeekStatus::End,
            Some(term) if term == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }
    fn doc_freq(&mut self) -> lucene_search::Result<i32> {
        Ok(self.positioned_term()?.postings.len() as i32)
    }
    fn total_term_freq(&mut self) -> lucene_search::Result<i64> {
        Ok(self
            .positioned_term()?
            .postings
            .iter()
            .map(|posting| {
                if self.model.has_freqs() {
                    i64::from(posting.freq)
                } else {
                    1
                }
            })
            .sum())
    }
    fn postings(
        &mut self,
        flags: PostingsFlags,
    ) -> lucene_search::Result<Box<dyn read::PostingsEnum>> {
        let postings = &self.positioned_term()?.postings;
        let docs = postings.iter().map(|posting| posting.doc).collect();
        let freqs = postings
            .iter()
            .map(|posting| {
                if flags == PostingsFlags::None || !self.model.has_freqs() {
                    1
                } else {
                    posting.freq
                }
            })
            .collect();
        let positions = (flags.wants_positions() && self.model.has_positions()).then(|| {
            postings
                .iter()
                .map(|posting| {
                    posting
                        .positions
                        .iter()
                        .map(|position| {
                            let mut position = position.clone();
                            if !matches!(flags, PostingsFlags::Offsets | PostingsFlags::All) {
                                position.start_offset = -1;
                                position.end_offset = -1;
                            }
                            if !matches!(flags, PostingsFlags::Payloads | PostingsFlags::All) {
                                position.payload.clear();
                            }
                            position
                        })
                        .collect()
                })
                .collect()
        });
        Ok(Box::new(MaterializedPostings::new(docs, freqs, positions)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelReadTerms<'static> {
        ModelReadTerms {
            info: Cow::Owned(
                FieldInfo::new("f", 0)
                    .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
            ),
            field: Cow::Owned(byte::FieldPostings {
                name: "f".into(),
                terms: vec![
                    byte::Term {
                        bytes: b"a".to_vec(),
                        postings: vec![byte::Posting {
                            doc: 2,
                            freq: 2,
                            positions: vec![
                                Position {
                                    position: 1,
                                    start_offset: 3,
                                    end_offset: 4,
                                    payload: vec![9],
                                },
                                Position {
                                    position: 2,
                                    start_offset: 5,
                                    end_offset: 7,
                                    payload: vec![],
                                },
                            ],
                        }],
                    },
                    byte::Term {
                        bytes: b"c".to_vec(),
                        postings: vec![byte::Posting {
                            doc: 4,
                            freq: 1,
                            positions: vec![Position {
                                position: 0,
                                start_offset: 0,
                                end_offset: 1,
                                payload: vec![],
                            }],
                        }],
                    },
                ],
            }),
        }
    }
    #[test]
    fn owned_write_enums_report_unpositioned_and_missing_occurrences() {
        let model = model();
        let terms = ModelWriteTerms {
            field: model.field,
            info: model.info,
        };
        assert!(terms.has_positions());
        assert!(terms.has_offsets());
        assert!(!terms.has_payloads());
        let mut e = terms.iterator().unwrap();
        assert!(e.postings(None, POSITIONS).is_err());
        assert_eq!(e.next().unwrap(), Some(b"a".to_vec()));
        let mut docs = e.postings(None, POSITIONS).unwrap();
        assert!(docs.freq().is_err());
        assert!(docs.next_position().is_err());
        assert_eq!(docs.next_doc().unwrap(), 2);
        assert_eq!(docs.freq().unwrap(), 2);
        assert!(docs.start_offset().is_err());
        assert!(docs.end_offset().is_err());
        assert!(docs.payload().is_err());
        assert_eq!(docs.next_position().unwrap(), 1);
        assert_eq!(docs.start_offset().unwrap(), 3);
        assert_eq!(docs.end_offset().unwrap(), 4);
        assert_eq!(docs.payload().unwrap(), Some(vec![9]));
        assert_eq!(docs.next_position().unwrap(), 2);
        assert_eq!(docs.payload().unwrap(), None);
        assert!(docs.next_position().is_err());
        assert_eq!(docs.next_doc().unwrap(), NO_MORE_DOCS);
        assert!(docs.freq().is_err());
        assert_eq!(e.next().unwrap(), Some(b"c".to_vec()));
        assert!(e.next().unwrap().is_none());
        assert!(e.postings(None, 0).is_err());
    }
    #[test]
    fn read_seeks_statistics_and_each_flag_preserve_only_requested_features() {
        let mut model = model();
        model.info.to_mut().store_payloads = true;
        assert_eq!(model.size(), 2);
        assert_eq!(model.sum_total_term_freq(), 3);
        assert_eq!(model.sum_doc_freq(), 2);
        assert_eq!(model.doc_count(), 2);
        assert!(model.has_freqs());
        assert!(model.has_positions());
        assert!(model.has_offsets());
        assert!(model.has_payloads());
        assert_eq!(model.min().unwrap(), Some(b"a".to_vec()));
        assert_eq!(model.max().unwrap(), Some(b"c".to_vec()));
        let mut e = model.iterator().unwrap();
        assert!(e.term().is_none());
        assert!(e.doc_freq().is_err());
        assert!(e.total_term_freq().is_err());
        assert!(e.postings(PostingsFlags::All).is_err());
        assert_eq!(e.try_seek_ceil(b"a").unwrap(), SeekStatus::Found);
        assert_eq!(e.term(), Some(b"a".as_slice()));
        assert_eq!(e.doc_freq().unwrap(), 1);
        assert_eq!(e.total_term_freq().unwrap(), 2);
        for flag in [
            PostingsFlags::None,
            PostingsFlags::Freqs,
            PostingsFlags::Positions,
            PostingsFlags::Offsets,
            PostingsFlags::Payloads,
            PostingsFlags::All,
        ] {
            let mut p = e.postings(flag).unwrap();
            assert_eq!(p.next_doc().unwrap(), 2);
            assert_eq!(p.freq(), if flag == PostingsFlags::None { 1 } else { 2 });
            if flag.wants_positions() {
                assert_eq!(p.next_position().unwrap(), 1);
                assert_eq!(
                    p.start_offset(),
                    if matches!(flag, PostingsFlags::Offsets | PostingsFlags::All) {
                        3
                    } else {
                        -1
                    }
                );
                assert_eq!(
                    p.end_offset(),
                    if matches!(flag, PostingsFlags::Offsets | PostingsFlags::All) {
                        4
                    } else {
                        -1
                    }
                );
                assert_eq!(
                    p.payload(),
                    matches!(flag, PostingsFlags::Payloads | PostingsFlags::All)
                        .then_some([9].as_slice())
                );
            }
        }
        assert_eq!(e.try_seek_ceil(b"b").unwrap(), SeekStatus::NotFound);
        assert_eq!(e.term(), Some(b"c".as_slice()));
        assert!(e.next().unwrap().is_none());
        assert_eq!(e.try_seek_ceil(b"z").unwrap(), SeekStatus::End);
        assert!(e.next().unwrap().is_none());
        drop(e);
        model.info.to_mut().index_options = IndexOptions::Docs;
        assert!(!model.has_freqs());
        assert!(!model.has_positions());
        assert!(!model.has_offsets());
        assert_eq!(model.sum_total_term_freq(), 2);
        let mut e = model.iterator().unwrap();
        assert!(e.next().unwrap().is_some());
        assert_eq!(e.total_term_freq().unwrap(), 1);
        let mut p = e.postings(PostingsFlags::All).unwrap();
        assert_eq!(p.next_doc().unwrap(), 2);
        assert_eq!(p.freq(), 1);
        let mut overlap = self::model();
        let posting = overlap.field.terms[0].postings[0].clone();
        overlap.field.to_mut().terms[1].postings.push(posting);
        assert_eq!(overlap.doc_count(), 2);
        let term = overlap.field.terms[0].clone();
        overlap.field.to_mut().terms = (0u16..129)
            .map(|n| {
                let mut term = term.clone();
                term.bytes = n.to_be_bytes().to_vec();
                term
            })
            .collect();
        assert_eq!(overlap.doc_count(), 1);
        overlap.field.to_mut().terms.pop();
        assert_eq!(overlap.doc_count(), 1);
    }
    struct BrokenFields;
    impl ByteFields for BrokenFields {
        fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
            Err(failure("fields failure"))
        }
    }
    struct Sink {
        closes: usize,
    }
    impl FieldsConsumer for Sink {
        fn write(&mut self, _: &dyn WriteFields, _: Option<&dyn NormsProducer>) -> Result<()> {
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            self.closes += 1;
            Ok(())
        }
    }
    #[test]
    fn lazy_native_bridge_forwards_source_errors_and_missing_field_metadata() {
        let infos = FieldInfos::new(vec![]).unwrap();
        let broken = NativeFieldsView::new(&BrokenFields, &infos);
        assert_eq!(
            broken.field_names().unwrap_err().to_string(),
            "fields failure"
        );
        assert!(broken.terms("f").is_err());
        assert!(EncodingFields {
            source: &broken,
            infos: &infos
        }
        .materialize()
        .is_err());
        let fields = byte::OwnedFields(vec![model().field.into_owned()]);
        let view = NativeFieldsView::new(&fields, &infos);
        assert_eq!(view.field_names().unwrap(), vec!["f"]);
        assert!(view.terms("absent").unwrap().is_none());
        assert!(view.terms("f").is_err());
        let mut wrapped = AssertingFieldsConsumer::new(Sink { closes: 0 }, &infos);
        assert_eq!(wrapped.delegate().closes, 0);
        wrapped.delegate_mut().close().unwrap();
        assert_eq!(wrapped.delegate().closes, 1);
    }
    #[test]
    fn native_acquisition_errors_and_producer_raw_accessors_remain_explicit() {
        let infos = FieldInfos::new(vec![]).unwrap();
        let id = [0; 16];
        let state = SegmentWriteState {
            max_doc: -1,
            segment_id: &id,
            segment_suffix: "",
            field_infos: &infos,
        };
        let mut format = AssertingPostingsFormat::new(Lucene104PostingsFormat);
        assert!(format.fields_consumer(state).is_err());
        let files = Files::new();
        assert!(format
            .fields_producer(ReadState {
                files: &files,
                segment: lucene_codecs::segment_state::SegmentReadState {
                    max_doc: 0,
                    segment_id: &id,
                    segment_suffix: "",
                    field_infos: &infos
                }
            })
            .is_err());
        let native = byte::PostingsFormat::fields_consumer(
            &mut byte::Lucene104PostingsFormat,
            SegmentWriteState {
                max_doc: 0,
                ..state
            },
        )
        .unwrap();
        let mut native = native;
        ByteConsumer::close(&mut native).unwrap();
        let files = byte::output_files(native.output().unwrap());
        let mut reader = format
            .fields_producer(ReadState {
                files: &files,
                segment: lucene_codecs::segment_state::SegmentReadState {
                    max_doc: 0,
                    segment_id: &id,
                    segment_suffix: "",
                    field_infos: &infos,
                },
            })
            .unwrap();
        assert_eq!(reader.delegate().size(), 0);
        let mut alias = reader.get_merge_instance();
        assert!(Rc::ptr_eq(
            reader.delegate().fields.as_ref().unwrap(),
            alias.delegate().fields.as_ref().unwrap()
        ));
        assert!(matches!(
            reader.delegate().materialize().unwrap(),
            Cow::Borrowed(_)
        ));
        reader
            .delegate()
            .shared
            .files
            .borrow_mut()
            .insert("short".into(), vec![]);
        assert!(reader.check_integrity().is_err());
        reader.delegate().shared.files.borrow_mut().remove("short");
        reader
            .delegate()
            .shared
            .files
            .borrow_mut()
            .get_mut("doc")
            .unwrap()[8] ^= 1;
        assert!(reader.check_integrity().is_err());
        reader.delegate_mut().close().unwrap();
        assert!(reader.delegate().fields.is_none());
        assert!(alias.delegate().materialize().is_err());
        assert!(alias.check_integrity().is_err());
        alias.close().unwrap();
        assert!(alias.delegate().fields.is_none());
        assert!(reader.terms("absent").unwrap().is_none());
    }

    struct TracedSource {
        label: &'static str,
        events: Rc<RefCell<Vec<String>>>,
        fields: byte::OwnedFields,
        calls: Cell<usize>,
        fail_at: Cell<usize>,
        integrity_fail: Cell<bool>,
    }
    impl ByteFields for TracedSource {
        fn materialize(&self) -> Result<Cow<'_, [byte::FieldPostings]>> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            self.events
                .borrow_mut()
                .push(format!("{}:{call}", self.label));
            if call == self.fail_at.get() {
                return Err(failure("later source failure"));
            }
            self.fields.materialize()
        }
    }
    impl byte::FieldsProducer for TracedSource {
        fn check_integrity(&self) -> Result<()> {
            self.events
                .borrow_mut()
                .push(format!("integrity:{}", self.label));
            if self.integrity_fail.get() {
                return Err(failure("integrity failure"));
            }
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    fn traced(label: &'static str, events: Rc<RefCell<Vec<String>>>) -> TracedSource {
        TracedSource {
            label,
            events,
            calls: Cell::new(0),
            fail_at: Cell::new(usize::MAX),
            integrity_fail: Cell::new(false),
            fields: byte::OwnedFields(
                ["f", "g"]
                    .into_iter()
                    .enumerate()
                    .map(|(doc, name)| byte::FieldPostings {
                        name: name.into(),
                        terms: vec![byte::Term {
                            bytes: b"x".to_vec(),
                            postings: vec![byte::Posting {
                                doc: doc as i32,
                                freq: 1,
                                positions: vec![],
                            }],
                        }],
                    })
                    .collect(),
            ),
        }
    }
    struct InspectMapped(Rc<RefCell<Vec<String>>>);
    impl FieldsConsumer for InspectMapped {
        fn write(&mut self, fields: &dyn WriteFields, _: Option<&dyn NormsProducer>) -> Result<()> {
            self.0.borrow_mut().push("write".into());
            assert_eq!(fields.field_names()?, vec!["f", "g"]);
            let terms = fields.terms("f")?.unwrap();
            let mut e = terms.iterator()?;
            assert_eq!(e.next()?, Some(b"x".to_vec()));
            let mut docs = e.postings(None, FREQS)?;
            assert_eq!(docs.next_doc()?, 0);
            assert_eq!(docs.next_doc()?, 2);
            assert_eq!(docs.next_doc()?, NO_MORE_DOCS);
            assert!(fields.terms("f")?.is_some());
            assert!(fields.terms("absent")?.is_none());
            Ok(())
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
    #[test]
    fn selective_merge_repeats_every_source_callback_after_integrity_and_abort_checks() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let first = traced("a", events.clone());
        let second = traced("b", events.clone());
        let infos = FieldInfos::new(
            ["f", "g"]
                .into_iter()
                .enumerate()
                .map(|(n, name)| {
                    FieldInfo::new(name, n as i32).with_index_options(IndexOptions::DocsAndFreqs)
                })
                .collect(),
        )
        .unwrap();
        let sources = [
            byte::MergeSource {
                reader: Some(&first),
                max_doc: 2,
                doc_map: &[0, 1],
            },
            byte::MergeSource {
                reader: None,
                max_doc: 0,
                doc_map: &[],
            },
            byte::MergeSource {
                reader: Some(&second),
                max_doc: 2,
                doc_map: &[2, 3],
            },
        ];
        let check = || {
            events.borrow_mut().push("abort".into());
            Ok(())
        };
        let state = MergeState {
            bytes: byte::MergeState {
                sources: &sources,
                check_aborted: &check,
            },
            field_infos: &infos,
        };
        InspectMapped(events.clone()).merge(&state, None).unwrap();
        assert_eq!(
            *events.borrow(),
            [
                "abort",
                "integrity:a",
                "abort",
                "integrity:b",
                "write",
                "a:1",
                "b:1",
                "a:2",
                "b:2",
                "a:3",
                "b:3",
                "a:4",
                "b:4"
            ]
        );
        second.fail_at.set(5);
        let view = MappedWriteFields(&state);
        assert_eq!(
            view.terms("f").err().unwrap().to_string(),
            "later source failure"
        );
        assert_eq!(
            &events.borrow()[events.borrow().len() - 2..],
            ["a:5", "b:5"]
        );
        assert!(view.terms("f").unwrap().is_some());
        assert_eq!(first.calls.get(), 6);
        assert_eq!(second.calls.get(), 6);
        events.borrow_mut().clear();
        second.integrity_fail.set(true);
        assert_eq!(
            InspectMapped(events.clone())
                .merge(&state, None)
                .err()
                .unwrap()
                .to_string(),
            "integrity failure"
        );
        assert_eq!(
            *events.borrow(),
            ["abort", "integrity:a", "abort", "integrity:b"]
        );
        events.borrow_mut().clear();
        let abort = || {
            events.borrow_mut().push("abort-failure".into());
            Err(failure("abort failure"))
        };
        let state = MergeState {
            bytes: byte::MergeState {
                sources: &sources,
                check_aborted: &abort,
            },
            field_infos: &infos,
        };
        assert_eq!(
            InspectMapped(events.clone())
                .merge(&state, None)
                .err()
                .unwrap()
                .to_string(),
            "abort failure"
        );
        assert_eq!(*events.borrow(), ["abort-failure"]);
    }
    #[test]
    fn selective_merge_checks_unrelated_fields_before_returning_requested_terms() {
        let infos = FieldInfos::new(vec![
            FieldInfo::new("f", 0).with_index_options(IndexOptions::DocsAndFreqs)
        ])
        .unwrap();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut producer = traced("a", events);
        for (max_doc, map, message) in [
            (2, &[0, -2][..], "invalid mapped document"),
            (2, &[0][..], "document map out of bounds"),
            (1, &[0, 1][..], "source document out of bounds"),
        ] {
            let sources = [byte::MergeSource {
                reader: Some(&producer),
                max_doc,
                doc_map: map,
            }];
            let state = MergeState {
                bytes: byte::MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(()),
                },
                field_infos: &infos,
            };
            let view = MappedWriteFields(&state);
            assert_eq!(view.field_names().unwrap_err().to_string(), message);
            assert_eq!(view.terms("f").err().unwrap().to_string(), message);
            assert_eq!(view.terms("absent").err().unwrap().to_string(), message);
        }
        producer.fields.0[1].terms[0].postings[0].doc = -1;
        let sources = [byte::MergeSource {
            reader: Some(&producer),
            max_doc: 2,
            doc_map: &[0, 1],
        }];
        let state = MergeState {
            bytes: byte::MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            field_infos: &infos,
        };
        assert!(MappedWriteFields(&state).terms("f").is_err());
        producer.fields.0[1].terms[0].postings[0].doc = 1;
        let sources = [byte::MergeSource {
            reader: Some(&producer),
            max_doc: 2,
            doc_map: &[-1, -1],
        }];
        let state = MergeState {
            bytes: byte::MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            field_infos: &infos,
        };
        let view = MappedWriteFields(&state);
        assert_eq!(view.field_names().unwrap(), vec!["f", "g"]);
        assert!(view
            .terms("f")
            .unwrap()
            .unwrap()
            .iterator()
            .unwrap()
            .next()
            .unwrap()
            .is_none());
        assert_eq!(
            view.terms("g").err().unwrap().to_string(),
            "unknown postings field"
        );
        assert!(view.terms("absent").unwrap().is_none());
    }
}
