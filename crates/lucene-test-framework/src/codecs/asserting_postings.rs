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
//! and ordinary next-term values are owned copies; installed native validation
//! borrows current term bytes and reuses one previous-term buffer. Read-side
//! term statistics and posting frequency/offset access inherit the public
//! reader's infallible accessors and its unsupported ordinal/state/reuse and
//! per-call thread-check limitations.
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
//! Native Display is a stable label. The optional installed writer wraps native
//! Lucene104 groups and checks their borrowed flat inputs after successful
//! delegation. Group ownership, Directory cleanup and publication belong to the
//! index layer. Source integrity/mapping timing remains index-owned; IOContext,
//! reader-side installation and panic cleanup remain incomplete. Native encoding
//! buffers bytes rather than streaming Java's callback-level I/O. No random
//! decisions are added.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings::Position;
use lucene_codecs::postings_consumer::{
    NativePostingsWriter, PostingsWriteSink, PostingsWriteState, PostingsWriterFactory,
};
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource};
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
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum<'_> + '_>>;
    fn has_positions(&self) -> bool;
    fn has_offsets(&self) -> bool;
    fn has_payloads(&self) -> bool;
}
pub trait WriteTermsEnum<'a> {
    fn next(&mut self) -> Result<Option<Vec<u8>>>;
    /// The default preserves owned enumerators and their callback order.
    /// Native validation borrows current bytes until its next enum operation.
    fn next_bytes(&mut self) -> Result<Option<Cow<'_, [u8]>>> {
        Ok(self.next()?.map(Cow::Owned))
    }
    fn postings(
        &mut self,
        reuse: Option<Box<dyn WritePostings + 'a>>,
        flags: u32,
    ) -> Result<Box<dyn WritePostings + 'a>>;
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
        validate_write_fields(fields, &self.fields.fields)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}
/// Java: `AssertingFieldsConsumer.write`, after successful delegation.
fn validate_write_fields(fields: &dyn WriteFields, infos: &[FieldInfo]) -> Result<()> {
    let mut last_field: Option<String> = None;
    for name in fields.field_names()? {
        let info = infos
            .iter()
            .find(|field| field.name == name)
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
        let mut postings: Option<Box<dyn WritePostings + '_>> = None;
        let has_freqs = has_freqs(info.index_options);
        let has_positions = info.index_options.subsumes_positions();
        let has_offsets = info.index_options.subsumes_offsets();
        let has_payloads = terms.has_payloads();
        assert_eq!(has_positions, terms.has_positions());
        assert_eq!(has_offsets, terms.has_offsets());
        loop {
            // Measured validation cost motivates borrowing current term bytes
            // and retaining one previous-term buffer, like Java BytesRefBuilder.
            // End the enum borrow before asking it for the current postings.
            {
                let Some(term) = terms_enum.next_bytes()? else {
                    break;
                };
                assert!(last_term
                    .as_ref()
                    .is_none_or(|last| last.as_slice() < term.as_ref()));
                match term {
                    Cow::Owned(term) => last_term = Some(term),
                    Cow::Borrowed(term) => {
                        let previous = last_term.get_or_insert_with(Vec::new);
                        previous.clear();
                        previous.extend_from_slice(term);
                    }
                }
            }
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

/// Native codec identity with assertions on each native Lucene104 postings group.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingPostingsCodec;
impl Lucene104Codec for AssertingPostingsCodec {
    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(self)
    }
}
impl PostingsWriterFactory for AssertingPostingsCodec {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
        Ok(Box::new(InstalledAssertingConsumer {
            delegate: NativePostingsWriter::new(state)?,
            infos: state.field_infos,
        }))
    }
}
struct InstalledAssertingConsumer<'a, W> {
    delegate: W,
    infos: &'a [FieldInfo],
}
impl<W: PostingsWriteSink> PostingsWriteSink for InstalledAssertingConsumer<'_, W> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.delegate.write(inputs, norms)?;
        validate_native_inputs(inputs, self.infos)
    }
    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        // Java's inherited merge invokes this wrapper's write. Native sources
        // have already been checked and mapped by the index layer.
        self.delegate.merge(inputs, norms)?;
        validate_native_inputs(inputs, self.infos)
    }
    fn close(&mut self) -> postings_writer::Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
}
fn validate_native_inputs(
    inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
    infos: &[FieldInfo],
) -> postings_writer::Result<()> {
    validate_write_fields(&BorrowedNativeFields { inputs, infos }, infos)
        .map_err(|error| lucene_store::Error::Io(std::io::Error::other(error.to_string())).into())
}

/// Re-enumerates the exact native runs after encoding; only names and term bytes
/// are copied, as on the existing assertion API. Posting lists remain borrowed.
struct BorrowedNativeFields<'a, 'b> {
    inputs: &'a [FieldPostingsInput<'b, &'b dyn TermSource>],
    infos: &'a [FieldInfo],
}
impl WriteFields for BorrowedNativeFields<'_, '_> {
    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self
            .inputs
            .iter()
            .map(|input| {
                self.infos
                    .iter()
                    .find(|info| info.number == input.field_number)
                    .expect("unknown postings field")
                    .name
                    .clone()
            })
            .collect())
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn WriteTerms + '_>>> {
        let Some(info) = self.infos.iter().find(|info| info.name == name) else {
            return Ok(None);
        };
        Ok(self
            .inputs
            .iter()
            .find(|input| input.field_number == info.number)
            .map(|input| {
                Box::new(BorrowedNativeTerms {
                    terms: input.terms,
                    options: input.index_options,
                    payloads: input.has_payloads,
                }) as Box<dyn WriteTerms + '_>
            }))
    }
}
struct BorrowedNativeTerms<'a> {
    terms: &'a [&'a dyn TermSource],
    options: IndexOptions,
    payloads: bool,
}
impl WriteTerms for BorrowedNativeTerms<'_> {
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum<'_> + '_>> {
        Ok(Box::new(BorrowedNativeEnum {
            terms: self.terms,
            next: 0,
            current: None,
        }))
    }
    fn has_positions(&self) -> bool {
        self.options.subsumes_positions()
    }
    fn has_offsets(&self) -> bool {
        self.options.subsumes_offsets()
    }
    fn has_payloads(&self) -> bool {
        self.payloads
    }
}
struct BorrowedNativeEnum<'a> {
    terms: &'a [&'a dyn TermSource],
    next: usize,
    current: Option<usize>,
}
impl<'a> WriteTermsEnum<'a> for BorrowedNativeEnum<'a> {
    fn next(&mut self) -> Result<Option<Vec<u8>>> {
        Ok(self.next_bytes()?.map(Cow::into_owned))
    }
    fn next_bytes(&mut self) -> Result<Option<Cow<'_, [u8]>>> {
        let Some(term) = self.terms.get(self.next) else {
            self.current = None;
            return Ok(None);
        };
        self.current = Some(self.next);
        self.next += 1;
        Ok(Some(Cow::Borrowed(term.term())))
    }
    fn postings(
        &mut self,
        _reuse: Option<Box<dyn WritePostings + 'a>>,
        flags: u32,
    ) -> Result<Box<dyn WritePostings + 'a>> {
        let current = self
            .current
            .ok_or_else(|| failure("unpositioned terms enum"))?;
        Ok(Box::new(BorrowedNativePostings {
            term: self.terms[current],
            flags,
            next: 0,
            current: None,
            position: 0,
            occurrence_base: 0,
        }))
    }
}
struct BorrowedNativePostings<'a> {
    term: &'a dyn TermSource,
    flags: u32,
    next: usize,
    current: Option<usize>,
    position: usize,
    occurrence_base: usize,
}
impl BorrowedNativePostings<'_> {
    fn doc_index(&self) -> Result<usize> {
        self.current.ok_or_else(|| failure("unpositioned postings"))
    }
    fn occurrence(&self) -> Result<usize> {
        self.position
            .checked_sub(1)
            .ok_or_else(|| failure("no position"))
    }
    fn offsets(&self) -> Result<(i32, i32)> {
        if self.flags & OFFSETS != OFFSETS {
            return Ok((-1, -1));
        }
        self.term
            .offsets(self.doc_index()?)
            .get(self.occurrence()?)
            .copied()
            .ok_or_else(|| failure("missing offsets"))
    }
}
impl WritePostings for BorrowedNativePostings<'_> {
    fn next_doc(&mut self) -> Result<i32> {
        if let Some(current) = self.current {
            self.occurrence_base += usize::try_from(self.term.docs()[current].1)
                .map_err(|_| failure("negative frequency"))?;
        }
        self.position = 0;
        let Some(&(doc, _)) = self.term.docs().get(self.next) else {
            self.current = None;
            return Ok(NO_MORE_DOCS);
        };
        self.current = Some(self.next);
        self.next += 1;
        Ok(doc)
    }
    fn freq(&mut self) -> Result<i32> {
        let index = self.doc_index()?;
        Ok(if self.flags & FREQS == FREQS {
            self.term.docs()[index].1
        } else {
            1
        })
    }
    fn next_position(&mut self) -> Result<i32> {
        if self.flags & POSITIONS != POSITIONS {
            return Err(failure("positions not requested"));
        }
        let value = self
            .term
            .positions(self.doc_index()?)
            .get(self.position)
            .copied()
            .ok_or_else(|| failure("missing position"))?;
        self.position += 1;
        Ok(value)
    }
    fn start_offset(&mut self) -> Result<i32> {
        Ok(self.offsets()?.0)
    }
    fn end_offset(&mut self) -> Result<i32> {
        Ok(self.offsets()?.1)
    }
    fn payload(&mut self) -> Result<Option<Vec<u8>>> {
        if self.flags & PAYLOADS != PAYLOADS {
            return Ok(None);
        }
        self.doc_index()?;
        let occurrence = self.occurrence_base + self.occurrence()?;
        let lengths = self.term.payload_lengths();
        let length = *lengths
            .get(occurrence)
            .ok_or_else(|| failure("missing payload length"))? as usize;
        let start = lengths[..occurrence]
            .iter()
            .try_fold(0usize, |sum, &length| {
                sum.checked_add(length as usize)
                    .ok_or_else(|| failure("payload length overflow"))
            })?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| failure("payload length overflow"))?;
        let bytes = self
            .term
            .payload_bytes()
            .get(start..end)
            .ok_or_else(|| failure("missing payload bytes"))?;
        Ok((!bytes.is_empty()).then(|| bytes.to_vec()))
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
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum<'_> + '_>> {
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
impl<'a> WriteTermsEnum<'a> for ModelWriteEnum<'a> {
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
        _reuse: Option<Box<dyn WritePostings + 'a>>,
        _flags: u32,
    ) -> Result<Box<dyn WritePostings + 'a>> {
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
mod tests;

#[cfg(test)]
mod installed_tests;
