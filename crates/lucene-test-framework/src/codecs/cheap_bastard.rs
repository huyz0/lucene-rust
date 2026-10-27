//! `CheapBastardCodec`: a default Lucene104 delegate and one stable direct
//! Lucene104 postings format with block-tree term blocks of 100..200 items.
//! Java's default format uses 25..48. No constructor or accessor consumes RNG.
//!
//! The native component adapter writes real `.doc/.psm/.pos/.pay/.tim/.tip/.tmd`
//! bytes through the production writer with those block settings; its reader
//! reuses the existing native Lucene104 reader. The owned Fields/norms, eager
//! decoding, buffered publication and merge/error-timing limits are the same as
//! `cranky_postings`' explicit adapter. Repeated writes are refused, close is
//! idempotent and an unwritten close publishes the empty format's files.
//! Merges retain invocation-local source views and map borrowed postings, so
//! position/payload graphs are copied only into the encoder's real columns.
//!
//! This is a direct component API. Java's ten other top-level format accessors
//! forward to its stable default codec. Here `delegate()` exposes the existing
//! default Lucene104 routing descriptor. `CodecDelegate` provides per-invocation
//! callback forwarding for all ten accessors, including fresh component identity
//! and the first delegate error. The native default descriptor supplies no such
//! factories; FilterCodec SPI/name installation and Directory/IOContext callbacks
//! are not modeled.
//! Installed in `IndexWriter` (its `Lucene104Codec` impl), segments record
//! `CheapBastard` and the writer's own flush and merge write the direct
//! format, unsuffixed and without per-field attributes, as Java's does.

use super::cranky_postings::{self as native, Fields, FieldsConsumer, PostingsFormat};
use lucene_codecs::codec::DefaultLucene104Codec;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::per_field_postings;
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, Output, TermPostings};
use lucene_codecs::segment_state::SegmentWriteState;
pub use native::{
    output_files, Error, Files, NormsProducer, OwnedFields, OwnedNorms, ReadState, Result,
};
use std::borrow::Cow;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};

fn failure(message: impl Into<String>) -> Error {
    Error::Message(message.into())
}

/// Java: CheapBastardCodec's construction and direct postings override.
pub struct CheapBastardCodec<D = DefaultLucene104Codec> {
    delegate: D,
    postings: CheapPostingsFormat,
}
impl Default for CheapBastardCodec {
    fn default() -> Self {
        Self::new()
    }
}
impl CheapBastardCodec {
    pub fn new() -> Self {
        Self::with_delegate(DefaultLucene104Codec)
    }
}
impl<D> CheapBastardCodec<D> {
    /// Local callback seam for Java FilterCodec's otherwise native-absent factories.
    pub fn with_delegate(delegate: D) -> Self {
        Self {
            delegate,
            postings: CheapPostingsFormat {
                settings: per_field_postings::Lucene104PostingsFormat::new(100, 200)
                    .expect("fixed valid Java block settings"),
            },
        }
    }
    pub fn name(&self) -> &str {
        "CheapBastard"
    }
    pub fn delegate(&self) -> &D {
        &self.delegate
    }
    pub fn postings_format(&self) -> &CheapPostingsFormat {
        &self.postings
    }
    pub fn postings_format_mut(&mut self) -> &mut CheapPostingsFormat {
        &mut self.postings
    }
}

impl<D> std::fmt::Debug for CheapBastardCodec<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheapBastardCodec").finish_non_exhaustive()
    }
}

/// Installation in `IndexWriter::set_codec`: Java's codec as `IndexWriter`
/// runs it. Every field's postings go to the one direct
/// `Lucene104PostingsFormat(100, 200)` (unsuffixed `_0.tim`, no per-field
/// attributes) and every other format is `Lucene104`'s; segments record
/// `CheapBastard`, which [`super::register_codecs`] makes readable.
impl lucene_codecs::codec::Lucene104Codec for CheapBastardCodec<DefaultLucene104Codec> {
    fn name(&self) -> &str {
        CheapBastardCodec::name(self)
    }

    fn postings_format_for_field(
        &self,
        _field: &str,
    ) -> per_field_postings::Lucene104PostingsFormat {
        self.postings.settings.direct()
    }
}

/// Java FilterCodec forwards these ten accessors on every call. A delegate may
/// return a fresh object (Lucene104's points format does), or return an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentKind {
    DocValues,
    FieldInfos,
    LiveDocs,
    Norms,
    SegmentInfo,
    StoredFields,
    TermVectors,
    Compound,
    Points,
    KnnVectors,
}
pub trait CodecDelegate {
    type Component;
    fn component(&self, kind: ComponentKind) -> Result<Self::Component>;
}
impl<D: CodecDelegate> CheapBastardCodec<D> {
    pub fn component(&self, kind: ComponentKind) -> Result<D::Component> {
        self.delegate.component(kind)
    }
}

pub struct CheapPostingsFormat {
    settings: per_field_postings::Lucene104PostingsFormat,
}
impl CheapPostingsFormat {
    pub fn settings(&self) -> &per_field_postings::Lucene104PostingsFormat {
        &self.settings
    }
}
pub struct CheapFieldsConsumer<'a> {
    state: SegmentWriteState<'a>,
    options: postings_writer::WriteOptions,
    output: Option<Output>,
    closed: bool,
}
impl CheapFieldsConsumer<'_> {
    pub fn output(&self) -> Option<&Output> {
        self.output.as_ref()
    }
}
impl PostingsFormat for CheapPostingsFormat {
    type Consumer<'a> = CheapFieldsConsumer<'a>;
    type Producer = native::Lucene104FieldsProducer;
    fn name(&self) -> &str {
        self.settings.name()
    }
    fn fields_consumer<'a>(
        &mut self,
        state: SegmentWriteState<'a>,
    ) -> Result<CheapFieldsConsumer<'a>> {
        if state.max_doc < 0 {
            return Err(failure("negative maxDoc"));
        }
        Ok(CheapFieldsConsumer {
            state,
            options: state.postings_options(self.settings.write_options()),
            output: None,
            closed: false,
        })
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        native::Lucene104PostingsFormat.fields_producer(state)
    }
}

struct MappedTerm<'a> {
    bytes: &'a [u8],
    postings: std::ops::Range<usize>,
}
#[derive(Default)]
struct MappedField<'a> {
    terms: Vec<MappedTerm<'a>>,
    postings: Vec<(i32, &'a native::Posting)>,
}
type MappedPostings<'a> = BTreeMap<&'a str, MappedField<'a>>;
fn encode_term<'p>(
    info: &FieldInfo,
    bytes: &[u8],
    postings: impl ExactSizeIterator<Item = (i32, &'p native::Posting)> + Clone,
    max_doc: i32,
) -> Result<TermPostings> {
    let count = postings.len();
    let positions = info.index_options.subsumes_positions();
    let offsets = info.index_options.subsumes_offsets();
    let mut encoded = TermPostings {
        term: bytes.to_vec(),
        docs: Vec::with_capacity(count),
        positions: Vec::with_capacity(if positions { count } else { 0 }),
        offsets: Vec::with_capacity(if offsets { count } else { 0 }),
        ..TermPostings::default()
    };
    if positions && info.store_payloads {
        encoded
            .payload_lengths
            .reserve(postings.clone().map(|(_, p)| p.positions.len()).sum());
    }
    for (doc, posting) in postings {
        if doc < 0 || doc >= max_doc {
            return Err(failure("destination document out of bounds"));
        }
        encoded.docs.push((
            doc,
            if info.index_options == IndexOptions::Docs {
                1
            } else {
                posting.freq
            },
        ));
        if positions {
            encoded
                .positions
                .push(posting.positions.iter().map(|p| p.position).collect());
            if offsets {
                encoded.offsets.push(
                    posting
                        .positions
                        .iter()
                        .map(|p| (p.start_offset, p.end_offset))
                        .collect(),
                );
            }
            if info.store_payloads {
                for position in &posting.positions {
                    encoded.payload_lengths.push(
                        u32::try_from(position.payload.len())
                            .map_err(|_| failure("payload too long"))?,
                    );
                    encoded.payload_bytes.extend_from_slice(&position.payload);
                }
            }
        }
    }
    Ok(encoded)
}
fn collect_norms(
    info: &FieldInfo,
    norms: Option<&dyn NormsProducer>,
    columns: &mut Vec<(i32, Vec<i64>)>,
) -> Result<()> {
    if !info.omit_norms {
        if let Some(norms) = norms {
            if let Some(values) = norms.norms(info)? {
                columns.push((
                    info.number,
                    values.iter().map(|value| value.unwrap_or(1)).collect(),
                ));
            }
        }
    }
    Ok(())
}
fn encode_fields(
    state: &SegmentWriteState<'_>,
    options: &postings_writer::WriteOptions,
    metadata: &[&FieldInfo],
    encoded: &[Vec<TermPostings>],
    norm_columns: &[(i32, Vec<i64>)],
) -> Result<Output> {
    let inputs: Vec<_> = metadata
        .iter()
        .zip(encoded)
        .map(|(info, terms)| FieldPostingsInput {
            field_number: info.number,
            index_options: info.index_options,
            has_payloads: info.store_payloads,
            doc_count: terms
                .iter()
                .flat_map(|term| term.docs.iter().map(|(doc, _)| *doc))
                .collect::<BTreeSet<_>>()
                .len() as i32,
            terms,
        })
        .collect();
    let norms: Vec<_> = norm_columns
        .iter()
        .map(|(field_number, values)| FieldNorms {
            field_number: *field_number,
            values,
        })
        .collect();
    Ok(if inputs.is_empty() {
        postings_writer::write_no_fields(options, state.segment_id, state.segment_suffix)?
    } else {
        postings_writer::write_fields_with_options(
            &inputs,
            &norms,
            options,
            state.segment_id,
            state.segment_suffix,
        )?
    })
}

impl<'a> CheapFieldsConsumer<'a> {
    fn ensure_writable(&self) -> Result<()> {
        if self.closed {
            return Err(failure("closed postings writer"));
        }
        if self.output.is_some() {
            return Err(failure("postings already written"));
        }
        Ok(())
    }
    fn info(&self, name: &str) -> Result<&'a FieldInfo> {
        let info = self
            .state
            .field_infos
            .field_by_name(name)
            .ok_or_else(|| failure("unrecognized postings field"))?;
        if info.index_options == IndexOptions::None {
            return Err(failure("field does not index postings"));
        }
        Ok(info)
    }
}
impl FieldsConsumer for CheapFieldsConsumer<'_> {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        self.ensure_writable()?;
        let fields = fields.materialize()?;
        let mut encoded = Vec::with_capacity(fields.len());
        let mut metadata = Vec::with_capacity(fields.len());
        let mut columns = Vec::new();
        for field in fields.iter() {
            let info = self.info(&field.name)?;
            let mut terms = Vec::with_capacity(field.terms.len());
            for term in &field.terms {
                if term.postings.is_empty() {
                    continue;
                }
                terms.push(encode_term(
                    info,
                    &term.bytes,
                    term.postings.iter().map(|posting| (posting.doc, posting)),
                    self.state.max_doc,
                )?);
            }
            if terms.is_empty() {
                continue;
            }
            collect_norms(info, norms, &mut columns)?;
            metadata.push(info);
            encoded.push(terms);
        }
        self.output = Some(encode_fields(
            &self.state,
            &self.options,
            &metadata,
            &encoded,
            &columns,
        )?);
        Ok(())
    }
    fn merge(
        &mut self,
        state: &native::MergeState<'_>,
        norms: Option<&dyn NormsProducer>,
    ) -> Result<()> {
        for source in state.sources {
            if let Some(reader) = source.reader {
                (state.check_aborted)()?;
                reader.check_integrity()?;
            }
        }
        self.ensure_writable()?;
        // Each callback runs once, in source order. Hold owned and borrowed Cow
        // results until encoding ends so mapped postings can retain references.
        let holders: Vec<OnceCell<Cow<'_, [native::FieldPostings]>>> =
            (0..state.sources.len()).map(|_| OnceCell::new()).collect();
        let mut fields: MappedPostings<'_> = BTreeMap::new();
        for (source, holder) in state.sources.iter().zip(&holders) {
            let Some(reader) = source.reader else {
                continue;
            };
            let materialized = reader.materialize()?;
            for field in holder.get_or_init(|| materialized).iter() {
                let mapped = fields.entry(&field.name).or_default();
                mapped.terms.reserve(field.terms.len());
                mapped
                    .postings
                    .reserve(field.terms.iter().map(|term| term.postings.len()).sum());
                for term in &field.terms {
                    let start = mapped.postings.len();
                    for posting in &term.postings {
                        if posting.doc < 0 || posting.doc >= source.max_doc {
                            return Err(failure("source document out of bounds"));
                        }
                        let at = usize::try_from(posting.doc)
                            .map_err(|_| failure("negative source document"))?;
                        let doc = *source
                            .doc_map
                            .get(at)
                            .ok_or_else(|| failure("document map out of bounds"))?;
                        // SENTINEL: MergeState.DocMap returns -1 for a deleted source document.
                        if doc == -1 {
                            continue;
                        }
                        if doc < 0 {
                            return Err(failure("invalid mapped document"));
                        }
                        mapped.postings.push((doc, posting));
                    }
                    mapped.terms.push(MappedTerm {
                        bytes: &term.bytes,
                        postings: start..mapped.postings.len(),
                    });
                }
            }
        }
        let mut fields: Vec<_> = fields.into_iter().collect();
        fields.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
        let mut encoded = Vec::with_capacity(fields.len());
        let mut metadata = Vec::with_capacity(fields.len());
        let mut columns = Vec::new();
        let mut scratch = Vec::new();
        for (name, mut field) in fields {
            let info = self.info(name)?;
            // Stable grouping retains source encounter order for equal terms,
            // including repeated terms/fields supplied by an arbitrary source.
            field
                .terms
                .sort_by(|left, right| left.bytes.cmp(right.bytes));
            let mut converted = Vec::with_capacity(field.terms.len());
            let mut at = 0;
            while at < field.terms.len() {
                let bytes = field.terms[at].bytes;
                scratch.clear();
                while at < field.terms.len() && field.terms[at].bytes == bytes {
                    scratch.extend_from_slice(&field.postings[field.terms[at].postings.clone()]);
                    at += 1;
                }
                if scratch.is_empty() {
                    continue;
                }
                scratch.sort_by_key(|(doc, _)| *doc);
                converted.push(encode_term(
                    info,
                    bytes,
                    scratch.iter().copied(),
                    self.state.max_doc,
                )?);
            }
            if converted.is_empty() {
                continue;
            }
            collect_norms(info, norms, &mut columns)?;
            metadata.push(info);
            encoded.push(converted);
        }
        self.output = Some(encode_fields(
            &self.state,
            &self.options,
            &metadata,
            &encoded,
            &columns,
        )?);
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.output.is_none() {
            self.output = Some(postings_writer::write_no_fields(
                &self.options,
                self.state.segment_id,
                self.state.segment_suffix,
            )?);
        }
        self.closed = true;
        Ok(())
    }
}
