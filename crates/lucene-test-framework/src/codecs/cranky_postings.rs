//! Java's `CrankyPostingsFormat`: one RNG shared by format and consumer.
//! Consumer acquisition and `write` draw before delegation; `close` draws only
//! after successful delegation. Producer acquisition and every reader callback
//! pass through unchanged. The inherited `FieldsConsumer.merge` checks abort
//! and integrity before calling this wrapper's guarded `write`.
//!
//! The explicit Lucene104 adapter uses the existing binary codec on real
//! `.doc/.psm/.pos/.pay/.tim/.tip/.tmd` buffers. `.psm` carries four little-endian
//! impact maxima and the final data-file lengths after its index header.
//! Field names use Java UTF-16 String order; term bytes retain unsigned byte order.
//! Fields/terms/postings are owned models rather than lazy Java enums. The writer
//! snapshots norms once per nonempty field, defaulting a missing producer/column
//! to 1; Java acquires an iterator per term and requires one for normed fields.
//! Per-term norms acquisition and its error timing are outside this native seam.
//! Mapping is deferred until the delegate asks for fields, but materialization then
//! groups and sorts all mapped postings; enum-level errors and exact streaming
//! error timing cannot be represented. The buffered consumer exposes complete
//! output files and rejects repeated writes; Directory open/publication/close
//! errors, IOContext and Codec/IndexWriter routing are absent. Native readers
//! eagerly decode on open and retain immutable files for integrity checks;
//! close clears their own fields/files and repeated close succeeds. Generic
//! callbacks preserve their delegate's errors and identity. Shared JavaRandom
//! uses single-threaded Rc/RefCell instead of Java's thread-safe Random.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::postings::{DocInput, PayInput, PosInput, Position};
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, Output, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::java_random::JavaRandom;

pub const FAKE_OPEN_ERROR: &str = "Fake IOException from PostingsFormat.fieldsConsumer()";
pub const FAKE_WRITE_ERROR: &str = "Fake IOException from FieldsConsumer.write()";
pub const FAKE_CLOSE_ERROR: &str = "Fake IOException from FieldsConsumer.close()";
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    Postings(#[from] postings_writer::Error),
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    #[error(transparent)]
    ReadPostings(#[from] lucene_codecs::postings::Error),
    #[error("{0}")]
    Message(String),
}
pub type Result<T> = std::result::Result<T, Error>;
fn failure(message: impl Into<String>) -> Error {
    Error::Message(message.into())
}
fn guard(random: &RefCell<JavaRandom>, message: &str) -> Result<()> {
    if random.borrow_mut().next_int_bounded(100) == 0 {
        Err(failure(message))
    } else {
        Ok(())
    }
}

/// A document's `PostingsEnum` result, including each positional occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Posting {
    pub doc: i32,
    pub freq: i32,
    pub positions: Vec<Position>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    pub bytes: Vec<u8>,
    pub postings: Vec<Posting>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPostings {
    pub name: String,
    pub terms: Vec<Term>,
}
/// Java: MultiFields' MergedIterator and BlockTreeTermsReader.sortFieldNames
/// compare String names by UTF-16 code units, unlike Rust's UTF-8 String order.
fn sort_field_names(fields: &mut [FieldPostings]) {
    fields.sort_by(|left, right| left.name.encode_utf16().cmp(right.name.encode_utf16()));
}
/// `Fields`: no statistics are required from a merge's mapped view.
pub trait Fields {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>>;
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnedFields(pub Vec<FieldPostings>);
impl Fields for OwnedFields {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
        Ok(Cow::Borrowed(&self.0))
    }
}
pub trait FieldsProducer: Fields {
    fn check_integrity(&self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
}
/// Dense optional values model a fresh `getNorms` iterator's advanceExact.
pub trait NormsProducer {
    fn norms(&self, field: &FieldInfo) -> Result<Option<&[Option<i64>]>>;
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnedNorms(pub BTreeMap<String, Vec<Option<i64>>>);
impl NormsProducer for OwnedNorms {
    fn norms(&self, field: &FieldInfo) -> Result<Option<&[Option<i64>]>> {
        Ok(self.0.get(&field.name).map(Vec::as_slice))
    }
}
pub struct MergeSource<'a> {
    pub reader: Option<&'a dyn FieldsProducer>,
    pub max_doc: i32,
    /// Java DocMap: -1 means deleted; every other entry is the destination ID.
    pub doc_map: &'a [i32],
}
pub struct MergeState<'a> {
    pub sources: &'a [MergeSource<'a>],
    pub check_aborted: &'a dyn Fn() -> Result<()>,
}
pub trait FieldsConsumer {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    /// Java: FieldsConsumer.merge. The mapped view stays lazy until `write`
    /// requests it, so a cranky guard precedes source enumeration failures.
    fn merge(&mut self, state: &MergeState<'_>, norms: Option<&dyn NormsProducer>) -> Result<()> {
        for source in state.sources {
            if let Some(reader) = source.reader {
                (state.check_aborted)()?;
                reader.check_integrity()?;
            }
        }
        self.write(&MappedFields(state), norms)
    }
}
struct MappedFields<'a, 'b>(&'a MergeState<'b>);
impl Fields for MappedFields<'_, '_> {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
        let mut fields: BTreeMap<String, BTreeMap<Vec<u8>, Vec<Posting>>> = BTreeMap::new();
        for source in self.0.sources {
            let Some(reader) = source.reader else {
                continue;
            };
            for field in reader.materialize()?.iter() {
                let terms = fields.entry(field.name.clone()).or_default();
                for term in &field.terms {
                    let postings = terms.entry(term.bytes.clone()).or_default();
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
                        // SENTINEL: MergeState.DocMap returns -1 for a deleted source document.
                        if doc == -1 {
                            continue;
                        }
                        if doc < 0 {
                            return Err(failure("invalid mapped document"));
                        }
                        let mut mapped = posting.clone();
                        mapped.doc = doc;
                        postings.push(mapped);
                    }
                }
            }
        }
        let mut fields: Vec<_> = fields
            .into_iter()
            .map(|(name, terms)| FieldPostings {
                name,
                terms: terms
                    .into_iter()
                    .filter_map(|(bytes, mut postings)| {
                        if postings.is_empty() {
                            return None;
                        }
                        postings.sort_by_key(|posting| posting.doc);
                        Some(Term { bytes, postings })
                    })
                    .collect(),
            })
            .collect();
        sort_field_names(&mut fields);
        Ok(Cow::Owned(fields))
    }
}

#[derive(Clone, Copy)]
pub struct ReadState<'a> {
    pub files: &'a Files,
    pub segment: SegmentReadState<'a>,
}
pub trait PostingsFormat {
    type Consumer<'a>: FieldsConsumer;
    type Producer: FieldsProducer;
    fn name(&self) -> &str;
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>>;
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer>;
}
pub struct CrankyPostingsFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}
impl<F> CrankyPostingsFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: PostingsFormat> PostingsFormat for CrankyPostingsFormat<F> {
    type Consumer<'a> = CrankyFieldsConsumer<F::Consumer<'a>>;
    type Producer = F::Producer;
    fn name(&self) -> &str {
        self.delegate.name()
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        guard(&self.random, FAKE_OPEN_ERROR)?;
        Ok(CrankyFieldsConsumer::new(
            self.delegate.fields_consumer(state)?,
            self.random.clone(),
        ))
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        self.delegate.fields_producer(state)
    }
}
pub struct CrankyFieldsConsumer<W> {
    delegate: W,
    random: Rc<RefCell<JavaRandom>>,
}
impl<W> CrankyFieldsConsumer<W> {
    pub fn new(delegate: W, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: FieldsConsumer> FieldsConsumer for CrankyFieldsConsumer<W> {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        guard(&self.random, FAKE_WRITE_ERROR)?;
        self.delegate.write(fields, norms)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()?;
        guard(&self.random, FAKE_CLOSE_ERROR)
    }
}

/// Extension-keyed immutable file buffers; segment names and routing are explicit.
pub type Files = BTreeMap<String, Vec<u8>>;
/// Existing Lucene104 writer/reader with the default 25/48 term-block geometry.
#[derive(Clone, Copy, Default)]
pub struct Lucene104PostingsFormat;
pub struct Lucene104FieldsConsumer<'a> {
    state: SegmentWriteState<'a>,
    output: Option<Output>,
    closed: bool,
}
impl Lucene104FieldsConsumer<'_> {
    pub fn output(&self) -> Option<&Output> {
        self.output.as_ref()
    }
}
pub struct Lucene104FieldsProducer {
    fields: Option<Vec<FieldPostings>>,
    files: Files,
}
impl Fields for Lucene104FieldsProducer {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
        self.fields
            .as_deref()
            .map(Cow::Borrowed)
            .ok_or_else(|| failure("closed postings reader"))
    }
}
impl FieldsProducer for Lucene104FieldsProducer {
    fn check_integrity(&self) -> Result<()> {
        self.materialize()?;
        for bytes in self.files.values() {
            let footer = bytes
                .len()
                .checked_sub(codec_util::FOOTER_LENGTH)
                .ok_or_else(|| failure("missing postings footer"))?;
            codec_util::check_whole_file_footer(bytes, footer)?;
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.fields = None;
        self.files.clear();
        Ok(())
    }
}
fn file<'a>(files: &'a Files, extension: &str) -> Result<&'a [u8]> {
    files
        .get(extension)
        .map(Vec::as_slice)
        .ok_or_else(|| failure(format!("missing postings file .{extension}")))
}
/// Java: Lucene104PostingsReader constructor's `.psm` metadata and lengths.
fn check_postings_meta(state: ReadState<'_>) -> Result<()> {
    let bytes = file(state.files, "psm")?;
    let mut input = SliceInput::new(bytes);
    codec_util::check_index_header(
        &mut input,
        "Lucene104PostingsWriterMeta",
        0,
        0,
        state.segment.segment_id,
        state.segment.segment_suffix,
    )?;
    for _ in 0..4 {
        input.read_i32()?;
    }
    let mut extensions = vec!["doc"];
    if state
        .segment
        .field_infos
        .fields
        .iter()
        .any(|field| field.index_options.subsumes_positions())
    {
        extensions.push("pos");
        if state
            .segment
            .field_infos
            .fields
            .iter()
            .any(|field| field.store_payloads || field.index_options.subsumes_offsets())
        {
            extensions.push("pay");
        }
    }
    for extension in extensions {
        let length = usize::try_from(input.read_i64()?)
            .map_err(|_| failure("negative postings file length"))?;
        codec_util::retrieve_checksum_with_expected_length(file(state.files, extension)?, length)?;
    }
    codec_util::check_whole_file_footer(bytes, input.position())?;
    Ok(())
}
impl PostingsFormat for Lucene104PostingsFormat {
    type Consumer<'a> = Lucene104FieldsConsumer<'a>;
    type Producer = Lucene104FieldsProducer;
    fn name(&self) -> &str {
        "Lucene104"
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        if state.max_doc < 0 {
            return Err(failure("negative maxDoc"));
        }
        Ok(Lucene104FieldsConsumer {
            state,
            output: None,
            closed: false,
        })
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        check_postings_meta(state)?;
        let segment = state.segment;
        let dictionaries = blocktree::open(
            file(state.files, "tim")?,
            file(state.files, "tip")?,
            file(state.files, "tmd")?,
            segment.field_infos,
            segment.segment_id,
            segment.segment_suffix,
            segment.max_doc,
        )?;
        let doc = DocInput::open(
            file(state.files, "doc")?,
            segment.segment_id,
            segment.segment_suffix,
        )?;
        let prox = segment
            .field_infos
            .fields
            .iter()
            .any(|field| field.index_options.subsumes_positions());
        let pos = if prox {
            Some(PosInput::open(
                file(state.files, "pos")?,
                segment.segment_id,
                segment.segment_suffix,
            )?)
        } else {
            None
        };
        let pay = if prox
            && segment
                .field_infos
                .fields
                .iter()
                .any(|field| field.store_payloads || field.index_options.subsumes_offsets())
        {
            Some(PayInput::open(
                file(state.files, "pay")?,
                segment.segment_id,
                segment.segment_suffix,
            )?)
        } else {
            None
        };
        let mut fields = Vec::new();
        for (name, dictionary) in dictionaries.iter_fields() {
            let mut walk = dictionary.iter();
            let mut terms = Vec::new();
            while let Some(bytes) = walk.try_next_term()? {
                let bytes = bytes.to_vec();
                let (docs, positions) = if dictionary.index_options().subsumes_positions() {
                    walk.try_current_postings_and_positions(
                        Some(&doc),
                        pos.as_ref().expect("segment has positions"),
                        pay.as_ref(),
                    )?
                    .expect("cursor is on a term")
                } else {
                    (
                        walk.try_current_postings(Some(&doc))?
                            .expect("cursor is on a term"),
                        Vec::new(),
                    )
                };
                let postings = docs
                    .docs
                    .into_iter()
                    .zip(docs.freqs)
                    .enumerate()
                    .map(|(index, (doc, freq))| Posting {
                        doc,
                        freq,
                        positions: positions.get(index).cloned().unwrap_or_default(),
                    })
                    .collect();
                terms.push(Term { bytes, postings });
            }
            fields.push(FieldPostings {
                name: name.to_string(),
                terms,
            });
        }
        sort_field_names(&mut fields);
        Ok(Lucene104FieldsProducer {
            fields: Some(fields),
            files: state.files.clone(),
        })
    }
}
impl FieldsConsumer for Lucene104FieldsConsumer<'_> {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        if self.closed {
            return Err(failure("closed postings writer"));
        }
        if self.output.is_some() {
            return Err(failure("postings already written"));
        }
        let fields = fields.materialize()?;
        let mut encoded = Vec::new();
        let mut metadata = Vec::new();
        let mut norm_columns = Vec::new();
        for field in fields.iter() {
            let info = self
                .state
                .field_infos
                .field_by_name(&field.name)
                .ok_or_else(|| failure("unrecognized postings field"))?;
            if info.index_options == IndexOptions::None {
                return Err(failure("field does not index postings"));
            }
            let mut terms = Vec::new();
            for term in &field.terms {
                if term.postings.is_empty() {
                    continue;
                }
                let mut native = TermPostings {
                    term: term.bytes.clone(),
                    ..TermPostings::default()
                };
                for posting in &term.postings {
                    if posting.doc < 0 || posting.doc >= self.state.max_doc {
                        return Err(failure("destination document out of bounds"));
                    }
                    native.docs.push((
                        posting.doc,
                        if info.index_options == IndexOptions::Docs {
                            1
                        } else {
                            posting.freq
                        },
                    ));
                    if info.index_options.subsumes_positions() {
                        native.positions.push(
                            posting
                                .positions
                                .iter()
                                .map(|position| position.position)
                                .collect(),
                        );
                        if info.index_options.subsumes_offsets() {
                            native.offsets.push(
                                posting
                                    .positions
                                    .iter()
                                    .map(|position| (position.start_offset, position.end_offset))
                                    .collect(),
                            )
                        }
                        if info.store_payloads {
                            for position in &posting.positions {
                                native.payload_lengths.push(
                                    u32::try_from(position.payload.len())
                                        .map_err(|_| failure("payload too long"))?,
                                );
                                native.payload_bytes.extend_from_slice(&position.payload);
                            }
                        }
                    }
                }
                terms.push(native);
            }
            if terms.is_empty() {
                continue;
            }
            if !info.omit_norms {
                if let Some(norms) = norms {
                    if let Some(values) = norms.norms(info)? {
                        norm_columns.push((
                            info.number,
                            values
                                .iter()
                                .map(|value| value.unwrap_or(1))
                                .collect::<Vec<_>>(),
                        ));
                    }
                }
            }
            metadata.push(info);
            encoded.push(terms);
        }
        let inputs: Vec<_> = metadata
            .iter()
            .zip(&encoded)
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
        let options = self
            .state
            .postings_options(postings_writer::WriteOptions::default());
        self.output = Some(if inputs.is_empty() {
            postings_writer::write_no_fields(
                &options,
                self.state.segment_id,
                self.state.segment_suffix,
            )?
        } else {
            postings_writer::write_fields_with_options(
                &inputs,
                &norms,
                &options,
                self.state.segment_id,
                self.state.segment_suffix,
            )?
        });
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.output.is_none() {
            let options = self
                .state
                .postings_options(postings_writer::WriteOptions::default());
            self.output = Some(postings_writer::write_no_fields(
                &options,
                self.state.segment_id,
                self.state.segment_suffix,
            )?);
        }
        self.closed = true;
        Ok(())
    }
}
/// Convert existing output buffers into extension-keyed input files.
pub fn output_files(output: &Output) -> Files {
    let mut files = Files::new();
    for (extension, bytes) in [
        ("doc", &output.doc),
        ("psm", &output.psm),
        ("pos", &output.pos),
        ("pay", &output.pay),
        ("tim", &output.tim),
        ("tip", &output.tip),
        ("tmd", &output.tmd),
    ] {
        if !bytes.is_empty() {
            files.insert(extension.to_string(), bytes.clone());
        }
    }
    files
}
#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
    const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    fn infos() -> FieldInfos {
        lucene_codecs::field_infos::parse(
            include_bytes!("../../../../fixtures/testframework/data/cranky_postings.fnm"),
            &ID,
            "",
        )
        .unwrap()
    }
    fn files() -> Files {
        [
            (
                "doc",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.doc")
                    .as_slice(),
            ),
            (
                "psm",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.psm")
                    .as_slice(),
            ),
            (
                "pos",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.pos")
                    .as_slice(),
            ),
            (
                "pay",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.pay")
                    .as_slice(),
            ),
            (
                "tim",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.tim")
                    .as_slice(),
            ),
            (
                "tip",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.tip")
                    .as_slice(),
            ),
            (
                "tmd",
                include_bytes!("../../../../fixtures/testframework/data/cranky_postings.tmd")
                    .as_slice(),
            ),
        ]
        .into_iter()
        .map(|(extension, bytes)| (extension.into(), bytes.to_vec()))
        .collect()
    }
    fn read<'a>(files: &'a Files, infos: &'a FieldInfos) -> ReadState<'a> {
        ReadState {
            files,
            segment: SegmentReadState {
                max_doc: 9,
                segment_id: &ID,
                segment_suffix: "s",
                field_infos: infos,
            },
        }
    }
    fn write(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
        SegmentWriteState {
            max_doc,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: infos,
        }
    }
    fn one(name: &str, doc: i32) -> OwnedFields {
        OwnedFields(vec![FieldPostings {
            name: name.into(),
            terms: vec![Term {
                bytes: b"a".to_vec(),
                postings: vec![Posting {
                    doc,
                    freq: 1,
                    positions: vec![],
                }],
            }],
        }])
    }
    fn reseal(bytes: &mut Vec<u8>) {
        bytes.truncate(bytes.len() - codec_util::FOOTER_LENGTH);
        codec_util::write_footer(bytes);
    }
    #[test]
    fn native_metadata_headers_lengths_missing_files_and_integrity_corruption() {
        let infos = infos();
        let good = files();
        let mut format = Lucene104PostingsFormat;
        for extension in ["psm", "doc", "pos", "pay", "tim", "tip", "tmd"] {
            let mut missing = good.clone();
            missing.remove(extension);
            assert!(
                format.fields_producer(read(&missing, &infos)).is_err(),
                "missing {extension}"
            );
            let mut bad = good.clone();
            bad.get_mut(extension).unwrap()[0] ^= 1;
            assert!(
                format.fields_producer(read(&bad, &infos)).is_err(),
                "header {extension}"
            );
        }
        let at = codec_util::index_header_length("Lucene104PostingsWriterMeta", "s") + 16;
        for length in [-1i64, 1] {
            let mut bad = good.clone();
            let psm = bad.get_mut("psm").unwrap();
            psm[at..at + 8].copy_from_slice(&length.to_le_bytes());
            reseal(psm);
            let error = format.fields_producer(read(&bad, &infos)).err().unwrap();
            if length == -1 {
                assert_eq!(error.to_string(), "negative postings file length");
            }
        }
        let mut bad = good.clone();
        bad.get_mut("psm").unwrap().truncate(at + 2);
        assert!(format.fields_producer(read(&bad, &infos)).is_err());
        let mut wrong = read(&good, &infos);
        let wrong_id = [77; 16];
        wrong.segment.segment_id = &wrong_id;
        assert!(format.fields_producer(wrong).is_err());
        let mut reader = format.fields_producer(read(&good, &infos)).unwrap();
        reader.files.get_mut("doc").unwrap()[0] ^= 1;
        assert!(reader.check_integrity().is_err());
        reader.files.insert("broken".into(), vec![]);
        assert!(reader.check_integrity().is_err());
    }
    #[test]
    fn buffer_lifecycle_empty_docs_only_and_recoverable_validation_errors() {
        let infos = FieldInfos::new(vec![
            FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs)
        ])
        .unwrap();
        let mut format = Lucene104PostingsFormat;
        assert_eq!(
            format
                .fields_consumer(write(&infos, -1))
                .err()
                .unwrap()
                .to_string(),
            "negative maxDoc"
        );
        let mut consumer = format.fields_consumer(write(&infos, 2)).unwrap();
        assert!(consumer.output().is_none());
        assert_eq!(
            consumer
                .write(&one("missing", 0), None)
                .unwrap_err()
                .to_string(),
            "unrecognized postings field"
        );
        for doc in [-1, 2] {
            assert_eq!(
                consumer
                    .write(&one("f", doc), None)
                    .unwrap_err()
                    .to_string(),
                "destination document out of bounds"
            );
        }
        consumer.write(&one("f", 0), None).unwrap();
        assert!(consumer.write(&one("f", 1), None).is_err());
        consumer.close().unwrap();
        consumer.close().unwrap();
        assert!(consumer.write(&one("f", 0), None).is_err());
        let output = output_files(consumer.output().unwrap());
        assert_eq!(output.len(), 5);
        let reader = format.fields_producer(read(&output, &infos)).unwrap();
        assert_eq!(reader.materialize().unwrap()[0].terms[0].postings[0].doc, 0);
        let mut empty = format.fields_consumer(write(&infos, 0)).unwrap();
        empty.close().unwrap();
        let output = output_files(empty.output().unwrap());
        let reader = format.fields_producer(read(&output, &infos)).unwrap();
        assert!(reader.materialize().unwrap().is_empty());
        reader.check_integrity().unwrap();
        let mut empty = format.fields_consumer(write(&infos, 0)).unwrap();
        empty
            .write(
                &OwnedFields(vec![FieldPostings {
                    name: "f".into(),
                    terms: vec![Term {
                        bytes: b"a".to_vec(),
                        postings: vec![],
                    }],
                }]),
                None,
            )
            .unwrap();
        assert!(empty.output().is_some());
        let none = FieldInfos::new(vec![FieldInfo::new("f", 0)]).unwrap();
        let mut consumer = format.fields_consumer(write(&none, 2)).unwrap();
        assert_eq!(
            consumer.write(&one("f", 0), None).unwrap_err().to_string(),
            "field does not index postings"
        );
    }
    struct Producer {
        fields: OwnedFields,
        events: Rc<RefCell<Vec<&'static str>>>,
        integrity_error: bool,
        fields_error: bool,
    }
    impl Fields for Producer {
        fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
            self.events.borrow_mut().push("fields");
            if self.fields_error {
                Err(failure("fields failure"))
            } else {
                self.fields.materialize()
            }
        }
    }
    impl FieldsProducer for Producer {
        fn check_integrity(&self) -> Result<()> {
            self.events.borrow_mut().push("integrity");
            if self.integrity_error {
                Err(failure("integrity failure"))
            } else {
                Ok(())
            }
        }
        fn close(&mut self) -> Result<()> {
            self.events.borrow_mut().push("close");
            Ok(())
        }
    }
    #[test]
    fn superclass_merge_orders_abort_integrity_guard_then_mapping_and_preserves_failures() {
        let infos = FieldInfos::new(vec![
            FieldInfo::new("f", 0).with_index_options(IndexOptions::Docs)
        ])
        .unwrap();
        let events = Rc::new(RefCell::new(vec![]));
        let mut producer = Producer {
            fields: one("f", 0),
            events: events.clone(),
            integrity_error: false,
            fields_error: true,
        };
        let mut format = Lucene104PostingsFormat;
        for fail_abort in [true, false] {
            let sources = [
                MergeSource {
                    reader: None,
                    max_doc: 0,
                    doc_map: &[],
                },
                MergeSource {
                    reader: Some(&producer),
                    max_doc: 1,
                    doc_map: &[0],
                },
            ];
            let check = || {
                events.borrow_mut().push("abort");
                if fail_abort {
                    Err(failure("aborted"))
                } else {
                    Ok(())
                }
            };
            let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
            events.borrow_mut().clear();
            let error = consumer
                .merge(
                    &MergeState {
                        sources: &sources,
                        check_aborted: &check,
                    },
                    None,
                )
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                if fail_abort {
                    "aborted"
                } else {
                    "fields failure"
                }
            );
            assert_eq!(
                *events.borrow(),
                if fail_abort {
                    vec!["abort"]
                } else {
                    vec!["abort", "integrity", "fields"]
                }
            );
            assert!(consumer.output().is_none());
        }
        producer.integrity_error = true;
        let sources = [MergeSource {
            reader: Some(&producer),
            max_doc: 1,
            doc_map: &[0],
        }];
        let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
        assert_eq!(
            consumer
                .merge(
                    &MergeState {
                        sources: &sources,
                        check_aborted: &|| Ok(())
                    },
                    None
                )
                .unwrap_err()
                .to_string(),
            "integrity failure"
        );
        producer.integrity_error = false;
        let failing_seed = (0..)
            .find(|seed| JavaRandom::new(*seed).next_int_bounded(100) == 0)
            .unwrap();
        let native = format.fields_consumer(write(&infos, 1)).unwrap();
        let mut wrapper =
            CrankyFieldsConsumer::new(native, Rc::new(RefCell::new(JavaRandom::new(failing_seed))));
        let sources = [MergeSource {
            reader: Some(&producer),
            max_doc: 1,
            doc_map: &[0],
        }];
        events.borrow_mut().clear();
        assert_eq!(
            wrapper
                .merge(
                    &MergeState {
                        sources: &sources,
                        check_aborted: &|| Ok(())
                    },
                    None
                )
                .unwrap_err()
                .to_string(),
            FAKE_WRITE_ERROR
        );
        assert_eq!(*events.borrow(), vec!["integrity"]);
        producer.fields_error = false;
        for (doc, max_doc, map, message) in [
            (-1, 1, vec![0], "source document out of bounds"),
            (1, 1, vec![0, 1], "source document out of bounds"),
            (0, 1, vec![], "document map out of bounds"),
            (0, 1, vec![-2], "invalid mapped document"),
        ] {
            producer.fields = one("f", doc);
            let sources = [MergeSource {
                reader: Some(&producer),
                max_doc,
                doc_map: &map,
            }];
            let mut consumer = format.fields_consumer(write(&infos, 2)).unwrap();
            assert_eq!(
                consumer
                    .merge(
                        &MergeState {
                            sources: &sources,
                            check_aborted: &|| Ok(())
                        },
                        None
                    )
                    .unwrap_err()
                    .to_string(),
                message
            );
        }
        producer.fields = one("f", 0);
        let sources = [MergeSource {
            reader: Some(&producer),
            max_doc: 1,
            doc_map: &[-1],
        }];
        let mut consumer = format.fields_consumer(write(&infos, 1)).unwrap();
        consumer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(()),
                },
                None,
            )
            .unwrap();
        let result = output_files(consumer.output().unwrap());
        assert!(format
            .fields_producer(read(&result, &infos))
            .unwrap()
            .materialize()
            .unwrap()
            .is_empty());
        producer.close().unwrap();
        assert_eq!(events.borrow().last(), Some(&"close"));
    }
    struct FailingNorms;
    impl NormsProducer for FailingNorms {
        fn norms(&self, _: &FieldInfo) -> Result<Option<&[Option<i64>]>> {
            Err(failure("norms failure"))
        }
    }
    #[test]
    fn norms_callbacks_optional_values_and_full_position_validation() {
        let infos = infos();
        let files = files();
        let mut format = Lucene104PostingsFormat;
        let input = format.fields_producer(read(&files, &infos)).unwrap();
        let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
        assert_eq!(
            consumer
                .write(&input, Some(&FailingNorms))
                .unwrap_err()
                .to_string(),
            "norms failure"
        );
        assert!(consumer.output().is_none());
        let mut norms = OwnedNorms::default();
        assert!(norms.norms(&infos.fields[0]).unwrap().is_none());
        norms.0.insert("rich".into(), vec![None; 9]);
        consumer.write(&input, Some(&norms)).unwrap();
        consumer.close().unwrap();
        let mut invalid = input.materialize().unwrap().into_owned();
        let field = invalid
            .iter_mut()
            .find(|field| field.name == "rich")
            .unwrap();
        field.terms[0].postings[0].positions.clear();
        let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
        assert!(consumer.write(&OwnedFields(invalid), None).is_err());
        let mut consumer = format.fields_consumer(write(&infos, 9)).unwrap();
        consumer
            .write(&input, Some(&OwnedNorms::default()))
            .unwrap();
        assert!(consumer.output().is_some());
    }
}
