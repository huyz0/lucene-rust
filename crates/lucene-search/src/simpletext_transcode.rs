//! A segment re-encoded with `SimpleTextCodec`'s formats: what
//! `IndexWriter` with `iwc.setCodec(new SimpleTextCodec())` writes for the
//! same documents, built from the segment as this port's IndexWriter wrote
//! it (`Lucene104`).
//!
//! The segment is opened as a [`SegmentReader`] and every component is
//! decoded through the [`LeafReader`] API (the reader's own files: deleted
//! documents included, as a flush or merge writes them), then handed to the
//! `lucene-codecs` SimpleText writers in the order Java's flush and merge
//! write them: fields by number (postings by name), documents ascending.
//!
//! Field and segment attributes that only the `Lucene104` formats record
//! (`PerFieldPostingsFormat.*`, `PerFieldDocValuesFormat.*`,
//! `PerFieldKnnVectorsFormat.*`, `Lucene90StoredFieldsFormat.mode`) are left
//! out, as SimpleText's formats never write them.

use lucene_codecs::doc_values::{
    BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::points::Relation;
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_codecs::simpletext as text;
use lucene_index::segment_info::{self, LuceneVersion, SegmentInfo, SimpleTextSegmentInfo};
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_store::Directory;
use lucene_util::version::Version;

use crate::directory_reader::SegmentReader;
use crate::reader::{IntersectVisitor, LeafReader, PostingsFlags, NO_MORE_DOCS};

/// The field attributes only `Lucene104`'s per-field formats write.
const PER_FIELD_PREFIXES: [&str; 3] = [
    "PerFieldPostingsFormat.",
    "PerFieldDocValuesFormat.",
    "PerFieldKnnVectorsFormat.",
];

/// The segment attribute only `Lucene90StoredFieldsFormat` writes.
const STORED_FIELDS_MODE: &str = "Lucene90StoredFieldsFormat.mode";

/// A segment's SimpleText files.
#[derive(Debug)]
pub struct SimpleTextSegment {
    /// The `.si`'s model: `files` lists every file below (the `.scf` alone
    /// for a compound segment) and the `.si` itself.
    pub info: SimpleTextSegmentInfo,
    /// The `.si`'s bytes.
    pub si: Vec<u8>,
    /// Every other file, by name: the `.scf` for a compound segment.
    pub files: Vec<(String, Vec<u8>)>,
    /// The live docs as `_<seg>_<gen>.liv`, when the commit has deletions.
    pub live_docs: Option<(String, Vec<u8>)>,
}

pub(crate) fn text_error(e: text::Error) -> crate::Error {
    crate::directory_reader::Error::SimpleText(e).into()
}

pub(crate) fn version(v: LuceneVersion) -> crate::Result<Version> {
    Version::new(v.major, v.minor, v.bugfix, 0)
        .map_err(|e| crate::Error::IllegalArgument(format!("segment version: {e}")))
}

/// SimpleText's field infos: the native ones without the per-field formats'
/// attributes.
fn simpletext_field_infos(native: &FieldInfos) -> FieldInfos {
    FieldInfos {
        fields: native
            .fields
            .iter()
            .map(|f| {
                let mut f = f.clone();
                f.attributes
                    .retain(|(k, _)| !PER_FIELD_PREFIXES.iter().any(|p| k.starts_with(p)));
                f
            })
            .collect(),
    }
}

/// `segment` (a `Lucene104` segment of `dir`) as SimpleText: its `.si`
/// says `info`'s version, diagnostics and sort; `commit` gives the live
/// docs' generation.
///
/// # Errors
/// What reading the segment or a SimpleText writer fails with.
pub fn to_simpletext(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    info: &SegmentInfo,
) -> crate::Result<SimpleTextSegment> {
    let reader = SegmentReader::open_one(dir, commit)?;
    let name = &commit.segment_name;
    let max_doc = info.doc_count;
    let docs = usize::try_from(max_doc).unwrap_or(0);
    let infos = simpletext_field_infos(reader.field_infos());
    let state = SegmentWriteState {
        max_doc,
        segment_id: &commit.segment_id,
        segment_suffix: "",
        field_infos: &infos,
    };
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();

    // Java's SegmentMerger/flush order: field infos last on disk, but the
    // file set is unordered; this keeps the merge's order of writes.
    // Stored fields: every document, deleted ones included.
    let mut stored = Vec::with_capacity(docs);
    for doc in 0..max_doc {
        stored.push(reader.stored_document(doc)?.unwrap_or_default());
    }
    files.push((
        text::stored_fields_writer::stored_fields_file_name(name),
        text::stored_fields_writer::write(&stored, &infos).map_err(text_error)?,
    ));
    drop(stored);

    // Norms, by field number.
    let mut norms: Vec<(i32, Vec<i64>)> = Vec::new();
    let mut norms_consumer = text::norms::SimpleTextNormsConsumer::new(max_doc);
    for field in infos.fields.iter().filter(|f| has_norms(f)) {
        let Some((sparse, dense)) = norms_of(&reader, field, docs)? else {
            continue;
        };
        norms_consumer
            .add_norms_field(field, &sparse)
            .map_err(text_error)?;
        norms.push((field.number, dense));
    }
    let any_norms = !norms.is_empty();

    // Postings, by field name.
    let postings = postings_of(&reader, &infos)?;
    if !postings.is_empty() {
        let inputs: Vec<FieldPostingsInput<'_, TermPostings>> = postings
            .iter()
            .map(|(field, doc_count, terms)| FieldPostingsInput {
                field_number: field.number,
                index_options: field.index_options,
                doc_count: *doc_count,
                has_payloads: field.store_payloads,
                terms,
            })
            .collect();
        let field_norms: Vec<FieldNorms<'_>> = norms
            .iter()
            .map(|(number, values)| FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        files.push((
            text::fields_writer::postings_file_name(name, ""),
            text::fields_writer::write(&inputs, &field_norms, &state).map_err(text_error)?,
        ));
    }
    drop(postings);
    if any_norms {
        files.push((
            text::norms::norms_file_name(name, ""),
            norms_consumer.finish().map_err(text_error)?,
        ));
    }

    // Doc values, in `IndexingChain`'s field-hash order for a flush, by
    // field number for a merge.
    let mut dv = text::doc_values_writer::SimpleTextDocValuesWriter::new(max_doc);
    let mut any_dv = false;
    let walk = flush_order(info, &infos);
    for field in walk
        .iter()
        .copied()
        .filter(|f| f.doc_values_type != DocValuesType::None)
    {
        add_doc_values(&reader, field, &mut dv)?;
        any_dv = true;
    }
    if any_dv {
        files.push((
            text::doc_values_writer::doc_values_file_name(
                name,
                "",
                text::doc_values_writer::DOC_VALUES_EXTENSION,
            ),
            dv.finish().map_err(text_error)?,
        ));
    }

    // Term vectors: every document, as `addAllDocVectors` copies them.
    if infos.fields.iter().any(|f| f.store_term_vectors) {
        let mut vectors = Vec::with_capacity(docs);
        for doc in 0..max_doc {
            let mut document = reader.term_vectors(doc)?.unwrap_or_default();
            document.fields.sort_by(|a, b| {
                let name = |n: i32| infos.field_by_number(n).map(|f| f.name.as_str());
                name(a.field_number).cmp(&name(b.field_number))
            });
            vectors.push(document);
        }
        files.push((
            text::term_vectors_writer::term_vectors_file_name(name),
            text::term_vectors_writer::write(&vectors, &infos).map_err(text_error)?,
        ));
    }

    // Points, in the doc values' order (`IndexingChain.writePoints`).
    let point_fields: Vec<&FieldInfo> = walk
        .iter()
        .copied()
        .filter(|f| f.point_dimension_count > 0)
        .collect();
    if !point_fields.is_empty() {
        let mut writer = text::points_writer::SimpleTextPointsWriter::new(name, docs, None);
        for field in point_fields {
            let Some(values) = reader.point_values(&field.name)? else {
                continue;
            };
            let mut all = AllPoints::default();
            values.intersect(&mut all)?;
            let size = u64::try_from(values.size()).unwrap_or(0);
            writer
                .write_field(
                    field,
                    size,
                    all.points.iter().map(|(v, d)| (v.as_slice(), *d)),
                )
                .map_err(text_error)?;
        }
        let (dim, dii) = writer.finish();
        files.push((text::points_writer::points_file_name(name, ""), dim));
        files.push((text::points_writer::points_index_file_name(name, ""), dii));
    }

    // Vectors, by field number.
    let vector_fields: Vec<&FieldInfo> = infos
        .fields
        .iter()
        .filter(|f| f.vector_dimension > 0)
        .collect();
    if !vector_fields.is_empty() {
        let mut writer = text::knn_vectors_writer::SimpleTextKnnVectorsWriter::new();
        for field in vector_fields {
            if let Some(values) = reader.float_vector_values(&field.name)? {
                let mut vectors = Vec::new();
                for ord in 0..values.size() {
                    vectors.push((values.ord_to_doc(ord)?, values.vector_value(ord)?));
                }
                writer
                    .write_float_field(field, vectors.iter().map(|(d, v)| (*d, v.as_slice())))
                    .map_err(text_error)?;
            } else if let Some(values) = reader.byte_vector_values(&field.name)? {
                let mut vectors = Vec::new();
                for ord in 0..values.size() {
                    vectors.push((values.ord_to_doc(ord)?, values.vector_value(ord)?));
                }
                writer
                    .write_byte_field(field, vectors.iter().map(|(d, v)| (*d, v.as_slice())))
                    .map_err(text_error)?;
            }
        }
        let (vec, gri) = writer.finish();
        files.push((
            text::knn_vectors_writer::vector_data_file_name(name, ""),
            vec,
        ));
        files.push((text::knn_vectors_writer::meta_file_name(name, ""), gri));
    }

    files.push((
        text::field_infos::field_infos_file_name(name, ""),
        text::field_infos::write(&infos.fields),
    ));

    let live_docs = match reader.live_docs() {
        Some(bits) if commit.del_gen > 0 => {
            text::live_docs::live_docs_file_name(name, commit.del_gen)
                .map(|file| (file, text::live_docs::write(bits)))
        }
        _ => None,
    };

    let files = if info.is_compound_file {
        let scf = text::compound::write(&files);
        vec![(text::compound::compound_file_name(name), scf)]
    } else {
        files
    };
    let mut model = SimpleTextSegmentInfo {
        name: name.clone(),
        version: version(info.version)?,
        min_version: info.min_version.map(version).transpose()?,
        doc_count: max_doc,
        is_compound_file: info.is_compound_file,
        has_blocks: info.has_blocks,
        diagnostics: info.diagnostics.clone(),
        attributes: info
            .attributes
            .iter()
            .filter(|(k, _)| k != STORED_FIELDS_MODE)
            .cloned()
            .collect(),
        files: files.iter().map(|(f, _)| f.clone()).collect(),
        id: commit.segment_id,
        index_sort: info.index_sort.clone().unwrap_or_default(),
    };
    let si =
        segment_info::write_simpletext(&mut model).map_err(crate::directory_reader::Error::from)?;
    Ok(SimpleTextSegment {
        info: model,
        si,
        files,
        live_docs,
    })
}

/// The order a segment's flush or merge hands fields to the doc-values and
/// points writers: `IndexingChain`'s field-hash walk for a flushed segment
/// (its fields first seen in field-number order), field-number order for a
/// merge.
fn flush_order<'a>(info: &SegmentInfo, infos: &'a FieldInfos) -> Vec<&'a FieldInfo> {
    let mut by_number: Vec<&FieldInfo> = infos.fields.iter().collect();
    by_number.sort_by_key(|f| f.number);
    let flushed = info
        .diagnostics
        .iter()
        .any(|(k, v)| k == "source" && v == "flush");
    if !flushed {
        return by_number;
    }
    let names: Vec<&str> = by_number.iter().map(|f| f.name.as_str()).collect();
    lucene_index::indexing_chain::field_hash_order(&names)
        .into_iter()
        .map(|i| by_number[i])
        .collect()
}

/// `commit` without its generations: the segment as its flush or merge
/// wrote it, which [`to_simpletext`] turns into the SimpleText segment's
/// base files; [`generation_files`] adds the rest.
pub fn base_commit(commit: &SegmentCommitInfo) -> SegmentCommitInfo {
    SegmentCommitInfo {
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        field_infos_files: Vec::new(),
        dv_update_files: Vec::new(),
        ..commit.clone()
    }
}

/// The SimpleText files a commit's generations add to its segment's base
/// files, as Java's `SimpleTextLiveDocsFormat`, `SimpleTextFieldInfosFormat`
/// and `SimpleTextDocValuesFormat` write them for a deletion, field-infos or
/// doc-values update generation (`_<seg>_<gen>.liv`, `.inf`, `.dat`).
#[derive(Debug, Default)]
pub struct GenerationFiles {
    /// Every file, with its bytes unless the caller had it already.
    pub files: Vec<(String, Option<Vec<u8>>)>,
    /// `SegmentCommitInfo.getFieldInfosFiles()`.
    pub field_infos_files: Vec<String>,
    /// `SegmentCommitInfo.getDocValuesUpdatesFiles()`, by field number.
    pub dv_update_files: Vec<(i32, Vec<String>)>,
}

/// [`GenerationFiles`] of `commit` (a `Lucene104` segment of `dir`); `known`
/// names the files already written, whose bytes are not built again.
///
/// A doc-values generation's `.dat` holds the fields whose latest update is
/// that generation, with their current values: the fields Java's
/// `ReadersAndUpdates.writeFieldUpdates` wrote into it that no later
/// generation superseded.
///
/// # Errors
/// What reading the segment or a SimpleText writer fails with.
pub fn generation_files(
    dir: &dyn Directory,
    commit: &SegmentCommitInfo,
    known: &dyn Fn(&str) -> bool,
) -> crate::Result<GenerationFiles> {
    let mut out = GenerationFiles::default();
    if commit.del_gen <= 0 && commit.field_infos_gen <= 0 {
        return Ok(out);
    }
    let reader = SegmentReader::open_one(dir, commit)?;
    let name = &commit.segment_name;
    if let Some(file) = text::live_docs::live_docs_file_name(name, commit.del_gen) {
        let bytes = if known(&file) {
            None
        } else {
            let bits = reader.live_docs().ok_or_else(|| {
                crate::Error::IllegalArgument(format!("{name}: no live docs at {file}"))
            })?;
            Some(text::live_docs::write(bits))
        };
        out.files.push((file, bytes));
    }
    if commit.field_infos_gen > 0 {
        let infos = simpletext_field_infos(reader.field_infos());
        let file = text::field_infos::field_infos_file_name(
            name,
            &lucene_util::base36::to_base36(commit.field_infos_gen),
        );
        let bytes = (!known(&file)).then(|| text::field_infos::write(&infos.fields));
        out.field_infos_files.push(file.clone());
        out.files.push((file, bytes));

        let mut generations: std::collections::BTreeMap<i64, Vec<&FieldInfo>> =
            std::collections::BTreeMap::new();
        for field in infos
            .fields
            .iter()
            .filter(|f| f.doc_values_gen > 0 && f.doc_values_type != DocValuesType::None)
        {
            generations
                .entry(field.doc_values_gen)
                .or_default()
                .push(field);
        }
        for (generation, fields) in generations {
            let file = text::doc_values_writer::doc_values_file_name(
                name,
                &lucene_util::base36::to_base36(generation),
                text::doc_values_writer::DOC_VALUES_EXTENSION,
            );
            out.dv_update_files
                .extend(fields.iter().map(|f| (f.number, vec![file.clone()])));
            let bytes = if known(&file) {
                None
            } else {
                let mut dv =
                    text::doc_values_writer::SimpleTextDocValuesWriter::new(reader.max_doc);
                for field in &fields {
                    add_doc_values(&reader, field, &mut dv)?;
                }
                Some(dv.finish().map_err(text_error)?)
            };
            out.files.push((file, bytes));
        }
        out.dv_update_files.sort_by_key(|(number, _)| *number);
    }
    Ok(out)
}

/// A field's norms by document, and as a column of every document.
pub(crate) type FieldNormValues = (Vec<(i32, i64)>, Vec<i64>);

/// One field's norms: the documents that have one, and a column of every
/// document's (`1` where missing); `None` when the field has none.
pub(crate) fn norms_of(
    reader: &SegmentReader,
    field: &FieldInfo,
    docs: usize,
) -> crate::Result<Option<FieldNormValues>> {
    let Some(mut values) = reader.norm_values(&field.name)? else {
        return Ok(None);
    };
    let mut sparse = Vec::new();
    let mut dense = vec![1i64; docs];
    loop {
        let doc = values.next_doc()?;
        if doc == NO_MORE_DOCS {
            break;
        }
        let value = values.long_value();
        sparse.push((doc, value));
        if let Some(slot) = usize::try_from(doc).ok().and_then(|d| dense.get_mut(d)) {
            *slot = value;
        }
    }
    Ok(Some((sparse, dense)))
}

/// Every indexed field's postings, by field name: each field with its
/// `docCount` and its terms' documents, frequencies, positions, offsets and
/// payloads.
pub(crate) fn postings_of<'i>(
    reader: &SegmentReader,
    infos: &'i FieldInfos,
) -> crate::Result<Vec<(&'i FieldInfo, i32, Vec<TermPostings>)>> {
    let mut indexed: Vec<&FieldInfo> = infos
        .fields
        .iter()
        .filter(|f| f.index_options != IndexOptions::None)
        .collect();
    indexed.sort_by(|a, b| a.name.cmp(&b.name));
    let mut postings = Vec::new();
    for field in &indexed {
        let Some(terms) = reader.terms(&field.name)? else {
            continue;
        };
        let doc_count = terms.doc_count();
        let flags = if field.index_options.subsumes_offsets() || field.store_payloads {
            PostingsFlags::All
        } else if field.index_options.subsumes_positions() {
            PostingsFlags::Positions
        } else {
            PostingsFlags::Freqs
        };
        let positions = field.index_options.subsumes_positions();
        let offsets = field.index_options.subsumes_offsets();
        let mut out = Vec::new();
        let mut terms_enum = terms.iterator()?;
        while let Some(term) = terms_enum.next()? {
            let mut posting = TermPostings {
                term: term.to_vec(),
                ..TermPostings::default()
            };
            let mut p = terms_enum.postings(flags)?;
            loop {
                let doc = p.next_doc()?;
                if doc == NO_MORE_DOCS {
                    break;
                }
                let freq = p.freq();
                posting.docs.push((doc, freq));
                if positions {
                    let mut ps = Vec::new();
                    let mut os = Vec::new();
                    for _ in 0..freq {
                        ps.push(p.next_position()?);
                        if offsets {
                            os.push((p.start_offset(), p.end_offset()));
                        }
                        if field.store_payloads {
                            let payload = p.payload().unwrap_or_default();
                            posting
                                .payload_lengths
                                .push(u32::try_from(payload.len()).unwrap_or(u32::MAX));
                            posting.payload_bytes.extend_from_slice(payload);
                        }
                    }
                    posting.positions.push(ps);
                    if offsets {
                        posting.offsets.push(os);
                    }
                }
            }
            out.push(posting);
        }
        postings.push((*field, doc_count, out));
    }
    Ok(postings)
}

pub(crate) fn has_norms(field: &FieldInfo) -> bool {
    field.index_options != IndexOptions::None && !field.omit_norms
}

/// Every point of a field with its document, in `intersect` order.
#[derive(Default)]
pub(crate) struct AllPoints {
    pub(crate) points: Vec<(Vec<u8>, i32)>,
}

impl IntersectVisitor for AllPoints {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {}
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.points.push((packed_value.to_vec(), doc_id));
    }
}

/// One field's doc values, read through the [`LeafReader`] API.
pub(crate) enum Column {
    Numeric(Vec<(i32, i64)>),
    Binary(BinaryColumn),
    Sorted(SortedOrdsColumn),
    SortedNumeric(SortedNumericColumn),
    SortedSet(SortedSetOrdsColumn),
}

/// `field`'s doc values (empty when the reader has none for it); `None` for
/// a field without doc values.
pub(crate) fn column_of(
    reader: &SegmentReader,
    field: &FieldInfo,
) -> crate::Result<Option<Column>> {
    Ok(Some(match field.doc_values_type {
        DocValuesType::None => return Ok(None),
        DocValuesType::Numeric => {
            let mut values = Vec::new();
            if let Some(mut it) = reader.numeric_doc_values(&field.name)? {
                while it.next_doc()? != NO_MORE_DOCS {
                    values.push((it.doc_id(), it.long_value()));
                }
            }
            Column::Numeric(values)
        }
        DocValuesType::Binary => {
            let mut column = BinaryColumn::default();
            if let Some(mut it) = reader.binary_doc_values(&field.name)? {
                while it.next_doc()? != NO_MORE_DOCS {
                    let value = it.binary_value();
                    column.docs.push(it.doc_id());
                    column
                        .lengths
                        .push(u32::try_from(value.len()).unwrap_or(u32::MAX));
                    column.bytes.extend_from_slice(value);
                }
            }
            Column::Binary(column)
        }
        DocValuesType::Sorted => {
            let mut column = SortedOrdsColumn::default();
            if let Some(mut it) = reader.sorted_doc_values(&field.name)? {
                for ord in 0..it.value_count() {
                    column.dict.push(it.lookup_ord(ord)?);
                }
                while it.next_doc()? != NO_MORE_DOCS {
                    column.docs.push(it.doc_id());
                    column.ords.push(i64::from(it.ord_value()));
                }
            }
            Column::Sorted(column)
        }
        DocValuesType::SortedNumeric => {
            let mut column = SortedNumericColumn::default();
            if let Some(mut it) = reader.sorted_numeric_doc_values(&field.name)? {
                while it.next_doc()? != NO_MORE_DOCS {
                    let count = it.doc_value_count();
                    column.docs.push(it.doc_id());
                    column.counts.push(u32::try_from(count).unwrap_or(0));
                    for _ in 0..count {
                        column.values.push(it.next_value()?);
                    }
                }
            }
            Column::SortedNumeric(column)
        }
        DocValuesType::SortedSet => {
            let mut column = SortedSetOrdsColumn::default();
            if let Some(mut it) = reader.sorted_set_doc_values(&field.name)? {
                for ord in 0..it.value_count() {
                    column.dict.push(it.lookup_ord(ord)?);
                }
                while it.next_doc()? != NO_MORE_DOCS {
                    let count = it.doc_value_count();
                    column.docs.push(it.doc_id());
                    column.counts.push(u32::try_from(count).unwrap_or(0));
                    for _ in 0..count {
                        column.ords.push(it.next_ord()?);
                    }
                }
            }
            Column::SortedSet(column)
        }
    }))
}

/// One field's doc values into the SimpleText writer.
fn add_doc_values(
    reader: &SegmentReader,
    field: &FieldInfo,
    dv: &mut text::doc_values_writer::SimpleTextDocValuesWriter,
) -> crate::Result<()> {
    use text::doc_values_writer::DocValuesInput;
    let Some(column) = column_of(reader, field)? else {
        return Ok(());
    };
    let values = match &column {
        Column::Numeric(values) => DocValuesInput::Numeric(values),
        Column::Binary(column) => DocValuesInput::Binary(column),
        Column::Sorted(column) => DocValuesInput::Sorted(column),
        Column::SortedNumeric(column) => DocValuesInput::SortedNumeric(column),
        Column::SortedSet(column) => DocValuesInput::SortedSet(column),
    };
    dv.add_field(field, values).map_err(text_error)
}
