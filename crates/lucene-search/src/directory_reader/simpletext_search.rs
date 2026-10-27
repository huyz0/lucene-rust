//! The binary forms a SimpleText segment is searched through.
//!
//! Java's SimpleText readers implement `Terms`/`PostingsEnum`, `PointValues`
//! and `KnnVectorsReader.search` themselves, and every query runs on them.
//! This port's term and phrase legs have native text cursors
//! (`exec::build::simpletext`); the rest of the scorer tree, the span and
//! interval weights, the point queries and the KNN searches run on the binary
//! codec structures of [`crate::multi_segment::OpenSegment`]. A SimpleText
//! segment hands those the forms here: its text postings (every frequency,
//! position, offset and payload, impacts from its text norms), points and
//! vectors re-encoded with the default codec's writers on first use and kept
//! with the segment core, as [`crate::memory`] encodes its one document.
//!
//! Differs: Java walks the text files; this walks their binary encoding,
//! which holds the same documents and values, so hits and scores agree
//! (statistics and norms still come from the text files). A KNN search over
//! the flat vectors with no graph is the exhaustive scan
//! `SimpleTextKnnVectorsReader.search` is. Each form is built once per core:
//! a delete-only reopen shares it, and a malformed text file fails the first
//! search that needs it rather than the open.

use std::sync::OnceLock;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::field_infos::{FieldInfos, IndexOptions, VectorEncoding};
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::simpletext as text;
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};
use lucene_store::codec_util::ID_LENGTH;

use super::simpletext::SimpleTextCore;
use super::Result;

/// The suffix the forms are written and opened with.
const SUFFIX: &str = "";

/// [`SimpleTextCore`]'s binary forms, each built on first use.
#[derive(Debug, Default)]
pub(crate) struct SearchForms {
    postings: OnceLock<Postings>,
    points: OnceLock<Option<PointsFiles>>,
    vectors: OnceLock<Option<VectorFiles>>,
    doc_values: OnceLock<Option<DocValuesFiles>>,
}

/// The text postings as a block tree and `.doc`/`.pos`/`.pay`.
#[derive(Debug)]
pub(crate) struct Postings {
    pub(crate) fields: BlockTreeFields,
    pub(crate) doc: Option<Vec<u8>>,
    pub(crate) pos: Option<Vec<u8>>,
    pub(crate) pay: Option<Vec<u8>>,
}

/// The text points as `.kdm`/`.kdi`/`.kdd`.
#[derive(Debug)]
pub(crate) struct PointsFiles {
    pub(crate) kdm: Vec<u8>,
    pub(crate) kdi: Vec<u8>,
    pub(crate) kdd: Vec<u8>,
}

/// The text doc values (the current generation of each field) as
/// `.dvm` (parsed), `.dvd` and the skip indexes' `.dvs`.
#[derive(Debug)]
pub(crate) struct DocValuesFiles {
    pub(crate) meta: lucene_codecs::doc_values::DocValuesMeta,
    pub(crate) data: Vec<u8>,
    pub(crate) skip: Vec<u8>,
}

/// `.kdm`, `.kdi` and `.kdd`.
pub(super) type PointsBytes<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// The text vectors as `.vemf`/`.vec`.
#[derive(Debug)]
pub(crate) struct VectorFiles {
    pub(crate) meta: Vec<u8>,
    pub(crate) data: Vec<u8>,
}

/// A writer refusing what the text readers decoded: the text files hold
/// something no Lucene index can.
fn unencodable(what: &str, e: impl std::fmt::Display) -> super::Error {
    lucene_store::Error::Corrupted(format!("SimpleText {what} do not re-encode: {e}")).into()
}

impl SearchForms {
    /// The postings form, once built.
    pub(crate) fn built_postings(&self) -> Option<&Postings> {
        self.postings.get()
    }

    /// The doc-values form, once built (`None` inside when the segment has
    /// no doc values).
    pub(crate) fn built_doc_values(&self) -> Option<&DocValuesFiles> {
        self.doc_values.get().and_then(Option::as_ref)
    }

    /// The points form, once built (`None` inside when the segment has no
    /// points).
    pub(crate) fn built_points(&self) -> Option<&PointsFiles> {
        self.points.get().and_then(Option::as_ref)
    }
}

impl SimpleTextCore {
    /// The postings form, built on first call.
    ///
    /// # Errors
    /// A text postings file that does not decode.
    pub(crate) fn search_postings(&self, segment_id: &[u8; ID_LENGTH]) -> Result<&Postings> {
        if let Some(built) = self.search.postings.get() {
            return Ok(built);
        }
        let built = self.build_postings(segment_id)?;
        // A racing build encoded the same postings; either copy is the answer.
        Ok(self.search.postings.get_or_init(|| built))
    }

    /// The points form, built on first call; `None` without points.
    ///
    /// # Errors
    /// A text points file that does not decode.
    pub(crate) fn search_points(
        &self,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<&PointsFiles>> {
        if let Some(built) = self.search.points.get() {
            return Ok(built.as_ref());
        }
        let built = self.build_points(segment_id)?;
        Ok(self.search.points.get_or_init(|| built).as_ref())
    }

    /// The doc-values form, built on first call; `None` without doc values.
    ///
    /// # Errors
    /// A text doc-values file that does not decode.
    pub(crate) fn search_doc_values(
        &self,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<&DocValuesFiles>> {
        if let Some(built) = self.search.doc_values.get() {
            return Ok(built.as_ref());
        }
        let built = self.build_doc_values(segment_id)?;
        Ok(self.search.doc_values.get_or_init(|| built).as_ref())
    }

    /// The vectors form, built on first call; `None` without vectors.
    ///
    /// # Errors
    /// A text vectors file that does not decode.
    pub(crate) fn search_vectors(
        &self,
        segment_id: &[u8; ID_LENGTH],
    ) -> Result<Option<&VectorFiles>> {
        if let Some(built) = self.search.vectors.get() {
            return Ok(built.as_ref());
        }
        let built = self.build_vectors(segment_id)?;
        Ok(self.search.vectors.get_or_init(|| built).as_ref())
    }

    // Java: SimpleTextFieldsReader's Terms/PostingsEnum, read as
    // FreqProxTermsWriter hands a flush's postings to the postings format.
    fn build_postings(&self, segment_id: &[u8; ID_LENGTH]) -> Result<Postings> {
        let infos: &FieldInfos = &self.field_infos;
        let mut fields: Vec<(i32, IndexOptions, bool, i32, Vec<TermPostings>)> = Vec::new();
        let mut norms: Vec<(i32, Vec<i64>)> = Vec::new();
        if let Some(reader) = &self.postings {
            for name in reader.field_names() {
                let Some(terms) = reader.terms(name)? else {
                    continue;
                };
                let info = terms.info();
                let options = info.index_options;
                let offsets = terms.has_offsets();
                let payloads = terms.has_payloads();
                let flags = match (offsets, payloads) {
                    (true, true) => text::ALL,
                    (true, false) => text::OFFSETS,
                    (false, true) => text::PAYLOADS,
                    (false, false) => text::POSITIONS,
                };
                let mut out = Vec::new();
                let mut cursor = terms.iterator();
                while let Some(term) = cursor.next()? {
                    let mut posting = TermPostings {
                        term: term.to_vec(),
                        ..TermPostings::default()
                    };
                    let mut docs = cursor.postings(flags)?;
                    loop {
                        let doc = docs.next_doc()?;
                        if doc == text::NO_MORE_DOCS {
                            break;
                        }
                        let freq = if options == IndexOptions::Docs {
                            1
                        } else {
                            docs.freq()
                        };
                        posting.docs.push((doc, freq));
                        if !options.subsumes_positions() {
                            continue;
                        }
                        let mut positions = Vec::new();
                        let mut starts = Vec::new();
                        for _ in 0..freq {
                            positions.push(docs.next_position()?);
                            if offsets {
                                starts.push((docs.start_offset(), docs.end_offset()));
                            }
                            if payloads {
                                let payload = docs.payload().unwrap_or_default();
                                posting.payload_bytes.extend_from_slice(payload);
                                posting.payload_lengths.push(
                                    u32::try_from(payload.len())
                                        .map_err(|e| unencodable("payloads", e))?,
                                );
                            }
                        }
                        posting.positions.push(positions);
                        if offsets {
                            posting.offsets.push(starts);
                        }
                    }
                    out.push(posting);
                }
                fields.push((info.number, options, payloads, terms.doc_count(), out));
                if let Some((values, data)) = self.norms(name)? {
                    let mut column = vec![1i64; usize::try_from(self.info.doc_count).unwrap_or(0)];
                    let mut cursor = values.numeric_from_data(name, data)?;
                    loop {
                        let doc = cursor.next_doc()?;
                        if doc == text::NO_MORE_DOCS {
                            break;
                        }
                        let value = cursor.long_value()?;
                        if let Some(slot) =
                            usize::try_from(doc).ok().and_then(|d| column.get_mut(d))
                        {
                            *slot = value;
                        }
                    }
                    norms.push((info.number, column));
                }
            }
        }
        let inputs: Vec<FieldPostingsInput<'_>> = fields
            .iter()
            .map(
                |(number, options, payloads, doc_count, terms)| FieldPostingsInput {
                    field_number: *number,
                    index_options: *options,
                    doc_count: *doc_count,
                    has_payloads: *payloads,
                    terms,
                },
            )
            .collect();
        let impacts: Vec<postings_writer::FieldNorms<'_>> = norms
            .iter()
            .map(|(number, values)| postings_writer::FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        let output =
            postings_writer::write_fields_with_norms(&inputs, &impacts, segment_id, SUFFIX)
                .map_err(|e| unencodable("postings", e))?;
        let fields = lucene_codecs::blocktree::open_shared(
            std::sync::Arc::new(output.tim),
            std::sync::Arc::new(output.tip),
            &output.tmd,
            infos,
            segment_id,
            SUFFIX,
            self.info.doc_count,
        )?;
        let present = |bytes: Vec<u8>| (!bytes.is_empty()).then_some(bytes);
        Ok(Postings {
            fields,
            doc: present(output.doc),
            pos: present(output.pos),
            pay: present(output.pay),
        })
    }

    // Java: SimpleTextBKDReader's PointValues, every point visited.
    fn build_points(&self, segment_id: &[u8; ID_LENGTH]) -> Result<Option<PointsFiles>> {
        let mut fields = Vec::new();
        for info in &self.field_infos.fields {
            if info.point_dimension_count == 0 {
                continue;
            }
            let Some(tree) = self.points(&info.name)? else {
                continue;
            };
            let mut all = AllPoints::default();
            tree.intersect(&mut all)?;
            let dims = |n: usize| i32::try_from(n).map_err(|e| unencodable("points", e));
            fields.push(lucene_codecs::points::PackedPointsField {
                field_number: info.number,
                num_dims: dims(tree.num_dimensions())?,
                num_index_dims: dims(tree.num_index_dimensions())?,
                bytes_per_dim: dims(tree.bytes_per_dimension())?,
                docs: all.docs,
                values: all.values,
            });
        }
        if fields.is_empty() {
            return Ok(None);
        }
        let (kdm, kdi, kdd) = lucene_codecs::points::write_packed(
            fields,
            lucene_codecs::points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            self.info.doc_count,
            segment_id,
            SUFFIX,
        )
        .map_err(|e| unencodable("points", e))?;
        Ok(Some(PointsFiles { kdm, kdi, kdd }))
    }

    // Java: SimpleTextDocValuesReader's five iterators, each field from the
    // generation its FieldInfo names.
    fn build_doc_values(&self, segment_id: &[u8; ID_LENGTH]) -> Result<Option<DocValuesFiles>> {
        use lucene_codecs::doc_values::{
            BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
        };
        use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType};
        enum Column {
            Numeric(Vec<(i32, i64)>),
            Binary(BinaryColumn),
            Sorted(SortedOrdsColumn),
            SortedNumeric(SortedNumericColumn),
            SortedSet(SortedSetOrdsColumn),
        }
        let count = |n: usize| u32::try_from(n).map_err(|e| unencodable("doc values", e));
        let mut columns = Vec::new();
        let mut skip_fields = Vec::new();
        for info in &self.field_infos.fields {
            let ty = info.doc_values_type;
            let Some((reader, data)) = self.doc_values(&info.name, ty)? else {
                continue;
            };
            let name = info.name.as_str();
            let column = match ty {
                DocValuesType::None => continue,
                DocValuesType::Numeric => {
                    let mut values = reader.numeric_from_data(name, data)?;
                    let mut out = Vec::new();
                    while values.next_doc()? != text::NO_MORE_DOCS {
                        out.push((values.doc_id(), values.long_value()?));
                    }
                    Column::Numeric(out)
                }
                DocValuesType::Binary => {
                    let mut values = reader.binary_from_data(name, data)?;
                    let mut out = BinaryColumn::default();
                    while values.next_doc()? != text::NO_MORE_DOCS {
                        let value = values.binary_value()?;
                        out.docs.push(values.doc_id());
                        out.lengths.push(count(value.len())?);
                        out.bytes.extend_from_slice(&value);
                    }
                    Column::Binary(out)
                }
                DocValuesType::Sorted => {
                    let mut values = reader.sorted_from_data(name, data)?;
                    let mut out = SortedOrdsColumn::default();
                    for ord in 0..values.value_count() {
                        out.dict.push(values.lookup_ord(ord)?.to_vec());
                    }
                    while values.next_doc()? != text::NO_MORE_DOCS {
                        out.docs.push(values.doc_id());
                        out.ords.push(i64::from(values.ord_value()));
                    }
                    Column::Sorted(out)
                }
                DocValuesType::SortedNumeric => {
                    let mut values = reader.sorted_numeric_from_data(name, data)?;
                    let mut out = SortedNumericColumn::default();
                    while values.next_doc()? != text::NO_MORE_DOCS {
                        let n = values.doc_value_count();
                        out.docs.push(values.doc_id());
                        out.counts.push(count(n)?);
                        for _ in 0..n {
                            out.values.push(values.next_value()?);
                        }
                    }
                    Column::SortedNumeric(out)
                }
                DocValuesType::SortedSet => {
                    let mut values = reader.sorted_set_from_data(name, data)?;
                    let mut out = SortedSetOrdsColumn::default();
                    for ord in 0..values.value_count() {
                        out.dict.push(values.lookup_ord(ord)?.to_vec());
                    }
                    while values.next_doc()? != text::NO_MORE_DOCS {
                        let n = values.doc_value_count();
                        out.docs.push(values.doc_id());
                        out.counts.push(count(n)?);
                        for _ in 0..n {
                            out.ords.push(values.next_ord()?);
                        }
                    }
                    Column::SortedSet(out)
                }
            };
            if info.doc_values_skip_index_type != DocValuesSkipIndexType::None
                && ty != DocValuesType::Binary
            {
                skip_fields.push(info.number);
            }
            columns.push((info.number, column));
        }
        if columns.is_empty() {
            return Ok(None);
        }
        use lucene_codecs::doc_values::DenseField;
        let fields: Vec<DenseField<'_>> = columns
            .iter()
            .map(|(number, column)| match column {
                Column::Numeric(values) => DenseField::SparseNumeric(*number, values),
                Column::Binary(values) => DenseField::BinaryColumn(*number, values),
                Column::Sorted(values) => DenseField::SortedOrds(*number, values),
                Column::SortedNumeric(values) => DenseField::SortedNumericColumn(*number, values),
                Column::SortedSet(values) => DenseField::SortedSetOrds(*number, values),
            })
            .collect();
        let (dvm, data, skip) = lucene_codecs::doc_values::write_fields_with_skip_indexes(
            &fields,
            &skip_fields,
            self.info.doc_count,
            segment_id,
            SUFFIX,
        )
        .map_err(|e| unencodable("doc values", e))?;
        let (_, meta) =
            lucene_codecs::doc_values::parse_meta(&dvm, segment_id, SUFFIX, &self.field_infos)?;
        Ok(Some(DocValuesFiles { meta, data, skip }))
    }

    // Java: SimpleTextKnnVectorsReader's FloatVectorValues/ByteVectorValues.
    fn build_vectors(&self, segment_id: &[u8; ID_LENGTH]) -> Result<Option<VectorFiles>> {
        let mut fields = Vec::new();
        for info in &self.field_infos.fields {
            if info.vector_dimension == 0 {
                continue;
            }
            let (docs, values) = match info.vector_encoding {
                VectorEncoding::Float32 => {
                    let Some(values) = self.vector_values::<f32>(&info.name)? else {
                        continue;
                    };
                    let (docs, flat) = flatten(&values)?;
                    (docs, FieldVectorData::Float32(flat))
                }
                VectorEncoding::Byte => {
                    let Some(values) = self.vector_values::<u8>(&info.name)? else {
                        continue;
                    };
                    let (docs, flat) = flatten(&values)?;
                    (docs, FieldVectorData::Byte(flat))
                }
            };
            fields.push(FlatVectorsField {
                field_number: info.number,
                similarity: info.vector_similarity_function,
                dimension: info.vector_dimension,
                docs,
                values,
            });
        }
        if fields.is_empty() {
            return Ok(None);
        }
        let (data, meta) = lucene_codecs::vectors::write_flat_vectors(
            &fields,
            self.info.doc_count,
            segment_id,
            SUFFIX,
        )
        .map_err(|e| unencodable("vectors", e))?;
        Ok(Some(VectorFiles { meta, data }))
    }
}

/// A field's vectors in ord order: each ord's document, and the components
/// one after another.
fn flatten<T: text::knn_vectors_reader::VectorElement>(
    values: &text::knn_vectors_reader::SimpleTextVectorValues<'_, T>,
) -> Result<(Vec<i32>, Vec<T>)> {
    let mut docs = Vec::with_capacity(values.size());
    let mut flat = Vec::new();
    for ord in 0..values.size() {
        let (Some(doc), Some(vector)) = (values.ord_to_doc(ord), values.vector_value(ord)) else {
            return Err(unencodable("vectors", format!("ord {ord} is missing")));
        };
        docs.push(doc);
        flat.extend_from_slice(vector);
    }
    Ok((docs, flat))
}

/// Every point of a tree, in its leaf order.
#[derive(Default)]
struct AllPoints {
    docs: Vec<i32>,
    values: Vec<u8>,
}

impl lucene_codecs::points::IntersectVisitor for AllPoints {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> lucene_codecs::points::Relation {
        // Crossing every cell walks to each leaf and reads its values.
        lucene_codecs::points::Relation::CellCrossesQuery
    }

    fn visit(&mut self, _doc_id: i32) {
        unreachable!("a crossing walk visits values")
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.docs.push(doc_id);
        self.values.extend_from_slice(packed_value);
    }
}

impl super::SegmentReader {
    /// The term dictionary the scorer tree reads: a SimpleText segment's
    /// postings form once [`DirectoryReader::open_segments`] or
    /// [`Self::with_open_segment`] has built it, else this segment's own.
    ///
    /// [`DirectoryReader::open_segments`]: super::DirectoryReader::open_segments
    pub(super) fn search_fields(&self) -> &BlockTreeFields {
        self.simpletext
            .as_ref()
            .and_then(|core| core.search.built_postings())
            .map_or(&self.fields, |forms| &forms.fields)
    }

    /// The `.kdm`/`.kdi`/`.kdd` a points query reads, a SimpleText segment's
    /// built on this call; `None` when the segment indexes no points.
    ///
    /// # Errors
    /// A SimpleText points file that does not decode.
    pub(super) fn search_points_files(&self) -> Result<Option<PointsBytes<'_>>> {
        match &self.simpletext {
            Some(core) => Ok(core
                .search_points(&self.segment_id)?
                .map(|files| (&*files.kdm, &*files.kdi, &*files.kdd))),
            None => Ok(self.points_files()),
        }
    }

    /// A SimpleText segment's vectors as a flat vectors reader, built on
    /// this call; `None` when it has none.
    ///
    /// # Errors
    /// A SimpleText vectors file that does not decode.
    pub(super) fn simpletext_flat_vectors(
        &self,
    ) -> crate::Result<Option<lucene_codecs::vectors::FlatVectorsReader<'_>>> {
        let Some(core) = &self.simpletext else {
            return Ok(None);
        };
        let Some(files) = core.search_vectors(&self.segment_id)? else {
            return Ok(None);
        };
        Ok(Some(lucene_codecs::vectors::FlatVectorsReader::open(
            &files.meta,
            &files.data,
            &self.segment_id,
            SUFFIX,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use crate::directory_reader::DirectoryReader;
    use crate::reader::segment::SegmentTerms;
    use crate::reader::{LeafReader, PostingsFlags, Terms, NO_MORE_DOCS};
    use lucene_codecs::postings::{DocInput, PayInput, PosInput};
    use lucene_store::FsDirectory;

    fn fixture(variant: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/codecs_simpletext/codec")
            .join(variant)
            .join("index")
    }

    /// Every field's postings form holds what the text files do: statistics,
    /// terms, documents, frequencies, positions, offsets and payloads
    /// (the native text reader is checked against Java's dump by
    /// `simpletext_terms_stats_seeks_and_all_postings_match_java_through_leaf_reader`).
    #[test]
    fn postings_forms_hold_every_text_posting() {
        let mut compared = 0;
        for variant in ["flush", "merged"] {
            let reader = DirectoryReader::open(&FsDirectory::open(fixture(variant))).unwrap();
            let opened = reader.open_segments().unwrap();
            for (segment, open) in reader
                .segment_readers()
                .iter()
                .zip(opened.as_open_segments())
            {
                assert!(
                    segment.points_files().is_none(),
                    "points are built on demand"
                );
                let core = segment.simpletext.as_deref().unwrap();
                let forms = core.search.built_postings().unwrap();
                for info in &segment.field_infos().fields {
                    let Some(text) = segment.terms(&info.name).unwrap() else {
                        assert!(open.fields.field(&info.name).is_none());
                        continue;
                    };
                    let binary = SegmentTerms {
                        field: open.fields.field(&info.name).unwrap(),
                        doc_in: forms.doc.as_deref().map(DocInput::validated),
                        pos_in: forms.pos.as_deref().map(PosInput::validated),
                        pay_in: forms.pay.as_deref().map(PayInput::validated),
                    };
                    let stats = |t: &dyn Terms| {
                        (
                            t.size(),
                            t.sum_total_term_freq(),
                            t.sum_doc_freq(),
                            t.doc_count(),
                            t.has_freqs(),
                            t.has_positions(),
                            t.has_offsets(),
                            t.has_payloads(),
                        )
                    };
                    assert_eq!(stats(&*text), stats(&binary), "{}", info.name);
                    let (mut want, mut got) =
                        (text.iterator().unwrap(), binary.iterator().unwrap());
                    while let Some(term) = want.next().unwrap().map(<[u8]>::to_vec) {
                        assert_eq!(got.next().unwrap(), Some(&term[..]));
                        assert_eq!(want.doc_freq().unwrap(), got.doc_freq().unwrap());
                        assert_eq!(
                            want.total_term_freq().unwrap(),
                            got.total_term_freq().unwrap()
                        );
                        let mut a = want.postings(PostingsFlags::All).unwrap();
                        let mut b = got.postings(PostingsFlags::All).unwrap();
                        loop {
                            let doc = a.next_doc().unwrap();
                            assert_eq!(b.next_doc().unwrap(), doc);
                            if doc == NO_MORE_DOCS {
                                break;
                            }
                            assert_eq!(a.freq(), b.freq());
                            if !text.has_positions() {
                                continue;
                            }
                            for _ in 0..a.freq() {
                                assert_eq!(a.next_position().unwrap(), b.next_position().unwrap());
                                assert_eq!(a.start_offset(), b.start_offset());
                                assert_eq!(a.end_offset(), b.end_offset());
                                assert_eq!(
                                    a.payload().filter(|p| !p.is_empty()),
                                    b.payload().filter(|p| !p.is_empty())
                                );
                            }
                        }
                        compared += 1;
                    }
                    assert!(got.next().unwrap().is_none());
                }
            }
        }
        assert!(compared > 500, "{compared} terms");
    }
}
