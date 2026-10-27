//! A SimpleText segment re-encoded with this port's default (`Lucene104`)
//! formats: what [`crate::simpletext_writer::SimpleTextDirectory`] hands
//! the `IndexWriter` for each segment of an index it appends to, so the
//! writer reads, deletes from, updates and merges SimpleText segments as it
//! does its own.
//!
//! The segment is opened as a [`SegmentReader`] (the SimpleText readers)
//! and every component is decoded through the [`LeafReader`] API, then
//! written with the codec writers a flush uses, under the segment's own
//! name, id and field numbers: stored fields, term vectors, norms, postings
//! (`Lucene104_0`), doc values (`Lucene90_0`), points and vectors
//! (`Lucene99HnswVectorsFormat_0`, with its graph). The base files hold the
//! segment as its flush or merge wrote it; each generation is written as
//! the writer's own would be (`_<seg>_<gen>.liv`, `_<seg>_<gen>.fnm`,
//! `_<seg>_<gen>_Lucene90_0.dvm`/`.dvd`/`.dvs`), at the generations the
//! SimpleText commit names, so the writer's next generation of each is the
//! one Java's would be.
//!
//! Differs: Java's `IndexWriter` reads the SimpleText files themselves.
//! The mirror holds the same documents, values and generations, and is
//! never written back: what the wrapped directory keeps of the segment is
//! its SimpleText files.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, VectorEncoding,
};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput};
use lucene_codecs::vectors::{FieldVectorData, FlatVectorsField};
use lucene_index::field_updates;
use lucene_index::index_writer::{
    per_field_codec_suffix, segment_file_name, DOC_VALUES_FORMAT_NAME, KNN_VECTORS_FORMAT_NAME,
    POSTINGS_FORMAT_NAME,
};
use lucene_index::segment_info::{self, LuceneVersion, SegmentInfo};
use lucene_index::segment_infos::SegmentCommitInfo;
use lucene_store::{DataOutput, Directory};
use lucene_util::version::Version;

use crate::directory_reader::SegmentReader;
use crate::reader::LeafReader;
use crate::simpletext_transcode::{
    base_commit, column_of, has_norms, norms_of, postings_of, AllPoints, Column,
};

/// The codec the mirror's segments name.
pub const NATIVE_CODEC: &str = "Lucene104";

const STORED_FIELDS_MODE: (&str, &str) = ("Lucene90StoredFieldsFormat.mode", "BEST_SPEED");

/// The per-field format attributes a flush gives a field
/// (`PerField*Format.format`/`.suffix`): `(key prefix, format)`.
const POSTINGS_FORMAT: (&str, &str) = ("PerFieldPostingsFormat", POSTINGS_FORMAT_NAME);
const DOC_VALUES_FORMAT: (&str, &str) = ("PerFieldDocValuesFormat", DOC_VALUES_FORMAT_NAME);
const VECTORS_FORMAT: (&str, &str) = ("PerFieldKnnVectorsFormat", KNN_VECTORS_FORMAT_NAME);

fn unwritable(what: &str, e: impl std::fmt::Display) -> crate::Error {
    crate::Error::from(crate::directory_reader::Error::from(
        lucene_store::Error::Corrupted(format!("SimpleText {what} do not re-encode natively: {e}")),
    ))
}

fn lucene_version(v: Version) -> LuceneVersion {
    LuceneVersion {
        major: v.major(),
        minor: v.minor(),
        bugfix: v.bugfix(),
    }
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) -> crate::Result<()> {
    let mut out = dir
        .create_output(name)
        .map_err(crate::directory_reader::Error::from)?;
    out.write_bytes(bytes);
    out.close().map_err(crate::directory_reader::Error::from)?;
    Ok(())
}

fn put_format(field: &mut FieldInfo, (prefix, format): (&str, &str)) {
    for (key, value) in [
        (format!("{prefix}.format"), format),
        (format!("{prefix}.suffix"), "0"),
    ] {
        match field.attributes.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value.to_string(),
            None => field.attributes.push((key, value.to_string())),
        }
    }
}

/// Mirrors `commit` (a SimpleText segment of `real`) into `native` and
/// returns its commit there.
///
/// # Errors
/// What reading the SimpleText segment, a native writer or `native` fails
/// with.
pub fn to_native(
    real: &dyn Directory,
    commit: &SegmentCommitInfo,
    native: &dyn Directory,
) -> crate::Result<SegmentCommitInfo> {
    let name = commit.segment_name.as_str();
    let id = commit.segment_id;
    let postings_suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let dv_suffix = per_field_codec_suffix(DOC_VALUES_FORMAT_NAME);
    let vectors_suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let si = segment_info::parse_simpletext(
        &real
            .open(&format!("{name}.si"))
            .map_err(crate::directory_reader::Error::from)?,
        name,
        &id,
    )
    .map_err(crate::directory_reader::Error::from)?;
    let base = SegmentReader::open_one(real, &base_commit(commit))?;
    let max_doc = si.doc_count;
    let docs = usize::try_from(max_doc).unwrap_or(0);
    let infos = base.field_infos().clone();
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let file = |suffix: &str, ext: &str| segment_file_name(name, suffix, ext);

    // Stored fields, every document.
    let mut stored = Vec::with_capacity(docs);
    for doc in 0..max_doc {
        stored.push(base.stored_document(doc)?.unwrap_or_default());
    }
    let (fdt, fdx, fdm) = lucene_codecs::stored_fields::write_best_speed(&stored, &id, "");
    drop(stored);
    files.extend([
        (file("", "fdt"), fdt),
        (file("", "fdx"), fdx),
        (file("", "fdm"), fdm),
    ]);

    // Term vectors, every document.
    if infos.fields.iter().any(|f| f.store_term_vectors) {
        let mut vectors = Vec::with_capacity(docs);
        for doc in 0..max_doc {
            vectors.push(base.term_vectors(doc)?.unwrap_or_default());
        }
        let (tvd, tvx, tvm) = lucene_codecs::term_vectors::write_best_speed(&vectors, &id, "");
        files.extend([
            (file("", "tvd"), tvd),
            (file("", "tvx"), tvx),
            (file("", "tvm"), tvm),
        ]);
    }

    // Norms, and the impacts the postings take from them.
    let mut sparse_norms: Vec<(i32, Vec<(i32, i64)>)> = Vec::new();
    let mut dense_norms: Vec<(i32, Vec<i64>)> = Vec::new();
    for field in infos.fields.iter().filter(|f| has_norms(f)) {
        if let Some((sparse, dense)) = norms_of(&base, field, docs)? {
            sparse_norms.push((field.number, sparse));
            dense_norms.push((field.number, dense));
        }
    }
    if !sparse_norms.is_empty() {
        let fields: Vec<lucene_codecs::norms::NormsField<'_>> = sparse_norms
            .iter()
            .map(|(number, values)| lucene_codecs::norms::NormsField::Sparse(*number, values))
            .collect();
        let (nvm, nvd) = lucene_codecs::norms::write_fields(&fields, max_doc, &id, "")
            .map_err(|e| unwritable("norms", e))?;
        files.extend([(file("", "nvm"), nvm), (file("", "nvd"), nvd)]);
    }

    // Postings, by field name.
    let postings = postings_of(&base, &infos)?;
    let postings_fields: Vec<i32> = postings.iter().map(|(f, _, _)| f.number).collect();
    if !postings.is_empty() {
        let inputs: Vec<FieldPostingsInput<'_>> = postings
            .iter()
            .map(|(field, doc_count, terms)| FieldPostingsInput {
                field_number: field.number,
                index_options: field.index_options,
                doc_count: *doc_count,
                has_payloads: field.store_payloads,
                terms,
            })
            .collect();
        let impacts: Vec<FieldNorms<'_>> = dense_norms
            .iter()
            .map(|(number, values)| FieldNorms {
                field_number: *number,
                values,
            })
            .collect();
        let out = lucene_codecs::postings_writer::write_fields_with_norms(
            &inputs,
            &impacts,
            &id,
            &postings_suffix,
        )
        .map_err(|e| unwritable("postings", e))?;
        for (ext, bytes) in [
            ("doc", out.doc),
            ("psm", out.psm),
            ("pos", out.pos),
            ("pay", out.pay),
            ("tim", out.tim),
            ("tip", out.tip),
            ("tmd", out.tmd),
        ] {
            if !bytes.is_empty() {
                files.push((file(&postings_suffix, ext), bytes));
            }
        }
    }
    drop(postings);

    // Doc values, as the segment's flush or merge wrote them.
    let dv_fields: Vec<&FieldInfo> = infos
        .fields
        .iter()
        .filter(|f| f.doc_values_type != DocValuesType::None)
        .collect();
    if let Some(dv) = doc_values_files(&base, &dv_fields, max_doc, &id, &dv_suffix)? {
        let (dvm, dvd, dvs) = dv;
        files.extend([
            (file(&dv_suffix, "dvm"), dvm),
            (file(&dv_suffix, "dvd"), dvd),
            (file(&dv_suffix, "dvs"), dvs),
        ]);
    }

    // Points.
    let mut points = Vec::new();
    for field in infos.fields.iter().filter(|f| f.point_dimension_count > 0) {
        let Some(values) = base.point_values(&field.name)? else {
            continue;
        };
        let mut all = AllPoints::default();
        values.intersect(&mut all)?;
        let mut packed = lucene_codecs::points::PackedPointsField {
            field_number: field.number,
            num_dims: field.point_dimension_count,
            num_index_dims: field.point_index_dimension_count,
            bytes_per_dim: field.point_num_bytes,
            ..Default::default()
        };
        for (value, doc) in all.points {
            packed.docs.push(doc);
            packed.values.extend_from_slice(&value);
        }
        points.push(packed);
    }
    if !points.is_empty() {
        let (kdm, kdi, kdd) = lucene_codecs::points::write_packed(
            points,
            lucene_codecs::points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            max_doc,
            &id,
            "",
        )
        .map_err(|e| unwritable("points", e))?;
        files.extend([
            (file("", "kdm"), kdm),
            (file("", "kdi"), kdi),
            (file("", "kdd"), kdd),
        ]);
    }

    // Vectors, with the graph a flush builds.
    let mut vectors = Vec::new();
    for field in infos.fields.iter().filter(|f| f.vector_dimension > 0) {
        let (docs, values) = match field.vector_encoding {
            VectorEncoding::Float32 => {
                let Some(values) = base.float_vector_values(&field.name)? else {
                    continue;
                };
                let (mut docs, mut flat) = (Vec::new(), Vec::new());
                for ord in 0..values.size() {
                    docs.push(values.ord_to_doc(ord)?);
                    flat.extend_from_slice(&values.vector_value(ord)?);
                }
                (docs, FieldVectorData::Float32(flat))
            }
            VectorEncoding::Byte => {
                let Some(values) = base.byte_vector_values(&field.name)? else {
                    continue;
                };
                let (mut docs, mut flat) = (Vec::new(), Vec::new());
                for ord in 0..values.size() {
                    docs.push(values.ord_to_doc(ord)?);
                    flat.extend_from_slice(&values.vector_value(ord)?);
                }
                (docs, FieldVectorData::Byte(flat))
            }
        };
        vectors.push(FlatVectorsField {
            field_number: field.number,
            similarity: field.vector_similarity_function,
            dimension: field.vector_dimension,
            docs,
            values,
        });
    }
    let vector_fields: Vec<i32> = vectors.iter().map(|f| f.field_number).collect();
    if !vectors.is_empty() {
        for (ext, bytes) in vector_files(&vectors, max_doc, &id)? {
            files.push((file(&vectors_suffix, ext), bytes));
        }
    }

    // Field infos, with the per-field formats a flush records.
    let mut fields = infos.fields.clone();
    for field in &mut fields {
        field.doc_values_gen = -1;
        if postings_fields.contains(&field.number) {
            put_format(field, POSTINGS_FORMAT);
        }
        if field.doc_values_type != DocValuesType::None {
            put_format(field, DOC_VALUES_FORMAT);
        }
        if vector_fields.contains(&field.number) {
            put_format(field, VECTORS_FORMAT);
        }
    }
    files.push((
        file("", "fnm"),
        lucene_codecs::field_infos::write(&fields, &id, ""),
    ));

    let si_name = format!("{name}.si");
    let mut listed: Vec<String> = files.iter().map(|(f, _)| f.clone()).collect();
    listed.push(si_name.clone());
    let mut attributes = si.attributes.clone();
    attributes.retain(|(k, _)| k != STORED_FIELDS_MODE.0);
    attributes.push((STORED_FIELDS_MODE.0.into(), STORED_FIELDS_MODE.1.into()));
    let native_si = SegmentInfo {
        id,
        version: lucene_version(si.version),
        min_version: si.min_version.map(lucene_version),
        doc_count: max_doc,
        is_compound_file: false,
        has_blocks: si.has_blocks,
        diagnostics: si.diagnostics.clone(),
        files: listed,
        attributes,
        index_sort: (!si.index_sort.is_empty()).then(|| si.index_sort.clone()),
    };
    files.push((si_name, segment_info::write(&native_si, "")));
    drop(base);

    let mut mirrored = SegmentCommitInfo {
        codec_name: NATIVE_CODEC.to_string(),
        field_infos_files: Vec::new(),
        dv_update_files: Vec::new(),
        ..commit.clone()
    };
    if commit.del_gen > 0 || commit.field_infos_gen > 0 {
        let current = SegmentReader::open_one(real, commit)?;
        if commit.del_gen > 0 {
            let bits = current.live_docs().ok_or_else(|| {
                unwritable(
                    "live docs",
                    format!("{name} has no live docs at {}", commit.del_gen),
                )
            })?;
            let del_count = usize::try_from(commit.del_count).unwrap_or(0);
            let liv = lucene_codecs::live_docs::write(bits, &id, commit.del_gen, del_count)
                .map_err(|e| unwritable("live docs", e))?;
            files.push((
                lucene_index::deletes::liv_file_name(name, commit.del_gen),
                liv,
            ));
        }
        if commit.field_infos_gen > 0 {
            let updated = generation_infos(current.field_infos(), &fields);
            let mut generations: std::collections::BTreeMap<i64, Vec<&FieldInfo>> =
                std::collections::BTreeMap::new();
            for field in updated
                .iter()
                .filter(|f| f.doc_values_gen > 0 && f.doc_values_type != DocValuesType::None)
            {
                generations
                    .entry(field.doc_values_gen)
                    .or_default()
                    .push(field);
            }
            for (generation, in_gen) in generations {
                let suffix = field_updates::generation_segment_suffix(generation, &dv_suffix);
                let Some((dvm, dvd, dvs)) =
                    doc_values_files(&current, &in_gen, max_doc, &id, &suffix)?
                else {
                    continue;
                };
                let mut names = Vec::new();
                for (ext, bytes) in [("dvm", dvm), ("dvd", dvd), ("dvs", dvs)] {
                    let file_name =
                        field_updates::generation_file_name(name, generation, &dv_suffix, ext);
                    names.push(file_name.clone());
                    files.push((file_name, bytes));
                }
                for field in in_gen {
                    mirrored.set_doc_values_updates_files(field.number, names.clone());
                }
            }
            mirrored.dv_update_files.sort_by_key(|(number, _)| *number);
            let fnm = field_updates::field_infos_gen_file_name(name, commit.field_infos_gen);
            files.push((
                fnm.clone(),
                lucene_codecs::field_infos::write(
                    &updated,
                    &id,
                    &lucene_util::base36::to_base36(commit.field_infos_gen),
                ),
            ));
            mirrored.field_infos_files = vec![fnm];
        }
    }

    for (file_name, bytes) in &files {
        write_file(native, file_name, bytes)?;
    }
    let names: Vec<String> = files.into_iter().map(|(f, _)| f).collect();
    native
        .sync(&names)
        .map_err(crate::directory_reader::Error::from)?;
    Ok(mirrored)
}

/// The current field infos with the per-field formats `base` (the mirror's
/// base `.fnm`) records, and the doc-values format on a field an update
/// gave doc values.
fn generation_infos(current: &FieldInfos, base: &[FieldInfo]) -> Vec<FieldInfo> {
    current
        .fields
        .iter()
        .map(|f| {
            let mut field = f.clone();
            if let Some(native) = base.iter().find(|b| b.name == f.name) {
                field.attributes = native.attributes.clone();
            }
            if field.doc_values_type != DocValuesType::None {
                put_format(&mut field, DOC_VALUES_FORMAT);
            }
            field
        })
        .collect()
}

/// `.dvm`, `.dvd` and `.dvs`.
type DocValuesBytes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// `fields`' doc values from `reader` as one `Lucene90DocValuesFormat`
/// instance's `.dvm`/`.dvd`/`.dvs`; `None` without fields.
fn doc_values_files(
    reader: &SegmentReader,
    fields: &[&FieldInfo],
    max_doc: i32,
    id: &[u8; lucene_store::codec_util::ID_LENGTH],
    suffix: &str,
) -> crate::Result<Option<DocValuesBytes>> {
    use lucene_codecs::doc_values::DenseField;
    let mut columns = Vec::new();
    let mut skip = Vec::new();
    for field in fields {
        let Some(column) = column_of(reader, field)? else {
            continue;
        };
        if field.doc_values_skip_index_type != DocValuesSkipIndexType::None
            && field.doc_values_type != DocValuesType::Binary
        {
            skip.push(field.number);
        }
        columns.push((field.number, column));
    }
    if columns.is_empty() {
        return Ok(None);
    }
    let dense: Vec<DenseField<'_>> = columns
        .iter()
        .map(|(number, column)| match column {
            Column::Numeric(values) => DenseField::SparseNumeric(*number, values),
            Column::Binary(values) => DenseField::BinaryColumn(*number, values),
            Column::Sorted(values) => DenseField::SortedOrds(*number, values),
            Column::SortedNumeric(values) => DenseField::SortedNumericColumn(*number, values),
            Column::SortedSet(values) => DenseField::SortedSetOrds(*number, values),
        })
        .collect();
    let files = lucene_codecs::doc_values::write_fields_with_skip_indexes(
        &dense, &skip, max_doc, id, suffix,
    )
    .map_err(|e| unwritable("doc values", e))?;
    Ok(Some(files))
}

/// `Lucene99HnswVectorsFormat`'s files for `fields`: the flat `.vec`/`.vemf`
/// and the `.vex`/`.vem` graph a flush builds over them.
fn vector_files(
    fields: &[FlatVectorsField],
    max_doc: i32,
    id: &[u8; lucene_store::codec_util::ID_LENGTH],
) -> crate::Result<Vec<(&'static str, Vec<u8>)>> {
    use lucene_codecs::hnsw;
    let vectors_suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let (vec, vemf) =
        lucene_codecs::vectors::write_flat_vectors(fields, max_doc, id, &vectors_suffix)
            .map_err(|e| unwritable("vectors", e))?;
    let flat = lucene_codecs::vectors::FlatVectorsReader::open(&vemf, &vec, id, &vectors_suffix)
        .map_err(|e| unwritable("vectors", e))?;
    let (m, beam_width) = (hnsw::DEFAULT_MAX_CONN, hnsw::DEFAULT_BEAM_WIDTH);
    let mut graphs = Vec::with_capacity(fields.len());
    for field in fields {
        let count = i32::try_from(field.docs.len()).map_err(|e| unwritable("vectors", e))?;
        if !hnsw::should_create_graph(hnsw::HNSW_GRAPH_THRESHOLD, count) {
            graphs.push(None);
            continue;
        }
        let built = match field.values.encoding() {
            VectorEncoding::Float32 => {
                let values = flat
                    .float_vector_values(field.field_number)
                    .map_err(|e| unwritable("vectors", e))?;
                hnsw::HnswGraphBuilder::new(
                    values.ord_scorer(),
                    m,
                    beam_width,
                    hnsw::DEFAULT_RAND_SEED,
                )
                .and_then(|b| b.build(count))
            }
            VectorEncoding::Byte => {
                let values = flat
                    .byte_vector_values(field.field_number)
                    .map_err(|e| unwritable("vectors", e))?;
                hnsw::HnswGraphBuilder::new(
                    values.ord_scorer(),
                    m,
                    beam_width,
                    hnsw::DEFAULT_RAND_SEED,
                )
                .and_then(|b| b.build(count))
            }
        };
        graphs.push(Some(built.map_err(|e| unwritable("vectors", e))?));
    }
    let hnsw_fields: Vec<lucene_codecs::hnsw_vectors::HnswVectorsField<'_>> = fields
        .iter()
        .zip(&graphs)
        .map(
            |(field, graph)| lucene_codecs::hnsw_vectors::HnswVectorsField {
                field_number: field.field_number,
                encoding: field.values.encoding(),
                similarity: field.similarity,
                dimension: field.dimension,
                count: i32::try_from(field.docs.len()).unwrap_or(i32::MAX),
                graph: graph.as_ref(),
                m,
            },
        )
        .collect();
    let (vex, vem) =
        lucene_codecs::hnsw_vectors::write_hnsw_vectors(&hnsw_fields, id, &vectors_suffix)
            .map_err(|e| unwritable("vectors", e))?;
    Ok(vec![
        ("vec", vec),
        ("vemf", vemf),
        ("vex", vex),
        ("vem", vem),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_format_attribute_is_replaced_not_repeated() {
        let mut field = FieldInfo::new("f", 0);
        field.attributes = vec![("PerFieldDocValuesFormat.format".into(), "Other".into())];
        put_format(&mut field, DOC_VALUES_FORMAT);
        put_format(&mut field, DOC_VALUES_FORMAT);
        assert_eq!(
            field.attributes,
            vec![
                (
                    "PerFieldDocValuesFormat.format".to_string(),
                    "Lucene90".to_string()
                ),
                (
                    "PerFieldDocValuesFormat.suffix".to_string(),
                    "0".to_string()
                ),
            ]
        );
    }
}
