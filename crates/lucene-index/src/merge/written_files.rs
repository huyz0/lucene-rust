use super::*;

/// Rewrites the merged field list so the merged `.fnm` describes the files
/// the merge **actually wrote**, the way `IndexWriter`'s
/// `fields_with_per_field_attributes` already does at flush time.
///
/// A merged `FieldInfo` is seeded from a source segment's own `.fnm` (see
/// [`reconcile_field_numbers`]), which describes *that* segment's files. Five
/// things in it can be wrong for the merged segment, and every one of them is
/// silent or fatal in real Lucene rather than merely cosmetic:
///
/// - **`PerFieldPostingsFormat.format`/`.suffix`, the `PerFieldDocValues`
///   pair, and the `PerFieldKnnVectors` pair.** Real Lucene routes a field to
///   a postings/doc-values/vectors format purely through these attributes.
///   Without them a merged segment's `.doc`/`.tim`/`.tip`/`.tmd` are never
///   opened at all: `MultiTerms.getTerms(reader, field)` returns `null` and
///   the field silently reads back as having no terms, with no error anywhere
///   -- the exact silent-failure shape `VerifyFullSegment` was written to
///   catch at flush time. Conversely, an attribute inherited from a source for
///   data this merge did *not* write sends a reader looking for a file that
///   does not exist.
/// - **`doc_values_type`.** A `.fnm` claiming a type the merged `.dvm` has no
///   entry for makes `PerFieldDocValuesFormat.FieldsReader` register no
///   producer, so `getNumeric` returns `null` and real
///   `CheckIndex.testDocValues` -- which iterates every field whose `.fnm`
///   claims doc values -- dereferences it.
/// - **`vector_dimension`.** Same shape: a positive dimension makes
///   `FieldInfo.hasVectorValues()` true, which `IncrementalHnswGraphMerger`
///   and `CheckIndex` key off, while `PerFieldKnnVectorsFormat` registers no
///   reader without the attribute.
/// - **`omit_norms`.** `DirectoryReader.open` throws on a missing `.nvm`
///   rather than degrading, so an indexed field whose merged `.fnm` still
///   claims norms the merge did not write makes the whole index unopenable.
/// - **`store_term_vectors`.** Same shape, for a segment with no `.tvd` at
///   all.
///
/// Nothing here invents data. But every one of these rewrites **tolerates a
/// loss** rather than refusing it, and that is a deliberate, uncomfortable
/// choice worth stating plainly: this function's job is to describe files, and
/// a merge that produced fewer of them than its sources had is already a fact
/// by the time it runs. Clearing the claim keeps the merged segment openable
/// and honest; it does not put the data back.
///
/// The guard against that ever mattering is **on the caller side**, and it is
/// the one thing to preserve when adding a format here:
/// [`crate::index_writer::IndexWriter::execute_merge`] opens every format its
/// own flush can write (stored fields, postings, term vectors, norms, all five
/// doc-values types, vectors and points), and
/// [`crate::index_writer::IndexWriter::segment_stats`] withholds from the
/// merge policy the one case it cannot round-trip (a doc-values *generation*).
/// A format opened by neither would be dropped here silently.
///
pub(super) fn describe_written_files(
    merged_fields: &mut [FieldInfo],
    postings_suffixes: &[(i32, &'static str, u32)],
    doc_values_suffixes: &[(i32, &'static str, u32)],
    wrote_term_vectors: bool,
    vector_fields: &[(i32, Option<(&'static str, u32)>)],
    _points_field_numbers: &[i32],
) {
    for f in merged_fields.iter_mut() {
        f.attributes.retain(|(key, _)| {
            key != "PerFieldPostingsFormat.format"
                && key != "PerFieldPostingsFormat.suffix"
                && key != "PerFieldDocValuesFormat.format"
                && key != "PerFieldDocValuesFormat.suffix"
                && key != "PerFieldKnnVectorsFormat.format"
                && key != "PerFieldKnnVectorsFormat.suffix"
        });
        if let Some((_, format_name, suffix)) =
            postings_suffixes.iter().find(|(n, _, _)| *n == f.number)
        {
            f.attributes.push((
                "PerFieldPostingsFormat.format".to_string(),
                format_name.to_string(),
            ));
            f.attributes.push((
                "PerFieldPostingsFormat.suffix".to_string(),
                suffix.to_string(),
            ));
        }
        if let Some((_, format_name, suffix)) =
            doc_values_suffixes.iter().find(|(n, _, _)| *n == f.number)
        {
            f.attributes.push((
                "PerFieldDocValuesFormat.format".to_string(),
                format_name.to_string(),
            ));
            f.attributes.push((
                "PerFieldDocValuesFormat.suffix".to_string(),
                suffix.to_string(),
            ));
        } else {
            // A `.fnm` claiming a `DocValuesType` the merged `.dvm` has no
            // entry for makes `PerFieldDocValuesFormat.FieldsReader` register
            // no producer, so `getNumeric` returns `null` and real
            // `CheckIndex.testDocValues` -- which iterates every field whose
            // `.fnm` claims doc values -- dereferences it. Same rule
            // `IndexWriter::fields_with_per_field_attributes` applies at
            // flush time.
            f.doc_values_type = lucene_codecs::field_infos::DocValuesType::None;
            f.doc_values_skip_index_type = lucene_codecs::field_infos::DocValuesSkipIndexType::None;
        }
        if !wrote_term_vectors {
            f.store_term_vectors = false;
        }
        if let Some((_, routing)) = vector_fields.iter().find(|(n, _)| *n == f.number) {
            if let Some((format, suffix)) = routing {
                f.attributes.push((
                    per_field_knn_vectors::PER_FIELD_FORMAT_KEY.to_string(),
                    format.to_string(),
                ));
                f.attributes.push((
                    per_field_knn_vectors::PER_FIELD_SUFFIX_KEY.to_string(),
                    suffix.to_string(),
                ));
            }
        } else {
            // The vectors twin of the doc-values rule above, for a field the
            // vectors merge wrote no entry for. That is every vector field
            // when no source supplies `MergeSource::vectors` (a merged
            // segment with no vectors and an honest `.fnm`); once any source
            // does, every vector field of the merged FieldInfos gets an
            // entry, empty when its sources supplied none (as
            // `KnnVectorsWriter.merge` writes it). `IndexWriter::execute_merge` always opens
            // them, so the loss is not reachable from this port's own writer.
            // A positive `vector_dimension` makes `FieldInfo.hasVectorValues()` true,
            // which `IncrementalHnswGraphMerger` and `CheckIndex` key off,
            // while `PerFieldKnnVectorsFormat` registers no reader without
            // the format attribute -- so the field reads back as
            // vector-capable and yields nothing, with no error anywhere.
            f.vector_dimension = 0;
        }
        // Java: SegmentMerger keeps merged point dimensions even when every
        // point was deleted. The writer still emits an empty points triple.
    }
}
