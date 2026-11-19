#![cfg(test)]
// The arithmetic gate is about values read off disk; a fixture builder's
// own index arithmetic is not one (see `docs/arithmetic-gate.md`).
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// `ShuffleForcedMergePolicy.interleaveList`: by name, then oldest with
/// newest, the smaller of each pair first; a middle one alone.
#[test]
fn shuffle_interleave_pairs_oldest_with_newest() {
    let stat = |name: &str, size| merge_policy::SegmentStat {
        name: name.to_string(),
        doc_count: 1,
        del_count: 0,
        size_bytes: size,
    };
    let stats = [
        stat("_0", 50),
        stat("_1", 10),
        stat("_2", 30),
        stat("_3", 20),
        stat("_4", 5),
    ];
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(
        super::shuffle_interleave(names(&["_3", "_0", "_4", "_2", "_1"]), &stats),
        names(&["_4", "_0", "_1", "_3", "_2"])
    );
    // Equal sizes: the newer first, as Java's `<` test leaves it.
    let even = [stat("_a", 1), stat("_b", 1)];
    assert_eq!(
        super::shuffle_interleave(names(&["_a", "_b"]), &even),
        names(&["_b", "_a"])
    );
    assert!(super::shuffle_interleave(Vec::new(), &even).is_empty());
}

use crate::segment_info::IndexSortField;
use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    self as fi, DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::hnsw::HnswGraphView;
use lucene_codecs::postings::DocInput;

/// An older Lucene's segment names its postings and vector files after
/// *its* formats; a merge and a buffered delete must find them by that
/// name.
#[test]
fn old_segments_name_their_postings_and_vectors_after_their_own_formats() {
    let files: Vec<String> = vec![
        "_0.si".into(),
        "_0_Lucene90_0.tim".into(),
        "_0_lucene92HnswVectorsFormat_0.vem".into(),
    ];
    assert_eq!(
        postings_file_base(&files, "_0"),
        ("_0_Lucene90_0".to_string(), "Lucene90_0".to_string())
    );
    assert_eq!(
        retired_vector_files(&files, "_0"),
        Some((
            RetiredHnswFormat::Lucene92,
            "lucene92HnswVectorsFormat_0".to_string()
        ))
    );
    // This writer's own segment: no `.tim` of another format, and a
    // current-format `.vem`.
    let current: Vec<String> = vec!["_1_Lucene99HnswVectorsFormat_0.vem".into()];
    assert_eq!(
        postings_file_base(&current, "_1"),
        ("_1_Lucene104_0".to_string(), "Lucene104_0".to_string())
    );
    assert_eq!(retired_vector_files(&current, "_1"), None);
    // Another segment's files are not this one's.
    assert_eq!(retired_vector_files(&files, "_1"), None);
    // Quantized groups, each under its own suffix, by their metadata
    // files; the current format's and another segment's are not theirs.
    let quantized: Vec<String> = vec![
        "_0_Lucene99HnswScalarQuantizedVectorsFormat_1.vemq".into(),
        "_0_Lucene102BinaryQuantizedVectorsFormat_0.vemb".into(),
        "_0_Lucene99HnswScalarQuantizedVectorsFormat_1.veq".into(),
        "_0_Lucene99HnswVectorsFormat_0.vemf".into(),
        "_1_Lucene99ScalarQuantizedVectorsFormat_0.vemq".into(),
        "_0_SomethingElse_0.vemq".into(),
    ];
    assert_eq!(
        quantized_vector_files(&quantized, "_0"),
        vec![
            (
                QuantizedFormat::Lucene102BinaryQuantized,
                "Lucene102BinaryQuantizedVectorsFormat_0".to_string()
            ),
            (
                QuantizedFormat::Lucene99HnswScalarQuantized,
                "Lucene99HnswScalarQuantizedVectorsFormat_1".to_string()
            ),
        ]
    );
    assert!(quantized_vector_files(&current, "_1").is_empty());
}
use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
use lucene_store::directory::FsDirectory;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn stored_only_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn doc_value(d: &Document) -> String {
    match &d.fields[0].value {
        FieldValue::String(s) => s.clone(),
        other => panic!("unexpected field value shape: {other:?}"),
    }
}

use lucene_util::small_float;
use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir(tag: &str) -> TempDir {
    TempDir::new(&format!("index-writer-{tag}"))
}

fn strict_fixture(relative: &str) -> lucene_store::ByteBuffersDirectory {
    let dir = lucene_store::ByteBuffersDirectory::new();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(relative);
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_str().unwrap();
        if entry.file_type().unwrap().is_file()
            && !name.ends_with(".raw")
            && name != "manifest.properties"
        {
            write_file(&dir, name, &std::fs::read(entry.path()).unwrap()).unwrap();
        }
    }
    dir
}

fn strict_snapshot(dir: &dyn Directory) -> std::collections::BTreeMap<String, Vec<u8>> {
    dir.list_all()
        .unwrap()
        .into_iter()
        .map(|name| {
            let bytes = dir.open(&name).unwrap().to_vec();
            (name, bytes)
        })
        .collect()
}

#[test]
fn default_config_rejects_existing_field_roles_before_reclaiming_files() {
    for (fixture, marker) in [
        (
            "field_infos_index",
            "soft-deletes field is not configured in IWC",
        ),
        ("misc/bp_blocks", "has no parent document field configured"),
        (
            "codecs_simpletext/codec/flush/index",
            "has no parent document field configured",
        ),
    ] {
        let dir = strict_fixture(fixture);
        write_file(&dir, "_z.orphan", b"must survive validation failure").unwrap();
        let before = strict_snapshot(&dir);
        let error = IndexWriter::open_with_default_config(&dir, "Lucene104", version(), None)
            .err()
            .expect("default field configuration must be rejected");
        assert!(error.to_string().contains(marker), "{fixture}: {error}");
        assert_eq!(strict_snapshot(&dir), before, "{fixture}");
        // Failed construction releases the lock before returning.
        drop(dir.obtain_lock(WRITE_LOCK_NAME).unwrap());
    }
}

fn seed_strict_schema_segment(
    dir: &dyn Directory,
    name: &str,
    salt: u8,
    fields: &[FieldInfo],
) -> SegmentCommitInfo {
    let id = [salt; ID_LENGTH];
    let field_file = format!("{name}.fnm");
    write_file(dir, &field_file, &fi::write(fields, &id, "")).unwrap();
    let info = segment_info::SegmentInfo {
        id,
        version: version(),
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: Vec::new(),
        files: vec![format!("{name}.si"), field_file],
        attributes: Vec::new(),
        index_sort: None,
    };
    write_file(dir, &format!("{name}.si"), &segment_info::write(&info, "")).unwrap();
    SegmentCommitInfo {
        segment_name: name.into(),
        segment_id: id,
        codec_name: "Lucene104".into(),
        ..Default::default()
    }
}

#[test]
fn default_config_compares_schemas_across_segments_before_deletion_policy() {
    let dir = lucene_store::ByteBuffersDirectory::new();
    let numeric = FieldInfo::new("value", 0).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    );
    let binary = FieldInfo::new("value", 7).with_doc_values(
        DocValuesType::Binary,
        DocValuesSkipIndexType::None,
        -1,
    );
    let mut infos = empty_segment_infos(version());
    infos.generation = 1;
    infos.counter = 2;
    infos.segments = vec![
        seed_strict_schema_segment(&dir, "_0", 1, std::slice::from_ref(&numeric)),
        seed_strict_schema_segment(&dir, "_1", 2, &[binary]),
    ];
    segment_infos::write(&infos, &dir).unwrap();
    write_file(&dir, "_z.orphan", b"not reclaimed on failure").unwrap();
    let before = strict_snapshot(&dir);
    let error = IndexWriter::open_with_default_config(&dir, "Lucene104", version(), None)
        .err()
        .expect("inconsistent field schemas must fail at open");
    assert!(error.to_string().contains(
            "cannot change field \"value\" from doc values type=NUMERIC to inconsistent doc values type=BINARY"
        ), "{error}");
    assert_eq!(strict_snapshot(&dir), before);
    drop(dir.obtain_lock(WRITE_LOCK_NAME).unwrap());

    // Field numbers, update generations and attributes are per segment,
    // so their differences must not be mistaken for schema changes.
    let mut consistent = numeric;
    consistent.number = 7;
    consistent.doc_values_gen = 3;
    consistent.attributes = vec![("per_segment".into(), "value".into())];
    dir.delete_file("_1.fnm").unwrap();
    write_file(
        &dir,
        "_1.fnm",
        &fi::write(&[consistent], &[2; ID_LENGTH], ""),
    )
    .unwrap();
    let writer = IndexWriter::open_with_default_config(
        &dir,
        "Lucene104",
        version(),
        Some(std::sync::Arc::new(
            crate::deletion_policy::NoDeletionPolicy,
        )),
    )
    .unwrap();
    assert!(!dir.list_all().unwrap().contains(&"_z.orphan".to_string()));
    assert_eq!(writer.segment_infos().segments.len(), 2);
    drop(writer);
    drop(dir.obtain_lock(WRITE_LOCK_NAME).unwrap());
}

#[test]
fn default_config_opens_empty_and_normal_java_indices() {
    let empty = lucene_store::ByteBuffersDirectory::new();
    let writer =
        IndexWriter::open_with_default_config(&empty, "Lucene104", version(), None).unwrap();
    assert!(writer.segment_infos().segments.is_empty());
    drop(writer);
    let dir = strict_fixture("compound_index");
    let before = strict_snapshot(&dir);
    let writer = IndexWriter::open_with_default_config(&dir, "Lucene104", version(), None).unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 1);
    assert_eq!(strict_snapshot(&dir), before);
}

#[test]
fn current_field_infos_updates_do_not_require_base_segment_or_compound_files() {
    for codec in ["Lucene104", "SimpleText"] {
        let dir = lucene_store::ByteBuffersDirectory::new();
        let sci = SegmentCommitInfo {
            segment_name: "_0".into(),
            segment_id: [4; ID_LENGTH],
            codec_name: codec.into(),
            field_infos_gen: 2,
            ..Default::default()
        };
        let fields = vec![FieldInfo::new("updated", 3).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        )];
        let (name, bytes) = if codec == "SimpleText" {
            (
                "_0_2.inf",
                lucene_codecs::simpletext::field_infos::write(&fields),
            )
        } else {
            ("_0_2.fnm", fi::write(&fields, &sci.segment_id, "2"))
        };
        write_file(&dir, name, &bytes).unwrap();
        let actual = segment_field_infos(&dir, &sci).unwrap();
        assert_eq!(actual.fields.len(), 1);
        assert_eq!(actual.fields[0].name, "updated");
        assert_eq!(actual.fields[0].number, 3);
        dir.delete_file(name).unwrap();
        assert!(segment_field_infos(&dir, &sci).is_err());
    }
}

#[test]
fn simpletext_current_and_base_field_infos_match_java_fixture_metadata() {
    use lucene_codecs::simpletext::{compound, field_infos};
    for stage in ["flush", "merged"] {
        let dir = strict_fixture(&format!("codecs_simpletext/codec/{stage}/index"));
        let infos = segment_infos::read_latest(&dir).unwrap();
        for sci in &infos.segments {
            let current = segment_field_infos(&dir, sci).unwrap();
            assert!(!current.fields.is_empty());
            if sci.field_infos_gen != -1 {
                let name = field_infos::field_infos_file_name(
                    &sci.segment_name,
                    &lucene_util::base36::to_base36(sci.field_infos_gen),
                );
                let expected = field_infos::read(&dir.open(&name).unwrap()).unwrap();
                assert_eq!(
                    field_infos::write(&current.fields),
                    field_infos::write(&expected.fields)
                );
            }
            let mut base = sci.clone();
            base.field_infos_gen = -1;
            let actual = segment_field_infos(&dir, &base).unwrap();
            let info = segment_info::parse_simpletext(
                &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
                &sci.segment_name,
                &sci.segment_id,
            )
            .unwrap();
            let name = field_infos::field_infos_file_name(&sci.segment_name, "");
            let expected = if info.is_compound_file {
                let data = dir
                    .open(&compound::compound_file_name(&sci.segment_name))
                    .unwrap();
                let archive =
                    compound::SimpleTextCompoundReader::open(&data, &sci.segment_name).unwrap();
                field_infos::read(archive.open_input(&data, &name).unwrap()).unwrap()
            } else {
                field_infos::read(&dir.open(&name).unwrap()).unwrap()
            };
            assert_eq!(
                field_infos::write(&actual.fields),
                field_infos::write(&expected.fields)
            );
        }
    }
}

/// Reads every document out of every segment `segment_infos` lists, in
/// commit order -- the minimal "is this readable/searchable" check this
/// crate can do on its own (the real reader/searcher stack lives one
/// layer up in `lucene-search`, which already depends on `lucene-index`
/// and so cannot be depended on back from here -- see the
/// `architecture` skill).
fn read_all_docs(dir: &FsDirectory, segment_infos: &SegmentInfos) -> Vec<String> {
    let mut out = Vec::new();
    for sci in &segment_infos.segments {
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let live = if sci.del_gen >= 0 {
            let liv = dir
                .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
                .unwrap();
            Some(
                lucene_codecs::live_docs::parse(
                    &liv,
                    &sci.segment_id,
                    sci.del_gen,
                    reader.max_doc() as usize,
                    sci.del_count as usize,
                )
                .unwrap(),
            )
        } else {
            None
        };
        for doc_id in 0..reader.max_doc() {
            let is_live = live
                .as_ref()
                .map(|bits| bits.get(doc_id as usize))
                .unwrap_or(true);
            if is_live {
                out.push(doc_value(&reader.document(doc_id).unwrap()));
            }
        }
    }
    out
}

#[test]
fn open_on_a_fresh_directory_starts_with_no_segments() {
    let tmp = tempdir("fresh");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert!(writer.segment_infos().segments.is_empty());
    assert_eq!(writer.pending_doc_count(), 0);
}

/// `IndexWriter`'s constructor takes `write.lock` first: a second writer
/// on the same directory -- through any `Directory` instance over the
/// path -- fails with `LockObtainFailedException` until the first is
/// closed (dropped), and a failed open does not keep the lock.
#[test]
fn a_second_writer_on_one_directory_fails_to_obtain_the_write_lock() {
    let tmp = tempdir("write-lock");
    let dir = FsDirectory::open(&tmp);
    let fields = || vec![stored_only_field("id", 0)];
    let mut first = IndexWriter::open(&dir, fields(), "Lucene104", version()).unwrap();
    assert!(tmp.join(WRITE_LOCK_NAME).is_file());
    first.ensure_write_lock_valid().unwrap();

    let other = FsDirectory::open(&tmp);
    let err = IndexWriter::open(&other, fields(), "Lucene104", version())
        .err()
        .expect("the lock is held");
    assert!(
        matches!(err, Error::Store(lucene_store::Error::LockObtainFailed(_))),
        "{err}"
    );
    // The holder is unaffected, and commits.
    first.add_document(doc("1")).unwrap();
    first.commit().unwrap();
    drop(first);

    // Released on drop; `write.lock` is not an index file the deleter
    // touches, and a reader-side listing still sees one commit.
    let mut second = IndexWriter::open(&other, fields(), "Lucene104", version()).unwrap();
    assert_eq!(second.segment_infos().segments.len(), 1);
    second.rollback();
    drop(second);

    // A failed open (bad field list) releases the lock it took.
    let bad = FieldInfo::new("body", 0)
        .with_index_options(IndexOptions::DocsAndFreqs)
        .with_store_payloads(true);
    assert!(IndexWriter::open(&dir, vec![bad], "Lucene104", version()).is_err());
    IndexWriter::open(&dir, fields(), "Lucene104", version()).unwrap();
}

/// A writer whose `write.lock` was lost (here: deleted and recreated
/// behind its back) refuses to flush or commit rather than publish into
/// an index another writer may now own.
#[test]
fn a_writer_that_lost_its_lock_refuses_to_commit() {
    let tmp = tempdir("write-lock-lost");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(
        &dir,
        vec![stored_only_field("id", 0)],
        "Lucene104",
        version(),
    )
    .unwrap();
    writer.add_document(doc("1")).unwrap();
    std::fs::remove_file(tmp.join(WRITE_LOCK_NAME)).unwrap();
    assert!(writer.ensure_write_lock_valid().is_err());
    assert!(writer.flush().is_err());
    assert!(writer.commit().is_err());
    assert!(crate::segment_infos::read_latest(&dir).is_err());
}

/// `IndexWriter::open` is the port's caller-facing door for a hand-built
/// field list, and Java's is the one place a `FieldInfo` can come from --
/// its constructor, which throws. An inconsistent field must be found at
/// open, not much later at `parse` time or (for the combinations
/// `field_infos::write` coerces) never.
#[test]
fn open_rejects_a_field_list_javas_fieldinfo_constructor_would_throw_on() {
    let tmp = tempdir("open-validates-fields");
    let dir = FsDirectory::open(&tmp);

    // Payloads without positions: an *indexed* field, so no coercion
    // rescues it -- `FieldInfo.checkConsistency` throws.
    let bad = FieldInfo::new("body", 0)
        .with_index_options(IndexOptions::DocsAndFreqs)
        .with_store_payloads(true);
    assert!(matches!(
        IndexWriter::open(&dir, vec![bad], "Lucene104", version()).err(),
        Some(Error::FieldInfos(fi::Error::Inconsistent(_, _)))
    ));

    // Cross-field: two fields sharing a number, which
    // `FieldInfos(FieldInfo[])` rejects.
    assert!(matches!(
        IndexWriter::open(
            &dir,
            vec![stored_only_field("a", 0), stored_only_field("b", 0)],
            "Lucene104",
            version()
        )
        .err(),
        Some(Error::FieldInfos(fi::Error::InvalidFieldInfos(_)))
    ));
}

/// The other half of Java's constructor: a *non*-indexed field's
/// `omitNorms`/`storePayloads`/`storeTermVector` are coerced off rather
/// than rejected, so the writer's stored copy is the one Lucene would
/// hold -- and the `.fnm` it later writes cannot disagree with it.
#[test]
fn open_coerces_the_indexed_only_flags_off_a_non_indexed_field() {
    let tmp = tempdir("open-coerces-fields");
    let dir = FsDirectory::open(&tmp);
    let writer = IndexWriter::open(
        &dir,
        vec![FieldInfo::new("id", 0)
            .with_omit_norms(true)
            .with_store_term_vectors(true)
            .with_store_payloads(true)],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert!(!writer.cfg.fields[0].omit_norms);
    assert!(!writer.cfg.fields[0].store_term_vectors);
    assert!(!writer.cfg.fields[0].store_payloads);
}

#[test]
fn add_documents_then_commit_produces_one_readable_segment() {
    let tmp = tempdir("add-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc("c")).unwrap();
    assert_eq!(writer.pending_doc_count(), 3);

    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    assert_eq!(writer.pending_doc_count(), 0);

    // Readable back through the on-disk segments_N this call wrote --
    // not just through the returned struct.
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.generation, sis.generation);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
}

#[test]
fn committed_doc_count_is_zero_on_a_fresh_directory() {
    let tmp = tempdir("committed-doc-count-fresh");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert_eq!(writer.committed_doc_count().unwrap(), 0);
}

#[test]
fn committed_doc_count_sums_across_multiple_commits_and_segments_and_is_distinct_from_pending() {
    let tmp = tempdir("committed-doc-count-multi");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    // First commit: 3 docs in one segment.
    writer.add_document(doc("a")).unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc("c")).unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.committed_doc_count().unwrap(), 3);
    assert_eq!(writer.pending_doc_count(), 0);

    // Second commit: 2 more docs, producing a second segment. Total
    // committed count must now sum both segments' `.si` doc_count.
    writer.add_document(doc("d")).unwrap();
    writer.add_document(doc("e")).unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 2);
    assert_eq!(writer.committed_doc_count().unwrap(), 5);
    assert_eq!(writer.pending_doc_count(), 0);

    // Buffer 2 more docs without committing: committed_doc_count must
    // stay at 5 (untouched by uncommitted buffering) while
    // pending_doc_count reports the 2 buffered docs -- the two accessors
    // must never conflate committed-to-disk state with in-memory
    // buffered state.
    writer.add_document(doc("f")).unwrap();
    writer.add_document(doc("g")).unwrap();
    assert_eq!(writer.committed_doc_count().unwrap(), 5);
    assert_eq!(writer.pending_doc_count(), 2);

    // Committing the buffered pair brings committed_doc_count to 7 and
    // drains pending back to 0.
    writer.commit().unwrap();
    assert_eq!(writer.committed_doc_count().unwrap(), 7);
    assert_eq!(writer.pending_doc_count(), 0);
}

#[test]
fn commit_with_no_pending_documents_is_a_valid_no_op_content_commit() {
    let tmp = tempdir("empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.generation, 1);
    assert!(sis.segments.is_empty());

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.generation, 1);
}

#[test]
fn prepare_commit_then_finish_commit_matches_commit_directly() {
    // Two writers over two separate directories, fed the identical
    // pending docs -- one calls `commit()`, the other calls
    // `prepare_commit()` then `finish_commit()`. Their on-disk results
    // (the returned `SegmentInfos` and every readable document) must be
    // identical.
    let fields = || vec![stored_only_field("id", 0)];

    let tmp_a = tempdir("two-phase-direct");
    let dir_a = FsDirectory::open(&tmp_a);
    let mut writer_a = IndexWriter::open(&dir_a, fields(), "Lucene104", version()).unwrap();
    writer_a.add_document(doc("a")).unwrap();
    writer_a.add_document(doc("b")).unwrap();
    let sis_a = writer_a.commit().unwrap().clone();

    let tmp_b = tempdir("two-phase-split");
    let dir_b = FsDirectory::open(&tmp_b);
    let mut writer_b = IndexWriter::open(&dir_b, fields(), "Lucene104", version()).unwrap();
    writer_b.add_document(doc("a")).unwrap();
    writer_b.add_document(doc("b")).unwrap();
    writer_b.prepare_commit().unwrap();
    // Not yet visible: the prepared generation hasn't been activated.
    assert_eq!(
        lucene_store::directory::last_commit_generation(&dir_b.list_all().unwrap()).unwrap(),
        -1
    );
    let sis_b = writer_b.finish_commit().unwrap().clone();

    assert_eq!(sis_a.generation, sis_b.generation);
    assert_eq!(sis_a.segments.len(), sis_b.segments.len());
    assert_eq!(read_all_docs(&dir_a, &sis_a), read_all_docs(&dir_b, &sis_b));

    let reopened_b = segment_infos::read_latest(&dir_b).unwrap();
    assert_eq!(reopened_b.generation, sis_b.generation);
    assert_eq!(read_all_docs(&dir_b, &reopened_b), vec!["a", "b"]);
}

#[test]
fn prepare_commit_without_finish_commit_leaves_the_previous_commit_current() {
    // Simulates "prepared but never activated" (e.g. a crash before
    // `finish_commit()`): a fresh reader of `dir` must still see the
    // prior commit, not the prepared-but-unpublished one -- this port
    // provides no crash-recoverable "prepared" marker at all (see
    // `prepare_commit`'s doc comment), so the only honest guarantee is
    // that nothing changes until `finish_commit()` actually runs.
    let tmp = tempdir("prepare-only");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    let committed_generation = writer.segment_infos().generation;

    writer.add_document(doc("b")).unwrap();
    writer.prepare_commit().unwrap();

    // The new segment's files are on disk, but no segments_N points at
    // them: a fresh open() still sees only the first commit.
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.generation, committed_generation);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);

    // The live writer's segments are likewise untouched -- only
    // `finish_commit()` installs them. Its generation is not: Java's
    // `startCommit` runs `updateGeneration` as soon as the pending file
    // is written, so a later prepare never asks for the same name.
    assert_eq!(writer.segment_infos().generation, committed_generation + 1);
    assert_eq!(
        writer.segment_infos().segments.len(),
        reopened.segments.len()
    );
}

#[test]
fn finish_commit_without_a_prior_prepare_commit_is_an_error() {
    let tmp = tempdir("finish-without-prepare");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.finish_commit().unwrap_err();
    assert!(matches!(err, Error::NoPreparedCommit));
}

#[test]
fn finish_commit_twice_in_a_row_is_an_error_the_second_time() {
    let tmp = tempdir("finish-twice");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();
    writer.finish_commit().unwrap();

    let err = writer.finish_commit().unwrap_err();
    assert!(matches!(err, Error::NoPreparedCommit));
}

#[test]
fn calling_prepare_commit_again_before_finish_commit_is_rejected_and_loses_nothing() {
    // Real `IndexWriter.prepareCommitInternal` throws
    // IllegalStateException("prepareCommit was already called with no
    // corresponding call to commit"). Replacing the prepared state
    // instead would silently discard every document the first prepare had
    // already flushed and synced: the second prepare would rebuild its
    // SegmentInfos from `self.segment_infos`, which never saw the first
    // prepare's segment.
    let tmp = tempdir("prepare-twice");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();

    writer.add_document(doc("b")).unwrap();
    let err = writer.prepare_commit().unwrap_err();
    assert!(
        matches!(err, Error::PrepareCommitAlreadyCalled),
        "second prepare_commit() must be refused, got {err:?}"
    );

    // The first prepare is untouched and still activatable, and the
    // document buffered after it is still buffered (not flushed, not lost).
    assert_eq!(writer.pending_doc_count(), 1);
    let sis = writer.finish_commit().unwrap().clone();
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.generation, sis.generation);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);

    // ...and "b" lands in the next commit, so nothing was dropped anywhere.
    writer.commit().unwrap();
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
}
/// The real two-phase protocol, on disk: after `prepare_commit()` the
/// commit exists as `pending_segments_N` -- a name
/// `last_commit_generation` deliberately does not scan for -- and only
/// `finish_commit()`'s rename makes it the current `segments_N`. This is
/// what makes a crash between the two phases recoverable: the previous
/// commit is still current and the pending file is inert, rather than a
/// half-written `segments_N` that would make the whole index unopenable.
#[test]
fn prepare_commit_writes_a_pending_segments_file_that_finish_commit_renames() {
    let tmp = tempdir("two-phase-pending-file");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.prepare_commit().unwrap();

    let listed = dir.list_all().unwrap();
    assert!(
        listed.iter().any(|f| f == "pending_segments_2"),
        "prepare_commit must leave a pending_segments_N behind: {listed:?}"
    );
    assert!(
        !listed.iter().any(|f| f == "segments_2"),
        "prepare_commit must NOT publish segments_N yet: {listed:?}"
    );
    // A fresh reader still sees the previous commit, exactly as it would
    // after a crash at this point.
    let before = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(before.generation, 1);
    assert_eq!(read_all_docs(&dir, &before), vec!["a"]);

    writer.finish_commit().unwrap();
    let listed = dir.list_all().unwrap();
    assert!(
        !listed.iter().any(|f| f == "pending_segments_2"),
        "finish_commit must rename the pending file away, not copy it: {listed:?}"
    );
    assert!(listed.iter().any(|f| f == "segments_2"), "{listed:?}");
    let after = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(after.generation, 2);
    assert_eq!(read_all_docs(&dir, &after), vec!["a", "b"]);
}

/// `rollback()` after a `prepare_commit()` must also remove the pending
/// commit file (`SegmentInfos.rollbackCommit`), or a later
/// `prepare_commit()` at the same generation would be overwriting a file
/// the caller already asked to discard.
#[test]
fn rollback_deletes_the_pending_segments_file_prepare_commit_wrote() {
    let tmp = tempdir("two-phase-rollback-pending");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();
    assert!(dir
        .list_all()
        .unwrap()
        .iter()
        .any(|f| f == "pending_segments_1"));

    writer.rollback();
    let listed = dir.list_all().unwrap();
    assert!(
        !listed.iter().any(|f| f == "pending_segments_1"),
        "rollback must delete the pending commit file: {listed:?}"
    );
    assert!(matches!(
        writer.finish_commit().unwrap_err(),
        Error::NoPreparedCommit
    ));
}

/// Every operation that writes its own `segments_N` claims
/// `generation + 1` -- the *same* generation a pending prepared commit has
/// already claimed. Letting them through meant `finish_commit()` later
/// overwrote that commit with a segment list built before it, silently
/// reverting the delete/merge/update.
#[test]
fn commit_writing_operations_are_refused_while_a_commit_is_prepared() {
    let tmp = tempdir("prepared-guard");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();

    assert!(matches!(
        writer
            .delete_documents_with_sources(&[], "id", b"a")
            .unwrap_err(),
        Error::PreparedCommitPending("delete_documents")
    ));
    assert!(matches!(
        writer.delete_all().unwrap_err(),
        Error::PreparedCommitPending("delete_all")
    ));
    assert!(matches!(
        writer
            .update_document_with_sources(&[], "id", b"a", doc("x"))
            .unwrap_err(),
        Error::PreparedCommitPending("update_document")
    ));
    let merged = SegmentCommitInfo {
        segment_name: "_9".to_string(),
        segment_id: [0u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    };
    assert!(matches!(
        writer.apply_merge(&[], merged).unwrap_err(),
        Error::PreparedCommitPending("apply_merge")
    ));

    // The prepared commit is still intact and still activatable.
    writer.finish_commit().unwrap();
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
}

/// Java writes a fresh `StringHelper.randomId()` into every `segments_N`
/// header. Cloning the previous commit's `SegmentInfos` (as every commit
/// path here does) carried the same id forward forever, so no two
/// generations of an index were distinguishable by id.
#[test]
fn every_commit_generation_gets_its_own_commit_id() {
    let tmp = tempdir("commit-id");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    let first = writer.commit().unwrap().id;
    writer.add_document(doc("b")).unwrap();
    let second = writer.commit().unwrap().id;
    assert_ne!(first, second, "each commit must carry its own id");
    assert_eq!(segment_infos::read_latest(&dir).unwrap().id, second);
}

/// `IndexWriter.deleteAll()`: buffered docs and every segment go away
/// in memory; nothing is committed until the next `commit()`.
#[test]
fn delete_all_drops_buffered_and_committed_segments_but_only_on_next_commit() {
    let tmp = tempdir("delete-all");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("b")).unwrap();
    assert_eq!(writer.pending_doc_count(), 1);

    writer.delete_all().unwrap();
    assert_eq!(writer.pending_doc_count(), 0);
    assert!(writer.segment_infos().segments.is_empty());

    // Not durable yet: a fresh reader still sees the pre-deleteAll commit,
    // matching Java's deleteAll() not writing a commit of its own.
    let still = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &still), vec!["a"]);

    writer.commit().unwrap();
    let after = segment_infos::read_latest(&dir).unwrap();
    assert!(after.segments.is_empty());
    assert_eq!(read_all_docs(&dir, &after), Vec::<String>::new());
}

/// `IndexWriter.setLiveCommitData`/`getLiveCommitData`: opaque caller
/// metadata carried in `SegmentInfos.userData` from the next commit
/// onwards, and still there when the directory is reopened.
#[test]
fn live_commit_data_is_written_into_the_commit_and_survives_reopen() {
    let tmp = tempdir("live-commit-data");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
    assert!(writer.live_commit_data().is_empty());

    writer.set_live_commit_data(vec![("translog".to_string(), "uuid-1".to_string())]);
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(
        reopened.user_data,
        vec![("translog".to_string(), "uuid-1".to_string())]
    );

    // ...and a writer resumed on that directory reports it back, then
    // carries it into its own next commit unless replaced. (Closed
    // first: the resumed writer needs `write.lock`.)
    drop(writer);
    let mut resumed = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert_eq!(
        resumed.live_commit_data(),
        [("translog".to_string(), "uuid-1".to_string())]
    );
    resumed.add_document(doc("b")).unwrap();
    resumed.commit().unwrap();
    assert_eq!(
        segment_infos::read_latest(&dir).unwrap().user_data,
        vec![("translog".to_string(), "uuid-1".to_string())]
    );
}

/// A failed publish must not swallow the prepared commit: if `finish_commit`
/// `take()`s it unconditionally, a failure at the rename leaves the
/// already-flushed-and-synced segment orphaned with no way to retry or roll
/// back. Deleting the pending file out from under it is the cheapest way to
/// make the rename fail for real.
#[test]
fn a_failed_finish_commit_keeps_the_prepared_commit_instead_of_dropping_it() {
    let tmp = tempdir("finish-commit-failure");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();
    dir.delete_file("pending_segments_1").unwrap();

    let err = writer.finish_commit().unwrap_err();
    assert!(
        !matches!(err, Error::NoPreparedCommit),
        "the failure must come from the publish, not from a missing prepared commit: {err:?}"
    );
    // Still prepared: a second attempt reports the publish failure again,
    // not "nothing to activate".
    assert!(!matches!(
        writer.finish_commit().unwrap_err(),
        Error::NoPreparedCommit
    ));
    // ...and rollback still cleans the state up.
    writer.rollback();
    assert!(matches!(
        writer.finish_commit().unwrap_err(),
        Error::NoPreparedCommit
    ));
}

#[test]
fn prepare_commit_and_finish_commit_with_no_pending_documents_is_a_valid_no_op() {
    let tmp = tempdir("two-phase-empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.prepare_commit().unwrap();
    let sis = writer.finish_commit().unwrap().clone();
    assert!(sis.segments.is_empty());

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.generation, sis.generation);
    assert!(reopened.segments.is_empty());
}

#[test]
fn multiple_commits_produce_multiple_independent_segments() {
    let tmp = tempdir("multi-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();

    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc("c")).unwrap();
    writer.commit().unwrap();

    let sis = writer.segment_infos().clone();
    assert_eq!(sis.segments.len(), 2);
    assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.segments.len(), 2);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b", "c"]);
}

#[test]
fn rollback_discards_pending_docs_so_next_commit_never_sees_them() {
    let tmp = tempdir("rollback-basic");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.add_document(doc("b")).unwrap();
    assert_eq!(writer.pending_doc_count(), 2);

    writer.rollback();
    assert_eq!(writer.pending_doc_count(), 0);

    // Nothing was ever written to disk -- a subsequent commit is a
    // no-op-content commit, same as if the docs had never been added.
    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert!(reopened.segments.is_empty());
    assert_eq!(read_all_docs(&dir, &reopened), Vec::<String>::new());
}

#[test]
fn rollback_after_prepare_commit_discards_the_prepared_state_too() {
    // rollback() clearing only pending_docs would leave `prepared_commit`
    // intact, so prepare_commit() -> rollback() -> finish_commit() would
    // silently activate the segment the caller just asked to roll back.
    // rollback() must discard prepared_commit as well, and finish_commit()
    // afterward must fail
    // with NoPreparedCommit, exactly like it would if prepare_commit()
    // had never been called.
    let tmp = tempdir("rollback-after-prepare");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.prepare_commit().unwrap();
    assert!(writer.pending_doc_count() == 0 || writer.pending_doc_count() == 1);

    writer.rollback();

    let err = writer.finish_commit().unwrap_err();
    assert!(matches!(err, Error::NoPreparedCommit));

    // Nothing was ever written to disk -- no segments_N file exists at all.
    assert!(segment_infos::read_latest(&dir).is_err());

    // The writer is still fully usable afterward.
    writer.add_document(doc("b")).unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    assert_eq!(read_all_docs(&dir, &sis), vec!["b"]);
}

#[test]
fn rollback_with_nothing_pending_is_a_safe_no_op() {
    let tmp = tempdir("rollback-noop");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    assert_eq!(writer.pending_doc_count(), 0);
    writer.rollback();
    assert_eq!(writer.pending_doc_count(), 0);

    // Still fully usable afterward.
    writer.add_document(doc("a")).unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
}

#[test]
fn rollback_never_affects_a_prior_commits_segments() {
    let tmp = tempdir("rollback-prior-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    // First commit: "a" is on disk for good.
    writer.add_document(doc("a")).unwrap();
    let first = writer.commit().unwrap().clone();
    assert_eq!(first.segments.len(), 1);

    // Second batch: buffered but never committed, then rolled back.
    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc("c")).unwrap();
    assert_eq!(writer.pending_doc_count(), 2);
    writer.rollback();
    assert_eq!(writer.pending_doc_count(), 0);

    // A commit right after the rollback only re-writes segments_N with
    // no new segment appended -- "a"'s segment from the first commit is
    // still exactly as it was, "b"/"c" never appear anywhere.
    let second = writer.commit().unwrap().clone();
    assert_eq!(second.segments.len(), 1);
    assert_eq!(
        second.segments[0].segment_name,
        first.segments[0].segment_name
    );

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a"]);
}

#[test]
fn rollback_preserves_writer_configuration() {
    let tmp = tempdir("rollback-config");
    let dir = FsDirectory::open(&tmp);
    let mut text_field = stored_only_field("text", 0);
    text_field.index_options = IndexOptions::DocsAndFreqs;
    text_field.store_term_vectors = true;
    let mut num_field = stored_only_field("num", 1);
    num_field.doc_values_type = DocValuesType::Numeric;
    let fields = vec![text_field, num_field];

    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("text")).unwrap();
    writer.set_term_vector_field(Some("text")).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 2,
        ..Default::default()
    }));

    // Buffer a doc with no values for the configured fields (so this
    // rolled-back batch doesn't have to satisfy the dense doc-values
    // requirement), then roll it back.
    writer.add_document(doc("placeholder")).unwrap();
    writer.rollback();
    assert_eq!(writer.pending_doc_count(), 0);

    // Configuration must have survived: a doc with real values for every
    // configured field, committed after the rollback, produces postings/
    // term-vector/doc-values files exactly as if the rollback never
    // happened.
    let real_doc = Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("hello world".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(42),
            },
        ],
    };
    writer.add_document(real_doc).unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);

    let segment_name = &sis.segments[0].segment_name;
    let si_bytes = dir.open(&format!("{segment_name}.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &sis.segments[0].segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".doc")));
    assert!(si.files.iter().any(|f| f.ends_with(".tvd")));
    assert!(si.files.iter().any(|f| f.ends_with(".dvd")));
}

#[test]
fn reopening_an_existing_directory_resumes_its_committed_state() {
    let tmp = tempdir("resume");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];

    {
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
    }

    let mut writer2 = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert_eq!(writer2.segment_infos().segments.len(), 1);

    writer2.add_document(doc("b")).unwrap();
    writer2.commit().unwrap();

    let sis = writer2.segment_infos();
    assert_eq!(sis.segments.len(), 2);
    // The second session's segment name must not collide with the
    // first's.
    assert_ne!(sis.segments[0].segment_name, sis.segments[1].segment_name);
}

// --- update_document/delete_documents: needs a real postings fixture,
// same one term_delete.rs/update_document.rs's own tests already use. ---

struct Fixture {
    fields: blocktree::BlockTreeFields,
    doc_bytes: Vec<u8>,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
    max_doc: usize,
}

impl Fixture {
    fn doc_in(&self) -> DocInput<'_> {
        DocInput::open(&self.doc_bytes, &self.segment_id, &self.suffix).expect("open .doc")
    }
}

fn open_fixture() -> Fixture {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTree)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let id_hex = get("id_hex");
    let mut segment_id = [0u8; ID_LENGTH];
    for (i, slot) in segment_id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix");
    let max_doc: i32 = get("max_doc").parse().unwrap();

    let read_raw = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
    };
    let fnm = read_raw(&get("fnm_file_name"));
    let field_infos = fi::parse(&fnm, &segment_id, "").expect("parse .fnm");
    let tim = read_raw(&get("tim_file_name"));
    let tip = read_raw(&get("tip_file_name"));
    let tmd = read_raw(&get("tmd_file_name"));
    let fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &segment_id,
        &suffix,
        max_doc,
    )
    .expect("open blocktree");
    let doc_bytes = read_raw(&get("doc_file_name"));

    Fixture {
        fields,
        doc_bytes,
        segment_id,
        suffix,
        max_doc: max_doc as usize,
    }
}

/// Seeds a writer's committed state with the real-Lucene fixture segment
/// as segment `_0`, without going through `add_document`/`commit`
/// (the fixture already has real postings; this facade's own
/// `flush_stored_only_segment` path never writes any).
///
/// The segment's `.si` **is** written into `dir`, listing only itself. That
/// is not decoration: `IndexFileDeleter` enumerates a segment's files
/// through its `.si` exactly the way `CheckIndex` and real Lucene's own
/// `SegmentCommitInfo.files()` do, so a `SegmentCommitInfo` naming a segment
/// with no `.si` on disk is a corrupt commit, not a shortcut. Listing only
/// the `.si` (and not the fixture's `.tim`/`.tip`/`.tmd`/`.doc`, which live
/// in `fixtures/data/` and were never copied here) keeps every file the
/// deleter refcounts a file that actually exists in `dir`.
fn writer_seeded_with_fixture<'d>(
    dir: &'d FsDirectory,
    fx: &Fixture,
    fields: Vec<FieldInfo>,
) -> IndexWriter<'d> {
    let si = segment_info::SegmentInfo {
        id: fx.segment_id,
        version: version(),
        min_version: None,
        doc_count: fx.max_doc as i32,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.si".to_string()],
        attributes: vec![],
        index_sort: None,
    };
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
    // Written *after* the open: the deleter's init sweep reclaims every
    // unreferenced index file in `dir`, and a `.si` no commit names is
    // exactly that. Writing it here and checkpointing below is the same
    // order a real flush uses (write the files, then tell the deleter).
    write_file(dir, "_0.si", &segment_info::write(&si, "")).unwrap();
    writer.segment_infos.segments.push(SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: fx.segment_id,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    });
    writer.segment_infos.counter = 1;
    // `IndexFileDeleter.checkpoint(segmentInfos, false)`: without it the
    // seeded segment's `.si` is still unreferenced, and the next
    // checkpoint/refresh would reclaim it.
    let live = writer.live_infos();
    writer.deleter.checkpoint(&live, false).unwrap();
    writer
}

#[test]
fn update_document_replaces_a_matched_doc_and_is_visible_after_commit() {
    let fx = open_fixture();
    let tmp = tempdir("update-doc");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: None, // "id"->"id0" is a singleton term, no .doc needed
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    let sis = writer
        .update_document_with_sources(&sources, "id", b"id0", doc("replacement"))
        .unwrap()
        .clone();
    assert_eq!(sis.segments.len(), 2);
    assert_eq!(sis.segments[0].del_count, 1);

    let reopened = segment_infos::read_latest(&dir).unwrap();
    // The old segment's real postings can't be read back through this
    // crate's stored-fields-only reader helper, so just confirm the new
    // segment (this writer's own flush) is visible and correct.
    let new_sci = reopened
        .segments
        .iter()
        .find(|s| s.segment_name != "_0")
        .unwrap();
    let fdt = dir.open(&format!("{}.fdt", new_sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", new_sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", new_sci.segment_name)).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &new_sci.segment_id, "").unwrap();
    assert_eq!(doc_value(&reader.document(0).unwrap()), "replacement");
}

/// Real `IndexWriter.newSegmentName()` bumps `segmentInfos.counter` at
/// the moment it hands out a name, so the counter written into the very
/// next commit already accounts for the flushed segment. Bumping it only
/// on the in-memory writer *after* `update_document` had written its
/// `segments_N` would leave the committed counter stale: a writer reopened
/// on the same directory would hand out the same `_N` again and the next
/// flush would truncate a live segment's files out from under the commit
/// that referenced them.
#[test]
fn update_document_persists_the_bumped_segment_counter_so_a_reopen_never_reuses_a_name() {
    let fx = open_fixture();
    let tmp = tempdir("update-doc-counter");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields.clone());

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: None,
        live_docs: None,
        max_doc: fx.max_doc,
    }];
    let sis = writer
        .update_document_with_sources(&sources, "id", b"id0", doc("replacement"))
        .unwrap()
        .clone();
    let new_name = sis
        .segments
        .iter()
        .find(|s| s.segment_name != "_0")
        .unwrap()
        .segment_name
        .clone();

    let committed = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(
        committed.counter, sis.counter,
        "the committed counter must match the writer's in-memory one"
    );

    // A fresh writer over the same directory must not hand out `new_name`
    // again. (The first one closes first: `write.lock`.)
    drop(writer);
    let mut resumed = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert_ne!(
        resumed.new_segment_name(),
        new_name,
        "a reopened writer reused the segment name update_document just flushed"
    );
    resumed.add_document(doc("later")).unwrap();
    let after = resumed.commit().unwrap().clone();
    let names: Vec<&str> = after
        .segments
        .iter()
        .map(|s| s.segment_name.as_str())
        .collect();
    assert_eq!(
        names.len(),
        3,
        "the reopened writer's flush must be a third, distinct segment: {names:?}"
    );
    // The replacement document is still readable, i.e. its files were not
    // overwritten by the resumed writer's flush.
    let replacement = after
        .segments
        .iter()
        .find(|s| s.segment_name == new_name)
        .unwrap();
    let fdt = dir.open(&format!("{new_name}.fdt")).unwrap();
    let fdx = dir.open(&format!("{new_name}.fdx")).unwrap();
    let fdm = dir.open(&format!("{new_name}.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &replacement.segment_id, "").unwrap();
    assert_eq!(doc_value(&reader.document(0).unwrap()), "replacement");
}

#[test]
fn delete_documents_marks_matching_docs_dead_and_is_visible_after_commit() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir("delete-doc");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

    // `writer_seeded_with_fixture` only seeds this writer's in-memory
    // `SegmentCommitInfo` -- it never writes a real `_0.si` to `dir`
    // (the fixture's real files are `.tim`/`.tip`/`.tmd`/`.doc` only).
    // `committed_doc_count()` reads doc counts straight off each
    // segment's `.si` file, so write a minimal one here for `_0` --
    // same pattern `check_index.rs`'s own fixture setup uses.
    std::fs::write(
        tmp.join("_0.si"),
        segment_info::write(
            &segment_info::SegmentInfo {
                id: fx.segment_id,
                version: LuceneVersion {
                    major: 10,
                    minor: 0,
                    bugfix: 0,
                },
                min_version: None,
                doc_count: fx.max_doc as i32,
                is_compound_file: false,
                has_blocks: false,
                diagnostics: vec![],
                files: vec![],
                attributes: vec![],
                index_sort: None,
            },
            "",
        ),
    )
    .unwrap();

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    // "body" -> "cat" matches docs [0, 2] per the checked-in fixture
    // (same contents `term_delete.rs`'s own tests document).
    let sis = writer
        .delete_documents_with_sources(&sources, "body", b"cat")
        .unwrap()
        .clone();
    assert_eq!(sis.segments.len(), 1);
    assert_eq!(sis.segments[0].del_count, 2);
    assert_eq!(sis.segments[0].del_gen, 1);

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.segments[0].del_count, 2);

    let liv = dir.open("_0_1.liv").unwrap();
    let parsed = lucene_codecs::live_docs::parse(&liv, &fx.segment_id, 1, fx.max_doc, 2).unwrap();
    assert!(!parsed.get(0));
    assert!(parsed.get(1));
    assert!(!parsed.get(2));

    // `committed_doc_count()` is total-including-deleted, not live: the
    // segment's `.si` doc_count (the fixture's full `max_doc`) is
    // unchanged by a delete -- only a merge ever actually drops a
    // deleted doc from disk (see that method's doc comment) -- so the
    // count stays at `fx.max_doc` even though 2 of those docs are now
    // dead.
    assert_eq!(writer.committed_doc_count().unwrap(), fx.max_doc);
}

#[test]
fn delete_documents_with_no_matching_source_leaves_segment_untouched() {
    let fx = open_fixture();
    let tmp = tempdir("delete-doc-no-match");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

    // No source targets "_0" at all, so the segment must pass through
    // unmodified (no .liv written, del_count stays 0) even though
    // segment_infos itself still advances a generation/version.
    let sources: [SegmentDeleteSource; 0] = [];
    let sis = writer
        .delete_documents_with_sources(&sources, "body", b"cat")
        .unwrap()
        .clone();
    assert_eq!(sis.segments.len(), 1);
    assert_eq!(sis.segments[0].segment_name, "_0");
    assert_eq!(sis.segments[0].del_count, 0);
    assert_eq!(sis.segments[0].del_gen, -1);
}

#[test]
fn a_failing_update_document_leaves_the_writer_state_unchanged() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir("update-fail");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);
    let before = writer.segment_infos().clone();

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: 1, // bogus: doc id 2 (from "cat") is out of range
    }];

    let result = writer.update_document_with_sources(&sources, "body", b"cat", doc("nope"));
    assert!(result.is_err());
    assert!(!tmp.join("segments_1").exists());

    // Everything a reader could observe is unchanged...
    let after = writer.segment_infos().clone();
    assert_eq!(after.segments, before.segments);
    assert_eq!(after.generation, before.generation);
    assert_eq!(after.version, before.version);
    assert_eq!(after.id, before.id);

    // ...but the segment name this attempt consumed is gone for good. Real
    // `IndexWriter.newSegmentName()` bumps `segmentInfos.counter` in the
    // same synchronized block that hands the name out, precisely so a name
    // can never be handed out twice -- a failed attempt burning a name is
    // the intended, safe direction (reusing it would let the next flush
    // create_output over files a previous attempt may have left behind).
    assert_eq!(
        after.counter,
        before.counter + 1,
        "a failed update_document must not make its segment name reusable"
    );
}

/// A tight [`MergePolicyConfig`] whose threshold (`segments_per_tier`)
/// this test suite deliberately crosses/stays-under, so `commit()`'s
/// automatic-merge behavior is exercised deterministically rather than
/// relying on the (much larger) real-Lucene-shaped defaults.
fn tight_merge_policy() -> MergePolicyConfig {
    MergePolicyConfig {
        max_merge_at_once: 10,
        segments_per_tier: 2.0,
        max_merged_segment_size: 1_000_000,
        reclaim_weight: 1.0,
        // Above every segment these tests flush, mirroring real Lucene's
        // 16MB `floorSegmentBytes` sitting far above a freshly-flushed
        // tiny segment: the level walk in `find_merges` then budgets one
        // level only, so `segments_per_tier` alone decides when a merge
        // fires. With a zero floor the budget instead grows level by
        // level and three tiny segments are considered in-budget.
        floor_segment_size: 100_000,
        force_merge_deletes_pct_allowed: 10.0,
        ..MergePolicyConfig::default()
    }
}

#[test]
fn commit_with_no_merge_policy_set_never_auto_merges() {
    let tmp = tempdir("no-merge-policy");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    for doc_id in 0..5 {
        writer.add_document(doc(&doc_id.to_string())).unwrap();
        writer.commit().unwrap();
    }

    // 5 commits, no merge policy set => still 5 independent segments,
    // exactly as before automatic merge triggering existed.
    assert_eq!(writer.segment_infos().segments.len(), 5);
}

#[test]
fn commit_below_merge_threshold_stays_unmerged() {
    let tmp = tempdir("below-threshold");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // tight_merge_policy's segments_per_tier is 2, so 2 commits (2
    // segments) must stay below/at threshold and remain unmerged.
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();

    assert_eq!(writer.segment_infos().segments.len(), 2);
    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.segments.len(), 2);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
}

#[test]
fn commit_above_merge_threshold_automatically_merges_and_stays_readable() {
    let tmp = tempdir("above-threshold");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // tight_merge_policy's segments_per_tier is 2; crossing it (5
    // one-document commits) must trigger at least one automatic merge,
    // ending with fewer segments than commits.
    let ids = ["a", "b", "c", "d", "e"];
    for id in ids {
        writer.add_document(doc(id)).unwrap();
        writer.commit().unwrap();
    }

    let final_count = writer.segment_infos().segments.len();
    assert!(
        final_count < ids.len(),
        "expected automatic merging to reduce segment count below {}, got {final_count}",
        ids.len()
    );

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.segments.len(), final_count);
    let mut docs = read_all_docs(&dir, &reopened);
    docs.sort();
    assert_eq!(docs, vec!["a", "b", "c", "d", "e"]);
}

#[test]
fn repeated_commits_with_auto_merge_converge_without_panicking_or_looping_forever() {
    let tmp = tempdir("converge");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // Many small single-document commits in a row: each commit() call
    // must return (no infinite auto-merge loop), and the segment count
    // must never run away unboundedly.
    for i in 0..20 {
        writer.add_document(doc(&i.to_string())).unwrap();
        writer.commit().unwrap();
        assert!(
            writer.segment_infos().segments.len() <= 20,
            "segment count should never exceed the number of commits made so far"
        );
    }

    let reopened = segment_infos::read_latest(&dir).unwrap();
    let mut docs = read_all_docs(&dir, &reopened);
    docs.sort();
    let mut expected: Vec<String> = (0..20).map(|i| i.to_string()).collect();
    expected.sort();
    assert_eq!(docs, expected);
}

#[test]
fn auto_merge_correctly_carries_forward_a_segments_existing_deletions() {
    let tmp = tempdir("auto-merge-with-deletions");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

    // Two ordinary commits (no merge policy yet), so segment "_0" has a
    // real, on-disk, flushed segment to apply a deletion to directly via
    // `deletes::apply_deletes` (the same primitive `delete_documents`
    // itself calls), independent of term resolution.
    writer.add_document(doc("a")).unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();

    let sci = writer.segment_infos().segments[0].clone();
    assert_eq!(sci.segment_name, "_0");
    let fdt = dir.open("_0.fdt").unwrap();
    let fdx = dir.open("_0.fdx").unwrap();
    let fdm = dir.open("_0.fdm").unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    let max_doc = reader.max_doc() as usize;

    // Delete doc 0 ("a") directly, bypassing term resolution entirely --
    // this is exactly what `execute_merge`'s `sci.del_gen >= 0` branch
    // must read back correctly during an automatic merge.
    let updated_sci = deletes::apply_deletes(&dir, &sci, None, max_doc, [0]).unwrap();
    assert_eq!(updated_sci.del_gen, 1);
    assert_eq!(updated_sci.del_count, 1);

    let mut new_segment_infos = writer.segment_infos().clone();
    new_segment_infos.segments[0] = updated_sci;
    new_segment_infos.generation += 1;
    new_segment_infos.version += 1;
    segment_infos::write(&new_segment_infos, &dir).unwrap();

    // Reopen the writer against this on-disk state (one segment with a
    // real deletion already applied), enable the merge policy, and cross
    // its threshold so the deleted segment gets folded into an automatic
    // merge. The first writer closes first: `write.lock`.
    drop(writer);
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));
    writer.add_document(doc("c")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("d")).unwrap();
    writer.commit().unwrap();

    let final_count = writer.segment_infos().segments.len();
    assert!(
        final_count < 3,
        "expected the deleted-doc segment to merge away, got {final_count} segments"
    );

    let reopened = segment_infos::read_latest(&dir).unwrap();
    let mut docs = read_all_docs(&dir, &reopened);
    docs.sort();
    // "a" was deleted before the merge, so only "b", "c", "d" survive.
    assert_eq!(docs, vec!["b", "c", "d"]);
}

/// A [`Directory`] that records every `create_output`, `open` and
/// successful `sync` by name -- the only way to see how many times one
/// commit rewrites or fsyncs the same file, which no assertion on the
/// resulting bytes can. While `fail_sync` is set, every `sync` fails
/// and records nothing.
struct CountingDirectory<'a> {
    inner: &'a FsDirectory,
    created: std::sync::Mutex<Vec<String>>,
    opened: std::sync::Mutex<Vec<String>>,
    synced: std::sync::Mutex<Vec<String>>,
    fail_sync: std::sync::atomic::AtomicBool,
}

impl<'a> CountingDirectory<'a> {
    fn new(inner: &'a FsDirectory) -> Self {
        CountingDirectory {
            inner,
            created: std::sync::Mutex::new(Vec::new()),
            opened: std::sync::Mutex::new(Vec::new()),
            synced: std::sync::Mutex::new(Vec::new()),
            fail_sync: std::sync::atomic::AtomicBool::new(false),
        }
    }
    /// Every name synced so far, `segments`/`pending_segments` files
    /// left out.
    fn synced_index_files(&self) -> Vec<String> {
        self.synced
            .lock()
            .unwrap()
            .iter()
            .filter(|n| !n.contains("segments"))
            .cloned()
            .collect()
    }
    fn count(list: &std::sync::Mutex<Vec<String>>, name: &str) -> usize {
        list.lock()
            .unwrap()
            .iter()
            .filter(|n| n.as_str() == name)
            .count()
    }
}

impl Directory for CountingDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
        self.opened.lock().unwrap().push(name.to_string());
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.created.lock().unwrap().push(name.to_string());
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        if self.fail_sync.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(lucene_store::Error::Io(std::io::Error::other("sync")));
        }
        self.inner.sync(names)?;
        self.synced.lock().unwrap().extend(names.iter().cloned());
        Ok(())
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// A [`Directory`] that, while `fail` is set, fails the close (after the
/// bytes reached the file) and the fsync of a `pending_segments_N` and
/// then the delete that would clean it up -- leaving the file behind, as
/// a failed commit on a sick disk can.
struct StuckPendingDirectory<'a> {
    inner: &'a FsDirectory,
    fail: std::sync::atomic::AtomicBool,
    /// Which names get stuck: `pending_segments_N` unless a test says.
    stuck: fn(&str) -> bool,
}

impl StuckPendingDirectory<'_> {
    fn failing(&self, name: &str) -> lucene_store::Result<()> {
        if (self.stuck)(name) && self.fail.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(lucene_store::Error::Io(std::io::Error::other("stuck")));
        }
        Ok(())
    }
}

/// Fails an output's close after its bytes were flushed.
struct FailClose;
impl lucene_store::index_output::OutputFaults for FailClose {
    fn before_write(&mut self, _: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
        lucene_store::index_output::WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> lucene_store::index_output::CloseVerdict {
        lucene_store::index_output::CloseVerdict::FailAfterFlush(std::io::Error::other("stuck"))
    }
}

impl Directory for StuckPendingDirectory<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        let mut output = self.inner.create_output(name)?;
        if self.failing(name).is_err() {
            output.set_faults(Box::new(FailClose));
        }
        Ok(output)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        for name in names {
            self.failing(name)?;
        }
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.failing(name)?;
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// `IndexWriter.startCommit` records the prepared generation "on error
/// or success so we don't double-write a segments_N file". This writer
/// recorded it on success only, so a retry after a failed commit asked
/// for the same `pending_segments_N` again -- which `createOutput`
/// refuses once the failed commit could not delete it, wedging every
/// later commit of the writer.
#[test]
fn a_pending_segments_file_a_failed_commit_left_is_never_asked_for_again() {
    let tmp = tempdir("stuck-pending");
    let fs = FsDirectory::open(&tmp);
    let dir = StuckPendingDirectory {
        inner: &fs,
        fail: std::sync::atomic::AtomicBool::new(false),
        stuck: |name| name.starts_with("pending_segments"),
    };
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    let first = writer.segment_infos().generation;

    writer.add_document(doc("b")).unwrap();
    dir.fail.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(writer.commit().is_err());
    let stuck = lucene_store::directory::pending_segments_file_name(first + 1).unwrap();
    assert!(
        fs.list_all().unwrap().contains(&stuck),
        "the failed commit left {stuck}"
    );

    dir.fail.store(false, std::sync::atomic::Ordering::SeqCst);
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().generation, first + 2);
    assert_eq!(writer.committed_doc_count().unwrap(), 2);
    assert_eq!(
        segment_infos::read_latest(&fs).unwrap().generation,
        first + 2
    );
}

/// A buffered delete whose `.liv` cannot be written fails the commit,
/// and the writer keeps every segment and every packet; dropping the
/// segment lists on the error would make the next commit publish a
/// `segments_N` naming no segment at all.
/// The partial `.liv` cannot be deleted either, so the retry must also
/// take the next delete generation (`PendingDeletes.writeLiveDocs`'
/// `advanceNextWriteDelGen`).
#[test]
fn a_failed_delete_apply_keeps_every_segment_for_the_next_commit() {
    let tmp = tempdir("failed-delete-apply");
    let fs = FsDirectory::open(&tmp);
    let dir = StuckPendingDirectory {
        inner: &fs,
        fail: std::sync::atomic::AtomicBool::new(false),
        stuck: |name| name.ends_with(".liv"),
    };
    let fields = vec![FieldInfo {
        index_options: lucene_codecs::field_infos::IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let mut w = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.add_document(doc("d0")).unwrap();
    w.add_document(doc("d1")).unwrap();
    w.commit().unwrap();
    w.add_document(doc("d2")).unwrap();
    w.commit().unwrap();
    let segments = w.segment_infos().segments.len();

    w.delete_documents_by_term(&[Term::new("id", "d0")])
        .unwrap();
    dir.fail.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(w.commit().is_err());
    assert_eq!(w.segment_infos().segments.len(), segments);
    assert!(fs.list_all().unwrap().iter().any(|f| f.ends_with(".liv")));

    dir.fail.store(false, std::sync::atomic::Ordering::SeqCst);
    w.commit().unwrap();
    let latest = segment_infos::read_latest(&fs).unwrap();
    assert_eq!(latest.segments.len(), segments);
    let deleted: i32 = latest.segments.iter().map(|s| s.del_count).sum();
    assert_eq!(deleted, 1, "the restored packet was applied on the retry");
}

/// `IndexWriter.sealFlushedSegment` writes a segment's `.si` **once**,
/// from the in-memory `SegmentInfo` every format's writer has added its
/// files to, not once per file group.
///
/// Counted rather than inferred: every intermediate rewrite produces the
/// same final bytes, so nothing about the resulting segment can see them.
#[test]
fn one_commit_writes_the_segments_si_exactly_once() {
    let tmp = tempdir("si-written-once");
    let fs = FsDirectory::open(&tmp);
    let dir = CountingDirectory::new(&fs);
    let fields = vec![
        stored_only_field("id", 0),
        // Indexed, not omitting norms, and advertising term vectors ->
        // postings *and* norms *and* term vectors; the numeric column
        // below adds doc values. Four of the five per-format file groups
        // in one commit, on top of the stored-fields flush itself.
        FieldInfo {
            store_term_vectors: true,
            ..body_field(1)
        },
        numeric_dv_field("num", 2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();

    for i in 0..8 {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("d{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(format!("term{i} shared")),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(i as i64),
                    },
                ],
            })
            .unwrap();
    }
    writer.commit().unwrap();

    // Every format really did run, so the count below is over the full
    // set of writers, not a degenerate stored-fields-only flush.
    let created = dir.created.lock().unwrap().clone();
    for ext in [".fdt", ".tim", ".tvd", ".dvd", ".nvd"] {
        assert!(
            created.iter().any(|n| n.ends_with(ext)),
            "no {ext} written; the counted commit did not exercise every format: {created:?}"
        );
    }

    assert_eq!(
        CountingDirectory::count(&dir.created, "_0.si"),
        1,
        "the segment's .si was written more than once: {created:?}"
    );
    // **Zero** reads. As in Java (`SegmentCommitInfo` holds its
    // `SegmentInfo`, so `SegmentInfos.files()` reads the list out of
    // memory), the deleter never re-parses the finished `.si`: `flush`
    // hands the deleter the same in-memory list `seal_flushed_segment`
    // encoded (`IndexFileDeleter::record_segment_files`).
    assert_eq!(
        CountingDirectory::count(&dir.opened, "_0.si"),
        0,
        "the segment's .si was read back during its own flush: {:?}",
        dir.opened.lock().unwrap()
    );
    assert_eq!(
        CountingDirectory::count(&dir.synced, "_0.si"),
        1,
        "the segment's .si was fsynced more than once"
    );

    // ...and the one `.si` that was written still lists every file.
    let sci = &writer.segment_infos().segments[0];
    let si_bytes = fs.open("_0.si").unwrap().to_vec();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in [
        ".fdt", ".fdx", ".fdm", ".fnm", ".si", ".tim", ".tvd", ".dvd", ".nvd",
    ] {
        assert!(
            si.files.iter().any(|f| f.ends_with(ext)),
            "the single .si write lost {ext}: {:?}",
            si.files
        );
    }
}

/// Java's model: a flush writes a segment without syncing it, and
/// `IndexWriter.startCommit` syncs every file the commit names. Each file
/// is synced exactly once -- the next commit leaves the files the last one
/// synced alone.
#[test]
fn a_flush_syncs_nothing_and_each_commit_syncs_only_its_new_files_once() {
    let tmp = tempdir("sync-at-commit");
    let fs = FsDirectory::open(&tmp);
    let dir = CountingDirectory::new(&fs);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.flush().unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.flush().unwrap();
    assert!(
        dir.synced.lock().unwrap().is_empty(),
        "a flush synced {:?}",
        dir.synced.lock().unwrap()
    );

    writer.commit().unwrap();
    let committed = segment_infos::read_latest(&fs).unwrap();
    let first = crate::index_file_deleter::files_of_infos(&fs, &committed, false).unwrap();
    assert!(first.iter().any(|f| f.starts_with("_0.")));
    assert!(first.iter().any(|f| f.starts_with("_1.")));
    let mut synced = dir.synced_index_files();
    synced.sort();
    let mut expected = first.clone();
    expected.sort();
    assert_eq!(
        synced, expected,
        "the commit syncs exactly the files it names"
    );

    // A second commit with a new segment syncs that segment only.
    writer.add_document(doc("c")).unwrap();
    writer.commit().unwrap();
    for f in &first {
        assert_eq!(
            CountingDirectory::count(&dir.synced, f),
            1,
            "{f} was synced again by the second commit"
        );
    }
    let second = dir.synced_index_files();
    assert!(second.iter().any(|f| f.starts_with("_2.")));
    assert_eq!(
        second.len(),
        second
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        "no file was synced twice: {second:?}"
    );
}

/// A writer that resumes an existing index trusts that commit's files to
/// be durable already: its first commit syncs only what it wrote itself.
#[test]
fn a_resumed_writer_does_not_resync_the_files_of_the_commit_it_opened() {
    let tmp = tempdir("sync-resumed");
    let fs = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    {
        let mut writer = IndexWriter::open(&fs, fields.clone(), "Lucene104", version()).unwrap();
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
    }
    let dir = CountingDirectory::new(&fs);
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();
    let synced = dir.synced_index_files();
    assert!(!synced.is_empty());
    assert!(
        synced.iter().all(|f| f.starts_with("_1.")),
        "only the new segment is synced: {synced:?}"
    );
}

/// A commit-time sync that fails publishes nothing and records nothing
/// as synced, so the next commit syncs the same files again.
#[test]
fn a_failed_commit_time_sync_is_retried_by_the_next_commit() {
    let tmp = tempdir("sync-retry");
    let fs = FsDirectory::open(&tmp);
    let dir = CountingDirectory::new(&fs);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    let generation = writer.segment_infos().generation;

    writer.add_document(doc("b")).unwrap();
    dir.fail_sync
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(writer.commit().is_err());
    assert_eq!(
        segment_infos::read_latest(&fs).unwrap().generation,
        generation,
        "a commit whose files could not be synced must not be published"
    );

    dir.fail_sync
        .store(false, std::sync::atomic::Ordering::SeqCst);
    writer.commit().unwrap();
    let latest = segment_infos::read_latest(&fs).unwrap();
    assert!(latest.generation > generation);
    assert_eq!(latest.segments.len(), 2);
    let new_files = crate::index_file_deleter::segment_files(&fs, &latest.segments[1]).unwrap();
    for f in &new_files {
        assert_eq!(
            CountingDirectory::count(&dir.synced, f),
            1,
            "{f} was not synced by the retry"
        );
    }
}

/// A `.liv` generation that a later one supersedes before any commit
/// names it is never synced: only the generation the commit names is.
#[test]
fn a_superseded_live_docs_generation_is_never_synced() {
    let tmp = tempdir("sync-superseded-liv");
    let fs = FsDirectory::open(&tmp);
    let dir = CountingDirectory::new(&fs);
    let fields = vec![FieldInfo {
        index_options: lucene_codecs::field_infos::IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let mut w = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    for id in ["d0", "d1", "d2"] {
        w.add_document(doc(id)).unwrap();
    }
    w.commit().unwrap();

    w.delete_documents_by_term(&[Term::new("id", "d0")])
        .unwrap();
    w.flush().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d1")])
        .unwrap();
    w.flush().unwrap();
    let livs: Vec<String> = dir
        .created
        .lock()
        .unwrap()
        .iter()
        .filter(|n| n.ends_with(".liv"))
        .cloned()
        .collect();
    assert_eq!(livs.len(), 2, "two delete generations: {livs:?}");
    assert!(dir
        .synced_index_files()
        .iter()
        .all(|f| !f.ends_with(".liv")));

    w.commit().unwrap();
    assert_eq!(CountingDirectory::count(&dir.synced, &livs[0]), 0);
    assert_eq!(CountingDirectory::count(&dir.synced, &livs[1]), 1);
    assert_eq!(
        segment_infos::read_latest(&fs).unwrap().segments[0].del_count,
        2
    );
}

/// `prepareCommitInternal`/`startCommit`: "skip startCommit(): no changes
/// pending". A commit with nothing to commit writes no new generation and
/// still succeeds; the first commit of a new index always writes.
#[test]
fn a_commit_with_no_changes_writes_no_new_generation() {
    let tmp = tempdir("skip-empty-commit");
    let fs = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&fs, fields.clone(), "Lucene104", version()).unwrap();
    assert!(
        writer.has_uncommitted_changes(),
        "a new index has its first commit pending"
    );

    // A new index's first commit is written even with nothing in it.
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 1);
    assert!(!writer.has_uncommitted_changes());

    // Nothing changed: no new generation, but the call succeeds.
    let same = writer.commit().unwrap().generation;
    assert_eq!(same, 1);
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 1);
    writer.prepare_commit().unwrap();
    writer.prepare_commit().unwrap();
    writer.finish_commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 1);
    // Nothing is prepared after a skipped prepare.
    assert!(matches!(
        writer.finish_commit(),
        Err(Error::NoPreparedCommit)
    ));

    // A document is a change.
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 2);
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 2);

    // Commit data alone is a change (`setLiveCommitData` increments).
    writer.set_live_commit_data(vec![("k".into(), "v".into())]);
    assert!(writer.has_uncommitted_changes());
    writer.commit().unwrap();
    let latest = segment_infos::read_latest(&fs).unwrap();
    assert_eq!(latest.generation, 3);
    assert_eq!(latest.user_data, vec![("k".to_string(), "v".to_string())]);

    // A delete that matches nothing changes nothing.
    writer
        .delete_documents_by_term(&[Term::new("id", "missing")])
        .unwrap();
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 3);

    // `deleteAll` is a change, and so is a second one on an empty index.
    writer.delete_all().unwrap();
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 4);
    writer.delete_all().unwrap();
    writer.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 5);
    drop(writer);

    // A writer resuming an existing index has nothing pending.
    let mut resumed = IndexWriter::open(&fs, fields, "Lucene104", version()).unwrap();
    assert!(!resumed.has_uncommitted_changes());
    resumed.commit().unwrap();
    assert_eq!(segment_infos::read_latest(&fs).unwrap().generation, 5);
}

/// `IndexWriter.mergeMiddle`: a merge whose result holds no live
/// document writes nothing and drops both the result and its sources
/// ("Merge would produce a 0-doc segment, so we do nothing except commit
/// the merge to remove all the 0-doc segments that we merged"), so no
/// zero-document segment is left for every later open, merge and
/// `CheckIndex` to pay for.
#[test]
fn a_merge_whose_sources_are_all_deleted_is_dropped_not_committed() {
    let tmp = tempdir("zero-doc-merge-dropped");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

    // Three one-document segments, then every document deleted through
    // `deletes::apply_deletes` (the field is stored-only, so a term
    // delete has no postings to resolve against). All three `.liv` files
    // are on disk and all three segments are 100% deleted before any
    // merge is proposed.
    for id in ["a", "b", "c"] {
        writer.add_document(doc(id)).unwrap();
        writer.commit().unwrap();
    }
    let mut infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 3);
    for i in 0..infos.segments.len() {
        let sci = infos.segments[i].clone();
        infos.segments[i] = deletes::apply_deletes(&dir, &sci, None, 1, [0]).unwrap();
    }
    infos.generation += 1;
    infos.version += 1;
    segment_infos::write(&infos, &dir).unwrap();

    drop(writer);
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));
    writer.commit().unwrap();

    let committed = segment_infos::read_latest(&dir).unwrap();
    // Nothing is left: the sources were retired and no empty segment took
    // their place. Before the fix the commit named one brand-new
    // zero-document segment instead.
    assert!(
        committed.segments.is_empty(),
        "expected the fully-deleted sources to be dropped with nothing published, got {:?}",
        committed
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect::<Vec<_>>()
    );
    assert!(read_all_docs(&dir, &committed).is_empty());
    // And no files were written for the merge that did not happen.
    let leftovers: Vec<String> = index_files(&dir)
        .into_iter()
        .filter(|f| !f.starts_with("segments"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a dropped merge left files behind: {leftovers:?}"
    );
}

#[test]
fn apply_merge_folds_a_merge_result_into_the_writers_committed_state() {
    let tmp = tempdir("apply-merge");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 2);

    let seg0_name = writer.segment_infos().segments[0].segment_name.clone();
    let seg1_name = writer.segment_infos().segments[1].segment_name.clone();

    let fdt0 = dir.open(&format!("{seg0_name}.fdt")).unwrap().to_vec();
    let fdx0 = dir.open(&format!("{seg0_name}.fdx")).unwrap().to_vec();
    let fdm0 = dir.open(&format!("{seg0_name}.fdm")).unwrap().to_vec();
    let fdt1 = dir.open(&format!("{seg1_name}.fdt")).unwrap().to_vec();
    let fdx1 = dir.open(&format!("{seg1_name}.fdx")).unwrap().to_vec();
    let fdm1 = dir.open(&format!("{seg1_name}.fdm")).unwrap().to_vec();

    // Segment ids are generated internally, so re-derive them from the
    // committed SegmentInfos rather than hard-coding a value.
    let seg0_id = writer.segment_infos().segments[0].segment_id;
    let seg1_id = writer.segment_infos().segments[1].segment_id;
    let reader0 = stored_fields::open(&fdt0, &fdx0, &fdm0, &seg0_id, "").unwrap();
    let reader1 = stored_fields::open(&fdt1, &fdx1, &fdm1, &seg1_id, "").unwrap();

    let sources = vec![
        merge::MergeSource::stored_only(&fields, &reader0, None, Some(version())),
        merge::MergeSource::stored_only(&fields, &reader1, None, Some(version())),
    ];
    let merged_sci = merge::merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let sis = writer
        .apply_merge(&[seg0_name.as_str(), seg1_name.as_str()], merged_sci)
        .unwrap()
        .clone();
    assert_eq!(sis.segments.len(), 1);
    assert_eq!(sis.segments[0].segment_name, "_merged");

    let reopened = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(reopened.segments.len(), 1);
    assert_eq!(read_all_docs(&dir, &reopened), vec!["a", "b"]);
}

// --- set_postings_field / commit()'s postings-writing path ---

fn body_field(number: i32) -> FieldInfo {
    FieldInfo {
        index_options: IndexOptions::DocsAndFreqs,
        ..stored_only_field("body", number)
    }
}

fn doc_with_body(id: &str, body: &str) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body.to_string()),
            },
        ],
    }
}

#[test]
fn set_postings_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.set_postings_field(Some("nonexistent")).unwrap_err();
    assert!(matches!(err, Error::UnknownPostingsField(name) if name == "nonexistent"));
}

#[test]
fn set_postings_field_rejects_a_field_with_no_index_options() {
    let tmp = tempdir("unindexed-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.set_postings_field(Some("id")).unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedPostingsIndexOptions(name, IndexOptions::None) if name == "id"
    ));
}

#[test]
fn add_postings_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-add-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    let err = writer.add_postings_field("nonexistent").unwrap_err();
    assert!(matches!(err, Error::UnknownPostingsField(name) if name == "nonexistent"));
}

#[test]
fn add_postings_field_rejects_a_duplicate_field() {
    let tmp = tempdir("duplicate-add-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    let err = writer.add_postings_field("body").unwrap_err();
    assert!(matches!(err, Error::DuplicatePostingsField(name) if name == "body"));
}

#[test]
fn add_postings_field_accumulates_on_top_of_set_postings_field() {
    let tmp = tempdir("accumulate-add-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        body_field(1),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..stored_only_field("title", 2)
        },
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.add_postings_field("title").unwrap();

    assert_eq!(writer.cfg.postings_fields.len(), 2);
    assert_eq!(writer.cfg.postings_fields[0].name, "body");
    assert_eq!(writer.cfg.postings_fields[1].name, "title");

    // `set_postings_field(None)` still clears the whole accumulated list,
    // not just the last entry.
    writer.set_postings_field(None).unwrap();
    assert!(writer.cfg.postings_fields.is_empty());
}

#[test]
fn commit_with_postings_field_writes_readable_postings_for_multiple_docs_and_terms() {
    let tmp = tempdir("postings-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "the lazy fox"))
        .unwrap();
    writer
        .add_document(doc_with_body("c", "the fox runs"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    // Stored fields are still intact (backward-compatible).
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

    // The postings files exist and are listed in `.si`.
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["doc", "tim", "tip", "tmd"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    // Readable via the existing, unmodified read side: `fox` occurs in
    // all 3 docs, `quick`/`lazy`/`runs` are singletons, `the` occurs in
    // all 3 too but is not a singleton either.
    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            body_field(1),
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        3,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .doc");
    let field = block_fields.field("body").unwrap();

    let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1, 2]);
    let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1, 2]);
    let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0]);
    let postings = field.postings(b"lazy", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![1]);
    let postings = field.postings(b"runs", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![2]);
    assert!(field.seek_exact(b"missing").is_none());
}

/// Task #211's headline proof at the `lucene-index` level (the fuller
/// `PhraseQuery`-based proof lives in
/// `lucene-search/src/directory_reader.rs`'s `phrase_query_e2e` module,
/// since `lucene-index` can't depend on `lucene-search`): a field opted
/// into `IndexOptions::DocsAndFreqsAndPositions` postings via
/// `set_postings_field` produces a real `.pos` file (registered in `.si`,
/// present on disk), and `blocktree::FieldTerms::positions` -- the same
/// read path a query layer would use -- decodes each doc's occurrences
/// back out correctly, not just the doc-ID/freq shape
/// `commit_with_postings_field_writes_readable_postings_for_multiple_docs_and_terms`
/// already covers for `DocsAndFreqs`.
#[test]
fn commit_with_positions_index_options_writes_a_readable_pos_file() {
    let tmp = tempdir("postings-positions-commit");
    let dir = FsDirectory::open(&tmp);
    let positions_body_field = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), positions_body_field.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer
        .add_document(doc_with_body("a", "quick fox jumps"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "the fox sleeps"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    let pos_name = format!(
        "{}.pos",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    );
    assert!(
        si.files.contains(&pos_name),
        "missing {pos_name} in .si files"
    );
    assert!(
        dir.list_all().unwrap().contains(&pos_name),
        "missing {pos_name} on disk"
    );
    // No offsets were requested, so `.pay` must not exist at all.
    assert!(!si.files.iter().any(|f| f.ends_with(".pay")));

    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let pos_bytes = dir.open(&pos_name).unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            positions_body_field,
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .doc");
    let pos_in = lucene_codecs::postings::PosInput::open(
        &pos_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .pos");
    let field = block_fields.field("body").unwrap();

    // "fox" occurs in both docs: doc 0 at position 1 ("quick fox
    // jumps"), doc 1 at position 1 ("the fox sleeps").
    let fox_positions = field
        .positions(b"fox", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(fox_positions.len(), 2);
    assert_eq!(fox_positions[0].len(), 1);
    assert_eq!(fox_positions[0][0].position, 1);
    assert_eq!(fox_positions[1].len(), 1);
    assert_eq!(fox_positions[1][0].position, 1);

    // "quick" is a doc-0-only singleton at position 0.
    let quick_positions = field
        .positions(b"quick", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(quick_positions.len(), 1);
    assert_eq!(
        quick_positions[0],
        vec![lucene_codecs::postings::Position {
            position: 0,
            start_offset: -1,
            end_offset: -1,
            payload: Vec::new(),
        }]
    );
}

/// Same shape as
/// `commit_with_positions_index_options_writes_a_readable_pos_file`, but
/// with `IndexOptions::DocsAndFreqsAndPositionsAndOffsets` -- proves the
/// `has_offsets` branch in `IndexWriter::build_postings_output` also
/// feeds real character offsets through to a readable `.pos`/`.si` file
/// set (small enough here to stay inline in `.pos` rather than needing a
/// `.pay` file at all -- see `blocktree::FieldTerms::positions`'s doc
/// comment on when `.pay` is actually required).
#[test]
fn commit_with_offsets_index_options_writes_readable_offsets() {
    let tmp = tempdir("postings-offsets-commit");
    let dir = FsDirectory::open(&tmp);
    let offsets_body_field = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), offsets_body_field.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer
        .add_document(doc_with_body("a", "quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".pos")));

    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let pos_bytes = dir
        .open(&format!(
            "{}.pos",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            offsets_body_field,
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .doc");
    let pos_in = lucene_codecs::postings::PosInput::open(
        &pos_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .pos");
    let field = block_fields.field("body").unwrap();

    // "quick fox" -> "quick" spans bytes [0, 5), "fox" spans [6, 9).
    let fox_positions = field
        .positions(b"fox", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(fox_positions[0][0].start_offset, 6);
    assert_eq!(fox_positions[0][0].end_offset, 9);
    let quick_positions = field
        .positions(b"quick", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(quick_positions[0][0].start_offset, 0);
    assert_eq!(quick_positions[0][0].end_offset, 5);
}

/// A helper for the payload tests below: opens `.tim`/`.tip`/`.tmd`,
/// `.doc`, `.pos` and (when present) `.pay` from a committed segment and
/// returns each document's occurrences for one term of one field, read
/// through the same path a query layer would take.
pub(crate) fn read_occurrences(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    field_infos: &fi::FieldInfos,
    field: &str,
    term: &[u8],
    max_doc: i32,
) -> Vec<Vec<lucene_codecs::postings::Position>> {
    let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let open = |ext: &str| dir.open(&format!("{seg}.{ext}"));
    let tim = open("tim").unwrap();
    let tip = open("tip").unwrap();
    let tmd = open("tmd").unwrap();
    let doc_bytes = open("doc").unwrap();
    let pos_bytes = open("pos").unwrap();
    let pay_bytes = open("pay").ok();
    let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        field_infos,
        &sci.segment_id,
        &suffix,
        max_doc,
    )
    .expect("blocktree::open");
    let doc_in = DocInput::open(&doc_bytes, &sci.segment_id, &suffix).expect("open .doc");
    let pos_in = lucene_codecs::postings::PosInput::open(&pos_bytes, &sci.segment_id, &suffix)
        .expect("open .pos");
    let pay_in = pay_bytes.as_ref().map(|b| {
        lucene_codecs::postings::PayInput::open(b, &sci.segment_id, &suffix).expect("open .pay")
    });
    block_fields
        .field(field)
        .unwrap()
        .positions(term, Some(&doc_in), &pos_in, pay_in.as_ref())
        .unwrap()
        .unwrap()
}

/// A `store_payloads` field whose `index_options` indexes positions but
/// **not** offsets needs a `.pay`, which exists only when a field has
/// offsets or payloads: if `build_postings_output` did not set
/// `has_payloads`, the `.fnm` would promise payloads real Lucene would then
/// look for in a `.pay` that was never written. Lucene's `Lucene104PostingsReader` opens `.pay`
/// unconditionally when `fieldInfos.hasPayloads()`, so that segment could
/// not be opened at all.
///
/// The negative control is the sibling test below: the same field without
/// `store_payloads` must produce no `.pay` at all, so this test cannot
/// pass merely because something always writes one.
#[test]
fn a_payloads_field_without_offsets_writes_a_pay_file_and_round_trips_its_bytes() {
    let tmp = tempdir("postings-payloads-no-offsets");
    let dir = FsDirectory::open(&tmp);
    let body = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_payloads: true,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), body.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            Some(format!("{}@{}", ctx.term, ctx.position).into_bytes())
        })))
        .unwrap();

    writer
        .add_document(doc_with_body("a", "quick fox jumps"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "the fox sleeps"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    let pay_name = format!(
        "{}.pay",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    );
    assert!(
        si.files.contains(&pay_name),
        "a store_payloads field must produce a .pay even with no offsets: {:?}",
        si.files
    );
    assert!(dir.list_all().unwrap().contains(&pay_name));

    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            body,
        ],
    };
    let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 2);
    assert_eq!(fox.len(), 2);
    assert_eq!(fox[0][0].position, 1);
    assert_eq!(fox[0][0].payload, b"fox@1".to_vec());
    assert_eq!(fox[1][0].position, 1);
    assert_eq!(fox[1][0].payload, b"fox@1".to_vec());
    // Offsets are not indexed, so both are Lucene's -1 sentinel -- a
    // writer that emitted an offset region here would show up as a
    // decoded value instead.
    assert_eq!(fox[0][0].start_offset, -1);
    assert_eq!(fox[0][0].end_offset, -1);
}

/// The negative control for the test above: the same field shape with
/// `store_payloads` off must produce **no** `.pay` file, and every
/// occurrence must read back with an empty payload. Without this, "a .pay
/// exists" would be evidence of nothing.
#[test]
fn a_positions_field_without_payloads_writes_no_pay_file() {
    let tmp = tempdir("postings-no-payloads");
    let dir = FsDirectory::open(&tmp);
    let body = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), body.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
        .add_document(doc_with_body("a", "quick fox jumps"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        !si.files.iter().any(|f| f.ends_with(".pay")),
        "a field with neither offsets nor payloads needs no .pay: {:?}",
        si.files
    );

    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            body,
        ],
    };
    let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 1);
    assert!(fox[0][0].payload.is_empty());
}

/// The payload bytes have to be the ones the source produced, not merely
/// *some* bytes: this indexes the same corpus twice with two different
/// sources and requires the read-back payloads to differ. A writer that
/// wrote a constant, or dropped the source and wrote zero-length payloads
/// everywhere, passes every "a .pay exists" assertion and fails this one.
#[test]
fn the_payload_source_decides_the_bytes_that_land_on_disk() {
    let read_with = |tag: &str, prefix: &'static str| -> Vec<Vec<u8>> {
        let tmp = tempdir(tag);
        let dir = FsDirectory::open(&tmp);
        let body = fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            store_payloads: true,
            ..stored_only_field("body", 1)
        };
        let fields = vec![stored_only_field("id", 0), body.clone()];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer
            .set_payload_source(Some(Box::new(move |ctx| {
                Some(format!("{prefix}{}", ctx.position).into_bytes())
            })))
            .unwrap();
        writer
            .add_document(doc_with_body("a", "quick fox jumps"))
            .unwrap();
        writer.commit().unwrap();
        let sis = writer.segment_infos().clone();
        let field_infos = fi::FieldInfos {
            fields: vec![
                fi::FieldInfo {
                    index_options: IndexOptions::None,
                    ..stored_only_field("id", 0)
                },
                body,
            ],
        };
        read_occurrences(&dir, &sis.segments[0], &field_infos, "body", b"fox", 1)[0]
            .iter()
            .map(|p| p.payload.clone())
            .collect()
    };

    let a = read_with("payload-source-a", "a");
    let b = read_with("payload-source-b", "bb");
    assert_eq!(a, vec![b"a1".to_vec()]);
    assert_eq!(b, vec![b"bb1".to_vec()]);
    assert_ne!(a, b);
}

/// Java's `FieldInfo.checkConsistency`: "indexed field cannot have
/// payloads without positions". Caught at `open` -- the writer
/// can no longer *hold* the field, so `set_postings_field` never sees it.
#[test]
fn open_rejects_payloads_on_a_field_without_positions() {
    let tmp = tempdir("payloads-without-positions");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        fi::FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            store_payloads: true,
            ..stored_only_field("body", 1)
        },
    ];
    let Err(err) = IndexWriter::open(&dir, fields, "Lucene104", version()) else {
        panic!("a field with payloads but no positions must not open a writer");
    };
    assert!(
        matches!(err, Error::FieldInfos(fi::Error::Inconsistent(ref name, msg))
                if name == "body" && msg.contains("payloads without positions")),
        "{err:?}"
    );
}

/// Installing a payload source when no declared field stores payloads
/// would discard every payload it produced, silently. That is a caller
/// mistake worth naming rather than a no-op.
#[test]
fn set_payload_source_rejects_a_writer_with_no_payload_field() {
    let tmp = tempdir("payload-source-no-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    assert!(matches!(
        writer
            .set_payload_source(Some(Box::new(|_| None)))
            .unwrap_err(),
        Error::NoPayloadFields
    ));
    // Clearing is always allowed, whatever the field list.
    writer.set_payload_source(None).unwrap();
}

/// A `store_payloads` field with no source installed still has to carry
/// the payload-length stream its `.fnm` promises: real Lucene frames
/// `.pay` off `FieldInfo.hasPayloads()` alone, so "declared but never
/// used" must be an all-zero-length stream, not an absent one.
#[test]
fn a_payloads_field_with_no_source_still_writes_the_payload_length_stream() {
    let tmp = tempdir("payloads-no-source");
    let dir = FsDirectory::open(&tmp);
    let body = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_payloads: true,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), body.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
        .add_document(doc_with_body("a", "quick fox jumps"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".pay")));

    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            body,
        ],
    };
    let fox = read_occurrences(&dir, sci, &field_infos, "body", b"fox", 1);
    assert!(fox[0][0].payload.is_empty());
}

/// Term vectors must carry the same axes the field's postings do:
/// recording positions only, whatever the field's options, would make
/// `CheckIndex.testTermVectors`' offset and payload cross-checks vacuous
/// rather than failing, since it compares only the axes a vector declares.
#[test]
fn term_vectors_carry_the_offsets_and_payloads_the_field_declares() {
    let tmp = tempdir("tv-axes");
    let dir = FsDirectory::open(&tmp);
    let body = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        store_payloads: true,
        store_term_vectors: true,
        ..stored_only_field("body", 1)
    };
    // The control: same segment, a term-vector field that declares
    // neither axis, so "the vector has offsets" cannot be something the
    // writer does unconditionally.
    let title = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqs,
        store_term_vectors: true,
        ..stored_only_field("title", 2)
    };
    let fields = vec![stored_only_field("id", 0), body, title];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.add_term_vector_field("title").unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            Some(vec![ctx.position as u8, ctx.term.len() as u8])
        })))
        .unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("quick fox".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::String("quick".to_string()),
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    let doc0 = reader.document(0).unwrap().unwrap();

    let body_vec = doc0
        .fields
        .iter()
        .find(|f| f.field_number == 1)
        .expect("body vector");
    assert!(body_vec.has_positions);
    assert!(body_vec.has_offsets, "body indexes offsets");
    assert!(body_vec.has_payloads, "body stores payloads");
    let fox = body_vec
        .terms
        .iter()
        .find(|t| t.term == b"fox")
        .expect("fox in the vector");
    assert_eq!(fox.positions.as_deref(), Some(&[1i32][..]));
    assert_eq!(fox.start_offsets.as_deref(), Some(&[6i32][..]));
    assert_eq!(fox.end_offsets.as_deref(), Some(&[9i32][..]));
    assert_eq!(fox.payloads.as_deref(), Some(&[vec![1u8, 3]][..]));

    let title_vec = doc0
        .fields
        .iter()
        .find(|f| f.field_number == 2)
        .expect("title vector");
    assert!(title_vec.has_positions);
    assert!(!title_vec.has_offsets, "title indexes no offsets");
    assert!(!title_vec.has_payloads, "title stores no payloads");
    assert!(title_vec.terms[0].start_offsets.is_none());
    assert!(title_vec.terms[0].payloads.is_none());
}

/// A document's term-vector fields must come out in ascending field-**name**
/// order, whatever order the caller configured them in.
///
/// Real Lucene's `CheckIndex.checkTermVectors` iterates `TVFields`, which
/// yields the order the fields were written, and `checkFields` throws
/// unless that order is sorted by name. The wire format carries field
/// *numbers*, and this writer's numbers come from the caller's field list,
/// so a caller who declares `zeta` as field 1 and `alpha` as field 2 --
/// and calls `set_term_vector_field("zeta")` before
/// `add_term_vector_field("alpha")` -- must not get a segment written in
/// call order, i.e. `zeta` then `alpha`, which real `CheckIndex` rejects.
///
/// The negative control is the field *numbers*: they must stay 1 and 2
/// (the caller's own numbering), so this is a reordering of the per-doc
/// field list, not a renumbering of the schema.
#[test]
fn term_vector_fields_are_written_in_ascending_field_name_order() {
    let tmp = tempdir("tv-field-order");
    let dir = FsDirectory::open(&tmp);
    // Name order and number order deliberately disagree: `zeta` is field
    // 1, `alpha` is field 2.
    let zeta = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_term_vectors: true,
        ..stored_only_field("zeta", 1)
    };
    let alpha = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        store_term_vectors: true,
        ..stored_only_field("alpha", 2)
    };
    let fields = vec![stored_only_field("id", 0), zeta, alpha];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    // Configured in the *wrong* order on purpose.
    writer.set_term_vector_field(Some("zeta")).unwrap();
    writer.add_term_vector_field("alpha").unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("quick".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::String("fox".to_string()),
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    let doc0 = reader.document(0).unwrap().unwrap();

    let numbers: Vec<i32> = doc0.fields.iter().map(|f| f.field_number).collect();
    assert_eq!(
        numbers,
        vec![2, 1],
        "the document's vectors must be ordered by field name (alpha=2, zeta=1), \
             not by configuration order"
    );
    assert_eq!(doc0.fields[0].terms[0].term, b"fox", "alpha's only term");
    assert_eq!(doc0.fields[1].terms[0].term, b"quick", "zeta's only term");
}

/// A `.fnm` this port writes must be one this port can re-open. Java makes
/// the failing combination unrepresentable -- `FieldInfo`'s constructor
/// coerces `storeTermVectors`/`storePayloads`/`omitNorms` to `false` for a
/// non-indexed field before `checkConsistency` ever looks -- but a Rust
/// `FieldInfo` is a plain struct, so a caller can hand `IndexWriter::open`
/// a stored-only field with `omit_norms` set. Written to the wire, real
/// Lucene would coerce those bits away on read, but this port's own
/// `field_infos::parse` would reject the file, so they are coerced here.
///
/// The negative control is the second half: the same three bits on an
/// *indexed* field must survive, so the fix cannot be "always clear them".
#[test]
fn a_non_indexed_field_with_indexed_only_flags_still_writes_a_reopenable_fnm() {
    let tmp = tempdir("fnm-coercion");
    let dir = FsDirectory::open(&tmp);
    let stored_only_but_flagged = fi::FieldInfo {
        index_options: IndexOptions::None,
        omit_norms: true,
        ..stored_only_field("id", 0)
    };
    let indexed = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        omit_norms: true,
        store_payloads: true,
        store_term_vectors: true,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_but_flagged, indexed];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer
        .add_document(doc_with_body("a", "quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();

    let fnm = dir
        .open(&format!("{}.fnm", sis.segments[0].segment_name))
        .unwrap();
    let parsed = fi::parse(&fnm, &sis.segments[0].segment_id, "")
        .expect("this port must be able to re-open the .fnm it just wrote");
    let id = parsed.fields.iter().find(|f| f.name == "id").unwrap();
    assert!(!id.omit_norms, "coerced away for a non-indexed field");
    assert!(!id.store_payloads);
    assert!(!id.store_term_vectors);
    let body = parsed.fields.iter().find(|f| f.name == "body").unwrap();
    assert!(body.omit_norms, "an indexed field keeps its flags");
    assert!(body.store_payloads);
    assert!(body.store_term_vectors);

    // And the whole segment passes this port's own CheckIndex, which is
    // what the bug was hiding: `fnm.open` failing skipped every postings
    // check in the segment.
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn commit_with_no_postings_field_configured_stays_stored_only() {
    // Backward compatibility: a writer that never calls
    // `set_postings_field` must produce an index with no postings files
    // -- no `.doc`/`.tim`/`.tip`/`.tmd`
    // files at all.
    let tmp = tempdir("no-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    for ext in ["doc", "tim", "tip", "tmd"] {
        assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
    }
}

fn custom_freq_field(number: i32) -> FieldInfo {
    FieldInfo {
        index_options: IndexOptions::DocsAndCustomFreqs,
        ..stored_only_field("score", number)
    }
}

#[test]
fn set_custom_freq_postings_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-custom-freq-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer
        .set_custom_freq_postings_field(Some("nonexistent"))
        .unwrap_err();
    assert!(matches!(err, Error::UnknownCustomFreqPostingsField(name) if name == "nonexistent"));
}

#[test]
fn set_custom_freq_postings_field_rejects_a_field_with_wrong_index_options() {
    let tmp = tempdir("custom-freq-field-wrong-options");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer
        .set_custom_freq_postings_field(Some("body"))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedCustomFreqPostingsIndexOptions(name, IndexOptions::DocsAndFreqs)
            if name == "body"
    ));
}

#[test]
fn set_custom_freq_postings_field_accepts_a_docs_and_custom_freqs_field() {
    let tmp = tempdir("custom-freq-field-accepted");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();
    let cfg = writer.cfg.custom_freq_postings_field.as_ref().unwrap();
    assert_eq!(cfg.field_number, 1);

    writer.set_custom_freq_postings_field(None).unwrap();
    assert!(writer.cfg.custom_freq_postings_field.is_none());
}

#[test]
fn set_custom_freq_postings_field_rejects_when_a_text_postings_field_is_already_set() {
    let tmp = tempdir("mutual-exclusion-a");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        body_field(1),
        custom_freq_field(2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    let err = writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::PostingsAndCustomFreqPostingsMutuallyExclusive("set_custom_freq_postings_field")
    ));
}

#[test]
fn set_postings_field_rejects_when_a_custom_freq_postings_field_is_already_set() {
    let tmp = tempdir("mutual-exclusion-b");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        body_field(1),
        custom_freq_field(2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    let err = writer.set_postings_field(Some("body")).unwrap_err();
    assert!(matches!(
        err,
        Error::PostingsAndCustomFreqPostingsMutuallyExclusive("set_postings_field")
    ));

    let err = writer.add_postings_field("body").unwrap_err();
    assert!(matches!(
        err,
        Error::PostingsAndCustomFreqPostingsMutuallyExclusive("add_postings_field")
    ));
}

#[test]
fn commit_with_custom_freq_postings_field_writes_readable_postings_with_explicit_freqs() {
    // The whole point of `DocsAndCustomFreqs`: the freq value written is
    // exactly the caller's explicit `custom_freq`, not a derived
    // occurrence count -- proven here by supplying a `custom_freq` that
    // differs from "1 occurrence" for every doc/term pair and reading it
    // back byte-for-byte through the existing, unmodified
    // `blocktree`/`postings::DocInput` read side.
    let tmp = tempdir("custom-freq-postings-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    writer
        .add_document_with_custom_freq_terms(
            doc("a"),
            vec![("alpha".to_string(), 7), ("beta".to_string(), 3)],
        )
        .unwrap();
    writer
        .add_document_with_custom_freq_terms(doc("b"), vec![("alpha".to_string(), 42)])
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["doc", "tim", "tip", "tmd"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files");
    }

    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            custom_freq_field(1),
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .expect("blocktree::open on custom-freq .tim/.tip/.tmd");
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .doc");
    let field = block_fields.field("score").unwrap();

    let postings = field.postings(b"alpha", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1]);
    assert_eq!(postings.freqs, vec![7, 42]);

    let postings = field.postings(b"beta", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0]);
    assert_eq!(postings.freqs, vec![3]);

    assert!(field.seek_exact(b"missing").is_none());
}

#[test]
fn commit_with_custom_freq_postings_field_but_no_pending_docs_writes_no_postings_files() {
    let tmp = tempdir("custom-freq-postings-empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());
}

#[test]
fn commit_with_custom_freq_postings_field_and_no_doc_supplying_terms_skips_postings() {
    // A doc added via plain `add_document` alongside custom-freq-terms
    // docs contributes an empty term list (see `add_document`'s own doc
    // comment on keeping `pending_custom_freq_terms` aligned) -- not an
    // error, just nothing to index for that doc.
    let tmp = tempdir("custom-freq-postings-no-terms");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    writer.add_document(doc("a")).unwrap(); // no custom-freq terms at all
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
}

#[test]
fn commit_with_no_custom_freq_postings_field_configured_stays_stored_only() {
    let tmp = tempdir("no-custom-freq-postings-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer
        .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 5)])
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    for ext in ["doc", "tim", "tip", "tmd"] {
        assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
    }
}

#[test]
fn custom_freqs_below_one_are_refused_and_repeats_sum() {
    // Java refuses a term frequency below 1 when the document is added
    // (the wire encoding has none), so the buffer never holds one.
    let tmp = tempdir("custom-freq-below-one");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    let err = writer
        .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 0)])
        .unwrap_err();
    assert!(matches!(err, Error::Document(_)));
    // Refused before buffering: the writer still commits.
    assert_eq!(writer.pending_doc_count(), 0);
    writer.commit().unwrap();

    // A repeated term sums its freqs (Java's `Math.addExact`) rather
    // than leaving a flush that can never succeed.
    writer
        .add_document_with_custom_freq_terms(
            doc("b"),
            vec![("alpha".to_string(), 2), ("alpha".to_string(), 3)],
        )
        .unwrap();
    writer.commit().unwrap();
    assert!(writer
        .add_document_with_custom_freq_terms(
            doc("c"),
            vec![("alpha".to_string(), i32::MAX), ("alpha".to_string(), 1)],
        )
        .is_err());
}

/// `IndexingChain.invert` counts a `DOCS_AND_CUSTOM_FREQS` field's tokens
/// once each (`isTermDocField`: `length += 1`, the custom frequency is
/// not added), so a document's norm is the encoded number of its terms --
/// and a document that supplies no term has no such field, so no norm.
#[test]
fn a_custom_freq_field_with_norms_gets_the_norm_of_its_own_terms() {
    let tmp = tempdir("custom-freq-norms");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();
    writer
        .add_document_with_custom_freq_terms(
            doc("a"),
            vec![
                ("alpha".to_string(), 7),
                ("beta".to_string(), 3),
                ("gamma".to_string(), 90),
            ],
        )
        .unwrap();
    writer
        .add_document_with_custom_freq_terms(doc("b"), vec![("alpha".to_string(), 42)])
        .unwrap();
    writer
        .add_document_with_custom_freq_terms(doc("c"), Vec::new())
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    let entry = meta.entry(1).expect("the custom-freq field keeps norms");
    let encoded = |terms: u32| i64::from(small_float::int_to_byte4(terms) as i8);
    assert_eq!(norms::norm_value(&nvd, entry, 0).unwrap(), Some(encoded(3)));
    assert_eq!(norms::norm_value(&nvd, entry, 1).unwrap(), Some(encoded(1)));
    assert_eq!(norms::norm_value(&nvd, entry, 2).unwrap(), None);
}

#[test]
fn ids_drawn_with_the_same_salt_differ() {
    // The update-with-sources path draws a segment id and a commit id
    // from the same counter, possibly in the same clock tick.
    let ids: std::collections::HashSet<_> = (0..1000).map(|_| generate_segment_id(7)).collect();
    assert_eq!(ids.len(), 1000);
}

#[test]
fn rollback_discards_pending_custom_freq_terms_too() {
    let tmp = tempdir("custom-freq-rollback");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), custom_freq_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_custom_freq_postings_field(Some("score"))
        .unwrap();

    writer
        .add_document_with_custom_freq_terms(doc("a"), vec![("alpha".to_string(), 5)])
        .unwrap();
    assert_eq!(writer.pending_custom_freq_terms.len(), 1);
    writer.rollback();
    assert!(writer.pending_custom_freq_terms.is_empty());

    // Next commit sees nothing from the rolled-back doc.
    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());
}

/// `postings_writer` emits real full `ForUtil`/`PForUtil` blocks for
/// `docFreq >= BLOCK_SIZE (256)` and level-1 skip entries
/// (`postings_writer::write_level1_span`) for `docFreq >= LEVEL1_NUM_DOCS`
/// (8192); the byte-level round-trip proofs are that module's
/// `docfreq_exactly_one_full_block_no_tail`,
/// `docfreq_spans_multiple_full_blocks_plus_tail` and
/// `docfreq_at_level1_boundaries_round_trips` unit tests. There is no
/// end-to-end `IndexWriter`-level test of the 8192 boundary because
/// reaching it needs >= 8192 pending docs in one flush. A term just under
/// the 256 boundary must commit successfully.
#[test]
fn commit_succeeds_below_the_doc_freq_boundary() {
    let tmp = tempdir("postings-docfreq-just-under");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    for i in 0..255 {
        writer
            .add_document(doc_with_body(&i.to_string(), "shared"))
            .unwrap();
    }
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
}

#[test]
fn commit_with_postings_field_but_no_pending_docs_writes_no_postings_files() {
    let tmp = tempdir("postings-empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());
}

#[test]
fn commit_with_postings_field_but_no_doc_has_that_fields_text_skips_postings() {
    // A document that omits the opted-in postings field entirely (no
    // `StoredField` for its `field_number`) contributes no postings --
    // this must not be an error, just "nothing to index this commit".
    let tmp = tempdir("postings-no-text");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer.add_document(doc("a")).unwrap(); // only field_number 0 ("id"), no "body"
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
}

#[test]
fn commit_with_postings_field_holding_a_non_string_value_skips_that_doc() {
    // A doc whose stored value for the opted-in postings field isn't a
    // `FieldValue::String` (e.g. `Int`) contributes no indexable text --
    // matches `set_postings_field`'s own doc comment.
    let tmp = tempdir("postings-non-string-value");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(42), // not a String -- must be skipped
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
}

#[test]
fn commit_with_postings_field_text_that_tokenizes_to_nothing_skips_postings() {
    // The opted-in field has a `String` value on every doc, but that
    // text tokenizes to zero terms (e.g. only whitespace) -- still not
    // an error, just nothing to index this commit, distinct from the
    // "field missing/non-String" case above.
    let tmp = tempdir("postings-empty-tokenization");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer.add_document(doc_with_body("a", "   ")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
}

#[test]
fn setting_postings_field_back_to_none_restores_stored_only_behavior() {
    let tmp = tempdir("postings-field-reset");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_postings_field(None).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
}

#[test]
fn a_merged_routed_field_without_terms_merges_again() {
    // A routed field outlives a merge that drops its last term, so the
    // merged segment keeps postings files whose dictionary names no term.
    // The next merge opens them and gets no `SourcePostings` back; the
    // coverage gate must not mistake that for files it never opened.
    let tmp = tempdir("routed-field-without-terms");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.add_document(doc_with_body("a", "apple")).unwrap();
    writer.add_document(doc_with_body("b", "")).unwrap();
    writer.commit().unwrap();
    writer
        .delete_documents_by_term(&[Term::new("body", "apple")])
        .unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();

    writer.add_document(doc_with_body("c", "")).unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();

    let segments = &writer.segment_infos().segments;
    assert_eq!(segments.len(), 1);
    let si_bytes = dir
        .open(&format!("{}.si", segments[0].segment_name))
        .unwrap();
    let si = segment_info::parse(&si_bytes, &segments[0].segment_id).unwrap();
    assert_eq!(si.doc_count, 2);
    assert!(si.files.iter().any(|f| f.ends_with(".tim")));
}

#[test]
fn segments_with_postings_are_automatically_merged_with_postings_preserved() {
    // Enabling both set_postings_field and set_merge_policy at once now
    // merges postings-carrying segments for real: execute_merge opens
    // each source's .tim/.tip/.tmd/.doc (when present) and feeds them
    // through crate::merge::merge_postings via MergeSource::postings, so
    // segment_stats() no longer needs to exclude postings-bearing
    // segments from find_merges' candidate pool to avoid silent data
    // loss.
    let tmp = tempdir("postings-and-merge-policy");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // tight_merge_policy's segments_per_tier is 2 -- three one-doc
    // commits, each producing a segment with real postings, must cross
    // that threshold and trigger a merge down to one segment.
    for (id, text) in [
        ("a", "shared apple"),
        ("b", "shared banana"),
        ("c", "shared cherry"),
    ] {
        writer.add_document(doc_with_body(id, text)).unwrap();
        writer.commit().unwrap();
    }

    let segments = writer.segment_infos().segments.clone();
    assert_eq!(
        segments.len(),
        1,
        "postings-carrying segments must now merge down like any other"
    );
    let sci = &segments[0];

    let files = dir.list_all().unwrap();
    assert!(files.contains(&format!(
        "{}.tim",
        per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
    )));
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".tim")));

    // The merged segment's postings must still resolve every term from
    // every source doc -- open the merged .tim/.tip/.tmd/.doc for real
    // and confirm "shared" (present in all three docs) and "banana"
    // (present in exactly one) both round-trip correctly.
    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: vec![body_field(1)],
    };
    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    let max_doc = stored.max_doc();
    let block_tree = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        max_doc,
    )
    .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let body_terms = block_tree.field("body").unwrap();

    let shared_stats = body_terms.seek_exact(b"shared").unwrap();
    assert_eq!(
        shared_stats.doc_freq, 3,
        "\"shared\" appears in all 3 merged docs"
    );
    let shared_postings = body_terms
        .postings(b"shared", Some(&doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(shared_postings.docs.len(), 3);

    let banana_stats = body_terms.seek_exact(b"banana").unwrap();
    assert_eq!(banana_stats.doc_freq, 1);
}

// --- set_term_vector_field / commit()'s term-vector-writing path ---

fn tv_body_field(number: i32) -> FieldInfo {
    FieldInfo {
        index_options: IndexOptions::DocsAndFreqs,
        store_term_vectors: true,
        ..stored_only_field("body", number)
    }
}

#[test]
fn set_term_vector_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer
        .set_term_vector_field(Some("nonexistent"))
        .unwrap_err();
    assert!(matches!(err, Error::UnknownTermVectorField(name) if name == "nonexistent"));
}

#[test]
fn set_term_vector_field_rejects_a_field_without_store_term_vectors() {
    let tmp = tempdir("unflagged-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)]; // body_field: store_term_vectors == false
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.set_term_vector_field(Some("body")).unwrap_err();
    assert!(matches!(err, Error::UnsupportedTermVectorField(name) if name == "body"));
}

#[test]
fn add_term_vector_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-add-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    let err = writer.add_term_vector_field("nonexistent").unwrap_err();
    assert!(matches!(err, Error::UnknownTermVectorField(name) if name == "nonexistent"));
}

#[test]
fn add_term_vector_field_rejects_a_duplicate_field() {
    let tmp = tempdir("duplicate-add-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    let err = writer.add_term_vector_field("body").unwrap_err();
    assert!(matches!(err, Error::DuplicateTermVectorField(name) if name == "body"));
}

#[test]
fn commit_with_term_vector_field_writes_readable_term_vectors_for_multiple_docs_and_terms() {
    let tmp = tempdir("tv-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "the lazy fox"))
        .unwrap();
    writer
        .add_document(doc_with_body("c", "the fox runs"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    // Stored fields are still intact (backward-compatible).
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

    // The term-vector files exist and are listed in `.si`.
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["tvd", "tvx", "tvm"] {
        let name = format!("{}.{ext}", sci.segment_name);
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    // Readable via the existing, unmodified read side.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "")
        .expect("term_vectors::open on IndexWriter-produced .tvd/.tvx/.tvm");
    assert_eq!(reader.max_doc(), 3);

    let doc0 = reader.document(0).unwrap().unwrap();
    assert_eq!(doc0.fields.len(), 1);
    assert_eq!(doc0.fields[0].field_number, 1);
    assert!(doc0.fields[0].has_positions);
    let mut terms0: Vec<String> = doc0.fields[0]
        .terms
        .iter()
        .map(|t| String::from_utf8(t.term.clone()).unwrap())
        .collect();
    terms0.sort();
    assert_eq!(terms0, vec!["fox", "quick", "the"]);

    let doc2 = reader.document(2).unwrap().unwrap();
    let mut terms2: Vec<String> = doc2.fields[0]
        .terms
        .iter()
        .map(|t| String::from_utf8(t.term.clone()).unwrap())
        .collect();
    terms2.sort();
    assert_eq!(terms2, vec!["fox", "runs", "the"]);
}

#[test]
fn commit_with_no_term_vector_field_configured_stays_stored_only() {
    // Backward compatibility: a writer that never calls
    // `set_term_vector_field` must produce an index with no term-vector
    // files -- no `.tvd`/`.tvx`/`.tvm`
    // files at all.
    let tmp = tempdir("no-tv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let infos = fi::parse(&fnm, &sci.segment_id, "").unwrap();
    assert!(infos.fields.iter().all(|field| !field.store_term_vectors));

    let files = dir.list_all().unwrap();
    for ext in ["tvd", "tvx", "tvm"] {
        assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
    }
}

#[test]
fn commit_with_term_vector_field_but_no_pending_docs_writes_no_term_vector_files() {
    let tmp = tempdir("tv-empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());
}

#[test]
fn commit_with_term_vector_field_but_no_text_writes_empty_vectors() {
    let tmp = tempdir("tv-no-text");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer.add_document(doc("a")).unwrap(); // only field_number 0 ("id"), no "body"
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let infos = fi::parse(&fnm, &sci.segment_id, "").unwrap();
    assert!(
        infos
            .fields
            .iter()
            .find(|field| field.name == "body")
            .unwrap()
            .store_term_vectors
    );

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let vectors = term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    vectors.check_integrity().unwrap();
    assert_eq!(vectors.max_doc(), 1);
    assert!(vectors.document(0).unwrap().is_none());
}

#[test]
fn commit_with_term_vector_field_holding_a_non_string_value_skips_that_doc() {
    let tmp = tempdir("tv-non-string-value");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(42), // not a String -- must be skipped
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let infos = fi::parse(&fnm, &sci.segment_id, "").unwrap();
    assert!(
        infos
            .fields
            .iter()
            .find(|field| field.name == "body")
            .unwrap()
            .store_term_vectors
    );

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let vectors = term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    vectors.check_integrity().unwrap();
    assert_eq!(vectors.max_doc(), 1);
    assert!(vectors.document(0).unwrap().is_none());
}

#[test]
fn commit_with_term_vector_field_text_that_tokenizes_to_nothing_writes_empty_vectors() {
    let tmp = tempdir("tv-empty-tokenization");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer.add_document(doc_with_body("a", "   ")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let infos = fi::parse(&fnm, &sci.segment_id, "").unwrap();
    assert!(
        infos
            .fields
            .iter()
            .find(|field| field.name == "body")
            .unwrap()
            .store_term_vectors
    );
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.max_doc(), 1);
    assert!(reader.document(0).unwrap().is_none());
}

#[derive(Debug, Default)]
struct TvSchemaInstalledCodec {
    factory: lucene_codecs::term_vectors_writer::NativeTermVectorsWriterFactory,
}
impl lucene_codecs::codec::Lucene104Codec for TvSchemaInstalledCodec {
    fn term_vectors_writer_factory(
        &self,
    ) -> Option<&dyn lucene_codecs::term_vectors_writer::TermVectorsWriterFactory> {
        Some(&self.factory)
    }
}

#[test]
fn configured_term_vector_schema_changes_preserve_the_buffer_and_committed_view() {
    for installed in [false, true] {
        for compound in [false, true] {
            for (kind, initial, reopen, flush_only) in [
                ("disable", true, false, false),
                ("enable", false, false, false),
                ("reopen_disabled", true, true, false),
                ("flushed_disable", true, false, true),
                ("flushed_enable", false, false, true),
            ] {
                let tmp = tempdir(&format!("tv-schema-{installed}-{compound}-{kind}"));
                let dir = FsDirectory::open(&tmp);
                let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
                let open = || {
                    let mut writer =
                        IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
                    writer.set_postings_field(Some("body")).unwrap();
                    writer.set_use_compound_file(compound);
                    if installed {
                        writer
                            .set_codec(Some(std::sync::Arc::new(TvSchemaInstalledCodec::default())))
                            .unwrap();
                    }
                    writer
                };
                let mut writer = open();
                writer
                    .set_term_vector_field(initial.then_some("body"))
                    .unwrap();
                writer.add_document(doc_with_body("a", "x x")).unwrap();
                if flush_only {
                    writer.flush().unwrap();
                } else {
                    writer.commit().unwrap();
                }
                if reopen {
                    drop(writer);
                    writer = open();
                }
                writer
                    .set_term_vector_field((!initial).then_some("body"))
                    .unwrap();
                writer.add_document(doc_with_body("b", "x x")).unwrap();
                let before = strict_snapshot(&dir);
                let before_counter = writer.segment_infos.counter;
                let before_committed = writer.segment_infos.segments.len();
                let before_flushed = writer.flushed_segments.len();
                let before_ram = writer.ram_bytes_used;
                let before_doc = doc_value(&writer.pending_docs[0]);
                assert!(
                    matches!(writer.flush(), Err(Error::FieldInfos(fi::Error::Inconsistent(ref name, "cannot change store_term_vectors across indexed segments"))) if name == "body")
                );
                assert_eq!(strict_snapshot(&dir), before);
                assert_eq!(writer.segment_infos.counter, before_counter);
                assert_eq!(writer.segment_infos.segments.len(), before_committed);
                assert_eq!(writer.flushed_segments.len(), before_flushed);
                assert_eq!(writer.pending_docs.len(), 1);
                assert_eq!(doc_value(&writer.pending_docs[0]), before_doc);
                assert_eq!(writer.ram_bytes_used, before_ram);
                writer
                    .set_term_vector_field(initial.then_some("body"))
                    .unwrap();
                let committed = writer.commit().unwrap().clone();
                assert_eq!(committed.segments.len(), 2);
                let mut actual = Vec::new();
                for segment in &committed.segments {
                    let cfs = compound.then(|| {
                        CompoundReader::open(&dir, &segment.segment_name, &segment.segment_id)
                            .unwrap()
                    });
                    let source: &dyn Directory = match &cfs {
                        Some(cfs) => cfs,
                        None => &dir,
                    };
                    let fdt = source
                        .open(&format!("{}.fdt", segment.segment_name))
                        .unwrap();
                    let fdx = source
                        .open(&format!("{}.fdx", segment.segment_name))
                        .unwrap();
                    let fdm = source
                        .open(&format!("{}.fdm", segment.segment_name))
                        .unwrap();
                    let stored =
                        stored_fields::open(&fdt, &fdx, &fdm, &segment.segment_id, "").unwrap();
                    for doc_id in 0..stored.max_doc() {
                        actual.push(doc_value(&stored.document(doc_id).unwrap()));
                    }
                }
                assert_eq!(actual, vec!["a", "b"]);
            }
        }
    }
}

#[test]
fn configured_term_vector_schema_is_validated_before_concurrent_capture() {
    for installed in [false, true] {
        for compound in [false, true] {
            for initial in [false, true] {
                let tmp = tempdir(&format!(
                    "tv-concurrent-schema-{installed}-{compound}-{initial}"
                ));
                let dir = FsDirectory::open(&tmp);
                let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
                let open = || {
                    let mut writer =
                        IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
                    writer.set_postings_field(Some("body")).unwrap();
                    writer.set_use_compound_file(compound);
                    if installed {
                        writer
                            .set_codec(Some(std::sync::Arc::new(TvSchemaInstalledCodec::default())))
                            .unwrap();
                    }
                    writer
                };
                let mut writer = open();
                writer
                    .set_term_vector_field(initial.then_some("body"))
                    .unwrap();
                writer.add_document(doc_with_body("a", "x x")).unwrap();
                writer.commit().unwrap();
                drop(writer);
                let mut writer = open();
                writer
                    .set_term_vector_field((!initial).then_some("body"))
                    .unwrap();
                let before = strict_snapshot(&dir);
                assert!(
                    matches!(crate::concurrent_writer::ConcurrentIndexWriter::new(writer, 2), Err(Error::FieldInfos(fi::Error::Inconsistent(ref name, "cannot change store_term_vectors across indexed segments"))) if name == "body")
                );
                assert_eq!(strict_snapshot(&dir), before);
                let mut writer = open();
                writer
                    .set_term_vector_field(initial.then_some("body"))
                    .unwrap();
                let concurrent =
                    crate::concurrent_writer::ConcurrentIndexWriter::new(writer, 2).unwrap();
                concurrent.add_document(doc_with_body("b", "x x")).unwrap();
                concurrent.commit().unwrap();
            }
        }
    }
}

#[test]
fn configured_term_vectors_survive_missing_first_segments_and_append_reopen() {
    #[derive(Debug, Default)]
    struct InstalledNativeTv {
        factory: lucene_codecs::term_vectors_writer::NativeTermVectorsWriterFactory,
    }
    impl lucene_codecs::codec::Lucene104Codec for InstalledNativeTv {
        fn term_vectors_writer_factory(
            &self,
        ) -> Option<&dyn lucene_codecs::term_vectors_writer::TermVectorsWriterFactory> {
            Some(&self.factory)
        }
    }
    for installed in [false, true] {
        for reopen in [false, true] {
            // The fixed schema retains vector capability even when the
            // pending documents contain no indexable value for it.
            for first_kind in ["missing", "non_string", "empty"] {
                let tmp = tempdir(&format!("tv-gap-{installed}-{reopen}-{first_kind}"));
                let dir = FsDirectory::open(&tmp);
                let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
                let mut writer =
                    IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
                writer.set_term_vector_field(Some("body")).unwrap();
                if installed {
                    writer
                        .set_codec(Some(std::sync::Arc::new(InstalledNativeTv::default())))
                        .unwrap();
                }
                let first = match first_kind {
                    "missing" => doc("a"),
                    "non_string" => Document {
                        fields: vec![
                            StoredField {
                                field_number: 0,
                                value: FieldValue::String("a".into()),
                            },
                            StoredField {
                                field_number: 1,
                                value: FieldValue::Int(42),
                            },
                        ],
                    },
                    _ => doc_with_body("a", ""),
                };
                writer.add_document(first).unwrap();
                let first_commit = writer.commit().unwrap().clone();
                assert_eq!(first_commit.segments.len(), 1);
                let first_segment = &first_commit.segments[0];
                let metadata = dir
                    .open(&format!("{}.fnm", first_segment.segment_name))
                    .unwrap();
                let parsed = fi::parse(&metadata, &first_segment.segment_id, "").unwrap();
                let body_has_vectors = parsed
                    .fields
                    .iter()
                    .any(|f| f.name == "body" && f.store_term_vectors);
                assert!(body_has_vectors);
                assert!(dir
                    .open(&format!("{}.tvd", first_segment.segment_name))
                    .is_ok());
                if reopen {
                    drop(writer);
                    writer = IndexWriter::open(&dir, fields, "Lucene104", version())
                        .expect("an absent configured vector value must not prevent append reopen");
                    writer.set_term_vector_field(Some("body")).unwrap();
                    if installed {
                        writer
                            .set_codec(Some(std::sync::Arc::new(InstalledNativeTv::default())))
                            .unwrap();
                    }
                }
                writer.add_document(doc_with_body("b", "x x")).unwrap();
                let appended = writer.commit().unwrap().clone();
                assert_eq!(appended.segments.len(), 2);
                writer.force_merge(1).unwrap();
                let merged = writer.commit().unwrap().clone();
                assert_eq!(merged.segments.len(), 1);
                let segment = &merged.segments[0];
                let data = dir.open(&format!("{}.tvd", segment.segment_name)).unwrap();
                let index = dir.open(&format!("{}.tvx", segment.segment_name)).unwrap();
                let meta = dir.open(&format!("{}.tvm", segment.segment_name)).unwrap();
                let vectors =
                    term_vectors::open(&data, &index, &meta, &segment.segment_id, "").unwrap();
                vectors.check_integrity().unwrap();
                assert_eq!(vectors.max_doc(), 2);
                let fdt = dir.open(&format!("{}.fdt", segment.segment_name)).unwrap();
                let fdx = dir.open(&format!("{}.fdx", segment.segment_name)).unwrap();
                let fdm = dir.open(&format!("{}.fdm", segment.segment_name)).unwrap();
                let stored =
                    stored_fields::open(&fdt, &fdx, &fdm, &segment.segment_id, "").unwrap();
                let metadata = dir.open(&format!("{}.fnm", segment.segment_name)).unwrap();
                let parsed = fi::parse(&metadata, &segment.segment_id, "").unwrap();
                let id_number = parsed
                    .fields
                    .iter()
                    .find(|f| f.name == "id")
                    .unwrap()
                    .number;
                let mut seen = std::collections::BTreeSet::new();
                // Merge policy may choose the larger source first. Pair
                // each vector document with its stored ID, not source order.
                for doc_id in 0..vectors.max_doc() {
                    let document = stored.document(doc_id).unwrap();
                    let FieldValue::String(id) = &document
                        .fields
                        .iter()
                        .find(|f| f.field_number == id_number)
                        .unwrap()
                        .value
                    else {
                        panic!("stored ID changed type")
                    };
                    assert!(seen.insert(id.clone()), "duplicate stored ID {id}");
                    match id.as_str() {
                        "a" => assert_eq!(vectors.document(doc_id).unwrap(), None),
                        "b" => {
                            let populated = vectors.document(doc_id).unwrap().unwrap();
                            assert_eq!(populated.fields.len(), 1);
                            assert_eq!(populated.fields[0].terms.len(), 1);
                            assert_eq!(populated.fields[0].terms[0].term, b"x");
                            assert_eq!(populated.fields[0].terms[0].freq, 2);
                            assert_eq!(populated.fields[0].terms[0].positions, Some(vec![0, 1]));
                        }
                        other => panic!("unexpected stored ID {other}"),
                    }
                }
                assert_eq!(
                    seen,
                    std::collections::BTreeSet::from(["a".to_owned(), "b".to_owned()])
                );
                assert!(parsed
                    .fields
                    .iter()
                    .any(|f| f.name == "body" && f.store_term_vectors));
            }
        }
    }
}

#[test]
fn commit_with_term_vector_field_where_only_some_docs_have_text_still_writes_an_entry_per_doc() {
    // A doc with no indexable text for the opted-in field still needs a
    // `TermVectorsDocument` entry (empty `fields`) so doc IDs stay
    // aligned with the segment's real doc count --
    // `write_best_speed` derives `max_doc` from `docs.len()`.
    let tmp = tempdir("tv-partial-docs");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.add_document(doc("b")).unwrap(); // no "body" field at all
    writer.add_document(doc_with_body("c", "gamma")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    assert_eq!(reader.max_doc(), 3);
    assert_eq!(reader.document(0).unwrap().unwrap().fields.len(), 1);
    // A doc contributing zero fields to this chunk decodes as `None`,
    // not `Some(fields: vec![])` -- see `TermVectorsReader::document`'s
    // own `doc_num_fields == 0 => Ok(None)` branch.
    assert!(reader.document(1).unwrap().is_none());
    assert_eq!(reader.document(2).unwrap().unwrap().fields.len(), 1);
}

#[test]
fn setting_term_vector_field_back_to_none_restores_stored_only_behavior() {
    let tmp = tempdir("tv-field-reset");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_term_vector_field(None).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!("{}.tvd", sci.segment_name)));
}

#[test]
fn segments_with_term_vectors_are_automatically_merged_with_term_vectors_preserved() {
    // Same class of fix as `segments_with_postings_are_automatically_
    // merged_with_postings_preserved`, for term vectors instead of
    // postings: enabling both `set_term_vector_field` and
    // `set_merge_policy` at once now merges term-vector-carrying
    // segments for real -- `execute_merge` opens each source's real
    // `.tvd`/`.tvx`/`.tvm` (when present) and feeds them through
    // `crate::merge::write_merged_term_vectors` via `MergeSource::term_vectors`,
    // so `segment_stats()` no longer needs to exclude term-vector-
    // bearing segments from `find_merges`' candidate pool to avoid
    // silent data loss.
    let tmp = tempdir("tv-and-merge-policy");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // tight_merge_policy's segments_per_tier is 2 -- three one-doc
    // commits, each producing a segment with real term vectors, must
    // cross that threshold and trigger a merge down to one segment.
    for (id, text) in [("a", "alpha one"), ("b", "beta two"), ("c", "gamma three")] {
        writer.add_document(doc_with_body(id, text)).unwrap();
        writer.commit().unwrap();
    }

    let segments = writer.segment_infos().segments.clone();
    assert_eq!(
        segments.len(),
        1,
        "term-vector-carrying segments must now merge down like any other"
    );
    let sci = &segments[0];

    let files = dir.list_all().unwrap();
    assert!(files.contains(&format!("{}.tvd", sci.segment_name)));
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".tvd")));

    // The merged segment's term vectors must still hold every source
    // doc's own distinct terms -- open the merged .tvd/.tvx/.tvm for real
    // and check each doc's fields/terms. Automatic merging doesn't
    // guarantee source-segment order in the merged segment, so this
    // checks the *set* of per-doc term-vector contents survived intact
    // rather than assuming a fixed doc-id assignment.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    assert_eq!(reader.max_doc(), 3);

    let mut actual_docs: Vec<Vec<String>> = (0..3)
        .map(|doc_id| {
            let doc = reader.document(doc_id).unwrap().unwrap();
            assert_eq!(doc.fields.len(), 1);
            doc.fields[0]
                .terms
                .iter()
                .map(|t| std::str::from_utf8(&t.term).unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    actual_docs.sort();

    let mut expected_docs = vec![
        vec!["alpha".to_string(), "one".to_string()],
        vec!["beta".to_string(), "two".to_string()],
        vec!["gamma".to_string(), "three".to_string()],
    ];
    expected_docs.sort();

    assert_eq!(
        actual_docs, expected_docs,
        "every source doc's own term-vector content must survive the merge intact"
    );
}

#[test]
fn a_field_with_both_postings_and_term_vectors_configured_at_once_writes_both_correctly() {
    // Real Lucene's ordinary case: a field is both indexed (postings)
    // and has term vectors stored, in the same commit. Both write-side
    // passes must coexist: the segment's `.si` lists all seven files,
    // and both remain independently readable.
    let tmp = tempdir("postings-and-tv-together");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), tv_body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();

    writer
        .add_document(doc_with_body("a", "the quick fox"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "the lazy fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["doc", "tim", "tip", "tmd"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files");
    }
    // Term vectors are not a per-field format, so they keep plain names.
    for ext in ["tvd", "tvx", "tvm"] {
        let name = format!("{}.{ext}", sci.segment_name);
        assert!(si.files.contains(&name), "missing {name} in .si files");
    }

    // Postings side still readable.
    let tim = dir
        .open(&format!(
            "{}.tim",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tip = dir
        .open(&format!(
            "{}.tip",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let tmd = dir
        .open(&format!(
            "{}.tmd",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            tv_body_field(1),
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .expect("blocktree::open on IndexWriter-produced .tim/.tip/.tmd");
    let field = block_fields.field("body").unwrap();
    let doc_bytes = dir
        .open(&format!(
            "{}.doc",
            per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME)
        ))
        .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .expect("open .doc");
    let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1]);

    // Term-vector side also readable.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    let doc0 = reader.document(0).unwrap().unwrap();
    let mut terms0: Vec<String> = doc0.fields[0]
        .terms
        .iter()
        .map(|t| String::from_utf8(t.term.clone()).unwrap())
        .collect();
    terms0.sort();
    assert_eq!(terms0, vec!["fox", "quick", "the"]);
}

fn numeric_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::Numeric,
        ..stored_only_field(name, number)
    }
}

fn sorted_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::Sorted,
        ..stored_only_field(name, number)
    }
}

fn sorted_set_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::SortedSet,
        ..stored_only_field(name, number)
    }
}

fn binary_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::Binary,
        ..stored_only_field(name, number)
    }
}

fn sorted_numeric_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::SortedNumeric,
        ..stored_only_field(name, number)
    }
}

fn doc_with_score(id: &str, score: i64) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(score),
            },
        ],
    }
}

#[test]
fn set_doc_values_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-dv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer
        .set_doc_values_field(Some("nonexistent"))
        .unwrap_err();
    assert!(matches!(err, Error::UnknownDocValuesField(name) if name == "nonexistent"));
}

#[test]
fn set_doc_values_field_rejects_a_field_with_no_doc_values_type() {
    let tmp = tempdir("untyped-dv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.set_doc_values_field(Some("id")).unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedDocValuesType(name, DocValuesType::None) if name == "id"
    ));
}

#[test]
fn commit_with_doc_values_field_writes_readable_numeric_values_for_multiple_docs() {
    let tmp = tempdir("dv-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    writer.add_document(doc_with_score("a", 5)).unwrap();
    writer.add_document(doc_with_score("b", 250)).unwrap();
    writer.add_document(doc_with_score("c", -7)).unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    // Stored fields are still intact (backward-compatible).
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

    // The doc-values files exist and are listed in `.si`.
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["dvm", "dvd", "dvs"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    // Readable via the existing, unmodified read side.
    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced .dvm");
    let entry = meta.numeric_entry(1).unwrap();
    assert!(entry.is_dense());
    for (doc, want) in [(0, 5i64), (1, 250), (2, -7)] {
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(&dvd, entry, doc).unwrap(),
            Some(want)
        );
    }
}

#[test]
fn commit_with_doc_values_field_writes_readable_sorted_values_for_multiple_docs() {
    let tmp = tempdir("dv-commit-sorted");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("category")).unwrap();

    let categories = ["fruit", "vegetable", "fruit"];
    for (i, cat) in categories.iter().enumerate() {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(cat.to_string()),
                    },
                ],
            })
            .unwrap();
    }
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_field("category", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced SORTED .dvm");
    let entry = meta.sorted_entry(1).unwrap();
    let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
    for (doc, want) in categories.iter().enumerate() {
        let ord = lucene_codecs::doc_values::sorted_ord(&dvd, entry, doc as i32)
            .unwrap()
            .unwrap();
        assert_eq!(dict[ord as usize], want.as_bytes());
    }
}

#[test]
fn commit_with_doc_values_field_writes_readable_sorted_set_values_for_multiple_docs() {
    let tmp = tempdir("dv-commit-sorted-set");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_set_field("tags", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("tags")).unwrap();

    // A doc opts into multiple SORTED_SET values by repeating the field.
    let doc_tags: Vec<Vec<&str>> = vec![vec!["fruit", "red"], vec!["vegetable"], vec!["red"]];
    for (i, tags) in doc_tags.iter().enumerate() {
        let mut doc_fields = vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("doc{i}")),
        }];
        for tag in tags {
            doc_fields.push(StoredField {
                field_number: 1,
                value: FieldValue::String(tag.to_string()),
            });
        }
        writer
            .add_document(Document { fields: doc_fields })
            .unwrap();
    }
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_set_field("tags", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced SORTED_SET .dvm");
    let entry = meta.sorted_set_entry(1).unwrap();
    let (ords_entry, terms_entry) = match &entry.kind {
        lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => (ords, terms),
        lucene_codecs::doc_values::SortedSetKind::Single(_) => {
            panic!("expected Multi (some doc has 2 values)")
        }
    };
    let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, terms_entry).unwrap();
    for (doc, want) in doc_tags.iter().enumerate() {
        let ords =
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, doc as i32).unwrap();
        let mut got: Vec<&str> = ords
            .iter()
            .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
            .collect();
        got.sort_unstable();
        let mut want_sorted = want.clone();
        want_sorted.sort_unstable();
        assert_eq!(got, want_sorted);
    }
}

#[test]
fn commit_with_doc_values_field_writes_readable_binary_values_for_multiple_docs() {
    let tmp = tempdir("dv-commit-binary");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("payload")).unwrap();

    let payloads: [&[u8]; 3] = [b"abc", b"", b"hello world"];
    for (i, payload) in payloads.iter().enumerate() {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(payload.to_vec()),
                    },
                ],
            })
            .unwrap();
    }
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), binary_field("payload", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced BINARY .dvm");
    let entry = meta.binary_entry(1).unwrap();
    for (doc, want) in payloads.iter().enumerate() {
        let got = lucene_codecs::doc_values::binary_value(&dvd, entry, doc as i32)
            .unwrap()
            .unwrap();
        assert_eq!(got, *want);
    }
}

#[test]
fn commit_with_doc_values_field_rejects_non_binary_binary_value() {
    let tmp = tempdir("dv-commit-binary-nonbinary");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("payload")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(1),
                },
            ],
        })
        .unwrap();
    let err = writer.commit().unwrap_err();
    assert!(matches!(
        err,
        Error::NonBinaryDocValue(name, 0, "Long") if name == "payload"
    ));
}

#[test]
fn commit_with_doc_values_field_writes_readable_sorted_numeric_values_for_multiple_docs() {
    let tmp = tempdir("dv-commit-sorted-numeric");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("nums")).unwrap();

    // Doc with multiple values exercises the address-array path; the
    // other two docs (one value each) alone would collapse it away.
    let doc_nums: Vec<Vec<i64>> = vec![vec![10, 20], vec![-5], vec![0, 1, 2]];
    for (i, nums) in doc_nums.iter().enumerate() {
        let mut doc_fields = vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("doc{i}")),
        }];
        for n in nums {
            doc_fields.push(StoredField {
                field_number: 1,
                value: FieldValue::Long(*n),
            });
        }
        writer
            .add_document(Document { fields: doc_fields })
            .unwrap();
    }
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced SORTED_NUMERIC .dvm");
    let entry = meta.sorted_numeric_entry(1).unwrap();
    for (doc, want) in doc_nums.iter().enumerate() {
        let got =
            lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, doc as i32).unwrap();
        assert_eq!(&got, want);
    }
}

#[test]
fn commit_with_doc_values_field_rejects_non_numeric_sorted_numeric_value() {
    let tmp = tempdir("dv-commit-sorted-numeric-nonnumeric");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("nums")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("not a number".to_string()),
                },
            ],
        })
        .unwrap();
    let err = writer.commit().unwrap_err();
    assert!(matches!(
        err,
        Error::NonNumericDocValue(name, 0, "String") if name == "nums"
    ));
}

#[test]
fn commit_with_doc_values_field_rejects_doc_with_no_sorted_numeric_values() {
    let tmp = tempdir("dv-commit-sorted-numeric-missing");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("nums")).unwrap();

    writer
        .add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("a".to_string()),
            }],
        })
        .unwrap();
    let err = writer.commit().unwrap_err();
    assert!(matches!(
        err,
        Error::MissingDenseDocValue(name, 0) if name == "nums"
    ));
}

#[test]
fn commit_with_doc_values_field_rejects_non_binary_sorted_value() {
    let tmp = tempdir("dv-commit-sorted-nonbinary");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("category")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(1),
                },
            ],
        })
        .unwrap();
    let err = writer.commit().unwrap_err();
    assert!(matches!(
        err,
        Error::NonBinaryDocValue(name, 0, "Long") if name == "category"
    ));
}

#[test]
fn commit_with_doc_values_field_accepts_int_values_not_just_long() {
    // build_doc_values_output accepts both FieldValue::Int and
    // FieldValue::Long -- the multi-doc test above only exercises Long,
    // so this covers the Int arm's i64 sign-extension explicitly.
    let tmp = tempdir("dv-commit-int");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced .dvm");
    let entry = meta.numeric_entry(1).unwrap();
    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
        Some(-42i64)
    );
}

#[test]
fn commit_with_no_doc_values_field_configured_stays_stored_only() {
    // Backward compatibility: a writer that never calls
    // `set_doc_values_field` must produce an index with no doc-values files
    // -- no `.dvm`/`.dvd`/`.dvs` files at
    // all.
    let tmp = tempdir("no-dv-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc_with_score("a", 5)).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    for ext in ["dvm", "dvd", "dvs"] {
        assert!(!files.contains(&format!("{}.{ext}", sci.segment_name)));
    }
}

#[test]
fn commit_with_doc_values_field_but_no_pending_docs_writes_no_doc_values_files() {
    let tmp = tempdir("dv-empty-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    let sis = writer.commit().unwrap().clone();
    assert!(sis.segments.is_empty());
}

#[test]
fn commit_writes_sparse_numeric_doc_values_when_a_pending_doc_has_no_value() {
    // A doc missing the opted-in NUMERIC field entirely no longer
    // rejects the whole commit: it routes the present docs' values
    // through `write_single_sparse_numeric_field` instead of the dense
    // writer, and the missing doc reads back as absent (`None`), not as
    // a wrong/zero value.
    let tmp = tempdir("dv-sparse-numeric");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    writer.add_document(doc_with_score("a", 5)).unwrap();
    writer.add_document(doc("b")).unwrap(); // only field_number 0 ("id"), no "score"
    writer.add_document(doc_with_score("c", 7)).unwrap();

    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced sparse .dvm");
    let entry = meta.numeric_entry(1).unwrap();

    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
        Some(5)
    );
    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 1).unwrap(),
        None
    );
    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 2).unwrap(),
        Some(7)
    );
}

#[test]
fn commit_writes_sparse_sorted_doc_values_when_a_pending_doc_has_no_value() {
    // Same sparse contract as the NUMERIC test above, but for SORTED,
    // whose sparse writer builds a terms dictionary of only the present
    // docs' values and writes per-doc ordinals only for those docs.
    let tmp = tempdir("dv-sparse-sorted");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_field("category", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("category")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("red".to_string()),
                },
            ],
        })
        .unwrap();
    writer.add_document(doc("b")).unwrap(); // no "category" value
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("c".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("blue".to_string()),
                },
            ],
        })
        .unwrap();

    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_field("category", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced sparse SORTED .dvm");
    let entry = meta.sorted_entry(1).unwrap();

    assert!(lucene_codecs::doc_values::sorted_ord(&dvd, entry, 0)
        .unwrap()
        .is_some());
    assert_eq!(
        lucene_codecs::doc_values::sorted_ord(&dvd, entry, 1).unwrap(),
        None
    );
    assert!(lucene_codecs::doc_values::sorted_ord(&dvd, entry, 2)
        .unwrap()
        .is_some());
}

#[test]
fn commit_writes_sparse_binary_doc_values_when_a_pending_doc_has_no_value() {
    let tmp = tempdir("dv-sparse-binary");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), binary_field("payload", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("payload")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Binary(b"abc".to_vec()),
                },
            ],
        })
        .unwrap();
    writer.add_document(doc("b")).unwrap(); // no "payload" value
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("c".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Binary(b"xyz".to_vec()),
                },
            ],
        })
        .unwrap();

    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), binary_field("payload", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced sparse BINARY .dvm");
    let entry = meta.binary_entry(1).unwrap();

    assert_eq!(
        lucene_codecs::doc_values::binary_value(&dvd, entry, 0).unwrap(),
        Some(b"abc".as_slice())
    );
    assert_eq!(
        lucene_codecs::doc_values::binary_value(&dvd, entry, 1).unwrap(),
        None
    );
    assert_eq!(
        lucene_codecs::doc_values::binary_value(&dvd, entry, 2).unwrap(),
        Some(b"xyz".as_slice())
    );
}

#[test]
fn commit_writes_sparse_sorted_numeric_doc_values_when_a_pending_doc_has_no_value() {
    let tmp = tempdir("dv-sparse-sorted-numeric");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("nums")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(10),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(20),
                },
            ],
        })
        .unwrap();
    writer.add_document(doc("b")).unwrap(); // no "nums" value at all
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("c".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(-5),
                },
            ],
        })
        .unwrap();

    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_numeric_field("nums", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced sparse SORTED_NUMERIC .dvm");
    let entry = meta.sorted_numeric_entry(1).unwrap();

    assert_eq!(
        lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 0).unwrap(),
        vec![10, 20]
    );
    assert_eq!(
        lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 1).unwrap(),
        Vec::<i64>::new()
    );
    assert_eq!(
        lucene_codecs::doc_values::sorted_numeric_values(&dvd, entry, 2).unwrap(),
        vec![-5]
    );
}

#[test]
fn commit_writes_sparse_sorted_set_doc_values_when_a_pending_doc_has_no_value() {
    let tmp = tempdir("dv-sparse-sorted-set");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_set_field("tags", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("tags")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("fruit".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("red".to_string()),
                },
            ],
        })
        .unwrap();
    writer.add_document(doc("b")).unwrap(); // no "tags" value at all
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("c".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("vegetable".to_string()),
                },
            ],
        })
        .unwrap();

    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![stored_only_field("id", 0), sorted_set_field("tags", 1)],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .expect("parse_meta on IndexWriter-produced sparse SORTED_SET .dvm");
    let entry = meta.sorted_set_entry(1).unwrap();
    let (ords_entry, terms_entry) = match &entry.kind {
        lucene_codecs::doc_values::SortedSetKind::Multi { ords, terms } => (ords, terms),
        lucene_codecs::doc_values::SortedSetKind::Single(_) => {
            panic!("expected Multi (some doc has 2 values)")
        }
    };
    let dict = lucene_codecs::terms_dict::decode_all_terms(&dvd, terms_entry).unwrap();

    let mut doc0: Vec<&str> = lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 0)
        .unwrap()
        .iter()
        .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
        .collect();
    doc0.sort_unstable();
    assert_eq!(doc0, vec!["fruit", "red"]);

    assert_eq!(
        lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 1).unwrap(),
        Vec::<i64>::new(),
        "doc without a \"tags\" value must report zero ordinals, not a stale/wrong one"
    );

    let doc2: Vec<&str> = lucene_codecs::doc_values::sorted_numeric_values(&dvd, ords_entry, 2)
        .unwrap()
        .iter()
        .map(|&ord| std::str::from_utf8(&dict[ord as usize]).unwrap())
        .collect();
    assert_eq!(doc2, vec!["vegetable"]);
}

// Note: a "one field dense, another field on the same docs sparse"
// mixed-commit test is not expressible against the current API --
// `IndexWriter` supports only one `doc_values_field` configured at a
// time (`doc_values_field: Option<DocValuesFieldConfig>`, not a list;
// see `set_doc_values_field`'s doc comment), a pre-existing limitation
// unrelated to sparse support. The dense
// vs. sparse build paths for a *single* field are independent match
// arms in `build_doc_values_output` (dense/sparse is decided per field,
// per call), so there is no shared mutable state between fields for
// multiple fields to interfere through even once that limitation is
// lifted.

#[test]
fn commit_rejects_when_a_pending_docs_value_is_not_numeric() {
    let tmp = tempdir("dv-non-numeric-value");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("not-a-number".to_string()), // wrong type
                },
            ],
        })
        .unwrap();

    let err = writer.commit().unwrap_err();
    assert!(matches!(
        err,
        Error::NonNumericDocValue(name, 0, "String") if name == "score"
    ));
}

#[test]
fn setting_doc_values_field_back_to_none_restores_stored_only_behavior() {
    let tmp = tempdir("dv-field-reset");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();
    writer.set_doc_values_field(None).unwrap();

    writer.add_document(doc_with_score("a", 5)).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let files = dir.list_all().unwrap();
    assert!(!files.contains(&format!(
        "{}.dvm",
        per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
    )));
}

#[test]
fn an_automatic_merge_carries_doc_values_through_instead_of_dropping_them() {
    // Doc-values-bearing segments take part in merges, so the thing to
    // assert is not that the merge happened but that **every document still has its own value**
    // afterwards, which is the property a lost or mis-mapped column
    // breaks while leaving a perfectly valid segment behind.
    let tmp = tempdir("dv-and-merge-policy");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    for (id, score) in [("a", 1i64), ("b", 2), ("c", 3)] {
        writer.add_document(doc_with_score(id, score)).unwrap();
        writer.commit().unwrap();
    }

    let infos = writer.segment_infos().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "a tight merge policy must now fold all three doc-values segments into one"
    );
    let sci = &infos.segments[0];
    let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
    let dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap().to_vec();
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &lucene_codecs::field_infos::FieldInfos {
            fields: vec![stored_only_field("id", 0), numeric_field("score", 1)],
        },
    )
    .unwrap();
    let entry = meta.numeric_entry(1).unwrap();
    // Stored `id` and doc-values `score` were written from the same
    // document, so they must still describe the same document at the same
    // merged doc id.
    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    assert_eq!(reader.max_doc(), 3);
    for d in 0..3 {
        let id = match &reader.document(d).unwrap().fields[0].value {
            FieldValue::String(v) => v.clone(),
            other => panic!("unexpected stored value {other:?}"),
        };
        let expected = match id.as_str() {
            "a" => 1,
            "b" => 2,
            "c" => 3,
            other => panic!("unexpected id {other}"),
        };
        assert_eq!(
            doc_values::numeric_value(&dvd, entry, d).unwrap(),
            Some(expected),
            "doc {d} (id={id})"
        );
    }
}

#[test]
fn postings_term_vectors_and_doc_values_configured_together_all_write_correctly() {
    // Three independent write-side passes (postings, term vectors, doc
    // values) each patching the same `.si` after the fact -- proves
    // there's no ordering bug where a later writer's read-modify-write
    // clobbers an earlier one's file-list additions.
    let tmp = tempdir("postings-tv-and-dv-together");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        tv_body_field(1),
        numeric_field("score", 2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_doc_values_field(Some("score")).unwrap();

    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("the quick fox".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(10),
                },
            ],
        })
        .unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("b".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("the lazy fox".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(20),
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for (exts, format) in [
        (
            ["doc", "tim", "tip", "tmd"].as_slice(),
            Some(POSTINGS_FORMAT_NAME),
        ),
        (
            ["dvm", "dvd", "dvs"].as_slice(),
            Some(DOC_VALUES_FORMAT_NAME),
        ),
        // Term vectors are not a per-field format: plain segment name.
        (["tvd", "tvx", "tvm"].as_slice(), None),
    ] {
        for ext in exts {
            let name = match format {
                Some(f) => format!("{}.{ext}", per_field_segment(&sci.segment_name, f)),
                None => format!("{}.{ext}", sci.segment_name),
            };
            assert!(si.files.contains(&name), "missing {name} in .si files");
        }
    }

    // Doc-values side readable.
    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            stored_only_field("id", 0),
            tv_body_field(1),
            numeric_field("score", 2),
        ],
    };
    let (_, meta) = lucene_codecs::doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = meta.numeric_entry(2).unwrap();
    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 0).unwrap(),
        Some(10)
    );
    assert_eq!(
        lucene_codecs::doc_values::numeric_value(&dvd, entry, 1).unwrap(),
        Some(20)
    );

    // Term-vector side still readable too.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    assert!(reader.document(0).unwrap().is_some());
}

#[test]
fn omit_norms_field_rejects_an_unknown_field_name() {
    let tmp = tempdir("unknown-norms-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.omit_norms_field("nonexistent").unwrap_err();
    assert!(matches!(err, Error::UnknownNormsField(name) if name == "nonexistent"));
}

#[test]
fn omit_norms_field_rejects_a_field_with_no_index_options() {
    let tmp = tempdir("unindexed-norms-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    let err = writer.omit_norms_field("id").unwrap_err();
    assert!(matches!(err, Error::UnsupportedNormsField(name) if name == "id"));
}

/// Naming the same field twice is a no-op, and a field that already
/// declares `omit_norms` is accepted (Java's `setOmitNorms(true)` is
/// idempotent), so neither leaves a second column or an error behind.
#[test]
fn omit_norms_field_is_idempotent() {
    let tmp = tempdir("omit-norms-idempotent");
    let dir = FsDirectory::open(&tmp);
    let omitted = FieldInfo {
        omit_norms: true,
        ..body_field(1)
    };
    let fields = vec![stored_only_field("id", 0), omitted];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert!(writer.cfg.norms_field_configs().is_empty());
    writer.omit_norms_field("body").unwrap();
    writer.omit_norms_field("body").unwrap();
    assert!(writer.cfg.norms_field_configs().is_empty());
}

/// An indexed field the caller never named gets
/// norms, because that is what Lucene does
/// (`IndexingChain.writeNorms`: every `FieldInfo` with
/// `omitsNorms() == false && indexOptions != NONE`). Without them BM25
/// would score the field against a constant length instead of each
/// document's own.
#[test]
fn an_indexed_field_gets_norms_with_no_opt_in_at_all() {
    let tmp = tempdir("norms-by-default");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.add_document(doc_with_body("a", "fox")).unwrap();
    writer
        .add_document(doc_with_body("b", "the quick brown fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
    let body = fis.fields.iter().find(|f| f.name == "body").unwrap();
    assert!(
        !body.omit_norms,
        "an indexed field keeps the norms its caller asked for"
    );

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        si.files.iter().any(|f| f.ends_with(".nvm")),
        "norms files must have been written: {:?}",
        si.files
    );

    // And the values are the documents' own lengths, not a constant.
    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    let entry = meta.entry(1).unwrap();
    let d0 = norms::norm_value(&nvd, entry, 0).unwrap().unwrap();
    let d1 = norms::norm_value(&nvd, entry, 1).unwrap().unwrap();
    assert_ne!(d0, d1, "norms must vary with field length");
    assert_eq!(small_float::byte4_to_int(d0 as u8), 1);
    assert_eq!(small_float::byte4_to_int(d1 as u8), 4);
}

/// `omit_norms_field` is the opt-out, and it is the only norms knob.
#[test]
fn omit_norms_field_removes_the_column_and_says_so_in_the_fnm() {
    let tmp = tempdir("norms-opted-out");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.omit_norms_field("body").unwrap();
    writer.add_document(doc_with_body("a", "fox")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
    let body = fis.fields.iter().find(|f| f.name == "body").unwrap();
    assert!(body.omit_norms);

    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        !si.files.iter().any(|f| f.ends_with(".nvm")),
        "no norms files should have been written"
    );
}

/// A document that does **not** carry an indexed field gets no norm at
/// all, and one that carries it but tokenizes to nothing gets an
/// explicit `0` -- Java's `PerField.finish`, which runs only for a field
/// the document actually has and writes `0` for `invertState.length == 0`.
/// So an absent field and an empty one are distinguishable in the `.nvd`.
#[test]
fn a_document_without_the_field_gets_no_norm_and_an_empty_one_gets_zero() {
    let tmp = tempdir("norms-sparse");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    // doc 0 has `body`; doc 1 has no `body` field at all; doc 2 has an
    // empty `body`.
    writer
        .add_document(doc_with_body("a", "fox jumps"))
        .unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.add_document(doc_with_body("c", "")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    let entry = meta.entry(1).unwrap();
    assert_eq!(
        norms::norm_value(&nvd, entry, 0).unwrap().map(|v| v as u8),
        Some(small_float::int_to_byte4(2))
    );
    assert_eq!(
        norms::norm_value(&nvd, entry, 1).unwrap(),
        None,
        "a doc that does not carry the field has no norm"
    );
    assert_eq!(
        norms::norm_value(&nvd, entry, 2).unwrap(),
        Some(0),
        "a doc that carries an empty field has an explicit zero norm"
    );
}

#[test]
fn commit_writes_readable_length_dependent_norms_for_multiple_docs() {
    let tmp = tempdir("norms-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    // doc 0: "fox" -> length 1; doc 1: "the quick brown fox jumps" ->
    // length 5; doc 2: "fox fox fox" -> length 3 (repeated term still
    // counts every occurrence, not just distinct terms).
    writer.add_document(doc_with_body("a", "fox")).unwrap();
    writer
        .add_document(doc_with_body("b", "the quick brown fox jumps"))
        .unwrap();
    writer
        .add_document(doc_with_body("c", "fox fox fox"))
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1);
    let sci = &sis.segments[0];

    // Stored fields are still intact (backward-compatible).
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);

    // The norms files exist and are listed in `.si`.
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    for ext in ["nvm", "nvd"] {
        let name = format!("{}.{ext}", sci.segment_name);
        assert!(si.files.contains(&name), "missing {name} in .si files");
        assert!(
            dir.list_all().unwrap().contains(&name),
            "missing {name} on disk"
        );
    }

    // Readable via the existing, unmodified norms read side.
    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_version, parsed) = lucene_codecs::norms::parse_meta(&nvm, &sci.segment_id, "")
        .expect("parse_meta on IndexWriter-produced .nvm");
    let entry = parsed.entry(1).expect("field 1 (body) has a norms entry");
    assert!(entry.is_dense());

    let norm_a = lucene_codecs::norms::norm_value(&nvd, entry, 0)
        .unwrap()
        .unwrap();
    let norm_b = lucene_codecs::norms::norm_value(&nvd, entry, 1)
        .unwrap()
        .unwrap();
    let norm_c = lucene_codecs::norms::norm_value(&nvd, entry, 2)
        .unwrap()
        .unwrap();

    // Decode back to lengths via this crate's own BM25 read-side
    // decoder -- these are exact for small lengths (below the 24-value
    // subnormal threshold), so this also pins down the exact expected
    // byte, not just "some nonzero value".
    assert_eq!(lucene_util::small_float::byte4_to_int(norm_a as u8), 1);
    assert_eq!(lucene_util::small_float::byte4_to_int(norm_b as u8), 5);
    assert_eq!(lucene_util::small_float::byte4_to_int(norm_c as u8), 3);

    // The core proof this is real, length-dependent computation and not
    // a constant: three different lengths produce three different norm
    // bytes.
    assert_ne!(norm_a, norm_b);
    assert_ne!(norm_b, norm_c);
    assert_ne!(norm_a, norm_c);
}

/// The inverse of `an_indexed_field_gets_norms_with_no_opt_in_at_all`:
/// once every indexed field has opted out, the commit is stored-only
/// again and writes no `.nvm`/`.nvd` at all.
#[test]
fn commit_with_every_field_opted_out_of_norms_stays_stored_only() {
    let tmp = tempdir("no-norms-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.omit_norms_field("body").unwrap();

    writer.add_document(doc_with_body("a", "fox")).unwrap();
    let sis = writer.commit().unwrap().clone();
    let sci = &sis.segments[0];

    assert!(!dir
        .list_all()
        .unwrap()
        .contains(&format!("{}.nvm", sci.segment_name)));
    assert!(!dir
        .list_all()
        .unwrap()
        .contains(&format!("{}.nvd", sci.segment_name)));
}

#[test]
fn commit_with_norms_but_no_pending_docs_writes_no_norms_files() {
    let tmp = tempdir("norms-commit-no-docs");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.commit().unwrap();
    assert!(dir
        .list_all()
        .unwrap()
        .iter()
        .all(|f| !f.ends_with(".nvm") && !f.ends_with(".nvd")));
}

// ---------------------------------------------------------------------
// File lifecycle: `IndexFileDeleter`.
//
// Every assertion below is on the *directory listing*, not on the writer's
// in-memory state. An orphan file is by definition something no in-memory
// structure points at, so only `list_all()` can prove it is gone.
// ---------------------------------------------------------------------

/// `dir.list_all()` filtered to Lucene index files, sorted -- the shape
/// every lifecycle assertion below compares.
fn index_files(dir: &FsDirectory) -> Vec<String> {
    let mut files: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|f| crate::index_file_deleter::is_index_file_name(f))
        .collect();
    files.sort();
    files
}

/// A prepared commit has already flushed its segment to disk and written a
/// `pending_segments_N`. Rolling it back must leave the directory exactly as
/// it was before the prepare, with none of those files left behind.
#[test]
fn rollback_after_a_prepared_commit_deletes_every_file_that_prepare_wrote() {
    let tmp = tempdir("deleter-rollback-prepare");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    writer.add_document(doc_with_body("a", "kept")).unwrap();
    writer.commit().unwrap();
    let after_first_commit = index_files(&dir);

    writer
        .add_document(doc_with_body("b", "rolled back"))
        .unwrap();
    writer.prepare_commit().unwrap();
    let after_prepare = index_files(&dir);
    assert!(
        after_prepare.len() > after_first_commit.len(),
        "prepare must have written the segment's files and a pending commit: {after_prepare:?}"
    );
    assert!(after_prepare
        .iter()
        .any(|f| f.starts_with("pending_segments")));

    writer.rollback();

    assert_eq!(
        index_files(&dir),
        after_first_commit,
        "rollback must leave exactly the files the last commit references"
    );
    // And the surviving commit is still readable.
    let sis = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(read_all_docs(&dir, &sis), vec!["a"]);
}

/// Each commit supersedes the previous `segments_N`; the default
/// `KeepOnlyLastCommitDeletionPolicy` deletes it.
#[test]
fn each_commit_deletes_the_segments_file_it_supersedes() {
    let tmp = tempdir("deleter-superseded-commit");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    assert!(index_files(&dir).contains(&"segments_1".to_string()));

    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();

    let files = index_files(&dir);
    assert!(files.contains(&"segments_2".to_string()));
    assert!(
        !files.contains(&"segments_1".to_string()),
        "the superseded commit generation must be deleted: {files:?}"
    );
    // Exactly one commit file at any time under the default policy.
    assert_eq!(
        files.iter().filter(|f| f.starts_with("segments")).count(),
        1,
        "{files:?}"
    );
    // Both segments' data files survive -- only the commit file died.
    let sis = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(sis.segments.len(), 2);
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b"]);
}

/// `NoDeletionPolicy`: every generation stays, which is the property a
/// replication setup relies on.
#[test]
fn the_keep_all_deletion_policy_retains_every_commit_generation() {
    let tmp = tempdir("deleter-keep-all");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer
        .set_deletion_policy(crate::index_file_deleter::DeletionPolicy::KeepAll)
        .unwrap();

    for id in ["a", "b", "c"] {
        writer.add_document(doc(id)).unwrap();
        writer.commit().unwrap();
    }

    let files = index_files(&dir);
    for name in ["segments_1", "segments_2", "segments_3"] {
        assert!(
            files.contains(&name.to_string()),
            "{name} missing: {files:?}"
        );
    }

    // Switching back to the default reclaims them immediately, which is
    // `IndexFileDeleter.revisitPolicy()`.
    writer
        .set_deletion_policy(crate::index_file_deleter::DeletionPolicy::KeepOnlyLastCommit)
        .unwrap();
    let files = index_files(&dir);
    assert!(!files.contains(&"segments_1".to_string()), "{files:?}");
    assert!(!files.contains(&"segments_2".to_string()), "{files:?}");
    assert!(files.contains(&"segments_3".to_string()), "{files:?}");
}

/// A merge's source segments are unreferenced the instant the commit that
/// replaced them supersedes the previous one -- their files must go.
#[test]
fn an_automatic_merge_deletes_its_source_segments_files() {
    let tmp = tempdir("deleter-merge-sources");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), body_field(1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    for (id, text) in [
        ("a", "shared apple"),
        ("b", "shared banana"),
        ("c", "shared cherry"),
    ] {
        writer.add_document(doc_with_body(id, text)).unwrap();
        writer.commit().unwrap();
    }

    let segments = writer.segment_infos().segments.clone();
    assert_eq!(segments.len(), 1, "the tight policy must have merged");
    let merged = &segments[0].segment_name;

    let files = index_files(&dir);
    for source in ["_0", "_1", "_2"] {
        assert!(
            !files
                .iter()
                .any(|f| f.starts_with(&format!("{source}."))
                    || f.starts_with(&format!("{source}_"))),
            "merge source {source}'s files must be reclaimed: {files:?}"
        );
    }
    assert!(
        files.iter().any(|f| f.starts_with(&format!("{merged}."))),
        "the merged segment's own files must survive: {files:?}"
    );
    // The merged segment is still fully readable (merge order is the
    // merge policy's, not insertion order, so compare as a set).
    let sis = segment_infos::read_latest(&dir).unwrap();
    let mut docs = read_all_docs(&dir, &sis);
    docs.sort();
    assert_eq!(docs, vec!["a", "b", "c"]);
}

/// A second delete round writes `_0_2.liv` and abandons `_0_1.liv`. The
/// superseded generation must be deleted, not accumulated.
#[test]
fn a_second_delete_round_deletes_the_live_docs_generation_it_supersedes() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir("deleter-liv-generations");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = writer_seeded_with_fixture(&dir, &fx, fields);

    let first_round = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    let sis = writer
        .delete_documents_with_sources(&first_round, "body", b"cat")
        .unwrap()
        .clone();
    let files = index_files(&dir);
    assert!(files.contains(&"_0_1.liv".to_string()), "{files:?}");

    // The second round has to build on the first round's live docs, exactly
    // as a reader-pool-backed caller would -- otherwise `apply_deletes`
    // rejects the round for a del_count that does not match the bitset.
    let liv = dir.open("_0_1.liv").unwrap().to_vec();
    let live_docs = lucene_codecs::live_docs::parse(
        &liv,
        &fx.segment_id,
        1,
        fx.max_doc,
        sis.segments[0].del_count as usize,
    )
    .unwrap();
    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: Some(&live_docs),
        max_doc: fx.max_doc,
    }];

    writer
        .delete_documents_with_sources(&sources, "body", b"dog")
        .unwrap();
    let files = index_files(&dir);
    assert!(files.contains(&"_0_2.liv".to_string()), "{files:?}");
    assert!(
        !files.contains(&"_0_1.liv".to_string()),
        "the superseded .liv generation must be deleted: {files:?}"
    );
}

/// Opening a writer over a directory a crashed session left behind reclaims
/// its orphans: a `pending_segments_N` from a prepare that never finished,
/// and the segment files of a flush no commit ever named. Files that do not
/// look like Lucene index files are never touched.
#[test]
fn opening_a_writer_reclaims_the_orphans_a_crashed_session_left() {
    let tmp = tempdir("deleter-open-sweep");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    {
        let mut writer = IndexWriter::open(&dir, fields.clone(), "Lucene104", version()).unwrap();
        writer.add_document(doc("a")).unwrap();
        writer.commit().unwrap();
    }
    let committed = index_files(&dir);

    // Simulate a crash: a half-finished prepare and an uncommitted flush's
    // files, plus one file that is none of this port's business.
    write_file(&dir, "pending_segments_7", b"garbage").unwrap();
    write_file(&dir, "_9.fdt", b"orphaned flush").unwrap();
    write_file(&dir, "_9_Lucene104_0.tim", b"orphaned flush").unwrap();
    std::fs::write(tmp.join("README.txt"), b"not ours").unwrap();

    let writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    assert_eq!(
        index_files(&dir),
        committed,
        "every orphan must be reclaimed at open, and nothing else touched"
    );
    assert!(
        tmp.join("README.txt").exists(),
        "a non-index file is not ours to delete"
    );

    // `inflateGens`: the reclaimed names must never be handed out again.
    assert!(
        writer.segment_infos().counter > 9,
        "counter must be inflated past the orphaned segment name, got {}",
        writer.segment_infos().counter
    );
    assert!(
        writer.segment_infos().generation >= 7,
        "generation must be inflated past the orphaned pending commit, got {}",
        writer.segment_infos().generation
    );
}

// ---------------------------------------------------------------------
// RAM accounting and the automatic flush trigger.
// ---------------------------------------------------------------------

#[test]
fn ram_bytes_used_tracks_the_buffered_documents_and_resets_on_flush() {
    let tmp = tempdir("ram-accounting");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    assert_eq!(writer.ram_bytes_used(), 0);
    writer.add_document(doc("a")).unwrap();
    let one = writer.ram_bytes_used();
    assert!(one > 0, "a buffered document must cost something");

    writer
        .add_document(doc("a much longer identifier value than the first one"))
        .unwrap();
    let two = writer.ram_bytes_used();
    assert!(
        two > 2 * one,
        "the counter must reflect the actual string bytes, not a per-doc constant: \
             {one} then {two}"
    );

    writer.commit().unwrap();
    assert_eq!(writer.ram_bytes_used(), 0, "a flush resets the counter");
}

/// `FlushByRamOrCountsPolicy` with buffered deletes: deletes alone at the
/// RAM buffer are applied at once (their `.liv` written with no flush),
/// and deletes count toward the buffer with the documents.
#[test]
fn buffered_deletes_are_weighed_against_the_ram_buffer() {
    let tmp = tempdir("delete-ram");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![FieldInfo {
        index_options: lucene_codecs::field_infos::IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
    let mut w = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    for k in 0..10 {
        w.add_document(doc(&format!("d{k}"))).unwrap();
    }
    w.commit().unwrap();
    let mb = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let nb = DeleteNode::terms(vec![Term::new("id", "d0")]).ram_bytes();
    w.set_ram_buffer_size_mb(mb(nb * 3)).unwrap();
    let liv = || dir.list_all().unwrap().iter().any(|f| f.ends_with(".liv"));
    w.delete_documents_by_term(&[Term::new("id", "d0")])
        .unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d1")])
        .unwrap();
    assert!(!liv());
    assert_eq!(w.delete_queue.global_ram_bytes(), nb * 2);
    w.delete_documents_by_term(&[Term::new("id", "d2")])
        .unwrap();
    assert!(
        liv(),
        "the third delete reached the buffer and applied all three"
    );
    assert_eq!(w.delete_queue.global_ram_bytes(), 0);

    // One document under the buffer, two deletes under it: the second
    // document is under it alone but not with the deletes.
    let d = document_ram_bytes(&doc("n0"));
    assert!(d < nb * 2, "the deletes decide: {d} vs {nb}");
    w.set_ram_buffer_size_mb(mb(d + nb * 2)).unwrap();
    w.add_document(doc("n0")).unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d3")])
        .unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d4")])
        .unwrap();
    assert_eq!(w.pending_docs.len(), 1);
    w.add_document(doc("n1")).unwrap();
    assert!(w.pending_docs.is_empty(), "documents and deletes filled it");
    assert_eq!(w.delete_queue.global_ram_bytes(), 0);
    w.commit().unwrap();
    assert_eq!(
        crate::segment_infos::read_latest(&dir)
            .unwrap()
            .segments
            .iter()
            .map(|s| s.del_count)
            .sum::<i32>(),
        5
    );
}

#[test]
fn max_buffered_docs_flushes_a_segment_without_committing_it() {
    let tmp = tempdir("max-buffered-docs");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_max_buffered_docs(2).unwrap();

    writer.add_document(doc("a")).unwrap();
    assert_eq!(writer.pending_doc_count(), 1);
    writer.add_document(doc("b")).unwrap();
    assert_eq!(
        writer.pending_doc_count(),
        0,
        "the second document must have tripped the flush"
    );

    // Flushed, but not committed: a fresh reader sees nothing yet.
    let files = index_files(&dir);
    assert!(files.iter().any(|f| f.ends_with(".fdt")), "{files:?}");
    assert!(
        !files.iter().any(|f| f.starts_with("segments")),
        "an automatic flush must not publish a commit: {files:?}"
    );

    writer.add_document(doc("c")).unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(
        sis.segments.len(),
        2,
        "the auto-flushed segment and the final one must both be in the commit"
    );
    assert_eq!(read_all_docs(&dir, &sis), vec!["a", "b", "c"]);
}

/// The whole point of the flush trigger: peak memory stops growing with the
/// number of documents added between commits.
#[test]
fn the_ram_buffer_bounds_the_buffer_no_matter_how_many_documents_arrive() {
    let tmp = tempdir("ram-buffer-bound");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    // Small enough that a few hundred short documents cross it.
    writer.set_ram_buffer_size_mb(0.01).unwrap();

    let mut peak = 0usize;
    for i in 0..2_000 {
        writer.add_document(doc(&format!("doc-{i}"))).unwrap();
        peak = peak.max(writer.ram_bytes_used());
    }
    assert!(
        peak < 2 * (0.01 * 1024.0 * 1024.0) as usize,
        "peak buffered bytes must stay near the configured limit, got {peak}"
    );

    let sis = writer.commit().unwrap().clone();
    assert!(
        sis.segments.len() > 1,
        "the run must have auto-flushed more than once, got {} segment(s)",
        sis.segments.len()
    );
    let docs = read_all_docs(&dir, &sis);
    assert_eq!(docs.len(), 2_000, "no document may be lost across flushes");
    assert_eq!(docs[0], "doc-0");
    assert_eq!(docs[1_999], "doc-1999");
}

/// A rollback must discard automatically flushed segments too -- and delete
/// their files, since no commit ever named them.
#[test]
fn rollback_discards_auto_flushed_segments_and_deletes_their_files() {
    let tmp = tempdir("rollback-auto-flush");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    writer.add_document(doc("kept")).unwrap();
    writer.commit().unwrap();
    let after_commit = index_files(&dir);

    writer.set_max_buffered_docs(2).unwrap();
    writer.add_document(doc("x")).unwrap();
    writer.add_document(doc("y")).unwrap();
    assert!(
        index_files(&dir).len() > after_commit.len(),
        "the auto-flush must have written a segment"
    );

    writer.rollback();
    assert_eq!(
        index_files(&dir),
        after_commit,
        "rollback must reclaim an auto-flushed but uncommitted segment"
    );

    let sis = writer.commit().unwrap().clone();
    assert_eq!(read_all_docs(&dir, &sis), vec!["kept"]);
}

#[test]
fn the_auto_flush_setters_port_javas_validation_exactly() {
    let tmp = tempdir("auto-flush-validation");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    assert_eq!(writer.ram_buffer_size_mb(), DEFAULT_RAM_BUFFER_SIZE_MB);
    assert_eq!(writer.max_buffered_docs(), DEFAULT_MAX_BUFFERED_DOCS);
    assert_eq!(DEFAULT_MAX_BUFFERED_DOCS, DISABLE_AUTO_FLUSH);

    // "ramBufferSize should be > 0.0 MB when enabled"
    assert!(matches!(
        writer.set_ram_buffer_size_mb(0.0).unwrap_err(),
        Error::InvalidRamBufferSize(_)
    ));
    assert!(matches!(
        writer.set_ram_buffer_size_mb(-2.0).unwrap_err(),
        Error::InvalidRamBufferSize(_)
    ));
    // "maxBufferedDocs must at least be 2 when enabled"
    assert!(matches!(
        writer.set_max_buffered_docs(1).unwrap_err(),
        Error::InvalidMaxBufferedDocs(1)
    ));
    assert!(matches!(
        writer.set_max_buffered_docs(0).unwrap_err(),
        Error::InvalidMaxBufferedDocs(0)
    ));

    // "at least one of ramBufferSize and maxBufferedDocs must be enabled"
    assert!(matches!(
        writer
            .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
            .unwrap_err(),
        Error::BothAutoFlushTriggersDisabled
    ));
    writer.set_max_buffered_docs(10).unwrap();
    // ...now that doc-count flushing is on, disabling RAM flushing is legal.
    writer
        .set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB)
        .unwrap();
    assert!(matches!(
        writer
            .set_max_buffered_docs(DISABLE_AUTO_FLUSH)
            .unwrap_err(),
        Error::BothAutoFlushTriggersDisabled
    ));
}

/// `IndexWriter.deleteUnusedFiles()` is the fallible way to force the sweep
/// `rollback()` performs silently.
#[test]
fn delete_unused_files_reclaims_an_orphan_written_behind_the_writers_back() {
    let tmp = tempdir("delete-unused-files");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_document(doc("a")).unwrap();
    writer.commit().unwrap();
    let committed = index_files(&dir);

    write_file(&dir, "_ff.fdt", b"orphan").unwrap();
    assert!(index_files(&dir).contains(&"_ff.fdt".to_string()));

    writer.delete_unused_files().unwrap();
    assert_eq!(index_files(&dir), committed);
}

// ---------------------------------------------------------------
// Sequence numbers and buffered deletes.
//
// These tests exist to prove the three properties
// `crate::buffered_updates` claims, not merely that the methods run:
//   1. every mutating call returns a strictly larger seqNo,
//   2. a delete reaches every document buffered before it and none
//      buffered after it, and
//   3. a delete reaches every segment flushed before it and none
//      flushed after it.
// ---------------------------------------------------------------

/// A `body` field with real postings, so the writer can resolve a delete
/// term against segments it wrote itself -- which is the whole point:
/// without postings there is no term dictionary to resolve against and
/// every delete is a silent no-op.
fn seq_fields() -> Vec<FieldInfo> {
    vec![stored_only_field("id", 0), body_field(1)]
}

fn seq_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
    let mut writer = IndexWriter::open(dir, seq_fields(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
}

fn body_term(text: &str) -> Term {
    Term::new("body", text.as_bytes())
}

/// **Nothing the writer keeps per segment outlives the segment**, over
/// hundreds of flushes, deletes, commits and automatic merges.
///
/// The time-based soak's replacement for the writer: it found two
/// per-segment caches that were never pruned (the deleter's recorded
/// file sets and `segment_versions`), each growing by one entry per
/// segment ever written. Here every round must leave them, and the
/// directory, holding only what the live commit and the unflushed view
/// need: an entry kept for a dead segment fails the round it appears in,
/// not after a week.
#[test]
fn per_segment_state_and_files_track_the_live_segments_over_many_rounds() {
    let tmp = tempdir("resource-bounds");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.set_merge_policy(Some(tight_merge_policy()));
    let mut merged = 0;
    for round in 0..400 {
        for k in 0..3 {
            let n = round * 3 + k;
            writer
                .add_document(doc_with_body(&n.to_string(), &format!("t{n}")))
                .unwrap();
        }
        // A delete into an older segment: new `.liv` generations, whose
        // predecessors must go too.
        if round >= 5 {
            let old = (round - 5) * 3;
            writer
                .delete_documents_by_term(&[body_term(&format!("t{old}"))])
                .unwrap();
        }
        if round % 3 == 2 {
            writer.flush().unwrap(); // a segment no commit has seen yet
            continue;
        }
        let before = writer.segment_infos().segments.len();
        writer.commit().unwrap();
        if writer.segment_infos().segments.len() < before + 1 {
            merged += 1;
        }

        let live: std::collections::HashSet<String> = writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        let stale: Vec<&String> = writer
            .segment_versions
            .keys()
            .filter(|k| !live.contains(*k))
            .collect();
        assert!(
            stale.is_empty(),
            "round {round}: versions kept for dead segments {stale:?}"
        );
        let (segments, files) = writer.deleter.tracked_sizes();
        assert!(
            segments <= live.len(),
            "round {round}: the deleter tracks {segments} segments, {} are live",
            live.len()
        );
        let on_disk: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|f| crate::index_file_deleter::is_index_file_name(f))
            .collect();
        let orphans: Vec<&String> = on_disk
            .iter()
            .filter(|f| {
                !f.starts_with("segments")
                    && !live.contains(crate::index_file_deleter::parse_segment_name(f))
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "round {round}: files of dead segments {orphans:?}"
        );
        // Exactly what the deleter references: a superseded `.liv`
        // generation of a live segment, which this loop's deletes write
        // every round, would be left behind with nothing naming it.
        let mut disk_sorted = on_disk.clone();
        disk_sorted.sort();
        assert_eq!(files, disk_sorted.len(), "round {round}");
        assert_eq!(
            writer.deleter.referenced_files(),
            disk_sorted,
            "round {round}"
        );
        // One commit point: old `segments_N` and `.liv` generations go.
        let commits = on_disk
            .iter()
            .filter(|f| f.starts_with("segments_"))
            .count();
        assert_eq!(commits, 1, "round {round}: {on_disk:?}");
    }
    assert!(merged > 50, "the merge policy must have merged: {merged}");
}

/// Every live document's `id`, in commit order.
fn visible_ids(dir: &FsDirectory, infos: &SegmentInfos) -> Vec<String> {
    read_all_docs(dir, infos)
}

/// **`IndexWriter` drops a 100%-deleted segment when the deletes are
/// applied, and real Lucene says so**.
///
/// `IndexWriter.finishApply` removes from the in-memory `SegmentInfos`
/// every segment `closeSegmentStates` found fully deleted
/// (`rld.isFullyDeleted() && mergePolicy.keepFullyDeletedSegment(..) ==
/// false`, where `PendingDeletes.isFullyDeleted` is
/// `getDelCount() == maxDoc()` -- hard deletes only). This port kept such
/// a segment in the commit forever: a segment nothing can ever match,
/// carried by every later open, merge and `CheckIndex`.
///
/// Ground truth is `fixtures/src/GenFullyDeletedDrop.java`, which runs the
/// same four scripts through a real `IndexWriter` and records the
/// committed segment count, each segment's `(maxDoc, delCount)`, and the
/// visible ids. No index is committed: the bytes of an index with a
/// dropped segment are indistinguishable from those of one that never had
/// it, so the *behaviour* is what there is to record.
///
/// `partial` is the control -- a segment with one of two documents deleted
/// survives -- so this cannot pass by dropping every segment with deletes.
#[test]
fn a_fully_deleted_segment_is_dropped_exactly_where_real_lucene_drops_it() {
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/fully_deleted_drop/manifest.properties"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenFullyDeletedDrop first");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };

    let scenarios: Vec<String> = get("scenarios").split(',').map(str::to_string).collect();
    assert_eq!(scenarios, ["drop", "partial", "all", "block"]);

    for name in &scenarios {
        let tmp = tempdir(&format!("fully-deleted-{name}"));
        let dir = FsDirectory::open(&tmp);
        let mut writer = seq_writer(&dir);
        match name.as_str() {
            "drop" => {
                writer.add_document(doc_with_body("a", "shared")).unwrap();
                writer.add_document(doc_with_body("b", "shared")).unwrap();
                writer.flush().unwrap();
                writer
                    .delete_documents_by_term(&[body_term("shared")])
                    .unwrap();
                writer.add_document(doc_with_body("c", "other")).unwrap();
                writer.add_document(doc_with_body("d", "other")).unwrap();
            }
            "partial" => {
                writer.add_document(doc_with_body("a", "shared")).unwrap();
                writer.add_document(doc_with_body("b", "kept")).unwrap();
                writer.flush().unwrap();
                writer
                    .delete_documents_by_term(&[body_term("shared")])
                    .unwrap();
                writer.add_document(doc_with_body("c", "other")).unwrap();
                writer.add_document(doc_with_body("d", "other")).unwrap();
            }
            "all" => {
                writer.add_document(doc_with_body("a", "shared")).unwrap();
                writer.add_document(doc_with_body("b", "shared")).unwrap();
                writer.flush().unwrap();
                writer.add_document(doc_with_body("c", "shared")).unwrap();
                writer.add_document(doc_with_body("d", "shared")).unwrap();
                writer.flush().unwrap();
                writer
                    .delete_documents_by_term(&[body_term("shared")])
                    .unwrap();
            }
            "block" => {
                writer
                    .add_documents(vec![doc_with_body("p1", "key"), doc_with_body("c1", "key")])
                    .unwrap();
                writer.commit().unwrap();
                writer
                    .update_documents(
                        body_term("key"),
                        vec![doc_with_body("p2", "key"), doc_with_body("c2", "key")],
                    )
                    .unwrap();
            }
            other => panic!("unknown scenario {other}"),
        }
        let infos = writer.commit().unwrap().clone();

        let expected_count: usize = get(&format!("{name}.segment_count")).parse().unwrap();
        assert_eq!(
            infos.segments.len(),
            expected_count,
            "{name}: real Lucene commits {expected_count} segment(s), this port \
                 {}",
            infos.segments.len()
        );

        let expected_shape = get(&format!("{name}.segment_shape"));
        let got_shape = infos
            .segments
            .iter()
            .map(|sci| {
                let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
                let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
                format!("{}:{}", si.doc_count, sci.del_count)
            })
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            got_shape, expected_shape,
            "{name}: per-segment (maxDoc, delCount)"
        );

        let mut expected_ids: Vec<String> = get(&format!("{name}.visible_ids"))
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        expected_ids.sort();
        let mut got_ids = visible_ids(&dir, &infos);
        got_ids.sort();
        assert_eq!(got_ids, expected_ids, "{name}: visible documents");
    }
}

/// [`MergePolicyConfig::keep_fully_deleted_segments`] is
/// `MergePolicy.keepFullyDeletedSegment`, and it is the whole reason the
/// drop is not unconditional: `SoftDeletesRetentionMergePolicy` returns
/// `true` from it, so a writer retaining soft-deleted documents must be
/// able to keep a segment the *hard* delete count has written off.
#[test]
fn keep_fully_deleted_segments_suppresses_the_drop() {
    let tmp = tempdir("keep-fully-deleted");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.set_merge_policy(Some(MergePolicyConfig {
        keep_fully_deleted_segments: true,
        // A merge would drop the emptied segment by the other route
        // (`execute_merge`'s zero-live-document check), so keep the policy
        // from proposing one.
        max_merged_segment_size: 0,
        ..MergePolicyConfig::default()
    }));
    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.add_document(doc_with_body("b", "shared")).unwrap();
    writer.flush().unwrap();
    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer.add_document(doc_with_body("c", "other")).unwrap();
    let infos = writer.commit().unwrap().clone();

    assert_eq!(
        infos.segments.len(),
        2,
        "the emptied segment must be kept when the policy says so"
    );
    assert_eq!(infos.segments[0].del_count, 2);
    assert_eq!(visible_ids(&dir, &infos), vec!["c"]);
}

#[test]
fn every_mutating_method_returns_a_strictly_increasing_sequence_number() {
    let tmp = tempdir("seqno-monotonic");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    let mut seqs = vec![writer.add_document(doc_with_body("a", "alpha")).unwrap()];
    seqs.push(
        writer
            .add_documents(vec![doc_with_body("b", "beta"), doc_with_body("c", "beta")])
            .unwrap(),
    );
    seqs.push(
        writer
            .update_document(body_term("alpha"), doc_with_body("a2", "alpha"))
            .unwrap(),
    );
    seqs.push(
        writer
            .update_documents(body_term("beta"), vec![doc_with_body("b2", "beta")])
            .unwrap(),
    );
    seqs.push(
        writer
            .delete_documents_by_term(&[body_term("gamma")])
            .unwrap(),
    );
    seqs.push(
        writer
            .delete_documents_by_query(&[DeleteQuery::Term(body_term("delta"))])
            .unwrap(),
    );
    seqs.push(
        writer
            .add_document_with_custom_freq_terms(doc_with_body("d", "delta"), Vec::new())
            .unwrap(),
    );

    // Java: "seqNo must start at 1 because some APIs negate this to also
    // return a boolean".
    assert_eq!(seqs[0], 1);
    assert!(
        seqs.windows(2).all(|w| w[0] < w[1]),
        "sequence numbers must be strictly increasing: {seqs:?}"
    );

    // The doc-values half of the surface needs a writer with doc-values
    // fields, so it gets its own writer -- but the same assertion, because
    // the property is about the whole mutating surface, not about one
    // configuration of it.
    let tmp2 = tempdir("seqno-monotonic-dv");
    let dir2 = FsDirectory::open(&tmp2);
    let mut dvw = dv_writer(&dir2);
    let mut dv_seqs = vec![dvw.add_document(doc_with_body("a", "alpha")).unwrap()];
    dv_seqs.push(
        dvw.update_numeric_doc_value(body_term("alpha"), "soft", 1)
            .unwrap(),
    );
    dv_seqs.push(
        dvw.update_binary_doc_value(body_term("alpha"), "payload", b"v")
            .unwrap(),
    );
    dv_seqs.push(
        dvw.update_doc_values(
            body_term("alpha"),
            &[DocValuesUpdate::Numeric {
                term: body_term("alpha"),
                field: "soft".into(),
                value: None,
            }],
        )
        .unwrap(),
    );
    dv_seqs.push(
        dvw.soft_update_document(
            body_term("alpha"),
            doc_with_body("b", "alpha"),
            &[DocValuesUpdate::Numeric {
                term: body_term("alpha"),
                field: "soft".into(),
                value: Some(1),
            }],
        )
        .unwrap(),
    );
    dv_seqs.push(
        dvw.soft_update_documents(
            body_term("alpha"),
            vec![doc_with_body("c", "alpha"), doc_with_body("d", "alpha")],
            &[DocValuesUpdate::Numeric {
                term: body_term("alpha"),
                field: "soft".into(),
                value: Some(1),
            }],
        )
        .unwrap(),
    );
    assert_eq!(dv_seqs[0], 1);
    assert!(
        dv_seqs.windows(2).all(|w| w[0] < w[1]),
        "sequence numbers must be strictly increasing: {dv_seqs:?}"
    );

    // `deleteDocuments(MatchAllDocsQuery)` short-circuits to `deleteAll()`
    // (LUCENE-6379) and still consumes a number, so it belongs here too.
    let tmp3 = tempdir("seqno-monotonic-matchall");
    let dir3 = FsDirectory::open(&tmp3);
    let mut w3 = seq_writer(&dir3);
    let first = w3.add_document(doc_with_body("a", "x")).unwrap();
    let match_all = w3
        .delete_documents_by_query(&[DeleteQuery::MatchAll])
        .unwrap();
    let after = w3.add_document(doc_with_body("b", "x")).unwrap();
    assert!(first < match_all && match_all < after);
}

#[test]
fn a_document_block_takes_exactly_one_sequence_number() {
    // Java's `updateDocuments(Node, docs)` returns one seqNo for the whole
    // block -- it is one operation, seen whole or not at all.
    let tmp = tempdir("seqno-block");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    let first = writer.add_document(doc_with_body("a", "x")).unwrap();
    let block = writer
        .add_documents(vec![
            doc_with_body("b", "x"),
            doc_with_body("c", "x"),
            doc_with_body("d", "x"),
        ])
        .unwrap();
    let after = writer.add_document(doc_with_body("e", "x")).unwrap();

    assert_eq!(first, 1);
    assert_eq!(block, 2, "a three-document block is one operation");
    assert_eq!(after, 3);
}

#[test]
fn a_rollback_never_reissues_a_sequence_number() {
    // Java's `rollbackInternal` builds a *fresh* delete queue whose seqNos
    // continue past the aborted ones.
    let tmp = tempdir("seqno-rollback");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    let before = writer.add_document(doc_with_body("a", "x")).unwrap();
    writer.rollback();
    let after = writer.add_document(doc_with_body("b", "x")).unwrap();
    assert!(after > before, "{after} must be past {before}");
}

#[test]
fn interleaved_adds_updates_and_deletes_produce_the_expected_visible_set() {
    // The ordering under test, all inside one segment:
    //   #1 add   a (doc 0)
    //   #2 add   b (doc 1)
    //   #3 update body:alpha -> a2   (deletes doc 0, adds doc 2)
    //   #4 add   c (doc 3)
    //   #5 delete body:beta          (deletes doc 1, limit 4)
    // Both deletes carry the buffer position they were issued at, so
    // neither can reach a document added after it.
    let tmp = tempdir("seqno-interleaved");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.add_document(doc_with_body("b", "beta")).unwrap();
    writer
        .update_document(body_term("alpha"), doc_with_body("a2", "alpha"))
        .unwrap();
    writer.add_document(doc_with_body("c", "gamma")).unwrap();
    writer
        .delete_documents_by_term(&[body_term("beta")])
        .unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a2", "c"]);
}

#[test]
fn a_delete_does_not_reach_a_document_added_after_it_in_the_same_segment() {
    // `docIDUpto`: the delete is issued when one document is buffered, so
    // it may clear doc 0 and nothing above it -- even though the document
    // added afterwards carries the very same term.
    let tmp = tempdir("seqno-docidupto");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer
        .add_document(doc_with_body("first", "shared"))
        .unwrap();
    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer
        .add_document(doc_with_body("second", "shared"))
        .unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["second"]);
}

#[test]
fn a_delete_issued_before_any_document_exists_deletes_nothing() {
    let tmp = tempdir("seqno-empty-delete");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer.add_document(doc_with_body("a", "shared")).unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
}

#[test]
fn a_delete_applies_to_the_segments_flushed_before_it_and_not_to_the_ones_after() {
    // The cross-segment half of the contract, and the one that is easy to
    // get subtly wrong: `_0` is already on disk when the delete is issued,
    // so it must be cleared; `_1` is flushed afterwards from documents
    // carrying the *same* term, so it must survive untouched.
    let tmp = tempdir("seqno-across-flush");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.set_max_buffered_docs(2).unwrap();

    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.add_document(doc_with_body("b", "shared")).unwrap();
    // The threshold tripped: `_0` exists on disk now.
    assert_eq!(writer.pending_doc_count(), 0);

    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();

    writer.add_document(doc_with_body("c", "shared")).unwrap();
    writer.add_document(doc_with_body("d", "shared")).unwrap();

    let infos = writer.commit().unwrap().clone();
    // `_0` ends up 100% deleted, so `IndexWriter.finishApply` drops it
    // (`drop_fully_deleted_segments`) -- and its absence is itself the
    // proof that the delete reached *both* of its documents. `_1` must
    // still be here, untouched: a delete that wrongly reached it too would
    // leave no segments at all.
    assert_eq!(
        infos
            .segments
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect::<Vec<_>>(),
        vec!["_1"],
        "`_0` fully deleted and dropped, `_1` survives"
    );
    assert_eq!(visible_ids(&dir, &infos), vec!["c", "d"]);
    assert_eq!(infos.segments[0].del_count, 0);
    assert!(
        !index_files(&dir).iter().any(|f| f.starts_with("_0.")),
        "a dropped segment's files are reclaimed at the commit: {:?}",
        index_files(&dir)
    );
}

#[test]
fn a_delete_issued_before_a_flush_still_reaches_the_segment_that_flush_produces() {
    // The mirror image: the delete is buffered while the documents are
    // still in RAM, so the segment the *next* flush produces must come out
    // with those documents already dead.
    let tmp = tempdir("seqno-before-flush");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.add_document(doc_with_body("b", "keep")).unwrap();
    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].del_count, 1);
    assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
}

#[test]
fn a_delete_issued_after_a_commit_reaches_the_committed_segment() {
    let tmp = tempdir("seqno-after-commit");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.add_document(doc_with_body("b", "keep")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
}

#[test]
fn update_document_replaces_across_a_flush_boundary() {
    // `updateDocument` is a delete of everything that came before plus an
    // add: the replacement lands in the new segment, the original dies in
    // the committed one.
    let tmp = tempdir("seqno-update-across");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("v1", "key")).unwrap();
    writer.commit().unwrap();

    writer
        .update_document(body_term("key"), doc_with_body("v2", "key"))
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["v2"]);
}

#[test]
fn two_updates_of_the_same_term_in_one_buffer_leave_only_the_newest() {
    // `BufferedUpdates.addTerm` keeps the *higher* `docIDUpto`, which is
    // exactly what makes the second update delete the first update's
    // replacement as well as the original.
    let tmp = tempdir("seqno-double-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("v1", "key")).unwrap();
    writer
        .update_document(body_term("key"), doc_with_body("v2", "key"))
        .unwrap();
    writer
        .update_document(body_term("key"), doc_with_body("v3", "key"))
        .unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["v3"]);
}

#[test]
fn a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list() {
    // The failure this pins: applying a buffered delete bumps a *committed*
    // segment's `del_gen` in the writer's in-memory view and writes a new
    // `.liv` that no commit names yet. `rollback()` then runs the deleter's
    // `refresh()`, which reclaims exactly such an unreferenced file. Without
    // `SegmentInfos.rollbackSegmentInfos(rollbackSegments)` the in-memory
    // view would survive the rollback pointing at a deleted file, and the
    // next commit would write a `segments_N` naming it.
    let tmp = tempdir("seqno-rollback-restores");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "shared")).unwrap();
    // A second document the delete does not match, so the segment is not
    // left 100% deleted: `finishApply`'s drop would otherwise remove it
    // before the rollback, and this test is about the rollback restoring a
    // segment whose `.liv` generation advanced, not about the drop.
    writer.add_document(doc_with_body("keep", "other")).unwrap();
    writer.commit().unwrap();
    let committed_del_gen = writer.segment_infos().segments[0].del_gen;
    assert_eq!(committed_del_gen, -1);

    // Force the delete to be resolved and written *before* the rollback.
    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer.flush().unwrap();
    assert_eq!(writer.segment_infos().segments[0].del_gen, 1);
    assert!(index_files(&dir).contains(&"_0_1.liv".to_string()));

    writer.rollback();

    // The in-memory view is back to the commit...
    assert_eq!(writer.segment_infos().segments[0].del_gen, -1);
    assert_eq!(writer.segment_infos().segments[0].del_count, 0);
    // ...the orphaned `.liv` is gone...
    assert!(!index_files(&dir).contains(&"_0_1.liv".to_string()));
    // ...and the next commit names only files that exist.
    let infos = writer.commit().unwrap().clone();
    for sci in &infos.segments {
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        for f in sci.files(&si.files) {
            assert!(index_files(&dir).contains(&f), "commit names missing {f}");
        }
    }
    assert_eq!(visible_ids(&dir, &infos), vec!["a", "keep"]);
}

/// End to end for `SegmentCommitInfo.generationAdvanced()`: the id a
/// commit records for a segment must change when that segment changes.
/// Read back out of `segments_N`, not off the in-memory writer, because
/// the commit file is what a replication or NRT consumer actually
/// compares.
#[test]
fn a_delete_changes_the_segment_commit_id_the_next_commit_records() {
    let tmp = tempdir("sci-id-changes-across-commits");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "keep")).unwrap();
    writer.add_document(doc_with_body("b", "doomed")).unwrap();
    writer.commit().unwrap();

    let before = segment_infos::read_latest(&dir).unwrap().segments[0].sci_id;
    assert!(before.is_some());

    writer
        .delete_documents_by_term(&[body_term("doomed")])
        .unwrap();
    writer.commit().unwrap();

    let after = segment_infos::read_latest(&dir).unwrap().segments[0].sci_id;
    assert!(after.is_some());
    assert_ne!(
        after, before,
        "a deleted document left the segment-commit id unchanged"
    );
}

#[test]
fn a_rollback_discards_buffered_deletes_along_with_buffered_documents() {
    let tmp = tempdir("seqno-rollback-deletes");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer.rollback();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
}

// --- delete-by-query ---

#[test]
fn delete_documents_by_query_resolves_a_term_query() {
    let tmp = tempdir("query-delete-term");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.add_document(doc_with_body("b", "beta")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_query(&[DeleteQuery::Term(body_term("alpha"))])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
}

#[test]
fn delete_documents_by_query_resolves_a_prefix_and_stops_at_the_span_end() {
    let tmp = tempdir("query-delete-prefix");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "apple")).unwrap();
    writer.add_document(doc_with_body("b", "apricot")).unwrap();
    writer.add_document(doc_with_body("c", "banana")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_query(&[DeleteQuery::Prefix {
            field: "body".into(),
            prefix: b"ap".to_vec(),
        }])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["c"]);
}

#[test]
fn delete_documents_by_query_resolves_an_inclusive_and_an_exclusive_range() {
    let tmp = tempdir("query-delete-range");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    for (id, body) in [("a", "aaa"), ("b", "bbb"), ("c", "ccc"), ("d", "ddd")] {
        writer.add_document(doc_with_body(id, body)).unwrap();
    }
    writer.commit().unwrap();

    // [bbb, ddd) -> bbb and ccc.
    writer
        .delete_documents_by_query(&[DeleteQuery::TermRange {
            field: "body".into(),
            lower: Some(b"bbb".to_vec()),
            upper: Some(b"ddd".to_vec()),
            include_lower: true,
            include_upper: false,
        }])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a", "d"]);
}

#[test]
fn delete_documents_by_query_resolves_boolean_composition() {
    let tmp = tempdir("query-delete-boolean");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer
        .add_document(doc_with_body("a", "red round"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "red square"))
        .unwrap();
    writer
        .add_document(doc_with_body("c", "blue round"))
        .unwrap();
    writer.commit().unwrap();

    // red AND round -> only "a".
    writer
        .delete_documents_by_query(&[DeleteQuery::All(vec![
            DeleteQuery::Term(body_term("red")),
            DeleteQuery::Term(body_term("round")),
        ])])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["b", "c"]);
}

#[test]
fn delete_documents_by_query_resolves_a_negation_over_live_docs_only() {
    let tmp = tempdir("query-delete-not");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "keep")).unwrap();
    writer.add_document(doc_with_body("b", "drop")).unwrap();
    writer.add_document(doc_with_body("c", "drop")).unwrap();
    writer.commit().unwrap();

    // NOT keep -> b and c.
    writer
        .delete_documents_by_query(&[DeleteQuery::Not(Box::new(DeleteQuery::Term(body_term(
            "keep",
        ))))])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a"]);
}

#[test]
fn delete_documents_by_query_resolves_a_union() {
    let tmp = tempdir("query-delete-any");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.add_document(doc_with_body("b", "beta")).unwrap();
    writer.add_document(doc_with_body("c", "gamma")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_query(&[DeleteQuery::Any(vec![
            DeleteQuery::Term(body_term("alpha")),
            DeleteQuery::Term(body_term("gamma")),
        ])])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
}

#[test]
fn delete_documents_by_query_specialises_match_all_into_delete_all() {
    // LUCENE-6379: a `MatchAllDocsQuery` short-circuits to `deleteAll()`,
    // which drops whole segments instead of writing an all-zero `.liv`.
    let tmp = tempdir("query-delete-matchall");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_query(&[DeleteQuery::MatchAll])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert!(infos.segments.is_empty(), "deleteAll drops the segments");
    assert!(visible_ids(&dir, &infos).is_empty());
}

#[test]
fn a_query_delete_honours_the_doc_id_upto_limit_within_its_own_segment() {
    let tmp = tempdir("query-delete-limit");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    writer
        .add_document(doc_with_body("first", "shared"))
        .unwrap();
    writer
        .delete_documents_by_query(&[DeleteQuery::Term(body_term("shared"))])
        .unwrap();
    writer
        .add_document(doc_with_body("second", "shared"))
        .unwrap();

    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["second"]);
}

// --- document blocks / `hasBlocks` ---

fn parsed_si(dir: &FsDirectory, sci: &SegmentCommitInfo) -> segment_info::SegmentInfo {
    let bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    segment_info::parse(&bytes, &sci.segment_id).unwrap()
}

#[test]
fn a_block_add_sets_has_blocks_on_the_flushed_segment() {
    let tmp = tempdir("block-has-blocks");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer
        .add_documents(vec![
            doc_with_body("parent", "p"),
            doc_with_body("child1", "c"),
            doc_with_body("child2", "c"),
        ])
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
    // Contiguous, ascending doc IDs in the order they were supplied.
    assert_eq!(
        visible_ids(&dir, &infos),
        vec!["parent", "child1", "child2"]
    );
}

#[test]
fn single_document_adds_leave_has_blocks_unset() {
    let tmp = tempdir("block-no-blocks");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "x")).unwrap();
    writer.add_document(doc_with_body("b", "x")).unwrap();
    // A one-document "block" is not a block, matching Java's `numDocs > 1`.
    writer.add_documents(vec![doc_with_body("c", "x")]).unwrap();
    let infos = writer.commit().unwrap().clone();

    assert!(!parsed_si(&dir, &infos.segments[0]).has_blocks);
}

#[test]
fn has_blocks_does_not_leak_from_one_flush_into_the_next() {
    let tmp = tempdir("block-not-leaked");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer
        .add_documents(vec![doc_with_body("p", "p"), doc_with_body("c", "c")])
        .unwrap();
    writer.commit().unwrap();
    writer.add_document(doc_with_body("plain", "x")).unwrap();
    let infos = writer.commit().unwrap().clone();

    assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
    assert!(!parsed_si(&dir, &infos.segments[1]).has_blocks);
}

#[test]
fn an_automatic_flush_never_splits_a_document_block() {
    // The contiguity guarantee: the threshold is consulted once per
    // `add_documents` call, so a block that crosses it still lands whole.
    let tmp = tempdir("block-not-split");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.set_max_buffered_docs(2).unwrap();

    writer
        .add_documents(vec![
            doc_with_body("p", "p"),
            doc_with_body("c1", "c"),
            doc_with_body("c2", "c"),
            doc_with_body("c3", "c"),
        ])
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    assert_eq!(infos.segments.len(), 1, "the block must not be split");
    assert_eq!(visible_ids(&dir, &infos), vec!["p", "c1", "c2", "c3"]);
}

#[test]
fn update_documents_deletes_the_old_block_and_adds_the_new_one_atomically() {
    let tmp = tempdir("block-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer
        .add_documents(vec![doc_with_body("p1", "key"), doc_with_body("c1", "key")])
        .unwrap();
    writer.commit().unwrap();

    writer
        .update_documents(
            body_term("key"),
            vec![doc_with_body("p2", "key"), doc_with_body("c2", "key")],
        )
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["p2", "c2"]);
    // The old block was the whole of its segment, so applying the delete
    // leaves it 100% deleted and `finishApply` drops it -- the replacement
    // is the only segment left.
    assert_eq!(infos.segments.len(), 1, "the emptied segment is dropped");
    assert!(parsed_si(&dir, &infos.segments[0]).has_blocks);
}

// --- doc-values updates and soft deletes ---

fn numeric_dv_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::Numeric,
        ..stored_only_field(name, number)
    }
}

fn binary_dv_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        doc_values_type: DocValuesType::Binary,
        ..stored_only_field(name, number)
    }
}

fn dv_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
    let fields = vec![
        stored_only_field("id", 0),
        body_field(1),
        numeric_dv_field("soft", 2),
        binary_dv_field("payload", 3),
    ];
    let mut writer = IndexWriter::open(dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
}

/// A writer whose `soft` field actually has a base doc-values column, so
/// an update has something to supersede. `dv_writer` deliberately does
/// not: a field that never had a column in the segment is Java's
/// `FieldInfos.FieldNumbers.constructFieldInfo` case, and both are worth
/// exercising.
fn dv_base_writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
    let mut writer = dv_writer(dir);
    writer.set_doc_values_field(Some("soft")).unwrap();
    writer
}

fn doc_with_dv(id: &str, body: &str, soft: i64) -> Document {
    let mut doc = doc_with_body(id, body);
    doc.fields.push(StoredField {
        field_number: 2,
        value: FieldValue::Long(soft),
    });
    doc
}

/// Opens `field_number`'s **current** doc-values column exactly the way
/// `SegmentDocValuesProducer` does: the generation
/// `SegmentCommitInfo.dv_update_files` names for the field, read with the
/// generation-suffixed index header those files carry, against the
/// generational `.fnm`'s one-field `FieldInfos`. Returns `None` when the
/// field has no update generation at all.
fn open_dv_generation(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    field_number: i32,
) -> Option<(lucene_codecs::doc_values::DocValuesMeta, Vec<u8>, i32)> {
    let (_, files) = sci
        .dv_update_files
        .iter()
        .find(|(n, _)| *n == field_number)?;
    let dvm_name = files.iter().find(|f| f.ends_with(".dvm")).unwrap().clone();
    let dvd_name = files.iter().find(|f| f.ends_with(".dvd")).unwrap().clone();
    let suffix = dvm_name
        .strip_prefix(&format!("{}_", sci.segment_name))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap()
        .to_string();

    let fnm_name = sci
        .field_infos_files
        .first()
        .expect("a doc-values update always writes a FieldInfos generation");
    let fnm = dir.open(fnm_name).unwrap();
    let infos = lucene_codecs::field_infos::parse(
        &fnm,
        &sci.segment_id,
        &lucene_util::base36::to_base36(sci.field_infos_gen),
    )
    .unwrap();
    let field = infos
        .fields
        .iter()
        .find(|f| f.number == field_number)
        .unwrap()
        .clone();
    assert_ne!(
        field.doc_values_gen, -1,
        "the generational .fnm must record the field's docValuesGen"
    );
    let only = lucene_codecs::field_infos::FieldInfos {
        fields: vec![field],
    };

    let dvm = dir.open(&dvm_name).unwrap();
    let dvd = dir.open(&dvd_name).unwrap();
    let (_, meta) =
        lucene_codecs::doc_values::parse_meta(&dvm, &sci.segment_id, &suffix, &only).unwrap();
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    Some((meta, dvd.to_vec(), si.doc_count))
}

/// The whole NUMERIC column a doc-values update generation left behind,
/// one entry per doc. `None` means the doc has no value -- a `reset`, or a
/// doc the base never had a value for.
pub(crate) fn read_numeric_column(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    field_number: i32,
) -> Vec<Option<i64>> {
    let Some((meta, data, max_doc)) = open_dv_generation(dir, sci, field_number) else {
        return Vec::new();
    };
    let entry = meta.numeric_entry(field_number).unwrap();
    (0..max_doc)
        .map(|doc| lucene_codecs::doc_values::numeric_value(&data, entry, doc).unwrap())
        .collect()
}

/// [`read_numeric_column`] for BINARY doc values.
fn read_binary_column(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    field_number: i32,
) -> Vec<Option<Vec<u8>>> {
    let Some((meta, data, max_doc)) = open_dv_generation(dir, sci, field_number) else {
        return Vec::new();
    };
    let entry = meta.binary_entry(field_number).unwrap();
    (0..max_doc)
        .map(|doc| {
            lucene_codecs::doc_values::binary_value(&data, entry, doc)
                .unwrap()
                .map(<[u8]>::to_vec)
        })
        .collect()
}

/// **`has_blocks` survives an automatic merge.** It is one byte in the
/// `.si` that nothing else in the write path sets, and a merged segment
/// holding blocks but reporting `hasBlocks = false` reads back perfectly
/// while silently invalidating every parent/child join query against it.
/// Java's `IndexWriter.mergeMiddle` ORs it across `merge.segments`.
#[test]
fn an_automatic_merge_keeps_has_blocks() {
    let tmp = tempdir("has-blocks-merge");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));
    // Two block-bearing commits and one plain one, so the merge has to
    // OR the flag rather than copy the first source's.
    writer.add_documents(vec![doc("a1"), doc("a2")]).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc("b")).unwrap();
    writer.commit().unwrap();
    writer.add_documents(vec![doc("c1"), doc("c2")]).unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "the three segments must have merged"
    );
    let sci = &infos.segments[0];
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        si.has_blocks,
        "the merged .si must still report that it holds document blocks"
    );
    // Each block is still contiguous and in order, which is what the flag
    // promises. (The *segments* may be merged in whatever order the
    // policy proposed them, so the block positions are not fixed.)
    let docs = read_all_docs(&dir, &infos);
    assert_eq!(docs.len(), 5);
    for (first, second) in [("a1", "a2"), ("c1", "c2")] {
        let i = docs.iter().position(|d| d == first).unwrap();
        assert_eq!(
            docs.get(i + 1).map(String::as_str),
            Some(second),
            "block {first}/{second} was split by the merge: {docs:?}"
        );
    }
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// **Positional postings survive an automatic merge.** `execute_merge`
/// filtered its merge-time postings fields to `Docs`/`DocsAndFreqs` back
/// when that was all this writer could flush; once positional flush
/// landed, that filter meant a merged segment carried no postings for the
/// field at all while its `.fnm` still declared
/// `DocsAndFreqsAndPositions` -- an indexed field with no registered
/// postings producer, which reads back as having no terms and raises
/// nothing.
#[test]
fn an_automatic_merge_carries_positional_postings_through() {
    let tmp = tempdir("positional-postings-merge");
    let dir = FsDirectory::open(&tmp);
    let positions_body_field = fi::FieldInfo {
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        ..stored_only_field("body", 1)
    };
    let fields = vec![stored_only_field("id", 0), positions_body_field.clone()];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    for (id, body) in [
        ("a", "quick fox jumps"),
        ("b", "the fox sleeps"),
        ("c", "a fox again"),
    ] {
        writer.add_document(doc_with_body(id, body)).unwrap();
        writer.commit().unwrap();
    }
    let infos = writer.segment_infos().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "the three segments must have merged"
    );
    let sci = &infos.segments[0];

    let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        si.files.contains(&format!("{seg}.pos")),
        "the merged segment must carry a .pos"
    );
    let tim = dir.open(&format!("{seg}.tim")).unwrap();
    let tip = dir.open(&format!("{seg}.tip")).unwrap();
    let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
    let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
    let pos_bytes = dir.open(&format!("{seg}.pos")).unwrap();
    let field_infos = fi::FieldInfos {
        fields: vec![
            fi::FieldInfo {
                index_options: IndexOptions::None,
                ..stored_only_field("id", 0)
            },
            positions_body_field,
        ],
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        3,
    )
    .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pos_in = lucene_codecs::postings::PosInput::open(
        &pos_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field = block_fields.field("body").unwrap();
    // "fox" is at position 1 in every one of the three documents, which
    // are now three merged doc ids of one segment.
    let fox = field
        .positions(b"fox", Some(&doc_in), &pos_in, None)
        .unwrap()
        .expect("the merged dictionary must still have \"fox\"");
    assert_eq!(fox.len(), 3);
    for doc_positions in &fox {
        assert_eq!(doc_positions.len(), 1);
        assert_eq!(doc_positions[0].position, 1);
    }
    // ...and a document-unique term still lands on its own document.
    let quick = field
        .positions(b"quick", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(quick.len(), 1);
    assert_eq!(quick[0][0].position, 0);
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// **The other four doc-values types survive an automatic merge too.**
/// `execute_merge` must build every doc-values type's merge sources, since
/// `segment_stats` offers every `.dvd`-bearing segment; otherwise a BINARY (or
/// SORTED / SORTED_NUMERIC / SORTED_SET) column is dropped by the merge
/// and `describe_written_files` then zeroed its `DocValuesType` to keep
/// the `.fnm` honest -- a valid, `CheckIndex`-clean segment with the data
/// gone. This asserts on the *values*, per document, rather than on the merge
/// having happened.
#[test]
fn an_automatic_merge_carries_every_doc_values_type_through() {
    let tmp = tempdir("dv-types-and-merge-policy");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        numeric_dv_field("num", 1),
        binary_dv_field("bin", 2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();
    writer.add_doc_values_field("bin").unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    for (id, num) in [("a", 1i64), ("b", 2), ("c", 3)] {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(id.to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(num),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String(format!("payload-{id}")),
                    },
                ],
            })
            .unwrap();
        writer.commit().unwrap();
    }

    let infos = writer.segment_infos().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "the three segments must have merged"
    );
    let sci = &infos.segments[0];
    let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
    let dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap().to_vec();
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &lucene_codecs::field_infos::FieldInfos {
            fields: vec![
                stored_only_field("id", 0),
                numeric_dv_field("num", 1),
                binary_dv_field("bin", 2),
            ],
        },
    )
    .unwrap();
    let numeric = meta.numeric_entry(1).expect("the NUMERIC column survived");
    let binary = meta.binary_entry(2).expect("the BINARY column survived");

    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    for d in 0..3 {
        let id = match &reader.document(d).unwrap().fields[0].value {
            FieldValue::String(v) => v.clone(),
            other => panic!("unexpected stored value {other:?}"),
        };
        let expected_num = match id.as_str() {
            "a" => 1,
            "b" => 2,
            "c" => 3,
            other => panic!("unexpected id {other}"),
        };
        assert_eq!(
            doc_values::numeric_value(&dvd, numeric, d).unwrap(),
            Some(expected_num),
            "num at doc {d} (id={id})"
        );
        assert_eq!(
            doc_values::binary_value(&dvd, binary, d).unwrap(),
            Some(format!("payload-{id}").as_bytes()),
            "bin at doc {d} (id={id})"
        );
    }
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// **A sort tier none of the merged documents has a value for.** The
/// merged column is all-missing, which
/// `Lucene90DocValuesConsumer.writeValues` writes as
/// `docsWithFieldOffset = -2` rather than omitting the field. Omitting it
/// would zero the field's `DocValuesType` in the merged `.fnm` while the
/// merged `.si` still declares the sort over it: real Lucene's
/// `DocValues.getNumeric` *throws* for a field whose `FieldInfo` exists
/// but declares no doc values, so `CheckIndex.testSort` would fail rather
/// than degrade -- and this port could never merge that segment again,
/// because `read_sort_keys` would return `MergeSortColumnMissing` out of
/// every subsequent `commit`.
/// A merge source none of whose documents has the sort field -- its
/// `.fnm` does not list it, as for a segment of delete tombstones only
/// -- sorts every document as missing instead of failing the merge.
#[test]
fn a_source_without_the_sort_field_sorts_as_missing() {
    let columns = SourceDocValueColumns {
        columns: Vec::new(),
        per_field: Vec::new(),
    };
    let numeric = sort_field("rank", false, SortMissingValue::Last);
    assert_eq!(
        IndexWriter::read_sort_keys(&columns, &[], &numeric, 3).unwrap(),
        vec![None; 3]
    );
    let mut terms = numeric.clone();
    terms.kind = segment_info::IndexSortKind::String(segment_info::StringMissingValue::Last);
    assert_eq!(
        IndexWriter::read_sort_terms(&columns, &[], &terms, 2).unwrap(),
        vec![None; 2]
    );
}

#[test]
fn a_sort_tier_no_merged_document_has_a_value_for_is_still_written() {
    let tmp = tempdir("sorted-merge-empty-tier");
    let dir = FsDirectory::open(&tmp);
    let mut writer = sorted_merge_writer(
        &dir,
        &[
            sort_field("rank", false, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ],
    );
    writer.set_merge_policy(None);
    // `tie` is never set on any document, so the second tier's merged
    // column has no values at all.
    for batch in [vec![("b", 20i64)], vec![("a", 10)], vec![("c", 30)]] {
        for (id, rank) in batch {
            let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
                .collect::<Vec<_>>()
                .join(" ");
            let mut document = sortable_doc(id, rank, 0, &body);
            // Drop the `tie` stored field, so no document has a value for
            // the second sort tier.
            document.fields.retain(|f| f.field_number != 2);
            writer
                .add_document_with_vectors(
                    document,
                    vec![DocumentVector::float32(
                        "v",
                        vec![rank as f32, 1.0, 2.0, 3.0],
                    )],
                )
                .unwrap();
        }
        writer.commit().unwrap();
    }

    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    writer.execute_merge(&names).unwrap();
    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];

    // The merged `.fnm` still declares `tie` as NUMERIC doc values, and
    // the merged `.dvm` has an entry for it with no values.
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let merged_fields = fi::parse(&fnm, &sci.segment_id, "").unwrap();
    let tie = merged_fields.field_by_number(2).unwrap();
    assert_eq!(tie.doc_values_type, DocValuesType::Numeric);
    assert_eq!(read_base_numeric_column(&dir, sci, 2, 3), vec![None; 3]);
    assert_eq!(
        read_base_numeric_column(&dir, sci, 1, 3),
        vec![Some(10), Some(20), Some(30)]
    );
    assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c"]);
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    // ...and the segment can be merged again, which is the half that
    // would have failed with `MergeSortColumnMissing`.
    writer
        .add_document_with_vectors(
            {
                let mut d = sortable_doc("d", 40, 0, "td td td td");
                d.fields.retain(|f| f.field_number != 2);
                d
            },
            vec![DocumentVector::float32("v", vec![40.0, 1.0, 2.0, 3.0])],
        )
        .unwrap();
    writer.commit().unwrap();
    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    writer.execute_merge(&names).unwrap();
    assert_eq!(
        read_all_docs(&dir, writer.segment_infos()),
        vec!["a", "b", "c", "d"]
    );
}

/// A doc-values update against a segment produced by a **merge**, not by
/// a flush. Worth its own case because the merged `.fnm` is written by
/// `merge::describe_written_files`, which zeroes
/// `doc_values_type` for a field the merge wrote no column for -- the
/// exact state `field_updates::check_updatable` would otherwise reject.
/// If the two disagreed, the update would fail on a
/// merged segment while working on a flushed one.
#[test]
fn a_doc_values_update_still_works_against_a_merged_segment() {
    let tmp = tempdir("dv-update-after-merge");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_base_writer(&dir);
    writer.set_merge_policy(Some(tight_merge_policy()));
    for (id, body, soft) in [("a", "alpha", 1i64), ("b", "beta", 2), ("c", "gamma", 3)] {
        let mut document = doc_with_body(id, body);
        document.fields.push(StoredField {
            field_number: 2,
            value: FieldValue::Long(soft),
        });
        writer.add_document(document).unwrap();
        writer.commit().unwrap();
    }
    let infos = writer.segment_infos().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "the three segments must have merged"
    );
    assert_eq!(infos.segments[0].doc_values_gen, -1, "no generation yet");

    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 42)
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(infos.segments[0].doc_values_gen, 1);
    // The generation supersedes only the updated document; the merged
    // base column still answers for the other two.
    let sci = &infos.segments[0];
    let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
    let base_dvm = dir.open(&format!("{seg}.dvm")).unwrap().to_vec();
    assert!(!base_dvm.is_empty(), "the merge wrote a base column");
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn update_numeric_doc_value_writes_a_generation_the_reader_can_replay() {
    let tmp = tempdir("dv-numeric-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.add_document(doc_with_body("b", "beta")).unwrap();
    writer.commit().unwrap();

    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 42)
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    let sci = &infos.segments[0];
    assert_eq!(sci.doc_values_gen, 1, "one generation was written");
    assert_eq!(
        sci.field_infos_gen, 1,
        "a doc-values update also writes a FieldInfos generation"
    );
    assert_eq!(sci.del_count, 0, "a doc-values update never deletes");
    // Lucene's own file names, base-36 generation and per-field suffix.
    let files: Vec<&String> = sci
        .dv_update_files
        .iter()
        .find(|(n, _)| *n == 2)
        .map(|(_, f)| f.iter().collect())
        .expect("field 2 has an update generation");
    assert_eq!(
        files.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
        vec![
            "_0_1_Lucene90_0.dvm",
            "_0_1_Lucene90_0.dvd",
            "_0_1_Lucene90_0.dvs"
        ]
    );
    assert_eq!(sci.field_infos_files, vec!["_0_1.fnm".to_string()]);
    assert_eq!(read_numeric_column(&dir, sci, 2), vec![Some(42), None]);
    // The documents themselves are untouched.
    assert_eq!(visible_ids(&dir, &infos), vec!["a", "b"]);
}

#[test]
fn update_doc_values_with_a_null_value_records_a_removal_not_a_zero() {
    // Java: "If a doc values fields data is null the existing value is
    // removed from all documents matching the term" -- which reaches
    // `DocValuesFieldUpdates.reset(doc)`, not `add(doc, 0)`.
    let tmp = tempdir("dv-reset");
    let dir = FsDirectory::open(&tmp);
    // A *base* column, so "removed" and "never had one" are actually
    // distinguishable in the rewritten generation.
    let mut writer = dv_base_writer(&dir);
    writer.add_document(doc_with_dv("a", "alpha", 7)).unwrap();
    writer.add_document(doc_with_dv("b", "beta", 8)).unwrap();
    writer.commit().unwrap();

    writer
        .update_doc_values(
            body_term("alpha"),
            &[DocValuesUpdate::Numeric {
                term: body_term("alpha"),
                field: "soft".into(),
                value: None,
            }],
        )
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(
        read_numeric_column(&dir, &infos.segments[0], 2),
        vec![None, Some(8)],
        "doc 0's value is removed (not zeroed), doc 1's base value survives"
    );
}

#[test]
fn update_binary_doc_value_writes_a_binary_generation() {
    let tmp = tempdir("dv-binary-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.commit().unwrap();

    writer
        .update_binary_doc_value(body_term("alpha"), "payload", b"hello")
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    let sci = &infos.segments[0];
    assert_eq!(
        read_binary_column(&dir, sci, 3),
        vec![Some(b"hello".to_vec())]
    );
}

#[test]
fn successive_doc_values_updates_supersede_each_other_at_a_new_generation() {
    // Lucene's format is a *full rewrite* per generation, not a stack of
    // deltas: generation 2 is the field's complete column, so generation
    // 1's files stop being referenced (and are reclaimed) the moment it
    // lands. What must survive is only the semantic contract -- the newest
    // write wins.
    let tmp = tempdir("dv-generations");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.commit().unwrap();

    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 1)
        .unwrap();
    writer.commit().unwrap();
    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 2)
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    let sci = &infos.segments[0];
    assert_eq!(sci.doc_values_gen, 2);
    assert_eq!(sci.field_infos_gen, 2);
    let (_, files) = sci.dv_update_files.iter().find(|(n, _)| *n == 2).unwrap();
    assert!(
        files.iter().all(|f| f.starts_with("_0_2_")),
        "only the newest generation stays referenced: {files:?}"
    );
    assert_eq!(read_numeric_column(&dir, sci, 2), vec![Some(2)]);
    assert!(
        !index_files(&dir).iter().any(|f| f.starts_with("_0_1_")),
        "the superseded generation must be reclaimed: {:?}",
        index_files(&dir)
    );
}

#[test]
fn update_doc_values_rejects_an_unknown_or_wrongly_typed_field() {
    let tmp = tempdir("dv-validation");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);

    let unknown = writer
        .update_numeric_doc_value(body_term("alpha"), "nope", 1)
        .unwrap_err();
    assert!(matches!(
        unknown,
        Error::UnknownDocValuesUpdateField(name) if name == "nope"
    ));

    // `payload` is BINARY, so a numeric update against it is refused --
    // Java's `verifyOrCreateDvOnlyField` throws for the same mismatch.
    let wrong_type = writer
        .update_numeric_doc_value(body_term("alpha"), "payload", 1)
        .unwrap_err();
    assert!(matches!(
        wrong_type,
        Error::WrongDocValuesUpdateType { ref field, .. } if field == "payload"
    ));

    let empty = writer
        .update_doc_values(body_term("alpha"), &[])
        .unwrap_err();
    assert!(matches!(empty, Error::NoDocValuesUpdatesSupplied));
}

/// **`CheckIndex` counts a soft delete that reached a segment as a
/// doc-values update.** Such a delete lands in a new `.fnm` and
/// `.dvm`/`.dvd` generation; the check read the base column instead,
/// counted only the soft deletes the segment was written with, and failed
/// a segment real Lucene's `CheckIndex` passes (the engine writer's
/// refresh-loop test found it: 4460 counted, 4500 recorded).
#[test]
fn check_index_counts_soft_deletes_applied_as_doc_values_updates() {
    let tmp = tempdir("soft-deletes-generation");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        body_field(1),
        FieldInfo {
            soft_deletes_field: true,
            ..numeric_dv_field("soft", 2)
        },
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    for id in ["a", "b", "c"] {
        writer
            .add_document(doc_with_body(id, &format!("k{id}")))
            .unwrap();
    }
    writer.commit().unwrap();
    let mark = |term: &str| {
        [DocValuesUpdate::Numeric {
            term: body_term(term),
            field: "soft".into(),
            value: Some(1),
        }]
    };
    writer
        .soft_update_document(body_term("ka"), doc_with_body("a2", "ka2"), &mark("ka"))
        .unwrap();
    writer
        .soft_update_document(body_term("kb"), doc_with_body("b2", "kb2"), &mark("kb"))
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    let first = &infos.segments[0];
    assert_eq!(first.soft_del_count, 2);
    assert_ne!(
        first.field_infos_gen, -1,
        "the marks must be a doc-values update"
    );
    let results = crate::check_index::check_directory(&dir).unwrap();
    let checked = results
        .iter()
        .find(|r| r.segment_name == first.segment_name)
        .unwrap();
    let soft = checked
        .checks
        .iter()
        .find(|c| c.name == "soft_deletes.count_matches")
        .unwrap();
    assert!(soft.passed(), "{soft:?}");
    assert!(checked.all_passed(), "{:?}", checked.failures());
}

#[test]
fn soft_update_document_adds_the_new_doc_and_marks_the_old_one_without_deleting_it() {
    // The whole point of a soft delete: the previous version keeps its
    // live bit and its postings, and is instead marked through a
    // doc-values field a retention policy can consult.
    let tmp = tempdir("soft-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.add_document(doc_with_body("v1", "key")).unwrap();
    writer.commit().unwrap();

    writer
        .soft_update_document(
            body_term("key"),
            doc_with_body("v2", "key"),
            &[DocValuesUpdate::Numeric {
                term: body_term("key"),
                field: "soft".into(),
                value: Some(1),
            }],
        )
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    // Nothing was hard-deleted anywhere.
    assert!(infos.segments.iter().all(|s| s.del_count == 0));
    assert_eq!(visible_ids(&dir, &infos), vec!["v1", "v2"]);
    // ...but the original is marked.
    assert_eq!(
        read_numeric_column(&dir, &infos.segments[0], 2),
        vec![Some(1)]
    );
    // ...and the replacement is not: the marking carries the buffer
    // position it was issued at, exactly like a hard delete would.
    assert!(
        infos.segments[1].dv_update_files.is_empty(),
        "the replacement must not mark itself: {:?}",
        infos.segments[1].dv_update_files
    );
}

#[test]
fn a_delete_after_a_rollback_still_reaches_every_committed_segment() {
    // Two committed segments carry `buffered_deletes_gen` 1 and 2. A
    // rollback must not rewind `BufferedUpdatesStream`'s counter to 1 (as
    // Java's `clear()` does -- safely, because Java then *closes* the
    // writer): the next packet would be stamped gen 1 and `applies_to`
    // would reject it for `_1`, so the delete would silently reach only
    // the oldest segment. One segment is not enough to see it -- gen 1 ==
    // gen 1 passes by luck -- which is why this test uses two.
    let tmp = tempdir("rollback-then-delete");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);

    // Each segment also carries a document the delete does not match, so
    // neither is left 100% deleted and `finishApply`'s drop does not
    // remove the very evidence this test reads (`del_count` per segment).
    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer
        .add_document(doc_with_body("a-keep", "other"))
        .unwrap();
    writer.commit().unwrap();
    writer.add_document(doc_with_body("b", "shared")).unwrap();
    writer
        .add_document(doc_with_body("b-keep", "other"))
        .unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 2);

    writer.rollback();

    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(
        visible_ids(&dir, &infos),
        vec!["a-keep", "b-keep"],
        "the delete must reach both segments, not just the oldest"
    );
    assert_eq!(
        infos
            .segments
            .iter()
            .map(|s| s.del_count)
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
}

#[test]
fn documents_and_deletes_buffered_during_a_prepared_commit_survive_into_the_next_one() {
    // The loss this pins: `finish_commit` installs the `SegmentInfos`
    // snapshot `prepare_commit` took and clears `flushed_segments`, so a
    // flush in between would have its segment *and* every buffered delete
    // it resolved thrown away. With `max_buffered_docs = 2`, the two adds
    // below trip the automatic flush inside the window.
    let tmp = tempdir("prepared-window");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.set_max_buffered_docs(2).unwrap();

    writer.add_document(doc_with_body("a", "shared")).unwrap();
    writer.commit().unwrap();

    // A change, so the prepare really holds a commit open (one with
    // nothing to commit prepares nothing, and defers nothing).
    writer.set_live_commit_data(vec![("k".into(), "v".into())]);
    writer.prepare_commit().unwrap();
    writer
        .delete_documents_by_term(&[body_term("shared")])
        .unwrap();
    writer.add_document(doc_with_body("b", "kept")).unwrap();
    writer.add_document(doc_with_body("c", "kept")).unwrap();
    // Deferred, not flushed: nothing may be published behind the prepare.
    assert_eq!(writer.pending_doc_count(), 2);

    // The prepared commit publishes exactly what it snapshotted...
    let after_finish = writer.finish_commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &after_finish), vec!["a"]);

    // ...and the next one carries everything buffered during the window.
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["b", "c"]);
}

#[test]
fn an_explicit_flush_is_refused_while_a_commit_is_prepared() {
    let tmp = tempdir("prepared-flush-refused");
    let dir = FsDirectory::open(&tmp);
    let mut writer = seq_writer(&dir);
    writer.add_document(doc_with_body("a", "x")).unwrap();
    writer.prepare_commit().unwrap();
    writer.add_document(doc_with_body("b", "x")).unwrap();

    let err = writer.flush().unwrap_err();
    assert!(matches!(err, Error::PreparedCommitPending("flush")));
    // Nothing was lost by the refusal.
    assert_eq!(writer.pending_doc_count(), 1);
    writer.finish_commit().unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(visible_ids(&dir, &infos), vec!["a", "b"]);
}

#[test]
fn every_generational_file_name_round_trips_through_parse_generation() {
    // The property that ties every generational name this writer produces
    // -- `field_updates`' `.dvm`/`.dvd`/`.dvs` and `.fnm`, and
    // `deletes::liv_file_name` -- to `IndexFileNames.parseGeneration`,
    // which is what `inflateGens` relies on after a crash. Checked across
    // the base-36 boundary (10, 35, 36) rather than at one value, because
    // decimal and base 36 agree below 10 -- which is exactly why writing
    // the generation in decimal went unnoticed until it was checked here.
    let per_field = per_field_codec_suffix(DOC_VALUES_FORMAT_NAME);
    for gen in 1..=100i64 {
        for ext in ["dvm", "dvd", "dvs"] {
            let dv = crate::field_updates::generation_file_name("_0", gen, &per_field, ext);
            assert_eq!(
                crate::index_file_deleter::parse_generation_for_test(&dv),
                gen,
                "doc-values generation name {dv} must read back as generation {gen}"
            );
        }
        let fnm = crate::field_updates::field_infos_gen_file_name("_0", gen);
        assert_eq!(
            crate::index_file_deleter::parse_generation_for_test(&fnm),
            gen,
            "field-infos generation name {fnm} must read back as generation {gen}"
        );
        let liv = deletes::liv_file_name("_0", gen);
        assert_eq!(
            crate::index_file_deleter::parse_generation_for_test(&liv),
            gen,
            "live-docs name {liv} must read back as generation {gen}"
        );
    }
}

#[test]
fn a_doc_values_update_skips_a_document_the_same_packet_just_deleted() {
    // Java runs `applyTermDeletes` -> `applyQueryDeletes` ->
    // `applyDocValuesUpdates` inside one `FrozenBufferedUpdates.apply`, and
    // the last of the three reads `segState.rld.getLiveDocs()` -- which by
    // then reflects the first two. So a document this packet just killed
    // takes no doc-values update. Here `a` is hard-deleted and separately
    // marked in the same buffer; only `b`'s mark may survive.
    let tmp = tempdir("dv-skips-deleted");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer
        .add_document(doc_with_body("a", "alpha shared"))
        .unwrap();
    writer
        .add_document(doc_with_body("b", "beta shared"))
        .unwrap();
    writer.commit().unwrap();

    writer
        .delete_documents_by_term(&[body_term("alpha")])
        .unwrap();
    writer
        .update_numeric_doc_value(body_term("shared"), "soft", 9)
        .unwrap();
    let infos = writer.commit().unwrap().clone();

    let sci = &infos.segments[0];
    assert_eq!(sci.del_count, 1, "only `a` is deleted");
    assert_eq!(
        read_numeric_column(&dir, sci, 2),
        vec![None, Some(9)],
        "doc 0 was deleted by this packet, so it takes no update"
    );
    assert_eq!(visible_ids(&dir, &infos), vec!["b"]);
}

#[test]
fn soft_update_document_requires_at_least_one_soft_delete_field() {
    let tmp = tempdir("soft-update-empty");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    let err = writer
        .soft_update_document(body_term("key"), doc_with_body("v", "key"), &[])
        .unwrap_err();
    assert!(matches!(err, Error::NoSoftDeletesSupplied));
}

/// **Merging a segment that carries a doc-values *update*.** Its newest
/// column lives in generational `.dvm`/`.dvd` files that no `.si` lists;
/// reading the base pair would silently resurrect the pre-update values
/// into a valid, `CheckIndex`-clean segment.
///
/// `execute_merge` resolves every field to its **current**
/// generation, through the same two functions the update path reads its
/// own base from. This asserts on the merged *values*, not on the merge
/// having happened: the failure mode is a merge that succeeds with the
/// old numbers.
///
/// Both shapes an update can take are exercised in one segment: `soft`
/// has **no base column at all** (the flush wrote no `.dvd`, so the
/// generation is the field's only column -- Java's
/// `FieldInfos.FieldNumbers.constructFieldInfo` case), while `payload`
/// has a base BINARY column this merge must carry forward untouched from
/// the base pair the same segment also has.
#[test]
fn a_segment_with_a_doc_values_update_merges_at_its_newest_generation() {
    let tmp = tempdir("dv-update-merges");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.set_doc_values_field(Some("payload")).unwrap();
    writer.set_merge_policy(None);
    writer
        .add_document({
            let mut doc = doc_with_body("a", "alpha");
            doc.fields.push(StoredField {
                field_number: 3,
                value: FieldValue::String("base-a".to_string()),
            });
            doc
        })
        .unwrap();
    writer.commit().unwrap();
    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 5)
        .unwrap();
    writer.commit().unwrap();

    // `soft` lives only in generation 1; `payload` only in the base pair.
    let updated = writer.segment_infos().segments[0].clone();
    assert_eq!(updated.doc_values_gen, 1);
    let si_bytes = dir.open(&format!("{}.si", updated.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &updated.segment_id).unwrap();
    assert!(
        !si.files
            .iter()
            .any(|f| f.ends_with(".dvd") && f.contains("_1_")),
        "the generational `.dvd` is not in the `.si`, which is the whole difficulty"
    );

    // A second segment, then a merge over both.
    writer.set_merge_policy(Some(tight_merge_policy()));
    for i in 0..2 {
        writer
            .add_document({
                let mut doc = doc_with_body(&format!("x{i}"), "beta");
                doc.fields.push(StoredField {
                    field_number: 3,
                    value: FieldValue::String(format!("base-x{i}")),
                });
                doc
            })
            .unwrap();
        writer.commit().unwrap();
    }

    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1, "the sources must have merged");
    let merged = &infos.segments[0];
    assert_eq!(
        merged.doc_values_gen, -1,
        "a merge folds every generation back into one base column"
    );

    // The merged columns live in the merged segment's *base*
    // `.dvm`/`.dvd`, not in a generation -- which is the claim.
    let seg = per_field_segment(&merged.segment_name, DOC_VALUES_FORMAT_NAME);
    let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &merged.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &fi::FieldInfos {
            fields: vec![
                stored_only_field("id", 0),
                body_field(1),
                numeric_dv_field("soft", 2),
                binary_dv_field("payload", 3),
            ],
        },
    )
    .unwrap();
    let soft = meta
        .numeric_entry(2)
        .expect("`soft`'s generational column reached the merged base column");
    let payload = meta.binary_entry(3).expect("`payload`'s base column too");

    // Resolve each merged doc id through its stored `id`, so this asserts
    // the pairing rather than an assumed merged order.
    let fdt = dir.open(&format!("{}.fdt", merged.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", merged.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", merged.segment_name)).unwrap();
    let stored = stored_fields::open(&fdt, &fdx, &fdm, &merged.segment_id, "").unwrap();
    assert_eq!(stored.max_doc(), 3);
    for doc_id in 0..3 {
        let id = doc_value(&stored.document(doc_id).unwrap());
        let expected_soft = if id == "a" { Some(5) } else { None };
        assert_eq!(
            doc_values::numeric_value(&dvd, soft, doc_id).unwrap(),
            expected_soft,
            "`soft` at doc {doc_id} (id={id}): the update's value, not the base's absence"
        );
        assert_eq!(
            doc_values::binary_value(&dvd, payload, doc_id).unwrap(),
            Some(format!("base-{id}").as_bytes()),
            "`payload` at doc {doc_id} (id={id}): carried from the base pair"
        );
    }
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn a_doc_values_update_generation_is_referenced_and_survives_the_deleters_sweep() {
    // The generation files are named in `segments_N` through
    // `dv_update_files` and `field_infos_files`, so
    // `SegmentCommitInfo::files()` -- and therefore the deleter -- must
    // see all four. None of them is listed in the `.si`: they did not
    // exist when it was written.
    let tmp = tempdir("dv-refcount");
    let dir = FsDirectory::open(&tmp);
    let mut writer = dv_writer(&dir);
    writer.add_document(doc_with_body("a", "alpha")).unwrap();
    writer.commit().unwrap();
    writer
        .update_numeric_doc_value(body_term("alpha"), "soft", 7)
        .unwrap();
    writer.commit().unwrap();

    let before = index_files(&dir);
    writer.delete_unused_files().unwrap();
    assert_eq!(index_files(&dir), before);
    for expected in [
        "_0_1_Lucene90_0.dvm",
        "_0_1_Lucene90_0.dvd",
        "_0_1_Lucene90_0.dvs",
        "_0_1.fnm",
    ] {
        assert!(
            before.iter().any(|f| f == expected),
            "{expected} must still be there: {before:?}"
        );
    }
}

// -----------------------------------------------------------------
// Vector fields (`set_vector_field` / `add_document_with_vectors`)
// -----------------------------------------------------------------

fn vector_field(
    name: &str,
    number: i32,
    dim: i32,
    encoding: VectorEncoding,
    similarity: VectorSimilarityFunction,
) -> FieldInfo {
    FieldInfo {
        vector_dimension: dim,
        vector_encoding: encoding,
        vector_similarity_function: similarity,
        ..stored_only_field(name, number)
    }
}

/// A deterministic, well-spread vector so a graph over these has real
/// structure rather than every node being equidistant.
fn test_vector(dim: usize, seed: i64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s as u64) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn open_vector_files(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let seg = per_field_segment(&sci.segment_name, KNN_VECTORS_FORMAT_NAME);
    (
        dir.open(&format!("{seg}.vec")).unwrap().to_vec(),
        dir.open(&format!("{seg}.vemf")).unwrap().to_vec(),
        dir.open(&format!("{seg}.vem")).unwrap().to_vec(),
        dir.open(&format!("{seg}.vex")).unwrap().to_vec(),
    )
}

/// The end-to-end shape: documents added with vectors produce a segment
/// whose four vector files decode back to exactly the vectors that went
/// in, dense and sparse, float and byte -- and whose graph is a real
/// graph, not an empty one.
#[test]
fn add_document_with_vectors_writes_a_readable_vector_segment() {
    let tmp = tempdir("vectors-round-trip");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "dense",
            1,
            8,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
        vector_field(
            "sparse_bytes",
            2,
            4,
            VectorEncoding::Byte,
            VectorSimilarityFunction::DotProduct,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("dense")).unwrap();
    writer.add_vector_field("sparse_bytes").unwrap();

    let n = 800;
    for i in 0..n {
        let mut vectors = vec![DocumentVector::float32("dense", test_vector(8, i as i64))];
        if i % 4 == 0 {
            vectors.push(DocumentVector::byte(
                "sparse_bytes",
                vec![(i % 251) as u8, 7, 200, (i / 3 % 251) as u8],
            ));
        }
        writer
            .add_document_with_vectors(doc(&format!("doc{i}")), vectors)
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    let (vec_bytes, vemf, vem, vex) = open_vector_files(&dir, sci);

    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let flat =
        vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
    let dense = flat.float_vector_values(1).unwrap();
    assert_eq!(dense.size(), n);
    assert_eq!(dense.dimension(), 8);
    assert_eq!(
        dense.similarity(),
        VectorSimilarityFunction::Euclidean,
        "the similarity must come from the FieldInfo, not a default"
    );
    for ord in 0..n {
        assert_eq!(dense.ord_to_doc(ord).unwrap(), ord, "dense ord == doc");
        assert_eq!(dense.vector(ord).unwrap(), test_vector(8, ord as i64));
    }

    let sparse = flat.byte_vector_values(2).unwrap();
    assert_eq!(sparse.size(), n / 4);
    for ord in 0..sparse.size() {
        let doc_id = ord * 4;
        assert_eq!(sparse.ord_to_doc(ord).unwrap(), doc_id);
        assert_eq!(
            sparse.vector(ord).unwrap(),
            &[(doc_id % 251) as u8, 7, 200, (doc_id / 3 % 251) as u8][..]
        );
    }

    // The graph: 800 vectors is well past HNSW_GRAPH_THRESHOLD, so the
    // dense field must carry a real one and every node must be on level 0.
    let graphs =
        hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
    let graph = graphs
        .graph(1)
        .unwrap()
        .expect("dense field must have a graph");
    assert!(graph.num_levels() >= 1);
    assert_eq!(
        graph.sorted_nodes_on_level(0).unwrap().len(),
        n as usize,
        "level 0 must contain every vector"
    );
    // 200 byte vectors is below the threshold: Lucene builds no graph, and
    // neither may this writer.
    assert!(
        graphs.graph(2).unwrap().is_none(),
        "a sub-threshold field must have numLevels = 0"
    );
}

/// The graph the flush built has to be *usable*, not merely present: an
/// approximate search over it must land on the same documents an
/// exhaustive scan over the same `.vec` does. (This is the port's own
/// end of the check; `VerifyVectorSegment` runs real Lucene's
/// `KnnFloatVectorQuery` over the same bytes.)
#[test]
fn the_flushed_graph_finds_the_same_documents_an_exhaustive_scan_does() {
    let tmp = tempdir("vectors-search");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            12,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    for i in 0..1000 {
        writer
            .add_document_with_vectors(
                doc(&format!("doc{i}")),
                vec![DocumentVector::float32("v", test_vector(12, 5000 + i))],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let (vec_bytes, vemf, vem, vex) = open_vector_files(&dir, sci);
    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let flat =
        vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
    let graphs =
        hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
    let values = flat.float_vector_values(1).unwrap();
    let graph = graphs.graph(1).unwrap().unwrap();

    let mut hits = 0usize;
    let mut total = 0usize;
    for q in 0..8 {
        let query = test_vector(12, 90_000 + q);
        let exact: Vec<i32> = values
            .exhaustive_search(&query, 10)
            .unwrap()
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let mut scorer = values.scorer(&query).unwrap();
        // `SearchOptions::default()` is Java's unfiltered, unseeded
        // `KnnFloatVectorQuery`; the second half of the return is the
        // collector's early-termination flag, which this recall check
        // does not consult.
        let (approx, _early_terminated) = hnsw_vectors::search(
            &mut scorer,
            Some(&graph),
            10,
            u64::MAX,
            hnsw_vectors::SearchOptions::default(),
        )
        .unwrap();
        let approx: Vec<i32> = approx
            .into_iter()
            .map(|(ord, _)| values.ord_to_doc(ord).unwrap())
            .collect();
        total += exact.len();
        hits += approx.iter().filter(|d| exact.contains(d)).count();
    }
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.8,
        "graph search over a flushed segment recalled {recall} of the exact top-10"
    );
}

/// A field declared with a dimension that no document ever carried must
/// not be claimed by the `.fnm`, and must get no per-field format
/// attributes -- see `fields_with_per_field_attributes`.
#[test]
fn a_vector_field_no_document_carried_is_not_claimed_in_the_fnm() {
    let tmp = tempdir("vectors-unused-field");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "used",
            1,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ),
        vector_field(
            "unused",
            2,
            6,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("used")).unwrap();
    writer.add_vector_field("unused").unwrap();
    for i in 0..3 {
        writer
            .add_document_with_vectors(
                doc(&format!("doc{i}")),
                vec![DocumentVector::float32("used", test_vector(4, i))],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let parsed = fi::parse(&fnm, &sci.segment_id, "").unwrap();

    let used = parsed.fields.iter().find(|f| f.name == "used").unwrap();
    assert_eq!(used.vector_dimension, 4);
    assert!(used
        .attributes
        .iter()
        .any(|(k, v)| k == "PerFieldKnnVectorsFormat.format" && v == KNN_VECTORS_FORMAT_NAME));
    assert!(used
        .attributes
        .iter()
        .any(|(k, v)| k == "PerFieldKnnVectorsFormat.suffix" && v == PER_FIELD_SUFFIX));

    let unused = parsed.fields.iter().find(|f| f.name == "unused").unwrap();
    assert_eq!(
        unused.vector_dimension, 0,
        "a field no document carried must record dimension 0, or a reader \
             sees a vector-capable field with no vectors"
    );
    assert!(!unused
        .attributes
        .iter()
        .any(|(k, _)| k.starts_with("PerFieldKnnVectorsFormat")));
    // And the `.vemf` must carry only the field that got values.
    let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let flat =
        vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
    assert_eq!(flat.fields().len(), 1);
    assert_eq!(flat.fields()[0].field_number, 1);
}

/// No opted-in field with any value at all: no vector files, and nothing
/// in the `.si` naming them.
#[test]
fn a_flush_with_no_vectors_writes_no_vector_files() {
    let tmp = tempdir("vectors-none");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    writer.add_document(doc("only-stored")).unwrap();
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(
        !si.files.iter().any(|f| f.ends_with(".vec")),
        "no vector files may be named: {:?}",
        si.files
    );
    assert!(dir
        .list_all()
        .unwrap()
        .iter()
        .all(|f| !f.ends_with(".vec") && !f.ends_with(".vem")));
}

/// The four files must be listed in `SegmentInfo.files`; `IndexFileDeleter`,
/// `CheckIndex` and this port's own `checksum_verify` all walk that list,
/// so a file missing from it is a file nothing knows about.
#[test]
fn the_segment_info_lists_the_four_vector_files() {
    let tmp = tempdir("vectors-si-files");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    for i in 0..4 {
        writer
            .add_document_with_vectors(
                doc(&format!("doc{i}")),
                vec![DocumentVector::float32("v", test_vector(4, i))],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    let seg = per_field_segment(&sci.segment_name, KNN_VECTORS_FORMAT_NAME);
    for ext in ["vec", "vemf", "vem", "vex"] {
        let name = format!("{seg}.{ext}");
        assert!(
            si.files.contains(&name),
            "{name} missing from {:?}",
            si.files
        );
        assert!(
            dir.list_all().unwrap().contains(&name),
            "{name} not written"
        );
    }
}

/// Every per-document validation `add_document_with_vectors` runs, one
/// case each -- and each of them must leave the buffer untouched, since a
/// half-buffered document would desynchronise `pending_vectors` from
/// `pending_docs`.
#[test]
fn add_document_with_vectors_rejects_malformed_input() {
    let tmp = tempdir("vectors-validation");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "f32",
            1,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
        vector_field(
            "bytes",
            2,
            4,
            VectorEncoding::Byte,
            VectorSimilarityFunction::DotProduct,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("f32")).unwrap();
    writer.add_vector_field("bytes").unwrap();

    // Wrong dimension.
    assert!(matches!(
        writer.add_document_with_vectors(
            doc("a"),
            vec![DocumentVector::float32("f32", vec![1.0, 2.0])]
        ),
        Err(Error::VectorDimensionMismatch(name, 0, 4, 2)) if name == "f32"
    ));
    // Float value for a BYTE field.
    assert!(matches!(
        writer.add_document_with_vectors(
            doc("a"),
            vec![DocumentVector::float32("bytes", vec![1.0, 2.0, 3.0, 4.0])]
        ),
        Err(Error::VectorEncodingMismatch(
            _,
            0,
            VectorEncoding::Byte,
            VectorEncoding::Float32
        ))
    ));
    // Byte value for a FLOAT32 field.
    assert!(matches!(
        writer.add_document_with_vectors(
            doc("a"),
            vec![DocumentVector::byte("f32", vec![1, 2, 3, 4])]
        ),
        Err(Error::VectorEncodingMismatch(
            _,
            0,
            VectorEncoding::Float32,
            VectorEncoding::Byte
        ))
    ));
    // A field that is not opted in.
    assert!(matches!(
        writer.add_document_with_vectors(
            doc("a"),
            vec![DocumentVector::float32("nope", vec![1.0, 2.0, 3.0, 4.0])]
        ),
        Err(Error::UnknownVectorField(name)) if name == "nope"
    ));
    // The same field twice on one document.
    assert!(matches!(
        writer.add_document_with_vectors(
            doc("a"),
            vec![
                DocumentVector::float32("f32", vec![1.0, 2.0, 3.0, 4.0]),
                DocumentVector::float32("f32", vec![5.0, 6.0, 7.0, 8.0]),
            ]
        ),
        Err(Error::DuplicateVectorField(name)) if name == "f32"
    ));
    assert_eq!(
        writer.pending_doc_count(),
        0,
        "a rejected document must not be buffered"
    );

    // A non-finite component is caught at flush, by `VectorUtil.checkFinite`
    // in the codec -- the last point that can see the whole field.
    writer
        .add_document_with_vectors(
            doc("a"),
            vec![DocumentVector::float32(
                "f32",
                vec![1.0, f32::NAN, 3.0, 4.0],
            )],
        )
        .unwrap();
    assert!(matches!(
        writer.commit(),
        Err(Error::Vectors(vectors::Error::NonFiniteValue(1, 1, _)))
    ));
}

#[test]
fn set_vector_field_validates_its_field() {
    let tmp = tempdir("vectors-config");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("plain", 0),
        vector_field(
            "v",
            1,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert!(matches!(
        writer.set_vector_field(Some("missing")),
        Err(Error::UnknownVectorField(_))
    ));
    assert!(matches!(
        writer.set_vector_field(Some("plain")),
        Err(Error::UnsupportedVectorField(_, 0))
    ));
    writer.set_vector_field(Some("v")).unwrap();
    assert!(matches!(
        writer.add_vector_field("v"),
        Err(Error::DuplicateVectorField(_))
    ));
    // `None` clears the whole list, the same "reassign, don't accumulate"
    // shape `set_postings_field` has.
    writer.set_vector_field(None).unwrap();
    writer.add_vector_field("v").unwrap();

    assert!(matches!(
        writer.set_hnsw_parameters(0, 100),
        Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
    ));
    assert!(matches!(
        writer.set_hnsw_parameters(513, 100),
        Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
    ));
    assert!(matches!(
        writer.set_hnsw_parameters(16, 0),
        Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
    ));
    assert!(matches!(
        writer.set_hnsw_parameters(16, 3201),
        Err(Error::Vectors(vectors::Error::InvalidGraphParameter(_)))
    ));
    writer.set_hnsw_parameters(8, 40).unwrap();
}

/// `pending_vectors` is indexed by doc id, so it must stay aligned 1:1
/// with `pending_docs` no matter which entry point buffered a document and
/// no matter what discarded the buffer. A drift here does not fail loudly:
/// it shifts every vector to the wrong document.
#[test]
fn the_vector_buffer_stays_aligned_with_the_document_buffer() {
    let tmp = tempdir("vectors-alignment");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();

    // A plain add, a block add, a vector add, a rollback, then the same
    // again -- every entry point that touches the buffer.
    writer.add_document(doc("plain")).unwrap();
    writer
        .add_documents(vec![doc("block-a"), doc("block-b")])
        .unwrap();
    writer
        .add_document_with_vectors(
            doc("with-vector"),
            vec![DocumentVector::float32("v", vec![1.0, 2.0])],
        )
        .unwrap();
    writer.rollback();

    writer.add_document(doc("d0")).unwrap();
    writer
        .add_document_with_vectors(
            doc("d1"),
            vec![DocumentVector::float32("v", vec![3.0, 4.0])],
        )
        .unwrap();
    writer.add_documents(vec![doc("d2"), doc("d3")]).unwrap();
    writer
        .add_document_with_vectors(
            doc("d4"),
            vec![DocumentVector::float32("v", vec![5.0, 6.0])],
        )
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let flat =
        vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix).unwrap();
    let values = flat.float_vector_values(1).unwrap();
    assert_eq!(values.size(), 2);
    assert_eq!(values.ord_to_doc(0).unwrap(), 1, "d1 is doc 1");
    assert_eq!(values.ord_to_doc(1).unwrap(), 4, "d4 is doc 4");
    assert_eq!(values.vector(0).unwrap(), vec![3.0, 4.0]);
    assert_eq!(values.vector(1).unwrap(), vec![5.0, 6.0]);
    assert_eq!(read_all_docs(&dir, &infos).len(), 5);
}

/// Two segments, each with its own vector files: the second flush must not
/// inherit the first's ordinals or overwrite its files.
#[test]
fn each_flushed_segment_gets_its_own_vector_files() {
    let tmp = tempdir("vectors-two-segments");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            3,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    for i in 0..3 {
        writer
            .add_document_with_vectors(
                doc(&format!("a{i}")),
                vec![DocumentVector::float32("v", vec![i as f32, 0.0, 1.0])],
            )
            .unwrap();
    }
    writer.flush().unwrap();
    for i in 0..2 {
        writer
            .add_document_with_vectors(
                doc(&format!("b{i}")),
                vec![DocumentVector::float32("v", vec![9.0, i as f32, 2.0])],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 2);
    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let sizes: Vec<i32> = infos
        .segments
        .iter()
        .map(|sci| {
            let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
            vectors::FlatVectorsReader::open(&vemf, &vec_bytes, &sci.segment_id, &suffix)
                .unwrap()
                .float_vector_values(1)
                .unwrap()
                .size()
        })
        .collect();
    assert_eq!(sizes, vec![3, 2]);
}

/// This port's own `CheckIndex` (`crate::check_index`) vector and HNSW
/// checks, run over a writer-produced vector segment.
#[test]
fn our_own_check_index_passes_over_a_writer_produced_vector_segment() {
    let tmp = tempdir("vectors-check-index");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "dense",
            1,
            8,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
        vector_field(
            "sparse",
            2,
            4,
            VectorEncoding::Byte,
            VectorSimilarityFunction::DotProduct,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("dense")).unwrap();
    writer.add_vector_field("sparse").unwrap();
    for i in 0..900 {
        let mut vectors = vec![DocumentVector::float32("dense", test_vector(8, i))];
        if i % 2 == 0 {
            vectors.push(DocumentVector::byte(
                "sparse",
                vec![(i % 251) as u8, 3, 9, (i / 5 % 251) as u8],
            ));
        }
        writer
            .add_document_with_vectors(doc(&format!("doc{i}")), vectors)
            .unwrap();
    }
    writer.commit().unwrap();

    let results = crate::check_index::check_directory(&dir).unwrap();
    for result in &results {
        assert!(
            result.all_passed(),
            "check_index failed on {}: {:?}",
            result.segment_name,
            result.failures()
        );
    }
    // ...and it must actually have *looked* at the vectors, not skipped
    // them: two fields, 900 + 450 values.
    let stats = &results
        .iter()
        .find(|r| r.max_doc.is_some())
        .expect("one segment")
        .stats;
    assert_eq!(stats.vector_fields, 2);
    assert_eq!(stats.vector_values, 900 + 450);
}

/// `set_hnsw_parameters` must actually reach the graph: `M` is recorded in
/// the `.vem` and bounds every node's neighbour count.
#[test]
fn set_hnsw_parameters_reaches_the_written_graph() {
    let tmp = tempdir("vectors-hnsw-params");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        vector_field(
            "v",
            1,
            6,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    writer.set_hnsw_parameters(6, 32).unwrap();
    for i in 0..800 {
        writer
            .add_document_with_vectors(
                doc(&format!("doc{i}")),
                vec![DocumentVector::float32("v", test_vector(6, i))],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let (_, _, vem, vex) = open_vector_files(&dir, sci);
    let suffix = per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME);
    let graphs =
        hnsw_vectors::HnswVectorsReader::open(&vem, &vex, &sci.segment_id, &suffix).unwrap();
    let graph = graphs.graph(1).unwrap().unwrap();
    assert_eq!(graph.max_conn(), 6);
    let mut neighbors = Vec::new();
    for node in graph.sorted_nodes_on_level(0).unwrap() {
        graph.neighbors_into(0, node, &mut neighbors).unwrap();
        assert!(
            neighbors.len() <= 12,
            "level 0 allows at most 2*M = 12 arcs, got {}",
            neighbors.len()
        );
    }
}

// ---------------------------------------------------------------
// Index sorting (`IndexWriterConfig.setIndexSort`)
// ---------------------------------------------------------------

/// The two `Long.MIN_VALUE`/`Long.MAX_VALUE` sentinels this port's own
/// sorted flush and sorted merge write, as a test-local shorthand.
/// `IndexSortField` models Java's whole missing-value space now (any
/// numeric sentinel, or none), so this names the two the tests below
/// care about rather than pretending they are all there is.
#[derive(Debug, Clone, Copy)]
enum SortMissingValue {
    /// `SortField.setMissingValue(Long.MIN_VALUE)`.
    First,
    /// `SortField.setMissingValue(Long.MAX_VALUE)`.
    Last,
}

fn sort_field(name: &str, reverse: bool, missing: SortMissingValue) -> IndexSortField {
    IndexSortField::long(
        name,
        reverse,
        Some(match missing {
            SortMissingValue::First => i64::MIN,
            SortMissingValue::Last => i64::MAX,
        }),
    )
}

/// A writer over `id` (stored), `rank`/`tie` (NUMERIC doc values, also
/// stored so the doc carries the value the flush reads its sort key
/// from), `body` (postings + norms + term vectors) and `v` (vectors).
/// One helper, because every assertion below wants a *different* format
/// to be the one that gets the sort wrong.
fn sortable_fields_with(postings: bool, term_vectors: bool) -> Vec<FieldInfo> {
    vec![
        stored_only_field("id", 0),
        numeric_field("rank", 1),
        numeric_field("tie", 2),
        FieldInfo {
            index_options: if postings {
                IndexOptions::DocsAndFreqs
            } else {
                IndexOptions::None
            },
            store_term_vectors: term_vectors,
            ..stored_only_field("body", 3)
        },
        vector_field(
            "v",
            4,
            4,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
    ]
}

/// The doc-values-only shape: same field numbers, but `body` is plain
/// stored, so a flush that opts into neither postings nor term vectors
/// still writes a `.fnm` that claims nothing the segment lacks.
fn sortable_fields() -> Vec<FieldInfo> {
    sortable_fields_with(false, false)
}

fn sortable_doc(id: &str, rank: i64, tie: i64, body: &str) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(rank),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(tie),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::String(body.to_string()),
            },
        ],
    }
}

fn read_index_sort(dir: &FsDirectory, sci: &SegmentCommitInfo) -> Option<Vec<IndexSortField>> {
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    segment_info::parse(&si_bytes, &sci.segment_id)
        .unwrap()
        .index_sort
}

/// Reads back one NUMERIC doc-values column of a *freshly flushed*
/// segment (no doc-values generation), in doc order.
fn read_base_numeric_column(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    field_number: i32,
    max_doc: i32,
) -> Vec<Option<i64>> {
    let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
    let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: sortable_fields(),
    };
    let (_, meta) = doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = meta.numeric_entry(field_number).unwrap();
    (0..max_doc)
        .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap())
        .collect()
}

#[test]
fn set_index_sort_rejects_an_empty_sort() {
    let tmp = tempdir("sort-empty");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    assert!(matches!(
        writer.set_index_sort(Some(&[])).unwrap_err(),
        Error::EmptyIndexSort
    ));
}

#[test]
fn set_index_sort_rejects_an_unknown_field() {
    let tmp = tempdir("sort-unknown");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    let err = writer
        .set_index_sort(Some(&[sort_field("nope", false, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(matches!(err, Error::UnknownIndexSortField(f) if f == "nope"));
}

/// `IndexingChain.validateIndexSortDVType`: a sort field that is not
/// NUMERIC doc values cannot be read by the `LONG` `SortField` this
/// port's `.si` encodes, so it is refused rather than written as a sort
/// no reader can evaluate.
#[test]
fn set_index_sort_rejects_a_field_that_is_not_numeric_doc_values() {
    let tmp = tempdir("sort-wrong-dv-type");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), sorted_field("name", 1)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("name")).unwrap();
    let err = writer
        .set_index_sort(Some(&[sort_field("name", false, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedIndexSortField(f, DocValuesType::Sorted) if f == "name"
    ));
    // The `id` field has no doc values at all -- the other half of the
    // same check.
    let err = writer
        .set_index_sort(Some(&[sort_field("id", false, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::UnsupportedIndexSortField(f, DocValuesType::None) if f == "id"
    ));
}

/// A sort over a column nothing writes is a sort no reader can check:
/// real Lucene's `DocValues.getNumeric` substitutes an all-missing
/// instance rather than failing, so `CheckIndex.testSort` would compare
/// `maxDoc` equal keys and pass. Refused at configuration time instead.
#[test]
fn set_index_sort_rejects_a_sort_field_no_doc_values_are_written_for() {
    let tmp = tempdir("sort-no-dv-optin");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    let err = writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(matches!(err, Error::IndexSortFieldWithoutDocValues(f) if f == "rank"));
    // Opting in makes the same call succeed.
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
}

/// The sort is read at flush, so changing it with documents already
/// buffered would order one part of the batch by one key and the rest by
/// another. Java snapshots `IndexWriterConfig` at construction and cannot
/// reach the state at all.
#[test]
fn set_index_sort_is_refused_once_documents_are_buffered() {
    let tmp = tempdir("sort-mid-buffer");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
    assert!(matches!(
        writer
            .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
            .unwrap_err(),
        Error::IndexSortChangedMidBuffer(1)
    ));
    // ...including clearing it.
    assert!(matches!(
        writer.set_index_sort(None).unwrap_err(),
        Error::IndexSortChangedMidBuffer(1)
    ));
    writer.commit().unwrap();
    writer.set_index_sort(None).unwrap();
}

/// `IndexWriter.validateIndexSort` + `isCongruentSort`: an existing
/// segment's sort must have the incoming one as a **prefix**. An
/// unsorted segment fails just as hard as a differently-sorted one.
#[test]
fn set_index_sort_must_be_congruent_with_the_segments_already_in_the_index() {
    let tmp = tempdir("sort-congruence");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
    writer.commit().unwrap();

    // The committed segment is unsorted, so *any* sort is incongruent.
    let err = writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::IncongruentIndexSort { segment, existing, .. }
            if segment == "_0" && existing == "<none>")
    );

    // A fresh index sorted by (rank, tie) accepts the prefix (rank), the
    // whole sort, and rejects a longer or different one.
    let tmp2 = tempdir("sort-congruence-2");
    let dir2 = FsDirectory::open(&tmp2);
    let mut w2 = IndexWriter::open(&dir2, sortable_fields(), "Lucene104", version()).unwrap();
    w2.set_doc_values_field(Some("rank")).unwrap();
    w2.add_doc_values_field("tie").unwrap();
    w2.set_index_sort(Some(&[
        sort_field("rank", false, SortMissingValue::Last),
        sort_field("tie", false, SortMissingValue::Last),
    ]))
    .unwrap();
    w2.add_document(sortable_doc("a", 1, 0, "x")).unwrap();
    w2.commit().unwrap();

    w2.set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .expect("a prefix of the existing sort is congruent");
    w2.set_index_sort(Some(&[
        sort_field("rank", false, SortMissingValue::Last),
        sort_field("tie", false, SortMissingValue::Last),
    ]))
    .expect("the identical sort is congruent");
    // Same fields, opposite direction: not a prefix.
    let err = w2
        .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
        .unwrap_err();
    assert!(matches!(err, Error::IncongruentIndexSort { .. }));
    // Longer than the existing sort: also not a prefix.
    let err = w2
        .set_index_sort(Some(&[
            sort_field("rank", false, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
            sort_field("rank", false, SortMissingValue::Last),
        ]))
        .unwrap_err();
    assert!(matches!(err, Error::IncongruentIndexSort { .. }));
}

/// The one hole a mutable configuration leaves that Java's immutable one
/// does not: `set_doc_values_field` *shrinks* the list, so after
/// `set_index_sort` it could strand the sort over a column nothing
/// writes.
#[test]
fn narrowing_the_doc_values_fields_cannot_strand_a_configured_index_sort() {
    let tmp = tempdir("sort-strand-dv");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer
        .set_index_sort(Some(&[
            sort_field("rank", true, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::First),
        ]))
        .unwrap();

    // Both of these would leave a tier with no column.
    assert!(matches!(
        writer.set_doc_values_field(Some("rank")).unwrap_err(),
        Error::IndexSortFieldWithoutDocValues(f) if f == "tie"
    ));
    assert!(matches!(
        writer.set_doc_values_field(None).unwrap_err(),
        Error::IndexSortFieldWithoutDocValues(f) if f == "rank"
    ));
    // The rejected calls left the list intact, so a flush still works.
    writer.add_document(sortable_doc("a", 1, 2, "x")).unwrap();
    writer.commit().unwrap();
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }

    // Clearing the sort first makes the same call legal.
    // Fresh directories, since the committed segment above pins the sort
    // every later `set_index_sort` on `dir` must be congruent with.
    let tmp2 = tempdir("sort-strand-dv-cleared");
    let dir2 = FsDirectory::open(&tmp2);
    let mut writer = IndexWriter::open(&dir2, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
        .unwrap();
    writer.set_index_sort(None).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();

    // ...and narrowing to exactly the single tier's own column is fine.
    let tmp3 = tempdir("sort-strand-dv-single");
    let dir3 = FsDirectory::open(&tmp3);
    let mut writer = IndexWriter::open(&dir3, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
}

/// The heart of the batch: **one** permutation reaches every format.
/// Stored fields, the doc-values column, postings doc ids, norms, term
/// vectors and vectors must all describe the *same* document at the same
/// doc id after a sorted flush -- the failure mode this guards against is
/// silent (each file is well-formed and checksums cleanly; only the
/// association between them is wrong).
#[test]
fn a_sorted_flush_reorders_every_format_together() {
    let tmp = tempdir("sort-all-formats");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(
        &dir,
        sortable_fields_with(true, true),
        "Lucene104",
        version(),
    )
    .unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();

    // Inserted in an order that has nothing to do with the sort, with a
    // per-document body whose token count (and therefore norm) and whose
    // unique term both identify the document.
    let inserted: Vec<(&str, i64)> = vec![("d", 40), ("a", 10), ("c", 30), ("b", 20)];
    for (id, rank) in &inserted {
        let body: String = std::iter::repeat_n(format!("t{id}"), *rank as usize / 10)
            .collect::<Vec<_>>()
            .join(" ");
        writer
            .add_document_with_vectors(
                sortable_doc(id, *rank, 0, &body),
                vec![DocumentVector::float32(
                    "v",
                    vec![*rank as f32, 1.0, 2.0, 3.0],
                )],
            )
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];

    // Stored fields: ascending by rank.
    assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c", "d"]);
    // The `.si` says so too.
    let sort = read_index_sort(&dir, sci).unwrap();
    assert_eq!(
        sort,
        vec![sort_field("rank", false, SortMissingValue::Last)]
    );
    // Doc values: the sort key column, in the new doc order.
    assert_eq!(
        read_base_numeric_column(&dir, sci, 1, 4),
        vec![Some(10), Some(20), Some(30), Some(40)]
    );

    // Postings: each document's unique term must resolve to its *new*
    // doc id. A permutation applied to stored fields but not to the
    // invert pass lands every term on the wrong document.
    let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let tim = dir.open(&format!("{seg}.tim")).unwrap();
    let tip = dir.open(&format!("{seg}.tip")).unwrap();
    let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
    let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
    let field_infos = fi::FieldInfos {
        fields: sortable_fields_with(true, true),
    };
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        4,
    )
    .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let body_terms = block_fields.field("body").unwrap();
    for (new_doc, id) in ["a", "b", "c", "d"].iter().enumerate() {
        let postings = body_terms
            .postings(format!("t{id}").as_bytes(), Some(&doc_in))
            .unwrap()
            .unwrap();
        assert_eq!(
            postings.docs,
            vec![new_doc as i32],
            "term t{id} must land on the sorted doc id"
        );
    }

    // Norms: doc `i`'s norm encodes its field length, which is
    // `rank / 10` tokens -- strictly increasing in the sorted order.
    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_, norms_meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    let entry = norms_meta.entry(3).unwrap();
    let lengths: Vec<i64> = (0..4)
        .map(|d| {
            lucene_util::small_float::byte4_to_int(
                norms::norm_value(&nvd, entry, d).unwrap().unwrap() as u8,
            ) as i64
        })
        .collect();
    assert_eq!(lengths, vec![1, 2, 3, 4]);

    // Term vectors: doc `i` must carry only its own term.
    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let tv = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    for (new_doc, id) in ["a", "b", "c", "d"].iter().enumerate() {
        let d = tv.document(new_doc as i32).unwrap().unwrap();
        let terms: Vec<String> = d.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        assert_eq!(terms, vec![format!("t{id}")]);
    }

    // Vectors: the ordinal -> doc mapping and the value itself. Every
    // vector's first component is its document's rank, so a vector that
    // drifted onto another document is immediately visible. This is the
    // "ordinals must be remapped" hazard, discharged by the buffer
    // permutation rather than by a per-format remap.
    let (vec_bytes, vemf, _, _) = open_vector_files(&dir, sci);
    let flat = vectors::FlatVectorsReader::open(
        &vemf,
        &vec_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
    )
    .unwrap();
    let values = flat.float_vector_values(4).unwrap();
    assert_eq!(values.size(), 4);
    for ord in 0..4 {
        let doc_id = values.ord_to_doc(ord).unwrap();
        assert_eq!(doc_id, ord, "dense field: ordinal == doc id");
        let v = values.vector(ord).unwrap();
        assert_eq!(
            v[0],
            ((ord + 1) * 10) as f32,
            "vector on doc {doc_id} belongs to another document"
        );
    }

    // ...and this port's own CheckIndex, which re-derives the sort's
    // comparators from the `.si` and the doc values and walks adjacent
    // doc ids.
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(
            result.all_passed(),
            "check_index failed: {:?}",
            result.failures()
        );
    }
}

/// Reads one segment back and asserts that **every** format describes the
/// same document at the same doc id, given the ids the segment is
/// expected to hold in physical order.
///
/// The corpus convention (shared with
/// `a_sorted_flush_reorders_every_format_together`): document `id` has
/// `rank` doc-values, a body of `rank / 10` copies of the unique term
/// `t{id}`, so its norm encodes `rank / 10`, and a vector whose first
/// component is `rank`. Every one of those is a different file, so a
/// doc map applied to one and not another shows up as a specific
/// mismatch rather than as "something is off".
fn assert_every_format_agrees(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
    expected: &[(&str, i64)],
) {
    let max_doc = expected.len() as i32;
    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    assert_eq!(stored.max_doc(), max_doc);
    for (doc_id, (id, _)) in expected.iter().enumerate() {
        assert_eq!(
            doc_value(&stored.document(doc_id as i32).unwrap()),
            *id,
            "stored id at doc {doc_id}"
        );
    }

    let ranks: Vec<Option<i64>> = expected.iter().map(|(_, rank)| Some(*rank)).collect();
    let seg = per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME);
    let dvm = dir.open(&format!("{seg}.dvm")).unwrap();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
    let field_infos = fi::FieldInfos {
        fields: sortable_fields_with(true, true),
    };
    let (_, dv_meta) = doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let dv_entry = dv_meta.numeric_entry(1).unwrap();
    let read_ranks: Vec<Option<i64>> = (0..max_doc)
        .map(|d| doc_values::numeric_value(&dvd, dv_entry, d).unwrap())
        .collect();
    assert_eq!(read_ranks, ranks, "doc-values rank column");

    let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let tim = dir.open(&format!("{seg}.tim")).unwrap();
    let tip = dir.open(&format!("{seg}.tip")).unwrap();
    let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
    let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        max_doc,
    )
    .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let body_terms = block_fields.field("body").unwrap();
    for (doc_id, (id, _)) in expected.iter().enumerate() {
        let postings = body_terms
            .postings(format!("t{id}").as_bytes(), Some(&doc_in))
            .unwrap()
            .unwrap_or_else(|| panic!("term t{id} is missing from the merged dictionary"));
        assert_eq!(
            postings.docs,
            vec![doc_id as i32],
            "term t{id} must land on doc {doc_id}"
        );
    }

    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_, norms_meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    let norms_entry = norms_meta.entry(3).unwrap();
    let lengths: Vec<i64> = (0..max_doc)
        .map(|d| {
            lucene_util::small_float::byte4_to_int(
                norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap() as u8,
            ) as i64
        })
        .collect();
    assert_eq!(
        lengths,
        expected.iter().map(|(_, r)| r / 10).collect::<Vec<_>>(),
        "norms"
    );

    let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
    let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
    let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
    let tv = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    for (doc_id, (id, _)) in expected.iter().enumerate() {
        let d = tv.document(doc_id as i32).unwrap().unwrap();
        let terms: Vec<String> = d.fields[0]
            .terms
            .iter()
            .map(|t| String::from_utf8(t.term.clone()).unwrap())
            .collect();
        assert_eq!(
            terms,
            vec![format!("t{id}")],
            "term vectors at doc {doc_id}"
        );
    }

    let (vec_bytes, vemf, vem, vex) = open_vector_files(dir, sci);
    let flat = vectors::FlatVectorsReader::open(
        &vemf,
        &vec_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
    )
    .unwrap();
    let values = flat.float_vector_values(4).unwrap();
    assert_eq!(values.size(), max_doc);
    for ord in 0..max_doc {
        let doc_id = values.ord_to_doc(ord).unwrap();
        assert_eq!(doc_id, ord, "dense vector field: ordinal == doc id");
        assert_eq!(
            values.vector(ord).unwrap()[0],
            expected[ord as usize].1 as f32,
            "vector on doc {doc_id} belongs to another document"
        );
    }
    // The graph must describe the merged ordinal space, not a source's.
    let graph_reader = hnsw_vectors::HnswVectorsReader::open(
        &vem,
        &vex,
        &sci.segment_id,
        &per_field_codec_suffix(KNN_VECTORS_FORMAT_NAME),
    )
    .unwrap();
    assert_eq!(graph_reader.field(4).unwrap().size, max_doc);
}

/// Adds one document of the corpus `assert_every_format_agrees` reads
/// back: unique term, length-encoding body, and a vector that names its
/// own rank.
fn add_sortable_doc(writer: &mut IndexWriter, id: &str, rank: i64, tie: i64) {
    let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
        .collect::<Vec<_>>()
        .join(" ");
    writer
        .add_document_with_vectors(
            sortable_doc(id, rank, tie, &body),
            vec![DocumentVector::float32(
                "v",
                vec![rank as f32, 1.0, 2.0, 3.0],
            )],
        )
        .unwrap();
}

/// **The merge-completeness gate, end to end.** The union of formats the
/// flush wrote across the source segments must reappear in the merged
/// segment's own `.si` -- not "the merge succeeded", which it does just
/// as happily with a format dropped.
///
/// This is the observable half of `merge::check_format_coverage`: the
/// gate refuses a merge that never *opened* a format, and this asserts
/// that every opened format was also *written*. Together they check
/// mechanically that `execute_merge` opens every format the flush can
/// write; a dropped format would otherwise produce a well-formed,
/// checksummed, `CheckIndex`-clean segment with the data gone.
///
/// The writer is configured with every format the flush path has an
/// opt-in for: stored fields (always), postings **with positions and
/// offsets**, term vectors, doc values (two NUMERIC columns), norms on
/// two fields, and KNN vectors. Points are the one format with no flush
/// path at all (`docs/parity.md`), so no segment can carry them; the
/// gate covers them anyway, and would fire the day one can.
#[test]
fn every_format_the_flush_writes_reappears_in_the_merged_si() {
    let tmp = tempdir("merge-format-completeness");
    let dir = FsDirectory::open(&tmp);
    let mut fields = sortable_fields_with(true, true);
    // Positions and offsets, so the postings half is the widest shape
    // `set_postings_field` accepts rather than the `DocsAndFreqs` one.
    fields[3].index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    // A second norms field, which was inexpressible before c26.
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.add_postings_field("id").unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    // No merging for the first three commits, so every source segment is
    // observable before anything folds it away.
    writer.set_merge_policy(None);

    for batch in [
        vec![("d", 40i64), ("a", 10)],
        vec![("e", 50), ("b", 20)],
        vec![("f", 60), ("c", 30)],
    ] {
        for (id, rank) in batch {
            add_sortable_doc(&mut writer, id, rank, 0);
        }
        writer.commit().unwrap();
    }

    let sources = writer.segment_infos().segments.clone();
    assert_eq!(sources.len(), 3, "three unmerged sources");
    let source_names: Vec<String> = sources.iter().map(|s| s.segment_name.clone()).collect();
    let source_formats: std::collections::BTreeSet<merge::SegmentFormat> = sources
        .iter()
        .flat_map(|sci| segment_formats(&dir, sci))
        .collect();

    // Now let the merge policy see them.
    writer.set_merge_policy(Some(tight_merge_policy()));
    writer.commit().unwrap();

    // Everything but points, which no flush path can write.
    let expected: std::collections::BTreeSet<merge::SegmentFormat> = merge::SegmentFormat::ALL
        .into_iter()
        .filter(|f| *f != merge::SegmentFormat::Points)
        .collect();
    assert_eq!(
        source_formats, expected,
        "the sources must exercise every format the flush can write, or this test proves \
             nothing about the ones it skipped"
    );

    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1, "the sources must have merged");
    let merged = &infos.segments[0];
    assert!(
        !source_names.contains(&merged.segment_name),
        "the merged segment must be a new one, not one of the sources"
    );
    assert_eq!(
        segment_formats(&dir, merged),
        source_formats,
        "the merged `.si` must carry every format its sources' `.si`s did"
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// The set of `merge::SegmentFormat`s a segment's own `.si` file list
/// says it has -- the same classification the gate uses, so this test
/// and `check_format_coverage` cannot disagree about what a format is.
fn segment_formats(
    dir: &FsDirectory,
    sci: &SegmentCommitInfo,
) -> std::collections::BTreeSet<merge::SegmentFormat> {
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    si.files
        .iter()
        .filter_map(|f| f.rsplit_once('.'))
        .filter_map(|(_, ext)| merge::SegmentFormat::for_extension(ext))
        .collect()
}

/// **Multi-field norms, flush and merge.** `norms::write_fields` writes
/// every field's column into one `.nvm`/`.nvd` pair, on both the flush's
/// per-field column loop and `merge_norms`. Both `id` and `body` are
/// indexed, so both get norms.
///
/// Asserted on the *values*, per document per field, rather than on the
/// merge having happened: a merge that carried only the first field
/// would still produce a valid segment.
#[test]
fn an_automatic_merge_carries_norms_for_every_field() {
    let tmp = tempdir("multi-field-norms-merge");
    let dir = FsDirectory::open(&tmp);
    let mut fields = sortable_fields_with(true, false);
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.add_postings_field("id").unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));

    // `id` is one token, `body` is `rank / 10` of them -- so the two
    // fields' norms differ per document and a column written twice, or
    // one field's column read for the other, is visible.
    let corpus = [("aa", 10i64), ("bb", 20), ("cc", 30)];
    for (id, rank) in corpus {
        let body: String = std::iter::repeat_n(format!("t{id}"), rank as usize / 10)
            .collect::<Vec<_>>()
            .join(" ");
        writer
            .add_document(sortable_doc(id, rank, 0, &body))
            .unwrap();
        writer.commit().unwrap();
    }

    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1, "the sources must have merged");
    let sci = &infos.segments[0];
    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let nvd = dir.open(&format!("{}.nvd", sci.segment_name)).unwrap();
    let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    assert_eq!(
        meta.entries.len(),
        2,
        "both norms fields must share the one merged `.nvm`"
    );
    let id_entry = *meta
        .entries
        .iter()
        .find(|e| e.field_number == 0)
        .expect("`id`'s norms survived the merge");
    let body_entry = *meta
        .entries
        .iter()
        .find(|e| e.field_number == 3)
        .expect("`body`'s norms survived the merge");

    // Which document landed at which merged doc id is the merge's
    // business; resolve it through the stored `id` the way
    // `assert_every_format_agrees` does, so this asserts the *pairing*
    // rather than an assumed order.
    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let stored = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    assert_eq!(stored.max_doc(), corpus.len() as i32);
    for doc_id in 0..stored.max_doc() {
        let id = doc_value(&stored.document(doc_id).unwrap());
        let rank = corpus
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, r)| *r)
            .unwrap_or_else(|| panic!("unexpected stored id {id}"));
        assert_eq!(
            norms::norm_value(&nvd, &id_entry, doc_id).unwrap(),
            Some(small_float::int_to_byte4(1) as i8 as i64),
            "`id` norm at doc {doc_id} (id={id}): one token"
        );
        assert_eq!(
            norms::norm_value(&nvd, &body_entry, doc_id).unwrap(),
            Some(small_float::int_to_byte4(rank as u32 / 10) as i8 as i64),
            "`body` norm at doc {doc_id} (id={id})"
        );
    }
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// The norms field set is derived from the schema, not accumulated: every
/// indexed field that has not opted out gets exactly one column, and
/// `omit_norms_field` takes one away: the `.fnm` and the `.nvm` say the
/// same thing.
#[test]
fn the_norms_columns_are_exactly_the_indexed_non_omitting_fields() {
    let tmp = tempdir("norms-column-set");
    let dir = FsDirectory::open(&tmp);
    let mut fields = sortable_fields_with(true, false);
    // `id` and `body` are indexed; `tie` is indexed but opts out; `rank`
    // is a plain numeric field (`index_options == None`).
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    fields[2].omit_norms = true;
    fields[2].index_options = IndexOptions::DocsAndFreqs;
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();

    assert!(matches!(
        writer.omit_norms_field("nonexistent"),
        Err(Error::UnknownNormsField(_))
    ));
    // `rank` is a plain numeric field: `index_options == None`, so there
    // are no norms to omit.
    assert!(matches!(
        writer.omit_norms_field("rank"),
        Err(Error::UnsupportedNormsField(_))
    ));

    writer.set_postings_field(Some("body")).unwrap();
    writer.add_postings_field("id").unwrap();
    writer
        .add_document(sortable_doc("a", 10, 0, "ta ta"))
        .unwrap();
    writer.commit().unwrap();

    let sci = writer.segment_infos().segments[0].clone();
    let nvm = dir.open(&format!("{}.nvm", sci.segment_name)).unwrap();
    let (_v, meta) = norms::parse_meta(&nvm, &sci.segment_id, "").unwrap();
    assert_eq!(
        meta.entries.len(),
        2,
        "`id` and `body` each get exactly one column; `tie` opted out and \
             `rank` is not indexed"
    );

    // Opting the remaining two out writes no norms at all for the next
    // segment.
    writer.omit_norms_field("body").unwrap();
    writer.omit_norms_field("id").unwrap();
    writer
        .add_document(sortable_doc("b", 20, 0, "tb tb"))
        .unwrap();
    writer.commit().unwrap();
    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    assert!(
        !dir.list_all()
            .unwrap()
            .contains(&format!("{}.nvm", names[1])),
        "opting every field out must write no norms at all"
    );
}

/// **Index sort, write side.** A `SortedNumericSortField` with the `MAX`
/// selector and **no missing value** (Java compares such a document as
/// `0`).
///
/// Asserted through this port's own `CheckIndex`, which re-derives the
/// comparator from the written `.si` and re-reads the keys out of the
/// segment's SORTED_NUMERIC column -- so a flush that picked the wrong
/// value out of the multi-valued column, or that ignored the selector,
/// fails here rather than producing a plausible segment.
#[test]
fn a_sorted_numeric_max_selector_sort_orders_the_flush_and_survives_a_merge() {
    let tmp = tempdir("sorted-numeric-max-sort");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        FieldInfo {
            doc_values_type: DocValuesType::SortedNumeric,
            ..stored_only_field("multi", 1)
        },
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("multi")).unwrap();
    let sort = IndexSortField {
        field: "multi".to_string(),
        reverse: false,
        kind: crate::segment_info::IndexSortKind::SortedNumeric {
            key: crate::segment_info::NumericSortKey::Int(None),
            selector: crate::segment_info::SortedNumericSelector::Max,
        },
    };
    writer
        .set_index_sort(Some(std::slice::from_ref(&sort)))
        .unwrap();
    // No merging while the flushes are being observed.
    writer.set_merge_policy(None);

    // `(id, values)`. The MIN of each document is deliberately in the
    // *opposite* order from the MAX, so a flush that took the wrong end
    // of the column produces a different physical order.
    // `f`'s values are supplied *descending* on purpose: a document's
    // SORTED_NUMERIC values must be stored ascending
    // (`SortedNumericDocValuesWriter.finishCurrentDoc`'s `Arrays.sort`,
    // which real `CheckIndex.checkSortedNumericDocValues` enforces), and
    // MAX is then the *last* stored value. A writer that kept the
    // caller's order would both write a segment real Lucene rejects and
    // make the selector pick `3` instead of `6`.
    let corpus: [(&str, &[i64]); 6] = [
        ("a", &[1, 9]),
        ("b", &[8, 8]),
        ("c", &[0, 5]),
        ("d", &[7, 7]),
        ("e", &[2, 2]),
        ("f", &[6, 3]),
    ];
    for batch in [&corpus[..2], &corpus[2..4], &corpus[4..]] {
        for (id, values) in batch {
            let mut doc_fields = vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }];
            for v in *values {
                doc_fields.push(StoredField {
                    field_number: 1,
                    value: FieldValue::Long(*v),
                });
            }
            writer
                .add_document(Document { fields: doc_fields })
                .unwrap();
        }
        writer.commit().unwrap();
    }

    // Ascending by MAX: b(8), a(9) in the first flush and c(5), d(7) in
    // the second; ascending by MIN it would have been a(1), b(8) and
    // c(0), d(7) -- so the first flush's order alone discriminates.
    let sis = writer.segment_infos().clone();
    assert_eq!(
        read_all_docs(&dir, &sis),
        vec!["b", "a", "c", "d", "e", "f"],
        "each flush is MAX-ordered on its own: b(8) before a(9), c(5) before \
             d(7), e(2) before f(6) -- by MIN it would have been a(1) before b(8)"
    );

    writer.set_merge_policy(Some(tight_merge_policy()));
    writer.commit().unwrap();
    let sis = writer.segment_infos().clone();
    assert_eq!(sis.segments.len(), 1, "the two flushes merged into one");
    assert_eq!(
        read_all_docs(&dir, &sis),
        vec!["e", "c", "f", "d", "b", "a"],
        "the sort-preserving merge is MAX-ordered across sources too: \
             e(2), c(5), f(6), d(7), b(8), a(9)"
    );

    // The merged `.si` still describes the widened sort, verbatim.
    let sci = &sis.segments[0];
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert_eq!(si.index_sort, Some(vec![sort]));

    // The column real Lucene requires: every document's values ascending,
    // so `f` came back as `[3, 6]` and not the `[6, 3]` it was given.
    let dvm = dir
        .open(&format!(
            "{}.dvm",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let dvd = dir
        .open(&format!(
            "{}.dvd",
            per_field_segment(&sci.segment_name, DOC_VALUES_FORMAT_NAME)
        ))
        .unwrap();
    let infos = lucene_codecs::field_infos::parse(
        &dir.open(&format!("{}.fnm", sci.segment_name)).unwrap(),
        &sci.segment_id,
        "",
    )
    .unwrap();
    let (_v, dv_meta) = doc_values::parse_meta(
        &dvm,
        &sci.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &infos,
    )
    .unwrap();
    let entry = dv_meta.sorted_numeric_entry(1).unwrap();
    for doc in 0..6 {
        let values = doc_values::sorted_numeric_values(&dvd, entry, doc).unwrap();
        assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "doc {doc} values are not ascending: {values:?}"
        );
    }
    assert_eq!(
        doc_values::sorted_numeric_values(&dvd, entry, 2).unwrap(),
        vec![3, 6],
        "`f` is the third document in the merged order and its values were sorted"
    );

    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// `set_index_sort` accepts the byte-keyed kinds over the doc-values type
/// each reads (`validateIndexSortDVType`) and refuses them over any
/// other -- a `STRING` sort over a SORTED_SET column is not quietly read
/// as its first value.
#[test]
fn set_index_sort_matches_each_byte_kind_to_its_doc_values_type() {
    use crate::segment_info::{IndexSortKind, SortedSetSelector, StringMissingValue};
    let tmp = tempdir("sort-kind-gate");
    let dir = FsDirectory::open(&tmp);
    let mut tags = sorted_field("tags", 3);
    tags.doc_values_type = DocValuesType::SortedSet;
    let mut blob = sorted_field("blob", 4);
    blob.doc_values_type = DocValuesType::Binary;
    let fields = vec![
        stored_only_field("id", 0),
        numeric_field("rank", 1),
        sorted_field("name", 2),
        tags,
        blob,
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("name")).unwrap();
    writer.add_doc_values_field("tags").unwrap();
    writer.add_doc_values_field("blob").unwrap();
    let string = IndexSortKind::String(StringMissingValue::First);
    let set = IndexSortKind::SortedSet {
        selector: SortedSetSelector::Min,
        missing: StringMissingValue::None,
    };
    let binary = IndexSortKind::Binary(StringMissingValue::Last);
    for (field, kind) in [("name", &string), ("tags", &set), ("blob", &binary)] {
        let sf = IndexSortField {
            field: field.to_string(),
            reverse: false,
            kind: kind.clone(),
        };
        writer.set_index_sort(Some(&[sf])).unwrap();
    }
    for (field, kind, dv) in [
        ("tags", &string, DocValuesType::SortedSet),
        ("name", &set, DocValuesType::Sorted),
        ("name", &binary, DocValuesType::Sorted),
    ] {
        let sf = IndexSortField {
            field: field.to_string(),
            reverse: false,
            kind: kind.clone(),
        };
        assert!(
            matches!(
                writer.set_index_sort(Some(&[sf])),
                Err(Error::UnsupportedIndexSortField(f, t)) if f == field && t == dv
            ),
            "{kind:?} over {field}"
        );
    }
    // And the doc-values type still has to match the kind: a
    // SortedNumeric sort over a NUMERIC column is refused by
    // `validateIndexSortDVType`'s rule, not silently read as NUMERIC.
    writer.set_index_sort(None).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    let sf = IndexSortField {
        field: "rank".to_string(),
        reverse: false,
        kind: IndexSortKind::SortedNumeric {
            key: crate::segment_info::NumericSortKey::Long(None),
            selector: crate::segment_info::SortedNumericSelector::Min,
        },
    };
    assert!(matches!(
        writer.set_index_sort(Some(&[sf])),
        Err(Error::UnsupportedIndexSortField(f, DocValuesType::Numeric)) if f == "rank"
    ));
}

/// A writer configured exactly the way the merge tests below need it.
fn sorted_merge_writer<'a>(dir: &'a FsDirectory, sort: &[IndexSortField]) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        dir,
        sortable_fields_with(true, true),
        "Lucene104",
        version(),
    )
    .unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer.set_vector_field(Some("v")).unwrap();
    writer.set_index_sort(Some(sort)).unwrap();
    writer.set_merge_policy(Some(tight_merge_policy()));
    writer
}

/// Three index-sorted flushes are folded
/// into one segment by the merge policy, and the merged segment is still
/// in sort order -- globally, across sources, not per source -- with
/// every format still describing the same document at the same doc id.
///
/// Concatenating instead would give a valid, `CheckIndex`-clean segment
/// that is no longer in the order its inputs were in.
#[test]
fn an_automatic_merge_preserves_the_index_sort_across_every_format() {
    let tmp = tempdir("sorted-merge-all-formats");
    let dir = FsDirectory::open(&tmp);
    let mut writer =
        sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);

    // Three flushes, each internally sorted, whose ranges interleave --
    // so a concatenating merge cannot accidentally come out ordered.
    for batch in [
        vec![("d", 40i64), ("a", 10)],
        vec![("e", 50), ("b", 20)],
        vec![("f", 60), ("c", 30)],
    ] {
        for (id, rank) in batch {
            add_sortable_doc(&mut writer, id, rank, 0);
        }
        writer.commit().unwrap();
    }

    let infos = writer.segment_infos().clone();
    assert_eq!(
        infos.segments.len(),
        1,
        "the three sorted segments must have merged into one"
    );
    let sci = &infos.segments[0];
    assert_eq!(
        read_index_sort(&dir, sci).unwrap(),
        vec![sort_field("rank", false, SortMissingValue::Last)],
        "the merged `.si` must still declare the sort"
    );
    assert_every_format_agrees(
        &dir,
        sci,
        &[
            ("a", 10),
            ("b", 20),
            ("c", 30),
            ("d", 40),
            ("e", 50),
            ("f", 60),
        ],
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(
            result.all_passed(),
            "check_index failed: {:?}",
            result.failures()
        );
    }
}

/// **Negative control for the sort itself.** The same three segments
/// merged with the sort *removed from their `.si` files* -- i.e. exactly
/// what this code did before -- concatenate. The merged segment is
/// entirely valid and passes this port's `CheckIndex`, because it makes
/// no claim about its order; only comparing it against the sorted result
/// shows the loss. Then stamping the sort onto the merged `.si` (which is
/// what a merge that kept the declaration while concatenating would have
/// produced) makes `CheckIndex.testSort` fail.
#[test]
fn a_concatenating_merge_of_the_same_segments_loses_the_order_silently() {
    let tmp = tempdir("sorted-merge-negative-control");
    let dir = FsDirectory::open(&tmp);
    let mut writer =
        sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);
    writer.set_merge_policy(None);
    for batch in [
        vec![("d", 40i64), ("a", 10)],
        vec![("e", 50), ("b", 20)],
        vec![("f", 60), ("c", 30)],
    ] {
        for (id, rank) in batch {
            add_sortable_doc(&mut writer, id, rank, 0);
        }
        writer.commit().unwrap();
    }
    assert_eq!(writer.segment_infos().segments.len(), 3);

    // Strip the sort declaration from every source, so `execute_merge`
    // takes the concatenating path over the very same bytes.
    for sci in &writer.segment_infos().segments.clone() {
        let si_name = format!("{}.si", sci.segment_name);
        let si_bytes = dir.open(&si_name).unwrap().to_vec();
        let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        si.index_sort = None;
        dir.delete_file(&si_name).unwrap();
        write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();
    }
    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    writer.execute_merge(&names).unwrap();

    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    assert!(
        read_index_sort(&dir, sci).is_none(),
        "a concatenating merge must not claim a sort"
    );
    // Every format is still self-consistent -- the data is not corrupt,
    // it is merely in source order. Each source is internally sorted (its
    // flush sorted it), so the concatenation is sorted *within* a source
    // and jumps back down at every source boundary: the order looks
    // almost right, which is why nothing downstream notices.
    assert_every_format_agrees(
        &dir,
        sci,
        &[
            ("a", 10),
            ("d", 40),
            ("b", 20),
            ("e", 50),
            ("c", 30),
            ("f", 60),
        ],
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(
            result.all_passed(),
            "a concatenated segment is perfectly valid: {:?}",
            result.failures()
        );
    }

    // ...and this is what makes it a *defect* rather than a choice: claim
    // the sort the inputs had, and the order check fails.
    let si_name = format!("{}.si", sci.segment_name);
    let si_bytes = dir.open(&si_name).unwrap().to_vec();
    let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    si.index_sort = Some(vec![sort_field("rank", false, SortMissingValue::Last)]);
    dir.delete_file(&si_name).unwrap();
    write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();
    let results = crate::check_index::check_directory(&dir).unwrap();
    let failures: Vec<String> = results
        .iter()
        .flat_map(|r| r.failures())
        .map(|c| format!("{}: {}", c.name, c.message))
        .collect();
    assert!(
        failures.iter().any(|f| f.contains("sort")),
        "expected the sort check to fail, got {failures:?}"
    );
}

/// A **multi-tier, reverse-first** sort across a merge, with the second
/// tier doing most of the ordering: one ascending key with distinct
/// values is the case most likely to come out right by accident.
#[test]
fn a_merge_honours_every_tier_of_a_multi_field_reverse_sort() {
    let tmp = tempdir("sorted-merge-multi-tier");
    let dir = FsDirectory::open(&tmp);
    let mut writer = sorted_merge_writer(
        &dir,
        &[
            sort_field("rank", true, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ],
    );
    // Two rank groups of three, split across three flushes so that every
    // group spans every source and the tie tier decides within them.
    for batch in [
        vec![("a", 20i64, 1i64), ("d", 10, 1)],
        vec![("b", 20, 2), ("e", 10, 2)],
        vec![("c", 20, 3), ("f", 10, 3)],
    ] {
        for (id, rank, tie) in batch {
            add_sortable_doc(&mut writer, id, rank, tie);
        }
        writer.commit().unwrap();
    }
    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    // rank descending, then tie ascending.
    assert_every_format_agrees(
        &dir,
        sci,
        &[
            ("a", 20),
            ("b", 20),
            ("c", 20),
            ("d", 10),
            ("e", 10),
            ("f", 10),
        ],
    );
    assert_eq!(
        read_index_sort(&dir, sci).unwrap(),
        vec![
            sort_field("rank", true, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ]
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// A sorted merge whose sources have deletions: the survivors must come
/// out in sort order and the deleted documents must be gone from *every*
/// format, not just from the stored fields. Deletions also disable the
/// stored-fields and term-vector bulk-copy paths, so this exercises the
/// per-document ones.
#[test]
fn a_sorted_merge_drops_deleted_documents_from_every_format() {
    let tmp = tempdir("sorted-merge-deletions");
    let dir = FsDirectory::open(&tmp);
    let mut writer =
        sorted_merge_writer(&dir, &[sort_field("rank", false, SortMissingValue::Last)]);
    writer.set_merge_policy(None);
    for batch in [
        vec![("d", 40i64), ("a", 10)],
        vec![("e", 50), ("b", 20)],
        vec![("f", 60), ("c", 30)],
    ] {
        for (id, rank) in batch {
            add_sortable_doc(&mut writer, id, rank, 0);
        }
        writer.commit().unwrap();
    }
    // Delete one document from each segment, by its unique body term.
    for id in ["a", "e", "f"] {
        writer
            .delete_documents_by_term(&[Term::new("body", format!("t{id}").into_bytes())])
            .unwrap();
    }
    writer.commit().unwrap();
    assert_eq!(
        writer
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.del_count)
            .sum::<i32>(),
        3
    );

    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    writer.execute_merge(&names).unwrap();
    let infos = writer.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    assert_eq!(sci.del_count, 0, "the merge drops the deleted documents");
    assert_every_format_agrees(&dir, sci, &[("b", 20), ("c", 30), ("d", 40)]);
    // The deleted documents' terms must be gone from the dictionary too,
    // not merely unreachable.
    let seg = per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let tim = dir.open(&format!("{seg}.tim")).unwrap();
    let tip = dir.open(&format!("{seg}.tip")).unwrap();
    let tmd = dir.open(&format!("{seg}.tmd")).unwrap();
    let doc_bytes = dir.open(&format!("{seg}.doc")).unwrap();
    let block_fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &fi::FieldInfos {
            fields: sortable_fields_with(true, true),
        },
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        3,
    )
    .unwrap();
    let doc_in = DocInput::open(
        &doc_bytes,
        &sci.segment_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let body_terms = block_fields.field("body").unwrap();
    for id in ["a", "e", "f"] {
        assert!(
            body_terms
                .postings(format!("t{id}").as_bytes(), Some(&doc_in))
                .unwrap()
                .is_none(),
            "deleted document {id}'s term survived the merge"
        );
    }
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// A multi-field sort where the **second** tier is what actually orders
/// most of the batch, plus a reverse first tier -- a single ascending key
/// is the case most likely to come out right by accident.
#[test]
fn a_multi_field_sort_breaks_ties_with_its_second_tier() {
    let tmp = tempdir("sort-multi-field");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer
        .set_index_sort(Some(&[
            sort_field("rank", true, SortMissingValue::Last),
            sort_field("tie", false, SortMissingValue::Last),
        ]))
        .unwrap();

    // Two rank groups; within each, `tie` decides. Inserted in an order
    // that is wrong on both tiers.
    for (id, rank, tie) in [
        ("a", 1, 30),
        ("b", 2, 20),
        ("c", 1, 10),
        ("d", 2, 10),
        ("e", 1, 20),
    ] {
        writer
            .add_document(sortable_doc(id, rank, tie, "x"))
            .unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];

    // rank descending: the 2s first (d then b, by tie ascending), then
    // the 1s (c, e, a).
    assert_eq!(read_all_docs(&dir, &infos), vec!["d", "b", "c", "e", "a"]);
    assert_eq!(
        read_base_numeric_column(&dir, sci, 1, 5),
        vec![Some(2), Some(2), Some(1), Some(1), Some(1)]
    );
    assert_eq!(
        read_base_numeric_column(&dir, sci, 2, 5),
        vec![Some(10), Some(20), Some(10), Some(20), Some(30)]
    );
    assert_eq!(read_index_sort(&dir, sci).unwrap().len(), 2);
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// A buffer that is already in sort order still gets its sort recorded.
/// Java's `Sorter.sortAndLeaveUnpacked` returns `null` (no `DocMap`) in
/// this case, but `SegmentInfo.setIndexSort` was already called.
#[test]
fn an_already_ordered_buffer_is_still_recorded_as_sorted() {
    let tmp = tempdir("sort-identity");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
    for (id, rank) in [("a", 1), ("b", 2), ("c", 3)] {
        writer.add_document(sortable_doc(id, rank, 0, "x")).unwrap();
    }
    let infos = writer.commit().unwrap().clone();
    assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c"]);
    assert_eq!(read_index_sort(&dir, &infos.segments[0]).unwrap().len(), 1);
}

/// A document with no value for the sort field takes that field's
/// sentinel (`Long.MIN_VALUE` for `First`, `Long.MAX_VALUE` for `Last`)
/// and is then compared like any other value -- so `reverse` moves it to
/// the *other* end. This is what the `.si` this flush writes actually
/// says, and what `CheckIndex.testSort` checks.
#[test]
fn a_missing_sort_key_takes_its_sentinel_and_reverses_with_it() {
    for (reverse, expected) in [
        (false, vec!["a", "c", "b"]),
        // `Last` == Long.MAX_VALUE; reversed, the largest comes first.
        (true, vec!["b", "c", "a"]),
    ] {
        let tmp = tempdir(&format!("sort-missing-{reverse}"));
        let dir = FsDirectory::open(&tmp);
        let mut writer =
            IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer
            .set_index_sort(Some(&[sort_field("rank", reverse, SortMissingValue::Last)]))
            .unwrap();
        writer.add_document(sortable_doc("a", 10, 0, "x")).unwrap();
        // `b` has no `rank` field at all.
        writer
            .add_document(Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("b".to_string()),
                }],
            })
            .unwrap();
        writer.add_document(sortable_doc("c", 20, 0, "x")).unwrap();
        let infos = writer.commit().unwrap().clone();
        assert_eq!(read_all_docs(&dir, &infos), expected, "reverse={reverse}");
        // The doc-values column is sparse (b has no value) and the
        // segment still passes our own sort check, which re-derives the
        // comparator from the `.si`.
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(result.all_passed(), "{:?}", result.failures());
        }
    }
}

/// A segment-private delete's `docIDUpto` is a **pre-sort** buffer
/// position. `update_document` buffers the delete at the position of the
/// document it replaces, so on a sorted flush the limit has to be
/// compared against `newToOld(doc)` -- Java keeps the `Sorter.DocMap` on
/// the pooled `ReadersAndUpdates` for exactly this. Without the mapping
/// the sort silently changes which documents an update deletes.
#[test]
fn a_private_delete_limit_is_mapped_back_through_the_sort() {
    let tmp = tempdir("sort-private-delete");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(
        &dir,
        sortable_fields_with(true, false),
        "Lucene104",
        version(),
    )
    .unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    // Descending, so the buffer order and the sorted order are reversed
    // and an unmapped limit cuts exactly the wrong end off.
    writer
        .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
        .unwrap();

    // Buffer positions 0..2 all carry body term "dup"; position 3 is an
    // `update_document` whose delete has `docIDUpto = 3`, so it must
    // delete exactly the first three and not itself.
    for (id, rank) in [("p0", 10), ("p1", 20), ("p2", 30)] {
        writer
            .add_document(sortable_doc(id, rank, 0, "dup"))
            .unwrap();
    }
    writer
        .update_document(
            Term {
                field: "body".to_string(),
                bytes: b"dup".to_vec(),
            },
            sortable_doc("p3", 40, 0, "dup"),
        )
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];

    // Sorted descending, doc 0 is p3 (rank 40) -- the replacement, which
    // in the *pre-sort* space sat at position 3 and is therefore NOT
    // below the limit. The other three are deleted.
    assert_eq!(sci.del_count, 3);
    let liv = dir
        .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
        .unwrap();
    let live = lucene_codecs::live_docs::parse(&liv, &sci.segment_id, sci.del_gen, 4, 3).unwrap();
    assert!(live.get(0), "p3 (sorted first) must survive");
    for d in 1..4 {
        assert!(!live.get(d), "doc {d} must be deleted");
    }
}

/// A flush that fails **after** the sort must leave the buffer in
/// insertion order, because every buffered delete's `docIDUpto` is a
/// position in that order. Without the restore the retry sees an
/// already-sorted buffer, short-circuits to "no sort map", and then
/// compares pre-sort limits against sorted doc ids -- deleting the exact
/// complement of the right documents, with a valid `.liv` and a clean
/// `CheckIndex`.
///
/// The failure is injected the way a caller would hit it: a second
/// doc-values field with a non-numeric value on one document, which
/// `build_doc_values_output` rejects *after* `sort_pending_buffer` has
/// run.
#[test]
fn a_flush_that_fails_after_the_sort_leaves_the_buffer_in_insertion_order() {
    let tmp = tempdir("sort-failed-flush-retry");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        numeric_field("rank", 1),
        numeric_field("tie", 2),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..stored_only_field("body", 3)
        },
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", true, SortMissingValue::Last)]))
        .unwrap();

    for (id, rank) in [("p0", 10), ("p1", 20), ("p2", 30)] {
        writer
            .add_document(sortable_doc(id, rank, 0, "dup"))
            .unwrap();
    }
    // Position 3: the replacement, whose delete carries `docIDUpto = 3`.
    // Its `tie` is a *string*, so the doc-values build fails -- after the
    // sort has already permuted the buffer.
    writer
        .update_document(
            Term {
                field: "body".to_string(),
                bytes: b"dup".to_vec(),
            },
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("p3".to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(40),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("not a number".to_string()),
                    },
                    StoredField {
                        field_number: 3,
                        value: FieldValue::String("dup".to_string()),
                    },
                ],
            },
        )
        .unwrap();
    // The document index in the message is the doc id in the *segment
    // being built* (0 -- p3 sorts first under a descending rank), not the
    // caller's buffer position. That is the only observable trace the
    // sort leaves on an error, and it is the honest one: the value that
    // was rejected is at that doc id in the segment this flush was
    // writing.
    assert!(matches!(
        writer.flush().unwrap_err(),
        Error::NonNumericDocValue(ref f, 0, _) if f == "tie"
    ));

    // The buffer is back in insertion order -- checked directly, because
    // that is the invariant, not just its consequence.
    assert_eq!(
        writer
            .pending_docs
            .iter()
            .map(doc_value)
            .collect::<Vec<_>>(),
        vec!["p0", "p1", "p2", "p3"]
    );

    // Repair the offending document and retry. The delete must still
    // reach exactly the three documents that preceded it.
    writer.pending_docs[3].fields[2].value = FieldValue::Long(0);
    writer.commit().unwrap();
    let infos = writer.segment_infos().clone();
    let sci = &infos.segments[0];
    assert_eq!(sci.del_count, 3);
    let liv = dir
        .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
        .unwrap();
    let live = lucene_codecs::live_docs::parse(&liv, &sci.segment_id, sci.del_gen, 4, 3).unwrap();
    // Descending by rank, so p3 (rank 40) is doc 0 -- and in the pre-sort
    // space it sat at position 3, which is not below the limit.
    assert!(live.get(0), "p3 must survive the retried flush");
    for d in 1..4 {
        assert!(!live.get(d), "doc {d} must be deleted");
    }
}

/// `DocumentsWriterPerThread.updateDocuments`' `IllegalArgumentException`:
/// a block of documents must stay contiguous and in order, which an
/// index sort would shred, so a sorted writer without a parent field
/// ([`IndexWriter::set_parent_field`]) refuses the block when it is
/// added -- before anything is buffered -- and the writer goes on (Java
/// throws before `reserveOneDoc`). Refusing it only at flush, as this
/// port once did, left a buffer no flush could ever write.
#[test]
fn document_blocks_and_an_index_sort_are_refused_together() {
    let tmp = tempdir("sort-blocks");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
    writer.add_document(sortable_doc("a", 2, 0, "x")).unwrap();
    let before = writer.pending_docs.len();
    for err in [
        writer
            .add_documents(vec![
                sortable_doc("b", 1, 0, "x"),
                sortable_doc("c", 3, 0, "x"),
            ])
            .unwrap_err(),
        writer
            .update_documents(
                Term::new("body", "x"),
                vec![sortable_doc("d", 1, 0, "x"), sortable_doc("e", 3, 0, "x")],
            )
            .unwrap_err(),
    ] {
        assert!(matches!(err, Error::BlocksWithIndexSortNeedParentField));
    }
    assert_eq!(writer.pending_docs.len(), before, "nothing was buffered");
    assert!(!writer.pending_has_blocks);
    // A one-document "block" is a plain document, as in Java.
    writer
        .add_documents(vec![sortable_doc("f", 1, 0, "x")])
        .unwrap();
    writer.commit().unwrap();
    assert_eq!(writer.segment_infos().segments.len(), 1);
    assert!(!parsed_si(&dir, &writer.segment_infos().segments[0]).has_blocks);
}

/// `IndexWriter.updateNumericDocValue`: rewriting the column the
/// segment's physical order is defined over would leave every existing
/// segment claiming a sort it no longer satisfies.
#[test]
fn a_doc_values_update_on_a_sort_field_is_refused() {
    let tmp = tempdir("sort-dv-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
    let term = Term {
        field: "body".to_string(),
        bytes: b"x".to_vec(),
    };
    let err = writer
        .update_numeric_doc_value(term.clone(), "rank", 7)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DocValuesUpdateOnIndexSortField { field, .. } if field == "rank"
    ));
    // `updateBinaryDocValue` checks the sort before the doc-values type
    // (Lucene 10.6, #16565): the numeric sort field is reported as a sort
    // field, not as a type mismatch.
    let err = writer
        .update_binary_doc_value(term.clone(), "rank", b"bar")
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DocValuesUpdateOnIndexSortField { field, .. } if field == "rank"
    ));
    // A doc-values field that is *not* part of the sort is still fine.
    writer.update_numeric_doc_value(term, "tie", 7).unwrap();
}

/// Two doc-values fields land in one `.dvm`/`.dvd` pair, which is what
/// makes a multi-field index sort expressible at all.
#[test]
fn two_doc_values_fields_share_one_dvm() {
    let tmp = tempdir("multi-dv");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    assert!(matches!(
        writer.add_doc_values_field("tie").unwrap_err(),
        Error::DuplicateDocValuesField(f) if f == "tie"
    ));
    writer.add_document(sortable_doc("a", 1, 100, "x")).unwrap();
    writer.add_document(sortable_doc("b", 2, 200, "x")).unwrap();
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    assert_eq!(
        read_base_numeric_column(&dir, sci, 1, 2),
        vec![Some(1), Some(2)]
    );
    assert_eq!(
        read_base_numeric_column(&dir, sci, 2, 2),
        vec![Some(100), Some(200)]
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// `doc_values::write_dense_fields` accepts a sparse column only for
/// NUMERIC, so a multi-field configuration cannot silently drop another
/// type's sparse docs -- and a sparse NUMERIC tier, which is what an
/// index sort with missing keys needs, goes through.
#[test]
fn a_multi_field_doc_values_flush_allows_a_sparse_numeric_but_not_a_sparse_sorted() {
    // Sparse NUMERIC: accepted, and readable as sparse.
    let tmp = tempdir("multi-dv-sparse-numeric");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("tie").unwrap();
    writer.add_document(sortable_doc("a", 1, 100, "x")).unwrap();
    // No `tie` value on this one.
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("b".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(2),
                },
            ],
        })
        .unwrap();
    let infos = writer.commit().unwrap().clone();
    let sci = &infos.segments[0];
    assert_eq!(
        read_base_numeric_column(&dir, sci, 2, 2),
        vec![Some(100), None]
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }

    // Sparse SORTED alongside it: written as a sparse SORTED column in the
    // same `.dvm`/`.dvd`, and clean.
    let tmp = tempdir("multi-dv-sparse-sorted");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        numeric_field("rank", 1),
        sorted_field("label", 2),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer.add_doc_values_field("label").unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(1),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::String("x".to_string()),
                },
            ],
        })
        .unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("b".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(2),
                },
            ],
        })
        .unwrap();
    writer.commit().unwrap();
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// An index-sorted segment *is* offered to the merge policy now, because
/// the merge preserves the sort. What must not happen is a merge whose
/// sources **disagree** about the sort: concatenating them would produce
/// a segment in one order whose `.si` describes another, which is valid,
/// `CheckIndex`-clean and wrong. Java raises
/// `IllegalArgumentException("cannot change index sort ...")`
/// (`IndexWriter.validateIndexSort`); this refuses the merge.
///
/// The `.si` is patched after the fact rather than written by a sorted
/// flush, so the sort declaration is the *only* difference between the
/// three segments.
/// Java's `SegmentMerger` takes `maxDoc` from `SegmentReader.maxDoc()`,
/// i.e. from the `.si`. This port's `execute_merge` read it off the
/// `.fdm` instead, where `stored_fields::open` checks only that it is
/// non-negative -- and `stored_fields.rs` says so at the site ("This port
/// has no `SegmentInfo` to hand, so the `.fdm` copy *is* the document
/// count").
///
/// `merge_segments` then sizes two `Vec`s from it, per source: the live
/// doc-id list and the dense doc-id map. A four-byte edit claiming
/// `maxDoc = i32::MAX` makes each of those ~8.6 GB -- an allocation
/// abort, which is the one failure `catch_unwind` at the FFI boundary
/// cannot intercept (`docs/arithmetic-gate.md`'s fourth row).
///
/// Without the cross-check this test does not fail politely: it takes the
/// container's whole memory cap and is killed.
#[test]
fn a_segment_whose_fdm_disagrees_with_its_si_about_max_doc_is_not_merged() {
    /// Overwrites the `.fdm`'s own `maxDoc`, leaving every other file
    /// alone, and re-signs the footer so only the semantic disagreement
    /// can fire.
    ///
    /// Layout: `CodecUtil.writeIndexHeader` (magic, codec name, version,
    /// id, suffix), a vint `chunkSize`, then the big-endian `maxDoc`.
    fn patch_fdm_max_doc(fdm: &mut [u8], expected_now: i32, max_doc: i32) {
        // The field is *found* rather than computed: the vint `chunkSize`
        // between the index header and `maxDoc` is a variable width. The
        // search asserts the match is unique in the window, so a wrong
        // offset cannot pass silently and patch some other field.
        let start = lucene_store::codec_util::index_header_length("Lucene90FieldsIndexMeta", "");
        // Little-endian: Lucene 9 switched `DataInput.readInt` to LE, and
        // this is `read_i32`, not `read_be_i32`. (The codec *footer*
        // below is still big-endian, which is why the two differ here.)
        let want = expected_now.to_le_bytes();
        let hits: Vec<usize> = (start..start + 8)
            .filter(|&p| fdm[p..p + 4] == want)
            .collect();
        assert_eq!(hits.len(), 1, "maxDoc offset is ambiguous: {hits:?}");
        let at = hits[0];
        fdm[at..at + 4].copy_from_slice(&max_doc.to_le_bytes());
        let n = fdm.len();
        let crc = crc32fast::hash(&fdm[..n - 8]) as u64;
        fdm[n - 8..].copy_from_slice(&crc.to_be_bytes());
    }

    let tmp = tempdir("fdm-vs-si-max-doc");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    // Three documents in the first segment and two in the second, so the
    // `maxDoc` this test patches is `3` -- a value that cannot be
    // confused with the run of zero bytes around it.
    for d in 0..3 {
        writer.add_document(doc(&format!("a{d}"))).unwrap();
    }
    writer.commit().unwrap();
    for d in 0..2 {
        writer.add_document(doc(&format!("b{d}"))).unwrap();
    }
    writer.commit().unwrap();
    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    assert_eq!(names.len(), 2);

    // The control first: these two segments merge cleanly, so the failure
    // below is the edit and not the fixture.
    {
        let tmp_ok = tempdir("fdm-vs-si-max-doc-control");
        let dir_ok = FsDirectory::open(&tmp_ok);
        let mut ok = IndexWriter::open(
            &dir_ok,
            vec![stored_only_field("id", 0)],
            "Lucene104",
            version(),
        )
        .unwrap();
        for d in 0..3 {
            ok.add_document(doc(&format!("a{d}"))).unwrap();
        }
        ok.commit().unwrap();
        for d in 0..2 {
            ok.add_document(doc(&format!("b{d}"))).unwrap();
        }
        ok.commit().unwrap();
        let ok_names: Vec<String> = ok
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        ok.execute_merge(&ok_names).unwrap();
        assert_eq!(ok.segment_infos().segments.len(), 1);
    }

    let fdm_name = format!("{}.fdm", names[0]);
    let mut fdm = dir.open(&fdm_name).unwrap().to_vec();
    patch_fdm_max_doc(&mut fdm, 3, i32::MAX);
    dir.delete_file(&fdm_name).unwrap();
    write_file(&dir, &fdm_name, &fdm).unwrap();

    assert!(
        matches!(
            writer.execute_merge(&names),
            Err(Error::SegmentDocCountMismatch {
                si_doc_count: 3,
                stored_fields_max_doc: i32::MAX,
                ..
            })
        ),
        "a .fdm claiming 2 billion documents must be reported, not reserved for"
    );
}

#[test]
fn a_field_length_past_integer_max_value_saturates_instead_of_wrapping() {
    // Java steps `FieldInvertState.length` with
    // `Math.addExact(invertState.length, 1)` -- it throws rather than
    // wrapping past `Integer.MAX_VALUE`. The `+=` this replaced wrapped a
    // `u32` instead: the longest document in the index would encode to a
    // *small* norm and score as one of the shortest, silently, in every
    // BM25 query over the field. It also trips `int_to_byte4`'s own
    // `debug_assert` on the way past `i32::MAX`.
    const MAX: u32 = i32::MAX as u32;
    assert_eq!(accumulate_field_length(0, 0), 0);
    assert_eq!(accumulate_field_length(3, 4), 7);
    assert_eq!(accumulate_field_length(MAX, 1), MAX, "clamped, not wrapped");
    assert_eq!(accumulate_field_length(MAX - 1, 5), MAX);
    assert_eq!(accumulate_field_length(u32::MAX, 1), MAX);
    // A negative frequency can only come from an occurrence count above
    // `i32::MAX`, i.e. "longer than the longest", never "shorter".
    assert_eq!(accumulate_field_length(0, -1), MAX);
    // And the whole reachable domain still encodes monotonically.
    assert!(
        small_float::int_to_byte4(accumulate_field_length(MAX, 1))
            >= small_float::int_to_byte4(accumulate_field_length(1_000, 1))
    );
}

#[test]
fn a_merge_of_segments_that_disagree_about_the_index_sort_is_refused() {
    let tmp = tempdir("sort-merge-disagreement");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    for batch in 0..3 {
        writer.add_document(doc(&format!("d{batch}"))).unwrap();
        writer.commit().unwrap();
    }
    assert_eq!(writer.segment_infos().segments.len(), 3);
    assert_eq!(writer.segment_stats().unwrap().len(), 3);

    // Declare a sort on the middle segment's `.si` only.
    let sci = writer.segment_infos().segments[1].clone();
    let si_name = format!("{}.si", sci.segment_name);
    let si_bytes = dir.open(&si_name).unwrap().to_vec();
    let mut si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    si.index_sort = Some(vec![sort_field("rank", false, SortMissingValue::Last)]);
    dir.delete_file(&si_name).unwrap();
    write_file(&dir, &si_name, &segment_info::write(&si, "")).unwrap();

    // Still a merge candidate -- the exclusion is gone.
    let stats = writer.segment_stats().unwrap();
    assert_eq!(stats.len(), 3);
    assert!(stats.iter().any(|s| s.name == sci.segment_name));

    let names: Vec<String> = writer
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    assert!(matches!(
        writer.execute_merge(&names),
        Err(Error::MergeSortDisagreement { .. })
    ));
}

/// Two sorted flushes from one writer produce two segments that each
/// claim the sort and are each internally ordered -- the precondition
/// a sort-preserving merge is built on.
#[test]
fn every_sorted_flush_gets_its_own_ordered_segment() {
    let tmp = tempdir("sort-two-segments");
    let dir = FsDirectory::open(&tmp);
    let mut writer = IndexWriter::open(&dir, sortable_fields(), "Lucene104", version()).unwrap();
    writer.set_doc_values_field(Some("rank")).unwrap();
    writer
        .set_index_sort(Some(&[sort_field("rank", false, SortMissingValue::Last)]))
        .unwrap();
    writer.add_document(sortable_doc("b", 20, 0, "x")).unwrap();
    writer.add_document(sortable_doc("a", 10, 0, "x")).unwrap();
    writer.commit().unwrap();
    writer.add_document(sortable_doc("d", 40, 0, "x")).unwrap();
    writer.add_document(sortable_doc("c", 30, 0, "x")).unwrap();
    let infos = writer.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 2);
    for sci in &infos.segments {
        assert_eq!(read_index_sort(&dir, sci).unwrap().len(), 1);
    }
    assert_eq!(read_all_docs(&dir, &infos), vec!["a", "b", "c", "d"]);
}

// --- points ---

fn points_field(name: &str, number: i32, dims: i32, index_dims: i32, bytes: i32) -> FieldInfo {
    FieldInfo {
        point_dimension_count: dims,
        point_index_dimension_count: index_dims,
        point_num_bytes: bytes,
        ..stored_only_field(name, number)
    }
}

fn points_config(dims: i32, bytes: i32) -> PointsFieldConfig {
    PointsFieldConfig {
        name: "p".to_string(),
        field_number: 1,
        num_dims: dims,
        num_index_dims: dims,
        bytes_per_dim: bytes,
    }
}

#[test]
fn add_points_field_validates_its_field() {
    let tmp = tempdir("points-config");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("plain", 0),
        points_field("p", 1, 2, 1, 4),
        points_field("too_many_index_dims", 2, 2, 3, 4),
        points_field("too_wide", 3, 1, 1, 17),
        points_field("too_many_dims", 4, 17, 1, 4),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    assert!(matches!(
        writer.add_points_field("missing"),
        Err(Error::UnknownPointsField(name)) if name == "missing"
    ));
    assert!(matches!(
        writer.add_points_field("plain"),
        Err(Error::UnsupportedPointsField(_, 0, 0, 0))
    ));
    for name in ["too_many_index_dims", "too_wide", "too_many_dims"] {
        assert!(
            matches!(
                writer.add_points_field(name),
                Err(Error::UnsupportedPointsField(..))
            ),
            "{name}"
        );
    }
    writer.add_points_field("p").unwrap();
    assert!(matches!(
        writer.add_points_field("p"),
        Err(Error::DuplicatePointsField(name)) if name == "p"
    ));
}

#[test]
fn encode_point_uses_the_sortable_encodings_and_rejects_wrong_shapes() {
    let encode = |config: &PointsFieldConfig, value: FieldValue| {
        let mut out = Vec::new();
        encode_point(config, 7, &value, &mut out).map(|()| out)
    };
    let (int, long) = (points_config(1, 4), points_config(1, 8));
    assert_eq!(
        encode(&int, FieldValue::Int(-1)).unwrap(),
        [0x7f, 0xff, 0xff, 0xff]
    );
    assert_eq!(
        encode(&long, FieldValue::Long(1)).unwrap(),
        [0x80, 0, 0, 0, 0, 0, 0, 1]
    );
    // Unsigned byte order is numeric order, negatives included.
    let floats: Vec<Vec<u8>> = [-2.5f32, -0.0, 0.0, 1.5]
        .into_iter()
        .map(|v| encode(&int, FieldValue::Float(v)).unwrap())
        .collect();
    assert!(floats.windows(2).all(|w| w[0] < w[1]), "{floats:?}");
    let doubles: Vec<Vec<u8>> = [-1e300f64, -1.0, 0.0, 2.0]
        .into_iter()
        .map(|v| encode(&long, FieldValue::Double(v)).unwrap())
        .collect();
    assert!(doubles.windows(2).all(|w| w[0] < w[1]), "{doubles:?}");
    // Every NaN encodes as Java's canonical one, above `+inf` -- a
    // negative or payload-carrying NaN included, which `to_bits` alone
    // would put below `-inf`.
    let canonical = encode(&int, FieldValue::Float(f32::NAN)).unwrap();
    assert!(canonical > encode(&int, FieldValue::Float(f32::INFINITY)).unwrap());
    for nan in [
        -f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0000),
    ] {
        assert_eq!(encode(&int, FieldValue::Float(nan)).unwrap(), canonical);
    }
    let canonical = encode(&long, FieldValue::Double(f64::NAN)).unwrap();
    assert!(canonical > encode(&long, FieldValue::Double(f64::INFINITY)).unwrap());
    for nan in [-f64::NAN, f64::from_bits(0x7ff8_0000_0000_0001)] {
        assert_eq!(encode(&long, FieldValue::Double(nan)).unwrap(), canonical);
    }
    let two_ints = points_config(2, 4);
    assert_eq!(
        encode(&two_ints, FieldValue::Binary(vec![1; 8])).unwrap(),
        vec![1; 8]
    );

    for (config, value, kind) in [
        (&int, FieldValue::Long(1), "Long"),
        (&long, FieldValue::Int(1), "Int"),
        (&long, FieldValue::Float(1.0), "Float"),
        (&int, FieldValue::Double(1.0), "Double"),
        (&two_ints, FieldValue::Int(1), "Int"),
        (&two_ints, FieldValue::Binary(vec![1; 4]), "Binary"),
        (&int, FieldValue::String("1".to_string()), "String"),
    ] {
        let err = encode(config, value).unwrap_err();
        assert!(
            matches!(&err, Error::PointValueShape(name, 7, k, _, _) if name == "p" && *k == kind),
            "{err:?}"
        );
        // The message names the shape it needed.
        assert!(err.to_string().contains("dimension(s)"), "{err}");
    }
}

#[test]
fn a_flush_writes_points_only_for_fields_that_got_one() {
    let tmp = tempdir("points-flush");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![
        stored_only_field("id", 0),
        points_field("n", 1, 1, 1, 8),
        points_field("unused", 2, 1, 1, 4),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_points_field("n").unwrap();
    writer.add_points_field("unused").unwrap();
    for i in 0..1200i64 {
        let mut fields = vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("d{i}")),
        }];
        // Two values on every third document, none on every fifth.
        if i % 5 != 0 {
            fields.push(StoredField {
                field_number: 1,
                value: FieldValue::Long(1000 - i),
            });
        }
        if i % 3 == 0 {
            fields.push(StoredField {
                field_number: 1,
                value: FieldValue::Long(-i),
            });
        }
        writer.add_document(Document { fields }).unwrap();
    }
    writer.commit().unwrap();
    let names: Vec<String> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(names.iter().any(|n| n.ends_with(".kdd")), "{names:?}");
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }

    // A value of the wrong shape fails the flush, naming the document.
    writer
        .add_document(Document {
            fields: vec![StoredField {
                field_number: 2,
                value: FieldValue::Long(1),
            }],
        })
        .unwrap();
    assert!(matches!(
        writer.commit(),
        Err(Error::PointValueShape(name, 0, "Long", 1, 4)) if name == "unused"
    ));
}

#[test]
fn a_flush_with_no_point_values_writes_no_points_files() {
    let tmp = tempdir("points-none");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), points_field("n", 1, 1, 1, 8)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_points_field("n").unwrap();
    writer
        .add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("only".to_string()),
            }],
        })
        .unwrap();
    writer.commit().unwrap();
    let names: Vec<String> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(!names.iter().any(|n| n.ends_with(".kdd")), "{names:?}");
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn a_merge_whose_sources_have_no_points_claims_none() {
    // Two flushes of a declared points field no document carries: each
    // flushed `.fnm` omits the absent field, and the merged one must too
    // -- a merged `.fnm` claiming point dimensions over a segment with no
    // `.kdm` is one real Lucene refuses to open.
    let tmp = tempdir("points-none-merged");
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0), points_field("n", 1, 1, 1, 8)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.add_points_field("n").unwrap();
    for batch in 0..3 {
        writer
            .add_document(Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("d{batch}")),
                }],
            })
            .unwrap();
        writer.commit().unwrap();
    }
    writer.set_merge_policy(Some(tight_merge_policy()));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1, "the three flushes merged");
    let sci = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
    assert!(fis.fields.iter().all(|f| f.name != "n"));

    let names: Vec<String> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(!names.iter().any(|n| n.ends_with(".kdm")), "{names:?}");
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

// --- segments real Lucene wrote ---

/// `fixtures/data/compound_index`, copied: one **compound** segment real
/// Lucene 10.5.0 flushed -- five documents, `id` a `StringField` ("0" ..
/// "4", postings only, field 0) and `num` a `NumericDocValuesField`
/// (`10i`, field 1).
fn lucene_compound_index(tag: &str) -> TempDir {
    let src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/compound_index");
    let tmp = tempdir(tag);
    for name in ["_0.cfs", "_0.cfe", "_0.si", "segments_1"] {
        std::fs::copy(src.join(name), tmp.join(name)).unwrap();
    }
    tmp
}

/// This writer's schema for the fixture's fields, numbered the **other**
/// way round, so nothing can pass by assuming the two agree.
fn compound_fixture_fields() -> Vec<FieldInfo> {
    vec![
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..stored_only_field("num", 0)
        },
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..stored_only_field("id", 1)
        },
    ]
}

#[test]
fn deletes_and_merges_reach_into_a_lucene_compound_segment() {
    let tmp = lucene_compound_index("lucene-compound");
    let dir = FsDirectory::open(&tmp);
    let mut writer =
        IndexWriter::open(&dir, compound_fixture_fields(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("id")).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();
    for i in 5..8i64 {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(i.to_string()),
                    },
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Long(i * 10),
                    },
                ],
            })
            .unwrap();
        if i == 5 {
            // Two segments of this writer's beside Lucene's one, so the
            // merge policy below has more than a tier's worth to merge.
            writer.commit().unwrap();
        }
    }
    // The delete has to find "2" in the Java segment's term dictionary,
    // inside its `.cfs`, through *its* numbering (`id` is field 0 there).
    writer
        .delete_documents_by_term(&[Term::new("id", b"2".to_vec())])
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let java = sis
        .segments
        .iter()
        .find(|s| s.segment_name == "_0")
        .unwrap();
    assert_eq!(java.del_count, 1, "the delete reached the compound segment");

    // Now merge the compound segment -- with its `.liv`, which lives
    // beside the archive -- and this writer's own.
    writer.set_merge_policy(Some(tight_merge_policy()));
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1, "merged");
    let merged = &sis.segments[0];
    assert_eq!(merged.del_count, 0);
    let fnm = dir.open(&format!("{}.fnm", merged.segment_name)).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &merged.segment_id, "").unwrap();
    let num = fis.fields.iter().find(|f| f.name == "num").unwrap();
    let id = fis.fields.iter().find(|f| f.name == "id").unwrap();
    assert_eq!(num.doc_values_type, DocValuesType::Numeric);
    assert_eq!(id.index_options, IndexOptions::Docs);
    let si = segment_info::parse(
        &dir.open(&format!("{}.si", merged.segment_name)).unwrap(),
        &merged.segment_id,
    )
    .unwrap();
    assert_eq!(si.doc_count, 7);
    let seg = per_field_segment(&merged.segment_name, DOC_VALUES_FORMAT_NAME);
    let (_, meta) = doc_values::parse_meta(
        &dir.open(&format!("{seg}.dvm")).unwrap(),
        &merged.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &fis,
    )
    .unwrap();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
    let entry = meta.numeric_entry(num.number).unwrap();
    let mut values: Vec<i64> = (0..si.doc_count)
        .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
        .collect();
    values.sort_unstable();
    assert_eq!(
        values,
        [0, 10, 30, 40, 50, 60, 70],
        "doc 2 gone, the rest kept"
    );

    // The merged postings still find a Java document's term, and the
    // deleted one's is gone.
    writer
        .delete_documents_by_term(&[
            Term::new("id", b"3".to_vec()),
            Term::new("id", b"2".to_vec()),
        ])
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments[0].del_count, 1);
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// A doc-values update -- a soft delete, to OpenSearch -- against a
/// segment Java flushed compound: the base column is read out of the
/// `.cfs`, the generation is written loose beside it, and a merge folds it.
#[test]
fn doc_values_updates_reach_into_a_lucene_compound_segment() {
    let tmp = lucene_compound_index("lucene-compound-dv-update");
    let dir = FsDirectory::open(&tmp);
    let mut writer =
        IndexWriter::open(&dir, compound_fixture_fields(), "Lucene104", version()).unwrap();
    writer.set_postings_field(Some("id")).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();
    writer
        .update_numeric_doc_value(Term::new("id", b"3".to_vec()), "num", 99)
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    let java = &sis.segments[0];
    assert_eq!(java.doc_values_gen, 1, "a generation beside the archive");
    assert!(dir.list_all().unwrap().iter().any(|f| f == "_0_1.fnm"));
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    // A second round reads the first generation, not the archive's base.
    writer
        .update_numeric_doc_value(Term::new("id", b"1".to_vec()), "num", 11)
        .unwrap();
    writer.commit().unwrap();

    writer.set_merge_policy(Some(tight_merge_policy()));
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("5".to_string()),
                },
                StoredField {
                    field_number: 0,
                    value: FieldValue::Long(50),
                },
            ],
        })
        .unwrap();
    writer.commit().unwrap();
    writer
        .add_document(Document {
            fields: vec![
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("6".to_string()),
                },
                StoredField {
                    field_number: 0,
                    value: FieldValue::Long(60),
                },
            ],
        })
        .unwrap();
    let sis = writer.commit().unwrap().clone();
    assert_eq!(sis.segments.len(), 1, "merged");
    let merged = &sis.segments[0];
    let fnm = dir.open(&format!("{}.fnm", merged.segment_name)).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &merged.segment_id, "").unwrap();
    let num = fis.fields.iter().find(|f| f.name == "num").unwrap();
    let seg = per_field_segment(&merged.segment_name, DOC_VALUES_FORMAT_NAME);
    let (_, meta) = doc_values::parse_meta(
        &dir.open(&format!("{seg}.dvm")).unwrap(),
        &merged.segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &fis,
    )
    .unwrap();
    let dvd = dir.open(&format!("{seg}.dvd")).unwrap();
    let entry = meta.numeric_entry(num.number).unwrap();
    let mut reader = doc_values::NumericReader::new(&dvd, entry);
    let mut values: Vec<i64> = (0..7).map(|d| reader.value(d).unwrap().unwrap()).collect();
    values.sort_unstable();
    assert_eq!(
        values,
        [0, 11, 20, 40, 50, 60, 99],
        "both updates survive the merge"
    );
    for result in crate::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// Terms given back with `recycle` come out of `recycled` empty, with their
/// arrays, and a term read from them is the one pushed to them last.
#[test]
fn recycled_inverted_terms_keep_arrays_and_no_terms() {
    let mut t = InvertedTerms::recycled(4, 64, 8);
    t.push(b"alpha", 2, &[0, 5], &[], &[] as &[&[u8]]);
    t.push(b"beta", 1, &[3], &[], &[] as &[&[u8]]);
    let ends_capacity = t.ram_bytes();
    t.recycle();
    let mut again = InvertedTerms::recycled(1, 1, 1);
    assert!(again.is_empty());
    assert!(again.ram_bytes() >= ends_capacity);
    again.push(b"gamma", 1, &[7], &[], &[] as &[&[u8]]);
    let got = again.get(0);
    assert_eq!(
        (got.term, got.freq, got.positions),
        (&b"gamma"[..], 1, &[7][..])
    );
    assert_eq!(again.len(), 1);
}

/// A big document's terms are not kept by the thread.
#[test]
fn oversized_inverted_terms_are_not_pooled() {
    let mut big = InvertedTerms::with_capacity(1, 1 << 20, 0);
    big.push(b"x", 1, &[], &[], &[] as &[&[u8]]);
    big.recycle();
    let next = InvertedTerms::recycled(0, 0, 0);
    assert!(next.ram_bytes() < 1 << 20);
}
