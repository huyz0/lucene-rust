#![cfg(test)]
use super::*;
use crate::multi_segment::search_term_query_multi_segment;
use crate::query::TermQuery;
use lucene_index::segment_infos::LuceneVersion;
use lucene_store::FsDirectory;

fn fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    ))
}

/// Opens the real single-segment `blocktree_index` fixture end-to-end
/// through `DirectoryReader::open` (reading its real `segments_1`, not a
/// hand-built `SegmentInfos`), confirming file-existence detection finds
/// the real postings files and produces a working, single-segment
/// `OpenSegment` whose query results match the manually wired
/// `search_term_query_multi_segment` test.
#[test]
fn opens_real_single_segment_fixture_and_matches_manual_wiring() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open real segments_1 commit");

    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    assert_eq!(seg.doc_base, 0);
    assert!(seg.doc_buf.is_some(), "fixture has real .doc postings");

    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();

    let query = TermQuery::new("body", "cat");
    let norms = [None];
    let hits = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    // Same fixture/query multi_segment.rs's own test uses
    // (`search_term_query_multi_segment_merges_two_real_segments`,
    // segment 0 half): must be non-empty and score-descending.
    assert!(!hits.is_empty());
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

/// A visitor that fails the test if it is asked about, or handed, a hidden field.
struct Spy<'a> {
    hidden: &'a std::collections::HashSet<i32>,
    asked: usize,
    saw: usize,
}

impl lucene_codecs::stored_fields::StoredFieldVisitor for Spy<'_> {
    fn needs_field(
        &mut self,
        n: i32,
    ) -> lucene_codecs::stored_fields::Result<lucene_codecs::stored_fields::VisitStatus> {
        assert!(!self.hidden.contains(&n), "asked about hidden field {n}");
        self.asked += 1;
        Ok(lucene_codecs::stored_fields::VisitStatus::Yes)
    }
    fn string_field(&mut self, n: i32, _: &str) -> lucene_codecs::stored_fields::Result<()> {
        assert!(!self.hidden.contains(&n));
        self.saw += 1;
        Ok(())
    }
}

/// A view hides every named field from postings, doc values, norms, points and stored
/// documents, shows the others as they were, and a hidden `_field_names` entry is not found.
#[test]
fn a_view_hides_fields_from_every_reader() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let names: std::collections::HashSet<String> = seg
        .field_infos
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    assert!(!names.is_empty());
    // Nothing to hide: a plain clone with no hidden set.
    let none = reader
        .with_hidden_fields(&vec![std::collections::HashSet::new(); 1])
        .unwrap();
    assert!(none.segment_readers()[0].hidden_fields().is_none());
    // A view is its own reader, whatever it hides.
    let view = reader
        .with_hidden_fields(std::slice::from_ref(&names))
        .unwrap();
    let v = &view.segment_readers()[0];
    assert!(v.field_infos.fields.is_empty());
    assert_eq!(v.hidden_fields().unwrap().len(), names.len());
    for f in &seg.field_infos.fields {
        assert!(v.is_hidden(f.number) && !seg.is_hidden(f.number));
        assert!(v.doc_values_for_field(f.number).is_none());
        assert!(v.norms_entry(f.number).is_none());
    }
    assert_eq!(view.doc_freq("body", b"cat").unwrap(), 0);
    assert!(reader.doc_freq("body", b"cat").unwrap() > 0);
    // Stored documents: the visitor is never offered a hidden field.
    let hidden = v.hidden_fields().unwrap();
    let mut spy = Spy {
        hidden,
        asked: 0,
        saw: 0,
    };
    v.visit_stored_document(0, &mut spy).unwrap();
    let mut plain = Spy {
        hidden: &std::collections::HashSet::new(),
        asked: 0,
        saw: 0,
    };
    seg.visit_stored_document(0, &mut plain).unwrap();
    assert_eq!(spy.asked, 0);
    assert!(plain.asked >= spy.asked);
    // Segment count mismatch is refused.
    assert!(reader.with_hidden_fields(&[]).is_err());
}

/// `docFreq` is the number of documents holding the term (the fixture has
/// no deletions), and 0 for a term or a field that is not there.
#[test]
fn doc_freq_counts_the_documents_holding_a_term() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let (_, total) = crate::multi_segment::search_term_query_multi_segment_counting(
        &segments,
        &TermQuery::new("body", "cat"),
        &[None],
        1000,
        u64::MAX,
    )
    .unwrap();
    assert!(total.value > 0);
    assert_eq!(reader.doc_freq("body", b"cat").unwrap(), total.value as i64);
    assert_eq!(reader.doc_freq("body", b"no-such-term").unwrap(), 0);
    assert_eq!(reader.doc_freq("no-such-field", b"cat").unwrap(), 0);
}

/// A stored-fields-only segment (no postings/doc-values/norms/term
/// vectors at all -- exactly `segment_writer.rs`'s
/// `flush_stored_only_segment` output) must open cleanly with no
/// postings files, not error out just because they're missing.
#[test]
fn a_segments_stored_documents_are_its_stored_fields_readers() {
    // The codec-level reader is verified against Lucene by the
    // stored-fields fixture tests; through the segment reader every
    // document must come back the same.
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/stored_fields_index"
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let seg = &reader.segment_readers()[0];
    let f = seg.stored.as_deref().expect("stored fields");
    let direct =
        lucene_codecs::stored_fields::open(&f.fdt, &f.fdx, &f.fdm, &seg.segment_id, &f.suffix)
            .unwrap();
    assert!(seg.max_doc > 0);
    for doc in 0..seg.max_doc {
        let got = seg.stored_document(doc).unwrap().expect("a document");
        let want = direct.document(doc).unwrap();
        assert_eq!(got.fields.len(), want.fields.len());
        for (g, w) in got.fields.iter().zip(&want.fields) {
            assert_eq!((g.field_number, &g.value), (w.field_number, &w.value));
        }
        assert!(!got.fields.is_empty());
    }
    assert!(
        seg.stored_document(seg.max_doc).is_err(),
        "past the segment"
    );
    // A segment with no stored-fields files answers none.
    let mut bare = seg.clone();
    bare.stored = None;
    assert!(bare.stored_document(0).unwrap().is_none());
}

#[test]
fn stored_fields_only_segment_opens_without_postings_files() {
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);

    let field_infos_list = vec![FieldInfo {
        name: "title".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
        vector_similarity_function: lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
    }];
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("hello".to_string()),
        }],
    }];

    let lucene_version = segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    let commit_info = lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        "_0",
        [7u8; ID_LENGTH],
        "Lucene104",
        lucene_version,
        &field_infos_list,
        &docs,
        false,
    )
    .expect("flush stored-only segment");

    let segment_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation: 1,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![commit_info],
        user_data: vec![],
    };

    let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open stored-only");
    let seg = &reader.segment_readers()[0];
    assert!(seg.doc_buf.is_none());
    assert!(seg.pos_buf.is_none());
    assert!(seg.pay_buf.is_none());
    assert!(seg.live_docs.is_none());
    assert_eq!(seg.max_doc, 1);

    std::fs::remove_dir_all(&dir_path).ok();
}

/// A stored-fields-only segment flushed with `use_compound_file: true`
/// (so only `_0.cfs`/`_0.cfe`/`_0.si` exist on disk -- no loose `.fnm`)
/// must open through `DirectoryReader` with the same doc count and "no
/// postings" shape as the non-compound flush of the same documents,
/// proving the `.fnm` read out of the compound archive succeeds (a
/// missing/unreadable `.fnm` would fail `DirectoryReader::open_at`
/// itself, since field infos are required to open any segment). Also
/// confirms a *loose* `.fnm` genuinely does not exist next to the
/// compound segment -- so success here can only come from reading
/// through `.cfs`, not from an accidental loose-file fallback. Does NOT
/// compare field name/number directly (`SegmentReader` doesn't expose
/// parsed field infos publicly); [`compound_file_segment_opens_and_is_queryable`]
/// below covers that indirectly by running a real term query, which
/// only succeeds if the field/term data read out of `.cfs` is correct.
#[test]
fn compound_flushed_segment_opens_with_field_infos_matching_non_compound_flush() {
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);

    let lucene_version = segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("hello".to_string()),
        }],
    }];

    // Loose flush of the same documents, for a same-shape baseline.
    let loose_commit = lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        "_0",
        [7u8; ID_LENGTH],
        "Lucene104",
        lucene_version,
        &stored_only_field_infos(),
        &docs,
        false,
    )
    .expect("flush loose segment");
    assert!(dir_path.join("_0.fnm").exists());

    let loose_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation: 1,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![loose_commit],
        user_data: vec![],
    };
    let loose_reader = DirectoryReader::open_at(&dir, loose_infos).expect("open loose");
    let loose_seg = &loose_reader.segment_readers()[0];

    // Compound flush of the same documents into a second segment, in
    // the same directory (a distinct segment name/id so both coexist).
    let compound_commit = lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        "_1",
        [8u8; ID_LENGTH],
        "Lucene104",
        lucene_version,
        &stored_only_field_infos(),
        &docs,
        true,
    )
    .expect("flush compound segment");
    assert!(dir_path.join("_1.cfs").exists());
    assert!(dir_path.join("_1.cfe").exists());
    assert!(
        !dir_path.join("_1.fnm").exists(),
        "compound flush must not also write a loose .fnm"
    );

    let compound_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation: 1,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![compound_commit],
        user_data: vec![],
    };
    let compound_reader =
        DirectoryReader::open_at(&dir, compound_infos).expect("open compound segment");
    let compound_seg = &compound_reader.segment_readers()[0];

    // Both must have the same doc count and same "no postings" shape --
    // the compound one exercised entirely through `.cfs`/`.cfe`.
    assert_eq!(compound_seg.max_doc, loose_seg.max_doc);
    assert!(compound_seg.doc_buf.is_none());
    assert!(compound_seg.pos_buf.is_none());
    assert!(compound_seg.pay_buf.is_none());
    assert!(compound_seg.live_docs.is_none());

    std::fs::remove_dir_all(&dir_path).ok();
}

/// A compound segment whose `.cfs` data has been truncated (corrupting
/// its header/footer) must surface as a typed error out of
/// [`CompoundArchive::open`]'s `compound_format::check_data_header_footer`
/// call, not panic or silently read garbage.
#[test]
fn compound_file_segment_with_truncated_cfs_is_a_typed_error() {
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let lucene_version = segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("hello".to_string()),
        }],
    }];
    let commit = lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        "_0",
        [7u8; ID_LENGTH],
        "Lucene104",
        lucene_version,
        &stored_only_field_infos(),
        &docs,
        true,
    )
    .expect("flush compound segment");

    let cfs_path = dir_path.join("_0.cfs");
    let mut bytes = std::fs::read(&cfs_path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&cfs_path, bytes).unwrap();

    let segment_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation: 1,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![commit],
        user_data: vec![],
    };
    let err = DirectoryReader::open_at(&dir, segment_infos)
        .expect_err("truncated .cfs must not open successfully");
    assert!(
        matches!(err, Error::CompoundFormat(_)),
        "expected a typed CompoundFormat error, got {err:?}"
    );

    std::fs::remove_dir_all(&dir_path).ok();
}

/// A compound segment whose `.cfe` entry table is truncated must
/// surface as a typed error out of `compound_format::parse_entries`
/// rather than panicking on an out-of-bounds read.
#[test]
fn compound_file_segment_with_truncated_cfe_is_a_typed_error() {
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let lucene_version = segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("hello".to_string()),
        }],
    }];
    let commit = lucene_index::segment_writer::flush_stored_only_segment(
        &dir,
        "_0",
        [7u8; ID_LENGTH],
        "Lucene104",
        lucene_version,
        &stored_only_field_infos(),
        &docs,
        true,
    )
    .expect("flush compound segment");

    let cfe_path = dir_path.join("_0.cfe");
    let mut bytes = std::fs::read(&cfe_path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&cfe_path, bytes).unwrap();

    let segment_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation: 1,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: None,
        segments: vec![commit],
        user_data: vec![],
    };
    let err = DirectoryReader::open_at(&dir, segment_infos)
        .expect_err("truncated .cfe must not open successfully");
    assert!(
        matches!(err, Error::CompoundFormat(_)),
        "expected a typed CompoundFormat error, got {err:?}"
    );

    std::fs::remove_dir_all(&dir_path).ok();
}

/// Confirms `doc_base` is computed as the running sum of previous
/// segments' `maxDoc`, by opening the same real fixture segment twice
/// under one hand-built two-segment `SegmentInfos` (the same "open the
/// same fixture twice" trick the cross-segment tests use).
#[test]
fn doc_base_is_running_sum_of_previous_max_docs() {
    let dir = FsDirectory::open(fixture_dir());
    let commit = read_commit_info(&dir);

    let mut segment_infos = read_commit(&dir);
    segment_infos.segments.push(commit);

    let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open two segments");
    assert_eq!(reader.segment_readers().len(), 2);
    let first_max_doc = reader.segment_readers()[0].max_doc;
    assert_eq!(reader.segment_readers()[0].doc_base, 0);
    assert_eq!(reader.segment_readers()[1].doc_base, first_max_doc);
}

/// Segments whose `.si` `maxDoc`s sum past `IndexWriter.MAX_DOCS` are a
/// corrupt index (`SegmentInfos.readCommit`'s "Too many documents"), not
/// an `i32` doc base that overflows. The two segments here are reused
/// readers whose `max_doc` is set as a damaged `.si` would set it.
#[test]
fn segments_summing_past_max_docs_are_a_corruption_error() {
    let dir = FsDirectory::open(fixture_dir());
    let mut segment_infos = read_commit(&dir);
    segment_infos.segments.push(read_commit_info(&dir));
    let reader = DirectoryReader::open_at(&dir, segment_infos.clone()).unwrap();
    let mut reusable = reader.segments.clone();
    for seg in &mut reusable {
        seg.max_doc = i32::MAX - 200;
    }
    let err = DirectoryReader::open_at_reusing(&dir, segment_infos.clone(), &reusable)
        .expect_err("a total past MAX_DOCS must be refused");
    assert!(
        matches!(&err, Error::Store(lucene_store::Error::Corrupted(m))
                if m.contains("Too many documents")),
        "{err:?}"
    );

    // Up to the limit is still an index. Both entries name the same
    // segment, so both reuse the first candidate.
    let limit = crate::reader::multi_reader::MAX_DOCS;
    reusable[0].max_doc = i32::try_from(limit / 2).unwrap();
    let reader = DirectoryReader::open_at_reusing(&dir, segment_infos, &reusable).unwrap();
    assert_eq!(reader.segment_readers()[1].doc_base, reusable[0].max_doc);
    assert_eq!(i64::from(reader.max_doc()), limit / 2 * 2);
}

/// `num_docs` is counted once, when the live docs are set, and follows
/// every path that sets them.
#[test]
fn num_docs_is_kept_in_step_with_the_live_docs() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    assert_eq!(seg.num_docs(), seg.max_doc);
    let mut live = FixedBitSet::new(seg.max_doc as usize);
    live.set(0);
    let filtered = seg.with_live_docs(live);
    assert_eq!(filtered.num_docs(), 1);
    assert_eq!(crate::reader::IndexReader::num_docs(&filtered), 1);
    assert_eq!(seg.num_docs(), seg.max_doc);
}

/// A mask per segment replaces that segment's live docs; `None` keeps them,
/// and a mask of another length than the segment is refused.
#[test]
fn live_docs_masks_replace_a_segments_live_docs() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let mut live = FixedBitSet::new(seg.max_doc as usize);
    live.set(0);
    let masked = reader.with_live_docs_masks(vec![Some(live)]).unwrap();
    assert_eq!(masked.segment_readers()[0].num_docs(), 1);
    assert_eq!(masked.max_doc(), seg.max_doc);
    assert_eq!(reader.segment_readers()[0].num_docs(), seg.max_doc);
    let kept = reader.with_live_docs_masks(vec![None]).unwrap();
    assert_eq!(kept.segment_readers()[0].num_docs(), seg.max_doc);
    let short = FixedBitSet::new(1);
    assert!(seg.max_doc == 1 || reader.with_live_docs_masks(vec![Some(short)]).is_err());
    assert!(reader.with_live_docs_masks(Vec::new()).is_err());
}

/// The host's view: a mask replaces the live docs, `None` makes every document live.
#[test]
fn host_live_docs_replace_a_segments_live_docs() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let mut live = FixedBitSet::new(seg.max_doc as usize);
    live.set(0);
    let masked = reader.with_host_live_docs(&[Some(live)]).unwrap();
    assert_eq!(masked.segment_readers()[0].num_docs(), 1);
    let all = reader.with_host_live_docs(&[None]).unwrap();
    assert_eq!(all.segment_readers()[0].num_docs(), seg.max_doc);
    assert!(reader.with_host_live_docs(&[]).is_err());
}

fn read_commit(dir: &FsDirectory) -> SegmentInfos {
    segment_infos::read_latest(dir).expect("read real segments_1")
}

fn read_commit_info(dir: &FsDirectory) -> segment_infos::SegmentCommitInfo {
    read_commit(dir).segments[0].clone()
}

/// A segment with real deletions (`del_gen != -1`) must open its `.liv`
/// file and populate `live_docs`, cross-checked against the real
/// `del_count` -- exercises the branch
/// `opens_real_single_segment_fixture_and_matches_manual_wiring` and
/// `stored_fields_only_segment_opens_without_postings_files` both skip.
#[test]
fn segment_with_deletions_opens_live_docs() {
    let dir_path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/live_docs_index/"
    ));
    let dir = FsDirectory::open(&dir_path);
    let reader = DirectoryReader::open(&dir).expect("open real live_docs_index commit");

    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    assert_eq!(seg.max_doc, 5);
    let live = seg.live_docs.as_ref().expect("segment has deletions");
    assert_eq!(live.cardinality(), 3);
    assert!(live.get(0) && live.get(2) && live.get(4));
    assert!(!live.get(1) && !live.get(3));

    // Also has real postings (.tim/.tip/.tmd/.doc) but no .pos/.pay,
    // exercising the "some but not all optional postings files" path.
    assert!(seg.doc_buf.is_some());
    assert!(seg.pos_buf.is_none());
    assert!(seg.pay_buf.is_none());
}

/// A real, Java-written compound-file (`.cfs`/`.cfe`) segment must open
/// through `DirectoryReader` exactly like a loose segment would -- same
/// field infos, same doc count, same postings query results -- reading
/// every sub-file (`.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`, this fixture has no
/// `.pos`/`.pay`) out of the `.cfs` archive instead of failing to find
/// loose files that were never written.
#[test]
fn compound_file_segment_opens_and_is_queryable() {
    let dir_path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/compound_index/"
    ));
    let dir = FsDirectory::open(&dir_path);
    let reader = DirectoryReader::open(&dir).expect("open real compound-file segment");

    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    assert_eq!(seg.max_doc, 5);
    assert!(
        seg.doc_buf.is_some(),
        "compound fixture has real .doc postings"
    );
    assert!(seg.pos_buf.is_none(), "compound fixture has no .pos");
    assert!(seg.pay_buf.is_none(), "compound fixture has no .pay");
    // A member is a window onto the `.cfs`, not a copy of it.
    let doc = seg.doc_buf.as_deref().unwrap();
    assert!(matches!(doc, Input::Slice { .. }), "{doc:?}");

    // GenCompoundFormat.java indexes docs 0..5 with a `StringField("id",
    // ...)` -- confirms the `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc` bytes
    // pulled out of the compound archive are actually usable for a real
    // query, not merely non-empty.
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let query = TermQuery::new("id", "3");
    let norms = [None];
    let hits = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 3);
}

/// A segment whose `.si` lists no `.fnm` file at all is corrupt (every
/// real segment has field infos); this must surface as an error, not a
/// panic.
#[test]
fn segment_missing_fnm_file_is_an_error() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);

    let si = SegmentInfo {
        id: [1u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 3,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![], // no .fnm listed
        attributes: vec![],
        index_sort: None,
    };
    let si_bytes = segment_info::write(&si, "");
    std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [1u8; ID_LENGTH],
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
    let err = SegmentReader::open(&dir, &commit, 0).expect_err("missing .fnm must error");
    assert!(matches!(err, Error::Store(_)));

    std::fs::remove_dir_all(&dir_path).ok();
}

/// `.si` listing only some of `.tim`/`.tip`/`.tmd` (a corrupt/partial
/// segment -- never produced by a real writer) must be rejected rather
/// than silently treated as "no postings" or panicking on an `unwrap`.
#[test]
fn partial_blocktree_files_is_an_error() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);

    let field_infos_list = vec![lucene_codecs::field_infos::FieldInfo {
        name: "title".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: lucene_codecs::field_infos::IndexOptions::None,
        doc_values_type: lucene_codecs::field_infos::DocValuesType::None,
        doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
        vector_similarity_function: lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
    }];
    let fnm = lucene_codecs::field_infos::write(&field_infos_list, &[2u8; ID_LENGTH], "");
    std::fs::write(dir_path.join("_0.fnm"), &fnm).unwrap();
    // Only `.tim` present -- `.tip`/`.tmd` deliberately missing.
    std::fs::write(dir_path.join("_0_x.tim"), [0u8; 4]).unwrap();

    let si = SegmentInfo {
        id: [2u8; ID_LENGTH],
        version: segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 1,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec!["_0.fnm".to_string(), "_0_x.tim".to_string()],
        attributes: vec![],
        index_sort: None,
    };
    let si_bytes = segment_info::write(&si, "");
    std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

    let commit = segment_infos::SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id: [2u8; ID_LENGTH],
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
    let err = SegmentReader::open(&dir, &commit, 0).expect_err("partial postings must error");
    assert!(matches!(err, Error::PartialBlockTreeFiles { found: 1, .. }));

    std::fs::remove_dir_all(&dir_path).ok();
}

fn stored_only_field_infos() -> Vec<lucene_codecs::field_infos::FieldInfo> {
    use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
    vec![FieldInfo {
        name: "title".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
        vector_similarity_function: lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
    }]
}

fn flush_stored_only(
    dir: &FsDirectory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    text: &str,
) -> segment_infos::SegmentCommitInfo {
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(text.to_string()),
        }],
    }];
    let lucene_version = segment_info::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    };
    lucene_index::segment_writer::flush_stored_only_segment(
        dir,
        segment_name,
        segment_id,
        "Lucene104",
        lucene_version,
        &stored_only_field_infos(),
        &docs,
        false,
    )
    .expect("flush stored-only segment")
}

fn write_commit(
    dir: &FsDirectory,
    generation: i64,
    segments: Vec<segment_infos::SegmentCommitInfo>,
) -> SegmentInfos {
    let segment_infos = SegmentInfos {
        id: [9u8; ID_LENGTH],
        generation,
        format_version: segment_infos::VERSION_86,
        lucene_version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        index_created_version_major: 10,
        version: generation,
        counter: segments.len() as i64,
        min_segment_lucene_version: None,
        segments,
        user_data: vec![],
    };
    segment_infos::write(&segment_infos, dir).expect("write segments_N");
    segment_infos
}

/// Reopening with no writes since `self` was opened returns `None` --
/// real `openIfChanged`'s "nothing to do" convention, checked purely off
/// the `segments_N` generation (no need to re-diff segment lists).
#[test]
fn open_if_changed_returns_none_when_generation_unchanged() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0]);

    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    let reopened = reader.open_if_changed(&dir).expect("open_if_changed");
    assert!(reopened.is_none());

    std::fs::remove_dir_all(&dir_path).ok();
}

/// Reopening after a genuinely new segment was added returns `Some` with
/// both segments visible, correct `doc_base`s, and the *first* segment's
/// reader genuinely reused rather than re-read from disk: proven by
/// deleting the old segment's on-disk files before reopening -- if
/// `open_if_changed` tried to re-open segment `_0` from disk it would
/// fail (file not found), so success here is only possible via reuse.
#[test]
fn open_if_changed_reuses_unchanged_segment_and_opens_new_one() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);

    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    assert_eq!(reader.segment_readers().len(), 1);

    let commit1 = flush_stored_only(&dir, "_1", [2u8; ID_LENGTH], "world");
    write_commit(&dir, 2, vec![commit0.clone(), commit1]);

    // Remove segment _0's on-disk files: a fresh open of _0 would now
    // fail, so if open_if_changed still succeeds and reports the right
    // doc counts, it must have reused the already-open reader instead of
    // re-reading the (now-missing) files.
    for ext in [".fdt", ".fdx", ".fdm", ".fnm"] {
        std::fs::remove_file(dir_path.join(format!("_0{ext}"))).ok();
    }

    let reopened = reader
        .open_if_changed(&dir)
        .expect("open_if_changed")
        .expect("segments_2 differs from segments_1");
    assert_eq!(reopened.segment_readers().len(), 2);
    assert_eq!(reopened.segment_readers()[0].segment_name, "_0");
    assert_eq!(reopened.segment_readers()[0].doc_base, 0);
    assert_eq!(reopened.segment_readers()[1].segment_name, "_1");
    assert_eq!(reopened.segment_readers()[1].doc_base, 1);

    std::fs::remove_dir_all(&dir_path).ok();
}

/// The correctness-critical case: a del_gen-only change on an *existing*
/// segment (no new segment, same name/id) must NOT be served from the
/// reused reader's stale `live_docs` -- it must reopen that segment and
/// pick up the new deletions.
#[test]
fn open_if_changed_reopens_segment_whose_del_gen_changed() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);

    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    assert!(reader.segment_readers()[0].live_docs.is_none());

    // Delete doc 0 in segment _0: bumps del_gen, writes a new .liv.
    let commit0_deleted = lucene_index::deletes::apply_deletes(&dir, &commit0, None, 1, [0i32])
        .expect("apply delete");
    assert_eq!(commit0_deleted.del_gen, 1);
    write_commit(&dir, 2, vec![commit0_deleted]);

    let reopened = reader
        .open_if_changed(&dir)
        .expect("open_if_changed")
        .expect("segments_2 differs from segments_1");
    assert_eq!(reopened.segment_readers().len(), 1);
    let live = reopened.segment_readers()[0]
        .live_docs
        .as_ref()
        .expect("del_gen change must be picked up as new live_docs");
    assert!(!live.get(0), "doc 0 must now be marked deleted");

    std::fs::remove_dir_all(&dir_path).ok();
}

/// A doc-values update commit bumps `fieldInfosGen`/`docValuesGen` and
/// leaves `delGen` alone. Java's reuse test is `delGen == ... &&
/// fieldInfosGen == ...` (`StandardDirectoryReader
/// .createOrReuseSegmentReader`); matching on `del_gen` alone, as this used
/// to, silently served the *previous* commit's doc values forever after a
/// numeric doc-values update.
///
/// Proof mechanism is the mirror of the reuse test above: the segment's
/// `.fnm` is deleted before reopening, so a genuine re-open of `_0` must
/// fail. Failure is therefore the assertion -- it is only reachable if the
/// reader declined to reuse.
#[test]
fn open_if_changed_reopens_a_segment_whose_doc_values_gen_changed() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);

    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    assert_eq!(reader.segment_readers()[0].doc_values_gen, -1);

    let mut updated = commit0.clone();
    updated.doc_values_gen = 0;
    updated.field_infos_gen = 0;
    write_commit(&dir, 2, vec![updated]);

    std::fs::remove_file(dir_path.join("_0.fnm")).expect("remove .fnm");
    let err = reader.open_if_changed(&dir);
    assert!(
        err.is_err(),
        "a doc-values generation bump must force a fresh open, not reuse a \
             reader holding the previous generation's doc values"
    );

    std::fs::remove_dir_all(&dir_path).ok();
}

/// A commit that changes nothing about a segment -- same name, id, and all
/// three generations -- reuses the whole reader, `.fnm` gone or not. The
/// companion to the test above: together they pin *both* directions of
/// Java's reuse condition, so neither can regress into "always reuse" or
/// "never reuse".
#[test]
fn open_if_changed_reuses_a_segment_whose_generations_all_match() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);
    let reader = DirectoryReader::open(&dir).expect("open segments_1");

    // A new commit generation, but `_0` itself is untouched.
    write_commit(&dir, 2, vec![commit0]);
    std::fs::remove_file(dir_path.join("_0.fnm")).expect("remove .fnm");

    let reopened = reader
        .open_if_changed(&dir)
        .expect("open_if_changed")
        .expect("segments_2 differs from segments_1");
    assert_eq!(reopened.segment_readers().len(), 1);
    assert_eq!(reopened.segment_readers()[0].max_doc, 1);

    std::fs::remove_dir_all(&dir_path).ok();
}

/// The delete-only reopen must be Java's "only liveDocs changed" branch:
/// the shared core (`SegmentCoreReaders` -- term dictionary, field infos,
/// postings buffers) is carried over and *only* the new `.liv` is read.
///
/// Every one of the segment's own files is deleted before reopening, so a
/// full re-open would fail; success plus refreshed live docs proves the
/// core was shared and the `.liv` re-read, rather than the whole segment
/// being decoded again to change one bitset.
#[test]
fn a_delete_only_reopen_shares_the_core_and_rereads_only_the_liv_file() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);

    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    assert!(reader.segment_readers()[0].live_docs().is_none());

    let commit0_deleted = lucene_index::deletes::apply_deletes(&dir, &commit0, None, 1, [0i32])
        .expect("apply delete");
    write_commit(&dir, 2, vec![commit0_deleted]);

    // Everything except the freshly written `.liv` and `segments_2`.
    for ext in [".fdt", ".fdx", ".fdm", ".fnm", ".si"] {
        std::fs::remove_file(dir_path.join(format!("_0{ext}"))).ok();
    }

    let reopened = reader
        .open_if_changed(&dir)
        .expect("a delete-only reopen must not re-read the segment's own files")
        .expect("segments_2 differs from segments_1");
    let live = reopened.segment_readers()[0]
        .live_docs()
        .expect("the new .liv must have been read");
    assert!(!live.get(0), "doc 0 must now be marked deleted");
    assert_eq!(reopened.segment_readers()[0].del_gen, 1);

    std::fs::remove_dir_all(&dir_path).ok();
}

/// [`DirectoryReader::reopen_at`] takes the new `SegmentInfos` from its
/// caller rather than from `segments_N` -- which is why it exists: an NRT
/// refresh's infos were never committed. Same proof as the
/// `open_if_changed` reuse test above: `_0`'s files are removed first, so
/// the reopen can only succeed by reusing the open reader, and no
/// `segments_2` is ever written.
#[test]
fn reopen_at_reuses_unchanged_segments_from_caller_supplied_infos() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);
    let reader = DirectoryReader::open(&dir).expect("open segments_1");

    let commit1 = flush_stored_only(&dir, "_1", [2u8; ID_LENGTH], "world");
    let mut infos = reader.segment_infos.clone();
    infos.segments = vec![commit0, commit1];
    for ext in [".fdt", ".fdx", ".fdm", ".fnm"] {
        std::fs::remove_file(dir_path.join(format!("_0{ext}"))).ok();
    }

    let reopened = reader.reopen_at(&dir, infos).expect("reopen_at");
    assert_eq!(reopened.segment_readers().len(), 2);
    assert_eq!(reopened.segment_readers()[1].segment_name, "_1");
    assert_eq!(reopened.segment_readers()[1].doc_base, 1);
    assert!(Arc::ptr_eq(
        &reader.segment_readers()[0].fields,
        &reopened.segment_readers()[0].fields
    ));

    std::fs::remove_dir_all(&dir_path).ok();
}

/// Reusing a segment must be a refcount bump, not a copy of its decoded
/// state. `Arc::ptr_eq` on the term dictionary is the observable form of
/// Java's `oldReader.incRef()`; without it, every reopen deep-copied the
/// segment's FST and field metadata.
#[test]
fn a_reused_segment_shares_its_core_rather_than_copying_it() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);
    let reader = DirectoryReader::open(&dir).expect("open segments_1");

    let commit1 = flush_stored_only(&dir, "_1", [2u8; ID_LENGTH], "world");
    write_commit(&dir, 2, vec![commit0, commit1]);
    let reopened = reader
        .open_if_changed(&dir)
        .expect("open_if_changed")
        .expect("segments_2 differs from segments_1");

    let before = &reader.segment_readers()[0];
    let after = &reopened.segment_readers()[0];
    assert!(Arc::ptr_eq(&before.fields, &after.fields));
    assert!(Arc::ptr_eq(&before.field_infos, &after.field_infos));

    std::fs::remove_dir_all(&dir_path).ok();
}

/// Norms must be reachable through the reader: a caller who opened an index
/// the normal way would otherwise have to pass `None` for norms and score
/// every document at `UNNORMED_FIELD_LENGTH` -- a wrong BM25
/// length-normalization term, not just a missing feature.
///
/// Checked against the real Java-written `blocktree_index` fixture, whose
/// `body` field has real `.nvm`/`.nvd` files.
#[test]
fn opens_real_norms_and_scores_differently_from_the_no_norms_fallback() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open real segments_1");
    let seg = &reader.segment_readers()[0];

    assert!(
        seg.norms_data().is_some(),
        "the fixture has real .nvd norms"
    );
    let body = seg
        .field_infos()
        .field_by_name("body")
        .expect("fixture has a body field");
    assert!(seg.norms_entry(body.number).is_some());
    // A field the segment does not have has no norms entry.
    assert!(seg.norms_entry(9999).is_none());

    let norms = seg.field_norms("body").unwrap().expect("body has norms");
    assert!(
        norms.avg_field_length > 0.0,
        "avgFieldLength must come from the field's real .tmd counters"
    );
    assert!(seg.field_norms("no_such_field").unwrap().is_none());

    // Scoring with real norms must actually differ from the no-norms
    // fallback -- otherwise wiring them in would be decoration.
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let query = TermQuery::new("body", "cat");
    let scored = search_term_query_multi_segment(&segments, &query, &[Some(&norms)], 10)
        .expect("scored with norms");
    let unnormed =
        search_term_query_multi_segment(&segments, &query, &[None], 10).expect("scored without");
    let mut scored_docs: Vec<i32> = scored.iter().map(|h| h.doc_id).collect();
    let mut unnormed_docs: Vec<i32> = unnormed.iter().map(|h| h.doc_id).collect();
    scored_docs.sort_unstable();
    unnormed_docs.sort_unstable();
    assert_eq!(
        scored_docs, unnormed_docs,
        "the same documents match either way -- norms change ranking, not matching"
    );
    // And they genuinely re-rank: the fixture's `body:cat` documents have
    // different lengths, so the shorter one outranks the longer one only
    // once real norms are in play.
    assert_ne!(
        scored.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        unnormed.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        "real norms must actually affect the ranking"
    );
}

/// A segment with no `.nvm`/`.nvd` at all (this port's stored-fields-only
/// flush) must report no norms rather than erroring -- the same
/// "missing is not an error" contract `.tim`/`.dvm` already have.
#[test]
fn a_segment_without_norms_files_reports_none() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0]);
    let reader = DirectoryReader::open(&dir).expect("open segments_1");
    let seg = &reader.segment_readers()[0];
    assert!(seg.norms_data().is_none());
    assert!(seg.norms_entry(0).is_none());
    assert!(seg.field_norms("title").unwrap().is_none());
    assert!(seg.soft_deletes_field().is_none());

    std::fs::remove_dir_all(&dir_path).ok();
}

/// `codec_suffix_of`'s doc comment describes a trap (a segment name ending
/// in a digit) and two on-disk shapes; three call sites now depend on it
/// getting both right, and the compound-stripped case reaches `""` through
/// `unwrap_or_default()` -- correct by fallback rather than by
/// construction, which is exactly the kind of thing to pin.
#[test]
fn codec_suffix_is_derived_for_every_on_disk_file_name_shape() {
    // Loose files: `<segment>_<suffix><ext>`.
    assert_eq!(
        codec_suffix_of("_0_Lucene104_0.tim", "_0", ".tim"),
        "Lucene104_0"
    );
    assert_eq!(
        codec_suffix_of("_0_Lucene90_0.dvm", "_0", ".dvm"),
        "Lucene90_0"
    );
    // No suffix at all -- this port's own writer. The generic path would
    // read the segment name's trailing digit as the suffix.
    assert_eq!(codec_suffix_of("_0.nvm", "_0", ".nvm"), "");
    assert_eq!(codec_suffix_of("_10.tim", "_10", ".tim"), "");
    // Compound entries: the writer already stripped the segment name
    // (`IndexFileNames.stripSegmentName`).
    assert_eq!(
        codec_suffix_of("_Lucene104_0.tim", "_0", ".tim"),
        "Lucene104_0"
    );
    assert_eq!(codec_suffix_of(".nvm", "_0", ".nvm"), "");
}

/// A segment carrying one of `.nvm`/`.nvd` is corrupt, and must say so
/// rather than degrading to "this field has no norms" -- scoring without
/// norms is silently wrong, which is the whole reason norms are opened.
#[test]
fn a_segment_with_only_half_a_norms_pair_is_an_error() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);

    // Fabricate a `.nvm` with no `.nvd` beside it, and list it in the `.si`
    // the reader consults.
    std::fs::write(dir_path.join("_0.nvm"), b"not really norms").unwrap();
    let si_bytes = dir.open("_0.si").unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit0.segment_id).unwrap();
    si.files.push("_0.nvm".to_string());
    si.files.sort();
    std::fs::write(dir_path.join("_0.si"), segment_info::write(&si, "")).unwrap();

    let err = DirectoryReader::open(&dir).unwrap_err();
    assert!(
        matches!(err, Error::PartialNormsFiles { .. }),
        "expected PartialNormsFiles, got {err:?}"
    );

    std::fs::remove_dir_all(&dir_path).ok();
}

/// Two of the three stored-fields files is a damaged segment, not one
/// with no stored fields.
#[test]
fn a_segment_with_part_of_its_stored_fields_files_is_an_error() {
    let dir_path = tempdir();
    let dir = FsDirectory::open(&dir_path);
    let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
    write_commit(&dir, 1, vec![commit0.clone()]);
    let si_bytes = dir.open("_0.si").unwrap();
    let mut si = segment_info::parse(&si_bytes, &commit0.segment_id).unwrap();
    assert!(si.files.iter().any(|f| f.ends_with(".fdx")));
    si.files.retain(|f| !f.ends_with(".fdx"));
    std::fs::write(dir_path.join("_0.si"), segment_info::write(&si, "")).unwrap();

    let err = DirectoryReader::open(&dir).unwrap_err();
    assert!(
        matches!(err, Error::PartialStoredFieldsFiles { found: 2, .. }),
        "expected PartialStoredFieldsFiles, got {err:?}"
    );
}

/// Stored-fields metadata that does not parse fails each read and is not
/// kept: the segment still opens when the file's checksum holds (Java
/// opens its stored-fields reader with the segment core; this one
/// verifies the `.fdm`'s footer at open and parses it on the first
/// document).
#[test]
fn stored_fields_metadata_that_does_not_parse_is_an_error_every_time() {
    let dir_path = tempdir();
    let src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/stored_fields_index"
    );
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dir_path.join(e.file_name())).unwrap();
    }
    let fdm = dir_path.join("_0.fdm");
    let bytes = std::fs::read(&fdm).unwrap();
    // Half the metadata, under a footer whose checksum matches it.
    let mut half = bytes[..bytes.len() / 2].to_vec();
    lucene_store::codec_util::write_footer(&mut half);
    std::fs::write(&fdm, &half).unwrap();
    let reader = DirectoryReader::open(&FsDirectory::open(&dir_path)).unwrap();
    let seg = &reader.segment_readers()[0];
    for _ in 0..2 {
        assert!(matches!(
            seg.stored_document(0),
            Err(Error::StoredFields(_))
        ));
    }
    assert!(seg.stored.as_deref().unwrap().meta.get().is_none());
    assert!(format!("{:?}", seg.stored.as_deref().unwrap()).contains("meta_read: false"));
}

/// A mapped `.fdt` advised random reads every document as one advised
/// otherwise does, through a prefetch and without one; advised auto it
/// is small next to memory and keeps the OS read-ahead.
#[test]
fn stored_fields_advised_random_read_as_unadvised() {
    use lucene_store::read_advice;
    let src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/stored_fields_index"
    );
    let dir = lucene_store::MmapDirectory::open(src);
    read_advice::set("fdt:random").unwrap();
    let advised = DirectoryReader::open(&dir);
    read_advice::set("").unwrap();
    let advised = advised.unwrap();
    let plain = DirectoryReader::open(&FsDirectory::open(src)).unwrap();
    for (a, p) in advised
        .segment_readers()
        .iter()
        .zip(plain.segment_readers())
    {
        assert!(a.stored.as_deref().unwrap().will_need_chunks);
        assert!(!p.stored.as_deref().unwrap().will_need_chunks);
        let docs: Vec<i32> = (0..a.max_doc).collect();
        a.prefetch_stored_documents(&docs).unwrap();
        for doc in 0..a.max_doc {
            assert_eq!(
                a.stored_document(doc).unwrap(),
                p.stored_document(doc).unwrap()
            );
        }
        // A read with no prefetch before it on another thread.
        std::thread::scope(|s| {
            s.spawn(|| {
                for doc in (0..a.max_doc).rev() {
                    assert_eq!(
                        a.stored_document(doc).unwrap(),
                        p.stored_document(doc).unwrap()
                    );
                }
            });
        });
    }
    let auto = DirectoryReader::open(&dir).unwrap();
    assert_eq!(
        read_advice::for_extension("fdt"),
        read_advice::ReadAdvice::Auto
    );
    assert!(
        !auto.segment_readers()[0]
            .stored
            .as_deref()
            .unwrap()
            .will_need_chunks
    );
}

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("directory-reader")
}

// -- "NRT reopen after sparse doc-values commits" (docs/parity.md) --
//
// Every other doc-values test in this crate (`doc_value_query.rs`,
// `facets.rs`) feeds hand-loaded fixture bytes straight to the low-level
// `lucene_codecs::doc_values` readers, never through
// `DirectoryReader`/`SegmentReader`. These tests are the real
// `IndexWriter` -> commit -> NRT reopen -> read-back path: a genuinely
// sparse NUMERIC field (some docs opted out entirely), read through a
// `DirectoryReader` opened by this module, refreshed via
// `open_if_changed` across a second commit that adds a brand-new sparse
// segment.
mod sparse_doc_values_nrt {
    use super::*;
    use lucene_codecs::doc_values;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

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

    fn numeric_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
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

    fn doc_without_score(id: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            }],
        }
    }

    /// The base scenario: one segment, one commit, a sparse NUMERIC
    /// field with a present/absent/present pattern, read back correctly
    /// through `DirectoryReader::open` -- not a hand-loaded fixture, the
    /// actual bytes `IndexWriter::commit` wrote.
    #[test]
    fn nrt_reader_reads_sparse_numeric_field_after_first_commit() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        writer.add_document(doc_without_score("b")).unwrap();
        writer.add_document(doc_with_score("c", 7)).unwrap();
        writer.commit().unwrap();

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];

        let field_number = seg
            .field_infos()
            .field_by_number(1)
            .expect("score field present")
            .number;
        let meta = seg
            .doc_values_meta()
            .expect("sparse-committed segment must have .dvm/.dvd wired in");
        let entry = meta
            .numeric_entry(field_number)
            .expect("score has a NumericEntry");
        let data = seg.doc_values_data().expect("segment has .dvd bytes");

        assert_eq!(doc_values::numeric_value(data, entry, 0).unwrap(), Some(5));
        assert_eq!(
            doc_values::numeric_value(data, entry, 1).unwrap(),
            None,
            "doc 1 opted out of the field entirely -- sparse, not zero"
        );
        assert_eq!(doc_values::numeric_value(data, entry, 2).unwrap(), Some(7));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// The genuine NRT-reopen scenario: a second commit adds a brand-new
    /// *second* segment, itself also sparse, with a different
    /// present/absent pattern. `open_if_changed` must refresh doc-values
    /// wiring for the newly-opened segment while continuing to serve the
    /// first (reused) segment's sparse data correctly too.
    #[test]
    fn nrt_reopen_reads_new_segments_sparse_numeric_values_correctly() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_doc_values_field(Some("score")).unwrap();

        writer.add_document(doc_with_score("a", 5)).unwrap();
        writer.add_document(doc_without_score("b")).unwrap();
        writer.commit().unwrap();

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert_eq!(reader.segment_readers().len(), 1);

        // Second commit: a new segment, sparse in the opposite pattern
        // (present, absent, present) so this is not just re-reading the
        // first segment's bytes.
        writer.add_document(doc_without_score("x")).unwrap();
        writer.add_document(doc_with_score("y", 42)).unwrap();
        writer.add_document(doc_without_score("z")).unwrap();
        writer.commit().unwrap();

        let reopened = reader
            .open_if_changed(&dir)
            .expect("open_if_changed")
            .expect("second commit differs from the first");
        assert_eq!(reopened.segment_readers().len(), 2);

        let seg0 = &reopened.segment_readers()[0];
        let seg1 = &reopened.segment_readers()[1];
        assert_eq!(seg0.doc_base, 0);
        assert_eq!(seg1.doc_base, 2);

        // First (reused) segment: still reads its original sparse values.
        let meta0 = seg0.doc_values_meta().expect("segment 0 has doc values");
        let entry0 = meta0.numeric_entry(1).unwrap();
        let data0 = seg0.doc_values_data().unwrap();
        assert_eq!(
            doc_values::numeric_value(data0, entry0, 0).unwrap(),
            Some(5)
        );
        assert_eq!(doc_values::numeric_value(data0, entry0, 1).unwrap(), None);

        // Second (freshly opened via reopen) segment: its own, different
        // sparse pattern.
        let meta1 = seg1
            .doc_values_meta()
            .expect("newly reopened segment must have doc-values wired in too");
        let entry1 = meta1.numeric_entry(1).unwrap();
        let data1 = seg1.doc_values_data().unwrap();
        assert_eq!(doc_values::numeric_value(data1, entry1, 0).unwrap(), None);
        assert_eq!(
            doc_values::numeric_value(data1, entry1, 1).unwrap(),
            Some(42)
        );
        assert_eq!(doc_values::numeric_value(data1, entry1, 2).unwrap(), None);

        std::fs::remove_dir_all(&dir_path).ok();
    }
}

/// End-to-end exercise of `lucene_index::merge::merge_postings` (task:
/// "wire a real end-to-end caller for postings merge logic") -- not a
/// hand-built `MergeSource` fixture (see `merge.rs`'s own unit tests for
/// that), but the real path: `IndexWriter` flushes two real segments,
/// each with real `.tim`/`.tip`/`.tmd`/`.doc` postings
/// (`set_postings_field`), a configured `MergePolicyConfig` triggers a
/// real automatic merge inside `commit()` (`IndexWriter::auto_merge` ->
/// `execute_merge`, which now opens each source's postings and builds
/// `MergeSource::postings` from real on-disk data), and the merged
/// result is read back through this crate's own `DirectoryReader` ->
/// `SegmentReader` -> `search_term_query_multi_segment` stack -- the
/// exact same read path a caller with a fully general multi-segment
/// index already uses, run here against a single post-merge segment.
///
/// (Points (`merge_points`) is not exercised here: this port's
/// `IndexWriter` has no points write path at flush time at all yet --
/// no `set_points_field`-equivalent, no points-carrying `Document`
/// field shape -- so there is no real segment it could ever produce
/// with `.kdm`/`.kdi`/`.kdd` files to merge in the first place. See
/// `docs/parity.md` and `merge.rs`'s module doc comment.)
mod postings_merge_e2e {
    use super::*;
    use crate::multi_segment::search_term_query_multi_segment;
    use crate::query::TermQuery;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::merge_policy::MergePolicyConfig;
    use lucene_index::segment_info::LuceneVersion;

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

    fn body_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqs,
            ..stored_only_field("body", number)
        }
    }

    fn body_field_with_term_vectors(number: i32) -> FieldInfo {
        FieldInfo {
            store_term_vectors: true,
            ..body_field(number)
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

    fn tight_merge_policy() -> MergePolicyConfig {
        MergePolicyConfig {
            max_merge_at_once: 10,
            segments_per_tier: 2.0,
            ..MergePolicyConfig::default()
        }
    }

    /// Flushes two real segments (one doc each, via two separate
    /// `commit()` calls) with real postings, then a third `commit()`
    /// crosses `tight_merge_policy`'s `segments_per_tier` threshold and
    /// triggers a real automatic merge down to one segment -- confirms
    /// via a real `TermQuery` that every source doc's terms survived
    /// the merge, not just that the merged segment's raw bytes exist.
    #[test]
    fn term_query_finds_docs_from_both_flushed_segments_after_automatic_merge() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let fields = vec![stored_only_field("id", 0), body_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        writer
            .add_document(doc_with_body("a", "quick fox"))
            .unwrap();
        writer.commit().unwrap();
        writer.add_document(doc_with_body("b", "lazy fox")).unwrap();
        writer.commit().unwrap();
        // Crosses segments_per_tier (2) with a third one-doc commit,
        // triggering `auto_merge` to fold all three segments into one.
        writer
            .add_document(doc_with_body("c", "quick dog"))
            .unwrap();
        writer.commit().unwrap();

        let segments_after = writer.segment_infos().segments.clone();
        assert_eq!(
            segments_after.len(),
            1,
            "postings-carrying segments must merge down like any other"
        );
        drop(segments_after);

        // Opened through the real `DirectoryReader` -> `SegmentReader`
        // stack: `SegmentReader::open`'s `.tim`-suffix derivation
        // special-cases this port's own no-codec-suffix convention
        // (segment_suffix == "", matching every write site in
        // `index_writer.rs`/`merge.rs`) the same way its `.dvm`-suffix
        // derivation does.
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms = [None];

        // "fox" was indexed by docs "a" and "b" -- originally in two
        // different flushed segments, now merged into one.
        let fox_hits =
            search_term_query_multi_segment(&segments, &TermQuery::new("body", "fox"), &norms, 10)
                .unwrap();
        assert_eq!(
            fox_hits.len(),
            2,
            "\"fox\" must resolve to both merged docs"
        );

        // "quick" was indexed by docs "a" and "c".
        let quick_hits = search_term_query_multi_segment(
            &segments,
            &TermQuery::new("body", "quick"),
            &norms,
            10,
        )
        .unwrap();
        assert_eq!(
            quick_hits.len(),
            2,
            "\"quick\" must resolve to both merged docs"
        );

        // "lazy" was indexed by doc "b" only.
        let lazy_hits =
            search_term_query_multi_segment(&segments, &TermQuery::new("body", "lazy"), &norms, 10)
                .unwrap();
        assert_eq!(lazy_hits.len(), 1);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Same shape as
    /// `term_query_finds_docs_from_both_flushed_segments_after_automatic_merge`,
    /// but with the "body" field also opted into term vectors -- proves
    /// postings and term vectors merge correctly together through the
    /// same automatic-merge path, and that the merged segment's term
    /// vectors (verified through `lucene_codecs::term_vectors::open`
    /// directly, since term vectors have no query/search surface to
    /// exercise via `DirectoryReader`) still hold every source doc's own
    /// content after the merge.
    #[test]
    fn term_vectors_survive_automatic_merge_alongside_postings() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let fields = vec![stored_only_field("id", 0), body_field_with_term_vectors(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();
        writer.set_term_vector_field(Some("body")).unwrap();
        writer.set_merge_policy(Some(tight_merge_policy()));

        writer
            .add_document(doc_with_body("a", "quick fox"))
            .unwrap();
        writer.commit().unwrap();
        writer.add_document(doc_with_body("b", "lazy fox")).unwrap();
        writer.commit().unwrap();
        // Crosses segments_per_tier (2) with a third one-doc commit,
        // triggering a real automatic merge of all three segments.
        writer
            .add_document(doc_with_body("c", "quick dog"))
            .unwrap();
        writer.commit().unwrap();

        let segments_after = writer.segment_infos().segments.clone();
        assert_eq!(
            segments_after.len(),
            1,
            "postings+term-vector-carrying segments must merge down like any other"
        );
        let sci = &segments_after[0];

        // Postings still searchable through the real DirectoryReader.
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms = [None];
        let fox_hits =
            search_term_query_multi_segment(&segments, &TermQuery::new("body", "fox"), &norms, 10)
                .unwrap();
        assert_eq!(
            fox_hits.len(),
            2,
            "\"fox\" must resolve to both merged docs"
        );

        // Term vectors: every source doc's own distinct terms must
        // survive the merge, checked as a set since automatic merging
        // doesn't guarantee source-segment order in the merged segment.
        let tvd = dir.open(&format!("{}.tvd", sci.segment_name)).unwrap();
        let tvx = dir.open(&format!("{}.tvx", sci.segment_name)).unwrap();
        let tvm = dir.open(&format!("{}.tvm", sci.segment_name)).unwrap();
        let tv_reader =
            lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
        assert_eq!(tv_reader.max_doc(), 3);

        let mut actual_docs: Vec<Vec<String>> = (0..3)
            .map(|doc_id| {
                let doc = tv_reader.document(doc_id).unwrap().unwrap();
                assert_eq!(doc.fields.len(), 1);
                let mut terms: Vec<String> = doc.fields[0]
                    .terms
                    .iter()
                    .map(|t| String::from_utf8(t.term.clone()).unwrap())
                    .collect();
                terms.sort();
                terms
            })
            .collect();
        actual_docs.sort();

        let mut expected_docs = vec![
            vec!["fox".to_string(), "quick".to_string()],
            vec!["fox".to_string(), "lazy".to_string()],
            vec!["dog".to_string(), "quick".to_string()],
        ];
        expected_docs.sort();

        assert_eq!(
            actual_docs, expected_docs,
            "every source doc's own term-vector content must survive the merge intact"
        );

        std::fs::remove_dir_all(&dir_path).ok();
    }
}

/// Real end-to-end proof that positions survive
/// `IndexWriter::add_document` (real, automatically-tokenized text, no
/// hand-built `TermPostings`) -> `invert_documents` -> `commit` -> a real
/// `PhraseQuery` search against the resulting on-disk segment. The other
/// `PhraseQuery` tests read hand-built or Java-written postings.
mod phrase_query_e2e {
    use super::*;
    use crate::{search_phrase_query, PhraseQuery, VecCollector};
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

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

    fn positions_field(number: i32) -> FieldInfo {
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositions,
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

    /// Flushes one real segment via `add_document`/`commit` with the
    /// "body" field opted into `DocsAndFreqsAndPositions` postings, then
    /// runs a real `PhraseQuery` against it through this crate's own
    /// `DirectoryReader` -> `search_phrase_query` path: "quick fox" must
    /// find doc "a" (an adjacent pair) but not doc "b" (same two terms,
    /// but not adjacent), and a non-adjacent pair like "fox lazy" must
    /// find nothing at all.
    #[test]
    fn phrase_query_finds_adjacent_terms_from_a_real_add_document_commit() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let fields = vec![stored_only_field("id", 0), positions_field(1)];
        let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
        writer.set_postings_field(Some("body")).unwrap();

        writer
            .add_document(doc_with_body("a", "the quick fox jumps"))
            .unwrap();
        writer
            .add_document(doc_with_body("b", "the fox is quick"))
            .unwrap();
        writer.commit().unwrap();

        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let seg = &segments[0];

        // "quick fox" is adjacent only in doc "a" (doc ID 0).
        let mut collector = VecCollector::default();
        search_phrase_query(
            seg.fields,
            seg.doc_in,
            seg.pos_in,
            seg.pay_in,
            seg.live_docs,
            &PhraseQuery::new("body", ["quick", "fox"]),
            &mut collector,
        )
        .unwrap();
        assert_eq!(
            collector.docs,
            vec![0],
            "\"quick fox\" must match only the doc where they're adjacent"
        );

        // Same two terms, non-adjacent in every doc -- must match
        // nothing.
        let mut collector = VecCollector::default();
        search_phrase_query(
            seg.fields,
            seg.doc_in,
            seg.pos_in,
            seg.pay_in,
            seg.live_docs,
            &PhraseQuery::new("body", ["fox", "quick"]),
            &mut collector,
        )
        .unwrap();
        assert!(
            collector.docs.is_empty(),
            "\"fox quick\" (reversed order) must not match either doc"
        );

        std::fs::remove_dir_all(&dir_path).ok();
    }
}
