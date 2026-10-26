use super::*;
use crate::buffered_updates::Term;
use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
use crate::segment_info::{IndexSortField, LuceneVersion};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn fields(rank_type: DocValuesType) -> Vec<FieldInfo> {
    vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("rank", 1)
            .with_omit_norms(true)
            .with_doc_values(rank_type, DocValuesSkipIndexType::None, -1),
    ]
}

fn writer(dir: &FsDirectory, rank_type: DocValuesType) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(
        dir,
        fields(rank_type),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_doc_values_field(Some("rank")).unwrap();
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn doc(id: &str, rank: i64) -> Document {
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
        ],
    }
}

/// Every live document's id, over the latest commit.
fn committed_ids(dir: &FsDirectory) -> Vec<String> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut out = Vec::new();
    for sci in &infos.segments {
        let name = &sci.segment_name;
        let si = segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
            .unwrap();
        let read = |ext: &str| dir.open(&format!("{name}.{ext}")).unwrap();
        let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
        let reader =
            lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let live = (sci.del_gen >= 0).then(|| {
            lucene_codecs::live_docs::parse(
                &dir.open(&crate::deletes::liv_file_name(name, sci.del_gen))
                    .unwrap(),
                &sci.segment_id,
                sci.del_gen,
                si.doc_count as usize,
                sci.del_count as usize,
            )
            .unwrap()
        });
        for d in 0..si.doc_count {
            if live.as_ref().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                out.push(s.clone());
            }
        }
    }
    out.sort();
    out
}

fn assert_clean(dir: &FsDirectory) {
    for r in crate::check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

/// A source index with two segments, one of which has a deletion.
fn source(tmp: &TempDir) -> FsDirectory {
    let dir = FsDirectory::open(tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    for (id, rank) in [("s0", 5), ("s1", 1), ("s2", 3)] {
        w.add_document(doc(id, rank)).unwrap();
    }
    w.commit().unwrap();
    for (id, rank) in [("s3", 2), ("s4", 4)] {
        w.add_document(doc(id, rank)).unwrap();
    }
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "s1")])
        .unwrap();
    w.commit().unwrap();
    drop(w);
    dir
}

/// `addIndexes(Directory...)`: the source's segments arrive under new
/// names, deletions and all; nothing is durable until the commit; the
/// result passes `CheckIndex`, and a delete by term reaches the added
/// documents.
#[test]
fn add_indexes_copies_segments_as_they_are() {
    let src_tmp = TempDir::new("add-indexes-src");
    let src = source(&src_tmp);
    let tmp = TempDir::new("add-indexes-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    w.add_document(doc("own", 0)).unwrap();
    w.commit().unwrap();
    w.add_indexes(&[&src]).unwrap();
    assert_eq!(
        committed_ids(&dir),
        ["own"],
        "not durable before the commit"
    );
    let names: Vec<&str> = w
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.as_str())
        .collect();
    assert_eq!(names, ["_0", "_1", "_2"]);
    assert_eq!(w.segment_infos().segments[1].del_count, 1);
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3", "s4"]);
    assert_clean(&dir);
    w.delete_documents_by_term(&[Term::new("id", "s4")])
        .unwrap();
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3"]);
    // The source is untouched and can be added again.
    w.add_indexes(&[&src]).unwrap();
    w.rollback();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3"]);
}

/// `addIndexes(CodecReader...)`: one new segment of the sources' live
/// documents, in the index sort.
#[test]
fn add_indexes_merged_makes_one_segment() {
    let src_tmp = TempDir::new("add-merged-src");
    let src = source(&src_tmp);
    let src2_tmp = TempDir::new("add-merged-src2");
    let src2 = source(&src2_tmp);
    let tmp = TempDir::new("add-merged-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    w.add_indexes_merged(&[]).unwrap();
    // One directory named twice: its second `write.lock` fails (Java's
    // `noDupDirs` throws first); nothing is added.
    assert!(matches!(
        w.add_indexes(&[&src, &src]),
        Err(Error::Store(lucene_store::Error::LockObtainFailed(_)))
    ));
    assert!(w.segment_infos().segments.is_empty());
    w.add_indexes_merged(&[&src, &src2]).unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    w.commit().unwrap();
    let ids = committed_ids(&dir);
    assert_eq!(ids.len(), 8);
    assert_eq!(
        w.segment_infos().segments[0].del_count,
        0,
        "deletions dropped"
    );
    // `IndexWriter.addIndexes(CodecReader...)`'s diagnostics: no merge details.
    let sci = &w.segment_infos().segments[0];
    let si = crate::segment_info::parse(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
    )
    .unwrap();
    assert_eq!(si.diagnostics[0].1, "addIndexes(CodecReader...)");
    assert!(si.diagnostics.iter().all(|(k, _)| !k.starts_with("merge")));
    assert_clean(&dir);
}

/// `addIndexes(CodecReader...)` over readers hiding documents: only the
/// kept, still-live documents arrive; a deleted one stays deleted whatever
/// the filter says; rejecting everything adds nothing.
#[test]
fn add_indexes_filtered_keeps_only_accepted_documents() {
    let src_tmp = TempDir::new("add-filtered-src");
    let src = source(&src_tmp);
    let tmp = TempDir::new("add-filtered-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    w.add_indexes_filtered(&[&src], &|_, _, _| false).unwrap();
    assert!(w.segment_infos().segments.is_empty());
    // Accept the whole second segment and the first two documents of the first (the first, s1, deleted in the source, is live again).
    w.add_indexes_filtered(&[&src], &|_, segment, doc| segment == 1 || doc <= 1)
        .unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["s1", "s2", "s3", "s4"]);
    assert_clean(&dir);
}

/// What `addIndexes` refuses: a source whose sort is not congruent with the
/// writer's, and a field whose schema conflicts; either way nothing is
/// added.
#[test]
fn add_indexes_refuses_incompatible_sources() {
    let src_tmp = TempDir::new("add-bad-src");
    let src = source(&src_tmp);

    let tmp = TempDir::new("add-bad-sort");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", true, None)]))
        .unwrap();
    assert!(matches!(
        w.add_indexes(&[&src]),
        Err(Error::IncongruentIndexSort { .. })
    ));
    assert!(w.segment_infos().segments.is_empty());

    let tmp = TempDir::new("add-bad-schema");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::SortedNumeric);
    let err = w.add_indexes(&[&src]).unwrap_err();
    assert!(
        matches!(&err, Error::AddIndexes(m) if m.contains("\"rank\"") && m.contains("doc values type")),
        "{err}"
    );
    assert!(w.segment_infos().segments.is_empty());

    // A source with a writer still open on it is locked.
    let tmp = TempDir::new("add-locked");
    let busy = FsDirectory::open(&tmp);
    let mut open = writer(&busy, DocValuesType::Numeric);
    let tmp2 = TempDir::new("add-locked-dst");
    let dir2 = FsDirectory::open(&tmp2);
    let mut w = writer(&dir2, DocValuesType::Numeric);
    assert!(matches!(
        w.add_indexes(&[&busy]),
        Err(Error::Store(lucene_store::Error::LockObtainFailed(_)))
    ));
    // `addIndexes(CodecReader...)` reads readers and locks nothing: the open
    // writer's committed documents arrive, and it keeps its lock.
    open.add_document(doc("busy", 1)).unwrap();
    open.commit().unwrap();
    w.add_indexes_merged(&[&busy]).unwrap();
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir2), vec!["busy".to_string()]);
    open.add_document(doc("still", 2)).unwrap();
    open.commit().unwrap();
    drop(open);
    assert_eq!(rename("_3.fdt", "_3", "_9"), "_9.fdt");
    assert_eq!(rename("_3_1.liv", "_3", "_9"), "_9_1.liv");
    assert_eq!(rename("_30.fdt", "_3", "_9"), "_30.fdt");
}

/// `FieldNumbers.verifySameSchema`, which `addIndexes` runs on every
/// incoming field (`addOrGet`/`verifyFieldInfo`): any difference is refused
/// with Java's message, a side that sets no doc values, index options,
/// points or vectors included; norms and term vectors count only for an
/// indexed field.
#[test]
fn schema_conflicts_are_the_ones_java_refuses() {
    use super::super::explicit::schema_change;
    use lucene_codecs::field_infos::{VectorEncoding, VectorSimilarityFunction};
    let base = FieldInfo::new("f", 0);
    assert_eq!(schema_change(&base, &base), None);
    let says = |ours: &FieldInfo, theirs: &FieldInfo| schema_change(ours, theirs).unwrap();
    let dv = |t| {
        base.clone()
            .with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    assert_eq!(
        says(&dv(DocValuesType::Numeric), &dv(DocValuesType::Sorted)),
        "cannot change field \"f\" from doc values type=NUMERIC to inconsistent doc values type=SORTED"
    );
    assert_eq!(
        says(&base, &dv(DocValuesType::Sorted)),
        "cannot change field \"f\" from doc values type=NONE to inconsistent doc values type=SORTED"
    );
    let skip =
        base.clone()
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::Range, -1);
    assert_eq!(
        says(&dv(DocValuesType::Numeric), &skip),
        "cannot change field \"f\" from docValuesSkipIndexType=NONE to inconsistent docValuesSkipIndexType=RANGE"
    );
    let io = |o| base.clone().with_index_options(o);
    assert_eq!(
        says(&io(IndexOptions::Docs), &io(IndexOptions::DocsAndFreqs)),
        "cannot change field \"f\" from index options=DOCS to inconsistent index options=DOCS_AND_FREQS"
    );
    assert_eq!(
        says(&io(IndexOptions::None), &io(IndexOptions::Docs)),
        "cannot change field \"f\" from index options=NONE to inconsistent index options=DOCS"
    );
    let mut tv = io(IndexOptions::Docs);
    tv.store_term_vectors = true;
    assert_eq!(
        says(&io(IndexOptions::Docs), &tv),
        "cannot change field \"f\" from storeTermVector=false to inconsistent storeTermVector=true"
    );
    let mut norms = io(IndexOptions::Docs);
    norms.omit_norms = true;
    assert_eq!(
        says(&io(IndexOptions::Docs), &norms),
        "cannot change field \"f\" from omitNorms=false to inconsistent omitNorms=true"
    );
    // Not indexed: norms and term vectors are not the field's.
    let mut unindexed = base.clone();
    unindexed.omit_norms = true;
    assert_eq!(schema_change(&base, &unindexed), None);
    let points = base.clone().with_points(1, 1, 8);
    assert_eq!(
        says(&base, &points),
        "cannot change field \"f\" from points dimensionCount=0, indexDimensionCount=0, numBytes=0 to inconsistent dimensionCount=1, indexDimensionCount=1, numBytes=8"
    );
    assert_eq!(schema_change(&points, &points), None);
    assert!(
        schema_change(&points, &base).is_some(),
        "no incoming points"
    );
    let vectors = |d, sim| base.clone().with_vectors(d, VectorEncoding::Float32, sim);
    assert_eq!(
        says(
            &vectors(4, VectorSimilarityFunction::DotProduct),
            &vectors(6, VectorSimilarityFunction::DotProduct)
        ),
        "cannot change field \"f\" from vector dimension=4, vector encoding=FLOAT32, vector similarity function=DOT_PRODUCT to inconsistent vector dimension=6, vector encoding=FLOAT32, vector similarity function=DOT_PRODUCT"
    );
    let cosine = vectors(4, VectorSimilarityFunction::Cosine);
    assert_eq!(schema_change(&cosine, &cosine), None);
    assert!(
        says(&base, &base.clone().with_soft_deletes_field(true)).contains("soft-deletes"),
        "soft-deletes role"
    );
}

/// `addIndexes` refuses to run between `prepareCommit` and its finish, and
/// past `maxDocs`; a source whose segment carries doc-values updates is
/// copied with its update generation, and a field this writer does not know
/// comes along as it is.
#[test]
fn add_indexes_limits_and_generations() {
    let src_tmp = TempDir::new("add-limits-src");
    let src = source(&src_tmp);

    let tmp = TempDir::new("add-limits");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.add_document(doc("own", 1)).unwrap();
    w.prepare_commit().unwrap();
    assert!(matches!(
        w.add_indexes(&[&src]),
        Err(Error::PreparedCommitPending("add_indexes"))
    ));
    w.finish_commit().unwrap();
    w.set_max_docs(3);
    assert!(matches!(w.add_indexes(&[&src]), Err(Error::TooManyDocs(3))));
    assert_eq!(w.segment_infos().segments.len(), 1);

    // A source with a doc-values update on a committed segment.
    let upd_tmp = TempDir::new("add-limits-upd");
    let upd = FsDirectory::open(&upd_tmp);
    let mut u = writer(&upd, DocValuesType::Numeric);
    u.add_document(doc("u0", 1)).unwrap();
    u.add_document(doc("u1", 2)).unwrap();
    u.commit().unwrap();
    u.update_numeric_doc_value(Term::new("id", "u1"), "rank", 7)
        .unwrap();
    u.commit().unwrap();
    drop(u);
    assert!(!segment_infos::read_latest(&upd).unwrap().segments[0]
        .dv_update_files
        .is_empty());

    // Into a writer that knows only `id`.
    let only_tmp = TempDir::new("add-limits-only-id");
    let only = FsDirectory::open(&only_tmp);
    let mut o = IndexWriter::open(
        &only,
        vec![FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)],
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    o.add_indexes(&[&upd]).unwrap();
    o.commit().unwrap();
    let added = &segment_infos::read_latest(&only).unwrap().segments[0];
    assert!(!added.dv_update_files.is_empty());
    assert!(added
        .dv_update_files
        .iter()
        .flat_map(|(_, f)| f)
        .all(|f| f.starts_with(&format!("{}_", added.segment_name))));
    assert_eq!(committed_ids(&only), ["u0", "u1"]);
    assert_clean(&only);
}

/// A compound source segment's field infos are read through its `.cfs`:
/// it is added (and checked against this writer's schema) like any other.
#[test]
fn add_indexes_reads_a_compound_source() {
    let src_tmp = TempDir::new("add-cfs-src");
    let src = FsDirectory::open(&src_tmp);
    let mut w = writer(&src, DocValuesType::Numeric);
    w.set_use_compound_file(true);
    w.add_document(doc("c0", 1)).unwrap();
    w.commit().unwrap();
    drop(w);
    assert!(src.list_all().unwrap().iter().any(|f| f.ends_with(".cfs")));

    let tmp = TempDir::new("add-cfs-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.add_indexes(&[&src]).unwrap();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    assert_clean(&dir);
    drop(w);

    let bad_tmp = TempDir::new("add-cfs-bad");
    let bad = FsDirectory::open(&bad_tmp);
    let mut w = writer(&bad, DocValuesType::Sorted);
    let err = w.add_indexes(&[&src]).unwrap_err();
    assert!(
        matches!(err, Error::AddIndexes(ref m) if m.contains("doc values type")),
        "{err:?}"
    );
}

/// `addIndexes(CodecReader...)` over remapped SORTED_NUMERIC values
/// (lucene-facet's `OrdinalMappingLeafReader`): a value outside the map
/// fails the merge, and then nothing of the source is added -- not even the
/// raw copies, whose values were never rewritten.
#[test]
fn a_failed_remapping_merge_adds_nothing() {
    let src_tmp = TempDir::new("add-remap-src");
    let src = FsDirectory::open(&src_tmp);
    {
        let mut w = writer(&src, DocValuesType::SortedNumeric);
        for (id, rank) in [("s0", 5), ("s1", 1), ("s2", 3)] {
            w.add_document(doc(id, rank)).unwrap();
        }
        w.commit().unwrap();
    }
    let tmp = TempDir::new("add-remap-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::SortedNumeric);
    w.add_document(doc("own", 0)).unwrap();
    w.commit().unwrap();
    let short = crate::merge::SortedNumericRemap {
        fields: vec!["rank".into()],
        map: vec![0; 3],
    };
    match w.add_indexes_merged_remapping(&[&src], short) {
        Err(Error::Merge(crate::merge::Error::InvalidRemap(m))) => {
            assert_eq!(m, "value 5 is outside the map of 3 ordinals")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(w.segment_infos().segments.len(), 1, "only the own segment");
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own"]);
    assert_clean(&dir);
    // A map that covers every value: one merged segment of the source.
    let full = crate::merge::SortedNumericRemap {
        fields: vec!["rank".into()],
        map: (0..6).rev().collect(),
    };
    w.add_indexes_merged_remapping(&[&src], full).unwrap();
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s1", "s2"]);
    assert_eq!(w.segment_infos().segments.len(), 2);
    assert_clean(&dir);
}

/// A remap of an index-sort field is refused: the merge would order the
/// documents by the values before the remap rewrote them.
#[test]
fn a_remap_of_the_index_sort_is_refused() {
    let src_tmp = TempDir::new("add-remap-sort-src");
    let src = FsDirectory::open(&src_tmp);
    let tmp = TempDir::new("add-remap-sort-dst");
    let dir = FsDirectory::open(&tmp);
    // The check reads only the sort's field names.
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    let remap = crate::merge::SortedNumericRemap {
        fields: vec!["rank".into()],
        map: vec![0],
    };
    match w.add_indexes_merged_remapping(&[&src], remap) {
        Err(Error::AddIndexes(m)) => {
            assert_eq!(m, "cannot remap \"rank\", a field of the index sort")
        }
        other => panic!("{other:?}"),
    }
}

/// Imports update the writer-lifetime number reservations before a new
/// explicit field can consume an imported name's number. Source removal must
/// not discard a reservation for a name this writer has not registered yet.
#[test]
fn explicit_field_numbers_reserve_imported_names_for_copied_and_merged_sources() {
    use crate::index_writer::explicit::{
        ExplicitDocument, ExplicitFields, InvertedField, InvertedTerm,
    };
    use lucene_store::ByteBuffersDirectory;
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let source = ByteBuffersDirectory::new();
    {
        let mut writer = IndexWriter::open(
            &source,
            vec![
                FieldInfo::new("imported", 0).with_index_options(IndexOptions::DocsAndFreqs),
                FieldInfo::new("unseen", 2),
            ],
            "Lucene104",
            version,
        )
        .unwrap();
        writer.enable_explicit_documents().unwrap();
        writer
            .add_explicit_documents(vec![ExplicitDocument {
                stored: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("imported".into()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("unseen".into()),
                    },
                ],
                fields: ExplicitFields {
                    inverted: vec![InvertedField {
                        field_number: 0,
                        terms: vec![InvertedTerm {
                            term: b"drop".to_vec(),
                            freq: 1,
                            positions: Vec::new(),
                            offsets: Vec::new(),
                            payloads: Vec::new(),
                        }],
                        norm: Some(1),
                    }],
                    ..ExplicitFields::default()
                },
            }])
            .unwrap();
        writer.commit().unwrap();
    }
    let changed_source = ByteBuffersDirectory::new();
    {
        let mut writer = IndexWriter::open(
            &changed_source,
            vec![FieldInfo::new("imported", 0)],
            "Lucene104",
            version,
        )
        .unwrap();
        writer
            .add_document(Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("changed".into()),
                }],
            })
            .unwrap();
        writer.commit().unwrap();
    }
    for merged in [false, true] {
        let destination = ByteBuffersDirectory::new();
        let mut writer = IndexWriter::open(&destination, Vec::new(), "Lucene104", version).unwrap();
        writer.enable_explicit_documents().unwrap();
        assert_eq!(
            writer.register_field(FieldInfo::new("local", 9)).unwrap(),
            0
        );
        if merged {
            writer.add_indexes_merged(&[&source]).unwrap();
        } else {
            writer.add_indexes(&[&source]).unwrap();
        }
        assert!(writer.registered_field("imported").is_none());
        assert!(writer.registered_field("unseen").is_none());
        writer
            .delete_documents_by_term(&[Term::new("imported", "drop")])
            .unwrap();
        writer.commit().unwrap();
        assert!(writer.segment_infos.segments.is_empty());
        for changed in [
            FieldInfo::new("imported", 9),
            FieldInfo::new("imported", 9)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_omit_norms(true),
        ] {
            assert!(
                matches!(writer.register_field(changed), Err(Error::Explicit(_))),
                "the removed source's index options and norms remain fixed"
            );
        }
        let changed_import = if merged {
            writer.add_indexes_merged(&[&changed_source])
        } else {
            writer.add_indexes(&[&changed_source])
        };
        assert!(
            matches!(changed_import, Err(Error::AddIndexes(_))),
            "imports also verify the retained source schema"
        );
        assert_eq!(writer.register_field(FieldInfo::new("late", 9)).unwrap(), 3);
        assert_eq!(
            writer
                .register_field(
                    FieldInfo::new("imported", 9).with_index_options(IndexOptions::DocsAndFreqs)
                )
                .unwrap(),
            1
        );
        assert_eq!(
            writer.register_field(FieldInfo::new("unseen", 9)).unwrap(),
            2
        );
        assert_eq!(writer.register_field(FieldInfo::new("next", 9)).unwrap(), 4);
    }
}

#[test]
fn delete_all_releases_unregistered_source_properties_and_preserves_configured_roles() {
    let source_tmp = TempDir::new("explicit-delete-all-source-properties");
    let source = source(&source_tmp);
    let destination_tmp = TempDir::new("explicit-delete-all-configured-properties");
    let destination = FsDirectory::open(&destination_tmp);
    let numeric = |name, number| {
        FieldInfo::new(name, number).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        )
    };
    let parent = numeric("parent", 2).with_parent_field(true);
    let soft = numeric("soft", 5).with_soft_deletes_field(true);
    let mut writer = IndexWriter::open(
        &destination,
        vec![parent.clone(), soft.clone()],
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.set_parent_field(Some("parent")).unwrap();
    writer.add_indexes(&[&source]).unwrap();
    assert!(
        writer.register_field(FieldInfo::new("id", 99)).is_err(),
        "the imported indexed schema remains fixed before deleteAll"
    );
    writer.delete_all().unwrap();
    writer.commit().unwrap();
    // Source-only fields lose their numbers and properties, as Java clears
    // FieldNumbers. The facade's configured schemas and IDs remain intact.
    assert_eq!(writer.register_field(FieldInfo::new("id", 99)).unwrap(), 0);
    assert_eq!(
        writer.register_field(FieldInfo::new("rank", 99)).unwrap(),
        1
    );
    assert_eq!(writer.register_field(parent).unwrap(), 2);
    assert_eq!(writer.register_field(soft).unwrap(), 5);
    assert_eq!(writer.parent_field(), Some("parent"));
    assert!(writer.registered_field("parent").unwrap().parent_field);
    assert!(writer.registered_field("soft").unwrap().soft_deletes_field);
}
