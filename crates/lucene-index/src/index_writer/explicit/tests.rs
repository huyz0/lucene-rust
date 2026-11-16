#![cfg(test)]
// Test fixtures' own arithmetic, not values read off disk -- see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::index_file_deleter::DeletionPolicy;
use crate::index_writer::SoftDeletesRetention;
use crate::index_writer::POSTINGS_FORMAT_NAME;
use crate::segment_info::LuceneVersion;
use lucene_store::directory::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// The field shapes an OpenSearch document is made of.
struct Fields {
    id: i32,
    source: i32,
    body: i32,
    tag: i32,
    num: i32,
    soft: i32,
    seq: i32,
}

fn register(w: &mut IndexWriter<'_>) -> Fields {
    w.enable_explicit_documents().unwrap();
    let reg = |w: &mut IndexWriter<'_>, f: FieldInfo| w.register_field(f).unwrap();
    Fields {
        id: reg(
            w,
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        ),
        source: reg(w, FieldInfo::new("_source", 0)),
        body: reg(
            w,
            FieldInfo::new("body", 0)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
        ),
        tag: reg(
            w,
            FieldInfo::new("tag", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1),
        ),
        num: reg(
            w,
            FieldInfo::new("num", 0)
                .with_points(1, 1, 8)
                .with_doc_values(
                    DocValuesType::SortedNumeric,
                    DocValuesSkipIndexType::None,
                    -1,
                ),
        ),
        soft: reg(
            w,
            FieldInfo::new("__soft_deletes", 0)
                .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                .with_soft_deletes_field(true),
        ),
        seq: reg(
            w,
            FieldInfo::new("_seq_no", 0).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
        ),
    }
}

fn term(t: &str, positions: &[i32]) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: positions.len() as i32,
        positions: positions.to_vec(),
        offsets: positions.iter().map(|&p| (p * 6, p * 6 + 5)).collect(),
        payloads: Vec::new(),
    }
}

/// Document `i`: `_id` "d{i}" (stored and indexed), a stored `_source`,
/// `body` with two or three terms, `tag` "t{i%3}", `num` = i. Every third
/// document has no `num`, so its columns are sparse.
fn doc(f: &Fields, i: i64) -> ExplicitDocument {
    let id = format!("d{i}");
    let mut body = vec![term("alpha", &[0]), term("beta", &[1, 3])];
    if i % 2 == 0 {
        body.push(term("gamma", &[2]));
    }
    let length = body.iter().map(|t| t.freq).sum::<i32>() as i64;
    let tag = format!("t{}", i % 3).into_bytes();
    let mut fields = ExplicitFields {
        inverted: vec![
            InvertedField {
                field_number: f.id,
                terms: vec![term_docs(&id)].into(),
                norm: None,
            },
            InvertedField {
                field_number: f.body,
                terms: body.into(),
                norm: Some(length),
            },
            InvertedField {
                field_number: f.tag,
                terms: vec![term_docs(std::str::from_utf8(&tag).unwrap())].into(),
                norm: None,
            },
        ],
        doc_values: vec![
            StoredField {
                field_number: f.tag,
                value: FieldValue::Binary(tag),
            },
            StoredField {
                field_number: f.seq,
                value: FieldValue::Long(i),
            },
        ],
        points: Vec::new(),
        registered: Vec::new(),
        term_vectors: Vec::new(),
    };
    if i % 3 != 0 {
        fields.doc_values.push(StoredField {
            field_number: f.num,
            value: FieldValue::Long(i),
        });
        fields.points.push(StoredField {
            field_number: f.num,
            value: FieldValue::Binary(((i as u64) ^ (1 << 63)).to_be_bytes().to_vec()),
        });
    }
    ExplicitDocument {
        stored: vec![
            StoredField {
                field_number: f.id,
                value: FieldValue::String(id),
            },
            StoredField {
                field_number: f.source,
                value: FieldValue::Binary(format!("{{\"n\":{i}}}").into_bytes()),
            },
        ],
        fields,
    }
}

fn term_docs(t: &str) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

fn check(dir: &FsDirectory) {
    let results = crate::check_index::check_directory(dir).unwrap();
    for r in &results {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

fn segment_fields(dir: &FsDirectory, sci: &SegmentCommitInfo) -> Vec<FieldInfo> {
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "")
        .unwrap()
        .fields
}

/// Explicit documents flush, merge and commit into segments the port's
/// `CheckIndex` accepts, each listing only the fields it carries.
/// `DocumentsWriterPerThread.updateDocuments` validates each document of
/// a block as it goes, and on a failure marks the block's earlier
/// documents deleted; this port validates the whole block first, so a
/// bad document `k` leaves nothing buffered and the block flag as it
/// was -- the same documents survive either way.
#[test]
fn a_block_with_an_invalid_document_buffers_nothing() {
    let tmp = TempDir::new("explicit-bad-block");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
    let (docs, explicit, ram) = (
        w.pending_docs.len(),
        w.pending_explicit.len(),
        w.ram_bytes_used(),
    );
    for k in 0..3 {
        let mut block: Vec<ExplicitDocument> = (1..4).map(|i| doc(&f, i)).collect();
        // Document `k` names an unregistered field.
        block[k].stored.push(StoredField {
            field_number: 99,
            value: FieldValue::Long(1),
        });
        assert!(w.add_explicit_documents(block).is_err(), "doc {k}");
        assert_eq!(w.pending_docs.len(), docs);
        assert_eq!(w.pending_explicit.len(), explicit);
        assert_eq!(w.ram_bytes_used(), ram);
        assert!(!w.pending_has_blocks, "doc {k} set the block flag");
    }
    w.commit().unwrap();
    let infos = w.segment_infos().clone();
    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].del_count, 0);
}

#[test]
fn explicit_documents_round_trip_through_check_index() {
    let tmp = TempDir::new("explicit-round-trip");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.set_max_buffered_docs(7).unwrap();
    let docs: Vec<ExplicitDocument> = (0..30).map(|i| doc(&f, i)).collect();
    for d in docs {
        w.add_explicit_documents(vec![d]).unwrap();
    }
    let infos = w.commit().unwrap().clone();
    assert!(infos.segments.len() >= 4, "several flushes");
    check(&dir);

    let fields = segment_fields(&dir, &infos.segments[0]);
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["_id", "_source", "body", "tag", "num", "_seq_no"],
        "no soft-deletes field yet"
    );
    let attr = |name: &str, key: &str| {
        fields
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .attributes
            .iter()
            .any(|(k, _)| k == key)
    };
    assert!(attr("body", "PerFieldPostingsFormat.format"));
    assert!(!attr("_source", "PerFieldPostingsFormat.format"));
    assert!(attr("tag", "PerFieldDocValuesFormat.format"));
    assert!(!attr("body", "PerFieldDocValuesFormat.format"));

    // A soft update reaches a segment that lacks the soft-deletes field in
    // its `.fnm`, and a merge keeps it all readable.
    let one = DocValuesUpdate::Numeric {
        term: Term {
            field: "_id".to_string(),
            bytes: b"d3".to_vec(),
        },
        field: "__soft_deletes".to_string(),
        value: Some(1),
    };
    w.soft_update_explicit_documents(
        Term {
            field: "_id".to_string(),
            bytes: b"d3".to_vec(),
        },
        vec![doc(&f, 3)],
        &[one],
    )
    .unwrap();
    let infos = w.commit().unwrap().clone();
    check(&dir);
    assert_eq!(f.soft, 5);
    // `softDelCount`: the old d3, and nothing else yet.
    let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
    assert_eq!(soft, 1);
    assert_eq!(
        infos
            .segments
            .iter()
            .filter(|s| s.soft_del_count == 1)
            .count(),
        1
    );

    // A tombstone born soft-deleted counts in its own segment.
    let mut tombstone = doc(&f, 100);
    tombstone.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    w.add_explicit_documents(vec![tombstone]).unwrap();
    let infos = w.commit().unwrap().clone();
    check(&dir);
    let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
    assert_eq!(soft, 2);
}

#[test]
fn registration_is_global_and_schema_is_fixed() {
    let tmp = TempDir::new("explicit-register");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let text = FieldInfo::new("t", 0).with_index_options(IndexOptions::DocsAndFreqs);
    assert!(
        w.register_field(text.clone()).is_err(),
        "needs explicit mode"
    );
    w.enable_explicit_documents().unwrap();
    w.enable_explicit_documents().unwrap();
    assert_eq!(w.register_field(text.clone()).unwrap(), 0);
    assert_eq!(w.register_field(FieldInfo::new("u", 7)).unwrap(), 1);
    assert_eq!(
        w.register_field(text.clone()).unwrap(),
        0,
        "same schema, same number"
    );
    let changed = text.clone().with_omit_norms(true);
    assert!(w.register_field(changed).is_err());
    let vectors = FieldInfo::new("v", 0).with_store_term_vectors(true);
    assert!(
        w.register_field(vectors).is_err(),
        "vectors on an unindexed field"
    );
    let vectors = text.clone().with_store_term_vectors(true);
    assert!(
        w.register_field(vectors).is_err(),
        "storeTermVectors changed"
    );
    let mut payloads = FieldInfo::new("p", 0).with_index_options(IndexOptions::DocsAndFreqs);
    payloads.store_payloads = true;
    assert!(w.register_field(payloads).is_err());
}

#[test]
fn reopened_writer_reserves_all_source_numbers_before_a_new_field_appears() {
    let dir = lucene_store::ByteBuffersDirectory::new();
    {
        let mut writer = IndexWriter::open(
            &dir,
            vec![
                FieldInfo::new("old_a", 2)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true),
                FieldInfo::new("old_b", 5),
            ],
            "Lucene104",
            VERSION,
        )
        .unwrap();
        writer.enable_explicit_documents().unwrap();
        writer
            .add_explicit_documents(vec![ExplicitDocument {
                stored: vec![
                    StoredField {
                        field_number: 2,
                        value: FieldValue::String("a".into()),
                    },
                    StoredField {
                        field_number: 5,
                        value: FieldValue::String("b".into()),
                    },
                ],
                fields: ExplicitFields {
                    inverted: vec![InvertedField {
                        field_number: 2,
                        terms: vec![InvertedTerm {
                            term: b"drop".to_vec(),
                            freq: 1,
                            positions: Vec::new(),
                            offsets: Vec::new(),
                            payloads: Vec::new(),
                        }]
                        .into(),
                        norm: None,
                    }],
                    ..ExplicitFields::default()
                },
            }])
            .unwrap();
        writer.commit().unwrap();
    }
    let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    writer.enable_explicit_documents().unwrap();
    assert!(
        writer.cfg.fields.is_empty(),
        "reservations must not populate configured schemas"
    );
    assert_eq!(
        writer.register_field(FieldInfo::new("late", 99)).unwrap(),
        0
    );
    assert_eq!(
        writer
            .register_field(
                FieldInfo::new("old_a", 99)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true)
            )
            .unwrap(),
        2
    );
    assert!(
        writer
            .register_field(FieldInfo::new("old_b", 99).with_index_options(IndexOptions::Docs))
            .is_err(),
        "source schemas are still checked"
    );
    // The source carrying old_b can disappear before this session has
    // registered it. Its reserved number survives for the writer's life.
    writer
        .delete_documents_by_term(&[Term::new("old_a", "drop")])
        .unwrap();
    writer.commit().unwrap();
    assert!(writer.segment_infos.segments.is_empty());
    assert_eq!(
        writer.register_field(FieldInfo::new("old_b", 99)).unwrap(),
        5
    );
    assert_eq!(
        writer
            .register_field(FieldInfo::new("another", 99))
            .unwrap(),
        1
    );
    assert_eq!(
        writer
            .register_field(FieldInfo::new("after_gap", 99))
            .unwrap(),
        3
    );
}

#[test]
fn delete_all_rollback_restores_source_numbers_and_schema_before_new_fields() {
    for (fallible, configured) in [(false, false), (true, false), (false, true), (true, true)] {
        let dir = lucene_store::ByteBuffersDirectory::new();
        let original = FieldInfo::new("old", 0).with_index_options(IndexOptions::DocsAndFreqs);
        {
            let mut writer =
                IndexWriter::open(&dir, vec![original.clone()], "Lucene104", VERSION).unwrap();
            writer.enable_explicit_documents().unwrap();
            writer
                .add_explicit_documents(vec![ExplicitDocument {
                    fields: ExplicitFields {
                        inverted: vec![InvertedField {
                            field_number: 0,
                            terms: Vec::new().into(),
                            norm: Some(0),
                        }],
                        ..ExplicitFields::default()
                    },
                    ..ExplicitDocument::default()
                }])
                .unwrap();
            writer.commit().unwrap();
        }
        let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        writer.enable_explicit_documents().unwrap();
        if configured {
            assert_eq!(
                writer.register_field(FieldInfo::new("pinned", 99)).unwrap(),
                1
            );
        }
        writer.delete_all().unwrap();
        if fallible {
            writer.try_rollback().unwrap();
        } else {
            writer.rollback();
        }
        assert_eq!(
            writer.register_field(FieldInfo::new("new", 99)).unwrap(),
            if configured { 2 } else { 1 }
        );
        if configured {
            assert_eq!(
                writer.register_field(FieldInfo::new("pinned", 99)).unwrap(),
                1
            );
        }
        assert!(writer.register_field(FieldInfo::new("old", 99)).is_err());
        assert!(writer
            .register_field(original.clone().with_omit_norms(true))
            .is_err());
        assert_eq!(writer.register_field(original).unwrap(), 0);
    }
}

#[test]
fn rollback_schema_conflict_keeps_current_writer_state_coherent() {
    for fallible in [false, true] {
        let dir = lucene_store::ByteBuffersDirectory::new();
        {
            let mut writer = IndexWriter::open(
                &dir,
                vec![FieldInfo::new("field", 0).with_index_options(IndexOptions::DocsAndFreqs)],
                "Lucene104",
                VERSION,
            )
            .unwrap();
            writer.enable_explicit_documents().unwrap();
            writer
                .add_explicit_documents(vec![ExplicitDocument {
                    fields: ExplicitFields {
                        inverted: vec![InvertedField {
                            field_number: 0,
                            terms: Vec::new().into(),
                            norm: Some(0),
                        }],
                        ..ExplicitFields::default()
                    },
                    ..ExplicitDocument::default()
                }])
                .unwrap();
            writer.commit().unwrap();
        }
        let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        writer.enable_explicit_documents().unwrap();
        let original_files: Vec<_> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .map(|name| {
                let bytes = dir.open(&name).unwrap().to_vec();
                (name, bytes)
            })
            .collect();
        writer.delete_all().unwrap();
        let changed = writer.register_field(FieldInfo::new("field", 99)).unwrap();
        assert_eq!(changed, 0);
        writer
            .add_explicit_documents(vec![ExplicitDocument {
                stored: vec![StoredField {
                    field_number: changed,
                    value: FieldValue::String("replacement".into()),
                }],
                ..ExplicitDocument::default()
            }])
            .unwrap();
        if fallible {
            assert!(matches!(writer.try_rollback(), Err(Error::Explicit(_))));
        } else {
            writer.rollback();
        }
        assert!(
            writer.segment_infos.segments.is_empty(),
            "incompatible committed sources were not restored"
        );
        for (name, bytes) in &original_files {
            assert_eq!(
                &*dir.open(name).unwrap(),
                bytes.as_slice(),
                "rejected rollback preserves committed source bytes"
            );
        }
        assert_eq!(
            writer.pending_docs.len(),
            1,
            "failed rollback leaves the current buffer intact"
        );
        assert_eq!(
            writer.register_field(FieldInfo::new("field", 99)).unwrap(),
            changed
        );
        let committed = writer.commit().unwrap().clone();
        assert_eq!(committed.segments.len(), 1);
        let fields = super::super::segment_field_infos(&dir, &committed.segments[0]).unwrap();
        assert_eq!(fields.fields[0].index_options, IndexOptions::None);
        for result in crate::check_index::check_directory(&dir).unwrap() {
            assert!(
                result.all_passed(),
                "{}: {:?}",
                result.segment_name,
                result.failures()
            );
        }
    }
}

#[test]
fn reserved_schemas_keep_only_global_properties_and_reject_changes() {
    let mut numbers = FieldNumbers::default();
    let mut info = FieldInfo::new("field", 3).with_index_options(IndexOptions::DocsAndFreqs);
    info.store_payloads = true;
    info.doc_values_gen = 19;
    info.attributes.push(("segment".into(), "attribute".into()));
    assert_eq!(numbers.register(&info, Some(3)).unwrap(), 3);
    let retained = numbers.schema("field").unwrap();
    assert_eq!(retained.number, 3);
    assert!(!retained.store_payloads);
    assert_eq!(retained.doc_values_gen, -1);
    assert!(retained.attributes.is_empty());
    assert_eq!(numbers.register(&info, None).unwrap(), 3);
    assert!(numbers.register(&info.with_omit_norms(true), None).is_err());
}

#[test]
fn source_number_conflicts_use_first_preference_then_lowest_free_number() {
    let mut numbers = FieldNumbers::default();
    assert_eq!(numbers.add_or_get("first", Some(2)).unwrap(), 2);
    assert_eq!(numbers.add_or_get("second", Some(2)).unwrap(), 0);
    // A repeated name keeps its first global assignment, regardless of
    // the number a later source segment assigned to the same name.
    assert_eq!(numbers.add_or_get("first", Some(1)).unwrap(), 2);
    assert_eq!(numbers.add_or_get("third", Some(0)).unwrap(), 1);
    assert_eq!(numbers.add_or_get("preferred", Some(7)).unwrap(), 7);
    assert_eq!(numbers.add_or_get("new", None).unwrap(), 3);
    assert_eq!(numbers.add_or_get("unset_source", Some(-1)).unwrap(), 4);
    // Saturating a field number must report exhaustion instead of
    // returning another field's already reserved number.
    numbers.lowest_unassigned = i32::MAX;
    assert_eq!(numbers.add_or_get("last", None).unwrap(), i32::MAX);
    assert!(numbers.add_or_get("overflow", None).is_err());
}

/// Term vectors arrive built; `validate_explicit` checks their shape
/// against the registered schema.
#[test]
fn term_vectors_are_checked_at_add_time() {
    use lucene_codecs::term_vectors::TermVectorTerm;
    let dir = lucene_store::ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let indexed = |name: &str| {
        FieldInfo::new(name, 0)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_omit_norms(true)
    };
    let tv = w
        .register_field(indexed("tv").with_store_term_vectors(true))
        .unwrap();
    let plain = w.register_field(indexed("plain")).unwrap();
    let term = |t: &str| TermVectorTerm {
        term: t.as_bytes().to_vec(),
        freq: 1,
        positions: Some(vec![0]),
        start_offsets: None,
        end_offsets: None,
        payloads: None,
    };
    let field = |n: i32, terms: Vec<TermVectorTerm>| TermVectorField {
        field_number: n,
        has_positions: true,
        has_offsets: false,
        has_payloads: false,
        terms,
    };
    let with = |fields: Vec<TermVectorField>| {
        let mut numbers: Vec<i32> = fields.iter().map(|f| f.field_number).collect();
        numbers.dedup();
        ExplicitDocument {
            stored: Vec::new(),
            fields: ExplicitFields {
                inverted: numbers
                    .into_iter()
                    .map(|n| InvertedField {
                        field_number: n,
                        terms: vec![term_docs("a")].into(),
                        norm: None,
                    })
                    .collect(),
                term_vectors: fields,
                ..ExplicitFields::default()
            },
        }
    };
    let mut not_inverted = with(vec![field(tv, vec![term("a")])]);
    not_inverted.fields.inverted.clear();
    let err = w.add_explicit_documents(vec![not_inverted]).unwrap_err();
    assert!(matches!(err, Error::Explicit(_)), "{err}");
    // Inverted into terms without the vector Java would have recorded.
    let mut no_vector = with(vec![field(tv, vec![term("a")])]);
    no_vector.fields.term_vectors.clear();
    let err = w.add_explicit_documents(vec![no_vector]).unwrap_err();
    assert!(err.to_string().contains("gives none"), "{err}");
    w.add_explicit_documents(vec![with(vec![field(tv, vec![term("a")])])])
        .unwrap();
    let mut payloads_alone = field(tv, vec![term("a")]);
    payloads_alone.has_positions = false;
    payloads_alone.has_payloads = true;
    let mut no_positions = term("a");
    no_positions.positions = None;
    let mut zero = term("a");
    zero.freq = 0;
    let cases = [
        ("unregistered", vec![field(99, vec![term("a")])]),
        ("no term vectors", vec![field(plain, vec![term("a")])]),
        (
            "twice",
            vec![field(tv, vec![term("a")]), field(tv, vec![term("b")])],
        ),
        ("payloads without positions", vec![payloads_alone]),
        ("unsorted", vec![field(tv, vec![term("b"), term("a")])]),
        ("duplicate", vec![field(tv, vec![term("a"), term("a")])]),
        ("missing positions", vec![field(tv, vec![no_positions])]),
        ("freq 0", vec![field(tv, vec![zero])]),
    ];
    for (what, fields) in cases {
        let err = w
            .add_explicit_documents(vec![with(fields)])
            .expect_err(what);
        assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
    }
    w.commit().unwrap();
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
}

#[test]
fn explicit_mode_must_come_first() {
    let tmp = TempDir::new("explicit-first");
    let dir = FsDirectory::open(tmp.path());
    let mut w =
        IndexWriter::open(&dir, vec![FieldInfo::new("s", 0)], "Lucene104", VERSION).unwrap();
    assert!(w
        .add_explicit_documents(vec![ExplicitDocument::default()])
        .is_err());
    w.add_document(Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::Long(1),
        }],
    })
    .unwrap();
    assert!(w.enable_explicit_documents().is_err());
}

/// Everything `validate_explicit` refuses, one case each.
#[test]
fn invalid_documents_are_refused_at_add_time() {
    let tmp = TempDir::new("explicit-invalid");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    assert_eq!(w.register_parent_field().unwrap(), None);
    let base = || doc(&f, 1);
    let mut cases: Vec<(&str, ExplicitDocument)> = Vec::new();
    let mut d = base();
    d.stored[0].field_number = 99;
    cases.push(("unknown stored field", d));
    let mut d = base();
    d.fields.inverted.push(d.fields.inverted[0].clone());
    cases.push(("field inverted twice", d));
    let mut d = base();
    d.fields.inverted[0].field_number = f.source;
    cases.push(("not indexed", d));
    let mut d = base();
    d.fields.inverted[1].norm = None;
    cases.push(("missing norm", d));
    let mut d = base();
    d.fields.inverted[0].norm = Some(1);
    cases.push(("norm on an omitNorms field", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| t.push(t[0].clone()));
    cases.push(("duplicate term", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[0].freq = 0;
    });
    cases.push(("freq 0", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[1].positions = vec![3, 1];
    });
    cases.push(("descending positions", d));
    let mut d = base();
    d.fields.inverted[0].terms.edit(|t| {
        t[0].positions = vec![0];
    });
    cases.push(("positions on a docs-only field", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[0].positions.pop();
    });
    cases.push(("positions out of step with freq", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[0].offsets.pop();
    });
    cases.push(("offsets out of step with freq", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[0].offsets = vec![(5, 1)];
    });
    cases.push(("end before start", d));
    let mut d = base();
    d.fields.inverted[0].terms.edit(|t| {
        t[0].offsets = vec![(0, 1)];
    });
    cases.push(("offsets on a docs-only field", d));
    let mut d = base();
    d.fields.doc_values[0].value = FieldValue::Long(1);
    cases.push(("a long for a sorted-set field", d));
    let mut d = base();
    d.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    d.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    cases.push(("two numeric values", d));
    let mut d = base();
    d.fields.points[0].value = FieldValue::Binary(vec![0; 3]);
    cases.push(("short point", d));
    let mut d = base();
    d.fields.inverted[1].terms.edit(|t| {
        t[0].payloads = vec![vec![1]; 7];
    });
    cases.push(("payloads out of step with positions", d));
    for (what, d) in cases {
        let err = w.add_explicit_documents(vec![d]).expect_err(what);
        assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
    }
    let soft = w.soft_update_explicit_documents(
        Term {
            field: "_id".to_string(),
            bytes: b"d1".to_vec(),
        },
        vec![base()],
        &[],
    );
    assert!(matches!(soft, Err(Error::NoSoftDeletesSupplied)));
}

/// Every refusal `validate_explicit_vectors` makes, and the vector-list
/// count check, on both the add and the soft-update entry points.
#[test]
fn invalid_vectors_are_refused_at_add_time() {
    use crate::index_writer::DocumentVector;
    use lucene_codecs::field_infos::{VectorEncoding, VectorSimilarityFunction};
    let tmp = TempDir::new("explicit-vectors-invalid");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.register_field(
        FieldInfo::new("v", 0)
            .with_index_options(IndexOptions::None)
            .with_vectors(2, VectorEncoding::Float32, VectorSimilarityFunction::Cosine),
    )
    .unwrap();
    let ok = || DocumentVector::float32("v", vec![1.0, 2.0]);
    let cases: Vec<(&str, Vec<DocumentVector>)> = vec![
        (
            "unregistered vector field",
            vec![DocumentVector::float32("nope", vec![1.0, 2.0])],
        ),
        (
            "registered but not a vector",
            vec![DocumentVector::float32("_id", vec![1.0, 2.0])],
        ),
        (
            "encoding mismatch",
            vec![DocumentVector::byte("v", vec![1, 2])],
        ),
        (
            "dimension mismatch",
            vec![DocumentVector::float32("v", vec![1.0])],
        ),
        ("repeated field", vec![ok(), ok()]),
    ];
    for (what, vectors) in cases {
        let err = w.cfg.validate_explicit_vectors(&vectors).expect_err(what);
        assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
        let err = w
            .add_explicit_documents_with_vectors(None, vec![doc(&f, 1)], vec![vectors])
            .expect_err(what);
        assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
    }
    w.cfg.validate_explicit_vectors(&[ok()]).unwrap();
    let err = w
        .add_explicit_documents_with_vectors(None, vec![doc(&f, 1)], Vec::new())
        .expect_err("vector list count");
    assert!(matches!(err, Error::Explicit(_)), "{err}");
    let err = w
        .soft_update_explicit_documents_with_vectors(
            Term {
                field: "_id".to_string(),
                bytes: b"d1".to_vec(),
            },
            vec![doc(&f, 1)],
            vec![vec![DocumentVector::byte("v", vec![1, 2])]],
            &[soft_delete(1)],
        )
        .expect_err("soft update encoding");
    assert!(matches!(err, Error::Explicit(_)), "{err}");
}

#[test]
fn stored_only_adds_mark_their_parent() {
    // Java's `addDocument`/`addDocuments` mark the last document of
    // every add once a parent field is configured; the stored-only
    // entry points must mark it too, or a block has no parent and
    // `MockRandomMergePolicy.reverse` sees none.
    let tmp = TempDir::new("explicit-stored-only-parent");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_parent_field(Some("_parent")).unwrap();
    let empty = || Document { fields: Vec::new() };
    w.add_documents(vec![empty(), empty()]).unwrap();
    w.add_document(empty()).unwrap();
    w.add_document_with_custom_freq_terms(empty(), Vec::new())
        .unwrap();
    w.add_document_with_vectors(empty(), Vec::new()).unwrap();
    w.add_documents(Vec::new()).unwrap();
    let parent = w.cfg.parent_field_number().unwrap();
    let marked: Vec<bool> = w
        .pending_explicit
        .iter()
        .map(|f| f.doc_values.iter().any(|v| v.field_number == parent))
        .collect();
    assert_eq!(marked, [false, true, true, true, true]);
    w.commit().unwrap();
    let sci = w.segment_infos.segments[0].clone();
    let infos = super::super::segment_field_infos(&dir, &sci).unwrap();
    assert_eq!(infos.parent_field(), Some("_parent"));
}

/// The refusals `register_field` and `set_parent_field` make: schema
/// shapes Lucene rejects, and the parent field's rules.
#[test]
fn field_registration_and_parent_field_refusals() {
    use crate::segment_info::IndexSortField;
    let explicit = |what: &str, r: Result<()>| {
        assert!(matches!(r, Err(Error::Explicit(_))), "{what}: {r:?}");
    };
    let tmp = TempDir::new("explicit-field-refusals");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    // Before explicit mode: both refused or switched on.
    explicit(
        "register before enabling",
        w.register_field(FieldInfo::new("a", 0)).map(drop),
    );
    w.enable_explicit_documents().unwrap();
    explicit(
        "term vectors",
        w.register_field(FieldInfo::new("a", 0).with_store_term_vectors(true))
            .map(drop),
    );
    explicit(
        "skip index on binary doc values",
        w.register_field(FieldInfo::new("a", 0).with_doc_values(
            DocValuesType::Binary,
            DocValuesSkipIndexType::Range,
            -1,
        ))
        .map(drop),
    );
    explicit(
        "parent field without a configured parent",
        w.register_field(FieldInfo::new("a", 0).with_parent_field(true))
            .map(drop),
    );
    // A conflicting schema for a known name.
    w.register_field(FieldInfo::new("c", 0).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    explicit(
        "changed doc values type",
        w.register_field(FieldInfo::new("c", 0).with_doc_values(
            DocValuesType::Sorted,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .map(drop),
    );
    // set_parent_field against what is registered.
    explicit(
        "parent named as a registered non-parent field",
        w.set_parent_field(Some("c")),
    );
    w.register_field(
        FieldInfo::new("soft", 0)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
            .with_soft_deletes_field(true),
    )
    .unwrap();
    explicit(
        "parent named as the soft-deletes field",
        w.set_parent_field(Some("soft")),
    );
    explicit(
        "parent added to an index that has other fields",
        w.set_parent_field(Some("p")),
    );

    // A fresh writer: the parent field proper.
    let tmp2 = TempDir::new("explicit-parent-field");
    let dir2 = FsDirectory::open(tmp2.path());
    let mut w = IndexWriter::open(&dir2, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_parent_field(Some("p")).unwrap();
    assert_eq!(w.parent_field(), Some("p"));
    explicit(
        "another field as the parent",
        w.register_field(FieldInfo::new("q", 0).with_parent_field(true))
            .map(drop),
    );
    explicit(
        "the parent name as an ordinary field",
        w.register_field(FieldInfo::new("p", 0)).map(drop),
    );
    w.register_parent_field().unwrap();
    explicit(
        "another parent over a registered one",
        w.set_parent_field(Some("r")),
    );
    explicit("clearing a registered parent", w.set_parent_field(None));
    let f = register(&mut w);
    w.add_explicit_documents(vec![doc(&f, 1)]).unwrap();
    explicit(
        "setting the parent with documents buffered",
        w.set_parent_field(Some("p")),
    );

    // Clearing a parent that was never registered.
    let tmp3 = TempDir::new("explicit-parent-clear");
    let dir3 = FsDirectory::open(tmp3.path());
    let mut w = IndexWriter::open(&dir3, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_parent_field(Some("p")).unwrap();
    w.set_parent_field(None).unwrap();
    assert_eq!(w.parent_field(), None);

    // An index sort names a field; registering it with other doc values
    // is refused.
    let tmp4 = TempDir::new("explicit-sort-mismatch");
    let dir4 = FsDirectory::open(tmp4.path());
    let mut w = IndexWriter::open(&dir4, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.set_index_sort(Some(&[IndexSortField::long("v", true, None)]))
        .unwrap();
    explicit(
        "sort field with the wrong doc values",
        w.register_field(FieldInfo::new("v", 0).with_doc_values(
            DocValuesType::Sorted,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .map(drop),
    );
    let err = w
        .set_index_sort(Some(&[IndexSortField::long("v", true, None)]))
        .map(drop);
    assert!(err.is_ok(), "{err:?}");
}

/// `update_explicit_documents` hard-deletes the earlier document.
#[test]
fn update_replaces_the_earlier_document() {
    let tmp = TempDir::new("explicit-update");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.add_explicit_documents((0..4).map(|i| doc(&f, i)).collect())
        .unwrap();
    w.commit().unwrap();
    w.update_explicit_documents(
        Term {
            field: "_id".to_string(),
            bytes: b"d2".to_vec(),
        },
        vec![doc(&f, 2)],
    )
    .unwrap();
    let infos = w.commit().unwrap().clone();
    // `committed_doc_count` is maxDoc; the live count subtracts deletes.
    let deleted: i32 = infos.segments.iter().map(|s| s.del_count).sum();
    assert_eq!(w.committed_doc_count().unwrap(), 5);
    assert_eq!(deleted, 1, "the earlier d2 is deleted");
    check(&dir);
}

fn id_term(i: i64) -> Term {
    Term {
        field: "_id".to_string(),
        bytes: format!("d{i}").into_bytes(),
    }
}

/// Document `i` re-indexed as the operation with sequence number `seq_no`.
fn doc_at(f: &Fields, i: i64, seq_no: i64) -> ExplicitDocument {
    let mut d = doc(f, i);
    for v in &mut d.fields.doc_values {
        if v.field_number == f.seq {
            v.value = FieldValue::Long(seq_no);
        }
    }
    d
}

fn soft_delete(i: i64) -> DocValuesUpdate {
    DocValuesUpdate::Numeric {
        term: id_term(i),
        field: "__soft_deletes".to_string(),
        value: Some(1),
    }
}

/// `(maxDoc, softDelCount)` over the committed segments.
fn counts(w: &mut IndexWriter<'_>) -> (usize, i32) {
    let soft = w
        .commit()
        .unwrap()
        .segments
        .iter()
        .map(|s| s.soft_del_count)
        .sum();
    (w.committed_doc_count().unwrap(), soft)
}

/// `SoftDeletesRetentionMergePolicy`: a merge carries a soft-deleted
/// document over while its `_seq_no` is retained, and drops it after.
#[test]
fn merges_drop_only_the_soft_deleted_history_no_longer_retained() {
    let tmp = TempDir::new("explicit-retention");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.set_max_buffered_docs(4).unwrap();
    for i in 0..10 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    // Operations 100..=105 replace d0..d5; seq_nos 0..=5 become history.
    for i in 0..6 {
        w.soft_update_explicit_documents(
            id_term(i),
            vec![doc_at(&f, i, 100 + i)],
            &[soft_delete(i)],
        )
        .unwrap();
    }
    assert_eq!(counts(&mut w), (16, 6));

    // No retention policy: every soft-deleted document survives.
    w.force_merge(1).unwrap();
    assert_eq!(w.commit().unwrap().segments.len(), 1);
    assert_eq!(counts(&mut w), (16, 6));
    check(&dir);

    // Seq_nos 0..=2 fall out of retention; 3..=5 are still needed.
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 3,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    w.force_merge_deletes().unwrap();
    assert_eq!(counts(&mut w), (13, 3));
    check(&dir);

    // Nothing retained: all history goes, and a clean index is left alone.
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: i64::MAX,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    w.force_merge_deletes().unwrap();
    assert_eq!(counts(&mut w), (10, 0));
    let before = w.commit_generations();
    w.force_merge_deletes().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(w.commit_generations(), before, "nothing to merge");
    check(&dir);

    assert!(matches!(
        w.force_merge(0),
        Err(Error::InvalidMaxNumSegments(0))
    ));
}

/// Merge hooks that keep every document the sources' hard deletes keep,
/// as hooks reading the sources through a reader do.
#[derive(Debug)]
struct KeepHardLive;

impl crate::merge_policy::MergeHooks for KeepHardLive {
    fn prepare(
        &self,
        dir: &dyn Directory,
        sources: &crate::segment_infos::SegmentInfos,
        _may_reorder: bool,
    ) -> std::result::Result<crate::merge_policy::PreparedMerge, String> {
        let mut live_docs = Vec::new();
        for sci in &sources.segments {
            let si = dir
                .open(&format!("{}.si", sci.segment_name))
                .map_err(|e| e.to_string())?;
            let max_doc = crate::segment_info::parse(&si, &sci.segment_id)
                .map_err(|e| e.to_string())?
                .doc_count;
            let mut bits =
                lucene_util::fixed_bit_set::FixedBitSet::new(usize::try_from(max_doc).unwrap());
            bits.set_range(0, bits.len());
            live_docs.push(Some(bits));
        }
        Ok(crate::merge_policy::PreparedMerge {
            live_docs,
            new_to_old: None,
            diagnostics: vec![("test.hooked".to_string(), "yes".to_string())],
        })
    }
}

/// A merge with hooks still drops the soft-deleted history retention no
/// longer keeps, though the hooks' own live documents keep it.
#[test]
fn hooked_merges_still_drop_unretained_soft_deletes() {
    let tmp = TempDir::new("explicit-retention-hooks");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.set_max_buffered_docs(4).unwrap();
    for i in 0..10 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    for i in 0..6 {
        w.soft_update_explicit_documents(
            id_term(i),
            vec![doc_at(&f, i, 100 + i)],
            &[soft_delete(i)],
        )
        .unwrap();
    }
    assert_eq!(counts(&mut w), (16, 6));
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 3,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    let names: Vec<String> = w
        .commit()
        .unwrap()
        .segments
        .iter()
        .map(|s| s.segment_name.clone())
        .collect();
    w.execute_merge_hooked(&names, Some(std::sync::Arc::new(KeepHardLive)))
        .unwrap();
    assert_eq!(counts(&mut w), (13, 3));
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    let si = crate::segment_info::parse(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
    )
    .unwrap();
    assert!(si
        .diagnostics
        .iter()
        .any(|(k, v)| k == "test.hooked" && v == "yes"));
    assert!(si
        .diagnostics
        .iter()
        .any(|(k, v)| k == "source" && v == "merge"));
    check(&dir);
}

/// The `_id` document frequency of each of `terms` in the index's only
/// segment, or `None` when that segment writes no `_id` terms at all.
fn id_doc_freqs(dir: &FsDirectory, w: &mut IndexWriter<'_>, terms: &[&str]) -> Option<Vec<i32>> {
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let sci = &infos.segments[0];
    let seg = crate::index_writer::per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
    let suffix = crate::index_writer::per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let open = |ext: &str| dir.open(&format!("{seg}.{ext}")).unwrap();
    let (tim, tip, tmd) = (open("tim"), open("tip"), open("tmd"));
    let field_infos = lucene_codecs::field_infos::FieldInfos {
        fields: segment_fields(dir, sci),
    };
    let tree = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &sci.segment_id,
        &suffix,
        i32::try_from(w.committed_doc_count().unwrap()).unwrap(),
    )
    .unwrap();
    let field = tree.field("_id")?;
    Some(
        terms
            .iter()
            .map(|t| field.seek_exact(t.as_bytes()).map_or(0, |s| s.doc_freq))
            .collect(),
    )
}

/// `PrunePostingsMergePolicy`: a retention merge keeps a soft-deleted
/// document for its history but drops its `_id` postings, and writes no
/// `_id` terms at all once every document is soft-deleted.
#[test]
fn retention_merges_prune_the_ids_of_soft_deleted_documents() {
    let tmp = TempDir::new("explicit-prune");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    for i in 0..3 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    w.soft_update_explicit_documents(id_term(0), vec![doc_at(&f, 0, 100)], &[soft_delete(0)])
        .unwrap();
    let retention = |prune: bool| SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 0,
        prune_postings_field: prune.then(|| "_id".to_string()),
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    };

    // Without pruning the retained old version still answers `d0`.
    w.set_soft_deletes_retention(Some(retention(false)));
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (4, 1));
    assert_eq!(id_doc_freqs(&dir, &mut w, &["d0", "d1"]), Some(vec![2, 1]));

    // With pruning, the merge keeps both old versions but not their ids.
    w.soft_update_explicit_documents(id_term(1), vec![doc_at(&f, 1, 101)], &[soft_delete(1)])
        .unwrap();
    w.set_soft_deletes_retention(Some(retention(true)));
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (5, 2), "the history is kept");
    assert_eq!(
        id_doc_freqs(&dir, &mut w, &["d0", "d1", "d2"]),
        Some(vec![1, 1, 1]),
        "only the live versions keep their ids"
    );
    check(&dir);

    // Every document soft-deleted: no `_id` term survives.
    let tmp = TempDir::new("explicit-prune-all");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
    w.commit().unwrap();
    let mut tombstone = doc_at(&f, 0, 1);
    tombstone.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    w.soft_update_explicit_documents(id_term(0), vec![tombstone], &[soft_delete(0)])
        .unwrap();
    w.set_soft_deletes_retention(Some(retention(true)));
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (2, 2));
    assert_eq!(id_doc_freqs(&dir, &mut w, &["d0"]), None);
    check(&dir);
}

/// `MergePolicy.numDeletesToMerge`: `force_merge(1)` leaves a single
/// segment whose soft-deleted documents are all still retained alone,
/// and rewrites it once retention lets them go.
#[test]
fn force_merge_rewrites_one_segment_only_for_reclaimable_soft_deletes() {
    let tmp = TempDir::new("explicit-reclaimable");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    for i in 0..3 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    w.soft_update_explicit_documents(id_term(0), vec![doc_at(&f, 0, 100)], &[soft_delete(0)])
        .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (4, 1));
    let retention = |min: i64| SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: min,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    };

    w.set_soft_deletes_retention(Some(retention(0)));
    let before = w.commit_generations();
    w.force_merge(1).unwrap();
    assert_eq!(w.commit_generations(), before, "all history is retained");
    w.force_merge_deletes().unwrap();
    assert_eq!(w.commit_generations(), before, "nothing to expunge either");

    w.set_soft_deletes_retention(Some(retention(1)));
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (3, 0), "seq_no 0 is reclaimed");
    check(&dir);
}

/// `RecoverySourcePruneMergePolicy`: a retention merge keeps the stored
/// `_recovery_source` and its doc-values flag only on documents whose
/// sequence number is still retained.
#[test]
fn retention_merges_prune_the_recovery_source_of_unretained_documents() {
    let tmp = TempDir::new("explicit-recovery-source");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    let recovery = w
        .register_field(FieldInfo::new("_recovery_source", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
    let with_recovery = |mut d: ExplicitDocument| {
        d.stored.push(StoredField {
            field_number: recovery,
            value: FieldValue::Binary(b"{}".to_vec()),
        });
        d.fields.doc_values.push(StoredField {
            field_number: recovery,
            value: FieldValue::Long(1),
        });
        d
    };
    for i in 0..3 {
        w.add_explicit_documents(vec![with_recovery(doc(&f, i))])
            .unwrap();
    }
    w.commit().unwrap();
    w.soft_update_explicit_documents(
        id_term(0),
        vec![with_recovery(doc_at(&f, 0, 100))],
        &[soft_delete(0)],
    )
    .unwrap();
    w.commit().unwrap();
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 2,
        prune_postings_field: None,
        prune_recovery_source_field: Some("_recovery_source".to_string()),
        seq_no_sentinel: Vec::new(),
    }));
    w.force_merge(1).unwrap();
    let infos = w.commit().unwrap().clone();
    check(&dir);

    // d0's old version (seq_no 0) is gone; d1 (1) is live but no longer
    // retained; d2 (2) and d0's new version (100) are retained.
    let sci = &infos.segments[0];
    assert_eq!((infos.segments.len(), sci.soft_del_count), (1, 0));
    let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
    let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
    let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
    let stored = lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    let si_files: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|n| {
            n.starts_with(&format!("{}.", sci.segment_name))
                || n.starts_with(&format!("{}_", sci.segment_name))
        })
        .collect();
    let fields = crate::field_updates::read_current_field_infos(&dir, sci, &si_files).unwrap();
    let number = |name: &str| fields.fields.iter().position(|f| f.name == name).unwrap();
    let (id_at, recovery_at) = (number("_id"), number("_recovery_source"));
    let mut kept = Vec::new();
    for doc in 0..stored.max_doc() {
        let d = stored.document(doc).unwrap();
        let id = d
            .fields
            .iter()
            .find_map(|v| match &v.value {
                FieldValue::String(s) if v.field_number == fields.fields[id_at].number => {
                    Some(s.clone())
                }
                _ => None,
            })
            .unwrap();
        if d.fields
            .iter()
            .any(|v| v.field_number == fields.fields[recovery_at].number)
        {
            kept.push(id);
        }
    }
    kept.sort();
    assert_eq!(kept, ["d0", "d2"], "stored");

    let per_field = crate::field_updates::per_field_component(
        &fields.fields[recovery_at],
        &crate::index_writer::per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    );
    let (meta, data) = crate::field_updates::read_current_column(
        &dir,
        sci,
        &si_files,
        &fields,
        recovery_at,
        &per_field,
    )
    .unwrap()
    .unwrap();
    let entry = meta
        .numeric_entry(fields.fields[recovery_at].number)
        .unwrap();
    let mut flags = lucene_codecs::doc_values::NumericReader::new(&data, entry);
    let mut flagged: Vec<String> = (0..stored.max_doc())
        .filter(|&d| flags.value(d).unwrap().is_some())
        .map(|d| {
            stored
                .document(d)
                .unwrap()
                .fields
                .into_iter()
                .find_map(|v| match v.value {
                    FieldValue::String(s) if v.field_number == fields.fields[id_at].number => {
                        Some(s)
                    }
                    _ => None,
                })
                .unwrap()
        })
        .collect();
    flagged.sort();
    assert_eq!(flagged, ["d0", "d2"], "doc values");
}

/// `seq_no_sentinel` (Elasticsearch's `index.disable_sequence_numbers`): a
/// retention merge writes the sentinel for every `_seq_no` below the
/// retained minimum and keeps the others, so the column stays dense and
/// history above the minimum still replays.
#[test]
fn retention_merges_write_the_sentinel_for_unretained_sequence_numbers() {
    let tmp = TempDir::new("explicit-seq-no-sentinel");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    for i in 0..4 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    w.soft_update_explicit_documents(id_term(3), vec![doc_at(&f, 3, 100)], &[soft_delete(3)])
        .unwrap();
    w.commit().unwrap();
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 2,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: vec![
            ("_seq_no".to_string(), -2),
            ("_primary_term".to_string(), 0),
        ],
    }));
    w.force_merge(1).unwrap();
    let infos = w.commit().unwrap().clone();
    check(&dir);

    let sci = &infos.segments[0];
    // d3's old version (seq_no 3) is soft-deleted but retained.
    assert_eq!((infos.segments.len(), sci.soft_del_count), (1, 1));
    let si_files: Vec<String> = dir
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|n| {
            n.starts_with(&format!("{}.", sci.segment_name))
                || n.starts_with(&format!("{}_", sci.segment_name))
        })
        .collect();
    let fields = crate::field_updates::read_current_field_infos(&dir, sci, &si_files).unwrap();
    let seq_at = fields
        .fields
        .iter()
        .position(|x| x.name == "_seq_no")
        .unwrap();
    let per_field = crate::field_updates::per_field_component(
        &fields.fields[seq_at],
        &crate::index_writer::per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    );
    let (meta, data) = crate::field_updates::read_current_column(
        &dir, sci, &si_files, &fields, seq_at, &per_field,
    )
    .unwrap()
    .unwrap();
    let entry = meta.numeric_entry(fields.fields[seq_at].number).unwrap();
    let mut seq = lucene_codecs::doc_values::NumericReader::new(&data, entry);
    // d0..d3, d3's old version kept soft-deleted, and its update.
    let mut values: Vec<i64> = (0..5).map(|d| seq.value(d).unwrap().unwrap()).collect();
    values.sort();
    assert_eq!(values, [-2, -2, 2, 3, 100]);
    // A sentinel field the segment does not have is skipped.
    assert!(fields.fields.iter().all(|x| x.name != "_primary_term"));
}

/// `IndexWriter.reserveDocs`: an add past the limit is refused before
/// anything -- its own delete included -- is buffered, deleted documents
/// count until a merge drops them, and the limit is clamped to
/// `MAX_DOCS`.
#[test]
fn adds_past_max_docs_are_refused_whole() {
    let tmp = TempDir::new("explicit-max-docs");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.set_max_docs(3);
    w.add_explicit_documents(vec![doc(&f, 0), doc(&f, 1)])
        .unwrap();
    w.commit().unwrap();
    w.add_explicit_documents(vec![doc(&f, 2)]).unwrap();
    let too_many = |r: Result<super::super::SeqNo>| {
        assert!(matches!(r, Err(Error::TooManyDocs(3))), "{r:?}");
    };
    too_many(w.add_explicit_documents(vec![doc(&f, 3)]));
    too_many(w.soft_update_explicit_documents(
        id_term(0),
        vec![doc_at(&f, 0, 100)],
        &[soft_delete(0)],
    ));
    assert_eq!(counts(&mut w), (3, 0), "the refused update deleted nothing");

    // A hard delete frees its slot only once a merge drops the document.
    w.update_explicit_documents(id_term(9), Vec::new()).unwrap();
    w.soft_update_explicit_documents(id_term(0), Vec::new(), &[soft_delete(0)])
        .unwrap();
    w.commit().unwrap();
    too_many(w.add_explicit_documents(vec![doc(&f, 3)]));
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: i64::MAX,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    w.force_merge(1).unwrap();
    w.add_explicit_documents(vec![doc(&f, 3)]).unwrap();
    assert_eq!(counts(&mut w), (3, 0));

    w.set_max_docs(usize::MAX);
    assert_eq!(w.max_docs, crate::index_writer::MAX_DOCS);
    check(&dir);
}

/// `mergeSortedField`/`mergeSortedSetField`: a term only deleted
/// documents used leaves the merged dictionary, for SORTED as for
/// SORTED_SET -- Java's `CheckIndex` refuses a dictionary with holes.
#[test]
fn merges_drop_terms_only_deleted_documents_used() {
    let tmp = TempDir::new("explicit-dead-terms");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    let kw = w
        .register_field(FieldInfo::new("kw", 0).with_doc_values(
            DocValuesType::Sorted,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
    for i in 0..4 {
        let mut d = doc(&f, i);
        d.fields.doc_values.push(StoredField {
            field_number: kw,
            value: FieldValue::Binary(format!("k{i}").into_bytes()),
        });
        w.add_explicit_documents(vec![d]).unwrap();
    }
    w.commit().unwrap();
    // d0 alone carries `k0` and `tag` t0 (d3 has t0 too); d1 alone `k1`.
    w.update_explicit_documents(id_term(0), Vec::new()).unwrap();
    w.update_explicit_documents(id_term(1), Vec::new()).unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (2, 0));
    check(&dir);
}

/// Doc-values skip indexes are written at flush, at merge and in a
/// doc-values update's new generation.
#[test]
fn skip_indexed_fields_are_written_everywhere_a_column_is() {
    let tmp = TempDir::new("explicit-skip-index");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    let rank = w
        .register_field(FieldInfo::new("rank", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::Range,
            -1,
        ))
        .unwrap();
    // `verifySameDocValuesSkipIndex`, and no skip index without numbers
    // or ordinals to summarize.
    for bad in [
        FieldInfo::new("rank", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ),
        FieldInfo::new("blob", 0).with_doc_values(
            DocValuesType::Binary,
            DocValuesSkipIndexType::Range,
            -1,
        ),
    ] {
        assert!(matches!(w.register_field(bad), Err(Error::Explicit(_))));
    }
    let ranked = |i: i64| {
        let mut d = doc(&f, i);
        d.fields.doc_values.push(StoredField {
            field_number: rank,
            value: FieldValue::Long(10 + i),
        });
        d
    };
    let skipper = |w: &mut IndexWriter<'_>| {
        let infos = w.commit().unwrap().clone();
        let sci = &infos.segments[0];
        let files: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|n| {
                n.starts_with(&format!("{}.", sci.segment_name))
                    || n.starts_with(&format!("{}_", sci.segment_name))
            })
            .collect();
        let fields = crate::field_updates::read_current_field_infos(&dir, sci, &files).unwrap();
        let at = fields.fields.iter().position(|f| f.name == "rank").unwrap();
        let per_field = crate::field_updates::per_field_component(
            &fields.fields[at],
            &crate::index_writer::per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        );
        let (meta, _) =
            crate::field_updates::read_current_column(&dir, sci, &files, &fields, at, &per_field)
                .unwrap()
                .unwrap();
        let s = *meta
            .skipper_meta(fields.fields[at].number)
            .expect("a skip index");
        (s.min_value, s.max_value, s.doc_count)
    };
    w.add_explicit_documents(vec![ranked(0), ranked(1)])
        .unwrap();
    assert_eq!(skipper(&mut w), (10, 11, 2), "flush");
    w.add_explicit_documents(vec![ranked(2)]).unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(skipper(&mut w), (10, 12, 3), "merge");
    let update = DocValuesUpdate::Numeric {
        term: id_term(1),
        field: "rank".to_string(),
        value: Some(99),
    };
    w.soft_update_explicit_documents(id_term(1), Vec::new(), &[update])
        .unwrap();
    assert_eq!(skipper(&mut w), (10, 99, 3), "update generation");
    check(&dir);
}

/// `findForcedDeletesMerges`: a segment is expunged only when more than
/// `forceMergeDeletesPctAllowed` (10%) of it can be reclaimed.
#[test]
fn force_merge_deletes_leaves_segments_under_the_threshold() {
    let tmp = TempDir::new("explicit-expunge-pct");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    let docs: Vec<ExplicitDocument> = (0..20).map(|i| doc(&f, i)).collect();
    w.add_explicit_documents(docs).unwrap();
    w.commit().unwrap();
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: i64::MAX,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    // 2 of 20 is exactly 10%: not above it.
    for i in 0..2 {
        w.soft_update_explicit_documents(id_term(i), Vec::new(), &[soft_delete(i)])
            .unwrap();
    }
    w.commit().unwrap();
    let before = w.commit_generations();
    w.force_merge_deletes().unwrap();
    assert_eq!(w.commit_generations(), before);
    w.soft_update_explicit_documents(id_term(2), Vec::new(), &[soft_delete(2)])
        .unwrap();
    w.commit().unwrap();
    w.force_merge_deletes().unwrap();
    assert_eq!(counts(&mut w), (17, 0), "3 of 20 is above it");
    check(&dir);
}

/// The latest commit's segments, each with its parsed `.si`.
fn committed_infos(dir: &FsDirectory) -> Vec<crate::segment_info::SegmentInfo> {
    crate::segment_infos::read_latest(dir)
        .unwrap()
        .segments
        .iter()
        .map(|sci| {
            let si = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            crate::segment_info::parse(&si, &sci.segment_id).unwrap()
        })
        .collect()
}

/// Four committed segments of 1, 2, 3 and 4 documents.
fn four_segments(w: &mut IndexWriter<'_>, f: &Fields) {
    let mut next = 0;
    for n in 1..=4 {
        let docs: Vec<ExplicitDocument> = (next..next + n).map(|i| doc(f, i)).collect();
        next += n;
        w.add_explicit_documents(docs).unwrap();
        w.commit().unwrap();
    }
}

/// With a merge policy set, `forceMergeDeletes` is the policy's
/// `findForcedDeletesMerges`, with its `forceMergeDeletesPctAllowed`.
#[test]
fn force_merge_deletes_takes_the_policys_pct_allowed() {
    let tmp = TempDir::new("explicit-expunge-policy-pct");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_merge_policy(Some(crate::merge_policy::MergePolicyConfig {
        force_merge_deletes_pct_allowed: 20.0,
        ..Default::default()
    }));
    let f = register(&mut w);
    let docs: Vec<ExplicitDocument> = (0..20).map(|i| doc(&f, i)).collect();
    w.add_explicit_documents(docs).unwrap();
    w.commit().unwrap();
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: i64::MAX,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    for i in 0..3 {
        w.soft_update_explicit_documents(id_term(i), Vec::new(), &[soft_delete(i)])
            .unwrap();
    }
    w.commit().unwrap();
    let before = w.commit_generations();
    w.force_merge_deletes().unwrap();
    assert_eq!(w.commit_generations(), before, "3 of 20 is not above 20%");
    for i in 3..5 {
        w.soft_update_explicit_documents(id_term(i), Vec::new(), &[soft_delete(i)])
            .unwrap();
    }
    w.commit().unwrap();
    w.force_merge_deletes().unwrap();
    assert_eq!(counts(&mut w), (15, 0), "5 of 20 is above it");
    check(&dir);
}

/// `ShuffleForcedMergePolicy`: a forced merge takes its sources
/// interleaved and says so in the merged segment's diagnostics; without
/// it the merged segment carries no such key.
#[test]
fn a_shuffled_forced_merge_is_marked_in_its_diagnostics() {
    for shuffle in [false, true] {
        let tmp = TempDir::new("explicit-shuffle");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_merge_policy(Some(crate::merge_policy::MergePolicyConfig::default()));
        w.set_shuffle_forced_merges(shuffle);
        assert_eq!(w.shuffle_forced_merges(), shuffle);
        let f = register(&mut w);
        four_segments(&mut w, &f);
        w.force_merge(1).unwrap();
        let infos = committed_infos(&dir);
        assert_eq!(infos.len(), 1);
        let marked = infos[0]
            .diagnostics
            .iter()
            .any(|(k, v)| k == crate::merge::SHUFFLE_MERGE_KEY && v.is_empty());
        assert_eq!(marked, shuffle);
        assert_eq!(w.committed_doc_count().unwrap(), 10);
        check(&dir);
    }
}

/// The merge policy's `noCFSRatio` decides whether a merged segment is
/// compound, whatever `use_compound_file` (flushed segments only) says.
#[test]
fn the_merge_compound_file_settings_decide_merged_segments() {
    for (ratio, compound) in [(1.0, true), (0.0, false)] {
        let tmp = TempDir::new("explicit-merge-cfs");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_merge_policy(Some(crate::merge_policy::MergePolicyConfig::default()));
        w.set_use_compound_file(!compound);
        let settings = crate::merge_policy::CompoundFileSettings::new(ratio, i64::MAX);
        w.set_merge_compound_file_settings(Some(settings));
        assert_eq!(w.merge_compound_file_settings(), Some(settings));
        let f = register(&mut w);
        four_segments(&mut w, &f);
        assert!(committed_infos(&dir)
            .iter()
            .all(|si| si.is_compound_file == !compound));
        w.force_merge(1).unwrap();
        let infos = committed_infos(&dir);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].is_compound_file, compound, "noCFSRatio {ratio}");
        check(&dir);
    }
}

/// A soft-deleted document with no sequence number is never retained.
#[test]
fn a_soft_deleted_document_without_a_seq_no_is_not_retained() {
    let tmp = TempDir::new("explicit-retention-noseq");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    let mut tombstone = doc(&f, 7);
    tombstone
        .fields
        .doc_values
        .retain(|v| v.field_number != f.seq);
    tombstone.fields.doc_values.push(StoredField {
        field_number: f.soft,
        value: FieldValue::Long(1),
    });
    w.add_explicit_documents(vec![doc(&f, 1), tombstone])
        .unwrap();
    w.commit().unwrap();
    w.add_explicit_documents(vec![doc(&f, 2)]).unwrap();
    assert_eq!(counts(&mut w), (3, 1));
    w.set_soft_deletes_retention(Some(SoftDeletesRetention {
        seq_no_field: "_seq_no".to_string(),
        min_retained_seq_no: 0,
        prune_postings_field: None,
        prune_recovery_source_field: None,
        seq_no_sentinel: Vec::new(),
    }));
    w.force_merge(1).unwrap();
    assert_eq!(counts(&mut w), (2, 0));
    check(&dir);
}

/// OpenSearch's `CombinedDeletionPolicy` decides which commits live, and a
/// reader's hold keeps a dropped commit's segment files on disk.
#[test]
fn caller_driven_commit_deletion_respects_holds() {
    let tmp = TempDir::new("explicit-holds");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_deletion_policy(DeletionPolicy::KeepAll).unwrap();
    let f = register(&mut w);
    w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
    let first = w.commit().unwrap().generation;
    w.add_explicit_documents(vec![doc(&f, 1)]).unwrap();
    let second = w.commit().unwrap().generation;
    assert_eq!(w.commit_generations(), [first, second]);

    let held = w.hold_commit(first).unwrap();
    assert!(!held.is_empty());
    assert!(held.iter().all(|f| !f.starts_with("segments")));
    w.force_merge(1).unwrap();
    let merged = w.commit_generations();
    assert_eq!(merged.len(), 3);
    let newest = *merged.last().unwrap();
    w.delete_commits(&merged).unwrap();
    assert_eq!(
        w.commit_generations(),
        [newest],
        "the newest always survives"
    );
    let on_disk = dir.list_all().unwrap();
    let first_name = lucene_store::directory::segments_file_name(first).unwrap();
    assert!(!on_disk.contains(&first_name));
    assert!(held.iter().all(|f| on_disk.contains(f)), "held files stay");
    check(&dir);

    w.release_files(&held).unwrap();
    let on_disk = dir.list_all().unwrap();
    assert!(
        held.iter().all(|f| !on_disk.contains(f)),
        "released files go"
    );
    check(&dir);

    w.delete_commits(&[newest, 12345]).unwrap();
    assert_eq!(w.commit_generations(), [newest]);
    assert!(w.hold_commit(999).is_err(), "no such commit");
}

/// A merged segment numbers its fields in the order its sources declare
/// them, not the writer's: a later soft delete must still land on the
/// soft-deletes field, found by name -- not on whichever field of the
/// merged segment happens to carry the writer's number for it.
#[test]
fn a_soft_delete_lands_in_a_merged_sparse_segment() {
    let tmp = TempDir::new("explicit-renumbered");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    // A first segment with only `_id` and `_seq_no`; the merge keeps the
    // writer's numbers rather than numbering those two first, as Java's
    // global field numbers do.
    let mut sparse = doc(&f, 0);
    sparse.stored.retain(|s| s.field_number == f.id);
    sparse.fields.inverted.retain(|i| i.field_number == f.id);
    sparse.fields.doc_values.retain(|v| v.field_number == f.seq);
    sparse.fields.points.clear();
    w.add_explicit_documents(vec![sparse]).unwrap();
    w.commit().unwrap();
    for i in 1..4 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    let merged = w.commit().unwrap().segments[0].clone();
    let renumbered = segment_fields(&dir, &merged);
    assert_eq!(
        renumbered
            .iter()
            .find(|x| x.name == "_seq_no")
            .unwrap()
            .number,
        f.seq,
        "the merged segment keeps the writer's field numbers"
    );

    w.soft_update_explicit_documents(id_term(2), vec![doc_at(&f, 2, 100)], &[soft_delete(2)])
        .unwrap();
    let infos = w.commit().unwrap().clone();
    check(&dir);
    let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
    assert_eq!(soft, 1);
    let seg = infos
        .segments
        .iter()
        .find(|s| s.segment_name == merged.segment_name)
        .unwrap();
    let fnm = dir
        .open(&crate::field_updates::field_infos_gen_file_name(
            &seg.segment_name,
            seg.field_infos_gen,
        ))
        .unwrap();
    let current = lucene_codecs::field_infos::parse(
        &fnm,
        &seg.segment_id,
        &lucene_util::base36::to_base36(seg.field_infos_gen),
    )
    .unwrap();
    for field in &current.fields {
        let expected = renumbered.iter().find(|x| x.name == field.name);
        match expected {
            Some(before) => assert_eq!(
                field.doc_values_type, before.doc_values_type,
                "{} keeps its doc-values type",
                field.name
            ),
            None => assert_eq!(field.name, "__soft_deletes", "only the soft field is new"),
        }
    }
}

/// A soft delete naming a field the writer never registered is refused,
/// rather than resolving against nothing and silently leaving the old
/// version live.
#[test]
fn a_soft_update_of_an_unregistered_field_is_refused() {
    let tmp = TempDir::new("explicit-unregistered-soft");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    let one = |i: &str| ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: id,
                terms: vec![term_docs(i)].into(),
                norm: None,
            }],
            doc_values: Vec::new(),
            points: Vec::new(),
            registered: Vec::new(),
            term_vectors: Vec::new(),
        },
    };
    w.add_explicit_documents(vec![one("a")]).unwrap();
    let soft = DocValuesUpdate::Numeric {
        term: id_term(0),
        field: "__soft_deletes".to_string(),
        value: Some(1),
    };
    assert!(matches!(
        w.soft_update_explicit_documents(id_term(0), vec![one("a")], std::slice::from_ref(&soft)),
        Err(Error::UnknownDocValuesUpdateField(ref f)) if f == "__soft_deletes"
    ));
    // Registered, the same update soft-deletes the buffered document.
    w.register_field(
        FieldInfo::new("__soft_deletes", 0)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
            .with_soft_deletes_field(true),
    )
    .unwrap();
    let soft = DocValuesUpdate::Numeric {
        term: Term {
            field: "_id".to_string(),
            bytes: b"a".to_vec(),
        },
        field: "__soft_deletes".to_string(),
        value: Some(1),
    };
    w.soft_update_explicit_documents(
        Term {
            field: "_id".to_string(),
            bytes: b"a".to_vec(),
        },
        vec![one("a")],
        &[soft],
    )
    .unwrap();
    let infos = w.commit().unwrap().clone();
    let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
    assert_eq!(soft, 1, "the buffered version is soft-deleted");
    check(&dir);
}

/// Document `i` of [`doc`] whose `body` term "beta" (positions 1 and 3)
/// carries a payload at its first occurrence and none at its second --
/// what a payload-producing analyzer such as Lucene's `MockAnalyzer`
/// hands `PayloadAttribute`.
fn doc_with_payload(f: &Fields, i: i64) -> ExplicitDocument {
    let mut d = doc(f, i);
    let body = d
        .fields
        .inverted
        .iter_mut()
        .find(|inv| inv.field_number == f.body)
        .unwrap();
    body.terms.edit(|terms| {
        let beta = terms.iter_mut().find(|t| t.term == b"beta").unwrap();
        beta.payloads = vec![format!("p{i}").into_bytes(), Vec::new()];
    });
    d
}

/// The payloads of `body:beta`, occurrence by occurrence, in every
/// document of `sci` that has it.
fn beta_payloads(dir: &FsDirectory, sci: &SegmentCommitInfo, max_doc: i32) -> Vec<Vec<u8>> {
    let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
    let infos = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
    crate::index_writer::tests::read_occurrences(dir, sci, &infos, "body", b"beta", max_doc)
        .into_iter()
        .flat_map(|occurrences| occurrences.into_iter().map(|o| o.payload))
        .collect()
}

fn stores_payloads(dir: &FsDirectory, sci: &SegmentCommitInfo, name: &str) -> bool {
    segment_fields(dir, sci)
        .iter()
        .find(|f| f.name == name)
        .unwrap()
        .store_payloads
}

/// A document whose analyzer produced payloads is indexed with them.
/// Refusing it would make OpenSearch record the refused primary operation
/// as a no-op tombstone and lose the document. The
/// field stores payloads in exactly the segments one of whose documents
/// gave it one, as `FieldInfo.setStorePayloads` does, the bytes read back
/// through the postings, and a merge keeps them.
#[test]
fn explicit_payloads_flush_merge_and_read_back() {
    let tmp = TempDir::new("explicit-payloads");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    // Documents 0 and 2 carry payloads, 1 does not.
    for i in 0..3 {
        let d = if i % 2 == 0 {
            doc_with_payload(&f, i)
        } else {
            doc(&f, i)
        };
        w.add_explicit_documents(vec![d]).unwrap();
    }
    w.commit().unwrap();
    // A segment none of whose documents has a payload.
    for i in 3..5 {
        w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
    }
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 2);
    check(&dir);

    let (with, without) = (&infos.segments[0], &infos.segments[1]);
    assert!(stores_payloads(&dir, with, "body"));
    assert!(
        !stores_payloads(&dir, with, "_id"),
        "only the field with one"
    );
    assert!(!stores_payloads(&dir, without, "body"));
    let none = Vec::new();
    assert_eq!(
        beta_payloads(&dir, with, 3),
        [
            b"p0".to_vec(),
            none.clone(),
            none.clone(),
            none.clone(),
            b"p2".to_vec(),
            none.clone()
        ]
    );

    w.force_merge(1).unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    let merged = w.segment_infos().segments[0].clone();
    check(&dir);
    assert!(stores_payloads(&dir, &merged, "body"));
    let mut payloads = beta_payloads(&dir, &merged, 5);
    payloads.retain(|p| !p.is_empty());
    payloads.sort();
    assert_eq!(payloads, [b"p0".to_vec(), b"p2".to_vec()]);
}

/// Payloads are one per occurrence, and only on a field with positions --
/// a malformed document is refused whole, before it is buffered.
#[test]
fn explicit_payloads_are_validated() {
    let tmp = TempDir::new("explicit-payloads-invalid");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);

    let mut short = doc_with_payload(&f, 0);
    for inv in &mut short.fields.inverted {
        inv.terms.edit(|terms| {
            for t in terms.iter_mut().filter(|t| t.term == b"beta") {
                t.payloads.pop();
            }
        });
    }
    let err = w
        .add_explicit_documents(vec![short])
        .unwrap_err()
        .to_string();
    assert!(err.contains("one per occurrence"), "{err}");

    let mut no_positions = doc(&f, 1);
    for inv in &mut no_positions.fields.inverted {
        if inv.field_number == f.id {
            inv.terms.edit(|t| {
                t[0].payloads = vec![b"x".to_vec()];
            });
        }
    }
    let err = w
        .add_explicit_documents(vec![no_positions])
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot store payloads"), "{err}");
    assert_eq!(w.pending_doc_count(), 0);
}

#[test]
fn a_doc_values_update_creates_an_unknown_field_doc_values_only() {
    // Java's `updateDocValues`/`tryUpdateDocValue` build their update
    // with `verifyOrCreateDvOnlyField(name, type, false)`: a field no
    // document has carried is created, not indexed, with the update's
    // type. `updateNumericDocValue`/`updateBinaryDocValue` pass
    // `fieldMustExist` and still refuse it.
    use crate::document::{Document as Doc, Store, StringField};
    let tmp = TempDir::new("explicit-dv-only-update");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let mut d = Doc::new();
    d.add(StringField::new("id", "0", Store::No));
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
    let reader_infos = w.segment_infos().clone();
    let id = || Term::new("id", "0");
    let err = w.update_numeric_doc_value(id(), "n", 1).unwrap_err();
    assert!(
        matches!(err, Error::UnknownDocValuesUpdateField(_)),
        "{err}"
    );
    let err = w.update_binary_doc_value(id(), "b", b"x").unwrap_err();
    assert!(
        matches!(err, Error::UnknownDocValuesUpdateField(_)),
        "{err}"
    );
    let update = |field: &str| DocValuesUpdate::Numeric {
        term: id(),
        field: field.to_string(),
        value: Some(7),
    };
    w.update_doc_values(id(), &[update("n")]).unwrap();
    assert!(w
        .try_update_doc_value(&reader_infos, 0, &[update("m")])
        .unwrap()
        .is_some());
    w.commit().unwrap();
    let sci = w.segment_infos.segments[0].clone();
    let infos = super::super::segment_field_infos(&dir, &sci).unwrap();
    for name in ["n", "m"] {
        let info = infos.field_by_name(name).unwrap();
        assert_eq!(info.doc_values_type, DocValuesType::Numeric, "{name}");
        assert_eq!(info.index_options, IndexOptions::None, "{name}");
    }
}

/// Several threads index explicit documents through the concurrent
/// writer, each re-indexing every document it added as a soft update.
/// Each update's soft delete reaches its earlier copy whichever slot
/// holds it, a field registered mid-run is indexed, and the segments
/// pass `CheckIndex`.
#[test]
fn explicit_documents_index_from_several_threads() {
    const THREADS: i64 = 4;
    const PER_THREAD: i64 = 200;
    let tmp = TempDir::new("explicit-concurrent");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let f = register(&mut w);
    w.set_max_buffered_docs(37).unwrap();
    let w = crate::concurrent_writer::ConcurrentIndexWriter::new(w, 4).unwrap();
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let (w, f) = (&w, &f);
            scope.spawn(move || {
                for k in 0..PER_THREAD {
                    let i = t * PER_THREAD + k;
                    w.add_explicit_documents(vec![doc(f, i)]).unwrap();
                    w.soft_update_explicit_documents(
                        id_term(i),
                        vec![doc_at(f, i, 1_000 + i)],
                        &[soft_delete(i)],
                    )
                    .unwrap();
                    if t == 0 && k == PER_THREAD / 2 {
                        let extra = w
                            .register_field(
                                FieldInfo::new("late", 0)
                                    .with_index_options(IndexOptions::Docs)
                                    .with_omit_norms(true),
                            )
                            .unwrap();
                        let mut d = doc(f, 10_000);
                        d.fields.inverted.push(InvertedField {
                            field_number: extra,
                            terms: vec![term_docs("x")].into(),
                            norm: None,
                        });
                        w.add_explicit_documents(vec![d]).unwrap();
                    }
                }
            });
        }
    });
    w.commit().unwrap();
    let mut w = w.into_writer().unwrap();
    let docs = usize::try_from(THREADS * PER_THREAD).unwrap();
    assert_eq!(counts(&mut w), (2 * docs + 1, i32::try_from(docs).unwrap()));
    check(&dir);
    let late = w
        .segment_infos()
        .segments
        .iter()
        .filter(|sci| segment_fields(&dir, sci).iter().any(|fi| fi.name == "late"))
        .count();
    assert_eq!(late, 1, "one segment holds the late field");
}
