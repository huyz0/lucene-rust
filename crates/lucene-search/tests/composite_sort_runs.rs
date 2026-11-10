//! Filters on the leading keys of a several-key index sort (a tenant, then
//! a sub-partition, then a time): matched as one run of doc ids, and every
//! query shape returns what the same documents return unsorted.

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey, SortedNumericSelector,
    SortedSetSelector, StringMissingValue,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{
    IndexOrDocValuesQuery, IndexSortSortedNumericDocValuesRangeQuery, NumericDocValuesRangeQuery,
};
use lucene_search::query::PointsRangeQuery;
use lucene_search::{search_boolean_query_multi_segment, BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

const DOCS: usize = 420;

const TENANT: i32 = 0;
const SUB: i32 = 1;
const TS: i32 = 2;
const BODY: i32 = 3;
const ID: i32 = 4;
const TS_MULTI: i32 = 5;

fn term(t: &[u8]) -> InvertedTerm {
    InvertedTerm {
        term: t.to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

/// Document `i`'s tenant(s): one of 4, interleaved; document 77 has two.
fn tenants(i: usize) -> Vec<String> {
    match i {
        77 => vec!["t0".to_string(), "t2".to_string()],
        _ => vec![format!("t{}", (i * 7) % 4)],
    }
}

/// Document `i`'s sub-partition, one of 3; every 13th has none.
fn sub(i: usize) -> Option<String> {
    (i % 13 != 12).then(|| format!("s{}", (i * 5) % 3))
}

/// Document `i`'s time; every 11th has none.
fn ts(i: usize) -> Option<i64> {
    (i % 11 != 10).then(|| ((i * 37) % 50) as i64 * 10 - 100)
}

fn doc(i: usize) -> ExplicitDocument {
    let tenants = tenants(i);
    let mut inverted = vec![
        InvertedField {
            field_number: BODY,
            terms: vec![term(b"all"), term(format!("b{}", i % 3).as_bytes())].into(),
            norm: None,
        },
        InvertedField {
            field_number: TENANT,
            terms: tenants
                .iter()
                .map(|t| term(t.as_bytes()))
                .collect::<Vec<_>>()
                .into(),
            norm: None,
        },
    ];
    let mut doc_values: Vec<StoredField> = tenants
        .iter()
        .map(|t| StoredField {
            field_number: TENANT,
            value: FieldValue::Binary(t.as_bytes().to_vec()),
        })
        .collect();
    if let Some(s) = sub(i) {
        inverted.push(InvertedField {
            field_number: SUB,
            terms: vec![term(s.as_bytes())].into(),
            norm: None,
        });
        doc_values.push(StoredField {
            field_number: SUB,
            value: FieldValue::Binary(s.into_bytes()),
        });
    }
    let mut points = Vec::new();
    if let Some(v) = ts(i) {
        doc_values.push(StoredField {
            field_number: TS,
            value: FieldValue::Long(v),
        });
        let mut packed = vec![0u8; 8];
        lucene_util::numeric_utils::long_to_sortable_bytes(v, &mut packed, 0);
        points.push(StoredField {
            field_number: TS,
            value: FieldValue::Binary(packed),
        });
        for x in [v, v + 1000 * (i % 2) as i64] {
            doc_values.push(StoredField {
                field_number: TS_MULTI,
                value: FieldValue::Long(x),
            });
        }
    }
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: ID,
            value: FieldValue::Int(i as i32),
        }],
        fields: ExplicitFields {
            inverted,
            doc_values,
            points,
            ..ExplicitFields::default()
        },
    }
}

/// `tenant` a SORTED_SET column, `sub` a SORTED one.
fn keyword(field: &str) -> IndexSortField {
    IndexSortField {
        field: field.to_string(),
        reverse: false,
        kind: if field == "sub" {
            IndexSortKind::String(StringMissingValue::None)
        } else {
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::Min,
                missing: StringMissingValue::None,
            }
        },
    }
}

fn index(dir: &FsDirectory, sort: Option<&[IndexSortField]>) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let keyword_field = |name: &str, number, dv| {
        FieldInfo::new(name, number)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)
            .with_doc_values(dv, DocValuesSkipIndexType::None, -1)
    };
    w.register_field(keyword_field("tenant", TENANT, DocValuesType::SortedSet))
        .unwrap();
    w.register_field(keyword_field("sub", SUB, DocValuesType::Sorted))
        .unwrap();
    w.register_field(
        FieldInfo::new("ts", TS)
            .with_points(1, 1, 8)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1),
    )
    .unwrap();
    w.register_field(
        FieldInfo::new("body", BODY)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    w.register_field(FieldInfo::new("id", ID)).unwrap();
    w.register_field(FieldInfo::new("ts_multi", TS_MULTI).with_doc_values(
        DocValuesType::SortedNumeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.set_index_sort(sort).unwrap();
    for i in 0..DOCS {
        w.add_explicit_documents(vec![doc(i)]).unwrap();
        if i % 150 == 149 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
}

fn t(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes()))
}

fn points(field: &str, min: i64, max: i64) -> Clause {
    Clause::PointsRange(PointsRangeQuery::new(field, min, max))
}

/// Each hit's document number and score, by document number.
fn hits(dir: &FsDirectory, q: &BooleanQuery) -> Vec<(i32, u32)> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut ids = Vec::new();
    for seg in reader.segment_readers() {
        for doc in 0..seg.max_doc {
            match seg.stored_document(doc).unwrap().unwrap().fields[0].value {
                FieldValue::Int(id) => ids.push(id),
                ref o => panic!("{o:?}"),
            }
        }
    }
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let mut out: Vec<(i32, u32)> = search_boolean_query_multi_segment(&segments, q, &norms, 1000)
        .unwrap()
        .iter()
        .map(|h| (ids[h.doc_id as usize], h.score.to_bits()))
        .collect();
    out.sort_unstable();
    out
}

/// The filter shapes a tenant, sub-partition and time conjunction comes in.
fn queries() -> Vec<BooleanQuery> {
    let mut filters: Vec<Vec<Clause>> = Vec::new();
    for tenant in ["t0", "t1", "t2", "t9"] {
        filters.push(vec![t("tenant", tenant)]);
        for s in ["s0", "s1", "s2", "s7"] {
            filters.push(vec![t("sub", s), t("tenant", tenant)]);
            filters.push(vec![t("tenant", tenant), t("sub", s), points("ts", 0, 200)]);
            filters.push(vec![t("tenant", tenant), t("sub", s), points("ts", 90, 90)]);
            filters.push(vec![
                t("tenant", tenant),
                t("sub", s),
                points("ts", 300, 100),
            ]);
        }
        filters.push(vec![t("tenant", tenant), points("ts", -50, 150)]);
        filters.push(vec![t("tenant", tenant), points("ts", 0, 0)]);
        filters.push(vec![
            t("tenant", tenant),
            points("ts", -100, 100),
            t("sub", "s1"),
        ]);
        filters.push(vec![
            t("tenant", tenant),
            NumericDocValuesRangeQuery::new("ts", 20, 260).into(),
        ]);
        filters.push(vec![
            t("tenant", tenant),
            IndexSortSortedNumericDocValuesRangeQuery::new("ts", 20, 260, points("ts", 20, 260))
                .into(),
        ]);
        filters.push(vec![
            t("tenant", tenant),
            IndexOrDocValuesQuery::new(
                points("ts", 40, 300),
                NumericDocValuesRangeQuery::new("ts", 40, 300),
            )
            .into(),
        ]);
        filters.push(vec![
            t("tenant", tenant),
            NumericDocValuesRangeQuery::new("ts_multi", 0, 1200).into(),
        ]);
    }
    filters.push(vec![points("ts", 0, 200)]);
    filters.push(vec![points("ts", 0, 200), t("sub", "s1")]);
    let mut out = Vec::new();
    for f in filters {
        out.push(BooleanQuery {
            filter: f.clone(),
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            must: vec![t("body", "b1")],
            filter: f.clone(),
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            should: vec![t("body", "b0"), t("body", "b2")],
            filter: f.clone(),
            minimum_should_match: 1,
            ..BooleanQuery::default()
        });
        // Required clauses of a boolean built without scores.
        out.push(BooleanQuery {
            must: vec![t("body", "all")],
            filter: vec![Clause::Boolean(Box::new(BooleanQuery {
                must: f.clone(),
                ..BooleanQuery::default()
            }))],
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            must: vec![t("body", "all")],
            filter: vec![Clause::Boolean(Box::new(BooleanQuery {
                must: f[..1].to_vec(),
                filter: f[1..].to_vec(),
                ..BooleanQuery::default()
            }))],
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            must: vec![t("body", "all")],
            must_not: vec![Clause::Boolean(Box::new(BooleanQuery {
                filter: f,
                ..BooleanQuery::default()
            }))],
            ..BooleanQuery::default()
        });
    }
    out
}

fn check(name: &str, sort: &[IndexSortField]) {
    let plain = TempDir::new(&format!("composite-plain-{name}"));
    let plain_dir = FsDirectory::open(&plain);
    index(&plain_dir, None);
    let sorted = TempDir::new(&format!("composite-sorted-{name}"));
    let sorted_dir = FsDirectory::open(&sorted);
    index(&sorted_dir, Some(sort));
    let reader = DirectoryReader::open(&sorted_dir).unwrap();
    assert!(reader.segment_readers().len() >= 2);
    let mut matched = 0;
    for q in &queries() {
        let expected = hits(&plain_dir, q);
        matched += expected.len();
        assert_eq!(hits(&sorted_dir, q), expected, "{name}: {q:?}");
    }
    assert!(matched > 0);
}

#[test]
fn tenant_then_sub_then_time() {
    check(
        "tst",
        &[
            keyword("tenant"),
            keyword("sub"),
            IndexSortField::long("ts", true, None),
        ],
    );
}

#[test]
fn tenant_then_time_then_sub() {
    let mut s = keyword("sub");
    s.reverse = true;
    s.kind = IndexSortKind::String(StringMissingValue::Last);
    check(
        "tts",
        &[
            keyword("tenant"),
            IndexSortField::long("ts", false, Some(-100)),
            s,
        ],
    );
}

#[test]
fn time_first() {
    check(
        "time",
        &[
            IndexSortField::long("ts", false, Some(i64::MAX)),
            keyword("sub"),
        ],
    );
}

#[test]
fn multi_valued_second_key() {
    check(
        "multi",
        &[
            keyword("tenant"),
            IndexSortField {
                field: "ts_multi".to_string(),
                reverse: false,
                kind: IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Long(None),
                    selector: SortedNumericSelector::Min,
                },
            },
            keyword("sub"),
        ],
    );
}
