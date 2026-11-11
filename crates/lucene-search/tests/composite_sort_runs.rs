//! Filters on the leading keys of a several-key index sort (a tenant, then
//! a sub-partition, then a time): matched as one run of doc ids, and every
//! query shape returns what the same documents return unsorted -- in
//! segments of document blocks too, where the run is of whole blocks. A
//! search sorted by the keys after the pinned ones ends each segment early
//! and finds the hits a full collection finds.

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
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
use lucene_search::{
    count_boolean_query_segment, search_boolean_query_multi_segment, BooleanQuery, Clause,
    TermQuery,
};
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

/// How the documents are laid out.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// One document each; document 77 has two tenants.
    Flat,
    /// Each document the parent of a block of up to two children, which
    /// hold only a body term and an id.
    Blocks,
    /// [`Shape::Blocks`], some children also holding a `sub` value.
    BlocksChildSub,
}

/// Document `i`'s tenant(s): one of 4, interleaved; document 77 has two
/// when the documents are flat.
fn tenants(i: usize, shape: Shape) -> Vec<String> {
    match i {
        77 if shape == Shape::Flat => vec!["t0".to_string(), "t2".to_string()],
        _ => vec![format!("t{}", (i * 7) % 4)],
    }
}

/// Document `i`'s children, before it in its block.
fn children(i: usize, shape: Shape) -> Vec<ExplicitDocument> {
    if shape == Shape::Flat {
        return Vec::new();
    }
    (0..i % 3)
        .map(|c| {
            let mut doc_values = Vec::new();
            let mut inverted = vec![InvertedField {
                field_number: BODY,
                terms: vec![term(b"child"), term(format!("b{}", (i + c) % 3).as_bytes())].into(),
                norm: None,
            }];
            if shape == Shape::BlocksChildSub && i.is_multiple_of(17) {
                inverted.push(InvertedField {
                    field_number: SUB,
                    terms: vec![term(b"s1")].into(),
                    norm: None,
                });
                doc_values.push(StoredField {
                    field_number: SUB,
                    value: FieldValue::Binary(b"s1".to_vec()),
                });
            }
            ExplicitDocument {
                stored: vec![StoredField {
                    field_number: ID,
                    value: FieldValue::Int((10_000 + i * 10 + c) as i32),
                }],
                fields: ExplicitFields {
                    inverted,
                    doc_values,
                    ..ExplicitFields::default()
                },
            }
        })
        .collect()
}

/// Document `i`'s sub-partition, one of 3; every 13th has none.
fn sub(i: usize) -> Option<String> {
    (i % 13 != 12).then(|| format!("s{}", (i * 5) % 3))
}

/// Document `i`'s time; every 11th has none.
fn ts(i: usize) -> Option<i64> {
    (i % 11 != 10).then(|| ((i * 37) % 50) as i64 * 10 - 100)
}

fn doc(i: usize, shape: Shape) -> ExplicitDocument {
    let tenants = tenants(i, shape);
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

fn index(dir: &FsDirectory, sort: Option<&[IndexSortField]>, shape: Shape) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    if shape != Shape::Flat {
        w.set_parent_field(Some("_parent")).unwrap();
    }
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
        let mut block = children(i, shape);
        block.push(doc(i, shape));
        w.add_explicit_documents(block).unwrap();
        if i % 150 == 149 {
            w.commit().unwrap();
        }
        if i == 300 {
            // Deleted documents inside the runs.
            w.delete_documents_by_term(&[Term::new("body", b"b2".as_slice())])
                .unwrap();
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

/// How many live documents `q` matches, counted without scores: the top
/// boolean's own required clauses fold into the run.
fn count(dir: &FsDirectory, q: &BooleanQuery) -> u64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    opened
        .as_open_segments()
        .iter()
        .map(|seg| count_boolean_query_segment(seg, q).unwrap())
        .sum()
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
        // Counted without scores, `MUST` clauses alone and beside others.
        out.push(BooleanQuery {
            must: f.clone(),
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            must: f[..1].to_vec(),
            filter: f[1..].to_vec(),
            ..BooleanQuery::default()
        });
        out.push(BooleanQuery {
            must: vec![t("body", "b1")].into_iter().chain(f.clone()).collect(),
            ..BooleanQuery::default()
        });
        // Children (in blocks) or the filter's documents.
        out.push(BooleanQuery {
            should: vec![
                t("body", "child"),
                Clause::Boolean(Box::new(BooleanQuery {
                    filter: f.clone(),
                    ..BooleanQuery::default()
                })),
            ],
            minimum_should_match: 1,
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
    check_shape(name, sort, Shape::Flat);
}

fn check_shape(name: &str, sort: &[IndexSortField], shape: Shape) {
    let plain = TempDir::new(&format!("composite-plain-{name}"));
    let plain_dir = FsDirectory::open(&plain);
    index(&plain_dir, None, shape);
    let sorted = TempDir::new(&format!("composite-sorted-{name}"));
    let sorted_dir = FsDirectory::open(&sorted);
    index(&sorted_dir, Some(sort), shape);
    let reader = DirectoryReader::open(&sorted_dir).unwrap();
    assert!(reader.segment_readers().len() >= 2);
    assert_eq!(
        reader.segment_readers().iter().all(|s| s.has_blocks()),
        shape != Shape::Flat
    );
    let mut matched = 0;
    for q in &queries() {
        let expected = hits(&plain_dir, q);
        matched += expected.len();
        assert_eq!(hits(&sorted_dir, q), expected, "{name}: {q:?}");
        assert_eq!(
            count(&sorted_dir, q),
            expected.len() as u64,
            "{name} count: {q:?}"
        );
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

fn tenant_sub_time() -> Vec<IndexSortField> {
    vec![
        keyword("tenant"),
        keyword("sub"),
        IndexSortField::long("ts", true, None),
    ]
}

#[test]
fn blocks_tenant_then_sub_then_time() {
    check_shape("btst", &tenant_sub_time(), Shape::Blocks);
}

#[test]
fn blocks_with_a_key_on_children() {
    check_shape("bchild", &tenant_sub_time(), Shape::BlocksChildSub);
}

#[test]
fn blocks_tenant_then_time_then_sub() {
    check_shape(
        "btts",
        &[
            keyword("tenant"),
            IndexSortField::long("ts", false, Some(-100)),
            keyword("sub"),
        ],
        Shape::Blocks,
    );
}

/// The leading sort keys a query pins in each segment of an index sorted
/// tenant, sub, time.
fn pinned(shape: Shape, q: &BooleanQuery) -> Vec<usize> {
    let dir = TempDir::new("composite-pinned");
    let fs = FsDirectory::open(&dir);
    index(&fs, Some(&tenant_sub_time()), shape);
    let reader = DirectoryReader::open(&fs).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    opened
        .as_open_segments()
        .iter()
        .map(|seg| lucene_search::pinned_sort_keys(seg, q).unwrap())
        .collect()
}

#[test]
fn pinned_sort_keys_follow_the_point_clauses() {
    let q = |filter: Vec<Clause>| BooleanQuery {
        must: vec![t("body", "all")],
        filter,
        ..BooleanQuery::default()
    };
    let tenant = q(vec![t("tenant", "t1")]);
    let both = q(vec![t("sub", "s2"), t("tenant", "t1")]);
    let nested = BooleanQuery {
        must: vec![Clause::Boolean(Box::new(both.clone()))],
        filter: vec![t("body", "b1")],
        ..BooleanQuery::default()
    };
    let ranged = q(vec![t("tenant", "t1"), points("ts", 0, 100)]);
    let point = q(vec![
        t("tenant", "t1"),
        t("sub", "s0"),
        points("ts", 90, 90),
    ]);
    // A document with two tenants: no key order inside the tenant's run in
    // its segment.
    let flat = pinned(Shape::Flat, &tenant);
    assert!(flat.contains(&0) && flat.contains(&1), "{flat:?}");
    for shape in [Shape::Blocks, Shape::BlocksChildSub] {
        assert!(pinned(shape, &tenant).iter().all(|&p| p == 1));
        assert!(pinned(shape, &ranged).iter().all(|&p| p == 1));
        assert!(pinned(shape, &q(vec![t("sub", "s2")]))
            .iter()
            .all(|&p| p == 0));
    }
    assert!(pinned(Shape::Blocks, &both).iter().all(|&p| p == 2));
    assert!(pinned(Shape::Blocks, &nested).iter().all(|&p| p == 2));
    assert!(pinned(Shape::Blocks, &point).iter().all(|&p| p == 3));
    // A child holds a `sub`: the sub key no longer orders the blocks.
    assert!(pinned(Shape::BlocksChildSub, &both).iter().all(|&p| p == 1));
}

/// A search sorted by the time after a pinned tenant and sub (and by the
/// sub and time after a pinned tenant) ends each segment early and finds
/// what a full collection finds.
#[test]
fn sorted_by_the_keys_after_pinned_ones() {
    use lucene_search::top_field::{search_sorted, SortField, SortType};
    let dir = TempDir::new("composite-early");
    let fs = FsDirectory::open(&dir);
    index(&fs, Some(&tenant_sub_time()), Shape::Blocks);
    let reader = DirectoryReader::open(&fs).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let plain = opened.as_open_segments();
    let norms = vec![None; plain.len()];
    let by_time = vec![SortField::numeric("ts", SortType::Long, true)];
    let by_sub_time = vec![SortField::string("sub", false), by_time[0].clone()];
    let mut early_hits = 0;
    for tenant in ["t0", "t1", "t3", "t9"] {
        for (sort, filter, pins) in [
            (&by_time, vec![t("tenant", tenant), t("sub", "s1")], 2),
            (&by_sub_time, vec![t("tenant", tenant)], 1),
        ] {
            for must in [vec![], vec![t("body", "b1")]] {
                let q = BooleanQuery {
                    must,
                    filter: filter.clone(),
                    ..BooleanQuery::default()
                };
                let mut early = opened.as_open_segments();
                for seg in &mut early {
                    seg.index_sort_prefix =
                        lucene_search::pinned_sort_keys(seg, &q).unwrap() >= pins;
                    assert!(seg.index_sort_prefix);
                }
                for (top_n, threshold) in [(1, 0), (3, 2), (10, 1000), (500, 0)] {
                    let full = search_sorted(
                        &plain,
                        reader.segment_readers(),
                        &q,
                        &norms,
                        sort,
                        top_n,
                        threshold,
                        None,
                    )
                    .unwrap();
                    let cut = search_sorted(
                        &early,
                        reader.segment_readers(),
                        &q,
                        &norms,
                        sort,
                        top_n,
                        threshold,
                        None,
                    )
                    .unwrap();
                    let key = |h: &lucene_search::top_field::FieldDoc| {
                        (h.doc, h.values.clone(), h.terms.clone())
                    };
                    assert_eq!(
                        cut.hits.iter().map(key).collect::<Vec<_>>(),
                        full.hits.iter().map(key).collect::<Vec<_>>(),
                        "{q:?} {top_n} {threshold}"
                    );
                    assert!(cut.total.value <= full.total.value);
                    early_hits += cut.hits.len();
                }
            }
        }
    }
    assert!(early_hits > 0);
}
