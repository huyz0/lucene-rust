//! A filter on the field a segment is sorted by first: its documents are one
//! run of doc ids, matched as that run, and every query shape returns what
//! the same documents return unsorted.

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, LuceneVersion, SortedSetSelector, StringMissingValue,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_boolean_query_multi_segment, BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn term(t: &[u8]) -> InvertedTerm {
    InvertedTerm {
        term: t.to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

/// Document `i`: `tenant` values `tenants` (postings and SORTED_SET doc
/// values), `body` "b{i % 3}" and "all".
fn doc(i: usize, tenants: &[String]) -> ExplicitDocument {
    let mut inverted = vec![InvertedField {
        field_number: 1,
        terms: vec![term(b"all"), term(format!("b{}", i % 3).as_bytes())].into(),
        norm: None,
    }];
    if !tenants.is_empty() {
        inverted.push(InvertedField {
            field_number: 0,
            terms: tenants
                .iter()
                .map(|t| term(t.as_bytes()))
                .collect::<Vec<_>>()
                .into(),
            norm: None,
        });
    }
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: 1,
            value: FieldValue::String(format!("d{i}")),
        }],
        fields: ExplicitFields {
            inverted,
            doc_values: tenants
                .iter()
                .map(|t| StoredField {
                    field_number: 0,
                    value: FieldValue::Binary(t.as_bytes().to_vec()),
                })
                .collect(),
            ..ExplicitFields::default()
        },
    }
}

/// Document `i`'s tenants: one of 5, interleaved; every 17th has none and
/// one has two, so `p1`'s documents are not one run in its segment.
fn tenants(i: usize) -> Vec<String> {
    match i {
        _ if i % 17 == 16 => Vec::new(),
        50 => vec!["p0".to_string(), "p1".to_string()],
        _ => vec![format!("p{}", (i * 7) % 5)],
    }
}

fn index(dir: &FsDirectory, sorted: bool) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new("tenant", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)
            .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1),
    )
    .unwrap();
    w.register_field(
        FieldInfo::new("body", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    if sorted {
        w.set_index_sort(Some(&[IndexSortField {
            field: "tenant".to_string(),
            reverse: false,
            kind: IndexSortKind::SortedSet {
                selector: SortedSetSelector::Min,
                missing: StringMissingValue::None,
            },
        }]))
        .unwrap();
    }
    for i in 0..300 {
        w.add_explicit_documents(vec![doc(i, &tenants(i))]).unwrap();
        if i % 120 == 119 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
}

fn t(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes()))
}

fn hits(dir: &FsDirectory, q: &BooleanQuery) -> Vec<f32> {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let mut scores: Vec<f32> = search_boolean_query_multi_segment(&segments, q, &norms, 1000)
        .unwrap()
        .iter()
        .map(|h| h.score)
        .collect();
    scores.sort_by(f32::total_cmp);
    scores
}

#[test]
fn filters_on_the_sort_field_match_an_unsorted_index() {
    let plain = TempDir::new("runs-plain");
    let plain_dir = FsDirectory::open(&plain);
    index(&plain_dir, false);
    let sorted = TempDir::new("runs-sorted");
    let sorted_dir = FsDirectory::open(&sorted);
    index(&sorted_dir, true);
    let reader = DirectoryReader::open(&sorted_dir).unwrap();
    assert!(reader.segment_readers().len() >= 2);
    assert!(reader
        .segment_readers()
        .iter()
        .all(|s| s.index_sort().is_some()));

    let mut queries = Vec::new();
    for p in ["p0", "p1", "p2", "p4", "p9"] {
        queries.push(BooleanQuery {
            must: vec![t("body", "b1")],
            filter: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
        queries.push(BooleanQuery {
            filter: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
        queries.push(BooleanQuery {
            filter: vec![t("tenant", p), t("body", "b2")],
            ..BooleanQuery::default()
        });
        queries.push(BooleanQuery {
            filter: vec![t("tenant", p)],
            should: vec![t("body", "b0")],
            ..BooleanQuery::default()
        });
        queries.push(BooleanQuery {
            must: vec![t("body", "all")],
            must_not: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
    }
    for q in &queries {
        assert_eq!(hits(&sorted_dir, q), hits(&plain_dir, q), "{q:?}");
    }
    let lone = |p: &str| BooleanQuery {
        filter: vec![t("tenant", p)],
        ..BooleanQuery::default()
    };
    let expected = |p: &str| {
        (0..300)
            .filter(|&i| tenants(i).iter().any(|x| x == p))
            .count()
    };
    for p in ["p0", "p1", "p3"] {
        assert_eq!(hits(&sorted_dir, &lone(p)).len(), expected(p), "{p}");
    }
}
