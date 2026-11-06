//! A term on the partition field, in segments that each hold one partition
//! (`SegmentPartitioning::field`): every query shape that prunes on it
//! returns what the same documents return in an unpartitioned index.

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm, SegmentPartitioning,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_boolean_query_multi_segment, BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn term(t: &str) -> InvertedTerm {
    InvertedTerm {
        term: t.as_bytes().to_vec(),
        freq: 1,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    }
}

/// Document `i` of tenant `p`: `tenant` "p{p}", `body` "b{i % 3}" and "all".
fn doc(p: usize, i: usize) -> ExplicitDocument {
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("p{p}-{i}")),
        }],
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: 0,
                    terms: vec![term(&format!("p{p}"))].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: 1,
                    terms: vec![term("all"), term(&format!("b{}", i % 3))].into(),
                    norm: None,
                },
            ],
            ..ExplicitFields::default()
        },
    }
}

/// 3 partitions of 40, 25 and 3 documents, interleaved; a pool for
/// partitions under 10 documents, so p2 and p0's and p1's first ones share
/// the default partition's segments.
fn index(dir: &FsDirectory, partitioning: Option<SegmentPartitioning>) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    for (n, name) in ["tenant", "body"].iter().enumerate() {
        w.register_field(
            FieldInfo::new(*name, n as i32)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    }
    w.set_segment_partitioning(partitioning);
    let w = ConcurrentIndexWriter::new(w, 8).unwrap();
    let sizes = [40, 25, 3];
    for i in 0..40 {
        for (p, n) in sizes.iter().enumerate() {
            if i < *n {
                let tenant = std::sync::Arc::from(format!("p{p}").as_str());
                w.add_explicit_documents_in(Some(tenant), vec![doc(p, i)])
                    .unwrap();
            }
        }
        if i % 13 == 12 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
}

fn t(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes()))
}

fn hits(dir: &FsDirectory, q: &BooleanQuery) -> usize {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    search_boolean_query_multi_segment(&segments, q, &norms, 1000)
        .unwrap()
        .len()
}

#[test]
fn pruned_searches_match_an_unpartitioned_index() {
    let plain = TempDir::new("pruning-plain");
    let plain_dir = FsDirectory::open(&plain);
    index(&plain_dir, None);
    let parted = TempDir::new("pruning-parted");
    let parted_dir = FsDirectory::open(&parted);
    index(
        &parted_dir,
        Some(SegmentPartitioning {
            pool_below_docs: 10,
            field: Some("tenant".to_string()),
            ..SegmentPartitioning::opensearch()
        }),
    );
    // The partitioned index has segments of their own for p0 and p1.
    let reader = DirectoryReader::open(&parted_dir).unwrap();
    let own = reader
        .segment_readers()
        .iter()
        .filter(|s| s.partition().is_some())
        .count();
    assert!(own >= 2, "{own} segments of one partition");

    let mut queries = Vec::new();
    for p in ["p0", "p1", "p2", "p9"] {
        // A filter beside a scoring clause.
        queries.push(BooleanQuery {
            must: vec![t("body", "b1")],
            filter: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
        // A lone filter: matches all of the partition.
        queries.push(BooleanQuery {
            filter: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
        // A filter with optional clauses only: they stay optional.
        queries.push(BooleanQuery {
            filter: vec![t("tenant", p)],
            should: vec![t("body", "b2")],
            ..BooleanQuery::default()
        });
        // An exclusion.
        queries.push(BooleanQuery {
            must: vec![t("body", "all")],
            must_not: vec![t("tenant", p)],
            ..BooleanQuery::default()
        });
    }
    for q in &queries {
        assert_eq!(hits(&parted_dir, q), hits(&plain_dir, q), "{q:?}");
    }
    assert_eq!(
        hits(
            &parted_dir,
            &BooleanQuery {
                filter: vec![t("tenant", "p0")],
                ..BooleanQuery::default()
            }
        ),
        40
    );
}
