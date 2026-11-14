#![allow(clippy::arithmetic_side_effects)]
//! A numeric doc-values range as a filter bit set (`filter_bitsets`, what a
//! filtered k-NN search starts from) read straight from the column, and its
//! zone-pruned scorer, match exactly the documents whose value is in range:
//! on an `Adaptive90` field with per-1024-document statistics (in document
//! order and shuffled, a partial last window), on a stock Lucene90 copy, and
//! with a skip index.

use lucene_codecs::doc_values::Lucene90DocValuesFormat;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::document::{Document, KnnFloatVectorField, NumericDocValuesField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{ExtendedQuery, NumericDocValuesRangeQuery};
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::vector_query::filter_bitsets;
use lucene_search::{BooleanQuery, Clause};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const DOCS: usize = 5_300;

fn value_of(layout: &str, i: usize) -> i64 {
    match layout {
        "ordered" => 1_000 + i as i64 * 7,
        // A permutation: 5_300 and 7 are coprime.
        _ => 1_000 + ((i * 2_399) % DOCS) as i64 * 7,
    }
}

fn build(dir: &FsDirectory, layout: &str, adaptive: bool, skip: bool) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    if adaptive {
        w.set_default_doc_values_format(Lucene90DocValuesFormat::rust_adaptive());
    }
    for i in 0..DOCS {
        let mut d = Document::new();
        d.add(
            KnnFloatVectorField::new(
                "v",
                vec![(i % 17) as f32, (i % 5) as f32],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        let v = value_of(layout, i);
        d.add(if skip {
            NumericDocValuesField::indexed_field("ts", v)
        } else {
            NumericDocValuesField::new("ts", v)
        });
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn range(lo: i64, hi: i64) -> Clause {
    Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(
        NumericDocValuesRangeQuery::new("ts", lo, hi),
    )))
}

#[test]
fn range_filters_match_the_values_on_every_layout() {
    let bounds = [
        (1_000, 1_000),
        (1_000, 1_000 + 7 * 1_023),
        (1_000 + 7 * 1_024, 1_000 + 7 * 2_047),
        (1_000 + 7 * 500, 1_000 + 7 * 3_000),
        (1_000 + 7 * 5_299, 1_000 + 7 * 5_299),
        (1_000 + 7 * 5_000, i64::MAX),
        (i64::MIN, 1_000 + 7 * 10),
        (0, 999),
        (2_000_000, 3_000_000),
        (5, 4),
    ];
    for layout in ["ordered", "shuffled"] {
        for (adaptive, skip) in [(true, false), (true, true), (false, false), (false, true)] {
            let tmp = TempDir::new("dense-range-bits");
            let dir = FsDirectory::open(tmp.path());
            build(&dir, layout, adaptive, skip);
            let reader = DirectoryReader::open(&dir).unwrap();
            let opened = reader.open_segments().unwrap();
            let segments = opened.as_open_segments();
            for &(lo, hi) in &bounds {
                let got = &filter_bitsets(&segments, &range(lo, hi)).unwrap()[0];
                let want = (0..DOCS)
                    .filter(|&i| (lo..=hi).contains(&value_of(layout, i)))
                    .count();
                assert_eq!(
                    got.cardinality(),
                    want,
                    "{layout} adaptive={adaptive} skip={skip} [{lo}, {hi}]"
                );
                // The scorer tree (a boolean wrapper skips the direct read)
                // answers with the same set.
                let wrapped = Clause::from(BooleanQuery::new().with_filter([range(lo, hi)]));
                let tree = &filter_bitsets(&segments, &wrapped).unwrap()[0];
                assert_eq!(
                    tree.cardinality(),
                    want,
                    "scorer {layout} adaptive={adaptive} skip={skip} [{lo}, {hi}]"
                );
            }
            // A filtered k-NN over the field answers from the same filter.
            let hits = knn_top_hits(
                &segments,
                "v",
                &KnnTarget::Float(vec![3.0, 1.0]),
                5,
                20,
                Some(&range(1_000 + 7 * 100, 1_000 + 7 * 2_000)),
            )
            .unwrap();
            assert_eq!(hits.len(), 5, "{layout} adaptive={adaptive} skip={skip}");
        }
    }
}
