//! The plugin's `terms` aggregation behind a filter, in process: the shape
//! OpenSearch sends (`size: 0`, a filter, `terms` on a keyword field), run
//! through `aggs::aggregate_sliced_counting` -- the call
//! `ffi_jvm_reader_aggregate` makes -- over the benchmark corpus
//! (`benchmarks/.corpus/merged`: `keyword` is `SORTED`, `cat`
//! `SORTED_SET`, `num` a `LongPoint`). The Java side,
//! `benchmarks/micro/java/AggsMicro.java`, runs OpenSearch's collector for
//! the same request over Lucene (Lucene has no terms aggregation):
//! `scripts/bench-micro.sh --bench aggs`. Each case prints a `#check`
//! digest of its buckets.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::aggs::aggregate_sliced_counting;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{BooleanQuery, Clause, PointsRangeQuery, TermQuery};
use lucene_search::terms_agg::TermsSpec;
use lucene_store::FsDirectory;

use super::measure;

pub fn bench_aggs(w: Duration, m: Duration, dir: &str) {
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let readers = reader.segment_readers();
    let slices = vec![(0..segments.len()).collect::<Vec<_>>()];
    let filter = |c: Clause| BooleanQuery {
        filter: vec![c],
        ..Default::default()
    };
    let term = |t: &str| Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()));
    let cases: Vec<(&str, BooleanQuery)> = vec![
        ("aggs_terms_dense_term", filter(term("t0"))),
        ("aggs_terms_mid_term", filter(term("t5"))),
        (
            "aggs_terms_dense_range",
            filter(Clause::PointsRange(PointsRangeQuery::new(
                "num", 0, 700_000,
            ))),
        ),
    ];
    for field in ["keyword", "cat"] {
        let terms = [TermsSpec {
            field: field.to_string(),
            shard_size: 25,
        }];
        let globals = [reader.global_ords(field).unwrap()];
        for (name, q) in &cases {
            let name = format!("{name}_{field}");
            let run = || {
                let (sliced, _) = aggregate_sliced_counting(
                    &segments,
                    readers,
                    q,
                    &[],
                    &terms,
                    &globals,
                    &slices,
                    None,
                )
                .unwrap();
                let mut h = 0u64;
                for (_, results) in &sliced {
                    for r in results {
                        h = h.wrapping_mul(31).wrapping_add(r.other_doc_count);
                        for (t, c) in &r.buckets {
                            h = h
                                .wrapping_mul(31)
                                .wrapping_add(*c)
                                .wrapping_add(t.len() as u64);
                        }
                    }
                }
                h
            };
            if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
                continue;
            }
            println!("#check\t{name}\t{:016x}\t1", run());
            measure(&name, w, m, || {
                black_box(run());
                1
            });
        }
    }
}
