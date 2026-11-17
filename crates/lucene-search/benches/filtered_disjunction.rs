//! A `MUST`-free disjunction under a `FILTER`, top-50 with pruning: the
//! `MaxScoreBulkScorer` filter path, for filters of falling density.
//!
//! Skips itself when `benchmarks/.corpus/merged` is absent
//! (`scripts/bench-corpus.sh`). Run with
//! `cargo bench -p lucene-search --bench filtered_disjunction`.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::norms;
use lucene_search::collector::TopDocsCollector;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::MmapDirectory;

const TOP_N: usize = 50;

fn corpus_dir() -> Option<String> {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/.corpus/merged"
    );
    std::path::Path::new(dir)
        .join("segments_1")
        .exists()
        .then(|| dir.to_string())
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t))
}

fn bench(c: &mut Criterion) {
    let Some(dir_path) = corpus_dir() else {
        eprintln!(
            "filtered_disjunction: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let seg = &segments[0];

    // Norms, read directly like the bench runner does -- without them BM25
    // loses length normalization, which is the very work a filter clause skips.
    let base = std::path::Path::new(&dir_path);
    let name = &reader.segment_readers()[0].segment_name;
    let commit_id = reader.segment_infos.segments[0].segment_id;
    let meta = std::fs::read(base.join(format!("{name}.nvm"))).expect("read .nvm");
    let data = std::fs::read(base.join(format!("{name}.nvd"))).expect("read .nvd");
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());
    let (_, parsed) = norms::parse_meta(&meta, &commit_id, "").expect("parse .nvm");
    let mut norms_map: HashMap<String, FieldNorms<'static>> = HashMap::new();
    for fi in &reader.segment_readers()[0].field_infos().fields {
        if let (Some(entry), Some(f)) = (parsed.entry(fi.number), seg.fields.field(&fi.name)) {
            norms_map.insert(
                fi.name.clone(),
                FieldNorms::from_field_stats(data, *entry, f.sum_total_term_freq, f.doc_count),
            );
        }
    }

    let shoulds3 = ["t1", "t2", "t3"];
    let shoulds6 = ["t1", "t2", "t3", "t4", "t5", "t6"];
    let mut cases: Vec<(String, BooleanQuery)> = Vec::new();
    for filter in ["t0", "t10", "t100", "t1000", "t10000"] {
        for (label, set) in [("3", &shoulds3[..]), ("6", &shoulds6[..])] {
            cases.push((
                format!("filter_{filter}_should{label}"),
                BooleanQuery::new()
                    .with_filter([term(filter)])
                    .with_should(set.iter().map(|t| term(t)).collect::<Vec<_>>())
                    .with_minimum_should_match(1),
            ));
        }
    }
    let mut group = c.benchmark_group("filtered_disjunction");
    for (name, query) in &cases {
        group.bench_function(name.as_str(), |b| {
            b.iter(|| {
                let mut top = TopDocsCollector::new(TOP_N);
                lucene_search::search_boolean_query_scored(
                    seg.fields,
                    seg.doc_in,
                    seg.pos_in,
                    seg.pay_in,
                    seg.live_docs,
                    None,
                    black_box(query),
                    Some(&norms_map),
                    &mut top,
                )
                .expect("search");
                black_box(top.top_docs().len())
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
