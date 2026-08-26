//! lucene-rust side of the M1 performance gate.
//!
//! Reads a Java-written index and a query file, runs each query, and emits one
//! TSV line per query. The Java runner (`benchmarks/java-runner`) emits the
//! identical schema so `scripts/bench-compare.sh` can join them.
//!
//! Output columns:
//!   id  hits  topdocs  qps  p50_us  p95_us  p99_us
//!
//! `hits` and `topdocs` exist so the comparison can cross-check recall before
//! it compares timings: if the two engines disagree on what matched, the
//! timings are measuring different work and mean nothing.

use std::collections::HashMap;
use std::time::Instant;

use lucene_codecs::norms;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::{
    search_boolean_query_multi_segment, search_boolean_query_multi_segment_maxscore,
    search_term_query_multi_segment,
};
use lucene_store::MmapDirectory;

const TOP_N: usize = 50;

/// One query from the checked-in query file.
struct Query {
    id: String,
    kind: String,
    field: String,
    args: Vec<String>,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: bench-runner <index-dir> <queries.tsv> <warmup> <iters>");
        std::process::exit(2);
    }
    let (dir_path, queries_path) = (&a[1], &a[2]);
    let warmup: usize = a[3].parse().expect("warmup");
    let iters: usize = a[4].parse().expect("iters");

    let queries = load_queries(queries_path);
    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();

    // Norms, wired per segment. DirectoryReader does not load these itself (see
    // the M1 findings), so the runner reads .nvm/.nvd directly -- without them
    // BM25 loses length normalization and would not match Lucene's scores.
    let norms_by_seg = load_norms(dir_path, &reader);
    let term_norms: Vec<Option<&FieldNorms<'_>>> = norms_by_seg
        .iter()
        .map(|m| m.as_ref().and_then(|m| m.get("body")))
        .collect();
    let bool_norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        norms_by_seg.iter().map(|m| m.as_ref()).collect();

    println!("id\thits\ttop1doc\ttop1score\ttopset\tqps\tp50_us\tp95_us\tp99_us");
    for q in &queries {
        let run = || -> Vec<lucene_search::collector::ScoreDoc> {
            match q.kind.as_str() {
                "term" => {
                    let tq = TermQuery { field: q.field.clone(), term: q.args[0].clone().into_bytes() };
                    search_term_query_multi_segment(&segments, &tq, &term_norms, TOP_N).expect("term")
                }
                "and" | "or" | "or_maxscore" => {
                    let clauses: Vec<Clause> = q
                        .args
                        .iter()
                        .map(|t| Clause::Term(TermQuery { field: q.field.clone(), term: t.clone().into_bytes() }))
                        .collect();
                    let bq = if q.kind == "and" {
                        BooleanQuery { must: clauses, ..Default::default() }
                    } else {
                        BooleanQuery { should: clauses, ..Default::default() }
                    };
                    if q.kind == "or_maxscore" {
                        search_boolean_query_multi_segment_maxscore(&segments, &bq, &bool_norms, TOP_N)
                            .expect("bool maxscore")
                    } else {
                        search_boolean_query_multi_segment(&segments, &bq, &bool_norms, TOP_N).expect("bool")
                    }
                }
                "phrase" => {
                    let pq = PhraseQuery {
                        field: q.field.clone(),
                        terms: q.args.iter().map(|t| t.clone().into_bytes()).collect(),
                        slop: 0,
                        ..Default::default()
                    };
                    let bq = BooleanQuery { must: vec![Clause::Phrase(pq)], ..Default::default() };
                    search_boolean_query_multi_segment(&segments, &bq, &bool_norms, TOP_N).expect("phrase")
                }
                other => panic!("unknown query kind: {other}"),
            }
        };

        for _ in 0..warmup {
            std::hint::black_box(run());
        }
        let mut samples = Vec::with_capacity(iters);
        let mut last = Vec::new();
        let t0 = Instant::now();
        for _ in 0..iters {
            let s = Instant::now();
            last = run();
            samples.push(s.elapsed().as_micros() as u64);
        }
        let wall = t0.elapsed().as_secs_f64();

        samples.sort_unstable();
        let pct = |p: f64| samples[((samples.len() - 1) as f64 * p) as usize];
        // Compare the top-k as a SET, not an ordered list: equal-scoring docs
        // may legitimately tie-break differently between engines, and treating
        // that as a recall mismatch would hide the real ones.
        let mut ids: Vec<i32> = last.iter().map(|d| d.doc_id).collect();
        ids.sort_unstable();
        let topset = ids.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        let (top1doc, top1score) = last
            .first()
            .map(|d| (d.doc_id, d.score))
            .unwrap_or((-1, 0.0));
        println!(
            "{}\t{}\t{}\t{:.6}\t{}\t{:.1}\t{}\t{}\t{}",
            q.id,
            last.len(),
            top1doc,
            top1score,
            topset,
            iters as f64 / wall,
            pct(0.50),
            pct(0.95),
            pct(0.99)
        );
    }
}

fn load_queries(path: &str) -> Vec<Query> {
    std::fs::read_to_string(path)
        .expect("read queries")
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Query {
                id: f[0].to_string(),
                kind: f[1].to_string(),
                field: f[2].to_string(),
                args: f[3..].iter().map(|s| s.to_string()).collect(),
            }
        })
        .collect()
}

/// Read each segment's `.nvm`/`.nvd` and build a field -> FieldNorms map.
fn load_norms<'a>(
    dir_path: &str,
    reader: &'a DirectoryReader,
) -> Vec<Option<HashMap<String, FieldNorms<'a>>>> {
    // Buffers must outlive the FieldNorms borrowing them, so leak them: the
    // runner is a short-lived process measuring steady-state query cost, and a
    // self-referential owner here would add lifetime noise for no benefit.
    reader
        .segment_readers()
        .iter()
        .zip(reader.segment_infos.segments.iter())
        .map(|(seg, commit)| {
            let base = std::path::Path::new(dir_path);
            let meta = std::fs::read(base.join(format!("{}.nvm", seg.segment_name))).ok()?;
            let data = std::fs::read(base.join(format!("{}.nvd", seg.segment_name))).ok()?;
            let data: &'a [u8] = Box::leak(data.into_boxed_slice());
            let (_, parsed) = norms::parse_meta(&meta, &commit.segment_id, "").ok()?;
            let mut out = HashMap::new();
            for fi in &seg.field_infos().fields {
                if let Some(entry) = parsed.entry(fi.number) {
                    if let Ok(fnorms) = FieldNorms::open(data, *entry, seg.max_doc, None) {
                        out.insert(fi.name.clone(), fnorms);
                    }
                }
            }
            Some(out)
        })
        .collect()
}
