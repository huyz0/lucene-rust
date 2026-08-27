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
use lucene_search::collector::ScoringCollector;
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
        eprintln!("usage: bench-runner <index-dir> <queries.tsv> <warmup_ms> <measure_ms>");
        std::process::exit(2);
    }
    let (dir_path, queries_path) = (&a[1], &a[2]);
    // Time-boxed, not count-based. A fixed iteration count cannot serve both a
    // 5us query and a 15s one: low enough for the slow query leaves the JVM's
    // JIT cold on the fast ones, which biases exactly the queries where Rust
    // wins. Time-boxing gives every query the same warmup *duration*.
    let warmup_ms: u128 = a[3].parse().expect("warmup_ms");
    let measure_ms: u128 = a[4].parse().expect("measure_ms");

    let queries = load_queries(queries_path);
    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();

    // Norms, wired per segment. DirectoryReader does not load these itself (see
    // the M1 findings), so the runner reads .nvm/.nvd directly -- without them
    // BM25 loses length normalization and would not match Lucene's scores.
    let norms_by_seg = load_norms(dir_path, &reader, &segments);
    let bool_norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        norms_by_seg.iter().map(|m| m.as_ref()).collect();

    if std::env::var("BENCH_DUMP_STATS").is_ok() {
        // Per-segment collection statistics, to show how far this port's
        // per-segment idf drifts from Lucene's global one.
        let mut tot_df = 0i64;
        let mut tot_dc = 0i64;
        for (i, seg) in segments.iter().enumerate() {
            if let Some(ft) = seg.fields.field("body") {
                let df = ft.seek_exact(b"t0").map(|s| s.doc_freq).unwrap_or(0) as i64;
                let dc = ft.doc_count as i64;
                tot_df += df;
                tot_dc += dc;
                let idf = ((dc as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
                eprintln!("  seg {i:>2}: docCount={dc:>9} docFreq(t0)={df:>9} idf={idf:.6}");
            }
        }
        let gidf = ((tot_dc as f64 - tot_df as f64 + 0.5) / (tot_df as f64 + 0.5) + 1.0).ln();
        eprintln!("  GLOBAL: docCount={tot_dc} docFreq={tot_df} idf={gidf:.6}  <- what Lucene uses");
        return;
    }
    println!("id\thits\ttop1doc\ttop1score\ttopset\tqps\tp50_us\tp95_us\tp99_us");
    for q in &queries {
        let run = || -> Vec<lucene_search::collector::ScoreDoc> {
            match q.kind.as_str() {
                // Probe: the existing impacts-pruned single-segment path, which
                // search_term_query_multi_segment does NOT call. Single-segment
                // only (the merged corpus), which is all the probe needs.
                // Diagnostic: the pre-M1.5 eager path, merged by hand exactly as
                // merge_multi_segment_scored does, to isolate whether a
                // multi-segment disagreement comes from the pruned path or was
                // already there.
                "term_eager" => {
                    let tq = TermQuery { field: q.field.clone(), term: q.args[0].clone().into_bytes() };
                    let mut merged = lucene_search::collector::TopDocsCollector::new(TOP_N);
                    for (i, seg) in segments.iter().enumerate() {
                        let mut local = lucene_search::collector::TopDocsCollector::new(TOP_N);
                        lucene_search::search_term_query_scored(
                            seg.fields, seg.doc_in, seg.live_docs, &tq,
                            norms_by_seg[i].as_ref().and_then(|m| m.get(&q.field)),
                            &mut local,
                        ).expect("term_eager");
                        for hit in local.top_docs() {
                            merged.collect(hit.doc_id + seg.doc_base, hit.score);
                        }
                    }
                    merged.top_docs().to_vec()
                }
                "term_ms" => {
                    let tq = TermQuery { field: q.field.clone(), term: q.args[0].clone().into_bytes() };
                    let seg = &segments[0];
                    let mut c = lucene_search::collector::TopDocsCollector::new(TOP_N);
                    lucene_search::search_term_query_scored_maxscore(
                        seg.fields, seg.doc_in, seg.live_docs, &tq,
                        norms_by_seg[0].as_ref().and_then(|m| m.get(&q.field)),
                        &mut c,
                    ).expect("term_ms");
                    c.top_docs().to_vec()
                }
                "term" => {
                    let tq = TermQuery { field: q.field.clone(), term: q.args[0].clone().into_bytes() };
                    // Per-query field, not a hardcoded one: scoring a `title` or
                    // `keyword` query with `body`'s norms silently produces a
                    // different ranking, which first showed up as a phantom
                    // recall mismatch against Java.
                    let tn: Vec<Option<&FieldNorms<'_>>> = norms_by_seg
                        .iter()
                        .map(|m| m.as_ref().and_then(|m| m.get(&q.field)))
                        .collect();
                    search_term_query_multi_segment(&segments, &tq, &tn, TOP_N).expect("term")
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

        // How effective is block pruning actually? Counting skips answers
        // whether the remaining gap is decode speed or blocks never skipped.
        if std::env::var("BENCH_COUNT_SKIPS").is_ok() {
            lucene_search::test_only_maxscore_block_skip_counter::reset();
            let hits = run();
            let skips = lucene_search::test_only_maxscore_block_skip_counter::count();
            eprintln!(
                "  {}: {} hits, {} blocks skipped",
                q.id,
                hits.len(),
                skips
            );
            continue;
        }

        let w = Instant::now();
        loop {
            std::hint::black_box(run());
            if w.elapsed().as_millis() >= warmup_ms { break; }
        }
        let mut samples = Vec::new();
        let mut last;
        let t0 = Instant::now();
        loop {
            let s = Instant::now();
            last = run();
            samples.push(s.elapsed().as_micros() as u64);
            // At least 5 samples so a percentile means something, even when a
            // single execution already exceeds the measurement budget.
            if t0.elapsed().as_millis() >= measure_ms && samples.len() >= 5 { break; }
        }
        let wall = t0.elapsed().as_secs_f64();
        let iters = samples.len();

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
    segments: &[lucene_search::multi_segment::OpenSegment<'a>],
) -> Vec<Option<HashMap<String, FieldNorms<'a>>>> {
    // Buffers must outlive the FieldNorms borrowing them, so leak them: the
    // runner is a short-lived process measuring steady-state query cost, and a
    // self-referential owner here would add lifetime noise for no benefit.
    reader
        .segment_readers()
        .iter()
        .zip(reader.segment_infos.segments.iter())
        .enumerate()
        .map(|(i, (seg, commit))| {
            let base = std::path::Path::new(dir_path);
            let meta = std::fs::read(base.join(format!("{}.nvm", seg.segment_name))).ok()?;
            let data = std::fs::read(base.join(format!("{}.nvd", seg.segment_name))).ok()?;
            let data: &'a [u8] = Box::leak(data.into_boxed_slice());
            let (_, parsed) = norms::parse_meta(&meta, &commit.segment_id, "").ok()?;
            let mut out = HashMap::new();
            for fi in &seg.field_infos().fields {
                if let Some(entry) = parsed.entry(fi.number) {
                    // Lucene-exact avgdl from the field's .tmd counters, not the
                    // average of lossy decoded norms. See FieldNorms::from_field_stats.
                    if let Some(ft) = segments[i].fields.field(&fi.name) {
                        out.insert(
                            fi.name.clone(),
                            FieldNorms::from_field_stats(
                                data, *entry, ft.sum_total_term_freq, ft.doc_count,
                            ),
                        );
                    }
                }
            }
            Some(out)
        })
        .collect()
}
