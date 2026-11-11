//! Newest-N (top N by a time field) over the Java-written Loghub index of
//! `benchmarks/newest-n/NewestN.java`: the same shapes, their median
//! latency, and their hits written beside Java's for comparison.
//!
//! `cargo run --release -p lucene-search --example newest_n -- <idx> <out dir> <label> [cold <shape>]`
#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, PointsRangeQuery, TermQuery};
use lucene_search::top_field::{search_sorted, FieldDoc, SortField, SortType, TopFieldDocs};
use lucene_store::MmapDirectory;
use std::io::Write;
use std::time::Instant;

const T0: i64 = 1_730_000_000_000;
const SPAN: i64 = 7 * 24 * 3600 * 1000;

struct Shape {
    name: &'static str,
    query: BooleanQuery,
    sort: SortField,
    n: usize,
    page2: bool,
}

fn key(field: &str, reverse: bool, missing: Option<i64>) -> SortField {
    let mut k = SortField::numeric(field, SortType::Long, reverse);
    if let Some(m) = missing {
        k.missing = m;
    }
    k
}

fn shapes(max_doc: i32) -> Vec<Shape> {
    let to = T0 + SPAN;
    let from = to - 24 * 3600 * 1000;
    let all = || BooleanQuery {
        filter: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))],
        ..Default::default()
    };
    let term = |s: &str| BooleanQuery {
        filter: vec![Clause::Term(TermQuery::new("src", s.as_bytes().to_vec()))],
        ..Default::default()
    };
    let range = || Clause::PointsRange(PointsRangeQuery::new("ts", from, to));
    let range_and = |s: &str| BooleanQuery {
        filter: vec![
            range(),
            Clause::Term(TermQuery::new("src", s.as_bytes().to_vec())),
        ],
        ..Default::default()
    };
    let mut out = Vec::new();
    for n in [10usize, 100, 1000] {
        let mut add = |name, query, sort, page2| {
            out.push(Shape {
                name,
                query,
                sort,
                n,
                page2,
            })
        };
        add("match_all ts desc", all(), key("ts", true, None), false);
        add("match_all ts asc", all(), key("ts", false, None), false);
        add(
            "match_all ts_s(skipper) desc",
            all(),
            key("ts_s", true, None),
            false,
        );
        add(
            "match_all ts_s(skipper) asc",
            all(),
            key("ts_s", false, None),
            false,
        );
        add(
            "src=openssh ts desc",
            term("openssh"),
            key("ts", true, None),
            false,
        );
        add(
            "src=hdfs ts desc",
            term("hdfs"),
            key("ts", true, None),
            false,
        );
        add(
            "src=openssh ts_s(skipper) desc",
            term("openssh"),
            key("ts_s", true, None),
            false,
        );
        add(
            "range24h ts desc",
            BooleanQuery {
                filter: vec![range()],
                ..Default::default()
            },
            key("ts", true, None),
            false,
        );
        add(
            "range24h&hdfs ts desc",
            range_and("hdfs"),
            key("ts", true, None),
            false,
        );
        add(
            "range24h&openssh(0 hits) ts desc",
            range_and("openssh"),
            key("ts", true, None),
            false,
        );
        add(
            "match_all tsec(ties) desc",
            all(),
            key("tsec", true, None),
            false,
        );
        add(
            "match_all tsec(ties) asc",
            all(),
            key("tsec", false, None),
            false,
        );
        add(
            "match_all tmiss desc missing_last",
            all(),
            key("tmiss", true, Some(i64::MIN)),
            false,
        );
        add(
            "match_all tmiss asc missing_last",
            all(),
            key("tmiss", false, Some(i64::MAX)),
            false,
        );
        add(
            "match_all tmiss desc missing_first",
            all(),
            key("tmiss", true, Some(i64::MAX)),
            false,
        );
        add(
            "match_all ts desc search_after",
            all(),
            key("ts", true, None),
            true,
        );
        add(
            "src=hdfs tsec(ties) desc search_after",
            term("hdfs"),
            key("tsec", true, None),
            true,
        );
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (idx, out_dir, label) = (&args[1], &args[2], &args[3]);
    let cold = args.get(4).map(String::as_str) == Some("cold");
    std::fs::create_dir_all(out_dir).unwrap();
    let start = Instant::now();
    let dir = MmapDirectory::open(idx);
    let reader = DirectoryReader::open(&dir).unwrap();
    let max: i32 = reader.segment_readers().iter().map(|s| s.max_doc).sum();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segs = opened.as_open_segments();
    let norms: Vec<_> = segs.iter().map(|_| None).collect();
    let all = shapes(max);
    let run = |sh: &Shape| -> TopFieldDocs {
        let sort = [sh.sort.clone()];
        let first = search_sorted(
            &segs,
            reader.segment_readers(),
            &sh.query,
            &norms,
            &sort,
            sh.n,
            1000,
            None,
        )
        .unwrap();
        if !sh.page2 {
            return first;
        }
        let Some(last) = first.hits.last().cloned() else {
            return first;
        };
        let after = FieldDoc {
            doc: last.doc,
            values: last.values,
            terms: Vec::new(),
        };
        search_sorted(
            &segs,
            reader.segment_readers(),
            &sh.query,
            &norms,
            &sort,
            sh.n,
            1000,
            Some(&after),
        )
        .unwrap()
    };
    if cold {
        let i: usize = args[5].parse().unwrap();
        let sh = &all[i];
        std::hint::black_box(run(sh));
        println!(
            "{},{},{:.3}",
            sh.name,
            sh.n,
            start.elapsed().as_secs_f64() * 1e3
        );
        return;
    }
    println!("segments={} docs={max}", segs.len());
    let mut csv = std::fs::File::create(format!("{out_dir}/{label}.csv")).unwrap();
    writeln!(csv, "shape,n,median_ms,p90_ms,reps,hits,total,relation").unwrap();
    let only = std::env::var("SHAPES").ok();
    for sh in &all {
        if let Some(o) = &only {
            if !format!("{} {}", sh.name, sh.n).contains(o.as_str()) {
                continue;
            }
        }
        let mut top = run(sh);
        let t = Instant::now();
        let mut warm = 0;
        while t.elapsed().as_millis() < 600 && warm < 30 {
            top = run(sh);
            warm += 1;
        }
        let mut v = Vec::new();
        let t = Instant::now();
        while (t.elapsed().as_millis() < 1200 || v.len() < 3) && v.len() < 400 {
            let s = Instant::now();
            top = std::hint::black_box(run(sh));
            v.push(s.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let p90 = v[(v.len() * 9 / 10).min(v.len() - 1)];
        let rel = format!("{:?}", top.total.relation);
        println!(
            "{:<45} n={:<5} median={med:9.3} ms reps={} hits={} total={}{}",
            sh.name,
            sh.n,
            v.len(),
            top.hits.len(),
            top.total.value,
            if rel.contains("Equal") { "" } else { "+" }
        );
        writeln!(
            csv,
            "{},{},{med:.4},{p90:.4},{},{},{},{rel}",
            sh.name,
            sh.n,
            v.len(),
            top.hits.len(),
            top.total.value
        )
        .unwrap();
        let safe: String = sh
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .split('_')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        let mut hw =
            std::fs::File::create(format!("{out_dir}/{label}-{safe}-{}.hits", sh.n)).unwrap();
        for h in &top.hits {
            writeln!(hw, "{} {}", h.doc, h.values[0]).unwrap();
        }
    }
}
