//! M14.f, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSandboxCollectors.java`.
//!
//! lucene-sandbox's collectors over a three-segment index with deletions:
//! `LargeNumHitsTopDocsCollector` read through its manager and through
//! `topDocs(howMany)` -- below, at and past the requested count, so both the
//! list and the popped queue are read (`large.tsv`); the collector
//! profilers' result trees and the hits of the collector they wrap
//! (`profiler.tsv`); and the arguments Java refuses (`errors.tsv`).

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::collector::{ScoringCollector, TopDocsCollector, TotalHitsRelation};
use lucene_search::collectors::CollectorManager;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::sandbox::search::{
    LargeNumHitsTopDocsCollector, LargeNumHitsTopDocsCollectorManager, ProfilerCollector,
    ProfilerCollectorManager, ProfilerCollectorResult,
};
use lucene_search::top_docs::TopDocs;
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sandbox_collectors")
}

/// `GenSandboxCollectors.query`.
fn query(spec: &str) -> BooleanQuery {
    let t: Vec<&str> = spec.split(' ').collect();
    let term = |w: &str| Clause::Term(TermQuery::new("body", w));
    match t[0] {
        "T" => BooleanQuery {
            must: vec![term(t[1])],
            ..Default::default()
        },
        "B" => {
            BooleanQuery::new().with_should(t[1..].iter().map(|w| term(w)).collect::<Vec<Clause>>())
        }
        "all" => BooleanQuery {
            must: vec![Clause::from(MatchAllDocsQuery::new(150))],
            ..Default::default()
        },
        other => panic!("{other}"),
    }
}

/// `GenSandboxCollectors.render`.
fn render(td: &TopDocs) -> String {
    let rel = match td.total_hits.relation {
        TotalHitsRelation::EqualTo => "EQUAL_TO",
        TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
    };
    let mut b = format!("{}\t{rel}\t", td.total_hits.value);
    for sd in &td.score_docs {
        b.push_str(&format!("{}:{:x},", sd.doc, sd.score.to_bits()));
    }
    b
}

/// `GenSandboxCollectors.profile`.
fn profile(r: &ProfilerCollectorResult) -> String {
    format!(
        "{}|{}|{}|{}",
        r.name,
        r.reason,
        if r.time > 0 { "timed" } else { "untimed" },
        r.children.len()
    )
}

fn open<T>(f: impl FnOnce(&IndexSearcher<'_, '_>) -> T) -> T {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let fields = vec!["body".to_string()];
    let owned = reader.field_norms_by_field(&fields).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&searcher)
}

/// One collector over the whole reader, then `topDocs(howMany)`.
fn top_docs(searcher: &IndexSearcher<'_, '_>, spec: &str, req: i32, how_many: i32) -> String {
    let mut c = LargeNumHitsTopDocsCollector::new(req).unwrap();
    searcher.search_collector(&query(spec), &mut c).unwrap();
    let held = c.total_hits().min(usize::try_from(req).unwrap());
    let td = c.top_docs_n(how_many).unwrap();
    // A queue smaller than `howMany` leaves Java's leading slots `null`;
    // the hits returned are exactly the ones held.
    let asked = usize::try_from(how_many).unwrap().min(c.total_hits());
    assert_eq!(
        td.score_docs.len(),
        asked.min(held),
        "{spec} {req} {how_many}"
    );
    let nulls = "null,".repeat(asked - td.score_docs.len());
    let r = render(&td);
    let (head, hits) = r.rsplit_once('\t').unwrap();
    format!("{head}\t{nulls}{hits}")
}

#[test]
fn large_num_hits_collects_as_lucene_does() {
    let text = std::fs::read_to_string(data().join("large.tsv")).unwrap();
    open(|searcher| {
        let mut bad = 0;
        for line in text.lines() {
            let f: Vec<&str> = line.splitn(4, '\t').collect();
            let (spec, req) = (f[0], f[1].parse::<i32>().unwrap());
            let got = if f[2] == "all" {
                let m = LargeNumHitsTopDocsCollectorManager::new(req).unwrap();
                render(&searcher.search_manager(&query(spec), &m).unwrap())
            } else {
                top_docs(searcher, spec, req, f[2].parse().unwrap())
            };
            if got != f[3] {
                bad += 1;
                eprintln!("{spec} {req} {}\n  java {}\n  rust {got}", f[2], f[3]);
            }
        }
        assert_eq!(bad, 0, "{bad} of {} lines differ", text.lines().count());
    });
}

#[test]
fn profilers_report_as_lucene_does() {
    let text = std::fs::read_to_string(data().join("profiler.tsv")).unwrap();
    open(|searcher| {
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let q = query(f[0]);
            // Java's `TopScoreDocCollector` is this port's `TopDocsCollector`.
            let rust_name = |s: &str| s.replace("TopScoreDocCollector", "TopDocsCollector");
            let m = ProfilerCollectorManager::new("search_top_hits", || {
                Ok(TopDocsCollector::with_total_hits_threshold(5, 1000))
            });
            // The one collector the manager makes, as Java's `reduce` sees it.
            let mut single = m.new_collector().unwrap();
            searcher.search_collector(&q, &mut single).unwrap();
            assert_eq!(profile(&single.profile_result()), rust_name(f[1]), "{line}");
            let reduced = searcher.search_manager(&q, &m).unwrap();
            assert_eq!(profile(&reduced), rust_name(f[2]), "{line}");

            let m = ProfilerCollectorManager::new("large", || LargeNumHitsTopDocsCollector::new(3));
            assert_eq!(profile(&searcher.search_manager(&q, &m).unwrap()), f[3]);

            let child = ProfilerCollector::new(
                LargeNumHitsTopDocsCollector::new(2).unwrap(),
                "child",
                Vec::new(),
            );
            let mut parent = ProfilerCollector::new(
                LargeNumHitsTopDocsCollector::new(4).unwrap(),
                "parent",
                vec![child.handle()],
            );
            searcher.search_collector(&q, &mut parent).unwrap();
            let tree = parent.profile_result();
            assert_eq!(profile(&tree), f[4], "{line}");
            assert_eq!(profile(&tree.children[0]), f[5], "{line}");
            assert_eq!(format!("{:?}", parent.score_mode()).to_uppercase(), f[6]);
            // The wrapped collector's hits are the unwrapped one's.
            assert_eq!(
                render(&parent.into_collector().top_docs()),
                f[7..].join("\t")
            );
        }
    });
}

/// Java's exception class and message.
fn java<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => "OK".into(),
        Err(Error::IllegalArgument(m)) => format!("IllegalArgumentException: {m}"),
        Err(other) => format!("{other:?}"),
    }
}

#[test]
fn collector_errors_are_lucenes() {
    let text = std::fs::read_to_string(data().join("errors.tsv")).unwrap();
    for line in text.lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let got = match spec {
            "collector 0" => java(LargeNumHitsTopDocsCollector::new(0)),
            "collector -3" => java(LargeNumHitsTopDocsCollector::new(-3)),
            "manager 0" => java(LargeNumHitsTopDocsCollectorManager::new(0)),
            "topDocs -1" => java(LargeNumHitsTopDocsCollector::new(5).unwrap().top_docs_n(-1)),
            other => panic!("{other}"),
        };
        assert_eq!(got, want, "{spec}");
    }
    // `CollectorManager::new_collector` hands out an empty collector.
    let m = LargeNumHitsTopDocsCollectorManager::new(3).unwrap();
    assert_eq!(m.new_collector().unwrap().total_hits(), 0);
}
