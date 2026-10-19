//! The lucene-sandbox benchmark pair (M14.f), against
//! `benchmarks/micro/java/SandboxMicro.java`: per range set, one top-10
//! search -- `IntPointMultiRangeBuilder` over `p`, the sorted-numeric
//! multi-range query over `sn` and over the skip-indexed `snx`, the
//! sorted-set one over `ss` -- and `FloatPointNearestNeighbor`'s 10 nearest
//! points of `v`, over the 200 000-document, four-segment index the Java side
//! builds (`SandboxMicro build <dir>`). Each case prints a `#check` digest of
//! its hits the report compares before it shows a ratio. The scoring
//! queries -- the sandbox `CombinedFieldQuery` over three terms,
//! `CoveringQuery`, `TermAutomatonQuery` and `PhraseWildcardQuery` -- run
//! over its second index, `text/`, four words a range set.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::collector::ScoreDoc;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::{search_top_docs, DocumentQuery};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery, TermQuery};
use lucene_search::sandbox::document::{FloatPointNearestNeighbor, IntPointMultiRangeBuilder};
use lucene_search::sandbox::search::{
    CombinedFieldQuery, CoveringQuery, PhraseWildcardBuilder, SortedNumericStabbingBuilder,
    SortedSetStabbingBuilder, TermAutomatonQuery,
};
use lucene_search::values_source::long_from_long_field;
use lucene_store::FsDirectory;

use super::measure;

/// FNV-1a over 64-bit words, identical to `SandboxMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn digest(f: &mut Fnv, docs: &[ScoreDoc]) {
    for d in docs {
        f.add(i64::from(d.doc_id));
        f.add(i64::from(d.score.to_bits() as i32));
    }
}

type Ranges = Vec<(i64, i64)>;

fn term(t: i64) -> Vec<u8> {
    format!("t{t:05}").into_bytes()
}

fn cases(
    name: &str,
    sets: &[Ranges],
    segments: &[OpenSegment<'_>],
    w: Duration,
    m: Duration,
    query: &dyn Fn(&Ranges) -> Box<dyn DocumentQuery>,
) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    let run = |f: &mut Fnv| {
        for set in sets {
            let q = query(set);
            digest(
                f,
                &search_top_docs(segments, q.as_ref(), 10)
                    .unwrap()
                    .score_docs,
            );
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\t{name}\t{:016x}\t{}", f.0, sets.len());
    measure(name, w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        sets.len() as u64
    });
}

/// `SandboxMicro.words(set)`.
fn words(set: &Ranges) -> [String; 4] {
    let w = |x: i64| format!("w{x:04}");
    [
        w(set[0].0 % 50),
        w(set[1].0 % 200),
        w(set[2].0 % 200),
        w(set[3].0 % 50),
    ]
}

fn scored_cases(
    name: &str,
    sets: &[Ranges],
    s: &IndexSearcher<'_, '_>,
    w: Duration,
    m: Duration,
    query: &dyn Fn(&Ranges) -> Clause,
) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != name) {
        return;
    }
    let run = |f: &mut Fnv| {
        for set in sets {
            let q = BooleanQuery {
                must: vec![query(set)],
                ..Default::default()
            };
            for sd in &s.search(&q, 10).unwrap().score_docs {
                f.add(i64::from(sd.doc));
                f.add(i64::from(sd.score.to_bits() as i32));
            }
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check	{name}	{:016x}	{}", f.0, sets.len());
    measure(name, w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        sets.len() as u64
    });
}

fn bench_text(w: Duration, m: Duration, dir: &str, sets: &[Ranges]) {
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let mut segments = opened.as_open_segments();
    for seg in &mut segments {
        seg.cache = None;
    }
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let s = IndexSearcher::new(&segments, &norms).unwrap();
    scored_cases("sb_combined_terms", sets, &s, w, m, &|set| {
        let t = words(set);
        CombinedFieldQuery::with_terms(t[..3].to_vec(), [("body", 1.0), ("title", 2.0)])
            .unwrap()
            .into()
    });
    scored_cases("sb_covering", sets, &s, w, m, &|set| {
        let qs = words(set).map(|t| Clause::Term(TermQuery::new("body", t)));
        CoveringQuery::new(qs, long_from_long_field("mm"))
            .unwrap()
            .into()
    });
    scored_cases("sb_term_automaton", sets, &s, w, m, &|set| {
        let t = words(set);
        let mut q = TermAutomatonQuery::new("body");
        for _ in 0..4 {
            q.create_state();
        }
        q.set_accept(3, true).unwrap();
        q.add_transition(0, 1, t[0].as_str());
        q.add_transition(1, 3, t[1].as_str());
        q.add_transition(1, 2, t[2].as_str());
        q.add_transition(2, 3, t[3].as_str());
        q.finish().unwrap();
        q.into()
    });
    scored_cases("sb_phrase_wildcard", sets, &s, w, m, &|set| {
        let t = words(set);
        let mut b = PhraseWildcardBuilder::new("body", 128);
        b.add_term(t[0].as_str());
        b.add_multi_term(&Clause::Prefix(PrefixQuery::new("body", &t[1][..4])))
            .unwrap();
        b.build().into()
    });
}

fn origin(set: &Ranges) -> [f32; 3] {
    [
        (set[0].0 % 100) as f32,
        (set[1].0 % 100) as f32,
        (set[2].0 % 100) as f32,
    ]
}

pub fn bench_sandbox(w: Duration, m: Duration, dir: &str) {
    let sets: Vec<Ranges> = std::fs::read_to_string(format!("{dir}/sandbox-ranges.tsv"))
        .expect("run SandboxMicro build first (scripts/bench-micro.sh --bench sandbox)")
        .lines()
        .map(|l| {
            l.split(';')
                .map(|r| {
                    let (lo, hi) = r.split_once(':').unwrap();
                    (lo.parse().unwrap(), hi.parse().unwrap())
                })
                .collect()
        })
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let mut segments = opened.as_open_segments();
    for seg in &mut segments {
        seg.cache = None;
    }

    cases("sb_multi_range_points", &sets, &segments, w, m, &|set| {
        let mut b = IntPointMultiRangeBuilder::new("p", 1).unwrap();
        for &(lo, hi) in set {
            b.add(&[lo as i32], &[hi as i32]).unwrap();
        }
        Box::new(b.build())
    });
    cases("sb_dv_multi_range", &sets, &segments, w, m, &|set| {
        let mut b = SortedNumericStabbingBuilder::new("sn");
        for &(lo, hi) in set {
            b.add(lo, hi);
        }
        b.build()
    });
    cases("sb_dv_multi_range_skip", &sets, &segments, w, m, &|set| {
        let mut b = SortedNumericStabbingBuilder::new("snx");
        for &(lo, hi) in set {
            b.add(lo, hi);
        }
        b.build()
    });
    cases("sb_ss_multi_range", &sets, &segments, w, m, &|set| {
        let mut b = SortedSetStabbingBuilder::new("ss");
        for &(lo, hi) in set {
            b.add(term(lo / 50), term(hi / 50));
        }
        b.build()
    });
    bench_text(w, m, &format!("{dir}/text"), &sets);
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != "sb_nearest") {
        return;
    }
    let run = |f: &mut Fnv| {
        for set in &sets {
            let hits =
                FloatPointNearestNeighbor::nearest(&segments, "v", 10, &origin(set)).unwrap();
            for (doc, distance) in hits.hits {
                f.add(i64::from(doc));
                f.add(i64::from(distance.to_bits() as i32));
            }
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\tsb_nearest\t{:016x}\t{}", f.0, sets.len());
    measure("sb_nearest", w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        sets.len() as u64
    });
}
