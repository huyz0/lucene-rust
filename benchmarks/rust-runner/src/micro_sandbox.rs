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
//! over its second index, `text/`, four words a range set, and so does
//! `LargeNumHitsTopDocsCollectorManager`'s top 1000 of two of them.
//! `IDVersionSegmentTermsEnum.seekExact(id, minVersion)` runs over a third,
//! `idv/`, and `KMeans.cluster` over the vectors in `kmeans/`.

use std::hint::black_box;
use std::time::Duration;

use lucene_codecs::idversion::VersionBlockTreeTermsReader;
use lucene_codecs::quantization::{cluster, InMemoryFloatVectors};
use lucene_search::collector::ScoreDoc;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::{search_top_docs, DocumentQuery};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery, TermQuery};
use lucene_search::sandbox::document::{FloatPointNearestNeighbor, IntPointMultiRangeBuilder};
use lucene_search::sandbox::search::{
    CombinedFieldQuery, CoveringQuery, LargeNumHitsTopDocsCollectorManager, PhraseWildcardBuilder,
    SortedNumericStabbingBuilder, SortedSetStabbingBuilder, TermAutomatonQuery,
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
    // `LargeNumHitsTopDocsCollector`: the top 1000 of two mid-frequency words.
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != "sb_large_num_hits") {
        return;
    }
    let qs: Vec<BooleanQuery> = sets
        .iter()
        .map(|set| {
            let t = words(set);
            BooleanQuery::new().with_should(
                [&t[1], &t[2]]
                    .map(|w| Clause::Term(TermQuery::new("body", w.as_str())))
                    .to_vec(),
            )
        })
        .collect();
    let run = |f: &mut Fnv| {
        for q in &qs {
            let m = LargeNumHitsTopDocsCollectorManager::new(1000).unwrap();
            for sd in &s.search_manager(q, &m).unwrap().score_docs {
                f.add(i64::from(sd.doc));
                f.add(i64::from(sd.score.to_bits() as i32));
            }
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\tsb_large_num_hits\t{:016x}\t{}", f.0, sets.len());
    measure("sb_large_num_hits", w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        sets.len() as u64
    });
}

/// `sb_idversion_seek`: `seekExact(id, minVersion)` on one enum, for each
/// of `idv/targets.tsv`'s lookups, over the `IDVersion` files Java wrote.
fn bench_idversion(w: Duration, m: Duration, dir: &str) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != "sb_idversion_seek") {
        return;
    }
    let targets: Vec<(Vec<u8>, i64)> = std::fs::read_to_string(format!("{dir}/targets.tsv"))
        .expect("run SandboxMicro build first (scripts/bench-micro.sh --bench sandbox)")
        .lines()
        .map(|l| {
            let (id, min) = l.split_once('\t').unwrap();
            (id.as_bytes().to_vec(), min.parse().unwrap())
        })
        .collect();
    let si = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .find(|n| n.ends_with(".si"))
        .unwrap();
    let name = si.trim_end_matches(".si");
    let buf = std::fs::read(format!("{dir}/{si}")).unwrap();
    // magic, codec name, version, then the id.
    let at = 4 + 1 + usize::from(buf[4]) + 4;
    let id: [u8; 16] = buf[at..at + 16].try_into().unwrap();
    let fnm = std::fs::read(format!("{dir}/{name}.fnm")).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
    let names = |n: i32| {
        fis.fields
            .iter()
            .find(|f| f.number == n)
            .map(|f| f.name.clone())
    };
    let suffix = "IDVersion_0";
    let reader = VersionBlockTreeTermsReader::open(
        std::fs::read(format!("{dir}/{name}_{suffix}.tiv")).unwrap(),
        &std::fs::read(format!("{dir}/{name}_{suffix}.tipv")).unwrap(),
        &id,
        suffix,
        i32::MAX,
        &names,
    )
    .unwrap();
    let run = |f: &mut Fnv| {
        let mut te = reader.iterator("id").unwrap();
        for (t, min) in &targets {
            f.add(if te.seek_exact_with_version(t, *min).unwrap() {
                te.version()
            } else {
                -1
            });
        }
    };
    let mut f = Fnv::new();
    run(&mut f);
    println!("#check\tsb_idversion_seek\t{:016x}\t{}", f.0, targets.len());
    measure("sb_idversion_seek", w, m, || {
        let mut g = Fnv::new();
        run(black_box(&mut g));
        black_box(g.0);
        targets.len() as u64
    });
}

/// `sb_kmeans`: `KMeans.cluster(vectors, EUCLIDEAN, 32)` over the Java
/// side's 10 000 16-dimensional vectors, per vector.
fn bench_kmeans(w: Duration, m: Duration, dir: &str) {
    if std::env::var("MICRO_CASE").is_ok_and(|only| only != "sb_kmeans") {
        return;
    }
    const DIM: usize = 16;
    let raw = std::fs::read(format!("{dir}/vectors.bin")).unwrap();
    let vs: Vec<Vec<f32>> = raw
        .chunks_exact(DIM * 4)
        .map(|v| {
            v.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    let values = InMemoryFloatVectors::new(&vs, DIM);
    let res = cluster(&values, false, 32).unwrap().unwrap();
    let mut f = Fnv::new();
    for x in res.centroids.iter().flatten() {
        f.add(i64::from(x.to_bits() as i32));
    }
    for &a in res.vector_centroids.as_ref().unwrap() {
        f.add(i64::from(a));
    }
    println!("#check\tsb_kmeans\t{:016x}\t{}", f.0, vs.len());
    measure("sb_kmeans", w, m, || {
        let r = cluster(black_box(&values), false, 32).unwrap().unwrap();
        black_box(r.vector_centroids.unwrap()[0]);
        vs.len() as u64
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
    bench_idversion(w, m, &format!("{dir}/idv"));
    bench_kmeans(w, m, &format!("{dir}/kmeans"));
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
