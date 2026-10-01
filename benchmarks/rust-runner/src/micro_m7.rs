//! M7's benchmark pairs, against `benchmarks/micro/java/M7Micro.java`: the
//! queries and similarities M7 brought into the scorer tree, the sorted
//! searches' comparator pruning, `QueryBuilder`, and stored-fields writes in
//! both modes (the zlib port behind `BEST_COMPRESSION`).
//!
//! The queries are the differential tests' own: each case parses query text
//! with the grammar the fixture tests read (`GenM7Queries`',
//! `GenSimilaritySearch`' and `GenSortedSearch`'s, included by path below),
//! and the Java side builds the same text with the generators' own `parse`.
//! So a case times exactly the shapes that are verified bit for bit, and
//! nothing is re-transcribed by hand.
//!
//! Every case also prints a `#check` line -- an FNV-1a digest of every top
//! hit's document and score bits (of the query string, of the written bytes)
//! -- which `scripts/bench-micro-report.py` compares across the engines
//! before it reports a ratio: two engines that disagree are timing different
//! work.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{search_boolean_query_multi_segment_with_similarity, OpenSegment};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::similarities::{Bm25Similarity, Similarity};
use lucene_store::MmapDirectory;

use super::{m7grammar, measure, sexpr, simgrammar};

/// FNV-1a over 64-bit words, identical to `M7Micro.fnv`.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: u64) {
        self.0 ^= x;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn add_hit(&mut self, doc: i32, score: f32) {
        self.add(doc as u32 as u64);
        self.add(score.to_bits() as u64);
    }
    fn add_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
}

/// `MICRO_CASE=<name>` runs one case alone: the others skip their digest
/// pass too, so a profile of it is not mixed with theirs.
fn selected(case: &str) -> bool {
    std::env::var("MICRO_CASE").map_or(true, |only| only == case)
}

/// `MICRO_QUERY=<n>` keeps only the `n`-th query of every case, so one
/// query of a family can be timed or profiled alone (`-Dquery=<n>` on the
/// Java side).
fn query_selected(i: usize) -> bool {
    std::env::var("MICRO_QUERY")
        .ok()
        .and_then(|q| q.parse::<usize>().ok())
        .is_none_or(|q| q == i)
}

fn check(case: &str, digest: Fnv, n: usize) {
    println!("#check\t{case}\t{:016x}\t{n}", digest.0);
}

/// `M7Micro.family`: the family a `GenM7Queries` query is reported under --
/// its first M7 op, the multi-term ones by rewrite method.
fn family(q: &str) -> String {
    let toks: Vec<&str> = q.split(' ').collect();
    for t in &toks {
        let f = match *t {
            "S" => "synonym",
            "CF" => "combined_field",
            "PP" => "phrase_positions",
            "NG" => "ngram_phrase",
            "MP" => "multi_phrase",
            "F" => "fuzzy",
            "PF" | "W" | "RE" | "R" | "A" => {
                let m = toks.last().unwrap().split(':').next().unwrap();
                return format!("mtq_{m}");
            }
            "BT" => "blended",
            "IA" => "indri",
            "LO" => "log_odds",
            "BY" => "bayesian",
            "NR" => "dv_range",
            "ISR" => "index_sort_range",
            "P2R" | "I1R" | "I1S" | "LS" | "LR" | "IPR" | "IPS" => "points",
            "IODV" => "index_or_dv",
            "VSF" | "VSFF" | "VSB" => "vector_similarity",
            "KF" | "KFF" | "KB" | "KFS" => "knn",
            "PKF" => "knn_patience",
            "SKF" => "knn_seeded",
            _ => continue,
        };
        return f.to_string();
    }
    "other".to_string()
}

fn is_vector(q: &str) -> bool {
    q.split(' ').any(|t| {
        matches!(
            t,
            "VSF" | "VSFF" | "VSB" | "KF" | "KFF" | "KB" | "KFS" | "PKF" | "SKF"
        )
    })
}

/// One search to time: the similarity, the query text (re-parsed per run
/// when it holds a vector op, whose rewrite is the KNN search itself, as
/// `AbstractKnnVectorQuery.rewrite` runs inside every `IndexSearcher.search`),
/// and how many hits to collect.
struct Search {
    sim: Arc<dyn Similarity>,
    text: String,
    parsed: Option<BooleanQuery>,
    top: usize,
}

/// Runs the M7 searches of one index, grouped into cases by `family`.
fn run_m7_cases(
    w: Duration,
    m: Duration,
    prefix: &str,
    dir: &std::path::Path,
    lines: &[(String, String, usize)],
    qvec: &[f32],
) {
    let mmap = MmapDirectory::open(dir.to_string_lossy().into_owned());
    let reader = DirectoryReader::open(&mmap).expect("open index");
    let mut opened = reader.open_segments().expect("open segments");
    opened.open_points().expect("open points");
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&[
        "body".to_string(),
        "title".to_string(),
        "gram".to_string(),
    ]);
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(Some).collect();
    let vectors = m7grammar::VectorFiles::read(dir, &reader);
    let knn = |op: &str, field: &str, arg: &str, clause: Option<Clause>| -> Option<Clause> {
        m7grammar::knn_clause(
            op, field, arg, clause, &reader, &segments, &norms, &vectors, qvec,
        )
    };
    let ctx = m7grammar::Ctx { knn: &knn };

    let mut groups: BTreeMap<String, Vec<Search>> = BTreeMap::new();
    for (sim, text, top) in lines {
        let parsed = (!is_vector(text)).then(|| m7grammar::query(text, &ctx).expect("parse"));
        groups
            .entry(format!("{prefix}_{}", family(text)))
            .or_default()
            .push(Search {
                sim: m7grammar::sim(sim),
                text: text.clone(),
                parsed,
                top: *top,
            });
    }
    for (case, searches) in &groups {
        let searches: Vec<&Search> = searches
            .iter()
            .enumerate()
            .filter(|&(i, _)| query_selected(i))
            .map(|(_, s)| s)
            .collect();
        let searches = &searches;
        let run_one = |s: &Search, digest: &mut Fnv| {
            let reparsed;
            let bq = match &s.parsed {
                Some(bq) => bq,
                None => {
                    reparsed = m7grammar::query(&s.text, &ctx).expect("parse");
                    &reparsed
                }
            };
            let hits = search_boolean_query_multi_segment_with_similarity(
                &segments,
                black_box(bq),
                &norms,
                s.top,
                s.sim.as_ref(),
            )
            .expect("search");
            for h in &hits {
                digest.add_hit(h.doc_id, h.score);
            }
        };
        if !selected(&case[..]) {
            continue;
        }
        let mut digest = Fnv::new();
        for s in searches {
            run_one(s, &mut digest);
        }
        check(case, digest, searches.len());
        measure(case, w, m, || {
            let mut sink = Fnv::new();
            for s in searches {
                run_one(s, &mut sink);
            }
            black_box(sink.0);
            searches.len() as u64
        });
    }
}

/// `searches.tsv` of a `GenM7Queries` index: `(similarity, query, top)`.
fn fixture_lines(name: &str, top: usize) -> Vec<(String, String, usize)> {
    let text = std::fs::read_to_string(format!("fixtures/data/{name}/searches.tsv"))
        .expect("run from the repository root");
    text.lines()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), f[1].to_string(), top)
        })
        .collect()
}

/// Every search `GenM7Queries` records, over its three fixture indexes, by
/// family: small indexes (180, 2,500 and 17,000 documents), so these weigh
/// a query's fixed costs -- rewrite, statistics, scorer construction -- as
/// much as its per-document loop. The KNN, vector-threshold, index-sort and
/// multi-dimensional point families exist only here (the corpus has no
/// vectors, no index sort and one-dimensional points).
pub fn bench_m7_fixture(w: Duration, m: Duration) {
    for (prefix, name, top, qvec) in [
        (
            "fx",
            "m7_queries_index",
            20,
            &m7grammar::QVEC[..],
        ),
        ("fxknn", "m7_knn_index", 50, &QVEC8[..]),
        ("fxdv", "m7_dv_index", 50, &m7grammar::QVEC[..]),
    ] {
        let dir = std::path::PathBuf::from(format!("fixtures/data/{name}"));
        run_m7_cases(w, m, prefix, &dir, &fixture_lines(name, top), qvec);
    }
}

/// `GenM7Queries.QVEC8`.
const QVEC8: [f32; 8] = [0.1, -0.3, 0.25, 0.6, -0.05, 0.4, -0.2, 0.15];

/// `M7Micro.CORPUS_QUERIES`: the M7 families the benchmark corpus can run
/// (`body`/`title` text, `keyword` SORTED doc values, `num` 1-D long points
/// and NUMERIC doc values), at a million documents, in `GenM7Queries`'
/// grammar.
pub const CORPUS_QUERIES: [&str; 47] = [
    "S body 2 t1 1 t2 1",
    "S body 3 tz 1 t2s 0.5 t1z4 1",
    "B 1 1 0 T body t0 S body 2 t3 1 t11 0.5",
    "CF t1 2 body 1 title 1",
    "CF tz 2 body 1 title 3",
    "B 0 2 0 CF t3 2 body 1 title 2 T body t4",
    "PP body 0 2 t0 0 t1 2",
    "PP body 1 3 t0 0 t1 1 t2 3",
    "NG 2 body 0 3 t0 t1 t2",
    "MP body 0 2 2 t0 t1 0 1 t2 1",
    "MP body 1 2 1 t1 0 2 t0 t3 1",
    "F body t123 1 0",
    "F title t1z4 2 1",
    "PF body t4a sb",
    "PF body t4a csbool",
    "PF body t1 tts:50",
    "PF body t1 ttb:50",
    "PF body t1 ttbf:50",
    "PF body t1 cs",
    "PF body t1 csb",
    "W body t?3 csb",
    "RE body t[0-2]. tts:20",
    "R body t10 t20 1 0 csb",
    "R body t4a t4b 1 0 sb",
    "A body t3.* csb",
    "A body (t1|t2)5? sb",
    "PF keyword t1z dv",
    "R keyword t10 t11 1 1 dv",
    "BT bool 3 body t1 1 body t2 1 title t1 1",
    "BT dismax:0.1 2 body tz 1 title tz 1",
    "IA 2 T body t1 T body t2",
    "IA 2 T body tz T body t2s",
    "LO 0.5 - - 2 T body t1 T title t1",
    "LO 1 0.7,0.3 - 2 T body tz T title t2",
    "BY 1.5 1 0 T body t2",
    "BY 0.5 2 0.1 B 0 2 0 T body t1 T title t1",
    "NR num 1000 2000",
    "B 1 0 0 T body t1 F2 NR num 0 100000",
    "B 1 0 0 T body tz F2 NR num 0 500000",
    "LR num 1000 2000",
    "LS num 5 7 99 1000 123456 777777",
    "B 1 0 0 T body t2 F2 LR num 0 300000",
    "IODV LR num 0 100000 NR num 0 100000",
    "B 1 0 0 T body t2s F2 IODV LR num 0 500000 NR num 0 500000",
    "B 1 0 0 T body t1z4 F2 IODV LR num 0 900000 NR num 0 900000",
    "B 0 2 0 IA 2 T body t1 T body t2 T title t3",
    "B 0 2 0 BY 1.5 1 0 T body t2 T body t5",
];

/// The M7 families over the million-document benchmark corpus, top 10 under
/// BM25.
pub fn bench_m7_corpus(w: Duration, m: Duration, index: &str) {
    let lines: Vec<(String, String, usize)> = CORPUS_QUERIES
        .iter()
        .map(|q| ("bm25".to_string(), q.to_string(), 10))
        .collect();
    run_m7_cases(w, m, "c", std::path::Path::new(index), &lines, &[]);
}

/// `M7Micro.SIM_QUERIES`, in `GenSimilaritySearch`' grammar.
const SIM_QUERIES: [&str; 6] = [
    "T body t1",
    "T body tz",
    "B 0 2 0 T body t1 T body t2",
    "B 2 0 0 T body t0 T body tz",
    "B 0 3 0 T body tz T body t2s T title t1",
    "P body 0 2 t0 t1",
];

/// `M7Micro.SPAN_QUERIES`: spans as scorer-tree clauses.
const SPAN_QUERIES: [&str; 4] = [
    "N 0 1 2 S body t0 S body t1",
    "N 3 0 2 S body t2 S body t0",
    "O 2 S body tz S body t2s",
    "B 0 2 0 N 1 0 2 S body t4 S body t0 T body t2",
];

/// `M7Micro.SPAN_CASES`: each span query alone under BM25.
const SPAN_CASES: [(&str, usize); 4] = [
    ("span_near_ordered", 0),
    ("span_near_unordered", 1),
    ("span_or", 2),
    ("span_in_boolean", 3),
];

/// `GenSimilaritySearch.sims()`' names.
const SIMS: [&str; 12] = [
    "bm25",
    "bm25_k2_b03",
    "classic",
    "boolean",
    "dfr_In_B_H2",
    "ib_LL_DF_H2",
    "dfi_saturated",
    "lmdirichlet",
    "lmjm_0.7",
    "ax_f2exp",
    "multi",
    "perfield",
];

/// Term, boolean and phrase searches under every similarity Lucene ships
/// (`IndexSearcher.setSimilarity`), and span queries under BM25 and
/// `ClassicSimilarity`, over the corpus.
pub fn bench_similarity(w: Duration, m: Duration, index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(Some).collect();
    let mut cases: Vec<(String, Arc<dyn Similarity>, &[&str])> = SIMS
        .iter()
        .map(|s| (format!("sim_{s}"), simgrammar::sim(s), &SIM_QUERIES[..]))
        .collect();
    for (name, i) in SPAN_CASES {
        cases.push((
            name.to_string(),
            Arc::new(Bm25Similarity::default()),
            &SPAN_QUERIES[i..=i],
        ));
    }
    cases.push((
        "span_classic".to_string(),
        simgrammar::sim("classic"),
        &SPAN_QUERIES[..],
    ));
    for (case, sim, texts) in &cases {
        let queries: Vec<BooleanQuery> = texts
            .iter()
            .enumerate()
            .filter(|&(i, _)| query_selected(i))
            .map(|(_, t)| simgrammar::query(t))
            .collect();
        let run = |digest: &mut Fnv| {
            for q in &queries {
                let hits = search_boolean_query_multi_segment_with_similarity(
                    &segments,
                    black_box(q),
                    &norms,
                    10,
                    sim.as_ref(),
                )
                .expect("search");
                for h in &hits {
                    digest.add_hit(h.doc_id, h.score);
                }
            }
        };
        if !selected(&case[..]) {
            continue;
        }
        let mut digest = Fnv::new();
        run(&mut digest);
        check(case, digest, queries.len());
        measure(case, w, m, || {
            let mut sink = Fnv::new();
            run(&mut sink);
            black_box(sink.0);
            queries.len() as u64
        });
    }
}

/// The `sorted` rows of `benchmarks/queries.tsv` (`GenSortedSearch`'s
/// grammar), each its own case: `TopFieldCollectorManager(sort, 10, null,
/// 1000)`, so the comparators' competitive iterators prune as they do in
/// Lucene.
pub fn bench_sort_pruning(w: Duration, m: Duration, index: &str) {
    let dir = MmapDirectory::open(index.to_string());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let mut opened = reader.open_segments().expect("open segments");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<Option<&std::collections::HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(Some).collect();
    let text = std::fs::read_to_string("benchmarks/queries.tsv").expect("queries.tsv");
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 || f[1] != "sorted" {
            continue;
        }
        let case = format!("sort_{}", f[0]);
        let bq = sexpr::root(sexpr::parse(f[2], f[3]));
        let sort = sexpr::sort(f[4]);
        let run = |digest: &mut Fnv| {
            let top = lucene_search::top_field::search_sorted(
                &segments,
                reader.segment_readers(),
                black_box(&bq),
                &norms,
                &sort,
                10,
                1000,
                None,
            )
            .expect("sorted");
            for h in &top.hits {
                digest.add(h.doc as u32 as u64);
            }
        };
        if !selected(&case[..]) {
            continue;
        }
        let mut digest = Fnv::new();
        run(&mut digest);
        check(&case, digest, 1);
        measure(&case, w, m, || {
            let mut sink = Fnv::new();
            run(&mut sink);
            black_box(sink.0);
            1
        });
    }
}

/// `M7Micro.queryTexts`: 2,000 query strings of 1-8 words over the corpus
/// vocabulary, some capitalised, some punctuated.
fn query_texts() -> Vec<String> {
    let mut r = super::Rng(0x5EED_0F00_D15E_A5E5);
    (0..2000)
        .map(|_| {
            let words = 1 + (r.next() % 8) as usize;
            let mut s = String::new();
            for wi in 0..words {
                if wi > 0 {
                    s.push(' ');
                }
                let x = r.next();
                let mut word = format!(
                    "t{}",
                    lucene_util::base36::to_base36(((x >> 8) % 5000) as i64)
                );
                if x & 7 == 0 {
                    word = word.to_uppercase();
                }
                s.push_str(&word);
                if x & 15 == 1 {
                    s.push(',');
                }
            }
            s
        })
        .collect()
}

/// `QueryBuilder` over `StandardAnalyzer`: the boolean (`SHOULD` and
/// `MUST`), phrase (exact and sloppy) and minimum-should-match forms, each
/// query's `toString("body")` digested.
pub fn bench_query_builder(w: Duration, m: Duration) {
    use lucene_analysis::{Analyzer, StandardAnalyzer};
    use lucene_search::query_visitor::Occur;
    use lucene_search::query_builder::{to_query_string, QueryBuilder};
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let qb = QueryBuilder::new(&analyzer);
    let texts = query_texts();
    type Build<'q> = Box<dyn Fn(&str) -> Option<lucene_search::query::Clause> + 'q>;
    let cases: Vec<(&str, Build<'_>)> = vec![
        (
            "qb_should",
            Box::new(|t| qb.create_boolean_query("body", t).unwrap()),
        ),
        (
            "qb_must",
            Box::new(|t| qb.create_boolean_query_with("body", t, Occur::Must).unwrap()),
        ),
        (
            "qb_phrase",
            Box::new(|t| qb.create_phrase_query("body", t).unwrap()),
        ),
        (
            "qb_phrase_slop",
            Box::new(|t| qb.create_phrase_query_with_slop("body", t, 2).unwrap()),
        ),
        (
            "qb_min_should_match",
            Box::new(|t| qb.create_min_should_match_query("body", t, 0.5).unwrap()),
        ),
    ];
    for (case, build) in &cases {
        if !selected(&case[..]) {
            continue;
        }
        let mut digest = Fnv::new();
        for t in &texts {
            digest.add_bytes(to_query_string(build(t).as_ref(), "body").as_bytes());
        }
        check(case, digest, texts.len());
        measure(case, w, m, || {
            for t in &texts {
                black_box(build(black_box(t)));
            }
            texts.len() as u64
        });
    }
}

/// `GenStoredFieldsDeflate`'s documents, from its LCG, with the field
/// numbers `M7Micro.storedDocs` assigns: text 0, blob 1, run 2, num 3, big 4.
fn stored_docs() -> Vec<lucene_codecs::stored_fields::Document> {
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
    const WORDS: [&str; 28] = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "labore",
        "magna",
        "aliqua",
        "lucene",
        "rust",
        "segment",
        "merge",
    ];
    let mut seed: u64 = 20_260_930;
    let mut next = |bound: u64| -> u64 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % bound
    };
    (0..500usize)
        .map(|i| {
            let mut fields = Vec::new();
            let kind = if i >= 497 { 0 } else { next(10) };
            let words = if kind == 0 {
                1 + next(3)
            } else if i.is_multiple_of(83) {
                5000 + next(2000)
            } else {
                20 + next(400)
            };
            let mut text = String::new();
            for _ in 0..words {
                text.push_str(WORDS[next(WORDS.len() as u64) as usize]);
                text.push_str(if next(5) == 0 { ". " } else { " " });
            }
            fields.push(StoredField {
                field_number: 0,
                value: FieldValue::String(text),
            });
            let blob_len = if i % 101 == 50 {
                20_000 + next(5000)
            } else {
                next(if kind == 1 { 1200 } else { 40 })
            };
            let blob: Vec<u8> = (0..blob_len).map(|_| next(256) as u8).collect();
            fields.push(StoredField {
                field_number: 1,
                value: FieldValue::Binary(blob),
            });
            if kind == 2 {
                let len = next(2000) as usize;
                let byte = b'a' + next(3) as u8;
                fields.push(StoredField {
                    field_number: 2,
                    value: FieldValue::Binary(vec![byte; len]),
                });
            }
            fields.push(StoredField {
                field_number: 3,
                value: FieldValue::Int(next(1_000_000) as i32 - 500_000),
            });
            if i == 496 {
                let sentences: Vec<String> = (0..40)
                    .map(|_| {
                        let mut s = String::new();
                        for _ in 0..12 {
                            s.push_str(WORDS[next(WORDS.len() as u64) as usize]);
                            s.push(' ');
                        }
                        s.push_str(". ");
                        s
                    })
                    .collect();
                let mut big = String::new();
                for _ in 0..9000 {
                    big.push_str(&sentences[next(40) as usize]);
                }
                fields.push(StoredField {
                    field_number: 4,
                    value: FieldValue::String(big),
                });
            }
            Document { fields }
        })
        .collect()
}

/// A segment's stored fields written in each mode: `BEST_SPEED` (LZ4) and
/// `BEST_COMPRESSION` (preset-dictionary DEFLATE, this port's zlib against
/// the JDK's native one). ns per document; the digest is the written
/// `.fdt`/`.fdx`, which must be Lucene's byte for byte.
pub fn bench_stored_fields_write(w: Duration, m: Duration) {
    use lucene_codecs::stored_fields::{write_best_compression, write_best_speed};
    let docs = stored_docs();
    let id = [7u8; 16];
    type Write = fn(
        &[lucene_codecs::stored_fields::Document],
        &[u8; 16],
        &str,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>);
    let cases: [(&str, Write); 2] = [
        ("sf_write_best_speed", write_best_speed),
        ("sf_write_best_compression", write_best_compression),
    ];
    for (case, write) in cases {
        let (fdt, fdx, _) = write(&docs, &id, "");
        if !selected(&case[..]) {
            continue;
        }
        let mut digest = Fnv::new();
        digest.add(crc32fast::hash(&fdt) as u64);
        digest.add(crc32fast::hash(&fdx) as u64);
        eprintln!("{case}: {} docs, .fdt {} bytes", docs.len(), fdt.len());
        check(case, digest, docs.len());
        measure(case, w, m, || {
            black_box(write(black_box(&docs), &id, ""));
            docs.len() as u64
        });
    }
}
