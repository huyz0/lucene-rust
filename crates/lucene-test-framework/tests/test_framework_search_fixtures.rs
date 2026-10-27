//! Differential test against `GenTestFrameworkSearch`
//! (`fixtures/testframework/data/search.txt`): the similarity bases'
//! random scoring and bulk scoring, `RandomSimilarity`'s picks, the
//! explanation index's top docs, score bits and explanations, and the
//! search-equivalence corpus, each equal to what Lucene 10.5.0 gives on the
//! same `java.util.Random` streams.

mod explanation_dsl;
mod similarity_cases;

use std::fmt::Write as _;

use explanation_dsl::{
    a, boost, bq, bq_msm, csq, dmq, match_all, mpq, phrase, synonym, t, F, M, N, S,
};
use lucene_search::misc::SweetSpotSimilarity;
use lucene_search::query::{Clause, TermQuery};
use lucene_search::similarities::{
    AfterEffect, BasicModel, Bm25Similarity, ClassicSimilarity, DfrSimilarity, Normalization,
    Similarity,
};
use lucene_test_framework::search::base_explanation::{top_level, BaseExplanation};
use lucene_test_framework::search::search_equivalence::SearchEquivalence;
use lucene_test_framework::search::similarities::{
    bulk_score, java_shuffle, random_scoring, Knobs, RandomSimilarity, ScoringCase,
};
use lucene_util::java_random::JavaRandom;

const DATA: &str = include_str!("../../../fixtures/testframework/data/search.txt");

/// `atLeast`/`rarely` as the twin runs them: multiplier 1, not nightly.
const KNOBS: Knobs = Knobs {
    nightly: false,
    multiplier: 1,
};

fn lines(prefix: &str) -> impl Iterator<Item = Vec<&'static str>> + '_ {
    DATA.lines()
        .filter(move |l| l.split(' ').next() == Some(prefix))
        .map(|l| l.split(' ').skip(1).collect())
}

/// FNV-1a over 64-bit values, as the twin's `feed`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn feed(&mut self, v: i64) {
        self.0 = (self.0 ^ v as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }

    /// `(long) Float.floatToRawIntBits(f)`: sign-extended.
    fn feed_f(&mut self, f: f32) {
        self.feed(i64::from(f.to_bits() as i32));
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

fn bits(f: f32) -> String {
    format!("{:08x}", f.to_bits())
}

fn feed_case(h: &mut Fnv, c: &ScoringCase) {
    for v in [
        c.corpus.max_doc,
        c.corpus.doc_count,
        c.corpus.sum_total_term_freq,
        c.corpus.sum_doc_freq,
        c.term.doc_freq,
        c.term.total_term_freq,
    ] {
        h.feed(v);
    }
    h.feed_f(c.boost);
    h.feed_f(c.freq);
    h.feed(i64::from(c.norm));
    for f in [c.max_score, c.score, c.prev_freq, c.prev_score] {
        h.feed_f(f);
    }
    for o in [c.prev_norm_score, c.prev_term_score] {
        match o {
            None => h.feed(0),
            Some(f) => {
                h.feed(1);
                h.feed_f(f);
            }
        }
    }
}

fn verbatim(c: &ScoringCase) -> String {
    let opt = |o: Option<f32>| o.map_or("-".to_string(), bits);
    format!(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        c.corpus.max_doc,
        c.corpus.doc_count,
        c.corpus.sum_total_term_freq,
        c.corpus.sum_doc_freq,
        c.term.doc_freq,
        c.term.total_term_freq,
        bits(c.boost),
        bits(c.freq),
        c.norm,
        bits(c.max_score),
        bits(c.score),
        bits(c.prev_freq),
        bits(c.prev_score),
        opt(c.prev_norm_score),
        opt(c.prev_term_score)
    )
}

/// Every `BaseSimilarityTestCase` subclass's `testRandomScoring` and
/// `testBulkScore`: the same cases, the same score bits, the same
/// `SimScorer.explain` text.
#[test]
fn similarity_scores_are_javas_bit_for_bit() {
    let sims: Vec<_> = lines("sim").collect();
    assert_eq!(sims.len(), similarity_cases::CASES.len() * 4);
    for sim in sims {
        let (name, kind, seed, count, hash) = (sim[0], sim[1], sim[2], sim[3], sim[4]);
        let get = similarity_cases::get(name);
        let seed_n: i64 = seed.parse().expect("seed");
        let mut random = JavaRandom::new(seed_n);
        let mut h = Fnv::new();
        let n = if kind == "scoring" {
            let want: Vec<String> = lines("case")
                .filter(|c| c[0] == name && c[1] == seed)
                .map(|c| c[3..].join(" "))
                .collect();
            let want_explanations: Vec<String> = lines("sexp")
                .filter(|c| c[0] == name && c[1] == seed)
                .map(|c| c[3..].join(" "))
                .collect();
            let mut explanations = Vec::new();
            let mut eh = Fnv::new();
            let mut n = 0;
            random_scoring(&mut random, KNOBS, get, |c| {
                if n < want.len() {
                    assert_eq!(verbatim(c), want[n], "{name} seed {seed} case {n}");
                }
                feed_case(&mut h, c);
                for text in &c.explanations {
                    if explanations.len() < want_explanations.len() {
                        explanations.push(text.trim().replace('\n', " | "));
                    }
                    for &b in text.as_bytes() {
                        eh.feed(i64::from(b));
                    }
                }
                n += 1;
            });
            assert_eq!(explanations, want_explanations, "{name} seed {seed}");
            let simexp = lines("simexp")
                .find(|l| l[0] == name && l[1] == seed)
                .expect("a simexp line");
            assert_eq!(eh.hex(), simexp[3], "{name} seed {seed}: explanation hash");
            n
        } else {
            let batches = bulk_score(&mut random, KNOBS, get);
            for b in &batches {
                h.feed(i64::try_from(b.len()).expect("small"));
                for &s in b {
                    h.feed_f(s);
                }
            }
            batches.len()
        };
        assert_eq!(n.to_string(), count, "{name} {kind} seed {seed}: count");
        assert_eq!(h.hex(), hash, "{name} {kind} seed {seed}: hash");
    }
}

/// `Collections.shuffle` and `RandomSimilarity.get`'s per-field picks.
#[test]
fn random_similarity_picks_are_javas() {
    let want = lines("shuffle").next().expect("a shuffle line")[0];
    let mut v: Vec<i32> = (0..10).collect();
    java_shuffle(&mut v, &mut JavaRandom::new(42));
    let got = format!(
        "[{}]",
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    assert_eq!(got, want);

    const FIELDS: [&str; 14] = [
        "field", "body", "title", "id", "KEY", "alt", "f0", "f1", "f2", "f3", "f4", "f5", "f6",
        "f7",
    ];
    let mut n = 0;
    for l in lines("rsim") {
        let sim = RandomSimilarity::new(&mut JavaRandom::new(l[0].parse().expect("seed")));
        let picks: Vec<String> = FIELDS
            .iter()
            .map(|f| sim.all_sims_index(f).to_string())
            .collect();
        assert_eq!(picks.join(","), l[1], "seed {}", l[0]);
        n += 1;
    }
    assert_eq!(n, 20);
}

/// The twin's explanation queries, by name.
fn explanation_query(name: &str) -> Clause {
    match name {
        "t_w1" => t("w1"),
        "t_xx" => t("xx"),
        "bq_must" => bq(vec![(M, t("w1")), (M, t("w2"))]),
        "bq_should" => bq(vec![(S, t("w1")), (S, t("w5")), (S, t("xx"))]),
        "bq_mixed" => bq(vec![(M, t("w1")), (N, t("xx")), (S, t("w2")), (S, a("w3"))]),
        "bq_msm" => bq_msm(
            2,
            vec![(S, t("w1")), (S, t("w2")), (S, t("w5")), (S, t("zz"))],
        ),
        "bq_filter" => bq(vec![(F, t("xx")), (S, t("w3"))]),
        "bq_dup" => bq(vec![(S, t("w1")), (S, boost(t("w1"), 1000.0))]),
        "bq_dup_nested" => bq(vec![
            (S, explanation_query("bq_dup")),
            (S, Clause::Term(TermQuery::new("NEVER", "MATCH"))),
        ]),
        "dmq8" => dmq(
            vec![
                bq(vec![(S, t("yy")), (S, boost(t("w5"), 100.0))]),
                boost(t("xx"), 100_000.0),
            ],
            0.5,
        ),
        "boost" => boost(t("w2"), 1000.0),
        "phrase" => phrase(0, &["w1", "w2"]),
        "phrase_sloppy" => phrase(2, &["w1", "w3"]),
        "dmq" => dmq(vec![t("w1"), a("w5"), t("xx")], 0.5),
        "csq" => csq(t("w1")),
        "match_all" => match_all(),
        "mpq" => mpq(0, &[&["w1"], &["w2", "w3"]]),
        "mpq_xx" => mpq(0, &[&["w1"], &["xx", "QQ"]]),
        "phrase_xx" => phrase(0, &["w1", "xx"]),
        "synonym" => synonym(&["w2", "w3"]),
        other => panic!("no explanation query {other}"),
    }
}

/// How much of a hit's explanation is compared with Java's.
#[derive(Clone, Copy, PartialEq)]
enum Compared {
    /// Every line.
    All,
    /// The top line's value: the ledger's "multi-term, span flat".
    Top,
}

fn compared_part(_sim_name: &str, q_name: &str) -> Compared {
    match q_name {
        "mpq" | "synonym" => Compared::Top,
        _ => Compared::All,
    }
}

/// The twin's similarities, by the names its `exp` rows carry.
fn explanation_similarity(name: &str) -> Option<Box<dyn Similarity>> {
    match name {
        "BM25Similarity" => None,
        "ClassicSimilarity" => Some(Box::new(ClassicSimilarity::default())),
        "BM25_k1.5_b0.5" => Some(Box::new(
            Bm25Similarity::new(1.5, 0.5, true).expect("valid parameters"),
        )),
        "DFR_In_L_H2" => Some(Box::new(
            DfrSimilarity::new(BasicModel::In, AfterEffect::L, Normalization::H2_DEFAULT)
                .expect("valid parameters"),
        )),
        "SweetSpot_3_5" => {
            let mut s = SweetSpotSimilarity::default();
            s.set_length_norm_factors(3, 5, 0.5);
            Some(Box::new(s))
        }
        other => panic!("no explanation similarity {other}"),
    }
}

/// One line of a hit's explanation as [`compared_part`] says (`None`: not
/// compared).
fn compared(part: Compared, line: &str, seen_top: &mut bool) -> Option<String> {
    if part == Compared::All || line.starts_with("hit ") {
        *seen_top = false;
        return Some(line.to_string());
    }
    if line.starts_with("expl ") || std::mem::replace(seen_top, true) {
        return None;
    }
    line.split(" = ").next().map(str::to_string)
}

/// `text`'s lines with the details of every `max ... of:` node sorted by
/// their text, as `GenTestFrameworkSearch.canonicalMaxOrder` records Java's:
/// `DisjunctionMaxQuery`'s disjunct order follows identity hashes there.
fn canonical_max_order(text: &str) -> Vec<String> {
    fn indent(line: &str) -> usize {
        line.len() - line.trim_start_matches(' ').len()
    }
    fn node(lines: &[&str], at: usize, out: &mut Vec<String>) -> usize {
        let depth = indent(lines[at]);
        let mut end = at + 1;
        let mut children = Vec::new();
        while end < lines.len() && indent(lines[end]) > depth {
            let mut child = Vec::new();
            end = node(lines, end, &mut child);
            children.push(child);
        }
        let head = lines[at].trim();
        let is_max = head.ends_with(" times others of:") && head.contains(" = max plus ")
            || head.ends_with(" = max of:");
        if is_max {
            children.sort_by_key(|c: &Vec<String>| c.join("\n"));
        }
        out.push(lines[at].to_string());
        out.extend(children.into_iter().flatten());
        end
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        at = node(&lines, at, &mut out);
    }
    out
}

/// `BaseExplanationTestCase`'s index under BM25, `ClassicSimilarity` and
/// (some queries) BM25 with its own `k1` and `b`, DFR and
/// `SweetSpotSimilarity` with a document outside its sweet spot: the same
/// hits, score bits and explanation text.
#[test]
fn explanations_are_javas() {
    let mut want = String::new();
    let mut got = String::new();
    let mut l = DATA
        .lines()
        .skip_while(|l| !l.starts_with("exp "))
        .peekable();
    while let Some(head) = l.next_if(|l| l.starts_with("exp ")) {
        writeln!(want, "{head}").expect("write");
        let parts: Vec<&str> = head.split(' ').collect();
        let (sim_name, q_name) = (parts[1], parts[2]);
        let part = compared_part(sim_name, q_name);
        let mut seen_top = false;
        while let Some(line) = l
            .next_if(|l| !l.starts_with("exp ") && !l.starts_with("miss ") && !l.starts_with("eq "))
        {
            if let Some(line) = compared(part, line, &mut seen_top) {
                writeln!(want, "{line}").expect("write");
            }
        }
        let mut base = BaseExplanation::new(&mut JavaRandom::new(0));
        if let Some(sim) = explanation_similarity(sim_name) {
            base = base.with_similarity(sim);
        }
        let q = top_level(explanation_query(q_name));
        base.with_searcher(|s| {
            let td = s.search(&q, 10).expect("search");
            writeln!(got, "exp {sim_name} {q_name} {}", td.total_hits.value).expect("write");
            for sd in &td.score_docs {
                let text = s.explain(&q, sd.doc).expect("explain").to_string();
                let text = canonical_max_order(text.trim_end_matches('\n'));
                let mut lines = vec![
                    format!("hit {} {}", sd.doc, bits(sd.score)),
                    format!("expl {}", text.len()),
                ];
                lines.extend(text.iter().map(|line| format!("| {line}")));
                let mut seen_top = false;
                for line in lines {
                    if let Some(line) = compared(part, &line, &mut seen_top) {
                        writeln!(got, "{line}").expect("write");
                    }
                }
            }
        });
    }
    assert!(!want.is_empty());
    let (w, g): (Vec<&str>, Vec<&str>) = (want.lines().collect(), got.lines().collect());
    if let Some(i) = w.iter().zip(&g).position(|(w, g)| w != g) {
        // the query's header and its lines on both sides, to locate it
        let from = w[..=i]
            .iter()
            .rposition(|l| l.starts_with("exp "))
            .unwrap_or(0);
        let to = (i + 20).min(w.len()).min(g.len());
        panic!(
            "line {i} of the explanation section differs\n--- Java\n{}\n--- Rust\n{}",
            w[from..to].join("\n"),
            g[from..to].join("\n")
        );
    }
    assert_eq!(g.len(), w.len());
}

/// `PhraseWeight.explain`'s two no-matches on the documents a phrase and a
/// multi-phrase miss: "no matching terms" where a term is absent, "no
/// matching phrase" where the positions do not line up.
#[test]
fn phrase_misses_explain_as_javas() {
    let mut n = 0;
    for l in lines("miss") {
        let (sim_name, q_name, doc) = (l[0], l[1], l[2].parse::<i32>().expect("doc"));
        let want = l[3..].join(" ");
        let mut base = BaseExplanation::new(&mut JavaRandom::new(0));
        if let Some(sim) = explanation_similarity(sim_name) {
            base = base.with_similarity(sim);
        }
        let q = top_level(explanation_query(q_name));
        let got = base.with_searcher(|s| s.explain(&q, doc).expect("explain").to_string());
        assert_eq!(
            got.lines().next().expect("a line"),
            want,
            "{sim_name} {q_name} doc {doc}"
        );
        n += 1;
    }
    assert_eq!(n, 16);
}

/// `SearchEquivalenceTestBase`'s corpus: the same stop letter, documents
/// and deletes, and BM25 top docs over it bit for bit.
#[test]
fn search_equivalence_corpus_is_javas() {
    let mut l = DATA
        .lines()
        .skip_while(|l| !l.starts_with("eq "))
        .peekable();
    let mut corpora = 0;
    while let Some(head) = l.next() {
        let parts: Vec<&str> = head.split(' ').collect();
        assert_eq!(parts[0], "eq");
        let seed: i64 = parts[1].parse().expect("seed");
        let s = SearchEquivalence::new(&mut JavaRandom::new(seed), false, 1);
        assert_eq!(s.stopword().to_string(), parts[2], "seed {seed}");
        assert_eq!(s.num_docs().to_string(), parts[3], "seed {seed}");
        for (i, text) in s.contents().iter().enumerate() {
            assert_eq!(
                l.next().expect("a doc line"),
                format!("doc {i} [{text}]"),
                "seed {seed}"
            );
        }
        let deleted: Vec<String> = s.deleted().iter().map(ToString::to_string).collect();
        assert_eq!(
            l.next().expect("a del line"),
            format!("del [{}]", deleted.join(","))
        );
        while let Some(hits) = l.next_if(|x| x.starts_with("eqhits ")) {
            let term = hits.split(' ').nth(1).expect("a term");
            let q = top_level(Clause::Term(TermQuery::new("field", term)));
            let got = s.with_searcher(|searcher| {
                let td = searcher.search(&q, 5).expect("search");
                let mut out = format!("eqhits {term} {}", td.total_hits.value);
                for sd in &td.score_docs {
                    write!(out, " {}:{}", sd.doc, bits(sd.score)).expect("write");
                }
                out
            });
            assert_eq!(got, hits, "seed {seed}");
        }
        corpora += 1;
    }
    assert_eq!(corpora, 4);
}

/// The default similarity names itself as Java's class does, which the
/// explanation section's headers rely on.
#[test]
fn classic_is_named_as_java() {
    assert_eq!(
        ClassicSimilarity::default().simple_name(),
        "ClassicSimilarity"
    );
}
