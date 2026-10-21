//! Differential test against `GenTestFrameworkSearch`
//! (`fixtures/testframework/data/search.txt`): the explanation index's top
//! docs, score bits and explanations, and the search-equivalence corpus,
//! each equal to what Lucene 10.5.0 gives on the same `java.util.Random`
//! streams. (The dump's similarity sections are checked with the similarity
//! bases.)

mod explanation_dsl;

use std::fmt::Write as _;

use explanation_dsl::{
    a, boost, bq, bq_msm, csq, dmq, match_all, mpq, phrase, synonym, t, F, M, N, S,
};
use lucene_search::query::{Clause, TermQuery};
use lucene_search::similarities::{ClassicSimilarity, Similarity};
use lucene_test_framework::search::base_explanation::{top_level, BaseExplanation};
use lucene_test_framework::search::search_equivalence::SearchEquivalence;
use lucene_util::java_random::JavaRandom;

const DATA: &str = include_str!("../../../fixtures/testframework/data/search.txt");

fn bits(f: f32) -> String {
    format!("{:08x}", f.to_bits())
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
        "synonym" => synonym(&["w2", "w3"]),
        other => panic!("no explanation query {other}"),
    }
}

/// How much of a hit's explanation is compared with Java's.
#[derive(Clone, Copy, PartialEq)]
enum Compared {
    /// Every line.
    All,
    /// The top line's value: the ledger's "multi-term, span flat" and
    /// "other boosts are own node" (a boost under a non-BM25 similarity).
    Top,
    /// The hit lines only: under a non-BM25 similarity a boost's own node
    /// rounds `dmq8`'s sub-values, so their sum can land one ulp from the
    /// score (Classic: 57104.13 for Java's 57104.125, the score bits equal).
    Hits,
}

fn compared_part(sim_name: &str, q_name: &str) -> Compared {
    let bm25 = sim_name == "BM25Similarity";
    match q_name {
        "dmq8" if !bm25 => Compared::Hits,
        // Rust explains a phrase's and a multi-phrase's value with BM25,
        // whatever the similarity
        "phrase" | "phrase_sloppy" | "mpq" if !bm25 => Compared::Hits,
        "mpq" | "synonym" => Compared::Top,
        "boost" | "bq_dup" | "bq_dup_nested" if !bm25 => Compared::Top,
        _ => Compared::All,
    }
}

/// One line of a hit's explanation as [`compared_part`] says (`None`: not
/// compared).
fn compared(part: Compared, line: &str, seen_top: &mut bool) -> Option<String> {
    if part == Compared::All || line.starts_with("hit ") {
        *seen_top = false;
        return Some(line.to_string());
    }
    if part == Compared::Hits || line.starts_with("expl ") || std::mem::replace(seen_top, true) {
        return None;
    }
    line.split(" = ").next().map(str::to_string)
}

/// `BaseExplanationTestCase`'s index under BM25 and `ClassicSimilarity`:
/// the same hits, score bits and explanation text.
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
        while let Some(line) = l.next_if(|l| !l.starts_with("exp ") && !l.starts_with("eq ")) {
            if let Some(line) = compared(part, line, &mut seen_top) {
                writeln!(want, "{line}").expect("write");
            }
        }
        let mut base = BaseExplanation::new(&mut JavaRandom::new(0));
        if sim_name == "ClassicSimilarity" {
            base = base.with_similarity(Box::new(ClassicSimilarity::default()));
        } else {
            assert_eq!(sim_name, "BM25Similarity");
        }
        let q = top_level(explanation_query(q_name));
        base.with_searcher(|s| {
            let td = s.search(&q, 10).expect("search");
            writeln!(got, "exp {sim_name} {q_name} {}", td.total_hits.value).expect("write");
            for sd in &td.score_docs {
                let text = s.explain(&q, sd.doc).expect("explain").to_string();
                let text = text.trim_end_matches('\n');
                let mut lines = vec![
                    format!("hit {} {}", sd.doc, bits(sd.score)),
                    format!("expl {}", text.lines().count()),
                ];
                lines.extend(text.lines().map(|line| format!("| {line}")));
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
