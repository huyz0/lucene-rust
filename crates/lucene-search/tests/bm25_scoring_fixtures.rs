//! **Cross-engine BM25 ground truth** for the term / boolean / phrase scoring
//! paths: `fixtures/src/AppendScoringManifest.java` opens the checked-in
//! `fixtures/data/blocktree_index/` segment read-only through a real
//! `DirectoryReader`/`IndexSearcher` and records real Lucene 10.5.0's own
//! `TopDocs` `(doc, score)` pairs -- as decimal *and* as raw `Float.floatToIntBits`
//! -- into `manifest.properties`' `scoring.*` keys. Every test here re-runs the
//! equivalent query through this port and compares against those recorded
//! values.
//!
//! Why this file exists alongside `scoring_fixtures.rs`: that file's assertions
//! are almost all *self-consistency* checks (it re-derives BM25 in the test and
//! compares against this port's own output, with `norms: None`), and
//! `dismax_query_fixtures.rs`'s one cross-engine test compares to a `1e-4`
//! tolerance. Neither can catch a divergence smaller than `1e-4`, and neither
//! covers the plain term / boolean / sloppy-phrase paths at all. The sweep that
//! added this file found a real scoring bug that way (`slop > 0` scored every
//! sloppy match as frequency `1`, where `SloppyPhraseMatcher.sloppyWeight()` is
//! `1 / (1 + matchLength)`), which no tolerance-based test in the crate could
//! have surfaced.

use std::collections::HashMap;

use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::{blocktree, field_infos, norms};
use lucene_search::collector::TopDocsCollector;
use lucene_search::{
    search_boolean_query_scored, search_multi_phrase_query, search_multi_phrase_query_scored,
    search_phrase_query_scored, search_term_query_scored, BooleanQuery, Clause, FieldNorms,
    MultiPhraseQuery, PhraseQuery, TermQuery, VecCollector,
};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenBlockTree)");
        Manifest {
            kv: text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| {
                panic!("manifest key {key} missing -- re-run scripts/gen-fixtures.sh")
            })
    }

    /// Real Lucene's recorded hits for `key`, as `(doc, score)` with the score
    /// reconstructed from its exact `Float.floatToIntBits` encoding so the
    /// comparison never goes through decimal rounding.
    fn lucene_hits(&self, key: &str) -> Vec<(i32, f32)> {
        let raw = self.get(&format!("{key}.bits"));
        if raw.is_empty() {
            return Vec::new();
        }
        raw.split(',')
            .map(|pair| {
                let (doc, bits) = pair.split_once(':').expect("doc:bits pair");
                (
                    doc.parse().unwrap(),
                    f32::from_bits(bits.parse::<u32>().unwrap()),
                )
            })
            .collect()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read_raw(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn read_plain(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{}", dir(), name)).unwrap_or_else(|_| panic!("missing {name}"))
}

struct Segment {
    fields: blocktree::BlockTreeFields,
    doc: &'static [u8],
    pos: &'static [u8],
    pay: &'static [u8],
    norms_data: &'static [u8],
    id: [u8; 16],
    suffix: String,
    max_doc: i32,
    manifest: Manifest,
}

/// Leaks the mapped-in file buffers so the returned `BlockTreeFields<'static>`
/// can be handed around freely inside one test -- the standard trick the other
/// fixture suites in this crate use for the same borrow-shape problem.
fn open_segment() -> Segment {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm: &'static [u8] = Box::leak(read_raw(m.get("fnm_file_name")).into_boxed_slice());
    let field_infos: &'static field_infos::FieldInfos = Box::leak(Box::new(
        field_infos::parse(fnm, &id, "").expect("parse .fnm"),
    ));

    let tim: &'static [u8] = Box::leak(read_raw(m.get("tim_file_name")).into_boxed_slice());
    let tip: &'static [u8] = Box::leak(read_raw(m.get("tip_file_name")).into_boxed_slice());
    let tmd: &'static [u8] = Box::leak(read_raw(m.get("tmd_file_name")).into_boxed_slice());
    let fields =
        blocktree::open(tim, tip, tmd, field_infos, &id, &suffix, max_doc).expect("open blocktree");

    Segment {
        fields,
        doc: Box::leak(read_raw(m.get("doc_file_name")).into_boxed_slice()),
        pos: Box::leak(read_raw(m.get("pos_file_name")).into_boxed_slice()),
        pay: Box::leak(read_raw(m.get("pay_file_name")).into_boxed_slice()),
        norms_data: Box::leak(read_plain("_0.nvd").into_boxed_slice()),
        id,
        suffix,
        max_doc,
        manifest: m,
    }
}

impl Segment {
    /// Opens this fixture's real `_0.nvm`/`_0.nvd` norms for one field -- real
    /// Lucene's default `IndexSearcher` always scores with real norms, so
    /// matching its recorded scores requires the same real norms here.
    fn field_norms(&self, field: &str) -> FieldNorms<'_> {
        let meta = read_plain("_0.nvm");
        let fnm = read_raw(self.manifest.get("fnm_file_name"));
        let infos = field_infos::parse(&fnm, &self.id, "").expect("parse .fnm");
        let number = infos
            .fields
            .iter()
            .find(|f| f.name == field)
            .unwrap_or_else(|| panic!("{field} field"))
            .number;
        let (_, parsed) = norms::parse_meta(&meta, &self.id, "").expect("parse .nvm");
        let entry = *parsed
            .entry(number)
            .unwrap_or_else(|| panic!("{field} has a norms entry"));
        FieldNorms::open(self.norms_data, entry, self.max_doc, None).expect("open norms")
    }

    fn norms_map(&self, field: &str) -> HashMap<String, FieldNorms<'_>> {
        let mut map = HashMap::new();
        map.insert(field.to_string(), self.field_norms(field));
        map
    }
}

/// Asserts this port's hits are real Lucene's hits, **bit for bit**, in the
/// same order.
fn assert_same_as_lucene(
    what: &str,
    got: &[lucene_search::collector::ScoreDoc],
    expected: &[(i32, f32)],
) {
    let got_pairs: Vec<(i32, f32)> = got.iter().map(|h| (h.doc_id, h.score)).collect();
    assert_eq!(
        got_pairs.len(),
        expected.len(),
        "{what}: hit count differs -- got {got_pairs:?}, Lucene {expected:?}"
    );
    for ((got_doc, got_score), (exp_doc, exp_score)) in got_pairs.iter().zip(expected) {
        assert_eq!(got_doc, exp_doc, "{what}: doc order differs");
        assert_eq!(
            got_score.to_bits(),
            exp_score.to_bits(),
            "{what}: doc {got_doc} scored {got_score} ({:#x}), real Lucene {exp_score} ({:#x})",
            got_score.to_bits(),
            exp_score.to_bits()
        );
    }
}

#[test]
fn term_query_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let norms = seg.field_norms("body");

    for (key, term) in [("scoring.term.cat", "cat"), ("scoring.term.bird", "bird")] {
        let mut top = TopDocsCollector::new(20);
        search_term_query_scored(
            &seg.fields,
            Some(&doc_in),
            None,
            &TermQuery::new("body", term),
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

#[test]
fn boolean_query_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let norms = seg.norms_map("body");

    let cases: [(&str, BooleanQuery); 2] = [
        (
            "scoring.boolean.should",
            BooleanQuery::new()
                .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
        ),
        (
            "scoring.boolean.must",
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
        ),
    ];

    for (key, query) in cases {
        let mut top = TopDocsCollector::new(20);
        search_boolean_query_scored(
            &seg.fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

/// `BooleanQuery.rewrite`'s two "Deduplicate … clauses by summing up their
/// boosts" blocks are **score-neutral**, and this is the ground truth for that
/// claim rather than an argument about BM25 being linear.
///
/// Lucene rewrites `body:cat body:cat body:dog` to `(body:cat)^2 body:dog`
/// and `+body:cat +body:cat` to `(body:cat)^2` *before* scoring; this port's
/// executor sums each duplicate clause separately
/// (`crate::clause_scores`). If the collapse moved a single bit, these two
/// entries would be where it showed — and the comparison is
/// `f32::to_bits`-exact, not a tolerance.
///
/// It is also why the dedup was worth implementing anyway: what it changes is
/// the query's *shape* (clause count, and so the explain tree and
/// `maxClauseCount` pressure), which `query.rs`'s own rewrite tests pin.
#[test]
fn duplicate_clause_dedup_is_score_neutral_against_real_lucene() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let norms = seg.norms_map("body");

    let cases: [(&str, BooleanQuery); 2] = [
        (
            "scoring.boolean.dupshould",
            BooleanQuery::new().with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "dog"),
            ]),
        ),
        (
            "scoring.boolean.dupmust",
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]),
        ),
    ];

    for (key, query) in cases {
        // Both forms are scored against the same recorded bits: the query as
        // written (which this port's executor sums clause by clause) and the
        // query as `rewrite` collapses it (one clause carrying the summed
        // boost, which is what Lucene actually scored). Checking only the
        // first would leave a wrong summed boost undetected; checking only the
        // second would not test the equivalence at all.
        let rewritten = match query.clone().rewrite() {
            Clause::Boolean(inner) => *inner,
            single => BooleanQuery::new().with_must([single]),
        };
        assert_ne!(rewritten, query, "{key}: the duplicates did not collapse");
        for (what, q) in [("as written", &query), ("rewritten", &rewritten)] {
            let mut top = TopDocsCollector::new(20);
            search_boolean_query_scored(
                &seg.fields,
                Some(&doc_in),
                None,
                None,
                None,
                None,
                q,
                Some(&norms),
                &mut top,
            )
            .unwrap();
            assert_same_as_lucene(
                &format!("{key} ({what})"),
                top.top_docs(),
                &seg.manifest.lucene_hits(key),
            );
        }
    }
}

/// The exact (`slop == 0`) phrase path: `ExactPhraseMatcher`'s frequency is the
/// number of matching start positions, and doc 8556 (`"alpha alpha"`) exercises
/// a repeated-term phrase where that count is a real count, not always `1`.
#[test]
fn exact_phrase_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");
    let norms = seg.field_norms("pos");

    let cases: [(&str, PhraseQuery); 2] = [
        (
            "scoring.phrase.exact",
            PhraseQuery::new("pos", ["alpha", "beta"]),
        ),
        (
            "scoring.phrase.repeat",
            PhraseQuery::new("pos", ["alpha", "alpha"]),
        ),
    ];
    for (key, query) in cases {
        let mut top = TopDocsCollector::new(20);
        search_phrase_query_scored(
            &seg.fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

/// The sloppy (`slop > 0`) phrase path. This is the test that pins
/// `SloppyPhraseMatcher.sloppyWeight() == 1 / (1 + matchLength)`: doc 8555 has
/// `alpha beta` adjacent (`matchLength == 0`, weight `1`) while doc 8557 has
/// them two positions apart (`matchLength == 2`, weight `1/3`), so real
/// Lucene scores them very differently even though both are single sloppy
/// matches. Scoring every sloppy match as frequency `1` -- which this port did
/// before this sweep -- gives both documents the same `tf`.
#[test]
fn sloppy_phrase_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");
    let norms = seg.field_norms("pos");

    for (key, slop) in [("scoring.phrase.slop2", 2u32), ("scoring.phrase.slop3", 3)] {
        let mut top = TopDocsCollector::new(20);
        search_phrase_query_scored(
            &seg.fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(slop),
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

/// A `Clause::Phrase` inside a `BooleanQuery` must score identically to the
/// standalone phrase search -- the boolean path reaches
/// `search_phrase_query_scored_with_stats` through `clause_scores`, a second
/// code path that could drift from the first.
#[test]
fn phrase_clause_inside_a_boolean_query_scores_like_the_standalone_phrase() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");
    let norms = seg.norms_map("pos");

    let query = BooleanQuery::new().with_should([Clause::Phrase(
        PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(2),
    )]);
    let mut top = TopDocsCollector::new(20);
    search_boolean_query_scored(
        &seg.fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &query,
        Some(&norms),
        &mut top,
    )
    .unwrap();
    assert_same_as_lucene(
        "scoring.phrase.slop2 (as a boolean SHOULD clause)",
        top.top_docs(),
        &seg.manifest.lucene_hits("scoring.phrase.slop2"),
    );
}

/// **`MultiPhraseQuery`** (`MultiPhraseWeight` + `UnionFullPostingsEnum`), the
/// three shapes that are each easy to get wrong by analogy with `PhraseQuery`:
///
/// - `union`: one position accepts two alternatives, so the matcher sees the
///   *merged* position list. Fixture doc 8555 (`alpha beta`) and doc 8556
///   (`alpha alpha`) both match `"alpha (beta|alpha)"`, and score identically,
///   which no single-term phrase query can produce.
/// - `bothslots`: alternatives at *both* positions, whose idf is the sum over
///   all four terms -- `MultiPhraseWeight` collects `TermStats` for every term
///   of every position and hands them all to `Similarity.scorer`.
/// - `single`: a one-position multi-phrase, which `MultiPhraseQuery.rewrite`
///   turns into a `BooleanQuery` of `SHOULD` `TermQuery`s -- **not** a one-slot
///   phrase over the merged union. The two disagree (per-term idf and freq,
///   summed, versus one summed idf against a merged freq), and this recorded
///   ground truth is what settles which one Lucene actually produces.
/// - `slop2`: the sloppy matcher reached through the multi-phrase path.
#[test]
fn multi_phrase_query_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");
    let norms = seg.field_norms("pos");

    let cases: [(&str, MultiPhraseQuery); 5] = [
        (
            "scoring.multiphrase.union",
            MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["beta", "alpha"]]),
        ),
        (
            "scoring.multiphrase.bothslots",
            MultiPhraseQuery::new("pos", [vec!["alpha", "delta"], vec!["beta", "gamma"]]),
        ),
        (
            "scoring.multiphrase.single",
            MultiPhraseQuery::new("pos", [vec!["alpha", "delta"]]),
        ),
        (
            "scoring.multiphrase.dup",
            MultiPhraseQuery::new("pos", [vec!["alpha", "alpha"], vec!["beta"]]),
        ),
        (
            "scoring.multiphrase.slop2",
            MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["beta"]]).with_slop(2),
        ),
    ];

    for (key, query) in cases {
        let mut top = TopDocsCollector::new(20);
        search_multi_phrase_query_scored(
            &seg.fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

/// The unscored multi-phrase path must report exactly the documents the scored
/// one does -- they run the same implementation precisely so they cannot drift,
/// and this is the test that would notice if they were ever split again.
#[test]
fn unscored_multi_phrase_matches_the_scored_paths_doc_set() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");

    for key in [
        "scoring.multiphrase.union",
        "scoring.multiphrase.bothslots",
        "scoring.multiphrase.single",
        "scoring.multiphrase.slop2",
    ] {
        let query = match key {
            "scoring.multiphrase.union" => {
                MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["beta", "alpha"]])
            }
            "scoring.multiphrase.bothslots" => {
                MultiPhraseQuery::new("pos", [vec!["alpha", "delta"], vec!["beta", "gamma"]])
            }
            "scoring.multiphrase.single" => MultiPhraseQuery::new("pos", [vec!["alpha", "delta"]]),
            _ => MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["beta"]]).with_slop(2),
        };
        let mut docs = VecCollector::default();
        search_multi_phrase_query(
            &seg.fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            &mut docs,
        )
        .unwrap();
        let mut expected: Vec<i32> = seg
            .manifest
            .lucene_hits(key)
            .into_iter()
            .map(|(doc, _)| doc)
            .collect();
        expected.sort_unstable();
        assert_eq!(docs.docs, expected, "{key}");
    }
}

/// A `Clause::MultiPhrase` nested in a `BooleanQuery` must score identically to
/// the standalone search -- the clause arm is a second path into the same
/// scorer.
#[test]
fn multi_phrase_clause_inside_a_boolean_query_scores_like_the_standalone_query() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");
    let norms = seg.norms_map("pos");

    let query = BooleanQuery::new().with_should([Clause::MultiPhrase(MultiPhraseQuery::new(
        "pos",
        [vec!["alpha", "delta"], vec!["beta", "gamma"]],
    ))]);
    let mut top = TopDocsCollector::new(20);
    search_boolean_query_scored(
        &seg.fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        &query,
        Some(&norms),
        &mut top,
    )
    .unwrap();
    assert_same_as_lucene(
        "scoring.multiphrase.bothslots (as a boolean SHOULD clause)",
        top.top_docs(),
        &seg.manifest.lucene_hits("scoring.multiphrase.bothslots"),
    );
}

/// Degenerate shapes, matching `MultiPhraseQuery.rewrite`'s own handling: an
/// empty `term_arrays` is a `MatchNoDocsQuery`, and a position whose whole
/// alternative set is absent from the segment can never be satisfied.
#[test]
fn multi_phrase_degenerate_shapes_match_nothing() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");

    let empty: [Vec<&str>; 0] = [];
    let cases = [
        MultiPhraseQuery::new("pos", empty.clone()),
        MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["nope-not-a-term"]]),
        MultiPhraseQuery::new("pos", [vec!["alpha"], vec![]]),
        MultiPhraseQuery::new("no-such-field", [vec!["alpha"], vec!["beta"]]),
    ];
    for query in cases {
        let mut docs = VecCollector::default();
        search_multi_phrase_query(
            &seg.fields,
            Some(&doc_in),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &query,
            &mut docs,
        )
        .unwrap();
        assert!(docs.docs.is_empty(), "{query:?} must match nothing");
    }
}

/// Deletions plus a scored phrase query. `read_positions_for_docs` validates
/// that the per-document frequencies it is handed sum to the term's
/// `totalTermFreq`, so a caller that filters deleted documents out of that
/// frequency list before calling it hands over a sum that no longer matches.
#[test]
fn scored_phrase_query_with_deletions_still_works() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");

    // Every doc live except 8556 ("alpha alpha"), which contains `alpha` and
    // so contributes to that term's postings and positions.
    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(seg.max_doc as usize);
    for d in 0..seg.max_doc {
        live.set(d as usize);
    }
    live.clear(8556);

    let mut top = TopDocsCollector::new(20);
    let result = search_phrase_query_scored(
        &seg.fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        Some(&live),
        &PhraseQuery::new("pos", ["alpha", "beta"]),
        None,
        &mut top,
    );
    assert!(
        result.is_ok(),
        "phrase search over a segment with deletions failed: {result:?}"
    );
    let docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
    assert_eq!(docs, vec![8555], "only doc 8555 has an adjacent alpha/beta");
}

/// The same deletion hazard as `scored_phrase_query_with_deletions_still_works`,
/// through the multi-phrase path -- it shares `positions_for_docs` and had to
/// learn the same rule.
#[test]
fn scored_multi_phrase_query_with_deletions_still_works() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let pos_in = PosInput::open(seg.pos, &seg.id, &seg.suffix).expect("open .pos");
    let pay_in = PayInput::open(seg.pay, &seg.id, &seg.suffix).expect("open .pay");

    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(seg.max_doc as usize);
    for d in 0..seg.max_doc {
        live.set(d as usize);
    }
    // 8556 is "alpha alpha" -- a real contributor to `alpha`'s postings, and a
    // hit of the query below when it is live.
    live.clear(8556);

    let mut top = TopDocsCollector::new(20);
    let result = search_multi_phrase_query_scored(
        &seg.fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        Some(&live),
        &MultiPhraseQuery::new("pos", [vec!["alpha"], vec!["beta", "alpha"]]),
        None,
        &mut top,
    );
    assert!(
        result.is_ok(),
        "multi-phrase over deletions failed: {result:?}"
    );
    let docs: Vec<i32> = top.top_docs().iter().map(|h| h.doc_id).collect();
    assert_eq!(
        docs,
        vec![8555],
        "8556 also matches this query but has been deleted"
    );
}

// ---------------------------------------------------------------------------
// `Occur.FILTER`
// ---------------------------------------------------------------------------

/// Every `Occur.FILTER` shape recorded by `AppendScoringManifest`, compared
/// bit for bit against real `IndexSearcher`.
///
/// Bit-for-bit is the only comparison that can see what these tests are for. A
/// filter clause's whole contract is that it contributes **exactly zero** to
/// the score; the two ways to break it are (a) summing it anyway, which a
/// tolerance would catch, and (b) merely *reordering* the surviving scoring
/// clauses around it, which no tolerance can catch, because `f32` addition is
/// not associative and the error is one ULP.
///
/// Each case pins one property:
///
/// - `filter` (`+body:cat #body:dog`) against `scoring.boolean.must`
///   (`+body:cat +body:dog`): the same matched set, and exactly the `dog`
///   clause's score less. This is the headline case.
/// - `filteronly` (`#body:cat #body:dog`): a query with no scoring clause at
///   all still *matches*, at score `0`. It is not a pure negative query.
/// - `filter.should` (`#body:cat body:dog`): the filter fixes the matched set,
///   the optional clause adds score only where it matches -- so doc 2 survives
///   at `0`.
/// - `filter.minshouldmatch` (`(#body:cat body:dog body:bird)~1`): filters do
///   not count toward `minimumNumberShouldMatch`, so doc 2 (matching the
///   filter, matching neither optional clause) drops out.
/// - `filter.dupmust` (`+body:cat +body:dog #body:dog`): bit-identical to
///   `scoring.boolean.must`. A filter double-counting its `MUST` twin shows up
///   here and nowhere else.
/// - `filter.nested` (`+(+body:cat #body:dog) body:dog`): a filter inside a
///   nested `BooleanQuery`, with a scoring clause at each level. Lucene rewrites
///   this to `+body:cat +body:dog`; executing the nested form directly has to
///   reach the same sum in the same order.
#[test]
fn filter_clause_scores_match_real_lucene_bit_for_bit() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let norms = seg.norms_map("body");

    let cases: [(&str, BooleanQuery); 6] = [
        (
            "scoring.boolean.filter",
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "dog")]),
        ),
        (
            "scoring.boolean.filteronly",
            BooleanQuery::new()
                .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
        ),
        (
            "scoring.boolean.filter.should",
            BooleanQuery::new()
                .with_filter([TermQuery::new("body", "cat")])
                .with_should([TermQuery::new("body", "dog")]),
        ),
        (
            "scoring.boolean.filter.minshouldmatch",
            BooleanQuery::new()
                .with_filter([TermQuery::new("body", "cat")])
                .with_should([
                    TermQuery::new("body", "dog"),
                    TermQuery::new("body", "bird"),
                ])
                .with_minimum_should_match(1),
        ),
        (
            "scoring.boolean.filter.dupmust",
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
                .with_filter([TermQuery::new("body", "dog")]),
        ),
        (
            "scoring.boolean.filter.nested",
            BooleanQuery::new()
                .with_must([BooleanQuery::new()
                    .with_must([TermQuery::new("body", "cat")])
                    .with_filter([TermQuery::new("body", "dog")])])
                .with_should([TermQuery::new("body", "dog")]),
        ),
    ];

    for (key, query) in cases {
        let mut top = TopDocsCollector::new(20);
        search_boolean_query_scored(
            &seg.fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            &query,
            Some(&norms),
            &mut top,
        )
        .unwrap();
        assert_same_as_lucene(key, top.top_docs(), &seg.manifest.lucene_hits(key));
    }
}

/// A single `FILTER` clause and nothing else. Real `BooleanQuery.rewrite`
/// turns this into `BoostQuery(ConstantScoreQuery(q), 0)` -- "no scoring
/// clauses, so return a score of 0" -- so real Lucene's recorded hits are the
/// *rewritten* query's. Both forms are checked against that one recording:
/// the unrewritten `BooleanQuery`, and the clause `rewrite()` actually
/// produces, which must be that exact `BoostQuery`/`ConstantScoreQuery` pair.
#[test]
fn a_lone_filter_clause_matches_at_score_zero_like_real_lucene() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");
    let norms = seg.norms_map("body");
    let key = "scoring.boolean.filter.single";
    let expected = seg.manifest.lucene_hits(key);
    assert_eq!(expected.len(), 2, "body:cat matches docs 0 and 2");

    let query = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);

    let mut top = TopDocsCollector::new(20);
    search_boolean_query_scored(
        &seg.fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query.clone(),
        Some(&norms),
        &mut top,
    )
    .unwrap();
    assert_same_as_lucene(key, top.top_docs(), &expected);

    // ... and the rewritten form Lucene actually executes.
    let rewritten = query.rewrite();
    assert_eq!(
        rewritten,
        Clause::Boost(Box::new(lucene_search::BoostQuery::new(
            Clause::ConstantScore(Box::new(lucene_search::ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                1.0,
            ))),
            0.0,
        ))),
        "Java: `case FILTER: return new BoostQuery(new ConstantScoreQuery(query), 0);`"
    );
    let mut top = TopDocsCollector::new(20);
    search_boolean_query_scored(
        &seg.fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &BooleanQuery::new().with_must([rewritten]),
        Some(&norms),
        &mut top,
    )
    .unwrap();
    assert_same_as_lucene(key, top.top_docs(), &expected);
}

/// A `FILTER` clause and the equivalent `MUST` clause must select **exactly**
/// the same documents -- the whole difference between them is the score. Run
/// through the unscored entry point, which is the one a filter-only query would
/// realistically use.
#[test]
fn filter_and_must_select_the_same_documents() {
    let seg = open_segment();
    let doc_in = DocInput::open(seg.doc, &seg.id, &seg.suffix).expect("open .doc");

    let docs = |query: &BooleanQuery| -> Vec<i32> {
        let mut v = VecCollector::default();
        lucene_search::search_boolean_query(
            &seg.fields,
            Some(&doc_in),
            None,
            None,
            None,
            None,
            query,
            &mut v,
        )
        .unwrap();
        v.docs
    };

    let as_must = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let as_filter = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let mixed = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")]);

    assert_eq!(docs(&as_must), vec![0]);
    assert_eq!(docs(&as_filter), docs(&as_must));
    assert_eq!(docs(&mixed), docs(&as_must));

    // And against real Lucene's own recorded doc list for the same conjunction.
    let lucene: Vec<i32> = seg
        .manifest
        .lucene_hits("scoring.boolean.must")
        .iter()
        .map(|(d, _)| *d)
        .collect();
    assert_eq!(docs(&as_filter), lucene);
}
