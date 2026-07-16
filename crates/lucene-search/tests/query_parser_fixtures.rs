//! Integration test for `query_parser::parse_query` against the same real
//! `IndexWriter`-produced segment `tests/term_query_fixtures.rs`/
//! `tests/boolean_query_fixtures.rs` already validate
//! (`fixtures/data/blocktree_index/`). There's no "real Lucene bytes" for
//! query-string *syntax* to decode -- the correctness question this test
//! answers is "does this string parse to the `Clause` a human would expect
//! and does it execute to the same doc set as an equivalent hand-built
//! `Clause`", using the already-differentially-verified single-clause
//! queries (`TermQuery`/`PhraseQuery`/`WildcardQuery`/`PrefixQuery`/
//! `FuzzyQuery`) as the comparison baseline, per task #44's scoping note.
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s
//! own doc comment): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_search::query_parser::parse_query;
use lucene_search::{search_boolean_query, BooleanQuery, Clause, TermQuery, VecCollector};

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
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
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

fn open_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String) {
    let m = Manifest::load();
    let id = id_from_hex(m.get("id_hex"));
    let suffix = m.get("segment_suffix").to_string();
    let max_doc: i32 = m.get("max_doc").parse().unwrap();

    let fnm = read_raw(m.get("fnm_file_name"));
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");

    let tim = read_raw(m.get("tim_file_name"));
    let tip = read_raw(m.get("tip_file_name"));
    let tmd = read_raw(m.get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");

    let doc = read_raw(m.get("doc_file_name"));
    (fields, doc, id, suffix)
}

/// Executes `clause` as a single `must` clause in a `BooleanQuery` and
/// returns the ascending matched doc-ID list -- same execution shape
/// `prefix_query_fixtures.rs` etc. already use for single-clause checks.
fn run(clause: Clause) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let query = BooleanQuery::new().with_must([clause]);
    let mut c = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &query,
        &mut c,
    )
    .unwrap();
    c.docs
}

/// A parsed single-term query against `body` must return exactly the same
/// doc set as the hand-built `TermQuery` -- the baseline
/// `term_query_fixtures.rs` already proves against real Lucene's own
/// postings.
#[test]
fn parsed_bare_term_matches_hand_built_term_query() {
    for term in ["cat", "dog", "bird"] {
        let parsed = parse_query(term, Some("body")).unwrap();
        assert_eq!(parsed, Clause::Term(TermQuery::new("body", term)));
        assert_eq!(run(parsed), run(Clause::from(TermQuery::new("body", term))));
    }
}

/// `field:term` syntax parses to, and executes identically to, the same
/// hand-built `TermQuery`.
#[test]
fn parsed_field_prefixed_term_matches_hand_built_term_query() {
    let parsed = parse_query("body:cat", None).unwrap();
    assert_eq!(
        run(parsed),
        run(Clause::from(TermQuery::new("body", "cat")))
    );
}

/// `+cat -dog` (must `cat`, must_not `dog`) must narrow to `cat`'s docs
/// minus `dog`'s docs: `cat` = {0,2}, `dog` = {0,1} -> {2}.
#[test]
fn parsed_plus_minus_combination_executes_like_hand_built_boolean_query() {
    let parsed = parse_query("+body:cat -body:dog", None).unwrap();
    let hand_built = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);

    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut parsed_collector = VecCollector::default();
    let Clause::Boolean(parsed_query) = parsed else {
        panic!("expected Boolean clause");
    };
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &parsed_query,
        &mut parsed_collector,
    )
    .unwrap();

    let mut hand_collector = VecCollector::default();
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &hand_built,
        &mut hand_collector,
    )
    .unwrap();

    assert_eq!(parsed_collector.docs, hand_collector.docs);
    assert_eq!(parsed_collector.docs, vec![2]);
}

/// A quoted phrase parses to, and executes identically to, the equivalent
/// hand-built `PhraseQuery`. A single-term phrase is used deliberately (see
/// [`lucene_search::search_phrase_query`]'s doc comment): it degenerates to
/// a plain term search with no `.pos` file needed, so this test can reuse
/// the same `run` helper (`must`-clause execution with no `pos_in`) every
/// other case here already uses, while still proving the parser builds a
/// genuine `Clause::Phrase` (not a `Clause::Term`) for quoted input.
#[test]
fn parsed_phrase_matches_hand_built_phrase_query() {
    let parsed = parse_query(r#"body:"cat""#, None).unwrap();
    let hand_built = lucene_search::PhraseQuery::new("body", ["cat"]);
    assert_eq!(parsed, Clause::Phrase(hand_built.clone()));
    assert_eq!(run(parsed), run(Clause::from(hand_built)));
}

/// A parenthesized group of bare (SHOULD) terms parses to, and executes
/// identically to, the equivalent hand-built `BooleanQuery` should-list:
/// `(cat dog)` -> should[cat, dog] -> union of {0,2} and {0,1} = {0,1,2}.
#[test]
fn parsed_group_of_should_terms_executes_like_hand_built_boolean_query() {
    let parsed = parse_query("(body:cat body:dog)", None).unwrap();
    let hand_built = Clause::from(
        BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
    );
    assert_eq!(run(parsed), run(hand_built));
}
