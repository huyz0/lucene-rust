//! Differential test for `WildcardQuery` (task #34) against the same real
//! `IndexWriter`-produced segment `tests/boolean_query_fixtures.rs`/
//! `tests/dismax_query_fixtures.rs` already validate
//! (`fixtures/data/blocktree_index/`) -- see those files' module docs for the
//! shared fixture-reuse rationale.
//!
//! `body`'s real terms/postings in this fixture (see `GenBlockTree.java`'s own
//! doc comment): `cat` = {0, 2}, `dog` = {0, 1}, `bird` = {1, 4}.
//!
//! `fixtures/src/AppendWildcardManifest.java` opens this exact fixture
//! directory read-only and runs genuine
//! `org.apache.lucene.search.WildcardQuery`s (a literal, a trailing `*`, a
//! `?`, a bare `*`, a no-match pattern, and two `\`-escaped patterns) through
//! a real `IndexSearcher`, recording real Lucene's own matched doc IDs into
//! `manifest.properties`' `wildcard.*` keys -- this is the actual
//! cross-engine proof this port's `WildcardQuery` matching is checked
//! against, not a hand-derived expectation.

use lucene_codecs::blocktree;
use lucene_codecs::postings::DocInput;
use lucene_search::{search_boolean_query, BooleanQuery, Clause, VecCollector, WildcardQuery};

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
            .expect("run fixtures generator first (GenBlockTree + AppendWildcardManifest)");
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

/// Runs `search_boolean_query` with a single `Clause::Wildcard(field,
/// pattern)` `must` clause and returns the matched, ascending doc-ID list.
fn wildcard_docs(field: &str, pattern: &str) -> Vec<i32> {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([Clause::from(WildcardQuery::new(
        field,
        pattern.as_bytes().to_vec(),
    ))]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, &query, &mut c).unwrap();
    c.docs
}

/// Every `wildcard.<case>.*` manifest entry `AppendWildcardManifest.java`
/// recorded must be reproduced exactly by this port's own `Clause::Wildcard`
/// matching -- the actual cross-engine check (not a self-consistency test).
#[test]
fn wildcard_matches_real_lucenes_own_wildcardquery_output_for_every_case() {
    let m = Manifest::load();
    let cases: Vec<&str> = m.get("wildcard.cases").split(',').collect();
    assert!(!cases.is_empty(), "manifest must record wildcard cases");

    for case in cases {
        let field = m.get(&format!("wildcard.{case}.field"));
        let pattern = m.get(&format!("wildcard.{case}.pattern"));
        let docs_key = m.get(&format!("wildcard.{case}.docs"));
        let expected: Vec<i32> = if docs_key.is_empty() {
            Vec::new()
        } else {
            docs_key.split(',').map(|d| d.parse().unwrap()).collect()
        };

        let got = wildcard_docs(field, pattern);
        assert_eq!(got, expected, "case={case} field={field} pattern={pattern}");
    }
}

/// Unit-level checks of `WildcardQuery` matching semantics not already
/// pinned by the real-Lucene cases above: an empty pattern (`WildcardQuery`
/// with `pattern=""`) matches only an empty term (`body` has none, so it
/// matches nothing here), and a pattern of only wildcards (`"**"`/`"?*"`)
/// behaves the same as `"*"`/`"?"` respectively (adjacent-star collapsing,
/// already unit-tested in `lucene_codecs::wildcard`, exercised here through
/// the full `Clause::Wildcard` path for good measure).
#[test]
fn wildcard_empty_and_pure_wildcard_patterns() {
    assert_eq!(wildcard_docs("body", ""), Vec::<i32>::new());
    assert_eq!(wildcard_docs("body", "**"), vec![0, 1, 2, 4]);
    assert_eq!(wildcard_docs("body", "?"), Vec::<i32>::new()); // no single-byte term
}

/// A `WildcardQuery` against a field that doesn't exist in this segment
/// matches nothing, same "missing field" convention every other clause
/// follows (see `term_doc_ids`'s doc comment).
#[test]
fn wildcard_missing_field_matches_nothing() {
    assert_eq!(wildcard_docs("no_such_field", "*"), Vec::<i32>::new());
}

/// Composing a `Clause::Wildcard` inside a `BooleanQuery` alongside a plain
/// `Clause::Term` must intersect (`must`) / union (`should`) exactly like any
/// other clause pair -- `b*` (bird={1,4}) AND `dog` ({0,1}) -> {1}.
#[test]
fn wildcard_composes_inside_boolean_query_must() {
    let (fields, doc, id, suffix) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let query = BooleanQuery::new().with_must([
        Clause::from(WildcardQuery::new("body", b"b*".to_vec())),
        Clause::Term(lucene_search::TermQuery::new("body", "dog")),
    ]);
    let mut c = VecCollector::default();
    search_boolean_query(&fields, Some(&doc_in), None, None, None, &query, &mut c).unwrap();
    assert_eq!(c.docs, vec![1]);
}
