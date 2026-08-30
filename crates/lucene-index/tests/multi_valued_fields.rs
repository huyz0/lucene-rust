//! Differential test for the `TokenStream` lifecycle and multi-valued fields:
//! `IndexingChain.PerField.invertTokenStream`'s
//!
//! ```java
//! stream.end();
//! invertState.position += invertState.posIncrAttribute.getPositionIncrement();
//! invertState.offset += invertState.offsetAttribute.endOffset();
//! ...
//! if (analyzed) {
//!   invertState.position += analyzer.getPositionIncrementGap(fieldInfo.name);
//!   invertState.offset += analyzer.getOffsetGap(fieldInfo.name);
//! }
//! ```
//!
//! Ground truth is the `mv_*` block of `fixtures/data/analysis/
//! manifest.properties`, which `fixtures/src/GenAnalysis.java` produces by
//! indexing the same values as repeated values of one field through a real
//! `IndexWriter` and reading the positions and offsets back off the postings.
//! Reading them off the *postings* rather than off a `TokenStream` is the
//! point: everything this accumulation does happens downstream of every
//! attribute a token list can show.

use lucene_analysis::Analyzer;
use lucene_index::indexing_chain::invert_documents;

fn manifest() -> Vec<(String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/analysis/manifest.properties"
    );
    std::fs::read_to_string(path)
        .expect("run scripts/gen-fixtures.sh --only GenAnalysis first")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

/// `term:position:start,end` triples in the manifest's order (position, then
/// term), which is how `GenAnalysis` sorts them.
fn rendered(occurrences: &mut [(i32, String, i32, i32)]) -> String {
    occurrences.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    occurrences
        .iter()
        .map(|(pos, term, start, end)| format!("{term}:{pos}:{start},{end}"))
        .collect::<Vec<_>>()
        .join(";")
}

/// Runs one `mv_*` case through `invert_documents` and compares every
/// occurrence's position and offsets against real Lucene's.
fn check_case(case: &str, stopwords: Option<&[&str]>) {
    let m = manifest();
    let values: Vec<&str> = get(&m, &format!("{case}.values")).split('|').collect();
    let position_gap: i32 = get(&m, &format!("{case}.position_increment_gap"))
        .parse()
        .unwrap();
    let offset_gap: i32 = get(&m, &format!("{case}.offset_gap")).parse().unwrap();
    let expected = get(&m, &format!("{case}.postings")).to_string();

    let stopword_set = stopwords.map(|words| {
        words
            .iter()
            .map(|w| (*w).to_string())
            .collect::<std::collections::HashSet<String>>()
    });
    let analyzer = Analyzer::standard(stopword_set.as_ref())
        .with_position_increment_gap(position_gap)
        .with_offset_gap(offset_gap);

    // Every value of the field, for one document, consecutively -- which is
    // what makes them one multi-valued field rather than several documents.
    let docs: Vec<(i32, &str, &str)> = values.iter().map(|v| (0i32, "body", *v)).collect();
    let index = invert_documents(&docs, &analyzer);

    let mut occurrences: Vec<(i32, String, i32, i32)> = Vec::new();
    for ((field, term), postings) in index.terms.iter() {
        assert_eq!(field, "body");
        for entry in &postings.entries {
            for occurrence in &entry.occurrences {
                occurrences.push((
                    occurrence.position,
                    term.clone(),
                    occurrence.start_offset,
                    occurrence.end_offset,
                ));
            }
        }
    }

    assert_eq!(
        rendered(&mut occurrences),
        expected,
        "case {case}: values {values:?}, positionIncrementGap {position_gap}, \
         offsetGap {offset_gap}"
    );
}

/// Java's base `Analyzer.getPositionIncrementGap` is **0**, so the second
/// value continues straight on from the first and a phrase *does* match
/// across the boundary. Recorded because it is the surprising direction: a
/// port that "fixed" multi-valued fields by always inserting a gap would fail
/// this case.
#[test]
fn multi_valued_field_at_the_default_gap_matches_lucene() {
    check_case("mv_default_gap", None);
}

/// A non-zero `positionIncrementGap` (the override every Lucene consumer
/// exposes) pushes the second value's positions out by exactly the gap.
#[test]
fn multi_valued_field_with_a_position_increment_gap_matches_lucene() {
    check_case("mv_gap_100", None);
}

/// The `end()` case: the first value ends in two stopwords, whose increments
/// `FilteringTokenFilter.end()` hands to the field's position counter. Drop
/// them -- as this port did, having no end-of-stream hook at all -- and `dog`
/// lands at position 1 instead of 3, and its offsets at `0,3` instead of
/// `12,15`.
#[test]
fn trailing_stopwords_advance_the_next_values_positions_as_lucene_does() {
    check_case("mv_trailing_stopwords", Some(&["the"]));
}

/// Trailing stopwords *and* both gaps, over three values, so the accumulation
/// is exercised more than once and the two additions compose.
#[test]
fn stopwords_and_both_gaps_compose_across_three_values() {
    check_case("mv_stopwords_and_gap", Some(&["the"]));
}
