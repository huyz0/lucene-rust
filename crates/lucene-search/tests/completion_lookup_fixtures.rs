//! M13 T13.3d against Lucene 10.5.0 (`fixtures/src/GenCompletion.java`):
//! `PrefixCompletionQuery` over a real Completion104 index, answered from the
//! `.cmp`/`.lkp` the writer left, 18 prefixes x 5 sizes x with and without
//! `skipDuplicates` (180 queries), and `RegexCompletionQuery` (96).

use std::fs;
use std::path::PathBuf;

use lucene_codecs::completion::CompletionFields;
use lucene_search::suggest::document::{lookup, SegmentView, TopSuggestDocsCollector};
use lucene_util::automaton::automata::make_binary;
use lucene_util::automaton::operations::determinize;
use lucene_util::automaton::{RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/completion")
}

fn segment_id() -> [u8; 16] {
    let hex = fs::read_to_string(dir().join("id.hex")).unwrap();
    let mut id = [0u8; 16];
    for (b, pair) in id.iter_mut().zip(hex.trim().as_bytes().chunks(2)) {
        *b = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    id
}

#[test]
fn prefix_completion_answers_as_lucene_does() {
    let suffix = fs::read_to_string(dir().join("suffix.txt")).unwrap();
    let fields = CompletionFields::open(
        "Completion104",
        &segment_id(),
        suffix.trim(),
        &fs::read(dir().join("suggest.cmp")).unwrap(),
        fs::read(dir().join("suggest.lkp")).unwrap(),
    )
    .unwrap();
    let suggester = fields.suggester(0).unwrap().unwrap();
    let max_doc = fs::read_to_string(dir().join("docs.tsv"))
        .unwrap()
        .lines()
        .count() as i32;
    let accept = |_: i32| true;
    let view = SegmentView {
        num_docs: max_doc,
        max_doc,
        filtered: false,
        accept: &accept,
    };
    let (mut checked, mut bad) = (0, Vec::new());
    for line in fs::read_to_string(dir().join("lookups.tsv"))
        .unwrap()
        .lines()
    {
        let c: Vec<&str> = line.split('\t').collect();
        let (prefix, n, dedup) = (c[0], c[1].parse::<usize>().unwrap(), c[2] == "1");
        let mut collector = TopSuggestDocsCollector::new(n, dedup).unwrap();
        let automaton = make_binary(prefix.as_bytes());
        lookup(&suggester, &automaton, &view, &mut collector).unwrap();
        collector.finish();
        let got = collector
            .get()
            .iter()
            .map(|h| format!("{}\u{1e}{}\u{1e}{}", h.key, h.doc, h.score as i64))
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let want = c.get(3).copied().unwrap_or("");
        checked += 1;
        if got != want {
            bad.push(format!(
                "{prefix:?} n={n} dedup={dedup}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(4).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 18 * 10);
}

/// `regex.tsv`: `RegexCompletionQuery`, whose automaton intersects into
/// several prefix paths, so the collector fills before the search ends
/// (`CollectionTerminatedException` in Java) for the small sizes.
#[test]
fn regex_completion_answers_as_lucene_does() {
    let suffix = fs::read_to_string(dir().join("suffix.txt")).unwrap();
    let fields = CompletionFields::open(
        "Completion104",
        &segment_id(),
        suffix.trim(),
        &fs::read(dir().join("suggest.cmp")).unwrap(),
        fs::read(dir().join("suggest.lkp")).unwrap(),
    )
    .unwrap();
    let suggester = fields.suggester(0).unwrap().unwrap();
    let max_doc = fs::read_to_string(dir().join("docs.tsv"))
        .unwrap()
        .lines()
        .count() as i32;
    let accept = |_: i32| true;
    let view = SegmentView {
        num_docs: max_doc,
        max_doc,
        filtered: false,
        accept: &accept,
    };
    let (mut checked, mut bad) = (0, Vec::new());
    for line in fs::read_to_string(dir().join("regex.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split("\t").collect();
        let (re, n, dedup) = (c[0], c[1].parse::<usize>().unwrap(), c[2] == "1");
        let automaton = determinize(
            &RegExp::new(re).unwrap().to_automaton().unwrap(),
            DEFAULT_DETERMINIZE_WORK_LIMIT,
        )
        .unwrap();
        let mut collector = TopSuggestDocsCollector::new(n, dedup).unwrap();
        lookup(&suggester, &automaton, &view, &mut collector).unwrap();
        collector.finish();
        let got = collector
            .get()
            .iter()
            .map(|h| format!("{}\u{1e}{}\u{1e}{}", h.key, h.doc, h.score as i64))
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let want = c.get(3).copied().unwrap_or("");
        checked += 1;
        if got != want {
            bad.push(format!(
                "{re:?} n={n} dedup={dedup}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(4).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 8 * 6 * 2);
}
