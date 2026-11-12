//! Expected snippets are OpenSearch 3.9's `unified` highlighter's on the same
//! text, a `text` field (`standard` analyzer, positions only, so `ANALYSIS`).

use super::*;
use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::query::{BooleanQuery, PhraseQuery, PrefixQuery, TermQuery};

const TEXT: &str = "The quick brown fox jumps over the lazy dog. A second sentence mentions \
the fox again & it ends here! Third <b>sentence</b> without the animal. The last one has quick \
foxes running quickly over the hills and far away from the lazy dogs of the village.";

fn options(breaks: Breaks, max_passages: i32) -> Options {
    Options {
        field: "body".into(),
        offset_source: OffsetSource::Analysis,
        breaks,
        max_passages,
        no_match_size: 0,
        pre_tag: "<em>".into(),
        post_tag: "</em>".into(),
        html: false,
        field_matcher: Arc::new(|f: &str| f == "body"),
        masked_fields: None,
    }
}

/// OpenSearch's defaults: `fragment_size` 100, five fragments.
fn defaults() -> Options {
    options(Breaks::BoundedSentence(100), 5)
}

fn analyzer() -> Option<Arc<Analyzer>> {
    Some(Arc::new(Analyzer::new(StandardAnalyzer::new())))
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t.as_bytes().to_vec()))
}

/// The snippets OpenSearch shows: those with text, in passage order.
fn run(options: Options, query: &Clause, content: &str) -> Vec<String> {
    let mut h = CustomUnifiedHighlighter::new(options, query, analyzer()).unwrap();
    let content: Vec<u16> = content.encode_utf16().collect();
    h.highlight(None, &content)
        .unwrap()
        .into_iter()
        .map(|s| String::from_utf16(&s.text).unwrap())
        .filter(|s| !s.trim().is_empty())
        .collect()
}

#[test]
fn a_term_in_bounded_sentences() {
    assert_eq!(
        run(defaults(), &term("fox"), TEXT),
        [
            "The quick brown <em>fox</em> jumps over the lazy dog.",
            "A second sentence mentions the <em>fox</em> again & it ends here! Third <b>sentence</b> without the animal."
        ]
    );
}

#[test]
fn no_fragments_is_the_whole_value() {
    assert_eq!(
        run(options(Breaks::Values, ALL_PASSAGES), &term("fox"), TEXT),
        [
            "The quick brown <em>fox</em> jumps over the lazy dog. A second sentence mentions the \
<em>fox</em> again & it ends here! Third <b>sentence</b> without the animal. The last one has \
quick foxes running quickly over the hills and far away from the lazy dogs of the village."
        ]
    );
}

#[test]
fn words_bound_each_passage() {
    let q = Clause::Boolean(Box::new(
        BooleanQuery::new().with_should([term("quick"), term("dog")]),
    ));
    assert_eq!(
        run(options(Breaks::Word, 5), &q, TEXT),
        ["<em>quick</em>", "<em>dog</em>", "<em>quick</em>"]
    );
}

#[test]
fn a_phrase_highlights_its_occurrence_only() {
    let q = Clause::Phrase(PhraseQuery::new(
        "body",
        [b"lazy".to_vec(), b"dog".to_vec()],
    ));
    assert_eq!(
        run(defaults(), &q, TEXT),
        ["The quick brown fox jumps over the <em>lazy</em> <em>dog</em>."]
    );
}

#[test]
fn a_prefix_highlights_every_expansion() {
    let q = Clause::Prefix(PrefixQuery::new("body", b"qui".to_vec()));
    assert_eq!(
        run(defaults(), &q, TEXT),
        [
            "The <em>quick</em> brown fox jumps over the lazy dog.",
            "The last one has <em>quick</em> foxes running <em>quickly</em> over the hills and far away from the lazy dogs of the village"
        ]
    );
}

#[test]
fn a_small_fragment_size_bounds_the_sentence() {
    assert_eq!(
        run(options(Breaks::BoundedSentence(20), 3), &term("fox"), TEXT),
        [
            "The quick brown <em>fox</em> jumps",
            "sentence mentions the <em>fox</em>"
        ]
    );
}

#[test]
fn fragment_size_zero_is_plain_sentences() {
    assert_eq!(
        run(options(Breaks::Sentence, 5), &term("fox"), TEXT),
        [
            "The quick brown <em>fox</em> jumps over the lazy dog.",
            "A second sentence mentions the <em>fox</em> again & it ends here!"
        ]
    );
}

#[test]
fn the_html_encoder_escapes_and_custom_tags_wrap() {
    let mut o = defaults();
    o.html = true;
    o.pre_tag = "<x>".into();
    o.post_tag = "</x>".into();
    assert_eq!(
        run(o, &term("sentence"), TEXT),
        [
            "A second <x>sentence</x> mentions the fox again &amp; it ends here! Third \
&lt;b&gt;<x>sentence</x>&lt;&#x2F;b&gt; without the animal."
        ]
    );
}

#[test]
fn a_query_on_another_field_is_a_noop_with_an_optional_summary() {
    let q = Clause::Term(TermQuery::new("title", b"none".to_vec()));
    let h = CustomUnifiedHighlighter::new(defaults(), &q, analyzer()).unwrap();
    assert!(h.is_noop());
    assert_eq!(h.no_match_size(), 0);
    assert_eq!(h.offset_source(), OffsetSource::NoneNeeded);
    assert_eq!(h.analysis_fields().count(), 0);
    let mut o = defaults();
    o.no_match_size = 30;
    assert_eq!(run(o, &q, TEXT), ["The quick brown fox jumps over"]);
}

#[test]
fn values_of_a_multi_valued_field_are_passages_of_their_own() {
    let content = "first value with fox\u{0}second value no match\u{0}third fox value";
    let expected = ["first value with <em>fox</em>", "third <em>fox</em> value"];
    assert_eq!(
        run(options(Breaks::Values, ALL_PASSAGES), &term("fox"), content),
        expected
    );
    assert_eq!(run(defaults(), &term("fox"), content), expected);
}

#[test]
fn one_fragment_is_the_best_passage() {
    let q = Clause::Boolean(Box::new(
        BooleanQuery::new().with_should([term("fox"), term("lazy")]),
    ));
    assert_eq!(
        run(options(Breaks::BoundedSentence(100), 1), &q, TEXT),
        ["The quick brown <em>fox</em> jumps over the <em>lazy</em> dog."]
    );
}

#[test]
fn analysis_needs_an_analyzer_and_reports_its_fields() {
    let h = CustomUnifiedHighlighter::new(defaults(), &term("fox"), analyzer()).unwrap();
    assert_eq!(h.analysis_fields().collect::<Vec<_>>(), ["body"]);
    assert_eq!(h.offset_source(), OffsetSource::Analysis);
    assert!(!h.is_noop());
    assert!(matches!(
        CustomUnifiedHighlighter::new(defaults(), &term("fox"), None),
        Err(Error::IllegalArgument(_))
    ));
}

#[test]
fn snippets_carry_scores_and_whether_they_are_highlighted() {
    let mut o = defaults();
    o.no_match_size = 10;
    let content: Vec<u16> = TEXT.encode_utf16().collect();
    let mut h = CustomUnifiedHighlighter::new(o, &term("fox"), analyzer()).unwrap();
    let snippets = h.highlight(None, &content).unwrap();
    assert!(snippets.iter().all(|s| s.highlighted && s.score > 0.0));
    let mut o = defaults();
    o.no_match_size = 10;
    let mut h = CustomUnifiedHighlighter::new(o, &term("absent"), analyzer()).unwrap();
    let snippets = h.highlight(None, &content).unwrap();
    assert_eq!(snippets.len(), 1);
    assert!(!snippets[0].highlighted);
    assert_eq!(
        String::from_utf16(&snippets[0].text).unwrap(),
        "The quick brown"
    );
}

#[test]
fn the_bounded_scanner_moves_like_its_sentence_iterator() {
    let mut b = BoundedBreakIteratorScanner::sentence(20);
    let text: Vec<u16> = "One two. Three four five six seven."
        .encode_utf16()
        .collect();
    b.set_text(&text);
    assert_eq!(b.first(), 0);
    assert_eq!(b.last(), text.len() as i32);
    assert_eq!(b.first(), 0);
    let next = b.next();
    assert!(next > 0);
    assert!(b.previous() < next);
}

fn phrase_prefix(arrays: &[&[&str]], slop: u32) -> Clause {
    let terms: Vec<Vec<Vec<u8>>> = arrays
        .iter()
        .map(|a| a.iter().map(|t| t.as_bytes().to_vec()).collect())
        .collect();
    let positions = (0..terms.len() as i32).collect();
    lucene_search::extended_query::MultiPhrasePrefixQuery::new("body", terms, positions, slop)
        .unwrap()
        .into()
}

#[test]
fn a_phrase_prefix_highlights_the_phrase_it_completes() {
    assert_eq!(
        run(defaults(), &phrase_prefix(&[&["quick"], &["fo"]], 0), TEXT),
        ["The last one has <em>quick</em> <em>foxes</em> running quickly over the hills and far away from the lazy dogs of the village"]
    );
}

#[test]
fn a_phrase_prefix_of_one_position_is_its_prefix() {
    assert_eq!(
        run(defaults(), &phrase_prefix(&[&["qui"]], 0), TEXT),
        run(
            defaults(),
            &Clause::Prefix(PrefixQuery::new("body", b"qui".to_vec())),
            TEXT
        ),
    );
}

#[test]
fn a_sloppy_phrase_prefix_matches_out_of_order() {
    // slop 2: "brown ... quick" within two moves
    assert_eq!(
        run(defaults(), &phrase_prefix(&[&["brown"], &["qu"]], 2), TEXT),
        ["The <em>quick</em> <em>brown</em> fox jumps over the lazy dog."]
    );
}

#[test]
fn a_phrase_prefix_s_whole_terms_alone_do_not_highlight() {
    // "lazy" is a whole term of the phrase: no "lazy z*" anywhere
    let h = CustomUnifiedHighlighter::new(
        defaults(),
        &phrase_prefix(&[&["lazy"], &["z"]], 0),
        analyzer(),
    )
    .unwrap();
    assert!(!h.is_noop());
    assert!(run(defaults(), &phrase_prefix(&[&["lazy"], &["z"]], 0), TEXT).is_empty());
}
