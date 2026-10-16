//! `FastVectorHighlighter`'s arguments, errors and accessors, over the
//! `vhighlight` fixture index (the docs of `matchhighlight-docs.tsv`).

use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::highlight::unified::break_iterators::{sentence_instance, BreakIterator};
use lucene_search::highlight::vectorhighlight::{
    BoundaryScanner, BreakIteratorBoundaryScanner, FastVectorHighlighter, FieldFragList,
    FieldPhraseList, FieldQuery, FieldTermStack, FragListBuilder, FragListKind, FragmentsBuilder,
    SimpleBoundaryScanner, SimpleFragListBuilder, SingleFragListBuilder, Toffs,
    WeightedFragListBuilder, WeightedPhraseInfo,
};
use lucene_search::highlight::{DefaultEncoder, Error};
use lucene_search::queryparser::classic::QueryParser;

fn reader() -> DirectoryReader {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/vhighlight");
    DirectoryReader::open(&lucene_store::FsDirectory::open(dir.join("index"))).unwrap()
}

fn query(reader: &DirectoryReader, text: &str) -> lucene_search::query::Clause {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    QueryParser::new("tv", analyzer)
        .parse(text)
        .unwrap()
        .to_clause(reader.max_doc())
}

fn tags(t: &str) -> Vec<String> {
    vec![t.to_string()]
}

#[test]
fn bad_arguments_are_refused() {
    let r = reader();
    assert!(matches!(
        SimpleFragListBuilder::with_margin(-1),
        Err(Error::IllegalArgument(_))
    ));
    assert!(WeightedFragListBuilder::with_margin(-1).is_err());
    let mut fvh = FastVectorHighlighter::default();
    let fq = fvh.get_field_query(&query(&r, "tv:fox"), Some(&r)).unwrap();
    // fragCharSize below `margin * 3`
    let e = fvh.get_best_fragments(&fq, &r, 0, "tv", 5, 2).unwrap_err();
    assert!(e.to_string().contains("too small"), "{e}");
    let e = fvh
        .get_best_fragments(&fq, &r, 0, "tv", 30, -1)
        .unwrap_err();
    assert!(e.to_string().contains("must be positive"), "{e}");
    // no matched fields
    let mut fb = FragmentsBuilder::score_order();
    let list = SimpleFragListBuilder::new();
    assert!(fvh
        .get_best_fragments_matched(
            &fq,
            &r,
            0,
            "tv",
            &[],
            30,
            2,
            &list,
            &mut fb,
            &tags("<b>"),
            &tags("</b>"),
            &DefaultEncoder
        )
        .is_err());
    // a document the index lacks
    assert!(fvh.get_best_fragment(&fq, &r, 99, "tv", 30).is_err());
    // no tags to pick from
    let e = fvh
        .get_best_fragment_with(
            &fq,
            &r,
            0,
            "tv",
            30,
            &list,
            &mut fb,
            &[],
            &[],
            &DefaultEncoder,
        )
        .unwrap_err();
    assert!(e.to_string().contains("by zero"), "{e}");
}

#[test]
fn a_field_without_a_usable_term_vector_has_no_fragments() {
    let r = reader();
    let mut fvh = FastVectorHighlighter::default();
    for field in ["tvpos", "plain", "nosuchfield"] {
        let fq = fvh
            .get_field_query(&query(&r, &format!("{field}:fox")), Some(&r))
            .unwrap();
        assert_eq!(
            fvh.get_best_fragment(&fq, &r, 0, field, 30).unwrap(),
            None,
            "{field}"
        );
    }
    // a query on another field has no terms for this one
    let fq = fvh
        .get_field_query(&query(&r, "tvb:fox"), Some(&r))
        .unwrap();
    assert!(fq.term_set("tv").is_none());
    assert!(fq.term_set("tvb").unwrap().contains("fox"));
    assert_eq!(fvh.get_best_fragment(&fq, &r, 0, "tv", 30).unwrap(), None);
    // an empty phrase limit finds nothing
    fvh.set_phrase_limit(0);
    assert_eq!(fvh.phrase_limit(), 0);
    let fq = fvh.get_field_query(&query(&r, "tv:fox"), Some(&r)).unwrap();
    assert_eq!(fvh.get_best_fragment(&fq, &r, 0, "tv", 30).unwrap(), None);
    fvh.set_phrase_limit(-4);
    assert_eq!(fvh.phrase_limit(), 0);
}

#[test]
fn the_highlighter_reports_what_it_was_built_with() {
    let fvh = FastVectorHighlighter::new(false, true);
    assert!(!fvh.is_phrase_highlight());
    assert!(fvh.is_field_match());
    let d = FastVectorHighlighter::default();
    assert!(d.is_phrase_highlight() && d.is_field_match());
    assert_eq!(d.phrase_limit(), usize::MAX);
}

#[test]
fn multi_term_queries_need_a_reader() {
    let r = reader();
    let fq = FieldQuery::new(&query(&r, "tv:qu*"), None, true, true).unwrap();
    assert!(fq.term_set("tv").is_none());
    let fq = FieldQuery::new(&query(&r, "tv:qu* tv:fox"), Some(&r), true, true).unwrap();
    let set = fq.term_set("tv").unwrap();
    assert!(set.contains("quick") && set.contains("fox"));
    // without field matching every field shares one set
    let fq = FieldQuery::new(&query(&r, "tvb:fox"), Some(&r), true, false).unwrap();
    assert!(fq.term_set("anything").unwrap().contains("fox"));
}

#[test]
fn the_phrase_tree_is_searched_by_term_and_position() {
    let r = reader();
    let fq = FieldQuery::new(&query(&r, "tv:\"quick brown\"~1"), Some(&r), true, true).unwrap();
    let root = fq.field_term_map("tv", "quick").unwrap();
    assert!(!root.is_terminal());
    let leaf = root.term_map("brown").unwrap();
    assert!(leaf.is_terminal());
    assert_eq!((leaf.slop(), leaf.boost()), (1, 1.0));
    assert_eq!(leaf.term_or_phrase_number(), 0);
    assert!(fq.field_term_map("tv", "zzz").is_none());
    assert!(fq.field_term_map("nofield", "quick").is_none());
}

#[test]
fn stack_and_phrase_list_show_what_they_hold() {
    let r = reader();
    let fq = FieldQuery::new(
        &query(&r, "tv:\"quick brown\" tv:fox"),
        Some(&r),
        true,
        true,
    )
    .unwrap();
    let mut stack = FieldTermStack::new(&r, 0, "tv", &fq).unwrap();
    assert_eq!(stack.field_name(), "tv");
    let first = stack.pop().unwrap();
    assert_eq!(stack.info(first).to_string(), "quick(4,9,1)");
    assert_eq!(stack.next_at_position(first), first);
    assert!(stack.info(first).weight() > 0.0);
    assert_eq!(stack.info(first).start_offset(), 4);
    assert_eq!(stack.info(first).end_offset(), 9);
    stack.push(first);
    assert!(!stack.is_empty());
    let list = FieldPhraseList::new(&mut stack, &fq, usize::MAX);
    assert!(stack.is_empty());
    let first = &list.phrase_list()[0];
    assert_eq!(first.text(), "quickbrown");
    assert_eq!(first.to_string(), "quickbrown(1)((4,15))");
    assert_eq!(first.seqnum(), 0);
    assert_eq!(first.terms_infos().len(), 2);
    assert_eq!(first.terms_offsets(), &[Toffs::new(4, 15)]);
    assert_eq!((first.start_offset(), first.end_offset()), (4, 15));
    assert_eq!(Toffs::new(4, 15).to_string(), "(4,15)");
    assert_eq!(
        Toffs::new(1, 2).compare(&Toffs::new(1, 3)),
        std::cmp::Ordering::Less
    );
    // the same list, merged on its own, is the same
    let merged = FieldPhraseList::merge(std::slice::from_ref(&list));
    assert_eq!(merged.phrase_list().len(), list.phrase_list().len());
    let none = FieldPhraseList::merge(&[]);
    assert!(none.phrase_list().is_empty());
    let one = WeightedPhraseInfo::merge(&list.phrase_list()[..1]);
    assert_eq!(one.terms_offsets(), first.terms_offsets());
}

#[test]
fn a_fragment_list_sums_or_weighs_its_phrases() {
    let r = reader();
    let fq = FieldQuery::new(&query(&r, "tv:fox tv:dog"), Some(&r), true, true).unwrap();
    let mut stack = FieldTermStack::new(&r, 0, "tv", &fq).unwrap();
    let phrases = FieldPhraseList::new(&mut stack, &fq, usize::MAX);
    let simple = SimpleFragListBuilder::new()
        .create_field_frag_list(&phrases, 30)
        .unwrap();
    let weighted = WeightedFragListBuilder::new()
        .create_field_frag_list(&phrases, 30)
        .unwrap();
    let single = SingleFragListBuilder
        .create_field_frag_list(&phrases, 1)
        .unwrap();
    assert!(!simple.frag_infos().is_empty());
    assert_eq!(single.frag_infos().len(), 1);
    assert_eq!(single.frag_infos()[0].end_offset, i32::MAX);
    assert!(simple.frag_infos()[0]
        .to_string()
        .starts_with("subInfos=(fox("));
    assert!(weighted.frag_infos()[0].total_boost > 0.0);
    // no phrase, no fragment
    let empty = FieldPhraseList::default();
    assert!(SingleFragListBuilder
        .create_field_frag_list(&empty, 1)
        .unwrap()
        .frag_infos()
        .is_empty());
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(0, 10, &[]);
    assert_eq!(list.into_frag_infos().len(), 1);
}

#[test]
fn boundary_scanners_stop_at_their_limits() {
    let text: Vec<u16> = "ab, cd ef gh".encode_utf16().collect();
    let mut s = SimpleBoundaryScanner::with_max_scan(2);
    // out of range is returned as it is
    assert_eq!(s.find_start_offset(&text, 0), 0);
    assert_eq!(s.find_start_offset(&text, 99), 99);
    assert_eq!(s.find_end_offset(&text, -1), -1);
    assert_eq!(s.find_end_offset(&text, 99), 99);
    // a boundary within reach
    assert_eq!(s.find_start_offset(&text, 4), 4);
    assert_eq!(s.find_end_offset(&text, 5), 6);
    // none within reach: the offset itself
    assert_eq!(s.find_end_offset(&text, 7), 7);
    let mut far = SimpleBoundaryScanner::with_chars(1, &[u16::from(b'z')]);
    assert_eq!(far.find_start_offset(&text, 5), 5);
    assert_eq!(far.find_end_offset(&text, 5), 5);
    // scanning back to the beginning
    let mut s = SimpleBoundaryScanner::with_chars(20, &[u16::from(b'z')]);
    assert_eq!(s.find_start_offset(&text, 5), 0);

    let mut b = BreakIteratorBoundaryScanner::new(sentence_instance());
    let text: Vec<u16> = "One. Two three. Four".encode_utf16().collect();
    assert_eq!(b.find_start_offset(&text, 0), 0);
    assert_eq!(b.find_start_offset(&text, 99), 99);
    assert_eq!(b.find_start_offset(&text, 10), 5);
    assert_eq!(b.find_end_offset(&text, 99), 99);
    assert_eq!(b.find_end_offset(&text, 6), 16);
    let mut bi: Box<dyn BreakIterator> = sentence_instance();
    bi.set_text(&text);
    assert_eq!(bi.first(), 0);
}

#[test]
fn fragments_builder_settings_are_kept() {
    let mut fb = FragmentsBuilder::simple().with_tags(&["[", "{"], &["]", "}"]);
    assert_eq!(fb.multi_valued_separator(), u16::from(b' '));
    fb.set_multi_valued_separator(u16::from(b'|'));
    assert_eq!(fb.multi_valued_separator(), u16::from(b'|'));
    assert!(!fb.is_discrete_multi_value_highlighting());
    fb.set_discrete_multi_value_highlighting(true);
    assert!(fb.is_discrete_multi_value_highlighting());
}

#[test]
fn the_builder_tags_and_encoder_default_to_java() {
    let r = reader();
    let mut fvh = FastVectorHighlighter::default();
    let fq = fvh.get_field_query(&query(&r, "tv:fox"), Some(&r)).unwrap();
    let best = fvh.get_best_fragment(&fq, &r, 4, "tv", 30).unwrap();
    assert_eq!(best.as_deref(), Some("<b>fox</b>"));
    let all = fvh
        .get_best_fragments(&fq, &r, 4, "tv", 30, 0)
        .unwrap()
        .unwrap();
    assert!(all.is_empty());
    // the builder's own tags and its fragments, again
    let mut fb = FragmentsBuilder::score_order().with_tags(&["<i>"], &["</i>"]);
    let list = SimpleFragListBuilder::new();
    let one = fvh
        .get_best_fragments_with(
            &fq,
            &r,
            4,
            "tv",
            30,
            2,
            &list,
            &mut fb,
            &tags("<i>"),
            &tags("</i>"),
            &DefaultEncoder,
        )
        .unwrap()
        .unwrap();
    assert_eq!(one, ["<i>fox</i>"]);
}

#[test]
fn merged_phrases_keep_apart_what_does_not_touch() {
    let r = reader();
    let fq = FieldQuery::new(&query(&r, "tv:quick tv:fox"), Some(&r), true, true).unwrap();
    let mut stack = FieldTermStack::new(&r, 0, "tv", &fq).unwrap();
    let list = FieldPhraseList::new(&mut stack, &fq, usize::MAX);
    let (quick, fox) = (&list.phrase_list()[0], &list.phrase_list()[1]);
    assert_eq!(
        (quick.text(), fox.text()),
        ("quick".to_string(), "fox".to_string())
    );
    let merged = WeightedPhraseInfo::merge(&list.phrase_list()[..2]);
    assert_eq!(
        merged.terms_offsets(),
        &[Toffs::new(4, 9), Toffs::new(16, 19)]
    );
    assert_eq!(merged.boost(), quick.boost() + fox.boost());
    // Java's quirk: the terms of the first phrase are not carried over
    assert_eq!(merged.terms_infos().len(), 1);
    assert_eq!(merged.terms_infos()[0].text(), "fox");
    // equal offsets: the boost decides
    let heavy = WeightedPhraseInfo::new(quick.terms_infos(), 2.0, 0);
    let light = WeightedPhraseInfo::new(quick.terms_infos(), 1.0, 0);
    use std::cmp::Ordering::{Equal, Greater, Less};
    assert_eq!(heavy.compare(&light), Greater);
    assert_eq!(light.compare(&heavy), Less);
    assert_eq!(light.compare(&light), Equal);
}

#[test]
fn a_fragment_outside_the_stored_text_is_an_error() {
    let r = reader();
    let fvh = FastVectorHighlighter::default();
    let fq = fvh.get_field_query(&query(&r, "tv:fox"), Some(&r)).unwrap();
    let mut stack = FieldTermStack::new(&r, 0, "tv", &fq).unwrap();
    let phrases = FieldPhraseList::new(&mut stack, &fq, usize::MAX);
    let far = &phrases.phrase_list()[1]; // fox at (56,59), past doc 4's text
    let (pre, post) = (tags("<b>"), tags("</b>"));
    let mut fb = FragmentsBuilder::simple();
    // the fragment starts past the end of every value
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(500, 600, &[far]);
    let e = fb
        .create_fragments_with(&r, 4, "tv", &list, 1, &pre, &post, &DefaultEncoder)
        .unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
    // the fragment fits, the phrase in it does not
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(0, 10, &[far]);
    let e = fb
        .create_fragments_with(&r, 4, "tv", &list, 1, &pre, &post, &DefaultEncoder)
        .unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
    // a document without the field stored has no fragments
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(0, 10, &[far]);
    assert_eq!(
        fb.create_fragments_with(
            &r,
            r.max_doc() - 4,
            "tv",
            &list,
            1,
            &pre,
            &post,
            &DefaultEncoder
        )
        .unwrap(),
        None
    );
    // a fragment ending at 0 reads no value: `values[index[0] - 1]`
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(0, 0, &[far]);
    let e = fb
        .create_fragments_with(&r, 4, "tv", &list, 1, &pre, &post, &DefaultEncoder)
        .unwrap_err();
    assert!(e.to_string().contains("-1"), "{e}");
    // discrete values skip a fragment without phrases that crosses values
    // (doc 2's first value is 27 characters long)
    fb.set_discrete_multi_value_highlighting(true);
    let mut list = FieldFragList::new(FragListKind::Simple);
    list.add(20, 40, &[]);
    assert_eq!(
        fb.create_fragments_with(&r, 2, "tv", &list, 1, &pre, &post, &DefaultEncoder)
            .unwrap(),
        Some(Vec::new())
    );
}

#[test]
fn every_variant_checks_the_fragment_size() {
    let r = reader();
    let fvh = FastVectorHighlighter::default();
    let fq = fvh.get_field_query(&query(&r, "tv:fox"), Some(&r)).unwrap();
    let (list, mut fb) = (
        SimpleFragListBuilder::new(),
        FragmentsBuilder::score_order(),
    );
    let (pre, post) = (tags("<b>"), tags("</b>"));
    assert!(fvh
        .get_best_fragment_with(
            &fq,
            &r,
            0,
            "tv",
            5,
            &list,
            &mut fb,
            &pre,
            &post,
            &DefaultEncoder
        )
        .is_err());
    assert!(fvh
        .get_best_fragments_with(
            &fq,
            &r,
            0,
            "tv",
            5,
            2,
            &list,
            &mut fb,
            &pre,
            &post,
            &DefaultEncoder
        )
        .is_err());
    assert!(fvh
        .get_best_fragments_matched(
            &fq,
            &r,
            0,
            "tv",
            &["tv", "tvb"],
            5,
            2,
            &list,
            &mut fb,
            &pre,
            &post,
            &DefaultEncoder
        )
        .is_err());
    // the default boundary scanner is `SimpleBoundaryScanner()`
    let text: Vec<u16> = "ab cd ef".encode_utf16().collect();
    let (mut a, mut b) = (
        SimpleBoundaryScanner::default(),
        SimpleBoundaryScanner::new(),
    );
    assert_eq!(a.find_start_offset(&text, 4), b.find_start_offset(&text, 4));
    assert_eq!(a.find_end_offset(&text, 4), b.find_end_offset(&text, 4));
}
