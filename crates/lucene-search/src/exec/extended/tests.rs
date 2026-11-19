#![cfg(test)]
use super::*;

fn imp(freq: i32, norm: i64) -> Impact {
    Impact { freq, norm }
}

/// `SynonymQuery.mergeImpacts`: the implicit impacts between a list's
/// own norms are counted (`{2, 10}` still bounds a document of norm 11),
/// equal norms sum, a sum that does not rise is dropped, and norms order
/// unsigned (a sign-extended byte norm sorts after every positive one).
#[test]
fn merged_impacts_sum_frequencies_by_norm_as_lucene_does() {
    assert!(merge_impacts(&[]).is_none());
    let one = vec![imp(2, 10), imp(4, 12)];
    assert_eq!(merge_impacts(std::slice::from_ref(&one)), Some(one.clone()));
    let other = vec![imp(1, 11), imp(3, 13)];
    assert_eq!(
        merge_impacts(&[one.clone(), other]),
        Some(vec![imp(2, 10), imp(3, 11), imp(5, 12), imp(7, 13)])
    );
    assert_eq!(
        merge_impacts(&[one, vec![imp(3, 10)]]),
        Some(vec![imp(5, 10), imp(7, 12)])
    );
    // Norm -1 is the largest unsigned: it comes last, and a frequency
    // that does not rise past the running maximum adds no impact.
    assert_eq!(
        merge_impacts(&[vec![imp(1, 5), imp(9, -1)], vec![imp(1, 3)]]),
        Some(vec![imp(1, 3), imp(2, 5), imp(10, -1)])
    );
    assert_eq!(
        merge_impacts(&[vec![imp(4, 1)], vec![imp(1, 2)], vec![imp(1, 1)]]),
        Some(vec![imp(5, 1), imp(6, 2)])
    );
}

/// A segment without the queried field, or without postings at all.
fn empty_ctx(fields: &BlockTreeFields) -> LeafContext<'_> {
    LeafContext {
        fields,
        doc_in: None,
        pos_in: None,
        pay_in: None,
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: Some(4),
        cache: None,
        reader: None,
        similarity: None,
    }
}

/// The two queries with no weight of their own refuse to build a scorer
/// until the searcher rewrites them, naming the rewrite that was missed.
#[test]
fn unrewritten_common_terms_and_mlt_queries_are_illegal_state() {
    let fields = BlockTreeFields::empty();
    let ctx = empty_ctx(&fields);
    let mut ct = crate::common_terms::CommonTermsQuery::new(
        crate::query_visitor::Occur::Should,
        crate::query_visitor::Occur::Should,
        0.1,
    )
    .unwrap();
    ct.add("f", "a");
    let mlt = crate::mlt::MoreLikeThisQuery::new(
        "a b",
        vec!["f".into()],
        std::sync::Arc::new(lucene_analysis::Analyzer::standard(None)),
        "f",
    );
    for (q, name) in [
        (ExtendedQuery::CommonTerms(ct), "CommonTermsQuery"),
        (ExtendedQuery::MoreLikeThis(mlt), "MoreLikeThisQuery"),
        (
            ExtendedQuery::FuzzyLikeThis(crate::fuzzy_like_this::FuzzyLikeThisQuery::new(
                50,
                std::sync::Arc::new(lucene_analysis::Analyzer::standard(None)),
            )),
            "FuzzyLikeThisQuery",
        ),
    ] {
        let built = build(&ctx, &q, 1.0, Mode::Complete, true);
        assert!(
            matches!(&built, Err(crate::Error::IllegalState(m)) if m.starts_with(name)),
            "{name}"
        );
    }
}

/// A phrase with explicit positions over a segment that lacks the field
/// or its postings has no scorer, as `PhraseWeight.scorer` returns
/// `null`; an empty one matches nothing, a one-term one is its term.
#[test]
fn positional_phrases_without_postings_have_no_scorer() {
    let fields = BlockTreeFields::empty();
    let ctx = empty_ctx(&fields);
    let two = PhraseQuery::with_positions("f", [("a", 0), ("b", 2)]).unwrap();
    assert!(positional_phrase(&ctx, &two, 1.0, Mode::Complete)
        .unwrap()
        .is_none());
    let one = PhraseQuery::with_positions("f", [("a", 3)]).unwrap();
    assert!(positional_phrase(&ctx, &one, 1.0, Mode::Complete)
        .unwrap()
        .is_none());
    let none = PhraseQuery::with_positions("f", Vec::<(&str, i32)>::new()).unwrap();
    assert!(positional_phrase(&ctx, &none, 1.0, Mode::NoScores)
        .unwrap()
        .is_none());
    // Without the `.doc` stream the term cursor is not opened at all.
    assert!(cursor(&ctx, "f", b"a", PostingsFlags::Freqs)
        .unwrap()
        .is_none());
}

/// A positional phrase over a real segment: a pulsed singleton term
/// (no `.doc` stream to walk) resolves the matches up front -- through
/// [`eager_phrase`], counted by [`EAGER_PHRASES`] -- a segment opened
/// without `.pos` cannot run it, and an absent field has no term cursor.
#[test]
fn positional_phrases_over_a_real_segment() {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/spans/index"
    ));
    let reader =
        crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
            .unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let mut eager = 0;
    let eager_before = EAGER_PHRASES.with(std::cell::Cell::get);
    for seg in &segs {
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
            reader: seg.reader,
            similarity: seg.similarity,
        };
        assert!(cursor(&ctx, "no_such_field", b"a", PostingsFlags::Freqs)
            .unwrap()
            .is_none());
        // `zeta` is in one document: a pulsed singleton where present.
        for w in [
            "apple", "ape", "bank", "band", "cat", "car", "dog", "egg", "fig",
        ] {
            for k in 1..=2 {
                for pair in [[("zeta", 0), (w, k)], [(w, 0), ("zeta", k)]] {
                    let q = PhraseQuery::with_positions("body", pair).unwrap();
                    if let Some(mut s) = positional_phrase(&ctx, &q, 1.0, Mode::Complete).unwrap() {
                        let doc = super::super::exact_next(&mut *s).unwrap();
                        if doc != NO_MORE_DOCS {
                            eager += 1;
                            assert!(s.score().unwrap() > 0.0);
                        }
                    }
                }
            }
        }
        let without_pos = LeafContext {
            pos_in: None,
            ..ctx
        };
        let two = PhraseQuery::with_positions("body", [("apple", 0), ("bank", 1)]).unwrap();
        assert!(matches!(
            positional_phrase(&without_pos, &two, 1.0, Mode::Complete),
            Err(crate::Error::MissingPosInput)
        ));
    }
    assert!(eager > 0, "zeta's document has neighbours");
    assert!(
        EAGER_PHRASES.with(std::cell::Cell::get) > eager_before,
        "the pulsed singleton's phrases resolve up front"
    );
}

/// A multi-phrase with one wide position (a prefix's expansion) matches
/// exactly the documents its alternatives' plain phrases do together.
#[test]
fn a_wide_position_is_the_union_of_its_plain_phrases() {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/spans/index"
    ));
    let reader =
        crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
            .unwrap();
    let opened = reader.open_segments().unwrap();
    let docs_of = |ctx: &LeafContext<'_>, q: &Clause| {
        let mut out = Vec::new();
        if let Some(mut s) = build::build(ctx, q, 1.0, Mode::Complete, false).unwrap() {
            loop {
                let d = super::super::exact_next(&mut *s).unwrap();
                if d == NO_MORE_DOCS {
                    break;
                }
                out.push(d);
            }
        }
        out
    };
    let words = [
        "apple", "ape", "bank", "band", "cat", "car", "dog", "egg", "fig", "zeta", "nope",
    ];
    let mut matched = 0;
    for seg in &opened.as_open_segments() {
        let ctx = LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: None,
            points: None,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: None,
            reader: seg.reader,
            similarity: seg.similarity,
        };
        for lead in words {
            for gap in [1, 2] {
                let wide = Clause::MultiPhrase(
                    MultiPhraseQuery::with_positions(
                        "body",
                        [(vec![lead], 0), (words.to_vec(), gap)],
                    )
                    .unwrap(),
                );
                let mut want = Vec::new();
                for w in words {
                    let q = PhraseQuery::with_positions("body", [(lead, 0), (w, gap)]).unwrap();
                    want.extend(docs_of(&ctx, &Clause::Phrase(q)));
                }
                want.sort_unstable();
                want.dedup();
                matched += want.len();
                assert_eq!(docs_of(&ctx, &wide), want, "{lead} +{gap}");
            }
        }
    }
    assert!(matched > 0, "some phrase matches");
}

/// The term-statistics queries over a segment without their field have
/// no scorer, as their weights' `scorer` returns `null`: a multi-phrase
/// (empty, or of absent terms), a synonym, a combined field and a
/// blended term.
#[test]
fn term_statistics_queries_without_the_field_have_no_scorer() {
    let fields = BlockTreeFields::empty();
    let ctx = empty_ctx(&fields);
    let none: Vec<Vec<&str>> = Vec::new();
    for (q, mode) in [
        (MultiPhraseQuery::new("f", none), Mode::Complete),
        (
            MultiPhraseQuery::new("f", [vec!["a", "b"], vec!["c"]]),
            Mode::Complete,
        ),
        (
            MultiPhraseQuery::new("f", [vec!["a"], vec!["c"]]),
            Mode::NoScores,
        ),
    ] {
        assert!(multi_phrase(&ctx, &q, 1.0, mode).unwrap().is_none());
    }
    let syn = SynonymQuery::new("f", [("a", 1.0), ("b", 0.5)]).unwrap();
    assert!(synonym(&ctx, &syn, 1.0, Mode::Complete).unwrap().is_none());
    let combined = CombinedFieldQuery::new("a", [("f", 1.0), ("g", 2.0)]).unwrap();
    assert!(combined_field(&ctx, &combined, 1.0, Mode::Complete)
        .unwrap()
        .is_none());
    let blend =
        BlendedTermQuery::new([("f", "a", 1.0), ("g", "a", 1.0)], BlendedRewrite::Boolean).unwrap();
    assert!(blended(&ctx, &blend, 1.0, Mode::Complete, true)
        .unwrap()
        .is_none());
}

/// `IndexOrDocValuesQuery`'s two sides over a segment without points or
/// a reader: a point-in-set index side has no supplier (`null`), any
/// other clause one of unknown cost; the doc-values side cannot be
/// asked without the segment's reader.
#[test]
fn index_or_doc_values_sides_without_points_or_reader() {
    let fields = BlockTreeFields::empty();
    let ctx = empty_ctx(&fields);
    let pis = Clause::Extended(Box::new(ExtendedQuery::PointInSet(
        crate::extended_query::PointInSetQuery::new("p", 1, 4, [vec![0, 0, 0, 1]]).unwrap(),
    )));
    assert!(matches!(index_side(&ctx, &pis).unwrap(), Supplier::Absent));
    let term = Clause::Term(TermQuery::new("f", "a"));
    assert!(matches!(
        index_side(&ctx, &term).unwrap(),
        Supplier::Present(None)
    ));
    let dv = Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(
        crate::extended_query::NumericDocValuesRangeQuery::new("n", 1, 5),
    )));
    assert!(matches!(
        dv_side_present(&ctx, &dv),
        Err(crate::Error::MissingSegmentReader(f)) if f == "n"
    ));
    assert!(dv_side_present(&ctx, &term).unwrap());
    let iodv = IndexOrDocValuesQuery::new(pis, dv);
    assert!(
        index_or_doc_values(&ctx, &iodv, 1.0, Mode::Complete, true, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn logit_sigmoid_softplus_follow_java() {
    assert_eq!(sigmoid(0.0), 0.5);
    assert!(sigmoid(-50.0) > 0.0 && sigmoid(-50.0) < 1e-20);
    assert_eq!(softplus(25.0), 25.0);
    assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
    assert!(logit(0.0) < -15.0);
    assert!(logit(1.0) > 15.0);
    assert_eq!(logit(0.5), 0.0);
}

#[test]
fn explicit_offsets_rebase_onto_slots() {
    let mut scratch = sloppy_phrase::SloppyScratch::default();
    let mut shifted = Vec::new();
    // "a _ b": a at 3, b at 5, the phrase has a hole at 1.
    let positions = vec![vec![3], vec![5]];
    let none = sloppy_phrase::PhraseRepeats::none(2);
    assert_eq!(
        phrase_freq_at(&positions, &[0, 2], &none, 0, &mut scratch, &mut shifted),
        1.0
    );
    assert_eq!(
        phrase_freq_at(&positions, &[0, 1], &none, 0, &mut scratch, &mut shifted),
        0.0
    );
    // One position of slop covers the hole's absence.
    assert!(phrase_freq_at(&positions, &[0, 1], &none, 1, &mut scratch, &mut shifted) > 0.0);
}

/// More slots than fit on the stack, and a rebased position past
/// `i32::MAX`, which saturates rather than wraps.
#[test]
fn explicit_offsets_over_many_slots_and_at_the_int_limit() {
    let mut scratch = sloppy_phrase::SloppyScratch::default();
    let mut shifted = Vec::new();
    // Ten terms at consecutive positions 10..20, offsets 0..10 with a
    // hole after the fifth: the phrase matches once, with the hole.
    let positions: Vec<Vec<i32>> = (0..10).map(|i| vec![10 + i + i32::from(i >= 5)]).collect();
    let offsets: Vec<i32> = (0..10).map(|i| i + i32::from(i >= 5)).collect();
    let none = sloppy_phrase::PhraseRepeats::none(10);
    assert_eq!(
        phrase_freq_at(&positions, &offsets, &none, 0, &mut scratch, &mut shifted),
        1.0
    );
    // Slot 1 at offset 0 moves up by one: from `i32::MAX` it saturates
    // and no longer lines up with slot 0.
    let edge = vec![vec![i32::MAX], vec![i32::MAX]];
    let none2 = sloppy_phrase::PhraseRepeats::none(2);
    assert_eq!(
        phrase_freq_at(&edge, &[0, 0], &none2, 0, &mut scratch, &mut shifted),
        0.0
    );
    assert_eq!(shifted[1], [i32::MAX]);
}

#[test]
fn log_odds_rewrite_drops_match_no_docs_and_renormalises() {
    let t = |s: &str| Clause::Term(TermQuery::new("f", s));
    let none = Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new());
    let q = LogOddsFusionQuery::new(
        [t("a"), none.clone(), t("b")],
        0.5,
        Some(vec![0.5, 0.25, 0.25]),
        Some((vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])),
    )
    .unwrap();
    let Rewritten::Fusion(r) = rewrite_log_odds(&q) else {
        panic!("two clauses stay a fusion")
    };
    assert_eq!(r.clauses.len(), 2);
    assert_eq!(r.weights.unwrap(), vec![0.5 / 0.75, 0.25 / 0.75]);
    assert_eq!(r.logit_bounds.unwrap(), (vec![0.0, 2.0], vec![3.0, 5.0]));
    let one = LogOddsFusionQuery::new([t("a"), none.clone()], 0.5, None, None).unwrap();
    assert!(matches!(rewrite_log_odds(&one), Rewritten::One(_)));
    let zero = LogOddsFusionQuery::new([none.clone(), none], 0.5, None, None).unwrap();
    assert!(matches!(rewrite_log_odds(&zero), Rewritten::None));
}

#[test]
fn multi_norms_reencode_the_weighted_length() {
    let mut none = Norms::Multi(Vec::new(), None);
    assert_eq!(none.norm(3, false).unwrap(), 1);
    let mut one = Norms::One(None);
    assert_eq!(one.norm(3, true).unwrap(), 1);
}

/// Dense one-byte norms, as `Lucene90NormsConsumer` writes them for an
/// ordinary analyzed field: `bytes[doc]` is the document's norm.
fn dense_norms(bytes: &[u8]) -> crate::field_norms::FieldNorms<'_> {
    let entry = lucene_codecs::norms::NormsEntry {
        field_number: 0,
        docs_with_field_offset: -1,
        docs_with_field_length: 0,
        jump_table_entry_count: -1,
        dense_rank_power: 0xFF,
        num_docs_with_field: bytes.len() as i32,
        bytes_per_norm: 1,
        norms_offset: 0,
    };
    crate::field_norms::FieldNorms::from_field_stats(bytes, entry, 100, 10)
}

/// [`DenseMemo`] is a cache of [`Norms::norm`]'s own arithmetic (which
/// the `CombinedFieldQuery` fixtures verify against Lucene): it must give
/// exactly the general path's value for every pair of norm bytes -- the
/// ones that decode as negative `i8` included -- for one field and two,
/// whatever the weights, on a first (computed) and a second (remembered)
/// lookup, and step aside for a document its arrays do not cover.
#[test]
fn dense_memo_is_the_general_multi_norm_for_every_byte_pair() {
    let n = 1usize << 16;
    let low: Vec<u8> = (0..n).map(|d| (d & 0xFF) as u8).collect();
    let high: Vec<u8> = (0..n).map(|d| (d >> 8) as u8).collect();
    let (a, b) = (dense_norms(&low), dense_norms(&high));
    for (wa, wb) in [(1.0f32, 1.0f32), (2.5, 3.0), (1.0, 7.25), (13.0, 1.5)] {
        let fields = vec![(a.cursor(), wa), (b.cursor(), wb)];
        let mut memo = DenseMemo::new(&fields).expect("both fields are dense");
        let mut general = Norms::Multi(fields, None);
        for pass in 0..2 {
            for batch in [false, true] {
                for doc in 0..n as i32 {
                    let want = general.norm(doc, batch).unwrap();
                    assert_eq!(
                        memo.norm(doc, batch),
                        Some(want),
                        "{wa}/{wb} doc {doc} pass {pass} batch {batch}"
                    );
                }
            }
        }
        assert_eq!(memo.norm(n as i32, false), None);
        assert_eq!(memo.norm(-1, true), None);
    }
    let fields = vec![(a.cursor(), 3.5f32)];
    let mut memo = DenseMemo::new(&fields).expect("one dense field");
    let mut general = Norms::Multi(fields, None);
    for doc in 0..256 {
        for batch in [false, true] {
            assert_eq!(
                memo.norm(doc, batch),
                Some(general.norm(doc, batch).unwrap())
            );
        }
    }
    // Three fields, or a field that is not dense one-byte, have none.
    let three = vec![(a.cursor(), 1.0), (b.cursor(), 1.0), (a.cursor(), 1.0)];
    assert!(DenseMemo::new(&three).is_none());
    let unnormed = crate::field_norms::FieldNorms::unnormed(10, 1.0);
    assert!(DenseMemo::new(&[(unnormed.cursor(), 1.0)]).is_none());
    // `Norms::norm` answers from the memo under either rule.
    let fields = vec![(a.cursor(), 1.0f32), (b.cursor(), 1.0f32)];
    let memo = DenseMemo::new(&fields);
    let mut with_memo = Norms::Multi(fields, memo);
    let mut without = Norms::Multi(vec![(a.cursor(), 1.0), (b.cursor(), 1.0)], None);
    for doc in [0, 1, 255, 256, 40_000] {
        for batch in [false, true] {
            assert_eq!(
                with_memo.norm(doc, batch).unwrap(),
                without.norm(doc, batch).unwrap()
            );
        }
    }
}
