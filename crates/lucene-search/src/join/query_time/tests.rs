#![cfg(test)]
use super::*;
use crate::query::TermQuery;

fn set(terms: &[&str]) -> Arc<[Vec<u8>]> {
    terms.iter().map(|t| t.as_bytes().to_vec()).collect()
}

#[test]
fn java_float_min_and_max() {
    assert!(java_min_f32(f32::NAN, 1.0).is_nan());
    assert!(java_max_f32(1.0, f32::NAN).is_nan());
    assert!(java_min_f32(0.0, -0.0).is_sign_negative());
    assert!(java_min_f32(-0.0, 0.0).is_sign_negative());
    assert!(java_max_f32(-0.0, 0.0).is_sign_positive());
    assert!(java_max_f32(0.0, -0.0).is_sign_positive());
    assert_eq!(java_min_f32(1.0, 2.0), 1.0);
    assert_eq!(java_min_f32(2.0, 1.0), 1.0);
    assert_eq!(java_max_f32(1.0, 2.0), 2.0);
    assert_eq!(java_max_f32(2.0, 1.0), 2.0);
    assert!(is_positive_zero_f64(0.0) && !is_positive_zero_f64(-0.0));
}

/// `SeekingTermSetTermsEnum.accept` over a walk of the enum's terms:
/// each seek, each match, the catch-up loop and the end.
#[test]
fn the_term_set_filter_seeks_and_accepts_as_java_does() {
    let mut f = SeekingTermSet::new(set(&["b", "d", "f"]));
    assert_eq!(f.next_seek(), Some(&b"b"[..]));
    assert_eq!(f.next_seek(), None, "a seek is handed over once");
    // The enum landed on `b`: accepted, seek on to `d`.
    assert_eq!(f.accept_term(b"b"), AcceptStatus::YesAndSeek);
    assert_eq!(f.next_seek(), Some(&b"d"[..]));
    // It landed past `d`, on `e`: catch up to `f`, which is past `e`.
    assert_eq!(f.accept_term(b"e"), AcceptStatus::NoAndSeek);
    assert_eq!(f.next_seek(), Some(&b"f"[..]));
    // On the last term: accepted, nothing more to seek.
    assert_eq!(f.accept_term(b"f"), AcceptStatus::Yes);
    assert_eq!(f.accept_term(b"g"), AcceptStatus::End);

    // Catching up onto an exact match, with more to come and at the end.
    let mut f = SeekingTermSet::new(set(&["a", "c", "e"]));
    f.next_seek();
    assert_eq!(f.accept_term(b"c"), AcceptStatus::YesAndSeek);
    assert_eq!(f.next_seek(), Some(&b"e"[..]));
    let mut f = SeekingTermSet::new(set(&["a", "c"]));
    f.next_seek();
    assert_eq!(f.accept_term(b"c"), AcceptStatus::Yes);
    // Behind on the last term: rejected.
    let mut f = SeekingTermSet::new(set(&["a", "c"]));
    f.next_seek();
    assert_eq!(f.accept_term(b"b"), AcceptStatus::NoAndSeek);
    assert_eq!(f.accept_term(b"bb"), AcceptStatus::No);
    let mut f = SeekingTermSet::new(set(&["a"]));
    assert_eq!(f.accept_term(b"0"), AcceptStatus::No);
    // An empty set ends at once.
    let mut f = SeekingTermSet::new(set(&[]));
    assert_eq!(f.next_seek(), None);
    assert_eq!(f.accept_term(b"a"), AcceptStatus::End);
    let mut f = SeekingTermSet::new(set(&["x"]));
    assert_eq!(f.accept(b"x").unwrap(), AcceptStatus::Yes);
    assert_eq!(f.next_seek_term(None, None).unwrap(), Some(b"x".to_vec()));
}

#[test]
fn numeric_types_encode_as_their_points() {
    assert_eq!(NumericType::Int.encode(1), vec![0x80, 0, 0, 1]);
    assert_eq!(NumericType::Int.encode(-1), vec![0x7f, 0xff, 0xff, 0xff]);
    assert_eq!(NumericType::Long.encode(0), vec![0x80, 0, 0, 0, 0, 0, 0, 0]);
    let f = NumericType::Float.encode(i64::from(1.5f32.to_bits() as i32));
    assert_eq!(f.len(), 4);
    let d = NumericType::Double.encode(2.5f64.to_bits() as i64);
    assert_eq!(d.len(), 8);
    // Every NaN encodes as Java's one NaN.
    assert_eq!(
        NumericType::Float.encode(0x7fc0_0001),
        NumericType::Float.encode(0x7fc0_0000)
    );
    assert_eq!(
        NumericType::Double.encode(0x7ff8_0000_0000_0001),
        NumericType::Double.encode(0x7ff8_0000_0000_0000)
    );
    assert_eq!(NumericType::Int.render(&NumericType::Int.encode(-7)), "-7");
    assert_eq!(NumericType::Long.render(&NumericType::Long.encode(9)), "9");
    assert_eq!(
        NumericType::Float.render(&NumericType::Float.encode(i64::from((-1.5f32).to_bits()))),
        "-1.5"
    );
    assert_eq!(
        NumericType::Double.render(&NumericType::Double.encode(0.25f64.to_bits() as i64)),
        "0.25"
    );
    assert_eq!(NumericType::Int.render(&[1, 2]), "unsupported");
    assert_eq!(NumericType::Long.bytes(), 8);
    assert_eq!(NumericType::Float.bytes(), 4);
    assert_eq!(bytes_ref_string(&[0x80, 0x0a]), "[80 a]");
}

#[test]
fn point_set_with_scores_checks_its_stream() {
    let q = || Arc::new(Clause::Term(TermQuery::new("f", b"x".to_vec())));
    let ok = PointInSetIncludingScoreQuery::new(
        ScoreMode::Max,
        q(),
        false,
        "p",
        4,
        NumericType::Int,
        vec![
            (NumericType::Int.encode(1), 1.0),
            (NumericType::Int.encode(2), 2.0),
        ],
    )
    .unwrap();
    assert_eq!(ok.points_and_scores().count(), 2);
    assert!(format!("{ok:?}").contains("p:{1 2}"));
    assert_eq!(ok, ok.clone());
    let err = |stream: Vec<(Vec<u8>, f32)>, bytes: usize| {
        PointInSetIncludingScoreQuery::new(
            ScoreMode::Max,
            q(),
            false,
            "p",
            bytes,
            NumericType::Int,
            stream,
        )
        .unwrap_err()
        .to_string()
    };
    assert!(err(Vec::new(), 0).contains("bytesPerDim must be > 0"));
    assert!(err(vec![(vec![1, 2], 1.0)], 4).contains("packed point length should be 4"));
    let one = NumericType::Int.encode(1);
    let two = NumericType::Int.encode(2);
    assert!(err(vec![(one.clone(), 1.0), (one.clone(), 1.0)], 4)
        .contains("unexpected duplicated value: [80 0 0 1]"));
    assert!(err(vec![(two, 1.0), (one, 1.0)], 4).contains("values are out of order"));
}

#[test]
fn blocks_allocate_a_block_only_when_one_of_its_slots_is_written() {
    let mut b = Blocks::new(BLOCK_SIZE * 3 + 1, f32::INFINITY);
    assert_eq!(b.blocks.len(), 4);
    assert!(b.blocks.iter().all(Option::is_none));
    assert_eq!(b.get(5), f32::INFINITY, "unset before any write");
    *b.get_mut(BLOCK_SIZE * 2 + 7).unwrap() = 1.5;
    assert_eq!(
        b.blocks.iter().map(Option::is_some).collect::<Vec<_>>(),
        [false, false, true, false]
    );
    assert_eq!(b.get(BLOCK_SIZE * 2 + 7), 1.5);
    assert_eq!(b.get(BLOCK_SIZE * 2 + 8), f32::INFINITY, "the block's fill");
    assert!(b.get_mut(BLOCK_SIZE * 4).is_none(), "past the end");
    assert_eq!(b.get(BLOCK_SIZE * 4), f32::INFINITY);
    assert!(Blocks::<i32>::new(0, 0).get_mut(0).is_none());
}

#[test]
fn pending_ords_fold_each_term_in_document_order_across_drains() {
    let mut p = PendingOrds::default();
    p.reset(10);
    let mut lookups = Vec::new();
    let mut seen: Vec<(usize, f32)> = Vec::new();
    let mut ids: HashMap<Vec<u8>, usize> = HashMap::new();
    fn drain(
        p: &mut PendingOrds,
        lookups: &mut Vec<i64>,
        seen: &mut Vec<(usize, f32)>,
        ids: &mut HashMap<Vec<u8>, usize>,
    ) {
        p.drain(
            |o| {
                lookups.push(o);
                Ok(format!("t{o}").into_bytes())
            },
            |t, score| {
                let id = match t {
                    Resolved::Term(t) => {
                        let n = ids.len();
                        *ids.entry(t.to_vec()).or_insert(n)
                    }
                    Resolved::Id(id) => id,
                };
                seen.push((id, score));
                Ok(id)
            },
        )
        .unwrap();
    }
    // Two drains: ordinal 3 spans both, and is looked up once.
    assert!(!p.push(3, 1.0));
    p.push(1, 2.0);
    p.push(3, 3.0);
    drain(&mut p, &mut lookups, &mut seen, &mut ids);
    p.push(3, 4.0);
    p.push(-1, 5.0);
    p.push(-1, 6.0);
    drain(&mut p, &mut lookups, &mut seen, &mut ids);
    assert_eq!(
        lookups,
        [1, 3, -1],
        "an ordinal's term is read once per segment"
    );
    let of = |id: usize| -> Vec<f32> {
        seen.iter()
            .filter(|&&(i, _)| i == id)
            .map(|&(_, s)| s)
            .collect()
    };
    assert_eq!(of(ids[b"t3".as_slice()]), [1.0, 3.0, 4.0], "document order");
    assert_eq!(of(ids[b"t1".as_slice()]), [2.0]);
    assert!(p.pairs.is_empty());
    // A new segment forgets the ids.
    p.reset(10);
    p.push(3, 7.0);
    drain(&mut p, &mut lookups, &mut seen, &mut ids);
    assert_eq!(lookups, [1, 3, -1, 3]);
    // The limit asks for a drain.
    let mut p = PendingOrds::default();
    assert!((0..PENDING_LIMIT)
        .map(|i| p.push(i as i64, 0.0))
        .last()
        .unwrap());
}

#[test]
fn numeric_join_sums_start_from_the_first_score_as_hppc_add_to_does() {
    // `LongFloatHashMap.addTo` inserts the first score itself, so a lone
    // `-0.0` stays `-0.0` (summed from `0.0` it would turn `+0.0`).
    for mode in [ScoreMode::Total, ScoreMode::Avg] {
        let mut c = NumericJoinCollector::new("n", false, mode);
        c.add(7, -0.0);
        c.add(8, 1.5);
        c.add(8, 2.0);
        let got = c.into_sorted();
        assert_eq!(got[0].0, 7);
        assert_eq!(got[0].1.to_bits(), (-0.0f32).to_bits(), "{mode:?}");
        let eight = if mode == ScoreMode::Avg { 1.75 } else { 3.5 };
        assert_eq!(got[1], (8, eight), "{mode:?}");
    }
}

#[test]
fn collected_ordinals_match_and_score() {
    let mut scores = Blocks::new(4, 0.0);
    *scores.get_mut(0).unwrap() = 6.0;
    let mut occurrences = Blocks::new(4, 0);
    *occurrences.get_mut(0).unwrap() = 3;
    let mut c = CollectedOrdinals {
        score_mode: ScoreMode::Avg,
        double: true,
        collected: FixedBitSet::new(4),
        scores,
        occurrences,
        do_min_max: true,
        min: 2,
        max: 3,
    };
    c.collected.set(0);
    assert!(c.matches(0));
    assert!(!c.matches(1), "never collected");
    assert!(!c.matches(-1) && !c.matches(9));
    assert_eq!(c.score(0), 2.0);
    assert_eq!(c.score(-1), 0.0);
    c.max = 2;
    assert!(!c.matches(0), "three occurrences, at most two");
    c.score_mode = ScoreMode::None;
    assert_eq!(c.score(0), 1.0);
    c.score_mode = ScoreMode::Max;
    assert_eq!(c.score(0), 6.0);
    c.do_min_max = false;
    assert!(c.matches(0));
}

#[test]
fn global_ordinal_lookups() {
    assert_eq!(global_ord(None, 5).unwrap(), 5);
    assert_eq!(global_ord(Some(&[3, 7]), 1).unwrap(), 7);
    assert!(global_ord(Some(&[3]), 1).is_err());
    assert!(global_ord(Some(&[3]), -1).is_err());
    let mut b = FixedBitSet::new(3);
    set_bit(&mut b, 2);
    set_bit(&mut b, 9);
    set_bit(&mut b, -1);
    assert!(bit(&b, 2) && !bit(&b, 9) && !bit(&b, -1) && !bit(&b, 0));
    assert!(segment_map(None, 0).unwrap().is_none());
}

#[test]
fn collectors_score_in_javas_order() {
    // Single-valued: the first score is taken, a later one combined --
    // and a combined score of exactly `+0.0` is overwritten.
    let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::Total);
    let id = c.slot(b"a").unwrap();
    c.collect_sv(id, 1.0);
    c.collect_sv(id, 2.0);
    assert_eq!(c.score_sums[id], 3.0);
    c.collect_sv(id, -3.0);
    c.collect_sv(id, 5.0);
    assert_eq!(c.score_sums[id], 5.0, "a zero sum restarts");
    for (mode, want) in [(ScoreMode::Min, 1.0), (ScoreMode::Max, 4.0)] {
        let mut c = TermsWithScoreCollector::new("f", false, mode);
        let id = c.slot(b"a").unwrap();
        for s in [2.0, 1.0, 4.0] {
            c.collect_sv(id, s);
        }
        assert_eq!(c.score_sums[id], want, "{mode}");
    }
    let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::Avg);
    let id = c.slot(b"a").unwrap();
    for s in [1.0, 2.0, 6.0] {
        c.collect_sv(id, s);
    }
    let (terms, ords, scores) = c.into_terms_and_scores();
    assert_eq!((terms.len(), ords, scores), (1, vec![0], vec![3.0]));

    for (mode, want) in [
        (ScoreMode::Total, 7.0),
        (ScoreMode::Min, 1.0),
        (ScoreMode::Max, 4.0),
        (ScoreMode::Avg, 7.0 / 3.0),
        (ScoreMode::None, 0.0),
    ] {
        let mut c = TermsWithScoreCollector::new("f", true, mode);
        let id = c.slot(b"z").unwrap();
        let other = c.slot(b"a").unwrap();
        for s in [2.0, 1.0, 4.0] {
            c.collect_mv(id, s);
        }
        assert_eq!(other, 1);
        let (terms, ords, scores) = c.into_terms_and_scores();
        assert_eq!(terms, vec![b"z".to_vec(), b"a".to_vec()]);
        assert_eq!(ords, vec![1, 0]);
        assert_eq!(scores[0], want as f32, "{mode}");
    }
    // `SV` with `None` (which the factory never builds) does nothing.
    let mut c = TermsWithScoreCollector::new("f", false, ScoreMode::None);
    let id = c.slot(b"a").unwrap();
    c.collect_sv(id, 2.0);
    c.collect_sv(id, 3.0);
    assert_eq!(c.score_sums[id], 2.0);
}

#[test]
fn numeric_collector_aggregates_per_value() {
    for (mode, want) in [
        (ScoreMode::Max, 4.0f32),
        (ScoreMode::Min, 1.0),
        (ScoreMode::Total, 7.0),
        (ScoreMode::Avg, 7.0 / 3.0),
        (ScoreMode::None, 0.0),
    ] {
        let mut c = NumericJoinCollector::new("f", false, mode);
        for s in [2.0, 1.0, 4.0] {
            c.add(5, s);
        }
        c.add(-1, 1.0);
        let sorted = c.into_sorted();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].0, -1);
        assert_eq!(sorted[1], (5, want), "{mode}");
    }
}

#[test]
fn queries_compare_and_print() {
    let from = Arc::new(Clause::Term(TermQuery::new("f", b"x".to_vec())));
    let source = TermSetSource {
        field: "to".into(),
        terms: set(&["a", "b"]),
        from_field: "from".into(),
        from_query: Arc::clone(&from),
    };
    assert!(source.contains(b"b") && !source.contains(b"c"));
    let mut copy = source.clone();
    copy.terms = set(&["a", "b"]);
    assert_eq!(source, copy, "equal sets");
    copy.terms = set(&["a"]);
    assert_ne!(source, copy);
    assert!(format!("{source:?}").starts_with("TermsQuery{field=to"));

    let q = TermsIncludingScoreQuery {
        score_mode: ScoreMode::Max,
        to_field: "to".into(),
        multiple_values_per_document: false,
        terms: set(&["a"]),
        scores: vec![1.5].into(),
        from_field: "from".into(),
        from_query: Arc::clone(&from),
    };
    let mut other = q.clone();
    assert_eq!(q, other);
    other.terms = set(&["a"]);
    assert_eq!(q, other, "equal terms and scores");
    other.scores = vec![2.5].into();
    assert_ne!(q, other);
    assert!(format!("{q:?}").starts_with("TermsIncludingScoreQuery{field=to;"));
    assert!(matches!(
        q.as_terms_query(),
        Clause::Extended(e) if matches!(*e, ExtendedQuery::MultiTerm(_))
    ));

    let g = GlobalOrdinalsQuery {
        found_ords: Arc::new(FixedBitSet::new(2)),
        join_field: "gj".into(),
        ordinal_map: None,
        leaves: Arc::new(LeafOrds::default()),
        to_query: Box::new((*from).clone()),
        from_query: Arc::clone(&from),
    };
    let mut g2 = g.clone();
    assert_eq!(g, g2);
    g2.found_ords = Arc::new(FixedBitSet::new(2));
    assert_eq!(g, g2, "equal ordinals");
    assert!(format!("{g:?}").starts_with("GlobalOrdinalsQuery{joinField=gj}"));
    let w = GlobalOrdinalsWithScoreQuery {
        collected: Arc::new(CollectedOrdinals {
            score_mode: ScoreMode::None,
            double: true,
            collected: FixedBitSet::new(2),
            scores: Blocks::new(0, 0.0),
            occurrences: Blocks::new(0, 0),
            do_min_max: false,
            min: 0,
            max: i32::MAX,
        }),
        score_mode: ScoreMode::None,
        join_field: "gj".into(),
        ordinal_map: None,
        leaves: Arc::new(LeafOrds::default()),
        to_query: Box::new((*from).clone()),
        from_query: from,
        min: 0,
        max: i32::MAX,
    };
    assert_eq!(w, w.clone());
    assert!(format!("{w:?}").contains("min=0"));
    assert_eq!(w.as_global_ordinals_query().join_field, "gj");
}

/// Total scores are summed in double and narrowed once (Lucene 10.6, #16378,
/// #16428), so the order the documents come in cannot change the float.
#[test]
fn numeric_join_totals_do_not_depend_on_document_order() {
    let scores = [1.0e8f32, 1.0, 1.0, -1.0e8, 0.1, 0.2];
    let mut forward = NumericJoinCollector::new("f", false, ScoreMode::Total);
    let mut backward = NumericJoinCollector::new("f", false, ScoreMode::Total);
    for &s in &scores {
        forward.add(7, s);
    }
    for &s in scores.iter().rev() {
        backward.add(7, s);
    }
    assert_eq!(forward.into_sorted(), backward.into_sorted());
}

/// Before Lucene 10.6 (OpenSearch 3.9's 10.5.1) the totals add up in `float`,
/// one rounding per document: exactly the `f32` running sum, in document order.
#[test]
fn numeric_join_totals_add_in_float_before_10_6() {
    let scores = [1.0e8f32, 1.0, 1.0, -1.0e8, 0.1, 0.2];
    let host = Version::LUCENE_10_5_1;
    let mut forward = NumericJoinCollector::new("f", false, ScoreMode::Total).for_lucene(host);
    let mut backward = NumericJoinCollector::new("f", false, ScoreMode::Total).for_lucene(host);
    for &s in &scores {
        forward.add(7, s);
    }
    for &s in scores.iter().rev() {
        backward.add(7, s);
    }
    let float_sum = scores[1..].iter().fold(scores[0], |a, &b| a + b);
    assert_eq!(forward.into_sorted(), vec![(7, float_sum)]);
    assert_ne!(
        backward.into_sorted()[0].1.to_bits(),
        float_sum.to_bits(),
        "float addition is order dependent"
    );
}

/// `Avg` divides in float before 10.6 too.
#[test]
fn numeric_join_averages_divide_in_float_before_10_6() {
    let scores = [0.1f32, 0.2, 0.7];
    let mut old =
        NumericJoinCollector::new("f", false, ScoreMode::Avg).for_lucene(Version::LUCENE_10_5_1);
    let mut new =
        NumericJoinCollector::new("f", false, ScoreMode::Avg).for_lucene(Version::LUCENE_10_6_0);
    for &s in &scores {
        old.add(1, s);
        new.add(1, s);
    }
    let float_avg = scores[1..].iter().fold(scores[0], |a, &b| a + b) / 3.0f32;
    let double_avg = ((f64::from(0.1f32) + f64::from(0.2f32) + f64::from(0.7f32)) / 3.0) as f32;
    assert_eq!(old.into_sorted(), vec![(1, float_avg)]);
    assert_eq!(new.into_sorted(), vec![(1, double_avg)]);
}
