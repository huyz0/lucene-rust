use super::*;

fn t(s: &str) -> IntervalsSource {
    Intervals::term(s)
}

/// The factories rewrite as Java's do: duplicates repeat, disjunctions are
/// pulled up and flattened, phrases flatten, single sources unwrap.
#[test]
fn the_factories_rewrite_as_lucene_does() {
    assert_eq!(Intervals::ordered(vec![t("a")]), t("a"));
    assert_eq!(Intervals::unordered(vec![t("a")]), t("a"));
    assert_eq!(
        Intervals::ordered(vec![t("a"), t("a")]).to_string(),
        "ORDERED(a,a)"
    );
    assert_eq!(
        Intervals::unordered(vec![t("a"), t("a")]).to_string(),
        "UNORDERED(a,a)"
    );
    assert_eq!(
        Intervals::ordered(vec![t("a"), t("a"), t("b"), t("a")]).to_string(),
        "ORDERED(a,a,b,a)"
    );
    assert_eq!(
        Intervals::unordered(vec![t("a"), t("b"), t("a")]).to_string(),
        "UNORDERED(a,a,b)"
    );
    let phrase =
        Intervals::phrase(vec![Intervals::phrase_terms(&["a", "b"]).unwrap(), t("c")]).unwrap();
    assert_eq!(phrase.to_string(), "BLOCK(a,b,c)");
    assert_eq!(Intervals::phrase_terms(&["a"]).unwrap(), t("a"));
    assert_eq!(Intervals::phrase(vec![t("x")]).unwrap(), t("x"));
    let pulled = Intervals::phrase(vec![
        Intervals::or(vec![t("a"), Intervals::phrase_terms(&["b", "c"]).unwrap()]).unwrap(),
        t("d"),
    ])
    .unwrap();
    assert_eq!(pulled.to_string(), "or(BLOCK(a,d),BLOCK(b,c,d))");
    let kept = Intervals::phrase(vec![
        Intervals::or_with_rewrite(
            false,
            vec![t("a"), Intervals::phrase_terms(&["b", "c"]).unwrap()],
        )
        .unwrap(),
        t("d"),
    ])
    .unwrap();
    assert_eq!(kept.to_string(), "BLOCK(or(BLOCK(b,c),a),d)");
    // A disjunction of one is the one; nested disjunctions flatten, deduplicated.
    assert_eq!(Intervals::or(vec![t("a"), t("a")]).unwrap(), t("a"));
    let flat = Intervals::or(vec![t("a"), Intervals::or(vec![t("b"), t("a")]).unwrap()]).unwrap();
    assert_eq!(flat.to_string(), "or(a,b)");
    // Set equality.
    assert_eq!(
        Intervals::or(vec![t("a"), t("b")]).unwrap(),
        Intervals::or(vec![t("b"), t("a")]).unwrap()
    );
    assert_eq!(
        Intervals::unordered_no_overlaps(t("a"), t("b"))
            .unwrap()
            .to_string(),
        "or(ORDERED(a,b),ORDERED(b,a))"
    );
}

/// `toString` and `minExtent` of every source.
#[test]
fn every_source_prints_and_measures_as_lucene() {
    let cases: Vec<(IntervalsSource, &str, i32)> = vec![
        (t("a"), "a", 1),
        (
            Intervals::term_with_payload_filter("a", PayloadFilter::new(|_| true)),
            "PAYLOAD_FILTERED(a)",
            1,
        ),
        (
            Intervals::maxgaps(2, Intervals::ordered(vec![t("a"), t("b")])).unwrap(),
            "MAXGAPS/2(ORDERED(a,b))",
            2,
        ),
        (
            Intervals::maxwidth(3, Intervals::unordered(vec![t("a"), t("b")])),
            "MAXWIDTH/3(UNORDERED(a,b))",
            2,
        ),
        (Intervals::extend(t("a"), 1, 2), "EXTEND(a,1,2)", 4),
        (
            Intervals::extend(t("a"), i32::MAX, 0),
            "EXTEND(a,2147483647,0)",
            i32::MAX,
        ),
        (Intervals::fix_field("g", t("a")), "FIELD(g,a)", 1),
        (Intervals::no_intervals("why"), "NOMATCH(why)", 0),
        (
            Intervals::containing(t("a"), t("b")).unwrap(),
            "CONTAINING(a,b)",
            1,
        ),
        (
            Intervals::contained_by(t("a"), t("b")).unwrap(),
            "CONTAINED_BY(a,b)",
            1,
        ),
        (
            Intervals::not_containing(t("a"), t("b")).unwrap(),
            "NOT_CONTAINING(a,b)",
            1,
        ),
        (
            Intervals::not_contained_by(t("a"), t("b")).unwrap(),
            "NOT_CONTAINED_BY(a,b)",
            1,
        ),
        (
            Intervals::overlapping(t("a"), t("b")),
            "OVERLAPPING(a,b)",
            1,
        ),
        (
            Intervals::non_overlapping(t("a"), t("b")),
            "NON_OVERLAPPING(a,b)",
            1,
        ),
        (
            Intervals::not_within(t("a"), 2, t("b")),
            "NON_OVERLAPPING(a,EXTEND(b,2,2))",
            1,
        ),
        (
            Intervals::within(t("a"), 2, t("b")).unwrap(),
            "CONTAINED_BY(a,EXTEND(b,2,2))",
            1,
        ),
        (
            Intervals::before(t("a"), t("b")).unwrap(),
            "CONTAINED_BY(a,EXTEND(PRECEDING(b),2147483647,0))",
            1,
        ),
        (
            Intervals::after(t("a"), t("b")).unwrap(),
            "CONTAINED_BY(a,EXTEND(FOLLOWING(b),0,2147483647))",
            1,
        ),
        (
            Intervals::at_least(
                2,
                vec![
                    t("a"),
                    Intervals::phrase_terms(&["b", "c"]).unwrap(),
                    t("d"),
                ],
            ),
            "AtLeast(a,BLOCK(b,c),d~2)",
            2,
        ),
        (
            Intervals::at_least(3, vec![t("a"), t("b")]),
            "NOMATCH(Too few sources to match minimum of [3]: [a, b])",
            0,
        ),
        (
            Intervals::at_least(2, vec![t("a"), t("b")]),
            "UNORDERED(a,b)",
            2,
        ),
        (Intervals::prefix("ap", 10).unwrap(), "MultiTerm(ap*)", 1),
        (Intervals::wildcard("a?c", 10).unwrap(), "MultiTerm(a?c)", 1),
        (Intervals::regexp("a.*", 10).unwrap(), "MultiTerm(a.*)", 1),
        (
            Intervals::range(None, Some(b"m".to_vec()), true, false, 10).unwrap(),
            "MultiTerm({* ,m})",
            1,
        ),
        (
            Intervals::range(Some(b"c".to_vec()), None, true, false, 10).unwrap(),
            "MultiTerm({c,*})",
            1,
        ),
        (
            Intervals::fuzzy_term("cat", 1, 0, true, 10).unwrap(),
            "MultiTerm(cat~1)",
            1,
        ),
        (
            Intervals::or(vec![t("b"), Intervals::ordered(vec![t("a"), t("c")])]).unwrap(),
            "or(ORDERED(a,c),b)",
            1,
        ),
        (Intervals::or(vec![]).unwrap(), "or()", i32::MAX),
    ];
    for (s, printed, extent) in cases {
        assert_eq!(s.to_string(), printed);
        assert_eq!(s.min_extent(), extent, "{printed}");
    }
    // `minExtent` sums wrap as Java's do.
    let big = Intervals::ordered(vec![
        Intervals::extend(t("a"), i32::MAX - 1, 0),
        t("b"),
        t("c"),
    ]);
    assert_eq!(big.min_extent(), i32::MAX.wrapping_add(2));
}

/// Equality is each Java class's: the payload filter and a repeat's name
/// are not compared, the score function is not part of the query's.
#[test]
fn equality_follows_the_java_classes() {
    let f1 = Intervals::term_with_payload_filter("a", PayloadFilter::new(|_| true));
    let f2 = Intervals::term_with_payload_filter("a", PayloadFilter::new(|_| false));
    assert_eq!(f1, f2);
    assert_ne!(f1, t("a"));
    assert_eq!(
        format!("{:?}", PayloadFilter::new(|_| true)),
        "PayloadFilter"
    );
    let named = Intervals::ordered(vec![t("a"), t("a")]);
    let unnamed = Intervals::ordered(vec![Intervals::ordered(vec![t("a"), t("a")]), t("b")]);
    let IntervalsSource::Ordered(subs) = &unnamed else {
        panic!("{unnamed}")
    };
    assert_eq!(subs[0], named);
    let q1 = IntervalQuery::new("f", t("a"));
    let q2 = IntervalQuery::with_pivot("f", t("a"), 3.0).unwrap();
    assert_eq!(q1, q2);
    assert_ne!(q1, IntervalQuery::new("g", t("a")));
    assert_eq!(q1.to_string(), "f:a");
    assert_eq!(q1.to_string_with_field("f"), "a");
    for (a, b) in [
        (
            Intervals::extend(t("a"), 1, 2),
            Intervals::extend(t("a"), 1, 3),
        ),
        (
            Intervals::fix_field("f", t("a")),
            Intervals::fix_field("g", t("a")),
        ),
        (
            Intervals::maxgaps(1, t("a")).unwrap(),
            Intervals::maxwidth(1, t("a")),
        ),
        (
            Intervals::non_overlapping(t("a"), t("b")),
            Intervals::overlapping(t("a"), t("b")),
        ),
        (
            Intervals::at_least(1, vec![t("a"), t("b")]),
            Intervals::at_least(1, vec![t("b"), t("a")]),
        ),
        (
            Intervals::prefix("a", 5).unwrap(),
            Intervals::prefix("a", 6).unwrap(),
        ),
        (Intervals::no_intervals("x"), Intervals::no_intervals("y")),
    ] {
        assert_ne!(a, b, "{a} vs {b}");
        assert_eq!(a, a.clone());
    }
    let offset = |before| IntervalsSource::Offset {
        source: Box::new(t("a")),
        before,
    };
    assert_ne!(offset(true), offset(false));
    let auto = |s: &str| Intervals::fuzzy_term(s, 1, 0, true, 5).unwrap();
    assert_eq!(auto("cat"), auto("cat"));
    let IntervalsSource::MultiTerm { pattern: a, .. } = auto("cat") else {
        panic!()
    };
    let IntervalsSource::MultiTerm { pattern: b, .. } = auto("cot") else {
        panic!()
    };
    assert_ne!(a, b);
    assert_ne!(a, MultiTermPattern::Prefix(b"c".to_vec()));
    assert_eq!(
        MultiTermPattern::Range {
            lower: None,
            upper: None,
            include_lower: true,
            include_upper: true
        },
        MultiTermPattern::Range {
            lower: None,
            upper: None,
            include_lower: true,
            include_upper: true
        }
    );
    assert_eq!(
        MultiTermPattern::Regexp("a".into()),
        MultiTermPattern::Regexp("a".into())
    );
}

/// Every pull-up: a source's disjuncts, combined.
#[test]
fn disjunctions_pull_up_through_every_gap_sensitive_source() {
    let or_ab =
        || Intervals::or(vec![t("a"), Intervals::phrase_terms(&["b", "c"]).unwrap()]).unwrap();
    let ordered = Intervals::ordered(vec![or_ab(), t("d")]);
    let pulled: Vec<String> = ordered
        .pull_up_disjunctions()
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(pulled, ["ORDERED(a,d)", "ORDERED(BLOCK(b,c),d)"]);
    let unordered = Intervals::unordered(vec![or_ab(), t("d")]);
    assert_eq!(unordered.pull_up_disjunctions().unwrap().len(), 2);
    let width = Intervals::maxwidth(3, or_ab());
    assert_eq!(width.pull_up_disjunctions().unwrap().len(), 2);
    let extended = Intervals::extend(or_ab(), 1, 1);
    assert_eq!(extended.pull_up_disjunctions().unwrap().len(), 2);
    let extended_one = Intervals::extend(t("a"), 1, 1);
    assert_eq!(
        extended_one.pull_up_disjunctions().unwrap(),
        vec![extended_one.clone()]
    );
    let fixed = Intervals::fix_field("g", or_ab());
    assert_eq!(fixed.pull_up_disjunctions().unwrap().len(), 2);
    let fixed_one = Intervals::fix_field("g", t("a"));
    assert_eq!(
        fixed_one.pull_up_disjunctions().unwrap(),
        vec![fixed_one.clone()]
    );
    let overlapping = Intervals::overlapping(or_ab(), t("z"));
    assert_eq!(overlapping.pull_up_disjunctions().unwrap().len(), 2);
    // A disjunction that keeps itself whole.
    let kept = Intervals::or_with_rewrite(false, vec![t("a"), t("b")]).unwrap();
    assert_eq!(kept.pull_up_disjunctions().unwrap(), vec![kept.clone()]);
    // `containing`/`containedBy`'s own pull-ups.
    let c = IntervalsSource::Containing {
        big: Box::new(or_ab()),
        small: Box::new(t("z")),
    };
    assert_eq!(c.pull_up_disjunctions().unwrap().len(), 2);
    let cb = IntervalsSource::ContainedBy {
        small: Box::new(t("z")),
        big: Box::new(or_ab()),
    };
    assert_eq!(cb.pull_up_disjunctions().unwrap().len(), 2);
}

/// Past `getMaxClauseCount()` disjunct combinations, the pull-up refuses.
#[test]
fn too_many_disjunctions_are_refused() {
    let wide = || {
        let alts: Vec<IntervalsSource> = (0..40)
            .map(|i| Intervals::phrase_terms(&[&format!("a{i}"), "x"]).unwrap())
            .collect();
        Intervals::or(alts).unwrap()
    };
    let e = Intervals::phrase(vec![wide(), wide()]).unwrap_err();
    assert!(
        matches!(e, Error::IllegalArgument(ref m) if m == "Too many disjunctions to expand"),
        "{e:?}"
    );
    assert!(Intervals::prefix("a", MAX_CLAUSE_COUNT + 1).is_err());
    assert!(Intervals::regexp("a[", 10).is_err());
    assert!(Intervals::fuzzy_term("a", 3, 0, true, 10).is_err());
    assert!(Intervals::fuzzy_term("a", 1, -1, true, 10).is_err());
}

/// The scoring functions: validation, the formulas, and their explanations.
#[test]
fn score_functions_validate_score_and_explain() {
    for bad in [0.0f32, -1.0, f32::INFINITY, f32::NAN] {
        assert!(IntervalScoreFunction::saturation(bad).is_err());
        assert!(IntervalScoreFunction::sigmoid(bad, 1.0).is_err());
        assert!(IntervalScoreFunction::sigmoid(1.0, bad).is_err());
        assert!(IntervalQuery::with_pivot("f", t("a"), bad).is_err());
        assert!(IntervalQuery::with_pivot_and_exp("f", t("a"), 1.0, bad).is_err());
    }
    let sat = IntervalScoreFunction::saturation(2.0).unwrap();
    assert_eq!(sat.score(3.0, 2.0), 3.0f32 * (1.0 - 2.0 / (2.0 + 2.0)));
    let e = sat.explain("f:a", 3.0, 2.0);
    assert_eq!(e.value, 1.5);
    assert_eq!(e.details.len(), 3);
    assert!(e.details[2].description.ends_with("f:a"));
    let sig = IntervalScoreFunction::sigmoid(1.5, 2.0).unwrap();
    let want = (3.0f64 * (1.0 - 1.5f64.powf(2.0) / (2.0f64.powf(2.0) + 1.5f64.powf(2.0)))) as f32;
    assert_eq!(sig.score(3.0, 2.0), want);
    let e = sig.explain("f:a", 3.0, 2.0);
    assert_eq!(e.value, want);
    assert_eq!(e.details.len(), 4);
}

/// The rest of the equality and pull-up cases: overlapping, an extension of
/// an empty disjunction, a source that keeps itself, and the automaton
/// factory.
#[test]
fn remaining_equalities_and_pull_ups() {
    let o = |a: &str, b: &str| Intervals::overlapping(t(a), t(b));
    assert_eq!(o("a", "b"), o("a", "b"));
    assert_ne!(o("a", "b"), o("b", "a"));
    let empty = Intervals::or(vec![]).unwrap();
    let ext = Intervals::extend(empty, 1, 1);
    assert_eq!(ext.pull_up_disjunctions().unwrap(), vec![ext.clone()]);
    let none = Intervals::no_intervals("n");
    assert_eq!(none.pull_up_disjunctions().unwrap(), vec![none.clone()]);
    let a = lucene_util::automaton::automata::make_string("abc");
    let m = Intervals::multiterm(a, false, 7, "abc").unwrap();
    assert_eq!(m.to_string(), "MultiTerm(abc)");
    assert_eq!(m.min_extent(), 1);
    assert!(Intervals::multiterm(
        lucene_util::automaton::automata::make_string("x"),
        true,
        MAX_CLAUSE_COUNT + 1,
        "x"
    )
    .is_err());
}
