use super::*;
use crate::intervals::{Intervals, PayloadFilter};
use crate::query::MatchAllDocsQuery;

fn occ(positions: &[(i32, i32, i32, u8)]) -> Occurrences {
    positions
        .iter()
        .map(|&(p, s, e, pay)| Position {
            position: p,
            start_offset: s,
            end_offset: e,
            payload: if pay == 0 { Vec::new() } else { vec![pay] },
        })
        .collect::<Vec<_>>()
        .into()
}

/// A document of field `f`: `a` at 0, 3, 6; `b` at 1, 4; `c` at 2.
fn doc() -> DocOccurrences {
    let mut d = DocOccurrences::default();
    let put = |d: &mut DocOccurrences, t: &str, o: Occurrences| {
        d.terms
            .insert(("f".to_string(), t.as_bytes().to_vec()), Some(o));
    };
    put(
        &mut d,
        "a",
        occ(&[(0, 0, 1, 1), (3, 6, 7, 2), (6, 12, 13, 0)]),
    );
    put(&mut d, "b", occ(&[(1, 2, 3, 1), (4, 8, 9, 0)]));
    put(&mut d, "c", occ(&[(2, 4, 5, 3)]));
    d.terms.insert(("f".to_string(), b"none".to_vec()), None);
    d
}

fn fallback() -> Arc<Clause> {
    Arc::new(Clause::MatchAllDocs(MatchAllDocsQuery { max_doc: 1 }))
}

fn with<R>(source: &IntervalsSource, f: impl FnOnce(Option<Shared>) -> R) -> R {
    let d = doc();
    let cx = MatchContext {
        occ: &d,
        doc: 5,
        query: fallback(),
    };
    f(source_matches(source, "f", &cx).unwrap())
}

fn t(s: &str) -> IntervalsSource {
    Intervals::term(s)
}

/// `(start, end, startOffset, endOffset)` of every match.
fn all(mi: &Shared) -> Vec<[i32; 4]> {
    let mut out = Vec::new();
    while mi.borrow_mut().next().unwrap() {
        let m = mi.borrow();
        out.push([
            m.start_position(),
            m.end_position(),
            m.start_offset(),
            m.end_offset(),
        ]);
    }
    out
}

#[test]
fn leaf_matches_report_their_term_or_refuse_a_query() {
    with(&t("a"), |mi| {
        let mi = mi.unwrap();
        assert_eq!(mi.borrow().gaps(), 0);
        assert_eq!(mi.borrow().width(), 1);
        assert!(matches!(&*mi.borrow().query().unwrap(), Clause::Term(_)));
        assert!(mi.borrow_mut().sub_matches().unwrap().is_none());
        assert_eq!(all(&mi), [[0, 0, 0, 1], [3, 3, 6, 7], [6, 6, 12, 13]]);
    });
    let filtered = Intervals::term_with_payload_filter("a", PayloadFilter::new(|p| p.is_some()));
    with(&filtered, |mi| {
        let mi = mi.unwrap();
        assert!(matches!(mi.borrow().query(), Err(Error::Unsupported(_))));
        // The plain view falls back to the enclosing query.
        let plain = as_plain(Rc::clone(&mi), &fallback());
        assert!(matches!(plain.query(), Clause::MatchAllDocs(_)));
        assert_eq!(all(&mi), [[0, 0, 0, 1], [3, 3, 6, 7]]);
    });
    with(&t("none"), |mi| assert!(mi.is_none()));
    with(&t("absent"), |mi| assert!(mi.is_none()));
    with(&Intervals::no_intervals("x"), |mi| assert!(mi.is_none()));
}

#[test]
fn conjunctions_report_positions_offsets_and_sub_matches() {
    let ordered = Intervals::ordered(vec![t("a"), t("b")]);
    with(&ordered, |mi| {
        let mi = mi.unwrap();
        assert!(matches!(mi.borrow().query(), Err(Error::Unsupported(_))));
        assert!(mi.borrow_mut().next().unwrap());
        assert_eq!(mi.borrow().gaps(), 0);
        assert_eq!(mi.borrow().width(), 2);
        let mut sub = mi.borrow_mut().sub_matches().unwrap().unwrap();
        let mut n = 0;
        while sub.next().unwrap() {
            n += 1;
            assert!(matches!(sub.query(), Clause::Term(_)));
        }
        assert_eq!(n, 2);
        let rest = all(&mi);
        assert_eq!(rest, [[3, 4, 6, 9]]);
    });
    let phrase = Intervals::phrase_terms(&["a", "b", "c"]).unwrap();
    with(&phrase, |mi| assert_eq!(all(&mi.unwrap()), [[0, 2, 0, 5]]));
    // A conjunction with a member absent from the document has no matches.
    with(&Intervals::phrase_terms(&["a", "absent"]).unwrap(), |mi| {
        assert!(mi.is_none())
    });
    with(&Intervals::unordered(vec![t("c"), t("absent")]), |mi| {
        assert!(mi.is_none())
    });
    let at_least = Intervals::at_least(2, vec![t("a"), t("b"), t("absent")]);
    with(&at_least, |mi| {
        let mi = mi.unwrap();
        assert!(mi.borrow().query().is_ok());
        assert!(mi.borrow_mut().next().unwrap());
        assert_eq!(mi.borrow().width(), 2);
        assert_eq!(mi.borrow().gaps(), 0);
        assert!(mi.borrow_mut().sub_matches().unwrap().is_some());
    });
    with(
        &Intervals::at_least(2, vec![t("a"), t("x"), t("y")]),
        |mi| assert!(mi.is_none()),
    );
    let contained =
        Intervals::contained_by(t("b"), Intervals::ordered(vec![t("a"), t("c")])).unwrap();
    with(&contained, |mi| {
        assert_eq!(all(&mi.unwrap()), [[1, 1, 2, 3]])
    });
    let overlapping = Intervals::overlapping(t("a"), t("absent"));
    with(&overlapping, |mi| assert!(mi.is_none()));
}

#[test]
fn disjunctions_and_repeats_delegate_to_their_current_sub() {
    let or = Intervals::or(vec![t("c"), Intervals::phrase_terms(&["a", "b"]).unwrap()]).unwrap();
    with(&or, |mi| {
        let mi = mi.unwrap();
        // Before the first match: no current sub.
        assert_eq!(mi.borrow().start_offset(), -1);
        assert!(mi.borrow_mut().sub_matches().unwrap().is_none());
        assert!(mi.borrow().query().is_ok());
        assert!(mi.borrow_mut().next().unwrap());
        assert_eq!(mi.borrow().width(), 2);
        assert_eq!(mi.borrow().gaps(), 0);
        assert!(matches!(mi.borrow().query(), Err(Error::Unsupported(_))));
        assert!(mi.borrow_mut().sub_matches().unwrap().is_some());
    });
    with(&Intervals::or(vec![t("x"), t("y")]).unwrap(), |mi| {
        assert!(mi.is_none())
    });
    let repeat = Intervals::ordered(vec![t("a"), t("a")]);
    with(&repeat, |mi| {
        let mi = mi.unwrap();
        assert!(matches!(mi.borrow().query(), Err(Error::Unsupported(_))));
        assert!(mi.borrow_mut().next().unwrap());
        assert_eq!(mi.borrow().width(), 2);
        assert_eq!(mi.borrow().gaps(), 2);
        assert!(mi.borrow_mut().sub_matches().unwrap().is_some());
    });
    // Not enough copies for a repeat.
    with(&Intervals::ordered(vec![t("c"), t("c")]), |mi| {
        assert!(mi.is_none())
    });
}

#[test]
fn wrappers_keep_their_source_s_offsets_or_hide_them() {
    let ext = Intervals::extend(t("b"), 1, 1);
    with(&ext, |mi| {
        let mi = mi.unwrap();
        assert!(mi.borrow().query().is_ok());
        assert_eq!(mi.borrow().gaps(), 0);
        assert_eq!(mi.borrow().width(), 3);
        assert!(mi.borrow_mut().sub_matches().unwrap().is_none());
        assert_eq!(all(&mi), [[0, 2, -1, -1], [3, 5, -1, -1]]);
    });
    let offset = IntervalsSource::Offset {
        source: Box::new(t("c")),
        before: true,
    };
    with(&offset, |mi| assert_eq!(all(&mi.unwrap()), [[1, 1, 4, 5]]));
    let absent_offset = IntervalsSource::Offset {
        source: Box::new(t("absent")),
        before: false,
    };
    with(&absent_offset, |mi| assert!(mi.is_none()));
    with(&Intervals::maxwidth(1, t("absent")), |mi| {
        assert!(mi.is_none())
    });
    with(&Intervals::extend(t("absent"), 1, 1), |mi| {
        assert!(mi.is_none())
    });
    let fixed = Intervals::fix_field("f", t("c"));
    with(&fixed, |mi| assert_eq!(all(&mi.unwrap()), [[2, 2, 4, 5]]));
}

#[test]
fn a_wrapped_matches_iterator_is_an_iterator_over_one_document() {
    with(&t("c"), |mi| {
        let mut w = wrap(&mi.unwrap(), 5);
        assert_eq!(w.doc_id(), -1);
        assert_eq!(w.cost(), 1);
        assert_eq!(w.match_cost(), 1.0);
        assert_eq!(w.next_doc().unwrap(), 5);
        assert_eq!(w.next_interval().unwrap(), 2);
        assert_eq!(w.next_interval().unwrap(), NO_MORE_INTERVALS);
        assert_eq!(
            (w.start(), w.end(), w.doc_id()),
            (NO_MORE_INTERVALS, NO_MORE_INTERVALS, 5)
        );
        assert_eq!(w.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(w.doc_id(), NO_MORE_DOCS);
        assert_eq!(w.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(w.advance(6).unwrap(), NO_MORE_DOCS);
    });
}
