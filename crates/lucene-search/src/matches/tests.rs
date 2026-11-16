#![cfg(test)]
use super::*;

fn term() -> Arc<Clause> {
    Arc::new(Clause::Term(TermQuery::new("f", b"t".to_vec())))
}

fn list(intervals: &[[i32; 4]]) -> BoxMatchesIterator {
    Box::new(ListMatchesIterator {
        query: term(),
        intervals: intervals.to_vec(),
        at: 0,
    })
}

fn drain(mut it: BoxMatchesIterator) -> Vec<[i32; 4]> {
    let mut out = Vec::new();
    while it.next().unwrap() {
        out.push([
            it.start_position(),
            it.end_position(),
            it.start_offset(),
            it.end_offset(),
        ]);
        assert!(it.sub_matches().unwrap().is_none());
        assert_eq!(clause_name(it.query()), "TermQuery");
    }
    out
}

#[test]
fn disjunction_merges_by_position_then_offset() {
    let a = list(&[[1, 1, 0, 3], [5, 5, 20, 23]]);
    let b = list(&[[2, 4, 5, 15], [5, 5, 20, 23]]);
    let c = list(&[]);
    let d = disjunction(vec![a, b, c]).unwrap().unwrap();
    assert_eq!(
        drain(d),
        vec![[1, 1, 0, 3], [2, 4, 5, 15], [5, 5, 20, 23], [5, 5, 20, 23]]
    );
    // Without positions, by offsets.
    let a = list(&[[-1, -1, 7, 9]]);
    let b = list(&[[-1, -1, 1, 4], [-1, -1, 7, 8]]);
    let d = disjunction(vec![a, b]).unwrap().unwrap();
    assert_eq!(
        drain(d),
        vec![[-1, -1, 1, 4], [-1, -1, 7, 8], [-1, -1, 7, 9]]
    );
    assert!(disjunction(vec![]).unwrap().is_none());
    let one = disjunction(vec![list(&[[3, 3, 1, 2]])]).unwrap().unwrap();
    assert_eq!(drain(one), vec![[3, 3, 1, 2]]);
    let mut empty = DisjunctionMatchesIterator::new(vec![list(&[]), list(&[])]).unwrap();
    assert!(!empty.next().unwrap());
    assert!(!empty.next().unwrap());
    assert!(empty.sub_matches().unwrap().is_none());
}

#[test]
fn filter_iterator_delegates() {
    let mut f = FilterMatchesIterator {
        inner: list(&[[4, 6, 10, 20]]),
    };
    assert!(f.next().unwrap());
    assert_eq!(
        [
            f.start_position(),
            f.end_position(),
            f.start_offset(),
            f.end_offset()
        ],
        [4, 6, 10, 20]
    );
    assert!(f.sub_matches().unwrap().is_none());
    assert_eq!(clause_name(f.query()), "TermQuery");
    assert!(!f.next().unwrap());
    let unstarted = ListMatchesIterator {
        query: term(),
        intervals: vec![[1, 1, 1, 1]],
        at: 0,
    };
    assert_eq!(unstarted.start_position(), -1);
}

fn field(name: &str, intervals: Vec<[i32; 4]>) -> BoxMatches {
    let q = term();
    for_field(
        name,
        Box::new(move || {
            Ok(Some(Box::new(ListMatchesIterator {
                query: Arc::clone(&q),
                intervals: intervals.clone(),
                at: 0,
            }) as BoxMatchesIterator))
        }),
    )
    .unwrap()
    .unwrap()
}

#[test]
fn from_sub_matches_amalgamates() {
    assert!(from_sub_matches(vec![]).is_none());
    let only_no_terms = from_sub_matches(vec![Box::new(MatchWithNoTerms)]).unwrap();
    assert!(only_no_terms.is_match_with_no_terms());
    assert!(only_no_terms.get_matches("f").unwrap().is_none());
    assert!(only_no_terms.fields().is_empty());
    let one = from_sub_matches(vec![
        Box::new(MatchWithNoTerms),
        field("a", vec![[0, 0, 0, 1]]),
    ])
    .unwrap();
    assert_eq!(one.fields(), vec!["a".to_string()]);
    assert!(one.sub_matches().is_empty());
    assert!(one.get_matches("b").unwrap().is_none());
    let many = from_sub_matches(vec![
        field("a", vec![[3, 3, 9, 10]]),
        Box::new(MatchWithNoTerms),
        field("b", vec![[1, 1, 0, 1]]),
        field("a", vec![[1, 1, 2, 3]]),
    ])
    .unwrap();
    assert_eq!(many.fields(), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(many.sub_matches().len(), 4);
    assert_eq!(
        drain(many.get_matches("a").unwrap().unwrap()),
        vec![[1, 1, 2, 3], [3, 3, 9, 10]]
    );
    assert!(many.get_matches("c").unwrap().is_none());
    // `forField` of a supplier with nothing is no matches at all.
    assert!(for_field("x", Box::new(|| Ok(None))).unwrap().is_none());
}

#[test]
fn named_matches_are_found_breadth_first() {
    let inner = NamedMatches::new("inner", field("a", vec![[0, 0, 0, 1]]));
    assert_eq!(inner.name(), "inner");
    assert_eq!(inner.fields(), vec!["a".to_string()]);
    assert!(inner.get_matches("a").unwrap().is_some());
    let tree = from_sub_matches(vec![
        Box::new(NamedMatches::new("x", Box::new(inner))),
        Box::new(NamedMatches::new("y", field("b", vec![[1, 1, 1, 2]]))),
    ])
    .unwrap();
    let names: Vec<&str> = find_named_matches(tree.as_ref())
        .iter()
        .map(|n| n.name())
        .collect();
    assert_eq!(names, vec!["x", "y", "inner"]);
}

fn occ(position: i32, start: i32, end: i32) -> Position {
    Position {
        position,
        start_offset: start,
        end_offset: end,
        payload: Vec::new(),
    }
}

#[test]
fn exact_phrase_spans_follow_the_lead() {
    // "a b" over a(0) b(1) a(2) x(3) a(4) b(5).
    let a = vec![occ(0, 0, 1), occ(2, 4, 5), occ(4, 8, 9)];
    let b = vec![occ(1, 2, 3), occ(5, 10, 11)];
    assert_eq!(
        exact_phrase_spans(&[a.clone(), b.clone()], &[0, 1]),
        vec![[0, 1, 0, 3], [4, 5, 8, 11]]
    );
    // "b a": b(1) a(2) only.
    assert_eq!(exact_phrase_spans(&[b, a], &[0, 1]), vec![[1, 2, 2, 5]]);
    assert!(exact_phrase_spans(&[], &[]).is_empty());
}
