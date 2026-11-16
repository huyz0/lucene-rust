#![cfg(test)]
#[test]
fn child_values_are_converted_before_they_are_picked() {
    let mut v = vec![-250i64, 0, 350];
    assert_eq!(ChildConvert::HalfFloat.apply_one(0x3c00), {
        let mut h = vec![0x3c00];
        ChildConvert::HalfFloat.apply(&mut h);
        h[0]
    });
    ChildConvert::None.apply(&mut v);
    assert_eq!(v, [-250, 0, 350]);
    ChildConvert::ScaledFloat { inverse: 0.01 }.apply(&mut v);
    assert_eq!(
        v.iter()
            .copied()
            .map(sortable_long_to_double)
            .collect::<Vec<_>>(),
        [-250.0 * 0.01, 0.0, 350.0 * 0.01]
    );
    let mut h = vec![
        i64::from(lucene_util::half_float::half_float_to_sortable_short(-1.5)),
        i64::from(lucene_util::half_float::half_float_to_sortable_short(2.25)),
    ];
    ChildConvert::HalfFloat.apply(&mut h);
    assert_eq!(
        h.iter()
            .copied()
            .map(sortable_int_to_float)
            .collect::<Vec<_>>(),
        [-1.5, 2.25]
    );
}

#[test]
fn cmp_bytes_is_unsigned_lexicographic_order() {
    let alphabet = [0u8, 1, 0x7f, 0x80, 0xff];
    let mut all: Vec<Vec<u8>> = vec![Vec::new()];
    for len in 1..=3 {
        let mut next = Vec::new();
        for v in all.iter().filter(|v| v.len() == len - 1) {
            for &b in &alphabet {
                let mut w = v.clone();
                w.push(b);
                next.push(w);
            }
        }
        all.extend(next);
    }
    all.push(vec![1; 9]);
    all.push(vec![1; 8]);
    all.push([vec![1; 8], vec![0]].concat());
    for a in &all {
        for b in &all {
            assert_eq!(super::cmp_bytes(a, b), a.as_slice().cmp(b), "{a:?} {b:?}");
        }
    }
}

use super::*;

/// Whether a key's comparator reverses the value order (`reverseMul`).
fn tf_reverse(key: &SortField) -> bool {
    TopField::new(std::slice::from_ref(key), 1, u64::MAX, None).comps[0].mul < 0
}

/// The single-key queue (keys in the entries, one wide comparison)
/// keeps the general comparator's hits, in its order, over extreme
/// values, ties and both directions.
#[test]
fn plain_key_queue_keeps_the_general_order() {
    let mut x = 7u64;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        x >> 33
    };
    let pool = [i64::MIN, i64::MIN + 1, -1, 0, 1, 5, i64::MAX - 1, i64::MAX];
    for (ty, reverse) in [SortType::Long, SortType::Int, SortType::Score]
        .into_iter()
        .flat_map(|t| [(t, false), (t, true)])
    {
        for n in [1usize, 2, 3, 10, 33] {
            let sort = [match ty {
                SortType::Score => SortField::score(),
                t => SortField::numeric("x", t, reverse),
            }];
            let reverse = tf_reverse(&sort[0]);
            let mut tf = TopField::new(&sort, n, u64::MAX, None);
            assert!(plain_key(&tf.comps).is_some());
            let docs: Vec<(i64, i32)> = (0..500)
                .map(|d| {
                    let r = next();
                    let v = if r % 3 == 0 {
                        pool[(r % 8) as usize]
                    } else {
                        (r % 40) as i64 - 20
                    };
                    (v, d)
                })
                .collect();
            for &(v, doc) in &docs {
                let len = tf.queue.heap.len();
                if len < n {
                    tf.comps[0].values[len] = v;
                    tf.queue.add(
                        &tf.comps,
                        Entry {
                            slot: len,
                            doc,
                            key: 0,
                        },
                    );
                    continue;
                }
                let top = tf.queue.top().unwrap();
                let r = tf.comps[0].mul * cmp(tf.comps[0].values[top.slot], v);
                if r > 0 {
                    tf.comps[0].values[top.slot] = v;
                    tf.queue.replace_top(&tf.comps, Entry { doc, ..top });
                }
            }
            let mut got = Vec::new();
            while let Some(e) = tf.queue.pop(&tf.comps) {
                got.push((tf.comps[0].values[e.slot], e.doc));
            }
            got.reverse();
            let mut want = docs.clone();
            want.sort_by(|a, b| {
                let o = a.0.cmp(&b.0);
                (if reverse { o.reverse() } else { o }).then(a.1.cmp(&b.1))
            });
            want.truncate(n);
            assert_eq!(got, want, "{ty:?} reverse={reverse} n={n}");
        }
    }
}

#[test]
fn keys_and_values_print_as_java() {
    use crate::values_source as vs;
    let keys = [
        (SortField::score(), "<score>"),
        (SortField::doc(), "<doc>"),
        (SortField::string("s", false), "<string: \"s\">"),
        (SortField::string_val("s", true), "<string_val: \"s\">!"),
        (
            SortField::numeric("i", SortType::Int, false),
            "<int: \"i\">",
        ),
        (
            SortField::numeric("l", SortType::Long, true),
            "<long: \"l\">!",
        ),
        (
            SortField::numeric("f", SortType::Float, false),
            "<float: \"f\">",
        ),
        (
            SortField::numeric("d", SortType::Double, false),
            "<double: \"d\">",
        ),
    ];
    for (key, want) in &keys {
        assert_eq!(sort_field_string(key), *want);
    }
    let all: Vec<SortField> = keys.iter().map(|(k, _)| k.clone()).collect();
    assert!(sort_string(&all).starts_with("<score>,<doc>,<string: \"s\">,"));
    assert_eq!(sort_string(&[]), "");

    // Values from the comparable long each key carries.
    let v = |key: &SortField, value: i64| sort_value_string(key, value, None);
    assert_eq!(v(&keys[0].0, i64::from(1.5f32.to_bits())), "1.5");
    assert_eq!(v(&keys[1].0, 7), "7");
    assert_eq!(v(&keys[5].0, -3), "-3");
    let d = crate::values_source::double_to_sortable_long(-2.0);
    assert_eq!(v(&keys[7].0, d), "-2.0");
    let f = i64::from(lucene_util::numeric_utils::float_to_sortable_int(0.25));
    assert_eq!(v(&keys[6].0, f), "0.25");
    assert_eq!(
        sort_value_string(&keys[2].0, 0, Some(Some(b"a\x0f"))),
        "[61 f]"
    );
    assert_eq!(sort_value_string(&keys[2].0, 0, Some(None)), "null");

    // A values-source key: `<source>`, its value a `Double` or `Long`.
    let double = vs::double_sort_field(vs::constant(1.0), true, 0.0);
    assert_eq!(sort_field_string(&double), "<constant(1.0)>!");
    assert_eq!(v(&double, vs::double_to_sortable_long(2.5)), "2.5");
    let long = vs::long_sort_field(vs::long_constant(4), false, 0);
    assert_eq!(sort_field_string(&long), "<constant(4)>");
    assert_eq!(v(&long, 9), "9");

    // A source without its own `toString`: the defaults; an unknown
    // registration prints the field alone.
    struct Plain(Arc<dyn FieldComparatorSource>);
    impl FieldComparatorSource for Plain {
        fn new_comparator(
            &self,
            field: &str,
            num_hits: usize,
            reverse: bool,
        ) -> Box<dyn FieldComparator> {
            self.0.new_comparator(field, num_hits, reverse)
        }
    }
    let SortType::Custom(id) = double.ty else {
        unreachable!()
    };
    let plain = Plain(registered_source(id).unwrap());
    let _ = plain.new_comparator("x", 1, false);
    let held = SortField::with_source("x", Arc::new(plain), true);
    assert_eq!(sort_field_string(&held), "<custom:\"x\">!");
    assert_eq!(v(&held, 5), "5");
    let unknown = SortField::custom("u", CustomSortId(u32::MAX - 1), false);
    assert_eq!(sort_field_string(&unknown), "<custom:\"u\">");
    assert_eq!(v(&unknown, 6), "6");
}

#[test]
fn modes_pick_as_multi_value_mode_does() {
    use SortType::{Double, Float, Long};
    let p = |sel: Selector, v: &[i64]| sel.pick(Long, v).unwrap();
    assert_eq!(p(Selector::Min, &[-3, 1, 9]), -3);
    assert_eq!(p(Selector::Max, &[-3, 1, 9]), 9);
    assert_eq!(p(Selector::Sum, &[-3, 1, 9]), 7);
    assert_eq!(
        p(Selector::Sum, &[i64::MAX, 1]),
        i64::MIN,
        "a long sum wraps"
    );
    // AVG: Math.round of the double quotient, ties toward +infinity.
    assert_eq!(p(Selector::Avg, &[1, 2]), 2);
    assert_eq!(p(Selector::Avg, &[-2, -1]), -1);
    assert_eq!(p(Selector::Avg, &[5]), 5);
    assert_eq!(p(Selector::Avg, &[1, 1, 2]), 1);
    // MEDIAN: the middle value, or the rounded mean of the middle two.
    assert_eq!(p(Selector::Median, &[1, 2, 3]), 2);
    assert_eq!(p(Selector::Median, &[1, 2, 3, 4]), 3);
    assert_eq!(p(Selector::Median, &[-3, -2]), -2);
    assert_eq!(p(Selector::Median, &[7]), 7);
    assert_eq!(Selector::Sum.pick(Long, &[]), None);
    // Doubles: over the decoded values, re-encoded.
    let d = double_to_sortable_long;
    let dv = |sel: Selector, v: &[f64]| {
        let enc: Vec<i64> = v.iter().map(|&x| d(x)).collect();
        sortable_long_to_double(sel.pick(Double, &enc).unwrap())
    };
    assert_eq!(dv(Selector::Sum, &[0.1, 0.2]), 0.1 + 0.2);
    assert_eq!(dv(Selector::Avg, &[1.0, 2.0, 4.0]), 7.0 / 3.0);
    assert_eq!(dv(Selector::Median, &[1.0, 2.0, 3.0, 10.0]), 2.5);
    assert_eq!(dv(Selector::Median, &[-1.5]), -1.5);
    // Floats: summed as doubles, the result cast back to float.
    let f = |x: f32| i64::from(float_to_sortable_int(x));
    let got = Selector::Sum.pick(Float, &[f(0.1), f(0.2)]).unwrap();
    assert_eq!(got, f((f64::from(0.1f32) + f64::from(0.2f32)) as f32));
    // Math.round's edges.
    assert_eq!(java_round(0.499_999_999_999_999_94), 0);
    assert_eq!(java_round(-0.5), 0);
    assert_eq!(java_round(2.5), 3);
    assert_eq!(java_round(f64::NAN), 0);
    assert_eq!(java_round(1e300), i64::MAX);
    assert_eq!(java_round(-1e300), i64::MIN);
}
use crate::directory_reader::DirectoryReader;
use lucene_store::FsDirectory;

fn fixture(name: &str) -> DirectoryReader {
    let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
    DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
}

fn all() -> BooleanQuery {
    let mut q = BooleanQuery::new();
    q.must
        .push(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
            0,
        )));
    q
}

fn run(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &[SortField],
    top_n: usize,
    after: Option<&FieldDoc>,
) -> Result<TopFieldDocs> {
    run_t(reader, q, sort, top_n, u64::MAX, after)
}

fn run_t(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &[SortField],
    top_n: usize,
    threshold: u64,
    after: Option<&FieldDoc>,
) -> Result<TopFieldDocs> {
    let mut opened = reader.open_segments().expect("open");
    opened.open_points().expect("points");
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    search_sorted(
        &segments,
        reader.segment_readers(),
        q,
        &norms,
        sort,
        top_n,
        threshold,
        after,
    )
}

/// A doc-values skip index lets a numeric key (`DVSkipperCompetitiveDISIBuilder`)
/// and a keyword key (`SkipperBasedCompetitiveState`) skip whole blocks
/// once the queue is full: the same hits as with skipping turned off,
/// over fewer documents (the total, a lower bound past the threshold,
/// counts every document visited).
#[test]
fn a_skip_index_skips_documents_without_changing_the_hits() {
    let numeric = fixture("sorted_search_index");
    let keyword = fixture("keyword_sort_index");
    let cases: Vec<(&DirectoryReader, Vec<SortField>)> = vec![
        (
            &numeric,
            vec![SortField::numeric("k", SortType::Long, false)],
        ),
        (
            &numeric,
            vec![SortField::numeric("k", SortType::Long, true)],
        ),
        (
            &numeric,
            vec![
                SortField::numeric("k", SortType::Long, false),
                SortField::score(),
            ],
        ),
        (&keyword, vec![keyword_last("ks", false)]),
        (&keyword, vec![keyword_last("ks", true)]),
    ];
    for (reader, sort) in cases {
        // The comparators' own skipping, not the bounded newest-N search's.
        let with = |disabled: bool| {
            DISABLE_SKIPPER.with(|d| d.set(disabled));
            newest_n::tests::DISABLED.with(|d| d.set(true));
            let r = run_t(reader, &all(), &sort, 10, 100, None).unwrap();
            newest_n::tests::DISABLED.with(|d| d.set(false));
            DISABLE_SKIPPER.with(|d| d.set(false));
            r
        };
        let skipping = with(false);
        let scanning = with(true);
        assert_eq!(skipping.hits, scanning.hits, "{sort:?}");
        assert!(
            skipping.total.value * 2 < scanning.total.value,
            "{sort:?}: visited {} with the skip index, {} without",
            skipping.total.value,
            scanning.total.value
        );
    }
}

fn keyword_last(field: &str, reverse: bool) -> SortField {
    SortField {
        missing: 1,
        ..SortField::string(field, reverse)
    }
}

/// `DocComparator`'s competitive iterator: nothing before the hits
/// threshold; then empty once the queue is full, or -- paging -- the
/// documents from the `after` document on, a segment before it skipped.
#[test]
fn the_doc_iterator_follows_the_threshold_the_bottom_and_the_top() {
    let mut d = DocCompetitive::new(100, 50, None);
    d.update(false, true);
    assert!(matches!(d.iter, Iter::All { .. }));
    d.update(true, true);
    assert_eq!(d.iter.advance(0).unwrap(), NO_MORE_DOCS);
    // Paging after global doc 120: this segment (100..150) from 20 on.
    let mut d = DocCompetitive::new(100, 50, Some(120));
    d.update(true, false);
    assert_eq!(d.iter.advance(0).unwrap(), 20);
    assert_eq!(d.iter.cost(), 30);
    assert_eq!(d.iter.advance(49).unwrap(), 49);
    assert_eq!(d.iter.advance(50).unwrap(), NO_MORE_DOCS);
    // A segment wholly before the top.
    let mut d = DocCompetitive::new(0, 50, Some(120));
    d.update(true, false);
    assert_eq!(d.iter.advance(0).unwrap(), NO_MORE_DOCS);
}

/// Every document of `field` in `reader`'s segments: `(segment, doc,
/// values)`, and the values' doc-values encoding.
fn column(reader: &DirectoryReader, field: &str) -> Vec<Vec<Vec<i64>>> {
    let mut out = Vec::new();
    for r in reader.segment_readers() {
        let info = r
            .field_infos()
            .fields
            .iter()
            .find(|i| i.name == field)
            .unwrap();
        let (meta, data) = r.doc_values_for_field(info.number).unwrap();
        let mut seg = Vec::new();
        let e = meta.sorted_numeric_entry(info.number);
        let n = meta.numeric_entry(info.number);
        for doc in 0..r.max_doc {
            let mut v = Vec::new();
            if let Some(e) = e {
                SortedNumericReader::new(data, e)
                    .values(doc, &mut v)
                    .unwrap();
            } else if let Some(n) = n {
                v.extend(NumericReader::new(data, n).value(doc).unwrap());
            }
            seg.push(v);
        }
        out.push(seg);
    }
    out
}

#[test]
fn a_nested_key_picks_from_the_children_before_each_root() {
    // Roots every seventh document; children the documents a points
    // range matches (deletions ignored, as Lucene's inner weight does).
    let reader = fixture("sorted_search_index");
    let mut opened = reader.open_segments().expect("open");
    opened.open_points().expect("points");
    let segments = opened.as_open_segments();
    let parents: Vec<Vec<i32>> = reader
        .segment_readers()
        .iter()
        .map(|r| (0..r.max_doc).filter(|d| d % 7 == 6).collect())
        .collect();
    let children = BooleanQuery {
        must: vec![Clause::PointsRange(crate::query::PointsRangeQuery::new(
            "r", 0, 2_000_000,
        ))],
        ..Default::default()
    };
    for (field, ty) in [
        ("m", SortType::Long),
        ("l", SortType::Long),
        ("d", SortType::Double),
        ("f", SortType::Float),
        ("i", SortType::Int),
    ] {
        let values = column(&reader, field);
        for selector in [Selector::Min, Selector::Max, Selector::Sum, Selector::Avg] {
            for max_children in [i32::MAX, 2] {
                for (s, seg) in segments.iter().enumerate() {
                    let child_docs = nested_docs(seg, &children).unwrap();
                    let (column, _) = {
                        let r = &reader.segment_readers()[s];
                        let info = r
                            .field_infos()
                            .fields
                            .iter()
                            .find(|i| i.name == field)
                            .unwrap();
                        let (meta, data) = r.doc_values_for_field(info.number).unwrap();
                        match meta.sorted_numeric_entry(info.number) {
                            Some(e) => (ChildValues::Multi(SortedNumericReader::new(data, e)), ()),
                            None => (
                                ChildValues::Single(NumericReader::new(
                                    data,
                                    meta.numeric_entry(info.number).unwrap(),
                                )),
                                (),
                            ),
                        }
                    };
                    let mut col = NestedColumn {
                        values: column,
                        parents: parents[s].clone(),
                        children: child_docs.clone(),
                        pos: None,
                        selector,
                        ty,
                        missing: -77,
                        max_children,
                        convert: ChildConvert::None,
                        last_parent: -1,
                        last_value: -77,
                        buf: Vec::new(),
                    };
                    // Doc ids ascend, so the oracle walks the children
                    // with one cursor rather than rescanning all of them
                    // per root (that rescan made this the suite's
                    // second-slowest test, ~1 minute).
                    assert!(child_docs.windows(2).all(|w| w[0] < w[1]));
                    let mut next_child = 0;
                    let mut prev = -1;
                    for &p in &parents[s] {
                        // The oracle: the children strictly between the
                        // previous root and this one, in order.
                        let from = next_child;
                        while next_child < child_docs.len() && child_docs[next_child] < p {
                            next_child += 1;
                        }
                        let kids: Vec<&Vec<i64>> = child_docs[from..next_child]
                            .iter()
                            .filter(|&&c| c > prev && c < p)
                            .map(|&c| &values[s][c as usize])
                            .filter(|v| !v.is_empty())
                            .take(usize::try_from(max_children).unwrap())
                            .collect();
                        prev = p;
                        let floating = matches!(ty, SortType::Double | SortType::Float);
                        let d = |v: i64| {
                            if ty == SortType::Float {
                                f64::from(sortable_int_to_float(v))
                            } else {
                                sortable_long_to_double(v)
                            }
                        };
                        let enc = |r: f64| {
                            if ty == SortType::Float {
                                i64::from(float_to_sortable_int(r as f32))
                            } else {
                                double_to_sortable_long(r)
                            }
                        };
                        let all: Vec<i64> = kids.iter().flat_map(|v| v.iter().copied()).collect();
                        let want = if kids.is_empty() {
                            -77
                        } else {
                            match (selector, floating) {
                                (Selector::Min, false) => kids.iter().map(|v| v[0]).min().unwrap(),
                                (Selector::Max, false) => {
                                    kids.iter().map(|v| *v.last().unwrap()).max().unwrap()
                                }
                                (Selector::Sum, false) => {
                                    all.iter().fold(0i64, |a, &b| a.wrapping_add(b))
                                }
                                (Selector::Avg, false) => {
                                    let t = all.iter().fold(0i64, |a, &b| a.wrapping_add(b));
                                    if all.len() > 1 {
                                        java_round(t as f64 / all.len() as f64)
                                    } else {
                                        t
                                    }
                                }
                                (Selector::Min, true) => {
                                    enc(kids.iter().fold(f64::INFINITY, |a, v| {
                                        crate::java_math::java_min(a, d(v[0]))
                                    }))
                                }
                                (Selector::Max, true) => {
                                    enc(kids.iter().fold(f64::NEG_INFINITY, |a, v| {
                                        crate::java_math::java_max(a, d(*v.last().unwrap()))
                                    }))
                                }
                                (Selector::Sum, true) => {
                                    enc(all.iter().fold(0.0, |a, &b| a + d(b)))
                                }
                                _ => enc(all.iter().fold(0.0, |a, &b| a + d(b)) / all.len() as f64),
                            }
                        };
                        let got = col.value(p).unwrap();
                        if p == 0 {
                            assert_eq!(got, -77);
                        } else {
                            assert_eq!(
                                got, want,
                                "{field} {selector:?} max {max_children} seg {s} root {p}"
                            );
                        }
                        // Asked again: the same, without reading.
                        assert_eq!(col.value(p).unwrap(), got);
                    }
                }
            }
        }
    }
    // Through the collector: a nested key sorts the roots by their picks.
    let nested = Arc::new(NestedSort {
        convert: ChildConvert::None,
        parents: BooleanQuery {
            must: vec![Clause::PointsRange(crate::query::PointsRangeQuery::new(
                "r", 0, 2_000_000,
            ))],
            ..Default::default()
        },
        children: children.clone(),
        max_children: i32::MAX,
    });
    let key = SortField {
        nested: Some(nested.clone()),
        rewritten: None,
        convert: ChildConvert::None,
        uninverted: false,
        ..SortField::numeric("l", SortType::Long, false)
    };
    assert_eq!(key, key.clone());
    assert_ne!(key, SortField::numeric("l", SortType::Long, false));
    let top = run(&reader, &all(), &[key, SortField::doc()], 5, None).unwrap();
    assert_eq!(top.hits.len(), 5);
    assert!(top
        .hits
        .windows(2)
        .all(|w| w[0].values[0] <= w[1].values[0]));
}

fn keyword(field: &str, reverse: bool, missing_last: bool) -> SortField {
    SortField {
        missing: i64::from(missing_last),
        ..SortField::string(field, reverse)
    }
}

#[test]
fn skipping_by_term_never_changes_the_keyword_hits() {
    // Every competitive-state form: postings (sparse `k`, multi-valued
    // `km`, dense `kd`), an indexed field without doc values (`id`: no
    // terms to compete, so nothing does), a field no segment has
    // (`EmptyCompetitiveState`), and the missing-first/last cases that
    // keep everything.
    let reader = fixture("keyword_sort_index");
    let mut sorts = Vec::new();
    for field in ["k", "km", "kd", "id", "nosuch"] {
        for reverse in [false, true] {
            for last in [false, true] {
                sorts.push(vec![keyword(field, reverse, last)]);
                sorts.push(vec![keyword(field, reverse, last), SortField::doc()]);
            }
        }
    }
    sorts.push(vec![SortField {
        selector: Selector::Max,
        ..keyword("km", true, true)
    }]);
    let mut checked = 0;
    for sort in &sorts {
        for top_n in [1, 7] {
            let exact = run(&reader, &all(), sort, top_n, None).unwrap();
            let pruned = run_t(&reader, &all(), sort, top_n, 0, None).unwrap();
            assert_eq!(pruned.hits, exact.hits, "{sort:?} top {top_n}");
            let Some(last) = exact.hits.last() else {
                continue;
            };
            let exact2 = run(&reader, &all(), sort, top_n, Some(last)).unwrap();
            let pruned2 = run_t(&reader, &all(), sort, top_n, 0, Some(last)).unwrap();
            assert_eq!(pruned2.hits, exact2.hits, "{sort:?} top {top_n} after");
            checked += 1;
        }
    }
    assert_eq!(checked, sorts.len() * 2);
    // A field no segment has sorts every document as missing: the first
    // documents, in order, and a search-after term looked up nowhere.
    let absent = [keyword("nosuch", false, true)];
    let got = run_t(&reader, &all(), &absent, 3, 0, None).unwrap();
    let docs: Vec<i32> = got.hits.iter().map(|h| h.doc).collect();
    assert!(
        docs.len() == 3 && docs.windows(2).all(|w| w[0] < w[1]),
        "{docs:?}"
    );
    assert!(got.hits.iter().all(|h| h.terms == [None]));
    assert_eq!(got.total.relation, TotalHitsRelation::GreaterThanOrEqualTo);
    let after = FieldDoc {
        doc: 5,
        values: vec![0],
        terms: vec![Some(b"m".to_vec())],
    };
    // Missing sorts last, so every document is after the term.
    let got = run_t(&reader, &all(), &absent, 2, 0, Some(&after)).unwrap();
    assert_eq!(
        got.hits.iter().map(|h| h.doc).collect::<Vec<_>>(),
        docs[..2]
    );
}

#[test]
fn a_keyword_key_needs_a_sorted_column_and_a_numeric_key_a_numeric_one() {
    let reader = fixture("keyword_sort_index");
    // A search-after page by a keyword key carries its term slot.
    let after = FieldDoc {
        doc: 1,
        values: vec![0],
        terms: Vec::new(),
    };
    let err = run(
        &reader,
        &all(),
        &[keyword("k", false, true)],
        3,
        Some(&after),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            crate::Error::Sort(SortError::AfterArity { got: 0, want: 1 })
        ),
        "{err}"
    );
    let err = run(&reader, &all(), &[keyword("i", false, true)], 3, None).unwrap_err();
    assert!(
        matches!(err, crate::Error::Sort(SortError::KeywordType(ref f)) if f == "i"),
        "{err}"
    );
    let err = run(
        &reader,
        &all(),
        &[SortField::numeric("k", SortType::Long, false)],
        3,
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, crate::Error::Sort(SortError::DocValuesType(ref f)) if f == "k"),
        "{err}"
    );
}

#[test]
fn a_sort_needs_keys_and_an_after_of_its_arity() {
    let r = fixture("sorted_search_index");
    assert!(matches!(
        run(&r, &all(), &[], 10, None),
        Err(crate::Error::Sort(SortError::NoKeys))
    ));
    let after = FieldDoc {
        doc: 3,
        values: vec![1, 2],
        terms: Vec::new(),
    };
    assert!(matches!(
        run(&r, &all(), &[SortField::doc()], 10, Some(&after)),
        Err(crate::Error::Sort(SortError::AfterArity {
            got: 2,
            want: 1
        }))
    ));
    let none = run(&r, &all(), &[SortField::doc()], 0, None).unwrap();
    assert!(none.hits.is_empty());
    assert_eq!(none.total.value, 0);
}

#[test]
fn points_of_another_width_are_refused_as_lucene_refuses_them() {
    // `i` is an `IntPoint`; sorting it as a long asks for 8-byte points.
    let r = fixture("sorted_search_index");
    let e = run(
        &r,
        &all(),
        &[SortField::numeric("i", SortType::Long, false)],
        10,
        None,
    );
    assert!(matches!(
        e,
        Err(crate::Error::Sort(SortError::PointsShape { want: 8, .. }))
    ));
}

#[test]
fn a_non_numeric_column_cannot_be_sorted_numerically() {
    let r = fixture("sorted_dv_index");
    let e = run(
        &r,
        &all(),
        &[SortField::numeric("sorted", SortType::Long, false)],
        3,
        None,
    );
    assert!(matches!(
        e,
        Err(crate::Error::Sort(SortError::DocValuesType(f))) if f == "sorted"
    ));
    // A field with no doc values at all sorts every document as missing.
    let ok = run(
        &r,
        &all(),
        &[SortField::numeric("nope", SortType::Long, true)],
        3,
        None,
    )
    .unwrap();
    assert_eq!(
        ok.hits
            .iter()
            .map(|h| (h.doc, h.values[0]))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 0), (2, 0)]
    );
}

#[test]
fn values_order_as_lucene_compares_them() {
    // Float.compare: -0.0 before 0.0, NaN after everything.
    let order = [
        f32::NEG_INFINITY,
        -1.5,
        -0.0,
        0.0,
        1e-30,
        2.0,
        f32::INFINITY,
        f32::NAN,
    ];
    for w in order.windows(2) {
        assert!(
            float_to_sortable_int(w[0]) < float_to_sortable_int(w[1]),
            "{w:?}"
        );
        // Relevance runs the other way.
        assert!(score_value(w[0]) > score_value(w[1]), "{w:?}");
    }
    for f in [0.25f32, -3.0, 0.0, -0.0, 7.5e10] {
        assert_eq!(
            sortable_int_to_float(-score_value(f)).to_bits(),
            f.to_bits()
        );
    }
    assert_eq!(
        sortable_bytes_to_long(&crate::points_query::pack_i64(-42)),
        -42
    );
    assert_eq!(sortable_bytes_to_long(&[0x80, 0, 0, 5]), 5);
    assert_eq!(sortable_bytes_to_long(&[0x7f, 0xff, 0xff, 0xff]), -1);
    // A half-float point: two bytes, the sign bit flipped.
    assert_eq!(sortable_bytes_to_long(&[0x80, 5]), 5);
    assert_eq!(sortable_bytes_to_long(&[0x7f, 0xff]), -1);
    assert_eq!(
        sortable_bytes_to_long(&[1, 2, 3]),
        0,
        "no other width is a sortable number"
    );
}

#[test]
fn every_competitive_iterator_form_advances_the_same_way() {
    let docs = vec![3, 9, 20];
    let mut bits = FixedBitSet::new(32);
    for &d in &docs {
        bits.set(d as usize);
    }
    let mut forms = [
        Iter::Docs {
            docs: docs.clone(),
            next: 0,
            doc: -1,
        },
        Iter::Bits {
            cost: bits.cardinality() as i64,
            bits,
            doc: -1,
        },
    ];
    for it in &mut forms {
        assert_eq!(it.doc_id(), -1);
        assert_eq!(it.advance(0).unwrap(), 3);
        assert_eq!(it.advance(4).unwrap(), 9);
        assert_eq!(it.advance(9).unwrap(), 9);
        assert_eq!(it.advance(21).unwrap(), NO_MORE_DOCS);
    }
    let mut all = Iter::All {
        max_doc: 5,
        doc: -1,
    };
    assert_eq!(all.advance(2).unwrap(), 2);
    assert_eq!(all.advance(5).unwrap(), NO_MORE_DOCS);
}

#[test]
fn the_skip_interval_widens_on_failures_and_narrows_on_success() {
    let points = PointsReader::empty();
    let mut c = Competitive {
        points: &points,
        skipper: None,
        field_number: 0,
        bytes: 8,
        convert: ChildConvert::None,
        point_doc_count: 0,
        max_doc: 10,
        leaf_top_set: false,
        iter: Iter::All {
            max_doc: 10,
            doc: -1,
        },
        min_value: i64::MIN,
        max_value: i64::MAX,
        max_doc_visited: -1,
        update_counter: 0,
        current_skip_interval: MIN_SKIP_INTERVAL,
        iterator_cost: 10,
        try_update_fail_count: 0,
        with_value: WithValue::None,
        scratch: Vec::new(),
        spare_bits: None,
        walk: PointsScratch::default(),
        missing_docs: None,
        missing_range: None,
    };
    // Below 257 updates the interval never moves.
    c.update_skip_interval(false);
    assert_eq!(c.current_skip_interval, MIN_SKIP_INTERVAL);
    c.update_counter = 300;
    for _ in 0..4 {
        c.update_skip_interval(false);
    }
    assert_eq!(
        c.current_skip_interval,
        2 * MIN_SKIP_INTERVAL,
        "the fourth failure doubles it"
    );
    c.current_skip_interval = MAX_SKIP_INTERVAL;
    for _ in 0..4 {
        c.update_skip_interval(false);
    }
    assert_eq!(c.current_skip_interval, MAX_SKIP_INTERVAL, "capped");
    c.update_skip_interval(true);
    assert_eq!(c.current_skip_interval, MAX_SKIP_INTERVAL / 2);
    c.current_skip_interval = MIN_SKIP_INTERVAL;
    c.update_skip_interval(true);
    assert_eq!(c.current_skip_interval, MIN_SKIP_INTERVAL, "floored");
}

#[test]
fn the_visitor_moves_to_a_bit_set_past_the_upgrade_size() {
    let mut v = CompetitiveVisitor {
        convert: ChildConvert::None,
        min: 10,
        max: 20,
        max_doc_visited: 2,
        docs: Vec::new(),
        bits: None,
        upgrade_at: 3,
        added: 0,
        spare: None,
        max_doc: 64,
        corrupt: None,
    };
    v.visit(1); // already visited: dropped
    v.visit(9);
    v.visit_with_value(5, &crate::points_query::pack_i64(30)); // out of range
    v.visit_with_value(7, &crate::points_query::pack_i64(15));
    assert!(v.bits.is_none());
    assert_eq!(v.docs, [9, 7]);
    v.visit(40); // the third id upgrades
    v.visit(41);
    let bits = v.bits.as_ref().expect("upgraded");
    assert!(v.docs.is_empty());
    assert_eq!(v.added, 4);
    for d in [7, 9, 40, 41] {
        assert!(bits.get(d), "{d}");
    }
    assert_eq!(bits.cardinality(), 4);

    // In bulk: below the floor dropped, the list kept until a run would
    // pass the upgrade size, then set directly.
    let mut v = CompetitiveVisitor {
        convert: ChildConvert::None,
        min: 0,
        max: 0,
        max_doc_visited: 4,
        docs: Vec::new(),
        bits: None,
        upgrade_at: 4,
        added: 0,
        spare: Some(FixedBitSet::new(64)),
        max_doc: 64,
        corrupt: None,
    };
    v.visit_many(&[3, 5, 6]);
    assert_eq!((v.docs.as_slice(), v.added), (&[5, 6][..], 2));
    v.visit_many(&[2, 8, 9]);
    let bits = v.bits.as_ref().expect("upgraded");
    assert!(v.spare.is_none(), "the spare set was used");
    assert_eq!(v.added, 4);
    assert_eq!(bits.cardinality(), 4);

    // A leaf naming a document outside the segment is remembered, not
    // indexed: singly or in bulk.
    v.visit(64);
    assert_eq!(v.corrupt, Some(64));
    let mut w = CompetitiveVisitor {
        convert: ChildConvert::None,
        min: 0,
        max: 0,
        max_doc_visited: -1,
        docs: Vec::new(),
        bits: None,
        upgrade_at: 100,
        added: 0,
        spare: None,
        max_doc: 8,
        corrupt: None,
    };
    // The visit stops at the bad id (what came before it may be added:
    // a recorded `corrupt` fails the whole walk, so it is never used).
    w.visit_many(&[1, -3, 2]);
    assert_eq!((w.corrupt, w.added), (Some(-3), 1));
    assert_eq!(w.docs, [1]);
}

fn comparator(reverse: bool, pruning: Pruning) -> Comparator {
    let mut f = SortField::numeric("l", SortType::Long, reverse);
    f.missing = 7;
    Comparator {
        field: f,
        mul: if reverse { -1 } else { 1 },
        values: vec![0; 4],
        bottom: 10,
        top: 3,
        top_set: false,
        pruning,
        single_sort: true,
        hits_threshold_reached: false,
        queue_full: false,
        strs: None,
        vals: None,
        custom: None,
    }
}

fn competitive<'a>(
    points: &'a PointsReader<'a>,
    field_number: i32,
    max_doc: i32,
) -> Competitive<'a> {
    Competitive {
        points,
        skipper: None,
        field_number,
        bytes: 8,
        convert: ChildConvert::None,
        point_doc_count: max_doc,
        max_doc,
        leaf_top_set: false,
        iter: Iter::All { max_doc, doc: -1 },
        min_value: i64::MIN,
        max_value: i64::MAX,
        max_doc_visited: -1,
        update_counter: 0,
        current_skip_interval: MIN_SKIP_INTERVAL,
        iterator_cost: i64::from(max_doc),
        try_update_fail_count: 0,
        with_value: WithValue::None,
        scratch: Vec::new(),
        spare_bits: None,
        walk: PointsScratch::default(),
        missing_docs: None,
        missing_range: None,
    }
}

#[test]
fn a_finished_segment_leaves_its_bit_sets_cleared_for_the_next_query() {
    // Test threads are shared, so each test keys on a length of its own.
    const LEN: usize = 1_003;
    let points = PointsReader::empty();
    let mut comp = competitive(&points, 0, LEN as i32);
    let mut used = FixedBitSet::new(LEN);
    used.set(7);
    used.set(LEN - 1);
    comp.iter = Iter::Bits {
        bits: used,
        doc: -1,
        cost: 2,
    };
    comp.spare_bits = Some(FixedBitSet::new(LEN));
    assert_eq!(crate::bit_set_pool::held(LEN), 0);
    drop(comp);
    // Both the live iterator's set and the spare come back.
    assert_eq!(crate::bit_set_pool::held(LEN), 2);
    for _ in 0..2 {
        let b = crate::bit_set_pool::take(LEN).expect("a pooled set");
        assert_eq!(b.len(), LEN);
        assert_eq!(b.cardinality(), 0, "handed back cleared");
    }
    assert!(crate::bit_set_pool::take(LEN).is_none());
    // A set is only reused for a segment of exactly its length.
    crate::bit_set_pool::give(FixedBitSet::new(LEN));
    assert!(crate::bit_set_pool::take(LEN + 1).is_none());
    assert!(crate::bit_set_pool::take(LEN).is_some());
    // An iterator that is not a bit set leaves nothing behind.
    drop(competitive(&points, 0, LEN as i32));
    assert_eq!(crate::bit_set_pool::held(LEN), 0);
}

#[test]
fn the_spare_bit_sets_stay_under_their_limit() {
    let limit_bits = crate::bit_set_pool::SPARE_BIT_SET_LIMIT_BYTES * 8;
    // Larger than the whole limit: dropped, not pinned.
    crate::bit_set_pool::give(FixedBitSet::new(limit_bits + 64));
    assert_eq!(crate::bit_set_pool::held(limit_bits + 64), 0);
    // Three sets of a third of the limit (plus a word) fit only twice.
    let third = limit_bits / 3 + 64;
    for _ in 0..3 {
        crate::bit_set_pool::give(FixedBitSet::new(third));
    }
    assert_eq!(crate::bit_set_pool::held(third), 2);
    while crate::bit_set_pool::take(third).is_some() {}
}

#[test]
fn bounds_follow_the_bottom_the_top_and_the_pruning() {
    let points = PointsReader::empty();
    // Ascending, `>=` pruning: the bottom itself cannot compete.
    let mut c = comparator(false, Pruning::GreaterThanOrEqualTo);
    let mut comp = competitive(&points, 0, 100);
    comp.encode_bottom(&c);
    assert_eq!((comp.min_value, comp.max_value), (i64::MIN, 9));
    // Descending with `>` pruning: the bottom still can (a later key).
    let d = comparator(true, Pruning::GreaterThan);
    comp.encode_bottom(&d);
    assert_eq!(comp.min_value, 10);
    // encodeTop tightens only for a single sort with a full queue.
    c.queue_full = true;
    comp.encode_top(&c);
    assert_eq!(comp.min_value, 4);
    let mut d = comparator(true, Pruning::GreaterThanOrEqualTo);
    d.queue_full = true;
    comp.encode_top(&d);
    assert_eq!(comp.max_value, 2);
    d.single_sort = false;
    comp.encode_top(&d);
    assert_eq!(comp.max_value, 3);

    // isMissingValueCompetitive: missing 7 against bottom 10 and top 3.
    let mut asc = comparator(false, Pruning::GreaterThanOrEqualTo);
    asc.queue_full = true;
    assert!(comp.missing_competitive(&asc), "7 < 10 ascending");
    asc.bottom = 7;
    assert!(!comp.missing_competitive(&asc), "a tie loses under >=");
    asc.pruning = Pruning::GreaterThan;
    assert!(comp.missing_competitive(&asc), "a tie can win under >");
    let mut desc = comparator(true, Pruning::GreaterThanOrEqualTo);
    desc.queue_full = true;
    assert!(!comp.missing_competitive(&desc), "7 < 10 descending");
    desc.pruning = Pruning::GreaterThan;
    desc.bottom = 7;
    assert!(comp.missing_competitive(&desc));
    // With a top value: ascending needs missing >= top, descending <= top.
    comp.leaf_top_set = true;
    asc.bottom = 20;
    assert!(comp.missing_competitive(&asc));
    desc.bottom = 1;
    assert!(!comp.missing_competitive(&desc));
    asc.queue_full = false;
    asc.top = 8;
    assert!(!comp.missing_competitive(&asc));
}

#[test]
fn an_update_waits_for_the_threshold_and_a_full_queue_then_falls_back_to_values() {
    let reader = fixture("sorted_search_index");
    let mut opened = reader.open_segments().expect("open");
    opened.open_points().expect("points");
    let segments = opened.as_open_segments();
    let p = segments[0].points.expect("points opened");
    let num = p.field_number("l").expect("l has points");
    let max_doc = reader.segment_readers()[0].max_doc;
    let mut c = comparator(false, Pruning::GreaterThanOrEqualTo);
    c.bottom = i64::MAX - 1;
    let mut comp = competitive(&p.reader, num, max_doc);
    comp.point_doc_count = max_doc; // no missing documents
    comp.update(&c).unwrap();
    assert_eq!(comp.update_counter, 0, "before the threshold, nothing");
    c.hits_threshold_reached = true;
    comp.update(&c).unwrap();
    assert_eq!(comp.update_counter, 0, "no top and no full queue, nothing");
    c.queue_full = true;
    // A range this wide cannot narrow 8-fold: with fewer points than the
    // iterator costs, the documents with values drive it instead.
    comp.point_doc_count = max_doc - 1;
    comp.iterator_cost = i64::from(max_doc) + 1;
    c.field.missing = i64::MAX; // missing sorts last: not competitive
    comp.with_value = WithValue::All;
    comp.update(&c).unwrap();
    assert!(matches!(comp.iter, Iter::All { .. }));
    assert_eq!(comp.iterator_cost, i64::from(max_doc - 1));
    comp.iterator_cost = i64::from(max_doc) + 1;
    comp.with_value = WithValue::None;
    comp.update(&c).unwrap();
    assert!(matches!(&comp.iter, Iter::Docs { docs, .. } if docs.is_empty()));
    // A missing value that could still compete stops Lucene's update;
    // here it goes to the missing-value path, sampled like any update.
    c.field.missing = i64::MIN;
    let before = comp.update_counter;
    comp.update(&c).unwrap();
    assert_eq!(comp.update_counter, before + 1);
    // The same range again is not rebuilt, and not counted.
    comp.update(&c).unwrap();
    assert_eq!(comp.update_counter, before + 1);
}

#[test]
fn docs_with_values_come_from_the_entry() {
    let reader = fixture("sorted_search_index");
    let r = &reader.segment_readers()[0];
    let info = r
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "l")
        .unwrap();
    let (meta, data) = r.doc_values_for_field(info.number).unwrap();
    let entry = meta
        .sorted_numeric_entry(info.number)
        .unwrap()
        .numeric
        .clone();
    assert!(
        matches!(WithValue::of(data, &entry), WithValue::Disi { .. }),
        "l is sparse"
    );
    let mut e = entry.clone();
    e.docs_with_field_offset = -1;
    assert!(matches!(WithValue::of(data, &e), WithValue::All));
    e.docs_with_field_offset = -2;
    assert!(matches!(WithValue::of(data, &e), WithValue::None));
    e.docs_with_field_offset = data.len() as i64;
    e.docs_with_field_length = 10;
    assert!(
        matches!(WithValue::of(data, &e), WithValue::None),
        "past the file"
    );
}

#[test]
fn the_scoring_tree_passes_through_and_reports_a_document_it_lacks() {
    let reader = fixture("sorted_search_index");
    let opened = reader.open_segments().expect("open");
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    let ctx = exec::LeafContext {
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
        reader: None,
        similarity: seg.similarity,
    };
    let all = Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
    let tree = exec::build::child(&ctx, &all, 1.0, Mode::Complete, true)
        .unwrap()
        .unwrap()
        .into_scorer(Mode::Complete);
    let mut s = ScoreAt {
        inner: tree,
        doc: 5,
    };
    assert_eq!(s.cost(), i64::from(seg.max_doc.unwrap()));
    assert_eq!(s.next_doc().unwrap(), 0);
    assert_eq!(s.doc_id(), 0);
    assert_eq!(s.score().unwrap(), 1.0, "moved to document 5 first");
    assert_eq!(s.doc_id(), 5);
    assert_eq!(s.advance(7).unwrap(), 7);
    assert!(s.max_score(10).unwrap() >= 1.0);
    s.doc = seg.max_doc.unwrap() + 3;
    assert!(matches!(
        s.score(),
        Err(crate::Error::Sort(SortError::ScoringTree(_)))
    ));
}

#[test]
fn counting_the_rest_stops_just_past_the_threshold() {
    let reader = fixture("sorted_search_index");
    let opened = reader.open_segments().expect("open");
    let segments = opened.as_open_segments();
    let seg = &segments[1];
    let ctx = exec::LeafContext {
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
        reader: None,
        similarity: seg.similarity,
    };
    let all = Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
    let scorer = || {
        exec::build::child(&ctx, &all, 1.0, Mode::NoScores, true)
            .unwrap()
            .unwrap()
            .into_scorer(Mode::NoScores)
    };
    let leaf = || Leaf {
        doc_base: 0,
        keys: vec![LeafKey::Doc],
        collected_all_competitive: true,
        after_doc: 0,
        terminated: false,
        early: false,
        score: 0.0,
        score_doc: -1,
        doc_competitive: None,
    };
    let max_doc = u64::try_from(seg.max_doc.unwrap()).unwrap();
    // A run: counted at once, to one past the threshold.
    let mut tf = TopField::new(&[SortField::doc()], 5, 100, None);
    let mut l = leaf();
    count_rest(&mut *scorer(), &mut tf, &mut l, None).unwrap();
    assert_eq!(tf.total_hits, 101);
    assert!(l.terminated);
    assert_eq!(tf.relation, TotalHitsRelation::GreaterThanOrEqualTo);
    // Exhaustive: every document but the one already on (next_doc skips it).
    let mut tf = TopField::new(&[SortField::doc()], 5, u64::MAX, None);
    let mut l = leaf();
    count_rest(&mut *scorer(), &mut tf, &mut l, None).unwrap();
    assert_eq!(tf.total_hits, max_doc);
    assert!(!l.terminated);
    // With live documents, one at a time.
    let mut live = FixedBitSet::new(max_doc as usize);
    for d in (0..max_doc as usize).step_by(2) {
        live.set(d);
    }
    let mut tf = TopField::new(&[SortField::doc()], 5, 3, None);
    let mut l = leaf();
    count_rest(&mut *scorer(), &mut tf, &mut l, Some(&live)).unwrap();
    assert_eq!(tf.total_hits, 6, "threshold max(3, 5 hits) + 1");
    assert!(l.terminated);
}

#[test]
fn terms_read_back_from_the_postings_give_each_document_its_term_ordinal() {
    let reader = fixture("sorted_search_index");
    let opened = reader.open_segments().expect("open");
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    let max_doc = seg.max_doc.unwrap();
    let SortColumn::Terms {
        ords,
        offsets,
        bytes,
    } = uninvert_terms(seg, "body", max_doc).unwrap()
    else {
        panic!("terms")
    };
    assert!(offsets.len() > 2, "{} terms", offsets.len());
    let terms: Vec<&[u8]> = offsets
        .windows(2)
        .map(|w| &bytes[w[0] as usize..w[1] as usize])
        .collect();
    assert!(terms.windows(2).all(|w| w[0] < w[1]), "in term order");
    assert!(ords.iter().any(|&o| o >= 0));
    let mut dict = Dict::Mem(Arc::new(SortColumn::Terms {
        ords,
        offsets,
        bytes: bytes.clone(),
    }));
    assert_eq!(dict.lookup_term(terms[1]).unwrap(), 1);
    assert_eq!(dict.lookup_ord(1).unwrap(), terms[1]);
    assert_eq!(dict.lookup_term(b"\x00").unwrap(), -1);
}

#[test]
fn the_bulk_collector_keeps_its_first_error_and_asks_for_no_scores_by_document() {
    let mut tf = TopField::new(&[SortField::doc()], 2, 10, None);
    let mut l = Leaf {
        doc_base: 0,
        keys: vec![LeafKey::Doc],
        collected_all_competitive: false,
        after_doc: 0,
        terminated: false,
        early: false,
        score: 0.0,
        score_doc: -1,
        doc_competitive: None,
    };
    let mut c = BulkLeaf {
        tf: &mut tf,
        leaf: &mut l,
        min_score: None,
        error: Some(SortError::NoKeys.into()),
    };
    assert_eq!(c.score_mode(), ScoreMode::CompleteNoScores);
    assert_eq!(c.min_competitive_score(), None);
    c.collect(1, 0.5);
    assert_eq!(c.tf.total_hits, 0, "nothing collected past an error");
    let mut tf = TopField::new(&[SortField::doc(), SortField::score()], 2, 10, None);
    let mut l = Leaf {
        doc_base: 0,
        keys: vec![LeafKey::Doc, LeafKey::Score],
        collected_all_competitive: false,
        after_doc: 0,
        terminated: false,
        early: false,
        score: 0.0,
        score_doc: -1,
        doc_competitive: None,
    };
    let mut c = BulkLeaf {
        tf: &mut tf,
        leaf: &mut l,
        min_score: Some(1.0),
        error: None,
    };
    assert_eq!(c.score_mode(), ScoreMode::Complete);
    // Below the minimum (and a NaN score): not collected.
    c.collect(1, 0.5);
    c.collect(2, f32::NAN);
    assert_eq!(c.tf.total_hits, 0, "nothing below min_score collected");
    c.collect(3, 1.0);
    assert_eq!(c.tf.total_hits, 1);
}

#[test]
fn the_empty_queue_pops_nothing() {
    let mut q = HitQueue { heap: Vec::new() };
    assert!(q.top().is_none());
    assert!(q.pop(&[]).is_none());
}
