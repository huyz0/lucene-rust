#![cfg(test)]
use super::super::{search_sorted, SortType};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::query::{Clause, MatchAllDocsQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture(name: &str) -> DirectoryReader {
    let dir = format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"));
    DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
}

fn all() -> BooleanQuery {
    let mut q = BooleanQuery::new();
    q.must.push(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
    q
}

fn term(field: &str, text: &str) -> BooleanQuery {
    let mut q = BooleanQuery::new();
    q.must
        .push(Clause::Term(TermQuery::new(field, text.as_bytes())));
    q
}

/// Every live document's collapse value, by global id.
fn values(reader: &DirectoryReader, field: &str, kind: CollapseKind) -> Vec<CollapseValue> {
    let mut out = Vec::new();
    for r in reader.segment_readers() {
        match kind {
            CollapseKind::Numeric => {
                let mut v = dv::get_sorted_numeric(r, field).unwrap();
                for doc in 0..r.max_doc {
                    out.push(if v.advance_exact(doc).unwrap() {
                        CollapseValue::Long(v.next_value().unwrap())
                    } else {
                        CollapseValue::Missing
                    });
                }
            }
            CollapseKind::Keyword => {
                let mut v = dv::get_sorted_set(r, field).unwrap();
                for doc in 0..r.max_doc {
                    out.push(if v.advance_exact(doc).unwrap() {
                        let o = v.next_ord().unwrap();
                        CollapseValue::Bytes(v.lookup_ord(o).unwrap())
                    } else {
                        CollapseValue::Missing
                    });
                }
            }
        }
    }
    out
}

/// The collapse by its definition: every match in the sort's order (ties
/// to the lower document), each group's first one kept, the first
/// `top_n` groups.
fn brute(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &[SortField],
    field: &str,
    kind: CollapseKind,
    top_n: usize,
) -> (Vec<FieldDoc>, Vec<CollapseValue>, u64) {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let every = search_sorted(
        &segments,
        reader.segment_readers(),
        q,
        &norms,
        sort,
        reader.max_doc() as usize,
        u64::MAX,
        None,
    )
    .unwrap();
    let by_doc = values(reader, field, kind);
    let mut seen = std::collections::HashSet::new();
    let (mut hits, mut groups) = (Vec::new(), Vec::new());
    for hit in every.hits {
        let v = by_doc[hit.doc as usize].clone();
        if seen.insert(v.clone()) && hits.len() < top_n {
            hits.push(hit);
            groups.push(v);
        }
    }
    (hits, groups, every.total.value)
}

#[allow(clippy::too_many_arguments)]
fn collapsed(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &[SortField],
    field: &str,
    kind: CollapseKind,
    top_n: usize,
    after: Option<&FieldDoc>,
    slices: &[Vec<usize>],
) -> Result<Vec<CollapseTopDocs>> {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    search_collapsed(
        &segments,
        reader.segment_readers(),
        q,
        &norms,
        sort,
        field,
        kind,
        top_n,
        after,
        false,
        slices,
        None,
    )
}

fn keyword(field: &str, reverse: bool, last: bool) -> SortField {
    SortField {
        missing: i64::from(last),
        ..SortField::string(field, reverse)
    }
}

#[test]
fn collapsing_keeps_each_groups_best_document_and_the_best_groups() {
    let numeric = fixture("sorted_search_index");
    let words = fixture("keyword_sort_index");
    let long = |f: &str, reverse| SortField::numeric(f, SortType::Long, reverse);
    let cases: Vec<(
        &DirectoryReader,
        BooleanQuery,
        Vec<SortField>,
        &str,
        CollapseKind,
    )> = vec![
        (
            &numeric,
            all(),
            vec![long("l", false)],
            "i",
            CollapseKind::Numeric,
        ),
        (
            &numeric,
            all(),
            vec![long("l", true), SortField::doc()],
            "i",
            CollapseKind::Numeric,
        ),
        (
            &numeric,
            term("body", "w3"),
            vec![SortField::score()],
            "s",
            CollapseKind::Numeric,
        ),
        (
            &numeric,
            term("body", "w0"),
            vec![
                SortField::numeric("i", SortType::Int, false),
                SortField::score(),
            ],
            "l",
            CollapseKind::Numeric,
        ),
        (
            &numeric,
            all(),
            vec![SortField::doc()],
            "l",
            CollapseKind::Numeric,
        ),
        (
            &words,
            all(),
            vec![keyword("kd", false, true)],
            "k",
            CollapseKind::Keyword,
        ),
        (
            &words,
            all(),
            vec![
                keyword("k", true, false),
                SortField::numeric("i", SortType::Int, false),
            ],
            "kd",
            CollapseKind::Keyword,
        ),
        (
            &words,
            all(),
            vec![SortField::numeric("i", SortType::Int, true)],
            "kn",
            CollapseKind::Keyword,
        ),
    ];
    for (reader, q, sort, field, kind) in &cases {
        for top_n in [1, 7, 120, 5000] {
            let (want_hits, want_values, want_total) = brute(reader, q, sort, field, *kind, top_n);
            for slices in [vec![], vec![vec![2, 0, 1]]] {
                let got = collapsed(reader, q, sort, field, *kind, top_n, None, &slices).unwrap();
                assert_eq!(got.len(), 1);
                let got = &got[0];
                let docs = |h: &[FieldDoc]| h.iter().map(|h| h.doc).collect::<Vec<_>>();
                assert_eq!(
                    docs(&got.hits),
                    docs(&want_hits),
                    "{field} {sort:?} top {top_n}"
                );
                assert_eq!(got.hits, want_hits, "{field} {sort:?} top {top_n}");
                assert_eq!(got.values, want_values, "{field} {sort:?} top {top_n}");
                assert_eq!(got.total, want_total);
                assert!(got.max_score.is_nan());
            }
        }
    }
}

#[test]
fn slices_each_collapse_on_their_own() {
    let reader = fixture("sorted_search_index");
    let sort = vec![SortField::numeric("l", SortType::Long, false)];
    let slices = vec![vec![1], vec![0, 2]];
    let got = collapsed(
        &reader,
        &all(),
        &sort,
        "i",
        CollapseKind::Numeric,
        10,
        None,
        &slices,
    )
    .unwrap();
    assert_eq!(got.len(), 2);
    let per_slice: u64 = got.iter().map(|g| g.total).sum();
    assert_eq!(per_slice, u64::from(reader.num_docs() as u32));
    // The second slice's segments: its hits come from segments 0 and 2.
    let bases: Vec<i32> = reader
        .segment_readers()
        .iter()
        .scan(0, |b, r| {
            let base = *b;
            *b += r.max_doc;
            Some(base)
        })
        .collect();
    for h in &got[0].hits {
        assert!(h.doc >= bases[1] && h.doc < bases[2], "{}", h.doc);
    }
    for h in &got[1].hits {
        assert!(h.doc < bases[1] || h.doc >= bases[2], "{}", h.doc);
    }
}

#[test]
fn search_after_continues_past_the_previous_page() {
    let reader = fixture("keyword_sort_index");
    for reverse in [false, true] {
        let sort = vec![keyword("k", reverse, false)];
        let (every, values, _) = brute(&reader, &all(), &sort, "k", CollapseKind::Keyword, 10_000);
        let page = 25;
        let first = &collapsed(
            &reader,
            &all(),
            &sort,
            "k",
            CollapseKind::Keyword,
            page,
            None,
            &[],
        )
        .unwrap()[0];
        assert_eq!(first.values, values[..page]);
        let last = first.hits.last().unwrap().clone();
        let second = &collapsed(
            &reader,
            &all(),
            &sort,
            "k",
            CollapseKind::Keyword,
            page,
            Some(&last),
            &[],
        )
        .unwrap()[0];
        assert_eq!(second.values, values[page..2 * page], "reverse {reverse}");
        assert_eq!(second.hits[0].doc, every[page].doc);
        // Every match is counted, before the page or after it.
        assert_eq!(second.total, first.total);
    }
    let numeric = fixture("sorted_search_index");
    let sort = vec![SortField::numeric("i", SortType::Int, true)];
    let first = &collapsed(
        &numeric,
        &all(),
        &sort,
        "i",
        CollapseKind::Numeric,
        3,
        None,
        &[],
    )
    .unwrap()[0];
    let next = &collapsed(
        &numeric,
        &all(),
        &sort,
        "i",
        CollapseKind::Numeric,
        3,
        first.hits.last(),
        &[],
    )
    .unwrap()[0];
    let (_, values, _) = brute(&numeric, &all(), &sort, "i", CollapseKind::Numeric, 6);
    assert_eq!(first.values, values[..3]);
    // Java's search_after: a later document tied with the last hit on its
    // value opens that group again on the next page.
    let mut both = [first.values.clone(), next.values.clone()].concat();
    assert!(
        next.values[0] == first.values[2] || next.values[0] == values[3],
        "{both:?}"
    );
    both.dedup();
    assert_eq!(both, values[..both.len()]);
}

#[test]
fn scores_track_and_min_score_filter() {
    let reader = fixture("sorted_search_index");
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let q = term("body", "w0");
    let sort = vec![SortField::numeric("l", SortType::Long, false)];
    let run = |track, min| {
        search_collapsed(
            &segments,
            reader.segment_readers(),
            &q,
            &norms,
            &sort,
            "i",
            CollapseKind::Numeric,
            5,
            None,
            track,
            &[],
            min,
        )
        .unwrap()
        .remove(0)
    };
    let tracked = run(true, None);
    assert!(tracked.max_score > 0.0);
    let by_score = search_sorted(
        &segments,
        reader.segment_readers(),
        &q,
        &norms,
        &[SortField::score()],
        1,
        u64::MAX,
        None,
    )
    .unwrap();
    assert_eq!(
        tracked.max_score.to_bits(),
        by_score.hits[0].values[0] as u32
    );
    let none = run(true, Some(f32::MAX));
    assert_eq!((none.total, none.hits.len()), (0, 0));
    assert!(none.max_score.is_nan());
    // A threshold between the distinct scores: only the documents above it count.
    let every = search_sorted(
        &segments,
        reader.segment_readers(),
        &q,
        &norms,
        &[SortField::score()],
        100_000,
        u64::MAX,
        None,
    )
    .unwrap();
    let mut scores: Vec<f32> = every
        .hits
        .iter()
        .map(|h| f32::from_bits(h.values[0] as u32))
        .collect();
    scores.dedup();
    assert!(scores.len() > 1, "{scores:?}");
    let cut = scores[scores.len() / 2];
    let above = every
        .hits
        .iter()
        .filter(|h| f32::from_bits(h.values[0] as u32) >= cut)
        .count() as u64;
    let some = run(false, Some(cut));
    assert_eq!(some.total, above);
    assert!(some.total > 0 && some.total < tracked.total);
}

#[test]
fn refusals_are_javas() {
    let reader = fixture("sorted_search_index");
    let sort = vec![SortField::numeric("l", SortType::Long, false)];
    // A multi-valued field, once a document with two values competes.
    let e = collapsed(
        &reader,
        &all(),
        &sort,
        "m",
        CollapseKind::Numeric,
        10,
        None,
        &[],
    )
    .unwrap_err();
    assert!(e.to_string().contains(MULTI_VALUED), "{e}");
    let words = fixture("keyword_sort_index");
    let e = collapsed(
        &words,
        &all(),
        &[keyword("k", false, true)],
        "km",
        CollapseKind::Keyword,
        10,
        None,
        &[],
    )
    .unwrap_err();
    assert!(e.to_string().contains(MULTI_VALUED), "{e}");
    // The wrong kind of doc values for the selector.
    let e = collapsed(
        &words,
        &all(),
        &sort,
        "kd",
        CollapseKind::Numeric,
        10,
        None,
        &[],
    )
    .unwrap_err();
    assert!(e.to_string().contains("unexpected doc values type"), "{e}");
    let e = collapsed(
        &reader,
        &all(),
        &sort,
        "l",
        CollapseKind::Keyword,
        10,
        None,
        &[],
    )
    .unwrap_err();
    assert!(e.to_string().contains("unexpected doc values type"), "{e}");
    // A field no segment has: one group of documents without a value.
    let none = &collapsed(
        &reader,
        &all(),
        &sort,
        "absent",
        CollapseKind::Keyword,
        10,
        None,
        &[],
    )
    .unwrap()[0];
    assert_eq!(none.values, vec![CollapseValue::Missing]);
    // The arguments OpenSearch checks.
    assert!(collapsed(
        &reader,
        &all(),
        &[],
        "i",
        CollapseKind::Numeric,
        10,
        None,
        &[]
    )
    .is_err());
    assert!(collapsed(
        &reader,
        &all(),
        &sort,
        "i",
        CollapseKind::Numeric,
        0,
        None,
        &[]
    )
    .is_err());
    let two = vec![sort[0].clone(), SortField::doc()];
    let after = FieldDoc {
        doc: 0,
        values: vec![0, 0],
        terms: Vec::new(),
    };
    assert!(collapsed(
        &reader,
        &all(),
        &two,
        "i",
        CollapseKind::Numeric,
        10,
        Some(&after),
        &[]
    )
    .is_err());
    assert!(collapsed(
        &reader,
        &all(),
        &sort,
        "i",
        CollapseKind::Numeric,
        10,
        None,
        &[vec![99]]
    )
    .is_err());
}
