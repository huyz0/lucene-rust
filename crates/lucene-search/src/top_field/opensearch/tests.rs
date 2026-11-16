#![cfg(test)]
use std::sync::Arc;

use super::super::{search_sorted, FieldDoc, SortField};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture() -> DirectoryReader {
    let dir = format!(
        "{}/../../fixtures/data/sorted_search_index",
        env!("CARGO_MANIFEST_DIR")
    );
    DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
}

fn query(term: Option<&str>) -> BooleanQuery {
    let mut q = BooleanQuery::new();
    match term {
        Some(t) => q
            .must
            .push(Clause::Term(TermQuery::new("body", t.as_bytes()))),
        None => q.must.push(Clause::MatchAllDocs(MatchAllDocsQuery::new(0))),
    }
    q
}

fn sorted(
    reader: &DirectoryReader,
    q: &BooleanQuery,
    sort: &[SortField],
    top_n: usize,
    after: Option<&FieldDoc>,
) -> Vec<FieldDoc> {
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    search_sorted(
        &segments,
        reader.segment_readers(),
        q,
        &norms,
        sort,
        top_n,
        u64::MAX,
        after,
    )
    .unwrap()
    .hits
}

fn docs(hits: &[FieldDoc]) -> Vec<i32> {
    hits.iter().map(|h| h.doc).collect()
}

#[test]
fn scaled_float_picks_the_scaled_doubles() {
    use lucene_util::numeric_utils::{double_to_sortable_long, sortable_long_to_double};
    let inverse = 1.0 / 100.0;
    let mut scratch = Vec::new();
    let pick = |selector, values: &[i64], scratch: &mut Vec<i64>| {
        let s = ScaledFloatSort {
            selector,
            missing: f64::INFINITY,
            inverse,
        };
        sortable_long_to_double(Pick::Scaled(s).value(values, scratch))
    };
    let values = [-250, 100, 350];
    assert_eq!(pick(Selector::Min, &values, &mut scratch), -250.0 * inverse);
    assert_eq!(pick(Selector::Max, &values, &mut scratch), 350.0 * inverse);
    let sum = (-250.0 * inverse + 100.0 * inverse) + 350.0 * inverse;
    assert_eq!(pick(Selector::Sum, &values, &mut scratch), sum);
    assert_eq!(pick(Selector::Avg, &values, &mut scratch), sum / 3.0);
    assert_eq!(
        pick(Selector::Median, &values, &mut scratch),
        100.0 * inverse
    );
    assert_eq!(
        pick(Selector::Median, &[100, 200], &mut scratch),
        (100.0 * inverse + 200.0 * inverse) / 2.0
    );
    // No value: the missing one, infinity here.
    assert_eq!(pick(Selector::Min, &[], &mut scratch), f64::INFINITY);
    let source = ScaledFloatSort {
        selector: Selector::Min,
        missing: 0.0,
        inverse,
    };
    assert_eq!(source.value_string(double_to_sortable_long(2.5)), "2.5");
}

#[test]
fn one_value_takes_the_short_path_with_the_same_answer() {
    let mut scratch = Vec::new();
    for selector in [
        Selector::Min,
        Selector::Max,
        Selector::Sum,
        Selector::Avg,
        Selector::Median,
    ] {
        let picks = [
            Pick::Scaled(ScaledFloatSort {
                selector,
                missing: f64::INFINITY,
                inverse: 0.01,
            }),
            Pick::Unsigned(UnsignedLongSort {
                selector,
                missing: -1,
            }),
            Pick::HalfFloat(HalfFloatSort {
                selector,
                missing: f32::INFINITY,
            }),
        ];
        for pick in picks {
            for v in [
                None,
                Some(0),
                Some(1234),
                Some(-77),
                Some(i64::from(i16::MIN) + 5),
            ] {
                let long = v.as_slice();
                assert_eq!(pick.single(v), pick.value(long, &mut scratch));
            }
        }
    }
}

#[test]
fn unsigned_picks_are_opensearchs() {
    let big = |v: u64| v as i64;
    // Ascending as signed: the values past i64::MAX come first.
    let values = [big(u64::MAX - 1), big(u64::MAX), 3, 9];
    assert_eq!(unsigned_pick(Selector::Min, &values), 3);
    assert_eq!(unsigned_pick(Selector::Max, &values), big(u64::MAX));
    assert_eq!(unsigned_pick(Selector::Max, &[2, 7]), 7);
    assert_eq!(
        unsigned_pick(Selector::Max, &[big(1 << 63), big(u64::MAX)]),
        big(u64::MAX)
    );
    assert_eq!(
        unsigned_pick(Selector::Min, &[big(1 << 63), big(u64::MAX)]),
        big(1 << 63)
    );
    // MultiValueMode.MIN's quirk: a leading 0 among several gives the next.
    assert_eq!(unsigned_pick(Selector::Min, &[0, 4, 6]), 4);
    assert_eq!(unsigned_pick(Selector::Min, &[0]), 0);
    assert_eq!(unsigned_pick(Selector::Sum, &[big(u64::MAX), 2]), 1);
    // The unsigned median: [3, 9, MAX-1, MAX] in unsigned order, signed order has the big ones first.
    assert_eq!(unsigned_pick(Selector::Median, &[3]), 3);
    assert_eq!(unsigned_pick(Selector::Median, &[1, 4, 6]), 4);
    assert_eq!(unsigned_pick(Selector::Median, &[2, 4]), 3);
    assert_eq!(
        unsigned_pick(Selector::Median, &[big(u64::MAX - 1), big(u64::MAX), 3, 9]),
        // 9 + (MAX - 1) wraps to 7: Java's overflow, kept.
        4
    );
    assert_eq!(
        unsigned_pick(Selector::Median, &[big(u64::MAX - 1), big(u64::MAX), 3]),
        big(u64::MAX - 1)
    );
    assert_eq!(unsigned_pick(Selector::Avg, &[5]), 5);
    assert_eq!(unsigned_pick(Selector::Avg, &[1, 2]), 2);
    for (total, n) in [
        (7u64, 2u64),
        (9, 4),
        (u64::MAX, 2),
        (u64::MAX - 5, 3),
        (1 << 63, 7),
        (10, 4),
    ] {
        // Rounded half up, as BigInteger division then Math.round would.
        let want = ((u128::from(total) * 2 + u128::from(n)) / (2 * u128::from(n))) as u64;
        assert_eq!(
            divide_unsigned_and_round_up(total as i64, n as i64) as u64,
            want,
            "{total} / {n}"
        );
    }
}

#[test]
fn shard_doc_sorts_by_the_global_document() {
    let reader = fixture();
    let prefix = 3i64 << 32;
    let key =
        |reverse| SortField::with_source("_shard_doc", Arc::new(ShardDocSort { prefix }), reverse);
    let q = query(Some("w1"));
    let by_doc = sorted(&reader, &q, &[SortField::doc()], 100_000, None);
    let asc = sorted(&reader, &q, &[key(false)], 100_000, None);
    assert_eq!(docs(&asc), docs(&by_doc));
    assert!(asc
        .iter()
        .all(|h| h.values == vec![prefix | i64::from(h.doc)]));
    let desc = sorted(&reader, &q, &[key(true)], 40, None);
    let mut want = docs(&by_doc);
    want.reverse();
    assert_eq!(docs(&desc), want[..40]);
    // search_after: the documents after the 40th, as the key orders them.
    let next = sorted(&reader, &q, &[key(true)], 40, desc.last());
    assert_eq!(docs(&next), want[40..80]);
    // Another shard's key: everything or nothing is after it.
    let other = FieldDoc {
        doc: 0,
        values: vec![4i64 << 32],
        terms: Vec::new(),
    };
    assert_eq!(
        sorted(&reader, &q, &[key(true)], 40, Some(&other)).len(),
        40
    );
    assert!(sorted(&reader, &q, &[key(false)], 40, Some(&other)).is_empty());
    assert_eq!(
        ShardDocSort { prefix }.sort_field_string("_shard_doc", false),
        "<custom:\"_shard_doc\">"
    );
}

#[test]
fn unsigned_long_orders_as_the_signed_key_over_small_values() {
    let reader = fixture();
    let q = query(Some("w0"));
    for (selector, reverse) in [
        (Selector::Min, false),
        (Selector::Max, true),
        (Selector::Sum, false),
        (Selector::Avg, true),
    ] {
        // Missing: the greatest unsigned long, as the greatest signed one.
        let source = UnsignedLongSort {
            selector,
            missing: -1,
        };
        let key = SortField::with_source("m", Arc::new(source), reverse);
        let signed = SortField {
            selector,
            missing: i64::MAX,
            ..SortField::numeric("m", SortType::Long, reverse)
        };
        let got = sorted(&reader, &q, &[key.clone(), SortField::doc()], 300, None);
        let want = sorted(&reader, &q, &[signed, SortField::doc()], 300, None);
        assert_eq!(docs(&got), docs(&want), "{selector:?}");
        for (g, w) in got.iter().zip(&want) {
            let w = if w.values[0] == i64::MAX {
                -1
            } else {
                w.values[0]
            };
            // MIN's quirk aside, which a 0 among several values shows.
            if selector != Selector::Min {
                assert_eq!(g.values[0], w, "{selector:?} doc {}", g.doc);
            }
        }
        let page = sorted(
            &reader,
            &q,
            &[key.clone(), SortField::doc()],
            50,
            Some(&got[49]),
        );
        assert_eq!(docs(&page), docs(&got[50..100]));
    }
    assert_eq!(
        UnsignedLongSort {
            selector: Selector::Min,
            missing: 0
        }
        .value_string(-1),
        u64::MAX.to_string()
    );
}

#[test]
fn half_float_reads_sortable_shorts() {
    let reader = fixture();
    let q = query(None);
    for (selector, reverse) in [
        (Selector::Min, false),
        (Selector::Max, true),
        (Selector::Median, false),
    ] {
        let source = HalfFloatSort {
            selector,
            missing: f32::INFINITY,
        };
        let key = SortField::with_source("l", Arc::new(source), reverse);
        let signed = SortField {
            selector,
            missing: i64::MAX,
            ..SortField::numeric("l", SortType::Long, reverse)
        };
        let got = sorted(&reader, &q, &[key, SortField::doc()], 500, None);
        let want = sorted(&reader, &q, &[signed, SortField::doc()], 500, None);
        assert_eq!(docs(&got), docs(&want), "{selector:?}");
        for (g, w) in got.iter().zip(&want) {
            let f = if w.values[0] == i64::MAX {
                f32::INFINITY
            } else {
                lucene_util::half_float::sortable_short_to_half_float(w.values[0] as i16)
            };
            assert_eq!(
                g.values[0],
                i64::from(float_to_sortable_int(f)),
                "doc {}",
                g.doc
            );
        }
    }
    let s = HalfFloatSort {
        selector: Selector::Avg,
        missing: 0.0,
    };
    assert_eq!(s.value_string(i64::from(float_to_sortable_int(1.5))), "1.5");
    // A document's values averaged as doubles, then cast to float.
    let pick = Pick::HalfFloat(s);
    let raw = |f: f32| i64::from(lucene_util::half_float::half_float_to_sortable_short(f));
    let mut scratch = Vec::new();
    assert_eq!(
        pick.value(&[raw(1.0), raw(2.5)], &mut scratch),
        i64::from(float_to_sortable_int(1.75))
    );
    assert_eq!(
        pick.value(&[], &mut scratch),
        i64::from(float_to_sortable_int(0.0))
    );
}

#[test]
fn wrong_doc_values_type_is_refused() {
    let dir = format!(
        "{}/../../fixtures/data/keyword_sort_index",
        env!("CARGO_MANIFEST_DIR")
    );
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let key = SortField::with_source(
        "k",
        Arc::new(UnsignedLongSort {
            selector: Selector::Min,
            missing: 0,
        }),
        false,
    );
    let e = search_sorted(
        &segments,
        reader.segment_readers(),
        &query(None),
        &norms,
        &[key],
        5,
        u64::MAX,
        None,
    )
    .unwrap_err();
    assert!(e.to_string().contains('k'), "{e}");
    let key = SortField::with_source(
        "absent",
        Arc::new(HalfFloatSort {
            selector: Selector::Max,
            missing: 2.0,
        }),
        false,
    );
    let hits = search_sorted(
        &segments,
        reader.segment_readers(),
        &query(None),
        &norms,
        &[key],
        5,
        u64::MAX,
        None,
    )
    .unwrap()
    .hits;
    assert!(hits
        .iter()
        .all(|h| h.values[0] == i64::from(float_to_sortable_int(2.0))));
}

#[test]
fn an_uninverted_keyword_key_sorts_by_the_terms_read_back() {
    let reader = fixture();
    let mut key = SortField::numeric("body", super::super::SortType::String, false);
    key.uninverted = true;
    let hits = sorted(&reader, &query(None), &[key.clone()], 5, None);
    let mut rev = key;
    rev.reverse = true;
    let back = sorted(&reader, &query(None), &[rev], 5, None);
    assert_eq!(hits.len(), 5);
    let first: Vec<_> = hits.iter().map(|h| h.terms[0].clone()).collect();
    let last: Vec<_> = back.iter().map(|h| h.terms[0].clone()).collect();
    assert!(first.iter().all(Option::is_some), "{first:?}");
    assert!(
        first[0] <= first[4] && last[0] >= last[4],
        "{first:?} {last:?}"
    );
    assert!(first[0] < last[0]);
}

#[test]
fn an_id_term_decodes_to_the_id_string() {
    use super::super::decode_id;
    assert_eq!(decode_id(&[0xff, b'a', b'1']), b"a1");
    assert_eq!(decode_id(&[0xfe, 0x0f]), b"0");
    assert_eq!(decode_id(&[0xfe, 0x10, 0x4f]), b"104");
    assert_eq!(decode_id(&[0xfe, 0x12, 0x34]), b"1234");
    assert_eq!(decode_id(&[0x0f]), b"Dw");
    assert_eq!(decode_id(&[0xfd, 0xff, 0xfe]), b"__4");
    assert_eq!(decode_id(&[]), b"");
}
