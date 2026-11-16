#![cfg(test)]
use super::*;
use crate::collector::VecCollector;
use lucene_codecs::points::{self, WritePointsField};
use lucene_store::codec_util::ID_LENGTH;

fn long_bytes(v: i64) -> [u8; 8] {
    ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
}

/// Same single-dimension `LongPoint`-shaped fixture
/// `lucene_index::points_delete`'s tests use: doc 0 -> 10, doc 1 -> 20,
/// doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
fn build_single_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
    let segment_id = [9u8; ID_LENGTH];
    let points: Vec<(i32, Vec<u8>)> = vec![
        (0, long_bytes(10).to_vec()),
        (1, long_bytes(20).to_vec()),
        (2, long_bytes(30).to_vec()),
        (3, long_bytes(40).to_vec()),
        (4, long_bytes(50).to_vec()),
    ];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    (kdm, kdi, kdd, segment_id)
}

/// `points_range_lowest_docs` returns the first `n` of what
/// `search_points_range` returns, and its full count -- on a single-valued
/// field (the selection path) and a multi-valued one (the sort path), with
/// and without deletions, for `n` below, at and above the match count.
#[test]
fn lowest_docs_is_a_prefix_of_the_full_sorted_result() {
    let segment_id = [3u8; ID_LENGTH];
    let mut x = 0x1357_9BDF_2468_ACE0u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let single: Vec<(i32, Vec<u8>)> = (0..3000)
        .map(|d| (d, long_bytes((next() % 1000) as i64).to_vec()))
        .collect();
    // Every doc twice: a doc id can match more than once.
    let multi: Vec<(i32, Vec<u8>)> = (0..3000)
        .flat_map(|d| [(d, (next() % 1000) as i64), (d, (next() % 1000) as i64)])
        .map(|(d, v)| (d, long_bytes(v).to_vec()))
        .collect();
    let mut live = FixedBitSet::new(3000);
    for d in 0..3000 {
        if next() % 4 != 0 {
            live.set(d);
        }
    }
    for points in [single, multi] {
        let field = WritePointsField {
            field_number: 1,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = points::write(&[field], 64, &segment_id, "").unwrap();
        let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();
        for live_docs in [None, Some(&live)] {
            for (lo, hi) in [(0i64, 5), (100, 400), (0, 999), (2000, 3000)] {
                let mut all = VecCollector::default();
                search_points_range(
                    &reader,
                    live_docs,
                    1,
                    &long_bytes(lo),
                    &long_bytes(hi),
                    &mut all,
                )
                .unwrap();
                for n in [0usize, 1, 10, 50, 100_000] {
                    let (got, total) = points_range_lowest_docs(
                        &reader,
                        live_docs,
                        1,
                        &long_bytes(lo),
                        &long_bytes(hi),
                        n,
                    )
                    .unwrap();
                    assert_eq!(total, all.docs.len(), "[{lo},{hi}] n={n}");
                    assert_eq!(
                        got[..],
                        all.docs[..n.min(all.docs.len())],
                        "[{lo},{hi}] n={n}"
                    );
                }
            }
        }
    }
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    assert_eq!(
        points_range_lowest_docs(&reader, None, 99, &long_bytes(0), &long_bytes(9), 5).unwrap(),
        (Vec::new(), 0)
    );
}

/// The multi-segment top-`n` returns the global lowest `n` matching doc
/// ids, and a count that is exact when it says so and otherwise a lower
/// bound past the threshold -- whatever order the segments are given in.
#[test]
fn top_n_multi_segment_matches_a_brute_force_merge() {
    let segment_id = [4u8; ID_LENGTH];
    let mut x = 0xDEAD_10CC_BEEF_CAFEu64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut files = Vec::new();
    let mut values: Vec<Vec<i64>> = Vec::new();
    for _ in 0..5 {
        let n = 50 + (next() % 800) as i32;
        let vals: Vec<i64> = (0..n).map(|_| (next() % 1000) as i64).collect();
        let points = vals
            .iter()
            .enumerate()
            .map(|(d, &v)| (d as i32, long_bytes(v).to_vec()))
            .collect();
        let field = WritePointsField {
            field_number: 1,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        files.push(points::write(&[field], 64, &segment_id, "").unwrap());
        values.push(vals);
    }
    let readers: Vec<PointsReader<'_>> = files
        .iter()
        .map(|(kdm, kdi, kdd)| points::open(kdm, kdi, kdd, &segment_id, "").unwrap())
        .collect();
    let mut base = 0i32;
    let mut segs: Vec<PointsSegment<'_>> = Vec::new();
    for (r, vals) in readers.iter().zip(&values) {
        segs.push(PointsSegment {
            reader: r,
            field_number: 1,
            live_docs: None,
            doc_base: base,
        });
        base += vals.len() as i32;
    }
    for (lo, hi) in [(0i64, 10), (0, 300), (0, 999), (5000, 6000)] {
        let mut all = Vec::new();
        let mut b = 0i32;
        for vals in &values {
            for (d, &v) in vals.iter().enumerate() {
                if v >= lo && v <= hi {
                    all.push(b + d as i32);
                }
            }
            b += vals.len() as i32;
        }
        for (n, threshold) in [(0usize, 0usize), (10, 5), (50, 1000), (3000, 100_000)] {
            // Same answer whatever order the segments arrive in.
            let mut shuffled = segs.clone();
            shuffled.reverse();
            for input in [&segs, &shuffled] {
                let (hits, total, exact) = points_range_top_n_multi_segment(
                    input,
                    &long_bytes(lo),
                    &long_bytes(hi),
                    n,
                    threshold,
                )
                .unwrap();
                assert_eq!(hits[..], all[..n.min(all.len())], "[{lo},{hi}] n={n}");
                if exact {
                    assert_eq!(total, all.len());
                } else {
                    assert!(total > threshold && total <= all.len());
                }
            }
        }
    }
}

/// 2D `LatLonPoint`-shaped fixture: doc 0 -> (0, 0), doc 1 -> (10, 10),
/// doc 2 -> (20, 20), doc 3 -> (10, 100) -- dimension 0 alone would match
/// doc 3 but dimension 1 must independently be in range too, exercising
/// this port's already-built multi-dimension BKD points support end to
/// end through the search-side entry point (not just the delete-side
/// one).
fn build_two_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
    let segment_id = [11u8; ID_LENGTH];
    let pack = |a: i64, b: i64| -> Vec<u8> {
        let mut v = long_bytes(a).to_vec();
        v.extend_from_slice(&long_bytes(b));
        v
    };
    let points: Vec<(i32, Vec<u8>)> = vec![
        (0, pack(0, 0)),
        (1, pack(10, 10)),
        (2, pack(20, 20)),
        (3, pack(10, 100)),
    ];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    (kdm, kdi, kdd, segment_id)
}

#[test]
fn matches_only_docs_within_bounds() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(15);
    let max = long_bytes(35);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![1, 2]); // values 20, 30
}

#[test]
fn boundary_values_are_inclusive_on_both_ends() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(10);
    let max = long_bytes(30);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![0, 1, 2]);
}

#[test]
fn empty_range_matches_no_docs() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(1000);
    let max = long_bytes(2000);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert!(collector.docs.is_empty());
}

#[test]
fn full_range_matches_every_doc() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(i64::MIN);
    let max = long_bytes(i64::MAX);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![0, 1, 2, 3, 4]);
}

#[test]
fn unknown_field_number_matches_nothing_not_an_error() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(i64::MIN);
    let max = long_bytes(i64::MAX);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 99, &min, &max, &mut collector).unwrap();
    assert!(collector.docs.is_empty());
}

#[test]
fn live_docs_filter_excludes_already_deleted_docs() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let mut live = FixedBitSet::new(5);
    for i in 0..5 {
        live.set(i);
    }
    live.clear(1); // doc 1 (value 20) already deleted
    let min = long_bytes(15);
    let max = long_bytes(35);
    let mut collector = VecCollector::default();
    search_points_range(&reader, Some(&live), 1, &min, &max, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![2]); // only doc 2 (value 30) is live
}

#[test]
fn two_dimension_range_checks_every_dimension_independently() {
    let (kdm, kdi, kdd, id) = build_two_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let pack = |a: i64, b: i64| -> Vec<u8> {
        let mut v = long_bytes(a).to_vec();
        v.extend_from_slice(&long_bytes(b));
        v
    };
    let min = pack(0, 0);
    let max = pack(20, 20);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    // doc 3's dim-0 value (10) is in range but dim-1 (100) is not, so it
    // must be excluded even though a single-dimension check would match.
    assert_eq!(collector.docs, vec![0, 1, 2]);
}

#[test]
fn corrupt_kdd_leaf_data_surfaces_as_points_error() {
    let (kdm, kdi, mut kdd, id) = build_single_dim_fixture();
    // `points::open` only validates the `.kdd` codec header + footer, not
    // every leaf block's content in between (that's only decoded lazily,
    // per field, during the tree walk) -- so scrambling the bytes strictly
    // between the header and the trailing
    // `lucene_store::codec_util::FOOTER_LENGTH`-byte footer keeps `open`
    // itself succeeding, but forces the leaf-block read to fail partway
    // through. This exercises the error path through
    // `search_points_range` itself (not just through `points::open`),
    // confirming the crate-level `Error::Points` wiring documented on
    // this module.
    //
    // The range below has to be a *narrow* one. With the real
    // `PointValues.intersect` pruning walk, an all-encompassing `[i64::MIN,
    // i64::MAX]` range hits `CELL_INSIDE_QUERY` at the root, which --
    // exactly like Java's `visitDocIDs` shortcut -- reads only the leaf's
    // doc-id block and never decodes a packed value, so it can happily
    // return garbage doc ids from garbage bytes without erroring (detecting
    // *that* is `checkIntegrity`'s/the footer checksum's job, not the tree
    // walk's, in Java too). A narrow range makes the root cell
    // `CELL_CROSSES_QUERY`, which is the branch that decodes the block.
    let footer_start = kdd.len() - lucene_store::codec_util::FOOTER_LENGTH;
    // The `.kdd` index header: magic (4) + `"Lucene90PointsFormatData"`
    // (1 + 24) + version (4) + id (16) + empty suffix (1). Scrambling from
    // the first leaf byte on corrupts its point count, which no leaf
    // layout can decode past.
    let header_end = 50;
    for b in kdd[header_end..footer_start].iter_mut() {
        *b = 0xFF;
    }
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(15);
    let max = long_bytes(35);
    let mut collector = VecCollector::default();
    let err = search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap_err();
    assert!(matches!(err, Error::Points(_)));
}

// --- `search_points_in_set` (`PointInSetQuery`) ---

#[test]
fn in_set_matches_exactly_the_requested_values() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let points = vec![
        long_bytes(20).to_vec(),
        long_bytes(50).to_vec(),
        long_bytes(45).to_vec(), // no doc has this value
    ];
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![1, 4]); // values 20 and 50
}

#[test]
fn in_set_is_exact_equality_not_a_range() {
    // A set spanning the whole value space's endpoints must NOT pull in
    // the values between them -- the single sharpest difference between
    // `PointInSetQuery` and `PointRangeQuery`.
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let points = vec![long_bytes(10).to_vec(), long_bytes(50).to_vec()];
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![0, 4]);
}

#[test]
fn in_set_unsorted_and_duplicated_input_is_normalized() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    // Deliberately descending, with a duplicate: `search_points_in_set`
    // must sort + dedup before the merge walk (Java's `sortedPackedPoints`).
    let points = vec![
        long_bytes(40).to_vec(),
        long_bytes(20).to_vec(),
        long_bytes(40).to_vec(),
    ];
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![1, 3]);
}

#[test]
fn in_set_empty_and_unknown_field_collect_nothing() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &[], &mut collector).unwrap();
    assert!(collector.docs.is_empty());
    search_points_in_set(
        &reader,
        None,
        99,
        &[long_bytes(10).to_vec()],
        &mut collector,
    )
    .unwrap();
    assert!(collector.docs.is_empty());
}

#[test]
fn in_set_respects_live_docs() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let mut live = FixedBitSet::new(5);
    for i in 0..5 {
        live.set(i);
    }
    live.clear(1); // doc 1 (value 20) deleted
    let points = vec![long_bytes(20).to_vec(), long_bytes(30).to_vec()];
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, Some(&live), 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![2]);
}

#[test]
fn in_set_multi_dimension_requires_every_dimension_to_match() {
    let (kdm, kdi, kdd, id) = build_two_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let pack = |a: i64, b: i64| -> Vec<u8> {
        let mut v = long_bytes(a).to_vec();
        v.extend_from_slice(&long_bytes(b));
        v
    };
    // (10, 10) is doc 1; (10, 20) matches no doc even though doc 1 shares
    // dim 0 and doc 2 shares dim 1.
    let points = vec![pack(10, 10), pack(10, 20)];
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![1]);
}

#[test]
fn in_set_wrong_length_point_matches_nothing() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &[vec![0u8; 3]], &mut collector).unwrap();
    assert!(collector.docs.is_empty());
}

#[test]
fn in_set_every_indexed_value_matches_every_doc() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let points: Vec<Vec<u8>> = (1..=5).map(|i| long_bytes(i * 10).to_vec()).collect();
    let mut collector = VecCollector::default();
    search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![0, 1, 2, 3, 4]);
}

// -----------------------------------------------------------------
// `PointValues.estimateDocCount`
// -----------------------------------------------------------------

/// Java's three branches, each with a case only that branch produces.
#[test]
fn estimate_doc_count_follows_javas_three_branches() {
    // 1. An estimate at or above the field's whole point count matches
    //    every document, however the points are distributed.
    assert_eq!(estimate_doc_count(100, 100, 40), 40);
    assert_eq!(estimate_doc_count(1_000, 100, 40), 40);

    // 2. Single-valued (`size == docCount`): the point estimate *is* the
    //    document estimate.
    assert_eq!(estimate_doc_count(37, 100, 100), 37);
    // ...and a zero estimate is zero whatever the shape, so a
    // provably-empty range never costs anything.
    assert_eq!(estimate_doc_count(0, 100, 40), 0);

    // 3. Multi-valued: `D * (1 - ((N - n)/N)^(N/D))`, floored at 1.
    //    N = 100 points over D = 50 docs (2 points per doc), n = 50:
    //    50 * (1 - 0.5^2) = 37.5 -> truncates to 37.
    assert_eq!(estimate_doc_count(50, 100, 50), 37);
    //    N = 1000 over D = 100 (10 points/doc), n = 100:
    //    100 * (1 - 0.9^10) = 65.13... -> 65.
    assert_eq!(estimate_doc_count(100, 1000, 100), 65);
    //    A tiny estimate rounds to zero documents, which Java floors to 1
    //    rather than letting a non-empty match report no cost at all.
    assert_eq!(estimate_doc_count(1, 1_000_000, 1), 1);
}

/// The `IndexOrDocValuesQuery` question: with a real
/// estimate in hand, does the planner make a *different* choice from the
/// `None` default -- and when?
///
/// The estimate comes from Java's own walk --
/// `PointsReader::estimate_range_point_count` -- over the real `fixtures/data/points_index/`
/// field, not from a match count standing in for it. The narrow range is
/// deliberately one where the two differ (256 estimated against 8
/// matched), so this test would not pass against the stand-in. The field
/// is single-valued (`point_count == doc_count`), so the estimate passes
/// straight through branch 2 of [`estimate_doc_count`].
#[test]
fn a_real_point_estimate_changes_the_index_or_doc_values_plan_only_when_another_clause_leads() {
    use crate::doc_value_query::{plan_index_or_doc_values, IndexOrDocValuesPlan};

    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/points_index/"
    );
    let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run scripts/gen-fixtures.sh first (GenPoints)");
    let kv: std::collections::HashMap<&str, &str> =
        text.lines().filter_map(|l| l.split_once('=')).collect();
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&kv["id_hex"][i * 2..i * 2 + 2], 16).unwrap();
    }
    let read = |key: &str| std::fs::read(format!("{dir}{}", kv[key])).unwrap();
    let (kdm, kdi, kdd) = (
        read("kdm_file_name"),
        read("kdi_file_name"),
        read("kdd_file_name"),
    );
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field_number: i32 = kv["field_number"].parse().unwrap();
    let field = reader.field(field_number).unwrap();
    let size = field.point_count;
    let doc_count = field.doc_count;
    assert_eq!(size, doc_count as i64, "the fixture field is single-valued");

    // The whole range: every point matches, so the estimate is the whole
    // field and `estimate_doc_count` short-circuits to `docCount`.
    let mut all = VecCollector::default();
    search_points_range(
        &reader,
        None,
        field_number,
        &long_bytes(i64::MIN),
        &long_bytes(i64::MAX),
        &mut all,
    )
    .unwrap();
    assert_eq!(all.docs.len(), doc_count as usize);
    let full_points = reader
        .estimate_range_point_count(field_number, &long_bytes(i64::MIN), &long_bytes(i64::MAX))
        .unwrap();
    assert_eq!(full_points, size, "a whole-field box is one INSIDE cell");
    let full_cost = estimate_doc_count(full_points, size, doc_count);
    assert_eq!(full_cost, doc_count as i64);

    // A narrow range, where the walk's estimate and the match count part
    // company.
    let mut narrow = VecCollector::default();
    search_points_range(
        &reader,
        None,
        field_number,
        &long_bytes(0),
        &long_bytes(100_000),
        &mut narrow,
    )
    .unwrap();
    let narrow_points = reader
        .estimate_range_point_count(field_number, &long_bytes(0), &long_bytes(100_000))
        .unwrap();
    // The estimate over-counts a partly-covered leaf on purpose -- Java's
    // "assume half the points matched" -- so it is *not* the match count.
    assert_eq!(narrow.docs.len(), 8);
    assert_eq!(narrow_points, 256);
    let narrow_cost = estimate_doc_count(narrow_points, size, doc_count);
    assert!(narrow_cost > 0 && narrow_cost < full_cost);

    // **The answer.** Java's rule is `cost >>> 3 <= leadCost`, so:
    //
    // - when this query leads (`lead_cost == cost`), the estimate cannot
    //   change the plan -- `cost/8 <= cost` for every non-negative cost --
    //   and the planner picks `Index`, which is exactly what `None`
    //   already picked. The estimate is *not* wasted work here; it is what
    //   proves the default was right rather than lucky.
    for cost in [full_cost, narrow_cost] {
        assert_eq!(
            plan_index_or_doc_values(Some(cost), cost),
            IndexOrDocValuesPlan::Index
        );
        assert_eq!(
            plan_index_or_doc_values(None, cost),
            IndexOrDocValuesPlan::Index
        );
    }

    // - the estimate changes the plan exactly when some *other* clause
    //   leads with fewer than `cost/8` documents. With the real numbers
    //   below that threshold is a concrete doc count, and on either side
    //   of it the two planners disagree.
    let threshold = full_cost / 8;
    assert!(threshold > 0);
    assert_eq!(
        plan_index_or_doc_values(Some(full_cost), threshold),
        IndexOrDocValuesPlan::Index,
        "at exactly cost/8 Java still runs the index"
    );
    assert_eq!(
        plan_index_or_doc_values(Some(full_cost), threshold - 1),
        IndexOrDocValuesPlan::DocValues,
        "below cost/8 Java verifies the lead clause's docs against doc values"
    );
    assert_eq!(
        plan_index_or_doc_values(None, threshold - 1),
        IndexOrDocValuesPlan::Index,
        "...and this is the one case the no-estimate default gets wrong"
    );

    // The narrower the query, the smaller its threshold, so a lead cost
    // that flips the full-range plan need not flip the narrow one -- the
    // estimate is doing real work, not scaling out of the comparison.
    assert!(narrow_cost / 8 < threshold);
    assert_eq!(
        plan_index_or_doc_values(Some(narrow_cost), threshold - 1),
        IndexOrDocValuesPlan::Index
    );
}
