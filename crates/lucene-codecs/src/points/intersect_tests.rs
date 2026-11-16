#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn id() -> [u8; codec_util::ID_LENGTH] {
    [9u8; codec_util::ID_LENGTH]
}

/// `NumericUtils.longToSortableBytes`: flip the sign bit so unsigned
/// big-endian byte order matches signed numeric order.
fn long_sortable_bytes(v: i64) -> Vec<u8> {
    ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec()
}

/// Records which relations `intersect` saw, so a test can prove that
/// pruning really happened rather than just that the answer was right.
struct CountingRange {
    inner: RangeVisitor,
    cells_compared: usize,
    leaves_fully_inside: usize,
    points_examined: usize,
}

impl IntersectVisitor for CountingRange {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        self.cells_compared += 1;
        self.inner.compare(min_packed, max_packed)
    }
    fn visit(&mut self, doc_id: i32) {
        self.leaves_fully_inside += 1;
        self.inner.visit(doc_id);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.points_examined += 1;
        self.inner.visit_with_value(doc_id, packed_value);
    }
}

fn counting(field: &PointsField, lower: &[u8], upper: &[u8]) -> CountingRange {
    CountingRange {
        inner: RangeVisitor {
            matcher: RangeMatcher::new(field, lower, upper),
            docs: Vec::new(),
        },
        cells_compared: 0,
        leaves_fully_inside: 0,
        points_examined: 0,
    }
}

/// Brute-force reference: exactly what `decode_all_points` + an in-memory
/// filter would return, which is what `lucene-search`'s points query does
/// today. `intersect` must agree with it on every box.
fn brute_force(
    reader: &PointsReader<'_>,
    field_number: i32,
    lower: &[u8],
    upper: &[u8],
) -> Vec<i32> {
    let field = reader.field(field_number).unwrap();
    let bpd = field.bytes_per_dim as usize;
    reader
        .decode_all_points(field_number)
        .unwrap()
        .into_iter()
        .filter(|p| {
            (0..field.num_index_dims as usize).all(|dim| {
                let r = dim * bpd..(dim + 1) * bpd;
                p.packed_value[r.clone()] >= lower[r.clone()]
                    && p.packed_value[r.clone()] <= upper[r]
            })
        })
        .map(|p| p.doc_id)
        .collect()
}

/// The two arms of [`PointsReader::intersect_with_scratch`] are the same
/// walk with the per-level scratch either reused or reallocated at every
/// inner node (kept so
/// `examples/bkd_walk_scratch.rs` can time both in one process). They must
/// visit exactly the same documents, in the same order, and produce the
/// same estimate -- otherwise the measurement compares two algorithms
/// rather than two allocation strategies.
///
/// A multi-dimensional tree, because the buffers being reused are indexed
/// by split dimension: a 1D tree would restore the one dimension it has
/// even if the level indexing were wrong.
/// **The integer comparison orders exactly as the byte-wise one.** Every
/// 4- and 8-byte value against inclusive bounds, including the bytes
/// whose high bit is set (a signed load would put them first) and the
/// bounds themselves; and the shapes it does not cover fall back.
#[test]
fn a_fixed_width_range_agrees_with_the_byte_wise_comparison() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for width in [4usize, 8] {
        for _ in 0..200 {
            let mut a = next().to_be_bytes()[..width].to_vec();
            let mut b = next().to_be_bytes()[..width].to_vec();
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            let range = FixedWidthRange::new(1, width, &a, &b).unwrap();
            let mut probes: Vec<Vec<u8>> = (0..20)
                .map(|_| next().to_be_bytes()[..width].to_vec())
                .collect();
            probes.extend([a.clone(), b.clone(), vec![0x80; width], vec![0xff; width]]);
            for v in probes {
                let bytewise = a.as_slice() <= v.as_slice() && v.as_slice() <= b.as_slice();
                assert_eq!(range.contains(&v), Some(bytewise), "{a:x?} {v:x?} {b:x?}");
            }
        }
    }
    assert!(FixedWidthRange::new(2, 8, &[0; 16], &[0; 16]).is_none());
    assert!(FixedWidthRange::new(1, 16, &[0; 16], &[0; 16]).is_none());
    let r = FixedWidthRange::new(1, 8, &[0; 8], &[1; 8]).unwrap();
    assert_eq!(r.contains(&[0; 4]), None, "too short to say");
}

#[test]
fn the_two_scratch_arms_visit_exactly_the_same_documents() {
    let points: Vec<(i32, Vec<u8>)> = (0..400)
        .map(|i| {
            let mut packed = long_sortable_bytes((i as i64 * 7919) % 1301);
            packed.extend_from_slice(&long_sortable_bytes((i as i64 * 104_729) % 907));
            packed.extend_from_slice(&long_sortable_bytes(i as i64 % 13));
            (i, packed)
        })
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 3,
        num_index_dims: 3,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap().clone();

    let boxes = [
        (-2_000i64, 2_000i64),
        (0, 400),
        (100, 900),
        (-1, 1),
        (500, 500),
        (12, 1_300),
    ];
    let mut any_hits = false;
    for (lo, hi) in boxes {
        let lower: Vec<u8> = (0..3).flat_map(|_| long_sortable_bytes(lo)).collect();
        let upper: Vec<u8> = (0..3).flat_map(|_| long_sortable_bytes(hi)).collect();

        let mut reuse = counting(&meta, &lower, &upper);
        reader.intersect_with_scratch(0, &mut reuse, true).unwrap();
        let mut realloc = counting(&meta, &lower, &upper);
        reader
            .intersect_with_scratch(0, &mut realloc, false)
            .unwrap();
        assert_eq!(
            reuse.inner.docs, realloc.inner.docs,
            "reused and reallocated scratch disagree on [{lo}, {hi}]"
        );
        assert_eq!(reuse.cells_compared, realloc.cells_compared);
        assert_eq!(reuse.points_examined, realloc.points_examined);
        any_hits |= !reuse.inner.docs.is_empty();

        let mut v = counting(&meta, &lower, &upper);
        let a = reader
            .estimate_point_count_with_scratch(0, &mut v, i64::MAX, true)
            .unwrap();
        let mut v = counting(&meta, &lower, &upper);
        let b = reader
            .estimate_point_count_with_scratch(0, &mut v, i64::MAX, false)
            .unwrap();
        assert_eq!(a, b, "estimates disagree on [{lo}, {hi}]");
    }
    assert!(any_hits, "every box was empty -- the comparison is vacuous");
}

#[test]
fn intersect_1d_matches_brute_force_on_every_boundary_box() {
    // 173 points across 44 leaves (same shape as
    // `write_then_read_many_leaves_round_trips`), so the traversal has
    // several levels and an unbalanced deepest level.
    let points: Vec<(i32, Vec<u8>)> = (0..300)
        .filter(|i| i % 3 != 0)
        .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap().clone();

    // Boxes chosen to straddle every interesting case: empty below the
    // whole range, empty above it, exactly the whole range, a single
    // value, and several interior windows.
    let bounds = [
        (-5_000_000i64, -2_000_000i64),
        (10_000_000, 20_000_000),
        (-1_000_000, 1_400_000),
        (0, 0),
        (-1_000_000, -1_000_000),
        (-500_000, 500_000),
        (123_456, 987_654),
        (-1_000_000 + 7919, -1_000_000 + 7919 * 5),
    ];
    for (lo, hi) in bounds {
        let lower = long_sortable_bytes(lo);
        let upper = long_sortable_bytes(hi);
        let mut expected = brute_force(&reader, 0, &lower, &upper);
        expected.sort_unstable();
        let mut got = reader.range_query(0, &lower, &upper).unwrap();
        got.sort_unstable();
        assert_eq!(got, expected, "range [{lo}, {hi}]");
    }

    // Pruning really happens: a narrow interior window must not compare
    // (let alone decode) anything close to all 44 leaves' cells, and a
    // whole-range query must take the CELL_INSIDE_QUERY shortcut for
    // every point (no packed value decoded at all).
    let lower = long_sortable_bytes(-1_000_000 + 7919);
    let upper = long_sortable_bytes(-1_000_000 + 7919 * 5);
    let mut narrow = counting(&meta, &lower, &upper);
    reader.intersect(0, &mut narrow).unwrap();
    assert!(
        narrow.cells_compared < 2 * meta.num_leaves as usize,
        "narrow query compared {} cells for {} leaves -- no pruning?",
        narrow.cells_compared,
        meta.num_leaves
    );
    assert!(narrow.points_examined < 40, "{}", narrow.points_examined);

    let lower = long_sortable_bytes(i64::MIN);
    let upper = long_sortable_bytes(i64::MAX);
    let mut everything = counting(&meta, &lower, &upper);
    reader.intersect(0, &mut everything).unwrap();
    assert_eq!(everything.points_examined, 0);
    assert_eq!(everything.leaves_fully_inside, meta.point_count as usize);
    assert_eq!(everything.cells_compared, 1, "root alone should be INSIDE");
}

#[test]
fn intersect_2d_matches_brute_force() {
    // Multi-index-dimension: split dimension alternates, so the
    // per-dimension bound clamping and the split-value reconstruction's
    // per-dimension `last_split_values`/`negative_deltas` state both get
    // exercised (a single-dimension tree can't tell them apart).
    fn int_sortable_bytes(v: i32) -> [u8; 4] {
        ((v as u32) ^ 0x8000_0000).to_be_bytes()
    }
    let mut points: Vec<(i32, Vec<u8>)> = Vec::new();
    for i in 0..120i32 {
        let mut packed = Vec::new();
        packed.extend_from_slice(&int_sortable_bytes(i * 13 % 97 - 40));
        packed.extend_from_slice(&int_sortable_bytes(i * 31 % 71 - 30));
        points.push((i, packed));
    }
    let field = WritePointsField {
        field_number: 3,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 4,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 5, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();

    for (x0, x1, y0, y1) in [
        (-40i32, 60i32, -30i32, 45i32),
        (0, 10, 0, 10),
        (-100, 100, -100, 100),
        (1000, 2000, 0, 5),
        (-5, -5, -5, 40),
    ] {
        let mut lower = Vec::new();
        lower.extend_from_slice(&int_sortable_bytes(x0));
        lower.extend_from_slice(&int_sortable_bytes(y0));
        let mut upper = Vec::new();
        upper.extend_from_slice(&int_sortable_bytes(x1));
        upper.extend_from_slice(&int_sortable_bytes(y1));

        let mut expected = brute_force(&reader, 3, &lower, &upper);
        expected.sort_unstable();
        let mut got = reader.range_query(3, &lower, &upper).unwrap();
        got.sort_unstable();
        assert_eq!(got, expected, "box ({x0}..{x1}, {y0}..{y1})");
    }
}

#[test]
fn intersect_single_leaf_tree_still_works() {
    // `num_leaves == 1`: the root *is* a leaf, so `intersect_node` takes
    // the leaf branch on its very first call with no packed-index inner
    // node bytes to parse at all.
    let points: Vec<(i32, Vec<u8>)> = (0..5)
        .map(|i| (i, long_sortable_bytes(i as i64 * 10)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    assert_eq!(reader.field(0).unwrap().num_leaves, 1);

    let got = reader
        .range_query(0, &long_sortable_bytes(10), &long_sortable_bytes(30))
        .unwrap();
    assert_eq!(got, vec![1, 2, 3]);
    // Fully-outside box: the single leaf is never decoded.
    let got = reader
        .range_query(0, &long_sortable_bytes(1000), &long_sortable_bytes(2000))
        .unwrap();
    assert!(got.is_empty());
}

/// Exercises the packed index's `suffix == 0` branch: when many points
/// share a value, consecutive splits in the same dimension can produce a
/// split value byte-identical to the previous one, and
/// `recursePackIndex` then writes `prefix == bytesPerDim` with no suffix
/// bytes at all ("our split value is == last split value in this dim,
/// which can happen when there are many duplicate values", per
/// `BKDReader.readNodeData`'s own comment). The reconstruction must leave
/// the running split value untouched in that case.
#[test]
fn intersect_with_heavy_duplicate_values_matches_brute_force() {
    // 400 points over only 5 distinct values, 4 per leaf => 100 leaves,
    // so most splits land inside a run of equal values.
    let points: Vec<(i32, Vec<u8>)> = (0..400)
        .map(|i| (i, long_sortable_bytes((i % 5) as i64)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    assert_eq!(reader.field(0).unwrap().num_leaves, 100);

    for (lo, hi) in [(0i64, 4i64), (2, 2), (1, 3), (5, 10), (-1, 0)] {
        let lower = long_sortable_bytes(lo);
        let upper = long_sortable_bytes(hi);
        let mut expected: Vec<i32> = points
            .iter()
            .filter(|(_, v)| *v >= lower && *v <= upper)
            .map(|(d, _)| *d)
            .collect();
        expected.sort_unstable();
        let mut got = reader.range_query(0, &lower, &upper).unwrap();
        got.sort_unstable();
        assert_eq!(got, expected, "duplicate-heavy range [{lo}, {hi}]");
    }
}

/// A visitor taking the bulk callbacks (`visit_range`, `visit_bitset`,
/// `visit_run`) sees exactly the documents a per-document visitor sees,
/// and the bulk callbacks are really used on leaves of consecutive ids,
/// of bit-set ids and of runs of one value.
#[test]
fn bulk_visitor_callbacks_match_per_document_visits() {
    struct Bulk<'q> {
        lower: &'q [u8],
        upper: &'q [u8],
        docs: Vec<i32>,
        ranges: usize,
        bitsets: usize,
        runs: usize,
    }
    impl IntersectVisitor for Bulk<'_> {
        fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
            if max < self.lower || min > self.upper {
                Relation::CellOutsideQuery
            } else if min >= self.lower && max <= self.upper {
                Relation::CellInsideQuery
            } else {
                Relation::CellCrossesQuery
            }
        }
        fn visit(&mut self, doc_id: i32) {
            self.docs.push(doc_id);
        }
        fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
            if packed >= self.lower && packed <= self.upper {
                self.docs.push(doc_id);
            }
        }
        fn visit_run(&mut self, doc_ids: &[i32], packed: &[u8]) {
            self.runs += 1;
            if packed >= self.lower && packed <= self.upper {
                self.docs.extend_from_slice(doc_ids);
            }
        }
        fn visit_range(&mut self, start: i32, count: usize) -> bool {
            self.ranges += 1;
            self.docs.extend(start..start + count as i32);
            true
        }
        fn visit_bitset(&mut self, words: &[u64], base: i32, count: usize) -> bool {
            self.bitsets += 1;
            let before = self.docs.len();
            for (i, &w) in words.iter().enumerate() {
                for bit in 0..64 {
                    if w >> bit & 1 == 1 {
                        self.docs.push(base + (i * 64 + bit) as i32);
                    }
                }
            }
            assert_eq!(self.docs.len() - before, count);
            true
        }
    }
    // Field 0: doc i has value i (consecutive ids per leaf). Field 1:
    // doc 2i has value i (dense, bit-set ids). Field 2: 8 distinct
    // values over 2000 docs (runs of one value).
    let fields = [
        (0..2000)
            .map(|i| (i, long_sortable_bytes(i as i64)))
            .collect::<Vec<_>>(),
        (0..2000)
            .map(|i| (2 * i, long_sortable_bytes(i as i64)))
            .collect(),
        (0..2000)
            .map(|i| (i, long_sortable_bytes((i % 8) as i64)))
            .collect(),
    ];
    for (n, points) in fields.into_iter().enumerate() {
        let field = WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = write(&[field], 64, &id(), "").unwrap();
        let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        let (mut ranges, mut bitsets, mut runs) = (0, 0, 0);
        for (lo, hi) in [(0i64, 1999i64), (100, 1500), (3, 5), (2, 2)] {
            let lower = long_sortable_bytes(lo);
            let upper = long_sortable_bytes(hi);
            let mut expected = reader.range_query(0, &lower, &upper).unwrap();
            expected.sort_unstable();
            let mut v = Bulk {
                lower: &lower,
                upper: &upper,
                docs: Vec::new(),
                ranges: 0,
                bitsets: 0,
                runs: 0,
            };
            reader.intersect(0, &mut v).unwrap();
            v.docs.sort_unstable();
            assert_eq!(v.docs, expected, "field {n}, range [{lo}, {hi}]");
            ranges += v.ranges;
            bitsets += v.bitsets;
            runs += v.runs;
        }
        match n {
            0 => assert!(ranges > 0, "no consecutive-id leaf"),
            1 => assert!(bitsets > 0, "no bit-set leaf"),
            _ => assert!(runs > 0, "no run of one value"),
        }
    }
}

#[test]
fn intersect_unknown_field_rejected() {
    let points = vec![(0, long_sortable_bytes(1))];
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    assert!(matches!(
        reader.range_query(7, &long_sortable_bytes(0), &long_sortable_bytes(1)),
        Err(Error::IllegalFieldNumber(7))
    ));
}

// --- BKDWriter.merge's one-pass path ---

/// `n` single-dimension points with distinct 4-byte values, in ascending
/// value order -- the shape an already-merged 1-D points stream has.
fn sorted_1d_points(n: usize) -> Vec<(i32, Vec<u8>)> {
    (0..n)
        .map(|i| {
            let v = i as u32 * 37 + 11;
            (i as i32, v.to_be_bytes().to_vec())
        })
        .collect()
}

#[test]
fn presorted_plan_matches_the_general_plan_byte_for_byte() {
    // The whole safety argument for skipping the sort: for a
    // single-index-dimension field whose points are already sorted, the
    // one-pass plan and the general recursive plan must be the *same*
    // plan. Distinct values make the general path's stable sort produce
    // exactly the ascending order, whatever order it is handed.
    for n in [1usize, 2, 8, 9, 16, 17, 100, 1_000, 4_097] {
        let sorted = sorted_1d_points(n);
        // A deterministic shuffle, so the general (sorting) path runs.
        let mut shuffled = sorted.clone();
        let len = shuffled.len();
        for i in 0..len {
            let j = (i * 7919 + 13) % len;
            shuffled.swap(i, j);
        }

        let field = |points: Vec<(i32, Vec<u8>)>| WritePointsField {
            field_number: 3,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            points,
        };
        let a = write(&[field(sorted)], 16, &[5u8; codec_util::ID_LENGTH], "").unwrap();
        let b = write(&[field(shuffled)], 16, &[5u8; codec_util::ID_LENGTH], "").unwrap();
        assert_eq!(a.0, b.0, "n={n}: .kdm differs");
        assert_eq!(a.1, b.1, "n={n}: .kdi differs");
        assert_eq!(a.2, b.2, "n={n}: .kdd differs");
    }
}

#[test]
fn a_presorted_one_dimension_field_round_trips_through_the_reader() {
    const N: usize = 5_000;
    let points = sorted_1d_points(N);
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    let id = [6u8; codec_util::ID_LENGTH];
    let (kdm, kdi, kdd) = write(&[field], 64, &id, "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let got = reader.decode_all_points(0).unwrap();
    assert_eq!(got.len(), N);
    for (i, point) in got.iter().enumerate() {
        assert_eq!(point.doc_id, points[i].0, "point {i}");
        assert_eq!(point.packed_value, points[i].1, "point {i}");
    }
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.point_count, N as i64);
    assert_eq!(meta.min_packed_value, points[0].1);
    assert_eq!(meta.max_packed_value, points[N - 1].1);
}

#[test]
fn a_presorted_field_with_trailing_data_only_dimensions_still_takes_the_one_pass_path() {
    // `num_index_dims == 1 < num_dims`: the trailing dimension rides along
    // in every packed value but never splits, so sortedness of dimension 0
    // is all the one-pass plan needs. Same byte-for-byte equivalence test.
    let n = 300usize;
    let sorted: Vec<(i32, Vec<u8>)> = (0..n)
        .map(|i| {
            let mut v = (i as u16).to_be_bytes().to_vec();
            v.extend_from_slice(&((n - i) as u16).to_be_bytes());
            (i as i32, v)
        })
        .collect();
    let mut shuffled = sorted.clone();
    for i in 0..n {
        let j = (i * 131 + 5) % n;
        shuffled.swap(i, j);
    }
    let field = |points: Vec<(i32, Vec<u8>)>| WritePointsField {
        field_number: 1,
        num_dims: 2,
        num_index_dims: 1,
        bytes_per_dim: 2,
        points,
    };
    let id = [4u8; codec_util::ID_LENGTH];
    let a = write(&[field(sorted)], 32, &id, "").unwrap();
    let b = write(&[field(shuffled)], 32, &id, "").unwrap();
    assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2));
}

#[test]
fn points_with_equal_values_keep_their_input_order_on_both_paths() {
    // Ties are where a stable sort and a plain chunking could diverge:
    // the one-pass path preserves input order by construction, and the
    // general path's `sort_by` is stable, so they agree.
    let points: Vec<(i32, Vec<u8>)> = (0..64)
        .map(|i| (i, ((i / 8) as u32).to_be_bytes().to_vec()))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    let id = [3u8; codec_util::ID_LENGTH];
    let (kdm, kdi, kdd) = write(&[field], 8, &id, "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let got = reader.decode_all_points(0).unwrap();
    let doc_ids: Vec<i32> = got.iter().map(|p| p.doc_id).collect();
    assert_eq!(doc_ids, (0..64).collect::<Vec<i32>>());
}

// ------------------------------------------------------------------
// Arithmetic-gate controls: the packed-index (`.kdi`) walk. These build a
// `PointsReader` over a hand-written packed index, which is the only way
// to reach the pruning traversal with values no writer emits.
// ------------------------------------------------------------------

fn hand_built_field(num_leaves: i32, num_index_bytes: i32) -> PointsField {
    PointsField {
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 1,
        max_points_in_leaf_node: 512,
        num_leaves,
        min_packed_value: vec![0x00],
        max_packed_value: vec![0xFF],
        point_count: 2,
        doc_count: 2,
        index_start_pointer: 0,
        num_index_bytes,
        bkd_version: BKD_VERSION_CURRENT,
    }
}

fn negative_vint(out: &mut Vec<u8>, v: i32) {
    let mut u = v as u32;
    loop {
        let b = (u & 0x7F) as u8;
        u >>= 7;
        if u != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}

/// A negative split-descriptor vint. Java's `code % numIndexDims` yields a
/// negative `splitDim` that indexes `splitValuesStack` out of bounds one
/// line later; here the same value cast to `usize` indexed
/// `negative_deltas` with an astronomic offset -- a panic, not a decode
/// error. The `0x10..=0x20` query box forces `CELL_CROSSES_QUERY` at the
/// root so the pruning path (not `addAll`) reads the descriptor.
#[test]
fn a_negative_split_descriptor_is_a_decode_error() {
    let mut kdi = vec![0u8]; // root FP-delta vlong = 0
    negative_vint(&mut kdi, -1);
    kdi.extend_from_slice(&[0u8; 16]);
    let field = hand_built_field(2, kdi.len() as i32);
    let kdd = vec![0u8; 64];
    let reader = PointsReader {
        kdi: &kdi,
        kdd: &kdd,
        fields: std::borrow::Cow::Owned(vec![(0, field)]),
    };
    let err = reader.range_query(0, &[0x10], &[0x20]).unwrap_err();
    assert!(
        format!("{err}").contains("negative BKD split descriptor"),
        "{err}"
    );
}

/// `leftNumBytes` is the skip-ahead hint the walk adds to the current
/// position to find the right sibling. A negative one became a huge
/// `usize` and overflowed the sum -- in a release build, a wrap to a
/// position inside the packed index, i.e. the walk silently resuming on
/// the wrong node.
#[test]
fn a_negative_left_num_bytes_is_a_decode_error() {
    // numLeaves = 3 so the root's left child is itself an inner node and
    // the `leftNumBytes` vint is present.
    let mut kdi = vec![0u8, 0u8]; // root FP delta = 0, split descriptor = 0
    negative_vint(&mut kdi, -1); // leftNumBytes
    kdi.extend_from_slice(&[0u8; 16]);
    let field = hand_built_field(3, kdi.len() as i32);
    let kdd = vec![0u8; 64];
    let reader = PointsReader {
        kdi: &kdi,
        kdd: &kdd,
        fields: std::borrow::Cow::Owned(vec![(0, field)]),
    };
    let err = reader.range_query(0, &[0x10], &[0x20]).unwrap_err();
    assert!(format!("{err}").contains("leftNumBytes"), "{err}");
}

/// A leaf's file pointer is its parent's baseline plus a `.kdi` delta, and
/// Java lets the `long` add wrap before failing at `seek`. The left child
/// here is entirely outside the query, so the walk reaches the right
/// child's delta without touching `.kdd` first.
#[test]
fn a_leaf_pointer_that_overflows_is_a_decode_error() {
    let mut kdi = Vec::new();
    write_vlong_test(&mut kdi, i64::MAX); // root FP baseline
    kdi.push(0); // split descriptor: splitDim 0, prefix 0, delta 0
    write_vlong_test(&mut kdi, 1); // right child's FP delta
    let field = hand_built_field(2, kdi.len() as i32);
    let kdd = vec![0u8; 64];
    let reader = PointsReader {
        kdi: &kdi,
        kdd: &kdd,
        fields: std::borrow::Cow::Owned(vec![(0, field)]),
    };
    // The split value decodes to 0x00, so the left cell is [0x00, 0x00],
    // entirely below the query box.
    let err = reader.range_query(0, &[0x80], &[0xFF]).unwrap_err();
    assert!(
        format!("{err}").contains("leaf block pointer overflows"),
        "{err}"
    );
}

fn write_vlong_test(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7F) as u8;
        v = ((v as u64) >> 7) as i64;
        if v != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}

/// `RangeVisitor` slices the caller's box per index dimension against the
/// *field's* shape, so a box of the wrong width indexed out of bounds
/// mid-traversal. Java's `PointRangeQuery` constructor checks the same
/// thing up front.
#[test]
fn range_query_rejects_bounds_of_the_wrong_width() {
    let points: Vec<(i32, Vec<u8>)> = (0..8)
        .map(|i| (i, long_sortable_bytes(i as i64 * 10)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    match reader.range_query(0, &[0u8; 4], &long_sortable_bytes(50)) {
        Err(Error::InvalidConfig(msg)) => assert!(msg.contains("8 bytes"), "{msg}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // The correctly-sized box still works.
    assert!(!reader
        .range_query(0, &long_sortable_bytes(0), &long_sortable_bytes(70))
        .unwrap()
        .is_empty());
}

/// The same 173-point / 44-leaf index
/// `intersect_1d_matches_brute_force_on_every_boundary_box` builds: deep
/// enough that the estimate walk has several levels, and unbalanced at the
/// deepest one.
fn single_dim_index() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; codec_util::ID_LENGTH]) {
    let points: Vec<(i32, Vec<u8>)> = (0..300)
        .filter(|i| i % 3 != 0)
        .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    (kdm, kdi, kdd, id())
}

/// A `PointsField` with nothing but the four counts `subtree_size` reads.
/// Built by hand because the formula is a function of those counts alone
/// -- no `.kdi`/`.kdd` is involved, which is the property that makes
/// `estimate_point_count` cheap in the first place.
fn counts_only_field(
    num_leaves: i32,
    max_points_in_leaf_node: i32,
    point_count: i64,
) -> PointsField {
    PointsField {
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        max_points_in_leaf_node,
        num_leaves,
        min_packed_value: vec![0u8; 8],
        max_packed_value: vec![0xFFu8; 8],
        point_count,
        doc_count: 1,
        index_start_pointer: 0,
        num_index_bytes: 0,
        bkd_version: BKD_VERSION_CURRENT,
    }
}

/// `BKDPointTree.size()` over the shape the committed `points_index`
/// fixture has for its `val` field: 1 333 points, 512 per leaf, so three
/// leaves -- the unbalanced case where the left subtree is one level
/// deeper than the right, and where the last leaf holds 309 points rather
/// than 512.
///
/// Hand-derived from Java's own arithmetic rather than round-tripped, so a
/// self-consistent-but-wrong formula cannot pass: `numLeaves = 3` gives
/// `treeDepth = log2(3) + 2 = 3` and `rightMostLeafNode = (1 << 2) - 1 = 3`.
#[test]
fn subtree_size_matches_javas_unbalanced_formula() {
    let field = counts_only_field(3, 512, 1333);
    // Root: every leaf, so the whole point count.
    assert_eq!(field.subtree_size(1).unwrap(), 1333);
    // Node 2's subtree is leaves 4 and 5 (it is above the leaf offset), a
    // full 512 each, and its rightmost leaf is not the tree's.
    assert_eq!(field.subtree_size(2).unwrap(), 1024);
    // Node 3 *is* a leaf (3 >= numLeaves), and it is the tree's rightmost,
    // so it holds `pointCount % 512 == 309`.
    assert_eq!(field.subtree_size(3).unwrap(), 309);
    assert_eq!(field.subtree_size(4).unwrap(), 512);
    assert_eq!(field.subtree_size(5).unwrap(), 512);
    // The three leaves partition the field.
    assert_eq!(
        field.subtree_size(4).unwrap()
            + field.subtree_size(5).unwrap()
            + field.subtree_size(3).unwrap(),
        1333
    );
}

/// A point count that is an exact multiple of the leaf size takes Java's
/// `lastLeafNodePointCount == 0 ? maxPointsInLeafNode` promotion -- a full
/// last leaf, not an empty one.
#[test]
fn subtree_size_treats_an_exact_multiple_as_a_full_last_leaf() {
    let field = counts_only_field(2, 512, 1024);
    assert_eq!(field.subtree_size(1).unwrap(), 1024);
    assert_eq!(field.subtree_size(2).unwrap(), 512);
    assert_eq!(field.subtree_size(3).unwrap(), 512);
}

/// One leaf: Java forces `isTreeBalanced` false for this case, and the
/// unbalanced formula answers the whole point count.
#[test]
fn subtree_size_handles_a_single_leaf_tree() {
    let field = counts_only_field(1, 512, 7);
    assert_eq!(field.subtree_size(1).unwrap(), 7);
}

/// The largest `numLeaves`/`maxPointsInLeafNode` a `.kdm` can encode still
/// answers a non-negative size. Java wraps here and hands the query
/// planner a negative cost.
#[test]
fn subtree_size_saturates_rather_than_wrapping_negative() {
    let field = counts_only_field(i32::MAX, i32::MAX, 1);
    assert!(field.subtree_size(1).unwrap() > 0);
    assert!(field.subtree_size(2).unwrap() > 0);
}

/// The two estimate entry points reject an unknown field the same way
/// `intersect`/`range_query` do, rather than answering zero.
#[test]
fn estimate_point_count_rejects_an_unknown_field() {
    let (kdm, kdi, kdd, id) = single_dim_index();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    assert!(matches!(
        reader.estimate_range_point_count(99, &long_sortable_bytes(0), &long_sortable_bytes(1)),
        Err(Error::IllegalFieldNumber(99))
    ));
    assert!(matches!(
        reader.estimate_point_count_bounded(99, &mut CountingVisitor::default(), i64::MAX),
        Err(Error::IllegalFieldNumber(99))
    ));
}

/// A bounds box of the wrong width is rejected before the walk starts, as
/// it is for `range_query`.
#[test]
fn estimate_range_point_count_rejects_mis_sized_bounds() {
    let (kdm, kdi, kdd, id) = single_dim_index();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let err = reader
        .estimate_range_point_count(0, &[0u8; 4], &[0xFFu8; 4])
        .unwrap_err();
    assert!(
        format!("{err}").contains("range query bounds must be 8 bytes"),
        "unexpected error: {err}"
    );
}

/// `count_points` is exact: the inside cells' sizes plus what the
/// visitor accepts of the crossing leaves is the number of points
/// `range_query` finds, for ranges inside one leaf, across many, over
/// everything and over nothing; an unknown field is an error.
#[test]
fn count_points_counts_what_range_query_finds() {
    let (kdm, kdi, kdd, id) = single_dim_index();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field = reader.field(0).unwrap();
    let bounds = [
        i64::MIN,
        -1_000_001,
        -1_000_000,
        -992_081,
        -500_000,
        0,
        7_919,
        1_000_000,
        1_367_000,
        i64::MAX,
    ];
    let (mut saw_inside, mut saw_crossing) = (false, false);
    for &lo in &bounds {
        for &hi in bounds.iter().filter(|&&h| h >= lo) {
            let (lower, upper) = (long_sortable_bytes(lo), long_sortable_bytes(hi));
            let want = reader.range_query(0, &lower, &upper).unwrap().len() as i64;
            let mut visitor = reader.range_visitor(field, &lower, &upper).unwrap();
            let inside = reader.count_points(0, &mut visitor).unwrap();
            let crossing = visitor.docs.len() as i64;
            assert_eq!(inside + crossing, want, "[{lo}, {hi}]");
            saw_inside |= inside > 0;
            saw_crossing |= crossing > 0;
        }
    }
    assert!(saw_inside && saw_crossing, "both kinds of cell counted");
    assert!(matches!(
        reader.count_points(99, &mut CountingVisitor::default()),
        Err(Error::IllegalFieldNumber(99))
    ));
}

/// `estimate_point_count` never visits a document and never decodes a
/// point -- Java's estimate uses only `compare`. The visitor here panics
/// if either happens, so this is a shape assertion, not a count.
#[derive(Default)]
struct CountingVisitor {
    compares: usize,
}

impl IntersectVisitor for CountingVisitor {
    fn compare(&mut self, _min_packed: &[u8], _max_packed: &[u8]) -> Relation {
        self.compares += 1;
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        panic!("estimate_point_count must not visit documents");
    }
    fn visit_with_value(&mut self, _doc_id: i32, _packed_value: &[u8]) {
        panic!("estimate_point_count must not decode points");
    }
}

/// An always-`CELL_CROSSES_QUERY` visitor drives the walk to every leaf
/// and gets Java's "assume half the points matched" at each -- and touches
/// no `.kdd` byte on the way, which is the whole point of the estimate.
#[test]
fn a_crossing_estimate_halves_every_leaf_and_decodes_nothing() {
    let (kdm, kdi, kdd, id) = single_dim_index();
    let reader = open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let field = reader.field(0).unwrap();
    let mut visitor = CountingVisitor::default();
    let got = reader.estimate_point_count(0, &mut visitor).unwrap();
    let mut want = 0i64;
    for leaf in 0..field.num_leaves {
        want += (field.subtree_size(field.num_leaves + leaf).unwrap() + 1) / 2;
    }
    assert_eq!(got, want);
    assert!(visitor.compares > 0);
}
