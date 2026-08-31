//! Alternating A/B measurement of the **BKD tree walk's per-node allocations**
//! (ledger item 23b): the same tree, the same descent, run once reusing the
//! per-level scratch buffers and once reallocating them at every inner node,
//! which is exactly the shape `points.rs` had before `c43-final-cleanup`
//! (`InnerNode` owning `saved_split_tail` and `split_value`, plus a `to_vec()`
//! of each cell bound at both call sites -- four `Vec`s per inner node
//! visited, where Java's `BKDPointTree` preallocates per-level stacks).
//!
//! Both arms are the shipped `points.rs` code, selected by
//! `PointsReader::intersect_with_scratch`'s `reuse_scratch` argument, so they
//! run in one process from one build. There is no second binary and no
//! rebuild between arms -- which is the trap `c42-readpath-perf` found in
//! `scripts/bench-micro.sh`, where the container had been timing a months-old
//! binary. Criterion is not used: it reported 83/91/129 µs for identical code
//! on this machine (`docs/sweep/m2/c24-arith-codecs.md`), so every figure here
//! is a **min of N alternating repetitions**.
//!
//! Run: `cargo run --release -p lucene-codecs --example bkd_walk_scratch`
// Benchmark support code opts out of the arithmetic gate at the file
// boundary, as the fixture writers do. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::points::{self, IntersectVisitor, Relation, WritePointsField};
use std::hint::black_box;
use std::time::Instant;

const FIELD: i32 = 0;
const SEGMENT_ID: [u8; 16] = [7u8; 16];

/// `PointRangeQuery`'s visitor, counting rather than collecting so the
/// measurement is the walk and not a `Vec` push.
struct CountingRange {
    lower: Vec<u8>,
    upper: Vec<u8>,
    bytes_per_dim: usize,
    dims: usize,
    matched: u64,
}

impl CountingRange {
    fn dim<'v>(&self, v: &'v [u8], d: usize) -> &'v [u8] {
        &v[d * self.bytes_per_dim..(d + 1) * self.bytes_per_dim]
    }
}

impl IntersectVisitor for CountingRange {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let mut inside = true;
        for d in 0..self.dims {
            let (lo, hi) = (self.dim(&self.lower, d), self.dim(&self.upper, d));
            let (cmin, cmax) = (self.dim(min, d), self.dim(max, d));
            if cmax < lo || cmin > hi {
                return Relation::CellOutsideQuery;
            }
            if cmin < lo || cmax > hi {
                inside = false;
            }
        }
        if inside {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }

    fn visit(&mut self, _doc_id: i32) {
        self.matched += 1;
    }

    fn visit_with_value(&mut self, _doc_id: i32, value: &[u8]) {
        for d in 0..self.dims {
            if self.dim(value, d) < self.dim(&self.lower, d)
                || self.dim(value, d) > self.dim(&self.upper, d)
            {
                return;
            }
        }
        self.matched += 1;
    }
}

/// A deterministic pseudo-random 32-bit value, so a run is reproducible
/// without pulling in a dependency.
fn splitmix(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

fn build(num_points: usize, dims: usize, max_points_in_leaf: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut state = 0x5DEE_CE66u64;
    let mut points = Vec::with_capacity(num_points);
    for doc in 0..num_points {
        let mut packed = Vec::with_capacity(dims * 4);
        for _ in 0..dims {
            packed.extend_from_slice(&splitmix(&mut state).to_be_bytes());
        }
        points.push((doc as i32, packed));
    }
    points::write(
        &[WritePointsField {
            field_number: FIELD,
            num_dims: dims as i32,
            num_index_dims: dims as i32,
            bytes_per_dim: 4,
            points,
        }],
        max_points_in_leaf,
        &SEGMENT_ID,
        "",
    )
    .expect("write points")
}

fn range(dims: usize, lo: u32, hi: u32) -> (Vec<u8>, Vec<u8>) {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for _ in 0..dims {
        lower.extend_from_slice(&lo.to_be_bytes());
        upper.extend_from_slice(&hi.to_be_bytes());
    }
    (lower, upper)
}

fn visitor(dims: usize, lo: u32, hi: u32) -> CountingRange {
    let (lower, upper) = range(dims, lo, hi);
    CountingRange {
        lower,
        upper,
        bytes_per_dim: 4,
        dims,
        matched: 0,
    }
}

fn run(label: &str, num_points: usize, dims: usize, max_points_in_leaf: i32, reps: usize) {
    let (meta, index, data) = build(num_points, dims, max_points_in_leaf);
    let reader = points::open(&meta, &index, &data, &SEGMENT_ID, "").expect("open");
    // A band wide enough that most inner cells *cross* it, which is the walk
    // that reads every inner node rather than short-circuiting.
    let (lo, hi) = (u32::MAX / 8, u32::MAX / 8 * 5);

    let mut best_reuse = u128::MAX;
    let mut best_alloc = u128::MAX;
    let mut matched_reuse = 0u64;
    let mut matched_alloc = 0u64;
    for _ in 0..reps {
        let mut v = visitor(dims, lo, hi);
        let t = Instant::now();
        reader
            .intersect_with_scratch(FIELD, black_box(&mut v), true)
            .expect("intersect");
        best_reuse = best_reuse.min(t.elapsed().as_nanos());
        matched_reuse = v.matched;

        let mut v = visitor(dims, lo, hi);
        let t = Instant::now();
        reader
            .intersect_with_scratch(FIELD, black_box(&mut v), false)
            .expect("intersect");
        best_alloc = best_alloc.min(t.elapsed().as_nanos());
        matched_alloc = v.matched;
    }
    assert_eq!(
        matched_reuse, matched_alloc,
        "the two arms must visit the same documents"
    );

    // The estimate is the walk with every leaf read removed, so it is where
    // the allocations are the largest share of the work -- which is why c39's
    // review raised this against `estimate_point_count` specifically.
    let mut best_est_reuse = u128::MAX;
    let mut best_est_alloc = u128::MAX;
    for _ in 0..reps {
        let mut v = visitor(dims, lo, hi);
        let t = Instant::now();
        let a = reader
            .estimate_point_count_with_scratch(FIELD, black_box(&mut v), i64::MAX, true)
            .expect("estimate");
        best_est_reuse = best_est_reuse.min(t.elapsed().as_nanos());

        let mut v = visitor(dims, lo, hi);
        let t = Instant::now();
        let b = reader
            .estimate_point_count_with_scratch(FIELD, black_box(&mut v), i64::MAX, false)
            .expect("estimate");
        best_est_alloc = best_est_alloc.min(t.elapsed().as_nanos());
        assert_eq!(a, b, "the two arms must produce the same estimate");
    }

    println!(
        "{label:<34} intersect  reuse {:>9.1} us   realloc {:>9.1} us   {:.2}x   ({matched_reuse} hits)",
        best_reuse as f64 / 1000.0,
        best_alloc as f64 / 1000.0,
        best_alloc as f64 / best_reuse as f64,
    );
    println!(
        "{:<34} estimate   reuse {:>9.1} us   realloc {:>9.1} us   {:.2}x",
        "",
        best_est_reuse as f64 / 1000.0,
        best_est_alloc as f64 / 1000.0,
        best_est_alloc as f64 / best_est_reuse as f64,
    );
}

fn main() {
    let reps = 40;
    println!("min of {reps} alternating repetitions, both arms in one process\n");
    run("1D, 200k points, leaf 512", 200_000, 1, 512, reps);
    run("2D, 200k points, leaf 512", 200_000, 2, 512, reps);
    run("2D, 200k points, leaf 64", 200_000, 2, 64, reps);
    run("4D, 100k points, leaf 64", 100_000, 4, 64, reps);
}
