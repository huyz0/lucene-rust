//! Settles b12 finding F-22: `BooleanScorer`'s 4,096-document window/bucket bulk
//! OR against this port's per-document min-scan disjunction
//! (`docid_set::Disjunction`) and, for `minimum_should_match > 1`, against the
//! whole-segment `HashMap<i32, usize>` tally that `should_match_counts` built.
//!
//! Shapes benchmarked, chosen to be the ones a real disjunction takes:
//!
//! - **`or4_dense`** -- four clauses over a 1,000,000-document segment, each
//!   matching roughly one document in ten. This is the benchmark corpus's
//!   `or t0 t1 t2 t3` shape, the query b12 measured at 0.26x of Java.
//! - **`or4_sparse`** -- the same four clauses at one match in a thousand, so
//!   most windows are empty. This is where a window-at-a-time scorer could
//!   plausibly *lose*, and the reason this port visits only non-empty windows
//!   and clears only the words it touched.
//! - **`or16_dense`** -- sixteen clauses, where the min-scan's `O(clauses)` per
//!   emitted document is at its worst and the window's per-clause contiguous run
//!   is at its best.
//! - **`or4_msm2`** -- four clauses with `minimum_should_match = 2`, against the
//!   `HashMap` tally.
//!
//! Run with `cargo bench -p lucene-search --bench boolean_bulk_or`.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lucene_search::docid_set::{BoxDocIter, Disjunction, WindowedDisjunction};

const MAX_DOC: i32 = 1_000_000;

/// `clauses` ascending doc-id lists over `0..MAX_DOC`, each matching about one
/// document in `sparsity`, with deliberately different strides so the clauses
/// overlap partially rather than being copies of each other.
fn clause_lists(clauses: usize, sparsity: i32) -> Vec<Vec<i32>> {
    (0..clauses)
        .map(|c| {
            let stride = sparsity + c as i32;
            let offset = (c as i32 * 7) % stride;
            (0..)
                .map(|i| offset + i * stride)
                .take_while(|&d| d < MAX_DOC)
                .collect()
        })
        .collect()
}

/// The pre-c6 path, reproduced verbatim so the two are measured against each
/// other: a min-scan across every clause per emitted document, plus (for
/// `min_should_match > 1`) a whole-segment `HashMap` tally.
fn min_scan(lists: &[Vec<i32>], min_should_match: usize) -> usize {
    let boxed: Vec<BoxDocIter<'static>> = lists
        .iter()
        .map(|v| Box::new(v.clone().into_iter()) as BoxDocIter<'static>)
        .collect();
    if min_should_match > 1 {
        let mut counts: HashMap<i32, usize> = HashMap::new();
        for docs in lists {
            for &doc in docs {
                *counts.entry(doc).or_insert(0) += 1;
            }
        }
        Disjunction::new(boxed)
            .filter(|d| counts.get(d).copied().unwrap_or(0) >= min_should_match)
            .count()
    } else {
        Disjunction::new(boxed).count()
    }
}

fn windowed(lists: &[Vec<i32>], min_should_match: usize) -> usize {
    WindowedDisjunction::new(lists.to_vec(), min_should_match).count()
}

fn bench(c: &mut Criterion) {
    let cases: Vec<(&str, Vec<Vec<i32>>, usize)> = vec![
        ("or4_dense", clause_lists(4, 10), 1),
        ("or4_sparse", clause_lists(4, 1_000), 1),
        ("or16_dense", clause_lists(16, 10), 1),
        ("or4_msm2", clause_lists(4, 10), 2),
    ];

    for (name, lists, msm) in cases {
        // Sanity: both must agree, or the comparison is meaningless.
        assert_eq!(min_scan(&lists, msm), windowed(&lists, msm), "{name}");

        let mut group = c.benchmark_group("boolean_bulk_or");
        group.bench_function(BenchmarkId::new("min_scan", name), |b| {
            b.iter(|| black_box(min_scan(black_box(&lists), msm)))
        });
        group.bench_function(BenchmarkId::new("windowed", name), |b| {
            b.iter(|| black_box(windowed(black_box(&lists), msm)))
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
