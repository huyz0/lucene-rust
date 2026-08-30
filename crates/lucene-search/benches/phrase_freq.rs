//! Settles b12 finding F-21: `phrase_freq_exact` binary-searches each
//! subsequent term's position list for `p0 + i`, where
//! `ExactPhraseMatcher.nextMatch` advances all lists together in one merge
//! pass.
//!
//! The two are `O(|p0| · (n-1) · log|list|)` against `O(sum of list lengths)`,
//! but that is not the interesting number: this project has twice measured a
//! binary search losing to a linear scan for reasons that have nothing to do
//! with the exponents (`LazyDocsCursor::advance` and `next_doc`, M1.6 findings
//! #O5 and "advance binary-searched where Lucene scans linearly"), because the
//! branch in a binary search is unpredictable and the one in a forward scan is
//! not.
//!
//! Because `p0` increases monotonically along the first term's list, every
//! target `p0 + i` does too, so a cursor per subsequent term never rewinds --
//! that is what makes the merge form legal here, and it is exactly Lucene's
//! shape.
//!
//! Shapes benchmarked: a short list (a normal document's occurrences of a
//! word), a long one (a stopword-frequency term in a long document), a
//! high-hit-rate case (the phrase matches almost everywhere) and a
//! low-hit-rate one (the terms co-occur but never adjacently), plus a
//! three-term phrase. Run with `cargo bench -p lucene-search --bench
//! phrase_freq`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// The implementation this crate shipped before F-21: iterate the first term's
/// positions, binary-search every other list.
fn freq_binary_search(term_positions: &[&[i32]]) -> i32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0;
    }
    let mut freq = 0;
    'candidate: for &p0 in first.iter() {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            if positions.binary_search(&target).is_err() {
                continue 'candidate;
            }
        }
        freq += 1;
    }
    freq
}

/// `ExactPhraseMatcher`'s shape: one forward cursor per subsequent term,
/// advanced (never rewound) as `p0` walks forward.
fn freq_leapfrog(term_positions: &[&[i32]]) -> i32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0;
    }
    let mut cursors = vec![0usize; rest.len()];
    let mut freq = 0;
    'candidate: for &p0 in first.iter() {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            let mut c = cursors[i];
            while c < positions.len() && positions[c] < target {
                c += 1;
            }
            cursors[i] = c;
            if c == positions.len() {
                // Exhausted: no later `p0` can match this term either.
                break 'candidate;
            }
            if positions[c] != target {
                continue 'candidate;
            }
        }
        freq += 1;
    }
    freq
}

/// `(name, per-term position lists)`.
fn shapes() -> Vec<(&'static str, Vec<Vec<i32>>)> {
    // A short, ordinary document: a word occurring a handful of times.
    let short_a: Vec<i32> = (0..8).map(|i| i * 37).collect();
    let short_b: Vec<i32> = short_a.iter().map(|p| p + 1).collect();

    // A long document with a frequent term: 4096 occurrences.
    let long_a: Vec<i32> = (0..4096).map(|i| i * 3).collect();
    // Half of them are followed by the second term.
    let long_b_half: Vec<i32> = (0..4096)
        .filter(|i| i % 2 == 0)
        .map(|i| i * 3 + 1)
        .collect();
    // None of them are (the terms co-occur but never adjacently).
    let long_b_none: Vec<i32> = (0..4096).map(|i| i * 3 + 2).collect();
    // All of them are.
    let long_b_all: Vec<i32> = (0..4096).map(|i| i * 3 + 1).collect();

    vec![
        ("short-2term", vec![short_a, short_b]),
        (
            "long-2term-half-hit",
            vec![long_a.clone(), long_b_half.clone()],
        ),
        ("long-2term-no-hit", vec![long_a.clone(), long_b_none]),
        ("long-2term-all-hit", vec![long_a.clone(), long_b_all]),
        (
            "long-3term-half-hit",
            vec![
                long_a.clone(),
                long_b_half,
                (0..4096)
                    .filter(|i| i % 2 == 0)
                    .map(|i| i * 3 + 2)
                    .collect(),
            ],
        ),
    ]
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("phrase_freq_exact");
    for (name, lists) in shapes() {
        let refs: Vec<&[i32]> = lists.iter().map(|v| v.as_slice()).collect();
        // Both implementations must agree, or the comparison is meaningless.
        assert_eq!(
            freq_binary_search(&refs),
            freq_leapfrog(&refs),
            "implementations disagree on {name}"
        );
        group.bench_with_input(BenchmarkId::new("binary_search", name), &refs, |b, refs| {
            b.iter(|| black_box(freq_binary_search(black_box(refs))))
        });
        group.bench_with_input(BenchmarkId::new("leapfrog", name), &refs, |b, refs| {
            b.iter(|| black_box(freq_leapfrog(black_box(refs))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
