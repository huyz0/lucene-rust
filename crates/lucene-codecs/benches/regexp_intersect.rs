//! Microbenchmark for the dead-prefix skip in
//! `lucene_codecs::regexp::RegexpPattern::dead_prefix_len`, the capability
//! that lets `blocktree::FieldTerms::regexp_intersect` jump past whole runs of
//! the sorted term array instead of testing every term -- the sorted-array
//! analogue of what `ByteRunAutomaton`'s dead state does for real Lucene's
//! `IntersectTermsEnum` (see `docs/sweep/m2/b8-automata-analysis.md`).
//!
//! The term dictionary is the shape the search benchmark's corpus has:
//! `t0`..`t999999`, sorted lexicographically, so the interesting patterns are
//! the interior-constrained ones (`t1[0-9]`, `t1*z`) that the b4 sweep
//! measured at 34-38x over-scan.
//!
//! Each pattern is run twice: `scan` tests every term in the literal-prefix
//! range (what this port did before), `skip` uses the dead-prefix jump plus the
//! adaptive give-up `RegexpIntersect` applies.
//!
//! Run with: `cargo bench -p lucene-codecs --bench regexp_intersect`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::regexp::RegexpPattern;

/// `blocktree::prefix_upper_bound`, duplicated here because it is private.
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(&last) = upper.last() {
        if last == 0xFF {
            upper.pop();
        } else {
            *upper.last_mut().expect("non-empty") += 1;
            return Some(upper);
        }
    }
    None
}

/// `TermIndex::lower_bound_from`'s galloping search.
fn lower_bound_from(terms: &[&[u8]], from: usize, end: usize, target: &[u8]) -> usize {
    let mut lo = from;
    let mut step = 1;
    loop {
        let hi = (lo + step).min(end);
        if hi >= end || terms[hi] >= target {
            return lo + terms[lo..hi].partition_point(|t| *t < target);
        }
        lo = hi;
        step *= 2;
    }
}

fn scan(terms: &[&[u8]], start: usize, end: usize, pattern: &RegexpPattern) -> usize {
    (start..end).filter(|&i| pattern.matches(terms[i])).count()
}

/// `RegexpIntersect::next`'s loop, including its adaptive give-up.
fn skip(terms: &[&[u8]], start: usize, end: usize, pattern: &RegexpPattern) -> usize {
    const WARMUP: u32 = 128;
    const MIN_SAVING: u64 = 16;
    let (mut hits, mut i) = (0usize, start);
    let (mut attempts, mut skipped, mut enabled) = (0u32, 0u64, true);
    while i < end {
        if pattern.matches(terms[i]) {
            hits += 1;
            i += 1;
            continue;
        }
        if !enabled {
            i += 1;
            continue;
        }
        let next = match pattern
            .dead_prefix_len(terms[i])
            .and_then(|k| prefix_upper_bound(&terms[i][..k]))
        {
            Some(upper) => lower_bound_from(terms, i, end, &upper).max(i + 1),
            None => i + 1,
        };
        attempts += 1;
        skipped += (next - i - 1) as u64;
        if attempts == WARMUP && skipped < u64::from(attempts) * MIN_SAVING {
            enabled = false;
        }
        i = next;
    }
    hits
}

fn bench_regexp_intersect(c: &mut Criterion) {
    let mut owned: Vec<String> = (0..1_000_000u32).map(|i| format!("t{i}")).collect();
    owned.sort_unstable();
    let terms: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();

    // Two patterns that the skip wins on, and two it must not lose on.
    for src in ["t1[0-9]", "t1*z", "t[0-9]{4}", "t.*99"] {
        let pattern = RegexpPattern::new(src.as_bytes()).expect("valid pattern");
        let prefix = pattern.literal_prefix();
        let start = terms.partition_point(|t| *t < prefix.as_slice());
        let end = match prefix_upper_bound(&prefix) {
            Some(upper) => terms.partition_point(|t| *t < upper.as_slice()),
            None => terms.len(),
        };
        assert_eq!(
            scan(&terms, start, end, &pattern),
            skip(&terms, start, end, &pattern),
            "{src}: the skip changed the match count"
        );

        let mut group = c.benchmark_group(format!("regexp_intersect/{src}"));
        group.sample_size(10);
        group.bench_function("scan", |b| {
            b.iter(|| black_box(scan(&terms, start, end, &pattern)))
        });
        group.bench_function("skip", |b| {
            b.iter(|| black_box(skip(&terms, start, end, &pattern)))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_regexp_intersect);
criterion_main!(benches);
