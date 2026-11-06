//! Calibrating a field's search depth when it is written.
//!
//! A sample of the field's own vectors serves as queries. Each one's exact
//! `k` nearest neighbours among the other vectors come from a brute-force
//! scan; then:
//!
//! 1. **nprobe**: the fewest clusters, from 1 doubling up to all of them,
//!    whose clusters hold at least `(1 + target) / 2` of those neighbours.
//!    This is the recall a full rerank of the probed clusters would reach,
//!    kept halfway between the target and 1 so a short rerank can reach the
//!    target.
//! 2. **rerank**: a cap on the candidates per hit, from [`MULTIPLES`], for
//!    a rerank with early stop at that nprobe (as `Rerank::Bounded`, which
//!    `Rerank::Auto` searches with). The smallest cap reaching `target`
//!    recall over the sample (the largest when none does) is raised to the
//!    largest cap whose rerank work, after the early stop, stays within a
//!    quarter more: a generous cap costs little where the quantizer's error
//!    bounds are tight, and keeps the recall the early stop would find.
//!
//! Both are recorded relative to the field, nprobe as a fraction of its
//! clusters and rerank as a multiple of `k`, so they carry over to any `k`
//! and to [`crate::reader::Probes::Auto`] searches of segments of other
//! sizes. The sample, the scans and the choice are deterministic.

use rayon::prelude::*;

use crate::kmeans::{self, SplitMix64};
use crate::reader::{reach, Candidate, IvfField, Order, RangeSource, SearchStats};
use crate::Result;

/// A field's calibrated search depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// The share of the field's clusters to probe, in `(0, 1]`.
    pub probe_fraction: f32,
    /// The candidates to rerank per hit, at least 1.
    pub rerank_multiple: f32,
}

impl Calibration {
    /// Whether both values are in range.
    pub(crate) fn is_valid(&self) -> bool {
        self.probe_fraction > 0.0
            && self.probe_fraction <= 1.0
            && self.rerank_multiple >= 1.0
            && self.rerank_multiple.is_finite()
    }
}

/// The recall [`crate::IvfParams::calibration_recall`] targets by default.
pub const DEFAULT_CALIBRATION_RECALL: f32 = 0.95;

/// Sampled queries, fewer in a smaller field.
pub(crate) const QUERIES: usize = 64;

/// Hits per calibration query.
pub(crate) const K: usize = 10;

/// The rerank multiples tried, ascending.
pub const MULTIPLES: [f32; 16] = [
    1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
];

/// `ceil(multiple * k)` candidates.
fn depth(multiple: f32, k: usize) -> usize {
    (f64::from(multiple) * k as f64).ceil() as usize
}

/// One sampled query and its exact neighbours.
struct Truth {
    query: usize,
    neighbours: Vec<u32>,
}

/// The calibrated depth of `field`, written in memory with its postings in
/// `postings`, each slot's document being its vector's index in `vectors`
/// (as indexed: `Cosine` vectors unit-normalized). `clusters_of[i]` holds
/// vector `i`'s primary cluster and its spilled one (`u32::MAX` for none).
/// `None` for a field of fewer than two vectors.
// ARITH: indices are below the vector or cluster counts; counts are sums of
// at most `QUERIES * K` hits.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn calibrate(
    field: &IvfField,
    postings: &dyn RangeSource,
    vectors: &[f32],
    clusters_of: &[[u32; 2]],
    target: f32,
    seed: u64,
) -> Result<Option<Calibration>> {
    let n = field.count();
    let nlist = field.clusters();
    if n < 2 || nlist == 0 {
        return Ok(None);
    }
    let dim = field.dimension();
    let similarity = field.similarity();
    let k = K.min(n - 1);
    let queries = kmeans::sample_indices(n, QUERIES, &mut SplitMix64::new(seed));

    let truths: Vec<Truth> = queries
        .par_iter()
        .map(|&qi| {
            let q = kmeans::row(vectors, dim, qi);
            let mut scored: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != qi)
                .map(|j| (similarity.score(q, kmeans::row(vectors, dim, j)), j as u32))
                .collect();
            let by_score = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
            if k < scored.len() {
                scored.select_nth_unstable_by(k, by_score);
                scored.truncate(k);
            }
            Truth {
                query: qi,
                neighbours: scored.into_iter().map(|(_, j)| j).collect(),
            }
        })
        .collect();
    let total: usize = truths.iter().map(|t| t.neighbours.len()).sum();

    // nprobe: the first of 1, 2, 4, ..., nlist whose clusters hold enough
    // neighbours. All clusters hold every one.
    let needed = f64::from((1.0 + target) / 2.0) * total as f64;
    let mut nprobe = 1usize;
    loop {
        // Probed as searches probe (through the field's routing tree, if
        // any), not by exact centroid rank.
        let held: usize = truths
            .par_iter()
            .map(|t| {
                let mut probed = vec![false; nlist];
                for c in field.probe_order(kmeans::row(vectors, dim, t.query), nprobe) {
                    probed[c] = true;
                }
                t.neighbours
                    .iter()
                    .filter(|&&j| {
                        clusters_of[j as usize]
                            .iter()
                            .any(|&c| probed.get(c as usize).copied().unwrap_or(false))
                    })
                    .count()
            })
            .sum();
        if nprobe >= nlist || held as f64 >= needed {
            break;
        }
        nprobe = (nprobe * 2).min(nlist);
    }

    // rerank: per cap, the hits and the candidates reranked of a bounded
    // rerank (as `Rerank::Bounded`), summed over the queries.
    let longest = depth(MULTIPLES[MULTIPLES.len() - 1], k);
    let per_query: Vec<Vec<(usize, usize)>> = truths
        .par_iter()
        .map(|t| -> Result<Vec<(usize, usize)>> {
            let q = kmeans::row(vectors, dim, t.query);
            let probes = field.probe_order(q, nprobe);
            let own = t.query as i32;
            let accept = move |doc: i32| doc != own;
            let candidates = field.scan(
                postings,
                q,
                &probes,
                longest,
                Some(&accept),
                true,
                None,
                Order::Score,
                &mut SearchStats::default(),
            )?;
            let exact: Vec<f32> = candidates
                .iter()
                .map(|c| similarity.score(q, kmeans::row(vectors, dim, c.doc as usize)))
                .collect();
            Ok(MULTIPLES
                .iter()
                .map(|&m| {
                    let cap = depth(m, k).min(candidates.len());
                    bounded(&candidates[..cap], &exact, k, &t.neighbours)
                })
                .collect())
        })
        .collect::<Result<_>>()?;
    let sums: Vec<(usize, usize)> = (0..MULTIPLES.len())
        .map(|m| {
            per_query
                .iter()
                .fold((0, 0), |(h, w), q| (h + q[m].0, w + q[m].1))
        })
        .collect();
    let goal = f64::from(target) * total as f64;
    let reached = sums
        .iter()
        .position(|&(hits, _)| hits as f64 >= goal)
        .unwrap_or(MULTIPLES.len() - 1);
    // A deeper cap costs only what the early stop lets through: take the
    // deepest whose work stays within `GENEROUS` of the first that reaches
    // the target.
    let budget = sums[reached].1 as f64 * GENEROUS;
    let chosen = (reached..MULTIPLES.len())
        .take_while(|&m| sums[m].1 as f64 <= budget)
        .last()
        .unwrap_or(reached);
    Ok(Some(Calibration {
        probe_fraction: nprobe as f32 / nlist as f32,
        rerank_multiple: MULTIPLES[chosen],
    }))
}

/// How much more rerank work than the shallowest cap reaching the target a
/// deeper cap may cost and still be chosen.
const GENEROUS: f64 = 1.25;

/// The hits among `neighbours` and the candidates reranked of a bounded
/// rerank of `candidates` (best estimate first, `exact` their exact scores)
/// for `k` hits: in order, stopping once the `k`-th exact score beats every
/// remaining estimate plus margin, skipping a candidate whose own cannot.
// ARITH: indices are below `candidates.len()`.
#[allow(clippy::arithmetic_side_effects)]
fn bounded(
    candidates: &[Candidate],
    exact: &[f32],
    k: usize,
    neighbours: &[u32],
) -> (usize, usize) {
    let reach = reach(candidates);
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    let mut reranked = 0;
    for (i, c) in candidates.iter().enumerate() {
        if best.len() == k {
            let floor = best[k - 1].0;
            if reach[i] < floor {
                break;
            }
            if c.score + c.bound < floor {
                continue;
            }
        }
        reranked += 1;
        let s = exact[i];
        let pos = best.partition_point(|b| b.0 >= s);
        if pos < k {
            best.insert(pos, (s, c.doc as u32));
            best.truncate(k);
        }
    }
    let hits = best.iter().filter(|b| neighbours.contains(&b.1)).count();
    (hits, reranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validity() {
        let ok = Calibration {
            probe_fraction: 0.25,
            rerank_multiple: 3.0,
        };
        assert!(ok.is_valid());
        for bad in [
            Calibration {
                probe_fraction: 0.0,
                ..ok
            },
            Calibration {
                probe_fraction: 1.5,
                ..ok
            },
            Calibration {
                probe_fraction: f32::NAN,
                ..ok
            },
            Calibration {
                rerank_multiple: 0.5,
                ..ok
            },
            Calibration {
                rerank_multiple: f32::INFINITY,
                ..ok
            },
        ] {
            assert!(!bad.is_valid(), "{bad:?}");
        }
    }

    #[test]
    fn depths_round_up() {
        assert_eq!(depth(1.5, 3), 5);
        assert_eq!(depth(50.0, 10), 500);
    }
}
