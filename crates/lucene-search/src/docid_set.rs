//! Minimal `DocIdSetIterator`-shaped merge combinators (`org.apache.lucene.search.
//! DocIdSetIterator`/`ConjunctionDISI`/`DisjunctionDISIApproximation`), pared down to
//! this slice's scope: given several already-materialized, ascending, duplicate-free
//! doc-ID sequences (one per clause), merge them into the AND (conjunction), OR
//! (disjunction), or AND-NOT (exclusion) result, still ascending and duplicate-free.
//!
//! **Why plain `Iterator<Item = i32>` instead of a bespoke `next_doc`/`advance` trait**:
//! Rust's `Iterator` already *is* a pull-based "give me the next doc ID or tell me
//! you're done" cursor — `Option<i32>` is `NO_MORE_DOCS` for free, `Peekable` gives the
//! one extra primitive (look-ahead without consuming) every merge algorithm below
//! needs, and every combinator composes with the rest of `std` (`.collect()`,
//! `for doc in iter`) with no adapter layer. Inventing a parallel `DocIdSet` trait with
//! its own `next_doc`/`advance` methods would just be re-deriving `Iterator` by hand —
//! transliterating Java's shape instead of using the idiomatic Rust one, which
//! `rust-performance` explicitly warns against.
//!
//! **Why `Box<dyn Iterator<Item = i32>>` instead of monomorphized generics**: `must`/
//! `should`/`must_not` are runtime-length `Vec<TermQuery>` (a `BooleanQuery` might have
//! two clauses or twenty), so the number and concrete type of per-clause iterators
//! being merged isn't known at compile time — there is no closed set of shapes to
//! monomorphize over the way a fixed 2-way merge would allow. `rust-performance`'s
//! "monomorphize per-doc loops" guidance is aimed at *scorers/DISIs* on the hot
//! single-query-type path; a boxed trait object at the clause-merge boundary (one
//! virtual call per doc per clause, not per byte) is the same tradeoff real Lucene's
//! own `Scorer`-hierarchy conjunctions make, and is the right place to pay it. This is
//! also explicitly a first cut ("correctness first, not final perf" per the task that
//! introduced it) — the merge algorithms below are the simple standard ones
//! (leapfrog conjunction, min-scan disjunction), not the skip-list-driven versions a
//! later performance pass would swap in.
//!
//! Every doc-ID sequence handed to these combinators is expected to already be
//! **ascending and duplicate-free** — [`crate::term_doc_ids`] (or any other producer)
//! is responsible for that invariant, same as real Lucene's `DocIdSetIterator` contract.

use std::iter::Peekable;

/// A boxed, type-erased doc-ID sequence — see the module doc for why this shape
/// (dynamic clause count) beats a monomorphized alternative here.
pub type BoxDocIter<'a> = Box<dyn Iterator<Item = i32> + 'a>;

/// AND across every wrapped clause: a doc is emitted only when **all** clauses agree
/// on it. Standard leapfrog: track the current maximum among all clauses' peeked
/// heads, fast-forward every clause whose head is behind that maximum, and repeat
/// until either all clauses agree (emit) or one is exhausted (done) — the same
/// algorithm as `ConjunctionDISI.doNext` minus its two-phase-iterator special case
/// (no `TwoPhaseIterator` exists in this port yet).
pub struct Conjunction<'a> {
    iters: Vec<Peekable<BoxDocIter<'a>>>,
}

impl<'a> Conjunction<'a> {
    /// Builds the conjunction over `iters`. An empty `iters` list matches nothing
    /// (mirrors this port's `search_boolean_query`, which never constructs a
    /// `Conjunction` with zero `must` clauses in the first place — a `BooleanQuery`
    /// with no `must`/`should` clauses at all is rejected before reaching here, see
    /// that function's doc comment) — included as a defined, tested edge case rather
    /// than a panic.
    pub fn new(iters: Vec<BoxDocIter<'a>>) -> Self {
        Self {
            iters: iters.into_iter().map(Iterator::peekable).collect(),
        }
    }
}

impl Iterator for Conjunction<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.iters.is_empty() {
            return None;
        }
        loop {
            let mut max = i32::MIN;
            for it in &mut self.iters {
                match it.peek() {
                    Some(&v) => max = max.max(v),
                    None => return None,
                }
            }

            let mut all_match = true;
            for it in &mut self.iters {
                while it.peek().is_some_and(|&v| v < max) {
                    it.next();
                }
                match it.peek() {
                    Some(&v) if v == max => {}
                    Some(_) => all_match = false,
                    None => return None,
                }
            }

            if all_match {
                for it in &mut self.iters {
                    it.next();
                }
                return Some(max);
            }
        }
    }
}

/// OR across every wrapped clause: a doc is emitted once if **any** clause matches
/// it, even if several clauses share it (dedup happens by construction: every
/// clause currently peeking the emitted minimum is advanced past it in the same
/// step, so no clause can re-offer that doc). Simple min-scan over all clauses'
/// peeked heads per step — `DisjunctionDISIApproximation`'s min-heap does the same
/// thing in `O(log n)` per step instead of this port's `O(n)`; fine for a first cut
/// per the module doc, revisit if clause counts get large.
pub struct Disjunction<'a> {
    iters: Vec<Peekable<BoxDocIter<'a>>>,
}

impl<'a> Disjunction<'a> {
    /// Builds the disjunction over `iters`. Like [`Conjunction::new`], an empty list
    /// is a defined "matches nothing" edge case, not a panic.
    pub fn new(iters: Vec<BoxDocIter<'a>>) -> Self {
        Self {
            iters: iters.into_iter().map(Iterator::peekable).collect(),
        }
    }
}

impl Iterator for Disjunction<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        let mut min: Option<i32> = None;
        for it in &mut self.iters {
            if let Some(&v) = it.peek() {
                if min.is_none_or(|m| v < m) {
                    min = Some(v);
                }
            }
        }
        let min = min?;
        for it in &mut self.iters {
            if it.peek() == Some(&min) {
                it.next();
            }
        }
        Some(min)
    }
}

/// AND-NOT: every doc from `base` that does **not** appear in `excluded` — the
/// `must_not` clause set's effect (`Occur.MUST_NOT`), applied as a final filter over
/// whatever `base` (a [`Conjunction`] or [`Disjunction`] of the query's `must`/
/// `should` clauses) already produced. Advances `excluded` in lockstep with `base`
/// rather than re-scanning from the start each time, since both sequences are
/// ascending.
pub struct Excluding<'a> {
    base: BoxDocIter<'a>,
    excluded: Peekable<BoxDocIter<'a>>,
}

impl<'a> Excluding<'a> {
    pub fn new(base: BoxDocIter<'a>, excluded: BoxDocIter<'a>) -> Self {
        Self {
            base,
            excluded: excluded.peekable(),
        }
    }
}

impl Iterator for Excluding<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        for doc in self.base.by_ref() {
            while self.excluded.peek().is_some_and(|&v| v < doc) {
                self.excluded.next();
            }
            if self.excluded.peek() != Some(&doc) {
                return Some(doc);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(v: Vec<i32>) -> BoxDocIter<'static> {
        Box::new(v.into_iter())
    }

    fn collect_conjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
        Conjunction::new(inputs.into_iter().map(boxed).collect()).collect()
    }

    fn collect_disjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
        Disjunction::new(inputs.into_iter().map(boxed).collect()).collect()
    }

    #[test]
    fn conjunction_no_overlap_matches_nothing() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2], vec![3, 4]]),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn conjunction_full_overlap_matches_every_doc() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn conjunction_partial_overlap_matches_only_shared_docs() {
        assert_eq!(
            collect_conjunction(vec![vec![0, 2, 5, 7], vec![0, 1, 5, 9]]),
            vec![0, 5]
        );
    }

    #[test]
    fn conjunction_three_way_partial_overlap() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![3, 4, 5, 6]]),
            vec![3, 4]
        );
    }

    #[test]
    fn conjunction_one_iterator_empty_matches_nothing() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3], vec![]]),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn conjunction_single_clause_passes_through() {
        assert_eq!(collect_conjunction(vec![vec![4, 5, 6]]), vec![4, 5, 6]);
    }

    #[test]
    fn conjunction_no_clauses_matches_nothing() {
        assert_eq!(collect_conjunction(vec![]), Vec::<i32>::new());
    }

    #[test]
    fn disjunction_no_overlap_merges_both_sorted() {
        assert_eq!(
            collect_disjunction(vec![vec![1, 3], vec![2, 4]]),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn disjunction_full_overlap_dedups() {
        assert_eq!(
            collect_disjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn disjunction_partial_overlap() {
        assert_eq!(
            collect_disjunction(vec![vec![0, 2, 4], vec![2, 3]]),
            vec![0, 2, 3, 4]
        );
    }

    #[test]
    fn disjunction_one_iterator_empty() {
        assert_eq!(collect_disjunction(vec![vec![1, 2], vec![]]), vec![1, 2]);
    }

    #[test]
    fn disjunction_no_clauses_matches_nothing() {
        assert_eq!(collect_disjunction(vec![]), Vec::<i32>::new());
    }

    #[test]
    fn excluding_removes_shared_docs() {
        let base = boxed(vec![0, 1, 2, 3, 4]);
        let excluded = boxed(vec![1, 3]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_with_no_exclusions_passes_base_through() {
        let base = boxed(vec![0, 2, 4]);
        let excluded = boxed(vec![]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_everything_matches_nothing() {
        let base = boxed(vec![0, 1, 2]);
        let excluded = boxed(vec![0, 1, 2]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn excluding_docs_not_in_base_have_no_effect() {
        let base = boxed(vec![0, 2, 4]);
        let excluded = boxed(vec![1, 3, 5, 6]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_empty_base_matches_nothing() {
        let base = boxed(vec![]);
        let excluded = boxed(vec![1, 2]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, Vec::<i32>::new());
    }
}
