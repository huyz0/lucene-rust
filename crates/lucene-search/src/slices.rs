//! Running a concurrent segment search's slices.
//!
//! Lucene hands each slice to its executor and waits for them all. Here the
//! first slice runs on the calling thread while rayon's pool takes the rest:
//! the handoff to a pool thread (a wake-up, for a pool that has gone to sleep
//! between requests) then overlaps work rather than preceding it, which is
//! what a slice doing little -- a rare term, a small segment -- would
//! otherwise pay for in full.

/// Below this many estimated matches ([`estimated_matches`]) a search's slices
/// run one after another on the calling thread: the work is smaller than the
/// wake-up of a pool thread asleep between requests, which rayon's scope waits
/// for even when the caller could have run the slice itself. Lucene's
/// `TaskExecutor` has the caller run any task not yet started; rayon cannot
/// take a spawned job back, so the choice is made before spawning.
pub(crate) const SEQUENTIAL_BELOW: u64 = 32_768;

/// [`run_slices`], or the slices in turn on this thread when `parallel` is
/// false -- the same results in the same order.
pub(crate) fn run_slices_if<T: Send>(
    parallel: bool,
    slices: &[Vec<usize>],
    f: impl Fn(&[usize]) -> T + Sync,
) -> Vec<T> {
    if parallel {
        run_slices(slices, f)
    } else {
        slices.iter().map(|s| f(s)).collect()
    }
}

/// An upper bound on how many documents `query` matches over `segments`,
/// from the term dictionaries alone: a term's `docFreq`, a conjunction's
/// smallest required clause, a disjunction's clauses summed, and every other
/// clause as all of each segment's documents. Only ever used to decide
/// whether to hand slices to other threads.
pub(crate) fn estimated_matches(
    segments: &[crate::multi_segment::OpenSegment<'_>],
    query: &crate::query::BooleanQuery,
) -> u64 {
    use crate::query::Clause;
    fn all(segments: &[crate::multi_segment::OpenSegment<'_>]) -> u64 {
        segments
            .iter()
            .map(|s| u64::try_from(s.max_doc.unwrap_or(i32::MAX)).unwrap_or(0))
            .fold(0, u64::saturating_add)
    }
    fn clause(segments: &[crate::multi_segment::OpenSegment<'_>], c: &Clause) -> u64 {
        match c {
            Clause::Term(t) => segments
                .iter()
                .map(|s| {
                    s.fields
                        .field(&t.field)
                        .and_then(|f| f.try_seek_exact(&t.term).ok().flatten())
                        .map_or(0, |st| u64::try_from(st.doc_freq).unwrap_or(0))
                })
                .fold(0, u64::saturating_add),
            Clause::MatchNoDocs(_) => 0,
            Clause::Boolean(b) => boolean(segments, b),
            Clause::ConstantScore(c) => clause(segments, &c.inner),
            Clause::Boost(b) => clause(segments, &b.inner),
            _ => all(segments),
        }
    }
    fn boolean(
        segments: &[crate::multi_segment::OpenSegment<'_>],
        b: &crate::query::BooleanQuery,
    ) -> u64 {
        let required = b.must.iter().chain(&b.filter);
        match required.map(|c| clause(segments, c)).min() {
            Some(min) => min,
            None => b
                .should
                .iter()
                .map(|c| clause(segments, c))
                .fold(0, u64::saturating_add),
        }
    }
    boolean(segments, query)
}

/// `f` over every slice, the results in slice order.
pub(crate) fn run_slices<T: Send>(
    slices: &[Vec<usize>],
    f: impl Fn(&[usize]) -> T + Sync,
) -> Vec<T> {
    let Some((first, rest)) = slices.split_first() else {
        return Vec::new();
    };
    if rest.is_empty() {
        return vec![f(first)];
    }
    let mut out: Vec<Option<T>> = std::iter::repeat_with(|| None).take(slices.len()).collect();
    let (head, tail) = out.split_at_mut(1);
    let f = &f;
    rayon::in_place_scope(|scope| {
        for (slot, slice) in tail.iter_mut().zip(rest) {
            scope.spawn(move |_| *slot = Some(f(slice)));
        }
        if let Some(slot) = head.first_mut() {
            *slot = Some(f(first));
        }
    });
    // Every slot is filled once the scope returns (a panicking slice
    // resumes its panic there instead).
    out.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slice_runs_once_and_answers_in_order() {
        let slices: Vec<Vec<usize>> = (0..5).map(|i| vec![i, i + 10]).collect();
        let got = run_slices(&slices, |s| s.iter().sum::<usize>());
        assert_eq!(got, vec![10, 12, 14, 16, 18]);
        assert_eq!(run_slices(&[vec![3]], |s| s[0]), vec![3]);
        assert!(run_slices(&[], |s: &[usize]| s.len()).is_empty());
    }

    #[test]
    fn sequential_slices_answer_as_parallel_ones_do() {
        let slices: Vec<Vec<usize>> = (0..4).map(|i| vec![i, i * 3]).collect();
        let sum = |s: &[usize]| s.iter().sum::<usize>();
        assert_eq!(
            run_slices_if(false, &slices, sum),
            run_slices_if(true, &slices, sum)
        );
        assert_eq!(run_slices_if(false, &slices, sum), vec![0, 4, 8, 12]);
    }
}
