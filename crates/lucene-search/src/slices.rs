//! Running a concurrent segment search's slices.
//!
//! Lucene hands each slice to its executor and waits for them all. Here the
//! first slice runs on the calling thread while rayon's pool takes the rest:
//! the handoff to a pool thread (a wake-up, for a pool that has gone to sleep
//! between requests) then overlaps work rather than preceding it, which is
//! what a slice doing little -- a rare term, a small segment -- would
//! otherwise pay for in full.

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
}
