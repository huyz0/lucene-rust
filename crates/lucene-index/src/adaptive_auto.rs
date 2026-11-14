//! `index.codec.adaptive_doc_values: auto`: one choice of doc-values format
//! for a whole index, made from its own data.
//!
//! Adaptive doc values pay off when some column of the index is clustered in
//! document order (a timestamp in arrival order, an index-sort key): whole
//! 1024-document windows then fold from their block statistics, and every
//! field an aggregation reads needs those statistics, not only the clustered
//! one. When nothing is clustered they only cost scan speed. So the choice is
//! one for every field, from one question: is any column clustered?
//!
//! The writer asks it of its first [`SAMPLE_DOCS`] documents. Until then it
//! writes stock doc values; from the flush that reaches that many it writes
//! the chosen format, and so do its merges. The answer is recorded in the
//! commit's user data under [`COMMIT_KEY`], so a reopened writer keeps it;
//! until then each commit records [`PENDING`] there.
//!
//! A column's clustering is the median, over its full windows, of the
//! window's value range divided by the column's whole range across the
//! sample: about `1 / windows` for a column in order, near 1 for one in no
//! order. A column counts when at least [`MIN_WINDOWS`] of its windows were
//! seen and its median is under [`CLUSTERED`]. Windows run across segment
//! boundaries, so a shard flushing small segments is judged like one
//! flushing large ones.

use std::collections::HashMap;
use std::sync::Mutex;

/// Documents seen before the choice is made.
pub const SAMPLE_DOCS: u64 = 65_536;
/// Documents per window, as the block statistics cover them.
pub const WINDOW: usize = 1024;
/// A column's median window range over its whole range below this is
/// clustered.
pub const CLUSTERED: f64 = 0.25;
/// Full windows a column needs before its median counts.
pub const MIN_WINDOWS: usize = 16;
/// The commit user-data key holding the choice (`true` or `false`).
pub const COMMIT_KEY: &str = "adaptive_doc_values";

/// [`COMMIT_KEY`]'s value while the shard is still sampling.
pub const PENDING: &str = "auto";

/// One column's windows so far.
#[derive(Default)]
struct Column {
    min: i64,
    max: i64,
    seen: bool,
    ranges: Vec<u64>,
    /// The values after the last full window, carried into the next
    /// segment so small flushes still add up to windows.
    tail: Vec<i64>,
}

impl Column {
    fn window(&mut self, window: &[i64]) {
        let (lo, hi) = window
            .iter()
            .fold((i64::MAX, i64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        self.ranges.push(hi.abs_diff(lo));
        if self.seen {
            self.min = self.min.min(lo);
            self.max = self.max.max(hi);
        } else {
            (self.min, self.max, self.seen) = (lo, hi, true);
        }
    }

    /// `values` follow the tail in document order.
    fn extend(&mut self, mut values: &[i64]) {
        if !self.tail.is_empty() {
            let take = WINDOW.saturating_sub(self.tail.len()).min(values.len());
            let (head, rest) = values.split_at(take);
            self.tail.extend_from_slice(head);
            values = rest;
            if self.tail.len() < WINDOW {
                return;
            }
            let tail = std::mem::take(&mut self.tail);
            self.window(&tail);
        }
        let mut windows = values.chunks_exact(WINDOW);
        for w in windows.by_ref() {
            self.window(w);
        }
        self.tail.extend_from_slice(windows.remainder());
    }
}

enum State {
    Sampling {
        docs: u64,
        columns: HashMap<i32, Column>,
    },
    Decided(bool),
}

/// The choice, or what has been seen towards it. Shared by the writer's
/// flushes, which feed it, and its merges, which read it.
pub struct AdaptiveAuto {
    state: Mutex<State>,
}

impl AdaptiveAuto {
    /// A writer's chooser: `recorded` is the choice its commit holds, if
    /// any.
    pub fn new(recorded: Option<bool>) -> Self {
        let state = match recorded {
            Some(choice) => State::Decided(choice),
            None => State::Sampling {
                docs: 0,
                columns: HashMap::new(),
            },
        };
        AdaptiveAuto {
            state: Mutex::new(state),
        }
    }

    /// The choice recorded under [`COMMIT_KEY`] in commit user data.
    pub fn recorded(user_data: &[(String, String)]) -> Option<bool> {
        user_data
            .iter()
            .find(|(k, _)| k == COMMIT_KEY)
            .and_then(|(_, v)| v.parse().ok())
    }

    /// The choice, once made.
    pub fn decided(&self) -> Option<bool> {
        decided(&self.state.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// A segment about to be written, of `max_doc` documents: `columns`
    /// gives each numeric field's values in document order where every
    /// document has exactly one (other fields are left out). Makes the
    /// choice once [`SAMPLE_DOCS`] documents have been seen, so the segment
    /// that reaches them is written in the chosen format; returns it.
    pub fn observe<'c>(
        &self,
        max_doc: usize,
        columns: impl IntoIterator<Item = (i32, &'c [i64])>,
    ) -> Option<bool> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let State::Sampling {
            docs,
            columns: seen,
        } = &mut *state
        else {
            return decided(&state);
        };
        let mut present = Vec::new();
        for (field, values) in columns {
            if values.len() != max_doc {
                continue;
            }
            seen.entry(field).or_default().extend(values);
            present.push(field);
        }
        // A column missing from this segment breaks its run: its carried
        // values no longer continue into the next segment's.
        for (field, column) in seen.iter_mut() {
            if !present.contains(field) {
                column.tail.clear();
            }
        }
        *docs = docs.saturating_add(max_doc as u64);
        if *docs < SAMPLE_DOCS {
            return None;
        }
        let choice = seen.values_mut().any(clustered);
        *state = State::Decided(choice);
        Some(choice)
    }
}

fn decided(state: &State) -> Option<bool> {
    match state {
        State::Decided(choice) => Some(*choice),
        State::Sampling { .. } => None,
    }
}

/// Whether `column`'s windows are narrow against its whole range.
fn clustered(column: &mut Column) -> bool {
    if column.ranges.len() < MIN_WINDOWS {
        return false;
    }
    let whole = column.max.abs_diff(column.min);
    if whole == 0 {
        // One value throughout: every window folds whatever the format.
        return false;
    }
    let middle = column.ranges.len() / 2;
    let (_, &mut median, _) = column.ranges.select_nth_unstable(middle);
    (median as f64) < CLUSTERED * whole as f64
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn feed(auto: &AdaptiveAuto, segments: usize, per: usize, f: impl Fn(usize) -> i64) {
        let mut doc = 0;
        for _ in 0..segments {
            let values: Vec<i64> = (doc..doc + per).map(&f).collect();
            auto.observe(per, [(0, values.as_slice())]);
            doc += per;
        }
    }

    /// A shuffle by a fixed odd multiplier: a column in no order.
    fn scramble(i: usize) -> i64 {
        (i.wrapping_mul(2_654_435_761) % 1_000_003) as i64
    }

    #[test]
    fn a_column_in_order_makes_the_index_adaptive_after_the_sample() {
        let auto = AdaptiveAuto::new(None);
        feed(&auto, 15, 4096, |i| 1_700_000_000_000 + i as i64 * 37);
        assert_eq!(auto.decided(), None, "61,440 documents: still sampling");
        feed(&auto, 1, 4096, |i| 1_700_000_000_000 + i as i64 * 37);
        assert_eq!(auto.decided(), Some(true));
    }

    #[test]
    fn columns_in_no_order_keep_the_index_stock() {
        let auto = AdaptiveAuto::new(None);
        feed(&auto, 8, 8192, scramble);
        assert_eq!(auto.decided(), Some(false));
    }

    #[test]
    fn one_clustered_column_among_shuffled_ones_is_enough() {
        let auto = AdaptiveAuto::new(None);
        for s in 0..4 {
            let base = s * 16_384;
            let ts: Vec<i64> = (base..base + 16_384).map(|i| i as i64 / 3).collect();
            let fare: Vec<i64> = (base..base + 16_384).map(scramble).collect();
            auto.observe(16_384, [(1, fare.as_slice()), (0, ts.as_slice())]);
        }
        assert_eq!(auto.decided(), Some(true));
    }

    #[test]
    fn small_flushes_add_up_to_windows() {
        // Flushes of 3,000 documents: windows run across them.
        let auto = AdaptiveAuto::new(None);
        feed(&auto, 22, 3_000, |i| i as i64);
        assert_eq!(auto.decided(), Some(true));
        // Flushes far smaller than a window (a refresh every second at a
        // few hundred documents a second) are judged the same.
        let auto = AdaptiveAuto::new(None);
        feed(&auto, 132, 500, |i| 1_700_000_000_000 + i as i64 * 37);
        assert_eq!(auto.decided(), Some(true));
        let auto = AdaptiveAuto::new(None);
        feed(&auto, 132, 500, scramble);
        assert_eq!(auto.decided(), Some(false));
    }

    #[test]
    fn a_segment_without_the_column_breaks_its_run() {
        let auto = AdaptiveAuto::new(None);
        let a: Vec<i64> = (0..1000).collect();
        auto.observe(1000, [(0, a.as_slice())]);
        auto.observe(500, std::iter::empty());
        let state = auto.state.lock().unwrap();
        let State::Sampling { columns, .. } = &*state else {
            panic!("still sampling");
        };
        assert!(columns[&0].tail.is_empty() && columns[&0].ranges.is_empty());
    }

    #[test]
    fn sparse_constant_and_short_columns_do_not_count() {
        let auto = AdaptiveAuto::new(None);
        let constant = vec![7i64; SAMPLE_DOCS as usize];
        let short = vec![1i64; 100];
        auto.observe(
            SAMPLE_DOCS as usize,
            [(0, constant.as_slice()), (1, short.as_slice())],
        );
        assert_eq!(auto.decided(), Some(false));
        // Too few windows for a median: a 4-window column never counts.
        let auto = AdaptiveAuto::new(None);
        let few: Vec<i64> = (0..4 * WINDOW as i64).collect();
        auto.observe(4 * WINDOW, [(0, few.as_slice())]);
        auto.observe(SAMPLE_DOCS as usize, std::iter::empty());
        assert_eq!(auto.decided(), Some(false));
    }

    #[test]
    fn a_recorded_choice_is_kept_and_never_resampled() {
        let user_data = vec![
            ("translog_uuid".to_string(), "x".to_string()),
            (COMMIT_KEY.to_string(), "true".to_string()),
        ];
        let auto = AdaptiveAuto::new(AdaptiveAuto::recorded(&user_data));
        assert_eq!(auto.decided(), Some(true));
        let shuffled: Vec<i64> = (0..SAMPLE_DOCS as usize).map(scramble).collect();
        assert_eq!(
            auto.observe(SAMPLE_DOCS as usize, [(0, shuffled.as_slice())]),
            Some(true)
        );
        assert_eq!(AdaptiveAuto::recorded(&[]), None);
        assert_eq!(
            AdaptiveAuto::recorded(&[(COMMIT_KEY.to_string(), "maybe".to_string())]),
            None
        );
        // Still sampling when the commit was written: sample again.
        assert_eq!(
            AdaptiveAuto::recorded(&[(COMMIT_KEY.to_string(), PENDING.to_string())]),
            None
        );
    }

    #[test]
    fn extreme_values_do_not_overflow() {
        let auto = AdaptiveAuto::new(None);
        let values: Vec<i64> = (0..SAMPLE_DOCS as usize)
            .map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX })
            .collect();
        assert_eq!(
            auto.observe(SAMPLE_DOCS as usize, [(0, values.as_slice())]),
            Some(false)
        );
    }
}
