//! Port of `org.apache.lucene.index.MultiBits`: a composite reader's live
//! documents, one bit per top-level document id, answered by the segment
//! that holds it -- `MultiBits.getLiveDocs(reader)`.
//!
//! Java returns the lone segment's own `Bits` for a one-segment reader; here
//! it is a one-segment [`MultiBits`] (same answers). A document id outside
//! the reader, which Java only asserts against, is not live here rather than
//! a panic.

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::directory_reader::DirectoryReader;

/// `MultiBits`.
#[derive(Debug, Clone)]
pub struct MultiBits<'r> {
    /// Each segment's live docs; `None` for a segment without deletions.
    subs: Vec<Option<&'r FixedBitSet>>,
    /// `starts`: each segment's doc base, then the reader's `maxDoc`.
    starts: Vec<i32>,
    /// `defaultValue`: what a segment without deletions answers.
    default_value: bool,
}

/// `ReaderUtil.subIndex(n, docStarts)`: the segment holding top-level doc
/// `n` -- the last of equal starts, so empty segments are skipped -- or `-1`
/// below the first start.
// SENTINEL: `-1` = `n` is below every start; `MultiBits::get` tests it
// (`usize::try_from` fails) before indexing.
fn sub_index(n: i32, doc_starts: &[i32]) -> isize {
    let size = doc_starts.len();
    let mut lo = 0isize;
    // ARITH: `doc_starts` is an in-memory slice, so its length fits `isize`.
    #[allow(clippy::arithmetic_side_effects)]
    let mut hi = size as isize - 1;
    while hi >= lo {
        // ARITH: `0 <= lo <= hi < size`, so the sum fits and halves back in
        // range (Java's `>>> 1`).
        #[allow(clippy::arithmetic_side_effects)]
        let mut mid = (lo + hi) / 2;
        let mid_value = doc_starts[mid as usize];
        if n < mid_value {
            // ARITH: `mid >= lo >= 0`, so `mid - 1 >= -1`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                hi = mid - 1;
            }
        } else if n > mid_value {
            // ARITH: `mid <= hi < size`, so `mid + 1 <= size`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                lo = mid + 1;
            }
        } else {
            // ARITH: `mid + 1 < size` is checked before each step.
            #[allow(clippy::arithmetic_side_effects)]
            while ((mid + 1) as usize) < size && doc_starts[(mid + 1) as usize] == mid_value {
                mid += 1;
            }
            return mid;
        }
    }
    hi
}

impl<'r> MultiBits<'r> {
    /// `MultiBits.getLiveDocs(reader)`: `None` when no segment has deletions
    /// (every document is live).
    pub fn live_docs(reader: &'r DirectoryReader) -> Option<MultiBits<'r>> {
        let segments = reader.segment_readers();
        if segments.iter().all(|s| s.live_docs().is_none()) {
            return None;
        }
        let mut starts: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
        starts.push(reader.max_doc());
        Some(MultiBits {
            subs: segments.iter().map(|s| s.live_docs()).collect(),
            starts,
            default_value: true,
        })
    }

    /// `get(doc)`: whether top-level document `doc` is live.
    pub fn get(&self, doc: i32) -> bool {
        let Ok(reader) = usize::try_from(sub_index(doc, &self.starts)) else {
            return false;
        };
        // `subIndex` answers the last slot (the `maxDoc` entry) for a doc at
        // or past the end: no segment holds it.
        let Some(sub) = self.subs.get(reader) else {
            return false;
        };
        match sub {
            None => self.default_value,
            Some(bits) => {
                let local = doc.saturating_sub(self.starts[reader]);
                match usize::try_from(local) {
                    Ok(local) if local < bits.len() => bits.get(local),
                    _ => false,
                }
            }
        }
    }

    /// `length()`: the reader's `maxDoc`.
    pub fn len(&self) -> i32 {
        self.starts.last().copied().unwrap_or(0)
    }

    /// Whether the reader holds no documents at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_index_follows_reader_util() {
        // Segments of 3, 0 (empty), 2 docs: starts 0, 3, 3, then maxDoc 5.
        let starts = [0, 3, 3, 5];
        assert_eq!(sub_index(0, &starts), 0);
        assert_eq!(sub_index(2, &starts), 0);
        assert_eq!(sub_index(3, &starts), 2, "the empty segment is skipped");
        assert_eq!(sub_index(4, &starts), 2);
        assert_eq!(sub_index(5, &starts), 3, "past the end: the maxDoc slot");
        assert_eq!(sub_index(-1, &starts), -1);
        assert_eq!(sub_index(7, &[]), -1);
    }

    #[test]
    fn get_answers_per_segment() {
        let mut first = FixedBitSet::new(3);
        first.set(0);
        first.set(2);
        let bits = MultiBits {
            subs: vec![Some(&first), None, None],
            starts: vec![0, 3, 3, 5],
            default_value: true,
        };
        let live: Vec<bool> = (0..5).map(|d| bits.get(d)).collect();
        assert_eq!(live, [true, false, true, true, true]);
        assert!(!bits.get(-1));
        assert!(!bits.get(5));
        assert_eq!(bits.len(), 5);
        assert!(!bits.is_empty());
        // A live-docs set shorter than its segment (never written by either
        // engine) answers "not live" past its end rather than panicking.
        let short = FixedBitSet::new(1);
        let bits = MultiBits {
            subs: vec![Some(&short)],
            starts: vec![0, 3],
            default_value: true,
        };
        assert!(!bits.get(2));
    }
}
