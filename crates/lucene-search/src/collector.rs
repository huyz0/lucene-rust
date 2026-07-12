//! `Collector`/`LeafCollector`-equivalent, pared down to this slice's scope:
//! a plain callback trait invoked once per matching (live) doc ID, in
//! ascending order. Real Lucene's `Collector`/`LeafCollector` split exists to
//! let a collector rebind per-segment state (e.g. a `Scorer` reference) when
//! `IndexSearcher` moves to the next leaf; that split has no work to do yet
//! since this slice never federates more than one segment (see `lib.rs`'s
//! module doc), so a single flat trait stands in for both.
//!
//! `search_term_query` (`lib.rs`) is generic over `C: Collector` rather than
//! taking `&mut dyn Collector`, per the `rust-performance` skill's
//! "monomorphize per-doc loops, `dyn` only at Query/Weight level" rule — the
//! per-doc `collect()` call in the hot loop is a direct (inlinable) call, not
//! a vtable dispatch.

/// Called once per matching, live doc ID, in ascending order — the entire
/// contract a collector needs for this slice (no scores, no per-segment
/// rebinding, no early-termination signal yet; see module doc).
pub trait Collector {
    fn collect(&mut self, doc_id: i32);
}

/// Collects every matching doc ID into a `Vec<i32>`, ascending — the
/// `TopDocs`-shaped "give me the actual hits" collector.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VecCollector {
    pub docs: Vec<i32>,
}

impl Collector for VecCollector {
    fn collect(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }
}

/// `TotalHitCountCollector`-equivalent: counts matches without retaining doc
/// IDs, for callers that only need "how many docs match".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CountCollector {
    pub count: i32,
}

impl Collector for CountCollector {
    fn collect(&mut self, _doc_id: i32) {
        self.count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_collector_collects_in_call_order() {
        let mut c = VecCollector::default();
        c.collect(3);
        c.collect(7);
        assert_eq!(c.docs, vec![3, 7]);
    }

    #[test]
    fn count_collector_counts_calls_not_values() {
        let mut c = CountCollector::default();
        c.collect(0);
        c.collect(0);
        c.collect(5);
        assert_eq!(c.count, 3);
    }
}
