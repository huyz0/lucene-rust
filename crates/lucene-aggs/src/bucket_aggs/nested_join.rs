//! The block join of `nested` and `reverse_nested`: the segment's parent and child lists, the
//! lookups over them, and the hand-over of buffered children to the sub-aggregations.

use super::*;

pub(super) struct NestedLeaf {
    /// The parent bitset's documents, ascending (deleted ones too, as the
    /// bitset holds them).
    pub(super) parents: Vec<i32>,
    /// The child filter's documents, ascending, and the iterator over them:
    /// `docID()` is `children[pos]`, -1 before the first `advance`.
    pub(super) children: Vec<i32>,
    pub(super) pos: Option<usize>,
    /// The document whose buckets are buffered, and those buckets.
    pub(super) current: i32,
    pub(super) buffer: Vec<u32>,
    /// The children `flush_nested` has counted but not yet handed to the sub-aggregations, with
    /// their buckets, ascending: handed over a window at a time ([`drain_nested`]).
    pub(super) pending_docs: Vec<i32>,
    pub(super) pending_ords: Vec<u32>,
    /// Where the last two parent lookups landed: documents arrive ascending, so the next one
    /// is found by galloping from there instead of a binary search over every parent.
    pub(super) hint_prev: std::cell::Cell<usize>,
    pub(super) hint_next: std::cell::Cell<usize>,
    /// `reverse_nested`'s `bucketOrdToLastCollectedParentDoc`.
    pub(super) last: FastMap<u32, i32>,
}

/// `v.partition_point(pred)` for an ascending `v`, searching forward from `hint` (the answer
/// of an earlier, smaller target) and doubling its steps: a nearby answer costs a few probes.
#[inline]
pub(super) fn gallop(
    v: &[i32],
    hint: &std::cell::Cell<usize>,
    pred: impl Fn(i32) -> bool + Copy,
) -> usize {
    let start = hint.get().min(v.len());
    let start = if start > 0 && !v.get(start - 1).is_some_and(|&x| pred(x)) {
        0
    } else {
        start
    };
    let rest = v.get(start..).unwrap_or(&[]);
    let mut lo = 0usize;
    let mut bound = 1usize;
    while lo + bound <= rest.len() && rest.get(lo + bound - 1).is_some_and(|&x| pred(x)) {
        lo += bound;
        bound <<= 1;
    }
    let hi = (lo + bound).min(rest.len());
    let at = start
        + lo
        + rest
            .get(lo..hi)
            .unwrap_or(&[])
            .partition_point(|&x| pred(x));
    hint.set(at);
    at
}

impl NestedLeaf {
    /// `childDocs.docID()`.
    //
    // SENTINEL: `-1` = "not yet advanced", `DocIdSetIterator.docID()`'s own
    // value before the first move; its one reader, `parent_and_child`, only
    // compares it with the previous parent (`<= prev`), where `-1` is at or
    // below every one, so the iterator advances, as in Java.
    pub(super) fn doc_id(&self) -> i32 {
        match self.pos {
            None => -1,
            Some(p) => self.children.get(p).copied().unwrap_or(i32::MAX),
        }
    }

    /// `advance(target)`: the first child at or past `target`.
    pub(super) fn advance(&mut self, target: i32) -> i32 {
        let from = self.pos.unwrap_or(0);
        let rest = self.children.get(from..).unwrap_or(&[]);
        self.pos = Some(from + gallop(rest, &std::cell::Cell::new(0), |d| d < target));
        self.doc_id()
    }

    pub(super) fn next_doc(&mut self) -> i32 {
        self.pos = Some(self.pos.map_or(0, |p| p + 1));
        self.doc_id()
    }

    /// `BitSet.nextSetBit(doc)`: the first parent at or after `doc`.
    pub(super) fn next_parent(&self, doc: i32) -> i32 {
        let i = gallop(&self.parents, &self.hint_next, |d| d < doc);
        self.parents.get(i).copied().unwrap_or(i32::MAX)
    }

    /// `NestedAggregator.getParentAndChildId`: the block-level parent of
    /// `doc`, and the iterator's next child under it.
    pub(super) fn parent_and_child(&mut self, doc: i32) -> (i32, i32) {
        // `i` parents are at or before `doc`: `prevSetBit(doc)` is the last of them, and
        // `nextSetBit` of anything past it the first after them -- one search, from where
        // the last block's was.
        let i = gallop(&self.parents, &self.hint_prev, |d| d <= doc);
        let at = |k: Option<usize>| k.and_then(|k| self.parents.get(k)).copied().unwrap_or(-1);
        let last = at(i.checked_sub(1));
        let (prev, current) = if last == -1 {
            (-1, self.parents.get(i).copied().unwrap_or(i32::MAX))
        } else if last == doc {
            (at(i.checked_sub(2)), doc)
        } else {
            (last, self.parents.get(i).copied().unwrap_or(i32::MAX))
        };
        let mut child = self.doc_id();
        if child <= prev {
            child = self.advance(prev + 1);
        }
        (current, child)
    }
}

/// `processBufferedChildBuckets`: the buffered document's children, each
/// handed every buffered bucket in turn.
pub(super) fn flush_nested(
    node: &AggNode,
    docs: &mut Vec<u64>,
    subs: &mut [State],
    leaf: &mut Leaf<'_>,
) -> Result<()> {
    let Some(n) = leaf.nested.as_deref_mut() else {
        return Ok(());
    };
    if n.buffer.is_empty() {
        return Ok(());
    }
    let (parent, mut child) = n.parent_and_child(n.current);
    if let (&[b], Some(from)) = (&n.buffer[..], n.pos) {
        // One bucket: the block's children are a run of the child list, taken whole.
        let rest = n.children.get(from..).unwrap_or(&[]);
        let run = &rest[..gallop(rest, &std::cell::Cell::new(0), |d| d < parent)];
        if !run.is_empty() {
            *slot(docs, b as usize) += run.len() as u64;
            n.pending_docs.extend_from_slice(run);
            n.pending_ords.resize(n.pending_docs.len(), b);
            n.pos = Some(from + run.len());
        }
    } else {
        while child < parent {
            for &b in &n.buffer {
                *slot(docs, b as usize) += 1;
                n.pending_docs.push(child);
                n.pending_ords.push(b);
            }
            child = n.next_doc();
        }
    }
    n.buffer.clear();
    if n.pending_docs.len() >= NESTED_PENDING {
        drain_nested(node, subs, leaf)?;
    }
    Ok(())
}

/// Children buffered before the sub-aggregations collect them.
pub(super) const NESTED_PENDING: usize = 4096;

/// Hands the buffered children to the sub-aggregations: a window of them at a time, each
/// column read ahead for the window, as the segment's own windows are, instead of one
/// dispatch (and one cursor seek) per child and bucket. Each sub-aggregation still sees its
/// children in ascending order.
pub(super) fn drain_nested(node: &AggNode, subs: &mut [State], leaf: &mut Leaf<'_>) -> Result<()> {
    let Some(n) = leaf.nested.as_deref_mut() else {
        return Ok(());
    };
    if n.pending_docs.is_empty() {
        return Ok(());
    }
    let (docs, ords) = (
        std::mem::take(&mut n.pending_docs),
        std::mem::take(&mut n.pending_ords),
    );
    let mut i = 0;
    while let Some(&first) = docs.get(i) {
        let base = first - first.rem_euclid(WINDOW);
        let end = base.saturating_add(WINDOW);
        let j = i + docs[i..].partition_point(|&d| d < end);
        let chunk = &docs[i..j];
        let last = chunk.last().copied().unwrap_or(first);
        // ARITH: `base <= first <= last < base + WINDOW`.
        #[allow(clippy::arithmetic_side_effects)]
        let len = (last - base + 1) as usize;
        for (sub, l) in node.subs().iter().zip(leaf.subs.iter_mut()) {
            if no_windows() {
                l.drop_prefetch();
            } else if l.prefetch_script(base, len, chunk)? {
            } else if chunk.len().saturating_mul(8) >= len {
                l.prefetch(base, len)?;
            } else if chunk.len() >= SPARSE_WINDOW && !matches!(sub, AggNode::Metric { .. }) {
                l.prefetch_docs(base, len, chunk)?;
            } else {
                l.drop_prefetch();
            }
        }
        for ((sub, st), l) in node
            .subs()
            .iter()
            .zip(subs.iter_mut())
            .zip(leaf.subs.iter_mut())
        {
            if no_windows() {
                for (&d, &o) in chunk.iter().zip(&ords[i..j]) {
                    collect(sub, st, l, d, o)?;
                }
            } else {
                collect_window(sub, st, l, chunk, &ords[i..j])?;
            }
        }
        i = j;
    }
    for l in leaf.subs.iter_mut() {
        l.drop_prefetch();
    }
    if let Some(n) = leaf.nested.as_deref_mut() {
        (n.pending_docs, n.pending_ords) = (docs, ords);
        n.pending_docs.clear();
        n.pending_ords.clear();
    }
    Ok(())
}

/// A `nested` or `reverse_nested`'s bitsets over the segment.
pub(super) fn nested_leaf(
    node: &AggNode,
    seg: &OpenSegment<'_>,
) -> Result<Option<Box<NestedLeaf>>> {
    let (parents, children) = match node {
        AggNode::Nested {
            parents, children, ..
        } => (parents, Some(children)),
        AggNode::ReverseNested { parents, .. } => (parents, None),
        _ => return Ok(None),
    };
    Ok(Some(Box::new(NestedLeaf {
        parents: lucene_search::top_field::nested_docs(seg, parents)?,
        children: match children {
            Some(c) => lucene_search::top_field::nested_docs(seg, c)?,
            None => Vec::new(),
        },
        pos: None,
        current: -1,
        buffer: Vec::new(),
        hint_prev: std::cell::Cell::new(0),
        hint_next: std::cell::Cell::new(0),
        pending_docs: Vec::new(),
        pending_ords: Vec::new(),
        last: FastMap::default(),
    })))
}
