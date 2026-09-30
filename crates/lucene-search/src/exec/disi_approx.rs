//! `DisjunctionDISIApproximation` over `DisiPriorityQueue2`/`DisiPriorityQueueN`,
//! member for member -- including the order `topList()` links the members
//! on the current document in. A scorer that folds its members' values in
//! that order in `float` (`SynonymScorer.freq`, `CombinedFieldScorer.freq`,
//! `LogOddsFusionScorer.score`) rounds differently in any other order once
//! three or more non-integral values meet, so the order is part of the
//! result, not a detail. [`super::disjunction`] keeps its own queue, which
//! only a `double` sum reads.

use crate::Result;

/// What the approximation needs from a member: a `DocIdSetIterator`.
pub(crate) trait DisiSub {
    fn doc_id(&self) -> i32;
    fn next_doc(&mut self) -> Result<i32>;
    fn advance(&mut self, target: i32) -> Result<i32>;
    fn cost(&self) -> i64;
}

impl DisiSub for super::BoxScorer<'_> {
    fn doc_id(&self) -> i32 {
        super::Scorer::doc_id(&**self)
    }
    fn next_doc(&mut self) -> Result<i32> {
        super::Scorer::next_doc(&mut **self)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        super::Scorer::advance(&mut **self, target)
    }
    fn cost(&self) -> i64 {
        super::Scorer::cost(&**self)
    }
}

/// `DisjunctionDISIApproximation`: `subs` are the `DisiWrapper`s, addressed
/// by their index in the order given.
pub(crate) struct DisiApprox<S> {
    pub(crate) subs: Vec<S>,
    /// `DisiWrapper.doc`, per member.
    docs: Vec<i32>,
    costs: Vec<i64>,
    /// `leadIterators`: `DisiPriorityQueue2` (`two`) or the `N` heap.
    lead: Vec<usize>,
    two: bool,
    /// `otherIterators`.
    others: Vec<usize>,
    min_other_doc: i32,
    doc: i32,
    cost: i64,
}

impl<S: DisiSub> DisiApprox<S> {
    /// `new DisjunctionDISIApproximation(subIterators, leadCost)`.
    pub(crate) fn new(subs: Vec<S>, lead_cost: i64) -> Self {
        let n = subs.len();
        let costs: Vec<i64> = subs.iter().map(DisiSub::cost).collect();
        // `Arrays.sort(wrappers, comparingLong(cost).reversed())`: stable.
        let mut wrappers: Vec<usize> = (0..n).collect();
        wrappers.sort_by(|&a, &b| costs[b].cmp(&costs[a]));
        let mut reorder_threshold = lead_cost.wrapping_add(lead_cost >> 1);
        if reorder_threshold < 0 {
            reorder_threshold = i64::MAX;
        }
        let mut cost = 0i64;
        let mut reorder_cost = 0i64;
        // `lastIdx`, as a signed index: `-1` means every wrapper leads.
        let mut last_idx = n as isize - 1;
        while last_idx >= 0 {
            let last_cost = costs[wrappers[last_idx as usize]];
            let inc = last_cost.min(lead_cost);
            let next = reorder_cost.wrapping_add(inc);
            if next < 0 || next > reorder_threshold {
                break;
            }
            reorder_cost = next;
            cost = cost.wrapping_add(last_cost);
            last_idx -= 1;
        }
        if last_idx == n as isize - 1 && n > 0 {
            cost = cost.wrapping_add(costs[wrappers[last_idx as usize]]);
            last_idx -= 1;
        }
        let split = (last_idx + 1) as usize;
        let lead_members: Vec<usize> = wrappers[split..].to_vec();
        let others: Vec<usize> = wrappers[..split].to_vec();
        let docs = vec![-1; n];
        let mut min_other_doc = i32::MAX;
        for &w in &others {
            cost = cost.wrapping_add(costs[w]);
            min_other_doc = min_other_doc.min(docs[w]);
        }
        let two = lead_members.len() <= 2;
        let mut out = Self {
            subs,
            docs,
            costs,
            lead: Vec::with_capacity(lead_members.len()),
            two,
            others,
            min_other_doc,
            doc: -1,
            cost,
        };
        if two {
            // `DisiPriorityQueue.addAll`: one `add` at a time.
            for &w in &lead_members {
                out.lead.push(w);
                if out.lead.len() == 2 {
                    out.update_top();
                }
            }
        } else {
            out.lead = lead_members;
            out.heapify();
        }
        out
    }

    /// `DisiPriorityQueueN.addAll`'s bulk heapify.
    fn heapify(&mut self) {
        let size = self.lead.len();
        let first_leaf = size >> 1;
        for root in (0..first_leaf).rev() {
            let mut parent_index = root;
            let parent = self.lead[parent_index];
            while parent_index < first_leaf {
                let mut child_index = left(parent_index);
                let right_index = child_index + 1;
                let mut child = self.lead[child_index];
                if right_index < size && self.docs[self.lead[right_index]] < self.docs[child] {
                    child = self.lead[right_index];
                    child_index = right_index;
                }
                if self.docs[child] >= self.docs[parent] {
                    break;
                }
                self.lead[parent_index] = child;
                parent_index = child_index;
            }
            self.lead[parent_index] = parent;
        }
    }

    fn top(&self) -> usize {
        self.lead[0]
    }

    /// `updateTop()`: the queue's order restored after the top's doc grew.
    fn update_top(&mut self) -> usize {
        if self.two {
            if self.lead.len() == 2 && self.docs[self.lead[1]] < self.docs[self.lead[0]] {
                self.lead.swap(0, 1);
            }
            return self.lead[0];
        }
        // `downHeap(size)`.
        let size = self.lead.len();
        let mut i = 0;
        let node = self.lead[0];
        let node_doc = self.docs[node];
        let mut j = left(i);
        if j < size {
            let mut k = j + 1;
            if k < size && self.docs[self.lead[k]] < self.docs[self.lead[j]] {
                j = k;
            }
            if self.docs[self.lead[j]] < node_doc {
                loop {
                    self.lead[i] = self.lead[j];
                    i = j;
                    j = left(i);
                    k = j + 1;
                    if k < size && self.docs[self.lead[k]] < self.docs[self.lead[j]] {
                        j = k;
                    }
                    if !(j < size && self.docs[self.lead[j]] < node_doc) {
                        break;
                    }
                }
                self.lead[i] = node;
            }
        }
        self.lead[0]
    }

    pub(crate) fn doc_id(&self) -> i32 {
        self.doc
    }

    pub(crate) fn cost(&self) -> i64 {
        self.cost
    }

    /// `nextDoc()`.
    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        let mut top = self.top();
        if self.docs[top] < self.min_other_doc {
            let cur = self.docs[top];
            loop {
                self.docs[top] = self.subs[top].next_doc()?;
                top = self.update_top();
                if self.docs[top] != cur {
                    break;
                }
            }
            self.doc = self.docs[top].min(self.min_other_doc);
            Ok(self.doc)
        } else {
            self.advance(self.min_other_doc.saturating_add(1))
        }
    }

    /// `advance(target)`.
    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        let mut top = self.top();
        while self.docs[top] < target {
            self.docs[top] = self.subs[top].advance(target)?;
            top = self.update_top();
        }
        self.min_other_doc = i32::MAX;
        for k in 0..self.others.len() {
            let w = self.others[k];
            if self.docs[w] < target {
                self.docs[w] = self.subs[w].advance(target)?;
            }
            self.min_other_doc = self.min_other_doc.min(self.docs[w]);
        }
        self.doc = self.docs[top].min(self.min_other_doc);
        Ok(self.doc)
    }

    /// `topList()`: the members on the current document, head first, in the
    /// order Java links them.
    pub(crate) fn top_list(&self, out: &mut Vec<usize>) {
        out.clear();
        let top = self.top();
        if self.docs[top] < self.min_other_doc {
            self.lead_top_list(out);
        } else {
            // `computeTopList`: the lead's list, then each matching other
            // member prepended in turn.
            if self.docs[top] == self.min_other_doc {
                self.lead_top_list(out);
            }
            for &w in &self.others {
                if self.docs[w] == self.min_other_doc {
                    out.insert(0, w);
                }
            }
        }
    }

    /// `leadIterators.topList()`.
    fn lead_top_list(&self, out: &mut Vec<usize>) {
        let top = self.lead[0];
        if self.two {
            if self.lead.len() == 2 && self.docs[self.lead[1]] == self.docs[top] {
                out.push(self.lead[1]);
            }
            out.push(top);
            return;
        }
        // Built by prepending, as Java links it: collected in prepend order
        // and reversed at the end.
        let mut rev = vec![top];
        let size = self.lead.len();
        if size >= 3 {
            self.collect(&mut rev, size, 1, self.docs[top]);
            self.collect(&mut rev, size, 2, self.docs[top]);
        } else if size == 2 && self.docs[self.lead[1]] == self.docs[top] {
            rev.push(self.lead[1]);
        }
        out.extend(rev.into_iter().rev());
    }

    /// `DisiPriorityQueueN.topList(list, heap, size, i)`.
    fn collect(&self, rev: &mut Vec<usize>, size: usize, i: usize, doc: i32) {
        let w = self.lead[i];
        if self.docs[w] == doc {
            rev.push(w);
            let l = left(i);
            let r = l + 1;
            if r < size {
                self.collect(rev, size, l, doc);
                self.collect(rev, size, r, doc);
            } else if l < size && self.docs[self.lead[l]] == doc {
                rev.push(self.lead[l]);
            }
        }
    }

    /// Member `i`'s `DisiWrapper.cost`.
    #[allow(dead_code)]
    pub(crate) fn member_cost(&self, i: usize) -> i64 {
        self.costs[i]
    }

    /// Member `i`'s document as the approximation last saw it.
    pub(crate) fn member_doc(&self, i: usize) -> i32 {
        self.docs[i]
    }
}

fn left(i: usize) -> usize {
    2 * i + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    struct V {
        docs: Vec<i32>,
        at: usize,
        doc: i32,
    }

    impl V {
        fn new(docs: &[i32]) -> Self {
            Self {
                docs: docs.to_vec(),
                at: 0,
                doc: -1,
            }
        }
    }

    impl DisiSub for V {
        fn doc_id(&self) -> i32 {
            self.doc
        }
        fn next_doc(&mut self) -> Result<i32> {
            self.doc = self.docs.get(self.at).copied().unwrap_or(i32::MAX);
            self.at += 1;
            Ok(self.doc)
        }
        fn advance(&mut self, target: i32) -> Result<i32> {
            loop {
                let d = self.next_doc()?;
                if d >= target {
                    return Ok(d);
                }
            }
        }
        fn cost(&self) -> i64 {
            self.docs.len() as i64
        }
    }

    fn walk(lists: &[&[i32]], lead_cost: i64) -> Vec<(i32, Vec<usize>)> {
        let subs: Vec<V> = lists.iter().map(|l| V::new(l)).collect();
        let mut d = DisiApprox::new(subs, lead_cost);
        let mut out = Vec::new();
        let mut list = Vec::new();
        loop {
            let doc = d.next_doc().unwrap();
            if doc == i32::MAX {
                break;
            }
            d.top_list(&mut list);
            out.push((doc, list.clone()));
        }
        out
    }

    #[test]
    fn union_and_members_on_each_document() {
        let lists: [&[i32]; 3] = [&[1, 3, 5], &[3, 4], &[0, 3, 5, 9]];
        for lead_cost in [i64::MAX, 1, 2] {
            let got = walk(&lists, lead_cost);
            let docs: Vec<i32> = got.iter().map(|(d, _)| *d).collect();
            assert_eq!(docs, vec![0, 1, 3, 4, 5, 9]);
            for (doc, members) in &got {
                let mut want: Vec<usize> = (0..3).filter(|&i| lists[i].contains(doc)).collect();
                let mut m = members.clone();
                m.sort_unstable();
                want.sort_unstable();
                assert_eq!(m, want, "doc {doc} lead cost {lead_cost}");
            }
        }
    }

    #[test]
    fn two_member_queue_links_the_second_first() {
        let got = walk(&[&[2, 4], &[2, 4]], i64::MAX);
        assert_eq!(got[0], (2, vec![1, 0]));
    }

    #[test]
    fn advance_skips_and_heap_orders_many_members() {
        let lists: Vec<Vec<i32>> = (0..7)
            .map(|i| (0..50).filter(|d| d % (i + 2) == 0).collect())
            .collect();
        let refs: Vec<&[i32]> = lists.iter().map(|l| l.as_slice()).collect();
        let subs: Vec<V> = refs.iter().map(|l| V::new(l)).collect();
        let mut d = DisiApprox::new(subs, i64::MAX);
        assert_eq!(d.advance(7).unwrap(), 8);
        let mut list = Vec::new();
        d.top_list(&mut list);
        list.sort_unstable();
        assert_eq!(list, vec![0, 2, 6]);
        assert_eq!(d.doc_id(), 8);
        assert!(d.cost() > 0);
        assert_eq!(d.member_doc(0), 8);
    }
}
