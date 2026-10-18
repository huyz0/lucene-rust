//! `FieldPhraseList`: the phrases (and single terms) of the query found in
//! a document, longest first, with the offsets they cover.

use std::cmp::Ordering;

use super::field_query::{FieldQuery, QueryPhraseMap};
use super::field_term_stack::{FieldTermStack, TermInfo};

/// `WeightedPhraseInfo.Toffs`: an offset range to mark up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Toffs {
    start_offset: i32,
    end_offset: i32,
}

impl Toffs {
    /// `new Toffs(startOffset, endOffset)`.
    pub fn new(start_offset: i32, end_offset: i32) -> Self {
        Self {
            start_offset,
            end_offset,
        }
    }

    /// `getStartOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `getEndOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `compareTo`: by start, then end offset.
    pub fn compare(&self, other: &Toffs) -> Ordering {
        self.start_offset
            .cmp(&other.start_offset)
            .then(self.end_offset.cmp(&other.end_offset))
    }
}

impl std::fmt::Display for Toffs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.start_offset, self.end_offset)
    }
}

/// `WeightedPhraseInfo`.
#[derive(Debug, Clone)]
pub struct WeightedPhraseInfo {
    terms_offsets: Vec<Toffs>,
    boost: f32,
    seqnum: i32,
    terms_infos: Vec<TermInfo>,
}

/// `MergedIterator(false, iterators)`: repeatedly the smallest head, the
/// earlier iterator among equals.
fn merged<T: Copy>(lists: &[Vec<T>], cmp: impl Fn(&T, &T) -> Ordering) -> Vec<T> {
    let mut at = vec![0usize; lists.len()];
    let mut out = Vec::new();
    loop {
        let mut best: Option<usize> = None;
        for (i, list) in lists.iter().enumerate() {
            let Some(head) = list.get(at[i]) else {
                continue;
            };
            let better = match best {
                None => true,
                Some(b) => cmp(head, &lists[b][at[b]]) == Ordering::Less,
            };
            if better {
                best = Some(i);
            }
        }
        let Some(b) = best else {
            return out;
        };
        out.push(lists[b][at[b]]);
        at[b] += 1;
    }
}

impl WeightedPhraseInfo {
    /// `WeightedPhraseInfo(terms, boost, seqnum)`: consecutive positions
    /// share one range.
    ///
    /// # Panics
    /// With no `terms`.
    pub fn new(terms: &[TermInfo], boost: f32, seqnum: i32) -> Self {
        let first = &terms[0];
        let mut terms_offsets = vec![Toffs::new(first.start_offset(), first.end_offset())];
        let mut pos = first.position();
        for ti in &terms[1..] {
            if ti.position() - pos == 1 {
                if let Some(last) = terms_offsets.last_mut() {
                    last.end_offset = ti.end_offset();
                }
            } else {
                terms_offsets.push(Toffs::new(ti.start_offset(), ti.end_offset()));
            }
            pos = ti.position();
        }
        Self {
            terms_offsets,
            boost,
            seqnum,
            terms_infos: terms.to_vec(),
        }
    }

    /// `WeightedPhraseInfo(Collection<WeightedPhraseInfo>)`: one phrase
    /// standing for all of `to_merge`, their ranges joined where they touch.
    ///
    /// # Panics
    /// With nothing to merge.
    pub fn merge(to_merge: &[WeightedPhraseInfo]) -> Self {
        let first = &to_merge[0];
        let mut boost = first.boost;
        let mut terms_infos = Vec::new();
        for info in &to_merge[1..] {
            boost += info.boost;
            terms_infos.extend(info.terms_infos.iter().cloned());
        }
        let lists: Vec<Vec<Toffs>> = to_merge.iter().map(|i| i.terms_offsets.clone()).collect();
        let all = merged(&lists, Toffs::compare);
        let mut terms_offsets = Vec::new();
        if let Some((&head, rest)) = all.split_first() {
            let mut work = head;
            for current in rest {
                if current.start_offset <= work.end_offset {
                    work.end_offset = work.end_offset.max(current.end_offset);
                } else {
                    terms_offsets.push(work);
                    work = *current;
                }
            }
            terms_offsets.push(work);
        }
        Self {
            terms_offsets,
            boost,
            seqnum: first.seqnum,
            terms_infos,
        }
    }

    /// `getText()`: the terms' texts, run together.
    pub fn text(&self) -> String {
        self.terms_infos.iter().map(TermInfo::text).collect()
    }

    /// `getTermsOffsets()`.
    pub fn terms_offsets(&self) -> &[Toffs] {
        &self.terms_offsets
    }

    /// `getBoost()`.
    pub fn boost(&self) -> f32 {
        self.boost
    }

    /// `getTermsInfos()`.
    pub fn terms_infos(&self) -> &[TermInfo] {
        &self.terms_infos
    }

    /// `getSeqnum()`.
    pub fn seqnum(&self) -> i32 {
        self.seqnum
    }

    /// `getStartOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.terms_offsets.first().map_or(0, Toffs::start_offset)
    }

    /// `getEndOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.terms_offsets.last().map_or(0, Toffs::end_offset)
    }

    /// `isOffsetOverlap(other)`.
    pub fn is_offset_overlap(&self, other: &WeightedPhraseInfo) -> bool {
        let (so, eo) = (self.start_offset(), self.end_offset());
        let (oso, oeo) = (other.start_offset(), other.end_offset());
        (so <= oso && oso < eo)
            || (so < oeo && oeo <= eo)
            || (oso <= so && so < oeo)
            || (oso < eo && eo <= oeo)
    }

    /// `compareTo`: by start, then end offset, then boost.
    pub fn compare(&self, other: &WeightedPhraseInfo) -> Ordering {
        self.start_offset()
            .cmp(&other.start_offset())
            .then(self.end_offset().cmp(&other.end_offset()))
            .then_with(|| {
                // `(int) Math.signum(getBoost() - other.getBoost())`
                let d = self.boost - other.boost;
                if d > 0.0 {
                    Ordering::Greater
                } else if d < 0.0 {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
    }
}

impl std::fmt::Display for WeightedPhraseInfo {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})(", self.text(), self.boost)?;
        for to in &self.terms_offsets {
            write!(f, "{to}")?;
        }
        write!(f, ")")
    }
}

/// The occurrences `ids` stand for.
fn views(stack: &FieldTermStack, ids: &[usize]) -> Vec<TermInfo> {
    ids.iter().map(|&i| stack.info(i).clone()).collect()
}

/// `FieldPhraseList`.
#[derive(Debug, Default)]
pub struct FieldPhraseList {
    phrase_list: Vec<WeightedPhraseInfo>,
}

impl FieldPhraseList {
    /// `FieldPhraseList(fieldTermStack, fieldQuery, phraseLimit)`: empties
    /// the stack, or until `phrase_limit` phrases are found.
    pub fn new(stack: &mut FieldTermStack, field_query: &FieldQuery, phrase_limit: usize) -> Self {
        let mut this = Self::default();
        let field = stack.field_name().to_string();
        while !stack.is_empty() && this.phrase_list.len() < phrase_limit {
            let Some(first) = stack.pop() else { break };
            let mut ti = first;
            let mut curr = field_query.field_term_map(&field, stack.info(ti).text());
            while curr.is_none() && stack.next_at_position(ti) != first {
                ti = stack.next_at_position(ti);
                curr = field_query.field_term_map(&field, stack.info(ti).text());
            }
            let Some(mut curr_map) = curr else { continue };
            let mut candidate: Vec<usize> = vec![ti];
            loop {
                let first = stack.pop();
                let mut ti = first;
                let mut next_map: Option<&QueryPhraseMap> = None;
                if let (Some(f), Some(t)) = (first, &mut ti) {
                    next_map = curr_map.term_map(stack.info(*t).text());
                    while next_map.is_none() && stack.next_at_position(*t) != f {
                        *t = stack.next_at_position(*t);
                        next_map = curr_map.term_map(stack.info(*t).text());
                    }
                }
                let (Some(t), Some(next_map)) = (ti, next_map) else {
                    if let Some(t) = ti {
                        stack.push(t);
                    }
                    let refs = views(stack, &candidate);
                    let as_refs: Vec<&TermInfo> = refs.iter().collect();
                    if curr_map.is_valid_term_or_phrase(&as_refs) {
                        this.add_if_no_overlap(WeightedPhraseInfo::new(
                            &refs,
                            curr_map.boost(),
                            curr_map.term_or_phrase_number(),
                        ));
                    } else {
                        while candidate.len() > 1 {
                            if let Some(last) = candidate.pop() {
                                stack.push(last);
                            }
                            let refs = views(stack, &candidate);
                            let as_refs: Vec<&TermInfo> = refs.iter().collect();
                            if let Some(found) = field_query.search_phrase(&field, &as_refs) {
                                this.add_if_no_overlap(WeightedPhraseInfo::new(
                                    &refs,
                                    found.boost(),
                                    found.term_or_phrase_number(),
                                ));
                                break;
                            }
                        }
                    }
                    break;
                };
                candidate.push(t);
                curr_map = next_map;
            }
        }
        this
    }

    /// `FieldPhraseList(FieldPhraseList[] toMerge)`: the phrases of all the
    /// lists in offset order, those that touch made one.
    pub fn merge(to_merge: &[FieldPhraseList]) -> Self {
        let mut at = vec![0usize; to_merge.len()];
        // heads compare as the phrases they stand for
        let mut all: Vec<&WeightedPhraseInfo> = Vec::new();
        loop {
            let mut best: Option<usize> = None;
            for (i, l) in to_merge.iter().enumerate() {
                let Some(head) = l.phrase_list.get(at[i]) else {
                    continue;
                };
                let better = match best {
                    None => true,
                    Some(b) => head.compare(&to_merge[b].phrase_list[at[b]]) == Ordering::Less,
                };
                if better {
                    best = Some(i);
                }
            }
            let Some(b) = best else { break };
            all.push(&to_merge[b].phrase_list[at[b]]);
            at[b] += 1;
        }
        let mut this = Self::default();
        let Some((first, rest)) = all.split_first() else {
            return this;
        };
        let mut work: Vec<WeightedPhraseInfo> = vec![(*first).clone()];
        let mut work_end = first.end_offset();
        for current in rest {
            if current.start_offset() <= work_end {
                work_end = work_end.max(current.end_offset());
                work.push((*current).clone());
            } else {
                this.flush(&mut work);
                work.push((*current).clone());
                work_end = current.end_offset();
            }
        }
        this.flush(&mut work);
        this
    }

    fn flush(&mut self, work: &mut Vec<WeightedPhraseInfo>) {
        if work.len() == 1 {
            self.phrase_list.push(work.remove(0));
        } else {
            self.phrase_list.push(WeightedPhraseInfo::merge(work));
            work.clear();
        }
    }

    /// `addIfNoOverlap(wpi)`: a phrase that overlaps an earlier one joins it.
    pub fn add_if_no_overlap(&mut self, wpi: WeightedPhraseInfo) {
        for exist in &mut self.phrase_list {
            if exist.is_offset_overlap(&wpi) {
                exist.terms_infos.extend(wpi.terms_infos.iter().cloned());
                exist.boost += wpi.boost;
                return;
            }
        }
        self.phrase_list.push(wpi);
    }

    /// `getPhraseList()`.
    pub fn phrase_list(&self) -> &[WeightedPhraseInfo] {
        &self.phrase_list
    }
}
