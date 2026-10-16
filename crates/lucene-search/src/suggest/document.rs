//! `org.apache.lucene.search.suggest.document` (M13 T13.3d): the lookup half
//! of the completion postings format. [`lookup`] is `NRTSuggester.lookup`
//! (an automaton intersected with the per-field FST, then a best-first search
//! over what is below the accepting nodes); [`TopSuggestDocsCollector`] and
//! [`SuggestScoreDoc`] are what it fills. The files themselves are
//! `lucene_codecs::completion`.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashSet;

use lucene_codecs::completion::{decode_weight, NrtOutputs, NrtSuggester};
use lucene_codecs::fst::Pair;
use lucene_codecs::fst_util::{FstPath, TopNSearcher};
use lucene_store::{DataInput, Error, Result, SliceInput};
use lucene_util::automaton::Automaton;

use super::fst_util::intersect_prefix_paths;
use super::lookup::charsequence_compare;

/// `NRTSuggester.MAX_TOP_N_QUEUE_SIZE`.
const MAX_TOP_N_QUEUE_SIZE: i64 = 5000;

type NrtValue = Pair<i64, Vec<u8>>;

/// `TopSuggestDocs.SuggestScoreDoc`.
#[derive(Debug, Clone, PartialEq)]
pub struct SuggestScoreDoc {
    /// The document, with the segment's `docBase` added.
    pub doc: i32,
    /// The surface form.
    pub key: String,
    /// The context, for context suggest fields.
    pub context: Option<String>,
    /// The weight, boosted.
    pub score: f32,
}

/// `Float.compare`.
fn float_compare(a: f32, b: f32) -> Ordering {
    a.total_cmp(&b)
}

/// `SuggestScoreDocPriorityQueue.lessThan`.
fn less_than(a: &SuggestScoreDoc, b: &SuggestScoreDoc) -> bool {
    if a.score == b.score {
        let cmp = charsequence_compare(&a.key, &b.key);
        return if cmp != Ordering::Equal {
            cmp == Ordering::Greater
        } else {
            a.doc > b.doc
        };
    }
    a.score < b.score
}

/// `SuggestScoreDocPriorityQueue`: Java's binary heap (1-indexed, up-heap on
/// insert, down-heap on replace and pop), so equal entries leave in Java's
/// order.
#[derive(Debug)]
struct SuggestQueue {
    heap: Vec<Option<SuggestScoreDoc>>,
    size: usize,
    max_size: usize,
}

impl SuggestQueue {
    fn new(max_size: usize) -> Self {
        Self {
            heap: (0..=max_size).map(|_| None).collect(),
            size: 0,
            max_size,
        }
    }

    fn lt(&self, i: usize, j: usize) -> bool {
        match (&self.heap[i], &self.heap[j]) {
            (Some(a), Some(b)) => less_than(a, b),
            _ => false,
        }
    }

    fn up_heap(&mut self, mut i: usize) {
        let mut j = i >> 1;
        while j > 0 && self.lt(i, j) {
            self.heap.swap(i, j);
            i = j;
            j >>= 1;
        }
    }

    fn down_heap(&mut self, mut i: usize) {
        let mut j = i << 1;
        let k = j + 1;
        if k <= self.size && self.lt(k, j) {
            j = k;
        }
        while j <= self.size && self.lt(j, i) {
            self.heap.swap(i, j);
            i = j;
            j = i << 1;
            let k = j + 1;
            if k <= self.size && self.lt(k, j) {
                j = k;
            }
        }
    }

    /// `insertWithOverflow`: `true` when `element` itself was turned away
    /// (Java returns the element rather than the evicted top).
    fn insert_with_overflow(&mut self, element: SuggestScoreDoc) -> bool {
        if self.size < self.max_size {
            self.size += 1;
            self.heap[self.size] = Some(element);
            self.up_heap(self.size);
            false
        } else if self.size > 0
            && self.heap[1]
                .as_ref()
                .is_some_and(|top| less_than(top, &element))
        {
            self.heap[1] = Some(element);
            self.down_heap(1);
            false
        } else {
            true
        }
    }

    fn pop(&mut self) -> Option<SuggestScoreDoc> {
        if self.size == 0 {
            return None;
        }
        let result = self.heap[1].take();
        self.heap[1] = self.heap[self.size].take();
        self.size -= 1;
        if self.size > 0 {
            self.down_heap(1);
        }
        result
    }

    /// `getResults()`: best first.
    fn results(&mut self) -> Vec<SuggestScoreDoc> {
        let n = self.size;
        let mut out: Vec<Option<SuggestScoreDoc>> = (0..n).map(|_| None).collect();
        for slot in out.iter_mut().rev() {
            *slot = self.pop();
        }
        out.into_iter().flatten().collect()
    }
}

/// `TopSuggestDocsCollector`.
#[derive(Debug)]
pub struct TopSuggestDocsCollector {
    queue: SuggestQueue,
    num: usize,
    pending: Option<Vec<SuggestScoreDoc>>,
    seen_surface_forms: Option<HashSet<String>>,
    doc_base: i32,
    terminated: bool,
}

impl TopSuggestDocsCollector {
    /// `new TopSuggestDocsCollector(num, skipDuplicates)`.
    pub fn new(num: usize, skip_duplicates: bool) -> Result<Self> {
        if num == 0 {
            return Err(Error::IllegalArgument("'num' must be > 0".into()));
        }
        Ok(Self {
            queue: SuggestQueue::new(num),
            num,
            pending: skip_duplicates.then(Vec::new),
            seen_surface_forms: skip_duplicates.then(HashSet::new),
            doc_base: 0,
            terminated: false,
        })
    }

    /// `doSkipDuplicates()`.
    pub fn do_skip_duplicates(&self) -> bool {
        self.seen_surface_forms.is_some()
    }

    /// `getCountToCollect()`.
    pub fn count_to_collect(&self) -> usize {
        self.num
    }

    /// `doSetNextReader`: the next segment starts at `doc_base`.
    pub fn set_next_reader(&mut self, doc_base: i32) {
        self.doc_base = doc_base;
        self.terminated = false;
    }

    /// `collect(docID, key, context, score)`. Java throws
    /// `CollectionTerminatedException` when the hit does not fit, ending the
    /// segment; here the segment is marked terminated and the lookup stops
    /// accepting.
    pub fn collect(&mut self, doc_id: i32, key: &str, context: Option<&str>, score: f32) {
        let current = SuggestScoreDoc {
            doc: self.doc_base.wrapping_add(doc_id),
            key: key.to_string(),
            context: context.map(str::to_string),
            score,
        };
        if self.queue.insert_with_overflow(current) {
            self.terminated = true;
        }
    }

    /// Whether the current segment's collection has ended.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// `finish()`: after each segment's lookup, moves a deduplicating
    /// collector's hits aside.
    pub fn finish(&mut self) {
        if self.seen_surface_forms.is_some() {
            let results = self.queue.results();
            if let Some(p) = &mut self.pending {
                p.extend(results);
            }
            if let Some(s) = &mut self.seen_surface_forms {
                s.clear();
            }
        }
    }

    /// `get()`: the best `num` hits, highest score first.
    pub fn get(&mut self) -> Vec<SuggestScoreDoc> {
        let Some(pending) = &mut self.pending else {
            return self.queue.results();
        };
        pending.sort_by(|a, b| {
            float_compare(b.score, a.score)
                .then_with(|| charsequence_compare(&a.key, &b.key))
                .then_with(|| a.doc.cmp(&b.doc))
        });
        let mut seen: HashSet<&str> = HashSet::new();
        let mut hits = Vec::new();
        for hit in pending.iter() {
            if seen.insert(hit.key.as_str()) {
                hits.push(hit.clone());
                if hits.len() == self.num {
                    break;
                }
            }
        }
        hits
    }
}

/// What `CompletionScorer` knows about the segment.
pub struct SegmentView<'a> {
    /// `reader.numDocs()`.
    pub num_docs: i32,
    /// `reader.maxDoc()`.
    pub max_doc: i32,
    /// Whether a `BitsProducer` filter applies (`filtered`).
    pub filtered: bool,
    /// `CompletionScorer.accept(docID, acceptDocs)`: filter and live docs.
    pub accept: &'a dyn Fn(i32) -> bool,
}

/// `CompletionScorer.score(weight, boost)`.
fn score(weight: f32, boost: f32) -> f32 {
    if boost == 0.0 {
        weight
    } else if weight == 0.0 {
        boost
    } else {
        weight * boost
    }
}

fn surface_form(output: &[u8]) -> String {
    String::from_utf8_lossy(output).into_owned()
}

/// What `NRTSuggester.lookup` asks of a `CompletionWeight`: for each matched
/// path prefix `setNextMatch`, then the `boost()` and `context()` it set.
pub trait MatchWeight {
    /// `setNextMatch(pathPrefix)`.
    fn set_next_match(&mut self, path_prefix: &[i32]);
    /// `boost()`.
    fn boost(&self) -> f32;
    /// `context()`.
    fn context(&self) -> Option<String>;
}

/// A weight with boost 0 and no context: a query without contexts.
struct NoBoost;

impl MatchWeight for NoBoost {
    fn set_next_match(&mut self, _: &[i32]) {}
    fn boost(&self) -> f32 {
        0.0
    }
    fn context(&self) -> Option<String> {
        None
    }
}

/// `NRTSuggester.lookup(scorer, acceptDocs, collector)` for a query without
/// context (boost 0, no context).
pub fn lookup(
    suggester: &NrtSuggester,
    automaton: &Automaton,
    view: &SegmentView<'_>,
    collector: &mut TopSuggestDocsCollector,
) -> Result<()> {
    lookup_weighted(suggester, automaton, &mut NoBoost, view, collector)
}

/// `NRTSuggester.lookup(scorer, acceptDocs, collector)`: `weight` sets the
/// boost and context of each matched path prefix.
pub fn lookup_weighted(
    suggester: &NrtSuggester,
    automaton: &Automaton,
    weight: &mut dyn MatchWeight,
    view: &SegmentView<'_>,
    collector: &mut TopSuggestDocsCollector,
) -> Result<()> {
    // Java's `CollectionTerminatedException` ends one lookup, not the leaf
    collector.terminated = false;
    if view.num_docs <= 0 {
        return Ok(());
    }
    let live_docs_ratio = f64::from(view.num_docs) / f64::from(view.max_doc);
    let prefix_paths = intersect_prefix_paths(automaton, &suggester.fst)?;
    let top_n = collector.count_to_collect() * prefix_paths.len();
    let queue_size = max_top_n_searcher_queue_size(
        top_n,
        view.num_docs,
        live_docs_ratio,
        view.filtered,
        suggester.max_analyzed_paths_per_output,
    );
    let payload_sep = u8::try_from(suggester.payload_sep)
        .map_err(|_| Error::Corrupted(format!("payload separator {}", suggester.payload_sep)))?;
    let skip = collector.do_skip_duplicates();
    let state = RefCell::new(collector);

    let mut searcher = TopNSearcher::<NrtOutputs>::with_path_comparator(
        &suggester.fst,
        top_n,
        queue_size,
        |a: &NrtValue, b: &NrtValue| a.first.cmp(&b.first),
        |a: &FstPath<NrtValue>, b: &FstPath<NrtValue>| {
            let sa = score(decode_weight(a.output.first) as f32, a.boost);
            let sb = score(decode_weight(b.output.first) as f32, b.boost);
            float_compare(sb, sa).then_with(|| a.input.cmp(&b.input))
        },
    );
    searcher.accept_partial_path = Some(Box::new(|path: &mut FstPath<NrtValue>| {
        if state.borrow().is_terminated() {
            return false;
        }
        if skip {
            if path.payload == -1 {
                let arc_output = &path.arc.output().second;
                if let Some(i) = arc_output.iter().position(|&b| b == payload_sep) {
                    // the index lies within `output`, as Java asserts
                    path.payload = (path.output.second.len() - arc_output.len() + i) as i32;
                }
            }
            if path.payload != -1 {
                let surface = surface_form(&path.output.second[..path.payload as usize]);
                if state
                    .borrow()
                    .seen_surface_forms
                    .as_ref()
                    .is_some_and(|s| s.contains(&surface))
                {
                    return false;
                }
            }
        }
        true
    }));
    searcher.accept_result = Some(Box::new(|path: &FstPath<NrtValue>| {
        let mut st = state.borrow_mut();
        if st.is_terminated() {
            return false;
        }
        let output = &path.output.second;
        let sep_index = if path.payload != -1 {
            path.payload as usize
        } else {
            match output.iter().position(|&b| b == payload_sep) {
                Some(i) => i,
                None => return false,
            }
        };
        let surface = surface_form(&output[..sep_index]);
        let mut rest = SliceInput::new(&output[sep_index + 1..]);
        let Ok(doc_id) = rest.read_vint() else {
            return false;
        };
        if !(view.accept)(doc_id) {
            return false;
        }
        if let Some(seen) = &mut st.seen_surface_forms {
            if !seen.insert(surface.clone()) {
                return false;
            }
        }
        let s = score(decode_weight(path.output.first) as f32, path.boost);
        st.collect(doc_id, &surface, path.context.as_deref(), s);
        true
    }));

    for path in prefix_paths {
        weight.set_next_match(&path.input);
        let payload = if skip {
            path.output
                .second
                .iter()
                .position(|&b| b == payload_sep)
                .map_or(-1, |j| j as i32)
        } else {
            -1
        };
        searcher
            .add_start_paths_with(
                &path.fst_node,
                path.output.clone(),
                false,
                path.input.clone(),
                weight.boost(),
                weight.context(),
                payload,
            )
            .map_err(|e| Error::Corrupted(e.to_string()))?;
    }
    searcher
        .search()
        .map_err(|e| Error::Corrupted(e.to_string()))?;
    Ok(())
}

/// `NRTSuggester.getMaxTopNSearcherQueueSize`.
fn max_top_n_searcher_queue_size(
    top_n: usize,
    num_docs: i32,
    live_docs_ratio: f64,
    filter_enabled: bool,
    max_analyzed_paths_per_output: i32,
) -> usize {
    let mut max_queue_size =
        (top_n as i64).saturating_mul(i64::from(max_analyzed_paths_per_output));
    max_queue_size = (max_queue_size as f64 / live_docs_ratio) as i64;
    if filter_enabled {
        max_queue_size = max_queue_size.saturating_add(i64::from(num_docs / 2));
    }
    max_queue_size.min(MAX_TOP_N_QUEUE_SIZE) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::completion::{CompletionFieldWriter, TermDoc};
    use lucene_store::DataOutput;
    use lucene_util::automaton::automata::make_binary;

    fn doc(id: i32, surface: &str, weight: i32) -> TermDoc {
        let mut p = Vec::new();
        p.write_vint(surface.len() as i32);
        p.extend_from_slice(surface.as_bytes());
        p.write_vint(weight + 1);
        p.push(0);
        TermDoc {
            doc_id: id,
            payloads: vec![p],
        }
    }

    fn suggester(rows: &[(&str, i32, i32)]) -> NrtSuggester {
        let mut w = CompletionFieldWriter::new();
        let mut by_term: Vec<(&str, Vec<TermDoc>)> = Vec::new();
        for &(t, id, weight) in rows {
            match by_term.iter_mut().find(|(k, _)| *k == t) {
                Some((_, v)) => v.push(doc(id, t, weight)),
                None => by_term.push((t, vec![doc(id, t, weight)])),
            }
        }
        by_term.sort_by(|a, b| a.0.cmp(b.0));
        for (t, docs) in by_term {
            w.write_term(t.as_bytes(), &docs).unwrap();
        }
        let mut out = Vec::new();
        assert!(w.finish(&mut out).unwrap());
        NrtSuggester::load(&out).unwrap()
    }

    fn run(
        s: &NrtSuggester,
        prefix: &str,
        n: usize,
        dedup: bool,
        view: &SegmentView<'_>,
    ) -> Vec<(String, i32, i64)> {
        let mut c = TopSuggestDocsCollector::new(n, dedup).unwrap();
        lookup(s, &make_binary(prefix.as_bytes()), view, &mut c).unwrap();
        c.finish();
        c.get()
            .into_iter()
            .map(|h| (h.key, h.doc, h.score as i64))
            .collect()
    }

    fn view(num_docs: i32, max_doc: i32, accept: &dyn Fn(i32) -> bool) -> SegmentView<'_> {
        SegmentView {
            num_docs,
            max_doc,
            filtered: false,
            accept,
        }
    }

    #[test]
    fn hits_come_best_first_and_stop_at_num() {
        let s = suggester(&[("ab", 0, 5), ("ac", 1, 9), ("ad", 2, 7)]);
        let accept = |_: i32| true;
        let v = view(3, 3, &accept);
        assert_eq!(
            run(&s, "a", 2, false, &v),
            vec![("ac".into(), 1, 9), ("ad".into(), 2, 7)]
        );
        assert!(run(&s, "zz", 2, false, &v).is_empty());
    }

    #[test]
    fn a_rejected_document_is_skipped_and_a_filter_widens_the_queue() {
        let s = suggester(&[("ab", 0, 5), ("ac", 1, 9)]);
        let accept = |d: i32| d != 1;
        let mut v = view(2, 4, &accept);
        v.filtered = true;
        assert_eq!(run(&s, "a", 5, false, &v), vec![("ab".into(), 0, 5)]);
        let none = view(0, 4, &accept);
        assert!(run(&s, "a", 5, false, &none).is_empty());
    }

    #[test]
    fn duplicates_collapse_to_the_best_document() {
        let s = suggester(&[("ab", 0, 5), ("ab", 1, 8), ("ab", 2, 6), ("ac", 3, 1)]);
        let accept = |_: i32| true;
        let v = view(4, 4, &accept);
        assert_eq!(
            run(&s, "a", 3, true, &v),
            vec![("ab".into(), 1, 8), ("ac".into(), 3, 1)]
        );
        assert_eq!(run(&s, "a", 3, false, &v).len(), 3);
    }

    #[test]
    fn a_full_collector_ends_the_segment() {
        let s = suggester(&[("a1", 0, 9), ("a2", 1, 8), ("a3", 2, 7), ("a4", 3, 6)]);
        let accept = |_: i32| true;
        let v = view(4, 4, &accept);
        assert_eq!(run(&s, "a", 1, false, &v), vec![("a1".into(), 0, 9)]);
    }

    /// `PriorityQueue.insertWithOverflow` replaces the top only with a
    /// strictly greater element: a hit equal to the last one that fit (the
    /// same document and key under another context) overflows and ends the
    /// segment, so the first of equals stays.
    #[test]
    fn an_equal_hit_does_not_replace_the_last_that_fit() {
        let mut c = TopSuggestDocsCollector::new(1, false).unwrap();
        c.set_next_reader(0);
        c.collect(7, "x", Some("first"), 2.0);
        assert!(!c.is_terminated());
        c.collect(7, "x", Some("second"), 2.0);
        assert!(c.is_terminated());
        assert_eq!(c.get()[0].context.as_deref(), Some("first"));
    }

    #[test]
    fn collector_validates_and_merges_segments() {
        assert!(TopSuggestDocsCollector::new(0, false).is_err());
        let mut c = TopSuggestDocsCollector::new(2, true).unwrap();
        assert!(c.do_skip_duplicates());
        c.set_next_reader(0);
        c.collect(1, "x", None, 3.0);
        c.collect(2, "y", Some("ctx"), 1.0);
        c.finish();
        c.set_next_reader(100);
        assert!(!c.is_terminated());
        c.collect(1, "x", None, 3.0);
        c.finish();
        let hits = c.get();
        assert_eq!(hits.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(hits[1].context.as_deref(), Some("ctx"));
        let mut plain = TopSuggestDocsCollector::new(1, false).unwrap();
        plain.collect(0, "a", None, 1.0);
        plain.collect(1, "b", None, 2.0);
        plain.collect(2, "c", None, 0.5);
        assert!(plain.is_terminated());
        plain.set_next_reader(0);
        assert!(!plain.is_terminated());
        assert_eq!(plain.get()[0].key, "b");
    }

    #[test]
    fn scoring_follows_the_scorer() {
        assert_eq!(score(5.0, 0.0), 5.0);
        assert_eq!(score(0.0, 3.0), 3.0);
        assert_eq!(score(2.0, 3.0), 6.0);
        assert_eq!(max_top_n_searcher_queue_size(10, 100, 0.5, true, 3), 110);
        assert_eq!(
            max_top_n_searcher_queue_size(10_000, 100, 1.0, false, 3),
            5000
        );
    }

    #[test]
    fn ties_leave_the_queue_by_key_then_document() {
        let s = suggester(&[("a1", 5, 4), ("a2", 3, 4), ("a1", 2, 4)]);
        let accept = |_: i32| true;
        let v = view(3, 3, &accept);
        let got = run(&s, "a", 3, false, &v);
        assert_eq!(got.len(), 3);
        assert!(got[0].0 <= got[1].0 && got[1].0 <= got[2].0);
    }
}
