//! `org.apache.lucene.search.suggest.document`: the lookup half
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

/// `TopSuggestDocs.merge(topN, shardHits)`: the best `top_n` hits of every
/// shard's, best first. Each shard's hits (best first) are offered in turn
/// until the queue turns one away, which ends that shard. Java's
/// `TopSuggestDocs` is the returned `Vec`: its `totalHits` is the length,
/// `EQUAL_TO`.
pub fn merge(top_n: usize, shard_hits: &[&[SuggestScoreDoc]]) -> Vec<SuggestScoreDoc> {
    let mut queue = SuggestQueue::new(top_n);
    for shard in shard_hits {
        for hit in shard.iter() {
            if queue.insert_with_overflow(hit.clone()) {
                break;
            }
        }
    }
    queue.results()
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
    fn context(&self) -> Option<std::rc::Rc<str>>;
}

/// A weight with boost 0 and no context: a query without contexts.
struct NoBoost;

impl MatchWeight for NoBoost {
    fn set_next_match(&mut self, _: &[i32]) {}
    fn boost(&self) -> f32 {
        0.0
    }
    fn context(&self) -> Option<std::rc::Rc<str>> {
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
                // Java searches the last arc's output, the rest of `output`
                // being known to hold no separator; the arc's output is not
                // kept here, so the search covers `output`, to the same
                // first separator
                if let Some(i) = path.output.second.iter().position(|&b| b == payload_sep) {
                    path.payload = i as i32;
                }
            }
            if path.payload != -1 {
                let surface = String::from_utf8_lossy(&path.output.second[..path.payload as usize]);
                if state
                    .borrow()
                    .seen_surface_forms
                    .as_ref()
                    .is_some_and(|s| s.contains(surface.as_ref()))
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
                path.output,
                false,
                path.input,
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
mod tests;
