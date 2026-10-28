//! `org.apache.lucene.facet.sortedset`: facet counting over the
//! `SORTED_SET` doc values [`super::SortedSetDocValuesFacetField`] indexes,
//! with no taxonomy.
//!
//! [`DefaultSortedSetDocValuesReaderState`] reads a reader's facet field
//! once -- its terms across the leaves, through an [`OrdinalMap`] when
//! there is more than one -- into the dim layer ([`FacetsState`]: each flat
//! dim's ordinal range, each hierarchical dim's tree).
//! [`SortedSetDocValuesFacetCounts`] and
//! [`ConcurrentSortedSetDocValuesFacetCounts`] count every live document or
//! a [`FacetsCollector`]'s hits into the state's global ordinals, and answer
//! the [`Facets`] calls through [`SortedSetFacetCounts`] (Java's
//! `AbstractSortedSetDocValueFacetCounts`).
//!
//! Differs:
//! * the state holds the terms decoded (the dim layer's dictionary) rather
//!   than a `MultiSortedSetDocValues` it reads them back through; it is not
//!   an `Accountable` (`ramBytesUsed`/`getChildResources` are not ported);
//! * a field whose leaves hold more than `Integer.MAX_VALUE` terms is refused
//!   before the state is built, as Java refuses it, but a term that is not
//!   UTF-8 is decoded lossily where Java's `utf8ToString` substitutes too;
//! * Java's `SortedSetDocValues.unwrapSingleton` fast path (a field the codec
//!   stored one value per document) is not taken: every leaf is read as a
//!   set, which counts the same;
//! * the concurrent counter counts the segments on the rayon pool, not on a
//!   caller's `ExecutorService`, and each task into counts of its own that are
//!   summed at the end rather than into one shared `AtomicIntegerArray` (the
//!   same sums: the additions wrap either way).

use super::facet_utils::{for_each_hit, for_each_hit_in};
use super::facets_collector::{FacetsCollector, MatchingDocs};
use super::{
    DimTree, FacetResult, FacetValue, Facets, FacetsConfig, FacetsState, FacetsStateError,
    OrdRange, SortedSetFacetCounts, DEFAULT_INDEX_FIELD_NAME,
};
use crate::ordinal_map::OrdinalMap;
use crate::reader::doc_values::get_sorted_set;
use crate::reader::{IndexReader, LeafReaderContext, SortedSetDocValues, NO_MORE_DOCS};
use crate::{Error, Result};

/// The message of the counters' `IllegalStateException` for hits from
/// another reader.
pub(crate) const STATE_MISMATCH: &str =
    "the SortedSetDocValuesReaderState provided to this class does not match the reader being \
     searched; you must create a new SortedSetDocValuesReaderState every time you open a new \
     IndexReader";

/// `DefaultSortedSetDocValuesReaderState`: the dims of one reader's
/// `SORTED_SET` facet field, built once per reader.
pub struct DefaultSortedSetDocValuesReaderState<'r> {
    reader: &'r dyn IndexReader,
    field: String,
    state: FacetsState,
    /// `MultiSortedSetDocValues.mapping`: `Some` for more than one leaf.
    ordinal_map: Option<OrdinalMap>,
}

impl<'r> DefaultSortedSetDocValuesReaderState<'r> {
    /// `new DefaultSortedSetDocValuesReaderState(reader, config)`: the
    /// default index field, `$facets`.
    ///
    /// # Errors
    /// As [`Self::with_field`].
    pub fn new(reader: &'r dyn IndexReader, config: &FacetsConfig) -> Result<Self> {
        Self::with_field(reader, DEFAULT_INDEX_FIELD_NAME, config)
    }

    /// `new DefaultSortedSetDocValuesReaderState(reader, field, config)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when no leaf holds `field` as `SORTED_SET`
    /// doc values (`field "f" was not indexed with SortedSetDocValues`), for
    /// more than `Integer.MAX_VALUE` terms, and for a flat dim holding a
    /// deeper path; whatever reading the terms reports.
    pub fn with_field(
        reader: &'r dyn IndexReader,
        field: &str,
        config: &FacetsConfig,
    ) -> Result<Self> {
        let not_indexed = || {
            Error::IllegalArgument(format!(
                "field \"{field}\" was not indexed with SortedSetDocValues"
            ))
        };
        let leaves = reader.leaves();
        let (labels, ordinal_map) = match leaves.as_slice() {
            [] => return Err(not_indexed()),
            [leaf] => {
                let Some(mut dv) = leaf.reader.sorted_set_doc_values(field)? else {
                    return Err(not_indexed());
                };
                check_value_count(dv.value_count())?;
                (all_terms(dv.as_mut())?, None)
            }
            _ => {
                // `MultiDocValues.getSortedSetValues`: `null` unless some
                // leaf holds the field as `SORTED_SET`.
                let mut any_real = false;
                let mut terms = Vec::with_capacity(leaves.len());
                for leaf in &leaves {
                    match leaf.reader.sorted_set_doc_values(field)? {
                        Some(mut dv) => {
                            any_real = true;
                            terms.push(all_terms(dv.as_mut())?);
                        }
                        None => terms.push(Vec::new()),
                    }
                }
                if !any_real {
                    return Err(not_indexed());
                }
                let map = OrdinalMap::build(&terms);
                check_value_count(map.value_count())?;
                let mut labels =
                    Vec::with_capacity(usize::try_from(map.value_count()).unwrap_or(0));
                let mut global = 0i64;
                while global < map.value_count() {
                    // Each segment term is the first of exactly one global
                    // ordinal, so it moves out rather than being copied.
                    let term = map
                        .first_segment(global)
                        .zip(map.first_segment_ord(global))
                        .and_then(|(seg, ord)| {
                            terms.get_mut(seg)?.get_mut(usize::try_from(ord).ok()?)
                        })
                        .ok_or_else(|| bad_ord(global))?;
                    labels.push(std::mem::take(term));
                    global = global.saturating_add(1);
                }
                (labels, Some(map))
            }
        };
        let state = FacetsState::new(labels, config.clone()).map_err(state_error)?;
        Ok(DefaultSortedSetDocValuesReaderState {
            reader,
            field: field.to_string(),
            state,
            ordinal_map,
        })
    }

    /// `getField()`.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// `getReader()`.
    pub fn reader(&self) -> &'r dyn IndexReader {
        self.reader
    }

    /// `getSize()`: the number of distinct labels.
    pub fn size(&self) -> i32 {
        i32::try_from(self.state.size()).unwrap_or(i32::MAX)
    }

    /// `getFacetsConfig()`.
    pub fn facets_config(&self) -> &FacetsConfig {
        self.state.config()
    }

    /// `getDims()`: the hierarchical dims, then the flat ones (see
    /// [`FacetsState::dims`] for the order).
    pub fn dims(&self) -> impl Iterator<Item = &str> {
        self.state.dims()
    }

    /// `getOrdRange(dim)`: a flat dim's ordinals, `None` when it was never
    /// indexed.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a dim configured hierarchical.
    pub fn ord_range(&self, dim: &str) -> Result<Option<OrdRange>> {
        if self.facets_config().dim_config(dim).hierarchical {
            return Err(Error::Unsupported(
                "This operation is only supported for flat dimensions".into(),
            ));
        }
        Ok(self.state.ord_range(dim).copied())
    }

    /// `getPrefixToOrdRange()`: every flat dim's ordinals.
    pub fn prefix_to_ord_range(&self) -> impl Iterator<Item = (&str, OrdRange)> {
        self.state
            .dims()
            .filter_map(|d| self.state.ord_range(d).map(|r| (d, *r)))
    }

    /// `getDimTree(dim)`: a hierarchical dim's tree, `None` when it was
    /// never indexed.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a dim not configured hierarchical.
    pub fn dim_tree(&self, dim: &str) -> Result<Option<&DimTree>> {
        if !self.facets_config().dim_config(dim).hierarchical {
            return Err(Error::Unsupported(
                "This operation is only supported for hierarchical facets".into(),
            ));
        }
        Ok(self.state.dim_tree(dim))
    }

    /// The dim layer itself.
    pub fn facets_state(&self) -> &FacetsState {
        &self.state
    }

    /// `MultiSortedSetDocValues.mapping`: the leaves' ordinals to the
    /// state's, `None` for a single leaf.
    pub fn ordinal_map(&self) -> Option<&OrdinalMap> {
        self.ordinal_map.as_ref()
    }

    /// `ReaderUtil.getTopLevelContext(context).reader() == reader`: the leaf
    /// is this state's reader's.
    pub(crate) fn validate(&self, context: &LeafReaderContext<'_>) -> Result<()> {
        let same = self
            .reader
            .leaves()
            .get(context.ord)
            .is_some_and(|l| std::ptr::addr_eq(l.reader, context.reader));
        if same {
            Ok(())
        } else {
            Err(Error::IllegalState(STATE_MISMATCH.into()))
        }
    }
}

impl std::fmt::Debug for DefaultSortedSetDocValuesReaderState<'_> {
    /// `toString()`, the reader as its leaf count.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DefaultSortedSetDocValuesReaderState(field={} reader={} leaves)",
            self.field,
            self.reader.leaves().len()
        )
    }
}

/// The constructor's `valueCount > Integer.MAX_VALUE` refusal.
fn check_value_count(n: i64) -> Result<()> {
    if n > i64::from(i32::MAX) {
        return Err(Error::IllegalArgument(format!(
            "can only handle valueCount < Integer.MAX_VALUE; got {n}"
        )));
    }
    Ok(())
}

/// Every term of `dv`, in ordinal order, as `utf8ToString` decodes it.
fn all_terms(dv: &mut (dyn SortedSetDocValues + '_)) -> Result<Vec<String>> {
    let n = dv.value_count();
    let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    let mut ord = 0i64;
    while ord < n {
        // One allocation per term: the looked-up bytes become the string
        // unless they are not UTF-8 (Java's `utf8ToString` replaces those).
        let bytes = dv.lookup_ord(ord)?;
        out.push(
            String::from_utf8(bytes)
                .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()),
        );
        ord = ord.saturating_add(1);
    }
    Ok(out)
}

/// The state builder's refusals, as Java words them.
fn state_error(e: FacetsStateError) -> Error {
    match e {
        FacetsStateError::NotHierarchical { path, term } => Error::IllegalArgument(format!(
            "dimension not configured to handle hierarchical field; got: [{}] {term}",
            path.join(", ")
        )),
        other => Error::IllegalArgument(other.to_string()),
    }
}

fn bad_ord(ord: i64) -> Error {
    Error::IllegalState(format!("ordinal {ord} is out of bounds"))
}

// ---------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------

/// Where a segment's counts go: an `int[]` of the state's ordinals.
trait CountSink {
    /// `counts[ord] += n`.
    fn add(&mut self, ord: i64, n: i32) -> Result<()>;
}

impl CountSink for [i32] {
    #[inline]
    fn add(&mut self, ord: i64, n: i32) -> Result<()> {
        let slot = usize::try_from(ord)
            .ok()
            .and_then(|o| self.get_mut(o))
            .ok_or_else(|| bad_ord(ord))?;
        *slot = slot.wrapping_add(n);
        Ok(())
    }
}

/// The documents to count of a segment (of its documents in `range` only,
/// when given), each handed to `f` with `values` positioned on it: the hits
/// that have values, or (`hits` `None`) every live document that has
/// (`countOneSegmentNHLD` without deletions).
#[inline]
fn for_each_counted(
    hits: Option<&MatchingDocs<'_>>,
    context: &LeafReaderContext<'_>,
    range: Option<(i32, i32)>,
    values: &mut (dyn SortedSetDocValues + '_),
    mut f: impl FnMut(&mut (dyn SortedSetDocValues + '_)) -> Result<()>,
) -> Result<()> {
    if let Some(hits) = hits {
        return match range {
            None => for_each_hit(&hits.bits, values, f),
            Some((lo, hi)) => for_each_hit_in(&hits.bits, values, lo, hi, f),
        };
    }
    let live = context.reader.live_docs();
    let (mut doc, hi) = match range {
        None => (values.next_doc()?, NO_MORE_DOCS),
        Some((lo, hi)) => (values.advance(lo.max(0))?, hi),
    };
    while doc < hi && doc != NO_MORE_DOCS {
        if live.is_none_or(|l| l.get_doc(doc)) {
            f(values)?;
        }
        doc = values.next_doc()?;
    }
    Ok(())
}

/// `f(ord)` for each ordinal of the current document, in bulk when `dv`
/// holds them decoded.
#[inline]
fn for_each_ord(
    dv: &mut (dyn SortedSetDocValues + '_),
    mut f: impl FnMut(i64) -> Result<()>,
) -> Result<()> {
    if let Some(ords) = dv.take_ords() {
        for &ord in ords {
            f(ord)?;
        }
        return Ok(());
    }
    for _ in 0..dv.doc_value_count() {
        f(dv.next_ord()?)?;
    }
    Ok(())
}

/// `countOneSegment` / `countOneSegmentNHLD` / `CountOneSegment.call`: one
/// segment's hits (or live documents) into `counts`, through the ordinal map
/// when one is in use -- remapping each ordinal as it comes when the hits
/// are few against the segment's terms, else counting in segment ordinals
/// first. With a `range`, only the segment's documents in it.
fn count_one_segment<S: CountSink + ?Sized>(
    field: &str,
    ordinal_map: Option<&OrdinalMap>,
    context: &LeafReaderContext<'_>,
    hits: Option<&MatchingDocs<'_>>,
    range: Option<(i32, i32)>,
    counts: &mut S,
) -> Result<()> {
    let mut values = get_sorted_set(context.reader, field)?;
    let values = values.as_mut();
    let Some(map) = ordinal_map else {
        // No ordinal map (one segment): segment ordinals are global.
        return for_each_counted(hits, context, range, values, |dv| {
            for_each_ord(dv, |ord| counts.add(ord, 1))
        });
    };
    let seg_ords = map.segment_ords(context.ord).ok_or_else(|| {
        Error::IllegalState(format!("leaf {} is not in the ordinal map", context.ord))
    })?;
    let num_seg_ords = i32::try_from(values.value_count()).unwrap_or(i32::MAX);
    let global = |ord: i64| -> Result<i64> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| seg_ords.get(o))
            .copied()
            .ok_or_else(|| bad_ord(ord))
    };
    if hits.is_some_and(|h| h.total_hits < num_seg_ords / 10) {
        // Remap every ordinal to its global one as it comes.
        return for_each_counted(hits, context, range, values, |dv| {
            for_each_ord(dv, |ord| counts.add(global(ord)?, 1))
        });
    }
    // Count in segment ordinals first, then migrate to global ones.
    let mut seg_counts = vec![0i32; usize::try_from(num_seg_ords).unwrap_or(0)];
    for_each_counted(hits, context, range, values, |dv| {
        for_each_ord(dv, |ord| seg_counts.as_mut_slice().add(ord, 1))
    })?;
    for (ord, &count) in (0i64..).zip(&seg_counts) {
        if count != 0 {
            counts.add(global(ord)?, count)?;
        }
    }
    Ok(())
}

/// The fewest documents a chunk of a segment counted on its own task holds:
/// below it a task's own count array and start-up cost more than the
/// documents.
const MIN_CHUNK_DOCS: i32 = 1 << 16;

/// One counting task: a segment, its hits, and the documents of it to count
/// (`None`: all of them).
type Chunk<'t, 'c, 'h> = (
    &'t LeafReaderContext<'c>,
    Option<&'t MatchingDocs<'h>>,
    Option<(i32, i32)>,
);

/// The segments to count as tasks for `threads` threads: each segment whole
/// while there are as many segments as threads, else every segment of at
/// least two `min_chunk`s ([`MIN_CHUNK_DOCS`]) cut into document ranges
/// (whole 64-document words of the hits) of about `total / threads`
/// documents each, so that a single large segment is counted on every
/// thread too.
fn split_into_chunks<'t, 'c, 'h>(
    todo: &[&'t (LeafReaderContext<'c>, Option<&'t MatchingDocs<'h>>)],
    threads: usize,
    min_chunk: i32,
) -> Vec<Chunk<'t, 'c, 'h>> {
    let whole = || todo.iter().map(|(c, h)| (c, *h, None)).collect::<Vec<_>>();
    if threads <= 1 || todo.len() >= threads {
        return whole();
    }
    let total: i64 = todo
        .iter()
        .map(|(c, _)| i64::from(c.reader.max_doc()))
        .sum();
    let per_thread = total / i64::try_from(threads).unwrap_or(1).max(1);
    // Rounded up to whole words of the hits' bits.
    let size = i32::try_from(per_thread)
        .unwrap_or(i32::MAX)
        .max(min_chunk)
        .saturating_add(63)
        & !63;
    let mut out = Vec::new();
    for (context, hits) in todo.iter().map(|t| (&t.0, t.1)) {
        let max_doc = context.reader.max_doc();
        if max_doc < min_chunk.saturating_mul(2) {
            out.push((context, hits, None));
            continue;
        }
        let mut lo = 0i32;
        while lo < max_doc {
            let hi = lo.saturating_add(size).min(max_doc);
            out.push((context, hits, Some((lo, hi))));
            lo = hi;
        }
    }
    out
}

/// `a[i] += b[i]`, wrapping as Java's `int` sums.
fn add_counts(a: &mut [i32], b: &[i32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = x.wrapping_add(y);
    }
}

/// `count(matchingDocs)`'s ordinal map: only when the hits span more than
/// one segment ("is this right?" -- Java's own TODO).
fn hits_map<'m>(
    state: &'m DefaultSortedSetDocValuesReaderState<'_>,
    matching: &[MatchingDocs<'_>],
) -> Option<&'m OrdinalMap> {
    if matching.len() > 1 {
        state.ordinal_map()
    } else {
        None
    }
}

/// `SortedSetDocValuesFacetCounts`: counts of a [`FacetsCollector`]'s hits
/// (or of every live document) per label of a
/// [`DefaultSortedSetDocValuesReaderState`].
pub struct SortedSetDocValuesFacetCounts<'s> {
    inner: SortedSetFacetCounts<'s>,
}

impl<'s> SortedSetDocValuesFacetCounts<'s> {
    /// `new SortedSetDocValuesFacetCounts(state)`: every live document of
    /// the state's reader.
    ///
    /// # Errors
    /// Whatever reading the doc values reports.
    pub fn new(state: &'s DefaultSortedSetDocValuesReaderState<'_>) -> Result<Self> {
        let mut counts: Option<Vec<i32>> = None;
        for leaf in state.reader().leaves() {
            let c = counts.get_or_insert_with(|| new_counts(state));
            count_one_segment(
                state.field(),
                state.ordinal_map(),
                &leaf,
                None,
                None,
                c.as_mut_slice(),
            )?;
        }
        Ok(Self::over(state, counts))
    }

    /// `new SortedSetDocValuesFacetCounts(state, hits)`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] when `hits` were not collected from the
    /// state's reader; whatever reading the doc values reports.
    pub fn with_hits(
        state: &'s DefaultSortedSetDocValuesReaderState<'_>,
        hits: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let matching = hits.matching_docs();
        let map = hits_map(state, matching);
        let mut counts: Option<Vec<i32>> = None;
        for m in matching {
            state.validate(&m.context)?;
            if m.total_hits == 0 {
                continue;
            }
            let c = counts.get_or_insert_with(|| new_counts(state));
            count_one_segment(
                state.field(),
                map,
                &m.context,
                Some(m),
                None,
                c.as_mut_slice(),
            )?;
        }
        Ok(Self::over(state, counts))
    }

    /// Over counts already taken (`None`: nothing was counted).
    pub(crate) fn over(
        state: &'s DefaultSortedSetDocValuesReaderState<'_>,
        counts: Option<Vec<i32>>,
    ) -> Self {
        SortedSetDocValuesFacetCounts {
            inner: SortedSetFacetCounts::from_counts(state.facets_state(), counts),
        }
    }
}

/// `new int[state.getSize()]`.
fn new_counts(state: &DefaultSortedSetDocValuesReaderState<'_>) -> Vec<i32> {
    vec![0; state.facets_state().size()]
}

/// The [`Facets`] calls, as `AbstractSortedSetDocValueFacetCounts` answers
/// them over the counts.
macro_rules! delegate_facets {
    ($t:ty) => {
        impl Facets for $t {
            fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
                self.inner.get_all_children(dim, path)
            }

            fn get_top_children(
                &self,
                top_n: i32,
                dim: &str,
                path: &[&str],
            ) -> Result<Option<FacetResult>> {
                self.inner.get_top_children(top_n, dim, path)
            }

            fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
                self.inner.get_specific_value(dim, path)
            }

            fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
                self.inner.get_all_dims(top_n)
            }

            fn get_top_dims(
                &self,
                top_n_dims: i32,
                top_n_children: i32,
            ) -> Result<Vec<FacetResult>> {
                self.inner.get_top_dims(top_n_dims, top_n_children)
            }
        }
    };
}

delegate_facets!(SortedSetDocValuesFacetCounts<'_>);
delegate_facets!(ConcurrentSortedSetDocValuesFacetCounts<'_>);

/// `ConcurrentSortedSetDocValuesFacetCounts`: as
/// [`SortedSetDocValuesFacetCounts`], each segment counted on its own task
/// into one shared array -- so the counts exist (zeros) even when nothing
/// matched.
pub struct ConcurrentSortedSetDocValuesFacetCounts<'s> {
    inner: SortedSetFacetCounts<'s>,
}

impl<'s> ConcurrentSortedSetDocValuesFacetCounts<'s> {
    /// `new ConcurrentSortedSetDocValuesFacetCounts(state, exec)`: every
    /// live document.
    ///
    /// # Errors
    /// Whatever reading the doc values reports.
    pub fn new(state: &'s DefaultSortedSetDocValuesReaderState<'_>) -> Result<Self> {
        let leaves = state.reader().leaves();
        let tasks: Vec<(LeafReaderContext<'_>, Option<&MatchingDocs<'_>>)> =
            leaves.into_iter().map(|l| (l, None)).collect();
        Self::run(state, state.ordinal_map(), &tasks)
    }

    /// `new ConcurrentSortedSetDocValuesFacetCounts(state, hits, exec)`.
    ///
    /// # Errors
    /// As [`SortedSetDocValuesFacetCounts::with_hits`].
    pub fn with_hits(
        state: &'s DefaultSortedSetDocValuesReaderState<'_>,
        hits: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let matching = hits.matching_docs();
        let mut tasks = Vec::with_capacity(matching.len());
        for m in matching {
            state.validate(&m.context)?;
            tasks.push((m.context, Some(m)));
        }
        Self::run(state, hits_map(state, matching), &tasks)
    }

    fn run(
        state: &'s DefaultSortedSetDocValuesReaderState<'_>,
        map: Option<&OrdinalMap>,
        tasks: &[(LeafReaderContext<'_>, Option<&MatchingDocs<'_>>)],
    ) -> Result<Self> {
        Self::run_chunked(
            state,
            map,
            tasks,
            rayon::current_num_threads(),
            MIN_CHUNK_DOCS,
        )
    }

    /// [`Self::run`] for `threads` threads, a chunk holding at least
    /// `min_chunk` documents.
    fn run_chunked(
        state: &'s DefaultSortedSetDocValuesReaderState<'_>,
        map: Option<&OrdinalMap>,
        tasks: &[(LeafReaderContext<'_>, Option<&MatchingDocs<'_>>)],
        threads: usize,
        min_chunk: i32,
    ) -> Result<Self> {
        use rayon::prelude::*;
        let n = state.facets_state().size();
        let todo: Vec<&(LeafReaderContext<'_>, Option<&MatchingDocs<'_>>)> = tasks
            .iter()
            .filter(|(_, hits)| !hits.is_some_and(|h| h.total_hits == 0))
            .collect();
        let chunks = split_into_chunks(&todo, threads, min_chunk);
        // Each task counts into an array of its own (no atomic per ordinal),
        // the arrays summed at the end (wrapping, so in any order); one task
        // runs on this thread.
        let counts = if chunks.len() <= 1 {
            let mut counts = vec![0i32; n];
            for &(context, hits, range) in &chunks {
                count_one_segment(
                    state.field(),
                    map,
                    context,
                    hits,
                    range,
                    counts.as_mut_slice(),
                )?;
            }
            counts
        } else {
            chunks
                .par_iter()
                .try_fold(
                    || vec![0i32; n],
                    |mut counts, &(context, hits, range)| {
                        count_one_segment(
                            state.field(),
                            map,
                            context,
                            hits,
                            range,
                            counts.as_mut_slice(),
                        )?;
                        Ok::<_, Error>(counts)
                    },
                )
                .try_reduce(
                    || vec![0i32; n],
                    |mut a, b| {
                        add_counts(&mut a, &b);
                        Ok(a)
                    },
                )?
        };
        Ok(ConcurrentSortedSetDocValuesFacetCounts {
            inner: SortedSetFacetCounts::from_counts(state.facets_state(), Some(counts)),
        })
    }
}

#[cfg(test)]
mod tests;
