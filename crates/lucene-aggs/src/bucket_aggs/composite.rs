//! Composite aggregation collectors: the source-by-source collection of a `composite` aggregation.

use super::*;

/// `CompositeAggregator`'s collection of one document: every combination of
/// its sources' values (`missing_bucket`'s missing one for a source without
/// any) past `after` and competitive, counted and passed to the
/// sub-aggregations. A segment whose first source is read through its points
/// or terms counts the document once per distinct first-source value; the
/// sub-aggregations, replayed from the doc values, see every combination.
/// `lead`: the first source's segment ordinal, in place of the document's
/// own values (the sorted read's term). Returns the competitive keys placed,
/// or `None` when the document has no key to offer (a source without a value
/// and without `missing_bucket`).
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_composite(
    sources: &[CompositeSource],
    size: usize,
    subs: &[AggNode],
    st: &mut CompositeState,
    c: &mut CompositeLeaf<'_>,
    leaf_subs: &mut [Leaf<'_>],
    doc: i32,
    lead: Option<i64>,
) -> Result<Option<usize>> {
    if size == 0 {
        return Ok(None);
    }
    let mut placed = 0usize;
    for (i, ((s, l), values)) in sources
        .iter()
        .zip(&mut c.sources)
        .zip(&mut c.values)
        .enumerate()
    {
        values.clear();
        if let (0, Some(ord)) = (i, lead) {
            let map = match &l.col {
                Col::Ords(_, map) => *map,
                _ => &[],
            };
            values.push(Some(s.key(map, ord)?));
        } else if l.read(doc)? {
            let map = match &l.col {
                Col::Ords(_, map) => *map,
                _ => &[],
            };
            for &v in &l.buf {
                values.push(Some(s.key(map, v)?));
            }
        } else if s.missing_bucket {
            values.push(None);
        } else {
            return Ok(None);
        }
    }
    if let ([s], [values]) = (sources, c.values.as_slice()) {
        // One source: its keys in turn, without the combinations' bookkeeping.
        for (i, &k) in values.iter().enumerate() {
            if c.after
                .as_ref()
                .is_some_and(|a| !s.compare_after(k, a[0]).is_gt())
                || st
                    .worst
                    .as_deref()
                    .is_some_and(|w| s.compare(k, w[0]).is_gt())
            {
                continue;
            }
            // A repeated doc-values value counts again; a sorted read does not.
            let count = !c.sorted_lead || i == 0 || values[i - 1] != k;
            place(sources, size, subs, st, &[k], count, leaf_subs, doc)?;
            placed += 1;
        }
        return Ok(Some(placed));
    }
    let at = &mut c.at;
    at.clear();
    at.extend(std::iter::repeat_n(0, sources.len()));
    loop {
        let lead = c.values[0][at[0]];
        // A first-source key before `after`'s, or past the worst kept one,
        // rules out every combination it leads.
        let hopeless = c
            .after
            .as_ref()
            .is_some_and(|a| s0_before_after(&sources[0], lead, a[0]))
            || st
                .worst
                .as_deref()
                .is_some_and(|w| sources[0].compare(lead, w[0]).is_gt());
        if !hopeless {
            c.key.clear();
            c.key.extend(at.iter().zip(&c.values).map(|(&i, v)| v[i]));
            let past_after = c.after.as_ref().is_none_or(|after| {
                for ((s, &k), &a) in sources.iter().zip(&c.key).zip(after) {
                    let o = s.compare_after(k, a);
                    if o.is_ne() {
                        return o.is_gt();
                    }
                }
                false
            });
            let competitive = past_after
                && st
                    .worst
                    .as_deref()
                    .is_none_or(|w| compare_keys(sources, &c.key, w).is_le());
            if competitive {
                let lead_run = at[0] == 0 || c.values[0][at[0] - 1] != lead;
                let count = !c.sorted_lead || lead_run;
                place(sources, size, subs, st, &c.key, count, leaf_subs, doc)?;
                placed += 1;
            }
        }
        // The next combination, the last source fastest.
        let mut i = sources.len();
        loop {
            if i == 0 {
                return Ok(Some(placed));
            }
            i -= 1;
            at[i] += 1;
            if at[i] < c.values[i].len() {
                break;
            }
            at[i] = 0;
        }
    }
}

/// A table slot not yet settled, and one that is never competitive.
pub(super) const PAIR_UNSEEN: u32 = u32::MAX;
pub(super) const PAIR_DEAD: u32 = u32::MAX - 1;

/// Largest first-source dictionary the two-keyword window pass tabulates.
pub(super) const MAX_TABULATED_ORDS: usize = 1 << 16;

/// Whether no key led by the first-source segment ordinal `ord` can be
/// competitive: before `after`'s first part, or past the worst kept key's.
pub(super) fn first_hopeless(
    s0: &CompositeSource,
    after: Option<&[AfterPart]>,
    st: &CompositeState,
    map0: &[i64],
    ord: i64,
) -> Result<bool> {
    let k0 = Some(s0.key(map0, ord)?);
    Ok(after.is_some_and(|a| s0_before_after(s0, k0, a[0]))
        || st
            .worst
            .as_deref()
            .is_some_and(|w| s0.compare(k0, w[0]).is_gt()))
}

/// A window of a `composite` over two keyword sources, both read ahead as
/// single-valued columns: the documents' ordinals are read straight from the
/// windows, a document whose first key cannot be competitive (before `after`,
/// or past the worst kept key) is dropped without building its key, and the
/// rest are placed as [`collect_composite`] places them. A document missing
/// either value goes through [`collect_composite`]. False when the window is
/// not of that shape, with nothing collected.
pub(super) fn collect_composite_pairs(
    sources: &[CompositeSource],
    size: usize,
    subs: &[AggNode],
    st: &mut CompositeState,
    leaf: &mut Leaf<'_>,
    docs: &[i32],
) -> Result<bool> {
    let Some(c) = leaf.composite.as_deref_mut() else {
        return Ok(false);
    };
    let [s0, s1] = sources else {
        return Ok(false);
    };
    let [l0, l1] = c.sources.as_slice() else {
        return Ok(false);
    };
    let (Col::Ords(_, map0), Col::Ords(_, map1)) = (&l0.col, &l1.col) else {
        return Ok(false);
    };
    let (map0, map1) = (*map0, *map1);
    let reads_ahead = |l: &Leaf<'_>| l.pre_len > 0 && !l.pre_multi && l.missing.is_none();
    if size == 0
        || c.sorted_lead
        || s0.values != CompositeValues::Keyword
        || s1.values != CompositeValues::Keyword
        || !reads_ahead(l0)
        || !reads_ahead(l1)
    {
        return Ok(false);
    }
    let tabulate = map0.len() <= MAX_TABULATED_ORDS;
    // A document's ordinals in the two read-ahead columns, when it has both.
    let ords_of = |c: &CompositeLeaf<'_>, doc: i32| -> Option<(i64, i64)> {
        let at = |l: &Leaf<'_>| {
            let i = doc.wrapping_sub(l.pre_base) as u32 as usize;
            let has = l
                .pre_bits
                .get(i >> 6)
                .is_some_and(|w| w >> (i & 63) & 1 == 1);
            (i < l.pre_len && has).then(|| l.pre_vals[i])
        };
        at(&c.sources[0]).zip(at(&c.sources[1]))
    };
    for &doc in docs {
        let Some((a, b)) = ords_of(c, doc) else {
            collect_composite(sources, size, subs, st, c, &mut leaf.subs, doc, None)?;
            continue;
        };
        if tabulate && c.dead_epoch != st.epoch {
            c.dead0.clear();
            for o in 0..map0.len() {
                c.dead0
                    .push(first_hopeless(s0, c.after.as_deref(), st, map0, o as i64)?);
            }
            c.dead_epoch = st.epoch;
        }
        if tabulate {
            if c.dead0.get(a as usize).copied().unwrap_or(false) {
                continue;
            }
        } else if first_hopeless(s0, c.after.as_deref(), st, map0, a)? {
            continue;
        }
        let k0 = Some(s0.key(map0, a)?);
        let key = [k0, Some(s1.key(map1, b)?)];
        let past_after = c.after.as_ref().is_none_or(|after| {
            for ((s, &k), &a) in sources.iter().zip(&key).zip(after) {
                let o = s.compare_after(k, a);
                if o.is_ne() {
                    return o.is_gt();
                }
            }
            false
        });
        if past_after
            && st
                .worst
                .as_deref()
                .is_none_or(|w| compare_keys(sources, &key, w).is_le())
        {
            place(sources, size, subs, st, &key, true, &mut leaf.subs, doc)?;
        }
    }
    Ok(true)
}

/// `TermsSortedDocsProducer.processLeaf`: a top-level `composite` whose
/// first source the aggregator reads sorted (a keyword over a `match_all`)
/// walks that field's terms in order from `after`'s, each term's live
/// documents collected with the term as their first-source key, and stops
/// after a term (not the first) whose documents offered keys none of which
/// was competitive while the queue was full: every later term's keys sort
/// past the worst kept one. As OpenSearch, documents without a first-source
/// value are not visited. Returns whether the segment
/// was collected; not when a sub-aggregation's columns need documents in
/// order, or the field has no terms here.
pub(super) fn collect_composite_sorted(
    node: &AggNode,
    state: &mut State,
    leaf: &mut Leaf<'_>,
    seg: &OpenSegment<'_>,
    reader: &SegmentReader,
) -> Result<bool> {
    let (
        AggNode::Composite {
            sources,
            size,
            subs,
            ..
        },
        State::Composite(st),
    ) = (node, &mut *state)
    else {
        return Ok(false);
    };
    let Some(c) = leaf.composite.as_deref_mut() else {
        return Ok(false);
    };
    let Some(first) = sources.first() else {
        return Ok(false);
    };
    if matches!(
        first.values,
        CompositeValues::Long | CompositeValues::Date(_)
    ) {
        return collect_composite_points(node, state, leaf, seg);
    }
    // `checkIfSortedDocsIsApplicable`: ascending, and a missing bucket only
    // past an `after` and sorted first.
    if !c.sorted_lead
        || first.reverse
        || (first.missing_bucket
            && (c.after.is_none() || matches!(first.missing_order, MissingOrder::Last)))
        || !matches!(first.values, CompositeValues::Keyword)
        || !leaf.subs.iter().all(any_order)
    {
        return Ok(false);
    }
    let Some(Col::Ords(_, map)) = c.sources.first().map(|l| &l.col) else {
        return Ok(false);
    };
    let map: &[i64] = map;
    let Some(terms) = seg.fields.field(&first.field) else {
        return Ok(false);
    };
    let Some(mut dict) = lucene_search::global_ords::open_ords(reader, &first.field)?.1 else {
        return Ok(false);
    };
    // `seekCeil(after)`: the first term at or past `after`'s.
    let from = match c.after.as_ref().and_then(|a| a.first()) {
        Some(AfterPart::Exact(g) | AfterPart::Before(g)) => map.partition_point(|&o| o < *g),
        Some(AfterPart::Missing) | None => 0,
    };
    for l in c.sources.iter_mut().chain(leaf.subs.iter_mut()) {
        l.drop_prefetch();
    }
    let live = seg.live_docs;
    let mut table: Vec<u32> = Vec::new();
    for ord in from..map.len() {
        let term = dict
            .lookup_ord(ord as i64)
            .map_err(|e| lucene_search::Error::from(lucene_codecs::doc_values::Error::from(e)))?;
        let docs = terms
            .postings_with_flags(
                term,
                seg.doc_in,
                lucene_codecs::postings::PostingsFlags::DocsOnly,
            )?
            .map(|p| p.docs)
            .unwrap_or_default();
        let (mut collected, mut placed) = (false, 0usize);
        // Every key this term leads sorts past the worst kept one: its
        // documents offer keys and place none, so the first that offers one
        // settles the term.
        let hopeless = st.worst.as_deref().is_some_and(|w| {
            first
                .key(map, ord as i64)
                .is_ok_and(|k| first.compare(Some(k), w[0]).is_gt())
        });
        let second = match (sources.as_slice(), c.sources.as_slice()) {
            ([_, s1], [_, l1])
                if s1.values == CompositeValues::Keyword
                    && matches!(l1.col, Col::Ords(Ords::Single(_), _))
                    && l1.missing.is_none() =>
            {
                Some(s1)
            }
            _ => None,
        };
        // The second source's segment ordinals settled under this term: the
        // bucket of one, or that its key is never competitive.
        table.clear();
        if second.is_some() {
            if let Col::Ords(_, m) = &c.sources[1].col {
                if m.len() <= MAX_TABULATED_ORDS {
                    table.resize(m.len(), PAIR_UNSEEN);
                }
            }
        }
        // A lone source without sub-aggregations: once the term's bucket is
        // placed, its other documents only add to its count.
        let lone = sources.len() == 1 && subs.is_empty();
        let mut placed_bucket: Option<u32> = None;
        for doc in docs {
            if !live.is_none_or(|l| l.get_doc(doc)) {
                continue;
            }
            if let (true, Some(b)) = (lone, placed_bucket) {
                *slot(&mut st.docs, b as usize) += 1;
                continue;
            }
            let offered = match second {
                Some(s1) if !hopeless => collect_sorted_pair(
                    sources,
                    s1,
                    *size,
                    subs,
                    st,
                    c,
                    &mut leaf.subs,
                    doc,
                    ord as i64,
                    &mut table,
                )?,
                _ => collect_composite(
                    sources,
                    *size,
                    subs,
                    st,
                    c,
                    &mut leaf.subs,
                    doc,
                    Some(ord as i64),
                )?,
            };
            if let Some(n) = offered {
                collected = true;
                placed += n;
                if hopeless {
                    break;
                }
                if lone && n == 1 {
                    let key = [Some(first.key(map, ord as i64)?)];
                    placed_bucket = st
                        .index
                        .get(&key[..])
                        .copied()
                        .filter(|&b| !st.keys[b as usize].is_empty());
                }
            }
        }
        // `processBucket`: `queue.isFull() && hasCollected && none competitive`.
        if collected && placed == 0 && ord != from && st.worst.is_some() {
            break;
        }
    }
    Ok(true)
}

/// What [`collect_composite_points`]' walk of a field's points does with each
/// run of equal values: the buckets it groups them in, ascending, handed to
/// `flush` one at a time. `flush` answers whether the walk is over.
pub(super) struct SortedPoints<'f> {
    pub(super) decode: PointDecode,
    /// The bucket of a value.
    pub(super) bucket: &'f dyn Fn(i64) -> Result<i64>,
    /// Values below it are not read (`after`'s).
    pub(super) lower: Option<i64>,
    pub(super) current: Option<i64>,
    pub(super) docs: Vec<i32>,
    /// Receives a bucket's documents, ascending and distinct.
    pub(super) flush: &'f mut dyn FnMut(i64, &[i32]) -> Result<bool>,
    pub(super) done: bool,
    pub(super) error: Option<lucene_search::Error>,
}

impl SortedPoints<'_> {
    pub(super) fn add(&mut self, doc: i32, value: i64) {
        if self.done || self.error.is_some() || self.lower.is_some_and(|l| value < l) {
            return;
        }
        let b = match (self.bucket)(value) {
            Ok(b) => b,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        if self.current != Some(b) {
            self.finish();
            self.current = Some(b);
        }
        self.docs.push(doc);
    }

    /// Hands the current bucket over.
    pub(super) fn finish(&mut self) {
        let Some(b) = self.current.take() else {
            return;
        };
        if self.done || self.error.is_some() {
            self.docs.clear();
            return;
        }
        self.docs.sort_unstable();
        self.docs.dedup();
        match (self.flush)(b, &self.docs) {
            Ok(over) => self.done = over,
            Err(e) => self.error = Some(e),
        }
        self.docs.clear();
    }
}

impl lucene_codecs::points::IntersectVisitor for SortedPoints<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> lucene_codecs::points::Relation {
        use lucene_codecs::points::Relation;
        let _ = min;
        if self.done || self.error.is_some() {
            return Relation::CellOutsideQuery;
        }
        match (self.lower, self.decode.decode(max)) {
            (Some(l), Some(m)) if m < l => Relation::CellOutsideQuery,
            _ => Relation::CellCrossesQuery,
        }
    }

    fn visit(&mut self, _doc_id: i32) {}

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if let Some(v) = self.decode.decode(packed) {
            self.add(doc_id, v);
        }
    }

    fn visit_run(&mut self, doc_ids: &[i32], packed: &[u8]) {
        if let Some(v) = self.decode.decode(packed) {
            for &d in doc_ids {
                self.add(d, v);
            }
        }
    }
}

/// [`collect_composite_sorted`] for a numeric or date first source:
/// `PointsSortedDocsProducer`, which reads the field's points in value order,
/// settles each bucket's documents (those of its values, ascending) with
/// `processBucket`, and stops at the first bucket that placed nothing once
/// the queue is full. Returns whether the segment was collected.
pub(super) fn collect_composite_points(
    node: &AggNode,
    state: &mut State,
    leaf: &mut Leaf<'_>,
    seg: &OpenSegment<'_>,
) -> Result<bool> {
    let (
        AggNode::Composite {
            sources,
            size,
            subs,
            ..
        },
        State::Composite(st),
    ) = (node, state)
    else {
        return Ok(false);
    };
    let Some(c) = leaf.composite.as_deref_mut() else {
        return Ok(false);
    };
    let Some(first) = sources.first() else {
        return Ok(false);
    };
    if !c.sorted_lead
        || first.reverse
        || (first.missing_bucket
            && (c.after.is_none() || matches!(first.missing_order, MissingOrder::Last)))
        || !leaf.subs.iter().all(any_order)
    {
        return Ok(false);
    }
    let Some(points) = seg.points else {
        return Ok(false);
    };
    let Some(num) = points.field_number(&first.field) else {
        return Ok(false);
    };
    let Some(pf) = points.reader.field(num) else {
        return Ok(false);
    };
    let Some(decode) = PointDecode::of(ValueKind::Long, pf.bytes_per_dim) else {
        return Ok(false);
    };
    if pf.num_dims != 1 {
        return Ok(false);
    }
    // `after`'s value: nothing below it is read, and its own bucket never
    // ends the walk.
    let after = match c.after.as_ref().and_then(|a| a.first()) {
        Some(AfterPart::Exact(v) | AfterPart::Before(v)) => Some(*v),
        Some(AfterPart::Missing) | None => None,
    };
    for l in c.sources.iter_mut().chain(leaf.subs.iter_mut()) {
        l.drop_prefetch();
    }
    let live = seg.live_docs;
    let bucket = |v: i64| first.key(&[], v);
    let mut table: Vec<u32> = Vec::new();
    let mut flush = |b: i64, docs: &[i32]| -> Result<bool> {
        let (mut collected, mut placed) = (false, 0usize);
        // Two keyword sources, the second single-valued: the pair pass of
        // the sorted keyword read, its table settled per bucket.
        let second = match (sources.as_slice(), c.sources.as_slice()) {
            ([_, s1], [_, l1])
                if s1.values == CompositeValues::Keyword
                    && matches!(l1.col, Col::Ords(Ords::Single(_), _))
                    && l1.missing.is_none() =>
            {
                Some(s1)
            }
            _ => None,
        };
        table.clear();
        if second.is_some() {
            if let Col::Ords(_, m) = &c.sources[1].col {
                if m.len() <= MAX_TABULATED_ORDS {
                    table.resize(m.len(), PAIR_UNSEEN);
                }
            }
        }
        for &doc in docs {
            if !live.is_none_or(|l| l.get_doc(doc)) {
                continue;
            }
            let offered = match second {
                Some(s1) => collect_sorted_pair(
                    sources,
                    s1,
                    *size,
                    subs,
                    st,
                    c,
                    &mut leaf.subs,
                    doc,
                    b,
                    &mut table,
                )?,
                None => {
                    collect_composite(sources, *size, subs, st, c, &mut leaf.subs, doc, Some(b))?
                }
            };
            if let Some(n) = offered {
                collected = true;
                placed += n;
            }
        }
        // `processBucket`: `queue.isFull() && hasCollected && none competitive`.
        Ok(collected && placed == 0 && after != Some(b) && st.worst.is_some())
    };
    let mut visitor = SortedPoints {
        decode,
        bucket: &bucket,
        lower: after,
        current: None,
        docs: Vec::new(),
        flush: &mut flush,
        done: false,
        error: None,
    };
    points.reader.intersect(num, &mut visitor)?;
    visitor.finish();
    if let Some(e) = visitor.error {
        return Err(e);
    }
    Ok(true)
}

/// [`collect_composite`] of one document of a sorted read's term over two
/// keyword sources, the second single-valued: its key is the term and the
/// document's one value, checked against `after` and the worst kept key
/// without the generic combination bookkeeping. A document without a
/// single second value goes the generic way.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_sorted_pair(
    sources: &[CompositeSource],
    s1: &CompositeSource,
    size: usize,
    subs: &[AggNode],
    st: &mut CompositeState,
    c: &mut CompositeLeaf<'_>,
    leaf_subs: &mut [Leaf<'_>],
    doc: i32,
    ord: i64,
    table: &mut [u32],
) -> Result<Option<usize>> {
    let l1 = &mut c.sources[1];
    if size == 0 || !l1.read(doc)? || l1.buf.len() != 1 {
        return collect_composite(sources, size, subs, st, c, leaf_subs, doc, Some(ord));
    }
    let v = l1.buf[0];
    let at = usize::try_from(v).ok().filter(|&i| i < table.len());
    if let Some(i) = at {
        let b = table[i];
        if b == PAIR_DEAD {
            return Ok(Some(0));
        }
        if b != PAIR_UNSEEN && !st.keys[b as usize].is_empty() {
            *slot(&mut st.docs, b as usize) += 1;
            for ((n, sub), l) in subs.iter().zip(&mut st.subs).zip(leaf_subs) {
                collect(n, sub, l, doc, b)?;
            }
            return Ok(Some(1));
        }
    }
    let map1 = match &l1.col {
        Col::Ords(_, map) => *map,
        _ => &[],
    };
    let map0 = match &c.sources[0].col {
        Col::Ords(_, map) => *map,
        _ => &[],
    };
    let key = [Some(sources[0].key(map0, ord)?), Some(s1.key(map1, v)?)];
    let past_after = c.after.as_ref().is_none_or(|after| {
        for ((s, &k), &a) in sources.iter().zip(&key).zip(after) {
            let o = s.compare_after(k, a);
            if o.is_ne() {
                return o.is_gt();
            }
        }
        false
    });
    let competitive = past_after
        && st
            .worst
            .as_deref()
            .is_none_or(|w| compare_keys(sources, &key, w).is_le());
    if !competitive {
        if let Some(i) = at {
            table[i] = PAIR_DEAD;
        }
        return Ok(Some(0));
    }
    place(sources, size, subs, st, &key, true, leaf_subs, doc)?;
    if let Some(i) = at {
        table[i] = st.index.get(&key[..]).copied().unwrap_or(PAIR_DEAD);
    }
    Ok(Some(1))
}

/// Whether a sub-aggregation's columns read documents in any order (each
/// rewinds when asked for an earlier one): none of its nodes follows a
/// document stream (nested, scores, scripts, sorted hits, text, filters).
pub(super) fn any_order(l: &Leaf<'_>) -> bool {
    l.fed.is_none()
        && l.nested.is_none()
        && l.text.is_none()
        && l.filter.is_none()
        && l.composite.is_none()
        && !matches!(l.col, Col::Script(_) | Col::Mapped(_) | Col::Scores(..))
        && l.subs.iter().all(any_order)
}

/// Whether a first-source key sorts before `after`'s first part: every key
/// it leads is then at or before `after`.
pub(super) fn s0_before_after(s: &CompositeSource, k: KeyPart, a: AfterPart) -> bool {
    s.compare_after(k, a).is_lt()
}

/// A competitive key's bucket: found (by ordinal for a lone keyword
/// source) or added, its count bumped when `count`, the document passed to
/// the sub-aggregations, and the queue pruned past `2 * size` buckets.
#[allow(clippy::too_many_arguments)]
pub(super) fn place(
    sources: &[CompositeSource],
    size: usize,
    subs: &[AggNode],
    st: &mut CompositeState,
    key: &[KeyPart],
    count: bool,
    leaf_subs: &mut [Leaf<'_>],
    doc: i32,
) -> Result<()> {
    let direct = direct_slot(sources, key);
    let cached = direct
        .and_then(|i| st.direct.get(i).copied())
        .filter(|&o| o != NO_BUCKET);
    let ord = match cached {
        Some(o) => o,
        None => {
            let o = match st.index.get(key) {
                Some(&o) => o,
                None => {
                    let o = st.keys.len() as u32;
                    let boxed: Box<[KeyPart]> = key.into();
                    st.index.insert(boxed.clone(), o);
                    st.keys.push(boxed);
                    o
                }
            };
            if let Some(i) = direct {
                *slot_or(&mut st.direct, i, NO_BUCKET) = o;
            }
            o
        }
    };
    if count {
        *slot(&mut st.docs, ord as usize) += 1;
    }
    for ((n, sub), l) in subs.iter().zip(&mut st.subs).zip(leaf_subs) {
        collect(n, sub, l, doc, ord)?;
    }
    if st.index.len() > size.saturating_mul(2) {
        st.prune(sources, size);
    }
    Ok(())
}
