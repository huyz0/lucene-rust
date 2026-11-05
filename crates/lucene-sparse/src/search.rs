//! A Seismic query over one segment: neural-search's `SparseQueryWeight`
//! scorer selection, `OrderedPostingWithClustersScorer` and `ExactMatchScorer`.

use crate::postings::ClusteredPostings;
use crate::vector::{self, SparseVector, TOKEN_MODULUS};
use crate::{Error, Result};
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::FieldInfo;
use lucene_util::fixed_bit_set::FixedBitSet;

/// `SEISMIC_HEAP_SIZE`: how many top scores decide whether a cluster is skipped.
const SKIP_HEAP_SIZE: usize = 10;

/// Defaults of `SparseConstants`.
const DEFAULT_CEILING_INGEST: f32 = 3.0;
const DEFAULT_CEILING_SEARCH: f32 = 16.0;
const DEFAULT_THRESHOLD: i32 = 1_000_000;

/// A `sparse_vector` field's Seismic settings, read off its field attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SeismicField {
    pub number: i32,
    pub name: String,
    /// `approximate_threshold`: segments with fewer docs hold plain postings.
    pub approximate_threshold: i32,
    pub ceiling_ingest: f32,
    pub ceiling_search: f32,
    /// `engine`: `lucene` (clustered postings) or `native` (nsparse files).
    pub native_engine: bool,
}

pub(crate) fn attribute<'a>(info: &'a FieldInfo, key: &str) -> Option<&'a str> {
    info.attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// `NumberUtils.createFloat` on a non-empty attribute, the default otherwise.
pub(crate) fn float_attribute(info: &FieldInfo, key: &str, default: f32) -> Result<f32> {
    match attribute(info, key) {
        None | Some("") => Ok(default),
        Some(v) => v
            .trim_end_matches(['f', 'F'])
            .parse()
            .map_err(|_| Error::Corrupted(format!("field {}: bad {key} [{v}]", info.name))),
    }
}

impl SeismicField {
    /// `SparseVectorField.isSparseField` plus the Seismic attributes; `None`
    /// for a field that is not a `sparse_vector`.
    pub fn from_field_info(info: &FieldInfo) -> Result<Option<Self>> {
        if !attribute(info, "sparse_vector_field").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            return Ok(None);
        }
        let approximate_threshold = match attribute(info, "approximate_threshold") {
            None => DEFAULT_THRESHOLD,
            Some(v) => v.parse().map_err(|_| {
                Error::Corrupted(format!(
                    "field {}: bad approximate_threshold [{v}]",
                    info.name
                ))
            })?,
        };
        Ok(Some(SeismicField {
            number: info.number,
            name: info.name.clone(),
            approximate_threshold,
            ceiling_ingest: float_attribute(
                info,
                "quantization_ceiling_ingest",
                DEFAULT_CEILING_INGEST,
            )?,
            ceiling_search: float_attribute(
                info,
                "quantization_ceiling_search",
                DEFAULT_CEILING_SEARCH,
            )?,
            native_engine: attribute(info, "engine")
                .is_some_and(|e| e.eq_ignore_ascii_case("native")),
        }))
    }

    /// `PredicateUtils.shouldRunSeisPredicate`: whether a segment of `max_doc`
    /// documents holds clustered postings for this field.
    pub fn is_seismic_segment(&self, max_doc: i32) -> bool {
        max_doc >= self.approximate_threshold
    }
}

/// A `neural_sparse` query with `method_parameters`, prepared for scoring.
#[derive(Debug, Clone)]
pub struct SeismicQuery {
    /// Terms whose clusters are walked, as `SparseQueryContext.tokens`.
    pub top_tokens: Vec<String>,
    /// The quantized query vector, dense.
    pub dense: Vec<u8>,
    pub k: usize,
    pub heap_factor: f32,
    /// `boost * ceilingIngest * ceilingSearch / 255 / 255`: turns an integer
    /// dot product into the score.
    pub score_scale: f32,
}

impl SeismicQuery {
    /// `SparseAnnQueryBuilder`: folds the tokens (keeping each folded token's
    /// largest weight), keeps the `top_n` by weight as the terms to walk, and
    /// quantizes every token into the query vector with the field's search
    /// ceiling. `k` of 0 means the default 10.
    ///
    /// Tokens with equal weights at the `top_n` cut are ordered by token here;
    /// neural-search takes them in `HashMap` order.
    pub fn new(
        tokens: &[(u32, f32)],
        top_n: usize,
        k: usize,
        heap_factor: f32,
        boost: f32,
        field: &SeismicField,
    ) -> Result<Self> {
        if heap_factor.is_nan() || heap_factor <= 0.0 {
            return Err(Error::IllegalArgument(format!(
                "heap_factor must be positive, got {heap_factor}"
            )));
        }
        let mut folded: Vec<(u32, f32)> = Vec::with_capacity(tokens.len());
        for &(t, w) in tokens {
            let t = t % TOKEN_MODULUS;
            match folded.iter_mut().find(|(ft, _)| *ft == t) {
                // `Math.max(float, float)`.
                Some(slot) => {
                    slot.1 = if w.is_nan() || slot.1.is_nan() {
                        f32::NAN
                    } else {
                        slot.1.max(w)
                    }
                }
                None => folded.push((t, w)),
            }
        }
        let mut by_weight = folded.clone();
        by_weight.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let top_tokens = by_weight
            .iter()
            .take(top_n)
            .map(|(t, _)| t.to_string())
            .collect();
        let dense = SparseVector::quantized(&folded, field.ceiling_search).to_dense();
        let score_scale = boost * field.ceiling_ingest * field.ceiling_search / 255.0 / 255.0;
        Ok(SeismicQuery {
            top_tokens,
            dense,
            k: if k == 0 { 10 } else { k },
            heap_factor,
            score_scale,
        })
    }
}

/// `java.util.PriorityQueue<Pair<doc, score>>` ordered by score: the order
/// equal scores leave the heap in decides which of them survive, so this is
/// Java's binary heap, not `BinaryHeap`.
struct ScoreHeap {
    heap: Vec<(i32, i32)>,
    k: usize,
    threshold: f32,
}

impl ScoreHeap {
    fn new(k: usize) -> Self {
        ScoreHeap {
            heap: Vec::new(),
            k,
            threshold: i32::MIN as f32,
        }
    }

    fn is_full(&self) -> bool {
        self.heap.len() == self.k
    }

    fn min_score(&self) -> Option<i32> {
        self.heap.first().map(|&(_, s)| s)
    }

    /// `HeapWrapper.add`.
    fn add(&mut self, doc: i32, score: i32) {
        if (score as f32) <= self.threshold {
            return;
        }
        self.offer((doc, score));
        if self.heap.len() > self.k {
            self.poll();
            if let Some(min) = self.min_score() {
                self.threshold = min as f32;
            }
        }
    }

    /// `PriorityQueue.offer`: append and sift up.
    fn offer(&mut self, x: (i32, i32)) {
        let mut k = self.heap.len();
        self.heap.push(x);
        while k > 0 {
            let parent = k.saturating_sub(1) >> 1;
            if x.1 >= self.heap[parent].1 {
                break;
            }
            self.heap[k] = self.heap[parent];
            k = parent;
        }
        self.heap[k] = x;
    }

    /// `PriorityQueue.poll`: take the root, sift the last element down.
    fn poll(&mut self) {
        let Some(x) = self.heap.pop() else { return };
        let n = self.heap.len();
        if n == 0 {
            return;
        }
        let half = n >> 1;
        let mut k = 0;
        while k < half {
            let mut child = k.saturating_mul(2).saturating_add(1);
            let right = child.saturating_add(1);
            if right < n && self.heap[child].1 > self.heap[right].1 {
                child = right;
            }
            if x.1 <= self.heap[child].1 {
                break;
            }
            self.heap[k] = self.heap[child];
            k = child;
        }
        self.heap[k] = x;
    }

    /// `toOrderedList`: the survivors by doc id.
    fn into_sorted_by_doc(mut self) -> Vec<(i32, i32)> {
        self.heap.sort_by_key(|&(d, _)| d);
        self.heap
    }
}

/// The forward index of a field in one segment, quantized: each document's
/// folded tokens, ascending, with their weights, as the `SparseVector` Java
/// reads for it. Read once per segment core, as neural-search's forward-index
/// cache keeps them.
pub struct ForwardVectors {
    /// Where each document's tokens start; `starts[max_doc]` is the end.
    starts: Vec<usize>,
    /// The documents that have a vector.
    present: FixedBitSet,
    tokens: Vec<u16>,
    weights: Vec<u8>,
}

impl ForwardVectors {
    /// Reads every document's vector from the field's BINARY doc values
    /// (`data` and `entry`), quantized at `ceiling`.
    pub fn read(
        data: &[u8],
        entry: &doc_values::BinaryEntry,
        max_doc: i32,
        ceiling: f32,
    ) -> Result<Self> {
        let docs = usize::try_from(max_doc).unwrap_or(0);
        let mut out = ForwardVectors {
            starts: Vec::with_capacity(docs.saturating_add(1)),
            present: FixedBitSet::new(docs),
            tokens: Vec::new(),
            weights: Vec::new(),
        };
        let mut seen = vec![0u64; (TOKEN_MODULUS / 64) as usize];
        let mut items: Vec<(u16, u8)> = Vec::new();
        for (d, doc) in (0..max_doc).enumerate() {
            out.starts.push(out.tokens.len());
            let Some(bytes) = doc_values::binary_value(data, entry, doc)? else {
                continue;
            };
            out.present.set(d);
            // Without two tokens folding together (or one token twice), the
            // vector is its pairs folded, quantized and sorted; otherwise
            // `SparseVector` resolves them as Java does.
            items.clear();
            let mut clash = false;
            for c in bytes.chunks_exact(8) {
                let token = u32::from_be_bytes([c[0], c[1], c[2], c[3]]) % TOKEN_MODULUS;
                let (word, bit) = ((token / 64) as usize, 1u64 << (token % 64));
                if let Some(w) = seen.get_mut(word) {
                    clash |= *w & bit != 0;
                    *w |= bit;
                }
                let value = f32::from_be_bytes([c[4], c[5], c[6], c[7]]);
                // Folded tokens are below 32768, so they fit.
                items.push((token as u16, vector::quantize(value, ceiling)));
            }
            for &(token, _) in &items {
                if let Some(w) = seen.get_mut(usize::from(token / 64)) {
                    *w = 0;
                }
            }
            if clash {
                let v = SparseVector::quantized(&vector::decode_forward(bytes), ceiling);
                out.tokens.extend_from_slice(v.tokens());
                out.weights.extend_from_slice(v.weights());
            } else {
                items.sort_unstable_by_key(|&(t, _)| t);
                out.tokens.extend(items.iter().map(|&(t, _)| t));
                out.weights.extend(items.iter().map(|&(_, w)| w));
            }
        }
        out.starts.push(out.tokens.len());
        Ok(out)
    }

    /// `SparseVector.dotProduct` of `doc`'s vector with the dense query,
    /// `None` when the document has no vector.
    fn dot(&self, doc: usize, dense: &[u8]) -> Option<i32> {
        if doc >= self.present.len() || !self.present.get(doc) {
            return None;
        }
        let (Some(&start), Some(&end)) =
            (self.starts.get(doc), self.starts.get(doc.saturating_add(1)))
        else {
            return None;
        };
        let (Some(tokens), Some(weights)) =
            (self.tokens.get(start..end), self.weights.get(start..end))
        else {
            return None;
        };
        let mut score = 0i32;
        for (&t, &w) in tokens.iter().zip(weights) {
            // Tokens ascend, so the first one past the query ends the scan.
            let Some(&q) = dense.get(usize::from(t)) else {
                break;
            };
            // At most 255 * 255; the sum wraps as Java's int does.
            score = score.wrapping_add(i32::from(w).wrapping_mul(i32::from(q)));
        }
        Some(score)
    }
}

/// One segment's view of a Seismic field.
pub struct SegmentView<'a> {
    pub max_doc: i32,
    pub live_docs: Option<&'a FixedBitSet>,
    /// The field's forward index, when the segment has its doc values.
    pub forward: Option<&'a ForwardVectors>,
    pub postings: &'a ClusteredPostings,
}

/// Scores the query on one Seismic segment. `filter` is the segment's filter
/// bitset (filter AND field exists AND live), when neural-search would have one:
/// it has none when the query has no filter, and also when the filter matched
/// nothing in this segment, in which case the segment is searched unfiltered.
///
/// Returns `(doc, score)` by ascending doc, as the scorer emits them.
pub fn search(
    segment: &SegmentView<'_>,
    field: &SeismicField,
    query: &SeismicQuery,
    filter: Option<&FixedBitSet>,
) -> Result<Vec<(i32, f32)>> {
    let dot = |doc: i32| -> Option<i32> {
        let doc = usize::try_from(doc).ok()?;
        segment.forward.and_then(|f| f.dot(doc, &query.dense))
    };
    let score = |raw: i32| query.score_scale * raw as f32;

    if let Some(filter) = filter {
        let cardinality = filter.cardinality();
        if cardinality <= query.k {
            // `ExactMatchScorer`: every filtered doc, without a top-k cut.
            let mut hits = Vec::with_capacity(cardinality);
            let mut doc = filter.next_set_bit(0);
            while let Some(d) = doc {
                let d32 =
                    i32::try_from(d).map_err(|_| Error::Corrupted("doc id overflow".into()))?;
                let s = dot(d32).unwrap_or(0);
                hits.push((d32, score(s)));
                doc = filter.next_set_bit(d.saturating_add(1));
            }
            return Ok(hits);
        }
    }

    // `SeismicBaseScorer.searchUpfront`.
    let max_doc = usize::try_from(segment.max_doc).unwrap_or(0);
    let mut visited = vec![false; max_doc];
    let mut skip_heap = ScoreHeap::new(SKIP_HEAP_SIZE);
    let mut results = ScoreHeap::new(query.k);
    for token in &query.top_tokens {
        let Some(clusters) = segment.postings.clusters(field.number, token.as_bytes())? else {
            continue;
        };
        for cluster in clusters {
            if !cluster.should_not_skip && skip_heap.is_full() {
                let summary_score = cluster.summary.as_ref().map_or(0, |s| s.dot(&query.dense));
                let bound = skip_heap.min_score().unwrap_or(0) as f32 / query.heap_factor;
                if (summary_score as f32) < bound {
                    continue;
                }
            }
            for &doc in &cluster.docs {
                let Ok(d) = usize::try_from(doc) else {
                    continue;
                };
                if segment.live_docs.is_some_and(|l| d >= l.len() || !l.get(d)) {
                    continue;
                }
                let Some(seen) = visited.get_mut(d) else {
                    continue;
                };
                if *seen {
                    continue;
                }
                *seen = true;
                let Some(s) = dot(doc) else { continue };
                skip_heap.add(doc, s);
                results.add(doc, s);
            }
        }
    }
    Ok(results
        .into_sorted_by_doc()
        .into_iter()
        .filter(|&(d, _)| {
            filter.is_none_or(|f| usize::try_from(d).is_ok_and(|d| d < f.len() && f.get(d)))
        })
        .map(|(d, s)| (d, score(s)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::Cluster;

    fn info(attrs: &[(&str, &str)]) -> FieldInfo {
        FieldInfo::new("emb", 1).with_attributes(
            attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn field() -> SeismicField {
        SeismicField::from_field_info(&info(&[("sparse_vector_field", "true")]))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn field_attributes_and_defaults() {
        assert_eq!(SeismicField::from_field_info(&info(&[])).unwrap(), None);
        let f = field();
        assert_eq!(f.approximate_threshold, DEFAULT_THRESHOLD);
        assert_eq!(f.ceiling_ingest, 3.0);
        assert_eq!(f.ceiling_search, 16.0);
        assert!(!f.native_engine);
        assert!(f.is_seismic_segment(DEFAULT_THRESHOLD));
        assert!(!f.is_seismic_segment(DEFAULT_THRESHOLD - 1));
        let f = SeismicField::from_field_info(&info(&[
            ("sparse_vector_field", "TRUE"),
            ("engine", "Native"),
            ("approximate_threshold", "5"),
            ("quantization_ceiling_ingest", "2.5f"),
            ("quantization_ceiling_search", ""),
        ]))
        .unwrap()
        .unwrap();
        assert!(f.native_engine);
        assert_eq!(f.approximate_threshold, 5);
        assert_eq!(f.ceiling_ingest, 2.5);
        assert_eq!(f.ceiling_search, 16.0);
        for bad in [
            ("approximate_threshold", "x"),
            ("quantization_ceiling_ingest", "x"),
            ("quantization_ceiling_search", "x"),
        ] {
            assert!(
                SeismicField::from_field_info(&info(&[("sparse_vector_field", "true"), bad]))
                    .is_err()
            );
        }
    }

    #[test]
    fn query_folds_tokens_and_cuts_by_weight() {
        let f = field();
        let q = SeismicQuery::new(
            &[(1, 0.5), (32769, 2.0), (3, 1.0), (4, 1.0)],
            2,
            0,
            1.0,
            2.0,
            &f,
        )
        .unwrap();
        // 32769 folds onto 1 and its larger weight wins; 3 and 4 tie, the lower token first.
        assert_eq!(q.top_tokens, vec!["1", "3"]);
        assert_eq!(q.k, 10);
        assert_eq!(q.dense[1], quantize_for(2.0));
        assert_eq!(q.score_scale, 2.0 * 3.0 * 16.0 / 255.0 / 255.0);
        let nan = SeismicQuery::new(&[(1, 0.5), (1, f32::NAN)], 1, 1, 1.0, 1.0, &f).unwrap();
        assert_eq!(nan.dense, vec![0, 0]);
        for hf in [0.0, -1.0, f32::NAN] {
            assert!(SeismicQuery::new(&[(1, 1.0)], 1, 1, hf, 1.0, &f).is_err());
        }
    }

    fn quantize_for(v: f32) -> u8 {
        crate::vector::quantize(v, 16.0)
    }

    #[test]
    fn score_heap_matches_java_priority_queue() {
        // java.util.PriorityQueue keeps, of equal scores, the one its heap
        // order leaves at the root longest: (1,5) is evicted before (2,5).
        let mut h = ScoreHeap::new(2);
        h.add(1, 5);
        h.add(2, 5);
        h.add(3, 7);
        assert_eq!(h.into_sorted_by_doc(), vec![(2, 5), (3, 7)]);
        let mut h = ScoreHeap::new(3);
        for (d, s) in [(0, 9), (1, 3), (2, 6), (3, 1), (4, 8), (5, 2)] {
            h.add(d, s);
        }
        // A score at or below the evicted minimum is not admitted.
        h.add(6, 6);
        assert_eq!(h.into_sorted_by_doc(), vec![(0, 9), (2, 6), (4, 8)]);
        let mut empty = ScoreHeap::new(1);
        empty.poll();
        assert_eq!(empty.min_score(), None);
        let mut none = ScoreHeap::new(0);
        none.add(1, 1);
        assert!(none.into_sorted_by_doc().is_empty());
    }

    #[test]
    fn search_without_doc_values_or_past_bounds_finds_nothing() {
        let f = field();
        let postings = ClusteredPostings::from_clusters(
            1,
            vec![(
                "2",
                vec![Cluster {
                    docs: vec![-1, 0, 1, 9],
                    weights: vec![1, 1, 1, 1],
                    should_not_skip: true,
                    summary: Some(SparseVector::from_items(vec![(2, 1)])),
                }],
            )],
        );
        let mut live = FixedBitSet::new(2);
        live.set(1);
        let view = SegmentView {
            max_doc: 2,
            live_docs: Some(&live),
            forward: None,
            postings: &postings,
        };
        let q = SeismicQuery::new(&[(2, 1.0), (5, 0.5)], 2, 10, 1.0, 1.0, &f).unwrap();
        assert!(search(&view, &f, &q, None).unwrap().is_empty());
        let mut filter = FixedBitSet::new(2);
        filter.set(0);
        // The filter holds fewer docs than k: exact scoring, a doc without a vector scores 0.
        assert_eq!(
            search(&view, &f, &q, Some(&filter)).unwrap(),
            vec![(0, 0.0)]
        );
        let view = SegmentView {
            live_docs: None,
            max_doc: 1,
            ..view
        };
        let q = SeismicQuery::new(&[(2, 1.0)], 1, 1, 1.0, 1.0, &f).unwrap();
        let mut wide = FixedBitSet::new(1);
        wide.set(0);
        let none = FixedBitSet::new(1);
        // A filter wider than k becomes a post-filter.
        let q0 = SeismicQuery { k: 0, ..q };
        assert!(search(&view, &f, &q0, Some(&none)).unwrap().is_empty());
        assert!(search(&view, &f, &q0, Some(&wide)).unwrap().is_empty());
    }
}
