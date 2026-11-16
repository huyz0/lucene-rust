//! Writing a segment's clustered postings: neural-search's
//! `SparsePostingsConsumer`, `ClusteredPostingTermsWriter`,
//! `SeismicPostingClusterer` and `RandomClusteringAlgorithm`.
//!
//! [`SparseCodec`] is the writer's codec for an `index.sparse` index. Every
//! segment it writes records [`crate::CODEC_NAME`] and gets a `.sit`/`.sip`
//! pair. For a `sparse_vector` field of a segment at or over the field's
//! `approximate_threshold`, the pair holds the field's postings in clusters,
//! built from the forward index (the field's BINARY doc values, already
//! written): each forward-index token is a term, with the weight the
//! `FeatureField` of the same token carries, quantized with the ingest
//! ceiling.
//!
//! The field keeps its ordinary `FeatureField` postings too, so a merge into
//! a segment under the threshold still has every document's postings.
//! neural-search reads the clustered postings in place of them.
//!
//! Cluster centers are drawn from a generator seeded by the segment id and
//! the term, so the same segment is always clustered the same way.
//! neural-search draws them from an unseeded one.

use crate::postings::{self, Cluster, FieldClusters};
use crate::search::{attribute, float_attribute, SeismicField};
use crate::vector::{self, SparseVector};
use crate::{Error, Result};
use lucene_codecs::codec::{Lucene104Codec, SegmentFilesWriter, SegmentWrite};
use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::per_field_doc_values::{PER_FIELD_FORMAT_KEY, PER_FIELD_SUFFIX_KEY};
use std::collections::BTreeMap;

/// `MINIMAL_DOC_SIZE_TO_CLUSTER`: shorter postings are one cluster.
const MIN_DOCS_TO_CLUSTER: usize = 10;
/// `DEFAULT_POSTING_PRUNE_RATIO` and `DEFAULT_POSTING_MINIMUM_LENGTH`: the
/// postings kept per term when `n_postings` is `-1`.
const POSTING_PRUNE_RATIO: f32 = 0.0005;
const POSTING_MINIMUM_LENGTH: i32 = 160;
const DEFAULT_CLUSTER_RATIO: f32 = 0.1;
const DEFAULT_SUMMARY_PRUNE_RATIO: f32 = 0.4;

/// The codec an `index.sparse` index is written with: `Lucene104`'s formats
/// under [`crate::CODEC_NAME`], plus the clustered postings.
#[derive(Debug, Default, Clone, Copy)]
pub struct SparseCodec;

impl Lucene104Codec for SparseCodec {
    fn name(&self) -> &str {
        crate::CODEC_NAME
    }

    fn segment_files_writer(&self) -> Option<&dyn SegmentFilesWriter> {
        Some(&ClusteredPostingsWriter)
    }
}

/// Writes `.sit`/`.sip` for each segment; see the module docs.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClusteredPostingsWriter;

impl SegmentFilesWriter for ClusteredPostingsWriter {
    fn extensions(&self) -> &'static [&'static str] {
        &[postings::TERMS_EXTENSION, postings::POSTINGS_EXTENSION]
    }

    fn write(&self, segment: SegmentWrite<'_>) -> std::io::Result<Vec<String>> {
        write_segment(segment).map_err(|e| match e {
            Error::Store(lucene_store::Error::Io(io)) => io,
            other => std::io::Error::other(other.to_string()),
        })
    }
}

/// A field's clustering settings, from its attributes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusteringParams {
    /// `n_postings`: documents kept per term; `-1` scales with the segment.
    pub n_postings: i32,
    pub cluster_ratio: f32,
    pub summary_prune_ratio: f32,
}

impl ClusteringParams {
    /// Reads `n_postings`, `cluster_ratio` and `summary_prune_ratio`.
    ///
    /// # Errors
    /// An attribute is not a number.
    pub fn from_field_info(info: &FieldInfo) -> Result<Self> {
        let n_postings = match attribute(info, "n_postings") {
            None | Some("") => -1,
            Some(v) => v.parse().map_err(|_| {
                Error::Corrupted(format!("field {}: bad n_postings [{v}]", info.name))
            })?,
        };
        Ok(ClusteringParams {
            n_postings,
            cluster_ratio: float_attribute(info, "cluster_ratio", DEFAULT_CLUSTER_RATIO)?,
            summary_prune_ratio: float_attribute(
                info,
                "summary_prune_ratio",
                DEFAULT_SUMMARY_PRUNE_RATIO,
            )?,
        })
    }

    /// `SparseFieldUtils.getNPostings`: the documents kept per term in a
    /// segment of `max_doc` documents.
    pub fn postings_per_term(&self, max_doc: i32) -> usize {
        let n = if self.n_postings == -1 {
            // `(int) (0.0005f * maxDoc)`: a float product, truncated.
            #[allow(clippy::cast_possible_truncation)]
            let scaled = (POSTING_PRUNE_RATIO * max_doc as f32) as i32;
            scaled.max(POSTING_MINIMUM_LENGTH)
        } else {
            self.n_postings
        };
        usize::try_from(n).unwrap_or(0)
    }
}

/// SplitMix64: a small generator for cluster centers.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (`bound > 0`), by rejection.
    fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let Some(rem) = u64::MAX.checked_rem(bound) else {
            return 0;
        };
        let zone = u64::MAX.wrapping_sub(rem);
        loop {
            let v = self.next();
            if v < zone {
                // Below `bound`, which came from a usize.
                return v.checked_rem(bound).unwrap_or(0) as usize;
            }
        }
    }
}

/// The seed for one term's centers: FNV-1a over the segment id and the term.
fn seed(segment_id: &[u8], field: i32, term: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in segment_id.iter().chain(&field.to_le_bytes()).chain(term) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
    }
    h
}

/// `ValueEncoder.decodeFeatureValue(encodeFeatureValue(v))`: the value a
/// `FeatureField` keeps, its low 15 mantissa bits dropped.
fn feature_value(value: f32) -> f32 {
    f32::from_bits((value.to_bits() >> 15) << 15)
}

/// One `sparse_vector` field of a segment being written: its postings and
/// forward vectors, built document by document, then clustered.
///
/// The Rust writer feeds it from the segment's doc values; the plugin's Java
/// codec, which writes the segments of OpenSearch's own engine, feeds it the
/// same values at flush and merge.
#[derive(Debug)]
pub struct FieldClusterer {
    number: i32,
    max_doc: i32,
    ceiling_ingest: f32,
    params: ClusteringParams,
    /// Term bytes (the decimal token) to `(doc, weight)`, docs ascending.
    postings: BTreeMap<Vec<u8>, Vec<(i32, u8)>>,
    /// Each doc's quantized forward vector, `None` for a doc without one.
    forward: Vec<Option<SparseVector>>,
}

impl FieldClusterer {
    /// A clusterer for `info` in a segment of `max_doc` documents.
    ///
    /// # Errors
    /// A clustering or quantization attribute is not a number.
    pub fn new(info: &FieldInfo, max_doc: i32) -> Result<Self> {
        Ok(FieldClusterer {
            number: info.number,
            max_doc,
            ceiling_ingest: float_attribute(
                info,
                "quantization_ceiling_ingest",
                crate::search::DEFAULT_CEILING_INGEST,
            )?,
            params: ClusteringParams::from_field_info(info)?,
            postings: BTreeMap::new(),
            forward: Vec::new(),
        })
    }

    /// Adds `doc`'s forward-index value: big-endian `(int token, float
    /// weight)` pairs. Documents come in ascending order; one without a
    /// value is skipped.
    ///
    /// # Errors
    /// `doc` is out of order or past the segment.
    pub fn add(&mut self, doc: i32, bytes: &[u8]) -> Result<()> {
        let at = usize::try_from(doc)
            .ok()
            .filter(|&d| d >= self.forward.len() && doc < self.max_doc)
            .ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "doc {doc} out of order or past max_doc {}",
                    self.max_doc
                ))
            })?;
        self.forward.resize(at, None);
        let pairs = vector::decode_forward(bytes);
        for &(token, value) in &pairs {
            // Negative tokens are refused at index time.
            if i32::try_from(token).is_err() {
                continue;
            }
            let weight = vector::quantize(feature_value(value), self.ceiling_ingest);
            let list = self
                .postings
                .entry(token.to_string().into_bytes())
                .or_default();
            match list.last_mut() {
                Some(last) if last.0 == doc => last.1 = weight,
                _ => list.push((doc, weight)),
            }
        }
        self.forward
            .push(Some(SparseVector::quantized(&pairs, self.ceiling_ingest)));
        Ok(())
    }

    /// The field's clustered postings, centers drawn from a generator seeded
    /// by `segment_id`, the field and the term.
    pub fn finish(self, segment_id: &[u8]) -> FieldClusters {
        let keep = self.params.postings_per_term(self.max_doc);
        let terms = self
            .postings
            .into_iter()
            .map(|(term, list)| {
                let mut rng = SplitMix(seed(segment_id, self.number, &term));
                let clusters = cluster(&list, &self.forward, &self.params, keep, &mut rng);
                (term, clusters)
            })
            .collect();
        FieldClusters {
            number: self.number,
            terms,
        }
    }
}

/// `PostingsProcessingUtils.getTopK`: the `k` heaviest postings, in doc
/// order. Equal weights keep the lower doc ids.
fn top_k(postings: &[(i32, u8)], k: usize) -> Vec<(i32, u8)> {
    if k >= postings.len() {
        return postings.to_vec();
    }
    let mut order: Vec<usize> = (0..postings.len()).collect();
    order.sort_by(|&a, &b| postings[b].1.cmp(&postings[a].1).then(a.cmp(&b)));
    order.truncate(k);
    order.sort_unstable();
    order.into_iter().map(|i| postings[i]).collect()
}

/// `PostingsProcessingUtils.summarize`: the per-token maximum of the
/// members' vectors, cut to the heaviest tokens holding more than
/// `prune_ratio` of the total weight.
fn summarize(
    members: &[(i32, u8)],
    forward: &[Option<SparseVector>],
    prune_ratio: f32,
) -> Option<SparseVector> {
    let mut max: BTreeMap<u16, u8> = BTreeMap::new();
    for &(doc, _) in members {
        if let Some(Some(v)) = usize::try_from(doc).ok().and_then(|d| forward.get(d)) {
            for (&t, &w) in v.tokens().iter().zip(v.weights()) {
                let slot = max.entry(t).or_insert(0);
                *slot = (*slot).max(w);
            }
        }
    }
    let mut items: Vec<(u16, u8)> = max.into_iter().collect();
    // Heaviest first; equal weights by token.
    items.sort_by_key(|&(_, w)| std::cmp::Reverse(w));
    let total: f64 = items.iter().map(|&(_, w)| f64::from(w)).sum();
    // `(int) Math.floor(totalWeight * summaryPruneRatio)`.
    #[allow(clippy::cast_possible_truncation)]
    let threshold = (total * f64::from(prune_ratio)).floor() as i64;
    let mut sum: i64 = 0;
    let mut keep = 0usize;
    for &(_, w) in &items {
        keep = keep.saturating_add(1);
        sum = sum.saturating_add(i64::from(w));
        if sum > threshold {
            break;
        }
    }
    items.truncate(keep);
    if items.is_empty() {
        return None;
    }
    Some(SparseVector::from_items(
        items.into_iter().map(|(t, w)| (u32::from(t), w)).collect(),
    ))
}

/// `SeismicPostingClusterer.cluster` with `RandomClusteringAlgorithm`.
fn cluster(
    postings: &[(i32, u8)],
    forward: &[Option<SparseVector>],
    params: &ClusteringParams,
    keep: usize,
    rng: &mut SplitMix,
) -> Vec<Cluster> {
    let kept = top_k(postings, keep);
    if kept.is_empty() {
        return Vec::new();
    }
    let whole = |docs: &[(i32, u8)]| Cluster {
        docs: docs.iter().map(|&(d, _)| d).collect(),
        weights: docs.iter().map(|&(_, w)| w).collect(),
        should_not_skip: true,
        summary: None,
    };
    if kept.len() < MIN_DOCS_TO_CLUSTER || params.cluster_ratio == 0.0 {
        return vec![whole(&kept)];
    }
    let vector_of = |doc: i32| -> Option<&SparseVector> {
        usize::try_from(doc)
            .ok()
            .and_then(|d| forward.get(d))
            .and_then(Option::as_ref)
    };
    let size = kept.len();
    // `Math.min(size, Math.max(1, (int) Math.ceil(size * clusterRatio)))`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let wanted = ((size as f32) * params.cluster_ratio).ceil() as usize;
    let count = wanted.clamp(1, size);
    // `ints(0, size).distinct().limit(count)`: distinct indices in draw order.
    let mut picked = vec![false; size];
    let mut centers = Vec::with_capacity(count);
    while centers.len() < count {
        let i = rng.below(size);
        if !picked[i] {
            picked[i] = true;
            centers.push(vector_of(kept[i].0));
        }
    }
    let mut assigned: Vec<Vec<(i32, u8)>> = vec![Vec::new(); count];
    for &(doc, weight) in &kept {
        let Some(v) = vector_of(doc) else {
            continue;
        };
        let dense = v.to_dense();
        let mut best = 0usize;
        // `Float.MIN_VALUE`: a zero score never beats it.
        let mut best_score = f32::from_bits(1);
        for (i, center) in centers.iter().enumerate() {
            let score = center.map_or(f32::from_bits(1), |c| c.dot(&dense) as f32);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        assigned[best].push((doc, weight));
    }
    assigned
        .into_iter()
        .filter(|docs| !docs.is_empty())
        .map(|docs| Cluster {
            summary: summarize(&docs, forward, params.summary_prune_ratio),
            should_not_skip: false,
            ..whole(&docs)
        })
        .collect()
}

/// The `.dvm`/`.dvd` names of `info`'s doc-values group.
fn doc_values_files(segment: &str, info: &FieldInfo) -> Option<(String, String, String)> {
    let format = attribute(info, PER_FIELD_FORMAT_KEY)?;
    let suffix = attribute(info, PER_FIELD_SUFFIX_KEY)?;
    let stem = format!("{segment}_{format}_{suffix}");
    Some((
        format!("{stem}.dvm"),
        format!("{stem}.dvd"),
        format!("{format}_{suffix}"),
    ))
}

/// The clustered postings of one field of `segment`.
fn field_clusters(segment: &SegmentWrite<'_>, info: &FieldInfo) -> Result<FieldClusters> {
    let out = FieldClusters {
        number: info.number,
        terms: Vec::new(),
    };
    let Some((dvm_name, dvd_name, suffix)) = doc_values_files(segment.segment, info) else {
        return Ok(out);
    };
    let dvm = segment.directory.open(&dvm_name)?;
    let dvd = segment.directory.open(&dvd_name)?;
    let infos = FieldInfos {
        fields: segment.field_infos.to_vec(),
    };
    let (_, meta) = doc_values::parse_meta(&dvm, segment.id, &suffix, &infos)?;
    let Some(entry) = meta.binary_entry(info.number) else {
        return Ok(out);
    };
    let mut clusterer = FieldClusterer::new(info, segment.max_doc)?;
    let mut values = doc_values::BinaryReader::new(&dvd, entry);
    for doc in 0..segment.max_doc {
        if let Some(bytes) = values.value(doc)? {
            clusterer.add(doc, bytes)?;
        }
    }
    Ok(clusterer.finish(segment.id))
}

/// `SparsePostingsConsumer`: `.sit`/`.sip` for `segment`, with the clustered
/// postings of each `sparse_vector` field (Lucene engine) at or over its
/// threshold, fields in name order.
pub fn write_segment(segment: SegmentWrite<'_>) -> Result<Vec<String>> {
    let mut fields: Vec<(&FieldInfo, SeismicField)> = Vec::new();
    for info in segment.field_infos {
        if let Some(field) = SeismicField::from_field_info(info)? {
            if !field.native_engine && field.is_seismic_segment(segment.max_doc) {
                fields.push((info, field));
            }
        }
    }
    fields.sort_by(|a, b| a.0.name.cmp(&b.0.name));
    let clusters = fields
        .iter()
        .map(|(info, _)| field_clusters(&segment, info))
        .collect::<Result<Vec<_>>>()?;
    let (sit, sip) = postings::write(segment.id, &clusters);
    let mut names = Vec::with_capacity(2);
    for (ext, bytes) in [
        (postings::TERMS_EXTENSION, &sit),
        (postings::POSTINGS_EXTENSION, &sip),
    ] {
        let name = format!("{}.{ext}", segment.segment);
        let mut out = segment.directory.create_output(&name)?;
        lucene_store::DataOutput::write_bytes(&mut out, bytes);
        out.close()?;
        names.push(name);
    }
    Ok(names)
}

#[cfg(test)]
mod tests;
