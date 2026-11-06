//! Writing a segment's IVF files.
//!
//! `.ivm`:
//! ```text
//! IndexHeader(META_CODEC, version, segment id, suffix)
//! vint field count
//! for each field:
//!   i32 field number, byte similarity ordinal,
//!   byte vector encoding ordinal (from VERSION_VECTOR_ENCODING),
//!   vint dimension, vint vector count,
//!   string quantizer name, vint state length, state bytes,
//!   vint cluster count,
//!   vlong postings offset, vlong vectors offset, vlong doc table offset,
//!   centroids (cluster count x dimension f32 LE),
//!   cluster entry counts (cluster count x vint),
//!   for the int8 encoding, each cluster's own (primary) vector count
//!   (cluster count x vint): cluster c's vectors are the slots from the sum
//!   of the counts before it
//! Footer
//! ```
//!
//! `.ivp`, from the field's postings offset, one block per cluster in order:
//! the cluster's entries as vector slots (`u32` LE, ascending), then their
//! codes, in the same order and encoded relative to this cluster's centroid,
//! as one block in the quantizer's layout (`Quantizer::write_block`,
//! `Quantizer::block_len` bytes: end to end by default, 32 codes interleaved
//! per block for OSQ's 1- and 2-bit encodings). A cluster's entries are its own
//! vectors and, with spilling, the vectors spilled into it from their
//! primary cluster; every vector has one primary entry and at most one
//! spilled one.
//!
//! `.ivv`, from the field's vectors offset (4-byte aligned): every vector
//! (`VectorEncoding::vector_bytes(dimension)` bytes in the field's
//! encoding, int8 relative to its primary cluster's centroid; `f32` LE
//! before `VERSION_VECTOR_ENCODING`) in primary-cluster order, then
//! document order within a cluster; a vector's position is its slot. From
//! the doc table offset: the field's documents ascending (`i32` LE) and, for
//! each, its slot (`u32` LE).
//!
//! A `Cosine` field stores its vectors unit-normalized.

use rayon::prelude::*;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::DataOutput;
use lucene_vector_quant::QuantizerFactory;

use crate::kmeans::{self, KMeansParams, SplitMix64};
use crate::{
    similarity_ordinal, Error, Result, Similarity, VectorEncoding, META_CODEC, POSTINGS_CODEC,
    VECTORS_CODEC, VERSION_CURRENT,
};

/// Build parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvfParams {
    /// The most vectors a cluster is split down to. A field with no more
    /// vectors than this is one cluster: a flat scan of its codes. The
    /// refinement that follows the split can leave a cluster somewhat over
    /// it, and spilling adds entries on top.
    pub target_cluster_size: usize,
    /// How the clusters are found.
    pub kmeans: KMeansParams,
    /// The most vectors the quantizer trains on.
    pub training_sample: usize,
    /// Spill every vector into a second cluster with SOAR
    /// ([`kmeans::soar_assign`]), weighting the residual term by this
    /// `lambda`; `None` keeps one cluster per vector. Spilling doubles the
    /// codes (the full-precision vectors are stored once) and lets a search
    /// probe fewer clusters for the same recall.
    pub spill: Option<f32>,
    /// How `.ivv` stores the full-precision vectors that rerank and exact
    /// search read: `f32` exactly, or `f16`, `bf16` or 8 bits for a half or
    /// about a quarter of the bytes per reranked candidate.
    pub rerank_encoding: VectorEncoding,
}

impl Default for IvfParams {
    fn default() -> Self {
        IvfParams {
            target_cluster_size: 384,
            kmeans: KMeansParams::default(),
            training_sample: 25_000,
            spill: None,
            rerank_encoding: VectorEncoding::F32,
        }
    }
}

/// One field's vectors, as a segment holds them.
#[derive(Debug, Clone, Copy)]
pub struct FieldVectors<'a> {
    /// The field number.
    pub number: i32,
    /// The field's similarity.
    pub similarity: Similarity,
    /// The vector dimension.
    pub dimension: usize,
    /// The documents with a vector, strictly ascending.
    pub docs: &'a [i32],
    /// One vector per document, laid end to end.
    pub vectors: &'a [f32],
}

/// The three files of a segment's IVF index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfFiles {
    /// `.ivm`.
    pub meta: Vec<u8>,
    /// `.ivp`.
    pub postings: Vec<u8>,
    /// `.ivv`.
    pub vectors: Vec<u8>,
}

/// Writes the IVF files of one segment, field by field.
#[derive(Debug)]
pub struct IvfWriter {
    fields: Vec<u8>,
    field_numbers: Vec<i32>,
    postings: Vec<u8>,
    vectors: Vec<u8>,
    meta: Vec<u8>,
}

/// A field's vectors: unit-normalized for `Cosine`.
fn prepared_vectors(field: &FieldVectors<'_>) -> Vec<f32> {
    let mut out = field.vectors.to_vec();
    if field.similarity == Similarity::Cosine {
        out.par_chunks_mut(field.dimension).for_each(|v| {
            // A zero vector has no direction and stays zero.
            let _ = lucene_util::vector_util::l2normalize(v, false);
        });
    }
    out
}

/// Whether a field's vectors are unit length and clustered spherically.
pub(crate) fn spherical(similarity: Similarity) -> bool {
    matches!(similarity, Similarity::Cosine | Similarity::DotProduct)
}

/// Zero-pads `out` to a multiple of four bytes.
fn align4(out: &mut Vec<u8>) {
    let aligned = out.len().next_multiple_of(4);
    out.resize(aligned, 0);
}

fn write_len(out: &mut Vec<u8>, v: usize) {
    // In-memory lengths fit an i64.
    out.write_vlong(i64::try_from(v).unwrap_or(i64::MAX));
}

fn write_count(out: &mut Vec<u8>, v: usize) -> Result<()> {
    let v = i32::try_from(v)
        .map_err(|_| Error::IllegalArgument(format!("count {v} does not fit an int")))?;
    out.write_vint(v);
    Ok(())
}

impl IvfWriter {
    /// A writer for the segment `segment_id`, its files named with `suffix`.
    pub fn new(segment_id: &[u8; ID_LENGTH], suffix: &str) -> Self {
        let mut meta = Vec::new();
        let mut postings = Vec::new();
        let mut vectors = Vec::new();
        codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, segment_id, suffix);
        codec_util::write_index_header(
            &mut postings,
            POSTINGS_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        codec_util::write_index_header(
            &mut vectors,
            VECTORS_CODEC,
            VERSION_CURRENT,
            segment_id,
            suffix,
        );
        IvfWriter {
            fields: Vec::new(),
            field_numbers: Vec::new(),
            postings,
            vectors,
            meta,
        }
    }

    fn check(&self, field: &FieldVectors<'_>) -> Result<()> {
        let bad = |m: String| Err(Error::IllegalArgument(m));
        if self.field_numbers.contains(&field.number) {
            return bad(format!("field {} written twice", field.number));
        }
        if field.dimension == 0 {
            return bad("dimension must be positive".into());
        }
        if Some(field.vectors.len()) != field.docs.len().checked_mul(field.dimension) {
            return bad(format!(
                "{} floats for {} documents of dimension {}",
                field.vectors.len(),
                field.docs.len(),
                field.dimension
            ));
        }
        if field.docs.first().is_some_and(|&d| d < 0) || field.docs.windows(2).any(|w| w[0] >= w[1])
        {
            return bad("documents must be non-negative and strictly ascending".into());
        }
        if field.vectors.iter().any(|x| !x.is_finite()) {
            return bad("vectors must be finite".into());
        }
        // Slots are u32 and u32::MAX means "no cluster".
        if field.docs.len() >= u32::MAX as usize {
            return bad("too many vectors".into());
        }
        Ok(())
    }

    /// Clusters, quantizes and writes `field`, training a quantizer from
    /// `factory`.
    ///
    /// # Errors
    ///
    /// [`Error::IllegalArgument`] for a field written twice, a zero
    /// dimension, a vector count that does not match the documents,
    /// documents out of order, a non-finite value, or a value the rerank
    /// encoding cannot hold (beyond `f16`'s range); the quantizer's
    /// training errors.
    // ARITH: every index and count below is bounded by the field's vector
    // count (checked to fit a u32) times its dimension, which is the length
    // of an existing buffer; offsets are lengths of in-memory buffers.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn add_field(
        &mut self,
        field: &FieldVectors<'_>,
        factory: &dyn QuantizerFactory,
        params: &IvfParams,
    ) -> Result<()> {
        self.check(field)?;
        let dim = field.dimension;
        let n = field.docs.len();
        let vectors = prepared_vectors(field);
        let metric = if spherical(field.similarity) {
            kmeans::Metric::InnerProduct
        } else {
            kmeans::Metric::L2
        };
        let clustering = kmeans::cluster(
            &vectors,
            dim,
            params.target_cluster_size,
            &params.kmeans,
            metric,
        );
        let nlist = clustering.len(dim);
        // Encoded before anything is written, so a refused field leaves the
        // writer as it was; int8 encodes the residual from the primary
        // centroid.
        let encoding = params.rerank_encoding;
        let vector_bytes = encoding
            .vector_bytes(dim)
            .ok_or_else(|| Error::IllegalArgument("dimension too large".into()))?;
        let mut encoded = vec![0u8; n * vector_bytes];
        let representable = encoded
            .par_chunks_mut(vector_bytes)
            .zip(vectors.par_chunks(dim))
            .zip(clustering.assignments.par_iter())
            .all(|((out, v), &a)| {
                encoding.encode(v, kmeans::row(&clustering.centroids, dim, a as usize), out)
            });
        if !representable {
            return Err(Error::IllegalArgument(format!(
                "field {}: a vector is out of {}'s range",
                field.number,
                encoding.name()
            )));
        }

        let mut rng = SplitMix64::new(params.kmeans.seed ^ 0x7a11_0f5a_3b1e);
        let mut sample = Vec::new();
        for i in kmeans::sample_indices(n, params.training_sample, &mut rng) {
            sample.extend_from_slice(kmeans::row(&vectors, dim, i));
        }
        let quantizer = factory.train(dim, field.similarity, &sample)?;
        let code_len = quantizer.code_len();

        // Slots: vectors ordered by cluster, then by document (the input is
        // in document order, and the counting sort is stable).
        let mut sizes = vec![0usize; nlist];
        for &a in &clustering.assignments {
            sizes[a as usize] += 1;
        }
        let mut next = Vec::with_capacity(nlist);
        let mut start = 0usize;
        for &s in &sizes {
            next.push(start);
            start += s;
        }
        let mut slot_of = vec![0u32; n];
        let mut order = vec![0u32; n];
        for (i, &a) in clustering.assignments.iter().enumerate() {
            let slot = next[a as usize];
            next[a as usize] += 1;
            slot_of[i] = slot as u32;
            order[slot] = i as u32;
        }

        // Posting entries: every cluster's own vectors (its slot range),
        // plus the vectors spilled into it, each cluster's in slot order.
        let mut entries: Vec<Vec<u32>> = Vec::with_capacity(nlist);
        let mut first = 0usize;
        for &size in &sizes {
            entries.push((first as u32..(first + size) as u32).collect());
            first += size;
        }
        if let Some(lambda) = params.spill {
            // Spill by the metric search probes with: for unit vectors the
            // distance to a unit centroid ranks clusters as their dot does.
            let mut probe_space = clustering.clone();
            metric.adjust(&mut probe_space.centroids, dim);
            let secondary = kmeans::soar_assign(&vectors, dim, &probe_space, lambda);
            for (i, &c) in secondary.iter().enumerate() {
                if c != u32::MAX {
                    entries[c as usize].push(slot_of[i]);
                }
            }
            for e in &mut entries {
                e.sort_unstable();
            }
        }
        let flat: Vec<(u32, u32)> = entries
            .iter()
            .enumerate()
            .flat_map(|(c, e)| e.iter().map(move |&slot| (c as u32, slot)))
            .collect();
        let mut codes = vec![0u8; flat.len() * code_len];
        if code_len > 0 {
            // Each centroid as the quantizer encodes against it, mapped once
            // per cluster rather than once per vector.
            let prepared: Vec<Option<Vec<f32>>> = clustering
                .centroids
                .par_chunks(dim)
                .map(|c| quantizer.prepare_centroid(c))
                .collect();
            codes
                .par_chunks_mut(code_len)
                .zip(flat.par_iter())
                .for_each(|(out, &(c, slot))| {
                    let vector = kmeans::row(&vectors, dim, order[slot as usize] as usize);
                    match &prepared[c as usize] {
                        Some(p) => quantizer.encode_prepared(vector, p, out),
                        None => quantizer.encode(
                            vector,
                            kmeans::row(&clustering.centroids, dim, c as usize),
                            out,
                        ),
                    }
                });
        }

        let postings_offset = self.postings.len();
        let mut at = 0usize;
        let mut block = Vec::new();
        for e in &entries {
            for slot in e {
                self.postings.write_bytes(&slot.to_le_bytes());
            }
            block.clear();
            quantizer.write_block(&codes[at * code_len..(at + e.len()) * code_len], &mut block);
            self.postings.write_bytes(&block);
            at += e.len();
        }

        align4(&mut self.vectors);
        let vectors_offset = self.vectors.len();
        for &i in &order {
            let i = i as usize;
            self.vectors
                .write_bytes(&encoded[i * vector_bytes..(i + 1) * vector_bytes]);
        }
        let doc_table_offset = self.vectors.len();
        for d in field.docs {
            self.vectors.write_bytes(&d.to_le_bytes());
        }
        for s in &slot_of {
            self.vectors.write_bytes(&s.to_le_bytes());
        }

        let m = &mut self.fields;
        m.write_bytes(&field.number.to_le_bytes());
        m.write_byte(similarity_ordinal(field.similarity));
        m.write_byte(encoding.ordinal());
        write_count(m, dim)?;
        write_count(m, n)?;
        m.write_string(quantizer.name());
        let state = quantizer.state();
        write_count(m, state.len())?;
        m.write_bytes(&state);
        write_count(m, nlist)?;
        write_len(m, postings_offset);
        write_len(m, vectors_offset);
        write_len(m, doc_table_offset);
        for x in &clustering.centroids {
            m.write_bytes(&x.to_le_bytes());
        }
        for e in &entries {
            write_count(m, e.len())?;
        }
        if encoding.residual() {
            for &size in &sizes {
                write_count(m, size)?;
            }
        }
        self.field_numbers.push(field.number);
        Ok(())
    }

    /// The finished files.
    pub fn finish(mut self) -> IvfFiles {
        // Field counts are tiny.
        self.meta
            .write_vint(i32::try_from(self.field_numbers.len()).unwrap_or(i32::MAX));
        self.meta.write_bytes(&self.fields);
        codec_util::write_footer(&mut self.meta);
        codec_util::write_footer(&mut self.postings);
        codec_util::write_footer(&mut self.vectors);
        IvfFiles {
            meta: self.meta,
            postings: self.postings,
            vectors: self.vectors,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use lucene_vector_quant::osq::OsqFactory;

    #[test]
    fn rejects_bad_fields() {
        let mut w = IvfWriter::new(&[0; ID_LENGTH], "");
        let f = OsqFactory::bbq();
        let p = IvfParams::default();
        let ok = FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: 2,
            docs: &[0, 3],
            vectors: &[1.0, 2.0, 3.0, 4.0],
        };
        w.add_field(&ok, &f, &p).unwrap();
        let cases = [
            ok,
            FieldVectors {
                number: 2,
                dimension: 0,
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[0],
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[3, 3],
                ..ok
            },
            FieldVectors {
                number: 2,
                docs: &[-1, 3],
                ..ok
            },
            FieldVectors {
                number: 2,
                vectors: &[1.0, f32::NAN, 3.0, 4.0],
                ..ok
            },
        ];
        for c in cases {
            assert!(
                matches!(w.add_field(&c, &f, &p), Err(Error::IllegalArgument(_))),
                "{c:?}"
            );
        }
    }

    #[test]
    fn empty_writer_has_headers_and_footers() {
        let files = IvfWriter::new(&[1; ID_LENGTH], "x").finish();
        for f in [&files.meta, &files.postings, &files.vectors] {
            assert!(codec_util::retrieve_checksum(f).is_ok());
        }
    }

    #[test]
    fn write_count_rejects_huge_values() {
        let mut out = Vec::new();
        assert!(write_count(&mut out, usize::MAX).is_err());
    }
}
