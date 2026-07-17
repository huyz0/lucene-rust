//! Flat (non-graph) KNN vector storage plus brute-force exact search.
//!
//! **Scope note (task #219, first slice):** real Lucene's vector support is
//! two layered formats: `Lucene99FlatVectorsFormat` (flat `.vec`/`.vem`
//! storage of raw per-doc vectors) and `Lucene99HnswVectorsFormat` (a `.vex`
//! HNSW approximate-nearest-neighbor graph built *on top of* that flat
//! storage). This module ports only the flat-storage layer, together with a
//! brute-force (`O(n)`) exact top-K search over it. It does **not** implement
//! any HNSW graph: no `.vex` file, no graph construction, no approximate
//! search, no multi-layer skip-list traversal. Brute-force search is correct
//! and useful on its own for small vector counts, and flat storage is the
//! exact foundation an HNSW layer would sit on top of later.
//!
//! **Also not yet done:** this module's [`FieldVectors`] carries its own
//! caller-supplied `dimension`/`similarity`, entirely decoupled from
//! `field_infos.rs`'s pre-existing `FieldInfo::vector_dimension`/
//! `vector_similarity_function` metadata fields -- nothing here reads or
//! cross-validates against them. Wiring this module to actually consult
//! (and be driven by) that metadata is left for the `IndexWriter`
//! integration this slice also doesn't attempt.
//!
//! **Wire format note:** this is *not* a byte-for-byte port of Lucene's own
//! `.vec`/`.vem` format (which additionally slices into meta-vector clusters,
//! and would take considerably longer to reverse-engineer precisely than this
//! slice's budget allows). Instead this defines this port's own format, using
//! the same `codec_util` header/footer framing convention every other module
//! in this crate follows (see [`crate::norms`] for the sibling
//! doc-comment style). Two files per field group, analogous in spirit to
//! Lucene's own split:
//!
//! `.vec` (data) -- raw vector bytes, `IndexHeader` + `Footer` framed:
//! ```text
//! IndexHeader(codec="LuceneRustFlatVectorsData", version=0, id, suffix)
//! for each field, in the order given to `write_vectors`:
//!   for each vector, in the order it was added:
//!     Component_0 .. Component_{dim-1} --> f32 (LE), or i8 raw byte for
//!                                          VectorEncoding::Byte
//! Footer
//! ```
//!
//! `.vem` (metadata) -- per-field directory, `IndexHeader` + `Footer` framed:
//! ```text
//! IndexHeader(codec="LuceneRustFlatVectorsMeta", version=0, id, suffix)
//! NumFields               --> vint
//! for each field:
//!   FieldNumber           --> vint
//!   VectorEncoding        --> u8 (0 Byte, 1 Float32; see field_infos::VectorEncoding)
//!   VectorSimilarityFunction --> u8 (0 Euclidean, 1 DotProduct, 2 Cosine, 3 MaximumInnerProduct)
//!   Dimension             --> vint
//!   NumVectors            --> vint
//!   DataOffset            --> vlong (byte offset into the .vec payload,
//!                                    i.e. relative to right after its
//!                                    IndexHeader, where this field's
//!                                    vectors begin)
//!   for each vector, in the same order as in .vec:
//!     DocID               --> vint
//! Footer
//! ```

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::field_infos::VectorSimilarityFunction;

const DATA_CODEC: &str = "LuceneRustFlatVectorsData";
const META_CODEC: &str = "LuceneRustFlatVectorsMeta";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("field {0}: vector dimension mismatch: expected {1}, got {2}")]
    DimensionMismatch(i32, i32, i32),
    #[error("field {0}: empty vector not allowed")]
    EmptyVector(i32),
    #[error("unknown field number {0} in vector metadata")]
    UnknownField(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One field's worth of vectors to write: `(doc_id, vector)` pairs, in
/// caller-supplied order (typically increasing doc id, though this writer
/// does not require it).
#[derive(Debug, Clone)]
pub struct FieldVectors {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    pub vectors: Vec<(i32, Vec<f32>)>,
}

/// Writes the `.vec` (data) and `.vem` (metadata) file bytes for a set of
/// fields. Returns `(vec_bytes, vem_bytes)`.
pub fn write_vectors(
    fields: &[FieldVectors],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut vec_out = Vec::new();
    codec_util::write_index_header(
        &mut vec_out,
        DATA_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let vec_header_end = vec_out.len();

    let mut vem_out = Vec::new();
    codec_util::write_index_header(
        &mut vem_out,
        META_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    vem_out.write_vint(fields.len() as i32);

    for field in fields {
        for (doc_id, v) in &field.vectors {
            if v.is_empty() {
                return Err(Error::EmptyVector(field.field_number));
            }
            if v.len() as i32 != field.dimension {
                return Err(Error::DimensionMismatch(
                    field.field_number,
                    field.dimension,
                    v.len() as i32,
                ));
            }
            let _ = doc_id;
        }

        let data_offset = (vec_out.len() - vec_header_end) as i64;
        for (_, v) in &field.vectors {
            for component in v {
                vec_out.write_i32(component.to_bits() as i32);
            }
        }

        vem_out.write_vint(field.field_number);
        vem_out.write_byte(1); // VectorEncoding::Float32 (the only encoding this slice writes)
        vem_out.write_byte(similarity_to_byte(field.similarity));
        vem_out.write_vint(field.dimension);
        vem_out.write_vint(field.vectors.len() as i32);
        vem_out.write_vlong(data_offset);
        for (doc_id, _) in &field.vectors {
            vem_out.write_vint(*doc_id);
        }
    }

    codec_util::write_footer(&mut vec_out);
    codec_util::write_footer(&mut vem_out);
    Ok((vec_out, vem_out))
}

fn similarity_to_byte(s: VectorSimilarityFunction) -> u8 {
    match s {
        VectorSimilarityFunction::Euclidean => 0,
        VectorSimilarityFunction::DotProduct => 1,
        VectorSimilarityFunction::Cosine => 2,
        VectorSimilarityFunction::MaximumInnerProduct => 3,
    }
}

fn similarity_from_byte(b: u8) -> Result<VectorSimilarityFunction> {
    match b {
        0 => Ok(VectorSimilarityFunction::Euclidean),
        1 => Ok(VectorSimilarityFunction::DotProduct),
        2 => Ok(VectorSimilarityFunction::Cosine),
        3 => Ok(VectorSimilarityFunction::MaximumInnerProduct),
        other => Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid vector similarity function: {other}"
        )))),
    }
}

/// A field's vectors as read back from `.vec`/`.vem`, queryable by doc id and
/// scannable for brute-force search.
#[derive(Debug, Clone)]
pub struct VectorField {
    pub field_number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: i32,
    /// `(doc_id, vector)` pairs in on-disk order.
    entries: Vec<(i32, Vec<f32>)>,
}

impl VectorField {
    /// Returns the vector stored for `doc_id`, if present.
    pub fn vector(&self, doc_id: i32) -> Option<&[f32]> {
        self.entries
            .iter()
            .find(|(d, _)| *d == doc_id)
            .map(|(_, v)| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Exact brute-force top-K nearest neighbor search: scores every stored
    /// vector against `query` using this field's similarity function and
    /// returns up to `k` `(doc_id, score)` pairs sorted by descending score
    /// (higher score == more similar, matching Lucene's own convention that
    /// all `VectorSimilarityFunction`s produce a score where higher is
    /// better). Uses a bounded `BinaryHeap` (min-heap over scores) rather
    /// than a full sort, so this is `O(n log k)` rather than `O(n log n)`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i32, f32)> {
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::with_capacity(k + 1);
        for (doc_id, v) in &self.entries {
            let score = self.similarity.score(query, v);
            let candidate = ScoredDoc {
                score,
                doc_id: *doc_id,
            };
            if heap.len() < k {
                heap.push(candidate);
            } else if let Some(min) = heap.peek() {
                // Min-heap ordering (see `ScoredDoc`'s `Ord` impl): the top
                // of the heap is the current worst-scoring kept candidate.
                if candidate.score > min.score {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
        let mut out: Vec<(i32, f32)> = heap.into_iter().map(|c| (c.doc_id, c.score)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

/// Wrapper giving `f32` scores a total order for the heap (vector components
/// and hence scores are never expected to be `NaN` for real stored vectors,
/// but `total_cmp` makes the ordering well-defined regardless).
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredDoc {
    score: f32,
    doc_id: i32,
}

impl Eq for ScoredDoc {}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so `BinaryHeap` (a max-heap) behaves as a min-heap on
        // score, keeping the *worst* kept candidate at the top for O(1)
        // eviction comparisons.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl VectorSimilarityFunction {
    /// Port of `VectorSimilarityFunction.compare` for each variant real
    /// Lucene defines (`org.apache.lucene.index.VectorSimilarityFunction`):
    /// every function maps its raw comparison onto a `[0, 1]`-ish range where
    /// **higher is always better** (unlike a raw distance/dot-product, which
    /// would be lower-is-better for distance or unbounded for dot product).
    ///
    /// - `Euclidean`: `1 / (1 + squareDistance)`.
    /// - `DotProduct`: assumes unit-normalized vectors; `max((1 + dot(a, b))
    ///   / 2, 0)` (Lucene does *not* return the raw dot product here -- it
    ///   rescales into `[0, 1]` the same way `Cosine` does, on the
    ///   assumption that callers normalize vectors before indexing with
    ///   `DOT_PRODUCT` -- the `max(_, 0)` floor only matters for
    ///   unnormalized inputs, where `dot` can fall below `-1` and the
    ///   rescaled value would otherwise go negative).
    /// - `Cosine`: cosine similarity of the raw (non-normalized) inputs,
    ///   rescaled the same way: `(1 + cosine(a, b)) / 2`.
    /// - `MaximumInnerProduct`: raw dot product, rescaled with Lucene's
    ///   piecewise `scaleMaxInnerProductScore`: `1 + dot` if `dot >= 0`, else
    ///   `1 / (1 - dot)` -- keeps the score positive and bounded above by
    ///   `Float.MAX_VALUE` in Java; here just a plain `f32` since this port
    ///   has no NaN-safety concerns beyond `total_cmp` in the caller.
    pub fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            VectorSimilarityFunction::Euclidean => {
                let d = square_distance(a, b);
                1.0 / (1.0 + d)
            }
            VectorSimilarityFunction::DotProduct => {
                let dot = dot_product(a, b);
                ((1.0 + dot) / 2.0).max(0.0)
            }
            VectorSimilarityFunction::Cosine => {
                let cos = cosine(a, b);
                (1.0 + cos) / 2.0
            }
            VectorSimilarityFunction::MaximumInnerProduct => {
                let dot = dot_product(a, b);
                scale_max_inner_product_score(dot)
            }
        }
    }
}

fn square_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Port of Lucene's `VectorUtil.scaleMaxInnerProductScore`.
fn scale_max_inner_product_score(dot: f32) -> f32 {
    if dot >= 0.0 {
        1.0 + dot
    } else {
        1.0 / (1.0 - dot)
    }
}

/// A whole segment's worth of flat vector fields, read back from `.vec` +
/// `.vem`.
#[derive(Debug, Clone)]
pub struct FlatVectorsReader {
    fields: Vec<VectorField>,
}

impl FlatVectorsReader {
    /// Parses `.vec` (data) and `.vem` (metadata) bytes already read into
    /// memory, verifying header and footer of both files.
    pub fn open(
        vec_buf: &[u8],
        vem_buf: &[u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let mut vec_input = SliceInput::new(vec_buf);
        codec_util::check_index_header(
            &mut vec_input,
            DATA_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        let vec_payload_start = vec_input.position();
        codec_util::check_whole_file_footer(vec_buf, vec_buf.len() - codec_util::FOOTER_LENGTH)?;

        let mut vem_input = SliceInput::new(vem_buf);
        codec_util::check_index_header(
            &mut vem_input,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::check_whole_file_footer(vem_buf, vem_buf.len() - codec_util::FOOTER_LENGTH)?;

        let num_fields = vem_input.read_vint()?;
        let mut fields = Vec::with_capacity(num_fields.max(0) as usize);
        for _ in 0..num_fields {
            let field_number = vem_input.read_vint()?;
            let _encoding = vem_input.read_byte()?; // only Float32 (1) is written by this slice
            let similarity = similarity_from_byte(vem_input.read_byte()?)?;
            let dimension = vem_input.read_vint()?;
            let num_vectors = vem_input.read_vint()?;
            let data_offset = vem_input.read_vlong()?;

            let mut doc_ids = Vec::with_capacity(num_vectors.max(0) as usize);
            for _ in 0..num_vectors {
                doc_ids.push(vem_input.read_vint()?);
            }

            let mut data_at = SliceInput::new(vec_buf);
            data_at.seek(vec_payload_start + data_offset as usize)?;
            let mut entries = Vec::with_capacity(doc_ids.len());
            for doc_id in doc_ids {
                let mut v = Vec::with_capacity(dimension as usize);
                for _ in 0..dimension {
                    let bits = data_at.read_i32()? as u32;
                    v.push(f32::from_bits(bits));
                }
                entries.push((doc_id, v));
            }

            fields.push(VectorField {
                field_number,
                similarity,
                dimension,
                entries,
            });
        }

        Ok(FlatVectorsReader { fields })
    }

    pub fn field(&self, field_number: i32) -> Option<&VectorField> {
        self.fields.iter().find(|f| f.field_number == field_number)
    }

    pub fn fields(&self) -> &[VectorField] {
        &self.fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEG_ID: [u8; ID_LENGTH] = [7u8; ID_LENGTH];
    const SUFFIX: &str = "t1";

    fn sample_fields() -> Vec<FieldVectors> {
        vec![
            FieldVectors {
                field_number: 0,
                similarity: VectorSimilarityFunction::Euclidean,
                dimension: 2,
                vectors: vec![
                    (1, vec![0.0, 0.0]),
                    (2, vec![1.0, 0.0]),
                    (3, vec![10.0, 10.0]),
                ],
            },
            FieldVectors {
                field_number: 3,
                similarity: VectorSimilarityFunction::DotProduct,
                dimension: 3,
                vectors: vec![(5, vec![1.0, 0.0, 0.0]), (6, vec![0.0, 1.0, 0.0])],
            },
        ]
    }

    #[test]
    fn round_trip_write_read() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let reader = FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX).unwrap();

        let f0 = reader.field(0).unwrap();
        assert_eq!(f0.dimension, 2);
        assert_eq!(f0.len(), 3);
        assert!(!f0.is_empty());
        assert_eq!(f0.vector(1), Some([0.0, 0.0].as_slice()));
        assert_eq!(f0.vector(2), Some([1.0, 0.0].as_slice()));
        assert_eq!(f0.vector(999), None);

        let f3 = reader.field(3).unwrap();
        assert_eq!(f3.dimension, 3);
        assert_eq!(f3.vector(5), Some([1.0, 0.0, 0.0].as_slice()));

        assert!(reader.field(42).is_none());
        assert_eq!(reader.fields().len(), 2);
    }

    #[test]
    fn brute_force_search_euclidean_known_top_k() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let reader = FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX).unwrap();
        let f0 = reader.field(0).unwrap();

        // Query at origin: doc 1 is exact match (score 1.0), doc 2 is
        // distance 1 (score 0.5), doc 3 is far (score close to 0).
        let results = f0.search(&[0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(results[1].0, 2);
        assert!((results[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn brute_force_search_dot_product_known_scores() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let reader = FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX).unwrap();
        let f3 = reader.field(3).unwrap();

        // Query [1,0,0]: dot with doc5=[1,0,0] is 1 -> score (1+1)/2=1.0;
        // dot with doc6=[0,1,0] is 0 -> score (1+0)/2=0.5.
        let results = f3.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (5, 1.0));
        assert_eq!(results[1].0, 6);
        assert!((results[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let reader = FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX).unwrap();
        assert!(reader.field(0).unwrap().search(&[0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn search_on_empty_field_returns_empty() {
        let fields = vec![FieldVectors {
            field_number: 0,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            vectors: vec![],
        }];
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let reader = FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX).unwrap();
        let f0 = reader.field(0).unwrap();
        assert!(f0.is_empty());
        assert!(f0.search(&[0.0, 0.0], 3).is_empty());
    }

    #[test]
    fn write_rejects_dimension_mismatch() {
        let fields = vec![FieldVectors {
            field_number: 0,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            vectors: vec![(1, vec![1.0, 2.0, 3.0])],
        }];
        assert!(matches!(
            write_vectors(&fields, &SEG_ID, SUFFIX),
            Err(Error::DimensionMismatch(0, 2, 3))
        ));
    }

    #[test]
    fn write_rejects_empty_vector() {
        let fields = vec![FieldVectors {
            field_number: 0,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 2,
            vectors: vec![(1, vec![])],
        }];
        assert!(matches!(
            write_vectors(&fields, &SEG_ID, SUFFIX),
            Err(Error::EmptyVector(0))
        ));
    }

    #[test]
    fn open_rejects_wrong_segment_id() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(FlatVectorsReader::open(&vec_bytes, &vem_bytes, &wrong_id, SUFFIX).is_err());
    }

    #[test]
    fn open_rejects_truncated_footer() {
        let fields = sample_fields();
        let (vec_bytes, vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        let truncated = &vec_bytes[..vec_bytes.len() - 1];
        assert!(FlatVectorsReader::open(truncated, &vem_bytes, &SEG_ID, SUFFIX).is_err());
    }

    #[test]
    fn open_rejects_corrupted_meta_similarity_byte() {
        let fields = sample_fields();
        let (vec_bytes, mut vem_bytes) = write_vectors(&fields, &SEG_ID, SUFFIX).unwrap();
        // Locate the first field's similarity byte and corrupt it to an
        // out-of-range value. Layout after IndexHeader: NumFields(vint),
        // FieldNumber(vint), Encoding(u8), Similarity(u8), ...
        // Find header length via a fresh index-header check to get the
        // exact payload start.
        let mut probe = SliceInput::new(&vem_bytes);
        codec_util::check_index_header(
            &mut probe,
            META_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            &SEG_ID,
            SUFFIX,
        )
        .unwrap();
        let payload_start = probe.position();
        // NumFields is a single-byte vint (2 fields), FieldNumber(0) is a
        // single-byte vint (0), Encoding is next byte, Similarity follows.
        let similarity_byte_pos = payload_start + 1 + 1 + 1;
        vem_bytes[similarity_byte_pos] = 0xFF;
        // Footer checksum now covers stale data; recompute is unnecessary
        // for this test since checksum failure would also be an acceptable
        // (different) rejection reason, but we want to specifically hit the
        // similarity decode error, so recompute the footer checksum.
        let footer_start = vem_bytes.len() - codec_util::FOOTER_LENGTH;
        let checksum = crc32fast::hash(&vem_bytes[..footer_start + 8]) as u64;
        vem_bytes[footer_start + 8..].copy_from_slice(&checksum.to_be_bytes());

        assert!(matches!(
            FlatVectorsReader::open(&vec_bytes, &vem_bytes, &SEG_ID, SUFFIX),
            Err(Error::Store(lucene_store::Error::Corrupted(_)))
        ));
    }

    #[test]
    fn score_functions_match_hand_computed_values() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(
            (VectorSimilarityFunction::Euclidean.score(&a, &b) - (1.0 / 3.0)).abs() < 1e-6,
            "square distance 2 -> 1/(1+2)=1/3"
        );
        assert!((VectorSimilarityFunction::DotProduct.score(&a, &b) - 0.5).abs() < 1e-6);
        assert!((VectorSimilarityFunction::Cosine.score(&a, &b) - 0.5).abs() < 1e-6);

        let c = [2.0f32, 0.0, 0.0];
        // dot(a,c) = 2 >= 0 -> 1 + 2 = 3
        assert!((VectorSimilarityFunction::MaximumInnerProduct.score(&a, &c) - 3.0).abs() < 1e-6);
        let neg = [-2.0f32, 0.0, 0.0];
        // dot(a,neg) = -2 < 0 -> 1 / (1 - (-2)) = 1/3
        assert!(
            (VectorSimilarityFunction::MaximumInnerProduct.score(&a, &neg) - (1.0 / 3.0)).abs()
                < 1e-6
        );
    }

    #[test]
    fn dot_product_score_floors_at_zero_for_unnormalized_vectors() {
        // Real Lucene clamps DOT_PRODUCT's rescaled score at 0
        // (Math.max((1+dot)/2, 0)) -- only observable for unnormalized
        // vectors, where dot can fall below -1 and (1+dot)/2 would
        // otherwise go negative. a=[2,0,0], b=[-2,0,0]: dot=-4,
        // (1+-4)/2 = -1.5, clamped to 0.
        let a = [2.0f32, 0.0, 0.0];
        let b = [-2.0f32, 0.0, 0.0];
        assert_eq!(VectorSimilarityFunction::DotProduct.score(&a, &b), 0.0);
    }

    #[test]
    fn cosine_zero_vector_scores_midpoint() {
        // Raw cosine is defined as 0 for a zero-magnitude vector (avoids a
        // div-by-zero); after Lucene's (1+cos)/2 rescale that's the
        // 0.5 midpoint, not 0.0.
        let zero = [0.0f32, 0.0, 0.0];
        let other = [1.0f32, 0.0, 0.0];
        assert_eq!(VectorSimilarityFunction::Cosine.score(&zero, &other), 0.5);
    }

    #[test]
    fn identical_vectors_euclidean_score_is_one() {
        let v = [3.0f32, -2.5, 7.0];
        assert!((VectorSimilarityFunction::Euclidean.score(&v, &v) - 1.0).abs() < 1e-6);
    }
}
