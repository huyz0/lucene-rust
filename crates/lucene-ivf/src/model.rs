//! A model shared by the fields of many segments: the coarse centroids and
//! the quantizer a field's postings are encoded against, held once instead
//! of once per segment.
//!
//! A model-mapped field's `.ivm` entry names its model (id and checksum) and
//! keeps only what is specific to the segment: the cluster sizes and radii.
//! The reader is handed the model ([`SharedModel`]) when it opens, and every
//! field of every segment that names the same model shares its centroids,
//! centroid routing structures and quantizer ([`IvfField::model`]).
//!
//! [`IvfField::model`]: crate::IvfField::model

use std::sync::Arc;

use lucene_vector_quant::Quantizer;
use rayon::prelude::*;

use crate::kmeans::Metric;
use crate::routing::{self, Router};
use crate::{Error, Result, Routing, Similarity};

/// What a field's search needs of its centroids and quantizer: shared by the
/// fields of a model, owned by an ordinary field.
#[derive(Debug, Clone)]
pub(crate) struct Resident {
    pub(crate) quantizer: Arc<dyn Quantizer>,
    pub(crate) centroids: Vec<f32>,
    /// The centroids as the quantizer encodes and scores against them
    /// (`Quantizer::prepare_centroid`), `prepared_len` floats each, mapped
    /// once; empty when the quantizer uses centroids as they are.
    pub(crate) prepared: Vec<f32>,
    pub(crate) prepared_len: usize,
    /// Ranks the centroids for a query.
    pub(crate) router: Router,
    /// Each centroid's parent in the routing tree, when there is one.
    pub(crate) tree_parents: Option<Vec<u32>>,
}

impl Resident {
    pub(crate) fn new(
        quantizer: Arc<dyn Quantizer>,
        centroids: Vec<f32>,
        dimension: usize,
        similarity: Similarity,
        routing: Routing,
        tree_parents: Option<Vec<u32>>,
    ) -> Self {
        let (prepared, prepared_len) = prepare_centroids(quantizer.as_ref(), &centroids, dimension);
        Resident {
            router: Router::new(
                &centroids,
                dimension,
                similarity,
                routing,
                tree_parents.as_deref(),
            ),
            quantizer,
            centroids,
            prepared,
            prepared_len,
            tree_parents,
        }
    }
}

/// `centroids` (rows of `dimension` floats) through
/// `Quantizer::prepare_centroid`, laid end to end, and the length of one; empty
/// and 0 unless the quantizer prepares every centroid to one non-zero length.
pub(crate) fn prepare_centroids(
    quantizer: &dyn Quantizer,
    centroids: &[f32],
    dimension: usize,
) -> (Vec<f32>, usize) {
    let rows = centroids.chunks_exact(dimension.max(1));
    let clusters = rows.len();
    let prepared: Vec<Vec<f32>> = rows.map_while(|c| quantizer.prepare_centroid(c)).collect();
    let len = prepared.first().map_or(0, Vec::len);
    if len == 0 || prepared.len() != clusters || prepared.iter().any(|p| p.len() != len) {
        return (Vec::new(), 0);
    }
    (prepared.concat(), len)
}

/// A trained model: `nlist` centroids and the quantizer that encodes a
/// vector's residual against its centroid. Built once, then shared by every
/// segment's field that names it (`Arc`).
#[derive(Debug)]
pub struct SharedModel {
    id: String,
    crc: u32,
    dimension: usize,
    similarity: Similarity,
    nlist: usize,
    pub(crate) resident: Arc<Resident>,
}

impl SharedModel {
    /// A model named `id`, whose serialized form checksums to `crc` (what a
    /// segment records, so one written under another model is refused).
    /// `centroids` are `nlist * dimension` floats; for `DotProduct` and
    /// `Cosine` they should be spherical ([`Metric::InnerProduct`]), as a
    /// field is assigned by the inner product with the unit centroid. The
    /// routing tree ([`routing::tree_parents`]) is built here, once, from
    /// [`routing::TREE_MIN_CLUSTERS`] clusters on.
    ///
    /// # Errors
    ///
    /// [`Error::IllegalArgument`] for an empty id, no centroids, centroids
    /// that are not whole vectors or not finite, or a quantizer of another
    /// dimension or similarity.
    pub fn new(
        id: impl Into<String>,
        crc: u32,
        dimension: usize,
        similarity: Similarity,
        centroids: Vec<f32>,
        quantizer: Arc<dyn Quantizer>,
    ) -> Result<Arc<Self>> {
        let id = id.into();
        let bad = |why: &str| Err(Error::IllegalArgument(format!("model [{id}]: {why}")));
        if id.is_empty() {
            return Err(Error::IllegalArgument("a model needs an id".into()));
        }
        if dimension == 0 || centroids.is_empty() || !centroids.len().is_multiple_of(dimension) {
            return bad("centroids are not whole vectors of a positive dimension");
        }
        if centroids.iter().any(|x| !x.is_finite()) {
            return bad("centroids must be finite");
        }
        if quantizer.dimension() != dimension || quantizer.similarity() != similarity {
            return bad("the quantizer's dimension or similarity differs from the model's");
        }
        let nlist = centroids.len().checked_div(dimension).unwrap_or(0);
        let tree = (nlist >= routing::TREE_MIN_CLUSTERS)
            .then(|| routing::tree_parents(&centroids, dimension, similarity));
        let resident = Resident::new(
            quantizer,
            centroids,
            dimension,
            similarity,
            Routing::Auto,
            tree,
        );
        Ok(Arc::new(SharedModel {
            id,
            crc,
            dimension,
            similarity,
            nlist,
            resident: Arc::new(resident),
        }))
    }

    /// The model's id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The checksum a segment records for this model.
    pub fn crc(&self) -> u32 {
        self.crc
    }

    /// The vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The similarity the model's fields score with.
    pub fn similarity(&self) -> Similarity {
        self.similarity
    }

    /// The number of centroids.
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// The centroids, `nlist * dimension` floats.
    pub fn centroids(&self) -> &[f32] {
        &self.resident.centroids
    }

    /// The quantizer.
    pub fn quantizer(&self) -> &dyn Quantizer {
        self.resident.quantizer.as_ref()
    }

    /// The quantizer, shared.
    pub fn quantizer_arc(&self) -> Arc<dyn Quantizer> {
        Arc::clone(&self.resident.quantizer)
    }

    /// Bytes the resident model holds: centroids, their prepared form and
    /// the routing structure's parent table.
    pub fn resident_bytes(&self) -> usize {
        let r = &self.resident;
        r.centroids
            .len()
            .saturating_add(r.prepared.len())
            .saturating_mul(4)
            .saturating_add(
                r.tree_parents
                    .as_ref()
                    .map_or(0, |p| p.len().saturating_mul(4)),
            )
    }

    /// Each vector's nearest centroid (`vectors` laid end to end, already
    /// as [`crate::prepare_vectors`] leaves them), by one matrix product per
    /// block of rows: squared distance for `Euclidean`, the inner product
    /// with the unit centroid otherwise. Ties go to the lower centroid, so
    /// the same vector lands in the same list in every segment and merge.
    // ARITH: `row * nlist` and `c * dimension` index buffers of exactly those
    // lengths; `dot_rows` checks the operand lengths.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn assign(&self, vectors: &[f32]) -> Vec<u32> {
        const BLOCK: usize = 256;
        let (d, k) = (self.dimension, self.nlist);
        let spherical = crate::writer::spherical(self.similarity);
        let metric = if spherical {
            Metric::InnerProduct
        } else {
            Metric::L2
        };
        let mut centroids = self.resident.centroids.clone();
        metric.adjust(&mut centroids, d);
        let half_norms: Vec<f32> = match metric {
            Metric::L2 => centroids
                .chunks_exact(d)
                .map(|c| 0.5 * c.iter().map(|v| v * v).sum::<f32>())
                .collect(),
            Metric::InnerProduct => vec![0.0; k],
        };
        let blocks: Vec<Vec<u32>> = vectors
            .par_chunks(BLOCK * d)
            .map(|rows| {
                let mut scores = vec![0f32; rows.len() / d * k];
                lucene_util::gemm::dot_rows(rows, &centroids, d, &mut scores);
                scores
                    .chunks_exact(k)
                    .map(|row| {
                        let mut best = 0u32;
                        let mut best_s = f32::INFINITY;
                        for (c, (&dot, &h)) in row.iter().zip(&half_norms).enumerate() {
                            let s = h - dot;
                            if s < best_s {
                                best = c as u32;
                                best_s = s;
                            }
                        }
                        best
                    })
                    .collect()
            })
            .collect();
        blocks.concat()
    }
}
