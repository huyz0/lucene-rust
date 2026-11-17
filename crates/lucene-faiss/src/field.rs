//! One field of `FaissKnnVectorsFormat` with Lucene's semantics: similarity
//! to Faiss metric, cosine through normalised vectors, doc ids as Faiss ids,
//! Lucene score scaling, and accepted-docs filtering.

use crate::build::build_index;
use crate::error::{Error, Result};
use crate::format::{Index, Metric};
use crate::search::SearchParams;

/// Lucene's `VectorSimilarityFunction`s the Faiss codec accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Similarity {
    /// `EUCLIDEAN`: Faiss `METRIC_L2`.
    Euclidean,
    /// `DOT_PRODUCT`: Faiss `METRIC_INNER_PRODUCT`.
    DotProduct,
    /// `COSINE`: inner product over L2-normalised vectors.
    Cosine,
    /// `MAXIMUM_INNER_PRODUCT`: refused by the sandbox codec.
    MaximumInnerProduct,
}

impl Similarity {
    fn metric(self) -> Result<Metric> {
        match self {
            Similarity::Euclidean => Ok(Metric::L2),
            Similarity::DotProduct | Similarity::Cosine => Ok(Metric::InnerProduct),
            Similarity::MaximumInnerProduct => Err(Error::Invalid(
                "MAXIMUM_INNER_PRODUCT is not supported by the Faiss format".into(),
            )),
        }
    }

    /// Lucene's score for a Faiss distance (`FaissKnnVectorsReader`).
    pub fn score(self, distance: f32) -> f32 {
        match self {
            Similarity::Euclidean => 1.0 / (1.0 + distance),
            _ => ((1.0 + distance) / 2.0).max(0.0),
        }
    }
}

/// A built or read field: the Faiss index and the similarity it answers.
#[derive(Debug)]
pub struct FaissField {
    similarity: Similarity,
    index: Index,
}

/// The factory string `FaissKnnVectorsFormat` uses when none is configured.
pub fn default_description(max_conn: usize) -> String {
    format!("IDMap,HNSW{max_conn}")
}

fn normalise(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

impl FaissField {
    /// Builds the field over `vectors` (`dims` floats per doc) for `doc_ids`.
    /// `description` is a Faiss factory string, or `None` for the format's
    /// default `IDMap,HNSW<max_conn>` with `efConstruction=<beam_width>`.
    ///
    /// # Errors
    /// [`Error::Description`] for an unsupported factory string, and
    /// [`Error::Invalid`] for a refused similarity or mismatched lengths.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        similarity: Similarity,
        dims: usize,
        vectors: &[f32],
        doc_ids: &[i32],
        description: Option<&str>,
        index_params: Option<&str>,
        max_conn: usize,
        beam_width: usize,
    ) -> Result<Self> {
        let metric = similarity.metric()?;
        let ids: Vec<i64> = doc_ids.iter().map(|&d| i64::from(d)).collect();
        let mut owned;
        let vectors = if similarity == Similarity::Cosine {
            owned = vectors.to_vec();
            owned.chunks_exact_mut(dims.max(1)).for_each(normalise);
            &owned[..]
        } else {
            vectors
        };
        let default = default_description(max_conn);
        let params = index_params
            .map(str::to_string)
            .unwrap_or_else(|| format!("efConstruction={beam_width}"));
        let index = build_index(
            description.unwrap_or(&default),
            &params,
            metric,
            dims,
            vectors,
            &ids,
        )?;
        Ok(FaissField { similarity, index })
    }

    /// Wraps an index read from a segment.
    ///
    /// # Errors
    /// [`Error::Invalid`] when the index metric does not suit `similarity`.
    pub fn from_index(similarity: Similarity, index: Index) -> Result<Self> {
        if index.metric() != similarity.metric()? {
            return Err(Error::Invalid(format!(
                "index metric {:?} does not match {similarity:?}",
                index.metric()
            )));
        }
        Ok(FaissField { similarity, index })
    }

    /// The Faiss index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// The `k` best docs for `query` as `(doc id, Lucene score)`, best first;
    /// `accept` filters doc ids (`acceptDocs`).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a query of the wrong length.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        accept: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<Vec<(i32, f32)>> {
        let mut q;
        let query = if self.similarity == Similarity::Cosine {
            q = query.to_vec();
            normalise(&mut q);
            &q[..]
        } else {
            query
        };
        let filter = accept.map(|f| move |id: i64| i32::try_from(id).is_ok_and(f));
        let found = match &filter {
            Some(f) => self.index.search(query, k, params, Some(f))?,
            None => self.index.search(query, k, params, None)?,
        };
        Ok(found
            .into_iter()
            .map(|(id, dist)| (id as i32, self.similarity.score(dist)))
            .collect())
    }
}
