//! The IVF files of a [`KnnVectorsFormat::Ivf`] instance (`.ivm`, `.ivp`,
//! `.ivv`, from `lucene-ivf`), written beside the instance's raw vectors at
//! flush and at merge. Only `FLOAT32` fields get an index; a byte field of
//! the instance has its raw vectors alone.
//!
//! A merge takes the merged raw vectors (exact, whatever the sources'
//! rerank encoding) and reuses the clusters of the source holding most of
//! them ([`lucene_ivf::MergeClustering::ReuseLargest`]).
//!
//! [`KnnVectorsFormat::Ivf`]: lucene_codecs::per_field_knn_vectors::KnnVectorsFormat::Ivf

use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::scalar_quantized_vectors::util_similarity;
use lucene_ivf::{FieldVectors, IvfParams, IvfReader, IvfWriter, MergeSource, VectorEncoding};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Input;
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::RotatedFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

type Result<T> = std::result::Result<T, lucene_ivf::Error>;

/// The IVF files' extensions, in the order a group lists them.
pub const EXTENSIONS: [&str; 3] = ["ivm", "ivp", "ivv"];

/// An instance's IVF files by extension.
pub(crate) type Files = Vec<(&'static str, Vec<u8>)>;

/// One float field of an IVF instance: its number, similarity, dimension,
/// documents (ascending) and vectors end to end.
pub(crate) struct IvfField<'a> {
    pub number: i32,
    pub similarity: VectorSimilarityFunction,
    pub dimension: usize,
    pub docs: &'a [i32],
    pub vectors: &'a [f32],
    /// The segment's partitions, when the writer clusters each on its own
    /// ([`crate::index_writer::IndexWriter::set_ivf_partition_clusters`]).
    pub partitions: Option<Partitions<'a>>,
}

/// A segment's documents grouped by partition, for IVF fields clustered per
/// partition.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Partitions<'a> {
    /// Each partition's first document, ascending.
    pub starts: &'a [i32],
    /// [`IvfParams::partition_min_vectors`].
    pub min_vectors: usize,
}

/// A source segment's IVF files, opened by a merge that writes the field
/// as IVF again ([`IvfInputs::open`]).
#[derive(Debug)]
pub struct IvfInputs {
    pub meta: Input,
    pub postings: Arc<Input>,
    pub vectors: Arc<Input>,
    pub segment_id: [u8; ID_LENGTH],
    pub suffix: String,
}

impl IvfInputs {
    /// The reader over these files.
    ///
    /// # Errors
    /// Files that do not decode.
    pub fn open(&self) -> Result<IvfReader> {
        open(
            &self.meta,
            self.postings.clone(),
            self.vectors.clone(),
            &self.segment_id,
            &self.suffix,
        )
    }
}

/// One merge source's IVF index of a field and its documents' new numbers
/// (`-1` for a dropped document).
pub(crate) struct IvfSource<'a> {
    pub reader: &'a IvfReader,
    pub doc_map: &'a [i32],
}

/// Writes one IVF instance's files, field by field.
pub(crate) struct IvfGroupWriter {
    writer: IvfWriter,
    params: IvfParams,
    factory: Box<dyn QuantizerFactory>,
    any: bool,
}

/// The quantizer factory of an [`IVF_QUANTIZERS`] wire number: OSQ at 1
/// (BBQ), 2, 4 or 7 bits, Extended RaBitQ at 2 to 8 bits, or 1-bit RaBitQ
/// (BBQ after a random rotation); `None` for an unknown number.
///
/// [`IVF_QUANTIZERS`]: lucene_codecs::per_field_knn_vectors::IVF_QUANTIZERS
pub fn quantizer_factory(quantizer: u8) -> Option<Box<dyn QuantizerFactory>> {
    let osq = |e| Some(Box::new(OsqFactory::new(e)) as Box<dyn QuantizerFactory>);
    match quantizer {
        0 => osq(ScalarEncoding::SingleBitQueryNibble),
        1 => osq(ScalarEncoding::DibitQueryNibble),
        2 => osq(ScalarEncoding::PackedNibble),
        3 => osq(ScalarEncoding::SevenBit),
        // ARITH: 4..=10 maps onto 2..=8 bits.
        #[allow(clippy::arithmetic_side_effects)]
        4..=10 => Some(Box::new(ExRabitqFactory::new(quantizer - 2))),
        11 => Some(Box::new(RotatedFactory::new(Arc::new(OsqFactory::bbq())))),
        _ => None,
    }
}

impl IvfGroupWriter {
    /// A writer for `format`'s IVF files (`None` for another format).
    ///
    /// # Errors
    /// A cluster size, rerank encoding or quantizer
    /// [`KnnVectorsFormat::ivf_quantized`] refuses.
    pub(crate) fn new(
        format: KnnVectorsFormat,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> Result<Option<Self>> {
        let Some((target_cluster_size, rerank_encoding, quantizer)) = format.ivf_params() else {
            return Ok(None);
        };
        KnnVectorsFormat::ivf_quantized(target_cluster_size, rerank_encoding, quantizer)
            .map_err(|e| lucene_ivf::Error::IllegalArgument(e.0))?;
        let factory = quantizer_factory(quantizer).ok_or_else(|| {
            lucene_ivf::Error::IllegalArgument(format!("unknown IVF quantizer {quantizer}"))
        })?;
        Ok(Some(IvfGroupWriter {
            writer: IvfWriter::new(segment_id, suffix),
            params: IvfParams {
                target_cluster_size: target_cluster_size as usize,
                rerank_encoding: match rerank_encoding {
                    1 => VectorEncoding::F16,
                    2 => VectorEncoding::Bf16,
                    3 => VectorEncoding::Int8,
                    _ => VectorEncoding::F32,
                },
                ..IvfParams::default()
            },
            factory,
            any: false,
        }))
    }

    /// Writes `field`, clustered from the `sources` when one holds most of
    /// its vectors (none at a flush). Rerank vectors are half floats only
    /// when every value fits them, so a field with larger values keeps
    /// exact ones rather than failing.
    pub(crate) fn add(&mut self, field: &IvfField<'_>, sources: &[IvfSource<'_>]) -> Result<()> {
        let fits = |v: &f32| v.abs() <= 65_504.0;
        let params = if self.params.rerank_encoding == VectorEncoding::F16
            && !field.vectors.iter().all(fits)
        {
            IvfParams {
                rerank_encoding: VectorEncoding::F32,
                ..self.params
            }
        } else {
            self.params
        };
        let maps: Vec<Box<dyn Fn(i32) -> Option<i32> + '_>> = sources
            .iter()
            .map(|s| {
                let map = s.doc_map;
                Box::new(move |doc: i32| {
                    usize::try_from(doc)
                        .ok()
                        .and_then(|d| map.get(d))
                        .copied()
                        .filter(|&d| d >= 0)
                }) as Box<dyn Fn(i32) -> Option<i32>>
            })
            .collect();
        let merge_sources: Vec<MergeSource<'_>> = sources
            .iter()
            .zip(&maps)
            .map(|(s, map)| MergeSource {
                reader: s.reader,
                doc_map: map.as_ref(),
            })
            .collect();
        let vectors = FieldVectors {
            number: field.number,
            similarity: util_similarity(field.similarity),
            dimension: field.dimension,
            docs: field.docs,
            vectors: field.vectors,
        };
        // Partitions are clustered afresh: a source's clusters mix the
        // partitions a merge brings together.
        match field.partitions {
            Some(partitions) => self.writer.add_partitioned_field(
                &vectors,
                partitions.starts,
                self.factory.as_ref(),
                &IvfParams {
                    partition_min_vectors: partitions.min_vectors,
                    ..params
                },
            )?,
            None => lucene_ivf::merge_field_with_vectors(
                &mut self.writer,
                &vectors,
                &merge_sources,
                self.factory.as_ref(),
                &params,
            )?,
        }
        self.any = true;
        Ok(())
    }

    /// The clusters of the fields written so far.
    pub(crate) fn clusters(&self) -> &[lucene_ivf::FieldClusters] {
        self.writer.clusters()
    }

    /// The three files, `None` when no float field was written.
    pub(crate) fn finish(self) -> Option<Files> {
        self.any.then(|| {
            let files = self.writer.finish();
            vec![
                (EXTENSIONS[0], files.meta),
                (EXTENSIONS[1], files.postings),
                (EXTENSIONS[2], files.vectors),
            ]
        })
    }
}

/// Each IVF field's clusters from the last flush that wrote it
/// ([`IndexingConfig::buffer_lists_for`]).
///
/// [`IndexingConfig::buffer_lists_for`]: crate::index_writer::IndexingConfig::buffer_lists_for
#[derive(Debug, Default)]
pub(crate) struct ClusterSink {
    fields: std::sync::Mutex<Vec<(i32, Arc<BufferRouter>)>>,
}

impl ClusterSink {
    /// Keeps `clusters`, replacing their fields' earlier ones unless those
    /// have more than twice as many clusters: flushes finish in any order,
    /// and a small one (a refresh of a few documents) would leave later
    /// buffers little to prune by.
    pub(crate) fn record(&self, clusters: &[lucene_ivf::FieldClusters]) {
        let mut fields = self
            .fields
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for c in clusters.iter().filter(|c| !c.is_empty()) {
            let router = Arc::new(BufferRouter::new(c.clone()));
            match fields.iter_mut().find(|(n, _)| *n == c.number) {
                Some(slot) if slot.1.clusters.len() > c.len().saturating_mul(2) => {}
                Some(slot) => slot.1 = router,
                None => fields.push((c.number, router)),
            }
        }
    }

    /// Field `number`'s router, if a flush wrote its clusters.
    pub(crate) fn get(
        &self,
        number: i32,
    ) -> Option<Arc<dyn lucene_codecs::buffer_lists::ClusterRouter>> {
        let fields = self
            .fields
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        fields
            .iter()
            .find(|(n, _)| *n == number)
            .map(|(_, r)| Arc::clone(r) as Arc<dyn lucene_codecs::buffer_lists::ClusterRouter>)
    }
}

/// How far past its cluster's radius a buffered vector may be and still
/// join it: farther, it is an outlier every search scores.
const OUTLIER_RADII: f32 = 1.5;

/// A flushed field's clusters as a buffer routes to them: the reader's
/// router, and a vector beyond [`OUTLIER_RADII`] of its nearest cluster's
/// radius (or of the mean radius, for a cluster of one) left out.
#[derive(Debug)]
pub(crate) struct BufferRouter {
    clusters: lucene_ivf::FieldClusters,
    router: lucene_ivf::routing::Router,
    mean_radius: f32,
    nprobe: usize,
}

impl BufferRouter {
    fn new(clusters: lucene_ivf::FieldClusters) -> Self {
        let n = clusters.radii.len().max(1);
        let mean_radius = clusters.radii.iter().sum::<f32>() / n as f32;
        BufferRouter {
            router: clusters.router(),
            nprobe: clusters.nprobe(),
            mean_radius,
            clusters,
        }
    }
}

impl lucene_codecs::buffer_lists::ClusterRouter for BufferRouter {
    fn clusters(&self) -> usize {
        self.clusters.len()
    }

    fn assign(&self, v: &[f32]) -> Option<usize> {
        let normalized;
        let v = if self.clusters.similarity == lucene_ivf::Similarity::Cosine {
            let mut u = v.to_vec();
            // A zero vector has no direction and stays zero.
            let _ = lucene_util::vector_util::l2normalize(&mut u, false);
            normalized = u;
            &normalized[..]
        } else {
            v
        };
        let dim = self.clusters.dimension;
        let c = if self.clusters.similarity == lucene_ivf::Similarity::MaximumInnerProduct {
            // Its clusters are Euclidean (k-means' `L2` metric, its radii
            // L2 distances): the nearest centroid, not the best product.
            self.clusters
                .centroids
                .chunks_exact(dim.max(1))
                .map(|m| lucene_util::simd::square_distance_f32(v, m))
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))?
                .0
        } else {
            *self.router.route(&self.clusters.centroids, v, 1).first()?
        };
        let centroid = self
            .clusters
            .centroids
            .get(c.checked_mul(dim)?..c.checked_add(1)?.checked_mul(dim)?)?;
        let radius = self
            .clusters
            .radii
            .get(c)
            .copied()
            .filter(|&r| r > 0.0)
            .unwrap_or(self.mean_radius);
        let far = OUTLIER_RADII * radius;
        (lucene_util::simd::square_distance_f32(v, centroid) <= far * far).then_some(c)
    }

    fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        self.router.route(&self.clusters.centroids, query, nprobe)
    }

    fn default_nprobe(&self) -> usize {
        self.nprobe
    }
}

/// Opens an instance's IVF index from its three files.
///
/// # Errors
/// Files that do not decode.
pub fn open(
    meta: &Input,
    postings: Arc<Input>,
    vectors: Arc<Input>,
    segment_id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<IvfReader> {
    IvfReader::open(
        meta,
        postings,
        vectors,
        segment_id,
        suffix,
        &QuantizerRegistry::with_builtins(),
    )
}

#[cfg(test)]
mod tests {
    use lucene_codecs::per_field_knn_vectors::IVF_QUANTIZERS;

    use super::quantizer_factory;

    #[test]
    fn every_ivf_quantizer_number_has_its_factory() {
        for (number, name) in IVF_QUANTIZERS.iter().enumerate() {
            let factory = quantizer_factory(number as u8).expect(name);
            let want = if name.starts_with("erq") {
                "erq"
            } else if *name == "rabitq" {
                "rot-osq"
            } else {
                "osq"
            };
            assert_eq!(factory.name(), want, "{name}");
        }
        assert!(quantizer_factory(IVF_QUANTIZERS.len() as u8).is_none());
    }
}
