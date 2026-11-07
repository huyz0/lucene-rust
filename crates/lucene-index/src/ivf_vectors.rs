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
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::QuantizerRegistry;

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
    any: bool,
}

impl IvfGroupWriter {
    /// A writer for `format`'s IVF files (`None` for another format).
    ///
    /// # Errors
    /// A cluster size or rerank encoding [`KnnVectorsFormat::ivf`] refuses.
    pub(crate) fn new(
        format: KnnVectorsFormat,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
    ) -> Result<Option<Self>> {
        let Some((target_cluster_size, rerank_encoding)) = format.ivf_params() else {
            return Ok(None);
        };
        KnnVectorsFormat::ivf(target_cluster_size, rerank_encoding)
            .map_err(|e| lucene_ivf::Error::IllegalArgument(e.0))?;
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
        lucene_ivf::merge_field_with_vectors(
            &mut self.writer,
            &FieldVectors {
                number: field.number,
                similarity: util_similarity(field.similarity),
                dimension: field.dimension,
                docs: field.docs,
                vectors: field.vectors,
            },
            &merge_sources,
            &OsqFactory::bbq(),
            &params,
        )?;
        self.any = true;
        Ok(())
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
