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
/// (BBQ), 2, 4 or 7 bits, or Extended RaBitQ at 2 to 8 bits; `None` for an
/// unknown number.
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
            self.factory.as_ref(),
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
            } else {
                "osq"
            };
            assert_eq!(factory.name(), want, "{name}");
        }
        assert!(quantizer_factory(IVF_QUANTIZERS.len() as u8).is_none());
    }
}
