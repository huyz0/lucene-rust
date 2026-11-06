//! Merging IVF fields: the merged segment's vectors are re-clustered and
//! re-quantized from scratch, never patched in place. Segments are
//! immutable, so a merge is where the clusters adapt to the data that
//! arrived since the inputs were written.

use lucene_vector_quant::QuantizerFactory;

use crate::reader::IvfReader;
use crate::writer::{FieldVectors, IvfParams, IvfWriter};
use crate::{Error, Result};

/// One merged segment's field: its reader and how its documents map into
/// the merged segment (`None` for a deleted document).
pub struct MergeSource<'a> {
    /// The input segment.
    pub reader: &'a IvfReader,
    /// Old document to new document, `None` when the document is dropped.
    pub doc_map: &'a dyn Fn(i32) -> Option<i32>,
}

impl std::fmt::Debug for MergeSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeSource")
            .field("reader", self.reader)
            .finish_non_exhaustive()
    }
}

/// Writes field `number` of the merged segment into `writer`: the live
/// vectors of every source that has the field, under their new document
/// numbers, clustered and quantized afresh with `factory`. Sources may store
/// their vectors in any [`crate::VectorEncoding`]; they are decoded, and
/// written in `params.rerank_encoding`.
///
/// # Errors
///
/// Sources that disagree on the field's similarity or dimension, two
/// sources mapping onto the same document, a read failure, or the writer's
/// own errors. A field no source has is not written.
pub fn merge_field(
    writer: &mut IvfWriter,
    number: i32,
    sources: &[MergeSource<'_>],
    factory: &dyn QuantizerFactory,
    params: &IvfParams,
) -> Result<()> {
    let mut shape = None;
    // (new doc, source, slot) for every live vector.
    let mut live: Vec<(i32, usize, u32)> = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let Some(field) = source.reader.field(number) else {
            continue;
        };
        let this = (field.similarity(), field.dimension());
        if *shape.get_or_insert(this) != this {
            return Err(Error::IllegalArgument(format!(
                "field {number}: sources disagree on similarity or dimension"
            )));
        }
        for (doc, slot) in source.reader.doc_table(number)? {
            if let Some(new_doc) = (source.doc_map)(doc) {
                live.push((new_doc, i, slot));
            }
        }
    }
    let Some((similarity, dimension)) = shape else {
        return Ok(());
    };
    live.sort_unstable_by_key(|&(d, _, _)| d);
    if live.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(Error::IllegalArgument(format!(
            "field {number}: two sources map onto the same document"
        )));
    }
    let docs: Vec<i32> = live.iter().map(|&(d, _, _)| d).collect();
    let mut vectors = vec![0f32; live.len().saturating_mul(dimension)];
    // One sequential read per source rather than one per vector.
    for (i, source) in sources.iter().enumerate() {
        let Some(field) = source.reader.field(number) else {
            continue;
        };
        let (encoding, stride) = (field.encoding(), field.vector_len());
        let bytes = source.reader.slot_vectors(number)?;
        // ARITH: slots were checked against the field's count at open, so
        // `(slot + 1) * stride` bytes lie inside `bytes`, and `out` indexes
        // `live`, whose length sized `vectors`.
        #[allow(clippy::arithmetic_side_effects)]
        for (out, &(_, _, slot)) in live.iter().enumerate().filter(|(_, l)| l.1 == i) {
            let from = slot as usize * stride;
            encoding.decode(
                &bytes[from..from + stride],
                field.slot_centroid(slot as usize),
                &mut vectors[out * dimension..(out + 1) * dimension],
            );
        }
    }
    writer.add_field(
        &FieldVectors {
            number,
            similarity,
            dimension,
            docs: &docs,
            vectors: &vectors,
        },
        factory,
        params,
    )
}
