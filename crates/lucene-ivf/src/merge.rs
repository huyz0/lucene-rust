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
/// numbers, clustered and quantized afresh with `factory`.
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
    let mut merged: Vec<(i32, Vec<f32>)> = Vec::new();
    for source in sources {
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
                merged.push((new_doc, source.reader.slot_vector(number, slot)?));
            }
        }
    }
    let Some((similarity, dimension)) = shape else {
        return Ok(());
    };
    merged.sort_unstable_by_key(|(d, _)| *d);
    if merged.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(Error::IllegalArgument(format!(
            "field {number}: two sources map onto the same document"
        )));
    }
    let docs: Vec<i32> = merged.iter().map(|(d, _)| *d).collect();
    let vectors: Vec<f32> = merged.into_iter().flat_map(|(_, v)| v).collect();
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
