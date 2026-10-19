//! `TaxonomyMergeUtils`: merges one index and its taxonomy into another
//! index and taxonomy, the source's facet ordinals rewritten to the
//! destination taxonomy's.
//!
//! Differs: Java adds the source's leaves through
//! [`super::ordinal_mapping_leaf_reader()`] (`addIndexes(CodecReader...)`);
//! this port's writer merges segments rather than readers, so
//! [`merge`] hands the same map to the merge itself
//! ([`IndexWriter::add_indexes_merged_remapping`]) -- the destination gets
//! the same one new segment of the source's live documents, its facet
//! fields' values mapped and each document's sorted. Copying the source's
//! segments in takes its `write.lock`, as `addIndexes(Directory...)` does,
//! so the source must have no open writer (Java only opens a reader on
//! it); and a destination sorted on a facet field is refused (the merge
//! would sort by the unmapped ordinals).

use lucene_index::index_writer::IndexWriter;
use lucene_index::merge::SortedNumericRemap;
use lucene_store::directory::Directory;

use super::directory::directory_taxonomy_writer::{iw_error, OrdinalMap};
use super::ordinal_mapping_leaf_reader::facet_fields;
use super::{DirectoryTaxonomyWriter, TaxonomyWriter};
use crate::facets::FacetsConfig;
use crate::Result;

/// `TaxonomyMergeUtils.merge(srcIndexDir, srcTaxoDir, map, destIndexWriter,
/// destTaxoWriter, srcConfig)`: adds the source taxonomy to the
/// destination's (recording the ordinals in `map`), the source index's live
/// documents to the destination index with their facet ordinals mapped,
/// then commits the taxonomy and the index, in that order.
///
/// # Errors
/// Whatever adding the taxonomy, merging or committing reports.
pub fn merge(
    src_index_dir: &dyn Directory,
    src_taxo_dir: &dyn Directory,
    map: &mut dyn OrdinalMap,
    dest_index_writer: &mut IndexWriter<'_>,
    dest_taxo_writer: &DirectoryTaxonomyWriter<'_>,
    src_config: &FacetsConfig,
) -> Result<()> {
    dest_taxo_writer.add_taxonomy(src_taxo_dir, map)?;
    let ordinal_map = map.get_map()?.to_vec();
    let mut fields: Vec<String> = facet_fields(src_config).into_iter().collect();
    fields.sort();
    dest_index_writer
        .add_indexes_merged_remapping(
            &[src_index_dir],
            SortedNumericRemap {
                fields,
                map: ordinal_map,
            },
        )
        .map_err(iw_error)?;
    dest_taxo_writer.commit()?;
    dest_index_writer.commit().map_err(iw_error)?;
    Ok(())
}
