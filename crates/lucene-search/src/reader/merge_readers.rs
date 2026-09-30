//! `MergePolicy.OneMerge`'s reader hooks -- `wrapForMerge(CodecReader)` and
//! `reorder(CodecReader, Directory, Executor)` -- and the part of
//! `IndexWriter.mergeMiddle` that applies them to a merge's readers.
//!
//! These live here, not beside `lucene_index::merge_policy::OneMerge`,
//! because they take and return [`CodecReader`]s, which `lucene-index` cannot
//! see (the dependency graph runs `index <- search`). A merge that wants them
//! reads its segments through [`prepare_merge_readers`]: every reader is
//! wrapped, and, when there is no index sort, the merged view
//! ([`SlowCompositeCodecReaderWrapper`]) is offered to `reorder`; a map it
//! returns turns the inputs into one [`SortingCodecReader`] over that view,
//! with the per-input doc maps `mergeMiddle` keeps as `reorderDocMaps`.
//!
//! `hasBlocksButNoParentField` (no reorder of an index with document blocks
//! but no parent field) is not checked: this layer's leaves do not expose
//! `LeafMetaData.hasBlocks`.

use std::sync::Arc;

use super::slow_codec::SlowCompositeCodecReaderWrapper;
use super::sorting::{DocMap, SortingCodecReader};
use super::CodecReader;
use crate::Result;

/// `OneMerge`'s overridable reader hooks; both default to Java's.
pub trait MergeReaderHooks: Send + Sync {
    /// `wrapForMerge(reader)`: the reader as the merge should read it
    /// (the reader itself by default).
    ///
    /// # Errors
    /// The hook's own.
    fn wrap_for_merge(&self, reader: Arc<dyn CodecReader>) -> Result<Arc<dyn CodecReader>> {
        Ok(reader)
    }

    /// `reorder(reader, dir, executor)`: a renumbering of the merged view's
    /// documents, or `None` (the default) to keep their order.
    ///
    /// # Errors
    /// The hook's own.
    fn reorder(&self, _reader: &dyn CodecReader) -> Result<Option<DocMap>> {
        Ok(None)
    }
}

/// The hooks a plain `OneMerge` has: none.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultMergeHooks;

impl MergeReaderHooks for DefaultMergeHooks {}

/// What a merge reads after its hooks ran.
pub struct MergeReaders {
    /// The readers to merge: the wrapped inputs, or the one reordered view.
    pub readers: Vec<Arc<dyn CodecReader>>,
    /// `reorderDocMaps`: when the merge was reordered, for each input reader
    /// its documents' new ids (`docMap.oldToNew(docBase + doc)`).
    pub reorder_doc_maps: Option<Vec<Vec<i32>>>,
}

/// `IndexWriter.mergeMiddle`'s reader preparation: `wrapForMerge` on every
/// input, then -- without an index sort -- `reorder` over the merged view.
///
/// # Errors
/// What the hooks or the views report.
pub fn prepare_merge_readers(
    hooks: &dyn MergeReaderHooks,
    readers: Vec<Arc<dyn CodecReader>>,
    has_index_sort: bool,
) -> Result<MergeReaders> {
    let readers = readers
        .into_iter()
        .map(|r| hooks.wrap_for_merge(r))
        .collect::<Result<Vec<_>>>()?;
    if has_index_sort || readers.is_empty() {
        return Ok(MergeReaders {
            readers,
            reorder_doc_maps: None,
        });
    }
    let merged_view = SlowCompositeCodecReaderWrapper::wrap(readers.clone())?;
    let Some(doc_map) = hooks.reorder(merged_view.as_ref())? else {
        return Ok(MergeReaders {
            readers,
            reorder_doc_maps: None,
        });
    };
    let mut maps = Vec::with_capacity(readers.len());
    let mut doc_base = 0i32;
    for r in &readers {
        let n = r.max_doc();
        maps.push((0..n).map(|d| doc_map.old_to_new(doc_base + d)).collect());
        doc_base += n;
    }
    let sorted = SortingCodecReader::wrap(merged_view, Some(doc_map), Vec::new())?;
    Ok(MergeReaders {
        readers: vec![Arc::new(sorted)],
        reorder_doc_maps: Some(maps),
    })
}
