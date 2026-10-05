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
//! but no parent field) is not checked by [`prepare_merge_readers`]: this
//! layer's leaves do not expose `LeafMetaData.hasBlocks`. The writer's own
//! merge checks it before it asks [`SegmentMergeHooks`].
//!
//! [`SegmentMergeHooks`] is how the writer's merge (which reads segment
//! files, below this crate) runs these hooks: it implements
//! `lucene_index::merge_policy::MergeHooks` -- the trait a
//! `lucene_index::merge_policy::OneMerge` carries -- by opening the merge's
//! sources as readers, applying the hooks, and handing back the wrapped
//! readers' live documents and the reorder's `newToOld`.

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

/// A `OneMerge` subclass's hooks as the writer runs them: see the module
/// doc. Attach with `OneMerge::with_hooks(Arc::new(SegmentMergeHooks::new(..)))`.
pub struct SegmentMergeHooks {
    hooks: Arc<dyn MergeReaderHooks>,
}

impl SegmentMergeHooks {
    pub fn new(hooks: Arc<dyn MergeReaderHooks>) -> Self {
        SegmentMergeHooks { hooks }
    }
}

impl std::fmt::Debug for SegmentMergeHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SegmentMergeHooks")
    }
}

impl lucene_index::merge_policy::MergeHooks for SegmentMergeHooks {
    fn prepare(
        &self,
        dir: &dyn lucene_store::Directory,
        sources: &lucene_index::segment_infos::SegmentInfos,
        may_reorder: bool,
    ) -> std::result::Result<lucene_index::merge_policy::PreparedMerge, String> {
        let run = || -> Result<lucene_index::merge_policy::PreparedMerge> {
            let reader = crate::directory_reader::DirectoryReader::open_at(dir, sources.clone())?;
            let readers: Vec<Arc<dyn CodecReader>> = reader
                .segment_readers()
                .iter()
                .map(|r| Arc::new(r.clone()) as Arc<dyn CodecReader>)
                .collect();
            // `open_at` opens one reader per listed segment, in order.
            debug_assert_eq!(readers.len(), sources.segments.len());
            let merge = prepare_merge_readers(self.hooks.as_ref(), readers.clone(), !may_reorder)?;
            // The wrapped readers' live documents: with a reorder, the one
            // sorted view hides them, so they are taken from the wrapped
            // inputs again (wrapping is expected to be deterministic).
            let wrapped: Vec<Arc<dyn CodecReader>> = if merge.reorder_doc_maps.is_some() {
                readers
                    .into_iter()
                    .map(|r| self.hooks.wrap_for_merge(r))
                    .collect::<Result<_>>()?
            } else {
                merge.readers
            };
            let live_docs = wrapped.iter().map(|r| r.live_docs().cloned()).collect();
            let new_to_old = merge.reorder_doc_maps.map(|maps| {
                // `maps[i][d]` is input `i`'s document `d`'s new id.
                let total: usize = maps.iter().map(Vec::len).sum();
                let mut new_to_old = vec![0i32; total];
                let mut base = 0i32;
                for map in &maps {
                    for (d, &new) in map.iter().enumerate() {
                        if let Some(slot) = usize::try_from(new)
                            .ok()
                            .and_then(|n| new_to_old.get_mut(n))
                        {
                            *slot = base + d as i32;
                        }
                    }
                    base += map.len() as i32;
                }
                new_to_old
            });
            Ok(lucene_index::merge_policy::PreparedMerge {
                live_docs,
                new_to_old,
            })
        };
        run().map_err(|e| e.to_string())
    }
}
