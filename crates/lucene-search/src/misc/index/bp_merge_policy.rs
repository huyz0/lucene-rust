//! `BPReorderingMergePolicy`: a merge policy that renumbers the documents of
//! the segments it merges with a [`BpIndexReorderer`]. A merge too small to
//! be worth it is left in order, and so is one the reorderer has not the RAM
//! for -- reordering must not fail a merge.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_index::merge_policy::api::{Error as PolicyError, Result as PolicyResult};
use lucene_index::merge_policy::filter::FilterMergePolicy;
use lucene_index::merge_policy::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification, MergeTrigger,
};

use super::bp::{BpIndexReorderer, ReorderError};
use crate::reader::merge_readers::{MergeReaderHooks, SegmentMergeHooks};
use crate::reader::sorting::DocMap;
use crate::reader::CodecReader;
use crate::{Error, Result};

/// `REORDERED`: the diagnostics key Java records on a merged segment.
pub const REORDERED: &str = "bp.reordered";

/// `BPReorderingMergePolicy`.
#[derive(Debug)]
pub struct BpReorderingMergePolicy {
    filter: FilterMergePolicy,
    reorderer: Arc<BpIndexReorderer>,
    min_natural_merge_num_docs: i32,
    min_natural_merge_ratio_from_biggest_segment: f32,
}

/// What a wrapped `OneMerge`'s `reorder` does.
struct BpHooks {
    reorderer: Arc<BpIndexReorderer>,
    min_num_docs: i32,
}

impl MergeReaderHooks for BpHooks {
    fn reorder(&self, reader: &dyn CodecReader) -> Result<Option<DocMap>> {
        if reader.num_docs() < self.min_num_docs {
            return Ok(None);
        }
        match self.reorderer.compute_doc_map(reader) {
            Ok(map) => Ok(Some(map)),
            // Skip reordering, we don't have enough RAM anyway.
            Err(ReorderError::NotEnoughRam(_)) => Ok(None),
            Err(ReorderError::Failed(e)) => Err(e),
        }
    }
}

impl BpReorderingMergePolicy {
    /// `new BPReorderingMergePolicy(in, reorderer)`: every merge `inner`
    /// proposes above the configured size, and every forced merge, is
    /// reordered.
    pub fn new(inner: Box<dyn MergePolicy>, reorderer: BpIndexReorderer) -> Self {
        Self {
            filter: FilterMergePolicy::new(inner),
            reorderer: Arc::new(reorderer),
            min_natural_merge_num_docs: 1,
            min_natural_merge_ratio_from_biggest_segment: 0.0,
        }
    }

    /// `setMinNaturalMergeNumDocs`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] below 1.
    pub fn set_min_natural_merge_num_docs(&mut self, n: i32) -> Result<()> {
        if n < 1 {
            return Err(Error::IllegalArgument(format!(
                "minNaturalMergeNumDocs must be at least 1, got {n}"
            )));
        }
        self.min_natural_merge_num_docs = n;
        Ok(())
    }

    /// `setMinNaturalMergeRatioFromBiggestSegment`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] outside `[0, 1)`.
    pub fn set_min_natural_merge_ratio_from_biggest_segment(&mut self, ratio: f32) -> Result<()> {
        if !(0.0..1.0).contains(&ratio) {
            return Err(Error::IllegalArgument(format!(
                "minNaturalMergeRatioFromBiggestSegment must be in [0, 1), got {ratio:?}"
            )));
        }
        self.min_natural_merge_ratio_from_biggest_segment = ratio;
        Ok(())
    }

    /// `maybeReorder`: each merge of `spec` carries the reordering hooks.
    fn maybe_reorder(
        &self,
        spec: Option<MergeSpecification>,
        forced: bool,
        infos: Option<&[MergeSegment]>,
    ) -> PolicyResult<Option<MergeSpecification>> {
        let Some(spec) = spec else {
            return Ok(None);
        };
        let min_num_docs = if forced {
            // No minimum size for forced merges.
            1
        } else {
            let max_max_doc = infos
                .unwrap_or_default()
                .iter()
                .map(|s| s.max_doc)
                .fold(0, i32::max);
            self.min_natural_merge_num_docs.max(
                (f64::from(self.min_natural_merge_ratio_from_biggest_segment)
                    * f64::from(max_max_doc)) as i32,
            )
        };
        let hooks = Arc::new(SegmentMergeHooks::new(Arc::new(BpHooks {
            reorderer: Arc::clone(&self.reorderer),
            min_num_docs,
        })));
        let mut out = MergeSpecification::new();
        for merge in spec.merges {
            if merge.hooks.is_some() {
                return Err(PolicyError::IllegalArgument(
                    "the wrapped merge policy's merges already carry reader hooks".to_string(),
                ));
            }
            out.add(merge.with_hooks(hooks.clone()));
        }
        Ok(Some(out))
    }
}

impl MergePolicy for BpReorderingMergePolicy {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> PolicyResult<Option<MergeSpecification>> {
        self.maybe_reorder(
            self.filter.find_merges(trigger, infos, ctx)?,
            false,
            Some(infos),
        )
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> PolicyResult<Option<MergeSpecification>> {
        self.maybe_reorder(
            self.filter
                .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)?,
            true,
            Some(infos),
        )
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> PolicyResult<Option<MergeSpecification>> {
        self.maybe_reorder(
            self.filter.find_forced_deletes_merges(infos, ctx)?,
            true,
            Some(infos),
        )
    }

    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        // `addIndexes` is considered a forced merge.
        self.maybe_reorder(self.filter.find_merges_for_readers(max_docs), true, None)
            .ok()
            .flatten()
    }

    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> PolicyResult<Option<MergeSpecification>> {
        self.maybe_reorder(
            self.filter.find_full_flush_merges(trigger, infos, ctx)?,
            false,
            Some(infos),
        )
    }

    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> PolicyResult<bool> {
        self.filter.use_compound_file(infos, merged, ctx)
    }

    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> PolicyResult<i64> {
        self.filter.size(info, ctx)
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        self.filter.max_full_flush_merge_size()
    }

    fn keep_fully_deleted_segment(&self) -> bool {
        self.filter.keep_fully_deleted_segment()
    }

    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.filter.num_deletes_to_merge(info, del_count)
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.filter.compound_file_settings()
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.filter.compound_file_settings_mut()
    }
}
