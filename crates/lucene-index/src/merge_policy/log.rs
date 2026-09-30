//! Port of `org.apache.lucene.index.LogMergePolicy` and its two concrete
//! subclasses, `LogByteSizeMergePolicy` (segment size in bytes) and
//! `LogDocMergePolicy` (segment size in documents).
//!
//! Java's abstract base with an overridden `size()` becomes one struct,
//! [`LogMergePolicy`], carrying a [`LogSizeUnit`]; the two subclasses are
//! newtypes over it with Java's defaults and unit-specific setters, and
//! `Deref` to the base for the shared ones (`setMergeFactor`,
//! `setMaxMergeDocs`, ...). Every decision method is a line-by-line port;
//! Java's `float`/`double` mix in the level computation is kept operation for
//! operation, since a level on the boundary of `LEVEL_LOG_SPAN` decides which
//! segments merge.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use super::api::{
    default_size, CompoundFileSettings, Error, MergeContext, MergePolicy, MergeSegment,
    MergeSpecification, MergeTrigger, OneMerge, Result,
};

/// What a [`LogMergePolicy`] measures a segment by -- the `size()` override
/// that distinguishes Java's two subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSizeUnit {
    /// `LogByteSizeMergePolicy.size = sizeBytes`.
    Bytes,
    /// `LogDocMergePolicy.size = sizeDocs`.
    Docs,
}

/// `LogMergePolicy`: merges segments of roughly equal size (levels on a log
/// scale of `mergeFactor`), `mergeFactor` at a time.
#[derive(Debug, Clone)]
pub struct LogMergePolicy {
    unit: LogSizeUnit,
    merge_factor: i32,
    /// `minMergeSize`: segments below it are all on the lowest level.
    pub min_merge_size: i64,
    /// `maxMergeSize`: a segment at or above it is never merged naturally.
    pub max_merge_size: i64,
    /// `maxMergeSizeForForcedMerge`.
    pub max_merge_size_for_forced_merge: i64,
    /// `maxMergeDocs`.
    pub max_merge_docs: i32,
    /// `calibrateSizeByDeletes`.
    pub calibrate_size_by_deletes: bool,
    target_search_concurrency: i32,
    compound: CompoundFileSettings,
}

impl LogMergePolicy {
    /// `LogMergePolicy.LEVEL_LOG_SPAN`.
    pub const LEVEL_LOG_SPAN: f64 = 0.75;
    /// `DEFAULT_MERGE_FACTOR`.
    pub const DEFAULT_MERGE_FACTOR: i32 = 10;
    /// `DEFAULT_MAX_MERGE_DOCS`.
    pub const DEFAULT_MAX_MERGE_DOCS: i32 = i32::MAX;
    /// `LogMergePolicy.DEFAULT_NO_CFS_RATIO`.
    pub const DEFAULT_NO_CFS_RATIO: f64 = 0.1;

    fn with_unit(unit: LogSizeUnit, min: i64, max: i64, max_forced: i64) -> Self {
        LogMergePolicy {
            unit,
            merge_factor: Self::DEFAULT_MERGE_FACTOR,
            min_merge_size: min,
            max_merge_size: max,
            max_merge_size_for_forced_merge: max_forced,
            max_merge_docs: Self::DEFAULT_MAX_MERGE_DOCS,
            calibrate_size_by_deletes: true,
            target_search_concurrency: 1,
            compound: CompoundFileSettings::new(
                Self::DEFAULT_NO_CFS_RATIO,
                CompoundFileSettings::DEFAULT_MAX_CFS_SEGMENT_SIZE,
            ),
        }
    }

    /// Which size this policy measures.
    pub fn unit(&self) -> LogSizeUnit {
        self.unit
    }

    /// `getMergeFactor`.
    pub fn merge_factor(&self) -> i32 {
        self.merge_factor
    }

    /// `setMergeFactor`.
    pub fn set_merge_factor(&mut self, merge_factor: i32) -> Result<()> {
        if merge_factor < 2 {
            return Err(Error::IllegalArgument(
                "mergeFactor cannot be less than 2".into(),
            ));
        }
        self.merge_factor = merge_factor;
        Ok(())
    }

    /// `getTargetSearchConcurrency`.
    pub fn target_search_concurrency(&self) -> i32 {
        self.target_search_concurrency
    }

    /// `setTargetSearchConcurrency`.
    pub fn set_target_search_concurrency(&mut self, target: i32) -> Result<()> {
        if target < 1 {
            return Err(Error::IllegalArgument(format!(
                "targetSearchConcurrency must be >= 1 (got {target})"
            )));
        }
        self.target_search_concurrency = target;
        Ok(())
    }

    /// `sizeDocs`.
    pub fn size_docs(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        if self.calibrate_size_by_deletes {
            let del_count = ctx.num_deletes_to_merge(info)?;
            Ok(i64::from(info.max_doc).wrapping_sub(i64::from(del_count)))
        } else {
            Ok(i64::from(info.max_doc))
        }
    }

    /// `sizeBytes`.
    pub fn size_bytes(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        if self.calibrate_size_by_deletes {
            return default_size(info, ctx);
        }
        Ok(info.size_in_bytes)
    }

    /// `isMerged(infos, maxNumSegments, segmentsToMerge, ctx)`.
    fn is_merged_all(
        &self,
        infos: &[MergeSegment],
        max_num_segments: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        let mut num_to_merge: i32 = 0;
        let mut merge_info: Option<&MergeSegment> = None;
        let mut segment_is_original = false;
        for info in infos {
            if num_to_merge > max_num_segments {
                break;
            }
            if let Some(is_original) = segments_to_merge.get(&info.name) {
                segment_is_original = *is_original;
                num_to_merge = num_to_merge.saturating_add(1);
                merge_info = Some(info);
            }
        }
        if num_to_merge > max_num_segments {
            return Ok(false);
        }
        if num_to_merge != 1 || !segment_is_original {
            return Ok(true);
        }
        match merge_info {
            Some(info) => self.is_merged(infos, info, ctx),
            None => Ok(true),
        }
    }

    fn too_large(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<bool> {
        Ok(self.size(info, ctx)? > self.max_merge_size_for_forced_merge
            || self.size_docs(info, ctx)? > i64::from(self.max_merge_docs))
    }

    /// `findForcedMergesSizeLimit`.
    fn find_forced_merges_size_limit(
        &self,
        infos: &[MergeSegment],
        mut last: usize,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let mut spec = MergeSpecification::new();
        let mf = usize::try_from(self.merge_factor).unwrap_or(usize::MAX);
        // `int start = last - 1; while (start >= 0)`, with `start` one higher
        // so it stays unsigned: `start1 == start + 1`.
        let mut start1 = last;
        while start1 > 0 {
            let start = start1.saturating_sub(1);
            let info = &infos[start];
            if self.too_large(info, ctx)? {
                // `last - start - 1 > 1 || (start != last - 1 && !isMerged(infos[start + 1]))`
                let span = last.saturating_sub(start).saturating_sub(1);
                if span > 1
                    || (start != last.saturating_sub(1)
                        && !self.is_merged(infos, &infos[start1], ctx)?)
                {
                    spec.add(OneMerge::new(infos[start1..last].to_vec()));
                }
                last = start;
            } else if last.saturating_sub(start) == mf {
                spec.add(OneMerge::new(infos[start..last].to_vec()));
                last = start;
            }
            start1 = start;
        }
        // After the loop Java's `start == -1`; `++start` makes it 0.
        if last > 0 && (1 < last || !self.is_merged(infos, &infos[0], ctx)?) {
            spec.add(OneMerge::new(infos[0..last].to_vec()));
        }
        Ok((!spec.merges.is_empty()).then_some(spec))
    }

    /// `findForcedMergesMaxNumSegments`.
    fn find_forced_merges_max_num_segments(
        &self,
        infos: &[MergeSegment],
        max_num_segments: i32,
        last: usize,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let mut spec = MergeSpecification::new();
        let mf = i64::from(self.merge_factor);
        let max = i64::from(max_num_segments);
        let mut last_i = i64::try_from(last).unwrap_or(i64::MAX);

        // First, enroll all "full" merges (size mergeFactor) to potentially be
        // run concurrently.
        while last_i.saturating_sub(max).saturating_add(1) >= mf {
            let end = usize::try_from(last_i).unwrap_or(0);
            let begin = usize::try_from(last_i.saturating_sub(mf)).unwrap_or(0);
            spec.add(OneMerge::new(infos[begin..end].to_vec()));
            last_i = last_i.saturating_sub(mf);
        }

        // Only if there are no full merges pending do we add a final partial
        // (< mergeFactor segments) merge.
        if spec.merges.is_empty() {
            let last = usize::try_from(last_i).unwrap_or(0);
            if max_num_segments == 1 {
                // Since we must merge down to 1 segment, the choice is simple.
                if last > 1 || !self.is_merged(infos, &infos[0], ctx)? {
                    spec.add(OneMerge::new(infos[0..last].to_vec()));
                }
            } else if last_i > max {
                // Take care to pick a partial merge that is least cost, but
                // does not make the index too lopsided.
                let final_merge_size =
                    usize::try_from(last_i.saturating_sub(max).saturating_add(1)).unwrap_or(0);
                let mut best_size: i64 = 0;
                let mut best_start: usize = 0;
                let candidates = last.saturating_sub(final_merge_size).saturating_add(1);
                for i in 0..candidates {
                    let mut sum_size: i64 = 0;
                    for info in &infos[i..i.saturating_add(final_merge_size)] {
                        sum_size = sum_size.wrapping_add(self.size(info, ctx)?);
                    }
                    if i == 0
                        || (sum_size < self.size(&infos[i.saturating_sub(1)], ctx)?.wrapping_mul(2)
                            && sum_size < best_size)
                    {
                        best_start = i;
                        best_size = sum_size;
                    }
                }
                spec.add(OneMerge::new(
                    infos[best_start..best_start.saturating_add(final_merge_size)].to_vec(),
                ));
            }
        }
        Ok((!spec.merges.is_empty()).then_some(spec))
    }
}

impl MergePolicy for LogMergePolicy {
    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        match self.unit {
            LogSizeUnit::Bytes => self.size_bytes(info, ctx),
            LogSizeUnit::Docs => self.size_docs(info, ctx),
        }
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        self.min_merge_size
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.compound
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        &mut self.compound
    }

    /// `findForcedMerges`: merges down to `max_segment_count` segments,
    /// honouring `maxMergeSizeForForcedMerge` and `maxMergeDocs`.
    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        if max_segment_count < 1 {
            return Err(Error::IllegalArgument(format!(
                "maxNumSegments must be > 0 (got {max_segment_count})"
            )));
        }
        if self.is_merged_all(infos, max_segment_count, segments_to_merge, ctx)? {
            return Ok(None);
        }

        // Find the newest (rightmost) segment that needs to be merged (other
        // segments may have been flushed since merging started).
        let mut last = infos.len();
        while last > 0 {
            last = last.saturating_sub(1);
            if segments_to_merge.contains_key(&infos[last].name) {
                last = last.saturating_add(1);
                break;
            }
        }
        if last == 0 {
            return Ok(None);
        }

        // There is only one segment already, and it is merged.
        if max_segment_count == 1 && last == 1 && self.is_merged(infos, &infos[0], ctx)? {
            return Ok(None);
        }

        // Check if there are any segments above the threshold.
        let mut any_too_large = false;
        for info in &infos[..last] {
            if self.too_large(info, ctx)? {
                any_too_large = true;
                break;
            }
        }
        if any_too_large {
            self.find_forced_merges_size_limit(infos, last, ctx)
        } else {
            self.find_forced_merges_max_num_segments(infos, max_segment_count, last, ctx)
        }
    }

    /// `findForcedDeletesMerges`: every run of adjacent segments with
    /// deletions, `mergeFactor` at a time. Returns an empty specification,
    /// not `None`, when nothing has deletions -- as Java does.
    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let mut spec = MergeSpecification::new();
        let mf = usize::try_from(self.merge_factor).unwrap_or(usize::MAX);
        let mut first_with_deletions: Option<usize> = None;
        for (i, info) in infos.iter().enumerate() {
            let del_count = ctx.num_deletes_to_merge(info)?;
            if del_count > 0 {
                match first_with_deletions {
                    None => first_with_deletions = Some(i),
                    Some(first) if i.saturating_sub(first) == mf => {
                        // We've seen mergeFactor segments in a row with
                        // deletions, so force a merge now.
                        spec.add(OneMerge::new(infos[first..i].to_vec()));
                        first_with_deletions = Some(i);
                    }
                    Some(_) => {}
                }
            } else if let Some(first) = first_with_deletions.take() {
                // End of a sequence of segments with deletions, so merge
                // those past segments even if it's fewer than mergeFactor.
                spec.add(OneMerge::new(infos[first..i].to_vec()));
            }
        }
        if let Some(first) = first_with_deletions {
            spec.add(OneMerge::new(infos[first..].to_vec()));
        }
        Ok(Some(spec))
    }

    /// `findMerges`: the level walk.
    fn find_merges(
        &self,
        _trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let num_segments = infos.len();
        // Compute levels, which is just log (base mergeFactor) of the size of
        // each segment.
        let mut levels: Vec<f32> = Vec::with_capacity(num_segments);
        let norm = (f64::from(self.merge_factor)).ln() as f32;
        let merging = ctx.merging_segments();

        let mut total_doc_count: i32 = 0;
        for info in infos {
            // `totalDocCount += sizeDocs(...)`: an `int += long` compound
            // assignment, i.e. `(int) (totalDocCount + sizeDocs)`.
            total_doc_count =
                i64::from(total_doc_count).wrapping_add(self.size_docs(info, ctx)?) as i32;
            let mut size = self.size(info, ctx)?;
            // Floor tiny segments.
            if size < 1 {
                size = 1;
            }
            // `(float) Math.log((double) size) / norm`: a float division.
            levels.push(((size as f64).ln() as f32) / norm);
        }

        let level_floor: f32 = if self.min_merge_size <= 0 {
            0.0
        } else {
            // `(float) (Math.log((double) minMergeSize) / norm)`: a double
            // division, then the cast.
            ((self.min_merge_size as f64).ln() / f64::from(norm)) as f32
        };

        // Now, we quantize the log values into levels. The first level is any
        // segment whose log size is within LEVEL_LOG_SPAN of the max size, or,
        // who has such as segment "to the right". Then, we find the max of all
        // other segments and use that to define the next level segment, etc.
        let mut spec: Option<MergeSpecification> = None;
        let n = levels.len();

        let mut max_levels: Vec<f32> = vec![0.0; n.saturating_add(1)];
        max_levels[n] = -1.0;
        for i in (0..n).rev() {
            max_levels[i] = levels[i].max(max_levels[i.saturating_add(1)]);
        }

        // `Math.ceilDiv(totalDocCount, targetSearchConcurrency)`.
        let max_merge_docs = self
            .max_merge_docs
            .min(ceil_div(total_doc_count, self.target_search_concurrency));
        let mf = usize::try_from(self.merge_factor).unwrap_or(usize::MAX);

        let mut start: usize = 0;
        while start < n {
            let max_level = max_levels[start];
            // Now search backwards for the rightmost segment that falls into
            // this level.
            let level_bottom: f32 = if max_level > level_floor {
                // With a merge factor of 10, this means that the biggest
                // segment and the smallest segment that take part of a merge
                // have a size difference of at most 5.6x.
                (f64::from(max_level) - Self::LEVEL_LOG_SPAN) as f32
            } else {
                // For segments below the floor size, we allow more unbalanced
                // merges, but still somewhat balanced to avoid running into
                // O(n^2) merging.
                (f64::from(max_level) - 2.0 * Self::LEVEL_LOG_SPAN) as f32
            };

            // `int upto = numMergeableSegments - 1; while (upto >= start)`,
            // kept one higher (`upto1 == upto + 1`) so it stays unsigned.
            let mut upto1 = n;
            while upto1 > start {
                if levels[upto1.saturating_sub(1)] >= level_bottom {
                    break;
                }
                upto1 = upto1.saturating_sub(1);
            }

            // Finally, record all merges that are viable at this level.
            let mut end = start.saturating_add(mf);
            while end <= upto1 {
                let mut any_merging = false;
                let mut merge_size: i64 = 0;
                let mut merge_docs: i64 = 0;
                let scan_end = end;
                for (i, info) in infos.iter().enumerate().take(scan_end).skip(start) {
                    if merging.contains(&info.name) {
                        any_merging = true;
                        break;
                    }
                    let segment_size = self.size(info, ctx)?;
                    let segment_docs = self.size_docs(info, ctx)?;
                    if merge_size.wrapping_add(segment_size) > self.max_merge_size
                        || merge_docs.wrapping_add(segment_docs) > i64::from(max_merge_docs)
                    {
                        // This merge is full, stop adding more segments to it.
                        if i == start {
                            // This segment alone is too large, return a
                            // singleton merge.
                            end = i.saturating_add(1);
                        } else {
                            // This segment would make the merge too large,
                            // exclude it.
                            end = i;
                        }
                        break;
                    }
                    merge_size = merge_size.wrapping_add(segment_size);
                    merge_docs = merge_docs.wrapping_add(segment_docs);
                }

                if end.saturating_sub(start) >= mf
                    && self.min_merge_size < self.max_merge_size
                    && merge_size < self.min_merge_size
                    && !any_merging
                {
                    // If the merge has mergeFactor segments but is still
                    // smaller than minMergeSize, keep packing candidate
                    // segments.
                    while end < upto1 {
                        let info = &infos[end];
                        if merging.contains(&info.name) {
                            any_merging = true;
                            break;
                        }
                        let segment_size = self.size(info, ctx)?;
                        let segment_docs = self.size_docs(info, ctx)?;
                        if merge_size.wrapping_add(segment_size) > self.min_merge_size
                            || merge_docs.wrapping_add(segment_docs) > i64::from(max_merge_docs)
                        {
                            break;
                        }
                        merge_size = merge_size.wrapping_add(segment_size);
                        merge_docs = merge_docs.wrapping_add(segment_docs);
                        end = end.saturating_add(1);
                    }
                }

                if !any_merging && end.saturating_sub(start) > 1 {
                    spec.get_or_insert_with(MergeSpecification::new)
                        .add(OneMerge::new(infos[start..end].to_vec()));
                }

                start = end;
                end = start.saturating_add(mf);
            }

            start = upto1;
        }
        Ok(spec)
    }
}

/// `Math.ceilDiv(int, int)` for a positive divisor.
fn ceil_div(x: i32, y: i32) -> i32 {
    let y = y.max(1);
    let q = div_positive_i32(x, y);
    // Java rounds toward positive infinity: add one when the signs agree and
    // the division was inexact.
    if (x ^ y) >= 0 && q.wrapping_mul(y) != x {
        q.wrapping_add(1)
    } else {
        q
    }
}

/// `LogByteSizeMergePolicy`: a [`LogMergePolicy`] measuring segments in bytes.
#[derive(Debug, Clone)]
pub struct LogByteSizeMergePolicy(LogMergePolicy);

impl LogByteSizeMergePolicy {
    /// `DEFAULT_MIN_MERGE_MB`.
    pub const DEFAULT_MIN_MERGE_MB: f64 = 16.0;
    /// `DEFAULT_MAX_MERGE_MB`.
    pub const DEFAULT_MAX_MERGE_MB: f64 = 2048.0;
    /// `DEFAULT_MAX_MERGE_MB_FOR_FORCED_MERGE` (`Long.MAX_VALUE` MB).
    pub const DEFAULT_MAX_MERGE_MB_FOR_FORCED_MERGE: f64 = i64::MAX as f64;

    pub fn new() -> Self {
        LogByteSizeMergePolicy(LogMergePolicy::with_unit(
            LogSizeUnit::Bytes,
            mb_to_bytes(Self::DEFAULT_MIN_MERGE_MB),
            mb_to_bytes(Self::DEFAULT_MAX_MERGE_MB),
            mb_to_bytes(Self::DEFAULT_MAX_MERGE_MB_FOR_FORCED_MERGE),
        ))
    }

    /// `setMaxMergeMB`.
    pub fn set_max_merge_mb(&mut self, mb: f64) {
        self.0.max_merge_size = mb_to_bytes(mb);
    }

    /// `getMaxMergeMB`.
    pub fn max_merge_mb(&self) -> f64 {
        self.0.max_merge_size as f64 / 1024. / 1024.
    }

    /// `setMaxMergeMBForForcedMerge`.
    pub fn set_max_merge_mb_for_forced_merge(&mut self, mb: f64) {
        self.0.max_merge_size_for_forced_merge = mb_to_bytes(mb);
    }

    /// `getMaxMergeMBForForcedMerge`.
    pub fn max_merge_mb_for_forced_merge(&self) -> f64 {
        self.0.max_merge_size_for_forced_merge as f64 / 1024. / 1024.
    }

    /// `setMinMergeMB`.
    pub fn set_min_merge_mb(&mut self, mb: f64) {
        self.0.min_merge_size = mb_to_bytes(mb);
    }

    /// `getMinMergeMB`.
    pub fn min_merge_mb(&self) -> f64 {
        self.0.min_merge_size as f64 / 1024. / 1024.
    }
}

impl Default for LogByteSizeMergePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// `(long) (mb * 1024 * 1024)`; Rust's `as` saturates where Java's
/// narrowing does too.
fn mb_to_bytes(mb: f64) -> i64 {
    (mb * 1024.0 * 1024.0) as i64
}

/// `LogDocMergePolicy`: a [`LogMergePolicy`] measuring segments in documents.
#[derive(Debug, Clone)]
pub struct LogDocMergePolicy(LogMergePolicy);

impl LogDocMergePolicy {
    /// `DEFAULT_MIN_MERGE_DOCS`.
    pub const DEFAULT_MIN_MERGE_DOCS: i32 = 1000;

    pub fn new() -> Self {
        LogDocMergePolicy(LogMergePolicy::with_unit(
            LogSizeUnit::Docs,
            i64::from(Self::DEFAULT_MIN_MERGE_DOCS),
            i64::MAX,
            i64::MAX,
        ))
    }

    /// `setMinMergeDocs`.
    pub fn set_min_merge_docs(&mut self, min_merge_docs: i32) {
        self.0.min_merge_size = i64::from(min_merge_docs);
    }

    /// `getMinMergeDocs`.
    pub fn min_merge_docs(&self) -> i32 {
        self.0.min_merge_size as i32
    }
}

impl Default for LogDocMergePolicy {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! delegate_log {
    ($t:ty) => {
        impl Deref for $t {
            type Target = LogMergePolicy;
            fn deref(&self) -> &LogMergePolicy {
                &self.0
            }
        }
        impl DerefMut for $t {
            fn deref_mut(&mut self) -> &mut LogMergePolicy {
                &mut self.0
            }
        }
        impl MergePolicy for $t {
            fn find_merges(
                &self,
                trigger: MergeTrigger,
                infos: &[MergeSegment],
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                self.0.find_merges(trigger, infos, ctx)
            }
            fn find_forced_merges(
                &self,
                infos: &[MergeSegment],
                max_segment_count: i32,
                segments_to_merge: &HashMap<String, bool>,
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                self.0
                    .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
            }
            fn find_forced_deletes_merges(
                &self,
                infos: &[MergeSegment],
                ctx: &dyn MergeContext,
            ) -> Result<Option<MergeSpecification>> {
                self.0.find_forced_deletes_merges(infos, ctx)
            }
            fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
                self.0.size(info, ctx)
            }
            fn max_full_flush_merge_size(&self) -> i64 {
                self.0.max_full_flush_merge_size()
            }
            fn compound_file_settings(&self) -> CompoundFileSettings {
                self.0.compound_file_settings()
            }
            fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
                self.0.compound_file_settings_mut()
            }
        }
    };
}

delegate_log!(LogByteSizeMergePolicy);
delegate_log!(LogDocMergePolicy);

/// `x / y` for a `y` that is positive by construction.
// ARITH: the only caller clamps `y` to at least 1, so `/` neither divides by
// zero nor overflows (`i32::MIN / -1` needs a negative divisor).
#[allow(clippy::arithmetic_side_effects)]
fn div_positive_i32(x: i32, y: i32) -> i32 {
    x / y.max(1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::super::api::BasicMergeContext;
    use super::*;

    fn segs(sizes: &[(i32, i32, i64)]) -> Vec<MergeSegment> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &(docs, del, bytes))| MergeSegment::new(format!("_{i}"), docs, del, bytes))
            .collect()
    }

    fn groups(spec: Option<MergeSpecification>) -> Vec<Vec<String>> {
        spec.map(|s| s.groups()).unwrap_or_default()
    }

    #[test]
    fn defaults_match_java() {
        let p = LogByteSizeMergePolicy::new();
        assert_eq!(p.merge_factor(), 10);
        assert_eq!(p.min_merge_mb(), 16.0);
        assert_eq!(p.max_merge_mb(), 2048.0);
        assert_eq!(p.max_merge_size_for_forced_merge, i64::MAX);
        assert_eq!(p.max_full_flush_merge_size(), 16 * 1024 * 1024);
        assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.1);
        let d = LogDocMergePolicy::new();
        assert_eq!(d.min_merge_docs(), 1000);
        assert_eq!(d.unit(), LogSizeUnit::Docs);
        assert_eq!(d.max_merge_size, i64::MAX);
    }

    #[test]
    fn setters_validate() {
        let mut p = LogDocMergePolicy::new();
        assert!(p.set_merge_factor(1).is_err());
        p.set_merge_factor(3).unwrap();
        assert_eq!(p.merge_factor(), 3);
        assert!(p.set_target_search_concurrency(0).is_err());
        p.set_target_search_concurrency(4).unwrap();
        assert_eq!(p.target_search_concurrency(), 4);
        p.set_min_merge_docs(7);
        assert_eq!(p.min_merge_docs(), 7);
        let mut b = LogByteSizeMergePolicy::default();
        b.set_max_merge_mb(1.0);
        b.set_min_merge_mb(0.5);
        b.set_max_merge_mb_for_forced_merge(2.0);
        assert_eq!(b.max_merge_mb(), 1.0);
        assert_eq!(b.min_merge_mb(), 0.5);
        assert_eq!(b.max_merge_mb_for_forced_merge(), 2.0);
    }

    #[test]
    fn natural_merges_take_merge_factor_equal_segments() {
        let ctx = BasicMergeContext::default();
        let mut p = LogDocMergePolicy::new();
        p.set_merge_factor(3).unwrap();
        p.set_min_merge_docs(1);
        let infos = segs(&[(10, 0, 1), (10, 0, 1), (10, 0, 1), (10, 0, 1)]);
        let got = groups(p.find_merges(MergeTrigger::Explicit, &infos, &ctx).unwrap());
        assert_eq!(got, vec![vec!["_0", "_1", "_2"]]);
        // A merging segment blocks its window.
        let ctx = BasicMergeContext::new(["_1".to_string()].into_iter().collect());
        assert!(p
            .find_merges(MergeTrigger::Explicit, &infos, &ctx)
            .unwrap()
            .is_none());
    }

    #[test]
    fn forced_deletes_merges_runs_of_deleted_segments() {
        let ctx = BasicMergeContext::default();
        let mut p = LogDocMergePolicy::new();
        p.set_merge_factor(2).unwrap();
        let infos = segs(&[(10, 1, 1), (10, 1, 1), (10, 1, 1), (10, 0, 1), (10, 2, 1)]);
        let got = groups(p.find_forced_deletes_merges(&infos, &ctx).unwrap());
        assert_eq!(got, vec![vec!["_0", "_1"], vec!["_2"], vec!["_4"]]);
        let clean = segs(&[(10, 0, 1)]);
        let spec = p.find_forced_deletes_merges(&clean, &ctx).unwrap().unwrap();
        assert!(spec.merges.is_empty());
    }

    #[test]
    fn forced_merges_validate_and_skip_merged() {
        let ctx = BasicMergeContext::default();
        let p = LogDocMergePolicy::new();
        let infos = segs(&[(10, 0, 1)]);
        let all: HashMap<String, bool> = [("_0".to_string(), true)].into_iter().collect();
        assert!(p.find_forced_merges(&infos, 0, &all, &ctx).is_err());
        // One segment without deletes, not compound (noCFSRatio 0.1 of a
        // 1-segment index means the merged segment would be compound... the
        // size equals the total, so useCompoundFile is false): merged.
        assert!(p
            .find_forced_merges(&infos, 1, &all, &ctx)
            .unwrap()
            .is_none());
        // No segment to merge at all.
        assert!(p
            .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
            .unwrap()
            .is_none());
    }

    #[test]
    fn ceil_div_matches_java() {
        assert_eq!(ceil_div(7, 2), 4);
        assert_eq!(ceil_div(8, 2), 4);
        assert_eq!(ceil_div(-7, 2), -3);
        assert_eq!(ceil_div(0, 3), 0);
    }
}
