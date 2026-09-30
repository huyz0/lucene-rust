//! The pluggable half of `org.apache.lucene.index.MergePolicy`: the
//! [`MergePolicy`] trait every policy implements, the [`MergeContext`] a
//! writer hands it, and the [`OneMerge`]/[`MergeSpecification`] it answers
//! with -- plus [`TieredMergePolicy`] and [`NoMergePolicy`] behind that trait.
//!
//! # Shape
//!
//! Java passes a live `SegmentInfos` and `MergeContext`; a policy reads four
//! things off each `SegmentCommitInfo` -- its name, `maxDoc`, `sizeInBytes()`
//! and whether it is a compound file -- and asks the context for
//! `numDeletesToMerge`/`getMergingSegments`. [`MergeSegment`] is exactly those
//! four facts plus the hard delete count, so a policy is a pure function of
//! data a caller can build without a writer (which is how the differential
//! test drives Java and Rust with identical inputs).
//!
//! Java's `null` "no merges" is `None`; an empty specification is
//! `Some(spec)` with no merges, exactly as Java distinguishes the two
//! (`LogMergePolicy.findForcedDeletesMerges` returns an empty specification,
//! the others `null`).
//!
//! Sizes are `i64`, Java's `long`, so every size comparison and the
//! `(long) (mb * 1024 * 1024)` conversions keep Java's semantics.
//!
//! # Rust-only differences
//!
//! - `infoStream` verbose messages are not produced (no `InfoStream` here).
//! - `findMerges(CodecReader...)` (the `addIndexes` entry point) lives on
//!   [`OneMerge::from_readers`] (reader-backed merges carry no segments).
//! - The compound-file knobs (`noCFSRatio`, `maxCFSSegmentSize`) are a
//!   [`CompoundFileSettings`] every policy exposes through the trait, so a
//!   [`super::filter::FilterMergePolicy`] can delegate them like Java does.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::{
    find_forced_delete_merges_excluding, find_merges_excluding, forced_merges_full,
    MergePolicyConfig, SegmentStat, UNLIMITED_SEGMENT_COUNT,
};

/// A merge-policy failure: Java's `IllegalArgumentException` from a setter or
/// an argument check, or an I/O failure from a context that had to read a
/// segment to answer (`numDeletesToMerge` under soft deletes).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("illegal argument: {0}")]
    IllegalArgument(String),
    #[error("merge context failed: {0}")]
    Context(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `MergeTrigger`: what asked the policy for merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergeTrigger {
    SegmentFlush,
    FullFlush,
    Explicit,
    MergeFinished,
    Closing,
    Commit,
    GetReader,
    AddIndexes,
}

/// One segment as a merge policy sees it: the `SegmentCommitInfo` facts every
/// policy in Lucene reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeSegment {
    /// `SegmentInfo.name`.
    pub name: String,
    /// `SegmentInfo.maxDoc()`.
    pub max_doc: i32,
    /// `SegmentCommitInfo.getDelCount()` -- hard deletes.
    pub del_count: i32,
    /// `SegmentCommitInfo.sizeInBytes()` -- raw, not pro-rated.
    pub size_in_bytes: i64,
    /// `SegmentInfo.getUseCompoundFile()`.
    pub use_compound_file: bool,
}

impl MergeSegment {
    pub fn new(name: impl Into<String>, max_doc: i32, del_count: i32, size_in_bytes: i64) -> Self {
        MergeSegment {
            name: name.into(),
            max_doc,
            del_count,
            size_in_bytes,
            use_compound_file: false,
        }
    }

    /// Sets [`Self::use_compound_file`].
    pub fn with_compound_file(mut self, use_compound_file: bool) -> Self {
        self.use_compound_file = use_compound_file;
        self
    }
}

impl From<&SegmentStat> for MergeSegment {
    fn from(stat: &SegmentStat) -> Self {
        MergeSegment::new(
            stat.name.clone(),
            stat.doc_count,
            stat.del_count,
            i64::try_from(stat.size_bytes).unwrap_or(i64::MAX),
        )
    }
}

/// `MergePolicy.MergeContext`: what a policy may ask the writer.
pub trait MergeContext {
    /// `numDeletesToMerge(info)`: the deletes a merge of `info` would reclaim
    /// (hard deletes, plus soft deletes a retention policy no longer keeps).
    fn num_deletes_to_merge(&self, info: &MergeSegment) -> Result<i32>;
    /// `numDeletedDocs(info)`: every deleted document, hard and soft.
    fn num_deleted_docs(&self, info: &MergeSegment) -> i32;
    /// `getMergingSegments()`, by segment name.
    fn merging_segments(&self) -> &HashSet<String>;
}

/// The context a writer with no soft deletes provides: `numDeletesToMerge ==
/// numDeletedDocs == getDelCount()`, and an explicit merging set.
#[derive(Debug, Clone, Default)]
pub struct BasicMergeContext {
    pub merging: HashSet<String>,
}

impl BasicMergeContext {
    pub fn new(merging: HashSet<String>) -> Self {
        BasicMergeContext { merging }
    }
}

impl MergeContext for BasicMergeContext {
    fn num_deletes_to_merge(&self, info: &MergeSegment) -> Result<i32> {
        Ok(info.del_count)
    }
    fn num_deleted_docs(&self, info: &MergeSegment) -> i32 {
        info.del_count
    }
    fn merging_segments(&self) -> &HashSet<String> {
        &self.merging
    }
}

/// `MergePolicy.OneMerge`: the segments one merge combines.
#[derive(Debug, Clone, Default)]
pub struct OneMerge {
    /// `OneMerge.segments`, in the order the policy chose.
    pub segments: Vec<MergeSegment>,
    /// `OneMerge.totalMaxDoc`.
    pub total_max_doc: i64,
    /// For `OneMerge(CodecReader...)` (`addIndexes`): how many readers the
    /// merge combines. `0` for a segment merge.
    pub reader_count: usize,
}

impl OneMerge {
    /// `new OneMerge(List<SegmentCommitInfo>)`.
    pub fn new(segments: Vec<MergeSegment>) -> Self {
        let total_max_doc = segments
            .iter()
            .fold(0i64, |acc, s| acc.saturating_add(i64::from(s.max_doc)));
        OneMerge {
            segments,
            total_max_doc,
            reader_count: 0,
        }
    }

    /// `new OneMerge(CodecReader...)`: a merge of `addIndexes` readers,
    /// holding `total_max_doc` documents between them.
    pub fn from_readers(reader_count: usize, total_max_doc: i64) -> Self {
        OneMerge {
            segments: Vec::new(),
            total_max_doc,
            reader_count,
        }
    }

    /// The merged segments' names, in merge order.
    pub fn segment_names(&self) -> Vec<String> {
        self.segments.iter().map(|s| s.name.clone()).collect()
    }
}

/// `MergePolicy.MergeSpecification`.
#[derive(Debug, Clone, Default)]
pub struct MergeSpecification {
    pub merges: Vec<OneMerge>,
}

impl MergeSpecification {
    pub fn new() -> Self {
        Self::default()
    }

    /// `MergeSpecification.add`.
    pub fn add(&mut self, merge: OneMerge) {
        self.merges.push(merge);
    }

    /// Every merge as its segment names -- what the differential tests compare.
    pub fn groups(&self) -> Vec<Vec<String>> {
        self.merges.iter().map(OneMerge::segment_names).collect()
    }
}

/// `noCFSRatio` and `maxCFSSegmentSize`, the compound-file knobs every
/// `MergePolicy` carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundFileSettings {
    /// `noCFSRatio`, `0.0..=1.0`.
    pub no_cfs_ratio: f64,
    /// `maxCFSSegmentSize`, bytes.
    pub max_cfs_segment_size: i64,
}

impl CompoundFileSettings {
    /// `MergePolicy.DEFAULT_NO_CFS_RATIO`.
    pub const DEFAULT_NO_CFS_RATIO: f64 = 1.0;
    /// `MergePolicy.DEFAULT_MAX_CFS_SEGMENT_SIZE`.
    pub const DEFAULT_MAX_CFS_SEGMENT_SIZE: i64 = i64::MAX;

    pub fn new(no_cfs_ratio: f64, max_cfs_segment_size: i64) -> Self {
        CompoundFileSettings {
            no_cfs_ratio,
            max_cfs_segment_size,
        }
    }

    /// `setNoCFSRatio`.
    pub fn set_no_cfs_ratio(&mut self, no_cfs_ratio: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&no_cfs_ratio) {
            return Err(Error::IllegalArgument(format!(
                "noCFSRatio must be 0.0 to 1.0 inclusive; got {no_cfs_ratio}"
            )));
        }
        self.no_cfs_ratio = no_cfs_ratio;
        Ok(())
    }

    /// `getMaxCFSSegmentSizeMB`.
    pub fn max_cfs_segment_size_mb(&self) -> f64 {
        self.max_cfs_segment_size as f64 / 1024. / 1024.
    }

    /// `setMaxCFSSegmentSizeMB`.
    pub fn set_max_cfs_segment_size_mb(&mut self, mb: f64) -> Result<()> {
        if mb < 0.0 || mb.is_nan() {
            return Err(Error::IllegalArgument(format!(
                "maxCFSSegmentSizeMB must be >=0 (got {mb})"
            )));
        }
        let v = mb * 1024.0 * 1024.0;
        // `v > Long.MAX_VALUE ? Long.MAX_VALUE : (long) v`; Rust's `as` already
        // saturates, which is the same answer.
        self.max_cfs_segment_size = v as i64;
        Ok(())
    }
}

impl Default for CompoundFileSettings {
    fn default() -> Self {
        CompoundFileSettings::new(
            Self::DEFAULT_NO_CFS_RATIO,
            Self::DEFAULT_MAX_CFS_SEGMENT_SIZE,
        )
    }
}

/// `MergePolicy`'s default `size(info, ctx)`: `sizeInBytes()` pro-rated by
/// the fraction of documents a merge would keep.
pub fn default_size(info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
    let byte_size = info.size_in_bytes;
    let del_count = ctx.num_deletes_to_merge(info)?;
    if info.max_doc <= 0 {
        return Ok(byte_size);
    }
    let del_ratio = f64::from(del_count) / f64::from(info.max_doc);
    Ok((byte_size as f64 * (1.0 - del_ratio)) as i64)
}

/// `org.apache.lucene.index.MergePolicy`: decides which segments merge.
///
/// The three `find*` methods are the abstract ones; everything else carries
/// Java's default behaviour and is overridden where Java overrides it.
pub trait MergePolicy: Send + Sync + fmt::Debug {
    /// `findMerges(MergeTrigger, SegmentInfos, MergeContext)`.
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>>;

    /// `findForcedMerges(SegmentInfos, maxSegmentCount, segmentsToMerge,
    /// MergeContext)`. `segments_to_merge` maps a segment name to Java's
    /// "is original" flag; a segment absent from it is not to be merged.
    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>>;

    /// `findForcedDeletesMerges(SegmentInfos, MergeContext)`.
    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>>;

    /// `findMerges(CodecReader...)`: `addIndexes` merges every reader into one.
    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        let total = max_docs.iter().fold(0i64, |a, d| a.saturating_add(*d));
        let mut spec = MergeSpecification::new();
        spec.add(OneMerge::from_readers(max_docs.len(), total));
        Some(spec)
    }

    /// `findFullFlushMerges`: the natural merges whose every input is below
    /// [`Self::max_full_flush_merge_size`] -- what merge-on-commit/refresh runs.
    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let Some(spec) = self.find_merges(trigger, infos, ctx)? else {
            return Ok(None);
        };
        let mut out: Option<MergeSpecification> = None;
        for merge in spec.merges {
            let mut below = true;
            for sci in &merge.segments {
                if self.size(sci, ctx)? >= self.max_full_flush_merge_size() {
                    below = false;
                    break;
                }
            }
            if below {
                out.get_or_insert_with(MergeSpecification::new).add(merge);
            }
        }
        Ok(out)
    }

    /// `useCompoundFile(infos, mergedInfo, ctx)`.
    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        let settings = self.compound_file_settings();
        if settings.no_cfs_ratio == 0.0 {
            return Ok(false);
        }
        let merged_size = self.size(merged, ctx)?;
        if merged_size > settings.max_cfs_segment_size {
            return Ok(false);
        }
        if settings.no_cfs_ratio >= 1.0 {
            return Ok(true);
        }
        let mut total: i64 = 0;
        for info in infos {
            // Java's `long +=`: wraps.
            total = total.wrapping_add(self.size(info, ctx)?);
        }
        Ok(merged_size as f64 <= settings.no_cfs_ratio * total as f64)
    }

    /// `size(info, ctx)`.
    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        default_size(info, ctx)
    }

    /// `maxFullFlushMergeSize()`.
    fn max_full_flush_merge_size(&self) -> i64 {
        0
    }

    /// `keepFullyDeletedSegment`: `false` unless a policy retains documents.
    fn keep_fully_deleted_segment(&self) -> bool {
        false
    }

    /// `numDeletesToMerge(info, delCount, readerSupplier)`.
    fn num_deletes_to_merge(&self, _info: &MergeSegment, del_count: i32) -> i32 {
        del_count
    }

    /// `noCFSRatio`/`maxCFSSegmentSize`.
    fn compound_file_settings(&self) -> CompoundFileSettings;

    /// `setNoCFSRatio`/`setMaxCFSSegmentSizeMB` land here.
    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings;

    /// `isMerged(infos, info, ctx)`: no deletes to reclaim and already in the
    /// compound-file state this policy would choose.
    fn is_merged(
        &self,
        infos: &[MergeSegment],
        info: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        let del_count = ctx.num_deletes_to_merge(info)?;
        Ok(del_count == 0 && self.use_compound_file(infos, info, ctx)? == info.use_compound_file)
    }
}

/// `NoMergePolicy.INSTANCE`: never merges.
#[derive(Debug, Clone, Default)]
pub struct NoMergePolicy {
    compound: CompoundFileSettings,
}

impl MergePolicy for NoMergePolicy {
    fn find_merges(
        &self,
        _trigger: MergeTrigger,
        _infos: &[MergeSegment],
        _ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(None)
    }
    fn find_forced_merges(
        &self,
        _infos: &[MergeSegment],
        _max_segment_count: i32,
        _segments_to_merge: &HashMap<String, bool>,
        _ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(None)
    }
    fn find_forced_deletes_merges(
        &self,
        _infos: &[MergeSegment],
        _ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(None)
    }
    fn find_full_flush_merges(
        &self,
        _trigger: MergeTrigger,
        _infos: &[MergeSegment],
        _ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(None)
    }
    fn use_compound_file(
        &self,
        _infos: &[MergeSegment],
        merged: &MergeSegment,
        _ctx: &dyn MergeContext,
    ) -> Result<bool> {
        Ok(merged.use_compound_file)
    }
    fn size(&self, _info: &MergeSegment, _ctx: &dyn MergeContext) -> Result<i64> {
        Ok(i64::MAX)
    }
    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.compound
    }
    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        &mut self.compound
    }
}

/// `TieredMergePolicy` behind the [`MergePolicy`] trait: the decision
/// functions of [`super`] (`find_merges_excluding`, the forced-merge bin
/// packing, `find_forced_delete_merges_excluding`) fed from a
/// [`MergeContext`], plus the parts only the trait has --
/// `findFullFlushMerges` (`maxFullFlushMergeSize() == floorSegmentBytes`) and
/// `useCompoundFile` (`DEFAULT_NO_CFS_RATIO == 0.1`).
#[derive(Debug, Clone)]
pub struct TieredMergePolicy {
    pub config: MergePolicyConfig,
    compound: CompoundFileSettings,
}

impl TieredMergePolicy {
    /// `TieredMergePolicy.DEFAULT_NO_CFS_RATIO`.
    pub const DEFAULT_NO_CFS_RATIO: f64 = 0.1;

    pub fn new(config: MergePolicyConfig) -> Self {
        TieredMergePolicy {
            config,
            compound: CompoundFileSettings::new(
                Self::DEFAULT_NO_CFS_RATIO,
                CompoundFileSettings::DEFAULT_MAX_CFS_SEGMENT_SIZE,
            ),
        }
    }

    /// The segment statistics the decision functions take, with `del_count`
    /// set to what `ctx.numDeletesToMerge` reports -- the figure Java's
    /// `SegmentSizeAndDocs` records.
    fn stats(infos: &[MergeSegment], ctx: &dyn MergeContext) -> Result<Vec<SegmentStat>> {
        infos
            .iter()
            .map(|info| {
                Ok(SegmentStat {
                    name: info.name.clone(),
                    doc_count: info.max_doc,
                    del_count: ctx.num_deletes_to_merge(info)?,
                    size_bytes: u64::try_from(info.size_in_bytes).unwrap_or(0),
                })
            })
            .collect()
    }

    fn to_spec(infos: &[MergeSegment], groups: Vec<Vec<String>>) -> Option<MergeSpecification> {
        if groups.is_empty() {
            return None;
        }
        let by_name: HashMap<&str, &MergeSegment> =
            infos.iter().map(|s| (s.name.as_str(), s)).collect();
        let mut spec = MergeSpecification::new();
        for group in groups {
            spec.add(OneMerge::new(
                group
                    .iter()
                    .filter_map(|n| by_name.get(n.as_str()).map(|s| (*s).clone()))
                    .collect(),
            ));
        }
        Some(spec)
    }
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        TieredMergePolicy::new(MergePolicyConfig::default())
    }
}

impl MergePolicy for TieredMergePolicy {
    fn find_merges(
        &self,
        _trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let stats = Self::stats(infos, ctx)?;
        let groups = find_merges_excluding(&stats, ctx.merging_segments(), &self.config);
        Ok(Self::to_spec(infos, groups))
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let stats = Self::stats(infos, ctx)?;
        let max = if max_segment_count == i32::MAX {
            UNLIMITED_SEGMENT_COUNT
        } else {
            usize::try_from(max_segment_count).unwrap_or(1)
        };
        // `isMerged(infos, infoZero, ctx)`, answered up front per segment.
        let mut merged: HashSet<String> = HashSet::new();
        for info in infos {
            if self.is_merged(infos, info, ctx)? {
                merged.insert(info.name.clone());
            }
        }
        let groups = forced_merges_full(
            &stats,
            max,
            segments_to_merge,
            ctx.merging_segments(),
            &merged,
            &self.config,
        );
        Ok(Self::to_spec(infos, groups))
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let stats = Self::stats(infos, ctx)?;
        let groups =
            find_forced_delete_merges_excluding(&stats, ctx.merging_segments(), &self.config);
        Ok(Self::to_spec(infos, groups))
    }

    fn max_full_flush_merge_size(&self) -> i64 {
        i64::try_from(self.config.floor_segment_size).unwrap_or(i64::MAX)
    }

    fn keep_fully_deleted_segment(&self) -> bool {
        self.config.keep_fully_deleted_segments
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.compound
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        &mut self.compound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(name: &str, max_doc: i32, del: i32, bytes: i64) -> MergeSegment {
        MergeSegment::new(name, max_doc, del, bytes)
    }

    #[test]
    fn default_size_prorates_by_deletes() {
        let ctx = BasicMergeContext::default();
        assert_eq!(default_size(&seg("_a", 100, 25, 1000), &ctx).unwrap(), 750);
        assert_eq!(default_size(&seg("_a", 0, 0, 1000), &ctx).unwrap(), 1000);
    }

    #[test]
    fn compound_settings_validate_like_java() {
        let mut s = CompoundFileSettings::default();
        assert!(s.set_no_cfs_ratio(1.5).is_err());
        assert!(s.set_no_cfs_ratio(-0.1).is_err());
        s.set_no_cfs_ratio(0.5).unwrap();
        assert_eq!(s.no_cfs_ratio, 0.5);
        assert!(s.set_max_cfs_segment_size_mb(-1.0).is_err());
        s.set_max_cfs_segment_size_mb(2.0).unwrap();
        assert_eq!(s.max_cfs_segment_size, 2 * 1024 * 1024);
        assert_eq!(s.max_cfs_segment_size_mb(), 2.0);
        s.set_max_cfs_segment_size_mb(f64::INFINITY).unwrap();
        assert_eq!(s.max_cfs_segment_size, i64::MAX);
    }

    #[test]
    fn use_compound_file_follows_the_ratio() {
        let ctx = BasicMergeContext::default();
        let infos = vec![seg("_a", 10, 0, 900), seg("_b", 10, 0, 100)];
        let mut p = TieredMergePolicy::default();
        // 10% of 1000 is 100: `_b` qualifies, `_a` does not.
        assert!(p.use_compound_file(&infos, &infos[1], &ctx).unwrap());
        assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
        p.compound_file_settings_mut()
            .set_no_cfs_ratio(0.0)
            .unwrap();
        assert!(!p.use_compound_file(&infos, &infos[1], &ctx).unwrap());
        p.compound_file_settings_mut()
            .set_no_cfs_ratio(1.0)
            .unwrap();
        assert!(p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
        p.compound_file_settings_mut()
            .set_max_cfs_segment_size_mb(0.0)
            .unwrap();
        assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
    }

    #[test]
    fn no_merge_policy_never_merges() {
        let ctx = BasicMergeContext::default();
        let infos = vec![seg("_a", 10, 5, 100), seg("_b", 10, 5, 100)];
        let mut p = NoMergePolicy::default();
        let t = MergeTrigger::Explicit;
        assert!(p.find_merges(t, &infos, &ctx).unwrap().is_none());
        assert!(p
            .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
            .unwrap()
            .is_none());
        assert!(p
            .find_forced_deletes_merges(&infos, &ctx)
            .unwrap()
            .is_none());
        assert!(p.find_full_flush_merges(t, &infos, &ctx).unwrap().is_none());
        let cfs = infos[0].clone().with_compound_file(true);
        assert!(p.use_compound_file(&infos, &cfs, &ctx).unwrap());
        assert!(!p.use_compound_file(&infos, &infos[0], &ctx).unwrap());
        assert_eq!(p.size(&infos[0], &ctx).unwrap(), i64::MAX);
        p.compound_file_settings_mut()
            .set_no_cfs_ratio(0.3)
            .unwrap();
        assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.3);
        assert!(!p.keep_fully_deleted_segment());
        assert_eq!(p.num_deletes_to_merge(&infos[0], 3), 3);
        let spec = p.find_merges_for_readers(&[5, 7]).unwrap();
        assert_eq!(spec.merges.len(), 1);
        assert_eq!(spec.merges[0].reader_count, 2);
        assert_eq!(spec.merges[0].total_max_doc, 12);
    }

    #[test]
    fn full_flush_merges_keep_only_small_inputs() {
        let ctx = BasicMergeContext::default();
        let mut config = MergePolicyConfig {
            segments_per_tier: 2,
            max_merge_at_once: 2,
            floor_segment_size: 1000,
            ..MergePolicyConfig::default()
        };
        let small: Vec<MergeSegment> = (0..4).map(|i| seg(&format!("_{i}"), 10, 0, 10)).collect();
        let p = TieredMergePolicy::new(config.clone());
        let t = MergeTrigger::FullFlush;
        let natural = p.find_merges(t, &small, &ctx).unwrap().unwrap();
        let full = p.find_full_flush_merges(t, &small, &ctx).unwrap().unwrap();
        assert_eq!(natural.groups(), full.groups());
        config.floor_segment_size = 5;
        let p = TieredMergePolicy::new(config);
        assert!(p.find_full_flush_merges(t, &small, &ctx).unwrap().is_none());
        assert_eq!(p.max_full_flush_merge_size(), 5);
    }

    #[test]
    fn merge_segment_from_stat_and_spec_groups() {
        let stat = SegmentStat {
            name: "_x".into(),
            doc_count: 3,
            del_count: 1,
            size_bytes: u64::MAX,
        };
        let s = MergeSegment::from(&stat);
        assert_eq!(s.size_in_bytes, i64::MAX);
        let merge = OneMerge::new(vec![s.clone(), seg("_y", 4, 0, 1)]);
        assert_eq!(merge.total_max_doc, 7);
        let mut spec = MergeSpecification::new();
        spec.add(merge);
        assert_eq!(
            spec.groups(),
            vec![vec!["_x".to_string(), "_y".to_string()]]
        );
    }
}
