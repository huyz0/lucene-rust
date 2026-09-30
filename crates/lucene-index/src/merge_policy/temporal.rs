//! Port of `org.apache.lucene.index.TemporalMergePolicy`: segments are
//! bucketed into time windows by the maximum value of a `LongPoint` timestamp
//! field, and only segments of the same window merge.
//!
//! # Where the date ranges come from
//!
//! Java reads each segment's `.fnm` and `PointValues.getMin/MaxPackedValue`
//! for the temporal field (`extractDateRangeFromSegment`), or takes a
//! package-private override map. Here a policy is a pure function of
//! [`MergeSegment`]s, so the ranges come from, in order:
//!
//! 1. [`TemporalMergePolicy::set_segment_date_range_overrides`] -- Java's
//!    `setSegmentDateRangeOverrides`, keyed by segment name;
//! 2. a resolver installed with [`TemporalMergePolicy::with_date_range_resolver`],
//!    normally [`DirectoryDateRanges`] -- which does exactly what Java's
//!    extraction does, against a [`Directory`] and the commit's segments.
//!
//! A segment the source cannot answer for is left out, as Java leaves out a
//! segment whose extraction throws or finds no points.
//!
//! # Iteration order
//!
//! `findForcedMerges`/`findForcedDeletesMerges` group segments by iterating a
//! `HashMap<SegmentCommitInfo, SegmentDateRange>` -- identity-hash order, which
//! differs from run to run of the same JVM. This port iterates in index order,
//! one of the orders Java may produce. `findMerges` is deterministic in Java
//! (it walks `SegmentInfos`) and identical here.
//!
//! # The clock
//!
//! Buckets depend on the age of a segment's newest timestamp, `now - max`.
//! Java reads `System.currentTimeMillis()`; so does this, unless a fixed clock
//! is installed with [`TemporalMergePolicy::set_now_millis`] (for tests).

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use lucene_store::directory::Directory;

use super::api::{
    CompoundFileSettings, Error, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge, Result,
};
use crate::segment_infos::SegmentCommitInfo;

/// `TemporalMergePolicy.SegmentDateRange`: a segment's smallest and largest
/// timestamp, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentDateRange {
    pub min_date: i64,
    pub max_date: i64,
}

/// Where [`TemporalMergePolicy`] reads a segment's date range from.
pub type DateRangeResolver = Arc<dyn Fn(&MergeSegment) -> Option<SegmentDateRange> + Send + Sync>;

/// `TemporalMergePolicy`.
#[derive(Clone)]
pub struct TemporalMergePolicy {
    temporal_field: String,
    base_time_seconds: i64,
    min_threshold: i32,
    use_exponential_buckets: bool,
    max_window_size_seconds: i64,
    max_age_seconds: i64,
    max_threshold: i32,
    compaction_ratio: f64,
    force_merge_deletes_pct_allowed: f64,
    overrides: Option<HashMap<String, SegmentDateRange>>,
    resolver: Option<DateRangeResolver>,
    now_millis: Option<i64>,
    compound: CompoundFileSettings,
}

impl fmt::Debug for TemporalMergePolicy {
    /// Java's `toString()` fields.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemporalMergePolicy")
            .field("temporal_field", &self.temporal_field)
            .field("base_time_seconds", &self.base_time_seconds)
            .field("min_threshold", &self.min_threshold)
            .field("max_threshold", &self.max_threshold)
            .field("use_exponential_buckets", &self.use_exponential_buckets)
            .field("max_window_size_seconds", &self.max_window_size_seconds)
            .field("max_age_seconds", &self.max_age_seconds)
            .field("compaction_ratio", &self.compaction_ratio)
            .field(
                "force_merge_deletes_pct_allowed",
                &self.force_merge_deletes_pct_allowed,
            )
            .finish()
    }
}

impl Default for TemporalMergePolicy {
    fn default() -> Self {
        TemporalMergePolicy {
            temporal_field: String::new(),
            base_time_seconds: 3600,
            min_threshold: 4,
            use_exponential_buckets: true,
            // `TimeUnit.DAYS.toSeconds(365)`.
            max_window_size_seconds: 365 * 24 * 3600,
            max_age_seconds: i64::MAX,
            max_threshold: 8,
            compaction_ratio: 1.2,
            force_merge_deletes_pct_allowed: 10.0,
            overrides: None,
            resolver: None,
            now_millis: None,
            compound: CompoundFileSettings::default(),
        }
    }
}

impl TemporalMergePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// `setTemporalField`.
    pub fn set_temporal_field(&mut self, field: &str) -> Result<&mut Self> {
        if field.trim().is_empty() {
            return Err(Error::IllegalArgument(
                "temporalField cannot be blank".into(),
            ));
        }
        self.temporal_field = field.to_string();
        Ok(self)
    }

    pub fn temporal_field(&self) -> &str {
        &self.temporal_field
    }

    /// `setBaseTimeInSeconds`.
    pub fn set_base_time_in_seconds(&mut self, seconds: i64) -> Result<&mut Self> {
        if seconds <= 0 {
            return Err(Error::IllegalArgument(
                "baseTimeSeconds must be positive".into(),
            ));
        }
        self.base_time_seconds = seconds;
        Ok(self)
    }

    pub fn base_time_in_seconds(&self) -> i64 {
        self.base_time_seconds
    }

    /// `setMinThreshold`.
    pub fn set_min_threshold(&mut self, min_threshold: i32) -> Result<&mut Self> {
        if min_threshold < 2 {
            return Err(Error::IllegalArgument(
                "minThreshold must be at least 2".into(),
            ));
        }
        if min_threshold > self.max_threshold {
            return Err(Error::IllegalArgument(format!(
                "minThreshold cannot exceed maxThreshold ({})",
                self.max_threshold
            )));
        }
        self.min_threshold = min_threshold;
        Ok(self)
    }

    pub fn min_threshold(&self) -> i32 {
        self.min_threshold
    }

    /// `disableExponentialBuckets`.
    pub fn disable_exponential_buckets(&mut self) -> &mut Self {
        self.use_exponential_buckets = false;
        self
    }

    pub fn use_exponential_buckets(&self) -> bool {
        self.use_exponential_buckets
    }

    /// `setMaxThreshold`.
    pub fn set_max_threshold(&mut self, max_threshold: i32) -> Result<&mut Self> {
        if max_threshold < self.min_threshold {
            return Err(Error::IllegalArgument(format!(
                "maxThreshold must be >= minThreshold ({})",
                self.min_threshold
            )));
        }
        self.max_threshold = max_threshold;
        Ok(self)
    }

    pub fn max_threshold(&self) -> i32 {
        self.max_threshold
    }

    /// `setCompactionRatio`.
    pub fn set_compaction_ratio(&mut self, ratio: f64) -> Result<&mut Self> {
        if ratio < 1.0 || ratio.is_nan() {
            return Err(Error::IllegalArgument(
                "compactionRatio must be >= 1.0".into(),
            ));
        }
        self.compaction_ratio = ratio;
        Ok(self)
    }

    pub fn compaction_ratio(&self) -> f64 {
        self.compaction_ratio
    }

    /// `setMaxWindowSizeSeconds`.
    pub fn set_max_window_size_seconds(&mut self, seconds: i64) -> Result<&mut Self> {
        if seconds <= 0 {
            return Err(Error::IllegalArgument(
                "maxWindowSizeSeconds must be positive".into(),
            ));
        }
        self.max_window_size_seconds = seconds;
        Ok(self)
    }

    pub fn max_window_size_seconds(&self) -> i64 {
        self.max_window_size_seconds
    }

    /// `setMaxAgeSeconds`.
    pub fn set_max_age_seconds(&mut self, seconds: i64) -> Result<&mut Self> {
        if seconds <= 0 {
            return Err(Error::IllegalArgument(
                "maxAgeSeconds must be positive".into(),
            ));
        }
        self.max_age_seconds = seconds;
        Ok(self)
    }

    pub fn max_age_seconds(&self) -> i64 {
        self.max_age_seconds
    }

    /// `setForceMergeDeletesPctAllowed`.
    pub fn set_force_merge_deletes_pct_allowed(&mut self, pct: f64) -> Result<&mut Self> {
        if pct < 0.0 || pct.is_nan() {
            return Err(Error::IllegalArgument(
                "forceMergeDeletesPctAllowed must be >= 0".into(),
            ));
        }
        self.force_merge_deletes_pct_allowed = pct;
        Ok(self)
    }

    pub fn force_merge_deletes_pct_allowed(&self) -> f64 {
        self.force_merge_deletes_pct_allowed
    }

    /// `setSegmentDateRangeOverrides`: when set, the only date ranges the
    /// policy sees.
    pub fn set_segment_date_range_overrides(
        &mut self,
        overrides: Option<HashMap<String, SegmentDateRange>>,
    ) -> &mut Self {
        self.overrides = overrides;
        self
    }

    /// Reads each segment's date range through `resolver` (see the module
    /// doc; [`DirectoryDateRanges`] is Java's extraction).
    pub fn with_date_range_resolver(&mut self, resolver: DateRangeResolver) -> &mut Self {
        self.resolver = Some(resolver);
        self
    }

    /// Fixes the clock at `now` (milliseconds since the epoch); `None` reads
    /// the system clock, as Java does.
    pub fn set_now_millis(&mut self, now: Option<i64>) -> &mut Self {
        self.now_millis = now;
        self
    }

    fn now(&self) -> i64 {
        self.now_millis.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
                .unwrap_or(0)
        })
    }

    /// `resolveSegmentDateRanges`.
    fn resolve(&self, infos: &[MergeSegment]) -> HashMap<String, SegmentDateRange> {
        if let Some(overrides) = &self.overrides {
            return overrides.clone();
        }
        let mut out = HashMap::new();
        if let Some(resolver) = &self.resolver {
            for info in infos {
                if let Some(range) = resolver(info) {
                    out.insert(info.name.clone(), range);
                }
            }
        }
        out
    }

    /// `getBucketForTimestamp`: the window start (seconds) holding
    /// `timestamp_seconds`, or `-1` for a segment too old to merge.
    // SENTINEL: `-1` = older than `maxAgeSeconds`, never a merge window. The
    // only caller, `assign`, files it as a bucket key; both readers of the
    // buckets (`findMerges`' window loop and `find_forced_merges`) skip the
    // `-1` window, as Java does.
    fn bucket_for_timestamp(&self, timestamp_seconds: i64, now_seconds: i64) -> i64 {
        let mut age = now_seconds.wrapping_sub(timestamp_seconds);
        if age < 0 {
            age = 0;
        }
        if age > self.max_age_seconds {
            return -1;
        }
        if !self.use_exponential_buckets {
            return floor_to(timestamp_seconds, self.base_time_seconds);
        }
        let mut bucket = self.base_time_seconds;
        let min = i64::from(self.min_threshold);
        while age >= bucket.wrapping_mul(min) && bucket < self.max_window_size_seconds {
            bucket = bucket.wrapping_mul(min);
        }
        if bucket > self.max_window_size_seconds {
            bucket = self.max_window_size_seconds;
        }
        floor_to(timestamp_seconds, bucket)
    }

    /// `assignToBucket`.
    fn assign<'s>(
        &self,
        buckets: &mut BTreeMap<i64, Vec<&'s MergeSegment>>,
        now: i64,
        segment: &'s MergeSegment,
        range: SegmentDateRange,
    ) {
        let bucket =
            self.bucket_for_timestamp(range.max_date.wrapping_div(1000), now.wrapping_div(1000));
        buckets.entry(bucket).or_default().push(segment);
    }

    /// `groupSegmentsByTimeWindow`, over `segments` in index order (see the
    /// module doc).
    fn group<'s>(
        &self,
        segments: &[(&'s MergeSegment, SegmentDateRange)],
    ) -> BTreeMap<i64, Vec<&'s MergeSegment>> {
        let mut buckets = BTreeMap::new();
        let now = self.now();
        for (segment, range) in segments {
            self.assign(&mut buckets, now, segment, *range);
        }
        buckets
    }

    /// `findMergeCandidates`.
    fn find_merge_candidates(
        &self,
        buckets: &BTreeMap<i64, Vec<&MergeSegment>>,
        ranges: &HashMap<String, SegmentDateRange>,
    ) -> Option<MergeSpecification> {
        let mut spec: Option<MergeSpecification> = None;
        let min = usize::try_from(self.min_threshold).unwrap_or(usize::MAX);
        for (&window_start, in_window) in buckets {
            if window_start == -1 {
                // Segments too old to merge.
                continue;
            }
            if in_window.len() < min {
                continue;
            }
            let merges = self.plan_window_merges(in_window, ranges);
            if merges.is_empty() {
                continue;
            }
            let spec = spec.get_or_insert_with(MergeSpecification::new);
            for segments in merges {
                spec.add(OneMerge::new(segments));
            }
        }
        spec
    }

    /// `planWindowMerges`: newest first, runs of `minThreshold..=maxThreshold`
    /// segments whose total document count reaches `compactionRatio` times the
    /// largest's.
    fn plan_window_merges(
        &self,
        in_window: &[&MergeSegment],
        ranges: &HashMap<String, SegmentDateRange>,
    ) -> Vec<Vec<MergeSegment>> {
        let mut ordered: Vec<&MergeSegment> = in_window.to_vec();
        // `Comparator.comparingLong(maxDate).reversed()`, stable.
        ordered.sort_by(|a, b| {
            let ma = ranges.get(&a.name).map_or(0, |r| r.max_date);
            let mb = ranges.get(&b.name).map_or(0, |r| r.max_date);
            mb.cmp(&ma)
        });
        let min = usize::try_from(self.min_threshold).unwrap_or(usize::MAX);
        let max = usize::try_from(self.max_threshold).unwrap_or(usize::MAX);
        let mut planned = Vec::new();
        let mut cursor = 0usize;
        while ordered.len().saturating_sub(cursor) >= min {
            let mut total_docs: i64 = 0;
            let mut largest_docs: i64 = 0;
            let mut end = cursor;
            let mut emitted = false;
            while end < ordered.len() && end.saturating_sub(cursor) < max {
                let docs = i64::from(ordered[end].max_doc);
                total_docs = total_docs.wrapping_add(docs);
                largest_docs = largest_docs.max(docs);
                end = end.saturating_add(1);
                let size = end.saturating_sub(cursor);
                if size < min {
                    continue;
                }
                let reached_max = size == max;
                let exhausted = end == ordered.len();
                let take = if self.compaction_ratio <= 1.0 {
                    reached_max || exhausted
                } else {
                    let ratio_satisfied =
                        total_docs as f64 >= (largest_docs as f64 * self.compaction_ratio).ceil();
                    ratio_satisfied || reached_max
                };
                if take {
                    planned.push(ordered[cursor..end].iter().map(|s| (*s).clone()).collect());
                    cursor = end;
                    emitted = true;
                    break;
                }
            }
            if !emitted {
                break;
            }
        }
        planned
    }

    /// `buildForcedMerges`.
    fn build_forced_merges(
        &self,
        candidates: &[&MergeSegment],
        max_segment_count: i32,
    ) -> Option<MergeSpecification> {
        if candidates.len() < 2 {
            return None;
        }
        let mut spec: Option<MergeSpecification> = None;
        let max_count = usize::try_from(max_segment_count).unwrap_or(0);
        let max_threshold = usize::try_from(self.max_threshold).unwrap_or(usize::MAX);
        let mut remaining = candidates.len();
        let mut offset = 0usize;
        while remaining > max_count && candidates.len().saturating_sub(offset) >= 2 {
            let needed = remaining.saturating_sub(max_count);
            let inputs = max_threshold
                .min(needed.saturating_add(1).max(2))
                .min(candidates.len().saturating_sub(offset));
            let batch = &candidates[offset..offset.saturating_add(inputs)];
            spec.get_or_insert_with(MergeSpecification::new)
                .add(OneMerge::new(batch.iter().map(|s| (*s).clone()).collect()));
            offset = offset.saturating_add(inputs);
            remaining = remaining.saturating_sub(inputs.saturating_sub(1));
        }
        spec
    }

    /// `buildSequentialMerges`.
    fn build_sequential_merges(&self, candidates: &[&MergeSegment]) -> Option<MergeSpecification> {
        if candidates.len() < 2 {
            return None;
        }
        let max_threshold = usize::try_from(self.max_threshold).unwrap_or(usize::MAX);
        let mut spec: Option<MergeSpecification> = None;
        let mut batch: Vec<MergeSegment> = Vec::new();
        for info in candidates {
            batch.push((*info).clone());
            if batch.len() == max_threshold {
                spec.get_or_insert_with(MergeSpecification::new)
                    .add(OneMerge::new(std::mem::take(&mut batch)));
            }
        }
        if batch.len() > 1 {
            spec.get_or_insert_with(MergeSpecification::new)
                .add(OneMerge::new(batch));
        }
        spec
    }
}

impl MergePolicy for TemporalMergePolicy {
    fn find_merges(
        &self,
        _trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        if self.temporal_field.trim().is_empty() || infos.is_empty() {
            return Ok(None);
        }
        let merging = ctx.merging_segments();
        let ranges = self.resolve(infos);
        if ranges.is_empty() {
            return Ok(None);
        }
        let mut buckets: BTreeMap<i64, Vec<&MergeSegment>> = BTreeMap::new();
        let now = self.now();
        for info in infos {
            if merging.contains(&info.name) {
                continue;
            }
            if let Some(range) = ranges.get(&info.name) {
                self.assign(&mut buckets, now, info, *range);
            }
        }
        if buckets.is_empty() {
            return Ok(None);
        }
        Ok(self.find_merge_candidates(&buckets, &ranges))
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        if max_segment_count < 1 {
            return Err(Error::IllegalArgument(
                "maxSegmentCount must be >= 1".into(),
            ));
        }
        let ranges = self.resolve(infos);
        if ranges.is_empty() {
            return Ok(None);
        }
        let merge_all = segments_to_merge.is_empty();
        let merging = ctx.merging_segments();
        let mut eligible: Vec<(&MergeSegment, SegmentDateRange)> = Vec::new();
        for info in infos {
            if merge_all || segments_to_merge.contains_key(&info.name) {
                if merging.contains(&info.name) {
                    // A forced merge is already running.
                    return Ok(None);
                }
                if let Some(range) = ranges.get(&info.name) {
                    eligible.push((info, *range));
                }
            }
        }
        if eligible.is_empty() {
            return Ok(None);
        }
        let buckets = self.group(&eligible);
        let window_count = i32::try_from(buckets.len()).unwrap_or(i32::MAX).max(1);
        // ARITH: `window_count >= 1` (the `max(1)` above), so this can neither
        // divide by zero nor overflow (`i32::MIN / -1` needs a negative divisor).
        #[allow(clippy::arithmetic_side_effects)]
        let per_window = (max_segment_count / window_count)
            .max(1)
            .min(self.max_threshold);
        let mut spec: Option<MergeSpecification> = None;
        for (&window_start, segments) in &buckets {
            if window_start == -1 || segments.len() < 2 {
                continue;
            }
            if let Some(bucket_spec) = self.build_forced_merges(segments, per_window) {
                spec.get_or_insert_with(MergeSpecification::new)
                    .merges
                    .extend(bucket_spec.merges);
            }
        }
        Ok(spec)
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let ranges = self.resolve(infos);
        if ranges.is_empty() {
            return Ok(None);
        }
        let merging = ctx.merging_segments();
        let mut candidates: Vec<(&MergeSegment, SegmentDateRange)> = Vec::new();
        let mut ratios: HashMap<&str, f64> = HashMap::new();
        for info in infos {
            if merging.contains(&info.name) {
                continue;
            }
            let del_count = ctx.num_deletes_to_merge(info)?;
            if del_count <= 0 {
                continue;
            }
            let max_doc = info.max_doc.max(1);
            let pct = (100.0 * f64::from(del_count)) / f64::from(max_doc);
            if pct > self.force_merge_deletes_pct_allowed {
                if let Some(range) = ranges.get(&info.name) {
                    candidates.push((info, *range));
                    ratios.insert(info.name.as_str(), pct);
                }
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        let buckets = self.group(&candidates);
        let mut spec: Option<MergeSpecification> = None;
        for segments in buckets.values() {
            if segments.len() < 2 {
                continue;
            }
            let mut sorted = segments.clone();
            // Highest delete ratio first (`Double.compare(b, a)`), stable.
            sorted.sort_by(|a, b| {
                let ra = ratios.get(a.name.as_str()).copied().unwrap_or(0.0);
                let rb = ratios.get(b.name.as_str()).copied().unwrap_or(0.0);
                rb.total_cmp(&ra)
            });
            if let Some(bucket_spec) = self.build_sequential_merges(&sorted) {
                spec.get_or_insert_with(MergeSpecification::new)
                    .merges
                    .extend(bucket_spec.merges);
            }
        }
        Ok(spec)
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.compound
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        &mut self.compound
    }
}

/// `getTemporalFieldDivisor` applied: converts a raw timestamp (seconds,
/// milliseconds or microseconds, told apart by magnitude) to milliseconds.
pub fn to_millis(min: i64, max: i64) -> SegmentDateRange {
    let divisor: i64 = if max > 100_000_000_000_000 {
        1_000
    } else if max > 100_000_000_000 {
        1
    } else {
        -1_000
    };
    if divisor < 0 {
        let multiplier = divisor.wrapping_neg();
        SegmentDateRange {
            min_date: min.wrapping_mul(multiplier),
            max_date: max.wrapping_mul(multiplier),
        }
    } else {
        SegmentDateRange {
            min_date: div_positive(min, divisor),
            max_date: div_positive(max, divisor),
        }
    }
}

/// `extractDateRangeFromSegment`: the temporal field's min/max `LongPoint`
/// values in one segment, read from its `.fnm` and `.kdm` (through the
/// compound archive when the segment has one). `Ok(None)` when the segment
/// does not index the field as points, as Java returns `null`.
pub fn segment_date_range(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    field: &str,
) -> std::result::Result<Option<SegmentDateRange>, crate::field_updates::Error> {
    let si_bytes = dir.open(&format!("{}.si", sci.segment_name))?;
    let si = crate::segment_info::parse(&si_bytes, &sci.segment_id)?;
    let compound;
    let (reader_dir, files): (&dyn Directory, Vec<String>) = if si.is_compound_file {
        compound =
            crate::compound_reader::CompoundReader::open(dir, &sci.segment_name, &sci.segment_id)?;
        let files = compound.member_files();
        (&compound, files)
    } else {
        (dir, si.files.clone())
    };
    // Java reads the segment's own `.fnm` (`fieldInfosFormat().read(dir, si,
    // "")`), not a generational one.
    let Some(fnm) = files.iter().find(|f| f.ends_with(".fnm")) else {
        return Ok(None);
    };
    let fnm_bytes = reader_dir.open(fnm)?;
    let infos = lucene_codecs::field_infos::parse(&fnm_bytes, &sci.segment_id, "")?;
    let Some(info) = infos.field_by_name(field) else {
        return Ok(None);
    };
    if info.point_dimension_count == 0 {
        return Ok(None);
    }
    let find = |ext: &str| files.iter().find(|f| f.ends_with(ext)).cloned();
    let (Some(kdm), Some(kdi), Some(kdd)) = (find(".kdm"), find(".kdi"), find(".kdd")) else {
        return Ok(None);
    };
    let kdm = reader_dir.open(&kdm)?;
    let kdi = reader_dir.open(&kdi)?;
    let kdd = reader_dir.open(&kdd)?;
    let fields =
        lucene_codecs::points::open_meta(&kdm, &kdi, &kdd, &sci.segment_id, "").map_err(|e| {
            crate::field_updates::Error::Store(lucene_store::Error::Corrupted(e.to_string()))
        })?;
    let Some((_, points)) = fields.iter().find(|(n, _)| *n == info.number) else {
        return Ok(None);
    };
    let (Some(min), Some(max)) = (
        decode_long_dimension(&points.min_packed_value),
        decode_long_dimension(&points.max_packed_value),
    ) else {
        return Ok(None);
    };
    Ok(Some(to_millis(min, max)))
}

/// `LongPoint.decodeDimension(packed, 0)`: big-endian with the sign bit
/// flipped.
fn decode_long_dimension(packed: &[u8]) -> Option<i64> {
    let bytes: [u8; 8] = packed.get(..8)?.try_into().ok()?;
    Some((u64::from_be_bytes(bytes) ^ 0x8000_0000_0000_0000) as i64)
}

/// Java's extraction as a resolver: every commit's segment's date range,
/// read once from `dir` and served by segment name.
#[derive(Debug, Clone, Default)]
pub struct DirectoryDateRanges {
    ranges: HashMap<String, SegmentDateRange>,
}

impl DirectoryDateRanges {
    /// Reads the date range of each of `segments` for `field`. A segment whose
    /// read fails is left out, as Java logs and skips it.
    pub fn read(dir: &dyn Directory, segments: &[SegmentCommitInfo], field: &str) -> Self {
        let mut ranges = HashMap::new();
        for sci in segments {
            if let Ok(Some(range)) = segment_date_range(dir, sci, field) {
                ranges.insert(sci.segment_name.clone(), range);
            }
        }
        DirectoryDateRanges { ranges }
    }

    /// The range read for `segment`.
    pub fn get(&self, segment: &str) -> Option<SegmentDateRange> {
        self.ranges.get(segment).copied()
    }

    /// This as a [`DateRangeResolver`].
    pub fn into_resolver(self) -> DateRangeResolver {
        Arc::new(move |s: &MergeSegment| self.ranges.get(&s.name).copied())
    }
}

/// `value / divisor` for a `divisor` that is positive by construction.
// ARITH: every caller passes a positive divisor (a validated setter's value,
// or the constant 1 or 1000), so `/` neither divides by zero nor overflows
// (`i64::MIN / -1` needs a negative divisor). Java's `/` truncates toward
// zero exactly as Rust's does.
#[allow(clippy::arithmetic_side_effects)]
fn div_positive(value: i64, divisor: i64) -> i64 {
    debug_assert!(divisor > 0);
    value / divisor.max(1)
}

/// `timestamp / unit * unit` -- Java's window start for a positive `unit`.
fn floor_to(timestamp: i64, unit: i64) -> i64 {
    div_positive(timestamp, unit).wrapping_mul(unit)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::super::api::BasicMergeContext;
    use super::*;

    const HOUR_MS: i64 = 3_600_000;

    fn policy(ranges: &[(&str, i64)]) -> TemporalMergePolicy {
        let mut p = TemporalMergePolicy::new();
        p.set_temporal_field("ts").unwrap();
        p.disable_exponential_buckets();
        p.set_now_millis(Some(1_000 * HOUR_MS));
        p.set_segment_date_range_overrides(Some(
            ranges
                .iter()
                .map(|(n, max)| {
                    (
                        n.to_string(),
                        SegmentDateRange {
                            min_date: max - 1,
                            max_date: *max,
                        },
                    )
                })
                .collect(),
        ));
        p
    }

    fn seg(name: &str, docs: i32, del: i32) -> MergeSegment {
        MergeSegment::new(name, docs, del, 100)
    }

    #[test]
    fn setters_validate_like_java() {
        let mut p = TemporalMergePolicy::new();
        assert!(p.set_temporal_field(" ").is_err());
        assert!(p.set_base_time_in_seconds(0).is_err());
        assert!(p.set_min_threshold(1).is_err());
        assert!(p.set_min_threshold(9).is_err());
        assert!(p.set_max_threshold(3).is_err());
        assert!(p.set_compaction_ratio(0.5).is_err());
        assert!(p.set_max_window_size_seconds(0).is_err());
        assert!(p.set_max_age_seconds(0).is_err());
        assert!(p.set_force_merge_deletes_pct_allowed(-1.0).is_err());
        p.set_temporal_field("t").unwrap();
        p.set_base_time_in_seconds(10).unwrap();
        p.set_max_threshold(12).unwrap();
        p.set_min_threshold(6).unwrap();
        p.set_compaction_ratio(1.0).unwrap();
        p.set_max_window_size_seconds(100).unwrap();
        p.set_max_age_seconds(1000).unwrap();
        p.set_force_merge_deletes_pct_allowed(5.0).unwrap();
        assert_eq!(p.temporal_field(), "t");
        assert_eq!(p.base_time_in_seconds(), 10);
        assert_eq!(p.min_threshold(), 6);
        assert_eq!(p.max_threshold(), 12);
        assert_eq!(p.compaction_ratio(), 1.0);
        assert_eq!(p.max_window_size_seconds(), 100);
        assert_eq!(p.max_age_seconds(), 1000);
        assert_eq!(p.force_merge_deletes_pct_allowed(), 5.0);
        assert!(p.use_exponential_buckets());
        assert!(format!("{p:?}").contains("temporal_field"));
    }

    #[test]
    fn no_field_or_no_ranges_means_no_merges() {
        let ctx = BasicMergeContext::default();
        let infos = vec![seg("_a", 1, 0)];
        let p = TemporalMergePolicy::new();
        assert!(p
            .find_merges(MergeTrigger::Explicit, &infos, &ctx)
            .unwrap()
            .is_none());
        let p = policy(&[]);
        assert!(p
            .find_merges(MergeTrigger::Explicit, &infos, &ctx)
            .unwrap()
            .is_none());
        assert!(p
            .find_forced_merges(&infos, 1, &HashMap::new(), &ctx)
            .unwrap()
            .is_none());
        assert!(p
            .find_forced_deletes_merges(&infos, &ctx)
            .unwrap()
            .is_none());
        assert!(p
            .find_forced_merges(&infos, 0, &HashMap::new(), &ctx)
            .is_err());
    }

    #[test]
    fn same_window_segments_merge_newest_first() {
        let ctx = BasicMergeContext::default();
        let names = ["_a", "_b", "_c", "_d", "_e"];
        let ranges: Vec<(&str, i64)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, 10 * HOUR_MS + i as i64))
            .collect();
        let p = policy(&ranges);
        let infos: Vec<MergeSegment> = names.iter().map(|n| seg(n, 10, 0)).collect();
        let spec = p
            .find_merges(MergeTrigger::Explicit, &infos, &ctx)
            .unwrap()
            .unwrap();
        // Newest first; 1.2 * 10 = 12 docs reached at the minimum of 4.
        assert_eq!(spec.groups(), vec![vec!["_e", "_d", "_c", "_b"]]);
    }

    #[test]
    fn bucket_sizes_grow_exponentially_and_cap() {
        let mut p = TemporalMergePolicy::new();
        // age 0 -> base window.
        assert_eq!(p.bucket_for_timestamp(7_300, 7_300), 7_200);
        // Old: the window grows by minThreshold until it covers the age.
        assert_eq!(p.bucket_for_timestamp(0, 4 * 3600), 0);
        p.set_max_age_seconds(10).unwrap();
        assert_eq!(p.bucket_for_timestamp(0, 11), -1);
        p.set_max_age_seconds(i64::MAX).unwrap();
        p.set_max_window_size_seconds(5_000).unwrap();
        assert_eq!(p.bucket_for_timestamp(0, 1_000_000_000), 0);
        assert_eq!(p.bucket_for_timestamp(12_000, 1_000_000_000), 10_000);
    }

    #[test]
    fn divisor_detects_units() {
        assert_eq!(to_millis(1, 2).max_date, 2_000);
        assert_eq!(to_millis(1, 200_000_000_000).max_date, 200_000_000_000);
        assert_eq!(
            to_millis(1_000, 200_000_000_000_000).max_date,
            200_000_000_000
        );
        assert_eq!(decode_long_dimension(&[0x80, 0, 0, 0, 0, 0, 0, 5]), Some(5));
        assert_eq!(decode_long_dimension(&[0x80]), None);
    }

    #[test]
    fn resolver_and_directory_ranges() {
        let mut p = TemporalMergePolicy::new();
        p.set_temporal_field("ts").unwrap();
        p.set_now_millis(Some(0));
        let r = DirectoryDateRanges::default();
        assert!(r.get("_a").is_none());
        p.with_date_range_resolver(Arc::new(|s: &MergeSegment| {
            (s.name != "_z").then_some(SegmentDateRange {
                min_date: 0,
                max_date: 0,
            })
        }));
        let infos: Vec<MergeSegment> = ["_a", "_b", "_c", "_d", "_z"]
            .iter()
            .map(|n| seg(n, 1, 0))
            .collect();
        let spec = p
            .find_merges(
                MergeTrigger::Explicit,
                &infos,
                &BasicMergeContext::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(spec.groups(), vec![vec!["_a", "_b", "_c", "_d"]]);
    }
}
