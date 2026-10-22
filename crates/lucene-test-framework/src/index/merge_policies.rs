//! The test framework's merge policies: `ForceMergePolicy` (forced merges
//! only), `AlcoholicMergePolicy` (a `LogMergePolicy` with a drunk `size`)
//! and `MockRandomMergePolicy` (random merges, random compound files,
//! sometimes reversed documents).
//!
//! Differs:
//!
//! - `AlcoholicMergePolicy`'s `TimeZone` is a fixed UTC offset (no daylight
//!   saving): the calendar never moves, so only its hour of day is read.
//! - `MockRandomOneMerge.wrapForMerge`'s two wrappers exist to defeat
//!   Java's bulk merges. The field renumbering is a real
//!   [`MismatchedCodecReader`]; the `SlowCodecReaderWrapper` over a
//!   `FilterLeafReader` (and `MergeReaderWrapper`) is the reader itself,
//!   since this writer's merge has no bulk path and no merge instances.
//!   Either way the merge reads the sources' own files: a wrapper changes
//!   only which documents are live as far as it can see
//!   (`lucene_index::merge_policy::MergeHooks`).
//! - `reorder` draws whether to submit a no-op task to the merge's executor,
//!   as Java's always-present same-thread executor does, but submits
//!   nothing.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_index::merge_policy::api::Result;
use lucene_index::merge_policy::filter::FilterMergePolicy;
use lucene_index::merge_policy::log::LogMergePolicy;
use lucene_index::merge_policy::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge,
};
use lucene_search::reader::merge_readers::{MergeReaderHooks, SegmentMergeHooks};
use lucene_search::reader::sorting::DocMap;
use lucene_search::reader::{CodecReader, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use super::reader_wrappers::MismatchedCodecReader;
use crate::util::test_util;

/// `Collections.shuffle(list, random)`: from the end, each element swapped
/// with one at or before it.
pub fn shuffle<T>(random: &mut JavaRandom, list: &mut [T]) {
    for i in (2..=list.len()).rev() {
        let j = random.next_int_bounded(i as i32) as usize;
        list.swap(i - 1, j);
    }
}

/// `ForceMergePolicy`: the wrapped policy's forced merges, never a natural
/// one -- so a test controls when segments merge.
#[derive(Debug)]
pub struct ForceMergePolicy(FilterMergePolicy);

impl ForceMergePolicy {
    /// `new ForceMergePolicy(in)`.
    pub fn new(inner: Box<dyn MergePolicy>) -> Self {
        Self(FilterMergePolicy::new(inner))
    }

    /// `FilterMergePolicy.unwrap()`.
    pub fn unwrap(&self) -> &dyn MergePolicy {
        self.0.unwrap()
    }
}

impl MergePolicy for ForceMergePolicy {
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
    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.0.find_full_flush_merges(trigger, infos, ctx)
    }
    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.0.use_compound_file(infos, merged, ctx)
    }
    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        self.0.size(info, ctx)
    }
    fn max_full_flush_merge_size(&self) -> i64 {
        self.0.max_full_flush_merge_size()
    }
    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        self.0.find_merges_for_readers(max_docs)
    }
    fn keep_fully_deleted_segment(&self) -> bool {
        self.0.keep_fully_deleted_segment()
    }
    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.0.num_deletes_to_merge(info, del_count)
    }
    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.0.compound_file_settings()
    }
    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.0.compound_file_settings_mut()
    }
}

/// `AlcoholicMergePolicy.Drink`: how much bigger a segment looks.
const DRUNK_FACTORS: [i64; 5] = [15, 17, 21, 22, 30];

/// `AlcoholicMergePolicy`: a [`LogMergePolicy`] that, at night (and at
/// random moments of the day: it is five o'clock somewhere), sees segments
/// 15 to 30 times their size -- so tests meet the bizarre merges a user's
/// policy might choose.
#[derive(Debug)]
pub struct AlcoholicMergePolicy;

impl AlcoholicMergePolicy {
    /// `new AlcoholicMergePolicy(tz, random)`, with the zone as its offset
    /// from UTC in milliseconds. The policy owns `random`, as Java's keeps
    /// the one it is given (`newAlcoholicMergePolicy` hands it
    /// `new Random(r.nextLong())`).
    // Java's subclass is a `LogMergePolicy` with another `size`: that is
    // what this returns.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(utc_offset_millis: i64, mut random: JavaRandom) -> LogMergePolicy {
        let millis = test_util::next_long(&mut random, 0, i64::MAX);
        let max_merge_size = test_util::next_int(&mut random, 1024 * 1024, i32::MAX);
        let hour_of_day = hour_of_day(millis, utc_offset_millis);
        let own = Mutex::new(random);
        let mut policy = LogMergePolicy::with_size_fn(Arc::new(
            move |info: &MergeSegment, _ctx: &dyn MergeContext| {
                let mut r = lock(&own);
                if !(6..=20).contains(&hour_of_day) || r.next_int_bounded(23) == 5 {
                    let drink =
                        DRUNK_FACTORS[r.next_int_bounded(DRUNK_FACTORS.len() as i32) as usize];
                    return Ok(drink.wrapping_mul(info.size_in_bytes));
                }
                Ok(info.size_in_bytes)
            },
        ));
        policy.max_merge_size = i64::from(max_merge_size);
        policy
    }
}

/// `Calendar.get(HOUR_OF_DAY)` at `millis` since the epoch, `offset` from
/// UTC.
pub fn hour_of_day(millis: i64, offset: i64) -> i64 {
    millis.wrapping_add(offset).rem_euclid(24 * 60 * 60 * 1000) / (60 * 60 * 1000)
}

fn lock(m: &Mutex<JavaRandom>) -> MutexGuard<'_, JavaRandom> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `MockRandomMergePolicy`: merges a random subset of the segments now and
/// then, packs forced merges in random runs, chooses compound files four
/// times in five, and sometimes merges through
/// [`MockRandomOneMerge`]'s hooks.
#[derive(Debug)]
pub struct MockRandomMergePolicy {
    random: Mutex<JavaRandom>,
    do_non_bulk_merges: bool,
    compound: CompoundFileSettings,
}

impl MockRandomMergePolicy {
    /// `new MockRandomMergePolicy(random)`: a private stream forked from
    /// `random`.
    pub fn new(random: &mut JavaRandom) -> Self {
        Self {
            random: Mutex::new(JavaRandom::new(random.next_long())),
            do_non_bulk_merges: true,
            compound: CompoundFileSettings::default(),
        }
    }

    /// `setDoNonBulkMerges`.
    pub fn set_do_non_bulk_merges(&mut self, v: bool) {
        self.do_non_bulk_merges = v;
    }

    fn one_merge(&self, r: &mut JavaRandom, segments: Vec<MergeSegment>) -> OneMerge {
        if self.do_non_bulk_merges && r.next_boolean() {
            let hooks = MockRandomOneMerge::new(r.next_long());
            OneMerge::new(segments).with_hooks(Arc::new(SegmentMergeHooks::new(Arc::new(hooks))))
        } else {
            OneMerge::new(segments)
        }
    }
}

impl MergePolicy for MockRandomMergePolicy {
    fn find_merges(
        &self,
        _trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let merging = ctx.merging_segments();
        let mut segments: Vec<MergeSegment> = infos
            .iter()
            .filter(|s| !merging.contains(&s.name))
            .cloned()
            .collect();
        let num_segments = segments.len();
        let mut r = lock(&self.random);
        if num_segments > 1 && (num_segments > 30 || r.next_int_bounded(5) == 3) {
            shuffle(&mut r, &mut segments);
            let segs_to_merge = test_util::next_int(&mut r, 1, num_segments as i32) as usize;
            segments.truncate(segs_to_merge);
            let mut spec = MergeSpecification::new();
            spec.add(self.one_merge(&mut r, segments));
            return Ok(Some(spec));
        }
        Ok(None)
    }

    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        _max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let mut eligible: Vec<MergeSegment> = infos
            .iter()
            .filter(|s| segments_to_merge.contains_key(&s.name))
            .cloned()
            .collect();
        let needs_merge = match eligible.as_slice() {
            [] => false,
            [one] => !self.is_merged(infos, one, ctx)?,
            _ => true,
        };
        if !needs_merge {
            return Ok(None);
        }
        let mut r = lock(&self.random);
        let mut spec = MergeSpecification::new();
        shuffle(&mut r, &mut eligible);
        let mut upto = 0usize;
        while upto < eligible.len() {
            let max = 10.min(eligible.len() - upto);
            let inc = if max <= 2 {
                max
            } else {
                test_util::next_int(&mut r, 2, max as i32) as usize
            };
            spec.add(self.one_merge(&mut r, eligible[upto..upto + inc].to_vec()));
            upto += inc;
        }
        for merge in &spec.merges {
            for info in &merge.segments {
                debug_assert!(segments_to_merge.contains_key(&info.name));
            }
        }
        Ok(Some(spec))
    }

    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.find_merges(MergeTrigger::Explicit, infos, ctx)
    }

    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        let Some(spec) = self.find_merges(trigger, infos, ctx)? else {
            return Ok(None);
        };
        // Do not return any merges involving already-merging segments.
        let merging: &HashSet<String> = ctx.merging_segments();
        let mut filtered = MergeSpecification::new();
        for one in spec.merges {
            let non_merging: Vec<MergeSegment> = one
                .segments
                .iter()
                .filter(|s| !merging.contains(&s.name))
                .cloned()
                .collect();
            if non_merging.len() != one.segments.len() {
                if !non_merging.is_empty() {
                    filtered.add(OneMerge::new(non_merging));
                }
            } else {
                filtered.add(one);
            }
        }
        Ok((!filtered.merges.is_empty()).then_some(filtered))
    }

    fn use_compound_file(
        &self,
        _infos: &[MergeSegment],
        _merged: &MergeSegment,
        _ctx: &dyn MergeContext,
    ) -> Result<bool> {
        // 80% of the time we create CFS:
        Ok(lock(&self.random).next_int_bounded(5) != 1)
    }

    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.compound
    }

    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        &mut self.compound
    }
}

/// `MockRandomMergePolicy.MockRandomOneMerge`'s hooks: a random
/// `wrapForMerge` (see the module doc) and, one time in two, the merged
/// documents in reverse order (blocks kept whole).
#[derive(Debug)]
pub struct MockRandomOneMerge {
    r: Mutex<JavaRandom>,
}

impl MockRandomOneMerge {
    /// `new MockRandomOneMerge(segments, seed)`'s stream.
    pub fn new(seed: i64) -> Self {
        Self {
            r: Mutex::new(JavaRandom::new(seed)),
        }
    }
}

impl MergeReaderHooks for MockRandomOneMerge {
    fn wrap_for_merge(
        &self,
        reader: Arc<dyn CodecReader>,
    ) -> lucene_search::Result<Arc<dyn CodecReader>> {
        // 0: a slow wrapper (read as the reader itself here, module doc),
        // 1: renumbered fields, else unchanged.
        let mut r = lock(&self.r);
        if r.next_int_bounded(7) == 1 {
            return Ok(Arc::new(MismatchedCodecReader::new(reader, &mut r)));
        }
        Ok(reader)
    }

    fn reorder(&self, reader: &dyn CodecReader) -> lucene_search::Result<Option<DocMap>> {
        let mut r = lock(&self.r);
        if r.next_boolean() {
            return reverse(reader).map(Some);
        }
        // Java's merge always has an intra-merge executor (a same-thread one
        // by default), so it draws whether to submit a no-op task to it.
        let _submit = r.next_boolean();
        Ok(None)
    }
}

/// `MockRandomMergePolicy.reverse(reader)`: the documents in reverse order;
/// with a parent field, each block (its children, then its parent) moves
/// as one, in order inside.
///
/// # Errors
/// [`lucene_search::Error::IllegalArgument`] when documents follow the last
/// parent, which no block covers.
pub fn reverse(reader: &dyn CodecReader) -> lucene_search::Result<DocMap> {
    let max_doc = reader.max_doc();
    let parents = match reader.field_infos().parent_field() {
        None => None,
        Some(field) => {
            let mut bits = FixedBitSet::new(max_doc as usize);
            if let Some(mut values) = reader.numeric_doc_values(field)? {
                loop {
                    let doc = values.next_doc()?;
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    // FBS: `bits` is `FixedBitSet::new(max_doc)` and a
                    // doc-values iterator stops below its reader's `max_doc`.
                    bits.set(doc as usize);
                }
            }
            Some(bits)
        }
    };
    let Some(parents) = parents else {
        return DocMap::from_new_to_old((0..max_doc).rev().collect());
    };
    let unblocked =
        || lucene_search::Error::IllegalArgument("a document follows the last parent".into());
    let old_to_new = |doc: i32| -> lucene_search::Result<i32> {
        let old_block_start = if doc == 0 {
            0
        } else {
            parents
                .prev_set_bit(doc as usize - 1)
                .map_or(0, |p| p as i32 + 1)
        };
        let old_block_end = parents.next_set_bit(doc as usize).ok_or_else(unblocked)? as i32;
        let new_block_end = max_doc - 1 - old_block_start;
        Ok(new_block_end - (old_block_end - doc))
    };
    let mut new_to_old = Vec::with_capacity(max_doc as usize);
    for doc in 0..max_doc {
        let old_block_end = parents
            .next_set_bit((max_doc - 1 - doc) as usize)
            .ok_or_else(unblocked)? as i32;
        let new_block_end = old_to_new(old_block_end)?;
        new_to_old.push(old_block_end - (new_block_end - doc));
    }
    DocMap::from_new_to_old(new_to_old)
}
