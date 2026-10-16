//! `BPIndexReorderer`: recursive graph bisection, an assignment of document
//! ids that lowers the sum of the log gaps between consecutive postings (and
//! tends to speed queries up by putting documents that share terms side by
//! side). Dhulipala et al., "Compressing graphs and inverted indexes with
//! recursive graph bisection", with Mackenzie et al.'s simulated-annealing
//! stop rule.
//!
//! Each partition is split in two halves; documents are then swapped across
//! the halves for up to `max_iters` rounds, each swap chosen by the document's
//! *bias* -- how much more it would gain from the other half's term
//! frequencies -- until no swap helps; both halves are then reordered the same
//! way.

use std::sync::{Arc, OnceLock};

use lucene_codecs::field_infos::IndexOptions;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::sorter::{intro_select, intro_sort, IntroTarget};
use lucene_util::splittable_random::SplittableRandom;

use crate::reader::sorting::{DocMap, SortingCodecReader};
use crate::reader::{CodecReader, PostingsFlags, NO_MORE_DOCS};
use crate::{Error, Result};

/// `DEFAULT_MIN_DOC_FREQ`.
pub const DEFAULT_MIN_DOC_FREQ: i32 = 4096;
/// `DEFAULT_MIN_PARTITION_SIZE`.
pub const DEFAULT_MIN_PARTITION_SIZE: i32 = 32;
/// `DEFAULT_MAX_ITERS`.
pub const DEFAULT_MAX_ITERS: i32 = 20;
/// The RAM budget a new reorderer gets. Java takes 10% of the JVM's heap;
/// there is no heap to ask here.
pub const DEFAULT_RAM_BUDGET_MB: f64 = 256.0;

/// Why a reordering failed: `NotEnoughRAMException` apart from every other
/// failure, because [`super::bp_merge_policy::BpReorderingMergePolicy`] skips
/// the first and fails the merge on the second.
#[derive(Debug)]
pub enum ReorderError {
    /// `BPIndexReorderer.NotEnoughRAMException`.
    NotEnoughRam(String),
    /// Anything else.
    Failed(Error),
}

impl std::fmt::Display for ReorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReorderError::NotEnoughRam(m) => f.write_str(m),
            ReorderError::Failed(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ReorderError {}

impl From<Error> for ReorderError {
    fn from(e: Error) -> Self {
        ReorderError::Failed(e)
    }
}

/// `ArrayUtil.MAX_ARRAY_LENGTH`.
const MAX_ARRAY_LENGTH: i64 = 2_147_483_639;

/// `BPIndexReorderer`.
#[derive(Debug, Clone)]
pub struct BpIndexReorderer {
    min_doc_freq: i32,
    max_doc_freq: f32,
    min_partition_size: i32,
    max_iters: i32,
    ram_budget_mb: f64,
    fields: Option<Vec<String>>,
}

impl Default for BpIndexReorderer {
    fn default() -> Self {
        Self {
            min_doc_freq: DEFAULT_MIN_DOC_FREQ,
            max_doc_freq: 1.0,
            min_partition_size: DEFAULT_MIN_PARTITION_SIZE,
            max_iters: DEFAULT_MAX_ITERS,
            ram_budget_mb: DEFAULT_RAM_BUDGET_MB,
            fields: None,
        }
    }
}

impl BpIndexReorderer {
    /// `new BPIndexReorderer()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `setMinDocFreq`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] below 1.
    pub fn set_min_doc_freq(&mut self, min_doc_freq: i32) -> Result<()> {
        if min_doc_freq < 1 {
            return Err(Error::IllegalArgument(format!(
                "minDocFreq must be at least 1, got {min_doc_freq}"
            )));
        }
        self.min_doc_freq = min_doc_freq;
        Ok(())
    }

    /// `setMaxDocFreq`: a ratio of `maxDoc` in (0, 1].
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] outside that range (including NaN).
    pub fn set_max_doc_freq(&mut self, max_doc_freq: f32) -> Result<()> {
        if !(max_doc_freq > 0.0 && max_doc_freq <= 1.0) {
            return Err(Error::IllegalArgument(format!(
                "maxDocFreq must be in (0, 1], got {max_doc_freq:?}"
            )));
        }
        self.max_doc_freq = max_doc_freq;
        Ok(())
    }

    /// `setMinPartitionSize`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] below 1.
    pub fn set_min_partition_size(&mut self, min_partition_size: i32) -> Result<()> {
        if min_partition_size < 1 {
            return Err(Error::IllegalArgument(format!(
                "minPartitionSize must be at least 1, got {min_partition_size}"
            )));
        }
        self.min_partition_size = min_partition_size;
        Ok(())
    }

    /// `setMaxIters`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] below 1.
    pub fn set_max_iters(&mut self, max_iters: i32) -> Result<()> {
        if max_iters < 1 {
            return Err(Error::IllegalArgument(format!(
                "maxIters must be at least 1, got {max_iters}"
            )));
        }
        self.max_iters = max_iters;
        Ok(())
    }

    /// `setRAMBudgetMB`.
    pub fn set_ram_budget_mb(&mut self, ram_budget_mb: f64) {
        self.ram_budget_mb = ram_budget_mb;
    }

    /// `setFields`: `None` partitions on every indexed field.
    pub fn set_fields(&mut self, fields: Option<Vec<String>>) {
        self.fields = fields;
    }

    /// `computeDocMap(reader, tempDir, executor)`: the new numbering. The
    /// forward index lives in memory, so there is no temp directory, and the
    /// recursion runs on the calling thread.
    ///
    /// # Errors
    /// [`ReorderError::NotEnoughRam`] when the RAM budget cannot hold the
    /// document metadata or the matching terms; [`Error::IllegalArgument`]
    /// for more terms than an array can hold; and read errors.
    pub fn compute_doc_map(
        &self,
        reader: &dyn CodecReader,
    ) -> std::result::Result<DocMap, ReorderError> {
        let max_doc = reader.max_doc();
        let doc_ram = doc_ram_requirements(max_doc);
        if doc_ram >= self.ram_budget_mb * 1024.0 * 1024.0 {
            return Err(ReorderError::NotEnoughRam(format!(
                "At least {:?}MB of RAM are required to hold metadata about documents in RAM, but current RAM budget is {:?}MB",
                (doc_ram / 1024.0 / 1024.0).ceil(),
                self.ram_budget_mb
            )));
        }
        let fields: Vec<String> = match &self.fields {
            Some(f) => f.clone(),
            None => reader
                .field_infos()
                .fields
                .iter()
                .filter(|fi| fi.index_options != IndexOptions::None)
                .map(|fi| fi.name.clone())
                .collect(),
        };
        let forward = self.build_forward_index(reader, &fields)?;
        let n = usize::try_from(max_doc).unwrap_or(0);
        let mut docs: Vec<i32> = (0..max_doc).collect();
        let parents = match reader.field_infos().parent_field() {
            Some(field) => Some(parent_bits(reader, field, n)?),
            None => None,
        };
        let mut task = Task {
            reorderer: self,
            forward: &forward,
            left_freqs: vec![0; forward.num_terms],
            right_freqs: vec![0; forward.num_terms],
            biases: vec![0.0; n],
            parents: parents.as_ref(),
            docs: &mut docs,
        };
        task.run(0, n, 0)?;
        Ok(DocMap::from_new_to_old(docs)?)
    }

    /// `reorder(reader, tempDir, executor)`: `reader` seen in the new order,
    /// for `IndexWriter.addIndexes`.
    ///
    /// # Errors
    /// What [`Self::compute_doc_map`] reports.
    pub fn reorder(
        &self,
        reader: Arc<dyn CodecReader>,
    ) -> std::result::Result<SortingCodecReader, ReorderError> {
        let map = self.compute_doc_map(reader.as_ref())?;
        Ok(SortingCodecReader::wrap(reader, Some(map), Vec::new())?)
    }

    /// `writePostings` and `buildForwardIndex`: each document's matching term
    /// ids, ascending.
    fn build_forward_index(
        &self,
        reader: &dyn CodecReader,
        fields: &[String],
    ) -> std::result::Result<Forward, ReorderError> {
        let max_doc = reader.max_doc();
        let n = usize::try_from(max_doc).unwrap_or(0);
        let max_num_terms = ((self.ram_budget_mb * 1024.0 * 1024.0
            - doc_ram_requirements(max_doc))
            / 1.0
            / TERM_RAM_PER_THREAD_PER_TERM as f64) as i32;
        let max_doc_freq = (f64::from(self.max_doc_freq) * f64::from(max_doc)) as i32;
        let mut per_doc: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut num_terms: i64 = 0;
        for field in fields {
            let Some(terms) = reader.terms(field)? else {
                continue;
            };
            if terms.size() != -1 {
                // Skip ID-like fields that have many terms where none is of interest.
                let max_possible_doc_freq = 1 + terms.sum_doc_freq() - terms.size();
                if max_possible_doc_freq < i64::from(self.min_doc_freq) {
                    continue;
                }
            }
            let mut iterator = terms.iterator()?;
            while iterator.next()?.is_some() {
                let doc_freq = iterator.doc_freq()?;
                if doc_freq < self.min_doc_freq || doc_freq > max_doc_freq {
                    continue;
                }
                if num_terms >= MAX_ARRAY_LENGTH {
                    return Err(Error::IllegalArgument(format!(
                        "Cannot perform recursive graph bisection on more than {MAX_ARRAY_LENGTH} terms, the maximum array length"
                    ))
                    .into());
                } else if num_terms >= i64::from(max_num_terms) {
                    return Err(ReorderError::NotEnoughRam(format!(
                        "Too many terms are matching given the RAM budget of {:?}MB. Consider raising the RAM budget, raising the minimum doc frequency, or decreasing concurrency",
                        self.ram_budget_mb
                    )));
                }
                let term_id = num_terms as u32;
                num_terms += 1;
                let mut postings = iterator.postings(PostingsFlags::None)?;
                loop {
                    let doc = postings.next_doc()?;
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    per_doc[usize::try_from(doc).map_err(|_| {
                        Error::IllegalState(format!("negative document {doc} in postings"))
                    })?]
                    .push(term_id);
                }
            }
        }
        let mut starts = Vec::with_capacity(n + 1);
        let mut term_ids = Vec::new();
        for terms in &per_doc {
            starts.push(term_ids.len());
            term_ids.extend_from_slice(terms);
        }
        starts.push(term_ids.len());
        Ok(Forward {
            starts,
            term_ids,
            num_terms: num_terms as usize,
        })
    }
}

/// `docRAMRequirements`: an int per doc for the doc map and a float for its
/// bias.
fn doc_ram_requirements(max_doc: i32) -> f64 {
    8.0 * f64::from(max_doc)
}

/// `termRAMRequirementsPerThreadPerTerm`: two ints per term.
const TERM_RAM_PER_THREAD_PER_TERM: i64 = 8;

/// `BitSet.of(DocValues.getNumeric(reader, parentField), maxDoc)`.
fn parent_bits(reader: &dyn CodecReader, field: &str, max_doc: usize) -> Result<FixedBitSet> {
    let mut bits = FixedBitSet::new(max_doc);
    if let Some(mut values) = reader.numeric_doc_values(field)? {
        loop {
            let doc = values.next_doc()?;
            if doc == NO_MORE_DOCS {
                break;
            }
            // FBS: `bits` is sized to `max_doc`, and doc values iterate only documents below it.
            bits.set(doc as usize);
        }
    }
    Ok(bits)
}

/// `ForwardIndex`: each document's term ids, ascending.
struct Forward {
    starts: Vec<usize>,
    term_ids: Vec<u32>,
    num_terms: usize,
}

impl Forward {
    fn terms(&self, doc: i32) -> &[u32] {
        let d = doc as usize;
        &self.term_ids[self.starts[d]..self.starts[d + 1]]
    }
}

fn log2_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0f32; 256];
        let one = 1f32.to_bits();
        for (i, slot) in table.iter_mut().enumerate() {
            let f = f32::from_bits(one | ((i as u32) << (23 - 8)));
            *slot = (f64::from(f).ln() / 2f64.ln()) as f32;
        }
        table
    })
}

/// `fastLog2`: an approximate base-2 logarithm that trades accuracy for speed.
///
/// `i` must be positive.
pub fn fast_log2(i: i32) -> f32 {
    debug_assert!(i > 0, "Cannot compute log of i={i}");
    let floor_log2 = 31 - i.leading_zeros() as i32;
    // The first 8 mantissa bits of `i` as a float, without the implicit one.
    let table_index = ((i as u32).wrapping_shl((32 - floor_log2) as u32)) >> (32 - 8);
    floor_log2 as f32 + log2_table()[table_index as usize]
}

/// One level of the recursion over `docs[lo..hi]`.
struct Task<'a> {
    reorderer: &'a BpIndexReorderer,
    forward: &'a Forward,
    left_freqs: Vec<i32>,
    right_freqs: Vec<i32>,
    biases: Vec<f32>,
    parents: Option<&'a FixedBitSet>,
    docs: &'a mut Vec<i32>,
}

impl Task<'_> {
    fn next_parent(&self, doc: i32) -> Result<i32> {
        let parents = self.parents.expect("only called with parents");
        parents
            .next_set_bit(doc as usize)
            .map(|p| p as i32)
            .ok_or_else(|| Error::IllegalState(format!("no parent document at or after {doc}")))
    }

    fn doc_freqs(&self, from: usize, to: usize, freqs: &mut [i32]) {
        freqs.fill(0);
        for &doc in &self.docs[from..to] {
            for &t in self.forward.terms(doc) {
                freqs[t as usize] += 1;
            }
        }
    }

    /// `IndexReorderingTask.call`.
    fn run(&mut self, lo: usize, hi: usize, depth: u32) -> Result<()> {
        if depth > 0 {
            self.docs[lo..hi].sort_unstable();
        }
        let len = hi - lo;
        let half = len / 2;
        if (half as i64) < i64::from(self.reorderer.min_partition_size) {
            return Ok(());
        }
        let mid = lo + half;

        let mut left_freqs = std::mem::take(&mut self.left_freqs);
        let mut right_freqs = std::mem::take(&mut self.right_freqs);
        self.doc_freqs(lo, mid, &mut left_freqs);
        self.doc_freqs(mid, hi, &mut right_freqs);
        self.left_freqs = left_freqs;
        self.right_freqs = right_freqs;

        for iter in 0..self.reorderer.max_iters {
            if !self.shuffle(lo, hi, mid, iter)? {
                break;
            }
        }

        let (mut left, mut right) = ((lo, mid), (mid, hi));
        if let Some(parents) = self.parents {
            // Make sure we split just after a parent doc.
            let last_left = i64::from(self.docs[mid - 1]);
            let next = i64::from(self.next_parent(self.docs[mid - 1])?);
            let mut split = mid as i64 + next - last_left;
            if split == hi as i64 {
                // No good split on the right side, look on the left side then.
                let prev = parents
                    .prev_set_bit(last_left as usize)
                    .map_or(-1, |p| p as i64);
                split = mid as i64 - (last_left - prev);
                if split == lo as i64 {
                    // A single parent document: no reordering is possible.
                    return Ok(());
                }
            }
            let split = split as usize;
            left = (lo, split);
            right = (split, hi);
        }

        self.run(left.0, left.1, depth + 1)?;
        self.run(right.0, right.1, depth + 1)
    }

    /// `computeBias`: negative when a document is attracted to the left.
    fn bias(&self, doc: i32) -> f32 {
        let mut bias = 0f64;
        for &t in self.forward.terms(doc) {
            let from = self.left_freqs[t as usize];
            let to = self.right_freqs[t as usize];
            let a: f32 = if to == 0 { 0.0 } else { fast_log2(to) };
            let b: f32 = if from == 0 { 0.0 } else { fast_log2(from) };
            bias += f64::from(a - b);
        }
        bias as f32
    }

    /// `shuffle`: swaps documents across the halves; whether any moved.
    fn shuffle(&mut self, lo: usize, hi: usize, mid: usize, iter: i32) -> Result<bool> {
        for i in lo..hi {
            self.biases[i] = self.bias(self.docs[i]);
        }

        if self.parents.is_some() {
            let mut i = lo;
            while i < hi {
                let first_child = self.docs[i];
                let num_children = (self.next_parent(first_child)? - first_child) as usize;
                let mut cumulative = 0f64;
                for j in 0..=num_children {
                    cumulative += f64::from(self.biases[i + j]);
                }
                // Every document of a block gets the block's summed bias, so the
                // sort below keeps the block together.
                self.biases[i..=i + num_children].fill(cumulative as f32);
                i += num_children + 1;
            }
        }

        let max_left = self.biases[lo..mid]
            .iter()
            .fold(f32::NEG_INFINITY, |m, &b| m.max(b));
        let min_right = self.biases[mid..hi]
            .iter()
            .fold(f32::INFINITY, |m, &b| m.min(b));
        let gain = max_left - min_right;
        // Simulated annealing: the gain of swapping the left document most
        // attracted to the right with the right one most attracted to the left
        // is compared against `iter` rather than zero.
        if gain <= iter as f32 {
            return Ok(false);
        }

        let mut selector = Selector {
            task: self,
            mid,
            pivot_doc: 0,
            pivot_bias: 0.0,
        };
        if selector.task.parents.is_none() {
            let mut random = SplittableRandom::new(0);
            intro_select(&mut selector, lo, hi, mid, &mut random);
        } else {
            // With parents a full sort is needed so the block structure holds.
            intro_sort(&mut selector, lo, hi);
        }
        Ok(true)
    }
}

/// `shuffle`'s `Selector`: orders `docs[lo..hi]` by bias, ties by doc id,
/// keeping both halves' document frequencies current when a swap crosses
/// the midpoint.
struct Selector<'t, 'a> {
    task: &'t mut Task<'a>,
    mid: usize,
    pivot_doc: i32,
    pivot_bias: f32,
}

impl IntroTarget for Selector<'_, '_> {
    fn swap(&mut self, i: usize, j: usize) {
        let t = &mut *self.task;
        t.biases.swap(i, j);
        if (i < self.mid) == (j < self.mid) {
            t.docs.swap(i, j);
        } else {
            // A swap across the halves: keep the cached doc freqs up to date,
            // which is much cheaper than recomputing them next round.
            let (left, right) = (i.min(j), i.max(j));
            let left_doc = t.docs[left];
            let right_doc = t.docs[right];
            for &term in t.forward.terms(left_doc) {
                t.left_freqs[term as usize] -= 1;
                t.right_freqs[term as usize] += 1;
            }
            for &term in t.forward.terms(right_doc) {
                t.left_freqs[term as usize] += 1;
                t.right_freqs[term as usize] -= 1;
            }
            t.docs[left] = right_doc;
            t.docs[right] = left_doc;
        }
    }

    fn set_pivot(&mut self, i: usize) {
        self.pivot_doc = self.task.docs[i];
        self.pivot_bias = self.task.biases[i];
    }

    fn compare_pivot(&mut self, j: usize) -> i32 {
        // `Float.compare`, then the doc id to keep doc id order where it can.
        let cmp = self.pivot_bias.total_cmp(&self.task.biases[j]) as i32;
        if cmp == 0 {
            self.pivot_doc.wrapping_sub(self.task.docs[j])
        } else {
            cmp
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log2_tracks_log2() {
        assert_eq!(fast_log2(1), 0.0);
        assert_eq!(fast_log2(2), 1.0);
        assert_eq!(fast_log2(1024), 10.0);
        for i in [3, 5, 7, 100, 4097, 1_000_000, i32::MAX] {
            let want = (f64::from(i)).log2() as f32;
            assert!((fast_log2(i) - want).abs() < 0.01, "{i}");
        }
    }

    #[test]
    fn setters_refuse_out_of_range_values() {
        let mut r = BpIndexReorderer::new();
        assert!(r.set_min_doc_freq(0).is_err());
        assert!(r.set_max_doc_freq(0.0).is_err());
        assert!(r.set_max_doc_freq(1.5).is_err());
        assert!(r.set_max_doc_freq(f32::NAN).is_err());
        assert!(r.set_min_partition_size(0).is_err());
        assert!(r.set_max_iters(0).is_err());
        assert!(r.set_min_doc_freq(1).is_ok());
        assert!(r.set_max_doc_freq(1.0).is_ok());
        assert!(r.set_min_partition_size(1).is_ok());
        assert!(r.set_max_iters(1).is_ok());
    }
}
