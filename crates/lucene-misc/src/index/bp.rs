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

use lucene_search::reader::sorting::{DocMap, SortingCodecReader};
use lucene_search::reader::{CodecReader, PostingsFlags, NO_MORE_DOCS};
use lucene_search::{Error, Result};

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
    /// Documents this reorderer cannot renumber and keep whole (a block
    /// without its root): the merge keeps their order.
    Unsupported(String),
    /// Anything else.
    Failed(Error),
}

impl std::fmt::Display for ReorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReorderError::NotEnoughRam(m) | ReorderError::Unsupported(m) => f.write_str(m),
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
    /// A numeric doc-values field only the last document of each block has
    /// (OpenSearch's `_primary_term`, on root documents only), naming the
    /// blocks of a reader without a parent field.
    block_root_field: Option<String>,
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
            block_root_field: None,
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

    /// The field whose numeric doc values mark each document block's last
    /// document when the reader has no parent field (Lucene's reorderer has
    /// none: it reorders blocks only by their parent field). A reader where
    /// every document has a value, or none has, is taken to have no blocks:
    /// the field must be one every block's root carries.
    pub fn set_block_root_field(&mut self, field: Option<String>) {
        self.block_root_field = field;
    }

    /// [`Self::set_block_root_field`]'s field.
    pub fn block_root_field(&self) -> Option<&str> {
        self.block_root_field.as_deref()
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
        self.compute_doc_map_forking_at(reader, FORK_THRESHOLD)
    }

    /// [`Self::compute_doc_map`], forking partitions of at least
    /// `fork_threshold` documents.
    fn compute_doc_map_forking_at(
        &self,
        reader: &dyn CodecReader,
        fork_threshold: usize,
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
        let parents = match (reader.field_infos().parent_field(), &self.block_root_field) {
            (Some(field), _) => Some(parent_bits(reader, field, n)?),
            (None, Some(root)) => root_bits(reader, root, n)?,
            (None, None) => None,
        };
        let ctx = Ctx {
            reorderer: self,
            forward: &forward,
            parents: parents.as_ref(),
            scratch: std::sync::Mutex::new(Vec::new()),
            fork_threshold,
        };
        let mut biases = vec![0f32; n];
        let mut scratch = ctx.take_scratch();
        ctx.run(&mut docs, &mut biases, &mut scratch, 0)?;
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
    /// ids, ascending. The matching terms' postings are read twice -- once
    /// to count each document's terms, once to place them -- so the forward
    /// index is built in its final, flat layout with no per-document vectors
    /// and no copy.
    fn build_forward_index(
        &self,
        reader: &dyn CodecReader,
        fields: &[String],
    ) -> std::result::Result<Forward, ReorderError> {
        let n = usize::try_from(reader.max_doc()).unwrap_or(0);
        let mut counts = vec![0usize; n + 1];
        let num_terms = self.for_each_posting(reader, fields, |_, doc| {
            counts[doc] += 1;
        })?;
        // `starts[d]` is where document `d`'s terms begin; `counts` becomes
        // each document's write cursor.
        let mut starts = Vec::with_capacity(n + 1);
        let mut total = 0usize;
        for c in counts.iter_mut().take(n) {
            starts.push(total);
            let here = *c;
            *c = total;
            total += here;
        }
        starts.push(total);
        let mut term_ids = vec![0u32; total];
        // Terms are visited in ascending id order, so each document's ids
        // land ascending.
        self.for_each_posting(reader, fields, |term_id, doc| {
            term_ids[counts[doc]] = term_id;
            counts[doc] += 1;
        })?;
        Ok(Forward {
            starts,
            term_ids,
            num_terms: num_terms as usize,
        })
    }

    /// Every `(term id, doc)` of the matching terms of `fields`, term ids
    /// assigned in field then term order; how many terms matched.
    fn for_each_posting(
        &self,
        reader: &dyn CodecReader,
        fields: &[String],
        mut visit: impl FnMut(u32, usize),
    ) -> std::result::Result<i64, ReorderError> {
        let max_doc = reader.max_doc();
        let n = usize::try_from(max_doc).unwrap_or(0);
        let threads = rayon::current_num_threads().max(1) as f64;
        let max_num_terms = ((self.ram_budget_mb * 1024.0 * 1024.0 - doc_ram_requirements(max_doc))
            / threads
            / TERM_RAM_PER_THREAD_PER_TERM as f64) as i32;
        let max_doc_freq = (f64::from(self.max_doc_freq) * f64::from(max_doc)) as i32;
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
                    let d = usize::try_from(doc)
                        .ok()
                        .filter(|&d| d < n)
                        .ok_or_else(|| {
                            Error::IllegalState(format!(
                                "document {doc} in postings is out of range"
                            ))
                        })?;
                    visit(term_id, d);
                }
            }
        }
        Ok(num_terms)
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

/// The blocks [`BpIndexReorderer::set_block_root_field`] names: the
/// documents with a value of `field`, `None` when every document has one or
/// none has (no blocks).
///
/// # Errors
/// [`ReorderError::Unsupported`] when the last document has none: a block
/// without its root, which no renumbering can keep whole.
fn root_bits(
    reader: &dyn CodecReader,
    field: &str,
    max_doc: usize,
) -> std::result::Result<Option<FixedBitSet>, ReorderError> {
    let bits = parent_bits(reader, field, max_doc)?;
    let roots = bits.cardinality();
    if roots == max_doc || roots == 0 {
        return Ok(None);
    }
    // FBS: `max_doc - 1 < bits.len()`, checked non-empty first.
    if max_doc == 0 || !bits.get(max_doc - 1) {
        return Err(ReorderError::Unsupported(format!(
            "the last document has no {field}: a block without its root"
        )));
    }
    Ok(Some(bits))
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

/// Partitions at least this long split their two halves across threads, as
/// Java's `FORK_THRESHOLD`.
const FORK_THRESHOLD: usize = 8192;

/// Bias chunks per thread when a large partition computes its biases.
const BIAS_CHUNK: usize = 4096;

/// What every level of the recursion shares.
struct Ctx<'a> {
    reorderer: &'a BpIndexReorderer,
    forward: &'a Forward,
    parents: Option<&'a FixedBitSet>,
    /// Document-frequency arrays no running task holds, for the next fork.
    scratch: std::sync::Mutex<Vec<Scratch>>,
    /// Partitions at least this long fork ([`FORK_THRESHOLD`]).
    fork_threshold: usize,
}

/// One task's per-term document frequencies on either side of its split.
struct Scratch {
    left_freqs: Vec<i32>,
    right_freqs: Vec<i32>,
}

impl Ctx<'_> {
    fn take_scratch(&self) -> Scratch {
        let pooled = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop();
        pooled.unwrap_or_else(|| Scratch {
            left_freqs: vec![0; self.forward.num_terms],
            right_freqs: vec![0; self.forward.num_terms],
        })
    }

    fn give_scratch(&self, scratch: Scratch) {
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(scratch);
    }

    fn next_parent(&self, doc: i32) -> Result<i32> {
        let parents = self.parents.expect("only called with parents");
        parents
            .next_set_bit(doc as usize)
            .map(|p| p as i32)
            .ok_or_else(|| Error::IllegalState(format!("no parent document at or after {doc}")))
    }

    fn doc_freqs(&self, docs: &[i32], freqs: &mut [i32]) {
        freqs.fill(0);
        for &doc in docs {
            for &t in self.forward.terms(doc) {
                freqs[t as usize] += 1;
            }
        }
    }

    /// `IndexReorderingTask.call` over `docs`, with `biases` the same length.
    fn run(
        &self,
        docs: &mut [i32],
        biases: &mut [f32],
        scratch: &mut Scratch,
        depth: u32,
    ) -> Result<()> {
        if depth > 0 {
            docs.sort_unstable();
        }
        let len = docs.len();
        let half = len / 2;
        if (half as i64) < i64::from(self.reorderer.min_partition_size) {
            return Ok(());
        }
        let mid = half;

        {
            let (left, right) = docs.split_at(mid);
            let Scratch {
                left_freqs,
                right_freqs,
            } = scratch;
            if len >= self.fork_threshold {
                rayon::join(
                    || self.doc_freqs(left, left_freqs),
                    || self.doc_freqs(right, right_freqs),
                );
            } else {
                self.doc_freqs(left, left_freqs);
                self.doc_freqs(right, right_freqs);
            }
        }

        for iter in 0..self.reorderer.max_iters {
            if !self.shuffle(docs, biases, scratch, mid, iter)? {
                break;
            }
        }

        let mut split = mid;
        if let Some(parents) = self.parents {
            // Make sure we split just after a parent doc.
            let last_left = i64::from(docs[mid - 1]);
            let next = i64::from(self.next_parent(docs[mid - 1])?);
            let mut at = mid as i64 + next - last_left;
            if at == len as i64 {
                // No good split on the right side, look on the left side then.
                let prev = parents
                    .prev_set_bit(last_left as usize)
                    .map_or(-1, |p| p as i64);
                at = mid as i64 - (last_left - prev);
                if at == 0 {
                    // A single parent document: no reordering is possible.
                    return Ok(());
                }
            }
            split = at as usize;
        }

        let (left_docs, right_docs) = docs.split_at_mut(split);
        let (left_biases, right_biases) = biases.split_at_mut(split);
        if len >= self.fork_threshold {
            let (l, r) = rayon::join(
                || self.run(left_docs, left_biases, scratch, depth + 1),
                || {
                    let mut own = self.take_scratch();
                    let r = self.run(right_docs, right_biases, &mut own, depth + 1);
                    self.give_scratch(own);
                    r
                },
            );
            l.and(r)
        } else {
            self.run(left_docs, left_biases, scratch, depth + 1)?;
            self.run(right_docs, right_biases, scratch, depth + 1)
        }
    }

    /// `computeBias`: negative when a document is attracted to the left.
    fn bias(&self, scratch: &Scratch, doc: i32) -> f32 {
        let mut bias = 0f64;
        for &t in self.forward.terms(doc) {
            let from = scratch.left_freqs[t as usize];
            let to = scratch.right_freqs[t as usize];
            let a: f32 = if to == 0 { 0.0 } else { fast_log2(to) };
            let b: f32 = if from == 0 { 0.0 } else { fast_log2(from) };
            bias += f64::from(a - b);
        }
        bias as f32
    }

    /// `shuffle`: swaps documents across the halves at `mid`; whether any
    /// moved.
    fn shuffle(
        &self,
        docs: &mut [i32],
        biases: &mut [f32],
        scratch: &mut Scratch,
        mid: usize,
        iter: i32,
    ) -> Result<bool> {
        if docs.len() >= self.fork_threshold {
            use rayon::prelude::*;
            let shared = &*scratch;
            biases
                .par_chunks_mut(BIAS_CHUNK)
                .zip(docs.par_chunks(BIAS_CHUNK))
                .for_each(|(b, d)| {
                    for (slot, &doc) in b.iter_mut().zip(d) {
                        *slot = self.bias(shared, doc);
                    }
                });
        } else {
            for (slot, &doc) in biases.iter_mut().zip(docs.iter()) {
                *slot = self.bias(scratch, doc);
            }
        }

        if self.parents.is_some() {
            let mut i = 0;
            while i < docs.len() {
                let first_child = docs[i];
                let num_children = (self.next_parent(first_child)? - first_child) as usize;
                let mut cumulative = 0f64;
                for j in 0..=num_children {
                    cumulative += f64::from(biases[i + j]);
                }
                // Every document of a block gets the block's summed bias, so the
                // sort below keeps the block together.
                biases[i..=i + num_children].fill(cumulative as f32);
                i += num_children + 1;
            }
        }

        let max_left = biases[..mid]
            .iter()
            .fold(f32::NEG_INFINITY, |m, &b| m.max(b));
        let min_right = biases[mid..].iter().fold(f32::INFINITY, |m, &b| m.min(b));
        let gain = max_left - min_right;
        // Simulated annealing: the gain of swapping the left document most
        // attracted to the right with the right one most attracted to the left
        // is compared against `iter` rather than zero.
        if gain <= iter as f32 {
            return Ok(false);
        }

        let len = docs.len();
        let mut selector = Selector {
            forward: self.forward,
            docs,
            biases,
            scratch,
            mid,
            pivot_doc: 0,
            pivot_bias: 0.0,
        };
        if self.parents.is_none() {
            let mut random = SplittableRandom::new(0);
            intro_select(&mut selector, 0, len, mid, &mut random);
        } else {
            // With parents a full sort is needed so the block structure holds.
            intro_sort(&mut selector, 0, len);
        }
        Ok(true)
    }
}

/// `shuffle`'s `Selector`: orders `docs` by bias, ties by doc id, keeping
/// both halves' document frequencies current when a swap crosses the
/// midpoint.
struct Selector<'s> {
    forward: &'s Forward,
    docs: &'s mut [i32],
    biases: &'s mut [f32],
    scratch: &'s mut Scratch,
    mid: usize,
    pivot_doc: i32,
    pivot_bias: f32,
}

impl IntroTarget for Selector<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.biases.swap(i, j);
        if (i < self.mid) == (j < self.mid) {
            self.docs.swap(i, j);
        } else {
            // A swap across the halves: keep the cached doc freqs up to date,
            // which is much cheaper than recomputing them next round.
            let (left, right) = (i.min(j), i.max(j));
            let left_doc = self.docs[left];
            let right_doc = self.docs[right];
            let Scratch {
                left_freqs,
                right_freqs,
            } = &mut *self.scratch;
            for &term in self.forward.terms(left_doc) {
                left_freqs[term as usize] -= 1;
                right_freqs[term as usize] += 1;
            }
            for &term in self.forward.terms(right_doc) {
                left_freqs[term as usize] += 1;
                right_freqs[term as usize] -= 1;
            }
            self.docs[left] = right_doc;
            self.docs[right] = left_doc;
        }
    }

    fn set_pivot(&mut self, i: usize) {
        self.pivot_doc = self.docs[i];
        self.pivot_bias = self.biases[i];
    }

    fn compare_pivot(&mut self, j: usize) -> i32 {
        // `Float.compare`, then the doc id to keep doc id order where it can.
        let cmp = self.pivot_bias.total_cmp(&self.biases[j]) as i32;
        if cmp == 0 {
            self.pivot_doc.wrapping_sub(self.docs[j])
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

    /// Forking changes nothing: over an index big enough to fork at every
    /// level, the doc map is the one the recursion computes on one thread.
    #[test]
    fn forked_doc_maps_equal_sequential_ones() {
        use lucene_codecs::field_infos::FieldInfo;
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_index::segment_info::LuceneVersion;
        use lucene_search::directory_reader::DirectoryReader;
        use lucene_store::FsDirectory;
        use lucene_util::test_support::TempDir;

        let tmp = TempDir::new("bp-fork");
        let dir = FsDirectory::open(tmp.path());
        let fields = vec![
            FieldInfo {
                index_options: IndexOptions::Docs,
                omit_norms: true,
                ..FieldInfo::new("id", 0)
            },
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                ..FieldInfo::new("body", 1)
            },
        ];
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, fields, "Lucene104", version).unwrap();
        w.set_postings_field(Some("id")).unwrap();
        w.add_postings_field("body").unwrap();
        let mut x: u64 = 17;
        let mut next = || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 33) as usize
        };
        for i in 0..20_000 {
            let topic = next() % 16;
            let words: Vec<String> = (0..6)
                .map(|_| {
                    let t = if next() % 5 == 0 {
                        (topic + 1) % 16
                    } else {
                        topic
                    };
                    format!("w{t}_{}", next() % 8)
                })
                .collect();
            w.add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("d{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(words.join(" ")),
                    },
                ],
            })
            .unwrap();
        }
        w.commit().unwrap();
        w.close().unwrap();
        let reader = DirectoryReader::open(&dir).unwrap();
        let leaf = &reader.segment_readers()[0];
        assert_eq!(leaf.max_doc, 20_000);
        let mut r = BpIndexReorderer::new();
        r.set_min_doc_freq(2).unwrap();
        r.set_fields(Some(vec!["body".to_string()]));
        let order = |map: DocMap| {
            (0..map.size())
                .map(|d| map.new_to_old(d))
                .collect::<Vec<_>>()
        };
        let forked = order(r.compute_doc_map(leaf).unwrap());
        let sequential = order(r.compute_doc_map_forking_at(leaf, usize::MAX).unwrap());
        assert_eq!(forked, sequential);
        assert_ne!(
            forked,
            (0..20_000).collect::<Vec<_>>(),
            "nothing was reordered"
        );
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
