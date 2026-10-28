//! `org.apache.lucene.search.suggest.fst.FSTCompletion`, `FSTCompletionBuilder`
//! and `FSTCompletionLookup`, with the `BytesRefSorter` they sort through
//! (M13 T13.3b): an automaton over the suggestion strings whose root arcs
//! are the weight buckets.
//!
//! Departures from Java, all forced by Rust or by this port's in-memory
//! sorting (see [`super::input`]):
//!
//! * `ExternalRefSorter`/`OfflineSorter` sort on disk; the sort here is in
//!   memory with the same order (unsigned bytes), so the FST built is the same
//!   one. [`BytesRefSorter`] stays a trait so a caller can supply its own.
//! * `FSTCompletion.lookup(CharSequence)` returns a lazy `Stream`; here
//!   [`FstCompletion::lookup_all`] collects it and [`FstCompletion::lookup`]
//!   stops the traversal at `num`.
//! * `FSTCompletionLookup.load` reads the rest of its input for the FST (Java
//!   reads exactly the FST), so the FST must be the last thing in the stream.
//! * The `Directory`/`tempFileNamePrefix` constructor arguments are gone.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_codecs::fst::InputType;
use lucene_codecs::fst_compiler::{FstCompilerBuilder, FstReadError, NoOutputs};
use lucene_codecs::fst_util::{to_ints_ref, Arc as FstArc, TypedFst, END_LABEL};
use lucene_store::{DataInput, DataOutput, Error, Result};

use super::input::InputIterator;
use super::lookup::{illegal_state, unsupported_operation, Lookup, LookupResult};

/// The FST type both completions share: an automaton, no outputs.
pub type Automaton = TypedFst<NoOutputs>;

fn fst_err(e: FstReadError) -> Error {
    Error::Corrupted(e.0)
}

fn compile_err(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// `BytesRefSorter`: collects byte strings, then hands them back sorted.
pub trait BytesRefSorter {
    /// `add(utf8)`.
    fn add(&mut self, utf8: &[u8]) -> Result<()>;
    /// `iterator()`: every string added, in the sorter's order. Adding after
    /// this is an error.
    fn iterator(&mut self) -> Result<Vec<Vec<u8>>>;
}

/// The byte-string comparator a sorter holds (`Comparator<BytesRef>` in Java).
type Comparator = Box<dyn Fn(&[u8], &[u8]) -> Ordering>;

/// `InMemorySorter`.
pub struct InMemorySorter {
    buffer: Vec<Vec<u8>>,
    closed: bool,
    comparator: Comparator,
}

impl InMemorySorter {
    /// `new InMemorySorter(comparator)`.
    pub fn new(comparator: impl Fn(&[u8], &[u8]) -> Ordering + 'static) -> Self {
        Self {
            buffer: Vec::new(),
            closed: false,
            comparator: Box::new(comparator),
        }
    }

    /// `new InMemorySorter(Comparator.naturalOrder())`: unsigned bytes.
    pub fn natural() -> Self {
        Self::new(<[u8]>::cmp)
    }
}

impl BytesRefSorter for InMemorySorter {
    fn add(&mut self, utf8: &[u8]) -> Result<()> {
        if self.closed {
            return Err(illegal_state(""));
        }
        self.buffer.push(utf8.to_vec());
        Ok(())
    }

    fn iterator(&mut self) -> Result<Vec<Vec<u8>>> {
        self.closed = true;
        let mut out = std::mem::take(&mut self.buffer);
        out.sort_by(|a, b| (self.comparator)(a, b));
        Ok(out)
    }
}

/// `FSTCompletion.Completion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// `utf8`.
    pub utf8: Vec<u8>,
    /// `bucket`.
    pub bucket: i32,
}

impl Completion {
    /// `toString()`: `utf8/bucket`.
    pub fn to_display(&self) -> String {
        format!("{}/{}", String::from_utf8_lossy(&self.utf8), self.bucket)
    }
}

impl PartialOrd for Completion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `compareTo`: by `utf8` alone.
impl Ord for Completion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.utf8.cmp(&other.utf8)
    }
}

/// `FSTCompletion.DEFAULT_BUCKETS`.
pub const DEFAULT_BUCKETS: i32 = 10;

/// `FSTCompletion`.
#[derive(Debug, Clone)]
pub struct FstCompletion {
    automaton: Option<Arc<Automaton>>,
    root_arcs: Vec<FstArc<()>>,
    exact_first: bool,
    higher_weights_first: bool,
}

impl FstCompletion {
    /// `new FSTCompletion(automaton, higherWeightsFirst, exactFirst)`.
    pub fn new(
        automaton: Option<Arc<Automaton>>,
        higher_weights_first: bool,
        exact_first: bool,
    ) -> Result<Self> {
        let root_arcs = match &automaton {
            Some(fst) => Self::cache_root_arcs(fst)?,
            None => Vec::new(),
        };
        Ok(Self {
            automaton,
            root_arcs,
            exact_first,
            higher_weights_first,
        })
    }

    /// `new FSTCompletion(automaton)`: higher weights first, exact first.
    pub fn with_defaults(automaton: Option<Arc<Automaton>>) -> Result<Self> {
        Self::new(automaton, true, true)
    }

    fn cache_root_arcs(fst: &Automaton) -> Result<Vec<FstArc<()>>> {
        let first = fst.first_arc();
        let mut arc = first.clone();
        let mut reader = fst.bytes_reader();
        fst.read_first_target_arc(&first, &mut arc, &mut reader)
            .map_err(fst_err)?;
        let mut arcs = Vec::new();
        loop {
            arcs.push(arc.clone());
            if arc.is_last() {
                break;
            }
            fst.read_next_arc(&mut arc, &mut reader).map_err(fst_err)?;
        }
        // we want highest weights first.
        arcs.reverse();
        Ok(arcs)
    }

    fn fst(&self) -> &Automaton {
        self.automaton.as_deref().expect("checked by the caller")
    }

    /// `descendWithPrefix(arc, utf8)`.
    fn descend_with_prefix(&self, arc: &mut FstArc<()>, utf8: &[u8]) -> Result<bool> {
        let fst = self.fst();
        let mut reader = fst.bytes_reader();
        for &b in utf8 {
            match fst
                .find_target_arc(i32::from(b), arc, &mut reader)
                .map_err(fst_err)?
            {
                Some(next) => *arc = next,
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// `getExactMatchStartingFromRootArc`: the bucket of `utf8` as a whole
    /// completion, or `-1`.
    // SENTINEL: `-1` = `utf8` is not a whole completion under any root arc.
    fn exact_match_from_root_arc(&self, mut index: usize, utf8: &[u8]) -> Result<i32> {
        let fst = self.fst();
        let mut reader = fst.bytes_reader();
        while index < self.root_arcs.len() {
            let root = &self.root_arcs[index];
            let mut arc = root.clone();
            if self.descend_with_prefix(&mut arc, utf8)? {
                let follow = arc.clone();
                fst.read_first_target_arc(&follow, &mut arc, &mut reader)
                    .map_err(fst_err)?;
                if arc.label() == END_LABEL {
                    // Normalize prefix-encoded weight.
                    return Ok(root.label());
                }
            }
            index += 1;
        }
        Ok(-1)
    }

    /// `lookup(key, num)`.
    pub fn lookup(&self, key: &str, num: usize) -> Result<Vec<Completion>> {
        if key.is_empty() || self.automaton.is_none() || num == 0 {
            return Ok(Vec::new());
        }
        if !self.higher_weights_first && self.root_arcs.len() > 1 {
            // An optimal strategy for alphabetically sorted suggestions would
            // be to add them with a constant weight.
            let mut all = self.lookup_all(key)?;
            all.sort();
            all.truncate(num);
            return Ok(all);
        }
        let mut out = Vec::new();
        self.visit(key.as_bytes(), &mut |c| {
            out.push(c);
            out.len() < num
        })?;
        Ok(out)
    }

    /// `lookup(key)`: the whole stream, collected.
    pub fn lookup_all(&self, key: &str) -> Result<Vec<Completion>> {
        if key.is_empty() || self.automaton.is_none() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        self.visit(key.as_bytes(), &mut |c| {
            out.push(c);
            true
        })?;
        Ok(out)
    }

    /// `lookupSortedByWeight`: feeds `f` each completion, best bucket first,
    /// until it returns `false`.
    fn visit(&self, key: &[u8], f: &mut dyn FnMut(Completion) -> bool) -> Result<()> {
        // Look for an exact match first.
        let mut exact: Option<Completion> = None;
        if self.exact_first {
            // Java asks from every root arc in turn, but the search from arc
            // 0 already walks them all and stops at the first exact match:
            // its answer is the loop's, and a miss is a miss from every arc
            let bucket = self.exact_match_from_root_arc(0, key)?;
            if bucket != -1 {
                // root arcs are sorted by decreasing weight so any first
                // exact match will always win.
                exact = Some(Completion {
                    utf8: key.to_vec(),
                    bucket,
                });
            }
        }
        if let Some(e) = &exact {
            if !f(e.clone()) {
                return Ok(());
            }
        }
        for root in &self.root_arcs {
            let mut arc = root.clone();
            if !self.descend_with_prefix(&mut arc, key)? {
                continue;
            }
            // A subgraph starting from the current node has the completions of
            // the key prefix. The arc we're at is the last key's byte, so we
            // will collect it too.
            let mut output = key.to_vec();
            let skip = exact.as_ref();
            let keep_going = self.completion_stream(&mut output, root.label(), &arc, &mut |c| {
                // omit the exact completion in any further completions.
                if skip.is_some_and(|e| e.utf8 == c.utf8) {
                    return true;
                }
                f(c)
            })?;
            if !keep_going {
                return Ok(());
            }
        }
        Ok(())
    }

    /// `completionStream`: a depth-first walk below `from`, one completion per
    /// `END_LABEL` arc. `false` when `f` asked to stop.
    fn completion_stream(
        &self,
        output: &mut Vec<u8>,
        bucket: i32,
        from: &FstArc<()>,
        f: &mut dyn FnMut(Completion) -> bool,
    ) -> Result<bool> {
        let fst = self.fst();
        let mut reader = fst.bytes_reader();
        let first_of = |arc: &FstArc<()>, reader: &mut _| -> Result<FstArc<()>> {
            let mut target = arc.clone();
            fst.read_first_target_arc(arc, &mut target, reader)
                .map_err(fst_err)?;
            Ok(target)
        };
        let mut states: Vec<(FstArc<()>, usize)> =
            vec![(first_of(from, &mut reader)?, output.len())];
        while let Some(top) = states.last_mut() {
            output.truncate(top.1);
            let label = top.0.label();
            if label == END_LABEL {
                let completion = Completion {
                    utf8: output.clone(),
                    bucket,
                };
                if top.0.is_last() {
                    states.pop();
                } else {
                    fst.read_next_arc(&mut top.0, &mut reader)
                        .map_err(fst_err)?;
                }
                if !f(completion) {
                    return Ok(false);
                }
            } else {
                output.push(label as u8);
                let next = (first_of(&top.0, &mut reader)?, output.len());
                if top.0.is_last() {
                    states.pop();
                } else {
                    fst.read_next_arc(&mut top.0, &mut reader)
                        .map_err(fst_err)?;
                }
                states.push(next);
            }
        }
        Ok(true)
    }

    /// `getBucketCount()`.
    pub fn bucket_count(&self) -> usize {
        self.root_arcs.len()
    }

    /// `getBucket(key)`: `-1` when `key` is not a completion.
    // SENTINEL: `-1` = `key` is not a completion (Java's own return value).
    pub fn bucket_of(&self, key: &str) -> Result<i32> {
        if self.automaton.is_none() {
            return Ok(-1);
        }
        // SENTINEL-OK: handed back unchanged as `bucket_of`'s own declared sentinel.
        self.exact_match_from_root_arc(0, key.as_bytes())
    }

    /// `getFST()`.
    pub fn fst_handle(&self) -> Option<&Arc<Automaton>> {
        self.automaton.as_ref()
    }
}

/// `FSTCompletionBuilder`.
pub struct FstCompletionBuilder {
    buckets: i32,
    sorter: Box<dyn BytesRefSorter>,
    suffix_ram_limit_mb: f64,
}

impl FstCompletionBuilder {
    /// `new FSTCompletionBuilder()`.
    pub fn new() -> Self {
        Self {
            buckets: DEFAULT_BUCKETS,
            sorter: Box::new(InMemorySorter::natural()),
            suffix_ram_limit_mb: f64::INFINITY,
        }
    }

    /// `new FSTCompletionBuilder(buckets, sorter, suffixRAMLimitMB)`.
    pub fn with_sorter(
        buckets: i32,
        sorter: Box<dyn BytesRefSorter>,
        suffix_ram_limit_mb: f64,
    ) -> Result<Self> {
        if !(1..=255).contains(&buckets) {
            return Err(Error::IllegalArgument(format!(
                "Buckets must be >= 1 and <= 255: {buckets}"
            )));
        }
        Ok(Self {
            buckets,
            sorter,
            suffix_ram_limit_mb,
        })
    }

    /// `add(utf8, bucket)`.
    pub fn add(&mut self, utf8: &[u8], bucket: i32) -> Result<()> {
        if bucket < 0 || bucket >= self.buckets {
            return Err(Error::IllegalArgument(format!(
                "Bucket outside of the allowed range [0, {}): {bucket}",
                self.buckets
            )));
        }
        let mut entry = Vec::with_capacity(utf8.len() + 1);
        entry.push(bucket as u8);
        entry.extend_from_slice(utf8);
        self.sorter.add(&entry)
    }

    /// `build()`: the completion over everything added, with a null automaton
    /// when nothing was.
    pub fn build(&mut self) -> Result<FstCompletion> {
        let automaton = self.build_automaton()?.map(Arc::new);
        FstCompletion::with_defaults(automaton)
    }

    fn build_automaton(&mut self) -> Result<Option<Automaton>> {
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1)
            .suffix_ram_limit_mb(self.suffix_ram_limit_mb)
            .map_err(compile_err)?
            .build::<NoOutputs>();
        let mut previous: Vec<u8> = Vec::new();
        let mut count = 0usize;
        for entry in self.sorter.iterator()? {
            count += 1;
            if previous != entry {
                compiler
                    .add(&to_ints_ref(&entry), ())
                    .map_err(compile_err)?;
                previous = entry;
            }
        }
        if count == 0 {
            return Ok(None);
        }
        Ok(compiler.compile().map(Automaton::from_compiled))
    }
}

impl Default for FstCompletionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// `FSTCompletionLookup`'s `sharedTailLength`.
const SHARED_TAIL_LENGTH: f64 = 5.0;

/// `FSTCompletionLookup`.
pub struct FstCompletionLookup {
    buckets: i32,
    exact_match_first: bool,
    higher_weights_completion: Option<FstCompletion>,
    normal_completion: Option<FstCompletion>,
    count: u64,
}

impl FstCompletionLookup {
    /// `new FSTCompletionLookup()`: ten buckets, exact match first.
    pub fn new() -> Self {
        Self::with_options(DEFAULT_BUCKETS, true)
    }

    /// `new FSTCompletionLookup(tempDir, prefix, buckets, exactMatchFirst)`.
    pub fn with_options(buckets: i32, exact_match_first: bool) -> Self {
        Self {
            buckets,
            exact_match_first,
            higher_weights_completion: None,
            normal_completion: None,
            count: 0,
        }
    }

    /// `new FSTCompletionLookup(tempDir, prefix, completion, exactMatchFirst)`:
    /// wraps an FST built elsewhere.
    pub fn from_completion(completion: &FstCompletion, exact_match_first: bool) -> Result<Self> {
        let fst = completion.fst_handle().cloned();
        Ok(Self {
            buckets: -1,
            exact_match_first,
            normal_completion: Some(FstCompletion::new(fst.clone(), false, exact_match_first)?),
            higher_weights_completion: Some(FstCompletion::new(fst, true, exact_match_first)?),
            count: 0,
        })
    }

    /// `encodeWeight`.
    fn encode_weight(value: i64) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| unsupported_operation(format!("cannot encode value: {value}")))
    }

    /// `get(key)`: the bucket of an exact key.
    pub fn get(&self, key: &str) -> Result<Option<i64>> {
        let Some(c) = &self.normal_completion else {
            return Err(illegal_state("not built"));
        };
        let bucket = c.bucket_of(key)?;
        Ok((bucket != -1).then_some(i64::from(bucket)))
    }
}

impl Default for FstCompletionLookup {
    fn default() -> Self {
        Self::new()
    }
}

impl Lookup for FstCompletionLookup {
    fn get_count(&self) -> Result<u64> {
        Ok(self.count)
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        if input.has_payloads() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support payloads".into(),
            ));
        }
        if input.has_contexts() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        // Push weights up front before the terms to sort on them. Java writes
        // `Integer.reverseBytes(weight)` with `writeInt`, which is
        // little-endian since Lucene 9: the bytes are the weight big-endian,
        // "for comparing in lexicographical order".
        let mut lines: Vec<Vec<u8>> = Vec::new();
        while let Some(term) = input.next() {
            let encoded = Self::encode_weight(input.weight())?;
            let mut line = Vec::with_capacity(term.len() + 4);
            line.extend_from_slice(&encoded.to_be_bytes());
            line.extend_from_slice(&term);
            lines.push(line);
        }
        let input_line_count = lines.len();
        lines.sort();
        // We don't know the distribution of scores and we need to bucket them,
        // so we'll sort and divide into equal buckets.
        let mut builder = FstCompletionBuilder::with_sorter(
            self.buckets,
            Box::new(InMemorySorter::natural()),
            SHARED_TAIL_LENGTH,
        )?;
        let (mut previous_bucket, mut previous_score) = (0i32, 0i32);
        let mut new_count = 0u64;
        for (line, scratch) in lines.iter().enumerate() {
            let current_score =
                i32::from_be_bytes([scratch[0], scratch[1], scratch[2], scratch[3]]);
            let bucket = if line > 0 && current_score == previous_score {
                previous_bucket
            } else {
                (line as i64 * i64::from(self.buckets) / input_line_count as i64) as i32
            };
            previous_score = current_score;
            previous_bucket = bucket;
            // Only append the input, discard the weight.
            builder.add(&scratch[4..], bucket)?;
            new_count += 1;
        }
        // The two FSTCompletions share the same automaton.
        let higher = builder.build()?;
        self.normal_completion = Some(FstCompletion::new(
            higher.fst_handle().cloned(),
            false,
            self.exact_match_first,
        )?);
        self.higher_weights_completion = Some(higher);
        self.count = new_count;
        Ok(())
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        higher_weights_first: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        if contexts.is_some() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        let completion = if higher_weights_first {
            &self.higher_weights_completion
        } else {
            &self.normal_completion
        };
        let Some(completion) = completion else {
            return Err(illegal_state("not built"));
        };
        Ok(completion
            .lookup(key, num)?
            .into_iter()
            .map(|c| {
                LookupResult::new(
                    String::from_utf8_lossy(&c.utf8).into_owned(),
                    i64::from(c.bucket),
                )
            })
            .collect())
    }

    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        output.write_vlong(self.count as i64);
        let Some(fst) = self
            .normal_completion
            .as_ref()
            .and_then(FstCompletion::fst_handle)
        else {
            return Ok(false);
        };
        output.write_bytes(&fst.save());
        Ok(true)
    }

    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        self.count = input.read_vlong()? as u64;
        let mut rest = vec![0u8; input.remaining()];
        input.read_bytes(&mut rest)?;
        let fst = Automaton::read(&rest).map_err(fst_err)?;
        let higher = FstCompletion::with_defaults(Some(Arc::new(fst)))?;
        self.normal_completion = Some(FstCompletion::new(
            higher.fst_handle().cloned(),
            false,
            self.exact_match_first,
        )?);
        self.higher_weights_completion = Some(higher);
        Ok(true)
    }
}
