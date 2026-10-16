//! `org.apache.lucene.search.suggest.fst.WFSTCompletionLookup` (M13 T13.3b):
//! a weighted automaton over the suggestions. Each term's FST output is its
//! weight, encoded as `Integer.MAX_VALUE - weight` so that the smallest output
//! is the best suggestion, and `Util.shortestPaths` finds the top completions
//! of a prefix.
//!
//! Java sorts the input with `SortedInputIterator` over an encoding of the
//! term and its weight; here the sort is in memory with the same order
//! `(term, encoded weight)`, so the heaviest duplicate comes first and is the
//! one kept. `load` reads the rest of its input for the FST (see
//! [`super::fst_completion`]).

use std::collections::BTreeSet;

use lucene_codecs::fst::{InputType, PositiveIntOutputs};
use lucene_codecs::fst_compiler::{FstCompilerBuilder, FstReadError};
use lucene_codecs::fst_util::{shortest_paths, to_ints_ref, Arc as FstArc, TypedFst};
use lucene_store::{DataInput, DataOutput, Error, Result};

use super::input::InputIterator;
use super::lookup::{unsupported_operation, Lookup, LookupResult};

type WeightedFst = TypedFst<PositiveIntOutputs>;

fn fst_err(e: FstReadError) -> Error {
    Error::Corrupted(e.0)
}

/// `WFSTCompletionLookup`.
pub struct WfstCompletionLookup {
    fst: Option<WeightedFst>,
    exact_first: bool,
    count: u64,
}

impl WfstCompletionLookup {
    /// `new WFSTCompletionLookup(tempDir, prefix)`: exact match first.
    pub fn new() -> Self {
        Self::with_exact_first(true)
    }

    /// `new WFSTCompletionLookup(tempDir, prefix, exactFirst)`.
    pub fn with_exact_first(exact_first: bool) -> Self {
        Self {
            fst: None,
            exact_first,
            count: 0,
        }
    }

    /// `lookupPrefix(scratch, arc)`: the summed output along `prefix`, with
    /// `arc` left on its last arc, or `None` when the prefix is not a path.
    fn lookup_prefix(fst: &WeightedFst, prefix: &[u8]) -> Result<Option<(i64, FstArc<i64>)>> {
        let mut arc = fst.first_arc();
        let mut reader = fst.bytes_reader();
        let mut output = 0i64;
        for &b in prefix {
            match fst
                .find_target_arc(i32::from(b), &arc, &mut reader)
                .map_err(fst_err)?
            {
                None => return Ok(None),
                Some(next) => {
                    output = output.wrapping_add(*next.output());
                    arc = next;
                }
            }
        }
        Ok(Some((output, arc)))
    }

    /// `get(key)`: the weight of an exact key.
    pub fn get(&self, key: &str) -> Result<Option<i32>> {
        let Some(fst) = &self.fst else {
            return Ok(None);
        };
        match Self::lookup_prefix(fst, key.as_bytes())? {
            Some((output, arc)) if arc.is_final() => Ok(Some(decode_weight(
                output.wrapping_add(*arc.next_final_output()),
            ))),
            _ => Ok(None),
        }
    }
}

impl Default for WfstCompletionLookup {
    fn default() -> Self {
        Self::new()
    }
}

/// `decodeWeight`.
fn decode_weight(encoded: i64) -> i32 {
    (i64::from(i32::MAX).wrapping_sub(encoded)) as i32
}

/// `encodeWeight`.
fn encode_weight(value: i64) -> Result<i32> {
    if !(0..=i64::from(i32::MAX)).contains(&value) {
        return Err(unsupported_operation(format!(
            "cannot encode value: {value}"
        )));
    }
    Ok(i32::MAX - value as i32)
}

impl Lookup for WfstCompletionLookup {
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
        let mut entries: Vec<(Vec<u8>, i32)> = Vec::new();
        while let Some(term) = input.next() {
            entries.push((term, encode_weight(input.weight())?));
        }
        // (term, encoded weight) ascending: for a duplicate the best weight
        // comes first, and is the one added.
        entries.sort();
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        let mut previous: Option<&[u8]> = None;
        let mut new_count = 0u64;
        for (term, cost) in &entries {
            if previous == Some(term.as_slice()) {
                continue; // for duplicate suggestions, the best weight is actually added
            }
            compiler
                .add(&to_ints_ref(term), i64::from(*cost))
                .map_err(|e| Error::IllegalArgument(e.to_string()))?;
            previous = Some(term);
            new_count += 1;
        }
        self.fst = compiler.compile().map(WeightedFst::from_compiled);
        self.count = new_count;
        Ok(())
    }

    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        output.write_vlong(self.count as i64);
        let Some(fst) = &self.fst else {
            return Ok(false);
        };
        output.write_bytes(&fst.save());
        Ok(true)
    }

    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        self.count = input.read_vlong()? as u64;
        let mut rest = vec![0u8; input.remaining()];
        input.read_bytes(&mut rest)?;
        self.fst = Some(WeightedFst::read(&rest).map_err(fst_err)?);
        Ok(true)
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        only_more_popular: bool,
        mut num: usize,
    ) -> Result<Vec<LookupResult>> {
        if contexts.is_some() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        if only_more_popular {
            return Err(Error::IllegalArgument(
                "this suggester only works with onlyMorePopular=false".into(),
            ));
        }
        let Some(fst) = &self.fst else {
            return Ok(Vec::new());
        };
        let mut scratch = key.as_bytes().to_vec();
        let prefix_length = scratch.len();
        // match the prefix portion exactly
        let Some((prefix_output, arc)) = Self::lookup_prefix(fst, &scratch)? else {
            return Ok(Vec::new());
        };
        let mut results = Vec::with_capacity(num);
        if self.exact_first && arc.is_final() {
            results.push(LookupResult::new(
                String::from_utf8_lossy(&scratch).into_owned(),
                i64::from(decode_weight(
                    prefix_output.wrapping_add(*arc.next_final_output()),
                )),
            ));
            num = num.saturating_sub(1);
            if num == 0 {
                return Ok(results); // that was quick
            }
        }
        // complete top-N
        let completions = shortest_paths(
            fst,
            &arc,
            prefix_output,
            |a: &i64, b: &i64| a.cmp(b),
            num,
            !self.exact_first,
        )
        .map_err(fst_err)?;
        for completion in completions.top_n {
            scratch.truncate(prefix_length);
            scratch.extend(completion.input.iter().map(|&l| l as u8));
            results.push(LookupResult::new(
                String::from_utf8_lossy(&scratch).into_owned(),
                i64::from(decode_weight(completion.output)),
            ));
        }
        Ok(results)
    }
}
