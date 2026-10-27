//! Port of lucene-codecs' `org.apache.lucene.codecs.uniformsplit.sharedterms`:
//! `STUniformSplitPostingsFormat`, [`super`]'s format with one terms
//! dictionary for every field of the group. Each block line holds a term and
//! the term states of every field that has it, so a term shared by several
//! fields is written once and the group has a single FST.
//!
//! # The files
//!
//! As [`super`]'s, under codec name `SharedTermsUniformSplit` and the
//! extensions `.stustb`/`.stustd`, with these differences:
//!
//! ```text
//! .stustb  the blocks hold every field's terms, merged in term order;
//!          a line's term states (STBlockLine.Serializer):
//!            one field:  zInt(-fieldNumber), its term state
//!            otherwise:  zInt(numFields), vInt(fieldNumber) per field
//!                        (ascending), then each field's term state
//!          the term state is DeltaBaseTermStateSerializer's, against the
//!          block's base pointers, which the fields share
//!          each field's metadata: dictionaryStartFP is the dictionary's
//!          (the same for all), first/lastBlockStartFP the first and last
//!          block holding one of its terms; fields without terms are left out
//! .stustd  one FST over every block (absent when no field has a term)
//! ```
//!
//! The postings (`.doc`/`.pos`/`.pay`) are written term by term, and for a
//! term, field by field in field-number order, so a group's fields'
//! postings interleave.
//!
//! A field's reader ([`StBlockReader`]) walks the shared blocks and skips
//! the lines whose term the field lacks; [`StMergingBlockReader`] walks them
//! for every field at once, as Java's merge does.
//!
//! # What differs from Java
//!
//! - [`StUniformSplitPostingsFormat::write`] is a flush of the fields as
//!   given (Java's `write(Fields, norms)`: the fields in the order the
//!   `Fields` iterate, name order for a flush).
//!   [`StUniformSplitPostingsFormat::merge`] scans each source union once,
//!   groups equal terms, remaps documents and drops deletes, with target
//!   fields in field-number order (`mergeFieldInfos`). The index layer
//!   supplies document maps, mapped norms, and the generic merge callback
//!   for sorted indexes or foreign/wrapped terms. Rust buffers only the
//!   current term/field's postings; Java uses reusable postings iterators.
//! - The postings writer validates each (term, field) pair alone, so an
//!   `EmptyPostings`/`UnsupportedIndexOptions` error names term index 0;
//!   `EmptyTerms` and `TermsNotSorted` are checked per field before (with
//!   the field's own term index).
//! - `readTermStateForField` reads a line's field numbers twice (Java: into
//!   an `int[numFields]`), so a corrupt count allocates nothing.
//! - `readFieldTermStatesMap` fills a list of `(fieldNumber, state)` in the
//!   line's order (Java: a `Map` by field name), and
//!   [`StMergingBlockReader::postings`] takes a field number (Java: a
//!   name).
//! - A field number the segment's `FieldInfos` lacks is `Corrupted` (Java:
//!   a `NullPointerException` or `IllegalArgumentException`), as are fields
//!   whose `dictionaryStartFP` differ (Java: an assertion, else the last
//!   one wins).
//! - The rest as [`super`]: whole-slice encoders, the dictionary decoded at
//!   open, `Unsupported` for Java's `NullPointerException`s and
//!   `UnsupportedOperationException`s, no `.psm` reader, no `reuse`, no
//!   `Accountable`.

mod block_line;
mod block_reader;
mod merge;
mod reader;
#[cfg(test)]
mod tests;
mod writer;

use std::sync::Arc;

pub use block_line::FieldMetadataTermState;
pub(crate) use block_line::SharedFieldInfos;
pub use block_reader::{StBlockReader, StIntersectBlockReader, StMergingBlockReader};
pub use merge::StMergeSegment;
pub use reader::{StUniformSplitTerms, StUniformSplitTermsReader, UnionFieldMetadataBuilder};
pub use writer::StUniformSplitTermsWriter;

use super::{
    BlockDecoder, BlockEncoder, Error, Result, UniformSplitOutput, UniformSplitPostingsFormat,
};
use crate::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource, WriteOptions};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `STUniformSplitPostingsFormat.TERMS_DICTIONARY_EXTENSION`.
pub const TERMS_DICTIONARY_EXTENSION: &str = "stustd";
/// `STUniformSplitPostingsFormat.TERMS_BLOCKS_EXTENSION`.
pub const TERMS_BLOCKS_EXTENSION: &str = "stustb";
/// `STUniformSplitPostingsFormat.NAME`: both files' codec name.
pub const NAME: &str = "SharedTermsUniformSplit";
/// `STUniformSplitPostingsFormat.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = super::VERSION_CURRENT;

/// `STUniformSplitPostingsFormat`: [`UniformSplitPostingsFormat`]'s
/// settings, its terms writer and reader swapped for the shared-terms ones.
#[derive(Debug, Clone, Default)]
pub struct StUniformSplitPostingsFormat {
    base: UniformSplitPostingsFormat,
}

impl StUniformSplitPostingsFormat {
    /// `new STUniformSplitPostingsFormat(targetNumBlockLines, deltaNumLines,
    /// blockEncoder, blockDecoder, dictionaryOnHeap)`.
    ///
    /// # Errors
    /// As [`UniformSplitPostingsFormat::new`].
    pub fn new(
        target_num_block_lines: i32,
        delta_num_lines: i32,
        block_encoder: Option<Arc<dyn BlockEncoder>>,
        block_decoder: Option<Arc<dyn BlockDecoder>>,
        dictionary_on_heap: bool,
    ) -> Result<Self> {
        Ok(StUniformSplitPostingsFormat {
            base: UniformSplitPostingsFormat::new(
                target_num_block_lines,
                delta_num_lines,
                block_encoder,
                block_decoder,
                dictionary_on_heap,
            )?,
        })
    }

    /// The settings, as the parent format holds them.
    pub fn settings(&self) -> &UniformSplitPostingsFormat {
        &self.base
    }

    /// `fieldsConsumer(state).write(fields, norms)` and `close()`: the
    /// fields in the order given (see the module docs), each sorted.
    /// [`UniformSplitOutput::ustb`]/[`UniformSplitOutput::ustd`] are the
    /// `.stustb`/`.stustd`.
    ///
    /// # Errors
    /// What the postings writer rejects in `inputs`; [`Error::Write`]
    /// `EmptyTerms`/`TermsNotSorted` for a field without terms or unsorted
    /// ones; [`Error::IllegalArgument`] for a field number given twice or
    /// unknown to `state`; what the dictionary and the encoder fail with.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<UniformSplitOutput> {
        let options = state.postings_options(WriteOptions::default());
        let fields: Vec<(i32, i32)> = inputs
            .iter()
            .map(|input| (input.field_number, input.doc_count))
            .collect();
        let mut terms = StUniformSplitTermsWriter::new(
            state,
            self.base.target_num_block_lines,
            self.base.delta_num_lines,
            self.base.block_encoder.clone(),
            &fields,
        )?;
        let lines = term_major_inputs(inputs)?;
        let postings = postings_writer::write_postings(
            &lines,
            norms,
            options.has_prox,
            options.has_payloads_or_offsets,
            state.segment_id,
            state.segment_suffix,
            |field| terms.add_term_states(field),
        )?;
        let (ustb, ustd) = terms.finish()?;
        Ok(UniformSplitOutput {
            postings,
            ustb,
            ustd,
        })
    }

    /// `fieldsProducer(state)`: the group's fields, from its
    /// `.stustb`/`.stustd`.
    ///
    /// # Errors
    /// See [`StUniformSplitTermsReader::open`].
    pub fn open(
        &self,
        stustb: Vec<u8>,
        stustd: &[u8],
        state: &SegmentReadState<'_>,
    ) -> Result<StUniformSplitTermsReader> {
        StUniformSplitTermsReader::open(stustb, stustd, state, self.base.block_decoder.clone())
    }
}

/// `writeSingleSegment`'s `TermIteratorQueue<FieldTerms>`: each (term,
/// field) pair as a one-term input, by term, then field number -- the order
/// `writePostingLines` hands them to the postings writer.
///
/// # Errors
/// `EmptyTerms`/`TermsNotSorted` per field; [`Error::IllegalArgument`] for
/// a field number given twice.
fn term_major_inputs<'a, T: TermSource>(
    inputs: &[FieldPostingsInput<'a, T>],
) -> Result<Vec<FieldPostingsInput<'a, T>>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut queue = BinaryHeap::with_capacity(inputs.len());
    let mut total = 0usize;
    for (index, input) in inputs.iter().enumerate() {
        if inputs[..index]
            .iter()
            .any(|other| other.field_number == input.field_number)
        {
            return Err(Error::IllegalArgument(format!(
                "field number {} given twice",
                input.field_number
            )));
        }
        let Some(first) = input.terms.first() else {
            return Err(postings_writer::Error::EmptyTerms.into());
        };
        if let Some(at) = input
            .terms
            .windows(2)
            .position(|w| w[0].term() >= w[1].term())
        {
            // ARITH: `at` indexes `terms.windows(2)`, so `at + 1 < len`.
            #[allow(clippy::arithmetic_side_effects)]
            let at = at + 1;
            return Err(postings_writer::Error::TermsNotSorted(at).into());
        }
        total = total.saturating_add(input.terms.len());
        queue.push(Reverse((first.term(), input.field_number, index, 0usize)));
    }
    let mut lines = Vec::with_capacity(total);
    while let Some(Reverse((_, field_number, index, ord))) = queue.pop() {
        let input = &inputs[index];
        let terms: &'a [T] = input.terms;
        lines.push(FieldPostingsInput {
            field_number,
            index_options: input.index_options,
            doc_count: input.doc_count,
            has_payloads: input.has_payloads,
            terms: std::slice::from_ref(&terms[ord]),
        });
        // ARITH: `ord < terms.len()`, so `ord + 1 <= len`.
        #[allow(clippy::arithmetic_side_effects)]
        let next = ord + 1;
        if let Some(term) = terms.get(next) {
            queue.push(Reverse((term.term(), field_number, index, next)));
        }
    }
    Ok(lines)
}

/// A shared-terms uniform-split format under the name it is registered by:
/// `SharedTermsUniformSplit`, or the test framework's
/// `SharedTermsUniformSplitRot13`.
#[derive(Debug, Clone)]
pub struct NamedStUniformSplit {
    /// `getName()`.
    pub name: &'static str,
    pub format: StUniformSplitPostingsFormat,
}

/// `new STUniformSplitPostingsFormat()`, the instance
/// [`crate::terms_dictionary::lookup`] reads its groups with.
pub static DEFAULT_FORMAT: NamedStUniformSplit = NamedStUniformSplit {
    name: NAME,
    format: StUniformSplitPostingsFormat {
        base: UniformSplitPostingsFormat {
            target_num_block_lines: super::DEFAULT_TARGET_NUM_BLOCK_LINES,
            delta_num_lines: super::DEFAULT_DELTA_NUM_LINES,
            block_encoder: None,
            block_decoder: None,
            dictionary_on_heap: false,
        },
    },
};

impl crate::terms_dictionary::TermsDictionaryFormat for NamedStUniformSplit {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extensions(&self) -> &'static [&'static str] {
        &[TERMS_DICTIONARY_EXTENSION, TERMS_BLOCKS_EXTENSION]
    }

    fn write(
        &self,
        inputs: &[crate::terms_dictionary::ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        _options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(postings_writer::PostingsFiles, Vec<Vec<u8>>)> {
        let out = self
            .format
            .write(inputs, norms, state)
            .map_err(super::to_writer_error)?;
        Ok((out.postings, vec![out.ustd, out.ustb]))
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> crate::blocktree::Result<Vec<postings_writer::DictionaryField>> {
        let err = |e| super::to_blocktree_error(self.name, e);
        let [stustd, stustb] = files else {
            return Err(crate::blocktree::Error::Unsupported(
                "a SharedTermsUniformSplit group needs its .stustd and .stustb",
            ));
        };
        let reader = self
            .format
            .open(stustb.to_vec(), stustd, state)
            .map_err(err)?;
        let mut fields = Vec::with_capacity(reader.size());
        for name in reader.field_names() {
            let Some(terms) = reader.terms(name) else {
                continue;
            };
            let info = terms.field_info();
            let mut field = postings_writer::DictionaryField {
                field_number: info.number,
                index_options: info.index_options,
                has_payloads: info.store_payloads,
                doc_count: terms.doc_count(),
                terms: Vec::new(),
            };
            let mut it = terms.iter();
            while let Some(bytes) = it.next().map_err(err)? {
                let bytes = bytes.to_vec();
                let state = it.term_state().map_err(err)?;
                field.terms.push(postings_writer::DictionaryTerm {
                    bytes,
                    doc_freq: state.doc_freq,
                    total_term_freq: state.total_term_freq,
                    meta: state.meta,
                });
            }
            fields.push(field);
        }
        Ok(fields)
    }
}
