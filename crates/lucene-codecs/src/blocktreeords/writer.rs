//! Port of `OrdsBlockTreeTermsWriter`: the `.tio`/`.tipo` of one
//! `BlockTreeOrds` group, over `Lucene104PostingsWriter`'s postings.
//!
//! The block cutting (`pushTerm`/`writeBlocks`/`writeBlock`) is Lucene 4's,
//! line for line. Each block also counts the terms below it
//! (`totalTermCount`, its own terms plus every sub-block's
//! `totFloorTermCount`), which goes on disk beside each sub-block pointer
//! and, as `startOrd`/`endOrd`, into the terms index.
//!
//! One difference in how, not what: Java compiles an FST per block and
//! re-adds every entry of the sub-blocks' FSTs (a `BytesRefFSTEnum` walk)
//! into its parent's, shifting their ords; an FST's entries are exactly the
//! inputs it was built from, in that order, so this port carries each
//! block's entries as a list and compiles only the root's, which is the one
//! written. Same bytes.

use lucene_store::codec_util;
use lucene_store::data_output::DataOutput;

use super::outputs::{FstOrdsOutputs, Output};
use super::{
    Result, OUTPUT_FLAGS_NUM_BITS, OUTPUT_FLAG_HAS_TERMS, OUTPUT_FLAG_IS_FLOOR, TERMS_CODEC_NAME,
    TERMS_INDEX_CODEC_NAME, VERSION_CURRENT,
};
use crate::field_infos::IndexOptions;
use crate::fst::InputType;
use crate::fst_compiler::FstCompilerBuilder;
use crate::postings_writer::{self, FieldTermStates};
use crate::segment_state::SegmentWriteState;

/// `OrdsBlockTreeTermsWriter.encodeOutput`.
// ARITH: a file pointer of an in-memory file, far below 2^62 (Java's assert).
#[allow(clippy::arithmetic_side_effects)]
fn encode_output(fp: i64, has_terms: bool, is_floor: bool) -> i64 {
    (fp << OUTPUT_FLAGS_NUM_BITS)
        | if has_terms { OUTPUT_FLAG_HAS_TERMS } else { 0 }
        | if is_floor { OUTPUT_FLAG_IS_FLOOR } else { 0 }
}

/// The `(input, output)` pairs of a block's FST, in input order.
type IndexEntries = Vec<(Vec<u8>, Output)>;

/// `PendingBlock`.
#[derive(Debug)]
struct PendingBlock {
    prefix: Vec<u8>,
    fp: i64,
    /// `index`: the entries of this block's FST once compiled.
    index: IndexEntries,
    /// `subIndices`: each sub-block's entries and `termOrdStart`.
    sub_indices: Option<Vec<(IndexEntries, i64)>>,
    has_terms: bool,
    is_floor: bool,
    floor_lead_byte: i32,
    tot_floor_term_count: i64,
    total_term_count: i64,
}

/// `PendingEntry`: a term (its index in the field's [`FieldTermStates`]) or
/// a written block.
#[derive(Debug)]
enum PendingEntry {
    Term(usize),
    Block(PendingBlock),
}

/// `FieldMetaData`.
#[derive(Debug)]
struct FieldMetaData {
    field_number: i32,
    index_options: IndexOptions,
    root_code: Vec<u8>,
    num_terms: i64,
    index_start_fp: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
}

/// `OrdsBlockTreeTermsWriter`: the group's terms dictionary, one field at a
/// time ([`Self::write_field`], handed each field's [`FieldTermStates`] by
/// the postings writer), then [`Self::finish`] (`close()`).
#[derive(Debug)]
pub struct OrdsBlockTreeTermsWriter {
    out: Vec<u8>,
    index_out: Vec<u8>,
    min_items_in_block: usize,
    max_items_in_block: usize,
    fields: Vec<FieldMetaData>,
}

impl OrdsBlockTreeTermsWriter {
    /// The constructor: both headers and `postingsWriter.init(out)`. The
    /// block sizes are the format's, already validated.
    pub fn new(
        state: &SegmentWriteState<'_>,
        min_items_in_block: usize,
        max_items_in_block: usize,
    ) -> Self {
        let mut out = Vec::new();
        codec_util::write_index_header(
            &mut out,
            TERMS_CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        let mut index_out = Vec::new();
        codec_util::write_index_header(
            &mut index_out,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        postings_writer::write_postings_terms_header(
            &mut out,
            state.segment_id,
            state.segment_suffix,
        );
        OrdsBlockTreeTermsWriter {
            out,
            index_out,
            min_items_in_block,
            max_items_in_block,
            fields: Vec::new(),
        }
    }

    /// `write(fields, norms)` for one field: every term through
    /// `TermsWriter.write`, then `TermsWriter.finish`.
    ///
    /// # Errors
    /// An FST the compiler refuses (terms out of order).
    // ARITH: sums of a field's statistics, which Java keeps in longs.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn write_field(&mut self, field: &FieldTermStates<'_>) -> Result<()> {
        let mut tw = TermsWriter::new(field);
        for i in 0..field.len() {
            tw.sum_doc_freq += i64::from(field.doc_freq(i));
            tw.sum_total_term_freq = tw
                .sum_total_term_freq
                .wrapping_add(field.total_term_freq(i));
            tw.push_term(self, field.term(i));
            tw.pending.push(PendingEntry::Term(i));
            tw.num_terms += 1;
        }
        tw.finish(self)
    }

    /// `close()`: the field directory, the trailers and the footers;
    /// returns the `.tio` and the `.tipo`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        let dir_start = self.out.len() as i64;
        let index_dir_start = self.index_out.len() as i64;
        // At most one entry per field of a segment.
        self.out.write_vint(self.fields.len() as i32);
        for field in &self.fields {
            self.out.write_vint(field.field_number);
            self.out.write_vlong(field.num_terms);
            self.out.write_vint(field.root_code.len() as i32);
            self.out.write_bytes(&field.root_code);
            if field.index_options != IndexOptions::Docs {
                self.out.write_vlong(field.sum_total_term_freq);
            }
            self.out.write_vlong(field.sum_doc_freq);
            self.out.write_vint(field.doc_count);
            self.index_out.write_vlong(field.index_start_fp);
            self.out.write_vint(field.min_term.len() as i32);
            self.out.write_bytes(&field.min_term);
            self.out.write_vint(field.max_term.len() as i32);
            self.out.write_bytes(&field.max_term);
        }
        self.out.write_i64(dir_start);
        codec_util::write_footer(&mut self.out);
        self.index_out.write_i64(index_dir_start);
        codec_util::write_footer(&mut self.index_out);
        (self.out, self.index_out)
    }
}

/// `PendingBlock.compileIndex`: this block's entry -- its frame data, its
/// floor followers' and the ords of all of them -- then its sub-blocks'
/// entries, their ords shifted by the terms before them.
// ARITH: file pointers and term counts of in-memory files and lists.
#[allow(clippy::arithmetic_side_effects)]
fn compile_index(blocks: &mut [PendingBlock]) {
    let (first, rest) = blocks
        .split_first_mut()
        .expect("writeBlocks writes at least one block");
    let mut last_sum_total_term_count = 0;
    let mut sum_total_term_count = first.total_term_count;
    let mut scratch = Vec::new();
    scratch.write_vlong(encode_output(first.fp, first.has_terms, first.is_floor));
    if first.is_floor {
        scratch.write_vint(rest.len() as i32);
        for sub in rest.iter() {
            scratch.write_byte(sub.floor_lead_byte as u8);
            scratch.write_vlong(sum_total_term_count - last_sum_total_term_count);
            last_sum_total_term_count = sum_total_term_count;
            sum_total_term_count += sub.total_term_count;
            scratch.write_vlong(((sub.fp - first.fp) << 1) | i64::from(sub.has_terms));
        }
    }
    let mut entries = vec![(
        first.prefix.clone(),
        Output::new(scratch, 0, i64::MAX - (sum_total_term_count - 1)),
    )];
    let mut term_ord_offset = 0;
    for block in std::iter::once(&mut *first).chain(rest.iter_mut()) {
        if let Some(sub_indices) = block.sub_indices.take() {
            for (sub, term_ord_start) in sub_indices {
                // `append`.
                let shift = term_ord_offset + term_ord_start;
                entries.extend(sub.into_iter().map(|(input, out)| {
                    let (start, end) = (out.start_ord + shift, out.end_ord - shift);
                    (input, Output::new(out.bytes, start, end))
                }));
            }
        }
        term_ord_offset += block.total_term_count;
    }
    first.tot_floor_term_count = term_ord_offset;
    first.index = entries;
}

/// `TermsWriter`: one field.
struct TermsWriter<'f, 's> {
    field: &'f FieldTermStates<'s>,
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    last_term: Vec<u8>,
    /// `prefixStarts`: Java's `int[]`, which goes negative after a block
    /// is written (that slot is rewritten before it is read again).
    prefix_starts: Vec<i64>,
    pending: Vec<PendingEntry>,
    new_blocks: Vec<PendingBlock>,
}

impl<'f, 's> TermsWriter<'f, 's> {
    fn new(field: &'f FieldTermStates<'s>) -> Self {
        TermsWriter {
            field,
            num_terms: 0,
            sum_total_term_freq: 0,
            sum_doc_freq: 0,
            last_term: Vec::new(),
            prefix_starts: Vec::new(),
            pending: Vec::new(),
            new_blocks: Vec::new(),
        }
    }

    /// The bytes of term `i`.
    fn term(&self, i: usize) -> &'f [u8] {
        let field: &'f FieldTermStates<'s> = self.field;
        field.term(i)
    }

    /// `pushTerm(text)`: writes the blocks the new term closes.
    // ARITH: indices into `last_term`/`prefix_starts`, bounded by their
    // lengths; counts of `pending` in i64.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_term(&mut self, w: &mut OrdsBlockTreeTermsWriter, text: &[u8]) {
        let limit = self.last_term.len().min(text.len());
        let mut pos = 0;
        while pos < limit && self.last_term[pos] == text[pos] {
            pos += 1;
        }
        for i in (pos..self.last_term.len()).rev() {
            let prefix_top_size = self.pending.len() as i64 - self.prefix_starts[i];
            if prefix_top_size >= w.min_items_in_block as i64 {
                self.write_blocks(w, i + 1, prefix_top_size as usize);
                self.prefix_starts[i] -= prefix_top_size - 1;
            }
        }
        if self.prefix_starts.len() < text.len() {
            self.prefix_starts.resize(text.len(), 0);
        }
        for i in pos..text.len() {
            self.prefix_starts[i] = self.pending.len() as i64;
        }
        self.last_term.clear();
        self.last_term.extend_from_slice(text);
    }

    /// `writeBlocks(prefixLength, count)`: the top `count` entries become
    /// one block, or several floor blocks, replaced by the first of them.
    // ARITH: indices into `pending`, bounded by its length.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_blocks(
        &mut self,
        w: &mut OrdsBlockTreeTermsWriter,
        prefix_length: usize,
        count: usize,
    ) {
        let mut last_suffix_lead_label: i32 = -1;
        let mut has_terms = false;
        let mut has_sub_blocks = false;
        let start = self.pending.len() - count;
        let end = self.pending.len();
        let mut next_block_start = start;
        let mut next_floor_lead_label: i32 = -1;
        for i in start..end {
            let suffix_lead_label = match &self.pending[i] {
                PendingEntry::Term(t) => {
                    let term = self.term(*t);
                    if term.len() == prefix_length {
                        -1
                    } else {
                        i32::from(term[prefix_length])
                    }
                }
                PendingEntry::Block(block) => i32::from(block.prefix[prefix_length]),
            };
            if suffix_lead_label != last_suffix_lead_label {
                let items_in_block = i - next_block_start;
                if items_in_block >= w.min_items_in_block
                    && end - next_block_start > w.max_items_in_block
                {
                    let is_floor = items_in_block < count;
                    let block = self.write_block(
                        w,
                        prefix_length,
                        is_floor,
                        next_floor_lead_label,
                        next_block_start,
                        i,
                        has_terms,
                        has_sub_blocks,
                    );
                    self.new_blocks.push(block);
                    has_terms = false;
                    has_sub_blocks = false;
                    next_floor_lead_label = suffix_lead_label;
                    next_block_start = i;
                }
                last_suffix_lead_label = suffix_lead_label;
            }
            match self.pending[i] {
                PendingEntry::Term(_) => has_terms = true,
                PendingEntry::Block(_) => has_sub_blocks = true,
            }
        }
        if next_block_start < end {
            let items_in_block = end - next_block_start;
            let is_floor = items_in_block < count;
            let block = self.write_block(
                w,
                prefix_length,
                is_floor,
                next_floor_lead_label,
                next_block_start,
                end,
                has_terms,
                has_sub_blocks,
            );
            self.new_blocks.push(block);
        }
        compile_index(&mut self.new_blocks);
        self.pending.truncate(self.pending.len() - count);
        let mut blocks = std::mem::take(&mut self.new_blocks).into_iter();
        let first = blocks.next().expect("compile_index ran on a block");
        self.pending.push(PendingEntry::Block(first));
    }

    /// `writeBlock(...)`: entries `start..end` of `pending` as one block.
    // ARITH: lengths and file pointers of in-memory buffers; suffix lengths
    // are differences of a longer term's length and a shared prefix's.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        w: &mut OrdsBlockTreeTermsWriter,
        prefix_length: usize,
        is_floor: bool,
        floor_lead_label: i32,
        start: usize,
        end: usize,
        has_terms: bool,
        has_sub_blocks: bool,
    ) -> PendingBlock {
        let start_fp = w.out.len() as i64;
        let has_floor_lead_label = is_floor && floor_lead_label != -1;
        let mut prefix = self.last_term[..prefix_length].to_vec();
        let num_entries = end - start;
        let mut code = (num_entries as i32) << 1;
        if end == self.pending.len() {
            code |= 1;
        }
        w.out.write_vint(code);
        let is_leaf_block = !has_sub_blocks;
        let field = self.field;
        let has_freqs = field.index_options != IndexOptions::Docs;
        let mut suffix_writer = Vec::new();
        let mut stats_writer = Vec::new();
        let mut block_terms = Vec::new();
        let mut total_term_count: i64 = 0;
        let mut sub_indices = if is_leaf_block {
            None
        } else {
            Some(Vec::new())
        };
        for i in start..end {
            match &mut self.pending[i] {
                PendingEntry::Term(t) => {
                    let t = *t;
                    let term = field.term(t);
                    let suffix = term.len() - prefix_length;
                    if is_leaf_block {
                        suffix_writer.write_vint(suffix as i32);
                    } else {
                        suffix_writer.write_vint((suffix as i32) << 1);
                    }
                    suffix_writer.write_bytes(&term[prefix_length..]);
                    let doc_freq = field.doc_freq(t);
                    stats_writer.write_vint(doc_freq);
                    if has_freqs {
                        stats_writer.write_vlong(field.total_term_freq(t) - i64::from(doc_freq));
                    }
                    block_terms.push(t);
                    total_term_count += 1;
                }
                PendingEntry::Block(block) => {
                    let suffix = block.prefix.len() - prefix_length;
                    suffix_writer.write_vint(((suffix as i32) << 1) | 1);
                    suffix_writer.write_bytes(&block.prefix[prefix_length..]);
                    suffix_writer.write_vlong(start_fp - block.fp);
                    suffix_writer.write_vlong(block.tot_floor_term_count);
                    if let Some(subs) = sub_indices.as_mut() {
                        subs.push((std::mem::take(&mut block.index), total_term_count));
                    }
                    total_term_count += block.tot_floor_term_count;
                }
            }
        }
        // `postingsWriter.encodeTerm(metaWriter, ..., absolute)` per term,
        // the first absolute.
        let mut meta_writer = Vec::new();
        field.encode_terms(&mut meta_writer, &block_terms);
        w.out
            .write_vint(((suffix_writer.len() as i32) << 1) | i32::from(is_leaf_block));
        w.out.write_bytes(&suffix_writer);
        w.out.write_vint(stats_writer.len() as i32);
        w.out.write_bytes(&stats_writer);
        w.out.write_vint(meta_writer.len() as i32);
        w.out.write_bytes(&meta_writer);
        if has_floor_lead_label {
            prefix.push(floor_lead_label as u8);
        }
        PendingBlock {
            prefix,
            fp: start_fp,
            index: Vec::new(),
            sub_indices,
            has_terms,
            is_floor,
            floor_lead_byte: floor_lead_label,
            tot_floor_term_count: 0,
            total_term_count,
        }
    }

    /// `finish()`: the root block, its FST into `.tipo`, the field's record.
    fn finish(mut self, w: &mut OrdsBlockTreeTermsWriter) -> Result<()> {
        if self.num_terms == 0 {
            return Ok(());
        }
        let count = self.pending.len();
        self.write_blocks(w, 0, count);
        let Some(PendingEntry::Block(root)) = self.pending.pop() else {
            unreachable!("writeBlocks(0, all) leaves the root block alone");
        };
        let index_start_fp = w.index_out.len() as i64;
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<FstOrdsOutputs>();
        let mut root_code = Vec::new();
        for (input, output) in root.index {
            if input.is_empty() {
                root_code.clone_from(&output.bytes);
            }
            compiler.add_bytes(&input, output)?;
        }
        let fst = compiler
            .compile()
            .expect("the root FST has the empty input");
        w.index_out.write_bytes(&fst.save::<FstOrdsOutputs>());
        let field = self.field;
        w.fields.push(FieldMetaData {
            field_number: field.field_number,
            index_options: field.index_options,
            root_code,
            num_terms: self.num_terms,
            index_start_fp,
            sum_total_term_freq: self.sum_total_term_freq,
            sum_doc_freq: self.sum_doc_freq,
            doc_count: field.doc_count,
            min_term: field.term(0).to_vec(),
            max_term: field.term(field.len().saturating_sub(1)).to_vec(),
        });
        Ok(())
    }
}
