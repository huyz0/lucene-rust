//! Port of `VersionBlockTreeTermsWriter` with `IDVersionPostingsWriter` as
//! its postings writer: the `.tiv`/`.tipv` of one `IDVersion` group.
//!
//! The block cutting (`pushTerm`/`writeBlocks`/`writeBlock`) is Lucene 4's,
//! line for line. Each block remembers the largest version among its terms
//! and the sub-blocks it points to (each sub-block's *own* maximum, not its
//! floor followers': Java's quirk, kept), and the terms index records
//! `Long.MAX_VALUE` minus that maximum for every prefix, over all of a floor
//! block's blocks.
//!
//! One difference in how, not what: Java compiles an FST per block and
//! re-adds every entry of the sub-blocks' FSTs (a `BytesRefFSTEnum` walk)
//! into its parent's; an FST's entries are exactly the inputs it was built
//! from, in that order, so this port carries each block's entries as a list
//! and compiles only the root's, which is the one written. Same bytes.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_output::DataOutput;
use lucene_store::error::Error;

use super::{
    IdVersionPostingsFormat, IndexOutput, IndexOutputs, MAX_VERSION, MIN_VERSION,
    OUTPUT_FLAGS_NUM_BITS, OUTPUT_FLAG_HAS_TERMS, OUTPUT_FLAG_IS_FLOOR, POSTINGS_TERMS_CODEC,
    TERMS_CODEC_NAME, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT,
};
use crate::field_infos::IndexOptions;
use crate::fst::{InputType, Pair};
use crate::fst_compiler::FstCompilerBuilder;
use crate::postings_writer::TermSource;

/// One field of the group, its terms in byte order.
#[derive(Debug)]
pub struct IdVersionFieldInput<'a, T> {
    pub field_number: i32,
    pub index_options: IndexOptions,
    /// `FieldInfo.hasTermVectors()`, which `setField` refuses.
    pub store_term_vectors: bool,
    /// `FieldInfo.hasPayloads()`: without it every occurrence's payload is
    /// `null`, which `addPosition` refuses.
    pub has_payloads: bool,
    pub terms: &'a [T],
}

impl<T> Clone for IdVersionFieldInput<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for IdVersionFieldInput<'_, T> {}

/// The group's two files.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdVersionOutput {
    /// `.tiv`: the blocks and the field directory.
    pub tiv: Vec<u8>,
    /// `.tipv`: each field's FST terms index.
    pub tipv: Vec<u8>,
}

/// `VersionBlockTreeTermsWriter.encodeOutput`.
// ARITH: a file pointer of an in-memory file, far below 2^61.
#[allow(clippy::arithmetic_side_effects)]
fn encode_output(fp: i64, has_terms: bool, is_floor: bool) -> i64 {
    (fp << OUTPUT_FLAGS_NUM_BITS)
        | if has_terms { OUTPUT_FLAG_HAS_TERMS } else { 0 }
        | if is_floor { OUTPUT_FLAG_IS_FLOOR } else { 0 }
}

struct PendingTerm {
    term_bytes: Vec<u8>,
    doc_id: i32,
    id_version: i64,
}

/// The `(input, output)` pairs of a block's FST, in input order.
type IndexEntries = Vec<(Vec<u8>, IndexOutput)>;

struct PendingBlock {
    prefix: Vec<u8>,
    fp: i64,
    /// `index`: the entries of this block's FST once compiled.
    index: IndexEntries,
    /// `subIndices`: the entries of the sub-blocks' FSTs.
    sub_indices: Option<Vec<IndexEntries>>,
    has_terms: bool,
    is_floor: bool,
    floor_lead_byte: i32,
    max_version: i64,
}

enum PendingEntry {
    Term(PendingTerm),
    Block(PendingBlock),
}

/// `FieldMetaData`.
struct FieldMetaData {
    field_number: i32,
    root_code: IndexOutput,
    num_terms: i64,
    index_start_fp: i64,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
}

/// `VersionBlockTreeTermsWriter`'s state across the group's fields.
struct Writer {
    out: Vec<u8>,
    index_out: Vec<u8>,
    min_items_in_block: usize,
    max_items_in_block: usize,
    fields: Vec<FieldMetaData>,
    /// `IDVersionPostingsWriter.lastEncodedVersion`.
    last_encoded_version: i64,
}

/// `VersionBlockTreeTermsWriter`: writes every field of `inputs` (in the
/// order given: the group's name order) and closes, returning the two files.
/// `live_docs` is `SegmentWriteState.liveDocs` -- a flush's deletions, which
/// `IDVersionPostingsWriter.startDoc` skips; `None` keeps every document (a
/// merge).
///
/// # Errors
/// `IllegalArgument` with `IDVersionPostingsWriter`'s messages: a field not
/// indexed with exactly `DOCS_AND_FREQS_AND_POSITIONS`, or with term
/// vectors; a term in two live documents, or twice in one; an occurrence
/// without an 8-byte payload, or whose version is out of range.
pub fn write_fields<T: TermSource>(
    inputs: &[IdVersionFieldInput<'_, T>],
    live_docs: Option<&dyn Fn(i32) -> bool>,
    format: &IdVersionPostingsFormat,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<IdVersionOutput, Error> {
    let mut w = Writer {
        out: Vec::new(),
        index_out: Vec::new(),
        min_items_in_block: format.min_terms_in_block(),
        max_items_in_block: format.max_terms_in_block(),
        fields: Vec::new(),
        last_encoded_version: 0,
    };
    codec_util::write_index_header(
        &mut w.out,
        TERMS_CODEC_NAME,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    codec_util::write_index_header(
        &mut w.index_out,
        TERMS_INDEX_CODEC_NAME,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    // `IDVersionPostingsWriter.init`.
    codec_util::write_index_header(
        &mut w.out,
        POSTINGS_TERMS_CODEC,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    for input in inputs {
        let mut tw = TermsWriter::new(input)?;
        for term in input.terms {
            tw.write(&mut w, term, live_docs)?;
        }
        tw.finish(&mut w)?;
    }
    Ok(w.close())
}

impl Writer {
    /// `close()`: the field directory, the trailers and the footers.
    fn close(mut self) -> IdVersionOutput {
        let dir_start = self.out.len() as i64;
        let index_dir_start = self.index_out.len() as i64;
        self.out.write_vint(self.fields.len() as i32);
        for field in &self.fields {
            self.out.write_vint(field.field_number);
            self.out.write_vlong(field.num_terms);
            self.out.write_vint(field.root_code.first.len() as i32);
            self.out.write_bytes(&field.root_code.first);
            self.out.write_vlong(field.root_code.second);
            self.index_out.write_vlong(field.index_start_fp);
            self.out.write_vint(field.min_term.len() as i32);
            self.out.write_bytes(&field.min_term);
            self.out.write_vint(field.max_term.len() as i32);
            self.out.write_bytes(&field.max_term);
        }
        self.out.write_bytes(&dir_start.to_le_bytes());
        codec_util::write_footer(&mut self.out);
        self.index_out.write_bytes(&index_dir_start.to_le_bytes());
        codec_util::write_footer(&mut self.index_out);
        IdVersionOutput {
            tiv: self.out,
            tipv: self.index_out,
        }
    }

    /// `IDVersionPostingsWriter.encodeTerm`.
    fn encode_term(&mut self, meta: &mut Vec<u8>, term: &PendingTerm, absolute: bool) {
        meta.write_vint(term.doc_id);
        if absolute {
            meta.write_vlong(term.id_version);
        } else {
            meta.write_zlong(term.id_version.wrapping_sub(self.last_encoded_version));
        }
        self.last_encoded_version = term.id_version;
    }
}

/// `PendingBlock.compileIndex`: this block's entry -- its output and those
/// of its floor followers -- then its sub-blocks' entries.
// ARITH: file pointers and counts of in-memory files and lists.
#[allow(clippy::arithmetic_side_effects)]
fn compile_index(blocks: &mut [PendingBlock]) {
    let (first, rest) = blocks
        .split_first_mut()
        .expect("writeBlocks writes at least one block");
    let mut max_version_index = first.max_version;
    let mut scratch = Vec::new();
    scratch.write_vlong(encode_output(first.fp, first.has_terms, first.is_floor));
    if first.is_floor {
        scratch.write_vint(rest.len() as i32);
        for sub in rest.iter() {
            max_version_index = max_version_index.max(sub.max_version);
            scratch.write_byte(sub.floor_lead_byte as u8);
            scratch.write_vlong(((sub.fp - first.fp) << 1) | i64::from(sub.has_terms));
        }
    }
    let mut entries = vec![(
        first.prefix.clone(),
        Pair {
            first: scratch,
            second: i64::MAX - max_version_index,
        },
    )];
    for block in std::iter::once(&mut *first).chain(rest.iter_mut()) {
        if let Some(sub_indices) = block.sub_indices.take() {
            for sub in sub_indices {
                entries.extend(sub);
            }
        }
    }
    first.index = entries;
}

/// `TermsWriter`: one field.
struct TermsWriter {
    field_number: i32,
    has_payloads: bool,
    num_terms: i64,
    last_term: Vec<u8>,
    /// `prefixStarts`: Java's `int[]`, which goes negative after a block
    /// is written (that slot is rewritten before it is read again).
    prefix_starts: Vec<i64>,
    pending: Vec<PendingEntry>,
    new_blocks: Vec<PendingBlock>,
    first_pending_term: Option<Vec<u8>>,
    last_pending_term: Option<Vec<u8>>,
}

impl TermsWriter {
    /// `new TermsWriter(fieldInfo)`: `IDVersionPostingsWriter.setField`'s
    /// checks.
    fn new<T>(input: &IdVersionFieldInput<'_, T>) -> Result<Self, Error> {
        if input.index_options != IndexOptions::DocsAndFreqsAndPositions {
            return Err(Error::IllegalArgument(
                "field must be index using IndexOptions.DOCS_AND_FREQS_AND_POSITIONS".into(),
            ));
        }
        if input.store_term_vectors {
            return Err(Error::IllegalArgument(
                "field cannot index term vectors: CheckIndex will report this as index corruption"
                    .into(),
            ));
        }
        Ok(TermsWriter {
            field_number: input.field_number,
            has_payloads: input.has_payloads,
            num_terms: 0,
            last_term: Vec::new(),
            prefix_starts: Vec::new(),
            pending: Vec::new(),
            new_blocks: Vec::new(),
            first_pending_term: None,
            last_pending_term: None,
        })
    }

    /// `write(text, termsEnum, norms)`: `PushPostingsWriterBase.writeTerm`
    /// through `IDVersionPostingsWriter`, then the term is pushed when it has
    /// a live document.
    // ARITH: counts of in-memory lists.
    #[allow(clippy::arithmetic_side_effects)]
    fn write<T: TermSource>(
        &mut self,
        w: &mut Writer,
        term: &T,
        live_docs: Option<&dyn Fn(i32) -> bool>,
    ) -> Result<(), Error> {
        let mut last: Option<(i32, i64)> = None;
        let lengths = term.payload_lengths();
        let bytes = term.payload_bytes();
        let (mut occurrence, mut byte_at) = (0usize, 0usize);
        for (doc_index, &(doc_id, freq)) in term.docs().iter().enumerate() {
            let positions = term.positions(doc_index);
            // The occurrences' payloads are one run over every document.
            let occurrences = positions.len();
            let skip = |occ: &mut usize, at: &mut usize| {
                for k in 0..occurrences {
                    *at += lengths.get(*occ + k).map_or(0, |&l| l as usize);
                }
                *occ += occurrences;
            };
            // `startDoc`.
            if live_docs.is_some_and(|live| !live(doc_id)) {
                skip(&mut occurrence, &mut byte_at);
                continue;
            }
            if let Some((last_doc, _)) = last {
                return Err(Error::IllegalArgument(format!(
                    "term appears in more than one document: {last_doc} and {doc_id}"
                )));
            }
            if freq != 1 {
                return Err(Error::IllegalArgument(
                    "term appears more than once in the document".into(),
                ));
            }
            // `finishDoc` refuses a document `addPosition` never saw; Java's
            // `writeTerm` always calls it for a positions field, so only a
            // caller without positions gets here.
            if positions.is_empty() {
                return Err(Error::IllegalArgument("missing addPosition".into()));
            }
            // `addPosition`, once (freq is 1).
            let len = if self.has_payloads {
                lengths.get(occurrence).map_or(0, |&l| l as usize)
            } else {
                0
            };
            if len == 0 {
                return Err(Error::IllegalArgument(
                    "token doens't have a payload".into(),
                ));
            }
            if len != 8 {
                return Err(Error::IllegalArgument(format!(
                    "payload.length != 8 (got {len})"
                )));
            }
            let payload: [u8; 8] = bytes
                .get(byte_at..byte_at + 8)
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| Error::IllegalArgument("payload run is short".into()))?;
            let version = i64::from_be_bytes(payload);
            if version < MIN_VERSION {
                return Err(Error::IllegalArgument(format!(
                    "version must be >= MIN_VERSION={MIN_VERSION} (got: {version}; payload={})",
                    bytes_ref_string(&payload)
                )));
            }
            if version > MAX_VERSION {
                return Err(Error::IllegalArgument(format!(
                    "version must be <= MAX_VERSION={MAX_VERSION} (got: {version}; payload={})",
                    bytes_ref_string(&payload)
                )));
            }
            skip(&mut occurrence, &mut byte_at);
            last = Some((doc_id, version));
        }
        let Some((doc_id, id_version)) = last else {
            return Ok(());
        };
        self.push_term(w, term.term())?;
        let pending = PendingTerm {
            term_bytes: term.term().to_vec(),
            doc_id,
            id_version,
        };
        if self.first_pending_term.is_none() {
            self.first_pending_term = Some(pending.term_bytes.clone());
        }
        self.last_pending_term = Some(pending.term_bytes.clone());
        self.pending.push(PendingEntry::Term(pending));
        self.num_terms += 1;
        Ok(())
    }

    /// `pushTerm(text)`: writes the blocks the new term closes.
    // ARITH: indices into `last_term`/`prefix_starts`, bounded by their
    // lengths; counts of `pending` in i64.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_term(&mut self, w: &mut Writer, text: &[u8]) -> Result<(), Error> {
        let limit = self.last_term.len().min(text.len());
        let mut pos = 0;
        while pos < limit && self.last_term[pos] == text[pos] {
            pos += 1;
        }
        for i in (pos..self.last_term.len()).rev() {
            let prefix_top_size = self.pending.len() as i64 - self.prefix_starts[i];
            if prefix_top_size >= w.min_items_in_block as i64 {
                self.write_blocks(w, i + 1, prefix_top_size as usize)?;
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
        Ok(())
    }

    /// `writeBlocks(prefixLength, count)`: the top `count` entries become
    /// one block, or several floor blocks, replaced by the first of them.
    // ARITH: indices into `pending`, bounded by its length.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_blocks(
        &mut self,
        w: &mut Writer,
        prefix_length: usize,
        count: usize,
    ) -> Result<(), Error> {
        let mut last_suffix_lead_label: i32 = -1;
        let mut has_terms = false;
        let mut has_sub_blocks = false;
        let start = self.pending.len() - count;
        let end = self.pending.len();
        let mut next_block_start = start;
        let mut next_floor_lead_label: i32 = -1;
        for i in start..end {
            let suffix_lead_label = match &self.pending[i] {
                PendingEntry::Term(term) => {
                    if term.term_bytes.len() == prefix_length {
                        -1
                    } else {
                        i32::from(term.term_bytes[prefix_length])
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
        Ok(())
    }

    /// `writeBlock(...)`: entries `start..end` of `pending` as one block.
    // ARITH: lengths and file pointers of in-memory buffers; suffix lengths
    // are differences of a longer term's length and a shared prefix's.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        w: &mut Writer,
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
        let mut suffix_writer = Vec::new();
        let mut meta_writer = Vec::new();
        let mut absolute = true;
        let mut max_version_in_block: i64 = -1;
        let mut sub_indices = if is_leaf_block {
            None
        } else {
            Some(Vec::new())
        };
        for i in start..end {
            match &mut self.pending[i] {
                PendingEntry::Term(term) => {
                    max_version_in_block = max_version_in_block.max(term.id_version);
                    let suffix = term.term_bytes.len() - prefix_length;
                    if is_leaf_block {
                        suffix_writer.write_vint(suffix as i32);
                    } else {
                        suffix_writer.write_vint((suffix as i32) << 1);
                    }
                    suffix_writer.write_bytes(&term.term_bytes[prefix_length..]);
                    w.encode_term(&mut meta_writer, term, absolute);
                    absolute = false;
                }
                PendingEntry::Block(block) => {
                    max_version_in_block = max_version_in_block.max(block.max_version);
                    let suffix = block.prefix.len() - prefix_length;
                    suffix_writer.write_vint(((suffix as i32) << 1) | 1);
                    suffix_writer.write_bytes(&block.prefix[prefix_length..]);
                    suffix_writer.write_vlong(start_fp - block.fp);
                    if let Some(subs) = sub_indices.as_mut() {
                        subs.push(std::mem::take(&mut block.index));
                    }
                }
            }
        }
        w.out
            .write_vint(((suffix_writer.len() as i32) << 1) | i32::from(is_leaf_block));
        w.out.write_bytes(&suffix_writer);
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
            max_version: max_version_in_block,
        }
    }

    /// `finish()`: the root block, its FST into `.tipv`, the field's record.
    fn finish(mut self, w: &mut Writer) -> Result<(), Error> {
        if self.num_terms == 0 {
            return Ok(());
        }
        let count = self.pending.len();
        self.write_blocks(w, 0, count)?;
        let Some(PendingEntry::Block(root)) = self.pending.pop() else {
            unreachable!("writeBlocks(0, all) leaves the root block alone");
        };
        let index_start_fp = w.index_out.len() as i64;
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<IndexOutputs>();
        let mut root_code = None;
        for (input, output) in root.index {
            if input.is_empty() {
                root_code = Some(output.clone());
            }
            compiler
                .add_bytes(&input, output)
                .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        }
        let fst = compiler
            .compile()
            .expect("the root FST has the empty input");
        w.index_out.write_bytes(&fst.save::<IndexOutputs>());
        w.fields.push(FieldMetaData {
            field_number: self.field_number,
            root_code: root_code.expect("the root block's prefix is empty"),
            num_terms: self.num_terms,
            index_start_fp,
            min_term: self.first_pending_term.unwrap_or_default(),
            max_term: self.last_pending_term.unwrap_or_default(),
        });
        Ok(())
    }
}

/// `BytesRef.toString()`: `[hex bytes]`.
fn bytes_ref_string(bytes: &[u8]) -> String {
    let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
    format!("[{}]", hex.join(" "))
}
