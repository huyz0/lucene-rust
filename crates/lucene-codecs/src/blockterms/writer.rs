//! Port of `BlockTermsWriter` and its `TermsWriter`: each field's terms
//! gathered until the terms index picks the next index term, then written as
//! one block -- the suffixes against the block's common prefix, the
//! statistics, the postings metadata -- and the fields' summaries at the
//! end of the `.tib`.

use lucene_store::codec_util;
use lucene_store::data_output::DataOutput;

use super::{Error, Result, TermStats, TermsIndexWriterBase, CODEC_NAME, VERSION_CURRENT};
use crate::field_infos::IndexOptions;
use crate::postings_writer::{self, FieldTermStates};
use crate::segment_state::SegmentWriteState;

/// `BlockTermsWriter.FieldMetaData`.
#[derive(Debug, Clone, Copy)]
struct FieldMetaData {
    number: i32,
    index_options: IndexOptions,
    num_terms: i64,
    terms_start_pointer: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
}

/// `BlockTermsWriter`: the `.tib` of one group, filled field by field from
/// the postings writer's term states, and the terms index beside it.
pub struct BlockTermsWriter<'s, I: TermsIndexWriterBase> {
    state: &'s SegmentWriteState<'s>,
    /// `out`.
    out: Vec<u8>,
    terms_index_writer: I,
    fields: Vec<FieldMetaData>,
}

impl<I: TermsIndexWriterBase> std::fmt::Debug for BlockTermsWriter<'_, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockTermsWriter")
            .field("fields", &self.fields.len())
            .field("tib_len", &self.out.len())
            .finish()
    }
}

/// `TermsWriter.sharedPrefix(term1, term2)`.
fn shared_prefix(term1: &[u8], term2: &[u8]) -> usize {
    term1
        .iter()
        .zip(term2)
        .position(|(a, b)| a != b)
        .unwrap_or(term1.len().min(term2.len()))
}

/// A length as Java's `int` (`Math.toIntExact`).
fn to_int(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| Error::IllegalArgument(format!("integer overflow: {n}")))
}

impl<'s, I: TermsIndexWriterBase> BlockTermsWriter<'s, I> {
    /// The constructor: the `.tib`'s header, then `postingsWriter.init`'s.
    pub fn new(terms_index_writer: I, state: &'s SegmentWriteState<'s>) -> Self {
        let mut out = Vec::new();
        codec_util::write_index_header(
            &mut out,
            CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        postings_writer::write_postings_terms_header(
            &mut out,
            state.segment_id,
            state.segment_suffix,
        );
        BlockTermsWriter {
            state,
            out,
            terms_index_writer,
            fields: Vec::new(),
        }
    }

    /// `write(fields, norms)` for one field (`addField`, `TermsWriter.write`
    /// per term, `TermsWriter.finish`), over its terms as the postings writer
    /// laid them out.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a field number the segment's
    /// `FieldInfos` lacks or terms out of order; what the terms index fails
    /// with.
    // ARITH: the sums add a field's `docFreq`s/`totalTermFreq`s, which the
    // postings writer bounded by the documents it wrote; `num_terms` counts
    // the field's terms.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn write_field(&mut self, field: &FieldTermStates<'_>) -> Result<()> {
        let field_info = self.state.field_info(field.field_number).ok_or_else(|| {
            Error::IllegalArgument(format!("unknown field number {}", field.field_number))
        })?;
        let index_options = field_info.index_options;
        // `addField(field)` and the `TermsWriter` constructor.
        let terms_start_pointer = self.out.len() as i64;
        self.terms_index_writer
            .add_field(field_info, terms_start_pointer)?;
        let mut pending: Vec<usize> = Vec::with_capacity(32);
        let mut last_prev_term: &[u8] = &[];
        let (mut num_terms, mut sum_doc_freq, mut sum_total_term_freq) = (0i64, 0i64, 0i64);
        for i in 0..field.len() {
            // `TermsWriter.write(text, termsEnum, norms)`.
            let text = field.term(i);
            let stats = TermStats {
                doc_freq: field.doc_freq(i),
                total_term_freq: field.total_term_freq(i),
            };
            sum_doc_freq += i64::from(stats.doc_freq);
            sum_total_term_freq += stats.total_term_freq;
            if self.terms_index_writer.check_index_term(text, stats) {
                if !pending.is_empty() {
                    last_prev_term = self.flush_block(field, &pending, last_prev_term)?;
                    pending.clear();
                }
                let fp = self.out.len() as i64;
                self.terms_index_writer.add(text, stats, fp)?;
            }
            pending.push(i);
            num_terms += 1;
        }
        // `TermsWriter.finish()`.
        if !pending.is_empty() {
            self.flush_block(field, &pending, last_prev_term)?;
        }
        // EOF marker:
        self.out.write_vint(0);
        let fp = self.out.len() as i64;
        self.terms_index_writer.finish(fp)?;
        if num_terms > 0 {
            self.fields.push(FieldMetaData {
                number: field_info.number,
                index_options,
                num_terms,
                terms_start_pointer,
                sum_total_term_freq: if index_options == IndexOptions::Docs {
                    -1
                } else {
                    sum_total_term_freq
                },
                sum_doc_freq,
                doc_count: field.doc_count,
            });
        }
        Ok(())
    }

    /// `TermsWriter.flushBlock()`: the pending terms (indices into `field`)
    /// as one block; returns the block's last term, the next block's
    /// `lastPrevTerm`.
    ///
    /// # Errors
    /// A blob longer than an `int`.
    // ARITH: `common_prefix` is a shared prefix of each term, so no longer
    // than any of them.
    #[allow(clippy::arithmetic_side_effects)]
    fn flush_block<'f>(
        &mut self,
        field: &'f FieldTermStates<'_>,
        pending: &[usize],
        last_prev_term: &[u8],
    ) -> Result<&'f [u8]> {
        // First pass: compute common prefix for all terms in the block,
        // against term before first term in this block.
        let common_prefix = pending
            .iter()
            .map(|&i| shared_prefix(last_prev_term, field.term(i)))
            .min()
            .unwrap_or(0);
        self.out.write_vint(to_int(pending.len())?);
        self.out.write_vint(to_int(common_prefix)?);

        // 2nd pass: write suffixes, as separate byte[] blob.
        let mut bytes_writer: Vec<u8> = Vec::new();
        for &i in pending {
            let term = field.term(i);
            let suffix = term.len() - common_prefix;
            bytes_writer.write_vint(to_int(suffix)?);
            bytes_writer.write_bytes(&term[common_prefix..]);
        }
        self.out.write_vint(to_int(bytes_writer.len())?);
        self.out.write_bytes(&bytes_writer);
        bytes_writer.clear();

        // 3rd pass: write the freqs as byte[] blob.
        let docs_only = field.index_options == IndexOptions::Docs;
        for &i in pending {
            let doc_freq = field.doc_freq(i);
            bytes_writer.write_vint(doc_freq);
            if !docs_only {
                bytes_writer.write_vlong(field.total_term_freq(i) - i64::from(doc_freq));
            }
        }
        self.out.write_vint(to_int(bytes_writer.len())?);
        self.out.write_bytes(&bytes_writer);
        bytes_writer.clear();

        // 4th pass: write the metadata, the first term absolute.
        field.encode_terms(&mut bytes_writer, pending);
        self.out.write_vint(to_int(bytes_writer.len())?);
        self.out.write_bytes(&bytes_writer);

        Ok(pending.last().map_or(&[][..], |&i| field.term(i)))
    }

    /// `close()`: the fields' summaries, the directory offset and the
    /// footer, then the terms index's `close()`. Returns the `.tib` and the
    /// index files.
    ///
    /// # Errors
    /// What the terms index fails with.
    pub fn close(mut self) -> Result<(Vec<u8>, I::Files)> {
        let dir_start = self.out.len() as i64;
        self.out.write_vint(to_int(self.fields.len())?);
        for field in &self.fields {
            self.out.write_vint(field.number);
            self.out.write_vlong(field.num_terms);
            self.out.write_vlong(field.terms_start_pointer);
            if field.index_options != IndexOptions::Docs {
                self.out.write_vlong(field.sum_total_term_freq);
            }
            self.out.write_vlong(field.sum_doc_freq);
            self.out.write_vint(field.doc_count);
        }
        // `writeTrailer(dirStart)`.
        self.out.write_i64(dir_start);
        codec_util::write_footer(&mut self.out);
        let index = self.terms_index_writer.close()?;
        Ok((self.out, index))
    }
}
