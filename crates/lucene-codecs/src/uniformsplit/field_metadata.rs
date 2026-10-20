//! Port of `FieldMetadata` and `FieldMetadata.Serializer`: a field's
//! statistics, where its dictionary and blocks are, and its last term.

use lucene_store::data_output::DataOutput;

use super::{corrupt, length, read_slice, read_vint, read_vlong, Result};
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};

/// `indexOptions.subsumes(DOCS_AND_FREQS)`.
pub(crate) fn has_freqs(index_options: IndexOptions) -> bool {
    !matches!(index_options, IndexOptions::None | IndexOptions::Docs)
}

/// `FieldMetadata`.
#[derive(Debug, Clone)]
pub struct FieldMetadata {
    /// `fieldInfo`.
    pub(crate) field_info: FieldInfo,
    /// `sumDocFreq`.
    pub(crate) sum_doc_freq: i64,
    /// `numTerms`.
    pub(crate) num_terms: i64,
    /// `sumTotalTermFreq` (`sumDocFreq` read back for a field without
    /// frequencies).
    pub(crate) sum_total_term_freq: i64,
    /// `docCount` (Java's writer: `docsSeen.cardinality()`).
    pub(crate) doc_count: i32,
    /// `dictionaryStartFP`.
    pub(crate) dictionary_start_fp: i64,
    /// `firstBlockStartFP`.
    pub(crate) first_block_start_fp: i64,
    /// `lastBlockStartFP`.
    pub(crate) last_block_start_fp: i64,
    /// `lastTerm`.
    pub(crate) last_term: Vec<u8>,
}

impl FieldMetadata {
    /// `new FieldMetadata(fieldInfo, maxDoc)`: the writer's, every pointer
    /// `-1` until set.
    pub(crate) fn new(field_info: FieldInfo, doc_count: i32) -> Self {
        FieldMetadata {
            field_info,
            sum_doc_freq: 0,
            num_terms: 0,
            sum_total_term_freq: 0,
            doc_count,
            dictionary_start_fp: -1,
            first_block_start_fp: -1,
            last_block_start_fp: -1,
            last_term: Vec::new(),
        }
    }

    /// `updateStats(state)`: `totalTermFreq` only counts when positive (a
    /// field without frequencies has `-1`).
    pub(crate) fn update_stats(&mut self, doc_freq: i32, total_term_freq: i64) {
        self.sum_doc_freq = self.sum_doc_freq.wrapping_add(i64::from(doc_freq));
        if total_term_freq > 0 {
            self.sum_total_term_freq = self.sum_total_term_freq.wrapping_add(total_term_freq);
        }
        self.num_terms = self.num_terms.wrapping_add(1);
    }

    /// `getFieldInfo()`.
    pub fn field_info(&self) -> &FieldInfo {
        &self.field_info
    }

    /// `getNumTerms()`.
    pub fn num_terms(&self) -> i64 {
        self.num_terms
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// `getDictionaryStartFP()`.
    pub fn dictionary_start_fp(&self) -> i64 {
        self.dictionary_start_fp
    }

    /// `getFirstBlockStartFP()`.
    pub fn first_block_start_fp(&self) -> i64 {
        self.first_block_start_fp
    }

    /// `getLastBlockStartFP()`.
    pub fn last_block_start_fp(&self) -> i64 {
        self.last_block_start_fp
    }

    /// `getLastTerm()`.
    pub fn last_term(&self) -> &[u8] {
        &self.last_term
    }

    /// `Serializer.write(output, fieldMetadata)`.
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.write_vint(self.field_info.number);
        out.write_vlong(self.num_terms);
        out.write_vlong(self.sum_doc_freq);
        if has_freqs(self.field_info.index_options) {
            out.write_vlong(self.sum_total_term_freq.wrapping_sub(self.sum_doc_freq));
        }
        out.write_vint(self.doc_count);
        out.write_vlong(self.dictionary_start_fp);
        out.write_vlong(self.first_block_start_fp);
        out.write_vlong(self.last_block_start_fp);
        out.write_vint(self.last_term.len() as i32);
        out.write_bytes(&self.last_term);
    }

    /// `Serializer.read(input, fieldInfos, maxNumDocs)` at `*pos`.
    ///
    /// # Errors
    /// `Corrupted` for a field number `field_infos` lacks and the statistics
    /// Java refuses (no terms, `sumTotalTermFreq < sumDocFreq`, a `docCount`
    /// outside `0..=maxNumDocs` or above `sumDocFreq`, a negative last term
    /// length), and metadata past the end of its input.
    pub(crate) fn read(
        buf: &[u8],
        pos: &mut usize,
        field_infos: &FieldInfos,
        max_num_docs: i32,
    ) -> Result<Self> {
        let field_id = read_vint(buf, pos)?;
        let field_info = field_infos
            .field_by_number(field_id)
            .ok_or_else(|| corrupt(format!("Illegal field id= {field_id}")))?
            .clone();
        let num_terms = read_vlong(buf, pos)?;
        if num_terms <= 0 {
            return Err(corrupt(format!(
                "Illegal number of terms= {num_terms} for field= {field_id}"
            )));
        }
        let sum_doc_freq = read_vlong(buf, pos)?;
        let mut sum_total_term_freq = sum_doc_freq;
        if has_freqs(field_info.index_options) {
            sum_total_term_freq = sum_total_term_freq.wrapping_add(read_vlong(buf, pos)?);
            if sum_total_term_freq < sum_doc_freq {
                return Err(corrupt(format!(
                    "Illegal sumTotalTermFreq= {sum_total_term_freq} sumDocFreq= \
                     {sum_doc_freq} for field= {field_id}"
                )));
            }
        }
        let doc_count = read_vint(buf, pos)?;
        if doc_count < 0 || doc_count > max_num_docs {
            return Err(corrupt(format!(
                "Illegal number of docs= {doc_count} maxNumDocs= {max_num_docs} for \
                 field={field_id}"
            )));
        }
        if sum_doc_freq < i64::from(doc_count) {
            return Err(corrupt(format!(
                "Illegal sumDocFreq= {sum_doc_freq} docCount= {doc_count} for field= {field_id}"
            )));
        }
        let dictionary_start_fp = read_vlong(buf, pos)?;
        let first_block_start_fp = read_vlong(buf, pos)?;
        let last_block_start_fp = read_vlong(buf, pos)?;
        let last_term_length = read_vint(buf, pos)?;
        if last_term_length < 0 {
            return Err(corrupt(format!(
                "Illegal last term length= {last_term_length} for field= {field_id}"
            )));
        }
        let last_term = read_slice(buf, pos, length(i64::from(last_term_length))?)?.to_vec();
        Ok(FieldMetadata {
            field_info,
            sum_doc_freq,
            num_terms,
            sum_total_term_freq,
            doc_count,
            dictionary_start_fp,
            first_block_start_fp,
            last_block_start_fp,
            last_term,
        })
    }
}
