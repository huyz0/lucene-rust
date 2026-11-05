//! A searchable view of documents still in the writer's buffer: the postings
//! of the fields and terms one search needs, encoded with the codec's own
//! writers into an in-memory segment.
//!
//! The scorer tree, MaxScore/WAND and the collectors run on codec structures
//! ([`OpenSegment`]), so the buffered documents are encoded rather than read
//! through an interface of their own: `.doc`/`.pos`/`.pay`, the terms
//! dictionary and `.nvd` bytes, opened in place with no directory, the way
//! [`crate::memory`] encodes its one document. Only the terms a search asks
//! for are encoded, so building one costs what those terms' postings cost.
//!
//! The field statistics BM25 reads (`docCount`, `sumTotalTermFreq`,
//! `sumDocFreq`, the number of terms) are the whole buffer's, given by the
//! caller, not the encoded subset's: the dictionary's are replaced after it
//! is opened, so a document scores as it will once flushed.

use std::collections::HashMap;

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::norms::{self, NormsEntry, NormsField};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::field_norms::FieldNorms;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// The id the in-memory files are written and checked with.
const SEGMENT_ID: [u8; 16] = [0; 16];

fn illegal(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(format!("live segment: {e}"))
}

/// One field of a [`LiveSegment`]: its schema, the terms to encode, its
/// norms and the whole buffer's statistics for it.
///
/// Encoding positions costs several times what documents and frequencies
/// do, so a search that reads no positions (no phrase, span or interval
/// clause on the field) should pass the field as `DocsAndFreqs` without
/// them: scores do not depend on positions.
#[derive(Debug, Clone)]
pub struct LiveField {
    /// The field's schema; `store_payloads` says whether its terms carry
    /// payloads.
    pub info: FieldInfo,
    /// The terms to encode, in term order, each with its documents
    /// ascending and below the segment's `max_doc`. May be empty: the field
    /// then has no terms dictionary (the codec writes none for a field with
    /// no terms), and only [`LiveSegment::field_stats`] reports it.
    pub terms: Vec<TermPostings>,
    /// Each document's norm, indexed by doc id (`max_doc` entries); `None`
    /// for a field without norms. A document without the field may hold any
    /// value: it is in no posting list.
    pub norms: Option<Vec<i64>>,
    /// The buffer's number of distinct terms in the field.
    pub num_terms: i64,
    /// The buffer's `sumTotalTermFreq` for the field.
    pub sum_total_term_freq: i64,
    /// The buffer's `sumDocFreq` for the field.
    pub sum_doc_freq: i64,
    /// The number of buffered documents with the field.
    pub doc_count: i32,
}

/// One field's norms: its `.nvm` entry and the statistics its BM25 average
/// length comes from.
#[derive(Debug)]
struct NormsOf {
    field: String,
    entry: NormsEntry,
    sum_total_term_freq: i64,
    doc_count: i32,
}

/// See the module documentation.
#[derive(Debug)]
pub struct LiveSegment {
    terms: BlockTreeFields,
    /// Every field's `[numTerms, sumTotalTermFreq, sumDocFreq, docCount]`.
    field_stats: HashMap<String, [i64; 4]>,
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    nvd: Vec<u8>,
    norms: Vec<NormsOf>,
    live_docs: Option<FixedBitSet>,
    max_doc: i32,
}

/// The opened postings files of a [`LiveSegment`], which an [`OpenSegment`]
/// borrows.
pub struct LiveInputs<'a> {
    doc: Option<DocInput<'a>>,
    pos: Option<PosInput<'a>>,
    pay: Option<PayInput<'a>>,
}

impl LiveSegment {
    /// Encodes `fields` for documents `0..max_doc`, of which `deleted` are
    /// deleted.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the postings are malformed (terms out
    /// of order, a document at or past `max_doc`, norms of the wrong length)
    /// or the codec refuses them.
    pub fn build(fields: &[LiveField], max_doc: i32, deleted: &[i32]) -> Result<Self> {
        let n = usize::try_from(max_doc).map_err(|_| illegal("negative max_doc"))?;
        for f in fields {
            if f.norms.as_ref().is_some_and(|v| v.len() != n) {
                return Err(illegal(format!(
                    "field {} has norms for other than {n} documents",
                    f.info.name
                )));
            }
            for pair in f.terms.windows(2) {
                if pair[0].term >= pair[1].term {
                    return Err(illegal(format!(
                        "field {} has terms out of order",
                        f.info.name
                    )));
                }
            }
            for t in &f.terms {
                if t.docs.iter().any(|&(d, _)| d < 0 || d >= max_doc) {
                    return Err(illegal(format!(
                        "field {} has a document out of range",
                        f.info.name
                    )));
                }
            }
        }
        let infos: Vec<FieldInfo> = fields.iter().map(|f| f.info.clone()).collect();
        let field_infos = FieldInfos::new(infos).map_err(illegal)?;
        // The dictionary is written with the encoded terms' own document
        // count, which its reader checks against their postings, and given the
        // whole field's after it is opened.
        let encoded_doc_counts: Vec<i32> = fields
            .iter()
            .map(|f| {
                let mut seen = FixedBitSet::new(n);
                let mut count = 0i32;
                for t in &f.terms {
                    for &(d, _) in &t.docs {
                        let d = d as usize;
                        if d < seen.len() && !seen.get(d) {
                            seen.set(d);
                            count = count.saturating_add(1);
                        }
                    }
                }
                count
            })
            .collect();
        let inputs: Vec<FieldPostingsInput<'_>> = fields
            .iter()
            .zip(&encoded_doc_counts)
            .filter(|(f, _)| !f.terms.is_empty())
            .map(|(f, &doc_count)| FieldPostingsInput {
                field_number: f.info.number,
                index_options: f.info.index_options,
                doc_count,
                has_payloads: f.info.store_payloads,
                terms: &f.terms,
            })
            .collect();
        let impacts: Vec<postings_writer::FieldNorms<'_>> = fields
            .iter()
            .filter_map(|f| {
                f.norms.as_ref().map(|v| postings_writer::FieldNorms {
                    field_number: f.info.number,
                    values: v,
                })
            })
            .collect();
        let (mut terms, doc, pos, pay) = if inputs.is_empty() {
            (BlockTreeFields::empty(), Vec::new(), Vec::new(), Vec::new())
        } else {
            let output =
                postings_writer::write_fields_with_norms(&inputs, &impacts, &SEGMENT_ID, "")
                    .map_err(illegal)?;
            let terms = blocktree::open_shared(
                std::sync::Arc::new(output.tim),
                std::sync::Arc::new(output.tip),
                &output.tmd,
                &field_infos,
                &SEGMENT_ID,
                "",
                max_doc,
            )?;
            (terms, output.doc, output.pos, output.pay)
        };
        for f in fields {
            terms.set_field_stats(
                &f.info.name,
                f.num_terms,
                f.sum_total_term_freq,
                f.sum_doc_freq,
                f.doc_count,
            );
        }
        let columns: Vec<NormsField<'_>> = fields
            .iter()
            .filter_map(|f| {
                f.norms
                    .as_ref()
                    .map(|v| NormsField::Dense(f.info.number, v))
            })
            .collect();
        let (nvd, norms) = if columns.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let (nvm, nvd) =
                norms::write_fields(&columns, max_doc, &SEGMENT_ID, "").map_err(illegal)?;
            let (_, meta) = norms::parse_meta(&nvm, &SEGMENT_ID, "")?;
            let norms = fields
                .iter()
                .filter(|f| f.norms.is_some())
                .filter_map(|f| {
                    meta.entry(f.info.number).map(|entry| NormsOf {
                        field: f.info.name.clone(),
                        entry: *entry,
                        sum_total_term_freq: f.sum_total_term_freq,
                        doc_count: f.doc_count,
                    })
                })
                .collect();
            (nvd, norms)
        };
        let live_docs = if deleted.is_empty() {
            None
        } else {
            let mut bits = FixedBitSet::new(n);
            for d in 0..bits.len() {
                bits.set(d);
            }
            for &d in deleted {
                if let Ok(d) = usize::try_from(d) {
                    if d < bits.len() {
                        bits.clear(d);
                    }
                }
            }
            Some(bits)
        };
        let field_stats = fields
            .iter()
            .map(|f| {
                (
                    f.info.name.clone(),
                    [
                        f.num_terms,
                        f.sum_total_term_freq,
                        f.sum_doc_freq,
                        i64::from(f.doc_count),
                    ],
                )
            })
            .collect();
        Ok(LiveSegment {
            terms,
            field_stats,
            doc,
            pos,
            pay,
            nvd,
            norms,
            live_docs,
            max_doc,
        })
    }

    /// The number of documents, deleted ones included.
    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// The terms dictionary.
    pub fn fields(&self) -> &BlockTreeFields {
        &self.terms
    }

    /// Field `name`'s `[numTerms, sumTotalTermFreq, sumDocFreq, docCount]`
    /// over the whole buffer, also for a field with no encoded term.
    pub fn field_stats(&self, name: &str) -> Option<[i64; 4]> {
        self.field_stats.get(name).copied()
    }

    /// Which documents are live, `None` when none is deleted.
    pub fn live_docs(&self) -> Option<&FixedBitSet> {
        self.live_docs.as_ref()
    }

    /// Opens the postings files for [`Self::segment`].
    ///
    /// # Errors
    /// Whatever opening the encoded files reports (never, for files this
    /// segment wrote).
    pub fn inputs(&self) -> Result<LiveInputs<'_>> {
        Ok(LiveInputs {
            doc: match self.doc.is_empty() {
                true => None,
                false => Some(DocInput::open(&self.doc, &SEGMENT_ID, "")?),
            },
            pos: match self.pos.is_empty() {
                true => None,
                false => Some(PosInput::open(&self.pos, &SEGMENT_ID, "")?),
            },
            pay: match self.pay.is_empty() {
                true => None,
                false => Some(PayInput::open(&self.pay, &SEGMENT_ID, "")?),
            },
        })
    }

    /// The segment as a search reads it, its first document at `doc_base`.
    pub fn segment<'a>(&'a self, inputs: &'a LiveInputs<'a>, doc_base: i32) -> OpenSegment<'a> {
        OpenSegment {
            fields: &self.terms,
            doc_in: inputs.doc.as_ref(),
            pos_in: inputs.pos.as_ref(),
            pay_in: inputs.pay.as_ref(),
            live_docs: self.live_docs.as_ref(),
            doc_base,
            max_doc: Some(self.max_doc),
            cache: None,
            points: None,
            reader: None,
            index_sort_prefix: false,
            similarity: None,
        }
    }

    /// Each field's norms, by field name, as a search scores with them.
    pub fn norms(&self) -> HashMap<String, FieldNorms<'_>> {
        self.norms
            .iter()
            .map(|n| {
                (
                    n.field.clone(),
                    FieldNorms::from_field_stats(
                        &self.nvd,
                        n.entry,
                        n.sum_total_term_freq,
                        n.doc_count,
                    ),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
