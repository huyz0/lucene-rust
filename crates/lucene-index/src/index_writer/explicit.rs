//! Documents whose indexing is already decided: the shape the OpenSearch
//! engine (M5) hands this writer.
//!
//! [`Document`] -- this writer's native input -- is a list of stored values,
//! and the writer's own configuration decides what else each one becomes: a
//! postings field is re-analysed from its stored text with this port's
//! standard analyzer, a doc-values or points field reads the same stored value.
//! That cannot express an OpenSearch document. There a `text` field is indexed
//! and never stored, is analysed by whatever analyzer its mapping names, and
//! `_seq_no` is a point and a doc value but no stored field at all.
//!
//! An [`ExplicitDocument`] carries what Lucene's `IndexingChain` itself
//! consumes from each `IndexableField`, already computed on the Java side:
//!
//! - **stored values**, and only those, written to `.fdt`;
//! - **inverted fields**: each term with its frequency, positions and
//!   offsets -- the output of the field's own analyzer, position-increment and
//!   offset gaps applied -- plus the field's **norm**, computed by the field's
//!   own `Similarity.computeNorm` (`0` for a field present with no tokens,
//!   exactly as `IndexingChain` records it);
//! - **doc values** (`numericValue()`/`binaryValue()`, one entry per value, so
//!   a `SORTED_NUMERIC`/`SORTED_SET` field repeats its field number);
//! - **points** (`binaryValue()`, the packed value).
//!
//! Field numbers are the writer's global ones, registered through
//! [`IndexWriter::register_field`] as the schema grows -- Java's
//! `FieldInfos.FieldNumbers`. A registered field's schema is fixed, as
//! Lucene's is; re-registering it with a different one is an error.
//!
//! # What a flush writes
//!
//! A segment lists **only the fields its documents carry** (Lucene's
//! per-segment `FieldInfos`), each with its full registered schema. A field's
//! postings-format attribute is written only when the segment holds terms for
//! it, and its doc-values attribute whenever it has a doc-values type -- the
//! same facts `PerFieldPostingsFormat`/`PerFieldDocValuesFormat` stamp. Every
//! format is written by the codec writers the rest of this module uses:
//! postings through [`postings_writer::write_fields_with_norms`] (so impacts
//! use the real norms), norms through [`norms::write_fields`], doc values and
//! points through [`IndexWriter::build_doc_values_output`] and
//! [`IndexWriter::build_points_output`] over one synthetic value list per
//! document.
//!
//! Not supported, and refused when a field is registered: term vectors,
//! payloads, KNN vectors, doc-values skip indexes. Index sorting is refused
//! when explicit documents are enabled.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    per_field_codec_suffix, DocValuesFieldConfig, DocumentBuffer, Error, IndexWriter,
    IndexingConfig, PointsFieldConfig, Result, DOC_VALUES_FORMAT_NAME, PER_FIELD_SUFFIX,
    POSTINGS_FORMAT_NAME,
};
use crate::segment_infos::SegmentCommitInfo;
use crate::segment_writer;
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::norms;
use lucene_codecs::postings_writer::{self, FieldPostingsInput, TermPostings};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;

/// One field's norms for [`IndexingConfig::build_and_write_explicit_segment`]:
/// `(field, dense column when every document has one, sparse pairs otherwise)`.
type NormsColumn = (i32, Option<Vec<i64>>, Vec<(i32, i64)>);

/// One document, indexing decided: see the module doc.
#[derive(Debug, Clone, Default)]
pub struct ExplicitDocument {
    /// Written to stored fields, nothing else.
    pub stored: Vec<StoredField>,
    /// Everything that is not a stored value.
    pub fields: ExplicitFields,
}

/// The non-stored half of an [`ExplicitDocument`].
#[derive(Debug, Clone, Default)]
pub struct ExplicitFields {
    pub inverted: Vec<InvertedField>,
    /// `FieldValue::Long` for `NUMERIC`/`SORTED_NUMERIC`, `FieldValue::Binary`
    /// for `BINARY`/`SORTED`/`SORTED_SET`.
    pub doc_values: Vec<StoredField>,
    /// `FieldValue::Binary`, `numDims * bytesPerDim` packed bytes.
    pub points: Vec<StoredField>,
}

/// One document's inverted occurrence of one field.
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedField {
    pub field_number: i32,
    /// Each distinct term once. Order does not matter.
    pub terms: Vec<InvertedTerm>,
    /// `Similarity.computeNorm` for this document, or `None` when the field
    /// omits norms.
    pub norm: Option<i64>,
}

/// One term of an [`InvertedField`].
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedTerm {
    pub term: Vec<u8>,
    /// Occurrences in this document; must equal `positions.len()` when the
    /// field indexes positions.
    pub freq: i32,
    /// Ascending, when the field indexes positions; empty otherwise.
    pub positions: Vec<i32>,
    /// Parallel to `positions`, when the field indexes offsets; empty
    /// otherwise.
    pub offsets: Vec<(i32, i32)>,
}

impl ExplicitFields {
    pub(crate) fn ram_bytes(&self) -> usize {
        let inverted: usize = self
            .inverted
            .iter()
            .flat_map(|f| &f.terms)
            .map(|t| {
                t.term
                    .capacity()
                    .saturating_add(t.positions.capacity().saturating_mul(4))
                    .saturating_add(t.offsets.capacity().saturating_mul(8))
                    .saturating_add(48)
            })
            .fold(0usize, usize::saturating_add);
        let values = |v: &[StoredField]| -> usize {
            v.iter()
                .map(|f| match &f.value {
                    FieldValue::Binary(b) => b.capacity().saturating_add(32),
                    _ => 32,
                })
                .fold(0usize, usize::saturating_add)
        };
        inverted
            .saturating_add(values(&self.doc_values))
            .saturating_add(values(&self.points))
    }
}

fn explicit_error(message: impl Into<String>) -> Error {
    Error::Explicit(message.into())
}

impl IndexingConfig {
    /// The registered field with number `n`.
    fn explicit_field(&self, n: i32) -> Result<&FieldInfo> {
        self.fields
            .iter()
            .find(|f| f.number == n)
            .ok_or_else(|| explicit_error(format!("field number {n} is not registered")))
    }

    /// Checks one document against the registered schema -- at add time,
    /// while the caller still has the document, as `IndexingChain` does.
    pub(crate) fn validate_explicit(&self, doc: &ExplicitDocument) -> Result<()> {
        for s in &doc.stored {
            self.explicit_field(s.field_number)?;
        }
        let mut seen = BTreeSet::new();
        for inv in &doc.fields.inverted {
            let f = self.explicit_field(inv.field_number)?;
            if !seen.insert(inv.field_number) {
                return Err(explicit_error(format!(
                    "field {:?} is inverted twice in one document; merge its occurrences",
                    f.name
                )));
            }
            if f.index_options == IndexOptions::None {
                return Err(explicit_error(format!("field {:?} is not indexed", f.name)));
            }
            let wants_norm = !f.omit_norms;
            if inv.norm.is_some() != wants_norm {
                return Err(explicit_error(format!(
                    "field {:?}: a norm must be given exactly when the field has norms",
                    f.name
                )));
            }
            let positions = matches!(
                f.index_options,
                IndexOptions::DocsAndFreqsAndPositions
                    | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
            );
            let offsets = f.index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
            let mut terms = BTreeSet::new();
            for t in &inv.terms {
                if !terms.insert(t.term.as_slice()) {
                    return Err(explicit_error(format!(
                        "field {:?}: term {:?} listed twice",
                        f.name, t.term
                    )));
                }
                if t.freq < 1 {
                    return Err(explicit_error(format!(
                        "field {:?}: freq {} < 1",
                        f.name, t.freq
                    )));
                }
                if positions {
                    if t.positions.len() != t.freq as usize
                        || t.positions.windows(2).any(|w| w[0] > w[1])
                        || t.positions.first().is_some_and(|&p| p < 0)
                    {
                        return Err(explicit_error(format!(
                            "field {:?}: positions must be freq non-negative ascending values",
                            f.name
                        )));
                    }
                } else if !t.positions.is_empty() {
                    return Err(explicit_error(format!(
                        "field {:?} does not index positions",
                        f.name
                    )));
                }
                if offsets {
                    if t.offsets.len() != t.freq as usize
                        || t.offsets.iter().any(|&(s, e)| s < 0 || e < s)
                    {
                        return Err(explicit_error(format!(
                            "field {:?}: offsets must be freq (start, end) pairs, 0 <= start <= end",
                            f.name
                        )));
                    }
                } else if !t.offsets.is_empty() {
                    return Err(explicit_error(format!(
                        "field {:?} does not index offsets",
                        f.name
                    )));
                }
            }
        }
        let mut single = BTreeSet::new();
        for dv in &doc.fields.doc_values {
            let f = self.explicit_field(dv.field_number)?;
            let ok = matches!(
                (f.doc_values_type, &dv.value),
                (
                    DocValuesType::Numeric | DocValuesType::SortedNumeric,
                    FieldValue::Long(_)
                ) | (
                    DocValuesType::Binary | DocValuesType::Sorted | DocValuesType::SortedSet,
                    FieldValue::Binary(_)
                )
            );
            if !ok {
                return Err(explicit_error(format!(
                    "field {:?}: a {:?} doc value cannot be {:?}",
                    f.name, f.doc_values_type, dv.value
                )));
            }
            let single_valued = matches!(
                f.doc_values_type,
                DocValuesType::Numeric | DocValuesType::Binary | DocValuesType::Sorted
            );
            if single_valued && !single.insert(dv.field_number) {
                return Err(explicit_error(format!(
                    "field {:?}: {:?} doc values take one value per document",
                    f.name, f.doc_values_type
                )));
            }
        }
        for p in &doc.fields.points {
            let f = self.explicit_field(p.field_number)?;
            // Registered shapes are small positive integers taken from Java's
            // `FieldType`, which caps both; saturation keeps this total anyway.
            let want = usize::try_from(f.point_dimension_count.saturating_mul(f.point_num_bytes))
                .unwrap_or(0);
            match &p.value {
                FieldValue::Binary(b) if want > 0 && b.len() == want => {}
                other => {
                    return Err(explicit_error(format!(
                        "field {:?}: a point must be {want} packed bytes, got {other:?}",
                        f.name
                    )))
                }
            }
        }
        Ok(())
    }

    /// Writes one flushed segment from explicit documents: see the module doc.
    pub(crate) fn build_and_write_explicit_segment(
        &self,
        dir: &dyn Directory,
        buf: &DocumentBuffer<'_>,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
    ) -> Result<(SegmentCommitInfo, Vec<String>)> {
        let max_doc = buf.docs.len();
        let explicit = buf.explicit;
        if explicit.len() != max_doc {
            return Err(explicit_error(
                "explicit document fields are out of step with the document buffer",
            ));
        }

        // The fields this segment carries, in any role.
        let mut present: BTreeSet<i32> = BTreeSet::new();
        for (doc, fields) in buf.docs.iter().zip(explicit) {
            present.extend(doc.fields.iter().map(|f| f.field_number));
            present.extend(fields.inverted.iter().map(|f| f.field_number));
            present.extend(fields.doc_values.iter().map(|f| f.field_number));
            present.extend(fields.points.iter().map(|f| f.field_number));
        }

        // Postings: one term dictionary per field, terms in byte order.
        struct Building {
            terms: BTreeMap<Vec<u8>, TermPostings>,
            docs: usize,
        }
        let mut postings: BTreeMap<i32, Building> = BTreeMap::new();
        let mut norm_values: BTreeMap<i32, Vec<Option<i64>>> = BTreeMap::new();
        for (doc_id, fields) in explicit.iter().enumerate() {
            for inv in &fields.inverted {
                if let Some(norm) = inv.norm {
                    norm_values
                        .entry(inv.field_number)
                        .or_insert_with(|| vec![None; max_doc])[doc_id] = Some(norm);
                }
                if inv.terms.is_empty() {
                    continue;
                }
                let b = postings
                    .entry(inv.field_number)
                    .or_insert_with(|| Building {
                        terms: BTreeMap::new(),
                        docs: 0,
                    });
                b.docs = b.docs.saturating_add(1);
                for t in &inv.terms {
                    let tp = b
                        .terms
                        .entry(t.term.clone())
                        .or_insert_with(|| TermPostings {
                            term: t.term.clone(),
                            docs: Vec::new(),
                            positions: Vec::new(),
                            offsets: Vec::new(),
                            payload_bytes: Vec::new(),
                            payload_lengths: Vec::new(),
                        });
                    tp.docs.push((doc_id as i32, t.freq));
                    tp.positions.push(t.positions.clone());
                    tp.offsets.push(t.offsets.clone());
                }
            }
        }
        // Dense norm columns for the postings writer's impacts: a document
        // without the field never appears in its postings, so its slot is
        // never read.
        let dense_norms: Vec<(i32, Vec<i64>)> = norm_values
            .iter()
            .map(|(&n, col)| (n, col.iter().map(|v| v.unwrap_or(1)).collect()))
            .collect();
        let term_lists: Vec<(i32, IndexOptions, i32, Vec<TermPostings>)> = postings
            .into_iter()
            .map(|(n, b)| {
                let options = self.explicit_field(n).map(|f| f.index_options)?;
                Ok((n, options, b.docs as i32, b.terms.into_values().collect()))
            })
            .collect::<Result<_>>()?;
        let postings_fields: BTreeSet<i32> = term_lists.iter().map(|(n, ..)| *n).collect();
        let postings_output = if term_lists.is_empty() {
            None
        } else {
            let inputs: Vec<FieldPostingsInput<'_>> = term_lists
                .iter()
                .map(|(n, options, docs, terms)| FieldPostingsInput {
                    field_number: *n,
                    index_options: *options,
                    doc_count: *docs,
                    has_payloads: false,
                    terms,
                })
                .collect();
            let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = dense_norms
                .iter()
                .map(|(n, values)| postings_writer::FieldNorms {
                    field_number: *n,
                    values,
                })
                .collect();
            Some(postings_writer::write_fields_with_norms(
                &inputs,
                &norms_for_impacts,
                &segment_id,
                &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
            )?)
        };

        // Norms: a column for every present indexed field with norms, dense
        // when every document has it -- `Lucene90NormsConsumer`'s choice.
        let norms_output = if norm_values.is_empty() {
            None
        } else {
            let columns: Vec<NormsColumn> = norm_values
                .iter()
                .map(|(&n, col)| {
                    if col.iter().all(Option::is_some) {
                        (
                            n,
                            Some(col.iter().map(|v| v.unwrap_or(0)).collect()),
                            Vec::new(),
                        )
                    } else {
                        let pairs = col
                            .iter()
                            .enumerate()
                            .filter_map(|(d, v)| v.map(|v| (d as i32, v)))
                            .collect();
                        (n, None, pairs)
                    }
                })
                .collect();
            let fields: Vec<norms::NormsField<'_>> = columns
                .iter()
                .map(|(n, dense, pairs)| match dense {
                    Some(values) => norms::NormsField::Dense(*n, values),
                    None => norms::NormsField::Sparse(*n, pairs),
                })
                .collect();
            Some(norms::write_fields(
                &fields,
                max_doc as i32,
                &segment_id,
                "",
            )?)
        };

        // Doc values and points through the native builders, over one
        // synthetic value list per document.
        let synthetic = |pick: fn(&ExplicitFields) -> &Vec<StoredField>| -> Vec<Document> {
            explicit
                .iter()
                .map(|f| Document {
                    fields: pick(f).clone(),
                })
                .collect()
        };
        let dv_configs: Vec<DocValuesFieldConfig> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number) && f.doc_values_type != DocValuesType::None)
            .map(|f| DocValuesFieldConfig {
                name: f.name.clone(),
                field_number: f.number,
                doc_values_type: f.doc_values_type,
            })
            .collect();
        let doc_values_output = if dv_configs.is_empty() {
            None
        } else {
            Some(IndexWriter::build_doc_values_output(
                &synthetic(|f| &f.doc_values),
                &dv_configs,
                &segment_id,
            )?)
        };
        let point_configs: Vec<PointsFieldConfig> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number) && f.point_dimension_count > 0)
            .map(|f| PointsFieldConfig {
                name: f.name.clone(),
                field_number: f.number,
                num_dims: f.point_dimension_count,
                num_index_dims: f.point_index_dimension_count,
                bytes_per_dim: f.point_num_bytes,
            })
            .collect();
        let points_output = if point_configs.is_empty() {
            None
        } else {
            IndexWriter::build_points_output(
                &synthetic(|f| &f.points),
                &point_configs,
                &segment_id,
            )?
        };

        // This segment's FieldInfos: the present fields, full schema, and the
        // per-field format attributes for what was written.
        let fnm_fields: Vec<FieldInfo> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number))
            .map(|f| {
                let mut f = f.clone();
                f.attributes.retain(|(k, _)| !k.starts_with("PerField"));
                if postings_fields.contains(&f.number) {
                    f.attributes.push((
                        "PerFieldPostingsFormat.format".to_string(),
                        POSTINGS_FORMAT_NAME.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldPostingsFormat.suffix".to_string(),
                        PER_FIELD_SUFFIX.to_string(),
                    ));
                }
                if f.doc_values_type != DocValuesType::None {
                    f.attributes.push((
                        "PerFieldDocValuesFormat.format".to_string(),
                        DOC_VALUES_FORMAT_NAME.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldDocValuesFormat.suffix".to_string(),
                        PER_FIELD_SUFFIX.to_string(),
                    ));
                }
                f
            })
            .collect();

        let mut flushed = segment_writer::write_stored_only_segment_files(
            dir,
            segment_name,
            segment_id,
            &self.codec_name,
            self.lucene_version,
            &fnm_fields,
            buf.docs,
            false,
            buf.has_blocks,
        )?;
        let mut record = |names: Vec<String>| {
            flushed.info.files.extend(names.iter().cloned());
            flushed.pending_sync.extend(names);
        };
        if let Some(output) = postings_output {
            record(IndexWriter::write_postings_files(
                dir,
                segment_name,
                &output,
            )?);
        }
        if let Some((dvm, dvd, dvs)) = doc_values_output {
            record(IndexWriter::write_doc_values_files(
                dir,
                segment_name,
                &dvm,
                &dvd,
                &dvs,
            )?);
        }
        if let Some((nvm, nvd)) = norms_output {
            record(IndexWriter::write_norms_files(
                dir,
                segment_name,
                &nvm,
                &nvd,
            )?);
        }
        if let Some(output) = &points_output {
            record(IndexWriter::write_points_files(dir, segment_name, output)?);
        }
        segment_writer::seal_flushed_segment(dir, segment_name, flushed).map_err(Error::from)
    }
}

impl<'d> IndexWriter<'d> {
    /// Switches this writer to [`ExplicitDocument`]s. Must be called before
    /// any document is buffered; refused with an index sort, and once on, the
    /// native [`Document`] entry points are refused.
    pub fn enable_explicit_documents(&mut self) -> Result<()> {
        if self.cfg.explicit {
            return Ok(());
        }
        if !self.pending_docs.is_empty() {
            return Err(explicit_error(
                "explicit documents must be enabled before any document is buffered",
            ));
        }
        if self.cfg.index_sort.is_some() {
            return Err(explicit_error(
                "explicit documents do not support an index sort",
            ));
        }
        self.cfg_mut().explicit = true;
        Ok(())
    }

    /// `FieldInfos.FieldNumbers.addOrGet`: the global number of a field with
    /// this schema, registering it on first sight. `info.number` is ignored.
    /// A known name with a different schema is refused, as Lucene refuses to
    /// change a field's index options, doc-values type or point shape.
    pub fn register_field(&mut self, info: FieldInfo) -> Result<i32> {
        if !self.cfg.explicit {
            return Err(explicit_error(
                "register_field needs explicit documents enabled",
            ));
        }
        if info.store_term_vectors
            || info.store_payloads
            || info.vector_dimension != 0
            || info.doc_values_skip_index_type != DocValuesSkipIndexType::None
        {
            return Err(explicit_error(format!(
                "field {:?}: term vectors, payloads, vectors and doc-values skip indexes are not supported",
                info.name
            )));
        }
        let same_schema = |a: &FieldInfo, b: &FieldInfo| {
            a.index_options == b.index_options
                && a.omit_norms == b.omit_norms
                && a.doc_values_type == b.doc_values_type
                && a.point_dimension_count == b.point_dimension_count
                && a.point_index_dimension_count == b.point_index_dimension_count
                && a.point_num_bytes == b.point_num_bytes
                && a.soft_deletes_field == b.soft_deletes_field
                && a.parent_field == b.parent_field
        };
        if let Some(existing) = self.cfg.fields.iter().find(|f| f.name == info.name) {
            if !same_schema(existing, &info) {
                return Err(explicit_error(format!(
                    "field {:?} is registered with a different schema",
                    info.name
                )));
            }
            return Ok(existing.number);
        }
        let number = self
            .cfg
            .fields
            .iter()
            .map(|f| f.number)
            .max()
            .map_or(0, |m| m.saturating_add(1));
        let mut info = info;
        info.number = number;
        info.doc_values_gen = -1;
        info.attributes.clear();
        self.cfg_mut().fields.push(info);
        Ok(number)
    }

    fn explicit_documents_check(&self, docs: &[ExplicitDocument]) -> Result<()> {
        if !self.cfg.explicit {
            return Err(explicit_error("explicit documents are not enabled"));
        }
        docs.iter().try_for_each(|d| self.cfg.validate_explicit(d))
    }

    /// `IndexWriter.addDocuments` for explicit documents (one document is a
    /// block of one).
    pub fn add_explicit_documents(&mut self, docs: Vec<ExplicitDocument>) -> Result<super::SeqNo> {
        self.explicit_documents_check(&docs)?;
        self.add_explicit_with_delete(None, docs)
    }

    /// `IndexWriter.updateDocuments(term, docs)` for explicit documents.
    pub fn update_explicit_documents(
        &mut self,
        term: super::Term,
        docs: Vec<ExplicitDocument>,
    ) -> Result<super::SeqNo> {
        self.explicit_documents_check(&docs)?;
        self.add_explicit_with_delete(
            Some(super::DeleteNode::Terms(vec![std::sync::Arc::new(term)])),
            docs,
        )
    }

    /// `IndexWriter.softUpdateDocuments(term, docs, softDeletes...)` for
    /// explicit documents: `docs` are added and every earlier document
    /// matching `term` gets `soft_deletes`.
    pub fn soft_update_explicit_documents(
        &mut self,
        term: super::Term,
        docs: Vec<ExplicitDocument>,
        soft_deletes: &[super::DocValuesUpdate],
    ) -> Result<super::SeqNo> {
        if soft_deletes.is_empty() {
            return Err(Error::NoSoftDeletesSupplied);
        }
        self.explicit_documents_check(&docs)?;
        // `verifyOrCreateDvOnlyField`: an update to a field this writer has
        // never registered would resolve against nothing and vanish.
        for update in soft_deletes {
            self.cfg.verify_doc_values_update_field(update)?;
        }
        let updates = soft_deletes
            .iter()
            .map(|u| super::retarget_update(u, &term))
            .collect();
        self.add_explicit_with_delete(Some(super::DeleteNode::DocValuesUpdates(updates)), docs)
    }

    fn add_explicit_with_delete(
        &mut self,
        delete: Option<super::DeleteNode>,
        docs: Vec<ExplicitDocument>,
    ) -> Result<super::SeqNo> {
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = match delete {
            Some(node) => self.buffer_delete_node(node, doc_id_upto),
            None => self.delete_queue.next_sequence_number(),
        };
        if docs.len() > 1 {
            self.pending_has_blocks = true;
        }
        for doc in docs {
            let extra = doc.fields.ram_bytes();
            self.buffer_document(Document { fields: doc.stored });
            *self
                .pending_explicit
                .last_mut()
                .expect("buffer_document pushed an entry") = doc.fields;
            self.ram_bytes_used = self.ram_bytes_used.saturating_add(extra);
        }
        self.maybe_flush()?;
        Ok(seq_no)
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic, not values read off disk -- see
    // `docs/arithmetic-gate.md`'s "Test code" section.
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::buffered_updates::{DocValuesUpdate, Term};
    use crate::index_file_deleter::DeletionPolicy;
    use crate::index_writer::SoftDeletesRetention;
    use crate::segment_info::LuceneVersion;
    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    const VERSION: LuceneVersion = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };

    /// The field shapes an OpenSearch document is made of.
    struct Fields {
        id: i32,
        source: i32,
        body: i32,
        tag: i32,
        num: i32,
        soft: i32,
        seq: i32,
    }

    fn register(w: &mut IndexWriter<'_>) -> Fields {
        w.enable_explicit_documents().unwrap();
        let reg = |w: &mut IndexWriter<'_>, f: FieldInfo| w.register_field(f).unwrap();
        Fields {
            id: reg(
                w,
                FieldInfo::new("_id", 0)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true),
            ),
            source: reg(w, FieldInfo::new("_source", 0)),
            body: reg(
                w,
                FieldInfo::new("body", 0)
                    .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
            ),
            tag: reg(
                w,
                FieldInfo::new("tag", 0)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true)
                    .with_doc_values(DocValuesType::SortedSet, DocValuesSkipIndexType::None, -1),
            ),
            num: reg(
                w,
                FieldInfo::new("num", 0)
                    .with_points(1, 1, 8)
                    .with_doc_values(
                        DocValuesType::SortedNumeric,
                        DocValuesSkipIndexType::None,
                        -1,
                    ),
            ),
            soft: reg(
                w,
                FieldInfo::new("__soft_deletes", 0)
                    .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                    .with_soft_deletes_field(true),
            ),
            seq: reg(
                w,
                FieldInfo::new("_seq_no", 0).with_doc_values(
                    DocValuesType::Numeric,
                    DocValuesSkipIndexType::None,
                    -1,
                ),
            ),
        }
    }

    fn term(t: &str, positions: &[i32]) -> InvertedTerm {
        InvertedTerm {
            term: t.as_bytes().to_vec(),
            freq: positions.len() as i32,
            positions: positions.to_vec(),
            offsets: positions.iter().map(|&p| (p * 6, p * 6 + 5)).collect(),
        }
    }

    /// Document `i`: `_id` "d{i}" (stored and indexed), a stored `_source`,
    /// `body` with two or three terms, `tag` "t{i%3}", `num` = i. Every third
    /// document has no `num`, so its columns are sparse.
    fn doc(f: &Fields, i: i64) -> ExplicitDocument {
        let id = format!("d{i}");
        let mut body = vec![term("alpha", &[0]), term("beta", &[1, 3])];
        if i % 2 == 0 {
            body.push(term("gamma", &[2]));
        }
        let length = body.iter().map(|t| t.freq).sum::<i32>() as i64;
        let tag = format!("t{}", i % 3).into_bytes();
        let mut fields = ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: f.id,
                    terms: vec![term_docs(&id)],
                    norm: None,
                },
                InvertedField {
                    field_number: f.body,
                    terms: body,
                    norm: Some(length),
                },
                InvertedField {
                    field_number: f.tag,
                    terms: vec![term_docs(std::str::from_utf8(&tag).unwrap())],
                    norm: None,
                },
            ],
            doc_values: vec![
                StoredField {
                    field_number: f.tag,
                    value: FieldValue::Binary(tag),
                },
                StoredField {
                    field_number: f.seq,
                    value: FieldValue::Long(i),
                },
            ],
            points: Vec::new(),
        };
        if i % 3 != 0 {
            fields.doc_values.push(StoredField {
                field_number: f.num,
                value: FieldValue::Long(i),
            });
            fields.points.push(StoredField {
                field_number: f.num,
                value: FieldValue::Binary(((i as u64) ^ (1 << 63)).to_be_bytes().to_vec()),
            });
        }
        ExplicitDocument {
            stored: vec![
                StoredField {
                    field_number: f.id,
                    value: FieldValue::String(id),
                },
                StoredField {
                    field_number: f.source,
                    value: FieldValue::Binary(format!("{{\"n\":{i}}}").into_bytes()),
                },
            ],
            fields,
        }
    }

    fn term_docs(t: &str) -> InvertedTerm {
        InvertedTerm {
            term: t.as_bytes().to_vec(),
            freq: 1,
            positions: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn check(dir: &FsDirectory) {
        let results = crate::check_index::check_directory(dir).unwrap();
        for r in &results {
            assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
        }
    }

    fn segment_fields(dir: &FsDirectory, sci: &SegmentCommitInfo) -> Vec<FieldInfo> {
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "")
            .unwrap()
            .fields
    }

    /// Explicit documents flush, merge and commit into segments the port's
    /// `CheckIndex` accepts, each listing only the fields it carries.
    #[test]
    fn explicit_documents_round_trip_through_check_index() {
        let tmp = TempDir::new("explicit-round-trip");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.set_max_buffered_docs(7).unwrap();
        let docs: Vec<ExplicitDocument> = (0..30).map(|i| doc(&f, i)).collect();
        for d in docs {
            w.add_explicit_documents(vec![d]).unwrap();
        }
        let infos = w.commit().unwrap().clone();
        assert!(infos.segments.len() >= 4, "several flushes");
        check(&dir);

        let fields = segment_fields(&dir, &infos.segments[0]);
        let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            ["_id", "_source", "body", "tag", "num", "_seq_no"],
            "no soft-deletes field yet"
        );
        let attr = |name: &str, key: &str| {
            fields
                .iter()
                .find(|f| f.name == name)
                .unwrap()
                .attributes
                .iter()
                .any(|(k, _)| k == key)
        };
        assert!(attr("body", "PerFieldPostingsFormat.format"));
        assert!(!attr("_source", "PerFieldPostingsFormat.format"));
        assert!(attr("tag", "PerFieldDocValuesFormat.format"));
        assert!(!attr("body", "PerFieldDocValuesFormat.format"));

        // A soft update reaches a segment that lacks the soft-deletes field in
        // its `.fnm`, and a merge keeps it all readable.
        let one = DocValuesUpdate::Numeric {
            term: Term {
                field: "_id".to_string(),
                bytes: b"d3".to_vec(),
            },
            field: "__soft_deletes".to_string(),
            value: Some(1),
        };
        w.soft_update_explicit_documents(
            Term {
                field: "_id".to_string(),
                bytes: b"d3".to_vec(),
            },
            vec![doc(&f, 3)],
            &[one],
        )
        .unwrap();
        let infos = w.commit().unwrap().clone();
        check(&dir);
        assert_eq!(f.soft, 5);
        // `softDelCount`: the old d3, and nothing else yet.
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 1);
        assert_eq!(
            infos
                .segments
                .iter()
                .filter(|s| s.soft_del_count == 1)
                .count(),
            1
        );

        // A tombstone born soft-deleted counts in its own segment.
        let mut tombstone = doc(&f, 100);
        tombstone.fields.doc_values.push(StoredField {
            field_number: f.soft,
            value: FieldValue::Long(1),
        });
        w.add_explicit_documents(vec![tombstone]).unwrap();
        let infos = w.commit().unwrap().clone();
        check(&dir);
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 2);
    }

    #[test]
    fn registration_is_global_and_schema_is_fixed() {
        let tmp = TempDir::new("explicit-register");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let text = FieldInfo::new("t", 0).with_index_options(IndexOptions::DocsAndFreqs);
        assert!(
            w.register_field(text.clone()).is_err(),
            "needs explicit mode"
        );
        w.enable_explicit_documents().unwrap();
        w.enable_explicit_documents().unwrap();
        assert_eq!(w.register_field(text.clone()).unwrap(), 0);
        assert_eq!(w.register_field(FieldInfo::new("u", 7)).unwrap(), 1);
        assert_eq!(
            w.register_field(text.clone()).unwrap(),
            0,
            "same schema, same number"
        );
        let changed = text.clone().with_omit_norms(true);
        assert!(w.register_field(changed).is_err());
        let vectors = FieldInfo::new("v", 0).with_store_term_vectors(true);
        assert!(w.register_field(vectors).is_err());
    }

    #[test]
    fn explicit_mode_must_come_first() {
        let tmp = TempDir::new("explicit-first");
        let dir = FsDirectory::open(tmp.path());
        let mut w =
            IndexWriter::open(&dir, vec![FieldInfo::new("s", 0)], "Lucene104", VERSION).unwrap();
        assert!(w
            .add_explicit_documents(vec![ExplicitDocument::default()])
            .is_err());
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::Long(1),
            }],
        })
        .unwrap();
        assert!(w.enable_explicit_documents().is_err());
    }

    /// Everything `validate_explicit` refuses, one case each.
    #[test]
    fn invalid_documents_are_refused_at_add_time() {
        let tmp = TempDir::new("explicit-invalid");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let base = || doc(&f, 1);
        let mut cases: Vec<(&str, ExplicitDocument)> = Vec::new();
        let mut d = base();
        d.stored[0].field_number = 99;
        cases.push(("unknown stored field", d));
        let mut d = base();
        d.fields.inverted.push(d.fields.inverted[0].clone());
        cases.push(("field inverted twice", d));
        let mut d = base();
        d.fields.inverted[0].field_number = f.source;
        cases.push(("not indexed", d));
        let mut d = base();
        d.fields.inverted[1].norm = None;
        cases.push(("missing norm", d));
        let mut d = base();
        d.fields.inverted[0].norm = Some(1);
        cases.push(("norm on an omitNorms field", d));
        let mut d = base();
        let dup = d.fields.inverted[1].terms[0].clone();
        d.fields.inverted[1].terms.push(dup);
        cases.push(("duplicate term", d));
        let mut d = base();
        d.fields.inverted[1].terms[0].freq = 0;
        cases.push(("freq 0", d));
        let mut d = base();
        d.fields.inverted[1].terms[1].positions = vec![3, 1];
        cases.push(("descending positions", d));
        let mut d = base();
        d.fields.inverted[0].terms[0].positions = vec![0];
        cases.push(("positions on a docs-only field", d));
        let mut d = base();
        d.fields.inverted[1].terms[0].offsets = vec![(5, 1)];
        cases.push(("end before start", d));
        let mut d = base();
        d.fields.inverted[0].terms[0].offsets = vec![(0, 1)];
        cases.push(("offsets on a docs-only field", d));
        let mut d = base();
        d.fields.doc_values[0].value = FieldValue::Long(1);
        cases.push(("a long for a sorted-set field", d));
        let mut d = base();
        d.fields.doc_values.push(StoredField {
            field_number: f.soft,
            value: FieldValue::Long(1),
        });
        d.fields.doc_values.push(StoredField {
            field_number: f.soft,
            value: FieldValue::Long(1),
        });
        cases.push(("two numeric values", d));
        let mut d = base();
        d.fields.points[0].value = FieldValue::Binary(vec![0; 3]);
        cases.push(("short point", d));
        for (what, d) in cases {
            let err = w.add_explicit_documents(vec![d]).expect_err(what);
            assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
        }
        let soft = w.soft_update_explicit_documents(
            Term {
                field: "_id".to_string(),
                bytes: b"d1".to_vec(),
            },
            vec![base()],
            &[],
        );
        assert!(matches!(soft, Err(Error::NoSoftDeletesSupplied)));
    }

    /// `update_explicit_documents` hard-deletes the earlier document.
    #[test]
    fn update_replaces_the_earlier_document() {
        let tmp = TempDir::new("explicit-update");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.add_explicit_documents((0..4).map(|i| doc(&f, i)).collect())
            .unwrap();
        w.commit().unwrap();
        w.update_explicit_documents(
            Term {
                field: "_id".to_string(),
                bytes: b"d2".to_vec(),
            },
            vec![doc(&f, 2)],
        )
        .unwrap();
        let infos = w.commit().unwrap().clone();
        // `committed_doc_count` is maxDoc; the live count subtracts deletes.
        let deleted: i32 = infos.segments.iter().map(|s| s.del_count).sum();
        assert_eq!(w.committed_doc_count().unwrap(), 5);
        assert_eq!(deleted, 1, "the earlier d2 is deleted");
        check(&dir);
    }

    fn id_term(i: i64) -> Term {
        Term {
            field: "_id".to_string(),
            bytes: format!("d{i}").into_bytes(),
        }
    }

    /// Document `i` re-indexed as the operation with sequence number `seq_no`.
    fn doc_at(f: &Fields, i: i64, seq_no: i64) -> ExplicitDocument {
        let mut d = doc(f, i);
        for v in &mut d.fields.doc_values {
            if v.field_number == f.seq {
                v.value = FieldValue::Long(seq_no);
            }
        }
        d
    }

    fn soft_delete(i: i64) -> DocValuesUpdate {
        DocValuesUpdate::Numeric {
            term: id_term(i),
            field: "__soft_deletes".to_string(),
            value: Some(1),
        }
    }

    /// `(maxDoc, softDelCount)` over the committed segments.
    fn counts(w: &mut IndexWriter<'_>) -> (usize, i32) {
        let soft = w
            .commit()
            .unwrap()
            .segments
            .iter()
            .map(|s| s.soft_del_count)
            .sum();
        (w.committed_doc_count().unwrap(), soft)
    }

    /// `SoftDeletesRetentionMergePolicy`: a merge carries a soft-deleted
    /// document over while its `_seq_no` is retained, and drops it after.
    #[test]
    fn merges_drop_only_the_soft_deleted_history_no_longer_retained() {
        let tmp = TempDir::new("explicit-retention");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.set_max_buffered_docs(4).unwrap();
        for i in 0..10 {
            w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        }
        w.commit().unwrap();
        // Operations 100..=105 replace d0..d5; seq_nos 0..=5 become history.
        for i in 0..6 {
            w.soft_update_explicit_documents(
                id_term(i),
                vec![doc_at(&f, i, 100 + i)],
                &[soft_delete(i)],
            )
            .unwrap();
        }
        assert_eq!(counts(&mut w), (16, 6));

        // No retention policy: every soft-deleted document survives.
        w.force_merge(1).unwrap();
        assert_eq!(w.commit().unwrap().segments.len(), 1);
        assert_eq!(counts(&mut w), (16, 6));
        check(&dir);

        // Seq_nos 0..=2 fall out of retention; 3..=5 are still needed.
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: 3,
        }));
        w.force_merge_deletes().unwrap();
        assert_eq!(counts(&mut w), (13, 3));
        check(&dir);

        // Nothing retained: all history goes, and a clean index is left alone.
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: i64::MAX,
        }));
        w.force_merge_deletes().unwrap();
        assert_eq!(counts(&mut w), (10, 0));
        let before = w.commit_generations();
        w.force_merge_deletes().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(w.commit_generations(), before, "nothing to merge");
        check(&dir);

        assert!(matches!(
            w.force_merge(0),
            Err(Error::InvalidMaxNumSegments(0))
        ));
    }

    /// A soft-deleted document with no sequence number is never retained.
    #[test]
    fn a_soft_deleted_document_without_a_seq_no_is_not_retained() {
        let tmp = TempDir::new("explicit-retention-noseq");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let mut tombstone = doc(&f, 7);
        tombstone
            .fields
            .doc_values
            .retain(|v| v.field_number != f.seq);
        tombstone.fields.doc_values.push(StoredField {
            field_number: f.soft,
            value: FieldValue::Long(1),
        });
        w.add_explicit_documents(vec![doc(&f, 1), tombstone])
            .unwrap();
        w.commit().unwrap();
        w.add_explicit_documents(vec![doc(&f, 2)]).unwrap();
        assert_eq!(counts(&mut w), (3, 1));
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: 0,
        }));
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (2, 0));
        check(&dir);
    }

    /// OpenSearch's `CombinedDeletionPolicy` decides which commits live, and a
    /// reader's hold keeps a dropped commit's segment files on disk.
    #[test]
    fn caller_driven_commit_deletion_respects_holds() {
        let tmp = TempDir::new("explicit-holds");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_deletion_policy(DeletionPolicy::KeepAll).unwrap();
        let f = register(&mut w);
        w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
        let first = w.commit().unwrap().generation;
        w.add_explicit_documents(vec![doc(&f, 1)]).unwrap();
        let second = w.commit().unwrap().generation;
        assert_eq!(w.commit_generations(), [first, second]);

        let held = w.hold_commit(first).unwrap();
        assert!(!held.is_empty());
        assert!(held.iter().all(|f| !f.starts_with("segments")));
        w.force_merge(1).unwrap();
        let merged = w.commit_generations();
        assert_eq!(merged.len(), 3);
        let newest = *merged.last().unwrap();
        w.delete_commits(&merged).unwrap();
        assert_eq!(
            w.commit_generations(),
            [newest],
            "the newest always survives"
        );
        let on_disk = dir.list_all().unwrap();
        let first_name = lucene_store::directory::segments_file_name(first).unwrap();
        assert!(!on_disk.contains(&first_name));
        assert!(held.iter().all(|f| on_disk.contains(f)), "held files stay");
        check(&dir);

        w.release_files(&held).unwrap();
        let on_disk = dir.list_all().unwrap();
        assert!(
            held.iter().all(|f| !on_disk.contains(f)),
            "released files go"
        );
        check(&dir);

        w.delete_commits(&[newest, 12345]).unwrap();
        assert_eq!(w.commit_generations(), [newest]);
        assert!(w.hold_commit(999).is_err(), "no such commit");
    }

    /// A merged segment numbers its fields in the order its sources declare
    /// them, not the writer's: a later soft delete must still land on the
    /// soft-deletes field, found by name -- not on whichever field of the
    /// merged segment happens to carry the writer's number for it.
    #[test]
    fn a_soft_delete_finds_its_field_by_name_in_a_renumbered_segment() {
        let tmp = TempDir::new("explicit-renumbered");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        // A first segment with only `_id` and `_seq_no`, so the merged
        // segment numbers those two first.
        let mut sparse = doc(&f, 0);
        sparse.stored.retain(|s| s.field_number == f.id);
        sparse.fields.inverted.retain(|i| i.field_number == f.id);
        sparse.fields.doc_values.retain(|v| v.field_number == f.seq);
        sparse.fields.points.clear();
        w.add_explicit_documents(vec![sparse]).unwrap();
        w.commit().unwrap();
        for i in 1..4 {
            w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        }
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        let merged = w.commit().unwrap().segments[0].clone();
        let renumbered = segment_fields(&dir, &merged);
        assert_ne!(
            renumbered
                .iter()
                .find(|x| x.name == "_seq_no")
                .unwrap()
                .number,
            f.seq,
            "the merged segment numbers its fields its own way"
        );

        w.soft_update_explicit_documents(id_term(2), vec![doc_at(&f, 2, 100)], &[soft_delete(2)])
            .unwrap();
        let infos = w.commit().unwrap().clone();
        check(&dir);
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 1);
        let seg = infos
            .segments
            .iter()
            .find(|s| s.segment_name == merged.segment_name)
            .unwrap();
        let fnm = dir
            .open(&crate::field_updates::field_infos_gen_file_name(
                &seg.segment_name,
                seg.field_infos_gen,
            ))
            .unwrap();
        let current = lucene_codecs::field_infos::parse(
            &fnm,
            &seg.segment_id,
            &lucene_util::base36::to_base36(seg.field_infos_gen),
        )
        .unwrap();
        for field in &current.fields {
            let expected = renumbered.iter().find(|x| x.name == field.name);
            match expected {
                Some(before) => assert_eq!(
                    field.doc_values_type, before.doc_values_type,
                    "{} keeps its doc-values type",
                    field.name
                ),
                None => assert_eq!(field.name, "__soft_deletes", "only the soft field is new"),
            }
        }
    }

    /// A soft delete naming a field the writer never registered is refused,
    /// rather than resolving against nothing and silently leaving the old
    /// version live.
    #[test]
    fn a_soft_update_of_an_unregistered_field_is_refused() {
        let tmp = TempDir::new("explicit-unregistered-soft");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.enable_explicit_documents().unwrap();
        let id = w
            .register_field(
                FieldInfo::new("_id", 0)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true),
            )
            .unwrap();
        let one = |i: &str| ExplicitDocument {
            stored: Vec::new(),
            fields: ExplicitFields {
                inverted: vec![InvertedField {
                    field_number: id,
                    terms: vec![term_docs(i)],
                    norm: None,
                }],
                doc_values: Vec::new(),
                points: Vec::new(),
            },
        };
        w.add_explicit_documents(vec![one("a")]).unwrap();
        let soft = DocValuesUpdate::Numeric {
            term: id_term(0),
            field: "__soft_deletes".to_string(),
            value: Some(1),
        };
        assert!(matches!(
            w.soft_update_explicit_documents(id_term(0), vec![one("a")], &[soft.clone()]),
            Err(Error::UnknownDocValuesUpdateField(ref f)) if f == "__soft_deletes"
        ));
        // Registered, the same update soft-deletes the buffered document.
        w.register_field(
            FieldInfo::new("__soft_deletes", 0)
                .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                .with_soft_deletes_field(true),
        )
        .unwrap();
        let soft = DocValuesUpdate::Numeric {
            term: Term {
                field: "_id".to_string(),
                bytes: b"a".to_vec(),
            },
            field: "__soft_deletes".to_string(),
            value: Some(1),
        };
        w.soft_update_explicit_documents(
            Term {
                field: "_id".to_string(),
                bytes: b"a".to_vec(),
            },
            vec![one("a")],
            &[soft],
        )
        .unwrap();
        let infos = w.commit().unwrap().clone();
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 1, "the buffered version is soft-deleted");
        check(&dir);
    }
}
