//! Documents whose indexing is already decided: the shape the OpenSearch
//! engine hands this writer.
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
//! [`IndexWriter::delete_all`] releases reservations and schemas inherited
//! only from source segments. This facade retains explicitly configured field
//! schemas, their returned IDs, and parent/soft-delete roles across that call;
//! Java clears all field properties and may assign new numbers afterward.
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
//! use the real norms), norms through the installed consumer or
//! [`norms::write_fields`], doc values and points through
//! [`IndexWriter::build_doc_values_output`] and
//! [`IndexWriter::build_points_output`] over one synthetic value list per
//! document.
//!
//! KNN vectors ride alongside (the document API's vector fields,
//! [`crate::document`]), one [`super::DocumentVector`] per field and document,
//! written by the same flat/HNSW writers the native path uses.
//!
//! **Payloads** are per occurrence, as `PayloadAttribute` hands them to
//! `FreqProxTermsWriterPerField.writeProx`, and only on a field that indexes
//! positions. As in Lucene, a field stores payloads in a segment
//! (`FieldInfo.setStorePayloads`) exactly when one of the segment's
//! documents gave it a non-empty payload; the registered schema never
//! carries the flag, since Lucene's global `FieldNumbers` does not either.
//!
//! **Term vectors** arrive built, one [`TermVectorField`] per document and
//! field registered `store_term_vectors` (the document API builds them as
//! `TermVectorsConsumerPerField` does), and are written by the same
//! `.tvd`/`.tvx`/`.tvm` writer the native path uses, each document's fields
//! in name order. A segment gets the files when one of its documents inverts
//! such a field, even into no term, as Java's consumer opens its writer then.
//!
//! # Index sorting and blocks
//!
//! A flush sorts the buffer by the index sort's keys, read from each
//! document's own doc values -- the column the segment will carry -- before
//! any format is built, as the native path does. With a parent field
//! ([`IndexWriter::set_parent_field`], Java's `IndexWriterConfig.setParentField`)
//! the last document of every add carries `-1` in it, and a buffer with blocks
//! sorts each document by the key of the parent closing its block
//! (`IndexingChain.maybeSortSegment`), so blocks move whole; a merge does the
//! same per source (`MultiSorter.sort`). A block added to a sorted index
//! without a parent field is refused at add time, as Java refuses it.

use std::collections::{BTreeMap, BTreeSet};

use super::inverted_terms::InvertedTerms;
use super::pending_postings::PendingPostings;
#[cfg(test)]
use super::DOC_VALUES_FORMAT_NAME;
use super::{
    DocValuesFieldConfig, DocumentBuffer, Error, IndexWriter, IndexingConfig, PointsFieldConfig,
    Result,
};
use crate::segment_infos::SegmentCommitInfo;
use crate::segment_writer;
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::norms;
use lucene_codecs::per_field_doc_values;
use lucene_codecs::per_field_postings;
use lucene_codecs::postings_writer::{self, FieldPostingsInput};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::term_vectors::{TermVectorField, TermVectorsDocument};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;

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
    /// Registered fields this document puts in its segment's `FieldInfos`
    /// without a value: `IndexingChain.processBatch` initializes every
    /// column's `FieldInfo` before reading a cell, so a batch column that
    /// yields nothing is still a field of the segment.
    pub registered: Vec<i32>,
    /// This document's term vectors (`TermVectorsConsumerPerField`), one per
    /// field registered `store_term_vectors` that produced a term, terms in
    /// byte order. A flush writes them in field-name order, as
    /// `TermVectorsConsumer.finishDocument` sorts them.
    pub term_vectors: Vec<TermVectorField>,
}

/// One document's inverted occurrence of one field.
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedField {
    pub field_number: i32,
    /// Each distinct term once. Order does not matter.
    pub terms: InvertedTerms,
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
    /// Parallel to `positions`: each occurrence's payload (`PayloadAttribute`),
    /// empty for an occurrence without one. Empty altogether when no
    /// occurrence has one; only a field that indexes positions may carry
    /// payloads. A segment whose documents give a field a non-empty payload
    /// records the field `storePayloads`, as `FreqProxTermsWriterPerField.finish`
    /// does when it `sawPayloads`.
    pub payloads: Vec<Vec<u8>>,
}

impl ExplicitFields {
    pub(crate) fn ram_bytes(&self) -> usize {
        let inverted: usize = self
            .inverted
            .iter()
            .map(|f| f.terms.ram_bytes().saturating_add(48))
            .fold(0usize, usize::saturating_add);
        let values = |v: &[StoredField]| -> usize {
            v.iter()
                .map(|f| match &f.value {
                    FieldValue::Binary(b) => b.capacity().saturating_add(32),
                    _ => 32,
                })
                .fold(0usize, usize::saturating_add)
        };
        let term_vectors: usize = self
            .term_vectors
            .iter()
            .flat_map(|f| &f.terms)
            .map(|t| {
                let ints = [&t.positions, &t.start_offsets, &t.end_offsets]
                    .iter()
                    .map(|v| v.as_ref().map_or(0, |v| v.capacity().saturating_mul(4)))
                    .fold(0usize, usize::saturating_add);
                let payloads = t.payloads.as_ref().map_or(0, |p| {
                    p.iter()
                        .map(|p| p.capacity().saturating_add(24))
                        .fold(0usize, usize::saturating_add)
                });
                t.term
                    .capacity()
                    .saturating_add(ints)
                    .saturating_add(payloads)
                    .saturating_add(96)
            })
            .fold(0usize, usize::saturating_add);
        inverted
            .saturating_add(term_vectors)
            .saturating_add(values(&self.doc_values))
            .saturating_add(values(&self.points))
    }
}

pub(crate) fn explicit_error(message: impl Into<String>) -> Error {
    Error::Explicit(message.into())
}

/// `FieldInfos.FieldNumbers.verifySameSchema(fi)`: what `theirs` would
/// change in the index's field `ours`, as Java's
/// `IllegalArgumentException` says it -- the first of index options, term
/// vectors and norms (compared only for an indexed field), doc-values type,
/// skip index, point shape and vector shape that differs. `None` when the
/// schemas agree. The soft-deletes and parent roles, which Java checks by
/// name before this, are the last comparison here.
pub(crate) fn schema_change(ours: &FieldInfo, theirs: &FieldInfo) -> Option<String> {
    use lucene_codecs::simpletext::field_infos::{
        doc_values_type_name, index_options_name, similarity_name, skip_index_type_name,
        vector_encoding_name,
    };
    let name = &ours.name;
    let change = |what: &str, from: String, to: String| {
        Some(format!(
            "cannot change field \"{name}\" from {what}{from} to inconsistent {what}{to}"
        ))
    };
    if ours.index_options != theirs.index_options {
        return change(
            "index options=",
            index_options_name(ours.index_options).to_string(),
            index_options_name(theirs.index_options).to_string(),
        );
    }
    if ours.index_options != IndexOptions::None {
        if ours.store_term_vectors != theirs.store_term_vectors {
            return change(
                "storeTermVector=",
                ours.store_term_vectors.to_string(),
                theirs.store_term_vectors.to_string(),
            );
        }
        if ours.omit_norms != theirs.omit_norms {
            return change(
                "omitNorms=",
                ours.omit_norms.to_string(),
                theirs.omit_norms.to_string(),
            );
        }
    }
    if ours.doc_values_type != theirs.doc_values_type {
        return change(
            "doc values type=",
            doc_values_type_name(ours.doc_values_type).to_string(),
            doc_values_type_name(theirs.doc_values_type).to_string(),
        );
    }
    if ours.doc_values_skip_index_type != theirs.doc_values_skip_index_type {
        return change(
            "docValuesSkipIndexType=",
            skip_index_type_name(ours.doc_values_skip_index_type).to_string(),
            skip_index_type_name(theirs.doc_values_skip_index_type).to_string(),
        );
    }
    let points = |f: &FieldInfo| {
        (
            f.point_dimension_count,
            f.point_index_dimension_count,
            f.point_num_bytes,
        )
    };
    if points(ours) != points(theirs) {
        let shape = |f: &FieldInfo| {
            format!(
                "dimensionCount={}, indexDimensionCount={}, numBytes={}",
                f.point_dimension_count, f.point_index_dimension_count, f.point_num_bytes
            )
        };
        return Some(format!(
            "cannot change field \"{name}\" from points {} to inconsistent {}",
            shape(ours),
            shape(theirs)
        ));
    }
    let vectors = |f: &FieldInfo| {
        (
            f.vector_dimension,
            f.vector_encoding,
            f.vector_similarity_function,
        )
    };
    if vectors(ours) != vectors(theirs) {
        let shape = |f: &FieldInfo| {
            format!(
                "vector dimension={}, vector encoding={}, vector similarity function={}",
                f.vector_dimension,
                vector_encoding_name(f.vector_encoding),
                similarity_name(f.vector_similarity_function)
            )
        };
        return Some(format!(
            "cannot change field \"{name}\" from {} to inconsistent {}",
            shape(ours),
            shape(theirs)
        ));
    }
    if (ours.soft_deletes_field, ours.parent_field)
        != (theirs.soft_deletes_field, theirs.parent_field)
    {
        return Some(format!(
            "cannot change field \"{name}\" from soft-deletes field={}, parent field={} to \
             inconsistent soft-deletes field={}, parent field={}",
            ours.soft_deletes_field,
            ours.parent_field,
            theirs.soft_deletes_field,
            theirs.parent_field
        ));
    }
    None
}

/// `IndexingChain.processDocument`'s `IllegalArgumentException` for a
/// document that carries the parent field itself.
fn reserved_parent_field(name: &str) -> Error {
    Error::Document(crate::document::Error::IllegalArgument(format!(
        "\"{name}\" is a reserved field and should not be added to any document"
    )))
}

/// The `FieldInfo` of the parent field: `IndexingChain`'s
/// `new NumericDocValuesField(parentField, -1)`, flagged
/// `FieldInfo.isParentField()`.
fn parent_field_info(name: &str) -> FieldInfo {
    let mut info = FieldInfo::new(name, 0);
    info.doc_values_type = DocValuesType::Numeric;
    info.parent_field = true;
    info
}

/// `parents.nextSetBit(doc)` for every document of a buffer or segment:
/// `keys[doc]` becomes the key of the parent closing `doc`'s block, which is
/// how `IndexingChain.maybeSortSegment`, `Sorter.sort` and `MultiSorter.sort`
/// wrap the index sort's comparator once a segment has blocks and a parent
/// field (`in.compare(parents.nextSetBit(d1), parents.nextSetBit(d2))`). A
/// document after the last parent -- which a writer configured with a parent
/// field never produces, every add ending in one -- keeps its own key.
pub(crate) fn key_of_parent(keys: &mut [Option<i64>], is_parent: &dyn Fn(usize) -> bool) {
    let mut parent_key: Option<Option<i64>> = None;
    for doc in (0..keys.len()).rev() {
        if is_parent(doc) {
            parent_key = Some(keys[doc]);
        } else if let Some(key) = parent_key {
            keys[doc] = key;
        }
    }
}

/// Buffered postings bytes from which a flush streams its `.doc`/`.pos`/`.pay`
/// to the directory: they are as large as the postings' encoding of the buffer.
/// A test lowers or raises it to compare the two ways of writing.
pub(crate) static POSTINGS_STREAM_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(4 << 20);

impl IndexingConfig {
    /// Whether this configuration takes explicit documents.
    pub(crate) fn takes_explicit(&self) -> bool {
        self.explicit
    }

    /// `FreqProxTermsWriter.flush`: the buffered postings' files. A
    /// block-tree segment takes each field's terms a batch at a time,
    /// straight from the buffer ([`PendingPostings::stream_field_below`]); an
    /// installed postings writer or another terms dictionary needs every
    /// field laid out first.
    fn write_pending_postings(
        &self,
        pending: &PendingPostings,
        fields: &[(super::pending_postings::FieldToFlush, IndexOptions)],
        old_to_new: Option<&[usize]>,
        range: Option<(i32, i32)>,
        norms: &[postings_writer::FieldNorms<'_>],
        state: lucene_codecs::postings_consumer::PostingsWriteState<'_>,
    ) -> Result<super::PostingsGroups> {
        let name_of = |n: i32| {
            self.fields
                .iter()
                .find(|f| f.number == n)
                .map(|f| f.name.clone())
        };
        let format_for = |name: &str| self.postings_format_for(name);
        if !self.has_installed_postings_writer() {
            let specs: Vec<per_field_postings::StreamedFieldSpec> = fields
                .iter()
                .map(|(f, options)| per_field_postings::StreamedFieldSpec {
                    field_number: f.number,
                    index_options: *options,
                    has_payloads: f.payloads,
                })
                .collect();
            // A big flush writes `.doc`/`.pos`/`.pay` as they are encoded: held
            // whole they are as large as the postings' encoding of the buffer.
            let stream_from = POSTINGS_STREAM_BYTES.load(std::sync::atomic::Ordering::Relaxed);
            let sink = (pending.bytes() >= stream_from).then_some((state.directory, state.segment));
            let streamed = per_field_postings::write_streaming::<Error>(
                sink,
                &specs,
                state.field_infos,
                &name_of,
                &format_for,
                state.id,
                &mut |i, writer| {
                    let (field, _) = fields[i];
                    pending.stream_field_below(field.number, old_to_new, range, &mut |terms| {
                        writer.add_terms(terms, norms)?;
                        Ok(())
                    })?;
                    Ok(field.doc_count)
                },
            )?;
            if let Some(groups) = streamed {
                return Ok(groups);
            }
        }
        let flushed = pending.flushed_below(old_to_new, range)?;
        let term_lists = flushed.term_lists();
        let inputs: Vec<FieldPostingsInput<'_, _>> = term_lists
            .iter()
            .filter_map(|(n, docs, terms)| {
                let (f, options) = fields.iter().find(|(f, _)| f.number == *n)?;
                Some((
                    n,
                    if range.is_some() { &f.doc_count } else { docs },
                    terms,
                    f,
                    options,
                ))
            })
            .map(|(n, docs, terms, f, options)| FieldPostingsInput {
                field_number: *n,
                index_options: *options,
                doc_count: *docs,
                has_payloads: f.payloads,
                terms,
            })
            .collect();
        self.write_postings_groups(&inputs, norms, state)
    }

    /// Whether the codec installs its own postings writer, which writes to
    /// the directory and takes every field laid out whole.
    fn has_installed_postings_writer(&self) -> bool {
        self.codec
            .as_deref()
            .and_then(|c| c.postings_writer_factory())
            .is_some()
    }

    /// Whether a flush sorts the buffered documents.
    pub(crate) fn sorts_documents(&self) -> bool {
        self.index_sort.is_some()
    }

    /// The stored-fields mode a buffer compresses its explicit documents'
    /// stored values in as they arrive, or `None` when its flush needs them
    /// as documents: an index sort reorders them, and an installed
    /// stored-fields writer writes them itself.
    pub(crate) fn streamed_stored_mode(&self) -> Option<lucene_codecs::stored_fields::Mode> {
        if !self.explicit || self.index_sort.is_some() {
            return None;
        }
        let codec = self.codec.as_deref();
        if codec.is_some_and(|c| c.stored_fields_writer_factory().is_some()) {
            return None;
        }
        Some(
            codec.map_or(lucene_codecs::stored_fields::Mode::BestSpeed, |c| {
                c.stored_fields_mode()
            }),
        )
    }

    /// Checks an add of explicit documents against the registered schema and
    /// returns, per document and inverted field, the field's index options
    /// (`None` for a field with no terms).
    pub(crate) fn explicit_add_options(
        &self,
        docs: &[ExplicitDocument],
    ) -> Result<Vec<Vec<Option<IndexOptions>>>> {
        if !self.explicit {
            return Err(explicit_error("explicit documents are not enabled"));
        }
        docs.iter()
            .map(|doc| {
                self.validate_explicit(doc)?;
                doc.fields
                    .inverted
                    .iter()
                    .map(|inv| {
                        if inv.terms.is_empty() {
                            Ok(None)
                        } else {
                            self.explicit_field(inv.field_number)
                                .map(|f| Some(f.index_options))
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// The registered number of the parent field, once a document has
    /// registered it.
    pub(crate) fn parent_field_number(&self) -> Option<i32> {
        let name = self.parent_field.as_deref()?;
        self.field_by_name(name)
            .filter(|f| f.parent_field)
            .map(|f| f.number)
    }

    /// The registered vector field named `name`.
    pub(crate) fn explicit_vector_field(&self, name: &str) -> Result<&FieldInfo> {
        self.fields
            .iter()
            .find(|f| f.name == name && f.vector_dimension > 0)
            .ok_or_else(|| {
                explicit_error(format!("field {name:?} is not a registered vector field"))
            })
    }

    /// One document's vectors against the registered schema: a known vector
    /// field, its encoding and dimension, at most once per document.
    pub(crate) fn validate_explicit_vectors(
        &self,
        vectors: &[super::DocumentVector],
    ) -> Result<()> {
        for (i, v) in vectors.iter().enumerate() {
            let f = self.explicit_vector_field(&v.field_name)?;
            if v.value.encoding() != f.vector_encoding {
                return Err(explicit_error(format!(
                    "field {:?}: vector encoding {:?} does not match the field's {:?}",
                    f.name,
                    v.value.encoding(),
                    f.vector_encoding
                )));
            }
            if i32::try_from(v.value.len()).ok() != Some(f.vector_dimension) {
                return Err(explicit_error(format!(
                    "field {:?}: vector of dimension {} does not match the field's {}",
                    f.name,
                    v.value.len(),
                    f.vector_dimension
                )));
            }
            if vectors[..i].iter().any(|w| w.field_name == v.field_name) {
                return Err(explicit_error(format!(
                    "VectorValuesField \"{}\" appears more than once in this document (only one \
                     value is allowed per field)",
                    f.name
                )));
            }
        }
        Ok(())
    }

    /// The registered field with number `n`.
    fn explicit_field(&self, n: i32) -> Result<&FieldInfo> {
        self.field_by_number(n)
            .ok_or_else(|| explicit_error(format!("field number {n} is not registered")))
    }

    /// Checks one document against the registered schema -- at add time,
    /// while the caller still has the document, as `IndexingChain` does.
    pub(crate) fn validate_explicit(&self, doc: &ExplicitDocument) -> Result<()> {
        // `IndexingChain.processDocument`: the parent field is the writer's to
        // set, on the last document of each block; a document carrying it is
        // refused.
        if let Some(parent) = self.parent_field_number() {
            let fields = &doc.fields;
            let touches = doc.stored.iter().any(|f| f.field_number == parent)
                || fields.inverted.iter().any(|f| f.field_number == parent)
                || fields.doc_values.iter().any(|f| f.field_number == parent)
                || fields.points.iter().any(|f| f.field_number == parent)
                || fields.term_vectors.iter().any(|f| f.field_number == parent)
                || fields.registered.contains(&parent);
            if touches {
                return Err(reserved_parent_field(
                    self.parent_field.as_deref().unwrap_or_default(),
                ));
            }
        }
        for s in &doc.stored {
            self.explicit_field(s.field_number)?;
        }
        let mut seen = InlineFieldNumbers::default();
        let mut sorted: Vec<u64> = Vec::new();
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
            // The terms' hashes, sorted, find a possible repeat with one
            // allocation per document and no byte comparison; only a
            // document with equal hashes walks the set, which finds the
            // first real repeat in order (or none, for a collision).
            // A field of one term (a keyword, an id) cannot repeat one.
            let repeats = inv.terms.len() > 1 && {
                sorted.clear();
                sorted.extend_from_slice(inv.terms.hashes());
                sorted.sort_unstable();
                sorted.windows(2).any(|w| w[0] == w[1])
            };
            let mut terms = BTreeSet::new();
            for t in inv.terms.iter() {
                if repeats && !terms.insert(t.term) {
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
                if !t.payloads.is_empty() {
                    if !positions {
                        return Err(explicit_error(format!(
                            "field {:?} does not index positions, so it cannot store payloads",
                            f.name
                        )));
                    }
                    if t.payloads.len() != t.freq as usize {
                        return Err(explicit_error(format!(
                            "field {:?}: payloads must be one per occurrence (freq {}), got {}",
                            f.name,
                            t.freq,
                            t.payloads.len()
                        )));
                    }
                }
            }
        }
        let mut seen = InlineFieldNumbers::default();
        for tv in &doc.fields.term_vectors {
            let f = self.explicit_field(tv.field_number)?;
            if !f.store_term_vectors {
                return Err(explicit_error(format!(
                    "field {:?} does not store term vectors",
                    f.name
                )));
            }
            if !seen.insert(tv.field_number) {
                return Err(explicit_error(format!(
                    "field {:?} has two term vectors in one document",
                    f.name
                )));
            }
            // A vector is a view of the document's own inversion.
            if !doc
                .fields
                .inverted
                .iter()
                .any(|inv| inv.field_number == tv.field_number)
            {
                return Err(explicit_error(format!(
                    "field {:?} has a term vector but is not inverted in this document",
                    f.name
                )));
            }
            if tv.has_payloads && !tv.has_positions {
                return Err(explicit_error(format!(
                    "field {:?}: term vector payloads need term vector positions",
                    f.name
                )));
            }
            if tv.terms.windows(2).any(|w| w[0].term >= w[1].term) {
                return Err(explicit_error(format!(
                    "field {:?}: term vector terms must be distinct and in byte order",
                    f.name
                )));
            }
            for t in &tv.terms {
                let n = usize::try_from(t.freq).unwrap_or(0);
                let shaped = |present: bool, v: Option<usize>| match v {
                    Some(len) => present && len == n,
                    None => !present,
                };
                if n == 0
                    || !shaped(tv.has_positions, t.positions.as_ref().map(Vec::len))
                    || !shaped(tv.has_offsets, t.start_offsets.as_ref().map(Vec::len))
                    || !shaped(tv.has_offsets, t.end_offsets.as_ref().map(Vec::len))
                    || !shaped(tv.has_payloads, t.payloads.as_ref().map(Vec::len))
                {
                    return Err(explicit_error(format!(
                        "field {:?}: a term vector term needs freq >= 1 and one position, \
                         offset and payload per occurrence for each axis the vector has",
                        f.name
                    )));
                }
            }
        }
        // Java records a vector for every term-vector field the document
        // inverts into terms; a caller that sends none would write empty ones.
        for inv in &doc.fields.inverted {
            if !inv.terms.is_empty()
                && !seen.contains(inv.field_number)
                && self.explicit_field(inv.field_number)?.store_term_vectors
            {
                return Err(explicit_error(format!(
                    "field {:?} stores term vectors but this document gives none",
                    self.explicit_field(inv.field_number)?.name
                )));
            }
        }
        let mut single = InlineFieldNumbers::default();
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
        let mut present = DocFieldNumbers::default();
        for doc_vectors in buf.vectors {
            for v in doc_vectors {
                present.insert(self.explicit_vector_field(&v.field_name)?.number);
            }
        }
        if let Some(stored) = buf.stored {
            stored.fields.iter().for_each(|&n| present.insert(n));
        }
        // One pass over the documents' fields, one document at a time:
        // the fields present, the norms, and whether any term vectors are
        // written (`TermVectorsConsumer.hasVectors`: some document inverts a
        // field that stores them, even one that produced no term).
        let mut norm_values: BTreeMap<i32, Vec<Option<i64>>> = BTreeMap::new();
        let mut has_vectors = false;
        let decoding = std::time::Instant::now();
        let columns = buf.limit.and_then(|l| l.columns.as_ref());
        let decode = |present: &mut DocFieldNumbers,
                      norm_values: &mut BTreeMap<i32, Vec<Option<i64>>>,
                      has_vectors: &mut bool| {
            explicit.for_each_with_roles(|i, fields| {
                if let Some(doc) = buf.docs.get(i) {
                    doc.fields
                        .iter()
                        .for_each(|f| present.insert(f.field_number));
                }
                let roles = [&fields.doc_values, &fields.points];
                for inv in &fields.inverted {
                    present.insert(inv.field_number);
                    if let Some(norm) = inv.norm {
                        norm_values
                            .entry(inv.field_number)
                            .or_insert_with(|| vec![None; max_doc])[i] = Some(norm);
                    }
                    *has_vectors |= self
                        .explicit_field(inv.field_number)
                        .is_ok_and(|info| info.store_term_vectors);
                }
                roles
                    .into_iter()
                    .flatten()
                    .for_each(|f| present.insert(f.field_number));
                fields
                    .term_vectors
                    .iter()
                    .for_each(|f| present.insert(f.field_number));
                *has_vectors |= !fields.term_vectors.is_empty();
                fields.registered.iter().for_each(|&n| present.insert(n));
                Ok(())
            })
        };
        // A run kept as columns is not decoded: its fields, norms, doc
        // values and points are read from them.
        let roles = match columns {
            Some(c) => {
                for doc in buf.docs {
                    doc.fields
                        .iter()
                        .for_each(|f| present.insert(f.field_number));
                }
                c.present.iter().for_each(|&n| present.insert(n));
                norm_values.extend(c.norms.iter().cloned());
                None
            }
            None => {
                let roles = decode(&mut present, &mut norm_values, &mut has_vectors)?;
                if buf.limit.is_some() {
                    crate::live_timing::record(
                        crate::live_timing::Phase::ChunkDecode,
                        decoding.elapsed(),
                    );
                }
                Some(roles)
            }
        };
        let (mut dv_role, points_role) = match roles {
            Some([dv, points]) => (Some(dv), Some(points)),
            None => (None, None),
        };
        let chunk = buf.limit.is_some();

        // Postings: one term dictionary per field, terms in byte order. The
        // writer inverted them as each document was buffered
        // (`pending_postings`); a buffer that still carries its terms is
        // inverted here the same way.
        let inverted_here;
        let pending = match buf.postings {
            Some(p) => p,
            None => {
                let mut p = PendingPostings::default();
                explicit.for_each(|doc_id, fields| {
                    let doc_id = i32::try_from(doc_id)
                        .map_err(|_| explicit_error("too many buffered documents"))?;
                    for inv in &fields.inverted {
                        let options = self.explicit_field(inv.field_number)?.index_options;
                        p.add_field(doc_id, inv, options)?;
                    }
                    Ok(())
                })?;
                inverted_here = p;
                &inverted_here
            }
        };
        let mut flush_fields = pending.fields_to_flush();
        if let Some(limit) = buf.limit {
            if !limit.postings {
                flush_fields.clear();
            }
            // The fields as the documents written hold them.
            flush_fields.retain_mut(|f| {
                f.doc_count = limit.doc_counts.get(&f.number).copied().unwrap_or(0);
                f.doc_count > 0
            });
        }
        let postings_fields: Vec<(super::pending_postings::FieldToFlush, IndexOptions)> =
            flush_fields
                .into_iter()
                .map(|f| Ok((f, self.explicit_field(f.number)?.index_options)))
                .collect::<Result<_>>()?;
        // `FieldInfo.setStorePayloads`: the fields a document of this segment
        // gave a non-empty payload. Every occurrence of such a field then
        // carries a payload length, zero where it has none -- what
        // `FreqProxTermsWriterPerField.writeProx` records once the flag is on.
        let payload_fields: BTreeSet<i32> = postings_fields
            .iter()
            .filter(|(f, _)| f.payloads)
            .map(|(f, _)| f.number)
            .collect();
        // A present norms field still owns an empty sparse column when no
        // document supplied a value, as Java's norms flush does.
        for field in &self.fields {
            if present.contains(&field.number)
                && field.index_options != IndexOptions::None
                && !field.omit_norms
            {
                norm_values
                    .entry(field.number)
                    .or_insert_with(|| vec![None; max_doc]);
            }
        }
        // Dense impact values and sparse presence share one owned column.
        let norm_columns: Vec<_> = norm_values
            .into_iter()
            .map(|(number, values)| (number, super::NormsColumn::from_values(values)))
            .collect();
        let postings_field_numbers: BTreeSet<i32> =
            postings_fields.iter().map(|(f, _)| f.number).collect();
        // This segment's FieldInfos before the per-field attributes: the
        // present fields, with the payloads this segment holds. `.psm` frames
        // `.pos`/`.pay` from them, and the `.fnm` below is written from them.
        let segment_fields: Vec<FieldInfo> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number))
            .map(|f| FieldInfo {
                store_payloads: payload_fields.contains(&f.number),
                ..f.clone()
            })
            .collect();
        let build_norms = || -> Result<_> {
            if norm_columns.is_empty() {
                return Ok(None);
            }
            let fields: Vec<_> = norm_columns
                .iter()
                .map(|(number, column)| match &column.sparse {
                    None => norms::NormsField::Dense(*number, &column.dense),
                    Some(pairs) => norms::NormsField::Sparse(*number, pairs),
                })
                .collect();
            let factory = self
                .codec
                .as_deref()
                .and_then(|codec| codec.norms_writer_factory());
            let output = if let Some(factory) = factory {
                let files = crate::norms_writer::write_installed(
                    factory,
                    lucene_codecs::norms_writer::NormsWriteState {
                        directory: dir,
                        segment: segment_name,
                        id: &segment_id,
                        suffix: "",
                        max_doc: max_doc as i32,
                    },
                    &segment_fields,
                    &fields,
                )?;
                (None, files)
            } else {
                (
                    Some(norms::write_fields(
                        &fields,
                        max_doc as i32,
                        &segment_id,
                        "",
                    )?),
                    Vec::new(),
                )
            };
            Ok(Some(output))
        };
        let installed_norms = self
            .codec
            .as_deref()
            .and_then(|codec| codec.norms_writer_factory())
            .is_some();
        let mut norms_output = if installed_norms {
            build_norms()?
        } else {
            None
        };
        if !installed_norms {
            norms_output = build_norms()?;
        }

        // The postings are the segment's largest build and need nothing
        // the doc values, points and vectors builds make: when they are
        // encoded in memory (no installed postings writer touches the
        // directory) they are built beside those on a thread of their own.
        let build_postings = || -> Result<Option<super::PostingsGroups>> {
            if postings_fields.is_empty() {
                return Ok(None);
            }
            let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = norm_columns
                .iter()
                .map(|(n, column)| postings_writer::FieldNorms {
                    field_number: *n,
                    values: &column.dense,
                })
                .collect();
            Ok(Some(self.write_pending_postings(
                pending,
                &postings_fields,
                buf.old_to_new,
                buf.limit.map(|l| (l.start, l.docs)),
                &norms_for_impacts,
                lucene_codecs::postings_consumer::PostingsWriteState {
                    directory: dir,
                    segment: segment_name,
                    id: &segment_id,
                    suffix: "",
                    max_doc: max_doc as i32,
                    field_infos: &segment_fields,
                    format: lucene_codecs::per_field_postings::Lucene104PostingsFormat::default(),
                },
            )?))
        };
        let postings_beside = !postings_fields.is_empty() && !self.has_installed_postings_writer();
        let (doc_values_output, points_output, vectors_output, postings_output) =
            std::thread::scope(|scope| -> Result<_> {
                let postings = postings_beside.then(|| scope.spawn(build_postings));
                // Doc values and points through the native builders, a role at a
                // time: each document's values read in place, or taken out of the
                // packed documents for that role alone and dropped once written.
                let dv_configs: Vec<DocValuesFieldConfig> = self
                    .fields
                    .iter()
                    .filter(|f| {
                        buf.limit.is_none_or(|l| l.doc_values)
                            && present.contains(&f.number)
                            && f.doc_values_type != DocValuesType::None
                    })
                    .map(|f| DocValuesFieldConfig {
                        name: f.name.clone(),
                        field_number: f.number,
                        doc_values_type: f.doc_values_type,
                        skip_index: f.doc_values_skip_index_type != DocValuesSkipIndexType::None,
                    })
                    .collect();
                let doc_values_output = if dv_configs.is_empty() {
                    None
                } else {
                    let names: Vec<&str> = self
                        .fields
                        .iter()
                        .filter(|f| present.contains(&f.number))
                        .map(|f| f.name.as_str())
                        .collect();
                    let building = std::time::Instant::now();
                    let prebuilt = columns.and_then(|c| {
                        self.build_prebuilt_doc_values_groups(
                            &c.doc_values,
                            max_doc as i32,
                            &dv_configs,
                            &names,
                            &segment_id,
                            !chunk,
                        )
                    });
                    let (groups, phase) = match prebuilt {
                        Some(groups) => (groups?, crate::live_timing::Phase::ChunkDocValuesColumns),
                        None => {
                            if dv_role.is_none() {
                                let [dv, _] = explicit.for_each_with_roles(|_, _| Ok(()))?;
                                dv_role = Some(dv);
                            }
                            let docs = dv_role
                                .as_ref()
                                .map(super::RoleValues::per_doc)
                                .unwrap_or_default();
                            let groups = self.build_doc_values_groups(
                                &docs,
                                &dv_configs,
                                &names,
                                &segment_id,
                                !chunk,
                            )?;
                            (groups, crate::live_timing::Phase::ChunkDocValuesDecoded)
                        }
                    };
                    if chunk {
                        crate::live_timing::record(phase, building.elapsed());
                    }
                    Some(groups)
                };
                // Each role's values are freed once written, not held while
                // the points, vectors and postings builds run.
                drop(dv_role);
                let point_configs: Vec<PointsFieldConfig> = self
                    .fields
                    .iter()
                    .filter(|f| {
                        buf.limit.is_none_or(|l| l.points)
                            && present.contains(&f.number)
                            && f.point_dimension_count > 0
                    })
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
                    let building = std::time::Instant::now();
                    let out = match (columns, &points_role) {
                        (_, Some(role)) => IndexWriter::build_points_output(
                            self,
                            &role.per_doc(),
                            &point_configs,
                            &segment_id,
                            dir,
                            segment_name,
                        )?,
                        (Some(c), None) => IndexWriter::build_prebuilt_points_output(
                            self,
                            &c.points,
                            &point_configs,
                            &segment_id,
                            dir,
                            segment_name,
                            max_doc as i32,
                        )?,
                        (None, None) => None,
                    };
                    if chunk {
                        crate::live_timing::record(
                            crate::live_timing::Phase::ChunkPoints,
                            building.elapsed(),
                        );
                    }
                    out
                };
                drop(points_role);

                let vector_configs: Vec<super::VectorFieldConfig> = self
                    .fields
                    .iter()
                    .filter(|f| present.contains(&f.number) && f.vector_dimension > 0)
                    .map(|f| super::VectorFieldConfig {
                        name: f.name.clone(),
                        field_number: f.number,
                        dimension: f.vector_dimension,
                        encoding: f.vector_encoding,
                        similarity: f.vector_similarity_function,
                    })
                    .collect();
                let vectors_output = if vector_configs.is_empty() {
                    None
                } else {
                    let partition_starts = self.ivf_partition_starts(buf)?;
                    IndexWriter::build_vectors_output(
                        self.ivf_partitions(partition_starts.as_deref()),
                        buf.vectors,
                        &vector_configs,
                        i32::try_from(max_doc).map_err(|_| explicit_error("too many documents"))?,
                        &|name: &str| self.knn_vectors_format_for(name),
                        self.knn_vectors_direct(),
                        buf.limit.is_none(),
                        buf.graphs,
                        buf.old_to_new,
                        &self.ivf_clusters,
                        &segment_id,
                    )?
                };
                let postings_output = match postings {
                    Some(job) => job
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?,
                    None => build_postings()?,
                };
                Ok((
                    doc_values_output,
                    points_output,
                    vectors_output,
                    postings_output,
                ))
            })?;
        let dv_suffix = |number: i32| {
            doc_values_output
                .as_deref()
                .and_then(|groups| groups.iter().find(|g| g.field_numbers.contains(&number)))
                .map(|g| (g.format_name, g.suffix))
        };

        // Term vectors: each document's fields in name order
        // (`TermVectorsConsumer.finishDocument` sorts its `perFields` by
        // name). Written when some document inverts a field that stores term
        // vectors, even one that produced no term: that is when
        // `TermVectorsConsumer.hasVectors` opens the writer.
        let term_vectors_output = if !has_vectors {
            None
        } else {
            let mut docs = Vec::with_capacity(max_doc);
            explicit.for_each(|_, f| {
                let mut named = Vec::with_capacity(f.term_vectors.len());
                for tv in &f.term_vectors {
                    named.push((
                        self.explicit_field(tv.field_number)?.name.as_str(),
                        tv.clone(),
                    ));
                }
                // `String.compareTo`: UTF-16 code units, not bytes.
                named.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
                docs.push(TermVectorsDocument {
                    fields: named.into_iter().map(|(_, tv)| tv).collect(),
                });
                Ok(())
            })?;
            Some(self.write_term_vectors_output(
                &docs,
                &segment_fields,
                lucene_codecs::term_vectors_writer::TermVectorsWriteState {
                    directory: dir,
                    segment: segment_name,
                    id: &segment_id,
                    suffix: self.term_vectors_format().map_or("", |f| f.segment_suffix),
                    max_doc: max_doc as i32,
                    format: self.term_vectors_format(),
                },
            )?)
        };
        let vector_fields_written: Vec<super::WrittenVectorField> = vectors_output
            .as_ref()
            .map(|o| o.written.clone())
            .unwrap_or_default();

        // This segment's FieldInfos: the present fields, full schema, and the
        // per-field format attributes for what was written.
        let fnm_fields: Vec<FieldInfo> = segment_fields
            .iter()
            .map(|f| {
                let mut f = f.clone();
                f.attributes.retain(|(k, _)| !k.starts_with("PerField"));
                // A direct postings group's fields record no attributes.
                let postings_suffix = postings_output.as_ref().and_then(|groups| {
                    groups
                        .iter()
                        .find(|g| g.field_numbers.contains(&f.number))
                        .filter(|g| !g.direct)
                        .map(|g| (g.format_name, g.suffix))
                });
                if let (true, Some((format_name, suffix))) =
                    (postings_field_numbers.contains(&f.number), postings_suffix)
                {
                    f.attributes.push((
                        "PerFieldPostingsFormat.format".to_string(),
                        format_name.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldPostingsFormat.suffix".to_string(),
                        suffix.to_string(),
                    ));
                }
                if let Some((format_name, suffix)) = dv_suffix(f.number) {
                    f.attributes.push((
                        per_field_doc_values::PER_FIELD_FORMAT_KEY.to_string(),
                        format_name.to_string(),
                    ));
                    f.attributes.push((
                        per_field_doc_values::PER_FIELD_SUFFIX_KEY.to_string(),
                        suffix.to_string(),
                    ));
                }
                if let Some(written) = vector_fields_written
                    .iter()
                    .find(|w| w.name == f.name && !w.direct)
                {
                    f.attributes.push((
                        "PerFieldKnnVectorsFormat.format".to_string(),
                        written.format.to_string(),
                    ));
                    f.attributes.push((
                        "PerFieldKnnVectorsFormat.suffix".to_string(),
                        written.suffix.to_string(),
                    ));
                }
                f
            })
            .collect();

        super::before_write(
            self.codec.as_deref(),
            lucene_codecs::codec::WritePoint::FieldInfos,
        )?;
        let fnm_fields = self.partition_fields(fnm_fields, buf.partition);
        let mut flushed = segment_writer::write_stored_only_segment_files_with_codec(
            dir,
            segment_name,
            segment_id,
            &self.codec_name,
            self.lucene_version,
            &fnm_fields,
            buf.docs,
            buf.has_blocks,
            self.codec.as_deref(),
            buf.stored.and_then(|s| {
                s.bytes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
            }),
        )?;
        let mut record = |names: Vec<String>| {
            flushed.info.files.extend(names);
        };
        if let Some(output) = postings_output {
            record(
                output
                    .iter()
                    .flat_map(|g| g.streamed_files.clone())
                    .collect(),
            );
            record(IndexWriter::write_postings_files(
                dir,
                segment_name,
                &output,
            )?);
        }
        for group in doc_values_output.iter().flatten() {
            record(super::with_doc_values_faults(
                self.codec.as_deref(),
                group.field_numbers.len(),
                || IndexWriter::write_doc_values_files(dir, segment_name, group),
            )?);
        }
        if let Some((bytes, files)) = norms_output {
            record(files);
            if let Some((nvm, nvd)) = bytes {
                record(IndexWriter::write_norms_files(
                    dir,
                    segment_name,
                    &nvm,
                    &nvd,
                )?);
            }
        }
        if let Some(output) = &points_output {
            record(IndexWriter::write_points_files(dir, segment_name, output)?);
        }
        if let Some(output) = &vectors_output {
            record(IndexWriter::write_vector_files(dir, segment_name, output)?);
        }
        if let Some(output) = term_vectors_output {
            record(output.files);
            if let Some((tvd, tvx, tvm)) = output.bytes {
                record(IndexWriter::write_term_vector_files(
                    dir,
                    segment_name,
                    self.term_vectors_format().map_or("", |f| f.segment_suffix),
                    &tvd,
                    &tvx,
                    &tvm,
                )?);
            }
        }
        // The buffer was put in sort order before this build (`sort_buffer`);
        // the `.si` records the sort, as the native path's does.
        if let Some(sort) = &self.index_sort {
            flushed.info.index_sort = Some(sort.clone());
        }
        let extra = super::write_codec_segment_files(
            self.codec.as_deref(),
            lucene_codecs::codec::SegmentWrite {
                directory: dir,
                segment: segment_name,
                id: &segment_id,
                max_doc: flushed.info.doc_count,
                field_infos: &fnm_fields,
                files: &flushed.info.files,
            },
        )?;
        flushed.info.files.extend(extra);
        if self.use_compound_file {
            super::before_write(
                self.codec.as_deref(),
                lucene_codecs::codec::WritePoint::Compound,
            )?;
            segment_writer::create_compound_file(dir, segment_name, &mut flushed.info)?;
        }
        super::before_write(
            self.codec.as_deref(),
            lucene_codecs::codec::WritePoint::SegmentInfo,
        )?;
        self.partition_segment(&mut flushed.info.attributes, buf.partition);
        self.record_list_attributes(&mut flushed.info.attributes);
        segment_writer::seal_flushed_segment(dir, segment_name, flushed).map_err(Error::from)
    }
}

/// What a reader of a run of buffered documents needs besides their doc
/// values ([`super::LazyDocValues`]) and terms, built in memory without
/// writing a segment: [`IndexingConfig::build_lean_parts`].
#[derive(Debug, Clone)]
pub struct LeanParts {
    /// The fields the documents have, as a written segment of them lists
    /// them.
    pub field_infos: Vec<FieldInfo>,
    /// How many documents there are.
    pub max_doc: i32,
    /// The norms as `.nvm` and `.nvd` bytes, when some field has norms.
    pub norms: Option<(Vec<u8>, Vec<u8>)>,
    /// The points as `.kdm`, `.kdi` and `.kdd` bytes, when they were asked
    /// for and some document has one.
    pub points: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// The segment id the norms and points are encoded for.
    pub segment_id: [u8; ID_LENGTH],
}

impl IndexingConfig {
    /// Whether a run kept as `columns` is read from
    /// [`Self::build_lean_parts`]' parts: no codec of the caller's writes
    /// its files and no document has a vector.
    pub(crate) fn takes_lean_parts(&self, columns: &super::ChunkColumns) -> bool {
        self.codec.is_none()
            && !self
                .fields
                .iter()
                .any(|f| f.vector_dimension > 0 && columns.present.contains(&f.number))
    }

    /// A run of `max_doc` buffered documents kept as `columns`, as the
    /// parts of a segment without postings, doc values or stored fields:
    /// what [`Self::build_and_write_explicit_segment`] writes of them,
    /// without the files. For a run [`Self::takes_lean_parts`] takes.
    pub(crate) fn build_lean_parts(
        &self,
        columns: &super::ChunkColumns,
        max_doc: i32,
        points: bool,
        partition: Option<&str>,
        segment_id: [u8; ID_LENGTH],
    ) -> Result<LeanParts> {
        let mut present = DocFieldNumbers::default();
        columns.present.iter().for_each(|&n| present.insert(n));
        let docs = usize::try_from(max_doc).map_err(|_| explicit_error("a negative run"))?;
        let mut norm_values: BTreeMap<i32, Vec<Option<i64>>> =
            columns.norms.iter().cloned().collect();
        for field in &self.fields {
            if present.contains(&field.number)
                && field.index_options != IndexOptions::None
                && !field.omit_norms
            {
                norm_values
                    .entry(field.number)
                    .or_insert_with(|| vec![None; docs]);
            }
        }
        let norm_columns: Vec<_> = norm_values
            .into_iter()
            .map(|(number, values)| (number, super::NormsColumn::from_values(values)))
            .collect();
        let norms = if norm_columns.is_empty() {
            None
        } else {
            let fields: Vec<_> = norm_columns
                .iter()
                .map(|(number, column)| match &column.sparse {
                    None => norms::NormsField::Dense(*number, &column.dense),
                    Some(pairs) => norms::NormsField::Sparse(*number, pairs),
                })
                .collect();
            Some(norms::write_fields(&fields, max_doc, &segment_id, "")?)
        };
        let points = if points {
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
            if point_configs.is_empty() {
                None
            } else {
                // Without a codec the points are encoded in memory: the
                // directory is not written to.
                let unused = lucene_store::ByteBuffersDirectory::new();
                IndexWriter::build_prebuilt_points_output(
                    self,
                    &columns.points,
                    &point_configs,
                    &segment_id,
                    &unused,
                    "",
                    max_doc,
                )?
                .and_then(|o| o.bytes)
            }
        } else {
            None
        };
        let field_infos: Vec<FieldInfo> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number))
            .map(|f| {
                let mut f = f.clone();
                f.store_payloads = false;
                f.attributes.retain(|(k, _)| !k.starts_with("PerField"));
                f
            })
            .collect();
        Ok(LeanParts {
            field_infos: self.partition_fields(field_infos, partition),
            max_doc,
            norms,
            points,
            segment_id,
        })
    }
}

/// `FieldInfos.FieldNumbers`' number reservations. Source segments supply
/// preferred numbers; newly registered explicit fields request the first free
/// number. Keeping this separate from the configured schema preserves fields
/// that have not reappeared yet, even after their source segments are removed.
/// Their schema remains fixed for the same lifetime as their reserved number.
#[derive(Clone, Default)]
pub(super) struct FieldNumbers {
    by_name: BTreeMap<String, i32>,
    schemas: BTreeMap<String, FieldInfo>,
    used: BTreeSet<i32>,
    lowest_unassigned: i32,
}

impl FieldNumbers {
    pub(super) fn schema(&self, name: &str) -> Option<&FieldInfo> {
        self.schemas.get(name)
    }

    pub(super) fn register(&mut self, info: &FieldInfo, preferred: Option<i32>) -> Result<i32> {
        if let Some(existing) = self.schema(&info.name) {
            if let Some(message) = schema_change(existing, info) {
                return Err(explicit_error(message));
            }
        }
        let number = self.add_or_get(&info.name, preferred)?;
        self.schemas.entry(info.name.clone()).or_insert_with(|| {
            let mut schema = info.clone();
            schema.number = number;
            schema.doc_values_gen = -1;
            schema.attributes.clear();
            schema.store_payloads = false;
            schema
        });
        Ok(number)
    }

    fn add_or_get(&mut self, name: &str, preferred: Option<i32>) -> Result<i32> {
        if let Some(&number) = self.by_name.get(name) {
            return Ok(number);
        }
        let number = if let Some(number) = preferred.filter(|n| *n >= 0 && !self.used.contains(n)) {
            number
        } else {
            while self.used.contains(&self.lowest_unassigned) {
                self.lowest_unassigned = self
                    .lowest_unassigned
                    .checked_add(1)
                    .ok_or_else(|| explicit_error("field number limit exceeded"))?;
            }
            self.lowest_unassigned
        };
        self.used.insert(number);
        self.by_name.insert(name.to_owned(), number);
        Ok(number)
    }
}

impl<'d> IndexWriter<'d> {
    /// Switches this writer to [`ExplicitDocument`]s. Must be called before
    /// any document is buffered; refused with an index sort, and once on, the
    /// native [`Document`] entry points are refused.
    pub fn enable_explicit_documents(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        if self.cfg.explicit {
            return Ok(());
        }
        if !self.pending_docs.is_empty() {
            return Err(explicit_error(
                "explicit documents must be enabled before any document is buffered",
            ));
        }
        // An index sort's fields were checked against the registered ones by
        // `set_index_sort`; fields registered from here are checked by
        // `register_field`.
        // `IndexWriter.getFieldNumberMap`: reserve every opened field before
        // the first new document can introduce another name. Preferred source
        // numbers may conflict across segments; the first name keeps its
        // number, and later conflicts take the first free number.
        let mut numbers = FieldNumbers::default();
        for field in &self.cfg.fields {
            numbers.register(field, Some(field.number))?;
        }
        for segment in self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
        {
            let mut fields = super::segment_field_infos(self.dir, segment)?.fields;
            // Java's FieldInfos iterator visits ascending field numbers even
            // when an imported file recorded entries in another order.
            fields.sort_by_key(|field| field.number);
            for field in fields {
                numbers.register(&field, Some(field.number))?;
            }
        }
        let cfg = self.cfg_mut();
        cfg.explicit_field_numbers = numbers;
        cfg.explicit = true;
        Ok(())
    }

    /// Drops source-only field properties after deleteAll while retaining
    /// this facade's configured schemas and the field IDs callers hold.
    pub(super) fn reset_explicit_field_numbers(&mut self) -> Result<()> {
        if self.cfg.explicit {
            let mut numbers = FieldNumbers::default();
            for field in &self.cfg.fields {
                numbers.register(field, Some(field.number))?;
            }
            self.cfg_mut().explicit_field_numbers = numbers;
        }
        Ok(())
    }

    /// Checks the restored sources before a reusable rollback changes any
    /// document state. A schema registered after deleteAll may conflict with
    /// those sources; rejecting the rollback keeps the current writer coherent.
    pub(super) fn prepare_rollback_field_numbers(&self) -> Result<Option<FieldNumbers>> {
        if !self.cfg.explicit {
            return Ok(None);
        }
        let mut numbers = self.cfg.explicit_field_numbers.clone();
        for field in &self.cfg.fields {
            numbers.register(field, Some(field.number))?;
        }
        for segment in &self.rollback_segments {
            let mut fields = super::segment_field_infos(self.dir, segment)?.fields;
            fields.sort_by_key(|field| field.number);
            for field in fields {
                numbers.register(&field, Some(field.number))?;
            }
        }
        Ok(Some(numbers))
    }

    /// `FieldInfos.FieldNumbers.addOrGet`: the global number of a field with
    /// this schema, registering it on first sight. `info.number` is ignored.
    /// A known name with a different schema is refused, as Lucene refuses to
    /// change a field's index options, doc-values type or point shape.
    pub fn register_field(&mut self, info: FieldInfo) -> Result<i32> {
        self.write_pending_try_changes()?;
        if !self.cfg.explicit {
            return Err(explicit_error(
                "register_field needs explicit documents enabled",
            ));
        }
        if info.store_payloads {
            return Err(explicit_error(format!(
                "field {:?}: payloads are a segment's, not a registered schema's",
                info.name
            )));
        }
        // `FieldInfo.checkConsistency`.
        if info.store_term_vectors && info.index_options == IndexOptions::None {
            return Err(explicit_error(format!(
                "non-indexed field {:?} cannot have term vectors",
                info.name
            )));
        }
        // `FieldInfo`'s own rule, caught here rather than at flush -- where it
        // would fail a whole segment -- as Java catches it at `addDocument`.
        if info.doc_values_skip_index_type != DocValuesSkipIndexType::None
            && matches!(
                info.doc_values_type,
                DocValuesType::Binary | DocValuesType::None
            )
        {
            return Err(explicit_error(format!(
                "field {:?}: a doc-values skip index needs NUMERIC, SORTED_NUMERIC, SORTED or \
                 SORTED_SET doc values",
                info.name
            )));
        }
        // `FieldInfos.FieldNumbers.verifyParentFieldName`.
        match (self.cfg.parent_field.as_deref(), info.parent_field) {
            (None, true) => {
                return Err(explicit_error(format!(
                    "can't add field [{}] as parent document field; this IndexWriter has no \
                     parent document field configured",
                    info.name
                )));
            }
            (Some(parent), true) if parent != info.name => {
                return Err(explicit_error(format!(
                    "can't add field [{}] as parent document field; this IndexWriter is \
                     configured with [{parent}] as parent document field",
                    info.name
                )));
            }
            (Some(parent), false) if parent == info.name => {
                return Err(explicit_error(format!(
                    "can't add [{}] as non parent document field; this IndexWriter is \
                     configured with [{parent}] as parent document field",
                    info.name
                )));
            }
            _ => {}
        }
        // `IndexingChain.validateIndexSortDVType`: a field the index sort
        // names must carry the doc values the sort reads.
        if let Some(sf) = self
            .cfg
            .index_sort
            .iter()
            .flatten()
            .find(|sf| sf.field == info.name)
        {
            let wanted = super::sort_doc_values_type(sf);
            if info.doc_values_type != wanted {
                return Err(explicit_error(format!(
                    "invalid doc value type: {:?} for sortField: {:?}, expected {wanted:?}",
                    info.doc_values_type, sf.field
                )));
            }
        }
        if let Some(existing) = self.cfg.field_by_name(&info.name) {
            if let Some(message) = schema_change(existing, &info) {
                return Err(explicit_error(message));
            }
            return Ok(existing.number);
        }
        // `FieldNumbers` starts from every segment of the index the writer
        // opened (`enable_explicit_documents`), so `register` checks a field
        // another writer indexed against its schema without re-reading every
        // segment's `.fnm` per new field.
        let number = self
            .cfg_mut()
            .explicit_field_numbers
            .register(&info, None)?;
        let mut info = info;
        info.number = number;
        info.doc_values_gen = -1;
        // A field type's own attributes (`FieldType.putAttribute`) reach
        // every segment's `.fnm`; the per-field format ones are the writer's.
        info.attributes.retain(|(k, _)| !k.starts_with("PerField"));
        self.cfg_mut().push_field(info);
        Ok(number)
    }

    /// The schema the index's segments give the field named `name`, from
    /// the first segment holding it: what `IndexWriter`'s constructor reads
    /// into `globalFieldNumberMap` (`getFieldNumberMap`) for a field this
    /// writer has not registered itself.
    pub(crate) fn segments_schema(&self, name: &str) -> Result<Option<FieldInfo>> {
        let segments: Vec<SegmentCommitInfo> = self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
            .cloned()
            .collect();
        for sci in &segments {
            let infos = super::segment_field_infos(self.dir, sci)?;
            if let Some(f) = infos.fields.into_iter().find(|f| f.name == name) {
                return Ok(Some(f));
            }
        }
        Ok(None)
    }

    /// The registered field named `name`, if any.
    pub(crate) fn registered_field(&self, name: &str) -> Option<&FieldInfo> {
        self.cfg.field_by_name(name)
    }

    /// `IndexWriterConfig.setParentField(name)`: from here on the last
    /// document of every add -- a single document, or the parent closing a
    /// block -- carries a NUMERIC doc value of `-1` in the field `name`,
    /// whose `FieldInfo` is flagged `isParentField`. With an index sort, a
    /// flush or merge then moves whole blocks, each ordered by its parent
    /// (`IndexingChain.maybeSortSegment`, `MultiSorter.sort`); without one,
    /// adding a block to a sorted index is refused.
    ///
    /// The parent field is a property of the whole index, as Java fixes it
    /// on `IndexWriterConfig` before the writer exists: it must be set
    /// before any document is buffered, switches the writer to explicit
    /// documents (the only ones it can mark), and is refused for an index
    /// that already has fields but not this one (`IndexWriter`'s
    /// constructor, "can't add a parent field to an already existing index
    /// without a parent field"), or that uses `name` for something else, or
    /// another field as its parent (`FieldNumbers.verifyParentFieldName`).
    /// `None` removes it, for a writer that has buffered nothing.
    pub fn set_parent_field(&mut self, name: Option<&str>) -> Result<()> {
        if !self.pending_docs.is_empty() {
            return Err(explicit_error(
                "the parent field must be set before any document is buffered",
            ));
        }
        let Some(name) = name else {
            if let Some(f) = self.cfg.fields.iter().find(|f| f.parent_field) {
                return Err(explicit_error(format!(
                    "this index uses [{}] as parent document field; the writer must too",
                    f.name
                )));
            }
            self.cfg_mut().parent_field = None;
            return Ok(());
        };
        if self
            .cfg
            .fields
            .iter()
            .any(|f| f.soft_deletes_field && f.name == name)
        {
            return Err(explicit_error(format!(
                "parent document and soft-deletes field can't be the same field \"{name}\""
            )));
        }
        for f in &self.cfg.fields {
            if f.parent_field && f.name != name {
                return Err(explicit_error(format!(
                    "can't add field [{}] as parent document field; this IndexWriter is \
                     configured with [{name}] as parent document field",
                    f.name
                )));
            }
            if !f.parent_field && f.name == name {
                return Err(explicit_error(format!(
                    "can't add [{name}] as non parent document field; this IndexWriter is \
                     configured with [{name}] as parent document field"
                )));
            }
        }
        // `IndexWriter`'s constructor: a non-empty index without the field.
        let mut any_field = !self.cfg.fields.is_empty();
        let mut has_parent = self.cfg.fields.iter().any(|f| f.name == name);
        let segments: Vec<SegmentCommitInfo> = self
            .segment_infos
            .segments
            .iter()
            .chain(&self.flushed_segments)
            .cloned()
            .collect();
        for sci in &segments {
            let infos = super::segment_field_infos(self.dir, sci)?;
            any_field |= !infos.fields.is_empty();
            for f in &infos.fields {
                if f.name == name {
                    if !f.parent_field {
                        return Err(explicit_error(format!(
                            "can't add [{name}] as non parent document field; this IndexWriter \
                             is configured with [{name}] as parent document field"
                        )));
                    }
                    has_parent = true;
                } else if f.parent_field {
                    return Err(explicit_error(format!(
                        "can't add field [{}] as parent document field; this IndexWriter is \
                         configured with [{name}] as parent document field",
                        f.name
                    )));
                }
            }
        }
        if any_field && !has_parent {
            return Err(explicit_error(
                "can't add a parent field to an already existing index without a parent field",
            ));
        }
        self.enable_explicit_documents()?;
        self.cfg_mut().parent_field = Some(name.to_string());
        Ok(())
    }

    /// `IndexWriterConfig.getParentField()`.
    pub fn parent_field(&self) -> Option<&str> {
        self.cfg.parent_field.as_deref()
    }

    /// The parent field's number, registering it on first use -- Java's
    /// `IndexingChain` initializes the parent field's `FieldInfo` the first
    /// time a segment sees a last-in-block document. `None` without a parent
    /// field.
    pub(crate) fn register_parent_field(&mut self) -> Result<Option<i32>> {
        self.write_pending_try_changes()?;
        // Already registered as the parent field: nothing to check again.
        // This runs once per added document, and `register_field` builds a
        // whole `FieldInfo` and scans the index sort each time.
        if let Some(number) = self.cfg.parent_field_number() {
            return Ok(Some(number));
        }
        let Some(name) = self.cfg.parent_field.clone() else {
            return Ok(None);
        };
        self.register_field(parent_field_info(&name)).map(Some)
    }

    /// `IndexingChain.processDocument(docId, doc, lastDocInBlock)` for an
    /// add of stored-only [`Document`]s: the last document buffered closes
    /// the add, so it is the parent, `-1` in the parent field. Without a
    /// parent field there is nothing to mark.
    pub(crate) fn mark_last_buffered_parent(&mut self) -> Result<()> {
        self.write_pending_try_changes()?;
        if let Some(parent) = self.register_parent_field()? {
            if let Some(last) = self.pending_explicit.last_mut() {
                last.doc_values.push(StoredField {
                    field_number: parent,
                    value: FieldValue::Long(-1),
                });
            }
        }
        Ok(())
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
        self.write_pending_try_changes()?;
        self.explicit_documents_check(&docs)?;
        self.add_explicit_with_delete(None, docs, None)
    }

    /// `IndexWriter.updateDocuments(term, docs)` for explicit documents.
    pub fn update_explicit_documents(
        &mut self,
        term: super::Term,
        docs: Vec<ExplicitDocument>,
    ) -> Result<super::SeqNo> {
        self.write_pending_try_changes()?;
        self.explicit_documents_check(&docs)?;
        self.add_explicit_with_delete(
            Some(super::DeleteNode::Terms(vec![std::sync::Arc::new(term)])),
            docs,
            None,
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
        self.write_pending_try_changes()?;
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
        self.add_explicit_with_delete(
            Some(super::DeleteNode::DocValuesUpdates(updates)),
            docs,
            None,
        )
    }

    /// Adds explicit documents with their KNN vectors (one list per
    /// document, aligned with `docs`) -- the document API's entry point, where
    /// a vector is one of the document's fields.
    pub(crate) fn add_explicit_documents_with_vectors(
        &mut self,
        delete: Option<super::Term>,
        docs: Vec<ExplicitDocument>,
        vectors: Vec<Vec<super::DocumentVector>>,
    ) -> Result<super::SeqNo> {
        self.write_pending_try_changes()?;
        let delete = delete.map(|term| super::DeleteNode::Terms(vec![std::sync::Arc::new(term)]));
        self.add_explicit_documents_with_node(delete, docs, vectors)
    }

    /// [`Self::add_explicit_documents_with_vectors`] deleting what `delete`
    /// matches: `updateDocuments(Node, docs)`, the shared tail of the term
    /// and query updates.
    pub(crate) fn add_explicit_documents_with_node(
        &mut self,
        delete: Option<super::DeleteNode>,
        docs: Vec<ExplicitDocument>,
        vectors: Vec<Vec<super::DocumentVector>>,
    ) -> Result<super::SeqNo> {
        self.write_pending_try_changes()?;
        // The document API builds these documents itself, from fields it has
        // just registered: they hold by construction what
        // `validate_explicit` checks of a caller's, so the check runs only
        // in debug builds, where the debug test run still exercises it (the
        // release run of `lucene-search`'s tests does not).
        if cfg!(debug_assertions) {
            self.explicit_documents_check(&docs)?;
        }
        if vectors.len() != docs.len() {
            return Err(explicit_error("one vector list per document"));
        }
        for doc_vectors in &vectors {
            self.cfg.validate_explicit_vectors(doc_vectors)?;
        }
        self.add_explicit_with_delete(delete, docs, Some(vectors))
    }

    /// [`Self::soft_update_explicit_documents`] for the document API, with
    /// each document's KNN vectors.
    pub(crate) fn soft_update_explicit_documents_with_vectors(
        &mut self,
        term: super::Term,
        docs: Vec<ExplicitDocument>,
        vectors: Vec<Vec<super::DocumentVector>>,
        soft_deletes: &[super::DocValuesUpdate],
    ) -> Result<super::SeqNo> {
        self.write_pending_try_changes()?;
        if soft_deletes.is_empty() {
            return Err(Error::NoSoftDeletesSupplied);
        }
        self.explicit_documents_check(&docs)?;
        if vectors.len() != docs.len() {
            return Err(explicit_error("one vector list per document"));
        }
        for doc_vectors in &vectors {
            self.cfg.validate_explicit_vectors(doc_vectors)?;
        }
        for update in soft_deletes {
            self.cfg.verify_doc_values_update_field(update)?;
        }
        let updates = soft_deletes
            .iter()
            .map(|u| super::retarget_update(u, &term))
            .collect();
        self.add_explicit_with_delete(
            Some(super::DeleteNode::DocValuesUpdates(updates)),
            docs,
            Some(vectors),
        )
    }

    fn add_explicit_with_delete(
        &mut self,
        delete: Option<super::DeleteNode>,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<super::DocumentVector>>>,
    ) -> Result<super::SeqNo> {
        // `DocumentsWriterPerThread.updateDocuments`: a block in a sorted
        // index needs a parent field to keep it together.
        self.cfg.check_block(docs.len())?;
        // Every fallible lookup runs before anything is buffered, so a
        // refused document cannot leave part of its block (or the block's
        // delete) behind.
        let options = docs
            .iter()
            .map(|doc| {
                doc.fields
                    .inverted
                    .iter()
                    .map(|inv| {
                        if inv.terms.is_empty() {
                            Ok(None)
                        } else {
                            self.cfg
                                .explicit_field(inv.field_number)
                                .map(|f| Some(f.index_options))
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        self.reserve_docs(docs.len())?;
        let mut docs = docs;
        // `IndexingChain.processDocument(docId, doc, lastDocInBlock)`: the
        // last document of the add is the parent, `-1` in the parent field.
        if let (Some(parent), Some(last)) = (self.register_parent_field()?, docs.last_mut()) {
            last.fields.doc_values.push(StoredField {
                field_number: parent,
                value: FieldValue::Long(-1),
            });
        }
        let doc_id_upto = self.pending_doc_id_upto();
        let seq_no = match delete {
            Some(node) => self.buffer_delete_node(node, doc_id_upto),
            None => self.delete_queue.next_sequence_number(),
        };
        if docs.len() > 1 {
            self.pending_has_blocks = true;
        }
        let mut vectors = vectors.map(Vec::into_iter);
        for (mut doc, options) in docs.into_iter().zip(options) {
            let mut extra = doc.fields.ram_bytes();
            self.buffer_document(Document { fields: doc.stored });
            // `IndexingChain.processDocument` inverts into the segment's
            // term hash as the document arrives: see `pending_postings`.
            let doc_id = i32::try_from(self.pending_docs.len().saturating_sub(1))
                .map_err(|_| explicit_error("too many buffered documents"))?;
            for (inv, options) in doc.fields.inverted.iter_mut().zip(options) {
                // the document API's are in the postings already
                let Some(options) = options else {
                    continue;
                };
                self.pending_postings.add_field(doc_id, inv, options)?;
                // the hash holds them now; the document keeps its norm
                inv.terms = InvertedTerms::default();
            }
            *self
                .pending_explicit
                .last_mut()
                .expect("buffer_document pushed an entry") = doc.fields;
            if let Some(doc_vectors) = vectors.as_mut().and_then(Iterator::next) {
                extra = doc_vectors
                    .iter()
                    .map(super::DocumentVector::ram_bytes)
                    .fold(extra, usize::saturating_add);
                *self
                    .pending_vectors
                    .last_mut()
                    .expect("buffer_document pushed an entry") = doc_vectors;
            }
            self.ram_bytes_used = self.ram_bytes_used.saturating_add(extra);
        }
        self.maybe_flush()?;
        Ok(seq_no)
    }
}

/// The field numbers a flushed segment carries: a flag per number, which
/// field numbers (dense from 0) make cheaper than an ordered set that every
/// field of every buffered document is inserted into.
#[derive(Default)]
struct DocFieldNumbers(Vec<bool>);

impl DocFieldNumbers {
    fn insert(&mut self, number: i32) {
        // field numbers are never negative
        if let Ok(i) = usize::try_from(number) {
            if i >= self.0.len() {
                self.0.resize(i.saturating_add(1), false);
            }
            self.0[i] = true;
        }
    }

    fn contains(&self, number: &i32) -> bool {
        usize::try_from(*number).is_ok_and(|i| self.0.get(i).copied().unwrap_or(false))
    }
}

/// The field numbers one document has used so far: a few kept inline (a
/// document rarely has more than a dozen fields of a role), the rest in a
/// tree, so checking a document allocates nothing in the usual case.
#[derive(Default)]
struct InlineFieldNumbers {
    inline: [i32; Self::INLINE],
    len: usize,
    spill: BTreeSet<i32>,
}

impl InlineFieldNumbers {
    const INLINE: usize = 16;

    fn contains(&self, n: i32) -> bool {
        self.inline.get(..self.len).is_some_and(|v| v.contains(&n)) || self.spill.contains(&n)
    }

    /// Adds `n`; false when it was there already.
    fn insert(&mut self, n: i32) -> bool {
        if self.contains(n) {
            return false;
        }
        match self.inline.get_mut(self.len) {
            Some(slot) => {
                *slot = n;
                self.len = self.len.saturating_add(1);
            }
            None => {
                self.spill.insert(n);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests;
