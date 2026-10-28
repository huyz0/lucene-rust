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
            .flat_map(|f| &f.terms)
            .map(|t| {
                t.term
                    .capacity()
                    .saturating_add(t.positions.capacity().saturating_mul(4))
                    .saturating_add(t.offsets.capacity().saturating_mul(8))
                    .saturating_add(
                        t.payloads
                            .iter()
                            .map(|p| p.capacity().saturating_add(24))
                            .fold(
                                t.payloads.capacity().saturating_mul(24),
                                usize::saturating_add,
                            ),
                    )
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

impl IndexingConfig {
    /// The registered number of the parent field, once a document has
    /// registered it.
    pub(crate) fn parent_field_number(&self) -> Option<i32> {
        let name = self.parent_field.as_deref()?;
        self.field_by_name(name)
            .filter(|f| f.parent_field)
            .map(|f| f.number)
    }

    /// The registered vector field named `name`.
    fn explicit_vector_field(&self, name: &str) -> Result<&FieldInfo> {
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
        let mut seen = BTreeSet::new();
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
                && !seen.contains(&inv.field_number)
                && self.explicit_field(inv.field_number)?.store_term_vectors
            {
                return Err(explicit_error(format!(
                    "field {:?} stores term vectors but this document gives none",
                    self.explicit_field(inv.field_number)?.name
                )));
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
        let mut present = FieldNumberSet::default();
        for doc_vectors in buf.vectors {
            for v in doc_vectors {
                present.insert(self.explicit_vector_field(&v.field_name)?.number);
            }
        }
        for (doc, fields) in buf.docs.iter().zip(explicit) {
            doc.fields
                .iter()
                .for_each(|f| present.insert(f.field_number));
            let roles = [&fields.doc_values, &fields.points];
            fields
                .inverted
                .iter()
                .for_each(|f| present.insert(f.field_number));
            roles
                .into_iter()
                .flatten()
                .for_each(|f| present.insert(f.field_number));
            fields
                .term_vectors
                .iter()
                .for_each(|f| present.insert(f.field_number));
            fields.registered.iter().for_each(|&n| present.insert(n));
        }

        // Postings: one term dictionary per field, terms in byte order. The
        // writer inverted them as each document was buffered
        // (`pending_postings`); a buffer that still carries its terms is
        // inverted here the same way.
        let inverted_here;
        let pending = match buf.postings {
            Some(p) => p,
            None => {
                let mut p = PendingPostings::default();
                for (doc_id, fields) in explicit.iter().enumerate() {
                    let doc_id = i32::try_from(doc_id)
                        .map_err(|_| explicit_error("too many buffered documents"))?;
                    for inv in &fields.inverted {
                        let options = self.explicit_field(inv.field_number)?.index_options;
                        p.add_field(doc_id, inv, options)?;
                    }
                }
                inverted_here = p;
                &inverted_here
            }
        };
        // `FieldInfo.setStorePayloads`: the fields a document of this segment
        // gave a non-empty payload. Every occurrence of such a field then
        // carries a payload length, zero where it has none -- what
        // `FreqProxTermsWriterPerField.writeProx` records once the flag is on.
        let payload_fields: BTreeSet<i32> = pending.payload_fields().collect();
        let mut norm_values: BTreeMap<i32, Vec<Option<i64>>> = BTreeMap::new();
        for (doc_id, fields) in explicit.iter().enumerate() {
            for inv in &fields.inverted {
                if let Some(norm) = inv.norm {
                    norm_values
                        .entry(inv.field_number)
                        .or_insert_with(|| vec![None; max_doc])[doc_id] = Some(norm);
                }
            }
        }
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
        let remapped = buf.old_to_new.map(|map| pending.remap(map)).transpose()?;
        let term_lists: Vec<(i32, IndexOptions, i32, Vec<_>)> = pending
            .term_lists(remapped.as_ref())
            .into_iter()
            .map(|(n, docs, terms)| {
                let options = self.explicit_field(n).map(|f| f.index_options)?;
                Ok((n, options, docs, terms))
            })
            .collect::<Result<_>>()?;
        let postings_fields: BTreeSet<i32> = term_lists.iter().map(|(n, ..)| *n).collect();
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

        // Doc values and points through the native builders, each
        // document's values read in place.
        let in_place = |pick: fn(&ExplicitFields) -> &[StoredField]| -> Vec<&[StoredField]> {
            explicit.iter().map(pick).collect()
        };
        let dv_configs: Vec<DocValuesFieldConfig> = self
            .fields
            .iter()
            .filter(|f| present.contains(&f.number) && f.doc_values_type != DocValuesType::None)
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
            Some(self.build_doc_values_groups(
                &in_place(|f| &f.doc_values),
                &dv_configs,
                &names,
                &segment_id,
            )?)
        };
        let dv_suffix = |number: i32| {
            doc_values_output
                .as_deref()
                .and_then(|groups| groups.iter().find(|g| g.field_numbers.contains(&number)))
                .map(|g| (g.format_name, g.suffix))
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
                self,
                &in_place(|f| &f.points),
                &point_configs,
                &segment_id,
                dir,
                segment_name,
            )?
        };

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
            IndexWriter::build_vectors_output(
                buf.vectors,
                &vector_configs,
                i32::try_from(max_doc).map_err(|_| explicit_error("too many documents"))?,
                &|name: &str| self.knn_vectors_format_for(name),
                self.knn_vectors_direct(),
                &segment_id,
            )?
        };
        let postings_output = if term_lists.is_empty() {
            None
        } else {
            let inputs: Vec<FieldPostingsInput<'_, _>> = term_lists
                .iter()
                .map(|(n, options, docs, terms)| FieldPostingsInput {
                    field_number: *n,
                    index_options: *options,
                    doc_count: *docs,
                    has_payloads: payload_fields.contains(n),
                    terms,
                })
                .collect();
            let norms_for_impacts: Vec<postings_writer::FieldNorms<'_>> = norm_columns
                .iter()
                .map(|(n, column)| postings_writer::FieldNorms {
                    field_number: *n,
                    values: &column.dense,
                })
                .collect();
            Some(self.write_postings_groups(
                &inputs,
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
            )?)
        };

        // Term vectors: each document's fields in name order
        // (`TermVectorsConsumer.finishDocument` sorts its `perFields` by
        // name). Written when some document inverts a field that stores term
        // vectors, even one that produced no term: that is when
        // `TermVectorsConsumer.hasVectors` opens the writer.
        let has_vectors = explicit.iter().any(|f| {
            !f.term_vectors.is_empty()
                || f.inverted.iter().any(|inv| {
                    self.explicit_field(inv.field_number)
                        .is_ok_and(|info| info.store_term_vectors)
                })
        });
        let term_vectors_output = if !has_vectors {
            None
        } else {
            let docs = explicit
                .iter()
                .map(|f| {
                    let mut fields = f.term_vectors.clone();
                    let mut named = Vec::with_capacity(fields.len());
                    for tv in fields.drain(..) {
                        named.push((self.explicit_field(tv.field_number)?.name.as_str(), tv));
                    }
                    // `String.compareTo`: UTF-16 code units, not bytes.
                    named.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
                    Ok(TermVectorsDocument {
                        fields: named.into_iter().map(|(_, tv)| tv).collect(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
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
                    (postings_fields.contains(&f.number), postings_suffix)
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
            None,
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
        if self.use_compound_file {
            super::before_write(
                self.codec.as_deref(),
                lucene_codecs::codec::WritePoint::Compound,
            )?;
            flushed.pending_sync =
                segment_writer::create_compound_file(dir, segment_name, &mut flushed.info)?;
        }
        super::before_write(
            self.codec.as_deref(),
            lucene_codecs::codec::WritePoint::SegmentInfo,
        )?;
        segment_writer::seal_flushed_segment(dir, segment_name, flushed).map_err(Error::from)
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
        info.attributes.clear();
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
        self.explicit_documents_check(&docs)?;
        self.add_explicit_with_delete(None, docs, None)
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
                inv.terms = Vec::new();
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
struct FieldNumberSet(Vec<bool>);

impl FieldNumberSet {
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

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic, not values read off disk -- see
    // `docs/arithmetic-gate.md`'s "Test code" section.
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::buffered_updates::{DocValuesUpdate, Term};
    use crate::index_file_deleter::DeletionPolicy;
    use crate::index_writer::SoftDeletesRetention;
    use crate::index_writer::POSTINGS_FORMAT_NAME;
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
            payloads: Vec::new(),
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
            registered: Vec::new(),
            term_vectors: Vec::new(),
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
            payloads: Vec::new(),
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
    /// `DocumentsWriterPerThread.updateDocuments` validates each document of
    /// a block as it goes, and on a failure marks the block's earlier
    /// documents deleted; this port validates the whole block first, so a
    /// bad document `k` leaves nothing buffered and the block flag as it
    /// was -- the same documents survive either way.
    #[test]
    fn a_block_with_an_invalid_document_buffers_nothing() {
        let tmp = TempDir::new("explicit-bad-block");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
        let (docs, explicit, ram) = (
            w.pending_docs.len(),
            w.pending_explicit.len(),
            w.ram_bytes_used(),
        );
        for k in 0..3 {
            let mut block: Vec<ExplicitDocument> = (1..4).map(|i| doc(&f, i)).collect();
            // Document `k` names an unregistered field.
            block[k].stored.push(StoredField {
                field_number: 99,
                value: FieldValue::Long(1),
            });
            assert!(w.add_explicit_documents(block).is_err(), "doc {k}");
            assert_eq!(w.pending_docs.len(), docs);
            assert_eq!(w.pending_explicit.len(), explicit);
            assert_eq!(w.ram_bytes_used(), ram);
            assert!(!w.pending_has_blocks, "doc {k} set the block flag");
        }
        w.commit().unwrap();
        let infos = w.segment_infos().clone();
        assert_eq!(infos.segments.len(), 1);
        assert_eq!(infos.segments[0].del_count, 0);
    }

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
        assert!(
            w.register_field(vectors).is_err(),
            "vectors on an unindexed field"
        );
        let vectors = text.clone().with_store_term_vectors(true);
        assert!(
            w.register_field(vectors).is_err(),
            "storeTermVectors changed"
        );
        let mut payloads = FieldInfo::new("p", 0).with_index_options(IndexOptions::DocsAndFreqs);
        payloads.store_payloads = true;
        assert!(w.register_field(payloads).is_err());
    }

    #[test]
    fn reopened_writer_reserves_all_source_numbers_before_a_new_field_appears() {
        let dir = lucene_store::ByteBuffersDirectory::new();
        {
            let mut writer = IndexWriter::open(
                &dir,
                vec![
                    FieldInfo::new("old_a", 2)
                        .with_index_options(IndexOptions::Docs)
                        .with_omit_norms(true),
                    FieldInfo::new("old_b", 5),
                ],
                "Lucene104",
                VERSION,
            )
            .unwrap();
            writer.enable_explicit_documents().unwrap();
            writer
                .add_explicit_documents(vec![ExplicitDocument {
                    stored: vec![
                        StoredField {
                            field_number: 2,
                            value: FieldValue::String("a".into()),
                        },
                        StoredField {
                            field_number: 5,
                            value: FieldValue::String("b".into()),
                        },
                    ],
                    fields: ExplicitFields {
                        inverted: vec![InvertedField {
                            field_number: 2,
                            terms: vec![InvertedTerm {
                                term: b"drop".to_vec(),
                                freq: 1,
                                positions: Vec::new(),
                                offsets: Vec::new(),
                                payloads: Vec::new(),
                            }],
                            norm: None,
                        }],
                        ..ExplicitFields::default()
                    },
                }])
                .unwrap();
            writer.commit().unwrap();
        }
        let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        writer.enable_explicit_documents().unwrap();
        assert!(
            writer.cfg.fields.is_empty(),
            "reservations must not populate configured schemas"
        );
        assert_eq!(
            writer.register_field(FieldInfo::new("late", 99)).unwrap(),
            0
        );
        assert_eq!(
            writer
                .register_field(
                    FieldInfo::new("old_a", 99)
                        .with_index_options(IndexOptions::Docs)
                        .with_omit_norms(true)
                )
                .unwrap(),
            2
        );
        assert!(
            writer
                .register_field(FieldInfo::new("old_b", 99).with_index_options(IndexOptions::Docs))
                .is_err(),
            "source schemas are still checked"
        );
        // The source carrying old_b can disappear before this session has
        // registered it. Its reserved number survives for the writer's life.
        writer
            .delete_documents_by_term(&[Term::new("old_a", "drop")])
            .unwrap();
        writer.commit().unwrap();
        assert!(writer.segment_infos.segments.is_empty());
        assert_eq!(
            writer.register_field(FieldInfo::new("old_b", 99)).unwrap(),
            5
        );
        assert_eq!(
            writer
                .register_field(FieldInfo::new("another", 99))
                .unwrap(),
            1
        );
        assert_eq!(
            writer
                .register_field(FieldInfo::new("after_gap", 99))
                .unwrap(),
            3
        );
    }

    #[test]
    fn delete_all_rollback_restores_source_numbers_and_schema_before_new_fields() {
        for (fallible, configured) in [(false, false), (true, false), (false, true), (true, true)] {
            let dir = lucene_store::ByteBuffersDirectory::new();
            let original = FieldInfo::new("old", 0).with_index_options(IndexOptions::DocsAndFreqs);
            {
                let mut writer =
                    IndexWriter::open(&dir, vec![original.clone()], "Lucene104", VERSION).unwrap();
                writer.enable_explicit_documents().unwrap();
                writer
                    .add_explicit_documents(vec![ExplicitDocument {
                        fields: ExplicitFields {
                            inverted: vec![InvertedField {
                                field_number: 0,
                                terms: Vec::new(),
                                norm: Some(0),
                            }],
                            ..ExplicitFields::default()
                        },
                        ..ExplicitDocument::default()
                    }])
                    .unwrap();
                writer.commit().unwrap();
            }
            let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
            writer.enable_explicit_documents().unwrap();
            if configured {
                assert_eq!(
                    writer.register_field(FieldInfo::new("pinned", 99)).unwrap(),
                    1
                );
            }
            writer.delete_all().unwrap();
            if fallible {
                writer.try_rollback().unwrap();
            } else {
                writer.rollback();
            }
            assert_eq!(
                writer.register_field(FieldInfo::new("new", 99)).unwrap(),
                if configured { 2 } else { 1 }
            );
            if configured {
                assert_eq!(
                    writer.register_field(FieldInfo::new("pinned", 99)).unwrap(),
                    1
                );
            }
            assert!(writer.register_field(FieldInfo::new("old", 99)).is_err());
            assert!(writer
                .register_field(original.clone().with_omit_norms(true))
                .is_err());
            assert_eq!(writer.register_field(original).unwrap(), 0);
        }
    }

    #[test]
    fn rollback_schema_conflict_keeps_current_writer_state_coherent() {
        for fallible in [false, true] {
            let dir = lucene_store::ByteBuffersDirectory::new();
            {
                let mut writer = IndexWriter::open(
                    &dir,
                    vec![FieldInfo::new("field", 0).with_index_options(IndexOptions::DocsAndFreqs)],
                    "Lucene104",
                    VERSION,
                )
                .unwrap();
                writer.enable_explicit_documents().unwrap();
                writer
                    .add_explicit_documents(vec![ExplicitDocument {
                        fields: ExplicitFields {
                            inverted: vec![InvertedField {
                                field_number: 0,
                                terms: Vec::new(),
                                norm: Some(0),
                            }],
                            ..ExplicitFields::default()
                        },
                        ..ExplicitDocument::default()
                    }])
                    .unwrap();
                writer.commit().unwrap();
            }
            let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
            writer.enable_explicit_documents().unwrap();
            let original_files: Vec<_> = dir
                .list_all()
                .unwrap()
                .into_iter()
                .map(|name| {
                    let bytes = dir.open(&name).unwrap().to_vec();
                    (name, bytes)
                })
                .collect();
            writer.delete_all().unwrap();
            let changed = writer.register_field(FieldInfo::new("field", 99)).unwrap();
            assert_eq!(changed, 0);
            writer
                .add_explicit_documents(vec![ExplicitDocument {
                    stored: vec![StoredField {
                        field_number: changed,
                        value: FieldValue::String("replacement".into()),
                    }],
                    ..ExplicitDocument::default()
                }])
                .unwrap();
            if fallible {
                assert!(matches!(writer.try_rollback(), Err(Error::Explicit(_))));
            } else {
                writer.rollback();
            }
            assert!(
                writer.segment_infos.segments.is_empty(),
                "incompatible committed sources were not restored"
            );
            for (name, bytes) in &original_files {
                assert_eq!(
                    &*dir.open(name).unwrap(),
                    bytes.as_slice(),
                    "rejected rollback preserves committed source bytes"
                );
            }
            assert_eq!(
                writer.pending_docs.len(),
                1,
                "failed rollback leaves the current buffer intact"
            );
            assert_eq!(
                writer.register_field(FieldInfo::new("field", 99)).unwrap(),
                changed
            );
            let committed = writer.commit().unwrap().clone();
            assert_eq!(committed.segments.len(), 1);
            let fields = super::super::segment_field_infos(&dir, &committed.segments[0]).unwrap();
            assert_eq!(fields.fields[0].index_options, IndexOptions::None);
            for result in crate::check_index::check_directory(&dir).unwrap() {
                assert!(
                    result.all_passed(),
                    "{}: {:?}",
                    result.segment_name,
                    result.failures()
                );
            }
        }
    }

    #[test]
    fn reserved_schemas_keep_only_global_properties_and_reject_changes() {
        let mut numbers = FieldNumbers::default();
        let mut info = FieldInfo::new("field", 3).with_index_options(IndexOptions::DocsAndFreqs);
        info.store_payloads = true;
        info.doc_values_gen = 19;
        info.attributes.push(("segment".into(), "attribute".into()));
        assert_eq!(numbers.register(&info, Some(3)).unwrap(), 3);
        let retained = numbers.schema("field").unwrap();
        assert_eq!(retained.number, 3);
        assert!(!retained.store_payloads);
        assert_eq!(retained.doc_values_gen, -1);
        assert!(retained.attributes.is_empty());
        assert_eq!(numbers.register(&info, None).unwrap(), 3);
        assert!(numbers.register(&info.with_omit_norms(true), None).is_err());
    }

    #[test]
    fn source_number_conflicts_use_first_preference_then_lowest_free_number() {
        let mut numbers = FieldNumbers::default();
        assert_eq!(numbers.add_or_get("first", Some(2)).unwrap(), 2);
        assert_eq!(numbers.add_or_get("second", Some(2)).unwrap(), 0);
        // A repeated name keeps its first global assignment, regardless of
        // the number a later source segment assigned to the same name.
        assert_eq!(numbers.add_or_get("first", Some(1)).unwrap(), 2);
        assert_eq!(numbers.add_or_get("third", Some(0)).unwrap(), 1);
        assert_eq!(numbers.add_or_get("preferred", Some(7)).unwrap(), 7);
        assert_eq!(numbers.add_or_get("new", None).unwrap(), 3);
        assert_eq!(numbers.add_or_get("unset_source", Some(-1)).unwrap(), 4);
        // Saturating a field number must report exhaustion instead of
        // returning another field's already reserved number.
        numbers.lowest_unassigned = i32::MAX;
        assert_eq!(numbers.add_or_get("last", None).unwrap(), i32::MAX);
        assert!(numbers.add_or_get("overflow", None).is_err());
    }

    /// Term vectors arrive built; `validate_explicit` checks their shape
    /// against the registered schema.
    #[test]
    fn term_vectors_are_checked_at_add_time() {
        use lucene_codecs::term_vectors::TermVectorTerm;
        let dir = lucene_store::ByteBuffersDirectory::new();
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.enable_explicit_documents().unwrap();
        let indexed = |name: &str| {
            FieldInfo::new(name, 0)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_omit_norms(true)
        };
        let tv = w
            .register_field(indexed("tv").with_store_term_vectors(true))
            .unwrap();
        let plain = w.register_field(indexed("plain")).unwrap();
        let term = |t: &str| TermVectorTerm {
            term: t.as_bytes().to_vec(),
            freq: 1,
            positions: Some(vec![0]),
            start_offsets: None,
            end_offsets: None,
            payloads: None,
        };
        let field = |n: i32, terms: Vec<TermVectorTerm>| TermVectorField {
            field_number: n,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms,
        };
        let with = |fields: Vec<TermVectorField>| {
            let mut numbers: Vec<i32> = fields.iter().map(|f| f.field_number).collect();
            numbers.dedup();
            ExplicitDocument {
                stored: Vec::new(),
                fields: ExplicitFields {
                    inverted: numbers
                        .into_iter()
                        .map(|n| InvertedField {
                            field_number: n,
                            terms: vec![term_docs("a")],
                            norm: None,
                        })
                        .collect(),
                    term_vectors: fields,
                    ..ExplicitFields::default()
                },
            }
        };
        let mut not_inverted = with(vec![field(tv, vec![term("a")])]);
        not_inverted.fields.inverted.clear();
        let err = w.add_explicit_documents(vec![not_inverted]).unwrap_err();
        assert!(matches!(err, Error::Explicit(_)), "{err}");
        // Inverted into terms without the vector Java would have recorded.
        let mut no_vector = with(vec![field(tv, vec![term("a")])]);
        no_vector.fields.term_vectors.clear();
        let err = w.add_explicit_documents(vec![no_vector]).unwrap_err();
        assert!(err.to_string().contains("gives none"), "{err}");
        w.add_explicit_documents(vec![with(vec![field(tv, vec![term("a")])])])
            .unwrap();
        let mut payloads_alone = field(tv, vec![term("a")]);
        payloads_alone.has_positions = false;
        payloads_alone.has_payloads = true;
        let mut no_positions = term("a");
        no_positions.positions = None;
        let mut zero = term("a");
        zero.freq = 0;
        let cases = [
            ("unregistered", vec![field(99, vec![term("a")])]),
            ("no term vectors", vec![field(plain, vec![term("a")])]),
            (
                "twice",
                vec![field(tv, vec![term("a")]), field(tv, vec![term("b")])],
            ),
            ("payloads without positions", vec![payloads_alone]),
            ("unsorted", vec![field(tv, vec![term("b"), term("a")])]),
            ("duplicate", vec![field(tv, vec![term("a"), term("a")])]),
            ("missing positions", vec![field(tv, vec![no_positions])]),
            ("freq 0", vec![field(tv, vec![zero])]),
        ];
        for (what, fields) in cases {
            let err = w
                .add_explicit_documents(vec![with(fields)])
                .expect_err(what);
            assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
        }
        w.commit().unwrap();
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{:?}", r.failures());
        }
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
        assert_eq!(w.register_parent_field().unwrap(), None);
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
        d.fields.inverted[1].terms[0].positions.pop();
        cases.push(("positions out of step with freq", d));
        let mut d = base();
        d.fields.inverted[1].terms[0].offsets.pop();
        cases.push(("offsets out of step with freq", d));
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
        let mut d = base();
        d.fields.inverted[1].terms[0].payloads = vec![vec![1]; 7];
        cases.push(("payloads out of step with positions", d));
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

    /// Every refusal `validate_explicit_vectors` makes, and the vector-list
    /// count check, on both the add and the soft-update entry points.
    #[test]
    fn invalid_vectors_are_refused_at_add_time() {
        use crate::index_writer::DocumentVector;
        use lucene_codecs::field_infos::{VectorEncoding, VectorSimilarityFunction};
        let tmp = TempDir::new("explicit-vectors-invalid");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.register_field(
            FieldInfo::new("v", 0)
                .with_index_options(IndexOptions::None)
                .with_vectors(2, VectorEncoding::Float32, VectorSimilarityFunction::Cosine),
        )
        .unwrap();
        let ok = || DocumentVector::float32("v", vec![1.0, 2.0]);
        let cases: Vec<(&str, Vec<DocumentVector>)> = vec![
            (
                "unregistered vector field",
                vec![DocumentVector::float32("nope", vec![1.0, 2.0])],
            ),
            (
                "registered but not a vector",
                vec![DocumentVector::float32("_id", vec![1.0, 2.0])],
            ),
            (
                "encoding mismatch",
                vec![DocumentVector::byte("v", vec![1, 2])],
            ),
            (
                "dimension mismatch",
                vec![DocumentVector::float32("v", vec![1.0])],
            ),
            ("repeated field", vec![ok(), ok()]),
        ];
        for (what, vectors) in cases {
            let err = w.cfg.validate_explicit_vectors(&vectors).expect_err(what);
            assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
            let err = w
                .add_explicit_documents_with_vectors(None, vec![doc(&f, 1)], vec![vectors])
                .expect_err(what);
            assert!(matches!(err, Error::Explicit(_)), "{what}: {err}");
        }
        w.cfg.validate_explicit_vectors(&[ok()]).unwrap();
        let err = w
            .add_explicit_documents_with_vectors(None, vec![doc(&f, 1)], Vec::new())
            .expect_err("vector list count");
        assert!(matches!(err, Error::Explicit(_)), "{err}");
        let err = w
            .soft_update_explicit_documents_with_vectors(
                Term {
                    field: "_id".to_string(),
                    bytes: b"d1".to_vec(),
                },
                vec![doc(&f, 1)],
                vec![vec![DocumentVector::byte("v", vec![1, 2])]],
                &[soft_delete(1)],
            )
            .expect_err("soft update encoding");
        assert!(matches!(err, Error::Explicit(_)), "{err}");
    }

    #[test]
    fn stored_only_adds_mark_their_parent() {
        // Java's `addDocument`/`addDocuments` mark the last document of
        // every add once a parent field is configured; the stored-only
        // entry points used to buffer it unmarked, so a block had no
        // parent and `MockRandomMergePolicy.reverse` saw none.
        let tmp = TempDir::new("explicit-stored-only-parent");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_parent_field(Some("_parent")).unwrap();
        let empty = || Document { fields: Vec::new() };
        w.add_documents(vec![empty(), empty()]).unwrap();
        w.add_document(empty()).unwrap();
        w.add_document_with_custom_freq_terms(empty(), Vec::new())
            .unwrap();
        w.add_document_with_vectors(empty(), Vec::new()).unwrap();
        w.add_documents(Vec::new()).unwrap();
        let parent = w.cfg.parent_field_number().unwrap();
        let marked: Vec<bool> = w
            .pending_explicit
            .iter()
            .map(|f| f.doc_values.iter().any(|v| v.field_number == parent))
            .collect();
        assert_eq!(marked, [false, true, true, true, true]);
        w.commit().unwrap();
        let sci = w.segment_infos.segments[0].clone();
        let infos = super::super::segment_field_infos(&dir, &sci).unwrap();
        assert_eq!(infos.parent_field(), Some("_parent"));
    }

    /// The refusals `register_field` and `set_parent_field` make: schema
    /// shapes Lucene rejects, and the parent field's rules.
    #[test]
    fn field_registration_and_parent_field_refusals() {
        use crate::segment_info::IndexSortField;
        let explicit = |what: &str, r: Result<()>| {
            assert!(matches!(r, Err(Error::Explicit(_))), "{what}: {r:?}");
        };
        let tmp = TempDir::new("explicit-field-refusals");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        // Before explicit mode: both refused or switched on.
        explicit(
            "register before enabling",
            w.register_field(FieldInfo::new("a", 0)).map(drop),
        );
        w.enable_explicit_documents().unwrap();
        explicit(
            "term vectors",
            w.register_field(FieldInfo::new("a", 0).with_store_term_vectors(true))
                .map(drop),
        );
        explicit(
            "skip index on binary doc values",
            w.register_field(FieldInfo::new("a", 0).with_doc_values(
                DocValuesType::Binary,
                DocValuesSkipIndexType::Range,
                -1,
            ))
            .map(drop),
        );
        explicit(
            "parent field without a configured parent",
            w.register_field(FieldInfo::new("a", 0).with_parent_field(true))
                .map(drop),
        );
        // A conflicting schema for a known name.
        w.register_field(FieldInfo::new("c", 0).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap();
        explicit(
            "changed doc values type",
            w.register_field(FieldInfo::new("c", 0).with_doc_values(
                DocValuesType::Sorted,
                DocValuesSkipIndexType::None,
                -1,
            ))
            .map(drop),
        );
        // set_parent_field against what is registered.
        explicit(
            "parent named as a registered non-parent field",
            w.set_parent_field(Some("c")),
        );
        w.register_field(
            FieldInfo::new("soft", 0)
                .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1)
                .with_soft_deletes_field(true),
        )
        .unwrap();
        explicit(
            "parent named as the soft-deletes field",
            w.set_parent_field(Some("soft")),
        );
        explicit(
            "parent added to an index that has other fields",
            w.set_parent_field(Some("p")),
        );

        // A fresh writer: the parent field proper.
        let tmp2 = TempDir::new("explicit-parent-field");
        let dir2 = FsDirectory::open(tmp2.path());
        let mut w = IndexWriter::open(&dir2, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_parent_field(Some("p")).unwrap();
        assert_eq!(w.parent_field(), Some("p"));
        explicit(
            "another field as the parent",
            w.register_field(FieldInfo::new("q", 0).with_parent_field(true))
                .map(drop),
        );
        explicit(
            "the parent name as an ordinary field",
            w.register_field(FieldInfo::new("p", 0)).map(drop),
        );
        w.register_parent_field().unwrap();
        explicit(
            "another parent over a registered one",
            w.set_parent_field(Some("r")),
        );
        explicit("clearing a registered parent", w.set_parent_field(None));
        let f = register(&mut w);
        w.add_explicit_documents(vec![doc(&f, 1)]).unwrap();
        explicit(
            "setting the parent with documents buffered",
            w.set_parent_field(Some("p")),
        );

        // Clearing a parent that was never registered.
        let tmp3 = TempDir::new("explicit-parent-clear");
        let dir3 = FsDirectory::open(tmp3.path());
        let mut w = IndexWriter::open(&dir3, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_parent_field(Some("p")).unwrap();
        w.set_parent_field(None).unwrap();
        assert_eq!(w.parent_field(), None);

        // An index sort names a field; registering it with other doc values
        // is refused.
        let tmp4 = TempDir::new("explicit-sort-mismatch");
        let dir4 = FsDirectory::open(tmp4.path());
        let mut w = IndexWriter::open(&dir4, Vec::new(), "Lucene104", VERSION).unwrap();
        w.enable_explicit_documents().unwrap();
        w.set_index_sort(Some(&[IndexSortField::long("v", true, None)]))
            .unwrap();
        explicit(
            "sort field with the wrong doc values",
            w.register_field(FieldInfo::new("v", 0).with_doc_values(
                DocValuesType::Sorted,
                DocValuesSkipIndexType::None,
                -1,
            ))
            .map(drop),
        );
        let err = w
            .set_index_sort(Some(&[IndexSortField::long("v", true, None)]))
            .map(drop);
        assert!(err.is_ok(), "{err:?}");
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
            prune_postings_field: None,
            prune_recovery_source_field: None,
        }));
        w.force_merge_deletes().unwrap();
        assert_eq!(counts(&mut w), (13, 3));
        check(&dir);

        // Nothing retained: all history goes, and a clean index is left alone.
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: i64::MAX,
            prune_postings_field: None,
            prune_recovery_source_field: None,
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

    /// The `_id` document frequency of each of `terms` in the index's only
    /// segment, or `None` when that segment writes no `_id` terms at all.
    fn id_doc_freqs(
        dir: &FsDirectory,
        w: &mut IndexWriter<'_>,
        terms: &[&str],
    ) -> Option<Vec<i32>> {
        let infos = w.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        let seg = crate::index_writer::per_field_segment(&sci.segment_name, POSTINGS_FORMAT_NAME);
        let suffix = crate::index_writer::per_field_codec_suffix(POSTINGS_FORMAT_NAME);
        let open = |ext: &str| dir.open(&format!("{seg}.{ext}")).unwrap();
        let (tim, tip, tmd) = (open("tim"), open("tip"), open("tmd"));
        let field_infos = lucene_codecs::field_infos::FieldInfos {
            fields: segment_fields(dir, sci),
        };
        let tree = lucene_codecs::blocktree::open(
            &tim,
            &tip,
            &tmd,
            &field_infos,
            &sci.segment_id,
            &suffix,
            i32::try_from(w.committed_doc_count().unwrap()).unwrap(),
        )
        .unwrap();
        let field = tree.field("_id")?;
        Some(
            terms
                .iter()
                .map(|t| field.seek_exact(t.as_bytes()).map_or(0, |s| s.doc_freq))
                .collect(),
        )
    }

    /// `PrunePostingsMergePolicy`: a retention merge keeps a soft-deleted
    /// document for its history but drops its `_id` postings, and writes no
    /// `_id` terms at all once every document is soft-deleted.
    #[test]
    fn retention_merges_prune_the_ids_of_soft_deleted_documents() {
        let tmp = TempDir::new("explicit-prune");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        for i in 0..3 {
            w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        }
        w.commit().unwrap();
        w.soft_update_explicit_documents(id_term(0), vec![doc_at(&f, 0, 100)], &[soft_delete(0)])
            .unwrap();
        let retention = |prune: bool| SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: 0,
            prune_postings_field: prune.then(|| "_id".to_string()),
            prune_recovery_source_field: None,
        };

        // Without pruning the retained old version still answers `d0`.
        w.set_soft_deletes_retention(Some(retention(false)));
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (4, 1));
        assert_eq!(id_doc_freqs(&dir, &mut w, &["d0", "d1"]), Some(vec![2, 1]));

        // With pruning, the merge keeps both old versions but not their ids.
        w.soft_update_explicit_documents(id_term(1), vec![doc_at(&f, 1, 101)], &[soft_delete(1)])
            .unwrap();
        w.set_soft_deletes_retention(Some(retention(true)));
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (5, 2), "the history is kept");
        assert_eq!(
            id_doc_freqs(&dir, &mut w, &["d0", "d1", "d2"]),
            Some(vec![1, 1, 1]),
            "only the live versions keep their ids"
        );
        check(&dir);

        // Every document soft-deleted: no `_id` term survives.
        let tmp = TempDir::new("explicit-prune-all");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.add_explicit_documents(vec![doc(&f, 0)]).unwrap();
        w.commit().unwrap();
        let mut tombstone = doc_at(&f, 0, 1);
        tombstone.fields.doc_values.push(StoredField {
            field_number: f.soft,
            value: FieldValue::Long(1),
        });
        w.soft_update_explicit_documents(id_term(0), vec![tombstone], &[soft_delete(0)])
            .unwrap();
        w.set_soft_deletes_retention(Some(retention(true)));
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (2, 2));
        assert_eq!(id_doc_freqs(&dir, &mut w, &["d0"]), None);
        check(&dir);
    }

    /// `MergePolicy.numDeletesToMerge`: `force_merge(1)` leaves a single
    /// segment whose soft-deleted documents are all still retained alone,
    /// and rewrites it once retention lets them go.
    #[test]
    fn force_merge_rewrites_one_segment_only_for_reclaimable_soft_deletes() {
        let tmp = TempDir::new("explicit-reclaimable");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        for i in 0..3 {
            w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        }
        w.commit().unwrap();
        w.soft_update_explicit_documents(id_term(0), vec![doc_at(&f, 0, 100)], &[soft_delete(0)])
            .unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (4, 1));
        let retention = |min: i64| SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: min,
            prune_postings_field: None,
            prune_recovery_source_field: None,
        };

        w.set_soft_deletes_retention(Some(retention(0)));
        let before = w.commit_generations();
        w.force_merge(1).unwrap();
        assert_eq!(w.commit_generations(), before, "all history is retained");
        w.force_merge_deletes().unwrap();
        assert_eq!(w.commit_generations(), before, "nothing to expunge either");

        w.set_soft_deletes_retention(Some(retention(1)));
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (3, 0), "seq_no 0 is reclaimed");
        check(&dir);
    }

    /// `RecoverySourcePruneMergePolicy`: a retention merge keeps the stored
    /// `_recovery_source` and its doc-values flag only on documents whose
    /// sequence number is still retained.
    #[test]
    fn retention_merges_prune_the_recovery_source_of_unretained_documents() {
        let tmp = TempDir::new("explicit-recovery-source");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let recovery = w
            .register_field(FieldInfo::new("_recovery_source", 0).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ))
            .unwrap();
        let with_recovery = |mut d: ExplicitDocument| {
            d.stored.push(StoredField {
                field_number: recovery,
                value: FieldValue::Binary(b"{}".to_vec()),
            });
            d.fields.doc_values.push(StoredField {
                field_number: recovery,
                value: FieldValue::Long(1),
            });
            d
        };
        for i in 0..3 {
            w.add_explicit_documents(vec![with_recovery(doc(&f, i))])
                .unwrap();
        }
        w.commit().unwrap();
        w.soft_update_explicit_documents(
            id_term(0),
            vec![with_recovery(doc_at(&f, 0, 100))],
            &[soft_delete(0)],
        )
        .unwrap();
        w.commit().unwrap();
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: 2,
            prune_postings_field: None,
            prune_recovery_source_field: Some("_recovery_source".to_string()),
        }));
        w.force_merge(1).unwrap();
        let infos = w.commit().unwrap().clone();
        check(&dir);

        // d0's old version (seq_no 0) is gone; d1 (1) is live but no longer
        // retained; d2 (2) and d0's new version (100) are retained.
        let sci = &infos.segments[0];
        assert_eq!((infos.segments.len(), sci.soft_del_count), (1, 0));
        let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
        let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
        let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
        let stored =
            lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let si_files: Vec<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|n| {
                n.starts_with(&format!("{}.", sci.segment_name))
                    || n.starts_with(&format!("{}_", sci.segment_name))
            })
            .collect();
        let fields = crate::field_updates::read_current_field_infos(&dir, sci, &si_files).unwrap();
        let number = |name: &str| fields.fields.iter().position(|f| f.name == name).unwrap();
        let (id_at, recovery_at) = (number("_id"), number("_recovery_source"));
        let mut kept = Vec::new();
        for doc in 0..stored.max_doc() {
            let d = stored.document(doc).unwrap();
            let id = d
                .fields
                .iter()
                .find_map(|v| match &v.value {
                    FieldValue::String(s) if v.field_number == fields.fields[id_at].number => {
                        Some(s.clone())
                    }
                    _ => None,
                })
                .unwrap();
            if d.fields
                .iter()
                .any(|v| v.field_number == fields.fields[recovery_at].number)
            {
                kept.push(id);
            }
        }
        kept.sort();
        assert_eq!(kept, ["d0", "d2"], "stored");

        let per_field = crate::field_updates::per_field_component(
            &fields.fields[recovery_at],
            &crate::index_writer::per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        );
        let (meta, data) = crate::field_updates::read_current_column(
            &dir,
            sci,
            &si_files,
            &fields,
            recovery_at,
            &per_field,
        )
        .unwrap()
        .unwrap();
        let entry = meta
            .numeric_entry(fields.fields[recovery_at].number)
            .unwrap();
        let mut flags = lucene_codecs::doc_values::NumericReader::new(&data, entry);
        let mut flagged: Vec<String> = (0..stored.max_doc())
            .filter(|&d| flags.value(d).unwrap().is_some())
            .map(|d| {
                stored
                    .document(d)
                    .unwrap()
                    .fields
                    .into_iter()
                    .find_map(|v| match v.value {
                        FieldValue::String(s) if v.field_number == fields.fields[id_at].number => {
                            Some(s)
                        }
                        _ => None,
                    })
                    .unwrap()
            })
            .collect();
        flagged.sort();
        assert_eq!(flagged, ["d0", "d2"], "doc values");
    }

    /// `IndexWriter.reserveDocs`: an add past the limit is refused before
    /// anything -- its own delete included -- is buffered, deleted documents
    /// count until a merge drops them, and the limit is clamped to
    /// `MAX_DOCS`.
    #[test]
    fn adds_past_max_docs_are_refused_whole() {
        let tmp = TempDir::new("explicit-max-docs");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        w.set_max_docs(3);
        w.add_explicit_documents(vec![doc(&f, 0), doc(&f, 1)])
            .unwrap();
        w.commit().unwrap();
        w.add_explicit_documents(vec![doc(&f, 2)]).unwrap();
        let too_many = |r: Result<super::super::SeqNo>| {
            assert!(matches!(r, Err(Error::TooManyDocs(3))), "{r:?}");
        };
        too_many(w.add_explicit_documents(vec![doc(&f, 3)]));
        too_many(w.soft_update_explicit_documents(
            id_term(0),
            vec![doc_at(&f, 0, 100)],
            &[soft_delete(0)],
        ));
        assert_eq!(counts(&mut w), (3, 0), "the refused update deleted nothing");

        // A hard delete frees its slot only once a merge drops the document.
        w.update_explicit_documents(id_term(9), Vec::new()).unwrap();
        w.soft_update_explicit_documents(id_term(0), Vec::new(), &[soft_delete(0)])
            .unwrap();
        w.commit().unwrap();
        too_many(w.add_explicit_documents(vec![doc(&f, 3)]));
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: i64::MAX,
            prune_postings_field: None,
            prune_recovery_source_field: None,
        }));
        w.force_merge(1).unwrap();
        w.add_explicit_documents(vec![doc(&f, 3)]).unwrap();
        assert_eq!(counts(&mut w), (3, 0));

        w.set_max_docs(usize::MAX);
        assert_eq!(w.max_docs, crate::index_writer::MAX_DOCS);
        check(&dir);
    }

    /// `mergeSortedField`/`mergeSortedSetField`: a term only deleted
    /// documents used leaves the merged dictionary, for SORTED as for
    /// SORTED_SET -- Java's `CheckIndex` refuses a dictionary with holes.
    #[test]
    fn merges_drop_terms_only_deleted_documents_used() {
        let tmp = TempDir::new("explicit-dead-terms");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let kw = w
            .register_field(FieldInfo::new("kw", 0).with_doc_values(
                DocValuesType::Sorted,
                DocValuesSkipIndexType::None,
                -1,
            ))
            .unwrap();
        for i in 0..4 {
            let mut d = doc(&f, i);
            d.fields.doc_values.push(StoredField {
                field_number: kw,
                value: FieldValue::Binary(format!("k{i}").into_bytes()),
            });
            w.add_explicit_documents(vec![d]).unwrap();
        }
        w.commit().unwrap();
        // d0 alone carries `k0` and `tag` t0 (d3 has t0 too); d1 alone `k1`.
        w.update_explicit_documents(id_term(0), Vec::new()).unwrap();
        w.update_explicit_documents(id_term(1), Vec::new()).unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(counts(&mut w), (2, 0));
        check(&dir);
    }

    /// Doc-values skip indexes are written at flush, at merge and in a
    /// doc-values update's new generation.
    #[test]
    fn skip_indexed_fields_are_written_everywhere_a_column_is() {
        let tmp = TempDir::new("explicit-skip-index");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let rank = w
            .register_field(FieldInfo::new("rank", 0).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::Range,
                -1,
            ))
            .unwrap();
        // `verifySameDocValuesSkipIndex`, and no skip index without numbers
        // or ordinals to summarize.
        for bad in [
            FieldInfo::new("rank", 0).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
            FieldInfo::new("blob", 0).with_doc_values(
                DocValuesType::Binary,
                DocValuesSkipIndexType::Range,
                -1,
            ),
        ] {
            assert!(matches!(w.register_field(bad), Err(Error::Explicit(_))));
        }
        let ranked = |i: i64| {
            let mut d = doc(&f, i);
            d.fields.doc_values.push(StoredField {
                field_number: rank,
                value: FieldValue::Long(10 + i),
            });
            d
        };
        let skipper = |w: &mut IndexWriter<'_>| {
            let infos = w.commit().unwrap().clone();
            let sci = &infos.segments[0];
            let files: Vec<String> = dir
                .list_all()
                .unwrap()
                .into_iter()
                .filter(|n| {
                    n.starts_with(&format!("{}.", sci.segment_name))
                        || n.starts_with(&format!("{}_", sci.segment_name))
                })
                .collect();
            let fields = crate::field_updates::read_current_field_infos(&dir, sci, &files).unwrap();
            let at = fields.fields.iter().position(|f| f.name == "rank").unwrap();
            let per_field = crate::field_updates::per_field_component(
                &fields.fields[at],
                &crate::index_writer::per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
            );
            let (meta, _) = crate::field_updates::read_current_column(
                &dir, sci, &files, &fields, at, &per_field,
            )
            .unwrap()
            .unwrap();
            let s = *meta
                .skipper_meta(fields.fields[at].number)
                .expect("a skip index");
            (s.min_value, s.max_value, s.doc_count)
        };
        w.add_explicit_documents(vec![ranked(0), ranked(1)])
            .unwrap();
        assert_eq!(skipper(&mut w), (10, 11, 2), "flush");
        w.add_explicit_documents(vec![ranked(2)]).unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
        assert_eq!(skipper(&mut w), (10, 12, 3), "merge");
        let update = DocValuesUpdate::Numeric {
            term: id_term(1),
            field: "rank".to_string(),
            value: Some(99),
        };
        w.soft_update_explicit_documents(id_term(1), Vec::new(), &[update])
            .unwrap();
        assert_eq!(skipper(&mut w), (10, 99, 3), "update generation");
        check(&dir);
    }

    /// `findForcedDeletesMerges`: a segment is expunged only when more than
    /// `forceMergeDeletesPctAllowed` (10%) of it can be reclaimed.
    #[test]
    fn force_merge_deletes_leaves_segments_under_the_threshold() {
        let tmp = TempDir::new("explicit-expunge-pct");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        let docs: Vec<ExplicitDocument> = (0..20).map(|i| doc(&f, i)).collect();
        w.add_explicit_documents(docs).unwrap();
        w.commit().unwrap();
        w.set_soft_deletes_retention(Some(SoftDeletesRetention {
            seq_no_field: "_seq_no".to_string(),
            min_retained_seq_no: i64::MAX,
            prune_postings_field: None,
            prune_recovery_source_field: None,
        }));
        // 2 of 20 is exactly 10%: not above it.
        for i in 0..2 {
            w.soft_update_explicit_documents(id_term(i), Vec::new(), &[soft_delete(i)])
                .unwrap();
        }
        w.commit().unwrap();
        let before = w.commit_generations();
        w.force_merge_deletes().unwrap();
        assert_eq!(w.commit_generations(), before);
        w.soft_update_explicit_documents(id_term(2), Vec::new(), &[soft_delete(2)])
            .unwrap();
        w.commit().unwrap();
        w.force_merge_deletes().unwrap();
        assert_eq!(counts(&mut w), (17, 0), "3 of 20 is above it");
        check(&dir);
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
            prune_postings_field: None,
            prune_recovery_source_field: None,
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
                registered: Vec::new(),
                term_vectors: Vec::new(),
            },
        };
        w.add_explicit_documents(vec![one("a")]).unwrap();
        let soft = DocValuesUpdate::Numeric {
            term: id_term(0),
            field: "__soft_deletes".to_string(),
            value: Some(1),
        };
        assert!(matches!(
            w.soft_update_explicit_documents(id_term(0), vec![one("a")], std::slice::from_ref(&soft)),
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

    /// Document `i` of [`doc`] whose `body` term "beta" (positions 1 and 3)
    /// carries a payload at its first occurrence and none at its second --
    /// what a payload-producing analyzer such as Lucene's `MockAnalyzer`
    /// hands `PayloadAttribute`.
    fn doc_with_payload(f: &Fields, i: i64) -> ExplicitDocument {
        let mut d = doc(f, i);
        let body = d
            .fields
            .inverted
            .iter_mut()
            .find(|inv| inv.field_number == f.body)
            .unwrap();
        let beta = body.terms.iter_mut().find(|t| t.term == b"beta").unwrap();
        beta.payloads = vec![format!("p{i}").into_bytes(), Vec::new()];
        d
    }

    /// The payloads of `body:beta`, occurrence by occurrence, in every
    /// document of `sci` that has it.
    fn beta_payloads(dir: &FsDirectory, sci: &SegmentCommitInfo, max_doc: i32) -> Vec<Vec<u8>> {
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let infos = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        crate::index_writer::tests::read_occurrences(dir, sci, &infos, "body", b"beta", max_doc)
            .into_iter()
            .flat_map(|occurrences| occurrences.into_iter().map(|o| o.payload))
            .collect()
    }

    fn stores_payloads(dir: &FsDirectory, sci: &SegmentCommitInfo, name: &str) -> bool {
        segment_fields(dir, sci)
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .store_payloads
    }

    /// A document whose analyzer produced payloads is indexed with them. The
    /// OpenSearch engine used to refuse it, and OpenSearch then records a
    /// refused primary operation as a no-op tombstone -- so on a seed whose
    /// `MockAnalyzer` put payloads on the `value` field, the engine tests
    /// lost almost every document (`testAppendConcurrently`: `expected:<1024>
    /// but was:<6>`, the six whose random payloads happened to be empty). The
    /// field stores payloads in exactly the segments one of whose documents
    /// gave it one, as `FieldInfo.setStorePayloads` does, the bytes read back
    /// through the postings, and a merge keeps them.
    #[test]
    fn explicit_payloads_flush_merge_and_read_back() {
        let tmp = TempDir::new("explicit-payloads");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);
        // Documents 0 and 2 carry payloads, 1 does not.
        for i in 0..3 {
            let d = if i % 2 == 0 {
                doc_with_payload(&f, i)
            } else {
                doc(&f, i)
            };
            w.add_explicit_documents(vec![d]).unwrap();
        }
        w.commit().unwrap();
        // A segment none of whose documents has a payload.
        for i in 3..5 {
            w.add_explicit_documents(vec![doc(&f, i)]).unwrap();
        }
        let infos = w.commit().unwrap().clone();
        assert_eq!(infos.segments.len(), 2);
        check(&dir);

        let (with, without) = (&infos.segments[0], &infos.segments[1]);
        assert!(stores_payloads(&dir, with, "body"));
        assert!(
            !stores_payloads(&dir, with, "_id"),
            "only the field with one"
        );
        assert!(!stores_payloads(&dir, without, "body"));
        let none = Vec::new();
        assert_eq!(
            beta_payloads(&dir, with, 3),
            [
                b"p0".to_vec(),
                none.clone(),
                none.clone(),
                none.clone(),
                b"p2".to_vec(),
                none.clone()
            ]
        );

        w.force_merge(1).unwrap();
        assert_eq!(w.segment_infos().segments.len(), 1);
        let merged = w.segment_infos().segments[0].clone();
        check(&dir);
        assert!(stores_payloads(&dir, &merged, "body"));
        let mut payloads = beta_payloads(&dir, &merged, 5);
        payloads.retain(|p| !p.is_empty());
        payloads.sort();
        assert_eq!(payloads, [b"p0".to_vec(), b"p2".to_vec()]);
    }

    /// Payloads are one per occurrence, and only on a field with positions --
    /// a malformed document is refused whole, before it is buffered.
    #[test]
    fn explicit_payloads_are_validated() {
        let tmp = TempDir::new("explicit-payloads-invalid");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        let f = register(&mut w);

        let mut short = doc_with_payload(&f, 0);
        for inv in &mut short.fields.inverted {
            for t in &mut inv.terms {
                if t.term == b"beta" {
                    t.payloads.pop();
                }
            }
        }
        let err = w
            .add_explicit_documents(vec![short])
            .unwrap_err()
            .to_string();
        assert!(err.contains("one per occurrence"), "{err}");

        let mut no_positions = doc(&f, 1);
        for inv in &mut no_positions.fields.inverted {
            if inv.field_number == f.id {
                inv.terms[0].payloads = vec![b"x".to_vec()];
            }
        }
        let err = w
            .add_explicit_documents(vec![no_positions])
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot store payloads"), "{err}");
        assert_eq!(w.pending_doc_count(), 0);
    }

    #[test]
    fn a_doc_values_update_creates_an_unknown_field_doc_values_only() {
        // Java's `updateDocValues`/`tryUpdateDocValue` build their update
        // with `verifyOrCreateDvOnlyField(name, type, false)`: a field no
        // document has carried is created, not indexed, with the update's
        // type. `updateNumericDocValue`/`updateBinaryDocValue` pass
        // `fieldMustExist` and still refuse it.
        use crate::document::{Document as Doc, Store, StringField};
        let tmp = TempDir::new("explicit-dv-only-update");
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.enable_explicit_documents().unwrap();
        let mut d = Doc::new();
        d.add(StringField::new("id", "0", Store::No));
        w.add_fields_document(&d).unwrap();
        w.commit().unwrap();
        let reader_infos = w.segment_infos().clone();
        let id = || Term::new("id", "0");
        let err = w.update_numeric_doc_value(id(), "n", 1).unwrap_err();
        assert!(
            matches!(err, Error::UnknownDocValuesUpdateField(_)),
            "{err}"
        );
        let err = w.update_binary_doc_value(id(), "b", b"x").unwrap_err();
        assert!(
            matches!(err, Error::UnknownDocValuesUpdateField(_)),
            "{err}"
        );
        let update = |field: &str| DocValuesUpdate::Numeric {
            term: id(),
            field: field.to_string(),
            value: Some(7),
        };
        w.update_doc_values(id(), &[update("n")]).unwrap();
        assert!(w
            .try_update_doc_value(&reader_infos, 0, &[update("m")])
            .unwrap()
            .is_some());
        w.commit().unwrap();
        let sci = w.segment_infos.segments[0].clone();
        let infos = super::super::segment_field_infos(&dir, &sci).unwrap();
        for name in ["n", "m"] {
            let info = infos.field_by_name(name).unwrap();
            assert_eq!(info.doc_values_type, DocValuesType::Numeric, "{name}");
            assert_eq!(info.index_options, IndexOptions::None, "{name}");
        }
    }
}
