//! A source read straight into the documents the writer applies: [`prepare`] reads `_source`
//! against a [`Plan`] like [`super::parse`], decodes the fragments in place, and adds the
//! metadata fields the engine's document mapper would (`_id`, `_source`, `_nested_path`), so no
//! operation blob is built or decoded. The sequence number, primary term and version are only
//! known when the operation is applied: [`Prepared::finish`] adds them.

use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::index_writer::{ExplicitDocument, InvertedField, InvertedTerms};

use super::{read_document, Plan, Sections, SECTIONS, UNESCAPED};
use crate::status::Status;
use crate::writer_blob::document_from_sections;

/// The field numbers of the metadata fields, and the `_nested_path` term of each nested object.
pub(super) struct Meta {
    pub(super) id: i32,
    pub(super) seq_no: i32,
    pub(super) primary_term: i32,
    pub(super) version: i32,
    pub(super) source: i32,
    pub(super) nested_path: i32,
    /// Whether `_seq_no` is also a point (an index that keeps only the doc value drops it).
    pub(super) seq_point: bool,
    pub(super) nested_paths: Vec<Box<[u8]>>,
}

/// Documents read from one source, children before their parent, without their sequence
/// number, primary term and version.
pub struct Prepared {
    docs: Vec<ExplicitDocument>,
    /// The encoded `_id` term the documents belong to.
    id: Vec<u8>,
    /// Per nested child, in order, the plan index of its nested object.
    pub objects: Vec<i32>,
    /// A measure of the bytes the documents hold, for memory accounting.
    pub weight: usize,
}

impl Prepared {
    /// Number of documents (the nested children and the root).
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// The encoded `_id` term of the source read.
    pub fn id(&self) -> &[u8] {
        &self.id
    }

    /// The documents as the writer applies them, with the values the engine assigned.
    pub fn finish(
        mut self,
        plan: &Plan,
        seq_no: i64,
        primary_term: i64,
        version: i64,
    ) -> Vec<ExplicitDocument> {
        let Some(meta) = plan.meta.as_ref() else {
            return Vec::new();
        };
        let last = self.docs.len().saturating_sub(1);
        for (i, doc) in self.docs.iter_mut().enumerate() {
            doc.fields.doc_values.push(long(meta.seq_no, seq_no));
            if i == last {
                doc.fields
                    .doc_values
                    .push(long(meta.primary_term, primary_term));
            }
            doc.fields.doc_values.push(long(meta.version, version));
            if meta.seq_point {
                doc.fields.points.push(StoredField {
                    field_number: meta.seq_no,
                    value: FieldValue::Binary(long_point(seq_no)),
                });
            }
        }
        self.docs
    }
}

fn long(field_number: i32, value: i64) -> StoredField {
    StoredField {
        field_number,
        value: FieldValue::Long(value),
    }
}

/// `LongPoint`'s packed value: big-endian with the sign bit flipped.
fn long_point(value: i64) -> Vec<u8> {
    ((value as u64) ^ (1 << 63)).to_be_bytes().to_vec()
}

fn single_term(field_number: i32, term: &[u8]) -> InvertedField {
    let mut terms = InvertedTerms::new();
    terms.start(term).freq(1);
    InvertedField {
        field_number,
        terms,
        norm: None,
    }
}

/// One fragment's counts and sections: `4 x i32` counts, `4 x i32` lengths, the sections.
fn split(fragment: &[u8]) -> Option<([i32; 4], [&[u8]; 4])> {
    let int = |i: usize| -> Option<i32> {
        let b = fragment.get(i * 4..i * 4 + 4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let mut counts = [0; 4];
    let mut lens = [0usize; 4];
    for i in 0..4 {
        counts[i] = int(i)?;
        lens[i] = usize::try_from(int(4 + i)?).ok()?;
    }
    let mut at = 32usize;
    let mut sections: [&[u8]; 4] = [&[]; 4];
    for i in 0..4 {
        sections[i] = fragment.get(at..at.checked_add(lens[i])?)?;
        at += lens[i];
    }
    Some((counts, sections))
}

/// Reads `src` against `plan` into documents with the metadata fields of document `id` (its
/// encoded `_id` term); `None` when the JVM's mapper has to take the document.
pub fn prepare(plan: &Plan, src: &[u8], id: &[u8]) -> Result<Option<Prepared>, Status> {
    let Some(meta) = plan.meta.as_ref() else {
        return Ok(None);
    };
    SECTIONS.with(|s| {
        UNESCAPED.with(|u| {
            let mut sections = s.borrow_mut();
            let mut scratch = u.borrow_mut();
            let (text, path) = &mut *scratch;
            sections.clear();
            if read_document(plan, src, &mut sections, text, path).is_none() {
                return Ok(None);
            }
            let Sections { d, kids, nkids, .. } = &*sections;
            let mut docs = Vec::with_capacity(*nkids as usize + 1);
            let mut weight = src.len();
            let mut objects = Vec::with_capacity(*nkids as usize);
            let mut at = 0;
            for _ in 0..*nkids {
                let header = kids.get(at..at + 8).ok_or(Status::Decode)?;
                let object = i32::from_le_bytes([header[0], header[1], header[2], header[3]]);
                let len = i32::from_le_bytes([header[4], header[5], header[6], header[7]]);
                let len = usize::try_from(len).map_err(|_| Status::Decode)?;
                objects.push(object);
                let fragment = kids.get(at + 8..at + 8 + len).ok_or(Status::Decode)?;
                at += 8 + len;
                weight += len;
                let (counts, parts) = split(fragment).ok_or(Status::Decode)?;
                let mut doc = document_from_sections(counts, parts)?;
                let path = usize::try_from(object)
                    .ok()
                    .and_then(|o| meta.nested_paths.get(o))
                    .ok_or(Status::Decode)?;
                doc.fields.inverted.push(single_term(meta.id, id));
                doc.fields
                    .inverted
                    .push(single_term(meta.nested_path, path));
                docs.push(doc);
            }
            let counts = d.counts;
            let mut root =
                document_from_sections(counts, [&d.stored, &d.inverted, &d.doc_values, &d.points])?;
            weight += d.stored.len() + d.inverted.len() + d.doc_values.len() + d.points.len();
            root.stored.push(StoredField {
                field_number: meta.id,
                value: FieldValue::Binary(id.to_vec()),
            });
            root.fields.inverted.push(single_term(meta.id, id));
            root.stored.push(StoredField {
                field_number: meta.source,
                value: FieldValue::Binary(src.to_vec()),
            });
            docs.push(root);
            weight += id.len() * docs.len();
            Ok(Some(Prepared {
                docs,
                id: id.to_vec(),
                objects,
                weight,
            }))
        })
    })
}

#[cfg(test)]
pub(super) fn long_point_for_test(value: i64) -> Vec<u8> {
    long_point(value)
}
