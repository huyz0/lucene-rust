//! The on-disk query index of `WritableQueryIndex` / `ReadonlyQueryIndex`:
//! a Lucene index with one document per decomposed query piece.
//!
//! A document is the presearcher's document (each indexed field's terms, with
//! positions and norms: `TermFilteredPresearcher.indexQuery`), plus
//! `_query_id` (a `StringField` and a sorted doc value), `_cache_id` (sorted
//! doc value) and `_mq` (a binary doc value holding the serialized
//! `MonitorQuery`). Reading the index back needs only `_query_id` and `_mq`:
//! the queries are rebuilt from their serialized form, as Java's
//! `populateQueryCache` does, and decomposed afresh.
//!
//! The writer is opened for each commit: Lucene keeps one open and holds its
//! write lock for the monitor's life; here the directory is free between
//! commits, which only loosens the lock.

use std::path::{Path, PathBuf};

use lucene_codecs::doc_values::{binary_value, sorted_ord};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_codecs::terms_dict::decode_all_terms;
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::FsDirectory;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{Error, Result};

/// `QueryIndex.FIELDS.query_id`.
pub const QUERY_ID: &str = "_query_id";
/// `QueryIndex.FIELDS.cache_id`.
pub const CACHE_ID: &str = "_cache_id";
/// `QueryIndex.FIELDS.mq`.
pub const MQ: &str = "_mq";

/// One document of the query index: a decomposed piece of a query.
#[derive(Debug, Clone)]
pub struct EntryDoc {
    pub query_id: String,
    pub cache_id: String,
    /// The serialized `MonitorQuery` (`_mq`).
    pub mq: Vec<u8>,
    /// The presearcher's fields: each with its terms, sorted.
    pub fields: Vec<(String, Vec<Vec<u8>>)>,
    /// The presearcher's metadata fields (a filter field's value).
    pub metadata: Vec<(String, String)>,
}

/// The directory a monitor persists its queries in.
#[derive(Debug, Clone)]
pub struct PersistentStore {
    path: PathBuf,
}

fn index_error(e: lucene_index::index_writer::Error) -> Error {
    Error::IllegalArgument(e.to_string())
}

impl PersistentStore {
    /// Opens (creating the directory if needed) the index at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(path.as_ref())
            .map_err(|e| Error::IllegalState(format!("cannot create {:?}: {e}", path.as_ref())))?;
        Ok(PersistentStore {
            path: path.as_ref().to_path_buf(),
        })
    }

    fn has_commit(&self) -> bool {
        std::fs::read_dir(&self.path)
            .map(|d| {
                d.flatten()
                    .any(|f| f.file_name().to_string_lossy().starts_with("segments_"))
            })
            .unwrap_or(false)
    }

    /// `ReadonlyQueryIndex` / `populateQueryCache`: the serialized form of
    /// each registered query, once per query id, in document order.
    pub fn load(&self) -> Result<Vec<(String, Vec<u8>)>> {
        if !self.has_commit() {
            return Ok(Vec::new());
        }
        let dir = FsDirectory::open(&self.path);
        let reader = DirectoryReader::open(&dir)?;
        let mut out: Vec<(String, Vec<u8>)> = Vec::new();
        for seg in reader.segment_readers() {
            let infos = seg.field_infos();
            let (Some(qid), Some(mq)) = (infos.field_by_name(QUERY_ID), infos.field_by_name(MQ))
            else {
                continue;
            };
            let Some((meta, data)) = seg.doc_values_for_field(qid.number) else {
                continue;
            };
            let (Some(q_entry), Some((mq_meta, mq_data))) = (
                meta.sorted_entry(qid.number),
                seg.doc_values_for_field(mq.number),
            ) else {
                continue;
            };
            let Some(mq_entry) = mq_meta.binary_entry(mq.number) else {
                continue;
            };
            let ids = decode_all_terms(data, &q_entry.terms)
                .map_err(|e| Error::IllegalState(e.to_string()))?;
            for doc in 0..seg.max_doc {
                if seg.live_docs().is_some_and(|l| !l.get_doc(doc)) {
                    continue;
                }
                let ord = sorted_ord(data, q_entry, doc)?;
                let bytes = binary_value(mq_data, mq_entry, doc)?;
                let (Some(ord), Some(bytes)) = (ord, bytes) else {
                    continue;
                };
                let id =
                    String::from_utf8_lossy(&ids[usize::try_from(ord).unwrap_or(0)]).into_owned();
                // a query split by decomposition has several documents
                if !out.iter().any(|(i, _)| *i == id) {
                    out.push((id, bytes.to_vec()));
                }
            }
        }
        Ok(out)
    }

    /// Deletes every document of `delete_ids`, adds `docs`, and commits:
    /// `WritableQueryIndex.commitWithoutNotify`.
    pub fn commit(&self, delete_ids: &[String], docs: &[EntryDoc]) -> Result<()> {
        let dir = FsDirectory::open(&self.path);
        let mut writer =
            IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
        writer.set_use_compound_file(false);
        writer.enable_explicit_documents().map_err(index_error)?;
        if !delete_ids.is_empty() {
            let terms: Vec<Term> = delete_ids
                .iter()
                .map(|id| Term {
                    field: QUERY_ID.to_string(),
                    bytes: id.as_bytes().to_vec(),
                })
                .collect();
            writer
                .delete_documents_by_term(&terms)
                .map_err(index_error)?;
        }
        let mut explicit = Vec::with_capacity(docs.len());
        for doc in docs {
            explicit.push(build_document(&mut writer, doc)?);
        }
        if !explicit.is_empty() {
            writer
                .add_explicit_documents(explicit)
                .map_err(index_error)?;
        }
        writer.commit().map_err(index_error)?;
        Ok(())
    }

    /// `WritableQueryIndex.clear`: every document deleted.
    pub fn clear(&self) -> Result<()> {
        if !self.has_commit() {
            return Ok(());
        }
        let dir = FsDirectory::open(&self.path);
        let mut writer =
            IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
        writer.delete_all().map_err(index_error)?;
        writer.commit().map_err(index_error)?;
        Ok(())
    }

    /// `numDocs()`: the documents in the index.
    pub fn num_docs(&self) -> Result<usize> {
        if !self.has_commit() {
            return Ok(0);
        }
        let dir = FsDirectory::open(&self.path);
        let reader = DirectoryReader::open(&dir)?;
        Ok(usize::try_from(reader.num_docs()).unwrap_or(0))
    }
}

/// A field of the monitor index, registered with the writer.
fn register(
    writer: &mut IndexWriter<'_>,
    name: &str,
    index_options: IndexOptions,
    omit_norms: bool,
    doc_values_type: DocValuesType,
) -> Result<i32> {
    let mut info = FieldInfo::new(name, 0);
    info.index_options = index_options;
    info.omit_norms = omit_norms;
    info.doc_values_type = doc_values_type;
    writer.register_field(info).map_err(index_error)
}

/// One entry as the writer's explicit document.
fn build_document(writer: &mut IndexWriter<'_>, doc: &EntryDoc) -> Result<ExplicitDocument> {
    let mut fields = ExplicitFields::default();
    // the presearcher's fields: `TextField`-like, positions and norms
    for (name, terms) in &doc.fields {
        let number = register(
            writer,
            name,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            DocValuesType::None,
        )?;
        let length = i32::try_from(terms.len()).unwrap_or(i32::MAX);
        fields.inverted.push(InvertedField {
            field_number: number,
            terms: terms
                .iter()
                .enumerate()
                .map(|(i, t)| InvertedTerm {
                    term: t.clone(),
                    freq: 1,
                    positions: vec![i32::try_from(i).unwrap_or(i32::MAX)],
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                })
                .collect(),
            norm: Some(length_norm(length)),
        });
    }
    let mut stored = Vec::new();
    // a filter field's value: a stored `TextField`, one keyword token
    for (name, value) in &doc.metadata {
        let number = register(
            writer,
            name,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            DocValuesType::None,
        )?;
        fields.inverted.push(InvertedField {
            field_number: number,
            terms: vec![InvertedTerm {
                term: value.as_bytes().to_vec(),
                freq: 1,
                positions: vec![0],
                offsets: Vec::new(),
                payloads: Vec::new(),
            }]
            .into(),
            norm: Some(length_norm(1)),
        });
        stored.push(StoredField {
            field_number: number,
            value: FieldValue::String(value.clone()),
        });
    }
    // `_query_id`: a `StringField` and a sorted doc value
    let query_id = register(
        writer,
        QUERY_ID,
        IndexOptions::Docs,
        true,
        DocValuesType::Sorted,
    )?;
    fields.inverted.push(InvertedField {
        field_number: query_id,
        terms: vec![InvertedTerm {
            term: doc.query_id.as_bytes().to_vec(),
            freq: 1,
            positions: Vec::new(),
            offsets: Vec::new(),
            payloads: Vec::new(),
        }]
        .into(),
        norm: None,
    });
    let cache_id = register(
        writer,
        CACHE_ID,
        IndexOptions::None,
        true,
        DocValuesType::Sorted,
    )?;
    let mq = register(writer, MQ, IndexOptions::None, true, DocValuesType::Binary)?;
    fields.doc_values.push(StoredField {
        field_number: cache_id,
        value: FieldValue::Binary(doc.cache_id.as_bytes().to_vec()),
    });
    fields.doc_values.push(StoredField {
        field_number: query_id,
        value: FieldValue::Binary(doc.query_id.as_bytes().to_vec()),
    });
    fields.doc_values.push(StoredField {
        field_number: mq,
        value: FieldValue::Binary(doc.mq.clone()),
    });
    Ok(ExplicitDocument { stored, fields })
}

/// `BM25Similarity.computeNorm` for a field of `length` distinct tokens, none
/// at an increment of zero.
fn length_norm(length: i32) -> i64 {
    let state = lucene_index::similarity::FieldInvertState {
        docs_only: false,
        position: length,
        length,
        num_overlap: 0,
        offset: 0,
        max_term_frequency: 1,
        unique_term_count: length,
        attribute_source: None,
    };
    lucene_index::similarity::default_compute_norm(true, &state)
}

#[cfg(test)]
mod tests;
