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

use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

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
            }],
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
        }],
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
mod tests {
    use super::*;
    use crate::monitor::config::{MonitorConfiguration, MonitorUpdateListener};
    use crate::monitor::facade::Monitor;
    use crate::monitor::index::MonitorQuery;
    use crate::monitor::presearcher::TermFilteredPresearcher;
    use crate::monitor::serializer::MonitorQuerySerializer;
    use crate::query::{Clause, TermQuery};
    use lucene_analysis::{Analyzer, StandardAnalyzer};
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn serializer() -> MonitorQuerySerializer {
        MonitorQuerySerializer::from_parser(Arc::new(|t: &str| {
            Ok(Clause::Term(TermQuery::new("f", t.as_bytes().to_vec())))
        }))
    }

    fn mq(id: &str, text: &str) -> MonitorQuery {
        let clause = Clause::Term(TermQuery::new("f", text.as_bytes().to_vec()));
        MonitorQuery::new(id, clause, Some(text.to_string()), BTreeMap::new()).unwrap()
    }

    fn open(path: &Path, read_only: bool) -> Result<Monitor> {
        let config = MonitorConfiguration::default().set_directory(path, serializer(), read_only);
        Monitor::with_configuration(
            Arc::new(Analyzer::new(StandardAnalyzer::new())),
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
    }

    #[test]
    fn queries_survive_a_reopen_and_deletes_and_replacements_persist() {
        let tmp = lucene_util::test_support::TempDir::new("monitor-persist");
        let path: &Path = &tmp;
        {
            let mut m = open(path, false).unwrap();
            m.register(vec![mq("a", "fox"), mq("b", "dog"), mq("c", "cat")])
                .unwrap();
            m.register(vec![mq("b", "wolf")]).unwrap();
            m.delete_by_id(&["c"]).unwrap();
        }
        let m = open(path, false).unwrap();
        assert_eq!(m.query_ids(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            m.get_query("b").unwrap().query_string.as_deref(),
            Some("wolf")
        );
        let hit = m.match_doc(&[("f", "the wolf")]).unwrap();
        assert_eq!(
            hit.matches
                .iter()
                .map(|x| x.query_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b"]
        );
        drop(m);
        let mut m = open(path, false).unwrap();
        m.clear().unwrap();
        drop(m);
        let m = open(path, false).unwrap();
        assert_eq!(m.query_count(), 0);
        assert_eq!(PersistentStore::open(path).unwrap().num_docs().unwrap(), 0);
    }

    #[test]
    fn a_query_without_text_and_an_index_without_a_serializer_are_refused() {
        let tmp = lucene_util::test_support::TempDir::new("monitor-persist-refuse");
        let path: &Path = &tmp;
        let mut m = open(path, false).unwrap();
        let no_text = MonitorQuery::new(
            "x",
            Clause::Term(TermQuery::new("f", b"x".to_vec())),
            None,
            BTreeMap::new(),
        )
        .unwrap();
        let e = m.register(vec![no_text]).unwrap_err().to_string();
        assert!(e.contains("null string representation"), "{e}");
        m.register(vec![mq("a", "fox")]).unwrap();
        drop(m);
        let config = MonitorConfiguration {
            index_path: Some(path.to_path_buf()),
            ..MonitorConfiguration::default()
        };
        let e = Monitor::with_configuration(
            Arc::new(Analyzer::new(StandardAnalyzer::new())),
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("no MonitorQuerySerializer"), "{e}");
        // an unparsable stored query is reported, not skipped
        let bad = MonitorQuerySerializer::from_parser(Arc::new(|_| Err("boom".to_string())));
        let config = MonitorConfiguration::default().set_directory(path, bad, true);
        let e = Monitor::with_configuration(
            Arc::new(Analyzer::new(StandardAnalyzer::new())),
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            e.contains("Couldn't parse some queries") && e.contains("boom"),
            "{e}"
        );
    }

    #[test]
    fn listeners_hear_updates_deletes_and_clears() {
        #[derive(Default)]
        struct Log(Mutex<Vec<String>>);
        impl MonitorUpdateListener for Log {
            fn after_update(&self, updates: &[MonitorQuery]) {
                self.0
                    .lock()
                    .unwrap()
                    .push(format!("update {}", updates.len()));
            }
            fn after_delete(&self, ids: &[String]) {
                self.0
                    .lock()
                    .unwrap()
                    .push(format!("delete {}", ids.join(",")));
            }
            fn after_clear(&self) {
                self.0.lock().unwrap().push("clear".into());
            }
        }
        let log = Arc::new(Log::default());
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        m.add_query_index_update_listener(log.clone());
        m.register(vec![mq("a", "fox"), mq("b", "dog")]).unwrap();
        m.delete_by_id(&["a"]).unwrap();
        m.clear().unwrap();
        assert_eq!(
            *log.0.lock().unwrap(),
            vec!["update 2", "delete a", "clear"]
        );
    }

    #[test]
    fn a_copy_of_lucenes_index_can_be_appended_to() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/monitor/index");
        let tmp = lucene_util::test_support::TempDir::new("monitor-persist-append");
        let path: &Path = &tmp;
        for f in std::fs::read_dir(&src).unwrap().flatten() {
            std::fs::copy(f.path(), path.join(f.file_name())).unwrap();
        }
        let parser = crate::queryparser::classic::QueryParser::new(
            "body",
            Arc::new(Analyzer::new(StandardAnalyzer::new())),
        );
        let ser = MonitorQuerySerializer::from_parser(Arc::new(move |t: &str| {
            parser
                .parse(t)
                .map(|p| p.to_clause(1))
                .map_err(|e| e.message)
        }));
        let config = MonitorConfiguration::default().set_directory(path, ser.clone(), false);
        let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
        let mut m = Monitor::with_configuration(
            Arc::clone(&analyzer),
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .unwrap();
        assert_eq!(m.query_count(), 40);
        let fox = ser
            .deserialize(&ser.serialize(&mq("new", "zebra")).unwrap())
            .unwrap();
        m.register(vec![fox]).unwrap();
        m.delete_by_id(&["q0"]).unwrap();
        drop(m);
        let m = Monitor::with_configuration(
            analyzer,
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .unwrap();
        assert_eq!(m.query_count(), 40);
        assert!(m.get_query("new").is_some() && m.get_query("q0").is_none());
        assert_eq!(m.match_doc(&[("body", "zebra")]).unwrap().match_count(), 1);
    }
}
