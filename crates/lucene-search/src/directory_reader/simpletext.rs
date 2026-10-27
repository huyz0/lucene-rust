//! `SegmentCoreReaders` and `SegmentReader`'s SimpleText format dispatch.
//!
//! The `.si` keeps its complete version and typed sort. Compound entries
//! come from `.scf`; field-info and doc-values updates stay in the outer
//! directory, as in Java. The component decoders remain in `lucene-codecs`.
//! Differs: compound entries are copied into independently owned inputs,
//! as the binary compound reader does, rather than cloned `IndexInput`s.
//! Stored-field/term-vector offsets and doc-values headers are cached
//! separately from their live bytes to avoid self-referential Rust values.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::segment_state::SegmentReadState;
use lucene_codecs::simpletext as text;
use text::doc_values_reader::SimpleTextDocValuesReader;
use text::fields_reader::SimpleTextFieldsReader;
use text::stored_fields_reader::SimpleTextStoredFieldsReader;
use text::term_vectors_reader::SimpleTextTermVectorsReader;

use super::*;

/// Files shared by every reader over one SimpleText segment core.
#[derive(Debug)]
pub(crate) struct SimpleTextCore {
    pub(crate) info: segment_info::SimpleTextSegmentInfo,
    pub(crate) field_infos: Arc<FieldInfos>,
    files: BTreeMap<String, Arc<Input>>,
    pub(crate) postings: Option<SimpleTextFieldsReader>,
    stored: SimpleTextStoredFieldsReader<'static>,
    term_vectors: Option<SimpleTextTermVectorsReader<'static>>,
    doc_values: BTreeMap<String, SimpleTextDocValuesReader<'static>>,
    points: BTreeMap<String, text::bkd_reader::SimpleTextBkdReader<'static>>,
    vector_entries: std::collections::HashMap<i32, text::knn_vectors_reader::FieldEntry>,
    /// The binary forms searches read, built on first use.
    pub(crate) search: super::simpletext_search::SearchForms,
}

impl SimpleTextCore {
    // Java: SegmentCoreReaders.<init>, SegmentReader.initFieldInfos,
    // SegmentDocValuesProducer.<init>.
    fn open(dir: &dyn Directory, commit: &segment_infos::SegmentCommitInfo) -> Result<Self> {
        let name = &commit.segment_name;
        let info = segment_info::parse_simpletext(
            &dir.open(&format!("{name}.si"))?,
            name,
            &commit.segment_id,
        )?;
        let mut files = BTreeMap::new();
        if info.is_compound_file {
            let data = dir.open(&text::compound::compound_file_name(name))?;
            let compound = text::compound::SimpleTextCompoundReader::open(&data, name)?;
            for entry in compound.list_all() {
                files.insert(
                    entry.clone(),
                    Arc::new(Input::Owned(compound.open_input(&data, entry)?.to_vec())),
                );
            }
        } else {
            for file in &info.files {
                if file != &format!("{name}.si") {
                    files.insert(file.clone(), Arc::new(dir.open(file)?));
                }
            }
        }
        let base_name = text::field_infos::field_infos_file_name(name, "");
        let base_bytes = files.get(&base_name).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("segment {name} has no {base_name}"))
        })?;
        let base_infos = text::field_infos::read(base_bytes)?;
        let field_infos = if commit.field_infos_gen == -1 {
            base_infos.clone()
        } else {
            let suffix = lucene_util::base36::to_base36(commit.field_infos_gen);
            let file = text::field_infos::field_infos_file_name(name, &suffix);
            let bytes = Arc::new(dir.open(&file)?);
            let infos = text::field_infos::read(&bytes)?;
            files.insert(file, bytes);
            infos
        };
        for (_, updates) in &commit.dv_update_files {
            for file in updates {
                if !files.contains_key(file) {
                    files.insert(file.clone(), Arc::new(dir.open(file)?));
                }
            }
        }
        let postings = if base_infos
            .fields
            .iter()
            .any(|field| field.index_options != lucene_codecs::field_infos::IndexOptions::None)
        {
            let file = text::fields_writer::postings_file_name(name, "");
            let bytes = files.get(&file).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {file}"))
            })?;
            Some(SimpleTextFieldsReader::open(
                bytes.to_vec(),
                &SegmentReadState {
                    max_doc: info.doc_count,
                    segment_id: &commit.segment_id,
                    segment_suffix: "",
                    field_infos: &base_infos,
                },
            )?)
        } else {
            None
        };
        let stored_name = text::stored_fields_writer::stored_fields_file_name(name);
        let stored_data = files.get(&stored_name).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("segment {name} has no {stored_name}"))
        })?;
        let stored = SimpleTextStoredFieldsReader::open(stored_data, info.doc_count)?.rebind(&[]);
        let term_vectors = if base_infos
            .fields
            .iter()
            .any(|field| field.store_term_vectors)
        {
            let file = text::term_vectors_writer::term_vectors_file_name(name);
            let data = files.get(&file).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {file}"))
            })?;
            Some(SimpleTextTermVectorsReader::open(data, info.doc_count)?.rebind(&[]))
        } else {
            None
        };
        let mut doc_values = BTreeMap::new();
        for field in &field_infos.fields {
            if field.doc_values_type != DocValuesType::None {
                let file = dv_file_name(name, field.doc_values_gen);
                if let std::collections::btree_map::Entry::Vacant(entry) = doc_values.entry(file) {
                    let data = files.get(entry.key()).ok_or_else(|| {
                        lucene_store::Error::Corrupted(format!(
                            "segment {name} has no {}",
                            entry.key()
                        ))
                    })?;
                    entry
                        .insert(SimpleTextDocValuesReader::open(data, info.doc_count)?.rebind(&[]));
                }
            }
        }
        if base_infos.fields.iter().any(|field| {
            !field.omit_norms
                && field.index_options != lucene_codecs::field_infos::IndexOptions::None
        }) {
            let file = text::norms::norms_file_name(name, "");
            let data = files.get(&file).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {file}"))
            })?;
            doc_values.insert(
                file,
                SimpleTextDocValuesReader::open(data, info.doc_count)?.rebind(&[]),
            );
        }
        let mut points = BTreeMap::new();
        if base_infos
            .fields
            .iter()
            .any(|field| field.point_dimension_count != 0)
        {
            let dim_name = text::points_writer::points_file_name(name, "");
            let dii_name = text::points_writer::points_index_file_name(name, "");
            let data = files.get(&dim_name).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {dim_name}"))
            })?;
            let index = files.get(&dii_name).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {dii_name}"))
            })?;
            let reader = text::points_reader::SimpleTextPointsReader::open(data, index)?;
            for field in reader.field_names() {
                if let Some(tree) = reader.field(field) {
                    points.insert(field.to_owned(), tree.rebind(&[]));
                }
            }
        }
        let vector_entries = if base_infos
            .fields
            .iter()
            .any(|field| field.vector_dimension != 0)
        {
            let data_name = text::knn_vectors_writer::vector_data_file_name(name, "");
            let meta_name = text::knn_vectors_writer::meta_file_name(name, "");
            let data = files.get(&data_name).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {data_name}"))
            })?;
            let meta = files.get(&meta_name).ok_or_else(|| {
                lucene_store::Error::Corrupted(format!("segment {name} has no {meta_name}"))
            })?;
            text::knn_vectors_reader::SimpleTextKnnVectorsReader::open(meta, data, &base_infos)?
                .into_field_entries()
        } else {
            std::collections::HashMap::new()
        };
        Ok(Self {
            info,
            field_infos: Arc::new(field_infos),
            files,
            postings,
            stored,
            term_vectors,
            doc_values,
            points,
            vector_entries,
            search: Default::default(),
        })
    }

    pub(crate) fn check_integrity(&self) -> crate::Result<()> {
        // Java: each component's checkIntegrity. Every component uses the
        // same text CRC footer; the .scf table was validated when opened.
        for data in self.files.values() {
            if data.starts_with(text::util::CHECKSUM) {
                // SimpleText's empty vectors file consists only of its CRC line.
                let mut position = 0;
                text::util::check_footer(data, 0, &mut position).map_err(Error::SimpleText)?;
            } else {
                text::util::check_integrity(data).map_err(Error::SimpleText)?;
            }
        }
        Ok(())
    }

    fn file(&self, name: &str) -> Result<&[u8]> {
        self.files.get(name).map(|data| &***data).ok_or_else(|| {
            lucene_store::Error::Corrupted(format!("segment {} has no {name}", self.info.name))
                .into()
        })
    }

    pub(crate) fn visit_stored_document(
        &self,
        doc: i32,
        visitor: &mut dyn lucene_codecs::stored_fields::StoredFieldVisitor,
    ) -> Result<bool> {
        let name = text::stored_fields_writer::stored_fields_file_name(&self.info.name);
        let data = self.file(&name)?;
        self.stored.rebind(data).visit_document(doc, visitor)?;
        Ok(true)
    }

    pub(crate) fn doc_values(
        &self,
        field: &str,
        ty: DocValuesType,
    ) -> Result<Option<(&SimpleTextDocValuesReader<'static>, &[u8])>> {
        let Some(info) = self.field_infos.field_by_name(field) else {
            return Ok(None);
        };
        if info.doc_values_type != ty {
            return Ok(None);
        }
        self.values_file(&dv_file_name(&self.info.name, info.doc_values_gen))
            .map(Some)
    }

    pub(crate) fn term_vectors(
        &self,
        doc: i32,
    ) -> Result<Option<lucene_codecs::term_vectors::TermVectorsDocument>> {
        let Some(reader) = &self.term_vectors else {
            return Ok(None);
        };
        let name = text::term_vectors_writer::term_vectors_file_name(&self.info.name);
        let data = self.file(&name)?;
        Ok(reader.rebind(data).get(doc)?.map(|fields| {
            lucene_codecs::term_vectors::TermVectorsDocument {
                fields: fields.into_iter().map(|(_, field)| field).collect(),
            }
        }))
    }

    pub(crate) fn norms(
        &self,
        field: &str,
    ) -> Result<Option<(&SimpleTextDocValuesReader<'static>, &[u8])>> {
        let Some(info) = self.field_infos.field_by_name(field) else {
            return Ok(None);
        };
        if info.omit_norms || info.index_options == lucene_codecs::field_infos::IndexOptions::None {
            return Ok(None);
        }
        self.values_file(&text::norms::norms_file_name(&self.info.name, ""))
            .map(Some)
    }

    pub(crate) fn points(
        &self,
        field: &str,
    ) -> Result<Option<text::bkd_reader::SimpleTextBkdReader<'_>>> {
        let Some(info) = self.field_infos.field_by_name(field) else {
            return Ok(None);
        };
        if info.point_dimension_count == 0 {
            return Ok(None);
        }
        let Some(tree) = self.points.get(field) else {
            return Ok(None);
        };
        let name = text::points_writer::points_file_name(&self.info.name, "");
        let data = self.file(&name)?;
        Ok(Some(tree.rebind(data)))
    }

    pub(crate) fn vector_values<T: text::knn_vectors_reader::VectorElement>(
        &self,
        field: &str,
    ) -> Result<Option<text::knn_vectors_reader::SimpleTextVectorValues<'_, T>>> {
        let Some(info) = self.field_infos.field_by_name(field) else {
            return Ok(None);
        };
        let Some(entry) = self.vector_entries.get(&info.number) else {
            return Ok(None);
        };
        if entry.dimension != info.vector_dimension {
            return Err(lucene_store::Error::Corrupted(format!(
                "Inconsistent vector dimension for field=\"{field}\"; {} != {}",
                info.vector_dimension, entry.dimension
            ))
            .into());
        }
        let name = text::knn_vectors_writer::vector_data_file_name(&self.info.name, "");
        let data = self.file(&name)?;
        Ok(Some(entry.vector_values(data)?))
    }

    fn values_file(&self, name: &str) -> Result<(&SimpleTextDocValuesReader<'static>, &[u8])> {
        match (self.doc_values.get(name), self.files.get(name)) {
            (Some(reader), Some(data)) => Ok((reader, data)),
            _ => Err(lucene_store::Error::Corrupted(format!(
                "segment {} has no doc-values reader for {name}",
                self.info.name
            ))
            .into()),
        }
    }

    pub(crate) fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i32> {
        let Some(fields) = &self.postings else {
            return Ok(0);
        };
        let Some(terms) = fields.terms(field)? else {
            return Ok(0);
        };
        let mut cursor = terms.iterator();
        Ok(if cursor.seek_exact(term)? {
            cursor.doc_freq()
        } else {
            0
        })
    }

    pub(crate) fn field_stats(&self, field: &str) -> Result<Option<(i64, i32)>> {
        let Some(fields) = &self.postings else {
            return Ok(None);
        };
        Ok(fields
            .terms(field)?
            .map(|terms| (terms.sum_total_term_freq(), terms.doc_count())))
    }
}

fn dv_file_name(segment: &str, generation: i64) -> String {
    let suffix = if generation == -1 {
        String::new()
    } else {
        lucene_util::base36::to_base36(generation)
    };
    text::doc_values_writer::doc_values_file_name(
        segment,
        &suffix,
        text::doc_values_writer::DOC_VALUES_EXTENSION,
    )
}

/// Java: `SegmentReader`'s constructor and `LiveDocsFormat.readLiveDocs`.
pub(super) fn read_live_docs(
    dir: &dyn Directory,
    commit: &segment_infos::SegmentCommitInfo,
    max_doc: i32,
) -> Result<Option<Arc<FixedBitSet>>> {
    let Some(name) = text::live_docs::live_docs_file_name(&commit.segment_name, commit.del_gen)
    else {
        return Ok(None);
    };
    let bits = text::live_docs::read(&dir.open(&name)?, max_doc)?;
    if usize::try_from(max_doc).ok() != Some(bits.len())
        || i64::try_from(bits.cardinality()).unwrap_or(i64::MAX)
            != i64::from(max_doc).saturating_sub(i64::from(commit.del_count))
    {
        return Err(lucene_store::Error::Corrupted(format!(
            "segment {}: live-doc size/count disagrees with maxDoc={max_doc}, delCount={}",
            commit.segment_name, commit.del_count
        ))
        .into());
    }
    Ok(Some(Arc::new(bits)))
}

impl SegmentReader {
    /// Java: `FieldExistsQuery.createWeight().scorerSupplier`'s source
    /// selection. Keep the existing scorer's owned document-set contract;
    /// the source iterators retain their normal fallible decode behavior.
    pub(super) fn simpletext_field_exists_docs(
        &self,
        info: &lucene_codecs::field_infos::FieldInfo,
        source: crate::doc_value_query::FieldExistsSource,
    ) -> crate::Result<ExistsDocs> {
        use crate::doc_value_query::FieldExistsSource;
        use crate::reader::{DocIdSetIterator, LeafReader, NO_MORE_DOCS};
        let field = &info.name;
        if source == FieldExistsSource::Vectors {
            return match info.vector_encoding {
                lucene_codecs::field_infos::VectorEncoding::Float32 => {
                    let Some(values) = self.float_vector_values(field)? else {
                        return Ok(ExistsDocs::None);
                    };
                    exists_docs_from_iter(
                        self.max_doc,
                        (0..values.size()).map(|ord| values.ord_to_doc(ord)),
                    )
                }
                lucene_codecs::field_infos::VectorEncoding::Byte => {
                    let Some(values) = self.byte_vector_values(field)? else {
                        return Ok(ExistsDocs::None);
                    };
                    exists_docs_from_iter(
                        self.max_doc,
                        (0..values.size()).map(|ord| values.ord_to_doc(ord)),
                    )
                }
            };
        }
        let values: Option<Box<dyn DocIdSetIterator + '_>> = match source {
            FieldExistsSource::Norms => self
                .norm_values(field)?
                .map(|v| v as Box<dyn DocIdSetIterator>),
            FieldExistsSource::DocValues => match info.doc_values_type {
                DocValuesType::Numeric => self
                    .numeric_doc_values(field)?
                    .map(|v| v as Box<dyn DocIdSetIterator>),
                DocValuesType::Binary => self
                    .binary_doc_values(field)?
                    .map(|v| v as Box<dyn DocIdSetIterator>),
                DocValuesType::Sorted => self
                    .sorted_doc_values(field)?
                    .map(|v| v as Box<dyn DocIdSetIterator>),
                DocValuesType::SortedNumeric => self
                    .sorted_numeric_doc_values(field)?
                    .map(|v| v as Box<dyn DocIdSetIterator>),
                DocValuesType::SortedSet => self
                    .sorted_set_doc_values(field)?
                    .map(|v| v as Box<dyn DocIdSetIterator>),
                DocValuesType::None => {
                    return Err(crate::Error::FieldExistsUnsupported(field.clone()))
                }
            },
            FieldExistsSource::Vectors => unreachable!("vectors handled above"),
        };
        let Some(mut values) = values else {
            return Ok(ExistsDocs::None);
        };
        exists_docs_from_iter(
            self.max_doc,
            std::iter::from_fn(|| match values.next_doc() {
                Ok(NO_MORE_DOCS) => None,
                doc => Some(doc),
            }),
        )
    }

    pub(super) fn open_simpletext(
        dir: &dyn Directory,
        commit: &segment_infos::SegmentCommitInfo,
        doc_base: i32,
    ) -> Result<Self> {
        let core = Arc::new(SimpleTextCore::open(dir, commit)?);
        let live_docs = read_live_docs(dir, commit, core.info.doc_count)?;
        Ok(Self {
            index_sort: (!core.info.index_sort.is_empty())
                .then(|| Arc::new(core.info.index_sort.clone())),
            max_doc: core.info.doc_count,
            field_infos: core.field_infos.clone(),
            simpletext: Some(core),
            segment_name: commit.segment_name.clone(),
            segment_id: commit.segment_id,
            doc_base,
            del_gen: commit.del_gen,
            field_infos_gen: commit.field_infos_gen,
            doc_values_gen: commit.doc_values_gen,
            live_docs,
            core_cache: crate::reader::CacheHelper::new(),
            reader_cache: crate::reader::CacheHelper::new(),
            query_cache: Arc::default(),
            dvs: None,
            skip_indexes: Arc::default(),
            docs_with_field: Arc::default(),
            segment_suffix: String::new(),
            fields: Arc::new(BlockTreeFields::empty()),
            doc_buf: None,
            pos_buf: None,
            pay_buf: None,
            postings_files: Vec::new(),
            dv_data: None,
            dv_meta: None,
            dv_generations: Vec::new(),
            kdm_buf: None,
            kdi_buf: None,
            kdd_buf: None,
            points_meta: Arc::default(),
            stored: None,
            norms_meta: None,
            norms_data: None,
            term_vectors: None,
            vectors: Vec::new(),
            completion: Vec::new(),
            postings_validated: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }
}

/// Materialize the source iterator without filtering deletions; the scorer's
/// caller applies `liveDocs`, as with binary `IndexedDISI` document sets.
fn exists_docs_from_iter(
    max_doc: i32,
    docs: impl Iterator<Item = crate::Result<i32>>,
) -> crate::Result<ExistsDocs> {
    let len = usize::try_from(max_doc)
        .map_err(|_| lucene_store::Error::Corrupted(format!("negative maxDoc {max_doc}")))?;
    let mut bits = FixedBitSet::new(len);
    for doc in docs {
        let doc = doc?;
        let index = usize::try_from(doc)
            .ok()
            .filter(|&index| index < bits.len())
            .ok_or_else(|| {
                lucene_store::Error::Corrupted(format!(
                    "field-existence doc {doc} outside maxDoc {max_doc}"
                ))
            })?;
        bits.set(index);
    }
    Ok(match bits.cardinality() {
        0 => ExistsDocs::None,
        count if count == bits.len() => ExistsDocs::All,
        _ => ExistsDocs::Bits(bits),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::LeafReader;
    use lucene_store::data_output::DataOutput;
    use lucene_store::{ByteBuffersDirectory, FsDirectory};

    #[test]
    fn existence_sets_preserve_empty_dense_sparse_and_reject_bad_documents() {
        assert!(matches!(
            exists_docs_from_iter(0, [].into_iter()).unwrap(),
            ExistsDocs::None
        ));
        assert!(matches!(
            exists_docs_from_iter(3, (0..3).map(Ok)).unwrap(),
            ExistsDocs::All
        ));
        let ExistsDocs::Bits(bits) = exists_docs_from_iter(3, [Ok(1)].into_iter()).unwrap() else {
            panic!("expected sparse set");
        };
        assert_eq!(bits.cardinality(), 1);
        assert!(bits.get_doc(1));
        assert!(!bits.get_doc(0));
        for (max_doc, doc) in [(0, 0), (3, -1), (3, 3), (3, i32::MAX)] {
            assert!(exists_docs_from_iter(max_doc, [Ok(doc)].into_iter()).is_err());
        }
        assert!(exists_docs_from_iter(-1, [].into_iter()).is_err());
        let error = exists_docs_from_iter(
            3,
            [Err(crate::Error::IllegalState("decode failed".into()))].into_iter(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("decode failed"));
    }

    fn fixture() -> FsDirectory {
        FsDirectory::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/data/codecs_simpletext/codec/flush/index"),
        )
    }

    fn put(dir: &dyn Directory, name: &str, bytes: &[u8]) {
        if dir.list_all().unwrap().iter().any(|file| file == name) {
            dir.delete_file(name).unwrap();
        }
        let mut output = dir.create_output(name).unwrap();
        output.write_bytes(bytes);
        output.close().unwrap();
    }

    fn copy_fixture() -> ByteBuffersDirectory {
        let source = fixture();
        let target = ByteBuffersDirectory::new();
        for name in source.list_all().unwrap() {
            put(&target, &name, &source.open(&name).unwrap());
        }
        target
    }

    #[test]
    fn a_segment_with_only_empty_documents_has_no_optional_components() {
        let dir = ByteBuffersDirectory::new();
        let name = "_empty";
        let mut info = segment_info::SimpleTextSegmentInfo {
            name: name.into(),
            version: lucene_util::version::Version::LATEST,
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: Vec::new(),
            attributes: Vec::new(),
            files: vec![format!("{name}.inf"), format!("{name}.fld")],
            id: [9; 16],
            index_sort: Vec::new(),
        };
        let field_infos = FieldInfos::new(Vec::new()).unwrap();
        put(&dir, &format!("{name}.inf"), &text::field_infos::write(&[]));
        put(
            &dir,
            &format!("{name}.fld"),
            &text::stored_fields_writer::write(
                &[lucene_codecs::stored_fields::Document::default()],
                &field_infos,
            )
            .unwrap(),
        );
        put(
            &dir,
            &format!("{name}.si"),
            &segment_info::write_simpletext(&mut info).unwrap(),
        );
        let commit = segment_infos::SegmentCommitInfo {
            segment_name: name.into(),
            segment_id: info.id,
            codec_name: "SimpleText".into(),
            ..Default::default()
        };
        let reader = SegmentReader::open_simpletext(&dir, &commit, 0).unwrap();
        assert_eq!(reader.max_doc, 1);
        assert_eq!(reader.num_docs(), 1);
        assert!(reader
            .stored_document(0)
            .unwrap()
            .unwrap()
            .fields
            .is_empty());
        assert_eq!(reader.doc_freq("absent", b"term").unwrap(), 0);
        assert!(reader.field_stats("absent").unwrap().is_none());
        assert!(reader.field_norms("absent").unwrap().is_none());
        assert!(reader.terms("absent").unwrap().is_none());
        assert!(reader.term_vectors(0).unwrap().is_none());
        assert!(reader.point_values("absent").unwrap().is_none());
        assert!(reader.float_vector_values("absent").unwrap().is_none());
        assert!(reader.byte_vector_values("absent").unwrap().is_none());
        assert!(reader.norm_values("absent").unwrap().is_none());
        assert!(reader.numeric_doc_values("absent").unwrap().is_none());
        assert!(reader.doc_values_skipper("absent").unwrap().is_none());
        reader.check_integrity().unwrap();
    }

    #[test]
    fn field_metadata_cannot_name_components_missing_from_the_segment() {
        for extension in [
            "inf", "pst", "fld", "dat", "len", "tvc", "dim", "dii", "gri", "vec",
        ] {
            let dir = copy_fixture();
            let infos = segment_infos::read_latest(&dir).unwrap();
            let commit = infos
                .segments
                .iter()
                .find(|info| info.segment_name == "_1")
                .unwrap();
            let file = format!("_1.{extension}");
            let mut info = segment_info::parse_simpletext(
                &dir.open("_1.si").unwrap(),
                "_1",
                &commit.segment_id,
            )
            .unwrap();
            assert!(!info.is_compound_file);
            assert!(info.files.contains(&file));
            info.files.retain(|name| name != &file);
            put(
                &dir,
                "_1.si",
                &segment_info::write_simpletext(&mut info).unwrap(),
            );
            let error = SimpleTextCore::open(&dir, commit).unwrap_err().to_string();
            assert!(error.contains(&file), "missing {file}: {error}");
        }
    }

    #[test]
    fn integrity_checks_include_updated_doc_values_outside_compound_or_core_files() {
        let dir = copy_fixture();
        let infos = segment_infos::read_latest(&dir).unwrap();
        let commit = infos
            .segments
            .iter()
            .find(|info| !info.dv_update_files.is_empty())
            .unwrap();
        let name = commit
            .dv_update_files
            .iter()
            .flat_map(|(_, files)| files)
            .find(|name| name.ends_with(".dat"))
            .unwrap();
        let mut data = dir.open(name).unwrap().to_vec();
        let checksum_digit = data.len().saturating_sub(2);
        data[checksum_digit] = if data[checksum_digit] == b'0' {
            b'1'
        } else {
            b'0'
        };
        put(&dir, name, &data);
        let core = SimpleTextCore::open(&dir, commit).unwrap();
        let error = core.check_integrity().unwrap_err().to_string();
        assert!(error.contains("checksum"), "{error}");
    }

    #[test]
    fn existence_queries_propagate_lazy_doc_values_errors() {
        let dir = copy_fixture();
        let mut damaged = dir.open("_1.dat").unwrap().to_vec();
        let pattern = b"  pattern 0\n0\nT\n";
        let at = damaged
            .windows(pattern.len())
            .position(|part| part == pattern)
            .unwrap();
        *damaged
            .get_mut(at.saturating_add(b"  pattern 0\n".len()))
            .unwrap() = b'x';
        let footer = damaged
            .windows(text::util::CHECKSUM.len())
            .rposition(|part| part == text::util::CHECKSUM)
            .unwrap();
        damaged.truncate(footer);
        text::util::write_checksum(&mut damaged);
        put(&dir, "_1.dat", &damaged);
        let reader = DirectoryReader::open(&dir).unwrap();
        let segment = reader
            .segment_readers()
            .iter()
            .find(|segment| segment.segment_name == "_1")
            .unwrap();
        // A plain numeric field has no count shortcut; opening metadata must
        // not hide the later value decoder error behind an empty result.
        let leaf = segment.field_exists_leaf("num", None).unwrap();
        assert_eq!(crate::weight_count::count_field_exists_leaf(&leaf), None);
        let error = segment.field_exists_docs("num").unwrap_err();
        assert!(
            error.to_string().contains("Unparseable number: \"x\""),
            "{error}"
        );
    }

    #[test]
    fn vector_existence_does_not_decode_unselected_postings() {
        use crate::weight_count::{
            count_field_exists_leaf, field_exists_rewrites_to_match_all_docs,
        };
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/simpletext_field_exists/clean");
        let source = FsDirectory::open(root.join("index"));
        let dir = ByteBuffersDirectory::new();
        for name in source.list_all().unwrap() {
            put(&dir, &name, &source.open(&name).unwrap());
        }
        put(
            &dir,
            "_0.pst",
            &std::fs::read(root.join("vector-corruption.pst")).unwrap(),
        );
        let reader = DirectoryReader::open(&dir).unwrap();
        let segment = &reader.segment_readers()[0];
        let leaf = segment.field_exists_leaf("vector_terms", None).unwrap();
        assert_eq!(
            leaf.source,
            Some(crate::doc_value_query::FieldExistsSource::Vectors)
        );
        assert_eq!(leaf.terms_doc_count, None);
        assert_eq!(leaf.skipper_doc_count, None);
        let count = count_field_exists_leaf(&leaf).unwrap();
        let rewrite = field_exists_rewrites_to_match_all_docs(&[leaf], None, reader.max_doc());
        let error = segment.field_stats("vector_terms").unwrap_err();
        assert!(error.to_string().contains("x"), "{error}");
        let actual = format!("count\t{count}\nrewrite\t{rewrite}\nterms\terror\n");
        assert_eq!(
            actual,
            std::fs::read_to_string(root.join("vector-corruption.tsv")).unwrap()
        );
        let ExistsDocs::Bits(bits) = segment.field_exists_docs("vector_terms").unwrap() else {
            panic!("expected sparse vector documents");
        };
        assert_eq!(bits.cardinality(), 4);
        for doc in 0..segment.max_doc {
            assert_eq!(bits.get_doc(doc), doc % 2 == 0);
        }
    }

    #[test]
    fn statistics_and_norms_propagate_lazy_term_errors() {
        let dir = copy_fixture();
        let mut bytes = dir.open("_1.pst").unwrap().to_vec();
        let at = bytes
            .windows(b"freq 1\n".len())
            .position(|part| part == b"freq 1\n")
            .unwrap();
        bytes.insert(at.saturating_add(b"freq ".len()), b'-');
        let footer = bytes
            .windows(text::util::CHECKSUM.len())
            .rposition(|part| part == text::util::CHECKSUM)
            .unwrap();
        let mut damaged = bytes[..footer].to_vec();
        text::util::write_checksum(&mut damaged);
        put(&dir, "_1.pst", &damaged);
        let reader = DirectoryReader::open(&dir).unwrap();
        let segment = reader
            .segment_readers()
            .iter()
            .find(|segment| segment.segment_name == "_1")
            .unwrap();
        let field = segment
            .field_infos
            .fields
            .iter()
            .find(|field| field.name == "freqs")
            .unwrap();
        let error = segment.field_stats(&field.name).unwrap_err().to_string();
        assert!(error.contains("negative"), "{error}");
        assert!(segment.field_norms(&field.name).is_err());
        assert!(segment.field_exists_leaf(&field.name, None).is_err());
        assert!(reader.avg_field_length(&field.name).is_err());
        assert!(reader.field_norms(&field.name).is_err());
        assert!(reader
            .field_norms_by_field(std::slice::from_ref(&field.name))
            .is_err());
    }

    #[test]
    fn vector_dimensions_must_agree_with_the_field_infos() {
        let dir = copy_fixture();
        let original = dir.open("_1.gri").unwrap();
        let source = String::from_utf8(original.to_vec()).unwrap();
        assert!(source.contains("vector-dimension 4\n"));
        let source = source.replacen("vector-dimension 4\n", "vector-dimension 5\n", 1);
        let footer = source.rfind("checksum ").unwrap();
        let mut damaged = source.as_bytes()[..footer].to_vec();
        text::util::write_checksum(&mut damaged);
        put(&dir, "_1.gri", &damaged);
        let reader = DirectoryReader::open(&dir).unwrap();
        let segment = reader
            .segment_readers()
            .iter()
            .find(|segment| segment.segment_name == "_1")
            .unwrap();
        let error = segment
            .float_vector_values("fvec")
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("Inconsistent vector dimension"), "{error}");
        assert!(segment.field_exists_docs("fvec").is_err());
        assert!(segment.field_exists_leaf("fvec", None).is_err());
    }

    #[test]
    fn delete_only_reopen_reads_text_live_docs_and_keeps_the_segment_core() {
        let dir = copy_fixture();
        let old = DirectoryReader::open(&dir).unwrap();
        let mut infos = old.segment_infos.clone();
        let original = &old.segments[0];
        let mut live = original.live_docs.as_deref().unwrap().clone();
        let doc = (0..live.len()).find(|&doc| live.get(doc)).unwrap();
        live.clear(doc);
        let commit = &mut infos.segments[0];
        commit.del_gen = commit.del_gen.saturating_add(1);
        commit.del_count = commit.del_count.saturating_add(1);
        let name =
            text::live_docs::live_docs_file_name(&commit.segment_name, commit.del_gen).unwrap();
        put(&dir, &name, &text::live_docs::write(&live));
        let reopened = old.reopen_at(&dir, infos).unwrap();
        let changed = &reopened.segments[0];
        assert!(Arc::ptr_eq(
            original.simpletext.as_ref().unwrap(),
            changed.simpletext.as_ref().unwrap()
        ));
        assert!(original.live_docs.as_ref().unwrap().get(doc));
        assert!(!changed.live_docs.as_ref().unwrap().get(doc));
        assert_eq!(original.num_docs().saturating_sub(1), changed.num_docs());
        assert_eq!(
            original.core_cache_helper().key(),
            changed.core_cache_helper().key()
        );
        assert_ne!(
            original.reader_cache_helper().key(),
            changed.reader_cache_helper().key()
        );
        let mut invalid = reopened.segment_infos.segments[0].clone();
        invalid.del_count = invalid.del_count.saturating_add(1);
        assert!(read_live_docs(&dir, &invalid, changed.max_doc).is_err());
    }
}
