//! Paged zstd stored fields and the storage layer's alignment contract: a
//! reader opens the `.fdt` (or the `.cfs` holding it) with the page size as
//! its alignment hint, and a compound file starts the `.fdt` on a page.

#![allow(clippy::arithmetic_side_effects)]

use std::sync::Mutex;

use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField, ZSTD_PAGED_PAGE_SIZE};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::{Directory, FsDirectory, Input};
use lucene_store::{FsIndexOutput, Lock};
use lucene_util::test_support::TempDir;

/// Records every `open_aligned` call.
struct Recording<'a> {
    inner: &'a dyn Directory,
    aligned: Mutex<Vec<(String, usize)>>,
}

impl Directory for Recording<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        self.inner.open(name)
    }
    fn open_aligned(&self, name: &str, alignment: usize) -> lucene_store::Result<Input> {
        self.aligned
            .lock()
            .unwrap()
            .push((name.to_string(), alignment));
        self.inner.open_aligned(name, alignment)
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<FsIndexOutput> {
        self.inner
            .create_output_with_estimate(name, estimated_bytes)
    }
    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<FsIndexOutput> {
        self.inner
            .create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
}

fn body_field() -> FieldInfo {
    FieldInfo {
        name: "body".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn body(i: usize) -> String {
    format!(
        "{{\"id\":{i},\"msg\":\"request {} took {} ms\"}}",
        i * 7919 % 10_007,
        i % 97
    )
}

/// Writes `n` documents in one segment, paged or not, compound or not.
fn write(dir: &dyn Directory, n: usize, paged: bool, compound: bool) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, vec![body_field()], "Lucene104", version).unwrap();
    if paged {
        let codec = lucene_codecs::codec::ZstdPagedLucene104Codec::new(3).unwrap();
        w.set_codec(Some(std::sync::Arc::new(codec))).unwrap();
    }
    w.set_use_compound_file(compound);
    for i in 0..n {
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body(i)),
            }],
        })
        .unwrap();
    }
    w.commit().unwrap();
}

/// Opens the index through a [`Recording`] directory, checks every document,
/// and returns the `open_aligned` calls.
fn read(dir: &dyn Directory, n: usize) -> Vec<(String, usize)> {
    let rec = Recording {
        inner: dir,
        aligned: Mutex::new(Vec::new()),
    };
    let reader = DirectoryReader::open(&rec).unwrap();
    let mut seen = 0;
    for seg in reader.segment_readers() {
        for d in 0..seg.max_doc {
            let doc = seg.stored_document(d).unwrap().unwrap();
            let FieldValue::String(s) = &doc.fields[0].value else {
                panic!("a string body");
            };
            assert_eq!(*s, body(seen));
            seen += 1;
        }
    }
    assert_eq!(seen, n);
    rec.aligned.into_inner().unwrap()
}

fn names_ending(calls: &[(String, usize)], ext: &str) -> Vec<usize> {
    calls
        .iter()
        .filter(|(name, _)| name.ends_with(ext))
        .map(|(_, a)| *a)
        .collect()
}

#[test]
fn a_loose_paged_fdt_is_opened_with_the_page_size() {
    let tmp = TempDir::new("zstd-paged-aligned-loose");
    let dir = FsDirectory::open(tmp.path());
    write(&dir, 3000, true, false);
    let calls = read(&dir, 3000);
    assert_eq!(
        names_ending(&calls, ".fdt"),
        [ZSTD_PAGED_PAGE_SIZE],
        "{calls:?}"
    );
}

#[test]
fn a_compound_file_starts_the_paged_fdt_on_a_page() {
    let tmp = TempDir::new("zstd-paged-aligned-cfs");
    let dir = FsDirectory::open(tmp.path());
    write(&dir, 3000, true, true);
    let calls = read(&dir, 3000);
    assert_eq!(
        names_ending(&calls, ".cfs"),
        [ZSTD_PAGED_PAGE_SIZE],
        "{calls:?}"
    );

    let cfe = dir
        .list_all()
        .unwrap()
        .into_iter()
        .find(|f| f.ends_with(".cfe"))
        .expect("a compound segment");
    let sis = lucene_index::segment_infos::read_latest(&dir).unwrap();
    let entries =
        compound_format::parse_entries(&dir.open(&cfe).unwrap(), &sis.segments[0].segment_id)
            .unwrap();
    for (name, entry) in &entries.entries {
        if name.ends_with(".fdt") {
            assert_eq!(entry.offset % ZSTD_PAGED_PAGE_SIZE as i64, 0, "{name}");
        } else {
            assert_eq!(entry.offset % 64, 0, "{name}");
        }
    }
}

#[test]
fn other_stored_fields_are_opened_as_before() {
    for compound in [false, true] {
        let tmp = TempDir::new("zstd-paged-aligned-lz4");
        let dir = FsDirectory::open(tmp.path());
        write(&dir, 500, false, compound);
        assert_eq!(read(&dir, 500), [], "compound {compound}");
    }
}
