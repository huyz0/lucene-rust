#![cfg(test)]
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
        let mut info =
            segment_info::parse_simpletext(&dir.open("_1.si").unwrap(), "_1", &commit.segment_id)
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
    use crate::weight_count::{count_field_exists_leaf, field_exists_rewrites_to_match_all_docs};
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
    let name = text::live_docs::live_docs_file_name(&commit.segment_name, commit.del_gen).unwrap();
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
