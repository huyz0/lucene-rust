//! The test framework's `Lucene104` filter codecs installed in
//! [`lucene_index::index_writer::IndexWriter`] (`IndexWriter::set_codec`):
//! `CheapBastard` writes every field's postings with its direct
//! `Lucene104PostingsFormat(100, 200)` and `ConfigurableMCodec` every
//! field's vectors with its direct `Lucene99HnswVectorsFormat(128, 100)`.
//! Both direct formats' files carry no codec suffix and their fields no
//! per-field attributes, flushed and merged alike, as Java writes them.
//! `Asserting` routes every field to its `Asserting` postings, doc-values
//! and vectors formats, whose files and attributes carry that name; every
//! segment records the codec's name; CheckIndex passes and the reader serves
//! every term, document and vector. `VerifyFilterCodecs` (verify-write-path)
//! has Lucene read the same indexes; [`java_written_segments`] reads
//! segments Lucene wrote with them.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

#[path = "../examples/write_filter_codecs_fixture.rs"]
mod fixture;

use lucene_codecs::compound_format;
use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::stored_fields::FieldValue;
use lucene_codecs::{per_field_doc_values, per_field_knn_vectors, per_field_postings};
use lucene_index::index_writer::IndexWriter;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_term_query, TermQuery, VecCollector};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

use fixture::CODECS;

/// Every live document's id, and the ids each `body` term matches, sorted.
fn read_back(dir: &dyn Directory) -> (Vec<String>, Vec<(String, Vec<String>)>) {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let readers = reader.segment_readers();
    let id = |index: usize, doc: i32| {
        let stored = readers[index].stored_document(doc).unwrap().unwrap();
        let FieldValue::String(id) = &stored.fields[0].value else {
            panic!("an id");
        };
        id.clone()
    };
    let mut live = Vec::new();
    for (index, seg) in readers.iter().enumerate() {
        for doc in 0..seg.max_doc {
            if seg.live_docs().is_none_or(|bits| bits.get(doc as usize)) {
                live.push(id(index, doc));
            }
        }
    }
    live.sort();
    let mut terms = Vec::new();
    let words = (0..5)
        .map(|k| format!("w{k}"))
        .chain((0..7).map(|k| format!("x{k}")))
        .chain(["common".to_string()]);
    for word in words {
        let mut docs = Vec::new();
        for (index, seg) in opened.as_open_segments().iter().enumerate() {
            let mut collector = VecCollector::default();
            search_term_query(
                seg.fields,
                seg.doc_in,
                seg.live_docs,
                &TermQuery::new("body", word.as_bytes()),
                &mut collector,
            )
            .unwrap();
            docs.extend(collector.docs.iter().map(|&d| id(index, d)));
        }
        docs.sort();
        terms.push((word, docs));
    }
    (live, terms)
}

/// What [`read_back`] must find in the documents `0..count` the fixture
/// writes, less the deleted ones.
fn expected(count: usize) -> (Vec<String>, Vec<(String, Vec<String>)>) {
    let ids = |keep: &dyn Fn(usize) -> bool| {
        let mut v: Vec<String> = (0..count)
            .filter(|&i| fixture::live(i) && keep(i))
            .map(|i| format!("d{i}"))
            .collect();
        v.sort();
        v
    };
    let mut terms = Vec::new();
    for k in 0..5 {
        terms.push((format!("w{k}"), ids(&|i| i % 5 == k)));
    }
    for k in 0..7 {
        terms.push((format!("x{k}"), ids(&|i| i % 7 == k)));
    }
    terms.push(("common".to_string(), ids(&|_| true)));
    (ids(&|_| true), terms)
}

/// The segment's files, by full name, and each one's bytes: a compound
/// segment's (Lucene writes its small flushed segments so) read out of its
/// `.cfs`.
fn segment_files(dir: &dyn Directory, name: &str, id: &[u8; 16]) -> Vec<(String, Vec<u8>)> {
    let si = dir.open(&format!("{name}.si")).unwrap();
    let si = lucene_index::segment_info::parse(&si, id).unwrap();
    if si.is_compound_file {
        let cfs = dir.open(&format!("{name}.cfs")).unwrap();
        let cfe = dir.open(&format!("{name}.cfe")).unwrap();
        let entries = compound_format::parse_entries(&cfe, id).unwrap();
        entries
            .names()
            .map(|member| {
                let bytes = compound_format::open_input(&cfs, &entries, member)
                    .unwrap()
                    .as_slice()
                    .to_vec();
                (format!("{name}{member}"), bytes)
            })
            .collect()
    } else {
        let prefix = format!("{name}.");
        let suffixed = format!("{name}_");
        dir.list_all()
            .unwrap()
            .into_iter()
            .filter(|f| f.starts_with(&prefix) || f.starts_with(&suffixed))
            .map(|f| {
                let bytes = dir.open(&f).unwrap().to_vec();
                (f, bytes)
            })
            .collect()
    }
}

/// The commit's segments, each with its codec, files and per-field
/// attributes checked against what `codec` writes.
fn check_segments(dir: &dyn Directory, codec: &str) {
    let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
    assert!(!infos.segments.is_empty());
    let reader = DirectoryReader::open(dir).unwrap();
    for (sci, seg) in infos.segments.iter().zip(reader.segment_readers()) {
        assert_eq!(sci.codec_name, codec);
        let name = &sci.segment_name;
        let files = segment_files(dir, name, &sci.segment_id);
        let file = |f: &str| files.iter().find(|(n, _)| n == f).map(|(_, b)| b);
        let has = |f: &str| file(f).is_some();
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        let attr = |key: &str| {
            seg.field_infos()
                .fields
                .iter()
                .any(|f| f.attributes.iter().any(|(k, _)| k == key))
        };
        match codec {
            "CheapBastard" => {
                for ext in ["tim", "tip", "tmd", "doc", "pos", "psm"] {
                    assert!(has(&format!("{name}.{ext}")), "{name}.{ext} in {names:?}");
                }
                assert!(!names.iter().any(|f| f.contains("_Lucene104_")));
                assert!(!attr(per_field_postings::PER_FIELD_FORMAT_KEY));
                assert!(attr(per_field_knn_vectors::PER_FIELD_FORMAT_KEY));
            }
            "Asserting" => {
                // One `Asserting` instance per per-field format, named so.
                let group = format!("{name}_Asserting_0");
                for ext in [
                    "tim", "tip", "tmd", "doc", "pos", "psm", "dvd", "dvm", "vec", "vemf", "vem",
                    "vex",
                ] {
                    assert!(has(&format!("{group}.{ext}")), "{group}.{ext} in {names:?}");
                }
                assert!(!names.iter().any(|f| f.contains("_Lucene")), "{names:?}");
                for key in [
                    per_field_postings::PER_FIELD_FORMAT_KEY,
                    per_field_doc_values::PER_FIELD_FORMAT_KEY,
                    per_field_knn_vectors::PER_FIELD_FORMAT_KEY,
                ] {
                    let values: Vec<&str> = seg
                        .field_infos()
                        .fields
                        .iter()
                        .flat_map(|f| f.attributes.iter())
                        .filter(|(k, _)| k == key)
                        .map(|(_, v)| v.as_str())
                        .collect();
                    assert!(!values.is_empty(), "{key}");
                    assert!(
                        values.iter().all(|v| *v == "Asserting"),
                        "{key}: {values:?}"
                    );
                }
            }
            _ => {
                for ext in ["vec", "vemf", "vem", "vex"] {
                    assert!(has(&format!("{name}.{ext}")), "{name}.{ext} in {names:?}");
                }
                assert!(!names
                    .iter()
                    .any(|f| f.contains("_Lucene99HnswVectorsFormat_")));
                assert!(!attr(per_field_knn_vectors::PER_FIELD_FORMAT_KEY));
                assert!(attr(per_field_postings::PER_FIELD_FORMAT_KEY));
                // The configured M, as `Lucene99HnswVectorsWriter` records it.
                let vem = file(&format!("{name}.vem")).unwrap();
                let vex = file(&format!("{name}.vex")).unwrap();
                let graph = HnswVectorsReader::open(vem, vex, &sci.segment_id, "").unwrap();
                assert!(
                    graph.fields().iter().all(|f| f.m == 128),
                    "{:?}",
                    graph.fields()
                );
            }
        }
        let v = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap()
            .number;
        let flat = seg
            .flat_vectors_reader_for(v)
            .unwrap()
            .expect("v's vectors");
        assert_eq!(flat.field(v).unwrap().size, seg.max_doc);
    }
    for result in lucene_index::check_index::check_directory(dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

#[test]
fn filter_codecs_write_their_direct_formats_through_index_writer() {
    for codec in CODECS {
        for merge in [false, true] {
            let tmp = TempDir::new("filter-codecs");
            let dir = FsDirectory::open(tmp.path());
            fixture::write(&dir, codec, merge);
            let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
            assert_eq!(
                infos.segments.len(),
                if merge { 1 } else { fixture::SEGMENTS },
                "{codec}"
            );
            check_segments(&dir, codec);
            assert_eq!(
                read_back(&dir),
                expected(fixture::PER_SEGMENT * fixture::SEGMENTS),
                "{codec} merge={merge}"
            );
        }
    }
}

#[test]
fn set_codec_none_returns_to_lucene104() {
    lucene_test_framework::codecs::register_codecs();
    let tmp = TempDir::new("filter-codecs-reset");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", fixture::version()).unwrap();
    w.set_codec(Some(fixture::codec("CheapBastard"))).unwrap();
    w.add_fields_document(&fixture::doc(0)).unwrap();
    w.commit().unwrap();
    w.set_codec(None).unwrap();
    w.add_fields_document(&fixture::doc(1)).unwrap();
    w.commit().unwrap();
    let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
    let names: Vec<&str> = infos
        .segments
        .iter()
        .map(|s| s.codec_name.as_str())
        .collect();
    assert_eq!(names, ["CheapBastard", "Lucene104"]);
    // A merge of the two writes the writer's current codec.
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    drop(w);
    check_segments_lucene104(&dir);
    assert_eq!(read_back(&dir), expected(2));
}

fn check_segments_lucene104(dir: &dyn Directory) {
    let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
    assert_eq!(infos.segments.len(), 1);
    assert_eq!(infos.segments[0].codec_name, "Lucene104");
    for result in lucene_index::check_index::check_directory(dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// verify-write-path hands this test the directory `VerifyFilterCodecs`
/// left behind: each codec's `merged` index, this port's merged segment with
/// one Lucene appended (a compound file) under the same codec. Both are
/// read, merged by this port into one, and read again.
/// Skipped (and reported so, which that job fails) without the directory.
#[test]
fn java_written_segments() {
    let Ok(root) = std::env::var("FILTER_CODECS_INDEX_DIR") else {
        println!("skipped: FILTER_CODECS_INDEX_DIR is not set");
        return;
    };
    lucene_test_framework::codecs::register_codecs();
    for codec in CODECS {
        let path = std::path::Path::new(&root).join(codec).join("merged");
        let dir = FsDirectory::open(&path);
        let total = fixture::PER_SEGMENT * (fixture::SEGMENTS + 1);
        check_segments(&dir, codec);
        assert_eq!(read_back(&dir), expected(total), "{codec}");
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", fixture::version()).unwrap();
        w.set_codec(Some(fixture::codec(codec))).unwrap();
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        drop(w);
        check_segments(&dir, codec);
        assert_eq!(read_back(&dir), expected(total), "{codec} merged");
    }
}
