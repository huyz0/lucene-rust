//! A segment real Lucene wrote with **two postings formats and two
//! doc-values formats** (`PerFieldPostingsFormat`/`PerFieldDocValuesFormat`
//! routing fields to `Lucene104PostingsFormat()` and
//! `Lucene104PostingsFormat(10, 20)`, `Lucene90DocValuesFormat()` and
//! `Lucene90DocValuesFormat(1024)`): one set of files per (format, suffix),
//! `_0_Lucene104_0.*`/`_0_Lucene104_1.*` and `_0_Lucene90_0.*`/
//! `_0_Lucene90_1.*`. Regenerate with `fixtures/src/GenPerFieldFormats.java`.
//!
//! `DirectoryReader` must route every field to its own format's files: each
//! field's `.fnm` attributes name the format and suffix, and a reader that
//! opened only the first `.tim` (or `.dvm`) it found reports the other
//! format's fields as having no terms (or no values) -- silently.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_term_query, TermQuery, VecCollector};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/per_field_formats_index/"
    )
    .to_string()
}

fn manifest(key: &str) -> String {
    let text = std::fs::read_to_string(format!("{}manifest.properties", fixture_dir()))
        .expect("run fixtures generator first (GenPerFieldFormats)");
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
        .to_string()
}

#[test]
fn every_field_is_read_from_its_own_postings_format() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open the per-field index");
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 1);
    let seg = &segments[0];
    for field in ["a_id", "a_text", "b_id", "b_tag"] {
        let terms = seg
            .fields
            .field(field)
            .unwrap_or_else(|| panic!("{field} has no terms"));
        let want_terms: i64 = manifest(&format!("num_terms.{field}")).parse().unwrap();
        assert_eq!(terms.num_terms, want_terms, "{field}");
        let mut checked = 0;
        for entry in manifest(&format!("terms.{field}")).split(';') {
            let Some((term, docs)) = entry.split_once('=') else {
                continue;
            };
            let want: Vec<i32> = docs
                .split(',')
                .map(|d| d.split_once(':').unwrap().0.parse().unwrap())
                .collect();
            let mut collector = VecCollector::default();
            search_term_query(
                seg.fields,
                seg.doc_in,
                None,
                &TermQuery::new(field, term.as_bytes()),
                &mut collector,
            )
            .unwrap_or_else(|e| panic!("{field}:{term}: {e}"));
            assert_eq!(collector.docs, want, "{field}:{term}");
            checked += 1;
        }
        assert!(checked >= 10, "{field}: only {checked} terms checked");
    }
}

#[test]
fn every_field_is_read_from_its_own_doc_values_format() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open the per-field index");
    let seg = &reader.segment_readers()[0];
    let number = |name: &str| -> i32 {
        seg.field_infos()
            .fields
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .number
    };
    for field in ["dva_num", "dvb_num"] {
        let (meta, data) = seg
            .doc_values_for_field(number(field))
            .unwrap_or_else(|| panic!("{field}: no doc values"));
        let entry = meta
            .numeric_entry(number(field))
            .unwrap_or_else(|| panic!("{field}: no numeric entry"));
        let want: Vec<i64> = manifest(&format!("dv.{field}"))
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        for (doc, want) in want.iter().enumerate() {
            let got = doc_values::numeric_value(data, entry, doc as i32).unwrap();
            assert_eq!(got, Some(*want), "{field} doc {doc}");
        }
    }
    let field = number("dvb_sorted");
    let (meta, data) = seg.doc_values_for_field(field).expect("dvb_sorted");
    let entry = meta.sorted_entry(field).expect("sorted entry");
    let want: Vec<String> = manifest("dv.dvb_sorted")
        .split(',')
        .map(str::to_string)
        .collect();
    let terms = lucene_codecs::terms_dict::decode_all_terms(data, &entry.terms).unwrap();
    for (doc, want) in want.iter().enumerate() {
        let ord = doc_values::sorted_ord(data, entry, doc as i32)
            .unwrap()
            .unwrap();
        assert_eq!(terms[ord as usize], want.as_bytes(), "dvb_sorted doc {doc}");
    }
}

/// This port's `IndexWriter` over the Java index: a delete by a term of the
/// *second* postings format (`b_id`) must find its document there, and a
/// force-merge must read every field from its own postings and doc-values
/// format -- the merged segment, written under this writer's single-format
/// routing, keeps every term and value of the survivors.
#[test]
fn the_writer_deletes_from_and_merges_a_multi_format_segment() {
    use lucene_index::buffered_updates::Term;
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

    let tmp = lucene_util::test_support::TempDir::new("per-field-java-merge");
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name != "manifest.properties" {
            std::fs::copy(entry.path(), tmp.join(&name)).unwrap();
        }
    }
    let dir = FsDirectory::open(&tmp);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    // `b_id` is `b%05d` of `doc * 3`: document 1.
    writer
        .delete_documents_by_term(&[Term {
            field: "b_id".to_string(),
            bytes: b"b00003".to_vec(),
        }])
        .unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    drop(writer);

    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 1);
    let seg = &segments[0];
    // The survivors keep their order: old doc `d` is new `d - 1` past doc 1.
    let remap = |d: i32| if d > 1 { d - 1 } else { d };
    for field in ["a_id", "b_tag"] {
        let mut checked = 0;
        for entry in manifest(&format!("terms.{field}")).split(';') {
            let Some((term, docs)) = entry.split_once('=') else {
                continue;
            };
            let want: Vec<i32> = docs
                .split(',')
                .map(|d| d.split_once(':').unwrap().0.parse().unwrap())
                .filter(|&d| d != 1)
                .map(remap)
                .collect();
            let mut collector = VecCollector::default();
            search_term_query(
                seg.fields,
                seg.doc_in,
                None,
                &TermQuery::new(field, term.as_bytes()),
                &mut collector,
            )
            .unwrap();
            assert_eq!(collector.docs, want, "{field}:{term}");
            checked += 1;
        }
        assert!(checked >= 10, "{field}");
    }
    let seg = &reader.segment_readers()[0];
    let number = |name: &str| -> i32 {
        seg.field_infos()
            .fields
            .iter()
            .find(|f| f.name == name)
            .unwrap()
            .number
    };
    let field = number("dvb_num");
    let (meta, data) = seg.doc_values_for_field(field).expect("dvb_num");
    let entry = meta.numeric_entry(field).expect("numeric entry");
    let want: Vec<i64> = manifest("dv.dvb_num")
        .split(',')
        .map(|v| v.parse().unwrap())
        .enumerate()
        .filter(|&(d, _)| d != 1)
        .map(|(_, v)| v)
        .collect();
    for (doc, want) in want.iter().enumerate() {
        let got = doc_values::numeric_value(data, entry, doc as i32).unwrap();
        assert_eq!(got, Some(*want), "dvb_num doc {doc}");
    }
}
