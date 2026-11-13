//! [`IndexWriter::set_default_postings_format`] with
//! [`Lucene104PostingsFormat::rust_zstd`]: flushed and merged segments put
//! every field nothing else routes into one `ZstdTerms104` group with zstd
//! term blocks, deletes by term find their documents, and `check_index` walks
//! every field clean. Setting the default back writes stock segments again.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::per_field_postings::{self, Lucene104PostingsFormat};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::check_index;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const DOCS: usize = 3000;

fn suffixes(dir: &FsDirectory) -> Vec<Vec<String>> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let files = dir.list_all().unwrap();
    infos
        .segments
        .iter()
        .map(|s| {
            let own: Vec<String> = files
                .iter()
                .filter(|f| f.starts_with(&format!("{}_", s.segment_name)))
                .cloned()
                .collect();
            per_field_postings::group_suffixes(&own, &s.segment_name)
        })
        .collect()
}

fn check(dir: &FsDirectory) {
    let results = check_index::check_directory(dir).unwrap();
    for result in &results {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
    let names: Vec<&str> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .map(|c| c.name.as_str())
        .collect();
    for field in ["body", "url"] {
        let wanted = format!("postings.field_summary:{field}");
        assert!(names.contains(&wanted.as_str()), "{wanted} did not run");
    }
}

#[test]
fn zstd_term_blocks_flush_merge_and_check_clean() {
    let path = TempDir::new("zstd-term-blocks");
    let dir = FsDirectory::open(&path);
    let fields = vec![
        FieldInfo::new("id", 0),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
        FieldInfo::new("url", 2).with_index_options(IndexOptions::Docs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version).unwrap();
    for name in ["body", "url"] {
        writer.add_postings_field(name).unwrap();
    }
    writer.set_default_postings_format(Lucene104PostingsFormat::rust_zstd(3).unwrap());
    let text = |field_number: i32, s: String| StoredField {
        field_number,
        value: FieldValue::String(s),
    };
    for i in 0..DOCS {
        writer
            .add_document(Document {
                fields: vec![
                    text(0, format!("doc{i}")),
                    text(1, format!("shared word{} other{}", i % 13, i % 101)),
                    text(2, format!("/logs/service-{}/requests/{i:08}", i % 9)),
                ],
            })
            .unwrap();
        if (i + 1) % 1000 == 0 {
            writer.commit().unwrap();
        }
    }
    writer
        .delete_documents_by_term(&[Term {
            field: "url".to_string(),
            bytes: b"/logs/service-3/requests/00000012".to_vec(),
        }])
        .unwrap();
    writer.commit().unwrap();
    assert_eq!(suffixes(&dir), vec![vec!["ZstdTerms104_0".to_string()]; 3]);
    check(&dir);

    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert_eq!(suffixes(&dir), vec![vec!["ZstdTerms104_0".to_string()]]);
    let infos = segment_infos::read_latest(&dir).unwrap();
    let seg = &infos.segments[0];
    let tim = dir
        .open(&format!("{}_ZstdTerms104_0.tim", seg.segment_name))
        .unwrap();
    assert_eq!(
        lucene_store::codec_util::peek_codec_name(&tim).as_deref(),
        Some("ZstdBlockTreeTermsDict")
    );
    assert_eq!(seg.del_count, 0, "the merge dropped the deleted document");
    check(&dir);

    // Back to the default: the next merge writes stock files again.
    writer.set_default_postings_format(Lucene104PostingsFormat::default());
    writer
        .add_document(Document {
            fields: vec![
                text(0, "late".into()),
                text(1, "late".into()),
                text(2, "/late".into()),
            ],
        })
        .unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert_eq!(suffixes(&dir), vec![vec!["Lucene104_0".to_string()]]);
    check(&dir);
}
