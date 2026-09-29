//! `terminate::cannot_reach` at its boundary: a limit equal to the number of
//! matches is reachable -- the search stops at the segment after the one
//! holding the last match, and reports it (`terminatedEarly`) -- so only a
//! limit above every possible match may skip the cut. The fixture runs in
//! `terminate_after_fixtures.rs` hold the rule to real Lucene, but their
//! deletions keep the dictionaries' bound above the true count; this index
//! has none, so the bound is exact.

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::terminate::{cannot_reach, terminate_after};
use lucene_search::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

#[test]
fn a_limit_equal_to_the_matches_is_reachable() {
    let path = TempDir::new("terminate-unreachable");
    let dir = FsDirectory::open(&path);
    let fields = vec![
        FieldInfo::new("id", 0),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    // Three segments: `a` three times, `a` twice, then none.
    for words in [&["a", "a", "a"][..], &["a", "a"], &["b", "b"]] {
        for (i, w) in words.iter().enumerate() {
            let fields = vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(i.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String((*w).to_string()),
                },
            ];
            writer.add_document(Document { fields }).unwrap();
        }
        writer.flush().unwrap();
    }
    writer.commit().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let q = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", b"a".to_vec()))],
        ..Default::default()
    };
    // Five matches: a limit of five is reached, and the third segment ends
    // the search early.
    let at = terminate_after(&segments, &q, 5).unwrap();
    assert_eq!((at.collected, at.terminated), (5, true));
    assert!(!cannot_reach(&segments, &q, 5));
    // Six cannot be reached, and nothing ends early.
    let past = terminate_after(&segments, &q, 6).unwrap();
    assert_eq!((past.collected, past.terminated), (5, false));
    assert!(cannot_reach(&segments, &q, 6));
}
