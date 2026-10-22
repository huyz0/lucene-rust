//! `BaseTestCheckIndex`: the `CheckIndex` tests a directory implementation
//! runs over itself, as functions taking that directory.
//!
//! Differs: the port's `CheckIndex` is
//! [`lucene_index::check_index::check_directory`], which reports counts as
//! [`lucene_index::check_index::CheckStats`]: the term, frequency, position,
//! norm and term-vector counts Java asserts are there, the stored-field
//! counts and `diagnostics` are not. `checkIndex(onlySegments)` is
//! [`lucene_index::check_index::check_segment_in_commit`] over the named
//! segments the commit has. Reading takes no lock here, so
//! `testObtainsLock` checks the one operation that does take the write lock,
//! [`lucene_index::check_index::exorcise`]. The writers are a default
//! `IndexWriter`, not `newIndexWriterConfig`'s random one: what is checked
//! (the counts after deletes, checksums, the lock) does not depend on it.

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::check_index::{check_directory, check_segment_in_commit, exorcise, CheckResult};
use lucene_index::document::{Document, Field, FieldType, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use crate::analysis::mock_analyzer::MockAnalyzer;
use crate::util::line_file_docs::LineFileDocs;
use crate::util::test_util;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: i32 = 32766;

fn writer(dir: &dyn Directory) -> IndexWriter<'_> {
    IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open an IndexWriter")
}

/// Every result of a `CheckIndex` run, which must all pass.
#[track_caller]
fn assert_clean(results: &[CheckResult]) {
    for r in results {
        assert!(
            r.all_passed(),
            "CheckIndex failed: {}: {:?}",
            r.segment_name,
            r.failures()
        );
    }
}

/// `testDeletedDocs(dir)`: 19 documents with term vectors merged into one
/// segment, one deleted; `CheckIndex` counts the other 18.
pub fn test_deleted_docs(r: &mut JavaRandom, dir: &dyn Directory) {
    let mut w = writer(dir);
    w.set_analyzer(Some(Arc::new(MockAnalyzer::new(r).build())));
    w.set_max_buffered_docs(2).unwrap();
    let mut t = FieldType::copy_of(&TextField::type_stored());
    t.set_store_term_vectors(true).unwrap();
    t.set_store_term_vector_positions(true).unwrap();
    t.set_store_term_vector_offsets(true).unwrap();
    for i in 0..19 {
        let mut doc = Document::new();
        doc.add(Field::from_string("field", format!("aaa{i}"), t.clone()).unwrap());
        w.add_fields_document(&doc).unwrap();
    }
    // The port's forceMerge merges the committed segments only (Java's
    // flushes first), so the buffered documents are committed before it.
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("field", "aaa5")])
        .unwrap();
    w.close().unwrap();

    let results = check_directory(dir).unwrap();
    assert_clean(&results);
    // The commit, then its one segment.
    assert_eq!(
        results.len(),
        2,
        "{:?}",
        results.iter().map(|r| &r.segment_name).collect::<Vec<_>>()
    );
    let seg = &results[1].stats;
    assert_eq!(seg.norm_fields, 1);
    assert_eq!(seg.term_count, 18);
    assert_eq!(seg.del_term_count, 1);
    assert_eq!(seg.term_doc_pairs, 18);
    assert_eq!(seg.token_count, 18);
    assert_eq!(seg.term_vector_fields, 18);

    // `checkIndex(["_0"])`: only the segments named, which after the
    // forced merge is none.
    let infos = segment_infos::read_latest(dir).unwrap();
    for commit in infos.segments.iter().filter(|c| c.segment_name == "_0") {
        assert!(check_segment_in_commit(dir, Some(&infos), commit).all_passed());
    }
}

/// `testChecksumsOnly(dir)`: 100 line-file documents and an empty one
/// under a random maximum token length; `CheckIndex` is clean.
pub fn test_checksums_only(r: &mut JavaRandom, dir: &dyn Directory) {
    let mut lf = LineFileDocs::new(r).unwrap();
    let max_token_length = test_util::next_int(r, 1, MAX_TERM_LENGTH);
    let analyzer = MockAnalyzer::new(r)
        .max_token_length(max_token_length)
        .build();
    let mut w = writer(dir);
    w.set_analyzer(Some(Arc::new(analyzer)));
    for _ in 0..100 {
        w.add_fields_document(&lf.next_doc().unwrap().to_document())
            .unwrap();
    }
    w.add_fields_document(&Document::new()).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    assert_clean(&check_directory(dir).unwrap());
    test_util::check_index(dir).unwrap();
}

/// `testChecksumsOnlyVerbose(dir)`: the same; Java differs only in where
/// the (unread) report goes.
pub fn test_checksums_only_verbose(r: &mut JavaRandom, dir: &dyn Directory) {
    test_checksums_only(r, dir);
}

/// `testObtainsLock(dir)`: while a writer holds the index, the
/// lock-taking `CheckIndex` operation refuses.
pub fn test_obtains_lock(dir: &dyn Directory) {
    let mut w = writer(dir);
    w.add_fields_document(&Document::new()).unwrap();
    w.commit().unwrap();
    let results = check_directory(dir).unwrap();
    let err = exorcise(dir, &results).expect_err("the write lock is held");
    assert!(err.to_string().contains("lock"), "{err}");
    w.close().unwrap();
    // Released, it is the clean index's "no problems were detected".
    assert_eq!(exorcise(dir, &check_directory(dir).unwrap()).unwrap(), None);
}
