//! `RandomIndexWriter`: differential against
//! `fixtures/src/testframework/GenTestFrameworkIndex.java`'s `riw.txt` (the
//! constructor's draws, and those of 300 adds, updates and deletes), and
//! what the writer leaves behind (every reader sees the documents a model
//! says are live; the index passes `CheckIndex`).
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkIndex`.

use std::collections::BTreeSet;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::{DeleteQuery, DocValuesUpdate, Term};
use lucene_index::document::{
    BinaryDocValuesField, Document, NumericDocValuesField, Store, StringField, TextField,
};
use lucene_index::merge_policy::api::NoMergePolicy;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::index::index_writer_config::{
    new_index_writer_config, Draws, LiveFlushMode, LiveMergePolicy, PolicyKind,
};
use lucene_test_framework::index::random_index_writer::{RandomIndexWriter, WriterEnv};
use lucene_test_framework::util::test_util::check_index;
use lucene_util::java_random::JavaRandom;

fn fixture() -> String {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data/riw.txt");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn doc(i: i32) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", i.to_string(), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("text {i} of {}", i % 13),
        Store::No,
    ));
    d
}

fn id(i: i32) -> Term {
    Term::new("id", i.to_string())
}

fn env(seed: i64, flush_mode: LiveFlushMode) -> WriterEnv {
    WriterEnv {
        random: JavaRandom::new(seed ^ 0x5eed),
        flush_mode,
        utc_offset_millis: 0,
    }
}

#[test]
fn the_writer_draws_what_java_draws() {
    let text = fixture();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 200);
    for pair in lines.chunks(2) {
        let riw: Vec<&str> = pair[0].split(' ').collect();
        let seed: i64 = riw[1].parse().unwrap();
        let mode = match riw[2] {
            "BY_RAM" => LiveFlushMode::ByRam,
            "BY_DOCS" => LiveFlushMode::ByDocs,
            _ => LiveFlushMode::Either,
        };
        let dir = ByteBuffersDirectory::new();
        let mut r = JavaRandom::new(seed);
        let mut w = RandomIndexWriter::new(&mut r, &dir, env(seed, mode)).unwrap();
        let actual = format!(
            "riw {seed} {} {} {} {} {}",
            riw[2],
            w.flush_at(),
            w.soft_deletes_ratio().to_bits() as i64,
            w.random_force_merge(),
            r.next_long()
        );
        assert_eq!(actual, pair[0], "seed {seed}");
        for i in 0..300 {
            if i % 7 == 3 {
                w.update_document(id(i / 2), &doc(i)).unwrap();
            } else if i % 11 == 5 {
                w.delete_documents(id(i / 3)).unwrap();
            } else if i % 17 == 9 {
                w.update_documents(id(i / 2), &[doc(i)]).unwrap();
            } else {
                w.add_document(&doc(i)).unwrap();
            }
        }
        let ops = format!(
            "ops {} {} {}",
            w.doc_count(),
            w.flush_at(),
            w.probe_stream()
        );
        assert_eq!(ops, pair[1], "seed {seed}");
        w.close().unwrap();
    }
}

fn live_ids(reader: &DirectoryReader) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for seg in reader.segment_readers() {
        for doc in 0..seg.max_doc {
            if seg.live_docs().is_none_or(|l| l.get(doc as usize)) {
                let stored = seg.stored_document(doc).unwrap().unwrap();
                let v = stored
                    .fields
                    .iter()
                    .find_map(|f| match &f.value {
                        lucene_codecs::stored_fields::FieldValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap();
                ids.insert(v);
            }
        }
    }
    ids
}

/// Random adds, updates, deletes, readers and commits over many seeds: every
/// reader shows what the model says is live, and the closed index passes
/// `CheckIndex` and reopens with the same documents.
#[test]
fn every_reader_sees_the_live_documents() {
    for seed in 0..40i64 {
        let dir = ByteBuffersDirectory::new();
        let mut r = JavaRandom::new(seed);
        let mode = LiveFlushMode::random(&mut JavaRandom::new(seed));
        let mut w = RandomIndexWriter::new(&mut r, &dir, env(seed, mode)).unwrap();
        let mut model = BTreeSet::new();
        let mut ops = JavaRandom::new(seed + 1000);
        for i in 0..400 {
            let k = ops.next_int_bounded(60);
            match ops.next_int_bounded(10) {
                0..=1 => {
                    w.update_document(id(k), &doc(k)).unwrap();
                    model.insert(k.to_string());
                }
                2 => {
                    w.delete_documents(id(k)).unwrap();
                    model.remove(&k.to_string());
                }
                3 => {
                    w.update_documents(id(k), &[doc(k)]).unwrap();
                    model.insert(k.to_string());
                }
                4 if i % 40 == 0 => {
                    let reader = w.get_reader().unwrap();
                    assert_eq!(live_ids(&reader), model, "seed {seed} op {i}");
                }
                5 if i % 50 == 0 => w.commit().unwrap(),
                _ => {
                    if model.insert(k.to_string()) {
                        w.add_document(&doc(k)).unwrap();
                    } else {
                        w.update_document(id(k), &doc(k)).unwrap();
                    }
                }
            }
        }
        let reader = w.get_reader_with(true, seed % 2 == 0).unwrap();
        assert_eq!(live_ids(&reader), model, "seed {seed}");
        drop(reader);
        w.close().unwrap();
        check_index(&dir).unwrap();
        assert_eq!(live_ids(&DirectoryReader::open(&dir).unwrap()), model);
        // A second writer over the same index resumes it.
        let mut r = JavaRandom::new(!seed);
        let mut w = RandomIndexWriter::new(&mut r, &dir, env(seed, mode)).unwrap();
        w.add_document(&doc(1000)).unwrap();
        model.insert("1000".into());
        let reader = w.get_reader().unwrap();
        assert_eq!(live_ids(&reader), model, "seed {seed} reopened");
        drop(reader);
        w.close().unwrap();
        check_index(&dir).unwrap();
    }
}

fn config(seed: i64) -> (JavaRandom, WriterEnv, Arc<Analyzer>) {
    let r = JavaRandom::new(seed);
    let analyzer = Arc::new(MockAnalyzer::new(&mut JavaRandom::new(seed)).build());
    (r, env(seed, LiveFlushMode::ByDocs), analyzer)
}

/// Soft deletes, when drawn, register the soft-deletes field (`___soft_deletes`);
/// the updates are hard ones (see the module doc), so a reader never sees
/// the old version.
#[test]
fn soft_deletes_hide_the_old_version() {
    let dir = ByteBuffersDirectory::new();
    let (mut r, mut e, analyzer) = config(7);
    let c = new_index_writer_config(&mut Draws::with_context(&mut r, &mut e.random), analyzer, 0);
    let mut w = RandomIndexWriter::with_soft_deletes(&mut r, &dir, c, true, e).unwrap();
    assert!(w.soft_deletes_ratio() >= 1.0);
    for i in 0..20 {
        w.add_document(&doc(i)).unwrap();
    }
    for i in 0..20 {
        w.update_document(id(i), &doc(i)).unwrap();
        w.update_documents(id(i), &[doc(i)]).unwrap();
    }
    w.commit_with(true).unwrap();
    let reader = w.get_reader().unwrap();
    assert_eq!(reader.num_docs(), 20);
    drop(reader);
    w.close().unwrap();
    check_index(&dir).unwrap();
    // A second soft-deletes writer over the index registers the field again.
    let (mut r, mut e, analyzer) = config(8);
    let c = new_index_writer_config(&mut Draws::with_context(&mut r, &mut e.random), analyzer, 0);
    let mut w = RandomIndexWriter::with_soft_deletes(&mut r, &dir, c, true, e).unwrap();
    w.update_document(id(0), &doc(0)).unwrap();
    assert_eq!(w.get_reader().unwrap().num_docs(), 20);
    w.close().unwrap();
    check_index(&dir).unwrap();
}

/// The rest of the surface: doc-values updates, deletes by query, merges,
/// `addIndexes`, `deleteAll`, and the writer itself.
#[test]
fn the_rest_of_the_writer_api() {
    let src = ByteBuffersDirectory::new();
    {
        let mut r = JavaRandom::new(3);
        let mut w = RandomIndexWriter::new(&mut r, &src, env(3, LiveFlushMode::Either)).unwrap();
        for i in 0..10 {
            w.add_document(&doc(100 + i)).unwrap();
        }
        w.close().unwrap();
    }
    let dir = ByteBuffersDirectory::new();
    let mut r = JavaRandom::new(4);
    let mut w = RandomIndexWriter::new(&mut r, &dir, env(4, LiveFlushMode::ByRam)).unwrap();
    for i in 0..30 {
        let mut d = doc(i);
        d.add(NumericDocValuesField::new("n", i64::from(i)));
        d.add(BinaryDocValuesField::new("b", vec![i as u8]));
        w.add_documents(&[d]).unwrap();
        if i % 10 == 9 {
            w.flush().unwrap();
        }
    }
    w.update_numeric_doc_value(id(1), "n", 100).unwrap();
    w.update_binary_doc_value(id(2), "b", b"x").unwrap();
    w.update_doc_values(
        id(3),
        &[DocValuesUpdate::Numeric {
            term: id(3),
            field: "n".into(),
            value: Some(7),
        }],
    )
    .unwrap();
    w.delete_documents_by_query(DeleteQuery::Term(id(4)))
        .unwrap();
    w.force_merge_deletes().unwrap();
    w.force_merge(2).unwrap();
    w.add_indexes(&[&src]).unwrap();
    assert!(w.writer().segment_count() > 0);
    let reader = DirectoryReader::open_nrt(w.nrt_source(), true, false).unwrap();
    assert_eq!(reader.num_docs(), 39);
    drop(reader);
    assert!(w.config().commit_on_close);
    w.delete_all().unwrap();
    w.commit().unwrap();
    let reader = w.get_reader().unwrap();
    assert_eq!(reader.num_docs(), 0);
    drop(reader);
    w.close().unwrap();
    check_index(&dir).unwrap();
}

/// Gate rule #10: the partial forced merge's limit check fires. A policy
/// that never merges (but is not `NoMergePolicy`, so random forced merges
/// stay on) leaves more segments than the limit.
#[test]
fn the_random_forced_merge_limit_check_is_seen_to_fire() {
    let dir = ByteBuffersDirectory::new();
    let (mut r, mut e, analyzer) = config(11);
    let mut c =
        new_index_writer_config(&mut Draws::with_context(&mut r, &mut e.random), analyzer, 0);
    c.merge_policy = LiveMergePolicy::new(PolicyKind::Other(Box::new(NoMergePolicy::default())));
    let mut w = RandomIndexWriter::with_soft_deletes(&mut r, &dir, c, false, e).unwrap();
    w.set_do_random_force_merge(true);
    let fired = catch_unwind(AssertUnwindSafe(|| {
        for i in 0..2000 {
            w.add_document(&doc(i)).unwrap();
            w.flush().unwrap();
            let _ = w.get_reader().unwrap();
        }
    }));
    let msg = fired.expect_err("the limit check never fired");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.starts_with("limit="), "{msg}");

    // Turned off, the same run passes.
    let dir = ByteBuffersDirectory::new();
    let (mut r, mut e, analyzer) = config(11);
    let mut c =
        new_index_writer_config(&mut Draws::with_context(&mut r, &mut e.random), analyzer, 0);
    c.merge_policy = LiveMergePolicy::new(PolicyKind::Other(Box::new(NoMergePolicy::default())));
    let mut w = RandomIndexWriter::with_soft_deletes(&mut r, &dir, c, false, e).unwrap();
    w.set_do_random_force_merge(true);
    w.set_do_random_force_merge_assert(false);
    for i in 0..300 {
        w.add_document(&doc(i)).unwrap();
        w.flush().unwrap();
        let _ = w.get_reader().unwrap();
    }
    w.close().unwrap();
}

/// `NoMergePolicy` turns random forced merges off, as Java's
/// `instanceof NoMergePolicy` does.
#[test]
fn no_merge_policy_turns_random_forced_merges_off() {
    for seed in 0..20 {
        let dir = ByteBuffersDirectory::new();
        let (mut r, mut e, analyzer) = config(seed);
        let mut c =
            new_index_writer_config(&mut Draws::with_context(&mut r, &mut e.random), analyzer, 0);
        c.merge_policy = LiveMergePolicy::new(PolicyKind::NoMerge(NoMergePolicy::default()));
        let w = RandomIndexWriter::with_config(&mut r, &dir, c, e).unwrap();
        assert!(!w.random_force_merge());
        w.close().unwrap();
    }
}
