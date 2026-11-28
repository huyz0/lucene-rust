//! The hot tier against the writer without it: an engine's operation stream
//! (index, delete, refresh, commit) must leave the same live documents.
#![allow(clippy::arithmetic_side_effects)] // Test arithmetic on small constants.

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::hot_tier::{HistoryStub, HotTierConfig};
use lucene_index::index_writer::{ExplicitDocument, IndexWriter, InvertedField, InvertedTerm};
use lucene_index::nrt::NrtSource;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

struct Ids {
    id: i32,
    soft: i32,
    tombstone: i32,
    source: i32,
    seq: i32,
    term: i32,
    version: i32,
}

fn numeric(name: &str) -> FieldInfo {
    FieldInfo::new(name, 0).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    )
}

/// A writer with the fields an engine registers, as the engine does.
fn open(dir: &FsDirectory, hot: bool) -> (ConcurrentIndexWriter<'_>, Ids) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let id = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    let soft = w
        .register_field(numeric("__soft_deletes").with_soft_deletes_field(true))
        .unwrap();
    let seq = w.register_field(numeric("_seq_no")).unwrap();
    let term = w.register_field(numeric("_primary_term")).unwrap();
    let version = w.register_field(numeric("_version")).unwrap();
    let tombstone = w.register_field(numeric("_tombstone")).unwrap();
    let source = w.register_field(FieldInfo::new("_source", 0)).unwrap();
    let w = ConcurrentIndexWriter::new(w, 2).unwrap();
    if hot {
        w.enable_hot_tier(
            HotTierConfig::with_budget(1 << 20).with_history(HistoryStub {
                copy: vec!["_seq_no".to_string(), "_primary_term".to_string()],
                constants: vec![("_version".to_string(), 1), ("_tombstone".to_string(), 1)],
                source: Some(("_source".to_string(), b"superseded".to_vec())),
            }),
        );
    }
    (
        w,
        Ids {
            id,
            soft,
            tombstone,
            source,
            seq,
            term,
            version,
        },
    )
}

fn term_of(i: u32) -> Term {
    Term::new("_id", format!("d{i}").into_bytes())
}

/// An index operation of row `i`, or its delete tombstone (born soft-deleted
/// as an engine writes it).
fn document(f: &Ids, i: u32, seq: i64, tombstone: bool) -> ExplicitDocument {
    let mut d = ExplicitDocument::default();
    let name = format!("d{i}");
    d.stored.push(StoredField {
        field_number: f.id,
        value: FieldValue::String(name.clone()),
    });
    d.fields.inverted.push(InvertedField {
        field_number: f.id,
        terms: vec![InvertedTerm {
            term: name.into_bytes(),
            freq: 1,
            positions: Vec::new(),
            offsets: Vec::new(),
            payloads: Vec::new(),
        }]
        .into(),
        norm: None,
    });
    let mut dv = vec![(f.seq, seq), (f.term, 1), (f.version, seq + 100)];
    if tombstone {
        dv.push((f.soft, 1));
        dv.push((f.tombstone, 1));
    }
    d.stored.push(StoredField {
        field_number: f.source,
        value: FieldValue::Binary(b"{}".to_vec()),
    });
    for (n, v) in dv {
        d.fields.doc_values.push(StoredField {
            field_number: n,
            value: FieldValue::Long(v),
        });
    }
    d
}

fn soft_delete(i: u32) -> DocValuesUpdate {
    DocValuesUpdate::Numeric {
        term: term_of(i),
        field: "__soft_deletes".to_string(),
        value: Some(1),
    }
}

fn apply(w: &ConcurrentIndexWriter<'_>, f: &Ids, i: u32, seq: i64, tombstone: bool) {
    w.soft_update_explicit_documents(
        term_of(i),
        vec![document(f, i, seq, tombstone)],
        &[soft_delete(i)],
    )
    .unwrap();
}

/// The documents a commit makes searchable: neither deleted nor soft-deleted.
fn live_docs(dir: &FsDirectory) -> i64 {
    let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
    infos
        .segments
        .iter()
        .map(|sci| {
            let si = lucene_index::segment_info::parse(
                &lucene_store::directory::Directory::open(dir, &format!("{}.si", sci.segment_name))
                    .unwrap(),
                &sci.segment_id,
            )
            .unwrap();
            i64::from(si.doc_count) - i64::from(sci.del_count) - i64::from(sci.soft_del_count)
        })
        .sum()
}

/// A small xorshift, so a failing stream is the same stream on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self, below: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % below
    }
}

/// One stream through a writer; returns the live documents after a final
/// commit and the number of rows the stream left live.
fn run(seed: u64, hot: bool) -> (i64, i64) {
    let tmp = TempDir::new("hot-tier-updates");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, hot);
    let mut rng = Rng(seed | 1);
    let mut live = [false; 6];
    for seq in 0..600i64 {
        let i = rng.next(6) as u32;
        match rng.next(20) {
            0 => {
                w.nrt_snapshot(true, true).unwrap();
            }
            1 => {
                w.commit().unwrap();
            }
            2..=5 => {
                apply(&w, &f, i, seq, true);
                live[i as usize] = false;
            }
            _ if !live[i as usize] && rng.next(2) == 0 => {
                // An index of a deleted row is an add, as the engine does it.
                w.add_explicit_documents(vec![document(&f, i, seq, false)])
                    .unwrap();
                live[i as usize] = true;
            }
            _ => {
                apply(&w, &f, i, seq, false);
                live[i as usize] = true;
            }
        }
    }
    w.commit().unwrap();
    let expected = live.iter().filter(|l| **l).count() as i64;
    if hot {
        assert!(
            w.hot_tier_stats().unwrap().coalesced > 0,
            "the stream never held a row"
        );
    }
    drop(w);
    (live_docs(&dir), expected)
}

#[test]
fn a_hot_row_stream_leaves_the_live_documents_a_writer_without_the_tier_does() {
    for seed in 1..12u64 {
        let (off, expected) = run(seed, false);
        assert_eq!(off, expected, "seed {seed}: without the tier");
        let (on, _) = run(seed, true);
        assert_eq!(on, expected, "seed {seed}: with the tier");
    }
}

/// A refresh asks whether the reader is current: an update the tier holds is
/// not in it.
#[test]
fn a_held_update_makes_the_reader_not_current() {
    let tmp = TempDir::new("hot-tier-current");
    let dir = FsDirectory::open(tmp.path());
    let (w, f) = open(&dir, true);
    for seq in 0..8 {
        apply(&w, &f, 1, seq, false);
    }
    let snapshot = w.nrt_snapshot(true, true).unwrap();
    assert!(w.nrt_is_current(&snapshot.segment_infos).unwrap());
    apply(&w, &f, 1, 8, false);
    assert_eq!(w.hot_tier_stats().unwrap().held_rows, 1);
    assert_eq!(w.pending_doc_count(), 0);
    assert!(
        !w.nrt_is_current(&snapshot.segment_infos).unwrap(),
        "a held update is invisible to the reader"
    );
}

#[test]
fn an_add_after_a_held_delete_stays_live() {
    for n in 1..8 {
        let tmp = TempDir::new("hot-tier-delete-index");
        let dir = FsDirectory::open(tmp.path());
        let (w, f) = open(&dir, true);
        let mut seq = 0;
        for _ in 0..n {
            apply(&w, &f, 1, seq, false);
            seq += 1;
        }
        apply(&w, &f, 1, seq, true);
        // The engine knows the row is deleted, so it adds rather than updates.
        w.add_explicit_documents(vec![document(&f, 1, seq + 1, false)])
            .unwrap();
        w.commit().unwrap();
        drop(w);
        assert_eq!(live_docs(&dir), 1, "{n} updates, a delete, an index");
    }
}
