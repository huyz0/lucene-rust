//! `TermsEnum.termState()`/`seekExact(term, state)`: differential against
//! `fixtures/src/testframework/GenTestFrameworkTermState.java`'s
//! `term_state.txt` -- the same documents through this port's writer, every
//! term's state saved by one enum, and fresh enums seeked through the saved
//! states read back the term's statistics, postings and the term after it.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkTermState`.

use std::fs;
use std::path::PathBuf;

use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{LeafReader, PostingsFlags, TermState, TermsEnum, NO_MORE_DOCS};
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

fn fixture() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/term_state.txt");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn index(dir: &ByteBuffersDirectory) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    let mut r = JavaRandom::new(42);
    for _ in 0..300 {
        let mut body = String::new();
        let words = 1 + r.next_int_bounded(12);
        for _ in 0..words {
            let bound = 1 + r.next_int_bounded(300);
            body.push_str(&format!(" w{}", r.next_int_bounded(bound)));
        }
        let mut d = Document::new();
        d.add(TextField::new("body", body, Store::No));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    w.close().unwrap();
}

fn describe(te: &mut dyn TermsEnum, r: &mut JavaRandom) -> String {
    let term = String::from_utf8(te.term().unwrap().to_vec()).unwrap();
    let mut line = format!(
        "{term} {} {} |",
        te.doc_freq().unwrap(),
        te.total_term_freq().unwrap()
    );
    let mut pe = te.postings(PostingsFlags::Positions).unwrap();
    let mut doc = pe.next_doc().unwrap();
    while doc != NO_MORE_DOCS {
        line.push_str(&format!(" {doc}:{}:", pe.freq()));
        for p in 0..pe.freq() {
            let sep = if p == 0 { "" } else { "," };
            line.push_str(&format!("{sep}{}", pe.next_position().unwrap()));
        }
        doc = pe.next_doc().unwrap();
    }
    if r.next_boolean() {
        let next = te
            .next()
            .unwrap()
            .map(|t| String::from_utf8(t.to_vec()).unwrap());
        line.push_str(&format!(" next={}", next.as_deref().unwrap_or("null")));
    }
    line
}

#[test]
fn seeks_through_saved_term_states_match_lucene() {
    let dir = ByteBuffersDirectory::new();
    index(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let terms = reader.segment_readers()[0].terms("body").unwrap().unwrap();
    let mut saved: Vec<(Vec<u8>, TermState)> = Vec::new();
    let mut all = terms.iterator().unwrap();
    while let Some(t) = all.next().unwrap().map(<[u8]>::to_vec) {
        saved.push((t, all.term_state().unwrap()));
    }
    let mut r = JavaRandom::new(7);
    let mut actual = Vec::new();
    for _ in 0..400 {
        let idx = r.next_int_bounded(saved.len() as i32) as usize;
        let mut te = terms.iterator().unwrap();
        te.seek_exact_state(&saved[idx].0, &saved[idx].1).unwrap();
        actual.push(describe(te.as_mut(), &mut r));
    }
    let expected = fixture();
    let expected: Vec<&str> = expected.lines().collect();
    assert_eq!(expected.len(), actual.len());
    for (i, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(*want, got, "line {}", i + 1);
    }
}

/// A seek through a state taken after the enum moved on, a seek by term after
/// one through a state, and the defaults of an enum with no states of its
/// own: an empty state seeks by the term, and a missing term is refused.
#[test]
fn term_state_edges() {
    let dir = ByteBuffersDirectory::new();
    index(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    let terms = reader.segment_readers()[0].terms("body").unwrap().unwrap();
    let mut te = terms.iterator().unwrap();
    // Unpositioned: no state to take.
    assert!(te.term_state().is_err());
    assert!(te.try_seek_exact(b"w5").unwrap());
    let w5 = te.term_state().unwrap();
    let df = te.doc_freq().unwrap();
    te.next().unwrap();
    let mut other = terms.iterator().unwrap();
    other.seek_exact_state(b"w5", &w5).unwrap();
    assert_eq!(other.doc_freq().unwrap(), df);
    // A state of a pinned term is that term's.
    assert_eq!(other.term_state().unwrap(), w5);
    // A seek by term leaves the pinned state.
    assert!(other.try_seek_exact(b"w7").unwrap());
    assert_ne!(other.term_state().unwrap(), w5);
    // An empty state seeks by the term.
    other.seek_exact_state(b"w5", &TermState::empty()).unwrap();
    assert_eq!(other.term(), Some(&b"w5"[..]));
    assert_eq!(other.doc_freq().unwrap(), df);
    let err = other
        .seek_exact_state(b"absent", &TermState::empty())
        .unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
    // Postings without positions through a pinned state.
    other.seek_exact_state(b"w5", &w5).unwrap();
    let mut pe = other.postings(PostingsFlags::Freqs).unwrap();
    let mut n = 0;
    while pe.next_doc().unwrap() != NO_MORE_DOCS {
        n += 1;
    }
    assert_eq!(n, df);
    // `next()` after a state seek on an enum that read the dictionary goes
    // on from the pinned term; on a fresh one it starts from the first term.
    other.seek_exact_state(b"w5", &w5).unwrap();
    let after_w5 = other.next().unwrap().map(<[u8]>::to_vec);
    assert!(te.try_seek_exact(b"w5").unwrap());
    assert_eq!(te.next().unwrap().map(<[u8]>::to_vec), after_w5);
    let mut fresh = terms.iterator().unwrap();
    fresh.seek_exact_state(b"w5", &w5).unwrap();
    let first = terms
        .iterator()
        .unwrap()
        .next()
        .unwrap()
        .map(<[u8]>::to_vec);
    assert_eq!(fresh.next().unwrap().map(<[u8]>::to_vec), first);
    assert_eq!(TermState::ord(3).as_ord(), Some(3));
    assert_eq!(TermState::empty().as_ord(), None);
}
