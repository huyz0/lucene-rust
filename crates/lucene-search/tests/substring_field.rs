//! BINARY doc values in the `RustSubstring` format: flushed, deleted from and
//! force-merged, searched by substring, wildcard and regexp, case sensitive
//! and not -- each search's documents equal a scan of every value and the
//! same search over a plain BINARY field of the same documents.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values::Lucene90DocValuesFormat;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{BinaryDocValuesField, Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{SubstringKind, SubstringQuery};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::reader::LeafReader;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DOCS: usize = 3000;

const WORDS: &[&str] = &[
    "Connection",
    "timeout",
    "java.net.SocketTimeoutException:",
    "read",
    "timed",
    "out",
    "ERROR",
    "error",
    "WARN",
    "disk",
    "full",
    "blk_-16089",
    "10.250.19.102",
    "postdrop",
    "İstanbul",
    "\u{212A}elvin",
    "parity",
    "corrected",
    "",
    "a",
    "ab",
];

fn noise(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x >> 16
    }
}

/// Document `i`'s value; one in eleven has none.
fn value(i: usize) -> Option<String> {
    if i % 11 == 3 {
        return None;
    }
    let mut r = noise(i as u64);
    let n = (r() % 8) as usize;
    let words: Vec<&str> = (0..n)
        .map(|_| WORDS[(r() % WORDS.len() as u64) as usize])
        .collect();
    Some(format!("{} {}", i % 97, words.join(" ")))
}

fn write(dir: &std::path::Path, substring: bool) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    if substring {
        w.set_doc_values_format_for_field("msg", Lucene90DocValuesFormat::rust_substring(3));
    }
    for i in 0..DOCS {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        if let Some(v) = value(i) {
            doc.add(BinaryDocValuesField::new("msg", v.into_bytes()));
        }
        w.add_fields_document(&doc).unwrap();
        if (i + 1) % 700 == 0 {
            w.commit().unwrap();
        }
    }
    for i in (0..DOCS).step_by(13) {
        w.delete_documents_by_term(&[Term::new("id", i.to_string())])
            .unwrap();
    }
    w.commit().unwrap();
}

fn force_merge(dir: &std::path::Path, substring: bool) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    if substring {
        w.set_doc_values_format_for_field("msg", Lucene90DocValuesFormat::rust_substring(3));
    }
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn queries() -> Vec<SubstringQuery> {
    let q = |k, p: &str, ci| SubstringQuery::new("msg", k, p, ci).unwrap();
    use SubstringKind::*;
    vec![
        q(Contains, "timeout", false),
        q(Contains, "timeout", true),
        q(Contains, "TIMEOUT", true),
        q(Contains, "Timeout", false),
        q(Contains, "error", true),
        q(Contains, "blk_-16089", false),
        q(Contains, "10.250.19.102", false),
        q(Contains, "istanbul", true),
        q(Contains, "kelvin", true),
        q(Contains, "zzz", false),
        q(Contains, "ab", false),
        q(Wildcard, "*", false),
        q(Wildcard, "1 *", false),
        q(Wildcard, "*corrected", false),
        q(Wildcard, "4? *out*", false),
        q(Wildcard, "*conn*time*", true),
        q(Exact, "5 ", false),
        q(Regexp, ".*[Tt]ime(d|out).*", false),
        q(Regexp, "1[0-9] .*error.*", true),
    ]
}

/// The global doc ids of `q`'s matches, ascending.
fn hits(searcher: &IndexSearcher<'_, '_>, q: &SubstringQuery) -> Vec<i32> {
    let query = BooleanQuery {
        must: vec![Clause::from(q.clone())],
        ..BooleanQuery::default()
    };
    let top = searcher.search(&query, DOCS).unwrap();
    let mut docs: Vec<i32> = top.score_docs.iter().map(|d| d.doc).collect();
    docs.sort_unstable();
    assert_eq!(searcher.count(&query).unwrap() as usize, docs.len());
    docs
}

/// Every search of [`queries`] over the index in `dir`: its hits, the hits a
/// scan of the field's values finds, and the hits' values, sorted (two
/// indexes of the same documents may merge them in another order).
fn search(dir: &std::path::Path) -> Vec<(Vec<i32>, Vec<i32>, Vec<Vec<u8>>)> {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    // Every live value, by global doc id, read back through BinaryDocValues.
    let mut values: Vec<(i32, Vec<u8>)> = Vec::new();
    for s in &segments {
        let r = s.reader.expect("a segment reader");
        let Some(mut dv) = r.binary_doc_values("msg").unwrap() else {
            continue;
        };
        while dv.next_doc().unwrap() != i32::MAX {
            let d = dv.doc_id();
            if r.live_docs().is_none_or(|l| l.get(d as usize)) {
                values.push((s.doc_base + d, dv.binary_value().to_vec()));
            }
        }
    }
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    for q in queries() {
        let want: Vec<i32> = values
            .iter()
            .filter(|(_, v)| q.matcher().matches(v, &mut scratch))
            .map(|(d, _)| *d)
            .collect();
        let got = hits(&searcher, &q);
        let mut got_values: Vec<Vec<u8>> = got
            .iter()
            .map(|d| {
                let k = values.binary_search_by_key(d, |(doc, _)| *doc).unwrap();
                values[k].1.clone()
            })
            .collect();
        got_values.sort_unstable();
        out.push((got, want, got_values));
    }
    out
}

#[test]
fn substring_searches_match_a_scan_and_plain_binary_values() {
    let sub = TempDir::new("substring-field");
    let plain = TempDir::new("substring-field-plain");
    write(sub.path(), true);
    write(plain.path(), false);
    for round in 0..2 {
        let a = search(sub.path());
        let b = search(plain.path());
        for (k, ((got, want, values), (plain_got, plain_want, plain_values))) in
            a.iter().zip(&b).enumerate()
        {
            assert_eq!(got, want, "query {k} round {round}");
            assert_eq!(plain_got, plain_want, "plain query {k} round {round}");
            assert_eq!(
                values, plain_values,
                "query {k} round {round} against plain"
            );
        }
        assert!(a.iter().any(|(g, _, _)| g.len() > 100));
        assert!(a.iter().any(|(g, _, _)| !g.is_empty() && g.len() < 50));
        force_merge(sub.path(), true);
        force_merge(plain.path(), false);
    }
    // The merged segment is still in the substring format.
    let names: Vec<String> = std::fs::read_dir(sub.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("RustSubstring")),
        "{names:?}"
    );
}

#[test]
fn a_malformed_regexp_is_an_error() {
    assert!(SubstringQuery::new("msg", SubstringKind::Regexp, "(", false).is_err());
}
