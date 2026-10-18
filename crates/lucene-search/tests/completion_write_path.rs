//! M13 T13.3d through [`IndexWriter`]: `GenCompletion`'s documents as
//! [`SuggestField`]s on a field routed to `Completion104`, flushed (one
//! segment) and force-merged (from three of falling sizes), must leave `.lkp`/`.cmp` whose
//! bodies are Lucene's, with the delegate's files under the same suffix, and
//! answer every `lookups.tsv` prefix as Lucene does through
//! [`SegmentReader::suggester`]. `scripts/verify-write-path.sh`'s
//! `completion` case has real Lucene read the same index.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_index::document::Document;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::suggest::document::{lookup, SegmentView, TopSuggestDocsCollector};
use lucene_search::suggest::suggest_field::SuggestField;
use lucene_store::FsDirectory;
use lucene_util::automaton::automata::make_binary;
use lucene_util::test_support::TempDir;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/completion")
}

fn docs() -> Vec<(String, i32)> {
    fs::read_to_string(fixtures().join("docs.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c[1].to_string(), c[2].parse().unwrap())
        })
        .collect()
}

/// Writes `docs.tsv`, committing after each document count in `commits`,
/// then merges whatever segments that left into one. Sizes that fall from
/// segment to segment keep the documents in order through the merge: the
/// tiered policy merges the largest segment first, as Java's does.
fn write(dir: &Path, commits: &[usize]) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::keyword())));
    w.set_use_compound_file(false);
    w.set_postings_format_for_field("suggest", Lucene104PostingsFormat::completion104());
    for (i, (surface, weight)) in docs().iter().enumerate() {
        let mut doc = Document::new();
        doc.add_boxed(Box::new(
            SuggestField::new("suggest", surface.as_str(), *weight).unwrap(),
        ));
        w.add_fields_document(&doc).unwrap();
        if commits.contains(&(i + 1)) {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    if !commits.is_empty() {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

/// `bytes` with its index header's segment id zeroed and its checksum
/// dropped: what two writers of the same content under different ids share.
fn body(bytes: &[u8]) -> Vec<u8> {
    // magic, codec name "Completion104" (1 + 13), version, then the id
    let id_at = 4 + 14 + 4;
    let mut out = bytes[..bytes.len() - 8].to_vec();
    out[id_at..id_at + 16].fill(0);
    out
}

/// `dir`'s one segment against the fixture files `<fixture>.lkp`/`.cmp`,
/// and, when its documents are in `docs.tsv` order, against `lookups.tsv`.
fn check(dir: &Path, fixture: &str, in_order: bool) {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let segments = reader.segment_readers();
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.starts_with(&format!("{}_", segment.segment_name)))
        .collect();
    names.sort();
    let suffix = fs::read_to_string(fixtures().join("suffix.txt")).unwrap();
    let stem = format!("{}_{}", segment.segment_name, suffix.trim());
    for ext in ["cmp", "doc", "lkp", "pos", "psm", "tim", "tip", "tmd"] {
        assert!(
            names.contains(&format!("{stem}.{ext}")),
            "{stem}.{ext} in {names:?}"
        );
    }
    for ext in ["lkp", "cmp"] {
        let ours = fs::read(dir.join(format!("{stem}.{ext}"))).unwrap();
        let lucene = fs::read(fixtures().join(format!("{fixture}.{ext}"))).unwrap();
        assert_eq!(body(&ours), body(&lucene), "{ext}");
    }

    let suggester = segment.suggester("suggest").unwrap().unwrap();
    assert!(segment.suggester("absent").unwrap().is_none());
    if !in_order {
        return;
    }
    let max_doc = segment.max_doc;
    let accept = |_: i32| true;
    let view = SegmentView {
        num_docs: max_doc,
        max_doc,
        filtered: false,
        accept: &accept,
    };
    let mut checked = 0;
    for line in fs::read_to_string(fixtures().join("lookups.tsv"))
        .unwrap()
        .lines()
    {
        let c: Vec<&str> = line.split('\t').collect();
        let (prefix, n, dedup) = (c[0], c[1].parse::<usize>().unwrap(), c[2] == "1");
        let mut collector = TopSuggestDocsCollector::new(n, dedup).unwrap();
        lookup(
            &suggester,
            &make_binary(prefix.as_bytes()),
            &view,
            &mut collector,
        )
        .unwrap();
        collector.finish();
        let got = collector
            .get()
            .iter()
            .map(|h| format!("{}\u{1e}{}\u{1e}{}", h.key, h.doc, h.score as i64))
            .collect::<Vec<_>>()
            .join("\u{1f}");
        assert_eq!(
            got,
            c.get(3).copied().unwrap_or(""),
            "{prefix:?} n={n} dedup={dedup}"
        );
        checked += 1;
    }
    assert_eq!(checked, 180);
}

#[test]
fn a_flushed_completion_field_is_lucenes() {
    let tmp = TempDir::new("completion-flush");
    write(tmp.path(), &[]);
    check(tmp.path(), "suggest", true);
}

#[test]
fn a_merged_completion_field_is_rebuilt_as_lucene_builds_it() {
    let tmp = TempDir::new("completion-merge");
    write(tmp.path(), &[60, 90]);
    // The `.lkp` stores doc ids, so equal bytes with Lucene's merge of the
    // same commits pin the merged document order (largest segment first);
    // the lookups, over the flushed order, are not repeated.
    check(tmp.path(), "merged", false);
}

#[test]
fn a_completion_instance_without_its_lkp_is_corrupt() {
    let tmp = TempDir::new("completion-no-lkp");
    write(tmp.path(), &[]);
    let si_path = tmp.path().join("_0.si");
    let bytes = fs::read(&si_path).unwrap();
    // magic, codec name "Lucene90SegmentInfo" (1 + 19), version, then the id
    let id: [u8; 16] = bytes[28..44].try_into().unwrap();
    let mut si = segment_info::parse(&bytes, &id).unwrap();
    si.files.retain(|f| !f.ends_with(".lkp"));
    fs::write(&si_path, segment_info::write(&si, "")).unwrap();
    let Err(err) = DirectoryReader::open(&FsDirectory::open(tmp.path())) else {
        panic!("a .cmp without its .lkp must not open");
    };
    let err = err.to_string();
    assert!(err.contains(".lkp"), "{err}");
}
