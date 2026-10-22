//! M14.h h5's and h6's acceptance run: each format suite's
//! `addRandomFields` (postings, stored fields, term vectors, doc values,
//! segment infos, field infos, compound, norms) fills a fresh
//! `FsDirectory` per seed through a `RandomIndexWriter` under a random
//! `newIndexWriterConfig`, with deletes, commits and sometimes a forced
//! merge. The closed index must pass this port's `CheckIndex`, and real
//! Lucene's with `--java-cp` (`org.apache.lucene.index.CheckIndex`, which
//! decodes every posting, stored document, term vector, doc value and
//! norm, and every deleted document's `.liv`),
//! having opened every segment the port wrote.
//!
//! Usage: `format_suites [--seeds A..B | --seed N] [--java-cp CP] [--dir DIR]`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField};
use lucene_store::FsDirectory;
use lucene_test_framework::index::base_compound_format::CompoundSuite;
use lucene_test_framework::index::base_field_info_format::FieldInfoSuite;
use lucene_test_framework::index::base_index_file_format::{new_riw, IndexFileFormatSuite};
use lucene_test_framework::index::base_norms_format::NormsSuite;
use lucene_test_framework::index::base_postings_format::PostingsSuite;
use lucene_test_framework::index::base_segment_info_format::SegmentInfoSuite;
use lucene_test_framework::index::base_stored_fields_format::StoredFieldsSuite;
use lucene_test_framework::index::base_term_vectors_format::TermVectorsSuite;
use lucene_test_framework::index::legacy_base_doc_values_format::DocValuesSuite;
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util;

const SUITES: [(&str, &dyn IndexFileFormatSuite); 8] = [
    ("postings", &PostingsSuite),
    ("stored fields", &StoredFieldsSuite),
    ("term vectors", &TermVectorsSuite),
    ("doc values", &DocValuesSuite),
    ("segment infos", &SegmentInfoSuite),
    ("field infos", &FieldInfoSuite),
    ("compound", &CompoundSuite),
    ("norms", &NormsSuite),
];

/// Port and (with `java_cp`) Lucene `CheckIndex` over `path`; the number of
/// segments checked.
fn check(dir: &FsDirectory, path: &Path, java_cp: Option<&str>) -> Result<usize, String> {
    test_util::check_index(dir)?;
    let ours = lucene_index::segment_infos::read_latest(dir)
        .map_err(|e| e.to_string())?
        .segments
        .len();
    if let Some(cp) = java_cp {
        let out = std::process::Command::new("java")
            .args(["-cp", cp, "org.apache.lucene.index.CheckIndex"])
            .arg(path)
            .output()
            .map_err(|e| format!("java: {e}"))?;
        let text = String::from_utf8_lossy(&out.stdout);
        // Gate rule #10: an exit status alone would pass over an index Java
        // never opened; it must have checked every segment the port wrote.
        let theirs = text
            .split_whitespace()
            .find_map(|w| w.strip_prefix("numSegments="))
            .and_then(|n| n.parse::<usize>().ok());
        if !out.status.success()
            || !text.contains("No problems were detected")
            || theirs != Some(ours)
        {
            // From Java's first failing check (its stack trace follows),
            // or its last lines when nothing says which check failed.
            let lines: Vec<&str> = text.lines().collect();
            let from = lines
                .iter()
                .position(|l| l.contains("FAILED") || l.contains("ERROR"))
                .unwrap_or(lines.len().saturating_sub(15));
            let shown = &lines[from..lines.len().min(from + 20)];
            return Err(format!(
                "Lucene CheckIndex failed (numSegments {theirs:?}, ours {ours}):\n{}",
                shown.join("\n")
            ));
        }
    }
    Ok(ours)
}

/// One suite's index for `seed`: documents, deletes, commits, maybe a
/// forced merge; then the checks. The segments checked.
fn round(
    suite: &dyn IndexFileFormatSuite,
    name: &str,
    seed: i64,
    base: &Path,
    java_cp: Option<&str>,
) -> Result<usize, String> {
    let path = base.join(format!("{}-{seed}", name.replace(' ', "-")));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    let dir = FsDirectory::open(&path);
    let mut t = LuceneTestCase::with_config(name, seed, false, 1);
    {
        let mut w = new_riw(&mut t, &dir);
        let num_docs = t.at_least(200);
        for i in 0..num_docs {
            let mut doc = Document::new();
            doc.add(StringField::new("id", i.to_string(), Store::No));
            suite.add_random_fields(&mut t, &mut doc);
            w.add_document(&doc).map_err(|e| e.to_string())?;
            if t.random().next_int_bounded(50) == 0 {
                w.commit().map_err(|e| e.to_string())?;
            }
            if t.random().next_int_bounded(10) == 0 {
                let victim = t.random().next_int_bounded(i + 1);
                w.delete_documents(Term::new("id", victim.to_string()))
                    .map_err(|e| e.to_string())?;
            }
        }
        if t.random().next_boolean() {
            w.force_merge(1).map_err(|e| e.to_string())?;
        }
        w.commit().map_err(|e| e.to_string())?;
        w.close().map_err(|e| e.to_string())?;
    }
    let segments = check(&dir, &path, java_cp)?;
    drop(dir);
    let _ = std::fs::remove_dir_all(&path);
    Ok(segments)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let (first, end): (i64, i64) = match (value("--seed"), value("--seeds")) {
        (Some(s), _) => {
            let s: i64 = s.parse().expect("--seed");
            (s, s + 1)
        }
        (None, Some(range)) => {
            let (a, b) = range.split_once("..").expect("--seeds A..B");
            (a.parse().expect("seed"), b.parse().expect("seed"))
        }
        (None, None) => (0, 30),
    };
    assert!(first < end, "--seeds A..B needs A < B");
    let java_cp = value("--java-cp");
    let base = value("--dir").map_or_else(
        || std::env::temp_dir().join(format!("format-suites-{}", std::process::id())),
        PathBuf::from,
    );
    let started = Instant::now();
    let (mut indexes, mut segments) = (0usize, 0usize);
    for (name, suite) in SUITES {
        for seed in first..end {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                round(suite, name, seed, &base, java_cp)
            }))
            .unwrap_or_else(|panic| {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Err(format!("panicked: {text}"))
            });
            match result {
                Ok(n) => {
                    println!("{name} seed {seed}: ok -- {n} segment(s)");
                    indexes += 1;
                    segments += n;
                }
                Err(why) => {
                    println!("{name} seed {seed}: FAIL -- {why}");
                    println!("replay: format_suites --seed {seed}");
                    std::process::exit(1);
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&base);
    println!(
        "format_suites: {indexes} index(es), {segments} segment(s) checked{} in {:.1}s",
        if java_cp.is_some() {
            " by Lucene too"
        } else {
            ""
        },
        started.elapsed().as_secs_f64()
    );
}
