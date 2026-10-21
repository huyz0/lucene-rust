//! M14.h's fault run: an `IndexWriter` over a [`MockDirectoryWrapper`]
//! failing 10% of its writes and opens, as Lucene's
//! `TestIndexWriterExceptions.testRandomIOExceptions` drives one.
//!
//! Each seed runs a stream of adds and commits (and flushes, from a small
//! buffered-document limit) over a fresh `FSDirectory`. An error drops the
//! writer, which is Java's rollback, and the next operation opens a new
//! one. When the stream ends the rates go to 0 and the mock directory
//! closes, which crashes it (unsynced files lost or damaged) and runs this
//! port's `CheckIndex`. Then, over the bare directory:
//!
//! 1. no operation panicked, and every error was an injected one;
//! 2. the index holds a document count some commit could have left: the
//!    last one that succeeded, or a later one that failed after publishing;
//! 3. this port's `CheckIndex` passes, and real Lucene's too with
//!    `--java-cp` (`org.apache.lucene.index.CheckIndex`);
//! 4. a new writer appends to it and commits.
//!
//! Usage: `io_faults [--seeds A..B | --seed N] [--ops N] [--java-cp CP] [--dir DIR]`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::store::base_directory_wrapper::{index_exists, BaseDirectoryWrapper};
use lucene_test_framework::store::base_lock_factory::{add_doc, open_writer};
use lucene_test_framework::store::mock_directory_wrapper::{
    is_injected_io_exception, MockDirectoryWrapper,
};
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

/// What one round saw, for the run's evidence check.
#[derive(Default)]
struct Round {
    injected: usize,
    commits_ok: usize,
    commits_failed: usize,
    docs: usize,
}

/// An error from the stream: injected (counted) or a failure.
fn injected(e: &dyn std::fmt::Display, round: &mut Round) -> Result<(), String> {
    let text = e.to_string();
    if is_injected_io_exception(&text) {
        round.injected += 1;
        Ok(())
    } else {
        Err(format!("an error nothing injected: {text}"))
    }
}

/// `addDoc`'s document, `content:aaa`.
fn doc() -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("aaa".to_string()),
        }],
    }
}

fn run_stream(
    dir: &MockDirectoryWrapper,
    rng: &mut JavaRandom,
    num_ops: usize,
    round: &mut Round,
) -> Result<BTreeSet<usize>, String> {
    // The document counts a commit may have left on disk.
    let mut possible = BTreeSet::from([0usize]);
    let mut writer: Option<(IndexWriter<'_>, usize, usize)> = None;
    for _ in 0..num_ops {
        let Some((w, base, added)) = writer.as_mut() else {
            let opened = open_writer(dir)
                .map_err(|e| e.to_string())
                .and_then(|mut w| {
                    w.set_max_buffered_docs(2 + rng.next_int_bounded(8))
                        .map_err(|e| e.to_string())?;
                    let base = w.committed_doc_count().map_err(|e| e.to_string())?;
                    Ok((w, base))
                });
            match opened {
                Ok((w, base)) => {
                    if !possible.contains(&base) {
                        return Err(format!(
                            "a writer opened {base} docs; commits could leave {possible:?}"
                        ));
                    }
                    possible = BTreeSet::from([base]);
                    writer = Some((w, base, 0));
                }
                Err(e) => injected(&e, round)?,
            }
            continue;
        };
        if rng.next_int_bounded(8) == 0 {
            let target = *base + *added;
            let committed = w.commit().map(drop);
            match committed {
                Ok(_) => {
                    round.commits_ok += 1;
                    possible = BTreeSet::from([target]);
                    *base = target;
                    *added = 0;
                }
                Err(e) => {
                    round.commits_failed += 1;
                    injected(&e, round)?;
                    // A commit can fail after publishing `segments_N`.
                    possible.insert(target);
                    writer = None;
                }
            }
        } else {
            match w.add_document(doc()) {
                Ok(_) => *added += 1,
                Err(e) => {
                    injected(&e, round)?;
                    writer = None;
                }
            }
        }
    }
    drop(writer);
    Ok(possible)
}

fn check(dir: &dyn Directory, path: &Path, java_cp: Option<&str>) -> Result<(), String> {
    test_util::check_index(dir)?;
    if let Some(cp) = java_cp {
        let out = std::process::Command::new("java")
            .args(["-cp", cp, "org.apache.lucene.index.CheckIndex"])
            .arg(path)
            .output()
            .map_err(|e| format!("java: {e}"))?;
        if !out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let tail: Vec<&str> = text.lines().rev().take(15).collect();
            return Err(format!(
                "Lucene CheckIndex failed:\n{}",
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
    }
    Ok(())
}

fn round(seed: u64, base: &Path, num_ops: usize, java_cp: Option<&str>) -> Result<Round, String> {
    let path = base.join(format!("seed-{seed}"));
    let _ = std::fs::remove_dir_all(&path);
    let mut rng = JavaRandom::new(seed as i64);
    let mut round = Round::default();
    let dir = MockDirectoryWrapper::new(&mut rng, FsDirectory::open(&path));
    dir.set_random_io_exception_rate(0.1);
    dir.set_random_io_exception_rate_on_open(0.1);
    let possible = run_stream(&dir, &mut rng, num_ops, &mut round)?;
    dir.set_random_io_exception_rate(0.0);
    dir.set_random_io_exception_rate_on_open(0.0);
    dir.set_check_index_on_close(true);
    dir.close()
        .map_err(|e| format!("close (crash + CheckIndex): {e}"))?;
    drop(dir);

    let bare = FsDirectory::open(&path);
    if index_exists(&bare).map_err(|e| e.to_string())? {
        let docs = DirectoryReader::open(&bare)
            .map_err(|e| e.to_string())?
            .num_docs();
        let docs = usize::try_from(docs).map_err(|e| e.to_string())?;
        if !possible.contains(&docs) {
            return Err(format!(
                "the index holds {docs} docs; commits could leave {possible:?}"
            ));
        }
        round.docs = docs;
        check(&bare, &path, java_cp)?;
    } else if !possible.contains(&0) {
        return Err(format!("no index; commits could leave {possible:?}"));
    }
    let mut w = open_writer(&bare).map_err(|e| format!("recovery open: {e}"))?;
    add_doc(&mut w);
    w.commit().map_err(|e| format!("recovery commit: {e}"))?;
    drop(w);
    test_util::check_index(&bare).map_err(|e| format!("after recovery: {e}"))?;
    let _ = std::fs::remove_dir_all(&path);
    Ok(round)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let num_ops: usize = value("--ops").map_or(200, |v| v.parse().expect("--ops"));
    let (first, end) = match (value("--seed"), value("--seeds")) {
        (Some(s), _) => {
            let s: u64 = s.parse().expect("--seed");
            (s, s + 1)
        }
        (None, Some(range)) => {
            let (a, b) = range.split_once("..").expect("--seeds A..B");
            (a.parse().expect("seed"), b.parse().expect("seed"))
        }
        (None, None) => (0, 50),
    };
    assert!(first < end, "--seeds A..B needs A < B");
    let java_cp = value("--java-cp");
    let base = value("--dir").map_or_else(
        || std::env::temp_dir().join(format!("io-faults-{}", std::process::id())),
        PathBuf::from,
    );
    // A panic is reported as this round's failure, not as a crash of the run.
    std::panic::set_hook(Box::new(|_| {}));

    let started = Instant::now();
    let mut total = Round::default();
    let mut rounds = 0usize;
    for seed in first..end {
        let result = std::panic::catch_unwind(|| round(seed, &base, num_ops, java_cp))
            .unwrap_or_else(|panic| {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Err(format!("panicked: {text}"))
            });
        match result {
            Ok(r) => {
                println!(
                    "seed {seed}: ok -- {} injected, {} commits ok, {} failed, {} docs",
                    r.injected, r.commits_ok, r.commits_failed, r.docs
                );
                total.injected += r.injected;
                total.commits_ok += r.commits_ok;
                total.commits_failed += r.commits_failed;
                total.docs += r.docs;
            }
            Err(why) => {
                println!("seed {seed}: FAIL -- {why}");
                println!("replay: io_faults --seed {seed} --ops {num_ops}");
                std::process::exit(1);
            }
        }
        rounds += 1;
    }
    let _ = std::fs::remove_dir_all(&base);
    // Gate rule #10: a run that injected nothing, or never saw a commit both
    // survive and fail, proved nothing.
    if rounds >= 20
        && (total.injected == 0
            || total.commits_ok == 0
            || total.commits_failed == 0
            || total.docs == 0)
    {
        println!(
            "io_faults: FAIL -- the run never exercised every outcome ({} injected, {} commits ok, {} failed, {} docs kept)",
            total.injected, total.commits_ok, total.commits_failed, total.docs
        );
        std::process::exit(1);
    }
    println!(
        "io_faults: {rounds} round(s) passed in {:.1}s ({} injected errors, {} commits ok, {} failed)",
        started.elapsed().as_secs_f64(),
        total.injected,
        total.commits_ok,
        total.commits_failed
    );
}
