//! M14.h h4's acceptance run: a `RandomIndexWriter` under a random
//! `newIndexWriterConfig` (merge policy, scheduler, buffers, flush mode)
//! drives a random stream of adds, updates, deletes, doc-values updates,
//! commits, flushes and forced merges over a fresh `FsDirectory`, per seed.
//!
//! Every so often, and at the end, a reader is opened (NRT, through the
//! writer) and:
//!
//! 1. its live documents are exactly the ids a model says are live, each
//!    with the model's `n` doc value;
//! 2. every leaf is walked through an `AssertingLeafReader` -- every term,
//!    every posting with its positions, every doc value -- so the asserting
//!    wrappers check what the writer produced.
//!
//! After the writer closes, this port's `CheckIndex` must pass, and real
//! Lucene's too with `--java-cp` (`org.apache.lucene.index.CheckIndex`).
//!
//! Usage: `random_index [--seeds A..B | --seed N] [--ops N] [--java-cp CP] [--dir DIR]`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, IntField, NumericDocValuesField, SortedDocValuesField, SortedSetDocValuesField,
    Store, StringField, TextField,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{LeafReader, PostingsFlags, NO_MORE_DOCS};
use lucene_store::FsDirectory;
use lucene_test_framework::analysis::mock_analyzer::MockAnalyzer;
use lucene_test_framework::index::asserting_leaf_reader::AssertingLeafReader;
use lucene_test_framework::index::random_index_writer::{RandomIndexWriter, WriterEnv};
use lucene_test_framework::util::test_util;
use lucene_util::java_random::JavaRandom;

const WORDS: [&str; 12] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu",
];

/// What one round saw, for the run's evidence check.
#[derive(Default)]
struct Round {
    readers: usize,
    deletes: usize,
    dv_updates: usize,
    commits: usize,
    live: usize,
    postings: u64,
}

fn document(id: i32, n: i64, ops: &mut JavaRandom) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", id.to_string(), Store::Yes));
    let len = 1 + ops.next_int_bounded(8);
    let body: Vec<&str> = (0..len)
        .map(|_| WORDS[ops.next_int_bounded(WORDS.len() as i32) as usize])
        .collect();
    d.add(TextField::new("body", body.join(" "), Store::No));
    d.add(NumericDocValuesField::new("n", n));
    d.add(SortedDocValuesField::new(
        "s",
        WORDS[(id as usize) % WORDS.len()],
    ));
    for _ in 0..ops.next_int_bounded(3) {
        d.add(SortedSetDocValuesField::new(
            "ss",
            WORDS[ops.next_int_bounded(WORDS.len() as i32) as usize],
        ));
    }
    d.add(IntField::new("p", id % 97, Store::No));
    d
}

fn id_term(id: i32) -> Term {
    Term::new("id", id.to_string())
}

/// Walks every term, posting, position and doc value of `leaf` through an
/// `AssertingLeafReader`, returning the postings visited.
fn walk(leaf: Arc<dyn LeafReader>) -> Result<u64, String> {
    let e = |e: lucene_search::Error| e.to_string();
    let r = AssertingLeafReader::new(leaf);
    let _ = r.live_docs();
    let mut visited = 0u64;
    for field in ["id", "body"] {
        let Some(terms) = r.terms(field).map_err(e)? else {
            continue;
        };
        let _ = (terms.sum_total_term_freq(), terms.doc_count());
        let mut te = terms.iterator().map_err(e)?;
        while te.next().map_err(e)?.is_some() {
            let _ = te.total_term_freq().map_err(e)?;
            let mut p = te.postings(PostingsFlags::Positions).map_err(e)?;
            while p.next_doc().map_err(e)? != NO_MORE_DOCS {
                for _ in 0..p.freq() {
                    p.next_position().map_err(e)?;
                }
                visited += 1;
            }
        }
    }
    if let Some(mut n) = r.numeric_doc_values("n").map_err(e)? {
        while n.next_doc().map_err(e)? != NO_MORE_DOCS {
            let _ = n.long_value();
        }
    }
    if let Some(mut s) = r.sorted_doc_values("s").map_err(e)? {
        while s.next_doc().map_err(e)? != NO_MORE_DOCS {
            let ord = s.ord_value();
            s.lookup_ord(ord).map_err(e)?;
        }
    }
    if let Some(mut ss) = r.sorted_set_doc_values("ss").map_err(e)? {
        while ss.next_doc().map_err(e)? != NO_MORE_DOCS {
            for _ in 0..ss.doc_value_count() {
                let ord = ss.next_ord().map_err(e)?;
                ss.lookup_ord(ord).map_err(e)?;
            }
        }
    }
    if let Some(mut p) = r.sorted_numeric_doc_values("p").map_err(e)? {
        while p.next_doc().map_err(e)? != NO_MORE_DOCS {
            for _ in 0..p.doc_value_count() {
                p.next_value().map_err(e)?;
            }
        }
    }
    if let Some(mut norms) = r.norm_values("body").map_err(e)? {
        while norms.next_doc().map_err(e)? != NO_MORE_DOCS {
            let _ = norms.long_value();
        }
    }
    Ok(visited)
}

/// The reader's live documents as id -> `n`, and the postings walked.
fn read(reader: &DirectoryReader) -> Result<(BTreeMap<i32, i64>, u64), String> {
    let e = |e: lucene_search::Error| e.to_string();
    let mut live = BTreeMap::new();
    let mut visited = 0;
    for seg in reader.segment_readers() {
        let leaf: Arc<dyn LeafReader> = Arc::new(seg.clone());
        visited += walk(Arc::clone(&leaf))?;
        let mut n = leaf
            .numeric_doc_values("n")
            .map_err(e)?
            .ok_or("a segment without n")?;
        for doc in 0..seg.max_doc {
            if !seg.live_docs().is_none_or(|l| l.get(doc as usize)) {
                continue;
            }
            let mut v = DocumentVisitor::all();
            seg.document(doc, &mut v).map_err(e)?;
            let stored = v.into_document();
            let id = stored
                .fields
                .iter()
                .find_map(|f| match &f.value {
                    FieldValue::String(s) => s.parse::<i32>().ok(),
                    _ => None,
                })
                .ok_or("no id")?;
            if !n.advance_exact(doc).map_err(e)? {
                return Err(format!("doc {doc} (id {id}) has no n"));
            }
            if live.insert(id, n.long_value()).is_some() {
                return Err(format!("id {id} is live twice"));
            }
        }
    }
    Ok((live, visited))
}

fn check(dir: &FsDirectory, path: &Path, java_cp: Option<&str>) -> Result<(), String> {
    test_util::check_index(dir)?;
    if let Some(cp) = java_cp {
        let out = std::process::Command::new("java")
            .args(["-cp", cp, "org.apache.lucene.index.CheckIndex"])
            .arg(path)
            .output()
            .map_err(|e| format!("java: {e}"))?;
        let text = String::from_utf8_lossy(&out.stdout);
        // Gate rule #10: an exit status alone would pass over an index Java
        // never opened; it must have checked every segment the port wrote.
        let ours = lucene_index::segment_infos::read_latest(dir)
            .map_err(|e| e.to_string())?
            .segments
            .len();
        let theirs = text
            .split_whitespace()
            .find_map(|w| w.strip_prefix("numSegments="))
            .and_then(|n| n.parse::<usize>().ok());
        if !out.status.success()
            || !text.contains("No problems were detected")
            || theirs != Some(ours)
        {
            let tail: Vec<&str> = text.lines().rev().take(15).collect();
            return Err(format!(
                "Lucene CheckIndex failed (numSegments {theirs:?}, ours {ours}):\n{}",
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
    }
    Ok(())
}

fn round(seed: u64, base: &Path, num_ops: i32, java_cp: Option<&str>) -> Result<Round, String> {
    let e = |e: lucene_search::directory_reader::Error| e.to_string();
    let path = base.join(format!("seed-{seed}"));
    let _ = std::fs::remove_dir_all(&path);
    let dir = FsDirectory::open(&path);
    let mut r = JavaRandom::new(seed as i64);
    let analyzer = Arc::new(MockAnalyzer::new(&mut r).build());
    let env = WriterEnv::new(seed as i64 ^ 0x5eed);
    let mut w = RandomIndexWriter::with_analyzer(&mut r, &dir, analyzer, env).map_err(e)?;
    let mut ops = JavaRandom::new(seed as i64 ^ 0x0b5);
    let mut model: BTreeMap<i32, i64> = BTreeMap::new();
    let mut next_id = 0i32;
    let mut out = Round::default();
    let verify = |w: &mut RandomIndexWriter<'_>, model: &BTreeMap<i32, i64>, out: &mut Round| {
        let reader = w.get_reader().map_err(|e| e.to_string())?;
        let (live, visited) = read(&reader)?;
        if &live != model {
            let missing: Vec<_> = model.keys().filter(|k| !live.contains_key(k)).collect();
            let extra: Vec<_> = live.keys().filter(|k| !model.contains_key(k)).collect();
            let wrong: Vec<_> = live
                .iter()
                .filter_map(|(k, v)| model.get(k).filter(|m| *m != v).map(|m| (k, v, m)))
                .take(5)
                .collect();
            return Err(format!(
                "the reader disagrees with the model: missing {missing:?}, extra {extra:?}, \
                 (id, n, model's n) {wrong:?}"
            ));
        }
        out.readers += 1;
        out.postings += visited;
        Ok::<(), String>(())
    };
    for _ in 0..num_ops {
        let roll = ops.next_int_bounded(100);
        let existing = |ops: &mut JavaRandom| {
            if next_id == 0 {
                0
            } else {
                ops.next_int_bounded(next_id)
            }
        };
        match roll {
            0..=49 => {
                let n = ops.next_long();
                w.add_document(&document(next_id, n, &mut ops)).map_err(e)?;
                model.insert(next_id, n);
                next_id += 1;
            }
            50..=64 => {
                let id = existing(&mut ops);
                let n = ops.next_long();
                w.update_document(id_term(id), &document(id, n, &mut ops))
                    .map_err(e)?;
                model.insert(id, n);
                next_id = next_id.max(id + 1);
            }
            65..=74 => {
                let id = existing(&mut ops);
                w.delete_documents(id_term(id)).map_err(e)?;
                model.remove(&id);
                out.deletes += 1;
            }
            // `updateNumericDocValue` refuses a field nothing has indexed yet.
            75..=82 if next_id > 0 => {
                let id = existing(&mut ops);
                let n = ops.next_long();
                w.update_numeric_doc_value(id_term(id), "n", n).map_err(e)?;
                if let Some(v) = model.get_mut(&id) {
                    *v = n;
                }
                out.dv_updates += 1;
            }
            75..=87 => {
                w.commit().map_err(e)?;
                out.commits += 1;
            }
            88..=93 => verify(&mut w, &model, &mut out)?,
            94..=96 => w.flush().map_err(e)?,
            97 => w.force_merge_deletes().map_err(e)?,
            _ => w
                .force_merge(1 + ops.next_int_bounded(3) as usize)
                .map_err(e)?,
        }
    }
    verify(&mut w, &model, &mut out)?;
    out.live = model.len();
    w.close().map_err(e)?;

    // The committed index, read cold, is the model too.
    let reader = DirectoryReader::open(&dir).map_err(|e| e.to_string())?;
    let (live, _) = read(&reader)?;
    if live != model {
        return Err(format!(
            "the committed index holds {} live docs, the model {}",
            live.len(),
            model.len()
        ));
    }
    if reader.num_docs() as usize != model.len() {
        return Err("numDocs disagrees with the model".into());
    }
    drop(reader);
    check(&dir, &path, java_cp)?;
    let _ = std::fs::remove_dir_all(&path);
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let num_ops: i32 = value("--ops").map_or(300, |v| v.parse().expect("--ops"));
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
        || std::env::temp_dir().join(format!("random-index-{}", std::process::id())),
        PathBuf::from,
    );
    // A panic (an asserting wrapper firing, say) is this round's failure.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if std::env::var_os("RANDOM_INDEX_BACKTRACE").is_some() {
            default_hook(info);
        }
    }));

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
                    "seed {seed}: ok -- {} readers, {} live, {} deletes, {} dv updates, {} commits, {} postings walked",
                    r.readers, r.live, r.deletes, r.dv_updates, r.commits, r.postings
                );
                total.readers += r.readers;
                total.deletes += r.deletes;
                total.dv_updates += r.dv_updates;
                total.commits += r.commits;
                total.live += r.live;
                total.postings += r.postings;
            }
            Err(why) => {
                println!("seed {seed}: FAIL -- {why}");
                println!("replay: random_index --seed {seed} --ops {num_ops}");
                std::process::exit(1);
            }
        }
        rounds += 1;
    }
    let _ = std::fs::remove_dir_all(&base);
    // Gate rule #10: a run that never deleted, updated, committed or read
    // proved nothing about those paths.
    if rounds >= 20
        && (total.readers == 0
            || total.deletes == 0
            || total.dv_updates == 0
            || total.commits == 0
            || total.postings == 0)
    {
        println!("random_index: FAIL -- the run never exercised every operation");
        std::process::exit(1);
    }
    println!(
        "random_index: {rounds} round(s) passed in {:.1}s ({} readers checked, {} postings walked, {} live docs)",
        started.elapsed().as_secs_f64(),
        total.readers,
        total.postings,
        total.live
    );
}
