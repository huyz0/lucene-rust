//! M4's crash fuzzer (T4.4): a seeded, replayable random operation stream
//! against a real on-disk index, crashed at a random point, then checked.
//!
//! Two kinds of crash:
//!
//! - **Power loss** (default). The writer runs over a
//!   [`CrashingDirectory`], which fails the n-th directory operation and
//!   every one after it -- n drawn uniformly over the whole run, so crashes
//!   land inside flushes, merges, `prepare_commit`, `finish_commit` and
//!   between the two -- and then rewrites the directory into a state a power
//!   loss could leave: unsynced files kept, truncated, zeroed or gone, and
//!   only a prefix of the unpublished creates, renames and deletes.
//! - **`kill -9`** (`--kill`). A child process runs the same stream over a
//!   plain directory and is killed at a random moment. The page cache
//!   survives this, so it tests process death, not `fsync` ordering.
//! - **Concurrent power loss** (`--concurrent`). Several threads update and
//!   delete through a `ConcurrentIndexWriter`, with its merge thread and a
//!   committer, over the crashing directory. What survives must be a clean
//!   prefix of the operations by sequence number, reaching at least the last
//!   commit that returned.
//!
//! Each seed also draws its conditions: an operation mix (balanced, update-,
//! delete-, commit- or flush-heavy), the buffer size and the merge policy's
//! shape -- so seeds spread over conditions rather than repeating one.
//!
//! After either, the checks are the milestone's:
//!
//! 1. The index opens.
//! 2. What it holds is **exactly** the last durable commit -- every live
//!    document by id and version, nothing partial, no deleted document back.
//!    A crash *during* a commit may leave that commit or the one before;
//!    nothing else.
//! 3. This port's `CheckIndex` passes, and real Lucene's too with
//!    `--java-cp` (`org.apache.lucene.index.CheckIndex`).
//! 4. A writer reopened on it recovers: it adds a document, commits, and the
//!    result passes 2 and 3 again.
//!
//! The restarted side reads through `FsDirectory` for even seeds and
//! `MmapDirectory` for odd ones.
//!
//! Every document carries `id` (a unique term, so updates and deletes can
//! find it) and two numeric doc values, `idn` (the id again) and `ver` (its
//! version, bumped by every update). This port's reader returns doc ids, not
//! stored documents, so `idn`/`ver` are how the check reads the index back.
//!
//! Usage:
//!
//! ```text
//! crash_fuzz [--seeds A..B | --seed S] [--duration SECS] [--ops N]
//!            [--kill | --concurrent] [--java-cp CLASSPATH] [--dir PATH]
//! ```
//!
//! A failure prints its seed; `--seed S` replays it exactly.
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::MergePolicyConfig;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::crashing_directory::CrashingDirectory;
use lucene_store::{Directory, FsDirectory, MmapDirectory};

/// id -> version: the logical content of an index.
type Model = BTreeMap<i64, i64>;

#[derive(Debug, Clone, Copy)]
enum Op {
    Add,
    Update,
    Delete,
    Flush,
    Commit,
    TwoPhaseCommit,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// The seeded stream: everything a run does is a function of the seed, so the
/// model can be replayed without the index (which is how `--kill` knows what
/// each commit held).
struct Stream {
    rng: Rng,
    next_id: i64,
    /// The writer's view: every buffered change applied.
    current: Model,
    max_buffered_docs: i32,
    segments_per_tier: usize,
    max_merge_at_once: usize,
    floor_segment_size: u64,
    /// Cumulative thresholds out of 100 for add / update / delete / flush /
    /// commit; the rest is two-phase commit. Drawn per seed from
    /// [`PROFILES`], so the seeds spread over update-, delete-, commit- and
    /// flush-heavy streams instead of all sharing one mix.
    mix: [u64; 5],
}

/// Operation mixes (cumulative percentages: add, update, delete, flush,
/// commit; the remainder two-phase commit).
const PROFILES: [[u64; 5]; 5] = [
    [58, 72, 82, 87, 95], // balanced
    [30, 75, 82, 86, 94], // update-heavy: deletes buffered against live docs
    [35, 45, 80, 85, 93], // delete-heavy: fully deleted segments, dropped ones
    [45, 58, 68, 72, 88], // commit-heavy: most crashes land in a commit
    [55, 68, 76, 94, 97], // flush-heavy: many small segments, many merges
];

impl Stream {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        // Small enough that `add_document` flushes on its own between
        // commits, so crashes land inside automatic flushes too.
        let max_buffered_docs = 2 + rng.below(15) as i32;
        let segments_per_tier = 2 + rng.below(4) as usize;
        let max_merge_at_once = 2 + rng.below(9) as usize;
        // A small floor merges eagerly; a huge one treats every segment as
        // tiny, so merges become rare and wide.
        let floor_segment_size = if rng.below(2) == 0 { 1 << 12 } else { 1 << 30 };
        let mix = PROFILES[rng.below(PROFILES.len() as u64) as usize];
        Stream {
            rng,
            next_id: 0,
            current: Model::new(),
            max_buffered_docs,
            segments_per_tier,
            max_merge_at_once,
            floor_segment_size,
            mix,
        }
    }

    /// The next operation, and the document or term it acts on.
    fn next_op(&mut self) -> (Op, i64) {
        let roll = self.rng.below(100);
        let existing = |s: &mut Self| -> Option<i64> {
            if s.current.is_empty() {
                return None;
            }
            let n = s.rng.below(s.current.len() as u64) as usize;
            s.current.keys().nth(n).copied()
        };
        let [add, update, delete, flush, commit] = self.mix;
        if roll < add {
            let id = self.next_id;
            self.next_id += 1;
            (Op::Add, id)
        } else if roll < update {
            match existing(self) {
                Some(id) => (Op::Update, id),
                None => self.next_op(),
            }
        } else if roll < delete {
            match existing(self) {
                Some(id) => (Op::Delete, id),
                None => self.next_op(),
            }
        } else if roll < flush {
            (Op::Flush, 0)
        } else if roll < commit {
            (Op::Commit, 0)
        } else {
            (Op::TwoPhaseCommit, 0)
        }
    }

    /// Applies a document change to the model.
    fn apply(&mut self, op: Op, id: i64) {
        match op {
            Op::Add => {
                self.current.insert(id, 0);
            }
            Op::Update => {
                *self.current.get_mut(&id).expect("updated id exists") += 1;
            }
            Op::Delete => {
                self.current.remove(&id);
            }
            _ => {}
        }
    }
}

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo::new(name, number)
}

fn schema() -> Vec<FieldInfo> {
    vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            omit_norms: true,
            ..field("id", 0)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("idn", 1)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("ver", 2)
        },
    ]
}

fn document(id: i64, version: i64) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("i{id}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(id),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(version),
            },
        ],
    }
}

fn id_term(id: i64) -> Term {
    Term {
        field: "id".to_string(),
        bytes: format!("i{id}").into_bytes(),
    }
}

fn open_writer<'d>(
    dir: &'d dyn Directory,
    stream: &Stream,
) -> lucene_index::index_writer::Result<IndexWriter<'d>> {
    let mut writer = IndexWriter::open(
        dir,
        schema(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )?;
    writer.set_postings_field(Some("id"))?;
    writer.set_doc_values_field(Some("idn"))?;
    writer.add_doc_values_field("ver")?;
    writer.set_max_buffered_docs(stream.max_buffered_docs)?;
    writer.set_ram_buffer_size_mb(4096.0)?;
    writer.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: stream.max_merge_at_once,
        segments_per_tier: stream.segments_per_tier,
        max_merged_segment_size: 1 << 30,
        floor_segment_size: stream.floor_segment_size,
        ..MergePolicyConfig::default()
    }));
    Ok(writer)
}

/// What a run left for the check: the last commit known durable, and a commit
/// that was in flight when the crash hit (which may or may not have landed).
#[derive(Debug, Default)]
struct Outcome {
    committed: Model,
    /// The commit in flight when the crash hit -- set only when the crash came
    /// **after** that commit's `segments_N` rename, since before it the commit
    /// must not be visible. (`--kill` cannot see the rename and journals the
    /// commit's start instead; see `kill_round`.)
    in_flight: Option<Model>,
    ops_done: usize,
    crashed_in: Option<Op>,
    /// A writer error that was **not** the crash: always a failure.
    error: Option<String>,
}

/// Runs `num_ops` operations of `seed`'s stream, stopping at the first error
/// (the crash). `on_commit(k, begun)` is told as commit `k` starts and ends.
fn run(
    dir: &dyn Directory,
    probe: Option<&CrashingDirectory>,
    seed: u64,
    num_ops: usize,
    mut on_commit: impl FnMut(usize, bool),
) -> Outcome {
    let mut stream = Stream::new(seed);
    let mut outcome = Outcome::default();
    let crashed = || probe.is_some_and(CrashingDirectory::crashed);
    let mut writer = match open_writer(dir, &stream) {
        Ok(w) => w,
        Err(e) => {
            if !crashed() {
                outcome.error = Some(format!("open: {e}"));
            }
            return outcome;
        }
    };
    let mut commits = 0usize;
    let mut published_before = None;
    for _ in 0..num_ops {
        let (op, id) = stream.next_op();
        let result: lucene_index::index_writer::Result<()> = match op {
            Op::Add => writer.add_document(document(id, 0)).map(drop),
            Op::Update => {
                let version = stream.current[&id] + 1;
                writer
                    .update_document(id_term(id), document(id, version))
                    .map(drop)
            }
            Op::Delete => writer.delete_documents_by_term(&[id_term(id)]).map(drop),
            Op::Flush => writer.flush(),
            Op::Commit | Op::TwoPhaseCommit => {
                published_before = probe.map(CrashingDirectory::published);
                commits += 1;
                on_commit(commits, true);
                let r = match op {
                    Op::Commit => writer.commit().map(drop),
                    _ => writer
                        .prepare_commit()
                        .and_then(|()| writer.finish_commit().map(drop)),
                };
                if r.is_ok() {
                    on_commit(commits, false);
                }
                r
            }
        };
        match result {
            Ok(()) => {
                stream.apply(op, id);
                if matches!(op, Op::Commit | Op::TwoPhaseCommit) {
                    outcome.committed = stream.current.clone();
                }
                outcome.ops_done += 1;
            }
            Err(e) => {
                if !crashed() {
                    outcome.error = Some(format!("{op:?}: {e}"));
                    break;
                }
                let published = match (probe, published_before) {
                    (Some(p), Some(before)) => p.published() > before,
                    _ => false,
                };
                if matches!(op, Op::Commit | Op::TwoPhaseCommit) && published {
                    outcome.in_flight = Some(stream.current.clone());
                }
                outcome.crashed_in = Some(op);
                break;
            }
        }
    }
    outcome
}

/// The models of commits `1..`, replayed from the seed alone.
fn commit_models(seed: u64, num_ops: usize) -> Vec<Model> {
    let mut stream = Stream::new(seed);
    let mut models = vec![Model::new()];
    for _ in 0..num_ops {
        let (op, id) = stream.next_op();
        stream.apply(op, id);
        if matches!(op, Op::Commit | Op::TwoPhaseCommit) {
            models.push(stream.current.clone());
        }
    }
    models
}

/// Reads the index's logical content, or why it could not.
fn read_model(dir: &dyn Directory) -> Result<Option<Model>, String> {
    let files = dir.list_all().map_err(|e| e.to_string())?;
    if !files.iter().any(|f| f.starts_with("segments_")) {
        return Ok(None);
    }
    let reader = DirectoryReader::open(dir).map_err(|e| format!("open: {e}"))?;
    let mut model = Model::new();
    for segment in reader.segment_readers() {
        let number = |name: &str| {
            segment
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.number)
        };
        let (Some(idn), Some(ver)) = (number("idn"), number("ver")) else {
            return Err("a segment lacks idn/ver".to_string());
        };
        let column = |n: i32| {
            segment
                .doc_values_for_field(n)
                .and_then(|(meta, data)| meta.numeric_entry(n).map(|e| (e, data)))
                .ok_or_else(|| format!("no doc values for field {n}"))
        };
        let (idn_entry, idn_data) = column(idn)?;
        let (ver_entry, ver_data) = column(ver)?;
        for doc in 0..segment.max_doc {
            if segment.live_docs().is_some_and(|l| !l.get(doc as usize)) {
                continue;
            }
            let read = |data, entry, doc| {
                doc_values::numeric_value(data, entry, doc)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| format!("doc {doc} has no value"))
            };
            let id = read(idn_data, idn_entry, doc)?;
            let version = read(ver_data, ver_entry, doc)?;
            if model.insert(id, version).is_some() {
                return Err(format!("id {id} is live twice"));
            }
        }
    }
    Ok(Some(model))
}

fn check_index(dir: &dyn Directory, path: &Path, java_cp: Option<&str>) -> Result<(), String> {
    for result in lucene_index::check_index::check_directory(dir).map_err(|e| e.to_string())? {
        if !result.all_passed() {
            return Err(format!(
                "CheckIndex {}: {:?}",
                result.segment_name,
                result.failures()
            ));
        }
    }
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

/// Checks 1-4 over the crashed index at `path`, given which states it may
/// legitimately hold.
fn check_after_crash(
    seed: u64,
    path: &Path,
    allowed: &[&Model],
    java_cp: Option<&str>,
) -> Result<&'static str, String> {
    let dir = restart_dir(path, seed);
    let visible = read_model(dir.as_ref())?.unwrap_or_default();
    let which = allowed.iter().position(|m| **m == visible).ok_or_else(|| {
        let describe = |m: &Model| format!("{} docs", m.len());
        format!(
            "visible state ({}) is none of the allowed ones ({}){}",
            describe(&visible),
            allowed
                .iter()
                .map(|m| describe(m))
                .collect::<Vec<_>>()
                .join(", "),
            first_difference(&visible, allowed[0])
        )
    })?;
    let has_commit = dir
        .list_all()
        .map_err(|e| e.to_string())?
        .iter()
        .any(|f| f.starts_with("segments_"));
    if has_commit {
        check_index(dir.as_ref(), path, java_cp)?;
    }

    // Recovery: a new writer must clean up and carry on.
    let stream = Stream::new(seed);
    let mut writer = open_writer(dir.as_ref(), &stream).map_err(|e| format!("reopen: {e}"))?;
    let marker = 1_000_000_000;
    writer
        .add_document(document(marker, 7))
        .map_err(|e| format!("recovery add: {e}"))?;
    writer
        .commit()
        .map_err(|e| format!("recovery commit: {e}"))?;
    drop(writer);
    let mut expected = visible.clone();
    expected.insert(marker, 7);
    let recovered = read_model(restart_dir(path, seed).as_ref())?.unwrap_or_default();
    if recovered != expected {
        return Err(format!(
            "after recovery: {} docs, want {}{}",
            recovered.len(),
            expected.len(),
            first_difference(&recovered, &expected)
        ));
    }
    check_index(restart_dir(path, seed).as_ref(), path, java_cp)?;
    check_only_live_files(restart_dir(path, seed).as_ref())?;
    Ok(if which == 0 {
        "last commit"
    } else {
        "in-flight commit"
    })
}

/// After the recovery commit, the directory holds exactly what that commit
/// references -- `IndexFileDeleter` has reclaimed every leftover of the crash
/// (an orphaned `pending_segments_N`, a half-written segment, a superseded
/// commit).
fn check_only_live_files(dir: &dyn Directory) -> Result<(), String> {
    let infos = lucene_index::segment_infos::read_latest(dir).map_err(|e| e.to_string())?;
    let mut live: BTreeSet<String> = BTreeSet::new();
    live.insert(format!(
        "segments_{}",
        lucene_util::base36::to_base36(infos.generation)
    ));
    for sci in &infos.segments {
        let si_bytes = dir
            .open(&format!("{}.si", sci.segment_name))
            .map_err(|e| e.to_string())?;
        let si = lucene_index::segment_info::parse(&si_bytes, &sci.segment_id)
            .map_err(|e| e.to_string())?;
        live.extend(sci.files(&si.files));
    }
    let leftovers: Vec<String> = dir
        .list_all()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|f| !live.contains(f) && f != "write.lock")
        .collect();
    if leftovers.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "files the recovered commit does not reference: {leftovers:?}"
        ))
    }
}

fn first_difference(got: &Model, want: &Model) -> String {
    for (id, v) in want {
        match got.get(id) {
            Some(g) if g == v => {}
            Some(g) => return format!("; id {id} at version {g}, want {v}"),
            None => return format!("; id {id} missing"),
        }
    }
    for (id, v) in got {
        if !want.contains_key(id) {
            return format!("; id {id} (version {v}) should not be there");
        }
    }
    String::new()
}

/// The directory a restarted process reads the crashed index through:
/// `FsDirectory` for even seeds, `MmapDirectory` for odd ones, so both
/// backends see every kind of crash.
fn restart_dir(path: &Path, seed: u64) -> Box<dyn Directory> {
    if seed.is_multiple_of(2) {
        Box::new(FsDirectory::open(path))
    } else {
        Box::new(MmapDirectory::open(path))
    }
}

fn fresh(dir: &Path) -> PathBuf {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("create dir");
    dir.to_path_buf()
}

/// What one round saw, for the per-run evidence check in `main`.
struct Round {
    summary: String,
    /// "last commit" or "in-flight commit".
    visible: &'static str,
    crashed_in: Option<Op>,
    crash_point: String,
    killed_mid_run: bool,
}

/// One power-loss round for `seed`.
fn power_loss_round(
    seed: u64,
    base: &Path,
    num_ops: usize,
    java_cp: Option<&str>,
) -> Result<Round, String> {
    // A crash-free run measures how many directory operations the stream
    // makes, so the crash point is uniform over all of them. It must also
    // *be* crash-free: a writer error here would otherwise shrink the range
    // and pass every round.
    let dry = fresh(&base.join("dry"));
    let dir = CrashingDirectory::new(&dry, seed);
    let dry_run = run(&dir, Some(&dir), seed, num_ops, |_, _| {});
    if let Some(error) = dry_run.error {
        return Err(format!("the crash-free run failed: {error}"));
    }
    if dry_run.ops_done != num_ops {
        return Err(format!(
            "the crash-free run stopped after {} ops",
            dry_run.ops_done
        ));
    }
    let total = dir.ops().max(1);

    let path = fresh(&base.join("crash"));
    let dir = CrashingDirectory::new(&path, seed);
    let crash_at = 1 + Rng::new(seed ^ 0xc4a5_4000).below(total);
    dir.crash_after(crash_at);
    let outcome = run(&dir, Some(&dir), seed, num_ops, |_, _| {});
    if let Some(error) = outcome.error {
        return Err(format!("a writer error that is not the crash: {error}"));
    }
    let crash_point = dir.crash_point().unwrap_or_default();
    let loss = dir.power_loss().map_err(|e| format!("power loss: {e}"))?;
    let mut allowed = vec![&outcome.committed];
    if let Some(m) = &outcome.in_flight {
        allowed.push(m);
    }
    let visible = check_after_crash(seed, &path, &allowed, java_cp)
        .map_err(|why| format!("crash at dir op {crash_at}/{total} ({crash_point}): {why}"))?;
    Ok(Round {
        summary: format!(
            "crash at dir op {crash_at}/{total} ({crash_point}) in {:?} after {} ops; kept {}/{} unpublished changes, {} unsynced files; visible = {visible}",
            outcome.crashed_in,
            outcome.ops_done,
            loss.kept_events,
            loss.pending_events,
            loss.damaged.len()
        ),
        visible,
        crashed_in: outcome.crashed_in,
        crash_point,
        killed_mid_run: false,
    })
}

/// One concurrent round: `2 + seed % 3` threads updating and deleting their
/// own ids through a `ConcurrentIndexWriter` over the crashing directory,
/// its merge thread running, and a committer committing between them -- the
/// crash lands wherever the directory operation count says, in a flush, a
/// merge, a commit or its cut, on whichever thread happens to be there. The
/// visible state must then be a **clean prefix** of the operations by
/// sequence number, no shorter than the last commit that returned: every
/// operation up to some point, none after it.
fn concurrent_round(
    seed: u64,
    base: &Path,
    num_ops: usize,
    java_cp: Option<&str>,
) -> Result<Round, String> {
    let dry = fresh(&base.join("dry"));
    let dir = CrashingDirectory::new(&dry, seed);
    let dry_run = concurrent_run(&dir, seed, num_ops);
    if let Some(error) = dry_run.error {
        return Err(format!("the crash-free run failed: {error}"));
    }
    let total = dir.ops().max(1);

    let path = fresh(&base.join("crash"));
    let dir = CrashingDirectory::new(&path, seed);
    let crash_at = 1 + Rng::new(seed ^ 0xc0c0_4000).below(total);
    dir.crash_after(crash_at);
    let run = concurrent_run(&dir, seed, num_ops);
    if let Some(error) = run.error {
        return Err(format!("a writer error that is not the crash: {error}"));
    }
    let crash_point = dir.crash_point().unwrap_or_default();
    dir.power_loss().map_err(|e| format!("power loss: {e}"))?;

    let visible = read_model(restart_dir(&path, seed).as_ref())?.unwrap_or_default();
    // The ops in sequence order; the prefix up to the last returned commit
    // is the least the index may hold.
    let mut ops = run.ops;
    ops.sort_by_key(|&(seq, _, _)| seq);
    let mut prefix = Model::new();
    let mut i = 0;
    while i < ops.len() && ops[i].0 <= run.last_commit {
        apply_op(&mut prefix, &ops[i]);
        i += 1;
    }
    // An operation a thread saw fail (the crash) may or may not be in: its
    // id is compared loosely -- either the prefix's value or the failed op's.
    let matches = |prefix: &Model| {
        let ids: BTreeSet<i64> = prefix.keys().chain(visible.keys()).copied().collect();
        ids.into_iter().all(|id| {
            let (want, got) = (prefix.get(&id), visible.get(&id));
            want == got
                || run
                    .uncertain
                    .iter()
                    .any(|&(u, v)| u == id && got == v.as_ref())
        })
    };
    let mut found = matches(&prefix).then_some(i);
    while found.is_none() && i < ops.len() {
        apply_op(&mut prefix, &ops[i]);
        i += 1;
        if matches(&prefix) {
            found = Some(i);
        }
    }
    let at = found.ok_or_else(|| {
        format!(
            "crash at dir op {crash_at}/{total} ({crash_point}): visible state ({} docs) is no prefix of the {} ops at or past the last commit (seq {}){}",
            visible.len(),
            ops.len(),
            run.last_commit,
            first_difference(&visible, &prefix)
        )
    })?;
    let past_commit = ops[..at].iter().filter(|o| o.0 > run.last_commit).count();
    let visible_label = if past_commit == 0 {
        "last commit"
    } else {
        "in-flight commit"
    };
    // CheckIndex, then recovery by a new writer, as for every other round.
    check_after_crash(seed, &path, &[&visible], java_cp)
        .map_err(|why| format!("crash at dir op {crash_at}/{total} ({crash_point}): {why}"))?;
    Ok(Round {
        summary: format!(
            "concurrent, {} threads: crash at dir op {crash_at}/{total} ({crash_point}) after {} ops, {} commits; visible = {visible_label} ({past_commit} ops past it)",
            concurrent_threads(seed),
            ops.len(),
            run.commits
        ),
        visible: visible_label,
        crashed_in: None,
        crash_point,
        killed_mid_run: false,
    })
}

fn concurrent_threads(seed: u64) -> usize {
    2 + (seed % 3) as usize
}

fn apply_op(model: &mut Model, &(_, id, version): &(i64, i64, Option<i64>)) {
    match version {
        Some(v) => model.insert(id, v),
        None => model.remove(&id),
    };
}

/// What a concurrent run did: each operation that returned, with its
/// sequence number; the operations that failed (at most one per thread);
/// the sequence number of the last commit that returned.
#[derive(Default)]
struct ConcurrentRun {
    ops: Vec<(i64, i64, Option<i64>)>,
    uncertain: Vec<(i64, Option<i64>)>,
    last_commit: i64,
    commits: usize,
    error: Option<String>,
}

fn concurrent_run(dir: &CrashingDirectory, seed: u64, num_ops: usize) -> ConcurrentRun {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    let stream = Stream::new(seed);
    let threads = concurrent_threads(seed);
    let mut out = ConcurrentRun::default();
    let writer =
        match open_writer(dir, &stream).and_then(|w| ConcurrentIndexWriter::new(w, threads)) {
            Ok(w) => w,
            Err(e) => {
                if !dir.crashed() {
                    out.error = Some(format!("open: {e}"));
                }
                return out;
            }
        };
    let stop = AtomicBool::new(false);
    let working = AtomicUsize::new(threads);
    let shared = Mutex::new(ConcurrentRun::default());
    let fail = |what: String| {
        let mut s = shared.lock().unwrap();
        s.error.get_or_insert(what);
    };
    std::thread::scope(|scope| {
        let merger = scope.spawn(|| writer.run_merges(&stop));
        for t in 0..threads {
            let (writer, stop, working, shared, fail) = (&writer, &stop, &working, &shared, &fail);
            scope.spawn(move || {
                let mut rng = Rng::new(seed ^ ((t as u64 + 1) << 40));
                let mut versions: BTreeMap<i64, i64> = BTreeMap::new();
                for _ in 0..num_ops / threads {
                    if stop.load(Ordering::Acquire) || dir.crashed() {
                        break;
                    }
                    let id = (t as i64) * 1_000_000 + rng.below(40) as i64;
                    let roll = rng.below(100);
                    let (result, after) = if roll < 75 || !versions.contains_key(&id) {
                        let v = versions.get(&id).map_or(0, |v| v + 1);
                        (
                            writer.update_document(id_term(id), document(id, v)),
                            Some(v),
                        )
                    } else {
                        (writer.delete_documents_by_term(&[id_term(id)]), None)
                    };
                    match result {
                        Ok(seq) => {
                            match after {
                                Some(v) => versions.insert(id, v),
                                None => versions.remove(&id),
                            };
                            shared.lock().unwrap().ops.push((seq, id, after));
                        }
                        Err(e) => {
                            if dir.crashed() {
                                shared.lock().unwrap().uncertain.push((id, after));
                            } else {
                                fail(format!("thread {t}: {e}"));
                            }
                            break;
                        }
                    }
                }
                working.fetch_sub(1, Ordering::AcqRel);
            });
        }
        // The committer.
        let mut rng = Rng::new(seed ^ 0x00c0_ffee);
        loop {
            let last = working.load(Ordering::Acquire) == 0;
            if dir.crashed() {
                break;
            }
            match writer.commit() {
                Ok(seq) => {
                    let mut s = shared.lock().unwrap();
                    s.last_commit = seq;
                    s.commits += 1;
                }
                Err(e) => {
                    if !dir.crashed() {
                        fail(format!("commit: {e}"));
                    }
                    break;
                }
            }
            if last {
                break;
            }
            std::thread::sleep(Duration::from_micros(200 + rng.below(3000)));
        }
        stop.store(true, Ordering::Release);
        if let Ok(Err(e)) = merger.join() {
            if !dir.crashed() {
                fail(format!("merge thread: {e}"));
            }
        }
    });
    let s = shared.into_inner().unwrap();
    out.ops = s.ops;
    out.uncertain = s.uncertain;
    out.last_commit = s.last_commit;
    out.commits = s.commits;
    out.error = s.error;
    out
}

/// The child side of `--kill`: runs the stream to the end over a plain
/// directory, journaling each commit's start and end with an fsync, so the
/// parent knows which commits could be durable when it kills us.
fn child(seed: u64, path: &Path, num_ops: usize) {
    use std::io::Write;
    let dir = FsDirectory::open(path);
    let mut journal = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.with_extension("journal"))
        .expect("journal");
    let outcome = run(&dir, None, seed, num_ops, |k, begun| {
        // One `write` per line, so a kill leaves whole lines or none.
        let line = format!("{} {k}\n", if begun { "begin" } else { "end" });
        journal.write_all(line.as_bytes()).expect("journal");
    });
    // Nothing crashes this process but the kill, so any error is the
    // writer's: exit 2, which the parent reports as a failure.
    if outcome.error.is_some() || outcome.ops_done != num_ops {
        eprintln!("child: {:?} after {} ops", outcome.error, outcome.ops_done);
        std::process::exit(2);
    }
}

/// One `kill -9` round for `seed`.
fn kill_round(
    seed: u64,
    base: &Path,
    num_ops: usize,
    java_cp: Option<&str>,
) -> Result<Round, String> {
    let path = fresh(&base.join("kill"));
    let journal = path.with_extension("journal");
    let _ = std::fs::remove_file(&journal);
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut child = std::process::Command::new(exe)
        .args(["--child", &seed.to_string(), "--ops", &num_ops.to_string()])
        .arg(&path)
        .spawn()
        .map_err(|e| format!("spawn: {e}"))?;
    let delay = Duration::from_millis(5 + Rng::new(seed ^ 0x4b11).below(400));
    let started = Instant::now();
    while started.elapsed() < delay {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let _ = child.kill(); // SIGKILL
    let status = child.wait().map_err(|e| e.to_string())?;
    // Killed: no exit code. Finished before the kill: code 0. Anything else
    // is the writer failing on its own.
    let killed_mid_run = status.code().is_none();
    if status.code().is_some_and(|c| c != 0) {
        return Err(format!("the child failed on its own: {status}"));
    }

    let text = std::fs::read_to_string(&journal).unwrap_or_default();
    let (mut ended, mut begun) = (0usize, 0usize);
    for line in text.lines() {
        let mut parts = line.split(' ');
        let (kind, k) = (parts.next(), parts.next().and_then(|k| k.parse().ok()));
        match (kind, k) {
            (Some("begin"), Some(k)) => begun = k,
            (Some("end"), Some(k)) => ended = k,
            _ => {}
        }
    }
    let models = commit_models(seed, num_ops);
    let mut allowed = vec![&models[ended]];
    if begun > ended {
        allowed.push(&models[begun]);
    }
    let visible = check_after_crash(seed, &path, &allowed, java_cp).map_err(|why| {
        format!(
            "killed after {} ms, commits ended {ended}, begun {begun}: {why}",
            delay.as_millis()
        )
    })?;
    Ok(Round {
        summary: format!(
            "killed after {} ms ({}), commits ended {ended}, begun {begun}; visible = {visible}",
            delay.as_millis(),
            if killed_mid_run {
                "mid-run"
            } else {
                "had finished"
            }
        ),
        visible,
        crashed_in: None,
        crash_point: String::new(),
        killed_mid_run,
    })
}

/// What a run of rounds has exercised.
#[derive(Default)]
struct Evidence {
    last_commit: u64,
    in_flight: u64,
    in_add: u64,
    in_two_phase: u64,
    at_publish_rename: u64,
    killed_mid_run: u64,
}

impl Evidence {
    fn record(&mut self, round: &Round) {
        match round.visible {
            "last commit" => self.last_commit += 1,
            _ => self.in_flight += 1,
        }
        match round.crashed_in {
            Some(Op::Add) => self.in_add += 1,
            Some(Op::TwoPhaseCommit) => self.in_two_phase += 1,
            _ => {}
        }
        if round.crash_point.starts_with("rename pending_segments_") {
            self.at_publish_rename += 1;
        }
        if round.killed_mid_run {
            self.killed_mid_run += 1;
        }
    }

    fn require_power_loss_windows(&self) {
        let windows = [
            ("a crash visible as the last commit", self.last_commit),
            ("a crash visible as the in-flight commit", self.in_flight),
            ("a crash inside an automatic flush (add)", self.in_add),
            ("a crash inside a two-phase commit", self.in_two_phase),
            (
                "a crash at the segments_N publish rename",
                self.at_publish_rename,
            ),
        ];
        for (what, count) in windows {
            if count == 0 {
                println!("crash_fuzz: FAIL -- no round produced {what}");
                std::process::exit(1);
            }
        }
    }

    fn require_kills_mid_run(&self, rounds: u64) {
        if self.killed_mid_run * 2 < rounds {
            println!(
                "crash_fuzz: FAIL -- only {}/{rounds} kills landed mid-run",
                self.killed_mid_run
            );
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| -> Option<&str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let num_ops: usize = value("--ops").map_or(400, |v| v.parse().expect("--ops"));
    if let Some(seed) = value("--child") {
        let path = PathBuf::from(args.last().expect("child dir"));
        child(seed.parse().expect("seed"), &path, num_ops);
        return;
    }
    let (first, last) = match (value("--seed"), value("--seeds")) {
        (Some(s), _) => {
            let s: u64 = s.parse().expect("--seed");
            (s, s)
        }
        (None, Some(range)) => {
            let (a, b) = range.split_once("..").expect("--seeds A..B");
            let a: u64 = a.parse().expect("seed");
            let last = b
                .parse::<u64>()
                .expect("seed")
                .checked_sub(1)
                .filter(|&last| last >= a)
                .expect("--seeds A..B needs A < B");
            (a, last)
        }
        (None, None) => (0, 49),
    };
    let duration = value("--duration").map(|s| Duration::from_secs(s.parse().expect("--duration")));
    let kill = args.iter().any(|a| a == "--kill");
    let concurrent = args.iter().any(|a| a == "--concurrent");
    let java_cp = value("--java-cp");
    let base = value("--dir").map_or_else(
        || std::env::temp_dir().join(format!("crash-fuzz-{}", std::process::id())),
        PathBuf::from,
    );

    let started = Instant::now();
    let mut seed = first;
    let mut rounds = 0u64;
    let mut evidence = Evidence::default();
    loop {
        let result = if kill {
            kill_round(seed, &base, num_ops, java_cp)
        } else if concurrent {
            concurrent_round(seed, &base, num_ops, java_cp)
        } else {
            power_loss_round(seed, &base, num_ops, java_cp)
        };
        match result {
            Ok(round) => {
                println!("seed {seed}: ok -- {}", round.summary);
                evidence.record(&round);
            }
            Err(why) => {
                println!("seed {seed}: FAIL -- {why}");
                println!(
                    "replay: crash_fuzz --seed {seed} --ops {num_ops}{}",
                    if kill {
                        " --kill"
                    } else if concurrent {
                        " --concurrent"
                    } else {
                        ""
                    }
                );
                std::process::exit(1);
            }
        }
        rounds += 1;
        seed += 1;
        let done = match duration {
            Some(d) => started.elapsed() >= d,
            None => seed > last,
        };
        if done {
            break;
        }
    }
    let _ = std::fs::remove_dir_all(&base);
    // A fixed seed range long enough to expect every window must have hit
    // every window -- otherwise a change to the stream or the writer could
    // quietly stop exercising, say, the publish rename, and every round
    // would still pass.
    if duration.is_none() && rounds >= 100 && !kill && !concurrent {
        evidence.require_power_loss_windows();
    }
    if duration.is_none() && rounds >= 20 && kill {
        evidence.require_kills_mid_run(rounds);
    }
    println!(
        "crash_fuzz: {rounds} {} round(s) passed in {:.1}s",
        if kill {
            "kill -9"
        } else if concurrent {
            "concurrent power-loss"
        } else {
            "power-loss"
        },
        started.elapsed().as_secs_f64()
    );
}
