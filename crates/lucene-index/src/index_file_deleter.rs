//! Reference-counted reclamation of unreferenced index files: real Lucene's
//! `org.apache.lucene.index.IndexFileDeleter` plus the
//! `org.apache.lucene.util.FileDeleter` it delegates the counting to, and the
//! `IndexDeletionPolicy`/`KeepOnlyLastCommitDeletionPolicy` pair it consults.
//!
//! # Why a refcount and not "delete what the current commit doesn't name"
//!
//! A file can be named by more than one commit point -- every segment that
//! survives a commit is named by the commit before it and the commit after it,
//! and a deletion policy may deliberately keep older commits alive. So "is this
//! file still needed" is not a property of the newest commit; it is a count over
//! every live commit *plus* the writer's own in-memory, not-yet-committed view
//! (segments already flushed to disk but not yet published by a `segments_N`).
//! That is exactly what [`IndexFileDeleter`] counts.
//!
//! Two operations drive it, both ported verbatim in spirit from Java:
//!
//! - [`IndexFileDeleter::checkpoint`] with `is_commit == false` -- "the writer
//!   made a consistent change to its in-memory segment list": incRef the new
//!   view's files, decRef the previous non-commit view's files. This is what
//!   protects a segment that an automatic flush wrote but no commit references
//!   yet, and what reclaims it when a rollback drops it.
//! - [`IndexFileDeleter::checkpoint`] with `is_commit == true` -- "a
//!   `segments_N` was just published": incRef the commit's files *including the
//!   `segments_N` itself*, append a commit point, then let the
//!   [`DeletionPolicy`] decide which older commit points die. Every file whose
//!   count reaches zero is deleted right there.
//!
//! # What this port deliberately does not do
//!
//! **No Windows delete-on-close emulation.** Java's `FileDeleter.delete` carries
//! a `Constants.WINDOWS` branch that swallows `NoSuchFileException` because
//! Windows leaves a deleted-but-still-open file visible in directory listings in
//! a "pending delete" state, and `FSDirectory` keeps a `pendingDeletes` set plus
//! a `deletePendingFiles()` retry loop (called from
//! `IndexWriter.deletePendingFiles`) to drain it. This port targets Linux, where
//! `unlink` on an open file succeeds immediately and the name disappears at
//! once, so there is no pending-deletion state to model: `deletePendingFiles`,
//! `Directory.getPendingDeletions()` and the `WINDOWS` branch are all omitted on
//! purpose, not by oversight. A `NoSuchFileException`-equivalent from
//! [`lucene_store::directory::Directory::delete_file`] is therefore a real
//! error here, exactly as it is for Java on a non-Windows platform.
//!
//! **No `SnapshotDeletionPolicy`.** [`DeletionPolicy`] is an enum over Lucene's
//! two *stateless* built-in policies rather than a trait with one
//! implementation; a snapshotting policy needs a handle type this port has no
//! caller for yet. See [`DeletionPolicy`].
//!
//! **No `IndexCommit` object.** Java's `CommitPoint` extends the public
//! `IndexCommit` so a deletion policy (and `DirectoryReader.listCommits`) can
//! inspect a commit's user data and segment count. [`CommitPoint`] here carries
//! only what the refcounting needs -- generation, file list, and the
//! `segments_N` name -- because nothing in this port consumes an `IndexCommit`.

use std::collections::{HashMap, HashSet};

use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;

use crate::segment_info;
use crate::segment_infos::{self, SegmentCommitInfo, SegmentInfos};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    /// Java's `IllegalStateException("file \"...\" has refCount=0, which should
    /// never happen on init")`: a `segments_N` found in the directory that no
    /// commit-point scan claimed. Since the scan loads *every* `segments*` file
    /// it lists, this can only mean the listing changed underneath us.
    #[error(
        "index file deleter: commit file {0:?} has refCount 0 after the init scan, which should \
         never happen -- the directory listing changed while the deleter was initializing"
    )]
    UnreferencedCommitFile(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Port of `IndexDeletionPolicy`, restricted to Lucene's two stateless
/// built-ins.
///
/// Java models this as an abstract class with `onInit(List<IndexCommit>)` /
/// `onCommit(List<IndexCommit>)`, whose implementations call `IndexCommit
/// .delete()` on the commits they no longer want. The two shipped
/// implementations that need no extra state are
/// `KeepOnlyLastCommitDeletionPolicy` (the default: `onInit` delegates to
/// `onCommit`, which deletes every commit except the last) and
/// `NoDeletionPolicy` (keeps everything, used by
/// `IndexWriterConfig.setIndexDeletionPolicy(NoDeletionPolicy.INSTANCE)` and by
/// replication setups that manage commit lifetime themselves).
///
/// `SnapshotDeletionPolicy` and `PersistentSnapshotDeletionPolicy` are *not*
/// modelled: both hand the caller a `IndexCommit` snapshot handle that pins a
/// commit until released, which is a lifecycle this port has no caller for. An
/// enum with the two stateless policies is honest about what exists; a trait
/// with a single implementation would be a transliteration of a Java extension
/// point nothing here extends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeletionPolicy {
    /// `KeepOnlyLastCommitDeletionPolicy`, Lucene's default: after each commit,
    /// every commit point except the newest is dropped, and its files are
    /// decRef'd (and deleted if nothing else names them).
    #[default]
    KeepOnlyLastCommit,
    /// `NoDeletionPolicy.INSTANCE`: no commit point is ever dropped, so a file
    /// named by any commit -- however old -- is never deleted. Files that no
    /// commit ever named (an aborted flush, a rolled-back prepare) are still
    /// reclaimed, since those never had a commit point to begin with.
    KeepAll,
}

/// Java's `IndexFileDeleter.CommitPoint`, minus the `IndexCommit` surface (see
/// the module doc comment).
#[derive(Debug, Clone)]
struct CommitPoint {
    generation: i64,
    segments_file_name: String,
    files: Vec<String>,
}

/// Port of `IndexFileDeleter` + `org.apache.lucene.util.FileDeleter`.
///
/// Not thread-safe, matching Java (`FileDeleter`'s own doc comment says so; in
/// Java, `IndexFileDeleter` is guarded by the `IndexWriter` monitor). This port
/// takes `&mut self` for every mutating operation, so the borrow checker
/// enforces what Java enforces with an `assert locked()`.
pub struct IndexFileDeleter<'d> {
    dir: &'d dyn Directory,
    policy: DeletionPolicy,
    /// `FileDeleter.refCounts`. A name present with count 0 is Java's
    /// `initRefCount` state: "this file exists and the deleter knows about it,
    /// but nothing references it" -- the set `getUnrefedFiles` returns.
    ref_counts: HashMap<String, u32>,
    /// `IndexFileDeleter.commits`, kept sorted oldest-to-newest.
    commits: Vec<CommitPoint>,
    /// `IndexFileDeleter.lastFiles`: the files incRef'd by the most recent
    /// non-commit checkpoint, decRef'd by the next one.
    last_files: Vec<String>,
    /// `SegmentInfo.files` per segment name, together with the segment id the
    /// entry was recorded for.
    ///
    /// Java gets this for free because `SegmentCommitInfo` holds a live
    /// `SegmentInfo` with its file set in memory; this port's
    /// [`SegmentCommitInfo`] deliberately does not own the parsed `.si`. Two
    /// things fill this map, and between them the deleter never parses a `.si`
    /// the writer itself produced:
    ///
    /// - [`record_segment_files`](Self::record_segment_files), called by
    ///   `IndexWriter` the moment it seals a flushed segment, straight off the
    ///   in-memory `SegmentInfo` -- which is exactly what Java reads.
    /// - [`ensure_si_files`](Self::ensure_si_files)' fallback parse, for a segment
    ///   this process did not write: one opened at `IndexFileDeleter::open`,
    ///   or one another writer left behind.
    ///
    /// The segment id is stored rather than made part of the key so a lookup
    /// needs no allocation; a mismatch means a *different* segment reused the
    /// name, which cannot happen within one index but is cheap to rule out and
    /// would otherwise be a silently wrong file set.
    si_files: HashMap<String, ([u8; ID_LENGTH], Vec<String>)>,
}

impl<'d> IndexFileDeleter<'d> {
    /// Port of the `IndexFileDeleter` constructor: walk the directory, refcount
    /// every commit point it finds, delete everything left unreferenced, then
    /// let the policy prune commits and protect `current`.
    ///
    /// The unreferenced files this removes at open are precisely the orphans a
    /// previous session leaked: a `pending_segments_N` from a prepare that never
    /// finished, the segment files of a flush that was never committed, and the
    /// files of any commit the policy drops here.
    ///
    /// Only files that look like Lucene index files are ever considered --
    /// `IndexFileNames.CODEC_FILE_PATTERN` (`_<name>[_<suffix>].<ext>`),
    /// `segments*` and `pending_segments*` (see [`is_index_file_name`]). A
    /// caller's own unrelated file in the same directory is never touched, which
    /// is the same guard Java relies on.
    pub fn open(
        dir: &'d dyn Directory,
        current: &SegmentInfos,
        policy: DeletionPolicy,
    ) -> Result<Self> {
        let mut deleter = IndexFileDeleter {
            dir,
            policy,
            ref_counts: HashMap::new(),
            commits: Vec::new(),
            last_files: Vec::new(),
            si_files: HashMap::new(),
        };

        let files = dir.list_all()?;

        for file_name in &files {
            if !is_index_file_name(file_name) {
                continue;
            }
            // Java's `fileDeleter.initRefCount`: the file is now known, at
            // count 0, so `getUnrefedFiles` can find it if nothing increfs it.
            deleter.ref_counts.entry(file_name.clone()).or_insert(0);

            if !is_segments_file_name(file_name) {
                continue;
            }
            // A commit: load it and incRef everything it names, including its
            // own `segments_N`.
            let generation =
                lucene_store::directory::generation_from_segments_file_name(file_name)?;
            let bytes = dir.open(file_name)?.to_vec();
            let sis = segment_infos::parse(&bytes, generation)?;
            let commit_files = deleter.commit_files(&sis, true)?;
            deleter.inc_ref_all(&commit_files);
            deleter.commits.push(CommitPoint {
                generation,
                segments_file_name: file_name.clone(),
                files: commit_files,
            });
        }

        // Java keeps `commits` in ascending generation order
        // (`CollectionUtil.timSort(commits)`); `Directory.list_all` sorts
        // lexicographically, under which `segments_a` precedes `segments_9`.
        deleter.commits.sort_by_key(|c| c.generation);

        // Now delete anything still at count 0: files no commit references.
        let unrefed = deleter.unrefed_files();
        for name in &unrefed {
            if is_segments_file_name(name) {
                return Err(Error::UnreferencedCommitFile(name.clone()));
            }
        }
        deleter.delete_files(&unrefed)?;

        // `policy.onInit(commits)` then `deleteCommits()`.
        deleter.apply_policy()?;

        // "Always protect the incoming segmentInfos since sometimes it may not
        // be the most recent commit."
        // Note this is deliberately *not* asserted to have referenced a
        // `segments_N`: a never-committed index's `SegmentInfos` carries
        // generation 0 in this port (Java models it as -1 / a null
        // `getSegmentsFileName()`), and there is no commit file for it yet.
        deleter.checkpoint(current, false)?;

        Ok(deleter)
    }

    /// Port of `IndexFileDeleter.inflateGens`, in full: both the commit-wide
    /// counters and the per-segment `nextWrite*Gen` half.
    ///
    /// Java's purpose: after a crash, the directory can hold files with higher
    /// generations/segment names than the last *committed* `SegmentInfos`
    /// records. Writing at those same generations again would overwrite a file
    /// that a still-open reader (or a not-yet-reclaimed orphan) occupies, so
    /// every counter is pushed past the highest value seen on disk.
    ///
    /// `infos` is updated in place, exactly as Java's
    /// `inflateGens(SegmentInfos, ...)` mutates the `SegmentInfos` it is given:
    /// `generation` and `counter` on the commit itself, plus each segment's
    /// [`SegmentCommitInfo::next_write_del_gen`] /
    /// `next_write_field_infos_gen` / `next_write_doc_values_gen`. Java's own
    /// comment about the last three is worth repeating, because it explains an
    /// otherwise surprising outcome: the per-segment maximum is the **union**
    /// of the live-docs, field-infos and doc-values generations, "since it
    /// means DV updates will suddenly write to the next gen after live docs'
    /// gen, for example, but we don't have the APIs to ask the codec which
    /// file is which". This port inherits that, deliberately, rather than
    /// inventing a per-group split real Lucene does not have.
    pub fn inflate_gens(files: &[String], infos: &mut SegmentInfos) {
        let mut max_segment_gen = i64::MIN;
        let mut max_segment_name = i64::MIN;
        // Java's `maxPerSegmentGen`: segment name -> highest generation seen in
        // any of that segment's file names.
        let mut max_per_segment_gen: std::collections::HashMap<&str, i64> =
            std::collections::HashMap::new();

        for file_name in files {
            if is_segments_file_name(file_name) {
                if let Some(g) = usable_generation(
                    lucene_store::directory::generation_from_segments_file_name(file_name).ok(),
                ) {
                    max_segment_gen = max_segment_gen.max(g);
                }
            } else if let Some(rest) = file_name.strip_prefix("pending_segments") {
                // Java: `generationFromSegmentsFileName(fileName.substring(8))`
                // -- i.e. re-read the tail as a plain `segments...` name.
                if let Some(g) = usable_generation(
                    lucene_store::directory::generation_from_segments_file_name(&format!(
                        "segments{rest}"
                    ))
                    .ok(),
                ) {
                    max_segment_gen = max_segment_gen.max(g);
                }
            } else if is_index_file_name(file_name) {
                if file_name.to_ascii_lowercase().ends_with(".tmp") {
                    continue;
                }
                let segment_name = parse_segment_name(file_name);
                if let Some(n) = usable_generation(
                    segment_name
                        .strip_prefix('_')
                        .and_then(lucene_util::base36::from_base36),
                ) {
                    max_segment_name = max_segment_name.max(n);
                }
                let entry = max_per_segment_gen.entry(segment_name).or_insert(0);
                *entry = (*entry).max(parse_generation(file_name));
            }
        }

        infos.generation = infos.generation.max(max_segment_gen);
        // ARITH: `usable_generation` caps `max_segment_name` at
        // `MAX_GENERATION` (`i64::MAX / 2`), so `+ 1` has 2^62 of headroom.
        // The "no segment file at all" start value is `i64::MIN`, where `+ 1`
        // is equally safe (only `- 1` would wrap) and stays below any real
        // `counter`, which is what the `max` below wants.
        #[allow(clippy::arithmetic_side_effects)]
        let inflated_counter = max_segment_name + 1;
        infos.counter = infos.counter.max(inflated_counter);

        // The per-segment half. Java only ever *raises* each counter
        // (`if (info.getNextWriteDelGen() < genLong + 1)`), so a segment whose
        // recorded generation already exceeds anything on disk is left alone.
        for sci in &mut infos.segments {
            let gen = max_per_segment_gen
                .get(sci.segment_name.as_str())
                .copied()
                .unwrap_or(0);
            // ARITH: `parse_generation` only ever yields `0..=MAX_GENERATION`
            // (`usable_generation` maps everything else to "no generation"), so
            // `+ 1` cannot overflow. Hoisted out of the three comparisons
            // below, which each used it twice.
            #[allow(clippy::arithmetic_side_effects)]
            let next = gen + 1;
            if sci.next_write_del_gen() < next {
                sci.set_next_write_del_gen(next);
            }
            if sci.next_write_field_infos_gen() < next {
                sci.set_next_write_field_infos_gen(next);
            }
            if sci.next_write_doc_values_gen() < next {
                sci.set_next_write_doc_values_gen(next);
            }
        }
    }

    /// Port of `IndexFileDeleter.checkpoint(SegmentInfos, boolean)`.
    ///
    /// `is_commit == false`: the writer changed its in-memory segment list (a
    /// flush landed, a rollback dropped one, `deleteAll` cleared them all).
    /// IncRef the new view, decRef the previous view. Nothing about commits
    /// changes.
    ///
    /// `is_commit == true`: a `segments_N` was just published. IncRef the
    /// commit's files *including that `segments_N`*, append a commit point, and
    /// give the [`DeletionPolicy`] its chance to drop older commits -- whose
    /// files are then decRef'd, and deleted when nothing else names them.
    pub fn checkpoint(&mut self, infos: &SegmentInfos, is_commit: bool) -> Result<()> {
        let files = self.commit_files(infos, is_commit)?;
        self.inc_ref_all(&files);

        if is_commit {
            let generation = infos.generation;
            let segments_file_name = lucene_store::directory::segments_file_name(generation)
                .unwrap_or_else(|| "segments".to_string());
            self.commits.push(CommitPoint {
                generation,
                segments_file_name,
                files,
            });
            self.commits.sort_by_key(|c| c.generation);
            self.apply_policy()?;
        } else {
            let previous = std::mem::take(&mut self.last_files);
            self.dec_ref_all(&previous)?;
            self.last_files = files;
        }
        Ok(())
    }

    /// `IndexWriterConfig.setIndexDeletionPolicy` + `IndexFileDeleter.revisitPolicy()`:
    /// switch policy and apply it immediately, so tightening from
    /// [`DeletionPolicy::KeepAll`] back to
    /// [`DeletionPolicy::KeepOnlyLastCommit`] reclaims the commits that were
    /// being held rather than waiting for the next one.
    pub fn set_policy(&mut self, policy: DeletionPolicy) -> Result<()> {
        self.policy = policy;
        self.apply_policy()
    }

    /// Port of `IndexFileDeleter.refresh()`: re-list the directory and delete
    /// every index-looking file the deleter does not currently hold a reference
    /// to. Java calls this from `rollbackInternal` -- after an abort there may be
    /// files on disk that no checkpoint ever saw, because they were written by
    /// the very operation that failed.
    ///
    /// Unlike Java's, this also reclaims a leftover `pending_segments_N`, for the
    /// same reason Java's does: the pending file is never refcounted, so a
    /// rollback is the one moment it is provably dead.
    pub fn refresh(&mut self) -> Result<()> {
        let files = self.dir.list_all()?;
        let to_delete: Vec<String> = files
            .into_iter()
            .filter(|name| is_index_file_name(name) && !self.exists(name))
            .collect();
        self.delete_files(&to_delete)
    }

    /// Port of `IndexFileDeleter.deleteNewFiles(Collection<String>)`: delete
    /// the named files, but only those nothing has incRef'd yet. Java's
    /// `IndexWriter` calls it after a `DocumentsWriterPerThread.abort()` and
    /// after a merge that threw, to drop exactly the files that operation
    /// created.
    pub fn delete_new_files(&mut self, files: &[String]) -> Result<()> {
        let to_delete: Vec<String> = files
            .iter()
            .filter(|name| !self.exists(name))
            .cloned()
            .collect();
        self.delete_files(&to_delete)
    }

    /// `FileDeleter.getRefCount`: 0 for a file the deleter has never seen.
    pub fn ref_count(&self, name: &str) -> u32 {
        self.ref_counts.get(name).copied().unwrap_or(0)
    }

    /// `FileDeleter.exists`: known *and* referenced. A file at count 0 is known
    /// but dead.
    pub fn exists(&self, name: &str) -> bool {
        self.ref_count(name) > 0
    }

    /// Number of live commit points (`IndexFileDeleter.commits.size()`). One
    /// under [`DeletionPolicy::KeepOnlyLastCommit`] once anything has committed.
    pub fn commit_count(&self) -> usize {
        self.commits.len()
    }

    /// The `segments_N` names of every live commit point, oldest first.
    pub fn commit_file_names(&self) -> Vec<&str> {
        self.commits
            .iter()
            .map(|c| c.segments_file_name.as_str())
            .collect()
    }

    /// Port of `SegmentInfos.files(boolean includeSegmentsFile)`: every file
    /// named by `infos`, optionally including the `segments_N` itself.
    ///
    /// Each segment contributes [`SegmentCommitInfo::files`] -- its `.si`-declared
    /// files plus the `.liv`, field-infos-generation and doc-values-update files
    /// only the commit entry knows about. The `.si` parse is cached per
    /// `(segment_name, segment_id)`; see [`Self::si_files`].
    fn commit_files(
        &mut self,
        infos: &SegmentInfos,
        include_segments_file: bool,
    ) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        if include_segments_file {
            if let Some(name) = lucene_store::directory::segments_file_name(infos.generation) {
                if seen.insert(name.clone()) {
                    out.push(name);
                }
            }
        }
        // Two passes, deliberately: filling the cache needs `&mut self` and
        // reading it needs a borrow, and the two cannot overlap. Cloning each
        // segment's file list to collapse them into one pass is what this
        // costs -- one `Vec<String>` per segment per checkpoint, which for a
        // hundred-segment index is a hundred allocations on a path that runs
        // twice per commit and reads nothing.
        for sci in &infos.segments {
            self.ensure_si_files(sci)?;
        }
        for sci in &infos.segments {
            let (_, si_files) = self
                .si_files
                .get(sci.segment_name.as_str())
                .expect("just ensured above");
            for f in sci.files(si_files) {
                if seen.insert(f.clone()) {
                    out.push(f);
                }
            }
        }
        Ok(out)
    }

    /// Records a segment's `.si` file list from an in-memory `SegmentInfo`,
    /// so no checkpoint ever has to open and parse the file.
    ///
    /// This is the port of what Java gets structurally: `SegmentCommitInfo`
    /// holds its `SegmentInfo`, so `SegmentInfos.files()` reads the file set
    /// out of memory. `IndexWriter` calls this as it seals a flushed segment,
    /// with the same `SegmentInfo` it just encoded into the `.si`
    /// ([`crate::segment_writer::FlushedSegment::info`]) -- so the bytes on
    /// disk and the entry here are the same list by construction, not by
    /// agreement between a writer and a parser.
    ///
    /// Idempotent, and safe to call for a segment already recorded: the entry
    /// is replaced only when the id matches, so a stale name cannot shadow a
    /// different segment's files.
    pub fn record_segment_files(&mut self, sci: &SegmentCommitInfo, si_files: &[String]) {
        let files = Self::with_self_listing(&sci.segment_name, si_files.to_vec());
        self.si_files
            .insert(sci.segment_name.clone(), (sci.segment_id, files));
    }

    /// Measurement-only: drop every recorded file set, so the next checkpoint
    /// has to open and parse each `.si` again.
    ///
    /// This is the pre-`c43-final-cleanup` state of the deleter, and it exists
    /// so `examples/deleter_checkpoint.rs` can time both arms in **one process
    /// from one build** -- the only shape of A/B this project trusts (see
    /// `docs/sweep/m2/c24-arith-codecs.md` on criterion and `c42`'s stale
    /// binary). Correct but pointless in production: the cache refills itself
    /// from disk.
    #[doc(hidden)]
    pub fn forget_segment_files(&mut self) {
        self.si_files.clear();
    }

    /// Fills [`Self::si_files`] for `sci`, parsing its `.si` only if
    /// [`record_segment_files`](Self::record_segment_files) has not already
    /// recorded it from memory.
    fn ensure_si_files(&mut self, sci: &SegmentCommitInfo) -> Result<()> {
        if let Some((id, _)) = self.si_files.get(sci.segment_name.as_str()) {
            if *id == sci.segment_id {
                return Ok(());
            }
        }
        let name = format!("{}.si", sci.segment_name);
        let bytes = self.dir.open(&name)?.to_vec();
        let si = segment_info::parse(&bytes, &sci.segment_id)?;
        let files = Self::with_self_listing(&sci.segment_name, si.files);
        self.si_files
            .insert(sci.segment_name.clone(), (sci.segment_id, files));
        Ok(())
    }

    /// `Lucene99SegmentInfoFormat.write` calls `si.addFile(fileName)` for the
    /// `.si` itself before encoding, so a correctly written `.si` already lists
    /// itself. Older segments this port wrote did not; adding it here keeps the
    /// deleter from reclaiming the file that names all the others.
    fn with_self_listing(segment_name: &str, mut files: Vec<String>) -> Vec<String> {
        let name = format!("{segment_name}.si");
        if !files.iter().any(|f| f == &name) {
            files.push(name);
        }
        files
    }

    fn inc_ref_all(&mut self, files: &[String]) {
        for name in files {
            let count = self.ref_counts.entry(name.clone()).or_insert(0);
            // Saturating, not checked: 2^32 live references to one file is
            // unreachable (each is a commit point or a checkpoint holding the
            // name in memory), and a saturated count only ever *keeps* a file
            // alive -- the safe direction. A wrap to 0 would delete a file the
            // current commit still names.
            *count = count.saturating_add(1);
        }
    }

    /// `FileDeleter.decRef(Collection<String>)`: decrement each, collect the
    /// ones that hit zero, then delete them all in one pass (`segments_N` first,
    /// see [`Self::delete_files`]).
    fn dec_ref_all(&mut self, files: &[String]) -> Result<()> {
        let mut to_delete: Vec<String> = Vec::new();
        for name in files {
            if let Some(count) = self.ref_counts.get_mut(name) {
                debug_assert!(*count > 0, "decRef below zero for {name:?}");
                // Java's `RefCount.DecRef` asserts `count > 0` and, with
                // assertions off, lets the count go *negative* -- after which
                // the file can never reach 0 again and is leaked. `u32` here
                // would instead wrap to `u32::MAX`, the same leak by a
                // different route. Saturating keeps the count at 0, which is
                // exactly the "known, unreferenced" state `open`'s init sweep
                // reclaims, so the file is deleted rather than leaked. Only a
                // caller bug can reach it either way (the `debug_assert` is
                // what catches that in tests).
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.ref_counts.remove(name);
                    if !to_delete.contains(name) {
                        to_delete.push(name.clone());
                    }
                }
            }
        }
        self.delete_files(&to_delete)
    }

    fn unrefed_files(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .ref_counts
            .iter()
            .filter(|(_, count)| **count == 0)
            .map(|(name, _)| name.clone())
            .collect();
        out.sort();
        out
    }

    /// `KeepOnlyLastCommitDeletionPolicy.onCommit` + `deleteCommits()`:
    /// everything except the newest commit point dies, and its files are
    /// decRef'd.
    fn apply_policy(&mut self) -> Result<()> {
        if self.policy == DeletionPolicy::KeepAll || self.commits.len() < 2 {
            return Ok(());
        }
        // ARITH: the early return above leaves `self.commits.len() >= 2`.
        #[allow(clippy::arithmetic_side_effects)]
        let keep_from = self.commits.len() - 1;
        let doomed: Vec<CommitPoint> = self.commits.drain(..keep_from).collect();
        for commit in doomed {
            self.dec_ref_all(&commit.files)?;
        }
        Ok(())
    }

    /// `FileDeleter.delete(Collection<String>)`'s two passes: every `segments_N`
    /// first, then everything else.
    ///
    /// The ordering is not cosmetic. A crash between the two passes must never
    /// leave a commit file whose segments are already gone -- that is a corrupt
    /// index. Removing the commit file first makes the worst case an index with
    /// orphaned segment files, which is exactly the safe direction.
    fn delete_files(&mut self, names: &[String]) -> Result<()> {
        for name in names {
            if is_segments_file_name(name) {
                self.ref_counts.remove(name);
                self.dir.delete_file(name)?;
            }
        }
        for name in names {
            if !is_segments_file_name(name) {
                self.ref_counts.remove(name);
                self.dir.delete_file(name)?;
            }
        }
        Ok(())
    }
}

/// Port of `IndexFileNames.CODEC_FILE_PATTERN` (`_[a-z0-9]+(_.*)?\..*`) plus
/// the `segments`/`pending_segments` prefixes Java tests alongside it, and the
/// `write.lock` exclusion.
///
/// Written as a hand-rolled matcher rather than a regex dependency: the pattern
/// is three checks (leading `_`, a lowercase-alphanumeric run, then a `.`
/// somewhere after it) and this runs once per directory entry per checkpoint.
pub fn is_index_file_name(name: &str) -> bool {
    if name.ends_with("write.lock") {
        return false;
    }
    if name.starts_with("segments") || name.starts_with("pending_segments") {
        return true;
    }
    let Some(rest) = name.strip_prefix('_') else {
        return false;
    };
    // `[a-z0-9]+`
    let base_len = rest
        .bytes()
        .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        .count();
    if base_len == 0 {
        return false;
    }
    // `(_.*)?\..*` -- the remainder must contain a `.`, and if it starts with
    // anything else it has to be `_`.
    let tail = &rest[base_len..];
    match tail.as_bytes().first() {
        Some(b'.') => true,
        Some(b'_') => tail.contains('.'),
        _ => false,
    }
}

/// The `fileName.startsWith(IndexFileNames.SEGMENTS)` test. Note this is
/// deliberately *false* for `pending_segments_N`: Java relies on that exact
/// asymmetry (a pending file is never refcounted, never treated as a commit,
/// and reclaimed only by `refresh()` or the init sweep).
fn is_segments_file_name(name: &str) -> bool {
    name.starts_with("segments") && name != "segments.gen"
}

/// Port of `IndexFileNames.parseSegmentName`: everything up to the first `_`
/// after the leading one, or up to the extension dot.
/// `IndexFileNames.parseGeneration`: the generation embedded in a segment
/// file's name, or `0` when it carries none.
///
/// Java strips the extension, drops the leading `_`, splits on `_`, and reads
/// part 1 in base 36 -- covering exactly its four documented shapes:
/// `segment.ext` (no gen), `segment_gen.ext`, `segment_codec_suffix.ext` and
/// `segment_gen_codec_suffix.ext`. Two parts or four means part 1 is the
/// generation; anything else means there is none. Note the consequence Java
/// lives with too: `_0_Lucene104_0.tim` has three parts, so it contributes no
/// generation, while `_0_1_Lucene104_0.tim` has four and contributes `1`.
fn parse_generation(file_name: &str) -> i64 {
    let stem = match file_name.rfind('.') {
        Some(dot) => &file_name[..dot],
        None => file_name,
    };
    let Some(rest) = stem.strip_prefix('_') else {
        return 0;
    };
    let parts: Vec<&str> = rest.split('_').collect();
    if parts.len() == 2 || parts.len() == 4 {
        usable_generation(lucene_util::base36::from_base36(parts[1])).unwrap_or(0)
    } else {
        0
    }
}

/// Filters a generation parsed out of a **file name** down to one this port can
/// safely step past.
///
/// Java reads these with `Long.parseLong(..., 36)` and catches only
/// `NumberFormatException` -- its own comment calls the leftovers "trash file:
/// we have to handle this since codec regex is only so good". A name that
/// parses but carries an absurd value (`_0_1y2p0ij32e8e7.liv` is a perfectly
/// well-formed base-36 `i64::MAX`) is the same kind of trash, and Java gets
/// away with taking it at face value because `genLong + 1` merely wraps in
/// Java. Here it **panics** in a debug build, and in a release build it would
/// hand `nextWriteDelGen` a wrapped negative -- so the value is rejected in the
/// same breath as an unparsable one, at the one place both are produced.
///
/// A negative generation goes the same way: Lucene's own
/// `IndexFileNames.fileNameFromGeneration` asserts `gen > 0` before it emits a
/// suffix, so nothing that wrote the directory can have meant it.
///
/// The range is deliberately **exclusive** of
/// [`segment_infos::MAX_GENERATION`]. Everything this function feeds is a
/// value the caller immediately adds 1 to and stores into a field that gets
/// *serialized* -- `infos.counter`, and each segment's `nextWrite*Gen`, which
/// `advance_*_gen` turns into the `delGen`/`fieldInfosGen`/`docValuesGen`
/// written into the next `segments_N`. Accepting the cap itself would let a
/// trash file name named `_<base36(MAX_GENERATION)>.si` manufacture a commit
/// `segment_infos::parse` refuses, i.e. an index this port wrote and can no
/// longer open. One below the cap keeps the whole round trip closed.
fn usable_generation(parsed: Option<i64>) -> Option<i64> {
    parsed.filter(|g| (0..segment_infos::MAX_GENERATION).contains(g))
}

/// [`parse_generation`], reachable from other modules' tests so a
/// generational file name can be checked to round-trip through it. Not part of
/// the public API: `IndexFileNames.parseGeneration` is package-private in Java
/// too.
#[cfg(test)]
pub(crate) fn parse_generation_for_test(file_name: &str) -> i64 {
    parse_generation(file_name)
}

/// Both slices here are UTF-8 boundary-safe on a name that came out of a
/// directory listing: this is only ever called on a name [`is_index_file_name`]
/// accepted and the `segments`/`pending_segments` branches did not claim, which
/// means it starts with the ASCII `_` (or is the fixed, all-ASCII
/// `segments.gen`), so byte 1 is a character boundary; and `find` returns a
/// boundary by construction.
fn parse_segment_name(file_name: &str) -> &str {
    // ARITH: `i` is a byte offset *within* `file_name[1..]`, so `i + 1` is at
    // most `file_name.len()` -- an in-memory length, hence at most
    // `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let idx = file_name[1..]
        .find(['_', '.'])
        .map(|i| i + 1)
        .unwrap_or(file_name.len());
    &file_name[..idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_info::{self, LuceneVersion, SegmentInfo};
    use lucene_store::data_output::DataOutput;
    use lucene_store::directory::FsDirectory;

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir(tag: &str) -> TempDir {
        TempDir::new(&format!("file-deleter-{tag}"))
    }

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn id(seed: u8) -> [u8; ID_LENGTH] {
        [seed; ID_LENGTH]
    }

    fn write(dir: &FsDirectory, name: &str, bytes: &[u8]) {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    /// Writes a minimal but real `.si` for `name` listing itself plus
    /// `extra_files`, and returns the matching [`SegmentCommitInfo`]. Those
    /// extra files are created empty, so every name the deleter refcounts is a
    /// name that actually exists.
    fn seed_segment(
        dir: &FsDirectory,
        name: &str,
        seed: u8,
        extra_files: &[&str],
    ) -> SegmentCommitInfo {
        let mut files: Vec<String> = vec![format!("{name}.si")];
        for f in extra_files {
            files.push((*f).to_string());
            write(dir, f, b"x");
        }
        let si = SegmentInfo {
            id: id(seed),
            version: version(),
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files,
            attributes: vec![],
            index_sort: None,
        };
        write(dir, &format!("{name}.si"), &segment_info::write(&si, ""));
        SegmentCommitInfo {
            segment_name: name.to_string(),
            segment_id: id(seed),
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        }
    }

    fn infos(generation: i64, segments: Vec<SegmentCommitInfo>) -> SegmentInfos {
        SegmentInfos {
            id: id(0xAA),
            generation,
            format_version: segment_infos::VERSION_86,
            lucene_version: segment_infos::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: generation,
            counter: 10,
            min_segment_lucene_version: None,
            segments,
            user_data: vec![],
        }
    }

    fn listing(dir: &FsDirectory) -> Vec<String> {
        let mut v = dir.list_all().unwrap();
        v.sort();
        v
    }

    #[test]
    fn codec_file_pattern_accepts_lucene_names_and_rejects_everything_else() {
        for name in [
            "_0.si",
            "_0.fdt",
            "_1a.cfs",
            "_0_1.liv",
            "_0_Lucene104_0.tim",
            "_ff_Lucene90_0.dvd",
            "segments",
            "segments_2",
            "segments.gen",
            "pending_segments_3",
        ] {
            assert!(is_index_file_name(name), "{name} should be an index file");
        }
        for name in [
            "write.lock",
            "some.write.lock",
            "README.txt",
            "_0",
            "_.si",
            "_A.si",
            "_0_nodot",
            "notours",
            "",
        ] {
            assert!(
                !is_index_file_name(name),
                "{name} should not be an index file"
            );
        }
    }

    #[test]
    fn segments_file_name_test_excludes_the_legacy_pointer_file() {
        assert!(is_segments_file_name("segments"));
        assert!(is_segments_file_name("segments_9"));
        // Lucene 4.0 dropped `segments.gen`; it is not a commit, so the deleter
        // treats it as an ordinary unreferenced index file (and reclaims it)
        // rather than trying to read a commit out of it.
        assert!(!is_segments_file_name("segments.gen"));
        assert!(!is_segments_file_name("pending_segments_9"));
    }

    #[test]
    fn parse_segment_name_stops_at_the_generation_or_the_extension() {
        assert_eq!(parse_segment_name("_0.si"), "_0");
        assert_eq!(parse_segment_name("_1a_1.liv"), "_1a");
        assert_eq!(parse_segment_name("_0_Lucene104_0.tim"), "_0");
        assert_eq!(parse_segment_name("_0"), "_0");
    }

    fn named_sci(segment_name: &str, del_gen: i64) -> SegmentCommitInfo {
        SegmentCommitInfo {
            segment_name: segment_name.to_string(),
            segment_id: id(0x11),
            codec_name: "Lucene104".to_string(),
            del_gen,
            ..Default::default()
        }
    }

    #[test]
    fn inflate_gens_pushes_both_counters_past_everything_on_disk() {
        let mut current = infos(2, vec![]);
        let files: Vec<String> = [
            "segments_2",
            "pending_segments_7",
            "_0.si",
            "_a.fdt",
            "_zz.tmp",
            "trash",
            "write.lock",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        IndexFileDeleter::inflate_gens(&files, &mut current);
        assert_eq!(
            current.generation, 7,
            "the leftover pending commit must win"
        );
        // `_a` is 10 in base 36, so the next name has to be at least 11. `_zz`
        // is skipped: Java refuses to read a generation out of a `.tmp` name.
        assert_eq!(current.counter, 11);
    }

    #[test]
    fn inflate_gens_pushes_a_segments_next_write_gens_past_a_crashed_sessions_liv() {
        // The exact crash this guards: the commit records `_0` at delGen 1, but
        // a `_0_3.liv` from a session that died before committing is still on
        // disk. Without the per-segment half the next delete would derive
        // `del_gen + 1 == 2`, then `3` -- landing straight on the orphan.
        let mut current = infos(1, vec![named_sci("_0", 1)]);
        assert_eq!(current.segments[0].next_write_del_gen(), 2);

        let files: Vec<String> = ["segments_1", "_0.si", "_0_1.liv", "_0_3.liv"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        IndexFileDeleter::inflate_gens(&files, &mut current);

        assert_eq!(current.segments[0].next_write_del_gen(), 4);
        // Java's own comment: the maximum is the *union* of the live-docs,
        // field-infos and doc-values generations, so all three move together.
        assert_eq!(current.segments[0].next_write_field_infos_gen(), 4);
        assert_eq!(current.segments[0].next_write_doc_values_gen(), 4);
        // `del_gen` itself is untouched -- only the *next write* moves.
        assert_eq!(current.segments[0].del_gen, 1);
    }

    #[test]
    fn inflate_gens_never_lowers_a_segments_next_write_gen() {
        // A segment already at delGen 5 with nothing above `_0_5.liv` on disk
        // keeps its derived next-write generation of 6.
        let mut current = infos(1, vec![named_sci("_0", 5)]);
        let files: Vec<String> = ["segments_1", "_0.si", "_0_5.liv"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        IndexFileDeleter::inflate_gens(&files, &mut current);
        assert_eq!(current.segments[0].next_write_del_gen(), 6);
    }

    #[test]
    fn inflate_gens_leaves_a_segment_with_no_files_on_disk_at_its_derived_gens() {
        let mut current = infos(1, vec![named_sci("_9", -1)]);
        IndexFileDeleter::inflate_gens(&["segments_1".to_string()], &mut current);
        assert_eq!(current.segments[0].next_write_del_gen(), 1);
    }

    /// A generation is parsed out of a **file name**, in base 36, so any
    /// process that can create a file in the index directory can hand
    /// `inflate_gens` an `i64::MAX` — `1y2p0ij32e8e7` is `Long.MAX_VALUE` in
    /// base 36, and it fits Lucene's own `_<seg>_<gen>.<ext>` shape exactly.
    /// Java's `genLong + 1` merely wraps; the bare `+ 1` here **panicked** in a
    /// debug build, taking the JVM down at index-open time. Such a name is
    /// trash, in the same sense Java's own comment uses for an unparsable one,
    /// so it contributes nothing rather than being followed.
    #[test]
    fn a_trash_file_name_claiming_an_absurd_generation_is_ignored_not_followed() {
        let cap = lucene_util::base36::to_base36(segment_infos::MAX_GENERATION);
        let mut current = infos(1, vec![named_sci("_0", 1)]);
        let files: Vec<String> = [
            "segments_1".to_string(),
            "_0.si".to_string(),
            "_0_2.liv".to_string(),
            // `i64::MAX` and `i64::MIN` in base 36.
            "_0_1y2p0ij32e8e7.liv".to_string(),
            "_0_-1y2p0ij32e8e8.liv".to_string(),
            // ...and the boundary, which is the one that actually mattered:
            // `MAX_GENERATION` is small enough to survive the `+ 1` in
            // `inflate_gens` but produces a `nextWriteDelGen` of
            // `MAX_GENERATION + 1`, which `advance_del_gen` would then write
            // into `segments_N` for `segment_infos::parse` to refuse. An
            // index this port wrote and could no longer open.
            format!("_0_{cap}.liv"),
        ]
        .to_vec();
        IndexFileDeleter::inflate_gens(&files, &mut current);

        // The one *usable* generation on disk still wins: 2 -> next write 3.
        assert_eq!(current.segments[0].next_write_del_gen(), 3);
        assert_eq!(current.segments[0].next_write_field_infos_gen(), 3);
        assert_eq!(current.segments[0].next_write_doc_values_gen(), 3);
    }

    /// The same shape for the commit-wide counters: a `segments_<absurd>` and
    /// a `_<absurd>.si` in the listing must not drive `generation`/`counter`
    /// to a value the next commit's `+ 1` cannot represent.
    #[test]
    fn trash_segment_and_commit_names_do_not_inflate_the_commit_counters() {
        let cap = lucene_util::base36::to_base36(segment_infos::MAX_GENERATION);
        let mut current = infos(1, vec![named_sci("_0", -1)]);
        let files: Vec<String> = [
            "segments_1".to_string(),
            "segments_1y2p0ij32e8e7".to_string(),
            "_0.si".to_string(),
            "_1y2p0ij32e8e7.si".to_string(),
            // The boundary: `_<base36(MAX_GENERATION)>.si` drove
            // `infos.counter` to `MAX_GENERATION + 1`, which
            // `segment_infos::write` serializes and `segment_infos::parse`
            // then refuses.
            format!("_{cap}.si"),
            format!("segments_{cap}"),
        ]
        .to_vec();
        IndexFileDeleter::inflate_gens(&files, &mut current);

        assert_eq!(current.generation, 1);
        assert!(
            current.counter <= segment_infos::MAX_GENERATION,
            "counter {} escaped the cap",
            current.counter
        );
    }

    #[test]
    fn parse_generation_reads_javas_four_file_name_shapes() {
        // `segment.ext` and `segment_codec_suffix.ext` carry no generation.
        assert_eq!(parse_generation("_0.si"), 0);
        assert_eq!(parse_generation("_0_Lucene104_0.tim"), 0);
        // `segment_gen.ext` and `segment_gen_codec_suffix.ext` do, in base 36.
        assert_eq!(parse_generation("_0_3.liv"), 3);
        assert_eq!(parse_generation("_0_a.liv"), 10);
        assert_eq!(parse_generation("_0_2_Lucene90_0.dvd"), 2);
        // Trash tails read as no generation rather than erroring, matching
        // Java's `catch (NumberFormatException)`.
        assert_eq!(parse_generation("_0_!!.liv"), 0);
        assert_eq!(parse_generation("write.lock"), 0);
        // The generational `.fnm` and `.dvm`/`.dvd`/`.dvs` a doc-values
        // update writes (`crate::field_updates`), which is where the
        // four-part `segment_gen_codec_suffix.ext` shape actually comes from.
        assert_eq!(parse_generation("_0_1.fnm"), 1);
        assert_eq!(parse_generation("_1a_z_Lucene90_0.dvm"), 35);
    }

    #[test]
    fn inflate_gens_never_lowers_a_counter_and_tolerates_an_empty_directory() {
        let mut current = infos(5, vec![]);
        let before_counter = current.counter;
        IndexFileDeleter::inflate_gens(&[], &mut current);
        assert_eq!(current.generation, 5);
        assert_eq!(current.counter, before_counter);

        // A file whose base-36 tail is unparsable is trash, not a counter.
        let files = vec!["segments_zzzzzzzzzzzzzzzzzzzz".to_string()];
        IndexFileDeleter::inflate_gens(&files, &mut current);
        assert_eq!(current.generation, 5);
        assert_eq!(current.counter, before_counter);
    }

    #[test]
    fn opening_on_an_empty_directory_holds_nothing_and_deletes_nothing() {
        let tmp = tempdir("empty");
        let dir = FsDirectory::open(&tmp);
        let deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();
        assert_eq!(deleter.commit_count(), 0);
        assert!(deleter.commit_file_names().is_empty());
        assert_eq!(deleter.ref_count("_0.si"), 0);
        assert!(!deleter.exists("_0.si"));
        assert!(listing(&dir).is_empty());
    }

    #[test]
    fn a_non_commit_checkpoint_holds_the_new_view_and_releases_the_previous_one() {
        let tmp = tempdir("noncommit-checkpoint");
        let dir = FsDirectory::open(&tmp);
        let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
        let b = seed_segment(&dir, "_1", 2, &["_1.fdt"]);

        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();
        // The init sweep already reclaimed both, since nothing referenced them.
        assert!(listing(&dir).is_empty());

        let a = {
            let _ = a;
            seed_segment(&dir, "_0", 1, &["_0.fdt"])
        };
        deleter
            .checkpoint(&infos(0, vec![a.clone()]), false)
            .unwrap();
        assert_eq!(deleter.ref_count("_0.si"), 1);
        assert_eq!(deleter.ref_count("_0.fdt"), 1);
        assert!(deleter.exists("_0.si"));

        let b = {
            let _ = b;
            seed_segment(&dir, "_1", 2, &["_1.fdt"])
        };
        // Replacing the view with a different segment releases the first one.
        deleter
            .checkpoint(&infos(0, vec![b.clone()]), false)
            .unwrap();
        assert_eq!(deleter.ref_count("_0.si"), 0);
        assert_eq!(deleter.ref_count("_1.si"), 1);
        assert_eq!(listing(&dir), vec!["_1.fdt", "_1.si"]);
    }

    #[test]
    fn a_commit_checkpoint_keeps_only_the_last_commit_and_deletes_the_rest() {
        let tmp = tempdir("commit-checkpoint");
        let dir = FsDirectory::open(&tmp);
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();

        let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
        let first = infos(1, vec![a.clone()]);
        write(&dir, "segments_1", b"placeholder");
        deleter.checkpoint(&first, true).unwrap();
        assert_eq!(deleter.commit_count(), 1);
        assert_eq!(deleter.commit_file_names(), vec!["segments_1"]);
        assert_eq!(deleter.ref_count("segments_1"), 1);

        // A second commit that drops `_0` entirely.
        let b = seed_segment(&dir, "_1", 2, &["_1.fdt"]);
        let second = infos(2, vec![b.clone()]);
        write(&dir, "segments_2", b"placeholder");
        deleter.checkpoint(&second, true).unwrap();

        assert_eq!(deleter.commit_count(), 1, "only the last commit survives");
        assert_eq!(deleter.commit_file_names(), vec!["segments_2"]);
        assert_eq!(
            listing(&dir),
            vec!["_1.fdt", "_1.si", "segments_2"],
            "the superseded commit and every file only it named must be gone"
        );
    }

    #[test]
    fn the_keep_all_policy_never_drops_a_commit_point() {
        let tmp = tempdir("keep-all");
        let dir = FsDirectory::open(&tmp);
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepAll).unwrap();

        let a = seed_segment(&dir, "_0", 1, &[]);
        write(&dir, "segments_1", b"placeholder");
        deleter.checkpoint(&infos(1, vec![a]), true).unwrap();
        let b = seed_segment(&dir, "_1", 2, &[]);
        write(&dir, "segments_2", b"placeholder");
        deleter
            .checkpoint(&infos(2, vec![b.clone()]), true)
            .unwrap();

        assert_eq!(deleter.commit_count(), 2);
        assert_eq!(
            listing(&dir),
            vec!["_0.si", "_1.si", "segments_1", "segments_2"]
        );

        // Tightening the policy reclaims what it was holding.
        deleter
            .set_policy(DeletionPolicy::KeepOnlyLastCommit)
            .unwrap();
        assert_eq!(deleter.commit_count(), 1);
        assert_eq!(listing(&dir), vec!["_1.si", "segments_2"]);
    }

    #[test]
    fn refresh_and_delete_new_files_reclaim_only_unreferenced_files() {
        let tmp = tempdir("refresh");
        let dir = FsDirectory::open(&tmp);
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();

        let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
        deleter.checkpoint(&infos(0, vec![a]), false).unwrap();

        write(&dir, "_5.fdt", b"orphan");
        write(&dir, "pending_segments_4", b"orphan");
        std::fs::write(tmp.join("README.txt"), b"not ours").unwrap();

        deleter.refresh().unwrap();
        assert_eq!(listing(&dir), vec!["README.txt", "_0.fdt", "_0.si"]);

        // `deleteNewFiles` only removes the names it is given, and only the
        // unreferenced ones.
        write(&dir, "_6.fdt", b"orphan");
        deleter
            .delete_new_files(&["_6.fdt".to_string(), "_0.fdt".to_string()])
            .unwrap();
        assert_eq!(
            listing(&dir),
            vec!["README.txt", "_0.fdt", "_0.si"],
            "a referenced file must survive deleteNewFiles"
        );
    }

    #[test]
    fn opening_over_an_existing_commit_reclaims_orphans_and_holds_the_commit() {
        let tmp = tempdir("open-existing");
        let dir = FsDirectory::open(&tmp);
        let a = seed_segment(&dir, "_0", 1, &["_0.fdt"]);
        let committed = infos(1, vec![a]);
        segment_infos::write(&committed, &dir).unwrap();

        // Orphans from a "crash".
        write(&dir, "_7.fdt", b"orphan");
        write(&dir, "pending_segments_9", b"orphan");

        let deleter =
            IndexFileDeleter::open(&dir, &committed, DeletionPolicy::KeepOnlyLastCommit).unwrap();

        assert_eq!(deleter.commit_count(), 1);
        assert_eq!(deleter.commit_file_names(), vec!["segments_1"]);
        assert_eq!(listing(&dir), vec!["_0.fdt", "_0.si", "segments_1"]);
        // The commit holds one reference and the in-memory view another.
        assert_eq!(deleter.ref_count("_0.si"), 2);
        assert_eq!(deleter.ref_count("segments_1"), 1);
    }

    #[test]
    fn a_commit_file_under_a_non_canonical_name_is_rejected_rather_than_deleted() {
        let tmp = tempdir("noncanonical-commit");
        let dir = FsDirectory::open(&tmp);
        let a = seed_segment(&dir, "_0", 1, &[]);
        let committed = infos(1, vec![a]);
        segment_infos::write(&committed, &dir).unwrap();
        // `segments_01` parses to generation 1, but the commit it holds names
        // itself `segments_1`, so nothing ever increfs `segments_01`. Java hits
        // the same case with `IllegalStateException("... refCount=0, which
        // should never happen on init")`; deleting a commit file we do not
        // understand is not an option.
        dir.rename("segments_1", "segments_01").unwrap();

        let err = match IndexFileDeleter::open(&dir, &committed, DeletionPolicy::KeepOnlyLastCommit)
        {
            Err(e) => e,
            Ok(_) => panic!("a commit file nothing references must be rejected"),
        };
        assert!(
            matches!(&err, Error::UnreferencedCommitFile(n) if n == "segments_01"),
            "{err:?}"
        );
        assert!(listing(&dir).contains(&"segments_01".to_string()));
    }

    #[test]
    fn a_segment_whose_si_is_missing_surfaces_as_an_error_not_a_silent_skip() {
        let tmp = tempdir("missing-si");
        let dir = FsDirectory::open(&tmp);
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();
        let phantom = SegmentCommitInfo {
            segment_name: "_ghost".to_string(),
            segment_id: id(9),
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
            ..Default::default()
        };
        let err = deleter
            .checkpoint(&infos(0, vec![phantom]), false)
            .unwrap_err();
        assert!(matches!(err, Error::Store(_)), "{err:?}");
    }

    #[test]
    fn a_corrupt_commit_file_is_an_error_at_open_not_a_fresh_index() {
        let tmp = tempdir("corrupt-commit");
        let dir = FsDirectory::open(&tmp);
        write(&dir, "segments_1", b"not a commit file at all");
        let err = match IndexFileDeleter::open(&dir, &infos(1, vec![]), DeletionPolicy::KeepAll) {
            Err(e) => e,
            Ok(_) => panic!("a corrupt commit file must be an error"),
        };
        assert!(matches!(err, Error::SegmentInfos(_)), "{err:?}");
    }

    #[test]
    fn generational_liv_field_infos_and_doc_values_files_are_all_refcounted() {
        let tmp = tempdir("generational-files");
        let dir = FsDirectory::open(&tmp);
        let mut deleter =
            IndexFileDeleter::open(&dir, &infos(0, vec![]), DeletionPolicy::KeepOnlyLastCommit)
                .unwrap();

        let mut sci = seed_segment(&dir, "_0", 1, &[]);
        sci.del_gen = 3;
        sci.field_infos_gen = 2;
        sci.field_infos_files = vec!["_0_2.fnm".to_string()];
        sci.dv_update_files = vec![(0, vec!["_0_2_Lucene90_0.dvd".to_string()])];
        for name in ["_0_3.liv", "_0_2.fnm", "_0_2_Lucene90_0.dvd"] {
            write(&dir, name, b"x");
        }

        deleter
            .checkpoint(&infos(0, vec![sci.clone()]), false)
            .unwrap();
        for name in ["_0.si", "_0_3.liv", "_0_2.fnm", "_0_2_Lucene90_0.dvd"] {
            assert_eq!(deleter.ref_count(name), 1, "{name} must be refcounted");
        }

        // Dropping the segment reclaims every generational file with it -- the
        // exact leak `SegmentCommitInfo.files()` exists to prevent.
        deleter.checkpoint(&infos(0, vec![]), false).unwrap();
        assert!(listing(&dir).is_empty(), "{:?}", listing(&dir));
    }
}
