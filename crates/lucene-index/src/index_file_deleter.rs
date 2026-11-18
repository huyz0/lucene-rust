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
//! **Two ways to choose a policy.** [`DeletionPolicy`] is an enum over Lucene's
//! two *stateless* built-ins, the fast path every existing caller uses. A
//! pluggable `IndexDeletionPolicy` ([`crate::deletion_policy`]:
//! `KeepLastNCommits`, `Snapshot`, `PersistentSnapshot`, or a caller's own) is
//! installed with [`IndexFileDeleter::open_with_policy`] or
//! [`IndexFileDeleter::set_custom_policy`] and then decides instead; each
//! commit point is handed to it as an [`IndexCommit`] carrying its
//! generation, files, segment count and user data, as Java's `CommitPoint`
//! does.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;

use crate::deletion_policy::{self, IndexCommit, IndexDeletionPolicy};
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
    /// A pluggable [`IndexDeletionPolicy`] failed.
    #[error(transparent)]
    Policy(#[from] deletion_policy::Error),
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

/// `SegmentInfos.files(includeSegmentsFile)` for a caller with no deleter
/// (the replicator's replica nodes): every file `infos` names, read off the
/// segments' `.si` files in `dir`.
///
/// # Errors
/// A segment's `.si` is missing or unreadable.
pub fn files_of_infos(
    dir: &dyn Directory,
    infos: &SegmentInfos,
    include_segments_file: bool,
) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    if include_segments_file {
        if let Some(name) = lucene_store::directory::segments_file_name(infos.generation) {
            seen.insert(name.clone());
            out.push(name);
        }
    }
    for sci in &infos.segments {
        for f in segment_files(dir, sci)? {
            if seen.insert(f.clone()) {
                out.push(f);
            }
        }
    }
    Ok(out)
}

/// `SegmentCommitInfo.files()` for a caller with no deleter: the files of one
/// segment, read off its `.si` in `dir`.
///
/// # Errors
/// The `.si` is missing or unreadable.
pub fn segment_files(dir: &dyn Directory, sci: &SegmentCommitInfo) -> Result<Vec<String>> {
    let si_files = segment_info_files(dir, sci)?;
    Ok(sci.files(&si_files))
}

/// Java: `Codec.forName(...).segmentInfoFormat().read`, then `SegmentInfo.files`.
/// SimpleText's `.si` is textual even though its enclosing commit is binary.
fn segment_info_files(dir: &dyn Directory, sci: &SegmentCommitInfo) -> Result<Vec<String>> {
    let bytes = dir.open(&format!("{}.si", sci.segment_name))?.to_vec();
    let files = if sci.codec_name == "SimpleText" {
        segment_info::parse_simpletext(&bytes, &sci.segment_name, &sci.segment_id)?.files
    } else {
        segment_info::parse_for_commit(&bytes, sci)?.files
    };
    Ok(IndexFileDeleter::with_self_listing(
        &sci.segment_name,
        files,
    ))
}

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
/// Every other policy -- `KeepLastNCommitsDeletionPolicy`,
/// `SnapshotDeletionPolicy`, `PersistentSnapshotDeletionPolicy`, a caller's
/// own -- is a [`crate::deletion_policy::IndexDeletionPolicy`], installed with
/// [`IndexFileDeleter::open_with_policy`]/[`IndexFileDeleter::set_custom_policy`].
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

/// Java's `IndexFileDeleter.CommitPoint`; [`IndexFileDeleter::index_commits`]
/// hands it to a policy as an [`IndexCommit`].
#[derive(Debug, Clone)]
struct CommitPoint {
    generation: i64,
    segments_file_name: String,
    files: Vec<String>,
    /// `IndexCommit.getSegmentCount()`, for a pluggable policy.
    segment_count: usize,
    /// `IndexCommit.getUserData()`, for a pluggable policy.
    user_data: Vec<(String, String)>,
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
    /// A pluggable `IndexDeletionPolicy` ([`crate::deletion_policy`]); when
    /// set it decides instead of [`Self::policy`].
    custom: Option<Arc<dyn IndexDeletionPolicy>>,
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
    /// Files known to be fsynced: every file a commit found on disk at
    /// [`Self::open`] names (a commit is only ever published after its files
    /// were synced), plus every file [`Self::sync_commit_files`] has synced
    /// since. A name leaves the set when the deleter deletes the file.
    ///
    /// This is the complement of Java's model, in which `IndexWriter.startCommit`
    /// syncs every file the new commit references: a file this writer wrote
    /// (a flushed or merged segment, a `.liv` or doc-values generation, a
    /// copied `addIndexes` segment) is not in the set, so the first commit that
    /// names it syncs it, and no later commit syncs it again. A file no commit
    /// ever names -- a superseded `.liv` generation, a segment merged away
    /// before it was committed -- is never synced at all; the deleter reclaims
    /// it.
    synced: HashSet<String>,
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
        Self::open_inner(dir, current, policy, None)
    }

    /// [`Self::open`] under a pluggable [`IndexDeletionPolicy`]: its
    /// `on_init` sees every commit found on disk, and its `on_commit` every
    /// live commit after each new one -- `IndexWriterConfig
    /// .setIndexDeletionPolicy` with a policy of the caller's own.
    pub fn open_with_policy(
        dir: &'d dyn Directory,
        current: &SegmentInfos,
        policy: Arc<dyn IndexDeletionPolicy>,
    ) -> Result<Self> {
        Self::open_inner(
            dir,
            current,
            DeletionPolicy::KeepOnlyLastCommit,
            Some(policy),
        )
    }

    fn open_inner(
        dir: &'d dyn Directory,
        current: &SegmentInfos,
        policy: DeletionPolicy,
        custom: Option<Arc<dyn IndexDeletionPolicy>>,
    ) -> Result<Self> {
        let mut deleter = IndexFileDeleter {
            dir,
            policy,
            custom,
            ref_counts: HashMap::new(),
            commits: Vec::new(),
            last_files: Vec::new(),
            si_files: HashMap::new(),
            synced: HashSet::new(),
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
            // Whoever published this commit synced its files first.
            deleter.synced.extend(commit_files.iter().cloned());
            deleter.commits.push(CommitPoint {
                generation,
                segments_file_name: file_name.clone(),
                files: commit_files,
                segment_count: sis.segments.len(),
                user_data: sis.user_data.clone(),
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

        // "Always protect the incoming segmentInfos since sometimes it may not
        // be the most recent commit."
        // Note this is deliberately *not* asserted to have referenced a
        // `segments_N`: a never-committed index's `SegmentInfos` carries
        // generation 0 in this port (Java models it as -1 / a null
        // `getSegmentsFileName()`), and there is no commit file for it yet.
        //
        // Before the policy's deletions, as in Java (`onInit`, `checkpoint`,
        // then `deleteCommits`): a policy that drops the very commit this
        // writer opened on must not take the files the writer is about to
        // use with it.
        deleter.checkpoint(current, false)?;

        // `policy.onInit(commits)` then `deleteCommits()`.
        deleter.apply_policy(true)?;

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
                segment_count: infos.segments.len(),
                user_data: infos.user_data.clone(),
            });
            self.commits.sort_by_key(|c| c.generation);
            self.apply_policy(false)?;
        } else {
            let previous = std::mem::take(&mut self.last_files);
            self.dec_ref_all(&previous)?;
            self.last_files = files;
        }
        self.forget_dead_segments();
        Ok(())
    }

    /// Drops the recorded file sets of segments no file of which is
    /// referenced any more. Java's live only as long as their `SegmentInfo`;
    /// kept here past that, one entry per segment ever flushed or merged, they
    /// were a leak -- thousands of entries an hour under a concurrent writer.
    fn forget_dead_segments(&mut self) {
        let referenced: std::collections::HashSet<&str> = self
            .ref_counts
            .keys()
            .map(|f| parse_segment_name(f))
            .collect();
        self.si_files
            .retain(|name, _| referenced.contains(name.as_str()));
    }

    /// `IndexWriterConfig.setIndexDeletionPolicy` + `IndexFileDeleter.revisitPolicy()`:
    /// switch policy and apply it immediately, so tightening from
    /// [`DeletionPolicy::KeepAll`] back to
    /// [`DeletionPolicy::KeepOnlyLastCommit`] reclaims the commits that were
    /// being held rather than waiting for the next one.
    pub fn set_policy(&mut self, policy: DeletionPolicy) -> Result<()> {
        self.policy = policy;
        self.custom = None;
        self.apply_policy(false)
    }

    /// [`Self::set_policy`] with a pluggable [`IndexDeletionPolicy`]:
    /// `revisitPolicy()` runs its `on_commit` over the live commits at once.
    pub fn set_custom_policy(&mut self, policy: Arc<dyn IndexDeletionPolicy>) -> Result<()> {
        self.custom = Some(policy);
        if self.commits.is_empty() {
            return Ok(());
        }
        self.apply_policy(false)
    }

    /// `IndexFileDeleter.revisitPolicy()`: run the policy's `on_commit` over
    /// the live commits again -- what `IndexWriter.deleteUnusedFiles` does, so
    /// a commit a released snapshot no longer holds is dropped without waiting
    /// for the next commit.
    pub fn revisit_policy(&mut self) -> Result<()> {
        if self.commits.is_empty() {
            return Ok(());
        }
        self.apply_policy(false)?;
        self.forget_dead_segments();
        Ok(())
    }

    /// An `IndexCommit` for every live commit point, oldest first -- what a
    /// deletion policy is handed, and `DirectoryReader.listCommits` over the
    /// writer's directory.
    pub fn index_commits(&self) -> Vec<IndexCommit> {
        self.commits
            .iter()
            .map(|c| {
                IndexCommit::new(
                    c.generation,
                    c.segments_file_name.clone(),
                    c.files.clone(),
                    c.segment_count,
                    c.user_data.clone(),
                )
            })
            .collect()
    }

    /// `IndexCommit.delete()` from a caller-run deletion policy -- how
    /// OpenSearch's `CombinedDeletionPolicy` drives a writer whose policy is
    /// [`DeletionPolicy::KeepAll`]: each named commit point is dropped and its
    /// files decRef'd, deleting whatever nothing else still names. The newest
    /// commit is never dropped (Java's `IndexFileDeleter` asserts a policy
    /// leaves it), and a generation with no live commit point is ignored.
    pub fn drop_commits(&mut self, generations: &[i64]) -> Result<()> {
        let newest = self.commits.last().map(|c| c.generation);
        let (doomed, kept): (Vec<CommitPoint>, Vec<CommitPoint>) =
            std::mem::take(&mut self.commits)
                .into_iter()
                .partition(|c| Some(c.generation) != newest && generations.contains(&c.generation));
        self.commits = kept;
        for commit in doomed {
            self.dec_ref_all(&commit.files)?;
        }
        self.forget_dead_segments();
        Ok(())
    }

    /// The generations of every live commit point, oldest first.
    pub fn commit_generations(&self) -> Vec<i64> {
        self.commits.iter().map(|c| c.generation).collect()
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
        self.delete_files(&to_delete)?;
        self.forget_dead_segments();
        Ok(())
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

    /// How many segments' file sets and how many files this deleter tracks:
    /// what a resource-bound test holds to the live segments and files.
    #[cfg(test)]
    pub(crate) fn tracked_sizes(&self) -> (usize, usize) {
        (self.si_files.len(), self.ref_counts.len())
    }

    /// Every file this deleter holds a reference to, sorted.
    #[cfg(test)]
    pub(crate) fn referenced_files(&self) -> Vec<String> {
        let mut names: Vec<String> = self.ref_counts.keys().cloned().collect();
        names.sort();
        names
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

    /// The sync half of `IndexWriter.startCommit`
    /// (`directory.sync(toSync.files(false))`): fsyncs every file `infos`
    /// names that is not yet known to be synced, so a commit about to name
    /// them can be published. Called immediately before a `segments_N` (or
    /// `pending_segments_N`) for `infos` is written.
    ///
    /// Each file is synced at most once per writer: a file a previous commit
    /// already synced, or one named by a commit that was on disk when this
    /// deleter opened, is skipped. On failure nothing is recorded as synced,
    /// so the next attempt syncs the same files again.
    pub fn sync_commit_files(&mut self, infos: &SegmentInfos) -> Result<()> {
        let mut to_sync: Vec<String> = self
            .commit_files(infos, false)?
            .into_iter()
            .filter(|name| !self.synced.contains(name))
            .collect();
        if to_sync.is_empty() {
            return Ok(());
        }
        to_sync.sort();
        self.dir.sync(&to_sync)?;
        self.synced.extend(to_sync);
        Ok(())
    }

    /// Whether [`Self::sync_commit_files`] (or a commit found at open) has
    /// already made `name` durable.
    pub fn is_synced(&self, name: &str) -> bool {
        self.synced.contains(name)
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
    /// It exists so `examples/deleter_checkpoint.rs` can time both arms in
    /// **one process from one build**, the only shape of A/B this project
    /// trusts. Correct but pointless in production: the cache refills itself
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
        let files = segment_info_files(self.dir, sci)?;
        self.si_files
            .insert(sci.segment_name.clone(), (sci.segment_id, files));
        Ok(())
    }

    /// `Lucene99SegmentInfoFormat.write` calls `si.addFile(fileName)` for the
    /// `.si` itself before encoding, so a correctly written `.si` already lists
    /// itself. Older segments this port wrote did not; adding it here keeps the
    /// deleter from reclaiming the file that names all the others.
    pub(crate) fn with_self_listing(segment_name: &str, mut files: Vec<String>) -> Vec<String> {
        let name = format!("{segment_name}.si");
        if !files.iter().any(|f| f == &name) {
            files.push(name);
        }
        files
    }

    /// Holds a reference on every file of `segments` -- what a merge in
    /// progress does to its sources (`IndexWriter.mergeInit`'s readers), so a
    /// delete or a checkpoint that releases one of them meanwhile cannot take
    /// a file the merge is about to read. Returns the files held, for
    /// [`Self::release_files`].
    pub(crate) fn hold_segment_files(
        &mut self,
        segments: &[SegmentCommitInfo],
    ) -> Result<Vec<String>> {
        let mut held = Vec::new();
        for sci in segments {
            self.ensure_si_files(sci)?;
            let (_, si_files) = self
                .si_files
                .get(sci.segment_name.as_str())
                .expect("just ensured above");
            held.extend(sci.files(si_files));
        }
        self.inc_ref_all(&held);
        Ok(held)
    }

    /// Holds a reference on files written before any segment names them
    /// (a buffer's `.fdt`), so no sweep takes them; [`Self::release_files`]
    /// drops it.
    pub(crate) fn hold_files(&mut self, files: &[String]) {
        self.inc_ref_all(files);
    }

    /// Drops [`Self::hold_segment_files`]' references, deleting whatever
    /// nothing else still holds.
    pub(crate) fn release_files(&mut self, files: &[String]) -> Result<()> {
        // A merge releases its sources after the commit that retired them:
        // their files go here, and so must their recorded file sets, not at
        // the next checkpoint.
        self.dec_ref_all(files)?;
        self.forget_dead_segments();
        Ok(())
    }

    fn inc_ref_all(&mut self, files: &[String]) {
        for name in files {
            // `get_mut` first: nearly every name is already counted, and
            // `entry` would allocate its key for each one.
            let count = match self.ref_counts.get_mut(name) {
                Some(count) => count,
                None => self.ref_counts.entry(name.clone()).or_insert(0),
            };
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
                    // Removed from `ref_counts`, so a repeat of `name` later
                    // in `files` finds no entry: no duplicate check needed.
                    self.ref_counts.remove(name);
                    to_delete.push(name.clone());
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
    fn apply_policy(&mut self, init: bool) -> Result<()> {
        if let Some(policy) = self.custom.clone() {
            // `policy.onInit`/`onCommit(commits)`, then `deleteCommits()`:
            // every commit the policy called `delete()` on is dropped and its
            // files decRef'd. The current view stays protected by the last
            // non-commit checkpoint's references, as in Java.
            let mut list = self.index_commits();
            if init {
                policy.on_init(&mut list)?;
            } else {
                policy.on_commit(&mut list)?;
            }
            let commits = std::mem::take(&mut self.commits);
            let mut doomed = Vec::new();
            for (commit, decided) in commits.into_iter().zip(&list) {
                if decided.is_deleted() {
                    doomed.push(commit);
                } else {
                    self.commits.push(commit);
                }
            }
            for commit in doomed {
                self.dec_ref_all(&commit.files)?;
            }
            return Ok(());
        }
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
    ///
    /// Within a pass every file is tried and the first failure is returned
    /// at the end, so one undeletable file does not leave the rest of the
    /// list on disk untracked; the second pass only runs once every
    /// `segments_N` is gone.
    fn delete_files(&mut self, names: &[String]) -> Result<()> {
        for segments_pass in [true, false] {
            let mut first: Option<Error> = None;
            for name in names {
                if is_segments_file_name(name) == segments_pass {
                    self.ref_counts.remove(name);
                    // Before the delete, not after: a delete that fails
                    // partway may still have removed the file, and a later
                    // file of the same name must then be synced afresh.
                    self.synced.remove(name);
                    if let Err(e) = self.dir.delete_file(name) {
                        first.get_or_insert(e.into());
                    }
                }
            }
            if let Some(e) = first {
                return Err(e);
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
    // Java's default regex `.` excludes these line terminators. The prefix
    // branches above are String.startsWith checks and have no such exclusion.
    if name.contains(['\n', '\r', '\u{0085}', '\u{2028}', '\u{2029}']) {
        return false;
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
pub(crate) fn parse_segment_name(file_name: &str) -> &str {
    // ARITH: `i` is a byte offset *within* `file_name[1..]`, so `i + 1` is at
    // most `file_name.len()` -- an in-memory length, hence at most
    // `isize::MAX`.
    //
    // `get`, not slicing: a `.si` names its own files, so a corrupt one can
    // put an empty or non-ASCII-led name here, which must not panic.
    #[allow(clippy::arithmetic_side_effects)]
    let idx = file_name
        .get(1..)
        .and_then(|rest| rest.find(['_', '.']))
        .map(|i| i + 1)
        .unwrap_or(file_name.len());
    file_name.get(..idx).unwrap_or(file_name)
}

/// `SegmentInfos.files(true)` read off disk: the commit's `segments_N`, then
/// every file each segment's `.si` names (the `.si` included), then its
/// `.liv`, field-infos and doc-values-update files.
fn commit_file_names(dir: &dyn Directory, infos: &SegmentInfos) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    if let Some(name) = lucene_store::directory::segments_file_name(infos.generation) {
        seen.insert(name.clone());
        out.push(name);
    }
    for sci in &infos.segments {
        for f in segment_files(dir, sci)? {
            if !seen.contains(&f) {
                seen.insert(f.clone());
                out.push(f);
            }
        }
    }
    Ok(out)
}

/// `DirectoryReader.listCommits(dir)`: an [`IndexCommit`] for every commit
/// point in `dir`, oldest first -- the latest commit, plus every older
/// `segments_N` still present. As in Java, an older `segments_N` the listing
/// names but that is gone by the time it is opened (a concurrent writer
/// deleted it) is skipped; any other failure to read one is an error, as is
/// an unparsable `segments*` name and a directory with no commit at all.
pub fn list_commits(dir: &dyn Directory) -> Result<Vec<IndexCommit>> {
    let files = dir.list_all()?;
    let latest = segment_infos::read_latest(dir)?;
    let mut all = vec![latest];
    let current = all[0].generation;
    for name in &files {
        // Java's test is `startsWith(SEGMENTS)` alone, so an unparsable
        // `segments*` name (`segments.gen` included) is an error, not skipped.
        if !name.starts_with("segments") {
            continue;
        }
        let generation = lucene_store::directory::generation_from_segments_file_name(name)?;
        if generation >= current {
            continue;
        }
        let bytes = match dir.open(name) {
            Ok(b) => b,
            Err(e) if e.is_no_such_file() => continue,
            Err(e) => return Err(e.into()),
        };
        all.push(segment_infos::parse(&bytes, generation)?);
    }
    all.sort_by_key(|infos| infos.generation);
    let mut commits = Vec::with_capacity(all.len());
    for infos in &all {
        let segments_file =
            lucene_store::directory::segments_file_name(infos.generation).unwrap_or_default();
        commits.push(IndexCommit::new(
            infos.generation,
            segments_file,
            commit_file_names(dir, infos)?,
            infos.segments.len(),
            infos.user_data.clone(),
        ));
    }
    Ok(commits)
}

#[cfg(test)]
mod tests;
