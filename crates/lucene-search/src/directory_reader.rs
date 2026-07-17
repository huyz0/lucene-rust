//! A minimal `DirectoryReader`/`SegmentReader`-equivalent (task #45): the
//! "open every segment a commit lists, in one call" object this port has
//! been missing since task #41 added multi-segment search. Every query
//! function in this crate up to now (including [`crate::multi_segment`]'s
//! own tests) required the *caller* to manually read `segments_N`, open each
//! segment's `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`/`.liv` files by
//! hand, and compute each segment's `doc_base` as a running sum of earlier
//! segments' `maxDoc` -- exactly the boilerplate real `DirectoryReader.open`
//! centralizes in Java. This module is that centralization, nothing more.
//!
//! # Scope
//!
//! Mirrors `DirectoryReader.open(Directory)`: find the latest `segments_N`
//! ([`lucene_index::segment_infos::read_latest`]), open one
//! [`SegmentReader`] per listed segment (only the files that segment
//! actually has -- most fixtures in this port are stored-fields-only and
//! have no postings/doc-values/norms/term-vectors files at all, see
//! `segment_writer.rs`'s doc comment), and hand back `doc_base`-labelled
//! [`OpenSegment`] values ready for [`crate::multi_segment`]'s existing
//! fan-out/merge functions.
//!
//! **Compound-file segments** (`.cfs`/`.cfe`, `SegmentInfo.is_compound_file`)
//! are opened transparently: [`SegmentReader::open`] reads the segment's
//! `.cfs`/`.cfe` pair through the existing
//! [`lucene_codecs::compound_format`] read API and pulls each sub-file
//! (`.fnm`, and `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay` when present) out of
//! that in-memory archive instead of `dir.open`-ing a loose file that
//! wouldn't exist on disk for such a segment -- see [`open_segment_file`]'s
//! doc comment for exactly which sub-files this port's own write side
//! (`segment_writer.rs`) ever actually packs into one today.
//!
//! **Deliberately excluded** (see `docs/parity.md` for the authoritative
//! list): NRT/reopen (`DirectoryReader.openIfChanged`), soft deletes, and
//! norms/term vectors (irrelevant here: [`OpenSegment`] itself has no fields
//! for them -- `crate::field_norms`/term-vectors query functions still take
//! their own already-opened readers directly, unchanged by this task).
//! **Doc-values are now wired in** (see the "NRT reopen after sparse
//! doc-values commits" entry in `docs/parity.md`): [`SegmentReader::open`]
//! also opens a segment's `.dvm`/`.dvd` when present (together or not at
//! all, same contract as `.tim`/`.tip`/`.tmd`), exposed via
//! [`SegmentReader::field_infos`]/[`SegmentReader::doc_values_meta`]/
//! [`SegmentReader::doc_values_data`] -- callers feed those straight into the
//! existing dense/sparse-agnostic `lucene_codecs::doc_values` value readers,
//! and [`DirectoryReader::open_if_changed`] refreshes them for free since
//! they live on `SegmentReader` alongside every other per-segment file this
//! module already reopens-or-reuses.
//!
//! # Why the two-step `open_segments`/`as_open_segments` API
//!
//! [`OpenSegment<'a>`] holds `&'a DocInput<'a>` (a *reference to* an already
//! constructed value), not an owned `DocInput` and not raw bytes -- so
//! producing a `Vec<OpenSegment>` needs somewhere for those `DocInput`/
//! `PosInput`/`PayInput` values to live at least as long as the
//! `OpenSegment`s that borrow them. [`SegmentReader`] can't store them
//! itself (that would be self-referential: the `Input` would borrow from a
//! byte buffer owned by the same struct). Instead, [`DirectoryReader::open_segments`]
//! returns an [`OpenedSegments`] that owns those freshly-opened `Input`
//! values (borrowing from the `SegmentReader`s' already-owned byte buffers,
//! not from itself), and [`OpenedSegments::as_open_segments`] hands back the
//! `Vec<OpenSegment>` borrowing from *that*. Two calls instead of one, but
//! no `unsafe`/self-referential tricks (`#![forbid(unsafe_code)]` in this
//! crate) and no behavior change from what a hand-written caller already had
//! to do.

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::compound_format;
use lucene_codecs::doc_values::{self, DocValuesMeta};
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_codecs::live_docs;
use lucene_codecs::postings::{self, DocInput, PayInput, PosInput};
use lucene_index::deletes::liv_file_name;
use lucene_index::segment_info::{self, SegmentInfo};
use lucene_index::segment_infos::{self, SegmentInfos};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::multi_segment::OpenSegment;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    SegmentInfos(#[from] segment_infos::Error),
    #[error(transparent)]
    SegmentInfo(#[from] segment_info::Error),
    #[error(transparent)]
    FieldInfos(#[from] field_infos::Error),
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    #[error(transparent)]
    Postings(#[from] postings::Error),
    #[error(transparent)]
    LiveDocs(#[from] live_docs::Error),
    #[error(transparent)]
    CompoundFormat(#[from] compound_format::Error),
    #[error(transparent)]
    DocValues(#[from] doc_values::Error),
    #[error("segment {segment} has {found} of .tim/.tip/.tmd (need all three or none)")]
    PartialBlockTreeFiles { segment: String, found: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One opened segment: everything [`OpenSegment`] needs, already decoded/
/// validated, plus the bookkeeping ([`SegmentReader::doc_base`]) real
/// `SegmentReader.docBase` carries. `fields`/`live_docs` are owned outright
/// (both are fully-decoded, non-borrowing types -- see `blocktree.rs`'s
/// `BlockTreeFields` and `lucene_util::fixed_bit_set::FixedBitSet`);
/// `.doc`/`.pos`/`.pay` are kept as raw bytes because [`DocInput`]/
/// [`PosInput`]/[`PayInput`] borrow their buffer and are cheap to
/// (re-)construct on demand -- see [`DirectoryReader::open_segments`].
#[derive(Debug, Clone)]
pub struct SegmentReader {
    pub segment_name: String,
    pub max_doc: i32,
    pub doc_base: i32,
    segment_id: [u8; ID_LENGTH],
    /// The generation of this segment's deletions at the time this reader
    /// was opened (`-1` if it has none) -- used by
    /// [`DirectoryReader::open_if_changed`] to tell "same segment, unchanged
    /// deletions" (safe to reuse) apart from "same segment, new deletions"
    /// (must reopen to pick up the new `.liv`/`live_docs`).
    del_gen: i64,
    segment_suffix: String,
    fields: BlockTreeFields,
    doc_buf: Option<Vec<u8>>,
    pos_buf: Option<Vec<u8>>,
    pay_buf: Option<Vec<u8>>,
    live_docs: Option<FixedBitSet>,
    field_infos: FieldInfos,
    /// The segment's doc-values data (`.dvd`), kept as raw bytes so the
    /// dense/sparse-agnostic `doc_values::{numeric_value, binary_value,
    /// sorted_ord, sorted_numeric_values}` readers (unchanged, no new
    /// low-level format logic here) can be called directly against it --
    /// `None` when the segment has no `.dvm`/`.dvd` at all (e.g. no field is
    /// opted into doc values, matching `segment_writer.rs`'s "no pending
    /// values -> no `.dvm`/`.dvd`/`.dvs` files" contract).
    dv_data: Option<Vec<u8>>,
    /// The segment's parsed `.dvm` (per-field entries -- `NumericEntry`,
    /// `SortedEntry`, etc., each already carrying whichever of the
    /// dense/sparse shapes `doc_values.rs`'s write side actually produced;
    /// there is no separate "is this field sparse" flag to track here since
    /// every entry type already self-describes dense vs.
    /// `IndexedDISI`-backed sparse, and the existing value readers dispatch
    /// on that internally).
    dv_meta: Option<DocValuesMeta>,
}

impl SegmentReader {
    /// A cheap in-memory copy of an already-open reader, used by
    /// [`DirectoryReader::open_at_reusing`] to reuse a segment without
    /// re-reading any files -- `Clone` derives from `BlockTreeFields`/
    /// `FixedBitSet` already being `Clone`, so this is a genuine "no disk
    /// I/O" reuse, not a relabeled fresh open.
    fn clone_reader(&self) -> Self {
        self.clone()
    }
}

impl SegmentReader {
    /// Opens one segment: reads its `.si`, then whichever of `.fnm`/
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`/`.liv` it actually has
    /// (checked via `SegmentInfo.files` -- a segment missing postings, e.g.
    /// this port's stored-fields-only fixtures, legitimately has none of
    /// `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`, and that is not an error).
    fn open(
        dir: &dyn Directory,
        commit: &segment_infos::SegmentCommitInfo,
        doc_base: i32,
    ) -> Result<Self> {
        let segment_name = commit.segment_name.clone();
        let segment_id = commit.segment_id;

        let si_bytes = dir.open(&format!("{segment_name}.si"))?;
        let si: SegmentInfo = segment_info::parse(&si_bytes, &segment_id)?;

        // Compound-file segments (`.cfs`/`.cfe`) pack every other codec file
        // into one archive -- see `open_segment_file`'s doc comment for how
        // every sub-file lookup below transparently reads from it instead of
        // `dir.open`-ing a loose file that wouldn't exist on disk.
        let compound = if si.is_compound_file {
            Some(CompoundArchive::open(dir, &segment_name, &segment_id)?)
        } else {
            None
        };

        let fnm_bytes =
            open_segment_file(dir, compound.as_ref(), &si.files, ".fnm")?.ok_or_else(|| {
                Error::Store(lucene_store::Error::Corrupted(format!(
                    "segment {segment_name} has no .fnm file"
                )))
            })?;
        let field_infos = field_infos::parse(&fnm_bytes, &segment_id, "")?;

        let tim_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tim")?;
        let tip_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tip")?;
        let tmd_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".tmd")?;
        let found = [&tim_bytes, &tip_bytes, &tmd_bytes]
            .iter()
            .filter(|f| f.is_some())
            .count();

        let (fields, segment_suffix, doc_buf, pos_buf, pay_buf) = if found == 3 {
            // Suffix is embedded in the sub-file's own name: strip the
            // `<segment_name>_` prefix (loose files, e.g.
            // `_0_Lucene104_0.tim` -> `Lucene104_0`) or the leading `_`
            // (compound entries, already stripped of the segment-name prefix
            // by `IndexFileNames.stripSegmentName` when packed -- e.g.
            // `_Lucene104_0.tim` -> `Lucene104_0`), then the `.tim`
            // extension.
            let tim_file_name = find_segment_file_name(&si.files, compound.as_ref(), ".tim")
                .expect("found == 3 implies a .tim entry exists");
            // No-codec-suffix case first (this port's own writer, e.g. loose
            // `_0.tim`): the generic strip-and-derive logic below would
            // otherwise misparse the segment name's own trailing digit as a
            // bogus suffix (`_0.tim` -> strip leading `_` -> `0.tim` ->
            // suffix `"0"`, wrong) -- same fix as the `.dvm` case below.
            let segment_suffix = if tim_file_name == format!("{segment_name}.tim") {
                String::new()
            } else {
                tim_file_name
                    .strip_prefix(&format!("{segment_name}_"))
                    .or_else(|| tim_file_name.strip_prefix('_'))
                    .and_then(|s| s.strip_suffix(".tim"))
                    .unwrap_or_default()
                    .to_string()
            };

            let fields = blocktree::open(
                tim_bytes.as_ref().unwrap(),
                tip_bytes.as_ref().unwrap(),
                tmd_bytes.as_ref().unwrap(),
                &field_infos,
                &segment_id,
                &segment_suffix,
                si.doc_count,
            )?;

            let doc_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".doc")?;
            let pos_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".pos")?;
            let pay_buf = open_segment_file(dir, compound.as_ref(), &si.files, ".pay")?;

            (fields, segment_suffix, doc_buf, pos_buf, pay_buf)
        } else if found == 0 {
            (BlockTreeFields::empty(), String::new(), None, None, None)
        } else {
            return Err(Error::PartialBlockTreeFiles {
                segment: segment_name,
                found,
            });
        };

        // `.liv` exists only when the segment has deletions (`del_gen !=
        // -1`, `SegmentCommitInfo.hasDeletions()`'s condition).
        let live_docs = if commit.del_gen != -1 {
            let liv_name = liv_file_name(&segment_name, commit.del_gen);
            let liv_bytes = dir.open(&liv_name)?;
            Some(live_docs::parse(
                &liv_bytes,
                &segment_id,
                commit.del_gen,
                si.doc_count as usize,
                commit.del_count as usize,
            )?)
        } else {
            None
        };

        // `.dvm`/`.dvd` (no `.dvs` reader exists yet in this port, matching
        // `doc_value_query.rs`/`facets.rs`'s own scope): present together or
        // not at all, exactly like `.tim`/`.tip`/`.tmd` above -- a segment
        // with no doc-values field opted in has neither (see
        // `segment_writer.rs`'s "no pending values -> no files" contract).
        // The codec suffix is derived the same way `.tim`'s is above --
        // this port's own writer uses `""` (matching the sparse/dense unit
        // tests in `index_writer.rs`), but a real Lucene-written segment
        // (e.g. `fixtures/data/compound_index/`) gives doc values their own
        // codec suffix (`Lucene90_<n>`), independent of the postings suffix.
        let dvm_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".dvm")?;
        let dvd_bytes = open_segment_file(dir, compound.as_ref(), &si.files, ".dvd")?;
        let (dv_meta, dv_data) = match (dvm_bytes, dvd_bytes) {
            (Some(dvm), Some(dvd)) => {
                let dvm_file_name = find_segment_file_name(&si.files, compound.as_ref(), ".dvm")
                    .expect("dvm_bytes.is_some() implies a .dvm entry exists");
                // No-codec-suffix case first (this port's own writer, e.g.
                // loose `_0.dvm`): the generic strip-and-derive logic below
                // would otherwise misparse the segment name's own trailing
                // digit as a bogus suffix (`_0.dvm` -> strip leading `_` ->
                // `0.dvm` -> suffix `"0"`, wrong).
                let dv_suffix = if dvm_file_name == format!("{segment_name}.dvm") {
                    String::new()
                } else {
                    dvm_file_name
                        .strip_prefix(&format!("{segment_name}_"))
                        .or_else(|| dvm_file_name.strip_prefix('_'))
                        .and_then(|s| s.strip_suffix(".dvm"))
                        .unwrap_or_default()
                        .to_string()
                };
                let (_, meta) =
                    doc_values::parse_meta(&dvm, &segment_id, &dv_suffix, &field_infos)?;
                (Some(meta), Some(dvd))
            }
            _ => (None, None),
        };

        Ok(SegmentReader {
            segment_name,
            max_doc: si.doc_count,
            doc_base,
            segment_id,
            del_gen: commit.del_gen,
            segment_suffix,
            fields,
            doc_buf,
            pos_buf,
            pay_buf,
            live_docs,
            field_infos,
            dv_data,
            dv_meta,
        })
    }

    /// The segment's `.fnm`-derived field metadata (field name/number
    /// mapping, doc-values type, etc.) -- callers use this to resolve a field
    /// name to the number [`Self::doc_values_meta`]'s entries are keyed by.
    pub fn field_infos(&self) -> &FieldInfos {
        &self.field_infos
    }

    /// The segment's parsed `.dvm`, or `None` if it has no doc-values files
    /// at all. Each entry (`NumericEntry`/`BinaryEntry`/`SortedEntry`/etc.)
    /// already self-describes dense vs. sparse (`IndexedDISI`-backed); feed
    /// it and [`Self::doc_values_data`] straight into the existing
    /// `lucene_codecs::doc_values::{numeric_value, binary_value, sorted_ord,
    /// sorted_numeric_values}` readers, unchanged.
    pub fn doc_values_meta(&self) -> Option<&DocValuesMeta> {
        self.dv_meta.as_ref()
    }

    /// The segment's `.dvd`, or `None` if it has no doc-values files at all.
    pub fn doc_values_data(&self) -> Option<&[u8]> {
        self.dv_data.as_deref()
    }
}

fn find_file_ending(files: &[String], ext: &str) -> Option<String> {
    files.iter().find(|f| f.ends_with(ext)).cloned()
}

/// A segment's already-opened, already-validated `.cfs`/`.cfe` pair --
/// opened once per compound segment in [`SegmentReader::open`] and then
/// queried by [`find_segment_file_name`]/[`open_segment_file`] for each
/// sub-file the reader needs, entirely through the read-side API
/// `compound_format` already exposes ([`compound_format::parse_entries`],
/// [`compound_format::check_data_header_footer`],
/// [`compound_format::open_input`]) -- nothing about the archive format
/// itself is reimplemented here.
struct CompoundArchive {
    data: Vec<u8>,
    entries: compound_format::CompoundEntries,
}

impl CompoundArchive {
    fn open(dir: &dyn Directory, segment_name: &str, segment_id: &[u8; ID_LENGTH]) -> Result<Self> {
        let cfs_bytes = dir.open(&format!("{segment_name}.cfs"))?.to_vec();
        let cfe_bytes = dir.open(&format!("{segment_name}.cfe"))?;
        let entries = compound_format::parse_entries(&cfe_bytes, segment_id)?;
        compound_format::check_data_header_footer(&cfs_bytes, segment_id, &entries)?;
        Ok(CompoundArchive {
            data: cfs_bytes,
            entries,
        })
    }
}

/// Resolves the on-disk (loose) or in-archive (compound) name of a segment's
/// sub-file ending in `ext`, without reading its bytes -- shared by
/// [`open_segment_file`] and [`SegmentReader::open`]'s `.tim`-suffix
/// derivation, both of which need the *name*, not just its contents. Loose
/// sub-file names are full file names (`SegmentInfo.files`, e.g.
/// `_0_Lucene104_0.tim`); compound entry ids are already stripped of the
/// segment-name prefix by the writer (`IndexFileNames.stripSegmentName`,
/// e.g. `_Lucene104_0.tim`) -- either way, matching by extension suffix is
/// exactly what real Lucene's per-format file-name lookup does.
fn find_segment_file_name(
    files: &[String],
    compound: Option<&CompoundArchive>,
    ext: &str,
) -> Option<String> {
    match compound {
        Some(archive) => archive
            .entries
            .names()
            .find(|name| name.ends_with(ext))
            .map(str::to_string),
        None => find_file_ending(files, ext),
    }
}

/// The one shared "read a segment's sub-file" call site every extension
/// (`.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay`) goes through: resolves
/// the file's name via [`find_segment_file_name`], then either reads it out
/// of the already-opened [`CompoundArchive`] via
/// [`compound_format::open_input`] (compound segments) or `dir.open`s the
/// loose file (everything else) -- unchanged behavior for non-compound
/// segments, transparent compound-file reading for segments whose `.si` set
/// `is_compound_file`. Returns `Ok(None)` when the segment simply doesn't
/// have that file (e.g. a stored-fields-only segment has no `.tim`/`.tip`/
/// `.tmd`/`.doc`/`.pos`/`.pay`), the same "missing is not an error" contract
/// the call sites had before this helper existed.
fn open_segment_file(
    dir: &dyn Directory,
    compound: Option<&CompoundArchive>,
    files: &[String],
    ext: &str,
) -> Result<Option<Vec<u8>>> {
    let name = match find_segment_file_name(files, compound, ext) {
        Some(name) => name,
        None => return Ok(None),
    };
    match compound {
        Some(archive) => Ok(Some(
            compound_format::open_input(&archive.data, &archive.entries, &name)?
                .as_slice()
                .to_vec(),
        )),
        None => Ok(Some(dir.open(&name)?.to_vec())),
    }
}

/// `DirectoryReader.open(Directory)`-equivalent: reads the latest
/// `segments_N`, opens every listed segment (whichever files it actually
/// has), and computes each segment's `doc_base` automatically as the running
/// sum of every earlier segment's `maxDoc` -- the two things
/// [`OpenSegment::doc_base`]'s doc comment previously left entirely to the
/// caller.
#[derive(Debug)]
pub struct DirectoryReader {
    pub segment_infos: SegmentInfos,
    segments: Vec<SegmentReader>,
}

impl DirectoryReader {
    /// Opens every segment listed in the latest commit found in `dir`.
    pub fn open(dir: &dyn Directory) -> Result<Self> {
        let segment_infos = segment_infos::read_latest(dir)?;
        Self::open_at(dir, segment_infos)
    }

    /// Opens every segment listed in an already-parsed [`SegmentInfos`] --
    /// useful for tests that build a commit by hand rather than reading one
    /// off disk (see this module's unit tests).
    pub fn open_at(dir: &dyn Directory, segment_infos: SegmentInfos) -> Result<Self> {
        Self::open_at_reusing(dir, segment_infos, &[])
    }

    /// Shared by [`Self::open_at`] (no reuse candidates) and
    /// [`Self::open_if_changed`] (reuse candidates = the currently-open
    /// reader's segments): opens every segment in `segment_infos`, taking an
    /// already-open [`SegmentReader`] out of `reusable` instead of
    /// re-reading it from disk whenever one matches -- same `segment_name`
    /// *and* same `segment_id` *and* same `del_gen` as the new commit's entry
    /// (see [`Self::open_if_changed`]'s doc comment for why all three must
    /// match: a del_gen bump means the `.liv` file changed even though the
    /// segment's own postings/stored fields didn't).
    fn open_at_reusing(
        dir: &dyn Directory,
        segment_infos: SegmentInfos,
        reusable: &[SegmentReader],
    ) -> Result<Self> {
        let mut segments = Vec::with_capacity(segment_infos.segments.len());
        let mut doc_base = 0i32;
        for commit in &segment_infos.segments {
            let reused = reusable.iter().find(|r| {
                r.segment_name == commit.segment_name
                    && r.segment_id == commit.segment_id
                    && r.del_gen == commit.del_gen
            });
            let mut reader = match reused {
                Some(r) => r.clone_reader(),
                None => SegmentReader::open(dir, commit, doc_base)?,
            };
            reader.doc_base = doc_base;
            doc_base += reader.max_doc;
            segments.push(reader);
        }
        Ok(DirectoryReader {
            segment_infos,
            segments,
        })
    }

    /// `DirectoryReader.openIfChanged(DirectoryReader)`-equivalent: checks
    /// whether `dir`'s latest `segments_N` generation differs from the one
    /// `self` was opened from. Returns `Ok(None)` if nothing changed (same
    /// generation), matching real Lucene's convention that callers should
    /// keep using their existing reader rather than redundantly rebuild it.
    ///
    /// If the generation *did* change, builds a new [`DirectoryReader`] that
    /// reuses `self`'s already-open [`SegmentReader`]s for every segment
    /// that is unchanged in the new commit -- same segment name, same
    /// segment id, *and* same `del_gen` -- and opens fresh readers (reading
    /// `.si`/`.fnm`/postings/`.liv` from disk, exactly like
    /// [`Self::open_at`]) for anything new or whose `del_gen` bumped. A
    /// segment merge or delete replaces the segment name/id, so it always
    /// falls into "open fresh"; a same-segment delete-only commit keeps the
    /// same name/id but bumps `del_gen`, which this deliberately treats as
    /// "must reopen" so the new commit's `live_docs` (not `self`'s stale
    /// ones) is what queries see -- getting this backwards would silently
    /// serve deleted documents as live.
    ///
    /// **Deliberately excluded** (see this module's doc comment and
    /// `docs/parity.md`): this is a self-contained reopen, not a
    /// reader-pool-wide one -- real Lucene's `openIfChanged` operates through
    /// a shared `ReaderPool` so *every* open reader in a process benefits
    /// from one segment's reuse, and supports warm-up listeners
    /// (`ReaderManager`/`SearcherFactory`) called on newly-opened sub-readers
    /// before they're handed back. Neither exists here; each
    /// `open_if_changed` call only reuses its own receiver's readers.
    pub fn open_if_changed(&self, dir: &dyn Directory) -> Result<Option<Self>> {
        let latest = segment_infos::read_latest(dir)?;
        if latest.generation == self.segment_infos.generation {
            return Ok(None);
        }
        Some(Self::open_at_reusing(dir, latest, &self.segments)).transpose()
    }

    /// Every opened segment's own reader, in commit order.
    pub fn segment_readers(&self) -> &[SegmentReader] {
        &self.segments
    }

    /// Opens the per-segment `DocInput`/`PosInput`/`PayInput` values needed
    /// to build `OpenSegment`s -- see this module's doc comment for why this
    /// is a separate call from [`OpenedSegments::as_open_segments`].
    pub fn open_segments(&self) -> Result<OpenedSegments<'_>> {
        let mut doc_ins = Vec::with_capacity(self.segments.len());
        let mut pos_ins = Vec::with_capacity(self.segments.len());
        let mut pay_ins = Vec::with_capacity(self.segments.len());
        for seg in &self.segments {
            let doc_in = match &seg.doc_buf {
                Some(buf) => Some(DocInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            let pos_in = match &seg.pos_buf {
                Some(buf) => Some(PosInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            let pay_in = match &seg.pay_buf {
                Some(buf) => Some(PayInput::open(buf, &seg.segment_id, &seg.segment_suffix)?),
                None => None,
            };
            doc_ins.push(doc_in);
            pos_ins.push(pos_in);
            pay_ins.push(pay_in);
        }
        Ok(OpenedSegments {
            readers: &self.segments,
            doc_ins,
            pos_ins,
            pay_ins,
        })
    }
}

/// The `DocInput`/`PosInput`/`PayInput` values [`DirectoryReader::open_segments`]
/// constructed, plus a reference back to their owning [`SegmentReader`]s --
/// see this module's doc comment for why this intermediate type exists.
pub struct OpenedSegments<'a> {
    readers: &'a [SegmentReader],
    doc_ins: Vec<Option<DocInput<'a>>>,
    pos_ins: Vec<Option<PosInput<'a>>>,
    pay_ins: Vec<Option<PayInput<'a>>>,
}

impl<'a> OpenedSegments<'a> {
    /// The final step: a `Vec<OpenSegment>` ready to feed directly into
    /// `search_term_query_multi_segment`/`search_boolean_query_multi_segment`
    /// -- `doc_base` already computed, every field already opened.
    pub fn as_open_segments(&self) -> Vec<OpenSegment<'_>> {
        self.readers
            .iter()
            .enumerate()
            .map(|(i, r)| OpenSegment {
                fields: &r.fields,
                doc_in: self.doc_ins[i].as_ref(),
                pos_in: self.pos_ins[i].as_ref(),
                pay_in: self.pay_ins[i].as_ref(),
                live_docs: r.live_docs.as_ref(),
                doc_base: r.doc_base,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_segment::search_term_query_multi_segment;
    use crate::query::TermQuery;
    use lucene_index::segment_infos::LuceneVersion;
    use lucene_store::FsDirectory;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        ))
    }

    /// Opens the real single-segment `blocktree_index` fixture end-to-end
    /// through `DirectoryReader::open` (reading its real `segments_1`, not a
    /// hand-built `SegmentInfos`), confirming file-existence detection finds
    /// the real postings files and produces a working, single-segment
    /// `OpenSegment` whose query results match task #41's manually-wired
    /// `search_term_query_multi_segment` test.
    #[test]
    fn opens_real_single_segment_fixture_and_matches_manual_wiring() {
        let dir = FsDirectory::open(fixture_dir());
        let reader = DirectoryReader::open(&dir).expect("open real segments_1 commit");

        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];
        assert_eq!(seg.doc_base, 0);
        assert!(seg.doc_buf.is_some(), "fixture has real .doc postings");

        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();

        let query = TermQuery::new("body", "cat");
        let norms = [None];
        let hits = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        // Same fixture/query task #41's own multi_segment.rs test uses
        // (`search_term_query_multi_segment_merges_two_real_segments`,
        // segment 0 half): must be non-empty and score-descending.
        assert!(!hits.is_empty());
        for pair in hits.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    /// A stored-fields-only segment (no postings/doc-values/norms/term
    /// vectors at all -- exactly `segment_writer.rs`'s
    /// `flush_stored_only_segment` output) must open cleanly with no
    /// postings files, not error out just because they're missing.
    #[test]
    fn stored_fields_only_segment_opens_without_postings_files() {
        use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let field_infos_list = vec![FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }];
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("hello".to_string()),
            }],
        }];

        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        let commit_info = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &field_infos_list,
            &docs,
            false,
        )
        .expect("flush stored-only segment");

        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![commit_info],
            user_data: vec![],
        };

        let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open stored-only");
        let seg = &reader.segment_readers()[0];
        assert!(seg.doc_buf.is_none());
        assert!(seg.pos_buf.is_none());
        assert!(seg.pay_buf.is_none());
        assert!(seg.live_docs.is_none());
        assert_eq!(seg.max_doc, 1);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// A stored-fields-only segment flushed with `use_compound_file: true`
    /// (so only `_0.cfs`/`_0.cfe`/`_0.si` exist on disk -- no loose `.fnm`)
    /// must open through `DirectoryReader` with the same doc count and "no
    /// postings" shape as the non-compound flush of the same documents,
    /// proving the `.fnm` read out of the compound archive succeeds (a
    /// missing/unreadable `.fnm` would fail `DirectoryReader::open_at`
    /// itself, since field infos are required to open any segment). Also
    /// confirms a *loose* `.fnm` genuinely does not exist next to the
    /// compound segment -- so success here can only come from reading
    /// through `.cfs`, not from an accidental loose-file fallback. Does NOT
    /// compare field name/number directly (`SegmentReader` doesn't expose
    /// parsed field infos publicly); [`compound_file_segment_opens_and_is_queryable`]
    /// below covers that indirectly by running a real term query, which
    /// only succeeds if the field/term data read out of `.cfs` is correct.
    #[test]
    fn compound_flushed_segment_opens_with_field_infos_matching_non_compound_flush() {
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("hello".to_string()),
            }],
        }];

        // Loose flush of the same documents, for a same-shape baseline.
        let loose_commit = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            false,
        )
        .expect("flush loose segment");
        assert!(dir_path.join("_0.fnm").exists());

        let loose_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![loose_commit],
            user_data: vec![],
        };
        let loose_reader = DirectoryReader::open_at(&dir, loose_infos).expect("open loose");
        let loose_seg = &loose_reader.segment_readers()[0];

        // Compound flush of the same documents into a second segment, in
        // the same directory (a distinct segment name/id so both coexist).
        let compound_commit = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_1",
            [8u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            true,
        )
        .expect("flush compound segment");
        assert!(dir_path.join("_1.cfs").exists());
        assert!(dir_path.join("_1.cfe").exists());
        assert!(
            !dir_path.join("_1.fnm").exists(),
            "compound flush must not also write a loose .fnm"
        );

        let compound_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![compound_commit],
            user_data: vec![],
        };
        let compound_reader =
            DirectoryReader::open_at(&dir, compound_infos).expect("open compound segment");
        let compound_seg = &compound_reader.segment_readers()[0];

        // Both must have the same doc count and same "no postings" shape --
        // the compound one exercised entirely through `.cfs`/`.cfe`.
        assert_eq!(compound_seg.max_doc, loose_seg.max_doc);
        assert!(compound_seg.doc_buf.is_none());
        assert!(compound_seg.pos_buf.is_none());
        assert!(compound_seg.pay_buf.is_none());
        assert!(compound_seg.live_docs.is_none());

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// A compound segment whose `.cfs` data has been truncated (corrupting
    /// its header/footer) must surface as a typed error out of
    /// [`CompoundArchive::open`]'s `compound_format::check_data_header_footer`
    /// call, not panic or silently read garbage.
    #[test]
    fn compound_file_segment_with_truncated_cfs_is_a_typed_error() {
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("hello".to_string()),
            }],
        }];
        let commit = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            true,
        )
        .expect("flush compound segment");

        let cfs_path = dir_path.join("_0.cfs");
        let mut bytes = std::fs::read(&cfs_path).unwrap();
        bytes.truncate(bytes.len() / 2);
        std::fs::write(&cfs_path, bytes).unwrap();

        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![commit],
            user_data: vec![],
        };
        let err = DirectoryReader::open_at(&dir, segment_infos)
            .expect_err("truncated .cfs must not open successfully");
        assert!(
            matches!(err, Error::CompoundFormat(_)),
            "expected a typed CompoundFormat error, got {err:?}"
        );

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// A compound segment whose `.cfe` entry table is truncated must
    /// surface as a typed error out of `compound_format::parse_entries`
    /// rather than panicking on an out-of-bounds read.
    #[test]
    fn compound_file_segment_with_truncated_cfe_is_a_typed_error() {
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("hello".to_string()),
            }],
        }];
        let commit = lucene_index::segment_writer::flush_stored_only_segment(
            &dir,
            "_0",
            [7u8; ID_LENGTH],
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            true,
        )
        .expect("flush compound segment");

        let cfe_path = dir_path.join("_0.cfe");
        let mut bytes = std::fs::read(&cfe_path).unwrap();
        bytes.truncate(bytes.len() / 2);
        std::fs::write(&cfe_path, bytes).unwrap();

        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation: 1,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 1,
            counter: 1,
            min_segment_lucene_version: None,
            segments: vec![commit],
            user_data: vec![],
        };
        let err = DirectoryReader::open_at(&dir, segment_infos)
            .expect_err("truncated .cfe must not open successfully");
        assert!(
            matches!(err, Error::CompoundFormat(_)),
            "expected a typed CompoundFormat error, got {err:?}"
        );

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Confirms `doc_base` is computed as the running sum of previous
    /// segments' `maxDoc`, by opening the same real fixture segment twice
    /// under one hand-built two-segment `SegmentInfos` (the same "open the
    /// same fixture twice" trick task #41's own cross-segment tests use).
    #[test]
    fn doc_base_is_running_sum_of_previous_max_docs() {
        let dir = FsDirectory::open(fixture_dir());
        let commit = read_commit_info(&dir);

        let mut segment_infos = read_commit(&dir);
        segment_infos.segments.push(commit);

        let reader = DirectoryReader::open_at(&dir, segment_infos).expect("open two segments");
        assert_eq!(reader.segment_readers().len(), 2);
        let first_max_doc = reader.segment_readers()[0].max_doc;
        assert_eq!(reader.segment_readers()[0].doc_base, 0);
        assert_eq!(reader.segment_readers()[1].doc_base, first_max_doc);
    }

    fn read_commit(dir: &FsDirectory) -> SegmentInfos {
        segment_infos::read_latest(dir).expect("read real segments_1")
    }

    fn read_commit_info(dir: &FsDirectory) -> segment_infos::SegmentCommitInfo {
        read_commit(dir).segments[0].clone()
    }

    /// A segment with real deletions (`del_gen != -1`) must open its `.liv`
    /// file and populate `live_docs`, cross-checked against the real
    /// `del_count` -- exercises the branch
    /// `opens_real_single_segment_fixture_and_matches_manual_wiring` and
    /// `stored_fields_only_segment_opens_without_postings_files` both skip.
    #[test]
    fn segment_with_deletions_opens_live_docs() {
        let dir_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/live_docs_index/"
        ));
        let dir = FsDirectory::open(&dir_path);
        let reader = DirectoryReader::open(&dir).expect("open real live_docs_index commit");

        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];
        assert_eq!(seg.max_doc, 5);
        let live = seg.live_docs.as_ref().expect("segment has deletions");
        assert_eq!(live.cardinality(), 3);
        assert!(live.get(0) && live.get(2) && live.get(4));
        assert!(!live.get(1) && !live.get(3));

        // Also has real postings (.tim/.tip/.tmd/.doc) but no .pos/.pay,
        // exercising the "some but not all optional postings files" path.
        assert!(seg.doc_buf.is_some());
        assert!(seg.pos_buf.is_none());
        assert!(seg.pay_buf.is_none());
    }

    /// A real, Java-written compound-file (`.cfs`/`.cfe`) segment must open
    /// through `DirectoryReader` exactly like a loose segment would -- same
    /// field infos, same doc count, same postings query results -- reading
    /// every sub-file (`.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`, this fixture has no
    /// `.pos`/`.pay`) out of the `.cfs` archive instead of failing to find
    /// loose files that were never written.
    #[test]
    fn compound_file_segment_opens_and_is_queryable() {
        let dir_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/compound_index/"
        ));
        let dir = FsDirectory::open(&dir_path);
        let reader = DirectoryReader::open(&dir).expect("open real compound-file segment");

        assert_eq!(reader.segment_readers().len(), 1);
        let seg = &reader.segment_readers()[0];
        assert_eq!(seg.max_doc, 5);
        assert!(
            seg.doc_buf.is_some(),
            "compound fixture has real .doc postings"
        );
        assert!(seg.pos_buf.is_none(), "compound fixture has no .pos");
        assert!(seg.pay_buf.is_none(), "compound fixture has no .pay");

        // GenCompoundFormat.java indexes docs 0..5 with a `StringField("id",
        // ...)` -- confirms the `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc` bytes
        // pulled out of the compound archive are actually usable for a real
        // query, not merely non-empty.
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let query = TermQuery::new("id", "3");
        let norms = [None];
        let hits = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 3);
    }

    /// A segment whose `.si` lists no `.fnm` file at all is corrupt (every
    /// real segment has field infos); this must surface as an error, not a
    /// panic.
    #[test]
    fn segment_missing_fnm_file_is_an_error() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let si = SegmentInfo {
            id: [1u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 3,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![], // no .fnm listed
            attributes: vec![],
            index_sort: None,
        };
        let si_bytes = segment_info::write(&si, "");
        std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [1u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };
        let err = SegmentReader::open(&dir, &commit, 0).expect_err("missing .fnm must error");
        assert!(matches!(err, Error::Store(_)));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// `.si` listing only some of `.tim`/`.tip`/`.tmd` (a corrupt/partial
    /// segment -- never produced by a real writer) must be rejected rather
    /// than silently treated as "no postings" or panicking on an `unwrap`.
    #[test]
    fn partial_blocktree_files_is_an_error() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);

        let field_infos_list = vec![lucene_codecs::field_infos::FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: lucene_codecs::field_infos::IndexOptions::None,
            doc_values_type: lucene_codecs::field_infos::DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }];
        let fnm = lucene_codecs::field_infos::write(&field_infos_list, &[2u8; ID_LENGTH], "");
        std::fs::write(dir_path.join("_0.fnm"), &fnm).unwrap();
        // Only `.tim` present -- `.tip`/`.tmd` deliberately missing.
        std::fs::write(dir_path.join("_0_x.tim"), [0u8; 4]).unwrap();

        let si = SegmentInfo {
            id: [2u8; ID_LENGTH],
            version: segment_info::LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 1,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec!["_0.fnm".to_string(), "_0_x.tim".to_string()],
            attributes: vec![],
            index_sort: None,
        };
        let si_bytes = segment_info::write(&si, "");
        std::fs::write(dir_path.join("_0.si"), &si_bytes).unwrap();

        let commit = segment_infos::SegmentCommitInfo {
            segment_name: "_0".to_string(),
            segment_id: [2u8; ID_LENGTH],
            codec_name: "Lucene104".to_string(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: vec![],
            dv_update_files: vec![],
        };
        let err = SegmentReader::open(&dir, &commit, 0).expect_err("partial postings must error");
        assert!(matches!(err, Error::PartialBlockTreeFiles { found: 1, .. }));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    fn stored_only_field_infos() -> Vec<lucene_codecs::field_infos::FieldInfo> {
        use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
        vec![FieldInfo {
            name: "title".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: lucene_codecs::field_infos::VectorEncoding::Byte,
            vector_similarity_function:
                lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
        }]
    }

    fn flush_stored_only(
        dir: &FsDirectory,
        segment_name: &str,
        segment_id: [u8; ID_LENGTH],
        text: &str,
    ) -> segment_infos::SegmentCommitInfo {
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(text.to_string()),
            }],
        }];
        let lucene_version = segment_info::LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        };
        lucene_index::segment_writer::flush_stored_only_segment(
            dir,
            segment_name,
            segment_id,
            "Lucene104",
            lucene_version,
            &stored_only_field_infos(),
            &docs,
            false,
        )
        .expect("flush stored-only segment")
    }

    fn write_commit(
        dir: &FsDirectory,
        generation: i64,
        segments: Vec<segment_infos::SegmentCommitInfo>,
    ) -> SegmentInfos {
        let segment_infos = SegmentInfos {
            id: [9u8; ID_LENGTH],
            generation,
            format_version: segment_infos::VERSION_86,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: generation,
            counter: segments.len() as i64,
            min_segment_lucene_version: None,
            segments,
            user_data: vec![],
        };
        segment_infos::write(&segment_infos, dir).expect("write segments_N");
        segment_infos
    }

    /// Reopening with no writes since `self` was opened returns `None` --
    /// real `openIfChanged`'s "nothing to do" convention, checked purely off
    /// the `segments_N` generation (no need to re-diff segment lists).
    #[test]
    fn open_if_changed_returns_none_when_generation_unchanged() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        let reopened = reader.open_if_changed(&dir).expect("open_if_changed");
        assert!(reopened.is_none());

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// Reopening after a genuinely new segment was added returns `Some` with
    /// both segments visible, correct `doc_base`s, and the *first* segment's
    /// reader genuinely reused rather than re-read from disk: proven by
    /// deleting the old segment's on-disk files before reopening -- if
    /// `open_if_changed` tried to re-open segment `_0` from disk it would
    /// fail (file not found), so success here is only possible via reuse.
    #[test]
    fn open_if_changed_reuses_unchanged_segment_and_opens_new_one() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0.clone()]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert_eq!(reader.segment_readers().len(), 1);

        let commit1 = flush_stored_only(&dir, "_1", [2u8; ID_LENGTH], "world");
        write_commit(&dir, 2, vec![commit0.clone(), commit1]);

        // Remove segment _0's on-disk files: a fresh open of _0 would now
        // fail, so if open_if_changed still succeeds and reports the right
        // doc counts, it must have reused the already-open reader instead of
        // re-reading the (now-missing) files.
        for ext in [".fdt", ".fdx", ".fdm", ".fnm"] {
            std::fs::remove_file(dir_path.join(format!("_0{ext}"))).ok();
        }

        let reopened = reader
            .open_if_changed(&dir)
            .expect("open_if_changed")
            .expect("segments_2 differs from segments_1");
        assert_eq!(reopened.segment_readers().len(), 2);
        assert_eq!(reopened.segment_readers()[0].segment_name, "_0");
        assert_eq!(reopened.segment_readers()[0].doc_base, 0);
        assert_eq!(reopened.segment_readers()[1].segment_name, "_1");
        assert_eq!(reopened.segment_readers()[1].doc_base, 1);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// The correctness-critical case: a del_gen-only change on an *existing*
    /// segment (no new segment, same name/id) must NOT be served from the
    /// reused reader's stale `live_docs` -- it must reopen that segment and
    /// pick up the new deletions.
    #[test]
    fn open_if_changed_reopens_segment_whose_del_gen_changed() {
        let dir_path = tempdir();
        let dir = FsDirectory::open(&dir_path);
        let commit0 = flush_stored_only(&dir, "_0", [1u8; ID_LENGTH], "hello");
        write_commit(&dir, 1, vec![commit0.clone()]);

        let reader = DirectoryReader::open(&dir).expect("open segments_1");
        assert!(reader.segment_readers()[0].live_docs.is_none());

        // Delete doc 0 in segment _0: bumps del_gen, writes a new .liv.
        let commit0_deleted = lucene_index::deletes::apply_deletes(&dir, &commit0, None, 1, [0i32])
            .expect("apply delete");
        assert_eq!(commit0_deleted.del_gen, 1);
        write_commit(&dir, 2, vec![commit0_deleted]);

        let reopened = reader
            .open_if_changed(&dir)
            .expect("open_if_changed")
            .expect("segments_2 differs from segments_1");
        assert_eq!(reopened.segment_readers().len(), 1);
        let live = reopened.segment_readers()[0]
            .live_docs
            .as_ref()
            .expect("del_gen change must be picked up as new live_docs");
        assert!(!live.get(0), "doc 0 must now be marked deleted");

        std::fs::remove_dir_all(&dir_path).ok();
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "lucene-rust-directory-reader-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    // -- "NRT reopen after sparse doc-values commits" (docs/parity.md) --
    //
    // The rest of this module never previously read a segment's `.dvm`/
    // `.dvd` at all (see this module's doc comment, before this task) --
    // every other doc-values test in this crate (`doc_value_query.rs`,
    // `facets.rs`) fed hand-loaded fixture bytes straight to the low-level
    // `lucene_codecs::doc_values` readers, never through
    // `DirectoryReader`/`SegmentReader`. These tests are the real
    // `IndexWriter` -> commit -> NRT reopen -> read-back path: a genuinely
    // sparse NUMERIC field (some docs opted out entirely), read through a
    // `DirectoryReader` opened by this module, refreshed via
    // `open_if_changed` across a second commit that adds a brand-new sparse
    // segment.
    mod sparse_doc_values_nrt {
        use super::*;
        use lucene_codecs::doc_values;
        use lucene_codecs::field_infos::{
            DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
            VectorSimilarityFunction,
        };
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_index::segment_info::LuceneVersion;

        fn version() -> LuceneVersion {
            LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            }
        }

        fn stored_only_field(name: &str, number: i32) -> FieldInfo {
            FieldInfo {
                name: name.to_string(),
                number,
                store_term_vectors: false,
                omit_norms: false,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: IndexOptions::None,
                doc_values_type: DocValuesType::None,
                doc_values_skip_index_type: DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: vec![],
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: VectorEncoding::Float32,
                vector_similarity_function: VectorSimilarityFunction::Euclidean,
            }
        }

        fn numeric_field(name: &str, number: i32) -> FieldInfo {
            FieldInfo {
                doc_values_type: DocValuesType::Numeric,
                ..stored_only_field(name, number)
            }
        }

        fn doc_with_score(id: &str, score: i64) -> Document {
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(id.to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(score),
                    },
                ],
            }
        }

        fn doc_without_score(id: &str) -> Document {
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                }],
            }
        }

        /// The base scenario: one segment, one commit, a sparse NUMERIC
        /// field with a present/absent/present pattern, read back correctly
        /// through `DirectoryReader::open` -- not a hand-loaded fixture, the
        /// actual bytes `IndexWriter::commit` wrote.
        #[test]
        fn nrt_reader_reads_sparse_numeric_field_after_first_commit() {
            let dir_path = tempdir();
            let dir = FsDirectory::open(&dir_path);
            let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
            let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
            writer.set_doc_values_field(Some("score")).unwrap();

            writer.add_document(doc_with_score("a", 5));
            writer.add_document(doc_without_score("b"));
            writer.add_document(doc_with_score("c", 7));
            writer.commit().unwrap();

            let reader = DirectoryReader::open(&dir).expect("open segments_1");
            assert_eq!(reader.segment_readers().len(), 1);
            let seg = &reader.segment_readers()[0];

            let field_number = seg
                .field_infos()
                .field_by_number(1)
                .expect("score field present")
                .number;
            let meta = seg
                .doc_values_meta()
                .expect("sparse-committed segment must have .dvm/.dvd wired in");
            let entry = meta
                .numeric_entry(field_number)
                .expect("score has a NumericEntry");
            let data = seg.doc_values_data().expect("segment has .dvd bytes");

            assert_eq!(doc_values::numeric_value(data, entry, 0).unwrap(), Some(5));
            assert_eq!(
                doc_values::numeric_value(data, entry, 1).unwrap(),
                None,
                "doc 1 opted out of the field entirely -- sparse, not zero"
            );
            assert_eq!(doc_values::numeric_value(data, entry, 2).unwrap(), Some(7));

            std::fs::remove_dir_all(&dir_path).ok();
        }

        /// The genuine NRT-reopen scenario: a second commit adds a brand-new
        /// *second* segment, itself also sparse, with a different
        /// present/absent pattern. `open_if_changed` must refresh doc-values
        /// wiring for the newly-opened segment while continuing to serve the
        /// first (reused) segment's sparse data correctly too.
        #[test]
        fn nrt_reopen_reads_new_segments_sparse_numeric_values_correctly() {
            let dir_path = tempdir();
            let dir = FsDirectory::open(&dir_path);
            let fields = vec![stored_only_field("id", 0), numeric_field("score", 1)];
            let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
            writer.set_doc_values_field(Some("score")).unwrap();

            writer.add_document(doc_with_score("a", 5));
            writer.add_document(doc_without_score("b"));
            writer.commit().unwrap();

            let reader = DirectoryReader::open(&dir).expect("open segments_1");
            assert_eq!(reader.segment_readers().len(), 1);

            // Second commit: a new segment, sparse in the opposite pattern
            // (present, absent, present) so this is not just re-reading the
            // first segment's bytes.
            writer.add_document(doc_without_score("x"));
            writer.add_document(doc_with_score("y", 42));
            writer.add_document(doc_without_score("z"));
            writer.commit().unwrap();

            let reopened = reader
                .open_if_changed(&dir)
                .expect("open_if_changed")
                .expect("second commit differs from the first");
            assert_eq!(reopened.segment_readers().len(), 2);

            let seg0 = &reopened.segment_readers()[0];
            let seg1 = &reopened.segment_readers()[1];
            assert_eq!(seg0.doc_base, 0);
            assert_eq!(seg1.doc_base, 2);

            // First (reused) segment: still reads its original sparse values.
            let meta0 = seg0.doc_values_meta().expect("segment 0 has doc values");
            let entry0 = meta0.numeric_entry(1).unwrap();
            let data0 = seg0.doc_values_data().unwrap();
            assert_eq!(
                doc_values::numeric_value(data0, entry0, 0).unwrap(),
                Some(5)
            );
            assert_eq!(doc_values::numeric_value(data0, entry0, 1).unwrap(), None);

            // Second (freshly opened via reopen) segment: its own, different
            // sparse pattern.
            let meta1 = seg1
                .doc_values_meta()
                .expect("newly reopened segment must have doc-values wired in too");
            let entry1 = meta1.numeric_entry(1).unwrap();
            let data1 = seg1.doc_values_data().unwrap();
            assert_eq!(doc_values::numeric_value(data1, entry1, 0).unwrap(), None);
            assert_eq!(
                doc_values::numeric_value(data1, entry1, 1).unwrap(),
                Some(42)
            );
            assert_eq!(doc_values::numeric_value(data1, entry1, 2).unwrap(), None);

            std::fs::remove_dir_all(&dir_path).ok();
        }
    }

    /// End-to-end exercise of `lucene_index::merge::merge_postings` (task:
    /// "wire a real end-to-end caller for postings merge logic") -- not a
    /// hand-built `MergeSource` fixture (see `merge.rs`'s own unit tests for
    /// that), but the real path: `IndexWriter` flushes two real segments,
    /// each with real `.tim`/`.tip`/`.tmd`/`.doc` postings
    /// (`set_postings_field`), a configured `MergePolicyConfig` triggers a
    /// real automatic merge inside `commit()` (`IndexWriter::auto_merge` ->
    /// `execute_merge`, which now opens each source's postings and builds
    /// `MergeSource::postings` from real on-disk data), and the merged
    /// result is read back through this crate's own `DirectoryReader` ->
    /// `SegmentReader` -> `search_term_query_multi_segment` stack -- the
    /// exact same read path a caller with a fully general multi-segment
    /// index already uses, run here against a single post-merge segment.
    ///
    /// (Points (`merge_points`) is not exercised here: this port's
    /// `IndexWriter` has no points write path at flush time at all yet --
    /// no `set_points_field`-equivalent, no points-carrying `Document`
    /// field shape -- so there is no real segment it could ever produce
    /// with `.kdm`/`.kdi`/`.kdd` files to merge in the first place. See
    /// `docs/parity.md` and `merge.rs`'s module doc comment.)
    mod postings_merge_e2e {
        use super::*;
        use crate::multi_segment::search_term_query_multi_segment;
        use crate::query::TermQuery;
        use lucene_codecs::field_infos::{
            DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
            VectorSimilarityFunction,
        };
        use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_index::merge_policy::MergePolicyConfig;
        use lucene_index::segment_info::LuceneVersion;

        fn version() -> LuceneVersion {
            LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            }
        }

        fn stored_only_field(name: &str, number: i32) -> FieldInfo {
            FieldInfo {
                name: name.to_string(),
                number,
                store_term_vectors: false,
                omit_norms: false,
                store_payloads: false,
                soft_deletes_field: false,
                parent_field: false,
                index_options: IndexOptions::None,
                doc_values_type: DocValuesType::None,
                doc_values_skip_index_type: DocValuesSkipIndexType::None,
                doc_values_gen: -1,
                attributes: vec![],
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: VectorEncoding::Float32,
                vector_similarity_function: VectorSimilarityFunction::Euclidean,
            }
        }

        fn body_field(number: i32) -> FieldInfo {
            FieldInfo {
                index_options: IndexOptions::DocsAndFreqs,
                ..stored_only_field("body", number)
            }
        }

        fn doc_with_body(id: &str, body: &str) -> Document {
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(id.to_string()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(body.to_string()),
                    },
                ],
            }
        }

        fn tight_merge_policy() -> MergePolicyConfig {
            MergePolicyConfig {
                max_merge_at_once: 10,
                segments_per_tier: 2,
                ..MergePolicyConfig::default()
            }
        }

        /// Flushes two real segments (one doc each, via two separate
        /// `commit()` calls) with real postings, then a third `commit()`
        /// crosses `tight_merge_policy`'s `segments_per_tier` threshold and
        /// triggers a real automatic merge down to one segment -- confirms
        /// via a real `TermQuery` that every source doc's terms survived
        /// the merge, not just that the merged segment's raw bytes exist.
        #[test]
        fn term_query_finds_docs_from_both_flushed_segments_after_automatic_merge() {
            let dir_path = tempdir();
            let dir = FsDirectory::open(&dir_path);
            let fields = vec![stored_only_field("id", 0), body_field(1)];
            let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
            writer.set_postings_field(Some("body")).unwrap();
            writer.set_merge_policy(Some(tight_merge_policy()));

            writer.add_document(doc_with_body("a", "quick fox"));
            writer.commit().unwrap();
            writer.add_document(doc_with_body("b", "lazy fox"));
            writer.commit().unwrap();
            // Crosses segments_per_tier (2) with a third one-doc commit,
            // triggering `auto_merge` to fold all three segments into one.
            writer.add_document(doc_with_body("c", "quick dog"));
            writer.commit().unwrap();

            let segments_after = writer.segment_infos().segments.clone();
            assert_eq!(
                segments_after.len(),
                1,
                "postings-carrying segments must merge down like any other"
            );
            drop(segments_after);

            // Opened through the real `DirectoryReader` -> `SegmentReader`
            // stack -- now that `SegmentReader::open`'s `.tim`-suffix
            // derivation special-cases this port's own no-codec-suffix
            // convention (segment_suffix == "", matching every write site in
            // `index_writer.rs`/`merge.rs`) the same way its `.dvm`-suffix
            // derivation already did, this no longer needs a manual
            // off-disk-file workaround.
            let reader = DirectoryReader::open(&dir).unwrap();
            assert_eq!(reader.segment_readers().len(), 1);
            let opened = reader.open_segments().unwrap();
            let segments = opened.as_open_segments();
            let norms = [None];

            // "fox" was indexed by docs "a" and "b" -- originally in two
            // different flushed segments, now merged into one.
            let fox_hits = search_term_query_multi_segment(
                &segments,
                &TermQuery::new("body", "fox"),
                &norms,
                10,
            )
            .unwrap();
            assert_eq!(
                fox_hits.len(),
                2,
                "\"fox\" must resolve to both merged docs"
            );

            // "quick" was indexed by docs "a" and "c".
            let quick_hits = search_term_query_multi_segment(
                &segments,
                &TermQuery::new("body", "quick"),
                &norms,
                10,
            )
            .unwrap();
            assert_eq!(
                quick_hits.len(),
                2,
                "\"quick\" must resolve to both merged docs"
            );

            // "lazy" was indexed by doc "b" only.
            let lazy_hits = search_term_query_multi_segment(
                &segments,
                &TermQuery::new("body", "lazy"),
                &norms,
                10,
            )
            .unwrap();
            assert_eq!(lazy_hits.len(), 1);

            std::fs::remove_dir_all(&dir_path).ok();
        }
    }
}
