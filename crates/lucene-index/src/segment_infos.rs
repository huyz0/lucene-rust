//! Port of `org.apache.lucene.index.SegmentInfos` (`segments_N` commit files).
//!
//! This is the top of the read path: `segments_N` is the file a `DirectoryReader`
//! opens first — it lists every segment in the commit (by name + id + codec) along
//! with per-segment delete/DV-update generations, but does *not* embed the segments'
//! own metadata (doc count, compound-file flag, ...). That lives in each segment's
//! `.si` file, parsed separately by [`crate::segment_info`]. Callers resolve
//! `SegmentCommitInfo::segment_name` to `<name>.si` themselves — this module has no
//! `Directory` dependency yet (Phase 1, still to come).
//!
//! Wire format (all ints little-endian unless noted "BE"; header/footer/BE
//! primitives per `lucene_store::codec_util`):
//! ```text
//! Header       --> IndexHeader(codec="segments", version in [VERSION_74, VERSION_CURRENT],
//!                   id, suffix=generation formatted base-36)
//! LuceneVersion --> vint major, vint minor, vint bugfix   (note: vint here, NOT the
//!                    fixed-i32 triple `.si` uses for its own SegVersion)
//! IndexCreatedVersionMajor --> vint
//! Version      --> BEi64             (commit's own monotonic version counter)
//! Counter      --> vlong             (next segment-name counter)
//! NumSegments  --> BEi32
//! MinSegmentLuceneVersion --> vint triple, present iff NumSegments > 0
//! per segment:
//!   SegName        --> String
//!   SegID          --> [u8; 16]
//!   CodecName      --> String
//!   DelGen         --> BEi64
//!   DelCount       --> BEi32
//!   FieldInfosGen  --> BEi64
//!   DocValuesGen   --> BEi64
//!   SoftDelCount   --> BEi32
//!   SciIdMarker    --> u8 (only if format > VERSION_74); 1 => SciId: [u8; 16] follows
//!   FieldInfosFiles --> SetOfStrings
//!   NumDVFields    --> BEi32
//!   per DV field: FieldNumber --> BEi32, Files --> SetOfStrings
//!   NumOverlayFields --> BEi32 (only if format >= VERSION_10_6)
//!   per overlay field: FieldNumber --> BEi32, BaseGen --> BEi64,
//!                      NumDeltas --> BEi32, DeltaGen --> BEi64 (x NumDeltas)
//! UserData     --> MapOfStrings
//! Footer
//! ```

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;

const CODEC_NAME: &str = "segments";
pub const VERSION_74: i32 = 9;
pub const VERSION_86: i32 = 10;
/// Lucene 10.6: every `segments_N` carries per-field doc-values overlay
/// generations after each segment's update files.
pub const VERSION_10_6: i32 = 11;
const VERSION_CURRENT: i32 = VERSION_10_6;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid segment count: {0}")]
    InvalidSegmentCount(i32),
    #[error("invalid deletion count: {0} vs maxDoc unknown at this layer (segment={1})")]
    InvalidDeletionCount(i32, String),
    #[error("invalid SegmentCommitInfo ID marker: {0}")]
    InvalidSciIdMarker(u8),
    #[error("invalid doc-values field count: {0} (segment={1})")]
    InvalidDocValuesFieldCount(i32, String),
    #[error(
        "creation version [{created}.x] can't be greater than the version that wrote the segment \
         infos: [{major}.{minor}.{bugfix}]"
    )]
    CreatedVersionAheadOfWriter {
        created: i32,
        major: i32,
        minor: i32,
        bugfix: i32,
    },
    #[error("illegal {which} version: {value}")]
    IllegalVersion { which: &'static str, value: i32 },
    /// A generation (`delGen`, `fieldInfosGen`, `docValuesGen`) or a
    /// commit-wide counter (`version`, `counter`) outside the range this port
    /// will accept. See [`MAX_GENERATION`].
    #[error("invalid {which}: {value} (segment={segment:?})")]
    InvalidGeneration {
        which: &'static str,
        value: i64,
        segment: String,
    },
}

/// The largest generation (or commit counter) [`parse`] will accept off disk.
///
/// Lucene has no such cap: Java reads every generation as a bare `long` and
/// derives the next one with `gen + 1`, which in Java silently wraps and in
/// Rust **panics** in a debug build. Every generation in this module is on the
/// receiving end of exactly that shape --
/// [`SegmentCommitInfo::next_write_del_gen`] and its two twins,
/// [`SegmentCommitInfo::advance_del_gen`] and its twins,
/// [`crate::index_file_deleter::IndexFileDeleter::inflate_gens`], and
/// `update_document`'s `generation`/`version` bumps -- so a single 8-byte flip
/// in a `segments_N` turns every later commit into a panic.
///
/// Capping the value on the way *in* is what makes all of those `+ 1`s
/// provably safe, and half the `i64` range is the cap that needs no
/// justification: a generation is advanced at most once per file this port
/// writes, so reaching `i64::MAX` from `MAX_GENERATION` would take 2^62 more
/// index writes -- while any value a real Lucene index can carry is smaller
/// than 2^62 by a factor no machine will close.
pub const MAX_GENERATION: i64 = i64::MAX / 2;

/// Bounds-checks one generation read off `segments_N`. `-1` is Lucene's "no
/// such file" sentinel (`IndexFileNames.fileNameFromGeneration` returns `null`
/// for it); `0` means "no generation suffix"; anything positive is a real
/// generation. Anything below `-1` would produce a `_-5.liv`-shaped file name
/// no Lucene can read, and anything above [`MAX_GENERATION`] is corruption by
/// construction.
fn check_generation(which: &'static str, value: i64, segment: &str) -> Result<i64> {
    if !(-1..=MAX_GENERATION).contains(&value) {
        return Err(Error::InvalidGeneration {
            which,
            value,
            segment: segment.to_string(),
        });
    }
    Ok(value)
}

/// The in-process counterpart of [`check_generation`], for the three
/// `set_next_write_*_gen` setters.
///
/// Those setters take a bare `i64` from a caller, and their fields are `pub`
/// on a `pub` type -- so nothing in the type system re-establishes the
/// [`MAX_GENERATION`] bound between an `lucene-ffi` caller and
/// [`SegmentCommitInfo::advance_del_gen`]. A `debug_assert` is the honest
/// enforcer for that: an in-process value is a *caller* contract, not a
/// disk-corruption one, and the write path
/// ([`check_writable_generations`]) is what stops a violated contract from
/// ever reaching a file.
fn debug_assert_generation(which: &'static str, value: i64) {
    debug_assert!(
        (-1..=MAX_GENERATION).contains(&value),
        "{which} = {value} is outside -1..={MAX_GENERATION}; a generation this \
         large cannot be advanced or written"
    );
}

/// The write half of [`check_generation`], applied to every generation and
/// counter a commit is about to serialize.
///
/// This is what closes the round trip, and it is not belt-and-braces: the
/// derivations this module performs are all `+ 1`, so a commit read back at
/// exactly [`MAX_GENERATION`] would derive `MAX_GENERATION + 1`, serialize it,
/// and produce a `segments_N` that [`parse`] then **refuses** -- an index this
/// port wrote and can no longer open. Refusing the *commit* is the honest
/// failure: nothing is written, the previous `segments_N` stays current, and
/// the caller is told which counter ran out rather than discovering it on the
/// next open.
///
/// Reaching it requires a generation within 1 of `i64::MAX / 2`, which no
/// sequence of real index writes can produce -- see [`MAX_GENERATION`]. What
/// makes the check worth its two comparisons per segment is that a *file name*
/// can carry an arbitrary value into these fields
/// ([`crate::index_file_deleter::IndexFileDeleter::inflate_gens`]), so
/// "unreachable by writing" is not the same as "unreachable".
fn check_writable_generations(segment_infos: &SegmentInfos) -> Result<()> {
    check_generation("generation", segment_infos.generation, "")?;
    check_generation("version", segment_infos.version, "")?;
    check_generation("counter", segment_infos.counter, "")?;
    for sci in &segment_infos.segments {
        check_generation("delGen", sci.del_gen, &sci.segment_name)?;
        check_generation("fieldInfosGen", sci.field_infos_gen, &sci.segment_name)?;
        check_generation("docValuesGen", sci.doc_values_gen, &sci.segment_name)?;
    }
    Ok(())
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
}

/// One segment's entry in a commit: everything `segments_N` records about it,
/// *excluding* what lives in the segment's own `.si` file.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentCommitInfo {
    pub segment_name: String,
    pub segment_id: [u8; ID_LENGTH],
    pub codec_name: String,
    pub del_gen: i64,
    pub del_count: i32,
    pub field_infos_gen: i64,
    pub doc_values_gen: i64,
    pub soft_del_count: i32,
    /// Present from format > VERSION_74 only.
    pub sci_id: Option<[u8; ID_LENGTH]>,
    pub field_infos_files: Vec<String>,
    /// field number -> doc-values update files for that field.
    pub dv_update_files: Vec<(i32, Vec<String>)>,
    /// Incremental doc-values overlays (format >= [`VERSION_10_6`]):
    /// `(field number, base generation, delta generations newest first)`.
    /// Empty unless the writer that made the commit enabled overlays
    /// (`IndexWriterConfig.setMaxDocValuesOverlays`).
    pub dv_overlays: Vec<(i32, i64, Vec<i64>)>,
    /// `SegmentCommitInfo.nextWriteDelGen`: the generation the *next* `.liv`
    /// this segment produces will be written at. Java's constructor derives
    /// it as `delGen == -1 ? 1 : delGen + 1` and
    /// `SegmentInfos.inflateGens` then pushes it past any higher-generation
    /// file a crashed session left in the directory, so a name that session
    /// may already have written is never handed out again.
    ///
    /// **Not serialized** — neither here nor in Java (`segments_N` records
    /// only `delGen` itself). `0` is this port's "not explicitly set"
    /// sentinel: [`SegmentCommitInfo::next_write_del_gen`] then returns
    /// Java's derived value. Keeping the sentinel rather than eagerly
    /// deriving means every construction site that does not care can write
    /// `..Default::default()` and still get Java's constructor semantics.
    pub next_write_del_gen: i64,
    /// `SegmentCommitInfo.nextWriteFieldInfosGen`, same `0`-means-derive
    /// convention as [`Self::next_write_del_gen`].
    pub next_write_field_infos_gen: i64,
    /// `SegmentCommitInfo.nextWriteDocValuesGen`, same `0`-means-derive
    /// convention as [`Self::next_write_del_gen`].
    pub next_write_doc_values_gen: i64,
    /// `SegmentCommitInfo.bufferedDeletesGen`: the
    /// [`crate::buffered_updates::BufferedUpdatesStream`] generation this
    /// segment was published at. A frozen delete packet applies to this
    /// segment iff this value is `<=` the packet's `delGen`, which is the
    /// single rule that makes a delete reach the segments that existed when
    /// it was issued and no others (see
    /// [`crate::buffered_updates::FrozenBufferedUpdates::applies_to`]).
    ///
    /// **Not serialized**, exactly as in Java. `-1` is Java's default, for a
    /// segment read back from a commit: everything in a commit predates every
    /// delete a fresh writer session can issue.
    pub buffered_deletes_gen: i64,
}

impl Default for SegmentCommitInfo {
    /// Java's `SegmentCommitInfo` constructor defaults for a segment with no
    /// generational files yet: every generation `-1` ("none"), no counts, no
    /// id. The three `next_write_*_gen` fields default to the `0` sentinel
    /// (see [`SegmentCommitInfo::next_write_del_gen`]), which derives to
    /// Java's `1`.
    fn default() -> Self {
        SegmentCommitInfo {
            segment_name: String::new(),
            segment_id: [0u8; ID_LENGTH],
            codec_name: String::new(),
            del_gen: -1,
            del_count: 0,
            field_infos_gen: -1,
            doc_values_gen: -1,
            soft_del_count: 0,
            sci_id: None,
            field_infos_files: Vec::new(),
            dv_update_files: Vec::new(),
            dv_overlays: Vec::new(),
            next_write_del_gen: 0,
            next_write_field_infos_gen: 0,
            next_write_doc_values_gen: 0,
            buffered_deletes_gen: -1,
        }
    }
}

impl SegmentCommitInfo {
    /// `SegmentCommitInfo.getNextDelGen()`: the generation the next `.liv`
    /// for this segment gets. Java's constructor sets it to
    /// `delGen == -1 ? 1 : delGen + 1`; `inflateGens` may raise it.
    pub fn next_write_del_gen(&self) -> i64 {
        derive_next_gen(self.next_write_del_gen, self.del_gen)
    }

    /// `SegmentCommitInfo.getNextFieldInfosGen()`.
    pub fn next_write_field_infos_gen(&self) -> i64 {
        derive_next_gen(self.next_write_field_infos_gen, self.field_infos_gen)
    }

    /// `SegmentCommitInfo.getNextDocValuesGen()`.
    pub fn next_write_doc_values_gen(&self) -> i64 {
        derive_next_gen(self.next_write_doc_values_gen, self.doc_values_gen)
    }

    /// `SegmentCommitInfo.setNextWriteDelGen(long)` — used by
    /// `SegmentInfos.inflateGens` to push past a crashed session's leftovers.
    pub fn set_next_write_del_gen(&mut self, v: i64) {
        debug_assert_generation("nextWriteDelGen", v);
        self.next_write_del_gen = v;
    }

    /// `SegmentCommitInfo.setNextWriteFieldInfosGen(long)`.
    pub fn set_next_write_field_infos_gen(&mut self, v: i64) {
        debug_assert_generation("nextWriteFieldInfosGen", v);
        self.next_write_field_infos_gen = v;
    }

    /// `SegmentCommitInfo.setNextWriteDocValuesGen(long)`.
    pub fn set_next_write_doc_values_gen(&mut self, v: i64) {
        debug_assert_generation("nextWriteDocValuesGen", v);
        self.next_write_doc_values_gen = v;
    }

    /// `SegmentCommitInfo.advanceDelGen()`: take the next-write generation as
    /// the current one and step the next-write generation past it.
    // ARITH: three independent gates keep `del_gen` at or below
    // `MAX_GENERATION` (`i64::MAX / 2`), which leaves this `+ 1` 2^62 of
    // headroom: `parse` rejects a larger one off disk, `usable_generation`
    // rejects one carried in by a file *name*, and `check_writable_generations`
    // refuses to serialize one. Note what is deliberately *not* claimed: the
    // fields are `pub` on a `pub` type, so an in-process caller can still set
    // one directly -- `debug_assert_generation` on the three setters is the
    // enforcer for that half, and the write gate is what stops a violated
    // caller contract from reaching a file. Every legitimate step is `+ 1`
    // after a real file has been written, so reaching `i64::MAX` from the cap
    // would take 2^62 further `.liv` writes.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_del_gen(&mut self) {
        self.del_gen = self.next_write_del_gen();
        self.next_write_del_gen = self.del_gen + 1;
        self.generation_advanced();
    }

    /// `SegmentCommitInfo.advanceDocValuesGen()`.
    // ARITH: as `advance_del_gen`, for the `.dvd`/`.dvm` generation.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_doc_values_gen(&mut self) {
        self.doc_values_gen = self.next_write_doc_values_gen();
        self.next_write_doc_values_gen = self.doc_values_gen + 1;
        self.generation_advanced();
    }

    /// `SegmentCommitInfo.advanceFieldInfosGen()`: a doc-values update round
    /// changes `FieldInfo.docValuesGen` for the fields it touched, so the
    /// segment's `FieldInfos` are rewritten at a new generation alongside the
    /// doc-values files themselves (`ReadersAndUpdates.writeFieldInfosGen`).
    // ARITH: as `advance_del_gen`, for the generational `.fnm`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_field_infos_gen(&mut self) {
        self.field_infos_gen = self.next_write_field_infos_gen();
        self.next_write_field_infos_gen = self.field_infos_gen + 1;
        self.generation_advanced();
    }

    /// `SegmentCommitInfo.advanceNextWriteFieldInfosGen()`: step *only* the
    /// next-write counter, leaving `field_infos_gen` where it is. Java calls
    /// this (with its doc-values twin) from `writeFieldUpdates`' failure path,
    /// so a second attempt writes to a name the failed one cannot have left a
    /// partial file under.
    // ARITH: as `advance_del_gen` -- one step per failed write attempt, from a
    // value `parse` capped at `MAX_GENERATION`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_next_write_field_infos_gen(&mut self) {
        self.next_write_field_infos_gen = self.next_write_field_infos_gen() + 1;
    }

    /// `SegmentCommitInfo.advanceNextWriteDocValuesGen()`, the twin of
    /// [`Self::advance_next_write_field_infos_gen`].
    // ARITH: as `advance_next_write_field_infos_gen`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_next_write_doc_values_gen(&mut self) {
        self.next_write_doc_values_gen = self.next_write_doc_values_gen() + 1;
    }

    /// `SegmentCommitInfo.setDocValuesUpdatesFiles(Map)`: install `field`'s
    /// doc-values-update files, **replacing** whatever generation was recorded
    /// for it before.
    ///
    /// Replacement, not accumulation, is the whole point of the format: a
    /// generation is the field's *complete* rewritten column, so the previous
    /// generation's files are dead the moment this one lands. Java gets the
    /// same result from `newDVFiles.put(fieldInfo.number, ...)` overwriting the
    /// carried-over entry in `writeFieldUpdates`. Accumulating instead would
    /// keep every superseded generation referenced forever -- unreadable by a
    /// reader that resolves only `FieldInfo.docValuesGen`, and never reclaimed
    /// by [`crate::index_file_deleter`].
    pub fn set_doc_values_updates_files(&mut self, field_number: i32, files: Vec<String>) {
        match self
            .dv_update_files
            .iter_mut()
            .find(|(n, _)| *n == field_number)
        {
            Some(slot) => slot.1 = files,
            None => self.dv_update_files.push((field_number, files)),
        }
    }

    /// `SegmentCommitInfo.setBufferedDeletesGen(long)`. Java only sets it
    /// while it is still `-1` (a segment is published exactly once), and this
    /// keeps that guard so a re-published segment cannot silently move out of
    /// the delete packets that target it.
    pub fn set_buffered_deletes_gen(&mut self, v: i64) {
        // Java throws `IllegalStateException("buffered deletes gen should only
        // be set once")` here. Silently ignoring a second call is the safer
        // production behaviour -- moving a published segment's generation
        // would change which delete packets reach it -- but the second call is
        // still a caller bug, so make it visible where bugs are cheap to see.
        debug_assert_eq!(
            self.buffered_deletes_gen, -1,
            "buffered deletes gen should only be set once (segment {:?})",
            self.segment_name
        );
        if self.buffered_deletes_gen == -1 {
            self.buffered_deletes_gen = v;
            self.generation_advanced();
        }
    }

    /// `SegmentCommitInfo.generationAdvanced()`: every generation-advancing
    /// mutator ends here, and its whole job is to give the segment-commit a
    /// **new id**. Java draws a fresh `StringHelper.randomId()`; the field is
    /// documented as "an Id that uniquely identifies this segment commit ...
    /// This ID changes each time the segment changes due to a delete,
    /// doc-value or field update", i.e. it is a *change token*, not a name.
    ///
    /// Nothing in Lucene validates it on read (`SegmentInfos.readCommit`
    /// accepts any 16 bytes, and `.si` ids are checked against
    /// `SegmentInfo.id`, never against this one), so leaving it frozen breaks
    /// no read. What it breaks is every consumer that compares two commits'
    /// ids to decide whether a segment changed -- `DirectoryReader.openIfChanged`'s
    /// per-segment reuse, an NRT/replication client's "do I need to fetch this
    /// segment again", and any cache keyed on it -- all of which would be told
    /// "unchanged" across a delete or a doc-values update.
    ///
    /// Derived rather than random, for the same reason
    /// [`crate::segment_writer::derive_sci_id`] is: this workspace carries no
    /// CSPRNG dependency, and the only property a consumer reads the id for is
    /// **"different whenever the segment-commit is different"**. Hashing the
    /// segment's own (unique) id together with all four generations gives
    /// that: the generations only ever increase, so no two states of one
    /// segment share an *input*, and two different segments differ in the
    /// segment id. Distinct inputs give distinct *outputs* only with the
    /// probability a 128-bit digest affords, which is the same bound Java's
    /// 128 random bits give and far beyond what a change token needs. The
    /// hasher is `DefaultHasher`, whose algorithm is explicitly unspecified
    /// across Rust releases -- irrelevant here, because nothing compares an id
    /// against a value computed by a different build: they are compared
    /// against ids read back out of `segments_N`.
    ///
    /// The converse ("same state, same id") deliberately is **not** claimed
    /// across sessions: `buffered_deletes_gen` is not serialized (neither here
    /// nor in Java), so a segment that reaches `delGen = 1` in a fresh writer
    /// session gets a different token than the same segment did in the session
    /// that first deleted from it. That is the safe direction -- a consumer is
    /// told "changed" and re-fetches -- and it is strictly better than Java,
    /// whose random draw makes *every* re-derivation look changed.
    ///
    /// This is the second id derivation in the crate, and the one that wins:
    /// [`crate::segment_writer::derive_sci_id`] gives a freshly flushed
    /// segment its first id, and `IndexWriter`'s publish step immediately
    /// calls `set_buffered_deletes_gen` on it, so every segment an
    /// `IndexWriter` flushes reaches its commit carrying *this* value.
    /// `derive_sci_id` still earns its place for a caller using
    /// `segment_writer`/`merge` directly, and Java has the same two-step
    /// (`StringHelper.randomId()` in the constructor, re-drawn by
    /// `setBufferedDeletesGen`).
    ///
    /// Java also resets its cached `sizeInBytes` here. This port has no such
    /// cache (`merge_policy::commit_byte_size` sizes a segment from the
    /// directory on demand), so there is nothing to invalidate.
    fn generation_advanced(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut id = [0u8; ID_LENGTH];
        for (half, chunk) in id.chunks_mut(8).enumerate() {
            let mut hasher = DefaultHasher::new();
            (
                self.segment_id,
                self.del_gen,
                self.field_infos_gen,
                self.doc_values_gen,
                self.buffered_deletes_gen,
                half as u8,
            )
                .hash(&mut hasher);
            chunk.copy_from_slice(&hasher.finish().to_le_bytes());
        }
        self.sci_id = Some(id);
    }
}

/// Java's `SegmentCommitInfo` constructor lines 113-117: the next generation
/// of a generational file group starts one past whatever the commit records,
/// `1` when the commit records none. `0` here is this port's "no explicit
/// value has been set" sentinel (see
/// [`SegmentCommitInfo::next_write_del_gen`]) — `0` is never a legal Lucene
/// generation, which is what makes it usable as one.
// ARITH: `current_gen` is one of the three generations bounded to
// `-1..=MAX_GENERATION` (`i64::MAX / 2`) on the way in from disk (`parse`),
// on the way in from a file name (`usable_generation`) and on the way out
// (`check_writable_generations`), so `+ 1` has 2^62 of headroom. See
// `advance_del_gen` for the one hole that leaves -- a `pub` field set
// in-process -- and what enforces it.
#[allow(clippy::arithmetic_side_effects)]
fn derive_next_gen(explicit: i64, current_gen: i64) -> i64 {
    if explicit != 0 {
        explicit
    } else if current_gen == -1 {
        1
    } else {
        current_gen + 1
    }
}

impl SegmentCommitInfo {
    /// Port of `SegmentCommitInfo.files()`: the segment's own `.si`-declared
    /// files (passed in as `si_files`, since this type deliberately does not
    /// own the parsed `.si`) **plus** the three groups only the commit knows
    /// about -- the current-generation `.liv` file when `del_gen != -1`
    /// (Java: `liveDocsFormat().files(this, files)`), every
    /// `field_infos_files` entry, and every per-field doc-values update file.
    ///
    /// Anything that walks "every file this segment owns" -- reference
    /// counting, checksum verification, a corruption check -- must use this,
    /// not `SegmentInfo.files` alone: a `.liv` or a generational `.fnm`/`.dvd`
    /// is never listed in the `.si` (it did not exist when the `.si` was
    /// written), so a tool that only reads `.si` silently skips exactly the
    /// files a delete/update round produced.
    ///
    /// Order is deterministic (`.si` files first, then `.liv`, then field
    /// infos, then doc-values updates in field-number order); duplicates are
    /// removed, matching Java's `HashSet` semantics without its arbitrary
    /// iteration order.
    pub fn files(&self, si_files: &[String]) -> Vec<String> {
        // ARITH: `si_files` is an in-memory slice, so its length is at most
        // `isize::MAX` and `+ 4` (one `.liv` plus a little slack for the
        // generational groups) cannot overflow `usize`.
        #[allow(clippy::arithmetic_side_effects)]
        let capacity = si_files.len() + 4;
        let mut files: Vec<String> = Vec::with_capacity(capacity);
        let push = |name: String, files: &mut Vec<String>| {
            if !files.contains(&name) {
                files.push(name);
            }
        };
        for f in si_files {
            push(f.clone(), &mut files);
        }
        if self.del_gen != -1 {
            push(
                crate::deletes::liv_file_name(&self.segment_name, self.del_gen),
                &mut files,
            );
        }
        for f in &self.field_infos_files {
            push(f.clone(), &mut files);
        }
        for (_, dv_files) in &self.dv_update_files {
            for f in dv_files {
                push(f.clone(), &mut files);
            }
        }
        files
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentInfos {
    pub id: [u8; ID_LENGTH],
    pub generation: i64,
    pub format_version: i32,
    pub lucene_version: LuceneVersion,
    pub index_created_version_major: i32,
    /// Commit's own monotonic version counter (`SegmentInfos.version`).
    pub version: i64,
    /// Next unused segment-name counter (`SegmentInfos.counter`).
    pub counter: i64,
    pub min_segment_lucene_version: Option<LuceneVersion>,
    pub segments: Vec<SegmentCommitInfo>,
    pub user_data: Vec<(String, String)>,
}

/// Parses a whole `segments_N` file already read into memory.
///
/// `generation` is the `N` from the filename (or the special generation for
/// `segments.gen`-less setups) — Lucene encodes it as a base-36 string in the
/// index header's suffix and we must match it exactly, just like the codec name
/// and id.
pub fn parse(buf: &[u8], generation: i64) -> Result<SegmentInfos> {
    let mut input = SliceInput::new(buf);

    // `generation` is the `N` parsed out of the `segments_N` *file name*, in
    // base 36 -- so a directory entry named `segments_1y2p0ij32e8e7` hands
    // this a perfectly well-formed `i64::MAX`, and every later commit's
    // `generation + 1` would then panic. Bound it with the rest.
    let generation = check_generation("generation", generation, "")?;

    let suffix = lucene_util::base36::to_base36(generation);
    // We don't yet know `id` (it's inside the file), so check the header without
    // the id/suffix-bound convenience wrapper and validate the suffix by hand —
    // mirrors Java's `checkHeaderNoMagic` + manual `checkIndexHeaderSuffix` split.
    let header = codec_util::check_header(&mut input, CODEC_NAME, VERSION_74, VERSION_CURRENT)?;
    let mut id = [0u8; ID_LENGTH];
    input.read_bytes(&mut id)?;
    codec_util::check_index_header_suffix(&mut input, &suffix)?;

    let lucene_version = read_vint_version(&mut input)?;
    let index_created_version_major = input.read_vint()?;
    // `SegmentInfos.readCommit`: a commit can't claim it was created by a
    // *newer* major than the version that wrote it. Without this a corrupt
    // (or forward-dated) commit silently skews every later
    // `indexCreatedVersionMajor >= 7` gate.
    if lucene_version.major < index_created_version_major {
        return Err(Error::CreatedVersionAheadOfWriter {
            created: index_created_version_major,
            major: lucene_version.major,
            minor: lucene_version.minor,
            bugfix: lucene_version.bugfix,
        });
    }

    // `version` and `counter` are both stepped by `+ 1` on every commit
    // (`update_document`, `index_writer`), so they get the same cap the
    // per-segment generations get -- see `MAX_GENERATION`.
    let version = check_generation("version", input.read_be_u64()? as i64, "")?;
    let counter = check_generation("counter", input.read_vlong()?, "")?;
    let num_segments = input.read_be_i32()?;
    if num_segments < 0 {
        return Err(Error::InvalidSegmentCount(num_segments));
    }
    // A `SegmentCommitInfo` costs well over one byte on the wire (a name, a
    // 16-byte id, a codec name, three 8-byte generations, two counts, two
    // string sets), so a count above the bytes still in the file is corrupt by
    // construction. Checking it *before* reserving is the point: a
    // `SegmentCommitInfo` is ~150 bytes of `String`s and `Vec`s, so an
    // unbounded `i32` count reserves up to 300 GB and **aborts** the process --
    // and an abort is not something `catch_unwind` can keep out of the JVM.
    if num_segments as usize > input.remaining() {
        return Err(Error::InvalidSegmentCount(num_segments));
    }

    let min_segment_lucene_version = if num_segments > 0 {
        Some(read_vint_version(&mut input)?)
    } else {
        None
    };

    let mut segments = Vec::with_capacity(num_segments as usize);
    for _ in 0..num_segments {
        let segment_name = input.read_string()?;
        let mut segment_id = [0u8; ID_LENGTH];
        input.read_bytes(&mut segment_id)?;
        let codec_name = input.read_string()?;

        let del_gen = check_generation("delGen", input.read_be_u64()? as i64, &segment_name)?;
        let del_count = input.read_be_i32()?;
        if del_count < 0 {
            return Err(Error::InvalidDeletionCount(del_count, segment_name));
        }
        let field_infos_gen =
            check_generation("fieldInfosGen", input.read_be_u64()? as i64, &segment_name)?;
        let doc_values_gen =
            check_generation("docValuesGen", input.read_be_u64()? as i64, &segment_name)?;
        let soft_del_count = input.read_be_i32()?;
        if soft_del_count < 0 {
            return Err(Error::InvalidDeletionCount(soft_del_count, segment_name));
        }

        let sci_id = if header.version > VERSION_74 {
            match input.read_byte()? {
                0 => None,
                1 => {
                    let mut sci = [0u8; ID_LENGTH];
                    input.read_bytes(&mut sci)?;
                    Some(sci)
                }
                other => return Err(Error::InvalidSciIdMarker(other)),
            }
        } else {
            None
        };

        let field_infos_files = input.read_set_of_strings()?;
        let num_dv_fields = input.read_be_i32()?;
        // Same shape as `num_segments`: each entry is a 4-byte field number
        // plus a string set, so a count past the remaining bytes is corrupt --
        // and reserving for an unbounded one aborts. Java sizes a `HashMap`
        // from this value with the same lack of a bound.
        if num_dv_fields < 0 || num_dv_fields as usize > input.remaining() {
            return Err(Error::InvalidDocValuesFieldCount(
                num_dv_fields,
                segment_name,
            ));
        }
        let mut dv_update_files = Vec::with_capacity(num_dv_fields as usize);
        for _ in 0..num_dv_fields {
            let field_number = input.read_be_i32()?;
            let files = input.read_set_of_strings()?;
            dv_update_files.push((field_number, files));
        }
        let mut dv_overlays = Vec::new();
        if header.version >= VERSION_10_6 {
            let num_overlay_fields = input.read_be_i32()?;
            if num_overlay_fields < 0 || num_overlay_fields as usize > input.remaining() {
                return Err(Error::InvalidDocValuesFieldCount(
                    num_overlay_fields,
                    segment_name,
                ));
            }
            for _ in 0..num_overlay_fields {
                let field_number = input.read_be_i32()?;
                let base_gen = check_generation("overlay baseGen", input.read_be_u64()? as i64, &segment_name)?;
                let num_deltas = input.read_be_i32()?;
                if num_deltas < 0 || num_deltas as usize > input.remaining() {
                    return Err(Error::InvalidDocValuesFieldCount(num_deltas, segment_name));
                }
                let mut deltas = Vec::with_capacity(num_deltas as usize);
                for _ in 0..num_deltas {
                    deltas.push(check_generation(
                        "overlay deltaGen",
                        input.read_be_u64()? as i64,
                        &segment_name,
                    )?);
                }
                dv_overlays.push((field_number, base_gen, deltas));
            }
        }

        segments.push(SegmentCommitInfo {
            segment_name,
            segment_id,
            codec_name,
            del_gen,
            del_count,
            field_infos_gen,
            doc_values_gen,
            soft_del_count,
            sci_id,
            field_infos_files,
            dv_update_files,
            dv_overlays,
            // Java's `SegmentCommitInfo` constructor derives all three from
            // the generations it just read; the `0` sentinel does that
            // lazily (see `SegmentCommitInfo::next_write_del_gen`).
            ..Default::default()
        });
    }

    let user_data = input.read_map_of_strings()?;

    codec_util::check_footer(&mut input, buf.len())?;

    Ok(SegmentInfos {
        id,
        generation,
        format_version: header.version,
        lucene_version,
        index_created_version_major,
        version,
        counter,
        min_segment_lucene_version,
        segments,
        user_data,
    })
}

/// Port of `SegmentInfos.FindSegmentsFile` + `SegmentInfos.read(Directory)`:
/// locates the highest-generation `segments_N` (or plain `segments`) file in
/// `dir` via `lucene_store::directory::read_latest_commit` (already-existing
/// listing/generation-picking logic, not reimplemented here) and parses it.
/// This is the entry point a `DirectoryReader.open(Directory)`-equivalent
/// needs first, before it can open any segment the commit lists.
pub fn read_latest(dir: &dyn Directory) -> Result<SegmentInfos> {
    let (generation, bytes) = lucene_store::directory::read_latest_commit(dir)?;
    parse(&bytes, generation)
}

/// Port of `SegmentInfos.write(Directory)`: the exact byte-level inverse of
/// [`parse`], plus the durability half of a real commit (`Directory.sync`
/// before the file is considered "there").
///
/// Design choice: unlike [`crate::segment_info::write`] and
/// `lucene_codecs::field_infos::write` (which return `Vec<u8>` and let the
/// caller route bytes through a `Directory` itself), this function takes a
/// `&dyn Directory` and writes+syncs the `segments_N` file directly. A
/// `segments_N` commit isn't just a byte format — its correctness as a
/// *commit* depends on being fsynced before anything can be considered
/// durably published (real `IndexWriter.commit()` calls `Directory.sync` on
/// this exact file right after writing it, before deleting the previous
/// generation). Returning bytes and leaving sync to the caller would make it
/// easy for a caller to "write" a commit that a crash could still lose;
/// baking the sync into `write` mirrors Java's own
/// `SegmentInfos.write`/`finishCommit` split, which never lets a caller skip
/// it.
///
/// `format_version` is not read from `segment_infos` -- this always writes
/// [`VERSION_CURRENT`], matching [`crate::segment_info::write`]'s stance that
/// this port only ever writes fresh segments, never round-trips an older
/// format version. The file name is derived from `segment_infos.generation`
/// via [`lucene_store::directory::segments_file_name`] (reused, not
/// reimplemented) so the base-36 suffix in the index header and the file's
/// own name can never drift apart.
///
/// Like Java's `SegmentInfos.commit`, this syncs only the `segments_N` and
/// the directory, not the segment files the commit names: a caller that
/// writes its own commit syncs those first (`IndexWriter` does, through
/// [`crate::index_file_deleter::IndexFileDeleter::sync_commit_files`]).
///
/// Returns the written file's name (`segments_N`) on success.
pub fn write(segment_infos: &SegmentInfos, dir: &dyn Directory) -> Result<String> {
    write_pending(segment_infos, dir)?;
    finish_pending(segment_infos, dir)
}

/// Phase one of `SegmentInfos.prepareCommit(Directory)`: serialize the whole
/// commit and write it to `pending_segments_N`, then fsync it.
///
/// The pending name is deliberately not a name
/// [`lucene_store::directory::last_commit_generation`] scans for, so a crash
/// anywhere inside this function leaves the *previous* `segments_N` as the
/// current commit and the half-written pending file as an inert orphan.
/// That is the entire reason Java never creates a `segments_N` by writing to
/// it directly.
///
/// Returns the written `pending_segments_N` file name.
pub fn write_pending(segment_infos: &SegmentInfos, dir: &dyn Directory) -> Result<String> {
    // Every generation this commit is about to serialize must be one `parse`
    // will accept back, or the write produces an index this port cannot open
    // -- see `check_writable_generations`.
    check_writable_generations(segment_infos)?;
    // `pending_segments_file_name` and `segments_file_name` refuse exactly the
    // same generations (both only reject a negative one), so validating here
    // also guarantees `finish_pending` can name the file it has to rename to.
    let pending_name = pending_segments_name(segment_infos)?;

    // Java syncs the directory's metadata before creating the pending file
    // (`SegmentInfos.prepareCommit` -> `dir.syncMetaData()`), so that every
    // file name the segments file is about to reference is itself durable.
    dir.sync_meta_data()?;

    let bytes = to_bytes(segment_infos);

    let mut output = dir.create_output(&pending_name)?;
    output.write_bytes(&bytes);
    // Java deletes a truncated pending file rather than leaving it behind;
    // errors from that cleanup are suppressed in favour of the original.
    if let Err(e) = output.close() {
        let _ = dir.delete_file(&pending_name);
        return Err(e.into());
    }
    if let Err(e) = dir.sync(std::slice::from_ref(&pending_name)) {
        let _ = dir.delete_file(&pending_name);
        return Err(e.into());
    }

    Ok(pending_name)
}

/// Phase two of the commit, `SegmentInfos.finishCommit(Directory)`: rename
/// the already-synced `pending_segments_N` written by [`write_pending`] onto
/// its final `segments_N` name and fsync the directory, which is the single
/// instant the new generation becomes visible to
/// [`lucene_store::directory::read_latest_commit`].
///
/// Returns the committed `segments_N` file name.
pub fn finish_pending(segment_infos: &SegmentInfos, dir: &dyn Directory) -> Result<String> {
    let pending_name = pending_segments_name(segment_infos)?;
    let file_name = segments_file_name(segment_infos)?;

    dir.rename(&pending_name, &file_name)?;
    if let Err(e) = dir.sync_meta_data() {
        // The rename landed but the directory entry is not durable; Java
        // deletes the renamed file rather than leave a commit that might
        // vanish under a crash. This port's writer keeps the prepared commit
        // for a retry, which renames `pending_segments_N` again, so the file
        // goes back under that name; only if that fails is it deleted.
        if dir.rename(&file_name, &pending_name).is_err() {
            let _ = dir.delete_file(&file_name);
        }
        return Err(e.into());
    }

    Ok(file_name)
}

/// `SegmentInfos.rollbackCommit(Directory)`: drop the `pending_segments_N`
/// [`write_pending`] left behind, ignoring any failure exactly as Java's
/// `IOUtils.deleteFilesIgnoringExceptions` does.
pub fn rollback_pending(segment_infos: &SegmentInfos, dir: &dyn Directory) {
    if let Some(pending_name) =
        lucene_store::directory::pending_segments_file_name(segment_infos.generation)
    {
        let _ = dir.delete_file(&pending_name);
    }
}

fn pending_segments_name(segment_infos: &SegmentInfos) -> Result<String> {
    lucene_store::directory::pending_segments_file_name(segment_infos.generation).ok_or_else(|| {
        Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid generation for a segments_N file name: {}",
            segment_infos.generation
        )))
    })
}

fn segments_file_name(segment_infos: &SegmentInfos) -> Result<String> {
    lucene_store::directory::segments_file_name(segment_infos.generation).ok_or_else(|| {
        Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid generation for a segments_N file name: {}",
            segment_infos.generation
        )))
    })
}

/// `SegmentInfos.write(IndexOutput)`: the whole `segments_N` file as bytes,
/// header and checksum footer included, for a caller that ships it (the
/// replicator's `CopyState.infosBytes`) instead of writing it to a directory.
pub fn to_bytes(segment_infos: &SegmentInfos) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let suffix = lucene_util::base36::to_base36(segment_infos.generation);
    let version = write_version(segment_infos);
    codec_util::write_index_header(&mut out, CODEC_NAME, version, &segment_infos.id, &suffix);

    write_vint_version(&mut out, segment_infos.lucene_version);
    out.write_vint(segment_infos.index_created_version_major);

    out.write_be_u64(segment_infos.version as u64);
    out.write_vlong(segment_infos.counter);
    out.write_be_u32(segment_infos.segments.len() as u32);

    if !segment_infos.segments.is_empty() {
        let min_version = segment_infos
            .min_segment_lucene_version
            .unwrap_or(segment_infos.lucene_version);
        write_vint_version(&mut out, min_version);
    }

    for seg in &segment_infos.segments {
        out.write_string(&seg.segment_name);
        out.write_bytes(&seg.segment_id);
        out.write_string(&seg.codec_name);
        out.write_be_u64(seg.del_gen as u64);
        out.write_be_u32(seg.del_count as u32);
        out.write_be_u64(seg.field_infos_gen as u64);
        out.write_be_u64(seg.doc_values_gen as u64);
        out.write_be_u32(seg.soft_del_count as u32);

        // Every written version is above VERSION_74, so the SciId marker
        // is always present, matching `parse`'s expectation for this format.
        match seg.sci_id {
            Some(sci_id) => {
                out.write_byte(1);
                out.write_bytes(&sci_id);
            }
            None => out.write_byte(0),
        }

        out.write_set_of_strings(&seg.field_infos_files);
        out.write_be_u32(seg.dv_update_files.len() as u32);
        for (field_number, files) in &seg.dv_update_files {
            out.write_be_u32(*field_number as u32);
            out.write_vint(files.len() as i32);
            for f in files {
                out.write_string(f);
            }
        }
        if version >= VERSION_10_6 {
            out.write_be_u32(seg.dv_overlays.len() as u32);
            for (field_number, base_gen, deltas) in &seg.dv_overlays {
                out.write_be_u32(*field_number as u32);
                out.write_be_u64(*base_gen as u64);
                out.write_be_u32(deltas.len() as u32);
                for g in deltas {
                    out.write_be_u64(*g as u64);
                }
            }
        }
    }

    out.write_map_of_strings(&segment_infos.user_data);
    codec_util::write_footer(&mut out);
    out
}

/// The `segments_N` format a commit is written in: [`VERSION_10_6`] when
/// `format_version` asks for it (Lucene 10.6 and later read it; 10.5 cannot) or
/// a segment carries overlays, which only that format can record; otherwise
/// [`VERSION_86`], which every Lucene 10 reads.
fn write_version(segment_infos: &SegmentInfos) -> i32 {
    let overlays = segment_infos.segments.iter().any(|s| !s.dv_overlays.is_empty());
    if segment_infos.format_version >= VERSION_10_6 || overlays {
        VERSION_10_6
    } else {
        VERSION_86
    }
}

fn write_vint_version(out: &mut Vec<u8>, v: LuceneVersion) {
    out.write_vint(v.major);
    out.write_vint(v.minor);
    out.write_vint(v.bugfix);
}

/// `Version.fromBits`: every component is packed into one byte, so anything
/// outside `0..=255` is an `IllegalArgumentException` in Java. Mirrored here
/// so a corrupt vint can't produce a version later comparisons silently
/// trust.
fn check_version_component(which: &'static str, value: i32) -> Result<i32> {
    if !(0..=255).contains(&value) {
        return Err(Error::IllegalVersion { which, value });
    }
    Ok(value)
}

fn read_vint_version(input: &mut SliceInput) -> Result<LuceneVersion> {
    let major = check_version_component("major", input.read_vint()?)?;
    let minor = check_version_component("minor", input.read_vint()?)?;
    let bugfix = check_version_component("bugfix", input.read_vint()?)?;
    Ok(LuceneVersion {
        major,
        minor,
        bugfix,
    })
}

#[cfg(test)]
mod tests;
