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
const VERSION_CURRENT: i32 = VERSION_86;

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
    }

    /// `SegmentCommitInfo.advanceDocValuesGen()`.
    // ARITH: as `advance_del_gen`, for the `.dvd`/`.dvm` generation.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn advance_doc_values_gen(&mut self) {
        self.doc_values_gen = self.next_write_doc_values_gen();
        self.next_write_doc_values_gen = self.doc_values_gen + 1;
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
        }
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
        // vanish under a crash.
        let _ = dir.delete_file(&file_name);
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

fn to_bytes(segment_infos: &SegmentInfos) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let suffix = lucene_util::base36::to_base36(segment_infos.generation);
    codec_util::write_index_header(
        &mut out,
        CODEC_NAME,
        VERSION_CURRENT,
        &segment_infos.id,
        &suffix,
    );

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

        // Always emitting VERSION_CURRENT (> VERSION_74), so the SciId marker
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
    }

    out.write_map_of_strings(&segment_infos.user_data);
    codec_util::write_footer(&mut out);
    out
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
mod tests {
    use super::*;

    /// Test-only `segments_N` byte builder, independent of the real
    /// `IndexWriter`-generated fixture under `tests/segment_infos_fixtures.rs`:
    /// that exercises real bytes end-to-end, this covers error paths (negative
    /// counts, bad markers, multiple segments/DV fields) that a real writer
    /// would never produce.
    struct SegBuilder {
        name: String,
        id: [u8; ID_LENGTH],
        codec: String,
        del_gen: i64,
        del_count: i32,
        field_infos_gen: i64,
        doc_values_gen: i64,
        soft_del_count: i32,
        sci_marker: Option<u8>, // None => omit entirely (format <= VERSION_74)
        dv_fields: Vec<(i32, Vec<String>)>,
        num_dv_fields_override: Option<i32>,
    }

    impl SegBuilder {
        fn valid(name: &str) -> Self {
            Self {
                name: name.to_string(),
                id: [2u8; ID_LENGTH],
                codec: "Lucene104".to_string(),
                del_gen: -1,
                del_count: 0,
                field_infos_gen: -1,
                doc_values_gen: -1,
                soft_del_count: 0,
                sci_marker: Some(0),
                dv_fields: vec![],
                num_dv_fields_override: None,
            }
        }
    }

    struct SisBuilder {
        generation: i64,
        format_version: i32,
        id: [u8; ID_LENGTH],
        segments: Vec<SegBuilder>,
        num_segments_override: Option<i32>,
        commit_version: i64,
        counter: i64,
        user_data: Vec<(String, String)>,
        lucene_version_major: i32,
        lucene_version_minor: i32,
        index_created_version_major: i32,
    }

    impl SisBuilder {
        fn valid(generation: i64) -> Self {
            Self {
                generation,
                format_version: VERSION_86,
                id: [3u8; ID_LENGTH],
                segments: vec![],
                num_segments_override: None,
                commit_version: 1,
                counter: 1,
                user_data: vec![],
                lucene_version_major: 10,
                lucene_version_minor: 0,
                index_created_version_major: 10,
            }
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, CODEC_NAME);
            out.extend_from_slice(&(self.format_version as u32).to_be_bytes());
            out.extend_from_slice(&self.id);
            let suffix = lucene_util::base36::to_base36(self.generation);
            out.push(suffix.len() as u8);
            out.extend_from_slice(suffix.as_bytes());

            write_vint(&mut out, self.lucene_version_major);
            write_vint(&mut out, self.lucene_version_minor);
            write_vint(&mut out, 0); // bugfix
            write_vint(&mut out, self.index_created_version_major);

            out.extend_from_slice(&(self.commit_version as u64).to_be_bytes());
            write_vlong(&mut out, self.counter);

            let num_segments = self
                .num_segments_override
                .unwrap_or(self.segments.len() as i32);
            out.extend_from_slice(&(num_segments as u32).to_be_bytes());

            if num_segments > 0 {
                write_vint(&mut out, 10); // minSegmentLuceneVersion major
                write_vint(&mut out, 0);
                write_vint(&mut out, 0);
            }

            for seg in &self.segments {
                write_string(&mut out, &seg.name);
                out.extend_from_slice(&seg.id);
                write_string(&mut out, &seg.codec);
                out.extend_from_slice(&(seg.del_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.del_count as u32).to_be_bytes());
                out.extend_from_slice(&(seg.field_infos_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.doc_values_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.soft_del_count as u32).to_be_bytes());
                if self.format_version > VERSION_74 {
                    if let Some(marker) = seg.sci_marker {
                        out.push(marker);
                        if marker == 1 {
                            out.extend_from_slice(&seg.id); // reuse id as a dummy sciId
                        }
                    }
                }
                write_vint(&mut out, 0); // fieldInfosFiles: empty set
                let num_dv_fields = seg
                    .num_dv_fields_override
                    .unwrap_or(seg.dv_fields.len() as i32);
                out.extend_from_slice(&(num_dv_fields as u32).to_be_bytes());
                for (field_number, files) in &seg.dv_fields {
                    out.extend_from_slice(&(*field_number as u32).to_be_bytes());
                    write_vint(&mut out, files.len() as i32);
                    for f in files {
                        write_string(&mut out, f);
                    }
                }
            }

            write_vint(&mut out, self.user_data.len() as i32);
            for (k, v) in &self.user_data {
                write_string(&mut out, k);
                write_string(&mut out, v);
            }

            out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            let checksum = crc32fast::hash(&out) as u64;
            out.extend_from_slice(&checksum.to_be_bytes());
            out
        }
    }

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn empty_commit_no_segments() {
        let b = SisBuilder::valid(1);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments.len(), 0);
        assert!(sis.min_segment_lucene_version.is_none());
    }

    #[test]
    fn single_segment_no_sci_id_and_no_dv_fields() {
        let mut b = SisBuilder::valid(2);
        b.segments.push(SegBuilder::valid("_0"));
        let sis = parse(&b.build(), 2).unwrap();
        assert_eq!(sis.segments.len(), 1);
        assert!(sis.segments[0].sci_id.is_none());
        assert!(sis.min_segment_lucene_version.is_some());
    }

    #[test]
    fn segment_with_sci_id_present() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = Some(1);
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments[0].sci_id, Some([2u8; ID_LENGTH]));
    }

    #[test]
    fn format_at_version_74_omits_sci_marker_entirely() {
        let mut b = SisBuilder::valid(1);
        b.format_version = VERSION_74;
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = None; // omitted at this format version, per real Lucene
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert!(sis.segments[0].sci_id.is_none());
    }

    #[test]
    fn doc_values_update_fields_are_parsed() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.dv_fields = vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ];
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments[0].dv_update_files, seg_dv_fields());
    }

    fn seg_dv_fields() -> Vec<(i32, Vec<String>)> {
        vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ]
    }

    #[test]
    fn multiple_segments_and_user_data() {
        let mut b = SisBuilder::valid(1);
        b.segments.push(SegBuilder::valid("_0"));
        b.segments.push(SegBuilder::valid("_1"));
        b.user_data.push(("k".to_string(), "v".to_string()));
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments.len(), 2);
        assert_eq!(sis.user_data, vec![("k".to_string(), "v".to_string())]);
    }

    #[test]
    fn negative_num_segments_rejected() {
        let mut b = SisBuilder::valid(1);
        b.num_segments_override = Some(-1);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidSegmentCount(-1))
        ));
    }

    /// `SegmentInfos.readCommit`: a commit that claims to have been created
    /// by a newer major than the one that wrote it is corrupt -- Java throws,
    /// and so must we, since every later `indexCreatedVersionMajor` gate
    /// trusts this value.
    #[test]
    fn created_version_newer_than_writer_version_rejected() {
        let mut b = SisBuilder::valid(1);
        b.lucene_version_major = 9;
        b.index_created_version_major = 10;
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::CreatedVersionAheadOfWriter {
                created: 10,
                major: 9,
                ..
            })
        ));
    }

    /// Equal majors are legal (an index created by 10.x and written by 10.x).
    #[test]
    fn created_version_equal_to_writer_version_accepted() {
        let mut b = SisBuilder::valid(1);
        b.lucene_version_major = 10;
        b.index_created_version_major = 10;
        assert!(parse(&b.build(), 1).is_ok());
    }

    /// `Version.fromBits` packs each component into one byte.
    #[test]
    fn out_of_range_lucene_version_component_rejected() {
        let mut b = SisBuilder::valid(1);
        b.lucene_version_minor = 300;
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::IllegalVersion {
                which: "minor",
                value: 300
            })
        ));
    }

    #[test]
    fn negative_del_count_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.del_count = -1;
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
        ));
    }

    #[test]
    fn negative_soft_del_count_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.soft_del_count = -1;
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
        ));
    }

    #[test]
    fn invalid_sci_marker_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = Some(7); // neither 0 nor 1
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidSciIdMarker(7))
        ));
    }

    #[test]
    fn wrong_generation_suffix_rejected() {
        let b = SisBuilder::valid(1);
        assert!(matches!(parse(&b.build(), 2), Err(Error::Store(_))));
    }

    // --- write() round-trips through parse(), via a real on-disk Directory ---

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("segment-infos-write")
    }

    fn sample_sis(generation: i64) -> SegmentInfos {
        SegmentInfos {
            id: [5u8; ID_LENGTH],
            generation,
            format_version: VERSION_CURRENT,
            lucene_version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            index_created_version_major: 10,
            version: 7,
            counter: 3,
            min_segment_lucene_version: None,
            segments: vec![],
            user_data: vec![],
        }
    }

    fn sample_segment(name: &str) -> SegmentCommitInfo {
        SegmentCommitInfo {
            segment_name: name.to_string(),
            segment_id: [6u8; ID_LENGTH],
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

    #[test]
    fn write_empty_commit_round_trips() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let sis = sample_sis(1);

        let file_name = write(&sis, &dir).unwrap();
        assert_eq!(file_name, "segments_1");

        let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
        let parsed = parse(&bytes, 1).unwrap();
        assert_eq!(parsed.id, sis.id);
        assert_eq!(parsed.version, sis.version);
        assert_eq!(parsed.counter, sis.counter);
        assert_eq!(
            parsed.index_created_version_major,
            sis.index_created_version_major
        );
        assert!(parsed.segments.is_empty());
        assert!(parsed.min_segment_lucene_version.is_none());

        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn write_single_segment_round_trips() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let mut sis = sample_sis(2);
        sis.min_segment_lucene_version = Some(sis.lucene_version);
        sis.segments.push(sample_segment("_0"));

        let file_name = write(&sis, &dir).unwrap();
        assert_eq!(file_name, "segments_2");

        let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
        let parsed = parse(&bytes, 2).unwrap();
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].segment_name, "_0");
        assert_eq!(parsed.segments[0].segment_id, [6u8; ID_LENGTH]);
        assert_eq!(parsed.segments[0].codec_name, "Lucene104");
        assert!(parsed.segments[0].sci_id.is_none());
        assert_eq!(parsed.min_segment_lucene_version, Some(sis.lucene_version));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn write_multi_segment_with_sci_id_del_and_dv_updates_round_trips() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let mut sis = sample_sis(3);

        let mut seg0 = sample_segment("_0");
        seg0.del_gen = 1;
        seg0.del_count = 2;
        seg0.field_infos_gen = 4;
        seg0.doc_values_gen = 5;
        seg0.soft_del_count = 1;
        seg0.sci_id = Some([9u8; ID_LENGTH]);
        seg0.field_infos_files = vec!["_0_4.fnm".to_string()];
        seg0.dv_update_files = vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ];

        let seg1 = sample_segment("_1");

        sis.segments.push(seg0);
        sis.segments.push(seg1);
        sis.user_data.push(("k".to_string(), "v".to_string()));

        let file_name = write(&sis, &dir).unwrap();
        let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
        let parsed = parse(&bytes, 3).unwrap();

        assert_eq!(parsed.segments.len(), 2);
        let s0 = &parsed.segments[0];
        assert_eq!(s0.del_gen, 1);
        assert_eq!(s0.del_count, 2);
        assert_eq!(s0.field_infos_gen, 4);
        assert_eq!(s0.doc_values_gen, 5);
        assert_eq!(s0.soft_del_count, 1);
        assert_eq!(s0.sci_id, Some([9u8; ID_LENGTH]));
        assert_eq!(s0.field_infos_files, vec!["_0_4.fnm".to_string()]);
        assert_eq!(
            s0.dv_update_files,
            vec![
                (0, vec!["_0_1.dvd".to_string()]),
                (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
            ]
        );
        assert_eq!(parsed.user_data, vec![("k".to_string(), "v".to_string())]);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn write_uses_lucene_version_as_min_segment_version_when_unset() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let mut sis = sample_sis(1);
        sis.min_segment_lucene_version = None; // deliberately unset
        sis.segments.push(sample_segment("_0"));

        let file_name = write(&sis, &dir).unwrap();
        let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
        let parsed = parse(&bytes, 1).unwrap();
        assert_eq!(parsed.min_segment_lucene_version, Some(sis.lucene_version));

        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn write_generation_zero_uses_plain_segments_file_name() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let sis = sample_sis(0);

        let file_name = write(&sis, &dir).unwrap();
        assert_eq!(file_name, "segments");

        let bytes = std::fs::read(dir_path.join(&file_name)).unwrap();
        let parsed = parse(&bytes, 0).unwrap();
        assert_eq!(parsed.generation, 0);

        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn write_negative_generation_is_rejected() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        let sis = sample_sis(-1);
        assert!(matches!(write(&sis, &dir), Err(Error::Store(_))));
        assert!(matches!(write_pending(&sis, &dir), Err(Error::Store(_))));
        assert!(matches!(finish_pending(&sis, &dir), Err(Error::Store(_))));
        // `rollback_pending` is infallible by contract and must simply do
        // nothing for a generation that has no pending file name at all.
        rollback_pending(&sis, &dir);
        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// A `Directory` that delegates everything to a real `FsDirectory` but can
    /// be told to fail `sync` or `sync_meta_data`, so the two "clean up the
    /// file we just created" paths in `write_pending`/`finish_pending` can be
    /// exercised. Both mirror Java (`IOUtils.deleteFilesSuppressingExceptions`
    /// in `SegmentInfos.write`/`finishCommit`), and both are unreachable
    /// through `FsDirectory` alone -- its `sync_meta_data` is best-effort and
    /// never reports failure.
    struct FailingDir {
        inner: lucene_store::FsDirectory,
        fail_sync: bool,
        fail_sync_meta_data: bool,
    }

    impl Directory for FailingDir {
        fn list_all(&self) -> lucene_store::Result<Vec<String>> {
            self.inner.list_all()
        }
        fn open(&self, name: &str) -> lucene_store::Result<lucene_store::directory::Input> {
            self.inner.open(name)
        }
        fn create_output(
            &self,
            name: &str,
        ) -> lucene_store::Result<lucene_store::index_output::FsIndexOutput> {
            self.inner.create_output(name)
        }
        fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
            if self.fail_sync {
                return Err(lucene_store::Error::Corrupted("sync failed".to_string()));
            }
            self.inner.sync(names)
        }
        fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
            self.inner.rename(source, dest)
        }
        fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
            self.inner.delete_file(name)
        }
        fn sync_meta_data(&self) -> lucene_store::Result<()> {
            if self.fail_sync_meta_data {
                return Err(lucene_store::Error::Corrupted(
                    "syncMetaData failed".to_string(),
                ));
            }
            self.inner.sync_meta_data()
        }
    }

    #[test]
    fn a_pending_commit_file_that_cannot_be_synced_is_deleted_not_left_behind() {
        let dir_path = tempdir();
        let dir = FailingDir {
            inner: lucene_store::FsDirectory::open(&dir_path),
            fail_sync: true,
            fail_sync_meta_data: false,
        };
        let sis = sample_sis(3);
        assert!(write_pending(&sis, &dir).is_err());
        assert!(
            dir.list_all().unwrap().is_empty(),
            "a pending commit file that could not be fsynced must not survive: {:?}",
            dir.list_all().unwrap()
        );
        std::fs::remove_dir_all(&dir_path).ok();
    }

    #[test]
    fn a_renamed_commit_file_whose_directory_cannot_be_synced_is_deleted() {
        let dir_path = tempdir();
        let write_dir = FailingDir {
            inner: lucene_store::FsDirectory::open(&dir_path),
            fail_sync: false,
            fail_sync_meta_data: false,
        };
        let sis = sample_sis(3);
        write_pending(&sis, &write_dir).unwrap();

        let finish_dir = FailingDir {
            inner: lucene_store::FsDirectory::open(&dir_path),
            fail_sync: false,
            fail_sync_meta_data: true,
        };
        assert!(finish_pending(&sis, &finish_dir).is_err());
        let listed = finish_dir.list_all().unwrap();
        assert!(
            !listed.iter().any(|f| f == "segments_3"),
            "a commit whose directory entry is not durable must not stay visible: {listed:?}"
        );
        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// `write_pending` leaves the previous commit current: the pending file is
    /// not a name `read_latest` can find.
    #[test]
    fn write_pending_alone_does_not_publish_a_commit() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);
        write(&sample_sis(1), &dir).unwrap();
        write_pending(&sample_sis(2), &dir).unwrap();

        assert_eq!(read_latest(&dir).unwrap().generation, 1);
        assert!(dir
            .list_all()
            .unwrap()
            .iter()
            .any(|f| f == "pending_segments_2"));

        finish_pending(&sample_sis(2), &dir).unwrap();
        assert_eq!(read_latest(&dir).unwrap().generation, 2);

        // ...and `rollback_pending` on a generation with no pending file is a
        // no-op rather than an error.
        rollback_pending(&sample_sis(9), &dir);
        std::fs::remove_dir_all(&dir_path).ok();
    }

    // --- the arithmetic gate (c28) ---

    /// Eight flipped bytes in `delGen` are enough to make every *later* commit
    /// panic: `getNextDelGen()` is `delGen + 1`, which Java wraps and Rust
    /// does not. Before the cap, `parse` accepted `i64::MAX` here and
    /// `next_write_del_gen()` overflowed on the first call.
    #[test]
    fn absurd_generations_are_decode_errors_not_overflowing_next_gens() {
        for (label, mutate) in [
            (
                "delGen",
                Box::new(|s: &mut SegBuilder| s.del_gen = i64::MAX) as Box<dyn Fn(&mut SegBuilder)>,
            ),
            (
                "fieldInfosGen",
                Box::new(|s: &mut SegBuilder| s.field_infos_gen = MAX_GENERATION + 1),
            ),
            (
                "docValuesGen",
                Box::new(|s: &mut SegBuilder| s.doc_values_gen = i64::MIN),
            ),
            // Below `-1`: `IndexFileNames.fileNameFromGeneration` asserts
            // `gen > 0` before it emits a suffix, so a `-5` would name a
            // `_0_-5.liv` no Lucene can read.
            (
                "negative delGen",
                Box::new(|s: &mut SegBuilder| s.del_gen = -5),
            ),
        ] {
            let mut b = SisBuilder::valid(1);
            let mut seg = SegBuilder::valid("_0");
            mutate(&mut seg);
            b.segments.push(seg);
            assert!(
                matches!(parse(&b.build(), 1), Err(Error::InvalidGeneration { .. })),
                "{label} should be rejected"
            );
        }
    }

    /// A generation exactly at the cap still parses, and the `+ 1` derivation
    /// that follows it stays representable — the cap is not off by one.
    ///
    /// Note what this does *not* bless: `MAX_GENERATION + 1` is a legal
    /// in-memory next-write generation but **not** a legal thing to serialize,
    /// which is what `a_commit_this_port_writes_is_always_one_it_can_read_back`
    /// pins down.
    #[test]
    fn generation_at_the_cap_parses_and_still_derives_a_next_gen() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.del_gen = MAX_GENERATION;
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments[0].next_write_del_gen(), MAX_GENERATION + 1);
    }

    /// The round-trip property the cap has to satisfy: **anything this crate
    /// can write is something `parse` accepts back**. Without the write-side
    /// gate, a commit read at exactly `MAX_GENERATION` derives
    /// `MAX_GENERATION + 1`, serializes it, and produces a `segments_N` this
    /// port then refuses — an index it wrote and can no longer open. Refusing
    /// the *commit* is the honest failure, and it leaves the previous
    /// `segments_N` current.
    #[test]
    fn a_commit_this_port_writes_is_always_one_it_can_read_back() {
        let dir_path = tempdir();
        let dir = lucene_store::FsDirectory::open(&dir_path);

        let at_cap = |generation: i64| {
            let mut sis = sample_sis(generation);
            sis.min_segment_lucene_version = Some(sis.lucene_version);
            sis.segments.push(sample_segment("_0"));
            sis
        };

        // The boundary itself round-trips.
        let mut sis = at_cap(1);
        sis.version = MAX_GENERATION;
        sis.counter = MAX_GENERATION;
        sis.segments[0].del_gen = MAX_GENERATION;
        let name = write(&sis, &dir).unwrap();
        let bytes = std::fs::read(dir_path.join(&name)).unwrap();
        let back = parse(&bytes, 1).unwrap();
        assert_eq!(back.version, MAX_GENERATION);
        assert_eq!(back.counter, MAX_GENERATION);
        assert_eq!(back.segments[0].del_gen, MAX_GENERATION);

        // One past it is refused at write time, by every counter in turn,
        // rather than written and discovered unreadable on the next open.
        for (which, mutate) in [
            (
                "version",
                Box::new(|s: &mut SegmentInfos| s.version = MAX_GENERATION + 1)
                    as Box<dyn Fn(&mut SegmentInfos)>,
            ),
            (
                "counter",
                Box::new(|s: &mut SegmentInfos| s.counter = MAX_GENERATION + 1),
            ),
            (
                "delGen",
                Box::new(|s: &mut SegmentInfos| s.segments[0].del_gen = MAX_GENERATION + 1),
            ),
            (
                "docValuesGen",
                Box::new(|s: &mut SegmentInfos| s.segments[0].doc_values_gen = MAX_GENERATION + 1),
            ),
            (
                "fieldInfosGen",
                Box::new(|s: &mut SegmentInfos| s.segments[0].field_infos_gen = MAX_GENERATION + 1),
            ),
        ] {
            let mut sis = at_cap(2);
            mutate(&mut sis);
            let err = write(&sis, &dir).unwrap_err();
            assert!(
                matches!(&err, Error::InvalidGeneration { which: w, .. } if *w == which),
                "{which}: unexpected error {err}"
            );
            // Nothing was published, and no pending file was left behind.
            assert_eq!(read_latest(&dir).unwrap().generation, 1);
            assert!(
                !dir.list_all().unwrap().iter().any(|f| f == "segments_2"),
                "{which}: an unreadable commit was published"
            );
        }
        std::fs::remove_dir_all(&dir_path).ok();
    }

    /// `version` and `counter` are stepped by `+ 1` on every commit
    /// (`update_document`), so they carry the same cap.
    #[test]
    fn absurd_commit_version_and_counter_are_decode_errors() {
        let mut b = SisBuilder::valid(1);
        b.commit_version = i64::MAX;
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidGeneration {
                which: "version",
                ..
            })
        ));

        let mut b = SisBuilder::valid(1);
        b.counter = MAX_GENERATION + 1;
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidGeneration {
                which: "counter",
                ..
            })
        ));
    }

    /// The `N` in `segments_N` is base 36, so a directory entry named
    /// `segments_1y2p0ij32e8e7` hands `parse` a well-formed `i64::MAX` — and
    /// `update_document` then does `generation += 1` on it.
    #[test]
    fn absurd_file_name_generation_is_a_decode_error() {
        let b = SisBuilder::valid(i64::MAX);
        assert!(matches!(
            parse(&b.build(), i64::MAX),
            Err(Error::InvalidGeneration {
                which: "generation",
                ..
            })
        ));
    }

    /// `numSegments` sized a `Vec<SegmentCommitInfo>` — ~150 bytes of `String`s
    /// and `Vec`s each — straight off a 4-byte header field. `i32::MAX` of them
    /// is a ~300 GB reservation, and an allocation failure is an **abort**,
    /// which `catch_unwind` cannot keep out of the JVM. Only the reservation
    /// was unbounded: the loop itself would have hit EOF immediately.
    #[test]
    fn absurd_segment_count_errors_instead_of_reserving_for_it() {
        let mut b = SisBuilder::valid(1);
        b.num_segments_override = Some(i32::MAX);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidSegmentCount(i32::MAX))
        ));
    }

    /// The same shape one level down: `numDVFields` sizes a `Vec<(i32,
    /// Vec<String>)>`. Java sizes a `HashMap` from it with no bound either.
    #[test]
    fn absurd_doc_values_field_count_errors_instead_of_reserving_for_it() {
        for count in [i32::MAX, -1] {
            let mut b = SisBuilder::valid(1);
            let mut seg = SegBuilder::valid("_0");
            seg.num_dv_fields_override = Some(count);
            b.segments.push(seg);
            assert!(
                matches!(
                    parse(&b.build(), 1),
                    Err(Error::InvalidDocValuesFieldCount(..))
                ),
                "numDVFields={count}"
            );
        }
    }
}
