//! Port of `org.apache.lucene.store.Directory` / `FSDirectory` / `MMapDirectory`,
//! plus the generation-lookup logic from `org.apache.lucene.index.SegmentInfos`
//! (`getLastCommitGeneration`, `generationFromSegmentsFileName`) that depends only
//! on a file listing.
//!
//! Two backends, one trait:
//! - [`FsDirectory`]: `std::fs::read` — safe, no `unsafe`, always correct. Default.
//! - [`MmapDirectory`]: `memmap2` — zero-copy reads matching Lucene's own default
//!   (`MMapDirectory`) for real workloads. Contains this crate's only `unsafe`
//!   (documented on the call site): mapping a file is only sound if nothing else
//!   truncates/mutates it concurrently, same caveat Lucene's own Javadoc carries.
//!
//! Both return an [`Input`] — an owned-or-mapped byte buffer that `Deref`s to
//! `&[u8]`, so callers (codec_util, segment_info, segment_infos) are unchanged
//! regardless of backend.

use std::fs;
use std::io::Read;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::index_output::{self, FsIndexOutput};

/// The `segments` file-name prefix (`IndexFileNames.SEGMENTS`). Excludes the
/// pre-4.0 `segments.gen` pointer file, which is not a valid commit file name.
const SEGMENTS_PREFIX: &str = "segments";
const OLD_SEGMENTS_GEN: &str = "segments.gen";
/// `IndexFileNames.PENDING_SEGMENTS`: the name a `segments_N` is written
/// under before it is renamed into place. Deliberately *not* prefixed with
/// [`SEGMENTS_PREFIX`], so a half-written commit file can never be picked up
/// by [`last_commit_generation`] -- that invisibility is the whole point of
/// Java's two-phase `prepareCommit`/`finishCommit` protocol.
const PENDING_SEGMENTS_PREFIX: &str = "pending_segments";

/// A file's bytes, however the backend obtained them.
pub enum Input {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::fmt::Debug for Input {
    /// Length and provenance only. The alternative is dumping a
    /// half-gigabyte mapping into a panic message.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Input::Owned(_) => "Owned",
            Input::Mapped(_) => "Mapped",
        };
        write!(f, "Input::{kind}({} bytes)", self.len())
    }
}

impl Deref for Input {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Input::Owned(v) => v,
            Input::Mapped(m) => m,
        }
    }
}

/// The same bytes as [`Deref`], as an `AsRef` — the bound a type-erased
/// shared buffer needs.
///
/// `Deref` alone cannot be used behind `dyn`: `Arc<dyn Deref<Target = [u8]>>`
/// is legal but every consumer would have to name the associated type, and
/// `Arc<[u8]>` (the obvious alternative) cannot alias a mapping — it owns its
/// allocation, so handing an `Input` to one always costs a full copy. With
/// this impl an `Arc<Input>` coerces straight to
/// `Arc<dyn AsRef<[u8]> + Send + Sync>`, which is how
/// `lucene_codecs::blocktree::open_shared` takes a `.tim`/`.tip` mapping
/// without copying it (c12, ~199 µs on a 4.7 MB `.tim`).
impl AsRef<[u8]> for Input {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// Directory abstraction covering both Lucene's read path (`listAll`, `open`
/// a whole file's bytes) and the write-path primitives this crate now
/// supports: `createOutput` (a real on-disk [`FsIndexOutput`]), `sync` (the
/// fsync-before-durable contract), and `rename`/`deleteFile`/`syncMetaData`
/// (what `SegmentInfos.prepareCommit`/`finishCommit`/`rollbackCommit` need to
/// publish a commit atomically). Locking (`NativeFSLockFactory`) and file
/// reference-counting (`IndexFileDeleter`) are still deferred — see
/// `docs/parity.md`.
pub trait Directory {
    /// Port of `Directory.listAll()`: every file name in the directory, sorted.
    fn list_all(&self) -> Result<Vec<String>>;

    /// Reads a whole file's bytes.
    fn open(&self, name: &str) -> Result<Input>;

    /// Port of `Directory.createOutput(name, context)`: creates (truncating
    /// any existing file of the same name) a new file for sequential
    /// writing, backed by a real `std::fs::File`.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput>;

    /// Port of `Directory.sync(Collection<String>)`: fsyncs every named
    /// file's contents (and, best-effort, the directory entry) to disk.
    /// Callers must sync a new segment's files before referencing them from
    /// a commit file — that's Lucene's actual durability contract.
    fn sync(&self, names: &[String]) -> Result<()>;

    /// Port of `Directory.rename(source, dest)`: atomically makes `source`'s
    /// contents visible under `dest`. This is the operation Lucene's
    /// `SegmentInfos.finishCommit` relies on to publish a commit — the
    /// `pending_segments_N` file is fully written and fsynced first, then a
    /// single rename makes it the new `segments_N`, so no crash can ever
    /// expose a half-written commit file under a name a reader scans for.
    fn rename(&self, source: &str, dest: &str) -> Result<()>;

    /// Port of `Directory.deleteFile(name)`.
    fn delete_file(&self, name: &str) -> Result<()>;

    /// Port of `Directory.syncMetaData()`: fsyncs the directory itself, so a
    /// rename/create of a *name* (not just a file's contents) survives a
    /// crash. Lucene calls this on both sides of the commit rename.
    fn sync_meta_data(&self) -> Result<()>;
}

/// Safe, copying backend (`std::fs::read`). No `unsafe` anywhere in this crate
/// when used exclusively.
pub struct FsDirectory {
    root: PathBuf,
}

impl FsDirectory {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl Directory for FsDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        list_all(&self.root)
    }

    fn open(&self, name: &str) -> Result<Input> {
        Ok(Input::Owned(fs::read(self.root.join(name))?))
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        index_output::create_output(&self.root, name)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        index_output::rename(&self.root, source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        index_output::delete_file(&self.root, name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        index_output::sync_meta_data(&self.root)
    }
}

/// The size at or below which [`MmapDirectory`] reads a file instead of
/// mapping it.
///
/// Mapping is only cheaper than reading once the mapping is large enough to
/// amortise its own syscalls. Measured on this project's 5 M-document
/// benchmark corpus (`crates/lucene-search/examples/reader_open_profile.rs`),
/// a few-hundred-byte file costs **1.86 µs** to `open`+`mmap`+`munmap` and
/// **1.19 µs** to `open`+`read`, before the mapping's first page fault --
/// and every file this small in a segment (`.si`, `.fnm`, `.tmd`, `.dvm`,
/// `.nvm`, `.kdm`, `segments_N`) is parsed whole at open, so it takes that
/// fault immediately.
///
/// 16 KiB, not larger, because the files a reader *holds* and then accesses
/// randomly -- `.tip`, `.kdi` -- are the ones a copy would pessimise, and
/// they are above this on any index big enough for it to matter. The worst
/// case the threshold can cost is one 16 KiB `memcpy`.
///
/// Real Lucene has no equivalent: `MMapDirectory.openInput` maps
/// unconditionally. This is a Rust-side win, not a port divergence -- the
/// bytes a caller sees are identical, and [`Input`] already had both
/// representations because [`FsDirectory`] produces the owned one.
pub const SMALL_FILE_READ_THRESHOLD: u64 = 16 * 1024;

/// Zero-copy backend (`memmap2`), matching Lucene's default `MMapDirectory`.
pub struct MmapDirectory {
    root: PathBuf,
    /// Files of at most this many bytes are read rather than mapped -- see
    /// [`SMALL_FILE_READ_THRESHOLD`]. `0` maps everything, which is what
    /// this backend did before and what
    /// [`MmapDirectory::with_read_threshold`] exists to reproduce.
    read_threshold: u64,
}

impl MmapDirectory {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_threshold: SMALL_FILE_READ_THRESHOLD,
        }
    }

    /// [`Self::open`] with an explicit small-file threshold: files of at most
    /// `read_threshold` bytes are read into memory instead of mapped. `0`
    /// maps every file.
    ///
    /// Exposed because it is the only way to measure the two arms in **one
    /// process**, which is how this project measures anything (see
    /// `docs/sweep/m2/c24-arith-codecs.md` on why criterion is not trusted
    /// here) -- and because a caller with an unusual access pattern (a
    /// long-lived reader over many small files it seeks in repeatedly) can
    /// turn it off.
    pub fn with_read_threshold(root: impl Into<PathBuf>, read_threshold: u64) -> Self {
        Self {
            root: root.into(),
            read_threshold,
        }
    }
}

impl Directory for MmapDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        list_all(&self.root)
    }

    fn open(&self, name: &str) -> Result<Input> {
        let mut file = fs::File::open(self.root.join(name))?;
        // A small file is read, not mapped: see `SMALL_FILE_READ_THRESHOLD`.
        // The `metadata` call is one `fstat` on an already-open descriptor,
        // which is cheaper than the `munmap` it avoids.
        let len = file.metadata()?.len();
        if self.read_threshold > 0 && len <= self.read_threshold {
            // Read from the descriptor already open, not `fs::read(path)`:
            // that re-opens by name and re-`fstat`s for its own size hint, so
            // the "cheaper than a mapping" arm would have paid *two*
            // `open`+`fstat` pairs -- and would have reopened a path that can
            // have changed underneath it in between. `len` is at most
            // `read_threshold`, so the `usize` cast cannot truncate on any
            // target this crate builds for.
            let mut buf = Vec::with_capacity(len as usize);
            file.read_to_end(&mut buf)?;
            return Ok(Input::Owned(buf));
        }
        // SAFETY: mapping is only unsound if another process truncates or
        // mutates this file while it's mapped, which we do not do ourselves and
        // which Lucene's own `MMapDirectory` accepts the same risk for (see its
        // Javadoc). The directory is opened read-only and outlives no writer in
        // the read-only phase this crate currently implements (PLAN.md Phase 2).
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Input::Mapped(mmap))
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        index_output::create_output(&self.root, name)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        index_output::sync(&self.root, names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        index_output::rename(&self.root, source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        index_output::delete_file(&self.root, name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        index_output::sync_meta_data(&self.root)
    }
}

fn list_all(root: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(root)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    names.sort();
    Ok(names)
}

/// Port of `SegmentInfos.generationFromSegmentsFileName`.
pub fn generation_from_segments_file_name(file_name: &str) -> Result<i64> {
    if file_name == OLD_SEGMENTS_GEN {
        return Err(Error::Corrupted(format!(
            "\"{OLD_SEGMENTS_GEN}\" is not a valid segment file name since 4.0"
        )));
    }
    if file_name == SEGMENTS_PREFIX {
        return Ok(0);
    }
    if let Some(suffix) = file_name.strip_prefix(&format!("{SEGMENTS_PREFIX}_")) {
        return lucene_util::base36::from_base36(suffix).ok_or_else(|| {
            Error::Corrupted(format!("fileName \"{file_name}\" is not a segments file"))
        });
    }
    Err(Error::Corrupted(format!(
        "fileName \"{file_name}\" is not a segments file"
    )))
}

/// Port of `SegmentInfos.getLastCommitGeneration(String[])`: the highest
/// generation among `segments`/`segments_N` file names (excluding the legacy
/// `segments.gen` pointer), or -1 if none exist.
///
/// Strict, like Java: an unparsable `segments*` name aborts the scan rather
/// than being skipped. Skipping it is not a safe simplification in either
/// direction — a reader would silently open the *previous* commit and hide
/// every document committed since, and a writer would read -1 from a
/// directory that does have a commit and create a fresh index over it.
pub fn last_commit_generation(files: &[String]) -> Result<i64> {
    let mut generation = -1i64;
    for file in files {
        if is_segments_candidate(file) {
            generation = generation.max(generation_from_segments_file_name(file)?);
        }
    }
    Ok(generation)
}

/// The `startsWith(SEGMENTS) && startsWith(OLD_SEGMENTS_GEN) == false` guard
/// Java applies before calling `generationFromSegmentsFileName`. Note the
/// second test is a prefix test in Java, not equality: `segments.gen_1` is
/// skipped too, not treated as a corrupt generation.
fn is_segments_candidate(file_name: &str) -> bool {
    file_name.starts_with(SEGMENTS_PREFIX) && !file_name.starts_with(OLD_SEGMENTS_GEN)
}

/// Port of `IndexFileNames.fileNameFromGeneration("segments", "", gen)`.
pub fn segments_file_name(generation: i64) -> Option<String> {
    match generation {
        -1 => None,
        0 => Some(SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Port of `IndexFileNames.fileNameFromGeneration("pending_segments", "",
/// gen)`: the name a commit file is written under before
/// `SegmentInfos.finishCommit` renames it to [`segments_file_name`]'s name.
///
/// Same `gen == -1 -> null`, `gen == 0 -> bare base name` shape
/// [`segments_file_name`] has, since both go through the same Java helper.
/// Generation 0 is unreachable in practice (`getNextPendingGeneration()`
/// returns `1` for a never-committed index and `generation + 1` otherwise),
/// but the mapping is kept total and exact rather than special-cased away.
pub fn pending_segments_file_name(generation: i64) -> Option<String> {
    match generation {
        g if g < 0 => None,
        0 => Some(PENDING_SEGMENTS_PREFIX.to_string()),
        gen => Some(format!(
            "{PENDING_SEGMENTS_PREFIX}_{}",
            lucene_util::base36::to_base36(gen)
        )),
    }
}

/// Finds and reads the most recent `segments_N` commit file in `dir`.
/// Returns `(generation, bytes)`; callers pass both to `segment_infos::parse`.
pub fn read_latest_commit(dir: &(impl Directory + ?Sized)) -> Result<(i64, Input)> {
    let files = dir.list_all()?;
    let generation = last_commit_generation(&files)?;
    let name = segments_file_name(generation)
        .ok_or_else(|| Error::Corrupted("no segments_N commit file found".to_string()))?;
    let bytes = dir.open(&name)?;
    Ok((generation, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;

    #[test]
    fn generation_from_segments_file_name_valid_cases() {
        assert_eq!(generation_from_segments_file_name("segments").unwrap(), 0);
        assert_eq!(generation_from_segments_file_name("segments_1").unwrap(), 1);
        assert_eq!(generation_from_segments_file_name("segments_2").unwrap(), 2);
        // base-36: "segments_a" -> 10
        assert_eq!(
            generation_from_segments_file_name("segments_a").unwrap(),
            10
        );
    }

    #[test]
    fn generation_from_segments_file_name_rejects_old_pointer_file() {
        assert!(matches!(
            generation_from_segments_file_name("segments.gen"),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn generation_from_segments_file_name_rejects_garbage() {
        assert!(matches!(
            generation_from_segments_file_name("not-a-segments-file"),
            Err(Error::Corrupted(_))
        ));
        // Has the prefix but a non-base-36 suffix.
        assert!(matches!(
            generation_from_segments_file_name("segments_!!!"),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn last_commit_generation_ignores_old_pointer_and_non_segments_files() {
        let files = vec![
            "segments.gen".to_string(),
            "_0.si".to_string(),
            "segments_1".to_string(),
            "segments_3".to_string(),
            "segments_2".to_string(),
        ];
        assert_eq!(last_commit_generation(&files).unwrap(), 3);
    }

    #[test]
    fn segments_file_name_all_branches() {
        assert_eq!(segments_file_name(-1), None);
        assert_eq!(segments_file_name(0), Some("segments".to_string()));
        assert_eq!(segments_file_name(1), Some("segments_1".to_string()));
        assert_eq!(segments_file_name(10), Some("segments_a".to_string()));
    }

    /// A `Directory` that can only be listed: every test using it asserts
    /// that `read_latest_commit` fails during the *scan*, before any file is
    /// opened, so the remaining methods must never be called.
    struct ListingOnlyDir(Vec<String>);

    impl Directory for ListingOnlyDir {
        fn list_all(&self) -> Result<Vec<String>> {
            Ok(self.0.clone())
        }
        fn open(&self, name: &str) -> Result<Input> {
            panic!("open({name}) must not be reached: the generation scan should have failed")
        }
        fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
            panic!("create_output({name}) is not part of the read path under test")
        }
        fn sync(&self, _names: &[String]) -> Result<()> {
            panic!("sync() is not part of the read path under test")
        }
        fn rename(&self, source: &str, dest: &str) -> Result<()> {
            panic!("rename({source}, {dest}) is not part of the read path under test")
        }
        fn delete_file(&self, name: &str) -> Result<()> {
            panic!("delete_file({name}) is not part of the read path under test")
        }
        fn sync_meta_data(&self) -> Result<()> {
            panic!("sync_meta_data() is not part of the read path under test")
        }
    }

    #[test]
    fn input_debug_reports_provenance_and_length_not_contents() {
        // The Debug impl exists so a panic message can't dump a mapped
        // half-gigabyte file; assert it stays that way.
        let owned = Input::Owned(vec![7u8; 42]);
        assert_eq!(format!("{owned:?}"), "Input::Owned(42 bytes)");

        let root = tempdir();
        // `with_read_threshold(_, 0)`: a ten-byte file is below
        // `SMALL_FILE_READ_THRESHOLD`, so the default backend would read it.
        let dir = MmapDirectory::with_read_threshold(&root, 0);
        index_output::write_all_bytes(&root, "_0.si", b"0123456789").unwrap();
        let mapped = dir.open("_0.si").unwrap();
        assert_eq!(format!("{mapped:?}"), "Input::Mapped(10 bytes)");
        fs::remove_dir_all(&root).ok();
    }

    /// [`SMALL_FILE_READ_THRESHOLD`] decides *how* a file is obtained and
    /// nothing else: the bytes a caller sees are identical either way.
    ///
    /// Both sides are asserted, because the interesting failure is silent --
    /// a threshold that never fires costs the syscalls it exists to remove
    /// and nothing reports it, and one that fires on a large file copies a
    /// mapping the reader meant to hold.
    #[test]
    fn small_files_are_read_and_large_ones_mapped() {
        let root = tempdir();
        let small = vec![3u8; 8];
        let large = vec![4u8; (SMALL_FILE_READ_THRESHOLD as usize) + 1];
        index_output::write_all_bytes(&root, "_0.nvm", &small).unwrap();
        index_output::write_all_bytes(&root, "_0.doc", &large).unwrap();

        let dir = MmapDirectory::open(&root);
        let got_small = dir.open("_0.nvm").unwrap();
        let got_large = dir.open("_0.doc").unwrap();
        assert!(matches!(got_small, Input::Owned(_)), "{got_small:?}");
        assert!(matches!(got_large, Input::Mapped(_)), "{got_large:?}");
        assert_eq!(&*got_small, &small[..]);
        assert_eq!(&*got_large, &large[..]);

        // Exactly at the threshold is still read (`<=`).
        let exact = vec![5u8; SMALL_FILE_READ_THRESHOLD as usize];
        index_output::write_all_bytes(&root, "_0.dvm", &exact).unwrap();
        assert!(matches!(dir.open("_0.dvm").unwrap(), Input::Owned(_)));

        // And `0` turns the whole thing off, which is the A/B arm
        // `reader_open_profile` measures against.
        let mapping = MmapDirectory::with_read_threshold(&root, 0);
        assert!(matches!(mapping.open("_0.nvm").unwrap(), Input::Mapped(_)));
        assert_eq!(&*mapping.open("_0.nvm").unwrap(), &small[..]);
        fs::remove_dir_all(&root).ok();
    }

    /// A **zero-length** file: the edge `SMALL_FILE_READ_THRESHOLD` moves
    /// across a code path, checked to move nothing observable.
    ///
    /// `0 <= threshold` is always true, so an empty file now takes the read
    /// arm where it used to be mapped. It is worth pinning because the
    /// obvious guess about the old behaviour is wrong: `mmap(2)` rejects a
    /// zero length, but `memmap2::Mmap::map` special-cases it and hands back
    /// an empty mapping rather than an error -- so the two arms *already*
    /// agreed and this change kept them agreeing. Asserted rather than
    /// assumed, because a reviewer of this batch predicted the opposite and
    /// only running it settled which.
    #[test]
    fn a_zero_length_file_reads_as_empty_on_every_backend() {
        let root = tempdir();
        index_output::write_all_bytes(&root, "_0.nvm", b"").unwrap();

        // The read arm, which an empty file now takes.
        let read = MmapDirectory::open(&root).open("_0.nvm").unwrap();
        assert!(matches!(read, Input::Owned(_)), "{read:?}");
        assert!(read.is_empty());

        // The mapping arm, which it used to take -- still `Ok`, still empty.
        let mapped = MmapDirectory::with_read_threshold(&root, 0)
            .open("_0.nvm")
            .expect("memmap2 maps a zero-length file rather than refusing it");
        assert!(matches!(mapped, Input::Mapped(_)), "{mapped:?}");
        assert!(mapped.is_empty());

        // And the copying backend, unchanged throughout.
        assert!(FsDirectory::open(&root).open("_0.nvm").unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn pending_segments_file_name_matches_file_name_from_generation() {
        assert_eq!(pending_segments_file_name(-1), None);
        assert_eq!(
            pending_segments_file_name(0),
            Some("pending_segments".to_string())
        );
        assert_eq!(
            pending_segments_file_name(1),
            Some("pending_segments_1".to_string())
        );
        // base-36, same radix as `segments_N`.
        assert_eq!(
            pending_segments_file_name(10),
            Some("pending_segments_a".to_string())
        );
    }

    /// The whole point of the pending name: `getLastCommitGeneration` must not
    /// see it, so a half-written commit can never become the current one.
    #[test]
    fn a_pending_segments_file_is_invisible_to_the_commit_generation_scan() {
        let files = vec![
            "segments_1".to_string(),
            "pending_segments_2".to_string(),
            "_0.si".to_string(),
        ];
        assert_eq!(last_commit_generation(&files).unwrap(), 1);
    }

    #[test]
    fn rename_publishes_a_file_under_a_new_name_and_delete_file_removes_it() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);

        index_output::write_all_bytes(&root, "pending_segments_1", b"commit").unwrap();
        assert!(dir
            .list_all()
            .unwrap()
            .contains(&"pending_segments_1".to_string()));

        dir.rename("pending_segments_1", "segments_1").unwrap();
        dir.sync_meta_data().unwrap();
        let listed = dir.list_all().unwrap();
        assert!(!listed.contains(&"pending_segments_1".to_string()));
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit");

        dir.delete_file("segments_1").unwrap();
        assert!(dir.list_all().unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rename_and_delete_file_surface_io_errors_for_a_missing_source() {
        let root = tempdir();
        let dir = MmapDirectory::open(&root);
        assert!(matches!(
            dir.rename("nope", "segments_1"),
            Err(Error::Io(_))
        ));
        assert!(matches!(dir.delete_file("nope"), Err(Error::Io(_))));
        // `sync_meta_data` is best-effort by design (not every platform lets a
        // directory be fsynced) -- it reports success either way.
        dir.sync_meta_data().unwrap();
        fs::remove_dir_all(&root).ok();
    }

    use lucene_util::test_support::TempDir;

    /// A scratch directory that removes itself when the test ends -- unless
    /// the test is panicking, in which case its bytes stay for inspection.
    fn tempdir() -> TempDir {
        TempDir::new("directory-write")
    }

    #[test]
    fn fs_directory_create_output_round_trips_through_open_and_list_all() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);

        let mut out = dir.create_output("_0.si").unwrap();
        out.write_bytes(b"hello lucene-rust");
        let checksum = out.close().unwrap();
        assert_eq!(checksum, crc32fast::hash(b"hello lucene-rust") as u64);

        dir.sync(&["_0.si".to_string()]).unwrap();

        assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
        let bytes = dir.open("_0.si").unwrap();
        assert_eq!(&*bytes, b"hello lucene-rust");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn mmap_directory_create_output_round_trips_through_open() {
        let root = tempdir();
        let dir = MmapDirectory::open(&root);

        let mut out = dir.create_output("_0.si").unwrap();
        out.write_bytes(b"mmap round trip");
        out.close().unwrap();

        let bytes = dir.open("_0.si").unwrap();
        assert_eq!(&*bytes, b"mmap round trip");

        assert_eq!(dir.list_all().unwrap(), vec!["_0.si".to_string()]);
        dir.sync(&["_0.si".to_string()]).unwrap();

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn slice_input_over_a_real_file_has_independent_file_pointers() {
        // Writes a real file to a temp dir via FsDirectory/FsIndexOutput, then
        // slices it via SliceInput::slice_input the way a merge would slice a
        // sub-range of a real on-disk `.cfs`. Guards against any real
        // OS-file-handle-sharing bug that an in-memory-only test can't catch
        // (e.g. accidentally sharing one file's read position across slices).
        use crate::data_input::{DataInput, SliceInput};

        let root = tempdir();
        let dir = FsDirectory::open(&root);
        let mut out = dir.create_output("_0.cfs").unwrap();
        out.write_bytes(b"HEADER|firstpart|secondpart|FOOTER");
        out.close().unwrap();

        let bytes = dir.open("_0.cfs").unwrap();
        let root_input = SliceInput::new(&bytes);

        // "firstpart" starts at offset 7, "secondpart" at offset 17.
        let mut first = root_input.slice_input("first", 7, 9).unwrap();
        let mut second = root_input.slice_input("second", 17, 10).unwrap();

        // Interleave reads through both real-file-backed slices.
        let mut buf1 = [0u8; 4];
        let mut buf2 = [0u8; 4];
        first.read_bytes(&mut buf1).unwrap();
        second.read_bytes(&mut buf2).unwrap();
        assert_eq!(&buf1, b"firs");
        assert_eq!(&buf2, b"seco");

        let mut buf1b = [0u8; 5];
        let mut buf2b = [0u8; 6];
        first.read_bytes(&mut buf1b).unwrap();
        second.read_bytes(&mut buf2b).unwrap();
        assert_eq!(&buf1b, b"tpart");
        assert_eq!(&buf2b, b"ndpart");

        // Both slices are now fully consumed; further reads are Eof, not a
        // leak into the other slice's or the footer's bytes.
        assert!(first.read_byte().is_err());
        assert!(second.read_byte().is_err());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn fs_directory_open_nonexistent_file_is_io_error() {
        let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
    }

    #[test]
    fn fs_directory_list_all_nonexistent_dir_is_io_error() {
        let dir = FsDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.list_all(), Err(Error::Io(_))));
    }

    #[test]
    fn mmap_directory_open_nonexistent_file_is_io_error() {
        let dir = MmapDirectory::open("/nonexistent-lucene-rust-test-path");
        assert!(matches!(dir.open("whatever"), Err(Error::Io(_))));
    }

    #[test]
    fn read_latest_commit_finds_highest_generation_segments_file() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);
        index_output::write_all_bytes(&root, "segments_1", b"old").unwrap();
        index_output::write_all_bytes(&root, "segments_2", b"newest").unwrap();

        let (generation, bytes) = read_latest_commit(&dir).unwrap();
        assert_eq!(generation, 2);
        assert_eq!(&*bytes, b"newest");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn read_latest_commit_rejects_an_unparsable_segments_file_name() {
        // Java's getLastCommitGeneration lets generationFromSegmentsFileName's
        // exception escape here. Skipping the bad name instead would open
        // segments_1 and silently drop whatever segments_2 committed.
        let files = vec![
            "segments_1".to_string(),
            "segments_zzzzzzzzzzzzz".to_string(),
        ];
        assert!(matches!(
            read_latest_commit(&ListingOnlyDir(files.clone())),
            Err(Error::Corrupted(_))
        ));
        // ... and the scan the write path shares with it is equally strict:
        // reporting generation 1 here would let `IndexWriter` create a fresh
        // index over a directory that does hold a commit.
        assert!(matches!(
            last_commit_generation(&files),
            Err(Error::Corrupted(_))
        ));
    }

    #[test]
    fn read_latest_commit_ignores_the_legacy_pointer_file() {
        let root = tempdir();
        let dir = FsDirectory::open(&root);
        index_output::write_all_bytes(&root, "segments.gen", b"legacy").unwrap();
        index_output::write_all_bytes(&root, "segments_1", b"real").unwrap();
        let (generation, bytes) = read_latest_commit(&dir).unwrap();
        assert_eq!(generation, 1);
        assert_eq!(&*bytes, b"real");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn read_latest_commit_no_segments_file_is_corrupted_error() {
        assert!(matches!(
            read_latest_commit(&ListingOnlyDir(vec!["_0.si".to_string()])),
            Err(Error::Corrupted(_))
        ));
    }
}
