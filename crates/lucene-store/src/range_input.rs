//! Explicit fallible input for immutable, version-bound codec files.
//!
//! Java: IndexInput.clone/slice/seek/readBytes and RandomAccessInput. Rust
//! keeps I/O out of Deref/AsRef: only read methods may call the source. Clones
//! have independent cursors but share a finite cache and physical I/O budget.
//! This is an additive prototype; existing materialized Input is unchanged.

use crate::{DataInput, Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// One immutable file/version. Implementations must bind every request to the
/// same version and must return an error if that binding cannot be maintained.
/// A successful short read is permitted here and rejected by RangeInput.
/// No implementation may claim bytes it did not initialize.
pub trait ReadAt: Send + Sync {
    /// Stable length of the selected immutable version.
    fn len(&self) -> usize;
    /// Whether this file is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// One physical attempt, with an initialized destination; no implicit retry.
    fn read_at(&self, offset: usize, dst: &mut [u8]) -> Result<usize>;
}

/// Finite shared envelope, not a measured service default.
#[derive(Clone, Copy, Debug)]
pub struct RangeLimits {
    pub block_bytes: usize,
    pub cache_blocks: usize,
    pub attempts: u64,
    pub transferred_bytes: u64,
}
impl RangeLimits {
    fn validate(self) -> Result<Self> {
        if self.block_bytes == 0
            || self.block_bytes > 1024 * 1024
            || self.cache_blocks == 0
            || self.cache_blocks > 256
            || self
                .block_bytes
                .checked_mul(self.cache_blocks)
                .is_none_or(|n| n > 16 * 1024 * 1024)
            || self.attempts == 0
            || self.attempts > 1_000_000
            || self.transferred_bytes == 0
            || self.transferred_bytes > 1024 * 1024 * 1024
        {
            return Err(Error::IllegalArgument(
                "invalid range input envelope".into(),
            ));
        }
        Ok(self)
    }
}

/// Physical attempts are charged before dispatch, including failed/short reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RangeUsage {
    pub attempts: u64,
    pub transferred_bytes: u64,
    pub cache_bytes: usize,
}
struct Block {
    start: usize,
    bytes: Vec<u8>,
}
struct State {
    blocks: VecDeque<Block>,
    usage: RangeUsage,
}
struct Shared {
    source: Arc<dyn ReadAt>,
    length: usize,
    limits: RangeLimits,
    state: Mutex<State>,
}

/// Independent cursor and checked sub-file bounds, sharing one source/cache.
/// No borrowed slice may outlive cache eviction because reads copy into caller
/// storage. On error the cursor does not advance; destination stays initialized
/// but may be partially changed and must not be interpreted as a complete read.
#[derive(Clone)]
pub struct RangeInput {
    shared: Arc<Shared>,
    start: usize,
    length: usize,
    position: usize,
}
impl std::fmt::Debug for RangeInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeInput")
            .field("length", &self.length)
            .field("position", &self.position)
            .finish()
    }
}
impl RangeInput {
    pub fn new(source: Arc<dyn ReadAt>, limits: RangeLimits) -> Result<Self> {
        let limits = limits.validate()?;
        let length = source.len();
        Ok(Self {
            shared: Arc::new(Shared {
                source,
                length,
                limits,
                state: Mutex::new(State {
                    blocks: VecDeque::new(),
                    usage: RangeUsage::default(),
                }),
            }),
            start: 0,
            length,
            position: 0,
        })
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.length {
            return Err(Error::Eof { offset: position });
        }
        self.position = position;
        Ok(())
    }
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        self.check(offset, length)?;
        let start = self.start.checked_add(offset).ok_or_else(overflow)?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            start,
            length,
            position: 0,
        })
    }
    pub fn usage(&self) -> Result<RangeUsage> {
        Ok(self.shared.state.lock().map_err(|_| poisoned())?.usage)
    }
    /// Evict all resident blocks. This does not reset charged I/O budgets.
    pub fn clear_cache(&self) -> Result<()> {
        let mut state = self.shared.state.lock().map_err(|_| poisoned())?;
        state.blocks.clear();
        state.usage.cache_bytes = 0;
        Ok(())
    }
    fn check(&self, offset: usize, length: usize) -> Result<()> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.length)
        {
            return Err(Error::Eof { offset });
        }
        Ok(())
    }
    /// Fallible random read; never changes this input's cursor.
    pub fn read_exact_at(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        self.check(offset, dst.len())?;
        let absolute = self.start.checked_add(offset).ok_or_else(overflow)?;
        let mut done = 0usize;
        // Serializing misses keeps both transient allocations and in-flight
        // source attempts bounded. No caller/backend lock may call back here.
        let mut state = self.shared.state.lock().map_err(|_| poisoned())?;
        while done < dst.len() {
            let at = absolute.checked_add(done).ok_or_else(overflow)?;
            let block_start = at
                .checked_div(self.shared.limits.block_bytes)
                .and_then(|n| n.checked_mul(self.shared.limits.block_bytes))
                .ok_or_else(overflow)?;
            if !state.blocks.iter().any(|b| b.start == block_start) {
                let count = self
                    .shared
                    .length
                    .checked_sub(block_start)
                    .ok_or_else(overflow)?
                    .min(self.shared.limits.block_bytes);
                let attempts = state.usage.attempts.checked_add(1).ok_or_else(overflow)?;
                let transferred = state
                    .usage
                    .transferred_bytes
                    .checked_add(count as u64)
                    .ok_or_else(overflow)?;
                if attempts > self.shared.limits.attempts
                    || transferred > self.shared.limits.transferred_bytes
                {
                    return Err(Error::Io(std::io::Error::other(
                        "range input budget exhausted",
                    )));
                }
                // Evict before allocation: resident plus in-flight <= cache cap.
                if state.blocks.len() == self.shared.limits.cache_blocks {
                    let removed = state.blocks.pop_front().ok_or_else(overflow)?;
                    state.usage.cache_bytes = state
                        .usage
                        .cache_bytes
                        .checked_sub(removed.bytes.len())
                        .ok_or_else(overflow)?;
                }
                let mut bytes = vec![0u8; count];
                state.usage.attempts = attempts;
                state.usage.transferred_bytes = transferred;
                let received = self.shared.source.read_at(block_start, &mut bytes)?;
                if received != count {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "incomplete range response",
                    )));
                }
                state.usage.cache_bytes = state
                    .usage
                    .cache_bytes
                    .checked_add(count)
                    .ok_or_else(overflow)?;
                state.blocks.push_back(Block {
                    start: block_start,
                    bytes,
                });
            }
            let block = state
                .blocks
                .iter()
                .find(|b| b.start == block_start)
                .ok_or_else(overflow)?;
            let within = at.checked_sub(block_start).ok_or_else(overflow)?;
            let count = block
                .bytes
                .len()
                .checked_sub(within)
                .ok_or_else(overflow)?
                .min(dst.len().checked_sub(done).ok_or_else(overflow)?);
            let end = done.checked_add(count).ok_or_else(overflow)?;
            let block_end = within.checked_add(count).ok_or_else(overflow)?;
            dst.get_mut(done..end)
                .ok_or_else(overflow)?
                .copy_from_slice(block.bytes.get(within..block_end).ok_or_else(overflow)?);
            done = end;
        }
        Ok(())
    }
}
impl RangeInput {
    /// Codec footer integrity without materializing the file.
    pub fn check_integrity(&self) -> Result<()> {
        let mut footer = [0u8; crate::codec_util::FOOTER_LENGTH];
        let at = self.len().checked_sub(footer.len()).ok_or_else(overflow)?;
        self.read_exact_at(at, &mut footer)?;
        let expected = crate::codec_util::retrieve_checksum(&footer)?;
        let end = self.len().checked_sub(8).ok_or_else(overflow)?;
        let mut hasher = crc32fast::Hasher::new();
        let mut chunk = [0u8; 4096];
        let mut offset = 0usize;
        while offset < end {
            let count = end
                .checked_sub(offset)
                .ok_or_else(overflow)?
                .min(chunk.len());
            self.read_exact_at(offset, &mut chunk[..count])?;
            hasher.update(&chunk[..count]);
            offset = offset.checked_add(count).ok_or_else(overflow)?;
        }
        if u64::from(hasher.finalize()) != expected {
            return Err(Error::Corrupted("range checksum mismatch".into()));
        }
        Ok(())
    }
}

impl DataInput for RangeInput {
    fn read_byte(&mut self) -> Result<u8> {
        let mut out = [0];
        self.read_bytes(&mut out)?;
        Ok(out[0])
    }
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        self.read_exact_at(self.position, out)?;
        self.position = self.position.checked_add(out.len()).ok_or_else(overflow)?;
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.length.saturating_sub(self.position)
    }
    fn peek_u32_le(&mut self) -> Result<u32> {
        let mut out = [0; 4];
        self.read_exact_at(self.position, &mut out)?;
        Ok(u32::from_le_bytes(out))
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.seek(self.position.checked_add(n).ok_or_else(overflow)?)
    }
}
fn overflow() -> Error {
    Error::Corrupted("range input bounds".into())
}
fn poisoned() -> Error {
    Error::Io(std::io::Error::other("range input state poisoned"))
}

/// Explicit resident-metadata policy: at most 128 files, 1 MiB per file and
/// 16 MiB total reserved scratch. The caller supplies a fresh, exclusively
/// owned scratch directory and retains it until every returned mapping dies.
/// There is no heap-buffer fallback and no automatic deletion/refund on error.
/// Physical reads remain charged to the supplied RangeInput's shared budget.
pub struct MetadataMaterializer {
    root: std::path::PathBuf,
    usage: Mutex<MetadataUsage>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetadataUsage {
    pub files: usize,
    pub reserved_bytes: usize,
}
impl MetadataMaterializer {
    pub fn new(root: &std::path::Path) -> Result<Self> {
        let info = std::fs::symlink_metadata(root)?;
        if !info.is_dir()
            || info.file_type().is_symlink()
            || std::fs::read_dir(root)?.next().is_some()
        {
            return Err(Error::IllegalArgument(
                "metadata scratch must be an empty owned directory".into(),
            ));
        }
        Ok(Self {
            root: root.canonicalize()?,
            usage: Mutex::new(MetadataUsage::default()),
        })
    }
    pub fn usage(&self) -> Result<MetadataUsage> {
        Ok(*self.usage.lock().map_err(|_| poisoned())?)
    }
    /// Fully initialize and close a fresh file before mapping it. A failed
    /// transfer leaves charged, unadvertised scratch; its name cannot be reused.
    /// Source must be bound to the requested immutable version. The trusted
    /// scratch owner must prohibit external modification/truncation of maps.
    pub fn materialize(&self, name: &str, source: &RangeInput) -> Result<crate::Input> {
        use crate::Directory;
        use std::io::Write;
        if name.is_empty()
            || name.len() > 128
            || matches!(name, "." | "..")
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || source.len() > 1024 * 1024
        {
            return Err(Error::IllegalArgument("metadata name/file envelope".into()));
        }
        let mut usage = self.usage.lock().map_err(|_| poisoned())?;
        let files = usage.files.checked_add(1).ok_or_else(overflow)?;
        let bytes = usage
            .reserved_bytes
            .checked_add(source.len())
            .ok_or_else(overflow)?;
        if files > 128 || bytes > 16 * 1024 * 1024 {
            return Err(Error::IllegalArgument("metadata scratch exhausted".into()));
        }
        // Reserve before creating or dispatching. Failed reservations/transfers
        // are never refunded as though the file were proven absent.
        *usage = MetadataUsage {
            files,
            reserved_bytes: bytes,
        };
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(name))?;
        let mut chunk = [0u8; 4096];
        let mut offset = 0usize;
        while offset < source.len() {
            let length = source
                .len()
                .checked_sub(offset)
                .ok_or_else(overflow)?
                .min(chunk.len());
            source.read_exact_at(offset, &mut chunk[..length])?;
            file.write_all(&chunk[..length])?;
            offset = offset.checked_add(length).ok_or_else(overflow)?;
        }
        file.flush()?;
        drop(file);
        // Zero threshold preserves the existing mapping implementation even
        // for small metadata files. No newly introduced unsafe is necessary.
        crate::MmapDirectory::with_read_threshold(&self.root, 0).open(name)
    }
}

/// Read-only immutable inventory for the native range reader prototype.
/// Data files cannot escape into the materialized open API. Other touched
/// files are streamed under MetadataMaterializer's finite charged policy.
pub struct RangeDirectory {
    files: std::collections::BTreeMap<String, RangeInput>,
    metadata: MetadataMaterializer,
    resident: Mutex<std::collections::BTreeMap<String, Arc<crate::Input>>>,
}
impl RangeDirectory {
    pub fn new(scratch: &std::path::Path, files: Vec<(String, RangeInput)>) -> Result<Self> {
        if files.len() > 128 {
            return Err(Error::IllegalArgument("range inventory file limit".into()));
        }
        let mut inventory = std::collections::BTreeMap::new();
        let mut total = 0usize;
        for (name, input) in files {
            if name.is_empty()
                || name.len() > 128
                || matches!(name.as_str(), "." | "..")
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            {
                return Err(Error::IllegalArgument(
                    "invalid range inventory name".into(),
                ));
            }
            total = total.checked_add(input.len()).ok_or_else(overflow)?;
            if total > 128 * 1024 * 1024 || inventory.insert(name, input).is_some() {
                return Err(Error::IllegalArgument(
                    "range inventory byte/identity limit".into(),
                ));
            }
        }
        Ok(Self {
            files: inventory,
            metadata: MetadataMaterializer::new(scratch)?,
            resident: Mutex::new(std::collections::BTreeMap::new()),
        })
    }
    pub fn metadata_usage(&self) -> Result<MetadataUsage> {
        self.metadata.usage()
    }
    fn data_file(name: &str) -> bool {
        [".tim", ".doc", ".dvd", ".fdt", ".cfs"]
            .iter()
            .any(|ext| name.ends_with(ext))
    }
    fn source(&self, name: &str) -> Result<&RangeInput> {
        self.files.get(name).ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "range file absent",
            ))
        })
    }
}
impl crate::Directory for RangeDirectory {
    fn list_all(&self) -> Result<Vec<String>> {
        Ok(self.files.keys().cloned().collect())
    }
    fn open(&self, name: &str) -> Result<crate::Input> {
        if Self::data_file(name) {
            return Err(Error::IllegalArgument(
                "range data cannot be materialized by Directory::open".into(),
            ));
        }
        let source = self.source(name)?;
        let mut resident = self.resident.lock().map_err(|_| poisoned())?;
        if !resident.contains_key(name) {
            resident.insert(
                name.to_owned(),
                Arc::new(self.metadata.materialize(name, source)?),
            );
        }
        let bytes = resident.get(name).ok_or_else(overflow)?;
        crate::Input::slice_of(bytes, 0, bytes.len())
    }
    fn open_range(&self, name: &str) -> Result<Option<RangeInput>> {
        if Self::data_file(name) {
            Ok(Some(self.source(name)?.clone()))
        } else {
            Ok(None)
        }
    }
    fn is_range_directory(&self) -> bool {
        true
    }
    fn file_length(&self, name: &str) -> Result<u64> {
        Ok(self.source(name)?.len() as u64)
    }
    fn create_output(&self, _name: &str) -> Result<crate::FsIndexOutput> {
        Err(read_only())
    }
    fn sync(&self, _names: &[String]) -> Result<()> {
        Err(read_only())
    }
    fn rename(&self, _source: &str, _dest: &str) -> Result<()> {
        Err(read_only())
    }
    fn delete_file(&self, _name: &str) -> Result<()> {
        Err(read_only())
    }
    fn sync_meta_data(&self) -> Result<()> {
        Err(read_only())
    }
    fn obtain_lock(&self, _name: &str) -> Result<Box<dyn crate::Lock>> {
        Err(read_only())
    }
}
fn read_only() -> Error {
    Error::IllegalArgument("immutable range directory".into())
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Source {
        fail: bool,
    }
    impl ReadAt for Source {
        fn len(&self) -> usize {
            128
        }
        fn read_at(&self, offset: usize, dst: &mut [u8]) -> Result<usize> {
            if self.fail {
                return Err(std::io::Error::other("injected").into());
            }
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = offset.checked_add(i).unwrap() as u8;
            }
            Ok(dst.len())
        }
    }
    fn source(fail: bool) -> RangeInput {
        RangeInput::new(
            Arc::new(Source { fail }),
            RangeLimits {
                block_bytes: 16,
                cache_blocks: 2,
                attempts: 64,
                transferred_bytes: 1024,
            },
        )
        .unwrap()
    }
    fn scratch() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lucene-range-metadata-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn maps_only_complete_initialized_files_and_counts_physical_reads() {
        let root = scratch();
        let materializer = MetadataMaterializer::new(&root).unwrap();
        let input = source(false);
        let mapped = materializer.materialize("segment.tip", &input).unwrap();
        assert_eq!(mapped.len(), 128);
        assert!(matches!(mapped, crate::Input::Mapped(_)));
        assert_eq!(&mapped[..4], &[0, 1, 2, 3]);
        assert_eq!(input.usage().unwrap().attempts, 8);
        assert_eq!(
            materializer.usage().unwrap(),
            MetadataUsage {
                files: 1,
                reserved_bytes: 128
            }
        );
        assert!(materializer.materialize("../escape", &input).is_err());
        assert!(materializer.materialize("segment.tip", &input).is_err());
        assert_eq!(materializer.usage().unwrap().reserved_bytes, 256);
        drop(mapped);
        drop(materializer);
        std::fs::remove_file(root.join("segment.tip")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn incomplete_file_stays_charged_and_cannot_be_reused() {
        let root = scratch();
        let materializer = MetadataMaterializer::new(&root).unwrap();
        assert!(materializer
            .materialize("failed.tip", &source(true))
            .is_err());
        assert_eq!(
            materializer.usage().unwrap(),
            MetadataUsage {
                files: 1,
                reserved_bytes: 128
            }
        );
        assert_eq!(std::fs::metadata(root.join("failed.tip")).unwrap().len(), 0);
        assert!(MetadataMaterializer::new(&root).is_err());
        assert!(materializer
            .materialize("failed.tip", &source(false))
            .is_err());
        drop(materializer);
        std::fs::remove_file(root.join("failed.tip")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Source {
        bytes: Vec<u8>,
        fail: AtomicUsize,
    }
    impl ReadAt for Source {
        fn len(&self) -> usize {
            self.bytes.len()
        }
        fn read_at(&self, offset: usize, dst: &mut [u8]) -> Result<usize> {
            match self.fail.load(Ordering::Relaxed) {
                1 => return Err(std::io::Error::other("injected").into()),
                2 => return Ok(0),
                3 => return Ok(usize::MAX),
                _ => {}
            }
            let end = offset.checked_add(dst.len()).unwrap();
            dst.copy_from_slice(&self.bytes[offset..end]);
            Ok(dst.len())
        }
    }
    fn input() -> (Arc<Source>, RangeInput) {
        let source = Arc::new(Source {
            bytes: (0..128).collect(),
            fail: AtomicUsize::new(0),
        });
        let input = RangeInput::new(
            source.clone(),
            RangeLimits {
                block_bytes: 4,
                cache_blocks: 2,
                attempts: 100,
                transferred_bytes: 400,
            },
        )
        .unwrap();
        (source, input)
    }
    #[test]
    fn evictions_slices_and_clone_cursors_do_not_change_bytes() {
        let (_, mut input) = input();
        let mut copy = input.clone();
        input.seek(3).unwrap();
        let mut bytes = [0; 11];
        input.read_bytes(&mut bytes).unwrap();
        assert_eq!(bytes, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        assert_eq!(input.position(), 14);
        assert_eq!(copy.position(), 0);
        assert_eq!(copy.read_byte().unwrap(), 0);
        let mut slice = input.slice(2, 20).unwrap().slice(3, 10).unwrap();
        assert_eq!(slice.read_byte().unwrap(), 5);
        assert!(slice.slice(9, 2).is_err());
        assert!(slice.slice(usize::MAX, 1).is_err());
        assert!(slice.seek(11).is_err());
        slice.seek(10).unwrap();
        assert!(slice.read_byte().is_err());
        let before = input.usage().unwrap();
        assert!(before.cache_bytes <= 8);
        input.clear_cache().unwrap();
        assert_eq!(input.usage().unwrap().cache_bytes, 0);
        assert_eq!(input.usage().unwrap().attempts, before.attempts);
    }
    #[test]
    fn failed_or_short_attempt_never_populates_cache_or_advances_cursor() {
        for mode in [1, 2, 3] {
            let (source, mut input) = input();
            source.fail.store(mode, Ordering::Relaxed);
            let mut out = [9; 4];
            assert!(input.read_bytes(&mut out).is_err());
            assert_eq!(input.position(), 0);
            assert_eq!(out, [9; 4]);
            assert_eq!(
                input.usage().unwrap(),
                RangeUsage {
                    attempts: 1,
                    transferred_bytes: 4,
                    cache_bytes: 0
                }
            );
            source.fail.store(0, Ordering::Relaxed);
            input.read_bytes(&mut out).unwrap();
            assert_eq!(out, [0, 1, 2, 3]);
        }
    }
    #[test]
    fn preflight_and_exhaustion_do_not_dispatch_extra_attempts() {
        let (source, _) = input();
        let mut input = RangeInput::new(
            source.clone(),
            RangeLimits {
                block_bytes: 4,
                cache_blocks: 1,
                attempts: 1,
                transferred_bytes: 4,
            },
        )
        .unwrap();
        assert!(input.read_exact_at(usize::MAX, &mut [0; 1]).is_err());
        assert_eq!(input.usage().unwrap().attempts, 0);
        assert_eq!(input.read_u32_le().unwrap(), 0x03020100);
        assert!(input.read_byte().is_err());
        assert_eq!(input.position(), 4);
        assert_eq!(input.usage().unwrap().attempts, 1);
        assert!(RangeInput::new(
            source,
            RangeLimits {
                block_bytes: 0,
                cache_blocks: 1,
                attempts: 1,
                transferred_bytes: 4
            }
        )
        .is_err());
    }
    #[test]
    fn concurrent_clones_share_cache_and_budget() {
        let (_, input) = input();
        let a = input.clone();
        let b = input.clone();
        let one = std::thread::spawn(move || {
            let mut out = [0; 8];
            a.read_exact_at(20, &mut out).unwrap();
            out
        });
        let two = std::thread::spawn(move || {
            let mut out = [0; 8];
            b.read_exact_at(20, &mut out).unwrap();
            out
        });
        assert_eq!(one.join().unwrap(), [20, 21, 22, 23, 24, 25, 26, 27]);
        assert_eq!(two.join().unwrap(), [20, 21, 22, 23, 24, 25, 26, 27]);
        assert_eq!(input.usage().unwrap().attempts, 2);
    }
}
