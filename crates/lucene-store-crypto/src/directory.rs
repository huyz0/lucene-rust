//! [`EncryptedDirectory`]: a [`Directory`] wrapper that encrypts every file
//! it writes and decrypts every file it opens (application-level
//! encryption), over any directory underneath -- a local filesystem, a
//! [`lucene_store::BlobDirectory`], anything else that implements the trait.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use lucene_store::directory::Input;
use lucene_store::lock::Lock;
use lucene_store::{DataOutput, Directory, Error, FsIndexOutput, IndexOutput, OutputSink, Result};

use crate::format::{plaintext_len, ChunkCipher, Header, DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE};
use crate::key_provider::{GeneratedKey, KeyProvider};

/// Encrypts the files of `D` with per-file data keys wrapped by a
/// [`KeyProvider`]. See [`crate::format`] for the file layout.
///
/// Every operation that does not touch file contents (listing, rename,
/// delete, sync, locking) passes straight through, so the wrapper keeps the
/// inner directory's commit and durability semantics. `write.lock` is the
/// inner directory's lock and is not encrypted (it is empty).
///
/// An [`Input`] is a contiguous byte slice, so by default a file is
/// decrypted whole when it is opened and costs its size in memory while a
/// reader holds it open (AES-GCM with AES-NI decrypts at several GB/s per
/// core). With [`EncryptedDirectory::with_lazy_decrypt`] on Linux, a file is
/// authenticated on open and then decrypted page by page on first touch
/// into a mapping bounded by a process-wide budget ([`crate::lazy`]).
///
/// Opened plaintext is shared through a [`PlaintextCache`]: while any input
/// of a file is alive, opening it again costs a reference count. Directories
/// over the same files can share one cache
/// ([`EncryptedDirectory::with_plaintext_cache`]).
pub struct EncryptedDirectory<D: Directory> {
    inner: D,
    keys: Arc<dyn KeyProvider>,
    chunk_size: u32,
    /// Each known file's header length, so [`Directory::file_length`] does
    /// not read the file (a whole download, over a blob store). Lucene files
    /// are written once under a name; a delete or a rename forgets the entry.
    header_lens: Mutex<HashMap<String, (u64, u32)>>,
    /// Whether [`is_commit_metadata`] files are stored in plaintext.
    plain_commit_metadata: bool,
    /// Whether files are demand-decrypted rather than decrypted whole.
    lazy: bool,
    cache: Arc<PlaintextCache>,
}

/// The index lock file's name (Lucene's `IndexWriter.WRITE_LOCK_NAME`).
const WRITE_LOCK_NAME: &str = "write.lock";

static LAZY_DEFAULT: AtomicBool = AtomicBool::new(false);

/// Sets whether new [`EncryptedDirectory`]s demand-decrypt their files
/// (default off), and returns whether demand decryption is available in
/// this process: on Linux, with `userfaultfd` permitted. Where it is not,
/// files are decrypted whole whatever this is set to.
pub fn set_lazy_decrypt_default(enabled: bool) -> bool {
    LAZY_DEFAULT.store(enabled, Ordering::Relaxed);
    lazy_available()
}

/// Whether new [`EncryptedDirectory`]s ask for demand decryption
/// ([`set_lazy_decrypt_default`]).
pub fn lazy_decrypt_default() -> bool {
    LAZY_DEFAULT.load(Ordering::Relaxed)
}

/// Whether demand decryption works in this process (starting its fault
/// handler if it does).
pub fn lazy_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        crate::lazy::Pager::global().is_some()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Opened plaintext by file name, held weakly: an entry lives exactly as
/// long as some input of the file does. Lucene writes a file once under a
/// name, and a delete, rename or new output of the name drops its entry; the
/// ciphertext length is checked on every hit as well, so a file replaced
/// behind the directory's back is never served from the cache.
#[derive(Default)]
pub struct PlaintextCache {
    files: Mutex<HashMap<String, (u64, Weak<Input>)>>,
}

impl PlaintextCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn files(&self) -> std::sync::MutexGuard<'_, HashMap<String, (u64, Weak<Input>)>> {
        self.files.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn get(&self, name: &str, sealed_len: u64) -> Option<Arc<Input>> {
        let mut files = self.files();
        let (len, weak) = files.get(name)?;
        match weak.upgrade() {
            Some(input) if *len == sealed_len => Some(input),
            _ => {
                files.remove(name);
                None
            }
        }
    }

    fn put(&self, name: &str, sealed_len: u64, input: &Arc<Input>) {
        let mut files = self.files();
        files.retain(|_, (_, w)| w.strong_count() > 0);
        files.insert(name.to_string(), (sealed_len, Arc::downgrade(input)));
    }

    fn forget(&self, name: &str) {
        self.files().remove(name);
    }
}

impl std::fmt::Debug for PlaintextCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PlaintextCache({} files)", self.files().len())
    }
}

/// Whether `name` is commit metadata: a `segments_N` (or `pending_segments_N`)
/// file, a segment's `.si` file, or an OpenSearch `corrupted_*` marker,
/// including under a recovery temp name (`recovery.<uuid>.<name>`).
///
/// These hold segment names, document counts, codec names, file lists and
/// commit user data, never field contents. OpenSearch reads them through a
/// plain `NIOFSDirectory` before any store opens (gateway allocation of a
/// primary after a restart, `Store.tryOpenIndex`), so an encrypted store
/// that OpenSearch must restart from keeps them in plaintext; see
/// [`EncryptedDirectory::with_plaintext_commit_metadata`].
pub fn is_commit_metadata(name: &str) -> bool {
    name.contains("segments_") || name.ends_with(".si") || name.starts_with("corrupted_")
}

impl<D: Directory> std::fmt::Debug for EncryptedDirectory<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EncryptedDirectory(chunk_size={})", self.chunk_size)
    }
}

impl<D: Directory> EncryptedDirectory<D> {
    /// Wraps `inner`, encrypting under `keys`' current key.
    pub fn new(inner: D, keys: Arc<dyn KeyProvider>) -> Self {
        Self {
            inner,
            keys,
            chunk_size: DEFAULT_CHUNK_SIZE,
            header_lens: Mutex::new(HashMap::new()),
            plain_commit_metadata: false,
            lazy: LAZY_DEFAULT.load(Ordering::Relaxed),
            cache: Arc::new(PlaintextCache::new()),
        }
    }

    /// Demand-decrypts files on open instead of decrypting them whole
    /// (default: [`set_lazy_decrypt_default`]). A no-op where demand
    /// decryption is unavailable ([`lazy_available`]).
    pub fn with_lazy_decrypt(mut self, enabled: bool) -> Self {
        self.lazy = enabled;
        self
    }

    /// Shares opened plaintext with every directory given the same cache;
    /// they must be over the same files.
    pub fn with_plaintext_cache(mut self, cache: Arc<PlaintextCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Stores commit metadata ([`is_commit_metadata`]) in plaintext and
    /// encrypts everything else. Off by default; the OpenSearch plugin's
    /// `rust_encrypted` store turns it on, on both its Java and Rust sides,
    /// because OpenSearch reads `segments_N` and `.si` files without the
    /// store's directory when it allocates a shard after a restart.
    pub fn with_plaintext_commit_metadata(mut self) -> Self {
        self.plain_commit_metadata = true;
        self
    }

    /// Whether `name` is stored as-is: the index's lock file (created
    /// empty by the inner directory's lock factory), and commit metadata when
    /// [`EncryptedDirectory::with_plaintext_commit_metadata`] is on.
    fn is_plain(&self, name: &str) -> bool {
        name == WRITE_LOCK_NAME || (self.plain_commit_metadata && is_commit_metadata(name))
    }

    /// An inner output, encrypting unless its name is stored in plaintext.
    fn wrap(&self, out: FsIndexOutput) -> Result<FsIndexOutput> {
        self.cache.forget(out.name());
        if self.is_plain(out.name()) {
            self.header_lens().remove(out.name());
            return Ok(out);
        }
        self.encrypting(out)
    }

    /// Sets the plaintext bytes per authenticated chunk for new files
    /// (default 64 KiB). Existing files keep the size in their header.
    pub fn with_chunk_size(mut self, chunk_size: u32) -> Result<Self> {
        if chunk_size == 0 || chunk_size > MAX_CHUNK_SIZE {
            return Err(Error::IllegalArgument(format!(
                "chunk size {chunk_size} must be in 1..={MAX_CHUNK_SIZE}"
            )));
        }
        self.chunk_size = chunk_size;
        Ok(self)
    }

    /// The directory underneath, which holds ciphertext.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    fn header_lens(&self) -> std::sync::MutexGuard<'_, HashMap<String, (u64, u32)>> {
        self.header_lens.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Wraps an output of the inner directory: a fresh data key and header.
    fn encrypting(&self, out: FsIndexOutput) -> Result<FsIndexOutput> {
        let GeneratedKey {
            key_id,
            data_key,
            wrapped: wrapped_key,
        } = self.keys.generate_data_key()?;
        let header = Header {
            chunk_size: self.chunk_size,
            key_id,
            wrapped_key,
        };
        let header_bytes = header.encode()?;
        let name = out.name().to_string();
        self.header_lens()
            .insert(name.clone(), (header_bytes.len() as u64, self.chunk_size));
        let mut out = out;
        out.write_bytes(&header_bytes);
        let cipher = ChunkCipher::new(&data_key, header_bytes)?;
        Ok(FsIndexOutput::from_sink(
            &name,
            Box::new(EncryptingSink {
                out,
                cipher,
                chunk: self.chunk_size as usize,
                buf: Vec::new(),
                index: 0,
            }),
        ))
    }

    /// The plaintext of `file`, demand-decrypted when that is on and
    /// available, else decrypted whole.
    fn decrypt(
        &self,
        file: Input,
        cipher: ChunkCipher,
        header: &Header,
        header_len: usize,
        name: &str,
    ) -> Result<Input> {
        #[cfg(target_os = "linux")]
        if let Some(pager) =
            crate::lazy::Pager::global().filter(|p| self.lazy && p.supports(header.chunk_size))
        {
            let body = &file[header_len..];
            cipher.verify_body(body, header.chunk_size, name)?;
            let plain_len = plaintext_len(body.len() as u64, header.chunk_size)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    Error::Corrupted(format!("encrypted file {name}: malformed length"))
                })?;
            return pager.open(file, header_len, plain_len, header.chunk_size, cipher, name);
        }
        let plain = cipher.open_body(&file[header_len..], header.chunk_size, name)?;
        Ok(Input::Owned(plain))
    }

    /// The header of `name` from its bytes, and the cipher to open it.
    pub(crate) fn cipher_for(
        &self,
        name: &str,
        file: &[u8],
    ) -> Result<(ChunkCipher, Header, usize)> {
        let (header, header_len) = Header::decode(file, name)?;
        let data_key = self.keys.unwrap_key(&header.key_id, &header.wrapped_key)?;
        let cipher = ChunkCipher::new(&data_key, file[..header_len].to_vec())?;
        Ok((cipher, header, header_len))
    }
}

/// Buffers plaintext into chunks and seals each into the inner output. The
/// last chunk is held back until [`OutputSink::finish`], which seals it as
/// final, so every chunk but the last is exactly `chunk` bytes.
struct EncryptingSink {
    out: FsIndexOutput,
    cipher: ChunkCipher,
    chunk: usize,
    buf: Vec<u8>,
    index: u32,
}

impl EncryptingSink {
    /// Seals `buf[start..end]` in place as chunk `index` and writes it out.
    fn seal(&mut self, start: usize, end: usize, last: bool) -> std::io::Result<()> {
        let piece = &mut self.buf[start..end];
        let (nonce, tag) = self
            .cipher
            .seal(self.index, last, piece)
            .map_err(lucene_store::blob_directory::into_io)?;
        self.out.write_bytes(&nonce);
        self.out.write_bytes(piece);
        self.out.write_bytes(&tag);
        self.index = self
            .index
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("encrypted file exceeds 2^32 chunks"))?;
        Ok(())
    }
}

impl OutputSink for EncryptingSink {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.buf.extend_from_slice(bytes);
        if self.buf.len() <= self.chunk {
            return Ok(());
        }
        // Seal every full chunk that is certainly not the last one.
        let mut sealed = 0usize;
        while self.buf.len().saturating_sub(sealed) > self.chunk {
            let end = sealed.saturating_add(self.chunk);
            self.seal(sealed, end, false)?;
            sealed = end;
        }
        self.buf.drain(..sealed);
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<()> {
        let len = self.buf.len();
        self.seal(0, len, true)?;
        let this = *self;
        this.out
            .close()
            .map(|_| ())
            .map_err(lucene_store::blob_directory::into_io)
    }
}

impl<D: Directory> Directory for EncryptedDirectory<D> {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }

    fn open(&self, name: &str) -> Result<Input> {
        if self.is_plain(name) {
            return self.inner.open(name);
        }
        let sealed_len = self.inner.file_length(name)?;
        if let Some(shared) = self.cache.get(name, sealed_len) {
            return Input::slice_of(&shared, 0, shared.len());
        }
        let file = self.inner.open(name)?;
        let (cipher, header, header_len) = self.cipher_for(name, &file)?;
        self.header_lens()
            .insert(name.to_string(), (header_len as u64, header.chunk_size));
        let plain = self.decrypt(file, cipher, &header, header_len, name)?;
        let shared = Arc::new(plain);
        self.cache.put(name, sealed_len, &shared);
        Input::slice_of(&shared, 0, shared.len())
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        if self.is_plain(name) {
            return self.inner.file_length(name);
        }
        let known = self.header_lens().get(name).copied();
        let (header_len, chunk_size) = match known {
            Some(k) => k,
            None => {
                let file = self.inner.open(name)?;
                let (header, len) = Header::decode(&file, name)?;
                let entry = (len as u64, header.chunk_size);
                self.header_lens().insert(name.to_string(), entry);
                entry
            }
        };
        let total = self.inner.file_length(name)?;
        total
            .checked_sub(header_len)
            .and_then(|body| plaintext_len(body, chunk_size))
            .ok_or_else(|| {
                Error::Corrupted(format!(
                    "encrypted file {name}: length {total} is malformed"
                ))
            })
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.wrap(self.inner.create_output(name)?)
    }

    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.wrap(
            self.inner
                .create_output_with_estimate(name, estimated_bytes)?,
        )
    }

    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.wrap(self.inner.create_temp_output(prefix, suffix)?)
    }

    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.wrap(
            self.inner
                .create_temp_output_with_estimate(prefix, suffix, estimated_bytes)?,
        )
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        self.inner.sync(names)
    }

    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        if self.is_plain(source) != self.is_plain(dest) {
            return Err(Error::IllegalArgument(format!(
                "cannot rename {source} to {dest}: one is stored encrypted, the other in plaintext"
            )));
        }
        self.inner.rename(source, dest)?;
        self.cache.forget(source);
        self.cache.forget(dest);
        let mut lens = self.header_lens();
        lens.remove(dest);
        if let Some(entry) = lens.remove(source) {
            lens.insert(dest.to_string(), entry);
        }
        Ok(())
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.header_lens().remove(name);
        self.cache.forget(name);
        self.inner.delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.inner.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.inner.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        self.inner.fs_directory_path()
    }
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::key_provider::StaticKeyProvider;
    use lucene_store::ByteBuffersDirectory;

    fn keys() -> Arc<dyn KeyProvider> {
        Arc::new(StaticKeyProvider::new("k1", &[1; 32]))
    }

    fn write(dir: &dyn Directory, name: &str, bytes: &[u8], piece: usize) -> u64 {
        let mut out = dir.create_output(name).unwrap();
        for p in bytes.chunks(piece.max(1)) {
            out.write_bytes(p);
        }
        if bytes.is_empty() {
            out.write_bytes(&[]);
        }
        assert_eq!(out.file_pointer(), bytes.len() as u64);
        let crc = out.checksum();
        assert_eq!(out.close().unwrap(), crc);
        crc
    }

    #[test]
    fn the_lock_file_is_stored_as_is() {
        // The lock factory creates `write.lock` empty in the inner directory;
        // sizing the store lists it with every other file.
        let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
        dir.inner()
            .create_output(WRITE_LOCK_NAME)
            .unwrap()
            .close()
            .unwrap();
        assert_eq!(dir.file_length(WRITE_LOCK_NAME).unwrap(), 0);
        assert!(dir.open(WRITE_LOCK_NAME).unwrap().is_empty());
        for name in dir.list_all().unwrap() {
            dir.file_length(&name).unwrap();
        }
    }

    #[test]
    fn files_round_trip_at_every_size_and_write_pattern() {
        let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys())
            .with_chunk_size(16)
            .unwrap();
        assert!(format!("{dir:?}").contains("chunk_size=16"));
        for len in [0usize, 1, 15, 16, 17, 32, 33, 100, 20_000] {
            for piece in [1usize, 7, 16, 64, 100_000] {
                let name = format!("f{len}_{piece}");
                let plain: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
                let crc = write(&dir, &name, &plain, piece);
                // The checksum is the plaintext's, as Lucene's footers expect.
                assert_eq!(crc, crc32fast_hash(&plain));
                assert_eq!(&*dir.open(&name).unwrap(), &plain[..], "{name}");
                assert_eq!(dir.file_length(&name).unwrap(), len as u64, "{name}");
                let raw = dir.inner().open(&name).unwrap();
                assert!(raw.len() > len, "ciphertext carries a header and tags");
                if len >= 16 {
                    assert!(!contains(&raw, &plain[..16]), "{name}: plaintext on disk");
                }
            }
        }
    }

    /// The CRC32 a plaintext output of `bytes` reports.
    fn crc32fast_hash(bytes: &[u8]) -> u64 {
        let mut out = ByteBuffersDirectory::new().create_output("x").unwrap();
        out.write_bytes(bytes);
        out.checksum()
    }

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn rename_delete_and_temp_outputs_keep_lengths_right() {
        let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
        write(&dir, "pending_segments_1", b"commit bytes", 3);
        dir.sync(&["pending_segments_1".into()]).unwrap();
        dir.rename("pending_segments_1", "segments_1").unwrap();
        dir.sync_meta_data().unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["segments_1".to_string()]);
        assert_eq!(dir.file_length("segments_1").unwrap(), 12);
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit bytes");
        // A fresh wrapper over the same files learns lengths from headers.
        let again = EncryptedDirectory::new(dir.inner, keys());
        assert_eq!(again.file_length("segments_1").unwrap(), 12);
        again.delete_file("segments_1").unwrap();
        assert!(again
            .file_length("segments_1")
            .unwrap_err()
            .is_no_such_file());
        let mut t = again.create_temp_output("sort", "tmp").unwrap();
        t.write_bytes(b"abc");
        let tname = t.name().to_string();
        t.close().unwrap();
        assert_eq!(&*again.open(&tname).unwrap(), b"abc");
        let mut t = again
            .create_temp_output_with_estimate("sort", "tmp", Some(3))
            .unwrap();
        t.write_bytes(b"de");
        let tname = t.name().to_string();
        t.close().unwrap();
        assert_eq!(again.file_length(&tname).unwrap(), 2);
        let mut o = again.create_output_with_estimate("e", Some(1)).unwrap();
        o.write_byte(4);
        o.close().unwrap();
        assert_eq!(&*again.open("e").unwrap(), &[4]);
        assert!(again.pending_deletions().unwrap().is_empty());
        assert!(again.fs_directory_path().is_none());
        let lock = again.obtain_lock("write.lock").unwrap();
        assert!(again.obtain_lock("write.lock").is_err());
        drop(lock);
    }

    #[test]
    fn plaintext_commit_metadata_keeps_only_commit_files_readable() {
        for name in [
            "segments_3",
            "pending_segments_4",
            "_0.si",
            "corrupted_x",
            "recovery.abc.segments_3",
            "recovery.abc._1.si",
        ] {
            assert!(is_commit_metadata(name), "{name}");
        }
        for name in [
            "_0.cfs",
            "_0_Lucene104_0.tim",
            "_0.fdt",
            "write.lock",
            "x_sort_1.tmp",
        ] {
            assert!(!is_commit_metadata(name), "{name}");
        }
        let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys())
            .with_plaintext_commit_metadata();
        write(&dir, "pending_segments_1", b"commit bytes", 5);
        write(&dir, "_0.si", b"segment info", 4);
        write(&dir, "_0.tim", b"secret terms", 4);
        let mut t = dir.create_temp_output("_0", "si").unwrap();
        t.write_bytes(b"tmp");
        let tname = t.name().to_string();
        t.close().unwrap();
        // A temp name ends in `.tmp`, so it is encrypted whatever its suffix.
        assert_eq!(&dir.inner().open(&tname).unwrap()[..4], b"LRSE");
        assert_eq!(&*dir.open(&tname).unwrap(), b"tmp");
        dir.rename("pending_segments_1", "segments_1").unwrap();
        assert_eq!(&*dir.inner().open("segments_1").unwrap(), b"commit bytes");
        assert_eq!(&*dir.inner().open("_0.si").unwrap(), b"segment info");
        assert_eq!(&dir.inner().open("_0.tim").unwrap()[..4], b"LRSE");
        assert_eq!(&*dir.open("segments_1").unwrap(), b"commit bytes");
        assert_eq!(dir.file_length("_0.si").unwrap(), 12);
        assert_eq!(dir.file_length("_0.tim").unwrap(), 12);
        assert_eq!(&*dir.open("_0.tim").unwrap(), b"secret terms");
        let err = dir.rename("_0.tim", "segments_9").unwrap_err();
        assert!(err.to_string().contains("plaintext"), "{err}");
        // Off by default: the same names are encrypted.
        let all = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
        write(&all, "segments_1", b"commit bytes", 5);
        assert_eq!(&all.inner().open("segments_1").unwrap()[..4], b"LRSE");
        all.rename("segments_1", "x").unwrap();
    }

    /// Two directories sharing a cache share one plaintext while an input
    /// is alive; a delete, a rename or a rewrite of the name is never served
    /// stale bytes, and nothing is kept once every input is gone.
    #[test]
    fn the_plaintext_cache_shares_live_inputs_and_forgets_changed_names() {
        let inner = Arc::new(ByteBuffersDirectory::new());
        let cache = Arc::new(PlaintextCache::new());
        let a = EncryptedDirectory::new(Arc::clone(&inner), keys())
            .with_plaintext_cache(Arc::clone(&cache));
        let b = EncryptedDirectory::new(Arc::clone(&inner), keys())
            .with_plaintext_cache(Arc::clone(&cache));
        write(&a, "f", b"first contents", 64);
        let one = a.open("f").unwrap();
        let two = b.open("f").unwrap();
        assert_eq!(
            one.as_ptr(),
            two.as_ptr(),
            "the second open shares the first"
        );
        drop((one, two));
        assert!(cache.get("f", inner.file_length("f").unwrap()).is_none());

        let held = a.open("f").unwrap();
        b.delete_file("f").unwrap();
        write(&b, "f", b"second, longer contents", 64);
        assert_eq!(&*a.open("f").unwrap(), b"second, longer contents");
        assert_eq!(&*held, b"first contents", "an open input keeps its bytes");

        let g = a.open("f").unwrap();
        a.rename("f", "g").unwrap();
        assert_eq!(&*b.open("g").unwrap(), b"second, longer contents");
        assert!(b.open("f").is_err());
        drop(g);

        // Replaced behind the directories' back, with a different length.
        let _keep = a.open("g").unwrap();
        let other = EncryptedDirectory::new(Arc::clone(&inner), keys());
        other.delete_file("g").unwrap();
        write(&other, "g", b"replaced", 64);
        assert_eq!(&*a.open("g").unwrap(), b"replaced");
    }

    #[test]
    fn copy_from_reencrypts_and_a_plaintext_source_reads_back() {
        let plain = ByteBuffersDirectory::new();
        write(&plain, "a", b"hello", 5);
        let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
        dir.copy_from(&plain, "a", "b").unwrap();
        assert_eq!(&*dir.open("b").unwrap(), b"hello");
        // A plaintext file under the wrapper is refused, not misread.
        write(dir.inner(), "plain", b"not encrypted", 4);
        let err = dir.open("plain").unwrap_err().to_string();
        assert!(err.contains("plaintext index"), "{err}");
        assert!(dir.file_length("plain").is_err());
    }

    #[test]
    fn the_wrong_key_or_a_tampered_file_fails_and_rotation_keeps_old_files() {
        let inner = Arc::new(ByteBuffersDirectory::new());
        let k1 = Arc::new(StaticKeyProvider::new("k1", &[1; 32]));
        let dir = EncryptedDirectory::new(Arc::clone(&inner), k1);
        write(&dir, "old", b"under k1", 8);

        let rotated = Arc::new(
            StaticKeyProvider::new("k1", &[1; 32])
                .with_key("k2", &[2; 32])
                .with_current("k2"),
        );
        let dir2 = EncryptedDirectory::new(Arc::clone(&inner), rotated);
        write(&dir2, "new", b"under k2", 8);
        assert_eq!(&*dir2.open("old").unwrap(), b"under k1");
        assert_eq!(&*dir2.open("new").unwrap(), b"under k2");
        let raw = inner.open("new").unwrap();
        assert!(contains(&raw, b"k2"), "the header names the key");

        // A provider without k2 cannot read the new file.
        let err = dir.open("new").unwrap_err();
        assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied));
        // A provider with a different k1 cannot unwrap the old one.
        let imposter = EncryptedDirectory::new(
            Arc::clone(&inner),
            Arc::new(StaticKeyProvider::new("k1", &[3; 32])),
        );
        assert!(matches!(imposter.open("old"), Err(Error::Corrupted(_))));

        // Flip one ciphertext byte: authentication fails.
        let mut bytes = raw.to_vec();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        let mut out = inner.create_output("tampered").unwrap();
        out.write_bytes(&bytes);
        out.close().unwrap();
        let err = dir2.open("tampered").unwrap_err().to_string();
        assert!(err.contains("failed authentication"), "{err}");
        // Truncate it: its length no longer parses either.
        let mut out = inner.create_output("short").unwrap();
        out.write_bytes(&raw[..raw.len() - 17]);
        out.close().unwrap();
        assert!(dir2.open("short").is_err());
    }

    #[test]
    fn chunk_size_is_checked() {
        let d = || EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
        assert!(d().with_chunk_size(0).is_err());
        assert!(d().with_chunk_size(MAX_CHUNK_SIZE + 1).is_err());
        assert!(d().with_chunk_size(MAX_CHUNK_SIZE).is_ok());
    }

    #[test]
    fn a_key_provider_failure_fails_create_output() {
        let dir = EncryptedDirectory::new(
            ByteBuffersDirectory::new(),
            Arc::new(StaticKeyProvider::new("k", &[0; 32]).with_current("missing")),
        );
        assert!(dir.create_output("x").is_err());
    }
}
