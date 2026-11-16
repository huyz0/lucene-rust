//! [`EncryptedDirectory`]: a [`Directory`] wrapper that encrypts every file
//! it writes and decrypts every file it opens (application-level
//! encryption), over any directory underneath -- a local filesystem, a
//! [`lucene_store::BlobDirectory`], anything else that implements the trait.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
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
/// Over a directory that reads pages itself ([`lucene_store::PagedFsDirectory`]
/// on local disk, [`lucene_store::CachedBlobDirectory`] over a blob store),
/// a file is opened through [`Directory::open_decoded`]: only its header is
/// read at open, and each page of whole chunks is authenticated and
/// decrypted once, as it is read into that directory's page cache. A chunk
/// that fails authentication then fails the read of it with an error. Over
/// any other directory an [`Input`] is a contiguous byte slice, so a file is
/// authenticated and decrypted whole when it is opened and costs its size in
/// memory while a reader holds it open (AES-GCM with AES-NI decrypts at
/// several GB/s per core).
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
    cache: Arc<PlaintextCache>,
}

/// The index lock file's name (Lucene's `IndexWriter.WRITE_LOCK_NAME`).
const WRITE_LOCK_NAME: &str = "write.lock";

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
            cache: Arc::new(PlaintextCache::new()),
        }
    }

    /// Shares opened plaintext with every directory given the same cache;
    /// they must be over the same files.
    pub fn with_plaintext_cache(mut self, cache: Arc<PlaintextCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Stores commit metadata ([`is_commit_metadata`]) in plaintext and
    /// encrypts everything else. Off by default; the OpenSearch plugin's
    /// `encrypted` store turns it on, on both its Java and Rust sides,
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
        let decoder = crate::pages::EncryptedDecoder {
            keys: self.keys.as_ref(),
            header_lens: &self.header_lens,
        };
        let plain = match self.inner.open_decoded(name, &decoder) {
            Some(opened) => opened?,
            None => {
                let file = self.inner.open(name)?;
                let (cipher, header, header_len) = self.cipher_for(name, &file)?;
                self.header_lens()
                    .insert(name.to_string(), (header_len as u64, header.chunk_size));
                Input::Owned(cipher.open_body(&file[header_len..], header.chunk_size, name)?)
            }
        };
        let shared = Arc::new(plain);
        self.cache.put(name, sealed_len, &shared);
        Input::slice_of(&shared, 0, shared.len())
    }

    /// Plain files are hinted as they are; encrypted ones with the decoder
    /// their opens use, so their headers and first and last decoded pages
    /// come in the batch.
    fn prefetch_files(&self, names: &[String]) -> Result<()> {
        let (plain, sealed): (Vec<String>, Vec<String>) =
            names.iter().cloned().partition(|n| self.is_plain(n));
        let decoder = crate::pages::EncryptedDecoder {
            keys: self.keys.as_ref(),
            header_lens: &self.header_lens,
        };
        let plain_result = self.inner.prefetch_files(&plain);
        self.inner.prefetch_files_decoded(&sealed, &decoder)?;
        plain_result
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        if self.is_plain(name) {
            return self.inner.file_length(name);
        }
        let known = self.header_lens().get(name).copied();
        let (header_len, chunk_size) = match known {
            Some(k) => k,
            None => {
                let (header, bytes) = crate::pages::read_header(&self.inner, name)?;
                let entry = (bytes.len() as u64, header.chunk_size);
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
mod tests;
