# Pluggable storage and encryption

Lucene's `Directory` is the seam between the index and wherever its bytes
live. This port keeps that seam open: an index can live on a local disk, in
an object store, behind an encrypting wrapper, or behind anything else that
implements `lucene_store::Directory`. The codecs, `IndexWriter`,
`DirectoryReader` and `CheckIndex` never know which.

## The extension points

| You want | Implement or use | Where |
|---|---|---|
| A new backend (S3, GCS, Azure Blob, a KV store) | `BlobStore` (list, get, put, delete; optional size, copy, streaming writer, sync), wrapped in `BlobDirectory` | `lucene-store/src/blob_directory.rs` |
| Full control over a backend | `Directory` directly; hand out outputs with `FsIndexOutput::from_sink(name, Box<dyn OutputSink>)` and inputs with `Input::external(bytes)` | `lucene-store/src/directory.rs`, `index_output.rs` |
| A wrapper (Lucene's `FilterDirectory`): metering, caching, encryption | a type holding the inner directory that implements `Directory`, forwarding what it does not change | `lucene-store-crypto/src/directory.rs` is a worked example |
| Encryption at rest (ALE) | `lucene_store_crypto::EncryptedDirectory::new(inner, keys)` | `lucene-store-crypto` |
| Customer-managed keys (BYOK) | `lucene_store_crypto::KeyProvider` over your KMS (`wrap_key` or `generate_data_key`, plus `unwrap_key`), usually behind `CachingKeyProvider` | `lucene-store-crypto/src/key_provider.rs` |
| Pick a store by name from configuration | `register_store_type` / `register_store_wrapper`, then `StoreSpec::of("mmapfs").wrapped("encrypted").with(k, v).open(path)` | `lucene-store/src/store_registry.rs` |

Built-in store types use OpenSearch's names: `fs`, `hybridfs` and `mmapfs`
open an `MmapDirectory`; `niofs` and `simplefs` an `FsDirectory`.
`lucene_store_crypto::register()` adds the `encrypted` wrapper, which reads
`encryption.key_provider` (a name given to `register_key_provider`) and
optionally `encryption.chunk_size`.

## Reading an object store by page

`CachedBlobDirectory` reads the same `BlobStore` by range
(`BlobStore::get_range`) through a `PageCache`, so a node holds the pages its
queries touch, not whole files. The backend supplies only the `BlobStore`
(list, whole and ranged reads, uploads, deletes); the cache is built in.

```rust
use std::sync::Arc;
use lucene_store::{CachedBlobDirectory, PageCache, PageCacheConfig};

let cache = Arc::new(PageCache::new(PageCacheConfig {
    disk_dir: Some("/nvme/lucene-cache".into()),   // emptied on start
    disk_budget: 200 << 30,
    ..PageCacheConfig::default()                    // 1 MiB pages, 8 MiB requests
})?);
let dir = CachedBlobDirectory::new(MyS3Store::new("bucket", "idx/"), cache, "bucket/idx");
```

- Pages are fixed and whole: a read fetches every page it touches, adjacent
  misses merged into one ranged read; the cache never keeps part of a page.
  Concurrent misses on a page send one request.
- Files read whole at open or on every query (metadata, terms index, norms,
  the stored-fields and points indexes, live docs) are pinned on disk and
  held in memory. Other files are mapped and filled on first touch (Linux
  `userfaultfd`), within a process-wide memory budget
  (`lucene_store::paged_input::Pager::set_budget`); elsewhere they are read
  whole.
- `Input::will_need(offset, len)` (Lucene's `IndexInput.prefetch`) fetches a
  range ahead of the reads and returns the fetch error. A fault that still
  cannot be filled after retries aborts the process, since a memory read
  cannot fail.
- With `EncryptedDirectory` on top, pages are whole 64 KiB chunks after the
  header: the disk tier holds ciphertext, and a page is authenticated and
  decrypted once each time it is brought into memory. Only the header is
  read when a file opens. Encryption stays opt-in; without it, pages are
  copied through with no crypto.

## Example: BYOK on an object store

```rust
use std::{sync::Arc, time::Duration};
use lucene_store::BlobDirectory;
use lucene_store_crypto::{CachingKeyProvider, EncryptedDirectory};

let kms = MyKms::new("arn:aws:kms:...:key/tenant-a");      // impl KeyProvider
let keys = Arc::new(CachingKeyProvider::new(kms, Duration::from_secs(300), 10_000));
let dir = EncryptedDirectory::new(BlobDirectory::new(MyS3Store::new("bucket", "idx/")), keys);
let writer = IndexWriter::open(&dir, fields, "Lucene104", version)?;
```

`crates/lucene-store-crypto/tests/index_round_trip.rs` runs a full index
(flushes, commits, merges, deletes, `CheckIndex`, a search, stored fields)
through an encrypted filesystem directory, a blob store and an encrypted
blob store.

## Encrypted file format

Each file gets its own random 256-bit data key, stored only wrapped by the
key provider's current key (envelope encryption), then the plaintext in
64 KiB chunks, each sealed with AES-256-GCM under a nonce the crypto module
draws (FIPS-approved IV generation) and stored before the chunk. The header
(key id, wrapped key, chunk size), the chunk's index and a final-chunk flag
are every chunk's associated data, so
header edits, reordered, dropped or truncated chunks fail authentication.
Rotation: change the provider's current key; old files keep naming the key
that wrapped them, and merges rewrite them under the new one over time.
Revoking a key in the KMS makes the files it wrapped unreadable (after
`CachingKeyProvider`'s TTL). Details: `lucene-store-crypto/src/format.rs`.

## FIPS

All crypto goes through AWS-LC via `aws-lc-rs`
(`lucene-store-crypto/src/backend.rs`), using only FIPS-approved services
(`RandomizedNonceKey` seal, `open_in_place`):

- default (feature `aws-lc`): the regular AWS-LC build;
- `--no-default-features --features fips`: AES-256-GCM and the DRBG from
  AWS-LC's FIPS 140-3 validated module (`aws-lc-fips-sys`), with no other
  AES implementation linked. `lucene_store_crypto::is_fips()`
  reports whether the module is in FIPS mode; the FIPS build's tests assert it.

`scripts/build-fips.sh [--test]` is the release recipe (needs CMake, Go, Perl,
a C compiler) and CI's `fips` job runs it. Both builds write the same format
(a NIST GCM vector pins the cipher), so an index moves between them freely.
Compliance also depends on the deployed module version and platform: check
AWS-LC's current CMVP certificate for the resolved `aws-lc-fips-sys`. A
key provider over a KMS must itself use FIPS endpoints for an end-to-end
FIPS deployment.

## In OpenSearch: the `rust_encrypted` store type

The plugin adds the store type `rust_encrypted`. Both Lucene (the JVM) and
the Rust engine read and write a shard's files in the format above, with the
same keys.

```yaml
index.store.type: rust_encrypted
index.store.rust_encrypted.key_provider_type: aws-kms     # a CryptoKeyProviderPlugin type
index.store.rust_encrypted.key_provider: my-kms           # its name
index.store.rust_encrypted.kms.key_arn: arn:aws:kms:...   # what that provider reads
index.store.rust_encrypted.kms.encryption_context: tenant=a
# optional
index.store.rust_encrypted.chunk_size: 64kb
index.store.rust_encrypted.data_key_ttl: 1h     # one KMS GenerateDataKey per period, not per file
index.store.rust_encrypted.key_cache_ttl: 5m    # how long a revoked key keeps working
```

- **Keys:** keys come from OpenSearch's `MasterKeyProvider`, through any
  installed `CryptoKeyProviderPlugin` (`crypto-kms` for AWS KMS). The
  key-encryption key never leaves the KMS.
  - The provider type `lucene_rust_local` instead wraps data keys with a
    key in the node keystore (`lucene_rust.store.local_key`, base64 of 32
    bytes; id `lucene_rust.store.local_key_id`). It is meant for tests and
    for clusters without a KMS.
- **How the engine gets keys:** when the plugin creates the directory, it
  binds the shard path in the native library to the same key source
  (`ffi_store_bind_encrypted`). The Rust writer and readers call back into
  the JVM for keys through one FFM upcall, and cache what they unwrap.
- **Commit metadata stays plaintext:** `segments_N`, `.si` files and
  OpenSearch's `corrupted_*` markers are not encrypted
  (`EncryptedDirectory::with_plaintext_commit_metadata`, on in both the Java
  and Rust sides). OpenSearch reads them through a plain `NIOFSDirectory`,
  not the store's directory, when it allocates a primary after a restart
  (`Store.tryOpenIndex`); encrypted, a shard would never come back. They
  hold segment and file names, document counts, codec names and commit user
  data (sequence numbers, translog and history UUIDs), never field contents.
  The lock file `write.lock` is empty and stored as-is too, on both sides.
- **Store sizing:** OpenSearch sizes a store from every file it lists
  (`Store.stats`, also while it allocates another shard of the same index),
  including files a writer still has open. The Java directory's
  `fileLength` counts the complete chunks of a file that ends inside its
  header or a chunk instead of failing; opening such a file still fails.
- **No silent plaintext:**
  - The Rust engine refuses an index whose `index.store.type` it cannot
    write through (another plugin's encryption, say). Use
    `rust_encrypted` for encryption at rest.
  - The native read path declines any store directory that changes bytes on
    disk.
- **FIPS:** the plugin built with `gradle -Pfips bundlePlugin` produces
  `lucene-rust-fips-*.zip`, which contains:
  - the native library built against AWS-LC's FIPS module
    (`scripts/build-fips.sh --ffi`);
  - the Amazon Corretto Crypto Provider's FIPS build, the same AWS-LC
    module, which does the JVM side's AES-GCM and random numbers. SunJCE is
    not FIPS validated and is never used.

  At startup the node checks that both are in FIPS mode, and refuses to
  start otherwise. The default zip uses the JDK's own providers.

## Demand decryption

An input is a whole contiguous file, which is what lets codecs read a
memory-mapped file at memory speed. By default an encrypted file is
therefore decrypted whole when it opens and costs its size in RAM while a
reader holds it.

With demand decryption on (`EncryptedDirectory::with_lazy_decrypt`, the
store wrapper setting `encryption.lazy_decrypt`, or the plugin's node
setting `lucene_rust.store.encrypted.lazy_decrypt`), an encrypted file is
instead authenticated in full when it opens, then mapped and decrypted
chunk by chunk on first touch (`lucene-store-crypto/src/lazy.rs`). Decrypted
chunks stay resident up to a process-wide budget (default 512 MiB; the
plugin's `lucene_rust.store.encrypted.decrypt_cache_size`); past it the
oldest are dropped, and the next touch decrypts that chunk again from the
ciphertext in the page cache. Plaintext is never written to disk.

Opened plaintext is shared through a `PlaintextCache`: while any input of a
file is alive, opening it again, from the same directory or from another
one given the same cache, costs a reference count. In the plugin the
writer's and the readers' directories of one shard share one cache.

On a 1M-document encrypted index (143 MB; 4 cores; the
`lucene-store-crypto` example `encrypted_memory`), with a reader open and
term queries run:

| | whole-file | demand |
|---|---|---|
| anonymous memory | 138 MB | 25 MB |
| reader open | ~50 ms | ~75 ms |
| first 16 term queries | 4.4 ms | 8 to 9 ms |
| warm, best of 30 passes | 2.8 to 3.1 ms | 2.8 to 3.1 ms |

The budget should cover the hot working set: at 8 MiB against a 25 MB
working set the same queries take ~35 ms.

What it needs and how it fails:

- Linux with `userfaultfd` permitted. Docker's and containerd's default
  seccomp profiles (and Kubernetes `RuntimeDefault`) block it; a container
  needs a profile that allows `userfaultfd`. Where it is unavailable, or a
  file's chunk size is not a whole number of pages, files are decrypted
  whole and the plugin logs a warning at startup.
- A process without `CAP_SYS_PTRACE` (and with
  `vm.unprivileged_userfaultfd = 0`) gets user-mode-only faults: the kernel
  cannot fault a page in, so a system call reading straight from a page
  that is not resident fails with `EFAULT`. `FsIndexOutput` copies such
  bytes through a buffer and writes them again; other code handing input
  bytes to the kernel must do the same.
- A chunk that fails authentication after the file opened means the
  ciphertext changed under an open reader. A memory access cannot return an
  error, so the process aborts rather than hand out unauthenticated bytes.
- One handler thread serves every fault, so cold reads are serialized
  behind it.

## Limits to know

- Without demand decryption, an encrypted or remote file is read whole into
  memory when opened (see above).
- `BlobDirectory`'s default lock only excludes writers in one process. A store
  shared by several writer processes needs its own `LockFactory`.
- File names are not encrypted, and file sizes are visible (plus a small
  overhead). In OpenSearch, commit metadata is plaintext too (see above).
- In OpenSearch, a replica's store metadata (`Store.readMetadataSnapshot`)
  is also read through a plain directory, which cannot read an encrypted
  file's footer. A node therefore never reuses a stale local copy of an
  encrypted replica: peer recovery copies its files again.
- In OpenSearch, Lucene's side reads an encrypted file a chunk at a time,
  decrypting the chunk under the read position. The Rust side decrypts
  whole files when it opens them, or on demand as above.
