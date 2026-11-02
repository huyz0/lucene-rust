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

## Limits to know

- An input is a whole contiguous file. `MmapDirectory` maps it, but an
  encrypted or remote file is read whole into memory when opened, so an
  encrypted index costs its size in RAM while a reader holds it. Paged inputs
  would lift this; they touch every codec reader and are not done.
- `BlobDirectory`'s default lock only excludes writers in one process. A store
  shared by several writer processes needs its own `LockFactory`.
- File names are not encrypted, and file sizes are visible (plus a small
  overhead).
- The OpenSearch plugin does not use this layer yet: the Rust engine opens
  the shard path with `MmapDirectory`/`FsDirectory` directly.
