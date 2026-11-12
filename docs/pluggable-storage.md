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
    ..PageCacheConfig::default()                    // 256 KiB pages, auto window
})?);
let dir = CachedBlobDirectory::new(MyS3Store::new("bucket", "idx/"), cache, "bucket/idx");
```

- Pages are fixed and whole (256 KiB by default): a read fetches every page
  it touches, adjacent misses merged into one ranged read; the cache never
  keeps part of a page. Concurrent misses on a page send one request.
- A miss also fetches the other missing pages of its aligned download
  window, each kept and evicted as its own page. The window is sized from
  measured latency times throughput (a power of two between one page and
  `max_request`), or fixed with `PageCacheConfig::download_window`.
- The disk tier keeps pages in slots of a few large slab files, evicted by
  CLOCK. Slots come in size classes from 4 KiB doubling up to a full page,
  so a small file or a file's short last page takes the smallest class that
  holds it; the budget counts the slots in use, and an evicted slot's space
  is given back to the file system (hole punching on Linux). It holds about
  24 bytes of RAM per page (2 TB of 256 KiB pages: about 200 MB).
  `PageCacheStats` reports it, with `disk_footprint_bytes`.
- File lengths come from the store's listing when it carries them
  (`BlobStore::list_with_sizes`, as S3's `ListObjectsV2` does), so opening a
  file sends no length request. Lucene writes each file once, so a length
  is kept until the file is deleted, written or renamed.
- Files read whole at open or on every query (metadata, terms index, norms,
  the stored-fields and points indexes, live docs) are pinned on disk and
  held in memory. Other files are mapped and filled on first touch (Linux
  `userfaultfd`), within a process-wide memory budget
  (`lucene_store::paged_input::Pager::set_budget`); elsewhere they are read
  whole.
- Readers announce their reads, as Lucene's `IndexInput.prefetch` does, so
  a cold query waits for few round trips: a segment open fetches every file
  it reads at once (`Directory::prefetch_files`), a search the terms
  dictionary blocks of all its terms, then the start of their postings, and
  `SegmentReader::prefetch_stored_documents` the stored-fields chunks of a
  page of hits. `lucene_store::prefetch(&[&[u8]])` is the hint by address
  for readers that hold byte slices, `Input::will_need` by offset; both
  return the fetch error. A fault that still cannot be filled after retries
  aborts the process, since a memory read cannot fail.
- With `EncryptedDirectory` on top, pages are whole 64 KiB chunks after the
  header: the disk tier holds ciphertext, and a page is authenticated and
  decrypted once each time it is brought into memory. Only the header is
  read when a file opens, once, and a segment open fetches every encrypted
  file's header and decoded first and last pages together
  (`Directory::prefetch_files_decoded`). Encryption stays opt-in; without
  it, pages are copied through with no crypto.
- In OpenSearch, the fetch phase reads its hits one at a time. The plugin
  announces them first (`SearchOperationListener.onPreFetchPhase`), so a
  segment read by range fetches the stored fields of the page of hits
  together, as `prefetch_stored_documents` does.

`cargo run --release -p lucene-search --example cold_range_reads` counts the
requests and round trips of cold queries over a store with injected latency.
At 30 ms and 100 MB/s per request, over one 1M-document segment, elapsed
time in round trips (transfer included, length requests and listings
costed as one round trip each): opening it about 4.5 in 19 requests (8.6 in
34 when the listing carries no lengths), a term or boolean query about 2.5, a phrase 3.5 to 5.5, and
fetching the top 10 hits' stored fields 2.5 to 5 (5 to 23 one hit at a
time).

## Reading local disk through the page cache

`PagedFsDirectory` reads an index on local disk (EBS, NVMe) the way
`CachedBlobDirectory` reads an object store: files open as anonymous
mappings filled page by page (Linux `userfaultfd`), within the pager's
process-wide budget, instead of `mmap`. The files are `MmapDirectory`'s, so
either directory opens the same index.

```rust
use lucene_store::{PagedFsConfig, PagedFsDirectory};

lucene_store::paged_input::Pager::global().unwrap().set_budget(2 << 30);
let dir = PagedFsDirectory::open("/data/idx");                  // every file
let dir = PagedFsDirectory::with_config("/data/idx", PagedFsConfig::vectors_only());
```

- Reads use `O_DIRECT` where the file system takes it, so a page is held
  once, in the budget, not again in the kernel's page cache. A read goes
  into a reused per-thread buffer and from there straight into the mapping
  (`UFFDIO_COPY`); each fault wakes one handler thread (`EPOLLEXCLUSIVE`),
  of four per CPU (16 to 64), so faults wait on the disk in parallel.
- Every open of one file shares its pages: a merge reading a segment the
  search reader has open reads the pages the reader holds, and no page is
  held twice. Files read whole at open are shared the same way.
- Pages are 64 KiB. A file larger than half the budget gets 4 KiB pages
  (vector data) or 16 KiB pages (the rest), since it cannot stay resident
  and a large page would bring in bytes nobody reads
  (`PagedFsConfig::page_size_for`). A format whose blocks start on fixed
  boundaries opens its files with `Directory::open_aligned`; their pages
  are then the block size times a power of two (`lucene_store::alignment`,
  base 4 KiB).
- Metadata, the terms index, norms, the stored-fields and points indexes and
  live docs are read whole at open, outside the budget, as on an object
  store. So are the HNSW graph (`.vex`), quantized vectors (`.veq`) and IVF
  codes (`.ivp`) when each is at most an eighth of the budget
  (`PagedFsConfig::pins`).
- Faults that run in sequence read ahead, doubling up to 1 MiB. The window
  after the faulting page is read on an I/O thread while the faulting page
  is read alone, so the reader runs on as the rest arrives. A batch of
  scattered pages (a prefetch) is read on the calling thread and 16 I/O
  threads at once. HNSW asks for each expanded node's neighbours' vectors in
  one batch, disk HNSW for each beam's records, IVF for its probed clusters
  and its rerank candidates. A file whose pages are all resident costs a
  prefetch one counter read.
- Past the budget, pages are released in segmented FIFO order. New pages
  start on probation. A page faulted in again within about a budget's worth
  of evictions (the kernel's refault distance), or announced again by a
  prefetch, is protected, and protected pages take at most three quarters of
  the budget, so a scan cannot push out the working set. A pass past the
  budget goes a little below it (1/256), and its victims are released in
  one `process_madvise` call where the kernel takes `MADV_DONTNEED` there
  (Linux 6.13+; 6.16+ flushes the TLB once per batch), else one `madvise`
  each.
- In-flight reads and pinned files are outside the budget: leave headroom
  under a hard memory limit (the bench needed about 15% at 256 MiB).
  `lucene_store::whole_file_bytes` reports the pinned files' bytes.
- `Pager::stats` counts faults, refaults, fault time, evictions and pages
  that could not be read.
- A page that cannot be read (an I/O error) is retried for about two
  seconds. A thread in a checked call (an FFI call that noted its shard,
  `read_failures::set_checked`) then reads it as zeros for one second
  (`UNREADABLE_HOLD`), and the failure is recorded under the file's path
  (`lucene-store/src/read_failures.rs`), so the calls that ran while it was
  served fail; after that the zeros are dropped and the next touch reads
  the disk again. Any other thread (a merge, a warmer), which nothing
  would fail, keeps retrying until the disk serves the page. A merge also
  fails if a failure under its directory overlapped it. Announced reads
  (`will_need`, `prefetch`) get the error directly. The process never
  aborts for a page.

In OpenSearch, `index.lucene_rust.store_mode` picks it per index: `mmap` (the
default), `page_cache`, or `page_cache_vectors` (vector files only, the rest
mapped). `lucene_rust.store.page_cache.size` (default 1 GB) is the node's
budget. A node where `userfaultfd` is blocked keeps every index mapped and
says so at startup. `GET /_plugins/lucene_rust/stats` reports the page
cache under `page_cache` (budget, resident and whole-file bytes, faults,
refaults, evictions, read errors, fault time), and the resident bytes are
charged to the `lucene_rust_writer` breaker, whose default limit includes
the budget, so they show in `_nodes/stats/breaker`.

### Against `mmap`

`lucene-search/examples/local_store.rs` (full-text, and `build-tenants`) and
`local_vectors.rs` (IVF, HNSW) measure both directories on one index, cold
(page cache dropped, first pass) and hot (last pass), each run in a memory
cgroup when a limit is given. Mean latency per query, one thread, 4 vCPUs,
virtio disk; raw results under `perf/paged-local-store/` in the project
files. Runs vary by about 10% from one to the next.

| workload | memory limit | `mmap` cold / hot (us) | page cache cold / hot (us) | memory held | read per pass |
|---|---|---|---|---|---|
| full-text, 3M docs, 1 GiB | none | 7106 / 5699 | 6514 / 5867 | 798 / 774 MiB | |
| full-text | 400 MiB | 6475 / 6422 | 6717 / 6586 | 400 / 377 MiB | 990 / 540 MiB |
| IVF, 2M x 128, 1.1 GiB | none | 376 / 64 | 348 / 65 | 958 / 927 MiB | |
| IVF | 256 MiB | 855 / 839 | 761 / 690 | 256 / 177 MiB | 1.9 / 0.5 GiB |
| HNSW, 500k x 128, 265 MiB | none | 224 / 151 | 302 / 152 | 278 / 285 MiB | |
| HNSW | 128 MiB | 35155 / 36705 | 11340 / 11164 | 128 / 101 MiB | 14.3 / 0.8 GiB |
| 2000 tenants, 3M docs, 90% of queries on 20 | 300 MiB | 220 / 101 | 287 / 96 | 300 / 205 MiB | |
| tenants | 150 MiB | 310 / 244 | 315 / 306 | 150 / 138 MiB | 310 / 170 MiB |

What it shows:

- When the index fits, hot queries cost the same, and the page cache holds
  no more than `mmap`'s page cache did, a third less with tenants, since it
  reads ahead only within runs of faults.
- When the index does not fit, vector search wins. HNSW's graph walk under
  `mmap` reads ahead around every random hop and re-reads the file over and
  over; with 4 KiB pages, the graph pinned and each node's neighbours read
  in one batch, it reads 18x less and runs 3.2x faster. IVF runs 10-18%
  faster than `mmap` in a third less memory and a quarter of the reads.
- A cold first touch costs 0-30% more than `mmap`, most with tenants:
  - Every miss hands work between threads (query thread, fault handler, I/O
    thread). On this VM a wake-up of another vCPU costs tens of
    microseconds, where the kernel resolves a missing page on the faulting
    thread itself. Removing the allocations, copies and herd wake-ups took
    a 4 KiB fault from 120 to about 60 us against 43 us for a plain `pread`
    and 95 us for an `mmap` fault (with its read-ahead).
  - Small pages mean more device reads: tenants at 300 MiB use 16 KiB pages
    and make 1.7x the requests `mmap` makes with its 128 KiB read-ahead, at
    the same cost per request. With 64 KiB pages the gap halves, but the
    memory saving goes. Reading 64-128 KiB around every miss, polling for
    faults before sleeping, and starting prefetches in the background were
    tried and did not help or made it slower.
- Full-text under pressure is 3-5% slower than `mmap` at 400 MiB, and
  tenants at 150 MiB about 25% slower hot. The kernel's LRU sees every
  access through the page tables' accessed bits. The pager sees only faults
  and prefetch hints, and Linux gives a process no way to read those bits
  for its own pages, so a page in use but never faulted again looks cold.
  Refault-distance protection keeps what is re-read soon after release,
  which is most of what can be recovered. Hence `page_cache_vectors`, which
  leaves full-text on `mmap`.
- With an index sorted by tenant, each tenant's postings and stored fields
  are one run inside each file, so a hot tenant's pages stay resident and a
  cold tenant's age out. A per-tenant prefetch hint was not tried: postings
  are cut by term, not tenant, and the readers already announce the blocks
  a query reads. IVF clusters group vectors by similarity, so a tenant's
  vectors spread over every cluster and sorting by tenant does not make
  them contiguous.

### Read advice for mapped files

A mapped file's page fault reads ahead of the faulting page by default (Linux
reads up to 128 KiB around it). For files read a few hundred bytes at a time
at scattered offsets, that wastes reads and memory once the files no longer
fit in memory, and on a volume capped in IOPS and bandwidth (EBS gp3: 3000
IOPS, 125 MB/s) the waste is what caps throughput. Lucene 9.11 to 10.2
advised the vector files, stored fields and term vectors `RANDOM`; 10.3 went
back to the OS default and made per-file advice opt-in
(`MMapDirectory.ADVISE_BY_CONTEXT`, `-Dorg.apache.lucene.store.defaultReadAdvice`).

The search reader advises each mapped file by extension
(`lucene_store::read_advice`). By default the files a vector search reads a
vector at a time are advised random: flat vectors (`vec`), the HNSW graph
(`vex`), IVF rerank vectors (`ivv`) and quantized codes (`veq`, `vqd`).
Stored fields (`fdt`) are advised `auto`: random when the file is larger
than an eighth of available memory, the OS default otherwise (see below).
Every other file keeps the OS default. Merges open their own mappings and
are never advised. A file advised random that is at most an eighth of
available memory (`MemAvailable`, capped by the memory cgroup's limit) is
also read in once at open, so a cold start still warms it in large reads;
exhaustive (filtered) scans read their vectors ahead in 1 MiB windows.

In OpenSearch, two node settings change it:

- `lucene_rust.store.mmap.read_advice`: a list of `ext:advice` entries
  (`normal`, `random`, `sequential`, `willneed`, `auto`; `*` for every
  extension not listed, as Lucene's `defaultReadAdvice`). A list replaces the whole table,
  so `["vec:random", "vex:random", "fdt:random"]` leaves `veq` at `normal`.
  `willneed` reads the whole file in at open, as OpenSearch's
  `index.store.preload` does.
- `lucene_rust.store.mmap.warm_random` (default `true`): the read-in at
  open of small random-advised files.

On local NVMe with free memory the advice changes little (hot latency was
within noise in every case below). It matters when vector files do not fit
in memory, most of all on IOPS-capped volumes. HNSW 500k x 128 (266 MB), 600
queries in a 160 MB memory cgroup, one thread; gp3 emulated with a blkio
throttle of 3000 read IOPS and 125 MB/s on this VM's disk (an emulation, not
EBS):

| | ms/query | device reads/query | MiB read/query | gp3 qps ceiling |
|---|---|---|---|---|
| `mmap`, OS default | 176 | 518 | 19 | ~6 |
| `mmap`, random | 66 | 247 | 1.0 | ~12 |
| page cache (pager) | 71 | 239 | 1.25 | ~12 |

Filtered to 2% (an exhaustive scan per segment) in the same setup: 3.5 s,
1.25 s and 1.32 s per query. The pager reads as `mmap` with random advice
does; it does not beat it.

### Stored fields

A fetch reads one chunk of the `.fdt` per document: about 60 KB for the
default codec (`best_speed`, 1 KB documents), more for `best_compression`,
the whole document for one larger than a chunk. Read through the OS
default, a fault reads 128 KiB around the faulting page, about twice what a
small-document chunk needs. Advised random alone, a fault reads one page:
a chunk arrives a few pages at a time and a 1 MB document 256 times, which
is 10x slower on gp3. So when the `.fdt` is advised random (the `auto`
default once the file is larger than an eighth of available memory, or an
explicit `fdt:random`), the stored-fields reader advises each chunk it is
about to read `MADV_WILLNEED`: the fetch phase's prefetch does it for all its
documents at once, and a read without a prefetch (a get by id) does it for
its own chunk. A chunk already in memory is skipped (one `mincore`; chunks
over 256 KiB are always advised). A file that fits keeps the OS default, so
an index in memory pays nothing; a mapped file is the only one advised, the
page cache and blob stores prefetch by range as before, and merges are never
advised.

`stored_fetch` example (`crates/lucene-search/examples/stored_fetch.rs`), 4
threads, documents drawn uniformly over the index, caches dropped before
each cold run. gp3 is emulated with a blkio throttle of 3000 read IOPS and
125 MB/s on this VM's disk (an emulation, not EBS). Fetches per second:

| index (`.fdt` size), fetch | memory | OS default | random | sequential | **auto (new default)** |
|---|---|---|---|---|---|
| 2M x 1 KB `best_speed` (1.3 GB), 10 docs | cached | 42-45k | 34-40k | 44k | 41-42k |
| | 4 GB, cold, gp3 | 27-29k | 100 | 27k | 24k |
| | 256 MB, gp3 | 110 | 97 | 130 | **268** |
| 2M x 1 KB `best_compression` (0.7 GB), 10 docs | cached | 3.8-3.9k | 3.9k | 3.8k | 3.9k |
| | 4 GB, cold, gp3 | 3.1-3.2k | 54 | 3.2k | 3.1k |
| | 256 MB, gp3 | 70 | 43 | 86 | **106** |
| 1500 x 1 MB (1 GB), 1 doc | cached | 4.8-5.6k | 5.1k | 5.6k | 5.2k |
| | 4 GB, cold, gp3 | 3.7-4.2k | 22 | 4.0k | 3.8k |
| | 256 MB, gp3 | 200 | 21 | 224 | **241** |

`random` is random advice without the chunk read-in. Starved, `auto` reads
the same bytes per second as the OS default but only the chunks asked for:
small documents 2.4x, `best_compression` 1.5x, 1 MB documents 1.2x the
fetches. The one case it costs is a file advised random that stays cached
(the 4 GB row for small documents, a file a third of memory): about 1 µs per
document for the `mincore`, 10% of this bench's bare fetch, a few percent of
an OpenSearch fetch phase. A force merge of four segments (1.3 GB) alone
takes the same time either way (22.0 s, 22.1 s, 512 MB, gp3); under a 4-thread
fetch load it took 70 s instead of 62 s while the fetches ran 2.4x faster
(235 against 96 per second), the two sharing the volume's budget.

Quantized HNSW (`Lucene104` 7-bit codes, `.veq` 72 MB, graph 21 MB, raw
vectors 256 MB for rescoring; `knn_formats --mmap`, 500 queries) in a 64 MB
cgroup: `veq` advised normal vs random (the default) took 22 vs 17 ms per
query on the local disk and 221 vs 124 ms under the gp3 throttle, reading
3.6 vs 0.85 GiB. With 1 GB of memory both read the same and run within
noise.

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

## Decryption through the page cache

An input is a whole contiguous file, which is what lets codecs read a
memory-mapped file at memory speed. Over a directory that reads pages
itself, an encrypted file is opened through `Directory::open_decoded`
instead: only its header is read at open (a reader then checks the footer,
as Lucene does), and each page of whole chunks is authenticated and
decrypted once, as it is read into that directory's page cache. Readers see
plain memory; nothing is decrypted per read.

- Local disk: `PagedFsDirectory` (the demand pager, `lucene-store/src/paged_input.rs`)
  reads the ciphertext with `O_DIRECT`, so it never sits in the kernel's
  page cache, and holds the plaintext in its process-wide budget (the
  plugin's `lucene_rust.store.page_cache.size`, shared with
  `store_mode: page_cache` shards). Pages are whole chunks: 64 KiB with the
  default chunk size, sized up against the budget like plain files.
- Blob store: `CachedBlobDirectory` caches pages of whole chunks as
  ciphertext (in memory or on the NVMe tier) and decrypts into the reader's
  mapping.
- Any other directory, or Linux without `userfaultfd`: each file is
  authenticated and decrypted whole when it opens and costs its size in RAM
  while a reader holds it.

The plugin's `rust_encrypted` shards always read through the pager where it
runs. The node settings `lucene_rust.store.encrypted.lazy_decrypt` and
`lucene_rust.store.encrypted.decrypt_cache_size` (the earlier demand mode,
which authenticated whole files at open and decrypted from a mapping of the
ciphertext) are deprecated and ignored.

Opened plaintext is shared through a `PlaintextCache`: while any input of a
file is alive, opening it again, from the same directory or from another
one given the same cache, costs a reference count. In the plugin the
writer's and the readers' directories of one shard share one cache.

On a 2M-document index in 8+ segments (290 MB; 4 cores, local disk;
`lucene-store-crypto` example `encrypted_store_bench`, kernel caches dropped
before each run, mean of two runs), with a reader open and 116 term queries
per pass:

| | plain, mmap | plain, pager | encrypted, whole-file | encrypted, earlier demand mode | encrypted, pager |
|---|---|---|---|---|---|
| reader open | 62 ms | 99 ms | 168 ms | 186 ms | 121 ms |
| anonymous memory after the queries | 3 MB | 52 MB | 280 MB | 48 MB | 58 MB |
| kernel page cache mapped | 45 MB | 7 MB | 7 MB | 284 MB (ciphertext) | 7 MB |
| first (cold) pass | 27 to 46 ms | 40 to 44 ms | 19 to 21 ms | 35 to 44 ms | 50 to 55 ms |
| warm pass, median of 20 | 15 ms | 14 to 18 ms | 16 ms | 16 to 17 ms | 17 to 18 ms |
| force merge to one segment | 5.4 s | 6.1 s | 5.6 s | 5.4 s | 6.4 s |

Whole-file decryption pays at open for what the first queries then find in
memory, and holds the whole index on the heap. With a 32 MiB budget the
paged reader holds 34 MB and its warm passes take 62 to 67 ms (53 to 57 ms
plain, through the pager).

What it needs and how it fails:

- Linux with `userfaultfd` permitted. Docker's and containerd's default
  seccomp profiles (and Kubernetes `RuntimeDefault`) block it; a container
  needs a profile that allows `userfaultfd`. Where it is unavailable, or a
  file's chunk size is not a whole number of pages, files are decrypted
  whole.
- A process without `CAP_SYS_PTRACE` (and with
  `vm.unprivileged_userfaultfd = 0`) gets user-mode-only faults: the kernel
  cannot fault a page in, so a system call reading straight from a page
  that is not resident fails with `EFAULT`. `FsIndexOutput` copies such
  bytes through a buffer and writes them again; other code handing input
  bytes to the kernel must do the same.
- A chunk that fails authentication (the ciphertext changed or rotted under
  an open reader) is not retried and does not end the process. Reads that
  announce themselves (`Input::will_need`, `prefetch`) get a corruption
  error, and so does every later one on that file. A plain memory access
  cannot return an error, so the page reads as zeros, and the failure is
  recorded under the file's path (`lucene-store/src/read_failures.rs`) until
  the file is closed. The plugin's FFI layer fails any call that read the
  shard while a failure is recorded under it (`Io`, "failed
  authentication"); a merge reading the bad chunk fails its checksum check.
  OpenSearch then fails the shard and recovers it from a replica.
- An I/O error reading a page fails the calls that read it, as for an
  unencrypted paged shard, and the page is read again on its next touch.

## Limits to know

- Without the pager (no `userfaultfd`), an encrypted or remote file is read
  whole into memory when opened (see above).
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
  each chunk once into the page cache, as above.
