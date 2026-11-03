//! What encryption costs, against the same work unencrypted:
//!
//! - `cipher/*`: AES-256-GCM seal and open of one 64 KiB chunk (the
//!   backend's raw throughput: AWS-LC, or its FIPS module with `fips`).
//! - `file/*`: writing a 16 MiB file and opening it, plain vs encrypted, over
//!   a filesystem directory (`FsDirectory` reads into memory; `MmapDirectory`
//!   maps, which is what the engine uses for plain shards).
//! - `index/*`: indexing 20 000 documents with commits and a force merge, then
//!   opening a reader and running a term query, plain vs encrypted.
//!
//! Run with: `cargo bench -p lucene-store-crypto --bench encryption`, and the
//! FIPS build with `--no-default-features --features fips`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lucene_index::document::{Document, Store, StringField, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::{DataOutput, Directory, FsDirectory, MmapDirectory};
use lucene_store_crypto::backend::Aes256Gcm;
use lucene_store_crypto::{EncryptedDirectory, KeyProvider, StaticKeyProvider};
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn keys() -> Arc<dyn KeyProvider> {
    Arc::new(StaticKeyProvider::new("bench", &[9; 32]))
}

fn bytes(n: usize) -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn bench_cipher(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("cipher/{}", lucene_store_crypto::backend_name()));
    let chunk = bytes(64 * 1024);
    group.throughput(Throughput::Bytes(chunk.len() as u64));
    let aes = Aes256Gcm::new(&[1; 32]).unwrap();
    group.bench_function("seal_64k", |b| {
        let mut buf = chunk.clone();
        b.iter(|| black_box(aes.seal(b"aad", &mut buf).unwrap()))
    });
    let mut sealed = chunk.clone();
    let (nonce, tag) = aes.seal(b"aad", &mut sealed).unwrap();
    sealed.extend_from_slice(&tag);
    group.bench_function("open_64k", |b| {
        let mut buf = sealed.clone();
        b.iter(|| {
            buf.copy_from_slice(&sealed);
            aes.open(&nonce, b"aad", &mut buf).unwrap();
            black_box(buf[0])
        })
    });
    group.finish();
}

fn write_file(dir: &dyn Directory, name: &str, data: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    for piece in data.chunks(8192) {
        out.write_bytes(piece);
    }
    out.close().unwrap();
}

fn bench_file(c: &mut Criterion) {
    let data = bytes(16 << 20);
    let mut group = c.benchmark_group(format!("file/{}", lucene_store_crypto::backend_name()));
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.sample_size(20);
    let tmp = TempDir::new("bench-crypto-file");
    let plain = FsDirectory::open(tmp.path());
    let enc = EncryptedDirectory::new(FsDirectory::open(tmp.path()), keys());
    let mut n = 0u64;
    group.bench_function("write_16m/plain", |b| {
        b.iter(|| {
            n += 1;
            let name = format!("p{n}");
            write_file(&plain, &name, &data);
            plain.delete_file(&name).unwrap();
        })
    });
    group.bench_function("write_16m/encrypted", |b| {
        b.iter(|| {
            n += 1;
            let name = format!("e{n}");
            write_file(&enc, &name, &data);
            enc.delete_file(&name).unwrap();
        })
    });
    write_file(&plain, "plain", &data);
    write_file(&enc, "enc", &data);
    let mmap = MmapDirectory::open(tmp.path());
    let enc_mmap = EncryptedDirectory::new(MmapDirectory::open(tmp.path()), keys());
    group.bench_function("open_16m/plain_mmap", |b| {
        b.iter(|| {
            black_box(
                mmap.open("plain")
                    .unwrap()
                    .iter()
                    .step_by(4096)
                    .map(|&b| b as u64)
                    .sum::<u64>(),
            )
        })
    });
    group.bench_function("open_16m/plain_read", |b| {
        b.iter(|| black_box(plain.open("plain").unwrap().len()))
    });
    group.bench_function("open_16m/encrypted", |b| {
        b.iter(|| black_box(enc_mmap.open("enc").unwrap().len()))
    });
    group.finish();
}

fn index(dir: &dyn Directory, docs: usize) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_max_buffered_docs(5000).unwrap();
    let words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    ];
    for i in 0..docs {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("doc-{i:06}"), Store::Yes));
        let body: Vec<&str> = (0..12)
            .map(|j| words[(i * 7 + j * 3) % words.len()])
            .collect();
        d.add(TextField::new("body", body.join(" "), Store::No));
        w.add_fields_document(&d).unwrap();
        if i % 5000 == 4999 {
            w.commit().unwrap();
        }
    }
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn search(dir: &dyn Directory) -> u64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let q = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", b"gamma".to_vec()))],
        ..Default::default()
    };
    searcher.search(&q, 10).unwrap().total_hits.value
}

fn bench_index(c: &mut Criterion) {
    const DOCS: usize = 20_000;
    let mut group = c.benchmark_group(format!("index/{}", lucene_store_crypto::backend_name()));
    group.sample_size(10);
    group.throughput(Throughput::Elements(DOCS as u64));
    for encrypted in [false, true] {
        let label = if encrypted { "encrypted" } else { "plain" };
        group.bench_with_input(
            BenchmarkId::new("build_and_merge_20k", label),
            &encrypted,
            |b, &enc| {
                b.iter(|| {
                    let tmp = TempDir::new("bench-crypto-index");
                    if enc {
                        index(
                            &EncryptedDirectory::new(FsDirectory::open(tmp.path()), keys()),
                            DOCS,
                        );
                    } else {
                        index(&FsDirectory::open(tmp.path()), DOCS);
                    }
                })
            },
        );
        let tmp = TempDir::new("bench-crypto-search");
        if encrypted {
            index(
                &EncryptedDirectory::new(FsDirectory::open(tmp.path()), keys()),
                DOCS,
            );
        } else {
            index(&FsDirectory::open(tmp.path()), DOCS);
        }
        group.bench_with_input(
            BenchmarkId::new("open_reader_and_search", label),
            &encrypted,
            |b, &enc| {
                b.iter(|| {
                    if enc {
                        black_box(search(&EncryptedDirectory::new(
                            MmapDirectory::open(tmp.path()),
                            keys(),
                        )))
                    } else {
                        black_box(search(&MmapDirectory::open(tmp.path())))
                    }
                })
            },
        );
    }
    group.finish();
}

/// Demand decryption: the first touch of every page of a 16 MiB file opened
/// lazily, by one reader and by four at once (each reading its own quarter,
/// as concurrent searches touch different regions). Opening (which
/// authenticates every chunk) is setup, not timed.
#[cfg(target_os = "linux")]
fn bench_lazy(c: &mut Criterion) {
    if lucene_store_crypto::lazy::Pager::global().is_none() {
        eprintln!("userfaultfd unavailable; skipping lazy benches");
        return;
    }
    let data = bytes(16 << 20);
    let mut group = c.benchmark_group(format!("lazy/{}", lucene_store_crypto::backend_name()));
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.sample_size(20);
    let tmp = TempDir::new("bench-crypto-lazy");
    let enc =
        EncryptedDirectory::new(MmapDirectory::open(tmp.path()), keys()).with_lazy_decrypt(true);
    write_file(&enc, "enc", &data);
    group.bench_function("open_16m", |b| {
        b.iter(|| black_box(enc.open("enc").unwrap().len()))
    });
    for threads in [1usize, 4] {
        group.bench_function(format!("first_touch_16m_x{threads}"), |b| {
            b.iter_batched(
                || enc.open("enc").unwrap(),
                |input| {
                    let part = input.len().div_ceil(threads);
                    let sum: u64 = std::thread::scope(|s| {
                        let handles: Vec<_> = input
                            .chunks(part)
                            .map(|piece| {
                                s.spawn(move || {
                                    piece.iter().step_by(4096).map(|&b| b as u64).sum::<u64>()
                                })
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).sum()
                    });
                    black_box(sum);
                    input
                },
                criterion::BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn bench_lazy(_c: &mut Criterion) {}

criterion_group!(benches, bench_cipher, bench_file, bench_lazy, bench_index);
criterion_main!(benches);
