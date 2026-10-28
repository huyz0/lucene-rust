//! Microbenchmarks for the write side: `DataOutput`'s varint encoders over a
//! `Vec<u8>` (what most codec writers buffer into) and `FsIndexOutput`'s
//! per-byte and bulk write path (what every file written through a
//! `Directory` goes through, CRC included).
//!
//! Run with: `cargo bench -p lucene-store --bench output_write`
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_store::{DataOutput, FsIndexOutput, IndexOutput};

/// Values with a postings-like magnitude mix: mostly 1-2 byte vints, some
/// wider, deterministic so runs compare.
fn values(n: usize) -> Vec<u32> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let shift = [25u32, 25, 25, 18, 18, 11, 4][(x % 7) as usize];
            (x >> 32) as u32 >> shift
        })
        .collect()
}

fn bench_vec_encode(c: &mut Criterion) {
    let vals = values(64 * 1024);
    let mut group = c.benchmark_group("data_output/vec");
    group.throughput(Throughput::Elements(vals.len() as u64));
    group.bench_function("write_vint", |b| {
        let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 5);
        b.iter(|| {
            out.clear();
            for &v in &vals {
                out.write_vint(v as i32);
            }
            black_box(out.len())
        })
    });
    group.bench_function("write_vlong", |b| {
        let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 10);
        b.iter(|| {
            out.clear();
            for &v in &vals {
                out.write_vlong((v as i64) << 3);
            }
            black_box(out.len())
        })
    });
    group.bench_function("write_group_vints", |b| {
        let mut out: Vec<u8> = Vec::with_capacity(vals.len() * 5);
        b.iter(|| {
            out.clear();
            out.write_group_vints(&vals);
            black_box(out.len())
        })
    });
    group.finish();
}

fn bench_fs_output(c: &mut Criterion) {
    let vals = values(256 * 1024);
    let dir = std::env::temp_dir().join(format!("lucene-store-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut group = c.benchmark_group("fs_index_output");
    group.sample_size(20);
    group.throughput(Throughput::Elements(vals.len() as u64));
    group.bench_function("write_vint_file", |b| {
        b.iter(|| {
            let mut out = FsIndexOutput::create(&dir, "vints.bin").unwrap();
            for &v in &vals {
                out.write_vint(v as i32);
            }
            black_box(out.checksum());
            out.close().unwrap()
        })
    });
    group.bench_function("write_byte_file", |b| {
        b.iter(|| {
            let mut out = FsIndexOutput::create(&dir, "bytes.bin").unwrap();
            for &v in &vals {
                out.write_byte(v as u8);
            }
            out.close().unwrap()
        })
    });
    let big = vec![7u8; 8 << 20];
    group.throughput(Throughput::Bytes(big.len() as u64));
    group.bench_function("write_bytes_8mib_file", |b| {
        b.iter(|| {
            let mut out = FsIndexOutput::create(&dir, "big.bin").unwrap();
            out.write_bytes(&big);
            out.close().unwrap()
        })
    });
    group.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

fn bench_footer_crc(c: &mut Criterion) {
    let buf = vec![0x5Au8; 128 << 20];
    let mut group = c.benchmark_group("codec_util/crc32_128mib");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("single_thread", |b| {
        b.iter(|| black_box(crc32fast::hash(&buf)))
    });
    group.bench_function("parallel", |b| {
        b.iter(|| black_box(lucene_store::codec_util::crc32(&buf)))
    });
    group.finish();
}

criterion_group!(benches, bench_vec_encode, bench_fs_output, bench_footer_crc);
criterion_main!(benches);
