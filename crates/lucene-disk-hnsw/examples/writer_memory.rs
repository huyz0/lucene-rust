//! Peak memory of writing one field, above what the inputs hold.
//!
//! ```text
//! cargo run --release -p lucene-disk-hnsw --example writer_memory -- [n] [dim] [l2|cosine]
//! ```
//!
//! Builds a graph over random vectors (untimed), then writes the field and
//! prints the most the writer held beyond the vectors and the graph, and
//! the size of its output.
#![allow(clippy::arithmetic_side_effects)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use lucene_disk_hnsw::graph::{build_graph, GraphParams};
use lucene_disk_hnsw::{DiskHnswParams, DiskHnswWriter, FieldVectors, Similarity};
use lucene_vector_quant::osq::OsqFactory;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(now, Relaxed);
        // SAFETY: the caller upholds the allocator contract.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(now, Relaxed);
        // SAFETY: the caller upholds the allocator contract.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        // SAFETY: the caller upholds the allocator contract.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let now = if new >= l.size() {
            LIVE.fetch_add(new - l.size(), Relaxed) + (new - l.size())
        } else {
            LIVE.fetch_sub(l.size() - new, Relaxed) - (l.size() - new)
        };
        PEAK.fetch_max(now, Relaxed);
        // SAFETY: the caller upholds the allocator contract.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(300_000, |s| s.parse().unwrap());
    let dim: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let sim = match args.get(3).map(String::as_str) {
        Some("cosine") => Similarity::Cosine,
        _ => Similarity::Euclidean,
    };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let vectors: Vec<f32> = (0..n * dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect();
    let docs: Vec<i32> = (0..n as i32).collect();
    let graph = build_graph(&vectors, dim, sim, &GraphParams::default()).unwrap();
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut writer = DiskHnswWriter::new(&[1; 16], "");
    let field = FieldVectors {
        number: 0,
        similarity: sim,
        dimension: dim,
        docs: &docs,
        vectors: &vectors,
    };
    writer
        .add_field_with_graph(
            &field,
            &graph,
            &OsqFactory::bbq(),
            &DiskHnswParams::default(),
        )
        .unwrap();
    let files = writer.finish();
    let hash = [&files.meta, &files.codes, &files.nodes]
        .iter()
        .flat_map(|f| f.iter())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    println!("hash {hash:016x}");
    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    println!(
        "n={n} dim={dim} vectors={:.0} MiB peak_above_inputs={:.0} MiB output={:.0} MiB (nodes {:.0}, codes {:.0})",
        mib(n * dim * 4),
        mib(PEAK.load(Relaxed) - base),
        mib(files.nodes.len() + files.codes.len() + files.meta.len()),
        mib(files.nodes.len()),
        mib(files.codes.len())
    );
}
