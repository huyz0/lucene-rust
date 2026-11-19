//! Native ingest cost of the OpenSearch plugin's write path for a logs-shaped
//! document, through the C ABI the plugin calls: the blob decode alone
//! (`ffi_engine_prepare`), then decode + apply, one thread, no counting
//! allocator, so the numbers are wall time per document. Run with
//! `cargo run --profile dist --bin ingest-bench -- [docs]`.

#![allow(dead_code)]

use std::ffi::c_char;

use lucene_ffi as _;

type CommitDataUpcall = unsafe extern "C" fn(u64, *mut u8, usize, *mut usize) -> i32;

/// The commit-data upcall: every commit's user data is empty (a zero count).
unsafe extern "C" fn empty_commit_data(_: u64, out: *mut u8, cap: usize, len: *mut usize) -> i32 {
    let empty = 0i32.to_le_bytes();
    // SAFETY: the caller passes `len` valid for one write and `out` for `cap`.
    unsafe {
        *len = empty.len();
        if cap < empty.len() {
            return 1;
        }
        std::ptr::copy_nonoverlapping(empty.as_ptr(), out, empty.len());
    }
    0
}

extern "C" {
    fn ffi_engine_writer_open(
        path: *const c_char,
        path_len: usize,
        ram_buffer_mb: f64,
        fault_injection: u8,
        index_sort: *const u8,
        index_sort_len: usize,
        parent_field: *const c_char,
        parent_field_len: usize,
        max_docs: i32,
        out_handle: *mut u64,
    ) -> i32;
    fn ffi_engine_writer_register_field(
        handle: u64,
        spec: *const u8,
        spec_len: usize,
        out_number: *mut i32,
    ) -> i32;
    fn ffi_engine_writer_apply(handle: u64, op: *const u8, op_len: usize) -> i32;
    fn ffi_engine_writer_commit(handle: u64, context: u64, out_generation: *mut i64) -> i32;
    fn ffi_engine_set_commit_data_upcall(upcall: Option<CommitDataUpcall>) -> i32;
    fn ffi_engine_writer_force_merge(
        handle: u64,
        max_segments: i32,
        only_deletes: u8,
        out_generation: *mut i64,
    ) -> i32;
    fn ffi_engine_writer_close(handle: u64) -> i32;
    fn ffi_engine_writer_stats(handle: u64, out: *mut i64, n: usize) -> i32;
    fn ffi_open_jvm_reader(
        path: *const c_char,
        path_len: usize,
        infos: *const u8,
        infos_len: usize,
        generation: i64,
        previous: u64,
        expected_max_docs: *const i32,
        segment_count: usize,
        live_words: *const u64,
        live_word_counts: *const usize,
        out_handle: *mut u64,
    ) -> i32;
    fn ffi_jvm_reader_search(
        handle: u64,
        query: *const u8,
        query_len: usize,
        top_n: usize,
        count_limit: i64,
        out_docs: *mut i32,
        out_scores: *mut f32,
        buf_len: usize,
        out_hit_count: *mut usize,
        out_total: *mut i64,
        out_total_is_lower_bound: *mut bool,
    ) -> i32;
    fn ffi_jvm_reader_document_alloc(
        handle: u64,
        segment: i32,
        doc: i32,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn ffi_jvm_free_bytes(ptr: *mut u8, len: usize);
    fn ffi_close_jvm_reader(handle: u64) -> i32;
}

/// A little-endian blob, as the plugin's encoders build them.
#[derive(Default)]
struct Blob(Vec<u8>);

impl Blob {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn i64(&mut self, v: i64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.i32(b.len() as i32);
        self.0.extend_from_slice(b);
        self
    }
}

/// `ffi_engine_writer_register_field`'s spec.
fn field(
    name: &str,
    index_options: u8,
    omit_norms: u8,
    dv: u8,
    point_bytes: i32,
    soft: u8,
) -> Vec<u8> {
    let mut b = Blob::default();
    b.bytes(name.as_bytes())
        .u8(index_options)
        .u8(omit_norms)
        .u8(0)
        .u8(0)
        .u8(dv)
        .u8(0)
        .i32(i32::from(point_bytes > 0))
        .i32(i32::from(point_bytes > 0))
        .i32(point_bytes)
        .i32(0)
        .u8(soft)
        // no field attributes
        .i32(0);
    b.0
}

fn check(rc: i32, what: &str) {
    assert_eq!(rc, 0, "{what} failed with status {rc}");
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Skewed towards small values, so a few terms are frequent.
    fn skewed(&mut self, n: u64) -> u64 {
        let a = self.next() % n;
        self.next() % (a + 1)
    }
}

const SVC: [&str; 8] = [
    "auth", "api", "billing", "search", "cart", "gateway", "worker", "cron",
];
const PATHS: [&str; 7] = [
    "/api/v1/users",
    "/api/v1/orders",
    "/search",
    "/login",
    "/health",
    "/cart/add",
    "/checkout",
];

struct Ids {
    id: i32,
    source: i32,
    seq: i32,
    term: i32,
    version: i32,
    ts: i32,
    host: i32,
    service: i32,
    level: i32,
    path: i32,
    trace: i32,
    status: i32,
    bytes: i32,
    latency: i32,
    message: i32,
}

fn keyword(b: &mut Blob, f: i32, v: &[u8]) {
    b.i32(f).u8(0).i32(1).bytes(v).i32(1).u8(0);
}

extern "C" {
    fn ffi_engine_prepare(op: *const u8, op_len: usize, out_id: *mut u64) -> i32;
    fn ffi_engine_prepared_free(id: u64) -> i32;
    fn ffi_engine_writer_apply_prepared(handle: u64, id: u64) -> i32;
    fn ffi_jvm_analysis_register(blob: *const u8, blob_len: usize, out_id: *mut i32) -> i32;
}

fn analyzer_id() -> i32 {
    let mut spec = Blob::default();
    spec.u8(1).i32(100).i32(1).i32(0);
    spec.bytes(b"standard")
        .i32(1)
        .bytes(b"maxTokenLength")
        .bytes(b"255");
    spec.i32(1).bytes(b"lowercase").i32(0).i32(0);
    let mut b = Blob::default();
    b.bytes(&spec.0).bytes(b"BM25");
    let mut id = -1;
    // SAFETY: live slice and out-pointer.
    check(
        unsafe { ffi_jvm_analysis_register(b.0.as_ptr(), b.0.len(), &mut id) },
        "analyzer",
    );
    id
}

fn op(f: &Ids, analyzer: i32, i: usize, rng: &mut Rng, buf: &mut Blob) {
    let id = format!("{i:020}");
    let trace = format!("{:032x}", rng.next() as u128 * 31 + i as u128);
    let host = format!("host-{:03}", rng.next() % 200);
    let ts = 1_793_491_200_000i64 + i as i64 * 37;
    let status = [200i32, 200, 200, 201, 204, 301, 400, 404, 500, 503][(rng.next() % 10) as usize];
    let nbytes = 100 + (rng.next() % 199_900) as i64;
    let lat = (rng.next() % 4000) as f32 / 100.0;
    let svc = SVC[(rng.next() % 8) as usize];
    let level = ["INFO", "WARN", "ERROR"][(rng.next() % 3) as usize];
    let path = PATHS[(rng.next() % 7) as usize];
    let msg = format!(
        "user u{} logged in from 10.{}.{}.{} after {} ms retrying upstream call attempt {}",
        rng.next() % 50000,
        rng.next() % 256,
        rng.next() % 256,
        rng.next() % 256,
        rng.next() % 5000,
        rng.next() % 4
    );
    let source = format!(
        "{{\"@timestamp\":\"2026-11-01T00:00:00.000Z\",\"host\":\"{host}\",\"service\":\"{svc}\",\"level\":\"{level}\",\"path\":\"{path}\",\"status\":{status},\"bytes\":{nbytes},\"latency_ms\":{lat},\"trace_id\":\"{trace}\",\"message\":\"{msg}\"}}"
    );
    let b = buf;
    b.0.clear();
    b.u8(0).i32(1);
    b.i32(2);
    b.i32(f.id).u8(1).bytes(id.as_bytes());
    b.i32(f.source).u8(1).bytes(source.as_bytes());
    b.i32(7);
    keyword(b, f.id, id.as_bytes());
    keyword(b, f.host, host.as_bytes());
    keyword(b, f.service, svc.as_bytes());
    keyword(b, f.level, level.as_bytes());
    keyword(b, f.path, path.as_bytes());
    keyword(b, f.trace, trace.as_bytes());
    b.i32(f.message)
        .u8(2)
        .i32(analyzer)
        .u8(3)
        .u8(1)
        .bytes(b"message")
        .i32(1)
        .bytes(msg.as_bytes());
    // doc values: 3 + ts + status + bytes + latency numeric, 5 sorted-set
    b.i32(13);
    b.i32(f.seq).u8(0).i64(i as i64);
    b.i32(f.term).u8(0).i64(1);
    b.i32(f.version).u8(0).i64(1);
    b.i32(f.ts).u8(0).i64(ts);
    b.i32(f.status).u8(0).i64(status as i64);
    b.i32(f.bytes).u8(0).i64(nbytes);
    b.i32(f.latency).u8(0).i64(lat.to_bits() as i64);
    for (field, v) in [
        (f.host, host.as_str()),
        (f.service, svc),
        (f.level, level),
        (f.path, path),
        (f.trace, trace.as_str()),
    ] {
        b.i32(field).u8(1).bytes(v.as_bytes());
    }
    b.i32(f.id).u8(1).bytes(id.as_bytes());
    // points: ts, status, bytes, latency, seq_no
    b.i32(5);
    b.i32(f.ts).bytes(&(ts ^ i64::MIN).to_be_bytes());
    b.i32(f.status)
        .bytes(&((status ^ i32::MIN) as u32).to_be_bytes());
    b.i32(f.bytes).bytes(&(nbytes ^ i64::MIN).to_be_bytes());
    b.i32(f.latency).bytes(&lat.to_bits().to_be_bytes());
    b.i32(f.seq).bytes(&((i as i64) ^ i64::MIN).to_be_bytes());
}

fn main() {
    let docs: usize = std::env::args()
        .nth(1)
        .map_or(300_000, |a| a.parse().expect("docs"));
    let root = std::env::temp_dir().join(format!("ingest-bench-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.to_str().unwrap().to_string();
    let mut h = 0u64;
    // SAFETY: live string and out-pointer.
    check(
        unsafe {
            ffi_engine_writer_open(
                path.as_ptr().cast(),
                path.len(),
                256.0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                i32::MAX,
                &mut h,
            )
        },
        "open writer",
    );
    let register = |spec: Vec<u8>| {
        let mut n = -1;
        // SAFETY: live slice and out-pointer.
        check(
            unsafe { ffi_engine_writer_register_field(h, spec.as_ptr(), spec.len(), &mut n) },
            "register",
        );
        n
    };
    let kw = |name: &str| register(field(name, 1, 1, 4, 0, 0));
    let f = Ids {
        id: register(field("_id", 1, 1, 4, 0, 0)),
        source: register(field("_source", 0, 0, 0, 0, 0)),
        seq: register(field("_seq_no", 0, 0, 1, 8, 0)),
        term: register(field("_primary_term", 0, 0, 1, 0, 0)),
        version: register(field("_version", 0, 0, 1, 0, 0)),
        ts: register(field("@timestamp", 0, 0, 5, 8, 0)),
        host: kw("host"),
        service: kw("service"),
        level: kw("level"),
        path: kw("path"),
        trace: kw("trace_id"),
        status: register(field("status", 0, 0, 5, 4, 0)),
        bytes: register(field("bytes", 0, 0, 5, 8, 0)),
        latency: register(field("latency_ms", 0, 0, 5, 4, 0)),
        message: register(field("message", 3, 0, 0, 0, 0)),
    };
    register(field("__soft_deletes", 0, 0, 1, 0, 1));
    // SAFETY: the upcall keeps its contract for the whole run.
    check(
        unsafe { ffi_engine_set_commit_data_upcall(Some(empty_commit_data)) },
        "upcall",
    );
    let analyzer = analyzer_id();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut ops: Vec<Vec<u8>> = Vec::with_capacity(docs);
    let mut scratch = Blob::default();
    for i in 0..docs {
        op(&f, analyzer, i, &mut rng, &mut scratch);
        ops.push(scratch.0.clone());
    }
    let mean = ops.iter().map(Vec::len).sum::<usize>() / docs;
    println!("docs {docs}, blob {mean} bytes mean");
    for pass in 0..3 {
        let t = std::time::Instant::now();
        for o in &ops {
            let mut id = 0u64;
            // SAFETY: live slice and out-pointer.
            check(
                unsafe { ffi_engine_prepare(o.as_ptr(), o.len(), &mut id) },
                "prepare",
            );
            // SAFETY: id from prepare.
            check(unsafe { ffi_engine_prepared_free(id) }, "free");
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / docs as f64;
        println!("pass {pass}: prepare only {us:.2} us/doc");
    }
    let mut generation = 0i64;
    for pass in 0..3 {
        let t = std::time::Instant::now();
        for o in &ops {
            let mut id = 0u64;
            // SAFETY: live slice and out-pointer.
            check(
                unsafe { ffi_engine_prepare(o.as_ptr(), o.len(), &mut id) },
                "prepare",
            );
            // SAFETY: id from prepare, handle from open.
            check(unsafe { ffi_engine_writer_apply_prepared(h, id) }, "apply");
        }
        let add = t.elapsed().as_secs_f64() * 1e6 / docs as f64;
        let t = std::time::Instant::now();
        // SAFETY: live out-pointer.
        check(
            unsafe { ffi_engine_writer_commit(h, 0, &mut generation) },
            "commit",
        );
        let flush = t.elapsed().as_secs_f64() * 1e6 / docs as f64;
        println!("pass {pass}: prepare+apply {add:.2} us/doc, commit {flush:.2} us/doc");
    }
    ffi_engine_writer_close_checked(h);
    let _ = std::fs::remove_dir_all(&root);
}

fn ffi_engine_writer_close_checked(h: u64) {
    // SAFETY: a handle from open.
    check(unsafe { ffi_engine_writer_close(h) }, "close");
}
