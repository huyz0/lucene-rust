//! Microbenchmarks for the read-path decode loops that are actually wired
//! up end to end in this port (see `docs/parity.md`): bit-packed monotonic
//! sequence lookup, stored-fields document decode (LZ4), BKD point
//! decoding, and per-doc numeric doc-values lookup. Each reuses real
//! Java-Lucene-produced fixture bytes from `fixtures/data/` (the same ones
//! the differential tests in `crates/lucene-codecs/tests/` verify against)
//! rather than synthetic data, so timings track something representative
//! of real segments.
//!
//! Run with: `cargo bench -p lucene-codecs`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::{direct_monotonic, doc_values as ndv, field_infos, points, stored_fields};

fn fixtures_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/").to_string()
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load(sub_dir: &str) -> Self {
        let text =
            std::fs::read_to_string(format!("{}{}/manifest.properties", fixtures_dir(), sub_dir))
                .expect("run fixtures generator first");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

/// `DirectMonotonicReader.get` bit-unpacking, at 16384 values/block -- the
/// same per-block granularity `Lucene90DocValuesFormat` uses for variable-
/// length binary doc-values address arrays.
fn bench_direct_monotonic_get(c: &mut Criterion) {
    const NUM_VALUES: i64 = 16384;
    const BLOCK_SHIFT: u32 = 10; // 1024 values/block, matches typical DV address blocks
    let values: Vec<i64> = (0..NUM_VALUES).map(|i| i * 7 + (i % 5)).collect();
    let (meta_bytes, data) = direct_monotonic::write(&values, BLOCK_SHIFT);
    let mut input = lucene_store::SliceInput::new(&meta_bytes);
    let meta = direct_monotonic::load_meta(&mut input, NUM_VALUES, BLOCK_SHIFT).unwrap();

    c.bench_function("direct_monotonic/get_block", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for i in 0..NUM_VALUES {
                sum = sum.wrapping_add(direct_monotonic::get(&data, &meta, black_box(i)).unwrap());
            }
            black_box(sum)
        })
    });
}

/// `StoredFieldsReader::document` -- per-document LZ4 chunk decompress plus
/// field decode, using the real `.fdt`/`.fdx`/`.fdm` fixture (6 docs, one
/// field of every supported type).
fn bench_stored_fields_document(c: &mut Criterion) {
    let manifest = Manifest::load("stored_fields_index");
    let dir = format!("{}stored_fields_index/", fixtures_dir());
    let id = id_from_hex(manifest.get("id_hex"));
    let fdt = std::fs::read(format!("{dir}{}.raw", manifest.get("fdt_file_name"))).unwrap();
    let fdx = std::fs::read(format!("{dir}{}.raw", manifest.get("fdx_file_name"))).unwrap();
    let fdm = std::fs::read(format!("{dir}{}.raw", manifest.get("fdm_file_name"))).unwrap();
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();

    let reader = stored_fields::open(&fdt, &fdx, &fdm, &id, "").unwrap();
    c.bench_function("stored_fields/document_all_docs", |b| {
        b.iter(|| {
            for doc_id in 0..max_doc {
                black_box(reader.document(black_box(doc_id)).unwrap());
            }
        })
    });
}

/// `PointsReader::decode_all_points` -- BKD leaf decode across ~2000 points
/// (several leaves past the default 512-points-per-leaf threshold).
fn bench_points_decode_all(c: &mut Criterion) {
    let manifest = Manifest::load("points_index");
    let dir = format!("{}points_index/", fixtures_dir());
    let id = id_from_hex(manifest.get("id_hex"));
    let kdm = std::fs::read(format!("{dir}{}.raw", manifest.get("kdm_file_name"))).unwrap();
    let kdi = std::fs::read(format!("{dir}{}.raw", manifest.get("kdi_file_name"))).unwrap();
    let kdd = std::fs::read(format!("{dir}{}.raw", manifest.get("kdd_file_name"))).unwrap();
    let field_number: i32 = manifest.get("field_number").parse().unwrap();

    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    c.bench_function("points/decode_all_points", |b| {
        b.iter(|| black_box(reader.decode_all_points(black_box(field_number)).unwrap()))
    });
}

/// `doc_values::numeric_value` -- per-doc numeric doc-values lookup. The
/// fixture segment is only 5 docs (small by construction of the
/// differential-test generator), so this loops over it many times per
/// `iter()` to get a stable per-call measurement rather than reflecting a
/// realistic single-block size like the other benchmarks here.
fn bench_doc_values_numeric_value(c: &mut Criterion) {
    let manifest = Manifest::load("doc_values_index");
    let dir = format!("{}doc_values_index/", fixtures_dir());
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let meta_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
    let data_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
    let segment_name = manifest.get("segment_name");
    let dvm_name = manifest.get("dvm_file_name");
    let suffix = dvm_name
        .strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap();
    let (_, parsed) = ndv::parse_meta(&meta_buf, &id, suffix, &fis).unwrap();
    let field_number: i32 = manifest
        .get("field_numbers")
        .split(',')
        .find_map(|kv| {
            let (name, num) = kv.split_once(':').unwrap();
            (name == "varying").then(|| num.parse().unwrap())
        })
        .unwrap();
    let entry = parsed.numeric_entry(field_number).unwrap();
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();

    c.bench_function("doc_values/numeric_value_repeated", |b| {
        b.iter(|| {
            // Repeat over the small fixture segment ~3300x to approximate
            // one 16384-doc block's worth of per-doc lookups.
            let mut sum: i64 = 0;
            for _ in 0..3300 {
                for doc in 0..max_doc {
                    if let Some(v) = ndv::numeric_value(&data_buf, entry, black_box(doc)).unwrap() {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum)
        })
    });
}

/// Sparse doc-values lookup, at three field cardinalities.
///
/// This one is here to demonstrate a scaling defect rather than to track a
/// speed. `norms::norm_value` and `doc_values`' sparse paths call
/// `indexed_disi::decode_doc_ids`, which decodes the **entire** `IndexedDISI`
/// region into a fresh `Vec<i32>` and then binary-searches it -- so one lookup
/// is O(number of documents with the field) in both time and allocation.
/// Lucene's `IndexedDISI` is a forward-only iterator with a jump table and
/// answers `advance(target)` in roughly constant time.
///
/// Both shapes are measured. `decode_all` is what every sparse lookup used to
/// do and is linear in `n`; `cursor` is `DisiCursor`, which walks at most one
/// block header per 65,536 documents and should be flat. The contrast between
/// the two curves is the finding and the fix in one chart. No Java counterpart:
/// the point is the shape on this side, not a ratio.
fn bench_sparse_doc_values_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_disi/sparse_lookup");
    for n in [1_000usize, 10_000, 100_000] {
        // Every 7th doc present, so the blocks are genuinely SPARSE rather
        // than degenerating to ALL.
        let doc_ids: Vec<i32> = (0..n).map(|i| (i * 7) as i32).collect();
        let region = lucene_codecs::indexed_disi::write(&doc_ids);
        let target = doc_ids[n / 2];
        // The old shape: decode the whole region, then binary-search it.
        group.bench_function(format!("decode_all/n{n}"), |b| {
            b.iter(|| {
                let decoded =
                    lucene_codecs::indexed_disi::decode_doc_ids(black_box(&region), 0).unwrap();
                black_box(lucene_codecs::indexed_disi::rank_of(
                    &decoded,
                    black_box(target),
                ));
            });
        });
        // What the sparse doc-values and norms paths do now: a forward-only
        // cursor that walks block headers. Flat in `n` where the above is
        // linear -- that contrast is the finding and the fix in one chart.
        group.bench_function(format!("cursor/n{n}"), |b| {
            b.iter(|| {
                let mut c = lucene_codecs::indexed_disi::DisiCursor::new(black_box(&region), 0);
                black_box(c.advance_exact(black_box(target)).unwrap());
            });
        });
    }
    group.finish();
}

/// Sequential `numeric_value` over a real varying-bits-per-value NUMERIC field
/// -- the access pattern a sort or a facet count has.
///
/// `numeric_value` re-reads the per-field jump table and the block header (one
/// byte, one `i64`, one `i32`) for **every value**. Lucene's
/// `VaryingBPVReader.getLongValue` opens with `if (this.block != block)` and
/// keeps the decoded block, paying that once per 16,384-value block instead.
///
/// **The two cases coming out equal is the finding, not a failure of the
/// benchmark.** `stride1` stays inside one block for 16,384 consecutive calls
/// and `stride16k` crosses a block on every call; an implementation that caches
/// the block is much faster on the first and no faster on the second. This
/// port's free-function path is identical on both, because it caches nothing.
/// [`ndv::NumericReader`] is the version that does, and is measured beside it.
///
/// Rust-only: what matters is the difference between access patterns on this
/// side, not a ratio against Java.
fn bench_varying_bpv_numeric(c: &mut Criterion) {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_varying_bpv/"
    );
    let text = match std::fs::read_to_string(format!("{dir}manifest.properties")) {
        Ok(t) => t,
        // The fixture is generated, not committed in every checkout; skip
        // rather than fail the whole bench binary.
        Err(_) => return,
    };
    let kv: Vec<(String, String)> = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let get = |key: &str| -> String {
        kv.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    };

    let id = id_from_hex(&get("id_hex"));
    let fnm = std::fs::read(format!("{dir}{}.raw", get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let meta = std::fs::read(format!("{dir}{}.raw", get("dvm_file_name"))).unwrap();
    let data = std::fs::read(format!("{dir}{}.raw", get("dvd_file_name"))).unwrap();

    let segment_name = get("segment_name");
    let dvm = get("dvm_file_name");
    let suffix = dvm
        .strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .expect("unexpected dvm file name shape")
        .to_string();

    let (_, parsed) = ndv::parse_meta(&meta, &id, &suffix, &fis).unwrap();
    let field_number: i32 = get("field_numbers")
        .split(',')
        .find_map(|kv| {
            let (name, num) = kv.split_once(':').unwrap();
            (name == "varying_bpv").then(|| num.parse().unwrap())
        })
        .expect("varying_bpv field missing");
    let entry = parsed.numeric_entry(field_number).unwrap().clone();
    assert!(
        entry.block_shift.is_some(),
        "fixture is not a varying-bits-per-value field, so this measures nothing"
    );
    let max_doc: i32 = get("max_doc").parse().unwrap();

    let mut group = c.benchmark_group("doc_values/varying_bpv");
    // Sequential: thousands of consecutive reads land in one block.
    group.bench_function("stride1", |b| {
        let mut doc = 0i32;
        b.iter(|| {
            doc = (doc + 1) % max_doc;
            black_box(ndv::numeric_value(black_box(&data), &entry, doc).unwrap());
        });
    });
    // Control: one block crossing per call, so per-block setup is unavoidable
    // for either implementation.
    group.bench_function("stride16k", |b| {
        let mut doc = 0i32;
        b.iter(|| {
            doc = (doc + 16384) % max_doc;
            black_box(ndv::numeric_value(black_box(&data), &entry, doc).unwrap());
        });
    });
    // The same two patterns through the block-caching reader.
    group.bench_function("reader_stride1", |b| {
        let mut reader = ndv::NumericReader::new(&data, &entry);
        let mut doc = 0i32;
        b.iter(|| {
            doc = (doc + 1) % max_doc;
            black_box(reader.value(doc).unwrap());
        });
    });
    group.bench_function("reader_stride16k", |b| {
        let mut reader = ndv::NumericReader::new(&data, &entry);
        let mut doc = 0i32;
        b.iter(|| {
            doc = (doc + 16384) % max_doc;
            black_box(reader.value(doc).unwrap());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_direct_monotonic_get,
    bench_stored_fields_document,
    bench_points_decode_all,
    bench_doc_values_numeric_value,
    bench_sparse_doc_values_lookup,
    bench_varying_bpv_numeric
);
criterion_main!(benches);
