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

criterion_group!(
    benches,
    bench_direct_monotonic_get,
    bench_stored_fields_document,
    bench_points_decode_all,
    bench_doc_values_numeric_value
);
criterion_main!(benches);
