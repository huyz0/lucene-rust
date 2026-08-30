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
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::hnsw::HnswGraphView as _;
use lucene_codecs::{
    blocktree, direct_monotonic, doc_values as ndv, field_infos, hnsw, hnsw_vectors, points,
    postings, postings_writer, stored_fields, vectors,
};

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

    // Pruning (`intersect`, the port of `PointValues.intersect`) vs. the
    // decode-everything-and-filter shape `lucene-search`'s points query
    // still uses. Same answer, different cost: `range_query` never touches a
    // leaf whose cell is outside the box, and decodes only doc ids (not
    // packed values) for leaves entirely inside it.
    let points = reader.decode_all_points(field_number).unwrap();
    let mut values: Vec<Vec<u8>> = points.iter().map(|p| p.packed_value.clone()).collect();
    values.sort();
    // A ~1%-selectivity window in the middle of the value range.
    let lo = values[values.len() / 2].clone();
    let hi = values[(values.len() / 2 + values.len() / 100).min(values.len() - 1)].clone();
    c.bench_function("points/range_query_selective", |b| {
        b.iter(|| {
            black_box(
                reader
                    .range_query(black_box(field_number), &lo, &hi)
                    .unwrap(),
            )
        })
    });
    // The Java fixture is only 1333 points => 3 leaves, so pruning there is
    // capped at ~3x by construction. This synthetic 200k-point / 391-leaf
    // tree (written by this port's own `points::write`, read back through
    // its own reader) shows the asymptotic shape: a selective query touches
    // O(log n) inner nodes and a handful of leaves instead of all of them.
    let synthetic: Vec<(i32, Vec<u8>)> = (0..200_000i32)
        .map(|i| {
            let v = (i as i64).wrapping_mul(2_654_435_761) % 1_000_000_007;
            (
                i,
                ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes().to_vec(),
            )
        })
        .collect();
    let (big_kdm, big_kdi, big_kdd) = points::write(
        &[points::WritePointsField {
            field_number: 0,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points: synthetic,
        }],
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        &id,
        "",
    )
    .unwrap();
    let big = points::open(&big_kdm, &big_kdi, &big_kdd, &id, "").unwrap();
    let mut big_values: Vec<Vec<u8>> = big
        .decode_all_points(0)
        .unwrap()
        .into_iter()
        .map(|p| p.packed_value)
        .collect();
    big_values.sort();
    let big_lo = big_values[big_values.len() / 2].clone();
    let big_hi = big_values[big_values.len() / 2 + big_values.len() / 1000].clone();
    c.bench_function("points/range_query_selective_200k", |b| {
        b.iter(|| black_box(big.range_query(black_box(0), &big_lo, &big_hi).unwrap()))
    });
    c.bench_function("points/decode_all_then_filter_selective_200k", |b| {
        b.iter(|| {
            let out: Vec<i32> = big
                .decode_all_points(black_box(0))
                .unwrap()
                .into_iter()
                .filter(|p| p.packed_value >= big_lo && p.packed_value <= big_hi)
                .map(|p| p.doc_id)
                .collect();
            black_box(out)
        })
    });

    c.bench_function("points/decode_all_then_filter_selective", |b| {
        b.iter(|| {
            let out: Vec<i32> = reader
                .decode_all_points(black_box(field_number))
                .unwrap()
                .into_iter()
                .filter(|p| p.packed_value >= lo && p.packed_value <= hi)
                .map(|p| p.doc_id)
                .collect();
            black_box(out)
        })
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
    // `indexed_disi::write` never emits a DENSE rank table, so the matching
    // metadata byte is `0xFF` (Java's `denseRankPower == -1`). `0` is not a
    // legal value for that byte and is now rejected.
    const NO_RANK: u8 = 0xFF;

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
                    lucene_codecs::indexed_disi::decode_doc_ids(black_box(&region), NO_RANK)
                        .unwrap();
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
                let mut c =
                    lucene_codecs::indexed_disi::DisiCursor::new(black_box(&region), NO_RANK);
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

/// `.doc` postings iteration through [`postings::LazyDocsCursor`] -- the
/// decode-on-demand cursor every scored query walks.
///
/// Two shapes, because they hit different code:
///
/// - **`tail_block`**: a `docFreq == 200` term, i.e. entirely one group-varint
///   tail block with no level-0 skip header at all. This is what the vast
///   majority of real terms look like, and `refillRemainder` is the only decode
///   involved.
/// - **`full_blocks`**: a `docFreq == 2600` term -- ten full 256-doc
///   `ForUtil`/`PForUtil` blocks plus a 40-doc tail -- which is where the
///   bit-unpacking kernel and the level-0 header walk actually show up.
///
/// Bytes come from this crate's own writer rather than a fixture, because no
/// checked-in fixture has a term with a tail block big enough to time (the
/// blocktree fixtures top out at a handful of docs per term).
fn bench_postings_lazy_cursor(c: &mut Criterion) {
    use field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };

    let seg_id = [7u8; 16];
    let term_with = |name: &[u8], doc_freq: i32| postings_writer::TermPostings {
        term: name.to_vec(),
        // Irregular gaps and freqs, so neither the all-consecutive nor the
        // all-equal fast path stands in for a realistic block.
        docs: {
            let mut doc = -1i32;
            (0..doc_freq)
                .map(|i| {
                    doc += 1 + (i % 7);
                    (doc, 1 + (i % 4))
                })
                .collect()
        },
        ..Default::default()
    };
    let terms = vec![term_with(b"big", 2600), term_with(b"small", 200)];
    let max_doc = terms
        .iter()
        .flat_map(|t| t.docs.iter())
        .map(|&(d, _)| d)
        .max()
        .unwrap()
        + 1;
    let input = postings_writer::FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        // Both terms use the same doc-ID progression, so the "small" term's
        // docs are a prefix of the "big" one's: 2600 distinct docs in all.
        doc_count: 2600,
        has_payloads: false,
        terms: &terms,
    };
    let out = postings_writer::write_single_field(&input, &seg_id, "").unwrap();

    let field_infos = FieldInfos {
        fields: vec![FieldInfo {
            name: "f".to_string(),
            number: 0,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::DocsAndFreqs,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }],
    };
    let fields = blocktree::open(
        &out.tim,
        &out.tip,
        &out.tmd,
        &field_infos,
        &seg_id,
        "",
        max_doc,
    )
    .unwrap();
    let doc_in = postings::DocInput::open(&out.doc, &seg_id, "").unwrap();
    let field = fields.field("f").unwrap();

    let mut group = c.benchmark_group("postings/lazy_cursor");
    for (name, term) in [("tail_block", &b"small"[..]), ("full_blocks", &b"big"[..])] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cursor = field.lazy_postings(term, &doc_in).unwrap().unwrap();
                let mut sum = 0i64;
                loop {
                    let doc = cursor.next_doc().unwrap();
                    if doc == postings::NO_MORE_DOCS {
                        break;
                    }
                    sum += doc as i64 + cursor.freq().unwrap() as i64;
                }
                black_box(sum)
            });
        });
    }
    group.finish();
}

/// Brute force vs HNSW on the same 50k x 128-dim field, plus the index-build
/// cost the graph adds.
///
/// The vectors are **clustered** (500 centroids, tight Gaussian-ish spread),
/// not uniform noise. That matters: at 128 dimensions of uniform noise all
/// distances concentrate, there is no neighbourhood structure for a graph to
/// exploit, and both this port and Lucene recover ~15% of the true top-10 --
/// a real property of the data, but a misleading thing to benchmark, because
/// the search terminates early for the wrong reason. Clustered data is the
/// shape a real embedding field has. See `docs/sweep/m2/c5-vectors.md`, which
/// reports both.
///
/// Three arms:
/// - `brute_force_50k_dim128_k10_x25`: the exhaustive `O(n*d)` scan, which is
///   what `Lucene99HnswVectorsReader.search` itself falls back to below its
///   `HNSW_GRAPH_THRESHOLD`. 25 queries per iteration.
/// - `hnsw_50k_dim128_k10_x25`: the same 25 queries through a graph built at
///   Lucene's defaults (`M = 16`, `beamWidth = 100`).
/// - `hnsw_build_50k_dim128`: one whole graph construction, i.e. what the
///   graph costs at flush/merge time to buy the query-side win.
fn bench_vectors_search(c: &mut Criterion) {
    let seg_id = [7u8; 16];
    let dimension = 128usize;
    let count = 50_000i32;
    let clusters = 500usize;
    let mut state = 12_345u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let centroids: Vec<f32> = (0..clusters * dimension).map(|_| next()).collect();
    let data: Vec<f32> = (0..count as usize)
        .flat_map(|i| {
            let base = ((i * 7919) % clusters) * dimension;
            (0..dimension)
                .map(|d| centroids[base + d] + next() * 0.15)
                .collect::<Vec<f32>>()
        })
        .collect();
    // 25 distinct queries per iteration, each a perturbed data point. A single
    // repeated query would sit entirely in cache and converge from the same
    // entry point every time, which flatters the graph by roughly 5x and is
    // not what a query workload looks like.
    const QUERIES: usize = 25;
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| {
            let base = ((q * 1_493) % count as usize) * dimension;
            (0..dimension)
                .map(|d| data[base + d] + next() * 0.2)
                .collect()
        })
        .collect();

    let field = vectors::FlatVectorsField {
        field_number: 0,
        similarity: field_infos::VectorSimilarityFunction::Euclidean,
        dimension: dimension as i32,
        docs: (0..count).collect(),
        values: vectors::FieldVectorData::Float32(data),
    };
    let (vec_bytes, vemf_bytes) =
        vectors::write_flat_vectors(&[field], count, &seg_id, "").unwrap();
    let reader = vectors::FlatVectorsReader::open(&vemf_bytes, &vec_bytes, &seg_id, "").unwrap();
    let values = reader.float_vector_values(0).unwrap();

    let graph = hnsw::HnswGraphBuilder::new(
        values.ord_scorer(),
        hnsw::DEFAULT_MAX_CONN,
        hnsw::DEFAULT_BEAM_WIDTH,
        hnsw::DEFAULT_RAND_SEED,
    )
    .unwrap()
    .build(count)
    .unwrap();

    let mut group = c.benchmark_group("vectors");
    // Both arms run all 25 queries per iteration, so per-query cost is the
    // reported time / 25 and the two are directly comparable.
    group.bench_function("brute_force_50k_dim128_k10_x25", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(values.exhaustive_search(black_box(query), 10).unwrap());
            }
        })
    });
    group.bench_function("hnsw_50k_dim128_k10_x25", |b| {
        b.iter(|| {
            for query in &queries {
                let mut scorer = values.scorer(black_box(query)).unwrap();
                black_box(
                    hnsw_vectors::search(
                        &mut scorer,
                        Some(&graph),
                        10,
                        u64::MAX,
                        hnsw_vectors::SearchOptions::default(),
                    )
                    .unwrap(),
                );
            }
        })
    });
    group.sample_size(10);
    group.bench_function("hnsw_build_50k_dim128", |b| {
        b.iter(|| {
            black_box(
                hnsw::HnswGraphBuilder::new(
                    values.ord_scorer(),
                    hnsw::DEFAULT_MAX_CONN,
                    hnsw::DEFAULT_BEAM_WIDTH,
                    hnsw::DEFAULT_RAND_SEED,
                )
                .unwrap()
                .build(count)
                .unwrap()
                .size(),
            )
        })
    });
    group.finish();
}

/// `IndexedDISI` cursor state, the way a sort/facet/range scan uses it: **one**
/// cursor walked forward over a whole field, not a fresh one per lookup.
///
/// This is the shape Lucene's `IndexedDISI` is built for -- `index`, `word`,
/// `wordIndex` and `numberOfOnes` all carry across `advanceExact` calls, so a
/// forward walk of a DENSE block reads each of its 1024 words exactly once for
/// the whole block. A cursor that recomputes the rank from the block start on
/// every call reads up to 1024 words *per lookup* instead, which is quadratic
/// in the block's cardinality.
///
/// Three arms, all Rust-only (the point is this port's own scaling):
/// - `dense_forward`: one cursor, every present doc of a DENSE block in order.
/// - `sparse_forward`: the same over a SPARSE block (<= 4095 docs/block).
/// - `dense_random`: a fresh cursor per lookup at a random target, which is the
///   worst case and the one a rank table exists to fix.
fn bench_indexed_disi_cursor(c: &mut Criterion) {
    use lucene_codecs::indexed_disi::{
        write, write_with_dense_rank_power, DisiCursor, DEFAULT_DENSE_RANK_POWER, NO_RANK,
    };

    let mut group = c.benchmark_group("indexed_disi/cursor");

    // 10,000 present docs inside one 65536-doc range: above MAX_ARRAY_LENGTH
    // (4095) and below BLOCK_SIZE, so genuinely DENSE.
    let dense_docs: Vec<i32> = (0..10_000).map(|i| i * 6).collect();
    let dense_region = write(&dense_docs);
    group.bench_function("dense_forward/n10000", |b| {
        b.iter(|| {
            let mut cursor = DisiCursor::new(black_box(&dense_region), NO_RANK);
            let mut sum = 0usize;
            for &doc in &dense_docs {
                sum += cursor.advance_exact(black_box(doc)).unwrap().unwrap();
            }
            black_box(sum)
        })
    });

    // 4,000 present docs in one range: SPARSE (explicit 16-bit doc ids).
    let sparse_docs: Vec<i32> = (0..4_000).map(|i| i * 16).collect();
    let sparse_region = write(&sparse_docs);
    group.bench_function("sparse_forward/n4000", |b| {
        b.iter(|| {
            let mut cursor = DisiCursor::new(black_box(&sparse_region), NO_RANK);
            let mut sum = 0usize;
            for &doc in &sparse_docs {
                sum += cursor.advance_exact(black_box(doc)).unwrap().unwrap();
            }
            black_box(sum)
        })
    });

    // A fresh cursor per lookup, deep inside the DENSE block: no incremental
    // state can help, so this is what the DENSE rank table is for. Without one
    // the cursor must popcount every word before the target (844 of them here);
    // with one it reads a single rank entry and at most 8 words.
    group.bench_function("dense_random/n10000", |b| {
        let target = dense_docs[9_000];
        b.iter(|| {
            let mut cursor = DisiCursor::new(black_box(&dense_region), NO_RANK);
            black_box(cursor.advance_exact(black_box(target)).unwrap())
        })
    });
    let ranked_region = write_with_dense_rank_power(&dense_docs, DEFAULT_DENSE_RANK_POWER);
    group.bench_function("dense_random_rank9/n10000", |b| {
        let target = dense_docs[9_000];
        b.iter(|| {
            let mut cursor = DisiCursor::new(black_box(&ranked_region), DEFAULT_DENSE_RANK_POWER);
            black_box(cursor.advance_exact(black_box(target)).unwrap())
        })
    });

    group.finish();
}

/// `NumericReader` over a **sparse** NUMERIC field, at three cardinalities.
///
/// The reader used to decode the field's whole `IndexedDISI` doc-id list into a
/// `Vec<i32>` at construction (4 bytes per present doc, plus the decode) so it
/// could binary-search it. Java's `Lucene90DocValuesProducer` composes an
/// `IndexedDISI` cursor with a `LongValues` and holds no per-doc array at all.
///
/// Measured as a full forward walk including construction, which is what a sort
/// or a facet count does: the `Vec` version pays its O(cardinality) decode once
/// and then binary-searches, the cursor version pays nothing up front and walks.
/// The field is constant-encoded (`bits_per_value == 0`) on purpose so that the
/// value decode is free and what is left is the docs-with-field lookup.
fn bench_sparse_numeric_reader(c: &mut Criterion) {
    let mut group = c.benchmark_group("doc_values/sparse_numeric_reader");
    for n in [1_000i64, 10_000, 100_000] {
        let doc_ids: Vec<i32> = (0..n).map(|i| (i * 7) as i32).collect();
        let data = lucene_codecs::indexed_disi::write(&doc_ids);
        let entry = ndv::NumericEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: data.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: 0xFF,
            num_values: n,
            table: None,
            bits_per_value: 0,
            min_value: 42,
            gcd: 1,
            values_offset: 0,
            values_length: 0,
            block_shift: None,
            value_jump_table_offset: 0,
        };
        // A forward walk over every present doc, construction included.
        group.bench_function(format!("forward/n{n}"), |b| {
            b.iter(|| {
                let mut reader = ndv::NumericReader::new(black_box(&data), black_box(&entry));
                let mut sum = 0i64;
                for &doc in &doc_ids {
                    sum = sum.wrapping_add(reader.value(black_box(doc)).unwrap().unwrap());
                }
                black_box(sum)
            })
        });
        // One lookup, construction included -- the "open a reader, ask one
        // question" pattern, where an O(cardinality) constructor is worst.
        group.bench_function(format!("single/n{n}"), |b| {
            let target = doc_ids[doc_ids.len() / 2];
            b.iter(|| {
                let mut reader = ndv::NumericReader::new(black_box(&data), black_box(&entry));
                black_box(reader.value(black_box(target)).unwrap())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_direct_monotonic_get,
    bench_stored_fields_document,
    bench_points_decode_all,
    bench_doc_values_numeric_value,
    bench_sparse_doc_values_lookup,
    bench_indexed_disi_cursor,
    bench_sparse_numeric_reader,
    bench_varying_bpv_numeric,
    bench_postings_lazy_cursor,
    bench_vectors_search
);
criterion_main!(benches);
