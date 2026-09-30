//! Differential tests for scalar vector quantization against Lucene 10.5.0,
//! over the fixtures `fixtures/src/GenScalarQuantized.java` writes:
//!
//! - `scalar_quantized/quantizer.txt`: `OptimizedScalarQuantizer` codes and
//!   corrective terms (bit for bit), `deQuantize`, the packing helpers, the
//!   byte kernels, and the legacy `ScalarQuantizer`'s quantiles, codes,
//!   offsets and quantized similarity scores;
//! - `scalar_quantized_index/`: a real `Lucene104HnswScalarQuantizedVectorsFormat`
//!   segment, one format instance per `ScalarEncoding`. The Rust writer,
//!   given the raw vectors Lucene wrote to `.vec`, must reproduce `.veq` and
//!   `.vemq` byte for byte, and a search through this port's reader over
//!   Lucene's graph must return Lucene's top-10 with Lucene's scores;
//! - `scalar_quantized_merge_{plain,deletes}/`: the merge path, whose centroid
//!   comes from the source segments' centroids (plain) or is recomputed from
//!   every source vector (a source with deletions). The merged `.veq`/`.vemq`
//!   must be byte-identical too.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::filtered_hnsw_searcher::search_with_strategy;
use lucene_codecs::hnsw_vectors::{self, HnswVectorsReader};
use lucene_codecs::scalar_quantized_vectors::{
    exhaustive_search, QuantizedMergeSource, QuantizedQueryScorer, QuantizedVectorsField,
    ScalarQuantizedVectorsReader, ScalarQuantizedVectorsWriter, HNSW_NAME,
};
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::quantization::{
    self, OptimizedScalarQuantizer, ScalarEncoding, ScalarQuantizedVectorSimilarity,
    ScalarQuantizer,
};
use lucene_util::vector_util;

fn data(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(rel)
}

fn floats(s: &str) -> Vec<f32> {
    if s == "-" {
        return Vec::new();
    }
    s.split(',')
        .map(|x| f32::from_bits(x.parse::<i32>().unwrap() as u32))
        .collect()
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn fbits(s: &str) -> u32 {
    s.parse::<i32>().unwrap() as u32
}

fn util_sim(ord: usize) -> quantization::VectorSimilarityFunction {
    [
        quantization::VectorSimilarityFunction::Euclidean,
        quantization::VectorSimilarityFunction::DotProduct,
        quantization::VectorSimilarityFunction::Cosine,
        quantization::VectorSimilarityFunction::MaximumInnerProduct,
    ][ord]
}

fn codec_sim(ord: usize) -> VectorSimilarityFunction {
    [
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::MaximumInnerProduct,
    ][ord]
}

fn result_matches(parts: &[&str], r: &quantization::QuantizationResult, ctx: &str) {
    assert_eq!(r.lower_interval.to_bits(), fbits(parts[0]), "{ctx} lower");
    assert_eq!(r.upper_interval.to_bits(), fbits(parts[1]), "{ctx} upper");
    assert_eq!(
        r.additional_correction.to_bits(),
        fbits(parts[2]),
        "{ctx} additional"
    );
    assert_eq!(
        r.quantized_component_sum,
        parts[3].parse::<i32>().unwrap(),
        "{ctx} sum"
    );
}

#[test]
fn quantizer_matches_lucene() {
    let text = std::fs::read_to_string(data("scalar_quantized/quantizer.txt"))
        .expect("scripts/gen-fixtures.sh --only GenScalarQuantized");
    let lines: Vec<Vec<&str>> = text.lines().map(|l| l.split(' ').collect()).collect();
    let (mut osq, mut pack, mut kernels, mut legacy) = (0, 0, 0, 0);
    let mut i = 0;
    while i < lines.len() {
        let l = &lines[i];
        match l[0] {
            "osq" => {
                let sim = util_sim(l[1].parse().unwrap());
                let bits: u8 = l[2].parse().unwrap();
                let vec = floats(l[3]);
                let centroid = floats(l[4]);
                let ctx = format!("osq #{osq} sim={sim:?} bits={bits} dim={}", vec.len());
                let q = OptimizedScalarQuantizer::new(sim);
                let mut copy = vec.clone();
                let mut dest = vec![0u8; vec.len()];
                let r = q.scalar_quantize(&mut copy, &mut dest, bits, &centroid);
                let res = &lines[i + 1];
                result_matches(&res[1..5], &r, &ctx);
                assert_eq!(dest, unhex(res[5]), "{ctx} codes");
                let centered: Vec<u32> = copy.iter().map(|x| x.to_bits()).collect();
                let want: Vec<u32> = floats(res[6]).iter().map(|x| x.to_bits()).collect();
                assert_eq!(centered, want, "{ctx} centred vector");
                let mut deq = vec![0f32; vec.len()];
                quantization::dequantize(
                    &dest,
                    &mut deq,
                    bits,
                    r.lower_interval,
                    r.upper_interval,
                    &centroid,
                );
                let want: Vec<u32> = floats(lines[i + 2][1])
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                assert_eq!(
                    deq.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    want,
                    "{ctx} dequantized"
                );
                let m = &lines[i + 3];
                let mbits = [m[1].parse::<u8>().unwrap(), m[2].parse::<u8>().unwrap()];
                let mut dests = vec![vec![0u8; vec.len()], vec![0u8; vec.len()]];
                let rs = q.multi_scalar_quantize(&mut vec.clone(), &mut dests, &mbits, &centroid);
                result_matches(&m[3..7], &rs[0], &format!("{ctx} multi0"));
                assert_eq!(dests[0], unhex(m[7]), "{ctx} multi0 codes");
                result_matches(&m[8..12], &rs[1], &format!("{ctx} multi1"));
                assert_eq!(dests[1], unhex(m[12]), "{ctx} multi1 codes");
                osq += 1;
                i += 4;
                continue;
            }
            "pack" => {
                let (q4, q1, q2) = (unhex(l[1]), unhex(l[2]), unhex(l[3]));
                let mut t = vec![0u8; q4.len() / 2];
                quantization::transpose_half_byte(&q4, &mut t);
                assert_eq!(t, unhex(l[4]), "transposeHalfByte");
                let mut b = vec![0u8; q1.len() / 8];
                quantization::pack_as_binary(&q1, &mut b);
                assert_eq!(b, unhex(l[5]), "packAsBinary");
                let mut d = vec![0u8; q2.len() / 4];
                quantization::transpose_dibit(&q2, &mut d);
                assert_eq!(d, unhex(l[6]), "transposeDibit");
                assert_eq!(
                    vector_util::int4_bit_dot_product(&t, &b),
                    l[7].parse::<i64>().unwrap()
                );
                assert_eq!(
                    vector_util::int4_dibit_dot_product(&t, &d),
                    l[8].parse::<i64>().unwrap()
                );
                let mut back = vec![0u8; q1.len()];
                quantization::unpack_binary(&b, &mut back);
                assert_eq!(back, q1);
                let mut back = vec![0u8; q2.len()];
                quantization::untranspose_dibit(&d, &mut back);
                assert_eq!(back, q2);
                pack += 1;
            }
            "kernels" => {
                let (a, b) = (unhex(l[1]), unhex(l[2]));
                let n = |k: usize| l[k].parse::<i32>().unwrap();
                assert_eq!(vector_util::dot_product_i8(&a, &b), n(3));
                assert_eq!(vector_util::uint8_dot_product(&a, &b), n(4));
                assert_eq!(vector_util::uint8_dot_product_scalar(&a, &b), n(4));
                assert_eq!(vector_util::square_distance_i8(&a, &b), n(5));
                assert_eq!(vector_util::uint8_square_distance(&a, &b), n(6));
                assert_eq!(vector_util::xor_bit_count(&a, &b), n(7));
                let (a4, p4) = (unhex(l[8]), unhex(l[9]));
                assert_eq!(vector_util::int4_dot_product_single_packed(&a4, &p4), n(10));
                assert_eq!(
                    vector_util::int4_square_distance_single_packed(&a4, &p4),
                    n(11)
                );
                assert_eq!(vector_util::int4_dot_product_both_packed(&p4, &p4), n(12));
                assert_eq!(
                    vector_util::int4_square_distance_both_packed(&p4, &a),
                    n(13)
                );
                kernels += 1;
            }
            "legacy" => {
                let kind = l[1];
                let sim_ord: usize = l[2].parse().unwrap();
                let bits: u8 = l[3].parse().unwrap();
                let ci = f32::from_bits(fbits(l[4]));
                let vs: Vec<Vec<f32>> = l[6].split(';').map(floats).collect();
                let sim = util_sim(sim_ord);
                let sq = if kind == "auto" {
                    ScalarQuantizer::from_vectors_auto_interval(&vs, sim, vs.len(), bits).unwrap()
                } else {
                    ScalarQuantizer::from_vectors(&vs, ci, vs.len(), bits).unwrap()
                };
                let r = &lines[i + 1];
                let ctx = format!("legacy #{legacy} {kind} sim={sim:?} n={}", vs.len());
                assert_eq!(sq.lower_quantile().to_bits(), fbits(r[1]), "{ctx} lower");
                assert_eq!(sq.upper_quantile().to_bits(), fbits(r[2]), "{ctx} upper");
                let dim = vs[0].len();
                let (mut q0, mut q1) = (vec![0u8; dim], vec![0u8; dim]);
                let c0 = sq.quantize(&vs[0], &mut q0, sim);
                let c1 = sq.quantize(&vs[1], &mut q1, sim);
                assert_eq!(q0, unhex(r[3]), "{ctx} q0");
                assert_eq!(c0.to_bits(), fbits(r[4]), "{ctx} c0");
                assert_eq!(q1, unhex(r[5]), "{ctx} q1");
                assert_eq!(c1.to_bits(), fbits(r[6]), "{ctx} c1");
                let other = ScalarQuantizer::new(
                    sq.lower_quantile() - 0.5,
                    sq.upper_quantile() + 0.5,
                    bits,
                )
                .unwrap();
                assert_eq!(
                    other.recalculate_corrective_offset(&q0, &sq, sim).to_bits(),
                    fbits(r[7]),
                    "{ctx} recalc"
                );
                let s = ScalarQuantizedVectorSimilarity::from_vector_similarity(
                    sim,
                    sq.constant_multiplier(),
                    bits,
                );
                assert_eq!(
                    s.score(&q0, c0, &q1, c1).to_bits(),
                    fbits(r[8]),
                    "{ctx} score"
                );
                legacy += 1;
                i += 2;
                continue;
            }
            other => panic!("unknown record {other}"),
        }
        i += 1;
    }
    assert_eq!((osq, pack, kernels, legacy), (80, 20, 20, 12));
}

// ---------------------------------------------------------------------------
// Format fixtures
// ---------------------------------------------------------------------------

struct Field {
    name: String,
    number: i32,
    suffix: String,
    dim: usize,
    sim: VectorSimilarityFunction,
    encoding: ScalarEncoding,
}

struct Segment {
    name: String,
    id: [u8; 16],
    max_doc: i32,
    del_count: i32,
    fields: Vec<Field>,
}

struct Query {
    field: String,
    vector: Vec<f32>,
    hits: Vec<(i32, u32)>,
    /// A `KnnSearchStrategy.Hnsw(60)` query filtered to documents `% 10 == 0`.
    filtered: bool,
}

/// Parses a `manifest.txt`; `section` selects the lines after a `sources` /
/// `merged` marker (or everything, for the single index).
fn manifest(dir: &std::path::Path, section: Option<&str>) -> (Vec<Segment>, Vec<Query>) {
    let text = std::fs::read_to_string(dir.join("manifest.txt")).unwrap();
    let mut segments: Vec<Segment> = Vec::new();
    let mut queries = Vec::new();
    let mut active = section.is_none();
    for line in text.lines() {
        let p: Vec<&str> = line.split(' ').collect();
        match p[0] {
            "sources" | "merged" => active = Some(p[0]) == section,
            _ if !active => {}
            "segment" => {
                let mut id = [0u8; 16];
                id.copy_from_slice(&unhex(p[2]));
                segments.push(Segment {
                    name: p[1].to_string(),
                    id,
                    max_doc: p[3].parse().unwrap(),
                    del_count: p[4].parse().unwrap(),
                    fields: Vec::new(),
                });
            }
            "field" => segments.last_mut().unwrap().fields.push(Field {
                name: p[1].to_string(),
                number: p[2].parse().unwrap(),
                suffix: format!("{HNSW_NAME}_{}", p[3]),
                dim: p[4].parse().unwrap(),
                sim: codec_sim(p[5].parse().unwrap()),
                encoding: ScalarEncoding::from_wire_number(p[6].parse().unwrap()).unwrap(),
            }),
            "query" | "fquery" => queries.push(Query {
                filtered: p[0] == "fquery",
                field: p[1].to_string(),
                vector: floats(p[2]),
                hits: if p[3] == "-" {
                    Vec::new()
                } else {
                    p[3].split(',')
                        .map(|h| {
                            let (d, s) = h.split_once(':').unwrap();
                            (d.parse().unwrap(), fbits(s))
                        })
                        .collect()
                },
            }),
            other => panic!("unknown manifest line {other}"),
        }
    }
    (segments, queries)
}

fn read(dir: &std::path::Path, seg: &str, suffix: &str, ext: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{seg}_{suffix}.{ext}"))).unwrap()
}

/// The raw vectors and documents of one field, from Lucene's `.vec`.
fn raw_field(flat: &FlatVectorsReader<'_>, number: i32) -> (Vec<i32>, Vec<f32>) {
    let values = flat.float_vector_values(number).unwrap();
    let mut docs = Vec::new();
    let mut vectors = Vec::new();
    for ord in 0..values.size() {
        docs.push(values.ord_to_doc(ord).unwrap());
        vectors.extend_from_slice(&values.vector(ord).unwrap());
    }
    (docs, vectors)
}

/// The suffixes of a segment's fields, each with its fields in number order.
fn by_suffix(seg: &Segment) -> Vec<(String, Vec<&Field>)> {
    let mut out: Vec<(String, Vec<&Field>)> = Vec::new();
    for f in &seg.fields {
        match out.iter_mut().find(|(s, _)| *s == f.suffix) {
            Some((_, v)) => v.push(f),
            None => out.push((f.suffix.clone(), vec![f])),
        }
    }
    for (_, v) in &mut out {
        v.sort_by_key(|f| f.number);
    }
    out
}

/// Filtered queries answered by the graph walk (`FilteredHnswGraphSearcher`)
/// rather than by the exact fallback.
static APPROXIMATE_FILTERED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `original(doc)` is the id the generator gave the document now at `doc`
/// (its `g` filter term is `g{original % 10}`).
fn check_search(
    dir: &std::path::Path,
    seg: &Segment,
    queries: &[Query],
    original: &dyn Fn(usize) -> usize,
) -> usize {
    let mut checked = 0;
    for q in queries {
        let f = seg.fields.iter().find(|f| f.name == q.field).unwrap();
        let (veq, vemq) = (
            read(dir, &seg.name, &f.suffix, "veq"),
            read(dir, &seg.name, &f.suffix, "vemq"),
        );
        let (vem, vex) = (
            read(dir, &seg.name, &f.suffix, "vem"),
            read(dir, &seg.name, &f.suffix, "vex"),
        );
        let reader = ScalarQuantizedVectorsReader::open(&vemq, &veq, &seg.id, &f.suffix).unwrap();
        let hnsw = HnswVectorsReader::open(&vem, &vex, &seg.id, &f.suffix).unwrap();
        let values = reader.quantized_vector_values(f.number).unwrap();
        assert_eq!(values.encoding(), f.encoding);
        let graph = hnsw.graph(f.number).unwrap();
        let mut scorer = QuantizedQueryScorer::new(&values, f.sim, &q.vector).unwrap();
        let hits = if q.filtered {
            // AbstractKnnVectorQuery.getLeafResults on a one-leaf index: the
            // filter passes 100 documents (> k), so an approximate search
            // with visitedLimit = cost + 1, falling back to an exact search
            // over the filter when it stopped early or came up short.
            let mut accept = FixedBitSet::new(seg.max_doc as usize);
            for d in 0..accept.len() {
                if original(d).is_multiple_of(10) {
                    accept.set(d);
                }
            }
            let cost = accept.cardinality() as i32;
            let opts = hnsw_vectors::SearchOptions {
                accept_ords: Some(&accept),
                filtered_doc_count: Some(cost),
                seed_ords: None,
            };
            let (hits, early) =
                search_with_strategy(&mut scorer, graph.as_ref(), 10, cost as u64 + 1, opts, 60)
                    .unwrap();
            if early || hits.len() < 10 {
                exhaustive_search(&values, f.sim, &q.vector, 10, Some(&accept)).unwrap()
            } else {
                APPROXIMATE_FILTERED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                hits
            }
        } else {
            hnsw_vectors::search(
                &mut scorer,
                graph.as_ref(),
                10,
                u64::MAX,
                hnsw_vectors::SearchOptions::default(),
            )
            .unwrap()
            .0
        };
        let mut got: Vec<(i32, u32)> = hits
            .into_iter()
            .map(|(ord, s)| (values.ord_to_doc(ord).unwrap(), s.to_bits()))
            .collect();
        let mut want = q.hits.clone();
        let order = |v: &mut Vec<(i32, u32)>| {
            v.sort_by(|a, b| {
                f32::from_bits(b.1)
                    .total_cmp(&f32::from_bits(a.1))
                    .then(a.0.cmp(&b.0))
            })
        };
        order(&mut got);
        order(&mut want);
        assert_eq!(
            got, want,
            "search {} ({:?}, {:?})",
            q.field, f.encoding, f.sim
        );
        checked += 1;
    }
    checked
}

#[test]
fn flushed_segment_bytes_and_search_match_lucene() {
    let dir = data("scalar_quantized_index");
    let (segments, queries) = manifest(&dir, None);
    assert_eq!(segments.len(), 1);
    let seg = &segments[0];
    let mut files = 0;
    for (suffix, fields) in by_suffix(seg) {
        let (vec, vemf) = (
            read(&dir, &seg.name, &suffix, "vec"),
            read(&dir, &seg.name, &suffix, "vemf"),
        );
        let flat = FlatVectorsReader::open(&vemf, &vec, &seg.id, &suffix).unwrap();
        let encoding = fields[0].encoding;
        let mut writer = ScalarQuantizedVectorsWriter::new(encoding, seg.max_doc, &seg.id, &suffix);
        let raw: Vec<(Vec<i32>, Vec<f32>)> =
            fields.iter().map(|f| raw_field(&flat, f.number)).collect();
        for (f, (docs, vectors)) in fields.iter().zip(&raw) {
            writer
                .write_field(&QuantizedVectorsField {
                    field_number: f.number,
                    similarity: f.sim,
                    dimension: f.dim as i32,
                    docs,
                    vectors,
                })
                .unwrap();
        }
        let (veq, vemq) = writer.finish();
        assert_eq!(
            vemq,
            read(&dir, &seg.name, &suffix, "vemq"),
            "{suffix} .vemq ({encoding:?})"
        );
        assert_eq!(
            veq,
            read(&dir, &seg.name, &suffix, "veq"),
            "{suffix} .veq ({encoding:?})"
        );
        files += 1;
    }
    assert_eq!(files, 5);
    assert_eq!(check_search(&dir, seg, &queries, &|d| d), 60 + 30);
    // The filtered queries must have exercised the filtered graph walk, not
    // only the exact fallback.
    assert!(APPROXIMATE_FILTERED.load(std::sync::atomic::Ordering::Relaxed) >= 10);
}

fn check_merge(name: &str) {
    let base = data(name);
    let (src_segments, _) = manifest(&base, Some("sources"));
    let (merged_segments, queries) = manifest(&base, Some("merged"));
    let src_dir = base.join("sources");
    let merged_dir = base.join("merged");
    assert_eq!(src_segments.len(), 2);
    assert_eq!(merged_segments.len(), 1);
    let merged = &merged_segments[0];
    for (suffix, fields) in by_suffix(merged) {
        let (vec, vemf) = (
            read(&merged_dir, &merged.name, &suffix, "vec"),
            read(&merged_dir, &merged.name, &suffix, "vemf"),
        );
        let flat = FlatVectorsReader::open(&vemf, &vec, &merged.id, &suffix).unwrap();
        let encoding = fields[0].encoding;
        let mut writer =
            ScalarQuantizedVectorsWriter::new(encoding, merged.max_doc, &merged.id, &suffix);
        // Every source segment's view of these fields.
        let mut source_data = Vec::new();
        for s in &src_segments {
            let (svec, svemf) = (
                read(&src_dir, &s.name, &suffix, "vec"),
                read(&src_dir, &s.name, &suffix, "vemf"),
            );
            let (sveq, svemq) = (
                read(&src_dir, &s.name, &suffix, "veq"),
                read(&src_dir, &s.name, &suffix, "vemq"),
            );
            source_data.push((s, svec, svemf, sveq, svemq));
        }
        for f in &fields {
            let (docs, vectors) = raw_field(&flat, f.number);
            let mut owned = Vec::new();
            for (s, svec, svemf, sveq, svemq) in &source_data {
                let sflat = FlatVectorsReader::open(svemf, svec, &s.id, &suffix).unwrap();
                let squant =
                    ScalarQuantizedVectorsReader::open(svemq, sveq, &s.id, &suffix).unwrap();
                let (_, all) = raw_field(&sflat, f.number);
                // Java's `getCentroid` unwraps `PerFieldKnnVectorsFormat.FieldsReader`
                // to the field's reader -- here a `Lucene99HnswVectorsReader`, which
                // is not the `Lucene104ScalarQuantizedVectorsReader` it tests for --
                // so under the HNSW format no source centroid is ever visible and the
                // merged centroid is always recomputed. The source's own centroid is
                // still read, to prove it is there to be (not) used.
                assert!(squant.centroid(f.number).is_some());
                owned.push((None::<Vec<f32>>, all, s.del_count > 0));
            }
            let merge_sources: Vec<QuantizedMergeSource<'_>> = owned
                .iter()
                .map(|(c, v, del)| QuantizedMergeSource {
                    centroid: c.as_deref(),
                    vectors: v,
                    has_deletions: *del,
                })
                .collect();
            writer
                .merge_field(
                    f.number,
                    f.sim,
                    f.dim as i32,
                    &merge_sources,
                    &docs,
                    &vectors,
                )
                .unwrap();
        }
        let (veq, vemq) = writer.finish();
        assert_eq!(
            vemq,
            read(&merged_dir, &merged.name, &suffix, "vemq"),
            "{name} {suffix} .vemq"
        );
        assert_eq!(
            veq,
            read(&merged_dir, &merged.name, &suffix, "veq"),
            "{name} {suffix} .veq"
        );
    }
    // Lucene's merged graph, searched through this port's quantized scorer.
    // The merge renumbers documents past a deletion; the generator deleted
    // originals 5 and 702 in the `_deletes` variant.
    let deleted: &[usize] = if name.ends_with("deletes") {
        &[5, 702]
    } else {
        &[]
    };
    let kept: Vec<usize> = (0..1200).filter(|o| !deleted.contains(o)).collect();
    assert_eq!(
        check_search(&merged_dir, merged, &queries, &|d| kept[d]),
        18 + 12
    );
}

#[test]
fn merged_segment_bytes_match_lucene_weighted_centroids() {
    check_merge("scalar_quantized_merge_plain");
}

#[test]
fn merged_segment_bytes_match_lucene_recomputed_centroids() {
    check_merge("scalar_quantized_merge_deletes");
}
