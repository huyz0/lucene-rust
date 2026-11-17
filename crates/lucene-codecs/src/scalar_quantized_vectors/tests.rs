#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::hnsw::{UpdateableVectorScorer, VectorScorer};

const ID: [u8; 16] = *b"quantizedvec0001";

fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n * dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn write(
    encoding: ScalarEncoding,
    sim: VectorSimilarityFunction,
    docs: &[i32],
    v: &[f32],
    dim: i32,
    max_doc: i32,
) -> (Vec<u8>, Vec<u8>) {
    let mut w = ScalarQuantizedVectorsWriter::new(encoding, max_doc, &ID, "suf");
    w.write_field(&QuantizedVectorsField {
        field_number: 3,
        similarity: sim,
        dimension: dim,
        docs,
        vectors: v,
    })
    .unwrap();
    w.finish()
}

#[test]
fn round_trip_every_encoding_and_similarity() {
    let dim = 13usize;
    let n = 40usize;
    let v = vectors(n, dim, 9);
    let dense: Vec<i32> = (0..n as i32).collect();
    let sparse: Vec<i32> = (0..n as i32).map(|d| d * 3 + 1).collect();
    for encoding in [
        ScalarEncoding::UnsignedByte,
        ScalarEncoding::PackedNibble,
        ScalarEncoding::SevenBit,
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
    ] {
        for sim in [
            VectorSimilarityFunction::Euclidean,
            VectorSimilarityFunction::DotProduct,
            VectorSimilarityFunction::Cosine,
            VectorSimilarityFunction::MaximumInnerProduct,
        ] {
            for (docs, max_doc) in [(&dense, n as i32), (&sparse, 3 * n as i32 + 1)] {
                let (veq, vemq) = write(encoding, sim, docs, &v, dim as i32, max_doc);
                let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").unwrap();
                let e = r.field(3).unwrap();
                assert_eq!(e.scalar_encoding, encoding);
                assert_eq!(e.size, n as i32);
                assert!(r.centroid(3).is_some());
                let values = r.quantized_vector_values(3).unwrap();
                assert_eq!(values.dimension(), dim);
                assert_eq!(values.similarity(), sim);
                assert_eq!(values.encoding(), encoding);
                assert_eq!(values.centroid().len(), dim);
                assert!(values.centroid_dp().is_finite());
                for ord in 0..n as i32 {
                    assert_eq!(values.ord_to_doc(ord).unwrap(), docs[ord as usize]);
                    assert_eq!(
                        values.vector(ord).unwrap().len(),
                        encoding.doc_packed_length(dim)
                    );
                    let c = values.corrective_terms(ord).unwrap();
                    assert!(c.lower_interval <= c.upper_interval);
                }
                let mut cursor = values.doc_to_ord().unwrap();
                assert_eq!(cursor.ordinal(docs[5]).unwrap(), Some(5));
                // Querying with a stored vector finds it first (or close).
                let target = &v[7 * dim..8 * dim];
                let hits = exhaustive_search(&values, sim, target, 5, None).unwrap();
                assert_eq!(hits.len(), 5);
                if encoding == ScalarEncoding::UnsignedByte
                    && sim != VectorSimilarityFunction::MaximumInnerProduct
                {
                    assert_eq!(hits[0].0, 7, "{encoding:?} {sim:?}");
                }
                let mut out = vec![0f32; dim];
                values.dequantized(7, &mut out).unwrap();
                if encoding == ScalarEncoding::UnsignedByte
                    && sim == VectorSimilarityFunction::Euclidean
                {
                    for (a, b) in out.iter().zip(target) {
                        assert!((a - b).abs() < 0.02, "{a} vs {b}");
                    }
                }
                // Merge scorer.
                let q = if encoding.is_asymmetric() {
                    Some(binarized_query_data(&values, sim, &v).unwrap())
                } else {
                    assert!(binarized_query_data(&values, sim, &v).is_err());
                    None
                };
                let mut s = merge_scorer_supplier(&values, sim, q.as_deref()).unwrap();
                assert!(s.score(0).is_err());
                s.set_scoring_ordinal(2).unwrap();
                assert!(s.score(3).unwrap().is_finite());
                assert_eq!(s.max_ord(), n as i32);
                assert!(s.set_scoring_ordinal(n as i32).is_err());
            }
        }
    }
}

/// Modelled on Java's `TestLucene104HnswScalarQuantizedVectorsFormat
/// .testMergeCosineWithNonUnitVectors` (GITHUB#16665): for `COSINE` over
/// an asymmetric encoding, the merge scorer must score a pair exactly as
/// the search-time scorer does for the same (non-unit) query vector.
#[test]
fn asymmetric_cosine_merge_scorer_matches_the_search_scorer() {
    let dim = 24usize;
    let n = 60usize;
    let mut v = vectors(n, dim, 31);
    // Scale every vector away from the unit sphere, each by its own factor.
    for (i, chunk) in v.chunks_exact_mut(dim).enumerate() {
        let scale = 2.0 + (i % 9) as f32;
        chunk.iter_mut().for_each(|x| *x *= scale);
    }
    let docs: Vec<i32> = (0..n as i32).collect();
    let sim = VectorSimilarityFunction::Cosine;
    for encoding in [
        ScalarEncoding::SingleBitQueryNibble,
        ScalarEncoding::DibitQueryNibble,
    ] {
        let (veq, vemq) = write(encoding, sim, &docs, &v, dim as i32, n as i32);
        let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").unwrap();
        let values = r.quantized_vector_values(3).unwrap();
        let q = binarized_query_data(&values, sim, &v).unwrap();
        let mut merge = merge_scorer_supplier(&values, sim, Some(&q)).unwrap();
        for (query_ord, target_ord) in [(0, 1), (5, 17), (23, 42), (59, 3), (31, 30)] {
            let raw = &v[query_ord * dim..(query_ord + 1) * dim];
            let mut search = QuantizedQueryScorer::new(&values, sim, raw).unwrap();
            merge.set_scoring_ordinal(query_ord as i32).unwrap();
            let expected = search.score(target_ord).unwrap();
            let actual = merge.score(target_ord).unwrap();
            assert!(
                (expected - actual).abs() <= 1e-5,
                "{encoding:?} query ord {query_ord} target ord {target_ord}: \
                     {expected} vs {actual}"
            );
        }
        // A zero vector cannot be normalized; Java's `l2normalize` throws.
        assert!(binarized_query_data(&values, sim, &vec![0.0; dim]).is_err());
    }
}

#[test]
fn empty_field_and_bad_inputs() {
    let (veq, vemq) = write(
        ScalarEncoding::UnsignedByte,
        VectorSimilarityFunction::Euclidean,
        &[],
        &[],
        4,
        10,
    );
    let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").unwrap();
    let values = r.quantized_vector_values(3).unwrap();
    assert_eq!(values.size(), 0);
    assert!(r.centroid(3).is_none());
    assert!(matches!(
        r.quantized_vector_values(4),
        Err(Error::UnknownField(4))
    ));
    assert!(values.vector(0).is_err());
    assert!(values.ord_to_doc(0).is_err());
    let q = QuantizedQueryScorer::new(&values, VectorSimilarityFunction::Euclidean, &[0.0; 3]);
    assert!(q.is_err());
    assert!(merge_scorer_supplier(&values, VectorSimilarityFunction::Euclidean, None).is_ok());

    let mut w = ScalarQuantizedVectorsWriter::new(ScalarEncoding::UnsignedByte, 10, &ID, "");
    let bad = QuantizedVectorsField {
        field_number: 0,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 2,
        docs: &[1, 1],
        vectors: &[0.0; 4],
    };
    assert!(w.write_field(&bad).is_err());
    let bad_len = QuantizedVectorsField { docs: &[1], ..bad };
    assert!(w.write_field(&bad_len).is_err());
    let bad_dim = QuantizedVectorsField {
        dimension: 0,
        ..bad
    };
    assert!(w.write_field(&bad_dim).is_err());

    // Corruption: flip a byte of the meta and re-sign nothing -> footer fails.
    let (veq, mut vemq) = write(
        ScalarEncoding::PackedNibble,
        VectorSimilarityFunction::DotProduct,
        &[0, 1],
        &[0.5, 0.1, -0.3, 0.2],
        2,
        2,
    );
    let n = vemq.len();
    vemq[n - 20] ^= 1;
    assert!(ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "suf").is_err());
    assert!(ScalarQuantizedVectorsReader::open(&[], &veq, &ID, "suf").is_err());
}

#[test]
fn merge_recomputes_or_weights_centroids() {
    let dim = 4usize;
    let a = vectors(10, dim, 1);
    let b = vectors(6, dim, 2);
    let mut merged = a.clone();
    merged.extend_from_slice(&b);
    let docs: Vec<i32> = (0..16).collect();
    let centroid = |v: &[f32]| -> Vec<f32> {
        let n = v.len() / dim;
        let mut c = vec![0f32; dim];
        for x in v.chunks(dim) {
            for (s, &y) in c.iter_mut().zip(x) {
                *s += y;
            }
        }
        c.iter().map(|s| s / n as f32).collect()
    };
    let (ca, cb) = (centroid(&a), centroid(&b));
    for (has_del, missing) in [(false, false), (true, false), (false, true)] {
        let sources = [
            QuantizedMergeSource {
                centroid: if missing { None } else { Some(&ca) },
                vectors: &a,
                has_deletions: has_del,
                live: None,
            },
            QuantizedMergeSource {
                centroid: Some(&cb),
                vectors: &b,
                has_deletions: false,
                live: None,
            },
        ];
        for sim in [
            VectorSimilarityFunction::Euclidean,
            VectorSimilarityFunction::Cosine,
        ] {
            let mut w = ScalarQuantizedVectorsWriter::new(ScalarEncoding::SevenBit, 16, &ID, "");
            w.merge_field(1, sim, dim as i32, &sources, &docs, &merged)
                .unwrap();
            let (veq, vemq) = w.finish();
            let r = ScalarQuantizedVectorsReader::open(&vemq, &veq, &ID, "").unwrap();
            let got = r.centroid(1).unwrap();
            if sim == VectorSimilarityFunction::Euclidean {
                let want = if has_del || missing {
                    centroid(&merged)
                } else {
                    (0..dim)
                        .map(|j| (ca[j] * 10.0 + cb[j] * 6.0) / 16.0)
                        .collect()
                };
                assert_eq!(got, &want[..]);
            } else {
                assert!(vector_util::is_unit_vector(got));
            }
        }
    }
    let mut c = vec![0f32; dim];
    assert_eq!(
        merge_and_recalculate_centroids(&[], false, dim, &mut c).unwrap(),
        0
    );
}

#[test]
fn score_parts_follow_java() {
    assert_eq!(java_max(-0.0, 0.0).to_bits(), 0f32.to_bits());
    assert!(java_max(f32::NAN, 0.0).is_nan());
    assert_eq!(java_max(0.0, -0.0).to_bits(), 0f32.to_bits());
    assert_eq!(
        util_similarity(VectorSimilarityFunction::MaximumInnerProduct),
        quantization::VectorSimilarityFunction::MaximumInnerProduct
    );
    let mut v = vec![1u8, 2, 3];
    assert_eq!(align(&mut v, 4), 4);
    assert_eq!(v, [1, 2, 3, 0]);
}

/// `.vemq` field records `readField`/`validateFieldEntry` refuse, each
/// written with a valid header and footer around it: a negative field
/// number, a non-positive dimension, a negative region, an unknown
/// scalar-encoding wire number, a centroid past the end of the file, a
/// data length that is not `size * (packed + 16)`, a region past the end
/// of `.veq` -- and a `BYTE` field asked for its float vectors.
#[test]
fn field_records_out_of_range_are_corrupt() {
    use lucene_store::data_output::DataOutput;
    let (veq, _) = write(
        ScalarEncoding::UnsignedByte,
        VectorSimilarityFunction::Euclidean,
        &[0],
        &[0.5, -0.5],
        2,
        1,
    );
    let meta = |record: &dyn Fn(&mut Vec<u8>)| -> Vec<u8> {
        let mut m = Vec::new();
        codec_util::write_index_header(&mut m, META_CODEC, VERSION_CURRENT, &ID, "suf");
        record(&mut m);
        m.write_i32(-1);
        codec_util::write_footer(&mut m);
        m
    };
    // field, encoding, similarity, dimension, offset, length, size.
    let head = |m: &mut Vec<u8>, field: i32, enc: i32, dim: i32, off: i64, len: i64, size: i32| {
        m.write_i32(field);
        m.write_i32(enc);
        m.write_i32(0);
        m.write_vint(dim);
        m.write_vlong(off);
        m.write_vlong(len);
        m.write_vint(size);
    };
    // A one-vector field's tail: wire 0 (unsigned byte), a two-float
    // centroid, its dot product, and a dense ordToDoc.
    let tail = |m: &mut Vec<u8>| {
        m.write_vint(0);
        for _ in 0..3 {
            m.write_i32(0);
        }
        m.write_i64(-1);
        m.write_i64(0);
        m.write_i16(-1);
        m.write_byte(0xff);
    };
    let open = |m: &[u8]| ScalarQuantizedVectorsReader::open(m, &veq, &ID, "suf");
    let err = |m: Vec<u8>| open(&m).err().map(|e| e.to_string()).unwrap_or_default();
    let header_end = codec_util::index_header_length(META_CODEC, "suf");
    let data_start = codec_util::index_header_length(DATA_CODEC, "suf") as i64;
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("Invalid field number", meta(&|m| m.write_i32(-5))),
        (
            "illegal vector dimension",
            meta(&|m| head(m, 3, 1, 0, 0, 0, 0)),
        ),
        (
            "illegal quantized vector region",
            meta(&|m| head(m, 3, 1, 2, 0, 0, -1)),
        ),
        (
            "Could not get ScalarEncoding",
            meta(&|m| {
                head(m, 3, 1, 2, 0, 0, 1);
                m.write_vint(99);
            }),
        ),
        (
            "not matching",
            meta(&|m| {
                head(m, 3, 1, 2, data_start, 7, 1);
                tail(m);
            }),
        ),
        (
            "past the end",
            meta(&|m| {
                head(m, 3, 1, 2, 1 << 40, 18, 1);
                tail(m);
            }),
        ),
    ];
    for (want, m) in cases {
        assert!(m.len() > header_end);
        let e = err(m);
        assert!(e.contains(want), "{want}: {e}");
    }
    // A centroid wider than what is left of the file.
    let wide = meta(&|m| {
        head(m, 3, 1, 1 << 20, 0, 0, 1);
        m.write_vint(0);
    });
    assert!(matches!(open(&wide), Err(Error::Store(_))));
    // A valid empty BYTE field: its float vectors are not there.
    let byte_field = meta(&|m| {
        head(m, 3, 0, 2, data_start, 0, 0);
        m.write_i64(-2);
        m.write_i64(0);
        m.write_i16(-1);
        m.write_byte(0xff);
    });
    let r = open(&byte_field).unwrap();
    assert_eq!(r.fields().len(), 1);
    assert!(matches!(
        r.quantized_vector_values(3),
        Err(Error::EncodingMismatch(
            3,
            VectorEncoding::Byte,
            VectorEncoding::Float32
        ))
    ));
}

/// A recomputed merge centroid is the mean of the live vectors only
/// (Lucene 10.6, #16764): deleted vectors do not pull it.
#[test]
fn recomputed_centroid_skips_deleted_vectors() {
    let dim = 2;
    let vectors = [1.0f32, 1.0, 100.0, 100.0, 3.0, 3.0];
    let live = [true, false, true];
    let sources = [QuantizedMergeSource {
        centroid: None,
        vectors: &vectors,
        has_deletions: true,
        live: Some(&live),
    }];
    let mut c = vec![0f32; dim];
    let count = merge_and_recalculate_centroids(&sources, false, dim, &mut c).unwrap();
    assert_eq!(count, 2);
    assert_eq!(c, [2.0, 2.0]);
}
