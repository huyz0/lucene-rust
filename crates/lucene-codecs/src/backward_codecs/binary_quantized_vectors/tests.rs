#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::backward_codecs::quantized_vectors::test_support::fixture_group;
use crate::field_infos::FieldInfo;
use crate::vectors::FlatVectorsReader;
use lucene_store::data_output::DataOutput;

const ID: [u8; ID_LENGTH] = *b"binary-quant-001";

struct TestField {
    number: i32,
    encoding: VectorEncoding,
    sim: VectorSimilarityFunction,
    dim: usize,
    docs: Vec<i32>,
    max_doc: i32,
    vectors: Vec<f32>,
    dim_override: Option<i32>,
    offset_delta: i64,
    length_delta: i64,
}

fn field(number: i32, sim: VectorSimilarityFunction, dim: usize) -> TestField {
    let n = 25usize;
    let mut s = 91u64 + number as u64;
    let vectors = (0..n * dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    TestField {
        number,
        encoding: VectorEncoding::Float32,
        sim,
        dim,
        docs: (0..n as i32).map(|d| d * 3).collect(),
        max_doc: 3 * n as i32,
        vectors,
        dim_override: None,
        offset_delta: 0,
        length_delta: 0,
    }
}

/// A minimal `Lucene102BinaryQuantizedVectorsWriter`: the centroid is the
/// mean, each vector one-bit quantized against it and packed.
fn write(fields: &[TestField]) -> (Vec<u8>, Vec<u8>) {
    let mut meta = Vec::new();
    let mut data = Vec::new();
    codec_util::write_index_header(&mut meta, META_CODEC, VERSION_CURRENT, &ID, "b");
    codec_util::write_index_header(&mut data, DATA_CODEC, VERSION_CURRENT, &ID, "b");
    for f in fields {
        let n = f.docs.len();
        let mut centroid = vec![0f32; f.dim];
        let normalized: Vec<Vec<f32>> = f
            .vectors
            .chunks(f.dim.max(1))
            .take(n)
            .map(|v| {
                let mut v = v.to_vec();
                if f.sim == VectorSimilarityFunction::Cosine {
                    vector_util::l2normalize(&mut v, true).unwrap();
                }
                v
            })
            .collect();
        for v in &normalized {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += x / n as f32;
            }
        }
        let quantizer = OptimizedScalarQuantizer::new(util_similarity(f.sim));
        let offset = data.len() as i64;
        for v in &normalized {
            let mut v = v.clone();
            let mut scratch = vec![0u8; quantization::discretize(f.dim, 64)];
            let r = quantizer.scalar_quantize(&mut v, &mut scratch, INDEX_BITS, &centroid);
            let mut packed = vec![0u8; binary_bytes(f.dim)];
            quantization::pack_as_binary(&scratch, &mut packed);
            data.extend_from_slice(&packed);
            data.write_i32(r.lower_interval.to_bits() as i32);
            data.write_i32(r.upper_interval.to_bits() as i32);
            data.write_i32(r.additional_correction.to_bits() as i32);
            data.write_i16(r.quantized_component_sum as i16);
        }
        let length = data.len() as i64 - offset;
        meta.write_i32(f.number);
        meta.write_i32(match f.encoding {
            VectorEncoding::Byte => 0,
            VectorEncoding::Float32 => 1,
        });
        meta.write_i32(match f.sim {
            VectorSimilarityFunction::Euclidean => 0,
            VectorSimilarityFunction::DotProduct => 1,
            VectorSimilarityFunction::Cosine => 2,
            VectorSimilarityFunction::MaximumInnerProduct => 3,
        });
        meta.write_vint(f.dim_override.unwrap_or(f.dim as i32));
        meta.write_vlong(offset + f.offset_delta);
        meta.write_vlong(length + f.length_delta);
        meta.write_vint(n as i32);
        if n > 0 {
            for c in &centroid {
                meta.write_i32(c.to_bits() as i32);
            }
            let dp: f32 = centroid.iter().map(|c| c * c).sum();
            meta.write_i32(dp.to_bits() as i32);
        }
        crate::vectors::write_stored_meta(&mut meta, &mut data, &f.docs, f.max_doc);
    }
    meta.write_i32(-1);
    codec_util::write_footer(&mut meta);
    codec_util::write_footer(&mut data);
    (meta, data)
}

fn infos(fields: &[TestField]) -> FieldInfos {
    FieldInfos::new(
        fields
            .iter()
            .map(|f| {
                let mut fi = FieldInfo::new(format!("f{}", f.number), f.number);
                fi.vector_dimension = f.dim as i32;
                fi.vector_encoding = f.encoding;
                fi.vector_similarity_function = f.sim;
                fi
            })
            .collect(),
    )
    .unwrap()
}

fn open<'a>(meta: &[u8], data: &'a [u8]) -> Result<Lucene102BinaryQuantizedVectorsReader<'a>> {
    Lucene102BinaryQuantizedVectorsReader::open(meta, data, &ID, "b")
}

fn err_of<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.unwrap_err().to_string()
}

/// Every fixture group: the merge-time supplier over the raw vectors'
/// query side scores an ordinal pair exactly as the query scorer does
/// with that raw vector as its target -- the two quantize the same way.
#[test]
fn fixture_groups_open_and_the_merge_supplier_agrees_with_the_query_scorer() {
    let mut checked = 0;
    for format in [NAME, HNSW_NAME] {
        for n in 0..6 {
            for seg in ["_0", "_1"] {
                let Some(g) = fixture_group("10.2.2", seg, format, n) else {
                    continue;
                };
                let r = Lucene102BinaryQuantizedVectorsReader::open(
                    g.file("vemb"),
                    g.file("veb"),
                    &g.id,
                    &g.suffix,
                )
                .unwrap();
                r.check_integrity().unwrap();
                let flat = FlatVectorsReader::open(g.file("vemf"), g.file("vec"), &g.id, &g.suffix)
                    .unwrap();
                for e in r.fields().to_vec() {
                    let n = e.field_number;
                    assert_eq!(r.centroid(n).map(<[f32]>::len), Some(e.dimension as usize));
                    let values = r.binarized_vector_values(n).unwrap();
                    assert_eq!(values.similarity(), e.similarity);
                    assert_eq!(values.centroid().len(), values.dimension());
                    assert_eq!(values.discretized_dimensions() % 64, 0);
                    assert_eq!(
                        values.vector_byte_length() * 8,
                        values.discretized_dimensions()
                    );
                    assert!(values.centroid_dp() >= 0.0);
                    let raw = flat.float_vector_values(n).unwrap();
                    let dim = values.dimension();
                    let mut all = Vec::new();
                    for ord in 0..raw.size() {
                        all.extend(raw.vector(ord).unwrap());
                    }
                    let query_side = binarized_query_data(&values, &all).unwrap();
                    let mut supplier =
                        Lucene102BinaryQuantizedOrdScorer::new(values.clone(), &query_side);
                    let mut cursor = values.doc_to_ord().unwrap();
                    for ord in (0..values.size()).step_by(7) {
                        let target = &all[ord as usize * dim..(ord as usize + 1) * dim];
                        let mut q = r.scorer(n, target).unwrap();
                        assert_eq!(q.max_ord(), values.size());
                        assert_eq!(supplier.max_ord(), values.size());
                        supplier.set_scoring_ordinal(ord).unwrap();
                        for other in [0, ord, values.size() - 1] {
                            assert_eq!(
                                q.score(other).unwrap().to_bits(),
                                supplier.score(other).unwrap().to_bits()
                            );
                        }
                        let doc = values.ord_to_doc(ord).unwrap();
                        assert_eq!(q.ord_to_doc(ord).unwrap(), doc);
                        assert_eq!(doc, raw.ord_to_doc(ord).unwrap());
                        assert_eq!(cursor.ordinal(doc).unwrap(), Some(ord));
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 12, "{checked} fields");
}

#[test]
fn hand_built_fields_round_trip() {
    let fields = [
        field(0, VectorSimilarityFunction::Euclidean, 40),
        field(1, VectorSimilarityFunction::DotProduct, 64),
        field(2, VectorSimilarityFunction::Cosine, 70),
        field(3, VectorSimilarityFunction::MaximumInnerProduct, 9),
    ];
    let (meta, data) = write(&fields);
    let r = open(&meta, &data).unwrap();
    r.check_field_infos(&infos(&fields)).unwrap();
    for f in &fields {
        let values = r.binarized_vector_values(f.number).unwrap();
        assert_eq!(values.size(), 25);
        assert_eq!(values.ord_to_doc(4).unwrap(), 12);
        // Scores are in range, and the stored vector itself outranks the
        // average of the others.
        let target = &f.vectors[..f.dim];
        let mut s = r.scorer(f.number, target).unwrap();
        let own = s.score(0).unwrap();
        let mean: f32 = (1..25).map(|o| s.score(o).unwrap()).sum::<f32>() / 24.0;
        assert!(own > mean, "{:?}: {own} vs {mean}", f.sim);
        assert!((0..25).all(|o| s.score(o).unwrap() >= 0.0));
        assert_eq!(
            s.query().0.len(),
            usize::from(QUERY_BITS) * values.discretized_dimensions() / 8
        );
        assert!(values.vector(25).is_err());
        assert!(values.corrective_terms(-1).is_err());
    }
}

#[test]
fn empty_and_byte_fields() {
    let mut empty = field(5, VectorSimilarityFunction::Euclidean, 16);
    empty.docs.clear();
    let mut byte = field(6, VectorSimilarityFunction::Euclidean, 16);
    byte.encoding = VectorEncoding::Byte;
    byte.docs.clear();
    let (meta, data) = write(&[empty, byte]);
    let r = open(&meta, &data).unwrap();
    assert!(r.centroid(5).is_none());
    let values = r.binarized_vector_values(5).unwrap();
    assert_eq!(values.size(), 0);
    assert!(matches!(
        values.doc_to_ord().unwrap(),
        DocToOrdCursor::Empty
    ));
    assert!(matches!(
        r.binarized_vector_values(6),
        Err(Error::EncodingMismatch(
            6,
            VectorEncoding::Byte,
            VectorEncoding::Float32
        ))
    ));
    assert!(matches!(
        r.binarized_vector_values(7),
        Err(Error::UnknownField(7))
    ));
}

#[test]
fn caller_mistakes_are_errors() {
    let fields = [field(1, VectorSimilarityFunction::Cosine, 16)];
    let (meta, data) = write(&fields);
    let r = open(&meta, &data).unwrap();
    assert!(matches!(
        r.scorer(1, &[0.1; 3]),
        Err(Error::QueryDimensionMismatch(3, 16))
    ));
    assert!(r.scorer(1, &[0.0; 16]).is_err());
    let values = r.binarized_vector_values(1).unwrap();
    assert!(binarized_query_data(&values, &[0.5; 17]).is_err());
    let side = binarized_query_data(&values, &fields[0].vectors).unwrap();
    let mut s = Lucene102BinaryQuantizedOrdScorer::new(values, &side);
    assert!(s.score(0).is_err());
    assert!(s.set_scoring_ordinal(25).is_err());
    assert!(s.set_scoring_ordinal(-1).is_err());

    let mut other = infos(&fields);
    other.fields[0].vector_similarity_function = VectorSimilarityFunction::Euclidean;
    assert!(err_of(r.check_field_infos(&other)).contains("similarity"));
    let mut other = infos(&fields);
    other.fields[0].vector_dimension = 15;
    assert!(err_of(r.check_field_infos(&other)).contains("dimension"));
    assert!(
        err_of(r.check_field_infos(&FieldInfos::new(Vec::new()).unwrap()))
            .contains("Invalid field number")
    );
}

#[test]
fn corrupt_metadata_is_rejected() {
    let good = || field(1, VectorSimilarityFunction::DotProduct, 16);
    let mut f = good();
    f.length_delta = 1;
    let (meta, data) = write(&[f]);
    assert!(err_of(open(&meta, &data)).contains("not matching"));
    let mut f = good();
    f.offset_delta = 1 << 20;
    let (meta, data) = write(&[f]);
    assert!(err_of(open(&meta, &data)).contains("past the end"));
    let mut f = good();
    f.dim_override = Some(0);
    let (meta, data) = write(&[f]);
    assert!(err_of(open(&meta, &data)).contains("illegal binarized vector entry"));
    // A centroid longer than what is left of the file.
    let mut f = good();
    f.dim_override = Some(1 << 20);
    let (meta, data) = write(&[f]);
    assert!(open(&meta, &data).is_err());
    let mut f = good();
    f.number = -3;
    let (meta, data) = write(&[f]);
    assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
    let (meta, data) = write(&[good(), good()]);
    assert!(err_of(open(&meta, &data)).contains("Invalid field number"));

    let (meta, data) = write(&[good()]);
    assert!(open(&meta[..8], &data).is_err());
    assert!(open(&meta, &data[..data.len() - 2]).is_err());
    // The data file at another version than the metadata.
    let mut data1 = Vec::new();
    codec_util::write_index_header(&mut data1, DATA_CODEC, 1, &ID, "b");
    let header = data1.len();
    data1.extend_from_slice(&data[header..data.len() - codec_util::FOOTER_LENGTH]);
    codec_util::write_footer(&mut data1);
    assert!(open(&meta, &data1).is_err());
    let mut flipped = data.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x10;
    assert!(open(&meta, &flipped).unwrap().check_integrity().is_err());
}

#[test]
fn scores_clamp_as_java_does() {
    assert!(java_max(f32::NAN, 0.0).is_nan());
    assert!(java_max(0.0, f32::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(0.0, -0.0).is_sign_positive());
    assert_eq!(java_max(-1.0, 0.0), 0.0);
}
