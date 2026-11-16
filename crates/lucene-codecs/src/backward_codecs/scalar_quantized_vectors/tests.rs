#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::backward_codecs::quantized_vectors::test_support::{fixture_group, Group};
use crate::field_infos::FieldInfo;
use lucene_store::data_output::DataOutput;

const ID: [u8; ID_LENGTH] = *b"legacy-sq-000001";

/// `compressBytes`: the writer's half of [`decompress_bytes`].
fn compress_bytes(raw: &[u8], compressed: &mut [u8]) {
    let n = compressed.len();
    for (i, c) in compressed.iter_mut().enumerate() {
        *c = (raw[i] << 4) | raw[n + i];
    }
}

/// One field for [`write`], every vector quantized with `quantizer`.
struct TestField {
    number: i32,
    encoding: VectorEncoding,
    sim: VectorSimilarityFunction,
    dim: usize,
    bits: u8,
    compress: bool,
    docs: Vec<i32>,
    max_doc: i32,
    vectors: Vec<f32>,
    /// `confidenceInterval` as written (version 0 insists on a real one).
    confidence_bits: i32,
    /// Added to the recorded data offset and length.
    offset_delta: i64,
    length_delta: i64,
}

fn field(
    number: i32,
    sim: VectorSimilarityFunction,
    dim: usize,
    bits: u8,
    compress: bool,
) -> TestField {
    let n = 30usize;
    let mut s = 17u64 + number as u64;
    let vectors = (0..n * dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 1.8 - 0.9
        })
        .collect();
    TestField {
        number,
        encoding: VectorEncoding::Float32,
        sim,
        dim,
        bits,
        compress,
        docs: (0..n as i32).map(|d| d * 2 + 1).collect(),
        max_doc: 2 * n as i32 + 1,
        vectors,
        confidence_bits: 0.95f32.to_bits() as i32,
        offset_delta: 0,
        length_delta: 0,
    }
}

/// A minimal `Lucene99ScalarQuantizedVectorsWriter`: the quantizer's
/// `quantize` per vector, codes compressed as `compressBytes` does.
fn write(version: i32, fields: &[TestField]) -> (Vec<u8>, Vec<u8>) {
    let mut meta = Vec::new();
    let mut data = Vec::new();
    codec_util::write_index_header(&mut meta, META_CODEC, version, &ID, "s");
    codec_util::write_index_header(&mut data, DATA_CODEC, version, &ID, "s");
    for f in fields {
        let quantizer = ScalarQuantizer::new(-1.0, 1.0, f.bits).unwrap();
        let offset = data.len() as i64;
        let n = f.docs.len();
        for v in f.vectors.chunks(f.dim.max(1)).take(n) {
            let mut codes = vec![0u8; f.dim];
            let correction = quantize_query(v, &mut codes, f.sim, &quantizer).unwrap();
            if f.compress {
                // An odd dimension is padded, which Java's writer cannot
                // even do -- only to build codes its reader rejects.
                codes.resize(f.dim.next_multiple_of(2), 0);
                let mut packed = vec![0u8; f.dim.div_ceil(2)];
                compress_bytes(&codes, &mut packed);
                data.extend_from_slice(&packed);
            } else {
                data.extend_from_slice(&codes);
            }
            data.extend_from_slice(&correction.to_le_bytes());
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
        meta.write_vlong(offset + f.offset_delta);
        meta.write_vlong(length + f.length_delta);
        meta.write_vint(f.dim as i32);
        meta.write_i32(n as i32);
        if n > 0 {
            meta.write_i32(f.confidence_bits);
            if version >= VERSION_ADD_BITS {
                meta.write_byte(f.bits);
                meta.write_byte(u8::from(f.compress));
            }
            meta.write_i32(quantizer.lower_quantile().to_bits() as i32);
            meta.write_i32(quantizer.upper_quantile().to_bits() as i32);
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

fn open<'a>(meta: &[u8], data: &'a [u8]) -> Result<Lucene99ScalarQuantizedVectorsReader<'a>> {
    Lucene99ScalarQuantizedVectorsReader::open(meta, data, &ID, "s")
}

fn err_of<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.unwrap_err().to_string()
}

/// The fixture's `Lucene99HnswScalarQuantizedVectorsFormat` groups.
fn groups(version: &str) -> Vec<Group> {
    (0..10)
        .filter_map(|i| fixture_group(version, "_0", HNSW_NAME, i))
        .chain(fixture_group(version, "_0", NAME, 0))
        .collect()
}

#[test]
fn every_fixture_group_opens_and_both_scorers_agree_bit_for_bit() {
    for (version, want_version) in [("9.9.2", 0), ("9.12.2", 1), ("10.2.2", 1)] {
        let groups = groups(version);
        assert!(groups.len() >= 5, "{version}: {} groups", groups.len());
        let mut saw_compressed = false;
        for g in &groups {
            let r = Lucene99ScalarQuantizedVectorsReader::open(
                g.file("vemq"),
                g.file("veq"),
                &g.id,
                &g.suffix,
            )
            .unwrap();
            assert_eq!(r.version(), want_version);
            r.check_integrity().unwrap();
            for entry in r.fields().to_vec() {
                if entry.vector_encoding == VectorEncoding::Byte {
                    // A BYTE field has no `.vemq` entry in Java; if one
                    // were there, the float API would refuse it.
                    assert!(r.quantized_vector_values(entry.field_number).is_err());
                    continue;
                }
                let n = entry.field_number;
                let values = r.quantized_vector_values(n).unwrap();
                saw_compressed |= values.is_compressed();
                assert_eq!(values.similarity(), entry.similarity);
                assert_eq!(values.scalar_quantizer(), entry.scalar_quantizer.as_ref());
                assert_eq!(r.quantization_state(n).unwrap(), entry.scalar_quantizer);
                let target: Vec<f32> = (0..values.dimension())
                    .map(|k| ((k + 1) as f64).sin() as f32)
                    .collect();
                let mut fast = r.scorer(n, &target).unwrap().unwrap();
                let mut generic =
                    ScalarQuantizedVectorScorer::new(values.clone(), &target).unwrap();
                let mut ords = Lucene99ScalarQuantizedOrdScorer::new(values.clone()).unwrap();
                let mut generic_ords =
                    ScalarQuantizedVectorScorer::for_ordinals(values.clone()).unwrap();
                assert_eq!(fast.max_ord(), values.size());
                assert_eq!(generic.max_ord(), values.size());
                assert_eq!(ords.max_ord(), values.size());
                assert_eq!(generic_ords.max_ord(), values.size());
                assert_eq!(fast.query().0.len(), values.dimension());
                let mut cursor = values.doc_to_ord().unwrap();
                let mut dq = vec![0f32; values.dimension()];
                for ord in 0..values.size() {
                    let a = fast.score(ord).unwrap();
                    assert_eq!(a.to_bits(), generic.score(ord).unwrap().to_bits());
                    let doc = values.ord_to_doc(ord).unwrap();
                    assert_eq!(fast.ord_to_doc(ord).unwrap(), doc);
                    assert_eq!(cursor.ordinal(doc).unwrap(), Some(ord));
                    values.dequantized(ord, &mut dq).unwrap();
                    assert!(dq.iter().all(|x| x.is_finite()));
                    // The merge-time scorers: ordinal `ord` against its
                    // neighbour, both classes, the same bits.
                    let other = (ord + 1) % values.size();
                    ords.set_scoring_ordinal(ord).unwrap();
                    generic_ords.set_scoring_ordinal(ord).unwrap();
                    assert_eq!(
                        ords.score(other).unwrap().to_bits(),
                        generic_ords.score(other).unwrap().to_bits()
                    );
                }
                assert!(values.ord_to_doc(values.size()).is_err());
                assert!(values.packed_vector(-1).is_err());
            }
        }
        if version != "9.9.2" {
            assert!(saw_compressed, "{version} has a compressed field");
        }
    }
}

#[test]
fn hand_built_segments_round_trip_at_both_versions() {
    let sims = [
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::MaximumInnerProduct,
    ];
    for version in [VERSION_START, VERSION_ADD_BITS] {
        let mut fields = Vec::new();
        for (i, sim) in sims.into_iter().enumerate() {
            fields.push(field(i as i32, sim, 8, 7, false));
            if version == VERSION_ADD_BITS {
                fields.push(field(10 + i as i32, sim, 8, 4, true));
                fields.push(field(20 + i as i32, sim, 8, 4, false));
            }
        }
        let (meta, data) = write(version, &fields);
        let r = open(&meta, &data).unwrap();
        r.check_field_infos(&infos(&fields)).unwrap();
        for f in &fields {
            let e = r.field(f.number).unwrap();
            assert_eq!((e.bits, e.compress), (f.bits, f.compress));
            let values = r.quantized_vector_values(f.number).unwrap();
            assert_eq!(values.vector_byte_length(), if f.compress { 4 } else { 8 });
            let q = &f.vectors[..f.dim];
            let mut s = r.scorer(f.number, q).unwrap().unwrap();
            // The first stored vector is the query: the best score, as
            // far as quantization allows, and a finite one.
            let first = s.score(0).unwrap();
            assert!(first.is_finite() && first >= 0.0, "{:?} {first}", f.sim);
            let mut codes = Vec::new();
            values.vector_into(0, &mut codes).unwrap();
            assert_eq!(codes, s.query().0);
            assert_eq!(values.ord_to_doc(3).unwrap(), 7);
        }
    }
}

#[test]
fn empty_fields_have_no_quantizer_and_no_scorer() {
    let mut f = field(4, VectorSimilarityFunction::Euclidean, 8, 7, false);
    f.docs.clear();
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    let r = open(&meta, &data).unwrap();
    let e = r.field(4).unwrap();
    assert!(e.scalar_quantizer.is_none());
    assert!(e.ord_to_doc.is_empty());
    assert!(r.scorer(4, &[0.0; 8]).unwrap().is_none());
    let values = r.quantized_vector_values(4).unwrap();
    assert_eq!(values.size(), 0);
    assert!(values.dequantized(0, &mut [0.0; 8]).is_err());
    assert!(Lucene99ScalarQuantizedScorer::new(values.clone(), &[0.0; 8]).is_err());
    assert!(Lucene99ScalarQuantizedOrdScorer::new(values.clone()).is_err());
    assert!(ScalarQuantizedVectorScorer::new(values.clone(), &[0.0; 8]).is_err());
    assert!(ScalarQuantizedVectorScorer::for_ordinals(values).is_err());
}

#[test]
fn caller_mistakes_are_errors() {
    let fields = [field(1, VectorSimilarityFunction::Cosine, 8, 7, false)];
    let (meta, data) = write(VERSION_ADD_BITS, &fields);
    let r = open(&meta, &data).unwrap();
    assert!(matches!(
        r.quantized_vector_values(9),
        Err(Error::UnknownField(9))
    ));
    assert!(matches!(
        r.scorer(1, &[0.5; 3]),
        Err(Error::QueryDimensionMismatch(3, 8))
    ));
    // COSINE normalizes the query, and a zero vector cannot be.
    assert!(r.scorer(1, &[0.0; 8]).is_err());

    let mut other = infos(&fields);
    other.fields[0].vector_similarity_function = VectorSimilarityFunction::Euclidean;
    assert!(err_of(r.check_field_infos(&other)).contains("similarity"));
    let mut other = infos(&fields);
    other.fields[0].vector_dimension = 9;
    assert!(err_of(r.check_field_infos(&other)).contains("dimension"));
    let empty = FieldInfos::new(Vec::new()).unwrap();
    assert!(err_of(r.check_field_infos(&empty)).contains("Invalid field number"));

    let mut byte = field(2, VectorSimilarityFunction::Euclidean, 8, 7, false);
    byte.encoding = VectorEncoding::Byte;
    let (meta, data) = write(VERSION_ADD_BITS, &[byte]);
    let r = open(&meta, &data).unwrap();
    assert!(matches!(
        r.quantized_vector_values(2),
        Err(Error::EncodingMismatch(
            2,
            VectorEncoding::Byte,
            VectorEncoding::Float32
        ))
    ));
    assert!(r.quantization_state(2).is_err());
}

#[test]
fn corrupt_metadata_is_rejected() {
    let good = || field(1, VectorSimilarityFunction::DotProduct, 8, 7, false);
    // Version 0: a null (-1) or dynamic (0) confidence interval.
    for bits in [-1, 0] {
        let mut f = good();
        f.confidence_bits = bits;
        let (meta, data) = write(VERSION_START, &[f]);
        assert!(err_of(open(&meta, &data)).contains("confidence interval"));
    }
    // Version 1 ignores it.
    let mut f = good();
    f.confidence_bits = -1;
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    open(&meta, &data).unwrap();

    // Illegal bits: rewrite the bits byte of an otherwise good entry.
    let (meta, data) = write(VERSION_ADD_BITS, &[good()]);
    let mut entry = Vec::new();
    entry.write_i32(1);
    entry.write_i32(1);
    entry.write_i32(1);
    let at = meta.windows(entry.len()).position(|w| w == entry).unwrap();
    // encoding, similarity, offset/length vlongs, dim vint, size int,
    // confidence int, then bits.
    let mut probe = SliceInput::new(&meta[at + 12..]);
    probe.read_vlong().unwrap();
    probe.read_vlong().unwrap();
    probe.read_vint().unwrap();
    probe.read_i32().unwrap();
    probe.read_i32().unwrap();
    let bits_at = at + 12 + probe.position();
    let mut bad = meta.clone();
    bad[bits_at] = 9;
    let payload = bad.len() - codec_util::FOOTER_LENGTH;
    bad.truncate(payload);
    codec_util::write_footer(&mut bad);
    assert!(err_of(open(&bad, &data)).contains("illegal quantization bits"));

    // A length that is not size * (dim + 4), and a region past the end.
    let mut f = good();
    f.length_delta = 1;
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    assert!(err_of(open(&meta, &data)).contains("not matching"));
    let mut f = good();
    f.offset_delta = 1 << 20;
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    assert!(err_of(open(&meta, &data)).contains("past the end"));

    // Negative and duplicate field numbers, and dimension 0.
    let mut f = good();
    f.number = -5;
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
    let (meta, data) = write(VERSION_ADD_BITS, &[good(), good()]);
    assert!(err_of(open(&meta, &data)).contains("Invalid field number"));
    let mut f = good();
    f.dim = 0;
    f.docs.clear();
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    assert!(err_of(open(&meta, &data)).contains("illegal quantized vector entry"));

    // Data and meta disagreeing on the version; a truncated data footer;
    // a meta shorter than a footer.
    let (meta, _) = write(VERSION_ADD_BITS, &[good()]);
    let (_, data0) = write(VERSION_START, &[good()]);
    assert!(err_of(open(&meta, &data0)).contains("Format versions mismatch"));
    let (meta, data) = write(VERSION_ADD_BITS, &[good()]);
    assert!(open(&meta, &data[..data.len() - 3]).is_err());
    assert!(open(&meta[..10], &data).is_err());
    // `checkIntegrity` catches a flipped payload byte `retrieveChecksum`
    // does not look at.
    let mut flipped = data.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x40;
    let r = open(&meta, &flipped).unwrap();
    assert!(r.check_integrity().is_err());
}

#[test]
fn decompress_is_compress_reversed_and_rejects_odd_lengths() {
    let raw: Vec<u8> = (0..10u8).map(|i| i % 16).collect();
    let mut packed = [0u8; 5];
    compress_bytes(&raw, &mut packed);
    assert_eq!(packed[0], raw[0] << 4 | raw[5]);
    let mut back = [0u8; 10];
    decompress_bytes(&packed, &mut back).unwrap();
    assert_eq!(&back[..], &raw[..]);
    assert!(decompress_bytes(&packed, &mut [0u8; 9]).is_err());
    // An odd dimension with compression: Java's `decompressBytes` throws.
    let mut f = field(1, VectorSimilarityFunction::Euclidean, 7, 4, true);
    f.vectors = f.vectors.iter().copied().chain([0.0; 30]).collect();
    let (meta, data) = write(VERSION_ADD_BITS, &[f]);
    let r = open(&meta, &data).unwrap();
    let values = r.quantized_vector_values(1).unwrap();
    assert!(values.vector_into(0, &mut Vec::new()).is_err());
}

#[test]
fn java_max_is_javas() {
    assert!(java_max(f32::NAN, 0.0).is_nan());
    assert!(java_max(0.0, f32::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(0.0, -0.0).is_sign_positive());
    assert_eq!(java_max(-2.0, 0.0), 0.0);
    assert_eq!(java_max(3.0, 0.0), 3.0);
}
