#![cfg(test)]
use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::hnsw::VectorScorer;
use lucene_codecs::per_field_knn_vectors::{KnnVectorsFormat, IVF_QUANTIZERS};
use lucene_store::codec_util::FOOTER_LENGTH;
use lucene_store::directory::Input;
use lucene_store::DataOutput;

use super::{centroid, write, CodesField, HnswCodesReader, CODES_ALIGNMENT};

const ID: [u8; 16] = [7; 16];
const SUFFIX: &str = "HnswQuantizedVectorsFormat_0";

fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n * dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn format(name: &str) -> KnnVectorsFormat {
    let q = IVF_QUANTIZERS.iter().position(|&n| n == name).unwrap();
    KnnVectorsFormat::hnsw_quantized(q as u8, 16, 100).unwrap()
}

/// `(meta, data)` of two fields: 300 x 12 Euclidean (number 1), 40 x 33
/// cosine (number 4).
fn files(name: &str) -> (Vec<u8>, Vec<u8>) {
    let a = vectors(300, 12, 1);
    let b = vectors(40, 33, 2);
    let fields = [
        CodesField {
            number: 1,
            similarity: VectorSimilarityFunction::Euclidean,
            dimension: 12,
            vectors: &a,
        },
        CodesField {
            number: 4,
            similarity: VectorSimilarityFunction::Cosine,
            dimension: 33,
            vectors: &b,
        },
    ];
    let mut out = write(format(name), &fields, &ID, SUFFIX).unwrap().unwrap();
    assert_eq!(out[0].0, "vqm");
    assert_eq!(out[1].0, "vqd");
    let data = out.pop().unwrap().1;
    (out.pop().unwrap().1, data)
}

fn open(meta: &[u8], data: Vec<u8>) -> super::Result<HnswCodesReader> {
    HnswCodesReader::open(meta, Arc::new(Input::Owned(data)), &ID, SUFFIX)
}

#[test]
fn codes_round_trip_and_score_like_the_quantizer() {
    for name in ["rabitq", "erq2", "erq4", "erq7", "osq4"] {
        let (meta, data) = files(name);
        let reader = open(&meta, data).unwrap();
        reader.check_integrity().unwrap();
        assert!(reader.field(2).is_none());
        for (number, n, dim, seed) in [(1, 300, 12, 1), (4, 40, 33, 2)] {
            let field = reader.field(number).unwrap();
            assert_eq!(field.count(), n);
            let codes = reader.codes(field);
            assert_eq!(codes.len(), n * field.code_len());
            let all = vectors(n, dim, seed);
            let query = &all[..dim];
            let mut scorer = reader.scorer(field, query);
            assert_eq!(scorer.max_ord(), n as i32);
            let direct = field.query_scorer(query);
            for ord in [0, n as i32 / 2, n as i32 - 1] {
                let at = ord as usize * field.code_len();
                let code = &codes[at..at + field.code_len()];
                assert_eq!(scorer.score(ord).unwrap(), direct.score(code), "{name}");
            }
            scorer.prefetch(&[0, 1, -1, n as i32]);
            assert!(scorer.score(n as i32).is_err());
            assert!(scorer.score(-1).is_err());
            assert!(format!("{scorer:?}").contains("CodeScorer"));
        }
    }
}

#[test]
fn each_fields_codes_start_aligned() {
    let (meta, data) = files("erq3");
    let reader = open(&meta, data.clone()).unwrap();
    for number in [1, 4] {
        let codes = reader.codes(reader.field(number).unwrap());
        let at = codes.as_ptr() as usize - reader.data.as_ptr() as usize;
        assert_eq!(at % CODES_ALIGNMENT, 0);
    }
}

#[test]
fn other_formats_and_empty_fields_write_nothing() {
    let v = vectors(4, 4, 3);
    let field = CodesField {
        number: 0,
        similarity: VectorSimilarityFunction::DotProduct,
        dimension: 4,
        vectors: &v,
    };
    let hnsw = KnnVectorsFormat::hnsw(16, 100).unwrap();
    assert!(write(hnsw, &[field], &ID, SUFFIX).unwrap().is_none());
    let empty = CodesField {
        number: 0,
        similarity: VectorSimilarityFunction::DotProduct,
        dimension: 4,
        vectors: &[],
    };
    assert!(write(format("erq2"), &[empty], &ID, SUFFIX)
        .unwrap()
        .is_none());
    let ragged = CodesField {
        number: 0,
        similarity: VectorSimilarityFunction::DotProduct,
        dimension: 3,
        vectors: &v,
    };
    assert!(write(format("erq2"), &[ragged], &ID, SUFFIX).is_err());
    let unknown = KnnVectorsFormat::HnswQuantized {
        quantizer: 200,
        max_conn: 16,
        beam_width: 100,
    };
    assert!(write(unknown, &[], &ID, SUFFIX).is_err());
    let auto = KnnVectorsFormat::HnswQuantized {
        quantizer: lucene_codecs::per_field_knn_vectors::IVF_AUTO_QUANTIZER,
        max_conn: 16,
        beam_width: 100,
    };
    assert!(write(auto, &[], &ID, SUFFIX).is_err());
}

#[test]
fn the_centroid_is_the_mean_of_the_prepared_vectors() {
    let v = [3.0, 4.0, 0.0, 0.0, 0.0, 2.0];
    assert_eq!(centroid(&v, 2, false), vec![1.0, 2.0]);
    // Unit-normalized first; a zero vector counts as zero.
    assert_eq!(centroid(&v, 2, true), vec![0.2, 0.6]);
    assert_eq!(centroid(&[], 2, false), vec![0.0, 0.0]);
}

#[test]
fn damaged_files_are_refused() {
    let (meta, data) = files("erq2");
    // A truncated or footless data file, another segment's files.
    assert!(open(&meta, data[..data.len() - 1].to_vec()).is_err());
    assert!(open(&meta, data[..FOOTER_LENGTH - 1].to_vec()).is_err());
    assert!(open(&meta, data[..20].to_vec()).is_err());
    assert!(HnswCodesReader::open(
        &meta,
        Arc::new(Input::Owned(data.clone())),
        &[8; 16],
        SUFFIX
    )
    .is_err());
    // A flipped meta byte breaks the footer's checksum (or a field).
    let mut bad = meta.clone();
    let last = bad.len() - FOOTER_LENGTH - 1;
    bad[last] ^= 0xff;
    assert!(open(&bad, data.clone()).is_err());
    // Codes that do not fit the data file.
    let short = {
        let header = lucene_store::codec_util::index_header_length(super::DATA_CODEC, SUFFIX);
        let mut d = data[..header].to_vec();
        lucene_store::codec_util::write_footer(&mut d);
        d
    };
    assert!(open(&meta, short).is_err());
}

/// One hand-written meta entry: number, similarity, dimension, quantizer,
/// state, count, code length and data offset.
type MetaField<'a> = (i32, u8, i32, &'a str, Vec<u8>, i32, i32, i64);

/// A meta file written by hand from the given entries.
fn meta_with(fields: &[MetaField<'_>], version: i32) -> Vec<u8> {
    let mut meta = Vec::new();
    lucene_store::codec_util::write_index_header(
        &mut meta,
        super::META_CODEC,
        version,
        &ID,
        SUFFIX,
    );
    meta.write_vint(fields.len() as i32);
    for (number, sim, dim, name, state, count, code_len, offset) in fields {
        meta.write_vint(*number);
        meta.write_byte(*sim);
        meta.write_vint(*dim);
        meta.write_string(name);
        meta.write_vint(state.len() as i32);
        meta.write_bytes(state);
        for _ in 0..*dim {
            meta.write_bytes(&0f32.to_le_bytes());
        }
        meta.write_vint(*count);
        meta.write_vint(*code_len);
        meta.write_vlong(*offset);
    }
    lucene_store::codec_util::write_footer(&mut meta);
    meta
}

#[test]
fn inconsistent_metadata_is_refused() {
    let (meta, data) = files("erq2");
    let reader = open(&meta, data.clone()).unwrap();
    let f = reader.field(1).unwrap();
    let state = f.quantizer().state();
    let (len, offset) = (f.code_len() as i32, f.offset as i64);
    let good = (1, 0u8, 12, "erq", state.clone(), 300, len, offset);
    assert!(open(&meta_with(std::slice::from_ref(&good), 0), data.clone()).is_ok());
    for bad in [
        (1, 9u8, 12, "erq", state.clone(), 300, len, offset),
        (1, 0, 12, "pq", state.clone(), 300, len, offset),
        (1, 0, 12, "erq", vec![1], 300, len, offset),
        (1, 0, 12, "erq", state.clone(), 300, len + 1, offset),
        (1, 0, 12, "erq", state.clone(), 30_000, len, offset),
        (1, 0, 12, "erq", state.clone(), 300, len, 3),
        (1, 0, 12, "erq", state.clone(), -1, len, offset),
        (1, 0, -12, "erq", state.clone(), 300, len, offset),
    ] {
        assert!(
            open(&meta_with(std::slice::from_ref(&bad), 0), data.clone()).is_err(),
            "{bad:?}"
        );
    }
    assert!(open(&meta_with(&[good.clone(), good.clone()], 0), data.clone()).is_err());
    assert!(open(&meta_with(&[good], 1), data).is_err());
}
