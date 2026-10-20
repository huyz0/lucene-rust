// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::field_infos::VectorSimilarityFunction;

const ID: [u8; ID_LENGTH] = *b"bitvectorstests!";
const SUFFIX: &str = "HnswBitVectorsFormat_0";

fn rng_bytes(seed: &mut u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed as u8
        })
        .collect()
}

/// A byte field of `count` vectors, on every `every`-th document.
fn field(number: i32, dimension: i32, count: i32, every: i32, seed: &mut u64) -> FlatVectorsField {
    FlatVectorsField {
        field_number: number,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension,
        docs: (0..count).map(|i| i * every).collect(),
        values: FieldVectorData::Byte(rng_bytes(seed, (count * dimension) as usize)),
    }
}

fn hamming_score(a: &[u8], b: &[u8]) -> f32 {
    let bits = a.len() as i32 * 8;
    let differ: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
    (bits - differ as i32) as f32 / bits as f32
}

/// Every vector scored against `query`, best first, doc ascending on ties.
fn brute_force(field: &FlatVectorsField, query: &[u8], k: usize) -> Vec<(i32, f32)> {
    let FieldVectorData::Byte(values) = &field.values else {
        unreachable!()
    };
    let dim = field.dimension as usize;
    let mut all: Vec<(i32, f32)> = field
        .docs
        .iter()
        .enumerate()
        .map(|(ord, &doc)| {
            (
                doc,
                hamming_score(query, &values[ord * dim..(ord + 1) * dim]),
            )
        })
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

#[test]
fn names_and_parameters() {
    assert_eq!(FlatBitVectorsScorer.to_string(), "FlatBitVectorsScorer()");
    let format = HnswBitVectorsFormat::default();
    assert_eq!(
        format.to_string(),
        "HnswBitVectorsFormat(name=HnswBitVectorsFormat, maxConn=16, beamWidth=100, \
         flatVectorFormat=Lucene99FlatVectorsFormat(vectorsScorer=FlatBitVectorsScorer()))"
    );
    assert_eq!((format.max_conn(), format.beam_width()), (16, 100));
    assert_eq!(format.max_dimensions("any"), 1024);
    let custom = HnswBitVectorsFormat::new(512, 3200).unwrap();
    assert_eq!((custom.max_conn(), custom.beam_width()), (512, 3200));
    for (m, beam, msg) in [
        (
            0,
            10,
            "maxConn must be positive and less than or equal to 512; maxConn=0",
        ),
        (513, 10, "maxConn=513"),
        (
            16,
            0,
            "beamWidth must be positive and less than or equal to 3200; beamWidth=0",
        ),
        (16, 3201, "beamWidth=3201"),
    ] {
        let err = HnswBitVectorsFormat::new(m, beam).unwrap_err().to_string();
        assert!(err.contains(msg), "{err}");
    }
}

#[test]
fn write_refuses_float_fields() {
    let float = FlatVectorsField {
        field_number: 0,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 2,
        docs: vec![0],
        values: FieldVectorData::Float32(vec![1.0, 2.0]),
    };
    let err = HnswBitVectorsFormat::default()
        .write(&[float], 1, &ID, SUFFIX)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "HnswBitVectorsFormat only supports BYTE encoding"
    );
}

#[test]
fn search_matches_brute_force() {
    let mut seed = 0x1234_5678_9abc_def1;
    // `small` has no graph (exhaustive scan); `big` has one.
    let small = field(0, 4, 40, 3, &mut seed);
    let big = field(1, 16, 2_000, 1, &mut seed);
    let format = HnswBitVectorsFormat::new(8, 60).unwrap();
    let files = format
        .write(&[small.clone(), big.clone()], 2_000, &ID, SUFFIX)
        .unwrap();
    let reader = format.open(&files, &ID, SUFFIX).unwrap();
    assert!(reader.hnsw.graph(0).unwrap().is_none());
    assert!(reader.hnsw.graph(1).unwrap().is_some());
    let (mut recalled, mut wanted) = (0usize, 0usize);
    for (number, f) in [(0, &small), (1, &big)] {
        let dim = f.dimension as usize;
        for q in 0..20 {
            let query = rng_bytes(&mut seed, dim);
            // The exhaustive scan (`small`, and `k` past the graph's worth)
            // is exact.
            for k in [1usize, 5, 2_100] {
                let (hits, early) = reader
                    .search(number, &query, k, u64::MAX, SearchOptions::default())
                    .unwrap();
                assert!(!early);
                let want = brute_force(f, &query, k);
                if number == 0 || k == 2_100 {
                    assert_eq!(hits, want, "field {number} q {q} k {k}");
                } else {
                    // A graph walk is approximate (Java's too: the fixture
                    // test pins it hit for hit); here only its shape and
                    // every score are checked.
                    assert_eq!(hits.len(), k);
                    // Recall, tie-aware, with a wider beam (`k = 50`, as
                    // a caller oversampling would): a hit scoring at least
                    // the exact fifth score is one of an exact top five.
                    if k == 5 {
                        let (wide, _) = reader
                            .search(number, &query, 50, u64::MAX, SearchOptions::default())
                            .unwrap();
                        let floor = want[4].1;
                        recalled += wide[..5].iter().filter(|h| h.1 >= floor).count();
                        wanted += 5;
                    }
                    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
                    for (doc, score) in &hits {
                        let ord = f.docs.binary_search(doc).unwrap();
                        let FieldVectorData::Byte(v) = &f.values else {
                            unreachable!()
                        };
                        assert_eq!(
                            *score,
                            hamming_score(&query, &v[ord * dim..(ord + 1) * dim])
                        );
                    }
                }
            }
        }
    }
    assert!(
        recalled * 10 >= wanted * 8,
        "graph recall {recalled}/{wanted} under 80%"
    );
    let values = reader.byte_vector_values(1).unwrap();
    assert_eq!(values.size(), 2_000);
}

#[test]
fn scorers() {
    let mut seed = 99;
    let f = field(0, 8, 10, 2, &mut seed);
    let files = HnswBitVectorsFormat::default()
        .write(std::slice::from_ref(&f), 20, &ID, SUFFIX)
        .unwrap();
    let reader = BitVectorsReader::open(&files, &ID, SUFFIX).unwrap();
    let values = reader.byte_vector_values(0).unwrap();
    let mut scorer = FlatBitVectorsScorer.ord_scorer(&values);
    assert_eq!(scorer.max_ord(), 10);
    scorer.set_scoring_ordinal(3).unwrap();
    // A vector against itself, and the complement of a query against it.
    assert_eq!(scorer.score(3).unwrap(), 1.0);
    let v3 = values.vector(3).unwrap().to_vec();
    let inverted: Vec<u8> = v3.iter().map(|b| !b).collect();
    let mut opposite = FlatBitVectorsScorer.scorer(&values, &inverted).unwrap();
    assert_eq!(opposite.score(3).unwrap(), 0.0);
    assert_eq!(opposite.ord_to_doc(3).unwrap(), 6);
    assert!(opposite.ord_to_doc(10).is_err());
    assert!(scorer.score(10).is_err());
    assert!(scorer.set_scoring_ordinal(10).is_err());

    let err = FlatBitVectorsScorer.scorer(&values, &[0; 3]).unwrap_err();
    assert_eq!(err.to_string(), "vector dimensions differ: 3!=8");
    assert!(reader
        .search(0, &[0; 9], 1, 10, SearchOptions::default())
        .is_err());
    let err = FlatBitVectorsScorer.float_scorer(&[1.0; 8]).unwrap_err();
    assert_eq!(
        err.to_string(),
        "bit vectors do not support float[] targets"
    );
}

#[test]
fn reader_errors() {
    let mut seed = 7;
    let f = field(0, 8, 200, 1, &mut seed);
    let files = HnswBitVectorsFormat::default()
        .write(std::slice::from_ref(&f), 200, &ID, SUFFIX)
        .unwrap();
    let reader = BitVectorsReader::open(&files, &ID, SUFFIX).unwrap();
    assert!(reader.byte_vector_values(5).is_err());
    assert!(reader
        .search(5, &[0; 8], 1, 10, SearchOptions::default())
        .is_err());
    // Wrong suffix, and a damaged file of each kind.
    assert!(BitVectorsReader::open(&files, &ID, "other").is_err());
    for which in 0..4 {
        let mut bad = files.clone();
        let buf = match which {
            0 => &mut bad.vec,
            1 => &mut bad.vemf,
            2 => &mut bad.vex,
            _ => &mut bad.vem,
        };
        // A flipped footer byte (the checksum itself): `open` checks every
        // file's footer, and checksums `.vemf`/`.vem` whole.
        let at = buf.len() - 8;
        buf[at] ^= 0xff;
        assert!(
            BitVectorsReader::open(&bad, &ID, SUFFIX).is_err(),
            "file {which}"
        );
    }
}
