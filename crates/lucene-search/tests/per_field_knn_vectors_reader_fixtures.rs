//! Reading segments whose vector fields are split across several
//! `PerFieldKnnVectorsFormat` instances (`GenPerFieldKnnVectors`: two
//! `Lucene99HnswVectorsFormat`s, a `Lucene104HnswScalarQuantizedVectorsFormat`
//! and a `Lucene104ScalarQuantizedVectorsFormat`): every field's vectors come
//! from the instance its attributes name, through the reader layer's
//! `getFloatVectorValues`/`getByteVectorValues`, and equal what the generator
//! indexed.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::LeafReader;
use lucene_store::FsDirectory;

const PER_SEGMENT: usize = 300;
const DIM: usize = 16;

fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn floats(i: usize, salt: i64) -> Vec<f32> {
    (0..DIM as i64)
        .map(|j| (value(i, salt * 100 + j) % 2001 - 1000) as f32 / 100.0)
        .collect()
}

fn bytes(i: usize) -> Vec<u8> {
    (0..DIM as i64)
        .map(|j| (value(i, 500 + j) % 255 - 127) as i8 as u8)
        .collect()
}

/// `GenPerFieldKnnVectors.doc`'s vector for `field`, if the document has one.
fn expected(field: &str, i: usize) -> Option<Vec<f32>> {
    match field {
        "v_sq" if !i.is_multiple_of(7) => Some(floats(i, 1)),
        "v_default" if i != 0 => Some(floats(i, 2)),
        "v_small" if i != PER_SEGMENT => Some(floats(i, 3)),
        "v_flat" => Some(floats(i, 4)),
        _ => None,
    }
}

#[test]
fn every_field_reads_from_its_own_format_instance() {
    let dir = FsDirectory::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/per_field_knn_vectors/flushed"
    ));
    let reader = DirectoryReader::open(&dir).unwrap();
    let mut checked = 0;
    for (s, seg) in reader.segment_readers().iter().enumerate() {
        for field in ["v_sq", "v_default", "v_small", "v_flat"] {
            let values = seg.float_vector_values(field).unwrap().expect(field);
            let mut seen = 0;
            for ord in 0..values.size() {
                let doc = values.ord_to_doc(ord).unwrap() as usize;
                let want = expected(field, s * PER_SEGMENT + doc).expect("a vector it indexed");
                assert_eq!(values.vector_value(ord).unwrap(), want, "{field} doc {doc}");
                seen += 1;
            }
            let indexed = (0..PER_SEGMENT)
                .filter(|d| expected(field, s * PER_SEGMENT + d).is_some())
                .count();
            assert_eq!(seen, indexed, "{field}");
            // `getVectorCount` (Lucene 10.6, `branch_10_6`, #16647): the
            // metadata's count equals the opened values' size.
            assert_eq!(seg.vector_count(field).unwrap(), values.size(), "{field}");
            checked += seen;
        }
        let values = seg.byte_vector_values("v_bytes").unwrap().unwrap();
        assert_eq!(values.size() as usize, PER_SEGMENT);
        for ord in 0..values.size() {
            let doc = values.ord_to_doc(ord).unwrap() as usize;
            assert_eq!(
                values.vector_value(ord).unwrap(),
                bytes(s * PER_SEGMENT + doc)
            );
        }
        assert_eq!(seg.vector_count("v_bytes").unwrap(), values.size());
        // `testGetVectorCountInvalidField`: a field without vectors throws
        // `IllegalArgumentException`.
        let plain = seg
            .field_infos()
            .fields
            .iter()
            .filter(|f| f.vector_dimension == 0)
            .map(|f| f.name.clone())
            .chain(["no_such_field".to_string()]);
        for name in plain {
            match seg.vector_count(&name) {
                Err(lucene_search::Error::IllegalArgument(m)) => {
                    assert_eq!(m, format!("field=\"{name}\" not found"));
                }
                other => panic!("{name}: {other:?}"),
            }
        }
        assert!(seg.byte_vector_values("v_small").unwrap().is_none());
        assert!(seg.float_vector_values("v_bytes").unwrap().is_none());
    }
    // v_sq: all but the 86 multiples of 7; v_default, v_small: all but one.
    assert_eq!(checked, 514 + 599 + 599 + 600);
}
