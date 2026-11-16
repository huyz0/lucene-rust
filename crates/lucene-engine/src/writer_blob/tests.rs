#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::test_support::Blob;
use super::*;

#[test]
fn every_value_kind_decodes() {
    let d = Blob::default()
        .i32(6)
        .i32(0)
        .u8(0)
        .bytes(b"s")
        .i32(0)
        .u8(1)
        .bytes(&[1, 2])
        .i32(0)
        .u8(2)
        .bytes(&7i32.to_le_bytes())
        .i32(0)
        .u8(3)
        .bytes(&7i64.to_le_bytes())
        .i32(0)
        .u8(4)
        .bytes(&1.5f32.to_le_bytes())
        .i32(0)
        .u8(5)
        .bytes(&1.5f64.to_le_bytes())
        .i32(0)
        .i32(2)
        .i32(1)
        .u8(0)
        .i64(7)
        .i32(1)
        .u8(1)
        .bytes(&[9])
        .i32(1)
        .i32(2)
        .bytes(&[0, 1, 2, 3]);
    let op = decode_op(
        &Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(d.0)
            .collect::<Vec<_>>(),
    );
    let Ok(Op::Add { docs, .. }) = op else {
        panic!("decoded");
    };
    let doc = &docs[0];
    assert_eq!(doc.stored.len(), 6);
    assert!(matches!(doc.stored[5].value, FieldValue::Double(v) if v == 1.5));
    assert!(matches!(doc.fields.doc_values[1].value, FieldValue::Binary(ref b) if b == &[9]));
    assert_eq!(doc.fields.points.len(), 1);
    for bad in [
        Blob::default()
            .i32(1)
            .i32(0)
            .u8(9)
            .bytes(b"x")
            .i32(0)
            .i32(0)
            .i32(0),
        Blob::default()
            .i32(1)
            .i32(0)
            .u8(2)
            .bytes(b"x")
            .i32(0)
            .i32(0)
            .i32(0),
        Blob::default()
            .i32(1)
            .i32(0)
            .u8(0)
            .bytes(&[0xff])
            .i32(0)
            .i32(0)
            .i32(0),
        Blob::default()
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(3)
            .i32(0)
            .i32(0)
            .i32(0),
        Blob::default().i32(0).i32(0).i32(1).i32(0).u8(9).i32(0),
        Blob::default().i32(-1),
    ] {
        let blob: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(bad.0)
            .collect();
        assert!(matches!(decode_op(&blob), Err(Status::Decode)));
    }
    // A freq whose positions or offsets cannot fit.
    let short = Blob::default()
        .i32(0)
        .i32(1)
        .i32(0)
        .u8(0)
        .i32(1)
        .bytes(b"t")
        .i32(1000)
        .u8(1);
    let blob: Vec<u8> = Blob::default()
        .u8(OP_ADD)
        .i32(1)
        .0
        .into_iter()
        .chain(short.0)
        .collect();
    assert!(matches!(decode_op(&blob), Err(Status::Decode)));
    let short = Blob::default()
        .i32(0)
        .i32(1)
        .i32(0)
        .u8(0)
        .i32(1)
        .bytes(b"t")
        .i32(1000)
        .u8(2);
    let blob: Vec<u8> = Blob::default()
        .u8(OP_ADD)
        .i32(1)
        .0
        .into_iter()
        .chain(short.0)
        .collect();
    assert!(matches!(decode_op(&blob), Err(Status::Decode)));
    let vector = Blob::default()
        .bytes(b"v")
        .u8(0)
        .u8(0)
        .u8(0)
        .u8(0)
        .u8(0)
        .u8(0)
        .i32(0)
        .i32(0)
        .i32(0)
        .i32(4)
        // float32, cosine, Lucene104HnswScalarQuantizedVectorsFormat(7 bits, 16, 100)
        .u8(0)
        .u8(2)
        .u8(2)
        .i32(16)
        .i32(100)
        .u8(2)
        .u8(0);
    let attributed = vector.clone().i32(1).bytes(b"k").bytes(b"v");
    assert_eq!(
        decode_field(&attributed.0).unwrap().0.attributes,
        vec![("k".to_string(), "v".to_string())]
    );
    assert!(decode_field(&vector.clone().i32(1).bytes(b"k").0).is_err());
    let vector = vector.i32(0);
    let (info, format) = decode_field(&vector.0).unwrap();
    assert_eq!(info.vector_dimension, 4);
    assert_eq!(info.vector_encoding, VectorEncoding::Float32);
    assert_eq!(
        info.vector_similarity_function,
        VectorSimilarityFunction::Cosine
    );
    assert_eq!(
        format,
        Some(KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SevenBit, 16, 100).unwrap())
    );
    // A byte field is never quantized; a graph's bounds are Lucene's.
    let schema = |encoding: u8, kind: u8, m: i32| {
        let b = Blob::default()
            .bytes(b"v")
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(4)
            .u8(encoding)
            .u8(0)
            .u8(kind);
        let b = match kind {
            1 | 4 | 5 => b.i32(m).i32(100),
            2 => b.i32(m).i32(100).u8(2),
            3 => b.u8(2),
            // `m` is the cluster size, the rerank encoding f16, the
            // quantizer bbq.
            6 => b.i32(m).u8(1).u8(0),
            // The quantizer erq2.
            7 | 8 => b.i32(m).i32(100).u8(4),
            _ => b,
        };
        b.u8(0).i32(0).0
    };
    // IVF: any field (a byte one keeps its raw vectors alone), a checked
    // cluster size.
    for encoding in [0, 1] {
        assert_eq!(
            decode_field(&schema(encoding, 6, 384)).unwrap().1,
            Some(KnnVectorsFormat::ivf(384, 1).unwrap())
        );
    }
    assert!(decode_field(&schema(0, 6, 8)).is_err());
    // Quantized HNSW: any field (a byte one gets the graph alone).
    for encoding in [0, 1] {
        assert_eq!(
            decode_field(&schema(encoding, 7, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_quantized(4, 16, 100).unwrap())
        );
    }
    assert!(decode_field(&schema(0, 7, 0)).is_err());
    // Disk HNSW likewise.
    for encoding in [0, 1] {
        assert_eq!(
            decode_field(&schema(encoding, 8, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_disk(4, 16, 100).unwrap())
        );
    }
    assert!(decode_field(&schema(0, 8, 0)).is_err());
    assert!(decode_field(&schema(0, 6, -1)).is_err());
    assert!(decode_field(&schema(1, 2, 16)).is_err());
    assert!(decode_field(&schema(1, 3, 16)).is_err());
    assert!(decode_field(&schema(1, 1, 16)).is_ok());
    // Hamming scoring is for byte (packed bit) fields only.
    assert_eq!(
        decode_field(&schema(1, 4, 16)).unwrap().1,
        Some(KnnVectorsFormat::hnsw_hamming(16, 100).unwrap())
    );
    assert!(decode_field(&schema(0, 4, 16)).is_err());
    // Half floats are for float fields only.
    assert_eq!(
        decode_field(&schema(0, 5, 16)).unwrap().1,
        Some(KnnVectorsFormat::hnsw_half_float(16, 100).unwrap())
    );
    assert!(decode_field(&schema(1, 5, 16)).is_err());
    assert!(decode_field(&schema(0, 1, 0)).is_err());
    assert!(decode_field(&schema(0, 1, 513)).is_err());
    assert_eq!(decode_field(&schema(0, 0, 0)).unwrap().1, None);
    assert!(decode_field(&schema(0, 9, 0)).is_err());
    assert_eq!(
        parse_vectors_format("hnsw:16:100"),
        Some(KnnVectorsFormat::hnsw(16, 100).unwrap())
    );
    assert_eq!(
        parse_vectors_format("hnsw_sq:3:32:200"),
        Some(
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SingleBitQueryNibble, 32, 200)
                .unwrap()
        )
    );
    assert_eq!(
        parse_vectors_format("hnsw_hamming:8:64"),
        Some(KnnVectorsFormat::hnsw_hamming(8, 64).unwrap())
    );
    assert_eq!(
        parse_vectors_format("hnsw_half:8:64"),
        Some(KnnVectorsFormat::hnsw_half_float(8, 64).unwrap())
    );
    assert_eq!(
        parse_vectors_format("ivf:384:1"),
        Some(KnnVectorsFormat::ivf(384, 1).unwrap())
    );
    assert_eq!(
        parse_vectors_format("ivf:384:1:6"),
        Some(KnnVectorsFormat::ivf_quantized(384, 1, 6).unwrap())
    );
    assert_eq!(
        parse_vectors_format("hnsw_rq:11:16:100"),
        Some(KnnVectorsFormat::hnsw_quantized(11, 16, 100).unwrap())
    );
    assert_eq!(
        parse_vectors_format("hnsw_disk:0:16:100"),
        Some(KnnVectorsFormat::hnsw_disk(0, 16, 100).unwrap())
    );
    assert_eq!(
        parse_vectors_format("sq:1"),
        Some(KnnVectorsFormat::scalar_quantized(
            ScalarEncoding::PackedNibble
        ))
    );
    for bad in [
        "",
        "hnsw",
        "hnsw:0:100",
        "hnsw_sq:9:16:100",
        "flat:1",
        "hnsw:a:b",
        "ivf:8:0",
        "ivf:384:4",
        "ivf:384",
        "ivf:384:1:13",
        "ivf:384:1:x",
        "hnsw_rq:12:16:100",
        "hnsw_rq:4:0:100",
        "hnsw_rq:4:16",
        "hnsw_disk:12:16:100",
        "hnsw_disk:0:0:100",
        "hnsw_disk:0:16",
    ] {
        assert_eq!(parse_vectors_format(bad), None, "{bad}");
    }
    assert!(decode_user_data(&Blob::default().i32(0).u8(0).0).is_err());
}
