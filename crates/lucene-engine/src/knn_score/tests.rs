#![cfg(test)]
use super::*;

#[test]
fn hamming_bit_aligns_at_the_low_end() {
    // BigInteger(1, {1, 0xff}) xor BigInteger(1, {0xff}): 0x01 and 0x00 high byte, 0xff ^ 0xff.
    assert_eq!(hamming_bit_bytes(&[1, 0xff], &[0xff]), 1);
    assert_eq!(hamming_bit_bytes(&[0xff], &[1, 0xff]), 1);
    assert_eq!(hamming_bit_bytes(&[0xf0, 0x0f], &[0x0f, 0xf0]), 16);
    assert_eq!(hamming_bit_bytes(&[], &[0xff, 0x01]), 9);
}

#[test]
fn first_binary_value_reads_the_custom_encoding() {
    // count 2, then (len 2, "ab"), (len 1, "z").
    let raw = [2u8, 2, b'a', b'b', 1, b'z'];
    assert_eq!(first_binary_value(&raw), Some(&b"ab"[..]));
    assert_eq!(first_binary_value(&[0]), None);
    assert_eq!(first_binary_value(&[1, 5, 1]), None);
    assert_eq!(first_binary_value(&[]), None);
}
use crate::query_blob::test_support::Blob;

fn node(space: u8, data_type: u8, target: Blob) -> Blob {
    Blob::default()
        .u8(NODE_KNN_SCORE)
        .term("f", "x")
        .str("v")
        .u8(space)
        .u8(data_type)
        .u8(0)
        .raw(&target)
        .u8(0)
}

fn decoded(blob: &Blob) -> std::result::Result<Clause, Status> {
    let mut c = Cursor::new(&blob.0);
    let mut nodes = 0;
    crate::query_blob::decode_node(&mut c, 0, &mut nodes)
}

fn source(space: Space, half_cosine: bool, target: Target) -> KnnScoreSource {
    let norm = match &target {
        Target::Floats(q) => q.iter().fold(0.0f32, |n, x| n + x * x),
        Target::Bytes(_) => 0.0,
    };
    KnnScoreSource {
        field: "v".into(),
        space,
        data_type: DataType::Float,
        half_cosine,
        target,
        norm,
        painless: None,
    }
}

#[test]
fn decodes_and_refuses_bad_nodes() {
    let floats = Blob::default().u8(0).i32(2).f32(0.5).f32(1.0);
    let bytes = Blob::default().u8(1).bytes(&[1, 2]);
    assert!(decoded(&node(0, 0, floats.clone())).is_ok());
    assert!(decoded(&node(5, 1, bytes.clone())).is_ok());
    let long = Blob::default().u8(1).bytes(&[0, 0, 0, 0, 0, 0, 0, 5]);
    assert!(
        decoded(&node(6, 3, long.clone())).is_ok(),
        "hammingbit over a long"
    );
    assert!(
        decoded(&node(6, 1, bytes.clone())).is_ok(),
        "hammingbit over binary"
    );
    assert!(
        decoded(&node(6, 3, bytes.clone())).is_err(),
        "a long target is 8 bytes"
    );
    assert!(decoded(&node(6, 0, floats.clone())).is_err(), "space");
    assert!(decoded(&node(0, 3, floats.clone())).is_err(), "data type");
    assert!(
        decoded(&node(0, 1, floats.clone())).is_err(),
        "floats for a byte field"
    );
    assert!(
        decoded(&node(0, 0, bytes.clone())).is_err(),
        "bytes for a float field"
    );
    assert!(
        decoded(&node(5, 0, floats.clone())).is_err(),
        "hamming over floats"
    );
    assert!(
        decoded(&node(0, 1, Blob::default().u8(2))).is_err(),
        "target kind"
    );
    let with = |data_type: u8, target: &Blob, formula: Blob| {
        let mut b = node(0, data_type, target.clone());
        b.0.pop();
        decoded(&b.raw(&formula))
    };
    let program = |function: u8| Blob::default().u8(1).u8(function).f32(1.0).i32(1).u8(2);
    assert!(with(0, &floats, program(0)).is_ok(), "painless");
    assert!(with(1, &bytes, program(4)).is_ok(), "painless over bytes");
    assert!(
        with(1, &bytes, program(5)).is_err(),
        "a norm over a byte field"
    );
    assert!(with(0, &floats, Blob::default().u8(2)).is_err(), "formula");
}

#[test]
fn float_spaces_score_as_the_script() {
    let q = vec![1.0f32, -2.0, 0.5];
    let v = [0.5f32, 1.0, -1.5];
    let s =
        |space, half| source(space, half, Target::Floats(q.clone().into())).score_floats(&q, &v);
    // Squared distance 0.25 + 9 + 4 = 13.25; l1 0.5 + 3 + 2 = 5.5; linf 3.
    assert_eq!(s(Space::L2, false), 1.0 / 14.25);
    assert_eq!(s(Space::L1, false), 1.0 / 6.5);
    assert_eq!(s(Space::LInf, false), 1.0 / 4.0);
    // Inner product 0.5 - 2 - 0.75 = -2.25 scales to 1 / 3.25.
    assert_eq!(s(Space::InnerProduct, false), 1.0 / 3.25);
    let cos = (-2.25f64 / (5.25f64 * 3.5).sqrt()) as f32;
    assert_eq!(s(Space::Cosine, false), 1.0 + cos);
    assert_eq!(s(Space::Cosine, true), ((1.0 + cos) / 2.0).max(0.0));
    // A positive inner product scales to ip + 1.
    assert_eq!(
        source(Space::InnerProduct, false, Target::Floats(q.clone().into())).score_floats(&q, &q),
        5.25 + 1.0
    );
}

#[test]
fn byte_spaces_score_as_the_script() {
    let q = vec![3u8, 0xFE, 1];
    let v = [1u8, 2, 0xFF];
    let s = |space| source(space, false, Target::Bytes(q.clone().into())).score_bytes(&q, &v);
    // Signed: q = (3, -2, 1), v = (1, 2, -1).
    assert_eq!(s(Space::L2), 1.0 / (1.0 + 4.0 + 16.0 + 4.0));
    assert_eq!(s(Space::L1), 1.0 / (1.0 + 8.0));
    assert_eq!(s(Space::LInf), 1.0 / 5.0);
    assert_eq!(
        s(Space::InnerProduct),
        scale_max_inner_product_score(3.0 - 4.0 - 1.0)
    );
    // Hamming over the raw bits: 3^1 = 2 (1 bit), 0xFE^2 = 0xFC (6), 1^0xFF = 0xFE (7).
    assert_eq!(s(Space::Hamming), 1.0 / 15.0);
}

#[test]
fn equality_names_the_target_and_zero_byte_cosine_is_nan() {
    let a = source(Space::L2, false, Target::Floats(vec![1.0, 2.0].into()));
    let b = source(Space::L2, false, Target::Floats(vec![1.0, 3.0].into()));
    assert_ne!(a.equality_key(), b.equality_key());
    assert_eq!(a.equality_key(), a.clone().equality_key());
    let c = source(Space::Cosine, false, Target::Bytes(vec![1, 2].into()));
    assert!(c.score_bytes(&[1, 2], &[0, 0]).is_nan());
}
