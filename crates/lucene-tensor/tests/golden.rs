//! Fixed values encoded byte for byte as the plugin's Java `MultiVectorEncoding`
//! encodes them, and their pooled mean as its `pooledMean` (`NativeSelfTestTensor`
//! holds the same hex).
#![allow(clippy::arithmetic_side_effects)]

use lucene_tensor::{encode, TokenEncoding};

/// 17 tokens of 3: a second block, an int8 tie at `k + 0.5`, signed zeros, a
/// subnormal-max token, float16's largest magnitudes.
fn tokens() -> Vec<f32> {
    let mut t = vec![
        127.0,
        2.5,
        -2.5,
        -0.0,
        0.0,
        0.0,
        f32::from_bits(5),
        -f32::from_bits(3),
        0.0,
        65504.0,
        -65504.0,
        1.0,
    ];
    for i in 4..17 {
        t.push(i as f32 * 0.25 - 2.0);
        t.push((i % 5) as f32 * 0.125);
        t.push(-(((i * 3) % 7) as f32) / 8.0);
    }
    t
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

const FLOAT: &str = "030000000000fe4200002040000020c000000080000000000000000005000000030000800000000000e07f4700e07fc70000803f000080bf0000003f000020bf000040bf00000000000000be000000bf0000003e000000bf000080be0000803e00000080000000000000c03e0000c0be0000803e0000003f000040bf0000003f00000000000080be0000403f0000003e000020bf0000803f0000803e000000be0000a03f0000c03e000000bf0000c03f0000003f000000800000e03f000000000000c0be000000400000003e000040bf";
const FLOAT16: &str = "ffffffff0300000011000000f05700800000ff7b00bc00ba00b800b4000000340038003a003c003d003e003f004100000080fffb00380000003000340036003800000030003400360038000000c100000000003c00b900b000b8008000b600ba00b400b900b000b8008000b60040000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000000000000000000000000000000000000ba000000000000000000000000000000000000000000000000000000000000";
const INT8: &str = "feffffff03000000110000000000803f0000803f01000000e4f100440402013c0683c13b0402813b0402013b0683413b0683c13b0402813b0683c13b0402013c8542213c0683413c87c3613c0402813c7f00057f81818181002a7f7f7f7f7f7f0300fd814000207f7f55001520262a00fd000000b1eb81008181c096f0cd00e57f00000000000000000000000000000008000000000000000000000000000000d0000000000000000000000000000000";

#[test]
fn every_encoding_writes_the_plugins_bytes() {
    for (e, want) in [
        (TokenEncoding::Float32, FLOAT),
        (TokenEncoding::Float16, FLOAT16),
        (TokenEncoding::Int8, INT8),
    ] {
        assert_eq!(hex(&encode(&tokens(), 3, e).unwrap()), want, "{}", e.name());
    }
}

/// `pooled_mean` of the tokens, little-endian (`MultiVectorEncoding.pooledMean`).
const POOLED: &str = "1736353fc2d334bfa90193b8";

#[test]
fn the_pooled_mean_is_the_plugins() {
    let m = lucene_tensor::pooled_mean(&tokens(), 3);
    let bytes: Vec<u8> = m.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(hex(&bytes), POOLED);
}
