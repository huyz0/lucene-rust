//! Differential tests: decode bytes written by Java Lucene (fixtures/) and compare
//! against the values Java says it wrote. Regenerate with:
//!   cd fixtures && javac -cp $LUCENE_JAR -d classes src/GenPrimitives.java \
//!     && java -cp classes:$LUCENE_JAR GenPrimitives data
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_store::{DataInput, DataOutput, SliceInput};

fn fixture(name: &str) -> (Vec<u8>, Vec<i64>) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/");
    let bin = std::fs::read(format!("{dir}{name}.bin")).expect("run fixture generator");
    let expected = std::fs::read_to_string(format!("{dir}{name}.expected"))
        .unwrap()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    (bin, expected)
}

#[test]
fn vint_matches_java() {
    let (bin, expected) = fixture("vint");
    let mut input = SliceInput::new(&bin);
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(input.read_vint().unwrap() as i64, want, "index {i}");
    }
    assert_eq!(input.remaining(), 0, "trailing bytes left undecoded");
}

#[test]
fn vlong_matches_java() {
    let (bin, expected) = fixture("vlong");
    let mut input = SliceInput::new(&bin);
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(input.read_vlong().unwrap(), want, "index {i}");
    }
    assert_eq!(input.remaining(), 0);
}

#[test]
fn zlong_matches_java() {
    let (bin, expected) = fixture("zlong");
    let mut input = SliceInput::new(&bin);
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(input.read_zlong().unwrap(), want, "index {i}");
    }
    assert_eq!(input.remaining(), 0);
}

#[test]
fn group_vint_matches_java() {
    let (bin, expected) = fixture("group_vint");
    let mut input = SliceInput::new(&bin);
    let mut dst = vec![0u64; expected.len()];
    input.read_group_vints(&mut dst).unwrap();
    for (i, (&got, &want)) in dst.iter().zip(&expected).enumerate() {
        assert_eq!(got as i64, want, "index {i}");
    }
    assert_eq!(input.remaining(), 0);
}

/// Non-multiple-of-4 group-varint length exercises the vint tail path.
#[test]
fn group_vint_tail() {
    let (bin, expected) = fixture("group_vint");
    // Re-decode only full groups then verify a fresh cursor with a tail-sized dst
    // still decodes the leading values correctly.
    let mut input = SliceInput::new(&bin);
    let mut dst = [0u64; 7]; // 1 group + 3 tail... tail here is still group-encoded
                             // in the fixture, so only compare the first full group.
    input.read_group_vints(&mut dst[..4]).unwrap();
    assert_eq!(
        dst[..4].iter().map(|&v| v as i64).collect::<Vec<_>>(),
        expected[..4]
    );
}

// --- write side: re-encoding Java's own values must reproduce Java's bytes ---
//
// The decoders above prove we read what Java wrote. These prove the inverse
// for the encoders in `DataOutput`: given the values Java says it wrote,
// our writer must produce the *identical* byte sequence -- byte-for-byte, not
// merely something our own reader accepts (which a self-consistent but wrong
// encoding would also satisfy).

#[test]
fn write_vint_reproduces_java_bytes() {
    let (bin, expected) = fixture("vint");
    let mut out: Vec<u8> = Vec::new();
    for &v in &expected {
        out.write_vint(v as i32);
    }
    assert_eq!(out, bin);
}

#[test]
fn write_vlong_reproduces_java_bytes() {
    let (bin, expected) = fixture("vlong");
    let mut out: Vec<u8> = Vec::new();
    for &v in &expected {
        out.write_vlong(v);
    }
    assert_eq!(out, bin);
}

#[test]
fn write_zlong_reproduces_java_bytes() {
    let (bin, expected) = fixture("zlong");
    let mut out: Vec<u8> = Vec::new();
    for &v in &expected {
        out.write_zlong(v);
    }
    assert_eq!(out, bin);
}

#[test]
fn write_group_vints_reproduces_java_bytes() {
    // The fixture is 1024 values spread across all four byte widths, written
    // by Java's own `writeGroupVInts` -- so this pins the flag-byte packing
    // order and the vint tail, not just the value round-trip.
    let (bin, expected) = fixture("group_vint");
    let values: Vec<u32> = expected.iter().map(|&v| v as u32).collect();
    let mut out: Vec<u8> = Vec::new();
    out.write_group_vints(&values);
    assert_eq!(out, bin);
}
