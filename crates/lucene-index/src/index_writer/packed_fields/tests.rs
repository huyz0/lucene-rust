#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i - 3` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_codecs::term_vectors::TermVectorField;

fn sample(i: i64) -> ExplicitFields {
    ExplicitFields {
        inverted: vec![
            InvertedField {
                field_number: 1,
                terms: InvertedTerms::default(),
                norm: Some(i - 3),
            },
            InvertedField {
                field_number: 7,
                terms: InvertedTerms::default(),
                norm: None,
            },
        ],
        doc_values: vec![
            StoredField {
                field_number: 2,
                value: FieldValue::Long(i64::MIN + i),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::Binary(vec![i as u8; (i % 5) as usize]),
            },
            StoredField {
                field_number: 4,
                value: FieldValue::Double(i as f64 * -1.5),
            },
            StoredField {
                field_number: 5,
                value: FieldValue::Float(f32::NAN),
            },
            StoredField {
                field_number: 6,
                value: FieldValue::Int(i32::MAX - i as i32),
            },
            StoredField {
                field_number: 8,
                value: FieldValue::String(format!("s{i}é")),
            },
        ],
        points: vec![StoredField {
            field_number: 2,
            value: FieldValue::Binary((i as u64).to_be_bytes().to_vec()),
        }],
        registered: vec![9, i32::MAX],
        term_vectors: Vec::new(),
    }
}

fn same(a: &ExplicitFields, b: &ExplicitFields) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

/// Every value kind comes back as it went in, in order, across blocks,
/// with documents kept whole between packed ones.
#[test]
fn fields_unpack_as_they_were_packed() {
    let mut packed = PackedFields::default();
    let mut expected = Vec::new();
    let mut allocated = 0;
    for i in 0..5000 {
        let mut f = sample(i);
        if i % 997 == 0 {
            f.term_vectors.push(TermVectorField {
                field_number: 1,
                has_positions: false,
                has_offsets: false,
                has_payloads: false,
                terms: Vec::new(),
            });
        }
        expected.push(f.clone());
        allocated += packed.push(f);
    }
    assert_eq!(packed.len(), 5000);
    assert!(packed.blocks.len() > 1, "the records span several blocks");
    assert!(allocated >= packed.blocks.len() * BLOCK);
    // one document at a time, in order, each over the last
    let mut seen = 0;
    packed
        .for_each(|i, f| {
            same(f, &expected[i]);
            assert_eq!(i, seen);
            seen += 1;
            f.doc_values.clear();
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, expected.len());
    // a range, from the block holding its first document
    for (start, end) in [(0, 1), (996, 998), (1500, 4100), (4999, 5000), (2000, 2000)] {
        let mut seen = Vec::new();
        packed
            .for_each_range(start, end, |i, f| {
                same(f, &expected[start + i]);
                seen.push(i);
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, (0..end - start).collect::<Vec<_>>());
    }
    let unpacked = packed.unpack().unwrap();
    assert_eq!(unpacked.len(), expected.len());
    for (a, b) in unpacked.iter().zip(&expected) {
        same(a, b);
    }
}

/// A kept document that opens a block is followed by a packed one with
/// its shape, which the block's reader has not seen yet.
#[test]
fn a_block_opened_by_a_kept_document_restates_the_shape() {
    let mut packed = PackedFields::default();
    let mut expected = Vec::new();
    let mut kept = sample(0);
    kept.term_vectors.push(TermVectorField {
        field_number: 1,
        has_positions: false,
        has_offsets: false,
        has_payloads: false,
        terms: Vec::new(),
    });
    let mut opened = 0;
    let mut i = 0;
    while opened < 3 {
        let room = packed.blocks.last().map_or(0, |b| b.capacity() - b.len());
        // Fill the block to its last byte, then a kept document opens
        // the next one.
        let mut f = sample(i);
        let mut probe = Vec::new();
        f.doc_values[1].value = FieldValue::Binary(Vec::new());
        encode_values(&f, &mut probe);
        let fill = (1 + probe.len()..1 + probe.len() + 100).contains(&room);
        if fill {
            f.doc_values[1].value = FieldValue::Binary(vec![9; room - 1 - probe.len()]);
        }
        packed.push(f.clone());
        expected.push(f);
        if fill
            && packed
                .blocks
                .last()
                .is_some_and(|b| b.len() == b.capacity())
        {
            let blocks = packed.blocks.len();
            packed.push(kept.clone());
            expected.push(kept.clone());
            assert_eq!(packed.blocks.len(), blocks + 1);
            opened += 1;
        }
        i += 1;
    }
    let mut n = 0;
    packed
        .for_each(|i, f| {
            same(f, &expected[i]);
            n += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(n, expected.len());
    for (a, b) in packed.unpack().unwrap().iter().zip(&expected) {
        same(a, b);
    }
}

/// A record larger than a block gets a block of its own.
#[test]
fn a_large_record_fits() {
    let mut packed = PackedFields::default();
    let mut f = sample(1);
    f.doc_values[1].value = FieldValue::Binary(vec![7; 3 * BLOCK]);
    packed.push(sample(0));
    packed.push(f.clone());
    packed.push(sample(2));
    let unpacked = packed.unpack().unwrap();
    same(&unpacked[1], &f);
    same(&unpacked[2], &sample(2));
}

/// A packed document is a few dozen bytes, where the struct costs
/// hundreds.
#[test]
fn a_packed_document_is_small() {
    let mut f = sample(1000);
    f.doc_values.truncate(1);
    let mut out = Vec::new();
    encode_shape(&f, &mut out);
    encode_values(&f, &mut out);
    assert!(out.len() < 40, "{} bytes", out.len());
    assert!(f.ram_bytes() > 4 * out.len());
}
