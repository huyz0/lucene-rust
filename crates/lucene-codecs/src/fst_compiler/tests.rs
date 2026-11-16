#![cfg(test)]
use super::*;

fn build<O: FstOutputs>(entries: &[(&[u8], O::Value)]) -> CompiledFst<O::Value> {
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<O>();
    for (k, v) in entries {
        c.add_bytes(k, v.clone()).unwrap();
    }
    c.compile().unwrap()
}

#[test]
fn outputs_algebra() {
    assert_eq!(PositiveIntOutputs::common(&5, &3), 3);
    assert_eq!(PositiveIntOutputs::common(&0, &3), 0);
    assert_eq!(PositiveIntOutputs::subtract(&5, &3), 2);
    assert_eq!(PositiveIntOutputs::add(&5, &3), 8);
    assert_eq!(
        ByteSequenceOutputs::common(&b"abc".to_vec(), &b"abd".to_vec()),
        b"ab"
    );
    assert_eq!(
        ByteSequenceOutputs::subtract(&b"abc".to_vec(), &b"ab".to_vec()),
        b"c"
    );
    assert_eq!(IntSequenceOutputs::add(&vec![1], &vec![2]), vec![1, 2]);
    assert_eq!(
        IntSequenceOutputs::common(&vec![1, 2], &vec![1, 3]),
        vec![1]
    );
    assert_eq!(IntSequenceOutputs::subtract(&vec![1, 2], &vec![1]), vec![2]);
    assert_eq!(CharSequenceOutputs::common(&vec![7, 8], &vec![7]), vec![7]);
    assert_eq!(
        CharSequenceOutputs::subtract(&vec![7, 8], &vec![7]),
        vec![8]
    );
    assert_eq!(CharSequenceOutputs::add(&vec![7], &vec![8]), vec![7, 8]);
    type P = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
    let a = Pair {
        first: 4i64,
        second: b"xy".to_vec(),
    };
    let b = Pair {
        first: 6i64,
        second: b"xz".to_vec(),
    };
    let c = P::common(&a, &b);
    assert_eq!(
        c,
        Pair {
            first: 4,
            second: b"x".to_vec()
        }
    );
    assert_eq!(P::add(&c, &P::subtract(&b, &c)), b);
    NoOutputs::common(&(), &());
    NoOutputs::subtract(&(), &());
    NoOutputs::add(&(), &());
    assert_eq!(PositiveIntOutputs::merge(&1, &2), None);

    let mut buf = Vec::new();
    P::write(&b, &mut buf);
    IntSequenceOutputs::write(&vec![300, 1], &mut buf);
    CharSequenceOutputs::write(&vec![65], &mut buf);
    NoOutputs::write(&(), &mut buf);
    buf.reverse();
    let mut r = ReverseReader::new(&buf);
    r.set_position(buf.len() as i64 - 1);
    assert_eq!(P::read(&mut r).unwrap(), b);
    assert_eq!(IntSequenceOutputs::read(&mut r).unwrap(), vec![300, 1]);
    assert_eq!(
        CharSequenceOutputs::read_final_output(&mut r).unwrap(),
        vec![65]
    );
    NoOutputs::read(&mut r).unwrap();
    assert_eq!(r.position(), -1);
    assert!(r.read_byte().is_err());
}

#[test]
fn reader_errors() {
    let bad = [0x80u8; 12];
    let mut r = ReverseReader::new(&bad);
    r.set_position(11);
    assert!(r.read_vlong().is_err());
    let neg = [0x0f, 0xff, 0xff, 0xff, 0xff];
    let mut r = ReverseReader::new(&neg);
    r.set_position(4);
    assert!(read_len(&mut r).is_err());
    let s = [0x01, 0x02];
    let mut r = ReverseReader::new(&s);
    r.set_position(1);
    assert_eq!(r.read_short().unwrap(), i16::from_le_bytes([2, 1]));
    r.set_position(1);
    r.skip_bytes(1);
    assert_eq!(r.read_byte().unwrap(), 1);
}

#[test]
fn builder_and_input_checks() {
    assert!(FstCompilerBuilder::new(InputType::Byte1)
        .suffix_ram_limit_mb(-1.0)
        .is_err());
    assert!(FstCompilerBuilder::new(InputType::Byte1)
        .version(7)
        .is_err());
    assert!(FstCompilerBuilder::new(InputType::Byte1).version(8).is_ok());
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
    assert!(matches!(
        c.add(&[256], 1),
        Err(CompileError::LabelOutOfRange { .. })
    ));
    c.add_bytes(b"b", 1).unwrap();
    assert_eq!(c.add_bytes(b"a", 1), Err(CompileError::OutOfOrder));
    assert_eq!(c.add_bytes(b"b", 2), Err(CompileError::MergeUnsupported));
    let mut e = FstCompilerBuilder::new(InputType::Byte2).build::<PositiveIntOutputs>();
    e.add(&[], 3).unwrap();
    assert_eq!(e.add(&[], 4), Err(CompileError::MergeUnsupported));
    assert!(e.add(&[70000], 1).is_err());
    e.add(&[65535], 1).unwrap();
    assert!(FstCompilerBuilder::new(InputType::Byte4)
        .build::<NoOutputs>()
        .add(&[-1], ())
        .is_err());
    let mut none = FstCompilerBuilder::new(InputType::Byte1).build::<NoOutputs>();
    assert!(none.compile().is_none());
}

#[test]
fn shared_suffixes_and_pushed_outputs() {
    let f = build::<PositiveIntOutputs>(&[(b"cat", 5), (b"cats", 7), (b"dog", 5), (b"dogs", 7)]);
    // "at"/"og" tails share one "s"-final node; outputs pushed to the root.
    assert!(f.start_node > 0);
    let saved = f.save::<PositiveIntOutputs>();
    assert!(saved.len() > f.bytes.len());
    let empty = {
        let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<ByteSequenceOutputs>();
        c.add_bytes(b"", b"e".to_vec()).unwrap();
        c.compile().unwrap()
    };
    assert_eq!(empty.start_node, 0);
    assert_eq!(empty.bytes, vec![0]);
    // ... accepts-empty, len 2, reversed [len 1, 'e'], BYTE1, start 0, 1 byte, body.
    let saved = empty.save::<ByteSequenceOutputs>();
    assert!(saved.ends_with(&[1, 2, b'e', 1, 0, 0, 1, 0]), "{saved:?}");
}

#[test]
fn an_empty_input_with_an_empty_output_reopens() {
    // The empty output's reversed bytes are just its length (0): reading
    // it leaves the reader before the first byte, where a zero-length
    // read must not underflow.
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<ByteSequenceOutputs>();
    c.add_bytes(b"", Vec::new()).unwrap();
    c.add_bytes(b"a", b"x".to_vec()).unwrap();
    let saved = c.compile().unwrap().save::<ByteSequenceOutputs>();
    let fst = crate::fst_util::TypedFst::<ByteSequenceOutputs>::read(&saved).unwrap();
    assert_eq!(
        crate::fst_util::get_bytes(&fst, b"").unwrap(),
        Some(Vec::new())
    );
    assert_eq!(
        crate::fst_util::get_bytes(&fst, b"a").unwrap(),
        Some(b"x".to_vec())
    );
    let mut r = ReverseReader::new(&[]);
    r.set_position(-1);
    assert_eq!(r.read_bytes_reversed(0).unwrap(), Vec::<u8>::new());
}

#[test]
fn no_outputs_merge_duplicates_and_fixed_nodes() {
    let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<NoOutputs>();
    c.add_bytes(b"", ()).unwrap();
    c.add_bytes(b"", ()).unwrap();
    for k in [
        b"a".as_slice(),
        b"a",
        b"b",
        b"c",
        b"d",
        b"e",
        b"f",
        b"h",
        b"z",
    ] {
        c.add_bytes(k, ()).unwrap();
    }
    let f = c.compile().unwrap();
    assert!(c.compile().is_none(), "already finished");
    let s = c.stats();
    assert_eq!(s.arc_count, 8);
    assert_eq!(s.node_count, 2);
    assert_eq!(
        s.binary_search_node_count + s.direct_addressing_node_count,
        1
    );
    assert_eq!(c.fst_size_in_bytes(), f.bytes.len() as i64);
    assert_eq!(oversize_bytes(0), 0);
    assert_eq!(oversize_bytes(20), 24);
    assert_eq!(bits_required(0), 1);
    assert_eq!(bits_required(255), 8);
}
