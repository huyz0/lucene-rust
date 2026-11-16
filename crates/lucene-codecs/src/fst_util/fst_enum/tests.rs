#![cfg(test)]
use super::*;
use crate::fst::{InputType, PositiveIntOutputs};
use crate::fst_compiler::FstCompilerBuilder;

/// Keys with dense, sparse and contiguous label runs, so list,
/// binary-search, direct-addressing and continuous nodes all appear.
fn keys(seed: u64, n: usize) -> Vec<Vec<u8>> {
    let mut x = seed;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut keys = std::collections::BTreeSet::new();
    for i in 0..n {
        let len = 1 + (next() % 5) as usize;
        let k: Vec<u8> = (0..len)
            .map(|j| match (i + j) % 3 {
                0 => b'a' + (next() % 26) as u8,
                1 => (next() % 256) as u8,
                _ => b'm' + (next() % 4) as u8,
            })
            .collect();
        keys.insert(k);
    }
    // A contiguous run of first labels.
    for b in b'0'..=b'9' {
        keys.insert(vec![b]);
        keys.insert(vec![b, b'x']);
    }
    keys.insert(Vec::new());
    keys.into_iter().collect()
}

fn build(keys: &[Vec<u8>], factor: f32) -> TypedFst<PositiveIntOutputs> {
    let mut c = FstCompilerBuilder::new(InputType::Byte1)
        .direct_addressing_max_oversizing_factor(factor)
        .build::<PositiveIntOutputs>();
    for (i, k) in keys.iter().enumerate() {
        c.add_bytes(k, i as i64 * 3).unwrap();
    }
    TypedFst::from_compiled(c.compile().unwrap())
}

#[test]
fn seek_floor_matches_a_sorted_scan() {
    for (seed, factor) in [(7u64, 0.0f32), (11, 1.0), (13, 64.0), (17, -1.0)] {
        for skip_empty in [false, true] {
            let mut keys = keys(seed, 600);
            if skip_empty {
                keys.remove(0);
            }
            let fst = build(&keys, factor);
            let mut e = BytesRefFstEnum::new(&fst);
            let mut probes: Vec<Vec<u8>> = keys.clone();
            for k in &keys {
                let mut longer = k.clone();
                longer.push(0);
                probes.push(longer);
                let mut bumped = k.clone();
                if let Some(last) = bumped.last_mut() {
                    if *last > 0 {
                        *last -= 1;
                        probes.push(bumped.clone());
                    }
                    if let Some(l) = bumped.last_mut() {
                        *l = l.saturating_add(2);
                    }
                    probes.push(bumped);
                }
            }
            probes.push(vec![0xff; 6]);
            probes.push(Vec::new());
            // In a scrambled order, so the enum rewinds as well as advances.
            let n = probes.len();
            for step in 0..n {
                let probe = &probes[(step * 7919) % n];
                let want = keys
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, k)| k.as_slice() <= probe.as_slice())
                    .map(|(i, k)| (k.clone(), i as i64 * 3));
                let got = e.seek_floor(probe).unwrap().map(|(k, v)| (k.to_vec(), *v));
                assert_eq!(got, want, "factor {factor} probe {probe:?}");
            }
        }
    }
}

#[test]
fn integer_labels_match_sorted_scan_for_all_seeks() {
    for (input_type, scale) in [
        (InputType::Byte1, 1),
        (InputType::Byte2, 257),
        (InputType::Byte4, 4096),
    ] {
        for factor in [0.0, 1.0, 64.0, -1.0] {
            let keys: Vec<Vec<i32>> = keys(19, 350)
                .iter()
                .map(|key| key.iter().map(|&b| i32::from(b) * scale).collect())
                .collect();
            let mut compiler = FstCompilerBuilder::new(input_type)
                .direct_addressing_max_oversizing_factor(factor)
                .build::<PositiveIntOutputs>();
            for (i, key) in keys.iter().enumerate() {
                compiler.add(key, i as i64 * 3).unwrap();
            }
            let fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiler.compile().unwrap());
            let mut cursor = IntsRefFstEnum::new(&fst);
            assert!(cursor.current().is_none());
            for (i, key) in keys.iter().enumerate() {
                assert_eq!(
                    cursor.next().unwrap(),
                    Some((key.as_slice(), &(i as i64 * 3)))
                );
                assert_eq!(cursor.current(), Some((key.as_slice(), &(i as i64 * 3))));
            }
            assert!(cursor.next().unwrap().is_none());
            let mut probes = keys.clone();
            for key in &keys {
                let mut longer = key.clone();
                longer.push(1);
                probes.push(longer);
                if !key.is_empty() {
                    let mut bumped = key.clone();
                    *bumped.last_mut().unwrap() += 1;
                    probes.push(bumped);
                }
            }
            probes.push(vec![0x110000]);
            // Reverse order forces rewind and floor backtracking.
            for probe in probes.iter().rev() {
                let position = keys.binary_search(probe);
                let expected =
                    |index: Option<usize>| index.map(|i| (keys[i].clone(), i as i64 * 3));
                let exact = position.ok();
                let ceil = match position {
                    Ok(i) => Some(i),
                    Err(i) => (i < keys.len()).then_some(i),
                };
                let floor = match position {
                    Ok(i) => Some(i),
                    Err(i) => i.checked_sub(1),
                };
                assert_eq!(
                    cursor
                        .seek_exact(probe)
                        .unwrap()
                        .map(|(k, v)| (k.to_vec(), *v)),
                    expected(exact)
                );
                assert_eq!(
                    cursor
                        .seek_ceil(probe)
                        .unwrap()
                        .map(|(k, v)| (k.to_vec(), *v)),
                    expected(ceil)
                );
                assert_eq!(
                    cursor
                        .seek_floor(probe)
                        .unwrap()
                        .map(|(k, v)| (k.to_vec(), *v)),
                    expected(floor)
                );
            }
        }
    }
}

#[test]
fn integer_singletons_empty_and_long_inputs() {
    for input in [vec![], vec![0], vec![0x10ffff], vec![0x10000; 32]] {
        let mut compiler = FstCompilerBuilder::new(InputType::Byte4).build::<PositiveIntOutputs>();
        compiler.add(&input, 7).unwrap();
        let fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiler.compile().unwrap());
        let mut cursor = IntsRefFstEnum::new(&fst);
        assert_eq!(cursor.next().unwrap(), Some((input.as_slice(), &7)));
        assert!(cursor.next().unwrap().is_none());
        assert!(cursor.current().is_none());
        assert_eq!(
            cursor.seek_exact(&input).unwrap(),
            Some((input.as_slice(), &7))
        );
        assert_eq!(
            cursor.seek_floor(&input).unwrap(),
            Some((input.as_slice(), &7))
        );
        assert_eq!(
            cursor.seek_ceil(&input).unwrap(),
            Some((input.as_slice(), &7))
        );
        assert!(cursor.seek_ceil(&[0x110000]).unwrap().is_none());
        if !input.is_empty() {
            assert!(cursor.seek_floor(&[]).unwrap().is_none());
        }
    }
}

#[test]
fn integer_enumeration_reports_truncated_arc_data() {
    let mut compiler = FstCompilerBuilder::new(InputType::Byte4).build::<PositiveIntOutputs>();
    compiler.add(&[0x10000, 0x10ffff], 7).unwrap();
    let mut fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiler.compile().unwrap());
    fst.bytes.clear();
    assert!(IntsRefFstEnum::new(&fst).next().is_err());
    assert!(IntsRefFstEnum::new(&fst).seek_exact(&[0x10000]).is_err());
    assert!(IntsRefFstEnum::new(&fst).seek_ceil(&[0x10000]).is_err());
    assert!(IntsRefFstEnum::new(&fst).seek_floor(&[0x10000]).is_err());
}

#[test]
fn public_current_is_valid_only_after_a_successful_operation() {
    macro_rules! check {
        ($enumerator:ident, $label:ty, $input_type:expr, $first:expr, $last:expr) => {{
            let first: [$label; 2] = [$first, 1];
            let last: [$label; 2] = [$last, 2];
            let mut compiler = FstCompilerBuilder::new($input_type).build::<PositiveIntOutputs>();
            compiler.add(&first.map(i32::from), 7).unwrap();
            compiler.add(&last.map(i32::from), 11).unwrap();
            let fst = TypedFst::<PositiveIntOutputs>::from_compiled(compiler.compile().unwrap());
            let mut cursor = $enumerator::new(&fst);
            assert!(cursor.current().is_none());
            assert_eq!(cursor.next().unwrap(), Some((first.as_slice(), &7)));
            assert!(cursor.seek_exact(&[first[0]]).unwrap().is_none());
            assert!(cursor.current().is_none());
            assert_eq!(
                cursor.seek_ceil(&first).unwrap(),
                Some((first.as_slice(), &7))
            );
            assert_eq!(cursor.current(), Some((first.as_slice(), &7)));
            assert!(cursor.seek_floor(&[]).unwrap().is_none());
            assert!(cursor.current().is_none());
            assert_eq!(cursor.next().unwrap(), Some((first.as_slice(), &7)));
            assert!(cursor.seek_ceil(&[last[0], 3]).unwrap().is_none());
            assert!(cursor.current().is_none());
            assert_eq!(cursor.next().unwrap(), Some((first.as_slice(), &7)));
            assert_eq!(
                cursor.seek_exact(&last).unwrap(),
                Some((last.as_slice(), &11))
            );
            assert!(cursor.next().unwrap().is_none());
            assert!(cursor.current().is_none());
            assert_eq!(
                cursor.seek_floor(&last).unwrap(),
                Some((last.as_slice(), &11))
            );
            assert_eq!(cursor.current(), Some((last.as_slice(), &11)));

            for operation in 0..4 {
                let mut cursor = $enumerator::new(&fst);
                assert_eq!(
                    cursor.seek_exact(&first).unwrap(),
                    Some((first.as_slice(), &7))
                );
                // A corrupt root address makes each traversal fail after
                // the public cursor was successfully positioned.
                let root_address = cursor.inner.arcs[0].target;
                cursor.inner.arcs[0].target = i64::MAX;
                cursor.inner.upto = 0;
                assert!(match operation {
                    0 => cursor.next(),
                    1 => cursor.seek_exact(&first),
                    2 => cursor.seek_ceil(&first),
                    _ => cursor.seek_floor(&first),
                }
                .is_err());
                assert!(cursor.current().is_none());
                cursor.inner.arcs[0].target = root_address;
                cursor.inner.upto = 0;
                assert_eq!(
                    cursor.seek_exact(&last).unwrap(),
                    Some((last.as_slice(), &11))
                );
                assert_eq!(cursor.current(), Some((last.as_slice(), &11)));
            }
        }};
    }
    check!(BytesRefFstEnum, u8, InputType::Byte1, 10, 20);
    check!(IntsRefFstEnum, i32, InputType::Byte4, 0x10000, 0x10ffff);
}
