#![cfg(test)]
use super::*;

fn sortable(x: f64) -> i64 {
    flip(x.to_bits() as i64)
}

fn round_trip(values: &[i64]) -> usize {
    let (bytes, addresses) = encode(values);
    assert_eq!(addresses.len(), values.len().div_ceil(BLOCK_SIZE) + 1);
    for (b, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
        let block = &bytes[addresses[b] as usize..];
        let mut out = vec![0i64; chunk.len()];
        decode_block(block, &mut out).unwrap();
        assert_eq!(out, chunk, "block {b}");
        for (i, &want) in chunk.iter().enumerate() {
            assert_eq!(get_one(block, i).unwrap(), want, "block {b} value {i}");
        }
    }
    bytes.len()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn constant_and_tiny_blocks() {
    assert_eq!(round_trip(&[]), PADDING);
    round_trip(&[7]);
    round_trip(&[i64::MIN]);
    round_trip(&[5; 300]);
    round_trip(&[1, 2]);
}

#[test]
fn extremes_need_all_64_bits() {
    let v: Vec<i64> = (0..300)
        .map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX - i })
        .collect();
    round_trip(&v);
    let mut r = Rng(42);
    let v: Vec<i64> = (0..1000).map(|_| r.next() as i64).collect();
    round_trip(&v);
}

#[test]
fn every_width_round_trips() {
    let mut r = Rng(7);
    for bpv in 0..=64u32 {
        let mask = if bpv == 64 {
            u64::MAX
        } else {
            (1u64 << bpv) - 1
        };
        let v: Vec<i64> = (0..257).map(|_| (r.next() & mask) as i64).collect();
        round_trip(&v);
    }
}

#[test]
fn timestamps_take_the_delta_path() {
    let mut r = Rng(3);
    let mut t = 1_700_000_000_000i64;
    let v: Vec<i64> = (0..1280)
        .map(|_| {
            t += (r.next() % 50) as i64 * 1000;
            t
        })
        .collect();
    let size = round_trip(&v);
    // 50 distinct second steps: 6 bits a value plus headers.
    assert!(size < 1280, "{size}");
    assert_eq!(header(&encode(&v).0).unwrap().token & DELTA, DELTA);
}

#[test]
fn gcd_is_found() {
    let v: Vec<i64> = (0..128).map(|i| 1000 + (i * 37 % 11) * 1000).collect();
    let (bytes, _) = encode(&v);
    let h = header(&bytes).unwrap();
    assert_eq!(h.token & GCD, GCD);
    assert_eq!(h.gcd % 1000, 0);
    round_trip(&v);
}

#[test]
fn decimal_doubles_become_small_integers() {
    let mut r = Rng(11);
    let v: Vec<i64> = (0..1024)
        .map(|_| sortable((r.next() % 10000) as f64 / 100.0))
        .collect();
    let size = round_trip(&v);
    assert!(size < 1024 * 2 + 200, "{size}");
    assert_eq!(header(&encode(&v).0).unwrap().token & DECIMAL, DECIMAL);
}

#[test]
fn awkward_doubles_stay_exact() {
    let specials = [
        -0.0,
        0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        5e-324,
        0.1 + 0.2,
        1e300,
        -123.456,
        9007199254740993.0,
    ];
    let v: Vec<i64> = (0..256)
        .map(|i| sortable(specials[i % specials.len()]))
        .collect();
    round_trip(&v);
    // A block of decimals with one -0.0 cannot take the decimal path.
    let mut v: Vec<i64> = (0..128).map(|i| sortable(i as f64 / 10.0)).collect();
    v[5] = sortable(-0.0);
    round_trip(&v);
    assert_eq!(header(&encode(&v).0).unwrap().token & DECIMAL, 0);
}

#[test]
fn negative_and_decreasing_runs() {
    let v: Vec<i64> = (0..500).map(|i| -10 * i).collect();
    round_trip(&v);
    let v: Vec<i64> = (0..500).map(|i| sortable(-(i as f64) * 0.25)).collect();
    round_trip(&v);
}

#[test]
fn a_short_last_block_of_wide_values_decodes() {
    // A last block of a few wide values sits right before the padding:
    // decoding it must not read past it.
    let mut r = Rng(5);
    for tail in 1..BLOCK_SIZE {
        for bpv in [17u32, 40, 56] {
            let mut v = vec![5i64; BLOCK_SIZE];
            v.extend((0..tail).map(|_| (r.next() >> (64 - bpv)) as i64));
            round_trip(&v);
        }
    }
}

#[test]
fn corrupt_blocks_are_errors() {
    assert!(decode_block(&[], &mut [0; 4]).is_err());
    assert!(decode_block(&[0, 65], &mut [0; 4]).is_err());
    assert!(decode_block(&[DECIMAL, 3, 40], &mut [0; 4]).is_err());
    // A varint that never ends.
    assert!(decode_block(
        &[CONSTANT, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        &mut [0; 4]
    )
    .is_err());
    // Packed values past the end.
    assert!(decode_block(&[0, 8, 0], &mut [0; 4]).is_err());
    assert!(get_one(&[0, 60, 0], 3).is_err());
    assert_eq!(get_one(&[DELTA, 1, 0, 0, 0, 0], 0).unwrap(), 0);
    assert!(get_one(&[], 0).is_err());
    assert!(decode_block(&[DELTA, 1, 0, 0], &mut []).is_ok());
}
