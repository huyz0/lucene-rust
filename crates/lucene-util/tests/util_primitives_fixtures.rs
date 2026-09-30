//! Differential test for the `org.apache.lucene.util` primitives against
//! Lucene 10.5.0: replays `fixtures/data/util_primitives/cases.txt` (written
//! by `fixtures/src/GenUtilPrimitives.java`) -- BitUtil, MathUtil,
//! NumericUtils, SmallFloat, StringHelper (MurmurHash3, the `randomId`
//! stream under `tests.seed`), Version parsing, and random FixedBitSet
//! operation scripts checked after every step.
//!
//! Everything is bit-exact except `MathUtil.log(double, double)` and the
//! inverse hyperbolics, which go through Java's `Math.log`/`Math.sqrt` (a
//! HotSpot intrinsic allowed 1 ulp of error): those are held to 1 ulp.

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::string_helper::{self, IdGenerator};
use lucene_util::version::Version;
use lucene_util::{bit_util, math_util, numeric_utils as nu, small_float as sf};

fn cases() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/util_primitives/cases.txt"
    ))
    .expect("run scripts/gen-fixtures.sh --only GenUtilPrimitives")
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".into();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn d(bits: &str) -> f64 {
    f64::from_bits(bits.parse::<i64>().unwrap() as u64)
}

fn within_one_ulp(got: f64, want: f64) -> bool {
    if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
        return true;
    }
    let (g, w) = (got.to_bits() as i64, want.to_bits() as i64);
    (g >= 0) == (w >= 0) && (g - w).abs() <= 1
}

/// Sign-extends two's-complement big-endian bytes to `n` bytes, so a
/// minimal `BigInteger.toByteArray()` compares with a padded one.
fn sign_extend(b: &[u8], n: usize) -> Vec<u8> {
    let fill = if b.first().is_some_and(|x| x & 0x80 != 0) {
        0xff
    } else {
        0
    };
    let mut v = vec![fill; n.saturating_sub(b.len())];
    v.extend_from_slice(b);
    v
}

fn opt_i(v: Option<usize>) -> String {
    v.map_or("-1".into(), |x| x.to_string())
}

fn fbs_to_hex(s: &FixedBitSet) -> String {
    let bytes: Vec<u8> = s.words().iter().flat_map(|w| w.to_be_bytes()).collect();
    hex(&bytes)
}

fn fbs_from_hex(h: &str, num_bits: usize) -> FixedBitSet {
    let b = unhex(h);
    let words = b
        .chunks(8)
        .map(|c| u64::from_be_bytes(c.try_into().unwrap()))
        .collect();
    FixedBitSet::from_words(words, num_bits)
}

#[test]
fn util_primitives_match_lucene() {
    let text = cases();
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    let mut ids: Option<IdGenerator> = None;
    let mut consts = 0usize;
    // FixedBitSet script state.
    let mut a = FixedBitSet::new(1);
    let mut b = FixedBitSet::new(1);
    let mut num_bits = 1usize;

    for (ln, line) in text.lines().enumerate() {
        let p: Vec<&str> = line.split(' ').collect();
        let ctx = || format!("line {}: {line}", ln + 1);
        let i64_ = |k: usize| p[k].parse::<i64>().unwrap();
        let i32_ = |k: usize| p[k].parse::<i32>().unwrap();
        let us = |k: usize| p[k].parse::<usize>().unwrap();
        *counts.entry(p[0]).or_default() += 1;
        match p[0] {
            // --- BitUtil ---
            "interleave" => {
                assert_eq!(bit_util::interleave(i32_(1), i32_(2)), i64_(3), "{}", ctx())
            }
            "deinterleave" => assert_eq!(bit_util::deinterleave(i64_(1)), i64_(2), "{}", ctx()),
            "flipflop" => assert_eq!(bit_util::flip_flop(i64_(1)), i64_(2), "{}", ctx()),
            "nhp32" => assert_eq!(
                bit_util::next_highest_power_of_two_i32(i32_(1)),
                i32_(2),
                "{}",
                ctx()
            ),
            "nhp64" => assert_eq!(
                bit_util::next_highest_power_of_two_i64(i64_(1)),
                i64_(2),
                "{}",
                ctx()
            ),
            "izp" => assert_eq!(
                bit_util::is_zero_or_power_of_two(i32_(1)).to_string(),
                p[2],
                "{}",
                ctx()
            ),
            "zz" => {
                assert_eq!(
                    bit_util::zig_zag_encode(i64_(1)) as i64,
                    i64_(2),
                    "{}",
                    ctx()
                );
            }
            // --- MathUtil ---
            "log" => {
                let want = if p[3] == "ERR" { None } else { Some(i32_(3)) };
                assert_eq!(math_util::log_long(i64_(1), i32_(2)), want, "{}", ctx());
            }
            "gcd" => assert_eq!(math_util::gcd(i64_(1), i64_(2)), i64_(3), "{}", ctx()),
            "logd" => assert!(
                within_one_ulp(math_util::log_double(d(p[1]), d(p[2])), d(p[3])),
                "{}",
                ctx()
            ),
            "asinh" => assert!(
                within_one_ulp(math_util::asinh(d(p[1])), d(p[2])),
                "{}",
                ctx()
            ),
            "acosh" => assert!(
                within_one_ulp(math_util::acosh(d(p[1])), d(p[2])),
                "{}",
                ctx()
            ),
            "atanh" => assert!(
                within_one_ulp(math_util::atanh(d(p[1])), d(p[2])),
                "{}",
                ctx()
            ),
            "sumrel" => assert_eq!(
                math_util::sum_relative_error_bound(i32_(1)).to_bits(),
                d(p[2]).to_bits(),
                "{}",
                ctx()
            ),
            "sumup" => assert_eq!(
                math_util::sum_upper_bound(d(p[1]), i32_(2)).to_bits(),
                d(p[3]).to_bits(),
                "{}",
                ctx()
            ),
            "umin" => assert_eq!(
                math_util::unsigned_min(i32_(1), i32_(2)),
                i32_(3),
                "{}",
                ctx()
            ),
            // --- NumericUtils ---
            "d2sl" => assert_eq!(nu::double_to_sortable_long(d(p[1])), i64_(2), "{}", ctx()),
            "sl2d" => assert_eq!(
                nu::sortable_long_to_double(i64_(1)).to_bits() as i64,
                i64_(2),
                "{}",
                ctx()
            ),
            "f2si" => assert_eq!(
                nu::float_to_sortable_int(f32::from_bits(i32_(1) as u32)),
                i32_(2),
                "{}",
                ctx()
            ),
            "i2sb" => {
                let mut buf = [0u8; 4];
                nu::int_to_sortable_bytes(i32_(1), &mut buf, 0);
                assert_eq!(hex(&buf), p[2], "{}", ctx());
                assert_eq!(nu::sortable_bytes_to_int(&buf, 0), i32_(1));
            }
            "l2sb" => {
                let mut buf = [0u8; 8];
                nu::long_to_sortable_bytes(i64_(1), &mut buf, 0);
                assert_eq!(hex(&buf), p[2], "{}", ctx());
                assert_eq!(nu::sortable_bytes_to_long(&buf, 0), i64_(1));
            }
            "bi2sb" => {
                let size = us(2);
                let mut buf = vec![0u8; size];
                let got = nu::big_int_to_sortable_bytes(&unhex(p[1]), size, &mut buf, 0)
                    .map(|()| hex(&buf))
                    .unwrap_or_else(|_| "ERR".into());
                assert_eq!(got, p[3], "{}", ctx());
            }
            "sb2bi" => {
                let enc = unhex(p[1]);
                let got = nu::sortable_bytes_to_big_int(&enc, 0, enc.len());
                assert_eq!(got, sign_extend(&unhex(p[2]), enc.len()), "{}", ctx());
            }
            "add" | "sub" => {
                let (bpd, dim) = (us(1), us(2));
                let (x, y) = (unhex(p[3]), unhex(p[4]));
                let mut r = vec![0u8; bpd];
                let res = if p[0] == "add" {
                    nu::add(bpd, dim, &x, &y, &mut r)
                } else {
                    nu::subtract(bpd, dim, &x, &y, &mut r)
                };
                let got = res.map(|()| hex(&r)).unwrap_or_else(|_| "ERR".into());
                assert_eq!(got, p[5], "{}", ctx());
            }
            // --- SmallFloat ---
            "b315" => assert_eq!(
                sf::byte315_to_float(us(1) as u8).to_bits() as i32,
                i32_(2),
                "{}",
                ctx()
            ),
            "b2f" => assert_eq!(
                sf::byte_to_float(us(1) as u8, p[2].parse().unwrap(), i32_(3)).to_bits() as i32,
                i32_(4),
                "{}",
                ctx()
            ),
            "b42i" => assert_eq!(sf::byte4_to_int(us(1) as u8) as i64, i64_(2), "{}", ctx()),
            "f315" => assert_eq!(
                sf::float_to_byte315(f32::from_bits(i32_(1) as u32)) as usize,
                us(2),
                "{}",
                ctx()
            ),
            "f2b" => assert_eq!(
                sf::float_to_byte(
                    f32::from_bits(i32_(1) as u32),
                    p[2].parse().unwrap(),
                    i32_(3)
                ) as usize,
                us(4),
                "{}",
                ctx()
            ),
            "l2i4" => assert_eq!(
                sf::long_to_int4(i64_(1) as u64) as i64,
                i64_(2),
                "{}",
                ctx()
            ),
            "i42l" => assert_eq!(
                sf::int4_to_long(i32_(1) as u32) as i64,
                i64_(2),
                "{}",
                ctx()
            ),
            "i2b4" => assert_eq!(
                sf::int_to_byte4(i32_(1) as u32) as usize,
                us(2),
                "{}",
                ctx()
            ),
            // --- StringHelper ---
            "mm32" => assert_eq!(
                string_helper::murmurhash3_x86_32(&unhex(p[1]), i32_(2)),
                i32_(3),
                "{}",
                ctx()
            ),
            "mm128" => assert_eq!(
                string_helper::murmurhash3_x64_128(&unhex(p[1]), i32_(2)),
                [i64_(3), i64_(4)],
                "{}",
                ctx()
            ),
            "mm128d" => assert_eq!(
                string_helper::murmurhash3_x64_128_default(&unhex(p[1])),
                [i64_(2), i64_(3)],
                "{}",
                ctx()
            ),
            "bdiff" => {
                let got = string_helper::bytes_difference(&unhex(p[1]), &unhex(p[2]));
                assert_eq!(
                    got.map_or("ERR".into(), |v| v.to_string()),
                    p[3],
                    "{}",
                    ctx()
                );
                let skl = string_helper::sort_key_length(&unhex(p[1]), &unhex(p[2]));
                assert_eq!(skl, got.map(|v| v + 1));
            }
            "sw" => assert_eq!(
                string_helper::starts_with(&unhex(p[1]), &unhex(p[2])).to_string(),
                p[3],
                "{}",
                ctx()
            ),
            "ew" => assert_eq!(
                string_helper::ends_with(&unhex(p[1]), &unhex(p[2])).to_string(),
                p[3],
                "{}",
                ctx()
            ),
            "id" => {
                let g = ids.get_or_insert_with(|| IdGenerator::from_tests_seed(p[1]).unwrap());
                assert_eq!(hex(&g.next_id()), p[3], "{}", ctx());
            }
            "idstr" => assert_eq!(
                string_helper::id_to_string(Some(&unhex(p[1]))),
                p[2..].join(" "),
                "{}",
                ctx()
            ),
            // --- Version ---
            "vparse" | "vlenient" => {
                let s = if p[1] == "<empty>" { "" } else { p[1] };
                let got = if p[0] == "vparse" {
                    Version::parse(s)
                } else {
                    Version::parse_leniently(s)
                };
                let got = match got {
                    Ok(v) => format!(
                        "{} {} {} {} {v}",
                        v.major(),
                        v.minor(),
                        v.bugfix(),
                        v.prerelease()
                    ),
                    Err(e) => format!("ERR {e}"),
                };
                assert_eq!(got, p[2..].join(" "), "{}", ctx());
            }
            "vconst" => {
                consts += 1;
                let v = match p[1] {
                    "LATEST" => Version::LATEST,
                    "LUCENE_CURRENT" => Version::LUCENE_CURRENT,
                    name => {
                        Version::ALL
                            .iter()
                            .find(|(n, _)| *n == name)
                            .unwrap_or_else(|| panic!("{}", ctx()))
                            .1
                    }
                };
                assert_eq!(v.to_string(), p[2], "{}", ctx());
                assert_eq!(v.encoded_value(), i32_(3), "{}", ctx());
            }
            "vmin" => assert_eq!(Version::MIN_SUPPORTED_MAJOR, i32_(1)),
            // --- FixedBitSet scripts ---
            "fbs" => {
                num_bits = us(1);
                a = FixedBitSet::new(num_bits);
                b = FixedBitSet::new(num_bits);
            }
            "set" => a.set(us(1)),
            "clear" => a.clear(us(1)),
            "getAndSet" => assert_eq!(a.get_and_set(us(1)).to_string(), p[2], "{}", ctx()),
            "getAndClear" => assert_eq!(a.get_and_clear(us(1)).to_string(), p[2], "{}", ctx()),
            "flip" => a.flip(us(1)),
            "flipRange" => a.flip_range(us(1), us(2)),
            "setRange" => a.set_range(us(1), us(2)),
            "clearRange" => a.clear_range(us(1), us(2)),
            "prevSetBit" => assert_eq!(opt_i(a.prev_set_bit(us(1))), p[2], "{}", ctx()),
            "nextSetBit" => {
                let want = if p[2] == "2147483647" { "-1" } else { p[2] };
                assert_eq!(opt_i(a.next_set_bit(us(1))), want, "{}", ctx());
            }
            "nextSetBitRange" if p[3] != "-" => {
                let want = if p[3] == "2147483647" { "-1" } else { p[3] };
                assert_eq!(
                    opt_i(a.next_set_bit_in_range(us(1), us(2))),
                    want,
                    "{}",
                    ctx()
                );
            }
            "nextClearBit" => {
                let want = if p[2] == "2147483647" { "-1" } else { p[2] };
                assert_eq!(opt_i(a.next_clear_bit(us(1))), want, "{}", ctx());
            }
            "nextClearBitRange" if p[3] != "-" => {
                let want = if p[3] == "2147483647" { "-1" } else { p[3] };
                assert_eq!(
                    opt_i(a.next_clear_bit_in_range(us(1), us(2))),
                    want,
                    "{}",
                    ctx()
                );
            }
            "nextSetBitRange" | "nextClearBitRange" => {}
            "cardRange" => assert_eq!(a.cardinality_range(us(1), us(2)), us(3), "{}", ctx()),
            "bset" => b = fbs_from_hex(p[1], num_bits),
            "counts" => {
                let got = format!(
                    "{} {} {} {}",
                    FixedBitSet::union_count(&a, &b),
                    FixedBitSet::and_not_count(&a, &b),
                    FixedBitSet::intersection_count(&a, &b),
                    a.intersects(&b)
                );
                assert_eq!(got, p[1..].join(" "), "{}", ctx());
            }
            "xor" => a.xor(&b),
            "orRange" => FixedBitSet::or_range(&b, us(1), &mut a, us(2), us(3)),
            "andRange" => FixedBitSet::and_range(&b, us(1), &mut a, us(2), us(3)),
            "orMask" => a.or_mask(us(1), i64_(2) as u64, us(3)),
            "state" => {
                let got = format!(
                    "{} {} {} {}",
                    a.cardinality(),
                    a.approximate_cardinality(),
                    a.java_hash_code(),
                    a.scan_is_empty()
                );
                assert_eq!(got, p[1..].join(" "), "{}", ctx());
            }
            "intoArray" => {
                let mut arr = vec![0i32; num_bits];
                let n = a.into_array(us(1), us(2), 7, &mut arr);
                let got = if n == 0 {
                    "-".to_string()
                } else {
                    arr[..n]
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                };
                assert_eq!(format!("{n} {got}"), p[3..].join(" "), "{}", ctx());
                let mut seen = Vec::new();
                a.for_each_in_range(us(1), us(2), 7, |x| seen.push(x));
                assert_eq!(seen, arr[..n].to_vec());
            }
            "ensureCapacity" => {
                let g = FixedBitSet::ensure_capacity(a.clone(), us(1));
                assert_eq!(
                    format!("{} {}", g.len(), g.cardinality()),
                    p[2..].join(" "),
                    "{}",
                    ctx()
                );
            }
            "sum" => assert_eq!(
                format!("{} {}", a.java_hash_code(), a.cardinality()),
                p[1..].join(" "),
                "{}",
                ctx()
            ),
            "words" => assert_eq!(fbs_to_hex(&a), p[1], "{}", ctx()),
            other => panic!("unknown op {other} at {}", ctx()),
        }
    }
    assert_eq!(
        consts,
        Version::ALL.len() + 2,
        "every Java constant is ported"
    );
    for (op, min) in [
        ("mm128", 150),
        ("fbs", 40),
        ("vparse", 39),
        ("id", 20),
        ("words", 40),
    ] {
        assert!(
            counts.get(op).copied().unwrap_or(0) >= min,
            "{op}: {counts:?}"
        );
    }
}
