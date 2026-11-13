//! OpenSearch's `cardinality` sketch for one bucket, built natively: the
//! value hashes `CardinalityAggregator` feeds it (`MurmurHash3.hash128`'s
//! `h1` for a term, `BitMixer.mix64` for a long or a double's bits) and
//! `HyperLogLogPlusPlus` itself -- linear counting over 32-bit encoded hashes
//! in an open-addressing table until it holds more than three quarters of
//! `2^p / 4` of them, then `2^p` HyperLogLog registers -- written as
//! `AbstractHyperLogLogPlusPlus.writeTo` writes it, for the plugin to read
//! back with `readFrom`.
//!
//! The plugin shipped each bucket's distinct values and hashed them into a
//! Java sketch; building the sketch here hands it `2^p` bytes at most per
//! bucket and leaves the JVM no per-value work.
//!
//! Bit-for-bit what Java builds: `tests` compares the hashes and whole
//! written sketches (the linear-counting table in its slot order included)
//! with vectors from OpenSearch 3.8.0's own classes
//! (`opensearch-plugin/tools/GenHll.java`).

/// `AbstractLinearCounting.P2`: the bits an encoded hash keeps.
const P2: u32 = 25;

/// `org.opensearch.common.hash.MurmurHash3.hash128(key, 0, len, seed, hash).h1`
/// (MurmurHash3 x64 128-bit, the first half).
pub fn murmur3_h1(key: &[u8], seed: i64) -> i64 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let (mut h1, mut h2) = (seed as u64, seed as u64);
    let mut blocks = key.chunks_exact(16);
    for b in &mut blocks {
        let k1 = u64::from_le_bytes(b[..8].try_into().expect("8 bytes"));
        let k2 = u64::from_le_bytes(b[8..].try_into().expect("8 bytes"));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    let tail = blocks.remainder();
    let (mut k1, mut k2) = (0u64, 0u64);
    for (i, &b) in tail.iter().enumerate() {
        if i < 8 {
            k1 ^= u64::from(b) << (8 * i);
        } else {
            k2 ^= u64::from(b) << (8 * (i - 8));
        }
    }
    if tail.len() > 8 {
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
    }
    if !tail.is_empty() {
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
    }
    let len = key.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1.wrapping_add(h2) as i64
}

fn fmix(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^ (k >> 33)
}

/// `org.opensearch.common.util.BitMixer.mix64`.
pub fn mix64(z: i64) -> i64 {
    let mut z = z as u64;
    z = (z ^ (z >> 32)).wrapping_mul(0x4cd6_944c_5cc2_0b6d);
    z = (z ^ (z >> 29)).wrapping_mul(0xfc12_c5b1_9d32_59e9);
    (z ^ (z >> 32)) as i64
}

/// One bucket's `HyperLogLogPlusPlus` at precision `p` (4 to 18).
pub struct Sketch {
    p: u32,
    /// Linear counting: `LinearCounting`'s table of encoded hashes (0 is
    /// empty), `2^p / 4` slots, linear probing from `encoded & mask`.
    /// `None` once upgraded.
    table: Option<(Vec<i32>, usize)>,
    /// `HyperLogLog`'s run lengths, one per register, once upgraded.
    registers: Vec<u8>,
}

impl Sketch {
    /// An empty sketch; `None` for a precision OpenSearch refuses
    /// (`AbstractCardinalityAlgorithm`'s 4 to 18).
    pub fn new(p: u32) -> Option<Self> {
        if !(4..=18).contains(&p) {
            return None;
        }
        Some(Self {
            p,
            table: Some((vec![0; (1usize << p) / 4], 0)),
            registers: Vec::new(),
        })
    }

    /// `LinearCounting.threshold`: `(int) (capacity * 0.75f)`.
    fn threshold(&self) -> usize {
        ((1usize << self.p) / 4) * 3 / 4
    }

    /// `HyperLogLogPlusPlus.collect(bucket, hash)`.
    pub fn collect(&mut self, hash: i64) {
        let p = self.p;
        let threshold = self.threshold();
        match &mut self.table {
            Some((slots, size)) => {
                if add_encoded(slots, size, encode_hash(hash, p)) && *size > threshold {
                    self.upgrade();
                }
            }
            None => {
                let index = index(hash, p);
                self.add_run_len(index, run_len(hash, p));
            }
        }
    }

    /// `upgradeToHll`: every encoded hash, in slot order, into registers.
    fn upgrade(&mut self) {
        let Some((slots, _)) = self.table.take() else {
            return;
        };
        self.registers = vec![0; 1 << self.p];
        for encoded in slots.into_iter().filter(|&e| e != 0) {
            self.add_run_len(
                decode_index(encoded, self.p),
                decode_run_len(encoded, self.p),
            );
        }
    }

    fn add_run_len(&mut self, register: usize, run_len: u32) {
        let r = &mut self.registers[register];
        // `runLen` is at most `64 - p + 1 <= 61`: it fits a byte.
        *r = (*r).max(run_len as u8);
    }

    /// `AbstractHyperLogLogPlusPlus.writeTo(bucket, out)`: `vint p`, the
    /// algorithm as a boolean byte, then the linear-counting hashes (`vlong`
    /// count, big-endian ints in slot order) or the `2^p` register bytes.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.push(self.p as u8); // a vint below 128 is its own byte
        match &self.table {
            Some((slots, size)) => {
                out.push(0);
                let mut n = *size as u64;
                while n >= 0x80 {
                    out.push((n as u8) | 0x80);
                    n >>= 7;
                }
                out.push(n as u8);
                for &e in slots.iter().filter(|&&e| e != 0) {
                    out.extend_from_slice(&e.to_be_bytes());
                }
            }
            None => {
                out.push(1);
                out.extend_from_slice(&self.registers);
            }
        }
    }
}

/// `LinearCounting.addEncoded`: whether `encoded` was new.
fn add_encoded(slots: &mut [i32], size: &mut usize, encoded: i32) -> bool {
    let mask = slots.len() - 1;
    let mut i = (encoded as usize) & mask;
    loop {
        match slots[i] {
            0 => {
                slots[i] = encoded;
                *size += 1;
                return true;
            }
            v if v == encoded => return false,
            _ => i = (i + 1) & mask,
        }
    }
}

/// `AbstractLinearCounting.encodeHash`.
fn encode_hash(hash: i64, p: u32) -> i32 {
    let hash = hash as u64;
    let e = hash >> (64 - P2);
    let encoded = if e & ((1u64 << (P2 - p)) - 1) == 0 {
        let run_len = 1 + (hash << P2).leading_zeros().min(64 - P2) as u64;
        (e << 7) | (run_len << 1) | 1
    } else {
        e << 1
    };
    encoded as i32
}

/// `AbstractHyperLogLog.index`.
fn index(hash: i64, p: u32) -> usize {
    ((hash as u64) >> (64 - p)) as usize
}

/// `AbstractHyperLogLog.runLen`.
fn run_len(hash: i64, p: u32) -> u32 {
    1 + ((hash as u64) << p).leading_zeros().min(64 - p)
}

/// `AbstractHyperLogLog.decodeRunLen`.
fn decode_run_len(encoded: i32, p: u32) -> u32 {
    let e = encoded as u32;
    if e & 1 == 1 {
        ((e >> 1) & 0x3F) + (P2 - p)
    } else {
        1 + (e << (31 + p - P2)).leading_zeros()
    }
}

/// `AbstractHyperLogLog.decodeIndex`.
fn decode_index(encoded: i32, p: u32) -> usize {
    let e = encoded as u32;
    let index = if e & 1 == 1 { e >> 7 } else { e >> 1 };
    (index >> (P2 - p)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// From OpenSearch 3.8.0's `MurmurHash3` and `BitMixer`
    /// (`opensearch-plugin/tools/GenHll.java`).
    #[test]
    fn hashes_are_opensearchs() {
        let murmur: &[(&str, i64)] = &[
            ("", 0),
            ("a", -8839064797231613815),
            ("hello", -3758069500696749310),
            ("0123456789abcdef", 5467490433528156583),
            ("0123456789abcdefXYZ", -7362412312553418723),
            ("héllo wörld, a longer term", -5747529415078925469),
        ];
        for (s, h) in murmur {
            assert_eq!(murmur3_h1(s.as_bytes(), 0), *h, "{s:?}");
        }
        let mix: &[(i64, i64)] = &[
            (0, 0),
            (1, -2508561340476696217),
            (-1, 8912229458432966433),
            (42, -335292003828100895),
            (i64::MIN, -6559957559849290354),
            (i64::MAX, -4767257309785776240),
            (2.5f64.to_bits() as i64, -8231812061951526888),
        ];
        for (v, h) in mix {
            assert_eq!(mix64(*v), *h, "{v}");
        }
    }

    /// `(p, seed, n, written length, CRC-32 of the bytes, the bytes when
    /// short)`: `n` longs `seed + i * 7919` through `mix64` into a Java
    /// `HyperLogLogPlusPlus`, written with `writeTo` -- across the
    /// linear-counting threshold at each precision.
    const SKETCHES: &[(u32, i64, u32, usize, u32, &str)] = &[
        (4, -4967725919621401576, 1, 7, 3221990432, "04000102e8760a"),
        (
            4,
            -4627004027837150407,
            3,
            15,
            1272575341,
            "04000301f6a3d800a6877201d24562",
        ),
        (4, 6425179856112732765, 4, 18, 2839851172, ""),
        (4, -1894902459288369262, 20, 18, 2389985963, ""),
        (4, -5383181422176253347, 700, 18, 1494951403, ""),
        (4, 6491681576930330529, 5000, 18, 2623130768, ""),
        (5, 2227187148198412255, 1, 7, 1237375645, "050001039fcf18"),
        (
            5,
            -2768614539681141252,
            3,
            15,
            3227952366,
            "050003002a3be0024e9f48008a47a0",
        ),
        (5, 1535132644386981093, 4, 19, 1499189735, ""),
        (5, -1314366000489323314, 20, 34, 89935270, ""),
        (5, -6611035329062026007, 700, 34, 1416222099, ""),
        (5, -3970480494965416777, 5000, 34, 3169984183, ""),
        (10, -7582205815346941847, 1, 7, 3305394838, "0a000102371b88"),
        (10, 8724841566003434993, 3, 15, 3791211959, ""),
        (10, 4693253153669096614, 4, 19, 2179982715, ""),
        (10, 7058350309194143667, 20, 83, 2754948998, ""),
        (10, -4229898276804867161, 700, 1026, 625517366, ""),
        (10, 4057322764255967705, 5000, 1026, 4192263724, ""),
        (14, 5774083749219235972, 1, 7, 3195826083, "0e00010058e37c"),
        (14, -2459624705444518716, 3, 15, 661077217, ""),
        (14, 1907363412328072160, 4, 19, 2364081067, ""),
        (14, 8139028941982247806, 20, 83, 578789030, ""),
        (14, -8806609805423055690, 700, 2804, 2519285365, ""),
        (14, 1901884635892196386, 5000, 16386, 1775186237, ""),
    ];

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[test]
    fn sketches_are_written_as_opensearch_writes_them() {
        for &(p, seed, n, len, crc, hex) in SKETCHES {
            let mut s = Sketch::new(p).unwrap();
            for i in 0..n {
                s.collect(mix64(seed.wrapping_add(i64::from(i) * 7919)));
            }
            let mut out = Vec::new();
            s.write_to(&mut out);
            let what = format!("p {p}, n {n}");
            assert_eq!(out.len(), len, "{what}");
            assert_eq!(crc32(&out), crc, "{what}");
            if !hex.is_empty() {
                let got: String = out.iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(got, hex, "{what}");
            }
        }
    }

    #[test]
    fn a_repeated_value_counts_once_and_precision_is_bounded() {
        let mut s = Sketch::new(10).unwrap();
        for _ in 0..5 {
            s.collect(mix64(7));
        }
        let mut out = Vec::new();
        s.write_to(&mut out);
        assert_eq!(out.len(), 7, "one linear-counting hash");
        assert!(Sketch::new(3).is_none());
        assert!(Sketch::new(19).is_none());
        // A long linear-counting list writes its count as a multi-byte vlong.
        let mut s = Sketch::new(18).unwrap();
        for i in 0..200 {
            s.collect(mix64(i));
        }
        let mut out = Vec::new();
        s.write_to(&mut out);
        assert_eq!(&out[..4], &[18, 0, 0xc8, 0x01]);
        assert_eq!(out.len(), 4 + 200 * 4);
    }
}
