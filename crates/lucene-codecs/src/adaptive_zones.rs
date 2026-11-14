//! Per-window statistics of an `Adaptive90` value array: for every
//! [`ZONE_SIZE`] values, their minimum, maximum and -- when every window's
//! fits an `i64` -- exact sum.
//!
//! An aggregation that accepts every document of a window of a dense,
//! single-valued field reads the window's statistics instead of its values:
//! `min`, `max`, `value_count` and -- for integers, where the running sum
//! stays an exact integer -- `sum` and `avg` come out the same as the
//! value-by-value fold, without decoding the window (see `lucene-aggs`'
//! metric fold).
//!
//! Only `Adaptive90` writes them, behind the
//! [`ADAPTIVE_ZONES_TABLE_SIZE`](crate::doc_values) marker, and only for a
//! value array whose values are documents in order (a dense NUMERIC field,
//! a dense SORTED_NUMERIC field with one value per document, or the ordinals of
//! a dense SORTED or single-valued SORTED_SET field, which carry no sums).
//!
//! # Layout
//!
//! Right after the value array's block-address descriptor in `.dvd`:
//!
//! ```text
//! zone_shift: u8                 log2 of the values per zone
//! has_sums: u8                   1 when the sum array follows, else 0
//! 2 or 3 x (min, max[, sum] arrays, one entry per zone):
//!   base: i64 (little-endian)    the array's smallest entry
//!   bits_per_value: u8           0..=64
//!   packed: ceil(zones * bits / 8) bytes, entry i = base + bits i*bpv..(i+1)*bpv
//!   PADDING zero bytes
//! ```
//!
//! The packed entries are a little-endian bit stream; the padding lets a
//! reader always load 16 bytes at an entry's first byte.

use lucene_store::Error;

/// log2 of [`ZONE_SIZE`].
pub const ZONE_SHIFT: u32 = 10;
/// Values per zone: the metric fold's window.
pub const ZONE_SIZE: usize = 1 << ZONE_SHIFT;
/// Zero bytes after each packed array.
const PADDING: usize = 16;

/// One zone's statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneStats {
    pub min: i64,
    pub max: i64,
    /// The exact sum of the zone's values; `None` for every zone of an
    /// array where some zone's sum does not fit an `i64` (a `double`
    /// field's sortable longs, typically).
    pub sum: Option<i64>,
}

/// The zones of `values`, encoded as in the module docs, or `None` when
/// `values` has fewer than [`ZONE_SIZE`] values.
// ARITH: zone counts and bit offsets of an in-memory array; the sums are
// widened to `i128` and checked before narrowing.
#[allow(clippy::arithmetic_side_effects)]
pub fn encode(values: &[i64]) -> Option<Vec<u8>> {
    encode_with(values, true)
}

/// [`encode`], with the sum array only when `sums` (ordinals have no use
/// for sums).
// ARITH: see `encode`.
#[allow(clippy::arithmetic_side_effects)]
pub fn encode_with(values: &[i64], sums: bool) -> Option<Vec<u8>> {
    if values.len() < ZONE_SIZE {
        return None;
    }
    let zones = values.len().div_ceil(ZONE_SIZE);
    let (mut mins, mut maxs, mut sums) = (
        Vec::with_capacity(zones),
        Vec::with_capacity(zones),
        sums.then(|| Vec::with_capacity(zones)),
    );
    for chunk in values.chunks(ZONE_SIZE) {
        let (mut lo, mut hi, mut sum) = (i64::MAX, i64::MIN, 0i128);
        for &v in chunk {
            lo = lo.min(v);
            hi = hi.max(v);
            sum += i128::from(v);
        }
        mins.push(lo);
        maxs.push(hi);
        if let Some(s) = &mut sums {
            match i64::try_from(sum) {
                Ok(sum) => s.push(sum),
                Err(_) => sums = None,
            }
        }
    }
    let mut out = vec![ZONE_SHIFT as u8, u8::from(sums.is_some())];
    for array in [Some(&mins), Some(&maxs), sums.as_ref()]
        .into_iter()
        .flatten()
    {
        write_packed(array, &mut out);
    }
    Some(out)
}

// ARITH: bit offsets of an in-memory array; `v - base` is taken in `u64`.
#[allow(clippy::arithmetic_side_effects)]
fn write_packed(values: &[i64], out: &mut Vec<u8>) {
    let base = values.iter().copied().min().unwrap_or(0);
    let range = values
        .iter()
        .map(|&v| v.wrapping_sub(base) as u64)
        .max()
        .unwrap_or(0);
    let bpv = 64 - range.leading_zeros();
    out.extend_from_slice(&base.to_le_bytes());
    out.push(bpv as u8);
    let start = out.len();
    out.resize(
        start + (values.len() * bpv as usize).div_ceil(8) + PADDING,
        0,
    );
    if bpv == 0 {
        return;
    }
    let packed = &mut out[start..];
    for (i, &v) in values.iter().enumerate() {
        let delta = v.wrapping_sub(base) as u64;
        let bit = i * bpv as usize;
        let mut rest = u128::from(delta) << (bit % 8);
        let mut at = bit / 8;
        while rest != 0 {
            packed[at] |= rest as u8;
            rest >>= 8;
            at += 1;
        }
    }
}

/// One packed array: where its entries start in `.dvd`.
#[derive(Debug, Clone, Copy)]
struct Packed {
    base: i64,
    bits_per_value: u8,
    offset: usize,
}

impl Packed {
    // ARITH: `i < zones` (checked by the caller) and `offset` was checked to
    // leave room for every entry plus the padding, so the 16-byte load is in
    // bounds; the shift is below 8 and `bits_per_value` at most 64.
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&self, data: &[u8], i: usize) -> Option<i64> {
        let bpv = usize::from(self.bits_per_value);
        if bpv == 0 {
            return Some(self.base);
        }
        let bit = i * bpv;
        let at = self.offset + bit / 8;
        let bytes: [u8; 16] = data.get(at..at + 16)?.try_into().ok()?;
        let word = u128::from_le_bytes(bytes) >> (bit % 8);
        let mask = if bpv == 64 {
            u64::MAX
        } else {
            (1u64 << bpv) - 1
        };
        Some(self.base.wrapping_add(((word as u64) & mask) as i64))
    }
}

/// A value array's zones, parsed from `.dvd`.
#[derive(Debug, Clone)]
pub struct Zones {
    shift: u32,
    zones: usize,
    num_values: usize,
    min: Packed,
    max: Packed,
    sum: Option<Packed>,
}

impl Zones {
    /// The zones that start at `offset` in `data` (the whole `.dvd` file) for
    /// a value array of `num_values` values.
    ///
    /// # Errors
    ///
    /// A zone shift other than [`ZONE_SHIFT`], a sums flag other than 0 or
    /// 1, a bit width over 64, or arrays that run past the end of `data`.
    // ARITH: offsets checked against `data.len()` as they advance.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn parse(data: &[u8], offset: usize, num_values: i64) -> Result<Zones, Error> {
        let eof = || Error::Eof { offset };
        let num_values = usize::try_from(num_values).map_err(|_| eof())?;
        let shift = u32::from(*data.get(offset).ok_or_else(eof)?);
        if shift != ZONE_SHIFT {
            return Err(Error::Corrupted(format!(
                "adaptive zones: shift {shift}, expected {ZONE_SHIFT}"
            )));
        }
        let has_sums = *data.get(offset + 1).ok_or_else(eof)?;
        if has_sums > 1 {
            return Err(Error::Corrupted(format!(
                "adaptive zones: sums flag {has_sums}"
            )));
        }
        let zones = num_values.div_ceil(ZONE_SIZE);
        let mut at = offset + 2;
        let mut arrays = [Packed {
            base: 0,
            bits_per_value: 0,
            offset: 0,
        }; 3];
        for array in &mut arrays[..2 + usize::from(has_sums)] {
            let head = data.get(at..at + 9).ok_or_else(eof)?;
            let base = i64::from_le_bytes(head[..8].try_into().map_err(|_| eof())?);
            let bits_per_value = head[8];
            if bits_per_value > 64 {
                return Err(Error::Corrupted(format!(
                    "adaptive zones: {bits_per_value} bits per value"
                )));
            }
            let len = (zones * usize::from(bits_per_value)).div_ceil(8) + PADDING;
            if data.len() < at + 9 + len {
                return Err(eof());
            }
            *array = Packed {
                base,
                bits_per_value,
                offset: at + 9,
            };
            at += 9 + len;
        }
        let [min, max, sum] = arrays;
        Ok(Zones {
            shift,
            zones,
            num_values,
            min,
            max,
            sum: (has_sums == 1).then_some(sum),
        })
    }

    /// Values per zone.
    pub fn zone_size(&self) -> usize {
        1 << self.shift
    }

    /// The number of zones.
    pub fn len(&self) -> usize {
        self.zones
    }

    /// Whether there are no zones.
    pub fn is_empty(&self) -> bool {
        self.zones == 0
    }

    /// How many values zone `zone` covers: [`Self::zone_size`], fewer for the
    /// last.
    // ARITH: `zone < zones`, so `zone << shift < num_values`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn count(&self, zone: usize) -> usize {
        if zone >= self.zones {
            return 0;
        }
        (self.num_values - (zone << self.shift)).min(self.zone_size())
    }

    /// Zone `zone`'s statistics, `None` past the last zone.
    pub fn get(&self, data: &[u8], zone: usize) -> Option<ZoneStats> {
        if zone >= self.zones {
            return None;
        }
        let sum = match &self.sum {
            Some(sum) => Some(sum.get(data, zone)?),
            None => None,
        };
        Some(ZoneStats {
            min: self.min.get(data, zone)?,
            max: self.max.get(data, zone)?,
            sum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[i64]) {
        let bytes = encode(values).unwrap();
        // Something before and after, as in a real `.dvd`.
        let mut data = vec![0xAB; 7];
        data.extend_from_slice(&bytes);
        data.extend_from_slice(&[0xCD; 3]);
        let zones = Zones::parse(&data, 7, values.len() as i64).unwrap();
        assert_eq!(zones.len(), values.len().div_ceil(ZONE_SIZE));
        for (z, chunk) in values.chunks(ZONE_SIZE).enumerate() {
            let s = zones.get(&data, z).unwrap();
            assert_eq!(s.min, *chunk.iter().min().unwrap());
            assert_eq!(s.max, *chunk.iter().max().unwrap());
            let sum = chunk.iter().map(|&v| i128::from(v)).sum::<i128>();
            if let Some(s) = s.sum {
                assert_eq!(i128::from(s), sum);
            }
            assert_eq!(zones.count(z), chunk.len());
        }
        assert!(zones.get(&data, zones.len()).is_none());
        assert_eq!(zones.count(zones.len()), 0);
    }

    #[test]
    fn zones_read_back() {
        let ts: Vec<i64> = (0..5000).map(|i| 1_700_000_000_000 + i * 37).collect();
        roundtrip(&ts);
        let constant = vec![42i64; 3 * ZONE_SIZE];
        roundtrip(&constant);
        let mut x = 7u64;
        let wide: Vec<i64> = (0..4100)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 12) as i64 - (1 << 50)
            })
            .collect();
        roundtrip(&wide);
        let extremes: Vec<i64> = (0..2048)
            .map(|i| {
                if i % 2 == 0 {
                    i64::MIN / 4096
                } else {
                    i64::MAX / 4096
                }
            })
            .collect();
        roundtrip(&extremes);
        assert!(
            !Zones::parse(&encode(&constant).unwrap(), 0, constant.len() as i64)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn too_short_arrays_get_no_zones_and_overflowing_ones_no_sums() {
        assert!(encode(&[1; ZONE_SIZE - 1]).is_none());
        let mut wide = vec![7i64; 2 * ZONE_SIZE];
        wide[ZONE_SIZE..].fill(i64::MAX);
        roundtrip(&wide);
        let bytes = encode(&wide).unwrap();
        assert_eq!(bytes[1], 0);
        let zones = Zones::parse(&bytes, 0, wide.len() as i64).unwrap();
        for z in 0..2 {
            assert_eq!(zones.get(&bytes, z).unwrap().sum, None);
        }
        assert_eq!(zones.get(&bytes, 1).unwrap().max, i64::MAX);
        let small = encode(&[7i64; 2 * ZONE_SIZE]).unwrap();
        assert_eq!(small[1], 1);
        let zones = Zones::parse(&small, 0, 2 * ZONE_SIZE as i64).unwrap();
        assert_eq!(
            zones.get(&small, 1).unwrap().sum,
            Some(7 * ZONE_SIZE as i64)
        );
    }

    #[test]
    fn corrupt_zones_are_refused() {
        let values = vec![5i64; 2 * ZONE_SIZE];
        let mut bytes = encode(&values).unwrap();
        assert!(Zones::parse(&bytes, 0, values.len() as i64).is_ok());
        assert!(Zones::parse(&bytes[..bytes.len() - 1], 0, values.len() as i64).is_err());
        assert!(Zones::parse(&bytes, bytes.len(), values.len() as i64).is_err());
        assert!(Zones::parse(&bytes, 0, -1).is_err());
        assert!(Zones::parse(&[ZONE_SHIFT as u8], 0, values.len() as i64).is_err());
        bytes[0] = 9;
        assert!(Zones::parse(&bytes, 0, values.len() as i64).is_err());
        bytes[0] = ZONE_SHIFT as u8;
        bytes[1] = 2;
        assert!(Zones::parse(&bytes, 0, values.len() as i64).is_err());
        bytes[1] = 1;
        bytes[10] = 65;
        assert!(Zones::parse(&bytes, 0, values.len() as i64).is_err());
        assert_eq!(
            Zones::parse(&encode(&values).unwrap(), 0, values.len() as i64)
                .unwrap()
                .zone_size(),
            ZONE_SIZE
        );
    }
}
