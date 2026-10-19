//! Port of `codecs/bloom/FuzzySet`: a bloom filter over a [`FixedBitSet`]
//! whose size is always `2^n - 1` bits plus one, so a hash is reduced to a
//! bit by `& bloomSize` and a full set can be folded into a smaller one by
//! masking each set bit with the smaller size ([`FuzzySet::downsize`]).
//!
//! # Serialized form (`FuzzySet.serialize`)
//!
//! ```text
//! vInt(hashCount) int(bloomSize) int(numLongs) long[numLongs]
//! ```
//!
//! All little-endian, as Lucene's `DataOutput` writes them; the longs are
//! the bit set's words, bit `i` at `words[i >> 6] >> (i & 63)`.
//!
//! The sizing math (`createOptimalSet`, the deprecated
//! `getNearestSetSize(int, float)` and the collision estimate) calls
//! `Math.log`/`Math.pow`; this port computes them with
//! [`lucene_util::strict_math`] (fdlibm, what `StrictMath` returns), which the
//! fixtures are generated against.

use std::fmt;

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::error::Error;
use lucene_util::fixed_bit_set::{bits2words, FixedBitSet};
use lucene_util::strict_math;
use lucene_util::string_helper::murmurhash3_x64_128_default;

/// `FuzzySet.ContainsResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainsResult {
    /// The value may be in the set.
    Maybe,
    /// The value is certainly not in the set.
    No,
}

/// `FuzzySet.usableBitSetSizes`: `2^(i + 6) - 1` for `i` in `0..26`, the
/// last one `Integer.MAX_VALUE` (Java's `1 << 31` wraps to
/// `Integer.MIN_VALUE`, and minus one wraps back).
pub const USABLE_BIT_SET_SIZES: [i32; 26] = {
    let mut sizes = [0i32; 26];
    let mut i = 0;
    while i < 26 {
        // ARITH: `i + 6 <= 31`, so the shift and the subtraction stay in u64.
        #[allow(clippy::arithmetic_side_effects)]
        {
            sizes[i] = ((1u64 << (i + 6)) - 1) as i32;
            i += 1;
        }
    }
    sizes
};

/// `FuzzySet`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzySet {
    /// `filter`: `bloom_size + 1` bits.
    filter: FixedBitSet,
    /// `bloomSize`: a mask, `2^n - 1`, non-negative.
    bloom_size: i32,
    /// `hashCount`: how many bits each value sets.
    hash_count: i32,
}

/// `(int) d`: Java's narrowing of a double, which saturates and maps NaN to
/// 0 -- exactly what Rust's `as` does.
fn d2i(d: f64) -> i32 {
    d as i32
}

/// `Math.round(double)`: the nearest long, ties towards positive infinity,
/// NaN to 0, saturating.
fn java_round(d: f64) -> i64 {
    if d.is_nan() {
        return 0;
    }
    let floor = d.floor();
    // `d - floor` is exact for every finite `d` (the fraction of a double is
    // representable), so the tie test is exact too.
    let rounded = if d - floor >= 0.5 { floor + 1.0 } else { floor };
    rounded as i64
}

/// `FuzzySet.getNearestSetSize(maxNumberOfBits)`: the largest usable size
/// not above `max_number_of_bits`, the smallest when none is.
pub fn get_nearest_set_size(max_number_of_bits: i32) -> i32 {
    let mut result = USABLE_BIT_SET_SIZES[0];
    for &size in &USABLE_BIT_SET_SIZES {
        if size <= max_number_of_bits {
            result = size;
        }
    }
    result
}

/// `FuzzySet.getNearestSetSize(maxNumberOfValuesExpected,
/// desiredSaturation)`: the first usable size that holds that many values
/// at that saturation, or -1.
// SENTINEL: `-1` = no usable size holds that many values (Java's own result).
pub fn get_nearest_set_size_for_saturation(
    max_number_of_values_expected: i32,
    desired_saturation: f32,
) -> i32 {
    for &size in &USABLE_BIT_SET_SIZES {
        // `(int) (usableBitSetSizes[i] * desiredSaturation)`: a float product.
        let num_set_bits = (size as f32 * desired_saturation) as i32;
        let estimated =
            get_estimated_number_unique_values_allowing_for_collisions(size, num_set_bits, 1);
        if estimated > max_number_of_values_expected {
            return size;
        }
    }
    -1
}

/// `FuzzySet.getEstimatedNumberUniqueValuesAllowingForCollisions(setSize,
/// numRecordedBits, hashCount)`.
pub fn get_estimated_number_unique_values_allowing_for_collisions(
    set_size: i32,
    num_recorded_bits: i32,
    hash_count: i32,
) -> i32 {
    let set_size = f64::from(set_size);
    let saturation = f64::from(num_recorded_bits) / set_size;
    let log_inverse_saturation = -strict_math::log(1.0 - saturation);
    d2i(set_size * log_inverse_saturation / f64::from(hash_count))
}

/// `new FixedBitSet(setSize + 1)` for a `setSize` from
/// [`get_nearest_set_size`]: Java's `setSize + 1` overflows for
/// `Integer.MAX_VALUE` and the constructor throws
/// `NegativeArraySizeException`.
fn bits_for(set_size: i32) -> Result<FixedBitSet, Error> {
    let num_bits = set_size
        .checked_add(1)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            Error::IllegalArgument(format!("bloom filter size {set_size} + 1 overflows an int"))
        })?;
    Ok(FixedBitSet::new(num_bits))
}

impl FuzzySet {
    /// `FuzzySet.createSetBasedOnMaxMemory(maxNumBytes)`: one hash, the
    /// largest usable size within `max_num_bytes * 8` bits.
    ///
    /// # Errors
    /// `IllegalArgument` where Java's bit set constructor throws.
    pub fn create_set_based_on_max_memory(max_num_bytes: i32) -> Result<Self, Error> {
        let set_size = get_nearest_set_size(max_num_bytes.wrapping_mul(8));
        Ok(FuzzySet {
            filter: bits_for(set_size)?,
            bloom_size: set_size,
            hash_count: 1,
        })
    }

    /// `FuzzySet.createSetBasedOnQuality(maxNumUniqueValues,
    /// desiredMaxSaturation, version)`.
    ///
    /// # Errors
    /// `IllegalArgument` where Java throws: no usable size fits (Java's
    /// `new FixedBitSet(0)` then masks with -1 and fails on first use), or
    /// the bit set constructor's overflow.
    pub fn create_set_based_on_quality(
        max_num_unique_values: i32,
        desired_max_saturation: f32,
    ) -> Result<Self, Error> {
        let set_size =
            get_nearest_set_size_for_saturation(max_num_unique_values, desired_max_saturation);
        if set_size < 0 {
            return Err(Error::IllegalArgument(format!(
                "no bloom filter size holds {max_num_unique_values} values at saturation {desired_max_saturation}"
            )));
        }
        Ok(FuzzySet {
            filter: bits_for(set_size)?,
            bloom_size: set_size,
            hash_count: 1,
        })
    }

    /// `FuzzySet.createOptimalSet(maxNumUniqueValues, targetMaxFpp)`: the
    /// size and hash count of the optimal bloom filter for that many values
    /// at that false-positive probability, the size rounded to a usable one
    /// with twice the room.
    ///
    /// # Errors
    /// `IllegalArgument` where Java's bit set constructor throws.
    pub fn create_optimal_set(
        max_num_unique_values: i32,
        target_max_fpp: f32,
    ) -> Result<Self, Error> {
        let n = f64::from(max_num_unique_values);
        let ln2 = strict_math::log(2.0);
        let set_size = d2i(((n * strict_math::log(f64::from(target_max_fpp)))
            / strict_math::log(1.0 / strict_math::pow(2.0, ln2)))
        .ceil());
        let set_size = get_nearest_set_size(set_size.wrapping_mul(2));
        // `(int) Math.round(...)`: a long narrowed to its low 32 bits.
        let optimal_k = java_round((f64::from(set_size) / n) * ln2) as i32;
        Ok(FuzzySet {
            filter: bits_for(set_size)?,
            bloom_size: set_size,
            hash_count: optimal_k,
        })
    }

    /// `hashCount`.
    pub fn hash_count(&self) -> i32 {
        self.hash_count
    }

    /// `bloomSize`.
    pub fn bloom_size(&self) -> i32 {
        self.bloom_size
    }

    /// The bit `pos` of the filter, `pos` already masked with `bloomSize`.
    // FBS: every constructor builds `filter` with `bloom_size + 1` bits and
    // `bloom_size >= 0` (`bits_for`, `deserialize`, `downsize`), and every
    // caller passes `x & bloom_size`, which lies in `0..=bloom_size` -- so
    // `pos < filter.len()`.
    fn may_contain_value(&self, pos: i32) -> bool {
        let pos = pos & self.bloom_size;
        self.filter.get(pos as usize)
    }

    /// `FuzzySet.contains(value)`: [`ContainsResult::No`] when one of the
    /// value's `hashCount` bits is clear.
    pub fn contains(&self, value: &[u8]) -> ContainsResult {
        let [msb, lsb] = murmurhash3_x64_128_default(value);
        for i in 0..self.hash_count {
            let bloom_pos =
                (lsb.wrapping_add(i64::from(i).wrapping_mul(msb)) as i32) & self.bloom_size;
            if !self.may_contain_value(bloom_pos) {
                return ContainsResult::No;
            }
        }
        ContainsResult::Maybe
    }

    /// `FuzzySet.addValue(value)`.
    pub fn add_value(&mut self, value: &[u8]) {
        let [msb, lsb] = murmurhash3_x64_128_default(value);
        for i in 0..self.hash_count {
            let bloom_pos =
                (lsb.wrapping_add(i64::from(i).wrapping_mul(msb)) as i32) & self.bloom_size;
            // FBS: `bloom_pos` is masked with `bloom_size`, so it lies in
            // `0..=bloom_size`, and `filter` has `bloom_size + 1` bits (see
            // `may_contain_value`).
            self.filter.set(bloom_pos as usize);
        }
    }

    /// `FuzzySet.serialize(out)`.
    pub fn serialize(&self, out: &mut impl DataOutput) {
        out.write_vint(self.hash_count);
        out.write_i32(self.bloom_size);
        let words = self.filter.words();
        // A filter has at most `2^31` bits, so `2^25` words.
        out.write_i32(i32::try_from(words.len()).unwrap_or(i32::MAX));
        for &word in words {
            out.write_i64(word as i64);
        }
    }

    /// `FuzzySet.deserialize(in)`.
    ///
    /// Java takes the long array as it comes; this port refuses what Java
    /// would fail on later or never check: a negative or `Integer.MAX_VALUE`
    /// `bloomSize` (the mask would reach outside the bits), a negative
    /// `numLongs` (`NegativeArraySizeException`), fewer longs than `bloomSize
    /// + 1` bits need (Java's `IllegalArgumentException`), or more longs
    /// than the input holds. Surplus longs past what the bits need are read
    /// and dropped, and bits past `bloomSize` in the last word are cleared:
    /// no `contains` reaches either.
    ///
    /// # Errors
    /// `Corrupted` for the cases above, `Eof` for a truncated input.
    pub fn deserialize<I: DataInput + ?Sized>(input: &mut I) -> Result<Self, Error> {
        let hash_count = input.read_vint()?;
        let bloom_size = input.read_i32()?;
        let num_longs = input.read_i32()?;
        let num_bits = bloom_size
            .checked_add(1)
            .filter(|_| bloom_size >= 0)
            .ok_or_else(|| Error::Corrupted(format!("invalid bloom filter size {bloom_size}")))?;
        let num_longs = usize::try_from(num_longs)
            .map_err(|_| Error::Corrupted(format!("invalid bloom filter numLongs {num_longs}")))?;
        let num_bits = num_bits as usize;
        let words_needed = bits2words(num_bits);
        if words_needed > num_longs {
            return Err(Error::Corrupted(format!(
                "The given long array is too small to hold {num_bits} bits"
            )));
        }
        // Read in bounded chunks: a corrupt `bloomSize` must cost a read
        // past the end, not an up-front allocation of up to 256 MiB.
        let mut words: Vec<u64> = Vec::new();
        let mut chunk = [0i64; 512];
        while words.len() < words_needed {
            // The loop condition keeps `words.len() < words_needed`.
            let n = chunk.len().min(words_needed.saturating_sub(words.len()));
            input.read_i64s(&mut chunk[..n])?;
            words.extend(chunk[..n].iter().map(|&w| w as u64));
        }
        for _ in words_needed..num_longs {
            input.read_i64()?;
        }
        let tail = num_bits & 63;
        if tail != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << tail).wrapping_sub(1);
            }
        }
        Ok(FuzzySet {
            filter: FixedBitSet::from_words(words, num_bits),
            bloom_size,
            hash_count,
        })
    }

    /// `FuzzySet.downsize(targetMaxSaturation)`: the same values folded into
    /// the smallest usable size whose saturation stays within
    /// `target_max_saturation`, or `None` when that is not smaller.
    pub fn downsize(&self, target_max_saturation: f32) -> Option<FuzzySet> {
        let num_bits_set = self.cardinality();
        let mut right_sized_size = self.bloom_size;
        for &candidate in &USABLE_BIT_SET_SIZES {
            let candidate_saturation = num_bits_set as f32 / candidate as f32;
            if candidate_saturation <= target_max_saturation {
                right_sized_size = candidate;
                break;
            }
        }
        if right_sized_size >= self.bloom_size {
            return None;
        }
        // `right_sized_size < bloom_size <= i32::MAX`, so `+ 1` cannot wrap.
        let mut right_sized = FixedBitSet::new((right_sized_size as usize).saturating_add(1));
        self.filter.for_each_set_bit(|bit| {
            // FBS: `bit <= bloom_size` fits an i32, and masking it with
            // `right_sized_size` keeps it within `0..=right_sized_size`, the
            // bits `right_sized` was built with.
            right_sized.set(((bit as i32) & right_sized_size) as usize);
        });
        Some(FuzzySet {
            filter: right_sized,
            bloom_size: right_sized_size,
            hash_count: self.hash_count,
        })
    }

    /// `filter.cardinality()`, an int in Java (`2^31` bits at most).
    fn cardinality(&self) -> i32 {
        i32::try_from(self.filter.cardinality()).unwrap_or(i32::MAX)
    }

    /// `FuzzySet.getEstimatedUniqueValues()`.
    pub fn get_estimated_unique_values(&self) -> i32 {
        get_estimated_number_unique_values_allowing_for_collisions(
            self.bloom_size,
            self.cardinality(),
            self.hash_count,
        )
    }

    /// `FuzzySet.getTargetMaxSaturation()`.
    pub fn target_max_saturation(&self) -> f32 {
        0.5
    }

    /// `FuzzySet.getSaturation()`: set bits over `bloomSize`, in floats.
    pub fn saturation(&self) -> f32 {
        self.cardinality() as f32 / self.bloom_size as f32
    }

    /// `FuzzySet.ramBytesUsed()`: `RamUsageEstimator.sizeOf(long[])`, a
    /// 16-byte array header plus the words, aligned to 8.
    pub fn ram_bytes_used(&self) -> u64 {
        let words = self.filter.words().len() as u64;
        words.saturating_mul(8).saturating_add(16)
    }
}

impl fmt::Display for FuzzySet {
    /// `FuzzySet.toString()`: `FuzzySet(k=4, bits=653/2048)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FuzzySet(k={}, bits={}/{})",
            self.hash_count,
            self.cardinality(),
            self.filter.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

    #[test]
    fn usable_sizes_are_all_ones_up_to_int_max() {
        assert_eq!(USABLE_BIT_SET_SIZES[0], 63);
        assert_eq!(USABLE_BIT_SET_SIZES[1], 127);
        assert_eq!(USABLE_BIT_SET_SIZES[25], i32::MAX);
        for s in USABLE_BIT_SET_SIZES {
            assert_eq!(s & s.wrapping_add(1), 0, "{s} is 2^n - 1");
        }
    }

    #[test]
    fn java_round_ties_up_and_handles_nan() {
        assert_eq!(java_round(2.5), 3);
        assert_eq!(java_round(-2.5), -2);
        assert_eq!(java_round(0.499_999_999_999_999_94), 0);
        assert_eq!(java_round(f64::NAN), 0);
        assert_eq!(java_round(f64::INFINITY), i64::MAX);
    }

    #[test]
    fn quality_sizing_refuses_what_no_size_holds() {
        let s = FuzzySet::create_set_based_on_quality(100, 0.5).unwrap();
        assert_eq!(s.hash_count(), 1);
        assert!(s.bloom_size() >= 127);
        assert!(FuzzySet::create_set_based_on_quality(i32::MAX, 0.1).is_err());
        assert!(bits_for(i32::MAX).is_err());
    }

    #[test]
    fn add_then_contains_never_misses() {
        let mut s = FuzzySet::create_optimal_set(1000, 0.1023).unwrap();
        for i in 0..1000 {
            s.add_value(format!("v{i}").as_bytes());
        }
        for i in 0..1000 {
            assert_eq!(
                s.contains(format!("v{i}").as_bytes()),
                ContainsResult::Maybe
            );
        }
        let misses = (1000..3000)
            .filter(|i| s.contains(format!("v{i}").as_bytes()) == ContainsResult::No)
            .count();
        assert!(misses > 1500, "{misses}");
        assert!(s.ram_bytes_used() > 16);
        assert!(s.get_estimated_unique_values() > 0);
    }

    #[test]
    fn downsize_keeps_every_value_and_refuses_to_grow() {
        let mut s = FuzzySet::create_optimal_set(10_000, 0.1023).unwrap();
        for i in 0..50 {
            s.add_value(&[i as u8, 1]);
        }
        let d = s.downsize(s.target_max_saturation()).unwrap();
        assert!(d.bloom_size() < s.bloom_size());
        assert_eq!(d.hash_count(), s.hash_count());
        for i in 0..50 {
            assert_eq!(d.contains(&[i as u8, 1]), ContainsResult::Maybe);
        }
        // The smallest size already: nothing to gain.
        let small = FuzzySet::create_set_based_on_max_memory(1).unwrap();
        assert_eq!(small.bloom_size(), 63);
        assert!(small.downsize(0.5).is_none());
    }

    #[test]
    fn serialize_round_trips() {
        let mut s = FuzzySet::create_optimal_set(300, 0.01).unwrap();
        for i in 0..300 {
            s.add_value(&(i as u32).to_le_bytes());
        }
        let mut buf = Vec::new();
        s.serialize(&mut buf);
        let back = FuzzySet::deserialize(&mut SliceInput::new(&buf)).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.to_string(), s.to_string());
    }

    fn encoded(hash_count: i32, bloom_size: i32, num_longs: i32, longs: &[i64]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_vint(hash_count);
        buf.write_i32(bloom_size);
        buf.write_i32(num_longs);
        for &l in longs {
            buf.write_i64(l);
        }
        buf
    }

    #[test]
    fn deserialize_refuses_corrupt_sizes() {
        let bad = |b: Vec<u8>| FuzzySet::deserialize(&mut SliceInput::new(&b)).unwrap_err();
        assert!(matches!(bad(encoded(1, -1, 1, &[0])), Error::Corrupted(_)));
        assert!(matches!(
            bad(encoded(1, i32::MAX, 1, &[0])),
            Error::Corrupted(_)
        ));
        assert!(matches!(bad(encoded(1, 63, -1, &[])), Error::Corrupted(_)));
        let short = bad(encoded(1, 127, 1, &[0]));
        assert!(
            short.to_string().contains("too small to hold 128 bits"),
            "{short}"
        );
        // More longs announced than present: a read past the end.
        assert!(matches!(bad(encoded(1, 63, 5, &[0])), Error::Eof { .. }));
    }

    #[test]
    fn deserialize_drops_surplus_longs_and_ghost_bits() {
        // 65 bits: two words, the second's bits past bit 0 are ghosts.
        let s =
            FuzzySet::deserialize(&mut SliceInput::new(&encoded(2, 64, 3, &[-1, -1, 7]))).unwrap();
        assert_eq!(s.to_string(), "FuzzySet(k=2, bits=65/65)");
        assert_eq!(s.saturation(), 65.0 / 64.0);
    }
}
