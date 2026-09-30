//! `java.util.Random`: the 48-bit linear congruential generator, bit for bit.
//!
//! Lucene seeds one in a few places whose output reaches the index -- the
//! legacy `ScalarQuantizer`'s reservoir sample is one -- so a port that must
//! pick the same sample has to draw the same numbers.

const MULTIPLIER: u64 = 0x5DEECE66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;

/// `java.util.Random` (not thread-safe; Java's is, via an `AtomicLong`).
#[derive(Debug, Clone)]
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    /// `new Random(seed)`.
    pub fn new(seed: i64) -> Self {
        JavaRandom {
            seed: (seed as u64 ^ MULTIPLIER) & MASK,
        }
    }

    /// `next(bits)`.
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND) & MASK;
        (self.seed >> (48 - bits)) as i64 as i32
    }

    /// `nextInt()`.
    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    /// `nextInt(bound)`. Panics on a non-positive bound, as Java throws.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");
        let mut r = self.next(31);
        let m = bound - 1;
        if bound & m == 0 {
            // power of two
            return ((bound as i64 * r as i64) >> 31) as i32;
        }
        let mut u = r;
        loop {
            r = u % bound;
            if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                return r;
            }
            u = self.next(31);
        }
    }

    /// `nextLong()`.
    pub fn next_long(&mut self) -> i64 {
        ((self.next(32) as i64) << 32).wrapping_add(self.next(32) as i64)
    }

    /// `nextBoolean()`.
    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }

    /// `nextFloat()`.
    pub fn next_float(&mut self) -> f32 {
        self.next(24) as f32 / (1 << 24) as f32
    }

    /// `nextDouble()`.
    pub fn next_double(&mut self) -> f64 {
        (((self.next(26) as i64) << 27) + self.next(27) as i64) as f64
            * (1.0f64 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values printed by `new java.util.Random(42)` on a JDK.
    #[test]
    fn matches_the_jdk_sequence() {
        let mut r = JavaRandom::new(42);
        assert_eq!(r.next_int(), -1170105035);
        assert_eq!(r.next_int_bounded(10), 3);
        assert_eq!(r.next_int_bounded(16), 10);
        let mut r = JavaRandom::new(0);
        assert_eq!(r.next_long(), -4962768465676381896);
        let _ = r.next_boolean();
        let f = r.next_float();
        assert!((0.0..1.0).contains(&f));
        let d = r.next_double();
        assert!((0.0..1.0).contains(&d));
    }

    #[test]
    fn bounded_is_in_range_for_awkward_bounds() {
        let mut r = JavaRandom::new(7);
        for bound in [1, 2, 3, 7, 1000, i32::MAX] {
            for _ in 0..100 {
                let v = r.next_int_bounded(bound);
                assert!((0..bound).contains(&v));
            }
        }
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn zero_bound_panics() {
        JavaRandom::new(1).next_int_bounded(0);
    }
}
