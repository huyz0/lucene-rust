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
    /// `nextNextGaussian`, when `haveNextNextGaussian`: the second value of
    /// the last polar-method pair.
    next_next_gaussian: Option<f64>,
}

impl JavaRandom {
    /// `new Random(seed)`.
    pub fn new(seed: i64) -> Self {
        JavaRandom {
            seed: (seed as u64 ^ MULTIPLIER) & MASK,
            next_next_gaussian: None,
        }
    }

    /// `setSeed(seed)`: as `new Random(seed)`, dropping a pending Gaussian.
    pub fn set_seed(&mut self, seed: i64) {
        *self = JavaRandom::new(seed);
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
    // CrankyTermVectors addProx remained 0.78x Java after payload allocation
    // removal. Expose this boundary so constant bounds avoid variable division.
    #[inline]
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

    /// `nextBytes(bytes)`: one `nextInt()` per four bytes, low byte first.
    pub fn next_bytes(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(4) {
            let mut rnd = self.next_int();
            for b in chunk {
                *b = rnd as u8;
                rnd >>= 8;
            }
        }
    }

    /// `nextGaussian()`: the legacy polar method `java.util.Random` keeps
    /// (Marsaglia), over `StrictMath.log` and `StrictMath.sqrt`, caching the
    /// pair's second value for the next call.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(g) = self.next_next_gaussian.take() {
            return g;
        }
        loop {
            let v1 = 2.0 * self.next_double() - 1.0;
            let v2 = 2.0 * self.next_double() - 1.0;
            let s = v1 * v1 + v2 * v2;
            if s < 1.0 && s != 0.0 {
                let multiplier = (-2.0 * crate::strict_math::log(s) / s).sqrt();
                self.next_next_gaussian = Some(v2 * multiplier);
                return v1 * multiplier;
            }
        }
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

    const GAUSS: [u64; 3] = [0x3ff1a4d76a1d7504, 0xbff26b7713e0ae21, 0xbff3b3e821a12136];

    /// `new Random(42)`: `nextBytes(new byte[6])`, then `nextGaussian()`
    /// three times (the pair, then a fresh pair), on a JDK.
    #[test]
    fn bytes_and_gaussians_match_the_jdk() {
        let mut r = JavaRandom::new(42);
        let mut b = [0u8; 6];
        r.next_bytes(&mut b);
        assert_eq!(b, [0x35, 0x9d, 0x41, 0xba, 0xf7, 0x8a]);
        assert_eq!(r.next_gaussian().to_bits(), GAUSS[0]);
        assert_eq!(r.next_gaussian().to_bits(), GAUSS[1]);
        assert_eq!(r.next_gaussian().to_bits(), GAUSS[2]);
        r.set_seed(42);
        assert_eq!(r.next_int(), -1170105035);
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
