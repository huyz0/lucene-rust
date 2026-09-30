//! Port of `java.util.SplittableRandom`'s root stream.
//!
//! A JDK class, not a Lucene one -- it lives here because Lucene's HNSW graph
//! builder draws every node's level from it, and a port that draws from any
//! other stream cannot be compared against Java's graph node for node. See
//! `lucene_codecs::hnsw::HnswGraphBuilder`.

/// Bit-exact port of `java.util.SplittableRandom`'s root stream (no `split()`,
/// which the HNSW builder never calls).
///
/// This exists so that a graph built here assigns the same level to ordinal
/// `n` as Java's builder does from the same seed. Getting the *distribution*
/// right would be enough for recall; getting the *stream* right is what makes
/// "our recall vs Lucene's recall on this fixture" a comparison of the graph
/// algorithm rather than of two different random draws.
///
/// Verified against `new SplittableRandom(42).nextDouble()` on the JDK -- see
/// `splittable_random_matches_java` and the `splittable_random` section of
/// `fixtures/data/vectors/manifest.properties`.
#[derive(Debug, Clone)]
pub struct SplittableRandom {
    seed: u64,
}

impl SplittableRandom {
    /// `SplittableRandom.GOLDEN_GAMMA`, the fixed gamma of a root instance.
    const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    pub fn new(seed: u64) -> Self {
        SplittableRandom { seed }
    }

    /// `SplittableRandom.mix64`.
    fn mix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `SplittableRandom.nextLong()`.
    pub fn next_long(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(Self::GOLDEN_GAMMA);
        Self::mix64(self.seed)
    }

    /// `SplittableRandom.nextDouble()`: `(nextLong() >>> 11) * 0x1.0p-53`.
    pub fn next_double(&mut self) -> f64 {
        (self.next_long() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `SplittableRandom.mix32`.
    fn mix32(z: u64) -> i32 {
        let z = (z ^ (z >> 33)).wrapping_mul(0x62a9_d9ed_7997_05f5);
        ((z ^ (z >> 28)).wrapping_mul(0xcb24_d0a5_c88c_35b3) >> 32) as u32 as i32
    }

    /// `SplittableRandom.nextInt()`: `mix32(nextSeed())`.
    pub fn next_int(&mut self) -> i32 {
        self.seed = self.seed.wrapping_add(Self::GOLDEN_GAMMA);
        Self::mix32(self.seed)
    }

    /// `SplittableRandom.nextInt(bound)`, i.e. JDK 17+'s
    /// `RandomSupport.boundedNextInt` (the same algorithm JDK 8 inlined):
    /// mask for a power of two, otherwise reject the over-represented top of
    /// the range. `Lucene90HnswVectorsReader` draws its random entry points
    /// from this, so a search over a Lucene 9.0 graph starts where Java's does
    /// only if this is the same stream.
    ///
    /// # Panics
    ///
    /// If `bound <= 0`, as Java throws `IllegalArgumentException`.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");
        let mut r = self.next_int();
        let m = bound.wrapping_sub(1);
        if bound & m == 0 {
            r & m
        } else {
            // `for (int u = r >>> 1; u + m - (r = u % bound) < 0; u = nextInt() >>> 1);`
            // -- `int` arithmetic, so the overflow test is a wrapping one.
            let mut u = ((r as u32) >> 1) as i32;
            loop {
                r = u % bound;
                if u.wrapping_add(m).wrapping_sub(r) >= 0 {
                    return r;
                }
                u = ((self.next_int() as u32) >> 1) as i32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new SplittableRandom(42).nextDouble()`, as raw double bits recorded
    /// from the JDK. `HnswGraphBuilder` seeds with 42, so this is literally
    /// the stream every graph's level assignment comes from -- if it drifts,
    /// every "our graph matches Lucene's" claim in this repo becomes a
    /// coincidence.
    #[test]
    fn reproduces_the_jdk_stream_for_seed_42() {
        let expected = [
            4604854642168692077i64,
            4594929399376720760,
            4598690451703514086,
            4599872008648626872,
            4585641545927528512,
        ];
        let mut r = SplittableRandom::new(42);
        for want in expected {
            assert_eq!(r.next_double().to_bits() as i64, want);
        }
        // `nextLong` is the same stream one step earlier.
        let mut r2 = SplittableRandom::new(1);
        assert_ne!(r2.next_long(), 0);
    }

    /// `nextDouble` is `(nextLong() >>> 11) * 0x1.0p-53`, so it is always in
    /// `[0, 1)` -- the HNSW level draw takes `-ln(u)`, undefined at 0.
    #[test]
    fn next_double_stays_in_the_unit_interval() {
        let mut random = SplittableRandom::new(7);
        for _ in 0..10_000 {
            let u = random.next_double();
            assert!((0.0..1.0).contains(&u), "{u} outside [0, 1)");
        }
    }

    /// `nextInt()` and `nextInt(bound)` as the JDK (25) printed them: a
    /// non-power-of-two bound, a power of two, one, and a bound just past
    /// 2^30, where the rejection loop can reject.
    #[test]
    fn next_int_reproduces_the_jdk_stream() {
        let mut r = SplittableRandom::new(-3_141_592_653_589_793_238i64 as u64);
        let got: Vec<i32> = [
            1500,
            200,
            1024,
            7,
            1,
            1500,
            1_073_741_825,
            1_073_741_825,
            1_073_741_825,
        ]
        .iter()
        .map(|&b| r.next_int_bounded(b))
        .collect();
        assert_eq!(
            got,
            [
                437,
                144,
                789,
                1,
                0,
                454,
                161_817_036,
                718_760_587,
                697_655_645
            ]
        );
        let mut q = SplittableRandom::new(99);
        assert_eq!([q.next_int(), q.next_int()], [-801_679_754, -1_217_134_426]);
    }

    #[test]
    #[should_panic(expected = "bound must be positive")]
    fn next_int_rejects_a_non_positive_bound() {
        SplittableRandom::new(1).next_int_bounded(0);
    }

    /// The stream is a pure function of the seed: same seed replays, different
    /// seeds diverge.
    #[test]
    fn the_stream_is_a_function_of_the_seed() {
        let draw = |seed: u64| {
            let mut r = SplittableRandom::new(seed);
            (0..4).map(|_| r.next_long()).collect::<Vec<_>>()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }
}
