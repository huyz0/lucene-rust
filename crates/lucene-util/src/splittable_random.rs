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
    fn mix32(mut z: u64) -> i32 {
        z = (z ^ (z >> 33)).wrapping_mul(0x62a9_d9ed_7997_05f5);
        ((z ^ (z >> 28)).wrapping_mul(0xcb24_d0a5_c88c_35b3) >> 32) as i32
    }

    fn next_int_raw(&mut self) -> i32 {
        self.seed = self.seed.wrapping_add(Self::GOLDEN_GAMMA);
        Self::mix32(self.seed)
    }

    /// `SplittableRandom.nextInt(origin, bound)` (`RandomSupport.
    /// boundedNextInt`): uniform in `origin..bound`; `origin` itself when the
    /// range is empty.
    pub fn next_int_range(&mut self, origin: i32, bound: i32) -> i32 {
        let mut r = self.next_int_raw();
        if origin < bound {
            let n = bound.wrapping_sub(origin);
            let m = n.wrapping_sub(1);
            if n & m == 0 {
                r = (r & m).wrapping_add(origin);
            } else if n > 0 {
                let mut u = ((r as u32) >> 1) as i32;
                loop {
                    r = u % n;
                    if u.wrapping_add(m).wrapping_sub(r) >= 0 {
                        break;
                    }
                    u = ((self.next_int_raw() as u32) >> 1) as i32;
                }
                r = r.wrapping_add(origin);
            } else {
                while r < origin || r >= bound {
                    r = self.next_int_raw();
                }
            }
        }
        r
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

    #[test]
    fn bounded_ints_stay_in_range() {
        let mut r = SplittableRandom::new(9);
        for (o, b) in [
            (0, 1),
            (3, 11),
            (0, 16),
            (-5, 5),
            (i32::MIN, i32::MAX),
            (7, 7),
        ] {
            for _ in 0..200 {
                let v = r.next_int_range(o, b);
                if o < b {
                    assert!(v >= o && v < b, "{v} not in {o}..{b}");
                }
            }
        }
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
