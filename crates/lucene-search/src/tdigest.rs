//! OpenSearch's `percentiles` state for one bucket, built natively:
//! `TDigestState`, a t-digest 3.3 `MergingDigest` with its defaults (the
//! `K_2` scale function under the weight limit, alternating merge passes,
//! two-level compression, a buffer five times the centroid array).
//!
//! The digest is the whole state, not its centroids alone: Java reads every
//! field back (`MergingDigest`'s arrays and counters, `AbstractTDigest`'s
//! `min`/`max`), so the state an aggregation hands the reduce, or a quantile
//! asked of it, is the one `TDigestPercentilesAggregator` would have built
//! from the same values in the same order.
//!
//! Bit-for-bit what Java builds: `tests` compares whole states with vectors
//! from t-digest 3.3 (`opensearch-plugin/tools/GenTDigest.java`). The only
//! transcendental is one `Math.log` per merge, in the weight limit's
//! normalizer; Rust's `ln` and HotSpot's agree on the vectors.

/// A `MergingDigest` (t-digest 3.3) as `TDigestState(compression)` makes it.
#[derive(Debug, Clone, PartialEq)]
pub struct MergingDigest {
    /// `publicCompression`: the compression asked for.
    pub public_compression: f64,
    /// `compression`: the working one (two-level compression).
    pub compression: f64,
    pub merge_count: i32,
    /// The merged centroids, `mean[..last_used_cell]` and its weights.
    pub last_used_cell: usize,
    pub total_weight: f64,
    pub weight: Vec<f64>,
    pub mean: Vec<f64>,
    /// The values added since the last merge.
    pub unmerged_weight: f64,
    pub temp_used: usize,
    pub temp_weight: Vec<f64>,
    pub temp_mean: Vec<f64>,
    order: Vec<usize>,
    /// `AbstractTDigest.min`/`max`.
    pub min: f64,
    pub max: f64,
}

/// `Double.isNaN(x)` in `MergingDigest.add`: Java throws
/// `IllegalArgumentException("Cannot add NaN to t-digest")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanValue;

impl MergingDigest {
    /// `new MergingDigest(compression)`: `MergingDigest(compression, -1, -1)`.
    pub fn new(compression: f64) -> Self {
        let compression = if compression < 10.0 { 10.0 } else { compression };
        // useWeightLimit: the weight limit needs a little extra room.
        let mut size_fudge = 10.0;
        if compression < 30.0 {
            size_fudge += 20.0;
        }
        let mut size = java_max(2.0 * compression + size_fudge, -1.0) as i32;
        let mut buffer_size = 5 * size;
        if buffer_size <= 2 * size {
            buffer_size = 2 * size;
        }
        // Integer division, as Java's `bufferSize / size`.
        let scale = f64::from(std::cmp::max(1, buffer_size / size - 1));
        let working = scale.sqrt() * compression;
        if f64::from(size) < working + size_fudge {
            size = (working + size_fudge).ceil() as i32;
        }
        if buffer_size <= 2 * size {
            buffer_size = 2 * size;
        }
        let (size, buffer_size) = (size as usize, buffer_size as usize);
        MergingDigest {
            public_compression: compression,
            compression: working,
            merge_count: 0,
            last_used_cell: 0,
            total_weight: 0.0,
            weight: vec![0.0; size],
            mean: vec![0.0; size],
            unmerged_weight: 0.0,
            temp_used: 0,
            temp_weight: vec![0.0; buffer_size],
            temp_mean: vec![0.0; buffer_size],
            order: vec![0; buffer_size],
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// `add(x)`: `add(x, 1)`.
    pub fn add(&mut self, x: f64) -> Result<(), NanValue> {
        if x.is_nan() {
            return Err(NanValue);
        }
        if self.temp_used >= self.temp_weight.len() - self.last_used_cell - 1 {
            self.merge_new_values(false, self.compression);
        }
        let at = self.temp_used;
        self.temp_used += 1;
        self.temp_weight[at] = 1.0;
        self.temp_mean[at] = x;
        self.unmerged_weight += 1.0;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
        Ok(())
    }

    /// `mergeNewValues(force, compression)`.
    fn merge_new_values(&mut self, force: bool, compression: f64) {
        if self.total_weight == 0.0 && self.unmerged_weight == 0.0 {
            return;
        }
        if force || self.unmerged_weight > 0.0 {
            // useAlternatingSort: every other merge runs backwards.
            let backwards = self.merge_count % 2 == 1;
            self.merge(self.unmerged_weight, backwards, compression);
            self.merge_count += 1;
            self.temp_used = 0;
            self.unmerged_weight = 0.0;
        }
    }

    /// `merge(tempMean, tempWeight, tempUsed, null, order, unmergedWeight,
    /// runBackwards, compression)`.
    fn merge(&mut self, unmerged_weight: f64, backwards: bool, compression: f64) {
        let last = self.last_used_cell;
        let mut count = self.temp_used;
        self.temp_mean[count..count + last].copy_from_slice(&self.mean[..last]);
        self.temp_weight[count..count + last].copy_from_slice(&self.weight[..last]);
        count += last;
        stable_sort(&mut self.order[..count], &self.temp_mean);
        self.total_weight += unmerged_weight;
        if backwards {
            self.order[..count].reverse();
        }
        let total = self.total_weight;
        let mut cell = 0;
        self.mean[0] = self.temp_mean[self.order[0]];
        self.weight[0] = self.temp_weight[self.order[0]];
        let mut w_so_far = 0.0;
        // ScaleFunction.K_2.normalizer: compression / Z(compression, n).
        let normalizer = compression / (4.0 * (total / compression).ln() + 24.0);
        let max_q = |q: f64| q * (1.0 - q) / normalizer;
        for i in 1..count {
            let ix = self.order[i];
            let proposed = self.weight[cell] + self.temp_weight[ix];
            // useWeightLimit.
            let q0 = w_so_far / total;
            let q2 = (w_so_far + proposed) / total;
            let mut add_this = proposed <= total * java_min(max_q(q0), max_q(q2));
            if i == 1 || i == count - 1 {
                // The first and last centroids never merge.
                add_this = false;
            }
            if add_this {
                self.weight[cell] += self.temp_weight[ix];
                self.mean[cell] = self.mean[cell]
                    + (self.temp_mean[ix] - self.mean[cell]) * self.temp_weight[ix]
                        / self.weight[cell];
                self.temp_weight[ix] = 0.0;
            } else {
                w_so_far += self.weight[cell];
                cell += 1;
                self.mean[cell] = self.temp_mean[ix];
                self.weight[cell] = self.temp_weight[ix];
                self.temp_weight[ix] = 0.0;
            }
        }
        cell += 1;
        self.last_used_cell = cell;
        if backwards {
            self.mean[..cell].reverse();
            self.weight[..cell].reverse();
        }
        if total > 0.0 {
            self.min = java_min(self.min, self.mean[0]);
            self.max = java_max(self.max, self.mean[cell - 1]);
        }
    }
}

/// `Sort.stableSort(order, values, n)`: the indexes `0..n` ordered by
/// `(values[i], i)` -- a total order, so the randomly pivoted quick sort
/// and insertion sort Java runs leave exactly this.
fn stable_sort(order: &mut [usize], values: &[f64]) {
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    order.sort_by(|&a, &b| {
        let (x, y) = (values[a], values[b]);
        // Java's `<` and `==`: -0.0 and 0.0 tie, broken by index.
        if x < y {
            std::cmp::Ordering::Less
        } else if x > y {
            std::cmp::Ordering::Greater
        } else {
            a.cmp(&b)
        }
    });
}

fn java_min(a: f64, b: f64) -> f64 {
    crate::aggs::java_min(a, b)
}

fn java_max(a: f64, b: f64) -> f64 {
    crate::aggs::java_max(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_util::java_random::JavaRandom;

    /// FNV-1a over the little-endian bytes of each value, as `GenTDigest` folds them.
    fn mix(h: &mut u64, v: i64) {
        for i in 0..8 {
            *h ^= (v as u64 >> (8 * i)) & 0xff;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn state_hash(d: &MergingDigest) -> i64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        let bits = |x: f64| x.to_bits() as i64;
        mix(&mut h, bits(d.public_compression));
        mix(&mut h, bits(d.compression));
        mix(&mut h, i64::from(d.merge_count));
        mix(&mut h, d.last_used_cell as i64);
        mix(&mut h, bits(d.total_weight));
        mix(&mut h, bits(d.unmerged_weight));
        mix(&mut h, d.temp_used as i64);
        mix(&mut h, bits(d.min));
        mix(&mut h, bits(d.max));
        mix(&mut h, d.mean.len() as i64);
        mix(&mut h, d.temp_mean.len() as i64);
        for i in 0..d.last_used_cell {
            mix(&mut h, bits(d.mean[i]));
            mix(&mut h, bits(d.weight[i]));
        }
        for i in 0..d.temp_used {
            mix(&mut h, bits(d.temp_mean[i]));
            mix(&mut h, bits(d.temp_weight[i]));
        }
        h as i64
    }

    /// Whole states from t-digest 3.3's `MergingDigest`
    /// (`opensearch-plugin/tools/GenTDigest.java`): `(compression, kind, n,
    /// hash)` for `n` values of each kind -- Gaussian, small integers with
    /// many ties, signed zeros among integers, and magnitudes across 80
    /// binary orders.
    #[test]
    fn states_are_t_digests() {
        let vectors: &[(f64, i32, i32, i64)] = &[
            (100.0, 0, 0, -5866674160906223567),
            (100.0, 0, 1, 3735494556968509888),
            (100.0, 0, 2, -5775891645816938313),
            (100.0, 0, 3, 959394676427270277),
            (100.0, 0, 50, 6928934339997949104),
            (100.0, 0, 1500, 4556425500873710697),
            (100.0, 0, 20000, -8405303600444154801),
            (100.0, 1, 0, -5866674160906223567),
            (100.0, 1, 1, 8828284827679830057),
            (100.0, 1, 2, -7701354041618919793),
            (100.0, 1, 3, 3306262618991884851),
            (100.0, 1, 50, -6158065055549813252),
            (100.0, 1, 1500, 8925550950845159235),
            (100.0, 1, 20000, -1517440312993760146),
            (100.0, 2, 0, -5866674160906223567),
            (100.0, 2, 1, 1018468664191800528),
            (100.0, 2, 2, -1335666399398988269),
            (100.0, 2, 3, 6704182984673271947),
            (100.0, 2, 50, 2799048750238355109),
            (100.0, 2, 1500, 935869094706402111),
            (100.0, 2, 20000, 6537748066499546796),
            (100.0, 3, 0, -5866674160906223567),
            (100.0, 3, 1, -6337543753160938553),
            (100.0, 3, 2, -1289410647781316405),
            (100.0, 3, 3, -1068736603273679353),
            (100.0, 3, 50, -4731072334723577254),
            (100.0, 3, 1500, -9055744511820844096),
            (100.0, 3, 20000, -2456959362244262959),
            (20.0, 0, 0, -6104665044589467950),
            (20.0, 0, 1, 5387544943175913895),
            (20.0, 0, 2, -4868277080954119240),
            (20.0, 0, 3, 1365258876454959426),
            (20.0, 0, 50, 7031437594486013259),
            (20.0, 0, 1500, 6782846855730930578),
            (20.0, 0, 20000, 6160442922516728831),
            (20.0, 1, 0, -6104665044589467950),
            (20.0, 1, 1, 8258351685995588526),
            (20.0, 1, 2, -4431274842705486616),
            (20.0, 1, 3, -7223844299454391796),
            (20.0, 1, 50, 8559052302305977179),
            (20.0, 1, 1500, -7919022509027014247),
            (20.0, 1, 20000, -5179046044013506591),
            (20.0, 2, 0, -6104665044589467950),
            (20.0, 2, 1, 4719798071349968807),
            (20.0, 2, 2, 2801819443732551252),
            (20.0, 2, 3, 6810364549079082892),
            (20.0, 2, 50, 3906091627360550566),
            (20.0, 2, 1500, -3630734025324959439),
            (20.0, 2, 20000, -6602972996068456864),
            (20.0, 3, 0, -6104665044589467950),
            (20.0, 3, 1, -4356219391146565864),
            (20.0, 3, 2, 8927801464304492036),
            (20.0, 3, 3, 3863429700147958828),
            (20.0, 3, 50, 3513864106205153857),
            (20.0, 3, 1500, 1120793443488827847),
            (20.0, 3, 20000, -4202058258741667870),
            (5.0, 0, 0, 6507312924680812509),
            (5.0, 0, 1, 4533550879580857460),
            (5.0, 0, 2, 476497416795893523),
            (5.0, 0, 3, -7131392414379737935),
            (5.0, 0, 50, 3776542961794659308),
            (5.0, 0, 1500, 8575060053025504929),
            (5.0, 0, 20000, -6161014133613786316),
            (5.0, 1, 0, 6507312924680812509),
            (5.0, 1, 1, -1916304036564217771),
            (5.0, 1, 2, 8165727092909963499),
            (5.0, 1, 3, 1493568573515083343),
            (5.0, 1, 50, 5877743518550317896),
            (5.0, 1, 1500, 1061502932334722265),
            (5.0, 1, 20000, 7259436276492563737),
            (5.0, 2, 0, 6507312924680812509),
            (5.0, 2, 1, -4560942905139787556),
            (5.0, 2, 2, 3087676254283123263),
            (5.0, 2, 3, 829250907546323127),
            (5.0, 2, 50, -7883758159429393039),
            (5.0, 2, 1500, -5572242146544132703),
            (5.0, 2, 20000, -4099345065033450982),
            (5.0, 3, 0, 6507312924680812509),
            (5.0, 3, 1, -2026799820622474557),
            (5.0, 3, 2, 6878559041616225079),
            (5.0, 3, 3, 6924245855277827179),
            (5.0, 3, 50, 8901457095702174342),
            (5.0, 3, 1500, 7872527417710774056),
            (5.0, 3, 20000, 6870585014078677405),
            (300.0, 0, 0, 1053990249688818138),
            (300.0, 0, 1, -6458208966157296129),
            (300.0, 0, 2, -5463862043170683488),
            (300.0, 0, 3, 6798776228189836962),
            (300.0, 0, 50, -8726067418323855509),
            (300.0, 0, 1500, 1772889442372358799),
            (300.0, 0, 20000, -8331302632248494113),
            (300.0, 1, 0, 1053990249688818138),
            (300.0, 1, 1, -2126928075935766618),
            (300.0, 1, 2, 806704279549884216),
            (300.0, 1, 3, -7533934243805186900),
            (300.0, 1, 50, 45737378229433939),
            (300.0, 1, 1500, 3512009717090419666),
            (300.0, 1, 20000, -5810353868454985039),
            (300.0, 2, 0, 1053990249688818138),
            (300.0, 2, 1, 5846162211979371167),
            (300.0, 2, 2, -8564006545081375828),
            (300.0, 2, 3, -5067289326435285220),
            (300.0, 2, 50, 8411087329421749182),
            (300.0, 2, 1500, -3612060347952409250),
            (300.0, 2, 20000, 6859012023775449434),
            (300.0, 3, 0, 1053990249688818138),
            (300.0, 3, 1, -7176435207804593472),
            (300.0, 3, 2, -6144362409848765628),
            (300.0, 3, 3, 8891893781755565476),
            (300.0, 3, 50, -1273020924403934687),
            (300.0, 3, 1500, 4555874044114911478),
            (300.0, 3, 20000, 7448432177273597113),
        ];
        for &(compression, kind, n, want) in vectors {
            let mut r = JavaRandom::new(i64::from(31 * kind + n));
            let mut d = MergingDigest::new(compression);
            for i in 0..n {
                let x = match kind {
                    0 => r.next_gaussian() * 100.0,
                    1 => f64::from(r.next_int_bounded(20)),
                    2 => match i % 3 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => f64::from(i),
                    },
                    _ => (r.next_double() - 0.5) * 2f64.powi(r.next_int_bounded(80) - 40),
                };
                d.add(x).unwrap();
            }
            assert_eq!(state_hash(&d), want, "compression {compression}, kind {kind}, n {n}");
        }
    }

    #[test]
    fn nan_is_refused() {
        let mut d = MergingDigest::new(100.0);
        assert_eq!(d.add(f64::NAN), Err(NanValue));
        assert_eq!(d.temp_used, 0);
    }
}
