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
    order: Vec<u32>,
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
        let compression = if compression < 10.0 {
            10.0
        } else {
            compression
        };
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
        sort_new_then_merge(&mut self.order[..count], &self.temp_mean, self.temp_used);
        self.total_weight += unmerged_weight;
        if backwards {
            self.order[..count].reverse();
        }
        let total = self.total_weight;
        let mut cell = 0;
        let first = self.order[0] as usize;
        self.mean[0] = self.temp_mean[first];
        self.weight[0] = self.temp_weight[first];
        let mut w_so_far = 0.0;
        // ScaleFunction.K_2.normalizer: compression / Z(compression, n).
        let normalizer = compression / (4.0 * (total / compression).ln() + 24.0);
        let max_q = |q: f64| q * (1.0 - q) / normalizer;
        for i in 1..count {
            let ix = self.order[i] as usize;
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

/// The keys sorted and the radix sort's scratch space.
type RadixBuffers = (Vec<(u64, u32)>, Vec<(u64, u32)>);

/// The order key of `v`: integers that compare as `v` does, with -0.0 as 0.0
/// (Java's `<` ties them).
#[inline]
fn order_key(v: f64) -> u64 {
    let bits = if v == 0.0 { 0 } else { v.to_bits() };
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

/// [`stable_sort`] of `values[..order.len()]` when `values[fresh..]` (the
/// centroids of the last merge) is already ascending: only the `fresh` new
/// values are sorted and the two runs merged, new values first on a tie (their
/// indexes are lower), which leaves the same order. Any other input takes the
/// full sort.
fn sort_new_then_merge(order: &mut [u32], values: &[f64], fresh: usize) {
    thread_local! {
        static KEYS: std::cell::RefCell<RadixBuffers> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    let n = order.len();
    let values = &values[..n];
    let old = &values[fresh..];
    let sorted = !values.iter().any(|v| v.is_nan())
        && old.windows(2).all(|w| order_key(w[0]) <= order_key(w[1]));
    if !sorted || fresh == 0 || fresh == n {
        stable_sort(order, values);
        return;
    }
    KEYS.with(|keys| {
        let mut keys = keys.borrow_mut();
        let keys = &mut *keys;
        keys.0.clear();
        keys.0.extend(
            values[..fresh]
                .iter()
                .enumerate()
                .map(|(i, &v)| (order_key(v), i as u32)),
        );
        radix_sort(&mut keys.0, &mut keys.1);
        let keys = &keys.0;
        let (mut a, mut b, mut out) = (0, fresh, 0);
        while a < fresh && b < n {
            let ka = keys[a].0;
            if ka <= order_key(values[b]) {
                order[out] = keys[a].1;
                a += 1;
            } else {
                order[out] = b as u32;
                b += 1;
            }
            out += 1;
        }
        while a < fresh {
            order[out] = keys[a].1;
            a += 1;
            out += 1;
        }
        while b < n {
            order[out] = b as u32;
            b += 1;
            out += 1;
        }
    });
}

/// Sorts `(key, index)` pairs, given in index order, by key then index: an
/// LSD radix sort on the key's bytes (stable, so the index order holds among
/// equal keys), skipping the bytes all keys share.
fn radix_sort(a: &mut Vec<(u64, u32)>, tmp: &mut Vec<(u64, u32)>) {
    let n = a.len();
    if n < 64 {
        a.sort_unstable();
        return;
    }
    let mut hist = [[0u32; 256]; 8];
    for &(k, _) in a.iter() {
        for (d, h) in hist.iter_mut().enumerate() {
            h[(k >> (8 * d)) as usize & 0xff] += 1;
        }
    }
    tmp.clear();
    tmp.resize(n, (0, 0));
    for (d, h) in hist.iter().enumerate() {
        if h.iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut sum = 0u32;
        let mut start = [0u32; 256];
        for (s, &c) in start.iter_mut().zip(h.iter()) {
            *s = sum;
            sum += c;
        }
        for &p in a.iter() {
            let b = (p.0 >> (8 * d)) as usize & 0xff;
            tmp[start[b] as usize] = p;
            start[b] += 1;
        }
        std::mem::swap(a, tmp);
    }
}

/// `Sort.stableSort(order, values, n)`: the indexes `0..n` ordered by
/// `(values[i], i)` -- a total order, so the randomly pivoted quick sort
/// and insertion sort Java runs leave exactly this.
fn stable_sort(order: &mut [u32], values: &[f64]) {
    thread_local! {
        static KEYS: std::cell::RefCell<Vec<u128>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let values = &values[..order.len()];
    if values.iter().any(|v| v.is_nan()) {
        // NaN compares neither below nor above anything: no total order to
        // key on, so the comparator as written.
        for (i, o) in order.iter_mut().enumerate() {
            *o = i as u32;
        }
        order.sort_by(|&a, &b| {
            let (x, y) = (values[a as usize], values[b as usize]);
            // Java's `<` and `==`: -0.0 and 0.0 tie, broken by index.
            if x < y {
                std::cmp::Ordering::Less
            } else if x > y {
                std::cmp::Ordering::Greater
            } else {
                a.cmp(&b)
            }
        });
        return;
    }
    // `(values[i], i)` as one integer: the value's bits in an order-keeping
    // form (-0.0 as 0.0, which Java's `<` ties with it) above the index. The
    // keys are distinct, so any sort leaves the one order.
    KEYS.with(|keys| {
        let mut keys = keys.borrow_mut();
        keys.clear();
        keys.extend(values.iter().enumerate().map(|(i, &v)| {
            let bits = if v == 0.0 { 0 } else { v.to_bits() };
            let key = if bits >> 63 == 1 {
                !bits
            } else {
                bits | 1 << 63
            };
            (u128::from(key) << 64) | i as u128
        }));
        keys.sort_unstable();
        for (o, &k) in order.iter_mut().zip(keys.iter()) {
            *o = k as u32;
        }
    });
}

fn java_min(a: f64, b: f64) -> f64 {
    lucene_search::java_math::java_min(a, b)
}

fn java_max(a: f64, b: f64) -> f64 {
    lucene_search::java_math::java_max(a, b)
}

#[cfg(test)]
mod tests;
