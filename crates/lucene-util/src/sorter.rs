//! Exact ports of Lucene's in-place sorting/selection framework:
//! `org.apache.lucene.util.Sorter` (insertion sort, heap sort),
//! `IntroSorter`, `IntroSelector`, `RadixSelector` and `MSBRadixSorter`.
//!
//! These exist because some callers depend on *which permutation* an
//! algorithm produces, not just on the sorted order: the BKD writer sorts
//! and partitions points by keys that deliberately leave some bytes out, so
//! points with equal keys end up wherever the algorithm's swaps put them --
//! and those positions are written to disk. `slice::sort` would produce a
//! different (equally valid) arrangement and different bytes. So these are
//! swap-for-swap transcriptions, driven through small traits the caller
//! implements over its own storage.
//!
//! One deliberate difference: `IntroSelector` shuffles with an *unseeded*
//! `SplittableRandom` when its recursion budget runs out (a pathological
//! input); here the caller supplies the generator, so a run is reproducible.
//! Java's own output is not in that case.

use crate::splittable_random::SplittableRandom;

/// A comparison target for [`intro_sort`]/[`intro_select`]: Java's
/// `swap`/`setPivot`/`comparePivot`/`compare` hooks.
pub trait IntroTarget {
    /// `swap(i, j)`.
    fn swap(&mut self, i: usize, j: usize);
    /// `setPivot(i)`.
    fn set_pivot(&mut self, i: usize);
    /// `comparePivot(j)`: sign of `pivot - value[j]`.
    fn compare_pivot(&mut self, j: usize) -> i32;
    /// `compare(i, j)`: by default `setPivot(i); comparePivot(j)`.
    fn compare(&mut self, i: usize, j: usize) -> i32 {
        self.set_pivot(i);
        self.compare_pivot(j)
    }
}

/// A byte-addressable target for [`radix_select`]/[`msb_radix_sort`]:
/// `byteAt(i, k)` is the `k`-th key byte of entry `i` (0..=255), or -1 past
/// the end of a variable-length key.
pub trait RadixTarget {
    /// `swap(i, j)`.
    fn swap(&mut self, i: usize, j: usize);
    /// `byteAt(i, k)`.
    fn byte_at(&mut self, i: usize, k: usize) -> i32;
}

const INSERTION_SORT_THRESHOLD: usize = 16;
const SINGLE_MEDIAN_THRESHOLD: usize = 40;

/// `MathUtil.log(x, 2)` for a positive `x`.
fn log2(x: usize) -> i32 {
    if x == 0 {
        0
    } else {
        (usize::BITS - 1 - x.leading_zeros()) as i32
    }
}

/// `Sorter.insertionSort(from, to)`.
pub fn insertion_sort<T: IntroTarget + ?Sized>(t: &mut T, from: usize, to: usize) {
    let mut i = from + 1;
    while i < to {
        let mut current = i;
        i += 1;
        loop {
            let previous = current - 1;
            if t.compare(previous, current) > 0 {
                t.swap(previous, current);
                if previous == from {
                    break;
                }
                current = previous;
            } else {
                break;
            }
        }
    }
}

fn heap_parent(from: usize, i: usize) -> usize {
    ((i - 1 - from) >> 1) + from
}

fn heap_child(from: usize, i: usize) -> usize {
    ((i - from) << 1) + 1 + from
}

fn sift_down<T: IntroTarget + ?Sized>(t: &mut T, mut i: usize, from: usize, to: usize) {
    let mut left = heap_child(from, i);
    while left < to {
        let right = left + 1;
        if t.compare(i, left) < 0 {
            if right < to && t.compare(left, right) < 0 {
                t.swap(i, right);
                i = right;
            } else {
                t.swap(i, left);
                i = left;
            }
        } else if right < to && t.compare(i, right) < 0 {
            t.swap(i, right);
            i = right;
        } else {
            break;
        }
        left = heap_child(from, i);
    }
}

/// `Sorter.heapSort(from, to)`.
pub fn heap_sort<T: IntroTarget + ?Sized>(t: &mut T, from: usize, to: usize) {
    if to - from <= 1 {
        return;
    }
    let mut i = heap_parent(from, to - 1) as i64;
    while i >= from as i64 {
        sift_down(t, i as usize, from, to);
        i -= 1;
    }
    let mut end = to - 1;
    while end > from {
        t.swap(from, end);
        sift_down(t, from, from, end);
        end -= 1;
    }
}

fn median<T: IntroTarget + ?Sized>(t: &mut T, i: usize, j: usize, k: usize) -> usize {
    if t.compare(i, j) < 0 {
        if t.compare(j, k) <= 0 {
            return j;
        }
        return if t.compare(i, k) < 0 { k } else { i };
    }
    if t.compare(j, k) >= 0 {
        return j;
    }
    if t.compare(i, k) < 0 {
        i
    } else {
        k
    }
}

/// Java's three-way partition around the pivot swapped into `from`
/// (Bentley-McIlroy): returns `(j, i)` with `from..=j` less, `i..to`
/// greater.
fn three_way_partition<T: IntroTarget + ?Sized>(
    t: &mut T,
    from: usize,
    to: usize,
    pivot: usize,
) -> (i64, i64) {
    let last = to - 1;
    t.set_pivot(pivot);
    t.swap(from, pivot);
    let mut i = from as i64;
    let mut j = to as i64;
    let mut p = from as i64 + 1;
    let mut q = last as i64;
    loop {
        let mut left_cmp;
        loop {
            i += 1;
            left_cmp = t.compare_pivot(i as usize);
            if left_cmp <= 0 {
                break;
            }
        }
        let mut right_cmp;
        loop {
            j -= 1;
            right_cmp = t.compare_pivot(j as usize);
            if right_cmp >= 0 {
                break;
            }
        }
        if i >= j {
            if i == j && right_cmp == 0 {
                t.swap(i as usize, p as usize);
            }
            break;
        }
        t.swap(i as usize, j as usize);
        if right_cmp == 0 {
            t.swap(i as usize, p as usize);
            p += 1;
        }
        if left_cmp == 0 {
            t.swap(j as usize, q as usize);
            q -= 1;
        }
    }
    i = j + 1;
    let mut k = from as i64;
    while k < p {
        t.swap(k as usize, j as usize);
        k += 1;
        j -= 1;
    }
    let mut k = last as i64;
    while k > q {
        t.swap(k as usize, i as usize);
        k -= 1;
        i += 1;
    }
    (j, i)
}

/// `IntroSorter.sort(from, to)`.
pub fn intro_sort<T: IntroTarget + ?Sized>(t: &mut T, from: usize, to: usize) {
    assert!(from <= to, "'to' must be >= 'from'");
    intro_sort_depth(t, from, to, 2 * log2(to - from));
}

fn intro_sort_depth<T: IntroTarget + ?Sized>(
    t: &mut T,
    mut from: usize,
    mut to: usize,
    mut max_depth: i32,
) {
    while to - from > INSERTION_SORT_THRESHOLD {
        let size = to - from;
        max_depth -= 1;
        if max_depth < 0 {
            heap_sort(t, from, to);
            return;
        }
        let last = to - 1;
        let mid = (from + last) >> 1;
        let pivot = if size <= SINGLE_MEDIAN_THRESHOLD {
            let range = size >> 2;
            median(t, mid - range, mid, mid + range)
        } else {
            let range = size >> 3;
            let double_range = range << 1;
            let first = median(t, from, from + range, from + double_range);
            let middle = median(t, mid - range, mid, mid + range);
            let last_m = median(t, last - double_range, last - range, last);
            median(t, first, middle, last_m)
        };
        let (j, i) = three_way_partition(t, from, to, pivot);
        if j - (from as i64) < (last as i64) - i {
            intro_sort_depth(t, from, (j + 1) as usize, max_depth);
            from = i as usize;
        } else {
            intro_sort_depth(t, i as usize, to, max_depth);
            to = (j + 1) as usize;
        }
    }
    insertion_sort(t, from, to);
}

fn min3<T: IntroTarget + ?Sized>(t: &mut T, i: usize, j: usize, k: usize) -> usize {
    if t.compare(i, j) <= 0 {
        return if t.compare(i, k) <= 0 { i } else { k };
    }
    if t.compare(j, k) <= 0 {
        j
    } else {
        k
    }
}

fn max3<T: IntroTarget + ?Sized>(t: &mut T, i: usize, j: usize, k: usize) -> usize {
    if t.compare(i, j) <= 0 {
        return if t.compare(j, k) < 0 { k } else { j };
    }
    if t.compare(i, k) < 0 {
        k
    } else {
        i
    }
}

fn sort3<T: IntroTarget + ?Sized>(t: &mut T, from: usize) {
    let mid = from + 1;
    let last = from + 2;
    if t.compare(from, mid) <= 0 {
        if t.compare(mid, last) > 0 {
            t.swap(mid, last);
            if t.compare(from, mid) > 0 {
                t.swap(from, mid);
            }
        }
    } else if t.compare(mid, last) >= 0 {
        t.swap(from, last);
    } else {
        t.swap(from, mid);
        if t.compare(mid, last) > 0 {
            t.swap(mid, last);
        }
    }
}

/// `IntroSelector.select(from, to, k)`: rearranges so that `k` holds the
/// value a full sort would put there, smaller-or-equal values before it and
/// greater-or-equal after. `random` stands in for Java's lazily created,
/// unseeded `SplittableRandom` (used only once the depth budget is spent).
pub fn intro_select<T: IntroTarget + ?Sized>(
    t: &mut T,
    from: usize,
    to: usize,
    k: usize,
    random: &mut SplittableRandom,
) {
    assert!(k >= from && k < to, "k must be in from..to");
    intro_select_depth(t, from, to, k, 2 * log2(to - from), random);
}

fn intro_select_depth<T: IntroTarget + ?Sized>(
    t: &mut T,
    mut from: usize,
    mut to: usize,
    k: usize,
    mut max_depth: i32,
    random: &mut SplittableRandom,
) {
    while to - from > 3 {
        let size = to - from;
        max_depth -= 1;
        if max_depth == -1 {
            // Durstenfeld shuffle.
            let mut i = to - 1;
            while i > from {
                let r = random.next_int_range(from as i32, i as i32 + 1) as usize;
                t.swap(i, r);
                i -= 1;
            }
        }
        let last = to - 1;
        let mid = (from + last) >> 1;
        let pivot = if size <= SINGLE_MEDIAN_THRESHOLD {
            let range = size >> 2;
            median(t, mid - range, mid, mid + range)
        } else {
            let range = size >> 3;
            let double_range = range << 1;
            let first = median(t, from, from + range, from + double_range);
            let middle = median(t, mid - range, mid, mid + range);
            let last_m = median(t, last - double_range, last - range, last);
            if k - from < range {
                min3(t, first, middle, last_m)
            } else if to - k <= range {
                max3(t, first, middle, last_m)
            } else {
                median(t, first, middle, last_m)
            }
        };
        let (j, i) = three_way_partition(t, from, to, pivot);
        if (k as i64) <= j {
            to = (j + 1) as usize;
        } else if (k as i64) >= i {
            from = i as usize;
        } else {
            return;
        }
    }
    match to - from {
        2 => {
            if t.compare(from, from + 1) > 0 {
                t.swap(from, from + 1);
            }
        }
        3 => sort3(t, from),
        _ => {}
    }
}

const RADIX_LEVEL_THRESHOLD: usize = 8;
const RADIX_HISTOGRAM_SIZE: usize = 257;
const RADIX_LENGTH_THRESHOLD: usize = 100;

/// `RadixSelector.computeCommonPrefixLengthAndBuildHistogram`: returns the
/// common prefix length of the keys at `k..`, filling `histogram` (bucket
/// `byte + 1`) when it is 0. `with_first_bucket` is `MSBRadixSorter`'s
/// variant, which also records the leading run in the histogram.
fn prefix_and_histogram<T: RadixTarget + ?Sized>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    histogram: &mut [i64; RADIX_HISTOGRAM_SIZE],
    msb_style: bool,
) -> usize {
    let cap = max_length.min(24);
    let mut common_prefix = [0i32; 24];
    let mut len = cap.min(max_length - k);
    let mut j = 0;
    while j < len {
        let b = t.byte_at(from, k + j);
        common_prefix[j] = b;
        if b == -1 {
            len = j + 1;
            break;
        }
        j += 1;
    }
    let mut i = from + 1;
    'outer: while i < to {
        let mut j = 0;
        while j < len {
            let b = t.byte_at(i, k + j);
            if b != common_prefix[j] {
                len = j;
                if len == 0 {
                    if !msb_style {
                        histogram[(common_prefix[0] + 1) as usize] = (i - from) as i64;
                        histogram[(b + 1) as usize] = 1;
                    }
                    break 'outer;
                }
                break;
            }
            j += 1;
        }
        i += 1;
    }
    if i < to {
        if msb_style {
            histogram[(common_prefix[0] + 1) as usize] = (i - from) as i64;
            for x in i..to {
                histogram[(t.byte_at(x, k) + 1) as usize] += 1;
            }
        } else {
            for x in i + 1..to {
                histogram[(t.byte_at(x, k) + 1) as usize] += 1;
            }
        }
    } else {
        histogram[(common_prefix[0] + 1) as usize] = (to - from) as i64;
    }
    len
}

/// `RadixSelector.select(from, to, k)`: `fallback(t, d, from, to, k)` is
/// `getFallbackSelector(d).select(from, to, k)`, used for small ranges and
/// deep recursion.
pub fn radix_select<T, F>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    fallback: &mut F,
) where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize, usize),
{
    assert!(k >= from && k < to, "k must be in from..to");
    radix_select_level(t, max_length, from, to, k, 0, 0, fallback);
}

#[allow(clippy::too_many_arguments)]
fn radix_select_level<T, F>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    d: usize,
    l: usize,
    fallback: &mut F,
) where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize, usize),
{
    if to - from <= RADIX_LENGTH_THRESHOLD || l >= RADIX_LEVEL_THRESHOLD {
        fallback(t, d, from, to, k);
    } else {
        radix_select_step(t, max_length, from, to, k, d, l, fallback);
    }
}

#[allow(clippy::too_many_arguments)]
fn radix_select_step<T, F>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    d: usize,
    l: usize,
    fallback: &mut F,
) where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize, usize),
{
    let mut histogram = [0i64; RADIX_HISTOGRAM_SIZE];
    let common = prefix_and_histogram(t, max_length, from, to, d, &mut histogram, false);
    if common > 0 {
        if d + common < max_length && histogram[0] < (to - from) as i64 {
            radix_select_step(t, max_length, from, to, k, d + common, l, fallback);
        }
        return;
    }
    let mut bucket_from = from;
    for (bucket, &count) in histogram.iter().enumerate() {
        let bucket_to = bucket_from + count as usize;
        if bucket_to > k {
            radix_partition(t, from, to, bucket as i32, bucket_from, bucket_to, d);
            if bucket != 0 && d + 1 < max_length {
                radix_select_level(
                    t,
                    max_length,
                    bucket_from,
                    bucket_to,
                    k,
                    d + 1,
                    l + 1,
                    fallback,
                );
            }
            return;
        }
        bucket_from = bucket_to;
    }
    unreachable!("the histogram covers from..to, and k < to");
}

/// `RadixSelector.partition`.
fn radix_partition<T: RadixTarget + ?Sized>(
    t: &mut T,
    from: usize,
    to: usize,
    bucket: i32,
    bucket_from: usize,
    bucket_to: usize,
    d: usize,
) {
    let mut left = from;
    let mut right = to - 1;
    let mut slot = bucket_from;
    loop {
        let mut left_bucket = t.byte_at(left, d) + 1;
        let mut right_bucket = t.byte_at(right, d) + 1;
        while left_bucket <= bucket && left < bucket_from {
            if left_bucket == bucket {
                t.swap(left, slot);
                slot += 1;
            } else {
                left += 1;
            }
            left_bucket = t.byte_at(left, d) + 1;
        }
        while right_bucket >= bucket && right >= bucket_to {
            if right_bucket == bucket {
                t.swap(right, slot);
                slot += 1;
            } else {
                right -= 1;
            }
            right_bucket = t.byte_at(right, d) + 1;
        }
        if left < bucket_from && right >= bucket_to {
            t.swap(left, right);
            left += 1;
            right -= 1;
        } else {
            break;
        }
    }
}

/// `MSBRadixSorter.sort(from, to)`: `fallback(t, k, from, to)` is
/// `getFallbackSorter(k).sort(from, to)`.
pub fn msb_radix_sort<T, F>(t: &mut T, max_length: usize, from: usize, to: usize, fallback: &mut F)
where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize),
{
    assert!(from <= to, "'to' must be >= 'from'");
    msb_sort_level(t, max_length, from, to, 0, 0, fallback);
}

fn msb_sort_level<T, F>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    l: usize,
    fallback: &mut F,
) where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize),
{
    if to - from <= RADIX_LENGTH_THRESHOLD || l >= RADIX_LEVEL_THRESHOLD {
        fallback(t, k, from, to);
    } else {
        msb_radix_step(t, max_length, from, to, k, l, fallback);
    }
}

fn msb_radix_step<T, F>(
    t: &mut T,
    max_length: usize,
    from: usize,
    to: usize,
    k: usize,
    l: usize,
    fallback: &mut F,
) where
    T: RadixTarget + ?Sized,
    F: FnMut(&mut T, usize, usize, usize),
{
    let mut histogram = [0i64; RADIX_HISTOGRAM_SIZE];
    let common = prefix_and_histogram(t, max_length, from, to, k, &mut histogram, true);
    if common > 0 {
        if k + common < max_length && histogram[0] < (to - from) as i64 {
            msb_radix_step(t, max_length, from, to, k + common, l, fallback);
        }
        return;
    }
    // sumHistogram: start offsets in `histogram`, end offsets beside.
    let mut end_offsets = [0i64; RADIX_HISTOGRAM_SIZE];
    let mut accum = 0i64;
    for i in 0..RADIX_HISTOGRAM_SIZE {
        let count = histogram[i];
        histogram[i] = accum;
        accum += count;
        end_offsets[i] = accum;
    }
    // reorder
    for i in 0..RADIX_HISTOGRAM_SIZE {
        let limit = end_offsets[i];
        let mut h1 = histogram[i];
        while h1 < limit {
            let b = (t.byte_at(from + h1 as usize, k) + 1) as usize;
            let h2 = histogram[b];
            histogram[b] += 1;
            t.swap(from + h1 as usize, from + h2 as usize);
            h1 = histogram[i];
        }
    }
    // After reordering, `histogram` holds each bucket's end offset.
    if k + 1 < max_length {
        let mut prev = histogram[0];
        for &h in histogram.iter().skip(1) {
            if h - prev > 1 {
                msb_sort_level(
                    t,
                    max_length,
                    from + prev as usize,
                    from + h as usize,
                    k + 1,
                    l + 1,
                    fallback,
                );
            }
            prev = h;
        }
    }
}

/// `getFallbackSelector(d)`/`getFallbackSorter(k)`'s default comparison:
/// the keys from byte `d` on, as unsigned bytes with -1 ending a key.
pub fn compare_from<T: RadixTarget + ?Sized>(
    t: &mut T,
    max_length: usize,
    i: usize,
    j: usize,
    d: usize,
) -> i32 {
    for o in d..max_length {
        let b1 = t.byte_at(i, o);
        let b2 = t.byte_at(j, o);
        if b1 != b2 {
            return b1 - b2;
        } else if b1 == -1 {
            break;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorts `(key, tag)` pairs by `key` only, so equal keys expose the
    /// algorithm's arrangement.
    struct Pairs {
        v: Vec<(u32, u32)>,
        pivot: u32,
        swaps: usize,
    }

    impl IntroTarget for Pairs {
        fn swap(&mut self, i: usize, j: usize) {
            self.v.swap(i, j);
            self.swaps += 1;
        }
        fn set_pivot(&mut self, i: usize) {
            self.pivot = self.v[i].0;
        }
        fn compare_pivot(&mut self, j: usize) -> i32 {
            self.pivot.cmp(&self.v[j].0) as i32
        }
    }

    impl RadixTarget for Pairs {
        fn swap(&mut self, i: usize, j: usize) {
            self.v.swap(i, j);
        }
        fn byte_at(&mut self, i: usize, k: usize) -> i32 {
            if k >= 4 {
                -1
            } else {
                i32::from(self.v[i].0.to_be_bytes()[k])
            }
        }
    }

    fn data(n: u32, modulo: u32) -> Pairs {
        Pairs {
            v: (0..n).map(|i| ((i * 7919 + 13) % modulo, i)).collect(),
            pivot: 0,
            swaps: 0,
        }
    }

    fn sorted_keys(p: &Pairs) -> bool {
        p.v.windows(2).all(|w| w[0].0 <= w[1].0)
    }

    #[test]
    fn sorts_and_selects() {
        for (n, m) in [
            (0, 1),
            (1, 1),
            (5, 3),
            (17, 5),
            (40, 100),
            (41, 7),
            (500, 37),
            (3000, 100_000),
        ] {
            let mut p = data(n, m);
            intro_sort(&mut p, 0, n as usize);
            assert!(sorted_keys(&p), "intro {n} {m}");
            let mut h = data(n, m);
            heap_sort(&mut h, 0, n as usize);
            assert!(sorted_keys(&h));
            let mut ins = data(n.min(30), m);
            let len = ins.v.len();
            insertion_sort(&mut ins, 0, len);
            assert!(sorted_keys(&ins));
            let mut r = data(n, m);
            let len = r.v.len();
            msb_radix_sort(&mut r, 4, 0, len, &mut |t: &mut Pairs, _k, f, to| {
                intro_sort(t, f, to)
            });
            assert!(sorted_keys(&r), "msb {n} {m}");
            if n > 0 {
                for k in [0, n as usize / 2, n as usize - 1] {
                    let mut want = data(n, m).v;
                    want.sort_by_key(|x| x.0);
                    let mut s = data(n, m);
                    let mut rng = SplittableRandom::new(1);
                    intro_select(&mut s, 0, n as usize, k, &mut rng);
                    assert_eq!(s.v[k].0, want[k].0);
                    assert!(s.v[..k].iter().all(|x| x.0 <= want[k].0));
                    assert!(s.v[k + 1..].iter().all(|x| x.0 >= want[k].0));
                    let mut rs = data(n, m);
                    radix_select(
                        &mut rs,
                        4,
                        0,
                        n as usize,
                        k,
                        &mut |t: &mut Pairs, _d, f, to, kk| {
                            let mut rng = SplittableRandom::new(2);
                            intro_select(t, f, to, kk, &mut rng)
                        },
                    );
                    assert_eq!(rs.v[k].0, want[k].0, "radix select {n} {m} {k}");
                    assert!(rs.v[..k].iter().all(|x| x.0 <= want[k].0));
                }
            }
        }
        let mut p = data(10, 3);
        assert_eq!(compare_from(&mut p, 4, 0, 0, 0), 0);
        assert_eq!(log2(1), 0);
        assert_eq!(log2(0), 0);
    }

    #[test]
    fn deep_recursion_falls_back() {
        // A budget of 0 forces heap sort / the shuffle immediately.
        let mut p = data(200, 50);
        intro_sort_depth(&mut p, 0, 200, 0);
        assert!(sorted_keys(&p));
        let mut s = data(200, 50);
        let mut rng = SplittableRandom::new(3);
        intro_select_depth(&mut s, 0, 200, 100, 0, &mut rng);
        let mut want = data(200, 50).v;
        want.sort_by_key(|x| x.0);
        assert_eq!(s.v[100].0, want[100].0);
    }
}
