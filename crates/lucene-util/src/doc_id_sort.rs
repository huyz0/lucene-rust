//! Sorting doc ids: what Lucene's `DocIdSetBuilder` does with a buffer of ids
//! collected out of order (a BKD walk visits leaves in value order, not doc
//! order) before handing them out ascending.
//!
//! Doc ids are non-negative and bounded by `maxDoc`, so an LSD radix sort on
//! 11-bit digits needs only as many passes as the largest id has digits --
//! two for a segment under two million documents, three past it -- where a
//! comparison sort pays `log n` comparisons per element. Lucene uses
//! `LSBRadixSorter` for the same reason.

/// Sorts `docs` ascending and removes duplicates -- `DocIdSetBuilder.build`.
/// Negative values (never a real doc id) fall back to a comparison sort, so
/// the result is correct for any input.
pub fn sort_dedup_doc_ids(docs: &mut Vec<i32>) {
    const SMALL: usize = 64;
    // The radix path counts in `u32` (see below), so an input longer than
    // that -- not a real doc-id set -- takes the comparison sort too.
    if docs.len() <= SMALL || docs.len() > u32::MAX as usize || docs.iter().any(|&d| d < 0) {
        docs.sort_unstable();
        docs.dedup();
        return;
    }
    let n = docs.len();
    // `u32` counts (checked above), and every digit's histogram built in one
    // read of the input rather than one read per pass.
    let max = docs.iter().copied().max().unwrap_or(0) as u32;
    let bits = (u32::BITS - max.leading_zeros()).max(1);
    // As few passes as 11-bit digits allow, then the bits spread evenly over
    // them: 23 bits (a 5M-document segment) is three 8-bit digits, not 11 +
    // 11 + 1. A top digit with one or two distinct values sends every
    // increment of its histogram to the same counter, a serial
    // store-to-load chain that cost more than the rest of the sort; and
    // narrower digits make the tables, which are zeroed and prefix-summed per
    // sort, a fraction of the size.
    let passes = bits.div_ceil(11) as usize;
    let width = bits.div_ceil(passes as u32);
    let buckets = 1usize << width;
    let mask = (1u32 << width) - 1;
    let mut hist = vec![0u32; passes * buckets];
    for &d in docs.iter() {
        let v = d as u32;
        for (p, h) in hist.chunks_exact_mut(buckets).enumerate() {
            h[((v >> (width * p as u32)) & mask) as usize] += 1;
        }
    }
    let mut buf = vec![0i32; n];
    let (mut src, mut dst): (&mut [i32], &mut [i32]) = (docs.as_mut_slice(), buf.as_mut_slice());
    let mut in_buf = false;
    for (p, counts) in hist.chunks_exact_mut(buckets).enumerate() {
        // A digit every element shares moves nothing -- `LSBRadixSorter`
        // skips such a pass too.
        if counts.iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut sum = 0u32;
        for c in counts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        let shift = width * p as u32;
        for &d in src.iter() {
            let slot = &mut counts[((d as u32 >> shift) & mask) as usize];
            // SAFETY: after the exclusive prefix sum, each bucket's slot
            // starts at the number of elements in lower buckets and is bumped
            // once per element of its own bucket, so it never exceeds `n -
            // 1`: the histogram was counted over exactly these `n` elements.
            // `dst` has length `n`.
            unsafe { *dst.get_unchecked_mut(*slot as usize) = d };
            *slot += 1;
        }
        std::mem::swap(&mut src, &mut dst);
        in_buf = !in_buf;
    }
    if in_buf {
        std::mem::swap(docs, &mut buf);
    }
    docs.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_comparison_sort_for_every_size_and_range() {
        let mut s = 0x0123_4567_89ab_cdefu64;
        for (len, range) in [
            (0usize, 1u64),
            (5, 10),
            (64, 100),
            (65, 3),
            (1000, 1 << 11),
            (5000, 5_000_000),
            (7000, u32::MAX as u64 >> 1),
        ] {
            let mut docs: Vec<i32> = (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % range) as i32
                })
                .collect();
            let mut want = docs.clone();
            want.sort_unstable();
            want.dedup();
            sort_dedup_doc_ids(&mut docs);
            assert_eq!(docs, want, "len {len} range {range}");
        }
        // Only the low digit varies: the two upper passes are skipped, and the
        // result must still come back in `docs`, not in the scratch buffer.
        // Then only the middle digit, so an odd number of passes runs.
        let spreads: [fn(u64) -> i32; 2] = [
            |x| (1 << 22) + (x % 2048) as i32,
            |x| (1 << 22) + ((x % 2048) << 11) as i32,
        ];
        for spread in spreads {
            let mut docs: Vec<i32> = (0..3000)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    spread(s)
                })
                .collect();
            let mut want = docs.clone();
            want.sort_unstable();
            want.dedup();
            sort_dedup_doc_ids(&mut docs);
            assert_eq!(docs, want);
        }
        let mut with_negative = vec![
            5, -1, 3, 3, 100, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54,
            53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
        ];
        let mut want = with_negative.clone();
        want.sort_unstable();
        want.dedup();
        sort_dedup_doc_ids(&mut with_negative);
        assert_eq!(with_negative, want);
    }
}
