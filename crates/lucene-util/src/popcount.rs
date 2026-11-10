//! Population counts over word slices, for [`crate::fixed_bit_set`].
//!
//! The build targets x86-64-v3, whose `popcnt` counts one word per
//! instruction. Where the CPU has AVX-512 VPOPCNTDQ (Ice Lake and later
//! Xeons), `vpopcntq` counts eight words per instruction; HotSpot's C2
//! vectorizes `Long.bitCount` loops to it on those CPUs, so Lucene's
//! `FixedBitSet.cardinality` and `intersectionCount` use it there. The
//! kernels below pick it at run time, through [`crate::cpu::avx512_popcount`].

/// How the two words of a pair combine before counting.
#[derive(Clone, Copy)]
pub(crate) enum Op {
    And,
    Or,
    AndNot,
}

#[inline(always)]
fn apply(op: Op, x: u64, y: u64) -> u64 {
    match op {
        Op::And => x & y,
        Op::Or => x | y,
        Op::AndNot => x & !y,
    }
}

/// Total set bits of `words`.
#[inline]
pub(crate) fn words(words: &[u64]) -> usize {
    // Short slices stay scalar: the feature check and the vector setup cost
    // more than they save under a few cache lines.
    #[cfg(target_arch = "x86_64")]
    if words.len() >= 32 && crate::cpu::avx512_popcount() {
        // SAFETY: `avx512_popcount` checked the CPU supports the target features.
        return unsafe { avx512::words(words) };
    }
    scalar_words(words)
}

/// `sum(popcount(op(a[i], b[i])))` over the first `min(a.len(), b.len())`
/// words.
#[inline]
pub(crate) fn zip(a: &[u64], b: &[u64], op: Op) -> usize {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && crate::cpu::avx512_popcount() {
        // SAFETY: `avx512_popcount` checked the CPU supports the target features.
        return unsafe { avx512::zip(a, b, op) };
    }
    scalar_zip(a, b, op)
}

/// Four independent accumulators rather than one `map().sum()`: a single
/// running sum serialises every `popcnt` behind the previous add, and the
/// loop is then bound by that dependency chain rather than by the
/// instruction's throughput.
pub(crate) fn scalar_words(words: &[u64]) -> usize {
    let mut chunks = words.chunks_exact(4);
    let mut acc = [0usize; 4];
    for w in &mut chunks {
        for k in 0..4 {
            acc[k] = acc[k].wrapping_add(w[k].count_ones() as usize);
        }
    }
    let tail = chunks
        .remainder()
        .iter()
        .map(|w| w.count_ones() as usize)
        .fold(0, usize::wrapping_add);
    acc.iter().fold(tail, |s, v| s.wrapping_add(*v))
}

/// [`scalar_words`] over `op` of two equal-length slices.
pub(crate) fn scalar_zip(a: &[u64], b: &[u64], op: Op) -> usize {
    let mut ac = a.chunks_exact(4);
    let mut bc = b.chunks_exact(4);
    let mut acc = [0usize; 4];
    for (x, y) in (&mut ac).zip(&mut bc) {
        for k in 0..4 {
            acc[k] = acc[k].wrapping_add(apply(op, x[k], y[k]).count_ones() as usize);
        }
    }
    let tail = ac
        .remainder()
        .iter()
        .zip(bc.remainder())
        .map(|(&x, &y)| apply(op, x, y).count_ones() as usize)
        .fold(0, usize::wrapping_add);
    acc.iter().fold(tail, |s, v| s.wrapping_add(*v))
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::Op;
    use std::arch::x86_64::*;

    /// Two 8-word vectors per step into two accumulators, so consecutive
    /// `vpopcntq`s do not wait on one add chain.
    ///
    /// # Safety
    /// The CPU supports AVX-512F and VPOPCNTDQ.
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn words(words: &[u64]) -> usize {
        let mut chunks = words.chunks_exact(16);
        let (mut s0, mut s1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        for c in &mut chunks {
            // SAFETY: `c` holds 16 words; unaligned loads read 8 each.
            let (x, y) = unsafe {
                (
                    _mm512_loadu_si512(c.as_ptr().cast()),
                    _mm512_loadu_si512(c.as_ptr().add(8).cast()),
                )
            };
            s0 = _mm512_add_epi64(s0, _mm512_popcnt_epi64(x));
            s1 = _mm512_add_epi64(s1, _mm512_popcnt_epi64(y));
        }
        let sum = _mm512_reduce_add_epi64(_mm512_add_epi64(s0, s1)) as usize;
        sum.wrapping_add(super::scalar_words(chunks.remainder()))
    }

    const AND: u8 = 0;
    const OR: u8 = 1;
    const AND_NOT: u8 = 2;

    #[inline(always)]
    fn combine<const OP: u8>(x: __m512i, y: __m512i) -> __m512i {
        // SAFETY: only reached from `zip_op`, which enables AVX-512F.
        unsafe {
            match OP {
                AND => _mm512_and_si512(x, y),
                OR => _mm512_or_si512(x, y),
                // `andnot(y, x)` is `!y & x`.
                _ => _mm512_andnot_si512(y, x),
            }
        }
    }

    /// [`words`] over `op` of two equal-length slices.
    ///
    /// # Safety
    /// The CPU supports AVX-512F and VPOPCNTDQ; `a.len() == b.len()`.
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn zip(a: &[u64], b: &[u64], op: Op) -> usize {
        // SAFETY: the caller's contract, passed on. One loop per `op`, so
        // the combine is not chosen per vector.
        unsafe {
            match op {
                Op::And => zip_op::<AND>(a, b),
                Op::Or => zip_op::<OR>(a, b),
                Op::AndNot => zip_op::<AND_NOT>(a, b),
            }
        }
    }

    /// # Safety
    /// As [`zip`].
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn zip_op<const OP: u8>(a: &[u64], b: &[u64]) -> usize {
        let op = match OP {
            AND => Op::And,
            OR => Op::Or,
            _ => Op::AndNot,
        };
        let mut ac = a.chunks_exact(16);
        let mut bc = b.chunks_exact(16);
        let (mut s0, mut s1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        for (x, y) in (&mut ac).zip(&mut bc) {
            // SAFETY: `x` and `y` hold 16 words each; unaligned loads read 8.
            let (v0, v1) = unsafe {
                let (x, y) = (x.as_ptr(), y.as_ptr());
                (
                    combine::<OP>(_mm512_loadu_si512(x.cast()), _mm512_loadu_si512(y.cast())),
                    combine::<OP>(
                        _mm512_loadu_si512(x.add(8).cast()),
                        _mm512_loadu_si512(y.add(8).cast()),
                    ),
                )
            };
            s0 = _mm512_add_epi64(s0, _mm512_popcnt_epi64(v0));
            s1 = _mm512_add_epi64(s1, _mm512_popcnt_epi64(v1));
        }
        let sum = _mm512_reduce_add_epi64(_mm512_add_epi64(s0, s1)) as usize;
        sum.wrapping_add(super::scalar_zip(ac.remainder(), bc.remainder(), op))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_of(n: usize, seed: u64) -> Vec<u64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            })
            .collect()
    }

    #[test]
    fn every_length_matches_a_naive_count() {
        for n in [0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 257] {
            let a = words_of(n, 0x9e37_79b9_7f4a_7c15);
            let b = words_of(n, 0x1234_5678_9abc_def1);
            let naive = |f: &dyn Fn(u64, u64) -> u64| -> usize {
                a.iter()
                    .zip(&b)
                    .map(|(&x, &y)| f(x, y).count_ones() as usize)
                    .sum()
            };
            let ones: usize = a.iter().map(|w| w.count_ones() as usize).sum();
            assert_eq!(words(&a), ones, "words, n={n}");
            assert_eq!(scalar_words(&a), ones, "scalar words, n={n}");
            for (op, f) in [
                (
                    Op::And,
                    &(|x: u64, y: u64| x & y) as &dyn Fn(u64, u64) -> u64,
                ),
                (Op::Or, &|x, y| x | y),
                (Op::AndNot, &|x, y| x & !y),
            ] {
                assert_eq!(zip(&a, &b, op), naive(f), "zip, n={n}");
                assert_eq!(scalar_zip(&a, &b, op), naive(f), "scalar zip, n={n}");
            }
        }
    }

    #[test]
    fn unequal_lengths_count_the_common_prefix() {
        let a = words_of(100, 3);
        let b = words_of(40, 5);
        let naive: usize = a[..40]
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (x & y).count_ones() as usize)
            .sum();
        assert_eq!(zip(&a, &b, Op::And), naive);
    }
}
