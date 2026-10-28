//! The spell package's string distances (M13 T13.3e): `StringDistance` and
//! its four implementations. Each answers a similarity in `[0, 1]` (1 is
//! equal), computed in `float` the way Java does, operation for operation, so
//! the bits match Lucene's.
//!
//! Java strings are UTF-16: `LevenshteinDistance`, `JaroWinklerDistance` and
//! `NGramDistance` compare UTF-16 code units (a supplementary character is two
//! of them, as in Java), `LuceneLevenshteinDistance` compares code points.
//! A `&str` cannot hold an unpaired surrogate, so neither can these inputs.
//!
//! An ASCII string's bytes are its UTF-16 units and its code points, so two
//! ASCII strings are compared as bytes, without decoding; the dynamic
//! programs run over generic units either way, and their rows live on the
//! stack up to [`STACK_CELLS`] cells. The arithmetic is unchanged, so the
//! answers are bit for bit the same.

use std::fmt;

use crate::function::java_float;

/// `org.apache.lucene.search.spell.StringDistance`.
pub trait StringDistance: fmt::Debug + Send + Sync {
    /// `getDistance(s1, s2)`: 1 for equal strings, less the more they differ.
    fn get_distance(&self, s1: &str, s2: &str) -> f32;
}

/// `Math.min(float, float)`: NaN wins, and `-0.0` is smaller than `0.0`
/// (Rust's `f32::min` ignores a NaN).
fn java_min(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.is_sign_negative() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// The most cells of scratch a distance keeps on the stack; a longer string
/// takes its rows from the heap.
pub const STACK_CELLS: usize = 192;

/// `f` over `len` cells of zeroed scratch: on the stack when they fit.
fn with_scratch<T: Copy + Default, R>(len: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    if len <= STACK_CELLS {
        let mut stack = [T::default(); STACK_CELLS];
        f(&mut stack[..len])
    } else {
        f(&mut vec![T::default(); len])
    }
}

/// `f` over both strings' UTF-16 units: their bytes when both are ASCII.
fn over_utf16<R>(
    a: &str,
    b: &str,
    ascii: impl FnOnce(&[u8], &[u8]) -> R,
    wide: impl FnOnce(&[u16], &[u16]) -> R,
) -> R {
    if a.is_ascii() && b.is_ascii() {
        ascii(a.as_bytes(), b.as_bytes())
    } else {
        wide(&utf16(a), &utf16(b))
    }
}

/// A row cell: an edit count (a `&str` is under 4 GiB, and so are its units).
fn cell(n: usize) -> u32 {
    u32::try_from(n).expect("a string under 2^32 units")
}

/// `LevenshteinDistance`: the edit distance over UTF-16 units, as
/// `1 - distance / max(length)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LevenshteinDistance;

impl LevenshteinDistance {
    /// `new LevenshteinDistance()`.
    pub fn new() -> Self {
        Self
    }

    fn units<T: Copy + Eq>(sa: &[T], other: &[T]) -> f32 {
        let n = sa.len();
        let m = other.len();
        if n == 0 || m == 0 {
            return if n == m { 1.0 } else { 0.0 };
        }
        with_scratch(2 * (n + 1), |rows: &mut [u32]| {
            // two rows, swapped after each character of `other`
            let (mut p, mut d) = rows.split_at_mut(n + 1);
            for (i, c) in p.iter_mut().enumerate() {
                *c = cell(i);
            }
            for (j, &t_j) in other.iter().enumerate() {
                d[0] = cell(j + 1);
                for i in 1..=n {
                    let cost = u32::from(sa[i - 1] != t_j);
                    d[i] = (d[i - 1] + 1).min(p[i] + 1).min(p[i - 1] + cost);
                }
                std::mem::swap(&mut p, &mut d);
            }
            1.0 - (p[n] as f32 / m.max(n) as f32)
        })
    }
}

impl StringDistance for LevenshteinDistance {
    fn get_distance(&self, target: &str, other: &str) -> f32 {
        over_utf16(target, other, Self::units, Self::units)
    }
}

impl fmt::Display for LevenshteinDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("levenshtein")
    }
}

/// `JaroWinklerDistance`: the Jaro similarity over UTF-16 units, raised by
/// the Winkler common-prefix bonus once it reaches the threshold.
#[derive(Debug, Clone, Copy)]
pub struct JaroWinklerDistance {
    threshold: f32,
}

impl Default for JaroWinklerDistance {
    fn default() -> Self {
        Self { threshold: 0.7 }
    }
}

impl JaroWinklerDistance {
    /// `new JaroWinklerDistance()`: threshold 0.7.
    pub fn new() -> Self {
        Self::default()
    }

    /// `setThreshold`: the similarity from which the prefix bonus applies.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// `getThreshold`.
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }

    /// `matches(s1, s2)`: `[matches, transpositions / 2, prefix, max length]`.
    fn matches<T: Copy + Eq>(s1: &[T], s2: &[T]) -> [usize; 4] {
        let (max, min) = if s1.len() > s2.len() {
            (s1, s2)
        } else {
            (s2, s1)
        };
        let range = (max.len() / 2).saturating_sub(1);
        // `matchIndexes` is only ever asked whether a unit matched: a flag
        // per unit of each string
        with_scratch(min.len() + max.len(), |flags: &mut [bool]| {
            let (min_flags, max_flags) = flags.split_at_mut(min.len());
            let mut matches = 0;
            for (mi, &c1) in min.iter().enumerate() {
                let xn = (mi + range + 1).min(max.len());
                for xi in mi.saturating_sub(range)..xn {
                    if !max_flags[xi] && c1 == max[xi] {
                        min_flags[mi] = true;
                        max_flags[xi] = true;
                        matches += 1;
                        break;
                    }
                }
            }
            let ms1 = min
                .iter()
                .zip(min_flags.iter())
                .filter(|(_, f)| **f)
                .map(|(c, _)| *c);
            let ms2 = max
                .iter()
                .zip(max_flags.iter())
                .filter(|(_, f)| **f)
                .map(|(c, _)| *c);
            let transpositions = ms1.zip(ms2).filter(|(a, b)| a != b).count();
            let prefix = (0..min.len()).take_while(|&mi| s1[mi] == s2[mi]).count();
            [matches, transpositions / 2, prefix, max.len()]
        })
    }

    fn units<T: Copy + Eq>(&self, a: &[T], b: &[T]) -> f32 {
        let mtp = Self::matches(a, b);
        let m = mtp[0] as f32;
        if m == 0.0 {
            return 0.0;
        }
        let j = (m / a.len() as f32 + m / b.len() as f32 + (m - mtp[1] as f32) / m) / 3.0;
        if j < self.threshold {
            j
        } else {
            j + java_min(0.1, 1.0 / mtp[3] as f32) * mtp[2] as f32 * (1.0 - j)
        }
    }
}

impl StringDistance for JaroWinklerDistance {
    fn get_distance(&self, s1: &str, s2: &str) -> f32 {
        over_utf16(s1, s2, |a, b| self.units(a, b), |a, b| self.units(a, b))
    }
}

/// `equals`: `Float.floatToIntBits` of the thresholds, so every NaN is equal.
impl PartialEq for JaroWinklerDistance {
    fn eq(&self, other: &Self) -> bool {
        let bits = |f: f32| {
            if f.is_nan() {
                f32::NAN.to_bits()
            } else {
                f.to_bits()
            }
        };
        bits(self.threshold) == bits(other.threshold)
    }
}

impl fmt::Display for JaroWinklerDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "jarowinkler({})", java_float(self.threshold))
    }
}

/// `NGramDistance`: Kondrak's n-gram edit distance over UTF-16 units, the
/// source padded in front with `n - 1` NUL units whose matches are
/// discounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NGramDistance {
    n: usize,
}

impl Default for NGramDistance {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl NGramDistance {
    /// `new NGramDistance(size)`. Java's `int` may be negative and then
    /// fails on first use; `usize` rules that out. Size 0 answers NaN for
    /// any two non-empty strings, as Java's `0f / 0` does.
    pub fn with_size(size: usize) -> Self {
        Self { n: size }
    }

    /// `new NGramDistance()`: bigrams.
    pub fn new() -> Self {
        Self::default()
    }

    /// The distance over units; `T::default()` is the NUL padding unit.
    fn units<T: Copy + Eq + Default>(&self, source: &[T], target: &[T]) -> f32 {
        let n = self.n;
        let (sl, tl) = (source.len(), target.len());
        if sl == 0 || tl == 0 {
            return if sl == tl { 1.0 } else { 0.0 };
        }
        if sl < n || tl < n {
            let cost = source.iter().zip(target).filter(|(a, b)| a == b).count();
            return cost as f32 / sl.max(tl) as f32;
        }
        let pad = T::default();
        // sl >= 1, so the padded source has at least sl - 1 units
        with_scratch(sl + n - 1 + n, |units: &mut [T]| {
            let (sa, t_j) = units.split_at_mut(sl + n - 1);
            // size 0 reads no unit of the source (every n-gram is empty)
            if n > 0 {
                sa[n - 1..].copy_from_slice(source);
            }
            with_scratch(2 * (sl + 1), |rows: &mut [f32]| {
                let (mut p, mut d) = rows.split_at_mut(sl + 1);
                for (i, c) in p.iter_mut().enumerate() {
                    *c = i as f32;
                }
                for j in 1..=tl {
                    if j < n {
                        t_j[..n - j].fill(pad);
                        t_j[n - j..].copy_from_slice(&target[..j]);
                    } else {
                        t_j.copy_from_slice(&target[j - n..j]);
                    }
                    d[0] = j as f32;
                    for i in 1..=sl {
                        let mut cost = 0u32;
                        let mut tn = n as i64;
                        for (&s, &t) in sa[i - 1..i - 1 + n].iter().zip(t_j.iter()) {
                            if s != t {
                                cost += 1;
                            } else if s == pad {
                                // discount matches on the padding
                                tn -= 1;
                            }
                        }
                        let ec = cost as f32 / tn as f32;
                        d[i] = java_min(java_min(d[i - 1] + 1.0, p[i] + 1.0), p[i - 1] + ec);
                    }
                    std::mem::swap(&mut p, &mut d);
                }
                1.0 - (p[sl] / tl.max(sl) as f32)
            })
        })
    }
}

impl StringDistance for NGramDistance {
    fn get_distance(&self, source: &str, target: &str) -> f32 {
        over_utf16(
            source,
            target,
            |a, b| self.units(a, b),
            |a, b| self.units(a, b),
        )
    }
}

impl fmt::Display for NGramDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ngram({})", self.n)
    }
}

/// `LuceneLevenshteinDistance`: Damerau-Levenshtein (optimal string
/// alignment) over code points, as `FuzzyTermsEnum` counts edits with
/// transpositions, normalised by the *shorter* length. Equal empty strings
/// answer 0 and an empty against a non-empty string its length, as Java's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LuceneLevenshteinDistance;

impl LuceneLevenshteinDistance {
    /// `new LuceneLevenshteinDistance()`.
    pub fn new() -> Self {
        Self
    }

    /// The distance over code points. Java keeps the whole matrix; the
    /// recurrence reads only the two previous columns (Java's source says
    /// as much), so three rolling columns compute the same cells.
    fn units<T: Copy + Eq>(target: &[T], other: &[T]) -> f32 {
        let n = target.len();
        let m = other.len();
        if n == 0 || m == 0 {
            return if n == m { 0.0 } else { n.max(m) as f32 };
        }
        with_scratch(3 * (n + 1), |cols: &mut [u32]| {
            let (prev2, rest) = cols.split_at_mut(n + 1);
            let (prev, cur) = rest.split_at_mut(n + 1);
            let (mut prev2, mut prev, mut cur) = (prev2, prev, cur);
            // column 0: `d[i][0] = i`
            for (i, c) in prev.iter_mut().enumerate() {
                *c = cell(i);
            }
            for j in 1..=m {
                let t_j = other[j - 1];
                cur[0] = cell(j);
                for i in 1..=n {
                    let cost = u32::from(target[i - 1] != t_j);
                    let mut v = (prev[i] + 1).min(cur[i - 1] + 1).min(prev[i - 1] + cost);
                    if i > 1
                        && j > 1
                        && target[i - 1] == other[j - 2]
                        && target[i - 2] == other[j - 1]
                    {
                        v = v.min(prev2[i - 2] + cost);
                    }
                    cur[i] = v;
                }
                // rotate: column j becomes the previous one
                std::mem::swap(&mut prev2, &mut prev);
                std::mem::swap(&mut prev, &mut cur);
            }
            1.0 - (prev[n] as f32 / m.min(n) as f32)
        })
    }
}

impl StringDistance for LuceneLevenshteinDistance {
    fn get_distance(&self, target: &str, other: &str) -> f32 {
        if target.is_ascii() && other.is_ascii() {
            return Self::units(target.as_bytes(), other.as_bytes());
        }
        let target: Vec<char> = target.chars().collect();
        let other: Vec<char> = other.chars().collect();
        Self::units(&target, &other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_min_lets_nan_and_negative_zero_win() {
        assert!(java_min(f32::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f32::NAN).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert_eq!(java_min(2.0, 1.0), 1.0);
    }

    #[test]
    fn the_distances_print_as_java_does() {
        assert_eq!(LevenshteinDistance::new().to_string(), "levenshtein");
        assert_eq!(NGramDistance::new().to_string(), "ngram(2)");
        assert_eq!(NGramDistance::with_size(3).to_string(), "ngram(3)");
        let mut jw = JaroWinklerDistance::new();
        assert_eq!(jw.to_string(), "jarowinkler(0.7)");
        jw.set_threshold(0.5);
        assert_eq!(jw.get_threshold(), 0.5);
        assert_eq!(jw.to_string(), "jarowinkler(0.5)");
    }

    #[test]
    fn jaro_winkler_equality_is_by_threshold_bits() {
        let mut a = JaroWinklerDistance::new();
        let mut b = JaroWinklerDistance::new();
        assert_eq!(a, b);
        b.set_threshold(0.8);
        assert_ne!(a, b);
        a.set_threshold(f32::NAN);
        b.set_threshold(-f32::NAN);
        assert_eq!(a, b);
        a.set_threshold(0.0);
        b.set_threshold(-0.0);
        assert_ne!(a, b);
        assert_eq!(LuceneLevenshteinDistance::new(), LuceneLevenshteinDistance);
    }
}

/// The distances as first ported -- every string decoded, the whole matrix
/// kept -- the reference the stack-row, byte-unit versions must equal bit
/// for bit.
#[cfg(test)]
#[allow(dead_code)]
mod reference {
    use std::fmt;

    use super::{java_min, StringDistance};
    use crate::function::java_float;

    fn utf16(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    /// `LevenshteinDistance`: the edit distance over UTF-16 units, as
    /// `1 - distance / max(length)`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct LevenshteinDistance;

    impl LevenshteinDistance {
        /// `new LevenshteinDistance()`.
        pub fn new() -> Self {
            Self
        }
    }

    impl StringDistance for LevenshteinDistance {
        fn get_distance(&self, target: &str, other: &str) -> f32 {
            let sa = utf16(target);
            let other = utf16(other);
            let n = sa.len();
            let m = other.len();
            if n == 0 || m == 0 {
                return if n == m { 1.0 } else { 0.0 };
            }
            // two rows, swapped after each character of `other`
            let mut p: Vec<usize> = (0..=n).collect();
            let mut d = vec![0usize; n + 1];
            for j in 1..=m {
                let t_j = other[j - 1];
                d[0] = j;
                for i in 1..=n {
                    let cost = usize::from(sa[i - 1] != t_j);
                    d[i] = (d[i - 1] + 1).min(p[i] + 1).min(p[i - 1] + cost);
                }
                std::mem::swap(&mut p, &mut d);
            }
            1.0 - (p[n] as f32 / m.max(n) as f32)
        }
    }

    impl fmt::Display for LevenshteinDistance {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("levenshtein")
        }
    }

    /// `JaroWinklerDistance`: the Jaro similarity over UTF-16 units, raised by
    /// the Winkler common-prefix bonus once it reaches the threshold.
    #[derive(Debug, Clone, Copy)]
    pub struct JaroWinklerDistance {
        threshold: f32,
    }

    impl Default for JaroWinklerDistance {
        fn default() -> Self {
            Self { threshold: 0.7 }
        }
    }

    impl JaroWinklerDistance {
        /// `new JaroWinklerDistance()`: threshold 0.7.
        pub fn new() -> Self {
            Self::default()
        }

        /// `setThreshold`: the similarity from which the prefix bonus applies.
        pub fn set_threshold(&mut self, threshold: f32) {
            self.threshold = threshold;
        }

        /// `getThreshold`.
        pub fn get_threshold(&self) -> f32 {
            self.threshold
        }

        /// `matches(s1, s2)`: `[matches, transpositions / 2, prefix, max length]`.
        fn matches(s1: &[u16], s2: &[u16]) -> [usize; 4] {
            let (max, min) = if s1.len() > s2.len() {
                (s1, s2)
            } else {
                (s2, s1)
            };
            let range = (max.len() / 2).saturating_sub(1);
            let mut match_indexes: Vec<Option<usize>> = vec![None; min.len()];
            let mut match_flags = vec![false; max.len()];
            let mut matches = 0;
            for (mi, &c1) in min.iter().enumerate() {
                let xn = (mi + range + 1).min(max.len());
                for xi in mi.saturating_sub(range)..xn {
                    if !match_flags[xi] && c1 == max[xi] {
                        match_indexes[mi] = Some(xi);
                        match_flags[xi] = true;
                        matches += 1;
                        break;
                    }
                }
            }
            let ms1 = min
                .iter()
                .zip(&match_indexes)
                .filter(|(_, m)| m.is_some())
                .map(|(c, _)| *c);
            let ms2 = max
                .iter()
                .zip(&match_flags)
                .filter(|(_, f)| **f)
                .map(|(c, _)| *c);
            let transpositions = ms1.zip(ms2).filter(|(a, b)| a != b).count();
            let prefix = (0..min.len()).take_while(|&mi| s1[mi] == s2[mi]).count();
            [matches, transpositions / 2, prefix, max.len()]
        }
    }

    impl StringDistance for JaroWinklerDistance {
        fn get_distance(&self, s1: &str, s2: &str) -> f32 {
            let (a, b) = (utf16(s1), utf16(s2));
            let mtp = Self::matches(&a, &b);
            let m = mtp[0] as f32;
            if m == 0.0 {
                return 0.0;
            }
            let j = (m / a.len() as f32 + m / b.len() as f32 + (m - mtp[1] as f32) / m) / 3.0;
            if j < self.threshold {
                j
            } else {
                j + java_min(0.1, 1.0 / mtp[3] as f32) * mtp[2] as f32 * (1.0 - j)
            }
        }
    }

    /// `equals`: `Float.floatToIntBits` of the thresholds, so every NaN is equal.
    impl PartialEq for JaroWinklerDistance {
        fn eq(&self, other: &Self) -> bool {
            let bits = |f: f32| {
                if f.is_nan() {
                    f32::NAN.to_bits()
                } else {
                    f.to_bits()
                }
            };
            bits(self.threshold) == bits(other.threshold)
        }
    }

    impl fmt::Display for JaroWinklerDistance {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "jarowinkler({})", java_float(self.threshold))
        }
    }

    /// `NGramDistance`: Kondrak's n-gram edit distance over UTF-16 units, the
    /// source padded in front with `n - 1` NUL units whose matches are
    /// discounted.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NGramDistance {
        n: usize,
    }

    impl Default for NGramDistance {
        fn default() -> Self {
            Self { n: 2 }
        }
    }

    impl NGramDistance {
        /// `new NGramDistance(size)`. Java's `int` may be negative and then
        /// fails on first use; `usize` rules that out. Size 0 answers NaN for
        /// any two non-empty strings, as Java's `0f / 0` does.
        pub fn with_size(size: usize) -> Self {
            Self { n: size }
        }

        /// `new NGramDistance()`: bigrams.
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl StringDistance for NGramDistance {
        fn get_distance(&self, source: &str, target: &str) -> f32 {
            let n = self.n;
            let (source, target) = (utf16(source), utf16(target));
            let (sl, tl) = (source.len(), target.len());
            if sl == 0 || tl == 0 {
                return if sl == tl { 1.0 } else { 0.0 };
            }
            if sl < n || tl < n {
                let cost = source.iter().zip(&target).filter(|(a, b)| a == b).count();
                return cost as f32 / sl.max(tl) as f32;
            }
            // sl >= 1, so the padded source has at least sl - 1 units
            let mut sa = vec![0u16; sl + n - 1];
            for (i, unit) in sa.iter_mut().enumerate() {
                if i + 1 >= n {
                    *unit = source[i + 1 - n];
                }
            }
            let mut p: Vec<f32> = (0..=sl).map(|i| i as f32).collect();
            let mut d = vec![0f32; sl + 1];
            let mut t_j = vec![0u16; n];
            for j in 1..=tl {
                if j < n {
                    t_j[..n - j].fill(0);
                    t_j[n - j..].copy_from_slice(&target[..j]);
                } else {
                    t_j.copy_from_slice(&target[j - n..j]);
                }
                d[0] = j as f32;
                for i in 1..=sl {
                    let mut cost = 0u32;
                    let mut tn = n as i64;
                    for ni in 0..n {
                        if sa[i - 1 + ni] != t_j[ni] {
                            cost += 1;
                        } else if sa[i - 1 + ni] == 0 {
                            // discount matches on the padding
                            tn -= 1;
                        }
                    }
                    let ec = cost as f32 / tn as f32;
                    d[i] = java_min(java_min(d[i - 1] + 1.0, p[i] + 1.0), p[i - 1] + ec);
                }
                std::mem::swap(&mut p, &mut d);
            }
            1.0 - (p[sl] / tl.max(sl) as f32)
        }
    }

    impl fmt::Display for NGramDistance {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "ngram({})", self.n)
        }
    }

    /// `LuceneLevenshteinDistance`: Damerau-Levenshtein (optimal string
    /// alignment) over code points, as `FuzzyTermsEnum` counts edits with
    /// transpositions, normalised by the *shorter* length. Equal empty strings
    /// answer 0 and an empty against a non-empty string its length, as Java's.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct LuceneLevenshteinDistance;

    impl LuceneLevenshteinDistance {
        /// `new LuceneLevenshteinDistance()`.
        pub fn new() -> Self {
            Self
        }
    }

    impl StringDistance for LuceneLevenshteinDistance {
        fn get_distance(&self, target: &str, other: &str) -> f32 {
            let target: Vec<char> = target.chars().collect();
            let other: Vec<char> = other.chars().collect();
            let n = target.len();
            let m = other.len();
            if n == 0 || m == 0 {
                return if n == m { 0.0 } else { n.max(m) as f32 };
            }
            let width = m + 1;
            let mut d = vec![0usize; (n + 1) * width];
            for i in 0..=n {
                d[i * width] = i;
            }
            for (j, cell) in d.iter_mut().enumerate().take(width) {
                *cell = j;
            }
            for j in 1..=m {
                let t_j = other[j - 1];
                for i in 1..=n {
                    let cost = usize::from(target[i - 1] != t_j);
                    let mut v = (d[(i - 1) * width + j] + 1)
                        .min(d[i * width + j - 1] + 1)
                        .min(d[(i - 1) * width + j - 1] + cost);
                    if i > 1
                        && j > 1
                        && target[i - 1] == other[j - 2]
                        && target[i - 2] == other[j - 1]
                    {
                        v = v.min(d[(i - 2) * width + j - 2] + cost);
                    }
                    d[i * width + j] = v;
                }
            }
            1.0 - (d[n * width + m] as f32 / m.min(n) as f32)
        }
    }
}

#[cfg(test)]
mod differential_tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// Short words over a small alphabet (so they share units), now and then
    /// a NUL, a non-ASCII or a supplementary character, or a long run past
    /// the stack scratch.
    fn word(rng: &mut Rng) -> String {
        const UNITS: [char; 9] = ['a', 'b', 'c', 'd', 'e', '\0', '\u{e9}', '\u{1F600}', 'x'];
        let len = match rng.below(10) {
            0 => 0,
            1 => 150 + rng.below(120),
            _ => 1 + rng.below(12),
        };
        let wide = rng.below(3) == 0;
        (0..len)
            .map(|_| UNITS[rng.below(if wide { UNITS.len() } else { 5 })])
            .collect()
    }

    #[test]
    fn every_distance_equals_the_reference_bit_for_bit() {
        let mut jw_low = JaroWinklerDistance::new();
        jw_low.set_threshold(0.2);
        let mut ref_jw_low = reference::JaroWinklerDistance::new();
        ref_jw_low.set_threshold(0.2);
        let pairs: Vec<(Box<dyn StringDistance>, Box<dyn StringDistance>)> = vec![
            (
                Box::new(LevenshteinDistance),
                Box::new(reference::LevenshteinDistance),
            ),
            (
                Box::new(JaroWinklerDistance::new()),
                Box::new(reference::JaroWinklerDistance::new()),
            ),
            (Box::new(jw_low), Box::new(ref_jw_low)),
            (
                Box::new(NGramDistance::new()),
                Box::new(reference::NGramDistance::new()),
            ),
            (
                Box::new(NGramDistance::with_size(0)),
                Box::new(reference::NGramDistance::with_size(0)),
            ),
            (
                Box::new(NGramDistance::with_size(1)),
                Box::new(reference::NGramDistance::with_size(1)),
            ),
            (
                Box::new(NGramDistance::with_size(3)),
                Box::new(reference::NGramDistance::with_size(3)),
            ),
            (
                Box::new(LuceneLevenshteinDistance),
                Box::new(reference::LuceneLevenshteinDistance),
            ),
        ];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..1500 {
            let (a, b) = (word(&mut rng), word(&mut rng));
            for (fast, slow) in &pairs {
                let (got, want) = (fast.get_distance(&a, &b), slow.get_distance(&a, &b));
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{fast:?} {a:?} {b:?}: {got} vs {want}"
                );
                let same = fast.get_distance(&a, &a).to_bits();
                assert_eq!(same, slow.get_distance(&a, &a).to_bits(), "{fast:?} {a:?}");
            }
        }
    }
}
