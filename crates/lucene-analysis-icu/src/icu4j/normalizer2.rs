//! `com.ibm.icu.text.Normalizer2` and its implementations
//! (`Norm2AllModes`' `ComposeNormalizer2`, `DecomposeNormalizer2`,
//! `FCDNormalizer2`, `NoopNormalizer2`, and `FilteredNormalizer2`).
//!
//! [`Normalizer2::get_instance`] is `Normalizer2.getInstance(null, name,
//! mode)` over the data ICU4J 77.1 ships (`nfc`, `nfkc`, `nfkc_cf`,
//! `nfkc_scf`, `uts46`, vendored from its jar); [`Normalizer2::from_data`]
//! is `getInstance(InputStream, name, mode)` for a caller's `.nrm` file,
//! which is how Lucene loads `utr30.nrm` ([`Normalizer2::utr30`]). Each
//! file's data is parsed once and shared.
//!
//! Text is UTF-16 code units, Java's `CharSequence`. Rust-forced change:
//! Java's `IllegalArgumentException` for `normalize(s, s)` (source and
//! destination the same object) cannot arise -- the borrow checker forbids
//! the aliasing.

use std::sync::{Arc, OnceLock};

use crate::icu4j::normalizer2_impl::{Normalizer2Impl, ReorderingBuffer};
use crate::icu4j::unicode_set::{SpanCondition, UnicodeSet};
use crate::{IcuError, IcuErrorKind};

/// `Normalizer2.Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// `COMPOSE`: NFC-like.
    Compose,
    /// `DECOMPOSE`: NFD-like.
    Decompose,
    /// `FCD`.
    Fcd,
    /// `COMPOSE_CONTIGUOUS`: FCC.
    ComposeContiguous,
}

/// `Normalizer.QuickCheckResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuickCheck {
    /// `NO`.
    No,
    /// `YES`.
    Yes,
    /// `MAYBE`.
    Maybe,
}

#[derive(Debug, Clone)]
enum Kind {
    WithImpl {
        imp: Arc<Normalizer2Impl>,
        mode: Mode,
        /// Per ASCII unit, what a string of ASCII alone normalizes it to,
        /// when that is one ASCII unit for every unit and no two of them
        /// compose ([`Normalizer2::ascii_map`]).
        ascii: Option<Arc<[u8; 128]>>,
    },
    Noop,
    Filtered {
        norm2: Box<Normalizer2>,
        set: Arc<UnicodeSet>,
    },
}

/// A normalizer: one mode of one `.nrm` data file, or a filtered one.
#[derive(Debug, Clone)]
pub struct Normalizer2 {
    kind: Kind,
}

pub(crate) const NFC_DATA: &[u8] = include_bytes!("../resources/nfc.nrm");
const NFKC_DATA: &[u8] = include_bytes!("../resources/nfkc.nrm");
const NFKC_CF_DATA: &[u8] = include_bytes!("../resources/nfkc_cf.nrm");
const NFKC_SCF_DATA: &[u8] = include_bytes!("../resources/nfkc_scf.nrm");
const UTS46_DATA: &[u8] = include_bytes!("../resources/uts46.nrm");
/// Lucene's `org/apache/lucene/analysis/icu/utr30.nrm` (Apache-2.0, built
/// by Lucene from UTR #30's foldings with ICU's `gennorm2`).
const UTR30_DATA: &[u8] = include_bytes!("../resources/utr30.nrm");

static NFC: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();

/// `Norm2AllModes.getNFCInstance().impl`: the NFC data, shared (collation
/// reads its FCD values and decompositions).
pub fn nfc_impl() -> Arc<Normalizer2Impl> {
    shared(&NFC, NFC_DATA)
}

fn shared(cell: &'static OnceLock<Arc<Normalizer2Impl>>, data: &[u8]) -> Arc<Normalizer2Impl> {
    cell.get_or_init(|| Arc::new(Normalizer2Impl::load(data).expect("vendored .nrm data loads")))
        .clone()
}

impl Normalizer2 {
    fn with_impl(imp: Arc<Normalizer2Impl>, mode: Mode) -> Normalizer2 {
        let mut n = Normalizer2 {
            kind: Kind::WithImpl {
                imp,
                mode,
                ascii: None,
            },
        };
        // The table costs 128 normalizations and 128 x 128 composition
        // probes: computed once per data and mode, not per normalizer (a
        // char filter builds one per document).
        let slot = match mode {
            Mode::Compose => 0,
            Mode::Decompose => 1,
            Mode::Fcd => 2,
            Mode::ComposeContiguous => 3,
        };
        let table = match &n.kind {
            Kind::WithImpl { imp, .. } => imp.ascii_maps[slot]
                .get_or_init(|| n.compute_ascii_map().map(Arc::new))
                .clone(),
            _ => None,
        };
        if let Kind::WithImpl { ascii, .. } = &mut n.kind {
            *ascii = table;
        }
        n
    }

    /// The ASCII table: each ASCII unit normalizes alone to one ASCII unit
    /// of combining class 0, and no two such units compose -- then a string
    /// of ASCII alone normalizes unit by unit (every unit is a starter that
    /// combines with nothing after it), and its normalization is the table
    /// applied to each unit.
    fn compute_ascii_map(&self) -> Option<[u8; 128]> {
        let mut t = [0u8; 128];
        for (c, slot) in t.iter_mut().enumerate() {
            let out = self.normalize(&[c as u16]);
            match out[..] {
                [m] if m < 0x80 && self.get_combining_class(i32::from(m)) == 0 => *slot = m as u8,
                _ => return None,
            }
        }
        for &a in &t {
            for &b in &t {
                if self.compose_pair(i32::from(a), i32::from(b)) >= 0 {
                    return None;
                }
            }
        }
        Some(t)
    }

    /// The table a string of ASCII alone normalizes through, unit by unit
    /// (see `compute_ascii_map`); `None` for a filtered normalizer or data
    /// for which that does not hold.
    pub fn ascii_map(&self) -> Option<&[u8; 128]> {
        match &self.kind {
            Kind::WithImpl { ascii, .. } => ascii.as_deref(),
            _ => None,
        }
    }

    /// Whether the UTF-16 [`Self::quick_check`] of UTF-8 `s` is known to be
    /// [`QuickCheck::Yes`] without converting it (the per-unit fast loop of
    /// the compose and decompose checks; `false` means "not known").
    pub fn quick_yes_utf8(&self, s: &str) -> bool {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Compose | Mode::ComposeContiguous => imp.quick_yes_utf8(s, false),
                Mode::Decompose => imp.quick_yes_utf8(s, true),
                Mode::Fcd => false,
            },
            Kind::Noop => true,
            // The fast loop judges each character alone, so passing it over
            // the whole string passes it over every filtered span.
            Kind::Filtered { norm2, .. } => norm2.quick_yes_utf8(s),
        }
    }

    /// `Normalizer2.getInstance(null, name, mode)`: one of the normalizers
    /// ICU ships (`nfc`, `nfkc`, `nfkc_cf`, `nfkc_scf`, `uts46`); any other
    /// name is Java's `MissingResourceException`.
    pub fn get_instance(name: &str, mode: Mode) -> Result<Normalizer2, IcuError> {
        static NFKC: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();
        static NFKC_CF: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();
        static NFKC_SCF: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();
        static UTS46: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();
        let imp = match name {
            "nfc" => shared(&NFC, NFC_DATA),
            "nfkc" => shared(&NFKC, NFKC_DATA),
            "nfkc_cf" => shared(&NFKC_CF, NFKC_CF_DATA),
            "nfkc_scf" => shared(&NFKC_SCF, NFKC_SCF_DATA),
            "uts46" => shared(&UTS46, UTS46_DATA),
            _ => {
                return Err(IcuError::with_kind(
                    IcuErrorKind::MissingResource,
                    format!("could not locate data {name}.nrm"),
                ))
            }
        };
        Ok(Self::with_impl(imp, mode))
    }

    /// `Normalizer2.getInstance(InputStream, name, mode)` over a `.nrm`
    /// file's bytes.
    pub fn from_data(data: &[u8], mode: Mode) -> Result<Normalizer2, IcuError> {
        Ok(Self::with_impl(
            Arc::new(Normalizer2Impl::load(data)?),
            mode,
        ))
    }

    /// `ICUFoldingFilter.NORMALIZER`: Lucene's `utr30.nrm`, `COMPOSE`.
    pub fn utr30() -> Normalizer2 {
        static UTR30: OnceLock<Arc<Normalizer2Impl>> = OnceLock::new();
        Self::with_impl(shared(&UTR30, UTR30_DATA), Mode::Compose)
    }

    /// `getNFCInstance()`.
    pub fn nfc() -> Normalizer2 {
        Self::get_instance("nfc", Mode::Compose).expect("nfc is vendored")
    }

    /// `getNFKCInstance()`.
    pub fn nfkc() -> Normalizer2 {
        Self::get_instance("nfkc", Mode::Compose).expect("nfkc is vendored")
    }

    /// `getNFKCCasefoldInstance()`.
    pub fn nfkc_casefold() -> Normalizer2 {
        Self::get_instance("nfkc_cf", Mode::Compose).expect("nfkc_cf is vendored")
    }

    /// `Norm2AllModes.NOOP_NORMALIZER2`.
    pub fn noop() -> Normalizer2 {
        Normalizer2 { kind: Kind::Noop }
    }

    /// `new FilteredNormalizer2(n2, filterSet)`. A set holding strings is
    /// refused (`UnsupportedOperation`): `UnicodeSet.span` over strings
    /// (`UnicodeSetStringSpan`) is not ported.
    pub fn filtered(norm2: Normalizer2, set: UnicodeSet) -> Result<Normalizer2, IcuError> {
        if set.has_strings() {
            return Err(IcuError::unsupported(
                "FilteredNormalizer2 over a UnicodeSet with strings is not ported",
            ));
        }
        Ok(Normalizer2 {
            kind: Kind::Filtered {
                norm2: Box::new(norm2),
                set: Arc::new(set),
            },
        })
    }

    /// `normalize(src, dest)`: `dest` is cleared, then receives the
    /// normalized `src`.
    pub fn normalize_to(&self, src: &[u16], dest: &mut Vec<u16>) {
        dest.clear();
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => {
                let mut buffer = ReorderingBuffer::new(imp, dest, src.len());
                normalize_impl(imp, *mode, src, &mut buffer);
            }
            Kind::Noop => dest.extend_from_slice(src),
            Kind::Filtered { norm2, set } => {
                filtered_normalize(norm2, set, src, dest, SpanCondition::Simple)
            }
        }
    }

    /// `normalize(src)`.
    pub fn normalize(&self, src: &[u16]) -> Vec<u16> {
        let mut dest = Vec::new();
        self.normalize_to(src, &mut dest);
        dest
    }

    /// `normalizeSecondAndAppend(first, second)`.
    pub fn normalize_second_and_append(&self, first: &mut Vec<u16>, second: &[u16]) {
        self.second_and_append(first, second, true);
    }

    /// `append(first, second)`.
    pub fn append(&self, first: &mut Vec<u16>, second: &[u16]) {
        self.second_and_append(first, second, false);
    }

    fn second_and_append(&self, first: &mut Vec<u16>, second: &[u16], do_normalize: bool) {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => {
                let cap = first.len().saturating_add(second.len());
                let mut buffer = ReorderingBuffer::new(imp, first, cap);
                match mode {
                    Mode::Decompose => imp.decompose_and_append(second, do_normalize, &mut buffer),
                    Mode::Compose => {
                        imp.compose_and_append(second, do_normalize, false, &mut buffer)
                    }
                    Mode::ComposeContiguous => {
                        imp.compose_and_append(second, do_normalize, true, &mut buffer)
                    }
                    Mode::Fcd => imp.make_fcd_and_append(second, do_normalize, &mut buffer),
                }
            }
            Kind::Noop => first.extend_from_slice(second),
            Kind::Filtered { norm2, set } => {
                filtered_second_and_append(norm2, set, first, second, do_normalize)
            }
        }
    }

    /// `getDecomposition(c)`.
    pub fn get_decomposition(&self, c: i32) -> Option<Vec<u16>> {
        match &self.kind {
            Kind::WithImpl { imp, .. } => imp.get_decomposition(c),
            Kind::Noop => None,
            Kind::Filtered { norm2, set } => {
                if set.contains(c) {
                    norm2.get_decomposition(c)
                } else {
                    None
                }
            }
        }
    }

    /// `getRawDecomposition(c)`.
    pub fn get_raw_decomposition(&self, c: i32) -> Option<Vec<u16>> {
        match &self.kind {
            Kind::WithImpl { imp, .. } => imp.get_raw_decomposition(c),
            Kind::Noop => None,
            Kind::Filtered { norm2, set } => {
                if set.contains(c) {
                    norm2.get_raw_decomposition(c)
                } else {
                    None
                }
            }
        }
    }

    /// `composePair(a, b)`.
    // SENTINEL: `-1` = no composite (Java's own result).
    pub fn compose_pair(&self, a: i32, b: i32) -> i32 {
        match &self.kind {
            Kind::WithImpl { imp, .. } => imp.compose_pair(a, b),
            Kind::Noop => -1,
            Kind::Filtered { norm2, set } => {
                if set.contains(a) && set.contains(b) {
                    norm2.compose_pair(a, b)
                } else {
                    -1
                }
            }
        }
    }

    /// `getCombiningClass(c)`.
    pub fn get_combining_class(&self, c: i32) -> i32 {
        match &self.kind {
            Kind::WithImpl { imp, .. } => imp.get_cc(imp.get_norm16(c)),
            Kind::Noop => 0,
            Kind::Filtered { norm2, set } => {
                if set.contains(c) {
                    norm2.get_combining_class(c)
                } else {
                    0
                }
            }
        }
    }

    /// `isNormalized(s)`.
    pub fn is_normalized(&self, s: &[u16]) -> bool {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Compose | Mode::ComposeContiguous => {
                    let mut scratch = Vec::new();
                    let mut buffer = ReorderingBuffer::new(imp, &mut scratch, 5);
                    imp.compose(
                        s,
                        0,
                        s.len(),
                        *mode == Mode::ComposeContiguous,
                        false,
                        &mut buffer,
                    )
                }
                _ => s.len() == self.span_quick_check_yes(s),
            },
            Kind::Noop => true,
            Kind::Filtered { norm2, set } => {
                let mut cond = SpanCondition::Simple;
                let mut prev = 0;
                while prev < s.len() {
                    let limit = set.span(s, prev, cond);
                    if cond == SpanCondition::NotContained {
                        cond = SpanCondition::Simple;
                    } else {
                        if !norm2.is_normalized(&s[prev..limit]) {
                            return false;
                        }
                        cond = SpanCondition::NotContained;
                    }
                    prev = limit;
                }
                true
            }
        }
    }

    /// `quickCheck(s)`.
    pub fn quick_check(&self, s: &[u16]) -> QuickCheck {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Compose | Mode::ComposeContiguous => {
                    let r = imp.compose_quick_check(
                        s,
                        0,
                        s.len(),
                        *mode == Mode::ComposeContiguous,
                        false,
                    );
                    if r & 1 != 0 {
                        QuickCheck::Maybe
                    } else if r >> 1 == s.len() {
                        QuickCheck::Yes
                    } else {
                        QuickCheck::No
                    }
                }
                _ => {
                    if self.is_normalized(s) {
                        QuickCheck::Yes
                    } else {
                        QuickCheck::No
                    }
                }
            },
            Kind::Noop => QuickCheck::Yes,
            Kind::Filtered { norm2, set } => {
                let mut result = QuickCheck::Yes;
                let mut cond = SpanCondition::Simple;
                let mut prev = 0;
                while prev < s.len() {
                    let limit = set.span(s, prev, cond);
                    if cond == SpanCondition::NotContained {
                        cond = SpanCondition::Simple;
                    } else {
                        match norm2.quick_check(&s[prev..limit]) {
                            QuickCheck::No => return QuickCheck::No,
                            QuickCheck::Maybe => result = QuickCheck::Maybe,
                            QuickCheck::Yes => {}
                        }
                        cond = SpanCondition::NotContained;
                    }
                    prev = limit;
                }
                result
            }
        }
    }

    /// `spanQuickCheckYes(s)`.
    pub fn span_quick_check_yes(&self, s: &[u16]) -> usize {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Decompose => imp.decompose(s, 0, s.len(), None),
                Mode::Compose => imp.compose_quick_check(s, 0, s.len(), false, true) >> 1,
                Mode::ComposeContiguous => imp.compose_quick_check(s, 0, s.len(), true, true) >> 1,
                Mode::Fcd => imp.make_fcd(s, 0, s.len(), None),
            },
            Kind::Noop => s.len(),
            Kind::Filtered { norm2, set } => {
                let mut cond = SpanCondition::Simple;
                let mut prev = 0;
                while prev < s.len() {
                    let limit = set.span(s, prev, cond);
                    if cond == SpanCondition::NotContained {
                        cond = SpanCondition::Simple;
                    } else {
                        let yes = prev.saturating_add(norm2.span_quick_check_yes(&s[prev..limit]));
                        if yes < limit {
                            return yes;
                        }
                        cond = SpanCondition::NotContained;
                    }
                    prev = limit;
                }
                s.len()
            }
        }
    }

    /// `hasBoundaryBefore(c)`.
    pub fn has_boundary_before(&self, c: i32) -> bool {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Decompose | Mode::Fcd => imp.has_decomp_boundary_before(c),
                _ => imp.has_comp_boundary_before(c),
            },
            Kind::Noop => true,
            Kind::Filtered { norm2, set } => !set.contains(c) || norm2.has_boundary_before(c),
        }
    }

    /// `hasBoundaryAfter(c)`.
    pub fn has_boundary_after(&self, c: i32) -> bool {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Decompose | Mode::Fcd => imp.has_decomp_boundary_after(c),
                Mode::Compose => imp.has_comp_boundary_after(c, false),
                Mode::ComposeContiguous => imp.has_comp_boundary_after(c, true),
            },
            Kind::Noop => true,
            Kind::Filtered { norm2, set } => !set.contains(c) || norm2.has_boundary_after(c),
        }
    }

    /// `isInert(c)`.
    pub fn is_inert(&self, c: i32) -> bool {
        match &self.kind {
            Kind::WithImpl { imp, mode, .. } => match mode {
                Mode::Decompose => imp.is_decomp_inert(c),
                Mode::Fcd => imp.is_fcd_inert(c),
                Mode::Compose => imp.is_comp_inert(c, false),
                Mode::ComposeContiguous => imp.is_comp_inert(c, true),
            },
            Kind::Noop => true,
            Kind::Filtered { norm2, set } => !set.contains(c) || norm2.is_inert(c),
        }
    }
}

/// `Normalizer2WithImpl.normalize(src, buffer)` per mode.
fn normalize_impl(
    imp: &Normalizer2Impl,
    mode: Mode,
    src: &[u16],
    buffer: &mut ReorderingBuffer<'_>,
) {
    match mode {
        Mode::Decompose => {
            imp.decompose(src, 0, src.len(), Some(buffer));
        }
        Mode::Compose => {
            imp.compose(src, 0, src.len(), false, true, buffer);
        }
        Mode::ComposeContiguous => {
            imp.compose(src, 0, src.len(), true, true, buffer);
        }
        Mode::Fcd => {
            imp.make_fcd(src, 0, src.len(), Some(buffer));
        }
    }
}

/// `FilteredNormalizer2.normalize(src, dest, spanCondition)`: appends.
fn filtered_normalize(
    norm2: &Normalizer2,
    set: &UnicodeSet,
    src: &[u16],
    dest: &mut Vec<u16>,
    mut cond: SpanCondition,
) {
    let mut temp = Vec::new();
    let mut prev = 0;
    while prev < src.len() {
        let limit = set.span(src, prev, cond);
        if cond == SpanCondition::NotContained {
            dest.extend_from_slice(&src[prev..limit]);
            cond = SpanCondition::Simple;
        } else {
            if limit != prev {
                norm2.normalize_to(&src[prev..limit], &mut temp);
                dest.extend_from_slice(&temp);
            }
            cond = SpanCondition::NotContained;
        }
        prev = limit;
    }
}

/// `FilteredNormalizer2.normalizeSecondAndAppend(first, second, doNormalize)`.
fn filtered_second_and_append(
    norm2: &Normalizer2,
    set: &UnicodeSet,
    first: &mut Vec<u16>,
    second: &[u16],
    do_normalize: bool,
) {
    if first.is_empty() {
        if do_normalize {
            filtered_normalize(norm2, set, second, first, SpanCondition::Simple);
        } else {
            first.extend_from_slice(second);
        }
        return;
    }
    let prefix_limit = set.span(second, 0, SpanCondition::Simple);
    if prefix_limit != 0 {
        let prefix = &second[..prefix_limit];
        let suffix_start = set.span_back(first, first.len(), SpanCondition::Simple);
        if suffix_start == 0 {
            norm2.second_and_append(first, prefix, do_normalize);
        } else {
            let mut middle = first[suffix_start..].to_vec();
            norm2.second_and_append(&mut middle, prefix, do_normalize);
            first.truncate(suffix_start);
            first.extend_from_slice(&middle);
        }
    }
    if prefix_limit < second.len() {
        let rest = &second[prefix_limit..];
        if do_normalize {
            filtered_normalize(norm2, set, rest, first, SpanCondition::NotContained);
        } else {
            first.extend_from_slice(rest);
        }
    }
}

#[cfg(test)]
mod tests;
