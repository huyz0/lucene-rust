//! `org.apache.lucene.analysis.util` (analysis-common's helpers), plus
//! lucene-core's `RollingBuffer`, which the graph filters share.

pub mod char_tokenizer;
pub mod csv_util;
mod elision_filter;
pub mod java_regex;
pub mod jflex;
pub mod rolling_buffer;
pub mod stemmer_util;

pub use char_tokenizer::{
    from_separator_char_predicate, CharTokenizer, JavaLetter, LetterTokenizer, NotJavaWhitespace,
    NotUnicodeWhitespace, TokenChar, UnicodeWhitespaceTokenizer, WhitespaceTokenizer,
};
pub use elision_filter::ElisionFilter;
pub use java_regex::JavaPattern;
pub use rolling_buffer::{Resettable, RollingBuffer};

use crate::attributes::AttributeSource;
use crate::AnalysisError;

/// The most bytes one array sized from configuration may take: Java's
/// `new int[n]` past the heap throws `OutOfMemoryError`, which an
/// allocation failure here would turn into an aborted process (and the
/// JVM with it, across the FFI).
pub const MAX_CONFIGURED_ALLOCATION: usize = 1 << 30;

/// Java's `OutOfMemoryError` for an array of `len` elements of `elem_size`
/// bytes sized by `what`, as an `IllegalArgument` naming it.
fn out_of_memory(what: &str, len: usize) -> AnalysisError {
    AnalysisError::IllegalArgument(format!(
        "OutOfMemoryError: Java heap space ({what}: {len} elements)"
    ))
}

/// Checks that `len` elements of `elem_size` bytes, sized by the
/// configuration value `what`, fit [`MAX_CONFIGURED_ALLOCATION`].
pub fn check_configured_allocation(
    len: usize,
    elem_size: usize,
    what: &str,
) -> Result<(), AnalysisError> {
    match len.checked_mul(elem_size.max(1)) {
        Some(bytes) if bytes <= MAX_CONFIGURED_ALLOCATION => Ok(()),
        _ => Err(out_of_memory(what, len)),
    }
}

/// `new T[len]` (filled with `value`) for a length taken from
/// configuration: past [`MAX_CONFIGURED_ALLOCATION`], or when the
/// allocator refuses, Java's `OutOfMemoryError` rather than an abort.
pub fn configured_vec<T: Clone>(len: usize, value: T, what: &str) -> Result<Vec<T>, AnalysisError> {
    check_configured_allocation(len, std::mem::size_of::<T>(), what)?;
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| out_of_memory(what, len))?;
    v.resize(len, value);
    Ok(v)
}

/// Runs `f` over the term's UTF-16 code units (Java's `buffer()` /
/// `length()`), in `buf`; when `f` returns `true` the units are written back
/// (`copyBuffer`/`setLength`), an unpaired surrogate becoming U+FFFD. A
/// write-back of the units the term already holds is skipped.
pub(crate) fn with_utf16_term(
    a: &mut AttributeSource,
    buf: &mut Vec<u16>,
    f: impl FnOnce(&mut Vec<u16>) -> bool,
) {
    buf.clear();
    push_utf16(a.term(), buf);
    if f(buf) && !utf16_eq(a.term(), buf) {
        a.set_term_utf16(buf);
    }
}

/// Appends `s`'s UTF-16 code units to `buf`: `str::encode_utf16`, decoded
/// byte-wise (the iterator's per-unit capacity checks were a stem filter's
/// largest single cost).
pub(crate) fn push_utf16(s: &str, buf: &mut Vec<u16>) {
    let b = s.as_bytes();
    buf.reserve(b.len());
    let cont = |i: usize| u32::from(b[i] & 0x3F);
    let mut i = 0;
    while i < b.len() {
        let x = u32::from(b[i]);
        if x < 0x80 {
            buf.push(x as u16);
            i += 1;
        } else if x < 0xE0 {
            buf.push((((x & 0x1F) << 6) | cont(i + 1)) as u16);
            i += 2;
        } else if x < 0xF0 {
            buf.push((((x & 0x0F) << 12) | (cont(i + 1) << 6) | cont(i + 2)) as u16);
            i += 3;
        } else {
            let cp = ((x & 0x07) << 18) | (cont(i + 1) << 12) | (cont(i + 2) << 6) | cont(i + 3);
            buf.push((0xD7C0 + (cp >> 10)) as u16);
            buf.push((0xDC00 | (cp & 0x3FF)) as u16);
            i += 4;
        }
    }
}

/// `s` holds exactly the UTF-16 units `units`.
pub(crate) fn utf16_eq(s: &str, units: &[u16]) -> bool {
    if s.len() == units.len() && s.is_ascii() {
        return s.bytes().zip(units).all(|(b, &u)| u16::from(b) == u);
    }
    s.len() >= units.len() && s.encode_utf16().eq(units.iter().copied())
}

/// Test support: Lucene test-framework's `CannedTokenStream`, written as the
/// compact `term:start:end:posInc:posLen ...|finalOffset|finalPosInc` spec the
/// unit tests carry next to the output Java's filter gave for it.
#[cfg(test)]
pub(crate) mod canned {
    use crate::attributes::AttributeSource;
    use crate::token_stream::TokenStream;
    use crate::AnalysisError;

    /// `CannedTokenStream`.
    pub(crate) struct Canned {
        atts: AttributeSource,
        tokens: Vec<AttributeSource>,
        upto: usize,
        final_offset: i32,
        final_inc: i32,
    }

    impl Canned {
        /// Parses the spec (see the module docs).
        pub(crate) fn parse(spec: &str) -> Self {
            let mut parts = spec.split('|');
            let toks = parts.next().unwrap().trim();
            let final_offset = parts.next().map_or(0, |s| s.parse().unwrap());
            let final_inc = parts.next().map_or(0, |s| s.parse().unwrap());
            let tokens = toks
                .split(' ')
                .filter(|t| !t.is_empty())
                .map(|t| {
                    let f: Vec<&str> = t.rsplitn(5, ':').collect();
                    let mut a = AttributeSource::new();
                    a.set_term(f[4]);
                    a.set_offset(f[3].parse().unwrap(), f[2].parse().unwrap())
                        .unwrap();
                    a.set_position_increment(f[1].parse().unwrap()).unwrap();
                    a.set_position_length(f[0].parse().unwrap()).unwrap();
                    a
                })
                .collect();
            Canned {
                atts: AttributeSource::new(),
                tokens,
                upto: 0,
                final_offset,
                final_inc,
            }
        }
    }

    impl Canned {
        /// Overrides each token's term (for text the spec cannot hold).
        pub(crate) fn set_terms(&mut self, terms: &[&str]) {
            for (t, s) in self.tokens.iter_mut().zip(terms) {
                t.set_term(s);
            }
        }

        /// Sets each token's flags.
        pub(crate) fn set_flags(&mut self, flags: &[i32]) {
            for (t, f) in self.tokens.iter_mut().zip(flags) {
                t.set_flags(*f);
            }
        }

        /// Sets each token's keyword flag.
        pub(crate) fn set_keywords(&mut self, k: &[bool]) {
            for (t, k) in self.tokens.iter_mut().zip(k) {
                t.set_keyword(*k);
            }
        }

        /// Sets each token's type.
        pub(crate) fn set_types(&mut self, types: &[&'static str]) {
            for (t, ty) in self.tokens.iter_mut().zip(types) {
                t.set_token_type(*ty);
            }
        }
    }

    impl TokenStream for Canned {
        /// A source, not a wrapper: no conditional wrapper below it.
        fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
            None
        }
        fn attributes(&self) -> &AttributeSource {
            &self.atts
        }
        fn attributes_mut(&mut self) -> &mut AttributeSource {
            &mut self.atts
        }
        fn increment_token(&mut self) -> Result<bool, AnalysisError> {
            match self.tokens.get(self.upto) {
                Some(t) => {
                    self.atts.restore_state(t);
                    self.upto += 1;
                    Ok(true)
                }
                None => Ok(false),
            }
        }
        fn reset(&mut self) -> Result<(), AnalysisError> {
            self.upto = 0;
            Ok(())
        }
        fn end(&mut self) -> Result<(), AnalysisError> {
            self.atts.end_attributes();
            self.atts.set_position_increment(self.final_inc)?;
            self.atts.set_offset(self.final_offset, self.final_offset)
        }
    }

    /// Runs `ts` and renders its output in the spec format.
    pub(crate) fn render(ts: &mut dyn TokenStream) -> String {
        let mut out = Vec::new();
        let end = crate::token_stream::consume(ts, |a| {
            out.push(format!(
                "{}:{}:{}:{}:{}",
                a.term(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
                a.position_length()
            ))
        })
        .unwrap();
        format!(
            "{}|{}|{}",
            out.join(" "),
            end.end_offset(),
            end.position_increment()
        )
    }
}
