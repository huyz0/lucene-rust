//! `TestUtil`'s random generators and comparison helpers.
//!
//! Each generator draws from a [`JavaRandom`] in exactly the order its Java
//! twin does, so the same seed gives the same value
//! (`tests/test_framework_random_fixtures.rs` compares 200 seeds against
//! Lucene's own output). Strings are built in UTF-16, as Java builds them --
//! lengths, truncation and surrogate handling are counted in code units --
//! and returned as `String` once they are known to be well formed (every
//! generator here produces well-formed UTF-16).
//!
//! `TestUtil`'s index helpers (`checkIndex`, `docs`, `cloneDocument`, the
//! codec pickers) arrive with `RandomIndexWriter` (M14.h batch h4);
//! `randomPattern` needs `java.util.regex` and is not ported.

use std::cmp::Ordering;

use lucene_util::automaton::{java_to_lower_case, java_to_upper_case};
use lucene_util::java_random::JavaRandom;

use super::random_numbers::{random_from, random_int_between};

/// `TestUtil.nextInt(r, start, end)`: both bounds inclusive.
pub fn next_int(r: &mut JavaRandom, start: i32, end: i32) -> i32 {
    random_int_between(r, start, end)
}

/// `TestUtil.nextLong(r, start, end)`: both bounds inclusive.
///
/// A range wider than `Integer.MAX_VALUE` is Java's
/// `new BigDecimal(range).multiply(new BigDecimal(r.nextDouble()))
/// .toBigInteger()`; `nextDouble()` is `k / 2^53` exactly, so the product
/// truncated is `(range * k) >> 53`, which `u128` holds exactly.
///
/// # Panics
///
/// When `end < start` (Java's assertion).
pub fn next_long(r: &mut JavaRandom, start: i64, end: i64) -> i64 {
    assert!(end >= start, "start={start},end={end}");
    let range = (i128::from(end) + 1 - i128::from(start)) as u128;
    if range <= i32::MAX as u128 {
        start + i64::from(r.next_int_bounded(range as i32))
    } else {
        let k = (r.next_double() * (1u64 << 53) as f64) as u128;
        let augend = (range * k) >> 53;
        (i128::from(start) + augend as i128) as i64
    }
}

/// `TestUtil.nextBigInteger(random, maxBytes)`: the `1..=maxBytes` random
/// bytes `new BigInteger(buffer)` reads -- big-endian two's complement.
pub fn next_big_integer(r: &mut JavaRandom, max_bytes: i32) -> Vec<u8> {
    let length = next_int(r, 1, max_bytes);
    let mut buffer = vec![0u8; length as usize];
    r.next_bytes(&mut buffer);
    buffer
}

/// `TestUtil.randomSimpleString(r, minLength, maxLength)`: `a`-`z`.
pub fn random_simple_string_between(
    r: &mut JavaRandom,
    min_length: i32,
    max_length: i32,
) -> String {
    let end = next_int(r, min_length, max_length);
    (0..end)
        .map(|_| char::from(next_int(r, i32::from(b'a'), i32::from(b'z')) as u8))
        .collect()
}

/// `TestUtil.randomSimpleString(r, maxLength)`.
pub fn random_simple_string_max(r: &mut JavaRandom, max_length: i32) -> String {
    random_simple_string_between(r, 0, max_length)
}

/// `TestUtil.randomSimpleString(r)`: up to 10 characters.
pub fn random_simple_string(r: &mut JavaRandom) -> String {
    random_simple_string_between(r, 0, 10)
}

/// `TestUtil.randomSimpleStringRange(r, minChar, maxChar, maxLength)`, as
/// UTF-16 code units: a range may include lone surrogates, as in Java.
pub fn random_simple_string_range(
    r: &mut JavaRandom,
    min_char: u16,
    max_char: u16,
    max_length: i32,
) -> Vec<u16> {
    let end = next_int(r, 0, max_length);
    (0..end)
        .map(|_| next_int(r, i32::from(min_char), i32::from(max_char)) as u16)
        .collect()
}

/// `TestUtil.randomUnicodeString(r)`: up to 20 code units, the full range.
pub fn random_unicode_string(r: &mut JavaRandom) -> String {
    random_unicode_string_max(r, 20)
}

/// `TestUtil.randomUnicodeString(r, maxLength)`.
pub fn random_unicode_string_max(r: &mut JavaRandom, max_length: i32) -> String {
    utf16(&random_unicode_units(r, max_length))
}

fn random_unicode_units(r: &mut JavaRandom, max_length: i32) -> Vec<u16> {
    let end = next_int(r, 0, max_length) as usize;
    let mut buffer = vec![0u16; end];
    random_fixed_length_unicode_string(r, &mut buffer, 0, end);
    buffer
}

/// `TestUtil.randomFixedLengthUnicodeString(random, chars, offset, length)`:
/// fills `chars[offset..offset + length]` with well-formed UTF-16.
///
/// Java's surrogate-pair guard compares against `length - 1`, not
/// `offset + length - 1`; kept, so a non-zero `offset` draws as Java draws:
/// it only means no surrogate pair starts in the last `offset` slots.
pub fn random_fixed_length_unicode_string(
    r: &mut JavaRandom,
    chars: &mut [u16],
    offset: usize,
    length: usize,
) {
    let mut i = offset;
    let end = offset + length;
    while i < end {
        let t = r.next_int_bounded(5);
        if t == 0 && i + 1 < length {
            chars[i] = next_int(r, 0xd800, 0xdbff) as u16;
            chars[i + 1] = next_int(r, 0xdc00, 0xdfff) as u16;
            i += 2;
            continue;
        }
        chars[i] = match t {
            0 | 1 => r.next_int_bounded(0x80) as u16,
            2 => next_int(r, 0x80, 0x7ff) as u16,
            3 => next_int(r, 0x800, 0xd7ff) as u16,
            _ => next_int(r, 0xe000, 0xffff) as u16,
        };
        i += 1;
    }
}

/// `maxRecursionBound`.
const MAX_RECURSION_BOUND: i32 = 5;

/// `ops`: the operators `randomRegexpishString` picks from.
const OPS: [&str; 10] = [".", "?", "{0,5}", "{1,5}", "(", ")", "-", "[", "]", "|"];

/// `TestUtil.randomRegexpishString(r)`: up to 20 pieces.
pub fn random_regexpish_string(r: &mut JavaRandom) -> String {
    random_regexpish_string_max(r, 20)
}

/// `TestUtil.randomRegexpishString(r, maxLength)`: letters and regular
/// expression operators, `*` and `+` replaced by bounded repeats
/// (LUCENE-4111).
pub fn random_regexpish_string_max(r: &mut JavaRandom, max_length: i32) -> String {
    debug_assert_eq!(OPS[2], format!("{{0,{MAX_RECURSION_BOUND}}}"));
    let mut s = String::new();
    let mut i = next_int(r, 0, max_length);
    while i > 0 {
        if r.next_boolean() {
            s.push(char::from(
                random_int_between(r, i32::from(b'a'), i32::from(b'z')) as u8,
            ));
        } else {
            s.push_str(random_from(r, &OPS));
        }
        i -= 1;
    }
    s
}

/// `"    ".substring(n)` and friends: `width - n` copies of `c`.
fn tail(c: char, width: i32, n: i32) -> String {
    std::iter::repeat_n(c, (width - n) as usize).collect()
}

/// `TestUtil.randomHtmlishString(random, numElements)`: tags, comments,
/// entities and text, deliberately often malformed.
pub fn random_htmlish_string(r: &mut JavaRandom, num_elements: i32) -> String {
    let end = next_int(r, 0, num_elements);
    let mut sb = String::new();
    for _ in 0..end {
        match r.next_int_bounded(25) {
            0 => sb.push_str("<p>"),
            1 => {
                sb.push('<');
                sb.push_str(&tail(' ', 4, next_int(r, 0, 4)));
                sb.push_str(&random_simple_string(r));
                // Java re-draws the bound on every iteration.
                let mut j = 0;
                while j < next_int(r, 0, 10) {
                    sb.push(' ');
                    sb.push_str(&random_simple_string(r));
                    sb.push_str(&tail(' ', 1, next_int(r, 0, 1)));
                    sb.push('=');
                    sb.push_str(&tail(' ', 1, next_int(r, 0, 1)));
                    sb.push_str(&tail('"', 1, next_int(r, 0, 1)));
                    sb.push_str(&random_simple_string(r));
                    sb.push_str(&tail('"', 1, next_int(r, 0, 1)));
                    j += 1;
                }
                sb.push_str(&tail(' ', 4, next_int(r, 0, 4)));
                sb.push_str(&tail('/', 1, next_int(r, 0, 1)));
                sb.push_str(&tail('>', 1, next_int(r, 0, 1)));
            }
            2 => {
                sb.push_str("</");
                sb.push_str(&tail(' ', 4, next_int(r, 0, 4)));
                sb.push_str(&random_simple_string(r));
                sb.push_str(&tail(' ', 4, next_int(r, 0, 4)));
                sb.push_str(&tail('>', 1, next_int(r, 0, 1)));
            }
            3 => sb.push('>'),
            4 => sb.push_str("</p>"),
            5 => sb.push_str("<!--"),
            6 => sb.push_str("<!--#"),
            7 => sb.push_str("<script><!-- f('"),
            8 => sb.push_str("</script>"),
            9 => sb.push_str("<?"),
            10 => sb.push_str("?>"),
            11 => sb.push('"'),
            12 => sb.push_str("\\\""),
            13 => sb.push('\''),
            14 => sb.push_str("\\'"),
            15 => sb.push_str("-->"),
            16 => {
                sb.push('&');
                match next_int(r, 0, 2) {
                    0 => sb.push_str(&random_simple_string(r)),
                    1 => sb.push_str(
                        HTML_CHAR_ENTITIES
                            [r.next_int_bounded(HTML_CHAR_ENTITIES.len() as i32) as usize],
                    ),
                    _ => {}
                }
                sb.push_str(&tail(';', 1, next_int(r, 0, 1)));
            }
            17 => {
                sb.push_str("&#");
                if next_int(r, 0, 1) == 0 {
                    sb.push_str(&next_int(r, 0, i32::MAX - 1).to_string());
                    sb.push_str(&tail(';', 1, next_int(r, 0, 1)));
                }
            }
            18 => {
                sb.push_str("&#x");
                if next_int(r, 0, 1) == 0 {
                    sb.push_str(&format!("{:x}", next_int(r, 0, i32::MAX - 1)));
                    sb.push_str(&tail(';', 1, next_int(r, 0, 1)));
                }
            }
            19 => sb.push(';'),
            20 => sb.push_str(&next_int(r, 0, i32::MAX - 1).to_string()),
            21 => sb.push('\n'),
            22 => sb.push_str(&tail(' ', 10, next_int(r, 0, 10))),
            23 => {
                sb.push('<');
                if next_int(r, 0, 3) == 0 {
                    sb.push_str(&tail(' ', 10, next_int(r, 1, 10)));
                }
                if next_int(r, 0, 1) == 0 {
                    sb.push('/');
                    if next_int(r, 0, 3) == 0 {
                        sb.push_str(&tail(' ', 10, next_int(r, 1, 10)));
                    }
                }
                match next_int(r, 0, 3) {
                    0 => sb.push_str(&randomly_recase_code_points(r, "script")),
                    1 => sb.push_str(&randomly_recase_code_points(r, "style")),
                    2 => sb.push_str(&randomly_recase_code_points(r, "br")),
                    _ => {}
                }
                sb.push_str(&tail('>', 1, next_int(r, 0, 1)));
            }
            _ => sb.push_str(&random_simple_string(r)),
        }
    }
    sb
}

/// `TestUtil.randomlyRecaseCodePoints(random, str)`: each code point
/// upper-cased, lower-cased or kept, by `Character.toUpperCase(int)` /
/// `toLowerCase(int)` (the simple, length-preserving mappings).
///
/// The case table is the JDK's ([`java_to_upper_case`]), generated from the
/// Unicode version of the JDK it was read from; a code point whose case
/// changed between Unicode versions maps as that table says.
pub fn randomly_recase_code_points(r: &mut JavaRandom, s: &str) -> String {
    s.chars()
        .map(|c| {
            let cp = c as i32;
            let mapped = match next_int(r, 0, 2) {
                0 => java_to_upper_case(cp),
                1 => java_to_lower_case(cp),
                _ => cp,
            };
            char::from_u32(mapped as u32).expect("a case mapping is a scalar value")
        })
        .collect()
}

/// `TestUtil.randomRealisticUnicodeString(r)`: up to 20 code points.
pub fn random_realistic_unicode_string(r: &mut JavaRandom) -> String {
    random_realistic_unicode_string_between(r, 0, 20)
}

/// `TestUtil.randomRealisticUnicodeString(r, maxLength)`.
pub fn random_realistic_unicode_string_max(r: &mut JavaRandom, max_length: i32) -> String {
    random_realistic_unicode_string_between(r, 0, max_length)
}

/// `TestUtil.randomRealisticUnicodeString(r, minLength, maxLength)`:
/// `minLength..=maxLength` code points, all from one Unicode block.
pub fn random_realistic_unicode_string_between(
    r: &mut JavaRandom,
    min_length: i32,
    max_length: i32,
) -> String {
    let end = next_int(r, min_length, max_length);
    let block = r.next_int_bounded(BLOCK_STARTS.len() as i32) as usize;
    (0..end)
        .map(|_| {
            let cp = next_int(r, BLOCK_STARTS[block], BLOCK_ENDS[block]);
            char::from_u32(cp as u32).expect("no block holds a surrogate")
        })
        .collect()
}

/// `TestUtil.randomFixedByteLengthUnicodeString(r, length)`: a string whose
/// UTF-8 encoding is exactly `length` bytes.
pub fn random_fixed_byte_length_unicode_string(r: &mut JavaRandom, length: i32) -> String {
    let mut buffer = vec![0u16; length as usize * 3];
    let mut bytes = length;
    let mut i = 0;
    while i < buffer.len() && bytes != 0 {
        let t = if bytes >= 4 {
            r.next_int_bounded(5)
        } else if bytes == 3 {
            r.next_int_bounded(4)
        } else if bytes == 2 {
            r.next_int_bounded(2)
        } else {
            0
        };
        match t {
            0 => {
                buffer[i] = r.next_int_bounded(0x80) as u16;
                bytes -= 1;
            }
            1 => {
                buffer[i] = next_int(r, 0x80, 0x7ff) as u16;
                bytes -= 2;
            }
            2 => {
                buffer[i] = next_int(r, 0x800, 0xd7ff) as u16;
                bytes -= 3;
            }
            3 => {
                buffer[i] = next_int(r, 0xe000, 0xffff) as u16;
                bytes -= 3;
            }
            _ => {
                buffer[i] = next_int(r, 0xd800, 0xdbff) as u16;
                i += 1;
                buffer[i] = next_int(r, 0xdc00, 0xdfff) as u16;
                bytes -= 4;
            }
        }
        i += 1;
    }
    utf16(&buffer[..i])
}

/// `TestUtil.randomBinaryTerm(r)`: `0..15` random bytes.
pub fn random_binary_term(r: &mut JavaRandom) -> Vec<u8> {
    let length = r.next_int_bounded(15);
    random_binary_term_len(r, length)
}

/// `TestUtil.randomBinaryTerm(r, length)`.
pub fn random_binary_term_len(r: &mut JavaRandom, length: i32) -> Vec<u8> {
    let mut b = vec![0u8; length as usize];
    r.next_bytes(&mut b);
    b
}

/// `TestUtil.randomAnalysisString(random, maxLength, simple)`: text shaped
/// like words, as `MockTokenizer` would see it.
///
/// # Panics
///
/// When `max_length` is not positive (`Random.nextInt(0)` throws in Java).
pub fn random_analysis_string(r: &mut JavaRandom, max_length: i32, simple: bool) -> String {
    assert!(max_length >= 0);
    if r.next_int_bounded(31) == 0 {
        let word_length = r.next_int_bounded(max_length);
        return utf16(&random_sub_units(r, word_length, simple));
    }
    let max_length = r.next_int_bounded(max_length) as usize;
    let avg_word_length = next_int(r, 3, 8);
    let mut sb: Vec<u16> = Vec::new();
    while sb.len() < max_length {
        if !sb.is_empty() {
            sb.push(u16::from(b' '));
        }
        let mut word_length = -1;
        while word_length < 0 {
            word_length = (r.next_gaussian() * 3.0 + f64::from(avg_word_length)) as i32;
        }
        let word_length = word_length.min((max_length - sb.len()) as i32);
        sb.extend(random_sub_units(r, word_length, simple));
    }
    utf16(&sb)
}

/// `TestUtil.randomSubString(random, wordLength, simple)`: one "word" of at
/// most `wordLength` code units, from a generator picked by an "evilness".
pub fn random_sub_string(r: &mut JavaRandom, word_length: i32, simple: bool) -> String {
    utf16(&random_sub_units(r, word_length, simple))
}

fn random_sub_units(r: &mut JavaRandom, word_length: i32, simple: bool) -> Vec<u16> {
    if word_length == 0 {
        return Vec::new();
    }
    let evilness = next_int(r, 0, 20);
    let wl = word_length as usize;
    let mut sb: Vec<u16> = Vec::new();
    while sb.len() < wl {
        let piece = if simple {
            if r.next_boolean() {
                random_simple_string_max(r, word_length)
            } else {
                random_htmlish_string(r, word_length)
            }
        } else if evilness < 10 {
            random_simple_string_max(r, word_length)
        } else if evilness < 15 {
            debug_assert!(sb.is_empty());
            random_realistic_unicode_string_between(r, word_length, word_length)
        } else if evilness == 16 {
            random_htmlish_string(r, word_length)
        } else if evilness == 17 {
            random_regexpish_string_max(r, word_length)
        } else {
            random_unicode_string_max(r, word_length)
        };
        sb.extend(piece.encode_utf16());
    }
    if sb.len() > wl {
        sb.truncate(wl);
        if is_high_surrogate(sb[wl - 1]) {
            sb.truncate(wl - 1);
        }
    }
    if r.next_int_bounded(17) == 0 {
        let mixed = randomly_recase_code_points(r, &utf16(&sb));
        debug_assert_eq!(mixed.encode_utf16().count(), sb.len());
        mixed.encode_utf16().collect()
    } else {
        sb
    }
}

fn is_high_surrogate(c: u16) -> bool {
    (0xd800..=0xdbff).contains(&c)
}

/// Well-formed UTF-16 to a `String`; every generator here produces it.
fn utf16(units: &[u16]) -> String {
    String::from_utf16(units).expect("the generators produce well-formed UTF-16")
}

/// `TestUtil.STRING_CODEPOINT_COMPARATOR`: UTF-16 strings compared by code
/// point (which is UTF-8 byte order, unlike UTF-16 code-unit order).
pub fn string_codepoint_compare(a: &[u16], b: &[u16]) -> Ordering {
    let a = char::decode_utf16(a.iter().copied()).map(|c| c.map_or(0xFFFD, |c| c as u32));
    let b = char::decode_utf16(b.iter().copied()).map(|c| c.map_or(0xFFFD, |c| c as u32));
    a.cmp(b)
}

/// `TestUtil.floatUlpEquals(x, y, maxUlps)`: within `maxUlps` floats of each
/// other, inclusive; false for NaN.
pub fn float_ulp_equals(x: f32, y: f32, max_ulps: i16) -> bool {
    let x_int = x.to_bits() as i32;
    let y_int = y.to_bits() as i32;
    let max = i32::from(max_ulps);
    if (x_int ^ y_int) < 0 {
        let delta_plus = x_int & i32::MAX;
        let delta_minus = y_int & i32::MAX;
        return delta_plus <= max && delta_minus <= max - delta_plus;
    }
    x_int.wrapping_sub(y_int).wrapping_abs() <= max && !x.is_nan() && !y.is_nan()
}

/// `TestUtil.doubleUlpEquals(x, y, maxUlps)`.
pub fn double_ulp_equals(x: f64, y: f64, max_ulps: i32) -> bool {
    let x_int = x.to_bits() as i64;
    let y_int = y.to_bits() as i64;
    let max = i64::from(max_ulps);
    if (x_int ^ y_int) < 0 {
        let delta_plus = x_int & i64::MAX;
        let delta_minus = y_int & i64::MAX;
        return delta_plus <= max && delta_minus <= max - delta_plus;
    }
    x_int.wrapping_sub(y_int).wrapping_abs() <= max && !x.is_nan() && !y.is_nan()
}

/// `HTML_CHAR_ENTITIES`.
const HTML_CHAR_ENTITIES: [&str; 259] = [
    "AElig", "Aacute", "Acirc", "Agrave", "Alpha", "AMP", "Aring", "Atilde", "Auml", "Beta",
    "COPY", "Ccedil", "Chi", "Dagger", "Delta", "ETH", "Eacute", "Ecirc", "Egrave", "Epsilon",
    "Eta", "Euml", "Gamma", "GT", "Iacute", "Icirc", "Igrave", "Iota", "Iuml", "Kappa", "Lambda",
    "LT", "Mu", "Ntilde", "Nu", "OElig", "Oacute", "Ocirc", "Ograve", "Omega", "Omicron", "Oslash",
    "Otilde", "Ouml", "Phi", "Pi", "Prime", "Psi", "QUOT", "REG", "Rho", "Scaron", "Sigma",
    "THORN", "Tau", "Theta", "Uacute", "Ucirc", "Ugrave", "Upsilon", "Uuml", "Xi", "Yacute",
    "Yuml", "Zeta", "aacute", "acirc", "acute", "aelig", "agrave", "alefsym", "alpha", "amp",
    "and", "ang", "apos", "aring", "asymp", "atilde", "auml", "bdquo", "beta", "brvbar", "bull",
    "cap", "ccedil", "cedil", "cent", "chi", "circ", "clubs", "cong", "copy", "crarr", "cup",
    "curren", "dArr", "dagger", "darr", "deg", "delta", "diams", "divide", "eacute", "ecirc",
    "egrave", "empty", "emsp", "ensp", "epsilon", "equiv", "eta", "eth", "euml", "euro", "exist",
    "fnof", "forall", "frac12", "frac14", "frac34", "frasl", "gamma", "ge", "gt", "hArr", "harr",
    "hearts", "hellip", "iacute", "icirc", "iexcl", "igrave", "image", "infin", "int", "iota",
    "iquest", "isin", "iuml", "kappa", "lArr", "lambda", "lang", "laquo", "larr", "lceil", "ldquo",
    "le", "lfloor", "lowast", "loz", "lrm", "lsaquo", "lsquo", "lt", "macr", "mdash", "micro",
    "middot", "minus", "mu", "nabla", "nbsp", "ndash", "ne", "ni", "not", "notin", "nsub",
    "ntilde", "nu", "oacute", "ocirc", "oelig", "ograve", "oline", "omega", "omicron", "oplus",
    "or", "ordf", "ordm", "oslash", "otilde", "otimes", "ouml", "para", "part", "permil", "perp",
    "phi", "pi", "piv", "plusmn", "pound", "prime", "prod", "prop", "psi", "quot", "rArr", "radic",
    "rang", "raquo", "rarr", "rceil", "rdquo", "real", "reg", "rfloor", "rho", "rlm", "rsaquo",
    "rsquo", "sbquo", "scaron", "sdot", "sect", "shy", "sigma", "sigmaf", "sim", "spades", "sub",
    "sube", "sum", "sup", "sup1", "sup2", "sup3", "supe", "szlig", "tau", "there4", "theta",
    "thetasym", "thinsp", "thorn", "tilde", "times", "trade", "uArr", "uacute", "uarr", "ucirc",
    "ugrave", "uml", "upsih", "upsilon", "uuml", "weierp", "xi", "yacute", "yen", "yuml", "zeta",
    "zwj", "zwnj",
];

/// `blockStarts`: the first code point of each Unicode block `randomRealisticUnicodeString` draws from.
const BLOCK_STARTS: [i32; 194] = [
    0x0000, 0x0080, 0x0100, 0x0180, 0x0250, 0x02B0, 0x0300, 0x0370, 0x0400, 0x0500, 0x0530, 0x0590,
    0x0600, 0x0700, 0x0750, 0x0780, 0x07C0, 0x0800, 0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80,
    0x0C00, 0x0C80, 0x0D00, 0x0D80, 0x0E00, 0x0E80, 0x0F00, 0x1000, 0x10A0, 0x1100, 0x1200, 0x1380,
    0x13A0, 0x1400, 0x1680, 0x16A0, 0x1700, 0x1720, 0x1740, 0x1760, 0x1780, 0x1800, 0x18B0, 0x1900,
    0x1950, 0x1980, 0x19E0, 0x1A00, 0x1A20, 0x1B00, 0x1B80, 0x1C00, 0x1C50, 0x1CD0, 0x1D00, 0x1D80,
    0x1DC0, 0x1E00, 0x1F00, 0x2000, 0x2070, 0x20A0, 0x20D0, 0x2100, 0x2150, 0x2190, 0x2200, 0x2300,
    0x2400, 0x2440, 0x2460, 0x2500, 0x2580, 0x25A0, 0x2600, 0x2700, 0x27C0, 0x27F0, 0x2800, 0x2900,
    0x2980, 0x2A00, 0x2B00, 0x2C00, 0x2C60, 0x2C80, 0x2D00, 0x2D30, 0x2D80, 0x2DE0, 0x2E00, 0x2E80,
    0x2F00, 0x2FF0, 0x3000, 0x3040, 0x30A0, 0x3100, 0x3130, 0x3190, 0x31A0, 0x31C0, 0x31F0, 0x3200,
    0x3300, 0x3400, 0x4DC0, 0x4E00, 0xA000, 0xA490, 0xA4D0, 0xA500, 0xA640, 0xA6A0, 0xA700, 0xA720,
    0xA800, 0xA830, 0xA840, 0xA880, 0xA8E0, 0xA900, 0xA930, 0xA960, 0xA980, 0xAA00, 0xAA60, 0xAA80,
    0xABC0, 0xAC00, 0xD7B0, 0xE000, 0xF900, 0xFB00, 0xFB50, 0xFE00, 0xFE10, 0xFE20, 0xFE30, 0xFE50,
    0xFE70, 0xFF00, 0xFFF0, 0x10000, 0x10080, 0x10100, 0x10140, 0x10190, 0x101D0, 0x10280, 0x102A0,
    0x10300, 0x10330, 0x10380, 0x103A0, 0x10400, 0x10450, 0x10480, 0x10800, 0x10840, 0x10900,
    0x10920, 0x10A00, 0x10A60, 0x10B00, 0x10B40, 0x10B60, 0x10C00, 0x10E60, 0x11080, 0x12000,
    0x12400, 0x13000, 0x1D000, 0x1D100, 0x1D200, 0x1D300, 0x1D360, 0x1D400, 0x1F000, 0x1F030,
    0x1F100, 0x1F200, 0x20000, 0x2A700, 0x2F800, 0xE0000, 0xE0100, 0xF0000, 0x100000,
];

/// `blockEnds`: the last code point of each block in [`BLOCK_STARTS`].
const BLOCK_ENDS: [i32; 194] = [
    0x007F, 0x00FF, 0x017F, 0x024F, 0x02AF, 0x02FF, 0x036F, 0x03FF, 0x04FF, 0x052F, 0x058F, 0x05FF,
    0x06FF, 0x074F, 0x077F, 0x07BF, 0x07FF, 0x083F, 0x097F, 0x09FF, 0x0A7F, 0x0AFF, 0x0B7F, 0x0BFF,
    0x0C7F, 0x0CFF, 0x0D7F, 0x0DFF, 0x0E7F, 0x0EFF, 0x0FFF, 0x109F, 0x10FF, 0x11FF, 0x137F, 0x139F,
    0x13FF, 0x167F, 0x169F, 0x16FF, 0x171F, 0x173F, 0x175F, 0x177F, 0x17FF, 0x18AF, 0x18FF, 0x194F,
    0x197F, 0x19DF, 0x19FF, 0x1A1F, 0x1AAF, 0x1B7F, 0x1BBF, 0x1C4F, 0x1C7F, 0x1CFF, 0x1D7F, 0x1DBF,
    0x1DFF, 0x1EFF, 0x1FFF, 0x206F, 0x209F, 0x20CF, 0x20FF, 0x214F, 0x218F, 0x21FF, 0x22FF, 0x23FF,
    0x243F, 0x245F, 0x24FF, 0x257F, 0x259F, 0x25FF, 0x26FF, 0x27BF, 0x27EF, 0x27FF, 0x28FF, 0x297F,
    0x29FF, 0x2AFF, 0x2BFF, 0x2C5F, 0x2C7F, 0x2CFF, 0x2D2F, 0x2D7F, 0x2DDF, 0x2DFF, 0x2E7F, 0x2EFF,
    0x2FDF, 0x2FFF, 0x303F, 0x309F, 0x30FF, 0x312F, 0x318F, 0x319F, 0x31BF, 0x31EF, 0x31FF, 0x32FF,
    0x33FF, 0x4DBF, 0x4DFF, 0x9FFF, 0xA48F, 0xA4CF, 0xA4FF, 0xA63F, 0xA69F, 0xA6FF, 0xA71F, 0xA7FF,
    0xA82F, 0xA83F, 0xA87F, 0xA8DF, 0xA8FF, 0xA92F, 0xA95F, 0xA97F, 0xA9DF, 0xAA5F, 0xAA7F, 0xAADF,
    0xABFF, 0xD7AF, 0xD7FF, 0xF8FF, 0xFAFF, 0xFB4F, 0xFDFF, 0xFE0F, 0xFE1F, 0xFE2F, 0xFE4F, 0xFE6F,
    0xFEFF, 0xFFEF, 0xFFFF, 0x1007F, 0x100FF, 0x1013F, 0x1018F, 0x101CF, 0x101FF, 0x1029F, 0x102DF,
    0x1032F, 0x1034F, 0x1039F, 0x103DF, 0x1044F, 0x1047F, 0x104AF, 0x1083F, 0x1085F, 0x1091F,
    0x1093F, 0x10A5F, 0x10A7F, 0x10B3F, 0x10B5F, 0x10B7F, 0x10C4F, 0x10E7F, 0x110CF, 0x123FF,
    0x1247F, 0x1342F, 0x1D0FF, 0x1D1FF, 0x1D24F, 0x1D35F, 0x1D37F, 0x1D7FF, 0x1F02F, 0x1F09F,
    0x1F1FF, 0x1F2FF, 0x2A6DF, 0x2B73F, 0x2FA1F, 0xE007F, 0xE01EF, 0xFFFFF, 0x10FFFF,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_long_small_and_large_ranges() {
        let mut r = JavaRandom::new(1);
        let mut twin = JavaRandom::new(1);
        assert_eq!(
            next_long(&mut r, 5, 9),
            5 + i64::from(twin.next_int_bounded(5))
        );
        for _ in 0..500 {
            let v = next_long(&mut r, i64::MIN, i64::MAX);
            let _ = v;
            let w = next_long(&mut r, -10, 1 << 40);
            assert!((-10..=1 << 40).contains(&w));
        }
        // The large branch is `start + floor(range * nextDouble)`.
        let mut r = JavaRandom::new(9);
        let d = JavaRandom::new(9).next_double();
        let want = ((1u128 << 40) as f64 * d) as i64;
        assert_eq!(next_long(&mut r, 0, (1 << 40) - 1), want);
    }

    #[test]
    #[should_panic(expected = "start=2,end=1")]
    fn next_long_refuses_an_empty_range() {
        next_long(&mut JavaRandom::new(1), 2, 1);
    }

    #[test]
    fn simple_strings_stay_in_range() {
        let mut r = JavaRandom::new(2);
        for _ in 0..200 {
            let s = random_simple_string(&mut r);
            assert!(s.len() <= 10 && s.bytes().all(|b| b.is_ascii_lowercase()));
            let s = random_simple_string_between(&mut r, 3, 5);
            assert!((3..=5).contains(&s.len()));
            let u = random_simple_string_range(&mut r, 0xd800, 0xd801, 4);
            assert!(u.len() <= 4 && u.iter().all(|&c| (0xd800..=0xd801).contains(&c)));
        }
        assert_eq!(random_simple_string_max(&mut r, 0), "");
    }

    #[test]
    fn unicode_strings_are_well_formed() {
        let mut r = JavaRandom::new(3);
        for _ in 0..500 {
            assert!(random_unicode_string(&mut r).encode_utf16().count() <= 20);
            let _ = random_realistic_unicode_string(&mut r);
            let s = random_realistic_unicode_string_max(&mut r, 3);
            assert!(s.chars().count() <= 3);
            for len in [0, 1, 2, 3, 4, 9] {
                assert_eq!(
                    random_fixed_byte_length_unicode_string(&mut r, len).len(),
                    len as usize
                );
            }
        }
        let mut buf = [0u16; 8];
        random_fixed_length_unicode_string(&mut r, &mut buf, 2, 4);
        assert_eq!(&buf[..2], &[0, 0]);
    }

    #[test]
    fn regexpish_and_htmlish_strings_draw_every_piece() {
        let mut r = JavaRandom::new(4);
        let mut seen = String::new();
        for _ in 0..300 {
            seen.push_str(&random_regexpish_string(&mut r));
            seen.push_str(&random_htmlish_string(&mut r, 20));
        }
        for piece in ["{0,5}", "<p>", "<!--#", "&#x", "</script>", "\'", "-->"] {
            assert!(seen.contains(piece), "{piece}");
        }
        assert_eq!(random_htmlish_string(&mut r, 0), "");
    }

    #[test]
    fn recasing_keeps_the_length() {
        let mut r = JavaRandom::new(5);
        for _ in 0..20 {
            let s = randomly_recase_code_points(&mut r, "ScRiPt\u{10400}");
            assert_eq!(s.encode_utf16().count(), 8);
            assert_eq!(s.to_lowercase(), "script\u{10428}");
        }
    }

    #[test]
    fn binary_terms_and_big_integers() {
        let mut r = JavaRandom::new(6);
        assert!(random_binary_term(&mut r).len() < 15);
        assert_eq!(random_binary_term_len(&mut r, 7).len(), 7);
        let b = next_big_integer(&mut r, 3);
        assert!((1..=3).contains(&b.len()));
    }

    #[test]
    fn analysis_strings_honour_their_bounds() {
        let mut r = JavaRandom::new(7);
        for _ in 0..300 {
            for simple in [true, false] {
                let s = random_analysis_string(&mut r, 30, simple);
                assert!(s.encode_utf16().count() <= 60);
            }
        }
        assert_eq!(random_sub_string(&mut r, 0, false), "");
        for _ in 0..300 {
            let s = random_sub_string(&mut r, 5, false);
            assert!(s.encode_utf16().count() <= 5, "{s:?}");
        }
    }

    #[test]
    fn codepoint_order_is_not_code_unit_order() {
        let supp: Vec<u16> = "\u{10000}".encode_utf16().collect();
        let bmp: Vec<u16> = "\u{ffff}".encode_utf16().collect();
        assert_eq!(string_codepoint_compare(&supp, &bmp), Ordering::Greater);
        assert!(supp < bmp);
        assert_eq!(
            string_codepoint_compare(&[0x61], &[0x61, 0x62]),
            Ordering::Less
        );
        assert_eq!(
            string_codepoint_compare(&[0xd800], &[0xd800]),
            Ordering::Equal
        );
    }

    #[test]
    fn ulp_equality() {
        assert!(float_ulp_equals(1.0, 1.0, 0));
        assert!(float_ulp_equals(
            1.0,
            f32::from_bits(1.0f32.to_bits() + 1),
            1
        ));
        assert!(!float_ulp_equals(
            1.0,
            f32::from_bits(1.0f32.to_bits() + 2),
            1
        ));
        assert!(float_ulp_equals(0.0, -0.0, 0));
        assert!(!float_ulp_equals(1.0, -1.0, 100));
        assert!(!float_ulp_equals(f32::NAN, f32::NAN, 100));
        assert!(double_ulp_equals(1.0, 1.0, 0));
        assert!(double_ulp_equals(f64::from_bits(1), -f64::from_bits(1), 2));
        assert!(!double_ulp_equals(f64::from_bits(1), -f64::from_bits(2), 2));
        assert!(!double_ulp_equals(1.0, -1.0, 100));
        assert!(!double_ulp_equals(
            1.0,
            f64::from_bits(1.0f64.to_bits() + 2),
            1
        ));
        assert!(!double_ulp_equals(f64::NAN, f64::NAN, 100));
    }

    #[test]
    fn tables_pair_up() {
        assert!(BLOCK_STARTS
            .iter()
            .zip(BLOCK_ENDS.iter())
            .all(|(s, e)| s < e));
        assert_eq!(HTML_CHAR_ENTITIES[0], "AElig");
    }
}
