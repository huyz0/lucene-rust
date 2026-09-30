//! `Automata`: factories for common automata.

use super::automaton::{Automaton, Builder};
use super::case_folding;
use super::error::{illegal_argument, AutomatonError};
use super::operations;
use super::strings_to_automaton;
use super::{MAX_CODE_POINT, MIN_CODE_POINT};

/// `Automata.MAX_STRING_UNION_TERM_LENGTH`.
pub const MAX_STRING_UNION_TERM_LENGTH: usize = 1000;

/// `makeEmpty()`: accepts nothing (zero states).
pub fn make_empty() -> Automaton {
    let mut a = Automaton::new();
    a.finish_state();
    a
}

/// `makeEmptyString()`: accepts only the empty string.
pub fn make_empty_string() -> Automaton {
    let mut a = Automaton::new();
    a.create_state();
    a.set_accept(0, true);
    a
}

/// `makeAnyString()`: accepts every code-point string.
pub fn make_any_string() -> Automaton {
    let mut a = Automaton::new();
    let s = a.create_state();
    a.set_accept(s, true);
    a.add_transition(s, s, MIN_CODE_POINT, MAX_CODE_POINT);
    a.finish_state();
    a
}

/// `makeAnyBinary()`: accepts every byte string.
pub fn make_any_binary() -> Automaton {
    let mut a = Automaton::new();
    let s = a.create_state();
    a.set_accept(s, true);
    a.add_transition(s, s, 0, 255);
    a.finish_state();
    a
}

/// `makeNonEmptyBinary()`: accepts every non-empty byte string.
pub fn make_non_empty_binary() -> Automaton {
    let mut a = Automaton::new();
    let s1 = a.create_state();
    let s2 = a.create_state();
    a.set_accept(s2, true);
    a.add_transition(s1, s2, 0, 255);
    a.add_transition(s2, s2, 0, 255);
    a.finish_state();
    a
}

/// `makeAnyChar()`: accepts any single code point.
pub fn make_any_char() -> Automaton {
    make_char_range(MIN_CODE_POINT, MAX_CODE_POINT)
}

/// `appendAnyChar(a, state)`: a new state reached from `state` on any code
/// point.
pub fn append_any_char(a: &mut Automaton, state: i32) -> i32 {
    let ns = a.create_state();
    a.add_transition(state, ns, MIN_CODE_POINT, MAX_CODE_POINT);
    ns
}

/// `makeChar(c)`.
pub fn make_char(c: i32) -> Automaton {
    make_char_range(c, c)
}

/// `makeCaseInsensitiveChar(c)`: `c` or any of its [`case_folding`]
/// variants.
pub fn make_case_insensitive_char(c: i32) -> Automaton {
    make_char_set(&to_case_insensitive_char(c))
}

/// `appendChar(a, state, c)`: a new state reached from `state` on `c`.
pub fn append_char(a: &mut Automaton, state: i32, c: i32) -> i32 {
    let ns = a.create_state();
    a.add_transition(state, ns, c, c);
    ns
}

/// `makeCharRange(min, max)`: any single code point in `min..=max`.
pub fn make_char_range(min: i32, max: i32) -> Automaton {
    if min > max {
        return make_empty();
    }
    let mut a = Automaton::new();
    let s1 = a.create_state();
    let s2 = a.create_state();
    a.set_accept(s2, true);
    a.add_transition(s1, s2, min, max);
    a.finish_state();
    a
}

/// `makeCharSet(codepoints)`.
pub fn make_char_set(codepoints: &[i32]) -> Automaton {
    make_char_class(codepoints, codepoints).expect("equal lengths")
}

/// `makeCharClass(starts, ends)`: any single code point in one of the
/// `starts[i]..=ends[i]` ranges.
///
/// # Errors
/// `IllegalArgument("starts must match ends")` on unequal lengths.
pub fn make_char_class(starts: &[i32], ends: &[i32]) -> Result<Automaton, AutomatonError> {
    if starts.len() != ends.len() {
        return illegal_argument("starts must match ends");
    }
    if starts.is_empty() {
        return Ok(make_empty());
    }
    let mut a = Automaton::new();
    let s1 = a.create_state();
    let s2 = a.create_state();
    a.set_accept(s2, true);
    for (&lo, &hi) in starts.iter().zip(ends) {
        a.add_transition(s1, s2, lo, hi);
    }
    a.finish_state();
    Ok(a)
}

fn digit(x: &[u8], n: usize) -> i32 {
    i32::from(x[n])
}

fn any_of_right_length(b: &mut Builder, x: &[u8], n: usize) -> i32 {
    let s = b.create_state();
    if x.len() == n {
        b.set_accept(s, true);
    } else {
        let t = any_of_right_length(b, x, n + 1);
        b.add_transition(s, t, '0' as i32, '9' as i32);
    }
    s
}

fn at_least(b: &mut Builder, x: &[u8], n: usize, initials: &mut Vec<i32>, zeros: bool) -> i32 {
    let s = b.create_state();
    if x.len() == n {
        b.set_accept(s, true);
    } else {
        if zeros {
            initials.push(s);
        }
        let c = digit(x, n);
        let t = at_least(b, x, n + 1, initials, zeros && c == '0' as i32);
        b.add_transition_label(s, t, c);
        if c < '9' as i32 {
            let t = any_of_right_length(b, x, n + 1);
            b.add_transition(s, t, c + 1, '9' as i32);
        }
    }
    s
}

fn at_most(b: &mut Builder, x: &[u8], n: usize) -> i32 {
    let s = b.create_state();
    if x.len() == n {
        b.set_accept(s, true);
    } else {
        let c = digit(x, n);
        let t = at_most(b, x, n + 1);
        b.add_transition_label(s, t, c);
        if c > '0' as i32 {
            let t = any_of_right_length(b, x, n + 1);
            b.add_transition(s, t, '0' as i32, c - 1);
        }
    }
    s
}

fn between(
    b: &mut Builder,
    x: &[u8],
    y: &[u8],
    n: usize,
    initials: &mut Vec<i32>,
    zeros: bool,
) -> i32 {
    let s = b.create_state();
    if x.len() == n {
        b.set_accept(s, true);
    } else {
        if zeros {
            initials.push(s);
        }
        let cx = digit(x, n);
        let cy = digit(y, n);
        if cx == cy {
            let t = between(b, x, y, n + 1, initials, zeros && cx == '0' as i32);
            b.add_transition_label(s, t, cx);
        } else {
            let t = at_least(b, x, n + 1, initials, zeros && cx == '0' as i32);
            b.add_transition_label(s, t, cx);
            let t = at_most(b, y, n + 1);
            b.add_transition_label(s, t, cy);
            if cx + 1 < cy {
                let t = any_of_right_length(b, x, n + 1);
                b.add_transition(s, t, cx + 1, cy - 1);
            }
        }
    }
    s
}

/// `makeBinaryInterval(min, minInclusive, max, maxInclusive)`: every byte
/// string in the interval, `None` meaning open-ended.
///
/// # Errors
/// `IllegalArgument` when an open end is not inclusive.
pub fn make_binary_interval(
    min: Option<&[u8]>,
    min_inclusive: bool,
    max: Option<&[u8]>,
    max_inclusive: bool,
) -> Result<Automaton, AutomatonError> {
    if min.is_none() && !min_inclusive {
        return illegal_argument("minInclusive must be true when min is null (open ended)");
    }
    if max.is_none() && !max_inclusive {
        return illegal_argument("maxInclusive must be true when max is null (open ended)");
    }
    let min_inclusive = min_inclusive || min.is_none();
    let min: &[u8] = min.unwrap_or(&[]);
    let cmp = match max {
        Some(max) => min.cmp(max),
        None => {
            if min.is_empty() {
                return Ok(if min_inclusive {
                    make_any_binary()
                } else {
                    make_non_empty_binary()
                });
            }
            std::cmp::Ordering::Less
        }
    };
    match cmp {
        std::cmp::Ordering::Equal => {
            return Ok(if !min_inclusive || !max_inclusive {
                make_empty()
            } else {
                make_binary(min)
            });
        }
        std::cmp::Ordering::Greater => return Ok(make_empty()),
        std::cmp::Ordering::Less => {}
    }
    if let Some(max) = max {
        if max.starts_with(min) && max[min.len()..].iter().all(|&b| b == 0) {
            let mut max_length = max.len();
            if !max_inclusive {
                max_length -= 1;
            }
            if max_length == min.len() {
                return Ok(if !min_inclusive {
                    make_empty()
                } else {
                    make_binary(min)
                });
            }
            let mut a = Automaton::new();
            let mut last = a.create_state();
            for &byte in min {
                let st = a.create_state();
                a.add_transition_label(last, st, i32::from(byte));
                last = st;
            }
            if min_inclusive {
                a.set_accept(last, true);
            }
            for _ in min.len()..max_length {
                let st = a.create_state();
                a.add_transition_label(last, st, 0);
                a.set_accept(st, true);
                last = st;
            }
            a.finish_state();
            return Ok(a);
        }
    }
    let mut a = Automaton::new();
    let start_state = a.create_state();
    let sink_state = a.create_state();
    a.set_accept(sink_state, true);
    a.add_transition(sink_state, sink_state, 0, 255);
    let mut equal_prefix = true;
    let mut last_state = start_state;
    let mut first_max_state = -1;
    let mut shared_prefix_length = 0usize;
    for i in 0..min.len() {
        let min_label = i32::from(min[i]);
        let max_label = match max {
            Some(max) if equal_prefix && i < max.len() => i32::from(max[i]),
            _ => -1,
        };
        let next_state =
            if min_inclusive && i == min.len() - 1 && (!equal_prefix || min_label != max_label) {
                sink_state
            } else {
                a.create_state()
            };
        if equal_prefix {
            if min_label == max_label {
                a.add_transition_label(last_state, next_state, min_label);
            } else if let Some(max) = max {
                a.add_transition_label(last_state, next_state, min_label);
                if max_label > min_label + 1 {
                    a.add_transition(last_state, sink_state, min_label + 1, max_label - 1);
                }
                if max_inclusive || i < max.len() - 1 {
                    first_max_state = a.create_state();
                    if i < max.len() - 1 {
                        a.set_accept(first_max_state, true);
                    }
                    a.add_transition_label(last_state, first_max_state, max_label);
                }
                equal_prefix = false;
                shared_prefix_length = i;
            } else {
                equal_prefix = false;
                shared_prefix_length = 0;
                a.add_transition(last_state, sink_state, min_label + 1, 0xff);
                a.add_transition_label(last_state, next_state, min_label);
            }
        } else {
            a.add_transition_label(last_state, next_state, min_label);
            if min_label < 255 {
                a.add_transition(last_state, sink_state, min_label + 1, 255);
            }
        }
        last_state = next_state;
    }
    if !equal_prefix && last_state != sink_state && last_state != start_state {
        a.add_transition(last_state, sink_state, 0, 255);
    }
    if min_inclusive {
        a.set_accept(last_state, true);
    }
    if let Some(max) = max {
        if first_max_state == -1 {
            shared_prefix_length = min.len();
        } else {
            last_state = first_max_state;
            shared_prefix_length += 1;
        }
        for i in shared_prefix_length..max.len() {
            let max_label = i32::from(max[i]);
            if max_label > 0 {
                a.add_transition(last_state, sink_state, 0, max_label - 1);
            }
            if max_inclusive || i < max.len() - 1 {
                let ns = a.create_state();
                if i < max.len() - 1 {
                    a.set_accept(ns, true);
                }
                a.add_transition_label(last_state, ns, max_label);
                last_state = ns;
            }
        }
        if max_inclusive {
            a.set_accept(last_state, true);
        }
    }
    a.finish_state();
    Ok(a)
}

/// `makeDecimalInterval(min, max, digits)`: decimal numerals of the values
/// in `min..=max`; with `digits > 0` exactly that many digits (zero-padded),
/// otherwise any number of leading zeros.
///
/// # Errors
/// `IllegalArgument` (Java's message-less one, rendered `""`) when
/// `min > max` or `max` needs more than `digits` digits.
pub fn make_decimal_interval(min: i32, max: i32, digits: i32) -> Result<Automaton, AutomatonError> {
    let mut x = min.to_string();
    let mut y = max.to_string();
    if min > max || (digits > 0 && y.len() > digits as usize) {
        return illegal_argument("");
    }
    let d = if digits > 0 { digits as usize } else { y.len() };
    x = format!("{}{x}", "0".repeat(d.saturating_sub(x.len())));
    y = format!("{}{y}", "0".repeat(d.saturating_sub(y.len())));
    let mut builder = Builder::new();
    if digits <= 0 {
        builder.create_state();
    }
    let mut initials = Vec::new();
    between(
        &mut builder,
        x.as_bytes(),
        y.as_bytes(),
        0,
        &mut initials,
        digits <= 0,
    );
    let mut a1 = builder.finish();
    if digits <= 0 {
        a1.add_transition_label(0, 0, '0' as i32);
        for p in initials {
            a1.add_epsilon(0, p);
        }
        a1.finish_state();
    }
    Ok(operations::remove_dead_states(&a1))
}

/// `makeString(String)`: accepts exactly `s` (as code points).
pub fn make_string(s: &str) -> Automaton {
    let mut a = Automaton::new();
    let mut last = a.create_state();
    for c in s.chars() {
        let st = a.create_state();
        a.add_transition_label(last, st, c as i32);
        last = st;
    }
    a.set_accept(last, true);
    a.finish_state();
    a
}

/// `makeCaseInsensitiveString(s)`: `s` with every code point matched
/// case-insensitively.
pub fn make_case_insensitive_string(s: &str) -> Automaton {
    let mut a = Automaton::new();
    let mut last = a.create_state();
    for c in s.chars() {
        let st = a.create_state();
        for alt in to_case_insensitive_char(c as i32) {
            a.add_transition_label(last, st, alt);
        }
        last = st;
    }
    a.set_accept(last, true);
    a.finish_state();
    a
}

/// `makeBinary(BytesRef)`: accepts exactly `term`'s bytes.
pub fn make_binary(term: &[u8]) -> Automaton {
    let mut a = Automaton::new();
    let mut last = a.create_state();
    for &b in term {
        let st = a.create_state();
        a.add_transition_label(last, st, i32::from(b));
        last = st;
    }
    a.set_accept(last, true);
    a.finish_state();
    a
}

/// `makeString(int[] word, offset, length)`: accepts exactly `word`.
pub fn make_string_ints(word: &[i32]) -> Automaton {
    let mut a = Automaton::new();
    a.create_state();
    let mut s = 0;
    for &label in word {
        let s2 = a.create_state();
        a.add_transition_label(s, s2, label);
        s = s2;
    }
    a.set_accept(s, true);
    a.finish_state();
    a
}

/// `makeStringUnion(Iterable<BytesRef>)`: the minimal automaton accepting
/// exactly these UTF-8 strings (code-point labelled). Input must be sorted
/// in byte order.
///
/// # Errors
/// As [`strings_to_automaton::build`].
pub fn make_string_union<'a, I>(utf8_strings: I) -> Result<Automaton, AutomatonError>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut it = utf8_strings.into_iter().peekable();
    if it.peek().is_none() {
        return Ok(make_empty());
    }
    strings_to_automaton::build(it, false)
}

/// `makeBinaryStringUnion(Iterable<BytesRef>)`: as [`make_string_union`],
/// byte labelled.
///
/// # Errors
/// As [`strings_to_automaton::build`].
pub fn make_binary_string_union<'a, I>(strings: I) -> Result<Automaton, AutomatonError>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut it = strings.into_iter().peekable();
    if it.peek().is_none() {
        return Ok(make_empty());
    }
    strings_to_automaton::build(it, true)
}

/// Java's private `toCaseInsensitiveChar`: the sorted [`case_folding`]
/// expansion of `codepoint` (duplicates kept, as Java's list keeps them).
pub(crate) fn to_case_insensitive_char(codepoint: i32) -> Vec<i32> {
    let mut list = Vec::new();
    case_folding::expand(codepoint, &mut |v| list.push(v));
    list.sort_unstable();
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::operations::{determinize, run};

    #[test]
    fn factories() {
        assert_eq!(make_empty().get_num_states(), 0);
        assert!(make_empty_string().is_accept(0));
        assert!(run(&make_any_string(), "anything"));
        assert!(make_any_binary().is_accept(0));
        assert!(!make_non_empty_binary().is_accept(0));
        assert!(run(&make_any_char(), "\u{10FFFF}"));
        assert_eq!(make_char_range(5, 4).get_num_states(), 0);
        assert!(make_char_class(&[1], &[]).is_err());
        assert_eq!(make_char_class(&[], &[]).unwrap().get_num_states(), 0);
        let mut a = Automaton::new();
        let s = a.create_state();
        let s1 = append_char(&mut a, s, 'q' as i32);
        let s2 = append_any_char(&mut a, s1);
        a.set_accept(s2, true);
        a.finish_state();
        assert!(run(&a, "q!"));
        assert!(run(&make_string_ints(&[1, 2]), "\u{1}\u{2}"));
        assert!(run(&make_case_insensitive_char('k' as i32), "\u{212A}"));
        assert!(make_decimal_interval(5, 1, 0).is_err());
        assert!(make_decimal_interval(1, 500, 2).is_err());
        let d = determinize(&make_decimal_interval(7, 12, 0).unwrap(), 1000).unwrap();
        assert!(run(&d, "0007") && run(&d, "12") && !run(&d, "13"));
        assert!(
            make_string_union(std::iter::empty())
                .unwrap()
                .get_num_states()
                == 0
        );
        assert!(
            make_binary_string_union(std::iter::empty())
                .unwrap()
                .get_num_states()
                == 0
        );
    }

    #[test]
    fn binary_interval_edges() {
        assert!(make_binary_interval(None, false, None, true).is_err());
        assert!(make_binary_interval(None, true, None, false).is_err());
        let all = make_binary_interval(None, true, None, true).unwrap();
        assert!(all.is_accept(0));
        let non_empty = make_binary_interval(Some(b""), false, None, true).unwrap();
        assert!(!non_empty.is_accept(0));
        assert_eq!(
            make_binary_interval(Some(b"a"), true, Some(b"a"), false)
                .unwrap()
                .get_num_states(),
            0
        );
        assert_eq!(
            make_binary_interval(Some(b"b"), true, Some(b"a"), true)
                .unwrap()
                .get_num_states(),
            0
        );
        let exact = make_binary_interval(Some(b"a"), true, Some(b"a"), true).unwrap();
        assert!(crate::automaton::operations::run_ints(&exact, &[97]));
        let zeros = make_binary_interval(Some(b"a"), false, Some(b"a\0"), false).unwrap();
        assert_eq!(zeros.get_num_states(), 0);
        let zeros = make_binary_interval(Some(b"a"), true, Some(b"a\0"), false).unwrap();
        assert!(crate::automaton::operations::run_ints(&zeros, &[97]));
    }
}
