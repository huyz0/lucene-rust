//! `AffixCondition`: the "condition" column of a `PFX`/`SFX` rule, checked
//! against a stem with both the strip and the affix removed.
//!
//! As in Lucene, a condition with no `[`, `.` or `-` is a literal compared
//! character for character; anything else becomes a `RegExp` (dashes
//! escaped) run by a `CharacterRunAutomaton`, after the part that overlaps
//! the strip is checked with `String.matches` (`java.util.regex`, here the
//! crate's `JavaPattern`).

use lucene_util::automaton::{
    operations, CharacterRunAutomaton, RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use super::HunspellError;
use crate::util::JavaPattern;

/// `AffixCondition.ALWAYS_TRUE_KEY`.
pub(crate) const ALWAYS_TRUE_KEY: &str = ".*";

/// `AffixKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AffixKind {
    /// `PREFIX`.
    Prefix,
    /// `SUFFIX`.
    Suffix,
}

/// A compiled `AffixCondition`.
#[derive(Debug)]
pub(crate) enum AffixCondition {
    /// `ALWAYS_TRUE`.
    AlwaysTrue,
    /// `ALWAYS_FALSE`.
    AlwaysFalse,
    /// `substringCondition`: the stem starts (prefix) or ends (suffix) with
    /// these units.
    Substring { for_suffix: bool, text: Vec<u16> },
    /// `regexpCondition`: the first (prefix) or last (suffix) `char_count`
    /// units run through the automaton.
    Regexp {
        for_suffix: bool,
        char_count: usize,
        automaton: CharacterRunAutomaton,
    },
}

fn s(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

fn contains(units: &[u16], c: u8) -> bool {
    units.contains(&u16::from(c))
}

/// `AffixCondition.isRegexp`.
fn is_regexp(condition: &[u16]) -> bool {
    contains(condition, b'[') || contains(condition, b'.') || contains(condition, b'-')
}

/// `AffixCondition.uniqueKey`.
pub(crate) fn unique_key(kind: AffixKind, strip: &[u16], condition: &[u16]) -> String {
    if condition == [u16::from(b'.')]
        || kind == AffixKind::Prefix && strip.starts_with(condition)
        || kind == AffixKind::Suffix && strip.ends_with(condition) && !is_regexp(condition)
    {
        return ALWAYS_TRUE_KEY.to_string();
    }
    format!(
        "{} {} {}",
        s(condition),
        if kind == AffixKind::Prefix {
            "PREFIX"
        } else {
            "SUFFIX"
        },
        s(strip)
    )
}

/// `AffixCondition.skipCharPattern`: past one literal or `[...]` at `pos`.
fn skip_char_pattern(condition: &[u16], pos: usize) -> Result<usize, HunspellError> {
    if condition[pos] == u16::from(b'[') {
        match condition[pos + 1..]
            .iter()
            .position(|&c| c == u16::from(b']'))
        {
            Some(p) => Ok(pos + 1 + p + 1),
            None => Err(HunspellError::IllegalArgument(format!(
                "Malformed condition {}",
                s(condition)
            ))),
        }
    } else {
        Ok(pos + 1)
    }
}

fn skip_char_patterns(condition: &[u16], count: usize) -> Result<usize, HunspellError> {
    let mut pos = 0;
    for _ in 0..count {
        pos = skip_char_pattern(condition, pos)?;
    }
    Ok(pos)
}

fn count_char_patterns(condition: &[u16]) -> Result<usize, HunspellError> {
    let (mut n, mut i) = (0, 0);
    while i < condition.len() {
        n += 1;
        i = skip_char_pattern(condition, i)?;
    }
    Ok(n)
}

/// `String.matches(regex)`: `None` when the pattern does not compile
/// (Java's `PatternSyntaxException`).
fn java_matches(text: &[u16], regex: &str) -> Option<bool> {
    JavaPattern::compile(regex)
        .ok()
        .map(|p| p.matches(&s(text)))
}

/// `AffixCondition.escapeDash`.
fn escape_dash(re: &[u16]) -> Vec<u16> {
    let dash = u16::from(b'-');
    let backslash = u16::from(b'\\');
    if !re.contains(&dash) {
        return re.to_vec();
    }
    let mut out = Vec::with_capacity(re.len() + 4);
    let mut i = 0;
    while i < re.len() {
        let c = re[i];
        if c == dash {
            out.extend_from_slice(&[backslash, dash]);
        } else {
            out.push(c);
            if c == backslash && i + 1 < re.len() {
                out.push(re[i + 1]);
                i += 1;
            }
        }
        i += 1;
    }
    out
}

impl AffixCondition {
    /// `AffixCondition.compile`.
    pub(crate) fn compile(
        kind: AffixKind,
        strip: &[u16],
        condition: &[u16],
        line: &[u16],
    ) -> Result<AffixCondition, HunspellError> {
        if !is_regexp(condition) {
            if kind == AffixKind::Suffix && condition.ends_with(strip) {
                return Ok(AffixCondition::Substring {
                    for_suffix: true,
                    text: condition[..condition.len() - strip.len()].to_vec(),
                });
            }
            if kind == AffixKind::Prefix && condition.starts_with(strip) {
                return Ok(AffixCondition::Substring {
                    for_suffix: false,
                    text: condition[strip.len()..].to_vec(),
                });
            }
            return Ok(AffixCondition::AlwaysFalse);
        }
        let mut condition = condition.to_vec();
        if let Some(last_bracket) = condition.iter().rposition(|&c| c == u16::from(b'[')) {
            if !condition[last_bracket + 1..].contains(&u16::from(b']')) {
                // An unclosed `[` is tolerated by Hunspell.
                condition.push(u16::from(b']'));
            }
        }
        let on_line =
            |e: HunspellError| HunspellError::IllegalArgument(format!("On line: {}: {e}", s(line)));
        let condition_chars = count_char_patterns(&condition).map_err(on_line)?;
        if condition_chars <= strip.len() {
            let regex = if kind == AffixKind::Prefix {
                format!(".*{}", s(&condition))
            } else {
                format!("{}.*", s(&condition))
            };
            return Ok(match java_matches(strip, &regex) {
                Some(true) => AffixCondition::AlwaysTrue,
                _ => AffixCondition::AlwaysFalse,
            });
        }
        let rest = condition_chars - strip.len();
        let (strip_part, stem_part) = if kind == AffixKind::Prefix {
            let split = skip_char_patterns(&condition, strip.len()).map_err(on_line)?;
            (condition[..split].to_vec(), condition[split..].to_vec())
        } else {
            let split = skip_char_patterns(&condition, rest).map_err(on_line)?;
            (condition[split..].to_vec(), condition[..split].to_vec())
        };
        match java_matches(strip, &s(&strip_part)) {
            Some(true) => {}
            _ => return Ok(AffixCondition::AlwaysFalse),
        }
        let automaton = RegExp::with_flags(&s(&escape_dash(&stem_part)), RegExp::NONE, 0)
            .and_then(|re| re.to_automaton())
            .map_err(|e| HunspellError::IllegalArgument(format!("On line: {}: {e}", s(line))))?;
        let det = operations::determinize(&automaton, DEFAULT_DETERMINIZE_WORK_LIMIT)
            .map_err(|e| HunspellError::IllegalArgument(format!("On line: {}: {e}", s(line))))?;
        let automaton = CharacterRunAutomaton::new(&det)
            .map_err(|e| HunspellError::IllegalArgument(format!("On line: {}: {e}", s(line))))?;
        Ok(AffixCondition::Regexp {
            for_suffix: kind == AffixKind::Suffix,
            char_count: rest,
            automaton,
        })
    }

    /// `AffixCondition.acceptsStem(char[], offset, length)` over `stem`.
    pub(crate) fn accepts_stem(&self, stem: &[u16]) -> bool {
        match self {
            AffixCondition::AlwaysTrue => true,
            AffixCondition::AlwaysFalse => false,
            AffixCondition::Substring { for_suffix, text } => {
                if stem.len() < text.len() {
                    return false;
                }
                if *for_suffix {
                    stem.ends_with(text)
                } else {
                    stem.starts_with(text)
                }
            }
            AffixCondition::Regexp {
                for_suffix,
                char_count,
                automaton,
            } => {
                if stem.len() < *char_count {
                    return false;
                }
                let part = if *for_suffix {
                    &stem[stem.len() - char_count..]
                } else {
                    &stem[..*char_count]
                };
                automaton.run_code_points(
                    char::decode_utf16(part.iter().copied()).map(|c| {
                        c.map_or_else(|e| i32::from(e.unpaired_surrogate()), |c| c as i32)
                    }),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn a_bracket_without_its_close_is_malformed() {
        // `compile` closes a trailing `[` first, so only a direct call sees it.
        assert!(matches!(
            skip_char_pattern(&units("[ab"), 0),
            Err(HunspellError::IllegalArgument(_))
        ));
    }

    #[test]
    fn a_regexp_condition_needs_enough_stem() {
        let c = AffixCondition::compile(AffixKind::Suffix, &[], &units("[ab]c"), &[]).unwrap();
        assert!(!c.accepts_stem(&units("c")));
        assert!(c.accepts_stem(&units("xbc")));
        let p = AffixCondition::compile(AffixKind::Prefix, &[], &units("[ab]c"), &[]).unwrap();
        assert!(p.accepts_stem(&units("acx")));
        assert!(!p.accepts_stem(&units("c")));
    }
}
