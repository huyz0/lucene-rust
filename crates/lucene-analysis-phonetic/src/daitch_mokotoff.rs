//! Commons Codec's `DaitchMokotoffSoundex` (1.17.2): six-digit codes from
//! the rules of `dmrules.txt` (vendored verbatim, Apache-2.0), with every
//! branch of the ambiguous rules (`soundex`) or the first one (`encode`).

use std::sync::LazyLock;

use crate::java::{is_whitespace, to_lower_char, trim};

const MAX_LENGTH: usize = 6;
const RULES_TEXT: &str = include_str!("resources/dmrules.txt");

/// `DaitchMokotoffSoundex.Rule`.
#[derive(Debug)]
struct Rule {
    pattern: Vec<u16>,
    at_start: Vec<Vec<u16>>,
    before_vowel: Vec<Vec<u16>>,
    default: Vec<Vec<u16>>,
}

/// `replacement.split("\\|")`: Java's split, so `"|6"` is `["", "6"]` and a
/// trailing empty part is dropped.
fn split_bar(s: &str) -> Vec<Vec<u16>> {
    let mut parts: Vec<Vec<u16>> = s.split('|').map(|p| p.encode_utf16().collect()).collect();
    while parts.len() > 1 && parts.last().is_some_and(Vec::is_empty) {
        parts.pop();
    }
    parts
}

impl Rule {
    // Java: Rule.getReplacements
    fn replacements(&self, context: &[u16], at_start: bool) -> &[Vec<u16>] {
        if at_start {
            return &self.at_start;
        }
        let next_is_vowel = context
            .get(self.pattern.len())
            .is_some_and(|&c| b"aeiou".iter().any(|&v| c == u16::from(v)));
        if next_is_vowel {
            &self.before_vowel
        } else {
            &self.default
        }
    }
}

/// The parsed rules: by first `char` (sorted longest pattern first, a
/// stable sort as `List.sort` is) and the ASCII foldings.
struct Rules {
    rules: Vec<(u16, Vec<Rule>)>,
    foldings: Vec<(u16, u16)>,
}

impl Rules {
    fn get(&self, ch: u16) -> Option<&[Rule]> {
        self.rules
            .binary_search_by_key(&ch, |(c, _)| *c)
            .ok()
            .map(|i| self.rules[i].1.as_slice())
    }

    fn folding(&self, ch: u16) -> Option<u16> {
        self.foldings
            .binary_search_by_key(&ch, |&(c, _)| c)
            .ok()
            .map(|i| self.foldings[i].1)
    }
}

/// `DaitchMokotoffSoundex.stripQuotes`.
fn strip_quotes(s: &str) -> &str {
    let s = s.strip_prefix('"').unwrap_or(s);
    s.strip_suffix('"').unwrap_or(s)
}

/// `DaitchMokotoffSoundex.parseRules` over the vendored file (which Java
/// accepts, so every malformed-line branch is unreachable here and the
/// parser asserts instead).
fn parse_rules(text: &str) -> Rules {
    let mut by_char: Vec<(u16, Vec<Rule>)> = Vec::new();
    let mut foldings: Vec<(u16, u16)> = Vec::new();
    let mut in_comment = false;
    for raw in text.lines() {
        if in_comment {
            if raw.ends_with("*/") {
                in_comment = false;
            }
            continue;
        }
        if raw.starts_with("/*") {
            in_comment = true;
            continue;
        }
        let line = raw.find("//").map_or(raw, |i| &raw[..i]);
        let line: Vec<u16> = line.encode_utf16().collect();
        let line = String::from_utf16_lossy(trim(&line));
        if line.is_empty() {
            continue;
        }
        if let Some((l, r)) = line.split_once('=') {
            let (l, r): (Vec<u16>, Vec<u16>) =
                (l.encode_utf16().collect(), r.encode_utf16().collect());
            assert!(l.len() == 1 && r.len() == 1, "malformed folding {raw:?}");
            match foldings.iter_mut().find(|(c, _)| *c == l[0]) {
                Some(f) => f.1 = r[0],
                None => foldings.push((l[0], r[0])),
            }
        } else {
            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            assert_eq!(parts.len(), 4, "malformed rule {raw:?}");
            let rule = Rule {
                pattern: strip_quotes(parts[0]).encode_utf16().collect(),
                at_start: split_bar(strip_quotes(parts[1])),
                before_vowel: split_bar(strip_quotes(parts[2])),
                default: split_bar(strip_quotes(parts[3])),
            };
            let key = rule.pattern[0];
            match by_char.iter_mut().find(|(c, _)| *c == key) {
                Some((_, v)) => v.push(rule),
                None => by_char.push((key, vec![rule])),
            }
        }
    }
    for (_, v) in &mut by_char {
        v.sort_by_key(|r| std::cmp::Reverse(r.pattern.len()));
    }
    by_char.sort_by_key(|(c, _)| *c);
    foldings.sort_unstable();
    Rules {
        rules: by_char,
        foldings,
    }
}

static RULES: LazyLock<Rules> = LazyLock::new(|| parse_rules(RULES_TEXT));

/// `DaitchMokotoffSoundex.Branch`: one code being built.
#[derive(Debug, Clone)]
struct Branch {
    builder: Vec<u16>,
    last_replacement: Option<Vec<u16>>,
}

impl Branch {
    // Java: Branch.processNextReplacement
    fn process_next_replacement(&mut self, replacement: &[u16], force_append: bool) {
        let append = match &self.last_replacement {
            None => true,
            Some(last) => !last.ends_with(replacement) || force_append,
        };
        if append && self.builder.len() < MAX_LENGTH {
            self.builder.extend_from_slice(replacement);
            self.builder.truncate(MAX_LENGTH);
        }
        self.last_replacement = Some(replacement.to_vec());
    }

    // Java: Branch.finish
    fn finish(&mut self) {
        self.builder
            .resize(self.builder.len().max(MAX_LENGTH), u16::from(b'0'));
    }
}

/// `org.apache.commons.codec.language.DaitchMokotoffSoundex`.
#[derive(Debug, Clone)]
pub struct DaitchMokotoffSoundex {
    folding: bool,
}

impl Default for DaitchMokotoffSoundex {
    /// `new DaitchMokotoffSoundex()`: ASCII folding on.
    fn default() -> Self {
        DaitchMokotoffSoundex { folding: true }
    }
}

impl DaitchMokotoffSoundex {
    /// `new DaitchMokotoffSoundex(boolean folding)`.
    pub fn new(folding: bool) -> Self {
        DaitchMokotoffSoundex { folding }
    }

    // Java: DaitchMokotoffSoundex.cleanup
    fn cleanup(&self, input: &[u16]) -> Vec<u16> {
        input
            .iter()
            .filter(|&&c| !is_whitespace(c))
            .map(|&c| {
                let c = to_lower_char(c);
                match RULES.folding(c) {
                    Some(f) if self.folding => f,
                    _ => c,
                }
            })
            .collect()
    }

    /// `DaitchMokotoffSoundex.encode(String)`: the first branch only.
    pub fn encode(&self, source: &[u16]) -> Vec<u16> {
        self.soundex_branches(source, false).swap_remove(0)
    }

    /// `DaitchMokotoffSoundex.soundex(String)`: every branch, `|`-joined.
    pub fn soundex(&self, source: &[u16]) -> Vec<u16> {
        let branches = self.soundex_branches(source, true);
        let mut out = Vec::with_capacity(branches.len() * (MAX_LENGTH + 1));
        for (i, b) in branches.into_iter().enumerate() {
            if i > 0 {
                out.push(u16::from(b'|'));
            }
            out.extend(b);
        }
        out
    }

    // Java: DaitchMokotoffSoundex.soundex(String, boolean)
    fn soundex_branches(&self, source: &[u16], branching: bool) -> Vec<Vec<u16>> {
        let input = self.cleanup(source);
        // Java: a LinkedHashSet of branches equal by their string.
        let mut current = vec![Branch {
            builder: Vec::new(),
            last_replacement: None,
        }];
        let mut last_char: u16 = 0;
        let mut index = 0;
        while index < input.len() {
            let ch = input[index];
            if is_whitespace(ch) {
                index += 1;
                continue;
            }
            let context = &input[index..];
            let Some(rules) = RULES.get(ch) else {
                index += 1;
                continue;
            };
            for rule in rules {
                if !context.starts_with(&rule.pattern) {
                    continue;
                }
                let replacements = rule.replacements(context, last_char == 0);
                let branching_required = replacements.len() > 1 && branching;
                let force = last_char == u16::from(b'm') && ch == u16::from(b'n')
                    || last_char == u16::from(b'n') && ch == u16::from(b'm');
                let mut next: Vec<Branch> = Vec::new();
                for branch in &mut current {
                    for replacement in replacements {
                        if branching_required {
                            let mut nb = Branch {
                                builder: branch.builder.clone(),
                                last_replacement: branch.last_replacement.clone(),
                            };
                            nb.process_next_replacement(replacement, force);
                            next.push(nb);
                        } else {
                            branch.process_next_replacement(replacement, force);
                            if branching {
                                next.push(branch.clone());
                            }
                            break;
                        }
                    }
                }
                if branching {
                    // Java: currentBranches.clear(); addAll(nextBranches) --
                    // equal strings collapse onto the first.
                    current.clear();
                    for b in next {
                        if !current.iter().any(|c| c.builder == b.builder) {
                            current.push(b);
                        }
                    }
                }
                index += rule.pattern.len() - 1;
                break;
            }
            last_char = ch;
            index += 1;
        }
        current
            .into_iter()
            .map(|mut b| {
                b.finish();
                b.builder
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    #[test]
    fn rules_parse_as_java() {
        assert_eq!(split_bar("|6"), [vec![], units("6")]);
        assert_eq!(split_bar(""), [Vec::<u16>::new()]);
        assert_eq!(split_bar("4|"), [units("4")]);
        assert_eq!(strip_quotes("\"ab\""), "ab");
        assert_eq!(RULES.folding(u16::from(b'z')), None);
        assert_eq!(RULES.folding(0xDF), Some(u16::from(b's')));
        // Longest pattern first.
        let s = RULES.get(u16::from(b's')).unwrap();
        assert!(s
            .windows(2)
            .all(|w| w[0].pattern.len() >= w[1].pattern.len()));
    }

    #[test]
    fn empty_and_unmapped() {
        let e = DaitchMokotoffSoundex::default();
        assert_eq!(string(&e.encode(&units(""))), "000000");
        assert_eq!(string(&e.soundex(&units(" 1 "))), "000000");
        assert!(!DaitchMokotoffSoundex::new(false).folding);
    }
}
