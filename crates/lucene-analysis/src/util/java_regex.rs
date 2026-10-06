//! `java.util.regex.Pattern`/`Matcher`, as the analysis-common `pattern`
//! package and `PatternKeywordMarkerFilter` use them, over the `regex`
//! crate.
//!
//! What is Java's here:
//!
//! - `matches()` is a whole-input match, `find()` a leftmost search,
//!   `group(n)`, `start()`/`end()` in **UTF-16 code units** (Java `char`
//!   indices), as Lucene reports them.
//! - `\d`, `\w`, `\s` (and `\D`, `\W`, `\S` outside a class) are Java's
//!   ASCII classes, rewritten before compiling (the `regex` crate's are
//!   Unicode).
//! - The replacement string is Java's `appendReplacement` syntax: `$n`
//!   (greedy over the digits while the group exists), `${name}`, and `\`
//!   quoting the next character -- not the `regex` crate's `$name` syntax,
//!   under which `"$2_$1"` would name a group `2_`.
//!
//! Differs: the pattern language is the `regex` crate's -- no
//! backreferences, lookaround or possessive quantifiers (such a pattern is
//! an `IllegalArgument` error where Java would accept it), and `\W`/`\D`/`\S`
//! *inside* a character class keep the crate's Unicode meaning.

use regex::{Captures, Regex};

use crate::AnalysisError;

/// A compiled `java.util.regex.Pattern`.
#[derive(Debug, Clone)]
pub struct JavaPattern {
    regex: Regex,
    /// `^(?:pattern)$`, for `matches()`.
    whole: Regex,
}

/// Rewrites Java's ASCII-only perl classes into explicit ones.
fn translate(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut in_class = 0usize;
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(n) = chars.next() else {
                    out.push('\\');
                    break;
                };
                let body = match n {
                    'd' => Some("0-9"),
                    'w' => Some("0-9A-Za-z_"),
                    's' => Some(" \\t\\n\\x0B\\f\\r"),
                    _ => None,
                };
                let negated = match n {
                    'D' => Some("0-9"),
                    'W' => Some("0-9A-Za-z_"),
                    'S' => Some(" \\t\\n\\x0B\\f\\r"),
                    _ => None,
                };
                match (body, negated, in_class > 0) {
                    (Some(b), _, true) => out.push_str(b),
                    (Some(b), _, false) => {
                        out.push('[');
                        out.push_str(b);
                        out.push(']');
                    }
                    (None, Some(b), false) => {
                        out.push_str("[^");
                        out.push_str(b);
                        out.push(']');
                    }
                    _ => {
                        out.push('\\');
                        out.push(n);
                    }
                }
            }
            '[' => {
                in_class += 1;
                out.push(c);
                // A ']' right after '[' or '[^' is a literal.
                if chars.peek() == Some(&'^') {
                    out.push(chars.next().unwrap_or('^'));
                }
                if chars.peek() == Some(&']') {
                    chars.next();
                    out.push_str("\\]");
                }
            }
            ']' if in_class > 0 => {
                in_class -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

impl JavaPattern {
    /// `Pattern.compile(String)`.
    pub fn compile(pattern: &str) -> Result<Self, AnalysisError> {
        let t = translate(pattern);
        let err = |e: regex::Error| {
            AnalysisError::IllegalArgument(format!("PatternSyntaxException: {e}"))
        };
        Ok(JavaPattern {
            regex: Regex::new(&t).map_err(err)?,
            whole: Regex::new(&format!("^(?:{t})$")).map_err(err)?,
        })
    }

    /// The compiled search regex.
    pub fn regex(&self) -> &Regex {
        &self.regex
    }

    /// `matcher(s).matches()`.
    pub fn matches(&self, s: &str) -> bool {
        self.whole.is_match(s)
    }

    /// `Matcher.replaceAll`/`replaceFirst` with Java's replacement syntax.
    pub fn replace(&self, s: &str, replacement: &str, all: bool) -> Result<String, AnalysisError> {
        let mut out = String::with_capacity(s.len());
        let mut last = 0;
        for caps in self.regex.captures_iter(s) {
            let m = caps.get(0).expect("group 0");
            out.push_str(&s[last..m.start()]);
            append_replacement(&mut out, &caps, replacement)?;
            last = m.end();
            if !all {
                break;
            }
        }
        out.push_str(&s[last..]);
        Ok(out)
    }
}

/// A `java.util.regex.Matcher` over one text: `find()` from where the last
/// match ended (one character on after an empty match, as Java), and
/// `start(group)`/`end(group)` in UTF-16 units (`-1` for a group that did
/// not participate).
#[derive(Debug, Clone)]
pub struct JavaMatcher {
    regex: Regex,
    text: String,
    /// UTF-16 offset of every byte offset that starts a `char` (and the end).
    utf16_at: Vec<i32>,
    last: Option<(usize, usize)>,
    groups: Vec<Option<(usize, usize)>>,
}

impl JavaMatcher {
    /// `pattern.matcher(text)`.
    pub fn new(pattern: &JavaPattern, text: &str) -> Self {
        let mut m = JavaMatcher {
            regex: pattern.regex.clone(),
            text: String::new(),
            utf16_at: Vec::new(),
            last: None,
            groups: Vec::new(),
        };
        m.reset(text);
        m
    }

    /// `reset(CharSequence)`.
    pub fn reset(&mut self, text: &str) {
        self.text.clear();
        self.text.push_str(text);
        self.utf16_at.clear();
        self.utf16_at.resize(text.len() + 1, 0);
        let mut u = 0i32;
        for (i, c) in text.char_indices() {
            self.utf16_at[i] = u;
            u += c.len_utf16() as i32;
        }
        self.utf16_at[text.len()] = u;
        self.last = None;
        self.groups.clear();
    }

    /// The text being matched.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// `groupCount()`.
    pub fn group_count(&self) -> usize {
        self.regex.captures_len() - 1
    }

    /// `find()`.
    pub fn find(&mut self) -> bool {
        let from = match self.last {
            None => 0,
            Some((s, e)) if s == e => match self.text[e..].chars().next() {
                Some(c) => e + c.len_utf8(),
                None => {
                    self.groups.clear();
                    return false;
                }
            },
            Some((_, e)) => e,
        };
        match self.regex.captures_at(&self.text, from) {
            Some(caps) => {
                self.groups = (0..caps.len())
                    .map(|g| caps.get(g).map(|m| (m.start(), m.end())))
                    .collect();
                let m = caps.get(0).expect("group 0");
                self.last = Some((m.start(), m.end()));
                true
            }
            None => {
                self.groups.clear();
                false
            }
        }
    }

    /// The byte range of `group` in the last match.
    pub fn group_bytes(&self, group: usize) -> Option<(usize, usize)> {
        self.groups.get(group).copied().flatten()
    }

    /// `start(group)` in UTF-16 units, `-1` if the group did not match.
    pub fn start(&self, group: usize) -> i32 {
        self.group_bytes(group)
            .map_or(-1, |(s, _)| self.utf16_at[s])
    }

    /// `end(group)` in UTF-16 units, `-1` if the group did not match.
    pub fn end(&self, group: usize) -> i32 {
        self.group_bytes(group)
            .map_or(-1, |(_, e)| self.utf16_at[e])
    }
}

/// `Matcher.appendReplacement`'s expansion of `replacement` for one match.
pub(crate) fn append_replacement(
    out: &mut String,
    caps: &Captures<'_>,
    replacement: &str,
) -> Result<(), AnalysisError> {
    let bad = |m: &str| AnalysisError::IllegalArgument(m.to_string());
    let rep: Vec<char> = replacement.chars().collect();
    let mut i = 0;
    while i < rep.len() {
        let c = rep[i];
        if c == '\\' {
            i += 1;
            let Some(&n) = rep.get(i) else {
                return Err(bad("character to be escaped is missing"));
            };
            out.push(n);
            i += 1;
        } else if c == '$' {
            i += 1;
            let Some(&n) = rep.get(i) else {
                return Err(bad("Illegal group reference: group index is missing"));
            };
            if n == '{' {
                let close = rep[i..].iter().position(|&x| x == '}').map(|p| p + i);
                let Some(close) = close else {
                    return Err(bad("named capturing group is missing trailing '}'"));
                };
                let name: String = rep[i + 1..close].iter().collect();
                // An unknown name appends nothing (Java throws); a group
                // that did not participate appends nothing, as in Java.
                if let Some(m) = caps.name(&name) {
                    out.push_str(m.as_str());
                }
                i = close + 1;
            } else {
                let Some(first) = n.to_digit(10) else {
                    return Err(bad("Illegal group reference"));
                };
                let mut group = first as usize;
                i += 1;
                if group >= caps.len() {
                    return Err(bad(&format!("No group {group}")));
                }
                // Java: take further digits while the group number stays valid.
                while let Some(d) = rep.get(i).and_then(|c| c.to_digit(10)) {
                    let next = group * 10 + d as usize;
                    if next >= caps.len() {
                        break;
                    }
                    group = next;
                    i += 1;
                }
                if let Some(m) = caps.get(group) {
                    out.push_str(m.as_str());
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_classes_are_ascii() {
        let p = JavaPattern::compile("\\w+").unwrap();
        assert!(p.matches("abc_9"));
        assert!(!p.matches("é"));
        let p = JavaPattern::compile("[\\d\\s]+").unwrap();
        assert!(p.matches("1 2\t3"));
        assert!(!p.matches("١"));
        let p = JavaPattern::compile("\\D\\W\\S").unwrap();
        assert!(p.matches("a-x"));
        let p = JavaPattern::compile("[]a]+").unwrap();
        assert!(p.matches("]a]"));
        let p = JavaPattern::compile("[^]a]").unwrap();
        assert!(p.matches("b") && !p.matches("]"));
        assert!(JavaPattern::compile("\\.").unwrap().matches("."));
        assert!(JavaPattern::compile("(a").is_err());
        assert!(JavaPattern::compile("a\\").is_err());
        assert_eq!(translate("a\\"), "a\\");
    }

    #[test]
    fn matcher_finds_like_java() {
        let p = JavaPattern::compile("a*").unwrap();
        let mut m = JavaMatcher::new(&p, "baa😀");
        let mut found = Vec::new();
        while m.find() {
            found.push((m.start(0), m.end(0)));
        }
        assert_eq!(found, vec![(0, 0), (1, 3), (3, 3), (5, 5)]);
        let p = JavaPattern::compile("(x)|(y)").unwrap();
        let mut m = JavaMatcher::new(&p, "😀y");
        assert_eq!(m.group_count(), 2);
        assert!(m.find());
        assert_eq!((m.start(1), m.end(1), m.start(2), m.end(2)), (-1, -1, 2, 3));
        assert_eq!(m.text(), "😀y");
        assert!(!m.find());
        assert_eq!(m.start(0), -1, "a failed find clears the groups");
        assert!(!m.find(), "an empty last match at the end stays exhausted");
    }

    #[test]
    fn replacement_syntax_is_javas() {
        let p = JavaPattern::compile("([a-z]+)-([a-z]+)").unwrap();
        assert_eq!(
            p.replace("ab-cd ef-gh", "$2_$1", true).unwrap(),
            "cd_ab gh_ef"
        );
        assert_eq!(
            p.replace("ab-cd ef-gh", "$2_$1", false).unwrap(),
            "cd_ab ef-gh"
        );
        assert_eq!(p.replace("ab-cd", "\\$1$10", true).unwrap(), "$1ab0");
        let n = JavaPattern::compile("(?P<w>x)").unwrap();
        assert_eq!(n.replace("axb", "[${w}]", true).unwrap(), "a[x]b");
        assert!(p.replace("ab-cd", "$9", true).is_err());
        assert!(p.replace("ab-cd", "$", true).is_err());
        assert!(p.replace("ab-cd", "$x", true).is_err());
        assert!(p.replace("ab-cd", "x\\", true).is_err());
        assert!(n.replace("x", "${w", true).is_err());
        assert_eq!(p.replace("none", "$1", true).unwrap(), "none");
        assert!(p.regex().is_match("a-b"));
    }
}
