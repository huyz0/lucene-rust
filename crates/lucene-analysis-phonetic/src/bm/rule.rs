//! `org.apache.commons.codec.language.bm.Rule`: the rule files' grammar
//! (`"pattern" "left context" "right context" "phoneme expression"`,
//! `#include`, `//` and `/* */` comments), the context matchers and the
//! phonemes.

use std::collections::HashMap;
use std::sync::LazyLock;

use lucene_analysis::util::java_regex::JavaPattern;

use super::resources::resource;
use super::{LanguageSet, Languages, NameType, RuleType};
use crate::java::string;

/// `Rule.Phoneme`: a phoneme's text and the languages it holds for.
#[derive(Debug, Clone)]
pub struct Phoneme {
    /// `phonemeText`.
    pub text: Vec<u16>,
    /// `languages`.
    pub languages: LanguageSet,
}

/// `Rule.PhonemeExpr`: one phoneme, or a `(a|b|...)` list of them.
#[derive(Debug)]
pub enum PhonemeExpr {
    /// `Phoneme`.
    One(Phoneme),
    /// `PhonemeList`.
    List(Vec<Phoneme>),
}

impl PhonemeExpr {
    /// `getPhonemes()`.
    pub fn phonemes(&self) -> &[Phoneme] {
        match self {
            PhonemeExpr::One(p) => std::slice::from_ref(p),
            PhonemeExpr::List(l) => l,
        }
    }
}

/// `Rule.RPattern`: what `Rule.pattern(String regex)` builds for a context.
#[derive(Debug)]
enum RPattern {
    /// `ALL_STRINGS_RMATCHER`.
    All,
    /// `^$`: the empty string.
    Empty,
    /// `^content$`.
    Equals(Vec<u16>),
    /// `^content`.
    StartsWith(Vec<u16>),
    /// `content$`.
    EndsWith(Vec<u16>),
    /// `^[box]$`, `^[box]`, `[box]$` (`negate` for `[^box]`): the one
    /// unit, first unit or last unit in (or not in) the box.
    Box {
        units: Vec<u16>,
        should_match: bool,
        starts: bool,
        ends: bool,
    },
    /// Anything else: `Pattern.compile(regex).matcher(input).find()`.
    Regex(JavaPattern),
}

impl RPattern {
    // Java: Rule.pattern(String regex)
    fn new(regex: &str) -> RPattern {
        let starts = regex.starts_with('^');
        let ends = regex.ends_with('$');
        let content = &regex[usize::from(starts)..regex.len() - usize::from(ends)];
        let units = |s: &str| -> Vec<u16> { s.encode_utf16().collect() };
        if !content.contains('[') {
            if starts && ends {
                return if content.is_empty() {
                    RPattern::Empty
                } else {
                    RPattern::Equals(units(content))
                };
            }
            if (starts || ends) && content.is_empty() {
                return RPattern::All;
            }
            if starts {
                return RPattern::StartsWith(units(content));
            }
            if ends {
                return RPattern::EndsWith(units(content));
            }
        } else if content.starts_with('[') && content.ends_with(']') {
            let box_content = &content[1..content.len() - 1];
            if !box_content.contains('[') && (starts || ends) {
                let negate = box_content.starts_with('^');
                let box_content = if negate {
                    &box_content[1..]
                } else {
                    box_content
                };
                return RPattern::Box {
                    units: units(box_content),
                    should_match: !negate,
                    starts,
                    ends,
                };
            }
        }
        RPattern::Regex(JavaPattern::compile(regex).expect("Commons Codec's rule contexts compile"))
    }

    // Java: RPattern.isMatch
    fn is_match(&self, input: &[u16]) -> bool {
        match self {
            RPattern::All => true,
            RPattern::Empty => input.is_empty(),
            RPattern::Equals(c) => input == c.as_slice(),
            RPattern::StartsWith(c) => input.starts_with(c),
            RPattern::EndsWith(c) => input.ends_with(c),
            RPattern::Box {
                units,
                should_match,
                starts,
                ends,
            } => {
                let unit = match (starts, ends) {
                    (true, true) if input.len() == 1 => input.first(),
                    (true, true) => None,
                    (true, false) => input.first(),
                    _ => input.last(),
                };
                unit.is_some_and(|u| units.contains(u) == *should_match)
            }
            RPattern::Regex(p) => p.find_in(&string(input)),
        }
    }
}

/// `org.apache.commons.codec.language.bm.Rule`.
#[derive(Debug)]
pub struct Rule {
    pattern: Vec<u16>,
    l_context: RPattern,
    r_context: RPattern,
    phoneme: PhonemeExpr,
}

impl Rule {
    /// `getPattern()`.
    pub fn pattern(&self) -> &[u16] {
        &self.pattern
    }

    /// `getPhoneme()`.
    pub fn phoneme(&self) -> &PhonemeExpr {
        &self.phoneme
    }

    /// `Rule.patternAndContextMatches(input, i)`.
    pub fn pattern_and_context_matches(&self, input: &[u16], i: usize) -> bool {
        let ipl = i + self.pattern.len();
        if ipl > input.len() {
            return false;
        }
        if input[i..ipl] != self.pattern[..] {
            return false;
        }
        if !self.r_context.is_match(&input[ipl..]) {
            return false;
        }
        self.l_context.is_match(&input[..i])
    }
}

/// A rule file's rules by the first `char` of their pattern
/// (`Map<String, List<Rule>>` keyed by `pattern.substring(0, 1)`).
pub type RuleMap = HashMap<u16, Vec<Rule>>;

/// `Rule.stripQuotes`.
fn strip_quotes(s: &str) -> &str {
    let s = s.strip_prefix('"').unwrap_or(s);
    s.strip_suffix('"').unwrap_or(s)
}

// Java: Rule.parsePhoneme
fn parse_phoneme(ph: &str, languages: &Languages) -> Phoneme {
    match ph.find('[') {
        Some(open) => {
            assert!(ph.ends_with(']'), "phoneme {ph:?} has '[' but no ']'");
            let before = &ph[..open];
            let inner = &ph[open + 1..ph.len() - 1];
            let mut bits = 0;
            for l in inner.split('+') {
                bits |= languages
                    .bit(l)
                    .unwrap_or_else(|| panic!("rule language {l:?} is not one of the name type's"));
            }
            Phoneme {
                text: before.encode_utf16().collect(),
                languages: LanguageSet::from_bits(bits, false),
            }
        }
        None => Phoneme {
            text: ph.encode_utf16().collect(),
            languages: LanguageSet::Any,
        },
    }
}

// Java: Rule.parsePhonemeExpr
fn parse_phoneme_expr(ph: &str, languages: &Languages) -> PhonemeExpr {
    if let Some(rest) = ph.strip_prefix('(') {
        let body = rest.strip_suffix(')').expect("a '(' phoneme ends with ')'");
        // Java: body.split("[|]") -- trailing empty parts dropped, so an
        // empty alternative is added back when the body starts or ends
        // with '|'.
        let mut parts: Vec<&str> = body.split('|').collect();
        while parts.len() > 1 && parts.last().is_some_and(|p| p.is_empty()) {
            parts.pop();
        }
        let mut phs: Vec<Phoneme> = if parts.len() == 1 && parts[0].is_empty() && body.contains('|')
        {
            Vec::new()
        } else {
            parts
                .into_iter()
                .map(|p| parse_phoneme(p, languages))
                .collect()
        };
        if body.starts_with('|') || body.ends_with('|') {
            phs.push(Phoneme {
                text: Vec::new(),
                languages: LanguageSet::Any,
            });
        }
        return PhonemeExpr::List(phs);
    }
    PhonemeExpr::One(parse_phoneme(ph, languages))
}

// Java: Rule.parseRules
fn parse_rules(text: &str, languages: &Languages) -> RuleMap {
    let mut lines: RuleMap = HashMap::new();
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
        let line = super::trim_str(line);
        if line.is_empty() {
            continue;
        }
        if let Some(incl) = line.strip_prefix("#include") {
            let incl = super::trim_str(incl);
            assert!(!incl.contains(' '), "malformed #include {raw:?}");
            let included = resource(incl).unwrap_or_else(|| panic!("no rule file {incl}"));
            // Java: lines.putAll(...) -- an included key replaces the list.
            lines.extend(parse_rules(included, languages));
        } else {
            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            assert_eq!(parts.len(), 4, "malformed rule {raw:?}");
            let pattern = strip_quotes(parts[0]);
            let rule = Rule {
                pattern: pattern.encode_utf16().collect(),
                l_context: RPattern::new(&format!("{}$", strip_quotes(parts[1]))),
                r_context: RPattern::new(&format!("^{}", strip_quotes(parts[2]))),
                phoneme: parse_phoneme_expr(strip_quotes(parts[3]), languages),
            };
            lines.entry(rule.pattern[0]).or_default().push(rule);
        }
    }
    lines
}

/// Every rule map: `RULES[nameType][ruleType][language]`.
struct AllRules {
    maps: [[HashMap<&'static str, RuleMap>; 3]; 3],
}

static RULES: LazyLock<AllRules> = LazyLock::new(|| {
    let load = |nt: NameType| -> [HashMap<&'static str, RuleMap>; 3] {
        let languages = Languages::get(nt);
        [RuleType::Approx, RuleType::Exact, RuleType::Rules].map(|rt| {
            let mut rs: HashMap<&'static str, RuleMap> = HashMap::new();
            let mut names: Vec<&'static str> = languages.names().to_vec();
            if rt != RuleType::Rules {
                names.push("common");
            }
            for l in names {
                let file = format!("{}_{}_{}", nt.name(), rt.name(), l);
                let text = resource(&file).unwrap_or_else(|| panic!("no rule file {file}"));
                rs.insert(l, parse_rules(text, languages));
            }
            rs
        })
    };
    AllRules {
        maps: [NameType::Ashkenazi, NameType::Generic, NameType::Sephardic].map(load),
    }
});

/// `Rule.getInstanceMap(nameType, ruleType, lang)`; `None` where Java
/// throws `IllegalArgumentException("No rules found for ...")`.
pub fn instance_map(
    name_type: NameType,
    rule_type: RuleType,
    lang: &str,
) -> Option<&'static RuleMap> {
    RULES.maps[name_type.index()][rule_type.index()].get(lang)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::units;

    #[test]
    fn contexts() {
        let m = |re: &str, s: &str| RPattern::new(re).is_match(&units(s));
        assert!(m("^$", "") && !m("^$", "a"));
        assert!(m("^ab$", "ab") && !m("^ab$", "abc"));
        assert!(m("^", "x") && m("$", "x"));
        assert!(m("^ab", "abc") && !m("^ab", "cab"));
        assert!(m("ab$", "cab") && !m("ab$", "abc"));
        assert!(m("^[ab]$", "a") && !m("^[ab]$", "ab") && !m("^[ab]$", ""));
        assert!(m("^[ab]", "bz") && !m("^[ab]", "") && !m("^[ab]", "z"));
        assert!(m("[^ab]$", "az") && !m("[^ab]$", "za") && !m("[^ab]$", ""));
        assert!(matches!(RPattern::new("^[ln][bdf]"), RPattern::Regex(_)));
        assert!(m("^[ln][bdf]", "lbx") && !m("^[ln][bdf]", "lx"));
        assert!(m("D[^aeiEIou]$", "xDz") && !m("D[^aeiEIou]$", "xDa"));
    }

    #[test]
    fn phoneme_exprs() {
        let gen = Languages::get(NameType::Generic);
        let p = parse_phoneme_expr("(a|b[english+german]|)", gen);
        let ph = p.phonemes();
        assert_eq!(ph.len(), 3);
        assert_eq!(ph[0].text, units("a"));
        assert!(matches!(ph[1].languages, LanguageSet::Some { .. }));
        assert!(ph[2].text.is_empty());
        // Java: ["", "x"] plus the empty alternative a leading '|' adds.
        let p = parse_phoneme_expr("(|x)", gen);
        assert_eq!(p.phonemes().len(), 3);
        let p = parse_phoneme_expr("(|)", gen);
        assert_eq!(p.phonemes().len(), 1);
        assert_eq!(
            parse_phoneme_expr("x", gen).phonemes()[0].languages,
            LanguageSet::Any
        );
        assert_eq!(strip_quotes("\"a\""), "a");
    }

    #[test]
    fn rule_maps_load() {
        assert!(instance_map(NameType::Generic, RuleType::Rules, "english").is_some());
        assert!(instance_map(NameType::Generic, RuleType::Approx, "common").is_some());
        assert!(instance_map(NameType::Generic, RuleType::Rules, "common").is_none());
        assert!(instance_map(NameType::Sephardic, RuleType::Exact, "english").is_none());
        let r =
            &instance_map(NameType::Generic, RuleType::Rules, "any").unwrap()[&u16::from(b'a')][0];
        assert_eq!(r.pattern()[0], u16::from(b'a'));
        assert!(!r.phoneme().phonemes().is_empty());
        assert!(!r.pattern_and_context_matches(&units("a"), 1));
    }
}
