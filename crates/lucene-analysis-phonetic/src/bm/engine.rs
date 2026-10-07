//! `org.apache.commons.codec.language.bm.PhoneticEngine`: a name's phonemes
//! by the first pass's rules (`<nt>_rules_<lang>`), then rewritten by the
//! common and the language's final rules (`<nt>_<approx|exact>_...`).

use std::collections::BTreeMap;
use std::sync::LazyLock;

use super::lang::Lang;
use super::rule::{instance_map, Phoneme, PhonemeExpr, RuleMap};
use super::{CallerLanguages, LanguageSet, Languages, NameType, RuleType};
use crate::java::{java_hash_order, split_whitespace, to_lower, trim, units};
use crate::EncoderError;

/// `PhoneticEngine.DEFAULT_MAX_PHONEMES`.
pub const DEFAULT_MAX_PHONEMES: i32 = 20;

/// `PhoneticEngine.NAME_PREFIXES`, in the order Java's `HashSet` iterates
/// them (the first prefix that matches wins, so the order is behaviour).
pub fn name_prefixes(name_type: NameType) -> &'static [Vec<u16>] {
    static PREFIXES: LazyLock<[Vec<Vec<u16>>; 3]> = LazyLock::new(|| {
        let order = |words: &[&str]| -> Vec<Vec<u16>> {
            let u: Vec<Vec<u16>> = words.iter().map(|w| units(w)).collect();
            let refs: Vec<&[u16]> = u.iter().map(Vec::as_slice).collect();
            // `new HashSet<>(Arrays.asList(...))` sizes its table for
            // `max(size, 12)` entries at load factor .75.
            let capacity = (refs.len().max(12) * 4).div_ceil(3);
            java_hash_order(&refs, capacity)
                .into_iter()
                .map(<[u16]>::to_vec)
                .collect()
        };
        [
            order(&["bar", "ben", "da", "de", "van", "von"]),
            order(&[
                "da", "dal", "de", "del", "dela", "de la", "della", "des", "di", "do", "dos", "du",
                "van", "von",
            ]),
            order(&[
                "al", "el", "da", "dal", "de", "del", "dela", "de la", "della", "des", "di", "do",
                "dos", "du", "van", "von",
            ]),
        ]
    });
    &PREFIXES[name_type.index()]
}

/// `PhoneticEngine.PhonemeBuilder`: the alternatives built so far.
struct PhonemeBuilder {
    phonemes: Vec<Phoneme>,
}

impl PhonemeBuilder {
    // Java: PhonemeBuilder.empty
    fn empty(languages: LanguageSet) -> Self {
        PhonemeBuilder {
            phonemes: vec![Phoneme {
                text: Vec::new(),
                languages,
            }],
        }
    }

    // Java: PhonemeBuilder.append
    fn append(&mut self, s: &[u16]) {
        for p in &mut self.phonemes {
            p.text.extend_from_slice(s);
        }
    }

    // Java: PhonemeBuilder.apply -- every left x right whose languages
    // intersect, up to `max_phonemes`.
    fn apply(&mut self, expr: &PhonemeExpr, max_phonemes: i32) {
        let max = usize::try_from(max_phonemes).unwrap_or(0);
        let mut new: Vec<Phoneme> = Vec::new();
        'expr: for left in &self.phonemes {
            for right in expr.phonemes() {
                let languages = left.languages.restrict_to(right.languages);
                if languages.is_empty() {
                    continue;
                }
                if new.len() < max {
                    let mut text = Vec::with_capacity(left.text.len() + right.text.len());
                    text.extend_from_slice(&left.text);
                    text.extend_from_slice(&right.text);
                    new.push(Phoneme { text, languages });
                    if new.len() >= max {
                        break 'expr;
                    }
                }
            }
        }
        self.phonemes = new;
    }

    // Java: PhonemeBuilder.makeString
    fn make_string(&self) -> Vec<u16> {
        let mut out = Vec::new();
        for (i, p) in self.phonemes.iter().enumerate() {
            if i > 0 {
                out.push(u16::from(b'|'));
            }
            out.extend_from_slice(&p.text);
        }
        out
    }
}

/// `PhoneticEngine.RulesApplication.invoke`: the first rule keyed by
/// `input[i]` that matches at `i` applies its phoneme; returns the next
/// position and whether one matched.
fn invoke(
    rules: &RuleMap,
    input: &[u16],
    builder: &mut PhonemeBuilder,
    i: usize,
    max_phonemes: i32,
) -> (usize, bool) {
    if let Some(list) = rules.get(&input[i]) {
        for rule in list {
            if rule.pattern_and_context_matches(input, i) {
                builder.apply(rule.phoneme(), max_phonemes);
                return (i + rule.pattern().len(), true);
            }
        }
    }
    (i + 1, false)
}

/// `org.apache.commons.codec.language.bm.PhoneticEngine`.
#[derive(Debug, Clone)]
pub struct PhoneticEngine {
    name_type: NameType,
    rule_type: RuleType,
    concat: bool,
    max_phonemes: i32,
}

impl PhoneticEngine {
    /// `new PhoneticEngine(nameType, ruleType, concat)`; `Err` for
    /// [`RuleType::Rules`], as Java throws.
    pub fn new(
        name_type: NameType,
        rule_type: RuleType,
        concat: bool,
    ) -> Result<Self, EncoderError> {
        Self::with_max_phonemes(name_type, rule_type, concat, DEFAULT_MAX_PHONEMES)
    }

    /// `new PhoneticEngine(nameType, ruleType, concat, maxPhonemes)`.
    pub fn with_max_phonemes(
        name_type: NameType,
        rule_type: RuleType,
        concat: bool,
        max_phonemes: i32,
    ) -> Result<Self, EncoderError> {
        if rule_type == RuleType::Rules {
            return Err(EncoderError::illegal_argument("ruleType must not be RULES"));
        }
        Ok(PhoneticEngine {
            name_type,
            rule_type,
            concat,
            max_phonemes,
        })
    }

    /// `getNameType()`.
    pub fn name_type(&self) -> NameType {
        self.name_type
    }

    /// `getRuleType()`.
    pub fn rule_type(&self) -> RuleType {
        self.rule_type
    }

    /// `isConcat()`.
    pub fn is_concat(&self) -> bool {
        self.concat
    }

    /// `getMaxPhonemes()`.
    pub fn max_phonemes(&self) -> i32 {
        self.max_phonemes
    }

    /// The caller's language names as this name type's set
    /// (`LanguageSet.from(Set)`).
    pub fn languages(&self, names: &[String]) -> CallerLanguages {
        Languages::get(self.name_type).from_names(names)
    }

    /// `Rule.getInstanceMap(nameType, ruleType, languageSet)`.
    fn rule_map(
        &self,
        rule_type: RuleType,
        langs: &CallerLanguages,
    ) -> Result<&'static RuleMap, EncoderError> {
        let lang = langs.singleton().unwrap_or("any");
        instance_map(self.name_type, rule_type, lang).ok_or_else(|| {
            EncoderError::illegal_argument(format!(
                "No rules found for {}, {}, {lang}.",
                self.name_type.name(),
                rule_type.name()
            ))
        })
    }

    /// `PhoneticEngine.encode(String)`: the languages guessed from the
    /// input.
    pub fn encode(&self, input: &[u16]) -> Result<Vec<u16>, EncoderError> {
        let langs = Lang::instance(self.name_type).guess_languages(input);
        self.encode_with(input, &langs)
    }

    /// `PhoneticEngine.encode(String, LanguageSet)`.
    pub fn encode_with(
        &self,
        input: &[u16],
        languages: &CallerLanguages,
    ) -> Result<Vec<u16>, EncoderError> {
        let rules = self.rule_map(RuleType::Rules, languages)?;
        let final_rules1 = instance_map(self.name_type, self.rule_type, "common")
            .expect("approx and exact have common rules");
        let final_rules2 = self.rule_map(self.rule_type, languages)?;
        let lowered: Vec<u16> = to_lower(input)
            .into_iter()
            .map(|c| {
                if c == u16::from(b'-') {
                    u16::from(b' ')
                } else {
                    c
                }
            })
            .collect();
        let input = trim(&lowered);
        if self.name_type == NameType::Generic {
            let split = |prefix: &[u16], skip: usize| -> Result<Vec<u16>, EncoderError> {
                let remainder = &input[skip..];
                let mut combined = prefix.to_vec();
                combined.extend_from_slice(remainder);
                let mut out = units("(");
                out.extend(self.encode(remainder)?);
                out.extend(units(")-("));
                out.extend(self.encode(&combined)?);
                out.push(u16::from(b')'));
                Ok(out)
            };
            if input.starts_with(&units("d'")) {
                return split(&units("d"), 2);
            }
            for l in name_prefixes(self.name_type) {
                if input.starts_with(l) && input.get(l.len()) == Some(&u16::from(b' ')) {
                    return split(l, l.len() + 1);
                }
            }
        }
        let words = split_whitespace(input);
        let prefixes = name_prefixes(self.name_type);
        let words2: Vec<&[u16]> = match self.name_type {
            NameType::Sephardic => words
                .iter()
                .map(|w| {
                    // Java: aWord.split("'", -1), the last part.
                    let last = w
                        .iter()
                        .rposition(|&c| c == u16::from(b'\''))
                        .map_or(0, |p| p + 1);
                    &w[last..]
                })
                .filter(|w| !prefixes.iter().any(|p| p.as_slice() == *w))
                .collect(),
            NameType::Ashkenazi => words
                .iter()
                .copied()
                .filter(|w| !prefixes.iter().any(|p| p.as_slice() == *w))
                .collect(),
            NameType::Generic => words.clone(),
        };
        let joined: Vec<u16>;
        let input: &[u16] = if self.concat {
            joined = words2.join(&u16::from(b' '));
            &joined
        } else if words2.len() == 1 {
            // Java: `words.iterator().next()` -- the first word before the
            // prefixes were removed.
            words[0]
        } else if !words2.is_empty() {
            let mut result = Vec::new();
            for w in &words2 {
                result.push(u16::from(b'-'));
                result.extend(self.encode(w)?);
            }
            result.remove(0);
            return Ok(result);
        } else {
            input
        };
        let mut builder = PhonemeBuilder::empty(languages.set);
        let mut i = 0;
        while i < input.len() {
            i = invoke(rules, input, &mut builder, i, self.max_phonemes).0;
        }
        let builder = self.apply_final_rules(builder, final_rules1);
        let builder = self.apply_final_rules(builder, final_rules2);
        Ok(builder.make_string())
    }

    // Java: PhoneticEngine.applyFinalRules -- each phoneme rewritten by the
    // final rules, the results merged by text (a `TreeMap` by
    // `Phoneme.COMPARATOR`, which is UTF-16 order).
    fn apply_final_rules(&self, builder: PhonemeBuilder, final_rules: &RuleMap) -> PhonemeBuilder {
        if final_rules.is_empty() {
            return builder;
        }
        let mut phonemes: BTreeMap<Vec<u16>, LanguageSet> = BTreeMap::new();
        for phoneme in builder.phonemes {
            let mut sub = PhonemeBuilder::empty(phoneme.languages);
            let text = &phoneme.text;
            let mut i = 0;
            while i < text.len() {
                let (next, found) = invoke(final_rules, text, &mut sub, i, self.max_phonemes);
                if !found {
                    sub.append(&text[i..i + 1]);
                }
                i = next;
            }
            for p in sub.phonemes {
                match phonemes.get_mut(&p.text) {
                    Some(old) => *old = old.merge(p.languages),
                    None => {
                        phonemes.insert(p.text, p.languages);
                    }
                }
            }
        }
        PhonemeBuilder {
            phonemes: phonemes
                .into_iter()
                .map(|(text, languages)| Phoneme { text, languages })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::string;

    #[test]
    fn engine_basics() {
        assert!(PhoneticEngine::new(NameType::Generic, RuleType::Rules, true).is_err());
        let e = PhoneticEngine::new(NameType::Generic, RuleType::Approx, true).unwrap();
        assert_eq!(e.name_type(), NameType::Generic);
        assert_eq!(e.rule_type(), RuleType::Approx);
        assert!(e.is_concat());
        assert_eq!(e.max_phonemes(), DEFAULT_MAX_PHONEMES);
        assert_eq!(string(&e.encode(&units("")).unwrap()), "");
        let err = e
            .encode_with(&units("x"), &e.languages(&["klingon".into()]))
            .unwrap_err();
        assert_eq!(err.message(), "No rules found for gen, rules, klingon.");
        let none = e.encode_with(&units("x"), &e.languages(&[])).unwrap();
        assert!(none.is_empty());
        let zero =
            PhoneticEngine::with_max_phonemes(NameType::Generic, RuleType::Exact, true, 0).unwrap();
        assert_eq!(string(&zero.encode(&units("abc")).unwrap()), "");
    }

    #[test]
    fn prefix_order_is_a_permutation() {
        for (nt, n) in [
            (NameType::Ashkenazi, 6),
            (NameType::Generic, 14),
            (NameType::Sephardic, 16),
        ] {
            assert_eq!(name_prefixes(nt).len(), n);
        }
    }
}
