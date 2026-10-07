//! `org.apache.commons.codec.language.bm.Lang`: guessing a name's
//! languages from the `<nt>_lang.txt` rules (a regex `find` each, which
//! keeps or removes its languages).

use std::sync::LazyLock;

use lucene_analysis::util::java_regex::JavaPattern;

use super::resources::resource;
use super::{CallerLanguages, LanguageSet, Languages, NameType};
use crate::java::{string, to_lower};

/// `Lang.LangRule`.
struct LangRule {
    pattern: JavaPattern,
    languages: u64,
    accept_on_match: bool,
}

/// `org.apache.commons.codec.language.bm.Lang`.
pub struct Lang {
    rules: Vec<LangRule>,
    languages: &'static Languages,
}

impl Lang {
    /// `Lang.instance(NameType)`.
    pub fn instance(name_type: NameType) -> &'static Lang {
        static ALL: LazyLock<[Lang; 3]> = LazyLock::new(|| {
            [NameType::Ashkenazi, NameType::Generic, NameType::Sephardic].map(Lang::load)
        });
        &ALL[name_type.index()]
    }

    // Java: Lang.loadFromResource
    fn load(name_type: NameType) -> Lang {
        let languages = Languages::get(name_type);
        let text =
            resource(&format!("{}_lang", name_type.name())).expect("a lang file per name type");
        let mut rules = Vec::new();
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
            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            assert_eq!(parts.len(), 3, "malformed lang rule {raw:?}");
            let pattern =
                JavaPattern::compile(parts[0]).expect("Commons Codec's lang patterns compile");
            let mut bits = 0;
            for l in parts[1].split('+') {
                // A rule naming a language the name type lacks matches
                // nothing in the guessed set, as in Java.
                bits |= languages.bit(l).unwrap_or(0);
            }
            rules.push(LangRule {
                pattern,
                languages: bits,
                accept_on_match: parts[2] == "true",
            });
        }
        Lang { rules, languages }
    }

    /// `Lang.guessLanguages(String)`: `ANY_LANGUAGE` when no language is
    /// left.
    pub fn guess_languages(&self, input: &[u16]) -> CallerLanguages {
        let text = string(&to_lower(input));
        let mut langs = self.languages.all();
        for rule in &self.rules {
            if rule.pattern.find_in(&text) {
                if rule.accept_on_match {
                    langs &= rule.languages;
                } else {
                    langs &= !rule.languages;
                }
            }
        }
        if langs == 0 {
            return CallerLanguages {
                set: LanguageSet::Any,
                names: Vec::<String>::new().into(),
            };
        }
        let names: Vec<String> = self
            .languages
            .names()
            .iter()
            .enumerate()
            .filter(|(i, _)| langs & (1 << i) != 0)
            .map(|(_, n)| (*n).to_string())
            .collect();
        CallerLanguages {
            set: LanguageSet::Some {
                bits: langs,
                unknown: false,
            },
            names: names.into(),
        }
    }

    /// `Lang.guessLanguage(String)`: the language if exactly one is left,
    /// else `any`.
    pub fn guess_language(&self, input: &[u16]) -> String {
        self.guess_languages(input)
            .singleton()
            .unwrap_or("any")
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::units;

    #[test]
    fn guesses() {
        let gen = Lang::instance(NameType::Generic);
        assert_eq!(gen.guess_language(&units("Renault")), "french");
        assert_eq!(gen.guess_language(&units("Mickiewicz")), "polish");
        assert_eq!(gen.guess_language(&units("Smith")), "any");
        assert_eq!(
            gen.guess_languages(&units("")).set,
            LanguageSet::Some {
                bits: Languages::get(NameType::Generic).all(),
                unknown: false
            }
        );
    }
}
