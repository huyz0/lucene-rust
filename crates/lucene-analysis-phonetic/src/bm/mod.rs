//! Commons Codec's Beider-Morse phonetic matching
//! (`org.apache.commons.codec.language.bm`, 1.17.2): `NameType`,
//! `RuleType`, `Languages` and its `LanguageSet`s, `Lang` (language
//! guessing), `Rule` and `PhoneticEngine`, over the rule files Commons Codec
//! ships ([`resources`]).
//!
//! # Language sets
//!
//! Java's `LanguageSet`s are `HashSet<String>`s behind two singletons
//! (`NO_LANGUAGES`, `ANY_LANGUAGE`). Only their contents are ever
//! observable -- `getAny()` is called on singletons alone -- so the port
//! keeps a bit per language of the name type's `languages` file
//! ([`LanguageSet::Some`]). A caller's set may name languages the name type
//! does not have (the factory's `languageSet` argument is free text): those
//! can only ever be kept whole (`restrictTo(ANY_LANGUAGE)`, `merge`) or
//! dropped (intersected with a rule's languages, which never hold them), so
//! one flag stands for all of them, and the caller's names are kept apart
//! ([`Languages::from_names`]) for the one place a name is read: looking up
//! a singleton's rules.

mod engine;
mod lang;
pub mod resources;
mod rule;

use std::fmt;
use std::sync::{Arc, LazyLock};

pub use engine::{name_prefixes, PhoneticEngine, DEFAULT_MAX_PHONEMES};
pub use lang::Lang;
pub use rule::{instance_map, Phoneme, PhonemeExpr, Rule, RuleMap};

/// `org.apache.commons.codec.language.bm.NameType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NameType {
    /// `ASHKENAZI` (`ash`).
    Ashkenazi,
    /// `GENERIC` (`gen`).
    Generic,
    /// `SEPHARDIC` (`sep`).
    Sephardic,
}

impl NameType {
    /// `NameType.getName()`.
    pub fn name(self) -> &'static str {
        match self {
            NameType::Ashkenazi => "ash",
            NameType::Generic => "gen",
            NameType::Sephardic => "sep",
        }
    }

    /// `NameType.valueOf(String)`: the constant's Java name.
    pub fn value_of(s: &str) -> Option<Self> {
        match s {
            "ASHKENAZI" => Some(NameType::Ashkenazi),
            "GENERIC" => Some(NameType::Generic),
            "SEPHARDIC" => Some(NameType::Sephardic),
            _ => None,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// `org.apache.commons.codec.language.bm.RuleType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleType {
    /// `APPROX`.
    Approx,
    /// `EXACT`.
    Exact,
    /// `RULES`: the first pass's rules; not an engine's rule type.
    Rules,
}

impl RuleType {
    /// `RuleType.getName()`.
    pub fn name(self) -> &'static str {
        match self {
            RuleType::Approx => "approx",
            RuleType::Exact => "exact",
            RuleType::Rules => "rules",
        }
    }

    /// `RuleType.valueOf(String)`.
    pub fn value_of(s: &str) -> Option<Self> {
        match s {
            "APPROX" => Some(RuleType::Approx),
            "EXACT" => Some(RuleType::Exact),
            "RULES" => Some(RuleType::Rules),
            _ => None,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// `Languages.LanguageSet` (see the module docs for the representation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageSet {
    /// `Languages.NO_LANGUAGES`.
    No,
    /// `Languages.ANY_LANGUAGE`.
    Any,
    /// `SomeLanguages`: a bit per language of the name type, plus whether
    /// the caller's unknown languages are in it. Never empty (Java's
    /// `LanguageSet.from` turns an empty set into `NO_LANGUAGES`).
    Some {
        /// Bit `i`: the name type's `i`th language.
        bits: u64,
        /// The caller's languages the name type does not have.
        unknown: bool,
    },
}

impl LanguageSet {
    /// `LanguageSet.from(Set)` over bits.
    fn from_bits(bits: u64, unknown: bool) -> Self {
        if bits == 0 && !unknown {
            LanguageSet::No
        } else {
            LanguageSet::Some { bits, unknown }
        }
    }

    /// `LanguageSet.isEmpty()`.
    pub fn is_empty(self) -> bool {
        matches!(self, LanguageSet::No)
    }

    /// `LanguageSet.restrictTo(other)`.
    pub fn restrict_to(self, other: LanguageSet) -> LanguageSet {
        match (self, other) {
            (LanguageSet::No, _) => LanguageSet::No,
            (LanguageSet::Any, o) => o,
            (s @ LanguageSet::Some { .. }, LanguageSet::Any) => s,
            (LanguageSet::Some { .. }, LanguageSet::No) => LanguageSet::No,
            (
                LanguageSet::Some {
                    bits: a,
                    unknown: ua,
                },
                LanguageSet::Some {
                    bits: b,
                    unknown: ub,
                },
            ) => Self::from_bits(a & b, ua && ub),
        }
    }

    /// `LanguageSet.merge(other)`.
    pub fn merge(self, other: LanguageSet) -> LanguageSet {
        match (self, other) {
            (LanguageSet::No | LanguageSet::Any, o) => o,
            (s @ LanguageSet::Some { .. }, LanguageSet::No) => s,
            (LanguageSet::Some { .. }, LanguageSet::Any) => LanguageSet::Any,
            (
                LanguageSet::Some {
                    bits: a,
                    unknown: ua,
                },
                LanguageSet::Some {
                    bits: b,
                    unknown: ub,
                },
            ) => Self::from_bits(a | b, ua || ub),
        }
    }
}

/// `org.apache.commons.codec.language.bm.Languages`: a name type's
/// languages, in the order bits are assigned (the file's).
#[derive(Debug)]
pub struct Languages {
    names: Vec<&'static str>,
}

impl Languages {
    /// `Languages.getInstance(NameType)`.
    pub fn get(name_type: NameType) -> &'static Languages {
        static ALL: LazyLock<[Languages; 3]> = LazyLock::new(|| {
            [NameType::Ashkenazi, NameType::Generic, NameType::Sephardic].map(Languages::load)
        });
        &ALL[name_type.index()]
    }

    // Java: Languages.getInstance(String languagesResourceName)
    fn load(name_type: NameType) -> Languages {
        let text = resources::resource(&format!("{}_languages", name_type.name()))
            .expect("every name type has a languages file");
        let mut names: Vec<&'static str> = Vec::new();
        let mut in_comment = false;
        for raw in text.lines() {
            let line = trim_str(raw);
            if in_comment {
                if line.ends_with("*/") {
                    in_comment = false;
                }
            } else if line.starts_with("/*") {
                in_comment = true;
            } else if !line.is_empty() && !names.contains(&line) {
                names.push(line);
            }
        }
        assert!(names.len() <= 64, "a bit per language");
        Languages { names }
    }

    /// The language names, bit order.
    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    /// The bit of `name`, if the name type has it.
    pub fn bit(&self, name: &str) -> Option<u64> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| 1u64 << i)
    }

    /// Every language (`new HashSet<>(languages)`).
    pub fn all(&self) -> u64 {
        if self.names.len() == 64 {
            u64::MAX
        } else {
            (1u64 << self.names.len()) - 1
        }
    }

    /// `LanguageSet.from(names)` for this name type, with the caller's
    /// names kept for [`CallerLanguages::singleton`].
    pub fn from_names(&self, names: &[String]) -> CallerLanguages {
        let mut distinct: Vec<String> = Vec::new();
        for n in names {
            if !distinct.contains(n) {
                distinct.push(n.clone());
            }
        }
        let mut bits = 0;
        let mut unknown = false;
        for n in &distinct {
            match self.bit(n) {
                Some(b) => bits |= b,
                None => unknown = true,
            }
        }
        CallerLanguages {
            set: LanguageSet::from_bits(bits, unknown),
            names: Arc::from(distinct),
        }
    }
}

/// A caller's `LanguageSet` (`LanguageSet.from(Set<String>)`, or a guessed
/// one): the set, and its names for the singleton rule lookup.
#[derive(Debug, Clone)]
pub struct CallerLanguages {
    /// The set as the engine computes with it.
    pub set: LanguageSet,
    /// The distinct names (empty for a guessed `ANY_LANGUAGE`).
    pub names: Arc<[String]>,
}

impl CallerLanguages {
    /// `LanguageSet.isSingleton() ? getAny() : null`.
    pub fn singleton(&self) -> Option<&str> {
        match self.set {
            LanguageSet::Some { .. } if self.names.len() == 1 => Some(&self.names[0]),
            _ => None,
        }
    }
}

impl fmt::Display for LanguageSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LanguageSet::No => f.write_str("NO_LANGUAGES"),
            LanguageSet::Any => f.write_str("ANY_LANGUAGE"),
            LanguageSet::Some { bits, unknown } => write!(f, "Languages({bits:#x}, {unknown})"),
        }
    }
}

/// `String.trim()` of a resource line: every character `<= ' '` off both
/// ends.
fn trim_str(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_algebra() {
        let a = LanguageSet::Some {
            bits: 0b011,
            unknown: true,
        };
        let b = LanguageSet::Some {
            bits: 0b110,
            unknown: false,
        };
        assert_eq!(
            a.restrict_to(b),
            LanguageSet::Some {
                bits: 0b010,
                unknown: false
            }
        );
        assert_eq!(
            LanguageSet::Some {
                bits: 1,
                unknown: false
            }
            .restrict_to(LanguageSet::Some {
                bits: 2,
                unknown: false
            }),
            LanguageSet::No
        );
        assert_eq!(a.restrict_to(LanguageSet::Any), a);
        assert_eq!(a.restrict_to(LanguageSet::No), LanguageSet::No);
        assert_eq!(LanguageSet::Any.restrict_to(b), b);
        assert_eq!(LanguageSet::No.restrict_to(b), LanguageSet::No);
        assert_eq!(
            a.merge(b),
            LanguageSet::Some {
                bits: 0b111,
                unknown: true
            }
        );
        assert_eq!(a.merge(LanguageSet::No), a);
        assert_eq!(a.merge(LanguageSet::Any), LanguageSet::Any);
        assert_eq!(LanguageSet::Any.merge(b), b);
        assert_eq!(LanguageSet::No.merge(b), b);
        assert!(LanguageSet::No.is_empty() && !a.is_empty());
        assert_eq!(LanguageSet::No.to_string(), "NO_LANGUAGES");
        assert_eq!(LanguageSet::Any.to_string(), "ANY_LANGUAGE");
        assert_eq!(b.to_string(), "Languages(0x6, false)");
    }

    #[test]
    fn languages_and_names() {
        let gen = Languages::get(NameType::Generic);
        assert_eq!(gen.names().len(), 19);
        assert!(gen.bit("english").is_some() && gen.bit("klingon").is_none());
        assert_eq!(gen.all().count_ones(), 19);
        let c = gen.from_names(&["english".into(), "english".into()]);
        assert_eq!(c.singleton(), Some("english"));
        let u = gen.from_names(&["klingon".into(), "english".into()]);
        assert_eq!(u.singleton(), None);
        assert!(matches!(u.set, LanguageSet::Some { unknown: true, .. }));
        assert_eq!(gen.from_names(&[]).set, LanguageSet::No);
        assert_eq!(Languages::get(NameType::Sephardic).names().len(), 6);
        for (s, t) in [
            ("ASHKENAZI", NameType::Ashkenazi),
            ("GENERIC", NameType::Generic),
            ("SEPHARDIC", NameType::Sephardic),
        ] {
            assert_eq!(NameType::value_of(s), Some(t));
        }
        assert_eq!(NameType::value_of("generic"), None);
        for (s, t) in [
            ("APPROX", RuleType::Approx),
            ("EXACT", RuleType::Exact),
            ("RULES", RuleType::Rules),
        ] {
            assert_eq!(RuleType::value_of(s), Some(t));
            assert!(!t.name().is_empty());
        }
        assert_eq!(RuleType::value_of("x"), None);
        assert_eq!(trim_str("\t a b \r"), "a b");
        assert_eq!(trim_str("ab"), "ab");
    }
}
