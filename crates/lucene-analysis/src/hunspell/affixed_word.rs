//! `AffixedWord`: one analysis of a word -- its dictionary entry and the
//! prefixes and suffixes removed to reach it.

use super::dictionary::{DictEntry, Dictionary, AFFIX_FLAG};

/// `AffixedWord.Affix`: an affix rule by its flag and internal id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Affix {
    pub(crate) affix_id: i32,
    flag: String,
}

impl Affix {
    pub(crate) fn new(dictionary: &Dictionary, affix_id: i32) -> Self {
        let encoded = dictionary.affix_data(affix_id, AFFIX_FLAG);
        Affix {
            affix_id,
            flag: dictionary.flag_parsing.print_flag(encoded),
        }
    }

    /// `getFlag()`.
    pub fn flag(&self) -> &str {
        &self.flag
    }
}

impl std::fmt::Display for Affix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(id={})", self.flag, self.affix_id)
    }
}

/// `AffixedWord`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffixedWord {
    /// `getWord()`.
    pub word: String,
    /// `getDictEntry()`.
    pub entry: DictEntry,
    /// `getPrefixes()`: outer first.
    pub prefixes: Vec<Affix>,
    /// `getSuffixes()`: outer first.
    pub suffixes: Vec<Affix>,
}

fn list(items: &[Affix]) -> String {
    let parts: Vec<String> = items.iter().map(ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

impl std::fmt::Display for AffixedWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AffixedWord[word={}, entry={}, prefixes={}, suffixes={}]",
            self.word,
            self.entry,
            list(&self.prefixes),
            list(&self.suffixes)
        )
    }
}
