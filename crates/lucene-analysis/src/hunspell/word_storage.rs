//! `WordStorage`: the `.dic` entries, looked up by word and enumerated for
//! suggestions.
//!
//! Lucene packs the entries into a byte array (each entry a delta pointer to
//! its prefix) behind an open hash table keyed by `String.hashCode()`. The
//! port keeps the hash table -- same size, same slot per word, the same
//! newest-first collision chains, so enumeration (`processAllWords`) visits
//! entries in Lucene's order -- and stores each entry's units and forms
//! directly. Differs: the packed byte layout (and its `OFFSET_BITS` size
//! limit) is not reproduced.

use super::flags::FlagEnumerator;
use super::{HunspellError, HIDDEN_FLAG};

/// `WordStorage.MAX_STORED_LENGTH`: lengths at or above it share one length
/// code in the hash table.
const MAX_STORED_LENGTH: usize = 0x20 - 1;

/// `String.hashCode()` over UTF-16 units.
pub(crate) fn java_hash(units: &[u16]) -> i32 {
    units
        .iter()
        .fold(0i32, |h, &u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// One stored word.
#[derive(Debug)]
struct Entry {
    word: Vec<u16>,
    /// `lookupWord`'s forms: flag-set ids, each followed by a morph-data id
    /// when the dictionary has custom morphological data.
    forms: Vec<i32>,
    suggestible: bool,
    /// The previous entry with the same hash (the chain runs newest first).
    next: Option<usize>,
}

/// `WordStorage`.
#[derive(Debug)]
pub(crate) struct WordStorage {
    /// Per slot, the newest entry hashing there.
    hash_table: Vec<Option<usize>>,
    entries: Vec<Entry>,
    max_entry_length: usize,
    has_custom_morph_data: bool,
}

/// A word visited by [`WordStorage::process_all_words`] (`FlyweightEntry`).
pub(crate) struct WordEntry<'a> {
    /// `root()`.
    pub(crate) root: &'a [u16],
    forms: &'a [i32],
    step: usize,
}

impl WordEntry<'_> {
    /// `forms()`: the flag-set ids, without morph-data ids.
    pub(crate) fn forms(&self) -> impl Iterator<Item = i32> + '_ {
        self.forms.iter().step_by(self.step).copied()
    }
}

impl WordStorage {
    /// `lookupWord`: the forms of `word` (non-empty), or `None`.
    pub(crate) fn lookup_word(&self, word: &[u16]) -> Option<&[i32]> {
        let slot = (java_hash(word) % self.hash_table.len() as i32).unsigned_abs() as usize;
        let mut cur = self.hash_table[slot];
        while let Some(i) = cur {
            let e = &self.entries[i];
            if e.word == word {
                return Some(&e.forms);
            }
            cur = e.next;
        }
        None
    }

    /// `processAllWords`: every entry with a length in `min..=max` (Java's
    /// length-code rule: an entry of `MAX_STORED_LENGTH` units or more
    /// passes when `max` reaches that code) and, when `suggestible_only`,
    /// a suggestible homonym, in hash-table order.
    pub(crate) fn process_all_words(
        &self,
        min_length: usize,
        max_length: usize,
        suggestible_only: bool,
        mut processor: impl FnMut(&WordEntry<'_>),
    ) {
        let max_length = self.max_entry_length.min(max_length);
        let step = if self.has_custom_morph_data { 2 } else { 1 };
        for &head in &self.hash_table {
            let mut cur = head;
            while let Some(i) = cur {
                let e = &self.entries[i];
                cur = e.next;
                let len_code = e.word.len().min(MAX_STORED_LENGTH);
                let length_ok = if len_code == MAX_STORED_LENGTH {
                    max_length >= MAX_STORED_LENGTH
                } else {
                    len_code >= min_length && len_code <= max_length
                };
                if (!suggestible_only || e.suggestible) && length_ok && e.word.len() <= max_length {
                    processor(&WordEntry {
                        root: &e.word,
                        forms: &e.forms,
                        step,
                    });
                }
            }
        }
    }
}

/// `WordStorage.Builder`.
pub(crate) struct WordStorageBuilder<'a> {
    has_custom_morph_data: bool,
    hash_table: Vec<Option<usize>>,
    chain_lengths: Vec<u32>,
    entries: Vec<Entry>,
    no_suggest_flags: Vec<u16>,
    flag_enumerator: &'a mut FlagEnumerator,
    group: Vec<Vec<u16>>,
    morph_data_ids: Vec<i32>,
    current_entry: Option<Vec<u16>>,
    word_count: usize,
    actual_words: usize,
    max_entry_length: usize,
}

impl<'a> WordStorageBuilder<'a> {
    /// `new WordStorage.Builder(wordCount, hashFactor, ...)`.
    pub(crate) fn new(
        word_count: usize,
        hash_factor: f64,
        has_custom_morph_data: bool,
        flag_enumerator: &'a mut FlagEnumerator,
        no_suggest_flags: Vec<u16>,
    ) -> Self {
        let size = (word_count as f64 * hash_factor) as usize;
        WordStorageBuilder {
            has_custom_morph_data,
            hash_table: vec![None; size],
            chain_lengths: vec![0; size],
            entries: Vec::new(),
            no_suggest_flags,
            flag_enumerator,
            group: Vec::new(),
            morph_data_ids: Vec::new(),
            current_entry: None,
            word_count,
            actual_words: 0,
            max_entry_length: 0,
        }
    }

    /// `Builder.add`: entries arrive sorted by `String.compareTo`.
    pub(crate) fn add(
        &mut self,
        entry: &[u16],
        flags: Vec<u16>,
        morph_data_id: i32,
    ) -> Result<(), HunspellError> {
        self.max_entry_length = self.max_entry_length.max(entry.len());
        if self.current_entry.as_deref() != Some(entry) {
            if let Some(current) = &self.current_entry {
                if entry < current.as_slice() {
                    return Err(HunspellError::IllegalArgument(format!(
                        "out of order: {} < {}",
                        String::from_utf16_lossy(entry),
                        String::from_utf16_lossy(current)
                    )));
                }
                self.flush_group()?;
            }
            self.current_entry = Some(entry.to_vec());
        }
        self.group.push(flags);
        if self.has_custom_morph_data {
            self.morph_data_ids.push(morph_data_id);
        }
        Ok(())
    }

    /// `Builder.flushGroup`.
    fn flush_group(&mut self) -> Result<(), HunspellError> {
        self.actual_words += 1;
        if self.actual_words > self.word_count {
            return Err(HunspellError::IllegalState(
                "Don't add more words than wordCount!".into(),
            ));
        }
        let word = self.current_entry.clone().unwrap_or_default();
        let has_non_hidden = self.group.iter().any(|f| !f.contains(&HIDDEN_FLAG));
        let is_suggestible = self
            .group
            .iter()
            .any(|f| !f.iter().any(|x| self.no_suggest_flags.contains(x)));
        let mut forms = Vec::new();
        let group = std::mem::take(&mut self.group);
        let group_len = group.len();
        for (i, mut flags) in group.into_iter().enumerate() {
            if has_non_hidden && group_len > 1 && flags.contains(&HIDDEN_FLAG) {
                continue;
            }
            forms.push(self.flag_enumerator.add(&mut flags)?);
            if self.has_custom_morph_data {
                forms.push(self.morph_data_ids[i]);
            }
        }
        self.morph_data_ids.clear();
        let slot = (java_hash(&word) % self.hash_table.len() as i32).unsigned_abs() as usize;
        self.chain_lengths[slot] += 1;
        if self.chain_lengths[slot] > 20 {
            return Err(HunspellError::IllegalState(
                "Too many collisions. Try a larger Dictionary#hashFactor (now 1.0). If this doesn't help, please report this to dev@lucene.apache.org".into(),
            ));
        }
        let next = self.hash_table[slot];
        self.entries.push(Entry {
            word,
            forms,
            suggestible: is_suggestible,
            next,
        });
        self.hash_table[slot] = Some(self.entries.len() - 1);
        Ok(())
    }

    /// `new WordStorage(builder)`.
    pub(crate) fn finish(mut self) -> Result<WordStorage, HunspellError> {
        if !self.hash_table.is_empty() && !self.group.is_empty() {
            self.flush_group()?;
        }
        let hash_table = if self.hash_table.is_empty() {
            vec![None]
        } else {
            self.hash_table
        };
        Ok(WordStorage {
            hash_table,
            entries: self.entries,
            max_entry_length: self.max_entry_length,
            has_custom_morph_data: self.has_custom_morph_data,
        })
    }
}
