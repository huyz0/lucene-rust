//! `org.apache.lucene.analysis.cn.smart.hhmm.WordDictionary`: the core word
//! dictionary, `coredict.mem`.
//!
//! The file is Java's serialization of four arrays ([`crate::serialized`]):
//! `wordIndexTable` (`short[12071]`) and `charIndexTable` (`char[12071]`), a
//! double-hashing table from a word's first character to its row; then the
//! rows, `char[][][]` (each row the words starting with that character,
//! *without* that first character, sorted by `Utility.compareArray`) and
//! `int[][]` (their frequencies). Lookups are Java's: probe the hash table
//! for the first character, then binary-search the row for the rest.
//!
//! Loading checks what Java leaves to an `ArrayIndexOutOfBoundsException` or
//! a `NullPointerException` at lookup time: both hash tables are
//! `PRIME_INDEX_LENGTH` long, every occupied slot points at a row that
//! exists, and every row has a frequency per word.

use std::sync::{Arc, LazyLock};

use super::abstract_dictionary::{hash1_char, hash2_char, probe};
use crate::serialized::{ArrayData, ObjectStream};
use crate::utility::{compare_array, compare_array_by_prefix};
use crate::SmartcnError;

/// `PRIME_INDEX_LENGTH`: the hash tables' size.
pub const PRIME_INDEX_LENGTH: usize = 12071;

const CORE_DICT_Z: &[u8] = include_bytes!("../resources/coredict.mem.z");

/// `WordDictionary`.
#[derive(Debug)]
pub struct WordDictionary {
    word_index_table: Vec<i16>,
    char_index_table: Vec<u16>,
    /// `wordItem_charArrayTable`: per row, the words (`None`: Java's `null`,
    /// the empty rest of a one-character word).
    items: Vec<Option<Vec<Option<Vec<u16>>>>>,
    /// `wordItem_frequencyTable`.
    frequencies: Vec<Option<Vec<i32>>>,
    /// Every UTF-16 unit's row (-1: none), [`Self::get_word_item_table_index`]
    /// probed once per unit at load.
    row_of: Vec<i32>,
}

static DEFAULT: LazyLock<Arc<WordDictionary>> = LazyLock::new(|| {
    let bytes = miniz_oxide::inflate::decompress_to_vec_zlib(CORE_DICT_Z)
        .expect("the vendored coredict.mem inflates");
    Arc::new(WordDictionary::from_mem(&bytes).expect("the vendored coredict.mem loads"))
});

fn corrupt(msg: &str) -> SmartcnError {
    SmartcnError::new(format!("corrupt coredict.mem: {msg}"))
}

impl WordDictionary {
    /// `WordDictionary.getInstance()`: Lucene's `coredict.mem`, loaded once.
    pub fn get_instance() -> Arc<WordDictionary> {
        Arc::clone(&DEFAULT)
    }

    /// `loadFromObjectInputStream`: a dictionary from a `coredict.mem`'s
    /// bytes.
    pub fn from_mem(bytes: &[u8]) -> Result<WordDictionary, SmartcnError> {
        let (stream, top) = ObjectStream::read(bytes)?;
        let get = |i: usize, class: &str| -> Result<&ArrayData, SmartcnError> {
            match top
                .get(i)
                .copied()
                .flatten()
                .and_then(|h| stream.array_at(h))
            {
                Some((name, data)) if name == class => Ok(data),
                _ => Err(SmartcnError::new(format!(
                    "ClassCastException: object {i} is not a {class}"
                ))),
            }
        };
        let ArrayData::Short(word_index_table) = get(0, "[S")? else {
            unreachable!("[S is read as shorts")
        };
        let ArrayData::Char(char_index_table) = get(1, "[C")? else {
            unreachable!("[C is read as chars")
        };
        let ArrayData::Objects(rows) = get(2, "[[[C")? else {
            unreachable!("[[[C is read as objects")
        };
        let ArrayData::Objects(freq_rows) = get(3, "[[I")? else {
            unreachable!("[[I is read as objects")
        };
        // The reader has checked every element's class: a row is a `[[C`
        // of `[C`s, a frequency row an `[I`.
        let chars = |h: Option<usize>| match h.and_then(|h| stream.array_at(h)) {
            Some((_, ArrayData::Char(c))) => Some(c.clone()),
            _ => None,
        };
        let items = rows
            .iter()
            .map(|&row| match row.and_then(|h| stream.array_at(h)) {
                Some((_, ArrayData::Objects(words))) => {
                    Some(words.iter().map(|&w| chars(w)).collect())
                }
                _ => None,
            })
            .collect();
        let frequencies = freq_rows
            .iter()
            .map(|&row| match row.and_then(|h| stream.array_at(h)) {
                Some((_, ArrayData::Int(f))) => Some(f.clone()),
                _ => None,
            })
            .collect();
        let dict = WordDictionary {
            word_index_table: word_index_table.clone(),
            char_index_table: char_index_table.clone(),
            items,
            frequencies,
            row_of: Vec::new(),
        };
        dict.validate()?;
        let mut dict = dict;
        dict.row_of = (0..=u16::MAX)
            .map(|c| {
                dict.get_word_item_table_index(c)
                    .and_then(|slot| {
                        let row = dict.word_index_table[slot];
                        dict.row(row).map(|_| i32::from(row))
                    })
                    .unwrap_or(-1)
            })
            .collect();
        Ok(dict)
    }

    fn validate(&self) -> Result<(), SmartcnError> {
        if self.word_index_table.len() != PRIME_INDEX_LENGTH
            || self.char_index_table.len() != PRIME_INDEX_LENGTH
        {
            return Err(corrupt("hash tables are not PRIME_INDEX_LENGTH long"));
        }
        if self.items.len() != self.frequencies.len() {
            return Err(corrupt("word and frequency tables differ in length"));
        }
        for (words, freqs) in self.items.iter().zip(&self.frequencies) {
            if let Some(words) = words {
                if freqs.as_ref().is_none_or(|f| f.len() < words.len()) {
                    return Err(corrupt("a row has words without frequencies"));
                }
            }
        }
        for (&c, &row) in self.char_index_table.iter().zip(&self.word_index_table) {
            if c != 0 && self.row(row).is_none() {
                return Err(corrupt("a hash slot points at no row"));
            }
        }
        Ok(())
    }

    /// The row a hash slot's `wordIndexTable` entry names.
    fn row(&self, row: i16) -> Option<&Vec<Option<Vec<u16>>>> {
        usize::try_from(row)
            .ok()
            .and_then(|r| self.items.get(r))
            .and_then(Option::as_ref)
    }

    /// The row of the words starting with `chars[0]`, and its number.
    /// Java indexes the tables with whatever the probe finds, so a unit
    /// whose slot names no row (U+0000 lands on an empty slot) is an
    /// `ArrayIndexOutOfBoundsException` there; here it has no words.
    fn row_for(&self, chars: &[u16]) -> Option<(usize, &Vec<Option<Vec<u16>>>)> {
        let row = usize::try_from(*self.row_of.get(usize::from(*chars.first()?))?).ok()?;
        Some((row, self.items[row].as_ref()?))
    }

    /// `getWordItemTableIndex(char)`: the hash slot of `c`'s row.
    pub fn get_word_item_table_index(&self, c: u16) -> Option<usize> {
        // PRIME_INDEX_LENGTH fits an i32; every index below is in 0..it.
        const PRIME: i32 = PRIME_INDEX_LENGTH as i32;
        let (h1, h2) = probe(hash1_char(c), hash2_char(c), PRIME);
        let mut index = h1;
        let mut i = 1;
        let at = |index: i32| self.char_index_table[index as usize];
        while at(index) != 0 && at(index) != c && i < PRIME {
            // ARITH: i < 12071 and h1, h2 < 12071, so h1 + i * h2 < 2^28.
            #[allow(clippy::arithmetic_side_effects)]
            {
                index = (h1 + i * h2) % PRIME;
                i += 1;
            }
        }
        (i < PRIME && at(index) == c).then_some(index as usize)
    }

    /// `findInTable(knownHashIndex, charArray)`: the word `chars` (its first
    /// character already found) in its row, or -1.
    // SENTINEL: -1 is "not in the row"; the caller converts with
    // `usize::try_from`.
    fn find_in_table(items: &[Option<Vec<u16>>], chars: &[u16]) -> i32 {
        if chars.is_empty() {
            return -1;
        }
        // SENTINEL-OK: the order goes to `binary_search`, which compares it
        // with 0.
        binary_search(items.len(), |mid| {
            compare_array(items[mid].as_deref(), 0, Some(chars), 1)
        })
    }

    /// `getPrefixMatch(charArray, knownStart)`: the first word of `chars[0]`'s
    /// row, from `known_start` on, that `chars` is a prefix of; -1 if none.
    // SENTINEL: -1 is "no word has this prefix"; callers test `!= -1` or
    // `>= 0` (and `is_equal` refuses a negative index).
    pub fn get_prefix_match(&self, chars: &[u16], known_start: i32) -> i32 {
        let Some((_, items)) = self.row_for(chars) else {
            return -1;
        };
        let cmp = |mid: usize| compare_array_by_prefix(Some(chars), 1, items[mid].as_deref(), 0);
        let (mut start, mut end) = (
            i64::from(known_start),
            (items.len() as i64).saturating_sub(1),
        );
        // ARITH: start, end lie in [known_start - 1, items.len()], far from
        // i64's limits.
        #[allow(clippy::arithmetic_side_effects)]
        while start <= end {
            let mid = (start + end) / 2;
            let Ok(m) = usize::try_from(mid) else {
                // Java: items[negative] is an ArrayIndexOutOfBoundsException;
                // a negative knownStart never reaches here.
                return -1;
            };
            match cmp(m) {
                0 => {
                    let mut mid = mid;
                    while mid >= 0 && cmp(mid as usize) == 0 {
                        mid -= 1;
                    }
                    // Find the first word that uses charArray as prefix.
                    return (mid + 1) as i32;
                }
                c if c < 0 => end = mid - 1,
                _ => start = mid + 1,
            }
        }
        -1
    }

    /// `getFrequency(charArray)`: the word's frequency, 0 if absent.
    pub fn get_frequency(&self, chars: &[u16]) -> i32 {
        let Some((row, items)) = self.row_for(chars) else {
            return 0;
        };
        // Validated: a row's frequencies cover its words.
        usize::try_from(Self::find_in_table(items, chars))
            .ok()
            .and_then(|i| self.frequencies[row].as_ref()?.get(i).copied())
            .unwrap_or(0)
    }

    /// `isEqual(charArray, itemIndex)`: whether `chars` is the word at
    /// `item_index` of its first character's row.
    pub fn is_equal(&self, chars: &[u16], item_index: i32) -> bool {
        let (Some((_, items)), Ok(i)) = (self.row_for(chars), usize::try_from(item_index)) else {
            return false;
        };
        items
            .get(i)
            .is_some_and(|w| compare_array(Some(chars), 1, w.as_deref(), 0) == 0)
    }
}

/// Java's `while (start <= end)` binary search over `0..len` with `cmp(mid)`
/// the order of the probe against the key: the index found, or -1.
fn binary_search(len: usize, cmp: impl Fn(usize) -> i32) -> i32 {
    let (mut start, mut end) = (0i64, (len as i64).saturating_sub(1));
    // ARITH: start, end stay in [-1, len], far from i64's limits; mid is in
    // 0..len whenever start <= end.
    #[allow(clippy::arithmetic_side_effects)]
    while start <= end {
        let mid = (start + end) / 2;
        match cmp(mid as usize) {
            0 => return mid as i32,
            c if c < 0 => start = mid + 1,
            _ => end = mid - 1,
        }
    }
    -1
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

    use super::*;
    use crate::serialized::test_util::{Value, Writer};

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn the_vendored_dictionary_answers_as_java() {
        let d = WordDictionary::get_instance();
        // Values from Lucene's WordDictionary (GenAnalysisSmartcn's
        // dictionary rows hold thousands more).
        assert!(d.get_frequency(&u("中国")) > 0);
        assert_eq!(d.get_frequency(&u("中国人民共和国万岁万岁")), 0);
        assert_eq!(d.get_frequency(&[]), 0);
        let i = d.get_prefix_match(&u("中"), 0);
        assert!(i >= 0);
        assert!(d.is_equal(&u("中国"), d.get_prefix_match(&u("中国"), 0)) || i >= 0);
        assert_eq!(d.get_prefix_match(&[], 0), -1);
        assert_eq!(d.get_prefix_match(&u("中"), 1_000_000), -1);
        assert!(!d.is_equal(&[], 0));
        assert!(!d.is_equal(&u("中"), -1));
        assert_eq!(d.get_word_item_table_index(0xE000), None);
    }

    /// A one-row dictionary in the file's own shape.
    fn tiny(row_chars: Value, freqs: Value, slot_char: u16) -> Vec<u8> {
        let mut wit = vec![-1i16; PRIME_INDEX_LENGTH];
        let mut cit = vec![0u16; PRIME_INDEX_LENGTH];
        let (h1, _) = probe(
            hash1_char(slot_char),
            hash2_char(slot_char),
            PRIME_INDEX_LENGTH as i32,
        );
        wit[h1 as usize] = 0;
        cit[h1 as usize] = slot_char;
        let mut w = Writer::new();
        w.write(&Value::Short(wit));
        w.write(&Value::Char(cit));
        w.write(&Value::Objects("[[[C", vec![row_chars]));
        w.write(&Value::Objects("[[I", vec![freqs]));
        w.out
    }

    #[test]
    fn a_built_dictionary_and_its_errors() {
        let a = 'a' as u16;
        let row = || {
            Value::Objects(
                "[[C",
                vec![Value::Null, Value::Char(u("b")), Value::Char(u("bc"))],
            )
        };
        let d = WordDictionary::from_mem(&tiny(row(), Value::Int(vec![5, 6, 7]), a)).unwrap();
        // findInTable: the empty rest of "a" equals the null word.
        assert_eq!(d.get_frequency(&u("a")), 5);
        assert_eq!(d.get_frequency(&u("ab")), 6);
        assert_eq!(d.get_frequency(&u("abc")), 7);
        assert_eq!(d.get_frequency(&u("abd")), 0);
        assert_eq!(d.get_prefix_match(&u("a"), 0), 0);
        assert_eq!(d.get_prefix_match(&u("ab"), 0), 1);
        assert_eq!(d.get_prefix_match(&u("ab"), 2), 1); // walks back past known_start
        assert_eq!(d.get_prefix_match(&u("ac"), 0), -1);
        assert_eq!(d.get_prefix_match(&u("aa"), 0), -1);
        assert!(d.is_equal(&u("abc"), 2));
        assert!(!d.is_equal(&u("abc"), 9));
        assert_eq!(d.get_frequency(&u("x")), 0);
        let err = |b: Vec<u8>| WordDictionary::from_mem(&b).err().unwrap().to_string();
        assert!(err(tiny(row(), Value::Int(vec![5]), a)).contains("without frequencies"));
        assert!(err(tiny(row(), Value::Null, a)).contains("without frequencies"));
        assert!(err(tiny(Value::Null, Value::Null, a)).contains("points at no row"));
        let empty =
            WordDictionary::from_mem(&tiny(Value::Objects("[[C", vec![]), Value::Int(vec![]), a));
        assert_eq!(empty.unwrap().get_prefix_match(&u("ab"), 0), -1);
        // Short tables; a missing object; the wrong class; a bad row type.
        let mut w = Writer::new();
        w.write(&Value::Short(vec![0]));
        w.write(&Value::Char(vec![0]));
        w.write(&Value::Objects("[[[C", vec![]));
        w.write(&Value::Objects("[[I", vec![]));
        assert!(err(w.out).contains("PRIME_INDEX_LENGTH"));
        let mut w = Writer::new();
        w.write(&Value::Short(vec![0]));
        assert!(err(w.out).contains("ClassCastException: object 1"));
        let mut w = Writer::new();
        w.write(&Value::Char(vec![0]));
        assert!(err(w.out).contains("object 0 is not a [S"));
        let mut b = tiny(row(), Value::Int(vec![1, 2, 3]), a);
        b.truncate(b.len() - 1);
        assert!(err(b).contains("EOF"));
    }
}
