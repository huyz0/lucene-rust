//! ICU4J 77.1's dictionary break engines (`com.ibm.icu.impl.breakiter`):
//! `DictionaryBreakEngine` (with `PossibleWord` and `DequeI`), the
//! Thai, Lao, Khmer and Burmese engines (one [`SeaEngine`] with each
//! script's sets and dictionary; Thai alone also attaches the PAIYANNOI and
//! MAIYAMOK suffixes, and its candidate search leaves the outer loop with a
//! labelled break where the others finish one more `backUp`), the
//! Chinese/Japanese and Korean `CjkBreakEngine`, `UnhandledBreakEngine`,
//! `DictionaryData` and the bytes/chars `DictionaryMatcher`s, and the
//! process-wide engine list `RuleBasedBreakIterator.getLanguageBreakEngine`
//! keeps.
//!
//! The dictionaries are ICU 77.1's (`thaidict`, `laodict`, `khmerdict`,
//! `burmesedict`, `cjdict`; vendored zlib-compressed; their third-party
//! terms are in `LICENSE`). Neither ICU4J's LSTM engines (the jar carries
//! no LSTM model, so ICU4J falls back to these) nor the ML phrase breaker
//! (off unless a system property says otherwise) is ported; phrase
//! breaking (`isPhraseBreaking`, only for the `ja@lw=phrase` locales) is
//! not ported either -- the word iterators Lucene builds never set it.
//!
//! The engine list is global, as in Java: the first dictionary character
//! of a script with no engine (Tai Tham, or a Common-script prolonged sound
//! mark before any Han or kana was ever segmented) makes the unhandled
//! engine claim that whole script from then on, process-wide.

use std::sync::{Arc, Mutex, OnceLock};

use crate::icu4j::binary::read_header;
use crate::icu4j::char_iter::{CharIter, DONE32};
use crate::icu4j::normalizer2::{Mode, Normalizer2, QuickCheck};
use crate::icu4j::tries::{BytesTrie, CharsTrie, TrieResult};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::icu4j::uprops;
use crate::IcuError;

/// `DictionaryBreakEngine.DequeI`: an int deque over an array, read as Java
/// reads it -- `peek` on an empty deque returns the slot below the base
/// (`data[3]` after `removeAllElements`), whatever was last `offer`ed
/// there, as ICU4J does with assertions off.
#[derive(Debug, Clone)]
pub struct DequeI {
    data: Vec<i32>,
    last_idx: i32,
    first_idx: i32,
}

impl Default for DequeI {
    fn default() -> Self {
        DequeI {
            data: vec![0; 50],
            last_idx: 4,
            first_idx: 4,
        }
    }
}

// ARITH: (the whole impl) indices move by one per call and stay within the
// grown array for any sequence the engines make; a read outside it is 0.
#[allow(clippy::arithmetic_side_effects)]
impl DequeI {
    fn get(&self, i: i32) -> i32 {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.data.get(i))
            .copied()
            .unwrap_or(0)
    }

    fn set(&mut self, i: i32, v: i32) {
        if let Some(slot) = usize::try_from(i).ok().and_then(|i| self.data.get_mut(i)) {
            *slot = v;
        }
    }

    /// `size()`.
    pub fn size(&self) -> i32 {
        self.first_idx - self.last_idx
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// `offer(v)`: below the base.
    pub fn offer(&mut self, v: i32) {
        self.last_idx -= 1;
        self.set(self.last_idx, v);
    }

    /// `push(v)`.
    pub fn push(&mut self, v: i32) {
        if self.first_idx >= self.data.len() as i32 {
            let n = self.data.len() * 2;
            self.data.resize(n, 0);
        }
        self.set(self.first_idx, v);
        self.first_idx += 1;
    }

    /// `pop()`.
    pub fn pop(&mut self) -> i32 {
        self.first_idx -= 1;
        self.get(self.first_idx)
    }

    /// `peek()`.
    pub fn peek(&self) -> i32 {
        self.get(self.first_idx - 1)
    }

    /// `elementAt(i)`.
    pub fn element_at(&self, i: i32) -> i32 {
        self.get(self.last_idx + i)
    }

    /// `removeAllElements()`.
    pub fn remove_all_elements(&mut self) {
        self.last_idx = 4;
        self.first_idx = 4;
    }
}

// --- dictionaries -----------------------------------------------------------

const TRIE_TYPE_BYTES: i32 = 0;
const TRIE_TYPE_UCHARS: i32 = 1;
const TRIE_TYPE_MASK: i32 = 7;
const TRANSFORM_TYPE_MASK: i32 = 0x7f00_0000;
const TRANSFORM_TYPE_OFFSET: i32 = 0x0100_0000;
const TRANSFORM_OFFSET_MASK: i32 = 0x1f_ffff;

/// `DictionaryMatcher`: a bytes trie with an offset transform (the
/// Southeast Asian dictionaries) or a chars trie with values (`cjdict`).
#[derive(Debug)]
pub enum DictionaryMatcher {
    /// `BytesDictionaryMatcher`.
    Bytes { trie: Vec<u8>, transform: i32 },
    /// `CharsDictionaryMatcher`.
    Chars { trie: Vec<u16> },
}

impl DictionaryMatcher {
    /// `DictionaryData.loadDictionaryFor`'s reading of a `.dict` file.
    // ARITH: offsets are i32s read off the file, compared and subtracted
    // only after checking they lie within it.
    pub fn load(bytes: &[u8]) -> Result<DictionaryMatcher, IcuError> {
        let (mut r, _) = read_header(bytes, 0x4469_6374, |_| true)?;
        let base = r.position();
        let ix = r.i32s(8)?;
        let offset = ix[0];
        let total = ix[3];
        if offset < 32 || total < offset {
            return Err(IcuError::new("dictionary data corrupt"));
        }
        let start =
            usize::try_from(offset).map_err(|_| IcuError::new("dictionary data corrupt"))?;
        r.seek(base.saturating_add(start))?;
        let size = usize::try_from(total.saturating_sub(offset))
            .map_err(|_| IcuError::new("dictionary data corrupt"))?;
        match ix[4] & TRIE_TYPE_MASK {
            TRIE_TYPE_BYTES => {
                let transform = ix[5];
                if transform & TRANSFORM_TYPE_MASK != TRANSFORM_TYPE_OFFSET {
                    return Err(IcuError::new("dictionary transform unsupported"));
                }
                Ok(DictionaryMatcher::Bytes {
                    trie: r.take(size)?.to_vec(),
                    transform,
                })
            }
            TRIE_TYPE_UCHARS => Ok(DictionaryMatcher::Chars {
                trie: r.u16s(size / 2)?,
            }),
            _ => Err(IcuError::new("dictionary trie type unsupported")),
        }
    }

    /// `BytesDictionaryMatcher.transform(c)`.
    // SENTINEL: `-1` = a code point outside the dictionary's 256-character
    // window; the caller passes it on as Java does (it then matches nothing).
    // ARITH: c is a code point and the offset is masked to 21 bits.
    #[allow(clippy::arithmetic_side_effects)]
    fn transform(transform: i32, c: i32) -> i32 {
        if c == 0x200d {
            return 0xff;
        } else if c == 0x200c {
            return 0xfe;
        }
        let delta = c - (transform & TRANSFORM_OFFSET_MASK);
        if !(0..=0xfd).contains(&delta) {
            return -1;
        }
        delta
    }

    /// `matches(text, maxLength, lengths, count, limit, values)`: the
    /// dictionary words starting at the text's index, at most `limit`, by
    /// length in code points; returns the code points examined. Advances
    /// the text.
    // ARITH: num_chars and count are bounded by max_length and limit.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn matches(
        &self,
        text: &mut CharIter<'_>,
        max_length: i32,
        lengths: &mut [i32],
        count_out: &mut i32,
        limit: i32,
        mut values: Option<&mut [i32]>,
    ) -> i32 {
        let mut c = text.next_code_point();
        if c == -1 {
            return 0;
        }
        let mut num_chars = 1;
        let mut count = 0;
        match self {
            DictionaryMatcher::Bytes { trie, transform } => {
                let mut bt = BytesTrie::new(trie, 0);
                let mut result = bt.first(Self::transform(*transform, c));
                loop {
                    if result.has_value() {
                        if count < limit {
                            if let Some(v) = values.as_deref_mut() {
                                set(v, count, bt.get_value());
                            }
                            set(lengths, count, num_chars);
                            count += 1;
                        }
                        if result == TrieResult::FinalValue {
                            break;
                        }
                    } else if result == TrieResult::NoMatch {
                        break;
                    }
                    if num_chars >= max_length {
                        break;
                    }
                    c = text.next_code_point();
                    if c == -1 {
                        break;
                    }
                    num_chars += 1;
                    result = bt.next(Self::transform(*transform, c));
                }
            }
            DictionaryMatcher::Chars { trie } => {
                let mut ct = CharsTrie::new(trie, 0);
                let mut result = ct.first_for_code_point(c);
                loop {
                    if result.has_value() {
                        if count < limit {
                            if let Some(v) = values.as_deref_mut() {
                                set(v, count, ct.get_value());
                            }
                            set(lengths, count, num_chars);
                            count += 1;
                        }
                        if result == TrieResult::FinalValue {
                            break;
                        }
                    } else if result == TrieResult::NoMatch {
                        break;
                    }
                    if num_chars >= max_length {
                        break;
                    }
                    c = text.next_code_point();
                    if c == -1 {
                        break;
                    }
                    num_chars += 1;
                    result = ct.next_for_code_point(c);
                }
            }
        }
        *count_out = count;
        num_chars
    }
}

fn set(a: &mut [i32], i: i32, v: i32) {
    if let Some(slot) = usize::try_from(i).ok().and_then(|i| a.get_mut(i)) {
        *slot = v;
    }
}

fn get(a: &[i32], i: i32) -> i32 {
    usize::try_from(i)
        .ok()
        .and_then(|i| a.get(i))
        .copied()
        .unwrap_or(0)
}

const THAI_DICT: &[u8] = include_bytes!("../resources/thaidict.dict.z");
const LAO_DICT: &[u8] = include_bytes!("../resources/laodict.dict.z");
const KHMER_DICT: &[u8] = include_bytes!("../resources/khmerdict.dict.z");
const BURMESE_DICT: &[u8] = include_bytes!("../resources/burmesedict.dict.z");
const CJ_DICT: &[u8] = include_bytes!("../resources/cjdict.dict.z");

fn load_vendored(z: &[u8]) -> DictionaryMatcher {
    let bytes =
        miniz_oxide::inflate::decompress_to_vec_zlib(z).expect("vendored dictionary inflates");
    DictionaryMatcher::load(&bytes).expect("vendored dictionary loads")
}

// --- PossibleWord -------------------------------------------------------------

const POSSIBLE_WORD_LIST_MAX: usize = 20;

/// `DictionaryBreakEngine.PossibleWord`.
#[derive(Debug, Clone)]
struct PossibleWord {
    lengths: [i32; POSSIBLE_WORD_LIST_MAX],
    count: i32,
    prefix: i32,
    offset: i32,
    mark: i32,
    current: i32,
}

impl Default for PossibleWord {
    fn default() -> Self {
        PossibleWord {
            lengths: [0; POSSIBLE_WORD_LIST_MAX],
            count: 0,
            prefix: 0,
            offset: -1,
            mark: 0,
            current: 0,
        }
    }
}

// ARITH: (the whole impl) offsets are text indices; lengths are at most
// the text length; count is at most POSSIBLE_WORD_LIST_MAX.
#[allow(clippy::arithmetic_side_effects)]
impl PossibleWord {
    /// `candidates(fIter, dict, rangeEnd)`.
    fn candidates(
        &mut self,
        it: &mut CharIter<'_>,
        dict: &DictionaryMatcher,
        range_end: i32,
    ) -> i32 {
        let start = it.index() as i32;
        if start != self.offset {
            self.offset = start;
            self.prefix = dict.matches(
                it,
                range_end - start,
                &mut self.lengths,
                &mut self.count,
                POSSIBLE_WORD_LIST_MAX as i32,
                None,
            );
            if self.count <= 0 {
                it.set_index(start as usize);
            }
        }
        if self.count > 0 {
            it.set_index((start + self.lengths[(self.count - 1) as usize]) as usize);
        }
        self.current = self.count - 1;
        self.mark = self.current;
        self.count
    }

    /// `acceptMarked(fIter)`.
    fn accept_marked(&self, it: &mut CharIter<'_>) -> i32 {
        let len = self.lengths.get(self.mark as usize).copied().unwrap_or(0);
        it.set_index((self.offset + len) as usize);
        len
    }

    /// `backUp(fIter)`.
    fn back_up(&mut self, it: &mut CharIter<'_>) -> bool {
        if self.current > 0 {
            self.current -= 1;
            it.set_index((self.offset + self.lengths[self.current as usize]) as usize);
            return true;
        }
        false
    }

    /// `longestPrefix()`.
    fn longest_prefix(&self) -> i32 {
        self.prefix
    }

    /// `markCurrent()`.
    fn mark_current(&mut self) {
        self.mark = self.current;
    }
}

// --- the Southeast Asian engines ------------------------------------------

const THAI_PAIYANNOI: i32 = 0x0e2f;
const THAI_MAIYAMOK: i32 = 0x0e46;
const ROOT_COMBINE_THRESHOLD: i32 = 3;
const PREFIX_COMBINE_THRESHOLD: i32 = 3;
const LOOKAHEAD: i32 = 3;

/// `ThaiBreakEngine`, `LaoBreakEngine`, `KhmerBreakEngine`,
/// `BurmeseBreakEngine`.
#[derive(Debug)]
pub struct SeaEngine {
    script: i32,
    thai: bool,
    min_span: i32,
    /// `fSet`: the characters the engine segments.
    set: UnicodeSet,
    end_word_set: UnicodeSet,
    begin_word_set: UnicodeSet,
    mark_set: UnicodeSet,
    suffix_set: UnicodeSet,
    dictionary: DictionaryMatcher,
}

fn uset(p: &str) -> UnicodeSet {
    UnicodeSet::from_pattern(p).expect("a fixed engine set parses")
}

impl SeaEngine {
    /// The engine of `script` (Thai 38, Lao 24, Khmer 23, Myanmar 28).
    fn new(script: i32) -> Option<SeaEngine> {
        let (code, dict, begin, min_span): (&str, &[u8], &[(u32, u32)], i32) = match script {
            38 => ("Thai", THAI_DICT, &[(0x0e01, 0x0e2e), (0x0e40, 0x0e44)], 4),
            24 => (
                "Laoo",
                LAO_DICT,
                &[(0x0e81, 0x0eae), (0x0ec0, 0x0ec4), (0x0edc, 0x0edd)],
                2,
            ),
            23 => ("Khmer", KHMER_DICT, &[(0x1780, 0x17b3)], 4),
            28 => ("Mymr", BURMESE_DICT, &[(0x1000, 0x102a)], 2),
            _ => return None,
        };
        let word_set = uset(&format!("[[:{code}:]&[:LineBreak=SA:]]"));
        let mut mark_set = uset(&format!("[[:{code}:]&[:LineBreak=SA:]&[:M:]]"));
        mark_set.add(0x20);
        let mut begin_word_set = UnicodeSet::new();
        for &(a, b) in begin {
            begin_word_set.add_range(a, b);
        }
        let mut end_word_set = word_set.clone();
        match script {
            38 => {
                end_word_set.remove_all(&UnicodeSet::from_range(0x0e31, 0x0e31));
                end_word_set.remove_all(&UnicodeSet::from_range(0x0e40, 0x0e44));
            }
            24 => end_word_set.remove_all(&UnicodeSet::from_range(0x0ec0, 0x0ec4)),
            23 => end_word_set.remove_all(&UnicodeSet::from_range(0x17d2, 0x17d2)),
            _ => {}
        }
        let mut suffix_set = UnicodeSet::new();
        if script == 38 {
            suffix_set.add(THAI_PAIYANNOI as u32);
            suffix_set.add(THAI_MAIYAMOK as u32);
        }
        Some(SeaEngine {
            script,
            thai: script == 38,
            min_span,
            set: word_set,
            end_word_set,
            begin_word_set,
            mark_set,
            suffix_set,
            dictionary: load_vendored(dict),
        })
    }

    /// `divideUpDictionaryRange(fIter, rangeStart, rangeEnd, foundBreaks)`.
    // ARITH: positions are text indices (< i32::MAX: a tokenizer buffer);
    // word counts are bounded by the range length.
    #[allow(clippy::arithmetic_side_effects)]
    fn divide_up_dictionary_range(
        &self,
        it: &mut CharIter<'_>,
        range_start: i32,
        range_end: i32,
        found_breaks: &mut DequeI,
    ) -> i32 {
        if range_end - range_start < self.min_span {
            return 0;
        }
        let dict = &self.dictionary;
        let mut words_found = 0i32;
        let mut words: [PossibleWord; 3] = Default::default();
        let w = |n: i32| (n % LOOKAHEAD) as usize;
        let mut uc;
        it.set_index(range_start as usize);
        loop {
            let current = it.index() as i32;
            if current >= range_end {
                break;
            }
            let mut word_length = 0;
            let candidates = words[w(words_found)].candidates(it, dict, range_end);
            if candidates == 1 {
                word_length = words[w(words_found)].accept_marked(it);
                words_found += 1;
            } else if candidates > 1 {
                if (it.index() as i32) < range_end {
                    if self.thai {
                        // foundBest: do { ... } while (words[wf].backUp(fIter));
                        'found_best: loop {
                            if words[w(words_found + 1)].candidates(it, dict, range_end) > 0 {
                                words[w(words_found)].mark_current();
                                if it.index() as i32 >= range_end {
                                    break 'found_best;
                                }
                                loop {
                                    if words[w(words_found + 2)].candidates(it, dict, range_end) > 0
                                    {
                                        words[w(words_found)].mark_current();
                                        break 'found_best;
                                    }
                                    if !words[w(words_found + 1)].back_up(it) {
                                        break;
                                    }
                                }
                            }
                            if !words[w(words_found)].back_up(it) {
                                break;
                            }
                        }
                    } else {
                        let mut found_best = false;
                        loop {
                            if words[w(words_found + 1)].candidates(it, dict, range_end) > 0 {
                                words[w(words_found)].mark_current();
                                if it.index() as i32 >= range_end {
                                    break;
                                }
                                loop {
                                    if words[w(words_found + 2)].candidates(it, dict, range_end) > 0
                                    {
                                        words[w(words_found)].mark_current();
                                        found_best = true;
                                        break;
                                    }
                                    if !words[w(words_found + 1)].back_up(it) {
                                        break;
                                    }
                                }
                            }
                            // `backUp(fIter) && !foundBest`: backUp runs first.
                            if !(words[w(words_found)].back_up(it) && !found_best) {
                                break;
                            }
                        }
                    }
                }
                word_length = words[w(words_found)].accept_marked(it);
                words_found += 1;
            }
            if (it.index() as i32) < range_end && word_length < ROOT_COMBINE_THRESHOLD {
                if words[w(words_found)].candidates(it, dict, range_end) <= 0
                    && (word_length == 0
                        || words[w(words_found)].longest_prefix() < PREFIX_COMBINE_THRESHOLD)
                {
                    let mut remaining = range_end - (current + word_length);
                    let mut pc = it.current();
                    let mut chars = 0;
                    loop {
                        it.next();
                        uc = it.current();
                        chars += 1;
                        remaining -= 1;
                        if remaining <= 0 {
                            break;
                        }
                        if self.end_word_set.contains(pc) && self.begin_word_set.contains(uc) {
                            let candidate =
                                words[w(words_found + 1)].candidates(it, dict, range_end);
                            it.set_index((current + word_length + chars) as usize);
                            if candidate > 0 {
                                break;
                            }
                        }
                        pc = uc;
                    }
                    if word_length <= 0 {
                        words_found += 1;
                    }
                    word_length += chars;
                } else {
                    it.set_index((current + word_length) as usize);
                }
            }
            loop {
                let curr_pos = it.index() as i32;
                if curr_pos >= range_end || !self.mark_set.contains(it.current()) {
                    break;
                }
                it.next();
                word_length += it.index() as i32 - curr_pos;
            }
            if self.thai && (it.index() as i32) < range_end && word_length > 0 {
                if words[w(words_found)].candidates(it, dict, range_end) <= 0 && {
                    uc = it.current();
                    self.suffix_set.contains(uc)
                } {
                    if uc == THAI_PAIYANNOI {
                        if !self.suffix_set.contains(it.previous()) {
                            it.next();
                            it.next();
                            word_length += 1;
                            uc = it.current();
                        } else {
                            it.next();
                        }
                    }
                    if uc == THAI_MAIYAMOK {
                        if it.previous() != THAI_MAIYAMOK {
                            it.next();
                            it.next();
                            word_length += 1;
                        } else {
                            it.next();
                        }
                    }
                } else {
                    it.set_index((current + word_length) as usize);
                }
            }
            if word_length > 0 {
                found_breaks.push(current + word_length);
            }
        }
        if found_breaks.peek() >= range_end {
            found_breaks.pop();
            words_found -= 1;
        }
        words_found
    }
}

// --- CjkBreakEngine ---------------------------------------------------------------

const MAX_KATAKANA_LENGTH: i32 = 8;
const MAX_KATAKANA_GROUP_LENGTH: i32 = 20;
const MAX_SNLP: i32 = 255;

/// `CjkBreakEngine`: the shortest-path segmentation over `cjdict`'s word
/// costs, with katakana runs costed by length.
#[derive(Debug)]
pub struct CjkEngine {
    korean: bool,
    set: UnicodeSet,
    hangul_word_set: UnicodeSet,
    dictionary: DictionaryMatcher,
}

impl CjkEngine {
    /// `new CjkBreakEngine(korean)`.
    fn new(korean: bool) -> CjkEngine {
        let hangul_word_set = UnicodeSet::from_range(0xac00, 0xd7a3);
        let set = if korean {
            hangul_word_set.clone()
        } else {
            uset("[[:Han:][:Hiragana:][:Katakana:]\\u30fc\\uff70\\uff9e\\uff9f]")
        };
        CjkEngine {
            korean,
            set,
            hangul_word_set,
            dictionary: load_vendored(CJ_DICT),
        }
    }

    fn get_katakana_cost(word_length: i32) -> i32 {
        const COST: [i32; 9] = [8192, 984, 408, 240, 204, 252, 300, 372, 480];
        if word_length > MAX_KATAKANA_LENGTH {
            8192
        } else {
            usize::try_from(word_length)
                .ok()
                .and_then(|i| COST.get(i))
                .copied()
                .unwrap_or(8192)
        }
    }

    fn is_katakana(value: i32) -> bool {
        ((0x30a1..=0x30fe).contains(&value) && value != 0x30fb)
            || (0xff66..=0xff9f).contains(&value)
    }

    /// `divideUpDictionaryRange(inText, startPos, endPos, foundBreaks,
    /// false)` (word, not phrase, breaking).
    // ARITH: positions and code point counts are bounded by the text
    // length; costs add at most 20 dictionary values (< 2^16 each) or the
    // katakana cost to a path cost below i32::MAX, which `best_snlp` starts
    // at and never exceeds (`kint32max` paths are skipped).
    #[allow(clippy::arithmetic_side_effects)]
    fn divide_up_dictionary_range(
        &self,
        in_text: &mut CharIter<'_>,
        start_pos: i32,
        end_pos: i32,
        found_breaks: &mut DequeI,
    ) -> i32 {
        if start_pos >= end_pos {
            return 0;
        }
        let input_length = (end_pos - start_pos) as usize;
        let src = &in_text.text()[start_pos as usize..end_pos as usize];
        let _ = input_length;
        let nfkc = nfkc();
        let is_normalized = nfkc.quick_check(src) == QuickCheck::Yes || nfkc.is_normalized(src);
        let norm_buf;
        let mut char_positions: Vec<i32>;
        let mut num_code_pts: i32 = 0;
        let text_units: &[u16] = if is_normalized {
            // ALLOC: one position per unit of the text's dictionary run (no
            // count read off the data).
            char_positions = vec![0; src.len() + 1];
            let mut index = 0usize;
            while index < src.len() {
                let cp = crate::icu4j::utf16::code_point_at(src, index);
                index += if cp > 0xffff { 2 } else { 1 };
                num_code_pts += 1;
                char_positions[num_code_pts as usize] = index as i32;
            }
            src
        } else {
            norm_buf = nfkc.normalize(src);
            char_positions = vec![0; norm_buf.len() + 1];
            let mut it = LegacyNormalizerIter::new(src, &nfkc);
            let mut index = 0usize;
            while index < src.len() {
                if it.next_cp() < 0 {
                    // Java: `next()` past the end returns DONE and leaves the
                    // index where it is, which loops forever; no NFKC segment
                    // normalizes to nothing, so this is unreachable.
                    break;
                }
                num_code_pts += 1;
                index = it.get_index();
                if let Some(slot) = char_positions.get_mut(num_code_pts as usize) {
                    *slot = index as i32;
                }
            }
            &norm_buf
        };
        let n = num_code_pts as usize;
        let mut best_snlp = vec![i32::MAX; n + 1];
        best_snlp[0] = 0;
        let mut prev = vec![-1i32; n + 1];
        const MAX_WORD_SIZE: i32 = 20;
        let mut values = vec![0i32; n];
        let mut lengths = vec![0i32; n];
        let mut text = CharIter::new(text_units);
        let mut ix = 0usize;
        text.set_index(ix);
        let mut is_prev_katakana = false;
        let mut i = 0i32;
        while i < num_code_pts {
            ix = text.index();
            if best_snlp[i as usize] != i32::MAX {
                let max_search_length = if i + MAX_WORD_SIZE < num_code_pts {
                    MAX_WORD_SIZE
                } else {
                    num_code_pts - i
                };
                let mut count = 0;
                self.dictionary.matches(
                    &mut text,
                    max_search_length,
                    &mut lengths,
                    &mut count,
                    max_search_length,
                    Some(&mut values),
                );
                text.set_index(ix);
                let c = text.current32();
                if (count == 0 || get(&lengths, 0) != 1)
                    && c != DONE32
                    && !self.hangul_word_set.contains(c)
                {
                    set(&mut values, count, MAX_SNLP);
                    set(&mut lengths, count, 1);
                    count += 1;
                }
                for j in 0..count {
                    let new_snlp = best_snlp[i as usize] + get(&values, j);
                    let k = (get(&lengths, j) + i) as usize;
                    if k <= n && new_snlp < best_snlp[k] {
                        best_snlp[k] = new_snlp;
                        prev[k] = i;
                    }
                }
                let is_katakana = Self::is_katakana(text.current32());
                if !is_prev_katakana && is_katakana {
                    let mut j = i + 1;
                    text.next32();
                    while j < num_code_pts
                        && (j - i) < MAX_KATAKANA_GROUP_LENGTH
                        && Self::is_katakana(text.current32())
                    {
                        text.next32();
                        j += 1;
                    }
                    if (j - i) < MAX_KATAKANA_GROUP_LENGTH {
                        let new_snlp = best_snlp[i as usize] + Self::get_katakana_cost(j - i);
                        if new_snlp < best_snlp[j as usize] {
                            best_snlp[j as usize] = new_snlp;
                            prev[j as usize] = i;
                        }
                    }
                }
                is_prev_katakana = is_katakana;
            }
            // for (...; i++, text.setIndex(ix), next32(text))
            i += 1;
            text.set_index(ix);
            text.next32();
        }
        let mut t_boundary = vec![0i32; n + 1];
        let mut num_breaks = 0usize;
        if best_snlp[n] == i32::MAX {
            t_boundary[num_breaks] = num_code_pts;
            num_breaks += 1;
        } else {
            let mut i = num_code_pts;
            while i > 0 {
                t_boundary[num_breaks] = i;
                num_breaks += 1;
                i = prev[i as usize];
            }
        }
        if found_breaks.size() == 0 || found_breaks.peek() < start_pos {
            if num_breaks < t_boundary.len() {
                t_boundary[num_breaks] = 0;
            } else {
                t_boundary.push(0);
            }
            num_breaks += 1;
        }
        let mut corrected = 0;
        let mut previous = -1;
        for i in (0..num_breaks).rev() {
            let cp = char_positions
                .get(t_boundary[i] as usize)
                .copied()
                .unwrap_or(0);
            let pos = cp + start_pos;
            in_text.set_index(pos as usize);
            if pos > previous && pos != start_pos {
                found_breaks.push(pos);
                corrected += 1;
            }
            previous = pos;
        }
        if !found_breaks.is_empty() && found_breaks.peek() == end_pos {
            found_breaks.pop();
            corrected -= 1;
        }
        if !found_breaks.is_empty() {
            in_text.set_index(found_breaks.peek() as usize);
        }
        let _ = self.korean;
        corrected
    }
}

fn nfkc() -> Normalizer2 {
    Normalizer2::get_instance("nfkc", Mode::Compose).expect("nfkc is vendored")
}

/// The legacy iterating `com.ibm.icu.text.Normalizer` (`new
/// Normalizer(text, NFKC, 0)`, `next()`, `getIndex()`), which
/// `CjkBreakEngine` reads to map normalized code points to source offsets:
/// the text is normalized one segment at a time (a segment ends before a
/// code point with a boundary before it), and `getIndex()` is the segment's
/// start while its normalized code points are being read, its end once the
/// last has been.
struct LegacyNormalizerIter<'a> {
    text: &'a [u16],
    norm: &'a Normalizer2,
    buffer: Vec<u16>,
    buffer_pos: usize,
    current_index: usize,
    next_index: usize,
}

impl<'a> LegacyNormalizerIter<'a> {
    fn new(text: &'a [u16], norm: &'a Normalizer2) -> Self {
        LegacyNormalizerIter {
            text,
            norm,
            buffer: Vec::new(),
            buffer_pos: 0,
            current_index: 0,
            next_index: 0,
        }
    }

    /// `next()`.
    // SENTINEL: `-1` = the end of the text (`Normalizer.DONE`).
    fn next_cp(&mut self) -> i32 {
        if self.buffer_pos < self.buffer.len() || self.next_normalize() {
            let c = crate::icu4j::utf16::code_point_at(&self.buffer, self.buffer_pos);
            self.buffer_pos = self
                .buffer_pos
                .saturating_add(if c > 0xffff { 2 } else { 1 });
            c
        } else {
            -1
        }
    }

    /// `getIndex()`.
    fn get_index(&self) -> usize {
        if self.buffer_pos < self.buffer.len() {
            self.current_index
        } else {
            self.next_index
        }
    }

    /// `nextNormalize()`.
    fn next_normalize(&mut self) -> bool {
        self.buffer.clear();
        self.buffer_pos = 0;
        self.current_index = self.next_index;
        let mut it = CharIter::new(self.text);
        it.set_index(self.next_index);
        let c = it.next_code_point();
        if c < 0 {
            return false;
        }
        let mut segment = Vec::new();
        crate::icu4j::utf16::push_code_point(&mut segment, c);
        loop {
            let before = it.index();
            let c = it.next_code_point();
            if c < 0 {
                break;
            }
            if self.norm.has_boundary_before(c) {
                it.set_index(before);
                break;
            }
            crate::icu4j::utf16::push_code_point(&mut segment, c);
        }
        self.next_index = it.index();
        self.norm.normalize_to(&segment, &mut self.buffer);
        !self.buffer.is_empty()
    }
}

// --- the engine list ------------------------------------------------------------------

/// A language break engine.
#[derive(Debug, Clone)]
pub enum Engine {
    /// `UnhandledBreakEngine`.
    Unhandled,
    /// A Southeast Asian dictionary engine.
    Sea(Arc<SeaEngine>),
    /// `CjkBreakEngine`.
    Cjk(Arc<CjkEngine>),
}

/// `RuleBasedBreakIterator.gAllBreakEngines` and the unhandled engine's
/// `fHandled` set.
#[derive(Debug, Default)]
struct Registry {
    engines: Vec<Engine>,
    unhandled: UnicodeSet,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(Registry {
            engines: vec![Engine::Unhandled],
            unhandled: UnicodeSet::new(),
        })
    })
}

impl Engine {
    fn handles(&self, c: i32, unhandled: &UnicodeSet) -> bool {
        match self {
            Engine::Unhandled => unhandled.contains(c),
            Engine::Sea(e) => uprops::script(c) == e.script,
            Engine::Cjk(e) => e.set.contains(c),
        }
    }
}

/// `RuleBasedBreakIterator.getLanguageBreakEngine(c)`.
pub fn language_break_engine(c: i32) -> Engine {
    let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
    let Registry { engines, unhandled } = &mut *r;
    if let Some(e) = engines.iter().find(|e| e.handles(c, unhandled)) {
        return e.clone();
    }
    let mut script = uprops::script(c);
    if script == 22 || script == 20 {
        // KATAKANA, HIRAGANA -> HAN
        script = 17;
    }
    let eng = match script {
        38 | 24 | 28 | 23 => SeaEngine::new(script).map(|e| Engine::Sea(Arc::new(e))),
        17 => Some(Engine::Cjk(Arc::new(CjkEngine::new(false)))),
        18 => Some(Engine::Cjk(Arc::new(CjkEngine::new(true)))),
        _ => None,
    };
    match eng {
        Some(e) => {
            engines.push(e.clone());
            e
        }
        None => {
            // UnhandledBreakEngine.handleChar(c)
            if !unhandled.contains(c) {
                let mut s = UnicodeSet::new();
                if s.apply_int_property_value(uprops::SCRIPT, uprops::script(c))
                    .is_ok()
                {
                    unhandled.add_all(&s);
                }
            }
            Engine::Unhandled
        }
    }
}

impl Engine {
    /// `LanguageBreakEngine.findBreaks(text, startPos, endPos, foundBreaks)`.
    // ARITH: indices are text positions.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn find_breaks(
        &self,
        text: &mut CharIter<'_>,
        _start_pos: i32,
        end_pos: i32,
        found_breaks: &mut DequeI,
    ) -> i32 {
        let set = match self {
            Engine::Unhandled => {
                let r = registry().lock().unwrap_or_else(|e| e.into_inner());
                let uniset = r.unhandled.clone();
                drop(r);
                let mut c = text.current32();
                while (text.index() as i32) < end_pos && uniset.contains(c) {
                    text.next32();
                    c = text.current32();
                }
                return 0;
            }
            Engine::Sea(e) => &e.set,
            Engine::Cjk(e) => &e.set,
        };
        // DictionaryBreakEngine.findBreaks
        let start = text.index() as i32;
        let mut c = text.current32();
        let mut current;
        loop {
            current = text.index() as i32;
            if current >= end_pos || !set.contains(c) {
                break;
            }
            text.next32();
            c = text.current32();
        }
        let result = match self {
            Engine::Sea(e) => e.divide_up_dictionary_range(text, start, current, found_breaks),
            Engine::Cjk(e) => e.divide_up_dictionary_range(text, start, current, found_breaks),
            Engine::Unhandled => 0,
        };
        text.set_index(current as usize);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deque_reads_like_java() {
        let mut d = DequeI::default();
        assert!(d.is_empty());
        d.offer(7);
        assert_eq!(d.size(), 1);
        assert_eq!(d.element_at(0), 7);
        d.remove_all_elements();
        // The stale slot below the base.
        assert_eq!(d.peek(), 7);
        for i in 0..100 {
            d.push(i);
        }
        assert_eq!(d.size(), 100);
        assert_eq!(d.pop(), 99);
        assert_eq!(d.peek(), 98);
        assert_eq!(d.element_at(0), 0);
        assert_eq!(d.element_at(-100), 0);
    }

    fn units(s: &str) -> Vec<u16> {
        crate::icu4j::utf16::units(s)
    }

    #[test]
    fn dictionaries_match_words() {
        let thai = load_vendored(THAI_DICT);
        let t = units("ภาษาไทย");
        let mut it = CharIter::new(&t);
        let mut lengths = [0i32; 20];
        let mut count = 0;
        let n = thai.matches(&mut it, 7, &mut lengths, &mut count, 20, None);
        assert!(n >= 4 && count >= 1, "{n} {count}");
        assert!(lengths[..count as usize].contains(&4));
        // Past the dictionary's characters the transform reads -1.
        let t = units("abc");
        let mut it = CharIter::new(&t);
        thai.matches(&mut it, 3, &mut lengths, &mut count, 20, None);
        assert_eq!(count, 0);
        let mut it = CharIter::new(&[]);
        assert_eq!(
            thai.matches(&mut it, 3, &mut lengths, &mut count, 20, None),
            0
        );
        assert_eq!(
            DictionaryMatcher::transform(0x0e00 | TRANSFORM_TYPE_OFFSET, 0x200d),
            0xff
        );
        assert_eq!(
            DictionaryMatcher::transform(0x0e00 | TRANSFORM_TYPE_OFFSET, 0x200c),
            0xfe
        );
        let cj = load_vendored(CJ_DICT);
        let t = units("東京大学\u{20000}");
        let mut it = CharIter::new(&t);
        let mut values = [0i32; 20];
        cj.matches(&mut it, 5, &mut lengths, &mut count, 20, Some(&mut values));
        assert!(count >= 2, "{count}");
        assert!(values[..count as usize].iter().all(|&v| v > 0));
        let t = units("\u{20000}\u{20001}");
        let mut it = CharIter::new(&t);
        cj.matches(&mut it, 2, &mut lengths, &mut count, 1, Some(&mut values));
        assert!(count <= 1);
    }

    #[test]
    fn dictionary_header_errors() {
        assert!(DictionaryMatcher::load(&[0; 10]).is_err());
        let good = miniz_oxide::inflate::decompress_to_vec_zlib(THAI_DICT).unwrap();
        let h = usize::from(u16::from_be_bytes([good[0], good[1]]));
        let mut b = good.clone();
        b[h..h + 4].copy_from_slice(&4i32.to_be_bytes());
        assert!(DictionaryMatcher::load(&b).is_err());
        let mut b = good.clone();
        b[h + 16..h + 20].copy_from_slice(&7i32.to_be_bytes());
        assert!(DictionaryMatcher::load(&b).is_err());
        let mut b = good.clone();
        b[h + 20..h + 24].copy_from_slice(&0i32.to_be_bytes());
        assert!(DictionaryMatcher::load(&b).is_err());
        assert!(SeaEngine::new(25).is_none());
        assert_eq!(CjkEngine::get_katakana_cost(3), 240);
        assert_eq!(CjkEngine::get_katakana_cost(30), 8192);
    }
}
