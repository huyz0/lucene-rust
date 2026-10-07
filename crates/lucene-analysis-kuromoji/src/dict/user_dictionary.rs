//! `ja.dict.UserDictionary` and `ja.dict.UserMorphData`: a user's
//! `surface,segmentation,readings,part-of-speech` lines (CSV, `#`
//! comments), each a phrase the tokenizer prefers (cost -100000) and
//! outputs as its segments.
//!
//! Differs: the phrases are a trie, not an FST
//! ([`TokenInfoFst::from_sorted`] says why that is not observable). A
//! segment's part of speech is `None` where Java's `getPartOfSpeech`
//! throws `ArrayIndexOutOfBoundsException` (an entry whose part-of-speech
//! field is empty, which `String.split` drops).

use std::sync::Arc;

use lucene_analysis::factory::{java_trim, FactoryError, JavaException};
use lucene_analysis::morph::{MorphData, TokenInfoFst};
use lucene_analysis::util::csv_util;

/// `UserDictionary.INTERNAL_SEPARATOR`.
pub const INTERNAL_SEPARATOR: char = '\u{0}';
/// `UserDictionary.CUSTOM_DICTIONARY_WORD_ID_OFFSET`.
pub const CUSTOM_DICTIONARY_WORD_ID_OFFSET: i32 = 100_000_000;

/// `UserMorphData.WORD_COST`.
pub const WORD_COST: i32 = -100_000;
/// `UserMorphData.LEFT_ID`.
pub const LEFT_ID: i32 = 5;
/// `UserMorphData.RIGHT_ID`.
pub const RIGHT_ID: i32 = 5;

/// `UserMorphData`: per word id, `reading \0 partOfSpeech`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMorphData {
    data: Vec<String>,
}

impl MorphData for UserMorphData {
    fn left_id(&self, _: i32) -> i32 {
        LEFT_ID
    }
    fn right_id(&self, _: i32) -> i32 {
        RIGHT_ID
    }
    fn word_cost(&self, _: i32) -> i32 {
        WORD_COST
    }
}

/// `String.split(regex)`'s result shaping: trailing empty strings dropped.
fn drop_trailing_empty(mut v: Vec<&str>) -> Vec<&str> {
    while v.last().is_some_and(|s| s.is_empty()) {
        v.pop();
    }
    v
}

/// `s.split(" +")`.
fn split_spaces(s: &str) -> Vec<&str> {
    if !s.contains(' ') {
        return vec![s];
    }
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find(' ') {
        out.push(&rest[..i]);
        rest = rest[i..].trim_start_matches(' ');
    }
    out.push(rest);
    drop_trailing_empty(out)
}

/// `WHITESPACE.matcher(s).replaceAll("")`: `\s` is `[ \t\n\x0B\f\r]`.
fn remove_whitespace(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, ' ' | '\t' | '\n' | '\u{B}' | '\u{C}' | '\r'))
        .collect()
}

fn utf16_len(s: &str) -> i32 {
    i32::try_from(s.encode_utf16().count()).unwrap_or(i32::MAX)
}

impl UserMorphData {
    /// `getAllFeaturesArray(wordId)`.
    fn features(&self, word_id: i32) -> Option<Vec<&str>> {
        let i = usize::try_from(word_id.checked_sub(CUSTOM_DICTIONARY_WORD_ID_OFFSET)?).ok()?;
        let all = self.data.get(i)?;
        Some(drop_trailing_empty(all.split(INTERNAL_SEPARATOR).collect()))
    }

    /// `getReading(morphId, ...)`: feature 0.
    pub fn reading(&self, word_id: i32) -> Option<String> {
        self.features(word_id)?.first().map(|s| s.to_string())
    }

    /// `getPartOfSpeech(morphId)`: feature 1.
    pub fn part_of_speech(&self, word_id: i32) -> Option<String> {
        self.features(word_id)?.get(1).map(|s| s.to_string())
    }
}

/// `UserDictionary`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDictionary {
    fst: Arc<TokenInfoFst>,
    /// wordId, length, length... indexed by phrase id.
    segmentations: Vec<Vec<i32>>,
    morph_atts: UserMorphData,
}

/// `LINE_COMMENT.matcher(line).replaceAll("")` for `^#.*$`: `.` stops at a
/// line terminator and `$` matches at the end or before a final one.
fn remove_comment(line: &str) -> &str {
    if !line.starts_with('#') {
        return line;
    }
    let is_terminator = |c: char| matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}');
    match line.char_indices().find(|&(_, c)| is_terminator(c)) {
        None => "",
        Some((i, c)) if i.saturating_add(c.len_utf8()) == line.len() => &line[i..],
        Some(_) => line,
    }
}

/// `BufferedReader.readLine`: lines end at `\n`, `\r` or `\r\n`.
pub(crate) fn read_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        match rest.find(['\n', '\r']) {
            None => {
                lines.push(rest);
                break;
            }
            Some(i) => {
                lines.push(&rest[..i]);
                let skip = if rest[i..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[i.saturating_add(skip)..];
            }
        }
    }
    lines
}

fn runtime(message: String) -> FactoryError {
    FactoryError::new(JavaException::Runtime, message)
}

impl UserDictionary {
    /// `UserDictionary.open(reader)`: `None` when the text has no entry.
    pub fn open(text: &str) -> Result<Option<UserDictionary>, FactoryError> {
        let mut entries: Vec<Vec<String>> = Vec::new();
        for line in read_lines(text) {
            // Remove comments; skip empty lines or comment lines.
            let line = remove_comment(line);
            if java_trim(line).is_empty() {
                continue;
            }
            entries.push(csv_util::parse(line));
        }
        if entries.is_empty() {
            Ok(None)
        } else {
            Self::new(entries).map(Some)
        }
    }

    // Java: UserDictionary(List<String[]>)
    fn new(mut entries: Vec<Vec<String>>) -> Result<UserDictionary, FactoryError> {
        let aioobe = |i: usize| {
            FactoryError::new(
                JavaException::ArrayIndexOutOfBounds,
                format!("Index {i} out of bounds for length 0"),
            )
        };
        if entries.iter().any(Vec::is_empty) {
            return Err(aioobe(0));
        }
        // left[0].compareTo(right[0]): UTF-16 order; a stable sort.
        entries.sort_by(|a, b| a[0].encode_utf16().cmp(b[0].encode_utf16()));

        let mut word_id = CUSTOM_DICTIONARY_WORD_ID_OFFSET;
        let mut data = Vec::with_capacity(entries.len());
        let mut segmentations = Vec::with_capacity(entries.len());
        let mut keys: Vec<Vec<u16>> = Vec::with_capacity(entries.len());
        for values in &entries {
            if values.len() < 4 {
                return Err(FactoryError::new(
                    JavaException::ArrayIndexOutOfBounds,
                    format!(
                        "Index {} out of bounds for length {}",
                        values.len(),
                        values.len()
                    ),
                ));
            }
            let surface = remove_whitespace(&values[0]);
            let concatenated_segment = remove_whitespace(&values[1]);
            let segmentation = split_spaces(&values[1]);
            let readings = split_spaces(&values[2]);
            let pos = &values[3];

            if segmentation.len() != readings.len() {
                return Err(runtime(format!(
                    "Illegal user dictionary entry {} - the number of segmentations ({}) does not the match number of readings ({})",
                    values[0],
                    segmentation.len(),
                    readings.len()
                )));
            }
            if surface != concatenated_segment {
                return Err(runtime(format!(
                    "Illegal user dictionary entry {} - the concatenated segmentation ({concatenated_segment}) does not match the surface form ({surface})",
                    values[0]
                )));
            }
            // wordId offset, length, length....
            let mut word_id_and_length = Vec::with_capacity(segmentation.len().saturating_add(1));
            word_id_and_length.push(word_id);
            for (seg, reading) in segmentation.iter().zip(&readings) {
                word_id_and_length.push(utf16_len(seg));
                data.push(format!("{reading}{INTERNAL_SEPARATOR}{pos}"));
                word_id = word_id.wrapping_add(1);
            }
            keys.push(values[0].encode_utf16().collect());
            segmentations.push(word_id_and_length);
        }
        let fst = TokenInfoFst::from_sorted(&keys, 0x30FF, 0x3040)
            .map_err(|_| FactoryError::new(JavaException::UnsupportedOperation, ""))?;
        Ok(UserDictionary {
            fst: Arc::new(fst),
            segmentations,
            morph_atts: UserMorphData { data },
        })
    }

    /// `getMorphAttributes()`.
    pub fn morph_attributes(&self) -> &UserMorphData {
        &self.morph_atts
    }

    /// `getFST()`.
    pub fn fst(&self) -> &Arc<TokenInfoFst> {
        &self.fst
    }

    /// `lookupSegmentation(phraseID)`: the first word id, then each
    /// segment's length (empty for an id that names no phrase).
    pub fn lookup_segmentation(&self, phrase_id: i32) -> &[i32] {
        usize::try_from(phrase_id)
            .ok()
            .and_then(|i| self.segmentations.get(i))
            .map_or(&[], Vec::as_slice)
    }

    /// `lookup(chars, off, len)`: every phrase found in `chars`, as
    /// `{wordId, position, length}` per segment, the longest phrase at each
    /// start.
    pub fn lookup(&self, chars: &[u16]) -> Vec<[i32; 3]> {
        let mut result = Vec::new();
        for start in 0..chars.len() {
            let mut arc = self.fst.first_arc();
            let mut output: i32 = 0;
            let mut found: Option<&[i32]> = None;
            for (i, &ch) in chars[start..].iter().enumerate() {
                match self.fst.find_target_arc(i32::from(ch), &arc, i == 0) {
                    Ok(Some(a)) => arc = a,
                    _ => break,
                }
                output = output.wrapping_add(arc.output() as i32);
                if arc.is_final() {
                    found =
                        Some(self.lookup_segmentation(
                            output.wrapping_add(arc.next_final_output() as i32),
                        ));
                }
            }
            if let Some(word_id_and_length) = found {
                let mut position = i32::try_from(start).unwrap_or(i32::MAX);
                if let Some((&word_id, lengths)) = word_id_and_length.split_first() {
                    for (j, &len) in lengths.iter().enumerate() {
                        let id = word_id.wrapping_add(i32::try_from(j).unwrap_or(i32::MAX));
                        result.push([id, position, len]);
                        position = position.wrapping_add(len);
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_split_and_comment_shapes() {
        assert_eq!(split_spaces(""), [""]);
        assert_eq!(split_spaces("a  b"), ["a", "b"]);
        assert_eq!(split_spaces(" a"), ["", "a"]);
        assert_eq!(split_spaces("a "), ["a"]);
        assert!(split_spaces("  ").is_empty());
        assert_eq!(remove_comment("#x"), "");
        assert_eq!(remove_comment("#x\u{2028}"), "\u{2028}");
        assert_eq!(remove_comment("#x\u{2028}y"), "#x\u{2028}y");
        assert_eq!(remove_comment("a#x"), "a#x");
        assert_eq!(read_lines("a\r\nb\rc\n\nd"), ["a", "b", "c", "", "d"]);
        assert_eq!(remove_whitespace(" a\tb\u{B}"), "ab");
    }

    #[test]
    fn entries_and_errors() {
        let d = UserDictionary::open(
            "# c\n関西国際空港,関西 国際 空港,カンサイ コクサイ クウコウ,カスタム名詞\n\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            d.lookup_segmentation(0),
            [CUSTOM_DICTIONARY_WORD_ID_OFFSET, 2, 2, 2]
        );
        assert!(d.lookup_segmentation(1).is_empty());
        let m = d.morph_attributes();
        assert_eq!(
            m.reading(CUSTOM_DICTIONARY_WORD_ID_OFFSET + 1).as_deref(),
            Some("コクサイ")
        );
        assert_eq!(
            m.part_of_speech(CUSTOM_DICTIONARY_WORD_ID_OFFSET)
                .as_deref(),
            Some("カスタム名詞")
        );
        assert_eq!(m.reading(5), None);
        assert_eq!(
            (m.left_id(0), m.right_id(0), m.word_cost(0)),
            (5, 5, -100000)
        );
        let text: Vec<u16> = "で関西国際空港へ".encode_utf16().collect();
        assert_eq!(
            d.lookup(&text),
            [
                [CUSTOM_DICTIONARY_WORD_ID_OFFSET, 1, 2],
                [CUSTOM_DICTIONARY_WORD_ID_OFFSET + 1, 3, 2],
                [CUSTOM_DICTIONARY_WORD_ID_OFFSET + 2, 5, 2]
            ]
        );
        assert!(UserDictionary::open("# only\n  \n").unwrap().is_none());
        let e = UserDictionary::open("ab,a b,x,n").unwrap_err();
        assert_eq!(e.kind, JavaException::Runtime);
        assert!(
            e.message.contains("number of segmentations (2)"),
            "{}",
            e.message
        );
        let e = UserDictionary::open("ab,a c,x y,n").unwrap_err();
        assert!(
            e.message.contains("does not match the surface form (ab)"),
            "{}",
            e.message
        );
        assert_eq!(
            UserDictionary::open("a,b").unwrap_err().kind,
            JavaException::ArrayIndexOutOfBounds
        );
        assert_eq!(
            UserDictionary::open("\"a,b").unwrap_err().kind,
            JavaException::ArrayIndexOutOfBounds
        );
        assert_eq!(
            UserDictionary::open("a,a,a,n\na,a,a,v").unwrap_err().kind,
            JavaException::UnsupportedOperation
        );
        // An empty part of speech: no feature 1.
        let d = UserDictionary::open("x,x,y,").unwrap().unwrap();
        assert_eq!(
            d.morph_attributes()
                .part_of_speech(CUSTOM_DICTIONARY_WORD_ID_OFFSET),
            None
        );
    }
}
