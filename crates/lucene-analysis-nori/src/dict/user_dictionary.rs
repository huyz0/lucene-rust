//! `ko.dict.UserDictionary` and `ko.dict.UserMorphData`: a user's nouns,
//! one per line (`#` comments), optionally followed by the segmentation of
//! a compound, each a general noun (`NNG`) the tokenizer prefers (cost
//! -100000).
//!
//! Differs: the entries are a trie, not an FST
//! (`lucene_analysis::morph::TokenInfoFst::from_sorted` says why that is
//! not observable).

use std::sync::Arc;

use lucene_analysis::factory::{java_trim, FactoryError, JavaException};
use lucene_analysis::morph::{MorphData, TokenInfoFst};

use super::character_definition::CharacterDefinition;
use super::Morpheme;
use crate::pos::{Tag, Type};

/// `UserMorphData.WORD_COST`.
pub const WORD_COST: i32 = -100_000;
/// `UserMorphData.LEFT_ID`: NNG left.
pub const LEFT_ID: i32 = 1781;
/// `UserDictionary.RIGHT_ID`: NNG right.
pub const RIGHT_ID: i16 = 3533;
/// `UserDictionary.RIGHT_ID_T`: NNG right, a Hangul last character with a
/// coda.
pub const RIGHT_ID_T: i16 = 3535;
/// `UserDictionary.RIGHT_ID_F`: NNG right, a Hangul last character
/// without a coda.
pub const RIGHT_ID_F: i16 = 3534;

/// `UserMorphData`: per entry, the segment lengths (`None`: a simple noun)
/// and the right id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMorphData {
    segmentations: Vec<Option<Vec<i32>>>,
    right_ids: Vec<i16>,
}

impl MorphData for UserMorphData {
    fn left_id(&self, _: i32) -> i32 {
        LEFT_ID
    }
    fn right_id(&self, id: i32) -> i32 {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.right_ids.get(i))
            .map_or(0, |&r| i32::from(r))
    }
    fn word_cost(&self, _: i32) -> i32 {
        WORD_COST
    }
}

impl UserMorphData {
    fn segmentation(&self, id: i32) -> Option<&Vec<i32>> {
        self.segmentations.get(usize::try_from(id).ok()?)?.as_ref()
    }

    /// `getPOSType(morphId)`.
    pub fn pos_type(&self, id: i32) -> Type {
        match self.segmentation(id) {
            None => Type::Morpheme,
            Some(_) => Type::Compound,
        }
    }

    /// `getMorphemes(morphId, surfaceForm, off, len)`: the segments, each an
    /// `NNG`.
    pub fn morphemes(&self, id: i32, surface: &[u16], off: i32) -> Option<Vec<Morpheme>> {
        let segs = self.segmentation(id)?;
        let mut offset = usize::try_from(off).ok()?;
        // ALLOC: `segs` is already in memory (a parsed user entry).
        let mut out = Vec::with_capacity(segs.len());
        for &len in segs {
            let end = offset.checked_add(usize::try_from(len).ok()?)?;
            out.push(Morpheme {
                pos_tag: Tag::Nng,
                surface_form: String::from_utf16_lossy(surface.get(offset..end)?),
            });
            offset = end;
        }
        Some(out)
    }
}

/// `UserDictionary`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDictionary {
    fst: Arc<TokenInfoFst>,
    morph_atts: UserMorphData,
}

/// Java regex `\s`: `[ \t\n\x0B\f\r]`.
fn is_java_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\u{B}' | '\u{C}' | '\r')
}

/// `s.split("\\s+")`: a leading empty string kept, trailing ones dropped,
/// `[s]` when nothing matches.
fn split_whitespace(s: &str) -> Vec<&str> {
    if !s.contains(is_java_space) {
        return vec![s];
    }
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find(is_java_space) {
        out.push(&rest[..i]);
        rest = rest[i..].trim_start_matches(is_java_space);
    }
    out.push(rest);
    while out.last().is_some_and(|p| p.is_empty()) {
        out.pop();
    }
    out
}

/// `line.replaceAll("#.*$", "")`: `.` stops at a line terminator and `$`
/// matches at the end or before a final one, so the first `#` after the
/// last inner terminator starts the comment.
fn remove_comment(line: &str) -> String {
    let is_terminator = |c: char| matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}');
    let last = line.chars().last();
    let body_end = match last {
        Some(c) if is_terminator(c) => line.len().saturating_sub(c.len_utf8()),
        _ => line.len(),
    };
    let body = &line[..body_end];
    let tail_start = body
        .char_indices()
        .rfind(|&(_, c)| is_terminator(c))
        .map_or(0, |(i, c)| i.saturating_add(c.len_utf8()));
    match body[tail_start..].find('#') {
        Some(h) => format!(
            "{}{}",
            &line[..tail_start.saturating_add(h)],
            &line[body_end..]
        ),
        None => line.to_string(),
    }
}

fn utf16_len(s: &str) -> i32 {
    i32::try_from(s.encode_utf16().count()).unwrap_or(i32::MAX)
}

impl UserDictionary {
    /// `UserDictionary.open(reader)`: `None` when the text has no entry.
    pub fn open(text: &str) -> Result<Option<UserDictionary>, FactoryError> {
        let mut entries = Vec::new();
        for line in read_lines(text) {
            // Remove comments; skip empty lines or comment lines.
            let line = remove_comment(line);
            if java_trim(&line).is_empty() {
                continue;
            }
            entries.push(line);
        }
        if entries.is_empty() {
            Ok(None)
        } else {
            Self::new(entries).map(Some)
        }
    }

    // Java: UserDictionary(List<String>)
    fn new(mut entries: Vec<String>) -> Result<UserDictionary, FactoryError> {
        let char_def = CharacterDefinition::instance();
        // Comparator.comparing(e -> e.split("\\s+")[0]): UTF-16 order, stable.
        entries.sort_by(|a, b| {
            let (a, b) = (split_whitespace(a)[0], split_whitespace(b)[0]);
            a.encode_utf16().cmp(b.encode_utf16())
        });
        let mut last_token: Option<String> = None;
        let mut segmentations = Vec::with_capacity(entries.len());
        let mut right_ids = Vec::with_capacity(entries.len());
        let mut keys: Vec<Vec<u16>> = Vec::with_capacity(entries.len());
        for entry in &entries {
            let splits = split_whitespace(entry);
            let token = splits[0];
            if last_token.as_deref() == Some(token) {
                continue;
            }
            let last_char = entry.encode_utf16().last().unwrap_or(0);
            right_ids.push(if char_def.is_hangul(last_char) {
                if char_def.has_coda(last_char) {
                    RIGHT_ID_T
                } else {
                    RIGHT_ID_F
                }
            } else {
                RIGHT_ID
            });
            if splits.len() == 1 {
                segmentations.push(None);
            } else {
                let lengths: Vec<i32> = splits[1..].iter().map(|s| utf16_len(s)).collect();
                let offset = lengths.iter().fold(0i32, |a, &l| a.wrapping_add(l));
                if offset > utf16_len(token) {
                    return Err(FactoryError::new(
                        JavaException::IllegalArgument,
                        format!(
                            "Illegal user dictionary entry {entry} - the segmentation is bigger than the surface form ({token})"
                        ),
                    ));
                }
                segmentations.push(Some(lengths));
            }
            keys.push(token.encode_utf16().collect());
            last_token = Some(token.to_string());
        }
        let fst = TokenInfoFst::from_sorted(&keys, 0xD7A3, 0xAC00)
            .map_err(|_| FactoryError::new(JavaException::UnsupportedOperation, ""))?;
        Ok(UserDictionary {
            fst: Arc::new(fst),
            morph_atts: UserMorphData {
                segmentations,
                right_ids,
            },
        })
    }

    /// `getFST()`.
    pub fn fst(&self) -> &Arc<TokenInfoFst> {
        &self.fst
    }

    /// `getMorphAttributes()`.
    pub fn morph_attributes(&self) -> &UserMorphData {
        &self.morph_atts
    }

    /// `lookup(chars, off, len)`: the entry ids found at every start.
    pub fn lookup(&self, chars: &[u16]) -> Vec<i32> {
        let mut result = Vec::new();
        for start in 0..chars.len() {
            let mut arc = self.fst.first_arc();
            let mut output: i32 = 0;
            for (i, &ch) in chars[start..].iter().enumerate() {
                match self.fst.find_target_arc(i32::from(ch), &arc, i == 0) {
                    Ok(Some(a)) => arc = a,
                    _ => break,
                }
                output = output.wrapping_add(arc.output() as i32);
                if arc.is_final() {
                    result.push(output.wrapping_add(arc.next_final_output() as i32));
                }
            }
        }
        result
    }
}

/// `BufferedReader.readLine`: lines end at `\n`, `\r` or `\r\n`.
fn read_lines(text: &str) -> Vec<&str> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_string_shapes() {
        assert_eq!(split_whitespace("a"), ["a"]);
        assert_eq!(split_whitespace(" a  b "), ["", "a", "b"]);
        assert_eq!(remove_comment("abc # x"), "abc ");
        assert_eq!(remove_comment("#x"), "");
        assert_eq!(remove_comment("a#b\u{2028}"), "a\u{2028}");
        assert_eq!(remove_comment("a#b\u{2028}c#d"), "a#b\u{2028}c");
        assert_eq!(remove_comment("abc"), "abc");
        assert_eq!(read_lines("a\r\nb\rc\n"), ["a", "b", "c"]);
    }

    #[test]
    fn entries_ids_and_errors() {
        let d = UserDictionary::open("# c\nc++\n세종시 세종 시\n세종시 x\n대한민국날씨\n")
            .unwrap()
            .unwrap();
        let m = d.morph_attributes();
        // sorted: c++, 대한민국날씨, 세종시
        assert_eq!(m.pos_type(0), Type::Morpheme);
        assert_eq!(m.pos_type(2), Type::Compound);
        assert_eq!(m.right_id(0), i32::from(RIGHT_ID));
        assert_eq!(m.right_id(2), i32::from(RIGHT_ID_F)); // 시: no coda
        assert_eq!(m.right_id(1), i32::from(RIGHT_ID_F)); // 씨
        assert_eq!(m.right_id(9), 0);
        assert_eq!((m.left_id(0), m.word_cost(0)), (LEFT_ID, WORD_COST));
        let s: Vec<u16> = "세종시".encode_utf16().collect();
        let morphs = m.morphemes(2, &s, 0).unwrap();
        assert_eq!(
            morphs
                .iter()
                .map(|x| x.surface_form.as_str())
                .collect::<Vec<_>>(),
            ["세종", "시"]
        );
        assert!(m.morphemes(0, &s, 0).is_none());
        let text: Vec<u16> = "세종시c++".encode_utf16().collect();
        assert_eq!(d.lookup(&text), [2, 0]);
        assert!(UserDictionary::open("#\n  \n").unwrap().is_none());
        let e = UserDictionary::open("ab abc").unwrap_err();
        assert_eq!(e.kind, JavaException::IllegalArgument);
        assert!(
            e.message.contains("bigger than the surface form (ab)"),
            "{}",
            e.message
        );
    }
}
