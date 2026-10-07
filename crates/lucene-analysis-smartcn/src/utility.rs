//! `org.apache.lucene.analysis.cn.smart.{Utility, CharType, WordType}`:
//! the segmenter's constants, its character classes and its array
//! comparisons over UTF-16 units.

/// `CharType`: a UTF-16 unit's class ([`get_char_type`]).
pub mod char_type {
    /// Punctuation.
    pub const DELIMITER: i32 = 0;
    /// Basic Latin letters.
    pub const LETTER: i32 = 1;
    /// ASCII digits.
    pub const DIGIT: i32 = 2;
    /// Han ideographs U+4E00..U+9FA5.
    pub const HANZI: i32 = 3;
    /// Space, tab, CR, LF, U+3000.
    pub const SPACE_LIKE: i32 = 4;
    /// Full-width Latin letters.
    pub const FULLWIDTH_LETTER: i32 = 5;
    /// Full-width digits.
    pub const FULLWIDTH_DIGIT: i32 = 6;
    /// Everything else.
    pub const OTHER: i32 = 7;
    /// Surrogates.
    pub const SURROGATE: i32 = 8;
}

/// `WordType`: a segment's type.
pub mod word_type {
    pub const SENTENCE_BEGIN: i32 = 0;
    pub const SENTENCE_END: i32 = 1;
    pub const CHINESE_WORD: i32 = 2;
    pub const STRING: i32 = 3;
    pub const NUMBER: i32 = 4;
    pub const DELIMITER: i32 = 5;
    pub const FULLWIDTH_STRING: i32 = 6;
    pub const FULLWIDTH_NUMBER: i32 = 7;
}

/// `Utility.STRING_CHAR_ARRAY`: `未##串`, the dictionary's word for a run of
/// letters.
pub const STRING_CHAR_ARRAY: [u16; 4] = [0x672A, 0x23, 0x23, 0x4E32];
/// `Utility.NUMBER_CHAR_ARRAY`: `未##数`.
pub const NUMBER_CHAR_ARRAY: [u16; 4] = [0x672A, 0x23, 0x23, 0x6570];
/// `Utility.START_CHAR_ARRAY`: `始##始`.
pub const START_CHAR_ARRAY: [u16; 4] = [0x59CB, 0x23, 0x23, 0x59CB];
/// `Utility.END_CHAR_ARRAY`: `末##末`.
pub const END_CHAR_ARRAY: [u16; 4] = [0x672B, 0x23, 0x23, 0x672B];
/// `Utility.COMMON_DELIMITER`: every delimiter's term, `,`.
pub const COMMON_DELIMITER: [u16; 1] = [0x2C];
/// `Utility.SPACES`.
pub const SPACES: &str = " \u{3000}\t\r\n";
/// `Utility.MAX_FREQUENCE`: the largest word frequency.
pub const MAX_FREQUENCE: i32 = 2_079_997 + 80_000;

/// `Utility.compareArray`: compares `larray[lstart..]` with
/// `rarray[rstart..]` (`None` is Java's `null`); -1, 0 or 1.
// SENTINEL: -1 is the order "less", not an absent value; callers compare it
// with 0.
// ARITH: the indices only grow while below their arrays' lengths.
#[allow(clippy::arithmetic_side_effects)]
pub fn compare_array(
    larray: Option<&[u16]>,
    lstart: usize,
    rarray: Option<&[u16]>,
    rstart: usize,
) -> i32 {
    let (l, r) = match (larray, rarray) {
        (None, None) => return 0,
        (None, Some(r)) => return if rstart >= r.len() { 0 } else { -1 },
        (Some(l), None) => return if lstart >= l.len() { 0 } else { 1 },
        (Some(l), Some(r)) => (l, r),
    };
    let (mut li, mut ri) = (lstart, rstart);
    while li < l.len() && ri < r.len() && l[li] == r[ri] {
        li += 1;
        ri += 1;
    }
    if li >= l.len() {
        // Both arrays are equivalent; or larray has ended first.
        if ri >= r.len() {
            0
        } else {
            -1
        }
    } else if ri >= r.len() {
        // larray > rarray because rarray has ended first.
        1
    } else if l[li] > r[ri] {
        1
    } else {
        -1
    }
}

/// `Utility.compareArrayByPrefix`: 0 when `short[short_index..]` is a prefix
/// of `long[long_index..]`, else the order of the first difference.
// SENTINEL: -1 is the order "less"; callers compare it with 0.
// ARITH: as for `compare_array`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compare_array_by_prefix(
    short: Option<&[u16]>,
    short_index: usize,
    long: Option<&[u16]>,
    long_index: usize,
) -> i32 {
    let Some(s) = short else {
        // a null prefix is a prefix of longArray
        return 0;
    };
    let Some(l) = long else {
        return if short_index < s.len() { 1 } else { 0 };
    };
    let (mut si, mut li) = (short_index, long_index);
    while si < s.len() && li < l.len() && s[si] == l[li] {
        si += 1;
        li += 1;
    }
    if si >= s.len() {
        // shortArray is a prefix of longArray
        0
    } else if li >= l.len() || s[si] > l[li] {
        1
    } else {
        -1
    }
}

/// `Utility.getCharType(char)`.
pub fn get_char_type(ch: u16) -> i32 {
    use char_type::*;
    if (0xD800..=0xDFFF).contains(&ch) {
        return SURROGATE;
    }
    // Most (but not all!) of these are Han Ideographic Characters
    if (0x4E00..=0x9FA5).contains(&ch) {
        return HANZI;
    }
    if (0x41..=0x5A).contains(&ch) || (0x61..=0x7A).contains(&ch) {
        return LETTER;
    }
    if (0x30..=0x39).contains(&ch) {
        return DIGIT;
    }
    if matches!(ch, 0x20 | 0x09 | 0x0D | 0x0A | 0x3000) {
        return SPACE_LIKE;
    }
    // Punctuation Marks
    if (0x21..=0xBB).contains(&ch)
        || (0x2010..=0x2642).contains(&ch)
        || (0x3001..=0x301E).contains(&ch)
    {
        return DELIMITER;
    }
    // Full-Width range
    if (0xFF21..=0xFF3A).contains(&ch) || (0xFF41..=0xFF5A).contains(&ch) {
        return FULLWIDTH_LETTER;
    }
    if (0xFF10..=0xFF19).contains(&ch) {
        return FULLWIDTH_DIGIT;
    }
    if (0xFE30..=0xFF63).contains(&ch) {
        return DELIMITER;
    }
    OTHER
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn compare_array_as_java() {
        let (a, b) = (u("abc"), u("abd"));
        assert_eq!(compare_array(Some(&a), 0, Some(&b), 0), -1);
        assert_eq!(compare_array(Some(&b), 0, Some(&a), 0), 1);
        assert_eq!(compare_array(Some(&a), 0, Some(&a), 0), 0);
        assert_eq!(compare_array(Some(&u("ab")), 0, Some(&a), 0), -1);
        assert_eq!(compare_array(Some(&a), 0, Some(&u("ab")), 0), 1);
        assert_eq!(compare_array(Some(&u("xbc")), 1, Some(&u("bc")), 0), 0);
        assert_eq!(compare_array(None, 0, None, 0), 0);
        assert_eq!(compare_array(None, 0, Some(&a), 3), 0);
        assert_eq!(compare_array(None, 0, Some(&a), 1), -1);
        assert_eq!(compare_array(Some(&a), 3, None, 0), 0);
        assert_eq!(compare_array(Some(&a), 0, None, 0), 1);
    }

    #[test]
    fn compare_by_prefix_as_java() {
        let abc = u("abc");
        assert_eq!(compare_array_by_prefix(Some(&u("ab")), 0, Some(&abc), 0), 0);
        assert_eq!(
            compare_array_by_prefix(Some(&u("xab")), 1, Some(&abc), 0),
            0
        );
        assert_eq!(
            compare_array_by_prefix(Some(&u("abcd")), 0, Some(&abc), 0),
            1
        );
        assert_eq!(
            compare_array_by_prefix(Some(&u("abd")), 0, Some(&abc), 0),
            1
        );
        assert_eq!(
            compare_array_by_prefix(Some(&u("abb")), 0, Some(&abc), 0),
            -1
        );
        assert_eq!(compare_array_by_prefix(None, 0, Some(&abc), 0), 0);
        assert_eq!(compare_array_by_prefix(Some(&abc), 0, None, 0), 1);
        assert_eq!(compare_array_by_prefix(Some(&abc), 3, None, 0), 0);
    }

    #[test]
    fn char_types() {
        use char_type::*;
        let t = |c: char| get_char_type(c as u16);
        assert_eq!(t('中'), HANZI);
        assert_eq!(t('a'), LETTER);
        assert_eq!(t('Z'), LETTER);
        assert_eq!(t('5'), DIGIT);
        assert_eq!(t('\u{3000}'), SPACE_LIKE);
        assert_eq!(t('。'), DELIMITER);
        assert_eq!(t('—'), DELIMITER);
        assert_eq!(t('!'), DELIMITER);
        assert_eq!(t('Ｂ'), FULLWIDTH_LETTER);
        assert_eq!(t('ｂ'), FULLWIDTH_LETTER);
        assert_eq!(t('３'), FULLWIDTH_DIGIT);
        assert_eq!(t('，'), DELIMITER);
        assert_eq!(t('é'), OTHER);
        assert_eq!(get_char_type(0xD800), SURROGATE);
        assert_eq!(SPACES.chars().count(), 5);
        assert_eq!(String::from_utf16(&STRING_CHAR_ARRAY).unwrap(), "未##串");
        assert_eq!(String::from_utf16(&NUMBER_CHAR_ARRAY).unwrap(), "未##数");
        assert_eq!(String::from_utf16(&START_CHAR_ARRAY).unwrap(), "始##始");
        assert_eq!(String::from_utf16(&END_CHAR_ARRAY).unwrap(), "末##末");
    }
}
