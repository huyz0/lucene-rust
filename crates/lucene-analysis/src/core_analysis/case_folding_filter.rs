//! Lucene 10.6's `org.apache.lucene.analysis.CaseFoldingFilter` (#16371),
//! with the pieces of lucene-core it calls: `util/SimpleCaseFolding.fold`,
//! `UnicodeUtil.foldCase` and `CharacterUtils.simpleCaseFold`. Not in the
//! 10.5.2 jars this port is pinned to: an additive port of branch_10_6.
//!
//! Unicode simple case folding (`CaseFolding.txt`'s `C` and `S` mappings) is
//! the JDK's `Character.toLowerCase(int)` except at the code points listed in
//! [`SIMPLE_CASE_FOLDING`], transcribed from Lucene's generated
//! `SimpleCaseFolding.java` (ICU4J 77.1, Unicode 16.0; Unicode data, see
//! `docs/licences.md`). Every other code point folds through
//! [`java_character::to_lower_case`] (JDK 25, Unicode 16.0), as Java's
//! `default` branch does.
//!
//! Differs: Java's `fold` is a `switch`; here the exceptions are a sorted
//! table searched by code point. `simpleCaseFold` rewrites the term's
//! UTF-16 buffer in place; a folded code point never leaves its plane, so
//! mapping the `String`'s code points is the same rewrite.

use crate::java_character;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

use super::filters::map_term_chars;

/// `SimpleCaseFolding`'s `case` labels: `(code point, its simple case
/// fold)` for the 225 code points whose fold is not
/// `Character.toLowerCase`, sorted by code point.
#[rustfmt::skip]
pub static SIMPLE_CASE_FOLDING: [(u32, u32); 225] = [
    (0x00B5, 0x03BC), (0x0130, 0x0130), (0x017F, 0x0073), (0x0345, 0x03B9),
    (0x03C2, 0x03C3), (0x03D0, 0x03B2), (0x03D1, 0x03B8), (0x03D5, 0x03C6),
    (0x03D6, 0x03C0), (0x03F0, 0x03BA), (0x03F1, 0x03C1), (0x03F5, 0x03B5),
    (0x13A0, 0x13A0), (0x13A1, 0x13A1), (0x13A2, 0x13A2), (0x13A3, 0x13A3),
    (0x13A4, 0x13A4), (0x13A5, 0x13A5), (0x13A6, 0x13A6), (0x13A7, 0x13A7),
    (0x13A8, 0x13A8), (0x13A9, 0x13A9), (0x13AA, 0x13AA), (0x13AB, 0x13AB),
    (0x13AC, 0x13AC), (0x13AD, 0x13AD), (0x13AE, 0x13AE), (0x13AF, 0x13AF),
    (0x13B0, 0x13B0), (0x13B1, 0x13B1), (0x13B2, 0x13B2), (0x13B3, 0x13B3),
    (0x13B4, 0x13B4), (0x13B5, 0x13B5), (0x13B6, 0x13B6), (0x13B7, 0x13B7),
    (0x13B8, 0x13B8), (0x13B9, 0x13B9), (0x13BA, 0x13BA), (0x13BB, 0x13BB),
    (0x13BC, 0x13BC), (0x13BD, 0x13BD), (0x13BE, 0x13BE), (0x13BF, 0x13BF),
    (0x13C0, 0x13C0), (0x13C1, 0x13C1), (0x13C2, 0x13C2), (0x13C3, 0x13C3),
    (0x13C4, 0x13C4), (0x13C5, 0x13C5), (0x13C6, 0x13C6), (0x13C7, 0x13C7),
    (0x13C8, 0x13C8), (0x13C9, 0x13C9), (0x13CA, 0x13CA), (0x13CB, 0x13CB),
    (0x13CC, 0x13CC), (0x13CD, 0x13CD), (0x13CE, 0x13CE), (0x13CF, 0x13CF),
    (0x13D0, 0x13D0), (0x13D1, 0x13D1), (0x13D2, 0x13D2), (0x13D3, 0x13D3),
    (0x13D4, 0x13D4), (0x13D5, 0x13D5), (0x13D6, 0x13D6), (0x13D7, 0x13D7),
    (0x13D8, 0x13D8), (0x13D9, 0x13D9), (0x13DA, 0x13DA), (0x13DB, 0x13DB),
    (0x13DC, 0x13DC), (0x13DD, 0x13DD), (0x13DE, 0x13DE), (0x13DF, 0x13DF),
    (0x13E0, 0x13E0), (0x13E1, 0x13E1), (0x13E2, 0x13E2), (0x13E3, 0x13E3),
    (0x13E4, 0x13E4), (0x13E5, 0x13E5), (0x13E6, 0x13E6), (0x13E7, 0x13E7),
    (0x13E8, 0x13E8), (0x13E9, 0x13E9), (0x13EA, 0x13EA), (0x13EB, 0x13EB),
    (0x13EC, 0x13EC), (0x13ED, 0x13ED), (0x13EE, 0x13EE), (0x13EF, 0x13EF),
    (0x13F0, 0x13F0), (0x13F1, 0x13F1), (0x13F2, 0x13F2), (0x13F3, 0x13F3),
    (0x13F4, 0x13F4), (0x13F5, 0x13F5), (0x13F8, 0x13F0), (0x13F9, 0x13F1),
    (0x13FA, 0x13F2), (0x13FB, 0x13F3), (0x13FC, 0x13F4), (0x13FD, 0x13F5),
    (0x1C80, 0x0432), (0x1C81, 0x0434), (0x1C82, 0x043E), (0x1C83, 0x0441),
    (0x1C84, 0x0442), (0x1C85, 0x0442), (0x1C86, 0x044A), (0x1C87, 0x0463),
    (0x1C88, 0xA64B), (0x1C89, 0x1C8A), (0x1E9B, 0x1E61), (0x1FBE, 0x03B9),
    (0x1FD3, 0x0390), (0x1FE3, 0x03B0), (0xA7CB, 0x0264), (0xA7CC, 0xA7CD),
    (0xA7DA, 0xA7DB), (0xA7DC, 0x019B), (0xAB70, 0x13A0), (0xAB71, 0x13A1),
    (0xAB72, 0x13A2), (0xAB73, 0x13A3), (0xAB74, 0x13A4), (0xAB75, 0x13A5),
    (0xAB76, 0x13A6), (0xAB77, 0x13A7), (0xAB78, 0x13A8), (0xAB79, 0x13A9),
    (0xAB7A, 0x13AA), (0xAB7B, 0x13AB), (0xAB7C, 0x13AC), (0xAB7D, 0x13AD),
    (0xAB7E, 0x13AE), (0xAB7F, 0x13AF), (0xAB80, 0x13B0), (0xAB81, 0x13B1),
    (0xAB82, 0x13B2), (0xAB83, 0x13B3), (0xAB84, 0x13B4), (0xAB85, 0x13B5),
    (0xAB86, 0x13B6), (0xAB87, 0x13B7), (0xAB88, 0x13B8), (0xAB89, 0x13B9),
    (0xAB8A, 0x13BA), (0xAB8B, 0x13BB), (0xAB8C, 0x13BC), (0xAB8D, 0x13BD),
    (0xAB8E, 0x13BE), (0xAB8F, 0x13BF), (0xAB90, 0x13C0), (0xAB91, 0x13C1),
    (0xAB92, 0x13C2), (0xAB93, 0x13C3), (0xAB94, 0x13C4), (0xAB95, 0x13C5),
    (0xAB96, 0x13C6), (0xAB97, 0x13C7), (0xAB98, 0x13C8), (0xAB99, 0x13C9),
    (0xAB9A, 0x13CA), (0xAB9B, 0x13CB), (0xAB9C, 0x13CC), (0xAB9D, 0x13CD),
    (0xAB9E, 0x13CE), (0xAB9F, 0x13CF), (0xABA0, 0x13D0), (0xABA1, 0x13D1),
    (0xABA2, 0x13D2), (0xABA3, 0x13D3), (0xABA4, 0x13D4), (0xABA5, 0x13D5),
    (0xABA6, 0x13D6), (0xABA7, 0x13D7), (0xABA8, 0x13D8), (0xABA9, 0x13D9),
    (0xABAA, 0x13DA), (0xABAB, 0x13DB), (0xABAC, 0x13DC), (0xABAD, 0x13DD),
    (0xABAE, 0x13DE), (0xABAF, 0x13DF), (0xABB0, 0x13E0), (0xABB1, 0x13E1),
    (0xABB2, 0x13E2), (0xABB3, 0x13E3), (0xABB4, 0x13E4), (0xABB5, 0x13E5),
    (0xABB6, 0x13E6), (0xABB7, 0x13E7), (0xABB8, 0x13E8), (0xABB9, 0x13E9),
    (0xABBA, 0x13EA), (0xABBB, 0x13EB), (0xABBC, 0x13EC), (0xABBD, 0x13ED),
    (0xABBE, 0x13EE), (0xABBF, 0x13EF), (0xFB05, 0xFB06), (0x10D50, 0x10D70),
    (0x10D51, 0x10D71), (0x10D52, 0x10D72), (0x10D53, 0x10D73), (0x10D54, 0x10D74),
    (0x10D55, 0x10D75), (0x10D56, 0x10D76), (0x10D57, 0x10D77), (0x10D58, 0x10D78),
    (0x10D59, 0x10D79), (0x10D5A, 0x10D7A), (0x10D5B, 0x10D7B), (0x10D5C, 0x10D7C),
    (0x10D5D, 0x10D7D), (0x10D5E, 0x10D7E), (0x10D5F, 0x10D7F), (0x10D60, 0x10D80),
    (0x10D61, 0x10D81), (0x10D62, 0x10D82), (0x10D63, 0x10D83), (0x10D64, 0x10D84),
    (0x10D65, 0x10D85),
];

/// `UnicodeUtil.foldCase(int)` (`SimpleCaseFolding.fold`): the Unicode
/// simple case fold of a code point.
pub fn fold_case(cp: u32) -> u32 {
    match SIMPLE_CASE_FOLDING.binary_search_by_key(&cp, |&(c, _)| c) {
        Ok(i) => SIMPLE_CASE_FOLDING[i].1,
        Err(_) => java_character::to_lower_case(cp),
    }
}

/// `CharacterUtils.simpleCaseFold` on one code point: ASCII `A`-`Z` set
/// bit 5, anything else [`fold_case`].
#[inline]
fn fold_char(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    char::from_u32(fold_case(u32::from(c))).unwrap_or(c)
}

/// `CharacterUtils.simpleCaseFold(char[], 0, length)` over a whole term, in
/// place.
pub fn simple_case_fold(term: &mut String) {
    map_term_chars(term, fold_char);
}

/// `org.apache.lucene.analysis.CaseFoldingFilter` (Lucene 10.6): applies
/// Unicode simple case folding to each term, without ICU.
pub struct CaseFoldingFilter<I> {
    input: I,
}

impl<I: TokenStream> CaseFoldingFilter<I> {
    /// `new CaseFoldingFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        CaseFoldingFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for CaseFoldingFilter<I> {
    crate::filter_input!();

    // Java: CaseFoldingFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input.increment_token()? {
            simple_case_fold(self.input.attributes_mut().term_mut());
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::{consume, Tokenizer};
    use crate::util::WhitespaceTokenizer;

    fn fold_str(s: &str) -> String {
        let mut t = s.to_string();
        simple_case_fold(&mut t);
        t
    }

    // TestCharacterUtils.testAsciiFoldMatchesToLowerCase
    #[test]
    fn ascii_fold_matches_to_lower_case() {
        for c in 0..128 {
            assert_eq!(fold_case(c), java_character::to_lower_case(c));
        }
    }

    // TestCharacterUtils.testFoldExceptions, plus a few of the table's
    // other kinds of row
    #[test]
    fn fold_exceptions() {
        assert_eq!(fold_case(0x00B5), 0x03BC);
        assert_eq!(fold_case(0x017F), 0x0073);
        assert_eq!(fold_case(0x03C2), 0x03C3);
        assert_eq!(fold_case(u32::from('0')), u32::from('0'));
        assert_eq!(fold_case(u32::from('a')), u32::from('a'));
        assert_eq!(fold_case(u32::from('A')), u32::from('a'));
        // Cherokee folds lowercase to uppercase, against `toLowerCase`
        assert_eq!(fold_case(0xAB70), 0x13A0);
        assert_eq!(fold_case(0x13A0), 0x13A0);
        assert_eq!(fold_case(0x0130), 0x0130);
        // a supplementary row, and the fallback past the table
        assert_eq!(fold_case(0x10D50), 0x10D70);
        assert_eq!(fold_case(0x10400), 0x10428);
    }

    // TestCharacterUtils.testFoldIdempotent
    #[test]
    fn fold_is_idempotent() {
        for c in 0..=0xFFFF {
            let folded = fold_case(c);
            assert_eq!(fold_case(folded), folded, "not idempotent at U+{c:04X}");
        }
    }

    #[test]
    fn the_table_is_sorted_and_stays_in_its_plane() {
        assert!(SIMPLE_CASE_FOLDING.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(SIMPLE_CASE_FOLDING
            .iter()
            .all(|&(c, f)| (c < 0x10000) == (f < 0x10000)));
    }

    // TestCharacterUtils.testSimpleCaseFold
    #[test]
    fn simple_case_fold_folds_final_sigma() {
        assert_eq!(fold_str("ABcΣσς"), "abcσσσ");
        assert_eq!(fold_str("abc"), "abc");
    }

    // TestCharacterUtils.testSimpleCaseFoldRandom: folding twice is folding
    // once -- here over every code point rather than random strings.
    #[test]
    fn simple_case_fold_twice_is_once() {
        let s: String = (0..=0x10FFFF).filter_map(char::from_u32).collect();
        let once = fold_str(&s);
        assert_eq!(fold_str(&once), once);
        assert_eq!(once.chars().count(), s.chars().count());
    }

    #[test]
    fn the_filter_folds_each_term() {
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new("STRAẞE ΟΔΟΣ ﬅ Ꭰꭰ µ ǅ")))
            .unwrap();
        let mut f = CaseFoldingFilter::new(t);
        let mut out = Vec::new();
        consume(&mut f, |a| {
            out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
        })
        .unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.0.as_str()).collect();
        assert_eq!(terms, ["straße", "οδοσ", "ﬆ", "ᎠᎠ", "μ", "ǆ"]);
        assert_eq!((out[1].1, out[1].2), (7, 11));
    }
}
