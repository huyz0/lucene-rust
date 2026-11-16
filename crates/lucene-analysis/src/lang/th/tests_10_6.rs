#![cfg(test)]
use super::*;
use crate::analyzer::collect_tokens;
use crate::reader::StrReader;
use crate::token_stream::Tokenizer;
use crate::util::WhitespaceTokenizer;
use crate::KeywordTokenizer;

fn ws(reader: Box<dyn CharReader>) -> WhitespaceTokenizer {
    let mut t = WhitespaceTokenizer::new();
    t.set_reader(reader).unwrap();
    t
}

fn ws_text(text: &str) -> WhitespaceTokenizer {
    ws(Box::new(StrReader::new(text)))
}

fn char_filter(text: &str) -> Box<dyn CharReader> {
    Box::new(ThaiCharFilter::new(
        Box::new(StrReader::new(text)) as Box<dyn CharReader>
    ))
}

/// `assertTokenStreamContents(ts, terms, starts, ends, finalOffset)`.
fn assert_ts(
    ts: &mut dyn TokenStream,
    terms: &[&str],
    starts: Option<&[i32]>,
    ends: Option<&[i32]>,
    final_offset: Option<i32>,
) {
    let got = collect_tokens(ts).unwrap();
    let t: Vec<&str> = got.tokens.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(t, terms);
    if let Some(starts) = starts {
        let s: Vec<i32> = got.tokens.iter().map(|t| t.start_offset).collect();
        assert_eq!(s, starts, "start offsets of {terms:?}");
    }
    if let Some(ends) = ends {
        let e: Vec<i32> = got.tokens.iter().map(|t| t.end_offset).collect();
        assert_eq!(e, ends, "end offsets of {terms:?}");
    }
    if let Some(f) = final_offset {
        assert_eq!(got.final_offset, f, "final offset of {terms:?}");
    }
    // every token here starts a position
    assert!(got.tokens.iter().all(|t| t.position_increment == 1));
}

fn check_char_filter(input: &str, term: &str, end: i32) {
    assert_ts(
        &mut ws(char_filter(input)),
        &[term],
        Some(&[0]),
        Some(&[end]),
        Some(end),
    );
}

// TestThaiCharFilter.testDoubleSaraE
#[test]
fn char_filter_double_sara_e() {
    check_char_filter("เเมว", "แมว", 4);
}

// TestThaiCharFilter.testSaraAmRecomposition
#[test]
fn char_filter_sara_am_recomposition() {
    check_char_filter("น\u{0E4D}\u{0E49}\u{0E32}", "น้ำ", 4);
    check_char_filter("ท\u{0E4D}\u{0E32}", "ทำ", 3);
}

// TestThaiCharFilter.testDuplicateDiacritics
#[test]
fn char_filter_duplicate_diacritics() {
    check_char_filter("ดีี", "ดี", 3);
    check_char_filter("ไม้้", "ไม้", 4);
}

// TestThaiCharFilter.testZeroWidth
#[test]
fn char_filter_zero_width() {
    check_char_filter("สวัส\u{200B}ดี", "สวัสดี", 7);
}

// TestThaiCharFilter.testToneReordering
#[test]
fn char_filter_tone_reordering() {
    check_char_filter("บ\u{0E48}\u{0E31}น", "บั่น", 4);
}

// TestThaiCharFilter.testLakkhangyao
#[test]
fn char_filter_lakkhangyao() {
    assert_ts(
        &mut ws(char_filter("ฤๅ ฦๅ กๅ")),
        &["ฤๅ", "ฦๅ", "กา"],
        Some(&[0, 3, 6]),
        Some(&[2, 5, 8]),
        Some(8),
    );
}

// TestThaiCharFilter.testEmpty
#[test]
fn char_filter_empty() {
    assert_ts(&mut ws(char_filter("")), &[], None, None, Some(0));
}

/// The rest of the char filter's paths, through `correct_offset`
/// directly: ZWJ and BOM, a removal after a pending Sara Am, the
/// leading `0` previous character, and reads one unit at a time.
#[test]
fn char_filter_offsets_and_small_reads() {
    // ก ZWJ BOM ํ ่ า ZWSP ข  ->  ก ่ ำ ข
    let text = "ก\u{200D}\u{FEFF}\u{0E4D}\u{0E48}\u{0E32}\u{200B}ข";
    let mut f = ThaiCharFilter::new(StrReader::new(text));
    let mut out = Vec::new();
    let mut one = [0u16; 1];
    while f.read(&mut one).unwrap() == 1 {
        out.push(one[0]);
    }
    assert_eq!(String::from_utf16(&out).unwrap(), "ก\u{0E48}\u{0E33}ข");
    let corrected: Vec<i32> = (0..=4).map(|o| f.correct_offset(o)).collect();
    assert_eq!(corrected, [0, 3, 4, 7, 8]);
    assert_eq!(f.read(&mut []).unwrap(), 0);
    f.close().unwrap();
}

/// `TestThaiCharFilter.testRandom`'s invariants (`checkRandomData`:
/// offsets within the input and never going backwards) over a fixed
/// pseudo-random Thai text.
#[test]
fn char_filter_offsets_stay_ordered_and_in_bounds() {
    const UNITS: [u16; 14] = [
        0x0E01, 0x0E24, 0x0E31, 0x0E32, 0x0E34, 0x0E40, 0x0E45, 0x0E47, 0x0E48, 0x0E4C, 0x0E4D,
        0x200B, 0xFEFF, 0x0020,
    ];
    let mut seed = 0x2545_F491_u32;
    for _ in 0..200 {
        let mut units = Vec::new();
        for _ in 0..(seed % 40) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            units.push(UNITS[(seed % UNITS.len() as u32) as usize]);
        }
        let text = String::from_utf16(&units).unwrap();
        let got = collect_tokens(&mut ws(char_filter(&text))).unwrap();
        let mut last = 0;
        for t in &got.tokens {
            assert!(
                last <= t.start_offset && t.start_offset <= t.end_offset,
                "{text:?}"
            );
            assert!(t.end_offset <= units.len() as i32, "{text:?}");
            last = t.start_offset;
        }
        assert!(got.final_offset <= units.len() as i32);
    }
}

fn check_normalizer(input: &str, output: &str) {
    assert_ts(
        &mut ThaiNormalizationFilter::new(ws_text(input)),
        &[output],
        None,
        None,
        None,
    );
}

// TestThaiNormalizer.testDoubleSaraE
#[test]
fn normalizer_double_sara_e() {
    check_normalizer(
        "\u{0E40}\u{0E40}\u{0E1B}\u{0E25}\u{0E01}",
        "\u{0E41}\u{0E1B}\u{0E25}\u{0E01}",
    );
}

// TestThaiNormalizer.testSaraAmDecomposition
#[test]
fn normalizer_sara_am_decomposition() {
    check_normalizer(
        "\u{0E17}\u{0E4D}\u{0E32}\u{0E07}\u{0E32}\u{0E19}",
        "\u{0E17}\u{0E33}\u{0E07}\u{0E32}\u{0E19}",
    );
    check_normalizer(
        "\u{0E19}\u{0E4D}\u{0E49}\u{0E32}",
        "\u{0E19}\u{0E49}\u{0E33}",
    );
    check_normalizer(
        "\u{0E19}\u{0E49}\u{0E4D}\u{0E32}",
        "\u{0E19}\u{0E49}\u{0E33}",
    );
}

// TestThaiNormalizer.testSaraAmWithFollowedTone
#[test]
fn normalizer_sara_am_with_followed_tone() {
    check_normalizer("\u{0E19}\u{0E33}\u{0E49}", "\u{0E19}\u{0E49}\u{0E33}");
}

// TestThaiNormalizer.testReorderToneAndAboveVowel
#[test]
fn normalizer_reorder_tone_and_above_vowel() {
    check_normalizer("\u{0E01}\u{0E49}\u{0E34}", "\u{0E01}\u{0E34}\u{0E49}");
}

// TestThaiNormalizer.testDuplicateVowelsAndTones
#[test]
fn normalizer_duplicate_vowels_and_tones() {
    check_normalizer("\u{0E14}\u{0E35}\u{0E35}", "\u{0E14}\u{0E35}");
    check_normalizer(
        "\u{0E41}\u{0E21}\u{0E49}\u{0E49}",
        "\u{0E41}\u{0E21}\u{0E49}",
    );
    check_normalizer("\u{0E19}\u{0E32}\u{0E32}", "\u{0E19}\u{0E32}");
}

// TestThaiNormalizer.testConsecutiveDifferentTones
#[test]
fn normalizer_consecutive_different_tones() {
    check_normalizer(
        "\u{0E44}\u{0E21}\u{0E48}\u{0E49}",
        "\u{0E44}\u{0E21}\u{0E49}",
    );
}

// TestThaiNormalizer.testZeroWidthCharacters
#[test]
fn normalizer_zero_width_characters() {
    check_normalizer(
        "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{200B}\u{0E44}\u{0E17}\u{0E22}",
        "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{0E44}\u{0E17}\u{0E22}",
    );
    check_normalizer(
        "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{200C}\u{0E44}\u{0E17}\u{0E22}",
        "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{0E44}\u{0E17}\u{0E22}",
    );
}

// TestThaiNormalizer.testLakkhangyao
#[test]
fn normalizer_lakkhangyao() {
    check_normalizer("\u{0E24}\u{0E45}", "\u{0E24}\u{0E45}");
    check_normalizer("\u{0E01}\u{0E45}", "\u{0E01}\u{0E32}");
}

// TestThaiNormalizer.testDanglingMark
#[test]
fn normalizer_dangling_mark() {
    check_normalizer("\u{0E49}\u{0E01}\u{0E32}", "\u{0E01}\u{0E32}");
}

// TestThaiNormalizer.testEmptyTerm
#[test]
fn normalizer_empty_term() {
    let mut t = KeywordTokenizer::new();
    t.set_reader(Box::new(StrReader::new(""))).unwrap();
    assert_ts(
        &mut ThaiNormalizationFilter::new(t),
        &[""],
        None,
        None,
        None,
    );
}

/// The normalizer's remaining branches on the units directly, and the
/// filter leaving a keyword alone.
#[test]
fn normalizer_edges_and_keywords() {
    let norm = |s: &str| {
        let mut u: Vec<u16> = s.encode_utf16().collect();
        let len = u.len();
        let n = ThaiNormalizer.stem(&mut u, len);
        String::from_utf16(&u[..n]).unwrap()
    };
    assert_eq!(norm(""), "");
    // only dangling marks and zero widths: nothing left
    assert_eq!(norm("\u{0E48}\u{200B}\u{0E4E}"), "");
    // Lakkhangyao first, after Lu, and a follow vowel before a tone
    assert_eq!(norm("\u{0E45}"), "\u{0E32}");
    assert_eq!(norm("\u{0E26}\u{0E45}"), "\u{0E26}\u{0E45}");
    assert_eq!(norm("\u{0E01}\u{0E30}\u{0E48}"), "\u{0E01}\u{0E48}\u{0E30}");
    // Thanthakhat before an above vowel
    assert_eq!(norm("\u{0E01}\u{0E4C}\u{0E34}"), "\u{0E01}\u{0E34}\u{0E4C}");

    let a = crate::miscellaneous::SetKeywordMarkerFilter::new(
        ws_text("\u{0E14}\u{0E35}\u{0E35}"),
        Arc::new(CharArraySet::from_words(
            ["\u{0E14}\u{0E35}\u{0E35}"],
            false,
        )),
    );
    assert_ts(
        &mut ThaiNormalizationFilter::new(a),
        &["\u{0E14}\u{0E35}\u{0E35}"],
        None,
        None,
        None,
    );
}

fn repeat(text: &str) -> ThaiRepeatFilter<WhitespaceTokenizer> {
    ThaiRepeatFilter::new(ws_text(text))
}

// TestThaiRepeatFilter.testStandaloneMaiyamok
#[test]
fn repeat_standalone_maiyamok() {
    assert_ts(
        &mut repeat("เร็ว ๆ"),
        &["เร็ว", "เร็ว"],
        Some(&[0, 5]),
        Some(&[4, 6]),
        None,
    );
}

// TestThaiRepeatFilter.testAttachedMaiyamok
#[test]
fn repeat_attached_maiyamok() {
    assert_ts(
        &mut repeat("เร็วๆ"),
        &["เร็ว", "เร็ว"],
        Some(&[0, 4]),
        Some(&[4, 5]),
        None,
    );
}

// TestThaiRepeatFilter.testMultipleMaiyamok
#[test]
fn repeat_multiple_maiyamok() {
    assert_ts(
        &mut repeat("มากๆๆ"),
        &["มาก", "มาก", "มาก"],
        Some(&[0, 3, 4]),
        Some(&[3, 4, 5]),
        None,
    );
    assert_ts(
        &mut repeat("มาก ๆ ๆ"),
        &["มาก", "มาก", "มาก"],
        Some(&[0, 4, 6]),
        Some(&[3, 5, 7]),
        None,
    );
}

// TestThaiRepeatFilter.testInSentence
#[test]
fn repeat_in_sentence() {
    assert_ts(
        &mut repeat("เด็ก ๆ กำลัง วิ่ง เล่น"),
        &["เด็ก", "เด็ก", "กำลัง", "วิ่ง", "เล่น"],
        Some(&[0, 5, 7, 13, 18]),
        Some(&[4, 6, 12, 17, 22]),
        None,
    );
}

// TestThaiRepeatFilter.testLeadingDanglingMaiyamok
#[test]
fn repeat_leading_dangling_maiyamok() {
    assert_ts(&mut repeat("ๆ ครับ"), &["ครับ"], None, None, None);
}

// TestThaiRepeatFilter.testEmpty
#[test]
fn repeat_empty() {
    assert_ts(&mut repeat(""), &[], None, None, None);
}

/// A standalone run of several marks, more marks than the initial
/// capacity, and a reused filter forgetting the previous term.
#[test]
fn repeat_long_runs_and_reset() {
    assert_ts(
        &mut repeat("ดี ๆๆๆ"),
        &["ดี", "ดี", "ดี", "ดี"],
        Some(&[0, 3, 4, 5]),
        Some(&[2, 4, 5, 6]),
        None,
    );
    assert_ts(
        &mut repeat("ดีๆๆๆๆๆ"),
        &["ดี", "ดี", "ดี", "ดี", "ดี", "ดี"],
        Some(&[0, 2, 3, 4, 5, 6]),
        Some(&[2, 3, 4, 5, 6, 7]),
        None,
    );
    let mut f = repeat("ดี");
    assert_ts(&mut f, &["ดี"], None, None, None);
    f.input_mut()
        .set_reader(Box::new(StrReader::new("ๆ")))
        .unwrap();
    assert_ts(&mut f, &[], None, None, None);
}
