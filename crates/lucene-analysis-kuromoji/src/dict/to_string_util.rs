//! `ja.dict.ToStringUtil`: English glosses of IPADIC's part-of-speech and
//! inflection names (for attribute reflection) and the modified-Hepburn
//! romanization of katakana (`JapaneseReadingFormFilter`'s `useRomaji`).
//!
//! The tables and the romanization's `switch` are Java's, converted
//! mechanically (one `match` arm per `case` group, `break main` ending the
//! arm); Java's `HashMap` lookups become sorted-slice searches.

/// `pos_translations`.
static POS_TRANSLATIONS: &[(&str, &str)] = &[
    ("その他", "other"),
    ("その他-間投", "other-interjection"),
    ("フィラー", "filler"),
    ("副詞", "adverb"),
    ("副詞-一般", "adverb-misc"),
    ("副詞-助詞類接続", "adverb-particle_conjunction"),
    ("助動詞", "auxiliary-verb"),
    ("助詞", "particle"),
    ("助詞-並立助詞", "particle-coordinate"),
    ("助詞-係助詞", "particle-dependency"),
    ("助詞-副助詞", "particle-adverbial"),
    (
        "助詞-副助詞／並立助詞／終助詞",
        "particle-adverbial/conjunctive/final",
    ),
    ("助詞-副詞化", "particle-adnominalizer"),
    ("助詞-接続助詞", "particle-conjunctive"),
    ("助詞-格助詞", "particle-case"),
    ("助詞-格助詞-一般", "particle-case-misc"),
    ("助詞-格助詞-引用", "particle-case-quote"),
    ("助詞-格助詞-連語", "particle-case-compound"),
    ("助詞-特殊", "particle-special"),
    ("助詞-終助詞", "particle-final"),
    ("助詞-連体化", "particle-adnominalizer"),
    ("助詞-間投助詞", "particle-interjective"),
    ("動詞", "verb"),
    ("動詞-接尾", "verb-suffix"),
    ("動詞-自立", "verb-main"),
    ("動詞-非自立", "verb-auxiliary"),
    ("名詞", "noun"),
    ("名詞-サ変接続", "noun-verbal"),
    ("名詞-ナイ形容詞語幹", "noun-nai_adjective"),
    ("名詞-一般", "noun-common"),
    ("名詞-代名詞", "noun-pronoun"),
    ("名詞-代名詞-一般", "noun-pronoun-misc"),
    ("名詞-代名詞-縮約", "noun-pronoun-contraction"),
    ("名詞-副詞可能", "noun-adverbial"),
    ("名詞-動詞非自立的", "noun-verbal_aux"),
    ("名詞-固有名詞", "noun-proper"),
    ("名詞-固有名詞-一般", "noun-proper-misc"),
    ("名詞-固有名詞-人名", "noun-proper-person"),
    ("名詞-固有名詞-人名-一般", "noun-proper-person-misc"),
    ("名詞-固有名詞-人名-名", "noun-proper-person-given_name"),
    ("名詞-固有名詞-人名-姓", "noun-proper-person-surname"),
    ("名詞-固有名詞-地域", "noun-proper-place"),
    ("名詞-固有名詞-地域-一般", "noun-proper-place-misc"),
    ("名詞-固有名詞-地域-国", "noun-proper-place-country"),
    ("名詞-固有名詞-組織", "noun-proper-organization"),
    ("名詞-引用文字列", "noun-quotation"),
    ("名詞-形容動詞語幹", "noun-adjective-base"),
    ("名詞-接尾", "noun-suffix"),
    ("名詞-接尾-サ変接続", "noun-suffix-verbal"),
    ("名詞-接尾-一般", "noun-suffix-misc"),
    ("名詞-接尾-人名", "noun-suffix-person"),
    ("名詞-接尾-副詞可能", "noun-suffix-adverbial"),
    ("名詞-接尾-助動詞語幹", "noun-suffix-aux"),
    ("名詞-接尾-助数詞", "noun-suffix-classifier"),
    ("名詞-接尾-地域", "noun-suffix-place"),
    ("名詞-接尾-形容動詞語幹", "noun-suffix-adjective-base"),
    ("名詞-接尾-特殊", "noun-suffix-special"),
    ("名詞-接続詞的", "noun-suffix-conjunctive"),
    ("名詞-数", "noun-numeric"),
    ("名詞-特殊", "noun-special"),
    ("名詞-特殊-助動詞語幹", "noun-special-aux"),
    ("名詞-非自立", "noun-affix"),
    ("名詞-非自立-一般", "noun-affix-misc"),
    ("名詞-非自立-副詞可能", "noun-affix-adverbial"),
    ("名詞-非自立-助動詞語幹", "noun-affix-aux"),
    ("名詞-非自立-形容動詞語幹", "noun-affix-adjective-base"),
    ("形容詞", "adjective"),
    ("形容詞-接尾", "adjective-suffix"),
    ("形容詞-自立", "adjective-main"),
    ("形容詞-非自立", "adjective-auxiliary"),
    ("感動詞", "interjection"),
    ("接続詞", "conjunction"),
    ("接頭詞", "prefix"),
    ("接頭詞-動詞接続", "prefix-verbal"),
    ("接頭詞-名詞接続", "prefix-nominal"),
    ("接頭詞-形容詞接続", "prefix-adjectival"),
    ("接頭詞-数接続", "prefix-numerical"),
    ("未知語", "unknown"),
    ("記号", "symbol"),
    ("記号-アルファベット", "symbol-alphabetic"),
    ("記号-一般", "symbol-misc"),
    ("記号-句点", "symbol-period"),
    ("記号-括弧閉", "symbol-close_bracket"),
    ("記号-括弧開", "symbol-open_bracket"),
    ("記号-空白", "symbol-space"),
    ("記号-読点", "symbol-comma"),
    ("語断片", "fragment"),
    ("連体詞", "adnominal"),
    ("非言語音", "non-verbal"),
];

/// `infl_type_translations`.
static INFL_TYPE_TRANSLATIONS: &[(&str, &str)] = &[
    ("*", "*"),
    ("カ変・クル", "kuru-kana"),
    ("カ変・来ル", "kuru-kanji"),
    ("サ変・−スル", "irregular-suffix-suru"),
    ("サ変・−ズル", "irregular-suffix-zuru"),
    ("サ変・スル", "irregular-suru"),
    ("ラ変", "irregular-cons-r"),
    ("一段", "1-row"),
    ("一段・クレル", "1-row-kureru"),
    ("一段・得ル", "1-row-eru"),
    ("上二・ダ行", "2-row-upper-cons-d"),
    ("上二・ハ行", "2-row-upper-cons-h"),
    ("下二・カ行", "2-row-lower-cons-k"),
    ("下二・ガ行", "2-row-lower-cons-g"),
    ("下二・タ行", "2-row-lower-cons-t"),
    ("下二・ダ行", "2-row-lower-cons-d"),
    ("下二・ハ行", "2-row-lower-cons-h"),
    ("下二・マ行", "2-row-lower-cons-m"),
    ("下二・得", "2-row-lower-u"),
    ("不変化型", "non-inflectional"),
    ("五段・カ行イ音便", "5-row-cons-k-i-onbin"),
    ("五段・カ行促音便", "5-row-cons-k-cons-onbin"),
    ("五段・カ行促音便ユク", "5-row-cons-k-cons-onbin-yuku"),
    ("五段・ガ行", "5-row-cons-g"),
    ("五段・サ行", "5-row-cons-s"),
    ("五段・タ行", "5-row-cons-t"),
    ("五段・ナ行", "5-row-cons-n"),
    ("五段・バ行", "5-row-cons-b"),
    ("五段・マ行", "5-row-cons-m"),
    ("五段・ラ行", "5-row-cons-r"),
    ("五段・ラ行アル", "5-row-aru"),
    ("五段・ラ行特殊", "5-row-cons-r-special"),
    ("五段・ワ行ウ音便", "5-row-cons-w-u-onbin"),
    ("五段・ワ行促音便", "5-row-cons-w-cons-onbin"),
    ("四段・サ行", "4-row-cons-s"),
    ("四段・タ行", "4-row-cons-t"),
    ("四段・ハ行", "4-row-cons-h"),
    ("四段・バ行", "4-row-cons-b"),
    ("形容詞・アウオ段", "adj-group-a-o-u"),
    ("形容詞・イイ", "adj-group-ii"),
    ("形容詞・イ段", "adj-group-i"),
    ("文語・キ", "classical-ki"),
    ("文語・ケリ", "classical-keri"),
    ("文語・ゴトシ", "classical-gotoshi"),
    ("文語・ナリ", "classical-nari"),
    ("文語・ベシ", "classical-beshi"),
    ("文語・マジ", "classical-maji"),
    ("文語・リ", "classical-ri"),
    ("文語・ル", "classical-ru"),
    ("特殊・ジャ", "special-ja"),
    ("特殊・タ", "special-da"),
    ("特殊・タイ", "special-tai"),
    ("特殊・ダ", "special-ta"),
    ("特殊・デス", "special-desu"),
    ("特殊・ナイ", "special-nai"),
    ("特殊・ヌ", "special-nu"),
    ("特殊・マス", "special-masu"),
    ("特殊・ヤ", "special-ya"),
];

/// `infl_form_translations`.
static INFL_FORM_TRANSLATIONS: &[(&str, &str)] = &[
    ("*", "*"),
    ("ガル接続", "garu-connection"),
    ("仮定形", "subjunctive"),
    ("仮定縮約１", "conditional-contracted-1"),
    ("仮定縮約２", "conditional-contracted-2"),
    ("体言接続", "uninflected-connection"),
    ("体言接続特殊", "adnominal-special"),
    ("体言接続特殊２", "uninflected-special-connection-2"),
    ("命令ｅ", "imperative-e"),
    ("命令ｉ", "imperative-i"),
    ("命令ｒｏ", "imperative-ro"),
    ("命令ｙｏ", "imperative-yo"),
    ("基本形", "base"),
    ("基本形-促音便", "base-onbin"),
    ("文語基本形", "classical-base"),
    ("未然ウ接続", "imperfective-u-connection"),
    ("未然ヌ接続", "imperfective-nu-connection"),
    ("未然レル接続", "imperfective-reru-connection"),
    ("未然形", "imperfective"),
    ("未然特殊", "imperfective-special"),
    ("現代基本形", "modern-base"),
    ("連用ゴザイ接続", "conjunctive-gozai-connection"),
    ("連用タ接続", "conjunctive-ta-connection"),
    ("連用テ接続", "conjunctive-te-connection"),
    ("連用デ接続", "conjunctive-de-connection"),
    ("連用ニ接続", "conjunctive-ni-connection"),
    ("連用形", "conjunctive"),
    ("音便基本形", "onbin-base"),
];

fn lookup(table: &'static [(&'static str, &'static str)], s: &str) -> Option<&'static str> {
    table
        .binary_search_by(|(k, _)| (*k).cmp(s))
        .ok()
        .map(|i| table[i].1)
}

/// `getPOSTranslation(s)`.
pub fn pos_translation(s: &str) -> Option<&'static str> {
    lookup(POS_TRANSLATIONS, s)
}

/// `getInflectionTypeTranslation(s)`.
pub fn inflection_type_translation(s: &str) -> Option<&'static str> {
    lookup(INFL_TYPE_TRANSLATIONS, s)
}

/// `getInflectedFormTranslation(s)`.
pub fn inflected_form_translation(s: &str) -> Option<&'static str> {
    lookup(INFL_FORM_TRANSLATIONS, s)
}

/// The romanization's output: UTF-16, so an unpaired surrogate passes
/// through as Java appends it.
struct Out(Vec<u16>);

impl Out {
    fn push(&mut self, c: char) {
        let mut b = [0u16; 2];
        self.0.extend_from_slice(c.encode_utf16(&mut b));
    }
    fn push_str(&mut self, s: &str) {
        self.0.extend(s.encode_utf16());
    }
    fn push_unit(&mut self, u: u16) {
        self.0.push(u);
    }
}

/// `getRomanization(s)` on UTF-16 units.
// ARITH: `i` indexes `s` (a slice, so at most `isize::MAX` units); each
// step adds at most 3 and the loop ends once `i >= len`, so `i + 2` and
// `i += 2` cannot overflow.
#[allow(clippy::arithmetic_side_effects)]
pub fn romanization_utf16(s: &[u16]) -> Vec<u16> {
    let mut out = Out(Vec::with_capacity(s.len().saturating_mul(2)));
    // A unit as a `char` for the case labels; a surrogate matches none.
    let c = |i: usize| {
        s.get(i)
            .and_then(|&u| char::from_u32(u32::from(u)))
            .unwrap_or('\u{FFFF}')
    };
    let len = s.len();
    let mut i = 0usize;
    while i < len {
        // maximum lookahead: 3
        let unit = s[i];
        let ch = c(i);
        let ch2 = if i + 1 < len { c(i + 1) } else { '\0' };
        let ch3 = if i + 2 < len { c(i + 2) } else { '\0' };
        match ch {
            'ッ' => match ch2 {
                'カ' | 'キ' | 'ク' | 'ケ' | 'コ' => {
                    out.push('k');
                }
                'サ' | 'シ' | 'ス' | 'セ' | 'ソ' => {
                    out.push('s');
                }
                'タ' | 'チ' | 'ツ' | 'テ' | 'ト' => {
                    out.push('t');
                }
                'パ' | 'ピ' | 'プ' | 'ペ' | 'ポ' => {
                    out.push('p');
                }
                _ => {}
            },
            'ア' => {
                out.push('a');
            }
            'イ' => {
                if ch2 == 'ィ' {
                    out.push_str("yi");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("ye");
                    i += 1;
                } else {
                    out.push('i');
                }
            }
            'ウ' => match ch2 {
                'ァ' => {
                    out.push_str("wa");
                    i += 1;
                }
                'ィ' => {
                    out.push_str("wi");
                    i += 1;
                }
                'ゥ' => {
                    out.push_str("wu");
                    i += 1;
                }
                'ェ' => {
                    out.push_str("we");
                    i += 1;
                }
                'ォ' => {
                    out.push_str("wo");
                    i += 1;
                }
                'ュ' => {
                    out.push_str("wyu");
                    i += 1;
                }
                _ => {
                    out.push('u');
                }
            },
            'エ' => {
                out.push('e');
            }
            'オ' => {
                if ch2 == 'ウ' {
                    out.push('ō');
                    i += 1;
                } else {
                    out.push('o');
                }
            }
            'カ' => {
                out.push_str("ka");
            }
            'キ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("kyō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("kyū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("kya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("kyo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("kyu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("kye");
                    i += 1;
                } else {
                    out.push_str("ki");
                }
            }
            'ク' => match ch2 {
                'ァ' => {
                    out.push_str("kwa");
                    i += 1;
                }
                'ィ' => {
                    out.push_str("kwi");
                    i += 1;
                }
                'ェ' => {
                    out.push_str("kwe");
                    i += 1;
                }
                'ォ' => {
                    out.push_str("kwo");
                    i += 1;
                }
                'ヮ' => {
                    out.push_str("kwa");
                    i += 1;
                }
                _ => {
                    out.push_str("ku");
                }
            },
            'ケ' => {
                out.push_str("ke");
            }
            'コ' => {
                if ch2 == 'ウ' {
                    out.push_str("kō");
                    i += 1;
                } else {
                    out.push_str("ko");
                }
            }
            'サ' => {
                out.push_str("sa");
            }
            'シ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("shō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("shū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("sha");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("sho");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("shu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("she");
                    i += 1;
                } else {
                    out.push_str("shi");
                }
            }
            'ス' => {
                if ch2 == 'ィ' {
                    out.push_str("si");
                    i += 1;
                } else {
                    out.push_str("su");
                }
            }
            'セ' => {
                out.push_str("se");
            }
            'ソ' => {
                if ch2 == 'ウ' {
                    out.push_str("sō");
                    i += 1;
                } else {
                    out.push_str("so");
                }
            }
            'タ' => {
                out.push_str("ta");
            }
            'チ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("chō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("chū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("cha");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("cho");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("chu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("che");
                    i += 1;
                } else {
                    out.push_str("chi");
                }
            }
            'ツ' => {
                if ch2 == 'ァ' {
                    out.push_str("tsa");
                    i += 1;
                } else if ch2 == 'ィ' {
                    out.push_str("tsi");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("tse");
                    i += 1;
                } else if ch2 == 'ォ' {
                    out.push_str("tso");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("tsyu");
                    i += 1;
                } else {
                    out.push_str("tsu");
                }
            }
            'テ' => {
                if ch2 == 'ィ' {
                    out.push_str("ti");
                    i += 1;
                } else if ch2 == 'ゥ' {
                    out.push_str("tu");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("tyu");
                    i += 1;
                } else {
                    out.push_str("te");
                }
            }
            'ト' => {
                if ch2 == 'ウ' {
                    out.push_str("tō");
                    i += 1;
                } else if ch2 == 'ゥ' {
                    out.push_str("tu");
                    i += 1;
                } else {
                    out.push_str("to");
                }
            }
            'ナ' => {
                out.push_str("na");
            }
            'ニ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("nyō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("nyū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("nya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("nyo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("nyu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("nye");
                    i += 1;
                } else {
                    out.push_str("ni");
                }
            }
            'ヌ' => {
                out.push_str("nu");
            }
            'ネ' => {
                out.push_str("ne");
            }
            'ノ' => {
                if ch2 == 'ウ' {
                    out.push_str("nō");
                    i += 1;
                } else {
                    out.push_str("no");
                }
            }
            'ハ' => {
                out.push_str("ha");
            }
            'ヒ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("hyō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("hyū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("hya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("hyo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("hyu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("hye");
                    i += 1;
                } else {
                    out.push_str("hi");
                }
            }
            'フ' => {
                if ch2 == 'ャ' {
                    out.push_str("fya");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("fyu");
                    i += 1;
                } else if ch2 == 'ィ' && ch3 == 'ェ' {
                    out.push_str("fye");
                    i += 2;
                } else if ch2 == 'ョ' {
                    out.push_str("fyo");
                    i += 1;
                } else if ch2 == 'ァ' {
                    out.push_str("fa");
                    i += 1;
                } else if ch2 == 'ィ' {
                    out.push_str("fi");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("fe");
                    i += 1;
                } else if ch2 == 'ォ' {
                    out.push_str("fo");
                    i += 1;
                } else {
                    out.push_str("fu");
                }
            }
            'ヘ' => {
                out.push_str("he");
            }
            'ホ' => {
                if ch2 == 'ウ' {
                    out.push_str("hō");
                    i += 1;
                } else if ch2 == 'ゥ' {
                    out.push_str("hu");
                    i += 1;
                } else {
                    out.push_str("ho");
                }
            }
            'マ' => {
                out.push_str("ma");
            }
            'ミ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("myō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("myū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("mya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("myo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("myu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("mye");
                    i += 1;
                } else {
                    out.push_str("mi");
                }
            }
            'ム' => {
                out.push_str("mu");
            }
            'メ' => {
                out.push_str("me");
            }
            'モ' => {
                if ch2 == 'ウ' {
                    out.push_str("mō");
                    i += 1;
                } else {
                    out.push_str("mo");
                }
            }
            'ヤ' => {
                out.push_str("ya");
            }
            'ユ' => {
                out.push_str("yu");
            }
            'ヨ' => {
                if ch2 == 'ウ' {
                    out.push_str("yō");
                    i += 1;
                } else {
                    out.push_str("yo");
                }
            }
            'ラ' => {
                if ch2 == '゜' {
                    out.push_str("la");
                    i += 1;
                } else {
                    out.push_str("ra");
                }
            }
            'リ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("ryō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("ryū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("rya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("ryo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("ryu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("rye");
                    i += 1;
                } else if ch2 == '゜' {
                    out.push_str("li");
                    i += 1;
                } else {
                    out.push_str("ri");
                }
            }
            'ル' => {
                if ch2 == '゜' {
                    out.push_str("lu");
                    i += 1;
                } else {
                    out.push_str("ru");
                }
            }
            'レ' => {
                if ch2 == '゜' {
                    out.push_str("le");
                    i += 1;
                } else {
                    out.push_str("re");
                }
            }
            'ロ' => {
                if ch2 == 'ウ' {
                    out.push_str("rō");
                    i += 1;
                } else if ch2 == '゜' {
                    out.push_str("lo");
                    i += 1;
                } else {
                    out.push_str("ro");
                }
            }
            'ワ' => {
                out.push_str("wa");
            }
            'ヰ' => {
                out.push_str("i");
            }
            'ヱ' => {
                out.push_str("e");
            }
            'ヲ' => {
                out.push_str("o");
            }
            'ン' => match ch2 {
                'バ' | 'ビ' | 'ブ' | 'ベ' | 'ボ' | 'パ' | 'ピ' | 'プ' | 'ペ' | 'ポ' | 'マ'
                | 'ミ' | 'ム' | 'メ' | 'モ' => {
                    out.push('m');
                }
                'ヤ' | 'ユ' | 'ヨ' | 'ア' | 'イ' | 'ウ' | 'エ' | 'オ' => {
                    out.push_str("n'");
                }
                _ => {
                    out.push_str("n");
                }
            },
            'ガ' => {
                out.push_str("ga");
            }
            'ギ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("gyō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("gyū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("gya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("gyo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("gyu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("gye");
                    i += 1;
                } else {
                    out.push_str("gi");
                }
            }
            'グ' => match ch2 {
                'ァ' => {
                    out.push_str("gwa");
                    i += 1;
                }
                'ィ' => {
                    out.push_str("gwi");
                    i += 1;
                }
                'ェ' => {
                    out.push_str("gwe");
                    i += 1;
                }
                'ォ' => {
                    out.push_str("gwo");
                    i += 1;
                }
                'ヮ' => {
                    out.push_str("gwa");
                    i += 1;
                }
                _ => {
                    out.push_str("gu");
                }
            },
            'ゲ' => {
                out.push_str("ge");
            }
            'ゴ' => {
                if ch2 == 'ウ' {
                    out.push_str("gō");
                    i += 1;
                } else {
                    out.push_str("go");
                }
            }
            'ザ' => {
                out.push_str("za");
            }
            'ジ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("jō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("jū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("ja");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("jo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("ju");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("je");
                    i += 1;
                } else {
                    out.push_str("ji");
                }
            }
            'ズ' => {
                if ch2 == 'ィ' {
                    out.push_str("zi");
                    i += 1;
                } else {
                    out.push_str("zu");
                }
            }
            'ゼ' => {
                out.push_str("ze");
            }
            'ゾ' => {
                if ch2 == 'ウ' {
                    out.push_str("zō");
                    i += 1;
                } else {
                    out.push_str("zo");
                }
            }
            'ダ' => {
                out.push_str("da");
            }
            // TODO: investigate all this
            'ヂ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("jō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("jū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("ja");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("jo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("ju");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("je");
                    i += 1;
                } else {
                    out.push_str("ji");
                }
            }
            'ヅ' => {
                out.push_str("zu");
            }
            'デ' => {
                if ch2 == 'ィ' {
                    out.push_str("di");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("dyu");
                    i += 1;
                } else {
                    out.push_str("de");
                }
            }
            'ド' => {
                if ch2 == 'ウ' {
                    out.push_str("dō");
                    i += 1;
                } else if ch2 == 'ゥ' {
                    out.push_str("du");
                    i += 1;
                } else {
                    out.push_str("do");
                }
            }
            'バ' => {
                out.push_str("ba");
            }
            'ビ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("byō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("byū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("bya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("byo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("byu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("bye");
                    i += 1;
                } else {
                    out.push_str("bi");
                }
            }
            'ブ' => {
                out.push_str("bu");
            }
            'ベ' => {
                out.push_str("be");
            }
            'ボ' => {
                if ch2 == 'ウ' {
                    out.push_str("bō");
                    i += 1;
                } else {
                    out.push_str("bo");
                }
            }
            'パ' => {
                out.push_str("pa");
            }
            'ピ' => {
                if ch2 == 'ョ' && ch3 == 'ウ' {
                    out.push_str("pyō");
                    i += 2;
                } else if ch2 == 'ュ' && ch3 == 'ウ' {
                    out.push_str("pyū");
                    i += 2;
                } else if ch2 == 'ャ' {
                    out.push_str("pya");
                    i += 1;
                } else if ch2 == 'ョ' {
                    out.push_str("pyo");
                    i += 1;
                } else if ch2 == 'ュ' {
                    out.push_str("pyu");
                    i += 1;
                } else if ch2 == 'ェ' {
                    out.push_str("pye");
                    i += 1;
                } else {
                    out.push_str("pi");
                }
            }
            'プ' => {
                out.push_str("pu");
            }
            'ペ' => {
                out.push_str("pe");
            }
            'ポ' => {
                if ch2 == 'ウ' {
                    out.push_str("pō");
                    i += 1;
                } else {
                    out.push_str("po");
                }
            }
            'ヷ' => {
                out.push_str("va");
            }
            'ヸ' => {
                out.push_str("vi");
            }
            'ヹ' => {
                out.push_str("ve");
            }
            'ヺ' => {
                out.push_str("vo");
            }
            'ヴ' => {
                if ch2 == 'ィ' && ch3 == 'ェ' {
                    out.push_str("vye");
                    i += 2;
                } else {
                    out.push('v');
                }
            }
            'ァ' => {
                out.push('a');
            }
            'ィ' => {
                out.push('i');
            }
            'ゥ' => {
                out.push('u');
            }
            'ェ' => {
                out.push('e');
            }
            'ォ' => {
                out.push('o');
            }
            'ヮ' => {
                out.push_str("wa");
            }
            'ャ' => {
                out.push_str("ya");
            }
            'ュ' => {
                out.push_str("yu");
            }
            'ョ' => {
                out.push_str("yo");
            }
            'ー' => {}
            _ => {
                out.push_unit(unit);
            }
        }

        i += 1;
    }
    out.0
}

/// `getRomanization(s)`.
pub fn romanization(s: &str) -> String {
    String::from_utf16_lossy(&romanization_utf16(&s.encode_utf16().collect::<Vec<_>>()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translations_and_romanization() {
        assert_eq!(pos_translation("名詞"), Some("noun"));
        assert_eq!(pos_translation("x"), None);
        assert!(inflection_type_translation("五段・カ行イ音便").is_some());
        assert!(inflected_form_translation("基本形").is_some());
        assert_eq!(romanization("トウキョウ"), "tōkyō");
        assert_eq!(romanization("ガッコウ"), "gakkō");
        assert_eq!(romanization("シンブン"), "shimbun");
        assert_eq!(romanization("abc"), "abc");
        assert_eq!(
            romanization_utf16(&[0xD800, 0x30A2]),
            [0xD800, u16::from(b'a')]
        );
    }
}
