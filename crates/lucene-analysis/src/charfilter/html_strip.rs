//! `org.apache.lucene.analysis.charfilter.HTMLStripCharFilter`: the JFlex
//! scanner (over its extracted tables, see [`crate::util::jflex`]) with its
//! user code and every action ported line for line.
//!
//! Java's `outputSegment` is a reference to either `inputSegment` or
//! `entitySegment`; here it is [`Out`], which of the two it names.

use std::collections::HashMap;
use std::sync::LazyLock;

use super::OffsetCorrections;
use crate::reader::{CharFilter, CharReader};
use crate::util::jflex::{JFlexScanner, JFlexTables, Scan};
use crate::{AnalysisError, CharArraySet};

static TABLES: LazyLock<JFlexTables> = LazyLock::new(|| {
    JFlexTables::load(include_bytes!("html_strip_tables.bin.z"))
        .expect("the generated HTMLStripCharFilter tables")
});

const ZZ_BUFFERSIZE: usize = 16384;
const YYINITIAL: usize = 0;
const AMPERSAND: usize = 2;
const NUMERIC_CHARACTER: usize = 4;
const CHARACTER_REFERENCE_TAIL: usize = 6;
const LEFT_ANGLE_BRACKET: usize = 8;
const BANG: usize = 10;
const COMMENT: usize = 12;
const SCRIPT: usize = 14;
const SCRIPT_COMMENT: usize = 16;
const LEFT_ANGLE_BRACKET_SLASH: usize = 18;
const LEFT_ANGLE_BRACKET_SPACE: usize = 20;
const CDATA: usize = 22;
const SERVER_SIDE_INCLUDE: usize = 24;
const SINGLE_QUOTED_STRING: usize = 26;
const DOUBLE_QUOTED_STRING: usize = 28;
const END_TAG_TAIL_INCLUDE: usize = 30;
const END_TAG_TAIL_EXCLUDE: usize = 32;
const END_TAG_TAIL_SUBSTITUTE: usize = 34;
const START_TAG_TAIL_INCLUDE: usize = 36;
const START_TAG_TAIL_EXCLUDE: usize = 38;
const START_TAG_TAIL_SUBSTITUTE: usize = 40;
const STYLE: usize = 42;
const STYLE_COMMENT: usize = 44;

const INITIAL_INPUT_SEGMENT_SIZE: usize = 1024;
const BLOCK_LEVEL_START_TAG_REPLACEMENT: u16 = b'\n' as u16;
const BLOCK_LEVEL_END_TAG_REPLACEMENT: u16 = b'\n' as u16;
const BR_START_TAG_REPLACEMENT: u16 = b'\n' as u16;
const BR_END_TAG_REPLACEMENT: u16 = b'\n' as u16;
const SCRIPT_REPLACEMENT: u16 = b'\n' as u16;
const STYLE_REPLACEMENT: u16 = b'\n' as u16;
const REPLACEMENT_CHARACTER: u16 = 0xFFFD;

/// `entityValues`: the HTML 4 character entities (and the upper-case
/// variants of `quot`, `copy`, `gt`, `lt`, `reg`, `amp`).
static ENTITY_VALUES: LazyLock<HashMap<&'static str, u16>> = LazyLock::new(|| {
    const ENTITIES: &[(&str, u16)] = &[
        ("AElig", 0x00C6),
        ("Aacute", 0x00C1),
        ("Acirc", 0x00C2),
        ("Agrave", 0x00C0),
        ("Alpha", 0x0391),
        ("Aring", 0x00C5),
        ("Atilde", 0x00C3),
        ("Auml", 0x00C4),
        ("Beta", 0x0392),
        ("Ccedil", 0x00C7),
        ("Chi", 0x03A7),
        ("Dagger", 0x2021),
        ("Delta", 0x0394),
        ("ETH", 0x00D0),
        ("Eacute", 0x00C9),
        ("Ecirc", 0x00CA),
        ("Egrave", 0x00C8),
        ("Epsilon", 0x0395),
        ("Eta", 0x0397),
        ("Euml", 0x00CB),
        ("Gamma", 0x0393),
        ("Iacute", 0x00CD),
        ("Icirc", 0x00CE),
        ("Igrave", 0x00CC),
        ("Iota", 0x0399),
        ("Iuml", 0x00CF),
        ("Kappa", 0x039A),
        ("Lambda", 0x039B),
        ("Mu", 0x039C),
        ("Ntilde", 0x00D1),
        ("Nu", 0x039D),
        ("OElig", 0x0152),
        ("Oacute", 0x00D3),
        ("Ocirc", 0x00D4),
        ("Ograve", 0x00D2),
        ("Omega", 0x03A9),
        ("Omicron", 0x039F),
        ("Oslash", 0x00D8),
        ("Otilde", 0x00D5),
        ("Ouml", 0x00D6),
        ("Phi", 0x03A6),
        ("Pi", 0x03A0),
        ("Prime", 0x2033),
        ("Psi", 0x03A8),
        ("Rho", 0x03A1),
        ("Scaron", 0x0160),
        ("Sigma", 0x03A3),
        ("THORN", 0x00DE),
        ("Tau", 0x03A4),
        ("Theta", 0x0398),
        ("Uacute", 0x00DA),
        ("Ucirc", 0x00DB),
        ("Ugrave", 0x00D9),
        ("Upsilon", 0x03A5),
        ("Uuml", 0x00DC),
        ("Xi", 0x039E),
        ("Yacute", 0x00DD),
        ("Yuml", 0x0178),
        ("Zeta", 0x0396),
        ("aacute", 0x00E1),
        ("acirc", 0x00E2),
        ("acute", 0x00B4),
        ("aelig", 0x00E6),
        ("agrave", 0x00E0),
        ("alefsym", 0x2135),
        ("alpha", 0x03B1),
        ("amp", 0x0026),
        ("and", 0x2227),
        ("ang", 0x2220),
        ("apos", 0x0027),
        ("aring", 0x00E5),
        ("asymp", 0x2248),
        ("atilde", 0x00E3),
        ("auml", 0x00E4),
        ("bdquo", 0x201E),
        ("beta", 0x03B2),
        ("brvbar", 0x00A6),
        ("bull", 0x2022),
        ("cap", 0x2229),
        ("ccedil", 0x00E7),
        ("cedil", 0x00B8),
        ("cent", 0x00A2),
        ("chi", 0x03C7),
        ("circ", 0x02C6),
        ("clubs", 0x2663),
        ("cong", 0x2245),
        ("copy", 0x00A9),
        ("crarr", 0x21B5),
        ("cup", 0x222A),
        ("curren", 0x00A4),
        ("dArr", 0x21D3),
        ("dagger", 0x2020),
        ("darr", 0x2193),
        ("deg", 0x00B0),
        ("delta", 0x03B4),
        ("diams", 0x2666),
        ("divide", 0x00F7),
        ("eacute", 0x00E9),
        ("ecirc", 0x00EA),
        ("egrave", 0x00E8),
        ("empty", 0x2205),
        ("emsp", 0x2003),
        ("ensp", 0x2002),
        ("epsilon", 0x03B5),
        ("equiv", 0x2261),
        ("eta", 0x03B7),
        ("eth", 0x00F0),
        ("euml", 0x00EB),
        ("euro", 0x20AC),
        ("exist", 0x2203),
        ("fnof", 0x0192),
        ("forall", 0x2200),
        ("frac12", 0x00BD),
        ("frac14", 0x00BC),
        ("frac34", 0x00BE),
        ("frasl", 0x2044),
        ("gamma", 0x03B3),
        ("ge", 0x2265),
        ("gt", 0x003E),
        ("hArr", 0x21D4),
        ("harr", 0x2194),
        ("hearts", 0x2665),
        ("hellip", 0x2026),
        ("iacute", 0x00ED),
        ("icirc", 0x00EE),
        ("iexcl", 0x00A1),
        ("igrave", 0x00EC),
        ("image", 0x2111),
        ("infin", 0x221E),
        ("int", 0x222B),
        ("iota", 0x03B9),
        ("iquest", 0x00BF),
        ("isin", 0x2208),
        ("iuml", 0x00EF),
        ("kappa", 0x03BA),
        ("lArr", 0x21D0),
        ("lambda", 0x03BB),
        ("lang", 0x2329),
        ("laquo", 0x00AB),
        ("larr", 0x2190),
        ("lceil", 0x2308),
        ("ldquo", 0x201C),
        ("le", 0x2264),
        ("lfloor", 0x230A),
        ("lowast", 0x2217),
        ("loz", 0x25CA),
        ("lrm", 0x200E),
        ("lsaquo", 0x2039),
        ("lsquo", 0x2018),
        ("lt", 0x003C),
        ("macr", 0x00AF),
        ("mdash", 0x2014),
        ("micro", 0x00B5),
        ("middot", 0x00B7),
        ("minus", 0x2212),
        ("mu", 0x03BC),
        ("nabla", 0x2207),
        ("nbsp", 0x0020),
        ("ndash", 0x2013),
        ("ne", 0x2260),
        ("ni", 0x220B),
        ("not", 0x00AC),
        ("notin", 0x2209),
        ("nsub", 0x2284),
        ("ntilde", 0x00F1),
        ("nu", 0x03BD),
        ("oacute", 0x00F3),
        ("ocirc", 0x00F4),
        ("oelig", 0x0153),
        ("ograve", 0x00F2),
        ("oline", 0x203E),
        ("omega", 0x03C9),
        ("omicron", 0x03BF),
        ("oplus", 0x2295),
        ("or", 0x2228),
        ("ordf", 0x00AA),
        ("ordm", 0x00BA),
        ("oslash", 0x00F8),
        ("otilde", 0x00F5),
        ("otimes", 0x2297),
        ("ouml", 0x00F6),
        ("para", 0x00B6),
        ("part", 0x2202),
        ("permil", 0x2030),
        ("perp", 0x22A5),
        ("phi", 0x03C6),
        ("pi", 0x03C0),
        ("piv", 0x03D6),
        ("plusmn", 0x00B1),
        ("pound", 0x00A3),
        ("prime", 0x2032),
        ("prod", 0x220F),
        ("prop", 0x221D),
        ("psi", 0x03C8),
        ("quot", 0x0022),
        ("rArr", 0x21D2),
        ("radic", 0x221A),
        ("rang", 0x232A),
        ("raquo", 0x00BB),
        ("rarr", 0x2192),
        ("rceil", 0x2309),
        ("rdquo", 0x201D),
        ("real", 0x211C),
        ("reg", 0x00AE),
        ("rfloor", 0x230B),
        ("rho", 0x03C1),
        ("rlm", 0x200F),
        ("rsaquo", 0x203A),
        ("rsquo", 0x2019),
        ("sbquo", 0x201A),
        ("scaron", 0x0161),
        ("sdot", 0x22C5),
        ("sect", 0x00A7),
        ("shy", 0x00AD),
        ("sigma", 0x03C3),
        ("sigmaf", 0x03C2),
        ("sim", 0x223C),
        ("spades", 0x2660),
        ("sub", 0x2282),
        ("sube", 0x2286),
        ("sum", 0x2211),
        ("sup", 0x2283),
        ("sup1", 0x00B9),
        ("sup2", 0x00B2),
        ("sup3", 0x00B3),
        ("supe", 0x2287),
        ("szlig", 0x00DF),
        ("tau", 0x03C4),
        ("there4", 0x2234),
        ("theta", 0x03B8),
        ("thetasym", 0x03D1),
        ("thinsp", 0x2009),
        ("thorn", 0x00FE),
        ("tilde", 0x02DC),
        ("times", 0x00D7),
        ("trade", 0x2122),
        ("uArr", 0x21D1),
        ("uacute", 0x00FA),
        ("uarr", 0x2191),
        ("ucirc", 0x00FB),
        ("ugrave", 0x00F9),
        ("uml", 0x00A8),
        ("upsih", 0x03D2),
        ("upsilon", 0x03C5),
        ("uuml", 0x00FC),
        ("weierp", 0x2118),
        ("xi", 0x03BE),
        ("yacute", 0x00FD),
        ("yen", 0x00A5),
        ("yuml", 0x00FF),
        ("zeta", 0x03B6),
        ("zwj", 0x200D),
        ("zwnj", 0x200C),
    ];
    const UPPER: &[(&str, &str)] = &[
        ("quot", "QUOT"),
        ("copy", "COPY"),
        ("gt", "GT"),
        ("lt", "LT"),
        ("reg", "REG"),
        ("amp", "AMP"),
    ];
    let mut m = HashMap::with_capacity(260);
    for &(name, v) in ENTITIES {
        m.insert(name, v);
        if let Some(&(_, upper)) = UPPER.iter().find(|(l, _)| *l == name) {
            m.insert(upper, v);
        }
    }
    m
});

/// `HTMLStripCharFilter.TextSegment` (an `OpenStringBuilder` plus a read
/// position).
#[derive(Debug, Clone)]
struct TextSegment {
    buf: Vec<u16>,
    pos: usize,
}

impl TextSegment {
    fn with_capacity(n: usize) -> Self {
        TextSegment {
            buf: Vec::with_capacity(n),
            pos: 0,
        }
    }

    /// `clear()`: `len = 0` and `pos = 0`.
    fn clear(&mut self) {
        self.buf.clear();
        self.pos = 0;
    }

    /// `reset()` (inherited): `len = 0`, `pos` kept.
    fn reset(&mut self) {
        self.buf.clear();
    }

    fn restart(&mut self) {
        self.pos = 0;
    }

    fn next_char(&mut self) -> u16 {
        let c = self.buf[self.pos];
        self.pos += 1;
        c
    }

    fn is_read(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn len(&self) -> i32 {
        self.buf.len() as i32
    }
}

/// Which segment Java's `outputSegment` references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Out {
    Input,
    Entity,
}

/// `org.apache.lucene.analysis.charfilter.HTMLStripCharFilter`: strips HTML
/// markup (tags, comments, `<script>`/`<style>` bodies, processing
/// instructions), decodes character references, and replaces block-level
/// tags and `<br>` with a newline, correcting offsets throughout.
pub struct HTMLStripCharFilter<R> {
    input: R,
    scanner: JFlexScanner,
    escaped_tags: Option<CharArraySet>,
    input_start: i64,
    cumulative_diff: i32,
    escape_br: bool,
    escape_script: bool,
    escape_style: bool,
    restore_state: usize,
    previous_restore_state: usize,
    output_char_count: i32,
    eof_return_value: Option<u16>,
    eof_done: bool,
    input_segment: TextSegment,
    entity_segment: TextSegment,
    out: Out,
    corrections: OffsetCorrections,
}

impl<R: CharReader> HTMLStripCharFilter<R> {
    /// `new HTMLStripCharFilter(Reader)`.
    pub fn new(input: R) -> Self {
        HTMLStripCharFilter {
            input,
            scanner: JFlexScanner::with_growth(&TABLES, ZZ_BUFFERSIZE, true),
            escaped_tags: None,
            input_start: 0,
            cumulative_diff: 0,
            escape_br: false,
            escape_script: false,
            escape_style: false,
            restore_state: 0,
            previous_restore_state: 0,
            output_char_count: 0,
            eof_return_value: None,
            eof_done: false,
            input_segment: TextSegment::with_capacity(INITIAL_INPUT_SEGMENT_SIZE),
            entity_segment: TextSegment::with_capacity(2),
            out: Out::Input,
            corrections: OffsetCorrections::default(),
        }
    }

    /// `new HTMLStripCharFilter(Reader, Set<String> escapedTags)`: the tags
    /// (start and end) kept in the output.
    pub fn with_escaped_tags<S: AsRef<str>>(
        input: R,
        escaped_tags: impl IntoIterator<Item = S>,
    ) -> Self {
        let mut f = Self::new(input);
        for tag in escaped_tags {
            let tag = tag.as_ref();
            if tag.eq_ignore_ascii_case("BR") {
                f.escape_br = true;
            } else if tag.eq_ignore_ascii_case("SCRIPT") {
                f.escape_script = true;
            } else if tag.eq_ignore_ascii_case("STYLE") {
                f.escape_style = true;
            } else {
                f.escaped_tags
                    .get_or_insert_with(|| CharArraySet::new(true))
                    .add(tag);
            }
        }
        f
    }

    fn output(&mut self) -> &mut TextSegment {
        match self.out {
            Out::Input => &mut self.input_segment,
            Out::Entity => &mut self.entity_segment,
        }
    }

    fn yylength(&self) -> i32 {
        self.scanner.yylength() as i32
    }

    fn yytext(&self) -> Vec<u16> {
        self.scanner.text().to_vec()
    }

    fn yybegin(&mut self, state: usize) {
        self.scanner.lexical_state = state;
    }

    fn add_off_correct_map(&mut self, off: i32, cumulative_diff: i32) {
        self.corrections.add(off, cumulative_diff);
    }

    /// `inputSegment.write(zzBuffer, zzStartRead, yylength())`.
    fn write_match_to_input(&mut self) {
        let t = self.scanner.text();
        self.input_segment.buf.extend_from_slice(t);
    }

    fn is_escaped_tag(&self) -> bool {
        let tag = String::from_utf16_lossy(self.scanner.text());
        self.escaped_tags.as_ref().is_some_and(|s| s.contains(&tag))
    }

    /// `entitySegment` holds `cp` (`Character.toChars`), or U+FFFD for a
    /// surrogate code point.
    fn set_entity_code_point(&mut self, cp: u32) {
        self.out = Out::Entity;
        self.entity_segment.clear();
        if (0xD800..=0xDFFF).contains(&cp) {
            self.entity_segment.buf.push(REPLACEMENT_CHARACTER);
        } else {
            crate::java_character::push_utf16(&mut self.entity_segment.buf, cp);
        }
    }

    // Java: zzDoEOF
    fn do_eof(&mut self) {
        if self.eof_done {
            return;
        }
        self.eof_done = true;
        match self.scanner.lexical_state {
            SCRIPT
            | COMMENT
            | SCRIPT_COMMENT
            | STYLE
            | STYLE_COMMENT
            | SINGLE_QUOTED_STRING
            | DOUBLE_QUOTED_STRING
            | END_TAG_TAIL_EXCLUDE
            | END_TAG_TAIL_SUBSTITUTE
            | START_TAG_TAIL_EXCLUDE
            | SERVER_SIDE_INCLUDE
            | START_TAG_TAIL_SUBSTITUTE => {
                // Exclude.
                self.cumulative_diff += (self.scanner.yychar - self.input_start) as i32;
                self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                self.output().clear();
                self.eof_return_value = None;
            }
            CHARACTER_REFERENCE_TAIL => {
                // Substitute: at end of file, allow char refs without semicolons.
                let out_len = self.output().len();
                self.cumulative_diff += self.input_segment.len() - out_len;
                self.add_off_correct_map(self.output_char_count + out_len, self.cumulative_diff);
                self.eof_return_value =
                    (!self.output().is_read()).then(|| self.output().next_char());
            }
            BANG
            | CDATA
            | AMPERSAND
            | NUMERIC_CHARACTER
            | END_TAG_TAIL_INCLUDE
            | START_TAG_TAIL_INCLUDE
            | LEFT_ANGLE_BRACKET
            | LEFT_ANGLE_BRACKET_SLASH
            | LEFT_ANGLE_BRACKET_SPACE => {
                // Include.
                self.out = Out::Input;
                self.eof_return_value =
                    (!self.output().is_read()).then(|| self.output().next_char());
            }
            _ => self.eof_return_value = None,
        }
    }

    /// The decimal or hexadecimal digits of a surrogate-pair reference, as a
    /// `char` (Java's `(char) Integer.parseInt(..)`).
    fn parse_unit(text: &[u16], from: usize, to: usize, radix: u32) -> u16 {
        let s = String::from_utf16_lossy(&text[from..to]);
        u32::from_str_radix(&s, radix).map_or(0, |v| v as u16)
    }

    // Java: nextChar (getNextToken): the scanner loop and its actions.
    #[allow(clippy::too_many_lines)]
    fn next_char(&mut self) -> Result<Option<u16>, AnalysisError> {
        loop {
            let action = match self.scanner.scan(&mut self.input)? {
                Scan::Eof => {
                    self.do_eof();
                    return Ok(self.eof_return_value);
                }
                Scan::Action(a) => a,
            };
            match action {
                1 | 21 => {
                    if self.yylength() == 1 {
                        return Ok(Some(self.scanner.yycharat(0)));
                    }
                    let t = self.yytext();
                    self.output().buf.extend_from_slice(&t);
                    return Ok(Some(self.output().next_char()));
                }
                2 => {
                    self.input_start = self.scanner.yychar;
                    self.input_segment.clear();
                    self.input_segment.buf.push(u16::from(b'&'));
                    self.yybegin(AMPERSAND);
                }
                3 => {
                    self.input_start = self.scanner.yychar;
                    self.input_segment.clear();
                    self.input_segment.buf.push(u16::from(b'<'));
                    self.yybegin(LEFT_ANGLE_BRACKET);
                }
                4 => {
                    let n = self.scanner.yylength();
                    self.scanner.yypushback(n)?;
                    self.out = Out::Input;
                    self.output().restart();
                    self.yybegin(YYINITIAL);
                    return Ok(Some(self.output().next_char()));
                }
                5 => {
                    self.input_segment.buf.push(u16::from(b'#'));
                    self.yybegin(NUMERIC_CHARACTER);
                }
                6 | 32 => {
                    // 6: decimal, 32: hexadecimal (after the 'x').
                    let match_length = self.yylength();
                    self.write_match_to_input();
                    let (max_len, skip, radix) = if action == 6 { (7, 0, 10) } else { (6, 1, 16) };
                    if match_length <= max_len {
                        let t = self.yytext();
                        let cp = Self::parse_unit_u32(&t, skip, radix);
                        if cp <= 0x10FFFF {
                            self.set_entity_code_point(cp);
                            self.yybegin(CHARACTER_REFERENCE_TAIL);
                            continue;
                        }
                    }
                    self.out = Out::Input;
                    self.yybegin(YYINITIAL);
                    return Ok(Some(self.output().next_char()));
                }
                7 => {
                    let out_len = self.output().len();
                    self.cumulative_diff += self.input_segment.len() + self.yylength() - out_len;
                    self.add_off_correct_map(
                        self.output_char_count + out_len,
                        self.cumulative_diff,
                    );
                    self.yybegin(YYINITIAL);
                    return Ok(Some(self.output().next_char()));
                }
                8 => {
                    self.write_match_to_input();
                    self.yybegin(LEFT_ANGLE_BRACKET_SPACE);
                }
                9 => {
                    self.input_segment.buf.push(u16::from(b'!'));
                    self.yybegin(BANG);
                }
                10 => {
                    self.input_segment.buf.push(u16::from(b'/'));
                    self.yybegin(LEFT_ANGLE_BRACKET_SLASH);
                }
                11 | 12 | 19 | 20 => {
                    self.write_match_to_input();
                    let state = if self.is_escaped_tag() {
                        if action <= 12 {
                            START_TAG_TAIL_INCLUDE
                        } else {
                            END_TAG_TAIL_INCLUDE
                        }
                    } else {
                        match action {
                            11 => START_TAG_TAIL_SUBSTITUTE,
                            12 => START_TAG_TAIL_EXCLUDE,
                            19 => END_TAG_TAIL_SUBSTITUTE,
                            _ => END_TAG_TAIL_EXCLUDE,
                        }
                    };
                    self.yybegin(state);
                }
                13 => {
                    let t = self.yytext();
                    self.input_segment.buf.extend_from_slice(&t);
                }
                14 => {
                    self.cumulative_diff += self.input_segment.len() + self.yylength();
                    self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                    self.input_segment.clear();
                    self.yybegin(YYINITIAL);
                }
                15 => {}
                16 => {
                    self.restore_state = SCRIPT_COMMENT;
                    self.yybegin(DOUBLE_QUOTED_STRING);
                }
                17 => {
                    self.restore_state = SCRIPT_COMMENT;
                    self.yybegin(SINGLE_QUOTED_STRING);
                }
                18 => self.write_match_to_input(),
                22 | 23 => {
                    self.previous_restore_state = self.restore_state;
                    self.restore_state = SERVER_SIDE_INCLUDE;
                    self.yybegin(if action == 22 {
                        DOUBLE_QUOTED_STRING
                    } else {
                        SINGLE_QUOTED_STRING
                    });
                }
                24 => {
                    self.yybegin(self.restore_state);
                    self.restore_state = self.previous_restore_state;
                }
                25 => {
                    self.write_match_to_input();
                    self.out = Out::Input;
                    self.yybegin(YYINITIAL);
                    return Ok(Some(self.output().next_char()));
                }
                26 | 28 => {
                    self.cumulative_diff += self.input_segment.len() + self.yylength() - 1;
                    self.add_off_correct_map(self.output_char_count + 1, self.cumulative_diff);
                    self.input_segment.clear();
                    self.yybegin(YYINITIAL);
                    return Ok(Some(if action == 26 {
                        BLOCK_LEVEL_END_TAG_REPLACEMENT
                    } else {
                        BLOCK_LEVEL_START_TAG_REPLACEMENT
                    }));
                }
                27 => {
                    self.cumulative_diff += self.input_segment.len() + self.yylength();
                    self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                    self.input_segment.clear();
                    self.out = Out::Input;
                    self.yybegin(YYINITIAL);
                }
                29 => {
                    self.restore_state = STYLE_COMMENT;
                    self.yybegin(DOUBLE_QUOTED_STRING);
                }
                30 => {
                    self.restore_state = STYLE_COMMENT;
                    self.yybegin(SINGLE_QUOTED_STRING);
                }
                31 => {
                    self.write_match_to_input();
                    self.entity_segment.clear();
                    let name = String::from_utf16_lossy(self.scanner.text());
                    let ch = *ENTITY_VALUES.get(name.as_str()).ok_or_else(|| {
                        AnalysisError::IllegalState(format!("unknown entity {name}"))
                    })?;
                    self.entity_segment.buf.push(ch);
                    self.out = Out::Entity;
                    self.yybegin(CHARACTER_REFERENCE_TAIL);
                }
                33 => {
                    if self.input_segment.len() > 2 {
                        // Chars between "<!" and "--" - this is not a comment.
                        let t = self.yytext();
                        self.input_segment.buf.extend_from_slice(&t);
                    } else {
                        self.yybegin(COMMENT);
                    }
                }
                34 | 37 => {
                    self.yybegin(YYINITIAL);
                    if self.escape_br {
                        self.write_match_to_input();
                        self.out = Out::Input;
                        return Ok(Some(self.output().next_char()));
                    }
                    self.cumulative_diff += self.input_segment.len() + self.yylength() - 1;
                    self.add_off_correct_map(self.output_char_count + 1, self.cumulative_diff);
                    self.input_segment.reset();
                    return Ok(Some(if action == 34 {
                        BR_START_TAG_REPLACEMENT
                    } else {
                        BR_END_TAG_REPLACEMENT
                    }));
                }
                35 => {
                    self.cumulative_diff += (self.scanner.yychar - self.input_start
                        + i64::from(self.yylength()))
                        as i32;
                    self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                    self.input_segment.clear();
                    self.yybegin(YYINITIAL);
                }
                36 => self.yybegin(SCRIPT),
                38 => {
                    self.cumulative_diff += self.yylength();
                    self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                    self.yybegin(YYINITIAL);
                }
                39 => self.yybegin(self.restore_state),
                40 => self.yybegin(STYLE),
                41 => self.yybegin(SCRIPT_COMMENT),
                42 => self.yybegin(STYLE_COMMENT),
                43..=45 => {
                    self.restore_state = match action {
                        43 => COMMENT,
                        44 => SCRIPT_COMMENT,
                        _ => STYLE_COMMENT,
                    };
                    self.yybegin(SERVER_SIDE_INCLUDE);
                }
                46 | 47 => {
                    let (state, escape) = if action == 46 {
                        (STYLE, self.escape_style)
                    } else {
                        (SCRIPT, self.escape_script)
                    };
                    self.yybegin(state);
                    if escape {
                        self.write_match_to_input();
                        self.out = Out::Input;
                        self.input_start += 1 + i64::from(self.yylength());
                        return Ok(Some(self.output().next_char()));
                    }
                }
                48 => {
                    if self.input_segment.len() > 2 {
                        // Chars between "<!" and "[CDATA[" - this is not a CDATA section.
                        let t = self.yytext();
                        self.input_segment.buf.extend_from_slice(&t);
                    } else {
                        self.cumulative_diff += self.input_segment.len() + self.yylength();
                        self.add_off_correct_map(self.output_char_count, self.cumulative_diff);
                        self.input_segment.clear();
                        self.yybegin(CDATA);
                    }
                }
                49 | 50 => {
                    let (escape, replacement) = if action == 49 {
                        (self.escape_style, STYLE_REPLACEMENT)
                    } else {
                        (self.escape_script, SCRIPT_REPLACEMENT)
                    };
                    self.input_segment.clear();
                    self.yybegin(YYINITIAL);
                    self.cumulative_diff += (self.scanner.yychar - self.input_start) as i32;
                    let mut offset_correction_pos = self.output_char_count;
                    let return_value = if escape {
                        self.write_match_to_input();
                        self.out = Out::Input;
                        self.output().next_char()
                    } else {
                        self.cumulative_diff += self.yylength() - 1;
                        offset_correction_pos += 1;
                        replacement
                    };
                    self.add_off_correct_map(offset_correction_pos, self.cumulative_diff);
                    return Ok(Some(return_value));
                }
                51..=54 => {
                    // Paired UTF-16 surrogates as references: 51 `#D;#D`,
                    // 52 `#D;#xH`, 53 `#xH;#D`, 54 `#xH;#xH` (D decimal, H hex).
                    let t = self.yytext();
                    let high = if action <= 52 {
                        Self::parse_unit(&t, 1, 6, 10)
                    } else {
                        Self::parse_unit(&t, 2, 6, 16)
                    };
                    let low = if action == 51 || action == 53 {
                        Self::parse_unit(&t, 9, 14, 10)
                    } else {
                        Self::parse_unit(&t, 10, 14, 16)
                    };
                    let paired = match action {
                        51 => {
                            crate::java_character::is_high_surrogate(high)
                                && crate::java_character::is_low_surrogate(low)
                        }
                        52 => crate::java_character::is_high_surrogate(high),
                        53 => crate::java_character::is_low_surrogate(low),
                        _ => true,
                    };
                    if paired {
                        self.out = Out::Entity;
                        self.entity_segment.clear();
                        self.entity_segment.buf.push(low);
                        self.cumulative_diff += self.input_segment.len() + self.yylength() - 2;
                        self.add_off_correct_map(self.output_char_count + 2, self.cumulative_diff);
                        self.input_segment.clear();
                        self.yybegin(YYINITIAL);
                        return Ok(Some(high));
                    }
                    // Consume only '#'.
                    self.scanner.yypushback(t.len() - 1)?;
                    self.input_segment.buf.push(u16::from(b'#'));
                    self.yybegin(NUMERIC_CHARACTER);
                }
                _ => {
                    return Err(AnalysisError::IllegalState(
                        "Error: could not match input".into(),
                    ))
                }
            }
        }
    }

    /// `Integer.parseInt(text[skip..], radix)` of a matched reference.
    fn parse_unit_u32(text: &[u16], skip: usize, radix: u32) -> u32 {
        let s = String::from_utf16_lossy(&text[skip..]);
        u32::from_str_radix(&s, radix).unwrap_or(0)
    }

    // Java: HTMLStripCharFilter.read()
    fn read_one(&mut self) -> Result<Option<u16>, AnalysisError> {
        let ch = if self.output().is_read() {
            if self.scanner.at_eof() {
                return Ok(None);
            }
            match self.next_char()? {
                Some(c) => c,
                None => return Ok(None),
            }
        } else {
            self.output().next_char()
        };
        self.output_char_count += 1;
        Ok(Some(ch))
    }
}

impl<R: CharReader> CharFilter for HTMLStripCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    // Java: HTMLStripCharFilter.read(char[], int, int)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut i = 0;
        while i < buf.len() {
            match self.read_one()? {
                Some(c) => {
                    buf[i] = c;
                    i += 1;
                }
                None => break,
            }
        }
        Ok(i)
    }

    fn correct(&self, current_off: i32) -> i32 {
        self.corrections.correct(current_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;

    fn strip(f: &mut HTMLStripCharFilter<StrReader>) -> (String, Vec<i32>) {
        let mut out = Vec::new();
        let mut buf = [0u16; 7];
        loop {
            let n = f.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        let corr = (0..=out.len() as i32)
            .map(|o| f.correct_offset(o))
            .collect();
        (String::from_utf16_lossy(&out), corr)
    }

    /// `(html, Lucene's output, Lucene's correctOffset(0..=len))`.
    const CASES: &[(&str, &str, &[i32])] = &[
        (
            "<p>a &amp; b</p>",
            "\na & b\n",
            &[0, 3, 4, 5, 10, 11, 12, 16],
        ),
        ("x<!-- c -->y<br/>z", "xy\nz", &[0, 11, 12, 17, 18]),
        ("<script>var a = '<b>';</script>t", "\nt", &[0, 31, 32]),
        ("<style>p{}</style>u", "\nu", &[0, 18, 19]),
        (
            "&#65;&#x42;&#55357;&#56832;&#xD83D;&#xDE00;",
            "AB😀😀",
            &[0, 5, 11, 12, 27, 28, 43],
        ),
        (
            "&#55357;&#xDE00;&#xD83D;&#56832;",
            "😀😀",
            &[0, 1, 16, 17, 32],
        ),
        (
            "&#xD800;&#99999999;&COPY;&unknown;&lt",
            "\u{FFFD}&#99999999;©&unknown;<",
            &[
                0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 37,
            ],
        ),
        ("<![CDATA[ x<y ]]>", " x<y ", &[9, 10, 11, 12, 13, 17]),
        ("<!DOCTYPE html><?php echo 1 ?>q", "q", &[30, 31]),
        ("a <b>b</b> < c", "a b ", &[0, 1, 5, 10, 14]),
        ("<b>ab</b>c", "abc", &[3, 4, 9, 10]),
        ("a<!-- unterminated", "a", &[0, 18]),
        ("a&amp", "a&", &[0, 1, 5]),
        ("a&#3", "a\u{3}", &[0, 1, 4]),
        ("a<", "a<", &[0, 1, 2]),
        ("a<b c=\"x", "a<b c=\"x", &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("a<!--#include virtual=\"x.html\" -->b", "ab", &[0, 34, 35]),
        ("<script>/* \"q\" 'r' */ x</script>c", "\nc", &[0, 32, 33]),
        (
            "<script><!-- s=\"a\" t='b' <!--#echo var=\"v\" --> --></script>d",
            "\nd",
            &[0, 59, 60],
        ),
        (
            "<style><!-- p{content:\"x\"} q{content:'y'} <!--#if expr=\"1\" --> --></style>e",
            "\ne",
            &[0, 74, 75],
        ),
        ("<!x[CDATA[ y ]]>f", "f", &[16, 17]),
        ("<!-- a <!--#include file=\"f\" --> b -->g", "g", &[38, 39]),
        ("<!ELEMENT note (#PCDATA)>h", "h", &[25, 26]),
        (
            "&#x110000;&#1114112;&#xFFFFFFF;&#12345678;i",
            "&#x110000;&#1114112;&#xFFFFFFF;&#12345678;i",
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            ],
        ),
        (
            "&#55357;&#56832x&#xD83D;&#xZZ;j",
            "\u{FFFD}&#56832x\u{FFFD}&#xZZ;j",
            &[
                0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
        ),
        ("&#56832;&#55357;k", "\u{FFFD}\u{FFFD}k", &[0, 8, 16, 17]),
        ("<p class=\"a\" id='b'>l</p >", "\nl\n", &[0, 20, 21, 26]),
        (
            "</p>m</ div>n< /p>",
            "\nm\nn< /p>",
            &[0, 4, 5, 12, 13, 14, 15, 16, 17, 18],
        ),
        (
            "<a href=x>o</a><unknowntag>p</unknowntag>",
            "o\np\n",
            &[10, 15, 27, 28, 41],
        ),
        (
            "&COPY&gt&nbsp;q",
            "&COPY&gt q",
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15],
        ),
        ("<br><br/><BR>r</br>", "\n\n\nr\n", &[0, 4, 9, 13, 14, 19]),
        ("<?xml version=\"1.0\"?>s", "s", &[21, 22]),
        (
            "x &amp y &; z &#; w &#x;",
            "x &amp y &; z &#; w &#x;",
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24,
            ],
        ),
        (
            "<scrip>t</scrip><styl>u</styl>",
            "\nt\n\nu\n",
            &[0, 7, 8, 16, 22, 23, 30],
        ),
        ("<script>v", "", &[9]),
        ("<style>w", "", &[8]),
        ("<![CDATA[x", "x", &[9, 10]),
        ("<!--y", "", &[5]),
        ("<b c='z", "<b c='z", &[0, 1, 2, 3, 4, 5, 6, 7]),
        ("&eacute", "é", &[0, 7]),
        ("<p", "", &[2]),
        ("</b", "", &[3]),
        ("<!", "<!", &[0, 1, 2]),
        ("<!-", "<!-", &[0, 1, 2, 3]),
    ];

    #[test]
    fn strips_as_lucene_does() {
        for (html, out, corr) in CASES {
            let (s, c) = strip(&mut HTMLStripCharFilter::new(StrReader::new(*html)));
            assert_eq!(&s, out, "{html}");
            assert_eq!(&c, corr, "{html}");
        }
    }

    #[test]
    fn escaped_tags_and_buffer_growth() {
        let mut f = HTMLStripCharFilter::with_escaped_tags(
            StrReader::new("<b>x</b><br><i>y</i><script>s</script>"),
            ["b", "BR", "script", "style"],
        );
        let (s, c) = strip(&mut f);
        assert_eq!(s, "<b>x</b><br>y<script></script>");
        assert_eq!(
            c,
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38
            ]
        );
        let mut f = HTMLStripCharFilter::with_escaped_tags(
            StrReader::new("<b>x</b><br>y</br><script>s</script><style>t</style>z<B>w</B>"),
            ["b", "BR", "script", "style"],
        );
        assert_eq!(
            strip(&mut f).0,
            "<b>x</b><br>y</br><script></script><style></style>z<B>w</B>"
        );
        for (html, out) in [("<script>s", "<script>"), ("<style>t", "<style>")] {
            let mut f =
                HTMLStripCharFilter::with_escaped_tags(StrReader::new(html), ["script", "style"]);
            assert_eq!(strip(&mut f).0, out);
        }
        // A tag name longer than the 16384-unit buffer grows it.
        let tag = format!("<{}>z", "a".repeat(20_000));
        assert_eq!(
            strip(&mut HTMLStripCharFilter::new(StrReader::new(tag))).0,
            "\nz"
        );
        let long = format!("<!--{}-->z", "x".repeat(40_000));
        assert_eq!(
            strip(&mut HTMLStripCharFilter::new(StrReader::new(long))).0,
            "z"
        );
    }
}
