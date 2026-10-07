//! `org.apache.lucene.analysis.wikipedia.WikipediaTokenizer`: the classic
//! grammar plus MediaWiki syntax -- internal and external links, citations,
//! categories, bold/italics, headings -- each typed (`il`, `el`, `elu`, `ci`,
//! `c`, `b`, `i`, `bi`, `h`, `sh`). The JFlex scanner
//! `WikipediaTokenizerImpl` runs over its extracted tables (see
//! [`crate::util::jflex`]), its actions and user state ported line for line.
//!
//! With `UNTOKENIZED_ONLY` or `BOTH`, a run of tokens of one of the
//! `untokenized_types` is collapsed into one token (flag
//! [`UNTOKENIZED_TOKEN_FLAG`]); `BOTH` then also replays the individual
//! tokens.

use std::collections::{HashSet, VecDeque};
use std::sync::LazyLock;

use crate::attributes::{AttributeSource, State};
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::util::jflex::{JFlexScanner, JFlexTables, Scan, YYEOF};
use crate::AnalysisError;

/// `WikipediaTokenizerImpl`'s tables (`tools/ExtractJFlexTables.java`).
static TABLES: LazyLock<JFlexTables> = LazyLock::new(|| {
    JFlexTables::load(include_bytes!("tables.bin.z"))
        .expect("the generated wikipedia scanner tables")
});

/// `WikipediaTokenizerImpl.ZZ_BUFFERSIZE`.
const ZZ_BUFFERSIZE: usize = 4096;

/// `WikipediaTokenizer.INTERNAL_LINK` .. `SUB_HEADING`, `EXTERNAL_LINK_URL`.
pub const INTERNAL_LINK: &str = "il";
/// `WikipediaTokenizer.EXTERNAL_LINK`.
pub const EXTERNAL_LINK: &str = "el";
/// `WikipediaTokenizer.EXTERNAL_LINK_URL`.
pub const EXTERNAL_LINK_URL: &str = "elu";
/// `WikipediaTokenizer.CITATION`.
pub const CITATION: &str = "ci";
/// `WikipediaTokenizer.CATEGORY`.
pub const CATEGORY: &str = "c";
/// `WikipediaTokenizer.BOLD`.
pub const BOLD: &str = "b";
/// `WikipediaTokenizer.ITALICS`.
pub const ITALICS: &str = "i";
/// `WikipediaTokenizer.BOLD_ITALICS`.
pub const BOLD_ITALICS: &str = "bi";
/// `WikipediaTokenizer.HEADING`.
pub const HEADING: &str = "h";
/// `WikipediaTokenizer.SUB_HEADING`.
pub const SUB_HEADING: &str = "sh";

const ALPHANUM_ID: i32 = 0;
const APOSTROPHE_ID: i32 = 1;
const ACRONYM_ID: i32 = 2;
const COMPANY_ID: i32 = 3;
const EMAIL_ID: i32 = 4;
const HOST_ID: i32 = 5;
const NUM_ID: i32 = 6;
const CJ_ID: i32 = 7;
const INTERNAL_LINK_ID: i32 = 8;
const EXTERNAL_LINK_ID: i32 = 9;
const CITATION_ID: i32 = 10;
const CATEGORY_ID: i32 = 11;
const BOLD_ID: i32 = 12;
const ITALICS_ID: i32 = 13;
const BOLD_ITALICS_ID: i32 = 14;
const HEADING_ID: i32 = 15;
const SUB_HEADING_ID: i32 = 16;
const EXTERNAL_LINK_URL_ID: i32 = 17;

/// `WikipediaTokenizer.TOKEN_TYPES`.
pub const TOKEN_TYPES: [&str; 18] = [
    "<ALPHANUM>",
    "<APOSTROPHE>",
    "<ACRONYM>",
    "<COMPANY>",
    "<EMAIL>",
    "<HOST>",
    "<NUM>",
    "<CJ>",
    INTERNAL_LINK,
    EXTERNAL_LINK,
    CITATION,
    CATEGORY,
    BOLD,
    ITALICS,
    BOLD_ITALICS,
    HEADING,
    SUB_HEADING,
    EXTERNAL_LINK_URL,
];

/// `WikipediaTokenizer.TOKENS_ONLY`.
pub const TOKENS_ONLY: i32 = 0;
/// `WikipediaTokenizer.UNTOKENIZED_ONLY`.
pub const UNTOKENIZED_ONLY: i32 = 1;
/// `WikipediaTokenizer.BOTH`.
pub const BOTH: i32 = 2;
/// `WikipediaTokenizer.UNTOKENIZED_TOKEN_FLAG`.
pub const UNTOKENIZED_TOKEN_FLAG: i32 = 1;

// Lexical states.
const YYINITIAL: usize = 0;
const CATEGORY_STATE: usize = 2;
const INTERNAL_LINK_STATE: usize = 4;
const EXTERNAL_LINK_STATE: usize = 6;
const TWO_SINGLE_QUOTES_STATE: usize = 8;
const THREE_SINGLE_QUOTES_STATE: usize = 10;
const FIVE_SINGLE_QUOTES_STATE: usize = 12;
const DOUBLE_EQUALS_STATE: usize = 14;
const DOUBLE_BRACE_STATE: usize = 16;
const STRING: usize = 18;

/// `WikipediaTokenizerImpl`: the scanner and its user state.
struct WikipediaTokenizerImpl {
    s: JFlexScanner,
    current_tok_type: i32,
    num_balanced: i32,
    position_inc: i32,
    num_link_toks: i32,
    num_wiki_tokens_seen: i32,
}

impl WikipediaTokenizerImpl {
    /// `reset()` (the user state; `yyreset` is the scanner's).
    fn reset(&mut self) {
        self.current_tok_type = 0;
        self.num_balanced = 0;
        self.position_inc = 1;
        self.num_link_toks = 0;
        self.num_wiki_tokens_seen = 0;
    }

    fn yychar(&self) -> i32 {
        self.s.yychar as i32
    }

    // Java: WikipediaTokenizerImpl.getNextToken (the action switch)
    fn get_next_token(&mut self, reader: &mut dyn CharReader) -> Result<i32, AnalysisError> {
        loop {
            let action = match self.s.scan(reader)? {
                Scan::Eof => return Ok(YYEOF),
                Scan::Action(a) => a,
            };
            match action {
                1 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                }
                2 => {
                    self.position_inc = 1;
                    return Ok(ALPHANUM_ID);
                }
                3 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    self.current_tok_type = EXTERNAL_LINK_URL_ID;
                    self.s.lexical_state = EXTERNAL_LINK_STATE;
                }
                4 => {
                    self.position_inc = 1;
                    return Ok(CJ_ID);
                }
                5 => self.position_inc = 1,
                6 => {
                    self.s.lexical_state = CATEGORY_STATE;
                    self.num_wiki_tokens_seen += 1;
                    return Ok(self.current_tok_type);
                }
                7 => {
                    self.s.lexical_state = INTERNAL_LINK_STATE;
                    self.num_wiki_tokens_seen += 1;
                    return Ok(self.current_tok_type);
                }
                8 | 18 => {} // ignore
                9 => {
                    self.position_inc = if self.num_link_toks == 0 { 0 } else { 1 };
                    self.num_wiki_tokens_seen += 1;
                    self.current_tok_type = EXTERNAL_LINK_ID;
                    self.s.lexical_state = EXTERNAL_LINK_STATE;
                    self.num_link_toks += 1;
                    return Ok(self.current_tok_type);
                }
                10 => {
                    self.num_link_toks = 0;
                    self.position_inc = 0;
                    self.s.lexical_state = YYINITIAL;
                }
                11 => {
                    self.current_tok_type = BOLD_ID;
                    self.s.lexical_state = THREE_SINGLE_QUOTES_STATE;
                }
                12 => {
                    self.current_tok_type = ITALICS_ID;
                    self.num_wiki_tokens_seen += 1;
                    self.s.lexical_state = STRING;
                    return Ok(self.current_tok_type);
                }
                13 => {
                    self.current_tok_type = EXTERNAL_LINK_ID;
                    self.num_wiki_tokens_seen = 0;
                    self.s.lexical_state = EXTERNAL_LINK_STATE;
                }
                14 | 19 => {
                    self.s.lexical_state = STRING;
                    self.num_wiki_tokens_seen += 1;
                    return Ok(self.current_tok_type);
                }
                15 => {
                    self.current_tok_type = HEADING_ID;
                    self.s.lexical_state = DOUBLE_EQUALS_STATE;
                    self.num_wiki_tokens_seen += 1;
                    return Ok(self.current_tok_type);
                }
                16 => {
                    self.current_tok_type = SUB_HEADING_ID;
                    self.num_wiki_tokens_seen = 0;
                    self.s.lexical_state = STRING;
                }
                17 => {
                    self.s.lexical_state = DOUBLE_BRACE_STATE;
                    self.num_wiki_tokens_seen = 0;
                    return Ok(self.current_tok_type);
                }
                20 => {
                    self.num_balanced = 0;
                    self.num_wiki_tokens_seen = 0;
                    self.current_tok_type = EXTERNAL_LINK_ID;
                    self.s.lexical_state = EXTERNAL_LINK_STATE;
                }
                21 => {
                    self.s.lexical_state = STRING;
                    return Ok(self.current_tok_type);
                }
                22 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    if self.num_balanced == 0 {
                        self.num_balanced += 1;
                        self.s.lexical_state = TWO_SINGLE_QUOTES_STATE;
                    } else {
                        self.num_balanced = 0;
                    }
                }
                23 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    self.s.lexical_state = DOUBLE_EQUALS_STATE;
                }
                24 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    self.current_tok_type = INTERNAL_LINK_ID;
                    self.s.lexical_state = INTERNAL_LINK_STATE;
                }
                25 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    self.current_tok_type = CITATION_ID;
                    self.s.lexical_state = DOUBLE_BRACE_STATE;
                }
                26 | 30 => self.s.lexical_state = YYINITIAL,
                27 => {
                    self.num_link_toks = 0;
                    self.s.lexical_state = YYINITIAL;
                }
                28 | 29 => {
                    self.current_tok_type = INTERNAL_LINK_ID;
                    self.num_wiki_tokens_seen = 0;
                    self.s.lexical_state = INTERNAL_LINK_STATE;
                }
                // end italics, end bold, end sub header, end bold italics
                31 | 38 | 39 | 42 => {
                    self.num_balanced = 0;
                    self.current_tok_type = ALPHANUM_ID;
                    self.s.lexical_state = YYINITIAL;
                }
                32 => {
                    self.num_balanced = 0;
                    self.num_wiki_tokens_seen = 0;
                    self.current_tok_type = INTERNAL_LINK_ID;
                    self.s.lexical_state = INTERNAL_LINK_STATE;
                }
                33 => {
                    self.position_inc = 1;
                    return Ok(NUM_ID);
                }
                34 => {
                    self.position_inc = 1;
                    return Ok(COMPANY_ID);
                }
                35 => {
                    self.position_inc = 1;
                    return Ok(APOSTROPHE_ID);
                }
                36 => {
                    self.position_inc = 1;
                    return Ok(HOST_ID);
                }
                37 => {
                    self.current_tok_type = BOLD_ITALICS_ID;
                    self.s.lexical_state = FIVE_SINGLE_QUOTES_STATE;
                }
                40 => {
                    self.position_inc = 1;
                    return Ok(ACRONYM_ID);
                }
                41 => {
                    self.position_inc = 1;
                    return Ok(EMAIL_ID);
                }
                43 => {
                    self.position_inc = 1;
                    self.num_wiki_tokens_seen += 1;
                    self.s.lexical_state = EXTERNAL_LINK_STATE;
                    return Ok(self.current_tok_type);
                }
                44 => {
                    self.num_wiki_tokens_seen = 0;
                    self.position_inc = 1;
                    self.current_tok_type = CATEGORY_ID;
                    self.s.lexical_state = CATEGORY_STATE;
                }
                45 => {
                    self.current_tok_type = CATEGORY_ID;
                    self.num_wiki_tokens_seen = 0;
                    self.s.lexical_state = CATEGORY_STATE;
                }
                46 => {
                    self.num_balanced = 0;
                    self.num_wiki_tokens_seen = 0;
                    self.current_tok_type = CATEGORY_ID;
                    self.s.lexical_state = CATEGORY_STATE;
                }
                _ => {
                    return Err(AnalysisError::IllegalState(
                        "Error: could not match input".into(),
                    ))
                }
            }
        }
    }
}

/// `org.apache.lucene.analysis.wikipedia.WikipediaTokenizer`.
pub struct WikipediaTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    scanner: WikipediaTokenizerImpl,
    token_output: i32,
    untokenized_types: HashSet<String>,
    tokens: Option<VecDeque<State>>,
    first: bool,
}

impl Default for WikipediaTokenizer {
    /// `new WikipediaTokenizer()`: `TOKENS_ONLY`.
    fn default() -> Self {
        Self::new(TOKENS_ONLY, HashSet::new()).expect("TOKENS_ONLY is valid")
    }
}

/// `String.trim()` over UTF-16 units.
fn trim_utf16(b: &[u16]) -> &[u16] {
    let start = b.iter().position(|&c| c > 0x20).unwrap_or(b.len());
    let end = b.iter().rposition(|&c| c > 0x20).map_or(start, |e| e + 1);
    &b[start..end]
}

impl WikipediaTokenizer {
    /// `new WikipediaTokenizer(tokenOutput, untokenizedTypes)`.
    pub fn new(
        token_output: i32,
        untokenized_types: HashSet<String>,
    ) -> Result<Self, AnalysisError> {
        if token_output != TOKENS_ONLY && token_output != UNTOKENIZED_ONLY && token_output != BOTH {
            return Err(AnalysisError::IllegalArgument(
                "tokenOutput must be TOKENS_ONLY, UNTOKENIZED_ONLY or BOTH".into(),
            ));
        }
        Ok(WikipediaTokenizer {
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
            scanner: WikipediaTokenizerImpl {
                s: JFlexScanner::with_growth(&TABLES, ZZ_BUFFERSIZE, true),
                current_tok_type: 0,
                num_balanced: 0,
                position_inc: 1,
                num_link_toks: 0,
                num_wiki_tokens_seen: 0,
            },
            token_output,
            untokenized_types,
            tokens: None,
            first: false,
        })
    }

    // Java: setupToken
    fn setup_token(&mut self) -> Result<(), AnalysisError> {
        let text = self.scanner.s.text();
        self.atts.set_term_utf16(text);
        let start = self.scanner.yychar();
        let len = text.len() as i32;
        let (s, e) = (
            self.input.correct_offset(start),
            self.input.correct_offset(start + len),
        );
        self.atts.set_offset(s, e)
    }

    /// The shared body of `collapseTokens` / `collapseAndSaveTokens`; `save`
    /// captures each token (`setupSavedToken`).
    fn collapse(
        &mut self,
        token_type: i32,
        ty: &'static str,
        save: bool,
    ) -> Result<(), AnalysisError> {
        let mut buffer: Vec<u16> = Vec::with_capacity(32);
        buffer.extend_from_slice(self.scanner.s.text());
        let mut num_added = self.scanner.s.yylength() as i32;
        let the_start = self.scanner.yychar();
        let mut last_pos = the_start + num_added;
        let mut num_seen = 0;
        let mut tmp: VecDeque<State> = VecDeque::new();
        if save {
            self.setup_saved_token(0, ty)?;
            tmp.push_back(self.atts.capture_state());
        }
        let mut tmp_tok_type;
        loop {
            tmp_tok_type = self.scanner.get_next_token(self.input.reader()?)?;
            if !(tmp_tok_type != YYEOF
                && tmp_tok_type == token_type
                && self.scanner.num_wiki_tokens_seen > num_seen)
            {
                break;
            }
            let curr_pos = self.scanner.yychar();
            for _ in 0..(curr_pos - last_pos).max(0) {
                buffer.push(u16::from(b' '));
            }
            buffer.extend_from_slice(self.scanner.s.text());
            num_added = self.scanner.s.yylength() as i32;
            if save {
                let inc = self.scanner.position_inc;
                self.setup_saved_token(inc, ty)?;
                tmp.push_back(self.atts.capture_state());
            }
            num_seen += 1;
            last_pos = curr_pos + num_added;
        }
        let s = trim_utf16(&buffer);
        self.atts.set_term_utf16(s);
        let (a, b) = (
            self.input.correct_offset(the_start),
            self.input.correct_offset(the_start + s.len() as i32),
        );
        self.atts.set_offset(a, b)?;
        self.atts.set_flags(UNTOKENIZED_TOKEN_FLAG);
        if tmp_tok_type != YYEOF {
            let n = self.scanner.s.yylength();
            self.scanner.s.yypushback(n)?;
        }
        if save {
            self.tokens = Some(tmp);
        } else if tmp_tok_type == YYEOF {
            self.tokens = None;
        }
        Ok(())
    }

    // Java: setupSavedToken
    fn setup_saved_token(
        &mut self,
        position_inc: i32,
        ty: &'static str,
    ) -> Result<(), AnalysisError> {
        self.setup_token()?;
        self.atts.set_position_increment(position_inc)?;
        self.atts.set_token_type(ty);
        Ok(())
    }
}

impl TokenStream for WikipediaTokenizer {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: WikipediaTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if let Some(state) = self.tokens.as_mut().and_then(VecDeque::pop_front) {
            self.atts.restore_state(&state);
            return Ok(true);
        }
        self.atts.clear_attributes();
        let token_type = self.scanner.get_next_token(self.input.reader()?)?;
        if token_type == YYEOF {
            return Ok(false);
        }
        let ty = TOKEN_TYPES[token_type as usize];
        // `TOKENS_ONLY` (the default) never looks the type up.
        if self.token_output == TOKENS_ONLY || !self.untokenized_types.contains(ty) {
            self.setup_token()?;
        } else if self.token_output == UNTOKENIZED_ONLY {
            self.collapse(token_type, ty, false)?;
        } else if self.token_output == BOTH {
            self.collapse(token_type, ty, true)?;
        }
        let mut posinc = self.scanner.position_inc;
        if self.first && posinc == 0 {
            posinc = 1; // don't emit posinc=0 for the first token!
        }
        self.atts.set_position_increment(posinc)?;
        self.atts.set_token_type(ty);
        self.first = false;
        Ok(true)
    }

    // Java: WikipediaTokenizer.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let f = self
            .input
            .correct_offset(self.scanner.yychar() + self.scanner.s.yylength() as i32);
        self.atts.set_offset(f, f)
    }

    // Java: WikipediaTokenizer.reset
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.scanner.s.yyreset();
        self.tokens = None;
        self.scanner.reset();
        self.first = true;
        Ok(())
    }

    // Java: WikipediaTokenizer.close
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.scanner.s.yyreset();
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for WikipediaTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;

    fn run(t: &mut WikipediaTokenizer, text: &str) -> Vec<String> {
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        crate::token_stream::consume(t, |a| {
            out.push(format!(
                "{}/{}/{}/{}",
                a.term(),
                a.token_type(),
                a.position_increment(),
                a.flags()
            ))
        })
        .unwrap();
        out
    }

    // Expected outputs are Lucene 10.5.0's.
    #[test]
    fn wiki_syntax_is_typed() {
        let mut t = WikipediaTokenizer::default();
        assert_eq!(
            run(&mut t, "[[Main Page]] '''bold''' [http://x.org site]"),
            vec![
                "Main/il/1/0",
                "Page/il/1/0",
                "bold/b/1/0",
                "http://x.org/elu/1/0",
                "site/el/0/0",
            ]
        );
    }

    #[test]
    fn untokenized_types_collapse() {
        let types: HashSet<String> = [INTERNAL_LINK.to_string()].into();
        let mut t = WikipediaTokenizer::new(BOTH, types.clone()).unwrap();
        assert_eq!(
            run(&mut t, "a [[Main Page]] b"),
            vec![
                "a/<ALPHANUM>/1/0",
                "Main Page/il/1/1",
                "Main/il/0/0",
                "Page/il/1/0",
                "b/<ALPHANUM>/1/0",
            ]
        );
        let mut t = WikipediaTokenizer::new(UNTOKENIZED_ONLY, types).unwrap();
        assert_eq!(
            run(&mut t, "a [[Main Page]]"),
            vec!["a/<ALPHANUM>/1/0", "Main Page/il/1/1"]
        );
        assert!(WikipediaTokenizer::new(7, HashSet::new()).is_err());
        assert_eq!(trim_utf16(&[0x20, 0x41, 0x9]), &[0x41]);
        assert_eq!(trim_utf16(&[0x20]), &[] as &[u16]);
    }
}
