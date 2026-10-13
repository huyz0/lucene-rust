//! `org.apache.lucene.queryparser.simple.SimpleQueryParser`: the forgiving
//! syntax of `+` (and), `|` (or), `-` (not), `"phrase"~slop`, `prefix*`,
//! `word~edits`, `( )` precedence and `\` escapes, each switchable by a flag.
//! Hand-written in Java, and ported statement for statement: a query is
//! never rejected, a stray operator is dropped or taken as text.

use lucene_analysis::java_character::decimal_digit_value;
use lucene_analysis::Analyzer;

use super::classic::{QueryParserError, MAX_CLAUSE_COUNT};
use super::{ParsedBoolean, ParsedQuery};
use crate::query::{FuzzyQuery, PrefixQuery};
use crate::query_builder::QueryBuilder;
use crate::query_visitor::Occur;

/// `AND_OPERATOR`: `+`.
pub const AND_OPERATOR: i32 = 1 << 0;
/// `NOT_OPERATOR`: `-`.
pub const NOT_OPERATOR: i32 = 1 << 1;
/// `OR_OPERATOR`: `|`.
pub const OR_OPERATOR: i32 = 1 << 2;
/// `PREFIX_OPERATOR`: a trailing `*`.
pub const PREFIX_OPERATOR: i32 = 1 << 3;
/// `PHRASE_OPERATOR`: `"`.
pub const PHRASE_OPERATOR: i32 = 1 << 4;
/// `PRECEDENCE_OPERATORS`: `(` and `)`.
pub const PRECEDENCE_OPERATORS: i32 = 1 << 5;
/// `ESCAPE_OPERATOR`: `\`.
pub const ESCAPE_OPERATOR: i32 = 1 << 6;
/// `WHITESPACE_OPERATOR`: space, tab, newline and carriage return separate
/// tokens.
pub const WHITESPACE_OPERATOR: i32 = 1 << 7;
/// `FUZZY_OPERATOR`: `~`.
pub const FUZZY_OPERATOR: i32 = 1 << 8;
/// `NEAR_OPERATOR`: `~` after a phrase.
pub const NEAR_OPERATOR: i32 = 1 << 9;

/// `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE`.
const MAXIMUM_SUPPORTED_DISTANCE: i32 = 2;

/// `SimpleQueryParser`.
pub struct SimpleQueryParser {
    analyzer: Analyzer,
    weights: Vec<(String, f32)>,
    flags: i32,
    default_operator: Occur,
}

/// `State`: a position in the text, the operators seen and the query so far.
struct State {
    index: usize,
    length: usize,
    current_operation: Option<Occur>,
    previous_operation: Option<Occur>,
    not: i32,
    top: Option<ParsedQuery>,
}

impl State {
    fn new(index: usize, length: usize) -> Self {
        State {
            index,
            length,
            current_operation: None,
            previous_operation: None,
            not: 0,
            top: None,
        }
    }
}

type PResult<T> = Result<T, QueryParserError>;

/// One call of `parse`: the text, and `State`'s shared scratch buffer.
struct Run<'a> {
    p: &'a SimpleQueryParser,
    data: Vec<u16>,
    buffer: Vec<u16>,
}

fn unit(c: char) -> u16 {
    c as u16
}

impl SimpleQueryParser {
    /// `SimpleQueryParser(analyzer, field)`: every operator, one field.
    pub fn new(analyzer: Analyzer, field: impl Into<String>) -> Self {
        Self::with_weights(analyzer, vec![(field.into(), 1.0)], -1)
    }

    /// `SimpleQueryParser(analyzer, weights, flags)`: the fields in the order
    /// given (Java iterates a `Map`; pass the order it would), each with the
    /// boost its queries get.
    pub fn with_weights(analyzer: Analyzer, weights: Vec<(String, f32)>, flags: i32) -> Self {
        SimpleQueryParser {
            analyzer,
            weights,
            flags,
            default_operator: Occur::Should,
        }
    }

    /// `getDefaultOperator()`.
    pub fn default_operator(&self) -> Occur {
        self.default_operator
    }

    /// `setDefaultOperator(operator)`.
    ///
    /// # Errors
    /// `IllegalArgumentException` for an operator other than `SHOULD` or
    /// `MUST`.
    pub fn set_default_operator(&mut self, operator: Occur) -> Result<(), QueryParserError> {
        if operator != Occur::Should && operator != Occur::Must {
            return Err(QueryParserError::illegal_argument(
                "invalid operator: only SHOULD or MUST are allowed",
            ));
        }
        self.default_operator = operator;
        Ok(())
    }

    /// `parse(queryText)`.
    ///
    /// # Errors
    /// The analyzer's errors, and `TooManyClauses`.
    pub fn parse(&self, query_text: &str) -> PResult<ParsedQuery> {
        // `String.trim()`: every character up to U+0020.
        if query_text.trim_matches(|c: char| c <= ' ') == "*" {
            return Ok(ParsedQuery::MatchAll);
        }
        let data: Vec<u16> = query_text.encode_utf16().collect();
        let length = data.len();
        let mut run = Run {
            p: self,
            buffer: vec![0; length],
            data,
        };
        let mut state = State::new(0, length);
        run.parse_sub_query(&mut state)?;
        Ok(state.top.unwrap_or_else(|| {
            ParsedQuery::MatchNoDocs("empty string passed to query parser".into())
        }))
    }
}

impl Run<'_> {
    fn has(&self, flag: i32) -> bool {
        self.p.flags & flag != 0
    }

    fn at(&self, i: usize) -> u16 {
        self.data[i]
    }

    /// `parseSubQuery(state)`.
    fn parse_sub_query(&mut self, state: &mut State) -> PResult<()> {
        while state.index < state.length {
            let c = self.at(state.index);
            if c == unit('(') && self.has(PRECEDENCE_OPERATORS) {
                self.consume_sub_query(state)?;
            } else if c == unit(')') && self.has(PRECEDENCE_OPERATORS) {
                state.index += 1;
            } else if c == unit('"') && self.has(PHRASE_OPERATOR) {
                self.consume_phrase(state)?;
            } else if c == unit('+') && self.has(AND_OPERATOR) {
                if state.current_operation.is_none() && state.top.is_some() {
                    state.current_operation = Some(Occur::Must);
                }
                state.index += 1;
            } else if c == unit('|') && self.has(OR_OPERATOR) {
                if state.current_operation.is_none() && state.top.is_some() {
                    state.current_operation = Some(Occur::Should);
                }
                state.index += 1;
            } else if c == unit('-') && self.has(NOT_OPERATOR) {
                state.not += 1;
                state.index += 1;
                // not reset: a run of `-` counts
                continue;
            } else if matches!(c, 0x20 | 0x09 | 0x0a | 0x0d) && self.has(WHITESPACE_OPERATOR) {
                state.index += 1;
            } else {
                self.consume_token(state)?;
            }
            state.not = 0;
        }
        Ok(())
    }

    /// `consumeSubQuery(state)`.
    fn consume_sub_query(&mut self, state: &mut State) -> PResult<()> {
        state.index += 1;
        let start = state.index;
        let mut precedence = 1;
        let mut escaped = false;
        while state.index < state.length {
            if !escaped {
                let c = self.at(state.index);
                if c == unit('\\') && self.has(ESCAPE_OPERATOR) {
                    escaped = true;
                    state.index += 1;
                    continue;
                } else if c == unit('(') {
                    precedence += 1;
                } else if c == unit(')') {
                    precedence -= 1;
                    if precedence == 0 {
                        break;
                    }
                }
            }
            escaped = false;
            state.index += 1;
        }
        if state.index == state.length {
            // no closing parenthesis: the `(` is skipped and the rest is
            // parsed again as the query
            state.index = start;
        } else if state.index == start {
            state.current_operation = None;
            state.index += 1;
        } else {
            let mut sub = State::new(start, state.index);
            self.parse_sub_query(&mut sub)?;
            self.build_query_tree(state, sub.top)?;
            state.index += 1;
        }
        Ok(())
    }

    /// `consumePhrase(state)`.
    fn consume_phrase(&mut self, state: &mut State) -> PResult<()> {
        state.index += 1;
        let start = state.index;
        let mut copied = 0;
        let mut escaped = false;
        let mut has_slop = false;
        while state.index < state.length {
            if !escaped {
                let c = self.at(state.index);
                if c == unit('\\') && self.has(ESCAPE_OPERATOR) {
                    escaped = true;
                    state.index += 1;
                    continue;
                } else if c == unit('"') {
                    if state.length > state.index + 1
                        && self.at(state.index + 1) == unit('~')
                        && self.has(NEAR_OPERATOR)
                    {
                        state.index += 1;
                        if state.length > state.index + 1 {
                            has_slop = true;
                        }
                    }
                    break;
                }
            }
            escaped = false;
            self.buffer[copied] = self.at(state.index);
            copied += 1;
            state.index += 1;
        }
        if state.index == state.length {
            state.index = start;
        } else if state.index == start {
            state.current_operation = None;
            state.index += 1;
        } else {
            let phrase = String::from_utf16_lossy(&self.buffer[..copied]);
            let slop = if has_slop {
                self.parse_fuzziness(state)
            } else {
                0
            };
            let branch = self.new_phrase_query(&phrase, slop)?;
            self.build_query_tree(state, branch)?;
            state.index += 1;
        }
        Ok(())
    }

    /// `consumeToken(state)`.
    fn consume_token(&mut self, state: &mut State) -> PResult<()> {
        let mut copied = 0;
        let mut escaped = false;
        let mut prefix = false;
        let mut fuzzy = false;
        while state.index < state.length {
            if !escaped {
                let c = self.at(state.index);
                if c == unit('\\') && self.has(ESCAPE_OPERATOR) {
                    escaped = true;
                    prefix = false;
                    state.index += 1;
                    continue;
                } else if self.token_finished(state) {
                    break;
                } else if copied > 0 && c == unit('~') && self.has(FUZZY_OPERATOR) {
                    fuzzy = true;
                    break;
                }
                prefix = copied > 0 && c == unit('*') && self.has(PREFIX_OPERATOR);
            }
            escaped = false;
            self.buffer[copied] = self.at(state.index);
            copied += 1;
            state.index += 1;
        }
        if copied > 0 {
            let branch = if fuzzy && self.has(FUZZY_OPERATOR) {
                let token = String::from_utf16_lossy(&self.buffer[..copied]);
                let fuzziness = self.parse_fuzziness(state).min(MAXIMUM_SUPPORTED_DISTANCE);
                if fuzziness == 0 {
                    self.new_default_query(&token)?
                } else {
                    self.new_fuzzy_query(&token, fuzziness)?
                }
            } else if prefix {
                let token = String::from_utf16_lossy(&self.buffer[..copied - 1]);
                self.new_prefix_query(&token)?
            } else {
                let token = String::from_utf16_lossy(&self.buffer[..copied]);
                self.new_default_query(&token)?
            };
            self.build_query_tree(state, branch)?;
        }
        Ok(())
    }

    /// `addClause(bq, query, occur)`.
    fn add_clause(bq: ParsedQuery, query: ParsedQuery, occur: Occur) -> PResult<ParsedQuery> {
        let ParsedQuery::Boolean(mut b) = bq else {
            // `(BooleanQuery) state.top`
            return Err(QueryParserError::runtime("ClassCastException"));
        };
        if b.clauses.len() >= MAX_CLAUSE_COUNT {
            return Err(QueryParserError::too_many_clauses());
        }
        b.clauses.push((occur, query));
        Ok(ParsedQuery::Boolean(b))
    }

    /// `buildQueryTree(state, branch)`.
    fn build_query_tree(&mut self, state: &mut State, branch: Option<ParsedQuery>) -> PResult<()> {
        let Some(mut branch) = branch else {
            return Ok(());
        };
        if state.not % 2 == 1 {
            branch = ParsedQuery::Boolean(ParsedBoolean {
                clauses: vec![
                    (Occur::MustNot, branch),
                    (Occur::Should, ParsedQuery::MatchAll),
                ],
                minimum_should_match: 0,
            });
        }
        match state.top.take() {
            None => state.top = Some(branch),
            Some(top) => {
                let op = *state
                    .current_operation
                    .get_or_insert(self.p.default_operator);
                let mut top = top;
                if state.previous_operation != Some(op) {
                    top = ParsedQuery::Boolean(ParsedBoolean {
                        clauses: vec![(op, top)],
                        minimum_should_match: 0,
                    });
                }
                state.top = Some(Self::add_clause(top, branch, op)?);
                state.previous_operation = Some(op);
            }
        }
        state.current_operation = None;
        Ok(())
    }

    /// `parseFuzziness(state)`: the number after a `~`, up to the next
    /// token boundary.
    fn parse_fuzziness(&mut self, state: &mut State) -> i32 {
        if self.at(state.index) != unit('~') {
            return 0;
        }
        let mut slop_text: Vec<u16> = Vec::new();
        while state.index < state.length {
            state.index += 1;
            if state.index < state.length {
                if self.token_finished(state) {
                    break;
                }
                slop_text.push(self.at(state.index));
            }
        }
        let mut fuzziness = if slop_text.is_empty() {
            2
        } else {
            parse_java_int(&slop_text).unwrap_or(0)
        };
        if fuzziness < 0 {
            fuzziness = 0;
        }
        fuzziness
    }

    /// `tokenFinished(state)`.
    fn token_finished(&self, state: &State) -> bool {
        let c = self.at(state.index);
        (c == unit('"') && self.has(PHRASE_OPERATOR))
            || (c == unit('|') && self.has(OR_OPERATOR))
            || (c == unit('+') && self.has(AND_OPERATOR))
            || (c == unit('(') && self.has(PRECEDENCE_OPERATORS))
            || (c == unit(')') && self.has(PRECEDENCE_OPERATORS))
            || (matches!(c, 0x20 | 0x09 | 0x0a | 0x0d) && self.has(WHITESPACE_OPERATOR))
    }

    fn builder(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(&self.p.analyzer)
    }

    /// The weight loop of `newDefaultQuery`, `newPhraseQuery`, ...: each
    /// field's query (when it has one), boosted unless its weight is 1,
    /// `SHOULD` in a boolean that `simplify` then flattens.
    fn each_field(
        &self,
        mut make: impl FnMut(&str) -> PResult<Option<ParsedQuery>>,
    ) -> PResult<Option<ParsedQuery>> {
        let mut clauses = Vec::new();
        for (field, boost) in &self.p.weights {
            if let Some(mut q) = make(field)? {
                if *boost != 1.0 {
                    q = ParsedQuery::Boost(Box::new(q), *boost);
                }
                if clauses.len() >= MAX_CLAUSE_COUNT {
                    return Err(QueryParserError::too_many_clauses());
                }
                clauses.push((Occur::Should, q));
            }
        }
        Ok(simplify(clauses))
    }

    /// `newDefaultQuery(text)`.
    fn new_default_query(&self, text: &str) -> PResult<Option<ParsedQuery>> {
        let op = self.p.default_operator;
        self.each_field(|field| {
            let q = self
                .builder()
                .create_boolean_query_with(field, text, op)
                .map_err(QueryParserError::from_search)?;
            Ok(q.map(ParsedQuery::from_clause))
        })
    }

    /// `newFuzzyQuery(text, fuzziness)`.
    fn new_fuzzy_query(&self, text: &str, fuzziness: i32) -> PResult<Option<ParsedQuery>> {
        self.each_field(|field| {
            let term = self
                .p
                .analyzer
                .normalize(field, text)
                .map_err(QueryParserError::from_analysis)?;
            let q = FuzzyQuery::new(field, term).with_max_edits(fuzziness as u8);
            Ok(Some(ParsedQuery::Fuzzy(q)))
        })
    }

    /// `newPhraseQuery(text, slop)`.
    fn new_phrase_query(&self, text: &str, slop: i32) -> PResult<Option<ParsedQuery>> {
        self.each_field(|field| {
            let q = self
                .builder()
                .create_phrase_query_with_slop(field, text, slop.max(0) as u32)
                .map_err(QueryParserError::from_search)?;
            Ok(q.map(ParsedQuery::from_clause))
        })
    }

    /// `newPrefixQuery(text)`.
    fn new_prefix_query(&self, text: &str) -> PResult<Option<ParsedQuery>> {
        self.each_field(|field| {
            let term = self
                .p
                .analyzer
                .normalize(field, text)
                .map_err(QueryParserError::from_analysis)?;
            Ok(Some(ParsedQuery::Prefix(PrefixQuery::new(field, term))))
        })
    }
}

/// `simplify(bq)`: no clauses is `null`, one is that clause's query.
fn simplify(mut clauses: Vec<(Occur, ParsedQuery)>) -> Option<ParsedQuery> {
    match clauses.len() {
        0 => None,
        1 => clauses.pop().map(|(_, q)| q),
        _ => Some(ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: 0,
        })),
    }
}

/// `Integer.parseInt(String)`: an optional sign, then decimal digits (any
/// Unicode `Nd` of the Basic Multilingual Plane, which `Character.digit`
/// accepts); `None` for what throws `NumberFormatException`, an overflow
/// included.
fn parse_java_int(units: &[u16]) -> Option<i32> {
    let (negative, digits) = match units.first()? {
        0x2d => (true, &units[1..]),
        0x2b => (false, &units[1..]),
        _ => (false, units),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for &u in digits {
        let d = decimal_digit_value(u32::from(u))?;
        value = value * 10 + i64::from(d);
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::StandardAnalyzer;

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn parse_java_int_is_integer_parse_int() {
        assert_eq!(parse_java_int(&units("12")), Some(12));
        assert_eq!(parse_java_int(&units("+7")), Some(7));
        assert_eq!(parse_java_int(&units("-7")), Some(-7));
        assert_eq!(parse_java_int(&units("-2147483648")), Some(i32::MIN));
        assert_eq!(parse_java_int(&units("2147483647")), Some(i32::MAX));
        assert_eq!(parse_java_int(&units("2147483648")), None);
        assert_eq!(parse_java_int(&units("99999999999")), None);
        assert_eq!(parse_java_int(&units("")), None);
        assert_eq!(parse_java_int(&units("-")), None);
        assert_eq!(parse_java_int(&units("+")), None);
        assert_eq!(parse_java_int(&units("1x")), None);
        assert_eq!(parse_java_int(&units("1.5")), None);
        // Arabic-Indic and fullwidth digits are digits to `Character.digit`
        assert_eq!(parse_java_int(&units("\u{661}\u{662}")), Some(12));
        assert_eq!(parse_java_int(&units("\u{ff11}")), Some(1));
        // a supplementary digit is two surrogate units, neither a digit
        assert_eq!(parse_java_int(&units("\u{1d7ce}")), None);
    }

    #[test]
    fn a_boolean_past_the_clause_limit_is_too_many_clauses() {
        let p = SimpleQueryParser::new(Analyzer::new(StandardAnalyzer::new()), "f");
        let words: Vec<String> = (0..1100).map(|i| format!("t{i}")).collect();
        let e = p.parse(&words.join(" ")).unwrap_err();
        assert_eq!(e.class, "TooManyClauses");
        assert_eq!(e.message, "maxClauseCount is set to 1024");
        assert!(p.parse(&words[..1024].join(" ")).is_ok());
    }

    #[test]
    fn too_many_fields_is_too_many_clauses() {
        let weights = (0..1100).map(|i| (format!("f{i}"), 1.0)).collect();
        let p =
            SimpleQueryParser::with_weights(Analyzer::new(StandardAnalyzer::new()), weights, -1);
        assert_eq!(p.parse("a").unwrap_err().class, "TooManyClauses");
    }

    #[test]
    fn a_star_alone_matches_everything_and_blank_matches_nothing() {
        let p = SimpleQueryParser::new(Analyzer::new(StandardAnalyzer::new()), "f");
        assert_eq!(p.parse(" \t* \n").unwrap(), ParsedQuery::MatchAll);
        assert_eq!(p.parse("   ").unwrap().class_name(), "MatchNoDocsQuery");
    }
}
