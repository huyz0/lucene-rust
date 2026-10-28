//! `org.apache.lucene.queryparser.flexible.standard.parser.StandardSyntaxParser`:
//! the productions, ported from the generated parser one for one (each
//! `jj_la1` slot and `jj_2_<n>` lookahead where Java has it), building the
//! query node tree. The lookaheads themselves are generated
//! (`lookahead.rs`), as are the token kinds (`constants.rs`).

use super::constants::*;
use super::lookahead::{jj_3, JJ_LA1_0, JJ_LA1_1, NUM_LOOKAHEADS};
use super::nodes::{FieldData, IntervalFn, Kind, Modifier, Node, Text};
use super::token_manager::TokenManager;
use crate::queryparser::javacc::{CharStream, Grammar, JjError, Runtime};

static GRAMMAR: Grammar<TokenManager> = Grammar {
    token_image: &TOKEN_IMAGE,
    num_tokens: 57,
    la1_masks: &[&JJ_LA1_0, &JJ_LA1_1],
    lookahead: jj_3,
    num_lookaheads: NUM_LOOKAHEADS,
};

/// What parsing can throw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Fail {
    /// A `ParseException`, or a `TokenMgrError` (`Lexical error ...`): the
    /// message Java's `ParseException`/`TokenMgrError` carries.
    Syntax(String),
    /// `Integer.parseInt`'s `NumberFormatException` from a production's
    /// action, which `StandardSyntaxParser.parse` lets through.
    NumberFormat(String),
}

impl From<JjError> for Fail {
    fn from(e: JjError) -> Self {
        Fail::Syntax(e.0)
    }
}

impl From<crate::queryparser::javacc::TokenMgrError> for Fail {
    fn from(e: crate::queryparser::javacc::TokenMgrError) -> Self {
        Fail::Syntax(e.0)
    }
}

type PResult<T> = Result<T, Fail>;

/// A `ParseException` an action throws (`new ParseException(message)`):
/// its text never reaches the caller, `parse` replaces it.
fn action_error(key: &str) -> Fail {
    Fail::Syntax(key.to_string())
}

/// `EscapeQuerySyntaxImpl.discardEscapeChar(input)`: the text with its
/// escape characters removed, each escaped unit marked; `\uXXXX` is the
/// unit it names, not marked.
pub(crate) fn discard_escape_char(input: &[u16]) -> PResult<Text> {
    let mut output = Vec::with_capacity(input.len());
    let mut was_escaped = Vec::with_capacity(input.len());
    let mut last_char_was_escape_char = false;
    let mut code_point_multiplier = 0u32;
    let mut code_point = 0u32;
    for &cur in input {
        if code_point_multiplier > 0 {
            code_point += hex_to_int(cur)? * code_point_multiplier;
            code_point_multiplier >>= 4;
            if code_point_multiplier == 0 {
                output.push(code_point as u16);
                was_escaped.push(false);
                code_point = 0;
            }
        } else if last_char_was_escape_char {
            if cur == u16::from(b'u') {
                code_point_multiplier = 16 * 16 * 16;
            } else {
                output.push(cur);
                was_escaped.push(true);
            }
            last_char_was_escape_char = false;
        } else if cur == u16::from(b'\\') {
            last_char_was_escape_char = true;
        } else {
            output.push(cur);
            was_escaped.push(false);
        }
    }
    if code_point_multiplier > 0 {
        return Err(action_error("INVALID_SYNTAX_ESCAPE_UNICODE_TRUNCATION"));
    }
    if last_char_was_escape_char {
        return Err(action_error("INVALID_SYNTAX_ESCAPE_CHARACTER"));
    }
    Ok(Text {
        units: output,
        escaped: Some(was_escaped),
    })
}

/// `hexToInt(c)`.
fn hex_to_int(c: u16) -> PResult<u32> {
    match char::from_u32(u32::from(c)).and_then(|ch| ch.to_digit(16)) {
        // `'0'..'9'`, `'a'..'f'`, `'A'..'F'` only
        Some(d) if c < 128 => Ok(d),
        _ => Err(action_error("INVALID_SYNTAX_ESCAPE_NONE_HEX_UNICODE")),
    }
}

/// `Integer.parseInt(token.image)`.
fn parse_int(image: &[u16]) -> PResult<i32> {
    let s = String::from_utf16_lossy(image);
    s.parse::<i32>()
        .map_err(|_| Fail::NumberFormat(format!("For input string: \"{s}\"")))
}

/// `Float.parseFloat(token.image)`: a `NUMBER` is digits with at most one
/// fraction, which always parses.
fn parse_float(image: &[u16]) -> f32 {
    String::from_utf16_lossy(image)
        .parse::<f32>()
        .unwrap_or(0.0)
}

fn unquote(image: &[u16]) -> Vec<u16> {
    image[1..image.len() - 1].to_vec()
}

/// One parse of one query string.
pub(crate) struct Parser {
    rt: Runtime<TokenManager>,
    /// How many `(` groups the parse is inside, bounded by
    /// [`MAX_GROUP_DEPTH`].
    depth: usize,
}

/// The deepest `(` nesting a parse accepts: a Rust stack overflow aborts the
/// process where Java's `StackOverflowError` can be caught, so a hostile
/// `((((...` fails to parse instead. Lower than the classic parser's 128:
/// this grammar's productions keep larger frames, and an unoptimised build
/// overflows a 2 MiB thread stack at about 125 levels.
pub(crate) const MAX_GROUP_DEPTH: usize = 64;

impl Parser {
    pub(crate) fn new(query: &str) -> Self {
        Parser {
            rt: Runtime::new(&GRAMMAR, TokenManager::new(CharStream::new(query))),
            depth: 0,
        }
    }

    fn ntk(&mut self) -> PResult<i32> {
        Ok(self.rt.ntk()?)
    }

    fn consume(&mut self, kind: i32) -> PResult<usize> {
        Ok(self.rt.consume(kind)?)
    }

    fn no_match(&mut self, n: usize) -> Fail {
        Fail::from(self.rt.no_match(n))
    }

    fn fail(&mut self) -> Fail {
        Fail::from(self.rt.fail())
    }

    fn jj_2(&mut self, n: usize) -> PResult<bool> {
        Ok(self.rt.jj_2(n - 1, 2)?)
    }

    fn image(&self, t: usize) -> Vec<u16> {
        self.rt.tokens[t].image.clone()
    }

    fn kind(&self, t: usize) -> i32 {
        self.rt.tokens[t].kind
    }

    /// The last consumed token (`token`).
    fn token(&self) -> usize {
        self.rt.token
    }

    // --------------------------------------------------------- productions

    /// `TopLevelQuery(field)`.
    pub(crate) fn top_level_query(&mut self, field: Option<&str>) -> PResult<Node> {
        let q = self.query(field)?;
        self.consume(EOF)?;
        Ok(q)
    }

    fn starts_clause(kind: i32) -> bool {
        matches!(
            kind,
            NOT | FN_PREFIX
                | PLUS
                | MINUS
                | QUOTED
                | NUMBER
                | TERM
                | REGEXPTERM
                | RANGEIN_START
                | RANGEEX_START
                | LPAREN
        )
    }

    /// `Query(field)`.
    fn query(&mut self, field: Option<&str>) -> PResult<Node> {
        let mut clauses = Vec::new();
        loop {
            clauses.push(self.disj_query(field)?);
            if !Self::starts_clause(self.ntk()?) {
                self.rt.skipped(0);
                break;
            }
        }
        // a "pure" negation is wrapped in a boolean, which keeps it
        if clauses.len() == 1 && clauses[0].modifier() == Some(Modifier::Not) {
            let first = clauses.remove(0);
            clauses.push(Node::with(Kind::Boolean, vec![first]));
        }
        Ok(if clauses.len() == 1 {
            clauses.remove(0)
        } else {
            Node::with(Kind::Boolean, clauses)
        })
    }

    /// `DisjQuery(field)`.
    fn disj_query(&mut self, field: Option<&str>) -> PResult<Node> {
        let mut clauses = vec![self.conj_query(field)?];
        loop {
            if self.ntk()? != OR {
                self.rt.skipped(1);
                break;
            }
            self.consume(OR)?;
            clauses.push(self.conj_query(field)?);
        }
        Ok(if clauses.len() == 1 {
            clauses.remove(0)
        } else {
            Node::with(Kind::Or, clauses)
        })
    }

    /// `ConjQuery(field)`.
    fn conj_query(&mut self, field: Option<&str>) -> PResult<Node> {
        let mut clauses = vec![self.mod_clause(field)?];
        loop {
            if self.ntk()? != AND {
                self.rt.skipped(2);
                break;
            }
            self.consume(AND)?;
            clauses.push(self.mod_clause(field)?);
        }
        Ok(if clauses.len() == 1 {
            clauses.remove(0)
        } else {
            Node::with(Kind::And, clauses)
        })
    }

    /// `ModClause(field)`.
    fn mod_clause(&mut self, field: Option<&str>) -> PResult<Node> {
        let mut modifier = Modifier::None;
        match self.ntk()? {
            NOT | PLUS | MINUS => match self.ntk()? {
                PLUS => {
                    self.consume(PLUS)?;
                    modifier = Modifier::Req;
                }
                NOT | MINUS => {
                    match self.ntk()? {
                        MINUS => {
                            self.consume(MINUS)?;
                        }
                        NOT => {
                            self.consume(NOT)?;
                        }
                        _ => return Err(self.no_match(3)),
                    }
                    modifier = Modifier::Not;
                }
                _ => return Err(self.no_match(4)),
            },
            _ => self.rt.skipped(5),
        }
        let q = self.clause(field)?;
        Ok(if modifier != Modifier::None {
            Node::with(Kind::Modifier(modifier), vec![q])
        } else {
            q
        })
    }

    /// `Clause(field)`.
    fn clause(&mut self, field: Option<&str>) -> PResult<Node> {
        if self.jj_2(3)? {
            return self.field_range_expr();
        }
        let mut field = field.map(str::to_string);
        match self.ntk()? {
            FN_PREFIX | QUOTED | NUMBER | TERM | REGEXPTERM | RANGEIN_START | RANGEEX_START
            | LPAREN => {
                if self.jj_2(1)? {
                    field = Some(self.field_name()?);
                    match self.ntk()? {
                        OP_COLON => {
                            self.consume(OP_COLON)?;
                        }
                        OP_EQUAL => {
                            self.consume(OP_EQUAL)?;
                        }
                        _ => return Err(self.no_match(6)),
                    }
                }
                if self.jj_2(2)? {
                    self.term(field.as_deref())
                } else {
                    match self.ntk()? {
                        LPAREN => self.grouping_expr(field.as_deref()),
                        FN_PREFIX | QUOTED | NUMBER | TERM => self.interval_expr(field.as_deref()),
                        _ => Err(self.no_match(7)),
                    }
                }
            }
            _ => Err(self.no_match(8)),
        }
    }

    /// `FieldName()`.
    fn field_name(&mut self) -> PResult<String> {
        let name = self.consume(TERM)?;
        Ok(discard_escape_char(&self.image(name))?.as_string())
    }

    /// `GroupingExpr(field)`.
    fn grouping_expr(&mut self, field: Option<&str>) -> PResult<Node> {
        self.consume(LPAREN)?;
        if self.depth >= MAX_GROUP_DEPTH {
            return Err(Fail::Syntax(format!(
                "Cannot parse query: more than {MAX_GROUP_DEPTH} nested groups"
            )));
        }
        self.depth += 1;
        let q = self.query(field);
        self.depth -= 1;
        let mut q = q?;
        self.consume(RPAREN)?;
        if self.ntk()? == CARAT {
            q = self.boost(q)?;
        } else {
            self.rt.skipped(9);
        }
        let mut min_should_match = None;
        if self.ntk()? == 56 {
            self.consume(56)?;
            min_should_match = Some(self.consume(NUMBER)?);
        } else {
            self.rt.skipped(10);
        }
        Ok(match min_should_match {
            Some(t) => {
                let n = parse_int(&self.image(t))?;
                Node::with(
                    Kind::MinShouldMatch(n),
                    vec![Node::with(Kind::Group, vec![q])],
                )
            }
            None => Node::with(Kind::Group, vec![q]),
        })
    }

    /// `IntervalExpr(field)`.
    fn interval_expr(&mut self, field: Option<&str>) -> PResult<Node> {
        let source = self.interval_fun()?;
        Ok(Node::leaf(Kind::Interval {
            field: field.map(str::to_string),
            source,
        }))
    }

    /// `IntervalFun()`.
    fn interval_fun(&mut self) -> PResult<IntervalFn> {
        if self.jj_2(4)? {
            return self.interval_at_least();
        }
        if self.jj_2(5)? {
            return self.interval_number_one(MAXWIDTH, |n, s| IntervalFn::MaxWidth(n, Box::new(s)));
        }
        if self.jj_2(6)? {
            return self.interval_number_one(MAXGAPS, |n, s| IntervalFn::MaxGaps(n, Box::new(s)));
        }
        if self.jj_2(7)? {
            return self.interval_list(ORDERED, 13, IntervalFn::Ordered);
        }
        if self.jj_2(8)? {
            return self.interval_list(UNORDERED, 12, IntervalFn::Unordered);
        }
        if self.jj_2(9)? {
            return self.interval_two(UNORDERED_NO_OVERLAPS, |a, b| {
                IntervalFn::UnorderedNoOverlaps(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(10)? {
            return self.interval_list(FN_OR, 14, IntervalFn::Or);
        }
        if self.jj_2(11)? {
            return self.interval_wildcard();
        }
        if self.jj_2(12)? {
            return self.interval_two(AFTER, |a, b| IntervalFn::After(Box::new(a), Box::new(b)));
        }
        if self.jj_2(13)? {
            return self.interval_two(BEFORE, |a, b| IntervalFn::Before(Box::new(a), Box::new(b)));
        }
        if self.jj_2(14)? {
            return self.interval_list(PHRASE, 15, IntervalFn::Phrase);
        }
        if self.jj_2(15)? {
            return self.interval_two(CONTAINING, |a, b| {
                IntervalFn::Containing(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(16)? {
            return self.interval_two(NOT_CONTAINING, |a, b| {
                IntervalFn::NotContaining(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(17)? {
            return self.interval_two(CONTAINED_BY, |a, b| {
                IntervalFn::ContainedBy(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(18)? {
            return self.interval_two(NOT_CONTAINED_BY, |a, b| {
                IntervalFn::NotContainedBy(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(19)? {
            return self.interval_positions(WITHIN, |a, n, b| {
                IntervalFn::Within(Box::new(a), n, Box::new(b))
            });
        }
        if self.jj_2(20)? {
            return self.interval_positions(NOT_WITHIN, |a, n, b| {
                IntervalFn::NotWithin(Box::new(a), n, Box::new(b))
            });
        }
        if self.jj_2(21)? {
            return self.interval_two(OVERLAPPING, |a, b| {
                IntervalFn::Overlapping(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(22)? {
            return self.interval_two(NON_OVERLAPPING, |a, b| {
                IntervalFn::NonOverlapping(Box::new(a), Box::new(b))
            });
        }
        if self.jj_2(23)? {
            return self.interval_extend();
        }
        if self.jj_2(24)? {
            return self.interval_fuzzy_term();
        }
        if self.jj_2(25)? {
            return self.interval_text();
        }
        Err(self.fail())
    }

    fn starts_interval(kind: i32) -> bool {
        matches!(kind, FN_PREFIX | QUOTED | NUMBER | TERM)
    }

    /// `IntervalAtLeast()`.
    fn interval_at_least(&mut self) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(ATLEAST)?;
        self.consume(LPAREN)?;
        let min = self.consume(NUMBER)?;
        let mut sources = Vec::new();
        loop {
            sources.push(self.interval_fun()?);
            if !Self::starts_interval(self.ntk()?) {
                self.rt.skipped(11);
                break;
            }
        }
        self.consume(RPAREN)?;
        Ok(IntervalFn::AtLeast(parse_int(&self.image(min))?, sources))
    }

    /// `IntervalMaxWidth()` and `IntervalMaxGaps()`: a number, then one
    /// source.
    fn interval_number_one(
        &mut self,
        kind: i32,
        make: fn(i32, IntervalFn) -> IntervalFn,
    ) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(kind)?;
        self.consume(LPAREN)?;
        let n = self.consume(NUMBER)?;
        let source = self.interval_fun()?;
        self.consume(RPAREN)?;
        Ok(make(parse_int(&self.image(n))?, source))
    }

    /// `IntervalOrdered()`, `IntervalUnordered()`, `IntervalOr()`,
    /// `IntervalPhrase()`: one or more sources.
    fn interval_list(
        &mut self,
        kind: i32,
        la1: usize,
        make: fn(Vec<IntervalFn>) -> IntervalFn,
    ) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(kind)?;
        self.consume(LPAREN)?;
        let mut sources = Vec::new();
        loop {
            sources.push(self.interval_fun()?);
            if !Self::starts_interval(self.ntk()?) {
                self.rt.skipped(la1);
                break;
            }
        }
        self.consume(RPAREN)?;
        Ok(make(sources))
    }

    /// The two-source functions.
    fn interval_two(
        &mut self,
        kind: i32,
        make: fn(IntervalFn, IntervalFn) -> IntervalFn,
    ) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(kind)?;
        self.consume(LPAREN)?;
        let a = self.interval_fun()?;
        let b = self.interval_fun()?;
        self.consume(RPAREN)?;
        Ok(make(a, b))
    }

    /// `IntervalWithin()` and `IntervalNotWithin()`: a source, a number of
    /// positions, a reference.
    fn interval_positions(
        &mut self,
        kind: i32,
        make: fn(IntervalFn, i32, IntervalFn) -> IntervalFn,
    ) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(kind)?;
        self.consume(LPAREN)?;
        let a = self.interval_fun()?;
        let n = self.consume(NUMBER)?;
        let b = self.interval_fun()?;
        self.consume(RPAREN)?;
        Ok(make(a, parse_int(&self.image(n))?, b))
    }

    /// `IntervalExtend()`.
    fn interval_extend(&mut self) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(EXTEND)?;
        self.consume(LPAREN)?;
        let source = self.interval_fun()?;
        let before = self.consume(NUMBER)?;
        let after = self.consume(NUMBER)?;
        self.consume(RPAREN)?;
        Ok(IntervalFn::Extend(
            Box::new(source),
            parse_int(&self.image(before))?,
            parse_int(&self.image(after))?,
        ))
    }

    /// The word of `IntervalWildcard`/`IntervalFuzzyTerm`: a term or number
    /// as is, a quoted one unquoted.
    fn interval_word(&mut self, la1_inner: usize, la1_outer: usize) -> PResult<String> {
        match self.ntk()? {
            NUMBER | TERM => {
                match self.ntk()? {
                    TERM => {
                        self.consume(TERM)?;
                    }
                    NUMBER => {
                        self.consume(NUMBER)?;
                    }
                    _ => return Err(self.no_match(la1_inner)),
                }
                Ok(String::from_utf16_lossy(&self.image(self.token())))
            }
            QUOTED => {
                self.consume(QUOTED)?;
                Ok(String::from_utf16_lossy(&unquote(
                    &self.image(self.token()),
                )))
            }
            _ => Err(self.no_match(la1_outer)),
        }
    }

    /// `IntervalWildcard()`.
    fn interval_wildcard(&mut self) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(WILDCARD)?;
        self.consume(LPAREN)?;
        let wildcard = self.interval_word(16, 17)?;
        let mut max_expansions = None;
        if self.ntk()? == NUMBER {
            max_expansions = Some(self.consume(NUMBER)?);
        } else {
            self.rt.skipped(18);
        }
        self.consume(RPAREN)?;
        let max = match max_expansions {
            Some(t) => parse_int(&self.image(t))?,
            None => 0,
        };
        Ok(IntervalFn::Wildcard(wildcard, max))
    }

    /// `IntervalFuzzyTerm()`.
    fn interval_fuzzy_term(&mut self) -> PResult<IntervalFn> {
        self.consume(FN_PREFIX)?;
        self.consume(FUZZYTERM)?;
        self.consume(LPAREN)?;
        let term = self.interval_word(19, 20)?;
        let mut max_edits = None;
        if self.jj_2(26)? {
            max_edits = Some(self.consume(NUMBER)?);
        }
        let mut max_expansions = None;
        if self.jj_2(27)? {
            max_expansions = Some(self.consume(NUMBER)?);
        }
        self.consume(RPAREN)?;
        // `FuzzyQuery.defaultMaxEdits`, `Intervals.DEFAULT_MAX_EXPANSIONS`
        let max_edits = match max_edits {
            Some(t) => parse_int(&self.image(t))?,
            None => 2,
        };
        let max_expansions = match max_expansions {
            Some(t) => parse_int(&self.image(t))?,
            None => 128,
        };
        Ok(IntervalFn::FuzzyTerm(term, max_edits, max_expansions))
    }

    /// `IntervalText()`.
    fn interval_text(&mut self) -> PResult<IntervalFn> {
        match self.ntk()? {
            QUOTED => {
                self.consume(QUOTED)?;
                Ok(IntervalFn::AnalyzedText(String::from_utf16_lossy(
                    &unquote(&self.image(self.token())),
                )))
            }
            NUMBER | TERM => {
                match self.ntk()? {
                    TERM => {
                        self.consume(TERM)?;
                    }
                    NUMBER => {
                        self.consume(NUMBER)?;
                    }
                    _ => return Err(self.no_match(21)),
                }
                Ok(IntervalFn::AnalyzedText(String::from_utf16_lossy(
                    &self.image(self.token()),
                )))
            }
            _ => Err(self.no_match(22)),
        }
    }

    /// `Boost(node)`.
    fn boost(&mut self, node: Node) -> PResult<Node> {
        self.consume(CARAT)?;
        let boost = self.consume(NUMBER)?;
        Ok(Node::with(
            Kind::Boost(parse_float(&self.image(boost))),
            vec![node],
        ))
    }

    /// `FuzzyOp(field, term, node)`.
    fn fuzzy_op(&mut self, field: Option<&str>, term: usize) -> PResult<Node> {
        self.consume(TILDE)?;
        let mut similarity = None;
        if self.jj_2(28)? {
            similarity = Some(self.consume(NUMBER)?);
        }
        // `FuzzyQuery.defaultMaxEdits`
        let mut fms = 2.0f32;
        if let Some(t) = similarity {
            fms = parse_float(&self.image(t));
            if fms < 0.0 {
                return Err(action_error("INVALID_SYNTAX_FUZZY_LIMITS"));
            } else if fms >= 1.0 && fms != (fms as i32) as f32 {
                return Err(action_error("INVALID_SYNTAX_FUZZY_EDITS"));
            }
        }
        let text = discard_escape_char(&self.image(term))?;
        Ok(Node::leaf(Kind::Fuzzy {
            data: FieldData::new(field.map(str::to_string), text),
            similarity: fms,
            prefix_length: 0,
        }))
    }

    /// `FieldRangeExpr(field)`.
    fn field_range_expr(&mut self) -> PResult<Node> {
        let field = Some(self.field_name()?);
        match self.ntk()? {
            k @ (OP_LESSTHAN | OP_LESSTHANEQ | OP_MORETHAN | OP_MORETHANEQ) => {
                self.consume(k)?;
            }
            _ => return Err(self.no_match(23)),
        }
        let operator = self.kind(self.token());
        match self.ntk()? {
            k @ (TERM | QUOTED | NUMBER) => {
                self.consume(k)?;
            }
            _ => return Err(self.no_match(24)),
        }
        let term = self.token();
        let mut image = self.image(term);
        if self.kind(term) == QUOTED {
            image = unquote(&image);
        }
        let star = || Text::plain("*");
        let bound = |t: Text| Node::leaf(Kind::Field(FieldData::new(field.clone(), t)));
        let (lower, upper, lower_inclusive, upper_inclusive) = match operator {
            OP_LESSTHAN => (
                bound(star()),
                bound(discard_escape_char(&image)?),
                true,
                false,
            ),
            OP_LESSTHANEQ => (
                bound(star()),
                bound(discard_escape_char(&image)?),
                true,
                true,
            ),
            OP_MORETHAN => (
                bound(discard_escape_char(&image)?),
                bound(star()),
                false,
                true,
            ),
            _ => (
                bound(discard_escape_char(&image)?),
                bound(star()),
                true,
                true,
            ),
        };
        Ok(Node::with(
            Kind::TermRange {
                lower_inclusive,
                upper_inclusive,
            },
            vec![lower, upper],
        ))
    }

    /// `Term(field)`.
    fn term(&mut self, field: Option<&str>) -> PResult<Node> {
        let fld = field.map(str::to_string);
        let q = match self.ntk()? {
            REGEXPTERM => {
                let term = self.consume(REGEXPTERM)?;
                let v = unquote(&self.image(term));
                Node::leaf(Kind::Regexp {
                    field: fld,
                    text: Text {
                        units: v,
                        escaped: None,
                    },
                })
            }
            NUMBER | TERM => {
                let term = match self.ntk()? {
                    TERM => self.consume(TERM)?,
                    NUMBER => self.consume(NUMBER)?,
                    _ => return Err(self.no_match(25)),
                };
                let q = Node::leaf(Kind::Field(FieldData::new(
                    fld,
                    discard_escape_char(&self.image(term))?,
                )));
                if self.ntk()? == TILDE {
                    self.fuzzy_op(field, term)?
                } else {
                    self.rt.skipped(26);
                    q
                }
            }
            RANGEIN_START | RANGEEX_START => self.term_range_expr(field)?,
            QUOTED => self.quoted_term(field)?,
            _ => return Err(self.no_match(27)),
        };
        if self.ntk()? == CARAT {
            self.boost(q)
        } else {
            self.rt.skipped(28);
            Ok(q)
        }
    }

    /// `QuotedTerm(field)`.
    fn quoted_term(&mut self, field: Option<&str>) -> PResult<Node> {
        let term = self.consume(QUOTED)?;
        let image = unquote(&self.image(term));
        let q = Node::leaf(Kind::Quoted(FieldData::new(
            field.map(str::to_string),
            discard_escape_char(&image)?,
        )));
        if self.ntk()? == TILDE {
            self.consume(TILDE)?;
            let slop = self.consume(NUMBER)?;
            Ok(Node::with(
                Kind::Slop(parse_int(&self.image(slop))?),
                vec![q],
            ))
        } else {
            self.rt.skipped(29);
            Ok(q)
        }
    }

    /// The bound of `TermRangeExpr`: `RANGE_GOOP`, `RANGE_QUOTED` (unquoted)
    /// or `RANGE_TO`.
    fn range_bound(&mut self, la1: usize) -> PResult<Vec<u16>> {
        match self.ntk()? {
            k @ (RANGE_GOOP | RANGE_QUOTED | RANGE_TO) => {
                self.consume(k)?;
            }
            _ => return Err(self.no_match(la1)),
        }
        let t = self.token();
        let image = self.image(t);
        Ok(if self.kind(t) == RANGE_QUOTED {
            unquote(&image)
        } else {
            image
        })
    }

    /// `TermRangeExpr(field)`.
    fn term_range_expr(&mut self, field: Option<&str>) -> PResult<Node> {
        let mut left_inclusive = false;
        let mut right_inclusive = false;
        match self.ntk()? {
            RANGEIN_START => {
                self.consume(RANGEIN_START)?;
                left_inclusive = true;
            }
            RANGEEX_START => {
                self.consume(RANGEEX_START)?;
            }
            _ => return Err(self.no_match(30)),
        }
        let left = self.range_bound(31)?;
        self.consume(RANGE_TO)?;
        let right = self.range_bound(32)?;
        match self.ntk()? {
            RANGEIN_END => {
                self.consume(RANGEIN_END)?;
                right_inclusive = true;
            }
            RANGEEX_END => {
                self.consume(RANGEEX_END)?;
            }
            _ => return Err(self.no_match(33)),
        }
        let fld = field.map(str::to_string);
        let lower = Node::leaf(Kind::Field(FieldData::new(
            fld.clone(),
            discard_escape_char(&left)?,
        )));
        let upper = Node::leaf(Kind::Field(FieldData::new(
            fld,
            discard_escape_char(&right)?,
        )));
        Ok(Node::with(
            Kind::TermRange {
                lower_inclusive: left_inclusive,
                upper_inclusive: right_inclusive,
            },
            vec![lower, upper],
        ))
    }
}
