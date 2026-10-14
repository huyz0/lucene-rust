//! `org.apache.lucene.queryparser.surround.parser.QueryParser` (the JavaCC 7
//! parser generated from the surround `QueryParser.jj`) over the shared
//! runtime: the productions as generated, and the grammar's own
//! `getXxxQuery` helpers.

use lucene_analysis::payloads::parse_java_float;

use super::constants::*;
use super::token_manager::TokenManager;
use crate::queryparser::classic::QueryParserError;
use crate::queryparser::javacc::{CharStream, Grammar, JjError, Runtime, Stop, TokenMgrError};
use crate::queryparser::surround::query::SrndQuery;

const MINIMUM_PREFIX_LENGTH: usize = 3;
const MINIMUM_CHARS_IN_TRUNC: usize = 3;
const TRUNCATION_ERROR_MESSAGE: &str = "Too unrestrictive truncation: ";
const BOOST_ERROR_MESSAGE: &str = "Cannot handle boost value: ";

/// `TRUNCATOR`.
const TRUNCATOR: char = '*';
/// `ANY_CHAR`.
const ANY_CHAR: char = '?';
/// `FIELD_OPERATOR`.
const FIELD_OPERATOR: char = ':';

/// `jj_la1_0`.
static JJ_LA1_0: [u32; 10] = [
    0x100, 0x200, 0x400, 0x1000, 0x800, 0x7c3b00, 0x1b00, 0x8000, 0x7c0000, 0x20000,
];

/// The surround grammar for the shared JavaCC runtime.
static GRAMMAR: Grammar<TokenManager> = Grammar {
    token_image: &TOKEN_IMAGE,
    num_tokens: 24,
    la1_masks: &[&JJ_LA1_0],
    lookahead: jj_3,
    num_lookaheads: 1,
};

/// `jj_3_1()`: `TERM ":"`.
fn jj_3(rt: &mut Runtime<TokenManager>, _n: usize) -> Result<bool, Stop> {
    if rt.scan_token(TERM)? {
        return Ok(true);
    }
    if rt.scan_token(COLON)? {
        return Ok(true);
    }
    Ok(false)
}

/// What a production can throw: a `ParseException` message, or another
/// exception.
enum Fail {
    Parse(String),
    Other(QueryParserError),
}

impl From<JjError> for Fail {
    fn from(e: JjError) -> Self {
        Fail::Parse(e.0)
    }
}

impl From<TokenMgrError> for Fail {
    fn from(e: TokenMgrError) -> Self {
        Fail::Parse(e.0)
    }
}

type PResult<T> = Result<T, Fail>;

fn units(image: &[u16]) -> String {
    String::from_utf16_lossy(image)
}

/// `parse(query)`: the query tree, or the `ParseException`.
///
/// # Errors
/// `ParseException` with the generated parser's message (token, column and
/// expected tokens), `TokenMgrError`'s message for a lexical error as a
/// `ParseException`, and the exceptions of the query constructors.
pub fn parse(query: &str) -> Result<SrndQuery, QueryParserError> {
    let mut p = Parser {
        rt: Runtime::new(&GRAMMAR, TokenManager::new(CharStream::new(query))),
    };
    match p.top_srnd_query() {
        Ok(q) => Ok(q),
        Err(Fail::Parse(msg)) => Err(QueryParserError::parse_exception(msg)),
        Err(Fail::Other(e)) => Err(e),
    }
}

struct Parser {
    rt: Runtime<TokenManager>,
}

impl Parser {
    fn ntk(&mut self) -> PResult<i32> {
        Ok(self.rt.ntk()?)
    }

    fn consume(&mut self, kind: i32) -> PResult<usize> {
        Ok(self.rt.consume(kind)?)
    }

    fn image(&self, t: usize) -> Vec<u16> {
        self.rt.tokens[t].image.clone()
    }

    /// `TopSrndQuery()`.
    fn top_srnd_query(&mut self) -> PResult<SrndQuery> {
        let q = self.fields_query()?;
        self.consume(EOF)?;
        Ok(q)
    }

    /// `FieldsQuery()`.
    fn fields_query(&mut self) -> PResult<SrndQuery> {
        let field_names = self.optional_fields()?;
        let q = self.or_query()?;
        Ok(match field_names {
            None => q,
            // `getFieldsQuery(q, fieldNames)`
            Some(names) => SrndQuery::fields(q, names, FIELD_OPERATOR),
        })
    }

    /// `OptionalFields()`.
    fn optional_fields(&mut self) -> PResult<Option<Vec<String>>> {
        let mut field_names: Option<Vec<String>> = None;
        while self.rt.jj_2(0, 2)? {
            let field_name = self.consume(TERM)?;
            self.consume(COLON)?;
            field_names
                .get_or_insert_with(Vec::new)
                .push(units(&self.image(field_name)));
        }
        Ok(field_names)
    }

    /// `OrQuery()`, `AndQuery()` and `NotQuery()`: `operand (OP operand)*`,
    /// keeping the last operator token.
    fn infix(
        &mut self,
        op: i32,
        la1: usize,
        operand: fn(&mut Self) -> PResult<SrndQuery>,
        make: fn(Vec<SrndQuery>, String) -> SrndQuery,
    ) -> PResult<SrndQuery> {
        let mut q = operand(self)?;
        let mut queries: Option<Vec<SrndQuery>> = None;
        let mut oprt = None;
        loop {
            if self.ntk()? != op {
                self.rt.skipped(la1);
                break;
            }
            oprt = Some(self.consume(op)?);
            // keep only last used operator
            if queries.is_none() {
                queries = Some(vec![q]);
            }
            q = operand(self)?;
            queries.get_or_insert_with(Vec::new).push(q.clone());
        }
        Ok(match (queries, oprt) {
            (Some(queries), Some(oprt)) => make(queries, units(&self.image(oprt))),
            _ => q,
        })
    }

    fn or_query(&mut self) -> PResult<SrndQuery> {
        self.infix(OR, 0, Self::and_query, |qs, name| {
            SrndQuery::or(qs, true, name)
        })
    }

    fn and_query(&mut self) -> PResult<SrndQuery> {
        self.infix(AND, 1, Self::not_query, |qs, name| {
            SrndQuery::and(qs, true, name)
        })
    }

    fn not_query(&mut self) -> PResult<SrndQuery> {
        self.infix(NOT, 2, Self::n_query, SrndQuery::not)
    }

    /// `NQuery()`.
    fn n_query(&mut self) -> PResult<SrndQuery> {
        let mut q = self.w_query()?;
        loop {
            if self.ntk()? != N {
                self.rt.skipped(3);
                break;
            }
            let dt = self.consume(N)?;
            let mut queries = vec![q];
            queries.push(self.w_query()?);
            q = self.distance_query(queries, true, dt, false)?;
        }
        Ok(q)
    }

    /// `WQuery()`.
    fn w_query(&mut self) -> PResult<SrndQuery> {
        let mut q = self.primary_query()?;
        loop {
            if self.ntk()? != W {
                self.rt.skipped(4);
                break;
            }
            let wt = self.consume(W)?;
            let mut queries = vec![q];
            queries.push(self.primary_query()?);
            q = self.distance_query(queries, true, wt, true)?;
        }
        Ok(q)
    }

    /// `PrimaryQuery()`.
    fn primary_query(&mut self) -> PResult<SrndQuery> {
        let mut q = match self.ntk()? {
            LPAREN => {
                self.consume(LPAREN)?;
                let q = self.fields_query()?;
                self.consume(RPAREN)?;
                q
            }
            OR | AND | W | N => self.prefix_operator_query()?,
            TRUNCQUOTED | QUOTED | SUFFIXTERM | TRUNCTERM | TERM => self.simple_term()?,
            _ => return Err(self.rt.no_match(5).into()),
        };
        self.optional_weights(&mut q)?;
        Ok(q)
    }

    /// `PrefixOperatorQuery()`.
    fn prefix_operator_query(&mut self) -> PResult<SrndQuery> {
        match self.ntk()? {
            OR => {
                let oprt = self.consume(OR)?;
                let queries = self.fields_query_list()?;
                Ok(SrndQuery::or(queries, false, units(&self.image(oprt))))
            }
            AND => {
                let oprt = self.consume(AND)?;
                let queries = self.fields_query_list()?;
                Ok(SrndQuery::and(queries, false, units(&self.image(oprt))))
            }
            N => {
                let oprt = self.consume(N)?;
                let queries = self.fields_query_list()?;
                self.distance_query(queries, false, oprt, false)
            }
            W => {
                let oprt = self.consume(W)?;
                let queries = self.fields_query_list()?;
                self.distance_query(queries, false, oprt, true)
            }
            _ => Err(self.rt.no_match(6).into()),
        }
    }

    /// `FieldsQueryList()`.
    fn fields_query_list(&mut self) -> PResult<Vec<SrndQuery>> {
        self.consume(LPAREN)?;
        let mut queries = vec![self.fields_query()?];
        loop {
            self.consume(COMMA)?;
            queries.push(self.fields_query()?);
            if self.ntk()? != COMMA {
                self.rt.skipped(7);
                break;
            }
        }
        self.consume(RPAREN)?;
        Ok(queries)
    }

    /// `SimpleTerm()`.
    fn simple_term(&mut self) -> PResult<SrndQuery> {
        match self.ntk()? {
            TERM => {
                let term = self.consume(TERM)?;
                Ok(SrndQuery::term(units(&self.image(term)), false))
            }
            QUOTED => {
                let term = self.consume(QUOTED)?;
                let image = self.image(term);
                Ok(SrndQuery::term(units(&image[1..image.len() - 1]), true))
            }
            SUFFIXTERM => {
                // ending in *
                let term = self.consume(SUFFIXTERM)?;
                let image = self.image(term);
                if image.len() - 1 < MINIMUM_PREFIX_LENGTH {
                    return Err(Fail::Parse(format!(
                        "{TRUNCATION_ERROR_MESSAGE}{}",
                        units(&image)
                    )));
                }
                Ok(SrndQuery::prefix(
                    units(&image[..image.len() - 1]),
                    false,
                    TRUNCATOR,
                ))
            }
            TRUNCTERM => {
                // with at least one * or ?
                let term = self.consume(TRUNCTERM)?;
                let image = self.image(term);
                let normal = image
                    .iter()
                    .filter(|&&c| c != u16::from(b'*') && c != u16::from(b'?'))
                    .count();
                let _ = (TRUNCATOR, ANY_CHAR);
                if normal < MINIMUM_CHARS_IN_TRUNC {
                    return Err(Fail::Parse(format!(
                        "{TRUNCATION_ERROR_MESSAGE}{}",
                        units(&image)
                    )));
                }
                SrndQuery::trunc(units(&image)).map_err(Fail::Other)
            }
            TRUNCQUOTED => {
                // eg. "9b-b,m"*
                let term = self.consume(TRUNCQUOTED)?;
                let image = self.image(term);
                if image.len() - 3 < MINIMUM_PREFIX_LENGTH {
                    return Err(Fail::Parse(format!(
                        "{TRUNCATION_ERROR_MESSAGE}{}",
                        units(&image)
                    )));
                }
                Ok(SrndQuery::prefix(
                    units(&image[1..image.len() - 2]),
                    true,
                    TRUNCATOR,
                ))
            }
            _ => Err(self.rt.no_match(8).into()),
        }
    }

    /// `OptionalWeights(q)`.
    fn optional_weights(&mut self, q: &mut SrndQuery) -> PResult<()> {
        loop {
            if self.ntk()? != CARAT {
                self.rt.skipped(9);
                break;
            }
            self.consume(CARAT)?;
            let weight = self.consume(NUMBER)?;
            let image = units(&self.image(weight));
            let f = parse_java_float(&image)
                .map_err(|e| Fail::Parse(format!("{BOOST_ERROR_MESSAGE}{image} ({e})")))?;
            if f <= 0.0 {
                return Err(Fail::Parse(format!("{BOOST_ERROR_MESSAGE}{image}")));
            }
            // left associative, fwiw
            q.set_weight(f * q.weight());
        }
        Ok(())
    }

    /// `getDistanceQuery(queries, infix, dToken, ordered)`.
    fn distance_query(
        &self,
        queries: Vec<SrndQuery>,
        infix: bool,
        token: usize,
        ordered: bool,
    ) -> PResult<SrndQuery> {
        let image = units(&self.image(token));
        // W, 2W, 3W etc -> 1, 2 3, etc. Same for N, 2N ...
        let distance = if image.chars().count() == 1 {
            1
        } else {
            image[..image.len() - 1].parse::<i32>().unwrap_or(1)
        };
        let dq = SrndQuery::distance(queries, infix, distance, image.clone(), ordered);
        if let Some(m) = dq.distance_sub_query_not_allowed() {
            return Err(Fail::Parse(format!("Operator {image}: {m}")));
        }
        Ok(dq)
    }
}
