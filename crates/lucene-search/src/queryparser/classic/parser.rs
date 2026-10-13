//! `QueryParser` (the JavaCC 7 parser generated from `QueryParser.jj`) and
//! the query building of `QueryParserBase`, ported production for
//! production: the lookahead (`jj_2_*`/`jj_3*`), the expected-token
//! bookkeeping (`jj_la1`, `jj_2_rtns`, `jj_add_error_token`) and
//! `ParseException`'s message are Java's, so a syntax error names the same
//! token, column and expected tokens.

use lucene_analysis::payloads::parse_java_float;
use lucene_util::automaton::{
    operations, Automaton, AutomatonProvider, RegExp, TooComplexToDeterminize,
    DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use super::constants::*;
use super::token_manager::{add_escapes, CharStream, Token, TokenManager, TokenMgrError};
use super::{Operator, QueryParser, QueryParserError, MAX_CLAUSE_COUNT};
use crate::extended_query::TermRangeQuery;
use crate::query::{FuzzyQuery, PrefixQuery, RegexpQuery, WildcardQuery};
use crate::query_builder::QueryBuilder;
use crate::query_visitor::Occur;
use crate::queryparser::{term_text, ParsedBoolean, ParsedQuery};

const CONJ_NONE: i32 = 0;
const CONJ_AND: i32 = 1;
const CONJ_OR: i32 = 2;
const MOD_NONE: i32 = 0;
const MOD_NOT: i32 = 10;
const MOD_REQ: i32 = 11;

/// `jj_la1_0`.
static JJ_LA1_0: [u32; 25] = [
    0x300, 0x300, 0x1c00, 0x1c00, 0xfda7c00, 0xfda7f00, 0xfda7f00, 0x120000, 0x40000, 0xfda6000,
    0x9d22000, 0x200000, 0x40000, 0x240000, 0x240000, 0x6000000, 0x90000000, 0x90000000,
    0x60000000, 0x40000, 0x200000, 0x40000, 0x240000, 0x240000, 0xfda2000,
];

/// `jj_la1_1`.
static JJ_LA1_1: [u32; 25] = [
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
];

/// What a production can throw.
pub(crate) enum Fail {
    /// `ParseException` (and `TokenMgrError`, which `parse` wraps alike):
    /// the message.
    Parse(String),
    /// `IndexSearcher.TooManyClauses`.
    TooManyClauses,
    /// Any other exception, which `parse` lets through.
    Runtime(QueryParserError),
}

impl From<TokenMgrError> for Fail {
    fn from(e: TokenMgrError) -> Self {
        Fail::Parse(e.0)
    }
}

type PResult<T> = Result<T, Fail>;

/// How a lookahead scan stops early.
enum Stop {
    /// `LookaheadSuccess`: the lookahead matched.
    Success,
    Lex(TokenMgrError),
}

impl From<TokenMgrError> for Stop {
    fn from(e: TokenMgrError) -> Self {
        Stop::Lex(e)
    }
}

/// `JJCalls`: a lookahead to rescan when the parse fails.
#[derive(Debug, Clone, Default)]
struct JjCall {
    gen: i32,
    first: Option<usize>,
    arg: i32,
}

fn parse_error(msg: impl Into<String>) -> Fail {
    Fail::Parse(msg.into())
}

fn illegal(msg: impl Into<String>) -> Fail {
    Fail::Runtime(QueryParserError::illegal_argument(msg))
}

fn string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// `RegexpQuery.DEFAULT_PROVIDER`: knows no named automaton.
struct NoAutomata;

impl AutomatonProvider for NoAutomata {
    fn get_automaton(&self, _name: &str) -> Result<Option<Automaton>, String> {
        Ok(None)
    }
}

/// One parse of one query string.
pub(crate) struct Parser<'p> {
    cfg: &'p QueryParser,
    tm: TokenManager,
    tokens: Vec<Token>,
    token: usize,
    jj_ntk: i32,
    jj_scanpos: usize,
    jj_lastpos: usize,
    jj_la: i32,
    jj_looking_ahead: bool,
    jj_gen: i32,
    jj_la1: [i32; 25],
    jj_2_rtns: [Vec<JjCall>; 3],
    jj_rescan: bool,
    jj_gc: i32,
    jj_expentries: Vec<Vec<i32>>,
    jj_kind: i32,
    jj_lasttokens: [i32; 100],
    jj_endpos: i32,
}

impl<'p> Parser<'p> {
    pub(crate) fn new(cfg: &'p QueryParser, query: &str) -> Self {
        Parser {
            cfg,
            tm: TokenManager::new(CharStream::new(query)),
            tokens: vec![Token::default()],
            token: 0,
            jj_ntk: -1,
            jj_scanpos: 0,
            jj_lastpos: 0,
            jj_la: 0,
            jj_looking_ahead: false,
            jj_gen: 0,
            jj_la1: [-1; 25],
            jj_2_rtns: [
                vec![JjCall::default()],
                vec![JjCall::default()],
                vec![JjCall::default()],
            ],
            jj_rescan: false,
            jj_gc: 0,
            jj_expentries: Vec::new(),
            jj_kind: -1,
            jj_lasttokens: [0; 100],
            jj_endpos: 0,
        }
    }

    // ------------------------------------------------------- the machinery

    /// The token after `t`, read from the lexer the first time.
    fn next_of(&mut self, t: usize) -> Result<usize, TokenMgrError> {
        if let Some(n) = self.tokens[t].next {
            return Ok(n);
        }
        let tok = self.tm.next_token()?;
        self.tokens.push(tok);
        let n = self.tokens.len() - 1;
        self.tokens[t].next = Some(n);
        Ok(n)
    }

    fn kind(&self, t: usize) -> i32 {
        self.tokens[t].kind
    }

    /// `(jj_ntk==-1)?jj_ntk_f():jj_ntk`.
    fn ntk(&mut self) -> PResult<i32> {
        if self.jj_ntk == -1 {
            let n = self.next_of(self.token)?;
            self.jj_ntk = self.kind(n);
        }
        Ok(self.jj_ntk)
    }

    /// `jj_consume_token(kind)`.
    fn consume(&mut self, kind: i32) -> PResult<usize> {
        let old = self.token;
        self.token = self.next_of(self.token)?;
        self.jj_ntk = -1;
        if self.kind(self.token) == kind {
            self.jj_gen += 1;
            self.jj_gc += 1;
            if self.jj_gc > 100 {
                self.jj_gc = 0;
                let gen = self.jj_gen;
                for list in &mut self.jj_2_rtns {
                    for c in list.iter_mut() {
                        if c.gen < gen {
                            c.first = None;
                        }
                    }
                }
            }
            return Ok(self.token);
        }
        self.token = old;
        self.jj_kind = kind;
        Err(self.generate_parse_exception())
    }

    /// `jj_consume_token(-1); throw new ParseException();` -- the second
    /// is never reached: `-1` is no token's kind.
    fn fail(&mut self) -> Fail {
        self.consume(-1).err().unwrap_or_else(|| parse_error(""))
    }

    /// A choice none of whose alternatives starts with the next token:
    /// `jj_la1[n] = jj_gen;`, then [`Self::fail`].
    fn no_match(&mut self, n: usize) -> Fail {
        self.jj_la1[n] = self.jj_gen;
        self.fail()
    }

    /// `jj_scan_token(kind)`: `Ok(true)` when the token is not `kind`.
    fn scan_token(&mut self, kind: i32) -> Result<bool, Stop> {
        if self.jj_scanpos == self.jj_lastpos {
            self.jj_la -= 1;
            let n = self.next_of(self.jj_scanpos)?;
            self.jj_scanpos = n;
            self.jj_lastpos = n;
        } else {
            self.jj_scanpos = self.next_of(self.jj_scanpos)?;
        }
        if self.jj_rescan {
            let mut i = 0;
            let mut tok = Some(self.token);
            while let Some(t) = tok {
                if t == self.jj_scanpos {
                    break;
                }
                i += 1;
                tok = self.tokens[t].next;
            }
            if tok.is_some() {
                self.add_error_token(kind, i);
            }
        }
        if self.kind(self.jj_scanpos) != kind {
            return Ok(true);
        }
        if self.jj_la == 0 && self.jj_scanpos == self.jj_lastpos {
            return Err(Stop::Success);
        }
        Ok(false)
    }

    /// `getToken(index)`.
    fn get_token(&mut self, index: usize) -> Result<usize, TokenMgrError> {
        let mut t = if self.jj_looking_ahead {
            self.jj_scanpos
        } else {
            self.token
        };
        for _ in 0..index {
            t = self.next_of(t)?;
        }
        Ok(t)
    }

    /// `allowedPostMultiTerm(kind)`.
    fn allowed_post_multi_term(kind: i32) -> bool {
        !matches!(kind, COLON | STAR | FUZZY_SLOP | CARAT | AND | OR)
    }

    /// `getToken(1).kind == TERM && allowedPostMultiTerm(getToken(2).kind)`.
    fn multi_term_continues(&mut self) -> Result<bool, TokenMgrError> {
        let t1 = self.get_token(1)?;
        if self.kind(t1) != TERM {
            return Ok(false);
        }
        let t2 = self.get_token(2)?;
        Ok(Self::allowed_post_multi_term(self.kind(t2)))
    }

    /// `jj_2_<n>(xla)`.
    fn jj_2(&mut self, n: usize, xla: i32) -> PResult<bool> {
        self.jj_la = xla;
        self.jj_scanpos = self.token;
        self.jj_lastpos = self.token;
        let result = match self.jj_3(n) {
            Ok(failed) => Ok(!failed),
            Err(Stop::Success) => Ok(true),
            Err(Stop::Lex(e)) => Err(Fail::from(e)),
        };
        self.jj_save(n, xla);
        result
    }

    /// `jj_3_1()`, `jj_3_2()`, `jj_3_3()`: `Ok(true)` when the lookahead
    /// fails.
    fn jj_3(&mut self, n: usize) -> Result<bool, Stop> {
        match n {
            0 | 1 => self.jj_3r_multi_term(),
            _ => {
                let xsp = self.jj_scanpos;
                if self.jj_3r_clause_field(TERM)? {
                    self.jj_scanpos = xsp;
                    if self.jj_3r_clause_field(STAR)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    /// `jj_3R_MultiTerm_381_3_3()`.
    fn jj_3r_multi_term(&mut self) -> Result<bool, Stop> {
        if self.scan_token(TERM)? {
            return Ok(true);
        }
        self.jj_looking_ahead = true;
        let sem_la = self.multi_term_continues()?;
        self.jj_looking_ahead = false;
        if !sem_la {
            return Ok(true);
        }
        if self.scan_token(TERM)? {
            return Ok(true);
        }
        loop {
            let xsp = self.jj_scanpos;
            if self.scan_token(TERM)? {
                self.jj_scanpos = xsp;
                break;
            }
        }
        Ok(false)
    }

    /// `jj_3R_Clause_305_7_4()` (`TERM ":"`) and `jj_3R_Clause_306_9_5()`
    /// (`"*" ":"`).
    fn jj_3r_clause_field(&mut self, first: i32) -> Result<bool, Stop> {
        if self.scan_token(first)? {
            return Ok(true);
        }
        if self.scan_token(COLON)? {
            return Ok(true);
        }
        Ok(false)
    }

    /// `jj_save(index, xla)`.
    fn jj_save(&mut self, index: usize, xla: i32) {
        let gen = self.jj_gen;
        let list = &mut self.jj_2_rtns[index];
        let mut k = 0;
        while list[k].gen > gen {
            if k + 1 == list.len() {
                list.push(JjCall::default());
                k += 1;
                break;
            }
            k += 1;
        }
        list[k].gen = gen + xla - self.jj_la;
        list[k].first = Some(self.token);
        list[k].arg = xla;
    }

    /// `jj_add_error_token(kind, pos)`.
    fn add_error_token(&mut self, kind: i32, pos: i32) {
        if pos >= 100 {
            return;
        }
        if pos == self.jj_endpos + 1 {
            self.jj_lasttokens[self.jj_endpos as usize] = kind;
            self.jj_endpos += 1;
        } else if self.jj_endpos != 0 {
            let entry: Vec<i32> = self.jj_lasttokens[..self.jj_endpos as usize].to_vec();
            // JavaCC 7 adds the sequence only when an equal one is already
            // listed.
            if self.jj_expentries.contains(&entry) {
                self.jj_expentries.push(entry);
            }
            if pos != 0 {
                self.jj_endpos = pos;
                self.jj_lasttokens[(pos - 1) as usize] = kind;
            }
        }
    }

    /// `jj_rescan_token()`.
    fn rescan_token(&mut self) -> Result<(), TokenMgrError> {
        self.jj_rescan = true;
        for i in 0..3 {
            let mut k = 0;
            while k < self.jj_2_rtns[i].len() {
                let p = self.jj_2_rtns[i][k].clone();
                if p.gen > self.jj_gen {
                    self.jj_la = p.arg;
                    let first = p.first.unwrap_or(0);
                    self.jj_scanpos = first;
                    self.jj_lastpos = first;
                    match self.jj_3(i) {
                        Ok(_) => {}
                        Err(Stop::Success) => break,
                        Err(Stop::Lex(e)) => {
                            self.jj_rescan = false;
                            return Err(e);
                        }
                    }
                }
                k += 1;
            }
        }
        self.jj_rescan = false;
        Ok(())
    }

    /// `generateParseException()`.
    fn generate_parse_exception(&mut self) -> Fail {
        self.jj_expentries.clear();
        let mut la1tokens = [false; 64];
        if self.jj_kind >= 0 {
            la1tokens[self.jj_kind as usize] = true;
            self.jj_kind = -1;
        }
        for i in 0..25 {
            if self.jj_la1[i] == self.jj_gen {
                for j in 0..32 {
                    if JJ_LA1_0[i] & (1 << j) != 0 {
                        la1tokens[j] = true;
                    }
                    if JJ_LA1_1[i] & (1 << j) != 0 {
                        la1tokens[32 + j] = true;
                    }
                }
            }
        }
        for (i, &set) in la1tokens.iter().enumerate().take(33) {
            if set {
                self.jj_expentries.push(vec![i as i32]);
            }
        }
        self.jj_endpos = 0;
        if let Err(e) = self.rescan_token() {
            return Fail::from(e);
        }
        self.add_error_token(0, 0);
        Fail::Parse(self.parse_exception_message())
    }

    /// `ParseException.initialise(token, jj_expentries, tokenImage)`.
    fn parse_exception_message(&self) -> String {
        const EOL: &str = "\n";
        let mut expected = String::new();
        let mut max_size = 0;
        for seq in &self.jj_expentries {
            max_size = max_size.max(seq.len());
            for &k in seq {
                expected.push_str(TOKEN_IMAGE[k as usize]);
                expected.push(' ');
            }
            if seq.last().copied().unwrap_or(0) != 0 {
                expected.push_str("...");
            }
            expected.push_str(EOL);
            expected.push_str("    ");
        }
        let mut retval = String::from("Encountered \"");
        let mut tok = self.tokens[self.token].next;
        for i in 0..max_size {
            if i != 0 {
                retval.push(' ');
            }
            let Some(t) = tok else { break };
            let t = &self.tokens[t];
            if t.kind == 0 {
                retval.push_str(TOKEN_IMAGE[0]);
                break;
            }
            retval.push(' ');
            retval.push_str(TOKEN_IMAGE[t.kind as usize]);
            retval.push_str(" \"");
            retval.push_str(&add_escapes(&t.image));
            retval.push_str(" \"");
            tok = t.next;
        }
        if let Some(next) = self.tokens[self.token].next {
            retval.push_str(&format!(
                "\" at line 1, column {}",
                self.tokens[next].begin_column
            ));
        }
        retval.push('.');
        retval.push_str(EOL);
        match self.jj_expentries.len() {
            0 => {}
            1 => {
                retval.push_str("Was expecting:");
                retval.push_str(EOL);
                retval.push_str("    ");
            }
            _ => {
                retval.push_str("Was expecting one of:");
                retval.push_str(EOL);
                retval.push_str("    ");
            }
        }
        retval.push_str(&expected);
        retval
    }

    // --------------------------------------------------------- productions

    /// `Conjunction()`.
    fn conjunction(&mut self) -> PResult<i32> {
        let mut ret = CONJ_NONE;
        match self.ntk()? {
            AND | OR => match self.ntk()? {
                AND => {
                    self.consume(AND)?;
                    ret = CONJ_AND;
                }
                OR => {
                    self.consume(OR)?;
                    ret = CONJ_OR;
                }
                _ => return Err(self.no_match(0)),
            },
            _ => self.jj_la1[1] = self.jj_gen,
        }
        Ok(ret)
    }

    /// `Modifiers()`.
    fn modifiers(&mut self) -> PResult<i32> {
        let mut ret = MOD_NONE;
        match self.ntk()? {
            NOT | PLUS | MINUS => match self.ntk()? {
                PLUS => {
                    self.consume(PLUS)?;
                    ret = MOD_REQ;
                }
                MINUS => {
                    self.consume(MINUS)?;
                    ret = MOD_NOT;
                }
                NOT => {
                    self.consume(NOT)?;
                    ret = MOD_NOT;
                }
                _ => return Err(self.no_match(2)),
            },
            _ => self.jj_la1[3] = self.jj_gen,
        }
        Ok(ret)
    }

    /// `TopLevelQuery(field)`.
    pub(crate) fn top_level_query(&mut self, field: Option<&str>) -> PResult<Option<ParsedQuery>> {
        let q = self.query(field)?;
        self.consume(EOF)?;
        Ok(q)
    }

    fn starts_clause(kind: i32) -> bool {
        matches!(
            kind,
            NOT | PLUS
                | MINUS
                | BAREOPER
                | LPAREN
                | STAR
                | QUOTED
                | TERM
                | PREFIXTERM
                | WILDTERM
                | REGEXPTERM
                | RANGEIN_START
                | RANGEEX_START
                | NUMBER
        )
    }

    /// `Query(field)`.
    fn query(&mut self, field: Option<&str>) -> PResult<Option<ParsedQuery>> {
        let mut clauses: Vec<(Occur, ParsedQuery)> = Vec::new();
        let mut first_query: Option<ParsedQuery> = None;
        if self.jj_2(0, 2)? {
            first_query = self.multi_term(field, &mut clauses)?;
        } else {
            let k = self.ntk()?;
            if Self::starts_clause(k) {
                let mods = self.modifiers()?;
                let q = self.clause(field)?;
                self.add_clause(&mut clauses, CONJ_NONE, mods, q.clone())?;
                if mods == MOD_NONE {
                    first_query = q;
                }
            } else {
                return Err(self.no_match(4));
            }
        }
        loop {
            let k = self.ntk()?;
            if !(k == AND || k == OR || Self::starts_clause(k)) {
                self.jj_la1[5] = self.jj_gen;
                break;
            }
            if self.jj_2(1, 2)? {
                self.multi_term(field, &mut clauses)?;
            } else {
                let k = self.ntk()?;
                if k == AND || k == OR || Self::starts_clause(k) {
                    let conj = self.conjunction()?;
                    let mods = self.modifiers()?;
                    let q = self.clause(field)?;
                    self.add_clause(&mut clauses, conj, mods, q)?;
                } else {
                    return Err(self.no_match(6));
                }
            }
        }
        if clauses.len() == 1 && first_query.is_some() {
            Ok(first_query)
        } else {
            self.get_boolean_query(clauses)
        }
    }

    /// `Clause(field)`.
    fn clause(&mut self, field: Option<&str>) -> PResult<Option<ParsedQuery>> {
        let mut field: Option<String> = field.map(str::to_string);
        let mut boost = None;
        if self.jj_2(2, 2)? {
            match self.ntk()? {
                TERM => {
                    let f = self.consume(TERM)?;
                    self.consume(COLON)?;
                    field = Some(string(
                        &self.discard_escape_char(&self.tokens[f].image.clone())?,
                    ));
                }
                STAR => {
                    self.consume(STAR)?;
                    self.consume(COLON)?;
                    field = Some("*".to_string());
                }
                _ => return Err(self.no_match(7)),
            }
        }
        let q = match self.ntk()? {
            BAREOPER | STAR | QUOTED | TERM | PREFIXTERM | WILDTERM | REGEXPTERM
            | RANGEIN_START | RANGEEX_START | NUMBER => self.term(field.as_deref())?,
            LPAREN => {
                self.consume(LPAREN)?;
                let q = self.query(field.as_deref())?;
                self.consume(RPAREN)?;
                match self.ntk()? {
                    CARAT => {
                        self.consume(CARAT)?;
                        boost = Some(self.consume(NUMBER)?);
                    }
                    _ => self.jj_la1[8] = self.jj_gen,
                }
                q
            }
            _ => return Err(self.no_match(9)),
        };
        Ok(self.handle_boost(q, boost))
    }

    /// The optional `^boost` and `~slop` after a term or a phrase, in
    /// either order: `(boost, fuzzy_slop)`. `la1` are the four `jj_la1`
    /// slots the productions use (11-14 after a term, 20-23 after a phrase).
    fn boost_and_slop(&mut self, la1: [usize; 4]) -> PResult<(Option<usize>, Option<usize>)> {
        let mut boost = None;
        let mut fuzzy_slop = None;
        match self.ntk()? {
            CARAT | FUZZY_SLOP => match self.ntk()? {
                CARAT => {
                    self.consume(CARAT)?;
                    boost = Some(self.consume(NUMBER)?);
                    match self.ntk()? {
                        FUZZY_SLOP => fuzzy_slop = Some(self.consume(FUZZY_SLOP)?),
                        _ => self.jj_la1[la1[0]] = self.jj_gen,
                    }
                }
                FUZZY_SLOP => {
                    fuzzy_slop = Some(self.consume(FUZZY_SLOP)?);
                    match self.ntk()? {
                        CARAT => {
                            self.consume(CARAT)?;
                            boost = Some(self.consume(NUMBER)?);
                        }
                        _ => self.jj_la1[la1[1]] = self.jj_gen,
                    }
                }
                _ => return Err(self.no_match(la1[2])),
            },
            _ => self.jj_la1[la1[3]] = self.jj_gen,
        }
        Ok((boost, fuzzy_slop))
    }

    /// `Term(field)`.
    fn term(&mut self, field: Option<&str>) -> PResult<Option<ParsedQuery>> {
        let q;
        let boost;
        match self.ntk()? {
            BAREOPER | STAR | TERM | PREFIXTERM | WILDTERM | REGEXPTERM | NUMBER => {
                let (mut prefix, mut wildcard, mut regexp) = (false, false, false);
                let term = match self.ntk()? {
                    TERM => self.consume(TERM)?,
                    STAR => {
                        wildcard = true;
                        self.consume(STAR)?
                    }
                    PREFIXTERM => {
                        prefix = true;
                        self.consume(PREFIXTERM)?
                    }
                    WILDTERM => {
                        wildcard = true;
                        self.consume(WILDTERM)?
                    }
                    REGEXPTERM => {
                        regexp = true;
                        self.consume(REGEXPTERM)?
                    }
                    NUMBER => self.consume(NUMBER)?,
                    BAREOPER => {
                        let t = self.consume(BAREOPER)?;
                        self.tokens[t].image.truncate(1);
                        t
                    }
                    _ => return Err(self.no_match(10)),
                };
                let (b, fuzzy_slop) = self.boost_and_slop([11, 12, 13, 14])?;
                boost = b;
                let image = self.tokens[term].image.clone();
                let slop_image = fuzzy_slop.map(|t| self.tokens[t].image.clone());
                q = self.handle_bare_token_query(
                    field,
                    &image,
                    slop_image.as_deref(),
                    prefix,
                    wildcard,
                    fuzzy_slop.is_some(),
                    regexp,
                )?;
            }
            RANGEIN_START | RANGEEX_START => {
                let mut start_inc = false;
                let mut end_inc = false;
                match self.ntk()? {
                    RANGEIN_START => {
                        self.consume(RANGEIN_START)?;
                        start_inc = true;
                    }
                    RANGEEX_START => {
                        self.consume(RANGEEX_START)?;
                    }
                    _ => return Err(self.no_match(15)),
                }
                let goop1 = self.range_part(16)?;
                self.consume(RANGE_TO)?;
                let goop2 = self.range_part(17)?;
                match self.ntk()? {
                    RANGEIN_END => {
                        self.consume(RANGEIN_END)?;
                        end_inc = true;
                    }
                    RANGEEX_END => {
                        self.consume(RANGEEX_END)?;
                    }
                    _ => return Err(self.no_match(18)),
                }
                boost = match self.ntk()? {
                    CARAT => {
                        self.consume(CARAT)?;
                        Some(self.consume(NUMBER)?)
                    }
                    _ => {
                        self.jj_la1[19] = self.jj_gen;
                        None
                    }
                };
                let bound = |p: &Self, t: usize| -> (Vec<u16>, bool) {
                    let tok = &p.tokens[t];
                    if tok.kind == RANGE_QUOTED {
                        (tok.image[1..tok.image.len() - 1].to_vec(), false)
                    } else {
                        let star = tok.image == [u16::from(b'*')];
                        (tok.image.clone(), star)
                    }
                };
                let (image1, start_open) = bound(self, goop1);
                let (image2, end_open) = bound(self, goop2);
                let part1 = if start_open {
                    None
                } else {
                    Some(string(&self.discard_escape_char(&image1)?))
                };
                let part2 = if end_open {
                    None
                } else {
                    Some(string(&self.discard_escape_char(&image2)?))
                };
                q = self.get_range_query(field, part1, part2, start_inc, end_inc)?;
            }
            QUOTED => {
                let term = self.consume(QUOTED)?;
                let (b, fuzzy_slop) = self.boost_and_slop([20, 21, 22, 23])?;
                boost = b;
                let image = self.tokens[term].image.clone();
                let slop_image = fuzzy_slop.map(|t| self.tokens[t].image.clone());
                q = self.handle_quoted_term(field, &image, slop_image.as_deref())?;
            }
            _ => return Err(self.no_match(24)),
        }
        Ok(self.handle_boost(q, boost))
    }

    /// A range bound: `RANGE_GOOP`, `RANGE_QUOTED` or `"TO"`.
    fn range_part(&mut self, la1: usize) -> PResult<usize> {
        match self.ntk()? {
            RANGE_GOOP => self.consume(RANGE_GOOP),
            RANGE_QUOTED => self.consume(RANGE_QUOTED),
            RANGE_TO => self.consume(RANGE_TO),
            _ => Err(self.no_match(la1)),
        }
    }

    /// `MultiTerm(field, clauses)`.
    fn multi_term(
        &mut self,
        field: Option<&str>,
        clauses: &mut Vec<(Occur, ParsedQuery)>,
    ) -> PResult<Option<ParsedQuery>> {
        let mut first_query = None;
        let text = self.consume(TERM)?;
        let mut text_image = self.tokens[text].image.clone();
        let split = self.cfg.split_on_whitespace;
        if split {
            let t = string(&self.discard_escape_char(&text_image)?);
            first_query = self.get_field_query(field, &t, false)?;
            self.add_clause(clauses, CONJ_NONE, MOD_NONE, first_query.clone())?;
        }
        if !self.multi_term_continues()? {
            return Err(self.fail());
        }
        loop {
            let following = self.consume(TERM)?;
            let following_image = self.tokens[following].image.clone();
            if split {
                let t = string(&self.discard_escape_char(&following_image)?);
                let q = self.get_field_query(field, &t, false)?;
                self.add_clause(clauses, CONJ_NONE, MOD_NONE, q)?;
            } else {
                text_image.push(u16::from(b' '));
                text_image.extend_from_slice(&following_image);
            }
            if !self.multi_term_continues()? {
                break;
            }
        }
        if !split {
            let t = string(&self.discard_escape_char(&text_image)?);
            first_query = self.get_field_query(field, &t, false)?;
            self.add_multi_term_clauses(clauses, first_query.clone());
        }
        Ok(first_query)
    }

    // ---------------------------------------------- QueryParserBase

    /// `addClause(clauses, conj, mods, q)`.
    fn add_clause(
        &self,
        clauses: &mut Vec<(Occur, ParsedQuery)>,
        conj: i32,
        mods: i32,
        q: Option<ParsedQuery>,
    ) -> PResult<()> {
        let and_operator = self.cfg.operator == Operator::And;
        if conj == CONJ_AND {
            if let Some(last) = clauses.last_mut() {
                if last.0 != Occur::MustNot {
                    last.0 = Occur::Must;
                }
            }
        }
        if and_operator && conj == CONJ_OR {
            if let Some(last) = clauses.last_mut() {
                if last.0 != Occur::MustNot {
                    last.0 = Occur::Should;
                }
            }
        }
        let Some(q) = q else { return Ok(()) };
        let prohibited = mods == MOD_NOT;
        let required = if !and_operator {
            mods == MOD_REQ || (conj == CONJ_AND && !prohibited)
        } else {
            !prohibited && conj != CONJ_OR
        };
        let occur = match (required, prohibited) {
            (true, false) => Occur::Must,
            (false, false) => Occur::Should,
            (false, true) => Occur::MustNot,
            (true, true) => {
                return Err(Fail::Runtime(QueryParserError::runtime(
                    "Clause cannot be both required and prohibited",
                )))
            }
        };
        clauses.push((occur, q));
        Ok(())
    }

    /// `addMultiTermClauses(clauses, q)`.
    fn add_multi_term_clauses(
        &self,
        clauses: &mut Vec<(Occur, ParsedQuery)>,
        q: Option<ParsedQuery>,
    ) {
        let Some(q) = q else { return };
        let all_nested_term_queries = match &q {
            ParsedQuery::Boolean(b) => b
                .clauses
                .iter()
                .all(|(_, c)| matches!(c, ParsedQuery::Term(_))),
            _ => false,
        };
        if all_nested_term_queries {
            if let ParsedQuery::Boolean(b) = q {
                clauses.extend(b.clauses);
            }
            return;
        }
        let occur = if self.cfg.operator == Operator::Or {
            Occur::Should
        } else {
            Occur::Must
        };
        match q {
            ParsedQuery::Boolean(b) => {
                clauses.extend(b.clauses.into_iter().map(|(_, c)| (occur, c)));
            }
            other => clauses.push((occur, other)),
        }
    }

    /// `QueryParserBase.getFieldQuery(field, queryText, quoted)`
    /// (`newFieldQuery`).
    fn get_field_query_single(
        &self,
        field: &str,
        text: &str,
        quoted: bool,
    ) -> PResult<Option<ParsedQuery>> {
        let occur = if self.cfg.operator == Operator::And {
            Occur::Must
        } else {
            Occur::Should
        };
        let mut builder = QueryBuilder::new(&self.cfg.analyzer);
        builder.set_enable_position_increments(self.cfg.enable_position_increments);
        builder.set_enable_graph_queries(self.cfg.enable_graph_queries);
        builder.set_auto_generate_multi_term_synonyms_phrase_query(
            self.cfg.auto_generate_multi_term_synonyms_phrase_query,
        );
        let q = builder
            .create_field_query_from_text(
                occur,
                field,
                text,
                quoted || self.cfg.auto_generate_phrase_queries,
                self.cfg.phrase_slop.max(0) as u32,
            )
            .map_err(|e| Fail::Runtime(QueryParserError::from_search(e)))?;
        // A negative slop is refused only by the phrase it would be set on:
        // `new PhraseQuery(...)` and `MultiPhraseQuery.Builder.setSlop`.
        if self.cfg.phrase_slop < 0 {
            match &q {
                Some(crate::query::Clause::Phrase(_)) => {
                    return Err(illegal(format!(
                        "Slop must be >= 0, got {}",
                        self.cfg.phrase_slop
                    )))
                }
                Some(crate::query::Clause::MultiPhrase(_)) => {
                    return Err(illegal("slop value cannot be negative"))
                }
                _ => {}
            }
        }
        Ok(q.map(ParsedQuery::from_clause))
    }

    /// `getRangeQuery(field, part1, part2, startInclusive, endInclusive)`:
    /// with no date resolution set (the default), the date parse's
    /// `DateTools.dateToString` fails and is ignored, so it is
    /// `newRangeQuery`.
    fn get_range_query_single(
        &self,
        field: &str,
        part1: Option<String>,
        part2: Option<String>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> PResult<Option<ParsedQuery>> {
        let start = match part1 {
            None => None,
            Some(p) => Some(self.normalize(field, &p)?),
        };
        let end = match part2 {
            None => None,
            Some(p) => Some(self.normalize(field, &p)?),
        };
        Ok(Some(ParsedQuery::TermRange(TermRangeQuery::new(
            field,
            start,
            end,
            start_inclusive,
            end_inclusive,
        ))))
    }

    /// `getBooleanQuery(clauses)`.
    fn get_boolean_query(
        &self,
        clauses: Vec<(Occur, ParsedQuery)>,
    ) -> PResult<Option<ParsedQuery>> {
        if clauses.is_empty() {
            return Ok(None);
        }
        // `BooleanQuery.Builder.add` refuses the clause past the limit.
        if clauses.len() > MAX_CLAUSE_COUNT {
            return Err(Fail::TooManyClauses);
        }
        Ok(Some(ParsedQuery::Boolean(ParsedBoolean {
            clauses,
            minimum_should_match: 0,
        })))
    }

    /// `getAnalyzer().normalize(field, text)`.
    fn normalize(&self, field: &str, text: &str) -> PResult<Vec<u8>> {
        self.cfg
            .analyzer
            .normalize(field, text)
            .map_err(|e| Fail::Runtime(QueryParserError::from_analysis(e)))
    }

    /// `QueryParserBase.getWildcardQuery(field, termStr)`.
    fn get_wildcard_query_single(&self, field: &str, term: &[u16]) -> PResult<Option<ParsedQuery>> {
        let star = [u16::from(b'*')];
        if field == "*" && term == star {
            return Ok(Some(ParsedQuery::MatchAll));
        }
        let first = term.first().copied();
        if !self.cfg.allow_leading_wildcard
            && (first == Some(u16::from(b'*')) || first == Some(u16::from(b'?')))
        {
            return Err(parse_error(
                "'*' or '?' not allowed as first character in WildcardQuery",
            ));
        }
        let pattern = self.analyze_wildcard(field, term)?;
        // `new WildcardQuery(...)` determinizes its automaton.
        crate::query_visitor::wildcard_automaton_checked(&pattern).map_err(too_complex)?;
        Ok(Some(ParsedQuery::Wildcard(WildcardQuery::new(
            field, pattern,
        ))))
    }

    /// `analyzeWildcard(field, termStr)`: the text between `\.` escapes and
    /// `[?*]+` runs (`WILDCARD_PATTERN`) normalized, the matches kept.
    fn analyze_wildcard(&self, field: &str, term: &[u16]) -> PResult<Vec<u8>> {
        let mut out = Vec::new();
        let mut last = 0;
        let mut i = 0;
        while i < term.len() {
            let Some(end) = wildcard_match_at(term, i) else {
                i += 1;
                continue;
            };
            if i > 0 {
                out.extend(self.normalize(field, &string(&term[last..i]))?);
            }
            out.extend(string(&term[i..end]).into_bytes());
            last = end;
            i = end;
        }
        if last < term.len() {
            out.extend(self.normalize(field, &string(&term[last..]))?);
        }
        Ok(out)
    }

    /// `QueryParserBase.getRegexpQuery(field, termStr)`.
    fn get_regexp_query_single(&self, field: &str, term: &str) -> PResult<Option<ParsedQuery>> {
        let text = term_text(&self.normalize(field, term)?);
        // `new RegexpQuery(...)` parses the expression and determinizes its
        // automaton within `DEFAULT_DETERMINIZE_WORK_LIMIT`.
        let re = RegExp::with_flags(&text, RegExp::ALL, 0).map_err(|e| automaton_error(&e))?;
        let a = re
            .to_automaton_with_provider(&NoAutomata)
            .map_err(|e| automaton_error(&e))?;
        operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).map_err(too_complex)?;
        Ok(Some(ParsedQuery::Regexp(RegexpQuery::new(field, text))))
    }

    /// `QueryParserBase.getPrefixQuery(field, termStr)`.
    fn get_prefix_query_single(&self, field: &str, term: &str) -> PResult<Option<ParsedQuery>> {
        if !self.cfg.allow_leading_wildcard && term.starts_with('*') {
            return Err(parse_error(
                "'*' not allowed as first character in PrefixQuery",
            ));
        }
        let bytes = self.normalize(field, term)?;
        Ok(Some(ParsedQuery::Prefix(PrefixQuery::new(field, bytes))))
    }

    /// `QueryParserBase.getFuzzyQuery(field, termStr, minSimilarity)`
    /// (`newFuzzyQuery`).
    fn get_fuzzy_query_single(
        &self,
        field: &str,
        term: &str,
        min_similarity: f32,
    ) -> PResult<Option<ParsedQuery>> {
        let bytes = self.normalize(field, term)?;
        let text = term_text(&bytes);
        let num_edits = float_to_edits(min_similarity, text.chars().count());
        if self.cfg.fuzzy_prefix_length < 0 {
            return Err(illegal("prefixLength cannot be negative."));
        }
        let q = FuzzyQuery::new(field, bytes)
            .with_max_edits(num_edits)
            .with_prefix_length(self.cfg.fuzzy_prefix_length as usize);
        Ok(Some(ParsedQuery::Fuzzy(q)))
    }

    // ------------------------------------------ MultiFieldQueryParser

    /// `MultiFieldQueryParser.applyBoost(q, field)`: the field's boost, if
    /// it has one (a plain `QueryParser` has none).
    fn apply_boost(&self, q: Option<ParsedQuery>, field: &str) -> Option<ParsedQuery> {
        match (q, self.cfg.boosts.get(field)) {
            (Some(q), Some(&b)) => Some(ParsedQuery::Boost(Box::new(q), b)),
            (q, _) => q,
        }
    }

    /// `MultiFieldQueryParser.getMultiFieldQuery(queries)`.
    fn get_multi_field_query(&self, queries: Vec<ParsedQuery>) -> PResult<Option<ParsedQuery>> {
        self.get_boolean_query(queries.into_iter().map(|q| (Occur::Should, q)).collect())
    }

    /// The query of every field, for a clause with no field of its own.
    fn each_field(
        &self,
        mut make: impl FnMut(&str) -> PResult<Option<ParsedQuery>>,
    ) -> PResult<Option<ParsedQuery>> {
        let mut clauses = Vec::new();
        for f in &self.cfg.fields {
            clauses.extend(make(f)?);
        }
        self.get_multi_field_query(clauses)
    }

    /// `getFieldQuery(field, queryText, quoted)`: across the fields when
    /// there is no field, the k-th word of every field's query in one
    /// boolean (`MultiFieldQueryParser`).
    fn get_field_query(
        &self,
        field: Option<&str>,
        text: &str,
        quoted: bool,
    ) -> PResult<Option<ParsedQuery>> {
        if let Some(f) = field {
            return self.get_field_query_single(f, text, quoted);
        }
        let fields = &self.cfg.fields;
        let mut field_queries: Vec<Option<ParsedQuery>> = Vec::with_capacity(fields.len());
        let mut max_terms = 0;
        for f in fields {
            let q = self.get_field_query_single(f, text, quoted)?;
            if let Some(q) = &q {
                max_terms = match q {
                    ParsedQuery::Boolean(b) => max_terms.max(b.clauses.len()),
                    _ => max_terms.max(1),
                };
            }
            field_queries.push(q);
        }
        let mut clauses = Vec::new();
        for term_num in 0..max_terms {
            let mut term_clauses = Vec::new();
            for (f, fq) in fields.iter().zip(&field_queries) {
                let Some(fq) = fq else { continue };
                let q = match fq {
                    ParsedQuery::Boolean(b) => b.clauses.get(term_num).map(|(_, c)| c.clone()),
                    other if term_num == 0 => Some(other.clone()),
                    _ => None,
                };
                if let Some(q) = self.apply_boost(q, f) {
                    term_clauses.push(q);
                }
            }
            if max_terms > 1 {
                if !term_clauses.is_empty() {
                    if let Some(b) = self.get_multi_field_query(term_clauses)? {
                        clauses.push(b);
                    }
                }
            } else {
                clauses.extend(term_clauses);
            }
        }
        if clauses.is_empty() {
            return Ok(None);
        }
        self.get_multi_field_query(clauses)
    }

    /// `getFieldQuery(field, queryText, slop)`: a phrase at `slop`, across
    /// the fields when there is no field.
    fn get_field_query_slop(
        &self,
        field: Option<&str>,
        text: &str,
        slop: i32,
    ) -> PResult<Option<ParsedQuery>> {
        if let Some(f) = field {
            return match self.get_field_query_single(f, text, true)? {
                Some(q) => Ok(Some(apply_slop(q, slop)?)),
                None => Ok(None),
            };
        }
        let mut clauses = Vec::new();
        for f in &self.cfg.fields {
            if let Some(q) = self.apply_boost(self.get_field_query_single(f, text, true)?, f) {
                clauses.push(apply_slop(q, slop)?);
            }
        }
        if clauses.is_empty() {
            return Ok(None);
        }
        self.get_multi_field_query(clauses)
    }

    /// `getRangeQuery(field, part1, part2, startInclusive, endInclusive)`.
    fn get_range_query(
        &self,
        field: Option<&str>,
        part1: Option<String>,
        part2: Option<String>,
        start_inclusive: bool,
        end_inclusive: bool,
    ) -> PResult<Option<ParsedQuery>> {
        match field {
            Some(f) => {
                let q =
                    self.get_range_query_single(f, part1, part2, start_inclusive, end_inclusive)?;
                Ok(self.apply_boost(q, f))
            }
            None => self.each_field(|f| {
                self.get_range_query(
                    Some(f),
                    part1.clone(),
                    part2.clone(),
                    start_inclusive,
                    end_inclusive,
                )
            }),
        }
    }

    /// `getWildcardQuery(field, termStr)`.
    fn get_wildcard_query(
        &self,
        field: Option<&str>,
        term: &[u16],
    ) -> PResult<Option<ParsedQuery>> {
        match field {
            Some(f) => Ok(self.apply_boost(self.get_wildcard_query_single(f, term)?, f)),
            None => self.each_field(|f| self.get_wildcard_query(Some(f), term)),
        }
    }

    /// `getRegexpQuery(field, termStr)`.
    fn get_regexp_query(&self, field: Option<&str>, term: &str) -> PResult<Option<ParsedQuery>> {
        match field {
            Some(f) => Ok(self.apply_boost(self.get_regexp_query_single(f, term)?, f)),
            None => self.each_field(|f| self.get_regexp_query(Some(f), term)),
        }
    }

    /// `getPrefixQuery(field, termStr)`.
    fn get_prefix_query(&self, field: Option<&str>, term: &str) -> PResult<Option<ParsedQuery>> {
        match field {
            Some(f) => Ok(self.apply_boost(self.get_prefix_query_single(f, term)?, f)),
            None => self.each_field(|f| self.get_prefix_query(Some(f), term)),
        }
    }

    /// `getFuzzyQuery(field, termStr, minSimilarity)`.
    fn get_fuzzy_query(
        &self,
        field: Option<&str>,
        term: &str,
        min_similarity: f32,
    ) -> PResult<Option<ParsedQuery>> {
        match field {
            Some(f) => {
                let q = self.get_fuzzy_query_single(f, term, min_similarity)?;
                Ok(self.apply_boost(q, f))
            }
            None => self.each_field(|f| self.get_fuzzy_query(Some(f), term, min_similarity)),
        }
    }

    /// `handleBareTokenQuery(qfield, term, fuzzySlop, prefix, wildcard,
    /// fuzzy, regexp)`.
    #[allow(clippy::too_many_arguments)]
    fn handle_bare_token_query(
        &self,
        field: Option<&str>,
        image: &[u16],
        fuzzy_slop: Option<&[u16]>,
        prefix: bool,
        wildcard: bool,
        fuzzy: bool,
        regexp: bool,
    ) -> PResult<Option<ParsedQuery>> {
        let term_image = string(&self.discard_escape_char(image)?);
        if wildcard {
            self.get_wildcard_query(field, image)
        } else if prefix {
            let t = string(&self.discard_escape_char(&image[..image.len() - 1])?);
            self.get_prefix_query(field, &t)
        } else if regexp {
            self.get_regexp_query(field, &string(&image[1..image.len() - 1]))
        } else if fuzzy {
            self.handle_bare_fuzzy(field, fuzzy_slop.unwrap_or(&[]), &term_image)
        } else {
            self.get_field_query(field, &term_image, false)
        }
    }

    /// `getFuzzyDistance(fuzzyToken, termStr)`.
    fn get_fuzzy_distance(&self, fuzzy_slop: &[u16]) -> f32 {
        parse_java_float(&string(&fuzzy_slop[1..])).unwrap_or(self.cfg.fuzzy_min_sim)
    }

    /// `handleBareFuzzy(qfield, fuzzySlop, termImage)`.
    fn handle_bare_fuzzy(
        &self,
        field: Option<&str>,
        fuzzy_slop: &[u16],
        term_image: &str,
    ) -> PResult<Option<ParsedQuery>> {
        let fms = self.get_fuzzy_distance(fuzzy_slop);
        if fms < 0.0 {
            return Err(parse_error(
                "Minimum similarity for a FuzzyQuery has to be between 0.0f and 1.0f !",
            ));
        } else if fms >= 1.0 && fms != java_f2i(fms) as f32 {
            return Err(parse_error("Fractional edit distances are not allowed!"));
        }
        self.get_fuzzy_query(field, term_image, fms)
    }

    /// `handleQuotedTerm(qfield, term, fuzzySlop)`.
    fn handle_quoted_term(
        &self,
        field: Option<&str>,
        image: &[u16],
        fuzzy_slop: Option<&[u16]>,
    ) -> PResult<Option<ParsedQuery>> {
        let mut s = self.cfg.phrase_slop;
        if let Some(slop) = fuzzy_slop {
            if let Ok(f) = parse_java_float(&string(&slop[1..])) {
                s = java_f2i(f);
            }
        }
        let text = string(&self.discard_escape_char(&image[1..image.len() - 1])?);
        self.get_field_query_slop(field, &text, s)
    }

    /// `handleBoost(q, boost)`.
    fn handle_boost(&self, q: Option<ParsedQuery>, boost: Option<usize>) -> Option<ParsedQuery> {
        let Some(b) = boost else { return q };
        let f = parse_java_float(&string(&self.tokens[b].image)).unwrap_or(1.0);
        q.map(|q| ParsedQuery::Boost(Box::new(q), f))
    }

    /// `discardEscapeChar(input)`.
    fn discard_escape_char(&self, input: &[u16]) -> PResult<Vec<u16>> {
        let mut output = Vec::with_capacity(input.len());
        let mut last_was_escape = false;
        let mut code_point_multiplier = 0u32;
        let mut code_point = 0u32;
        for &c in input {
            if code_point_multiplier > 0 {
                code_point += hex_to_int(c)? * code_point_multiplier;
                code_point_multiplier >>= 4;
                if code_point_multiplier == 0 {
                    output.push(code_point as u16);
                    code_point = 0;
                }
            } else if last_was_escape {
                if c == u16::from(b'u') {
                    code_point_multiplier = 16 * 16 * 16;
                } else {
                    output.push(c);
                }
                last_was_escape = false;
            } else if c == u16::from(b'\\') {
                last_was_escape = true;
            } else {
                output.push(c);
            }
        }
        if code_point_multiplier > 0 {
            return Err(parse_error("Truncated Unicode escape sequence."));
        }
        if last_was_escape {
            return Err(parse_error("Term can not end with escape character."));
        }
        Ok(output)
    }
}

/// `MultiFieldQueryParser.applySlop(q, slop)` (and `QueryParserBase`'s
/// `addSlopToPhrase`): a phrase rebuilt at `slop`, a multi-phrase too when
/// its slop differs, through a boost.
fn apply_slop(q: ParsedQuery, slop: i32) -> PResult<ParsedQuery> {
    Ok(match q {
        ParsedQuery::Phrase(mut p) => {
            if slop < 0 {
                return Err(illegal(format!("Slop must be >= 0, got {slop}")));
            }
            p.slop = slop as u32;
            ParsedQuery::Phrase(p)
        }
        ParsedQuery::MultiPhrase(mut m) => {
            if slop != m.slop as i32 {
                if slop < 0 {
                    return Err(illegal("slop value cannot be negative"));
                }
                m.slop = slop as u32;
            }
            ParsedQuery::MultiPhrase(m)
        }
        ParsedQuery::Boost(q, b) => ParsedQuery::Boost(Box::new(apply_slop(*q, slop)?), b),
        other => other,
    })
}

/// `hexToInt(c)`.
fn hex_to_int(c: u16) -> PResult<u32> {
    match c {
        0x30..=0x39 => Ok(u32::from(c - 0x30)),
        0x61..=0x66 => Ok(u32::from(c - 0x61 + 10)),
        0x41..=0x46 => Ok(u32::from(c - 0x41 + 10)),
        _ => Err(parse_error(format!(
            "Non-hex character in Unicode escape sequence: {}",
            String::from_utf16_lossy(&[c])
        ))),
    }
}

/// Java's `(int) f`: truncation, `NaN` to 0, saturating.
fn java_f2i(f: f32) -> i32 {
    f as i32
}

/// `FuzzyQuery.floatToEdits(minimumSimilarity, termLen)`.
pub(crate) fn float_to_edits(min_similarity: f32, term_len: usize) -> u8 {
    const MAX: f32 = 2.0;
    if min_similarity >= 1.0 {
        min_similarity.min(MAX) as u8
    } else if min_similarity == 0.0 {
        0
    } else {
        // `(int) ((1D - minimumSimilarity) * termLen)`, at most 2.
        let edits = ((1.0 - f64::from(min_similarity)) * term_len as f64) as i32;
        edits.clamp(0, 2) as u8
    }
}

/// `WILDCARD_PATTERN` (`(\\.)|([?*]+)`) matched at `i`: the match's end.
/// `.` is any code point but a line terminator.
fn wildcard_match_at(term: &[u16], i: usize) -> Option<usize> {
    let c = term[i];
    if c == u16::from(b'\\') {
        let next = *term.get(i + 1)?;
        if matches!(next, 0x0a | 0x0d | 0x85 | 0x2028 | 0x2029) {
            return None;
        }
        let high = (0xd800..0xdc00).contains(&next);
        let low_follows = term
            .get(i + 2)
            .is_some_and(|&l| (0xdc00..0xe000).contains(&l));
        return Some(if high && low_follows { i + 3 } else { i + 2 });
    }
    if c == u16::from(b'?') || c == u16::from(b'*') {
        let mut end = i + 1;
        while end < term.len() && (term[end] == u16::from(b'?') || term[end] == u16::from(b'*')) {
            end += 1;
        }
        return Some(end);
    }
    None
}

/// `TooComplexToDeterminizeException`, which `parse` lets through.
fn too_complex(e: TooComplexToDeterminize) -> Fail {
    Fail::Runtime(QueryParserError::too_complex(e.to_string()))
}

fn automaton_error(e: &lucene_util::automaton::AutomatonError) -> Fail {
    match e {
        lucene_util::automaton::AutomatonError::IllegalArgument(m) => illegal(m.clone()),
        other => Fail::Runtime(QueryParserError::runtime(other.to_string())),
    }
}
