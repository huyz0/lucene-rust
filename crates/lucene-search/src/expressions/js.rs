//! `org.apache.lucene.expressions.js`: the JavaScript-like expression
//! language and its compiler.
//!
//! Java lexes and parses with an ANTLR grammar (`Javascript.g4`) and compiles
//! the parse tree to JVM bytecode with ASM. Here the lexer and parser are
//! written by hand against the same grammar and precedence, and "compiling"
//! lowers the tree to a [`Node`] tree evaluated with the same typing the
//! bytecode has: every sub-expression is produced as the JVM type its parent
//! expects (`double`, `long` for bitwise operators, `int` for logic and shift
//! counts), converted with the JVM's own narrowing rules (`d2i`, `d2l`, `l2i`),
//! so `!0.5` is `1` (`d2i(0.5) == 0`) exactly as Lucene's compiled code has it.
//!
//! # Deviations
//!
//! * A syntax error's [`ParseException::error_offset`] is the offending
//!   token's start, as Java's; its message is not ANTLR's (which lists the
//!   expected token set of its generated DFA) but names the same token and
//!   position.
//! * Java's `picky` mode (ANTLR's ambiguity diagnostics) has no counterpart.
//! * A function is a closure over `f64`s ([`ExprFunction`]) instead of a
//!   `MethodHandle`; "static, takes and returns only `double`" holds by type.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use lucene_util::{sloppy_math, strict_math};

use super::Error as ExprError;
use crate::values_source::BoxDoubleValues;

/// `java.text.ParseException`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseException {
    /// `getMessage()`.
    pub message: String,
    /// `getErrorOffset()`.
    pub error_offset: usize,
}

impl std::fmt::Display for ParseException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseException {}

fn clone_error(e: &ExprError) -> ExprError {
    match e {
        ExprError::Parse(p) => ExprError::Parse(p.clone()),
        ExprError::NumberFormat(m) => ExprError::NumberFormat(m.clone()),
    }
}

fn parse_error(message: String, offset: usize) -> ExprError {
    ExprError::Parse(ParseException {
        message,
        error_offset: offset,
    })
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A function's body: the default functions as plain `fn`s of their
/// arguments (called without collecting them), a custom one as a closure
/// over its argument slice.
#[derive(Clone)]
enum FunctionBody {
    F1(fn(f64) -> f64),
    F2(fn(f64, f64) -> f64),
    F4(fn(f64, f64, f64, f64) -> f64),
    Any(CustomFunction),
}

/// A custom function's body, called with its arguments as a slice.
type CustomFunction = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A function an expression may call: `arity` doubles in, one double out
/// (`JavascriptCompiler`'s `MethodHandle`, which `checkFunction` restricts to
/// exactly that).
#[derive(Clone)]
pub struct ExprFunction {
    arity: usize,
    call: FunctionBody,
}

impl ExprFunction {
    /// A function of `arity` arguments.
    pub fn new(arity: usize, call: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            arity,
            call: FunctionBody::Any(Arc::new(call)),
        }
    }

    /// The number of arguments.
    pub fn arity(&self) -> usize {
        self.arity
    }
}

impl std::fmt::Debug for ExprFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExprFunction/{}", self.arity)
    }
}

/// `Math.max(double, double)`: `NaN` wins, and `0.0` beats `-0.0`.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.is_sign_negative() && a.is_sign_positive() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.is_sign_negative() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
}

/// `Math.min(double, double)`.
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.is_sign_negative() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.is_sign_negative() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `MathUtil.asinh`, over `StrictMath.log` (what `Math.log` is on every CPU
/// with the JVM's intrinsic off).
fn asinh(a: f64) -> f64 {
    let (a, sign) = if (a.to_bits() as i64) < 0 {
        (a.abs(), -1.0)
    } else {
        (a, 1.0)
    };
    sign * strict_math::log((a * a + 1.0).sqrt() + a)
}

/// `MathUtil.acosh`.
fn acosh(a: f64) -> f64 {
    strict_math::log((a * a - 1.0).sqrt() + a)
}

/// `MathUtil.atanh`.
fn atanh(a: f64) -> f64 {
    let (a, mult) = if (a.to_bits() as i64) < 0 {
        (a.abs(), -0.5)
    } else {
        (a, 0.5)
    };
    mult * strict_math::log((1.0 + a) / (1.0 - a))
}

/// `MathUtil.log(base, x)`.
fn logn(base: f64, x: f64) -> f64 {
    strict_math::log(x) / strict_math::log(base)
}

/// `JavascriptCompiler.DEFAULT_FUNCTIONS`
/// (`JavascriptCompiler.properties`), a copy to extend.
pub fn default_functions() -> HashMap<String, ExprFunction> {
    defaults().clone()
}

/// The [`default_functions`], built once (Java's are a static map).
fn defaults() -> &'static HashMap<String, ExprFunction> {
    static DEFAULTS: std::sync::OnceLock<HashMap<String, ExprFunction>> =
        std::sync::OnceLock::new();
    DEFAULTS.get_or_init(build_defaults)
}

fn build_defaults() -> HashMap<String, ExprFunction> {
    let mut m: HashMap<String, ExprFunction> = HashMap::new();
    let mut put1 = |name: &str, f: fn(f64) -> f64| {
        let call = FunctionBody::F1(f);
        m.insert(name.to_string(), ExprFunction { arity: 1, call });
    };
    put1("abs", f64::abs);
    put1("acos", strict_math::acos);
    put1("acosh", acosh);
    put1("asin", strict_math::asin);
    put1("asinh", asinh);
    put1("atan", strict_math::atan);
    put1("atanh", atanh);
    put1("ceil", f64::ceil);
    put1("cos", strict_math::cos);
    put1("cosh", strict_math::cosh);
    put1("exp", strict_math::exp);
    put1("floor", f64::floor);
    put1("ln", strict_math::log);
    put1("log10", strict_math::log10);
    put1("sin", strict_math::sin);
    put1("sinh", strict_math::sinh);
    put1("sqrt", f64::sqrt);
    put1("tan", strict_math::tan);
    put1("tanh", strict_math::tanh);
    let mut put2 = |name: &str, f: fn(f64, f64) -> f64| {
        let call = FunctionBody::F2(f);
        m.insert(name.to_string(), ExprFunction { arity: 2, call });
    };
    put2("atan2", strict_math::atan2);
    put2("logn", logn);
    put2("max", java_max);
    put2("min", java_min);
    put2("pow", strict_math::pow);
    for (name, f) in [
        (
            "haversin",
            haversin_kilometers as fn(f64, f64, f64, f64) -> f64,
        ),
        ("haversinMeters", haversin_meters),
    ] {
        let call = FunctionBody::F4(f);
        m.insert(name.to_string(), ExprFunction { arity: 4, call });
    }
    m
}

/// `SloppyMath.haversinMeters`, with a far cosine that is `StrictMath.cos`
/// as [`haversin_kilometers`]'s is.
fn haversin_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    sloppy_math::haversin_meters_from_sort_key(sloppy_math::haversin_sort_key_fdlibm(
        lat1, lon1, lat2, lon2,
    ))
}

/// `ExpressionMath.haversinKilometers`.
pub fn haversin_kilometers(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversin_meters(lat1, lon1, lat2, lon2) * (1.0 / 1000.0)
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tok {
    Lp,
    Rp,
    Comma,
    BoolNot,
    BwNot,
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lsh,
    Rsh,
    Ush,
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    Ne,
    BwAnd,
    BwXor,
    BwOr,
    BoolAnd,
    BoolOr,
    Cond,
    Colon,
    Variable,
    Octal,
    Hex,
    Decimal,
    Eof,
    /// Stands for the lexical error the lexer hit: ANTLR lexes lazily, so
    /// it surfaces only when the parser reaches it.
    LexError,
}

#[derive(Debug, Clone)]
struct Token {
    kind: Tok,
    text: String,
    start: usize,
    line: usize,
    column: usize,
}

fn is_id_start(c: char) -> bool {
    c == '_' || c == '$' || c.is_ascii_alphabetic()
}

fn is_id_part(c: char) -> bool {
    is_id_start(c) || c.is_ascii_digit()
}

/// The end of `ARRAY` at `i` (`'[' (STRING | INTEGER) ']'`), if there is one.
fn match_array(s: &[char], i: usize) -> Option<usize> {
    if s.get(i) != Some(&'[') {
        return None;
    }
    let j = i + 1;
    let after = match s.get(j) {
        Some(&q) if q == '\'' || q == '"' => {
            // `STRING`: non-greedy, so the first closing quote not escaped
            let mut k = j + 1;
            loop {
                match s.get(k) {
                    None => return None,
                    // `'\\\'' | '\\\\' | ~[\\']`: a backslash only escapes
                    // the quote or another backslash
                    Some('\\') => match s.get(k + 1) {
                        Some(&n) if n == q || n == '\\' => k += 2,
                        _ => return None,
                    },
                    Some(&c) if c == q => break k + 1,
                    Some(_) => k += 1,
                }
            }
        }
        Some('0') => j + 1,
        Some(c) if c.is_ascii_digit() => {
            let mut k = j;
            while s.get(k).is_some_and(char::is_ascii_digit) {
                k += 1;
            }
            k
        }
        _ => return None,
    };
    (s.get(after) == Some(&']')).then_some(after + 1)
}

/// The longest `VARIABLE` at `i`: `ID ARRAY* ('.' ID ARRAY*)*`.
fn match_variable(s: &[char], i: usize) -> usize {
    let mut end;
    let id = |from: usize| -> Option<usize> {
        if s.get(from).is_some_and(|&c| is_id_start(c)) {
            let mut k = from + 1;
            while s.get(k).is_some_and(|&c| is_id_part(c)) {
                k += 1;
            }
            Some(k)
        } else {
            None
        }
    };
    let Some(mut k) = id(i) else { return i };
    loop {
        while let Some(n) = match_array(s, k) {
            k = n;
        }
        end = k;
        if s.get(k) == Some(&'.') {
            if let Some(n) = id(k + 1) {
                k = n;
                continue;
            }
        }
        return end;
    }
}

fn digits(s: &[char], mut i: usize, ok: fn(&char) -> bool) -> usize {
    while s.get(i).is_some_and(ok) {
        i += 1;
    }
    i
}

/// The longest of `OCTAL`, `HEX` and `DECIMAL` at `i` (`0` when none).
fn match_number(s: &[char], i: usize) -> (Tok, usize) {
    let mut best = (Tok::Decimal, i);
    let mut consider = |kind: Tok, end: usize| {
        if end > best.1 {
            best = (kind, end);
        }
    };
    if s.get(i) == Some(&'0') {
        let end = digits(s, i + 1, |c| ('0'..='7').contains(c));
        if end > i + 1 {
            consider(Tok::Octal, end);
        }
        if matches!(s.get(i + 1), Some('x' | 'X')) {
            let end = digits(s, i + 2, char::is_ascii_hexdigit);
            if end > i + 2 {
                consider(Tok::Hex, end);
            }
        }
    }
    // DECIMAL: ( INTEGER ( '.' [0-9]* )? | '.' [0-9]+ ) ( [eE] [+-]? [0-9]+ )?
    let mantissa = if s.get(i) == Some(&'.') {
        let end = digits(s, i + 1, char::is_ascii_digit);
        (end > i + 1).then_some(end)
    } else if s.get(i) == Some(&'0') {
        Some(i + 1)
    } else if s.get(i).is_some_and(|c| ('1'..='9').contains(c)) {
        Some(digits(s, i + 1, char::is_ascii_digit))
    } else {
        None
    };
    if let Some(mut end) = mantissa {
        if s.get(i) != Some(&'.') && s.get(end) == Some(&'.') {
            end = digits(s, end + 1, char::is_ascii_digit);
        }
        if matches!(s.get(end), Some('e' | 'E')) {
            let mut k = end + 1;
            if matches!(s.get(k), Some('+' | '-')) {
                k += 1;
            }
            let e = digits(s, k, char::is_ascii_digit);
            if e > k {
                end = e;
            }
        }
        consider(Tok::Decimal, end);
    }
    best
}

fn lex(source: &str) -> (Vec<Token>, Option<ExprError>) {
    let s: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut line, mut line_start) = (0usize, 1usize, 0usize);
    while i < s.len() {
        let c = s[i];
        if matches!(c, ' ' | '\t' | '\n' | '\r') {
            if c == '\n' {
                line += 1;
                line_start = i + 1;
            }
            i += 1;
            continue;
        }
        let (kind, end) = if is_id_start(c) {
            (Tok::Variable, match_variable(&s, i))
        } else if c.is_ascii_digit() || c == '.' {
            let (k, e) = match_number(&s, i);
            if e == i {
                return lex_failed(out, unexpected_char(&s, i, line, i - line_start), i);
            }
            (k, e)
        } else {
            let two: String = s[i..(i + 3).min(s.len())].iter().collect();
            if two == ">>>" {
                (Tok::Ush, i + 3)
            } else if two.starts_with("<<") {
                (Tok::Lsh, i + 2)
            } else if two.starts_with(">>") {
                (Tok::Rsh, i + 2)
            } else if two.starts_with("<=") {
                (Tok::Lte, i + 2)
            } else if two.starts_with(">=") {
                (Tok::Gte, i + 2)
            } else if two.starts_with("==") {
                (Tok::Eq, i + 2)
            } else if two.starts_with("!=") {
                (Tok::Ne, i + 2)
            } else if two.starts_with("&&") {
                (Tok::BoolAnd, i + 2)
            } else if two.starts_with("||") {
                (Tok::BoolOr, i + 2)
            } else {
                let k = match c {
                    '(' => Tok::Lp,
                    ')' => Tok::Rp,
                    ',' => Tok::Comma,
                    '!' => Tok::BoolNot,
                    '~' => Tok::BwNot,
                    '*' => Tok::Mul,
                    '/' => Tok::Div,
                    '%' => Tok::Rem,
                    '+' => Tok::Add,
                    '-' => Tok::Sub,
                    '<' => Tok::Lt,
                    '>' => Tok::Gt,
                    '&' => Tok::BwAnd,
                    '^' => Tok::BwXor,
                    '|' => Tok::BwOr,
                    '?' => Tok::Cond,
                    ':' => Tok::Colon,
                    _ => return lex_failed(out, unexpected_char(&s, i, line, i - line_start), i),
                };
                (k, i + 1)
            }
        };
        out.push(Token {
            kind,
            text: s[i..end].iter().collect(),
            start: i,
            line,
            column: i - line_start,
        });
        i = end;
    }
    out.push(Token {
        kind: Tok::Eof,
        text: "<EOF>".to_string(),
        start: s.len(),
        line,
        column: s.len() - line_start,
    });
    (out, None)
}

fn lex_failed(mut out: Vec<Token>, error: ExprError, at: usize) -> (Vec<Token>, Option<ExprError>) {
    out.push(Token {
        kind: Tok::LexError,
        text: String::new(),
        start: at,
        line: 0,
        column: 0,
    });
    (out, Some(error))
}

/// `JavascriptErrorHandlingLexer.recover`: the text from the token's start
/// through the character the lexer's DFA died on. Only `.` (a decimal
/// without its digits) and `=` (half of `==`) are viable for one character
/// without being a token; every other failure dies on its first character.
fn unexpected_char(s: &[char], at: usize, line: usize, column: usize) -> ExprError {
    let end = if matches!(s[at], '.' | '=') {
        (at + 2).min(s.len())
    } else {
        at + 1
    };
    // `Lexer.getErrorDisplay`: only line breaks and tabs are escaped
    let mut text = String::new();
    for &c in &s[at..end] {
        match c {
            '\n' => text.push_str("\\n"),
            '\t' => text.push_str("\\t"),
            '\r' => text.push_str("\\r"),
            c => text.push(c),
        }
    }
    parse_error(
        format!("unexpected character '{text}' on line ({line}) position ({column})"),
        at,
    )
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bin {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lsh,
    Rsh,
    Ush,
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    Ne,
    BwAnd,
    BwXor,
    BwOr,
    And,
    Or,
}

fn binary(kind: Tok) -> Option<(Bin, u8)> {
    Some(match kind {
        Tok::Mul => (Bin::Mul, 12),
        Tok::Div => (Bin::Div, 12),
        Tok::Rem => (Bin::Rem, 12),
        Tok::Add => (Bin::Add, 11),
        Tok::Sub => (Bin::Sub, 11),
        Tok::Lsh => (Bin::Lsh, 10),
        Tok::Rsh => (Bin::Rsh, 10),
        Tok::Ush => (Bin::Ush, 10),
        Tok::Lt => (Bin::Lt, 9),
        Tok::Lte => (Bin::Lte, 9),
        Tok::Gt => (Bin::Gt, 9),
        Tok::Gte => (Bin::Gte, 9),
        Tok::Eq => (Bin::Eq, 8),
        Tok::Ne => (Bin::Ne, 8),
        Tok::BwAnd => (Bin::BwAnd, 7),
        Tok::BwXor => (Bin::BwXor, 6),
        Tok::BwOr => (Bin::BwOr, 5),
        Tok::BoolAnd => (Bin::And, 4),
        Tok::BoolOr => (Bin::Or, 3),
        _ => return None,
    })
}

/// The conditional's precedence: below every binary operator, right
/// associative.
const COND_PREC: u8 = 2;

/// The deepest expression tree compiled. Parsing, lowering and evaluation
/// all recurse over the tree, and a Rust stack overflow aborts the process
/// (Java's ANTLR parser and generated bytecode would throw a
/// `StackOverflowError` somewhere past a few thousand levels instead). A
/// deeper expression is a `ParseException` at the token that crosses it.
/// At 256 the deepest tree compiles and evaluates in under 256 KiB of stack
/// in an optimised build, a quarter of a JVM thread's default 1 MiB (the
/// FFI's calls run on JVM threads); `nesting_is_bounded` runs it there.
/// A run of a left-associative operator (`a + b + c ...`) longer than
/// [`CHAIN`] counts one level whatever its length, so a flat sum of
/// thousands of terms compiles as in Java.
pub const MAX_DEPTH: u32 = 256;

/// The longest run of operators of one precedence level kept as nested
/// binary nodes; a longer one is a chain, evaluated by a loop.
const CHAIN: usize = 16;

#[derive(Debug, Clone)]
enum Ast {
    Paren(Box<Ast>),
    /// More than [`CHAIN`] left-associative operators of one level: the
    /// first operand, then each operator with its right operand.
    Chain(Box<Ast>, Vec<(Bin, Ast)>),
    Number(Tok, String),
    External {
        name: String,
        text: String,
        args: Vec<Ast>,
        parens: bool,
        start: usize,
    },
    Unary(Tok, Box<Ast>),
    Binary(Bin, Box<Ast>, Box<Ast>),
    Conditional(Box<Ast>, Box<Ast>, Box<Ast>),
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    lex_error: Option<ExprError>,
    /// `prefix` frames currently open.
    nesting: u32,
}

impl Parser {
    /// The current token; a lexical error is raised when the parser reaches
    /// it (ANTLR's token stream lexes on demand).
    fn peek(&self) -> Result<&Token, ExprError> {
        let t = &self.tokens[self.pos];
        match (&t.kind, &self.lex_error) {
            (Tok::LexError, Some(e)) => Err(clone_error(e)),
            _ => Ok(t),
        }
    }

    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if t.kind != Tok::Eof {
            self.pos += 1;
        }
        t
    }

    /// `JavascriptParserErrorStrategy.recoverInline`.
    fn mismatch(&self, expecting: &str) -> ExprError {
        let t = &self.tokens[self.pos];
        parse_error(
            format!(
                "unexpected token '{}' on line ({}) position ({}) was expecting {expecting}",
                t.text, t.line, t.column
            ),
            t.start,
        )
    }

    /// `JavascriptParserErrorStrategy.recover` for a `NoViableAltException`.
    fn no_viable(&self) -> ExprError {
        let t = &self.tokens[self.pos];
        let message = if t.kind == Tok::Eof {
            "unexpected end of expression".to_string()
        } else {
            format!(
                "invalid sequence of tokens near '{}' on line ({}) position ({})",
                t.text, t.line, t.column
            )
        };
        parse_error(message, t.start)
    }

    fn expect(&mut self, kind: Tok, what: &str) -> Result<Token, ExprError> {
        if self.peek()?.kind == kind {
            Ok(self.next())
        } else {
            Err(self.mismatch(what))
        }
    }

    fn compile(&mut self) -> Result<Ast, ExprError> {
        let (e, _) = self.expression(0)?;
        if self.peek()?.kind != Tok::Eof {
            return Err(self.mismatch("<EOF>"));
        }
        Ok(e)
    }

    /// A node `depth` levels deep, or the depth error at the current token.
    fn deepen(&self, depth: u32) -> Result<u32, ExprError> {
        if depth > MAX_DEPTH {
            let t = &self.tokens[self.pos];
            return Err(parse_error(
                format!("expression nested deeper than {MAX_DEPTH} levels"),
                t.start,
            ));
        }
        Ok(depth)
    }

    /// An expression and its tree's depth.
    fn expression(&mut self, min_prec: u8) -> Result<(Ast, u32), ExprError> {
        self.nesting += 1;
        let r = self
            .deepen(self.nesting)
            .and_then(|_| self.expression_inner(min_prec));
        self.nesting -= 1;
        r
    }

    fn expression_inner(&mut self, min_prec: u8) -> Result<(Ast, u32), ExprError> {
        let (mut lhs, mut depth) = self.prefix()?;
        // the operators of one precedence level that follow `lhs`, each with
        // its right operand and that operand's depth
        let mut run: Vec<(Bin, Ast, u32)> = Vec::new();
        let mut run_prec = 0;
        loop {
            let kind = self.peek()?.kind;
            if let Some((op, prec)) = binary(kind) {
                if prec >= min_prec {
                    if !run.is_empty() && prec != run_prec {
                        (lhs, depth) = self.chain(lhs, depth, std::mem::take(&mut run))?;
                    }
                    self.next();
                    let (rhs, d) = self.expression(prec + 1)?;
                    run.push((op, rhs, d));
                    run_prec = prec;
                    continue;
                }
            }
            (lhs, depth) = self.chain(lhs, depth, std::mem::take(&mut run))?;
            if kind == Tok::Cond && COND_PREC >= min_prec {
                self.next();
                let (then, d1) = self.expression(0)?;
                self.expect(Tok::Colon, "':'")?;
                let (other, d2) = self.expression(COND_PREC)?;
                depth = self.deepen(depth.max(d1).max(d2) + 1)?;
                lhs = Ast::Conditional(Box::new(lhs), Box::new(then), Box::new(other));
            } else {
                return Ok((lhs, depth));
            }
        }
    }

    /// `lhs` followed by a run of left-associative operators of one
    /// precedence level. A short run is the nested binary nodes Java's parse
    /// tree has; a longer one is a single [`Ast::Chain`], one level deep
    /// however many operands it has, evaluated by a loop.
    fn chain(
        &self,
        lhs: Ast,
        depth: u32,
        run: Vec<(Bin, Ast, u32)>,
    ) -> Result<(Ast, u32), ExprError> {
        if run.len() > CHAIN {
            let deepest = run.iter().fold(depth, |m, (_, _, d)| m.max(*d));
            let steps = run.into_iter().map(|(op, a, _)| (op, a)).collect();
            return Ok((Ast::Chain(Box::new(lhs), steps), self.deepen(deepest + 1)?));
        }
        let (mut lhs, mut depth) = (lhs, depth);
        for (op, rhs, d) in run {
            depth = self.deepen(depth.max(d) + 1)?;
            lhs = Ast::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok((lhs, depth))
    }

    /// A prefix expression and its tree's depth. `expression` and `prefix`
    /// count their open frames in `self.nesting` before descending, so the
    /// parser's own stack stays within [`MAX_DEPTH`] frames too.
    fn prefix(&mut self) -> Result<(Ast, u32), ExprError> {
        self.nesting += 1;
        let r = self.deepen(self.nesting).and_then(|_| self.prefix_inner());
        self.nesting -= 1;
        r
    }

    fn prefix_inner(&mut self) -> Result<(Ast, u32), ExprError> {
        let t = self.peek()?.clone();
        match t.kind {
            Tok::BoolNot | Tok::BwNot | Tok::Add | Tok::Sub => {
                self.next();
                // the unary alternative outranks every binary one
                let (operand, d) = self.prefix()?;
                Ok((Ast::Unary(t.kind, Box::new(operand)), self.deepen(d + 1)?))
            }
            Tok::Lp => {
                self.next();
                let (e, d) = self.expression(0)?;
                self.expect(Tok::Rp, "')'")?;
                Ok((Ast::Paren(Box::new(e)), self.deepen(d + 1)?))
            }
            Tok::Octal | Tok::Hex | Tok::Decimal => {
                self.next();
                Ok((Ast::Number(t.kind, t.text), 1))
            }
            Tok::Variable => {
                self.next();
                let mut text = t.text.clone();
                let mut args = Vec::new();
                let mut parens = false;
                let mut depth = 1;
                if self.peek()?.kind == Tok::Lp {
                    self.next();
                    parens = true;
                    text.push('(');
                    if self.peek()?.kind != Tok::Rp {
                        loop {
                            let (a, d) = self.expression(0)?;
                            depth = self.deepen(depth.max(d + 1))?;
                            text.push_str(&ast_text(&a));
                            args.push(a);
                            if self.peek()?.kind == Tok::Comma {
                                self.next();
                                text.push(',');
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Tok::Rp, "')'")?;
                    text.push(')');
                }
                Ok((
                    Ast::External {
                        name: t.text,
                        text,
                        args,
                        parens,
                        start: t.start,
                    },
                    depth,
                ))
            }
            _ => Err(self.no_viable()),
        }
    }
}

/// `ParseTree.getText()`: the tokens, without the whitespace between them.
fn ast_text(a: &Ast) -> String {
    match a {
        Ast::Paren(e) => format!("({})", ast_text(e)),
        Ast::Number(_, t) => t.clone(),
        Ast::External { text, .. } => text.clone(),
        Ast::Unary(op, e) => format!("{}{}", tok_text(*op), ast_text(e)),
        Ast::Binary(op, l, r) => format!("{}{}{}", ast_text(l), bin_text(*op), ast_text(r)),
        Ast::Chain(first, steps) => {
            let mut text = ast_text(first);
            for (op, a) in steps {
                text.push_str(bin_text(*op));
                text.push_str(&ast_text(a));
            }
            text
        }
        Ast::Conditional(c, a, b) => {
            format!("{}?{}:{}", ast_text(c), ast_text(a), ast_text(b))
        }
    }
}

fn tok_text(t: Tok) -> &'static str {
    match t {
        Tok::BoolNot => "!",
        Tok::BwNot => "~",
        Tok::Add => "+",
        _ => "-",
    }
}

fn bin_text(b: Bin) -> &'static str {
    match b {
        Bin::Mul => "*",
        Bin::Div => "/",
        Bin::Rem => "%",
        Bin::Add => "+",
        Bin::Sub => "-",
        Bin::Lsh => "<<",
        Bin::Rsh => ">>",
        Bin::Ush => ">>>",
        Bin::Lt => "<",
        Bin::Lte => "<=",
        Bin::Gt => ">",
        Bin::Gte => ">=",
        Bin::Eq => "==",
        Bin::Ne => "!=",
        Bin::BwAnd => "&",
        Bin::BwXor => "^",
        Bin::BwOr => "|",
        Bin::And => "&&",
        Bin::Or => "||",
    }
}

// ---------------------------------------------------------------------------
// Lowering ("compiling")
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Node {
    /// `pushLong` of a hexadecimal or octal literal.
    Integer(i64),
    Decimal(f64),
    Variable(usize),
    Call(ExprFunction, Vec<Node>),
    Not(Box<Node>),
    BwNot(Box<Node>),
    Neg(Box<Node>),
    /// Unary plus: its operand, in the type the parent expects.
    Plus(Box<Node>),
    Arith(Bin, Box<Node>, Box<Node>),
    Shift(Bin, Box<Node>, Box<Node>),
    Compare(Bin, Box<Node>, Box<Node>),
    Bitwise(Bin, Box<Node>, Box<Node>),
    And(Box<Node>, Box<Node>),
    Or(Box<Node>, Box<Node>),
    Conditional(Box<Node>, Box<Node>, Box<Node>),
    /// [`Ast::Chain`]: `((first op1 a1) op2 a2) ...`, each step typed as its
    /// binary node would be.
    Chain(Box<Node>, Vec<(Bin, Node)>),
}

/// What a chain's operators compute, and in which JVM type.
#[derive(Clone, Copy, PartialEq)]
enum ChainKind {
    /// `double`
    Arith,
    /// `long`, the counts `int`
    Shift,
    /// `double` operands, an `int` result
    Compare,
    /// `long`
    Bitwise,
    /// `int`, short-circuiting
    And,
    Or,
}

fn chain_kind(steps: &[(Bin, Node)]) -> ChainKind {
    match steps[0].0 {
        Bin::Mul | Bin::Div | Bin::Rem | Bin::Add | Bin::Sub => ChainKind::Arith,
        Bin::Lsh | Bin::Rsh | Bin::Ush => ChainKind::Shift,
        Bin::Lt | Bin::Lte | Bin::Gt | Bin::Gte | Bin::Eq | Bin::Ne => ChainKind::Compare,
        Bin::BwAnd | Bin::BwXor | Bin::BwOr => ChainKind::Bitwise,
        Bin::And => ChainKind::And,
        Bin::Or => ChainKind::Or,
    }
}

/// An arithmetic chain, as a `double`.
fn arith_chain(first: &Node, steps: &[(Bin, Node)]) -> Compiled<f64> {
    let first = to_double(first);
    let steps: Vec<(Bin, Compiled<f64>)> =
        steps.iter().map(|(op, a)| (*op, to_double(a))).collect();
    compiled(move |r| {
        let mut v = first(r);
        for (op, a) in &steps {
            let b = a(r);
            v = match op {
                Bin::Mul => v * b,
                Bin::Div => v / b,
                Bin::Rem => v % b,
                Bin::Add => v + b,
                _ => v - b,
            };
        }
        v
    })
}

/// A shift or bitwise chain, as a `long`.
fn long_chain(first: &Node, steps: &[(Bin, Node)]) -> Compiled<i64> {
    let first = to_long(first);
    if chain_kind(steps) == ChainKind::Shift {
        let steps: Vec<(Bin, Compiled<i32>)> =
            steps.iter().map(|(op, a)| (*op, to_int(a))).collect();
        return compiled(move |r| {
            let mut v = first(r);
            for (op, n) in &steps {
                // Java masks a long shift's count to its low six bits
                let n = (n(r) & 63) as u32;
                v = match op {
                    Bin::Lsh => v.wrapping_shl(n),
                    Bin::Rsh => v >> n,
                    _ => ((v as u64) >> n) as i64,
                };
            }
            v
        });
    }
    let steps: Vec<(Bin, Compiled<i64>)> = steps.iter().map(|(op, a)| (*op, to_long(a))).collect();
    compiled(move |r| {
        let mut v = first(r);
        for (op, a) in &steps {
            let b = a(r);
            v = match op {
                Bin::BwAnd => v & b,
                Bin::BwXor => v ^ b,
                _ => v | b,
            };
        }
        v
    })
}

/// A comparison or logical chain, as an `int`.
fn int_chain(first: &Node, steps: &[(Bin, Node)]) -> Compiled<i32> {
    match chain_kind(steps) {
        ChainKind::Compare => {
            let first = to_double(first);
            let steps: Vec<(Bin, Compiled<f64>)> =
                steps.iter().map(|(op, a)| (*op, to_double(a))).collect();
            compiled(move |r| {
                // each comparison's `int` the next one's left `double`
                let mut v = first(r);
                let mut result = 0;
                for (op, a) in &steps {
                    let b = a(r);
                    result = i32::from(match op {
                        Bin::Lt => v < b,
                        Bin::Lte => v <= b,
                        Bin::Gt => v > b,
                        Bin::Gte => v >= b,
                        Bin::Eq => v == b,
                        _ => v != b,
                    });
                    v = f64::from(result);
                }
                result
            })
        }
        kind => {
            // `((a && b) && c) ...` is `0` at its first `0` operand, the rest
            // unevaluated; `||` is `1` at its first non-zero one
            let stop_at_zero = kind == ChainKind::And;
            let operands: Vec<Compiled<i32>> = std::iter::once(first)
                .chain(steps.iter().map(|(_, a)| a))
                .map(to_int)
                .collect();
            compiled(move |r| {
                for a in &operands {
                    if (a(r) == 0) == stop_at_zero {
                        return i32::from(!stop_at_zero);
                    }
                }
                i32::from(stop_at_zero)
            })
        }
    }
}

// The tree evaluated: `Node` compiled once more, into closures, each
// sub-expression producing the JVM type its parent expects
// (`GeneratorAdapter.cast` and `pushBoolean` resolved when compiling rather
// than per evaluation; Rust's `as` is the JVM's `d2i`/`d2l` -- saturating,
// `NaN` to `0` -- and `l2i`, truncating). An operand that is a variable or a
// constant is read inside its parent's closure rather than through a closure
// of its own, and an operation over two constants is folded.

/// A compiled sub-expression producing a `T`.
type Compiled<T> = Arc<dyn Fn(&mut Reads<'_, '_>) -> T + Send + Sync>;

/// `f` as a [`Compiled`] (the bound gives the closure its signature).
fn compiled<T>(f: impl Fn(&mut Reads<'_, '_>) -> T + Send + Sync + 'static) -> Compiled<T> {
    Arc::new(f)
}

/// A constant as a [`Compiled`].
fn constant<T: Copy + Send + Sync + 'static>(v: T) -> Compiled<T> {
    compiled(move |_| v)
}

/// An operand read in place.
#[derive(Clone, Copy)]
enum Leaf {
    Const(f64),
    Var(usize),
}

/// `node` as an operand read in place, as a `double`.
fn leaf(node: &Node) -> Option<Leaf> {
    match node {
        Node::Integer(i) => Some(Leaf::Const(*i as f64)),
        Node::Decimal(d) => Some(Leaf::Const(*d)),
        Node::Variable(i) => Some(Leaf::Var(*i)),
        Node::Plus(e) => leaf(e),
        _ => None,
    }
}

/// `op` over two `double` operands, the left evaluated first.
fn binary_op<T: Copy + Send + Sync + 'static>(
    lhs: &Node,
    rhs: &Node,
    op: impl Fn(f64, f64) -> T + Copy + Send + Sync + 'static,
) -> Compiled<T> {
    match (leaf(lhs), leaf(rhs)) {
        (Some(Leaf::Const(a)), Some(Leaf::Const(b))) => constant(op(a, b)),
        (Some(Leaf::Var(i)), Some(Leaf::Const(b))) => compiled(move |r| op(r.read(i), b)),
        (Some(Leaf::Const(a)), Some(Leaf::Var(j))) => compiled(move |r| op(a, r.read(j))),
        (Some(Leaf::Var(i)), Some(Leaf::Var(j))) => compiled(move |r| {
            let a = r.read(i);
            op(a, r.read(j))
        }),
        (None, Some(Leaf::Const(b))) => {
            let f = to_double(lhs);
            compiled(move |r| op(f(r), b))
        }
        (None, Some(Leaf::Var(j))) => {
            let f = to_double(lhs);
            compiled(move |r| {
                let a = f(r);
                op(a, r.read(j))
            })
        }
        (Some(Leaf::Const(a)), None) => {
            let g = to_double(rhs);
            compiled(move |r| op(a, g(r)))
        }
        (Some(Leaf::Var(i)), None) => {
            let g = to_double(rhs);
            compiled(move |r| {
                let a = r.read(i);
                op(a, g(r))
            })
        }
        (None, None) => {
            let (f, g) = (to_double(lhs), to_double(rhs));
            compiled(move |r| {
                let a = f(r);
                op(a, g(r))
            })
        }
    }
}

/// `node` producing a `double`.
fn to_double(node: &Node) -> Compiled<f64> {
    if let Some(leaf) = leaf(node) {
        return match leaf {
            Leaf::Const(c) => constant(c),
            Leaf::Var(i) => compiled(move |r| r.read(i)),
        };
    }
    match node {
        Node::Call(f, args) => call(&f.call, args),
        Node::Neg(e) => match leaf(e) {
            Some(Leaf::Const(c)) => constant(-c),
            Some(Leaf::Var(i)) => compiled(move |r| -r.read(i)),
            None => {
                let f = to_double(e);
                compiled(move |r| -f(r))
            }
        },
        Node::Arith(op, l, r) => match op {
            Bin::Mul => binary_op(l, r, |a, b| a * b),
            Bin::Div => binary_op(l, r, |a, b| a / b),
            Bin::Rem => binary_op(l, r, |a, b| a % b),
            Bin::Add => binary_op(l, r, |a, b| a + b),
            _ => binary_op(l, r, |a, b| a - b),
        },
        Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
            let f = to_long(node);
            compiled(move |r| f(r) as f64)
        }
        Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
            let f = to_int(node);
            compiled(move |r| f64::from(f(r)))
        }
        Node::Conditional(c, a, e) => {
            let (c, a, e) = (to_int(c), to_double(a), to_double(e));
            compiled(move |r| if c(r) != 0 { a(r) } else { e(r) })
        }
        Node::Plus(e) => to_double(e),
        Node::Chain(first, steps) => match chain_kind(steps) {
            ChainKind::Arith => arith_chain(first, steps),
            ChainKind::Shift | ChainKind::Bitwise => {
                let f = long_chain(first, steps);
                compiled(move |r| f(r) as f64)
            }
            _ => {
                let f = int_chain(first, steps);
                compiled(move |r| f64::from(f(r)))
            }
        },
        // the leaves, above
        Node::Integer(_) | Node::Decimal(_) | Node::Variable(_) => unreachable!("a leaf"),
    }
}

/// A call of `body` over `args`, each produced as a `double`.
fn call(body: &FunctionBody, args: &[Node]) -> Compiled<f64> {
    match body {
        FunctionBody::F1(f) => {
            let f = *f;
            match leaf(&args[0]) {
                Some(Leaf::Var(i)) => compiled(move |r| f(r.read(i))),
                _ => {
                    let a = to_double(&args[0]);
                    compiled(move |r| f(a(r)))
                }
            }
        }
        FunctionBody::F2(f) => {
            let f = *f;
            binary_op(&args[0], &args[1], f)
        }
        FunctionBody::F4(f) => {
            let f = *f;
            let [a, b, c, d] = [0, 1, 2, 3].map(|i| to_double(&args[i]));
            compiled(move |r| {
                let (a, b, c) = (a(r), b(r), c(r));
                f(a, b, c, d(r))
            })
        }
        FunctionBody::Any(f) => {
            let f = f.clone();
            let args: Vec<Compiled<f64>> = args.iter().map(to_double).collect();
            compiled(move |r| {
                // the arguments on the stack for the common arities
                let mut small = [0.0; 4];
                if args.len() <= small.len() {
                    for (slot, a) in small.iter_mut().zip(&args) {
                        *slot = a(r);
                    }
                    f(&small[..args.len()])
                } else {
                    let all: Vec<f64> = args.iter().map(|a| a(r)).collect();
                    f(&all)
                }
            })
        }
    }
}

/// `node` producing a `long`.
fn to_long(node: &Node) -> Compiled<i64> {
    let two = |l: &Node, r: &Node| (to_long(l), to_long(r));
    match node {
        Node::Integer(i) => constant(*i),
        Node::Decimal(d) => constant(*d as i64),
        Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
            let f = to_double(node);
            compiled(move |r| f(r) as i64)
        }
        Node::Plus(e) => to_long(e),
        Node::BwNot(e) => {
            let f = to_long(e);
            compiled(move |r| !f(r))
        }
        Node::Shift(op, l, n) => {
            let (a, n) = (to_long(l), to_int(n));
            // Java masks a long shift's count to its low six bits
            match op {
                Bin::Lsh => compiled(move |r| {
                    let v = a(r);
                    v.wrapping_shl((n(r) & 63) as u32)
                }),
                Bin::Rsh => compiled(move |r| {
                    let v = a(r);
                    v >> (n(r) & 63)
                }),
                _ => compiled(move |r| {
                    let v = a(r) as u64;
                    (v >> (n(r) & 63)) as i64
                }),
            }
        }
        Node::Bitwise(op, l, r) => {
            let (a, b) = two(l, r);
            match op {
                Bin::BwAnd => compiled(move |r| {
                    let v = a(r);
                    v & b(r)
                }),
                Bin::BwXor => compiled(move |r| {
                    let v = a(r);
                    v ^ b(r)
                }),
                _ => compiled(move |r| {
                    let v = a(r);
                    v | b(r)
                }),
            }
        }
        Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
            let f = to_int(node);
            compiled(move |r| i64::from(f(r)))
        }
        Node::Conditional(c, a, e) => {
            let (c, a, e) = (to_int(c), to_long(a), to_long(e));
            compiled(move |r| if c(r) != 0 { a(r) } else { e(r) })
        }
        Node::Chain(first, steps) => match chain_kind(steps) {
            ChainKind::Arith => {
                let f = arith_chain(first, steps);
                compiled(move |r| f(r) as i64)
            }
            ChainKind::Shift | ChainKind::Bitwise => long_chain(first, steps),
            _ => {
                let f = int_chain(first, steps);
                compiled(move |r| i64::from(f(r)))
            }
        },
    }
}

/// `node` producing an `int` (logic yields `0` or `1`).
fn to_int(node: &Node) -> Compiled<i32> {
    match node {
        Node::Integer(i) => constant(*i as i32),
        Node::Decimal(d) => constant(*d as i32),
        Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
            let f = to_double(node);
            compiled(move |r| f(r) as i32)
        }
        Node::Plus(e) => to_int(e),
        Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
            let f = to_long(node);
            compiled(move |r| f(r) as i32)
        }
        Node::Not(e) => {
            let f = to_int(e);
            compiled(move |r| i32::from(f(r) == 0))
        }
        Node::Compare(op, l, r) => match op {
            Bin::Lt => binary_op(l, r, |a, b| i32::from(a < b)),
            Bin::Lte => binary_op(l, r, |a, b| i32::from(a <= b)),
            Bin::Gt => binary_op(l, r, |a, b| i32::from(a > b)),
            Bin::Gte => binary_op(l, r, |a, b| i32::from(a >= b)),
            Bin::Eq => binary_op(l, r, |a, b| i32::from(a == b)),
            _ => binary_op(l, r, |a, b| i32::from(a != b)),
        },
        Node::And(l, r) => {
            let (a, b) = (to_int(l), to_int(r));
            compiled(move |r| i32::from(a(r) != 0 && b(r) != 0))
        }
        Node::Or(l, r) => {
            let (a, b) = (to_int(l), to_int(r));
            compiled(move |r| i32::from(a(r) != 0 || b(r) != 0))
        }
        Node::Conditional(c, a, e) => {
            let (c, a, e) = (to_int(c), to_int(a), to_int(e));
            compiled(move |r| if c(r) != 0 { a(r) } else { e(r) })
        }
        Node::Chain(first, steps) => match chain_kind(steps) {
            ChainKind::Arith => {
                let f = arith_chain(first, steps);
                compiled(move |r| f(r) as i32)
            }
            ChainKind::Shift | ChainKind::Bitwise => {
                let f = long_chain(first, steps);
                compiled(move |r| f(r) as i32)
            }
            _ => int_chain(first, steps),
        },
    }
}

/// `JavascriptCompiler.normalizeQuotes`: a variable's name with its quotes
/// all single.
pub(crate) fn normalize_quotes(text: &str) -> String {
    let t: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut in_double = false;
    let mut i = 0;
    while i < t.len() {
        let mut c = t[i];
        if c == '\\' {
            i += 1;
            c = t[i];
            if c == '\\' {
                out.push('\\');
            }
        } else if c == '\'' {
            if in_double {
                out.push('\\');
            } else {
                let j = find_single_quote_string_end(&t, i);
                out.extend(&t[i..j]);
                i = j;
                c = t[i];
            }
        } else if c == '"' {
            c = '\'';
            in_double = !in_double;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn find_single_quote_string_end(t: &[char], mut start: usize) -> usize {
    start += 1;
    while t[start] != '\'' {
        if t[start] == '\\' {
            start += 1;
        }
        start += 1;
    }
    start
}

struct Lower<'a> {
    source: &'a str,
    functions: &'a HashMap<String, ExprFunction>,
    externals: Vec<String>,
}

impl Lower<'_> {
    fn node(&mut self, ast: &Ast) -> Result<Node, ExprError> {
        Ok(match ast {
            Ast::Paren(e) => self.node(e)?,
            Ast::Number(kind, text) => match kind {
                Tok::Hex => Node::Integer(parse_radix(&text[2..], 16)?),
                Tok::Octal => Node::Integer(parse_radix(&text[1..], 8)?),
                _ => Node::Decimal(parse_decimal(text)),
            },
            Ast::External {
                name,
                text,
                args,
                parens,
                start,
            } => {
                let function = if *parens {
                    self.functions.get(name)
                } else {
                    None
                };
                if let Some(f) = function {
                    if args.len() != f.arity {
                        return Err(parse_error(
                            format!(
                                "Invalid expression '{}': Expected ({}) arguments for function \
                                 call ({}), but found ({}).",
                                self.source,
                                f.arity,
                                name,
                                args.len()
                            ),
                            *start,
                        ));
                    }
                    let lowered = args
                        .iter()
                        .map(|a| self.node(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    Node::Call(f.clone(), lowered)
                } else if !*parens || (args.is_empty() && name.contains('.')) {
                    let normalized = normalize_quotes(text);
                    let index = match self.externals.iter().position(|e| *e == normalized) {
                        Some(i) => i,
                        None => {
                            self.externals.push(normalized);
                            self.externals.len() - 1
                        }
                    };
                    Node::Variable(index)
                } else {
                    return Err(parse_error(
                        format!(
                            "Invalid expression '{}': Unrecognized function call ({}).",
                            self.source, name
                        ),
                        *start,
                    ));
                }
            }
            Ast::Unary(op, e) => {
                let inner = Box::new(self.node(e)?);
                match op {
                    Tok::BoolNot => Node::Not(inner),
                    Tok::BwNot => Node::BwNot(inner),
                    Tok::Add => Node::Plus(inner),
                    _ => Node::Neg(inner),
                }
            }
            Ast::Binary(op, l, r) => {
                let (l, r) = (Box::new(self.node(l)?), Box::new(self.node(r)?));
                match op {
                    Bin::Mul | Bin::Div | Bin::Rem | Bin::Add | Bin::Sub => Node::Arith(*op, l, r),
                    Bin::Lsh | Bin::Rsh | Bin::Ush => Node::Shift(*op, l, r),
                    Bin::Lt | Bin::Lte | Bin::Gt | Bin::Gte | Bin::Eq | Bin::Ne => {
                        Node::Compare(*op, l, r)
                    }
                    Bin::BwAnd | Bin::BwXor | Bin::BwOr => Node::Bitwise(*op, l, r),
                    Bin::And => Node::And(l, r),
                    Bin::Or => Node::Or(l, r),
                }
            }
            Ast::Chain(first, steps) => Node::Chain(
                Box::new(self.node(first)?),
                steps
                    .iter()
                    .map(|(op, a)| Ok((*op, self.node(a)?)))
                    .collect::<Result<Vec<_>, ExprError>>()?,
            ),
            Ast::Conditional(c, a, b) => Node::Conditional(
                Box::new(self.node(c)?),
                Box::new(self.node(a)?),
                Box::new(self.node(b)?),
            ),
        })
    }
}

/// `Long.parseLong(digits, radix)` on a literal's digits: a value past
/// `i64::MAX` is a `NumberFormatException`.
fn parse_radix(digits: &str, radix: u32) -> Result<i64, ExprError> {
    i64::from_str_radix(digits, radix).map_err(|_| {
        ExprError::NumberFormat(format!(
            "For input string: \"{digits}\" under radix {radix}"
        ))
    })
}

/// `Double.parseDouble` of a `DECIMAL` token.
fn parse_decimal(text: &str) -> f64 {
    let t = if text.ends_with('.') {
        format!("{text}0")
    } else {
        text.to_string()
    };
    t.parse::<f64>()
        .expect("the lexer's DECIMAL is a Rust float")
}

// ---------------------------------------------------------------------------
// Expression
// ---------------------------------------------------------------------------

/// `Expression`: a compiled expression, evaluated over the values of its
/// [`variables`](Self::variables).
#[derive(Clone)]
pub struct Expression {
    /// `sourceText`.
    pub source_text: String,
    /// `variables`: the distinct external names, in order of first use.
    pub variables: Vec<String>,
    root: Compiled<f64>,
}

impl std::fmt::Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expression")
            .field("source_text", &self.source_text)
            .field("variables", &self.variables)
            .finish_non_exhaustive()
    }
}

impl Expression {
    /// `evaluate(functionValues)`: `values[i]` are the values of
    /// `variables[i]`.
    ///
    /// # Errors
    /// What reading a variable's value reports; `IllegalArgument` when
    /// `values` has fewer entries than there are variables.
    pub fn evaluate(&self, values: &mut [BoxDoubleValues<'_>]) -> crate::Result<f64> {
        if values.len() < self.variables.len() {
            return Err(crate::Error::IllegalArgument(format!(
                "{} values for the {} variables of {}",
                values.len(),
                self.variables.len(),
                self.source_text
            )));
        }
        self.run(Values::Boxed(values), 0)
    }

    /// The value at `doc` over `slots` (one per variable): each variable
    /// read is its slot's, positioned on `doc` at its first read (Java's
    /// `ZeroWhenUnpositioned` under `ExpressionFunctionValues`).
    ///
    /// # Errors
    /// As [`Self::evaluate`].
    pub(crate) fn evaluate_at(&self, slots: &mut [VarSlot<'_>], doc: i32) -> crate::Result<f64> {
        if slots.len() < self.variables.len() {
            return Err(crate::Error::IllegalArgument(format!(
                "{} values for the {} variables of {}",
                slots.len(),
                self.variables.len(),
                self.source_text
            )));
        }
        self.run(Values::Slots(slots), doc)
    }

    /// The value at each of a batch's `out.len()` documents, `cols[i * n +
    /// j]` the value (`0` for none) of variable `i` at the batch's `j`th:
    /// what [`Self::evaluate_at`] gives over the same variable values.
    ///
    /// # Errors
    /// `IllegalArgument` when `cols` has fewer values than the variables
    /// over the batch, or the batch is past an `i32`.
    pub(crate) fn evaluate_batch(&self, cols: &[f64], out: &mut [f64]) -> crate::Result<()> {
        let n = out.len();
        if cols.len() < self.variables.len() * n || i32::try_from(n).is_err() {
            return Err(crate::Error::IllegalArgument(format!(
                "{} values for the {} variables of {} over {n} documents",
                cols.len(),
                self.variables.len(),
                self.source_text
            )));
        }
        let mut reads = Reads {
            values: Values::Columns { cols, n },
            doc: 0,
            error: None,
        };
        for (j, o) in (0i32..).zip(out.iter_mut()) {
            reads.doc = j;
            *o = (self.root)(&mut reads);
        }
        // a column read cannot fail
        debug_assert!(reads.error.is_none());
        Ok(())
    }

    #[inline]
    fn run(&self, values: Values<'_, '_>, doc: i32) -> crate::Result<f64> {
        let mut reads = Reads {
            values,
            doc,
            error: None,
        };
        let v = (self.root)(&mut reads);
        match reads.error {
            Some(e) => Err(e),
            None => Ok(v),
        }
    }
}

/// What a variable's slot reads: its own values, or values it shares with
/// other slots (a caching source's variable that nested expressions name
/// too).
pub(crate) enum VarValues<'c> {
    /// Read by this slot alone.
    Owned(BoxDoubleValues<'c>),
    /// Shared with other slots.
    Shared(Rc<RefCell<SharedValues<'c>>>),
}

/// Values several readers share (the variables naming one source under a
/// caching expression, the scores every variable reads): the values, and
/// the last batch they filled, which a second reader of the same batch is
/// handed rather than reading the values again (they read forward only).
pub(crate) struct SharedValues<'c> {
    pub(crate) values: BoxDoubleValues<'c>,
    /// The last batch's documents and scores, and its values.
    docs: Vec<i32>,
    scores: Vec<f32>,
    out: Vec<f64>,
    has: Vec<bool>,
}

impl<'c> SharedValues<'c> {
    pub(crate) fn new(values: BoxDoubleValues<'c>) -> Self {
        Self {
            values,
            docs: Vec::new(),
            scores: Vec::new(),
            out: Vec::new(),
            has: Vec::new(),
        }
    }

    /// [`DoubleValues::fill_batch`] of the values, once per batch: the same
    /// documents and scores as the last batch are its values again.
    pub(crate) fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> crate::Result<()> {
        let n = docs.len();
        let scores = &scores[..n];
        let same = self.docs == docs
            && self
                .scores
                .iter()
                .map(|s| s.to_bits())
                .eq(scores.iter().map(|s| s.to_bits()));
        if !same {
            // a failed batch leaves nothing to hand on
            self.docs.clear();
            self.out.resize(n, 0.0);
            self.has.resize(n, false);
            self.values
                .fill_batch(docs, scores, &mut self.out, &mut self.has)?;
            self.docs.extend_from_slice(docs);
            self.scores.clear();
            self.scores.extend_from_slice(scores);
        }
        out[..n].copy_from_slice(&self.out[..n]);
        has[..n].copy_from_slice(&self.has[..n]);
        Ok(())
    }
}

/// One variable of an evaluation: `ExpressionValueSource.zeroWhenUnpositioned`
/// -- always a value, `0` where the wrapped values have none, read at most
/// once per document.
pub(crate) struct VarSlot<'c> {
    values: VarValues<'c>,
    /// The document `value` is for; `-1` before the first read.
    doc: i32,
    value: f64,
}

impl<'c> VarSlot<'c> {
    pub(crate) fn new(values: VarValues<'c>) -> Self {
        Self {
            values,
            doc: -1,
            value: 0.0,
        }
    }

    /// The value at `doc`, read from the wrapped values only at the first
    /// read for `doc` (a failed read is tried again at the next one, as
    /// Java's `computed` stays `false` when the read throws).
    #[inline]
    fn value_at(&mut self, doc: i32) -> crate::Result<f64> {
        if self.doc != doc {
            self.value = match &mut self.values {
                VarValues::Owned(v) => {
                    if v.advance_exact(doc)? {
                        v.double_value()?
                    } else {
                        0.0
                    }
                }
                VarValues::Shared(v) => {
                    let v = &mut v.borrow_mut().values;
                    if v.advance_exact(doc)? {
                        v.double_value()?
                    } else {
                        0.0
                    }
                }
            };
            self.doc = doc;
        }
        Ok(self.value)
    }

    /// Whether the wrapped values can be read a batch at a time
    /// ([`DoubleValues::batch_capable`]).
    pub(crate) fn batch_capable(&self) -> bool {
        match &self.values {
            VarValues::Owned(v) => v.batch_capable(),
            VarValues::Shared(v) => v.borrow().values.batch_capable(),
        }
    }

    /// The values at each of `docs` into `out` (`0` where there is none),
    /// `has` scratch, both as long as `docs`; the slot is left on the last
    /// document, with its value.
    pub(crate) fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> crate::Result<()> {
        let n = docs.len();
        let (out, has) = (&mut out[..n], &mut has[..n]);
        match &mut self.values {
            VarValues::Owned(v) => v.fill_batch(docs, scores, out, has)?,
            VarValues::Shared(v) => v.borrow_mut().fill_batch(docs, scores, out, has)?,
        }
        for (o, &h) in out.iter_mut().zip(has.iter()) {
            if !h {
                *o = 0.0;
            }
        }
        if let (Some(&doc), Some(&value)) = (docs.last(), out.last()) {
            self.doc = doc;
            self.value = value;
        }
        Ok(())
    }
}

/// The values an evaluation reads its variables from.
enum Values<'a, 'c> {
    /// [`Expression::evaluate`]'s, each already positioned.
    Boxed(&'a mut [BoxDoubleValues<'c>]),
    /// [`Expression::evaluate_at`]'s, positioned on [`Reads::doc`] as read.
    Slots(&'a mut [VarSlot<'c>]),
    /// [`Expression::evaluate_batch`]'s: variable `i`'s values for the
    /// batch at `cols[i * n..]`, read at the batch index [`Reads::doc`].
    Columns { cols: &'a [f64], n: usize },
}

/// The variables' values as an evaluation reads them. A failed read parks
/// its error and reads `0`: the evaluation runs to its end and then reports
/// the first error (Java's generated code throws at the read; the result is
/// the same error, and no value). Once an error is parked, later reads
/// return `0` without reading, but the rest of the tree still runs: a
/// custom function may be called with that `0`.
struct Reads<'a, 'c> {
    values: Values<'a, 'c>,
    /// The document [`Values::Slots`] are read at (the batch index
    /// [`Values::Columns`] are).
    doc: i32,
    error: Option<crate::Error>,
}

impl Reads<'_, '_> {
    #[inline]
    fn read(&mut self, i: usize) -> f64 {
        if self.error.is_some() {
            return 0.0;
        }
        let read = match &mut self.values {
            Values::Slots(slots) => slots[i].value_at(self.doc),
            Values::Boxed(values) => values[i].double_value(),
            Values::Columns { cols, n } => Ok(cols[i * *n + self.doc as usize]),
        };
        match read {
            Ok(v) => v,
            Err(e) => self.park(e),
        }
    }

    #[cold]
    #[inline(never)]
    fn park(&mut self, e: crate::Error) -> f64 {
        self.error = Some(e);
        0.0
    }
}

/// `JavascriptCompiler.compile(sourceText)`: with the
/// [`default_functions`].
///
/// # Errors
/// [`ExprError::Parse`] for a syntax error, a call to an unknown function or
/// with the wrong number of arguments; [`ExprError::NumberFormat`] for an
/// integer literal that does not fit a `long`.
pub fn compile(source_text: &str) -> Result<Expression, ExprError> {
    compile_with(source_text, defaults())
}

/// `JavascriptCompiler.compile(sourceText, functions)`.
///
/// # Errors
/// As [`compile`].
pub fn compile_with(
    source_text: &str,
    functions: &HashMap<String, ExprFunction>,
) -> Result<Expression, ExprError> {
    let (tokens, lex_error) = lex(source_text);
    let ast = Parser {
        tokens,
        pos: 0,
        lex_error,
        nesting: 0,
    }
    .compile()?;
    let mut lower = Lower {
        source: source_text,
        functions,
        externals: Vec::new(),
    };
    let root = to_double(&lower.node(&ast)?);
    Ok(Expression {
        source_text: source_text.to_string(),
        variables: lower.externals,
        root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::values_source::ConstantDoubleValues;

    fn run(src: &str, vars: &[f64]) -> f64 {
        let e = compile(src).unwrap();
        let mut vals: Vec<BoxDoubleValues<'_>> = vars
            .iter()
            .map(|v| Box::new(ConstantDoubleValues(*v)) as BoxDoubleValues<'_>)
            .collect();
        e.evaluate(&mut vals).unwrap()
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(run("1 + 2 * 3", &[]), 7.0);
        assert_eq!(run("10 - 4 - 3", &[]), 3.0);
        assert_eq!(run("-2 * 3", &[]), -6.0);
        assert_eq!(run("1 ? 2 : 0 ? 3 : 4", &[]), 2.0);
        assert_eq!(run("0 ? 2 : 0 ? 3 : 4", &[]), 4.0);
        assert_eq!(run("1 < 2 == 1", &[]), 1.0);
        assert_eq!(run("1 | 6 & 3 ^ 1", &[]), 3.0);
    }

    #[test]
    fn jvm_conversions_match() {
        // `!0.5`: d2i(0.5) is 0, so it is true
        assert_eq!(run("!0.5", &[]), 1.0);
        assert_eq!(run("!1", &[]), 0.0);
        assert_eq!(run("~5", &[]), -6.0);
        assert_eq!(run("1 << 40", &[]), (1i64 << 40) as f64);
        assert_eq!(run("-1 >>> 60", &[]), 15.0);
        assert_eq!(run("1.9 & 3", &[]), 1.0);
        assert_eq!(run("0x10 + 010", &[]), 24.0);
    }

    #[test]
    fn variables_are_distinct_and_normalised() {
        let e = compile("a + b + a + doc[\"x\"].value").unwrap();
        assert_eq!(e.variables, vec!["a", "b", "doc['x'].value"]);
        assert_eq!(run("a * a", &[3.0]), 9.0);
    }

    #[test]
    fn functions_and_their_arity() {
        assert_eq!(run("max(1, 2) + min(5, 4)", &[]), 6.0);
        assert_eq!(run("sqrt(16)", &[]), 4.0);
        let e = compile("tan(1, 1)").unwrap_err();
        assert!(matches!(e, ExprError::Parse(p) if p.error_offset == 0));
        let e = compile("1 + nosuch(2)").unwrap_err();
        assert!(matches!(e, ExprError::Parse(p) if p.error_offset == 4));
        // a custom function past four arguments, and one over a variable
        let mut functions = default_functions();
        functions.insert(
            "digits".into(),
            ExprFunction::new(6, |a| a.iter().fold(0.0, |n, d| n * 10.0 + d)),
        );
        let e = compile_with("digits(1, 2, 3, x, 5, 6) + abs(-x)", &functions).unwrap();
        let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(ConstantDoubleValues(4.0))];
        assert_eq!(e.evaluate(&mut vals).unwrap(), 123_460.0);
        assert!(format!("{e:?}").starts_with("Expression { source_text: \"digits("));
    }

    /// Values whose every read fails.
    struct Failing;

    impl crate::values_source::DoubleValues for Failing {
        fn advance_exact(&mut self, _doc: i32) -> crate::Result<bool> {
            Ok(true)
        }
        fn double_value(&mut self) -> crate::Result<f64> {
            Err(crate::Error::IllegalState("unreadable".into()))
        }
    }

    #[test]
    fn a_failed_read_is_the_evaluations_error() {
        // the first failure is reported, whichever branch read it
        for src in ["x * 2 + 1", "x > 0 ? 1 : x", "(x << 1) | 1"] {
            let e = compile(src).unwrap();
            let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(Failing)];
            let err = e.evaluate(&mut vals).unwrap_err();
            assert!(err.to_string().contains("unreadable"), "{src}: {err}");
        }
        // fewer values than variables is refused, not an out-of-bounds read
        let e = compile("x + y").unwrap();
        let mut vals: Vec<BoxDoubleValues<'_>> = vec![Box::new(ConstantDoubleValues(1.0))];
        let err = e.evaluate(&mut vals).unwrap_err().to_string();
        assert!(err.contains("1 values for the 2 variables"), "{err}");
    }

    #[test]
    fn syntax_errors_carry_the_offending_offset() {
        for (src, at) in [
            ("100 100", 4),
            ("7*/-8", 2),
            ("", 0),
            ("()", 1),
            ("0y1234", 1),
        ] {
            match compile(src) {
                Err(ExprError::Parse(p)) => assert_eq!(p.error_offset, at, "{src}"),
                other => panic!("{src}: {other:?}"),
            }
        }
        assert!(matches!(compile("a @ b"), Err(ExprError::Parse(p)) if p.error_offset == 2));
    }

    #[test]
    fn oversized_literals_are_number_format_errors() {
        assert!(matches!(
            compile("0xffffffffffffffff"),
            Err(ExprError::NumberFormat(_))
        ));
    }

    /// Trees past [`MAX_DEPTH`] are refused, whichever way they nest; ones at
    /// the limit compile and evaluate (on a test thread's 2 MiB stack, in a
    /// debug build).
    #[test]
    fn nesting_is_bounded() {
        let deep = |n: usize| {
            [
                format!("{}1{}", "(".repeat(n), ")".repeat(n)),
                format!("{}1", "-".repeat(n)),
                format!("{}1{}", "1-(".repeat(n), ")".repeat(n)),
                format!("{}1{}", "abs(".repeat(n), ")".repeat(n)),
                format!("{}1{}", "1?".repeat(n), ":0".repeat(n)),
            ]
        };
        // a parenthesis or a conditional's branch opens two parser frames, a
        // parenthesised right operand three
        for text in deep(MAX_DEPTH as usize / 3 - 1) {
            run(&text, &[]);
        }
        for text in deep(MAX_DEPTH as usize + 1)
            .into_iter()
            .chain(deep(200_000))
        {
            match compile(&text) {
                Err(ExprError::Parse(p)) => assert!(p.message.contains("deeper than"), "{p:?}"),
                other => panic!("{:?}", other.map(|e| e.source_text.len())),
            }
        }
        // The deepest tree of each shape compiles and evaluates on a quarter
        // of a JVM thread's default 1 MiB stack in an optimised build (it
        // needs between 128 and 256 KiB); an unoptimised one needs about
        // 1.5 MiB.
        let stack = if cfg!(debug_assertions) {
            4 << 20
        } else {
            256 << 10
        };
        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                for shape in 0..5 {
                    let deepest = (1..=MAX_DEPTH as usize)
                        .rev()
                        .map(|n| deep(n)[shape].clone())
                        .find(|text| compile(text).is_ok())
                        .unwrap();
                    run(&deepest, &[]);
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn long_chains_are_flat_and_typed_as_nested_operators() {
        // each level's run, past `CHAIN`, equals the left-nested parenthesised
        // form (binary nodes) under several assignments
        let vars = ["a", "b", "c"];
        let values = [[3.0, -2.5, 7.0], [0.0, 1.0, -1.0], [f64::NAN, 1e300, 0.5]];
        for ops in [
            &["*", "/", "%"][..],
            &["+", "-"],
            &["<<", ">>", ">>>"],
            &["<", "<=", ">", ">="],
            &["==", "!="],
            &["&"],
            &["^"],
            &["|"],
            &["&&"],
            &["||"],
        ] {
            for n in [CHAIN, CHAIN + 1, 40] {
                let mut flat = String::from("a");
                let mut nested = String::from("a");
                for i in 0..n {
                    let (op, v) = (ops[i % ops.len()], vars[(i + 1) % vars.len()]);
                    flat = format!("{flat} {op} {v}");
                    nested = format!("({nested} {op} {v})");
                }
                for vals in values {
                    for wrap in ["{}", "-({})", "!({})", "~({})", "({}) ? 2 : 3"] {
                        let (f, g) = (wrap.replace("{}", &flat), wrap.replace("{}", &nested));
                        assert_eq!(
                            run(&f, &vals).to_bits(),
                            run(&g, &vals).to_bits(),
                            "{f} at {vals:?}"
                        );
                    }
                }
            }
        }
        // a chain is one level, however long, so only real nesting is bounded
        let sum = format!("a{}", " + a".repeat(100_000));
        assert_eq!(run(&sum, &[1.0]), 100_001.0);
        let nested = format!("{}{}{}", "(".repeat(100), sum, ")".repeat(100));
        assert_eq!(run(&nested, &[0.5]), 50_000.5);
        // `&&` and `||` chains stop at their first deciding operand
        let mut counting: Vec<BoxDoubleValues<'_>> = vec![Box::new(Failing)];
        let and = compile(&format!("0{} && x", " && 1".repeat(CHAIN))).unwrap();
        assert_eq!(and.evaluate(&mut counting).unwrap(), 0.0);
        let or = compile(&format!("1{} || x", " || 0".repeat(CHAIN))).unwrap();
        assert_eq!(or.evaluate(&mut counting).unwrap(), 1.0);
    }

    #[test]
    fn java_math_edge_cases() {
        assert!(run("pow(1, 1 / 0)", &[]).is_nan());
        assert!(run("max(0 / 0, 1)", &[]).is_nan());
        assert_eq!(run("max(-0, 0)", &[]).to_bits(), 0.0f64.to_bits());
        assert_eq!(run("min(0, -0)", &[]).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn normalize_quotes_unifies_quoting() {
        assert_eq!(normalize_quotes("a[\"b\"]"), "a['b']");
        assert_eq!(normalize_quotes("a['it\\'s']"), "a['it\\'s']");
        assert_eq!(normalize_quotes("a[\"it's\"]"), "a['it\\'s']");
    }
}
