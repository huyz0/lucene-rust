//! The lexer: `PainlessLexer.g4`, longest match with the rule order breaking
//! ties, `/` read as a regex or a division by the token before it, and the
//! name after a `.` lexed on its own (so `a.0.1` is two list shortcuts).

use super::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Tok {
    // punctuation
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Lp,
    Rp,
    Dot,
    NsDot,
    Comma,
    Semicolon,
    // keywords
    If,
    In,
    Else,
    While,
    Do,
    For,
    Continue,
    Break,
    Return,
    New,
    Try,
    Catch,
    Throw,
    This,
    InstanceOf,
    // operators
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
    EqRef,
    Ne,
    NeRef,
    BwAnd,
    Xor,
    BwOr,
    BoolAnd,
    BoolOr,
    Cond,
    Colon,
    Elvis,
    Ref,
    Arrow,
    Find,
    Match,
    Incr,
    Decr,
    Assign,
    AAdd,
    ASub,
    AMul,
    ADiv,
    ARem,
    AAnd,
    AXor,
    AOr,
    ALsh,
    ARsh,
    AUsh,
    // literals and names
    Octal(String),
    Hex(String),
    Integer(String),
    Decimal(String),
    Str(String),
    Regex(String, String),
    True,
    False,
    Null,
    Primitive(String),
    Def,
    Id(String),
    DotInteger(String),
    DotId(String),
    Eof,
}

#[derive(Debug, Clone)]
pub(super) struct Token {
    pub tok: Tok,
    pub pos: usize,
}

fn err(offset: usize, message: impl Into<String>) -> ParseError {
    ParseError {
        offset,
        message: message.into(),
    }
}

const OPERATORS: &[(&str, Tok)] = &[
    (">>>=", Tok::AUsh),
    ("===", Tok::EqRef),
    ("!==", Tok::NeRef),
    ("==~", Tok::Match),
    ("<<=", Tok::ALsh),
    (">>=", Tok::ARsh),
    (">>>", Tok::Ush),
    ("<<", Tok::Lsh),
    (">>", Tok::Rsh),
    ("<=", Tok::Lte),
    (">=", Tok::Gte),
    ("==", Tok::Eq),
    ("!=", Tok::Ne),
    ("&&", Tok::BoolAnd),
    ("||", Tok::BoolOr),
    ("?:", Tok::Elvis),
    ("?.", Tok::NsDot),
    ("::", Tok::Ref),
    ("->", Tok::Arrow),
    ("=~", Tok::Find),
    ("++", Tok::Incr),
    ("--", Tok::Decr),
    ("+=", Tok::AAdd),
    ("-=", Tok::ASub),
    ("*=", Tok::AMul),
    ("%=", Tok::ARem),
    ("&=", Tok::AAnd),
    ("^=", Tok::AXor),
    ("|=", Tok::AOr),
    ("{", Tok::LBrack),
    ("}", Tok::RBrack),
    ("[", Tok::LBrace),
    ("]", Tok::RBrace),
    ("(", Tok::Lp),
    (")", Tok::Rp),
    (".", Tok::Dot),
    (",", Tok::Comma),
    (";", Tok::Semicolon),
    ("!", Tok::BoolNot),
    ("~", Tok::BwNot),
    ("*", Tok::Mul),
    ("%", Tok::Rem),
    ("+", Tok::Add),
    ("-", Tok::Sub),
    ("<", Tok::Lt),
    (">", Tok::Gt),
    ("&", Tok::BwAnd),
    ("^", Tok::Xor),
    ("|", Tok::BwOr),
    ("?", Tok::Cond),
    (":", Tok::Colon),
    ("=", Tok::Assign),
];

fn keyword(s: &str) -> Option<Tok> {
    Some(match s {
        "if" => Tok::If,
        "in" => Tok::In,
        "else" => Tok::Else,
        "while" => Tok::While,
        "do" => Tok::Do,
        "for" => Tok::For,
        "continue" => Tok::Continue,
        "break" => Tok::Break,
        "return" => Tok::Return,
        "new" => Tok::New,
        "try" => Tok::Try,
        "catch" => Tok::Catch,
        "throw" => Tok::Throw,
        "this" => Tok::This,
        "instanceof" => Tok::InstanceOf,
        "true" => Tok::True,
        "false" => Tok::False,
        "null" => Tok::Null,
        "def" => Tok::Def,
        "boolean" | "byte" | "short" | "char" | "int" | "long" | "float" | "double" => {
            Tok::Primitive(s.to_string())
        }
        _ => return None,
    })
}

/// Whether a `/` after `prev` starts a regex (`EnhancedPainlessLexer`).
fn slash_is_regex(prev: Option<&Tok>) -> bool {
    !matches!(
        prev,
        Some(
            Tok::RBrace
                | Tok::Rp
                | Tok::Octal(_)
                | Tok::Hex(_)
                | Tok::Integer(_)
                | Tok::Decimal(_)
                | Tok::Id(_)
                | Tok::DotInteger(_)
                | Tok::DotId(_)
        )
    )
}

fn is_id_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_id_part(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

fn digits(b: &[u8], mut i: usize, ok: impl Fn(u8) -> bool) -> usize {
    while i < b.len() && ok(b[i]) {
        i += 1;
    }
    i
}

/// The longest numeric token at `i` (rule order OCTAL, HEX, INTEGER, DECIMAL
/// breaks ties).
fn number(src: &str, i: usize) -> (Tok, usize) {
    let b = src.as_bytes();
    let suffix_l = |j: usize| (j < b.len() && matches!(b[j], b'l' | b'L')) as usize;
    let mut best: Option<(Tok, usize)> = None;
    let mut take = |tok: Tok, end: usize| {
        if best.as_ref().is_none_or(|(_, e)| end > *e) {
            best = Some((tok, end));
        }
    };
    let text = |end: usize| src[i..end].to_string();
    if b[i] == b'0' {
        let d = digits(b, i + 1, |c| (b'0'..=b'7').contains(&c));
        if d > i + 1 {
            let end = d + suffix_l(d);
            take(Tok::Octal(text(end)), end);
        }
        if i + 1 < b.len() && matches!(b[i + 1], b'x' | b'X') {
            let d = digits(b, i + 2, |c| c.is_ascii_hexdigit());
            if d > i + 2 {
                let end = d + suffix_l(d);
                take(Tok::Hex(text(end)), end);
            }
        }
    }
    let int_end = if b[i] == b'0' {
        i + 1
    } else {
        digits(b, i, |c| c.is_ascii_digit())
    };
    let sfx = |j: usize| {
        (j < b.len() && matches!(b[j], b'l' | b'L' | b'f' | b'F' | b'd' | b'D')) as usize
    };
    let end = int_end + sfx(int_end);
    take(Tok::Integer(text(end)), end);
    // DECIMAL: int (. digits)? (exp)? [fFdD]?
    let mut j = int_end;
    if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
        j = digits(b, j + 1, |c| c.is_ascii_digit());
    }
    if j < b.len() && matches!(b[j], b'e' | b'E') {
        let mut k = j + 1;
        if k < b.len() && matches!(b[k], b'+' | b'-') {
            k += 1;
        }
        let e = digits(b, k, |c| c.is_ascii_digit());
        if e > k {
            j = e;
        }
    }
    if j < b.len() && matches!(b[j], b'f' | b'F' | b'd' | b'D') {
        j += 1;
    }
    take(Tok::Decimal(text(j)), j);
    best.expect("INTEGER always matches")
}

/// A string literal at `i` (its opening quote); the text with `\\` and the
/// quote's escape resolved, and the end offset.
fn string(src: &str, i: usize) -> Result<(String, usize), ParseError> {
    let b = src.as_bytes();
    let q = b[i];
    let mut out = String::new();
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            c if c == q => return Ok((out, j + 1)),
            b'\\' => match b.get(j + 1) {
                Some(&c) if c == q || c == b'\\' => {
                    out.push(c as char);
                    j += 2;
                }
                _ => {
                    return Err(err(
                        j,
                        format!(
                            "invalid escape sequence: the only valid escape sequences in strings starting with [{}] are [\\\\] and [\\{}]",
                            q as char, q as char
                        ),
                    ))
                }
            },
            _ => {
                let ch = src[j..].chars().next().expect("in bounds");
                out.push(ch);
                j += ch.len_utf8();
            }
        }
    }
    Err(err(i, "unterminated string literal"))
}

/// A regex at `i` (its opening slash): pattern, flags, end offset.
fn regex(src: &str, i: usize) -> Option<(String, String, usize)> {
    let b = src.as_bytes();
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\n' => return None,
            b'\\' => {
                if b.get(j + 1).is_none_or(|&c| c == b'\n') {
                    return None;
                }
                j += 2;
            }
            b'/' => {
                if j == i + 1 {
                    return None;
                }
                let pat = src[i + 1..j].to_string();
                let f = digits(b, j + 1, |c| b"cilmsUux".contains(&c));
                return Some((pat, src[j + 1..f].to_string(), f));
            }
            _ => j += 1,
        }
    }
    None
}

pub(super) fn lex(src: &str) -> Result<Vec<Token>, ParseError> {
    let b = src.as_bytes();
    let mut out: Vec<Token> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // After a dot only a name or an integer may follow, with no space.
        if matches!(out.last().map(|t| &t.tok), Some(Tok::Dot | Tok::NsDot)) {
            if is_id_start(c) {
                let e = digits(b, i, is_id_part);
                out.push(Token {
                    tok: Tok::DotId(src[i..e].to_string()),
                    pos: i,
                });
                i = e;
            } else if c.is_ascii_digit() {
                let e = if c == b'0' {
                    i + 1
                } else {
                    digits(b, i, |c| c.is_ascii_digit())
                };
                out.push(Token {
                    tok: Tok::DotInteger(src[i..e].to_string()),
                    pos: i,
                });
                i = e;
            } else {
                return Err(err(i, "expected a name or an integer after '.'"));
            }
            continue;
        }
        if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            i = digits(b, i, |c| c != b'\n' && c != b'\r');
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            match src[i + 2..].find("*/") {
                Some(e) => i += 2 + e + 2,
                None => return Err(err(i, "unterminated comment")),
            }
            continue;
        }
        let pos = i;
        if c == b'/' {
            let prev = out.last().map(|t| &t.tok);
            let is_regex = slash_is_regex(prev);
            let re = if is_regex { regex(src, i) } else { None };
            let adiv = b.get(i + 1) == Some(&b'=');
            let (tok, end) = match (re, adiv) {
                (Some((p, f, e)), _) => (Tok::Regex(p, f), e),
                (_, true) => (Tok::ADiv, i + 2),
                _ if !is_regex => (Tok::Div, i + 1),
                _ => return Err(err(i, "unexpected '/'")),
            };
            out.push(Token { tok, pos });
            i = end;
            continue;
        }
        if c == b'"' || c == b'\'' {
            let (s, e) = string(src, i)?;
            out.push(Token {
                tok: Tok::Str(s),
                pos,
            });
            i = e;
            continue;
        }
        if c.is_ascii_digit() {
            let (tok, e) = number(src, i);
            out.push(Token { tok, pos });
            i = e;
            continue;
        }
        if is_id_start(c) {
            let e = digits(b, i, is_id_part);
            let word = &src[i..e];
            let tok = keyword(word).unwrap_or_else(|| Tok::Id(word.to_string()));
            out.push(Token { tok, pos });
            i = e;
            continue;
        }
        let rest = &src[i..];
        match OPERATORS.iter().find(|(s, _)| rest.starts_with(s)) {
            Some((s, tok)) => {
                out.push(Token {
                    tok: tok.clone(),
                    pos,
                });
                i += s.len();
            }
            None => {
                let ch = rest.chars().next().expect("in bounds");
                return Err(err(i, format!("unexpected character [{ch}]")));
            }
        }
    }
    out.push(Token {
        tok: Tok::Eof,
        pos: src.len(),
    });
    Ok(out)
}
