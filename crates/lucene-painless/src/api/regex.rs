//! `java.util.regex.Pattern` and `Matcher` on the `regex` crate.
//!
//! Java's syntax maps onto `regex`'s for what scripts use (classes,
//! quantifiers, groups, named groups, inline flags); backreferences and
//! lookaround, which `regex` does not have, refuse the script at compile
//! time.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::api::{self, int_arg, npe, str_arg};
use crate::lambda::invoke;
use crate::object::{downcast, JObject};
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

const CASE_INSENSITIVE: i32 = 2;
const COMMENTS: i32 = 4;
const MULTILINE: i32 = 8;
const LITERAL: i32 = 16;
const DOTALL: i32 = 32;
const UNICODE_CASE: i32 = 64;

#[derive(Debug)]
pub struct JPattern {
    pub source: String,
    pub flags: i32,
    pub re: regex::Regex,
}

/// Translates a Java pattern into `regex` syntax.
fn translate(p: &str) -> Result<String, String> {
    let mut out = String::with_capacity(p.len());
    let mut chars = p.chars().peekable();
    let mut in_class = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let n = chars.next().ok_or("trailing backslash")?;
                match n {
                    '1'..='9' if !in_class => return Err("backreference".into()),
                    'Q' => {
                        let mut lit = String::new();
                        loop {
                            match chars.next() {
                                Some('\\') if chars.peek() == Some(&'E') => {
                                    chars.next();
                                    break;
                                }
                                Some(ch) => lit.push(ch),
                                None => break,
                            }
                        }
                        out.push_str(&regex::escape(&lit));
                    }
                    'h' => out.push_str("[\\t \\x{A0}\\x{1680}\\x{180e}\\x{2000}-\\x{200a}\\x{202f}\\x{205f}\\x{3000}]"),
                    'Z' => out.push_str("(?:\\n?\\z)"),
                    'p' | 'P' => {
                        out.push('\\');
                        out.push(n);
                        if chars.peek() == Some(&'{') {
                            let mut name = String::new();
                            for ch in chars.by_ref() {
                                name.push(ch);
                                if ch == '}' {
                                    break;
                                }
                            }
                            let inner = name.trim_matches(['{', '}']);
                            let mapped = match inner {
                                "Lower" | "javaLowerCase" => "Ll",
                                "Upper" | "javaUpperCase" => "Lu",
                                "Alpha" => "Alphabetic",
                                "Digit" => "Nd",
                                "Punct" => "P",
                                "Space" | "javaWhitespace" => "White_Space",
                                "Alnum" => return Err("\\p{Alnum}".into()),
                                o => o.trim_start_matches("Is").trim_start_matches("In"),
                            };
                            out.push('{');
                            out.push_str(mapped);
                            out.push('}');
                        }
                    }
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            }
            '[' => {
                in_class = true;
                out.push(c);
                if chars.peek() == Some(&'^') {
                    out.push(chars.next().unwrap_or('^'));
                }
                if chars.peek() == Some(&']') {
                    chars.next();
                    out.push_str("\\]");
                }
            }
            ']' => {
                in_class = false;
                out.push(c);
            }
            '(' if !in_class && chars.peek() == Some(&'?') => {
                out.push('(');
                out.push(chars.next().unwrap_or('?'));
                match chars.peek() {
                    Some('=') | Some('!') => return Err("lookahead".into()),
                    Some('<') => {
                        chars.next();
                        match chars.peek() {
                            Some('=') | Some('!') => return Err("lookbehind".into()),
                            _ => out.push_str("P<"),
                        }
                    }
                    Some('>') => return Err("atomic group".into()),
                    _ => {}
                }
            }
            '*' | '+' | '?' | '}' if !in_class && chars.peek() == Some(&'+') => {
                chars.next();
                return Err("possessive quantifier".into());
            }
            _ => out.push(c),
        }
    }
    Ok(out)
}

/// Compiles a regex constant (`/.../flags`).
pub fn compile_pattern(p: &str, flags: i32) -> Result<Value, String> {
    let body = if flags & LITERAL != 0 {
        regex::escape(p)
    } else {
        translate(p)?
    };
    let mut prefix = String::new();
    if flags & CASE_INSENSITIVE != 0 {
        prefix.push('i');
    }
    if flags & MULTILINE != 0 {
        prefix.push('m');
    }
    if flags & DOTALL != 0 {
        prefix.push('s');
    }
    if flags & COMMENTS != 0 {
        prefix.push('x');
    }
    let _ = UNICODE_CASE;
    let full = if prefix.is_empty() {
        body
    } else {
        format!("(?{prefix}){body}")
    };
    let re = regex::Regex::new(&full).map_err(|e| e.to_string())?;
    Ok(Value::object(JPattern {
        source: p.to_string(),
        flags,
        re,
    }))
}

fn pattern_of(v: &Value) -> Result<&JPattern, Throw> {
    downcast::<JPattern>(v).ok_or_else(|| {
        if v.is_null() {
            npe()
        } else {
            Throw::class_cast(format!("{} is not a Pattern", v.class_name()))
        }
    })
}

/// `s =~ /re/`: whether the pattern occurs in `s`.
pub fn find_op(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let (p, s) = if downcast::<JPattern>(&args[1]).is_some() {
        (&args[1], &args[0])
    } else {
        (&args[0], &args[1])
    };
    let s = s.java_to_string_checked()?;
    Ok(Value::Bool(pattern_of(p)?.re.is_match(&s)))
}

/// `s ==~ /re/`: whether the pattern matches all of `s`.
pub fn match_op(_vm: &mut Vm, args: &[Value]) -> Result<Value, Throw> {
    let (p, s) = if downcast::<JPattern>(&args[1]).is_some() {
        (&args[1], &args[0])
    } else {
        (&args[0], &args[1])
    };
    let s = s.java_to_string_checked()?;
    Ok(Value::Bool(full_match(&pattern_of(p)?.re, &s).is_some()))
}

fn full_match<'a>(re: &regex::Regex, s: &'a str) -> Option<regex::Captures<'a>> {
    // A match of the whole input: anchor the pattern.
    let anchored = regex::Regex::new(&format!("^(?:{})$", re.as_str())).ok()?;
    anchored.captures(s)
}

trait CharSeq {
    fn java_to_string_checked(&self) -> Result<String, Throw>;
}

impl CharSeq for Value {
    fn java_to_string_checked(&self) -> Result<String, Throw> {
        match self {
            Value::Null => Err(npe()),
            Value::Str(s) => Ok(s.to_string()),
            other => Ok(other.java_to_string()),
        }
    }
}

pub fn static_lookup(name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (name, arity) {
        ("quote", 1) => |_vm, a| Ok(Value::string(format!("\\Q{}\\E", str_arg(&a[0])?))),
        _ => return None,
    })
}

/// Java's replacement syntax (`$1`, `${name}`, `\$`) to `regex`'s.
fn replacement(r: &str) -> Result<String, Throw> {
    let mut out = String::new();
    let mut chars = r.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('$') => out.push_str("$$"),
                Some(o) => out.push(o),
                None => return Err(Throw::illegal_arg("character to be escaped is missing")),
            },
            '$' => match chars.peek() {
                Some('{') => {
                    chars.next();
                    let name: String = chars.by_ref().take_while(|&ch| ch != '}').collect();
                    out.push_str(&format!("${{{name}}}"));
                }
                Some(d) if d.is_ascii_digit() => {
                    let d = chars.next().unwrap_or('0');
                    out.push_str(&format!("${{{d}}}"));
                }
                _ => return Err(Throw::illegal_arg("Illegal group reference")),
            },
            o => out.push(o),
        }
    }
    Ok(out)
}

/// `CharSequence.replaceAll(Pattern, Function<Matcher, String>)` (and
/// `replaceFirst`).
pub fn replace_with(
    vm: &mut Vm,
    recv: &Value,
    pattern: &Value,
    f: &Value,
    all: bool,
) -> Result<Value, Throw> {
    let s = recv.java_to_string_checked()?;
    let p = pattern_of(pattern)?;
    let mut out = String::new();
    let mut last = 0;
    let caps: Vec<_> = p.re.captures_iter(&s).collect();
    for c in caps {
        let m = c.get(0).expect("group 0");
        out.push_str(&s[last..m.start()]);
        let matcher = Value::object(JMatcher::at(pattern.clone(), &s, &c));
        let r = invoke(vm, f, &[matcher])?;
        out.push_str(&r.java_to_string());
        last = m.end();
        if !all {
            break;
        }
    }
    out.push_str(&s[last..]);
    Ok(Value::string(out))
}

/// A `Matcher`: the input and the current match's groups.
#[derive(Debug)]
pub struct JMatcher {
    pattern: Value,
    input: Rc<str>,
    /// Byte position the next `find` starts at.
    pos: Cell<usize>,
    groups: RefCell<Option<Vec<Option<(usize, usize)>>>>,
    names: Vec<Option<String>>,
}

impl JMatcher {
    fn new(pattern: Value, input: &str) -> Self {
        let names = downcast::<JPattern>(&pattern)
            .map(|p| p.re.capture_names().map(|n| n.map(String::from)).collect())
            .unwrap_or_default();
        JMatcher {
            pattern,
            input: Rc::from(input),
            pos: Cell::new(0),
            groups: RefCell::new(None),
            names,
        }
    }

    fn at(pattern: Value, input: &str, c: &regex::Captures) -> Self {
        let m = JMatcher::new(pattern, input);
        *m.groups.borrow_mut() = Some(c.iter().map(|g| g.map(|g| (g.start(), g.end()))).collect());
        m
    }

    fn re(&self) -> &regex::Regex {
        &downcast::<JPattern>(&self.pattern).expect("pattern").re
    }

    fn group(&self, i: usize) -> Result<Value, Throw> {
        let g = self.groups.borrow();
        let Some(g) = g.as_ref() else {
            return Err(Throw::new(
                "java.lang.IllegalStateException",
                "No match found",
            ));
        };
        match g.get(i) {
            Some(Some((a, b))) => Ok(Value::str(&self.input[*a..*b])),
            Some(None) => Ok(Value::Null),
            None => Err(Throw::new(
                "java.lang.IndexOutOfBoundsException",
                format!("No group {i}"),
            )),
        }
    }

    fn utf16_offset(&self, byte: usize) -> i32 {
        self.input[..byte].encode_utf16().count() as i32
    }
}

impl JObject for JMatcher {
    fn class_name(&self) -> &str {
        "java.util.regex.Matcher"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        let r = (|| -> Result<Value, Throw> {
            Ok(match (name, args.len()) {
                ("matches", 0) => {
                    let c = full_match(self.re(), &self.input);
                    let ok = c.is_some();
                    *self.groups.borrow_mut() =
                        c.map(|c| c.iter().map(|g| g.map(|g| (g.start(), g.end()))).collect());
                    Value::Bool(ok)
                }
                ("lookingAt", 0) => {
                    let anchored = regex::Regex::new(&format!("^(?:{})", self.re().as_str()))
                        .map_err(|e| Throw::illegal_arg(e.to_string()))?;
                    let c = anchored.captures(&self.input);
                    let ok = c.is_some();
                    *self.groups.borrow_mut() =
                        c.map(|c| c.iter().map(|g| g.map(|g| (g.start(), g.end()))).collect());
                    Value::Bool(ok)
                }
                ("find", 0) | ("find", 1) => {
                    if args.len() == 1 {
                        let i = int_arg(&args[0])?;
                        let byte = self
                            .input
                            .char_indices()
                            .map(|(b, _)| b)
                            .chain(std::iter::once(self.input.len()))
                            .nth(i.max(0) as usize)
                            .unwrap_or(self.input.len());
                        self.pos.set(byte);
                    }
                    let start = self.pos.get();
                    if start > self.input.len() {
                        *self.groups.borrow_mut() = None;
                        return Ok(Value::Bool(false));
                    }
                    match self.re().captures_at(&self.input, start) {
                        Some(c) => {
                            let m = c.get(0).expect("group 0");
                            let next = if m.end() == m.start() {
                                self.input[m.end()..]
                                    .chars()
                                    .next()
                                    .map_or(self.input.len() + 1, |ch| m.end() + ch.len_utf8())
                            } else {
                                m.end()
                            };
                            self.pos.set(next);
                            *self.groups.borrow_mut() =
                                Some(c.iter().map(|g| g.map(|g| (g.start(), g.end()))).collect());
                            Value::Bool(true)
                        }
                        None => {
                            *self.groups.borrow_mut() = None;
                            Value::Bool(false)
                        }
                    }
                }
                ("group", 0) => self.group(0)?,
                ("group", 1) | ("namedGroup", 1) => match &args[0] {
                    Value::Str(n) => {
                        let i = self
                            .names
                            .iter()
                            .position(|x| x.as_deref() == Some(&**n))
                            .ok_or_else(|| {
                                Throw::illegal_arg(format!("No group with name <{n}>"))
                            })?;
                        self.group(i)?
                    }
                    v => self.group(int_arg(v)? as usize)?,
                },
                ("groupCount", 0) => Value::Int(self.re().captures_len() as i32 - 1),
                ("start", 0) | ("end", 0) | ("start", 1) | ("end", 1) => {
                    let i = match args.first() {
                        Some(Value::Str(n)) => self
                            .names
                            .iter()
                            .position(|x| x.as_deref() == Some(&**n))
                            .unwrap_or(usize::MAX),
                        Some(v) => int_arg(v)? as usize,
                        None => 0,
                    };
                    let g = self.groups.borrow();
                    let g = g.as_ref().ok_or_else(|| {
                        Throw::new("java.lang.IllegalStateException", "No match available")
                    })?;
                    match g.get(i) {
                        Some(Some((a, b))) => {
                            Value::Int(self.utf16_offset(if name == "start" { *a } else { *b }))
                        }
                        Some(None) => Value::Int(-1),
                        None => {
                            return Err(Throw::new(
                                "java.lang.IndexOutOfBoundsException",
                                format!("No group {i}"),
                            ))
                        }
                    }
                }
                ("hitEnd", 0) => Value::Bool(self.pos.get() >= self.input.len()),
                ("replaceAll", 1) | ("replaceFirst", 1) => {
                    let rep = replacement(&str_arg(&args[0])?)?;
                    let s = if name == "replaceAll" {
                        self.re()
                            .replace_all(&self.input, rep.as_str())
                            .into_owned()
                    } else {
                        self.re().replace(&self.input, rep.as_str()).into_owned()
                    };
                    Value::string(s)
                }
                ("reset", 0) => {
                    self.pos.set(0);
                    *self.groups.borrow_mut() = None;
                    Value::Null
                }
                ("pattern", 0) => self.pattern.clone(),
                _ => return Err(Throw::new(api::UNSUPPORTED, "")),
            })
        })();
        match r {
            Err(t) if t.class() == api::UNSUPPORTED && t.message().as_deref() == Some("") => None,
            r => Some(r),
        }
    }
}

impl JObject for JPattern {
    fn class_name(&self) -> &str {
        "java.util.regex.Pattern"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn java_to_string(&self) -> String {
        self.source.clone()
    }

    fn call(
        &self,
        _vm: &mut Vm,
        this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        // A typed call carries the injected regex limit factor first.
        let args = match (name, args) {
            ("matcher" | "split" | "splitAsStream", [Value::Int(_), rest @ ..])
                if !rest.is_empty() && !matches!(rest[0], Value::Int(_)) =>
            {
                rest
            }
            _ => args,
        };
        Some(match (name, args.len()) {
            ("matcher", 1) => args[0]
                .java_to_string_checked()
                .map(|s| Value::object(JMatcher::new(this.clone(), &s))),
            ("pattern", 0) | ("toString", 0) => Ok(Value::str(&self.source)),
            ("flags", 0) => Ok(Value::Int(self.flags)),
            ("split", 1) | ("split", 2) => (|| {
                let s = args[0].java_to_string_checked()?;
                let limit = if args.len() == 2 {
                    int_arg(&args[1])?
                } else {
                    0
                };
                Ok(Value::array(
                    "java.lang.String",
                    split(&self.re, &s, limit)
                        .into_iter()
                        .map(Value::string)
                        .collect(),
                ))
            })(),
            ("asPredicate", 0) => Err(crate::compile::missing("Pattern.asPredicate")),
            _ => return None,
        })
    }
}

/// `Pattern.split`: Java drops a leading empty string for a zero-width
/// match at 0 and, with limit 0, trailing empty strings.
fn split(re: &regex::Regex, s: &str, limit: i32) -> Vec<String> {
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(s) {
        if limit > 0 && out.len() as i32 == limit - 1 {
            break;
        }
        if m.end() == 0 {
            continue;
        }
        if m.start() == m.end() && m.start() == last && last == 0 {
            continue;
        }
        out.push(s[last..m.start()].to_string());
        last = m.end();
    }
    out.push(s[last..].to_string());
    if limit == 0 {
        while out.len() > 1 && out.last().is_some_and(String::is_empty) {
            out.pop();
        }
        if out.len() == 1 && out[0].is_empty() && !s.is_empty() {
            out.clear();
        }
    }
    out
}
