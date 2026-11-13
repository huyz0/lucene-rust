//! JSON paths as Jayway JsonPath reads them, which ml-commons applies to a
//! remote model's response (`response_filter`, the built-in post-process
//! functions): `$`, `.name` / `['name']`, `[n]` (negative from the end),
//! `[*]` / `.*`, `[a,b]`, `[start:end]` and `..name`.
//!
//! A path with no wildcard, deep scan, slice or index list is definite and
//! reads one value; a missing step is an error. Any other path reads the
//! list of every match, leaving out the branches that do not match.

use super::json::Json;

/// Why a path read nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    /// The path is well formed but the document has nothing there
    /// (Jayway's `PathNotFoundException`).
    NotFound(String),
    /// The path cannot be compiled, or uses a construct (a filter) this
    /// reader does not implement.
    Invalid(String),
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::NotFound(m) | PathError::Invalid(m) => f.write_str(m),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Step {
    Prop(String),
    Index(i64),
    Indexes(Vec<i64>),
    Slice(Option<i64>, Option<i64>),
    Wildcard,
    Deep(Box<Step>),
}

/// A compiled path.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonPath {
    steps: Vec<Step>,
}

impl JsonPath {
    /// Compiles `path`.
    // ARITH: positions within the path's characters, two past at most.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compile(path: &str) -> Result<JsonPath, PathError> {
        let invalid = |why: &str| PathError::Invalid(format!("Invalid path {path}: {why}"));
        let s: Vec<char> = path.trim().chars().collect();
        if s.first() != Some(&'$') {
            return Err(invalid("a path must start with $"));
        }
        let mut i = 1;
        let mut steps = Vec::new();
        while i < s.len() {
            match s[i] {
                '.' if s.get(i + 1) == Some(&'.') => {
                    i += 2;
                    let step = if s.get(i) == Some(&'[') {
                        let (step, next) = bracket(&s, i).ok_or_else(|| invalid("bad brackets"))?;
                        i = next;
                        step
                    } else {
                        let (name, next) = name(&s, i);
                        i = next;
                        if name == "*" {
                            Step::Wildcard
                        } else if name.is_empty() {
                            return Err(invalid("empty property after .."));
                        } else {
                            Step::Prop(name)
                        }
                    };
                    steps.push(Step::Deep(Box::new(step)));
                }
                '.' => {
                    let (name, next) = name(&s, i + 1);
                    i = next;
                    match name.as_str() {
                        "" => return Err(invalid("empty property")),
                        "*" => steps.push(Step::Wildcard),
                        _ => steps.push(Step::Prop(name)),
                    }
                }
                '[' => {
                    let (step, next) = bracket(&s, i).ok_or_else(|| invalid("bad brackets"))?;
                    i = next;
                    steps.push(step);
                }
                _ => return Err(invalid("unexpected character")),
            }
        }
        Ok(JsonPath { steps })
    }

    /// Whether the path reads one value.
    pub fn is_definite(&self) -> bool {
        self.steps
            .iter()
            .all(|s| matches!(s, Step::Prop(_) | Step::Index(_)))
    }

    /// Reads the path from `doc`: the value for a definite path, the array
    /// of matches otherwise.
    pub fn read(&self, doc: &Json) -> Result<Json, PathError> {
        if self.is_definite() {
            let mut cur = doc;
            let mut at = String::from("$");
            for step in &self.steps {
                cur = match (step, cur) {
                    (Step::Prop(p), Json::Object(_)) => cur.get(p).ok_or_else(|| {
                        PathError::NotFound(format!("No results for path: {at}['{p}']"))
                    })?,
                    (Step::Prop(p), _) => {
                        return Err(PathError::NotFound(format!(
                            "Expected to find an object with property ['{p}'] in path {at} but found '{}'. This is not a json object according to the JsonProvider: 'com.jayway.jsonpath.spi.json.JsonSmartJsonProvider'.",
                            kind(cur)
                        )))
                    }
                    (Step::Index(n), Json::Array(a)) => index(a, *n).ok_or_else(|| {
                        PathError::NotFound(format!("Index out of bounds when evaluating path {at}[{n}]"))
                    })?,
                    (Step::Index(n), _) => {
                        return Err(PathError::NotFound(format!(
                            "Filter: [{n}] can only be applied to arrays. Current context is: {}",
                            cur.to_compact()
                        )))
                    }
                    _ => unreachable!("definite paths hold properties and indexes"),
                };
                match step {
                    Step::Prop(p) => at.push_str(&format!("['{p}']")),
                    Step::Index(n) => at.push_str(&format!("[{n}]")),
                    _ => {}
                }
            }
            return Ok(cur.clone());
        }
        let mut current = vec![doc];
        for step in &self.steps {
            let mut next = Vec::new();
            for v in current {
                apply(step, v, &mut next);
            }
            current = next;
        }
        Ok(Json::Array(current.into_iter().cloned().collect()))
    }
}

fn kind(v: &Json) -> &'static str {
    match v {
        Json::Array(_) => "net.minidev.json.JSONArray",
        Json::String(_) => "java.lang.String",
        Json::Number(_) => "java.lang.Number",
        Json::Bool(_) => "java.lang.Boolean",
        Json::Null => "null",
        Json::Object(_) => "net.minidev.json.JSONObject",
    }
}

// ARITH: `n` is negative when added, and a slice's length fits an i64.
#[allow(clippy::arithmetic_side_effects)]
fn index(a: &[Json], n: i64) -> Option<&Json> {
    let i = if n < 0 { a.len() as i64 + n } else { n };
    usize::try_from(i).ok().and_then(|i| a.get(i))
}

// ARITH: a slice bound is added to the length only when negative.
#[allow(clippy::arithmetic_side_effects)]
fn apply<'a>(step: &Step, v: &'a Json, out: &mut Vec<&'a Json>) {
    match (step, v) {
        (Step::Prop(p), Json::Object(_)) => out.extend(v.get(p)),
        (Step::Index(n), Json::Array(a)) => out.extend(index(a, *n)),
        (Step::Indexes(ns), Json::Array(a)) => out.extend(ns.iter().filter_map(|n| index(a, *n))),
        (Step::Slice(from, to), Json::Array(a)) => {
            let len = a.len() as i64;
            let norm = |x: i64| if x < 0 { (len + x).max(0) } else { x.min(len) };
            let from = norm(from.unwrap_or(0));
            let to = norm(to.unwrap_or(len));
            if from < to {
                out.extend(&a[from as usize..to as usize]);
            }
        }
        (Step::Wildcard, Json::Array(a)) => out.extend(a.iter()),
        (Step::Wildcard, Json::Object(m)) => out.extend(m.iter().map(|(_, v)| v)),
        (Step::Deep(inner), _) => deep(inner, v, out),
        _ => {}
    }
}

fn deep<'a>(inner: &Step, v: &'a Json, out: &mut Vec<&'a Json>) {
    apply(inner, v, out);
    match v {
        Json::Array(a) => a.iter().for_each(|c| deep(inner, c, out)),
        Json::Object(m) => m.iter().for_each(|(_, c)| deep(inner, c, out)),
        _ => {}
    }
}

/// A property name after `.`: up to the next `.` or `[`.
// ARITH: positions within the path's characters, one past at most.
#[allow(clippy::arithmetic_side_effects)]
fn name(s: &[char], from: usize) -> (String, usize) {
    let mut i = from;
    while i < s.len() && s[i] != '.' && s[i] != '[' {
        i += 1;
    }
    (s[from..i].iter().collect(), i)
}

/// The step in `[...]` starting at `s[from]`, and the index after `]`.
// ARITH: positions within the path's characters, one or two past at most.
#[allow(clippy::arithmetic_side_effects)]
fn bracket(s: &[char], from: usize) -> Option<(Step, usize)> {
    let mut i = from + 1;
    if matches!(s.get(i), Some('\'' | '"')) {
        let q = s[i];
        let mut names = Vec::new();
        loop {
            if s.get(i) != Some(&q) {
                return None;
            }
            i += 1;
            let start = i;
            while s.get(i)? != &q {
                i += if s[i] == '\\' { 2 } else { 1 };
            }
            names.push(s[start..i].iter().collect::<String>().replace("\\'", "'"));
            i += 1;
            while s.get(i) == Some(&' ') {
                i += 1;
            }
            match s.get(i)? {
                ']' => break,
                ',' => {
                    i += 1;
                    while s.get(i) == Some(&' ') {
                        i += 1;
                    }
                }
                _ => return None,
            }
        }
        // Several names read a map of them, which ml-commons never asks for.
        return (names.len() == 1).then(|| (Step::Prop(names.remove(0)), i + 1));
    }
    let close = (i..s.len()).find(|&j| s[j] == ']')?;
    let body: String = s[i..close].iter().collect::<String>().replace(' ', "");
    i = close + 1;
    if body == "*" {
        return Some((Step::Wildcard, i));
    }
    if body.starts_with('?') {
        return None;
    }
    if let Some((a, b)) = body.split_once(':') {
        let num = |t: &str| -> Option<Option<i64>> {
            if t.is_empty() {
                Some(None)
            } else {
                t.parse().ok().map(Some)
            }
        };
        return Some((Step::Slice(num(a)?, num(b)?), i));
    }
    let ns = body
        .split(',')
        .map(|t| t.parse::<i64>().ok())
        .collect::<Option<Vec<_>>>()?;
    Some((
        if ns.len() == 1 {
            Step::Index(ns[0])
        } else {
            Step::Indexes(ns)
        },
        i,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(path: &str, doc: &str) -> Result<String, PathError> {
        JsonPath::compile(path)?
            .read(&Json::parse(doc).unwrap())
            .map(|v| v.to_compact())
    }

    const DOC: &str = r#"{"data":[{"embedding":[0.1,2]},{"embedding":[3e-5]},{"x":1}],
        "embedding":[0.5],"results":[{"index":1,"relevance_score":0.25}],"o":{"a":{"b":"c"},"b":2}}"#;

    #[test]
    fn reads_definite_paths() {
        assert_eq!(read("$.embedding", DOC).unwrap(), "[0.5]");
        assert_eq!(read("$['embedding'][0]", DOC).unwrap(), "0.5");
        assert_eq!(read("$.data[-1].x", DOC).unwrap(), "1");
        assert_eq!(read("$[\"o\"].a.b", DOC).unwrap(), "\"c\"");
        assert_eq!(read("$", "[1]").unwrap(), "[1]");
        assert!(JsonPath::compile("$.a[0]").unwrap().is_definite());
        assert!(
            matches!(read("$.nope", DOC), Err(PathError::NotFound(m)) if m == "No results for path: $['nope']")
        );
        assert!(
            matches!(read("$.data.embedding", DOC), Err(PathError::NotFound(m)) if m.contains("JSONArray"))
        );
        assert!(matches!(
            read("$.data[9]", DOC),
            Err(PathError::NotFound(_))
        ));
        assert!(matches!(read("$.o[0]", DOC), Err(PathError::NotFound(_))));
    }

    #[test]
    fn reads_indefinite_paths() {
        assert_eq!(
            read("$.data[*].embedding", DOC).unwrap(),
            "[[0.1,2],[3e-5]]"
        );
        assert_eq!(read("$..b", DOC).unwrap(), "[2,\"c\"]");
        assert_eq!(
            read("$.data[0,2]", DOC).unwrap(),
            r#"[{"embedding":[0.1,2]},{"x":1}]"#
        );
        assert_eq!(
            read("$.data[1:]", DOC).unwrap(),
            r#"[{"embedding":[3e-5]},{"x":1}]"#
        );
        assert_eq!(read("$.data[:-2].embedding", DOC).unwrap(), "[[0.1,2]]");
        assert_eq!(read("$.o.*", DOC).unwrap(), r#"[{"b":"c"},2]"#);
        assert_eq!(read("$[*]", "[1,[2]]").unwrap(), "[1,[2]]");
        assert_eq!(read("$..[0]", "[[1],[2]]").unwrap(), "[[1],1,2]");
        assert_eq!(read("$..*", "[[1]]").unwrap(), "[[1],1]");
        assert_eq!(read("$.data[5:1]", DOC).unwrap(), "[]");
    }

    #[test]
    fn refuses_bad_paths() {
        for bad in [
            "data",
            "$..",
            "$.",
            "$[?(@.a)]",
            "$[",
            "$['a','b']",
            "$x",
            "$[a]",
            "$['a]",
        ] {
            assert!(
                matches!(JsonPath::compile(bad), Err(PathError::Invalid(_))),
                "{bad}"
            );
        }
        assert_eq!(PathError::Invalid("x".into()).to_string(), "x");
    }
}
