//! `org.apache.lucene.expressions.js.VariableContext`: a variable name split
//! into the member accesses, indexes and method calls it is made of.

/// `VariableContext.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    /// `.name`, or the base name.
    Member,
    /// `['key']`.
    StrIndex,
    /// `[3]`.
    IntIndex,
    /// `.name()`.
    Method,
}

/// `VariableContext`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableContext {
    /// `type`.
    pub kind: Type,
    /// `text`: the member, method or key (`None` for an integer index).
    pub text: Option<String>,
    /// `integer`: the index (`-1` otherwise).
    pub integer: i32,
}

/// What `VariableContext.parse` throws: a bad index, or a name that runs out
/// (`ArrayIndexOutOfBounds`/`NumberFormat` in Java).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableError(pub String);

impl std::fmt::Display for VariableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VariableError {}

fn at(text: &[u16], i: usize) -> Result<u16, VariableError> {
    text.get(i)
        .copied()
        .ok_or_else(|| VariableError(format!("Index {i} out of bounds for length {}", text.len())))
}

fn string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// `VariableContext.parse(variable)`.
///
/// # Errors
/// Where Java throws: an unterminated index, or an integer index that is not
/// an `int`.
pub fn parse(variable: &str) -> Result<Vec<VariableContext>, VariableError> {
    let text: Vec<u16> = variable.encode_utf16().collect();
    let mut contexts = Vec::new();
    let mut i = add_member(&text, 0, &mut contexts)?;
    while i < text.len() {
        if text[i] == u16::from(b'[') {
            i += 1;
            if at(&text, i)? == u16::from(b'\'') {
                i = add_string_index(&text, i, &mut contexts)?;
            } else {
                i = add_int_index(&text, i, &mut contexts)?;
            }
            i += 1;
        } else {
            i = add_member(&text, i + 1, &mut contexts)?;
        }
    }
    Ok(contexts)
}

fn add_member(
    text: &[u16],
    i: usize,
    contexts: &mut Vec<VariableContext>,
) -> Result<usize, VariableError> {
    let mut j = i + 1;
    while j < text.len()
        && text[j] != u16::from(b'[')
        && text[j] != u16::from(b'.')
        && text[j] != u16::from(b'(')
    {
        j += 1;
    }
    // `new String(text, i, j - i)` past the end of `text`
    at(text, j - 1)?;
    let name = string(&text[i..j]);
    if j + 1 < text.len() && text[j] == u16::from(b'(') && text[j + 1] == u16::from(b')') {
        contexts.push(VariableContext {
            kind: Type::Method,
            text: Some(name),
            integer: -1,
        });
        j += 2;
    } else {
        contexts.push(VariableContext {
            kind: Type::Member,
            text: Some(name),
            integer: -1,
        });
    }
    Ok(j)
}

fn add_string_index(
    text: &[u16],
    mut i: usize,
    contexts: &mut Vec<VariableContext>,
) -> Result<usize, VariableError> {
    i += 1;
    let mut j = i;
    while at(text, j)? != u16::from(b'\'') {
        if text[j] == u16::from(b'\\') {
            j += 1;
        }
        j += 1;
    }
    let mut buf = Vec::with_capacity(j - i);
    while i < j {
        if text[i] == u16::from(b'\\') {
            i += 1;
        }
        buf.push(at(text, i)?);
        i += 1;
    }
    contexts.push(VariableContext {
        kind: Type::StrIndex,
        text: Some(string(&buf)),
        integer: -1,
    });
    Ok(j + 1)
}

fn add_int_index(
    text: &[u16],
    i: usize,
    contexts: &mut Vec<VariableContext>,
) -> Result<usize, VariableError> {
    let mut j = i + 1;
    while at(text, j)? != u16::from(b']') {
        j += 1;
    }
    let digits = string(&text[i..j]);
    let index = parse_java_int(&digits)?;
    contexts.push(VariableContext {
        kind: Type::IntIndex,
        text: None,
        integer: index,
    });
    Ok(j)
}

/// `Integer.parseInt(s)`: an optional sign and ASCII digits only.
fn parse_java_int(s: &str) -> Result<i32, VariableError> {
    let body = s.strip_prefix(['-', '+']).unwrap_or(s);
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
        return Err(VariableError(format!("For input string: \"{s}\"")));
    }
    s.strip_prefix('+')
        .unwrap_or(s)
        .parse::<i32>()
        .map_err(|_| VariableError(format!("For input string: \"{s}\"")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(v: &str) -> Vec<(Type, Option<String>, i32)> {
        parse(v)
            .unwrap()
            .into_iter()
            .map(|c| (c.kind, c.text, c.integer))
            .collect()
    }

    #[test]
    fn members_indexes_and_methods() {
        assert_eq!(kinds("a"), vec![(Type::Member, Some("a".into()), -1)]);
        assert_eq!(
            kinds("a.b[2]['k'].m()"),
            vec![
                (Type::Member, Some("a".into()), -1),
                (Type::Member, Some("b".into()), -1),
                (Type::IntIndex, None, 2),
                (Type::StrIndex, Some("k".into()), -1),
                (Type::Method, Some("m".into()), -1),
            ]
        );
    }

    #[test]
    fn string_index_unescapes() {
        assert_eq!(
            kinds(r"s['it\'s']"),
            vec![
                (Type::Member, Some("s".into()), -1),
                (Type::StrIndex, Some("it's".into()), -1),
            ]
        );
    }

    #[test]
    fn malformed_names_error_where_java_throws() {
        assert!(parse("x['abc").is_err());
        assert!(parse("x[1").is_err());
        assert!(parse("x[2147483648]").is_err());
        assert!(parse("x[-1]").is_ok());
        assert!(parse("x[1.0]").is_err());
        assert!(parse("x[").is_err());
    }
}
