//! `${prefix.name}` placeholders replaced as Apache Commons Text's
//! `StringSubstitutor` replaces them, which ml-commons uses for every
//! connector template: `${parameters.x}`, `${credential.x}`.
//!
//! A placeholder ends at the first `}`; `name:-default` gives the text used
//! when `name` has no value; a value is itself substituted, and a value that
//! leads back to a name being substituted is an error; `$` before a
//! placeholder keeps it literal (the `$` is dropped); a placeholder with no
//! value and no default stays as written.

use crate::{MlError, Result};

/// The names a template reads its values from.
pub trait Lookup {
    /// The value of `name`, or `None` when it has none.
    fn lookup(&self, name: &str) -> Option<&str>;
}

impl<S: AsRef<str>> Lookup for Vec<(S, Option<S>)> {
    fn lookup(&self, name: &str) -> Option<&str> {
        self.iter()
            .rev()
            .find(|(k, _)| k.as_ref() == name)
            .and_then(|(_, v)| v.as_ref().map(AsRef::as_ref))
    }
}

impl Lookup for std::collections::HashMap<String, Option<String>> {
    fn lookup(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(|v| v.as_deref())
    }
}

impl Lookup for std::collections::HashMap<String, String> {
    fn lookup(&self, name: &str) -> Option<&str> {
        self.get(name).map(String::as_str)
    }
}

const ESCAPE: char = '$';
const SUFFIX: char = '}';
const DEFAULT: &str = ":-";

/// `text` with every `prefix` placeholder (`prefix` is `"${parameters."`)
/// replaced from `values`.
pub fn substitute(text: &str, prefix: &str, values: &dyn Lookup) -> Result<String> {
    let mut buf: Vec<char> = text.chars().collect();
    let prefix: Vec<char> = prefix.chars().collect();
    let mut prior = vec![text.to_string()];
    let len = buf.len();
    run(&mut buf, 0, len, &prefix, values, &mut prior)?;
    Ok(buf.into_iter().collect())
}

/// Substitutes `buf[offset..offset + length]` in place; the change in length.
// ARITH: every position is within `buf`, an in-memory buffer whose length
// fits an isize, and `delta` is the change in that length.
#[allow(clippy::arithmetic_side_effects)]
fn run(
    buf: &mut Vec<char>,
    offset: usize,
    length: usize,
    prefix: &[char],
    values: &dyn Lookup,
    prior: &mut Vec<String>,
) -> Result<isize> {
    let mut end = offset + length;
    let mut pos = offset;
    let mut change: isize = 0;
    while pos < end {
        if !matches_at(buf, pos, end, prefix) {
            pos += 1;
            continue;
        }
        if pos > offset && buf[pos - 1] == ESCAPE {
            buf.remove(pos - 1);
            change -= 1;
            end -= 1;
            continue;
        }
        let start = pos;
        let name_start = pos + prefix.len();
        let Some(close) = (name_start..end).find(|&i| buf[i] == SUFFIX) else {
            // No closing brace: nothing more can be a placeholder.
            break;
        };
        let expr: String = buf[name_start..close].iter().collect();
        let stop = close + 1;
        let (name, default) = match expr.find(DEFAULT) {
            Some(i) => (
                expr[..i].to_string(),
                Some(expr[i + DEFAULT.len()..].to_string()),
            ),
            None => (expr.clone(), None),
        };
        if prior[1..].contains(&name) {
            let mut chain: Vec<&str> = prior[1..].iter().map(String::as_str).collect();
            chain.push(&name);
            return Err(MlError::Model(format!(
                "Infinite loop in property interpolation of {}: {}",
                prior[0],
                chain.join("->")
            )));
        }
        prior.push(name.clone());
        let value = values.lookup(&name).map(str::to_string).or(default);
        match value {
            Some(value) => {
                let value: Vec<char> = value.chars().collect();
                let n = value.len();
                buf.splice(start..stop, value);
                let inner = run(buf, start, n, prefix, values, prior)?;
                let delta = inner + n as isize - (stop - start) as isize;
                pos = (stop as isize + delta) as usize;
                end = (end as isize + delta) as usize;
                change += delta;
            }
            None => pos = stop,
        }
        prior.pop();
    }
    Ok(change)
}

// ARITH: `pos` and `prefix.len()` are within in-memory buffers.
#[allow(clippy::arithmetic_side_effects)]
fn matches_at(buf: &[char], pos: usize, end: usize, prefix: &[char]) -> bool {
    pos + prefix.len() <= end && buf[pos..pos + prefix.len()] == *prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: &str = "${parameters.";

    fn sub(text: &str, values: &[(&str, Option<&str>)]) -> Result<String> {
        substitute(text, P, &values.to_vec())
    }

    #[test]
    fn replaces_as_string_substitutor() {
        let v = [
            ("a", Some("A")),
            ("b", Some("${parameters.a}!")),
            ("n", None),
        ];
        assert_eq!(sub("x${parameters.a}y", &v).unwrap(), "xAy");
        assert_eq!(sub("${parameters.b}${parameters.b}", &v).unwrap(), "A!A!");
        assert_eq!(sub("${parameters.c}", &v).unwrap(), "${parameters.c}");
        assert_eq!(
            sub("${parameters.c:-d}|${parameters.n:-null}", &v).unwrap(),
            "d|null"
        );
        assert_eq!(sub("${parameters.a:-d}", &v).unwrap(), "A");
        assert_eq!(
            sub("$${parameters.a} ${parameters.a}", &v).unwrap(),
            "${parameters.a} A"
        );
        assert_eq!(sub("${parameters.a", &v).unwrap(), "${parameters.a");
        assert_eq!(
            sub("${other.a} ${parameters.a}", &v).unwrap(),
            "${other.a} A"
        );
        assert_eq!(sub("é${parameters.a}é", &v).unwrap(), "éAé");
    }

    #[test]
    fn refuses_a_loop() {
        let v = [
            ("a", Some("${parameters.b}")),
            ("b", Some("x${parameters.a}")),
        ];
        let e = sub("<${parameters.a}>", &v).unwrap_err().to_string();
        assert!(
            e.contains("Infinite loop in property interpolation of <${parameters.a}>: a->b->a"),
            "{e}"
        );
        let mut m = std::collections::HashMap::new();
        m.insert("k".to_string(), "v".to_string());
        assert_eq!(
            substitute("${credential.k}", "${credential.", &m).unwrap(),
            "v"
        );
        let mut m: std::collections::HashMap<String, Option<String>> = Default::default();
        m.insert("k".into(), None);
        assert_eq!(substitute("${p.k}", "${p.", &m).unwrap(), "${p.k}");
    }
}
