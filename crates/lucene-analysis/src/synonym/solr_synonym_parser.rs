//! `org.apache.lucene.analysis.synonym.SolrSynonymParser`: the Solr rule
//! format.
//!
//! - Blank lines and lines starting with `#` are skipped.
//! - `a, b => c, d` maps every left phrase to every right one, dropping the
//!   original (`expand` is ignored).
//! - `a, b, c` is an equivalence: with `expand`, every phrase maps to every
//!   other one, keeping the original; without, every phrase (the first too)
//!   maps to the first, dropping the original.
//! - `\` escapes the next char (a `,`, `=>` or `\` inside a phrase).
//!
//! Differs: [`SolrSynonymParser::parse`] takes the rules as a `&str` rather
//! than a `Reader`.

use crate::analyzer::Analyzer;
use crate::AnalysisError;

use super::synonym_map::{java_lines, SynonymMap, SynonymParseError, SynonymParserBase};

/// `SolrSynonymParser`.
pub struct SolrSynonymParser<'a> {
    base: SynonymParserBase<'a>,
    expand: bool,
}

impl<'a> SolrSynonymParser<'a> {
    /// `new SolrSynonymParser(dedup, expand, analyzer)`.
    pub fn new(dedup: bool, expand: bool, analyzer: &'a Analyzer) -> Self {
        SolrSynonymParser {
            base: SynonymParserBase::new(dedup, analyzer),
            expand,
        }
    }

    /// `parse(Reader)`: adds every rule of `rules`; the first bad rule is a
    /// [`SynonymParseError::InvalidRule`] naming its line.
    pub fn parse(&mut self, rules: &str) -> Result<(), SynonymParseError> {
        for (i, line) in java_lines(rules).into_iter().enumerate() {
            self.add_line(line)
                .map_err(|cause| SynonymParseError::InvalidRule { line: i + 1, cause })?;
        }
        Ok(())
    }

    /// `build()` (see [`SynonymMapBuilder::build`](super::SynonymMapBuilder::build)).
    pub fn build(self) -> Result<SynonymMap, crate::AnalysisError> {
        self.base.builder.build()
    }

    fn analyze_all(&self, phrases: &[String]) -> Result<Vec<String>, AnalysisError> {
        phrases
            .iter()
            .map(|p| self.base.analyze(java_trim(&unescape(p))))
            .collect()
    }

    // Java: SolrSynonymParser.addInternal, one line
    fn add_line(&mut self, line: &str) -> Result<(), AnalysisError> {
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }
        let sides = split(line, "=>");
        if sides.len() > 1 {
            if sides.len() != 2 {
                return Err(AnalysisError::IllegalArgument(
                    "more than one explicit mapping specified on the same line".to_string(),
                ));
            }
            let inputs = self.analyze_all(&split(&sides[0], ","))?;
            let outputs = self.analyze_all(&split(&sides[1], ","))?;
            for input in &inputs {
                for output in &outputs {
                    self.base.builder.add(input, output, false)?;
                }
            }
        } else {
            let inputs = self.analyze_all(&split(line, ","))?;
            if self.expand {
                for (i, a) in inputs.iter().enumerate() {
                    for (j, b) in inputs.iter().enumerate() {
                        if i != j {
                            self.base.builder.add(a, b, true)?;
                        }
                    }
                }
            } else {
                for input in &inputs {
                    self.base.builder.add(input, &inputs[0], false)?;
                }
            }
        }
        Ok(())
    }
}

/// `SolrSynonymParser.split`: splits at `separator`, keeping escapes (a `\`
/// and the char after it) and dropping empty pieces.
fn split(s: &str, separator: &str) -> Vec<String> {
    let mut list = Vec::new();
    let mut sb = String::new();
    let mut rest = s;
    while let Some(ch) = rest.chars().next() {
        if rest.starts_with(separator) {
            if !sb.is_empty() {
                list.push(std::mem::take(&mut sb));
            }
            rest = &rest[separator.len()..];
            continue;
        }
        rest = &rest[ch.len_utf8()..];
        sb.push(ch);
        if ch == '\\' {
            match rest.chars().next() {
                None => break,
                Some(next) => {
                    rest = &rest[next.len_utf8()..];
                    sb.push(next);
                }
            }
        }
    }
    if !sb.is_empty() {
        list.push(sb);
    }
    list
}

/// `SolrSynonymParser.unescape`: a `\` before any char but the last is
/// dropped.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(ch) = it.next() {
        if ch == '\\' {
            if let Some(next) = it.next() {
                out.push(next);
                continue;
            }
        }
        out.push(ch);
    }
    out
}

/// `String.trim()`: strips chars up to U+0020 from both ends.
pub(crate) fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_keeps_escapes_and_drops_empty_pieces() {
        assert_eq!(split("a,,b", ","), vec!["a", "b"]);
        assert_eq!(split("a\\,b,c", ","), vec!["a\\,b", "c"]);
        assert_eq!(split("a=>b=>c", "=>"), vec!["a", "b", "c"]);
        assert_eq!(split("a\\", ","), vec!["a\\"]);
        assert_eq!(split("", ","), Vec::<String>::new());
    }

    #[test]
    fn unescape_and_trim_follow_java() {
        assert_eq!(unescape("a\\,b\\\\c\\"), "a,b\\c\\");
        assert_eq!(unescape("plain"), "plain");
        assert_eq!(java_trim("\u{1} a \t"), "a");
    }

    #[test]
    fn rules_build_the_map_java_builds() {
        let ws = Analyzer::new(crate::core_analysis::WhitespaceAnalyzer::default());
        let mut p = SolrSynonymParser::new(true, true, &ws);
        p.parse("# c\n\nfoo, bar baz\nx => y, z\n").unwrap();
        let m = p.build().unwrap();
        let e: Vec<_> = m
            .entries()
            .into_iter()
            .map(|(k, e)| {
                (
                    k,
                    e.keep_orig,
                    e.ords
                        .iter()
                        .map(|&o| m.word(o).to_string())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            e,
            vec![
                ("bar\0baz".to_string(), true, vec!["foo".to_string()]),
                ("foo".to_string(), true, vec!["bar\0baz".to_string()]),
                (
                    "x".to_string(),
                    false,
                    vec!["y".to_string(), "z".to_string()]
                ),
            ]
        );

        let mut p = SolrSynonymParser::new(true, false, &ws);
        p.parse("a, b").unwrap();
        let m = p.build().unwrap();
        assert!(m
            .entries()
            .iter()
            .all(|(_, e)| !e.keep_orig && m.word(e.ords[0]) == "a"));
    }

    #[test]
    fn bad_rules_name_their_line() {
        let ws = Analyzer::new(crate::core_analysis::WhitespaceAnalyzer::default());
        let mut p = SolrSynonymParser::new(true, true, &ws);
        match p.parse("a, b\na => b => c") {
            Err(SynonymParseError::InvalidRule { line: 2, cause }) => {
                assert!(cause.to_string().contains("more than one"))
            }
            other => panic!("{other:?}"),
        }
        let mut p = SolrSynonymParser::new(true, true, &ws);
        assert!(matches!(
            p.parse("a, \\ "),
            Err(SynonymParseError::InvalidRule { line: 1, .. })
        ));
    }
}
