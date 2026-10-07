//! The configuration text of `fixtures/corpus/analysis-factories.conf`
//! (`GenAnalysisFactories.java` is the Java twin): parses a line into
//! builder steps, builds the `CustomAnalyzer`, and writes the fixture's rows.
//! Shared with `lucene-search`'s word2vec factory test.
#![allow(dead_code)]

use std::path::PathBuf;

use lucene_analysis::factory::{CustomAnalyzer, CustomAnalyzerBuilder, FactoryError};
use lucene_util::version::Version;

/// One builder step and its arguments.
pub struct Step {
    pub kind: String,
    pub name: String,
    pub params: Vec<String>,
}

/// `GenAnalysisFactories.unesc`.
pub fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap());
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

/// `GenAnalysisFactories.parse`.
pub fn parse(fields: &[&str]) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::new();
    for f in &fields[1..] {
        let colon = f.find(':');
        let eq = f.find('=');
        if *f == "endwhen" {
            steps.push(Step {
                kind: "endwhen".into(),
                name: String::new(),
                params: Vec::new(),
            });
        } else if colon.is_some_and(|c| c > 0 && eq.is_none_or(|e| c < e)) {
            let c = colon.unwrap();
            steps.push(Step {
                kind: f[..c].to_string(),
                name: unesc(&f[c + 1..]),
                params: Vec::new(),
            });
        } else {
            let e = eq.expect("a key=value argument");
            let last = steps.last_mut().expect("an argument after a step");
            last.params.push(f[..e].to_string());
            last.params.push(unesc(&f[e + 1..]));
        }
    }
    steps
}

/// The resource directory the configurations name files in.
pub fn resource_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-factories"
    ))
}

/// `GenAnalysisFactories.build`.
pub fn build(steps: &[Step]) -> Result<CustomAnalyzer, FactoryError> {
    let mut b: Option<CustomAnalyzerBuilder> =
        Some(CustomAnalyzer::builder_with_dir(resource_dir())?);
    let mut cb = None;
    for s in steps {
        let p: Vec<&str> = s.params.iter().map(String::as_str).collect();
        match s.kind.as_str() {
            "tok" => b = Some(b.take().unwrap().with_tokenizer(&s.name, &p)?),
            "cf" => b = Some(b.take().unwrap().add_char_filter(&s.name, &p)?),
            "tf" => match cb.take() {
                Some(c) => {
                    cb = Some(
                        lucene_analysis::factory::ConditionBuilder::add_token_filter(
                            c, &s.name, &p,
                        )?,
                    )
                }
                None => b = Some(b.take().unwrap().add_token_filter(&s.name, &p)?),
            },
            "when" => cb = Some(b.take().unwrap().when(&s.name, &p)?),
            "whenTerm" => {
                let len: usize = s.name.parse().unwrap();
                cb = Some(
                    b.take()
                        .unwrap()
                        .when_term(move |t| t.encode_utf16().count() > len),
                );
            }
            "endwhen" => b = Some(cb.take().unwrap().endwhen()?),
            "version" => {
                b = Some(
                    b.take()
                        .unwrap()
                        .with_default_match_version(Version::parse(&s.name).unwrap())?,
                )
            }
            "posgap" => {
                b = Some(
                    b.take()
                        .unwrap()
                        .with_position_increment_gap(s.name.parse().unwrap())?,
                )
            }
            "offgap" => b = Some(b.take().unwrap().with_offset_gap(s.name.parse().unwrap())?),
            other => panic!("unknown step {other}"),
        }
    }
    b.unwrap().build()
}

/// `GenAnalysisFactories.stable`, plus the one message the port cannot
/// reproduce: `java.util.regex`'s multi-line `PatternSyntaxException` text
/// (the generator drops a `StringIndexOutOfBoundsException`'s, the JDK's).
pub fn stable(class: &str, message: &str) -> String {
    if let Some(i) = message.find("The current classpath supports the following names: ") {
        return message[..i].to_string();
    }
    if message.starts_with("Unable to load hunspell data!") {
        return "Unable to load hunspell data!".to_string();
    }
    if class == "PatternSyntaxException" || class == "StringIndexOutOfBoundsException" {
        return String::new();
    }
    message.to_string()
}

/// The configurations: `(name, fields)`.
pub fn configs() -> Vec<(String, Vec<String>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-factories.conf"
    );
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let fields: Vec<String> = l.split('\t').map(str::to_string).collect();
            (fields[0].clone(), fields)
        })
        .collect()
}
