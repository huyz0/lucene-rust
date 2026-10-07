//! The analysis harness's chain builders and row writer, shared by the
//! fixture tests of the generators written after `GenAnalysisCommon`
//! (`fixtures/src/AnalysisRows.java` is the Java twin). Rows are
//! `GenAnalysisCommon`'s; see `analysis_common_fixtures.rs` for the format
//! and the lone-surrogate normalisation.
#![allow(dead_code)]

use std::collections::BTreeSet;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::reader::CharReader;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, TokenStream, TokenStreamComponents,
};

pub type Sink = Result<TokenStreamComponents, AnalysisError>;
type ReaderWrap = Box<dyn Fn(Box<dyn CharReader>) -> Box<dyn CharReader> + Send + Sync>;

struct Chain {
    components: Box<dyn Fn() -> Sink + Send + Sync>,
    char_filters: Option<ReaderWrap>,
}

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.components)()
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        match &self.char_filters {
            Some(wrap) => wrap(reader),
            None => reader,
        }
    }
}

/// `AnalysisRows.chain(tokenizer, filters)`.
pub fn chain(f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain {
        components: Box::new(f),
        char_filters: None,
    })
}

/// [`chain`] with an `initReader` char filter.
pub fn chain_cf(
    cf: impl Fn(Box<dyn CharReader>) -> Box<dyn CharReader> + Send + Sync + 'static,
    f: impl Fn() -> Sink + Send + Sync + 'static,
) -> Analyzer {
    Analyzer::new(Chain {
        components: Box::new(f),
        char_filters: Some(Box::new(cf)),
    })
}

pub fn comps(sink: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(sink))
}

pub fn data_dir(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}/", env!("CARGO_MANIFEST_DIR"))
}

/// `AnalysisRows.corpus`: a corpus file's lines, split on '\n' only.
pub fn corpus(name: &str) -> Vec<String> {
    let path = format!(
        "{}/../../fixtures/corpus/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

/// `AnalysisRows.esc`.
pub fn esc(s: &str) -> String {
    let mut b = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => b.push_str("\\\\"),
            '\t' => b.push_str("\\t"),
            '\n' => b.push_str("\\n"),
            '\r' => b.push_str("\\r"),
            c if (c as u32) < 0x20 => b.push_str(&format!("\\u{:04X}", c as u32)),
            c => b.push(c),
        }
    }
    b
}

/// The inverse of [`esc`].
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

pub fn hex(b: Option<&[u8]>) -> String {
    match b {
        None => "-".to_string(),
        Some(b) => b.iter().map(|x| format!("{x:02x}")).collect(),
    }
}

/// Java's exception class for an [`AnalysisError`].
pub fn exception_name(e: &AnalysisError) -> &'static str {
    match e {
        AnalysisError::IllegalArgument(m) if m.starts_with("NumberFormatException") => {
            "NumberFormatException"
        }
        AnalysisError::IllegalArgument(_) => "IllegalArgumentException",
        AnalysisError::IllegalState(m) if m.starts_with("NullPointerException") => {
            "NullPointerException"
        }
        AnalysisError::IllegalState(_) => "IllegalStateException",
        AnalysisError::AlreadyClosed(_) => "AlreadyClosedException",
        AnalysisError::Io(_) => "IOException",
    }
}

pub fn token_row(ln: usize, a: &AttributeSource) -> String {
    let term = match a.bytes_term() {
        Some(bytes) => format!("#{}", hex(Some(bytes))),
        None => esc(a.term()),
    };
    format!(
        "T\t{ln}\t{term}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        a.start_offset(),
        a.end_offset(),
        a.position_increment(),
        a.position_length(),
        esc(a.token_type()),
        a.flags(),
        hex(a.payload()),
        u8::from(a.is_keyword()),
        a.term_frequency()
    )
}

/// `AnalysisRows.rows` for one line.
pub fn analyze_line(a: &Analyzer, ln: usize, line: &str, out: &mut Vec<String>) {
    let mut ts = match a.token_stream("f", line) {
        Ok(ts) => ts,
        Err(e) => {
            out.push(format!("X\t{ln}\t{}", exception_name(&e)));
            return;
        }
    };
    let run = |ts: &mut dyn TokenStream, out: &mut Vec<String>| -> Result<(), AnalysisError> {
        ts.reset()?;
        while ts.increment_token()? {
            out.push(token_row(ln, ts.attributes()));
        }
        ts.end()?;
        let e = ts.attributes();
        out.push(format!(
            "E\t{ln}\t{}\t{}\t{}",
            e.start_offset(),
            e.end_offset(),
            e.position_increment()
        ));
        Ok(())
    };
    if let Err(e) = run(&mut ts, out) {
        out.push(format!("X\t{ln}\t{}", exception_name(&e)));
    }
    let _ = ts.close();
}

/// Expected rows with Java's lone-surrogate escapes read as U+FFFD.
pub fn normalise_expected(row: &str) -> String {
    let mut out = String::with_capacity(row.len());
    let mut rest = row;
    while let Some(i) = rest.find("\\u") {
        let (head, tail) = rest.split_at(i);
        out.push_str(head);
        let escaped_backslash = head.chars().rev().take_while(|&c| c == '\\').count() % 2 == 1;
        let code = tail.get(2..6).and_then(|h| u32::from_str_radix(h, 16).ok());
        match code {
            Some(c) if !escaped_backslash && (0xD800..=0xDFFF).contains(&c) => {
                out.push('\u{FFFD}');
                rest = &tail[6..];
            }
            _ => {
                out.push_str(&tail[..2]);
                rest = &tail[2..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// A `.rows` fixture of many configurations: each row is `prefix<TAB>row`,
/// the prefix `prefix_fields` tab-separated fields naming the
/// configuration. The rows grouped by prefix, in file order, normalised.
pub fn prefixed_rows(dir: &str, file: &str, prefix_fields: usize) -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(data_dir(dir) + file).unwrap();
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for line in text.lines() {
        let cut = line
            .match_indices('\t')
            .nth(prefix_fields - 1)
            .map(|(i, _)| i)
            .unwrap_or_else(|| panic!("{file}: no prefix in {line:?}"));
        let (prefix, row) = (&line[..cut], &line[cut + 1..]);
        if groups.last().is_none_or(|g| g.0 != prefix) {
            groups.push((prefix.to_string(), Vec::new()));
        }
        groups.last_mut().unwrap().1.push(normalise_expected(row));
    }
    groups
}

/// `analyzer` over `lines`, row for row against `expected`.
pub fn check_rows(what: &str, analyzer: &Analyzer, lines: &[String], expected: &[String]) {
    let mut actual = Vec::new();
    for (ln, line) in lines.iter().enumerate() {
        analyze_line(analyzer, ln, line, &mut actual);
    }
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(a, e, "{what}: row {i} differs");
    }
    assert_eq!(actual.len(), expected.len(), "{what}: row count");
}

/// The `.tsv` chain fixtures of `fixtures/data/<dir>`.
pub fn fixture_names(dir: &str) -> BTreeSet<String> {
    std::fs::read_dir(data_dir(dir))
        .unwrap_or_else(|e| panic!("fixtures/data/{dir}: {e}"))
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|f| f.strip_suffix(".tsv").map(str::to_string))
        .collect()
}

/// Runs every chain fixture of `fixtures/data/<dir>` named by `names` over
/// the corpus through `build`'s analyzer, row for row. Every name must build.
/// Returns how many chains were checked.
pub fn check_chains(
    dir: &str,
    corpus_name: &str,
    names: &BTreeSet<String>,
    build: impl Fn(&str) -> Option<Analyzer>,
) -> usize {
    check_chains_over(dir, &corpus(corpus_name), names, build)
}

/// [`check_chains`] over explicit lines.
pub fn check_chains_over(
    dir: &str,
    lines: &[String],
    names: &BTreeSet<String>,
    build: impl Fn(&str) -> Option<Analyzer>,
) -> usize {
    let mut checked = 0;
    for name in names {
        let analyzer = build(name).unwrap_or_else(|| panic!("{dir}/{name}: no Rust chain"));
        let expected_text =
            std::fs::read_to_string(format!("{}{name}.tsv", data_dir(dir))).unwrap();
        let expected: Vec<String> = expected_text.lines().map(normalise_expected).collect();
        let mut actual = Vec::new();
        for (ln, line) in lines.iter().enumerate() {
            analyze_line(&analyzer, ln, line, &mut actual);
        }
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(a, e, "{dir}/{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{dir}/{name}: row count");
        checked += 1;
    }
    checked
}
