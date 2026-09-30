//! Differential test against real Lucene 10.5.0 for M7's analysis port:
//! `StandardTokenizer` (maxTokenLength 255, 5 and the limits around it),
//! `StandardAnalyzer` (with and without stopwords), a custom chain built from
//! the core APIs (a `CharFilter`, a `TokenFilter`, a `FilteringTokenFilter`,
//! a case-insensitive `StopFilter`, `LowerCaseFilter`), `Analyzer.normalize`,
//! a `GraphTokenFilter` subclass and both automaton converters.
//!
//! The corpus -- Unicode's word-break and emoji conformance inputs (the
//! 5,843 strings of Lucene's `WordBreakTestUnicode_12_1_0` and
//! `EmojiTokenizationTestUnicode_12_1`), fixed multilingual texts (CJK, Thai/Lao/Myanmar/Khmer,
//! Hangul syllables and jamo, kana, emoji with ZWJ, flags and keycaps,
//! numbers, URLs and emails, very long runs), UAX#29 conformance-style pairs
//! and triples over every Word_Break class, 800 seeded random strings and a
//! long multilingual document -- and every expected token come from
//! `fixtures/src/GenStandardTokenizer.java`. Every token's term, offsets,
//! position increment, position length and type, and the `end()` offset and
//! increment, must match exactly.

use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::reader::{read_to_string, CharFilter, CharReader, StrReader};
use lucene_analysis::token_stream::{consume, TokenFilter, TokenStream, Tokenizer};
use lucene_analysis::{
    automaton_to_token_stream, AnalysisError, Analyzer, AnalyzerDefinition, Automaton,
    CharArraySet, FilteringTokenFilter, GraphTokenFilter, LowerCaseFilter, StandardAnalyzer,
    StandardTokenizer, StopFilter, TokenStreamComponents, TokenStreamToAutomaton,
};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/standard_tokenizer/"
    )
    .to_string()
}

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{}{name}", dir()))
        .unwrap_or_else(|e| panic!("{name}: {e} (run GenStandardTokenizer)"))
}

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// What `GenStandardTokenizer.tokensOf` writes for `ts`: reset, every token,
/// end (the caller closes). A term equal to its offsets' slice of `source`
/// is "=".
fn tokens_of(ts: &mut dyn TokenStream, source: Option<&[u16]>) -> String {
    ts.reset().unwrap();
    let mut out = Vec::new();
    while ts.increment_token().unwrap() {
        let a = ts.attributes();
        let mut h = hex(a.term_bytes());
        if let Some(src) = source {
            let (s, e) = (a.start_offset() as usize, a.end_offset() as usize);
            if e <= src.len() && h == hex(String::from_utf16_lossy(&src[s..e]).as_bytes()) {
                h = "=".to_string();
            }
        }
        out.push(format!(
            "{h},{},{},{},{},{}",
            a.start_offset(),
            a.end_offset(),
            a.position_increment(),
            a.position_length(),
            a.token_type()
        ));
    }
    ts.end().unwrap();
    let a = ts.attributes();
    format!(
        "{}|{},{}",
        out.join(";"),
        a.end_offset(),
        a.position_increment()
    )
}

// ------------------------------------------------------- the custom chain

/// `GenStandardTokenizer.MapCharFilter`: deletes '-', expands 'ß' to "ss",
/// maps '_' to ' '; corrects through a per-output-unit table.
struct MapCharFilter {
    input: Box<dyn CharReader>,
    state: Option<(Vec<u16>, Vec<i32>, i32)>,
    pos: usize,
}

impl MapCharFilter {
    fn new(input: Box<dyn CharReader>) -> Self {
        MapCharFilter {
            input,
            state: None,
            pos: 0,
        }
    }

    fn fill(&mut self) -> Result<(), AnalysisError> {
        let src: Vec<u16> = read_to_string(&mut *self.input)?.encode_utf16().collect();
        let mut out = Vec::new();
        let mut map = Vec::new();
        for (i, &c) in src.iter().enumerate() {
            if c == u16::from(b'-') {
                continue;
            }
            if c == 0xDF {
                out.extend([u16::from(b's'); 2]);
                map.extend([i as i32; 2]);
            } else {
                out.push(if c == u16::from(b'_') {
                    u16::from(b' ')
                } else {
                    c
                });
                map.push(i as i32);
            }
        }
        map.push(src.len() as i32);
        self.state = Some((out, map, src.len() as i32));
        Ok(())
    }
}

impl CharFilter for MapCharFilter {
    fn input(&self) -> &dyn CharReader {
        &*self.input
    }
    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut *self.input
    }
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        if self.state.is_none() {
            self.fill()?;
        }
        let (out, ..) = self.state.as_ref().unwrap();
        let n = buf.len().min(out.len() - self.pos);
        buf[..n].copy_from_slice(&out[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn correct(&self, off: i32) -> i32 {
        let (out, map, in_len) = self.state.as_ref().unwrap();
        match map.get(off as usize) {
            Some(&m) => m,
            None => off + (in_len - out.len() as i32),
        }
    }
}

/// `GenStandardTokenizer.NumberTagFilter`.
struct NumberTagFilter<I> {
    input: I,
}

impl<I: TokenStream> TokenFilter for NumberTagFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if a.token_type() == "<NUM>" {
            let t = format!("n:{}", a.term());
            a.set_term(&t);
            a.set_token_type("number");
        }
        Ok(true)
    }
}

/// `GenStandardTokenizer.ChainAnalyzer`.
struct ChainAnalyzer;

impl AnalyzerDefinition for ChainAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut src = StandardTokenizer::new();
        src.set_max_token_length(10)?;
        let ts = NumberTagFilter { input: src };
        let stop = CharArraySet::from_words(["THE", "und", "Le"], true);
        let ts = StopFilter::new(ts, Arc::new(stop));
        let ts = FilteringTokenFilter::new(ts, |a: &AttributeSource| a.term_utf16_len() > 1);
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(ts)))
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(MapCharFilter::new(reader))
    }

    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        Box::new(MapCharFilter::new(reader))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

const STOP: [&str; 7] = ["the", "a", "of", "and", "is", "und", "le"];

#[test]
fn every_case_matches_lucene() {
    let std255 = Analyzer::new(StandardAnalyzer::new());
    let std5 = Analyzer::new(StandardAnalyzer::new().with_max_token_length(5));
    let stop = Analyzer::new(StandardAnalyzer::with_stopwords(CharArraySet::from_words(
        STOP, false,
    )));
    let chain = Analyzer::new(ChainAnalyzer);
    // One tokenizer reused across every text and length, as Lucene reuses it.
    let mut tok = StandardTokenizer::new();

    let cases = read("cases.txt");
    let texts: std::collections::HashMap<&str, String> = cases
        .lines()
        .filter_map(|l| l.strip_prefix("T|"))
        .map(|l| {
            let (name, h) = l.split_once('|').unwrap();
            (name, String::from_utf8(unhex(h)).unwrap())
        })
        .collect();
    let (mut checked, mut failures) = (0usize, Vec::new());
    for line in cases.lines() {
        let mut parts = line.splitn(3, '|');
        let (a, b, rest) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        if a == "T" {
            continue;
        }
        if a == "N" {
            let (t, want) = rest.split_once('|').unwrap();
            let t = String::from_utf8(unhex(t)).unwrap();
            let analyzer = if b == "norm-std" { &std255 } else { &chain };
            let got = hex(&analyzer.normalize("f", &t).unwrap());
            if got != want {
                failures.push(format!("normalize {b} {t:?}: got {got}, want {want}"));
            }
            checked += 1;
            continue;
        }
        let name = a;
        let text = &texts[name];
        let utf16: Vec<u16> = text.encode_utf16().collect();
        let got = match b {
            _ if b.starts_with("tok:") => {
                let max: i32 = b[4..].parse().unwrap();
                tok.set_max_token_length(max).unwrap();
                tok.set_reader(Box::new(StrReader::new(text.as_str())))
                    .unwrap();
                let got = tokens_of(&mut tok, Some(&utf16));
                tok.close().unwrap();
                got
            }
            "std:255" | "std:5" | "stop" | "chain" => {
                let (analyzer, source) = match b {
                    "std:255" => (&std255, Some(&utf16[..])),
                    "std:5" => (&std5, Some(&utf16[..])),
                    "stop" => (&stop, Some(&utf16[..])),
                    _ => (&chain, None),
                };
                let mut ts = analyzer.token_stream("f", text).unwrap();
                let got = tokens_of(&mut ts, source);
                ts.close().unwrap();
                got
            }
            other => panic!("unknown config {other}"),
        };
        if got != rest {
            failures.push(format!(
                "{name} [{b}] text={text:?}\n   got: {got}\n  want: {rest}"
            ));
        }
        checked += 1;
    }
    assert!(checked > 5000, "only {checked} cases");
    assert!(
        failures.is_empty(),
        "{} of {checked} cases diverged from Lucene; first:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ----------------------------------------------------------------- graphs

/// `GenStandardTokenizer.CannedStream`.
struct Canned {
    atts: AttributeSource,
    toks: Vec<(String, i32, i32, i32, i32)>,
    end: (i32, i32),
    upto: usize,
}

impl Canned {
    fn new(spec: &str) -> Self {
        let parts: Vec<&str> = spec.split(' ').collect();
        let (last, toks) = parts.split_last().unwrap();
        let toks = toks
            .iter()
            .map(|t| {
                let f: Vec<&str> = t.split('/').collect();
                (
                    f[0].to_string(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                    f[4].parse().unwrap(),
                )
            })
            .collect();
        let e: Vec<&str> = last.split('/').collect();
        Canned {
            atts: AttributeSource::new(),
            toks,
            end: (e[1].parse().unwrap(), e[2].parse().unwrap()),
            upto: 0,
        }
    }
}

impl TokenStream for Canned {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        let Some((t, inc, len, s, e)) = self.toks.get(self.upto).cloned() else {
            return Ok(false);
        };
        self.upto += 1;
        self.atts.set_term(&t);
        self.atts.set_position_increment(inc)?;
        self.atts.set_position_length(len)?;
        self.atts.set_offset(s, e)?;
        Ok(true)
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_position_increment(self.end.0)?;
        self.atts.set_offset(self.end.1, self.end.1)
    }
}

/// `GenStandardTokenizer.PathsFilter`.
struct PathsFilter<I> {
    graph: GraphTokenFilter<I>,
    depth: usize,
    pending: std::collections::VecDeque<String>,
    base_state: AttributeSource,
}

impl<I: TokenStream> TokenFilter for PathsFilter<I> {
    type Input = GraphTokenFilter<I>;
    fn input(&self) -> &Self::Input {
        &self.graph
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.graph
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        while self.pending.is_empty() {
            if !self.graph.increment_base_token()? {
                return Ok(false);
            }
            self.base_state = self.graph.attributes().capture_state();
            loop {
                let mut path = self.graph.attributes().term().to_string();
                let mut n = 1;
                while n < self.depth && self.graph.increment_graph_token()? {
                    path.push('_');
                    path.push_str(self.graph.attributes().term());
                    n += 1;
                }
                self.pending.push_back(path);
                if !self.graph.increment_graph()? {
                    break;
                }
            }
        }
        let state = self.base_state.clone();
        let atts = self.graph.attributes_mut();
        atts.restore_state(&state);
        atts.set_term(&self.pending.pop_front().unwrap());
        Ok(true)
    }
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.graph.reset()?;
        self.pending.clear();
        Ok(())
    }
}

/// `GenStandardTokenizer.dump`.
fn dump(a: &Automaton) -> String {
    let mut s = a.get_num_states().to_string();
    for (st, ts) in a.get_sorted_transitions().iter().enumerate() {
        s.push(';');
        if a.is_accept(st as i32) {
            s.push('A');
        }
        for t in ts {
            s.push_str(&format!(" {}:{}-{}", t.dest, t.min, t.max));
        }
    }
    s
}

#[test]
fn graph_filter_and_automaton_converters_match_lucene() {
    let graph = read("graph.txt");
    let mut checked = 0;
    for line in graph.lines() {
        let mut parts = line.splitn(3, '|');
        let (kind, spec, want) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        let got = match kind {
            "graph" => {
                let mut f = PathsFilter {
                    graph: GraphTokenFilter::new(Canned::new(spec)),
                    depth: 3,
                    pending: Default::default(),
                    base_state: AttributeSource::new(),
                };
                tokens_of(&mut f, None)
            }
            "a2ts" => {
                let a = TokenStreamToAutomaton::new()
                    .to_automaton(&mut Canned::new(spec))
                    .unwrap();
                let mut ts = automaton_to_token_stream(&a).unwrap();
                tokens_of(&mut ts, None)
            }
            _ => {
                let mut conv = TokenStreamToAutomaton::new();
                match kind {
                    "ts2a-default" => {}
                    "ts2a-nopreserve" => conv.set_preserve_position_increments(false),
                    "ts2a-finalhole-unicode" => {
                        conv.set_final_offset_gap_as_hole(true);
                        conv.set_unicode_arcs(true);
                    }
                    other => panic!("unknown kind {other}"),
                }
                dump(&conv.to_automaton(&mut Canned::new(spec)).unwrap())
            }
        };
        assert_eq!(got, want, "{kind} over {spec:?}");
        checked += 1;
    }
    assert_eq!(checked, 35);
}

/// The tokenizer reached through a `TokenStreamComponents` custom source
/// (`TokenStreamComponents(Consumer<Reader>, TokenStream)`), consumed with
/// the crate's `consume` helper, agrees with the direct path.
#[test]
fn custom_source_hook_feeds_the_tokenizer() {
    struct Hooked;
    impl AnalyzerDefinition for Hooked {
        fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
            Ok(TokenStreamComponents::with_source(
                |sink: &mut dyn TokenStream, r: Box<dyn CharReader>| {
                    sink.as_tokenizer()
                        .expect("tokenizer at the source")
                        .set_reader(r)
                },
                LowerCaseFilter::new(StandardTokenizer::new()),
            ))
        }
    }
    let a = Analyzer::new(Hooked);
    let mut ts = a.token_stream("f", "Hello 世界").unwrap();
    let mut terms = Vec::new();
    consume(&mut ts, |x| terms.push(x.term().to_string())).unwrap();
    assert_eq!(terms, vec!["hello", "世", "界"]);
}
