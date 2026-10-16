//! M13 T13.2: the classic `Highlighter` against Lucene 10.5.0. Every query of
//! `fixtures/corpus/highlight-queries.txt` over every text of
//! `highlight-texts.txt`, under eleven configurations of scorer, fragmenter,
//! formatter, encoder and limits, gives the fragments Lucene gives
//! (`fixtures/src/GenHighlight.java`); and what the span-term extractor
//! extracts -- terms, weights, position sensitivity and spans -- is Lucene's.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::attributes::java_float_to_string;
use lucene_analysis::{Analyzer, StandardAnalyzer, TokenStream};
use lucene_search::highlight::{
    DocFreqSource, DynStream, GradientFormatter, Highlighter, NullFragmenter, QueryScorer,
    QueryTermScorer, SimpleFragmenter, SimpleHtmlEncoder, SimpleHtmlFormatter,
    SimpleSpanFragmenter, SpanGradientFormatter, WeightedSpanTermExtractor,
};
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::spans::SpanNode;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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

/// `AnalysisRows.esc`.
fn esc(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

fn lines(name: &str) -> Vec<String> {
    let text = fs::read_to_string(dir().join("corpus").join(name)).unwrap();
    let mut v: Vec<String> = text.split('\n').map(str::to_string).collect();
    if v.last().is_some_and(String::is_empty) {
        v.pop();
    }
    v
}

fn span(toks: &[String], pos: &mut usize) -> SpanNode {
    assert_eq!(toks[*pos], "(");
    *pos += 1;
    let kind = toks[*pos].clone();
    *pos += 1;
    let node = match kind.as_str() {
        "term" => {
            let t = toks[*pos].clone();
            *pos += 1;
            SpanNode::term("body", t.into_bytes())
        }
        "near" => {
            let slop: i32 = toks[*pos].parse().unwrap();
            let ordered: bool = toks[*pos + 1].parse().unwrap();
            *pos += 2;
            let mut cs = Vec::new();
            while toks[*pos] != ")" {
                cs.push(span(toks, pos));
            }
            SpanNode::near(cs, slop, ordered).unwrap()
        }
        "or" => {
            let mut cs = Vec::new();
            while toks[*pos] != ")" {
                cs.push(span(toks, pos));
            }
            SpanNode::or(cs).unwrap()
        }
        "first" => {
            let end: i32 = toks[*pos].parse().unwrap();
            *pos += 1;
            SpanNode::first(span(toks, pos), end)
        }
        "not" => {
            let inc = span(toks, pos);
            let exc = span(toks, pos);
            SpanNode::not(inc, exc, 0, 0).unwrap()
        }
        other => panic!("{other}"),
    };
    assert_eq!(toks[*pos], ")");
    *pos += 1;
    node
}

fn parse(parser: &QueryParser, line: &str) -> Clause {
    if let Some(rest) = line.strip_prefix("span: ") {
        let toks: Vec<String> = rest
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(str::to_string)
            .collect();
        return Clause::from(span(&toks, &mut 0));
    }
    parser.parse(line).unwrap().to_clause(1)
}

fn join(frags: Vec<String>) -> String {
    frags.join("<|>")
}

fn stream<'a>(analyzer: &'a Analyzer, text: &str) -> DynStream<'a> {
    DynStream::new(analyzer.token_stream("body", text).unwrap())
}

/// The document frequencies of an index of the texts, one document each.
struct TextsIndex {
    docs: Vec<std::collections::HashSet<String>>,
}

impl TextsIndex {
    fn new(analyzer: &Analyzer, texts: &[String]) -> Self {
        let docs = texts
            .iter()
            .map(|t| {
                let mut ts = analyzer.token_stream("body", t).unwrap();
                ts.reset().unwrap();
                let mut set = std::collections::HashSet::new();
                while ts.increment_token().unwrap() {
                    set.insert(ts.attributes().term().to_string());
                }
                set
            })
            .collect();
        TextsIndex { docs }
    }
}

impl DocFreqSource for TextsIndex {
    fn max_doc(&self) -> i32 {
        i32::try_from(self.docs.len()).unwrap()
    }
    fn doc_freq(&self, field: &str, term: &str) -> i32 {
        if field != "body" {
            return 0;
        }
        i32::try_from(self.docs.iter().filter(|d| d.contains(term)).count()).unwrap()
    }
}

fn scored_fragments(h: &mut Highlighter, analyzer: &Analyzer, t: &str) -> Result<String, String> {
    h.get_best_text_fragments(stream(analyzer, t), t, true, 3)
        .map(|frags| {
            join(
                frags
                    .iter()
                    .map(|f| format!("{}:{}:{f}", java_float_to_string(f.score()), f.frag_num()))
                    .collect(),
            )
        })
        .map_err(|e| e.to_string())
}

fn run(
    cfg: &str,
    q: &Clause,
    analyzer: &Analyzer,
    t: &str,
    index: &Arc<TextsIndex>,
) -> Result<String, String> {
    let e = |e: lucene_search::highlight::Error| e.to_string();
    match cfg {
        "L" => {
            let mut h =
                Highlighter::new(QueryScorer::with_reader(q, Arc::clone(index) as _, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(40));
            scored_fragments(&mut h, analyzer, t)
        }
        "M" => {
            let mut h = Highlighter::new(QueryTermScorer::with_reader(q, index.as_ref(), "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(40));
            scored_fragments(&mut h, analyzer, t)
        }
        "A" => {
            let mut h = Highlighter::new(QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(40));
            h.get_best_fragments(stream(analyzer, t), t, 3)
                .map(join)
                .map_err(e)
        }
        "B" => {
            let s = QueryScorer::for_field(q, "body");
            let f = SimpleSpanFragmenter::with_size(&s, 40);
            let mut h = Highlighter::new(s);
            h.set_text_fragmenter(f);
            h.get_best_fragments(stream(analyzer, t), t, 3)
                .map(join)
                .map_err(e)
        }
        "C" => {
            let mut h = Highlighter::new(QueryTermScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(30));
            h.get_best_fragment(stream(analyzer, t), t)
                .map(Option::unwrap_or_default)
                .map_err(e)
        }
        "D" => {
            let mut h = Highlighter::with_encoder(
                SimpleHtmlFormatter::new("[", "]"),
                SimpleHtmlEncoder,
                QueryScorer::new(q),
            );
            h.set_text_fragmenter(SimpleFragmenter::new(25));
            h.get_best_fragments(stream(analyzer, t), t, 5)
                .map(join)
                .map_err(e)
        }
        "E" => {
            let mut h = Highlighter::new(QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(NullFragmenter);
            h.get_best_fragments(stream(analyzer, t), t, 1)
                .map(join)
                .map_err(e)
        }
        "F" => {
            let mut h = Highlighter::new(QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(20));
            h.set_max_doc_chars_to_analyze(30);
            h.get_best_fragments(stream(analyzer, t), t, 4)
                .map(join)
                .map_err(e)
        }
        "G" => {
            let mut h = Highlighter::new(QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(35));
            h.get_best_text_fragments(stream(analyzer, t), t, false, 4)
                .map(|frags| {
                    join(
                        frags
                            .iter()
                            .map(|f| {
                                format!("{}:{}:{f}", java_float_to_string(f.score()), f.frag_num())
                            })
                            .collect(),
                    )
                })
                .map_err(e)
        }
        "H" => {
            let g = GradientFormatter::new(
                2.0,
                Some("#000000"),
                Some("#FF0000"),
                Some("#FFFFFF"),
                Some("#00FF00"),
            )
            .unwrap();
            let mut h = Highlighter::with_formatter(g, QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(60));
            h.get_best_fragments(stream(analyzer, t), t, 2)
                .map(join)
                .map_err(e)
        }
        "I" => {
            let g = SpanGradientFormatter::new(2.0, None, None, Some("#112233"), Some("#AABBCC"))
                .unwrap();
            let mut h = Highlighter::with_formatter(g, QueryScorer::for_field(q, "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(60));
            h.get_best_fragments(stream(analyzer, t), t, 2)
                .map(join)
                .map_err(e)
        }
        "J" => {
            let mut s = QueryScorer::for_field(q, "body");
            s.set_expand_multi_term_query(false);
            let mut h = Highlighter::new(s);
            h.set_text_fragmenter(SimpleFragmenter::new(40));
            h.get_best_fragments(stream(analyzer, t), t, 3)
                .map(join)
                .map_err(e)
        }
        "K" => {
            let mut h = Highlighter::new(QueryScorer::for_field_with_default(q, "title", "body"));
            h.set_text_fragmenter(SimpleFragmenter::new(40));
            h.get_best_fragments_joined(stream(analyzer, t), t, 2, "...")
                .map_err(e)
        }
        other => panic!("{other}"),
    }
}

#[test]
fn highlighted_fragments_match_lucene() {
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let parser = QueryParser::new("body", Arc::new(Analyzer::new(StandardAnalyzer::new())));
    let queries = lines("highlight-queries.txt");
    let texts = lines("highlight-texts.txt");
    let want = fs::read_to_string(dir().join("data/highlight/highlight.tsv")).unwrap();
    let parsed: Vec<Clause> = queries.iter().map(|q| parse(&parser, q)).collect();
    let index = Arc::new(TextsIndex::new(&analyzer, &texts));
    let (mut checked, mut bad) = (0, Vec::new());
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let (cfg, qi, ti) = (
            c[0],
            c[1].parse::<usize>().unwrap(),
            c[2].parse::<usize>().unwrap(),
        );
        let expected = format!("{}\t{}", c[3], unesc(c[4]));
        let got = match run(cfg, &parsed[qi], &analyzer, &texts[ti], &index) {
            Ok(s) => format!("R\t{s}"),
            Err(m) => format!("X\t{m}"),
        };
        checked += 1;
        if got != expected {
            bad.push(format!(
                "config {cfg} query {qi} {:?} text {ti}\n  want {}\n  got  {}",
                queries[qi],
                esc(&expected),
                esc(&got)
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 31 * 9 * 13);
}

#[test]
fn the_span_term_extractor_matches_lucene() {
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let parser = QueryParser::new("body", Arc::new(Analyzer::new(StandardAnalyzer::new())));
    let queries = lines("highlight-queries.txt");
    let texts = lines("highlight-texts.txt");
    let want = fs::read_to_string(dir().join("data/highlight/spanterms.tsv")).unwrap();
    let parsed: Vec<Clause> = queries.iter().map(|q| parse(&parser, q)).collect();
    let (mut checked, mut bad) = (0, Vec::new());
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let (qi, ti) = (
            c[0].parse::<usize>().unwrap(),
            c[1].parse::<usize>().unwrap(),
        );
        let mut qse = WeightedSpanTermExtractor::new(None);
        qse.set_expand_multi_term_query(true);
        qse.set_max_doc_chars_to_analyze(50 * 1024);
        let extraction = qse
            .get_weighted_span_terms(
                &parsed[qi],
                1.0,
                stream(&analyzer, &texts[ti]),
                Some("body"),
            )
            .unwrap();
        let mut keys: Vec<&String> = extraction.terms.keys().collect();
        keys.sort();
        let cells: Vec<String> = keys
            .into_iter()
            .map(|k| {
                let w = &extraction.terms[k];
                let mut spans: Vec<String> = w
                    .position_spans()
                    .iter()
                    .map(|s| format!("{}-{}", s.start(), s.end()))
                    .collect();
                spans.sort();
                format!(
                    "{k}={}={}={}",
                    java_float_to_string(w.weight()),
                    w.is_position_sensitive(),
                    spans.join(",")
                )
            })
            .collect();
        let got = cells.join(" ");
        let expected = unesc(c[3]);
        checked += 1;
        if got != expected {
            bad.push(format!(
                "query {qi} {:?} text {ti}\n  want {expected}\n  got  {got}",
                queries[qi]
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 31 * 9);
}

/// The term vectors Lucene wrote for the texts (one segment, so document
/// numbers are the texts' indexes).
fn term_vectors() -> (
    lucene_search::directory_reader::DirectoryReader,
    Vec<Option<lucene_codecs::term_vectors::TermVectorsDocument>>,
) {
    use lucene_search::directory_reader::DirectoryReader;
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(
        dir().join("data/highlight/tvindex"),
    ))
    .unwrap();
    let docs = {
        let seg = &reader.segment_readers()[0];
        let tv = seg.term_vectors_reader().unwrap().unwrap();
        (0..tv.max_doc()).map(|d| tv.document(d).unwrap()).collect()
    };
    (reader, docs)
}

fn vector_of<'a>(
    reader: &lucene_search::directory_reader::DirectoryReader,
    docs: &'a [Option<lucene_codecs::term_vectors::TermVectorsDocument>],
    doc: usize,
    field: &str,
) -> Option<&'a lucene_codecs::term_vectors::TermVectorField> {
    let number = reader.segment_readers()[0]
        .field_infos()
        .field_by_name(field)?
        .number;
    docs[doc]
        .as_ref()?
        .fields
        .iter()
        .find(|f| f.field_number == number)
}

#[test]
fn term_vector_token_streams_match_lucene() {
    use lucene_search::highlight::TokenSources;
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let texts = lines("highlight-texts.txt");
    let (reader, docs) = term_vectors();
    let want = fs::read_to_string(dir().join("data/highlight/termvectors.tsv")).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let (field, doc) = (c[0], c[1].parse::<usize>().unwrap());
        let vector = vector_of(&reader, &docs, doc, field);
        let got = if c[2] == "only" {
            match TokenSources::get_term_vector_token_stream_or_null(vector, -1).unwrap() {
                Some(_) => "stream".to_string(),
                None => "null".to_string(),
            }
        } else {
            let max: i32 = c[2].parse().unwrap();
            match TokenSources::get_token_stream(field, vector, &texts[doc], &analyzer, max) {
                Err(e) => format!("X\t{e}"),
                Ok(mut ts) => {
                    ts.reset().unwrap();
                    let mut toks = Vec::new();
                    while ts.increment_token().unwrap() {
                        let a = ts.attributes();
                        toks.push(format!(
                            "{}@{}-{}+{}",
                            a.term(),
                            a.start_offset(),
                            a.end_offset(),
                            a.position_increment()
                        ));
                    }
                    format!("R\t{}", toks.join(" "))
                }
            }
        };
        let expected = if c[2] == "only" {
            c[3].to_string()
        } else {
            format!("{}\t{}", c[3], unesc(c[4]))
        };
        checked += 1;
        if got != expected {
            bad.push(format!("{row}\n  got {got}"));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 3 * 9 * 4);
}

#[test]
fn highlighting_through_term_vectors_matches_lucene() {
    use lucene_search::highlight::TokenSources;
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let parser = QueryParser::new("body", Arc::new(Analyzer::new(StandardAnalyzer::new())));
    let queries = lines("highlight-queries.txt");
    let texts = lines("highlight-texts.txt");
    let parsed: Vec<Clause> = queries.iter().map(|q| parse(&parser, q)).collect();
    let (reader, docs) = term_vectors();
    let want = fs::read_to_string(dir().join("data/highlight/tv_highlight.tsv")).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let field = c[0];
        let (qi, doc) = (
            c[1].parse::<usize>().unwrap(),
            c[2].parse::<usize>().unwrap(),
        );
        let vector = vector_of(&reader, &docs, doc, field);
        let got = TokenSources::get_token_stream(field, vector, &texts[doc], &analyzer, -1)
            .and_then(|ts| {
                let mut h = Highlighter::new(QueryScorer::for_field(&parsed[qi], "body"));
                h.set_text_fragmenter(SimpleFragmenter::new(40));
                h.get_best_fragments(ts, &texts[doc], 3)
            })
            .map(|f| format!("R\t{}", join(f)))
            .unwrap_or_else(|e| format!("X\t{e}"));
        let expected = format!("{}\t{}", c[3], unesc(c[4]));
        checked += 1;
        if got != expected {
            bad.push(format!(
                "{field} query {qi} doc {doc}\n  want {expected}\n  got  {got}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 31 * 9 * 2);
}
