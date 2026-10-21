//! M13 T13.2 (`uhighlight`) against Lucene 10.5.0: the unified highlighter's
//! snippets (for parsed and hand-built queries), its stored-field batching
//! and objects, and its break iterators.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod, SynonymQuery};
use lucene_search::field_norms::FieldNorms;
use lucene_search::highlight::unified::break_iterators::{
    sentence_instance, word_instance, BreakIterator, CustomSeparatorBreakIterator,
    LengthGoalBreakIterator, SplittingBreakIterator, WholeBreakIterator,
};
use lucene_search::highlight::unified::{
    DefaultPassageFormatter, HighlightObject, Passage, PassageFormatter, UnifiedHighlighter,
};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, MultiPhraseQuery,
    PhraseQuery, PrefixQuery, RegexpQuery, TermQuery, WildcardQuery,
};
use lucene_search::queryparser::classic::QueryParser;
use lucene_search::spans::SpanNode;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/uhighlight")
}

/// `AnalysisRows.esc` undone, to UTF-16 (a lone surrogate survives).
fn unesc16(s: &str) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match it.next() {
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some(o) => out.push(o as u16),
            None => out.push(u16::from(b'\\')),
        }
    }
    out
}

fn unesc(s: &str) -> String {
    String::from_utf16_lossy(&unesc16(s))
}

fn iterator(n: usize) -> Box<dyn BreakIterator> {
    match n {
        0 => Box::new(WholeBreakIterator::new()),
        1 => Box::new(CustomSeparatorBreakIterator::new(u16::from(b','))),
        2 => Box::new(SplittingBreakIterator::new(
            sentence_instance(),
            u16::from(b';'),
        )),
        3 => Box::new(LengthGoalBreakIterator::create_min_length(
            sentence_instance(),
            20,
            0.5,
        )),
        4 => Box::new(LengthGoalBreakIterator::create_closest_to_length(
            sentence_instance(),
            20,
            0.0,
        )),
        5 => Box::new(LengthGoalBreakIterator::create_closest_to_length(
            word_instance(),
            10,
            1.0,
        )),
        _ => Box::new(LengthGoalBreakIterator::create_min_length(
            Box::new(SplittingBreakIterator::new(
                sentence_instance(),
                u16::from(b';'),
            )),
            30,
            0.25,
        )),
    }
}

#[test]
fn break_iterators_break_as_lucenes_do() {
    let text = fs::read_to_string(dir().join("breaks.tsv")).unwrap();
    let (mut rows, mut bad) = (0, Vec::new());
    for line in text.split('\n').filter(|l| !l.is_empty()) {
        let c: Vec<&str> = line.split('\t').collect();
        let n: usize = c[0].parse().unwrap();
        let t = unesc16(c[1]);
        let mut bi = iterator(n);
        bi.set_text(&t);
        let mut problems = Vec::new();
        for cell in c[2].split(' ').filter(|x| !x.is_empty()) {
            let (got, want) = match cell.as_bytes()[0] {
                b'F' => (format!("F{}", bi.first()), cell.to_string()),
                b'L' => (format!("L{}", bi.last()), cell.to_string()),
                b'N' => (format!("N{}", bi.next()), cell.to_string()),
                b'C' => (format!("C{}", bi.current()), cell.to_string()),
                b'P' => (format!("P{}", bi.previous()), cell.to_string()),
                b'f' | b'p' => {
                    let (head, _) = cell.split_once('=').unwrap();
                    let off: i32 = head[1..].parse().unwrap();
                    let r = if cell.starts_with('f') {
                        bi.following(off)
                    } else {
                        bi.preceding(off)
                    };
                    (format!("{head}={r}"), cell.to_string())
                }
                _ => unreachable!(),
            };
            if got != want {
                problems.push(format!("{want} got {got}"));
                break;
            }
        }
        rows += 1;
        if !problems.is_empty() {
            bad.push(format!("iterator {n} {:?}: {}", c[1], problems.join("; ")));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {rows} differ:\n{}",
        bad.len(),
        bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(rows, 3000);
}

#[test]
fn unified_highlighting_equals_lucenes() {
    let reader =
        DirectoryReader::open(&lucene_store::FsDirectory::open(dir().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let names: Vec<String> = ["id", "offs", "pos", "docs", "tv", "both", "tvpos"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = {
        let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
        for field in &names {
            for (i, n) in reader.field_norms(field).into_iter().enumerate() {
                if owned.len() <= i {
                    owned.push(HashMap::new());
                }
                if let Some(n) = n {
                    owned[i].insert(field.clone(), n);
                }
            }
        }
        owned
    };
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = QueryParser::new("offs", Arc::clone(&analyzer));
    let queries: Vec<String> =
        fs::read_to_string(dir().join("../../corpus/uhighlight-queries.txt"))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
    let all: Vec<i32> = (0..reader.max_doc()).collect();
    let want = fs::read_to_string(dir().join("uhighlight.tsv")).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let (cfg, qi): (usize, usize) = (c[0].parse().unwrap(), c[1].parse().unwrap());
        let field = c[2];
        let text = queries[qi].replace("F:", &format!("{field}:"));
        let clause = parser.parse(&text).unwrap().to_clause(reader.max_doc());
        let mut uh = UnifiedHighlighter::new(Some(&reader), Some(Arc::clone(&analyzer)));
        let mut max_passages = 2;
        match cfg {
            0 => {}
            1 => uh = uh.with_flags(true, false, true, false),
            2 => uh = uh.with_flags(true, false, false, false),
            3 => {
                uh = uh
                    .with_break_iterator(Arc::new(|| {
                        Box::new(LengthGoalBreakIterator::create_min_length(
                            sentence_instance(),
                            50,
                            0.5,
                        ))
                    }))
                    .with_formatter(Arc::new(DefaultPassageFormatter::new(
                        "[", "]", " ... ", true,
                    )))
                    .with_max_no_highlight_passages(1);
                max_passages = 3;
            }
            4 => {
                uh = uh
                    .with_field_matcher(Arc::new(|_| true))
                    .with_break_iterator(Arc::new(|| {
                        Box::new(CustomSeparatorBreakIterator::new(u16::from(b',')))
                    }))
                    .with_max_length(60);
            }
            _ => {
                uh = uh
                    .with_flags(false, true, true, true)
                    .with_break_iterator(Arc::new(|| Box::new(WholeBreakIterator::new())))
                    .with_max_no_highlight_passages(0);
                max_passages = 1;
            }
        }
        let got = match uh.highlight_fields(&searcher, &[field], &clause, &all, &[max_passages]) {
            Ok(mut m) => {
                let snippets = m.remove(field).unwrap();
                let joined: Vec<String> = snippets
                    .into_iter()
                    .map(|s| s.unwrap_or_else(|| "\u{2}".to_string()))
                    .collect();
                format!("R\t{}", joined.join("\u{1f}"))
            }
            // the fixture records the exception's class, not its message
            Err(e) => format!("X\t{}", e.to_string().split(": ").next().unwrap_or("")),
        };
        let expected = format!("{}\t{}", c[3], unesc(c[4]));
        checked += 1;
        if got != expected {
            bad.push(format!(
                "config {cfg} query {qi} {text:?} field {field}\n  want {}\n  got  {}",
                expected.replace('\n', "\\n"),
                got.replace('\n', "\\n")
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(6).cloned().collect::<Vec<_>>().join(
            "
"
        )
    );
    assert_eq!(checked, 810);
}

/// The fixture index's reader and a searcher over it, with every field's
/// norms, handed to `f`.
fn with_index(f: impl FnOnce(&DirectoryReader, &IndexSearcher<'_, '_>)) {
    let reader =
        DirectoryReader::open(&lucene_store::FsDirectory::open(dir().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
    for field in ["id", "offs", "pos", "docs", "tv", "both", "tvpos"] {
        for (i, n) in reader.field_norms(field).into_iter().enumerate() {
            if owned.len() <= i {
                owned.push(HashMap::new());
            }
            if let Some(n) = n {
                owned[i].insert(field.to_string(), n);
            }
        }
    }
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&reader, &searcher);
}

fn st(f: &str, t: &str) -> SpanNode {
    SpanNode::term(f, t)
}

fn tq(f: &str, t: &str) -> Clause {
    TermQuery::new(f, t).into()
}

fn phrase(f: &str, terms: &[&str]) -> PhraseQuery {
    PhraseQuery::new(f, terms.iter().map(|s| s.as_bytes().to_vec()))
}

fn near(clauses: Vec<SpanNode>, slop: i32, in_order: bool) -> SpanNode {
    SpanNode::near(clauses, slop, in_order).unwrap()
}

fn span_mtq(source: MultiTermSource) -> SpanNode {
    SpanNode::multi_term(MultiTermQuery::new(source, RewriteMethod::default()))
}

/// `BooleanQuery` of `(occur, clause)`: `+` must, `-` must not, else should.
fn boolean(msm: usize, clauses: Vec<(char, Clause)>) -> Clause {
    let mut b = BooleanQuery::new();
    b.minimum_should_match = msm;
    for (occur, c) in clauses {
        match occur {
            '+' => b.must.push(c),
            '-' => b.must_not.push(c),
            _ => b.should.push(c),
        }
    }
    b.into()
}

/// `GenUnifiedHighlight.built(i, f)`.
fn built(i: usize, f: &str) -> Clause {
    match i {
        0 => near(vec![st(f, "quick"), st(f, "fox")], 1, false).into(),
        1 => phrase(f, &["fox", "quick"]).with_slop(2).into(),
        2 => PhraseQuery::with_positions(f, [("quick", 0), ("fox", 2)])
            .unwrap()
            .into(),
        3 => MultiPhraseQuery::new(
            f,
            [
                vec![b"quick".to_vec(), b"lazy".to_vec()],
                vec![b"brown".to_vec(), b"dog".to_vec()],
            ],
        )
        .into(),
        4 => boolean(
            0,
            vec![
                (
                    ' ',
                    SynonymQuery::new(f, [(b"fox".to_vec(), 1.0), (b"quick".to_vec(), 1.0)])
                        .unwrap()
                        .into(),
                ),
                (' ', phrase(f, &["quick", "brown"]).into()),
            ],
        ),
        5 => near(
            vec![
                span_mtq(MultiTermSource::Prefix(PrefixQuery::new(f, "qu"))),
                st(f, "brown"),
            ],
            0,
            true,
        )
        .into(),
        6 => SpanNode::or(vec![
            st(f, "lazy"),
            span_mtq(MultiTermSource::Wildcard(WildcardQuery::new(f, "f?x"))),
        ])
        .unwrap()
        .into(),
        7 => boolean(0, vec![('+', tq(f, "fox")), ('+', tq("id", "d0"))]),
        8 => boolean(0, vec![('+', tq(f, "fox")), ('-', tq("id", "d4"))]),
        9 => boolean(
            2,
            vec![
                (' ', tq(f, "fox")),
                (' ', tq("id", "d1")),
                (' ', tq(f, "zzz")),
            ],
        ),
        10 => boolean(
            2,
            vec![
                (' ', phrase(f, &["quick", "brown"]).into()),
                (' ', tq("docs", "cat")),
            ],
        ),
        11 => SpanNode::not(st(f, "fox"), st(f, "quick"), 1, 1)
            .unwrap()
            .into(),
        12 => SpanNode::first(st(f, "fox"), 5).into(),
        13 => boolean(0, vec![(' ', tq(f, "fox")), (' ', tq("pos", "quick"))]),
        14 => ConstantScoreQuery::new(
            BoostQuery::new(
                Clause::from(near(vec![st(f, "the"), st(f, "fox")], 3, true)),
                2.0,
            ),
            1.0,
        )
        .into(),
        15 => DisjunctionMaxQuery::new(
            [Clause::from(phrase(f, &["lazy", "dog"])), tq(f, "fox")],
            0.1,
        )
        .into(),
        16 => SpanNode::within(
            near(vec![st(f, "quick"), st(f, "fox")], 3, true),
            st(f, "brown"),
        )
        .unwrap()
        .into(),
        17 => phrase(f, &["fox"]).into(),
        18 => boolean(
            0,
            vec![
                (
                    '+',
                    near(vec![st(f, "quick"), st(f, "fox")], 1, false).into(),
                ),
                ('-', tq(f, "cat")),
            ],
        ),
        19 => boolean(
            0,
            vec![
                (' ', phrase(f, &["quick", "brown"]).into()),
                (' ', PrefixQuery::new(f, "qu").into()),
            ],
        ),
        20 => near(
            vec![
                span_mtq(MultiTermSource::Regexp(RegexpQuery::new(f, "la.y"))),
                st(f, "dog"),
            ],
            0,
            true,
        )
        .into(),
        21 => SpanNode::containing(
            near(vec![st(f, "quick"), st(f, "fox")], 2, true),
            st(f, "brown"),
        )
        .unwrap()
        .into(),
        _ => boolean(
            0,
            vec![
                (' ', phrase(f, &["quick", "brown"]).into()),
                (' ', st("pos", "quick").into()),
            ],
        ),
    }
}

/// The documents' snippets as the fixture joins them.
fn joined<T: ToString>(snippets: Vec<Option<T>>) -> String {
    snippets
        .into_iter()
        .map(|s| s.map_or_else(|| "\u{2}".to_string(), |s| s.to_string()))
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

#[test]
fn built_queries_highlight_as_lucenes_do() {
    with_index(|reader, searcher| {
        let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
        let all: Vec<i32> = (0..reader.max_doc()).collect();
        let want = fs::read_to_string(dir().join("uhighlight-built.tsv")).unwrap();
        let (mut checked, mut bad) = (0, Vec::new());
        for row in want.lines() {
            let c: Vec<&str> = row.split('\t').collect();
            let (cfg, qi): (usize, usize) = (c[0].parse().unwrap(), c[1].parse().unwrap());
            let field = c[2];
            let mut uh = UnifiedHighlighter::new(Some(reader), Some(Arc::clone(&analyzer)));
            uh = match cfg {
                0 => uh,
                1 => uh.with_flags(true, true, true, false),
                2 => uh.with_flags(false, true, true, false),
                3 => uh
                    .with_flags(true, true, true, false)
                    .with_field_matcher(Arc::new(|_| true)),
                _ => uh.with_field_matcher(Arc::new(|_| true)),
            };
            let query = built(qi, field);
            let got = match uh.highlight_fields(searcher, &[field], &query, &all, &[2]) {
                Ok(mut m) => format!("R\t{}", joined(m.remove(field).unwrap())),
                Err(e) => format!("X\t{}", e.class()),
            };
            let expected = format!("{}\t{}", c[3], unesc(c[4]));
            checked += 1;
            if got != expected {
                bad.push(format!(
                    "config {cfg} query {qi} field {field}\n  want {}\n  got  {}",
                    expected.replace('\n', "\\n"),
                    got.replace('\n', "\\n")
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {checked} differ:\n{}",
            bad.len(),
            bad.iter().take(8).cloned().collect::<Vec<_>>().join("\n")
        );
        assert_eq!(checked, 575);
    });
}

/// `GenUnifiedHighlight.Snip`.
struct Snip(usize, String);

impl std::fmt::Display for Snip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{}", self.0, self.1)
    }
}

/// `GenUnifiedHighlight.LoggingFormatter`: logs the head of the content of
/// each call, in call order.
#[derive(Default)]
struct LoggingFormatter {
    log: Mutex<Vec<String>>,
    inner: DefaultPassageFormatter,
}

impl PassageFormatter for LoggingFormatter {
    fn format(&self, passages: &[Passage], content: &[u16]) -> String {
        self.format_object(passages, content).to_string()
    }
    fn format_object(&self, passages: &[Passage], content: &[u16]) -> HighlightObject {
        let head = String::from_utf16_lossy(&content[..content.len().min(12)]);
        self.log.lock().unwrap().push(head);
        Box::new(Snip(passages.len(), self.inner.format(passages, content)))
    }
}

const CALL_FIELDS: [&str; 4] = ["tv", "offs", "pos", "both"];
const CALL_MAX_PASSAGES: [i32; 4] = [1, 2, 1, 2];
const CALL_DOCS: [i32; 6] = [5, 0, 3, 0, 7, 1];

fn call_query(i: usize) -> Clause {
    let mut b = BooleanQuery::new();
    for f in CALL_FIELDS {
        b.should.push(if i == 0 {
            tq(f, "fox")
        } else {
            PrefixQuery::new(f, "qu").into()
        });
    }
    b.into()
}

#[test]
fn stored_values_are_batched_and_objects_returned_as_lucene_does() {
    with_index(|reader, searcher| {
        let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
        let want = fs::read_to_string(dir().join("uhighlight-calls.tsv")).unwrap();
        let mut checked = 0;
        for row in want.lines() {
            let c: Vec<&str> = row.split('\t').collect();
            let (threshold, qi): (i32, usize) = (c[0].parse().unwrap(), c[1].parse().unwrap());
            let formatter = Arc::new(LoggingFormatter::default());
            let mut uh = UnifiedHighlighter::new(Some(reader), Some(Arc::clone(&analyzer)))
                .with_formatter(Arc::clone(&formatter) as _);
            if threshold >= 0 {
                uh = uh.with_cache_field_val_chars_threshold(threshold);
            }
            let got = uh
                .highlight_fields_as_objects(
                    searcher,
                    &CALL_FIELDS,
                    &call_query(qi),
                    &CALL_DOCS,
                    &CALL_MAX_PASSAGES,
                )
                .unwrap();
            let log = formatter.log.lock().unwrap().join("\u{1f}");
            assert_eq!(
                log,
                unesc(c[2]),
                "threshold {threshold} query {qi}: the calls"
            );
            let fields: Vec<String> = got
                .into_iter()
                .map(|(field, objects)| {
                    for o in objects.iter().flatten() {
                        assert!(o.as_any().downcast_ref::<Snip>().is_some());
                    }
                    format!("{field}={}", joined(objects))
                })
                .collect();
            let expected: Vec<String> = c[3..].iter().map(|s| unesc(s)).collect();
            assert_eq!(
                fields, expected,
                "threshold {threshold} query {qi}: the objects"
            );
            checked += 1;
        }
        assert_eq!(checked, 6);
    });
}
