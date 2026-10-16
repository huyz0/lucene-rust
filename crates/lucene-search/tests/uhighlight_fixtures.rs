//! M13 T13.2 (`uhighlight`) against Lucene 10.5.0: the unified highlighter's
//! snippets and its break iterators.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::highlight::unified::break_iterators::{
    sentence_instance, word_instance, BreakIterator, CustomSeparatorBreakIterator,
    LengthGoalBreakIterator, SplittingBreakIterator, WholeBreakIterator,
};
use lucene_search::highlight::unified::{DefaultPassageFormatter, UnifiedHighlighter};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::queryparser::classic::QueryParser;

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
