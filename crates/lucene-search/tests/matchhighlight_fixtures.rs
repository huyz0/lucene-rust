//! M13 T13.2 (`matchhighlight`) against Lucene 10.5.0 and the JDK it ran on.

use std::fs;
use std::path::PathBuf;

use lucene_search::highlight::WordBreakIterator;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/matchhighlight")
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
                out.push(
                    char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap_or('\u{fffd}'),
                );
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

fn bounds(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect()
}

#[test]
fn word_boundaries_match_the_jdk() {
    let text = fs::read_to_string(dir().join("words.tsv")).unwrap();
    let mut n = 0;
    for line in text.split('\n').filter(|l| !l.is_empty()) {
        let (t, b) = line.split_once('\t').unwrap();
        let t = unesc(t);
        assert_eq!(WordBreakIterator::new(&t).boundaries(), bounds(b), "{t:?}");
        n += 1;
    }
    assert_eq!(n, 16);
}

#[test]
fn random_strings_break_as_the_jdk_breaks_them() {
    let text = fs::read_to_string(dir().join("wordfuzz.tsv")).unwrap();
    let (mut rows, mut bad) = (0, Vec::new());
    for line in text.split('\n').filter(|l| !l.is_empty()) {
        let c: Vec<&str> = line.split('\t').collect();
        let t = unesc(c[0]);
        // lone surrogates survive only as UTF-16: the generator wrote them
        // as the escape `unesc` turns into U+FFFD, so skip those rows
        if c[0].contains("\\ud8") || c[0].contains("\\udc") {
            continue;
        }
        let fresh = || WordBreakIterator::new(&t);
        let mut it = fresh();
        let mut problems = Vec::new();
        let got_bounds = it.boundaries();
        if got_bounds != bounds(c[1]) {
            problems.push(format!("boundaries {got_bounds:?} want {}", c[1]));
        }
        let mut it = fresh();
        let opt = |v: Option<usize>| v.map_or(-1, |x| i64::try_from(x).unwrap());
        for cell in c[2].split(' ').filter(|x| !x.is_empty()) {
            let (off, rest) = cell.split_once(':').unwrap();
            let off: usize = off.parse().unwrap();
            let want = format!(
                "{},{},{}",
                fresh().is_boundary(off),
                opt(fresh().following(off)),
                opt(fresh().preceding(off))
            );
            if want != rest {
                problems.push(format!("offset {off}: got {want} want {rest}"));
            }
        }
        for cell in c[3].split(' ').filter(|x| !x.is_empty()) {
            let k = cell.find(['b', 'f', 'p']).unwrap();
            let off: usize = cell[..k].parse().unwrap();
            let got = match &cell[k..=k] {
                "b" => format!("b{}", it.is_boundary(off)),
                "f" => format!("f{}", opt(it.following(off))),
                _ => format!("p{}", opt(it.preceding(off))),
            };
            if got != cell[k..] {
                problems.push(format!("shared {cell}: got {got}"));
            }
        }
        rows += 1;
        if !problems.is_empty() {
            bad.push(format!("{t:?}: {}", problems.join("; ")));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {rows} differ:\n{}",
        bad.len(),
        bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(rows > 3800, "{rows}");
}

mod highlighting {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use lucene_analysis::{Analyzer, StandardAnalyzer};
    use lucene_search::directory_reader::DirectoryReader;
    use lucene_search::field_norms::FieldNorms;
    use lucene_search::highlight::{
        DocHighlights, FieldValueHighlighters, MatchHighlighter, PassageFormatter,
    };
    use lucene_search::index_searcher::IndexSearcher;
    use lucene_search::query::{BooleanQuery, Clause};
    use lucene_search::queryparser::classic::QueryParser;

    fn render(hits: &[DocHighlights]) -> String {
        hits.iter()
            .map(|dh| {
                let fields: Vec<String> = dh
                    .fields
                    .iter()
                    .map(|(f, v)| format!("{f}:{}", v.join("\u{1f}")))
                    .collect();
                format!("{}={}", dh.doc_id, fields.join("|"))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn highlighter(
        cfg: usize,
        reader: &DirectoryReader,
        analyzer: &Arc<Analyzer>,
    ) -> MatchHighlighter {
        let mut mh = MatchHighlighter::new(reader, Arc::clone(analyzer));
        match cfg {
            0 => {
                mh.append_field_highlighter(FieldValueHighlighters::highlighted(
                    32,
                    2,
                    PassageFormatter::new("...", "<em>", "</em>"),
                    |f| f != "id" && f != "title",
                ));
                mh.append_field_highlighter(FieldValueHighlighters::verbatim_value(&["id"]));
                mh.append_field_highlighter(FieldValueHighlighters::skip_remaining());
            }
            1 => mh.append_field_highlighter(FieldValueHighlighters::highlighted(
                80,
                3,
                PassageFormatter::with_markers(" … ", |r| format!("[{}:", r.from), |_| "]".into()),
                |_| true,
            )),
            _ => {
                mh.append_field_highlighter(FieldValueHighlighters::max_leading_characters(
                    40,
                    "…",
                    ["title".to_string()].into_iter().collect(),
                ));
                mh.append_field_highlighter(FieldValueHighlighters::highlighted(
                    16,
                    4,
                    PassageFormatter::new("~", "(", ")"),
                    |f| f == "offs" || f == "pos",
                ));
                mh.append_field_highlighter(FieldValueHighlighters::verbatim_value(&["id"]));
                mh.append_field_highlighter(FieldValueHighlighters::skip_remaining());
            }
        }
        mh
    }

    #[test]
    fn match_highlighting_equals_lucenes() {
        let reader =
            DirectoryReader::open(&lucene_store::FsDirectory::open(dir().join("index"))).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let names: Vec<String> = ["id", "offs", "pos", "docs", "title"]
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
            fs::read_to_string(dir().join("../../corpus/matchhighlight-queries.txt"))
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect();
        let want = fs::read_to_string(dir().join("matchhighlight.tsv")).unwrap();
        let (mut checked, mut bad) = (0, Vec::new());
        for row in want.lines() {
            let c: Vec<&str> = row.split('\t').collect();
            let (cfg, qi): (usize, usize) = (c[0].parse().unwrap(), c[1].parse().unwrap());
            let clause: Clause = parser
                .parse(&queries[qi])
                .unwrap()
                .to_clause(reader.max_doc());
            let mut bq = BooleanQuery::new();
            bq.must.push(clause.clone().rewrite());
            let top = searcher.search(&bq, 20).unwrap();
            let docs: Vec<i32> = top.score_docs.iter().map(|d| d.doc).collect();
            let mh = highlighter(cfg, &reader, &analyzer);
            let got = match mh.highlight(&searcher, &reader, &docs, &[clause]) {
                Ok(hits) => format!("R\t{}", render(&hits)),
                Err(e) => format!("X\t{e}"),
            };
            let expected = format!("{}\t{}", c[2], unesc(c[3]));
            checked += 1;
            if got != expected {
                bad.push(format!(
                    "config {cfg} query {qi} {:?}\n  want {}\n  got  {}",
                    queries[qi],
                    expected.replace('\n', "\\n"),
                    got.replace('\n', "\\n")
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {checked} differ:\n{}",
            bad.len(),
            bad.iter().take(6).cloned().collect::<Vec<_>>().join("\n")
        );
        assert_eq!(checked, 75);
    }
}
