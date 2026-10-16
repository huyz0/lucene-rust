//! M13 T13.2 (`vectorhighlight`) against Lucene 10.5.0: the
//! `FastVectorHighlighter`'s fragments.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::SynonymQuery;
use lucene_search::highlight::unified::break_iterators::sentence_instance;
use lucene_search::highlight::vectorhighlight::{
    BreakIteratorBoundaryScanner, FastVectorHighlighter, FragListBuilder, FragmentsBuilder,
    SimpleBoundaryScanner, SimpleFragListBuilder, SingleFragListBuilder, WeightedFragListBuilder,
    COLORED_POST_TAGS, COLORED_PRE_TAGS,
};
use lucene_search::highlight::{DefaultEncoder, Encoder, SimpleHtmlEncoder};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, MultiPhraseQuery,
    PhraseQuery, TermInSetQuery, TermQuery,
};
use lucene_search::queryparser::classic::QueryParser;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/vhighlight")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
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
    String::from_utf16_lossy(&out)
}

fn tags(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

/// `GenVectorHighlight.built`: the queries the classic parser cannot write.
const BUILT: usize = 10;

fn built(i: usize) -> Clause {
    let t = |s: &str| TermQuery::new("tv", s);
    let p = |terms: &[&str]| PhraseQuery::new("tv", terms.iter().map(|s| s.as_bytes().to_vec()));
    match i {
        0 => DisjunctionMaxQuery::new(
            [Clause::from(t("fox")), Clause::from(p(&["lazy", "dog"]))],
            0.1,
        )
        .into(),
        1 => SynonymQuery::new("tv", [(b"fox".to_vec(), 1.0), (b"dog".to_vec(), 0.5)])
            .unwrap()
            .into(),
        2 => ConstantScoreQuery::new(t("quick"), 1.0).into(),
        3 => MultiPhraseQuery::new("tv", [vec![b"quick".to_vec(), b"lazy".to_vec()]]).into(),
        4 => MultiPhraseQuery::new("tv", [vec![b"quick".to_vec()], vec![b"brown".to_vec()]]).into(),
        5 => TermInSetQuery::new("tv", [b"fox".to_vec(), b"dog".to_vec(), b"zzz".to_vec()]).into(),
        6 => {
            let mut b = BooleanQuery::new();
            b.should.push(t("fox").into());
            b.must_not.push(t("cat").into());
            BoostQuery::new(b, 2.0).into()
        }
        7 => p(&["fox"]).into(),
        8 => {
            let mut b = BooleanQuery::new();
            b.should
                .push(PhraseQuery::new("tvsyn", ["the", "quick"]).into());
            b.should
                .push(PhraseQuery::new("tvsyn", ["fox", "qb"]).into());
            b.into()
        }
        _ => {
            let mut b = BooleanQuery::new();
            b.should
                .push(BoostQuery::new(p(&["quick", "brown"]), 2.0).into());
            b.should.push(p(&["brown", "fox"]).with_slop(1).into());
            b.should.push(p(&["brown", "fox"]).into());
            b.into()
        }
    }
}

/// One configuration of `GenVectorHighlight`.
struct Config {
    fvh: FastVectorHighlighter,
    frag_list: Box<dyn FragListBuilder>,
    fragments: FragmentsBuilder,
    size: i32,
    max: i32,
    pre: Vec<String>,
    post: Vec<String>,
    encoder: Box<dyn Encoder>,
    matched: Option<Vec<&'static str>>,
}

fn config(cfg: usize) -> Config {
    let mut c = Config {
        fvh: FastVectorHighlighter::default(),
        frag_list: Box::new(SimpleFragListBuilder::new()),
        fragments: FragmentsBuilder::score_order(),
        size: 30,
        max: 3,
        pre: tags(&["<b>"]),
        post: tags(&["</b>"]),
        encoder: Box::new(DefaultEncoder),
        matched: None,
    };
    match cfg {
        0 => {}
        1 => {
            c.fvh = FastVectorHighlighter::new(false, false);
            c.size = 20;
            c.max = 2;
        }
        2 => {
            c.frag_list = Box::new(WeightedFragListBuilder::with_margin(3).unwrap());
            let mut fb = FragmentsBuilder::simple();
            fb.set_discrete_multi_value_highlighting(true);
            c.fragments = fb;
            c.pre = tags(&COLORED_PRE_TAGS);
            c.post = tags(&COLORED_POST_TAGS);
            c.size = 25;
            c.max = 4;
        }
        3 => {
            c.fvh = FastVectorHighlighter::new(true, false);
            c.frag_list = Box::new(SingleFragListBuilder);
            c.fragments = FragmentsBuilder::score_order().with_boundary_scanner(Box::new(
                BreakIteratorBoundaryScanner::new(sentence_instance()),
            ));
            c.max = 1;
        }
        4 => {
            c.fragments = FragmentsBuilder::score_order().with_boundary_scanner(Box::new(
                SimpleBoundaryScanner::with_chars(5, &[u16::from(b','), u16::from(b'.'), 32]),
            ));
            c.encoder = Box::new(SimpleHtmlEncoder);
            c.matched = Some(vec!["tv", "tvb"]);
        }
        5 => {
            c.fvh.set_phrase_limit(2);
            c.frag_list = Box::new(SimpleFragListBuilder::with_margin(0).unwrap());
            let mut fb = FragmentsBuilder::simple();
            fb.set_multi_valued_separator(u16::from(b'|'));
            c.fragments = fb;
            c.encoder = Box::new(SimpleHtmlEncoder);
            c.pre = tags(&["["]);
            c.post = tags(&["]"]);
            c.size = 10;
            c.max = 5;
        }
        7 => {
            let mut fb = FragmentsBuilder::simple();
            fb.set_discrete_multi_value_highlighting(true);
            c.fragments = fb;
            c.size = 60;
            c.max = 4;
        }
        _ => {
            c.fvh = FastVectorHighlighter::new(false, true);
            c.frag_list = Box::new(WeightedFragListBuilder::new());
            c.size = 40;
            c.max = 2;
        }
    }
    c
}

#[test]
fn fast_vector_highlighting_equals_lucenes() {
    let reader =
        DirectoryReader::open(&lucene_store::FsDirectory::open(dir().join("index"))).unwrap();
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = QueryParser::new("tv", Arc::clone(&analyzer));
    let queries: Vec<String> =
        fs::read_to_string(dir().join("../../corpus/vhighlight-queries.txt"))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
    let want = fs::read_to_string(dir().join("vhighlight.tsv")).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    let mut cache: Option<(usize, usize, String, Config, _)> = None;
    for row in want.lines() {
        let c: Vec<&str> = row.split('\t').collect();
        let cfg: usize = c[0].parse().unwrap();
        let qi: usize = c[1].parse().unwrap();
        let (field, doc): (&str, i32) = (c[2], c[3].parse().unwrap());
        let text = queries
            .get(qi)
            .map_or_else(String::new, |q| q.replace("F:", &format!("{field}:")));
        if cache
            .as_ref()
            .is_none_or(|(a, b, f, _, _)| (*a, *b, f.as_str()) != (cfg, qi, field))
        {
            let clause = if qi >= queries.len() {
                assert!(qi < queries.len() + BUILT);
                built(qi - queries.len())
            } else {
                parser.parse(&text).unwrap().to_clause(reader.max_doc())
            };
            let conf = config(cfg);
            let fq = conf.fvh.get_field_query(&clause, Some(&reader)).unwrap();
            cache = Some((cfg, qi, field.to_string(), conf, fq));
        }
        let (_, _, _, conf, fq) = cache.as_mut().unwrap();
        let result = match &conf.matched {
            Some(matched) => conf.fvh.get_best_fragments_matched(
                fq,
                &reader,
                doc,
                field,
                matched,
                conf.size,
                conf.max,
                &*conf.frag_list,
                &mut conf.fragments,
                &conf.pre,
                &conf.post,
                &*conf.encoder,
            ),
            None => conf.fvh.get_best_fragments_with(
                fq,
                &reader,
                doc,
                field,
                conf.size,
                conf.max,
                &*conf.frag_list,
                &mut conf.fragments,
                &conf.pre,
                &conf.post,
                &*conf.encoder,
            ),
        };
        let got = match result {
            Ok(None) => "N\t".to_string(),
            Ok(Some(f)) => format!("R\t{}", f.join("\u{1f}")),
            Err(_) => "X\t".to_string(),
        };
        let expected = format!("{}\t{}", c[4], unesc(c.get(5).copied().unwrap_or("")));
        checked += 1;
        if got != expected {
            bad.push(format!(
                "config {cfg} query {qi} {text:?} field {field} doc {doc}\n  want {}\n  got  {}",
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
    assert!(checked > 0);
}
