//! M13 T13.3c against Lucene 10.5.0 (`fixtures/src/GenAnalyzingSuggest.java`):
//! `AnalyzingSuggester` and `FuzzySuggester` answer as Lucene's do over a
//! phrase corpus (stop words, case variants of one analyzed form, a synonym
//! graph, non-ASCII), write Lucene's bytes, and read them back.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::synonym::{SynonymGraphFilter, SynonymMapBuilder};
use lucene_analysis::{
    Analyzer, AnalyzerDefinition, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter,
    TokenStream, TokenStreamComponents,
};
use lucene_search::suggest::analyzing::{
    AnalyzingSuggester, FuzzySuggester, SuggestStopFilter, EXACT_FIRST, PRESERVE_SEP,
};
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::{Lookup, LookupResult};
use lucene_store::{SliceInput, VecDataOutput};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/suggest_analyzing")
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

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The corpus, with or without payloads, as an `InputIterator`.
struct Phrases {
    rows: Vec<(String, i64, String)>,
    payloads: bool,
    next: usize,
}

impl Phrases {
    fn read(payloads: bool) -> Self {
        let text = fs::read_to_string(dir().join("phrases.tsv")).unwrap();
        let rows = text
            .lines()
            .map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                (unesc(c[0]), c[1].parse().unwrap(), c[2].to_string())
            })
            .collect();
        Phrases {
            rows,
            payloads,
            next: 0,
        }
    }

    fn of(rows: Vec<(&str, i64, &str)>, payloads: bool) -> Self {
        Phrases {
            rows: rows
                .into_iter()
                .map(|(s, w, p)| (s.to_string(), w, p.to_string()))
                .collect(),
            payloads,
            next: 0,
        }
    }
}

impl InputIterator for Phrases {
    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row.0.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.rows[self.next - 1].1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.payloads.then(|| self.rows[self.next - 1].2.as_bytes())
    }
    fn has_payloads(&self) -> bool {
        self.payloads
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

fn stops() -> HashSet<String> {
    ["the", "of", "a"].iter().map(|s| s.to_string()).collect()
}

fn standard() -> Arc<Analyzer> {
    Arc::new(Analyzer::standard(Some(&stops())))
}

/// Tokenizer, lower case, stop words, then a one-way synonym graph
/// (`wifi` -> `wireless`, keeping the original).
struct Synonyms {
    stops: Arc<CharArraySet>,
    map: Arc<lucene_analysis::synonym::SynonymMap>,
}

impl AnalyzerDefinition for Synonyms {
    fn create_components(
        &self,
        _field: &str,
    ) -> Result<TokenStreamComponents, lucene_analysis::AnalysisError> {
        let ts = LowerCaseFilter::new(StandardTokenizer::new());
        let ts = StopFilter::new(ts, Arc::clone(&self.stops));
        Ok(TokenStreamComponents::new(SynonymGraphFilter::new(
            ts,
            Arc::clone(&self.map),
            true,
        )))
    }
}

fn synonyms() -> Arc<Analyzer> {
    let mut b = SynonymMapBuilder::new(true);
    b.add("wifi", "wireless", true).unwrap();
    Arc::new(Analyzer::new(Synonyms {
        stops: Arc::new(CharArraySet::from(&stops())),
        map: Arc::new(b.build().unwrap()),
    }))
}

/// Tokenizer, lower case, then `SuggestStopFilter`.
struct SuggestStop {
    stops: Arc<CharArraySet>,
}

impl AnalyzerDefinition for SuggestStop {
    fn create_components(
        &self,
        _field: &str,
    ) -> Result<TokenStreamComponents, lucene_analysis::AnalysisError> {
        let ts = LowerCaseFilter::new(StandardTokenizer::new());
        Ok(TokenStreamComponents::new(SuggestStopFilter::new(
            ts,
            Arc::clone(&self.stops),
        )))
    }
}

fn suggest_stop() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(SuggestStop {
        stops: Arc::new(CharArraySet::from(&stops())),
    }))
}

const BOTH: i32 = EXACT_FIRST | PRESERVE_SEP;

fn an(
    a: Arc<Analyzer>,
    options: i32,
    surfaces: i32,
    graph: i32,
    preserve_pos: bool,
) -> AnalyzingSuggester {
    AnalyzingSuggester::with_options(a.clone(), a, options, surfaces, graph, preserve_pos).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn fz(
    a: Arc<Analyzer>,
    options: i32,
    edits: i32,
    transpositions: bool,
    prefix: i32,
    min: i32,
    unicode: bool,
    preserve_pos: bool,
) -> FuzzySuggester {
    FuzzySuggester::with_options(
        a.clone(),
        a,
        options,
        256,
        -1,
        preserve_pos,
        edits,
        transpositions,
        prefix,
        min,
        unicode,
    )
    .unwrap()
}

fn rendered(res: &[LookupResult]) -> String {
    res.iter()
        .map(|r| {
            format!(
                "{}\u{1e}{}\u{1e}{}",
                r.key,
                r.value,
                r.payload.as_deref().map(hex).unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

/// Every row of a lookup answer file against `lk`; the row count.
fn check_lookups(lk: &dyn Lookup, file: &str) -> usize {
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text.lines() {
        let c: Vec<&str> = line.splitn(4, '\t').collect();
        let query = unesc(c[0]);
        let num: usize = c[1].parse().unwrap();
        assert_eq!(c[2], "R", "{file}: {line}");
        let want: String = c[3]
            .split('\u{1f}')
            .filter(|r| !r.is_empty())
            .map(|r| {
                let f: Vec<&str> = r.split('\u{1e}').collect();
                format!("{}\u{1e}{}\u{1e}{}", unesc(f[0]), f[1], f[2])
            })
            .collect::<Vec<_>>()
            .join("\u{1f}");
        let got = rendered(&lk.lookup_no_contexts(&query, false, num).unwrap());
        checked += 1;
        if got != want {
            bad.push(format!(
                "{file} {query:?} n={num}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked}:\n{}",
        bad.len(),
        bad[..bad.len().min(6)].join("\n")
    );
    checked
}

fn stored(lk: &dyn Lookup) -> (bool, Vec<u8>) {
    let mut out = VecDataOutput::new();
    let ok = lk.store(&mut out).unwrap();
    (ok, out.into_inner())
}

/// Build, answer every query, write the same bytes as Lucene, and answer the
/// same after loading them.
fn check_config(
    name: &str,
    mut lk: impl Lookup,
    mut fresh: impl FnMut() -> Box<dyn Lookup>,
    payloads: bool,
) {
    lk.build(&mut Phrases::read(payloads)).unwrap();
    assert_eq!(check_lookups(&lk, &format!("{name}.tsv")), 480);
    let count: u64 = fs::read_to_string(dir().join(format!("{name}.count")))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(lk.get_count().unwrap(), count, "{name} count");
    let (ok, bytes) = stored(&lk);
    assert!(ok);
    assert_eq!(
        fs::read_to_string(dir().join(format!("{name}.stored"))).unwrap(),
        "true\n"
    );
    assert_eq!(
        bytes,
        fs::read(dir().join(format!("{name}.bin"))).unwrap(),
        "{name} store()"
    );
    let mut again = fresh();
    assert!(again
        .load(&mut SliceInput::new(
            &fs::read(dir().join(format!("{name}.bin"))).unwrap()
        ))
        .unwrap());
    assert_eq!(again.get_count().unwrap(), count);
    assert_eq!(
        check_lookups(again.as_ref(), &format!("{name}_reload.tsv")),
        480
    );
    assert_eq!(
        stored(again.as_ref()).1,
        bytes,
        "{name} store() after load()"
    );
}

#[test]
fn analyzing_default() {
    check_config(
        "a_default",
        AnalyzingSuggester::new(standard()),
        || Box::new(AnalyzingSuggester::new(standard())),
        false,
    );
}

#[test]
fn analyzing_without_separators_or_position_increments() {
    let make = || an(standard(), EXACT_FIRST, 256, -1, false);
    check_config("a_nosep", make(), || Box::new(make()), false);
}

#[test]
fn analyzing_plain_with_two_surface_forms() {
    let make = || an(standard(), 0, 2, -1, true);
    check_config("a_plain", make(), || Box::new(make()), false);
}

#[test]
fn analyzing_with_payloads() {
    let make = || an(standard(), BOTH, 256, -1, true);
    check_config("a_payload", make(), || Box::new(make()), true);
}

#[test]
fn analyzing_a_synonym_graph() {
    let make = || an(synonyms(), BOTH, 256, -1, true);
    check_config("a_syn", make(), || Box::new(make()), false);
}

#[test]
fn analyzing_a_synonym_graph_with_a_path_limit() {
    let make = || an(synonyms(), BOTH, 256, 1, true);
    check_config("a_syn_limit", make(), || Box::new(make()), false);
}

#[test]
fn fuzzy_default() {
    check_config(
        "f_default",
        FuzzySuggester::new(standard()),
        || Box::new(FuzzySuggester::new(standard())),
        false,
    );
}

#[test]
fn fuzzy_unicode_aware_with_two_edits() {
    let make = || fz(standard(), BOTH, 2, false, 0, 2, true, true);
    check_config("f_unicode", make(), || Box::new(make()), false);
}

#[test]
fn fuzzy_without_separators() {
    let make = || fz(standard(), EXACT_FIRST, 1, true, 2, 3, false, false);
    check_config("f_nosep", make(), || Box::new(make()), false);
}

#[test]
fn fuzzy_with_payloads() {
    let make = || fz(standard(), BOTH, 1, true, 1, 3, false, true);
    check_config("f_payload", make(), || Box::new(make()), true);
}

#[test]
fn fuzzy_a_synonym_graph() {
    let make = || fz(synonyms(), BOTH, 1, true, 1, 3, false, true);
    check_config("f_syn", make(), || Box::new(make()), false);
}

#[test]
fn an_empty_build_stores_a_count_and_refuses_to_save() {
    let mut empty = AnalyzingSuggester::new(standard());
    empty.build(&mut Phrases::of(Vec::new(), false)).unwrap();
    let (ok, bytes) = stored(&empty);
    assert!(!ok);
    assert_eq!(bytes, fs::read(dir().join("a_empty.bin")).unwrap());
    assert_eq!(
        fs::read_to_string(dir().join("a_empty.stored")).unwrap(),
        "false\n"
    );
    let got = format!(
        "R\t{}\n",
        rendered(&empty.lookup_no_contexts("a", false, 3).unwrap())
    );
    assert_eq!(
        got,
        fs::read_to_string(dir().join("a_empty.lookup")).unwrap()
    );
}

/// A refusal as `errors.tsv` renders it: `Class\tmessage`, or `ok`.
fn refusal(e: &lucene_store::Error) -> (String, String) {
    let text = e.to_string();
    let class = if text.contains("nsupported") || text.contains("cannot encode") || text == "null" {
        "UnsupportedOperationException"
    } else {
        "IllegalArgumentException"
    };
    (class.to_string(), text)
}

#[test]
fn refuses_what_lucene_refuses() {
    let text = fs::read_to_string(dir().join("errors.tsv")).unwrap();
    let rows: std::collections::HashMap<&str, Vec<&str>> = text
        .lines()
        .map(|l| {
            let mut c = l.split('\t');
            (c.next().unwrap(), c.collect())
        })
        .collect();
    let message = |name: &str| -> String {
        let r = &rows[name];
        unesc(r.last().unwrap())
    };
    let mut ok = AnalyzingSuggester::new(standard());
    ok.build(&mut Phrases::read(false)).unwrap();

    let err = ok.lookup("ne", None, true, 3).unwrap_err();
    assert!(err.to_string().contains(&message("only_more_popular")));
    let ctx: BTreeSet<Vec<u8>> = [b"c".to_vec()].into();
    let err = ok.lookup("ne", Some(&ctx), false, 3).unwrap_err();
    assert!(err.to_string().contains(&message("contexts")));
    for (name, key) in [("hole_key", "a\u{1e}b"), ("sep_key", "a\u{1f}b")] {
        let err = ok.lookup(key, None, false, 3).unwrap_err();
        assert!(err.to_string().contains(&message(name)), "{name}");
    }
    assert!(ok.lookup("a", None, false, 0).unwrap().is_empty());

    // constructor options
    let a = standard();
    let ctor_error = |name: &str, r: Result<AnalyzingSuggester, lucene_store::Error>| match r {
        Ok(_) => assert_eq!(rows[name], ["ok"], "{name}"),
        Err(e) => assert!(
            e.to_string().contains(&message(name)),
            "{name}: {e} vs {rows:?}",
            rows = rows[name]
        ),
    };
    let make = |options, surfaces, graph| {
        AnalyzingSuggester::with_options(a.clone(), a.clone(), options, surfaces, graph, true)
    };
    for options in [4, -1, 3] {
        ctor_error(&format!("options_{options}"), make(options, 256, -1));
    }
    for surfaces in [0, 257, 256, 1] {
        ctor_error(&format!("surfaces_{surfaces}"), make(BOTH, surfaces, -1));
    }
    for graph in [0, -2, -1, 1] {
        ctor_error(&format!("graph_{graph}"), make(BOTH, 256, graph));
    }
    let fuzzy = |edits, prefix, min| {
        FuzzySuggester::with_options(
            a.clone(),
            a.clone(),
            BOTH,
            256,
            -1,
            true,
            edits,
            true,
            prefix,
            min,
            false,
        )
        .map(|_| ())
    };
    for (name, r) in [
        ("fuzzy_edits_3", fuzzy(3, 1, 3)),
        ("fuzzy_edits_-1", fuzzy(-1, 1, 3)),
        ("fuzzy_prefix_-1", fuzzy(1, -1, 3)),
        ("fuzzy_min_-1", fuzzy(1, 1, -1)),
    ] {
        let e = r.unwrap_err();
        assert!(e.to_string().contains(&message(name)), "{name}: {e}");
    }

    // get is unsupported
    let e = ok.get("net").unwrap_err();
    assert_eq!(refusal(&e).0, "UnsupportedOperationException");
    assert_eq!(rows["get"][0], "UnsupportedOperationException");
    assert!(FuzzySuggester::new(standard()).get("net").is_err());

    // build refusals
    for (name, phrases, payloads) in [
        ("negative_weight", vec![("a b", -1, "")], false),
        (
            "big_weight",
            vec![("a b", i64::from(i32::MAX) + 1, "")],
            false,
        ),
        ("sep_in_surface", vec![("a\u{1f}b", 1, "p")], true),
    ] {
        let mut s = AnalyzingSuggester::new(standard());
        let e = s.build(&mut Phrases::of(phrases, payloads)).unwrap_err();
        assert!(e.to_string().contains(&message(name)), "{name}: {e}");
    }
    // without payloads the separator is only a character the analyzer drops
    let mut s = AnalyzingSuggester::new(standard());
    s.build(&mut Phrases::of(vec![("a\u{1f}b", 1, "")], false))
        .unwrap();
    assert_eq!(rows["sep_in_surface_no_payload"][0], "built");
    let got = format!(
        "R\t{}",
        rendered(&s.lookup_no_contexts("a", false, 3).unwrap())
    );
    assert_eq!(
        got,
        unesc(&rows["sep_in_surface_no_payload"][1..].join("\t"))
    );
}

#[test]
fn analyzing_with_a_query_analyzer_that_keeps_a_trailing_stop_word() {
    let make = || {
        AnalyzingSuggester::with_options(standard(), suggest_stop(), BOTH, 256, -1, true).unwrap()
    };
    check_config("a_suggeststop", make(), || Box::new(make()), false);
}

#[test]
fn suggest_stop_filter_keeps_only_a_stop_word_nothing_follows() {
    let text = fs::read_to_string(dir().join("suggest_stop.tsv")).unwrap();
    let analyzer = suggest_stop();
    let mut checked = 0;
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let input = unesc(c[0]);
        let mut tokens = Vec::new();
        let (inc, end) = analyzer
            .try_for_each_token(&input, |_, _, _, _| {})
            .unwrap();
        let mut ts = analyzer.token_stream("", &input).unwrap();
        ts.reset().unwrap();
        while ts.increment_token().unwrap() {
            let a = ts.attributes();
            tokens.push(format!(
                "{},{},{},{},{}",
                a.term(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
                a.is_keyword()
            ));
        }
        ts.end().unwrap();
        let a = ts.attributes();
        assert_eq!(
            (a.position_increment(), a.end_offset()),
            (inc, end),
            "{input:?}: end()"
        );
        assert_eq!(tokens.join("|"), unesc(c[1]), "{input:?}");
        assert_eq!(
            format!("{},{}", a.position_increment(), a.end_offset()),
            c[2],
            "{input:?}: end state"
        );
        checked += 1;
    }
    assert_eq!(checked, 21);
}
