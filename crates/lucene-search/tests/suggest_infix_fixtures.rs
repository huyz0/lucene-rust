//! M13 T13.3c against Lucene 10.5.0 (`fixtures/src/GenInfixSuggest.java`):
//! `AnalyzingInfixSuggester` and `BlendedInfixSuggester` answer as Lucene's do
//! over a titled, weighted, payloaded and contexted corpus, for each lookup
//! flavour (contexts as a set or as an occur map, `allTermsRequired`, highlight),
//! through the add/update/refresh/commit flow, and on their error paths.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::query_visitor::Occur;
use lucene_search::suggest::infix::{
    AnalyzingInfixSuggester, BlendedInfixSuggester, BlenderType, DEFAULT_CLOSE_INDEXWRITER_ON_BUILD,
};
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::{Lookup, LookupResult};
use lucene_store::{SliceInput, VecDataOutput};

type Answer = lucene_store::Result<Vec<LookupResult>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/suggest_infix")
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

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn set(names: &[&str]) -> BTreeSet<Vec<u8>> {
    names.iter().map(|n| n.as_bytes().to_vec()).collect()
}

fn info(pairs: &[(&str, Occur)]) -> BTreeMap<Vec<u8>, Occur> {
    pairs
        .iter()
        .map(|(c, o)| (c.as_bytes().to_vec(), *o))
        .collect()
}

/// One result per `\u{1f}`-separated record, with the same fields and
/// separators `GenInfixSuggest.results` writes.
fn rendered(res: &[LookupResult]) -> String {
    res.iter()
        .map(|r| {
            let contexts = r.contexts.as_ref().map_or_else(
                || "~null".to_string(),
                |s| s.iter().map(|c| hex(c)).collect::<Vec<_>>().join(","),
            );
            format!(
                "{}\u{1e}{}\u{1e}{}\u{1e}{}\u{1e}{}",
                r.key,
                r.highlight_key.as_deref().unwrap_or("~null"),
                r.value,
                r.payload
                    .as_deref()
                    .map_or_else(|| "~null".to_string(), hex),
                contexts
            )
        })
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

fn plain() -> Arc<Analyzer> {
    Arc::new(Analyzer::standard(None))
}

fn stopping() -> Arc<Analyzer> {
    let stops: HashSet<String> = ["the", "of", "a"].iter().map(|s| s.to_string()).collect();
    Arc::new(Analyzer::standard(Some(&stops)))
}

/// One corpus row: `text`, `weight`, `payload`, `contexts`.
#[derive(Clone)]
struct Item {
    text: String,
    weight: i64,
    payload: Vec<u8>,
    contexts: BTreeSet<Vec<u8>>,
}

/// `GenInfixSuggest.Entries`: the corpus as an `InputIterator`.
struct Entries {
    items: Vec<Item>,
    has_payloads: bool,
    has_contexts: bool,
    next: usize,
    current: usize,
}

impl Entries {
    fn corpus(has_payloads: bool, has_contexts: bool) -> Self {
        let text = fs::read_to_string(dir().join("corpus.tsv")).unwrap();
        let items = text
            .lines()
            .map(|line| {
                let f: Vec<&str> = line.split('\t').collect();
                Item {
                    text: unesc(f[0]),
                    weight: f[1].parse().unwrap(),
                    payload: f[2].as_bytes().to_vec(),
                    contexts: if f[3].is_empty() {
                        BTreeSet::new()
                    } else {
                        f[3].split(',').map(|c| c.as_bytes().to_vec()).collect()
                    },
                }
            })
            .collect();
        Entries {
            items,
            has_payloads,
            has_contexts,
            next: 0,
            current: 0,
        }
    }

    fn head(mut self, n: usize) -> Self {
        self.items.truncate(n);
        self
    }
}

impl InputIterator for Entries {
    fn next(&mut self) -> Option<Vec<u8>> {
        let item = self.items.get(self.next)?;
        self.current = self.next;
        self.next += 1;
        Some(item.text.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.items[self.current].weight
    }
    fn payload(&self) -> Option<&[u8]> {
        self.has_payloads
            .then(|| self.items[self.current].payload.as_slice())
    }
    fn has_payloads(&self) -> bool {
        self.has_payloads
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.has_contexts
            .then(|| &self.items[self.current].contexts)
    }
    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

/// The suggester a grid configuration names.
enum Target {
    Plain(AnalyzingInfixSuggester),
    Blended(BlendedInfixSuggester),
}

impl Target {
    fn build(&mut self, input: &mut Entries) -> lucene_store::Result<()> {
        match self {
            Target::Plain(s) => s.build(input),
            Target::Blended(b) => b.build(input),
        }
    }

    fn count(&self) -> u64 {
        match self {
            Target::Plain(s) => s.get_count().unwrap(),
            Target::Blended(b) => b.get_count().unwrap(),
        }
    }

    /// The `lookup(key, contexts, onlyMorePopular, num)` every grid row ends in
    /// `default` / `red` form.
    fn lookup_default(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        num: usize,
    ) -> Answer {
        match self {
            Target::Plain(s) => s.lookup(key, contexts, false, num),
            Target::Blended(b) => b.lookup(key, contexts, false, num),
        }
    }

    fn lookup_with(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        num: usize,
        all: bool,
        highlight: bool,
    ) -> Answer {
        match self {
            Target::Plain(s) => s.lookup_with(key, contexts, num, all, highlight),
            Target::Blended(b) => b.lookup_with(key, contexts, num, all, highlight),
        }
    }

    /// The occur-map lookup.
    fn lookup_info(
        &self,
        key: &str,
        context_info: &BTreeMap<Vec<u8>, Occur>,
        num: usize,
        all: bool,
        highlight: bool,
    ) -> Answer {
        match self {
            Target::Plain(s) => s.lookup_context_info(key, context_info, num, all, highlight),
            Target::Blended(b) => b.lookup_context_info(key, context_info, num, all, highlight),
        }
    }

    /// One row of a grid file: the variant it names.
    fn answer(&self, variant: &str, key: &str, num: usize) -> Answer {
        let red = set(&["red"]);
        let red_blue = set(&["red", "blue"]);
        let green = set(&["green"]);
        let none = set(&["none"]);
        match variant {
            "default" => self.lookup_default(key, None, num),
            "any" => self.lookup_with(key, None, num, false, true),
            "all_nohl" => self.lookup_with(key, None, num, true, false),
            "red" => self.lookup_default(key, Some(&red), num),
            "red_blue_any" => self.lookup_with(key, Some(&red_blue), num, false, true),
            "green_all" => self.lookup_with(key, Some(&green), num, true, false),
            "none" => self.lookup_with(key, Some(&none), num, true, true),
            "must_not_blue" => {
                self.lookup_info(key, &info(&[("blue", Occur::MustNot)]), num, true, true)
            }
            "must_not_blue_any" => {
                self.lookup_info(key, &info(&[("blue", Occur::MustNot)]), num, false, true)
            }
            "red_must_blue_not" => self.lookup_info(
                key,
                &info(&[("red", Occur::Must), ("blue", Occur::MustNot)]),
                num,
                true,
                false,
            ),
            "red_should_green_should_any" => self.lookup_info(
                key,
                &info(&[("red", Occur::Should), ("green", Occur::Should)]),
                num,
                false,
                true,
            ),
            "empty_info" => self.lookup_info(key, &info(&[]), num, true, true),
            other => panic!("unknown variant {other}"),
        }
    }
}

/// The configuration a grid file names, as `GenInfixSuggest.main` builds it.
fn make(name: &str) -> Target {
    let infix = |index, query, min_prefix, all, highlight| {
        Target::Plain(
            AnalyzingInfixSuggester::with_options(
                index,
                query,
                min_prefix,
                false,
                all,
                highlight,
                DEFAULT_CLOSE_INDEXWRITER_ON_BUILD,
            )
            .unwrap(),
        )
    };
    let blended = |blender, factor, exponent, min_prefix| {
        Target::Blended(
            BlendedInfixSuggester::with_options(
                plain(),
                plain(),
                min_prefix,
                blender,
                factor,
                exponent,
                false,
                true,
                true,
            )
            .unwrap(),
        )
    };
    match name {
        "infix_default" => Target::Plain(AnalyzingInfixSuggester::new(plain())),
        "infix_prefix0" => infix(plain(), plain(), 0, true, true),
        "infix_prefix1" => infix(plain(), plain(), 1, true, true),
        "infix_prefix2" => infix(plain(), plain(), 2, true, true),
        "infix_prefix9" => infix(plain(), plain(), 9, true, true),
        "infix_any_nohl" => infix(plain(), plain(), 3, false, false),
        "infix_stop" => infix(stopping(), stopping(), 4, true, true),
        "infix_mixed" => infix(plain(), stopping(), 2, true, true),
        "blend_linear" => blended(BlenderType::PositionLinear, 10, None, 4),
        "blend_reciprocal" => blended(BlenderType::PositionReciprocal, 10, None, 4),
        "blend_exp" => blended(BlenderType::PositionExponentialReciprocal, 10, None, 4),
        "blend_exp15" => blended(BlenderType::PositionExponentialReciprocal, 3, Some(1.5), 2),
        "blend_custom" => blended(BlenderType::Custom, 5, None, 4),
        "blend_factor1" => blended(BlenderType::PositionLinear, 1, None, 4),
        other => panic!("no configuration {other}"),
    }
}

/// `name\trest` rows of a fixture file, `rest` kept raw.
fn fixture_rows(file: &str) -> BTreeMap<String, String> {
    fs::read_to_string(dir().join(file))
        .unwrap()
        .lines()
        .map(|line| {
            let (name, rest) = line.split_once('\t').unwrap();
            (name.to_string(), rest.to_string())
        })
        .collect()
}

/// Asserts a named row of `rows` is `got`, both as `unesc`-ed text.
fn expect_row(rows: &BTreeMap<String, String>, name: &str, got: String) {
    let want = rows.get(name).unwrap_or_else(|| panic!("no row {name}"));
    assert_eq!(unesc(want), got, "{name}");
}

/// Asserts a named row of `rows` is an error whose text contains the message
/// the fixture recorded (`X\tClass\tmessage`).
fn expect_failure<T>(rows: &BTreeMap<String, String>, name: &str, got: lucene_store::Result<T>) {
    let want: Vec<&str> = rows[name].splitn(3, '\t').collect();
    assert_eq!(want[0], "X", "{name}");
    let message = unesc(want[2]);
    match got {
        Err(e) => assert!(e.to_string().contains(&message), "{name}: {e} vs {message}"),
        Ok(_) => panic!("{name}: expected {message}"),
    }
}

/// Every row of a lookup grid: the variant, the query, the size, then the
/// answer, checked against Lucene's.
fn check_grid(name: &str) -> usize {
    let queries = fs::read_to_string(dir().join("queries.tsv"))
        .unwrap()
        .lines()
        .count();
    let mut target = make(name);
    target.build(&mut Entries::corpus(true, true)).unwrap();
    let count: u64 = fs::read_to_string(dir().join(format!("{name}.count")))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(target.count(), count, "{name} count");

    let text = fs::read_to_string(dir().join(format!("{name}.tsv"))).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text.lines() {
        let c: Vec<&str> = line.splitn(4, '\t').collect();
        let (variant, query, num) = (c[0], unesc(c[1]), c[2].parse::<usize>().unwrap());
        let rest = c[3];
        let got = target.answer(variant, &query, num);
        checked += 1;
        let agrees = if let Some(message) = rest.strip_prefix("X\t") {
            let message = unesc(message.split_once('\t').unwrap().1);
            matches!(&got, Err(e) if e.to_string().contains(&message))
        } else {
            let want = unesc(rest.strip_prefix("R\t").unwrap());
            matches!(&got, Ok(res) if rendered(res) == want)
        };
        if !agrees {
            bad.push(format!(
                "{variant} {query:?} n={num}: want {rest:?}, got {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{name}: {} of {checked}:\n{}",
        bad.len(),
        bad[..bad.len().min(6)].join("\n")
    );
    assert_eq!(checked, queries * 3 * 12, "{name} rows");
    checked
}

#[test]
fn default_infix_suggester_matches_lucene() {
    check_grid("infix_default");
}

#[test]
fn infix_suggester_with_prefix_limits_matches_lucene() {
    for name in [
        "infix_prefix0",
        "infix_prefix1",
        "infix_prefix2",
        "infix_prefix9",
    ] {
        check_grid(name);
    }
}

#[test]
fn infix_suggester_without_highlighting_or_all_terms_matches_lucene() {
    check_grid("infix_any_nohl");
}

#[test]
fn infix_suggester_with_stop_words_matches_lucene() {
    check_grid("infix_stop");
    check_grid("infix_mixed");
}

#[test]
fn blended_suggesters_match_lucene() {
    for name in [
        "blend_linear",
        "blend_reciprocal",
        "blend_exp",
        "blend_exp15",
        "blend_custom",
        "blend_factor1",
    ] {
        check_grid(name);
    }
}

#[test]
fn near_real_time_flow_matches_lucene() {
    let rows = fixture_rows("nrt.tsv");
    let red = set(&["red"]);
    let blue = set(&["blue"]);
    let mut s = AnalyzingInfixSuggester::new(plain());

    s.add(b"early bird", Some(&red), 5, Some(b"e")).unwrap();
    expect_row(
        &rows,
        "added_before_build",
        format!("R\t{}", s.get_count().unwrap()),
    );
    let before = rendered(&s.lookup("early", None, false, 5).unwrap());
    expect_row(&rows, "lookup_unrefreshed", format!("R\t{before}"));
    s.refresh().unwrap();
    expect_row(&rows, "refresh", format!("R\t{}", s.get_count().unwrap()));
    let found = rendered(&s.lookup("early", None, false, 5).unwrap());
    expect_row(&rows, "lookup_refreshed", format!("R\t{found}"));

    s.add(b"early riser", None, 9, Some(b"r")).unwrap();
    s.add(b"early start", None, 9, None).unwrap();
    s.refresh().unwrap();
    expect_row(
        &rows,
        "add_no_contexts",
        format!("R\t{}", s.get_count().unwrap()),
    );
    let second = rendered(&s.lookup("early", None, false, 5).unwrap());
    expect_row(&rows, "lookup_second_segment", format!("R\t{second}"));

    s.update(b"early bird", Some(&blue), 50, Some(b"e2"))
        .unwrap();
    s.refresh().unwrap();
    expect_row(&rows, "update", format!("R\t{}", s.get_count().unwrap()));
    let updated = rendered(&s.lookup("early", None, false, 5).unwrap());
    expect_row(&rows, "lookup_updated", format!("R\t{updated}"));
    let updated_blue = rendered(&s.lookup("early", Some(&blue), false, 5).unwrap());
    expect_row(&rows, "lookup_updated_blue", format!("R\t{updated_blue}"));
    s.commit().unwrap();
    expect_row(&rows, "commit", "R\tok".to_string());

    s.build(&mut Entries::corpus(true, true)).unwrap();
    expect_row(&rows, "rebuilt_count", s.get_count().unwrap().to_string());
    let rebuilt = rendered(&s.lookup("early", None, false, 5).unwrap());
    expect_row(&rows, "rebuilt_lookup", format!("R\t{rebuilt}"));
    let apple = rendered(&s.lookup("app", None, false, 5).unwrap());
    expect_row(&rows, "rebuilt_lookup_apple", format!("R\t{apple}"));
    s.commit().unwrap();
    expect_row(&rows, "commit_after_close_on_build", "R\tok".to_string());
    s.refresh().unwrap();
    expect_row(
        &rows,
        "refresh_after_build",
        format!("R\t{}", s.get_count().unwrap()),
    );
    s.close();
    expect_row(&rows, "closed_count", s.get_count().unwrap().to_string());
}

#[test]
fn payloads_outlive_the_documents_that_brought_them() {
    let rows = fixture_rows("payloads.tsv");
    let mut p = AnalyzingInfixSuggester::new(plain());
    let one = |p: &AnalyzingInfixSuggester| {
        format!("R\t{}", rendered(&p.lookup("one", None, false, 5).unwrap()))
    };
    p.add(b"xray one", None, 3, Some(b"px")).unwrap();
    p.add(b"yankee one", None, 2, None).unwrap();
    p.refresh().unwrap();
    expect_row(&rows, "mixed", one(&p));
    p.update(b"xray one", None, 3, None).unwrap();
    p.refresh().unwrap();
    expect_row(&rows, "payload_deleted", one(&p));
    p.update(b"yankee one", None, 2, None).unwrap();
    p.refresh().unwrap();
    expect_row(&rows, "first_segment_gone", one(&p));
}

#[test]
fn open_writer_sees_added_documents_only_after_refresh() {
    let rows = fixture_rows("open_writer.tsv");
    let mut open =
        AnalyzingInfixSuggester::with_options(plain(), plain(), 4, true, true, true, false)
            .unwrap();
    open.build(&mut Entries::corpus(true, true).head(5))
        .unwrap();
    expect_row(&rows, "count", open.get_count().unwrap().to_string());
    open.add(b"lending library", None, 1000, Some(b"lib"))
        .unwrap();
    let before = rendered(&open.lookup("lend", None, false, 3).unwrap());
    expect_row(&rows, "before_refresh", format!("R\t{before}"));
    open.refresh().unwrap();
    let after = rendered(&open.lookup("lend", None, false, 3).unwrap());
    expect_row(&rows, "after_refresh", format!("R\t{after}"));
    open.commit().unwrap();
    expect_row(&rows, "commit", "ok".to_string());
}

#[test]
fn error_paths_match_lucene() {
    let rows = fixture_rows("errors.tsv");
    let mut unbuilt = AnalyzingInfixSuggester::new(plain());
    expect_failure(&rows, "unbuilt_lookup", unbuilt.lookup("a", None, false, 3));
    expect_failure(&rows, "unbuilt_refresh", unbuilt.refresh());
    expect_failure(&rows, "unbuilt_commit", unbuilt.commit());
    expect_row(
        &rows,
        "unbuilt_count",
        unbuilt.get_count().unwrap().to_string(),
    );

    expect_failure(
        &rows,
        "negative_min_prefix",
        AnalyzingInfixSuggester::with_options(plain(), plain(), -1, false, true, true, true)
            .map(|_| ()),
    );

    let mut ok = AnalyzingInfixSuggester::new(plain());
    ok.build(&mut Entries::corpus(true, true)).unwrap();
    // `num` is a `usize` here, so Java's `numHits = -1` cannot be passed: it
    // takes the same `numHits <= 0` path as `num_0`.
    expect_failure(&rows, "num_0", ok.lookup("apple", None, false, 0));
    expect_row(
        &rows,
        "count_after_build",
        ok.get_count().unwrap().to_string(),
    );

    let mut out = VecDataOutput::new();
    expect_row(
        &rows,
        "store",
        format!("R\t{}", ok.store(&mut out).unwrap()),
    );
    expect_row(
        &rows,
        "load",
        format!("R\t{}", ok.load(&mut SliceInput::new(&[])).unwrap()),
    );

    let mut blended = BlendedInfixSuggester::new(plain());
    blended.build(&mut Entries::corpus(true, true)).unwrap();
    expect_failure(
        &rows,
        "blended_num_0",
        blended.lookup("apple", None, false, 0),
    );

    let mut bare = AnalyzingInfixSuggester::new(plain());
    bare.build(&mut Entries::corpus(false, false)).unwrap();
    expect_row(
        &rows,
        "bare",
        format!(
            "R\t{}",
            rendered(&bare.lookup("apple", None, false, 3).unwrap())
        ),
    );
    let red = set(&["red"]);
    expect_row(
        &rows,
        "bare_ctx",
        format!(
            "R\t{}",
            rendered(&bare.lookup("apple", Some(&red), false, 3).unwrap())
        ),
    );
}
