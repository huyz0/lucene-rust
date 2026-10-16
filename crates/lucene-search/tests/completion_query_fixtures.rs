//! M13 T13.3d: the completion queries against an index Lucene wrote.
//! `GenCompletionQuery` indexes three segments (one with deletions) of
//! `SuggestField`s under the default and a separator-dropping
//! `CompletionAnalyzer` and of `ContextSuggestField`s, and records, for every
//! query kind (prefix, regex, fuzzy, and each inside a `ContextQuery`) at four
//! sizes with and without duplicates skipped, with and without a filter, what
//! `SuggestIndexSearcher` answers: key, document, the score's float bits and
//! the context. Every answer must be Lucene's, over Lucene's index and over
//! the same documents written by this port's `IndexWriter`, whose
//! `.lkp`/`.cmp` must equal Lucene's for every field and segment.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::suggest::completion_query::{
    BitsProducer, CompletionAnalyzer, CompletionQuery, ContextQuery, FilterBits,
    FuzzyCompletionQuery, PrefixCompletionQuery, RegexCompletionQuery, SuggestIndexSearcher,
};
use lucene_search::suggest::suggest_field::{CompletionOptions, ContextSuggestField, SuggestField};
use lucene_search::Result;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/completion_query")
}

/// `GenCompletionQuery.EVEN`.
struct Even;

impl BitsProducer for Even {
    fn get_bits(&self, _: &SegmentReader) -> Result<FilterBits> {
        Ok(FilterBits::Bits(Box::new(|doc| doc % 2 == 0)))
    }
}

/// `GenCompletionQuery.NONE`.
struct NoneMatch;

impl BitsProducer for NoneMatch {
    fn get_bits(&self, _: &SegmentReader) -> Result<FilterBits> {
        Ok(FilterBits::MatchNoBits)
    }
}

/// `new StandardAnalyzer(STOPS)`.
fn standard() -> Analyzer {
    let stops: HashSet<String> = ["the", "of"].iter().map(|s| s.to_string()).collect();
    Analyzer::standard(Some(&stops))
}

fn analyzer(preserve_sep: bool) -> CompletionAnalyzer {
    CompletionAnalyzer::with_options(
        Arc::new(standard()),
        preserve_sep,
        true,
        lucene_analysis::miscellaneous::DEFAULT_MAX_GRAPH_EXPANSIONS,
    )
}

fn filter(name: &str) -> Option<Arc<dyn BitsProducer>> {
    match name {
        "even" => Some(Arc::new(Even)),
        "none" => Some(Arc::new(NoneMatch)),
        _ => None,
    }
}

/// `addContexts`: `name`, `name:boost`, `prefix*:boost` or `*all`.
fn add_contexts(query: &mut ContextQuery, specs: &str) {
    for spec in specs.split(',').filter(|s| !s.is_empty()) {
        if spec == "*all" {
            query.add_all_contexts();
            continue;
        }
        let (name, boost) = match spec.split_once(':') {
            Some((n, b)) => (n, Some(b.parse::<f32>().unwrap())),
            None => (spec, None),
        };
        let exact = !name.ends_with('*');
        let name = name.trim_end_matches('*');
        query
            .add_context(name, boost.unwrap_or(1.0), exact)
            .unwrap();
    }
}

fn build(kind: &str, field: &str, text: &str, options: &str) -> CompletionQuery {
    let a = analyzer(field != "nosep");
    match kind {
        "prefix" => CompletionQuery::Prefix(
            PrefixCompletionQuery::new(a, field, text, filter(options)).unwrap(),
        ),
        "regex" => {
            CompletionQuery::Regex(RegexCompletionQuery::new(field, text, filter(options)).unwrap())
        }
        "fuzzy" => {
            // "maxEdits,transpositions,nonFuzzyPrefix,minFuzzyLength,unicodeAware|filter"
            let (opts, filt) = options.split_once('|').unwrap_or((options, ""));
            let q = if opts.is_empty() {
                FuzzyCompletionQuery::new(a, field, text, filter(filt))
            } else {
                let p: Vec<&str> = opts.split(',').collect();
                FuzzyCompletionQuery::with_options(
                    a,
                    field,
                    text,
                    filter(filt),
                    p[0].parse().unwrap(),
                    p[1] == "true",
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4] == "true",
                    lucene_util::automaton::DEFAULT_DETERMINIZE_WORK_LIMIT,
                )
            };
            CompletionQuery::Fuzzy(q.unwrap())
        }
        other => panic!("unknown kind {other}"),
    }
}

/// Every `queries.tsv` row against `dir`.
fn replay(dir: &Path) {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    let searcher = SuggestIndexSearcher::new(&reader);
    let rows = fs::read_to_string(data().join("queries.tsv")).unwrap();
    let mut checked = 0;
    let mut failures: Vec<String> = Vec::new();
    for line in rows.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (kind, field, text, options) = (c[0], c[1], c[2], c[3]);
        let n: usize = c[4].parse().unwrap();
        let dedup = c[5] == "1";
        let want = c.get(6).copied().unwrap_or("");
        let query = match kind {
            "context-prefix" | "context-fuzzy" | "context-regex" => {
                let (specs, filt) = options.split_once('|').unwrap_or((options, ""));
                let inner = match kind {
                    "context-prefix" => build("prefix", field, text, ""),
                    "context-fuzzy" => build("fuzzy", field, text, ""),
                    _ => CompletionQuery::Regex(
                        RegexCompletionQuery::new(field, text, filter(filt)).unwrap(),
                    ),
                };
                let mut q = ContextQuery::new(inner).unwrap();
                add_contexts(&mut q, specs);
                CompletionQuery::Context(q)
            }
            _ => build(kind, field, text, options),
        };
        let got = searcher
            .suggest(&query, n, dedup)
            .unwrap()
            .iter()
            .map(|h| {
                format!(
                    "{}\u{1e}{}\u{1e}{}\u{1e}{}",
                    h.key,
                    h.doc,
                    h.score.to_bits() as i32,
                    h.context.as_deref().unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\u{1f}");
        if got != want {
            failures.push(format!(
                "{kind} {field} {text:?} options={options:?} n={n} dedup={dedup}
 want {want:?}
 got  {got:?}"
            ));
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} differ; first:
{}",
        failures.len(),
        failures[..failures.len().min(6)].join(
            "
"
        )
    );
    assert_eq!(checked, 2960);
}

#[test]
fn every_completion_query_answers_as_lucene_does() {
    replay(&data().join("index"));
}

/// `bytes` with its index header's segment id zeroed and its checksum
/// dropped: what two writers of the same content under different ids share.
fn body(bytes: &[u8]) -> Vec<u8> {
    // magic, codec name "Completion104" (1 + 13), version, then the id
    let id_at = 4 + 14 + 4;
    let mut out = bytes[..bytes.len() - 8].to_vec();
    out[id_at..id_at + 16].fill(0);
    out
}

/// `GenCompletionQuery`'s documents, commits and deletes through this port's
/// `IndexWriter`: plain, separator-dropping and context suggestions, each
/// field's `.lkp`/`.cmp` Lucene's in every segment, and every query answered
/// as over Lucene's index.
#[test]
fn the_same_documents_written_here_are_lucenes_bytes_and_answers() {
    let tmp = TempDir::new("completion-query-write");
    let fs = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(standard())));
    w.set_use_compound_file(false);
    for field in ["suggest", "nosep", "ctx"] {
        w.set_postings_format_for_field(field, Lucene104PostingsFormat::completion104());
    }
    let nosep = CompletionOptions {
        preserve_sep: false,
        ..CompletionOptions::default()
    };
    for line in fs::read_to_string(data().join("docs.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (id, value, weight) = (c[0], c[1], c[2].parse::<i32>().unwrap());
        let contexts: Vec<&str> = c[3].split('\u{1f}').filter(|s| !s.is_empty()).collect();
        let mut d = Document::new();
        d.add(StringField::new("id", id.to_string(), Store::No));
        d.add_boxed(Box::new(
            SuggestField::new("suggest", value, weight).unwrap(),
        ));
        d.add_boxed(Box::new(
            SuggestField::new("nosep", value, weight)
                .unwrap()
                .with_options(nosep),
        ));
        d.add_boxed(Box::new(
            ContextSuggestField::new("ctx", value, weight, &contexts).unwrap(),
        ));
        w.add_fields_document(&d).unwrap();
        if id == "39" || id == "69" {
            w.commit().unwrap();
        }
    }
    for i in (41..70).step_by(3) {
        w.delete_documents_by_term(&[Term::new("id", i.to_string())])
            .unwrap();
    }
    w.commit().unwrap();
    drop(w);
    // `GenCompletionQuery` shares one `Completion104` instance between the
    // three fields, as this port's per-field routing does: one suffix.
    for segment in 0..3 {
        for ext in ["lkp", "cmp"] {
            let name = format!("_{segment}_Completion104_0.{ext}");
            let ours = fs::read(tmp.path().join(&name)).unwrap();
            let lucene = fs::read(data().join("index").join(&name)).unwrap();
            assert_eq!(body(&ours), body(&lucene), "{name}");
        }
        let one = format!("_{segment}_Completion104_1.lkp");
        assert!(!data().join("index").join(&one).exists(), "{one}");
    }
    replay(tmp.path());
}
