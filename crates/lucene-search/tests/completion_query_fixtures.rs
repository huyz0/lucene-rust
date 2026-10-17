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
use lucene_search::suggest::document::{merge, SuggestScoreDoc};
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

/// `GenCompletionQuery.hits`: `key \u{1e} doc \u{1e} score bits \u{1e}
/// context`, `\u{1f}`-separated.
fn format_hits(hits: &[SuggestScoreDoc]) -> String {
    hits.iter()
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
        .join("\u{1f}")
}

/// Every `queries.tsv` row of `expected` against the index in `dir`.
fn replay(dir: &Path, expected: &Path) {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    let searcher = SuggestIndexSearcher::new(&reader);
    let rows = fs::read_to_string(expected.join("queries.tsv")).unwrap();
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
        let got = format_hits(&searcher.suggest(&query, n, dedup).unwrap());
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
    replay(&data().join("index"), &data());
}

/// `weights.tsv`: every segment's `CompletionTerms` weights and type.
fn check_weights(dir: &Path, expected: &Path) {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let segments = reader.segment_readers();
    let mut rows = 0;
    for line in fs::read_to_string(expected.join("weights.tsv"))
        .unwrap()
        .lines()
    {
        let c: Vec<&str> = line.split('\t').collect();
        let segment = &segments[c[0].parse::<usize>().unwrap()];
        let got = match (
            segment.completion_weights(c[1]),
            segment.completion_type(c[1]),
        ) {
            (Some((min, max)), Some(t)) => format!("{min}\t{max}\t{t}"),
            (None, None) => "-".to_string(),
            other => panic!("{line}: {other:?}"),
        };
        assert_eq!(got, c[2..].join("\t"), "{line}");
        rows += 1;
    }
    assert_eq!(rows, 12);
}

/// `merges.tsv`: `TopSuggestDocs.merge` of three answers, one per field.
fn check_merges(dir: &Path, expected: &Path) {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
    let searcher = SuggestIndexSearcher::new(&reader);
    let mut rows = 0;
    for line in fs::read_to_string(expected.join("merges.tsv"))
        .unwrap()
        .lines()
    {
        let c: Vec<&str> = line.split('\t').collect();
        let (p, n) = (c[0], c[1].parse::<usize>().unwrap());
        let shards = [
            searcher
                .suggest(&build("prefix", "suggest", p, ""), 10, false)
                .unwrap(),
            searcher
                .suggest(&build("prefix", "nosep", p, ""), 5, true)
                .unwrap(),
            searcher
                .suggest(&build("fuzzy", "ctx", p, ""), 10, false)
                .unwrap(),
        ];
        let shards: Vec<&[SuggestScoreDoc]> = shards.iter().map(Vec::as_slice).collect();
        let merged = merge(n, &shards);
        assert_eq!(merged.len().to_string(), c[2], "{line}");
        assert_eq!(c[3], "EQUAL_TO");
        assert_eq!(
            format_hits(&merged),
            c.get(4).copied().unwrap_or(""),
            "{line}"
        );
        rows += 1;
    }
    assert_eq!(rows, 35);
}

#[test]
fn completion_terms_weights_are_lucenes() {
    check_weights(&data().join("index"), &data());
}

#[test]
fn merged_shard_answers_are_lucenes() {
    check_merges(&data().join("index"), &data());
}

/// `fixtures/bwc/BwcCompletion.java`: the same corpus written by Lucene 9.0,
/// 9.11, 9.12, 10.2 and 10.3 under their own `CompletionXXPostingsFormat`
/// and delegate postings format (`Completion101` over two), and Lucene
/// 10.5.0's answers over it.
#[test]
fn older_completion_formats_answer_as_lucene_does() {
    let root = data().join("../bwc-completion");
    for (version, format, delegate) in [
        ("9.0.0", "Completion90", "Lucene90"),
        ("9.11.1", "Completion99", "Lucene99"),
        ("9.12.2", "Completion912", "Lucene912"),
        ("10.2.2", "Completion101", "Lucene101"),
        ("10.3.2", "Completion101", "Lucene103"),
    ] {
        let expected = root.join(version);
        let written_by = fs::read_to_string(expected.join("written_by.txt")).unwrap();
        assert_eq!(written_by, format!("{version}\t{format}\n"));
        let index = expected.join("index");
        let reader = DirectoryReader::open(&FsDirectory::open(&index)).unwrap();
        for segment in reader.segment_readers() {
            let info = segment.field_infos().field_by_name("suggest").unwrap();
            assert!(
                info.attributes
                    .iter()
                    .any(|(k, v)| k == "PerFieldPostingsFormat.format" && v == format),
                "{version}: {:?}",
                info.attributes
            );
        }
        // The `.cmp` names the delegate its fields' postings are in, after
        // the index header: magic, codec name, version, id, suffix
        // (`<format>_0`).
        let cmp = fs::read(index.join(format!("_0_{format}_0.cmp"))).unwrap();
        let header = 4 + 1 + format.len() + 4 + 16 + 1 + format.len() + 2;
        assert_eq!(cmp[header] as usize, delegate.len(), "{version}");
        assert_eq!(
            &cmp[header + 1..header + 1 + delegate.len()],
            delegate.as_bytes()
        );
        replay(&index, &expected);
        check_weights(&index, &expected);
        check_merges(&index, &expected);
    }
}

/// Every older index force-merged by this port's `IndexWriter` into one
/// `Completion104` segment answers every query with Lucene's hits over the
/// unmerged index: the same keys, scores and contexts (the documents are
/// renumbered, so not the same ids). Only unfiltered answers that are not
/// cut off by `n` are compared, since a filter and a cut through a tie both
/// depend on the doc ids.
#[test]
fn older_completion_segments_merge_into_the_current_format() {
    let root = data().join("../bwc-completion");
    for version in ["9.0.0", "9.11.1", "9.12.2", "10.2.2", "10.3.2"] {
        let src = root.join(version).join("index");
        let tmp = TempDir::new("completion-bwc-merge");
        for entry in fs::read_dir(&src).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
        let fs_dir = FsDirectory::open(tmp.path());
        let current = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&fs_dir, Vec::new(), "Lucene104", current).unwrap();
        for field in ["suggest", "nosep", "ctx"] {
            w.set_postings_format_for_field(field, Lucene104PostingsFormat::completion104());
        }
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        drop(w);
        let reader = DirectoryReader::open(&fs_dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 1, "{version}");
        assert!(
            tmp.path().join("_3_Completion104_0.cmp").exists(),
            "{version}"
        );
        let searcher = SuggestIndexSearcher::new(&reader);
        let rows = fs::read_to_string(root.join(version).join("queries.tsv")).unwrap();
        let mut compared = 0;
        for line in rows.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let (kind, field, text, options) = (c[0], c[1], c[2], c[3]);
            let (n, dedup) = (c[4].parse::<usize>().unwrap(), c[5] == "1");
            let want = c.get(6).copied().unwrap_or("");
            // the filters select by segment-local doc id, which the merge changes
            let filtered = options.contains("even") || options.contains("none");
            if kind.starts_with("context-")
                || dedup
                || filtered
                || want.split('\u{1f}').count() >= n
            {
                continue;
            }
            let without_doc = |hits: &str| {
                let mut v: Vec<String> = hits
                    .split('\u{1f}')
                    .filter(|h| !h.is_empty())
                    .map(|h| {
                        let f: Vec<&str> = h.split('\u{1e}').collect();
                        format!("{}\u{1e}{}\u{1e}{}", f[0], f[2], f[3])
                    })
                    .collect();
                v.sort();
                v
            };
            let got = format_hits(
                &searcher
                    .suggest(&build(kind, field, text, options), n, false)
                    .unwrap(),
            );
            assert_eq!(
                without_doc(&got),
                without_doc(want),
                "{version} {kind} {field} {text:?} {options:?} {n}"
            );
            compared += 1;
        }
        assert!(compared > 300, "{version}: {compared}");
    }
}

/// `merge` stops offering a shard's hits at the first one the queue turns
/// away, and a `top_n` of 0 keeps nothing.
#[test]
fn merge_stops_a_shard_at_its_first_rejected_hit() {
    let hit = |key: &str, doc: i32, score: f32| SuggestScoreDoc {
        doc,
        key: key.to_string(),
        context: None,
        score,
    };
    let a = [hit("a", 0, 5.0), hit("b", 1, 1.0), hit("z", 9, 9.0)];
    let b = [hit("c", 2, 3.0), hit("d", 3, 2.0)];
    // "b" is turned away by the full queue (5, 3), so "z" is never offered
    let merged = merge(2, &[&b, &a]);
    let keys: Vec<&str> = merged.iter().map(|h| h.key.as_str()).collect();
    assert_eq!(keys, ["a", "c"]);
    assert!(merge(0, &[&a, &b]).is_empty());
    assert!(merge(3, &[]).is_empty());
}

/// A `.cmp` whose suffix names no completion format Lucene 10.5.0 reads
/// (`Completion84`, a Lucene 8 delegate) refuses the segment.
#[test]
fn a_completion_suffix_of_no_known_format_is_refused() {
    let tmp = TempDir::new("completion-unknown-format");
    let src = data().join("../bwc-completion/9.0.0/index");
    for entry in fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
    }
    // the `.cmp` renamed in the `.si`'s file list, its footer recomputed
    let si = tmp.path().join("_0.si");
    let bytes = fs::read(&si).unwrap();
    let mut renamed = replace_all(
        &bytes[..bytes.len() - 16],
        b"_0_Completion90_0.cmp",
        b"_0_Completion84_0.cmp",
    );
    assert_ne!(renamed, bytes[..bytes.len() - 16]);
    lucene_store::codec_util::write_footer(&mut renamed);
    fs::write(&si, renamed).unwrap();
    fs::rename(
        tmp.path().join("_0_Completion90_0.cmp"),
        tmp.path().join("_0_Completion84_0.cmp"),
    )
    .unwrap();
    let Err(err) = DirectoryReader::open(&FsDirectory::open(tmp.path())) else {
        panic!("an unknown completion format must fail the open");
    };
    let msg = err.to_string();
    assert!(msg.contains("Completion84_0"), "{msg}");
    assert!(msg.contains("is not a completion postings format"), "{msg}");
}

fn replace_all(haystack: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if haystack[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(haystack[i]);
            i += 1;
        }
    }
    out
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
    replay(tmp.path(), &data());
}
