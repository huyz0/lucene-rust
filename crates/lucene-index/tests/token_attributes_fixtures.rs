//! What `IndexingChain` reads from a token stream besides the term, against
//! Java (`GenTokenAttributes`): a `TokenFilter`'s `PayloadAttribute` and
//! `TermFrequencyAttribute` (the analysis crate's streaming model, an
//! analyzer set with [`IndexWriter::set_analyzer`]), and the whole
//! `FieldInvertState` a similarity's `computeNorm` is handed --
//! `maxTermFrequency`, `position`, `offset`, the attribute source as `end()`
//! left it.
//!
//! The same documents go through the document API with the same analyzer and
//! a similarity whose norm packs every state value; each state must be the one
//! Java recorded (`states.txt`), and every segment file but the `.si` must be
//! Java's byte for byte (segment id normalised): the payloads in `.pay`, the
//! custom frequencies in `.doc`, the packed norms in `.nvd`, `storePayloads`
//! in `.fnm` -- after a merge too, where only the second segment had payloads.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::{Arc, Mutex};

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, LowerCaseFilter, ReuseStrategy, StandardTokenizer,
    TokenFilter, TokenStream, TokenStreamComponents,
};
use lucene_index::check_index;
use lucene_index::document::{Document, Field, FieldType, IndexOptions, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_infos;
use lucene_index::similarity::{FieldInvertState, NormSimilarity};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

const PER_SEGMENT: usize = 120;
const WORDS: [&str; 26] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
];
const PLAIN_FROM: usize = 12;
const PLAIN_TO: usize = 19;

fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn word(i: usize, k: i64, plain: bool) -> &'static str {
    if plain {
        WORDS[PLAIN_FROM + (value(i, k) % (PLAIN_TO - PLAIN_FROM) as i64) as usize]
    } else {
        WORDS[(value(i, k) % WORDS.len() as i64) as usize]
    }
}

fn words(i: usize, from: i64, n: i64, plain: bool) -> String {
    let mut b = String::new();
    for k in from..from + n {
        if !b.is_empty() {
            b.push_str(if k % 3 == 0 { ", " } else { " " });
        }
        b.push_str(word(i, k, plain));
    }
    b
}

/// `GenTokenAttributes.Tag`.
struct Tag {
    input: LowerCaseFilter<StandardTokenizer>,
    field: String,
}

impl TokenFilter for Tag {
    type Input = LowerCaseFilter<StandardTokenizer>;
    fn input(&self) -> &Self::Input {
        &self.input
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        let c = a.term().chars().next().unwrap_or('\0');
        let len = a.term_utf16_len();
        if self.field == "pay" {
            if c < 'm' {
                a.set_payload(Some(vec![len as u8, c as u8]));
            } else if c >= 't' {
                a.set_payload(Some(Vec::new()));
            }
        } else if self.field == "tf" {
            a.set_term_frequency(len as i32)?;
        }
        Ok(true)
    }
}

/// `GenTokenAttributes.Attrs`.
struct Attrs;

impl AnalyzerDefinition for Attrs {
    fn create_components(&self, field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(Tag {
            input: LowerCaseFilter::new(StandardTokenizer::new()),
            field: field.to_string(),
        }))
    }
    fn position_increment_gap(&self, _field: &str) -> i32 {
        3
    }
    fn offset_gap(&self, _field: &str) -> i32 {
        5
    }
}

/// `GenTokenAttributes.StateSim`.
#[derive(Debug, Default)]
struct StateSim {
    states: Mutex<Vec<String>>,
}

impl NormSimilarity for StateSim {
    fn compute_norm(&self, field: &str, s: &FieldInvertState) -> i64 {
        let (end_offset, end_inc) = s.attribute_source.as_ref().map_or((-1, -1), |a| {
            (i64::from(a.end_offset()), i64::from(a.position_increment()))
        });
        self.states.lock().unwrap().push(format!(
            "{field} length={} maxTermFrequency={} position={} offset={} uniqueTermCount={} \
             numOverlap={} endOffset={end_offset} endIncrement={end_inc}",
            s.length,
            s.max_term_frequency,
            s.position,
            s.offset,
            s.unique_term_count,
            s.num_overlap
        ));
        1 + i64::from(s.length)
            + 7 * i64::from(s.max_term_frequency)
            + 131 * i64::from(s.position)
            + 1031 * i64::from(s.offset)
            + 10007 * i64::from(s.unique_term_count)
            + 100_003 * i64::from(s.num_overlap)
            + 1_000_003 * (end_offset + 2)
            + 10_000_019 * (end_inc + 2)
    }
}

fn field_type(options: IndexOptions) -> FieldType {
    let mut ft = FieldType::new();
    ft.set_tokenized(true).unwrap();
    ft.set_index_options(options).unwrap();
    ft.frozen()
}

/// `GenTokenAttributes.doc`.
fn doc(i: usize, plain: bool) -> Document {
    let pay = field_type(IndexOptions::DocsAndFreqsAndPositionsAndOffsets);
    let tf = field_type(IndexOptions::DocsAndFreqs);
    let docs = field_type(IndexOptions::Docs);
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    d.add(Field::from_string("pay", words(i, 0, 4, plain), pay.clone()).unwrap());
    d.add(Field::from_string("pay", words(i, 4, 3, plain) + " ", pay).unwrap());
    d.add(
        Field::from_string(
            "tf",
            format!("{} {}", words(i, 7, 3, false), word(i, 7, false)),
            tf,
        )
        .unwrap(),
    );
    d.add(
        Field::from_string(
            "docs",
            format!(
                "{} {} {}",
                word(i, 10, false),
                word(i, 10, false),
                word(i, 11, false)
            ),
            docs,
        )
        .unwrap(),
    );
    d
}

fn write(dir: &FsDirectory, merge: bool, sim: &Arc<StateSim>) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::with_reuse_strategy(
        Attrs,
        ReuseStrategy::PerField,
    ))));
    w.set_similarity(Some(Arc::clone(sim) as Arc<dyn NormSimilarity>));
    w.set_max_full_flush_merge_wait_millis(0);
    if merge {
        let mut tmp = TieredMergePolicy::default();
        tmp.compound_file_settings_mut()
            .set_no_cfs_ratio(0.0)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(tmp)));
    }
    let segments = if merge { 2 } else { 1 };
    for seg in 0..segments {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i, merge && seg == 0)).unwrap();
        }
        w.commit().unwrap();
    }
    if merge {
        w.force_merge(1).unwrap();
    }
}

fn fixture(which: &str) -> String {
    format!(
        "{}/../../fixtures/data/token_attributes/{which}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `ours` with its segment id replaced by `id` -- in every index header it
/// holds (`.tmd` has the terms dictionary's and the postings writer's) --
/// and its footer re-signed.
fn with_segment_id(ours: &[u8], our_id: &[u8; 16], id: &[u8; 16]) -> Vec<u8> {
    let mut bytes = ours.to_vec();
    let mut at = 0;
    while at + 16 <= bytes.len() {
        if &bytes[at..at + 16] == our_id {
            bytes[at..at + 16].copy_from_slice(id);
            at += 16;
        } else {
            at += 1;
        }
    }
    let n = bytes.len();
    let crc = u64::from(crc32fast::hash(&bytes[..n - 8]));
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    bytes
}

fn check(which: &str, merge: bool) {
    let tmp = TempDir::new("token-attributes");
    let dir = FsDirectory::open(&tmp);
    let sim = Arc::new(StateSim::default());
    write(&dir, merge, &sim);

    let mut states = sim.states.lock().unwrap().clone();
    states.sort();
    let expected: Vec<String> = std::fs::read_to_string(format!("{}/states.txt", fixture(which)))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(states.len(), expected.len(), "{which}: computeNorm calls");
    for (a, b) in states.iter().zip(&expected) {
        assert_eq!(a, b, "{which}: a FieldInvertState differs from Java's");
    }

    let java = FsDirectory::open(fixture(which));
    let ours = segment_infos::read_latest(&dir).unwrap();
    let theirs = segment_infos::read_latest(&java).unwrap();
    assert_eq!(ours.segments.len(), 1);
    assert_eq!(theirs.segments.len(), 1);
    let (a, b) = (&ours.segments[0], &theirs.segments[0]);
    let mut compared = Vec::new();
    for name in java.list_all().unwrap() {
        let Some(rest) = name
            .strip_prefix(&format!("{}_", b.segment_name))
            .map(|r| format!("_{r}"))
            .or_else(|| name.strip_prefix(&b.segment_name).map(str::to_string))
        else {
            continue;
        };
        if rest == ".si" {
            continue;
        }
        let mine_name = format!("{}{rest}", a.segment_name);
        let mine = dir.open(&mine_name).unwrap();
        let java_bytes = java.open(&name).unwrap();
        assert!(
            with_segment_id(&mine, &a.segment_id, &b.segment_id) == java_bytes[..],
            "{which}: {mine_name} differs from Java's {name}"
        );
        compared.push(rest);
    }
    compared.sort();
    for ext in [
        ".fnm",
        ".nvd",
        "_Lucene104_0.doc",
        "_Lucene104_0.pay",
        "_Lucene104_0.pos",
    ] {
        assert!(compared.iter().any(|c| c == ext), "{which}: {ext} compared");
    }
    let results = check_index::check_directory(&dir).unwrap();
    assert!(
        results.iter().all(|r| r.failures().is_empty()),
        "{results:?}"
    );
}

#[test]
fn a_token_filters_payloads_frequencies_and_the_invert_state_are_javas() {
    check("flushed", false);
}

#[test]
fn a_merge_keeps_payloads_one_source_segment_lacked() {
    check("merged", true);
}
