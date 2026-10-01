//! Writes `GenTokenAttributes`' documents for `VerifyTokenAttributes`: an
//! analyzer whose `TokenFilter` sets payloads (field `pay`) and term
//! frequencies (field `tf`), set with [`IndexWriter::set_analyzer`], and a
//! similarity whose norm packs the whole `FieldInvertState`.
//!
//! `<out>/flushed` holds one flushed segment; `<out>/merged` a segment
//! without payloads and one with them, force-merged into one.
//!
//! Usage: `write_token_attributes_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::{Arc, Mutex};

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, LowerCaseFilter, ReuseStrategy, StandardTokenizer,
    TokenFilter, TokenStream, TokenStreamComponents,
};
use lucene_index::document::{Document, Field, FieldType, IndexOptions, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::similarity::{FieldInvertState, NormSimilarity};
use lucene_store::FsDirectory;

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

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_token_attributes_fixture <output-dir>");
    for (sub, merge) in [("flushed", false), ("merged", true)] {
        let path = std::path::Path::new(&out).join(sub);
        std::fs::create_dir_all(&path).unwrap();
        let sim = Arc::new(StateSim::default());
        write(&FsDirectory::open(&path), merge, &sim);
    }
}
