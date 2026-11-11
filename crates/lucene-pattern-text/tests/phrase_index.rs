//! Pattern-text fields written by the index writer and searched by phrase:
//! the hits are exactly the documents whose message holds the phrase, as a
//! positional phrase on the same messages finds them.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::document::{
    Document, Field, FieldType, IndexOptions, Store, StoredValue, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_pattern_text::{
    fields, pattern_phrase_query, PatternPhraseSource, PhraseTokens, Tokenizer,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery};
use lucene_store::ByteBuffersDirectory;

fn analyze(a: &Analyzer, text: &str, out: &mut Vec<(i32, String)>) {
    let mut pos = -1i32;
    for t in a.analyze(text) {
        pos += t.position_increment;
        out.push((pos, t.term));
    }
}

/// Log lines whose phrases sit in the static text, cross a variable, sit
/// inside variables, or appear in variable-free templates.
fn messages() -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..400u32 {
        let ip = format!("10.0.{}.{}", i % 7, i % 13);
        out.push(match i % 8 {
            0 => format!(
                "Dec 10 06:55:{i:02} sshd[{}]: Connection closed by {ip} [preauth]",
                24000 + i
            ),
            1 => format!("Failed password for root from {ip} port {} ssh2", 40000 + i),
            2 => format!("Failed password for invalid user admin{} from {ip}", i % 3),
            3 => "instruction cache parity error corrected".to_string(),
            4 => "instruction cache parity error".to_string(),
            5 => format!("connection {} closed by peer", i % 5),
            6 => format!("Receiving block blk_{} src: /{ip}:50010", i * 31),
            _ => format!("  spaced   message {i}  "),
        });
    }
    out
}

const PHRASES: &[&str] = &[
    "connection closed by",
    "failed password for root",
    "failed password",
    "instruction cache parity error corrected",
    "parity error",
    "closed by peer",
    "receiving block",
    "spaced message",
    "password for invalid user",
    "no such phrase anywhere",
    "closed",
    // only the variables decide these
    "user admin0 from",
    "admin1 from",
    "port 40009",
    "connection 3 closed",
    "connection 3 closed by peer",
    "closed by 10.0.1.1",
];

fn positional_type() -> FieldType {
    let mut t = FieldType::new();
    t.set_tokenized(true).unwrap();
    t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    t.freeze();
    t
}

fn index(analyzer: Option<Arc<Analyzer>>) -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(
        &dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_analyzer(analyzer);
    for (i, m) in messages().iter().enumerate() {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::Yes));
        fields(&mut doc, "message", m).unwrap();
        // a stored field after the variables: the reader stops early
        doc.add(Field::stored("after", StoredValue::String(m.clone())));
        doc.add(Field::from_string("positional", m.as_str(), positional_type()).unwrap());
        w.add_fields_document(&doc).unwrap();
        if i == 199 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    w.close().unwrap();
    dir
}

type Oracle<'x> = dyn Fn(&IndexSearcher<'_, '_>, &[(i32, String)]) -> u64 + 'x;

/// Each phrase's hits by `pattern_phrase_query`, with and without the
/// template shortcut, checked against `want`.
fn check(dir: &ByteBuffersDirectory, analyzer: Arc<Analyzer>, want: &Oracle<'_>) {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 2);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let a = Arc::clone(&analyzer);
    let tokenizer: Arc<Tokenizer> = Arc::new(move |text: &str, out: &mut Vec<(i32, String)>| {
        analyze(&a, text, out);
    });
    let mut nonzero = 0;
    for phrase in PHRASES {
        let mut toks = Vec::new();
        analyze(&analyzer, phrase, &mut toks);
        let expected = want(&searcher, &toks);
        if expected > 0 {
            nonzero += 1;
        }
        for shortcut in [true, false] {
            let mut v = PatternPhraseSource::new(
                "message",
                PhraseTokens::new(toks.clone()),
                Arc::clone(&tokenizer),
                "standard",
            );
            if !shortcut {
                v = v.whole_messages_only();
            }
            let q = BooleanQuery::new().with_must([pattern_phrase_query(
                "message",
                PhraseTokens::new(toks.clone()),
                v,
            )]);
            assert_eq!(
                searcher.count(&q).unwrap(),
                expected,
                "{phrase} shortcut={shortcut}"
            );
            let top = searcher.search(&q, 10).unwrap();
            assert_eq!(top.total_hits.value, expected, "{phrase}");
        }
    }
    assert!(nonzero >= 12, "most phrases must match something");
}

#[test]
fn phrase_hits_equal_positional_phrase_hits() {
    let dir = index(None);
    check(
        &dir,
        Arc::new(Analyzer::standard(None)),
        &|searcher, toks| {
            let q = BooleanQuery::new().with_must([Clause::Phrase(PhraseQuery::new(
                "positional",
                toks.iter()
                    .map(|(_, t)| t.as_bytes().to_vec())
                    .collect::<Vec<Vec<u8>>>(),
            ))]);
            searcher.count(&q).unwrap()
        },
    );
}

#[test]
fn removed_stopwords_leave_the_same_gaps() {
    let stop: HashSet<String> = ["by", "for", "from"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let analyzer = Arc::new(Analyzer::standard(Some(&stop)));
    let dir = index(Some(Arc::clone(&analyzer)));
    let msgs = messages();
    let a = Arc::clone(&analyzer);
    // brute force: the phrase's tokens, at its positions, in any message
    check(&dir, analyzer, &move |_, toks| {
        let base = toks.first().map_or(0, |t| t.0);
        msgs.iter()
            .filter(|m| {
                let mut doc = Vec::new();
                analyze(&a, m, &mut doc);
                let mut at: HashMap<i32, Vec<&str>> = HashMap::new();
                for (p, t) in &doc {
                    at.entry(*p).or_default().push(t.as_str());
                }
                doc.iter().any(|(start, _)| {
                    toks.iter().all(|(p, t)| {
                        at.get(&(start + p - base))
                            .is_some_and(|v| v.contains(&t.as_str()))
                    })
                })
            })
            .count() as u64
    });
}
