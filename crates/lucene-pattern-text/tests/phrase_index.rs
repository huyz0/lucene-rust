//! Pattern-text fields written by the index writer and searched by phrase:
//! the hits are exactly the documents whose message holds the phrase, as a
//! positional phrase on the same messages finds them.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Field, FieldType, IndexOptions, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_pattern_text::{fields, pattern_phrase_query, PhraseTokens, Tokenizer};
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

/// Log lines whose phrases sit in the static text, cross a variable, or
/// appear in variable-free templates.
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

fn positional_type() -> FieldType {
    let mut t = FieldType::new();
    t.set_tokenized(true).unwrap();
    t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    t.freeze();
    t
}

#[test]
fn phrase_hits_equal_positional_phrase_hits() {
    let msgs = messages();
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
    for (i, m) in msgs.iter().enumerate() {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::Yes));
        fields(&mut doc, "message", m).unwrap();
        doc.add(Field::from_string("positional", m.as_str(), positional_type()).unwrap());
        w.add_fields_document(&doc).unwrap();
        if i == 199 {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    w.close().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 2);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();

    let analyzer = Arc::new(Analyzer::standard(None));
    let a = Arc::clone(&analyzer);
    let tokenizer: Arc<Tokenizer> = Arc::new(move |text: &str, out: &mut Vec<(i32, String)>| {
        analyze(&a, text, out);
    });

    for phrase in [
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
    ] {
        let mut toks = Vec::new();
        analyze(&analyzer, phrase, &mut toks);
        let want = {
            let q = BooleanQuery::new().with_must([Clause::Phrase(PhraseQuery::new(
                "positional",
                toks.iter()
                    .map(|(_, t)| t.as_bytes().to_vec())
                    .collect::<Vec<Vec<u8>>>(),
            ))]);
            searcher.count(&q).unwrap()
        };
        let q = BooleanQuery::new().with_must([pattern_phrase_query(
            "message",
            PhraseTokens::new(toks.clone()),
            Arc::clone(&tokenizer),
        )]);
        let got = searcher.count(&q).unwrap();
        assert_eq!(got, want, "{phrase}");
        let top = searcher.search(&q, 10).unwrap();
        assert_eq!(top.total_hits.value, want, "{phrase}");
    }
}
