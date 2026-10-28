//! `BasePostingsFormatTestCase.testInvertedWrite`: a codec whose postings
//! consumer walks the `body` postings it is handed, twice, at every flush
//! and merge, gathering its own term statistics, which must match what the
//! final reader says.
//!
//! Differs: Java's `FilterCodec` wraps the default `PostingsFormat`'s
//! `FieldsConsumer`; here the codec's
//! [`PostingsWriterFactory`] wraps the native writer
//! ([`NativePostingsWriter`]), and the consumer is handed each field's
//! terms as a sorted slice rather than a `Fields`: walking it twice is
//! iterating it twice, `seekExact` a binary search and `seekCeil` a
//! partition point. Java's `random()` inside the consumer (another
//! thread's stream during a merge) is a stream of its own seeded off the
//! test's. `RamUsageTester.ramUsed(doc)` is the body's UTF-16 size.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_consumer::{
    NativePostingsWriter, PostingsWriteSink, PostingsWriteState, PostingsWriterFactory,
};
use lucene_codecs::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource};
use lucene_index::document::{Document, Field, FieldType, TextField};
use lucene_search::multi_terms::MultiTerms;
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{new_config, riw_with};
use crate::util::line_file_docs::LineFileDocs;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `TermFreqs`.
#[derive(Debug, Default, Clone, Copy)]
struct TermFreqs {
    doc_freq: i64,
    total_term_freq: i64,
}

/// What the consumers gather, shared by every flush and merge.
#[derive(Debug)]
struct Stats {
    random: JavaRandom,
    term_freqs: HashMap<Vec<u8>, TermFreqs>,
    sum_doc_freq: i64,
    sum_total_term_freq: i64,
    flushes: u32,
    merges: u32,
}

#[derive(Debug)]
struct InvertedCodec(Arc<Mutex<Stats>>);

impl Lucene104Codec for InvertedCodec {
    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(self)
    }
}

impl PostingsWriterFactory for InvertedCodec {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
        let body = state
            .field_infos
            .iter()
            .find(|fi| fi.name == "body")
            .map(|fi| fi.number);
        Ok(Box::new(InvertedSink {
            delegate: NativePostingsWriter::new(state)?,
            stats: Arc::clone(&self.0),
            body,
        }))
    }
}

struct InvertedSink<'a> {
    delegate: NativePostingsWriter<'a>,
    stats: Arc<Mutex<Stats>>,
    body: Option<i32>,
}

/// One term's `docFreq` and `totalTermFreq`, read the way the test reads
/// a `PostingsEnum`: freqs only, or positions up to a random limit.
fn walk(r: &mut JavaRandom, term: &dyn TermSource) -> (i64, i64) {
    let no_positions = r.next_boolean();
    let mut doc_freq = 0;
    let mut total_term_freq = 0;
    for (i, &(_, freq)) in term.docs().iter().enumerate() {
        doc_freq += 1;
        total_term_freq += i64::from(freq);
        let limit = test_util::next_int(r, 1, freq);
        if !no_positions && i < term.position_lists() {
            let positions = term.positions(i);
            assert!(positions.len() >= limit as usize);
        }
    }
    (doc_freq, total_term_freq)
}

impl InvertedSink<'_> {
    fn check(&self, inputs: &[FieldPostingsInput<'_, &dyn TermSource>], is_merge: bool) {
        let body = self.body.expect("terms for body");
        let input = inputs
            .iter()
            .find(|i| i.field_number == body)
            .expect("terms for body");
        let mut guard = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        let stats = &mut *guard;
        if is_merge {
            stats.merges += 1;
        } else {
            stats.flushes += 1;
        }
        // The terms are walked twice; with `addOnSecondPass` the stats are
        // added up on the second walk.
        let add_on_second_pass = stats.random.next_boolean();
        for term in input.terms {
            let (doc_freq, total_term_freq) = walk(&mut stats.random, *term);
            let key = term.term().to_vec();
            // During merge we should only see terms we had already seen
            // during a previous flush.
            assert!(!is_merge || stats.term_freqs.contains_key(&key));
            if !is_merge {
                if !add_on_second_pass {
                    let tf = stats.term_freqs.entry(key).or_default();
                    tf.doc_freq += doc_freq;
                    tf.total_term_freq += total_term_freq;
                    stats.sum_doc_freq += doc_freq;
                    stats.sum_total_term_freq += total_term_freq;
                } else {
                    // A placeholder the second pass fills in.
                    stats.term_freqs.entry(key).or_default();
                }
            }
        }
        // Also seek the terms: every term seen so far (`seekExact`).
        let keys: Vec<Vec<u8>> = stats.term_freqs.keys().cloned().collect();
        for key in keys {
            let Ok(at) = input
                .terms
                .binary_search_by(|t| t.term().cmp(key.as_slice()))
            else {
                continue;
            };
            let (doc_freq, total_term_freq) = walk(&mut stats.random, input.terms[at]);
            if !is_merge && add_on_second_pass {
                let tf = stats.term_freqs.get_mut(&key).expect("a placeholder");
                tf.doc_freq += doc_freq;
                tf.total_term_freq += total_term_freq;
                stats.sum_doc_freq += doc_freq;
                stats.sum_total_term_freq += total_term_freq;
            }
            let tf = stats.term_freqs[&key];
            assert!(doc_freq <= tf.doc_freq);
            assert!(total_term_freq <= tf.total_term_freq);
        }
        // Also `seekCeil`.
        for _ in 0..10 {
            let target = test_util::random_realistic_unicode_string(&mut stats.random).into_bytes();
            let at = input
                .terms
                .partition_point(|t| t.term() < target.as_slice());
            if let Some(found) = input.terms.get(at) {
                // NOT_FOUND lands on the next term.
                assert!(found.term() >= target.as_slice());
            }
        }
    }
}

impl PostingsWriteSink for InvertedSink<'_> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.delegate.write(inputs, norms)?;
        self.check(inputs, false);
        Ok(())
    }

    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.delegate.merge(inputs, norms)?;
        self.check(inputs, true);
        Ok(())
    }

    fn close(&mut self) -> postings_writer::Result<()> {
        self.delegate.close()
    }
}

/// How many flushes and merges the consumers saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvertedWriteStats {
    pub flushes: u32,
    pub merges: u32,
}

/// `testInvertedWrite`, indexing `atLeast(kib) * 1024` bytes of bodies.
/// With `merged`, the writer flushes every 50 documents and force-merges
/// to one segment before the reader opens, so the merge-time checks run
/// whatever the random configuration draws (Java's test merges only when
/// its policy does).
pub fn inverted_write(t: &mut LuceneTestCase, kib: i32, merged: bool) -> InvertedWriteStats {
    let dir = ByteBuffersDirectory::new();
    // `IndexWriter.MAX_TERM_LENGTH`.
    let max_token_length = test_util::next_int(t.random(), 1, 32766);
    let analyzer = Arc::new(
        crate::analysis::mock_analyzer::MockAnalyzer::new(t.random())
            .max_token_length(max_token_length)
            .build(),
    );
    let mut config = new_config(t, analyzer);
    if merged {
        config.max_buffered_docs = Some(50);
    }
    let stats = Arc::new(Mutex::new(Stats {
        random: JavaRandom::new(t.random().next_long()),
        term_freqs: HashMap::new(),
        sum_doc_freq: 0,
        sum_total_term_freq: 0,
        flushes: 0,
        merges: 0,
    }));
    let mut w = riw_with(t, &dir, config);
    w.writer()
        .set_codec(Some(Arc::new(InvertedCodec(Arc::clone(&stats)))))
        .expect("setCodec");

    let mut docs = LineFileDocs::new(t.random()).expect("LineFileDocs");
    let bytes_to_index = i64::from(t.at_least(kib)) * 1024;
    let mut bytes_indexed = 0i64;
    while bytes_indexed < bytes_to_index {
        let line = docs.next_doc().expect("nextDoc");
        let mut just_body = Document::new();
        just_body.add(body_field(&line.body));
        w.add_document(&just_body).expect("addDocument");
        bytes_indexed += 2 * line.body.encode_utf16().count() as i64;
    }
    if merged {
        super::base_index_file_format::riw_force_merge(&mut w, 1);
    }
    let r = w.get_reader().expect("getReader");
    w.close().expect("close");

    let terms = MultiTerms::get_terms(&r, "body")
        .expect("terms")
        .expect("terms for body");
    let stats = stats.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(stats.sum_doc_freq, terms.sum_doc_freq());
    assert_eq!(stats.sum_total_term_freq, terms.sum_total_term_freq());
    InvertedWriteStats {
        flushes: stats.flushes,
        merges: stats.merges,
    }
}

/// `doc.getField("body")` of a `LineFileDocs` document: stored text with
/// positions and term vectors with positions and offsets.
fn body_field(body: &str) -> Field {
    let mut ft = FieldType::copy_of(&TextField::type_stored());
    for r in [
        ft.set_index_options(IndexOptions::DocsAndFreqsAndPositions),
        ft.set_store_term_vectors(true),
        ft.set_store_term_vector_offsets(true),
        ft.set_store_term_vector_positions(true),
    ] {
        r.expect("an unfrozen FieldType");
    }
    Field::from_string("body", body, ft).expect("stored text is valid")
}

/// `testInvertedWrite`.
pub fn test_inverted_write(t: &mut LuceneTestCase) {
    inverted_write(t, 100, false);
}
