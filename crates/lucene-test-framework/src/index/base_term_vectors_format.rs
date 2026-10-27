//! `BaseTermVectorsFormatTestCase`: the term vectors suite, on the one term
//! vectors format this port writes (`Lucene90TermVectorsFormat`).
//!
//! Differs:
//!
//! - A document's term vectors decode into a [`TermVectorsDocument`];
//!   `Terms`/`TermsEnum`/`PostingsEnum` are lucene-search's
//!   [`TermVectorTermsEnum`] over one of its fields, so the field-level
//!   statistics (`size`, `hasPositions`, ...) are read off the value.
//! - Enum reuse (`postings(reuse, flags)`) has no counterpart: each call
//!   hands out a new enum, checked as Java checks the reused one.
//! - `assertEquals(RandomDocument, Fields)` seeks back to every term,
//!   alternating `seekExact` and `seekCeil`, not to five random ones.
//! - `RandomTokenStream`'s `PermissiveOffsetAttributeImpl` is not needed: a
//!   [`FieldTokens`] carries its offsets unchecked.
//! - `maybeWrapWithMergingReader` is not applied (see
//!   [`super::base_stored_fields_format`]).

use super::base_index_file_format::SuiteTest;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::term_vectors::TermVectorsDocument;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Field, FieldToken, FieldTokens, FieldType, NumericDocValuesField, Store, StringField,
    TextField,
};
use lucene_index::segment_info::{IndexSortField, IndexSortKind, NumericSortKey};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::term_vector_terms::TermVectorTermsEnum;
use lucene_search::reader::{LeafReader, PostingsEnum, PostingsFlags, TermsEnum, NO_MORE_DOCS};
use lucene_store::ByteBuffersDirectory;
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{
    add_indexes_slowly, doc_with_id, leaf_of, mock_analyzer, new_config, new_riw, only_leaf,
    riw_force_merge, riw_with, set_index_sort, writer_with, IndexFileFormatSuite,
};
use super::merge_policies::shuffle;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `Options`: a combination of term vectors options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Options {
    None,
    Positions,
    Offsets,
    PositionsAndOffsets,
    PositionsAndPayloads,
    PositionsAndOffsetsAndPayloads,
}

impl Options {
    /// Every option, in declaration order (`EnumSet.allOf`).
    pub const ALL: [Options; 6] = [
        Options::None,
        Options::Positions,
        Options::Offsets,
        Options::PositionsAndOffsets,
        Options::PositionsAndPayloads,
        Options::PositionsAndOffsetsAndPayloads,
    ];

    /// `(positions, offsets, payloads)`.
    pub fn flags(self) -> (bool, bool, bool) {
        match self {
            Options::None => (false, false, false),
            Options::Positions => (true, false, false),
            Options::Offsets => (false, true, false),
            Options::PositionsAndOffsets => (true, true, false),
            Options::PositionsAndPayloads => (true, false, true),
            Options::PositionsAndOffsetsAndPayloads => (true, true, true),
        }
    }

    fn java_name(self) -> &'static str {
        match self {
            Options::None => "NONE",
            Options::Positions => "POSITIONS",
            Options::Offsets => "OFFSETS",
            Options::PositionsAndOffsets => "POSITIONS_AND_OFFSETS",
            Options::PositionsAndPayloads => "POSITIONS_AND_PAYLOADS",
            Options::PositionsAndOffsetsAndPayloads => "POSITIONS_AND_OFFSETS_AND_PAYLOADS",
        }
    }
}

/// `fieldType(options)`: unstored text with term vectors as `options` says.
pub fn field_type(options: Options) -> FieldType {
    let (positions, offsets, payloads) = options.flags();
    let mut ft = FieldType::copy_of(&TextField::type_not_stored());
    for r in [
        ft.set_store_term_vectors(true),
        ft.set_store_term_vector_positions(positions),
        ft.set_store_term_vector_offsets(offsets),
        ft.set_store_term_vector_payloads(payloads),
    ] {
        r.expect("an unfrozen FieldType");
    }
    ft.frozen()
}

/// `BaseTermVectorsFormatTestCase` itself: its `addRandomFields`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TermVectorsSuite;

impl IndexFileFormatSuite for TermVectorsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        for opts in Options::ALL {
            let ft = field_type(opts);
            let num_fields = t.random().next_int_bounded(5);
            for _ in 0..num_fields {
                let s = test_util::random_simple_string_max(t.random(), 2);
                doc.add(
                    Field::from_string(format!("f_{}", opts.java_name()), s, ft.clone())
                        .expect("a term vectors field"),
                );
            }
        }
    }
}

/// `RandomTokenStream`: a random token stream over sample terms, with what
/// it produced kept for the checks.
#[derive(Debug, Clone)]
pub struct RandomTokenStream {
    pub terms: Vec<String>,
    pub positions: Vec<i32>,
    pub position_increments: Vec<i32>,
    pub start_offsets: Vec<i32>,
    pub end_offsets: Vec<i32>,
    /// `None` is Java's `null` payload.
    pub payloads: Vec<Option<Vec<u8>>>,
    pub freqs: HashMap<String, i32>,
    position_to_terms: HashMap<i32, BTreeSet<usize>>,
    start_offset_to_terms: HashMap<i32, BTreeSet<usize>>,
}

impl RandomTokenStream {
    /// `new RandomTokenStream(len, sampleTerms, sampleTermBytes)`.
    pub fn new(t: &mut LuceneTestCase, len: usize, sample_terms: &[String]) -> Self {
        let mut terms = Vec::with_capacity(len);
        let mut position_increments = Vec::with_capacity(len);
        let mut start_offsets: Vec<i32> = Vec::with_capacity(len);
        let mut end_offsets = Vec::with_capacity(len);
        for i in 0..len {
            let o = t.random().next_int_bounded(sample_terms.len() as i32) as usize;
            terms.push(sample_terms[o].clone());
            position_increments.push(test_util::next_int(
                t.random(),
                if i == 0 { 1 } else { 0 },
                10,
            ));
            let start = if i == 0 {
                test_util::next_int(t.random(), 0, 1 << 16)
            } else {
                let gap = if t.rarely() { 1 << 16 } else { 20 };
                start_offsets[i - 1] + test_util::next_int(t.random(), 0, gap)
            };
            start_offsets.push(start);
            let span = if t.rarely() { 1 << 10 } else { 20 };
            end_offsets.push(start + test_util::next_int(t.random(), 0, span));
        }
        let mut positions: Vec<i32> = Vec::with_capacity(len);
        for i in 0..len {
            positions.push(if i == 0 {
                position_increments[i] - 1
            } else {
                positions[i - 1] + position_increments[i]
            });
        }
        let payloads = if t.rarely() {
            let p = random_payload(t.random());
            vec![p; len]
        } else {
            (0..len).map(|_| random_payload(t.random())).collect()
        };
        let mut position_to_terms: HashMap<i32, BTreeSet<usize>> = HashMap::new();
        let mut start_offset_to_terms: HashMap<i32, BTreeSet<usize>> = HashMap::new();
        for i in 0..len {
            position_to_terms.entry(positions[i]).or_default().insert(i);
            start_offset_to_terms
                .entry(start_offsets[i])
                .or_default()
                .insert(i);
        }
        let mut freqs = HashMap::new();
        for term in &terms {
            *freqs.entry(term.clone()).or_insert(0) += 1;
        }
        RandomTokenStream {
            terms,
            positions,
            position_increments,
            start_offsets,
            end_offsets,
            payloads,
            freqs,
            position_to_terms,
            start_offset_to_terms,
        }
    }

    /// `hasPayloads()`: some token has a non-empty payload.
    pub fn has_payloads(&self) -> bool {
        self.payloads
            .iter()
            .any(|p| p.as_ref().is_some_and(|p| !p.is_empty()))
    }

    /// The stream as the writer consumes it.
    pub fn tokens(&self) -> FieldTokens {
        FieldTokens {
            tokens: (0..self.terms.len())
                .map(|i| FieldToken {
                    term: self.terms[i].as_bytes().to_vec(),
                    start_offset: self.start_offsets[i],
                    end_offset: self.end_offsets[i],
                    position_increment: self.position_increments[i],
                    term_frequency: 1,
                    payload: self.payloads[i].clone(),
                })
                .collect(),
            final_position_increment: 0,
            final_offset: 0,
            end_attributes: None,
        }
    }
}

/// `randomPayload()`: `null` one time in five, else 1 to 4 random bytes.
fn random_payload(r: &mut JavaRandom) -> Option<Vec<u8>> {
    let len = r.next_int_bounded(5) as usize;
    if len == 0 {
        return None;
    }
    let mut payload = vec![0; len];
    r.next_bytes(&mut payload);
    Some(payload)
}

/// `RandomDocument`: fields with distinct names, each a random token stream.
#[derive(Debug, Clone)]
pub struct RandomDocument {
    pub field_names: Vec<String>,
    pub options: Options,
    pub token_streams: Vec<RandomTokenStream>,
}

impl RandomDocument {
    /// `toDocument()`.
    pub fn to_document(&self) -> Document {
        let ft = field_type(self.options);
        let mut doc = Document::new();
        for (name, tk) in self.field_names.iter().zip(&self.token_streams) {
            doc.add(
                Field::from_token_stream(name.clone(), tk.tokens(), ft.clone())
                    .expect("a token stream field"),
            );
        }
        doc
    }
}

/// `RandomDocumentFactory`.
#[derive(Debug, Clone)]
pub struct RandomDocumentFactory {
    field_names: Vec<String>,
    terms: Vec<String>,
}

impl RandomDocumentFactory {
    /// `new RandomDocumentFactory(distinctFieldNames, distinctTerms)`.
    pub fn new(t: &mut LuceneTestCase, distinct_field_names: usize, distinct_terms: usize) -> Self {
        let mut names = BTreeSet::new();
        while names.len() < distinct_field_names {
            names.insert(test_util::random_simple_string(t.random()));
            names.remove("id");
        }
        let terms = (0..distinct_terms)
            .map(|_| test_util::random_realistic_unicode_string(t.random()))
            .collect();
        RandomDocumentFactory {
            field_names: names.into_iter().collect(),
            terms,
        }
    }

    /// `newDocument(fieldCount, maxTermCount, options)`.
    pub fn new_document(
        &self,
        t: &mut LuceneTestCase,
        field_count: usize,
        max_term_count: i32,
        options: Options,
    ) -> RandomDocument {
        assert!(
            field_count <= self.field_names.len(),
            "IllegalArgumentException"
        );
        let mut used = BTreeSet::new();
        let mut field_names = Vec::with_capacity(field_count);
        let mut token_streams = Vec::with_capacity(field_count);
        for _ in 0..field_count {
            let name = loop {
                let n = &self.field_names
                    [t.random().next_int_bounded(self.field_names.len() as i32) as usize];
                if !used.contains(n) {
                    break n.clone();
                }
            };
            used.insert(name.clone());
            field_names.push(name);
            let len = test_util::next_int(t.random(), 1, max_term_count) as usize;
            token_streams.push(RandomTokenStream::new(t, len, &self.terms));
        }
        RandomDocument {
            field_names,
            options,
            token_streams,
        }
    }
}

/// `assertEquals(RandomDocument, Fields)`: `fields` are `doc`'s term
/// vectors, read from `leaf`.
///
/// # Panics
/// When they are not.
pub fn assert_doc_equals(
    doc: &RandomDocument,
    leaf: &SegmentReader,
    fields: Option<&TermVectorsDocument>,
) {
    let fields = fields.expect("term vectors");
    assert_eq!(doc.field_names.len(), fields.fields.len());
    let name_of = |n: i32| {
        leaf.field_infos()
            .field_by_number(n)
            .expect("a known field")
            .name
            .clone()
    };
    let fields1: BTreeSet<&String> = doc.field_names.iter().collect();
    let names: Vec<String> = fields
        .fields
        .iter()
        .map(|f| name_of(f.field_number))
        .collect();
    let fields2: BTreeSet<&String> = names.iter().collect();
    assert_eq!(fields1, fields2);
    let (positions, offsets, payloads) = doc.options.flags();
    for (name, tk) in doc.field_names.iter().zip(&doc.token_streams) {
        let at = names.iter().position(|n| n == name).expect("the field");
        let terms = &fields.fields[at];
        let unique: BTreeSet<&[u8]> = tk.terms.iter().map(|s| s.as_bytes()).collect();
        assert_eq!(unique.len(), terms.terms.len(), "{name}: terms.size()");
        assert_eq!(positions, terms.has_positions, "{name}: hasPositions");
        assert_eq!(offsets, terms.has_offsets, "{name}: hasOffsets");
        assert_eq!(
            payloads && tk.has_payloads(),
            terms.has_payloads,
            "{name}: hasPayloads"
        );
        let mut te = TermVectorTermsEnum::new(terms);
        for expected in &unique {
            let next = te.next().expect("next").map(<[u8]>::to_vec);
            assert_eq!(Some(expected.to_vec()), next, "{name}");
            assert_eq!(Some(*expected), te.term(), "{name}");
            assert_eq!(1, te.doc_freq().expect("docFreq"), "{name}");
            let text = std::str::from_utf8(expected).expect("a UTF-8 term");

            let mut postings = te.postings(PostingsFlags::Freqs).expect("postings");
            assert_eq!(0, postings.next_doc().expect("nextDoc"));
            assert_eq!(0, postings.doc_id());
            assert_eq!(tk.freqs[text], postings.freq(), "{name}: freq of {text:?}");
            assert_eq!(NO_MORE_DOCS, postings.next_doc().expect("nextDoc"));

            if !(terms.has_positions || terms.has_offsets) {
                continue;
            }
            let mut dpe = te.postings(PostingsFlags::Positions).expect("postings");
            assert_eq!(0, dpe.next_doc().expect("nextDoc"));
            let freq = dpe.freq();
            assert_eq!(tk.freqs[text], freq, "{name}: freq of {text:?}");
            for _ in 0..freq {
                let position = dpe.next_position().expect("nextPosition");
                let (start, end) = (dpe.start_offset(), dpe.end_offset());
                let indexes = if terms.has_positions {
                    tk.position_to_terms.get(&position)
                } else {
                    tk.start_offset_to_terms.get(&start)
                }
                .expect("a token there");
                let is_term = |i: &usize| tk.terms[*i].as_bytes() == *expected;
                if terms.has_positions {
                    assert!(
                        indexes
                            .iter()
                            .any(|i| is_term(i) && tk.positions[*i] == position),
                        "{name}: position {position}"
                    );
                }
                if terms.has_offsets {
                    assert!(
                        indexes.iter().any(|i| is_term(i)
                            && tk.start_offsets[*i] == start
                            && tk.end_offsets[*i] == end),
                        "{name}: offsets {start}-{end}"
                    );
                }
                if terms.has_payloads {
                    let got = dpe.payload();
                    assert!(
                        indexes
                            .iter()
                            .any(|i| is_term(i) && tk.payloads[*i].as_deref() == got),
                        "{name}: payload {got:?}"
                    );
                }
            }
            assert!(
                dpe.next_position().is_err(),
                "{name}: read past last position"
            );
            assert_eq!(NO_MORE_DOCS, dpe.next_doc().expect("nextDoc"));
        }
        assert_eq!(None, te.next().expect("next"), "{name}");
        // Java seeks to five random terms; here every term, alternating
        // `seekExact` and `seekCeil`.
        for (i, expected) in unique.iter().enumerate() {
            if i % 2 == 0 {
                assert!(te.try_seek_exact(expected).expect("seekExact"), "{name}");
            } else {
                assert_eq!(
                    SeekStatus::Found,
                    te.try_seek_ceil(expected).expect("seekCeil"),
                    "{name}"
                );
            }
        }
    }
}

/// `addId(doc, id)`.
pub fn add_id(mut doc: Document, id: &str) -> Document {
    doc.add(StringField::new("id", id, Store::No));
    doc
}

/// `termVectors.get(docID)` on a reader, with the segment it came from.
fn tv(reader: &DirectoryReader, doc: i32) -> (&SegmentReader, Option<TermVectorsDocument>) {
    let (leaf, local) = leaf_of(reader, doc);
    (leaf, leaf.term_vectors(local).expect("term vectors"))
}

fn assert_tv(doc: &RandomDocument, reader: &DirectoryReader, id: i32) {
    let (leaf, fields) = tv(reader, id);
    assert_doc_equals(doc, leaf, fields.as_ref());
}

/// `testRareVectors`: only one document has vectors.
pub fn test_rare_vectors(t: &mut LuceneTestCase) {
    let factory = RandomDocumentFactory::new(t, 10, 20);
    for options in Options::ALL {
        let num_docs = t.at_least(200);
        let doc_with_vectors = t.random().next_int_bounded(num_docs);
        let dir = ByteBuffersDirectory::new();
        let mut writer = new_riw(t, &dir);
        let field_count = test_util::next_int(t.random(), 1, 3) as usize;
        let doc = factory.new_document(t, field_count, 20, options);
        for i in 0..num_docs {
            if i == doc_with_vectors {
                writer
                    .add_document(&add_id(doc.to_document(), "42"))
                    .expect("addDocument");
            } else {
                writer.add_document(&Document::new()).expect("addDocument");
            }
        }
        let reader = writer.get_reader().expect("getReader");
        let with_vectors = doc_with_id(&reader, "42");
        for _ in 0..10 {
            let id = t.random().next_int_bounded(num_docs);
            let (leaf, fields) = tv(&reader, id);
            if id == with_vectors {
                assert_doc_equals(&doc, leaf, fields.as_ref());
            } else {
                assert!(fields.is_none(), "doc {id}: {fields:?}");
            }
        }
        assert_tv(&doc, &reader, with_vectors);
        writer.close().expect("close");
    }
}

/// `testHighFreqs`.
pub fn test_high_freqs(t: &mut LuceneTestCase) {
    let factory = RandomDocumentFactory::new(t, 3, 5);
    for options in Options::ALL {
        if options == Options::None {
            continue;
        }
        let dir = ByteBuffersDirectory::new();
        let mut writer = new_riw(t, &dir);
        let field_count = test_util::next_int(t.random(), 1, 2) as usize;
        let max = t.at_least(2000);
        let doc = factory.new_document(t, field_count, max, options);
        writer
            .add_document(&doc.to_document())
            .expect("addDocument");
        let reader = writer.get_reader().expect("getReader");
        assert_tv(&doc, &reader, 0);
        writer.close().expect("close");
    }
}

/// `testLotsOfFields`.
pub fn test_lots_of_fields(t: &mut LuceneTestCase) {
    let field_count = if t.nightly() {
        t.at_least(100)
    } else {
        t.at_least(10)
    };
    let factory = RandomDocumentFactory::new(t, field_count as usize, 10);
    for options in Options::ALL {
        let dir = ByteBuffersDirectory::new();
        let mut writer = new_riw(t, &dir);
        let n = test_util::next_int(t.random(), 5, field_count) as usize;
        let doc = factory.new_document(t, n, 5, options);
        writer
            .add_document(&doc.to_document())
            .expect("addDocument");
        let reader = writer.get_reader().expect("getReader");
        assert_tv(&doc, &reader, 0);
        writer.close().expect("close");
    }
}

/// `testMixedOptions`: different options for the same field.
pub fn test_mixed_options(t: &mut LuceneTestCase) {
    let num_fields = test_util::next_int(t.random(), 1, 3) as usize;
    let factory = RandomDocumentFactory::new(t, num_fields, 10);
    for options1 in Options::ALL {
        for options2 in Options::ALL {
            if options1 == options2 {
                continue;
            }
            let dir = ByteBuffersDirectory::new();
            let mut writer = new_riw(t, &dir);
            let doc1 = factory.new_document(t, num_fields, 20, options1);
            let doc2 = factory.new_document(t, num_fields, 20, options2);
            writer
                .add_document(&add_id(doc1.to_document(), "1"))
                .expect("addDocument");
            writer
                .add_document(&add_id(doc2.to_document(), "2"))
                .expect("addDocument");
            let reader = writer.get_reader().expect("getReader");
            assert_tv(&doc1, &reader, doc_with_id(&reader, "1"));
            assert_tv(&doc2, &reader, doc_with_id(&reader, "2"));
            writer.close().expect("close");
        }
    }
}

fn random_options(t: &mut LuceneTestCase) -> Options {
    Options::ALL[t.random().next_int_bounded(Options::ALL.len() as i32) as usize]
}

/// `testRandom`.
pub fn test_random(t: &mut LuceneTestCase) {
    let factory = RandomDocumentFactory::new(t, 5, 20);
    let num_docs = t.at_least(50);
    let docs: Vec<RandomDocument> = (0..num_docs)
        .map(|_| {
            let fc = test_util::next_int(t.random(), 1, 3) as usize;
            let max = test_util::next_int(t.random(), 10, 50);
            let o = random_options(t);
            factory.new_document(t, fc, max, o)
        })
        .collect();
    let dir = ByteBuffersDirectory::new();
    let mut writer = new_riw(t, &dir);
    for (i, d) in docs.iter().enumerate() {
        writer
            .add_document(&add_id(d.to_document(), &i.to_string()))
            .expect("addDocument");
    }
    let reader = writer.get_reader().expect("getReader");
    for (i, d) in docs.iter().enumerate() {
        assert_tv(d, &reader, doc_with_id(&reader, &i.to_string()));
    }
    writer.close().expect("close");
}

fn do_test_merge(
    t: &mut LuceneTestCase,
    index_sort: Option<Vec<IndexSortField>>,
    allow_deletes: bool,
) {
    let factory = RandomDocumentFactory::new(t, 5, 20);
    let num_docs = if t.nightly() {
        t.at_least(100)
    } else {
        t.at_least(10)
    };
    for options in Options::ALL {
        let mut docs: BTreeMap<String, RandomDocument> = BTreeMap::new();
        for i in 0..num_docs {
            let fc = test_util::next_int(t.random(), 1, 3) as usize;
            let max = t.at_least(10);
            docs.insert(i.to_string(), factory.new_document(t, fc, max, options));
        }
        let dir = ByteBuffersDirectory::new();
        let analyzer = mock_analyzer(t);
        let iwc = new_config(t, analyzer);
        let mut writer = riw_with(t, &dir, iwc);
        if let Some(sort) = &index_sort {
            set_index_sort(&mut writer.writer(), sort);
        }
        let mut live: Vec<String> = Vec::new();
        let mut ids: Vec<String> = docs.keys().cloned().collect();
        shuffle(t.random(), &mut ids);
        let verify = |writer: &mut super::random_index_writer::RandomIndexWriter<'_>,
                      live: &[String]| {
            let reader = writer.get_reader().expect("getReader");
            for id in live {
                assert_tv(&docs[id], &reader, doc_with_id(&reader, id));
            }
        };
        for id in ids {
            let mut doc = add_id(docs[&id].to_document(), &id);
            if let Some(sort) = &index_sort {
                for f in sort {
                    doc.add(NumericDocValuesField::new(
                        f.field.clone(),
                        i64::from(test_util::next_int(t.random(), 0, 1024)),
                    ));
                }
            }
            if t.random().next_int_bounded(100) < 5 {
                // add via foreign writer
                let other_dir = ByteBuffersDirectory::new();
                let analyzer = mock_analyzer(t);
                let other_iwc = new_config(t, analyzer);
                let mut other = riw_with(t, &other_dir, other_iwc);
                if let Some(sort) = &index_sort {
                    set_index_sort(&mut other.writer(), sort);
                }
                other.add_document(&doc).expect("addDocument");
                other.commit().expect("commit");
                add_indexes_slowly(&mut writer.writer(), &[&other_dir]);
                other.close().expect("close");
            } else {
                writer.add_document(&doc).expect("addDocument");
            }
            live.push(id);
            if allow_deletes && t.random().next_int_bounded(100) < 20 {
                let at = t.random().next_int_bounded(live.len() as i32) as usize;
                let delete_id = live.remove(at);
                writer
                    .delete_documents(Term::new("id", delete_id))
                    .expect("deleteDocuments");
            }
            if t.rarely() {
                writer.commit().expect("commit");
                verify(&mut writer, &live);
            }
            if t.rarely() {
                riw_force_merge(&mut writer, 1);
                verify(&mut writer, &live);
            }
        }
        verify(&mut writer, &live);
        riw_force_merge(&mut writer, 1);
        verify(&mut writer, &live);
        writer.close().expect("close");
    }
}

fn random_long_sort(t: &mut LuceneTestCase) -> Vec<IndexSortField> {
    let n = test_util::next_int(t.random(), 1, 2);
    (0..n)
        .map(|i| IndexSortField {
            field: format!("sort_field_{i}"),
            reverse: false,
            kind: IndexSortKind::Numeric(NumericSortKey::Long(None)),
        })
        .collect()
}

/// `testMerge`.
pub fn test_merge(t: &mut LuceneTestCase) {
    do_test_merge(t, None, false);
}

/// `testMergeWithDeletes`.
pub fn test_merge_with_deletes(t: &mut LuceneTestCase) {
    do_test_merge(t, None, true);
}

/// `testMergeWithIndexSort`.
pub fn test_merge_with_index_sort(t: &mut LuceneTestCase) {
    let sort = random_long_sort(t);
    do_test_merge(t, Some(sort), false);
}

/// `testMergeWithIndexSortAndDeletes`.
pub fn test_merge_with_index_sort_and_deletes(t: &mut LuceneTestCase) {
    let sort = random_long_sort(t);
    do_test_merge(t, Some(sort), true);
}

/// `testClone`: random reads from two threads at once, so per-thread
/// readers must not share mutable state.
pub fn test_clone(t: &mut LuceneTestCase) {
    let factory = RandomDocumentFactory::new(t, 5, 20);
    let num_docs = t.at_least(50);
    for options in Options::ALL {
        let docs: Vec<RandomDocument> = (0..num_docs)
            .map(|_| {
                let fc = test_util::next_int(t.random(), 1, 3) as usize;
                let max = t.at_least(10);
                factory.new_document(t, fc, max, options)
            })
            .collect();
        let dir = ByteBuffersDirectory::new();
        let mut writer = new_riw(t, &dir);
        for (i, d) in docs.iter().enumerate() {
            writer
                .add_document(&add_id(d.to_document(), &i.to_string()))
                .expect("addDocument");
        }
        let reader = writer.get_reader().expect("getReader");
        for (i, d) in docs.iter().enumerate() {
            assert_tv(d, &reader, doc_with_id(&reader, &i.to_string()));
        }
        let plans: Vec<Vec<usize>> = (0..2)
            .map(|_| {
                let n = t.at_least(100);
                (0..n)
                    .map(|_| t.random().next_int_bounded(num_docs) as usize)
                    .collect()
            })
            .collect();
        let (reader, docs) = (&reader, &docs);
        std::thread::scope(|s| {
            let threads: Vec<_> = plans
                .iter()
                .map(|plan| {
                    s.spawn(move || {
                        for &idx in plan {
                            assert_tv(&docs[idx], reader, doc_with_id(reader, &idx.to_string()));
                        }
                    })
                })
                .collect();
            for th in threads {
                th.join().expect("One thread threw an exception");
            }
        });
        writer.close().expect("close");
    }
}

/// One `testPostingsEnum*` case: `foo` holding `bar` twice, at 0-3 and 4-7,
/// with payloads `pay1` and `pay2` when `payloads`.
fn postings_enum_case(t: &mut LuceneTestCase, positions: bool, offsets: bool, payloads: bool) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut iw = writer_with(&dir, &config);
    let mut ft = FieldType::copy_of(&TextField::type_not_stored());
    for r in [
        ft.set_store_term_vectors(true),
        ft.set_store_term_vector_positions(positions),
        ft.set_store_term_vector_offsets(offsets),
        ft.set_store_term_vector_payloads(payloads),
    ] {
        r.expect("an unfrozen FieldType");
    }
    let mut doc = Document::new();
    if payloads {
        // `new CannedTokenStream(token1, token2)`
        let token = |start, end, payload: &str| FieldToken {
            payload: Some(payload.as_bytes().to_vec()),
            ..FieldToken::new("bar", start, end)
        };
        let tokens = FieldTokens {
            tokens: vec![token(0, 3, "pay1"), token(4, 7, "pay2")],
            ..FieldTokens::default()
        };
        doc.add(Field::from_token_stream("foo", tokens, ft).expect("foo"));
    } else {
        doc.add(Field::from_string("foo", "bar bar", ft).expect("foo"));
    }
    iw.add_fields_document(&doc).expect("addDocument");
    iw.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    let leaf = only_leaf(&reader);
    let tv = leaf.term_vectors(0).expect("termVectors").expect("vectors");
    assert_eq!(tv.fields.len(), 1);
    let mut te = TermVectorTermsEnum::new(&tv.fields[0]);
    assert_eq!(Some(&b"bar"[..]), te.next().expect("next"));
    let stored = Stored {
        positions,
        offsets,
        payloads,
    };
    // Java asks twice per flag: the second time reusing the first enum.
    for _reuse in 0..2 {
        // Simple usage (FREQS).
        let mut postings = te.postings(PostingsFlags::Freqs).expect("postings");
        assert_eq!(-1, postings.doc_id());
        assert_eq!(0, postings.next_doc().expect("nextDoc"));
        assert_eq!(2, postings.freq());
        assert_eq!(NO_MORE_DOCS, postings.next_doc().expect("nextDoc"));
        // Asking for docs only: ok.
        let mut docs_only = te.postings(PostingsFlags::None).expect("postings");
        assert_eq!(-1, docs_only.doc_id());
        assert_eq!(0, docs_only.next_doc().expect("nextDoc"));
        assert!(docs_only.freq() == 1 || docs_only.freq() == 2);
        assert_eq!(NO_MORE_DOCS, docs_only.next_doc().expect("nextDoc"));
    }
    for flags in [
        PostingsFlags::Positions,
        PostingsFlags::Payloads,
        PostingsFlags::Offsets,
        PostingsFlags::All,
    ] {
        for _reuse in 0..2 {
            let mut dpe = te.postings(flags).expect("postings");
            stored.check_bar(dpe.as_mut(), flags);
        }
    }
    iw.close().expect("close");
}

/// What a `testPostingsEnum*` field stores.
struct Stored {
    positions: bool,
    offsets: bool,
    payloads: bool,
}

impl Stored {
    /// `bar`'s two occurrences read under `flags`: exact where the field
    /// stores the data and the flags ask for it; where they do not, Java's
    /// "we don't define what it is" (`-1`/`null`, or the stored value).
    fn check_bar(&self, dpe: &mut dyn PostingsEnum, flags: PostingsFlags) {
        let wants_offsets = matches!(flags, PostingsFlags::Offsets | PostingsFlags::All);
        let wants_payloads = matches!(flags, PostingsFlags::Payloads | PostingsFlags::All);
        let occurrences: [(i32, i32, i32, &[u8]); 2] = [(0, 0, 3, b"pay1"), (1, 4, 7, b"pay2")];
        assert_eq!(-1, dpe.doc_id());
        assert_eq!(0, dpe.next_doc().expect("nextDoc"));
        assert_eq!(2, dpe.freq());
        for (position, start, end, payload) in occurrences {
            let expected = if self.positions { position } else { -1 };
            assert_eq!(expected, dpe.next_position().expect("nextPosition"));
            if self.offsets && wants_offsets {
                assert_eq!((start, end), (dpe.start_offset(), dpe.end_offset()));
            } else {
                assert!(dpe.start_offset() == -1 || dpe.start_offset() == start);
                assert!(dpe.end_offset() == -1 || dpe.end_offset() == end);
            }
            if self.payloads && wants_payloads {
                assert_eq!(Some(payload), dpe.payload());
            } else if !self.payloads {
                assert_eq!(None, dpe.payload());
            } else {
                assert!(dpe.payload().is_none_or(|p| p == payload));
            }
        }
        assert_eq!(NO_MORE_DOCS, dpe.next_doc().expect("nextDoc"));
    }
}

/// `testPostingsEnumFreqs`.
pub fn test_postings_enum_freqs(t: &mut LuceneTestCase) {
    postings_enum_case(t, false, false, false);
}

/// `testPostingsEnumPositions`.
pub fn test_postings_enum_positions(t: &mut LuceneTestCase) {
    postings_enum_case(t, true, false, false);
}

/// `testPostingsEnumOffsets`.
pub fn test_postings_enum_offsets(t: &mut LuceneTestCase) {
    postings_enum_case(t, true, true, false);
}

/// `testPostingsEnumOffsetsWithoutPositions`.
pub fn test_postings_enum_offsets_without_positions(t: &mut LuceneTestCase) {
    postings_enum_case(t, false, true, false);
}

/// `testPostingsEnumPayloads`.
pub fn test_postings_enum_payloads(t: &mut LuceneTestCase) {
    postings_enum_case(t, true, false, true);
}

/// `testPostingsEnumAll`.
pub fn test_postings_enum_all(t: &mut LuceneTestCase) {
    postings_enum_case(t, true, true, true);
}

/// Every test of the suite, with its name.
pub const TESTS: &[SuiteTest] = &[
    ("testRareVectors", test_rare_vectors),
    ("testHighFreqs", test_high_freqs),
    ("testLotsOfFields", test_lots_of_fields),
    ("testMixedOptions", test_mixed_options),
    ("testRandom", test_random),
    ("testMerge", test_merge),
    ("testMergeWithDeletes", test_merge_with_deletes),
    ("testMergeWithIndexSort", test_merge_with_index_sort),
    (
        "testMergeWithIndexSortAndDeletes",
        test_merge_with_index_sort_and_deletes,
    ),
    ("testClone", test_clone),
    ("testPostingsEnumFreqs", test_postings_enum_freqs),
    ("testPostingsEnumPositions", test_postings_enum_positions),
    ("testPostingsEnumOffsets", test_postings_enum_offsets),
    (
        "testPostingsEnumOffsetsWithoutPositions",
        test_postings_enum_offsets_without_positions,
    ),
    ("testPostingsEnumPayloads", test_postings_enum_payloads),
    ("testPostingsEnumAll", test_postings_enum_all),
];
