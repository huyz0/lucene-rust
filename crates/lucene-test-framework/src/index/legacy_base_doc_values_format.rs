//! `LegacyBaseDocValuesFormatTestCase`: the doc values suite without the
//! skipper checks (those are [`super::base_doc_values_format`]), on the one
//! doc values format this port writes (`Lucene90DocValuesFormat`).
//!
//! Differs:
//!
//! - The `IndexSearcher` searches (`count(TermQuery)`, `search(query, n)`)
//!   are postings lookups ([`docs_with_term`]): a `TermQuery` over one
//!   term finds exactly the live documents holding it, and these tests
//!   assert nothing about scores.
//! - `SortedDocValues.termsEnum()`/`intersect` are [`DvTermsEnum`], Java's
//!   `SortedDocValuesTermsEnum` over `lookupOrd`/`lookupTerm`: this port's
//!   doc values have no terms enum of their own. Each enum opens its own
//!   values, so two enums of one field never share an iterator. An
//!   intersection is the list of accepted `(term, ord)` pairs, which is
//!   what the tests walk.
//! - `TestUtil.checkReader(reader)` is CheckIndex over the latest commit
//!   (a commit first when a writer is still open), and
//!   `CheckIndex.testDocValues(reader)` is [`check_leaf_doc_values`].
//! - `DirectoryReader.open(writer)` is a commit and an open.
//! - `maybeWrapWithMergingReader` is not applied (see
//!   [`super::base_stored_fields_format`]).
//! - `FloatDocValuesField` is its `NumericDocValuesField`:
//!   `Float.floatToRawIntBits(value)`.
//! - `testCodecUsesOwnBytes`/`testCodecUsesOwnSortedBytes`: a returned
//!   `Vec<u8>` is always the caller's own, so only the values are checked.
//! - `HashSet`/`HashMap` iteration (`doTestSortedSetVsStoredFields`' unique
//!   values, `testRandomSortedBytes`' documents) is sorted order: the
//!   values drawn differ from Java's, not what is checked about them.
//! - Every directory is a `ByteBuffersDirectory`; `newFSDirectory` writes
//!   the same files.
//! - `@Nightly` `testThreads2`, `testThreads3`, `testEmptyBinaryValueOnPageSizes`
//!   and `testHighOrdsSortedSetDV` take their sizes as parameters
//!   ([`threads2`], [`threads3`], [`empty_binary_value_on_page_sizes`],
//!   [`high_ords_sorted_set_dv`]) so a default run reaches the same code.

use super::base_index_file_format::SuiteTest;
use lucene_search::reader::IndexReader as _;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Barrier;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::DocValuesType;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, IndexableField, NumericDocValuesField,
    SortedDocValuesField, SortedNumericDocValuesField, SortedSetDocValuesField, Store, StoredValue,
    StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::doc_values::{get_binary, get_numeric, get_sorted};
use lucene_search::reader::multi_doc_values;
use lucene_search::reader::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, LeafReader, NumericDocValues,
    SortedDocValues, SortedNumericDocValues, SortedSetDocValues, NO_MORE_DOCS,
};
use lucene_store::{ByteBuffersDirectory, Directory};
use lucene_util::automaton::{
    operations, AutomatonType, CompiledAutomaton, RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT,
};
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{
    docs_with_term, force_merge, get, get_all, leaf_stored, mock_analyzer, new_config, new_riw,
    only_leaf, plain_writer, riw_force_merge, riw_with, stored, IndexFileFormatSuite,
};
use super::index_writer_config::{new_log_merge_policy, Draws, LiveMergePolicy, PolicyKind};
use super::merge_policies::shuffle;
use super::random_index_writer::RandomIndexWriter;
use super::random_postings_tester::run_bytes;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `LegacyBaseDocValuesFormatTestCase` itself: its `addRandomFields`.
#[derive(Debug, Default, Clone, Copy)]
pub struct DocValuesSuite;

impl IndexFileFormatSuite for DocValuesSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        if t.usually() {
            let n = t.random().next_int_bounded(1 << 12);
            doc.add(NumericDocValuesField::new("ndv", i64::from(n)));
            let s = test_util::random_simple_string(t.random());
            doc.add(BinaryDocValuesField::new("bdv", s.into_bytes()));
            let s = test_util::random_simple_string_max(t.random(), 2);
            doc.add(SortedDocValuesField::new("sdv", s.into_bytes()));
        }
        let num_values = t.random().next_int_bounded(5);
        for _ in 0..num_values {
            let s = test_util::random_simple_string_max(t.random(), 2);
            doc.add(SortedSetDocValuesField::new("ssdv", s.into_bytes()));
        }
        let num_values = t.random().next_int_bounded(5);
        for _ in 0..num_values {
            let v = test_util::next_long(t.random(), i64::MIN, i64::MAX);
            doc.add(SortedNumericDocValuesField::new("sndv", v));
        }
    }
}

// --- iterator shorthands -------------------------------------------------

/// `it.nextDoc()`.
pub fn next<I: DocIdSetIterator + ?Sized>(it: &mut I) -> i32 {
    it.next_doc().expect("nextDoc")
}

/// `it.advance(target)`.
pub fn advance<I: DocIdSetIterator + ?Sized>(it: &mut I, target: i32) -> i32 {
    it.advance(target).expect("advance")
}

/// `it.advanceExact(target)`.
pub fn exact<I: DocValuesIterator + ?Sized>(it: &mut I, target: i32) -> bool {
    it.advance_exact(target).expect("advanceExact")
}

/// `getNumericDocValues(field)`, which must exist.
pub fn numeric<'a>(leaf: &'a SegmentReader, field: &str) -> Box<dyn NumericDocValues + 'a> {
    leaf.numeric_doc_values(field)
        .expect("getNumericDocValues")
        .unwrap_or_else(|| panic!("no numeric doc values for {field}"))
}

/// `getBinaryDocValues(field)`, which must exist.
pub fn binary<'a>(leaf: &'a SegmentReader, field: &str) -> Box<dyn BinaryDocValues + 'a> {
    leaf.binary_doc_values(field)
        .expect("getBinaryDocValues")
        .unwrap_or_else(|| panic!("no binary doc values for {field}"))
}

/// `getSortedDocValues(field)`, which must exist.
pub fn sorted<'a>(leaf: &'a SegmentReader, field: &str) -> Box<dyn SortedDocValues + 'a> {
    leaf.sorted_doc_values(field)
        .expect("getSortedDocValues")
        .unwrap_or_else(|| panic!("no sorted doc values for {field}"))
}

/// `getSortedNumericDocValues(field)`, which must exist.
pub fn sorted_numeric<'a>(
    leaf: &'a SegmentReader,
    field: &str,
) -> Box<dyn SortedNumericDocValues + 'a> {
    leaf.sorted_numeric_doc_values(field)
        .expect("getSortedNumericDocValues")
        .unwrap_or_else(|| panic!("no sorted numeric doc values for {field}"))
}

/// `getSortedSetDocValues(field)`, which must exist.
pub fn sorted_set<'a>(leaf: &'a SegmentReader, field: &str) -> Box<dyn SortedSetDocValues + 'a> {
    leaf.sorted_set_doc_values(field)
        .expect("getSortedSetDocValues")
        .unwrap_or_else(|| panic!("no sorted set doc values for {field}"))
}

/// The current document's values, `docValueCount` of `nextValue`.
pub fn values_of(dv: &mut (dyn SortedNumericDocValues + '_)) -> Vec<i64> {
    (0..dv.doc_value_count())
        .map(|_| dv.next_value().expect("nextValue"))
        .collect()
}

/// The current document's ordinals, `docValueCount` of `nextOrd`.
pub fn ords_of(dv: &mut (dyn SortedSetDocValues + '_)) -> Vec<i64> {
    (0..dv.doc_value_count())
        .map(|_| dv.next_ord().expect("nextOrd"))
        .collect()
}

/// `dv.lookupOrd(dv.ordValue())`.
pub fn current_term(dv: &mut (dyn SortedDocValues + '_)) -> Vec<u8> {
    let ord = dv.ord_value();
    dv.lookup_ord(ord).expect("lookupOrd")
}

// --- terms enums -----------------------------------------------------------

/// The ordinal dictionary a [`DvTermsEnum`] walks.
pub enum OrdValues<'a> {
    /// A `SORTED` field's.
    Sorted(Box<dyn SortedDocValues + 'a>),
    /// A `SORTED_SET` field's.
    SortedSet(Box<dyn SortedSetDocValues + 'a>),
}

impl OrdValues<'_> {
    fn value_count(&self) -> i64 {
        match self {
            OrdValues::Sorted(v) => i64::from(v.value_count()),
            OrdValues::SortedSet(v) => v.value_count(),
        }
    }

    fn lookup_ord(&mut self, ord: i64) -> Vec<u8> {
        match self {
            OrdValues::Sorted(v) => v
                .lookup_ord(i32::try_from(ord).expect("an int ord"))
                .expect("lookupOrd"),
            OrdValues::SortedSet(v) => v.lookup_ord(ord).expect("lookupOrd"),
        }
    }

    fn lookup_term(&mut self, key: &[u8]) -> i64 {
        match self {
            OrdValues::Sorted(v) => i64::from(v.lookup_term(key).expect("lookupTerm")),
            OrdValues::SortedSet(v) => v.lookup_term(key).expect("lookupTerm"),
        }
    }
}

/// A per-document value reader over one leaf.
type LeafWalk = for<'l> fn(&'l SegmentReader) -> Box<dyn FnMut(i32) -> Option<Vec<u8>> + 'l>;

/// `SortedDocValuesTermsEnum`/`SortedSetDocValuesTermsEnum`: a field's
/// terms in ordinal order, positioned by `lookupTerm` and read by
/// `lookupOrd`.
pub struct DvTermsEnum<'a> {
    values: OrdValues<'a>,
    current_ord: i64,
    term: Vec<u8>,
}

/// `next()`: the next term in ordinal order.
impl Iterator for DvTermsEnum<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        self.current_ord += 1;
        if self.current_ord >= self.values.value_count() {
            return None;
        }
        self.term = self.values.lookup_ord(self.current_ord);
        Some(self.term.clone())
    }
}

impl<'a> DvTermsEnum<'a> {
    /// `values.termsEnum()`.
    pub fn new(values: OrdValues<'a>) -> Self {
        DvTermsEnum {
            values,
            current_ord: -1,
            term: Vec::new(),
        }
    }

    /// `termsEnum()` of `leaf`'s `SORTED` field.
    pub fn sorted(leaf: &'a SegmentReader, field: &str) -> Self {
        Self::new(OrdValues::Sorted(sorted(leaf, field)))
    }

    /// `termsEnum()` of `leaf`'s `SORTED_SET` field.
    pub fn sorted_set(leaf: &'a SegmentReader, field: &str) -> Self {
        Self::new(OrdValues::SortedSet(sorted_set(leaf, field)))
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        &self.term
    }

    /// `ord()`.
    pub fn ord(&self) -> i64 {
        self.current_ord
    }

    /// `seekCeil(text)`.
    pub fn seek_ceil_term(&mut self, text: &[u8]) -> SeekStatus {
        let ord = self.values.lookup_term(text);
        if ord >= 0 {
            self.current_ord = ord;
            self.term = text.to_vec();
            return SeekStatus::Found;
        }
        self.current_ord = -ord - 1;
        if self.current_ord == self.values.value_count() {
            SeekStatus::End
        } else {
            self.term = self.values.lookup_ord(self.current_ord);
            SeekStatus::NotFound
        }
    }

    /// `seekExact(text)`.
    pub fn seek_exact_term(&mut self, text: &[u8]) -> bool {
        let ord = self.values.lookup_term(text);
        if ord >= 0 {
            self.current_ord = ord;
            self.term = text.to_vec();
            true
        } else {
            false
        }
    }

    /// `seekExact(ord)`.
    pub fn seek_exact_ord(&mut self, ord: i64) {
        self.current_ord = ord;
        self.term = self.values.lookup_ord(ord);
    }

    /// `values.intersect(compiled)`, walked: every accepted term with its
    /// ordinal, in order.
    pub fn intersect(mut self, compiled: &CompiledAutomaton) -> Vec<(Vec<u8>, i64)> {
        let mut out = Vec::new();
        match compiled.automaton_type {
            AutomatonType::NONE => {}
            AutomatonType::SINGLE => {
                let term = compiled.term.as_deref().expect("a SINGLE automaton's term");
                if self.seek_exact_term(term) {
                    out.push((term.to_vec(), self.ord()));
                }
            }
            AutomatonType::ALL | AutomatonType::NORMAL => {
                let run = compiled.run_automaton.as_ref().map(|r| &r.0);
                while let Some(term) = self.next() {
                    if run.is_none_or(|run| run_bytes(run, &term)) {
                        out.push((term, self.ord()));
                    }
                }
            }
        }
        out
    }
}

/// `new CompiledAutomaton(determinize(new RegExp(re).toAutomaton()))`.
///
/// # Panics
/// When `re` does not compile.
pub fn compiled(re: &str) -> CompiledAutomaton {
    let a = RegExp::new(re)
        .expect("a regexp")
        .to_automaton()
        .expect("an automaton");
    let a = operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).expect("determinize");
    CompiledAutomaton::new(&a).expect("compile")
}

/// The terms enum checks every merged-away `SORTED`/`SORTED_SET` test ends
/// with: nothing to find, `lookupTerm` says insert at 0.
fn assert_empty_dictionary(mut te: DvTermsEnum<'_>) {
    assert!(!te.seek_exact_term(b"lucene"));
    assert_eq!(SeekStatus::End, te.seek_ceil_term(b"lucene"));
    assert_eq!(-1, te.values.lookup_term(b"lucene"));
}

// --- writers -----------------------------------------------------------------

/// `RandomIndexWriter` over `newIndexWriterConfig(new MockAnalyzer(random()))`
/// with `setMergePolicy(newLogMergePolicy())`.
pub fn log_riw<'d>(t: &mut LuceneTestCase, dir: &'d dyn Directory) -> RandomIndexWriter<'d> {
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
    riw_with(t, dir, config)
}

/// `new IndexWriter(dir, new IndexWriterConfig(null).setMergePolicy(newLogMergePolicy()))`.
pub fn log_writer<'d>(t: &mut LuceneTestCase, dir: &'d dyn Directory) -> IndexWriter<'d> {
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    let mut w = plain_writer(dir);
    w.set_pluggable_merge_policy(Some(std::sync::Arc::new(LiveMergePolicy::new(
        PolicyKind::Log(log),
    ))));
    w
}

/// `iwriter.addDocument(doc)`.
pub fn add(w: &mut RandomIndexWriter<'_>, doc: &Document) {
    w.add_document(doc).expect("addDocument");
}

/// A document of the given fields.
pub fn doc_of(fields: Vec<Box<dyn IndexableField>>) -> Document {
    let mut d = Document::new();
    for f in fields {
        d.add_boxed(f);
    }
    d
}

/// `w.getReader()` then `w.close()`.
pub fn reader_then_close(mut w: RandomIndexWriter<'_>) -> DirectoryReader {
    let reader = w.get_reader().expect("getReader");
    w.close().expect("close");
    reader
}

/// `TestUtil.checkReader` over what `w` has written: a commit, then
/// CheckIndex over it.
pub fn check_riw(w: &mut RandomIndexWriter<'_>, dir: &dyn Directory) {
    w.commit().expect("commit");
    test_util::check_index(dir).expect("CheckIndex");
}

/// The `*MergeAwayAllValues` tests' index: a document without `field`
/// (`id` 0) and one with it (`id` 1), committed; `id` 1 deleted and the
/// index merged to one segment.
pub fn merged_away(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    field: Box<dyn IndexableField>,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    add(
        &mut w,
        &doc_of(vec![Box::new(StringField::new("id", "0", Store::No))]),
    );
    add(
        &mut w,
        &doc_of(vec![
            Box::new(StringField::new("id", "1", Store::No)),
            field,
        ]),
    );
    w.commit().expect("commit");
    w.delete_documents(Term::new("id", "1"))
        .expect("deleteDocuments");
    riw_force_merge(&mut w, 1);
    reader_then_close(w)
}

/// The `*MergeAwayAllValuesLargeSegment` tests' index: `field` on `id` 1,
/// then `atLeast(1024)` empty documents (a sparse field), `id` 1 deleted.
pub fn merged_away_large(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    field: Box<dyn IndexableField>,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    add(
        &mut w,
        &doc_of(vec![
            Box::new(StringField::new("id", "1", Store::No)),
            field,
        ]),
    );
    let num_empty_docs = t.at_least(1024);
    for _ in 0..num_empty_docs {
        add(&mut w, &Document::new());
    }
    w.commit().expect("commit");
    w.delete_documents(Term::new("id", "1"))
        .expect("deleteDocuments");
    riw_force_merge(&mut w, 1);
    reader_then_close(w)
}

// --- the one-document searches -------------------------------------------

/// The long term the one-document tests index.
pub fn long_term() -> String {
    "longterm".repeat(18)
}

/// `"This is the text to be indexed. " + longTerm`.
pub fn text() -> String {
    format!("This is the text to be indexed. {}", long_term())
}

/// Index one document -- `text()` stored in `fieldname`, plus `fields` --
/// through a `RandomIndexWriter`, close it, and open the index.
fn one_text_doc(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    fields: Vec<Box<dyn IndexableField>>,
) -> DirectoryReader {
    let mut w = new_riw(t, dir);
    let mut doc = doc_of(vec![Box::new(TextField::new(
        "fieldname",
        text(),
        Store::Yes,
    ))]);
    for f in fields {
        doc.add_boxed(f);
    }
    add(&mut w, &doc);
    w.close().expect("close");
    DirectoryReader::open(dir).expect("open")
}

/// The searches: one hit for the long term, one for `text`, whose stored
/// `fieldname` is the text. The hit, in the only segment.
fn hit(reader: &DirectoryReader) -> i32 {
    assert_eq!(
        1,
        docs_with_term(reader, "fieldname", long_term().as_bytes()).len()
    );
    let hits = docs_with_term(reader, "fieldname", b"text");
    assert_eq!(1, hits.len());
    let doc = hits[0];
    assert_eq!(
        Some(&StoredValue::String(text())),
        get(&stored(reader, doc), "fieldname")
    );
    assert_eq!(1, reader.segment_readers().len());
    doc
}

/// `testOneNumber`.
pub fn test_one_number(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(t, &dir, vec![Box::new(NumericDocValuesField::new("dv", 5))]);
    let doc = hit(&reader);
    let mut dv = numeric(only_leaf(&reader), "dv");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(5, dv.long_value());
}

/// `testOneFloat`.
pub fn test_one_float(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    // `FloatDocValuesField`: the float's raw bits.
    let bits = i64::from(5.7f32.to_bits() as i32);
    let reader = one_text_doc(
        t,
        &dir,
        vec![Box::new(NumericDocValuesField::new("dv", bits))],
    );
    let doc = hit(&reader);
    let mut dv = numeric(only_leaf(&reader), "dv");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(bits, dv.long_value());
}

/// `testTwoNumbers`.
pub fn test_two_numbers(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(
        t,
        &dir,
        vec![
            Box::new(NumericDocValuesField::new("dv1", 5)),
            Box::new(NumericDocValuesField::new("dv2", 17)),
        ],
    );
    let doc = hit(&reader);
    let leaf = only_leaf(&reader);
    let mut dv = numeric(leaf, "dv1");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(5, dv.long_value());
    let mut dv = numeric(leaf, "dv2");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(17, dv.long_value());
}

/// `testTwoBinaryValues`.
pub fn test_two_binary_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(
        t,
        &dir,
        vec![
            Box::new(BinaryDocValuesField::new("dv1", long_term().into_bytes())),
            Box::new(BinaryDocValuesField::new("dv2", text().into_bytes())),
        ],
    );
    let doc = hit(&reader);
    let leaf = only_leaf(&reader);
    let mut dv = binary(leaf, "dv1");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(long_term().as_bytes(), dv.binary_value());
    let mut dv = binary(leaf, "dv2");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(text().as_bytes(), dv.binary_value());
}

/// `testVariouslyCompressibleBinaryValues`.
pub fn test_variously_compressible_binary_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = new_riw(t, &dir);
    let num_docs = 1 + t.random().next_int_bounded(100);
    let mut written = BTreeMap::new();
    // Small vocabulary ranges will be highly compressible
    let vocab_range = 1 + t.random().next_int_bounded(i32::from(i8::MAX) - 1);
    for i in 0..num_docs {
        let len = 500 + t.random().next_int_bounded(1024);
        let value: Vec<u8> = (0..len)
            .map(|_| t.random().next_int_bounded(vocab_range) as u8)
            .collect();
        written.insert(i, value.clone());
        let doc = doc_of(vec![
            Box::new(TextField::new("id", i.to_string(), Store::Yes)),
            Box::new(BinaryDocValuesField::new("dv1", value)),
        ]);
        add(&mut w, &doc);
    }
    riw_force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    let leaf = only_leaf(&reader);
    for i in 0..num_docs {
        let id = i.to_string();
        let hits = docs_with_term(&reader, "id", id.as_bytes());
        assert_eq!(1, hits.len());
        let doc = hits[0];
        assert_eq!(
            Some(&StoredValue::String(id)),
            get(&stored(&reader, doc), "id")
        );
        let mut dv = binary(leaf, "dv1");
        assert_eq!(doc, advance(dv.as_mut(), doc));
        assert_eq!(written[&i].as_slice(), dv.binary_value());
    }
}

/// `testTwoFieldsMixed`.
pub fn test_two_fields_mixed(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(
        t,
        &dir,
        vec![
            Box::new(NumericDocValuesField::new("dv1", 5)),
            Box::new(BinaryDocValuesField::new("dv2", b"hello world".to_vec())),
        ],
    );
    let doc = hit(&reader);
    let leaf = only_leaf(&reader);
    let mut dv = numeric(leaf, "dv1");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(5, dv.long_value());
    let mut dv2 = binary(leaf, "dv2");
    assert_eq!(doc, advance(dv2.as_mut(), doc));
    assert_eq!(b"hello world", dv2.binary_value());
}

/// `testThreeFieldsMixed` (`sorted`, `numeric`, `binary` name the fields
/// `dv1`, `dv2`, `dv3`) and `testThreeFieldsMixed2` (`dv2`, `dv3`, `dv1`).
fn three_fields_mixed(t: &mut LuceneTestCase, sorted_f: &str, numeric_f: &str, binary_f: &str) {
    let dir = ByteBuffersDirectory::new();
    let mut fields: Vec<(&str, Box<dyn IndexableField>)> = vec![
        (
            sorted_f,
            Box::new(SortedDocValuesField::new(sorted_f, b"hello hello".to_vec())),
        ),
        (
            numeric_f,
            Box::new(NumericDocValuesField::new(numeric_f, 5)),
        ),
        (
            binary_f,
            Box::new(BinaryDocValuesField::new(binary_f, b"hello world".to_vec())),
        ),
    ];
    // Added in field-name order, as Java's two tests do.
    fields.sort_by_key(|(name, _)| *name);
    let reader = one_text_doc(t, &dir, fields.into_iter().map(|(_, f)| f).collect());
    let doc = hit(&reader);
    let leaf = only_leaf(&reader);
    let mut dv = sorted(leaf, sorted_f);
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(b"hello hello".to_vec(), current_term(dv.as_mut()));
    let mut dv2 = numeric(leaf, numeric_f);
    assert_eq!(doc, advance(dv2.as_mut(), doc));
    assert_eq!(5, dv2.long_value());
    let mut dv3 = binary(leaf, binary_f);
    assert_eq!(doc, advance(dv3.as_mut(), doc));
    assert_eq!(b"hello world", dv3.binary_value());
}

/// `testThreeFieldsMixed`.
pub fn test_three_fields_mixed(t: &mut LuceneTestCase) {
    three_fields_mixed(t, "dv1", "dv2", "dv3");
}

/// `testThreeFieldsMixed2`.
pub fn test_three_fields_mixed2(t: &mut LuceneTestCase) {
    three_fields_mixed(t, "dv2", "dv3", "dv1");
}

/// Index `docs` through a log-merge-policy writer, merge to one segment,
/// close, and open.
fn merged(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    docs: Vec<Vec<Box<dyn IndexableField>>>,
    commit_after_first: bool,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    for (i, fields) in docs.into_iter().enumerate() {
        add(&mut w, &doc_of(fields));
        if i == 0 && commit_after_first {
            w.commit().expect("commit");
        }
    }
    riw_force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    reader
}

/// `testTwoDocumentsNumeric`.
pub fn test_two_documents_numeric(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(NumericDocValuesField::new("dv", 1))],
            vec![Box::new(NumericDocValuesField::new("dv", 2))],
        ],
        false,
    );
    let mut dv = numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(1, dv.long_value());
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(2, dv.long_value());
}

/// The stored `id` of a segment's document.
fn id_of(leaf: &SegmentReader, doc: i32) -> String {
    match get(&leaf_stored(leaf, doc), "id") {
        Some(StoredValue::String(s)) => s.clone(),
        other => panic!("doc {doc} has no string id: {other:?}"),
    }
}

/// `testTwoDocumentsMerged`.
pub fn test_two_documents_merged(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![
                Box::new(StringField::new("id", "0", Store::Yes)),
                Box::new(NumericDocValuesField::new("dv", -10)),
            ],
            vec![
                Box::new(StringField::new("id", "1", Store::Yes)),
                Box::new(NumericDocValuesField::new("dv", 99)),
            ],
        ],
        true,
    );
    let leaf = only_leaf(&reader);
    let mut dv = numeric(leaf, "dv");
    for i in 0..2 {
        let expected = if id_of(leaf, i) == "0" { -10 } else { 99 };
        assert_eq!(i, next(dv.as_mut()));
        assert_eq!(expected, dv.long_value());
    }
}

/// `testBigNumericRange` and `testBigNumericRange2`: two values, read back.
fn big_numeric_range(t: &mut LuceneTestCase, a: i64, b: i64) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(NumericDocValuesField::new("dv", a))],
            vec![Box::new(NumericDocValuesField::new("dv", b))],
        ],
        false,
    );
    let mut dv = numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(a, dv.long_value());
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(b, dv.long_value());
}

/// `testBigNumericRange`.
pub fn test_big_numeric_range(t: &mut LuceneTestCase) {
    big_numeric_range(t, i64::MIN, i64::MAX);
}

/// `testBigNumericRange2`.
pub fn test_big_numeric_range2(t: &mut LuceneTestCase) {
    big_numeric_range(t, -8_841_491_950_446_638_677, 9_062_230_939_892_376_225);
}

/// `testBytes`.
pub fn test_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(
        t,
        &dir,
        vec![Box::new(BinaryDocValuesField::new(
            "dv",
            b"hello world".to_vec(),
        ))],
    );
    let doc = hit(&reader);
    let mut dv = binary(only_leaf(&reader), "dv");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(b"hello world", dv.binary_value());
}

/// `testBytesTwoDocumentsMerged`.
pub fn test_bytes_two_documents_merged(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![
                Box::new(StringField::new("id", "0", Store::Yes)),
                Box::new(BinaryDocValuesField::new("dv", b"hello world 1".to_vec())),
            ],
            vec![
                Box::new(StringField::new("id", "1", Store::Yes)),
                Box::new(BinaryDocValuesField::new("dv", b"hello 2".to_vec())),
            ],
        ],
        true,
    );
    let leaf = only_leaf(&reader);
    let mut dv = binary(leaf, "dv");
    for i in 0..2 {
        let expected: &[u8] = if id_of(leaf, i) == "0" {
            b"hello world 1"
        } else {
            b"hello 2"
        };
        assert_eq!(i, next(dv.as_mut()));
        assert_eq!(expected, dv.binary_value());
    }
}

/// `testBytesMergeAwayAllValues`.
pub fn test_bytes_merge_away_all_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("field", b"hi".to_vec())),
    );
    let mut dv = binary(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testSortedBytes`.
pub fn test_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_text_doc(
        t,
        &dir,
        vec![Box::new(SortedDocValuesField::new(
            "dv",
            b"hello world".to_vec(),
        ))],
    );
    let doc = hit(&reader);
    let mut dv = sorted(only_leaf(&reader), "dv");
    assert_eq!(doc, advance(dv.as_mut(), doc));
    assert_eq!(b"hello world".to_vec(), current_term(dv.as_mut()));
}

/// `testSortedBytesTwoDocuments`.
pub fn test_sorted_bytes_two_documents(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(SortedDocValuesField::new(
                "dv",
                b"hello world 1".to_vec(),
            ))],
            vec![Box::new(SortedDocValuesField::new(
                "dv",
                b"hello world 2".to_vec(),
            ))],
        ],
        false,
    );
    let mut dv = sorted(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"hello world 1".to_vec(), current_term(dv.as_mut()));
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(b"hello world 2".to_vec(), current_term(dv.as_mut()));
}

/// `testSortedBytesThreeDocuments`.
pub fn test_sorted_bytes_three_documents(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(SortedDocValuesField::new(
                "dv",
                b"hello world 1".to_vec(),
            ))],
            vec![Box::new(SortedDocValuesField::new(
                "dv",
                b"hello world 2".to_vec(),
            ))],
            vec![Box::new(SortedDocValuesField::new(
                "dv",
                b"hello world 1".to_vec(),
            ))],
        ],
        false,
    );
    let mut dv = sorted(only_leaf(&reader), "dv");
    assert_eq!(2, dv.value_count());
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(0, dv.ord_value());
    assert_eq!(
        b"hello world 1".to_vec(),
        dv.lookup_ord(0).expect("lookupOrd")
    );
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(1, dv.ord_value());
    assert_eq!(
        b"hello world 2".to_vec(),
        dv.lookup_ord(1).expect("lookupOrd")
    );
    assert_eq!(2, next(dv.as_mut()));
    assert_eq!(0, dv.ord_value());
}

/// `testSortedBytesTwoDocumentsMerged`.
pub fn test_sorted_bytes_two_documents_merged(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![
                Box::new(StringField::new("id", "0", Store::Yes)),
                Box::new(SortedDocValuesField::new("dv", b"hello world 1".to_vec())),
            ],
            vec![
                Box::new(StringField::new("id", "1", Store::Yes)),
                Box::new(SortedDocValuesField::new("dv", b"hello world 2".to_vec())),
            ],
        ],
        true,
    );
    let leaf = only_leaf(&reader);
    let mut dv = sorted(leaf, "dv");
    assert_eq!(2, dv.value_count()); // 2 ords
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"hello world 1".to_vec(), current_term(dv.as_mut()));
    assert_eq!(
        b"hello world 2".to_vec(),
        dv.lookup_ord(1).expect("lookupOrd")
    );
    for i in 0..2 {
        let expected: &[u8] = if id_of(leaf, i) == "0" {
            b"hello world 1"
        } else {
            b"hello world 2"
        };
        if dv.doc_id() < i {
            assert_eq!(i, next(dv.as_mut()));
        }
        assert_eq!(expected.to_vec(), current_term(dv.as_mut()));
    }
}

/// `testSortedMergeAwayAllValues`.
pub fn test_sorted_merge_away_all_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away(
        t,
        &dir,
        Box::new(SortedDocValuesField::new("field", b"hello".to_vec())),
    );
    let leaf = only_leaf(&reader);
    let mut dv = sorted(leaf, "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
    assert_empty_dictionary(DvTermsEnum::sorted(leaf, "field"));
}

/// `testBytesWithNewline`.
pub fn test_bytes_with_newline(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    add(
        &mut w,
        &doc_of(vec![Box::new(BinaryDocValuesField::new(
            "dv",
            b"hello\nworld\r1".to_vec(),
        ))]),
    );
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    let mut dv = binary(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"hello\nworld\r1", dv.binary_value());
}

/// `testMissingSortedBytes`.
pub fn test_missing_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    add(
        &mut w,
        &doc_of(vec![Box::new(SortedDocValuesField::new(
            "dv",
            b"hello world 2".to_vec(),
        ))]),
    );
    // 2nd doc missing the DV field
    add(&mut w, &Document::new());
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    let mut dv = sorted(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"hello world 2".to_vec(), current_term(dv.as_mut()));
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// The terms enum walk `testSortedTermsEnum` and `testSortedSetTermsEnum`
/// share, over the terms `beer`, `hello`, `world`; `then_aba` is the
/// `SORTED` one's extra `seekCeil("aba")`.
fn check_three_terms<'a>(open: impl Fn() -> DvTermsEnum<'a>, then_aba: bool) {
    let mut te = open();
    // next()
    assert_eq!(Some(b"beer".to_vec()), te.next());
    assert_eq!(0, te.ord());
    assert_eq!(Some(b"hello".to_vec()), te.next());
    assert_eq!(1, te.ord());
    assert_eq!(Some(b"world".to_vec()), te.next());
    assert_eq!(2, te.ord());

    // seekCeil()
    assert_eq!(SeekStatus::NotFound, te.seek_ceil_term(b"ha!"));
    assert_eq!(b"hello", te.term());
    assert_eq!(1, te.ord());
    assert_eq!(SeekStatus::Found, te.seek_ceil_term(b"beer"));
    assert_eq!(b"beer", te.term());
    assert_eq!(0, te.ord());
    assert_eq!(SeekStatus::End, te.seek_ceil_term(b"zzz"));
    if then_aba {
        assert_eq!(SeekStatus::NotFound, te.seek_ceil_term(b"aba"));
        assert_eq!(0, te.ord());
    }

    // seekExact()
    for (ord, term) in [(0, &b"beer"[..]), (1, b"hello"), (2, b"world")] {
        assert!(te.seek_exact_term(term));
        assert_eq!(term, te.term());
        assert_eq!(ord, te.ord());
    }
    assert!(!te.seek_exact_term(b"bogus"));

    // seek(ord)
    for (ord, term) in [(0, &b"beer"[..]), (1, b"hello"), (2, b"world")] {
        te.seek_exact_ord(ord);
        assert_eq!(term, te.term());
        assert_eq!(ord, te.ord());
    }

    // NORMAL automaton
    let normal = compiled(".*l.*");
    assert_eq!(AutomatonType::NORMAL, normal.automaton_type);
    assert_eq!(
        vec![(b"hello".to_vec(), 1), (b"world".to_vec(), 2)],
        open().intersect(&normal)
    );

    // SINGLE automaton
    let single = CompiledAutomaton::new(
        &RegExp::new("hello")
            .expect("a regexp")
            .to_automaton()
            .expect("an automaton"),
    )
    .expect("compile");
    assert_eq!(AutomatonType::SINGLE, single.automaton_type);
    assert_eq!(vec![(b"hello".to_vec(), 1)], open().intersect(&single));
}

/// `testSortedTermsEnum`.
pub fn test_sorted_terms_enum(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    for term in ["hello", "world", "beer"] {
        add(
            &mut w,
            &doc_of(vec![Box::new(SortedDocValuesField::new(
                "field",
                term.as_bytes().to_vec(),
            ))]),
        );
    }
    riw_force_merge(&mut w, 1);
    let reader = reader_then_close(w);
    let leaf = only_leaf(&reader);
    assert_eq!(3, sorted(leaf, "field").value_count());
    check_three_terms(|| DvTermsEnum::sorted(leaf, "field"), true);
}

/// `testEmptySortedBytes`.
pub fn test_empty_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(SortedDocValuesField::new("dv", Vec::new()))],
            vec![Box::new(SortedDocValuesField::new("dv", Vec::new()))],
        ],
        false,
    );
    let mut dv = sorted(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(0, dv.ord_value());
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(0, dv.ord_value());
    assert_eq!(Vec::<u8>::new(), dv.lookup_ord(0).expect("lookupOrd"));
}

/// `testEmptyBytes`.
pub fn test_empty_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged(
        t,
        &dir,
        vec![
            vec![Box::new(BinaryDocValuesField::new("dv", Vec::new()))],
            vec![Box::new(BinaryDocValuesField::new("dv", Vec::new()))],
        ],
        false,
    );
    let mut dv = binary(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert!(dv.binary_value().is_empty());
    assert_eq!(1, next(dv.as_mut()));
    assert!(dv.binary_value().is_empty());
}

/// One document of `make(bytes)` through a log-policy writer, closed and
/// opened.
fn one_doc(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    field: Box<dyn IndexableField>,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    add(&mut w, &doc_of(vec![field]));
    w.close().expect("close");
    let reader = DirectoryReader::open(dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    reader
}

/// `testVeryLargeButLegalBytes`.
pub fn test_very_large_but_legal_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut bytes = vec![0u8; 32766];
    t.random().next_bytes(&mut bytes);
    let reader = one_doc(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("dv", bytes.clone())),
    );
    let mut dv = binary(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(bytes.as_slice(), dv.binary_value());
}

/// `testVeryLargeButLegalSortedBytes`.
pub fn test_very_large_but_legal_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut bytes = vec![0u8; 32766];
    t.random().next_bytes(&mut bytes);
    let reader = one_doc(
        t,
        &dir,
        Box::new(SortedDocValuesField::new("dv", bytes.clone())),
    );
    let mut dv = get_sorted(only_leaf(&reader), "dv").expect("getSorted");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(bytes, current_term(dv.as_mut()));
}

/// `testCodecUsesOwnBytes`.
pub fn test_codec_uses_own_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_doc(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("dv", b"boo!".to_vec())),
    );
    let mut dv = binary(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"boo!", dv.binary_value());
}

/// `testCodecUsesOwnSortedBytes`.
pub fn test_codec_uses_own_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_doc(
        t,
        &dir,
        Box::new(SortedDocValuesField::new("dv", b"boo!".to_vec())),
    );
    let mut dv = get_sorted(only_leaf(&reader), "dv").expect("getSorted");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(b"boo!".to_vec(), current_term(dv.as_mut()));
}

/// `testDocValuesSimple`.
pub fn test_doc_values_simple(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let _analyzer = mock_analyzer(t);
    let mut w = log_writer(t, &dir);
    for i in 0..5 {
        let mut doc = Document::new();
        doc.add(NumericDocValuesField::new("docId", i64::from(i)));
        doc.add(TextField::new("docId", i.to_string(), Store::No));
        w.add_fields_document(&doc).expect("addDocument");
    }
    force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    // The five SHOULD clauses: every document matches one.
    let mut hits: Vec<i32> = (0..5)
        .flat_map(|i: i32| docs_with_term(&reader, "docId", i.to_string().as_bytes()))
        .collect();
    hits.sort_unstable();
    assert_eq!(vec![0, 1, 2, 3, 4], hits);
    let mut dv = numeric(only_leaf(&reader), "docId");
    for i in hits {
        assert_eq!(i, advance(dv.as_mut(), i));
        assert_eq!(i64::from(i), dv.long_value());
    }
}

/// `testRandomSortedBytes`.
pub fn test_random_sorted_bytes(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    let num_docs = t.at_least(100);
    let mut hash: BTreeSet<Vec<u8>> = BTreeSet::new();
    let mut doc_to_string: BTreeMap<String, String> = BTreeMap::new();
    let max_length = test_util::next_int(t.random(), 1, 50);
    for i in 0..num_docs {
        let s = test_util::random_realistic_unicode_string_between(t.random(), 1, max_length);
        hash.insert(s.clone().into_bytes());
        doc_to_string.insert(i.to_string(), s.clone());
        add(
            &mut w,
            &doc_of(vec![
                Box::new(TextField::new("id", i.to_string(), Store::Yes)),
                Box::new(SortedDocValuesField::new("field", s.into_bytes())),
            ]),
        );
    }
    if t.rarely() {
        w.commit().expect("commit");
    }
    let num_docs_no_value = t.at_least(10);
    for _ in 0..num_docs_no_value {
        add(
            &mut w,
            &doc_of(vec![Box::new(TextField::new("id", "noValue", Store::Yes))]),
        );
    }
    if t.rarely() {
        w.commit().expect("commit");
    }
    for i in 0..num_docs {
        let id = (i + num_docs).to_string();
        let s = test_util::random_realistic_unicode_string_between(t.random(), 1, max_length);
        hash.insert(s.clone().into_bytes());
        doc_to_string.insert(id.clone(), s.clone());
        add(
            &mut w,
            &doc_of(vec![
                Box::new(TextField::new("id", id, Store::Yes)),
                Box::new(SortedDocValuesField::new("field", s.into_bytes())),
            ]),
        );
    }
    w.commit().expect("commit");
    let reader = w.get_reader().expect("getReader");
    let mut dv = multi_doc_values::sorted_values(&reader, "field")
        .expect("getSortedValues")
        .expect("field");
    assert_eq!(hash.len() as i32, dv.value_count());
    for (i, expected) in hash.iter().enumerate() {
        let i = i as i32;
        assert_eq!(*expected, dv.lookup_ord(i).expect("lookupOrd"));
        assert_eq!(i, dv.lookup_term(expected).expect("lookupTerm"));
    }
    for (id, value) in &doc_to_string {
        // pk lookup
        let doc_id = docs_with_term(&reader, "id", id.as_bytes())[0];
        let mut dv = multi_doc_values::sorted_values(&reader, "field")
            .expect("getSortedValues")
            .expect("field");
        assert_eq!(doc_id, advance(dv.as_mut(), doc_id));
        assert_eq!(value.as_bytes().to_vec(), current_term(dv.as_mut()));
    }
    drop(dv);
    drop(reader);
    w.close().expect("close");
}

/// `doTestNumericsVsStoredFields(density, longs, minDocs)`.
fn numerics_vs_stored_fields(
    t: &mut LuceneTestCase,
    density: f64,
    longs: &mut dyn FnMut(&mut JavaRandom) -> i64,
    min_docs: i32,
) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    // index some docs
    let num_docs = t.at_least((f64::from(min_docs) * 1.172) as i32);
    // numDocs should be always > 256 so that in case of a codec that optimizes
    // for numbers of values <= 256, all storage layouts are tested
    assert!(num_docs > 256);
    for i in 0..num_docs {
        if t.random().next_double() > density {
            add(&mut w, &Document::new());
            continue;
        }
        let value = longs(t.random());
        add(
            &mut w,
            &doc_of(vec![
                Box::new(StringField::new("id", i.to_string(), Store::No)),
                Box::new(StringField::new("stored", value.to_string(), Store::Yes)),
                Box::new(NumericDocValuesField::new("dv", value)),
            ]),
        );
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    // merge some segments and ensure that at least one of them has more than
    // max(256, minDocs) values
    riw_force_merge(&mut w, (num_docs / 256.max(min_docs)) as usize);
    w.close().expect("close");
    // compare
    assert_dv_iterate(&dir);
}

/// `assertDVIterate(dir)`: every segment's `dv` against its `stored`,
/// one document at a time.
pub fn assert_dv_iterate(dir: &dyn Directory) {
    let reader = DirectoryReader::open(dir).expect("open");
    test_util::check_index(dir).expect("CheckIndex");
    for leaf in reader.segment_readers() {
        let mut dv = get_numeric(leaf, "dv").expect("getNumeric");
        next(dv.as_mut());
        for i in 0..leaf.max_doc() {
            match get(&leaf_stored(leaf, i), "stored") {
                None => assert!(dv.doc_id() > i),
                Some(StoredValue::String(s)) => {
                    assert_eq!(i, dv.doc_id());
                    assert_eq!(s.parse::<i64>().expect("a long"), dv.long_value());
                    next(dv.as_mut());
                }
                Some(other) => panic!("stored is not a string: {other:?}"),
            }
        }
        assert_eq!(NO_MORE_DOCS, dv.doc_id());
    }
}

/// The stored strings named `field` (`document(doc).getValues(field)`).
fn stored_strings(leaf: &SegmentReader, doc: i32, field: &str) -> Vec<String> {
    get_all(&leaf_stored(leaf, doc), field)
        .into_iter()
        .map(|v| match v {
            StoredValue::String(s) => s.clone(),
            other => panic!("{field} is not a string: {other:?}"),
        })
        .collect()
}

/// `document(doc).getBinaryValue(field)`.
fn stored_binary(leaf: &SegmentReader, doc: i32, field: &str) -> Option<Vec<u8>> {
    match get(&leaf_stored(leaf, doc), field) {
        None => None,
        Some(StoredValue::Binary(b)) => Some(b.clone()),
        Some(other) => panic!("{field} is not binary: {other:?}"),
    }
}

/// The positioning `compareStoredFieldWith*DV` draws: `nextDoc`, `advance`
/// or `advanceExact` to `doc`.
fn position<I: DocValuesIterator + ?Sized>(r: &mut JavaRandom, dv: &mut I, doc: i32) {
    match r.next_int_bounded(3) {
        0 => assert_eq!(doc, next(dv)),
        1 => assert_eq!(doc, advance(dv, doc)),
        _ => assert!(exact(dv, doc)),
    }
    assert_eq!(doc, dv.doc_id());
}

/// `compareStoredFieldWithSortedNumericsDV(reader, storedField, dvField)`.
pub fn compare_stored_field_with_sorted_numerics_dv(
    r: &mut JavaRandom,
    reader: &DirectoryReader,
    stored_field: &str,
    dv_field: &str,
) {
    // Reads `doc`'s values `1 + nextInt(3)` times, re-positioning first on
    // every repeat but (randomly) the first.
    fn repeats(
        r: &mut JavaRandom,
        dv: &mut (dyn SortedNumericDocValues + '_),
        doc: i32,
        stored: &[String],
    ) {
        let repeats = 1 + r.next_int_bounded(3);
        for rep in 0..repeats {
            if rep > 0 || r.next_boolean() {
                assert!(exact(dv, doc));
            }
            assert_eq!(stored.len() as i32, dv.doc_value_count());
            let got: Vec<String> = values_of(dv).iter().map(i64::to_string).collect();
            assert_eq!(stored, got.as_slice());
        }
    }
    for leaf in reader.segment_readers() {
        let max_doc = leaf.max_doc();
        let Some(mut dv) = leaf
            .sorted_numeric_doc_values(dv_field)
            .expect("getSortedNumericDocValues")
        else {
            // no stored values at all
            for doc in 0..max_doc {
                assert!(stored_strings(leaf, doc, stored_field).is_empty());
            }
            continue;
        };
        for doc in 0..max_doc {
            let stored = stored_strings(leaf, doc, stored_field);
            if stored.is_empty() {
                assert!(!exact(dv.as_mut(), doc));
                continue;
            }
            position(r, dv.as_mut(), doc);
            repeats(r, dv.as_mut(), doc, &stored);
        }
        // jump with advanceExact
        let iters = 1 + r.next_int_bounded(3);
        for _ in 0..iters {
            let mut dv = sorted_numeric(leaf, dv_field);
            let mut doc = r.next_int_bounded(max_doc);
            while doc < max_doc {
                let stored = stored_strings(leaf, doc, stored_field);
                if exact(dv.as_mut(), doc) {
                    assert_eq!(doc, dv.doc_id());
                    repeats(r, dv.as_mut(), doc, &stored);
                } else {
                    assert!(stored.is_empty());
                }
                doc += r.next_int_bounded(5); // skip some docs
                doc += 1;
            }
        }
        // jump with advance
        for _ in 0..iters {
            let mut dv = sorted_numeric(leaf, dv_field);
            let mut doc = r.next_int_bounded(max_doc);
            while doc != NO_MORE_DOCS {
                let next_doc = advance(dv.as_mut(), doc);
                // no stored fields in between
                let end = if next_doc == NO_MORE_DOCS {
                    max_doc
                } else {
                    next_doc
                };
                for d in doc..end {
                    assert!(stored_strings(leaf, d, stored_field).is_empty());
                }
                doc = next_doc;
                if doc != NO_MORE_DOCS {
                    let stored = stored_strings(leaf, doc, stored_field);
                    repeats(r, dv.as_mut(), doc, &stored);
                    doc = next_doc + 1;
                    doc += r.next_int_bounded(5); // skip some docs
                }
            }
        }
    }
}

/// `doTestSortedNumericsVsStoredFields(counts, values)`.
fn sorted_numerics_vs_stored_fields(
    t: &mut LuceneTestCase,
    counts: &mut dyn FnMut(&mut JavaRandom) -> i64,
    values: &mut dyn FnMut(&mut JavaRandom) -> i64,
) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    // index some docs
    let num_docs = t.at_least(300);
    // numDocs should be always > 256 so that in case of a codec that optimizes
    // for numbers of values <= 256, all storage layouts are tested
    assert!(num_docs > 256);
    for i in 0..num_docs {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        let value_count = counts(t.random());
        let mut value_array = Vec::new();
        for _ in 0..value_count {
            let value = values(t.random());
            value_array.push(value);
            doc.add(SortedNumericDocValuesField::new("dv", value));
        }
        value_array.sort_unstable();
        for v in value_array {
            doc.add(Field::stored("stored", StoredValue::String(v.to_string())));
        }
        add(&mut w, &doc);
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    w.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    test_util::check_index(&dir).expect("CheckIndex");
    compare_stored_field_with_sorted_numerics_dv(t.random(), &reader, "stored", "dv");
    // merge some segments and ensure that at least one of them has more than
    // 256 values
    riw_force_merge(&mut w, (num_docs / 256) as usize);
    let reader = DirectoryReader::open(&dir).expect("open");
    test_util::check_index(&dir).expect("CheckIndex");
    compare_stored_field_with_sorted_numerics_dv(t.random(), &reader, "stored", "dv");
    w.close().expect("close");
}

/// `atLeast(1)` runs of [`numerics_vs_stored_fields`] at `density` (drawn
/// per run when `None`) over `longs`.
fn numerics_runs(
    t: &mut LuceneTestCase,
    density: Option<f64>,
    longs: &mut dyn FnMut(&mut JavaRandom) -> i64,
) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        let d = density.unwrap_or_else(|| t.random().next_double());
        numerics_vs_stored_fields(t, d, longs, 256);
    }
}

/// `testBooleanNumericsVsStoredFields`.
pub fn test_boolean_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, Some(1.0), &mut |r| i64::from(r.next_int_bounded(2)));
}

/// `testSparseBooleanNumericsVsStoredFields`.
pub fn test_sparse_boolean_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, None, &mut |r| i64::from(r.next_int_bounded(2)));
}

/// `testByteNumericsVsStoredFields`.
pub fn test_byte_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, Some(1.0), &mut |r| {
        i64::from(test_util::next_int(
            r,
            i32::from(i8::MIN),
            i32::from(i8::MAX),
        ))
    });
}

/// `testSparseByteNumericsVsStoredFields`.
pub fn test_sparse_byte_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, None, &mut |r| {
        i64::from(test_util::next_int(
            r,
            i32::from(i8::MIN),
            i32::from(i8::MAX),
        ))
    });
}

/// `testShortNumericsVsStoredFields`.
pub fn test_short_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, Some(1.0), &mut |r| {
        i64::from(test_util::next_int(
            r,
            i32::from(i16::MIN),
            i32::from(i16::MAX),
        ))
    });
}

/// `testSparseShortNumericsVsStoredFields`.
pub fn test_sparse_short_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, None, &mut |r| {
        i64::from(test_util::next_int(
            r,
            i32::from(i16::MIN),
            i32::from(i16::MAX),
        ))
    });
}

/// `testIntNumericsVsStoredFields`.
pub fn test_int_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, Some(1.0), &mut |r| i64::from(r.next_int()));
}

/// `testSparseIntNumericsVsStoredFields`.
pub fn test_sparse_int_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, None, &mut |r| i64::from(r.next_int()));
}

/// `testLongNumericsVsStoredFields`.
pub fn test_long_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, Some(1.0), &mut JavaRandom::next_long);
}

/// `testSparseLongNumericsVsStoredFields`.
pub fn test_sparse_long_numerics_vs_stored_fields(t: &mut LuceneTestCase) {
    numerics_runs(t, None, &mut JavaRandom::next_long);
}

/// The binary values a forward walk of `dv` gives each document, in the
/// shape `doTestBinaryVsStoredFields`' loop checks: after `nextDoc`, a
/// document without a stored value must be behind the iterator.
fn binary_walk(leaf: &SegmentReader) -> impl FnMut(i32) -> Option<Vec<u8>> + '_ {
    let mut dv = get_binary(leaf, "dv").expect("getBinary");
    next(dv.as_mut());
    move |i| {
        if dv.doc_id() == i {
            let v = dv.binary_value().to_vec();
            next(dv.as_mut());
            Some(v)
        } else {
            assert!(dv.doc_id() > i);
            None
        }
    }
}

/// The same for a `SORTED` field: `lookupOrd(ordValue())`.
fn sorted_walk(leaf: &SegmentReader) -> impl FnMut(i32) -> Option<Vec<u8>> + '_ {
    let mut dv = get_sorted(leaf, "dv").expect("getSorted");
    next(dv.as_mut());
    move |i| {
        if dv.doc_id() == i {
            let v = current_term(dv.as_mut());
            next(dv.as_mut());
            Some(v)
        } else {
            assert!(dv.doc_id() > i);
            None
        }
    }
}

/// Index `num_docs` documents, each (with probability `density`) holding
/// `bytes()` both stored and in `dv` (`make` builds the doc-values field),
/// delete some, and check, before and after a merge to one segment, that
/// `walk` reads every stored value back.
fn bytes_vs_stored_fields(
    t: &mut LuceneTestCase,
    num_docs: i32,
    density: f64,
    bytes: &mut dyn FnMut(&mut JavaRandom) -> Vec<u8>,
    make: fn(Vec<u8>) -> Box<dyn IndexableField>,
    walk: LeafWalk,
) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    // index some docs
    for i in 0..num_docs {
        if t.random().next_double() > density {
            add(&mut w, &Document::new());
            continue;
        }
        let buffer = bytes(t.random());
        add(
            &mut w,
            &doc_of(vec![
                Box::new(StringField::new("id", i.to_string(), Store::No)),
                Box::new(Field::stored("stored", StoredValue::Binary(buffer.clone()))),
                make(buffer),
            ]),
        );
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    // compare, then compare again after a merge
    for merge in [false, true] {
        if merge {
            riw_force_merge(&mut w, 1);
        }
        let reader = w.get_reader().expect("getReader");
        check_riw(&mut w, &dir);
        for leaf in reader.segment_readers() {
            let mut at = walk(leaf);
            for i in 0..leaf.max_doc() {
                assert_eq!(stored_binary(leaf, i, "stored"), at(i), "doc {i}");
            }
        }
    }
    w.close().expect("close");
}

fn binary_field(b: Vec<u8>) -> Box<dyn IndexableField> {
    Box::new(BinaryDocValuesField::new("dv", b))
}

fn sorted_field(b: Vec<u8>) -> Box<dyn IndexableField> {
    Box::new(SortedDocValuesField::new("dv", b))
}

fn boxed_binary_walk(leaf: &SegmentReader) -> Box<dyn FnMut(i32) -> Option<Vec<u8>> + '_> {
    Box::new(binary_walk(leaf))
}

fn boxed_sorted_walk(leaf: &SegmentReader) -> Box<dyn FnMut(i32) -> Option<Vec<u8>> + '_> {
    Box::new(sorted_walk(leaf))
}

/// `random().nextBytes(new byte[length])`.
fn random_bytes(r: &mut JavaRandom, length: i32) -> Vec<u8> {
    let mut b = vec![0u8; length as usize];
    r.next_bytes(&mut b);
    b
}

/// `doTestBinaryVsStoredFields(density, bytes)`.
fn binary_vs_stored_fields(
    t: &mut LuceneTestCase,
    density: f64,
    bytes: &mut dyn FnMut(&mut JavaRandom) -> Vec<u8>,
) {
    let num_docs = t.at_least(300);
    bytes_vs_stored_fields(t, num_docs, density, bytes, binary_field, boxed_binary_walk);
}

/// `doTestBinaryFixedLengthVsStoredFields(density)`.
fn binary_fixed_length_vs_stored_fields(t: &mut LuceneTestCase, density: f64) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        let fixed_length = test_util::next_int(t.random(), 0, 10);
        binary_vs_stored_fields(t, density, &mut |r| random_bytes(r, fixed_length));
    }
}

/// `testBinaryFixedLengthVsStoredFields`.
pub fn test_binary_fixed_length_vs_stored_fields(t: &mut LuceneTestCase) {
    binary_fixed_length_vs_stored_fields(t, 1.0);
}

/// `testSparseBinaryFixedLengthVsStoredFields`.
pub fn test_sparse_binary_fixed_length_vs_stored_fields(t: &mut LuceneTestCase) {
    let density = t.random().next_double();
    binary_fixed_length_vs_stored_fields(t, density);
}

/// `doTestBinaryVariableLengthVsStoredFields(density)`.
fn binary_variable_length_vs_stored_fields(t: &mut LuceneTestCase, density: f64) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        binary_vs_stored_fields(t, density, &mut |r| {
            let length = r.next_int_bounded(10);
            random_bytes(r, length)
        });
    }
}

/// `testBinaryVariableLengthVsStoredFields`.
pub fn test_binary_variable_length_vs_stored_fields(t: &mut LuceneTestCase) {
    binary_variable_length_vs_stored_fields(t, 1.0);
}

/// `testSparseBinaryVariableLengthVsStoredFields`.
pub fn test_sparse_binary_variable_length_vs_stored_fields(t: &mut LuceneTestCase) {
    let density = t.random().next_double();
    binary_variable_length_vs_stored_fields(t, density);
}

/// `doTestSortedVsStoredFields(numDocs, density, minLength, maxLength)`.
pub fn sorted_vs_stored_fields(
    t: &mut LuceneTestCase,
    num_docs: i32,
    density: f64,
    min_length: i32,
    max_length: i32,
) {
    bytes_vs_stored_fields(
        t,
        num_docs,
        density,
        &mut |r| {
            let length = test_util::next_int(r, min_length, max_length);
            random_bytes(r, length)
        },
        sorted_field,
        boxed_sorted_walk,
    );
}

/// `atLeast(1)` runs of [`sorted_vs_stored_fields`]; a `None` density or
/// fixed length is drawn per run.
fn sorted_runs(t: &mut LuceneTestCase, density: Option<f64>, fixed: bool) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        let (min, max) = if fixed {
            let l = test_util::next_int(t.random(), 1, 10);
            (l, l)
        } else {
            (1, 10)
        };
        let num_docs = t.at_least(300);
        let d = density.unwrap_or_else(|| t.random().next_double());
        sorted_vs_stored_fields(t, num_docs, d, min, max);
    }
}

/// `testSortedFixedLengthVsStoredFields`.
pub fn test_sorted_fixed_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_runs(t, Some(1.0), true);
}

/// `testSparseSortedFixedLengthVsStoredFields`.
pub fn test_sparse_sorted_fixed_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_runs(t, None, true);
}

/// `testSortedVariableLengthVsStoredFields`.
pub fn test_sorted_variable_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_runs(t, Some(1.0), false);
}

/// `testSparseSortedVariableLengthVsStoredFields`.
pub fn test_sparse_sorted_variable_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_runs(t, None, false);
}

/// A `RandomIndexWriter` (default configuration) over `docs`, then
/// `getReader` and close.
fn riw_docs(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    docs: Vec<Vec<Box<dyn IndexableField>>>,
) -> DirectoryReader {
    let mut w = new_riw(t, dir);
    for fields in docs {
        add(&mut w, &doc_of(fields));
    }
    reader_then_close(w)
}

fn ss(field: &str, v: &str) -> Box<dyn IndexableField> {
    Box::new(SortedSetDocValuesField::new(field, v.as_bytes().to_vec()))
}

/// `testSortedSetOneValue`.
pub fn test_sorted_set_one_value(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(t, &dir, vec![vec![ss("field", "hello")]]);
    let mut dv = sorted_set(only_leaf(&reader), "field");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![0], ords_of(dv.as_mut()));
    assert_eq!(b"hello".to_vec(), dv.lookup_ord(0).expect("lookupOrd"));
}

/// `testSortedSetTwoFields`.
pub fn test_sorted_set_two_fields(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(
        t,
        &dir,
        vec![vec![ss("field", "hello"), ss("field2", "world")]],
    );
    let leaf = only_leaf(&reader);
    for (field, value) in [("field", b"hello"), ("field2", b"world")] {
        let mut dv = sorted_set(leaf, field);
        assert_eq!(0, next(dv.as_mut()));
        assert_eq!(vec![0], ords_of(dv.as_mut()));
        assert_eq!(value.to_vec(), dv.lookup_ord(0).expect("lookupOrd"));
    }
}

/// The `SORTED_SET` tests' log-policy index of `docs`, a commit after the
/// first document when `commit_after_first`, merged to one segment, read
/// through `getReader`.
fn sorted_set_merged(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    docs: Vec<Vec<Box<dyn IndexableField>>>,
    commit_after_first: bool,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    for (i, fields) in docs.into_iter().enumerate() {
        add(&mut w, &doc_of(fields));
        if i == 0 && commit_after_first {
            w.commit().expect("commit");
        }
    }
    riw_force_merge(&mut w, 1);
    reader_then_close(w)
}

/// `testSortedSetTwoDocumentsMerged`.
pub fn test_sorted_set_two_documents_merged(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = sorted_set_merged(
        t,
        &dir,
        vec![vec![ss("field", "hello")], vec![ss("field", "world")]],
        true,
    );
    let mut dv = sorted_set(only_leaf(&reader), "field");
    assert_eq!(2, dv.value_count());
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![0], ords_of(dv.as_mut()));
    assert_eq!(b"hello".to_vec(), dv.lookup_ord(0).expect("lookupOrd"));
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(vec![1], ords_of(dv.as_mut()));
    assert_eq!(b"world".to_vec(), dv.lookup_ord(1).expect("lookupOrd"));
}

/// `testSortedSetTwoValues` (`hello`, `world`) and
/// `testSortedSetTwoValuesUnordered` (`world`, `hello`).
fn sorted_set_two_values(t: &mut LuceneTestCase, a: &str, b: &str) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(t, &dir, vec![vec![ss("field", a), ss("field", b)]]);
    let mut dv = sorted_set(only_leaf(&reader), "field");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![0, 1], ords_of(dv.as_mut()));
    assert_eq!(b"hello".to_vec(), dv.lookup_ord(0).expect("lookupOrd"));
    assert_eq!(b"world".to_vec(), dv.lookup_ord(1).expect("lookupOrd"));
}

/// `testSortedSetTwoValues`.
pub fn test_sorted_set_two_values(t: &mut LuceneTestCase) {
    sorted_set_two_values(t, "hello", "world");
}

/// `testSortedSetTwoValuesUnordered`.
pub fn test_sorted_set_two_values_unordered(t: &mut LuceneTestCase) {
    sorted_set_two_values(t, "world", "hello");
}

/// `testSortedSetThreeValuesTwoDocs`.
pub fn test_sorted_set_three_values_two_docs(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = sorted_set_merged(
        t,
        &dir,
        vec![
            vec![ss("field", "hello"), ss("field", "world")],
            vec![ss("field", "hello"), ss("field", "beer")],
        ],
        true,
    );
    let mut dv = sorted_set(only_leaf(&reader), "field");
    assert_eq!(3, dv.value_count());
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![1, 2], ords_of(dv.as_mut()));
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(vec![0, 1], ords_of(dv.as_mut()));
    for (ord, term) in [(0, &b"beer"[..]), (1, b"hello"), (2, b"world")] {
        assert_eq!(term.to_vec(), dv.lookup_ord(ord).expect("lookupOrd"));
    }
}

/// The four `testSortedSetTwoDocuments{Last,First}Missing[Merge]` tests:
/// `hello` on one of two documents (the first unless `first_missing`).
fn sorted_set_one_missing(t: &mut LuceneTestCase, first_missing: bool, commit: bool) {
    let dir = ByteBuffersDirectory::new();
    let docs = if first_missing {
        vec![vec![], vec![ss("field", "hello")]]
    } else {
        vec![vec![ss("field", "hello")], vec![]]
    };
    let reader = sorted_set_merged(t, &dir, docs, commit);
    let mut dv = sorted_set(only_leaf(&reader), "field");
    assert_eq!(1, dv.value_count());
    assert_eq!(i32::from(first_missing), next(dv.as_mut()));
    assert_eq!(vec![0], ords_of(dv.as_mut()));
    assert_eq!(b"hello".to_vec(), dv.lookup_ord(0).expect("lookupOrd"));
}

/// `testSortedSetTwoDocumentsLastMissing`.
pub fn test_sorted_set_two_documents_last_missing(t: &mut LuceneTestCase) {
    sorted_set_one_missing(t, false, false);
}

/// `testSortedSetTwoDocumentsLastMissingMerge`.
pub fn test_sorted_set_two_documents_last_missing_merge(t: &mut LuceneTestCase) {
    sorted_set_one_missing(t, false, true);
}

/// `testSortedSetTwoDocumentsFirstMissing`.
pub fn test_sorted_set_two_documents_first_missing(t: &mut LuceneTestCase) {
    sorted_set_one_missing(t, true, false);
}

/// `testSortedSetTwoDocumentsFirstMissingMerge`.
pub fn test_sorted_set_two_documents_first_missing_merge(t: &mut LuceneTestCase) {
    sorted_set_one_missing(t, true, true);
}

/// `testSortedSetMergeAwayAllValues`.
pub fn test_sorted_set_merge_away_all_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away(t, &dir, ss("field", "hello"));
    let leaf = only_leaf(&reader);
    assert_eq!(0, sorted_set(leaf, "field").value_count());
    assert_empty_dictionary(DvTermsEnum::sorted_set(leaf, "field"));
}

/// `testSortedSetTermsEnum`.
pub fn test_sorted_set_terms_enum(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    add(
        &mut w,
        &doc_of(vec![
            ss("field", "hello"),
            ss("field", "world"),
            ss("field", "beer"),
        ]),
    );
    let reader = reader_then_close(w);
    let leaf = only_leaf(&reader);
    assert_eq!(3, sorted_set(leaf, "field").value_count());
    check_three_terms(|| DvTermsEnum::sorted_set(leaf, "field"), false);
}

/// `compareStoredFieldWithSortedSetDV(reader, storedField, dvField)`.
pub fn compare_stored_field_with_sorted_set_dv(
    r: &mut JavaRandom,
    reader: &DirectoryReader,
    stored_field: &str,
    dv_field: &str,
) {
    // `1 + nextInt(3)` reads of `doc`'s ordinals, as terms.
    fn repeats(
        r: &mut JavaRandom,
        dv: &mut (dyn SortedSetDocValues + '_),
        doc: i32,
        stored: &[String],
    ) {
        let repeats = 1 + r.next_int_bounded(3);
        for rep in 0..repeats {
            if rep > 0 || r.next_boolean() {
                assert!(exact(dv, doc));
            }
            let ords = ords_of(dv);
            let got: Vec<String> = ords
                .into_iter()
                .map(|o| String::from_utf8(dv.lookup_ord(o).expect("lookupOrd")).expect("utf-8"))
                .collect();
            assert_eq!(stored, got.as_slice());
        }
    }
    for leaf in reader.segment_readers() {
        let max_doc = leaf.max_doc();
        let Some(mut dv) = leaf
            .sorted_set_doc_values(dv_field)
            .expect("getSortedSetDocValues")
        else {
            // no stored values at all
            for doc in 0..max_doc {
                assert!(stored_strings(leaf, doc, stored_field).is_empty());
            }
            continue;
        };
        // sequentially
        for doc in 0..max_doc {
            let stored = stored_strings(leaf, doc, stored_field);
            if stored.is_empty() {
                assert!(!exact(dv.as_mut(), doc));
                continue;
            }
            position(r, dv.as_mut(), doc);
            assert_eq!(stored.len() as i32, dv.doc_value_count());
            repeats(r, dv.as_mut(), doc, &stored);
        }
        // jump with advanceExact
        let iters = 1 + r.next_int_bounded(3);
        for _ in 0..iters {
            let mut dv = sorted_set(leaf, dv_field);
            let mut doc = r.next_int_bounded(max_doc);
            while doc < max_doc {
                let stored = stored_strings(leaf, doc, stored_field);
                if exact(dv.as_mut(), doc) {
                    assert_eq!(doc, dv.doc_id());
                    assert_eq!(stored.len() as i32, dv.doc_value_count());
                    repeats(r, dv.as_mut(), doc, &stored);
                } else {
                    assert!(stored.is_empty());
                }
                doc += r.next_int_bounded(5); // skip some docs
                doc += 1;
            }
        }
        // jump with advance
        for _ in 0..iters {
            let mut dv = sorted_set(leaf, dv_field);
            let mut doc = r.next_int_bounded(max_doc);
            while doc != NO_MORE_DOCS {
                let next_doc = advance(dv.as_mut(), doc);
                // no stored fields in between
                let end = if next_doc == NO_MORE_DOCS {
                    max_doc
                } else {
                    next_doc
                };
                for d in doc..end {
                    assert!(stored_strings(leaf, d, stored_field).is_empty());
                }
                doc = next_doc;
                if doc != NO_MORE_DOCS {
                    let stored = stored_strings(leaf, doc, stored_field);
                    repeats(r, dv.as_mut(), doc, &stored);
                    doc = next_doc + 1;
                    doc += r.next_int_bounded(5); // skip some docs
                }
            }
        }
    }
}

/// `doTestSortedSetVsStoredFields(numDocs, minLength, maxLength,
/// maxValuesPerDoc, maxUniqueValues)`.
pub fn sorted_set_vs_stored_fields(
    t: &mut LuceneTestCase,
    num_docs: i32,
    min_length: i32,
    max_length: i32,
    max_values_per_doc: i32,
    max_unique_values: usize,
) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    let mut value_set = BTreeSet::new();
    let mut i = 0;
    while i < 10_000 && value_set.len() < max_unique_values {
        let length = test_util::next_int(t.random(), min_length, max_length);
        value_set.insert(test_util::random_simple_string_max(t.random(), length));
        i += 1;
    }
    let unique_values: Vec<String> = value_set.into_iter().collect();
    // index some docs
    for i in 0..num_docs {
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        let num_values = test_util::next_int(t.random(), 0, max_values_per_doc);
        // create a random set of strings
        let mut values = BTreeSet::new();
        for _ in 0..num_values {
            let pick = t.random().next_int_bounded(unique_values.len() as i32);
            values.insert(unique_values[pick as usize].clone());
        }
        // add ordered to the stored field
        for v in &values {
            doc.add(Field::stored("stored", StoredValue::String(v.clone())));
        }
        // add in any order to the dv field
        let mut unordered: Vec<String> = values.into_iter().collect();
        shuffle(t.random(), &mut unordered);
        for v in unordered {
            doc.add(SortedSetDocValuesField::new("dv", v.into_bytes()));
        }
        add(&mut w, &doc);
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    for merge in [false, true] {
        if merge {
            riw_force_merge(&mut w, 1);
        }
        let reader = w.get_reader().expect("getReader");
        check_riw(&mut w, &dir);
        compare_stored_field_with_sorted_set_dv(t.random(), &reader, "stored", "dv");
    }
    w.close().expect("close");
}

/// `atLeast(1)` runs of [`sorted_set_vs_stored_fields`]; `fixed` draws one
/// length per run.
fn sorted_set_runs(
    t: &mut LuceneTestCase,
    num_docs_at_least: i32,
    lengths: Option<(i32, i32)>,
    max_values_per_doc: i32,
    max_unique_values: usize,
) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        let (min, max) = lengths.unwrap_or_else(|| {
            let l = test_util::next_int(t.random(), 1, 10);
            (l, l)
        });
        let num_docs = t.at_least(num_docs_at_least);
        sorted_set_vs_stored_fields(t, num_docs, min, max, max_values_per_doc, max_unique_values);
    }
}

/// `testSortedSetFixedLengthVsStoredFields`.
pub fn test_sorted_set_fixed_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, None, 16, 100);
}

/// `atLeast(1)` runs of [`sorted_numerics_vs_stored_fields`].
fn sorted_numerics_runs(
    t: &mut LuceneTestCase,
    counts: &mut dyn FnMut(&mut JavaRandom) -> i64,
    values: &mut dyn FnMut(&mut JavaRandom) -> i64,
) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        sorted_numerics_vs_stored_fields(t, counts, values);
    }
}

/// `testSortedNumericsSingleValuedVsStoredFields`.
pub fn test_sorted_numerics_single_valued_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_numerics_runs(t, &mut |_| 1, &mut JavaRandom::next_long);
}

/// `testSortedNumericsSingleValuedMissingVsStoredFields`.
pub fn test_sorted_numerics_single_valued_missing_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_numerics_runs(
        t,
        &mut |r| if r.next_boolean() { 0 } else { 1 },
        &mut JavaRandom::next_long,
    );
}

/// `testSortedNumericsMultipleValuesVsStoredFields`.
pub fn test_sorted_numerics_multiple_values_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_numerics_runs(
        t,
        &mut |r| test_util::next_long(r, 0, 50),
        &mut JavaRandom::next_long,
    );
}

/// `testSortedNumericsFewUniqueSetsVsStoredFields`.
pub fn test_sorted_numerics_few_unique_sets_vs_stored_fields(t: &mut LuceneTestCase) {
    let n = test_util::next_int(t.random(), 2, 6);
    let values: Vec<i64> = (0..n).map(|_| t.random().next_long()).collect();
    sorted_numerics_runs(t, &mut |r| test_util::next_long(r, 0, 6), &mut |r| {
        values[r.next_int_bounded(values.len() as i32) as usize]
    });
}

/// `testSortedSetVariableLengthVsStoredFields`.
pub fn test_sorted_set_variable_length_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, Some((1, 10)), 16, 100);
}

/// `testSortedSetFixedLengthSingleValuedVsStoredFields`.
pub fn test_sorted_set_fixed_length_single_valued_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, None, 1, 100);
}

/// `testSortedSetVariableLengthSingleValuedVsStoredFields`.
pub fn test_sorted_set_variable_length_single_valued_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, Some((1, 10)), 1, 100);
}

/// `testSortedSetFixedLengthFewUniqueSetsVsStoredFields`.
pub fn test_sorted_set_fixed_length_few_unique_sets_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, Some((10, 10)), 6, 6);
}

/// `testSortedSetVariableLengthFewUniqueSetsVsStoredFields`.
pub fn test_sorted_set_variable_length_few_unique_sets_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 300, Some((1, 10)), 6, 6);
}

/// `testSortedSetVariableLengthManyValuesPerDocVsStoredFields`.
pub fn test_sorted_set_variable_length_many_values_per_doc_vs_stored_fields(
    t: &mut LuceneTestCase,
) {
    sorted_set_runs(t, 20, Some((1, 10)), 500, 1000);
}

/// `testSortedSetFixedLengthManyValuesPerDocVsStoredFields`.
pub fn test_sorted_set_fixed_length_many_values_per_doc_vs_stored_fields(t: &mut LuceneTestCase) {
    sorted_set_runs(t, 20, Some((10, 10)), 500, 1000);
}

/// `doTestGCDCompression(density)`.
fn gcd_compression(t: &mut LuceneTestCase, density: f64) {
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        let min = -(i64::from(t.random().next_int_bounded(1 << 30)) << 32);
        let mul = i64::from(t.random().next_int()) & 0xFFFF_FFFF;
        numerics_vs_stored_fields(
            t,
            density,
            &mut |r| min.wrapping_add(mul.wrapping_mul(i64::from(r.next_int_bounded(1 << 20)))),
            256,
        );
    }
}

/// `testGCDCompression`.
pub fn test_gcd_compression(t: &mut LuceneTestCase) {
    gcd_compression(t, 1.0);
}

/// `testSparseGCDCompression`.
pub fn test_sparse_gcd_compression(t: &mut LuceneTestCase) {
    let density = t.random().next_double();
    gcd_compression(t, density);
}

/// `testZeros`.
pub fn test_zeros(t: &mut LuceneTestCase) {
    numerics_vs_stored_fields(t, 1.0, &mut |_| 0, 256);
}

/// `testSparseZeros`.
pub fn test_sparse_zeros(t: &mut LuceneTestCase) {
    let density = t.random().next_double();
    numerics_vs_stored_fields(t, density, &mut |_| 0, 256);
}

/// `testZeroOrMin`.
pub fn test_zero_or_min(t: &mut LuceneTestCase) {
    // try to make GCD compression fail if the format did not anticipate that
    // the GCD of 0 and MIN_VALUE is negative
    let num_iterations = t.at_least(1);
    for _ in 0..num_iterations {
        numerics_vs_stored_fields(
            t,
            1.0,
            &mut |r| if r.next_boolean() { 0 } else { i64::MIN },
            256,
        );
    }
}

/// The `test{Two,Three}{Numbers,Bytes}OneMissing*` tests: documents `id`
/// 0, 1 (and 2 when `third` is given) under a log policy, `dv1` on 0 (and
/// on 2), a commit after document `commit_after` when given, merged to one
/// segment. The reader.
fn one_missing(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    first: Box<dyn IndexableField>,
    third: Option<Box<dyn IndexableField>>,
    commit_after: Option<usize>,
) -> DirectoryReader {
    let mut w = log_riw(t, dir);
    let mut docs = vec![
        vec![
            Box::new(StringField::new("id", "0", Store::Yes)) as Box<dyn IndexableField>,
            first,
        ],
        vec![Box::new(StringField::new("id", "1", Store::Yes))],
    ];
    if let Some(third) = third {
        docs.push(vec![
            Box::new(StringField::new("id", "2", Store::Yes)),
            third,
        ]);
    }
    for (i, fields) in docs.into_iter().enumerate() {
        add(&mut w, &doc_of(fields));
        if commit_after == Some(i) {
            w.commit().expect("commit");
        }
    }
    riw_force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    reader
}

/// `testTwoNumbersOneMissing` and `testTwoNumbersOneMissingWithMerging`.
fn two_numbers_one_missing(t: &mut LuceneTestCase, commit_after: Option<usize>) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_missing(
        t,
        &dir,
        Box::new(NumericDocValuesField::new("dv1", 0)),
        None,
        commit_after,
    );
    let mut dv = numeric(only_leaf(&reader), "dv1");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(0, dv.long_value());
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testTwoNumbersOneMissing`.
pub fn test_two_numbers_one_missing(t: &mut LuceneTestCase) {
    two_numbers_one_missing(t, None);
}

/// `testTwoNumbersOneMissingWithMerging`.
pub fn test_two_numbers_one_missing_with_merging(t: &mut LuceneTestCase) {
    two_numbers_one_missing(t, Some(0));
}

/// `testThreeNumbersOneMissingWithMerging`.
pub fn test_three_numbers_one_missing_with_merging(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_missing(
        t,
        &dir,
        Box::new(NumericDocValuesField::new("dv1", 0)),
        Some(Box::new(NumericDocValuesField::new("dv1", 5))),
        Some(1),
    );
    let mut dv = numeric(only_leaf(&reader), "dv1");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(0, dv.long_value());
    assert_eq!(2, next(dv.as_mut()));
    assert_eq!(5, dv.long_value());
}

/// `testTwoBytesOneMissing` and `testTwoBytesOneMissingWithMerging`.
fn two_bytes_one_missing(t: &mut LuceneTestCase, commit_after: Option<usize>) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_missing(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("dv1", Vec::new())),
        None,
        commit_after,
    );
    let mut dv = binary(only_leaf(&reader), "dv1");
    assert_eq!(0, next(dv.as_mut()));
    assert!(dv.binary_value().is_empty());
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testTwoBytesOneMissing`.
pub fn test_two_bytes_one_missing(t: &mut LuceneTestCase) {
    two_bytes_one_missing(t, None);
}

/// `testTwoBytesOneMissingWithMerging`.
pub fn test_two_bytes_one_missing_with_merging(t: &mut LuceneTestCase) {
    two_bytes_one_missing(t, Some(0));
}

/// `testThreeBytesOneMissingWithMerging`.
pub fn test_three_bytes_one_missing_with_merging(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = one_missing(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("dv1", Vec::new())),
        Some(Box::new(BinaryDocValuesField::new("dv1", b"boo".to_vec()))),
        Some(1),
    );
    let mut dv = binary(only_leaf(&reader), "dv1");
    assert_eq!(0, next(dv.as_mut()));
    assert!(dv.binary_value().is_empty());
    assert_eq!(2, next(dv.as_mut()));
    assert_eq!(b"boo", dv.binary_value());
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `CheckIndex.testDocValues(reader, infoStream, true)`: every doc-values
/// field of `leaf` walked through the iterator its type opens -- documents
/// strictly increasing below `maxDoc`, sorted-numeric values and
/// sorted-set ordinals ascending, ordinals inside the dictionary, which
/// looks up in order. The number of fields walked.
///
/// # Errors
/// What the first broken invariant is.
pub fn check_leaf_doc_values(leaf: &SegmentReader) -> Result<usize, String> {
    fn walk<I: DocIdSetIterator + ?Sized>(
        it: &mut I,
        max_doc: i32,
        mut each: impl FnMut(&mut I) -> Result<(), String>,
    ) -> Result<i32, String> {
        let (mut last, mut count) = (-1, 0);
        loop {
            let doc = it.next_doc().map_err(|e| e.to_string())?;
            if doc == NO_MORE_DOCS {
                return Ok(count);
            }
            if doc <= last || doc >= max_doc {
                return Err(format!("doc {doc} after {last} (maxDoc={max_doc})"));
            }
            each(it)?;
            last = doc;
            count += 1;
        }
    }
    let err = |e: lucene_search::Error| e.to_string();
    let max_doc = leaf.max_doc();
    let mut fields = 0;
    for fi in &leaf.field_infos().fields {
        let name = fi.name.as_str();
        match fi.doc_values_type {
            DocValuesType::None => continue,
            DocValuesType::Numeric => {
                let mut dv = leaf.numeric_doc_values(name).map_err(err)?.ok_or(name)?;
                walk(dv.as_mut(), max_doc, |d| {
                    let _ = d.long_value();
                    Ok(())
                })?;
            }
            DocValuesType::Binary => {
                let mut dv = leaf.binary_doc_values(name).map_err(err)?.ok_or(name)?;
                walk(dv.as_mut(), max_doc, |d| {
                    let _ = d.binary_value();
                    Ok(())
                })?;
            }
            DocValuesType::Sorted => {
                let mut dv = leaf.sorted_doc_values(name).map_err(err)?.ok_or(name)?;
                let count = dv.value_count();
                walk(dv.as_mut(), max_doc, |d| {
                    let ord = d.ord_value();
                    if ord < 0 || ord >= count {
                        return Err(format!("{name}: ord {ord} of {count}"));
                    }
                    Ok(())
                })?;
                check_dictionary((0..count).map(|o| dv.lookup_ord(o).map_err(err)))?;
            }
            DocValuesType::SortedNumeric => {
                let mut dv = leaf
                    .sorted_numeric_doc_values(name)
                    .map_err(err)?
                    .ok_or(name)?;
                walk(dv.as_mut(), max_doc, |d| {
                    let values = values_of(d);
                    if values.is_empty() || values.windows(2).any(|w| w[0] > w[1]) {
                        return Err(format!("{name}: values {values:?}"));
                    }
                    Ok(())
                })?;
            }
            DocValuesType::SortedSet => {
                let mut dv = leaf.sorted_set_doc_values(name).map_err(err)?.ok_or(name)?;
                let count = dv.value_count();
                walk(dv.as_mut(), max_doc, |d| {
                    let ords = ords_of(d);
                    if ords.is_empty()
                        || ords.windows(2).any(|w| w[0] >= w[1])
                        || ords.iter().any(|&o| o < 0 || o >= count)
                    {
                        return Err(format!("{name}: ords {ords:?} of {count}"));
                    }
                    Ok(())
                })?;
                check_dictionary((0..count).map(|o| dv.lookup_ord(o).map_err(err)))?;
            }
        }
        fields += 1;
    }
    Ok(fields)
}

/// A dictionary's terms must ascend strictly.
fn check_dictionary(terms: impl Iterator<Item = Result<Vec<u8>, String>>) -> Result<(), String> {
    let mut last: Option<Vec<u8>> = None;
    for term in terms {
        let term = term?;
        if last.as_ref().is_some_and(|l| *l >= term) {
            return Err(format!("terms out of order at {term:?}"));
        }
        last = Some(term);
    }
    Ok(())
}

/// One `testThreads` reader thread: every segment's values against its
/// stored fields, then CheckIndex.
fn threads_check_leaves(reader: &DirectoryReader, dir: &dyn Directory) {
    for leaf in reader.segment_readers() {
        let mut binaries = binary(leaf, "dvBin");
        let mut sorted_dv = sorted(leaf, "dvSorted");
        let mut numerics = numeric(leaf, "dvNum");
        for j in 0..leaf.max_doc() {
            let stored = leaf_stored(leaf, j);
            let Some(StoredValue::Binary(binary_value)) = get(&stored, "storedBin") else {
                panic!("doc {j} has no storedBin");
            };
            assert_eq!(j, next(binaries.as_mut()));
            assert_eq!(binary_value.as_slice(), binaries.binary_value());
            assert_eq!(j, next(sorted_dv.as_mut()));
            assert_eq!(*binary_value, current_term(sorted_dv.as_mut()));
            let Some(StoredValue::String(expected)) = get(&stored, "storedNum") else {
                panic!("doc {j} has no storedNum");
            };
            assert_eq!(j, next(numerics.as_mut()));
            assert_eq!(
                expected.parse::<i64>().expect("a long"),
                numerics.long_value()
            );
        }
    }
    test_util::check_index(dir).expect("CheckIndex");
}

/// Run `body` on `n` threads released together (Java's `startingGun`).
fn on_threads(n: i32, body: impl Fn() + Sync) {
    let gun = Barrier::new(n as usize);
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                gun.wait();
                body();
            });
        }
    });
}

/// `testThreads`.
pub fn test_threads(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    // index some docs
    let num_docs = t.at_least(300);
    for i in 0..num_docs {
        let length = test_util::next_int(t.random(), 0, 8);
        let buffer = random_bytes(t.random(), length);
        let numeric_value = t.random().next_long();
        add(
            &mut w,
            &doc_of(vec![
                Box::new(StringField::new("id", i.to_string(), Store::No)),
                Box::new(Field::stored(
                    "storedBin",
                    StoredValue::Binary(buffer.clone()),
                )),
                Box::new(BinaryDocValuesField::new("dvBin", buffer.clone())),
                Box::new(SortedDocValuesField::new("dvSorted", buffer)),
                Box::new(Field::stored(
                    "storedNum",
                    StoredValue::String(numeric_value.to_string()),
                )),
                Box::new(NumericDocValuesField::new("dvNum", numeric_value)),
            ]),
        );
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    w.close().expect("close");
    // compare
    let reader = DirectoryReader::open(&dir).expect("open");
    let num_threads = test_util::next_int(t.random(), 2, 7);
    on_threads(num_threads, || threads_check_leaves(&reader, &dir));
}

/// One `testThreads2` reader thread.
fn threads2_check_leaves(reader: &DirectoryReader, dir: &dyn Directory) {
    for leaf in reader.segment_readers() {
        let mut binaries = leaf.binary_doc_values("dvBin").expect("getBinaryDocValues");
        let mut sorted_dv = leaf
            .sorted_doc_values("dvSorted")
            .expect("getSortedDocValues");
        let mut numerics = leaf
            .numeric_doc_values("dvNum")
            .expect("getNumericDocValues");
        let mut sorted_set_dv = leaf
            .sorted_set_doc_values("dvSortedSet")
            .expect("getSortedSetDocValues");
        let mut sorted_numeric_dv = leaf
            .sorted_numeric_doc_values("dvSortedNumeric")
            .expect("getSortedNumericDocValues");
        for j in 0..leaf.max_doc() {
            let stored = leaf_stored(leaf, j);
            if let Some(StoredValue::Binary(binary_value)) = get(&stored, "storedBin") {
                if let Some(binaries) = binaries.as_mut() {
                    assert_eq!(j, next(binaries.as_mut()));
                    assert_eq!(binary_value.as_slice(), binaries.binary_value());
                    let sorted_dv = sorted_dv.as_mut().expect("dvSorted");
                    assert_eq!(j, next(sorted_dv.as_mut()));
                    assert_eq!(*binary_value, current_term(sorted_dv.as_mut()));
                }
            }
            if let Some(StoredValue::String(number)) = get(&stored, "storedNum") {
                if let Some(numerics) = numerics.as_mut() {
                    assert_eq!(j, advance(numerics.as_mut(), j));
                    assert_eq!(
                        number.parse::<i64>().expect("a long"),
                        numerics.long_value()
                    );
                }
            }
            let values = stored_strings(leaf, j, "storedSortedSet");
            if !values.is_empty() {
                let dv = sorted_set_dv.as_mut().expect("dvSortedSet");
                assert_eq!(j, next(dv.as_mut()));
                assert_eq!(values.len() as i32, dv.doc_value_count());
                for s in &values {
                    let ord = dv.next_ord().expect("nextOrd");
                    assert_eq!(
                        s.as_bytes().to_vec(),
                        dv.lookup_ord(ord).expect("lookupOrd")
                    );
                }
            }
            let num_values = stored_strings(leaf, j, "storedSortedNumeric");
            if !num_values.is_empty() {
                let dv = sorted_numeric_dv.as_mut().expect("dvSortedNumeric");
                assert_eq!(j, next(dv.as_mut()));
                assert_eq!(num_values.len() as i32, dv.doc_value_count());
                for v in &num_values {
                    assert_eq!(*v, dv.next_value().expect("nextValue").to_string());
                }
            }
        }
    }
    test_util::check_index(dir).expect("CheckIndex");
}

/// `@Nightly testThreads2` over `num_docs` documents (Java draws
/// 1025..=2047).
pub fn threads2(t: &mut LuceneTestCase, num_docs: i32) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    // index some docs
    for i in 0..num_docs {
        let length = test_util::next_int(t.random(), 0, 8);
        let buffer = random_bytes(t.random(), length);
        let numeric_value = t.random().next_long();
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        if t.random().next_int_bounded(4) > 0 {
            doc.add(Field::stored(
                "storedBin",
                StoredValue::Binary(buffer.clone()),
            ));
            doc.add(BinaryDocValuesField::new("dvBin", buffer.clone()));
            doc.add(SortedDocValuesField::new("dvSorted", buffer));
        }
        if t.random().next_int_bounded(4) > 0 {
            doc.add(Field::stored(
                "storedNum",
                StoredValue::String(numeric_value.to_string()),
            ));
            doc.add(NumericDocValuesField::new("dvNum", numeric_value));
        }
        let num_sorted_set_fields = t.random().next_int_bounded(3);
        let mut values = BTreeSet::new();
        for _ in 0..num_sorted_set_fields {
            values.insert(test_util::random_simple_string(t.random()));
        }
        for v in values {
            doc.add(SortedSetDocValuesField::new(
                "dvSortedSet",
                v.clone().into_bytes(),
            ));
            doc.add(Field::stored("storedSortedSet", StoredValue::String(v)));
        }
        let num_sorted_numeric_fields = t.random().next_int_bounded(3);
        let mut num_values = BTreeSet::new();
        for _ in 0..num_sorted_numeric_fields {
            num_values.insert(test_util::next_long(t.random(), i64::MIN, i64::MAX));
        }
        for l in num_values {
            doc.add(SortedNumericDocValuesField::new("dvSortedNumeric", l));
            doc.add(Field::stored(
                "storedSortedNumeric",
                StoredValue::String(l.to_string()),
            ));
        }
        add(&mut w, &doc);
        if t.random().next_int_bounded(31) == 0 {
            w.commit().expect("commit");
        }
    }
    // delete some docs
    let num_deletions = t.random().next_int_bounded(num_docs / 10);
    for _ in 0..num_deletions {
        let id = t.random().next_int_bounded(num_docs);
        w.delete_documents(Term::new("id", id.to_string()))
            .expect("deleteDocuments");
    }
    w.close().expect("close");
    // compare
    let reader = DirectoryReader::open(&dir).expect("open");
    let num_threads = test_util::next_int(t.random(), 2, 7);
    on_threads(num_threads, || threads2_check_leaves(&reader, &dir));
}

/// `@Nightly testThreads2`.
pub fn test_threads2(t: &mut LuceneTestCase) {
    let num_docs = test_util::next_int(t.random(), 1025, 2047);
    threads2(t, num_docs);
}

/// `@Nightly testThreads3` over `num_docs` documents (Java draws
/// 2025..=2047) and `rounds` reader rounds (Java runs 10).
pub fn threads3(t: &mut LuceneTestCase, num_docs: i32, rounds: i32) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut w = riw_with(t, &dir, config);
    let num_sorted_sets = t.random().next_int_bounded(21);
    let num_binaries = t.random().next_int_bounded(21);
    let num_sorted_nums = t.random().next_int_bounded(21);
    for _ in 0..num_docs {
        let mut doc = Document::new();
        for j in 0..num_sorted_sets {
            for _ in 0..2 {
                let s = test_util::random_simple_string(t.random());
                doc.add(SortedSetDocValuesField::new(
                    format!("ss{j}"),
                    s.into_bytes(),
                ));
            }
        }
        for j in 0..num_binaries {
            let s = test_util::random_simple_string(t.random());
            doc.add(BinaryDocValuesField::new(format!("b{j}"), s.into_bytes()));
        }
        for j in 0..num_sorted_nums {
            for _ in 0..2 {
                let v = test_util::next_long(t.random(), i64::MIN, i64::MAX);
                doc.add(SortedNumericDocValuesField::new(format!("sn{j}"), v));
            }
        }
        add(&mut w, &doc);
    }
    w.close().expect("close");
    // now check with threads
    for _ in 0..rounds {
        let reader = DirectoryReader::open(&dir).expect("open");
        let threads = test_util::next_int(t.random(), 4, 10);
        on_threads(threads, || {
            for leaf in reader.segment_readers() {
                check_leaf_doc_values(leaf).expect("testDocValues");
            }
        });
    }
}

/// `@Nightly testThreads3`.
pub fn test_threads3(t: &mut LuceneTestCase) {
    let num_docs = test_util::next_int(t.random(), 2025, 2047);
    threads3(t, num_docs, 10);
}

/// `@Nightly testEmptyBinaryValueOnPageSizes` up to values of
/// `1 << (max_shift - 1)` bytes (Java goes to `1 << 19`).
pub fn empty_binary_value_on_page_sizes(t: &mut LuceneTestCase, max_shift: u32) {
    // Test larger and larger power-of-two sized values,
    // followed by empty string value:
    for i in 0..max_shift {
        // `codecAcceptsHugeBinaryValues` is true for this format.
        let dir = ByteBuffersDirectory::new();
        let mut w = new_riw(t, &dir);
        let bytes = vec![0u8; 1 << i];
        for _ in 0..4 {
            add(
                &mut w,
                &doc_of(vec![Box::new(BinaryDocValuesField::new(
                    "field",
                    bytes.clone(),
                ))]),
            );
        }
        add(
            &mut w,
            &doc_of(vec![
                Box::new(Field::stored("id", StoredValue::String("5".into()))),
                Box::new(BinaryDocValuesField::new("field", Vec::new())),
            ]),
        );
        let reader = w.get_reader().expect("getReader");
        w.close().expect("close");
        let mut values = multi_doc_values::binary_values(&reader, "field")
            .expect("getBinaryValues")
            .expect("field");
        for j in 0..5 {
            assert_eq!(j, next(values.as_mut()));
            let len = values.binary_value().len();
            assert!(len == 0 || len == 1 << i);
        }
    }
}

/// `@Nightly testEmptyBinaryValueOnPageSizes`.
pub fn test_empty_binary_value_on_page_sizes(t: &mut LuceneTestCase) {
    empty_binary_value_on_page_sizes(t, 20);
}

/// One document of `fields` through a plain writer
/// (`new IndexWriterConfig(null)`), then an empty one when `then_empty`.
fn plain_docs(
    dir: &dyn Directory,
    fields: Vec<Box<dyn IndexableField>>,
    then_empty: bool,
) -> DirectoryReader {
    let mut w = plain_writer(dir);
    w.add_fields_document(&doc_of(fields)).expect("addDocument");
    if then_empty {
        w.add_fields_document(&Document::new())
            .expect("addDocument");
    }
    w.close().expect("close");
    let reader = DirectoryReader::open(dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    reader
}

fn sn(field: &str, v: i64) -> Box<dyn IndexableField> {
    Box::new(SortedNumericDocValuesField::new(field, v))
}

/// `testOneSortedNumber`.
pub fn test_one_sorted_number(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(t, &dir, vec![vec![sn("dv", 5)]]);
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![5], values_of(dv.as_mut()));
}

/// `testOneSortedNumberOneMissing`.
pub fn test_one_sorted_number_one_missing(_t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = plain_docs(&dir, vec![sn("dv", 5)], true);
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![5], values_of(dv.as_mut()));
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testNumberMergeAwayAllValues`.
pub fn test_number_merge_away_all_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away(t, &dir, Box::new(NumericDocValuesField::new("field", 5)));
    let mut dv = numeric(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testTwoSortedNumber`.
pub fn test_two_sorted_number(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(t, &dir, vec![vec![sn("dv", 11), sn("dv", -5)]]);
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![-5, 11], values_of(dv.as_mut()));
}

/// `testTwoSortedNumberSameValue`.
pub fn test_two_sorted_number_same_value(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = riw_docs(t, &dir, vec![vec![sn("dv", 11), sn("dv", 11)]]);
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![11, 11], values_of(dv.as_mut()));
}

/// `testTwoSortedNumberOneMissing`.
pub fn test_two_sorted_number_one_missing(_t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = plain_docs(&dir, vec![sn("dv", 11), sn("dv", -5)], true);
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![-5, 11], values_of(dv.as_mut()));
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testSortedNumberMerge`.
pub fn test_sorted_number_merge(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_writer(t, &dir);
    w.add_fields_document(&doc_of(vec![sn("dv", 11)]))
        .expect("addDocument");
    w.commit().expect("commit");
    w.add_fields_document(&doc_of(vec![sn("dv", -5)]))
        .expect("addDocument");
    force_merge(&mut w, 1);
    w.close().expect("close");
    let reader = DirectoryReader::open(&dir).expect("open");
    assert_eq!(1, reader.segment_readers().len());
    let mut dv = sorted_numeric(only_leaf(&reader), "dv");
    assert_eq!(0, next(dv.as_mut()));
    assert_eq!(vec![11], values_of(dv.as_mut()));
    assert_eq!(1, next(dv.as_mut()));
    assert_eq!(vec![-5], values_of(dv.as_mut()));
}

/// `testSortedNumberMergeAwayAllValues`.
pub fn test_sorted_number_merge_away_all_values(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away(t, &dir, sn("field", 5));
    let mut dv = sorted_numeric(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testSortedEnumAdvanceIndependently`.
pub fn test_sorted_enum_advance_independently(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    for v in ["2", "1", "3"] {
        add(
            &mut w,
            &doc_of(vec![Box::new(SortedDocValuesField::new(
                "field",
                v.as_bytes().to_vec(),
            ))]),
        );
    }
    w.commit().expect("commit");
    riw_force_merge(&mut w, 1);
    let reader = reader_then_close(w);
    let leaf = only_leaf(&reader);
    // `DocValues.singleton(dv)`: the same dictionary, as a set.
    let count = i64::from(sorted(leaf, "field").value_count());
    sorted_set_enum_advance_independently(count, || DvTermsEnum::sorted(leaf, "field"));
}

/// `testSortedSetEnumAdvanceIndependently`.
pub fn test_sorted_set_enum_advance_independently(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    for (a, b) in [("2", "3"), ("1", "3"), ("1", "2")] {
        add(&mut w, &doc_of(vec![ss("field", a), ss("field", b)]));
    }
    w.commit().expect("commit");
    riw_force_merge(&mut w, 1);
    let reader = reader_then_close(w);
    let leaf = only_leaf(&reader);
    let count = sorted_set(leaf, "field").value_count();
    sorted_set_enum_advance_independently(count, || DvTermsEnum::sorted_set(leaf, "field"));
}

/// `doTestSortedSetEnumAdvanceIndependently(dv)`, `open` being
/// `dv.termsEnum()`.
pub fn sorted_set_enum_advance_independently<'a>(
    value_count: i64,
    open: impl Fn() -> DvTermsEnum<'a>,
) {
    if value_count < 2 {
        return;
    }
    let mut te = open();
    let terms = [te.next().expect("a term"), te.next().expect("a term")];

    // Make sure that calls to next() does not modify the term of the other enum
    let (mut enum1, mut enum2) = (open(), open());
    enum1.next();
    let term2 = enum2.next().expect("a term");
    let term1 = enum1.next().expect("a term");
    assert_eq!(term1.as_slice(), enum1.term());
    assert_eq!(term2.as_slice(), enum2.term());

    // Same for seekCeil
    let (mut enum1, mut enum2) = (open(), open());
    let term2 = enum2.next().expect("a term");
    let mut seek_term = terms[0].clone();
    seek_term.push(0);
    enum1.seek_ceil_term(&seek_term);
    let term1 = enum1.term().to_vec();
    assert_eq!(term1.as_slice(), enum1.term());
    assert_eq!(term2.as_slice(), enum2.term());

    // Same for seekCeil on an exact value
    let (mut enum1, mut enum2) = (open(), open());
    let term2 = enum2.next().expect("a term");
    enum1.seek_ceil_term(&terms[1]);
    let term1 = enum1.term().to_vec();
    assert_eq!(term1.as_slice(), enum1.term());
    assert_eq!(term2.as_slice(), enum2.term());

    // Same for seekExact
    let (mut enum1, mut enum2) = (open(), open());
    enum2.next();
    assert!(enum1.seek_exact_term(&terms[1]));

    // Same for seek by ord
    let (mut enum1, mut enum2) = (open(), open());
    let term2 = enum2.next().expect("a term");
    enum1.seek_exact_ord(1);
    let term1 = enum1.term().to_vec();
    assert_eq!(term1.as_slice(), enum1.term());
    assert_eq!(term2.as_slice(), enum2.term());
}

/// `testSortedMergeAwayAllValuesLargeSegment`.
pub fn test_sorted_merge_away_all_values_large_segment(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away_large(
        t,
        &dir,
        Box::new(SortedDocValuesField::new("field", b"hello".to_vec())),
    );
    let leaf = only_leaf(&reader);
    let mut dv = sorted(leaf, "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
    assert_empty_dictionary(DvTermsEnum::sorted(leaf, "field"));
}

/// `testSortedSetMergeAwayAllValuesLargeSegment`.
pub fn test_sorted_set_merge_away_all_values_large_segment(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away_large(t, &dir, ss("field", "hello"));
    let leaf = only_leaf(&reader);
    let mut dv = sorted_set(leaf, "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
    assert_empty_dictionary(DvTermsEnum::sorted_set(leaf, "field"));
}

/// `testNumericMergeAwayAllValuesLargeSegment`.
pub fn test_numeric_merge_away_all_values_large_segment(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away_large(t, &dir, Box::new(NumericDocValuesField::new("field", 42)));
    let mut dv = numeric(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testSortedNumericMergeAwayAllValuesLargeSegment`.
pub fn test_sorted_numeric_merge_away_all_values_large_segment(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away_large(t, &dir, sn("field", 42));
    let mut dv = sorted_numeric(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `testBinaryMergeAwayAllValuesLargeSegment`.
pub fn test_binary_merge_away_all_values_large_segment(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let reader = merged_away_large(
        t,
        &dir,
        Box::new(BinaryDocValuesField::new("field", b"hello".to_vec())),
    );
    let mut dv = binary(only_leaf(&reader), "field");
    assert_eq!(NO_MORE_DOCS, next(dv.as_mut()));
}

/// `doTestRandomAdvance(fieldCreator)`: `field` makes the value, `numeric`
/// picks `MultiDocValues.getNumericValues` over `getBinaryValues`.
fn random_advance(
    t: &mut LuceneTestCase,
    field: &mut dyn FnMut(&mut JavaRandom) -> Box<dyn IndexableField>,
    numeric_field: bool,
) {
    let dir = ByteBuffersDirectory::new();
    let mut w = log_riw(t, &dir);
    let num_chunks = t.at_least(10);
    let mut id = 0;
    let mut missing_set = HashSet::new();
    for _ in 0..num_chunks {
        // change sparseness for each chunk
        let sparse_chance = t.random().next_double();
        let doc_count = t.at_least(1000);
        for _ in 0..doc_count {
            let mut doc = Document::new();
            doc.add(Field::stored("id", StoredValue::Int(id)));
            if t.random().next_double() > sparse_chance {
                doc.add_boxed(field(t.random()));
            } else {
                missing_set.insert(id);
            }
            id += 1;
            add(&mut w, &doc);
        }
    }
    if t.random().next_boolean() {
        riw_force_merge(&mut w, 1);
    }
    // Now search the index:
    let reader = w.get_reader().expect("getReader");
    let max_doc = reader.max_doc();
    let missing: Vec<bool> = (0..max_doc)
        .map(|d| match get(&stored(&reader, d), "id") {
            Some(StoredValue::Int(i)) => missing_set.contains(i),
            other => panic!("doc {d} has no int id: {other:?}"),
        })
        .collect();
    let num_iters = t.at_least(10);
    for _ in 0..num_iters {
        let mut values: Box<dyn DocIdSetIterator + '_> = if numeric_field {
            multi_doc_values::numeric_values(&reader, "field")
                .expect("getNumericValues")
                .expect("field")
        } else {
            multi_doc_values::binary_values(&reader, "field")
                .expect("getBinaryValues")
                .expect("field")
        };
        assert_eq!(-1, values.doc_id());
        loop {
            let doc_id = if t.random().next_boolean() {
                next(values.as_mut())
            } else {
                let range = if t.random().next_int_bounded(10) == 7 {
                    // big jump
                    max_doc - values.doc_id()
                } else {
                    // small jump
                    25
                };
                let inc = test_util::next_int(t.random(), 1, range);
                let target = values.doc_id() + inc;
                advance(values.as_mut(), target)
            };
            if doc_id == NO_MORE_DOCS {
                break;
            }
            assert!(!missing[doc_id as usize], "doc {doc_id} has no value");
        }
    }
    drop(reader);
    w.close().expect("close");
}

/// `testRandomAdvanceNumeric`.
pub fn test_random_advance_numeric(t: &mut LuceneTestCase) {
    let long_range = if t.random().next_boolean() {
        i64::from(test_util::next_int(t.random(), 1, 1024))
    } else {
        test_util::next_long(t.random(), 1, i64::MAX)
    };
    random_advance(
        t,
        &mut |r| {
            Box::new(NumericDocValuesField::new(
                "field",
                test_util::next_long(r, 0, long_range),
            ))
        },
        true,
    );
}

/// `testRandomAdvanceBinary`.
pub fn test_random_advance_binary(t: &mut LuceneTestCase) {
    random_advance(
        t,
        &mut |r| {
            let length = r.next_int_bounded(10);
            Box::new(BinaryDocValuesField::new("field", random_bytes(r, length)))
        },
        false,
    );
}

/// `@Nightly testHighOrdsSortedSetDV` over `num_docs` documents (Java
/// draws 20 000..30 000).
pub fn high_ords_sorted_set_dv(t: &mut LuceneTestCase, num_docs: i32) {
    let dir = ByteBuffersDirectory::new();
    let mut w = plain_writer(&dir);
    let ram = 8.0 + f64::from(t.random().next_int_bounded(64));
    w.set_ram_buffer_size_mb(ram).expect("setRAMBufferSizeMB");
    // many docs with some of them have very high ords
    for _ in 1..num_docs {
        let num_ords = if t.random().next_int_bounded(100) <= 5 {
            1000 + t.random().next_int_bounded(500)
        } else {
            t.random().next_int_bounded(10)
        };
        let mut doc = Document::new();
        for _ in 0..num_ords {
            let term = test_util::random_binary_term_len(t.random(), 2);
            doc.add(SortedSetDocValuesField::new("sorted_set_dv", term));
        }
        w.add_fields_document(&doc).expect("addDocument");
    }
    force_merge(&mut w, 1);
    w.close().expect("close");
    test_util::check_index(&dir).expect("CheckIndex");
    let reader = DirectoryReader::open(&dir).expect("open");
    for leaf in reader.segment_readers() {
        check_leaf_doc_values(leaf).expect("testDocValues");
    }
}

/// `@Nightly testHighOrdsSortedSetDV`.
pub fn test_high_ords_sorted_set_dv(t: &mut LuceneTestCase) {
    let num_docs = 20_000 + t.random().next_int_bounded(10_000);
    high_ords_sorted_set_dv(t, num_docs);
}

/// Every test of the suite, with its name, for a runner to loop over. The
/// `@Nightly` four are last.
pub const TESTS: &[SuiteTest] = &[
    ("testOneNumber", test_one_number),
    ("testOneFloat", test_one_float),
    ("testTwoNumbers", test_two_numbers),
    ("testTwoBinaryValues", test_two_binary_values),
    (
        "testVariouslyCompressibleBinaryValues",
        test_variously_compressible_binary_values,
    ),
    ("testTwoFieldsMixed", test_two_fields_mixed),
    ("testThreeFieldsMixed", test_three_fields_mixed),
    ("testThreeFieldsMixed2", test_three_fields_mixed2),
    ("testTwoDocumentsNumeric", test_two_documents_numeric),
    ("testTwoDocumentsMerged", test_two_documents_merged),
    ("testBigNumericRange", test_big_numeric_range),
    ("testBigNumericRange2", test_big_numeric_range2),
    ("testBytes", test_bytes),
    (
        "testBytesTwoDocumentsMerged",
        test_bytes_two_documents_merged,
    ),
    (
        "testBytesMergeAwayAllValues",
        test_bytes_merge_away_all_values,
    ),
    ("testSortedBytes", test_sorted_bytes),
    (
        "testSortedBytesTwoDocuments",
        test_sorted_bytes_two_documents,
    ),
    (
        "testSortedBytesThreeDocuments",
        test_sorted_bytes_three_documents,
    ),
    (
        "testSortedBytesTwoDocumentsMerged",
        test_sorted_bytes_two_documents_merged,
    ),
    (
        "testSortedMergeAwayAllValues",
        test_sorted_merge_away_all_values,
    ),
    ("testBytesWithNewline", test_bytes_with_newline),
    ("testMissingSortedBytes", test_missing_sorted_bytes),
    ("testSortedTermsEnum", test_sorted_terms_enum),
    ("testEmptySortedBytes", test_empty_sorted_bytes),
    ("testEmptyBytes", test_empty_bytes),
    (
        "testVeryLargeButLegalBytes",
        test_very_large_but_legal_bytes,
    ),
    (
        "testVeryLargeButLegalSortedBytes",
        test_very_large_but_legal_sorted_bytes,
    ),
    ("testCodecUsesOwnBytes", test_codec_uses_own_bytes),
    (
        "testCodecUsesOwnSortedBytes",
        test_codec_uses_own_sorted_bytes,
    ),
    ("testDocValuesSimple", test_doc_values_simple),
    ("testRandomSortedBytes", test_random_sorted_bytes),
    (
        "testBooleanNumericsVsStoredFields",
        test_boolean_numerics_vs_stored_fields,
    ),
    (
        "testSparseBooleanNumericsVsStoredFields",
        test_sparse_boolean_numerics_vs_stored_fields,
    ),
    (
        "testByteNumericsVsStoredFields",
        test_byte_numerics_vs_stored_fields,
    ),
    (
        "testSparseByteNumericsVsStoredFields",
        test_sparse_byte_numerics_vs_stored_fields,
    ),
    (
        "testShortNumericsVsStoredFields",
        test_short_numerics_vs_stored_fields,
    ),
    (
        "testSparseShortNumericsVsStoredFields",
        test_sparse_short_numerics_vs_stored_fields,
    ),
    (
        "testIntNumericsVsStoredFields",
        test_int_numerics_vs_stored_fields,
    ),
    (
        "testSparseIntNumericsVsStoredFields",
        test_sparse_int_numerics_vs_stored_fields,
    ),
    (
        "testLongNumericsVsStoredFields",
        test_long_numerics_vs_stored_fields,
    ),
    (
        "testSparseLongNumericsVsStoredFields",
        test_sparse_long_numerics_vs_stored_fields,
    ),
    (
        "testBinaryFixedLengthVsStoredFields",
        test_binary_fixed_length_vs_stored_fields,
    ),
    (
        "testSparseBinaryFixedLengthVsStoredFields",
        test_sparse_binary_fixed_length_vs_stored_fields,
    ),
    (
        "testBinaryVariableLengthVsStoredFields",
        test_binary_variable_length_vs_stored_fields,
    ),
    (
        "testSparseBinaryVariableLengthVsStoredFields",
        test_sparse_binary_variable_length_vs_stored_fields,
    ),
    (
        "testSortedFixedLengthVsStoredFields",
        test_sorted_fixed_length_vs_stored_fields,
    ),
    (
        "testSparseSortedFixedLengthVsStoredFields",
        test_sparse_sorted_fixed_length_vs_stored_fields,
    ),
    (
        "testSortedVariableLengthVsStoredFields",
        test_sorted_variable_length_vs_stored_fields,
    ),
    (
        "testSparseSortedVariableLengthVsStoredFields",
        test_sparse_sorted_variable_length_vs_stored_fields,
    ),
    ("testSortedSetOneValue", test_sorted_set_one_value),
    ("testSortedSetTwoFields", test_sorted_set_two_fields),
    (
        "testSortedSetTwoDocumentsMerged",
        test_sorted_set_two_documents_merged,
    ),
    ("testSortedSetTwoValues", test_sorted_set_two_values),
    (
        "testSortedSetTwoValuesUnordered",
        test_sorted_set_two_values_unordered,
    ),
    (
        "testSortedSetThreeValuesTwoDocs",
        test_sorted_set_three_values_two_docs,
    ),
    (
        "testSortedSetTwoDocumentsLastMissing",
        test_sorted_set_two_documents_last_missing,
    ),
    (
        "testSortedSetTwoDocumentsLastMissingMerge",
        test_sorted_set_two_documents_last_missing_merge,
    ),
    (
        "testSortedSetTwoDocumentsFirstMissing",
        test_sorted_set_two_documents_first_missing,
    ),
    (
        "testSortedSetTwoDocumentsFirstMissingMerge",
        test_sorted_set_two_documents_first_missing_merge,
    ),
    (
        "testSortedSetMergeAwayAllValues",
        test_sorted_set_merge_away_all_values,
    ),
    ("testSortedSetTermsEnum", test_sorted_set_terms_enum),
    (
        "testSortedSetFixedLengthVsStoredFields",
        test_sorted_set_fixed_length_vs_stored_fields,
    ),
    (
        "testSortedNumericsSingleValuedVsStoredFields",
        test_sorted_numerics_single_valued_vs_stored_fields,
    ),
    (
        "testSortedNumericsSingleValuedMissingVsStoredFields",
        test_sorted_numerics_single_valued_missing_vs_stored_fields,
    ),
    (
        "testSortedNumericsMultipleValuesVsStoredFields",
        test_sorted_numerics_multiple_values_vs_stored_fields,
    ),
    (
        "testSortedNumericsFewUniqueSetsVsStoredFields",
        test_sorted_numerics_few_unique_sets_vs_stored_fields,
    ),
    (
        "testSortedSetVariableLengthVsStoredFields",
        test_sorted_set_variable_length_vs_stored_fields,
    ),
    (
        "testSortedSetFixedLengthSingleValuedVsStoredFields",
        test_sorted_set_fixed_length_single_valued_vs_stored_fields,
    ),
    (
        "testSortedSetVariableLengthSingleValuedVsStoredFields",
        test_sorted_set_variable_length_single_valued_vs_stored_fields,
    ),
    (
        "testSortedSetFixedLengthFewUniqueSetsVsStoredFields",
        test_sorted_set_fixed_length_few_unique_sets_vs_stored_fields,
    ),
    (
        "testSortedSetVariableLengthFewUniqueSetsVsStoredFields",
        test_sorted_set_variable_length_few_unique_sets_vs_stored_fields,
    ),
    (
        "testSortedSetVariableLengthManyValuesPerDocVsStoredFields",
        test_sorted_set_variable_length_many_values_per_doc_vs_stored_fields,
    ),
    (
        "testSortedSetFixedLengthManyValuesPerDocVsStoredFields",
        test_sorted_set_fixed_length_many_values_per_doc_vs_stored_fields,
    ),
    ("testGCDCompression", test_gcd_compression),
    ("testSparseGCDCompression", test_sparse_gcd_compression),
    ("testZeros", test_zeros),
    ("testSparseZeros", test_sparse_zeros),
    ("testZeroOrMin", test_zero_or_min),
    ("testTwoNumbersOneMissing", test_two_numbers_one_missing),
    (
        "testTwoNumbersOneMissingWithMerging",
        test_two_numbers_one_missing_with_merging,
    ),
    (
        "testThreeNumbersOneMissingWithMerging",
        test_three_numbers_one_missing_with_merging,
    ),
    ("testTwoBytesOneMissing", test_two_bytes_one_missing),
    (
        "testTwoBytesOneMissingWithMerging",
        test_two_bytes_one_missing_with_merging,
    ),
    (
        "testThreeBytesOneMissingWithMerging",
        test_three_bytes_one_missing_with_merging,
    ),
    ("testThreads", test_threads),
    ("testOneSortedNumber", test_one_sorted_number),
    (
        "testOneSortedNumberOneMissing",
        test_one_sorted_number_one_missing,
    ),
    (
        "testNumberMergeAwayAllValues",
        test_number_merge_away_all_values,
    ),
    ("testTwoSortedNumber", test_two_sorted_number),
    (
        "testTwoSortedNumberSameValue",
        test_two_sorted_number_same_value,
    ),
    (
        "testTwoSortedNumberOneMissing",
        test_two_sorted_number_one_missing,
    ),
    ("testSortedNumberMerge", test_sorted_number_merge),
    (
        "testSortedNumberMergeAwayAllValues",
        test_sorted_number_merge_away_all_values,
    ),
    (
        "testSortedEnumAdvanceIndependently",
        test_sorted_enum_advance_independently,
    ),
    (
        "testSortedSetEnumAdvanceIndependently",
        test_sorted_set_enum_advance_independently,
    ),
    (
        "testSortedMergeAwayAllValuesLargeSegment",
        test_sorted_merge_away_all_values_large_segment,
    ),
    (
        "testSortedSetMergeAwayAllValuesLargeSegment",
        test_sorted_set_merge_away_all_values_large_segment,
    ),
    (
        "testNumericMergeAwayAllValuesLargeSegment",
        test_numeric_merge_away_all_values_large_segment,
    ),
    (
        "testSortedNumericMergeAwayAllValuesLargeSegment",
        test_sorted_numeric_merge_away_all_values_large_segment,
    ),
    (
        "testBinaryMergeAwayAllValuesLargeSegment",
        test_binary_merge_away_all_values_large_segment,
    ),
    ("testRandomAdvanceNumeric", test_random_advance_numeric),
    ("testRandomAdvanceBinary", test_random_advance_binary),
    ("testThreads2", test_threads2),
    ("testThreads3", test_threads3),
    (
        "testEmptyBinaryValueOnPageSizes",
        test_empty_binary_value_on_page_sizes,
    ),
    ("testHighOrdsSortedSetDV", test_high_ords_sorted_set_dv),
];

/// The `@Nightly` tests, at the end of [`TESTS`].
pub const NIGHTLY: &[&str] = &[
    "testThreads2",
    "testThreads3",
    "testEmptyBinaryValueOnPageSizes",
    "testHighOrdsSortedSetDV",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_java_test_is_listed_once() {
        let names: BTreeSet<&str> = TESTS.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), TESTS.len());
        // LegacyBaseDocValuesFormatTestCase's 106 public tests.
        assert_eq!(106, TESTS.len());
        for n in NIGHTLY {
            assert!(names.contains(n));
        }
    }

    #[test]
    fn check_dictionary_refuses_terms_out_of_order() {
        let ok = [b"a".to_vec(), b"b".to_vec()].into_iter().map(Ok);
        assert!(check_dictionary(ok).is_ok());
        let bad = [b"b".to_vec(), b"a".to_vec()].into_iter().map(Ok);
        assert!(check_dictionary(bad).unwrap_err().contains("out of order"));
        let failed = [Err("lookupOrd".to_string())].into_iter();
        assert_eq!(Err("lookupOrd".to_string()), check_dictionary(failed));
    }

    #[test]
    fn an_all_and_a_none_automaton_intersect_everything_and_nothing() {
        let dir = ByteBuffersDirectory::new();
        let mut w = plain_writer(&dir);
        for v in ["b", "a"] {
            w.add_fields_document(&doc_of(vec![ss("f", v)]))
                .expect("addDocument");
        }
        w.close().expect("close");
        let reader = DirectoryReader::open(&dir).expect("open");
        let leaf = only_leaf(&reader);
        let all = compiled(".*");
        assert_eq!(AutomatonType::ALL, all.automaton_type);
        assert_eq!(
            vec![(b"a".to_vec(), 0), (b"b".to_vec(), 1)],
            DvTermsEnum::sorted_set(leaf, "f").intersect(&all)
        );
        let none = compiled("#");
        assert_eq!(AutomatonType::NONE, none.automaton_type);
        assert!(DvTermsEnum::sorted_set(leaf, "f")
            .intersect(&none)
            .is_empty());
        let absent = CompiledAutomaton::new(
            &RegExp::new("zz")
                .expect("a regexp")
                .to_automaton()
                .expect("an automaton"),
        )
        .expect("compile");
        assert!(DvTermsEnum::sorted_set(leaf, "f")
            .intersect(&absent)
            .is_empty());
        // One term: nothing to advance independently.
        sorted_set_enum_advance_independently(1, || unreachable!());
        assert_eq!(2, check_leaf_doc_values(leaf).expect("testDocValues") + 1);
    }
}
