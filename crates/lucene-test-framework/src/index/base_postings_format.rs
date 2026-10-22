//! `BasePostingsFormatTestCase`: the postings suite, on the one postings
//! format this port writes (`Lucene104PostingsFormat`).
//!
//! Differs:
//!
//! - Java builds one [`RandomPostingsTester`] per class (`@BeforeClass`);
//!   here each test builds its own from the test's stream.
//! - `testInvertedWrite` is not ported: it wraps the codec's postings format
//!   in a `FilterCodec` to visit the postings twice during flush and merge,
//!   and this port's writer has no codec plug-in point.
//! - `testPostingsEnumReuse` and the `assertReused` halves of the
//!   `testPostingsEnum*` tests: this port's `TermsEnum.postings` takes no
//!   enum to reuse (`isPostingsEnumReuseImplemented() == false`), so they
//!   check that a second call returns an independent enum over the same
//!   postings.
//! - `testFields`' `Fields.iterator().remove()` check has no iterator to run
//!   on (see [`RandomPostingsTester::test_fields`]).
//! - `testMismatchedFields` reaches a mismatched field numbering by adding
//!   the fields in a different order to each index, not through
//!   `MismatchedCodecReader`, and copies with `addIndexes(Directory...)`.
//! - `testLineFileDocs` (`@Nightly`) and `testDocIDRunEnd` take their sizes
//!   as parameters ([`line_file_docs`], [`doc_id_run_end`]) so a non-nightly
//!   run can reach the same code at a smaller size.

use super::base_index_file_format::SuiteTest;
use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::IndexOptions;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    index_options_name, Document, Field, FieldToken, FieldTokens, FieldType, Store, StringField,
    TextField,
};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::IndexReader as _;
use lucene_search::reader::{LeafReader, PostingsEnum, PostingsFlags, NO_MORE_DOCS};
use lucene_store::{ByteBuffersDirectory, Directory};

use super::base_index_file_format::{
    docs_with_term, force_merge, mock_analyzer, new_config, only_leaf, plain_writer,
    riw_force_merge, riw_with, IndexFileFormatSuite,
};
use super::index_writer_config::{
    new_log_merge_policy, new_tiered_merge_policy, Draws, LiveMergePolicy, PolicyKind,
};
use super::merge_policies::shuffle;
use super::random_postings_tester::{RandomPostingsTester, TestOption, TestOptions, INDEX_OPTIONS};
use crate::util::line_file_docs::LineFileDocs;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;

/// `BasePostingsFormatTestCase` itself: its `addRandomFields`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PostingsSuite;

impl IndexFileFormatSuite for PostingsSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        for opts in &INDEX_OPTIONS[1..] {
            let mut ft = FieldType::new();
            ft.set_index_options(*opts).expect("an unfrozen FieldType");
            let ft = ft.frozen();
            let num_fields = t.random().next_int_bounded(5);
            for _ in 0..num_fields {
                let s = test_util::random_simple_string_max(t.random(), 2);
                doc.add(
                    Field::from_string(format!("f_{}", index_options_name(*opts)), s, ft.clone())
                        .expect("an indexed field"),
                );
            }
        }
    }
}

/// `postingsTester.testFull(codec, path, options, withPayloads)`: index every
/// field at exactly `options` and test at every weaker one too.
///
/// # Panics
/// When a check fails.
pub fn test_full(t: &mut LuceneTestCase, options: IndexOptions, with_payloads: bool) {
    let mut tester = RandomPostingsTester::from_test(t);
    let dir = ByteBuffersDirectory::new();
    let reader = tester.build_index(&dir, options, with_payloads, true);
    let leaf = only_leaf(&reader);
    tester.test_fields(leaf);
    let max_index_option = super::random_postings_tester::rank(options);
    for test_options in &INDEX_OPTIONS[..=max_index_option] {
        tester.test_terms(leaf, TestOptions::ALL, *test_options, options, true);
        if with_payloads {
            // If we indexed w/ payloads, also test enums w/o accessing
            // payloads:
            tester.test_terms(
                leaf,
                TestOptions::ALL.without(TestOption::Payloads),
                *test_options,
                options,
                true,
            );
        }
    }
}

/// `testDocsOnly`.
pub fn test_docs_only(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::Docs, false);
}

/// `testDocsAndFreqs`.
pub fn test_docs_and_freqs(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::DocsAndFreqs, false);
}

/// `testDocsAndFreqsAndPositions`.
pub fn test_docs_and_freqs_and_positions(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::DocsAndFreqsAndPositions, false);
}

/// `testDocsAndFreqsAndPositionsAndPayloads`.
pub fn test_docs_and_freqs_and_positions_and_payloads(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::DocsAndFreqsAndPositions, true);
}

/// `testDocsAndFreqsAndPositionsAndOffsets`.
pub fn test_docs_and_freqs_and_positions_and_offsets(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::DocsAndFreqsAndPositionsAndOffsets, false);
}

/// `testDocsAndFreqsAndPositionsAndOffsetsAndPayloads`.
pub fn test_docs_and_freqs_and_positions_and_offsets_and_payloads(t: &mut LuceneTestCase) {
    test_full(t, IndexOptions::DocsAndFreqsAndPositionsAndOffsets, true);
}

/// `testRandom`.
pub fn test_random(t: &mut LuceneTestCase) {
    let all = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let mut tester = RandomPostingsTester::from_test(t);
    for _ in 0..5 {
        let dir = ByteBuffersDirectory::new();
        let index_payloads = t.random().next_boolean();
        let reader = tester.build_index(&dir, all, index_payloads, false);
        let leaf = only_leaf(&reader);
        tester.test_fields(leaf);
        // NOTE: you can also test "weaker" index options than you indexed
        // with:
        tester.test_terms(leaf, TestOptions::ALL, all, all, false);
    }
}

/// Every document and freq of `p`, from its start.
fn drain(p: &mut dyn PostingsEnum) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    loop {
        let d = p.next_doc().expect("nextDoc");
        if d == NO_MORE_DOCS {
            return out;
        }
        out.push((d, p.freq()));
    }
}

/// `testPostingsEnumReuse`: `checkReuse(te, FREQS, ALL, false)`; reuse is
/// not implemented, so the second enum is a fresh one over the same docs.
pub fn test_postings_enum_reuse(t: &mut LuceneTestCase) {
    let mut tester = RandomPostingsTester::from_test(t);
    let dir = ByteBuffersDirectory::new();
    let payloads = t.random().next_boolean();
    let reader = tester.build_index(
        &dir,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        payloads,
        true,
    );
    shuffle(t.random(), &mut tester.all_terms);
    let fat = tester.all_terms[0].clone();
    let leaf = only_leaf(&reader);
    let terms = leaf.terms(&fat.field).expect("terms").expect("terms");
    let mut te = terms.iterator().expect("iterator");
    assert!(te.try_seek_exact(&fat.term).expect("seekExact"));
    let mut postings1 = te.postings(PostingsFlags::Freqs).expect("postings");
    let first = postings1.next_doc().expect("nextDoc");
    let mut postings2 = te.postings(PostingsFlags::All).expect("postings");
    assert_eq!(postings2.doc_id(), -1, "a second enum starts over");
    assert_eq!(
        postings1.doc_id(),
        first,
        "and leaves the first where it was"
    );
    let mut rest = vec![(first, postings1.freq())];
    rest.extend(drain(postings1.as_mut()));
    assert_eq!(rest, drain(postings2.as_mut()));
}

/// The fields of `leaf` other than a `RandomIndexWriter`'s soft-deletes
/// field, which Java's soft-deleting writer only adds on first use.
fn user_fields(leaf: &SegmentReader) -> usize {
    leaf.field_infos()
        .fields
        .iter()
        .filter(|f| !f.soft_deletes_field)
        .count()
}

fn one_string_field_doc(t: &mut LuceneTestCase, value: &str) -> (Vec<u8>, usize) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut iw = riw_with(t, &dir, config);
    let mut doc = Document::new();
    doc.add(StringField::new("", value, Store::No));
    iw.add_document(&doc).expect("addDocument");
    let ir = iw.get_reader().expect("getReader");
    let ar = only_leaf(&ir);
    let fields = user_fields(ar);
    let terms = ar.terms("").expect("terms").expect("terms");
    let mut te = terms.iterator().expect("iterator");
    let term = te.next().expect("next").expect("a term").to_vec();
    assert!(te.next().expect("next").is_none());
    drop(te);
    drop(terms);
    drop(ir);
    iw.close().expect("close");
    (term, fields)
}

/// `testJustEmptyField`.
pub fn test_just_empty_field(t: &mut LuceneTestCase) {
    let (term, fields) = one_string_field_doc(t, "something");
    assert_eq!(fields, 1);
    assert_eq!(term, b"something");
}

/// `testEmptyFieldAndEmptyTerm`.
pub fn test_empty_field_and_empty_term(t: &mut LuceneTestCase) {
    let (term, fields) = one_string_field_doc(t, "");
    assert_eq!(fields, 1);
    assert_eq!(term, b"");
}

fn two_docs_one_term(t: &mut LuceneTestCase, text: bool, flags: PostingsFlags) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let config = new_config(t, analyzer);
    let mut iw = riw_with(t, &dir, config);
    let mut doc = Document::new();
    if text {
        doc.add(TextField::new("field", "value", Store::No));
    } else {
        doc.add(StringField::new("field", "value", Store::No));
    }
    iw.add_document(&doc).expect("addDocument");
    iw.add_document(&doc).expect("addDocument");
    let ir = iw.get_reader().expect("getReader");
    let ar = only_leaf(&ir);
    let terms = ar.terms("field").expect("terms").expect("terms");
    let mut te = terms.iterator().expect("iterator");
    assert!(te.try_seek_exact(b"value").expect("seekExact"));
    let mut docs = te.postings(flags).expect("postings");
    assert_eq!(0, docs.next_doc().expect("nextDoc"));
    assert_eq!(1, docs.freq());
    assert_eq!(1, docs.next_doc().expect("nextDoc"));
    assert_eq!(1, docs.freq());
    drop(docs);
    drop(te);
    drop(terms);
    drop(ir);
    iw.close().expect("close");
}

/// `testDidntWantFreqsButAskedAnyway`.
pub fn test_didnt_want_freqs_but_asked_anyway(t: &mut LuceneTestCase) {
    two_docs_one_term(t, true, PostingsFlags::None);
}

/// `testAskForPositionsWhenNotThere`.
pub fn test_ask_for_positions_when_not_there(t: &mut LuceneTestCase) {
    two_docs_one_term(t, false, PostingsFlags::Positions);
}

/// `testGhosts`: ghost fields still work.
pub fn test_ghosts(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let analyzer = mock_analyzer(t);
    let mut config = new_config(t, analyzer);
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    config.merge_policy = LiveMergePolicy::new(PolicyKind::Log(log));
    let mut iw = riw_with(t, &dir, config);
    let mut doc = Document::new();
    iw.add_document(&doc).expect("addDocument");
    doc.add(StringField::new("ghostField", "something", Store::No));
    iw.add_document(&doc).expect("addDocument");
    riw_force_merge(&mut iw, 1);
    // delete the only term for the field
    iw.delete_documents(Term::new("ghostField", "something"))
        .expect("deleteDocuments");
    riw_force_merge(&mut iw, 1);
    let ir = iw.get_reader().expect("getReader");
    let ar = only_leaf(&ir);
    // Ghost busting terms dict impls will have fields.size() == 0; all
    // others must be == 1:
    assert!(user_fields(ar) <= 1);
    if let Some(terms) = ar.terms("ghostField").expect("terms") {
        let mut te = terms.iterator().expect("iterator");
        if te.next().expect("next").is_some() {
            let mut p = te.postings(PostingsFlags::Freqs).expect("postings");
            assert_eq!(p.next_doc().expect("nextDoc"), NO_MORE_DOCS);
        }
    }
    drop(ir);
    iw.close().expect("close");
}

fn tiered_writer<'d>(
    t: &mut LuceneTestCase,
    dir: &'d dyn Directory,
) -> lucene_index::index_writer::IndexWriter<'d> {
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let tmp = new_tiered_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    let mut iw = plain_writer(dir);
    iw.set_pluggable_merge_policy(Some(std::sync::Arc::new(LiveMergePolicy::new(
        PolicyKind::Tiered(tmp),
    ))));
    iw
}

/// `testDisorder`: seek in disorder.
pub fn test_disorder(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut iw = tiered_writer(t, &dir);
    for i in 0..10_000 {
        let mut document = Document::new();
        document.add(StringField::new("id", i.to_string(), Store::No));
        iw.add_fields_document(&document).expect("addDocument");
    }
    force_merge(&mut iw, 1);
    iw.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    let terms = only_leaf(&reader)
        .terms("id")
        .expect("terms")
        .expect("terms");
    let mut te = terms.iterator().expect("iterator");
    for _ in 0..20_000 {
        let n = t.random().next_int_bounded(10_000);
        let target = n.to_string().into_bytes();
        // seekExact.
        assert!(te.try_seek_exact(&target).expect("seekExact"));
        assert_eq!(te.term(), Some(target.as_slice()));
        // seekCeil.
        assert_eq!(
            SeekStatus::Found,
            te.try_seek_ceil(&target).expect("seekCeil")
        );
        assert_eq!(te.term(), Some(target.as_slice()));
    }
    drop(te);
    drop(terms);
    drop(reader);
    iw.close().expect("close");
}

/// `testBinarySearchTermLeaf`.
pub fn test_binary_search_term_leaf(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let mut iw = tiered_writer(t, &dir);
    for i in (100_000..=100_400).filter(|i| i % 2 == 1) {
        // only add odd number
        let mut document = Document::new();
        document.add(StringField::new("id", i.to_string(), Store::No));
        iw.add_fields_document(&document).expect("addDocument");
    }
    force_merge(&mut iw, 1);
    iw.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    let terms = only_leaf(&reader)
        .terms("id")
        .expect("terms")
        .expect("terms");
    let mut te = terms.iterator().expect("iterator");
    // test seekExact
    for i in 100_000..=100_400 {
        let target = i.to_string().into_bytes();
        if i % 2 == 1 {
            assert!(te.try_seek_exact(&target).expect("seekExact"));
            assert_eq!(te.term(), Some(target.as_slice()));
        } else {
            assert!(!te.try_seek_exact(&target).expect("seekExact"));
        }
    }
    // test seekCeil
    for i in 100_000..100_400 {
        let target = i.to_string().into_bytes();
        if i % 2 == 1 {
            assert_eq!(
                SeekStatus::Found,
                te.try_seek_ceil(&target).expect("seekCeil")
            );
            assert_eq!(te.term(), Some(target.as_slice()));
            if i <= 100_397 {
                let next = (i + 2).to_string().into_bytes();
                assert_eq!(te.next().expect("next"), Some(next.as_slice()));
            }
        } else {
            assert_eq!(
                SeekStatus::NotFound,
                te.try_seek_ceil(&target).expect("seekCeil")
            );
            let next = (i + 1).to_string().into_bytes();
            assert_eq!(te.term(), Some(next.as_slice()));
        }
    }
    assert_eq!(
        SeekStatus::End,
        te.try_seek_ceil(b"100400").expect("seekCeil")
    );
    drop(te);
    drop(terms);
    drop(reader);
    iw.close().expect("close");
}

/// `testLevel2Ghosts`: level 2 ghost fields still work.
pub fn test_level2_ghosts(t: &mut LuceneTestCase) {
    let dir = ByteBuffersDirectory::new();
    let (nightly, multiplier) = (t.nightly(), t.multiplier());
    let log = new_log_merge_policy(&mut Draws::new(t.random()).knobs(nightly, multiplier));
    let mut iw = plain_writer(&dir);
    iw.set_pluggable_merge_policy(Some(std::sync::Arc::new(LiveMergePolicy::new(
        PolicyKind::Log(log),
    ))));
    let mut document = Document::new();
    document.add(StringField::new("id", "0", Store::No));
    document.add(StringField::new("suggest_field", "apples", Store::No));
    iw.add_fields_document(&document).expect("addDocument");
    // need another document so whole segment isn't deleted
    iw.add_fields_document(&Document::new())
        .expect("addDocument");
    iw.commit().expect("commit");
    let mut document = Document::new();
    document.add(StringField::new("id", "1", Store::No));
    document.add(StringField::new("suggest_field2", "apples", Store::No));
    iw.add_fields_document(&document).expect("addDocument");
    iw.commit().expect("commit");
    iw.delete_documents_by_term(&[Term::new("id", "0")])
        .expect("deleteDocuments");
    // first force merge creates a level 1 ghost field
    force_merge(&mut iw, 1);
    // second force merge creates a level 2 ghost field
    iw.add_fields_document(&Document::new())
        .expect("addDocument");
    force_merge(&mut iw, 1);
    let reader = DirectoryReader::open(&dir).expect("open");
    assert_eq!(docs_with_term(&reader, "id", b"1").len(), 1);
    drop(reader);
    iw.close().expect("close");
}

/// What one `PostingsEnum` over `foo:bar` (two occurrences, offsets 0-3 and
/// 4-7, payloads `pay1`/`pay2` when indexed) must report under `flags`:
/// Java's `testPostingsEnum*` assertions, folded into one rule per feature.
fn check_bar(
    p: &mut dyn PostingsEnum,
    flags: PostingsFlags,
    options: IndexOptions,
    payloads: bool,
) {
    let has_freqs = options != IndexOptions::Docs;
    let has_positions =
        super::random_postings_tester::subsumes(options, IndexOptions::DocsAndFreqsAndPositions);
    let has_offsets = options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let want_offsets = matches!(flags, PostingsFlags::Offsets | PostingsFlags::All);
    let want_payloads = matches!(flags, PostingsFlags::Payloads | PostingsFlags::All);
    assert_eq!(-1, p.doc_id());
    assert_eq!(0, p.next_doc().expect("nextDoc"));
    if !has_freqs {
        assert_eq!(1, p.freq());
    } else if flags == PostingsFlags::None {
        // we don't define what it is, but if its something else, we should
        // look into it?
        assert!(p.freq() == 1 || p.freq() == 2);
    } else {
        assert_eq!(2, p.freq());
    }
    if has_positions && flags.wants_positions() {
        let expected = [(0, 3, b"pay1"), (4, 7, b"pay2")];
        for (pos, (start, end, payload)) in expected.into_iter().enumerate() {
            assert_eq!(pos as i32, p.next_position().expect("nextPosition"));
            let offsets = (p.start_offset(), p.end_offset());
            if !has_offsets {
                assert_eq!(offsets, (-1, -1));
            } else if want_offsets {
                assert_eq!(offsets, (start, end));
            } else {
                // we don't define what it is
                assert!(
                    offsets == (-1, -1) || offsets == (start, end),
                    "{offsets:?}"
                );
            }
            if !payloads {
                assert_eq!(p.payload(), None);
            } else if want_payloads {
                assert_eq!(p.payload(), Some(&payload[..]));
            } else {
                // we don't define what it is
                assert!(p.payload().is_none() || p.payload() == Some(&payload[..]));
            }
        }
    }
    assert_eq!(NO_MORE_DOCS, p.next_doc().expect("nextDoc"));
}

/// `testPostingsEnum*`: one document whose `foo` field holds `bar` twice,
/// indexed with `options` (`None` = a `StringField` holding `bar` once,
/// `testPostingsEnumDocsOnly`), with payloads when `payloads`.
fn postings_enum_case(options: Option<IndexOptions>, payloads: bool) {
    let dir = ByteBuffersDirectory::new();
    let mut iw = plain_writer(&dir);
    let mut doc = Document::new();
    let options = match options {
        None => {
            doc.add(StringField::new("foo", "bar", Store::No));
            IndexOptions::Docs
        }
        Some(options) => {
            let mut ft = FieldType::copy_of(&TextField::type_not_stored());
            ft.set_index_options(options)
                .expect("an unfrozen FieldType");
            let token = |start, end, payload: &str| FieldToken {
                payload: payloads.then(|| payload.as_bytes().to_vec()),
                ..FieldToken::new("bar", start, end)
            };
            // `new CannedTokenStream(token1, token2)`, or "bar bar" through
            // `MockTokenizer`: the same two tokens.
            let tokens = FieldTokens {
                tokens: vec![token(0, 3, "pay1"), token(4, 7, "pay2")],
                ..FieldTokens::default()
            };
            doc.add(Field::from_token_stream("foo", tokens, ft.frozen()).expect("foo"));
            options
        }
    };
    iw.add_fields_document(&doc).expect("addDocument");
    iw.commit().expect("commit");
    let reader = DirectoryReader::open(&dir).expect("open");
    let leaf = only_leaf(&reader);
    let docs_only = options == IndexOptions::Docs;
    let check = |p: &mut dyn PostingsEnum, flags| {
        if docs_only {
            // `testPostingsEnumDocsOnly`: docs and freq 1 under any flags.
            assert_eq!(-1, p.doc_id());
            assert_eq!(0, p.next_doc().expect("nextDoc"));
            assert_eq!(1, p.freq());
            assert_eq!(NO_MORE_DOCS, p.next_doc().expect("nextDoc"));
        } else {
            check_bar(p, flags, options, payloads);
        }
    };
    // sugar method (FREQS)
    let mut postings = leaf
        .postings("foo", b"bar", PostingsFlags::Freqs)
        .expect("postings")
        .expect("foo:bar");
    check(postings.as_mut(), PostingsFlags::Freqs);
    let terms = leaf.terms("foo").expect("terms").expect("terms");
    let mut te = terms.iterator().expect("iterator");
    assert!(te.try_seek_exact(b"bar").expect("seekExact"));
    // asking for any flags: ok, and twice (Java's reuse): a fresh enum each.
    for flags in [
        PostingsFlags::None,
        PostingsFlags::Freqs,
        PostingsFlags::Positions,
        PostingsFlags::Payloads,
        PostingsFlags::Offsets,
        PostingsFlags::All,
    ] {
        for _ in 0..2 {
            let mut p = te.postings(flags).expect("postings");
            check(p.as_mut(), flags);
        }
        let mut p = leaf
            .postings("foo", b"bar", flags)
            .expect("postings")
            .expect("foo:bar");
        check(p.as_mut(), flags);
    }
    drop(te);
    drop(terms);
    drop(reader);
    iw.close().expect("close");
}

/// `testPostingsEnumDocsOnly`.
pub fn test_postings_enum_docs_only(_t: &mut LuceneTestCase) {
    postings_enum_case(None, false);
}

/// `testPostingsEnumFreqs`.
pub fn test_postings_enum_freqs(_t: &mut LuceneTestCase) {
    postings_enum_case(Some(IndexOptions::DocsAndFreqs), false);
}

/// `testPostingsEnumPositions`.
pub fn test_postings_enum_positions(_t: &mut LuceneTestCase) {
    postings_enum_case(Some(IndexOptions::DocsAndFreqsAndPositions), false);
}

/// `testPostingsEnumOffsets`.
pub fn test_postings_enum_offsets(_t: &mut LuceneTestCase) {
    postings_enum_case(
        Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
        false,
    );
}

/// `testPostingsEnumPayloads`.
pub fn test_postings_enum_payloads(_t: &mut LuceneTestCase) {
    postings_enum_case(Some(IndexOptions::DocsAndFreqsAndPositions), true);
}

/// `testPostingsEnumAll`.
pub fn test_postings_enum_all(_t: &mut LuceneTestCase) {
    postings_enum_case(Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets), true);
}

/// `testLineFileDocs` at `num_docs` documents: realistic data, often
/// better at uncovering real bugs.
pub fn line_file_docs(t: &mut LuceneTestCase, num_docs: i32) {
    let dir = ByteBuffersDirectory::new();
    {
        let mut docs = LineFileDocs::new(t.random()).expect("LineFileDocs");
        let mut w = plain_writer(&dir);
        for _ in 0..num_docs {
            // Only keep the body field, and don't index term vectors on it,
            // we only care about postings
            let doc = docs.next_doc().expect("nextDoc");
            let mut body = Document::new();
            body.add(TextField::new("body", doc.body, Store::No));
            w.add_fields_document(&body).expect("addDocument");
        }
        force_merge(&mut w, 1);
        w.close().expect("close");
    }
    test_util::check_index(&dir).expect("CheckIndex");
}

/// `testLineFileDocs` (`@Nightly`: skipped otherwise).
pub fn test_line_file_docs(t: &mut LuceneTestCase) {
    if t.nightly() {
        let num_docs = t.at_least(10_000);
        line_file_docs(t, num_docs);
    }
}

/// `testMismatchedFields`.
pub fn test_mismatched_fields(_t: &mut LuceneTestCase) {
    let dir1 = ByteBuffersDirectory::new();
    let mut w1 = plain_writer(&dir1);
    let mut doc = Document::new();
    // `MismatchedCodecReader` renumbers the fields; adding them in the
    // other order gives the source the other numbering.
    doc.add(StringField::new("g", "b", Store::No));
    doc.add(StringField::new("f", "a", Store::No));
    w1.add_fields_document(&doc).expect("addDocument");
    w1.commit().expect("commit");
    w1.close().expect("close");

    let dir2 = ByteBuffersDirectory::new();
    let mut w2 = plain_writer(&dir2);
    let mut doc = Document::new();
    doc.add(StringField::new("f", "a", Store::No));
    doc.add(StringField::new("g", "b", Store::No));
    w2.add_fields_document(&doc).expect("addDocument");
    w2.commit().expect("commit");
    w2.add_indexes(&[&dir1]).expect("addIndexes");
    force_merge(&mut w2, 1);
    w2.close().expect("close");

    let reader = DirectoryReader::open(&dir2).expect("open");
    let leaf = only_leaf(&reader);
    for (field, term) in [("f", b"a"), ("g", b"b")] {
        let terms = leaf.terms(field).expect("terms").expect("terms");
        let mut te = terms.iterator().expect("iterator");
        assert_eq!(te.next().expect("next"), Some(&term[..]));
        assert_eq!(2, te.doc_freq().expect("docFreq"));
        assert!(te.next().expect("next").is_none());
    }
}

/// `testDocIDRunEnd` over `iters` indexes (Java runs 100).
pub fn doc_id_run_end(t: &mut LuceneTestCase, iters: i32) {
    for _ in 0..iters {
        // `OpenMode.CREATE`: a fresh directory per iteration.
        let dir = ByteBuffersDirectory::new();
        let mut iw = plain_writer(&dir);
        // Prevent randomization from slowing down indexing too much
        iw.set_max_buffered_docs(1_000).expect("maxBufferedDocs");
        let empty_doc = Document::new();
        let mut doc = Document::new();
        doc.add(StringField::new("", "something", Store::No));
        let num_empty_docs = test_util::next_int(t.random(), 0, 5_000);
        let num_docs = test_util::next_int(t.random(), 4096, 20_000);
        for _ in 0..num_empty_docs {
            iw.add_fields_document(&empty_doc).expect("addDocument");
        }
        for _ in 0..num_docs {
            iw.add_fields_document(&doc).expect("addDocument");
        }
        force_merge(&mut iw, 1);
        iw.close().expect("close");
        let reader = DirectoryReader::open(&dir).expect("open");
        let leaf = only_leaf(&reader);
        assert_eq!(leaf.max_doc(), num_empty_docs + num_docs);
        let mut p = leaf
            .postings("", b"something", PostingsFlags::Freqs)
            .expect("postings")
            .expect(":something");
        // `TestUtil.checkReader(ar)`: CheckIndex over the index, plus the
        // run the test is named for, walked here. Where the run lands is not
        // checked (nor by Java): a forced merge orders its segments by size,
        // not by age, so the empty documents need not come first.
        let docs = drain(p.as_mut());
        assert_eq!(docs.len(), num_docs as usize);
        let bad = docs
            .windows(2)
            .find(|w| w[0].0 >= w[1].0 || w[1].1 != 1)
            .or_else(|| docs.first().filter(|&&(_, f)| f != 1).map(|_| &docs[..1]));
        assert!(
            bad.is_none(),
            "{num_empty_docs} empty + {num_docs}: postings {bad:?}"
        );
        test_util::check_index(&dir).expect("CheckIndex");
    }
}

/// `testDocIDRunEnd`.
pub fn test_doc_id_run_end(t: &mut LuceneTestCase) {
    doc_id_run_end(t, 100);
}

/// Every test of the suite, with its name, for a runner to loop over.
pub const TESTS: &[SuiteTest] = &[
    ("testDocsOnly", test_docs_only),
    ("testDocsAndFreqs", test_docs_and_freqs),
    (
        "testDocsAndFreqsAndPositions",
        test_docs_and_freqs_and_positions,
    ),
    (
        "testDocsAndFreqsAndPositionsAndPayloads",
        test_docs_and_freqs_and_positions_and_payloads,
    ),
    (
        "testDocsAndFreqsAndPositionsAndOffsets",
        test_docs_and_freqs_and_positions_and_offsets,
    ),
    (
        "testDocsAndFreqsAndPositionsAndOffsetsAndPayloads",
        test_docs_and_freqs_and_positions_and_offsets_and_payloads,
    ),
    ("testRandom", test_random),
    ("testPostingsEnumReuse", test_postings_enum_reuse),
    ("testJustEmptyField", test_just_empty_field),
    (
        "testEmptyFieldAndEmptyTerm",
        test_empty_field_and_empty_term,
    ),
    (
        "testDidntWantFreqsButAskedAnyway",
        test_didnt_want_freqs_but_asked_anyway,
    ),
    (
        "testAskForPositionsWhenNotThere",
        test_ask_for_positions_when_not_there,
    ),
    ("testGhosts", test_ghosts),
    ("testDisorder", test_disorder),
    ("testBinarySearchTermLeaf", test_binary_search_term_leaf),
    ("testLevel2Ghosts", test_level2_ghosts),
    ("testPostingsEnumDocsOnly", test_postings_enum_docs_only),
    ("testPostingsEnumFreqs", test_postings_enum_freqs),
    ("testPostingsEnumPositions", test_postings_enum_positions),
    ("testPostingsEnumOffsets", test_postings_enum_offsets),
    ("testPostingsEnumPayloads", test_postings_enum_payloads),
    ("testPostingsEnumAll", test_postings_enum_all),
    ("testLineFileDocs", test_line_file_docs),
    ("testMismatchedFields", test_mismatched_fields),
    ("testDocIDRunEnd", test_doc_id_run_end),
];
