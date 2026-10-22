//! `RandomPostingsTester`: random postings, enumerable again from their
//! seeds, written through this port's writer and read back through every
//! `TermsEnum`/`PostingsEnum`/`ImpactsEnum` entry point a `LeafReader`
//! offers, the way `BasePostingsFormatTestCase` drives it.
//!
//! Differs:
//!
//! - **`buildIndex` writes through [`IndexWriter`](lucene_index::index_writer::IndexWriter)'s explicit documents**,
//!   not a bare `FieldsConsumer`: this port has no codec plug-in point that
//!   takes Java's `Fields` view, and its segment reader is the only
//!   `Terms` the suite can read back. The postings are transposed into one
//!   [`ExplicitDocument`] per document (each document's terms, freqs,
//!   positions, offsets and payloads, and norm `1 + (doc & 0x0f)`, which is
//!   Java's `DOC_TO_NORM`), so what reaches
//!   `postings_writer::write_fields_with_norms` is exactly the postings
//!   Java's `SeedFields` hands its consumer. A field's payload flag is the
//!   segment's own (set when one occurrence carries a non-empty payload), as
//!   explicit documents define it; the checks key off what the suite meant
//!   to index (`currentFieldInfos`), which agrees whenever a payload exists.
//! - **What has no Rust counterpart is not exercised**: `TermState`
//!   (`Option::TermState` saves nothing, so no seek goes through one),
//!   `PostingsEnum` reuse (`Option::ReuseEnums` draws but every call returns
//!   a fresh enum), `DocIdSetIterator.intoBitSet` (`Option::IntoBitSet`) and
//!   `PostingsEnum.nextPostings` (`Option::NextDocsAndFreqs`).
//! - The segment reader's `TermsEnum.impacts` is a `SlowImpactsEnum`
//!   (`(Integer.MAX_VALUE, 1)` over every document), so the impacts checks
//!   run but can only find a malformed list; the real per-block impacts are
//!   held to Java's by `CheckIndex`'s port and the codec's own fixtures.
//! - Each thread of `testTerms` shuffles its own copy of `allTerms` (Java's
//!   threads shuffle the one shared list, a race Java tolerates).
//! - The tester draws from its own stream, forked from the test's at
//!   construction, rather than holding the class's `random()`.

use std::collections::{BTreeMap, BTreeSet};

use lucene_codecs::automaton::{ByteDfa, Nfa};
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::postings::Impact;
use lucene_index::index_writer::{ExplicitDocument, InvertedField, InvertedTerm};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::reader::{
    ImpactsEnum, LeafReader, PostingsEnum, PostingsFlags, TermsEnum, NO_MORE_DOCS,
};
use lucene_search::Error as SearchError;
use lucene_store::Directory;
use lucene_util::automaton::{
    operations, AutomatonType, CompiledAutomaton, RunAutomaton, DEFAULT_DETERMINIZE_WORK_LIMIT,
};
use lucene_util::java_random::JavaRandom;

use super::base_index_file_format::{force_merge, plain_writer};
use super::merge_policies::shuffle;
use crate::util::automaton_test_util::{random_automaton, RandomAcceptedStrings};
use crate::util::lucene_test_case::{self, LuceneTestCase};
use crate::util::test_util::{next_int, random_simple_string};

/// `DOC_TO_NORM`: every field's norm in every document.
pub fn doc_to_norm(doc: i32) -> i64 {
    1 + i64::from(doc & 0x0f)
}

/// `IndexOptions.values()`, without `DOCS_AND_CUSTOM_FREQS` (never drawn:
/// the suites index at most `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`).
pub const INDEX_OPTIONS: [IndexOptions; 5] = [
    IndexOptions::None,
    IndexOptions::Docs,
    IndexOptions::DocsAndFreqs,
    IndexOptions::DocsAndFreqsAndPositions,
    IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
];

/// `Arrays.asList(IndexOptions.values()).indexOf(o)`, with
/// `DOCS_AND_CUSTOM_FREQS` ranked as `DOCS_AND_FREQS` (`subsumes`' rule).
pub fn rank(o: IndexOptions) -> usize {
    match o {
        IndexOptions::None => 0,
        IndexOptions::Docs => 1,
        IndexOptions::DocsAndFreqs | IndexOptions::DocsAndCustomFreqs => 2,
        IndexOptions::DocsAndFreqsAndPositions => 3,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => 4,
    }
}

/// `IndexOptions.subsumes(other)`.
pub fn subsumes(this: IndexOptions, other: IndexOptions) -> bool {
    rank(this) >= rank(other)
}

/// `PostingsEnum` flags as this port spells them: `POSITIONS` with or
/// without `OFFSETS` and `PAYLOADS`, else `FREQS` or `NONE`.
pub fn postings_flags(
    freqs: bool,
    positions: bool,
    offsets: bool,
    payloads: bool,
) -> PostingsFlags {
    match (positions, offsets, payloads) {
        (false, _, _) if freqs => PostingsFlags::Freqs,
        (false, _, _) => PostingsFlags::None,
        (true, false, false) => PostingsFlags::Positions,
        (true, true, false) => PostingsFlags::Offsets,
        (true, false, true) => PostingsFlags::Payloads,
        (true, true, true) => PostingsFlags::All,
    }
}

/// `Option`: which features a run tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOption {
    /// Sometimes use `advance()`.
    Skipping,
    /// Sometimes reuse the `PostingsEnum` across terms (draws only here).
    ReuseEnums,
    /// Sometimes pass non-null live docs (unused in Java too).
    LiveDocs,
    /// Sometimes seek through a saved `TermState` (no `TermState` here).
    TermState,
    /// Sometimes don't fully consume a term's documents.
    PartialDocConsume,
    /// Sometimes don't fully consume a document's positions.
    PartialPosConsume,
    /// `DocIdSetIterator#intoBitSet` (no counterpart here).
    IntoBitSet,
    /// `PostingsEnum#nextPostings` (no counterpart here).
    NextDocsAndFreqs,
    /// Sometimes check payloads.
    Payloads,
    /// Test with several threads.
    Threads,
}

/// An `EnumSet<Option>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestOptions(u16);

impl TestOptions {
    /// `EnumSet.allOf(Option.class)`.
    pub const ALL: TestOptions = TestOptions((1 << 10) - 1);

    /// Whether `o` is in the set.
    pub fn contains(self, o: TestOption) -> bool {
        self.0 & (1 << o as u16) != 0
    }

    /// The set without `o` (`EnumSet.complementOf(EnumSet.of(o))` on `ALL`).
    pub fn without(self, o: TestOption) -> TestOptions {
        TestOptions(self.0 & !(1 << o as u16))
    }
}

/// `getSeedPostings(term, seed, options, allowPayloads)`.
pub fn get_seed_postings(
    term: &[u8],
    seed: i64,
    options: IndexOptions,
    allow_payloads: bool,
    multiplier: i32,
) -> SeedPostings {
    let (min_doc_freq, max_doc_freq) = if term.starts_with(b"big_") {
        (multiplier * 50_000, multiplier * 70_000)
    } else if term.starts_with(b"medium_") {
        (multiplier * 3_000, multiplier * 6_000)
    } else if term.starts_with(b"low_") {
        (multiplier, multiplier * 40)
    } else {
        (1, 3)
    };
    SeedPostings::new(seed, min_doc_freq, max_doc_freq, options, allow_payloads)
}

/// `SeedPostings`: given the same seed, always the same postings.
#[derive(Debug, Clone)]
pub struct SeedPostings {
    /// Only for doc IDs, so pulling with or without positions yields the
    /// same documents.
    doc_random: JavaRandom,
    random: JavaRandom,
    pub doc_freq: i32,
    max_doc_spacing: i32,
    payload_size: i32,
    fixed_payloads: bool,
    payload: Vec<u8>,
    payload_len: usize,
    do_positions: bool,
    allow_payloads: bool,
    doc_id: i32,
    freq: i32,
    pub upto: i32,
    pos: i32,
    offset: i32,
    start_offset: i32,
    end_offset: i32,
    pos_spacing: i32,
    pos_upto: i32,
}

impl SeedPostings {
    /// `new SeedPostings(seed, minDocFreq, maxDocFreq, options, allowPayloads)`.
    pub fn new(
        seed: i64,
        min_doc_freq: i32,
        max_doc_freq: i32,
        options: IndexOptions,
        allow_payloads: bool,
    ) -> Self {
        let mut random = JavaRandom::new(seed);
        let doc_random = JavaRandom::new(random.next_long());
        let doc_freq = next_int(&mut random, min_doc_freq, max_doc_freq);
        let max_doc_spacing = next_int(&mut random, 1, 100);
        let payload_size = if random.next_int_bounded(10) == 7 {
            // 10% of the time create big payloads:
            1 + random.next_int_bounded(3)
        } else {
            1 + random.next_int_bounded(1)
        };
        let fixed_payloads = random.next_boolean();
        SeedPostings {
            doc_random,
            random,
            doc_freq,
            max_doc_spacing,
            payload_size,
            fixed_payloads,
            payload: vec![0; payload_size as usize],
            payload_len: 0,
            do_positions: subsumes(options, IndexOptions::DocsAndFreqsAndPositions),
            allow_payloads,
            doc_id: -1,
            freq: 0,
            upto: 0,
            pos: 0,
            offset: 0,
            start_offset: 0,
            end_offset: 0,
            pos_spacing: 0,
            pos_upto: 0,
        }
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        if self.doc_id == -1 {
            self.doc_id = 0;
        }
        // Must consume random:
        while self.pos_upto < self.freq {
            self.next_position();
        }
        if self.upto < self.doc_freq {
            if self.upto == 0 && self.doc_random.next_boolean() {
                // Sometimes index docID = 0
            } else if self.max_doc_spacing == 1 {
                self.doc_id += 1;
            } else {
                self.doc_id += next_int(&mut self.doc_random, 1, self.max_doc_spacing);
            }
            // Java's second branch tests `nextInt(10) == 17`, which never
            // holds; the draw is kept.
            self.freq = if self.random.next_int_bounded(200) == 17 {
                next_int(&mut self.random, 1, 1000)
            } else if self.random.next_int_bounded(10) == 17 {
                next_int(&mut self.random, 1, 20)
            } else {
                next_int(&mut self.random, 1, 4)
            };
            self.pos = 0;
            self.offset = 0;
            self.pos_upto = 0;
            self.pos_spacing = next_int(&mut self.random, 1, 100);
            self.upto += 1;
        } else {
            self.doc_id = NO_MORE_DOCS;
        }
        self.doc_id
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.doc_id
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        self.freq
    }

    /// `nextPosition()`: `-1` when positions are not enumerated.
    // SENTINEL: -1 = positions not enumerated (Java's own value); callers
    // compare it to the expected position, never index with it.
    pub fn next_position(&mut self) -> i32 {
        if !self.do_positions {
            self.pos_upto = self.freq;
            return -1;
        }
        assert!(self.pos_upto < self.freq);
        if self.pos_upto == 0 && self.random.next_boolean() {
            // Sometimes index pos = 0
        } else if self.pos_spacing == 1 {
            self.pos += 1;
        } else {
            self.pos += next_int(&mut self.random, 1, self.pos_spacing);
        }
        // Java's quirk, kept: a non-fixed payload is either empty or the
        // full `payloadSize` bytes.
        if self.payload_size != 0 {
            if self.fixed_payloads {
                self.payload_len = self.payload_size as usize;
                self.random.next_bytes(&mut self.payload);
            } else {
                let this_payload_size = self.random.next_int_bounded(self.payload_size);
                if this_payload_size != 0 {
                    self.payload_len = self.payload_size as usize;
                    self.random.next_bytes(&mut self.payload);
                } else {
                    self.payload_len = 0;
                }
            }
        } else {
            self.payload_len = 0;
        }
        if !self.allow_payloads {
            self.payload_len = 0;
        }
        self.start_offset = self.offset + self.random.next_int_bounded(5);
        self.end_offset = self.start_offset + self.random.next_int_bounded(10);
        self.offset = self.end_offset;
        self.pos_upto += 1;
        self.pos
    }

    /// `startOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `endOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `getPayload()`: `None` for an empty one.
    pub fn payload(&self) -> Option<&[u8]> {
        (self.payload_len != 0).then(|| &self.payload[..self.payload_len])
    }

    /// `advance(target)`: `slowAdvance`.
    pub fn advance(&mut self, target: i32) -> i32 {
        loop {
            let doc = self.next_doc();
            if doc >= target {
                return doc;
            }
        }
    }
}

/// `FieldAndTerm`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAndTerm {
    pub field: String,
    pub term: Vec<u8>,
    pub ord: i64,
}

#[derive(Debug, Clone, Copy)]
struct SeedAndOrd {
    seed: i64,
    ord: i64,
}

/// What `buildIndex` indexed one field with: `currentFieldInfos`.
#[derive(Debug, Clone, Copy)]
struct Indexed {
    options: IndexOptions,
    payloads: bool,
}

/// `RandomPostingsTester`.
pub struct RandomPostingsTester {
    /// Holds all postings, by field then term.
    fields: BTreeMap<String, BTreeMap<Vec<u8>, SeedAndOrd>>,
    /// The fields in creation order: Java's `FieldInfo` numbers.
    field_order: Vec<String>,
    pub all_terms: Vec<FieldAndTerm>,
    max_doc: i32,
    nightly: bool,
    multiplier: i32,
    pub random: JavaRandom,
    current: BTreeMap<String, Indexed>,
}

impl RandomPostingsTester {
    /// `new RandomPostingsTester(random())`, on a stream forked from the
    /// test's.
    pub fn from_test(t: &mut LuceneTestCase) -> Self {
        let seed = t.random().next_long();
        Self::new(JavaRandom::new(seed), t.nightly(), t.multiplier())
    }

    /// `new RandomPostingsTester(random)`.
    pub fn new(mut random: JavaRandom, nightly: bool, multiplier: i32) -> Self {
        let r = &mut random;
        let mut fields: BTreeMap<String, BTreeMap<Vec<u8>, SeedAndOrd>> = BTreeMap::new();
        let mut field_order = Vec::new();
        let num_fields = next_int(r, 1, 5);
        let mut max_doc = 0;
        let mut field_upto = 0;
        while field_upto < num_fields {
            let field = random_simple_string(r);
            if fields.contains_key(&field) {
                continue;
            }
            field_upto += 1;
            let mut postings = BTreeMap::new();
            let mut seen_terms = BTreeSet::new();
            let num_terms = if r.next_int_bounded(10) == 7 {
                lucene_test_case::at_least(r, multiplier, 50)
            } else {
                next_int(r, 2, 20)
            };
            while (postings.len() as i32) < num_terms {
                let term_upto = postings.len();
                // Cannot contain surrogates else default Java string sort
                // order (by UTF16 code unit) is different from Lucene:
                let term = random_simple_string(r);
                if !seen_terms.insert(term.clone()) {
                    continue;
                }
                let term = if nightly && term_upto == 0 && field_upto == 1 {
                    // Make 1 big term:
                    format!("big_{term}")
                } else if term_upto == 1 && field_upto == 1 {
                    // Make 1 medium term:
                    format!("medium_{term}")
                } else if r.next_boolean() {
                    // Low freq term:
                    format!("low_{term}")
                } else {
                    // Very low freq term (don't multiply by RANDOM_MULTIPLIER):
                    format!("verylow_{term}")
                };
                let term_seed = r.next_long();
                postings.insert(
                    term.clone().into_bytes(),
                    SeedAndOrd {
                        seed: term_seed,
                        ord: 0,
                    },
                );
                // NOTE: sort of silly: we enum all the docs just to get the
                // maxDoc
                let mut p = get_seed_postings(
                    term.as_bytes(),
                    term_seed,
                    IndexOptions::Docs,
                    true,
                    multiplier,
                );
                let mut last_doc = 0;
                loop {
                    let doc = p.next_doc();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    last_doc = doc;
                }
                max_doc = max_doc.max(last_doc);
            }
            // assign ords
            for (ord, ent) in postings.values_mut().enumerate() {
                ent.ord = ord as i64;
            }
            field_order.push(field.clone());
            fields.insert(field, postings);
        }
        // It's the count, not the last docID:
        max_doc += 1;
        let mut all_terms = Vec::new();
        for (field, terms) in &fields {
            for (ord, term) in terms.keys().enumerate() {
                all_terms.push(FieldAndTerm {
                    field: field.clone(),
                    term: term.clone(),
                    ord: ord as i64,
                });
            }
        }
        RandomPostingsTester {
            fields,
            field_order,
            all_terms,
            max_doc,
            nightly,
            multiplier,
            random,
            current: BTreeMap::new(),
        }
    }

    /// `buildIndex(codec, dir, maxAllowed, allowPayloads, alwaysTestMax)`:
    /// one segment in `dir` holding every field and term, each field at a
    /// random `IndexOptions` up to `max_allowed` (exactly it when
    /// `always_test_max`).
    ///
    /// # Panics
    /// When the writer or the reader fails.
    pub fn build_index(
        &mut self,
        dir: &dyn Directory,
        max_allowed: IndexOptions,
        allow_payloads: bool,
        always_test_max: bool,
    ) -> DirectoryReader {
        let max_index_option = rank(max_allowed);
        self.current.clear();
        for name in &self.field_order {
            // Randomly picked the IndexOptions to index this field with:
            let options = if always_test_max {
                max_allowed
            } else {
                INDEX_OPTIONS[next_int(&mut self.random, 1, max_index_option as i32) as usize]
            };
            let payloads =
                subsumes(options, IndexOptions::DocsAndFreqsAndPositions) && allow_payloads;
            self.current
                .insert(name.clone(), Indexed { options, payloads });
        }

        let mut w = plain_writer(dir);
        // One flushed segment: Java writes the whole of it in one call.
        w.set_ram_buffer_size_mb(1024.0).expect("ramBufferSizeMB");
        w.enable_explicit_documents().expect("explicit documents");
        let mut numbers = BTreeMap::new();
        for name in &self.field_order {
            let info =
                FieldInfo::new(name.clone(), 0).with_index_options(self.current[name].options);
            numbers.insert(
                name.clone(),
                w.register_field(info).expect("register_field"),
            );
        }

        let mut docs = vec![ExplicitDocument::default(); self.max_doc as usize];
        for (field, terms) in &self.fields {
            let indexed = self.current[field];
            let number = numbers[field];
            let positions = subsumes(indexed.options, IndexOptions::DocsAndFreqsAndPositions);
            let offsets = subsumes(
                indexed.options,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            );
            for (term, seed) in terms {
                // `SeedTermsEnum.postings`: always enumerated at `maxAllowed`.
                let mut p = get_seed_postings(
                    term,
                    seed.seed,
                    max_allowed,
                    allow_payloads,
                    self.multiplier,
                );
                loop {
                    let doc = p.next_doc();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    let freq = p.freq();
                    let mut t = InvertedTerm {
                        term: term.clone(),
                        freq,
                        positions: Vec::new(),
                        offsets: Vec::new(),
                        payloads: Vec::new(),
                    };
                    if positions {
                        for _ in 0..freq {
                            t.positions.push(p.next_position());
                            if offsets {
                                t.offsets.push((p.start_offset(), p.end_offset()));
                            }
                            if indexed.payloads {
                                t.payloads
                                    .push(p.payload().map(<[u8]>::to_vec).unwrap_or_default());
                            }
                        }
                        if t.payloads.iter().all(Vec::is_empty) {
                            t.payloads.clear();
                        }
                    }
                    let inverted = &mut docs[doc as usize].fields.inverted;
                    if inverted.last().is_none_or(|f| f.field_number != number) {
                        inverted.push(InvertedField {
                            field_number: number,
                            terms: Vec::new(),
                            norm: Some(doc_to_norm(doc)),
                        });
                    }
                    inverted.last_mut().expect("pushed above").terms.push(t);
                }
            }
        }
        for doc in docs {
            w.add_explicit_documents(vec![doc]).expect("addDocument");
        }
        force_merge(&mut w, 1);
        w.close().expect("close");
        let reader = DirectoryReader::open(dir).expect("open the index");
        assert_eq!(reader.segment_readers().len(), 1, "one segment");
        assert_eq!(reader.max_doc(), self.max_doc);
        reader
    }

    /// `testFields(fields)`: Java checks its `Fields` iterator refuses
    /// `remove()`; here, that the segment holds exactly these fields, each
    /// with terms.
    ///
    /// # Panics
    /// When a field is missing or extra.
    pub fn test_fields(&self, leaf: &SegmentReader) {
        let mut names: Vec<&str> = leaf
            .field_infos()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        names.sort_unstable();
        let expected: Vec<&str> = self.fields.keys().map(String::as_str).collect();
        assert_eq!(names, expected);
        for name in expected {
            assert!(
                leaf.terms(name).expect("terms").is_some(),
                "field {name:?} has no terms"
            );
        }
    }

    /// `testTerms(fieldsSource, options, maxTestOptions, maxIndexOptions,
    /// alwaysTestMax)`.
    ///
    /// # Panics
    /// When a check fails.
    pub fn test_terms(
        &mut self,
        leaf: &SegmentReader,
        options: TestOptions,
        max_test_options: IndexOptions,
        max_index_options: IndexOptions,
        always_test_max: bool,
    ) {
        if options.contains(TestOption::Threads) {
            let num_threads = if self.nightly {
                next_int(&mut self.random, 2, 5)
            } else {
                2
            };
            let seeds: Vec<i64> = (0..num_threads).map(|_| self.random.next_long()).collect();
            let this = &*self;
            std::thread::scope(|s| {
                for seed in seeds {
                    s.spawn(move || {
                        this.test_terms_one_thread(
                            &mut JavaRandom::new(seed),
                            leaf,
                            options,
                            max_test_options,
                            max_index_options,
                            always_test_max,
                        );
                    });
                }
            });
        } else {
            let mut r = std::mem::replace(&mut self.random, JavaRandom::new(0));
            self.test_terms_one_thread(
                &mut r,
                leaf,
                options,
                max_test_options,
                max_index_options,
                always_test_max,
            );
            self.random = r;
        }
    }

    fn test_terms_one_thread(
        &self,
        r: &mut JavaRandom,
        leaf: &SegmentReader,
        options: TestOptions,
        max_test_options: IndexOptions,
        max_index_options: IndexOptions,
        always_test_max: bool,
    ) {
        let check = Check {
            max_test_options,
            max_index_options,
            options,
            always_test_max,
        };
        let mut supports_ords = true;
        let mut all_terms = self.all_terms.clone();
        shuffle(r, &mut all_terms);
        for fat in &all_terms {
            // No `TermState` in this port: every seek is by term or by ord.
            let use_term_ord = supports_ords && r.next_int_bounded(5) == 1;
            let terms = leaf
                .terms(&fat.field)
                .expect("terms")
                .expect("the field has terms");
            let mut te = terms.iterator().expect("iterator");
            if use_term_ord {
                // Try seek by ord sometimes:
                match te.seek_exact_ord(fat.ord) {
                    Ok(()) => {}
                    Err(SearchError::Unsupported(_)) => {
                        supports_ords = false;
                        assert!(te.try_seek_exact(&fat.term).expect("seekExact"));
                    }
                    Err(e) => panic!("seekExact(ord): {e}"),
                }
            } else {
                assert!(te.try_seek_exact(&fat.term).expect("seekExact"));
            }
            // check we really seeked to the right place
            assert_eq!(te.term(), Some(fat.term.as_slice()));
            let term_ord = if supports_ords {
                match te.ord() {
                    Ok(ord) => ord,
                    Err(SearchError::Unsupported(_)) => {
                        supports_ords = false;
                        -1
                    }
                    Err(e) => panic!("ord: {e}"),
                }
            } else {
                -1
            };
            if term_ord != -1 {
                // PostingsFormat supports ords
                assert_eq!(fat.ord, term_ord);
            }
            self.verify_enum(r, &fat.field, &fat.term, te.as_mut(), &check);
            // 10% of the time make sure you can pull another enum from the
            // same term:
            if always_test_max || r.next_int_bounded(10) == 7 {
                self.verify_enum(r, &fat.field, &fat.term, te.as_mut(), &check);
            }
        }

        // Test Terms.intersect:
        for field in self.fields.keys() {
            loop {
                let a = random_automaton(r);
                let a = operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT)
                    .expect("determinize a random automaton");
                let ca = CompiledAutomaton::with_options(&a, false, true, false)
                    .expect("compile a random automaton");
                if ca.automaton_type != AutomatonType::NORMAL {
                    // Keep retrying until we get an A that will really "use"
                    // the PF's intersect code:
                    continue;
                }
                let mut start_term: Option<Vec<u8>> = None;
                if r.next_boolean() {
                    let ras = RandomAcceptedStrings::new(&a).expect("RandomAcceptedStrings");
                    for _ in 0..100 {
                        let code_points = ras
                            .get_random_accepted_string(r)
                            .expect("an accepted string");
                        if code_points.is_empty() {
                            continue;
                        }
                        start_term = Some(utf8_of(&code_points));
                        break;
                    }
                    // Don't allow empty string startTerm:
                    if start_term.is_none() {
                        continue;
                    }
                }
                let run = ca.run_automaton.as_ref().map(|b| b.run_automaton());
                let dfa = run.map_or_else(empty_dfa, byte_dfa);
                let terms = leaf
                    .terms(field)
                    .expect("terms")
                    .expect("the field has terms");
                let mut intersected = terms
                    .intersect(&dfa, start_term.as_deref())
                    .expect("intersect");
                let mut intersected_terms = BTreeSet::new();
                while let Some(term) = intersected.next().expect("next").map(<[u8]>::to_vec) {
                    if let Some(start) = &start_term {
                        // NOTE: not <=
                        assert!(start.as_slice() < term.as_slice());
                    }
                    self.verify_enum(r, field, &term, intersected.as_mut(), &check);
                    intersected_terms.insert(term);
                }
                match run {
                    None => assert!(intersected_terms.is_empty()),
                    Some(run) => {
                        for term2 in self.fields[field].keys() {
                            let expected = if start_term
                                .as_ref()
                                .is_some_and(|s| s.as_slice() >= term2.as_slice())
                            {
                                false
                            } else {
                                run_bytes(run, term2)
                            };
                            assert_eq!(
                                expected,
                                intersected_terms.contains(term2),
                                "term={:?}",
                                String::from_utf8_lossy(term2)
                            );
                        }
                    }
                }
                break;
            }
        }
    }

    fn verify_enum(
        &self,
        r: &mut JavaRandom,
        field: &str,
        term: &[u8],
        te: &mut dyn TermsEnum,
        c: &Check,
    ) {
        let always = c.always_test_max;
        // Make sure TermsEnum really is positioned on the expected term:
        assert_eq!(te.term(), Some(term));
        let indexed = self.current[field];
        let field_options = indexed.options;
        let mut expected = get_seed_postings(
            term,
            self.fields[field][term].seed,
            c.max_index_options,
            true,
            self.multiplier,
        );
        assert_eq!(expected.doc_freq, te.doc_freq().expect("docFreq"));

        let allow_freqs = subsumes(field_options, IndexOptions::DocsAndFreqs)
            && subsumes(c.max_test_options, IndexOptions::DocsAndFreqs);
        let do_check_freqs = allow_freqs && (always || r.next_int_bounded(3) <= 2);
        let allow_positions = subsumes(field_options, IndexOptions::DocsAndFreqsAndPositions)
            && subsumes(c.max_test_options, IndexOptions::DocsAndFreqsAndPositions);
        let do_check_positions = allow_positions && (always || r.next_int_bounded(3) <= 2);
        let allow_offsets = subsumes(
            field_options,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ) && subsumes(
            c.max_test_options,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        );
        let do_check_offsets = allow_offsets && (always || r.next_int_bounded(3) <= 2);
        let do_check_payloads = c.options.contains(TestOption::Payloads)
            && allow_positions
            && indexed.payloads
            && (always || r.next_int_bounded(3) <= 2);

        let reuse = |r: &mut JavaRandom| {
            // `options.contains(REUSE_ENUMS) && random.nextInt(10) < 9`:
            // drawn, but this port never reuses an enum.
            if c.options.contains(TestOption::ReuseEnums) {
                let _ = r.next_int_bounded(10);
            }
        };
        let mut postings: Box<dyn PostingsEnum> = if !do_check_positions {
            if allow_positions && r.next_int_bounded(10) == 7 {
                // 10% of the time, even though we will not check positions,
                // pull a DocsAndPositions enum
                reuse(r);
                let offsets = always || r.next_boolean();
                let payloads = always || r.next_boolean();
                te.postings(postings_flags(true, true, offsets, payloads))
            } else {
                reuse(r);
                te.postings(postings_flags(do_check_freqs, false, false, false))
            }
        } else {
            reuse(r);
            let offsets = always || do_check_offsets || r.next_int_bounded(3) == 1;
            let payloads = always || do_check_payloads || r.next_int_bounded(3) == 1;
            te.postings(postings_flags(true, true, offsets, payloads))
        }
        .expect("postings");
        assert_eq!(postings.doc_id(), -1, "initial docID should be -1");

        // 10% of the time don't consume all docs:
        let stop_at = if !always
            && c.options.contains(TestOption::PartialDocConsume)
            && expected.doc_freq > 1
            && r.next_int_bounded(10) == 7
        {
            r.next_int_bounded(expected.doc_freq - 1)
        } else {
            expected.doc_freq
        };
        let skip_chance = if always { 0.5 } else { r.next_double() };
        let num_skips = if expected.doc_freq < 3 {
            1
        } else {
            next_int(r, 1, 20.min(expected.doc_freq / 3))
        };
        let skip_inc = expected.doc_freq / num_skips;
        let skip_doc_inc = self.max_doc / num_skips;
        // Sometimes do 100% skipping:
        let do_all_skipping =
            c.options.contains(TestOption::Skipping) && r.next_int_bounded(7) == 1;
        let freq_ask_chance = if always { 1.0 } else { r.next_double() };
        let payload_check_chance = if always { 1.0 } else { r.next_double() };
        let offset_check_chance = if always { 1.0 } else { r.next_double() };

        while expected.upto <= stop_at {
            if expected.upto == stop_at {
                if stop_at == expected.doc_freq {
                    assert_eq!(
                        NO_MORE_DOCS,
                        postings.next_doc().expect("nextDoc"),
                        "DocsEnum should have ended but didn't"
                    );
                    // Common bug is to forget to set this.doc=NO_MORE_DOCS in
                    // the enum!:
                    assert_eq!(
                        NO_MORE_DOCS,
                        postings.doc_id(),
                        "DocsEnum should have ended but didn't"
                    );
                }
                break;
            }
            if c.options.contains(TestOption::Skipping)
                && (do_all_skipping || r.next_double() <= skip_chance)
            {
                let mut target_doc_id = -1;
                if expected.upto < stop_at && r.next_boolean() {
                    // Pick target we know exists:
                    let skip_count = next_int(r, 1, skip_inc);
                    for _ in 0..skip_count {
                        if expected.next_doc() == NO_MORE_DOCS {
                            break;
                        }
                    }
                } else {
                    // Pick random target (might not exist):
                    let skip_doc_ids = next_int(r, 1, skip_doc_inc);
                    if skip_doc_ids > 0 {
                        target_doc_id = expected.doc_id() + skip_doc_ids;
                        expected.advance(target_doc_id);
                    }
                }
                if expected.upto >= stop_at {
                    let target = if r.next_boolean() {
                        self.max_doc
                    } else {
                        NO_MORE_DOCS
                    };
                    assert_eq!(
                        NO_MORE_DOCS,
                        postings.advance(target).expect("advance"),
                        "DocsEnum should have ended but didn't"
                    );
                    break;
                }
                let target = if target_doc_id != -1 {
                    target_doc_id
                } else {
                    expected.doc_id()
                };
                let doc = postings.advance(target).expect("advance");
                assert_eq!(expected.doc_id(), doc, "docID is wrong");
            } else {
                expected.next_doc();
                let doc = postings.next_doc().expect("nextDoc");
                assert_eq!(expected.doc_id(), doc, "docID is wrong");
                if doc == NO_MORE_DOCS {
                    break;
                }
            }

            if do_check_freqs && r.next_double() <= freq_ask_chance {
                assert_eq!(expected.freq(), postings.freq(), "freq is wrong");
            }

            if do_check_positions {
                let freq = postings.freq();
                let num_pos_to_consume = if !always
                    && c.options.contains(TestOption::PartialPosConsume)
                    && r.next_int_bounded(5) == 1
                {
                    r.next_int_bounded(freq)
                } else {
                    freq
                };
                for _ in 0..num_pos_to_consume {
                    let pos = expected.next_position();
                    assert_eq!(
                        pos,
                        postings.next_position().expect("nextPosition"),
                        "position is wrong"
                    );
                    if do_check_payloads {
                        let expected_payload = expected.payload();
                        if r.next_double() <= payload_check_chance {
                            match expected_payload {
                                None => {
                                    assert!(postings.payload().is_none(), "should not have payload")
                                }
                                Some(e) => {
                                    let payload = postings.payload().map(<[u8]>::to_vec);
                                    assert_eq!(
                                        payload.as_deref(),
                                        Some(e),
                                        "payload bytes are wrong"
                                    );
                                    assert_eq!(
                                        payload.as_deref(),
                                        postings.payload(),
                                        "2nd call to getPayload returns something different!"
                                    );
                                }
                            }
                        }
                    }
                    if do_check_offsets {
                        if r.next_double() <= offset_check_chance {
                            assert_eq!(
                                expected.start_offset(),
                                postings.start_offset(),
                                "startOffset is wrong"
                            );
                            assert_eq!(
                                expected.end_offset(),
                                postings.end_offset(),
                                "endOffset is wrong"
                            );
                        }
                    } else if !subsumes(
                        field_options,
                        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                    ) {
                        assert_eq!(-1, postings.start_offset(), "startOffset isn't -1");
                        assert_eq!(-1, postings.end_offset(), "endOffset isn't -1");
                    }
                }
            }
        }

        if c.options.contains(TestOption::Skipping) {
            let flags = postings_flags(
                true,
                do_check_positions,
                do_check_positions && do_check_offsets,
                do_check_positions && do_check_payloads,
            );
            check_impacts_enum(r, te, flags, do_check_offsets, do_check_payloads);
        }
    }
}

/// `verifyEnum`'s `SKIPPING` tail: the impacts enum walks the same
/// documents as the postings, and its impacts bound every one.
fn check_impacts_enum(
    r: &mut JavaRandom,
    te: &mut dyn TermsEnum,
    flags: PostingsFlags,
    do_check_offsets: bool,
    do_check_payloads: bool,
) {
    // Every field here has norms.
    let same_positions = |postings: &mut dyn PostingsEnum, impacts: &mut dyn ImpactsEnum| {
        let freq = postings.freq();
        assert_eq!(freq, impacts.freq(), "freq is wrong");
        for _ in 0..freq {
            let pos = postings.next_position().expect("nextPosition");
            assert_eq!(
                pos,
                impacts.next_position().expect("nextPosition"),
                "position is wrong"
            );
            if do_check_offsets {
                assert_eq!(
                    postings.start_offset(),
                    impacts.start_offset(),
                    "startOffset is wrong"
                );
                assert_eq!(
                    postings.end_offset(),
                    impacts.end_offset(),
                    "endOffset is wrong"
                );
            }
            if do_check_payloads {
                assert_eq!(postings.payload(), impacts.payload(), "payload is wrong");
            }
        }
    };

    // First check impacts and block uptos
    let max = -1;
    let mut impacts_copy: Vec<Impact> = Vec::new();
    let mut impacts = te.impacts(flags).expect("impacts");
    let mut postings = te.postings(flags).expect("postings");
    loop {
        let doc = impacts.next_doc().expect("nextDoc");
        assert_eq!(postings.next_doc().expect("nextDoc"), doc);
        if doc == NO_MORE_DOCS {
            break;
        }
        same_positions(postings.as_mut(), impacts.as_mut());
        // Java never raises `max` here, so every document re-reads its block.
        if doc > max {
            impacts.advance_shallow(doc).expect("advanceShallow");
            let levels = impacts.impacts();
            check_impacts(&levels, doc).unwrap_or_else(|e| panic!("{e}"));
            impacts_copy = levels[0].1.clone();
        }
        assert_competitive(&impacts_copy, impacts.freq(), doc_to_norm(doc));
    }

    // Now check advancing. Java's `int` arithmetic is kept: from `docID() ==
    // -1`, `NO_MORE_DOCS - doc` overflows to `Integer.MIN_VALUE`, so a first
    // `advance` targets `NO_MORE_DOCS`.
    let mut impacts = te.impacts(flags).expect("impacts");
    let mut postings = te.postings(flags).expect("postings");
    let mut max = -1i32;
    loop {
        let doc = impacts.doc_id();
        let (advance, target) = if r.next_boolean() {
            (false, doc.wrapping_add(1))
        } else {
            let delta = (1 + r.next_int_bounded(512)).min(NO_MORE_DOCS.wrapping_sub(doc));
            (true, impacts.doc_id().wrapping_add(delta))
        };
        if target > max && r.next_boolean() {
            let delta = r
                .next_int_bounded(512)
                .min(NO_MORE_DOCS.wrapping_sub(target));
            max = target.wrapping_add(delta);
            impacts.advance_shallow(target).expect("advanceShallow");
            let levels = impacts.impacts();
            check_impacts(&levels, target).unwrap_or_else(|e| panic!("{e}"));
            impacts_copy = level_covering(&levels, max);
        }
        let doc = if advance {
            impacts.advance(target).expect("advance")
        } else {
            impacts.next_doc().expect("nextDoc")
        };
        assert_eq!(postings.advance(target).expect("advance"), doc);
        if doc == NO_MORE_DOCS {
            break;
        }
        same_positions(postings.as_mut(), impacts.as_mut());
        if doc > max {
            let delta = (1 + r.next_int_bounded(512)).min(NO_MORE_DOCS.wrapping_sub(doc));
            max = doc.wrapping_add(delta);
            let levels = impacts.impacts();
            check_impacts(&levels, doc).unwrap_or_else(|e| panic!("{e}"));
            impacts_copy = level_covering(&levels, max);
        }
        assert_competitive(&impacts_copy, impacts.freq(), doc_to_norm(doc));
    }
}

/// The first level whose `docIdUpTo` reaches `max`, else
/// `[(Integer.MAX_VALUE, 1)]`.
fn level_covering(levels: &[(i32, Vec<Impact>)], max: i32) -> Vec<Impact> {
    levels.iter().find(|(up_to, _)| *up_to >= max).map_or_else(
        || {
            vec![Impact {
                freq: i32::MAX,
                norm: 1,
            }]
        },
        |(_, impacts)| impacts.clone(),
    )
}

/// `Collections.binarySearch` by freq, then: some impact at least as
/// competitive as `(freq, norm)`.
fn assert_competitive(impacts: &[Impact], freq: i32, norm: i64) {
    let idx = impacts.partition_point(|i| i.freq < freq);
    assert!(
        idx < impacts.len() && impacts[idx].norm <= norm,
        "Got ({freq},{norm}) in postings, but no impact triggers equal or better scores in {impacts:?}"
    );
}

/// `CheckIndex.checkImpacts(impacts, lastTarget)`, which this suite reaches
/// through `IndexPackageAccess`. Ported as written, including that the
/// within-level ordering loop compares every impact with the first.
///
/// # Errors
/// What `CheckIndex` throws, as its message.
pub fn check_impacts(levels: &[(i32, Vec<Impact>)], last_target: i32) -> Result<(), String> {
    let Some((doc_id_up_to0, _)) = levels.first() else {
        return Err("The number of impact levels must be >= 1, got 0".to_string());
    };
    if *doc_id_up_to0 < last_target {
        return Err(format!(
            "getDocIdUpTo returned {doc_id_up_to0} on level 0, which is less than the target {last_target}"
        ));
    }
    for pair in levels.windows(2) {
        if pair[1].0 < pair[0].0 {
            return Err(format!(
                "Decreasing return for getDocIdUpTo: {} then {} for target {last_target}",
                pair[0].0, pair[1].0
            ));
        }
    }
    for (level, (_, impacts)) in levels.iter().enumerate() {
        let Some(first) = impacts.first() else {
            return Err(format!("Got empty list of impacts on level {level}"));
        };
        if first.freq < 1 {
            return Err(format!("First impact had a freq <= 0: {}", first.freq));
        }
        if first.norm == 0 {
            return Err(format!("First impact had a norm == 0: {}", first.norm));
        }
        let (prev_freq, prev_norm) = (first.freq, first.norm as u64);
        for impact in &impacts[1..] {
            if impact.freq <= prev_freq || (impact.norm as u64) <= prev_norm {
                return Err(format!(
                    "Impacts are not ordered or contain dups, got ({prev_freq},{prev_norm}) then ({},{})",
                    impact.freq, impact.norm
                ));
            }
        }
        if level > 0 {
            // Every impact of the level below must be matched by one of this
            // level's that scores at least as well.
            let mut index = 1usize;
            let (mut freq, mut norm) = (first.freq, first.norm as u64);
            for p in &levels[level - 1].1 {
                if p.freq <= freq && (p.norm as u64) >= norm {
                    continue;
                }
                let Some(next) = impacts.get(index) else {
                    return Err(format!(
                        "Found impact ({},{}) on level {} but no impact on level {level} triggers a better score",
                        p.freq,
                        p.norm,
                        level - 1
                    ));
                };
                (freq, norm) = (next.freq, next.norm as u64);
                index += 1;
            }
        }
    }
    Ok(())
}

struct Check {
    max_test_options: IndexOptions,
    max_index_options: IndexOptions,
    options: TestOptions,
    always_test_max: bool,
}

/// `new BytesRef(UnicodeUtil.newString(codePoints, ...))`: a lone surrogate
/// becomes U+FFFD, as `UnicodeUtil.UTF16toUTF8` writes it.
pub fn utf8_of(code_points: &[i32]) -> Vec<u8> {
    code_points
        .iter()
        .map(|&c| char::from_u32(c as u32).unwrap_or('\u{FFFD}'))
        .collect::<String>()
        .into_bytes()
}

/// `ca.runAutomaton.run(bytes, offset, length)`.
pub fn run_bytes(run: &RunAutomaton, bytes: &[u8]) -> bool {
    let mut state = 0;
    for &b in bytes {
        state = run.step(state, i32::from(b));
        if state == -1 {
            return false;
        }
    }
    run.is_accept(state)
}

/// The byte DFA `Terms.intersect` takes, holding exactly the language of a
/// compiled automaton's byte run automaton.
///
/// # Panics
/// When the automaton is too large for this port's DFA.
pub fn byte_dfa(run: &RunAutomaton) -> ByteDfa {
    let mut nfa = Nfa::new();
    let size = run.get_size();
    let states: Vec<u32> = (0..size)
        .map(|_| nfa.state().expect("an NFA state"))
        .collect();
    let accept = nfa.state().expect("an NFA state");
    for s in 0..size {
        let from = states[s as usize];
        if run.is_accept(s) {
            nfa.epsilon(from, accept);
        }
        let mut lo = 0i32;
        while lo < 256 {
            let to = run.step(s, lo);
            let mut hi = lo;
            while hi < 255 && run.step(s, hi + 1) == to {
                hi += 1;
            }
            if to != -1 {
                nfa.range(from, lo as u8, hi as u8, states[to as usize]);
            }
            lo = hi + 1;
        }
    }
    nfa.determinize(states[0], accept)
        .expect("determinize the byte automaton")
}

/// A DFA accepting nothing.
fn empty_dfa() -> ByteDfa {
    let mut nfa = Nfa::new();
    let start = nfa.state().expect("an NFA state");
    let accept = nfa.state().expect("an NFA state");
    nfa.determinize(start, accept).expect("determinize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_postings_replay_from_their_seed() {
        let all = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let mut a = get_seed_postings(b"low_x", 42, all, true, 1);
        let mut b = get_seed_postings(b"low_x", 42, IndexOptions::Docs, true, 1);
        loop {
            let (da, db) = (a.next_doc(), b.next_doc());
            assert_eq!(da, db, "docs do not depend on the options");
            if da == NO_MORE_DOCS {
                break;
            }
            assert_eq!(b.next_position(), -1);
        }
        assert_eq!(a.upto, a.doc_freq);
    }

    /// `testTerms` without `Option.THREADS` walks the terms on this
    /// thread, every check the threaded walk makes.
    #[test]
    fn terms_are_tested_on_one_thread_without_the_threads_option() {
        let mut tester = RandomPostingsTester::new(JavaRandom::new(17), false, 1);
        let dir = lucene_store::ByteBuffersDirectory::new();
        let all = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let reader = tester.build_index(&dir, all, true, true);
        let leaf = super::super::base_index_file_format::only_leaf(&reader);
        tester.test_fields(leaf);
        let options = TestOptions::ALL.without(TestOption::Threads);
        tester.test_terms(leaf, options, all, all, true);
    }

    /// Past every level's `docIdUpTo`, the impacts are Java's
    /// `(Integer.MAX_VALUE, 1)`.
    #[test]
    fn no_level_covers_a_target_past_the_last() {
        let one = vec![Impact { freq: 3, norm: 2 }];
        assert_eq!(level_covering(&[(5, one.clone())], 4), one);
        assert_eq!(
            level_covering(&[(5, one)], 6),
            [Impact {
                freq: i32::MAX,
                norm: 1
            }]
        );
    }

    #[test]
    fn options_set_operations() {
        let o = TestOptions::ALL.without(TestOption::Payloads);
        assert!(!o.contains(TestOption::Payloads));
        assert!(o.contains(TestOption::Threads));
        assert!(TestOptions::ALL.contains(TestOption::Skipping));
    }

    #[test]
    fn flags_follow_javas_bits() {
        assert_eq!(
            postings_flags(false, false, false, false),
            PostingsFlags::None
        );
        assert_eq!(
            postings_flags(true, false, true, true),
            PostingsFlags::Freqs
        );
        assert_eq!(
            postings_flags(true, true, false, false),
            PostingsFlags::Positions
        );
        assert_eq!(
            postings_flags(true, true, true, false),
            PostingsFlags::Offsets
        );
        assert_eq!(
            postings_flags(true, true, false, true),
            PostingsFlags::Payloads
        );
        assert_eq!(postings_flags(true, true, true, true), PostingsFlags::All);
        assert_eq!(rank(IndexOptions::DocsAndCustomFreqs), 2);
    }

    #[test]
    fn check_impacts_rejects_each_malformed_shape() {
        let i = |freq, norm| Impact { freq, norm };
        assert!(check_impacts(&[], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(1, 1)])], 6).is_err());
        assert!(check_impacts(&[(5, vec![i(1, 1)]), (4, vec![i(1, 1)])], 0).is_err());
        assert!(check_impacts(&[(5, vec![])], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(0, 1)])], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(1, 0)])], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(2, 1), i(1, 2)])], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(3, 1)]), (9, vec![i(2, 1)])], 0).is_err());
        assert!(check_impacts(&[(5, vec![i(2, 1)]), (9, vec![i(3, 1)])], 0).is_ok());
        assert!(check_impacts(
            &[(5, vec![i(1, 1), i(3, 2)]), (9, vec![i(2, 1), i(3, 2)])],
            0
        )
        .is_ok());
    }

    #[test]
    fn competitive_needs_an_impact_at_or_above_the_freq() {
        let impacts = [Impact { freq: 2, norm: 3 }, Impact { freq: 5, norm: 7 }];
        assert_competitive(&impacts, 2, 3);
        assert_competitive(&impacts, 4, 9);
        let r = std::panic::catch_unwind(|| assert_competitive(&impacts, 6, 9));
        assert!(r.is_err());
    }

    #[test]
    fn byte_dfa_matches_its_run_automaton() {
        let a = lucene_util::automaton::RegExp::new("ab*c|d[e-g]")
            .expect("regexp")
            .to_automaton()
            .expect("automaton");
        let a = operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).expect("determinize");
        let ca = CompiledAutomaton::new(&a).expect("compile");
        let run = ca.run_automaton.as_ref().expect("NORMAL").run_automaton();
        let dfa = byte_dfa(run);
        for s in ["ac", "abbc", "de", "dg", "dh", "a", "", "abcx"] {
            assert_eq!(dfa.run(s.as_bytes()), run_bytes(run, s.as_bytes()), "{s}");
        }
        assert!(!empty_dfa().run(b""));
        assert_eq!(utf8_of(&[0x61, 0xD800]), "a\u{FFFD}".as_bytes());
    }
}
