//! Unit tests of the reader layer's own edges: the paths a real index read
//! through `tests/reader_api_fixtures.rs` does not take (errors, empty
//! views, the delegation of every filter, the defaults of every trait).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_codecs::stored_fields::{Result as StoredResult, StoredFieldVisitor, VisitStatus};
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, NumericSortKey, SortedNumericSelector, SortedSetSelector,
    StringMissingValue,
};
use lucene_store::FsDirectory;

use super::exitable::{
    exitable_leaf_reader, ExitableDirectoryReader, ExitableSubReaderWrapper, QueryTimeout,
    QueryTimeoutImpl,
};
use super::filter::{
    FilterBinaryDocValues, FilterDirectoryReader, FilterLeafReader, FilterNumericDocValues,
    FilterPostingsEnum, FilterSortedDocValues, FilterSortedNumericDocValues,
    FilterSortedSetDocValues, FilterTerms, FilterTermsEnum, LeafFilter, NoFilter, SubReaderWrapper,
};
use super::filtered_terms_enum::{AcceptStatus, FilteredTermsEnum, TermFilter};
use super::merge_readers::{prepare_merge_readers, DefaultMergeHooks, MergeReaderHooks};
use super::multi_doc_values;
use super::multi_reader::MultiReader;
use super::parallel::{ParallelCompositeReader, ParallelLeafReader};
use super::slow_codec::{SlowCodecReaderWrapper, SlowCompositeCodecReaderWrapper};
use super::sorting::{sort_doc_map, DocMap, SortingCodecReader};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::multi_terms::MultiTerms;
use crate::Error;

fn fixture(name: &str) -> Arc<DirectoryReader> {
    let dir = format!(
        "{}/../../fixtures/data/reader_api/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    Arc::new(DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture"))
}

fn seg(r: &DirectoryReader, i: usize) -> Arc<dyn LeafReader> {
    Arc::new(r.segment_readers()[i].clone())
}

fn codec(r: &DirectoryReader, i: usize) -> Arc<dyn CodecReader> {
    Arc::new(r.segment_readers()[i].clone())
}

/// Every document of an iterator.
fn docs(it: &mut dyn DocIdSetIterator) -> Vec<i32> {
    let mut out = Vec::new();
    loop {
        let d = it.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            return out;
        }
        out.push(d);
    }
}

fn numerics(v: &mut dyn NumericDocValues) -> Vec<(i32, i64)> {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            return out;
        }
        out.push((d, v.long_value()));
    }
}

fn terms_of(te: &mut dyn TermsEnum) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(t) = te.next().unwrap() {
        out.push(String::from_utf8(t.to_vec()).unwrap());
    }
    out
}

/// A timeout that exits on its `exit_at`-th question.
#[derive(Debug)]
struct Countdown {
    exit_at: u32,
    calls: AtomicU32,
}

impl Countdown {
    #[allow(clippy::new_ret_no_self)]
    fn new(exit_at: u32) -> Arc<dyn QueryTimeout> {
        Arc::new(Self {
            exit_at,
            calls: AtomicU32::new(0),
        })
    }
}

impl QueryTimeout for Countdown {
    fn should_exit(&self) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst) + 1 >= self.exit_at
    }
}

/// Records every value it is handed.
struct Recorder(Vec<String>, VisitStatus);

impl Default for Recorder {
    fn default() -> Self {
        Recorder(Vec::new(), VisitStatus::Yes)
    }
}

impl StoredFieldVisitor for Recorder {
    fn needs_field(&mut self, n: i32) -> StoredResult<VisitStatus> {
        self.0.push(format!("needs:{n}"));
        Ok(self.1)
    }
    fn string_field(&mut self, n: i32, v: &str) -> StoredResult<()> {
        self.0.push(format!("{n}=s:{v}"));
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> StoredResult<()> {
        self.0.push(format!("{n}=b:{v:?}"));
        Ok(())
    }
    fn int_field(&mut self, n: i32, v: i32) -> StoredResult<()> {
        self.0.push(format!("{n}=i:{v}"));
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> StoredResult<()> {
        self.0.push(format!("{n}=l:{v}"));
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> StoredResult<()> {
        self.0.push(format!("{n}=f:{v}"));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> StoredResult<()> {
        self.0.push(format!("{n}=d:{v}"));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Postings, impacts, trait defaults
// ---------------------------------------------------------------------------

#[test]
fn materialized_postings_edges() {
    assert!(matches!(
        MaterializedPostings::new(vec![1], vec![], None),
        Err(Error::IllegalArgument(_))
    ));
    assert!(MaterializedPostings::new(vec![1], vec![1], Some(vec![])).is_err());
    let pos = |p: i32, pay: &[u8]| Position {
        position: p,
        start_offset: p * 2,
        end_offset: p * 2 + 1,
        payload: pay.to_vec(),
    };
    let mut pe = MaterializedPostings::new(
        vec![2, 5, 9],
        vec![2, 1, 1],
        Some(vec![
            vec![pos(0, b"x"), pos(3, b"")],
            vec![pos(1, b"")],
            vec![pos(4, b"")],
        ]),
    )
    .unwrap();
    assert_eq!(pe.doc_id(), -1);
    assert_eq!(pe.freq(), 0);
    assert_eq!(pe.start_offset(), -1);
    assert_eq!(pe.next_position().unwrap(), -1, "no current document");
    assert_eq!(pe.next_doc().unwrap(), 2);
    assert_eq!(pe.start_offset(), -1, "no position read yet");
    assert_eq!(pe.next_position().unwrap(), 0);
    assert_eq!((pe.start_offset(), pe.end_offset()), (0, 1));
    assert_eq!(pe.payload(), Some(&b"x"[..]));
    assert_eq!(pe.next_position().unwrap(), 3);
    assert_eq!(pe.payload(), None, "an empty payload reads as none");
    assert!(matches!(pe.next_position(), Err(Error::IllegalState(_))));
    assert_eq!(pe.advance(6).unwrap(), 9);
    assert_eq!(pe.cost(), 3);
    assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(pe.next_doc().unwrap(), NO_MORE_DOCS);
    let (d, f, p) = pe.into_parts();
    assert_eq!((d.len(), f.len(), p.map(|p| p.len())), (3, 3, Some(3)));

    let mut slow = SlowImpactsEnum::new(Box::new(
        MaterializedPostings::new(vec![1, 4], vec![3, 1], None).unwrap(),
    ));
    assert_eq!(slow.advance_shallow(0).unwrap(), NO_MORE_DOCS);
    assert_eq!(slow.impacts()[0].1[0].freq, i32::MAX);
    assert_eq!(slow.doc_id(), -1);
    assert_eq!(slow.advance(2).unwrap(), 4);
    assert_eq!(slow.freq(), 1);
    assert_eq!(slow.next_position().unwrap(), -1);
    assert_eq!(
        (slow.start_offset(), slow.end_offset(), slow.payload()),
        (-1, -1, None)
    );
    assert_eq!(slow.cost(), 2);
    assert_eq!(slow.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(PostingsFlags::Positions.wants_positions());
    assert!(!PostingsFlags::Freqs.wants_positions());
}

/// Terms over a sorted list, using every `Terms`/`TermsEnum` default.
struct VecTerms(Vec<&'static str>);

struct VecTermsEnum<'a> {
    terms: &'a [&'static str],
    at: Option<usize>,
}

impl TermsEnum for VecTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        let next = self.at.map_or(0, |i| i + 1);
        self.at = Some(next.min(self.terms.len()));
        Ok(self.term())
    }
    fn term(&self) -> Option<&[u8]> {
        self.at
            .and_then(|i| self.terms.get(i))
            .map(|t| t.as_bytes())
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let i = self.terms.partition_point(|t| t.as_bytes() < target);
        self.at = Some(i);
        Ok(match self.terms.get(i) {
            None => SeekStatus::End,
            Some(t) if t.as_bytes() == target => SeekStatus::Found,
            Some(_) => SeekStatus::NotFound,
        })
    }
    fn doc_freq(&mut self) -> Result<i32> {
        Ok(1)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        Ok(2)
    }
    fn postings(&mut self, _flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        let doc = self.at.unwrap_or(0) as i32;
        Ok(Box::new(MaterializedPostings::new(
            vec![doc],
            vec![2],
            None,
        )?))
    }
}

impl Terms for VecTerms {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(VecTermsEnum {
            terms: &self.0,
            at: None,
        }))
    }
    fn size(&self) -> i64 {
        self.0.len() as i64
    }
    fn sum_total_term_freq(&self) -> i64 {
        2 * self.0.len() as i64
    }
    fn sum_doc_freq(&self) -> i64 {
        self.0.len() as i64
    }
    fn doc_count(&self) -> i32 {
        self.0.len() as i32
    }
    fn has_freqs(&self) -> bool {
        true
    }
    fn has_offsets(&self) -> bool {
        false
    }
    fn has_positions(&self) -> bool {
        false
    }
    fn has_payloads(&self) -> bool {
        false
    }
}

#[test]
fn trait_defaults() {
    let t = VecTerms(vec!["apple", "banana", "fig", "foo", "fox", "zoo"]);
    assert_eq!(t.min().unwrap().unwrap(), b"apple");
    assert_eq!(t.max().unwrap().unwrap(), b"zoo");
    assert_eq!(VecTerms(vec![]).max().unwrap(), None);
    let mut te = t.iterator().unwrap();
    assert!(te.try_seek_exact(b"fig").unwrap());
    assert!(!te.try_seek_exact(b"fiz").unwrap());
    assert!(matches!(te.ord(), Err(Error::Unsupported(_))));
    assert!(matches!(te.seek_exact_ord(0), Err(Error::Unsupported(_))));
    let mut ie = te.impacts(PostingsFlags::Freqs).unwrap();
    assert_eq!(ie.next_doc().unwrap(), 3);

    // The default intersect is an `AutomatonTermsEnum` over `iterator()`.
    let dfa = lucene_codecs::regexp::RegexpPattern::new(b"f.*")
        .unwrap()
        .to_dfa()
        .unwrap();
    let mut te = t.intersect(&dfa, None).unwrap();
    assert_eq!(terms_of(&mut *te), ["fig", "foo", "fox"]);
    let mut te = t.intersect(&dfa, Some(b"foo")).unwrap();
    assert_eq!(terms_of(&mut *te), ["fox"]);
    assert!(te.try_seek_ceil(b"a").is_err());
    let mut te = t.intersect(&dfa, None).unwrap();
    assert_eq!(te.next().unwrap(), Some(&b"fig"[..]));
    assert_eq!(te.term(), Some(&b"fig"[..]));
    assert_eq!(te.doc_freq().unwrap(), 1);
    assert_eq!(te.total_term_freq().unwrap(), 2);
    assert_eq!(te.postings(PostingsFlags::Freqs).unwrap().cost(), 1);
}

/// A filter walking every `AcceptStatus`.
struct EveryStatus {
    seen: Vec<String>,
}

impl TermFilter for EveryStatus {
    fn accept(&mut self, term: &[u8]) -> Result<AcceptStatus> {
        self.seen.push(String::from_utf8(term.to_vec()).unwrap());
        Ok(match term {
            b"apple" => AcceptStatus::YesAndSeek,
            b"fig" => AcceptStatus::No,
            b"foo" => AcceptStatus::NoAndSeek,
            b"zoo" => AcceptStatus::End,
            _ => AcceptStatus::Yes,
        })
    }
    fn next_seek_term(
        &mut self,
        current: Option<&[u8]>,
        initial: Option<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        Ok(match current {
            None => initial,
            Some(b"apple") => Some(b"f".to_vec()),
            Some(_) => Some(b"x".to_vec()),
        })
    }
}

#[test]
fn filtered_terms_enum_statuses() {
    let t = VecTerms(vec!["apple", "banana", "fig", "foo", "fox", "zoo", "zzz"]);
    let mut fte = FilteredTermsEnum::new(
        t.iterator().unwrap(),
        EveryStatus { seen: Vec::new() },
        true,
    );
    fte.set_initial_seek_term(Some(b"a".to_vec()));
    // apple (yes+seek to "f") -> fig (no) -> foo (no+seek to "x") -> zoo (end).
    assert_eq!(terms_of(&mut fte), ["apple"]);
    assert_eq!(fte.filter().seen, ["apple", "fig", "foo", "zoo"]);
    assert!(fte.try_seek_ceil(b"a").is_err());
    assert!(fte.try_seek_exact(b"a").is_err());
    assert!(fte.seek_exact_ord(1).is_err());
    assert!(fte.ord().is_err());

    // Starting with a seek but no seek term: nothing, as in Java.
    let mut empty =
        FilteredTermsEnum::new(t.iterator().unwrap(), EveryStatus { seen: vec![] }, true);
    assert_eq!(empty.next().unwrap(), None);

    // Accepting everything but the end marker, without a seek.
    struct Pass;
    impl TermFilter for Pass {
        fn accept(&mut self, _: &[u8]) -> Result<AcceptStatus> {
            Ok(AcceptStatus::Yes)
        }
    }
    let mut all = FilteredTermsEnum::new(t.iterator().unwrap(), Pass, false);
    assert_eq!(all.next().unwrap(), Some(&b"apple"[..]));
    assert_eq!(all.doc_freq().unwrap(), 1);
    assert_eq!(all.total_term_freq().unwrap(), 2);
    assert_eq!(all.postings(PostingsFlags::Freqs).unwrap().cost(), 1);
    assert_eq!(all.impacts(PostingsFlags::Freqs).unwrap().cost(), 1);
    assert_eq!(terms_of(&mut all).len(), 6);
    // A seek past the last term ends the walk.
    struct SeekPast;
    impl TermFilter for SeekPast {
        fn accept(&mut self, _: &[u8]) -> Result<AcceptStatus> {
            Ok(AcceptStatus::NoAndSeek)
        }
        fn next_seek_term(
            &mut self,
            _: Option<&[u8]>,
            _: Option<Vec<u8>>,
        ) -> Result<Option<Vec<u8>>> {
            Ok(Some(b"zzzz".to_vec()))
        }
    }
    let mut past = FilteredTermsEnum::new(t.iterator().unwrap(), SeekPast, false);
    assert_eq!(past.next().unwrap(), None);
}

// ---------------------------------------------------------------------------
// Cache helpers, IndexReader defaults, field infos
// ---------------------------------------------------------------------------

#[test]
fn cache_helpers_and_closed_listeners() {
    let a = CacheHelper::new();
    let b = CacheHelper::default();
    assert_ne!(a.key(), b.key());
    assert_eq!(a.key(), a.clone().key());
    #[allow(clippy::mutable_key_type)]
    let mut set = std::collections::HashSet::new();
    set.insert(a.key());
    assert!(set.contains(&a.key()));
    assert!(format!("{:?}{:?}", a, a.key()).contains("CacheKey("));

    let fired = Arc::new(AtomicU32::new(0));
    let r = fixture("multi");
    let id = r.segment_readers()[0].core_cache_helper().key_id();
    let f = fired.clone();
    r.segment_readers()[0]
        .core_cache_helper()
        .add_closed_listener(Box::new(move |key| {
            assert_eq!(key, id);
            f.fetch_add(1, Ordering::SeqCst);
        }));
    let f = fired.clone();
    r.reader_cache_helper()
        .add_closed_listener(Box::new(move |_| {
            f.fetch_add(10, Ordering::SeqCst);
        }));
    let held = r.segment_readers()[0].clone();
    drop(r);
    assert_eq!(fired.load(Ordering::SeqCst), 10, "the core is still held");
    drop(held);
    assert_eq!(fired.load(Ordering::SeqCst), 11);
}

#[test]
fn index_reader_defaults() {
    let r = fixture("multi");
    assert!(r.has_deletions());
    assert_eq!(IndexReader::num_deleted_docs(&*r), 2);
    let leaves = r.leaves();
    assert!(matches!(
        leaf_for_doc(&leaves, 15),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        leaf_for_doc(&leaves, -1),
        Err(Error::IllegalArgument(_))
    ));
    let (leaf, local) = leaf_for_doc(&leaves, 7).unwrap();
    assert_eq!((leaf.ord, local), (1, 1));
    let mut rec = Recorder::default();
    r.stored_document(7, &mut rec).unwrap();
    assert!(rec.0.contains(&"0=s:d7".to_string()));
    assert!(r.document_term_vectors(7).unwrap().is_some());
    assert_eq!(IndexReader::doc_freq(&*r, "body", b"nope").unwrap(), 0);
    assert_eq!(IndexReader::total_term_freq(&*r, "nope", b"x").unwrap(), 0);
    let leaf = seg(&r, 0);
    assert_eq!(leaf.leaves().len(), 1);
    assert!(leaf.reader_cache_helper().is_some());
    assert!(leaf
        .postings("body", b"fox", PostingsFlags::Freqs)
        .unwrap()
        .is_some());
    assert!(leaf
        .postings("body", b"nope", PostingsFlags::Freqs)
        .unwrap()
        .is_none());
    assert!(leaf
        .postings("nope", b"fox", PostingsFlags::Freqs)
        .unwrap()
        .is_none());
    assert_eq!(r.leaf_handles().len(), 3);
}

#[test]
fn field_infos_builder_renumbers_conflicts() {
    let mut b = FieldInfosBuilder::default();
    b.add(&FieldInfo::new("a", 0));
    b.add(&FieldInfo::new("b", 0)); // taken: lowest free
    b.add(&FieldInfo::new("c", 1)); // taken by b: next free
    b.add(&FieldInfo::new("a", 7)); // seen: kept as first
    b.add(&FieldInfo::new("d", 9));
    let infos = b.finish();
    let got: Vec<_> = infos
        .fields
        .iter()
        .map(|f| format!("{}:{}", f.name, f.number))
        .collect();
    assert_eq!(got, ["a:0", "b:1", "c:2", "d:9"]);
}

#[test]
fn remapping_visitor_renumbers_every_kind() {
    let from = FieldInfos {
        fields: vec![FieldInfo::new("x", 0), FieldInfo::new("y", 1)],
    };
    let to = FieldInfos {
        fields: vec![FieldInfo::new("y", 5), FieldInfo::new("x", 6)],
    };
    let mut rec = Recorder::default();
    let mut v = RemappingVisitor {
        inner: &mut rec,
        from: &from,
        to: &to,
    };
    v.needs_field(0).unwrap();
    v.string_field(0, "s").unwrap();
    v.binary_field(1, b"b").unwrap();
    v.int_field(0, 1).unwrap();
    v.long_field(1, 2).unwrap();
    v.float_field(0, 1.5).unwrap();
    v.double_field(1, 2.5).unwrap();
    v.string_field(9, "unknown keeps its number").unwrap();
    assert_eq!(
        rec.0,
        [
            "needs:6",
            "6=s:s",
            "5=b:[98]",
            "6=i:1",
            "5=l:2",
            "6=f:1.5",
            "5=d:2.5",
            "9=s:unknown keeps its number"
        ]
    );
}

// ---------------------------------------------------------------------------
// Segment doc values iterators
// ---------------------------------------------------------------------------

#[test]
fn segment_doc_values_positioning() {
    let r = fixture("multi");
    let s = seg(&r, 0);
    // Sparse numeric: docs 0,1,2,4,5.
    let mut v = s.numeric_doc_values("rank").unwrap().unwrap();
    assert_eq!(v.cost(), 5);
    assert!(!v.advance_exact(3).unwrap());
    assert_eq!(v.doc_id(), 3);
    assert!(v.advance_exact(4).unwrap());
    assert_eq!(v.long_value(), 3);
    assert_eq!(v.next_doc().unwrap(), 5);
    assert_eq!(v.advance(6).unwrap(), NO_MORE_DOCS);
    // Dense norms.
    let mut n = s.norm_values("body").unwrap().unwrap();
    assert_eq!(n.advance(3).unwrap(), 3);
    assert!(n.advance_exact(5).unwrap());
    assert!(!n.advance_exact(6).unwrap());
    assert_eq!(n.advance(9).unwrap(), NO_MORE_DOCS);
    // Wrong type or absent: none.
    assert!(s.numeric_doc_values("kw").unwrap().is_none());
    assert!(s.binary_doc_values("nope").unwrap().is_none());
    assert!(
        s.norm_values("id").unwrap().is_none(),
        "StringField omits norms"
    );
    assert!(s.norm_values("nope").unwrap().is_none());
    assert!(s.point_values("body").unwrap().is_none());
    assert!(s.point_values("nope").unwrap().is_none());
    assert!(s.float_vector_values("bvec").unwrap().is_none());
    assert!(s.byte_vector_values("vec").unwrap().is_none());
    assert!(s.float_vector_values("nope").unwrap().is_none());
    assert!(s.byte_vector_values("nope").unwrap().is_none());
    assert!(s.terms("nope").unwrap().is_none());
    // Sorted: lookup_term default, advance, past the values.
    let mut kw = s.sorted_doc_values("kw").unwrap().unwrap();
    assert_eq!(kw.lookup_term(b"kilo").unwrap(), 2);
    assert_eq!(kw.lookup_term(b"beta").unwrap(), -2);
    assert_eq!(kw.lookup_term(b"zz").unwrap(), -6);
    assert_eq!(kw.advance(2).unwrap(), 2);
    assert!(kw.advance_exact(3).unwrap());
    let mut sn = s.sorted_numeric_doc_values("nums").unwrap().unwrap();
    assert_eq!(sn.next_doc().unwrap(), 0);
    assert_eq!(sn.doc_value_count(), 2);
    sn.next_value().unwrap();
    sn.next_value().unwrap();
    assert!(matches!(sn.next_value(), Err(Error::IllegalState(_))));
    assert!(!sn.advance_exact(2).unwrap());
    let mut ss = s.sorted_set_doc_values("tags").unwrap().unwrap();
    assert!(ss.advance_exact(3).unwrap());
    assert_eq!(ss.doc_value_count(), 3);
    for _ in 0..3 {
        ss.next_ord().unwrap();
    }
    assert!(ss.next_ord().is_err());
    assert_eq!(ss.advance(4).unwrap(), 4);
    let mut b = s.binary_doc_values("bin").unwrap().unwrap();
    assert!(b.advance_exact(2).unwrap());
    assert_eq!(b.binary_value(), [2, 253, 7]);
    // Points: the estimate, through the dyn visitor.
    let pv = s.point_values("pt").unwrap().unwrap();
    struct Inside;
    impl IntersectVisitor for Inside {
        fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
            Relation::CellInsideQuery
        }
        fn visit(&mut self, _: i32) {}
        fn visit_with_value(&mut self, _: i32, _: &[u8]) {}
    }
    assert_eq!(pv.estimate_point_count(&mut Inside).unwrap(), 6);
    let mut seen = Recorded::default();
    pv.intersect(&mut seen).unwrap();
    assert_eq!(seen.0.len(), 6, "inside cells hand over doc ids only");
    assert_eq!(seen.1, 6, "the walk's grow reaches the visitor");
}

/// Collects `visit`/`visit_many` doc ids, and sums the `grow` hints.
#[derive(Default)]
struct Recorded(Vec<i32>, usize);

impl IntersectVisitor for Recorded {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellInsideQuery
    }
    fn visit(&mut self, doc: i32) {
        self.0.push(doc);
    }
    fn visit_with_value(&mut self, doc: i32, _: &[u8]) {
        self.0.push(doc);
    }
    fn grow(&mut self, count: usize) {
        self.1 += count;
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

/// Hides every even document and one stored field.
struct HideEven {
    live: FixedBitSet,
}

impl LeafFilter for HideEven {
    fn live_docs<'a>(&'a self, _inner: Option<&'a FixedBitSet>) -> Option<&'a FixedBitSet> {
        Some(&self.live)
    }
    fn document(
        &self,
        inner: &dyn LeafReader,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> Result<()> {
        inner.document(doc + 1, visitor)
    }
    fn term_vectors(&self, _: &dyn LeafReader, _: i32) -> Result<Option<TermVectorsDocument>> {
        Ok(None)
    }
}

#[test]
fn filter_leaf_reader_delegates_and_overrides() {
    let r = fixture("multi");
    let s = seg(&r, 0);
    let f = FilterLeafReader::new(s.clone(), Box::new(NoFilter));
    assert_eq!(f.max_doc(), 6);
    assert_eq!(f.num_docs(), 5);
    assert_eq!(f.leaves().len(), 1);
    assert!(f.reader_cache_helper().is_some() && f.core_cache_helper().is_some());
    assert_eq!(f.field_infos().fields.len(), s.field_infos().fields.len());
    assert!(f.live_docs().is_some());
    assert!(Arc::ptr_eq(f.delegate(), &s));
    let t = f.terms("body").unwrap().unwrap();
    assert_eq!(t.doc_count(), 6);
    assert!(f.terms("nope").unwrap().is_none());
    assert_eq!(
        numerics(&mut *f.numeric_doc_values("rank").unwrap().unwrap()).len(),
        5
    );
    assert!(f.numeric_doc_values("nope").unwrap().is_none());
    assert_eq!(
        docs(&mut *f.binary_doc_values("bin").unwrap().unwrap()),
        [0, 2, 4]
    );
    assert!(f.binary_doc_values("nope").unwrap().is_none());
    assert_eq!(
        docs(&mut *f.sorted_doc_values("kw").unwrap().unwrap()).len(),
        6
    );
    assert!(f.sorted_doc_values("nope").unwrap().is_none());
    assert_eq!(
        docs(&mut *f.sorted_numeric_doc_values("nums").unwrap().unwrap()).len(),
        5
    );
    assert!(f.sorted_numeric_doc_values("nope").unwrap().is_none());
    assert_eq!(
        docs(&mut *f.sorted_set_doc_values("tags").unwrap().unwrap()).len(),
        5
    );
    assert!(f.sorted_set_doc_values("nope").unwrap().is_none());
    assert_eq!(
        numerics(&mut *f.norm_values("body").unwrap().unwrap()).len(),
        6
    );
    assert!(f.norm_values("nope").unwrap().is_none());
    assert_eq!(f.point_values("pt").unwrap().unwrap().size(), 6);
    assert!(f.point_values("nope").unwrap().is_none());
    assert_eq!(f.float_vector_values("vec").unwrap().unwrap().size(), 4);
    assert!(f.float_vector_values("nope").unwrap().is_none());
    assert_eq!(f.byte_vector_values("bvec").unwrap().unwrap().size(), 3);
    assert!(f.byte_vector_values("nope").unwrap().is_none());
    let mut rec = Recorder::default();
    f.document(1, &mut rec).unwrap();
    assert!(rec.0.contains(&"0=s:d1".to_string()));
    assert!(f.term_vectors(1).unwrap().is_some());
    assert!(f.index_sort().is_none());

    let mut live = FixedBitSet::new(6);
    for d in [1, 3, 5] {
        // FBS: the bitset holds six bits and every index here is below six.
        live.set(d);
    }
    let hide = FilterLeafReader::new(s.clone(), Box::new(HideEven { live }));
    assert_eq!(hide.num_docs(), 3);
    assert!(hide.core_cache_helper().is_none() && hide.reader_cache_helper().is_none());
    let mut rec = Recorder::default();
    hide.document(1, &mut rec).unwrap();
    assert!(rec.0.contains(&"0=s:d2".to_string()));
    assert!(hide.term_vectors(1).unwrap().is_none());

    // `FilterCodecReader` is a codec reader.
    let fc: Arc<dyn CodecReader> = Arc::new(super::filter::FilterCodecReader::new(
        codec(&r, 1),
        Box::new(NoFilter),
    ));
    assert_eq!(fc.max_doc(), 5);
    let mut rec = Recorder::default();
    fc.document(0, &mut rec).unwrap();
    assert!(fc.term_vectors(0).unwrap().is_some());
}

#[test]
fn access_object_filters_delegate() {
    let r = fixture("multi");
    let s = seg(&r, 0);
    let t = FilterTerms {
        in_: s.terms("body").unwrap().unwrap(),
    };
    assert_eq!(
        (
            t.size(),
            t.sum_total_term_freq(),
            t.sum_doc_freq(),
            t.doc_count()
        ),
        (13, 22, 20, 6)
    );
    assert!(t.has_freqs() && t.has_positions() && t.has_offsets() && !t.has_payloads());
    assert_eq!(t.min().unwrap().unwrap(), b"and");
    assert_eq!(t.max().unwrap().unwrap(), b"the");
    let dfa = lucene_codecs::regexp::RegexpPattern::new(b"f.*")
        .unwrap()
        .to_dfa()
        .unwrap();
    assert_eq!(terms_of(&mut *t.intersect(&dfa, None).unwrap()), ["fox"]);
    let mut te = FilterTermsEnum {
        in_: t.iterator().unwrap(),
    };
    assert_eq!(te.try_seek_ceil(b"fo").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.term(), Some(&b"fox"[..]));
    assert!(te.try_seek_exact(b"quick").unwrap());
    assert_eq!(te.doc_freq().unwrap(), 2);
    assert_eq!(te.total_term_freq().unwrap(), 3);
    assert!(te.ord().is_err());
    assert!(te.seek_exact_ord(0).is_err());
    assert_eq!(te.impacts(PostingsFlags::Freqs).unwrap().cost(), 2);
    let mut pe = FilterPostingsEnum {
        in_: te.postings(PostingsFlags::All).unwrap(),
    };
    assert_eq!(pe.doc_id(), -1);
    assert_eq!(pe.next_doc().unwrap(), 0);
    assert_eq!(pe.freq(), 1);
    assert_eq!(pe.next_position().unwrap(), 1);
    assert_eq!((pe.start_offset(), pe.end_offset()), (4, 9));
    assert_eq!(pe.payload(), None);
    assert_eq!(pe.advance(1).unwrap(), 1);
    assert_eq!(pe.cost(), 2);
    assert_eq!(te.next().unwrap(), Some(&b"quiet"[..]));

    let mut n = FilterNumericDocValues {
        in_: s.numeric_doc_values("rank").unwrap().unwrap(),
    };
    assert_eq!(n.advance(3).unwrap(), 4);
    assert_eq!(n.long_value(), 3);
    assert!(n.advance_exact(5).unwrap());
    assert_eq!((n.doc_id(), n.cost()), (5, 5));
    assert_eq!(n.next_doc().unwrap(), NO_MORE_DOCS);
    let mut b = FilterBinaryDocValues {
        in_: s.binary_doc_values("bin").unwrap().unwrap(),
    };
    assert_eq!(b.next_doc().unwrap(), 0);
    assert_eq!(b.binary_value(), [0, 255, 7]);
    let mut so = FilterSortedDocValues {
        in_: s.sorted_doc_values("kw").unwrap().unwrap(),
    };
    assert!(so.advance_exact(1).unwrap());
    assert_eq!(so.ord_value(), 0);
    assert_eq!(so.lookup_ord(0).unwrap(), b"alpha");
    assert_eq!(so.value_count(), 5);
    assert_eq!(so.lookup_term(b"mike").unwrap(), 3);
    let mut sn = FilterSortedNumericDocValues {
        in_: s.sorted_numeric_doc_values("nums").unwrap().unwrap(),
    };
    assert_eq!(sn.next_doc().unwrap(), 0);
    assert_eq!(sn.doc_value_count(), 2);
    assert_eq!(sn.next_value().unwrap(), -10);
    let mut ss = FilterSortedSetDocValues {
        in_: s.sorted_set_doc_values("tags").unwrap().unwrap(),
    };
    assert_eq!(ss.next_doc().unwrap(), 0);
    assert_eq!(ss.doc_value_count(), 2);
    assert_eq!(ss.next_ord().unwrap(), 1);
    assert_eq!(ss.lookup_ord(1).unwrap(), b"blue");
    assert_eq!(ss.value_count(), 4);
}

struct Identity;

impl SubReaderWrapper for Identity {
    fn wrap(&self, reader: Arc<dyn LeafReader>) -> Result<Arc<dyn LeafReader>> {
        Ok(reader)
    }
}

#[test]
fn filter_directory_reader() {
    let r = fixture("multi");
    let dir = FsDirectory::open(format!(
        "{}/../../fixtures/data/reader_api/multi",
        env!("CARGO_MANIFEST_DIR")
    ));
    let f = FilterDirectoryReader::new(r.clone(), Arc::new(Identity)).unwrap();
    assert!(Arc::ptr_eq(f.delegate(), &r) && Arc::ptr_eq(f.unwrap(), &r));
    assert_eq!(f.wrapped_leaves().len(), 3);
    assert_eq!((f.max_doc(), f.num_docs()), (15, 13));
    assert_eq!(f.leaves().len(), 3);
    assert_eq!(f.leaf_handles().len(), 3);
    assert!(f.reader_cache_helper().is_some());
    assert_eq!(f.version(), r.segment_infos.version);
    assert!(f.open_if_changed(&dir).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Exitable
// ---------------------------------------------------------------------------

#[test]
fn query_timeout_impl() {
    assert!(!QueryTimeoutImpl::new(-1).should_exit());
    assert!(QueryTimeoutImpl::new(-1).timeout_at().is_none());
    let t = QueryTimeoutImpl::new(0);
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert!(t.should_exit());
    assert!(!QueryTimeoutImpl::new(60_000).should_exit());
}

#[test]
fn exitable_points_vectors_and_terms() {
    let r = fixture("multi");
    // Points: a timeout already fired stops the intersection up front...
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(1));
    let pv = x.point_values("pt").unwrap().unwrap();
    assert!(matches!(
        pv.intersect(&mut Recorded::default()),
        Err(Error::ExitingReader(_))
    ));
    // ...one firing inside it unwinds the walk and reports the exit...
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(2));
    let pv = x.point_values("pt").unwrap().unwrap();
    let mut seen = Recorded::default();
    assert!(matches!(
        pv.intersect(&mut seen),
        Err(Error::ExitingReader(_))
    ));
    assert!(seen.0.is_empty());
    // ...and one that never fires lets it finish.
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(1000));
    let pv = x.point_values("pt").unwrap().unwrap();
    let mut seen = Recorded::default();
    pv.intersect(&mut seen).unwrap();
    assert_eq!(seen.0.len(), 6);
    assert_eq!(seen.1, 6, "grow is forwarded");
    assert_eq!(
        pv.estimate_point_count(&mut Recorded::default()).unwrap(),
        6
    );
    assert_eq!(
        (
            pv.num_dimensions(),
            pv.num_index_dimensions(),
            pv.bytes_per_dimension(),
            pv.size(),
            pv.doc_count()
        ),
        (2, 2, 4, 6, 6)
    );
    assert_eq!(pv.min_packed_value().len(), 8);
    assert_eq!(pv.max_packed_value().len(), 8);
    // Visits sampled every 16th call: a crossing walk with a late exit.
    struct Crossing(usize);
    impl IntersectVisitor for Crossing {
        fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
            Relation::CellCrossesQuery
        }
        fn visit(&mut self, _: i32) {
            self.0 += 1;
        }
        fn visit_with_value(&mut self, _: i32, _: &[u8]) {
            self.0 += 1;
        }
    }
    let mut c = Crossing(0);
    pv.intersect(&mut c).unwrap();
    assert_eq!(c.0, 6);

    // Vectors: `ord_to_doc` checks every 1000 documents.
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(1));
    let fv = x.float_vector_values("vec").unwrap().unwrap();
    assert_eq!((fv.dimension(), fv.size()), (3, 4));
    assert_eq!(fv.vector_value(1).unwrap(), [2.0, 3.0, -2.0]);
    assert!(matches!(fv.ord_to_doc(0), Err(Error::ExitingReader(_))));
    let bv = x.byte_vector_values("bvec").unwrap().unwrap();
    assert_eq!((bv.dimension(), bv.size()), (2, 3));
    assert_eq!(bv.vector_value(1).unwrap(), [2, 254]);
    assert!(bv.ord_to_doc(0).is_err());
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(2));
    let bv = x.byte_vector_values("bvec").unwrap().unwrap();
    assert_eq!(bv.ord_to_doc(0).unwrap(), 0);
    assert_eq!(bv.ord_to_doc(1).unwrap(), 2, "within 1000 docs: no check");

    // Terms: statistics and seeks pass through; intersect is checked.
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(3));
    let t = x.terms("body").unwrap().unwrap();
    assert_eq!(
        (
            t.size(),
            t.sum_total_term_freq(),
            t.sum_doc_freq(),
            t.doc_count()
        ),
        (13, 22, 20, 6)
    );
    assert!(t.has_freqs() && t.has_positions() && t.has_offsets() && !t.has_payloads());
    assert_eq!(t.min().unwrap().unwrap(), b"and");
    assert_eq!(t.max().unwrap().unwrap(), b"the");
    let mut te = t.iterator().unwrap();
    assert!(te.try_seek_exact(b"fox").unwrap());
    assert_eq!(te.try_seek_ceil(b"fp").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.term(), Some(&b"jumps"[..]));
    assert_eq!(te.doc_freq().unwrap(), 1);
    assert_eq!(te.total_term_freq().unwrap(), 1);
    assert_eq!(te.postings(PostingsFlags::Freqs).unwrap().cost(), 1);
    assert_eq!(te.impacts(PostingsFlags::Freqs).unwrap().cost(), 1);
    assert!(te.ord().is_err() && te.seek_exact_ord(1).is_err());
    let dfa = lucene_codecs::regexp::RegexpPattern::new(b"q.*")
        .unwrap()
        .to_dfa()
        .unwrap();
    assert_eq!(terms_of(&mut *t.intersect(&dfa, None).unwrap()).len(), 4);
    let x1 = exitable_leaf_reader(seg(&r, 0), Countdown::new(1));
    let t1 = x1.terms("body").unwrap().unwrap();
    assert!(matches!(
        t1.intersect(&dfa, None),
        Err(Error::ExitingReader(_))
    ));
    assert!(x.core_cache_helper().is_some());

    // Doc-values wrappers expose the values they iterate.
    let x = exitable_leaf_reader(seg(&r, 0), Countdown::new(1000));
    let mut kw = x.sorted_doc_values("kw").unwrap().unwrap();
    assert_eq!(kw.advance(1).unwrap(), 1);
    assert_eq!(kw.ord_value(), 0);
    assert_eq!(kw.lookup_ord(0).unwrap(), b"alpha");
    assert_eq!(kw.value_count(), 5);
    let mut ss = x.sorted_set_doc_values("tags").unwrap().unwrap();
    assert_eq!(ss.advance(3).unwrap(), 3);
    assert!(ss.advance_exact(4).unwrap());
    assert_eq!(ss.doc_value_count(), 2);
    assert_eq!(ss.next_ord().unwrap(), 1);
    assert_eq!(ss.lookup_ord(3).unwrap(), b"red");
    assert_eq!(ss.value_count(), 4);
    assert_eq!(ss.doc_id(), 4);
    assert!(ss.cost() > 0);
    let mut sn = x.sorted_numeric_doc_values("nums").unwrap().unwrap();
    assert!(sn.advance_exact(4).unwrap());
    assert_eq!(sn.doc_value_count(), 2);
    assert_eq!(sn.next_value().unwrap(), -4);
    let mut b = x.binary_doc_values("bin").unwrap().unwrap();
    assert!(b.advance_exact(4).unwrap());
    assert_eq!(b.binary_value(), [4, 251, 7]);
    let mut n = x.numeric_doc_values("rank").unwrap().unwrap();
    assert!(n.advance_exact(1).unwrap());
    assert_eq!(n.long_value(), 2);

    // The directory-level wrapper.
    let d = ExitableDirectoryReader::wrap(r.clone(), Countdown::new(1000)).unwrap();
    assert_eq!(d.leaves().len(), 3);
    let w = ExitableSubReaderWrapper::new(Countdown::new(1));
    let leaf = w.wrap(seg(&r, 1)).unwrap();
    assert!(leaf.terms("body").unwrap().unwrap().iterator().is_err());
}

// ---------------------------------------------------------------------------
// Composites
// ---------------------------------------------------------------------------

#[test]
fn multi_reader_edges() {
    let m = fixture("multi");
    let a = fixture("par_a");
    let mr = MultiReader::new(vec![
        ReaderHandle::Leaf(seg(&m, 0)),
        ReaderHandle::Composite(a.clone()),
    ])
    .unwrap();
    assert_eq!(mr.max_doc(), 11);
    assert_eq!(mr.reader_index(6).unwrap(), 1);
    assert_eq!(mr.reader_index(5).unwrap(), 0);
    assert!(mr.reader_index(11).is_err() && mr.reader_index(-1).is_err());
    assert_eq!(mr.reader_base(1), Some(6));
    assert_eq!(mr.sub_readers().len(), 2);
    assert!(mr.reader_cache_helper().is_none());
    assert_eq!(mr.leaf_handles().len(), 3);
    assert_eq!(mr.sequential_sub_readers().len(), 2);
    let one = MultiReader::new(vec![ReaderHandle::Composite(a.clone())]).unwrap();
    assert!(one.reader_cache_helper().is_some());
    let one_leaf = MultiReader::new(vec![ReaderHandle::Leaf(seg(&m, 0))]).unwrap();
    assert!(one_leaf.reader_cache_helper().is_some());
    let sorted = MultiReader::with_sorter(
        vec![
            ReaderHandle::Composite(m.clone()),
            ReaderHandle::Leaf(seg(&a, 1)),
        ],
        |x, y| x.max_doc().cmp(&y.max_doc()),
    )
    .unwrap();
    assert_eq!(sorted.reader_base(1), Some(2));
    assert_eq!(ReaderHandle::Leaf(seg(&m, 0)).num_docs(), 5);
    // Nested composites flatten depth first.
    let nested = MultiReader::new(vec![ReaderHandle::Composite(Arc::new(mr))]).unwrap();
    let leaves = nested.leaves();
    assert_eq!(
        leaves.iter().map(|l| l.doc_base).collect::<Vec<_>>(),
        [0, 6, 9]
    );
}

#[test]
fn multi_doc_values_edges() {
    let m = fixture("multi");
    let r = &*m;
    let mut rank = multi_doc_values::numeric_values(r, "rank")
        .unwrap()
        .unwrap();
    assert_eq!(rank.advance(3).unwrap(), 4);
    assert!(matches!(rank.advance(4), Err(Error::IllegalArgument(_))));
    assert_eq!(rank.advance(10).unwrap(), 11, "leaf 2 starts at 11");
    assert!(!rank.advance_exact(12).unwrap_or(true) || rank.long_value() >= 0);
    assert!(rank.advance_exact(13).unwrap());
    assert_eq!(rank.long_value(), 1);
    assert!(matches!(
        rank.advance_exact(2),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        rank.advance_exact(99),
        Err(Error::IllegalArgument(_))
    ));
    assert_eq!(rank.advance(99).unwrap(), NO_MORE_DOCS);
    assert_eq!(rank.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(rank.cost() > 0);
    let mut bin = multi_doc_values::binary_values(r, "bin").unwrap().unwrap();
    assert!(!bin.advance_exact(7).unwrap());
    assert!(bin.binary_value().is_empty(), "positioned on no value");
    assert_eq!(bin.advance(13).unwrap(), 14);
    let mut kw = multi_doc_values::sorted_values(r, "kw").unwrap().unwrap();
    assert_eq!(kw.advance(6).unwrap(), 6);
    assert!(kw.advance_exact(7).unwrap());
    assert_eq!(kw.lookup_ord(kw.ord_value()).unwrap(), b"kilo");
    assert!(kw.lookup_ord(99).is_err());
    assert!(kw.cost() > 0);
    let mut tags = multi_doc_values::sorted_set_values(r, "tags")
        .unwrap()
        .unwrap();
    assert!(tags.advance_exact(7).unwrap());
    assert_eq!(tags.doc_value_count(), 3);
    assert_eq!(tags.advance(8).unwrap(), 8);
    assert!(tags.lookup_ord(99).is_err());
    assert_eq!(tags.doc_id(), 8);
    assert!(tags.cost() > 0);
    let mut nums = multi_doc_values::sorted_numeric_values(r, "nums")
        .unwrap()
        .unwrap();
    assert!(!nums.advance_exact(6).unwrap());
    assert_eq!(nums.doc_value_count(), 0);
    // Absent fields, and the one- and zero-leaf shortcuts.
    assert!(multi_doc_values::numeric_values(r, "kw").unwrap().is_none());
    assert!(multi_doc_values::binary_values(r, "nope")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_values(r, "nope")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_set_values(r, "nope")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_numeric_values(r, "nope")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::norm_values(r, "id").unwrap().is_none());
    let leaf = seg(&m, 0);
    assert!(multi_doc_values::numeric_values(&*leaf, "rank")
        .unwrap()
        .is_some());
    assert!(multi_doc_values::binary_values(&*leaf, "bin")
        .unwrap()
        .is_some());
    assert!(multi_doc_values::sorted_values(&*leaf, "kw")
        .unwrap()
        .is_some());
    assert!(multi_doc_values::sorted_set_values(&*leaf, "tags")
        .unwrap()
        .is_some());
    assert!(multi_doc_values::sorted_numeric_values(&*leaf, "nums")
        .unwrap()
        .is_some());
    assert!(multi_doc_values::norm_values(&*leaf, "body")
        .unwrap()
        .is_some());
    let empty = MultiReader::new(Vec::new()).unwrap();
    assert!(multi_doc_values::numeric_values(&empty, "rank")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::binary_values(&empty, "bin")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_values(&empty, "kw")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_set_values(&empty, "tags")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::sorted_numeric_values(&empty, "nums")
        .unwrap()
        .is_none());
    assert!(multi_doc_values::norm_values(&empty, "body")
        .unwrap()
        .is_none());
    // A leaf without the field reads as no values there.
    let a = fixture("par_a");
    let mixed = MultiReader::new(vec![
        ReaderHandle::Leaf(seg(&a, 0)),
        ReaderHandle::Leaf(seg(&m, 0)),
    ])
    .unwrap();
    let mut rank = multi_doc_values::numeric_values(&mixed, "rank")
        .unwrap()
        .unwrap();
    assert!(!rank.advance_exact(1).unwrap());
    assert_eq!(rank.long_value(), 0);
    assert_eq!(rank.advance(2).unwrap(), 3);
    let mut sn = multi_doc_values::sorted_numeric_values(&mixed, "nums")
        .unwrap()
        .unwrap();
    assert!(!sn.advance_exact(0).unwrap());
    assert!(sn.next_value().is_err());
    let mut ss = multi_doc_values::sorted_set_values(&mixed, "tags")
        .unwrap()
        .unwrap();
    assert!(!ss.advance_exact(0).unwrap());
    assert!(ss.next_ord().is_err());
    assert_eq!(ss.doc_value_count(), 0);
    let mut kw = multi_doc_values::sorted_values(&mixed, "kw")
        .unwrap()
        .unwrap();
    assert!(!kw.advance_exact(0).unwrap());
    assert_eq!(kw.ord_value(), -1);
}

#[test]
fn multi_terms_edges() {
    let m = fixture("multi");
    // One leaf: its own terms.
    let leaf = seg(&m, 0);
    let t = MultiTerms::get_terms(&*leaf, "body").unwrap().unwrap();
    assert_eq!(t.size(), 13);
    let t = MultiTerms::get_terms(&*m, "body").unwrap().unwrap();
    assert!(t.has_freqs() && t.has_positions() && t.has_offsets() && !t.has_payloads());
    assert!(MultiTerms::get_terms(&*m, "nope").unwrap().is_none());
    let mut te = t.iterator().unwrap();
    assert_eq!(te.try_seek_ceil(b"zzz").unwrap(), SeekStatus::End);
    assert_eq!(te.try_seek_ceil(b"quick").unwrap(), SeekStatus::Found);
    let mut pe = te.postings(PostingsFlags::Positions).unwrap();
    assert_eq!(pe.advance(4).unwrap(), 7);
    assert_eq!(pe.freq(), 1);
    assert_eq!(pe.next_position().unwrap(), 3);
    assert_eq!(
        (pe.start_offset(), pe.end_offset(), pe.payload()),
        (12, 17, None)
    );
    assert!(pe.cost() > 0 && pe.doc_id() == 7);
    assert_eq!(
        crate::multi_terms::indexed_fields(&*m),
        ["body", "id", "kw", "pay"]
    );
}

#[test]
fn parallel_edges() {
    let a = fixture("par_a");
    let b = fixture("par_b");
    let m = fixture("multi");
    let a0 = seg(&a, 0);
    let b0 = seg(&b, 0);
    assert!(matches!(
        ParallelLeafReader::with_stored_fields_readers(vec![], vec![a0.clone()]),
        Err(Error::IllegalArgument(_))
    ));
    assert!(matches!(
        ParallelLeafReader::new(vec![a0.clone(), seg(&m, 0)]),
        Err(Error::IllegalArgument(_))
    ));
    let empty = ParallelLeafReader::new(vec![]).unwrap();
    assert_eq!((empty.max_doc(), empty.num_docs()), (0, 0));
    // One reader serving everything keeps its cache helpers.
    let single = ParallelLeafReader::new(vec![a0.clone()]).unwrap();
    assert!(single.core_cache_helper().is_some() && single.reader_cache_helper().is_some());
    assert_eq!(single.parallel_readers().len(), 1);
    assert_eq!(single.stored_fields_readers().len(), 1);
    assert_eq!(single.leaves().len(), 1);
    assert!(single.term_vectors(0).unwrap().is_none(), "par_a has none");
    let p = ParallelLeafReader::new(vec![a0.clone(), b0.clone()]).unwrap();
    assert!(p.core_cache_helper().is_none() && p.reader_cache_helper().is_none());
    for f in ["nope", "title"] {
        assert!(p.numeric_doc_values(f).unwrap().is_none());
        assert!(p.binary_doc_values(f).unwrap().is_none());
        assert!(p.sorted_doc_values(f).unwrap().is_none());
        assert!(p.sorted_numeric_doc_values(f).unwrap().is_none());
        assert!(p.sorted_set_doc_values(f).unwrap().is_none());
        assert!(p.point_values(f).unwrap().is_none());
        assert!(p.float_vector_values(f).unwrap().is_none());
        assert!(p.byte_vector_values(f).unwrap().is_none());
    }
    assert!(p.terms("nope").unwrap().is_none());
    assert!(p.norm_values("nope").unwrap().is_none());
    // Index sorts must agree.
    let sorted_a: Arc<dyn LeafReader> = Arc::new(
        SortingCodecReader::wrap(
            codec(&a, 0),
            None,
            vec![IndexSortField::long("na", false, None)],
        )
        .unwrap(),
    );
    let sorted_b: Arc<dyn LeafReader> = Arc::new(
        SortingCodecReader::wrap(
            codec(&b, 0),
            None,
            vec![IndexSortField::long("x", true, None)],
        )
        .unwrap(),
    );
    assert!(
        ParallelLeafReader::new(vec![sorted_a.clone(), sorted_a.clone()])
            .unwrap()
            .index_sort()
            .is_some()
    );
    assert!(ParallelLeafReader::new(vec![sorted_a, sorted_b]).is_err());

    // Composite: leaf structures must agree.
    assert!(ParallelCompositeReader::new(vec![a.clone(), m.clone()]).is_err());
    assert!(ParallelCompositeReader::with_stored_fields_readers(vec![], vec![a.clone()]).is_err());
    let none = ParallelCompositeReader::new(vec![]).unwrap();
    assert_eq!(none.max_doc(), 0);
    let solo = ParallelCompositeReader::new(vec![a.clone()]).unwrap();
    assert!(solo.reader_cache_helper().is_some());
    assert_eq!(solo.leaf_handles().len(), 2);
    let two = ParallelCompositeReader::new(vec![a.clone(), b.clone()]).unwrap();
    assert!(two.reader_cache_helper().is_none());
    // Same leaf count, different leaf sizes.
    let a_leaves = MultiReader::new(vec![
        ReaderHandle::Leaf(seg(&a, 1)),
        ReaderHandle::Leaf(seg(&a, 0)),
    ])
    .unwrap();
    assert!(ParallelCompositeReader::new(vec![a.clone(), Arc::new(a_leaves)]).is_err());
}

#[test]
fn slow_composite_edges() {
    let m = fixture("multi");
    assert!(matches!(
        SlowCompositeCodecReaderWrapper::wrap(vec![]),
        Err(Error::IllegalArgument(_))
    ));
    let one = codec(&m, 0);
    let same = SlowCompositeCodecReaderWrapper::wrap(vec![one.clone()]).unwrap();
    assert!(Arc::ptr_eq(&one, &same));
    let slow =
        SlowCompositeCodecReaderWrapper::wrap(vec![codec(&m, 0), codec(&m, 1), codec(&m, 2)])
            .unwrap();
    assert!(slow.core_cache_helper().is_none() && slow.reader_cache_helper().is_none());
    assert_eq!(slow.leaves().len(), 1);
    let mut rec = Recorder::default();
    assert!(slow.document(15, &mut rec).is_err());
    assert!(slow.term_vectors(-1).is_err());
    assert!(slow.terms("nope").unwrap().is_none());
    assert!(slow.point_values("body").unwrap().is_none());
    assert!(slow.float_vector_values("nope").unwrap().is_none());
    let pv = slow.point_values("pt").unwrap().unwrap();
    assert_eq!(
        pv.estimate_point_count(&mut Recorded::default()).unwrap(),
        15
    );
    let mut seen = Recorded::default();
    pv.intersect(&mut seen).unwrap();
    assert_eq!(
        (seen.0.len(), seen.1),
        (15, 15),
        "grow is forwarded per sub"
    );
    struct Rel(Relation);
    impl IntersectVisitor for Rel {
        fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
            self.0
        }
        fn visit(&mut self, _: i32) {}
        fn visit_with_value(&mut self, _: i32, _: &[u8]) {}
    }
    assert_eq!(
        pv.estimate_point_count(&mut Rel(Relation::CellOutsideQuery))
            .unwrap(),
        0
    );
    assert_eq!(
        pv.estimate_point_count(&mut Rel(Relation::CellCrossesQuery))
            .unwrap(),
        8
    );
    pv.intersect(&mut Rel(Relation::CellOutsideQuery)).unwrap();
    let mut inside = Recorded::default();
    pv.intersect(&mut inside).unwrap();
    inside.0.sort_unstable();
    assert_eq!(inside.0, (0..15).collect::<Vec<_>>());
    let fv = slow.float_vector_values("vec").unwrap().unwrap();
    assert_eq!(fv.dimension(), 3);
    assert!(fv.ord_to_doc(99).is_err() && fv.vector_value(-1).is_err());
    let bv = slow.byte_vector_values("bvec").unwrap().unwrap();
    assert_eq!((bv.dimension(), bv.size()), (2, 8));
    assert_eq!(bv.vector_value(3).unwrap(), [6, 250]);

    // The single-leaf wrapper delegates its cache helpers too.
    let w = SlowCodecReaderWrapper::wrap(seg(&m, 0));
    assert!(w.core_cache_helper().is_some() && w.reader_cache_helper().is_some());
    assert_eq!(w.leaves().len(), 1);
}

#[test]
fn sorting_edges() {
    assert!(DocMap::from_new_to_old(vec![0, 0]).is_err());
    assert!(DocMap::from_new_to_old(vec![0, 5]).is_err());
    assert!(DocMap::from_new_to_old(vec![-1]).is_err());
    let map = DocMap::from_new_to_old(vec![2, 0, 1]).unwrap();
    assert_eq!(
        (map.old_to_new(2), map.new_to_old(0), map.size()),
        (0, 2, 3)
    );

    let m = fixture("multi");
    assert!(matches!(
        SortingCodecReader::wrap(codec(&m, 0), Some(map), vec![]),
        Err(Error::IllegalArgument(_))
    ));
    // Already in order: no map, every read is the inner reader's.
    let ids = vec![IndexSortField {
        field: "rank".into(),
        reverse: false,
        kind: IndexSortKind::Numeric(NumericSortKey::Long(Some(-1))),
    }];
    let s2 = codec(&m, 2); // ranks 2,4,1,3: not in order.
    assert!(sort_doc_map(&*s2, &ids).unwrap().is_some());
    let by_id = vec![IndexSortField {
        field: "nope".into(),
        reverse: false,
        kind: IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(None),
            selector: SortedNumericSelector::Min,
        },
    }];
    let same = SortingCodecReader::wrap_sorted(codec(&m, 2), by_id).unwrap();
    assert!(
        same.doc_map().is_none(),
        "no document has the sort field: already in order"
    );
    assert!(same.live_docs().is_some());
    assert!(same.terms("body").unwrap().is_some());
    assert!(same.numeric_doc_values("rank").unwrap().is_some());
    assert!(same.binary_doc_values("bin").unwrap().is_some());
    assert!(same.sorted_doc_values("kw").unwrap().is_some());
    assert!(same.sorted_numeric_doc_values("nums").unwrap().is_some());
    assert!(same.sorted_set_doc_values("tags").unwrap().is_some());
    assert!(same.norm_values("body").unwrap().is_some());
    assert!(same.point_values("pt").unwrap().is_some());
    assert!(same.float_vector_values("vec").unwrap().is_some());
    assert!(same.byte_vector_values("bvec").unwrap().is_some());
    assert!(same.index_sort().is_some());
    assert!(same.core_cache_helper().is_none() && same.reader_cache_helper().is_none());
    assert_eq!(same.leaves().len(), 1);
    {
        let f = "nope";
        assert!(same.terms(f).unwrap().is_none());
        assert!(same.numeric_doc_values(f).unwrap().is_none());
        assert!(same.binary_doc_values(f).unwrap().is_none());
        assert!(same.sorted_doc_values(f).unwrap().is_none());
        assert!(same.sorted_numeric_doc_values(f).unwrap().is_none());
        assert!(same.sorted_set_doc_values(f).unwrap().is_none());
        assert!(same.norm_values(f).unwrap().is_none());
        assert!(same.point_values(f).unwrap().is_none());
        assert!(same.float_vector_values(f).unwrap().is_none());
        assert!(same.byte_vector_values(f).unwrap().is_none());
    }
    // Selectors and the refused kind.
    let s0 = codec(&m, 0);
    for selector in [
        SortedSetSelector::Min,
        SortedSetSelector::Max,
        SortedSetSelector::MiddleMin,
        SortedSetSelector::MiddleMax,
    ] {
        let sort = vec![IndexSortField {
            field: "tags".into(),
            reverse: false,
            kind: IndexSortKind::SortedSet {
                selector,
                missing: StringMissingValue::Last,
            },
        }];
        let map = sort_doc_map(&*s0, &sort).unwrap().unwrap();
        assert_eq!(map.old_to_new(2), 5, "no tags: missing last ({selector:?})");
    }
    let max = vec![IndexSortField {
        field: "nums".into(),
        reverse: true,
        kind: IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(None),
            selector: SortedNumericSelector::Max,
        },
    }];
    assert!(sort_doc_map(&*s0, &max).unwrap().is_some());
    let binary = vec![IndexSortField {
        field: "bin".into(),
        reverse: false,
        kind: IndexSortKind::Binary(StringMissingValue::None),
    }];
    // Binary: bytes order, missing first, ties by doc id.
    let mut expected: Vec<(Option<Vec<u8>>, i32)> = Vec::new();
    let mut bin = s0.binary_doc_values("bin").unwrap().unwrap();
    for d in 0..s0.max_doc() {
        let v = if bin.advance_exact(d).unwrap() {
            Some(bin.binary_value().to_vec())
        } else {
            None
        };
        expected.push((v, d));
    }
    drop(bin);
    expected.sort();
    let got = sort_doc_map(&*s0, &binary).unwrap();
    let new_to_old: Vec<i32> = (0..s0.max_doc())
        .map(|d| got.as_ref().map_or(d, |m| m.new_to_old(d)))
        .collect();
    assert_eq!(
        new_to_old,
        expected.iter().map(|&(_, d)| d).collect::<Vec<_>>()
    );
    // Absent sort fields sort nothing.
    let absent = vec![IndexSortField::long("nope", false, None)];
    assert!(sort_doc_map(&*s0, &absent).unwrap().is_none());
    for kind in [
        IndexSortKind::String(StringMissingValue::None),
        IndexSortKind::SortedSet {
            selector: SortedSetSelector::Min,
            missing: StringMissingValue::None,
        },
        IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(None),
            selector: SortedNumericSelector::Min,
        },
    ] {
        let sort = vec![IndexSortField {
            field: "nope".into(),
            reverse: false,
            kind,
        }];
        assert!(sort_doc_map(&*s0, &sort).unwrap().is_none());
    }

    // The sorted view's iterators, positioned by hand.
    let v =
        SortingCodecReader::wrap_sorted(s0.clone(), vec![IndexSortField::long("rank", true, None)])
            .unwrap();
    let mut n = v.numeric_doc_values("rank").unwrap().unwrap();
    assert!(n.advance_exact(0).unwrap());
    assert_eq!(n.long_value(), 4);
    assert_eq!(n.advance(3).unwrap(), 3);
    assert!(n.cost() > 0);
    let mut kw = v.sorted_doc_values("kw").unwrap().unwrap();
    assert_eq!(kw.advance(2).unwrap(), 2);
    assert!(kw.advance_exact(3).unwrap());
    assert_eq!(kw.doc_id(), 3);
    assert_eq!(kw.value_count(), 5);
    assert!(kw.cost() > 0);
    let mut ss = v.sorted_set_doc_values("tags").unwrap().unwrap();
    assert_eq!(ss.advance(1).unwrap(), 1);
    assert!(ss.advance_exact(2).unwrap());
    assert_eq!(ss.doc_id(), 2);
    assert!(ss.cost() > 0);
    let mut sn = v.sorted_numeric_doc_values("nums").unwrap().unwrap();
    assert!(!sn.advance_exact(0).unwrap(), "old doc 2 has no nums");
    assert_ne!(sn.next_doc().unwrap(), NO_MORE_DOCS);
    for _ in 0..sn.doc_value_count() {
        sn.next_value().unwrap();
    }
    assert!(sn.next_value().is_err());
    let pv = v.point_values("pt").unwrap().unwrap();
    assert_eq!(
        pv.estimate_point_count(&mut Recorded::default()).unwrap(),
        6
    );
    assert_eq!(pv.num_index_dimensions(), 2);
    let fv = v.float_vector_values("vec").unwrap().unwrap();
    assert_eq!(fv.dimension(), 3);
    assert!(fv.ord_to_doc(9).is_err());
    let bv = v.byte_vector_values("bvec").unwrap().unwrap();
    assert_eq!(bv.dimension(), 2);
    let vt = v.terms("body").unwrap().unwrap();
    let mut te = vt.iterator().unwrap();
    assert!(te.try_seek_exact(b"fox").unwrap());
    assert_eq!(te.try_seek_ceil(b"fp").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.impacts(PostingsFlags::Freqs).unwrap().cost(), 1);
    let t = v.terms("body").unwrap().unwrap();
    let dfa = lucene_codecs::regexp::RegexpPattern::new(b"q.*")
        .unwrap()
        .to_dfa()
        .unwrap();
    assert_eq!(
        terms_of(&mut *t.intersect(&dfa, None).unwrap()),
        ["quick", "quiet", "quilt", "quota"]
    );
    assert_eq!(t.min().unwrap().unwrap(), b"and");
    assert_eq!(t.max().unwrap().unwrap(), b"the");
}

struct Reverse;

impl MergeReaderHooks for Reverse {
    fn reorder(&self, reader: &dyn CodecReader) -> Result<Option<DocMap>> {
        let n = reader.max_doc();
        DocMap::from_new_to_old((0..n).rev().collect()).map(Some)
    }
}

/// Wraps every input so the hook is seen to run.
struct Counting(AtomicU32);

impl MergeReaderHooks for Counting {
    fn wrap_for_merge(&self, reader: Arc<dyn CodecReader>) -> Result<Arc<dyn CodecReader>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(reader)
    }
}

#[test]
fn merge_reader_hooks() {
    let m = fixture("multi");
    let inputs = || vec![codec(&m, 0), codec(&m, 1)];
    let plain = prepare_merge_readers(&DefaultMergeHooks, inputs(), false).unwrap();
    assert_eq!(plain.readers.len(), 2);
    assert!(plain.reorder_doc_maps.is_none());
    let counting = Counting(AtomicU32::new(0));
    let sorted = prepare_merge_readers(&counting, inputs(), true).unwrap();
    assert_eq!(sorted.readers.len(), 2);
    assert_eq!(counting.0.load(Ordering::SeqCst), 2);
    assert!(prepare_merge_readers(&Reverse, vec![], false)
        .unwrap()
        .readers
        .is_empty());

    let reordered = prepare_merge_readers(&Reverse, inputs(), false).unwrap();
    let maps = reordered.reorder_doc_maps.unwrap();
    assert_eq!(maps[0][0], 10);
    assert_eq!(maps[1][4], 0);
    let view = &reordered.readers[0];
    assert_eq!(view.max_doc(), 11);
    assert!(view.index_sort().is_none());
    let mut rec = Recorder::default();
    view.document(0, &mut rec).unwrap();
    assert!(rec.0.contains(&"0=s:d10".to_string()), "{:?}", rec.0);
    let flag = AtomicBool::new(false);
    assert!(!flag.load(Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// The remaining shapes
// ---------------------------------------------------------------------------

#[test]
fn segment_shapes_single_valued_sets_sparse_norms_and_empty_sets() {
    let e = fixture("exitable");
    let s = seg(&e, 0);
    // A single-valued sparse SORTED_SET is stored as SORTED ordinals.
    let mut ss1 = s.sorted_set_doc_values("ss1").unwrap().unwrap();
    assert_eq!(ss1.cost(), 1250);
    assert!(ss1.advance_exact(4).unwrap());
    assert_eq!(ss1.doc_value_count(), 1);
    let ord = ss1.next_ord().unwrap();
    assert_eq!(ss1.lookup_ord(ord).unwrap(), b"t04");
    assert!(!ss1.advance_exact(5).unwrap());
    assert_eq!(ss1.doc_value_count(), 0);
    assert_eq!(ss1.advance(7).unwrap(), 8);
    // Sparse norms.
    let mut n = s.norm_values("txt").unwrap().unwrap();
    assert_eq!(docs(&mut *n).len(), 834);
    let mut n = s.norm_values("txt").unwrap().unwrap();
    assert!(!n.advance_exact(1).unwrap());
    assert!(n.advance_exact(3).unwrap());
    assert!(n.long_value() > 0);
    assert_eq!(n.advance(4).unwrap(), 6);
    assert_eq!(n.advance(2499).unwrap(), 2499);
    assert_eq!(n.advance(2500).unwrap(), NO_MORE_DOCS);

    // An empty set, and a region outside its file.
    let mut c = super::segment::DvCursor::new(super::segment::DocSet::Empty);
    assert_eq!(c.next_doc(), NO_MORE_DOCS);
    assert_eq!(c.advance(3), NO_MORE_DOCS);
    assert!(!c.advance_exact(1));
    assert!(matches!(
        super::segment::DocSet::read(&[0u8; 4], 2, 8, 9, 10),
        Err(Error::Store(_))
    ));
    assert!(matches!(
        super::segment::DocSet::read(&[], -2, 0, 9, 10).unwrap(),
        super::segment::DocSet::Empty
    ));

    // A sparse set's rank index: positions, misses, targets past its end,
    // the current document again, and a step back (outside the contract).
    let sparse = std::sync::Arc::new(super::SparseDocs::new(vec![2, 5, 64, 65, 130], 200).unwrap());
    let mut c = super::segment::DvCursor::new(super::segment::DocSet::Sparse(sparse));
    assert!(!c.advance_exact(1));
    assert!(c.advance_exact(5));
    assert!(c.advance_exact(5), "the current document again");
    assert!(c.advance_exact(65));
    assert_eq!(c.next_doc(), 130);
    assert!(!c.advance_exact(131));
    assert!(!c.advance_exact(9999));
    assert!(!c.advance_exact(-1));
    let sparse = std::sync::Arc::new(super::SparseDocs::new(vec![2, 5, 64, 65, 130], 200).unwrap());
    let mut c = super::segment::DvCursor::new(super::segment::DocSet::Sparse(sparse));
    assert_eq!(c.advance(3), 5);
    assert_eq!(c.advance(5), 64, "never behind the current position");
    assert_eq!(c.advance(66), 130);
    assert_eq!(c.advance(131), NO_MORE_DOCS);

    // Postings of an unpositioned enum.
    let t = s.terms("t").unwrap().unwrap();
    let mut te = t.iterator().unwrap();
    assert!(matches!(
        te.postings(PostingsFlags::Freqs),
        Err(Error::IllegalState(_))
    ));
    assert!(matches!(te.doc_freq(), Err(Error::IllegalState(_))));
    let b = fixture("multi");
    let body = seg(&b, 0);
    let bt = body.terms("body").unwrap().unwrap();
    let mut bte = bt.iterator().unwrap();
    assert!(matches!(
        bte.postings(PostingsFlags::All),
        Err(Error::IllegalState(_))
    ));
    assert!(bte.try_seek_exact(b"fox").unwrap());
    let mut pe = bte.postings(PostingsFlags::None).unwrap();
    pe.next_doc().unwrap();
    assert_eq!(pe.freq(), 1, "no freqs asked for: 1");
    // No term vectors at all.
    let a = fixture("par_a");
    assert!(seg(&a, 0).term_vectors(0).unwrap().is_none());
    assert_eq!(
        IndexReader::total_term_freq(&*b, "body", b"fox").unwrap(),
        11
    );
}

#[test]
fn delegating_wrappers_reach_every_accessor() {
    let m = fixture("multi");
    // `SlowCodecReaderWrapper` over a segment.
    let w = SlowCodecReaderWrapper::wrap(seg(&m, 0));
    assert!(w.binary_doc_values("bin").unwrap().is_some());
    assert!(w.sorted_doc_values("kw").unwrap().is_some());
    assert!(w.sorted_numeric_doc_values("nums").unwrap().is_some());
    assert!(w.sorted_set_doc_values("tags").unwrap().is_some());
    assert!(w.norm_values("body").unwrap().is_some());
    assert!(w.point_values("pt").unwrap().is_some());
    assert!(w.float_vector_values("vec").unwrap().is_some());
    assert!(w.byte_vector_values("bvec").unwrap().is_some());
    assert!(w.index_sort().is_none());
    // A filter over a sized reader.
    let sized: Arc<crate::directory_reader::SegmentReader> =
        Arc::new(m.segment_readers()[0].clone());
    let f = FilterLeafReader::new(sized, Box::new(NoFilter));
    let mut rec = Recorder::default();
    f.document(0, &mut rec).unwrap();
    assert!(!rec.0.is_empty());
    // A composite view of readers without deletions has no live docs.
    let b = fixture("par_b");
    let slow = SlowCompositeCodecReaderWrapper::wrap(vec![codec(&b, 0), codec(&b, 1)]).unwrap();
    assert!(slow.live_docs().is_none());
    assert_eq!(slow.num_docs(), 5);
}

#[test]
fn filtered_terms_default_seek_and_rejects() {
    struct Pass;
    impl TermFilter for Pass {
        fn accept(&mut self, _: &[u8]) -> Result<AcceptStatus> {
            Ok(AcceptStatus::Yes)
        }
    }
    let t = VecTerms(vec!["fo", "foo", "fox", "g"]);
    let mut from = FilteredTermsEnum::new(t.iterator().unwrap(), Pass, true);
    from.set_initial_seek_term(Some(b"foo".to_vec()));
    assert_eq!(terms_of(&mut from), ["foo", "fox", "g"]);
    let dfa = lucene_codecs::regexp::RegexpPattern::new(b"foo|fox")
        .unwrap()
        .to_dfa()
        .unwrap();
    // "fo" is a live prefix the automaton does not accept.
    assert_eq!(
        terms_of(&mut *t.intersect(&dfa, None).unwrap()),
        ["foo", "fox"]
    );
}

/// A leaf of nearly `MAX_DOCS` documents and nothing else.
struct Huge(FieldInfos);

impl IndexReader for Huge {
    fn max_doc(&self) -> i32 {
        i32::MAX - 200
    }
    fn num_docs(&self) -> i32 {
        num_docs_of(self.max_doc(), None)
    }
    fn leaves(&self) -> Vec<LeafReaderContext<'_>> {
        vec![LeafReaderContext {
            reader: self,
            ord: 0,
            doc_base: 0,
        }]
    }
    fn reader_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

impl LeafReader for Huge {
    fn field_infos(&self) -> &FieldInfos {
        &self.0
    }
    fn live_docs(&self) -> Option<&FixedBitSet> {
        None
    }
    fn terms(&self, _: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        Ok(None)
    }
    fn numeric_doc_values(&self, _: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        Ok(None)
    }
    fn binary_doc_values(&self, _: &str) -> Result<Option<Box<dyn BinaryDocValues + '_>>> {
        Ok(None)
    }
    fn sorted_doc_values(&self, _: &str) -> Result<Option<Box<dyn SortedDocValues + '_>>> {
        Ok(None)
    }
    fn sorted_numeric_doc_values(
        &self,
        _: &str,
    ) -> Result<Option<Box<dyn SortedNumericDocValues + '_>>> {
        Ok(None)
    }
    fn sorted_set_doc_values(&self, _: &str) -> Result<Option<Box<dyn SortedSetDocValues + '_>>> {
        Ok(None)
    }
    fn norm_values(&self, _: &str) -> Result<Option<Box<dyn NumericDocValues + '_>>> {
        Ok(None)
    }
    fn point_values(&self, _: &str) -> Result<Option<Box<dyn PointValues + '_>>> {
        Ok(None)
    }
    fn float_vector_values(&self, _: &str) -> Result<Option<Box<dyn FloatVectorValues + '_>>> {
        Ok(None)
    }
    fn byte_vector_values(&self, _: &str) -> Result<Option<Box<dyn ByteVectorValues + '_>>> {
        Ok(None)
    }
    fn document(&self, _: i32, _: &mut dyn StoredFieldVisitor) -> Result<()> {
        Ok(())
    }
    fn term_vectors(&self, _: i32) -> Result<Option<TermVectorsDocument>> {
        Ok(None)
    }
    fn index_sort(&self) -> Option<&[IndexSortField]> {
        None
    }
    fn core_cache_helper(&self) -> Option<&CacheHelper> {
        None
    }
}

#[test]
fn composites_refuse_what_java_refuses() {
    let huge: Arc<dyn LeafReader> = Arc::new(Huge(FieldInfos { fields: vec![] }));
    assert_eq!(huge.num_docs(), i32::MAX - 200);
    assert_eq!(huge.leaves().len(), 1);
    assert!(huge.reader_cache_helper().is_none());
    assert!(matches!(
        MultiReader::new(vec![
            ReaderHandle::Leaf(huge.clone()),
            ReaderHandle::Leaf(huge)
        ]),
        Err(Error::IllegalArgument(_))
    ));
    // Same maxDoc, different leaf counts.
    let m = fixture("multi");
    let a = fixture("par_a");
    let one_leaf = MultiReader::new(vec![ReaderHandle::Leaf(seg(&m, 1))]).unwrap();
    assert!(ParallelCompositeReader::new(vec![a.clone(), Arc::new(one_leaf)]).is_err());
    // MultiTerms keeps the smallest min and largest max whichever leaf holds
    // them.
    let s1 = seg(&m, 1);
    let s0 = seg(&m, 0);
    let t = MultiTerms::new(vec![
        (s1.terms("body").unwrap().unwrap(), 0),
        (s0.terms("body").unwrap().unwrap(), 5),
    ]);
    assert_eq!(t.min().unwrap().unwrap(), b"and");
    assert_eq!(t.max().unwrap().unwrap(), b"zone");
    // A sorted view's own enumerations.
    let v = SortingCodecReader::wrap_sorted(
        codec(&m, 0),
        vec![IndexSortField::long("rank", false, None)],
    )
    .unwrap();
    let vt = v.terms("kw").unwrap().unwrap();
    assert_eq!(terms_of(&mut *vt.iterator().unwrap()).len(), 5);
    let mut n = v.numeric_doc_values("rank").unwrap().unwrap();
    while n.next_doc().unwrap() != NO_MORE_DOCS {}
    assert_eq!(n.next_doc().unwrap(), NO_MORE_DOCS);
    let mut inside = Recorded::default();
    v.point_values("pt")
        .unwrap()
        .unwrap()
        .intersect(&mut inside)
        .unwrap();
    inside.0.sort_unstable();
    assert_eq!(inside.0, [0, 1, 2, 3, 4, 5]);
    assert_eq!(inside.1, 6, "grow is forwarded through the doc map");
    let by_min = vec![IndexSortField {
        field: "nums".into(),
        reverse: false,
        kind: IndexSortKind::SortedNumeric {
            key: NumericSortKey::Long(None),
            selector: SortedNumericSelector::Min,
        },
    }];
    assert!(sort_doc_map(&*codec(&m, 0), &by_min).unwrap().is_some());
    let x = exitable_leaf_reader(seg(&m, 0), Countdown::new(1000));
    let fv = x.float_vector_values("vec").unwrap().unwrap();
    assert_eq!(fv.ord_to_doc(1).unwrap(), 2);
    // Every leaf's values, through one without the field.
    let mixed = MultiReader::new(vec![
        ReaderHandle::Leaf(seg(&a, 0)),
        ReaderHandle::Leaf(seg(&m, 0)),
    ])
    .unwrap();
    let mut rank = multi_doc_values::numeric_values(&mixed, "rank")
        .unwrap()
        .unwrap();
    assert_eq!(rank.next_doc().unwrap(), 3);
    let mut kw = multi_doc_values::sorted_values(&mixed, "kw")
        .unwrap()
        .unwrap();
    assert_eq!(kw.next_doc().unwrap(), 3);
    assert_eq!(kw.doc_id(), 3);
}

/// Walks an iterator the three ways a caller can: `advance`, `advanceExact`
/// and `nextDoc`, over a leaf boundary.
fn walk(it: &mut dyn DocValuesIterator) -> (i32, bool, i32, i64) {
    let a = it.advance(4).unwrap();
    let exact = it.advance_exact(7).unwrap();
    let doc = it.doc_id();
    let next = it.next_doc().unwrap();
    assert!(next > doc);
    (a, exact, next, it.cost())
}

#[test]
fn multi_doc_values_every_kind_positions_alike() {
    let m = fixture("multi");
    let r = &*m;
    let (a, exact, next, cost) =
        walk(&mut *multi_doc_values::norm_values(r, "body").unwrap().unwrap());
    assert_eq!((a, exact, next, cost), (4, true, 8, 15));
    let (a, exact, _, _) = walk(
        &mut *multi_doc_values::numeric_values(r, "rank")
            .unwrap()
            .unwrap(),
    );
    assert_eq!((a, exact), (4, true));
    let (a, exact, next, _) =
        walk(&mut *multi_doc_values::binary_values(r, "bin").unwrap().unwrap());
    assert_eq!((a, exact, next), (4, false, 8));
    let (a, exact, _, _) = walk(
        &mut *multi_doc_values::sorted_numeric_values(r, "nums")
            .unwrap()
            .unwrap(),
    );
    assert_eq!((a, exact), (4, true));
    let (a, exact, _, _) = walk(&mut *multi_doc_values::sorted_values(r, "kw").unwrap().unwrap());
    assert_eq!((a, exact), (4, true));
    let (a, exact, _, _) = walk(
        &mut *multi_doc_values::sorted_set_values(r, "tags")
            .unwrap()
            .unwrap(),
    );
    assert_eq!((a, exact), (4, true));
    // Backwards is refused by every kind.
    for mut it in [
        multi_doc_values::norm_values(r, "body").unwrap().unwrap() as Box<dyn DocValuesIterator>,
        multi_doc_values::numeric_values(r, "rank")
            .unwrap()
            .unwrap(),
        multi_doc_values::binary_values(r, "bin").unwrap().unwrap(),
        multi_doc_values::sorted_numeric_values(r, "nums")
            .unwrap()
            .unwrap(),
        multi_doc_values::sorted_values(r, "kw").unwrap().unwrap(),
        multi_doc_values::sorted_set_values(r, "tags")
            .unwrap()
            .unwrap(),
    ] {
        let d = it.advance(9).unwrap();
        assert!((9..15).contains(&d));
        assert!(it.advance(d).is_err());
        assert!(it.advance_exact(d - 1).is_err());
        assert!(it.advance_exact(15).is_err(), "out of range");
        assert_eq!(it.advance(20).unwrap(), NO_MORE_DOCS);
        assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
    }
}

#[test]
fn check_integrity_and_skippers() {
    let m = fixture("multi");
    for leaf in m.leaves() {
        leaf.reader.check_integrity().unwrap();
        assert!(leaf.reader.doc_values_skipper("nope").unwrap().is_none());
    }
    let s = seg(&m, 0);
    let f = FilterLeafReader::new(s.clone(), Box::new(NoFilter));
    f.check_integrity().unwrap();
    let rank = s
        .doc_values_skipper("rank")
        .unwrap()
        .map(|k| k.global_doc_count());
    assert_eq!(
        f.doc_values_skipper("rank")
            .unwrap()
            .map(|k| k.global_doc_count()),
        rank
    );
    let w = SlowCodecReaderWrapper::wrap(s.clone());
    w.check_integrity().unwrap();
    assert_eq!(
        w.doc_values_skipper("rank")
            .unwrap()
            .map(|k| k.global_doc_count()),
        rank
    );
    let p = ParallelLeafReader::new(vec![s.clone()]).unwrap();
    p.check_integrity().unwrap();
    assert_eq!(
        p.doc_values_skipper("rank")
            .unwrap()
            .map(|k| k.global_doc_count()),
        rank
    );
    assert!(p.doc_values_skipper("nope").unwrap().is_none());
    SlowCompositeCodecReaderWrapper::wrap(vec![codec(&m, 0), codec(&m, 1)])
        .unwrap()
        .check_integrity()
        .unwrap();
    SortingCodecReader::wrap(codec(&m, 0), None, vec![])
        .unwrap()
        .check_integrity()
        .unwrap();
    assert!(Huge(FieldInfos { fields: vec![] })
        .check_integrity()
        .is_ok());

    // A flipped byte in the middle of a stored-fields file.
    let src = format!(
        "{}/../../fixtures/data/reader_api/par_a",
        env!("CARGO_MANIFEST_DIR")
    );
    let tmp = lucene_util::test_support::TempDir::new("reader-check-integrity");
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    let fdt = tmp.path().join("_0.fdt");
    let mut bytes = std::fs::read(&fdt).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&fdt, bytes).unwrap();
    let r = DirectoryReader::open(&FsDirectory::open(tmp.path())).unwrap();
    assert!(r.segment_readers()[0].check_integrity().is_err());
    r.segment_readers()[1].check_integrity().unwrap();
}

/// A sparse field's `IndexedDISI` region, as bytes: each `(block, docs)` a
/// SPARSE block of those low 16 bits, then the `NO_MORE_DOCS` block.
fn sparse_region(blocks: &[(u16, &[u16])]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(block, lows) in blocks {
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&u16::try_from(lows.len() - 1).unwrap().to_le_bytes());
        for low in lows {
            out.extend_from_slice(&low.to_le_bytes());
        }
    }
    out.extend_from_slice(&[0xFF, 0x7F, 0, 0, 0xFF, 0xFF]);
    out
}

#[test]
fn corrupt_sparse_docs_are_refused_before_anything_is_sized_from_them() {
    use super::segment::DocSet;
    let read = |region: &[u8], max_doc: i32| {
        DocSet::read(
            region,
            0,
            region.len() as i64,
            lucene_codecs::indexed_disi::NO_RANK,
            max_doc,
        )
    };
    // Eight bytes naming document 0x7FFF_FFFE in a ten-document segment: sized
    // from the document, this was a ~400 MB bit set cached for the reader's
    // life (or an abort when the allocation failed).
    let huge = sparse_region(&[(0x7FFF, &[0xFFFE])]);
    assert!(matches!(read(&huge, 10), Err(Error::Store(_))));
    // The last legal document, and the first past the segment.
    let mut c = super::segment::DvCursor::new(read(&sparse_region(&[(0, &[3, 9])]), 10).unwrap());
    assert_eq!(
        (c.next_doc(), c.next_doc(), c.next_doc()),
        (3, 9, NO_MORE_DOCS)
    );
    assert!(matches!(
        read(&sparse_region(&[(0, &[3, 10])]), 10),
        Err(Error::Store(_))
    ));
    // Out of order, and repeated, inside a block and across blocks.
    for blocks in [
        &[(0u16, &[5u16, 3][..])][..],
        &[(0, &[5, 5])],
        &[(1, &[0]), (0, &[7])],
        &[(1, &[4]), (1, &[4])],
    ] {
        assert!(
            matches!(read(&sparse_region(blocks), 1 << 20), Err(Error::Store(_))),
            "{blocks:?}"
        );
    }
    // An ALL block repeated: 65,536 documents per four bytes, refused at the
    // second header rather than decoded into a list sized by the file.
    let mut all = Vec::new();
    for _ in 0..64 {
        all.extend_from_slice(&[0, 0, 0xFF, 0xFF]);
    }
    all.extend_from_slice(&[0xFF, 0x7F, 0, 0, 0xFF, 0xFF]);
    assert!(matches!(read(&all, 1 << 20), Err(Error::Store(_))));
    // One ALL block that runs past the segment.
    assert!(matches!(read(&all[60..], 1000), Err(Error::Store(_))));
    // The rank index refuses the same shapes when handed a list directly.
    for (docs, max_doc) in [
        (vec![-1], 10),
        (vec![3, 3], 10),
        (vec![4, 2], 10),
        (vec![10], 10),
    ] {
        assert!(
            matches!(
                super::SparseDocs::new(docs.clone(), max_doc),
                Err(Error::Store(_))
            ),
            "{docs:?}"
        );
    }
}
