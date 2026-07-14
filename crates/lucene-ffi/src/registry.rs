//! Process-wide handle registries. A JNI caller has no way to hand this
//! crate a Rust reference across calls (see the `ffi-safety` skill), so
//! every opened `Directory`/segment/result set lives in one of these global
//! [`SlotMap`]s, guarded by a `Mutex` (JNI callers may call from more than
//! one JVM thread) behind a `u64` handle the caller carries between calls.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::doc_values::DocValuesMeta;
use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::norms::Norms;
use lucene_search::ScoreDoc;
use lucene_store::directory::FsDirectory;

use crate::handle::{RegistryTag, SlotMap};

/// Locks `mutex`, recovering the inner value if a previous holder of this
/// same lock panicked instead of propagating the poison as a second panic.
///
/// **Why this is sound for these three registries specifically**: every
/// mutation [`SlotMap`] performs (`insert`/`remove`) is a single, non-panicking
/// sequence of `Vec`/field writes with no possibility of observing a torn
/// write from within this crate (`insert`/`remove` never call into
/// arbitrary/foreign code that could panic mid-mutation -- see `handle.rs`).
/// A panic that poisons one of these mutexes therefore always happens while
/// the guard is held read-only (e.g. mid-query, borrowing a `&SegmentHandle`
/// while decoding adversarial bytes) or entirely outside any `SlotMap`
/// method body, never mid-`insert`/mid-`remove` -- so the slotmap itself is
/// never left in a half-written state, only "the operation using its
/// borrowed contents failed." Recovering it is safe: every subsequent access
/// still goes through the same generation-tag (and, since the handle-tag
/// fix, registry-tag) validation, so a wrong/stale/cross-registry handle is
/// still rejected as before -- recovery only prevents *this* mutex from
/// wedging every future call into a permanent [`crate::error::FfiStatus::Panic`]
/// (defeating `catch_unwind`'s purpose of isolating one bad call), it does
/// not weaken any handle-validation guarantee.
pub(crate) fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One opened segment's decoded term dictionary plus the raw bytes of
/// whichever postings files were opened alongside it. `postings::DocInput`/
/// `PosInput`/`PayInput` all borrow from a byte slice (see
/// `lucene-codecs/src/postings.rs`), so this struct owns the bytes and a
/// fresh `DocInput::open(&self.doc, ...)` etc. is (cheaply -- header/footer
/// checks only) reconstructed per query call rather than stored as a
/// self-referential field.
pub struct SegmentHandle {
    pub fields: BlockTreeFields,
    pub doc_bytes: Option<Vec<u8>>,
    pub pos_bytes: Option<Vec<u8>>,
    // `.pay` (payload) support is deferred -- see `lib.rs`'s module doc --
    // so there is no `pay_bytes` field yet; `search_phrase_query` is always
    // called with `pay_in: None` in `query.rs`.
    pub segment_id: [u8; 16],
    pub segment_suffix: String,
    pub max_doc: i32,
    /// This segment's parsed `.fnm` (task #30) -- kept around (rather than
    /// dropped once `blocktree::open` has consumed it) so a scored query can
    /// map a field *name* (the only thing a caller passes over the C ABI) to
    /// the field *number* `norms`/`NormsEntry` are keyed by. `BlockTreeFields`
    /// has no such name->number mapping of its own (see `blocktree.rs`), so
    /// this is the only place left to look it up from.
    pub field_infos: FieldInfos,
    /// This segment's whole `.nvd` (norms data) file, opened by
    /// `ffi_open_segment`'s optional `nvd_name`/`nvm_name` parameters (task
    /// #30) -- `None` when the caller opened the segment without norms
    /// (every scored query then falls back to
    /// `lucene_search::similarity::UNNORMED_FIELD_LENGTH`, the same
    /// documented approximation `FieldNorms`'s absence already means
    /// elsewhere in this port).
    pub norms_data: Option<Vec<u8>>,
    /// This segment's parsed `.nvm` (norms metadata) -- one [`Norms`] entry
    /// per field that has norms, looked up by field number via
    /// `field_infos` above. `Some` iff `norms_data` is `Some` (both come from
    /// the same `nvd_name`/`nvm_name is-null` check in `ffi_open_segment`).
    pub norms: Option<Norms>,
    /// This segment's whole `.dvd` (doc-values data) file, opened by
    /// `ffi_open_segment`'s optional `dvm_name`/`dvd_name`/`dv_suffix`
    /// parameters (task #40) -- `None` when the caller opened the segment
    /// without doc values, in which case `ffi_sort_by_doc_value`/
    /// `ffi_sort_by_multi_valued_doc_value` return [`crate::error::FfiStatus::InvalidArgument`]
    /// (there is nothing to sort by, unlike norms' "fall back to a constant"
    /// story -- a sort with no values for its field has no sensible
    /// fallback).
    pub dv_data: Option<Vec<u8>>,
    /// This segment's parsed `.dvm` (doc-values metadata) -- one entry per
    /// doc-values field, looked up by field number via `field_infos` above,
    /// same pattern as `norms`. `Some` iff `dv_data` is `Some`.
    pub dv_meta: Option<DocValuesMeta>,
}

/// A completed unscored query's collected, ascending, live doc IDs -- read
/// back via `ffi_results_len`/`ffi_results_copy`, then released via
/// `ffi_close_results`.
pub struct ResultsHandle {
    pub docs: Vec<i32>,
}

/// A completed *scored* query's `(doc_id, score)` hits, kept in `TopDocsCollector`
/// order (best-first, ties broken by lower doc ID -- see `collector.rs`'s
/// `rank_order`) -- read back via `ffi_scored_results_len`/`ffi_scored_results_copy`,
/// then released via `ffi_close_scored_results`.
///
/// **Why a new registry/handle type instead of widening `ResultsHandle` with an
/// optional `Vec<f32>`**: `ResultsHandle` is a public, already-shipped shape read by
/// `ffi_results_len`/`ffi_results_copy`'s existing (unscored) contract -- adding an
/// optional scores field there would force every existing caller of the unscored
/// path to reason about a field that's always `None` for them, and would let a
/// caller accidentally call `ffi_results_copy` against a handle that was actually
/// populated by a scored query (or vice versa) since both would share one handle
/// type and one registry tag. A separate `ScoredResultsHandle`/`RegistryTag::
/// ScoredResults` keeps the two result shapes as distinct as the two collector
/// traits they come from (`Collector` vs `ScoringCollector`, see `collector.rs`'s
/// module doc for that same non-breaking-addition reasoning) -- a results handle
/// from the wrong search flavor is rejected by the registry-tag check before it
/// can be misread, exactly like a directory handle passed to a segment call is.
pub struct ScoredResultsHandle {
    pub hits: Vec<ScoreDoc>,
}

/// A completed doc-value sort's `(doc_id, value)` pairs (task #40, wrapping
/// `lucene_search::sort_by_numeric_doc_value`/`sort_by_multi_valued_doc_value`)
/// -- ascending by value, ties broken by ascending doc ID (see those
/// functions' own doc comments) -- read back via
/// `ffi_sorted_results_len`/`ffi_sorted_results_copy`, then released via
/// `ffi_close_sorted_results`.
///
/// **Why a new registry/handle type instead of reusing `ScoredResultsHandle`**:
/// a sort result's second element is the actual doc-value used for
/// ordering (an arbitrary `i64` -- a raw NUMERIC value, or a SORTED_NUMERIC/
/// SORTED_SET reduced value/ordinal), not a BM25 `f32` score -- a different
/// wire type (`i64` vs `f32`), a different scale/meaning a caller must not
/// confuse with a relevance score, and a different collector-less code path
/// (a plain sort over an already-known candidate set, not a
/// `TopDocsCollector` scored search, see `lucene-search`'s
/// `doc_value_query.rs` module doc for that design rationale). Keeping this
/// as its own registry/tag means a scored-results handle can never be
/// accidentally passed to `ffi_sorted_results_copy` (or vice versa) and
/// misread as the wrong element type -- exactly the same reasoning
/// `ScoredResultsHandle`'s own doc comment gives for not widening
/// `ResultsHandle`.
pub struct SortedResultsHandle {
    pub pairs: Vec<(i32, i64)>,
}

/// An opened `DirectoryReader` (task #51, wrapping task #45's
/// `lucene_search::directory_reader::DirectoryReader`): every segment listed
/// in a commit's `segments_N`, already opened with each segment's `doc_base`
/// computed -- read back via [`crate::directory_reader::ffi_search_term_query_multi_segment`]/
/// [`crate::directory_reader::ffi_search_boolean_query_multi_segment`] (task #41's
/// multi-segment fan-out/merge), released via
/// [`crate::directory_reader::ffi_close_directory_reader`].
///
/// **Why its own registry/tag instead of reusing `Directory`**: a
/// `DirectoryReader` owns a whole tree of already-opened, already-decoded
/// segment readers (term dictionaries, postings byte buffers, live docs) --
/// a fundamentally different lifetime and size class from `Directory`'s bare
/// `FsDirectory` (a filesystem root with no segment state at all) or
/// `Segment`'s single already-opened segment. Folding it into either would
/// let a directory/segment handle be silently accepted where a
/// `DirectoryReader` handle is expected (or vice versa) since they'd share
/// one registry tag -- exactly the cross-registry confusion `RegistryTag`
/// exists to rule out (see `handle.rs`'s module doc).
///
/// **No norms/doc-values plumbing**: task #45's `DirectoryReader` has no
/// `.nvm`/`.nvd`/`.dvm`/`.dvd` support at all (see that module's doc
/// comment) -- every multi-segment scored query built on this handle always
/// passes `norms: None` per segment, the same documented
/// `UNNORMED_FIELD_LENGTH` fallback `lucene_search`'s own scored functions
/// already use for a bare `None`, not a new gap introduced by this handle.
pub struct DirectoryReaderHandle {
    pub reader: lucene_search::directory_reader::DirectoryReader,
}

/// A completed SortedSet facet count's resolved `(ord, label, count)` triples
/// (Faceted search FFI exposure, wrapping `lucene_search::facets::facet_counts`/
/// `resolve_labels`/`top_n_facets`) -- read back via
/// `results_facets.rs`'s `ffi_facet_results_len`/`ffi_facet_results_copy`/
/// `ffi_facet_result_label`, then released via `ffi_close_facet_results`.
///
/// **Why a new registry/handle type instead of reusing `SortedResultsHandle`**:
/// a facet result's element is `(ord, label, count)` -- it carries a resolved
/// string label alongside a `u64` count, not a `(doc_id, value)` `i64` pair --
/// a different wire shape needing its own string-accessor
/// (`ffi_facet_result_label`) that a sorted-results handle has no equivalent
/// of. Keeping this as its own registry/tag means a sorted-results handle can
/// never be accidentally passed to a facet-results accessor (or vice versa)
/// and misread as the wrong element type -- exactly the same reasoning
/// `SortedResultsHandle`'s own doc comment gives for not widening
/// `ScoredResultsHandle`.
///
/// **NUMERIC range facet counts have no equivalent handle**: `ffi_range_facet_counts`
/// (also in `facets.rs`) writes counts directly into a caller-allocated
/// buffer instead -- every range's label is caller-supplied input, not
/// resolved from the index, so the caller already owns every label string
/// and there is nothing new to hand back behind a handle. See `facets.rs`'s
/// module doc for the full rationale.
pub struct FacetResultsHandle {
    pub facets: Vec<lucene_search::facets::FacetCount>,
}

/// A completed highlight fragment assembly's [`lucene_search::highlighter::Fragment`]s
/// (`highlighter.rs`'s `ffi_assemble_fragments`, wrapping
/// `lucene_search::highlighter::assemble_fragments`) -- read back via
/// `results_fragments.rs`'s `ffi_fragment_results_len`/`ffi_fragment_result_text`/
/// `ffi_fragment_result_matched_terms_len`/`ffi_fragment_result_matched_term`,
/// then released via `ffi_close_fragment_results`.
///
/// **Why a new registry/handle type instead of reusing `FacetResultsHandle`**:
/// a fragment carries a highlighted `text` string *and* a variable-length list
/// of `matched_terms` strings per element -- a two-level variable-length shape
/// none of this crate's existing handles have (`FacetResultsHandle`'s element
/// has exactly one string field, `label`). Keeping this as its own
/// registry/tag means a facet- or sorted-results handle can never be
/// accidentally passed to a fragment accessor (or vice versa) and misread --
/// the same reasoning every other handle type in this file already gives for
/// not widening an existing one.
pub struct FragmentResultsHandle {
    pub fragments: Vec<lucene_search::highlighter::Fragment>,
}

/// One node of a flattened `lucene_search::explain::Explanation` tree (Query
/// explain FFI exposure, `explain.rs`'s `ffi_explain_term_query`/
/// `ffi_explain_phrase_query`/`ffi_explain_boolean_query`, wrapping
/// `lucene_search::explain::explain_clause`) -- see `explain.rs`'s module doc
/// for the full flattening scheme (depth-first pre-order, root always index
/// `0`, each node's `children` a list of *indices into the same flat `Vec`*
/// rather than nested owned `Explanation`s).
pub struct ExplainNode {
    pub matched: bool,
    pub value: f32,
    pub description: String,
    pub children: Vec<usize>,
}

/// A completed explain call's flattened [`ExplainNode`] tree -- read back via
/// `results_explain.rs`'s `ffi_explain_node_value`/`ffi_explain_node_matched`/
/// `ffi_explain_node_description`/`ffi_explain_node_child_count`/
/// `ffi_explain_node_child_at`, then released via `ffi_close_explain_results`.
///
/// **Why a new registry/handle type instead of reusing `FragmentResultsHandle`**:
/// an explain node's shape (a `bool` + an `f32` + a `String` + a list of
/// *sibling node indices*, forming a recursive tree) has no correspondence to
/// a fragment's flat `text` + `matched_terms` list -- see this crate's other
/// handle doc comments for why each genuinely distinct result shape gets its
/// own registry/tag rather than widening an existing one (a fragment-results
/// handle must never be accidentally accepted by an explain accessor, or vice
/// versa).
pub struct ExplainResultsHandle {
    pub nodes: Vec<ExplainNode>,
}

pub fn directories() -> &'static Mutex<SlotMap<FsDirectory>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<FsDirectory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Directory)))
}

pub fn segments() -> &'static Mutex<SlotMap<SegmentHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<SegmentHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Segment)))
}

pub fn results() -> &'static Mutex<SlotMap<ResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<ResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::Results)))
}

pub fn scored_results() -> &'static Mutex<SlotMap<ScoredResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<ScoredResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::ScoredResults)))
}

pub fn sorted_results() -> &'static Mutex<SlotMap<SortedResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<SortedResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::SortedResults)))
}

pub fn directory_readers() -> &'static Mutex<SlotMap<DirectoryReaderHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<DirectoryReaderHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::DirectoryReader)))
}

pub fn facet_results() -> &'static Mutex<SlotMap<FacetResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<FacetResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::FacetResults)))
}

pub fn fragment_results() -> &'static Mutex<SlotMap<FragmentResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<FragmentResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::FragmentResults)))
}

pub fn explain_results() -> &'static Mutex<SlotMap<ExplainResultsHandle>> {
    static REGISTRY: OnceLock<Mutex<SlotMap<ExplainResultsHandle>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(SlotMap::new(RegistryTag::ExplainResults)))
}
