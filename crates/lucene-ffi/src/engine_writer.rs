//! The writer the OpenSearch plugin's `RustEngine` indexes through: one
//! handle per shard, wrapping a [`ConcurrentIndexWriter`] in
//! *explicit-documents* mode.
//!
//! ## Why explicit documents
//!
//! OpenSearch builds each Lucene document in the JVM: its mappers decide which
//! fields are stored, indexed, doc-valued or pointed, and its analyzers and
//! similarities decide the terms and the norms. Re-implementing any of that in
//! Rust would make the Rust engine index differently from the Java one. So the
//! JVM inverts each document itself -- terms, frequencies, positions,
//! offsets, the norm its `Similarity` computed -- and hands the result over
//! as one byte blob per operation ([`decode_op`] documents the layout). The
//! writer only has to lay those bytes out in Lucene's formats, which is the
//! part this port already does byte-for-byte.
//!
//! ## Indexing threads
//!
//! Handles live in their own registry as `Arc<EngineWriter>`; a call takes
//! the registry's shared guard only long enough to clone the `Arc`. Adds run
//! on the caller's thread without a shard-wide lock: each indexing thread
//! buffers into one of the writer's per-thread buffers and builds the
//! segment when its buffer fills, as Lucene's `DocumentsWriterPerThread`s
//! do. Commits, forced merges and commit points take the handle's maintenance
//! lock, one at a time.
//!
//! ## Failures
//!
//! A document the writer refuses (a field it cannot index, a malformed blob)
//! is [`FfiStatus::InvalidArgument`] and changes nothing -- Java's
//! document-level failure. Any other failure while writing leaves the writer
//! in a state nothing vouches for, so it is recorded as the handle's *tragic*
//! error (`IndexWriter.getTragicException()`): every later call fails with it,
//! and the engine fails the shard. A panic inside a call is recorded the same
//! way. Neither ever takes down another shard.
//!
//! ## Commits and merges
//!
//! A commit does not stop indexing. It flushes every buffer holding
//! operations from before it, and only then asks the JVM for its user data
//! (OpenSearch's local checkpoint and maximum sequence number), as Lucene's
//! `IndexWriter.commit` iterates its live commit data after the flush; other
//! threads go on indexing into the emptied buffers meanwhile. Merges run on
//! the handle's `ConcurrentMergeScheduler` threads over committed segments,
//! and a merged segment is in the next commit -- so every `segments_N` this
//! writer leaves is a consistent recovery point.
//! Commit points are never dropped by the writer itself
//! ([`DeletionPolicy::KeepAll`]): OpenSearch's `CombinedDeletionPolicy` runs
//! in the JVM and names the ones to drop ([`ffi_engine_writer_delete_commits`]).

use std::collections::HashMap;
use std::os::raw::c_char;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_file_deleter::DeletionPolicy;
use lucene_index::index_writer::{
    self, DocumentVector, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField,
    InvertedTerms, MergeEvent, MergeListener, MergePolicyConfig, MergeTimes, MergedSegmentWarmer,
    SegmentPartitioning, SoftDeletesRetention,
};
use lucene_index::merge_policy::log::LogByteSizeMergePolicy;
use lucene_index::merge_policy::{
    CompoundFileSettings, MergeOnFlushMergePolicy, MergePolicy, NoMergePolicy, TieredMergePolicy,
};
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_index::segment_info::LuceneVersion;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_store::directory::Directory;
use lucene_util::quantization::ScalarEncoding;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{engine_writers, lock_recovering, read_recovering, WriterHandle};

/// The Lucene version every segment this writer produces claims.
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 2,
};

/// The doc-values field OpenSearch keeps each document's sequence number in
/// (`SeqNoFieldMapper.NAME`), which retention is decided by.
const SEQ_NO_FIELD: &str = "_seq_no";
/// The field OpenSearch keeps each document's id in (`IdFieldMapper.NAME`),
/// whose postings a retention merge prunes (`PrunePostingsMergePolicy`).
const ID_FIELD: &str = "_id";
/// The stored copy of a filtered source OpenSearch keeps for peer recovery
/// (`SourceFieldMapper.RECOVERY_SOURCE_NAME`), which a retention merge prunes
/// (`RecoverySourcePruneMergePolicy`).
const RECOVERY_SOURCE_FIELD: &str = "_recovery_source";

/// Operation kinds of [`decode_op`].
const OP_ADD: u8 = 0;
const OP_SOFT_UPDATE: u8 = 1;
/// [`OP_ADD`] and [`OP_SOFT_UPDATE`] whose documents belong to a partition
/// (the value of the index's partition field): the kind is followed by the
/// partition (`i32 len, utf8`), then the operation as without one.
const OP_ADD_PARTITIONED: u8 = 2;
const OP_SOFT_UPDATE_PARTITIONED: u8 = 3;
/// Panics inside the writer -- the fault the "a panic fails exactly one
/// shard" test injects. Refused unless the handle was opened with fault
/// injection enabled.
const OP_PANIC: u8 = 0xFF;

/// Indices into [`ffi_engine_writer_stats`]' output.
pub const STAT_RAM_BYTES: usize = 0;
pub const STAT_FLUSHING_BYTES: usize = 1;
pub const STAT_PENDING_DOCS: usize = 2;
pub const STAT_UNCOMMITTED: usize = 3;
pub const STAT_GENERATION: usize = 4;
pub const STAT_SEGMENTS: usize = 5;
pub const STAT_COUNT: usize = 6;

/// One shard's writer and the state the JVM side drives it with.
pub struct EngineWriter {
    handle: WriterHandle,
    /// Held for the whole of a commit, merge or commit-point change, so they
    /// run one at a time; adds do not take it.
    maintenance: Mutex<Holds>,
    /// `IndexWriter.getTragicException()`: once set, every call fails.
    tragic: OnceLock<String>,
    fault_injection: bool,
    /// Which merge policies wrap the shard's `TieredMergePolicy`.
    policies: Mutex<MergePolicies>,
    /// Whether the shard's codec is the sparse one, which BP reordering
    /// does not support.
    sparse: bool,
    /// The buffers' segments live views read ([`crate::live`]).
    live_chunks: Mutex<crate::live::Chunks>,
    /// Writes those segments ahead of the searches ([`crate::live`]).
    live_warmer: crate::live::Warmer,
}

/// The policies [`install_merge_policies`] builds around the writer's
/// [`MergePolicyConfig`].
#[derive(Debug, Clone, Default)]
struct MergePolicies {
    /// `index.merge_on_flush.policy: merge-on-flush`.
    merge_on_flush: bool,
    /// `BPReorderingMergePolicy`'s settings, when reordering is on.
    bp: Option<BpSettings>,
    /// The shard's policy when it is not the tiered one
    /// ([`ffi_engine_writer_set_pluggable_merge_policy`]).
    other: Option<PluggablePolicy>,
}

/// `BPReorderingMergePolicy` with its `BPIndexReorderer`.
#[derive(Debug, Clone)]
struct BpSettings {
    reorderer: BpIndexReorderer,
    min_merge_docs: i32,
    min_merge_ratio: f32,
}

/// Rebuilds the policies `policies` names over the writer's current
/// [`MergePolicyConfig`]: merge-on-flush's `MergeOnFlushMergePolicy`, and
/// `BPReorderingMergePolicy` deciding every other merge, each around the
/// shard's own policy (tiered, or [`PluggablePolicy`]). With BP on, a
/// commit or refresh still picks its merges with the plain tiered policy:
/// the segments it merges are small, and its merges then run on the merge
/// threads instead of the refreshing one. A writer with merges off keeps
/// them off.
fn install_merge_policies(iw: &mut IndexWriter<'_>, policies: &MergePolicies) {
    let merges_on = iw.merge_policy().is_some();
    let config = iw.merge_policy().cloned().unwrap_or_default();
    let compound = iw.use_compound_file();
    let merged_compound = iw.merge_compound_file_settings();
    let tiered = || {
        let mut t = TieredMergePolicy::new(config.clone());
        if let Some(settings) = merged_compound {
            // `index.compound_format`, as the writer's own merges use it.
            *t.compound_file_settings_mut() = settings;
        } else if !compound {
            // Merged segments stay loose files, as without a policy.
            let _ = t.compound_file_settings_mut().set_no_cfs_ratio(0.0);
        }
        t
    };
    // The shard's own policy: tiered, or `log_byte_size`/no merges.
    let base = || -> Box<dyn MergePolicy> {
        match &policies.other {
            Some(other) => other.build(),
            None => Box::new(tiered()),
        }
    };
    let full_flush: Option<Arc<dyn MergePolicy>> = if policies.merge_on_flush {
        Some(Arc::new(MergeOnFlushMergePolicy::new(base())))
    } else if merges_on && policies.bp.is_some() {
        Some(Arc::from(base()))
    } else {
        None
    };
    iw.set_full_flush_merge_policy(full_flush);
    let pluggable = match policies.bp.as_ref().filter(|_| merges_on) {
        Some(bp) => {
            let mut policy = BpReorderingMergePolicy::new(base(), bp.reorderer.clone());
            // Both checked when the settings came in.
            let _ = policy.set_min_natural_merge_num_docs(bp.min_merge_docs);
            let _ = policy.set_min_natural_merge_ratio_from_biggest_segment(bp.min_merge_ratio);
            Some(Arc::new(policy) as Arc<dyn MergePolicy>)
        }
        None => policies
            .other
            .as_ref()
            .map(|other| Arc::from(other.build())),
    };
    iw.set_pluggable_merge_policy(pluggable);
}

impl EngineWriter {
    fn policies(&self) -> MutexGuard<'_, MergePolicies> {
        self.policies.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A merge policy other than `TieredMergePolicy`, as OpenSearch configures it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PluggablePolicy {
    /// `LogByteSizeMergePolicy` (`index.merge.policy: log_byte_size`).
    LogByteSize {
        merge_factor: i32,
        min_merge_bytes: i64,
        max_merge_bytes: i64,
        max_forced_merge_bytes: i64,
        max_merge_docs: i32,
        compound: CompoundFileSettings,
    },
    /// `NoMergePolicy` (`index.merge.enabled: false`).
    NoMerges,
}

impl PluggablePolicy {
    fn build(&self) -> Box<dyn MergePolicy> {
        match *self {
            PluggablePolicy::LogByteSize {
                merge_factor,
                min_merge_bytes,
                max_merge_bytes,
                max_forced_merge_bytes,
                max_merge_docs,
                compound,
            } => {
                let mut p = LogByteSizeMergePolicy::new();
                // Checked by the caller: at least 2.
                let _ = p.set_merge_factor(merge_factor);
                p.min_merge_size = min_merge_bytes;
                p.max_merge_size = max_merge_bytes;
                p.max_merge_size_for_forced_merge = max_forced_merge_bytes;
                p.max_merge_docs = max_merge_docs;
                *p.compound_file_settings_mut() = compound;
                Box::new(p)
            }
            PluggablePolicy::NoMerges => Box::new(NoMergePolicy::default()),
        }
    }
}

/// Files pinned by [`ffi_engine_writer_hold_commit`] and
/// [`ffi_engine_writer_nrt_snapshot`], by hold id (one id space).
#[derive(Default)]
struct Holds {
    files: HashMap<u64, Vec<String>>,
    /// Near-real-time snapshots' pins: releasing the last reference to one
    /// releases its files at once.
    nrt: HashMap<u64, Arc<lucene_index::nrt::NrtFileHold>>,
    next: u64,
}

impl EngineWriter {
    pub(crate) fn writer(&self) -> &ConcurrentIndexWriter<'static> {
        &self.handle.writer
    }

    pub(crate) fn live_chunks(&self) -> MutexGuard<'_, crate::live::Chunks> {
        self.live_chunks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn live_warmer(&self) -> &crate::live::Warmer {
        &self.live_warmer
    }

    fn maintenance(&self) -> MutexGuard<'_, Holds> {
        self.maintenance
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The generation of the newest commit written or attempted.
    fn generation(&self) -> i64 {
        self.writer().with_writer(|w| w.segment_infos().generation)
    }
}

/// A little-endian reader over a caller's blob; running off the end is
/// [`FfiStatus::Decode`], never a panic.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn decode_error(what: &str) -> FfiStatus {
    set_last_error(format!("engine writer: malformed blob: {what}"));
    FfiStatus::Decode
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], FfiStatus> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| decode_error(what))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self, what: &str) -> Result<u8, FfiStatus> {
        Ok(self.take(1, what)?[0])
    }

    fn i32(&mut self, what: &str) -> Result<i32, FfiStatus> {
        let b = self.take(4, what)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i64(&mut self, what: &str) -> Result<i64, FfiStatus> {
        let b: [u8; 8] = self
            .take(8, what)?
            .try_into()
            .map_err(|_| decode_error(what))?;
        Ok(i64::from_le_bytes(b))
    }

    /// A count, refused when negative or when even `min_bytes` per element
    /// could not fit in what is left -- which bounds every allocation sized
    /// by a count to the blob's own length.
    fn count(&mut self, min_bytes: usize, what: &str) -> Result<usize, FfiStatus> {
        let n = usize::try_from(self.i32(what)?).map_err(|_| decode_error(what))?;
        if n.saturating_mul(min_bytes.max(1)) > self.remaining() {
            return Err(decode_error(what));
        }
        Ok(n)
    }

    fn bytes(&mut self, what: &str) -> Result<&'a [u8], FfiStatus> {
        let n = self.count(1, what)?;
        self.take(n, what)
    }

    fn string(&mut self, what: &str) -> Result<String, FfiStatus> {
        let b = self.bytes(what)?;
        std::str::from_utf8(b)
            .map(str::to_string)
            .map_err(|_| decode_error(what))
    }

    fn finish(&self, what: &str) -> Result<(), FfiStatus> {
        if self.remaining() != 0 {
            return Err(decode_error(what));
        }
        Ok(())
    }
}

fn stored_value(c: &mut Cursor<'_>) -> Result<FieldValue, FfiStatus> {
    let kind = c.u8("stored kind")?;
    let bytes = c.bytes("stored value")?;
    let fixed = |n: usize| -> Result<&[u8], FfiStatus> {
        if bytes.len() == n {
            Ok(bytes)
        } else {
            Err(decode_error("stored value width"))
        }
    };
    Ok(match kind {
        0 => FieldValue::String(
            std::str::from_utf8(bytes)
                .map_err(|_| decode_error("stored string"))?
                .to_string(),
        ),
        1 => FieldValue::Binary(bytes.to_vec()),
        2 => FieldValue::Int(i32::from_le_bytes(
            fixed(4)?.try_into().map_err(|_| decode_error("int"))?,
        )),
        3 => FieldValue::Long(i64::from_le_bytes(
            fixed(8)?.try_into().map_err(|_| decode_error("long"))?,
        )),
        4 => FieldValue::Float(f32::from_le_bytes(
            fixed(4)?.try_into().map_err(|_| decode_error("float"))?,
        )),
        5 => FieldValue::Double(f64::from_le_bytes(
            fixed(8)?.try_into().map_err(|_| decode_error("double"))?,
        )),
        _ => return Err(decode_error("stored kind")),
    })
}

fn inverted_field(c: &mut Cursor<'_>) -> Result<InvertedField, FfiStatus> {
    let field_number = c.i32("inverted field number")?;
    let norm = match c.u8("norm flag")? {
        0 => None,
        1 => Some(c.i64("norm")?),
        2 => return analyzed_field(c, field_number),
        _ => return Err(decode_error("norm flag")),
    };
    // A term is at least its length, its freq and its flags.
    let n = c.count(9, "term count")?;
    let mut terms = InvertedTerms::new();
    for _ in 0..n {
        let term = c.bytes("term")?;
        let freq = c.i32("freq")?;
        let flags = c.u8("term flags")?;
        if flags & !7 != 0 {
            return Err(decode_error("term flags"));
        }
        let occurrences = usize::try_from(freq).map_err(|_| decode_error("freq"))?;
        // Each occurrence's position, offsets and payload length are at
        // least 4 bytes apiece.
        let per = [1u8, 2, 4]
            .iter()
            .map(|&bit| if flags & bit != 0 { 4usize } else { 0 })
            .sum::<usize>();
        if occurrences.saturating_mul(per) > c.remaining() {
            return Err(decode_error("occurrences"));
        }
        let mut t = terms.start(term);
        t.freq(freq);
        if flags & 1 != 0 {
            for _ in 0..occurrences {
                t.position(c.i32("position")?);
            }
        }
        if flags & 2 != 0 {
            for _ in 0..occurrences {
                t.offsets(c.i32("start offset")?, c.i32("end offset")?);
            }
        }
        if flags & 4 != 0 {
            for _ in 0..occurrences {
                t.payload(c.bytes("payload")?);
            }
        }
    }
    Ok(InvertedField {
        field_number,
        terms,
        norm,
    })
}

/// An inverted field the JVM left to Rust to analyze ([`crate::jvm_invert`]):
/// `i32 analyzer` (a [`crate::jvm_analysis::register`] id), `u8 index
/// options` (ordinal), `u8 norm` ([`crate::jvm_invert::Norm`]), `i32 len,
/// name`, `i32 instances`, then each instance's text (`i32 len`, UTF-8).
fn analyzed_field(c: &mut Cursor<'_>, field_number: i32) -> Result<InvertedField, FfiStatus> {
    use crate::jvm_invert::{invert, Norm, Options};
    let registered = crate::jvm_analysis::registered(c.i32("analyzer")?)?;
    let options = Options::from_ordinal(c.u8("index options")?)
        .ok_or_else(|| decode_error("index options"))?;
    let norm = Norm::from_byte(c.u8("norm mode")?).ok_or_else(|| decode_error("norm mode"))?;
    let name = c.string("field name")?;
    let n = c.count(4, "instances")?;
    let mut texts = try_with_capacity(n)?;
    for _ in 0..n {
        texts.push(
            std::str::from_utf8(c.bytes("instance text")?)
                .map_err(|_| decode_error("instance text"))?,
        );
    }
    invert(field_number, &name, &registered, options, norm, &texts)
}

/// One document: its stored values, inverted fields, doc values and points,
/// each a count followed by that many entries.
///
/// - stored: `i32 field, u8 kind (0 string, 1 binary, 2 int, 3 long,
///   4 float, 5 double), i32 len, bytes`
/// - inverted: `i32 field, u8 has_norm, [i64 norm], i32 terms`, then per term
///   `i32 len, bytes, i32 freq, u8 flags (1 positions, 2 offsets,
///   4 payloads), [freq x i32 position], [freq x (i32 start, i32 end)],
///   [freq x (i32 len, bytes) payload]` -- a zero-length payload is an
///   occurrence without one; or `i32 field, u8 2` and a field to analyze
///   here ([`analyzed_field`])
/// - doc values: `i32 field, u8 kind (0 long, 1 bytes), i64 | (i32 len, bytes)`
/// - points: `i32 field, i32 len, packed bytes`
///
/// A document's KNN vectors follow the operation's documents ([`decode_op`]).
fn document(c: &mut Cursor<'_>) -> Result<ExplicitDocument, FfiStatus> {
    let n = c.count(9, "stored count")?;
    let mut stored = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("stored field number")?;
        stored.push(StoredField {
            field_number,
            value: stored_value(c)?,
        });
    }
    let n = c.count(9, "inverted count")?;
    let mut inverted = try_with_capacity(n)?;
    for _ in 0..n {
        inverted.push(inverted_field(c)?);
    }
    let n = c.count(9, "doc values count")?;
    let mut doc_values = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("doc values field number")?;
        let value = match c.u8("doc values kind")? {
            0 => FieldValue::Long(c.i64("doc value")?),
            1 => FieldValue::Binary(c.bytes("doc value")?.to_vec()),
            _ => return Err(decode_error("doc values kind")),
        };
        doc_values.push(StoredField {
            field_number,
            value,
        });
    }
    let n = c.count(8, "points count")?;
    let mut points = try_with_capacity(n)?;
    for _ in 0..n {
        let field_number = c.i32("points field number")?;
        points.push(StoredField {
            field_number,
            value: FieldValue::Binary(c.bytes("packed point")?.to_vec()),
        });
    }
    Ok(ExplicitDocument {
        stored,
        fields: ExplicitFields {
            inverted,
            doc_values,
            points,
            registered: Vec::new(),
            term_vectors: Vec::new(),
        },
    })
}

/// One document's value of one vector field: `i32 len, utf8 field name,
/// u8 encoding (0 float32, 1 byte), i32 dimension`, then the components
/// (`f32` bits as `i32` each, or one byte each) -- a
/// `KnnFloatVectorField`/`KnnByteVectorField`.
fn document_vector(c: &mut Cursor<'_>) -> Result<DocumentVector, FfiStatus> {
    let name = c.string("vector field name")?;
    let encoding = c.u8("vector encoding")?;
    let dim = usize::try_from(c.i32("vector dimension")?)
        .map_err(|_| decode_error("vector dimension"))?;
    Ok(match encoding {
        0 => {
            let raw = c.take(
                dim.checked_mul(4)
                    .ok_or_else(|| decode_error("vector dimension"))?,
                "vector",
            )?;
            let mut v = try_with_capacity(dim)?;
            v.extend(
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            DocumentVector::float32(name, v)
        }
        1 => DocumentVector::byte(name, c.take(dim, "vector")?.to_vec()),
        _ => return Err(decode_error("vector encoding")),
    })
}

/// An operation blob, decoded.
enum Op {
    Add {
        partition: Option<Arc<str>>,
        docs: Vec<ExplicitDocument>,
        /// One list per document; `None` when no document has a vector.
        vectors: Option<Vec<Vec<DocumentVector>>>,
    },
    SoftUpdate {
        partition: Option<Arc<str>>,
        term: Term,
        soft_field: String,
        soft_value: i64,
        docs: Vec<ExplicitDocument>,
        vectors: Option<Vec<Vec<DocumentVector>>>,
    },
    Panic,
}

/// `u8 kind`, then for a soft update `term field (i32 len, utf8), term bytes
/// (i32 len, bytes), soft-deletes field (i32 len, utf8), i64 value`, then
/// `i32 count` documents ([`document`]), then, when any of them has a KNN
/// vector, per document `i32 n` and `n` vectors ([`document_vector`]). The
/// whole blob must be consumed.
///
/// A soft update is Lucene's `softUpdateDocuments(term, docs, softDeletes)`:
/// every live document matching `term` gets `soft_field = soft_value` as a
/// doc-values update, and `docs` are added, atomically. A delete's tombstone
/// is a soft update whose one document carries the soft-deletes value itself;
/// a stale operation is an add whose documents do.
fn decode_op(blob: &[u8]) -> Result<Op, FfiStatus> {
    let mut c = Cursor::new(blob);
    let kind = c.u8("op kind")?;
    let op = match kind {
        OP_PANIC => Op::Panic,
        OP_ADD | OP_SOFT_UPDATE | OP_ADD_PARTITIONED | OP_SOFT_UPDATE_PARTITIONED => {
            let partition = if matches!(kind, OP_ADD_PARTITIONED | OP_SOFT_UPDATE_PARTITIONED) {
                Some(Arc::<str>::from(c.string("partition")?))
            } else {
                None
            };
            let update = if matches!(kind, OP_SOFT_UPDATE | OP_SOFT_UPDATE_PARTITIONED) {
                let field = c.string("term field")?;
                let bytes = c.bytes("term bytes")?.to_vec();
                let soft_field = c.string("soft-deletes field")?;
                let soft_value = c.i64("soft-deletes value")?;
                Some((Term { field, bytes }, soft_field, soft_value))
            } else {
                None
            };
            let n = c.count(16, "document count")?;
            if n == 0 {
                return Err(decode_error("an operation needs at least one document"));
            }
            let mut docs = try_with_capacity(n)?;
            for _ in 0..n {
                docs.push(document(&mut c)?);
            }
            // The documents' vectors, when any has one: per document a count
            // and that many [`document_vector`]s.
            let vectors = if c.remaining() > 0 {
                let mut vectors: Vec<Vec<DocumentVector>> = try_with_capacity(n)?;
                for _ in 0..n {
                    let m = c.count(9, "vector count")?;
                    let mut doc_vectors = try_with_capacity(m)?;
                    for _ in 0..m {
                        doc_vectors.push(document_vector(&mut c)?);
                    }
                    vectors.push(doc_vectors);
                }
                Some(vectors)
            } else {
                None
            };
            match update {
                None => Op::Add {
                    partition,
                    docs,
                    vectors,
                },
                Some((term, soft_field, soft_value)) => Op::SoftUpdate {
                    partition,
                    term,
                    soft_field,
                    soft_value,
                    docs,
                    vectors,
                },
            }
        }
        _ => return Err(decode_error("op kind")),
    };
    c.finish("trailing bytes after the operation")?;
    Ok(op)
}

/// `i32 len, utf8 name, u8 index_options (0 none .. 4 docs+freqs+positions+
/// offsets), u8 omit_norms, u8 store_term_vectors, u8 store_payloads,
/// u8 doc_values_type (0 none, 1 numeric, 2 binary, 3 sorted, 4 sorted_set,
/// 5 sorted_numeric), u8 doc_values_skip_index, i32 point_dimension_count,
/// i32 point_index_dimension_count, i32 point_num_bytes, i32 vector_dimension,
/// and for a vector field (a non-zero dimension) its [`decode_vector_schema`],
/// then u8 soft_deletes_field, i32 attribute count, then (i32 len, utf8) key and
/// value pairs` -- Lucene's `FieldInfo`. The attributes are the field type's
/// own (`FieldType.putAttribute`); the writer adds its per-field format ones
/// per segment.
fn decode_field(blob: &[u8]) -> Result<(FieldInfo, Option<KnnVectorsFormat>), FfiStatus> {
    let mut c = Cursor::new(blob);
    let name = c.string("field name")?;
    let index_options = match c.u8("index options")? {
        0 => IndexOptions::None,
        1 => IndexOptions::Docs,
        2 => IndexOptions::DocsAndFreqs,
        3 => IndexOptions::DocsAndFreqsAndPositions,
        4 => IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        _ => return Err(decode_error("index options")),
    };
    let flag = |c: &mut Cursor<'_>, what: &str| -> Result<bool, FfiStatus> {
        match c.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(decode_error(what)),
        }
    };
    let omit_norms = flag(&mut c, "omit norms")?;
    let store_term_vectors = flag(&mut c, "store term vectors")?;
    let store_payloads = flag(&mut c, "store payloads")?;
    let doc_values_type = match c.u8("doc values type")? {
        0 => DocValuesType::None,
        1 => DocValuesType::Numeric,
        2 => DocValuesType::Binary,
        3 => DocValuesType::Sorted,
        4 => DocValuesType::SortedSet,
        5 => DocValuesType::SortedNumeric,
        _ => return Err(decode_error("doc values type")),
    };
    let skip = if flag(&mut c, "doc values skip index")? {
        DocValuesSkipIndexType::Range
    } else {
        DocValuesSkipIndexType::None
    };
    let dims = c.i32("point dimensions")?;
    let index_dims = c.i32("point index dimensions")?;
    let num_bytes = c.i32("point bytes")?;
    let vector_dimension = c.i32("vector dimension")?;
    let vector = if vector_dimension != 0 {
        Some(decode_vector_schema(&mut c)?)
    } else {
        None
    };
    let soft = flag(&mut c, "soft deletes")?;
    let attributes = decode_user_data_pairs(
        &mut c,
        [
            "field attribute count",
            "field attribute key",
            "field attribute value",
        ],
    )?;
    c.finish("trailing bytes after the field")?;
    let mut info = FieldInfo::new(name, 0)
        .with_index_options(index_options)
        .with_omit_norms(omit_norms)
        .with_store_term_vectors(store_term_vectors)
        .with_store_payloads(store_payloads)
        .with_doc_values(doc_values_type, skip, -1)
        .with_points(dims, index_dims, num_bytes)
        .with_soft_deletes_field(soft)
        .with_attributes(attributes);
    let format = match vector {
        Some((encoding, similarity, format)) => {
            info = info.with_vectors(vector_dimension, encoding, similarity);
            format
        }
        None => None,
    };
    Ok((info, format))
}

/// The field attribute naming the `KnnVectorsFormat` the plugin's mapping
/// chose for a vector field ([`parse_vectors_format`]), kept in the
/// segments' `FieldInfos` so a writer reopened on the index merges the field
/// in the same format before any document registers it again.
pub(crate) const VECTORS_FORMAT_ATTRIBUTE: &str = "lucene_rust.knn_format";

/// A [`VECTORS_FORMAT_ATTRIBUTE`] value: `hnsw:<max_conn>:<beam_width>`,
/// `hnsw_hamming:<max_conn>:<beam_width>`, `hnsw_half:<max_conn>:<beam_width>`,
/// `hnsw_sq:<scalar encoding wire number>:<max_conn>:<beam_width>`,
/// `sq:<scalar encoding wire number>` or
/// `ivf:<target_cluster_size>:<rerank encoding>`; `None` for anything else.
pub(crate) fn parse_vectors_format(value: &str) -> Option<KnnVectorsFormat> {
    let parts: Vec<&str> = value.split(':').collect();
    let int = |s: &str| s.parse::<i32>().ok();
    match parts.as_slice() {
        ["hnsw", m, beam] => KnnVectorsFormat::hnsw(int(m)?, int(beam)?).ok(),
        ["hnsw_hamming", m, beam] => KnnVectorsFormat::hnsw_hamming(int(m)?, int(beam)?).ok(),
        ["hnsw_half", m, beam] => KnnVectorsFormat::hnsw_half_float(int(m)?, int(beam)?).ok(),
        ["hnsw_sq", e, m, beam] => KnnVectorsFormat::hnsw_scalar_quantized(
            ScalarEncoding::from_wire_number(int(e)?)?,
            int(m)?,
            int(beam)?,
        )
        .ok(),
        ["sq", e] => Some(KnnVectorsFormat::scalar_quantized(
            ScalarEncoding::from_wire_number(int(e)?)?,
        )),
        ["ivf", size, e] => KnnVectorsFormat::ivf(size.parse().ok()?, e.parse().ok()?).ok(),
        _ => None,
    }
}

/// The field attribute marking a vector field whose `_source` value the plugin stores masked, as
/// k-NN's derived source does: `true` for a field of the root document, `nested` for one of nested
/// documents. Every segment flushed with the field lists it under
/// [`lucene_search::derived_source::DERIVED_VECTOR_FIELDS`] or
/// [`lucene_search::derived_source::NESTED_DERIVED_VECTOR_FIELDS`], so a reader (the plugin's or
/// stock k-NN's) puts the vector back.
pub(crate) const DERIVED_SOURCE_ATTRIBUTE: &str = "lucene_rust.derived_source";

/// The `.si` list that names `info` when it carries [`DERIVED_SOURCE_ATTRIBUTE`].
fn derived_source_list(info: &FieldInfo) -> Option<&'static str> {
    use lucene_search::derived_source::{DERIVED_VECTOR_FIELDS, NESTED_DERIVED_VECTOR_FIELDS};
    info.attributes
        .iter()
        .find(|(k, _)| k == DERIVED_SOURCE_ATTRIBUTE)
        .and_then(|(_, v)| match v.as_str() {
            "true" => Some(DERIVED_VECTOR_FIELDS),
            "nested" => Some(NESTED_DERIVED_VECTOR_FIELDS),
            _ => None,
        })
}

/// k-NN's derived-source `.si` lists, for a k-NN index ([`OPEN_KNN`]): flushed segments list the
/// derived fields, and a merge keeps every name its sources list, so masked `_source` values
/// never lose their vectors.
fn derived_source_attributes(writer: &mut IndexWriter<'_>) {
    use lucene_search::derived_source::{DERIVED_VECTOR_FIELDS, NESTED_DERIVED_VECTOR_FIELDS};
    writer.set_merge_union_attributes(vec![
        DERIVED_VECTOR_FIELDS.to_string(),
        NESTED_DERIVED_VECTOR_FIELDS.to_string(),
    ]);
    let derived: Vec<(&'static str, String)> = writer
        .fields()
        .iter()
        .filter_map(|f| derived_source_list(f).map(|list| (list, f.name.clone())))
        .collect();
    for (list, name) in derived {
        writer.add_segment_list_value(list, &name);
    }
}

/// Routes every vector field of `writer`'s existing segments to the format
/// its [`VECTORS_FORMAT_ATTRIBUTE`] names.
fn restore_vectors_formats(writer: &mut IndexWriter<'_>) {
    let found: Vec<(String, KnnVectorsFormat)> = writer
        .fields()
        .iter()
        .filter(|f| f.vector_dimension > 0)
        .filter_map(|f| {
            let value = f
                .attributes
                .iter()
                .find(|(k, _)| k == VECTORS_FORMAT_ATTRIBUTE)
                .map(|(_, v)| v)?;
            Some((f.name.clone(), parse_vectors_format(value)?))
        })
        .collect();
    for (name, format) in found {
        writer.set_knn_vectors_format_for_field(&name, format);
    }
}

/// A vector field's `u8 encoding (0 float32, 1 byte), u8 similarity
/// (VectorSimilarityFunction ordinal: 0 euclidean, 1 dot product, 2 cosine,
/// 3 maximum inner product)`, then its `KnnVectorsFormat`: `u8 kind` --
/// `0` the codec's default (`Lucene99HnswVectorsFormat()`), `1`
/// `Lucene99HnswVectorsFormat(max_conn, beam_width)`, `2`
/// `Lucene104HnswScalarQuantizedVectorsFormat(encoding, max_conn,
/// beam_width)`, `3` `Lucene104ScalarQuantizedVectorsFormat(encoding)`, `4`
/// k-NN's `KNN9120HnswBinaryVectorsFormat(max_conn, beam_width)` (byte
/// fields only; recorded as `KNN990HnswBinaryVectorsFormat`) or `5` k-NN's
/// `KNN1040HnswHalfFloatVectorsFormat(max_conn, beam_width)` (float fields
/// only) or `6` `LuceneRustIvfVectorsFormat(target_cluster_size,
/// rerank_encoding)`; kinds `1`, `2`, `4` and `5` are followed by `i32 max_conn,
/// i32 beam_width`, kinds `2` and `3` by `u8 scalar encoding`
/// (`ScalarEncoding.getWireNumber`), kind `6` by `i32 target_cluster_size, u8
/// rerank_encoding` (`0` f32, `1` f16, `2` bf16, `3` int8).
fn decode_vector_schema(
    c: &mut Cursor<'_>,
) -> Result<
    (
        VectorEncoding,
        VectorSimilarityFunction,
        Option<KnnVectorsFormat>,
    ),
    FfiStatus,
> {
    let encoding = match c.u8("vector encoding")? {
        0 => VectorEncoding::Float32,
        1 => VectorEncoding::Byte,
        _ => return Err(decode_error("vector encoding")),
    };
    let similarity = match c.u8("vector similarity")? {
        0 => VectorSimilarityFunction::Euclidean,
        1 => VectorSimilarityFunction::DotProduct,
        2 => VectorSimilarityFunction::Cosine,
        3 => VectorSimilarityFunction::MaximumInnerProduct,
        _ => return Err(decode_error("vector similarity")),
    };
    let kind = c.u8("vectors format")?;
    let graph = |c: &mut Cursor<'_>| -> Result<(i32, i32), FfiStatus> {
        Ok((c.i32("max conn")?, c.i32("beam width")?))
    };
    let scalar = |c: &mut Cursor<'_>| -> Result<ScalarEncoding, FfiStatus> {
        ScalarEncoding::from_wire_number(i32::from(c.u8("scalar encoding")?))
            .ok_or_else(|| decode_error("scalar encoding"))
    };
    let invalid = |e: lucene_codecs::per_field_knn_vectors::InvalidFormat| {
        set_last_error(e.0);
        FfiStatus::InvalidArgument
    };
    let format = match kind {
        0 => None,
        1 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw(m, beam).map_err(invalid)?)
        }
        2 => {
            let (m, beam) = graph(c)?;
            let e = scalar(c)?;
            Some(KnnVectorsFormat::hnsw_scalar_quantized(e, m, beam).map_err(invalid)?)
        }
        3 => Some(KnnVectorsFormat::scalar_quantized(scalar(c)?)),
        4 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw_hamming(m, beam).map_err(invalid)?)
        }
        5 => {
            let (m, beam) = graph(c)?;
            Some(KnnVectorsFormat::hnsw_half_float(m, beam).map_err(invalid)?)
        }
        6 => {
            let size = u32::try_from(c.i32("target cluster size")?)
                .map_err(|_| decode_error("target cluster size"))?;
            let rerank = c.u8("rerank encoding")?;
            Some(KnnVectorsFormat::ivf(size, rerank).map_err(invalid)?)
        }
        _ => return Err(decode_error("vectors format")),
    };
    if encoding == VectorEncoding::Byte
        && matches!(format, Some(KnnVectorsFormat::HnswHalfFloat { .. }))
    {
        set_last_error("a byte vector field cannot be stored as half floats");
        return Err(FfiStatus::InvalidArgument);
    }
    if encoding == VectorEncoding::Float32
        && matches!(format, Some(KnnVectorsFormat::HnswHamming { .. }))
    {
        set_last_error("a float vector field cannot be scored by Hamming distance");
        return Err(FfiStatus::InvalidArgument);
    }
    if format.as_ref().is_some_and(|f| {
        encoding == VectorEncoding::Byte
            && matches!(
                f,
                KnnVectorsFormat::HnswScalarQuantized { .. }
                    | KnnVectorsFormat::ScalarQuantized { .. }
            )
    }) {
        set_last_error("a byte vector field cannot be scalar quantized");
        return Err(FfiStatus::InvalidArgument);
    }
    Ok((encoding, similarity, format))
}

/// `i32 count`, then `count` pairs of `(i32 len, utf8)` key and value.
fn decode_user_data(blob: &[u8]) -> Result<Vec<(String, String)>, FfiStatus> {
    let mut c = Cursor::new(blob);
    let out = decode_user_data_pairs(
        &mut c,
        ["user data count", "user data key", "user data value"],
    )?;
    c.finish("trailing bytes after the user data")?;
    Ok(out)
}

/// `i32 count`, then `count` pairs of `(i32 len, utf8)` key and value;
/// `what` names the count, a key and a value in errors.
fn decode_user_data_pairs(
    c: &mut Cursor<'_>,
    [count, key, value]: [&str; 3],
) -> Result<Vec<(String, String)>, FfiStatus> {
    let n = c.count(8, count)?;
    let mut out = try_with_capacity(n)?;
    for _ in 0..n {
        let k = c.string(key)?;
        let v = c.string(value)?;
        out.push((k, v));
    }
    Ok(out)
}

/// Maps a writer error to a status. What Java raises as
/// `IllegalArgumentException` (caller misuse, a refused document, an add past
/// `maxDocs`, an inconsistent configuration) is [`FfiStatus::InvalidArgument`]
/// and fails only the operation; I/O and corrupt-index errors are
/// [`FfiStatus::Io`] and tragic. Every variant is listed, so a new one must be
/// classified here.
fn writer_error(e: index_writer::Error) -> FfiStatus {
    let status = match &e {
        index_writer::Error::NoPreparedCommit
        | index_writer::Error::PrepareCommitAlreadyCalled
        | index_writer::Error::PreparedCommitPending(_)
        | index_writer::Error::ConcurrentUnsupported(_)
        | index_writer::Error::Explicit(_)
        | index_writer::Error::LiveSearch(_)
        | index_writer::Error::Document(_)
        | index_writer::Error::ZeroNorm(_)
        | index_writer::Error::MergeHook(_)
        | index_writer::Error::InvalidMaxNumSegments(_)
        | index_writer::Error::UnknownPostingsField(_)
        | index_writer::Error::UnsupportedPostingsIndexOptions(_, _)
        | index_writer::Error::FieldInfos(_)
        | index_writer::Error::NoPayloadFields
        | index_writer::Error::UnknownTermVectorField(_)
        | index_writer::Error::UnsupportedTermVectorField(_)
        | index_writer::Error::UnknownDocValuesField(_)
        | index_writer::Error::UnsupportedDocValuesType(_, _)
        | index_writer::Error::MissingDenseDocValue(_, _)
        | index_writer::Error::NonNumericDocValue(_, _, _)
        | index_writer::Error::NonBinaryDocValue(_, _, _)
        | index_writer::Error::UnknownVectorField(_)
        | index_writer::Error::UnknownPointsField(_)
        | index_writer::Error::UnsupportedPointsField(_, _, _, _)
        | index_writer::Error::DuplicatePointsField(_)
        | index_writer::Error::PointValueShape(_, _, _, _, _)
        | index_writer::Error::UnsupportedVectorField(_, _)
        | index_writer::Error::DuplicateVectorField(_)
        | index_writer::Error::VectorDimensionMismatch(_, _, _, _)
        | index_writer::Error::VectorEncodingMismatch(_, _, _, _)
        | index_writer::Error::DuplicatePostingsField(_)
        | index_writer::Error::UnknownCustomFreqPostingsField(_)
        | index_writer::Error::UnsupportedCustomFreqPostingsIndexOptions(_, _)
        | index_writer::Error::PostingsAndCustomFreqPostingsMutuallyExclusive(_)
        | index_writer::Error::DuplicateTermVectorField(_)
        | index_writer::Error::UnknownNormsField(_)
        | index_writer::Error::UnsupportedNormsField(_)
        | index_writer::Error::InvalidRamBufferSize(_)
        | index_writer::Error::InvalidMaxBufferedDocs(_)
        | index_writer::Error::BothAutoFlushTriggersDisabled
        | index_writer::Error::NoSoftDeletesSupplied
        | index_writer::Error::TooManyDocs(_)
        | index_writer::Error::NoDocValuesUpdatesSupplied
        | index_writer::Error::UnknownDocValuesUpdateField(_)
        | index_writer::Error::WrongDocValuesUpdateType { .. }
        | index_writer::Error::EmptyIndexSort
        | index_writer::Error::UnknownIndexSortField(_)
        | index_writer::Error::UnsupportedIndexSortField(_, _)
        | index_writer::Error::IndexSortFieldWithoutDocValues(_)
        | index_writer::Error::IncongruentIndexSort { .. }
        | index_writer::Error::IndexSortChangedMidBuffer(_)
        | index_writer::Error::IndexSortWithBlocksAndNoParentField
        | index_writer::Error::BlocksWithIndexSortNeedParentField
        | index_writer::Error::DocValuesUpdateOnIndexSortField { .. }
        | index_writer::Error::DuplicateDocValuesField(_)
        | index_writer::Error::MergeSortDisagreement { .. }
        | index_writer::Error::InvalidMergeRate(_)
        | index_writer::Error::InvalidMergeScheduler(_)
        | index_writer::Error::AddIndexes(_)
        | index_writer::Error::UnknownSortField(_) => FfiStatus::InvalidArgument,
        // I/O, decode and corrupt-index failures; `MergeAbortedException` is an
        // `IOException` in Java.
        index_writer::Error::Store(_)
        | index_writer::Error::MergeThreadPanicked(_)
        | index_writer::Error::SegmentWriter(_)
        | index_writer::Error::SegmentInfos(_)
        | index_writer::Error::UpdateDocument(_)
        | index_writer::Error::TermDelete(_)
        | index_writer::Error::Deletes(_)
        | index_writer::Error::FieldUpdates(_)
        | index_writer::Error::Deleter(_)
        | index_writer::Error::Merge(_)
        | index_writer::Error::SegmentInfo(_)
        | index_writer::Error::SimpleText(_)
        | index_writer::Error::StoredFields(_)
        | index_writer::Error::LiveDocs(_)
        | index_writer::Error::PostingsWriter(_)
        | index_writer::Error::Blocktree(_)
        | index_writer::Error::Postings(_)
        | index_writer::Error::TermVectors(_)
        | index_writer::Error::DocValues(_)
        | index_writer::Error::Norms(_)
        | index_writer::Error::Vectors(_)
        | index_writer::Error::Ivf(_)
        | index_writer::Error::Points(_)
        | index_writer::Error::DocValuesRead(_)
        | index_writer::Error::NormsRead(_)
        | index_writer::Error::MergeSortColumnMissing(_)
        | index_writer::Error::SegmentDocCountMismatch { .. }
        | index_writer::Error::MergeAborted
        | index_writer::Error::UnreadableSegmentPostings { .. } => FfiStatus::Io,
    };
    set_last_error(format!("engine writer: {e}"));
    status
}

pub(crate) fn lookup(handle: u64) -> Result<Arc<EngineWriter>, FfiStatus> {
    read_recovering(engine_writers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error("engine writer: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })
}

/// Runs `body` on the handle's writer. A recorded tragic error fails the
/// call; a failure of `body` records one unless it is the document's own
/// ([`FfiStatus::InvalidArgument`], [`FfiStatus::Decode`]), and so does a
/// panic, which then carries on to the caller's guard.
fn with_writer<T>(
    handle: u64,
    body: impl FnOnce(&EngineWriter) -> Result<T, FfiStatus>,
) -> Result<T, FfiStatus> {
    let w = lookup(handle)?;
    if let Some(t) = w.tragic.get() {
        set_last_error(format!("engine writer closed by a tragic event: {t}"));
        return Err(FfiStatus::Io);
    }
    match catch_unwind(AssertUnwindSafe(|| body(&w))) {
        Ok(Err(status)) if status != FfiStatus::InvalidArgument && status != FfiStatus::Decode => {
            let _ = w.tragic.set(crate::error::last_error());
            Err(status)
        }
        Ok(out) => out,
        Err(panic) => {
            let _ = w
                .tragic
                .set("a previous call panicked inside the writer".to_string());
            resume_unwind(panic)
        }
    }
}

/// [`ffi_engine_writer_open`]'s flag allowing [`OP_PANIC`].
pub const OPEN_FAULT_INJECTION: u8 = 1;
/// [`ffi_engine_writer_open`]'s flag for an `index.sparse` index.
pub const OPEN_SPARSE: u8 = 2;
/// [`ffi_engine_writer_open`]'s flag for an index searched without a refresh
/// (`crate::live`).
pub const OPEN_LIVE_SEARCH: u8 = 4;
/// [`ffi_engine_writer_open`]'s flag for
/// `index.lucene_rust.stored_fields.merge_lz4_high_compression`: merges write
/// `BEST_SPEED` stored fields with the high-compression LZ4 match finder,
/// which stock Lucene reads
/// ([`IndexWriter::set_merge_lz4_high_compression`]).
pub const OPEN_MERGE_LZ4_HC: u8 = 8;
/// [`ffi_engine_writer_open`]'s flag for an `index.knn` index: segments record OpenSearch
/// k-NN's codec name ([`KNN_CODEC`]) over the stored-fields codec, whose name they record under
/// [`KNN_DELEGATE_CODEC_ATTRIBUTE`], as stock k-NN writes them; so k-NN's derived source reads
/// back through that codec on any node.
pub const OPEN_KNN: u8 = 128;
/// The codec name a k-NN index's segments record (`KNN1040Codec`).
pub(crate) const KNN_CODEC: &str = "KNN1040Codec";
/// The `.si` attribute naming the codec whose stored fields a k-NN segment holds
/// (`KNN10010DerivedSourceStoredFieldsFormat.KNN_DELEGATE_CODEC_NAME`).
pub(crate) const KNN_DELEGATE_CODEC_ATTRIBUTE: &str = "knn_delegate_stored_fields_codec_key";
/// [`ffi_engine_writer_open`]'s flag for `index.lucene_rust.zstd_terms`: new
/// segments write every field's terms in the `RustZstd104` postings format,
/// whose term blocks are zstd-compressed
/// ([`lucene_codecs::per_field_postings::Lucene104PostingsFormat::rust_zstd`]).
pub const OPEN_ZSTD_TERMS: u8 = 16;
/// The zstd level of [`OPEN_ZSTD_TERMS`]' term blocks. Blocks decode at the
/// same speed whatever the level; higher ones only write slower.
const ZSTD_TERMS_LEVEL: i32 = 3;
/// [`ffi_engine_writer_open`]'s flag for `index.lucene_rust.adaptive_doc_values`:
/// new segments write their doc values in the `RustAdaptive` format
/// ([`lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_adaptive`]).
pub const OPEN_ADAPTIVE_DOC_VALUES: u8 = 32;
/// [`ffi_engine_writer_open`]'s flag for an index with a partition field
/// (`index.lucene_rust.partition_field`): every segment holds one partition
/// and records it, and merges never mix partitions
/// ([`lucene_index::index_writer::SegmentPartitioning::opensearch`]).
pub const OPEN_PARTITIONED: u8 = 64;
/// Indexing buffers in a partitioned writer: a buffer holds one partition,
/// so each partition in flight keeps its own until the RAM buffer flushes the
/// largest or a refresh flushes them all. Past this many partitions in flight
/// a buffer of another partition is flushed to make room.
const PARTITIONED_SLOTS: usize = 1024;

/// Opens the shard's writer over the index at `path`, which must already hold
/// a commit (OpenSearch's `Store.createEmpty` or a recovered one).
/// `ram_buffer_mb` is the flush threshold. `flags`: [`OPEN_FAULT_INJECTION`]
/// allows the [`OP_PANIC`] operation; [`OPEN_SPARSE`] writes the segments of
/// an `index.sparse` index ([`lucene_sparse::SparseCodec`]);
/// [`OPEN_LIVE_SEARCH`] keeps what a search of the buffers reads;
/// [`OPEN_ADAPTIVE_DOC_VALUES`] writes doc values in the `RustAdaptive`
/// format; [`OPEN_ZSTD_TERMS`] writes terms dictionaries with zstd blocks;
/// [`OPEN_PARTITIONED`] keeps partitions in segments of their own;
/// [`OPEN_MERGE_LZ4_HC`] has merges compress stored fields harder.
/// `index_sort` is the index's sort as a `.si` encodes it
/// ([`lucene_index::segment_info::parse_index_sort`]; empty for none):
/// every segment is written in that order, as `IndexWriterConfig.setIndexSort`
/// has it. `parent_field` (empty for none) names the field that marks the
/// last document of each block (`IndexWriterConfig.setParentField`).
///
/// # Safety
/// `path` valid for `path_len` bytes, `index_sort` for `index_sort_len`,
/// `parent_field` for `parent_field_len`; `out_handle` valid for one write.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_engine_writer_open(
    path: *const c_char,
    path_len: usize,
    ram_buffer_mb: f64,
    flags: u8,
    index_sort: *const u8,
    index_sort_len: usize,
    parent_field: *const c_char,
    parent_field_len: usize,
    max_docs: i32,
    out_handle: *mut u64,
) -> i32 {
    // SAFETY: the same contract.
    unsafe {
        ffi_engine_writer_open_codec(
            path,
            path_len,
            ram_buffer_mb,
            flags,
            index_sort,
            index_sort_len,
            parent_field,
            parent_field_len,
            max_docs,
            STORED_FIELDS_BEST_SPEED,
            0,
            out_handle,
        )
    }
}

/// [`ffi_engine_writer_open`]'s stored fields: `Lucene104` with
/// `BEST_SPEED` (`index.codec: default`).
pub const STORED_FIELDS_BEST_SPEED: u8 = 0;
/// `Lucene104` with `BEST_COMPRESSION` (`index.codec: best_compression`).
pub const STORED_FIELDS_BEST_COMPRESSION: u8 = 1;
/// OpenSearch custom-codecs' `ZSTD104` (`index.codec: zstd`).
pub const STORED_FIELDS_ZSTD: u8 = 2;
/// OpenSearch custom-codecs' `ZSTDNODICT104` (`index.codec: zstd_no_dict`).
pub const STORED_FIELDS_ZSTD_NO_DICT: u8 = 3;
/// The Rust engine's paged zstd stored fields, `LuceneRustZstdPaged104`
/// (`index.lucene_rust.stored_fields.paged_zstd`): 16 KiB chunks packed into
/// 16 KiB pages, with a dictionary trained at large merges.
pub const STORED_FIELDS_ZSTD_PAGED: u8 = 4;

/// [`ffi_engine_writer_open`] writing the stored fields `stored_fields`
/// names (`STORED_FIELDS_*`), the zstd ones at `compression_level`
/// (`index.codec.compression_level`; ignored by the others). An
/// `index.sparse` index ([`OPEN_SPARSE`]) takes the default stored fields
/// only.
///
/// # Safety
/// As [`ffi_engine_writer_open`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_engine_writer_open_codec(
    path: *const c_char,
    path_len: usize,
    ram_buffer_mb: f64,
    flags: u8,
    index_sort: *const u8,
    index_sort_len: usize,
    parent_field: *const c_char,
    parent_field_len: usize,
    max_docs: i32,
    stored_fields: u8,
    compression_level: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        // SAFETY: caller contract.
        let index_sort = unsafe { bytes_from_raw(index_sort, index_sort_len)? };
        let index_sort = if index_sort.is_empty() {
            None
        } else {
            lucene_index::segment_info::parse_index_sort(index_sort).map_err(|e| {
                set_last_error(format!("engine writer: index sort: {e}"));
                FfiStatus::InvalidArgument
            })?
        };
        // SAFETY: caller contract.
        let parent_field = unsafe { str_from_raw(parent_field, parent_field_len)? };
        // A search of the buffers reads documents in arrival order, which a
        // sorted flush renumbers.
        if index_sort.is_some() && flags & OPEN_LIVE_SEARCH != 0 {
            set_last_error("engine writer: a search of the buffers does not support an index sort");
            return Err(FfiStatus::InvalidArgument);
        }
        // Mapped, not read onto the heap: a merge opens every source file of
        // every merged segment, and Lucene's own `MMapDirectory` -- what
        // OpenSearch writes with -- maps them too. Segment files are written
        // once and only deleted (never truncated) once unreferenced, which is
        // the condition mapping needs.
        // A `rust_encrypted` shard's path is bound to an encrypting directory
        // over the mapping (`store.rs`).
        let dir: Box<dyn Directory> = crate::store::shard_directory(path).map_err(|e| {
            set_last_error(format!("opening the shard directory [{path}]: {e}"));
            FfiStatus::Io
        })?;
        let dir_ref: &dyn Directory = &*dir;
        // SAFETY: `WriterHandle`'s own argument -- the boxed directory
        // outlives the writer that borrows it (see `writer.rs`).
        let dir_ref: &'static dyn Directory = unsafe { std::mem::transmute(dir_ref) };
        // Merges run on the scheduler's threads, as under OpenSearch's
        // `ConcurrentMergeScheduler`, with `index.merge.scheduler.*`'s
        // defaults until the engine passes the shard's settings
        // (`ffi_engine_writer_set_merge_scheduler`).
        let merges = ConcurrentMergeScheduler::new();
        merges.set_default_max_merges_and_threads(false);
        merges.enable_auto_io_throttle();
        let sparse = flags & OPEN_SPARSE != 0;
        let stored_codec = stored_fields_codec(stored_fields, compression_level)?;
        if sparse && stored_codec.is_some() {
            set_last_error(
                "an index.sparse index takes the default stored fields only".to_string(),
            );
            return Err(FfiStatus::InvalidArgument);
        }
        let knn = flags & OPEN_KNN != 0;
        if sparse && knn {
            set_last_error("an index.sparse index is not a k-NN index".to_string());
            return Err(FfiStatus::InvalidArgument);
        }
        let open = || -> index_writer::Result<Arc<ConcurrentIndexWriter<'static>>> {
            let codec = if sparse {
                lucene_sparse::CODEC_NAME
            } else if knn {
                KNN_CODEC
            } else {
                "Lucene104"
            };
            let mut writer = IndexWriter::open(dir_ref, Vec::new(), codec, VERSION)?;
            if sparse {
                writer.set_codec(Some(Arc::new(lucene_sparse::SparseCodec)))?;
            } else if knn {
                let delegate: Arc<dyn lucene_codecs::codec::Lucene104Codec> =
                    match stored_codec.clone() {
                        Some(c) => c,
                        None => Arc::new(lucene_codecs::codec::DefaultLucene104Codec),
                    };
                writer.set_segment_attribute(KNN_DELEGATE_CODEC_ATTRIBUTE, delegate.name());
                writer.set_codec(Some(Arc::new(
                    lucene_codecs::codec::NamedLucene104Codec::new(KNN_CODEC, delegate),
                )))?;
            } else if let Some(codec) = stored_codec.clone() {
                writer.set_codec(Some(codec))?;
            }
            if flags & OPEN_MERGE_LZ4_HC != 0 {
                writer.set_merge_lz4_high_compression(true);
            }
            if flags & OPEN_ADAPTIVE_DOC_VALUES != 0 {
                writer.set_default_doc_values_format(
                    lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_adaptive(),
                );
            }
            if flags & OPEN_ZSTD_TERMS != 0 {
                // A valid level, so this always builds the format.
                if let Ok(format) =
                    lucene_codecs::per_field_postings::Lucene104PostingsFormat::rust_zstd(
                        ZSTD_TERMS_LEVEL,
                    )
                {
                    writer.set_default_postings_format(format);
                }
            }
            writer.enable_explicit_documents()?;
            restore_vectors_formats(&mut writer);
            if knn {
                derived_source_attributes(&mut writer);
            }
            if !parent_field.is_empty() {
                writer.set_parent_field(Some(parent_field))?;
            }
            if index_sort.is_some() {
                writer.set_index_sort(index_sort.as_deref())?;
            }
            writer.set_deletion_policy(DeletionPolicy::KeepAll)?;
            writer.set_merge_policy(Some(MergePolicyConfig::default()));
            // `OpenSearchTieredMergePolicy`: forced merges have no merged-size
            // cap; forced-deletes merges keep the policy's.
            writer.set_forced_merge_max_segment_size(Some(u64::MAX));
            // Merge-on-commit/refresh stays off until the engine passes
            // the shard's `index.merge_on_flush.*` settings
            // (`ffi_engine_writer_set_merge_on_flush`).
            writer.set_max_full_flush_merge_wait_millis(0);
            // Loose segment files until the engine passes the shard's
            // `index.use_compound_file` and `index.compound_format`
            // (`ffi_engine_writer_set_options`).
            writer.set_use_compound_file(false);
            writer.set_ram_buffer_size_mb(ram_buffer_mb)?;
            // The JVM's `IndexWriter.getActualMaxDocs()`, so a test lowering
            // Java's limit lowers this one too.
            writer.set_max_docs(usize::try_from(max_docs).unwrap_or(0));
            // One buffer per thread that can index at once.
            let mut slots = std::thread::available_parallelism().map_or(1, |n| n.get());
            if flags & OPEN_PARTITIONED != 0 {
                writer.set_segment_partitioning(Some(SegmentPartitioning::opensearch()));
                slots = slots.max(PARTITIONED_SLOTS);
            }
            ConcurrentIndexWriter::with_merge_scheduler(writer, slots, Arc::new(merges.clone()))
        };
        let writer = open().map_err(writer_error)?;
        if flags & OPEN_LIVE_SEARCH != 0 {
            writer.enable_live_search().map_err(writer_error)?;
        }
        let engine = EngineWriter {
            handle: WriterHandle {
                writer,
                merges,
                dir,
            },
            maintenance: Mutex::new(Holds {
                files: HashMap::new(),
                nrt: HashMap::new(),
                next: 1,
            }),
            tragic: OnceLock::new(),
            fault_injection: flags & OPEN_FAULT_INJECTION != 0,
            policies: Mutex::new(MergePolicies::default()),
            sparse: flags & OPEN_SPARSE != 0,
            live_chunks: Mutex::default(),
            live_warmer: crate::live::Warmer::default(),
        };
        let handle = lock_recovering(engine_writers()).insert_checked(Arc::new(engine))?;
        if let Some(&upcall) = MERGE_UPCALL.get() {
            // OpenSearch's engine hears of every merge, and throttles indexing
            // itself when merges fall behind instead of the scheduler stalling.
            with_writer(handle, |w| {
                w.handle.merges.set_stall_producers(false);
                let listener = Arc::new(MergeUpcallListener { handle, upcall });
                w.writer()
                    .with_writer(|iw| iw.set_merge_listener(Some(listener)));
                Ok(())
            })?;
        }
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// The codec [`ffi_engine_writer_open_codec`]'s `stored_fields` names, or
/// `None` for the default one.
fn stored_fields_codec(
    stored_fields: u8,
    level: i32,
) -> Result<Option<Arc<dyn lucene_codecs::codec::Lucene104Codec>>, FfiStatus> {
    use lucene_codecs::codec::{
        BestCompressionLucene104Codec, ZstdLucene104Codec, ZstdPagedLucene104Codec,
    };
    use lucene_codecs::stored_fields::ZstdMode;
    let zstd = |mode: ZstdMode| -> Result<Option<Arc<dyn lucene_codecs::codec::Lucene104Codec>>, FfiStatus> {
        let codec = ZstdLucene104Codec::new(mode).map_err(|e| {
            set_last_error(format!("index.codec.compression_level: {e}"));
            FfiStatus::InvalidArgument
        })?;
        Ok(Some(Arc::new(codec)))
    };
    match stored_fields {
        STORED_FIELDS_BEST_SPEED => Ok(None),
        STORED_FIELDS_BEST_COMPRESSION => Ok(Some(Arc::new(BestCompressionLucene104Codec))),
        STORED_FIELDS_ZSTD => zstd(ZstdMode::zstd(level)),
        STORED_FIELDS_ZSTD_NO_DICT => zstd(ZstdMode::zstd_no_dict(level)),
        STORED_FIELDS_ZSTD_PAGED => {
            let codec = ZstdPagedLucene104Codec::new(level).map_err(|e| {
                set_last_error(format!("index.codec.compression_level: {e}"));
                FfiStatus::InvalidArgument
            })?;
            Ok(Some(Arc::new(codec)))
        }
        other => {
            set_last_error(format!("unknown stored fields format {other}"));
            Err(FfiStatus::InvalidArgument)
        }
    }
}

/// Registers a field ([`decode_field`]) and writes its global number.
///
/// # Safety
/// `spec` valid for `spec_len` bytes; `out_number` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_register_field(
    handle: u64,
    spec: *const u8,
    spec_len: usize,
    out_number: *mut i32,
) -> i32 {
    guard(|| {
        if out_number.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let spec = unsafe { bytes_from_raw(spec, spec_len)? };
        let (info, format) = decode_field(spec)?;
        let number = with_writer(handle, |w| {
            let name = info.name.clone();
            let derived = derived_source_list(&info);
            w.writer()
                .with_writer(|iw| {
                    let number = iw.register_field(info)?;
                    if let Some(format) = format {
                        iw.set_knn_vectors_format_for_field(&name, format);
                    }
                    // Only a k-NN index's codec reads the list back.
                    if let Some(list) = derived {
                        if iw.segment_attribute(KNN_DELEGATE_CODEC_ATTRIBUTE).is_some() {
                            iw.add_segment_list_value(list, &name);
                        }
                    }
                    Ok::<_, lucene_index::index_writer::Error>(number)
                })
                .map_err(|e| {
                    // A field the writer cannot index is the mapping's problem,
                    // not the writer's: document-level.
                    set_last_error(e.to_string());
                    FfiStatus::InvalidArgument
                })
        })?;
        // SAFETY: caller contract.
        unsafe { *out_number = number };
        Ok(())
    })
}

/// Applies one decoded operation on the calling thread.
fn apply_op(handle: u64, op: Op) -> Result<(), FfiStatus> {
    with_writer(handle, |w| {
        let writer = w.writer();
        match op {
            Op::Add {
                partition,
                docs,
                vectors,
            } => match vectors {
                None => writer.add_explicit_documents_in(partition, docs),
                Some(vectors) => {
                    writer.add_explicit_documents_with_vectors_in(partition, docs, vectors)
                }
            }
            .map_err(writer_error)?,
            Op::SoftUpdate {
                partition,
                term,
                soft_field,
                soft_value,
                docs,
                vectors,
            } => {
                let soft = DocValuesUpdate::Numeric {
                    term: term.clone(),
                    field: soft_field,
                    value: Some(soft_value),
                };
                match vectors {
                    None => {
                        writer.soft_update_explicit_documents_in(partition, term, docs, &[soft])
                    }
                    Some(vectors) => writer.soft_update_explicit_documents_with_vectors_in(
                        partition,
                        term,
                        docs,
                        vectors,
                        &[soft],
                    ),
                }
                .map_err(writer_error)?
            }
            Op::Panic => {
                if !w.fault_injection {
                    set_last_error("engine writer: fault injection is not enabled");
                    return Err(FfiStatus::InvalidArgument);
                }
                panic!("injected fault: engine writer panic");
            }
        };
        crate::live::after_add(w);
        Ok(())
    })
}

/// Applies one operation blob ([`decode_op`]).
///
/// # Safety
/// `op` valid for `op_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_apply(handle: u64, op: *const u8, op_len: usize) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let op = decode_op(unsafe { bytes_from_raw(op, op_len)? })?;
        apply_op(handle, op)
    })
}

/// Decoded operations waiting for [`ffi_engine_writer_apply_prepared`], by id.
static PREPARED: std::sync::LazyLock<Mutex<HashMap<u64, Op>>> =
    std::sync::LazyLock::new(Default::default);
static NEXT_PREPARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Decodes an operation blob -- analyzing the fields it leaves to Rust,
/// the expensive part -- without any writer's lock, so indexing threads
/// analyze in parallel; `*out_id` names the result for
/// [`ffi_engine_writer_apply_prepared`] (or [`ffi_engine_prepared_free`]).
/// A document the decoder refuses is [`FfiStatus::InvalidArgument`], as
/// from [`ffi_engine_writer_apply`].
///
/// # Safety
/// `op` valid for `op_len` bytes; `out_id` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_prepare(op: *const u8, op_len: usize, out_id: *mut u64) -> i32 {
    guard(|| {
        if out_id.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let op = decode_op(unsafe { bytes_from_raw(op, op_len)? })?;
        let id = NEXT_PREPARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PREPARED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, op);
        // SAFETY: caller contract.
        unsafe { *out_id = id };
        Ok(())
    })
}

/// Applies (and forgets) the operation [`ffi_engine_prepare`] decoded.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_apply_prepared(handle: u64, id: u64) -> i32 {
    guard(|| {
        let op = PREPARED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id)
            .ok_or_else(|| {
                set_last_error(format!("no prepared operation {id}"));
                FfiStatus::InvalidHandle
            })?;
        apply_op(handle, op)
    })
}

/// Forgets a prepared operation that will not be applied.
#[no_mangle]
pub extern "C" fn ffi_engine_prepared_free(id: u64) -> i32 {
    guard(|| {
        PREPARED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&id);
        Ok(())
    })
}

/// The JVM's commit-data upcall: `(context, out, cap, out_len) -> status`.
/// It writes the user data of the commit `context` names ([`decode_user_data`]'s
/// layout) into `out` and its length into `*out_len`, and returns
/// [`COMMIT_DATA_OK`]; when `cap` is too small it writes only the length
/// needed and returns [`COMMIT_DATA_BUFFER_TOO_SMALL`]. Any other status is a
/// failure the JVM keeps and reports itself. It is called on the thread that
/// called [`ffi_engine_writer_commit`], while that call is running.
pub type CommitDataUpcall = unsafe extern "C" fn(u64, *mut u8, usize, *mut usize) -> i32;
/// The commit-data upcall succeeded.
pub const COMMIT_DATA_OK: i32 = 0;
/// The commit-data upcall's buffer was too small; `*out_len` is the size needed.
pub const COMMIT_DATA_BUFFER_TOO_SMALL: i32 = 1;
/// The JVM could not produce the commit data; it keeps its own error.
pub const COMMIT_DATA_FAILED: i32 = 2;

static COMMIT_DATA_UPCALL: OnceLock<CommitDataUpcall> = OnceLock::new();

/// Registers the commit-data upcall ([`CommitDataUpcall`]), once per process;
/// later registrations are ignored.
///
/// # Safety
/// `upcall` must have [`CommitDataUpcall`]'s contract and stay callable for
/// the life of the process.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_set_commit_data_upcall(
    upcall: Option<CommitDataUpcall>,
) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        let _ = COMMIT_DATA_UPCALL.set(upcall);
        Ok(())
    })
}

/// The JVM's merge upcall, `OpenSearchConcurrentMergeScheduler`'s
/// `beforeMerge`/`afterMerge` for the Rust writer's merges: `(writer handle,
/// event, merge id, total docs, total bytes, maxNumSegments, took ms, stopped
/// ms, throttled ms, failure kind, segment names, their length, failure
/// message, its length)`. `event` is [`MERGE_EVENT_BEFORE`] or
/// [`MERGE_EVENT_AFTER`]. The segment names are UTF-8, one per line. After a
/// failed merge the failure kind is [`MERGE_FAILED_IO`] or
/// [`MERGE_FAILED_OTHER`] and the message is UTF-8; otherwise the kind is
/// [`MERGE_OK`] and the message null. Called on the thread
/// running the merge -- a Rust merge thread, or the caller of
/// [`ffi_engine_writer_force_merge`] while it holds the writer -- so the JVM
/// side must not call back into the writer except for
/// [`ffi_engine_writer_merge_state`].
pub type MergeUpcall = unsafe extern "C" fn(
    u64,
    i32,
    i64,
    i64,
    i64,
    i32,
    i64,
    i64,
    i64,
    i32,
    *const u8,
    usize,
    *const u8,
    usize,
);
/// A merge is about to run.
pub const MERGE_EVENT_BEFORE: i32 = 0;
/// A merge ended, successfully or not.
pub const MERGE_EVENT_AFTER: i32 = 1;
/// The merge did not fail (or has not ended).
pub const MERGE_OK: i32 = 0;
/// The merge failed reading or writing files (`IOException`).
pub const MERGE_FAILED_IO: i32 = 1;
/// The merge failed some other way.
pub const MERGE_FAILED_OTHER: i32 = 2;

static MERGE_UPCALL: OnceLock<MergeUpcall> = OnceLock::new();

/// Registers the merge upcall ([`MergeUpcall`]), once per process; later
/// registrations are ignored. Writers opened before it hear nothing.
///
/// # Safety
/// `upcall` must have [`MergeUpcall`]'s contract and stay callable for the
/// life of the process.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_set_merge_upcall(upcall: Option<MergeUpcall>) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        let _ = MERGE_UPCALL.set(upcall);
        Ok(())
    })
}

/// Hands one writer's merges to the merge upcall.
struct MergeUpcallListener {
    handle: u64,
    upcall: MergeUpcall,
}

impl MergeUpcallListener {
    fn send(
        &self,
        event: i32,
        merge: &MergeEvent,
        times: MergeTimes,
        failure: Option<&index_writer::Error>,
    ) {
        let ms = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        let names = merge.segments.join("\n");
        let kind = match failure {
            None => MERGE_OK,
            Some(index_writer::Error::Store(lucene_store::Error::Io(_))) => MERGE_FAILED_IO,
            Some(_) => MERGE_FAILED_OTHER,
        };
        let message = failure.map(|e| e.to_string());
        let (ptr, len) = message
            .as_deref()
            .map_or((std::ptr::null(), 0), |f| (f.as_ptr(), f.len()));
        // SAFETY: `ffi_engine_set_merge_upcall`'s contract; `names` and
        // `message` outlive the call.
        unsafe {
            (self.upcall)(
                self.handle,
                event,
                i64::try_from(merge.id).unwrap_or(i64::MAX),
                merge.total_docs,
                merge.total_bytes,
                merge.max_num_segments,
                ms(times.took_ms),
                ms(times.stopped_ms),
                ms(times.throttled_ms),
                kind,
                names.as_ptr(),
                names.len(),
                ptr,
                len,
            );
        }
    }
}

impl MergeListener for MergeUpcallListener {
    fn before_merge(&self, merge: &MergeEvent) {
        self.send(MERGE_EVENT_BEFORE, merge, MergeTimes::default(), None);
    }

    fn after_merge(
        &self,
        merge: &MergeEvent,
        times: MergeTimes,
        failure: Option<&index_writer::Error>,
    ) {
        self.send(MERGE_EVENT_AFTER, merge, times, failure);
    }
}

/// The JVM's merged-segment warmer (`IndexWriter.IndexReaderWarmer`):
/// `(writer handle, segment name, its length, segment id)` with the id's
/// [`lucene_store::codec_util::ID_LENGTH`] bytes, called on the merge
/// thread once the merged segment's files are written and before the merge
/// is published. Returns [`WARM_OK`], or anything else to fail the merge.
pub type WarmUpcall = unsafe extern "C" fn(u64, *const u8, usize, *const u8) -> i32;
/// The merged segment was warmed.
pub const WARM_OK: i32 = 0;

static WARM_UPCALL: OnceLock<WarmUpcall> = OnceLock::new();

/// Registers the merged-segment warmer upcall ([`WarmUpcall`]), once per
/// process; later registrations are ignored.
///
/// # Safety
/// `upcall` must have [`WarmUpcall`]'s contract and stay callable for the
/// life of the process.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_set_warm_upcall(upcall: Option<WarmUpcall>) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        let _ = WARM_UPCALL.set(upcall);
        Ok(())
    })
}

/// Hands one writer's merged segments to the warm upcall.
struct UpcallWarmer {
    handle: u64,
    upcall: WarmUpcall,
}

impl MergedSegmentWarmer for UpcallWarmer {
    fn warm(
        &self,
        _dir: &dyn Directory,
        segment: &lucene_index::segment_infos::SegmentCommitInfo,
    ) -> index_writer::Result<()> {
        let name = &segment.segment_name;
        // SAFETY: `ffi_engine_set_warm_upcall`'s contract; `name` and the id
        // outlive the call.
        let status = unsafe {
            (self.upcall)(
                self.handle,
                name.as_ptr(),
                name.len(),
                segment.segment_id.as_ptr(),
            )
        };
        if status == WARM_OK {
            return Ok(());
        }
        Err(index_writer::Error::Store(lucene_store::Error::Io(
            std::io::Error::other(format!("warming merged segment {name} failed")),
        )))
    }
}

/// `IndexWriterConfig.setMergedSegmentWarmer`: with `on != 0`, every merged
/// segment goes to the warm upcall before its merge is published, and the
/// compound-file decision is made before it; `0` warms nothing. Refused
/// before [`ffi_engine_set_warm_upcall`].
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_merged_segment_warmer(handle: u64, on: u8) -> i32 {
    guard(|| {
        let warmer = if on != 0 {
            let Some(&upcall) = WARM_UPCALL.get() else {
                set_last_error("no warm upcall is registered".to_string());
                return Err(FfiStatus::InvalidArgument);
            };
            Some(Arc::new(UpcallWarmer { handle, upcall }) as Arc<dyn MergedSegmentWarmer>)
        } else {
            None
        };
        with_writer(handle, |w| {
            w.writer()
                .with_writer(|iw| iw.set_merged_segment_warmer(warmer));
            Ok(())
        })
    })
}

/// Writes whether merges are registered that no merge thread has taken yet
/// (`IndexWriter.hasPendingMerges`, `0`/`1`) and the merge scheduler's IO
/// rate in MB/s (`getIORateLimitMBPerSec`). Takes no lock the writer holds
/// while merging, so the merge upcall may call it.
///
/// # Safety
/// `out_pending` and `out_mb_per_sec` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_merge_state(
    handle: u64,
    out_pending: *mut u8,
    out_mb_per_sec: *mut f64,
) -> i32 {
    guard(|| {
        if out_pending.is_null() || out_mb_per_sec.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let (pending, rate) = with_writer(handle, |w| {
            Ok((
                w.writer().has_pending_merges(),
                w.handle.merges.io_rate_limit_mb_per_sec(),
            ))
        })?;
        // SAFETY: caller contract.
        unsafe {
            *out_pending = u8::from(pending);
            *out_mb_per_sec = rate;
        }
        Ok(())
    })
}

/// Asks the JVM for the user data of the commit `context` names.
fn commit_data(context: u64) -> Result<Vec<(String, String)>, FfiStatus> {
    let Some(upcall) = COMMIT_DATA_UPCALL.get() else {
        set_last_error("engine writer: no commit-data upcall is registered");
        return Err(FfiStatus::InvalidArgument);
    };
    let mut buf = crate::raw::try_with_capacity::<u8>(512)?;
    buf.resize(512, 0);
    for _ in 0..2 {
        let mut len = 0usize;
        // SAFETY: `buf` is valid for `buf.len()` writes and `len` for one;
        // the upcall's contract is `ffi_engine_set_commit_data_upcall`'s.
        let status = unsafe { upcall(context, buf.as_mut_ptr(), buf.len(), &mut len) };
        match status {
            COMMIT_DATA_OK if len <= buf.len() => {
                return decode_user_data(buf.get(..len).unwrap_or_default());
            }
            COMMIT_DATA_BUFFER_TOO_SMALL if len > buf.len() => {
                buf = crate::raw::try_with_capacity::<u8>(len)?;
                buf.resize(len, 0);
            }
            COMMIT_DATA_FAILED => {
                // A refusal, not the writer's failure: the JVM reports its own error.
                set_last_error("engine writer: the commit-data upcall failed");
                return Err(FfiStatus::InvalidArgument);
            }
            _ => break,
        }
    }
    set_last_error("engine writer: the commit-data upcall returned an inconsistent length");
    Err(FfiStatus::InvalidArgument)
}

/// A commit's failure: the writer's, or the commit data's.
enum CommitError {
    Writer(index_writer::Error),
    Status(FfiStatus),
}

impl From<index_writer::Error> for CommitError {
    fn from(e: index_writer::Error) -> Self {
        CommitError::Writer(e)
    }
}

/// Commits, and writes the new commit's generation. Every buffer holding
/// operations from before the call is flushed first; then the commit-data
/// upcall gives the commit's user data for `context` -- read at this point,
/// as `IndexWriter.commit` iterates its live commit data after
/// `flushAllThreads`, so a maximum sequence number in it covers every
/// operation in the commit. Other threads go on indexing meanwhile, into
/// buffers this commit does not hold. Merges the commit makes possible run
/// on the writer's merge threads; the merged segments are in the next commit.
///
/// # Safety
/// `out_generation` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_commit(
    handle: u64,
    context: u64,
    out_generation: *mut i64,
) -> i32 {
    guard(|| {
        if out_generation.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let generation = with_writer(handle, |w| {
            let _maintenance = w.maintenance();
            let writer = w.writer();
            writer
                .commit_with(|| {
                    let data = commit_data(context).map_err(CommitError::Status)?;
                    writer.with_writer(|iw| iw.set_live_commit_data(data));
                    Ok::<(), CommitError>(())
                })
                .map_err(|e| match e {
                    CommitError::Writer(e) => writer_error(e),
                    CommitError::Status(status) => status,
                })?;
            Ok(w.generation())
        })?;
        // SAFETY: caller contract.
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// Sets `index.merge.scheduler.*` as `OpenSearchConcurrentMergeScheduler.refreshConfig`
/// does: at most `max_merge_count` merges at once, `max_thread_count` of them
/// running big merges, the automatic IO throttle (`auto_throttle != 0`), and
/// forced merges' rate (`max_force_merge_mb_per_sec`, infinite for none).
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_merge_scheduler(
    handle: u64,
    max_merge_count: i32,
    max_thread_count: i32,
    auto_throttle: u8,
    max_force_merge_mb_per_sec: f64,
) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            let merges = &w.handle.merges;
            if merges.max_merge_count() != max_merge_count
                || merges.max_thread_count() != max_thread_count
            {
                merges
                    .set_max_merges_and_threads(max_merge_count, max_thread_count)
                    .map_err(|e| {
                        set_last_error(e.to_string());
                        FfiStatus::InvalidArgument
                    })?;
            }
            if (auto_throttle != 0) != merges.auto_io_throttle() {
                if auto_throttle != 0 {
                    merges.enable_auto_io_throttle();
                } else {
                    merges.disable_auto_io_throttle();
                }
            }
            if max_force_merge_mb_per_sec != merges.force_merge_mb_per_sec() {
                merges.set_force_merge_mb_per_sec(max_force_merge_mb_per_sec);
            }
            Ok(())
        })
    })
}

/// Writes the generations of every commit point the writer still holds,
/// oldest first, into `out` (capacity `cap`) and their number into
/// `out_len` -- [`FfiStatus::BufferTooSmall`] when they do not fit, with
/// `out_len` still set.
///
/// # Safety
/// `out` valid for `cap` writes; `out_len` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_commit_generations(
    handle: u64,
    out: *mut i64,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if out_len.is_null() || (out.is_null() && cap != 0) {
            return Err(FfiStatus::NullPointer);
        }
        let gens = with_writer(handle, |w| {
            let _maintenance = w.maintenance();
            Ok(w.writer().with_writer(|iw| iw.commit_generations()))
        })?;
        // SAFETY: caller contract.
        unsafe { *out_len = gens.len() };
        if gens.len() > cap {
            return Err(FfiStatus::BufferTooSmall);
        }
        // SAFETY: `gens.len() <= cap` and `out` is valid for `cap` writes.
        unsafe { std::ptr::copy_nonoverlapping(gens.as_ptr(), out, gens.len()) };
        Ok(())
    })
}

/// Drops the named commit points (never the newest).
///
/// # Safety
/// `generations` valid for `n` reads.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_delete_commits(
    handle: u64,
    generations: *const i64,
    n: usize,
) -> i32 {
    guard(|| {
        let gens: Vec<i64> = if n == 0 {
            Vec::new()
        } else {
            if generations.is_null() {
                return Err(FfiStatus::NullPointer);
            }
            // SAFETY: caller contract.
            crate::raw::try_to_vec(unsafe { std::slice::from_raw_parts(generations, n) })?
        };
        with_writer(handle, |w| {
            let _maintenance = w.maintenance();
            w.writer()
                .with_writer(|iw| iw.delete_commits(&gens))
                .map_err(writer_error)
        })
    })
}

/// Pins commit `generation`'s segment files for a reader opened on it; the
/// hold id goes to `out_hold`.
///
/// # Safety
/// `out_hold` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_hold_commit(
    handle: u64,
    generation: i64,
    out_hold: *mut u64,
) -> i32 {
    guard(|| {
        if out_hold.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let id = with_writer(handle, |w| {
            let mut holds = w.maintenance();
            let files = w
                .writer()
                .with_writer(|iw| iw.hold_commit(generation))
                .map_err(|e| {
                    // No such commit is the caller's mistake, not the writer's.
                    set_last_error(e.to_string());
                    FfiStatus::InvalidArgument
                })?;
            let id = holds.next;
            holds.next = holds.next.saturating_add(1);
            holds.files.insert(id, files);
            Ok(id)
        })?;
        // SAFETY: caller contract.
        unsafe { *out_hold = id };
        Ok(())
    })
}

/// A near-real-time view without a commit (`IndexWriter.getReader`): flushes
/// every buffer into a segment, applies every buffered delete, and returns
/// the live segment list as `segments_N` bytes (`SegmentInfos.write`), with
/// the last commit's generation (`*out_generation`, which the bytes' header
/// names) and user data. Its files stay pinned under `*out_hold` until
/// [`ffi_engine_writer_release_hold`]. Nothing is synced: the segments become
/// durable at the next commit.
///
/// # Safety
/// Out pointers valid for one write each; the bytes go back through
/// `ffi_jvm_free_bytes`.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_nrt_snapshot(
    handle: u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    out_hold: *mut u64,
    out_generation: *mut i64,
) -> i32 {
    guard(|| {
        if out_ptr.is_null() || out_len.is_null() || out_hold.is_null() || out_generation.is_null()
        {
            return Err(FfiStatus::NullPointer);
        }
        let (bytes, id, generation) = with_writer(handle, |w| {
            use lucene_index::nrt::NrtSource;
            // Not under the maintenance lock: a forced merge holds it for
            // its whole run, and a refresh must not wait for that.
            let snapshot = w.writer().nrt_snapshot(true, true).map_err(writer_error)?;
            let bytes = lucene_index::segment_infos::to_bytes(&snapshot.segment_infos);
            let mut holds = w.maintenance();
            let id = holds.next;
            holds.next = holds.next.saturating_add(1);
            holds.nrt.insert(id, snapshot.hold);
            Ok((bytes, id, snapshot.segment_infos.generation))
        })?;
        // SAFETY: caller contract.
        unsafe {
            crate::ffm_bridge::give(bytes, out_ptr, out_len);
            *out_hold = id;
            *out_generation = generation;
        }
        Ok(())
    })
}

/// Releases a [`ffi_engine_writer_hold_commit`] pin.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_hold(handle: u64, hold: u64) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            let mut holds = w.maintenance();
            if let Some(nrt) = holds.nrt.remove(&hold) {
                drop(holds);
                // The last reference hands the files back to the writer;
                // release them now, as Java's reader close does, rather than
                // at the next flush or snapshot -- an index that goes idle
                // after a merge would otherwise keep the merged-away segments
                // on disk. A failed delete is not the writer's failure (Java's
                // close listener ignores it): the file is unreferenced and the
                // deleter's next sweep removes it.
                if Arc::into_inner(nrt).is_some() {
                    let _ = w.writer().with_writer(|iw| iw.release_nrt_holds());
                }
                return Ok(());
            }
            let Some(files) = holds.files.remove(&hold) else {
                set_last_error(format!("engine writer: no hold {hold}"));
                return Err(FfiStatus::InvalidArgument);
            };
            w.writer()
                .with_writer(|iw| iw.release_files(&files))
                .map_err(writer_error)
        })
    })
}

/// Sets the soft-deletes retention merges apply: `enabled == 0` keeps every
/// soft-deleted document, otherwise those whose `_seq_no` is below
/// `min_retained_seq_no` are dropped.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_retention(
    handle: u64,
    enabled: u8,
    min_retained_seq_no: i64,
) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            let retention = (enabled != 0).then(|| SoftDeletesRetention {
                seq_no_field: SEQ_NO_FIELD.to_string(),
                min_retained_seq_no,
                prune_postings_field: Some(ID_FIELD.to_string()),
                prune_recovery_source_field: Some(RECOVERY_SOURCE_FIELD.to_string()),
            });
            w.writer()
                .with_writer(|iw| iw.set_soft_deletes_retention(retention));
            Ok(())
        })
    })
}

/// `index.merge.policy.*` as OpenSearch's `TieredMergePolicyProvider` sets
/// them on its `TieredMergePolicy`: merges at once, segments per tier, the
/// largest merged segment and the floor size in bytes, the deletes
/// percentage natural merges tolerate, and the one `forceMergeDeletes`
/// tolerates. Applies to the merges proposed from the next flush on, and to
/// the merge-on-flush policy the next time it is set. A writer with merges
/// off keeps them off.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_merge_policy(
    handle: u64,
    max_merge_at_once: i32,
    segments_per_tier: f64,
    max_merged_segment_bytes: i64,
    floor_segment_bytes: i64,
    deletes_pct_allowed: f64,
    force_merge_deletes_pct_allowed: f64,
) -> i32 {
    guard(|| {
        let invalid = |what: &str| {
            set_last_error(format!("invalid merge policy setting: {what}"));
            FfiStatus::InvalidArgument
        };
        let max_merge_at_once = usize::try_from(max_merge_at_once)
            .ok()
            .filter(|&n| n >= 2)
            .ok_or_else(|| invalid("max_merge_at_once < 2"))?;
        if segments_per_tier.is_nan() || segments_per_tier < 2.0 {
            return Err(invalid("segments_per_tier < 2"));
        }
        let max_merged_segment_size = u64::try_from(max_merged_segment_bytes)
            .map_err(|_| invalid("max_merged_segment < 0"))?;
        let floor_segment_size =
            u64::try_from(floor_segment_bytes).map_err(|_| invalid("floor_segment < 0"))?;
        if !(5.0..=50.0).contains(&deletes_pct_allowed) {
            return Err(invalid("deletes_pct_allowed outside [5, 50]"));
        }
        if !(0.0..=100.0).contains(&force_merge_deletes_pct_allowed) {
            return Err(invalid("expunge_deletes_allowed outside [0, 100]"));
        }
        with_writer(handle, |w| {
            let policies = w.policies();
            w.writer().with_writer(|iw| {
                // A writer with merges off keeps them off.
                let Some(mut config) = iw.merge_policy().cloned() else {
                    return;
                };
                config.max_merge_at_once = max_merge_at_once;
                config.segments_per_tier = segments_per_tier;
                config.max_merged_segment_size = max_merged_segment_size;
                config.floor_segment_size = floor_segment_size;
                config.deletes_pct_allowed = deletes_pct_allowed;
                config.force_merge_deletes_pct_allowed = force_merge_deletes_pct_allowed;
                iw.set_merge_policy(Some(config));
                install_merge_policies(iw, &policies);
            });
            Ok(())
        })
    })
}

/// No pluggable policy: [`ffi_engine_writer_set_merge_policy`]'s tiered one decides.
pub const MERGE_POLICY_TIERED: i32 = 0;
/// `LogByteSizeMergePolicy`.
pub const MERGE_POLICY_LOG_BYTE_SIZE: i32 = 1;
/// `NoMergePolicy`.
pub const MERGE_POLICY_NONE: i32 = 2;

/// The shard's merge policy when it is not `TieredMergePolicy`: `kind` is
/// [`MERGE_POLICY_LOG_BYTE_SIZE`] with `index.merge.log_byte_size_policy.*`
/// (merge factor, smallest and largest merged segment in bytes, the largest
/// for forced merges, the most documents a merged segment may hold, and the
/// policy's `noCFSRatio` and `maxCFSSegmentSize` in bytes), or
/// [`MERGE_POLICY_NONE`] for `index.merge.enabled: false`, whose other
/// arguments are ignored; [`MERGE_POLICY_TIERED`] goes back to the tiered
/// policy. It decides natural, forced and forced-deletes merges from the next
/// one on, and the merge-on-flush policy the next time that is set.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_pluggable_merge_policy(
    handle: u64,
    kind: i32,
    merge_factor: i32,
    min_merge_bytes: i64,
    max_merge_bytes: i64,
    max_forced_merge_bytes: i64,
    max_merge_docs: i32,
    no_cfs_ratio: f64,
    max_cfs_segment_bytes: i64,
) -> i32 {
    guard(|| {
        let invalid = |what: String| {
            set_last_error(format!("invalid merge policy setting: {what}"));
            FfiStatus::InvalidArgument
        };
        let policy = match kind {
            MERGE_POLICY_TIERED => None,
            MERGE_POLICY_NONE => Some(PluggablePolicy::NoMerges),
            MERGE_POLICY_LOG_BYTE_SIZE => {
                if merge_factor < 2 {
                    return Err(invalid(format!("merge_factor {merge_factor} < 2")));
                }
                if min_merge_bytes < 0 || max_merge_bytes < 0 || max_forced_merge_bytes < 0 {
                    return Err(invalid("a merge size < 0".to_string()));
                }
                if max_merge_docs < 0 {
                    return Err(invalid(format!("max_merged_docs {max_merge_docs} < 0")));
                }
                let mut compound = CompoundFileSettings::default();
                compound
                    .set_no_cfs_ratio(no_cfs_ratio)
                    .map_err(|e| invalid(e.to_string()))?;
                if max_cfs_segment_bytes < 0 {
                    return Err(invalid(format!(
                        "maxCFSSegmentSize {max_cfs_segment_bytes} < 0"
                    )));
                }
                compound.max_cfs_segment_size = max_cfs_segment_bytes;
                Some(PluggablePolicy::LogByteSize {
                    merge_factor,
                    min_merge_bytes,
                    max_merge_bytes,
                    max_forced_merge_bytes,
                    max_merge_docs,
                    compound,
                })
            }
            other => return Err(invalid(format!("unknown merge policy kind {other}"))),
        };
        with_writer(handle, |w| {
            let mut policies = w.policies();
            if policies.other == policy {
                return Ok(());
            }
            policies.other = policy;
            w.writer()
                .with_writer(|iw| install_merge_policies(iw, &policies));
            Ok(())
        })
    })
}

/// The writer options OpenSearch's `NativeLuceneIndexWriterFactory` sets
/// beside the merge policy: whether flushed segments are compound
/// (`index.use_compound_file`); the merge policy's `noCFSRatio`
/// (`index.compound_format`, `0.0..=1.0`) and `maxCFSSegmentSize` in bytes,
/// which decide it for merged segments; whether indexing threads help with
/// pending flushes (`index.check_pending_flush.enabled`); and whether forced
/// merges interleave their sources (`ShuffleForcedMergePolicy`, on unless the
/// JVM's `opensearch.shuffle_forced_merge` is false).
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_options(
    handle: u64,
    use_compound_file: u8,
    no_cfs_ratio: f64,
    max_cfs_segment_bytes: i64,
    check_pending_flush: u8,
    shuffle_forced_merges: u8,
) -> i32 {
    guard(|| {
        let mut compound = CompoundFileSettings::default();
        compound.set_no_cfs_ratio(no_cfs_ratio).map_err(|e| {
            set_last_error(e.to_string());
            FfiStatus::InvalidArgument
        })?;
        if max_cfs_segment_bytes < 0 {
            set_last_error(format!(
                "maxCFSSegmentSize must be >= 0; got {max_cfs_segment_bytes}"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        with_writer(handle, |w| {
            let policies = w.policies();
            w.writer().with_writer(|iw| {
                iw.set_use_compound_file(use_compound_file != 0);
                compound.max_cfs_segment_size = max_cfs_segment_bytes;
                iw.set_merge_compound_file_settings(Some(compound));
                iw.set_shuffle_forced_merges(shuffle_forced_merges != 0);
                // The wrapping policies decide merged segments' compound files too.
                install_merge_policies(iw, &policies);
            });
            drop(policies);
            w.writer()
                .set_check_pending_flush_on_update(check_pending_flush != 0);
            Ok(())
        })
    })
}

/// Merge-on-commit/refresh as OpenSearch's `NativeLuceneIndexWriterFactory`
/// configures it from `index.merge_on_flush.*`: commits and refreshes wait
/// up to `wait_millis` for the merges the policy asks of the segments they
/// publish (`0` or less turns them off), and `merge_on_flush_policy != 0`
/// picks those merges with `MergeOnFlushMergePolicy` around the writer's
/// policy (`index.merge_on_flush.policy: merge-on-flush`).
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_merge_on_flush(
    handle: u64,
    wait_millis: i64,
    merge_on_flush_policy: u8,
) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            let mut policies = w.policies();
            policies.merge_on_flush = merge_on_flush_policy != 0;
            w.writer().with_writer(|iw| {
                iw.set_max_full_flush_merge_wait_millis(wait_millis);
                install_merge_policies(iw, &policies);
            });
            Ok(())
        })
    })
}

/// `BPReorderingMergePolicy` around the shard's merge policy
/// (`index.lucene_rust.bp_reorder.*`), or off when `enabled == 0`: merges of
/// at least `max(min_merge_docs, min_merge_ratio * the biggest segment's
/// maxDoc)` live documents, and every forced merge, renumber their documents
/// with `BPIndexReorderer` so documents sharing terms sit together. The
/// merged segments are ordinary Lucene segments. `fields` is the reorderer's
/// fields as NUL-separated UTF-8 names, every indexed field when empty; the
/// other values are `BPIndexReorderer`'s setters'. Not available on a sparse
/// shard, and a no-op for a sorted index, which keeps its sort.
///
/// # Safety
/// `fields` valid for `fields_len` bytes (or null with `fields_len == 0`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_engine_writer_set_bp_reorder(
    handle: u64,
    enabled: u8,
    min_doc_freq: i32,
    max_doc_freq: f32,
    min_partition_size: i32,
    max_iters: i32,
    ram_budget_mb: f64,
    min_merge_docs: i32,
    min_merge_ratio: f32,
    fields: *const u8,
    fields_len: usize,
) -> i32 {
    guard(|| {
        let invalid = |what: String| {
            set_last_error(format!("invalid bp_reorder setting: {what}"));
            FfiStatus::InvalidArgument
        };
        // SAFETY: caller contract.
        let fields = unsafe { bytes_from_raw(fields, fields_len)? };
        let bp = if enabled == 0 {
            None
        } else {
            let mut reorderer = BpIndexReorderer::new();
            reorderer
                .set_min_doc_freq(min_doc_freq)
                .map_err(|e| invalid(e.to_string()))?;
            reorderer
                .set_max_doc_freq(max_doc_freq)
                .map_err(|e| invalid(e.to_string()))?;
            reorderer
                .set_min_partition_size(min_partition_size)
                .map_err(|e| invalid(e.to_string()))?;
            reorderer
                .set_max_iters(max_iters)
                .map_err(|e| invalid(e.to_string()))?;
            if !(ram_budget_mb.is_finite() && ram_budget_mb > 0.0) {
                return Err(invalid(format!(
                    "ram_budget_mb must be positive, got {ram_budget_mb}"
                )));
            }
            reorderer.set_ram_budget_mb(ram_budget_mb);
            if fields.is_empty() {
                reorderer.set_fields(None);
            } else {
                let names = std::str::from_utf8(fields)
                    .map_err(|_| invalid("fields are not UTF-8".to_string()))?
                    .split('\0')
                    .filter(|f| !f.is_empty())
                    .map(str::to_string)
                    .collect();
                reorderer.set_fields(Some(names));
            }
            // The policy's own checks, before anything changes.
            let mut probe = BpReorderingMergePolicy::new(
                Box::new(TieredMergePolicy::default()),
                reorderer.clone(),
            );
            probe
                .set_min_natural_merge_num_docs(min_merge_docs)
                .map_err(|e| invalid(e.to_string()))?;
            probe
                .set_min_natural_merge_ratio_from_biggest_segment(min_merge_ratio)
                .map_err(|e| invalid(e.to_string()))?;
            Some(BpSettings {
                reorderer,
                min_merge_docs,
                min_merge_ratio,
            })
        };
        with_writer(handle, |w| {
            if bp.is_some() && w.sparse {
                set_last_error("bp_reorder is not supported on a sparse index");
                return Err(FfiStatus::InvalidArgument);
            }
            let mut policies = w.policies();
            policies.bp = bp;
            w.writer()
                .with_writer(|iw| install_merge_policies(iw, &policies));
            Ok(())
        })
    })
}

/// `forceMerge(max_segments)`, or `forceMergeDeletes()` when
/// `only_deletes != 0`, over the committed segments; writes the newest
/// commit's generation.
///
/// # Safety
/// `out_generation` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_force_merge(
    handle: u64,
    max_segments: i32,
    only_deletes: u8,
    out_generation: *mut i64,
) -> i32 {
    guard(|| {
        if out_generation.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let max = usize::try_from(max_segments).map_err(|_| {
            set_last_error(format!("max_segments must be >= 1; got {max_segments}"));
            FfiStatus::InvalidArgument
        })?;
        if max == 0 {
            set_last_error("max_segments must be >= 1; got 0");
            return Err(FfiStatus::InvalidArgument);
        }
        let generation = with_writer(handle, |w| {
            let _maintenance = w.maintenance();
            w.writer()
                .with_writer_merges_aborted(|iw| {
                    if only_deletes != 0 {
                        iw.force_merge_deletes()
                    } else {
                        iw.force_merge(max)
                    }
                })
                .map_err(writer_error)?;
            // The merged segments become durable with a commit.
            w.writer().commit().map_err(writer_error)?;
            Ok(w.generation())
        })?;
        // SAFETY: caller contract.
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// Writes `min(n, STAT_COUNT)` counters into `out`: RAM bytes buffered, the
/// part of them in buffers being flushed, buffered documents, whether anything is uncommitted (0/1), the last
/// commit generation written or attempted (a failed or only prepared commit
/// advances it, as Java's `updateGeneration` does), and the segment count
/// (`STAT_*`). Only the counters asked for are read: the two RAM counts
/// take no lock an indexing thread holds for long.
///
/// # Safety
/// `out` valid for `n` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_stats(handle: u64, out: *mut i64, n: usize) -> i32 {
    guard(|| {
        if out.is_null() && n != 0 {
            return Err(FfiStatus::NullPointer);
        }
        let k = n.min(STAT_COUNT);
        let stats = with_writer(handle, |w| {
            let writer = w.writer();
            let mut s = [0i64; STAT_COUNT];
            s[STAT_RAM_BYTES] = i64::try_from(writer.ram_bytes_used()).unwrap_or(i64::MAX);
            if k > STAT_FLUSHING_BYTES {
                s[STAT_FLUSHING_BYTES] = i64::try_from(writer.flushing_bytes()).unwrap_or(i64::MAX);
            }
            if k > STAT_PENDING_DOCS {
                s[STAT_PENDING_DOCS] =
                    i64::try_from(writer.pending_doc_count()).unwrap_or(i64::MAX);
            }
            if k > STAT_UNCOMMITTED {
                s[STAT_UNCOMMITTED] = i64::from(writer.has_uncommitted_changes());
            }
            if k > STAT_GENERATION {
                (s[STAT_GENERATION], s[STAT_SEGMENTS]) = writer.with_writer(|iw| {
                    let infos = iw.segment_infos();
                    (
                        infos.generation,
                        i64::try_from(infos.segments.len()).unwrap_or(i64::MAX),
                    )
                });
            }
            Ok(s)
        })?;
        // SAFETY: `k <= n` and `out` is valid for `n` writes.
        unsafe { std::ptr::copy_nonoverlapping(stats.as_ptr(), out, k) };
        Ok(())
    })
}

/// Closes the writer without committing -- Lucene's `rollback()`, which is
/// also how OpenSearch closes an engine's writer. Buffered documents are
/// discarded; commits stay. Works on a tragically failed writer too.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_close(handle: u64) -> i32 {
    guard(|| {
        let removed = lock_recovering(engine_writers()).remove(handle);
        let Some(shared) = removed else {
            set_last_error("ffi_engine_writer_close: unknown or already-closed handle");
            return Err(FfiStatus::InvalidHandle);
        };
        // Another thread may still be inside a call; the last `Arc` drops the
        // writer. Holds die with it, as a JVM's readers' pins die with the JVM.
        let _maintenance = shared.maintenance();
        // The shared live view holds the writer.
        shared.live_warmer().stop_sharing();
        shared.writer().stop_merges();
        shared.writer().rollback();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use lucene_store::directory::FsDirectory;
    use lucene_util::test_support::TempDir;

    /// A little-endian blob builder mirroring the JVM side's.
    #[derive(Default, Clone)]
    struct Blob(Vec<u8>);

    impl Blob {
        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn i64(mut self, v: i64) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn bytes(self, b: &[u8]) -> Self {
            let mut s = self.i32(b.len() as i32);
            s.0.extend_from_slice(b);
            s
        }
        fn f32(self, v: f32) -> Self {
            self.i32(v.to_bits() as i32)
        }
    }

    fn field(name: &str, index_options: u8, omit_norms: u8, dv: u8, soft: u8) -> Vec<u8> {
        Blob::default()
            .bytes(name.as_bytes())
            .u8(index_options)
            .u8(omit_norms)
            .u8(0)
            .u8(0)
            .u8(dv)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .u8(soft)
            .i32(0)
            .0
    }

    struct Numbers {
        id: i32,
        body: i32,
        seq: i32,
        soft: i32,
    }

    /// A document: stored `_id`, indexed `_id`, `body` with `words` at
    /// positions 0.., `_seq_no` doc value, and optionally born soft-deleted.
    fn doc(b: Blob, f: &Numbers, id: &str, words: &[&str], seq: i64, soft: bool) -> Blob {
        let mut b = b
            .i32(1)
            .i32(f.id)
            .u8(1)
            .bytes(id.as_bytes())
            // inverted: _id and body
            .i32(2)
            .i32(f.id)
            .u8(0)
            .i32(1)
            .bytes(id.as_bytes())
            .i32(1)
            .u8(0)
            .i32(f.body)
            .u8(1)
            .i64(words.len() as i64)
            .i32(words.len() as i32);
        for (p, w) in words.iter().enumerate() {
            b = b
                .bytes(w.as_bytes())
                .i32(1)
                .u8(3)
                .i32(p as i32)
                .i32(p as i32 * 4)
                .i32(p as i32 * 4 + 3);
        }
        b = b.i32(if soft { 2 } else { 1 }).i32(f.seq).u8(0).i64(seq);
        if soft {
            b = b.i32(f.soft).u8(0).i64(1);
        }
        b.i32(0)
    }

    fn apply(h: u64, blob: &[u8]) -> i32 {
        // SAFETY: a live slice.
        unsafe { ffi_engine_writer_apply(h, blob.as_ptr(), blob.len()) }
    }

    thread_local! {
        /// The user data the test upcall hands the next commit on this thread.
        static COMMIT_DATA: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The tests' commit-data upcall: the bytes [`commit`] set for this thread.
    unsafe extern "C" fn test_commit_data(
        _: u64,
        out: *mut u8,
        cap: usize,
        len: *mut usize,
    ) -> i32 {
        COMMIT_DATA.with(|d| {
            let d = d.borrow();
            // SAFETY: the upcall's contract.
            unsafe {
                *len = d.len();
                if cap < d.len() {
                    return COMMIT_DATA_BUFFER_TOO_SMALL;
                }
                std::ptr::copy_nonoverlapping(d.as_ptr(), out, d.len());
            }
            COMMIT_DATA_OK
        })
    }

    fn commit(h: u64, data: &[(&str, &str)]) -> i64 {
        let mut b = Blob::default().i32(data.len() as i32);
        for (k, v) in data {
            b = b.bytes(k.as_bytes()).bytes(v.as_bytes());
        }
        COMMIT_DATA.with(|d| *d.borrow_mut() = b.0);
        // SAFETY: the upcall keeps its contract for the life of the process.
        assert_eq!(
            unsafe { ffi_engine_set_commit_data_upcall(Some(test_commit_data)) },
            0
        );
        let mut gen = 0;
        // SAFETY: a live out-pointer.
        let rc = unsafe { ffi_engine_writer_commit(h, 0, &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        gen
    }

    /// Commits, then waits for the merges that commit set off and commits
    /// their results, until a commit leaves nothing uncommitted: a commit
    /// whose files are all the writer has on disk.
    fn commit_settled(h: u64) -> i64 {
        let mut gen = commit(h, &[]);
        loop {
            lookup(h).unwrap().handle.merges.sync().unwrap();
            if !lookup(h).unwrap().writer().has_uncommitted_changes() {
                return gen;
            }
            gen = commit(h, &[]);
        }
    }

    fn register(h: u64, spec: &[u8]) -> Result<i32, i32> {
        let mut n = -1;
        // SAFETY: live slice and out-pointer.
        let rc = unsafe { ffi_engine_writer_register_field(h, spec.as_ptr(), spec.len(), &mut n) };
        if rc == 0 {
            Ok(n)
        } else {
            Err(rc)
        }
    }

    fn stats(h: u64) -> [i64; STAT_COUNT] {
        let mut s = [0i64; STAT_COUNT];
        // SAFETY: `s` holds `STAT_COUNT` values.
        assert_eq!(
            unsafe { ffi_engine_writer_stats(h, s.as_mut_ptr(), STAT_COUNT) },
            0
        );
        s
    }

    fn generations(h: u64) -> Vec<i64> {
        let mut out = [0i64; 16];
        let mut n = 0;
        // SAFETY: `out` holds 16 values.
        let rc = unsafe { ffi_engine_writer_commit_generations(h, out.as_mut_ptr(), 16, &mut n) };
        assert_eq!(rc, 0);
        out[..n].to_vec()
    }

    /// An empty index with one commit, as OpenSearch's `Store.createEmpty`
    /// leaves it.
    fn empty_index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.commit().unwrap();
        tmp
    }

    fn open(tmp: &TempDir, fault_injection: u8) -> u64 {
        open_with_max_docs(tmp, fault_injection, i32::MAX)
    }

    fn open_with_max_docs(tmp: &TempDir, fault_injection: u8, max_docs: i32) -> u64 {
        open_with(tmp, 16.0, fault_injection, max_docs)
    }

    fn open_with(tmp: &TempDir, ram_mb: f64, fault_injection: u8, max_docs: i32) -> u64 {
        let path = tmp.path().to_str().unwrap();
        let mut h = 0;
        // SAFETY: live string and out-pointer.
        let rc = unsafe {
            ffi_engine_writer_open(
                path.as_ptr() as *const c_char,
                path.len(),
                ram_mb,
                fault_injection,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                max_docs,
                &mut h,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        h
    }

    fn setup(h: u64) -> Numbers {
        Numbers {
            id: register(h, &field("_id", 1, 1, 0, 0)).unwrap(),
            body: register(h, &field("body", 4, 0, 0, 0)).unwrap(),
            seq: register(h, &field("_seq_no", 0, 0, 1, 0)).unwrap(),
            soft: register(h, &field("__soft_deletes", 0, 0, 1, 1)).unwrap(),
        }
    }

    fn check(tmp: &TempDir) {
        let dir = FsDirectory::open(tmp.path());
        for r in lucene_index::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
        }
    }

    /// [`ffi_engine_writer_open_codec`]: each stored-fields choice records
    /// the codec and `.si` mode stock OpenSearch records for that
    /// `index.codec`, flushed and merged, and reads back.
    #[test]
    fn the_stored_fields_codec_is_opensearchs() {
        for (stored, level, codec, mode) in [
            (STORED_FIELDS_BEST_SPEED, 0, "Lucene104", "BEST_SPEED"),
            (
                STORED_FIELDS_BEST_COMPRESSION,
                0,
                "Lucene104",
                "BEST_COMPRESSION",
            ),
            (STORED_FIELDS_ZSTD, 3, "ZSTD104", "ZSTD"),
            (
                STORED_FIELDS_ZSTD_NO_DICT,
                -1,
                "ZSTDNODICT104",
                "ZSTD_NO_DICT",
            ),
            (
                STORED_FIELDS_ZSTD_PAGED,
                3,
                "LuceneRustZstdPaged104",
                "ZSTD_PAGED",
            ),
        ] {
            let tmp = empty_index("engine-writer-codec");
            let path = tmp.path().to_str().unwrap();
            let mut h = 0;
            // SAFETY: live string and out-pointer.
            let rc = unsafe {
                ffi_engine_writer_open_codec(
                    path.as_ptr() as *const c_char,
                    path.len(),
                    16.0,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    i32::MAX,
                    stored,
                    level,
                    &mut h,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let f = setup(h);
            for round in 0..2 {
                let mut b = Blob::default().u8(OP_ADD).i32(50);
                for i in 0..50 {
                    b = doc(
                        b,
                        &f,
                        &format!("d{round}-{i}"),
                        &["some", "text"],
                        i64::from(round * 50 + i),
                        false,
                    );
                }
                assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
                commit(h, &[]);
            }
            let mut generation = 0;
            // SAFETY: a live handle.
            assert_eq!(
                unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut generation) },
                0,
                "{}",
                crate::error::last_error()
            );
            commit(h, &[]);
            ffi_engine_writer_close(h);
            let dir = FsDirectory::open(tmp.path());
            let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
            assert_eq!(infos.segments.len(), 1);
            let seg = &infos.segments[0];
            assert_eq!(seg.codec_name, codec);
            let si = dir.open(&format!("{}.si", seg.segment_name)).unwrap();
            let si = lucene_index::segment_info::parse(&si, &seg.segment_id).unwrap();
            assert!(
                si.attributes
                    .iter()
                    .any(|(k, v)| k.ends_with("StoredFieldsFormat.mode") && v == mode),
                "{codec}: {:?}",
                si.attributes
            );
            let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
            let seg = &reader.segment_readers()[0];
            assert_eq!(seg.max_doc, 100);
            let mut ids: Vec<_> = (0..100)
                .map(
                    |d| match &seg.stored_document(d).unwrap().unwrap().fields[0].value {
                        lucene_codecs::stored_fields::FieldValue::Binary(b) => {
                            String::from_utf8(b.clone()).unwrap()
                        }
                        v => panic!("{v:?}"),
                    },
                )
                .collect();
            ids.sort();
            let mut want: Vec<_> = (0..2)
                .flat_map(|r| (0..50).map(move |i| format!("d{r}-{i}")))
                .collect();
            want.sort();
            assert_eq!(ids, want, "{codec}");
            check(&tmp);
        }
    }

    /// A bad level or stored-fields code, or a sparse index asking for other
    /// stored fields, is refused before anything opens.
    #[test]
    fn a_stored_fields_codec_it_cannot_write_is_refused() {
        let tmp = empty_index("engine-writer-codec-bad");
        let path = tmp.path().to_str().unwrap();
        for (flags, stored, level, msg) in [
            (0, STORED_FIELDS_ZSTD, 1000, "compression_level"),
            (0, 9, 0, "unknown stored fields"),
            (OPEN_SPARSE, STORED_FIELDS_ZSTD, 3, "index.sparse"),
        ] {
            let mut h = 0;
            // SAFETY: live string and out-pointer.
            let rc = unsafe {
                ffi_engine_writer_open_codec(
                    path.as_ptr() as *const c_char,
                    path.len(),
                    16.0,
                    flags,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    i32::MAX,
                    stored,
                    level,
                    &mut h,
                )
            };
            assert_eq!(rc, FfiStatus::InvalidArgument as i32);
            assert!(
                crate::error::last_error().contains(msg),
                "{}",
                crate::error::last_error()
            );
        }
    }

    /// [`OPEN_ADAPTIVE_DOC_VALUES`]: numeric doc values land in `RustAdaptive`
    /// files, read back value for value, survive a merge, and pass
    /// `CheckIndex`.
    #[test]
    fn an_adaptive_writer_writes_rust_adaptive_doc_values() {
        let tmp = empty_index("engine-writer-adaptive");
        let h = open(&tmp, OPEN_ADAPTIVE_DOC_VALUES);
        let f = setup(h);
        let n = 600i64;
        for round in 0..2 {
            for i in 0..n / 2 {
                let seq = round * n / 2 + i;
                let id = format!("d{seq}");
                let b = doc(
                    Blob::default().u8(OP_ADD).i32(1),
                    &f,
                    &id,
                    &["w"],
                    seq * 3,
                    false,
                );
                assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            }
            commit(h, &[]);
        }
        let read_seq_nos = |tmp: &TempDir| {
            let dir = FsDirectory::open(tmp.path());
            let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
            let mut all = Vec::new();
            for seg in reader.segment_readers() {
                let number = seg
                    .field_infos()
                    .fields
                    .iter()
                    .find(|fi| fi.name == "_seq_no")
                    .unwrap()
                    .number;
                let (meta, data) = seg.doc_values_for_field(number).unwrap();
                let entry = meta.numeric_entry(number).unwrap();
                assert!(entry.adaptive.is_some());
                let mut r = lucene_codecs::doc_values::NumericReader::new(data, entry);
                for d in 0..seg.max_doc {
                    all.push(r.value(d).unwrap().unwrap());
                }
            }
            all.sort_unstable();
            all
        };
        let want: Vec<i64> = (0..n).map(|s| s * 3).collect();
        assert_eq!(read_seq_nos(&tmp), want);
        let names: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(
            names.iter().any(|n| n.contains("_RustAdaptive_0.dvd")),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("_Lucene90_0.dv")),
            "{names:?}"
        );
        check(&tmp);
        // A forced merge rewrites both segments in the same format.
        let mut gen = 0;
        // SAFETY: a live handle and out-pointer.
        let rc = unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        commit_settled(h);
        ffi_engine_writer_close(h);
        assert_eq!(read_seq_nos(&tmp), want);
        check(&tmp);
    }

    /// [`OPEN_ZSTD_TERMS`]: every postings field lands in `RustZstd104`, a
    /// term lookup finds each document, and a forced merge keeps the format.
    #[test]
    fn a_zstd_terms_writer_writes_rust_zstd_postings() {
        let tmp = empty_index("engine-writer-zstd-terms");
        let h = open(&tmp, OPEN_ZSTD_TERMS);
        let f = setup(h);
        let n = 600i64;
        for round in 0..2 {
            for i in 0..n / 2 {
                let seq = round * n / 2 + i;
                let id = format!("d{seq}");
                let b = doc(
                    Blob::default().u8(OP_ADD).i32(1),
                    &f,
                    &id,
                    &["w"],
                    seq,
                    false,
                );
                assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            }
            commit(h, &[]);
        }
        let files = |tmp: &TempDir| -> Vec<String> {
            std::fs::read_dir(tmp.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect()
        };
        let ids_found = |tmp: &TempDir| {
            let dir = FsDirectory::open(tmp.path());
            let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
            (0..n)
                .filter(|seq| {
                    reader
                        .doc_freq("_id", format!("d{seq}").as_bytes())
                        .unwrap()
                        == 1
                })
                .count()
        };
        assert_eq!(ids_found(&tmp), n as usize);
        let names = files(&tmp);
        assert!(
            names.iter().any(|n| n.ends_with("_RustZstd104_0.tim")),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|n| n.ends_with("_Lucene104_0.tim")),
            "{names:?}"
        );
        check(&tmp);
        let mut gen = 0;
        // SAFETY: a live handle and out-pointer.
        let rc = unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        commit_settled(h);
        ffi_engine_writer_close(h);
        assert_eq!(ids_found(&tmp), n as usize);
        check(&tmp);
    }

    /// [`OPEN_SPARSE`]: segments record neural-search's codec and carry its
    /// files, and a field keeps the attributes its spec gave it.
    #[test]
    fn a_sparse_writer_writes_neural_search_segments() {
        let tmp = empty_index("engine-writer-sparse");
        let h = open(&tmp, OPEN_SPARSE);
        let f = setup(h);
        // The spec less its empty attribute list, then one attribute.
        let mut spec = field("emb", 2, 1, 2, 0);
        spec.truncate(spec.len() - 4);
        spec.extend_from_slice(
            &Blob::default()
                .i32(1)
                .bytes(b"sparse_vector_field")
                .bytes(b"true")
                .0,
        );
        let emb = register(h, &spec).unwrap();
        // `_id`, and `emb` as neural-search's mapper writes it: one
        // `FeatureField` term per token, the forward index as binary doc values.
        let freq = |v: f32| (v.to_bits() >> 15) as i32;
        let mut forward = Vec::new();
        for (t, v) in [(1i32, 0.5f32), (7, 2.0)] {
            forward.extend_from_slice(&t.to_be_bytes());
            forward.extend_from_slice(&v.to_be_bytes());
        }
        let add = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .i32(1)
            .i32(f.id)
            .u8(1)
            .bytes(b"a")
            .i32(2)
            .i32(f.id)
            .u8(0)
            .i32(1)
            .bytes(b"a")
            .i32(1)
            .u8(0)
            .i32(emb)
            .u8(0)
            .i32(2)
            .bytes(b"1")
            .i32(freq(0.5))
            .u8(0)
            .bytes(b"7")
            .i32(freq(2.0))
            .u8(0)
            .i32(2)
            .i32(f.seq)
            .u8(0)
            .i64(0)
            .i32(emb)
            .u8(1)
            .bytes(&forward)
            .i32(0);
        assert_eq!(apply(h, &add.0), 0, "{}", crate::error::last_error());
        commit(h, &[]);
        ffi_engine_writer_close(h);
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let seg = &reader.segment_readers()[0];
        assert!(seg
            .codec_file(lucene_sparse::postings::TERMS_EXTENSION)
            .is_some());
        let emb = seg
            .field_infos()
            .fields
            .iter()
            .find(|i| i.name == "emb")
            .unwrap();
        assert!(emb
            .attributes
            .iter()
            .any(|(k, v)| k == "sparse_vector_field" && v == "true"));
        check(&tmp);
    }

    /// A partitioned writer: each operation's partition decides the
    /// segment, every segment records its partition (OpenSearch's `bucket`,
    /// on the `.si` and on `_id`), an update's earlier version is soft
    /// deleted whatever partition holds it, and a forced merge leaves one
    /// segment per partition.
    #[test]
    fn a_partitioned_writer_keeps_each_partition_in_its_own_segments() {
        let tmp = empty_index("engine-writer-partitioned");
        let h = open_with(&tmp, 16.0, OPEN_PARTITIONED, i32::MAX);
        let f = setup(h);
        let add = |p: &str, id: &str, seq: i64| {
            doc(
                Blob::default()
                    .u8(OP_ADD_PARTITIONED)
                    .bytes(p.as_bytes())
                    .i32(1),
                &f,
                id,
                &["x"],
                seq,
                false,
            )
        };
        for (seq, (p, id)) in [("a", "1"), ("b", "2"), ("a", "3"), ("c", "4"), ("b", "5")]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                apply(h, &add(p, id, seq as i64).0),
                0,
                "{}",
                crate::error::last_error()
            );
        }
        commit(h, &[]);
        // "1" moves from partition a to c: the old one is soft deleted.
        let update = doc(
            Blob::default()
                .u8(OP_SOFT_UPDATE_PARTITIONED)
                .bytes(b"c")
                .bytes(b"_id")
                .bytes(b"1")
                .bytes(b"__soft_deletes")
                .i64(1)
                .i32(1),
            &f,
            "1",
            &["y"],
            5,
            false,
        );
        assert_eq!(apply(h, &update.0), 0, "{}", crate::error::last_error());
        commit(h, &[]);
        let buckets = || {
            let dir = FsDirectory::open(tmp.path());
            let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
            let mut out: Vec<(String, i32, i32)> = infos
                .segments
                .iter()
                .map(|sci| {
                    let name = &sci.segment_name;
                    let si = lucene_index::segment_info::parse(
                        &dir.open(&format!("{name}.si")).unwrap(),
                        &sci.segment_id,
                    )
                    .unwrap();
                    let fnm = lucene_codecs::field_infos::parse(
                        &dir.open(&format!("{name}.fnm")).unwrap(),
                        &sci.segment_id,
                        "",
                    )
                    .unwrap();
                    let bucket = |a: &[(String, String)]| {
                        a.iter()
                            .find(|(k, _)| k == "bucket")
                            .map(|(_, v)| v.clone())
                    };
                    let on_id = fnm.field_by_name("_id").and_then(|f| bucket(&f.attributes));
                    let on_si = bucket(&si.attributes).unwrap();
                    assert_eq!(on_id.as_deref(), Some(on_si.as_str()));
                    (on_si, si.doc_count, sci.soft_del_count)
                })
                .collect();
            out.sort();
            out
        };
        let before = buckets();
        assert!(before
            .iter()
            .all(|(b, _, _)| ["a", "b", "c"].contains(&b.as_str())));
        let soft: i32 = before.iter().map(|(_, _, s)| s).sum();
        assert_eq!(soft, 1, "the first version of 1, in partition a");
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let after: Vec<(String, i32)> = buckets().into_iter().map(|(b, n, _)| (b, n)).collect();
        assert_eq!(
            after,
            [
                ("a".to_string(), 2),
                ("b".to_string(), 2),
                ("c".to_string(), 2)
            ],
            "soft-deleted documents are kept without a retention policy"
        );
        ffi_engine_writer_close(h);
        check(&tmp);
    }

    /// `IndexWriter.tooManyDocs`: an add past the JVM's `maxDocs` fails that
    /// operation only -- Java's `IllegalArgumentException` -- and the writer
    /// carries on.
    #[test]
    fn an_add_past_max_docs_is_refused_without_failing_the_writer() {
        let tmp = empty_index("engine-writer-max-docs");
        let h = open_with_max_docs(&tmp, 0, 1);
        let f = setup(h);
        let add = |id: &str| doc(Blob::default().u8(OP_ADD).i32(1), &f, id, &["x"], 0, false);
        assert_eq!(apply(h, &add("a").0), 0);
        assert_eq!(apply(h, &add("b").0), FfiStatus::InvalidArgument.code());
        assert!(crate::error::last_error().contains("cannot exceed 1"));
        commit(h, &[]);
        assert_eq!(stats(h)[STAT_SEGMENTS], 1);
        ffi_engine_writer_close(h);
        check(&tmp);
    }

    /// A buffered document is charged for the heap its postings, compressed
    /// stored values and packed fields hold: 70 positions with offsets come
    /// to a few bytes each in the postings' byte slices (about 1.8 KB a
    /// document when they were kept as ints and structs).
    #[test]
    fn a_buffered_document_charges_its_packed_size() {
        let tmp = empty_index("engine-writer-ram-per-doc");
        // a buffer large enough that nothing flushes
        let h = open_with(&tmp, 1024.0, 0, i32::MAX);
        let f = setup(h);
        let vocab: Vec<String> = (0..20000).map(|i| format!("t{i:x}")).collect();
        let mut x: u64 = 7;
        let n = 20000;
        let source = "x".repeat(400);
        for d in 0..n {
            let mut words = Vec::new();
            for _ in 0..70 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (x >> 33) as f64 / (1u64 << 31) as f64;
                words.push(vocab[((r * r * r) * 20000.0) as usize].as_str());
            }
            let id = format!("{d}");
            let b = Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .i32(1)
                .i32(f.id)
                .u8(1)
                .bytes(source.as_bytes())
                .i32(2)
                .i32(f.id)
                .u8(0)
                .i32(1)
                .bytes(id.as_bytes())
                .i32(1)
                .u8(0);
            let mut terms: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
            for (p, w) in words.iter().enumerate() {
                terms.entry(w).or_default().push(p as i32);
            }
            let mut b = b
                .i32(f.body)
                .u8(1)
                .i64(words.len() as i64)
                .i32(terms.len() as i32);
            for (w, ps) in &terms {
                b = b.bytes(w.as_bytes()).i32(ps.len() as i32).u8(3);
                for &p in ps {
                    b = b.i32(p);
                }
                for &p in ps {
                    b = b.i32(p * 4).i32(p * 4 + 3);
                }
            }
            let b = b.i32(1).i32(f.seq).u8(0).i64(d).i32(0);
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        }
        let s = stats(h);
        assert_eq!(s[STAT_SEGMENTS], 0, "nothing flushed");
        let per_doc = s[STAT_RAM_BYTES] / n;
        assert!(per_doc < 850, "{per_doc} bytes per document");
        ffi_engine_writer_close(h);
    }

    /// A buffer writes its stored fields to its segment's `.fdt` as they
    /// arrive, so they are not charged as buffered RAM; a sweep for
    /// unreferenced files leaves that file alone, the commit reads them back
    /// whole, and a close that discards the buffer deletes the file.
    #[test]
    fn a_buffer_writes_its_stored_fields_to_disk_as_it_fills() {
        let fdt_files = |tmp: &TempDir| -> Vec<String> {
            FsDirectory::open(tmp.path())
                .list_all()
                .unwrap()
                .into_iter()
                .filter(|n| n.ends_with(".fdt"))
                .collect()
        };
        let add_docs = |h: u64, f: &Numbers, n: usize| {
            let mut x: u64 = 0x2545_f491_4f6c_dd1d;
            for d in 0..n {
                let source: Vec<u8> = (0..2000)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        b'a' + (x % 26) as u8
                    })
                    .collect();
                let id = format!("{d}");
                let b = Blob::default()
                    .u8(OP_ADD)
                    .i32(1)
                    .i32(1)
                    .i32(f.id)
                    .u8(1)
                    .bytes(&source)
                    .i32(1)
                    .i32(f.id)
                    .u8(0)
                    .i32(1)
                    .bytes(id.as_bytes())
                    .i32(1)
                    .u8(0)
                    .i32(1)
                    .i32(f.seq)
                    .u8(0)
                    .i64(d as i64)
                    .i32(0);
                assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            }
        };
        let n = 3000;
        let tmp = empty_index("engine-writer-stored-on-disk");
        let h = open_with(&tmp, 1024.0, 0, i32::MAX);
        let f = setup(h);
        add_docs(h, &f, n);
        let s = stats(h);
        assert_eq!(s[STAT_SEGMENTS], 0, "nothing flushed");
        assert!(
            s[STAT_RAM_BYTES] / (n as i64) < 1000,
            "{} bytes per document",
            s[STAT_RAM_BYTES] / (n as i64)
        );
        assert!(
            !fdt_files(&tmp).is_empty(),
            "the .fdt is on disk before the flush"
        );
        // A sweep for unreferenced files leaves the buffer's file alone.
        lookup(h)
            .unwrap()
            .writer()
            .with_writer(|iw| iw.delete_unused_files())
            .unwrap();
        commit(h, &[]);
        let stored: Vec<String> = stored_ids_and_bp(&tmp)
            .into_iter()
            .flat_map(|(values, _)| values)
            .collect();
        assert_eq!(stored.len(), n);
        assert!(stored.iter().all(|v| v.len() == 2000));
        check(&tmp);
        ffi_engine_writer_close(h);

        let tmp = empty_index("engine-writer-stored-discarded");
        let h = open_with(&tmp, 1024.0, 0, i32::MAX);
        let f = setup(h);
        add_docs(h, &f, n);
        assert!(!fdt_files(&tmp).is_empty());
        ffi_engine_writer_close(h);
        assert!(fdt_files(&tmp).is_empty(), "{:?}", fdt_files(&tmp));
    }

    /// The ids stored in the newest commit's segments, in doc-id order, and
    /// each segment's `bp.reordered` diagnostic.
    fn stored_ids_and_bp(tmp: &TempDir) -> Vec<(Vec<String>, Option<String>)> {
        let dir = FsDirectory::open(tmp.path());
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        infos
            .segments
            .iter()
            .map(|sci| {
                let name = &sci.segment_name;
                let si = lucene_index::segment_info::parse(
                    &dir.open(&format!("{name}.si")).unwrap(),
                    &sci.segment_id,
                )
                .unwrap();
                let fdt = dir.open(&format!("{name}.fdt")).unwrap();
                let fdx = dir.open(&format!("{name}.fdx")).unwrap();
                let fdm = dir.open(&format!("{name}.fdm")).unwrap();
                let reader =
                    lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "")
                        .unwrap();
                let ids = (0..reader.max_doc())
                    .map(|d| match &reader.document(d).unwrap().fields[0].value {
                        FieldValue::String(s) => s.clone(),
                        FieldValue::Binary(b) => String::from_utf8(b.clone()).unwrap(),
                        other => panic!("{other:?}"),
                    })
                    .collect();
                let bp = si
                    .diagnostics
                    .iter()
                    .find(|(k, _)| k == "bp.reordered")
                    .map(|(_, v)| v.clone());
                (ids, bp)
            })
            .collect()
    }

    /// `index.lucene_rust.bp_reorder.*`: a forced merge renumbers the
    /// documents so the ones sharing terms sit together, and records it in
    /// the merged segment's diagnostics; turned off, a forced merge keeps
    /// the documents' order again.
    #[test]
    fn bp_reordering_clusters_the_documents_of_a_forced_merge() {
        let tmp = empty_index("engine-writer-bp");
        let h = open(&tmp, 0);
        let f = setup(h);
        let set_bp = |enabled: u8, fields: &[u8]| {
            // SAFETY: a live slice.
            unsafe {
                ffi_engine_writer_set_bp_reorder(
                    h,
                    enabled,
                    1,
                    1.0,
                    2,
                    20,
                    64.0,
                    1,
                    0.0,
                    fields.as_ptr(),
                    fields.len(),
                )
            }
        };
        let force_merge = || {
            let mut generation = 0i64;
            // SAFETY: a live out-pointer.
            let rc = unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut generation) };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
        };
        // Two topics, scattered.
        let topic_of = |i: usize| (i.wrapping_mul(2_654_435_761) >> 9) % 2;
        let add = |i: usize| {
            let (words, prefix): (&[&str], &str) = if topic_of(i) == 0 {
                (&["apple", "banana", "cherry"], "a")
            } else {
                (&["xenon", "yttrium", "zinc"], "x")
            };
            let id = format!("{prefix}{i:03}");
            let b = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                &id,
                words,
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        for i in 0..256 {
            add(i);
            if i % 64 == 63 {
                commit(h, &[]);
            }
        }
        assert_eq!(stats(h)[STAT_SEGMENTS], 4);
        assert_eq!(set_bp(1, b"body"), 0, "{}", crate::error::last_error());
        force_merge();
        let segments = stored_ids_and_bp(&tmp);
        assert_eq!(segments.len(), 1);
        let (ids, bp) = &segments[0];
        assert_eq!(bp.as_deref(), Some("true"));
        assert_eq!(ids.len(), 256);
        let switches = |ids: &[String]| ids.windows(2).filter(|w| w[0][..1] != w[1][..1]).count();
        let before = (0..256).map(topic_of).collect::<Vec<_>>();
        let before = before.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(before > 64, "{before} switches before: a weak test");
        assert!(
            switches(ids) < 8,
            "{} topic switches: not clustered",
            switches(ids)
        );
        check(&tmp);

        // Off again: the next forced merge concatenates in order.
        assert_eq!(set_bp(0, b""), 0);
        for i in 256..320 {
            add(i);
        }
        commit(h, &[]);
        force_merge();
        let segments = stored_ids_and_bp(&tmp);
        assert_eq!(segments.len(), 1);
        let (ids, bp) = &segments[0];
        assert_eq!(*bp, None);
        assert_eq!(ids.len(), 320);
        check(&tmp);

        // Refused settings change nothing.
        let bad = FfiStatus::InvalidArgument.code();
        // SAFETY: no fields.
        let refuse = |min_doc_freq: i32,
                      max_doc_freq: f32,
                      ram: f64,
                      min_merge_docs: i32,
                      ratio: f32| unsafe {
            ffi_engine_writer_set_bp_reorder(
                h,
                1,
                min_doc_freq,
                max_doc_freq,
                32,
                20,
                ram,
                min_merge_docs,
                ratio,
                std::ptr::null(),
                0,
            )
        };
        assert_eq!(refuse(0, 1.0, 64.0, 1, 0.0), bad);
        assert_eq!(refuse(1, 0.0, 64.0, 1, 0.0), bad);
        assert_eq!(refuse(1, 1.0, 0.0, 1, 0.0), bad);
        assert_eq!(refuse(1, 1.0, f64::NAN, 1, 0.0), bad);
        assert_eq!(refuse(1, 1.0, 64.0, 0, 0.0), bad);
        assert_eq!(refuse(1, 1.0, 64.0, 1, 1.0), bad);
        assert_eq!(set_bp(1, &[0xff, 0xfe]), bad);
        lookup(h).unwrap().writer().with_writer(|w| {
            assert!(w.pluggable_merge_policy().is_none());
        });
        ffi_engine_writer_close(h);
    }

    /// A sparse shard refuses BP reordering; turning it off is still fine.
    #[test]
    fn a_sparse_writer_refuses_bp_reordering() {
        let tmp = empty_index("engine-writer-sparse-bp");
        let h = open(&tmp, OPEN_SPARSE);
        // SAFETY: no fields.
        let set = |enabled: u8| unsafe {
            ffi_engine_writer_set_bp_reorder(
                h,
                enabled,
                1,
                1.0,
                2,
                20,
                64.0,
                1,
                0.0,
                std::ptr::null(),
                0,
            )
        };
        assert_eq!(set(1), FfiStatus::InvalidArgument.code());
        assert!(crate::error::last_error().contains("sparse"));
        assert_eq!(set(0), 0);
        ffi_engine_writer_close(h);
    }

    /// With BP on, the merge policy settings and merge-on-flush rebuild the
    /// policies around the new settings, and a refresh's merges are picked
    /// without reordering.
    #[test]
    fn bp_reordering_follows_the_merge_policy_settings() {
        let tmp = empty_index("engine-writer-bp-policy");
        let h = open(&tmp, 0);
        // SAFETY: no fields.
        let rc = unsafe {
            ffi_engine_writer_set_bp_reorder(h, 1, 1, 1.0, 2, 20, 64.0, 1, 0.0, std::ptr::null(), 0)
        };
        assert_eq!(rc, 0);
        lookup(h).unwrap().writer().with_writer(|w| {
            assert!(w.pluggable_merge_policy().is_some());
        });
        let mb = 1024 * 1024;
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 30, 10.0, 5 * 1024 * mb, 16 * mb, 20.0, 10.0),
            0
        );
        assert_eq!(ffi_engine_writer_set_merge_on_flush(h, 1000, 0), 0);
        assert_eq!(ffi_engine_writer_set_merge_on_flush(h, 1000, 1), 0);
        lookup(h).unwrap().writer().with_writer(|w| {
            assert!(w.pluggable_merge_policy().is_some());
            assert_eq!(w.merge_policy().unwrap().max_merge_at_once, 30);
        });
        ffi_engine_writer_close(h);
    }

    /// The engine keeps the commits it has always written: merge-on-commit
    /// off (as OpenSearch configures Lucene) and loose segment files, set
    /// explicitly rather than inherited from `IndexWriter`'s defaults.
    #[test]
    fn the_engine_writer_turns_merge_on_commit_off_and_writes_loose_segments() {
        let tmp = empty_index("engine-writer-config");
        let h = open(&tmp, 0);
        let engine = lookup(h).unwrap();
        engine.writer().with_writer(|w| {
            assert_eq!(w.max_full_flush_merge_wait_millis(), 0);
            assert!(!w.use_compound_file());
        });
        ffi_engine_writer_close(h);
    }

    /// What the test merge upcall heard: `(handle, event, docs, max segments,
    /// failure kind, names)`.
    type HeardMerge = (u64, i32, i64, i32, i32, String);
    static HEARD: Mutex<Vec<HeardMerge>> = Mutex::new(Vec::new());

    unsafe extern "C" fn record_merge(
        handle: u64,
        event: i32,
        _id: i64,
        docs: i64,
        _bytes: i64,
        max_segments: i32,
        _took: i64,
        _stopped: i64,
        _throttled: i64,
        kind: i32,
        names: *const u8,
        names_len: usize,
        _failure: *const u8,
        _failure_len: usize,
    ) {
        // SAFETY: the upcall's contract.
        let names = unsafe { std::slice::from_raw_parts(names, names_len) };
        let names = String::from_utf8(names.to_vec()).unwrap();
        HEARD
            .lock()
            .unwrap()
            .push((handle, event, docs, max_segments, kind, names));
    }

    /// The merge upcall hears a forced merge before and after, with its
    /// sources; the writer's merge scheduler stops stalling producers, and
    /// the merge state reads without the writer's lock.
    #[test]
    fn the_merge_upcall_hears_merges_and_the_merge_state_reads() {
        // SAFETY: `record_merge` has the upcall's contract and lives forever.
        assert_eq!(
            unsafe { ffi_engine_set_merge_upcall(Some(record_merge)) },
            0
        );
        // SAFETY: a null upcall is refused.
        assert_eq!(
            unsafe { ffi_engine_set_merge_upcall(None) },
            FfiStatus::NullPointer.code()
        );
        let tmp = empty_index("engine-writer-merge-upcall");
        let h = open(&tmp, 0);
        assert!(!lookup(h).unwrap().handle.merges.stall_producers());
        let f = setup(h);
        for (i, id) in ["a", "b", "c"].into_iter().enumerate() {
            let d = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                id,
                &["x"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &d.0), 0);
            commit(h, &[]);
        }
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let heard: Vec<_> = HEARD
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.0 == h)
            .cloned()
            .collect();
        let forced: Vec<_> = heard.iter().filter(|e| e.3 == 1).collect();
        assert_eq!(forced.len(), 2, "{heard:?}");
        assert_eq!(forced[0].1, MERGE_EVENT_BEFORE);
        assert_eq!(forced[1].1, MERGE_EVENT_AFTER);
        assert_eq!(forced[1].2, 3);
        assert_eq!(forced[1].4, MERGE_OK);
        assert_eq!(forced[1].5.lines().count(), 3);
        let (mut pending, mut rate) = (9u8, 0.0);
        // SAFETY: live out-pointers.
        assert_eq!(
            unsafe { ffi_engine_writer_merge_state(h, &mut pending, &mut rate) },
            0
        );
        assert_eq!(pending, 0);
        assert!(rate > 0.0);
        // SAFETY: null out-pointers are refused before any write.
        assert_eq!(
            unsafe { ffi_engine_writer_merge_state(h, std::ptr::null_mut(), &mut rate) },
            FfiStatus::NullPointer.code()
        );
        ffi_engine_writer_close(h);
    }

    /// `index.use_compound_file`, `index.compound_format`,
    /// `index.check_pending_flush.enabled` and the shuffled forced merge reach
    /// the writer; a ratio or cap OpenSearch would refuse is refused.
    #[test]
    fn the_shard_writer_options_reach_the_writer() {
        let tmp = empty_index("engine-writer-options");
        let h = open(&tmp, 0);
        let mb = 1024 * 1024;
        assert_eq!(ffi_engine_writer_set_options(h, 1, 0.1, 5 * mb, 0, 1), 0);
        let engine = lookup(h).unwrap();
        assert!(!engine.writer().check_pending_flush_on_update());
        engine.writer().with_writer(|w| {
            assert!(w.use_compound_file());
            assert_eq!(
                w.merge_compound_file_settings(),
                Some(CompoundFileSettings::new(0.1, 5 * mb))
            );
            assert!(w.shuffle_forced_merges());
        });
        let bad = FfiStatus::InvalidArgument.code();
        assert_eq!(ffi_engine_writer_set_options(h, 1, 1.5, mb, 1, 1), bad);
        assert_eq!(ffi_engine_writer_set_options(h, 1, f64::NAN, mb, 1, 1), bad);
        assert_eq!(ffi_engine_writer_set_options(h, 1, 0.1, -1, 1, 1), bad);
        assert_eq!(ffi_engine_writer_set_options(h, 0, 0.0, 0, 1, 0), 0);
        assert!(engine.writer().check_pending_flush_on_update());
        engine.writer().with_writer(|w| {
            assert!(!w.use_compound_file());
            assert!(!w.shuffle_forced_merges());
        });
        drop(engine);
        ffi_engine_writer_close(h);
    }

    /// `OpenSearchTieredMergePolicy`: a forced merge ignores the policy's
    /// merged-size cap and leaves the policy as it found it; with the
    /// shuffle on, the merged segment is marked interleaved.
    #[test]
    fn a_forced_merge_has_no_size_cap_and_restores_the_policy() {
        let tmp = empty_index("engine-writer-forced-cap");
        let h = open(&tmp, 0);
        let f = setup(h);
        for (i, id) in ["a", "b", "c", "d"].into_iter().enumerate() {
            let d = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                id,
                &["x"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &d.0), 0);
            commit(h, &[]);
        }
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 10, 10.0, 1, 1, 20.0, 10.0),
            0
        );
        assert_eq!(ffi_engine_writer_set_options(h, 0, 0.1, i64::MAX, 1, 1), 0);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 2, 0, &mut gen) },
            0
        );
        let dir = FsDirectory::open(tmp.path());
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        assert!(
            infos.segments.len() <= 2,
            "{} segments",
            infos.segments.len()
        );
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        assert_eq!(infos.segments.len(), 1);
        let sci = &infos.segments[0];
        let si = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = lucene_index::segment_info::parse(&si, &sci.segment_id).unwrap();
        assert!(si
            .diagnostics
            .iter()
            .any(|(k, _)| k == lucene_index::merge::SHUFFLE_MERGE_KEY));
        lookup(h).unwrap().writer().with_writer(|w| {
            assert_eq!(w.merge_policy().unwrap().max_merged_segment_size, 1);
        });
        check(&tmp);
        ffi_engine_writer_close(h);
    }

    /// The segments the test warm upcall saw, and whether it fails them.
    static WARMED: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static WARM_FAILS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    unsafe extern "C" fn record_warm(
        _handle: u64,
        name: *const u8,
        len: usize,
        id: *const u8,
    ) -> i32 {
        // SAFETY: the upcall's contract.
        let (name, id) = unsafe {
            (
                std::slice::from_raw_parts(name, len),
                std::slice::from_raw_parts(id, lucene_store::codec_util::ID_LENGTH),
            )
        };
        assert!(id.iter().any(|&b| b != 0));
        WARMED
            .lock()
            .unwrap()
            .push(String::from_utf8(name.to_vec()).unwrap());
        i32::from(WARM_FAILS.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The merged-segment warmer upcall sees a forced merge's segment before
    /// it is published, and a warmer failure fails the merge.
    #[test]
    fn the_warm_upcall_sees_merged_segments() {
        // SAFETY: `record_warm` has the upcall's contract and lives forever.
        assert_eq!(unsafe { ffi_engine_set_warm_upcall(Some(record_warm)) }, 0);
        // SAFETY: a null upcall is refused.
        assert_eq!(
            unsafe { ffi_engine_set_warm_upcall(None) },
            FfiStatus::NullPointer.code()
        );
        let tmp = empty_index("engine-writer-warm");
        let h = open(&tmp, 0);
        let f = setup(h);
        let add = |ids: &[&str]| {
            for (i, id) in ids.iter().enumerate() {
                let d = doc(
                    Blob::default().u8(OP_ADD).i32(1),
                    &f,
                    id,
                    &["x"],
                    i as i64,
                    false,
                );
                assert_eq!(apply(h, &d.0), 0);
                commit(h, &[]);
            }
        };
        add(&["a", "b", "c"]);
        assert_eq!(ffi_engine_writer_set_merged_segment_warmer(h, 1), 0);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let dir = FsDirectory::open(tmp.path());
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        assert_eq!(infos.segments.len(), 1);
        assert!(WARMED
            .lock()
            .unwrap()
            .contains(&infos.segments[0].segment_name));
        add(&["d"]);
        WARM_FAILS.store(true, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: live out-pointer.
        let failed = unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) };
        WARM_FAILS.store(false, std::sync::atomic::Ordering::Relaxed);
        assert_ne!(failed, 0, "a failed warm fails the merge");
        ffi_engine_writer_close(h);
        let h = open(&tmp, 0);
        assert_eq!(ffi_engine_writer_set_merged_segment_warmer(h, 0), 0);
        ffi_engine_writer_close(h);
    }

    /// `index.merge.enabled: false` stops every merge, forced ones included;
    /// `log_byte_size` then decides them, and the tiered policy again once
    /// set back. Settings OpenSearch would refuse are refused.
    #[test]
    fn a_pluggable_merge_policy_decides_the_shards_merges() {
        let tmp = empty_index("engine-writer-pluggable-policy");
        let h = open(&tmp, 0);
        let f = setup(h);
        for (i, id) in ["a", "b", "c", "d"].into_iter().enumerate() {
            let d = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                id,
                &["x"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &d.0), 0);
            commit(h, &[]);
        }
        let set = |kind, factor, ratio| {
            ffi_engine_writer_set_pluggable_merge_policy(
                h,
                kind,
                factor,
                0,
                i64::MAX,
                i64::MAX,
                i32::MAX,
                ratio,
                i64::MAX,
            )
        };
        let dir = FsDirectory::open(tmp.path());
        let segments = || {
            lucene_index::segment_infos::read_latest(&dir)
                .unwrap()
                .segments
                .len()
        };
        let before = segments();
        assert!(before >= 2, "{before} segments");
        assert_eq!(set(MERGE_POLICY_NONE, 0, 0.0), 0);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        assert_eq!(segments(), before, "merges are off");
        assert_eq!(set(MERGE_POLICY_LOG_BYTE_SIZE, 2, 0.1), 0);
        assert_eq!(set(MERGE_POLICY_LOG_BYTE_SIZE, 2, 0.1), 0, "unchanged");
        assert_eq!(ffi_engine_writer_set_merge_on_flush(h, 500, 1), 0);
        lookup(h).unwrap().writer().with_writer(|w| {
            assert!(w.pluggable_merge_policy().is_some());
        });
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        assert_eq!(segments(), 1);
        assert_eq!(set(MERGE_POLICY_TIERED, 0, 0.0), 0);
        lookup(h).unwrap().writer().with_writer(|w| {
            assert!(w.pluggable_merge_policy().is_none());
        });
        let invalid = FfiStatus::InvalidArgument.code();
        assert_eq!(set(MERGE_POLICY_LOG_BYTE_SIZE, 1, 0.1), invalid);
        assert_eq!(set(MERGE_POLICY_LOG_BYTE_SIZE, 2, 1.5), invalid);
        assert_eq!(set(9, 2, 0.1), invalid);
        assert_eq!(
            ffi_engine_writer_set_pluggable_merge_policy(
                h,
                MERGE_POLICY_LOG_BYTE_SIZE,
                2,
                -1,
                0,
                0,
                0,
                0.1,
                0
            ),
            invalid
        );
        assert_eq!(
            ffi_engine_writer_set_pluggable_merge_policy(
                h,
                MERGE_POLICY_LOG_BYTE_SIZE,
                2,
                0,
                0,
                0,
                -1,
                0.1,
                0
            ),
            invalid
        );
        assert_eq!(
            ffi_engine_writer_set_pluggable_merge_policy(
                h,
                MERGE_POLICY_LOG_BYTE_SIZE,
                2,
                0,
                0,
                0,
                0,
                0.1,
                -1
            ),
            invalid
        );
        check(&tmp);
        ffi_engine_writer_close(h);
    }

    /// `index.merge.policy.*` reaches the writer's tiered policy, and the
    /// merge-on-flush policy set afterwards wraps those settings; values
    /// OpenSearch's settings would refuse are refused.
    #[test]
    fn the_shard_merge_policy_settings_reach_the_writer() {
        let tmp = empty_index("engine-writer-merge-policy");
        let h = open(&tmp, 0);
        let mb = 1024 * 1024;
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 30, 10.0, 5 * 1024 * mb, 16 * mb, 20.0, 10.0),
            0
        );
        lookup(h).unwrap().writer().with_writer(|w| {
            let p = w.merge_policy().unwrap();
            assert_eq!(p.max_merge_at_once, 30);
            assert_eq!(p.segments_per_tier, 10.0);
            assert_eq!(p.max_merged_segment_size, 5 * 1024 * mb as u64);
            assert_eq!(p.floor_segment_size, 16 * mb as u64);
            assert_eq!(p.deletes_pct_allowed, 20.0);
            assert_eq!(p.force_merge_deletes_pct_allowed, 10.0);
        });
        assert_eq!(ffi_engine_writer_set_merge_on_flush(h, 1000, 1), 0);
        let bad = FfiStatus::InvalidArgument.code();
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 1, 10.0, mb, mb, 20.0, 10.0),
            bad
        );
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 10, 1.5, mb, mb, 20.0, 10.0),
            bad
        );
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 10, f64::NAN, mb, mb, 20.0, 10.0),
            bad
        );
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 10, 10.0, -1, mb, 20.0, 10.0),
            bad
        );
        assert_eq!(
            ffi_engine_writer_set_merge_policy(h, 10, 10.0, mb, mb, 60.0, 10.0),
            bad
        );
        lookup(h).unwrap().writer().with_writer(|w| {
            assert_eq!(w.merge_policy().unwrap().max_merge_at_once, 30);
        });
        ffi_engine_writer_close(h);
    }

    /// A document whose analyzer gave `body` payloads -- Lucene's
    /// `MockAnalyzer` does so on some seeds, which OpenSearch's engine tests
    /// draw -- is indexed with them (term flag 4, one length-prefixed payload
    /// per occurrence, empty for none), and its segment stores payloads for
    /// that field alone. A term flag this layout does not define is refused.
    #[test]
    fn a_document_with_payloads_is_indexed_with_them() {
        let tmp = empty_index("engine-writer-payloads");
        let h = open(&tmp, 0);
        let f = setup(h);
        let words: [(&str, &[u8]); 2] = [("x", b"px"), ("y", b"")];
        let doc_with = |flags: u8| {
            let mut b = Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .i32(1)
                .i32(f.id)
                .u8(1)
                .bytes(b"a")
                .i32(2)
                .i32(f.id)
                .u8(0)
                .i32(1)
                .bytes(b"a")
                .i32(1)
                .u8(0)
                .i32(f.body)
                .u8(1)
                .i64(2)
                .i32(2);
            for (p, (w, payload)) in words.iter().enumerate() {
                b = b
                    .bytes(w.as_bytes())
                    .i32(1)
                    .u8(flags)
                    .i32(p as i32)
                    .i32(p as i32 * 2)
                    .i32(p as i32 * 2 + 1)
                    .bytes(payload);
            }
            b.i32(1).i32(f.seq).u8(0).i64(0).i32(0)
        };
        assert_eq!(apply(h, &doc_with(3 | 8).0), FfiStatus::Decode.code());
        assert!(crate::error::last_error().contains("term flags"));
        assert_eq!(
            apply(h, &doc_with(3 | 4).0),
            0,
            "{}",
            crate::error::last_error()
        );
        commit(h, &[]);
        ffi_engine_writer_close(h);
        check(&tmp);

        let dir = FsDirectory::open(tmp.path());
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        let sci = &infos.segments[0];
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let fields = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        let stores = |name: &str| {
            fields
                .fields
                .iter()
                .find(|fi| fi.name == name)
                .unwrap()
                .store_payloads
        };
        assert!(stores("body"));
        assert!(!stores("_id"));

        // Payloads that cannot fit in what is left of the blob.
        let short: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(0)
            .i32(1)
            .bytes(b"t")
            .i32(1000)
            .u8(4)
            .0;
        assert!(matches!(decode_op(&short), Err(FfiStatus::Decode)));
    }

    #[test]
    fn a_shard_indexes_updates_deletes_and_commits_with_user_data() {
        let tmp = empty_index("engine-writer-basic");
        let h = open(&tmp, 0);
        let f = setup(h);
        assert_eq!(stats(h)[STAT_UNCOMMITTED], 0);

        let add = doc(
            Blob::default().u8(OP_ADD).i32(1),
            &f,
            "a",
            &["x", "y"],
            0,
            false,
        );
        assert_eq!(apply(h, &add.0), 0);
        let add = doc(Blob::default().u8(OP_ADD).i32(1), &f, "b", &["y"], 1, false);
        assert_eq!(apply(h, &add.0), 0);
        let s = stats(h);
        assert_eq!(s[STAT_PENDING_DOCS], 2);
        assert_eq!(s[STAT_UNCOMMITTED], 1);
        assert!(s[STAT_RAM_BYTES] > 0);
        let first = commit(h, &[("local_checkpoint", "1"), ("max_seq_no", "1")]);

        // Update "a" (seq 2), then delete "b" with a tombstone (seq 3).
        let update = doc(
            Blob::default()
                .u8(OP_SOFT_UPDATE)
                .bytes(b"_id")
                .bytes(b"a")
                .bytes(b"__soft_deletes")
                .i64(1)
                .i32(1),
            &f,
            "a",
            &["z"],
            2,
            false,
        );
        assert_eq!(apply(h, &update.0), 0);
        let tombstone = doc(
            Blob::default()
                .u8(OP_SOFT_UPDATE)
                .bytes(b"_id")
                .bytes(b"b")
                .bytes(b"__soft_deletes")
                .i64(1)
                .i32(1),
            &f,
            "b",
            &[],
            3,
            true,
        );
        assert_eq!(apply(h, &tombstone.0), 0);
        let second = commit(h, &[("local_checkpoint", "3"), ("max_seq_no", "3")]);
        assert!(second > first);
        assert_eq!(stats(h)[STAT_UNCOMMITTED], 0);
        assert_eq!(stats(h)[STAT_GENERATION], second);
        check(&tmp);

        let dir = FsDirectory::open(tmp.path());
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        assert_eq!(
            infos.user_data,
            [
                ("local_checkpoint".to_string(), "3".to_string()),
                ("max_seq_no".to_string(), "3".to_string())
            ]
        );
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 3, "old a, old b and b's tombstone");

        // Every commit point survives until the caller drops it.
        let gens = generations(h);
        assert_eq!(*gens.last().unwrap(), second);
        assert!(gens.len() >= 3);
        // SAFETY: live slice.
        let rc = unsafe { ffi_engine_writer_delete_commits(h, gens.as_ptr(), gens.len()) };
        assert_eq!(rc, 0);
        assert_eq!(generations(h), [second]);
        check(&tmp);

        // Retention: nothing below seq 3 is needed; force-merging deletes
        // drops the old a and b, and keeps b's tombstone (seq 3).
        assert_eq!(ffi_engine_writer_set_retention(h, 1, 3), 0);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 1, &mut gen) },
            0
        );
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        let soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
        assert_eq!(soft, 1);
        assert_eq!(infos.generation, gen);
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        check(&tmp);

        assert_eq!(ffi_engine_writer_close(h), 0);
        assert_eq!(ffi_engine_writer_close(h), FfiStatus::InvalidHandle.code());
    }

    /// Files of this index a process-wide table lists: `/proc/self/fd`'s
    /// link targets or `/proc/self/maps`' paths under `dir`, `(deleted)`
    /// marker included. Scoped to `dir`, so tests running alongside cannot
    /// disturb the count.
    #[cfg(target_os = "linux")]
    fn open_under(dir: &std::path::Path) -> (Vec<String>, Vec<String>) {
        let root = dir.to_string_lossy().into_owned();
        let fds = std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(|e| std::fs::read_link(e.ok()?.path()).ok())
            .map(|t| t.to_string_lossy().into_owned())
            .filter(|t| t.starts_with(&root))
            .collect();
        let mut maps: Vec<String> = std::fs::read_to_string("/proc/self/maps")
            .unwrap()
            .lines()
            .filter_map(|l| l.find(&root).map(|i| l[i..].to_string()))
            .collect();
        maps.sort();
        maps.dedup();
        (fds, maps)
    }

    /// **The NRT refresh loop releases everything the previous reader held.**
    /// What OpenSearch does on every refresh, 150 times: write, commit, hold
    /// the commit, open a JVM reader over it sharing the previous reader's
    /// unchanged segments, search it, close the previous reader, release its
    /// hold, drop the older commits. Each round must leave no descriptor
    /// open under the index, no mapping of a file that is gone, no file on
    /// disk that neither the live commit nor the held one names -- and when
    /// the last reader closes, no mapping at all. The soak watched RSS, fds
    /// and disk for days to see the same thing.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_nrt_refresh_loop_releases_every_file_mapping_and_hold() {
        use crate::jvm_reader::{ffi_close_jvm_reader, ffi_jvm_reader_search, ffi_open_jvm_reader};
        let tmp = empty_index("engine-writer-refresh-loop");
        let h = open(&tmp, 0);
        let f = setup(h);
        let words: Vec<String> = (0..60).map(|w| format!("w{w}")).collect();
        let (mut reader, mut hold) = (0u64, 0u64);
        let mut seq = 0i64;
        let mut max_maps = 0;
        for round in 0..150 {
            for k in 0..40 {
                let id = format!("{}", (round * 40 + k) % 1500);
                let text: Vec<&str> = (0..50)
                    .map(|j| words[(round + k * 7 + j) % words.len()].as_str())
                    .collect();
                let op = Blob::default()
                    .u8(OP_SOFT_UPDATE)
                    .bytes(b"_id")
                    .bytes(id.as_bytes())
                    .bytes(b"__soft_deletes")
                    .i64(1)
                    .i32(1);
                assert_eq!(apply(h, &doc(op, &f, &id, &text, seq, false).0), 0);
                seq += 1;
            }
            let gen = commit_settled(h);
            let mut new_hold = 0;
            // SAFETY: live out-pointer.
            assert_eq!(
                unsafe { ffi_engine_writer_hold_commit(h, gen, &mut new_hold) },
                0
            );

            let name = lucene_store::directory::segments_file_name(gen).unwrap();
            let infos_bytes = std::fs::read(tmp.path().join(&name)).unwrap();
            let infos = lucene_index::segment_infos::parse(&infos_bytes, gen).unwrap();
            let max_docs: Vec<i32> = infos
                .segments
                .iter()
                .map(|s| {
                    let si = std::fs::read(tmp.path().join(format!("{}.si", s.segment_name)));
                    lucene_index::segment_info::parse(&si.unwrap(), &s.segment_id)
                        .unwrap()
                        .doc_count
                })
                .collect();
            let path = tmp.path().to_str().unwrap();
            let mut next = 0u64;
            // SAFETY: live slices and out-pointer; no live-docs words.
            let rc = unsafe {
                ffi_open_jvm_reader(
                    path.as_ptr().cast(),
                    path.len(),
                    infos_bytes.as_ptr(),
                    infos_bytes.len(),
                    gen,
                    reader,
                    max_docs.as_ptr(),
                    max_docs.len(),
                    std::ptr::null(),
                    std::ptr::null(),
                    &mut next,
                )
            };
            assert_eq!(rc, 0, "round {round}: {}", crate::error::last_error());
            let query = crate::jvm_reader::tests::term_blob("body", "w3");
            let (mut docs, mut scores) = ([0i32; 10], [0f32; 10]);
            let (mut n, mut total, mut lower) = (0usize, 0i64, false);
            // SAFETY: live buffers of 10 and out-pointers.
            let rc = unsafe {
                ffi_jvm_reader_search(
                    next,
                    query.as_ptr(),
                    query.len(),
                    10,
                    i64::MAX,
                    docs.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    10,
                    &mut n,
                    &mut total,
                    &mut lower,
                )
            };
            assert_eq!(rc, 0, "round {round}: {}", crate::error::last_error());
            assert!(total > 0);

            if reader != 0 {
                assert_eq!(ffi_close_jvm_reader(reader), 0);
                assert_eq!(ffi_engine_writer_release_hold(h, hold), 0);
            }
            (reader, hold) = (next, new_hold);
            let gens = generations(h);
            let old: Vec<i64> = gens.iter().copied().filter(|&g| g != gen).collect();
            // SAFETY: live slice.
            assert_eq!(
                unsafe { ffi_engine_writer_delete_commits(h, old.as_ptr(), old.len()) },
                0
            );

            let (fds, maps) = open_under(tmp.path());
            // The writer's own `write.lock` descriptor is the one that must
            // stay open: it is what holds the OS lock.
            let lock = tmp.path().join("write.lock").to_string_lossy().into_owned();
            assert!(
                fds == [lock.clone()],
                "round {round}: descriptors left open {fds:?}"
            );
            let gone: Vec<&String> = maps.iter().filter(|m| m.ends_with("(deleted)")).collect();
            assert!(
                gone.is_empty(),
                "round {round}: mappings of deleted files {gone:?}"
            );
            // Exactly the live commit's files: its `segments_N` and each
            // segment's `.si`-declared files plus the `.liv`, field-infos and
            // doc-values generations it names. A superseded generation of a
            // live segment -- what this loop writes every round -- counts.
            let mut want: std::collections::BTreeSet<String> =
                std::iter::once(name.clone()).collect();
            for sci in &infos.segments {
                let si = lucene_index::segment_info::parse(
                    &std::fs::read(tmp.path().join(format!("{}.si", sci.segment_name))).unwrap(),
                    &sci.segment_id,
                )
                .unwrap();
                want.extend(sci.files(&si.files));
            }
            let have: std::collections::BTreeSet<String> = std::fs::read_dir(tmp.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .filter(|n| lucene_index::index_file_deleter::is_index_file_name(n))
                .collect();
            let extra: Vec<&String> = have.difference(&want).collect();
            let missing: Vec<&String> = want.difference(&have).collect();
            assert!(
                extra.is_empty() && missing.is_empty(),
                "round {round}: on disk but not in the commit {extra:?}; missing {missing:?}"
            );
            max_maps = max_maps.max(maps.len());
        }
        assert!(
            max_maps > 0,
            "no file was ever mapped: the mapping checks tested nothing"
        );
        assert_eq!(ffi_close_jvm_reader(reader), 0);
        assert_eq!(ffi_engine_writer_release_hold(h, hold), 0);
        // What is still mapped is the writer's own: the postings its reader
        // pool keeps for resolving the next deletes, of live segments only.
        let (fds, maps) = open_under(tmp.path());
        let lock = tmp.path().join("write.lock").to_string_lossy().into_owned();
        assert!(
            fds == [lock] && maps.iter().all(|m| !m.ends_with("(deleted)")),
            "after the last close: {fds:?} {maps:?}"
        );
        check(&tmp);
        assert_eq!(ffi_engine_writer_close(h), 0);
        // Closing the writer releases the lock and its pool, and with them
        // the last descriptor and mapping.
        let (fds, maps) = open_under(tmp.path());
        assert!(
            fds.is_empty() && maps.is_empty(),
            "after the writer closed: {fds:?} {maps:?}"
        );
    }

    #[test]
    fn holds_pin_a_readers_files_until_released() {
        let tmp = empty_index("engine-writer-holds");
        let h = open(&tmp, 0);
        let f = setup(h);
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            let add = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                id,
                &["w"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &add.0), 0);
            commit(h, &[]);
        }
        let held_gen = *generations(h).last().unwrap();
        let mut hold = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_hold_commit(h, held_gen, &mut hold) },
            0
        );
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let gens = generations(h);
        // SAFETY: live slice.
        assert_eq!(
            unsafe { ffi_engine_writer_delete_commits(h, gens.as_ptr(), gens.len()) },
            0
        );
        let dir = FsDirectory::open(tmp.path());
        let held_infos = lucene_index::segment_infos::parse(
            &std::fs::read(tmp.path().join(format!("segments_{held_gen}"))).unwrap_or_default(),
            held_gen,
        );
        assert!(held_infos.is_err(), "the held commit's segments_N is gone");
        let before = dir.list_all().unwrap();
        assert!(
            before.iter().any(|n| n.starts_with("_0.")),
            "held segment files stay"
        );
        assert_eq!(ffi_engine_writer_release_hold(h, hold), 0);
        let after = dir.list_all().unwrap();
        assert!(
            !after.iter().any(|n| n.starts_with("_0.")),
            "released files go"
        );
        assert_eq!(
            ffi_engine_writer_release_hold(h, hold),
            FfiStatus::InvalidArgument.code()
        );
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_hold_commit(h, 999, &mut hold) },
            FfiStatus::InvalidArgument.code()
        );
        check(&tmp);
        assert_eq!(ffi_engine_writer_close(h), 0);
    }

    #[test]
    fn an_nrt_snapshot_shows_buffered_documents_without_a_commit() {
        let tmp = empty_index("engine-writer-nrt");
        let h = open(&tmp, 0);
        let f = setup(h);
        let add = |i: i64, id: &str| {
            let op = doc(Blob::default().u8(OP_ADD).i32(1), &f, id, &["w"], i, false);
            assert_eq!(apply(h, &op.0), 0);
        };
        add(0, "a");
        commit(h, &[]);
        let committed = generations(h);
        add(1, "b");
        add(2, "c");

        let (mut ptr, mut len, mut hold, mut gen) = (std::ptr::null_mut(), 0usize, 0u64, 0i64);
        // SAFETY: live out-pointers.
        let rc =
            unsafe { ffi_engine_writer_nrt_snapshot(h, &mut ptr, &mut len, &mut hold, &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call just gave.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };

        assert_eq!(generations(h), committed, "a snapshot writes no commit");
        assert_eq!(Some(&gen), committed.last());
        let infos = lucene_index::segment_infos::parse(&bytes, gen).unwrap();
        let docs: i32 = infos
            .segments
            .iter()
            .map(|s| {
                let si = std::fs::read(tmp.path().join(format!("{}.si", s.segment_name)));
                lucene_index::segment_info::parse(&si.unwrap(), &s.segment_id)
                    .unwrap()
                    .doc_count
            })
            .sum();
        assert_eq!(docs, 3, "the snapshot holds the buffered documents");

        assert_eq!(ffi_engine_writer_release_hold(h, hold), 0);
        assert_eq!(
            ffi_engine_writer_release_hold(h, hold),
            FfiStatus::InvalidArgument.code()
        );
        commit(h, &[]);
        check(&tmp);
        assert_eq!(ffi_engine_writer_close(h), 0);
    }

    /// Closing the last reader over merged-away segments deletes their files
    /// at once, with no write, flush or refresh after it: an index that goes
    /// idle after a force merge must not keep both copies on disk.
    #[test]
    fn releasing_an_nrt_hold_deletes_merged_away_files_at_once() {
        let tmp = empty_index("engine-writer-nrt-release");
        let h = open(&tmp, 0);
        let f = setup(h);
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            let add = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                id,
                &["w"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &add.0), 0);
            commit(h, &[]);
        }
        let snapshot = |h: u64| {
            let (mut ptr, mut len, mut hold, mut gen) = (std::ptr::null_mut(), 0usize, 0u64, 0i64);
            // SAFETY: live out-pointers.
            let rc = unsafe {
                ffi_engine_writer_nrt_snapshot(h, &mut ptr, &mut len, &mut hold, &mut gen)
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            // SAFETY: the buffer the call gave, freed once.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            hold
        };
        let old = snapshot(h);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) },
            0
        );
        let gens = generations(h);
        let older: Vec<i64> = gens.iter().copied().filter(|&g| g != gen).collect();
        // SAFETY: live slice.
        assert_eq!(
            unsafe { ffi_engine_writer_delete_commits(h, older.as_ptr(), older.len()) },
            0
        );
        let new = snapshot(h);
        let dir = FsDirectory::open(tmp.path());
        let merged_away = |n: &String| ["_0.", "_1.", "_2."].iter().any(|p| n.starts_with(p));
        let before = dir.list_all().unwrap();
        for p in ["_0.", "_1.", "_2."] {
            assert!(
                before.iter().any(|n| n.starts_with(p)),
                "the open reader keeps {p}*: {before:?}"
            );
        }
        assert_eq!(ffi_engine_writer_release_hold(h, old), 0);
        let after = dir.list_all().unwrap();
        assert!(
            !after.iter().any(merged_away),
            "the released segments' files are gone: {after:?}"
        );
        assert_eq!(ffi_engine_writer_release_hold(h, new), 0);
        check(&tmp);
        assert_eq!(ffi_engine_writer_close(h), 0);
    }

    #[test]
    fn a_refused_document_is_not_tragic_but_a_panic_is() {
        let tmp = empty_index("engine-writer-failures");
        let h = open(&tmp, 1);
        let f = setup(h);
        // A schema Lucene refuses -- term vectors on a field that is not
        // indexed (`FieldInfo.checkConsistency`) -- is refused at
        // registration, as the mapping's problem.
        let mut tv = field("tv", 0, 0, 0, 0);
        tv[2 + 4 + 2] = 1;
        assert_eq!(register(h, &tv), Err(FfiStatus::InvalidArgument.code()));
        // A document naming an unregistered field is refused; the writer
        // carries on.
        let bad = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .i32(1)
            .i32(99)
            .u8(1)
            .bytes(b"x")
            .i32(0)
            .i32(0)
            .i32(0);
        assert_eq!(apply(h, &bad.0), FfiStatus::InvalidArgument.code());
        let ok = doc(Blob::default().u8(OP_ADD).i32(1), &f, "a", &["x"], 0, false);
        assert_eq!(apply(h, &ok.0), 0);
        // Truncated and trailing bytes are decode errors, also not tragic.
        assert_eq!(apply(h, &ok.0[..ok.0.len() - 1]), FfiStatus::Decode.code());
        let mut long = ok.0.clone();
        long.push(0);
        assert_eq!(apply(h, &long), FfiStatus::Decode.code());
        assert_eq!(apply(h, &[OP_ADD, 0, 0, 0, 0]), FfiStatus::Decode.code());
        assert_eq!(apply(h, &[7]), FfiStatus::Decode.code());
        assert_eq!(apply(h, &ok.0), 0);

        // An injected panic is caught, and poisons only this handle.
        assert_eq!(apply(h, &[OP_PANIC]), FfiStatus::Panic.code());
        assert_eq!(apply(h, &ok.0), FfiStatus::Io.code());
        assert!(crate::error::last_error().contains("tragic"));
        let other = empty_index("engine-writer-failures-other");
        let h2 = open(&other, 0);
        let f2 = setup(h2);
        let ok2 = doc(
            Blob::default().u8(OP_ADD).i32(1),
            &f2,
            "a",
            &["x"],
            0,
            false,
        );
        assert_eq!(apply(h2, &ok2.0), 0);
        assert_eq!(apply(h2, &[OP_PANIC]), FfiStatus::InvalidArgument.code());
        commit(h2, &[]);
        assert_eq!(ffi_engine_writer_close(h2), 0);
        assert_eq!(
            ffi_engine_writer_close(h),
            0,
            "a tragic writer still closes"
        );
    }

    #[test]
    fn bad_arguments_are_status_codes() {
        let tmp = empty_index("engine-writer-args");
        let h = open(&tmp, 0);
        let mut gen = 0;
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, 0, 0, &mut gen) },
            FfiStatus::InvalidArgument.code()
        );
        // SAFETY: live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_force_merge(h, -1, 0, &mut gen) },
            FfiStatus::InvalidArgument.code()
        );
        // SAFETY: null out-pointers are refused before any write.
        unsafe {
            assert_eq!(
                ffi_engine_writer_force_merge(h, 1, 0, std::ptr::null_mut()),
                FfiStatus::NullPointer.code()
            );
            assert_eq!(
                ffi_engine_writer_commit(h, 0, std::ptr::null_mut()),
                FfiStatus::NullPointer.code()
            );
            assert_eq!(
                ffi_engine_writer_hold_commit(h, 1, std::ptr::null_mut()),
                FfiStatus::NullPointer.code()
            );
            assert_eq!(
                ffi_engine_writer_register_field(h, [0u8; 4].as_ptr(), 4, std::ptr::null_mut()),
                FfiStatus::NullPointer.code()
            );
            assert_eq!(
                ffi_engine_writer_stats(h, std::ptr::null_mut(), 1),
                FfiStatus::NullPointer.code()
            );
            assert_eq!(
                ffi_engine_writer_delete_commits(h, std::ptr::null(), 1),
                FfiStatus::NullPointer.code()
            );
            let mut n = 0;
            assert_eq!(
                ffi_engine_writer_commit_generations(h, std::ptr::null_mut(), 0, &mut n),
                FfiStatus::BufferTooSmall.code()
            );
            assert_eq!(n, 1);
            assert_eq!(
                ffi_engine_writer_open(
                    "x".as_ptr() as *const c_char,
                    1,
                    16.0,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    i32::MAX,
                    std::ptr::null_mut()
                ),
                FfiStatus::NullPointer.code()
            );
        }
        // A malformed field spec and user data.
        assert_eq!(register(h, &[1, 0, 0, 0]), Err(FfiStatus::Decode.code()));
        let mut spec = field("f", 9, 0, 0, 0);
        assert_eq!(register(h, &spec), Err(FfiStatus::Decode.code()));
        spec = field("f", 1, 2, 0, 0);
        assert_eq!(register(h, &spec), Err(FfiStatus::Decode.code()));
        spec = field("f", 1, 0, 9, 0);
        assert_eq!(register(h, &spec), Err(FfiStatus::Decode.code()));
        commit(h, &[]);
        COMMIT_DATA.with(|d| *d.borrow_mut() = Blob::default().i32(1).bytes(b"k").0);
        let mut g = 0;
        // SAFETY: a live out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_commit(h, 0, &mut g) },
            FfiStatus::Decode.code()
        );
        // A bad RAM buffer is refused at open.
        let path = tmp.path().to_str().unwrap();
        let mut h2 = 0;
        // SAFETY: live string and out-pointer.
        let rc = unsafe {
            ffi_engine_writer_open(
                path.as_ptr() as *const c_char,
                path.len(),
                -2.0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                i32::MAX,
                &mut h2,
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(ffi_engine_writer_close(h), 0);
        assert_eq!(
            ffi_engine_writer_release_hold(h, 1),
            FfiStatus::InvalidHandle.code()
        );
    }

    #[test]
    fn every_value_kind_decodes() {
        let d = Blob::default()
            .i32(6)
            .i32(0)
            .u8(0)
            .bytes(b"s")
            .i32(0)
            .u8(1)
            .bytes(&[1, 2])
            .i32(0)
            .u8(2)
            .bytes(&7i32.to_le_bytes())
            .i32(0)
            .u8(3)
            .bytes(&7i64.to_le_bytes())
            .i32(0)
            .u8(4)
            .bytes(&1.5f32.to_le_bytes())
            .i32(0)
            .u8(5)
            .bytes(&1.5f64.to_le_bytes())
            .i32(0)
            .i32(2)
            .i32(1)
            .u8(0)
            .i64(7)
            .i32(1)
            .u8(1)
            .bytes(&[9])
            .i32(1)
            .i32(2)
            .bytes(&[0, 1, 2, 3]);
        let op = decode_op(
            &Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .0
                .into_iter()
                .chain(d.0)
                .collect::<Vec<_>>(),
        );
        let Ok(Op::Add { docs, .. }) = op else {
            panic!("decoded");
        };
        let doc = &docs[0];
        assert_eq!(doc.stored.len(), 6);
        assert!(matches!(doc.stored[5].value, FieldValue::Double(v) if v == 1.5));
        assert!(matches!(doc.fields.doc_values[1].value, FieldValue::Binary(ref b) if b == &[9]));
        assert_eq!(doc.fields.points.len(), 1);
        for bad in [
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(9)
                .bytes(b"x")
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(2)
                .bytes(b"x")
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(1)
                .i32(0)
                .u8(0)
                .bytes(&[0xff])
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default()
                .i32(0)
                .i32(1)
                .i32(0)
                .u8(3)
                .i32(0)
                .i32(0)
                .i32(0),
            Blob::default().i32(0).i32(0).i32(1).i32(0).u8(9).i32(0),
            Blob::default().i32(-1),
        ] {
            let blob: Vec<u8> = Blob::default()
                .u8(OP_ADD)
                .i32(1)
                .0
                .into_iter()
                .chain(bad.0)
                .collect();
            assert!(matches!(decode_op(&blob), Err(FfiStatus::Decode)));
        }
        // A freq whose positions or offsets cannot fit.
        let short = Blob::default()
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(0)
            .i32(1)
            .bytes(b"t")
            .i32(1000)
            .u8(1);
        let blob: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(short.0)
            .collect();
        assert!(matches!(decode_op(&blob), Err(FfiStatus::Decode)));
        let short = Blob::default()
            .i32(0)
            .i32(1)
            .i32(0)
            .u8(0)
            .i32(1)
            .bytes(b"t")
            .i32(1000)
            .u8(2);
        let blob: Vec<u8> = Blob::default()
            .u8(OP_ADD)
            .i32(1)
            .0
            .into_iter()
            .chain(short.0)
            .collect();
        assert!(matches!(decode_op(&blob), Err(FfiStatus::Decode)));
        let vector = Blob::default()
            .bytes(b"v")
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(4)
            // float32, cosine, Lucene104HnswScalarQuantizedVectorsFormat(7 bits, 16, 100)
            .u8(0)
            .u8(2)
            .u8(2)
            .i32(16)
            .i32(100)
            .u8(2)
            .u8(0);
        let attributed = vector.clone().i32(1).bytes(b"k").bytes(b"v");
        assert_eq!(
            decode_field(&attributed.0).unwrap().0.attributes,
            vec![("k".to_string(), "v".to_string())]
        );
        assert!(decode_field(&vector.clone().i32(1).bytes(b"k").0).is_err());
        let vector = vector.i32(0);
        let (info, format) = decode_field(&vector.0).unwrap();
        assert_eq!(info.vector_dimension, 4);
        assert_eq!(info.vector_encoding, VectorEncoding::Float32);
        assert_eq!(
            info.vector_similarity_function,
            VectorSimilarityFunction::Cosine
        );
        assert_eq!(
            format,
            Some(
                KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SevenBit, 16, 100).unwrap()
            )
        );
        // A byte field is never quantized; a graph's bounds are Lucene's.
        let schema = |encoding: u8, kind: u8, m: i32| {
            let b = Blob::default()
                .bytes(b"v")
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .u8(0)
                .i32(0)
                .i32(0)
                .i32(0)
                .i32(4)
                .u8(encoding)
                .u8(0)
                .u8(kind);
            let b = match kind {
                1 | 4 | 5 => b.i32(m).i32(100),
                2 => b.i32(m).i32(100).u8(2),
                3 => b.u8(2),
                // `m` is the cluster size, the rerank encoding f16.
                6 => b.i32(m).u8(1),
                _ => b,
            };
            b.u8(0).i32(0).0
        };
        // IVF: any field (a byte one keeps its raw vectors alone), a checked
        // cluster size.
        for encoding in [0, 1] {
            assert_eq!(
                decode_field(&schema(encoding, 6, 384)).unwrap().1,
                Some(KnnVectorsFormat::ivf(384, 1).unwrap())
            );
        }
        assert!(decode_field(&schema(0, 6, 8)).is_err());
        assert!(decode_field(&schema(0, 6, -1)).is_err());
        assert!(decode_field(&schema(1, 2, 16)).is_err());
        assert!(decode_field(&schema(1, 3, 16)).is_err());
        assert!(decode_field(&schema(1, 1, 16)).is_ok());
        // Hamming scoring is for byte (packed bit) fields only.
        assert_eq!(
            decode_field(&schema(1, 4, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_hamming(16, 100).unwrap())
        );
        assert!(decode_field(&schema(0, 4, 16)).is_err());
        // Half floats are for float fields only.
        assert_eq!(
            decode_field(&schema(0, 5, 16)).unwrap().1,
            Some(KnnVectorsFormat::hnsw_half_float(16, 100).unwrap())
        );
        assert!(decode_field(&schema(1, 5, 16)).is_err());
        assert!(decode_field(&schema(0, 1, 0)).is_err());
        assert!(decode_field(&schema(0, 1, 513)).is_err());
        assert_eq!(decode_field(&schema(0, 0, 0)).unwrap().1, None);
        assert!(decode_field(&schema(0, 9, 0)).is_err());
        assert_eq!(
            parse_vectors_format("hnsw:16:100"),
            Some(KnnVectorsFormat::hnsw(16, 100).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_sq:3:32:200"),
            Some(
                KnnVectorsFormat::hnsw_scalar_quantized(
                    ScalarEncoding::SingleBitQueryNibble,
                    32,
                    200
                )
                .unwrap()
            )
        );
        assert_eq!(
            parse_vectors_format("hnsw_hamming:8:64"),
            Some(KnnVectorsFormat::hnsw_hamming(8, 64).unwrap())
        );
        assert_eq!(
            parse_vectors_format("hnsw_half:8:64"),
            Some(KnnVectorsFormat::hnsw_half_float(8, 64).unwrap())
        );
        assert_eq!(
            parse_vectors_format("ivf:384:1"),
            Some(KnnVectorsFormat::ivf(384, 1).unwrap())
        );
        assert_eq!(
            parse_vectors_format("sq:1"),
            Some(KnnVectorsFormat::scalar_quantized(
                ScalarEncoding::PackedNibble
            ))
        );
        for bad in [
            "",
            "hnsw",
            "hnsw:0:100",
            "hnsw_sq:9:16:100",
            "flat:1",
            "hnsw:a:b",
            "ivf:8:0",
            "ivf:384:4",
            "ivf:384",
        ] {
            assert_eq!(parse_vectors_format(bad), None, "{bad}");
        }
        assert!(decode_user_data(&Blob::default().i32(0).u8(0).0).is_err());
    }

    /// Opens a JVM reader of commit `gen`, every segment fully live.
    fn open_reader(tmp: &TempDir, gen: i64) -> u64 {
        let name = lucene_store::directory::segments_file_name(gen).unwrap();
        let infos_bytes = std::fs::read(tmp.path().join(&name)).unwrap();
        let infos = lucene_index::segment_infos::parse(&infos_bytes, gen).unwrap();
        let max_docs: Vec<i32> = infos
            .segments
            .iter()
            .map(|s| {
                let si = std::fs::read(tmp.path().join(format!("{}.si", s.segment_name)));
                lucene_index::segment_info::parse(&si.unwrap(), &s.segment_id)
                    .unwrap()
                    .doc_count
            })
            .collect();
        let path = tmp.path().to_str().unwrap();
        let mut reader = 0u64;
        // SAFETY: live slices and out-pointer; no live-docs words.
        let rc = unsafe {
            crate::jvm_reader::ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                infos_bytes.as_ptr(),
                infos_bytes.len(),
                gen,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                &mut reader,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        reader
    }

    /// `(doc, score)` hits and the total of `query` over `reader` and
    /// `view`'s buffers; `None` when the buffers cannot answer it.
    fn search_live(reader: u64, view: u64, query: &[u8]) -> Option<(Vec<(i32, f32)>, i64)> {
        let (mut docs, mut scores) = ([0i32; 64], [0f32; 64]);
        let (mut n, mut total, mut lower, mut supported) = (0usize, 0i64, false, false);
        // SAFETY: live buffers of 64 and out-pointers.
        let rc = unsafe {
            crate::live::ffi_jvm_reader_search_live(
                reader,
                view,
                query.as_ptr(),
                query.len(),
                64,
                i64::MAX,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                64,
                &mut n,
                &mut total,
                &mut lower,
                &mut supported,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        supported.then(|| {
            (
                docs[..n]
                    .iter()
                    .copied()
                    .zip(scores[..n].iter().copied())
                    .collect(),
                total,
            )
        })
    }

    /// [`OPEN_LIVE_SEARCH`]: a reader of the last commit and a view of the
    /// buffers together find and score what a reader of a commit of
    /// everything does, and a buffered hit's stored fields read back.
    #[test]
    fn a_view_searches_the_buffers_beside_a_reader() {
        use crate::live::{
            ffi_engine_writer_live_view_open, ffi_engine_writer_release_live, ffi_live_view_close,
            ffi_live_view_document_alloc,
        };
        let tmp = empty_index("engine-writer-live-view");
        let h = open_with(&tmp, 64.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let words = ["alpha", "beta", "gamma", "delta"];
        let add = |from: usize, to: usize| {
            for i in from..to {
                let text: Vec<&str> = (0..1 + i % 4).map(|j| words[(i + j) % 4]).collect();
                let id = format!("d{i}");
                let b = doc(
                    Blob::default().u8(OP_ADD).i32(1),
                    &f,
                    &id,
                    &text,
                    i as i64,
                    false,
                );
                assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            }
        };
        add(0, 40);
        let gen = commit_settled(h);
        let reader = open_reader(&tmp, gen);
        add(40, 70);

        let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
        // SAFETY: live out-pointers.
        let rc = unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call just gave.
        let blob = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let count = i32::from_le_bytes(blob[8..12].try_into().unwrap());
        let mut at = 12;
        let mut buffered = 0;
        for _ in 0..count {
            buffered += i32::from_le_bytes(blob[at..at + 4].try_into().unwrap());
            assert_eq!(
                i32::from_le_bytes(blob[at + 4..at + 8].try_into().unwrap()),
                0
            );
            let chunks = i32::from_le_bytes(blob[at + 8..at + 12].try_into().unwrap());
            at += 12 + 4 * chunks as usize;
        }
        assert_eq!((buffered, at), (30, blob.len()));

        let query = crate::jvm_reader::tests::term_blob("body", "gamma");
        let (live, live_total) = search_live(reader, view, &query).unwrap();
        let all_gen = commit_settled(h);
        let all = open_reader(&tmp, all_gen);
        let (mut docs, mut scores) = ([0i32; 64], [0f32; 64]);
        let (mut n, mut want_total, mut lower) = (0usize, 0i64, false);
        // SAFETY: live buffers of 64 and out-pointers.
        let rc = unsafe {
            crate::jvm_reader::ffi_jvm_reader_search(
                all,
                query.as_ptr(),
                query.len(),
                64,
                i64::MAX,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                64,
                &mut n,
                &mut want_total,
                &mut lower,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let want: Vec<(i32, f32)> = docs[..n]
            .iter()
            .copied()
            .zip(scores[..n].iter().copied())
            .collect();
        assert_eq!(live_total, want_total);
        let mut got: Vec<f32> = live.iter().map(|h| h.1).collect();
        let mut exp: Vec<f32> = want.iter().map(|h| h.1).collect();
        got.sort_by(f32::total_cmp);
        exp.sort_by(f32::total_cmp);
        assert_eq!(got, exp);

        let (doc, _) = *live.iter().find(|h| h.0 >= 40).unwrap();
        let (mut ptr, mut len) = (std::ptr::null_mut(), 0usize);
        // SAFETY: live out-pointers.
        let rc = unsafe { ffi_live_view_document_alloc(view, 0, doc - 40, &mut ptr, &mut len) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call just gave.
        let stored = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let id = format!("d{doc}");
        assert!(stored.windows(id.len()).any(|w| w == id.as_bytes()));

        assert_eq!(ffi_live_view_close(view), 0);
        assert_ne!(ffi_live_view_close(view), 0);
        assert_eq!(ffi_engine_writer_release_live(h, gen), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(all), 0);
    }

    /// A reader's documents updated or deleted since its commit are hidden
    /// from a search with a view, and the new versions found in the buffers.
    #[test]
    fn a_view_hides_the_readers_updated_documents() {
        use crate::live::{ffi_engine_writer_live_view_open, ffi_live_view_close};
        let tmp = empty_index("engine-writer-live-updates");
        let h = open_with(&tmp, 64.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let words = ["alpha", "beta", "gamma", "delta"];
        for i in 0..20usize {
            let text: Vec<&str> = (0..1 + i % 4).map(|j| words[(i + j) % 4]).collect();
            let b = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                &format!("d{i}"),
                &text,
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        }
        let gen = commit_settled(h);
        let reader = open_reader(&tmp, gen);
        let soft_update = |id: &str, text: &[&str], seq: i64, tombstone: bool| {
            let b = doc(
                Blob::default()
                    .u8(OP_SOFT_UPDATE)
                    .bytes(b"_id")
                    .bytes(id.as_bytes())
                    .bytes(b"__soft_deletes")
                    .i64(1)
                    .i32(1),
                &f,
                id,
                text,
                seq,
                tombstone,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        // d2 holds gamma; its new version does not. d3 goes.
        soft_update("d2", &["zeta"], 20, false);
        soft_update("d3", &[], 21, true);

        let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
        // SAFETY: live out-pointers.
        let rc = unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let gamma = crate::jvm_reader::tests::term_blob("body", "gamma");
        let zeta = crate::jvm_reader::tests::term_blob("body", "zeta");
        let (gamma_hits, gamma_total) = search_live(reader, view, &gamma).unwrap();
        // Twice: the second search reads the masks the first one built.
        assert_eq!(search_live(reader, view, &gamma).unwrap().1, gamma_total);
        let (zeta_hits, zeta_total) = search_live(reader, view, &zeta).unwrap();

        let total = |r: u64, q: &[u8]| {
            let (mut docs, mut scores) = ([0i32; 64], [0f32; 64]);
            let (mut n, mut total, mut lower) = (0usize, 0i64, false);
            // SAFETY: live buffers of 64 and out-pointers.
            let rc = unsafe {
                crate::jvm_reader::ffi_jvm_reader_search(
                    r,
                    q.as_ptr(),
                    q.len(),
                    64,
                    i64::MAX,
                    docs.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    64,
                    &mut n,
                    &mut total,
                    &mut lower,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            total
        };
        // The reader alone still has d2 and d3.
        assert_eq!(gamma_total, total(reader, &gamma) - 2);
        assert_eq!(zeta_total, 1);
        assert!(gamma_hits.iter().all(|&(d, _)| d != 2 && d != 3));
        assert_eq!(zeta_hits.len(), 1);
        assert!(zeta_hits[0].0 >= 20);

        assert_eq!(ffi_live_view_close(view), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
    }

    /// Under a staleness bound a view opened within it is shared, without
    /// the writes since; without one each view holds every write before it.
    #[test]
    fn under_a_staleness_bound_searches_share_a_view() {
        use crate::live::{
            ffi_engine_writer_live_view_open, ffi_engine_writer_set_live_staleness,
            ffi_live_view_close,
        };
        let tmp = empty_index("engine-writer-live-stale");
        let h = open_with(&tmp, 64.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let add = |i: usize| {
            let b = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                &format!("d{i}"),
                &["alpha"],
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        let open = || {
            let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
            // SAFETY: live out-pointers.
            let rc =
                unsafe { ffi_engine_writer_live_view_open(h, -1, &mut view, &mut ptr, &mut len) };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            // SAFETY: the buffer the call gave, read then freed once.
            let blob = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            (view, blob)
        };
        (0..5).for_each(add);
        assert_eq!(ffi_engine_writer_set_live_staleness(h, 60_000), 0);
        let (a, first) = open();
        add(5);
        // Within the bound: the first view, without the later document.
        let (b, second) = open();
        assert_ne!(a, b);
        assert_eq!(first, second);
        // After a wait_for release: a view holding the write.
        assert_eq!(crate::live::ffi_engine_writer_live_refresh_shared(h), 0);
        let (r, refreshed) = open();
        assert_ne!(first, refreshed);
        assert_eq!(ffi_live_view_close(r), 0);
        assert_eq!(ffi_live_view_close(a), 0);
        assert_eq!(ffi_live_view_close(b), 0);
        // Without one: every write before the view.
        assert_eq!(ffi_engine_writer_set_live_staleness(h, 0), 0);
        let (c, third) = open();
        assert_ne!(first, third);
        assert_eq!(ffi_live_view_close(c), 0);
        assert_ne!(ffi_engine_writer_set_live_staleness(h, -1), 0);
        assert_eq!(ffi_engine_writer_close(h), 0);
    }

    /// [`crate::live::ffi_live_view_reader`]: a reader handle over the
    /// reader and the view's buffers as segments finds what the live search
    /// finds, numbered alike, reads the buffered documents' doc values, and
    /// leaves out what was added or deleted after the view opened.
    #[test]
    fn a_view_reader_holds_the_buffers_as_segments() {
        use crate::live::{
            ffi_engine_writer_live_view_open, ffi_live_view_close, ffi_live_view_reader,
        };
        use lucene_search::reader::LeafReader;
        let tmp = empty_index("engine-writer-live-reader");
        let h = open_with(&tmp, 64.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let words = ["alpha", "beta", "gamma", "delta"];
        let add = |i: usize| {
            let text: Vec<&str> = (0..1 + i % 4).map(|j| words[(i + j) % 4]).collect();
            let b = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                &format!("d{i}"),
                &text,
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        let update = |id: &str, text: &[&str], seq: i64| {
            let b = doc(
                Blob::default()
                    .u8(OP_SOFT_UPDATE)
                    .bytes(b"_id")
                    .bytes(id.as_bytes())
                    .bytes(b"__soft_deletes")
                    .i64(1)
                    .i32(1),
                &f,
                id,
                text,
                seq,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        (0..20).for_each(add);
        let gen = commit_settled(h);
        let reader = open_reader(&tmp, gen);
        (20..30).for_each(add);
        update("d2", &["zeta"], 30);
        update("d21", &["zeta"], 31);

        let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
        // SAFETY: live out-pointers.
        let rc = unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        // After the view: in no search through it.
        update("d22", &["zeta"], 32);
        add(40);

        let mut whole = 0u64;
        // SAFETY: a live out-pointer.
        let rc = unsafe {
            ffi_live_view_reader(
                reader,
                view,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut whole,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let mut again = 0u64;
        // SAFETY: a live out-pointer.
        assert_eq!(
            unsafe {
                ffi_live_view_reader(
                    reader,
                    view,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut again,
                )
            },
            0
        );
        assert_eq!(again, whole, "one reader per view");

        let total = |r: u64, q: &[u8]| {
            let (mut docs, mut scores) = ([0i32; 64], [0f32; 64]);
            let (mut n, mut total, mut lower) = (0usize, 0i64, false);
            // SAFETY: live buffers of 64 and out-pointers.
            let rc = unsafe {
                crate::jvm_reader::ffi_jvm_reader_search(
                    r,
                    q.as_ptr(),
                    q.len(),
                    64,
                    i64::MAX,
                    docs.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    64,
                    &mut n,
                    &mut total,
                    &mut lower,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let mut hits: Vec<(i32, u32)> = docs[..n]
                .iter()
                .copied()
                .zip(scores[..n].iter().map(|s| s.to_bits()))
                .collect();
            hits.sort_unstable();
            (hits, total)
        };
        for term in ["gamma", "zeta", "alpha"] {
            let q = crate::jvm_reader::tests::term_blob("body", term);
            let (mut live, live_total) = search_live(reader, view, &q).unwrap();
            live.sort_by_key(|&(d, _)| d);
            let (hits, got) = total(whole, &q);
            assert_eq!(got, live_total, "{term}");
            let live: Vec<(i32, u32)> = live.iter().map(|&(d, s)| (d, s.to_bits())).collect();
            assert_eq!(hits, live, "{term}: the same hits, scored alike");
            // A handle for this request alone: its buffers written without
            // postings, the query's terms read from them.
            let mut lean = 0u64;
            // SAFETY: live slices and out-pointer.
            let rc = unsafe {
                ffi_live_view_reader(
                    reader,
                    view,
                    q.as_ptr(),
                    q.len(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut lean,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            assert_ne!(lean, whole, "{term}: a handle of its own");
            assert_eq!(total(lean, &q), (hits, got), "{term}: as with every term");
        }
        // zeta: d2' and d21' only; d22' came after the view.
        assert_eq!(
            total(whole, &crate::jvm_reader::tests::term_blob("body", "zeta")).1,
            2
        );

        {
            let w = crate::jvm_reader::lookup(whole, "test").unwrap();
            let readers = w.reader.segment_readers();
            let base = crate::jvm_reader::lookup(reader, "test")
                .unwrap()
                .live_docs
                .len();
            // One segment per buffer, holding the documents of the view.
            assert_eq!(readers.len(), base + 1);
            assert_eq!(readers[base].max_doc, 12);
            let mut found = Vec::new();
            for (r, live) in readers.iter().zip(&w.live_docs).skip(base) {
                let mut dv = r.numeric_doc_values("_seq_no").unwrap().unwrap();
                for d in 0..r.max_doc {
                    if live.as_ref().is_none_or(|l| l.get(d as usize))
                        && dv.advance_exact(d).unwrap()
                    {
                        found.push(dv.long_value());
                    }
                }
            }
            found.sort_unstable();
            let mut want: Vec<i64> = (20..30).filter(|&i| i != 21).collect();
            want.extend([30, 31]);
            assert_eq!(found, want);
        }

        assert_eq!(ffi_live_view_close(view), 0);
        assert_ne!(
            crate::jvm_reader::ffi_close_jvm_reader(whole),
            0,
            "closed with the view"
        );
        // SAFETY: a live out-pointer.
        let rc = unsafe {
            ffi_live_view_reader(
                reader,
                view,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut again,
            )
        };
        assert_ne!(rc, 0, "a closed view makes no reader");

        // A view whose buffers were published since reads their segments,
        // and finds what it found before they were. (Enough documents that
        // the buffer is one chunk again: see `a_growing_buffer_is_read_as_chunks`.)
        (41..70).for_each(add);
        // SAFETY: live out-pointers.
        let rc = unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        // SAFETY: the buffer the call gave, freed once.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        let terms = ["gamma", "zeta", "alpha"];
        let before: Vec<i64> = terms
            .iter()
            .map(|t| {
                search_live(
                    reader,
                    view,
                    &crate::jvm_reader::tests::term_blob("body", t),
                )
                .unwrap()
                .1
            })
            .collect();
        commit_settled(h);
        // SAFETY: a live out-pointer.
        let rc = unsafe {
            ffi_live_view_reader(
                reader,
                view,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut whole,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        for (t, want) in terms.iter().zip(before) {
            let q = crate::jvm_reader::tests::term_blob("body", t);
            assert_eq!(total(whole, &q).1, want, "{t}");
        }
        // zeta: d2', d21' and d22', each once.
        assert_eq!(
            total(whole, &crate::jvm_reader::tests::term_blob("body", "zeta")).1,
            3
        );
        assert_eq!(ffi_live_view_close(view), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
    }

    /// Views of a growing buffer: each view's reader holds the buffer as
    /// chunks, written once and read again by later views, which together
    /// hold the documents, positions, offsets and statistics of the view,
    /// less its deletes, as a search of the buffers finds them.
    /// Every hit of the query in `q` over the reader at `r`, with its
    /// score's bits, by document, and the total.
    fn scored_hits(r: u64, q: &[u8]) -> (Vec<(i32, u32)>, i64) {
        let (mut docs, mut scores) = ([0i32; 256], [0f32; 256]);
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        // SAFETY: live buffers of 256 and out-pointers.
        let rc = unsafe {
            crate::jvm_reader::ffi_jvm_reader_search(
                r,
                q.as_ptr(),
                q.len(),
                256,
                i64::MAX,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                256,
                &mut n,
                &mut total,
                &mut lower,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let mut hits: Vec<(i32, u32)> = docs[..n]
            .iter()
            .copied()
            .zip(scores[..n].iter().map(|s| s.to_bits()))
            .collect();
        hits.sort_unstable();
        (hits, total)
    }

    #[test]
    fn a_growing_buffer_is_read_as_chunks() {
        use crate::live::{
            ffi_engine_writer_live_view_open, ffi_live_view_close, ffi_live_view_reader,
        };
        use lucene_search::reader::{LeafReader, PostingsFlags};
        let tmp = empty_index("engine-writer-live-chunks");
        let h = open_with(&tmp, 64.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let words = ["alpha", "beta", "gamma", "delta", "eps"];
        let text = |i: usize| -> Vec<&str> { (0..1 + i % 4).map(|j| words[(i + j) % 5]).collect() };
        let add = |i: usize, id: &str| {
            let b = doc(
                Blob::default()
                    .u8(OP_SOFT_UPDATE)
                    .bytes(b"_id")
                    .bytes(id.as_bytes())
                    .bytes(b"__soft_deletes")
                    .i64(1)
                    .i32(1),
                &f,
                id,
                &text(i),
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        let gen = commit_settled(h);
        let reader = open_reader(&tmp, gen);
        let mut added = 0usize;
        // Each document's id; an update gives an id a new document.
        let mut ids: Vec<String> = Vec::new();
        let mut leaves_seen = Vec::new();
        for round in 0..24usize {
            // Every third view opens while a chunk of the documents before
            // its last one is still being written: it reads that chunk as it
            // is and writes only the documents after it.
            let ew = lookup(h).unwrap();
            let hold = round % 3 == 2;
            let mut held = None;
            let n = 1 + round % 5 + usize::from(hold);
            if hold {
                // A view waits for a chunk being written only while the
                // warmer is idle: else it writes those documents itself.
                crate::live::idle_warmer(&ew);
            }
            for k in 0..n {
                if hold && k + 1 == n {
                    held = crate::live::hold_tail(&ew);
                }
                let id = if added % 7 == 6 {
                    ids[added / 2].clone()
                } else {
                    format!("d{added}")
                };
                add(added, &id);
                ids.push(id);
                added += 1;
            }
            let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
            // SAFETY: live out-pointers.
            let rc =
                unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            // SAFETY: the buffer the call gave, freed once.
            let blob = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            // SAFETY: the buffer the call gave, freed once.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            let held = held.map(|(start, end, guard)| {
                drop(guard);
                (start, end)
            });
            let mut whole = 0u64;
            // SAFETY: a live out-pointer.
            let rc = unsafe {
                ffi_live_view_reader(
                    reader,
                    view,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut whole,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());

            // The blob's chunk ends are the segments' sizes.
            let w = crate::jvm_reader::lookup(whole, "test").unwrap();
            let base = crate::jvm_reader::lookup(reader, "test")
                .unwrap()
                .live_docs
                .len();
            let readers = &w.reader.segment_readers()[base..];
            let count = i32::from_le_bytes(blob[8..12].try_into().unwrap());
            let mut at = 12;
            let mut sizes = Vec::new();
            for _ in 0..count {
                let deleted = i32::from_le_bytes(blob[at + 4..at + 8].try_into().unwrap());
                at += 8 + 4 * deleted as usize;
                let chunks = i32::from_le_bytes(blob[at..at + 4].try_into().unwrap());
                at += 4;
                let mut from = 0;
                for _ in 0..chunks {
                    let end = i32::from_le_bytes(blob[at..at + 4].try_into().unwrap());
                    sizes.push(end - from);
                    from = end;
                    at += 4;
                }
            }
            assert_eq!(at, blob.len());
            assert_eq!(sizes, readers.iter().map(|r| r.max_doc).collect::<Vec<_>>());
            if let Some((start, end)) = held {
                let ends: Vec<i32> = sizes
                    .iter()
                    .scan(0, |at, n| {
                        *at += n;
                        Some(*at)
                    })
                    .collect();
                assert!(
                    ends.contains(&end),
                    "round {round}: {ends:?} lack {start}..{end}"
                );
            }
            leaves_seen.push(readers.len());

            // Every live document's postings, by sequence number.
            let mut got = Vec::new();
            let (mut doc_count, mut sum_doc_freq) = (0i64, 0i64);
            for (r, live) in readers.iter().zip(&w.live_docs[base..]) {
                let mut seq = Vec::new();
                let mut dv = r.numeric_doc_values("_seq_no").unwrap().unwrap();
                for d in 0..r.max_doc {
                    seq.push(dv.advance_exact(d).unwrap().then(|| dv.long_value()));
                }
                let Some(terms) = r.terms("body").unwrap() else {
                    continue;
                };
                doc_count += i64::from(terms.doc_count());
                sum_doc_freq += terms.sum_doc_freq();
                let mut te = terms.iterator().unwrap();
                while let Some(term) = te.next().unwrap() {
                    let term = String::from_utf8(term.to_vec()).unwrap();
                    let mut pe = te.postings(PostingsFlags::All).unwrap();
                    loop {
                        let d = pe.next_doc().unwrap();
                        if d == lucene_search::reader::NO_MORE_DOCS {
                            break;
                        }
                        let p = pe.next_position().unwrap();
                        assert_eq!(pe.freq(), 1);
                        if live.as_ref().is_none_or(|l| l.get(d as usize)) {
                            got.push((
                                seq[d as usize].unwrap(),
                                term.clone(),
                                p,
                                pe.start_offset(),
                                pe.end_offset(),
                            ));
                        }
                    }
                }
            }
            got.sort();
            let mut want = Vec::new();
            for i in 0..added {
                if ids[i + 1..].contains(&ids[i]) {
                    continue;
                }
                for (p, word) in text(i).iter().enumerate() {
                    let p = p as i32;
                    want.push((i as i64, word.to_string(), p, p * 4, p * 4 + 3));
                }
            }
            want.sort();
            assert_eq!(got, want, "round {round}");
            assert_eq!(
                doc_count, added as i64,
                "round {round}: deleted ones included"
            );
            assert_eq!(
                sum_doc_freq,
                (0..added).map(|i| text(i).len() as i64).sum::<i64>()
            );
            for term in words {
                let q = crate::jvm_reader::tests::term_blob("body", term);
                let (_, live_total) = search_live(reader, view, &q).unwrap();
                let want = (0..added)
                    .filter(|&i| !ids[i + 1..].contains(&ids[i]) && text(i).contains(&term))
                    .count();
                assert_eq!(live_total, want as i64, "round {round}: {term}");

                // A handle for the query and an aggregation filtering on
                // another term: the chunks it writes carry only those terms
                // (one written with every term may be reused), and sorts and
                // aggregations over it agree with `whole`.
                let other = crate::jvm_reader::tests::term_blob("body", words[round % 5]);
                let tree = crate::jvm_aggs::tests::histogram_of_filters("_seq_no", 5.0, &other);
                let mut lean = 0u64;
                // SAFETY: live slices and out-pointer.
                let rc = unsafe {
                    ffi_live_view_reader(
                        reader,
                        view,
                        q.as_ptr(),
                        q.len(),
                        tree.as_ptr(),
                        tree.len(),
                        std::ptr::null(),
                        0,
                        &mut lean,
                    )
                };
                assert_eq!(rc, 0, "{}", crate::error::last_error());
                {
                    let body_terms = |r: &lucene_search::directory_reader::SegmentReader| {
                        let mut out = Vec::new();
                        if let Some(terms) = r.terms("body").unwrap() {
                            let mut te = terms.iterator().unwrap();
                            while let Some(t) = te.next().unwrap() {
                                out.push(String::from_utf8(t.to_vec()).unwrap());
                            }
                        }
                        out
                    };
                    let l = crate::jvm_reader::lookup(lean, "test").unwrap();
                    let w = crate::jvm_reader::lookup(whole, "test").unwrap();
                    for r in &l.reader.segment_readers()[base..] {
                        let got = body_terms(r);
                        let requested = got
                            .iter()
                            .all(|t| [term, words[round % 5]].contains(&t.as_str()));
                        // The very chunk `whole` holds (the same core).
                        let reused = w.reader.segment_readers()[base..].iter().any(|c| {
                            c.core_cache_helper().key_id() == r.core_cache_helper().key_id()
                        });
                        assert!(
                            requested || reused,
                            "round {round}: only the request's terms in {}: {got:?}",
                            r.segment_name
                        );
                    }
                }
                assert_eq!(
                    scored_hits(lean, &q),
                    scored_hits(whole, &q),
                    "round {round}: {term} scores"
                );
                let aggs = |r: u64| crate::jvm_aggs::aggregate_tree_blobs(r, &q, &tree).unwrap();
                assert_eq!(aggs(lean), aggs(whole), "round {round}: {term} aggs");
                let sort = crate::jvm_reader::tests::sort_blob_terms(
                    &[(crate::jvm_reader::SORT_LONG, 1, "_seq_no", 0)],
                    None,
                );
                let sorted = |r: u64| {
                    let out =
                        crate::jvm_reader::search_sorted_blobs(r, &q, &sort, 10, i64::MAX).unwrap();
                    (out.hits, out.total)
                };
                let (hits, total) = sorted(lean);
                assert_eq!(
                    (hits.clone(), total),
                    sorted(whole),
                    "round {round}: {term}"
                );
                assert_eq!(total, want as i64);
            }
            // A prefix query reads the buffered terms it matches; a
            // rescore's query, given with the request, its own terms.
            let lean_of = |q: &[u8], extra: &[u8]| {
                let mut lean = 0u64;
                // SAFETY: live slices and out-pointer.
                let rc = unsafe {
                    ffi_live_view_reader(
                        reader,
                        view,
                        q.as_ptr(),
                        q.len(),
                        std::ptr::null(),
                        0,
                        extra.as_ptr(),
                        extra.len(),
                        &mut lean,
                    )
                };
                assert_eq!(rc, 0, "{}", crate::error::last_error());
                assert_ne!(lean, whole, "round {round}: a handle of its own");
                lean
            };
            // The terms of the chunks a handle wrote: a chunk of `whole`'s,
            // written with every term, may be reused.
            let held = |lean: u64| {
                let l = crate::jvm_reader::lookup(lean, "test").unwrap();
                let w = crate::jvm_reader::lookup(whole, "test").unwrap();
                let mut held = std::collections::BTreeSet::new();
                for r in &l.reader.segment_readers()[base..] {
                    let reused = w.reader.segment_readers()[base..]
                        .iter()
                        .any(|c| c.core_cache_helper().key_id() == r.core_cache_helper().key_id());
                    if reused {
                        continue;
                    }
                    if let Some(terms) = r.terms("body").unwrap() {
                        let mut te = terms.iterator().unwrap();
                        while let Some(t) = te.next().unwrap() {
                            held.insert(String::from_utf8(t.to_vec()).unwrap());
                        }
                    }
                }
                held
            };
            // A plain search reads the chunks the view planned (with their
            // kept overlays from the second time) and the documents after:
            // the same hits and scores as the reader with postings.
            for word in ["alpha", "gamma", "eps", "zz"] {
                let q = crate::jvm_reader::tests::term_blob("body", word);
                for _ in 0..2 {
                    let (hits, total) = search_live(reader, view, &q).unwrap();
                    let mut hits: Vec<(i32, u32)> =
                        hits.iter().map(|&(d, s)| (d, s.to_bits())).collect();
                    hits.sort_unstable();
                    let (mut want, want_total) = scored_hits(whole, &q);
                    want.sort_unstable();
                    assert_eq!((hits, total), (want, want_total), "round {round}: {word}");
                }
            }
            for prefix in ["d", "ga", "e", "zz"] {
                let q = crate::jvm_reader::tests::prefix_blob("body", prefix);
                let lean = lean_of(&q, &[]);
                assert_eq!(
                    scored_hits(lean, &q),
                    scored_hits(whole, &q),
                    "round {round}: {prefix}*"
                );
                assert!(
                    held(lean).iter().all(|t| t.starts_with(prefix)),
                    "round {round}: {prefix}* holds {:?}",
                    held(lean)
                );
            }
            let (q, rescore) = (
                crate::jvm_reader::tests::term_blob("body", words[round % 5]),
                crate::jvm_reader::tests::term_blob("body", words[(round + 2) % 5]),
            );
            let mut extra = (rescore.len() as i32).to_le_bytes().to_vec();
            extra.extend_from_slice(&rescore);
            let lean = lean_of(&q, &extra);
            assert_eq!(
                scored_hits(lean, &rescore),
                scored_hits(whole, &rescore),
                "round {round}: the rescore's query"
            );
            // A term of one chunk: the others have its field but not it,
            // and count in its statistics all the same.
            let q = crate::jvm_reader::tests::term_blob("_id", &ids[added - 1]);
            let mut lean = 0u64;
            // SAFETY: live slices and out-pointer.
            let rc = unsafe {
                ffi_live_view_reader(
                    reader,
                    view,
                    q.as_ptr(),
                    q.len(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut lean,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let (hits, total) = scored_hits(lean, &q);
            assert_eq!(total, 1, "round {round}");
            assert_eq!((hits, total), scored_hits(whole, &q), "round {round}");
            assert_eq!(ffi_live_view_close(view), 0);
        }
        // A few chunks per buffer, not one per view.
        assert!(leaves_seen.iter().all(|&n| n <= 12), "{leaves_seen:?}");
        assert!(leaves_seen.iter().any(|&n| n > 1), "{leaves_seen:?}");

        // A buffer published after views of it opened: one whose chunks
        // were all written reads them; one whose last chunk was not cannot.
        let open = || {
            let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
            // SAFETY: live out-pointers.
            let rc =
                unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            // SAFETY: the buffer the call gave, freed once.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            view
        };
        let written = open();
        add(added, "last");
        let unwritten = open();
        commit_settled(h);
        let mut whole = 0u64;
        // SAFETY: a live out-pointer.
        let rc = unsafe {
            ffi_live_view_reader(
                reader,
                written,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut whole,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let w = crate::jvm_reader::lookup(whole, "test").unwrap();
        let base = crate::jvm_reader::lookup(reader, "test")
            .unwrap()
            .live_docs
            .len();
        assert_eq!(
            w.reader.segment_readers().len(),
            base + leaves_seen.last().unwrap()
        );
        // SAFETY: a live out-pointer.
        let rc = unsafe {
            ffi_live_view_reader(
                reader,
                unwritten,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut whole,
            )
        };
        assert_ne!(rc, 0);
        assert!(crate::error::last_error().contains("being flushed"));
        assert_eq!(ffi_live_view_close(written), 0);
        assert_eq!(ffi_live_view_close(unwritten), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
    }

    /// How long a view's reader of a large growing buffer takes to build,
    /// a few hundred documents after the last. `LIVE_DOCS` sets the
    /// buffer's size (default 100 000).
    #[test]
    #[ignore = "timing; run with --release --ignored --nocapture"]
    fn view_reader_of_a_growing_buffer_timing() {
        use crate::live::{
            ffi_engine_writer_live_view_open, ffi_live_view_close, ffi_live_view_reader,
        };
        let docs: usize = std::env::var("LIVE_DOCS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000);
        let tmp = empty_index("engine-writer-live-timing");
        let h = open_with(&tmp, 2048.0, OPEN_LIVE_SEARCH, i32::MAX);
        let f = setup(h);
        let vocab: Vec<String> = (0..2000).map(|i| format!("w{i}")).collect();
        let mut x = 7u64;
        let mut add = |i: usize| {
            let words: Vec<&str> = (0..15)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    // Skewed: low numbers far more often.
                    let r = (x % 2000) as usize;
                    vocab[r * r / 2000].as_str()
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            let b = doc(
                Blob::default().u8(OP_ADD).i32(1),
                &f,
                &format!("d{i}"),
                &words,
                i as i64,
                false,
            );
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
        };
        let gen = commit_settled(h);
        let reader = open_reader(&tmp, gen);
        (0..docs).for_each(&mut add);
        let mut n = docs;
        for round in 0..25 {
            for _ in 0..300 {
                add(n);
                n += 1;
            }
            let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
            let t = std::time::Instant::now();
            // SAFETY: live out-pointers.
            let rc =
                unsafe { ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len) };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let opened = t.elapsed();
            // SAFETY: the buffer the call gave, freed once.
            unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
            let t = std::time::Instant::now();
            let mut whole = 0u64;
            // SAFETY: a live out-pointer.
            let rc = unsafe {
                ffi_live_view_reader(
                    reader,
                    view,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    &mut whole,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            let built = t.elapsed();
            let w = crate::jvm_reader::lookup(whole, "test").unwrap();
            let sizes: Vec<i32> = w
                .reader
                .segment_readers()
                .iter()
                .map(|r| r.max_doc)
                .collect();
            eprintln!("round {round}: open {opened:?}, reader {built:?}, segments {sizes:?}");
            assert_eq!(ffi_live_view_close(view), 0);
        }
    }

    /// The 1M OpenSearch bench's documents (`bench_nodes.py gen`), mapped as
    /// OpenSearch maps them, buffered by four threads, then one refresh
    /// timed. `FLUSH_DOCS` sets how many (default 500 000), `THREADS` the
    /// buffering threads (default 4). With `LIVE_ROUNDS` set, live search is
    /// on and that many views' readers are timed instead, `LIVE_STEP` more
    /// documents (default 2000) buffered between them.
    #[test]
    #[ignore = "timing; run with --release --ignored --nocapture"]
    fn refresh_of_the_bench_corpus_timing() {
        fn spec(name: &str, io: u8, omit_norms: u8, dv: u8, dims: i32, bytes: i32) -> Vec<u8> {
            Blob::default()
                .bytes(name.as_bytes())
                .u8(io)
                .u8(omit_norms)
                .u8(0)
                .u8(0)
                .u8(dv)
                .u8(0)
                .i32(dims)
                .i32(dims)
                .i32(bytes)
                .i32(0)
                .u8(0)
                .i32(0)
                .0
        }
        fn b36(mut n: usize) -> String {
            let d = b"0123456789abcdefghijklmnopqrstuvwxyz";
            let mut s = Vec::new();
            loop {
                s.push(d[n % 36]);
                n /= 36;
                if n == 0 {
                    break;
                }
            }
            s.reverse();
            String::from_utf8(s).unwrap()
        }
        fn zipf_cum(n: usize) -> Vec<f64> {
            let mut acc = 0.0;
            (1..=n)
                .map(|r| {
                    acc += 1.0 / (r as f64).powf(1.07);
                    acc
                })
                .collect()
        }
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn unit(&mut self) -> f64 {
                (self.next() >> 11) as f64 / (1u64 << 53) as f64
            }
            fn pick(&mut self, cum: &[f64]) -> usize {
                let x = self.unit() * cum[cum.len() - 1];
                cum.partition_point(|&c| c < x).min(cum.len() - 1)
            }
            fn range(&mut self, lo: u64, hi: u64) -> u64 {
                lo + self.next() % (hi - lo)
            }
        }
        fn text(b: Blob, field: i32, words: &[&str]) -> Blob {
            let mut terms: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
            for (p, w) in words.iter().enumerate() {
                terms.entry(w).or_default().push(p as i32);
            }
            let mut b = b
                .i32(field)
                .u8(1)
                .i64(words.len() as i64)
                .i32(terms.len() as i32);
            for (w, ps) in &terms {
                b = b.bytes(w.as_bytes()).i32(ps.len() as i32).u8(1);
                for &p in ps {
                    b = b.i32(p);
                }
            }
            b
        }
        fn keyword(b: Blob, field: i32, v: &str) -> Blob {
            b.i32(field).u8(0).i32(1).bytes(v.as_bytes()).i32(1).u8(0)
        }
        fn point(v: i64) -> [u8; 8] {
            ((v as u64) ^ (1 << 63)).to_be_bytes()
        }

        let docs: usize = std::env::var("FLUSH_DOCS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500_000);
        let tmp = TempDir::new("refresh-timing");
        let live_rounds: usize = std::env::var("LIVE_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let h = open_with(
            &tmp,
            4096.0,
            if live_rounds > 0 { OPEN_LIVE_SEARCH } else { 0 },
            i32::MAX,
        );
        let id = register(h, &spec("_id", 1, 1, 0, 0, 0)).unwrap();
        let source = register(h, &spec("_source", 0, 1, 0, 0, 0)).unwrap();
        let body = register(h, &spec("body", 3, 0, 0, 0, 0)).unwrap();
        let title = register(h, &spec("title", 3, 0, 0, 0, 0)).unwrap();
        let tag = register(h, &spec("tag", 1, 1, 4, 0, 0)).unwrap();
        let category = register(h, &spec("category", 1, 1, 4, 0, 0)).unwrap();
        let num = register(h, &spec("num", 0, 1, 5, 1, 8)).unwrap();
        let price = register(h, &spec("price", 0, 1, 5, 1, 8)).unwrap();
        let ts = register(h, &spec("ts", 0, 1, 5, 1, 8)).unwrap();
        let seq = register(h, &spec("_seq_no", 0, 1, 1, 1, 8)).unwrap();
        let term = register(h, &spec("_primary_term", 0, 1, 1, 0, 0)).unwrap();
        let version = register(h, &spec("_version", 0, 1, 1, 0, 0)).unwrap();
        let words: Vec<String> = (0..200_000).map(|i| format!("t{}", b36(i))).collect();
        let cum = zipf_cum(words.len());
        let tag_cum = zipf_cum(1000);
        let threads: usize = std::env::var("THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let buffer = |from: usize, to: usize| {
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (words, cum, tag_cum) = (&words, &cum, &tag_cum);
                    s.spawn(move || {
                    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (t as u64 + 1 + from as u64));
                    for i in (from + t..to).step_by(threads) {
                        let body_words: Vec<&str> = (0..rng.range(20, 121))
                            .map(|_| words[rng.pick(cum)].as_str())
                            .collect();
                        let title_words: Vec<&str> = (0..rng.range(2, 9))
                            .map(|_| words[rng.pick(cum)].as_str())
                            .collect();
                        let tag_v = format!("g{}", b36(rng.pick(tag_cum)));
                        let cat_v = format!("c{}", rng.range(0, 50));
                        let (n, p, when) = (
                            rng.range(0, 1_000_000) as i64,
                            rng.unit() * 1000.0,
                            1_700_000_000_000 + rng.range(0, 365 * 24 * 60) as i64 * 60_000,
                        );
                        let id_v = i.to_string();
                        let src = format!(
                            r#"{{"body":"{}","title":"{}","tag":"{tag_v}","category":"{cat_v}","num":{n},"price":{p:.2},"ts":{when}}}"#,
                            body_words.join(" "),
                            title_words.join(" ")
                        );
                        let b = Blob::default()
                            .u8(OP_ADD)
                            .i32(1)
                            .i32(2)
                            .i32(id)
                            .u8(1)
                            .bytes(id_v.as_bytes())
                            .i32(source)
                            .u8(1)
                            .bytes(src.as_bytes())
                            .i32(5);
                        let b = keyword(b, id, &id_v);
                        let b = text(b, body, &body_words);
                        let b = text(b, title, &title_words);
                        let b = keyword(b, tag, &tag_v);
                        let b = keyword(b, category, &cat_v);
                        let pd = (p * 100.0).round() as i64;
                        let b = b
                            .i32(8)
                            .i32(tag)
                            .u8(1)
                            .bytes(tag_v.as_bytes())
                            .i32(category)
                            .u8(1)
                            .bytes(cat_v.as_bytes())
                            .i32(num)
                            .u8(0)
                            .i64(n)
                            .i32(price)
                            .u8(0)
                            .i64(pd)
                            .i32(ts)
                            .u8(0)
                            .i64(when)
                            .i32(seq)
                            .u8(0)
                            .i64(i as i64)
                            .i32(term)
                            .u8(0)
                            .i64(1)
                            .i32(version)
                            .u8(0)
                            .i64(1)
                            .i32(4)
                            .i32(num)
                            .bytes(&point(n))
                            .i32(price)
                            .bytes(&point(pd))
                            .i32(ts)
                            .bytes(&point(when))
                            .i32(seq)
                            .bytes(&point(i as i64));
                        assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
                    }
                });
                }
            });
        };
        if live_rounds > 0 {
            use crate::live::{
                ffi_engine_writer_live_view_open, ffi_live_view_close, ffi_live_view_reader,
            };
            let gen = commit_settled(h);
            let reader = open_reader(&tmp, gen);
            let step: usize = std::env::var("LIVE_STEP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000);
            let mut n = 0;
            let mut total = std::time::Duration::ZERO;
            buffer(0, docs);
            n += docs;
            // LIVE_SEARCH=1: each round also times the bench's request
            // shapes over a handle made for each, by phase, and at the end
            // over the same documents committed.
            let searches = std::env::var("LIVE_SEARCH").is_ok_and(|v| v == "1");
            let term_q = crate::jvm_reader::tests::term_blob("body", "t0");
            let price = crate::jvm_reader::tests::sort_blob_terms(
                &[(crate::jvm_reader::SORT_LONG, 0, "price", 0)],
                None,
            );
            let tags = crate::jvm_aggs::tests::terms_tree("tag", 10);
            let shapes = |r: u64| -> [std::time::Duration; 3] {
                let t = std::time::Instant::now();
                std::hint::black_box(super::tests::scored_hits(r, &term_q));
                let term = t.elapsed();
                let t = std::time::Instant::now();
                let sorted =
                    crate::jvm_reader::search_sorted_blobs(r, &term_q, &price, 10, 10_000).unwrap();
                std::hint::black_box(sorted.total);
                let sort = t.elapsed();
                let t = std::time::Instant::now();
                std::hint::black_box(
                    crate::jvm_aggs::aggregate_tree_blobs(r, &term_q, &tags).unwrap(),
                );
                [term, sort, t.elapsed()]
            };
            let mut sums = [std::time::Duration::ZERO; 10];
            let mut builds: Vec<std::time::Duration> = Vec::new();
            // LIVE_LOAD=1: documents keep arriving, LIVE_STEP at a time, while
            // the views are read, as under the bench's bulk load.
            let load = std::env::var("LIVE_LOAD").is_ok_and(|v| v == "1");
            let stop = std::sync::atomic::AtomicBool::new(false);
            let loaded = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                if load {
                    s.spawn(|| {
                        let mut at = 10_000_000;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            buffer(at, at + step);
                            at += step;
                            loaded.fetch_add(step, std::sync::atomic::Ordering::Relaxed);
                        }
                    });
                }
                for round in 0..live_rounds {
                    let (mut view, mut ptr, mut len) = (0u64, std::ptr::null_mut(), 0usize);
                    // SAFETY: live out-pointers.
                    let rc = unsafe {
                        ffi_engine_writer_live_view_open(h, gen, &mut view, &mut ptr, &mut len)
                    };
                    assert_eq!(rc, 0, "{}", crate::error::last_error());
                    // SAFETY: the buffer the call gave, freed once.
                    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
                    if searches {
                        // The plain search of the term, as a scored query
                        // with no sort or aggregation runs.
                        let t = std::time::Instant::now();
                        std::hint::black_box(super::tests::search_live(reader, view, &term_q));
                        sums[9] += t.elapsed();
                        let _ = crate::live::phases::take();
                    }
                    if !searches {
                        let t = std::time::Instant::now();
                        let mut whole = 0u64;
                        // SAFETY: a live out-pointer.
                        let rc = unsafe {
                            ffi_live_view_reader(
                                reader,
                                view,
                                std::ptr::null(),
                                0,
                                std::ptr::null(),
                                0,
                                std::ptr::null(),
                                0,
                                &mut whole,
                            )
                        };
                        assert_eq!(rc, 0, "{}", crate::error::last_error());
                        let took = t.elapsed();
                        total += took;
                        let w = crate::jvm_reader::lookup(whole, "test").unwrap();
                        eprintln!(
                            "round {round} ({n} docs): reader {took:?}, {} leaves",
                            w.reader.segment_readers().len()
                        );
                    }
                    if searches {
                        crate::live::phases::take();
                        let t = std::time::Instant::now();
                        let mut lean = 0u64;
                        // SAFETY: live slices and out-pointer.
                        let rc = unsafe {
                            ffi_live_view_reader(
                                reader,
                                view,
                                term_q.as_ptr(),
                                term_q.len(),
                                tags.as_ptr(),
                                tags.len(),
                                std::ptr::null(),
                                0,
                                &mut lean,
                            )
                        };
                        assert_eq!(rc, 0, "{}", crate::error::last_error());
                        let build = t.elapsed();
                        let phases = crate::live::phases::take();
                        let [term, sort, agg] = shapes(lean);
                        eprintln!(
                        "round {round} ({n} docs): lean reader {build:?} ({}), term {term:?}, sort {sort:?}, agg {agg:?}",
                        crate::live::phases::NAMES
                            .iter()
                            .zip(phases)
                            .map(|(n, d)| format!("{n} {d:?}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                        sums[0] += build;
                        builds.push(build);
                        for (s, d) in sums[1..6].iter_mut().zip(phases) {
                            *s += d;
                        }
                        for (s, d) in sums[6..].iter_mut().zip([term, sort, agg]) {
                            *s += d;
                        }
                    }
                    assert_eq!(ffi_live_view_close(view), 0);
                    // LIVE_PAUSE: milliseconds between searches.
                    if let Some(ms) = std::env::var("LIVE_PAUSE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                    {
                        std::thread::sleep(std::time::Duration::from_millis(ms));
                    }
                    if load {
                        n = docs + loaded.load(std::sync::atomic::Ordering::Relaxed);
                    } else {
                        buffer(n, n + step);
                        n += step;
                    }
                }
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            eprintln!("views' readers took {total:?} in all");
            if searches && live_rounds > 0 {
                let r = live_rounds as u32;
                eprintln!(
                    "mean over {r} rounds: lean reader {:?} (chunks {:?}, snapshot {:?}, overlay {:?}, append {:?}, live_docs {:?}), term {:?}, sort {:?}, agg {:?}, plain term {:?}",
                    sums[0] / r, sums[1] / r, sums[2] / r, sums[3] / r, sums[4] / r, sums[5] / r,
                    sums[6] / r, sums[7] / r, sums[8] / r, sums[9] / r
                );
                builds.sort_unstable();
                let at = |q: f64| builds[((builds.len() as f64 - 1.0) * q).round() as usize];
                eprintln!(
                    "lean reader p50 {:?}, p90 {:?}, max {:?}",
                    at(0.5),
                    at(0.9),
                    at(1.0)
                );
                let gen = commit_settled(h);
                let committed = open_reader(&tmp, gen);
                for pass in 0..3 {
                    eprintln!("committed ({n} docs) pass {pass}: {:?}", shapes(committed));
                }
                assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(committed), 0);
            }
            assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
            ffi_engine_writer_close(h);
            return;
        }
        let start = std::time::Instant::now();
        buffer(0, docs);
        let buffered = start.elapsed();
        let s = stats(h);
        fn cpu_ticks() -> u64 {
            let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
            let fields: Vec<&str> = stat
                .rsplit(')')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            fields
                .get(11..13)
                .map_or(0, |f| f.iter().map(|v| v.parse::<u64>().unwrap_or(0)).sum())
        }
        let cpu = cpu_ticks();
        let refresh = std::time::Instant::now();
        commit(h, &[]);
        let took = refresh.elapsed();
        let cpu = cpu_ticks() - cpu;
        let after = stats(h);
        eprintln!(
            "docs {docs}: buffered in {buffered:?}, {} MB charged; refresh {took:?} ({} cpu ticks), {} segments",
            s[STAT_RAM_BYTES] >> 20,
            cpu,
            after[STAT_SEGMENTS]
        );
        let mut fdt: Vec<u64> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".fdt"))
            .map(|e| e.metadata().unwrap().len() >> 20)
            .collect();
        fdt.sort_unstable();
        eprintln!("stored MB per segment: {fdt:?}");
        ffi_engine_writer_close(h);
    }

    fn knn_vector(i: usize) -> [f32; 4] {
        let f = i as f32;
        [
            (f * 0.37).sin(),
            (f * 0.11).cos(),
            ((i * 7) % 13) as f32 / 13.0,
            0.5,
        ]
    }

    fn knn_score(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        1.0 / (1.0 + d)
    }

    /// **k-NN's derived source.** In a k-NN index ([`OPEN_KNN`]) a vector
    /// field marked [`DERIVED_SOURCE_ATTRIBUTE`] makes every flushed segment
    /// list it under `derived_vector_fields`, a forced merge keeps the list,
    /// and a read of the masked `_source` gets the vector back, before and
    /// after the merge; the segments record k-NN's codec over `Lucene104`.
    #[test]
    fn derived_source_fields_are_listed_kept_by_merges_and_read_back() {
        use lucene_search::derived_source::DERIVED_VECTOR_FIELDS;
        let tmp = empty_index("engine-writer-derived-source");
        let h = open(&tmp, OPEN_KNN);
        let f = setup(h);
        let source = register(h, &field("_source", 0, 1, 0, 0)).unwrap();
        let spec = Blob::default()
            .bytes(b"v")
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(4)
            .u8(0)
            .u8(0)
            .u8(1)
            .i32(8)
            .i32(50)
            .u8(0)
            .i32(2)
            .bytes(VECTORS_FORMAT_ATTRIBUTE.as_bytes())
            .bytes(b"hnsw:8:50")
            .bytes(DERIVED_SOURCE_ATTRIBUTE.as_bytes())
            .bytes(b"true")
            .0;
        register(h, &spec).unwrap();
        let mut seq = 0;
        for batch in 0..2 {
            let ids: Vec<usize> = (batch * 5..batch * 5 + 5).collect();
            let mut b = Blob::default().u8(OP_ADD).i32(ids.len() as i32);
            for &i in &ids {
                let id = format!("d{i}");
                let masked = format!(r#"{{"n":{i},"v":1}}"#);
                b = b
                    .i32(2)
                    .i32(f.id)
                    .u8(1)
                    .bytes(id.as_bytes())
                    .i32(source)
                    .u8(1)
                    .bytes(masked.as_bytes())
                    .i32(1)
                    .i32(f.id)
                    .u8(0)
                    .i32(1)
                    .bytes(id.as_bytes())
                    .i32(1)
                    .u8(0)
                    .i32(1)
                    .i32(f.seq)
                    .u8(0)
                    .i64(seq)
                    .i32(0);
                seq += 1;
            }
            for &i in &ids {
                b = b.i32(1).bytes(b"v").u8(0).i32(4);
                for x in knn_vector(i) {
                    b = b.f32(x);
                }
            }
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            commit(h, &[]);
        }
        commit_settled(h);
        let read_all = |want_segments: usize| {
            let dir = FsDirectory::open(tmp.path());
            let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
            assert_eq!(reader.segment_readers().len(), want_segments);
            let mut seen = 0;
            for seg in reader.segment_readers() {
                assert!(seg.derived_source().is_some(), "{}", seg.segment_name);
                for d in 0..seg.max_doc {
                    let mut visitor = lucene_codecs::stored_fields::DocumentVisitor::all();
                    assert!(seg.visit_source_document(d, &mut visitor).unwrap());
                    let doc = visitor.into_document();
                    let src = doc
                        .fields
                        .iter()
                        .find(|f| f.field_number == source)
                        .unwrap();
                    let lucene_codecs::stored_fields::FieldValue::Binary(bytes) = &src.value else {
                        panic!("binary _source");
                    };
                    let text = String::from_utf8(bytes.clone()).unwrap();
                    let n: usize = text[5..text.find(',').unwrap()].parse().unwrap();
                    let want: Vec<String> = knn_vector(n)
                        .iter()
                        .map(|x| lucene_analysis::attributes::java_float_to_string(*x))
                        .collect();
                    assert_eq!(text, format!(r#"{{"n":{n},"v":[{}]}}"#, want.join(",")));
                    seen += 1;
                }
            }
            assert_eq!(seen, 10);
        };
        read_all(2);
        let mut gen = 0;
        // SAFETY: a live handle and out-pointer.
        let rc = unsafe { ffi_engine_writer_force_merge(h, 1, 0, &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        commit_settled(h);
        ffi_engine_writer_close(h);
        check(&tmp);
        read_all(1);
        // The plugin's fetch: the document leaves the derived `_source` out, and the second call
        // reads it with its vectors back, as the segment reader splices it.
        {
            let dir = FsDirectory::open(tmp.path());
            let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
            let file = format!(
                "segments_{}",
                lucene_util::base36::to_base36(infos.generation)
            );
            let bytes = std::fs::read(tmp.path().join(&file)).unwrap();
            let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
            let seg = &reader.segment_readers()[0];
            let max_docs = [seg.max_doc];
            let counts = [0usize];
            let path = tmp.path_str();
            let mut jh = 0u64;
            // SAFETY: every pointer is valid for its length.
            let rc = unsafe {
                crate::jvm_reader::ffi_open_jvm_reader(
                    path.as_ptr().cast(),
                    path.len(),
                    bytes.as_ptr(),
                    bytes.len(),
                    infos.generation,
                    0,
                    max_docs.as_ptr(),
                    1,
                    std::ptr::null(),
                    counts.as_ptr(),
                    &mut jh,
                )
            };
            assert_eq!(rc, 0, "{}", crate::error::last_error());
            for d in 0..seg.max_doc {
                let blob = crate::jvm_fetch::document_blob(jh, 0, d).unwrap();
                let mut want = source.to_le_bytes().to_vec();
                want.push(crate::jvm_fetch::DERIVED_SOURCE);
                assert!(
                    blob.windows(want.len()).any(|w| w == want.as_slice()),
                    "document {d}: _source left out"
                );
                let mut visitor = lucene_codecs::stored_fields::DocumentVisitor::all();
                assert!(seg.visit_source_document(d, &mut visitor).unwrap());
                let spliced = visitor
                    .into_document()
                    .fields
                    .into_iter()
                    .find(|f| f.field_number == source)
                    .unwrap()
                    .value;
                assert_eq!(
                    lucene_codecs::stored_fields::FieldValue::Binary(
                        crate::jvm_fetch::source_blob(jh, 0, d).unwrap()
                    ),
                    spliced
                );
            }
            assert!(crate::jvm_fetch::source_blob(jh, 0, seg.max_doc).is_err());
            crate::jvm_reader::ffi_close_jvm_reader(jh);
        }
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let name = &reader.segment_readers()[0].segment_name;
        let infos = lucene_index::segment_infos::read_latest(&dir).unwrap();
        let sci = infos
            .segments
            .iter()
            .find(|s| &s.segment_name == name)
            .unwrap();
        let si = lucene_index::segment_info::parse(
            &std::fs::read(tmp.path().join(format!("{name}.si"))).unwrap(),
            &sci.segment_id,
        )
        .unwrap();
        let attr = |key: &str| {
            si.attributes
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(attr(DERIVED_VECTOR_FIELDS), Some("v"));
        assert_eq!(attr(KNN_DELEGATE_CODEC_ATTRIBUTE), Some("Lucene104"));
        assert_eq!(sci.codec_name, KNN_CODEC);
    }

    /// **A vector field is written in its mapping's format and searched
    /// natively.** The plugin's field spec (dimension, encoding, similarity,
    /// HNSW parameters) and its documents' vectors section make two segments
    /// of Lucene99 HNSW vectors that pass `CheckIndex`; the reader's
    /// whole-index KNN ([`crate::jvm_knn::ffi_jvm_reader_knn`]) finds the
    /// exact top `k`, and a per-segment KNN node (39) `k` per segment.
    #[test]
    fn a_vector_field_is_written_in_its_format_and_searched() {
        let tmp = empty_index("engine-writer-knn");
        let h = open(&tmp, 0);
        let f = setup(h);
        let spec = Blob::default()
            .bytes(b"v")
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .u8(0)
            .i32(0)
            .i32(0)
            .i32(0)
            .i32(4)
            .u8(0)
            .u8(0)
            .u8(1)
            .i32(8)
            .i32(50)
            .u8(0)
            .i32(1)
            .bytes(VECTORS_FORMAT_ATTRIBUTE.as_bytes())
            .bytes(b"hnsw:8:50")
            .0;
        register(h, &spec).unwrap();
        let mut seq = 0;
        for batch in 0..2 {
            let ids: Vec<usize> = (batch * 60..batch * 60 + 60).collect();
            let mut b = Blob::default().u8(OP_ADD).i32(ids.len() as i32);
            for &i in &ids {
                b = doc(b, &f, &format!("d{i}"), &["w"], seq, false);
                seq += 1;
            }
            for &i in &ids {
                b = b.i32(1).bytes(b"v").u8(0).i32(4);
                for x in knn_vector(i) {
                    b = b.f32(x);
                }
            }
            assert_eq!(apply(h, &b.0), 0, "{}", crate::error::last_error());
            commit(h, &[]);
        }
        let gen = commit_settled(h);
        ffi_engine_writer_close(h);
        check(&tmp);

        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), 2);
        for seg in reader.segment_readers() {
            let v = seg
                .field_infos()
                .fields
                .iter()
                .find(|i| i.name == "v")
                .unwrap();
            let attr = |k: &str| {
                v.attributes
                    .iter()
                    .find(|(a, _)| a == k)
                    .map(|(_, b)| b.as_str())
            };
            assert_eq!(attr(VECTORS_FORMAT_ATTRIBUTE), Some("hnsw:8:50"));
            assert_eq!(
                attr("PerFieldKnnVectorsFormat.format"),
                Some("Lucene99HnswVectorsFormat")
            );
        }

        let r = open_reader(&tmp, gen);
        let target = [0.2f32, -0.4, 0.6, 0.5];
        let mut exact: Vec<f32> = (0..120)
            .map(|i| knn_score(&target, &knn_vector(i)))
            .collect();
        exact.sort_by(|a, b| b.total_cmp(a));

        let mut request = Blob::default()
            .bytes(b"v")
            .u8(crate::jvm_queries::KNN_TARGET_FLOAT)
            .i32(4);
        for x in target {
            request = request.f32(x);
        }
        let request = request.i32(5).i32(40).u8(0).u8(0).i32(0).0;
        let (mut docs, mut scores, mut n) = ([0i32; 5], [0f32; 5], 0usize);
        // SAFETY: live request and buffers of 5.
        let rc = unsafe {
            crate::jvm_knn::ffi_jvm_reader_knn(
                r,
                request.as_ptr(),
                request.len(),
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                5,
                &mut n,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        assert_eq!(n, 5);
        for (got, want) in scores.iter().zip(&exact) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }

        let mut tree = Blob::default()
            .u8(crate::jvm_reader::QUERY_TREE)
            .u8(crate::jvm_queries::NODE_KNN)
            .bytes(b"v")
            .u8(crate::jvm_queries::KNN_TARGET_FLOAT)
            .i32(4);
        for x in target {
            tree = tree.f32(x);
        }
        let tree = tree
            .u8(crate::jvm_queries::KNN_TOP_K)
            .i32(3)
            .i32(20)
            .u8(0)
            .0;
        let (mut docs, mut scores) = ([0i32; 20], [0f32; 20]);
        let (mut n, mut total, mut lower) = (0usize, 0i64, false);
        // SAFETY: live buffers of 20 and out-pointers.
        let rc = unsafe {
            crate::jvm_reader::ffi_jvm_reader_search(
                r,
                tree.as_ptr(),
                tree.len(),
                20,
                i64::MAX,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                20,
                &mut n,
                &mut total,
                &mut lower,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        assert_eq!((n, total), (6, 6), "three hits from each of two segments");
        assert!((scores[0] - exact[0]).abs() < 1e-6);
        crate::jvm_reader::ffi_close_jvm_reader(r);
    }
}
