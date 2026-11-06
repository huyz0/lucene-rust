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
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_file_deleter::DeletionPolicy;
use lucene_index::index_writer::{
    self, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerms,
    MergePolicyConfig, SoftDeletesRetention,
};
use lucene_index::merge_policy::{MergeOnFlushMergePolicy, MergePolicy, TieredMergePolicy};
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_index::segment_info::LuceneVersion;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_store::directory::Directory;

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
}

/// The policies [`install_merge_policies`] builds around the writer's
/// [`MergePolicyConfig`].
#[derive(Debug, Clone, Default)]
struct MergePolicies {
    /// `index.merge_on_flush.policy: merge-on-flush`.
    merge_on_flush: bool,
    /// `BPReorderingMergePolicy`'s settings, when reordering is on.
    bp: Option<BpSettings>,
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
/// `BPReorderingMergePolicy` deciding every other merge. With BP on, a
/// commit or refresh still picks its merges with the plain tiered policy:
/// the segments it merges are small, and its merges then run on the merge
/// threads instead of the refreshing one. A writer with merges off keeps
/// them off.
fn install_merge_policies(iw: &mut IndexWriter<'_>, policies: &MergePolicies) {
    let merges_on = iw.merge_policy().is_some();
    let config = iw.merge_policy().cloned().unwrap_or_default();
    let compound = iw.use_compound_file();
    let tiered = || {
        let mut t = TieredMergePolicy::new(config.clone());
        if !compound {
            // Merged segments stay loose files, as without a policy.
            let _ = t.compound_file_settings_mut().set_no_cfs_ratio(0.0);
        }
        t
    };
    let full_flush: Option<Arc<dyn MergePolicy>> = if policies.merge_on_flush {
        Some(Arc::new(MergeOnFlushMergePolicy::new(Box::new(tiered()))))
    } else if merges_on && policies.bp.is_some() {
        Some(Arc::new(tiered()))
    } else {
        None
    };
    iw.set_full_flush_merge_policy(full_flush);
    let pluggable = policies.bp.as_ref().filter(|_| merges_on).map(|bp| {
        let mut policy = BpReorderingMergePolicy::new(Box::new(tiered()), bp.reorderer.clone());
        // Both checked when the settings came in.
        let _ = policy.set_min_natural_merge_num_docs(bp.min_merge_docs);
        let _ = policy.set_min_natural_merge_ratio_from_biggest_segment(bp.min_merge_ratio);
        Arc::new(policy) as Arc<dyn MergePolicy>
    });
    iw.set_pluggable_merge_policy(pluggable);
}

impl EngineWriter {
    fn policies(&self) -> MutexGuard<'_, MergePolicies> {
        self.policies.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Files pinned by [`ffi_engine_writer_hold_commit`] and
/// [`ffi_engine_writer_nrt_snapshot`], by hold id (one id space).
#[derive(Default)]
struct Holds {
    files: HashMap<u64, Vec<String>>,
    /// Near-real-time snapshots' pins: dropping one hands its files back to
    /// the writer, which releases them at its next flush or commit.
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

/// An operation blob, decoded.
enum Op {
    Add(Vec<ExplicitDocument>),
    SoftUpdate {
        term: Term,
        soft_field: String,
        soft_value: i64,
        docs: Vec<ExplicitDocument>,
    },
    Panic,
}

/// `u8 kind`, then for a soft update `term field (i32 len, utf8), term bytes
/// (i32 len, bytes), soft-deletes field (i32 len, utf8), i64 value`, then
/// `i32 count` documents ([`document`]). The whole blob must be consumed.
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
        OP_ADD | OP_SOFT_UPDATE => {
            let update = if kind == OP_SOFT_UPDATE {
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
            match update {
                None => Op::Add(docs),
                Some((term, soft_field, soft_value)) => Op::SoftUpdate {
                    term,
                    soft_field,
                    soft_value,
                    docs,
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
/// u8 soft_deletes_field, i32 attribute count, then (i32 len, utf8) key and
/// value pairs` -- Lucene's `FieldInfo`. The attributes are the field type's
/// own (`FieldType.putAttribute`); the writer adds its per-field format ones
/// per segment.
fn decode_field(blob: &[u8]) -> Result<FieldInfo, FfiStatus> {
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
    if vector_dimension != 0 {
        info = info.with_vectors(
            vector_dimension,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        );
    }
    Ok(info)
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
/// [`ffi_engine_writer_open`]'s flag for `index.lucene_rust.adaptive_doc_values`:
/// new segments write their doc values in the `RustAdaptive` format
/// ([`lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_adaptive`]).
pub const OPEN_ADAPTIVE_DOC_VALUES: u8 = 32;

/// Opens the shard's writer over the index at `path`, which must already hold
/// a commit (OpenSearch's `Store.createEmpty` or a recovered one).
/// `ram_buffer_mb` is the flush threshold. `flags`: [`OPEN_FAULT_INJECTION`]
/// allows the [`OP_PANIC`] operation; [`OPEN_SPARSE`] writes the segments of
/// an `index.sparse` index ([`lucene_sparse::SparseCodec`]);
/// [`OPEN_LIVE_SEARCH`] keeps what a search of the buffers reads;
/// [`OPEN_ADAPTIVE_DOC_VALUES`] writes doc values in the `RustAdaptive`
/// format.
///
/// # Safety
/// `path` valid for `path_len` bytes; `out_handle` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_open(
    path: *const c_char,
    path_len: usize,
    ram_buffer_mb: f64,
    flags: u8,
    max_docs: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
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
        let open = || -> index_writer::Result<Arc<ConcurrentIndexWriter<'static>>> {
            let sparse = flags & OPEN_SPARSE != 0;
            let codec = if sparse {
                lucene_sparse::CODEC_NAME
            } else {
                "Lucene104"
            };
            let mut writer = IndexWriter::open(dir_ref, Vec::new(), codec, VERSION)?;
            if sparse {
                writer.set_codec(Some(Arc::new(lucene_sparse::SparseCodec)))?;
            }
            if flags & OPEN_ADAPTIVE_DOC_VALUES != 0 {
                writer.set_default_doc_values_format(
                    lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_adaptive(),
                );
            }
            writer.enable_explicit_documents()?;
            writer.set_deletion_policy(DeletionPolicy::KeepAll)?;
            writer.set_merge_policy(Some(MergePolicyConfig::default()));
            // Merge-on-commit/refresh stays off until the engine passes
            // the shard's `index.merge_on_flush.*` settings
            // (`ffi_engine_writer_set_merge_on_flush`).
            writer.set_max_full_flush_merge_wait_millis(0);
            // Loose segment files, as this engine has always written them.
            // OpenSearch's default is compound (`index.compound_format`);
            // the setting is not wired through the engine yet.
            writer.set_use_compound_file(false);
            writer.set_ram_buffer_size_mb(ram_buffer_mb)?;
            // The JVM's `IndexWriter.getActualMaxDocs()`, so a test lowering
            // Java's limit lowers this one too.
            writer.set_max_docs(usize::try_from(max_docs).unwrap_or(0));
            // One buffer per thread that can index at once.
            let slots = std::thread::available_parallelism().map_or(1, |n| n.get());
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
        };
        let handle = lock_recovering(engine_writers()).insert_checked(Arc::new(engine))?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
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
        let info = decode_field(spec)?;
        let number = with_writer(handle, |w| {
            w.writer().register_field(info).map_err(|e| {
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
            Op::Add(docs) => writer.add_explicit_documents(docs).map_err(writer_error)?,
            Op::SoftUpdate {
                term,
                soft_field,
                soft_value,
                docs,
            } => {
                let soft = DocValuesUpdate::Numeric {
                    term: term.clone(),
                    field: soft_field,
                    value: Some(soft_value),
                };
                writer
                    .soft_update_explicit_documents(term, docs, &[soft])
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
            if holds.nrt.remove(&hold).is_some() {
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
                .with_writer(|iw| {
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
        let Ok(Op::Add(docs)) = op else {
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
            .u8(0);
        let attributed = vector.clone().i32(1).bytes(b"k").bytes(b"v");
        assert_eq!(
            decode_field(&attributed.0).unwrap().attributes,
            vec![("k".to_string(), "v".to_string())]
        );
        assert!(decode_field(&vector.clone().i32(1).bytes(b"k").0).is_err());
        let vector = vector.i32(0);
        assert_eq!(decode_field(&vector.0).unwrap().vector_dimension, 4);
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
        let rc = unsafe { ffi_live_view_reader(reader, view, &mut whole) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        let mut again = 0u64;
        // SAFETY: a live out-pointer.
        assert_eq!(unsafe { ffi_live_view_reader(reader, view, &mut again) }, 0);
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
        let rc = unsafe { ffi_live_view_reader(reader, view, &mut again) };
        assert_ne!(rc, 0, "a closed view makes no reader");

        // A view whose buffers were published since reads their segments,
        // and finds what it found before they were. (Enough documents that
        // the buffer is one chunk again: see `a_growing_buffer_is_read_as_chunks`.)
        (41..47).for_each(add);
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
        let rc = unsafe { ffi_live_view_reader(reader, view, &mut whole) };
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
            for _ in 0..1 + round % 5 {
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
            let mut whole = 0u64;
            // SAFETY: a live out-pointer.
            let rc = unsafe { ffi_live_view_reader(reader, view, &mut whole) };
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
            }
            assert_eq!(ffi_live_view_close(view), 0);
        }
        // A few chunks per buffer, not one per view.
        assert!(leaves_seen.iter().all(|&n| n <= 8), "{leaves_seen:?}");
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
        let rc = unsafe { ffi_live_view_reader(reader, written, &mut whole) };
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
        let rc = unsafe { ffi_live_view_reader(reader, unwritten, &mut whole) };
        assert_ne!(rc, 0);
        assert!(crate::error::last_error().contains("being flushed"));
        assert_eq!(ffi_live_view_close(written), 0);
        assert_eq!(ffi_live_view_close(unwritten), 0);
        assert_eq!(crate::jvm_reader::ffi_close_jvm_reader(reader), 0);
    }

    /// The 1M OpenSearch bench's documents (`bench_nodes.py gen`), mapped as
    /// OpenSearch maps them, buffered by four threads, then one refresh
    /// timed. `FLUSH_DOCS` sets how many (default 500 000).
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
        let h = open_with(&tmp, 4096.0, 0, i32::MAX);
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
        let threads = 4;
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            for t in 0..threads {
                let (words, cum, tag_cum) = (&words, &cum, &tag_cum);
                s.spawn(move || {
                    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (t as u64 + 1));
                    for i in (t..docs).step_by(threads) {
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
}
