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
//! as one byte blob per operation (`decode_op` documents the layout). The
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
//! is `FfiStatus::InvalidArgument` and changes nothing -- Java's
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
//! (`DeletionPolicy::KeepAll`): OpenSearch's `CombinedDeletionPolicy` runs
//! in the JVM and names the ones to drop (`ffi_engine_writer_delete_commits`).

use std::collections::HashMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use lucene_index::buffered_updates::DocValuesUpdate;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogByteSizeMergePolicy;
use lucene_index::merge_policy::{
    CompoundFileSettings, MergeOnFlushMergePolicy, MergePolicy, NoMergePolicy, TieredMergePolicy,
};
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_index::segment_info::LuceneVersion;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};

use crate::writer_blob::Op;
use crate::writer_error::writer_error;

use crate::registry::{engine_writers, read_recovering};
use crate::status::{set_last_error, Status};

/// What owns a shard's writer and the directory it borrows. `lucene-ffi`
/// implements it for the owner that erases the directory's borrow, so this
/// crate keeps no `unsafe`.
pub trait WriterParts: Send + Sync {
    /// The shard's writer.
    fn writer(&self) -> &std::sync::Arc<ConcurrentIndexWriter<'static>>;
    /// The scheduler whose threads merge for the writer.
    fn merges(&self) -> &ConcurrentMergeScheduler;
}

static YIELD_HOOK: OnceLock<fn()> = OnceLock::new();

/// Installs the function indexing threads call to give searches the CPU
/// ([`yield_hook`]); the boundary crate owns the syscall behind it.
pub fn set_yield_hook(hook: fn()) {
    let _ = YIELD_HOOK.set(hook);
}

/// Lowers the calling indexing thread's priority once the boundary crate
/// has installed the hook; a no-op before.
pub fn yield_hook() {
    if let Some(hook) = YIELD_HOOK.get() {
        hook();
    }
}

/// The Lucene version every segment this writer produces claims.
pub const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 2,
};

/// The doc-values field OpenSearch keeps each document's sequence number in
/// (`SeqNoFieldMapper.NAME`), which retention is decided by.
pub const SEQ_NO_FIELD: &str = "_seq_no";
/// OpenSearch's `SeqNoFieldMapper.PRIMARY_TERM_NAME`.
pub const PRIMARY_TERM_FIELD: &str = "_primary_term";
/// The field OpenSearch keeps each document's id in (`IdFieldMapper.NAME`),
/// whose postings a retention merge prunes (`PrunePostingsMergePolicy`).
pub const ID_FIELD: &str = "_id";
/// The stored copy of a filtered source OpenSearch keeps for peer recovery
/// (`SourceFieldMapper.RECOVERY_SOURCE_NAME`), which a retention merge prunes
/// (`RecoverySourcePruneMergePolicy`).
pub const RECOVERY_SOURCE_FIELD: &str = "_recovery_source";

/// Indices into `ffi_engine_writer_stats`' output.
pub const STAT_RAM_BYTES: usize = 0;
pub const STAT_FLUSHING_BYTES: usize = 1;
pub const STAT_PENDING_DOCS: usize = 2;
pub const STAT_UNCOMMITTED: usize = 3;
pub const STAT_GENERATION: usize = 4;
pub const STAT_SEGMENTS: usize = 5;
pub const STAT_COUNT: usize = 6;

/// One shard's writer and the state the JVM side drives it with.
pub struct EngineWriter {
    pub handle: Box<dyn WriterParts>,
    /// Held for the whole of a commit, merge or commit-point change, so they
    /// run one at a time; adds do not take it.
    pub maintenance: Mutex<Holds>,
    /// `IndexWriter.getTragicException()`: once set, every call fails.
    pub tragic: OnceLock<String>,
    pub fault_injection: bool,
    /// Which merge policies wrap the shard's `TieredMergePolicy`.
    pub policies: Mutex<MergePolicies>,
    /// Whether the shard's codec is the sparse one, which BP reordering
    /// does not support.
    pub sparse: bool,
    /// The buffers' segments live views read ([`crate::live`]).
    pub live_chunks: Mutex<crate::live::Chunks>,
    /// Writes those segments ahead of the searches ([`crate::live`]).
    pub live_warmer: crate::live::Warmer,
    /// The k-NN derived-source fields of buffered documents, as of how many
    /// fields were registered ([`EngineWriter::live_derived`]).
    pub live_derived: Mutex<(usize, Option<Arc<crate::live::LiveDerived>>)>,
}

/// The policies [`install_merge_policies`] builds around the writer's
/// `MergePolicyConfig`.
#[derive(Debug, Clone, Default)]
pub struct MergePolicies {
    /// `index.merge_on_flush.policy: merge-on-flush`.
    pub merge_on_flush: bool,
    /// `BPReorderingMergePolicy`'s settings, when reordering is on.
    pub bp: Option<BpSettings>,
    /// The shard's policy when it is not the tiered one
    /// (`ffi_engine_writer_set_pluggable_merge_policy`).
    pub other: Option<PluggablePolicy>,
}

/// `BPReorderingMergePolicy` with its `BPIndexReorderer`.
#[derive(Debug, Clone)]
pub struct BpSettings {
    pub reorderer: BpIndexReorderer,
    pub min_merge_docs: i32,
    pub min_merge_ratio: f32,
}

/// Rebuilds the policies `policies` names over the writer's current
/// `MergePolicyConfig`: merge-on-flush's `MergeOnFlushMergePolicy`, and
/// `BPReorderingMergePolicy` deciding every other merge, each around the
/// shard's own policy (tiered, or [`PluggablePolicy`]). With BP on, a
/// commit or refresh still picks its merges with the plain tiered policy:
/// the segments it merges are small, and its merges then run on the merge
/// threads instead of the refreshing one. A writer with merges off keeps
/// them off.
pub fn install_merge_policies(iw: &mut IndexWriter<'_>, policies: &MergePolicies) {
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
    pub fn policies(&self) -> MutexGuard<'_, MergePolicies> {
        self.policies.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A merge policy other than `TieredMergePolicy`, as OpenSearch configures it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PluggablePolicy {
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
    pub fn build(&self) -> Box<dyn MergePolicy> {
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

/// Files pinned by `ffi_engine_writer_hold_commit` and
/// `ffi_engine_writer_nrt_snapshot`, by hold id (one id space).
#[derive(Default)]
pub struct Holds {
    pub files: HashMap<u64, Vec<String>>,
    /// Near-real-time snapshots' pins: releasing the last reference to one
    /// releases its files at once.
    pub nrt: HashMap<u64, Arc<lucene_index::nrt::NrtFileHold>>,
    pub next: u64,
}

impl EngineWriter {
    /// The scheduler whose threads merge for the writer.
    pub fn merges(&self) -> &ConcurrentMergeScheduler {
        self.handle.merges()
    }

    pub fn writer(&self) -> &ConcurrentIndexWriter<'static> {
        self.handle.writer()
    }

    /// The fields whose `_source` values buffered documents hold masked
    /// ([`crate::writer_blob::DERIVED_SOURCE_ATTRIBUTE`]), `None` when there are none; resolved
    /// again only once more fields are registered.
    pub fn live_derived(&self) -> Option<Arc<crate::live::LiveDerived>> {
        let count = self.writer().field_count();
        let mut cached = self
            .live_derived
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cached.0 != count {
            let infos = self.writer().field_infos();
            *cached = (
                infos.len(),
                crate::live::LiveDerived::of(infos).map(Arc::new),
            );
        }
        cached.1.clone()
    }

    pub fn live_chunks(&self) -> MutexGuard<'_, crate::live::Chunks> {
        self.live_chunks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn live_warmer(&self) -> &crate::live::Warmer {
        &self.live_warmer
    }

    pub fn maintenance(&self) -> MutexGuard<'_, Holds> {
        self.maintenance
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The generation of the newest commit written or attempted.
    pub fn generation(&self) -> i64 {
        self.writer().with_writer(|w| w.segment_infos().generation)
    }
}

pub fn lookup(handle: u64) -> Result<Arc<EngineWriter>, Status> {
    read_recovering(engine_writers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error("engine writer: unknown or already-closed handle");
            Status::InvalidHandle
        })
}

/// Runs `body` on the handle's writer. A recorded tragic error fails the
/// call; a failure of `body` records one unless it is the document's own
/// ([`Status::InvalidArgument`], [`Status::Decode`]), and so does a
/// panic, which then carries on to the caller's guard.
pub fn with_writer<T>(
    handle: u64,
    body: impl FnOnce(&EngineWriter) -> Result<T, Status>,
) -> Result<T, Status> {
    let w = lookup(handle)?;
    if let Some(t) = w.tragic.get() {
        set_last_error(format!("engine writer closed by a tragic event: {t}"));
        return Err(Status::Io);
    }
    match catch_unwind(AssertUnwindSafe(|| body(&w))) {
        Ok(Err(status)) if status != Status::InvalidArgument && status != Status::Decode => {
            let _ = w.tragic.set(crate::status::last_error());
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

/// `ffi_engine_writer_open`'s flag allowing [`crate::writer_blob::OP_PANIC`].
pub const OPEN_FAULT_INJECTION: u8 = 1;
/// `ffi_engine_writer_open`'s flag for an `index.sparse` index.
pub const OPEN_SPARSE: u8 = 2;
/// `ffi_engine_writer_open`'s flag for an index searched without a refresh
/// (`crate::live`).
pub const OPEN_LIVE_SEARCH: u8 = 4;
/// `ffi_engine_writer_open`'s flag for
/// `index.codec.stored_fields.recompress_on_merge`: merges write
/// `BEST_SPEED` stored fields with the high-compression LZ4 match finder,
/// which stock Lucene reads
/// ([`IndexWriter::set_merge_lz4_high_compression`]).
pub const OPEN_MERGE_LZ4_HC: u8 = 8;
/// `ffi_engine_writer_open`'s flag for an `index.knn` index: segments record OpenSearch
/// k-NN's codec name ([`KNN_CODEC`]) over the stored-fields codec, whose name they record under
/// [`KNN_DELEGATE_CODEC_ATTRIBUTE`], as stock k-NN writes them; so k-NN's derived source reads
/// back through that codec on any node.
pub const OPEN_KNN: u8 = 128;
/// The codec name a k-NN index's segments record (`KNN1040Codec`).
pub const KNN_CODEC: &str = "KNN1040Codec";
/// The `.si` attribute naming the codec whose stored fields a k-NN segment holds
/// (`KNN10010DerivedSourceStoredFieldsFormat.KNN_DELEGATE_CODEC_NAME`).
pub const KNN_DELEGATE_CODEC_ATTRIBUTE: &str = "knn_delegate_stored_fields_codec_key";
/// `ffi_engine_writer_open`'s flag for `index.codec.terms_compression: zstd`: new
/// segments write every field's terms in the `ZstdTerms104` postings format,
/// whose term blocks are zstd-compressed
/// ([`lucene_codecs::per_field_postings::Lucene104PostingsFormat::rust_zstd`]).
pub const OPEN_ZSTD_TERMS: u8 = 16;
/// The zstd level of [`OPEN_ZSTD_TERMS`]' term blocks. Blocks decode at the
/// same speed whatever the level; higher ones only write slower.
pub const ZSTD_TERMS_LEVEL: i32 = 3;
/// `ffi_engine_writer_open`'s flag for `index.codec.adaptive_doc_values`:
/// new segments write their doc values in the `Adaptive90` format
/// ([`lucene_codecs::doc_values::Lucene90DocValuesFormat::rust_adaptive`]).
pub const OPEN_ADAPTIVE_DOC_VALUES: u8 = 32;
/// `ffi_engine_writer_open`'s flag for an index with a partition field
/// (`index.segment_partition.field`): every segment holds one partition
/// and records it, and merges never mix partitions
/// ([`lucene_index::index_writer::SegmentPartitioning::opensearch`]).
pub const OPEN_PARTITIONED: u8 = 64;
/// Indexing buffers in a partitioned writer: a buffer holds one partition,
/// so each partition in flight keeps its own until the RAM buffer flushes the
/// largest or a refresh flushes them all. Past this many partitions in flight
/// a buffer of another partition is flushed to make room.
pub const PARTITIONED_SLOTS: usize = 1024;

/// Opens the shard's writer over the index at `path`, which must already hold
/// a commit (OpenSearch's `Store.createEmpty` or a recovered one).
/// `ram_buffer_mb` is the flush threshold. `flags`: [`OPEN_FAULT_INJECTION`]
/// allows the [`crate::writer_blob::OP_PANIC`] operation; [`OPEN_SPARSE`] writes the segments of
/// an `index.sparse` index ([`lucene_sparse::SparseCodec`]);
/// [`OPEN_LIVE_SEARCH`] keeps what a search of the buffers reads;
/// [`OPEN_ADAPTIVE_DOC_VALUES`] writes doc values in the `Adaptive90`
/// format; [`OPEN_ZSTD_TERMS`] writes terms dictionaries with zstd blocks;
/// [`OPEN_PARTITIONED`] keeps partitions in segments of their own;
/// `ffi_engine_writer_open`'s stored fields: `Lucene104` with
/// `BEST_SPEED` (`index.codec: default`).
pub const STORED_FIELDS_BEST_SPEED: u8 = 0;
/// `Lucene104` with `BEST_COMPRESSION` (`index.codec: best_compression`).
pub const STORED_FIELDS_BEST_COMPRESSION: u8 = 1;
/// OpenSearch custom-codecs' `ZSTD104` (`index.codec: zstd`).
pub const STORED_FIELDS_ZSTD: u8 = 2;
/// OpenSearch custom-codecs' `ZSTDNODICT104` (`index.codec: zstd_no_dict`).
pub const STORED_FIELDS_ZSTD_NO_DICT: u8 = 3;
/// The Rust engine's paged zstd stored fields, `PagedZstd104`
/// (`index.codec.stored_fields.page_aligned`): 16 KiB chunks packed into
/// 16 KiB pages, with a dictionary trained at large merges.
pub const STORED_FIELDS_ZSTD_PAGED: u8 = 4;

/// The codec `ffi_engine_writer_open_codec`'s `stored_fields` names, or
/// `None` for the default one.
pub fn stored_fields_codec(
    stored_fields: u8,
    level: i32,
) -> Result<Option<Arc<dyn lucene_codecs::codec::Lucene104Codec>>, Status> {
    use lucene_codecs::codec::{
        BestCompressionLucene104Codec, ZstdLucene104Codec, ZstdPagedLucene104Codec,
    };
    use lucene_codecs::stored_fields::ZstdMode;
    let zstd =
        |mode: ZstdMode| -> Result<Option<Arc<dyn lucene_codecs::codec::Lucene104Codec>>, Status> {
            let codec = ZstdLucene104Codec::new(mode).map_err(|e| {
                set_last_error(format!("index.codec.compression_level: {e}"));
                Status::InvalidArgument
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
                Status::InvalidArgument
            })?;
            Ok(Some(Arc::new(codec)))
        }
        other => {
            set_last_error(format!("unknown stored fields format {other}"));
            Err(Status::InvalidArgument)
        }
    }
}

/// Applies one decoded operation on the calling thread.
pub fn apply_op(handle: u64, op: Op) -> Result<(), Status> {
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
                    return Err(Status::InvalidArgument);
                }
                panic!("injected fault: engine writer panic");
            }
        };
        crate::live::after_add(w);
        Ok(())
    })
}

/// The commit-data upcall succeeded.
pub const COMMIT_DATA_OK: i32 = 0;
/// The commit-data upcall's buffer was too small; `*out_len` is the size needed.
pub const COMMIT_DATA_BUFFER_TOO_SMALL: i32 = 1;
/// The JVM could not produce the commit data; it keeps its own error.
pub const COMMIT_DATA_FAILED: i32 = 2;

/// Registers the commit-data upcall (`CommitDataUpcall`), once per process;
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

/// The merged segment was warmed.
pub const WARM_OK: i32 = 0;

/// `ffi_engine_writer_set_retention` flag: retention is on.
pub const RETENTION_ENABLED: u8 = 1;
/// `ffi_engine_writer_set_retention` flag: merges write
/// [`UNASSIGNED_SEQ_NO`] and [`UNASSIGNED_PRIMARY_TERM`] for every document
/// whose `_seq_no` is below the retained minimum.
pub const RETENTION_SEQ_NO_SENTINEL: u8 = 2;
/// OpenSearch's `SequenceNumbers.UNASSIGNED_SEQ_NO`.
pub const UNASSIGNED_SEQ_NO: i64 = -2;
/// OpenSearch's `SequenceNumbers.UNASSIGNED_PRIMARY_TERM`.
pub const UNASSIGNED_PRIMARY_TERM: i64 = 0;

/// No pluggable policy: `ffi_engine_writer_set_merge_policy`'s tiered one decides.
pub const MERGE_POLICY_TIERED: i32 = 0;
/// `LogByteSizeMergePolicy`.
pub const MERGE_POLICY_LOG_BYTE_SIZE: i32 = 1;
/// `NoMergePolicy`.
pub const MERGE_POLICY_NONE: i32 = 2;
