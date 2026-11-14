//! What each `ffi_engine_writer_*` entry point does once its arguments are
//! plain values: opening a shard's writer over a directory, registering
//! fields, the merge policy and scheduler settings, commit points, merges and
//! the counters. `lucene-ffi` reads the caller's memory, looks the writer up
//! and calls these.

use std::sync::{Arc, Mutex, OnceLock};

use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_file_deleter::DeletionPolicy;
use lucene_index::index_writer::{
    self, IndexWriter, MergePolicyConfig, SegmentPartitioning, SoftDeletesRetention,
};
use lucene_index::merge_policy::{CompoundFileSettings, TieredMergePolicy};
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_store::directory::Directory;

use crate::status::{set_last_error, Status};
use crate::writer::*;
use crate::writer_blob::{
    derived_source_attributes, derived_source_list, restore_substring_formats,
    restore_vectors_formats, substring_format,
};
use crate::writer_error::writer_error;

/// The index sort as a `.si` encodes it (empty for none), checked against
/// the open flags: a search of the buffers reads documents in arrival
/// order, which a sorted flush renumbers.
pub fn parse_open_sort(
    index_sort: &[u8],
    flags: u8,
) -> Result<Option<Vec<lucene_index::segment_info::IndexSortField>>, Status> {
    let index_sort = if index_sort.is_empty() {
        None
    } else {
        lucene_index::segment_info::parse_index_sort(index_sort).map_err(|e| {
            set_last_error(format!("engine writer: index sort: {e}"));
            Status::InvalidArgument
        })?
    };
    if index_sort.is_some() && flags & OPEN_LIVE_SEARCH != 0 {
        set_last_error("engine writer: a search of the buffers does not support an index sort");
        return Err(Status::InvalidArgument);
    }
    Ok(index_sort)
}

/// What opening a writer takes besides its directory.
pub struct OpenParams<'a> {
    pub ram_buffer_mb: f64,
    pub flags: u8,
    pub index_sort: Option<Vec<lucene_index::segment_info::IndexSortField>>,
    pub parent_field: &'a str,
    pub max_docs: i32,
    pub stored_fields: u8,
    pub compression_level: i32,
}

/// Builds the writer for a shard over `dir`, and the scheduler whose threads
/// merge for it.
pub fn open_writer(
    dir: &'static dyn Directory,
    p: &OpenParams<'_>,
) -> Result<
    (
        Arc<ConcurrentIndexWriter<'static>>,
        ConcurrentMergeScheduler,
    ),
    Status,
> {
    let flags = p.flags;
    // Merges run on the scheduler's threads, as under OpenSearch's
    // `ConcurrentMergeScheduler`, with `index.merge.scheduler.*`'s
    // defaults until the engine passes the shard's settings
    // (`ffi_engine_writer_set_merge_scheduler`).
    let merges = ConcurrentMergeScheduler::new();
    merges.set_default_max_merges_and_threads(false);
    merges.enable_auto_io_throttle();
    let sparse = flags & OPEN_SPARSE != 0;
    let stored_codec = stored_fields_codec(p.stored_fields, p.compression_level)?;
    if sparse && stored_codec.is_some() {
        set_last_error("an index.sparse index takes the default stored fields only".to_string());
        return Err(Status::InvalidArgument);
    }
    let knn = flags & OPEN_KNN != 0;
    if sparse && knn {
        set_last_error("an index.sparse index is not a k-NN index".to_string());
        return Err(Status::InvalidArgument);
    }
    let open = || -> index_writer::Result<Arc<ConcurrentIndexWriter<'static>>> {
        let codec = if sparse {
            lucene_sparse::CODEC_NAME
        } else if knn {
            KNN_CODEC
        } else {
            "Lucene104"
        };
        let mut writer = IndexWriter::open(dir, Vec::new(), codec, VERSION)?;
        if sparse {
            writer.set_codec(Some(Arc::new(lucene_sparse::SparseCodec)))?;
        } else if knn {
            let delegate: Arc<dyn lucene_codecs::codec::Lucene104Codec> = match stored_codec.clone()
            {
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
        // The segments' own fields: the writer's list starts empty, and a
        // merge before any document registers a field again must still
        // write it in the format its mapping chose.
        let existing = writer.segment_fields()?;
        restore_vectors_formats(&mut writer, &existing);
        restore_substring_formats(&mut writer, &existing);
        if knn {
            derived_source_attributes(&mut writer);
        }
        if !p.parent_field.is_empty() {
            writer.set_parent_field(Some(p.parent_field))?;
        }
        if p.index_sort.is_some() {
            writer.set_index_sort(p.index_sort.as_deref())?;
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
        writer.set_ram_buffer_size_mb(p.ram_buffer_mb)?;
        // The JVM's `IndexWriter.getActualMaxDocs()`, so a test lowering
        // Java's limit lowers this one too.
        writer.set_max_docs(usize::try_from(p.max_docs).unwrap_or(0));
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
    Ok((writer, merges))
}

impl EngineWriter {
    /// A shard's writer over `handle`, with nothing held, applied or failed.
    pub fn new(handle: Box<dyn WriterParts>, flags: u8) -> EngineWriter {
        EngineWriter {
            handle,
            maintenance: Mutex::new(Holds {
                files: Default::default(),
                nrt: Default::default(),
                next: 1,
            }),
            tragic: OnceLock::new(),
            fault_injection: flags & OPEN_FAULT_INJECTION != 0,
            policies: Mutex::new(MergePolicies::default()),
            sparse: flags & OPEN_SPARSE != 0,
            live_chunks: Mutex::default(),
            live_warmer: crate::live::Warmer::default(),
            live_derived: Mutex::new((0, None)),
        }
    }
}

/// Registers a decoded field and writes its global number.
pub fn register_field(
    w: &EngineWriter,
    info: lucene_codecs::field_infos::FieldInfo,
    format: Option<lucene_codecs::per_field_knn_vectors::KnnVectorsFormat>,
) -> Result<i32, Status> {
    let name = info.name.clone();
    let derived = derived_source_list(&info);
    let substring = substring_format(&info);
    w.writer()
        .with_writer(|iw| {
            let number = iw.register_field(info)?;
            if let Some(format) = format {
                iw.set_knn_vectors_format_for_field(&name, format);
            }
            if let Some(format) = substring {
                iw.set_doc_values_format_for_field(&name, format);
            }
            // Only a k-NN index's codec reads the list back.
            if let Some(list) = derived {
                if iw.segment_attribute(KNN_DELEGATE_CODEC_ATTRIBUTE).is_some() {
                    iw.add_segment_list_value(list, &name);
                }
            }
            Ok::<_, index_writer::Error>(number)
        })
        .map_err(|e| {
            // A field the writer cannot index is the mapping's problem,
            // not the writer's: document-level.
            set_last_error(e.to_string());
            Status::InvalidArgument
        })
}

/// `index.merge.scheduler.*` as `OpenSearchConcurrentMergeScheduler.refreshConfig`
/// sets it.
pub fn set_merge_scheduler(
    w: &EngineWriter,
    max_merge_count: i32,
    max_thread_count: i32,
    auto_throttle: bool,
    max_force_merge_mb_per_sec: f64,
) -> Result<(), Status> {
    let merges = w.merges();
    if merges.max_merge_count() != max_merge_count || merges.max_thread_count() != max_thread_count
    {
        merges
            .set_max_merges_and_threads(max_merge_count, max_thread_count)
            .map_err(|e| {
                set_last_error(e.to_string());
                Status::InvalidArgument
            })?;
    }
    if auto_throttle != merges.auto_io_throttle() {
        if auto_throttle {
            merges.enable_auto_io_throttle();
        } else {
            merges.disable_auto_io_throttle();
        }
    }
    if max_force_merge_mb_per_sec != merges.force_merge_mb_per_sec() {
        merges.set_force_merge_mb_per_sec(max_force_merge_mb_per_sec);
    }
    Ok(())
}

/// The generations of every commit point the writer still holds, oldest first.
pub fn commit_generations(w: &EngineWriter) -> Vec<i64> {
    let _maintenance = w.maintenance();
    w.writer().with_writer(|iw| iw.commit_generations())
}

/// Drops the named commit points (never the newest).
pub fn delete_commits(w: &EngineWriter, gens: &[i64]) -> Result<(), Status> {
    let _maintenance = w.maintenance();
    w.writer()
        .with_writer(|iw| iw.delete_commits(gens))
        .map_err(writer_error)
}

/// Pins commit `generation`'s segment files; the hold's id.
pub fn hold_commit(w: &EngineWriter, generation: i64) -> Result<u64, Status> {
    let mut holds = w.maintenance();
    let files = w
        .writer()
        .with_writer(|iw| iw.hold_commit(generation))
        .map_err(|e| {
            // No such commit is the caller's mistake, not the writer's.
            set_last_error(e.to_string());
            Status::InvalidArgument
        })?;
    let id = holds.next;
    holds.next = holds.next.saturating_add(1);
    holds.files.insert(id, files);
    Ok(id)
}

/// A near-real-time view without a commit: `segments_N` bytes, the hold
/// pinning its files, and the last commit's generation.
pub fn nrt_snapshot(w: &EngineWriter) -> Result<(Vec<u8>, u64, i64), Status> {
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
}

/// Releases a hold ([`hold_commit`]'s or [`nrt_snapshot`]'s).
pub fn release_hold(w: &EngineWriter, hold: u64) -> Result<(), Status> {
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
        return Err(Status::InvalidArgument);
    };
    w.writer()
        .with_writer(|iw| iw.release_files(&files))
        .map_err(writer_error)
}

/// The soft-deletes retention merges apply.
pub fn set_retention(w: &EngineWriter, flags: u8, min_retained_seq_no: i64) {
    let retention = (flags & RETENTION_ENABLED != 0).then(|| SoftDeletesRetention {
        seq_no_field: SEQ_NO_FIELD.to_string(),
        min_retained_seq_no,
        prune_postings_field: Some(ID_FIELD.to_string()),
        prune_recovery_source_field: Some(RECOVERY_SOURCE_FIELD.to_string()),
        seq_no_sentinel: if flags & RETENTION_SEQ_NO_SENTINEL != 0 {
            vec![
                (SEQ_NO_FIELD.to_string(), UNASSIGNED_SEQ_NO),
                (PRIMARY_TERM_FIELD.to_string(), UNASSIGNED_PRIMARY_TERM),
            ]
        } else {
            Vec::new()
        },
    });
    w.writer()
        .with_writer(|iw| iw.set_soft_deletes_retention(retention));
}

/// `index.merge.policy.*` as OpenSearch's `TieredMergePolicyProvider` sets
/// them on its `TieredMergePolicy`, checked.
#[allow(clippy::too_many_arguments)]
pub fn set_merge_policy(
    w: &EngineWriter,
    max_merge_at_once: i32,
    segments_per_tier: f64,
    max_merged_segment_bytes: i64,
    floor_segment_bytes: i64,
    deletes_pct_allowed: f64,
    force_merge_deletes_pct_allowed: f64,
) -> Result<(), Status> {
    let invalid = |what: &str| {
        set_last_error(format!("invalid merge policy setting: {what}"));
        Status::InvalidArgument
    };
    let max_merge_at_once = usize::try_from(max_merge_at_once)
        .ok()
        .filter(|&n| n >= 2)
        .ok_or_else(|| invalid("max_merge_at_once < 2"))?;
    if segments_per_tier.is_nan() || segments_per_tier < 2.0 {
        return Err(invalid("segments_per_tier < 2"));
    }
    let max_merged_segment_size =
        u64::try_from(max_merged_segment_bytes).map_err(|_| invalid("max_merged_segment < 0"))?;
    let floor_segment_size =
        u64::try_from(floor_segment_bytes).map_err(|_| invalid("floor_segment < 0"))?;
    if !(5.0..=50.0).contains(&deletes_pct_allowed) {
        return Err(invalid("deletes_pct_allowed outside [5, 50]"));
    }
    if !(0.0..=100.0).contains(&force_merge_deletes_pct_allowed) {
        return Err(invalid("expunge_deletes_allowed outside [0, 100]"));
    }
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
}

/// The shard's merge policy when it is not `TieredMergePolicy`, checked;
/// the policy to install (`None` for the tiered one).
#[allow(clippy::too_many_arguments)]
pub fn pluggable_policy(
    kind: i32,
    merge_factor: i32,
    min_merge_bytes: i64,
    max_merge_bytes: i64,
    max_forced_merge_bytes: i64,
    max_merge_docs: i32,
    no_cfs_ratio: f64,
    max_cfs_segment_bytes: i64,
) -> Result<Option<PluggablePolicy>, Status> {
    let invalid = |what: String| {
        set_last_error(format!("invalid merge policy setting: {what}"));
        Status::InvalidArgument
    };
    Ok(match kind {
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
    })
}

/// Installs the shard's non-tiered merge policy ([`pluggable_policy`]).
pub fn set_pluggable_policy(w: &EngineWriter, policy: Option<PluggablePolicy>) {
    let mut policies = w.policies();
    if policies.other == policy {
        return;
    }
    policies.other = policy;
    w.writer()
        .with_writer(|iw| install_merge_policies(iw, &policies));
}

/// The compound-file and flush options OpenSearch sets beside the merge policy.
pub fn set_options(
    w: &EngineWriter,
    use_compound_file: bool,
    no_cfs_ratio: f64,
    max_cfs_segment_bytes: i64,
    check_pending_flush: bool,
    shuffle_forced_merges: bool,
) -> Result<(), Status> {
    let mut compound = CompoundFileSettings::default();
    compound.set_no_cfs_ratio(no_cfs_ratio).map_err(|e| {
        set_last_error(e.to_string());
        Status::InvalidArgument
    })?;
    if max_cfs_segment_bytes < 0 {
        set_last_error(format!(
            "maxCFSSegmentSize must be >= 0; got {max_cfs_segment_bytes}"
        ));
        return Err(Status::InvalidArgument);
    }
    let policies = w.policies();
    w.writer().with_writer(|iw| {
        iw.set_use_compound_file(use_compound_file);
        compound.max_cfs_segment_size = max_cfs_segment_bytes;
        iw.set_merge_compound_file_settings(Some(compound));
        iw.set_shuffle_forced_merges(shuffle_forced_merges);
        // The wrapping policies decide merged segments' compound files too.
        install_merge_policies(iw, &policies);
    });
    drop(policies);
    w.writer()
        .set_check_pending_flush_on_update(check_pending_flush);
    Ok(())
}

/// Merge-on-commit/refresh from `index.merge_on_flush.*`.
pub fn set_merge_on_flush(w: &EngineWriter, wait_millis: i64, merge_on_flush_policy: bool) {
    let mut policies = w.policies();
    policies.merge_on_flush = merge_on_flush_policy;
    w.writer().with_writer(|iw| {
        iw.set_max_full_flush_merge_wait_millis(wait_millis);
        install_merge_policies(iw, &policies);
    });
}

/// `index.knn.ivf.partition_min_vectors`.
pub fn set_ivf_partition_clusters(w: &EngineWriter, min_vectors: i64) -> Result<(), Status> {
    let min = match usize::try_from(min_vectors) {
        Ok(0) => None,
        Ok(n) => Some(n),
        Err(_) => {
            set_last_error(format!(
                "ivf_partition_clusters must be >= 0; got {min_vectors}"
            ));
            return Err(Status::InvalidArgument);
        }
    };
    w.writer()
        .with_writer(|iw| iw.set_ivf_partition_clusters(min));
    Ok(())
}

/// What `ffi_engine_writer_set_bp_reorder` passes besides the writer.
pub struct BpParams<'a> {
    pub min_doc_freq: i32,
    pub max_doc_freq: f32,
    pub min_partition_size: i32,
    pub max_iters: i32,
    pub ram_budget_mb: f64,
    pub min_merge_docs: i32,
    pub min_merge_ratio: f32,
    /// NUL-separated UTF-8 field names, every indexed field when empty.
    pub fields: &'a [u8],
}

/// The reorderer's settings, checked.
pub fn bp_settings(p: &BpParams<'_>) -> Result<BpSettings, Status> {
    let invalid = |what: String| {
        set_last_error(format!("invalid bp_reorder setting: {what}"));
        Status::InvalidArgument
    };
    let mut reorderer = BpIndexReorderer::new();
    reorderer
        .set_min_doc_freq(p.min_doc_freq)
        .map_err(|e| invalid(e.to_string()))?;
    reorderer
        .set_max_doc_freq(p.max_doc_freq)
        .map_err(|e| invalid(e.to_string()))?;
    reorderer
        .set_min_partition_size(p.min_partition_size)
        .map_err(|e| invalid(e.to_string()))?;
    reorderer
        .set_max_iters(p.max_iters)
        .map_err(|e| invalid(e.to_string()))?;
    if !(p.ram_budget_mb.is_finite() && p.ram_budget_mb > 0.0) {
        return Err(invalid(format!(
            "ram_budget_mb must be positive, got {}",
            p.ram_budget_mb
        )));
    }
    reorderer.set_ram_budget_mb(p.ram_budget_mb);
    if p.fields.is_empty() {
        reorderer.set_fields(None);
    } else {
        let names = std::str::from_utf8(p.fields)
            .map_err(|_| invalid("fields are not UTF-8".to_string()))?
            .split('\0')
            .filter(|f| !f.is_empty())
            .map(str::to_string)
            .collect();
        reorderer.set_fields(Some(names));
    }
    // OpenSearch sets no parent field on an unsorted index with nested
    // documents, so its blocks are found by their root documents: only
    // those have a `_primary_term` (`Queries.newNonNestedFilter`).
    reorderer.set_block_root_field(Some("_primary_term".to_string()));
    // The policy's own checks, before anything changes.
    let mut probe =
        BpReorderingMergePolicy::new(Box::new(TieredMergePolicy::default()), reorderer.clone());
    probe
        .set_min_natural_merge_num_docs(p.min_merge_docs)
        .map_err(|e| invalid(e.to_string()))?;
    probe
        .set_min_natural_merge_ratio_from_biggest_segment(p.min_merge_ratio)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(BpSettings {
        reorderer,
        min_merge_docs: p.min_merge_docs,
        min_merge_ratio: p.min_merge_ratio,
    })
}

/// Turns BP reordering on (`Some`) or off around the shard's merge policy.
pub fn set_bp_reorder(w: &EngineWriter, bp: Option<BpSettings>) -> Result<(), Status> {
    if bp.is_some() && w.sparse {
        set_last_error("bp_reorder is not supported on a sparse index");
        return Err(Status::InvalidArgument);
    }
    let mut policies = w.policies();
    policies.bp = bp;
    w.writer()
        .with_writer(|iw| install_merge_policies(iw, &policies));
    Ok(())
}

/// `forceMerge(max)`, or `forceMergeDeletes()`, then a commit; the newest
/// commit's generation.
pub fn force_merge(w: &EngineWriter, max: usize, only_deletes: bool) -> Result<i64, Status> {
    let _maintenance = w.maintenance();
    w.writer()
        .with_writer_merges_aborted(|iw| {
            if only_deletes {
                iw.force_merge_deletes()
            } else {
                iw.force_merge(max)
            }
        })
        .map_err(writer_error)?;
    // The merged segments become durable with a commit.
    w.writer().commit().map_err(writer_error)?;
    Ok(w.generation())
}

/// OpenSearch's segment upgrade: the newest commit's generation and how
/// many segments were rewritten.
pub fn upgrade(
    w: &EngineWriter,
    current: lucene_index::segment_info::LuceneVersion,
    only_ancient: bool,
) -> Result<(i64, usize), Status> {
    let _maintenance = w.maintenance();
    let upgraded = w
        .writer()
        .with_writer_merges_aborted(|iw| iw.upgrade_segments(current, only_ancient))
        .map_err(writer_error)?;
    if upgraded > 0 {
        w.writer().commit().map_err(writer_error)?;
    }
    Ok((w.generation(), upgraded))
}

/// The first `k` of the writer's `STAT_*` counters (the rest are zero).
pub fn stats(w: &EngineWriter, k: usize) -> [i64; STAT_COUNT] {
    let writer = w.writer();
    let mut s = [0i64; STAT_COUNT];
    s[STAT_RAM_BYTES] = i64::try_from(writer.ram_bytes_used()).unwrap_or(i64::MAX);
    if k > STAT_FLUSHING_BYTES {
        s[STAT_FLUSHING_BYTES] = i64::try_from(writer.flushing_bytes()).unwrap_or(i64::MAX);
    }
    if k > STAT_PENDING_DOCS {
        s[STAT_PENDING_DOCS] = i64::try_from(writer.pending_doc_count()).unwrap_or(i64::MAX);
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
    s
}

/// Closes the writer without committing (Lucene's `rollback()`).
pub fn close(shared: &EngineWriter) {
    // Another thread may still be inside a call; the last `Arc` drops the
    // writer. Holds die with it, as a JVM's readers' pins die with the JVM.
    let _maintenance = shared.maintenance();
    // The shared live view holds the writer.
    shared.live_warmer().stop_sharing();
    shared.writer().stop_merges();
    shared.writer().rollback();
}
