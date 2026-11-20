//! The writer the OpenSearch plugin's `AcceleratedEngine` indexes through, over
//! [`lucene_engine::writer`]: the directory a shard's writer opens over,
//! the JVM upcalls, and the `extern "C"` entry points, which read the
//! caller's memory and call [`lucene_engine::writer_ops`].

use std::collections::HashMap;
use std::os::raw::c_char;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    self, MergeEvent, MergeListener, MergeTimes, MergedSegmentWarmer,
};
use lucene_index::merge_scheduler::ConcurrentMergeScheduler;
use lucene_store::directory::Directory;

use lucene_engine::writer_blob::{decode_field, decode_op, decode_user_data, Op};
use lucene_engine::writer_error::writer_error;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::{engine_writers, lock_recovering, WriterHandle};

pub(crate) use lucene_engine::writer::*;
pub(crate) use lucene_engine::writer_ops::*;

/// The owner of a shard's writer and the directory it borrows, erased to
/// `'static`: it implements [`WriterParts`] for the engine crate, which holds
/// no `unsafe`.
pub(crate) struct OwnedWriter(pub crate::registry::WriterHandle);

impl WriterParts for OwnedWriter {
    fn writer(&self) -> &Arc<ConcurrentIndexWriter<'static>> {
        &self.0.writer
    }
    fn merges(&self) -> &ConcurrentMergeScheduler {
        &self.0.merges
    }
}

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
        // SAFETY: caller contract.
        let parent_field = unsafe { str_from_raw(parent_field, parent_field_len)? };
        let index_sort = parse_open_sort(index_sort, flags)?;
        // Mapped, not read onto the heap: a merge opens every source file of
        // every merged segment, and Lucene's own `MMapDirectory` -- what
        // OpenSearch writes with -- maps them too. Segment files are written
        // once and only deleted (never truncated) once unreferenced, which is
        // the condition mapping needs.
        // A `encrypted` shard's path is bound to an encrypting directory
        // over the mapping (`store.rs`).
        let dir: Box<dyn Directory> = crate::store::shard_directory(path).map_err(|e| {
            set_last_error(format!("opening the shard directory [{path}]: {e}"));
            FfiStatus::Io
        })?;
        let dir_ref: &dyn Directory = &*dir;
        // SAFETY: `WriterHandle`'s own argument -- the boxed directory
        // outlives the writer that borrows it (see `writer.rs`).
        let dir_ref: &'static dyn Directory = unsafe { std::mem::transmute(dir_ref) };
        let (writer, merges) = open_writer(
            dir_ref,
            &OpenParams {
                ram_buffer_mb,
                flags,
                index_sort,
                parent_field,
                max_docs,
                stored_fields,
                compression_level,
            },
        )?;
        lucene_engine::writer::set_yield_hook(yield_to_searches);
        let engine = EngineWriter::new(
            Box::new(OwnedWriter(WriterHandle {
                writer,
                merges,
                dir,
            })),
            flags,
        );
        let handle = lock_recovering(engine_writers()).insert_checked(Arc::new(engine))?;
        if let Some(&upcall) = MERGE_UPCALL.get() {
            // OpenSearch's engine hears of every merge, and throttles indexing
            // itself when merges fall behind instead of the scheduler stalling.
            with_writer(handle, |w| {
                w.merges().set_stall_producers(false);
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
        let number = with_writer(handle, |w| register_field(w, info, format))?;
        // SAFETY: caller contract.
        unsafe { *out_number = number };
        Ok(())
    })
}

/// How much lower than a search's an indexing thread's scheduling priority
/// is, as a nice increment, unless `ACCELERATED_INDEX_NICE` says otherwise
/// (0 leaves it).
const INDEX_NICE: i32 = 5;

/// Lowers the calling indexing thread's scheduling priority, once per
/// thread: on a machine indexing keeps busy, a search then runs as soon as
/// it is ready instead of waiting behind the indexing threads for a core,
/// and indexing gives up only the time searches take.
pub(crate) fn yield_to_searches() {
    #[cfg(target_os = "linux")]
    {
        thread_local!(static LOWERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
        static NICE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        let nice = *NICE.get_or_init(|| {
            std::env::var("ACCELERATED_INDEX_NICE")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(INDEX_NICE)
        });
        if nice <= 0 || LOWERED.with(|d| d.replace(true)) {
            return;
        }
        // A thread inherits the priority of the thread starting it: the
        // shared worker pool, which merges and searches use too, starts
        // now if no thread started it yet, at the priority it had before.
        let _ = rayon::current_num_threads();
        // SAFETY: syscalls about the calling thread alone, which Linux
        // schedules by its thread id; failing leaves the priority as it is.
        unsafe {
            let Ok(tid) = libc::id_t::try_from(libc::syscall(libc::SYS_gettid)) else {
                return;
            };
            // -1 is a priority too: only errno tells a failure.
            *libc::__errno_location() = 0;
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            if current == -1 && *libc::__errno_location() != 0 {
                return;
            }
            libc::setpriority(
                libc::PRIO_PROCESS,
                tid,
                current.saturating_add(nice).min(19),
            );
        }
    }
}

/// Applies one operation blob ([`decode_op`]).
///
/// # Safety
/// `op` valid for `op_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_apply(handle: u64, op: *const u8, op_len: usize) -> i32 {
    guard(|| {
        yield_to_searches();
        crate::purge::writing();
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
        yield_to_searches();
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
        yield_to_searches();
        crate::purge::writing();
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

thread_local! {
    /// The document the calling thread last read with `ffi_jvm_ingest_prepare`, as
    /// `(token, plan, documents)`: the engine applies it on the thread that prepared it, right
    /// after, so a thread holds at most one and an unused one is dropped by the next.
    pub(crate) static INGESTED: std::cell::RefCell<Option<(u64, u32, lucene_engine::ingest::Prepared)>> =
        const { std::cell::RefCell::new(None) };
}
pub(crate) static NEXT_INGESTED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Applies the document [`INGESTED`] holds under `token` (see `ffi_jvm_ingest_prepare`) with
/// the values the engine assigned it: `kind` 0 adds, 1 replaces the live documents of `id`
/// (their `soft_field` becomes `soft_value`) as `softUpdateDocuments` does. A thread holding
/// no such document is [`FfiStatus::InvalidHandle`]: the caller prepares it again.
///
/// # Safety
/// `soft_field` valid for `soft_len` bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ffi_engine_writer_apply_ingested(
    handle: u64,
    token: u64,
    kind: u8,
    soft_field: *const u8,
    soft_len: usize,
    soft_value: i64,
    seq_no: i64,
    primary_term: i64,
    version: i64,
) -> i32 {
    guard(|| {
        yield_to_searches();
        // SAFETY: caller contract.
        let soft_field = unsafe { bytes_from_raw(soft_field, soft_len)? };
        let held = INGESTED.with(|slot| {
            let mut slot = slot.borrow_mut();
            match slot.take() {
                Some(held) if held.0 == token => Some(held),
                other => {
                    *slot = other;
                    None
                }
            }
        });
        let Some((_, plan, prepared)) = held else {
            return Err(FfiStatus::InvalidHandle);
        };
        let plan = lucene_engine::ingest::plan(plan).ok_or(FfiStatus::InvalidHandle)?;
        let id = prepared.id().to_vec();
        let docs = prepared.finish(&plan, seq_no, primary_term, version);
        crate::purge::writing();
        let op = if kind == 0 {
            Op::Add {
                partition: None,
                docs,
                vectors: None,
            }
        } else {
            let soft_field = std::str::from_utf8(soft_field).map_err(|_| FfiStatus::InvalidUtf8)?;
            Op::SoftUpdate {
                partition: None,
                term: lucene_index::buffered_updates::Term::new("_id", id),
                soft_field: soft_field.to_owned(),
                soft_value,
                docs,
                vectors: None,
            }
        };
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
static COMMIT_DATA_UPCALL: OnceLock<CommitDataUpcall> = OnceLock::new();

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
                w.merges().io_rate_limit_mb_per_sec(),
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
        crate::purge::writing();
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
            set_merge_scheduler(
                w,
                max_merge_count,
                max_thread_count,
                auto_throttle != 0,
                max_force_merge_mb_per_sec,
            )
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
        let gens = with_writer(handle, |w| Ok(commit_generations(w)))?;
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
        with_writer(handle, |w| delete_commits(w, &gens))
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
        let id = with_writer(handle, |w| hold_commit(w, generation))?;
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
        let (bytes, id, generation) = with_writer(handle, nrt_snapshot)?;
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
    guard(|| with_writer(handle, |w| release_hold(w, hold)))
}

/// Sets the soft-deletes retention merges apply. `flags` without
/// [`RETENTION_ENABLED`] keeps every soft-deleted document; with it, those
/// whose `_seq_no` is below `min_retained_seq_no` are dropped, and with
/// [`RETENTION_SEQ_NO_SENTINEL`] as well every document below it gets the
/// sentinel `_seq_no` and `_primary_term`.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_retention(
    handle: u64,
    flags: u8,
    min_retained_seq_no: i64,
) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            set_retention(w, flags, min_retained_seq_no);
            Ok(())
        })
    })
}

/// `index.codec.adaptive_doc_values: auto`: the writer writes stock doc
/// values until it has seen its first documents, then stock or
/// `Adaptive90` for every field that would get the default format, one
/// choice made from whether any numeric column is clustered in document
/// order and kept in the commit's user data
/// ([`lucene_index::adaptive_auto`]). A writer opened on a commit that holds
/// the choice keeps it. Call it in place of [`OPEN_ADAPTIVE_DOC_VALUES`],
/// before the first document.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_adaptive_auto(handle: u64) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            w.writer()
                .with_writer(|iw| iw.set_adaptive_doc_values_auto());
            Ok(())
        })
    })
}

/// `auto`'s choice ([`ffi_engine_writer_set_adaptive_auto`]): 1 adaptive, 0
/// stock, -1 not made yet (or `auto` not set).
#[no_mangle]
pub extern "C" fn ffi_engine_writer_adaptive_choice(handle: u64, out: *mut i32) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        with_writer(handle, |w| {
            let choice = w.writer().with_writer(|iw| iw.adaptive_doc_values_choice());
            // SAFETY: `out` is non-null and writable (caller contract).
            unsafe { *out = choice.map_or(-1, i32::from) };
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
        with_writer(handle, |w| {
            set_merge_policy(
                w,
                max_merge_at_once,
                segments_per_tier,
                max_merged_segment_bytes,
                floor_segment_bytes,
                deletes_pct_allowed,
                force_merge_deletes_pct_allowed,
            )
        })
    })
}

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
        let policy = pluggable_policy(
            kind,
            merge_factor,
            min_merge_bytes,
            max_merge_bytes,
            max_forced_merge_bytes,
            max_merge_docs,
            no_cfs_ratio,
            max_cfs_segment_bytes,
        )?;
        with_writer(handle, |w| {
            set_pluggable_policy(w, policy);
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
        with_writer(handle, |w| {
            set_options(
                w,
                use_compound_file != 0,
                no_cfs_ratio,
                max_cfs_segment_bytes,
                check_pending_flush != 0,
                shuffle_forced_merges != 0,
            )
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
            set_merge_on_flush(w, wait_millis, merge_on_flush_policy != 0);
            Ok(())
        })
    })
}

/// IVF fields' vectors clustered per partition of the index sort's first
/// field at flush and merge (`index.knn.ivf.partition_min_vectors`):
/// each partition of at least `min_vectors` vectors on its own, the smaller
/// ones pooled, so a search filtered to one partition probes only its
/// clusters. `0` (the default) clusters each field as a whole; no effect
/// without an index sort, or on a flush holding nested document blocks.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_ivf_partition_clusters(
    handle: u64,
    min_vectors: i64,
) -> i32 {
    guard(|| with_writer(handle, |w| set_ivf_partition_clusters(w, min_vectors)))
}

/// The Lucene format generation the shard writes: `105` or `106`.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_index_format(handle: u64, format: i32) -> i32 {
    guard(|| with_writer(handle, |w| set_index_format(w, format)))
}

/// `index.codec.stored_fields.recompress_on_merge: non_force` (with the open
/// flag for the setting on): forced merges bulk-copy stored-field chunks
/// instead of recompressing them. `on == 0` clears it.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_set_merge_lz4_hc_non_force(handle: u64, on: u8) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            set_merge_lz4_hc_non_force(w, on != 0);
            Ok(())
        })
    })
}

/// `BPReorderingMergePolicy` around the shard's merge policy
/// (`index.merge.reorder.*`), or off when `enabled == 0`: merges of
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
        // SAFETY: caller contract.
        let fields = unsafe { bytes_from_raw(fields, fields_len)? };
        let bp = if enabled == 0 {
            None
        } else {
            Some(bp_settings(&BpParams {
                min_doc_freq,
                max_doc_freq,
                min_partition_size,
                max_iters,
                ram_budget_mb,
                min_merge_docs,
                min_merge_ratio,
                fields,
            })?)
        };
        with_writer(handle, |w| set_bp_reorder(w, bp))
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
        crate::purge::writing();
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
        let generation = with_writer(handle, |w| force_merge(w, max, only_deletes != 0))?;
        // SAFETY: caller contract.
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// OpenSearch's segment upgrade (`IndexShard.upgrade`): rewrites, each on
/// its own, every committed segment written by a Lucene older than
/// `major.minor` (OpenSearch's `Version.CURRENT.luceneVersion`) -- by an older
/// major only when `only_ancient != 0` -- as `OpenSearchMergePolicy` does
/// while an upgrade is in progress, and commits when any was. Writes the
/// newest commit's generation and how many segments were rewritten.
///
/// # Safety
/// `out_generation` and `out_upgraded` valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_upgrade(
    handle: u64,
    major: i32,
    minor: i32,
    only_ancient: u8,
    out_generation: *mut i64,
    out_upgraded: *mut i32,
) -> i32 {
    guard(|| {
        if out_generation.is_null() || out_upgraded.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let current = lucene_index::segment_info::LuceneVersion {
            major,
            minor,
            bugfix: 0,
        };
        let (generation, upgraded) =
            with_writer(handle, |w| upgrade(w, current, only_ancient != 0))?;
        // SAFETY: caller contract.
        unsafe {
            *out_generation = generation;
            *out_upgraded = i32::try_from(upgraded).unwrap_or(i32::MAX);
        }
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
        let stats = with_writer(handle, |w| Ok(stats(w, k)))?;
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
        close(&shared);
        Ok(())
    })
}

#[cfg(test)]
mod tests;
