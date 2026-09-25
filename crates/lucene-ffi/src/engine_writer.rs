//! The writer the OpenSearch plugin's `RustEngine` indexes through
//! (`opensearch-plugin/`, milestone M5): one handle per shard, wrapping a
//! [`IndexWriter`] in *explicit-documents* mode.
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
//! ## One lock per shard
//!
//! Handles live in their own registry as `Arc<Mutex<EngineWriter>>`: a call
//! takes the registry's shared guard only long enough to clone the `Arc`, then
//! serializes on the one shard's mutex. Two shards never wait for each other.
//!
//! ## Failures
//!
//! A document the writer refuses (a field it cannot index, a malformed blob)
//! is [`FfiStatus::InvalidArgument`] and changes nothing -- Java's
//! document-level failure. Any other failure while writing leaves the writer
//! in a state nothing vouches for, so it is recorded as the handle's *tragic*
//! error (`IndexWriter.getTragicException()`): every later call fails with it,
//! and the engine fails the shard. A panic poisons the handle's mutex, which
//! is read the same way. Neither ever takes down another shard.
//!
//! ## Commits
//!
//! Every commit publishes the caller's user data and then runs the merge
//! policy, each merge publishing its own commit with the same user data --
//! so every `segments_N` this writer leaves is a consistent recovery point.
//! Commit points are never dropped by the writer itself
//! ([`DeletionPolicy::KeepAll`]): OpenSearch's `CombinedDeletionPolicy` runs
//! in the JVM and names the ones to drop ([`ffi_engine_writer_delete_commits`]).

use std::collections::HashMap;
use std::os::raw::c_char;
use std::sync::{Arc, Mutex, PoisonError};

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::index_file_deleter::DeletionPolicy;
use lucene_index::index_writer::{
    self, ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
    MergePolicyConfig, SoftDeletesRetention,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::directory::{Directory, FsDirectory};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::{bytes_from_raw, str_from_raw, try_with_capacity};
use crate::registry::{engine_writers, lock_recovering, read_recovering, WriterHandle};

/// The Lucene version every segment this writer produces claims.
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
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
/// Panics inside the writer's lock -- the fault the "a panic fails exactly one
/// shard" test injects. Refused unless the handle was opened with fault
/// injection enabled.
const OP_PANIC: u8 = 0xFF;

/// Indices into [`ffi_engine_writer_stats`]' output.
pub const STAT_RAM_BYTES: usize = 0;
pub const STAT_PENDING_DOCS: usize = 1;
pub const STAT_UNCOMMITTED: usize = 2;
pub const STAT_GENERATION: usize = 3;
pub const STAT_SEGMENTS: usize = 4;
pub const STAT_COUNT: usize = 5;

/// One shard's writer and the state the JVM side drives it with.
pub struct EngineWriter {
    handle: WriterHandle,
    /// Files pinned by [`ffi_engine_writer_hold_commit`], by hold id.
    holds: HashMap<u64, Vec<String>>,
    next_hold: u64,
    /// `IndexWriter.getTragicException()`: once set, every call fails.
    tragic: Option<String>,
    fault_injection: bool,
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
        _ => return Err(decode_error("norm flag")),
    };
    // A term is at least its length, its freq and its flags.
    let n = c.count(9, "term count")?;
    let mut terms = try_with_capacity(n)?;
    for _ in 0..n {
        let term = c.bytes("term")?.to_vec();
        let freq = c.i32("freq")?;
        let flags = c.u8("term flags")?;
        let occurrences = usize::try_from(freq).map_err(|_| decode_error("freq"))?;
        let positions = if flags & 1 != 0 {
            if occurrences.saturating_mul(4) > c.remaining() {
                return Err(decode_error("positions"));
            }
            let mut p = try_with_capacity(occurrences)?;
            for _ in 0..occurrences {
                p.push(c.i32("position")?);
            }
            p
        } else {
            Vec::new()
        };
        let offsets = if flags & 2 != 0 {
            if occurrences.saturating_mul(8) > c.remaining() {
                return Err(decode_error("offsets"));
            }
            let mut o = try_with_capacity(occurrences)?;
            for _ in 0..occurrences {
                o.push((c.i32("start offset")?, c.i32("end offset")?));
            }
            o
        } else {
            Vec::new()
        };
        terms.push(InvertedTerm {
            term,
            freq,
            positions,
            offsets,
        });
    }
    Ok(InvertedField {
        field_number,
        terms,
        norm,
    })
}

/// One document: its stored values, inverted fields, doc values and points,
/// each a count followed by that many entries.
///
/// - stored: `i32 field, u8 kind (0 string, 1 binary, 2 int, 3 long,
///   4 float, 5 double), i32 len, bytes`
/// - inverted: `i32 field, u8 has_norm, [i64 norm], i32 terms`, then per term
///   `i32 len, bytes, i32 freq, u8 flags (1 positions, 2 offsets),
///   [freq x i32 position], [freq x (i32 start, i32 end)]`
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
/// u8 soft_deletes_field` -- Lucene's `FieldInfo` minus the attributes, which
/// the writer sets per segment.
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
    c.finish("trailing bytes after the field")?;
    let mut info = FieldInfo::new(name, 0)
        .with_index_options(index_options)
        .with_omit_norms(omit_norms)
        .with_store_term_vectors(store_term_vectors)
        .with_store_payloads(store_payloads)
        .with_doc_values(doc_values_type, skip, -1)
        .with_points(dims, index_dims, num_bytes)
        .with_soft_deletes_field(soft);
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
    let n = c.count(8, "user data count")?;
    let mut out = try_with_capacity(n)?;
    for _ in 0..n {
        let key = c.string("user data key")?;
        let value = c.string("user data value")?;
        out.push((key, value));
    }
    c.finish("trailing bytes after the user data")?;
    Ok(out)
}

/// The general writer's mapping: what Java raises as
/// `IllegalArgumentException` -- a refused document, an add past `maxDocs` --
/// is [`FfiStatus::InvalidArgument`] and fails only the operation; the rest
/// is I/O and tragic.
fn writer_error(e: index_writer::Error) -> FfiStatus {
    crate::writer::map_writer_error("engine writer", e)
}

fn lookup(handle: u64) -> Result<Arc<Mutex<EngineWriter>>, FfiStatus> {
    read_recovering(engine_writers())
        .get(handle)
        .cloned()
        .ok_or_else(|| {
            set_last_error("engine writer: unknown or already-closed handle");
            FfiStatus::InvalidHandle
        })
}

/// Runs `body` on the handle's writer. A poisoned mutex (an earlier call
/// panicked mid-write) and a recorded tragic error both fail the call; so
/// does `body`, which then records a tragic error unless `document_level`
/// says the failure is the document's own ([`FfiStatus::InvalidArgument`]).
fn with_writer<T>(
    handle: u64,
    body: impl FnOnce(&mut EngineWriter) -> Result<T, FfiStatus>,
) -> Result<T, FfiStatus> {
    let shared = lookup(handle)?;
    let mut w = match shared.lock() {
        Ok(w) => w,
        Err(poisoned) => {
            let mut w = poisoned.into_inner();
            if w.tragic.is_none() {
                w.tragic = Some("a previous call panicked inside the writer".to_string());
            }
            w
        }
    };
    if let Some(t) = &w.tragic {
        set_last_error(format!("engine writer closed by a tragic event: {t}"));
        return Err(FfiStatus::Io);
    }
    let out = body(&mut w);
    if let Err(status) = out {
        if status != FfiStatus::InvalidArgument && status != FfiStatus::Decode {
            w.tragic = Some(crate::error::last_error());
        }
    }
    out
}

/// Opens the shard's writer over the index at `path`, which must already hold
/// a commit (OpenSearch's `Store.createEmpty` or a recovered one).
/// `ram_buffer_mb` is the flush threshold; `fault_injection != 0` allows the
/// [`OP_PANIC`] operation.
///
/// # Safety
/// `path` valid for `path_len` bytes; `out_handle` valid for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_open(
    path: *const c_char,
    path_len: usize,
    ram_buffer_mb: f64,
    fault_injection: u8,
    max_docs: i32,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        let dir = Box::new(FsDirectory::open(path));
        let dir_ref: &dyn Directory = &*dir;
        // SAFETY: `WriterHandle`'s own argument -- the boxed directory
        // outlives the writer that borrows it (see `writer.rs`).
        let dir_ref: &'static dyn Directory = unsafe { std::mem::transmute(dir_ref) };
        let open = || -> index_writer::Result<IndexWriter<'static>> {
            let mut writer = IndexWriter::open(dir_ref, Vec::new(), "Lucene104", VERSION)?;
            writer.enable_explicit_documents()?;
            writer.set_deletion_policy(DeletionPolicy::KeepAll)?;
            writer.set_merge_policy(Some(MergePolicyConfig::default()));
            writer.set_ram_buffer_size_mb(ram_buffer_mb)?;
            // The JVM's `IndexWriter.getActualMaxDocs()`, so a test lowering
            // Java's limit lowers this one too.
            writer.set_max_docs(usize::try_from(max_docs).unwrap_or(0));
            Ok(writer)
        };
        let writer = open().map_err(writer_error)?;
        let engine = EngineWriter {
            handle: WriterHandle { writer, dir },
            holds: HashMap::new(),
            next_hold: 1,
            tragic: None,
            fault_injection: fault_injection != 0,
        };
        let handle =
            lock_recovering(engine_writers()).insert_checked(Arc::new(Mutex::new(engine)))?;
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
            w.handle.writer.register_field(info).map_err(|e| {
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

/// Applies one operation blob ([`decode_op`]).
///
/// # Safety
/// `op` valid for `op_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_apply(handle: u64, op: *const u8, op_len: usize) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let op = decode_op(unsafe { bytes_from_raw(op, op_len)? })?;
        with_writer(handle, |w| {
            let writer = &mut w.handle.writer;
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
    })
}

/// Commits with `user_data` ([`decode_user_data`]) as the commit's user data,
/// runs the merge policy, and writes the newest commit's generation.
///
/// # Safety
/// `user_data` valid for `user_data_len` bytes; `out_generation` valid for
/// one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_commit(
    handle: u64,
    user_data: *const u8,
    user_data_len: usize,
    out_generation: *mut i64,
) -> i32 {
    guard(|| {
        if out_generation.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let data = decode_user_data(unsafe { bytes_from_raw(user_data, user_data_len)? })?;
        let generation = with_writer(handle, |w| {
            let writer = &mut w.handle.writer;
            writer.set_live_commit_data(data);
            writer.commit().map_err(writer_error)?;
            Ok(writer.segment_infos().generation)
        })?;
        // SAFETY: caller contract.
        unsafe { *out_generation = generation };
        Ok(())
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
        let gens = with_writer(handle, |w| Ok(w.handle.writer.commit_generations()))?;
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
            w.handle.writer.delete_commits(&gens).map_err(writer_error)
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
            let files = w.handle.writer.hold_commit(generation).map_err(|e| {
                // No such commit is the caller's mistake, not the writer's.
                set_last_error(e.to_string());
                FfiStatus::InvalidArgument
            })?;
            let id = w.next_hold;
            w.next_hold = w.next_hold.saturating_add(1);
            w.holds.insert(id, files);
            Ok(id)
        })?;
        // SAFETY: caller contract.
        unsafe { *out_hold = id };
        Ok(())
    })
}

/// Releases a [`ffi_engine_writer_hold_commit`] pin.
#[no_mangle]
pub extern "C" fn ffi_engine_writer_release_hold(handle: u64, hold: u64) -> i32 {
    guard(|| {
        with_writer(handle, |w| {
            let Some(files) = w.holds.remove(&hold) else {
                set_last_error(format!("engine writer: no hold {hold}"));
                return Err(FfiStatus::InvalidArgument);
            };
            w.handle.writer.release_files(&files).map_err(writer_error)
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
            w.handle
                .writer
                .set_soft_deletes_retention((enabled != 0).then(|| SoftDeletesRetention {
                    seq_no_field: SEQ_NO_FIELD.to_string(),
                    min_retained_seq_no,
                    prune_postings_field: Some(ID_FIELD.to_string()),
                    prune_recovery_source_field: Some(RECOVERY_SOURCE_FIELD.to_string()),
                }));
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
            let writer = &mut w.handle.writer;
            if only_deletes != 0 {
                writer.force_merge_deletes().map_err(writer_error)?;
            } else {
                writer.force_merge(max).map_err(writer_error)?;
            }
            Ok(writer.segment_infos().generation)
        })?;
        // SAFETY: caller contract.
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// Writes `min(n, STAT_COUNT)` counters into `out`: RAM bytes buffered,
/// buffered documents, whether anything is uncommitted (0/1), the newest
/// commit's generation, and its segment count (`STAT_*`).
///
/// # Safety
/// `out` valid for `n` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_engine_writer_stats(handle: u64, out: *mut i64, n: usize) -> i32 {
    guard(|| {
        if out.is_null() && n != 0 {
            return Err(FfiStatus::NullPointer);
        }
        let stats = with_writer(handle, |w| {
            let writer = &w.handle.writer;
            let mut s = [0i64; STAT_COUNT];
            s[STAT_RAM_BYTES] = i64::try_from(writer.ram_bytes_used()).unwrap_or(i64::MAX);
            s[STAT_PENDING_DOCS] = i64::try_from(writer.pending_doc_count()).unwrap_or(i64::MAX);
            s[STAT_UNCOMMITTED] = i64::from(writer.has_uncommitted_changes());
            s[STAT_GENERATION] = writer.segment_infos().generation;
            s[STAT_SEGMENTS] =
                i64::try_from(writer.segment_infos().segments.len()).unwrap_or(i64::MAX);
            Ok(s)
        })?;
        let k = n.min(STAT_COUNT);
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
        let mut w = shared.lock().unwrap_or_else(PoisonError::into_inner);
        w.handle.writer.rollback();
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use lucene_util::test_support::TempDir;

    /// A little-endian blob builder mirroring the JVM side's.
    #[derive(Default)]
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

    fn commit(h: u64, data: &[(&str, &str)]) -> i64 {
        let mut b = Blob::default().i32(data.len() as i32);
        for (k, v) in data {
            b = b.bytes(k.as_bytes()).bytes(v.as_bytes());
        }
        let mut gen = 0;
        // SAFETY: live slice and out-pointer.
        let rc = unsafe { ffi_engine_writer_commit(h, b.0.as_ptr(), b.0.len(), &mut gen) };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        gen
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
        let path = tmp.path().to_str().unwrap();
        let mut h = 0;
        // SAFETY: live string and out-pointer.
        let rc = unsafe {
            ffi_engine_writer_open(
                path.as_ptr() as *const c_char,
                path.len(),
                16.0,
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
    fn a_refused_document_is_not_tragic_but_a_panic_is() {
        let tmp = empty_index("engine-writer-failures");
        let h = open(&tmp, 1);
        let f = setup(h);
        // Term vectors are refused at registration, as the mapping's problem.
        let mut tv = field("tv", 4, 0, 0, 0);
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
                ffi_engine_writer_commit(h, [0u8; 4].as_ptr(), 4, std::ptr::null_mut()),
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
        let data = Blob::default().i32(1).bytes(b"k").0;
        let mut g = 0;
        // SAFETY: live slice and out-pointer.
        assert_eq!(
            unsafe { ffi_engine_writer_commit(h, data.as_ptr(), data.len(), &mut g) },
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
                .u8(2)
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
        assert_eq!(decode_field(&vector.0).unwrap().vector_dimension, 4);
        assert!(decode_user_data(&Blob::default().i32(0).u8(0).0).is_err());
    }
}
