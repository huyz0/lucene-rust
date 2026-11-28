//! Read-only SQL over a shard's reader: the entry points the plugin's
//! `_plugins/_accelerated/sql` endpoint calls.
//!
//! A statement runs against the reader handle the plugin already holds for the
//! shard's searcher (a view of it for a user under field- and document-level
//! security), so what the statement sees is what a search of that user sees.
//! The table is described by a blob the plugin builds from the index's mapping
//! ([`decode_table`]). The engine built from a handle and a table is kept
//! until the handle closes, so a repeated statement reuses its parsed shapes
//! and cached plans.

use std::collections::HashMap;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use lucene_sql::{Path, Snapshot, SqlConfig, SqlEngine, TableSpec};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::raw::{bytes_from_raw, str_from_raw};
use crate::registry::JvmReaderHandle;
use lucene_analysis::TokenStream;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The layout version of the table blob.
const TABLE_VERSION: u32 = 1;

/// Column kinds in the table blob.
const KEY: u8 = 0;
const LONG: u8 = 1;
const KEYWORD: u8 = 2;
const TEXT: u8 = 3;
const VECTOR: u8 = 4;

struct Cursor<'a> {
    bytes: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FfiStatus> {
        if n > self.bytes.len() {
            set_last_error("the table blob ends early");
            return Err(FfiStatus::Decode);
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, FfiStatus> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, FfiStatus> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn blob(&mut self) -> Result<&'a [u8], FfiStatus> {
        let n = self.u32()? as usize;
        self.take(n)
    }

    fn string(&mut self) -> Result<&'a str, FfiStatus> {
        std::str::from_utf8(self.blob()?).map_err(|_| {
            set_last_error("the table blob holds a name that is not UTF-8");
            FfiStatus::InvalidUtf8
        })
    }
}

/// The table a blob describes.
///
/// | field | layout |
/// |---|---|
/// | version | `u32`, [`TABLE_VERSION`] |
/// | name | `u32` length, UTF-8 |
/// | columns | `u32` count, then per column: kind `u8` (key 0, long 1, keyword 2, text 3, vector 4), name (`u32` length, UTF-8), argument `u32` (a vector's dimensions), analyzer (`u32` length, an analysis spec; empty except for text) |
fn decode_table(blob: &[u8]) -> Result<TableSpec, FfiStatus> {
    let mut c = Cursor { bytes: blob };
    let version = c.u32()?;
    if version != TABLE_VERSION {
        set_last_error(format!(
            "table blob version {version}, expected {TABLE_VERSION}"
        ));
        return Err(FfiStatus::InvalidArgument);
    }
    let mut spec = TableSpec::new(c.string()?);
    for _ in 0..c.u32()? {
        let kind = c.u8()?;
        let name = c.string()?;
        let arg = c.u32()?;
        let analyzer = c.blob()?;
        spec = match kind {
            KEY => spec.key(name),
            LONG => spec.long(name),
            KEYWORD => spec.keyword(name),
            TEXT => spec.text_with(name, analyzer_of(name, analyzer)?),
            VECTOR => spec.vector(name, arg as usize),
            other => {
                set_last_error(format!("column `{name}` has kind {other}"));
                return Err(FfiStatus::InvalidArgument);
            }
        };
    }
    if !c.bytes.is_empty() {
        set_last_error("the table blob has trailing bytes");
        return Err(FfiStatus::Decode);
    }
    Ok(spec)
}

/// `spec`'s analyzer as the SQL layer takes one: the terms it makes of a text.
fn analyzer_of(field: &str, spec: &[u8]) -> Result<lucene_sql::spec::Analyzer, FfiStatus> {
    let built = lucene_engine::analysis::analyzer(spec)?;
    let field = field.to_string();
    Ok(lucene_sql::spec::Analyzer::new(move |text| {
        let mut terms = Vec::new();
        let Ok(mut stream) = built.analyzer().token_stream(&field, text) else {
            return terms;
        };
        if stream.reset().is_err() {
            return terms;
        }
        while stream.increment_token().unwrap_or(false) {
            terms.push(String::from_utf8_lossy(stream.attributes().term_bytes()).into_owned());
        }
        let _ = stream.end();
        let _ = stream.close();
        terms
    }))
}

/// What the engines are built with, set by [`ffi_jvm_sql_configure`] before
/// the first statement.
static CONFIG: Mutex<Option<SqlConfig>> = Mutex::new(None);

fn config() -> SqlConfig {
    lock(&CONFIG)
        .get_or_insert_with(SqlConfig::default)
        .clone()
}

/// One runtime for every engine: an engine per reader would start a thread
/// pool per refresh.
fn runtime() -> Result<Arc<lucene_sql::Runtime>, FfiStatus> {
    static RUNTIME: OnceLock<Arc<lucene_sql::Runtime>> = OnceLock::new();
    if let Some(r) = RUNTIME.get() {
        return Ok(Arc::clone(r));
    }
    let built = Arc::new(SqlEngine::runtime_for(&config()).map_err(|e| {
        set_last_error(format!("starting the SQL runtime: {e}"));
        FfiStatus::Io
    })?);
    Ok(Arc::clone(RUNTIME.get_or_init(|| built)))
}

struct Cached {
    table: Vec<u8>,
    engine: OnceLock<Result<Arc<SqlEngine>, String>>,
}

fn engines() -> &'static Mutex<HashMap<u64, Arc<Cached>>> {
    static ENGINES: OnceLock<Mutex<HashMap<u64, Arc<Cached>>>> = OnceLock::new();
    ENGINES.get_or_init(Mutex::default)
}

/// Statements answered, by path.
static ROUTED: AtomicU64 = AtomicU64::new(0);
static STANDARD: AtomicU64 = AtomicU64::new(0);

/// Drops the engine kept for reader `handle`; called when the handle closes.
pub(crate) fn forget(handle: u64) {
    lock(engines()).remove(&handle);
}

fn snapshot_of(h: &JvmReaderHandle, table: &[u8]) -> Result<Arc<Snapshot>, FfiStatus> {
    let spec = decode_table(table)?;
    let reader = h.reader.with_live_docs(&h.live_docs).map_err(|e| {
        set_last_error(format!("applying the live docs: {e}"));
        FfiStatus::Decode
    })?;
    Snapshot::new(reader, spec).map(Arc::new).map_err(|e| {
        set_last_error(format!("opening the table: {e}"));
        FfiStatus::InvalidArgument
    })
}

fn engine_for(handle: u64, table: &[u8]) -> Result<Arc<SqlEngine>, FfiStatus> {
    let h = lucene_engine::registry::lookup_jvm_reader(
        handle,
        "ffi_jvm_sql_query: unknown or already-closed reader handle",
    )?;
    let cached = {
        let mut map = lock(engines());
        match map.get(&handle) {
            Some(c) if c.table == table => Arc::clone(c),
            _ => {
                let c = Arc::new(Cached {
                    table: table.to_vec(),
                    engine: OnceLock::new(),
                });
                map.insert(handle, Arc::clone(&c));
                c
            }
        }
    };
    let built = cached.engine.get_or_init(|| {
        let build = || -> Result<Arc<SqlEngine>, String> {
            let snapshot = snapshot_of(&h, table).map_err(|_| lucene_engine::status::last_error())?;
            let runtime = runtime().map_err(|_| lucene_engine::status::last_error())?;
            SqlEngine::with_runtime(snapshot, &config(), runtime)
                .map(Arc::new)
                .map_err(|e| format!("starting the SQL engine: {e}"))
        };
        build()
    });
    match built {
        Ok(engine) => Ok(Arc::clone(engine)),
        Err(message) => {
            // A failed build is not kept: the next statement tries again.
            lock(engines()).remove(&handle);
            set_last_error(message.clone());
            Err(FfiStatus::InvalidArgument)
        }
    }
}

/// Runs `sql` over reader `handle` read as the table in `table`, and returns
/// the result as the SQL plugin's JSON response.
pub(crate) fn query_json(handle: u64, table: &[u8], sql: &str) -> Result<Vec<u8>, FfiStatus> {
    let engine = engine_for(handle, table)?;
    let (batches, path) = engine.query_traced(sql).map_err(|e| {
        set_last_error(format!("{e}"));
        FfiStatus::Search
    })?;
    let counter = if path == Path::Router { &ROUTED } else { &STANDARD };
    counter.fetch_add(1, Ordering::Relaxed);
    let schema = lucene_sql::output::result_schema(&batches, &engine.snapshot().spec().arrow_schema());
    let mut out = Vec::with_capacity(256);
    lucene_sql::output::write_json(&mut out, &schema, &batches).map_err(|e| {
        set_last_error(format!("writing the result: {e}"));
        FfiStatus::Search
    })?;
    Ok(out)
}

/// Sets what the SQL engines are built with: the most rows a statement may
/// return, the bytes DataFusion's operators may hold per engine and the
/// threads scans run on (`0`: the number of CPUs). Takes effect for engines
/// built afterwards, and the thread count only before the first statement.
#[no_mangle]
pub extern "C" fn ffi_jvm_sql_configure(max_rows: u64, memory_limit: u64, worker_threads: u32) -> i32 {
    guard(|| {
        let mut config = SqlConfig::default();
        if max_rows > 0 {
            config.max_rows = usize::try_from(max_rows).unwrap_or(usize::MAX);
        }
        if memory_limit > 0 {
            config.memory_limit = usize::try_from(memory_limit).unwrap_or(usize::MAX);
        }
        if worker_threads > 0 {
            config.worker_threads = worker_threads as usize;
            config.target_partitions = worker_threads as usize;
        }
        *lock(&CONFIG) = Some(config);
        Ok(())
    })
}

/// The name of the table the statement reads, as a buffer the caller frees
/// with `ffi_jvm_free_bytes`: what the plugin resolves to an index before it
/// opens anything.
///
/// # Safety
/// `sql` must be valid for `sql_len` bytes; `out_ptr` and `out_len` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_sql_table(
    sql: *const c_char,
    sql_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let sql = unsafe { str_from_raw(sql, sql_len)? };
        let name = lucene_sql::statement::table_name(sql).map_err(|e| {
            set_last_error(format!("{e}"));
            FfiStatus::InvalidArgument
        })?;
        // SAFETY: caller contract.
        unsafe { give(name.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

/// Runs a statement over the shard reader `handle`, which `table` (see
/// [`decode_table`]) describes, and leaves the result as JSON in an owned
/// buffer (`ffi_jvm_free_bytes`).
///
/// # Safety
/// `table` and `sql` must be valid for their lengths; `out_ptr` and `out_len`
/// for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_sql_query(
    handle: u64,
    table: *const u8,
    table_len: usize,
    sql: *const c_char,
    sql_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len)? };
        // SAFETY: caller contract.
        let table = unsafe { bytes_from_raw(table, table_len)? };
        // SAFETY: caller contract.
        let sql = unsafe { str_from_raw(sql, sql_len)? };
        let json = query_json(handle, table, sql)?;
        // SAFETY: caller contract.
        unsafe { give(json, out_ptr, out_len) };
        Ok(())
    })
}

/// Statements answered by the router and by DataFusion since the library
/// loaded.
///
/// # Safety
/// `out` must be valid for two `u64` writes.
#[no_mangle]
pub unsafe extern "C" fn ffi_jvm_sql_stats(out: *mut u64) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        unsafe {
            *out = ROUTED.load(Ordering::Relaxed);
            *out.add(1) = STANDARD.load(Ordering::Relaxed);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests;
