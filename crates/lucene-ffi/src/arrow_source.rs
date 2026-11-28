//! The table of an index as an Arrow source: record batches over the Arrow C
//! stream interface, which pyarrow, Polars, DuckDB and the JVM's Arrow
//! (through the Spark connector) import without a copy.
//!
//! A caller opens a table ([`ffi_arrow_table_open`]: the index directory and
//! the table declaration as JSON), then starts any number of streams over it
//! ([`ffi_arrow_scan`]: columns, filters and a limit as JSON, one partition
//! of several), each filling an `ArrowArrayStream` it owns and releases
//! itself. [`ffi_arrow_sql`] (feature `arrow-sql`) starts a stream from a SQL
//! statement.
//!
//! The declaration is `{"name": "t", "columns": [{"name": "id", "kind":
//! "key"}, ...]}`, kinds `key`, `long`, `keyword`, `text` and `vector` (with
//! `dim`). A request is `{"columns": [...], "filters": [{"column": "price",
//! "op": ">=", "value": 10}, {"column": "cat", "in": ["a", "b"]}, {"column":
//! "x", "is_null": true}], "limit": 100, "batch_rows": 65536, "dictionary":
//! true, "threads": 0}`.

use std::sync::{Arc, OnceLock, RwLock};

use lucene_engine::handle::{RegistryTag, SlotMap};
use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::arrow::ffi_stream::FFI_ArrowArrayStream;
use lucene_sql::export::{Filter, Literal, Op, Scan, ScanOptions};
use lucene_sql::{Snapshot, TableSpec};
use lucene_store::FsDirectory;
use serde_json::Value;

use crate::error::{guard, set_last_error, FfiStatus};
use crate::ffm_bridge::{clear_out, give};
use crate::raw::str_from_raw;
use crate::registry::{lock_recovering, read_recovering};

struct Table {
    snapshot: Arc<Snapshot>,
    #[cfg(feature = "arrow-sql")]
    sql: std::sync::Mutex<Option<Arc<lucene_sql::SqlEngine>>>,
}

fn tables() -> &'static RwLock<SlotMap<Arc<Table>>> {
    static REGISTRY: OnceLock<RwLock<SlotMap<Arc<Table>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(SlotMap::new(RegistryTag::ArrowTable)))
}

fn invalid(e: impl std::fmt::Display) -> FfiStatus {
    set_last_error(e.to_string());
    FfiStatus::InvalidArgument
}

fn table(handle: u64) -> Result<Arc<Table>, FfiStatus> {
    read_recovering(tables())
        .get(handle)
        .cloned()
        .ok_or(FfiStatus::InvalidHandle)
}

fn parse_spec(json: &str) -> Result<TableSpec, FfiStatus> {
    let v: Value = serde_json::from_str(json).map_err(invalid)?;
    let name = v["name"].as_str().ok_or_else(|| invalid("table `name`"))?;
    let mut spec = TableSpec::new(name);
    for c in v["columns"].as_array().into_iter().flatten() {
        let n = c["name"].as_str().ok_or_else(|| invalid("column `name`"))?;
        spec = match c["kind"].as_str() {
            Some("key") => spec.key(n),
            Some("long") => spec.long(n),
            Some("keyword") => spec.keyword(n),
            Some("text") => spec.text(n),
            Some("vector") => spec.vector(
                n,
                c["dim"]
                    .as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| invalid("vector column `dim`"))?,
            ),
            other => return Err(invalid(format!("column `{n}` kind {other:?}"))),
        };
    }
    Ok(spec)
}

fn literal(v: &Value) -> Result<Literal, FfiStatus> {
    match v {
        Value::String(s) => Ok(Literal::Str(s.clone())),
        Value::Number(n) => n
            .as_i64()
            .map(Literal::Long)
            .ok_or_else(|| invalid(format!("{n} is not an integer"))),
        other => Err(invalid(format!("filter value {other}"))),
    }
}

fn op_of(s: &str) -> Result<Op, FfiStatus> {
    Ok(match s {
        "=" | "==" => Op::Eq,
        "!=" | "<>" => Op::Ne,
        "<" => Op::Lt,
        "<=" => Op::Le,
        ">" => Op::Gt,
        ">=" => Op::Ge,
        other => return Err(invalid(format!("operator `{other}`"))),
    })
}

/// A non-negative integer option, `None` when absent.
fn count(v: &Value, name: &str) -> Result<Option<usize>, FfiStatus> {
    match v {
        Value::Null => Ok(None),
        other => other
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| invalid(format!("`{name}` is a non-negative integer"))),
    }
}

fn parse_request(json: &str) -> Result<ScanOptions, FfiStatus> {
    let v: Value = if json.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(json).map_err(invalid)?
    };
    let mut o = ScanOptions::default();
    if let Some(list) = v["columns"].as_array() {
        o.columns = Some(
            list.iter()
                .map(|c| {
                    c.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| invalid("column names are strings"))
                })
                .collect::<Result<_, _>>()?,
        );
    }
    for f in v["filters"].as_array().into_iter().flatten() {
        let column = f["column"]
            .as_str()
            .ok_or_else(|| invalid("filter `column`"))?
            .to_string();
        o.filters.push(if let Some(list) = f["in"].as_array() {
            Filter::In {
                column,
                values: list.iter().map(literal).collect::<Result<_, _>>()?,
            }
        } else if let Some(null) = f["is_null"].as_bool() {
            if null {
                Filter::IsNull { column }
            } else {
                Filter::IsNotNull { column }
            }
        } else {
            Filter::Cmp {
                column,
                op: op_of(f["op"].as_str().ok_or_else(|| invalid("filter `op`"))?)?,
                value: literal(&f["value"])?,
            }
        });
    }
    if let Some(l) = count(&v["limit"], "limit")? {
        o.limit = Some(l);
    }
    if let Some(b) = count(&v["batch_rows"], "batch_rows")? {
        o.batch_rows = b;
    }
    if let Some(d) = v["dictionary"].as_bool() {
        o.dictionary = d;
    }
    if let Some(t) = count(&v["threads"], "threads")? {
        o.threads = t;
    }
    Ok(o)
}

/// Opens the index in directory `path` as the table `spec` (JSON) declares.
pub(crate) fn open_table(path: &str, spec: &str) -> Result<u64, FfiStatus> {
    let spec = parse_spec(spec)?;
    let reader = DirectoryReader::open(&FsDirectory::open(path.to_string())).map_err(invalid)?;
    let snapshot = Arc::new(Snapshot::new(reader, spec).map_err(invalid)?);
    Ok(lock_recovering(tables()).insert_checked(Arc::new(Table {
        snapshot,
        #[cfg(feature = "arrow-sql")]
        sql: std::sync::Mutex::new(None),
    }))?)
}

/// Releases a table.
pub(crate) fn close_table(handle: u64) -> Result<(), FfiStatus> {
    lock_recovering(tables())
        .remove(handle)
        .map(drop)
        .ok_or(FfiStatus::InvalidHandle)
}

/// `{"segments": n, "docs": n, "max_doc": n}` of a table.
pub(crate) fn table_info(handle: u64) -> Result<String, FfiStatus> {
    let t = table(handle)?;
    let s = &t.snapshot;
    Ok(format!(
        "{{\"segments\":{},\"docs\":{},\"max_doc\":{}}}",
        s.segment_count(),
        s.num_docs(),
        s.max_doc()
    ))
}

/// The stream of partition `partition` of `partitions` of the scan `request`
/// describes.
pub(crate) fn start_scan(
    handle: u64,
    request: &str,
    partition: u32,
    partitions: u32,
) -> Result<FFI_ArrowArrayStream, FfiStatus> {
    let t = table(handle)?;
    let mut options = parse_request(request)?;
    if partitions > 1 && options.threads == 0 {
        // The caller runs the partitions side by side already.
        options.threads = 1;
    }
    let scan = Scan::new(t.snapshot.clone(), &options).map_err(invalid)?;
    let reader = if partitions <= 1 {
        scan.open()
    } else {
        let groups = scan.partitions(partitions as usize);
        scan.open_segments(groups.get(partition as usize).cloned().unwrap_or_default())
    };
    Ok(FFI_ArrowArrayStream::new(Box::new(reader)))
}

/// Opens the index in directory `path` as the table `spec` (JSON) declares;
/// its handle goes to `*out_handle`. Release it with
/// [`ffi_arrow_table_close`] once every stream over it is released.
///
/// # Safety
/// Each pointer must be valid for its length; `out_handle` for one write.
#[no_mangle]
pub unsafe extern "C" fn ffi_arrow_table_open(
    path: *const u8,
    path_len: usize,
    spec: *const u8,
    spec_len: usize,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if out_handle.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path.cast(), path_len) }?;
        // SAFETY: caller contract.
        let spec = unsafe { str_from_raw(spec.cast(), spec_len) }?;
        let handle = open_table(path, spec)?;
        // SAFETY: caller contract.
        unsafe { *out_handle = handle };
        Ok(())
    })
}

/// Closes a table. Streams already started keep reading it.
#[no_mangle]
pub extern "C" fn ffi_arrow_table_close(handle: u64) -> i32 {
    guard(|| close_table(handle))
}

/// The table's facts as UTF-8 JSON in an owned buffer (free it with
/// `ffi_jvm_free_bytes`): `{"segments": n, "docs": n, "max_doc": n}`.
///
/// # Safety
/// `out_ptr` and `out_len` must be valid for one write each.
#[no_mangle]
pub unsafe extern "C" fn ffi_arrow_table_info(
    handle: u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        unsafe { clear_out(out_ptr, out_len) }?;
        let json = table_info(handle)?;
        // SAFETY: caller contract.
        unsafe { give(json.into_bytes(), out_ptr, out_len) };
        Ok(())
    })
}

/// Starts a stream over `partition` of `partitions` (`0` of `1`: the whole
/// table) of the scan `request` (JSON, may be empty) describes, into
/// `*out_stream`. The consumer releases the stream; `limit` applies to each
/// partition.
///
/// # Safety
/// `request` must be valid for its length; `out_stream` for one write of an
/// `ArrowArrayStream`.
#[no_mangle]
pub unsafe extern "C" fn ffi_arrow_scan(
    handle: u64,
    request: *const u8,
    request_len: usize,
    partition: u32,
    partitions: u32,
    out_stream: *mut FFI_ArrowArrayStream,
) -> i32 {
    guard(|| {
        if out_stream.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: caller contract.
        let request = unsafe { str_from_raw(request.cast(), request_len) }?;
        let stream = start_scan(handle, request, partition, partitions)?;
        // SAFETY: caller contract: `out_stream` is valid for one write, and
        // the stream owns what it points to.
        unsafe { std::ptr::write(out_stream, stream) };
        Ok(())
    })
}

/// How many partitions `request` splits into when asked for at most
/// `partitions`: the readers [`ffi_arrow_scan`] can fill with segments.
#[no_mangle]
pub unsafe extern "C" fn ffi_arrow_partition_count(
    handle: u64,
    partitions: u32,
    out_count: *mut u32,
) -> i32 {
    guard(|| {
        if out_count.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let t = table(handle)?;
        let n = t.snapshot.segment_count().min(partitions.max(1) as usize);
        // SAFETY: caller contract.
        unsafe { *out_count = u32::try_from(n).unwrap_or(u32::MAX) };
        Ok(())
    })
}

/// Starts a stream over the rows `sql` returns.
///
/// # Safety
/// `sql` must be valid for its length; `out_stream` for one write of an
/// `ArrowArrayStream`.
#[cfg(feature = "arrow-sql")]
#[no_mangle]
pub unsafe extern "C" fn ffi_arrow_sql(
    handle: u64,
    sql: *const u8,
    sql_len: usize,
    out_stream: *mut FFI_ArrowArrayStream,
) -> i32 {
    guard(|| {
        if out_stream.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        let t = table(handle)?;
        // SAFETY: caller contract.
        let sql = unsafe { str_from_raw(sql.cast(), sql_len) }?;
        let engine = {
            let mut slot = t
                .sql
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match &*slot {
                Some(e) => e.clone(),
                None => {
                    let e = Arc::new(
                        lucene_sql::SqlEngine::new(
                            t.snapshot.clone(),
                            &lucene_sql::SqlConfig::default(),
                        )
                        .map_err(invalid)?,
                    );
                    *slot = Some(e.clone());
                    e
                }
            }
        };
        let reader = engine.stream_blocking(sql).map_err(invalid)?;
        let stream = FFI_ArrowArrayStream::new(reader);
        // SAFETY: caller contract.
        unsafe { std::ptr::write(out_stream, stream) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::{
        Document, KeywordField, NumericDocValuesField, Store, StringField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_sql::arrow::ffi_stream::ArrowArrayStreamReader;
    use lucene_util::test_support::TempDir;

    const SPEC: &str = r#"{"name":"t","columns":[{"name":"id","kind":"key"},
        {"name":"price","kind":"long"},{"name":"cat","kind":"keyword"},
        {"name":"body","kind":"text"},{"name":"vec","kind":"vector","dim":4}]}"#;

    fn index() -> TempDir {
        let dir = TempDir::new("arrow-source");
        let fs = FsDirectory::open(dir.path_str());
        let v = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", v).unwrap();
        for i in 0..300 {
            let mut d = Document::new();
            d.add(StringField::new("id", format!("d{i:04}"), Store::Yes));
            d.add(NumericDocValuesField::new("price", i % 50));
            d.add(KeywordField::new("cat", format!("c{}", i % 3), Store::No));
            w.add_fields_document(&d).unwrap();
            if i % 100 == 99 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        dir
    }

    fn open(dir: &TempDir) -> u64 {
        let path = dir.path_str();
        let mut h = 0u64;
        // SAFETY: the pointers are valid for their lengths and `h` for one write.
        let st = unsafe {
            ffi_arrow_table_open(path.as_ptr(), path.len(), SPEC.as_ptr(), SPEC.len(), &mut h)
        };
        assert_eq!(st, 0, "{}", crate::error::last_error());
        h
    }

    fn rows(h: u64, request: &str, partition: u32, partitions: u32) -> usize {
        let mut stream = std::mem::MaybeUninit::<FFI_ArrowArrayStream>::uninit();
        // SAFETY: valid pointers; the stream is written on success.
        let st = unsafe {
            ffi_arrow_scan(
                h,
                request.as_ptr(),
                request.len(),
                partition,
                partitions,
                stream.as_mut_ptr(),
            )
        };
        assert_eq!(st, 0, "{}", crate::error::last_error());
        // SAFETY: the scan wrote a stream, which the reader now owns.
        let reader = unsafe { ArrowArrayStreamReader::from_raw(stream.as_mut_ptr()) }.unwrap();
        reader.map(|b| b.unwrap().num_rows()).sum()
    }

    #[test]
    fn a_table_opens_scans_reports_and_closes() {
        let dir = index();
        let h = open(&dir);
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: valid out-pointers.
        assert_eq!(unsafe { ffi_arrow_table_info(h, &mut ptr, &mut len) }, 0);
        // SAFETY: the library returned `len` bytes at `ptr`.
        let info = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
        // SAFETY: the pair is the one the call returned.
        unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
        assert_eq!(
            String::from_utf8(info).unwrap(),
            r#"{"segments":3,"docs":300,"max_doc":300}"#
        );

        assert_eq!(rows(h, "", 0, 1), 300);
        assert_eq!(rows(h, r#"{"columns":["price"],"limit":10}"#, 0, 1), 10);
        let filtered = r#"{"columns":["id","cat"],"dictionary":false,"threads":2,"batch_rows":64,
            "filters":[{"column":"price","op":">=","value":40},
                       {"column":"cat","in":["c0","c1"]},
                       {"column":"cat","is_null":false}]}"#;
        let expect = (0..300).filter(|i| i % 50 >= 40 && i % 3 != 2).count();
        assert_eq!(rows(h, filtered, 0, 1), expect);
        let parts: usize = (0..2).map(|p| rows(h, filtered, p, 2)).sum();
        assert_eq!(parts, expect);
        assert_eq!(rows(h, r#"{"columns":[]}"#, 0, 1), 300);
        let null = r#"{"filters":[{"column":"price","is_null":true}]}"#;
        assert_eq!(rows(h, null, 0, 1), 0);

        let mut n = 0u32;
        // SAFETY: valid out-pointer.
        assert_eq!(unsafe { ffi_arrow_partition_count(h, 8, &mut n) }, 0);
        assert_eq!(n, 3);
        assert_eq!(ffi_arrow_table_close(h), 0);
        assert_ne!(ffi_arrow_table_close(h), 0, "a closed handle is refused");
    }

    #[test]
    fn bad_input_is_refused_with_a_message() {
        let dir = index();
        let h = open(&dir);
        let mut stream = std::mem::MaybeUninit::<FFI_ArrowArrayStream>::uninit();
        for bad in [
            "{",
            r#"{"columns":[1]}"#,
            r#"{"columns":["nope"]}"#,
            r#"{"filters":[{"column":"price"}]}"#,
            r#"{"filters":[{"column":"price","op":"~","value":1}]}"#,
            r#"{"filters":[{"column":"price","op":"=","value":1.5}]}"#,
            r#"{"filters":[{"column":"price","op":"=","value":true}]}"#,
            r#"{"filters":[{"op":"=","value":1}]}"#,
            r#"{"filters":[{"column":"cat","op":"=","value":1}]}"#,
            r#"{"limit":-1}"#,
        ] {
            // SAFETY: valid pointers.
            let st =
                unsafe { ffi_arrow_scan(h, bad.as_ptr(), bad.len(), 0, 1, stream.as_mut_ptr()) };
            assert_ne!(st, 0, "{bad}");
            assert!(!crate::error::last_error().is_empty());
        }
        // SAFETY: a null out-pointer is checked for.
        assert_ne!(
            unsafe { ffi_arrow_scan(h, b"".as_ptr(), 0, 0, 1, std::ptr::null_mut()) },
            0
        );
        // SAFETY: as above.
        assert_ne!(
            unsafe { ffi_arrow_table_open(b"".as_ptr(), 0, b"".as_ptr(), 0, std::ptr::null_mut()) },
            0
        );
        for spec in [
            "{",
            r#"{"columns":[]}"#,
            r#"{"name":"t","columns":[{"kind":"long"}]}"#,
            r#"{"name":"t","columns":[{"name":"a","kind":"blob"}]}"#,
            r#"{"name":"t","columns":[{"name":"a","kind":"vector"}]}"#,
        ] {
            let path = dir.path_str();
            let mut out = 0u64;
            // SAFETY: valid pointers.
            let st = unsafe {
                ffi_arrow_table_open(
                    path.as_ptr(),
                    path.len(),
                    spec.as_ptr(),
                    spec.len(),
                    &mut out,
                )
            };
            assert_ne!(st, 0, "{spec}");
        }
        let missing = "/nonexistent/index";
        let mut out = 0u64;
        // SAFETY: valid pointers.
        let st = unsafe {
            ffi_arrow_table_open(
                missing.as_ptr(),
                missing.len(),
                SPEC.as_ptr(),
                SPEC.len(),
                &mut out,
            )
        };
        assert_ne!(st, 0);
        let mut n = 0u32;
        // SAFETY: valid pointer / null checked.
        assert_ne!(
            unsafe { ffi_arrow_partition_count(h, 1, std::ptr::null_mut()) },
            0
        );
        assert_ne!(unsafe { ffi_arrow_partition_count(0, 1, &mut n) }, 0);
        // SAFETY: null out-pointers are checked for.
        assert_ne!(
            unsafe { ffi_arrow_table_info(h, std::ptr::null_mut(), std::ptr::null_mut()) },
            0
        );
        assert_eq!(ffi_arrow_table_close(h), 0);
    }

    #[cfg(feature = "arrow-sql")]
    #[test]
    fn sql_streams_rows() {
        let dir = index();
        let h = open(&dir);
        let run = |sql: &str| -> Result<usize, i32> {
            let mut stream = std::mem::MaybeUninit::<FFI_ArrowArrayStream>::uninit();
            // SAFETY: valid pointers.
            let st = unsafe { ffi_arrow_sql(h, sql.as_ptr(), sql.len(), stream.as_mut_ptr()) };
            if st != 0 {
                return Err(st);
            }
            // SAFETY: the call wrote a stream the reader owns.
            let reader = unsafe { ArrowArrayStreamReader::from_raw(stream.as_mut_ptr()) }.unwrap();
            Ok(reader.map(|b| b.unwrap().num_rows()).sum())
        };
        assert_eq!(run("SELECT id FROM t WHERE price >= 40"), Ok(60));
        // The engine is kept, so a second statement reuses it.
        assert_eq!(run("SELECT cat, count(*) FROM t GROUP BY cat"), Ok(3));
        assert!(run("SELEC").is_err());
        // SAFETY: a null out-pointer is checked for.
        assert_ne!(
            unsafe { ffi_arrow_sql(h, b"".as_ptr(), 0, std::ptr::null_mut()) },
            0
        );
        assert_eq!(ffi_arrow_table_close(h), 0);
    }
}
