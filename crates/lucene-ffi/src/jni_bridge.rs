//! JNI entry points for `org.lucenerust.opensearch.NativeBridge`, the
//! OpenSearch plugin's binding (`opensearch-plugin/`, M2 T2.3).
//!
//! **JNI, not Panama/FFM.** OpenSearch 3.8.0 supports JDK 21 as its minimum
//! runtime, and on JDK 21 `java.lang.foreign` is still a preview API -- a
//! plugin cannot use it without `--enable-preview` on the node's command
//! line. JNI works on every JDK OpenSearch runs on. The decision and its
//! measurement are recorded in `docs/parity.md`'s `lucene-ffi` section.
//!
//! **Marshalling only.** Every function here copies its Java arguments into
//! Rust memory, calls the matching C-ABI function in `jvm_reader.rs` -- which
//! owns the validation, the `catch_unwind` and the error message -- and
//! copies results back. Nothing here decides anything, so a JVM-free unit
//! test of `jvm_reader.rs` tests the behaviour, and the Java harness
//! (`opensearch-plugin/`'s `NativeBridgeSelfTest`) tests this marshalling.
//!
//! The contract with Java is the `ffi-safety` one: every call returns an
//! `int` status (`FfiStatus`), results go into caller-allocated arrays, and a
//! failure's message is read back with `lastError()`. A JNI failure while
//! marshalling (a null array, a short one) is [`FfiStatus::InvalidArgument`],
//! and any Java exception it left pending is cleared first, so that the
//! status code -- not an exception -- is what the caller sees.
//!
//! Every body runs inside [`guard`], so a panic in the marshalling itself is
//! caught here, not unwound into the JVM.

use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray, JLongArray};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;

use crate::error::{guard, last_error, set_last_error, FfiStatus};
use crate::jvm_reader;
use crate::raw::try_with_capacity;

/// A zeroed buffer of a caller-supplied length, as a status code rather than
/// an abort when the length is absurd -- see `raw::try_with_capacity`.
fn zeroed<T: Copy + Default>(len: usize) -> Result<Vec<T>, FfiStatus> {
    let mut v = try_with_capacity(len)?;
    v.resize(len, T::default());
    Ok(v)
}

/// Turns a JNI failure into [`FfiStatus::InvalidArgument`], clearing any Java
/// exception it left pending.
fn jni_err(env: &JNIEnv<'_>, what: &str, e: jni::errors::Error) -> FfiStatus {
    let _ = env.exception_clear();
    set_last_error(format!("JNI: {what}: {e}"));
    FfiStatus::InvalidArgument
}

/// Runs `body` under [`guard`]; `body`'s own `i32` is the C-ABI call's status.
fn run(body: impl FnOnce() -> Result<i32, FfiStatus>) -> jint {
    let mut inner = FfiStatus::Ok.code();
    let outer = guard(std::panic::AssertUnwindSafe(|| {
        inner = body()?;
        Ok(())
    }));
    if outer != FfiStatus::Ok.code() {
        outer
    } else {
        inner
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_abiVersion(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jint {
    jvm_reader::ffi_jvm_abi_version() as jint
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_lastError(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jstring {
    // Reading a thread-local and building a string cannot panic short of an
    // allocation failure, which `catch_unwind` could not catch either.
    match env.new_string(last_error()) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            let _ = env.exception_clear();
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_openReader<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    path: JByteArray<'l>,
    infos: JByteArray<'l>,
    generation: jlong,
    previous: jlong,
    max_docs: JIntArray<'l>,
    out_handle: JLongArray<'l>,
) -> jint {
    run(|| {
        let path = env
            .convert_byte_array(&path)
            .map_err(|e| jni_err(&env, "path", e))?;
        let infos = env
            .convert_byte_array(&infos)
            .map_err(|e| jni_err(&env, "infos", e))?;
        let n = env
            .get_array_length(&max_docs)
            .map_err(|e| jni_err(&env, "maxDocs", e))?;
        let mut docs: Vec<i32> = zeroed(usize::try_from(n).unwrap_or(0))?;
        env.get_int_array_region(&max_docs, 0, &mut docs)
            .map_err(|e| jni_err(&env, "maxDocs", e))?;
        let mut handle = 0u64;
        // SAFETY: every pointer/length pair describes a live Rust buffer.
        let status = unsafe {
            jvm_reader::ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                infos.as_ptr(),
                infos.len(),
                generation,
                previous as u64,
                docs.as_ptr(),
                docs.len(),
                &mut handle,
            )
        };
        if status == FfiStatus::Ok.code() {
            env.set_long_array_region(&out_handle, 0, &[handle as jlong])
                .map_err(|e| jni_err(&env, "outHandle", e))?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_setLiveDocs<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    segment: jint,
    words: JLongArray<'l>,
) -> jint {
    run(|| {
        let segment = usize::try_from(segment).map_err(|_| {
            set_last_error(format!("segment {segment} is negative"));
            FfiStatus::IndexOutOfBounds
        })?;
        if words.is_null() {
            // SAFETY: null with length 0 is the documented "no deletions".
            return Ok(unsafe {
                jvm_reader::ffi_jvm_reader_set_live_docs(
                    handle as u64,
                    segment,
                    std::ptr::null(),
                    0,
                )
            });
        }
        let n = env
            .get_array_length(&words)
            .map_err(|e| jni_err(&env, "words", e))?;
        let mut buf: Vec<i64> = zeroed(usize::try_from(n).unwrap_or(0))?;
        env.get_long_array_region(&words, 0, &mut buf)
            .map_err(|e| jni_err(&env, "words", e))?;
        let buf: Vec<u64> = buf.into_iter().map(|w| w as u64).collect();
        // SAFETY: `buf` is live for its length.
        Ok(unsafe {
            jvm_reader::ffi_jvm_reader_set_live_docs(
                handle as u64,
                segment,
                buf.as_ptr(),
                buf.len(),
            )
        })
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_search<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    query: JByteArray<'l>,
    top_n: jint,
    count_limit: jlong,
    out_docs: JIntArray<'l>,
    out_scores: JFloatArray<'l>,
    out_counts: JLongArray<'l>,
) -> jint {
    run(|| {
        let top_n = usize::try_from(top_n).map_err(|_| {
            set_last_error(format!("topN {top_n} is negative"));
            FfiStatus::InvalidArgument
        })?;
        let blob = env
            .convert_byte_array(&query)
            .map_err(|e| jni_err(&env, "query", e))?;
        let docs_len = if top_n == 0 {
            0
        } else {
            let a = env
                .get_array_length(&out_docs)
                .map_err(|e| jni_err(&env, "outDocs", e))?;
            let b = env
                .get_array_length(&out_scores)
                .map_err(|e| jni_err(&env, "outScores", e))?;
            usize::try_from(a.min(b)).unwrap_or(0)
        };
        if docs_len < top_n {
            set_last_error(format!(
                "output arrays hold {docs_len} hits, topN is {top_n}"
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        let mut docs: Vec<i32> = zeroed(top_n)?;
        let mut scores: Vec<f32> = zeroed(top_n)?;
        let mut hit_count = 0usize;
        let mut total = 0i64;
        let mut lower_bound = false;
        // SAFETY: every pointer/length pair describes a live Rust buffer.
        let status = unsafe {
            jvm_reader::ffi_jvm_reader_search(
                handle as u64,
                blob.as_ptr(),
                blob.len(),
                top_n,
                count_limit,
                docs.as_mut_ptr(),
                scores.as_mut_ptr(),
                top_n,
                &mut hit_count,
                &mut total,
                &mut lower_bound,
            )
        };
        if status == FfiStatus::Ok.code() {
            env.set_int_array_region(&out_docs, 0, &docs[..hit_count])
                .map_err(|e| jni_err(&env, "outDocs", e))?;
            env.set_float_array_region(&out_scores, 0, &scores[..hit_count])
                .map_err(|e| jni_err(&env, "outScores", e))?;
            env.set_long_array_region(
                &out_counts,
                0,
                &[hit_count as jlong, total, jlong::from(lower_bound)],
            )
            .map_err(|e| jni_err(&env, "outCounts", e))?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_closeReader(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jint {
    jvm_reader::ffi_close_jvm_reader(handle as u64)
}
