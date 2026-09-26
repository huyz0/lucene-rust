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

use jni::objects::{JByteArray, JClass, JFloatArray, JIntArray, JLongArray, JObjectArray};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;

use crate::engine_writer;
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
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_openReader<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    path: JByteArray<'l>,
    infos: JByteArray<'l>,
    generation: jlong,
    previous: jlong,
    max_docs: JIntArray<'l>,
    live_docs: JObjectArray<'l>,
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
        let segments = usize::try_from(n).unwrap_or(0);
        let mut docs: Vec<i32> = zeroed(segments)?;
        env.get_int_array_region(&max_docs, 0, &mut docs)
            .map_err(|e| jni_err(&env, "maxDocs", e))?;
        // `liveDocs[i]` is segment i's words, or null for no deletions; a null
        // array means no segment has any.
        let mut counts: Vec<usize> = zeroed(segments)?;
        let mut words: Vec<u64> = Vec::new();
        if !live_docs.is_null() {
            let m = env
                .get_array_length(&live_docs)
                .map_err(|e| jni_err(&env, "liveDocs", e))?;
            if m != n {
                set_last_error(format!("liveDocs has {m} entries for {n} segments"));
                return Err(FfiStatus::InvalidArgument);
            }
            for (i, count) in counts.iter_mut().enumerate() {
                let entry = env
                    .get_object_array_element(&live_docs, i as jint)
                    .map_err(|e| jni_err(&env, "liveDocs", e))?;
                if entry.is_null() {
                    continue;
                }
                let arr = JLongArray::from(entry);
                let len = env
                    .get_array_length(&arr)
                    .map_err(|e| jni_err(&env, "liveDocs", e))?;
                let mut buf: Vec<i64> = zeroed(usize::try_from(len).unwrap_or(0))?;
                env.get_long_array_region(&arr, 0, &mut buf)
                    .map_err(|e| jni_err(&env, "liveDocs", e))?;
                let _ = env.delete_local_ref(arr);
                *count = buf.len();
                words
                    .try_reserve(buf.len())
                    .map_err(|_| FfiStatus::InvalidArgument)?;
                words.extend(buf.into_iter().map(|w| w as u64));
            }
        }
        let mut handle = 0u64;
        // SAFETY: every pointer/length pair describes a live Rust buffer, and
        // `words` holds exactly the sum of `counts`.
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
                words.as_ptr(),
                counts.as_ptr(),
                &mut handle,
            )
        };
        if status == FfiStatus::Ok.code() {
            if let Err(e) = env.set_long_array_region(&out_handle, 0, &[handle as jlong]) {
                // Java will never see this handle, so nothing would close it.
                jvm_reader::ffi_close_jvm_reader(handle);
                return Err(jni_err(&env, "outHandle", e));
            }
        }
        Ok(status)
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
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_searchSorted<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    query: JByteArray<'l>,
    sort: JByteArray<'l>,
    top_n: jint,
    count_limit: jlong,
    out_docs: JIntArray<'l>,
    out_values: JLongArray<'l>,
    out_counts: JLongArray<'l>,
    out_terms: JObjectArray<'l>,
) -> jint {
    run(|| {
        let top_n = usize::try_from(top_n).map_err(|_| {
            set_last_error(format!("topN {top_n} is negative"));
            FfiStatus::InvalidArgument
        })?;
        let blob = env
            .convert_byte_array(&query)
            .map_err(|e| jni_err(&env, "query", e))?;
        let sort_blob = env
            .convert_byte_array(&sort)
            .map_err(|e| jni_err(&env, "sort", e))?;
        let keys = usize::from(sort_blob.first().copied().unwrap_or(0));
        let docs_len = env
            .get_array_length(&out_docs)
            .map_err(|e| jni_err(&env, "outDocs", e))?;
        let values_len = env
            .get_array_length(&out_values)
            .map_err(|e| jni_err(&env, "outValues", e))?;
        let docs_len = usize::try_from(docs_len).unwrap_or(0);
        let values_len = usize::try_from(values_len).unwrap_or(0);
        let want_values = top_n.checked_mul(keys).ok_or(FfiStatus::InvalidArgument)?;
        if docs_len < top_n || values_len < want_values {
            set_last_error(format!(
                "output arrays hold {docs_len} hits and {values_len} values, topN is {top_n} with {keys} keys"
            ));
            return Err(FfiStatus::BufferTooSmall);
        }
        let mut docs: Vec<i32> = zeroed(top_n)?;
        let mut values: Vec<i64> = zeroed(want_values)?;
        let mut hit_count = 0usize;
        let mut total = 0i64;
        let mut lower_bound = false;
        // Keyword keys hand back terms: room for short ones first, and the
        // exact room the search reports when they are longer.
        // A malformed blob is reported by the search itself.
        let string_keys = jvm_reader::decode_sort(&sort_blob).map_or(0, |(k, _)| {
            k.iter()
                .filter(|k| k.ty == lucene_search::top_field::SortType::String)
                .count()
        });
        let mut cap = top_n.saturating_mul(string_keys).saturating_mul(32);
        let mut status = FfiStatus::Ok.code();
        let mut terms: Vec<u8> = Vec::new();
        let mut terms_len = 0usize;
        // The same reader answers the same way, so a second try fits.
        for _ in 0..2 {
            terms = zeroed(cap)?;
            // SAFETY: every pointer/length pair describes a live Rust buffer;
            // `values` holds `top_n` hits of `keys` values, the sort blob's
            // count, and `terms` `cap` bytes.
            status = unsafe {
                jvm_reader::ffi_jvm_reader_search_sorted(
                    handle as u64,
                    blob.as_ptr(),
                    blob.len(),
                    sort_blob.as_ptr(),
                    sort_blob.len(),
                    top_n,
                    count_limit,
                    docs.as_mut_ptr(),
                    values.as_mut_ptr(),
                    top_n,
                    terms.as_mut_ptr(),
                    cap,
                    &mut hit_count,
                    &mut total,
                    &mut lower_bound,
                    &mut terms_len,
                )
            };
            if status != FfiStatus::BufferTooSmall.code() || terms_len <= cap {
                break;
            }
            cap = terms_len;
        }
        if status == FfiStatus::Ok.code() {
            if string_keys > 0 {
                let arr = env
                    .byte_array_from_slice(&terms[..terms_len])
                    .map_err(|e| jni_err(&env, "outTerms", e))?;
                env.set_object_array_element(&out_terms, 0, &arr)
                    .map_err(|e| jni_err(&env, "outTerms", e))?;
            }
            env.set_int_array_region(&out_docs, 0, &docs[..hit_count])
                .map_err(|e| jni_err(&env, "outDocs", e))?;
            env.set_long_array_region(&out_values, 0, &values[..hit_count * keys])
                .map_err(|e| jni_err(&env, "outValues", e))?;
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

// ---------------------------------------------------------------------------
// The engine writer (`engine_writer.rs`, M5): same marshalling-only rule.
// ---------------------------------------------------------------------------

/// Copies `arr[..len]` (or all of `arr` when `len < 0`).
fn bytes_of(
    env: &JNIEnv<'_>,
    arr: &JByteArray<'_>,
    len: jint,
    what: &str,
) -> Result<Vec<u8>, FfiStatus> {
    let total = env
        .get_array_length(arr)
        .map_err(|e| jni_err(env, what, e))?;
    let n = if len < 0 { total } else { len };
    if n > total {
        set_last_error(format!("{what}: length {n} exceeds the array's {total}"));
        return Err(FfiStatus::InvalidArgument);
    }
    let mut buf: Vec<i8> = zeroed(usize::try_from(n).unwrap_or(0))?;
    env.get_byte_array_region(arr, 0, &mut buf)
        .map_err(|e| jni_err(env, what, e))?;
    Ok(buf.into_iter().map(|b| b as u8).collect())
}

fn set_long(
    env: &JNIEnv<'_>,
    arr: &JLongArray<'_>,
    value: i64,
    what: &str,
) -> Result<(), FfiStatus> {
    env.set_long_array_region(arr, 0, &[value])
        .map_err(|e| jni_err(env, what, e))
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerOpen<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    path: JByteArray<'l>,
    ram_buffer_mb: f64,
    fault_injection: jni::sys::jboolean,
    max_docs: jint,
    out_handle: JLongArray<'l>,
) -> jint {
    run(|| {
        let path = bytes_of(&env, &path, -1, "path")?;
        let mut handle = 0u64;
        // SAFETY: live buffer and out-pointer.
        let status = unsafe {
            engine_writer::ffi_engine_writer_open(
                path.as_ptr().cast(),
                path.len(),
                ram_buffer_mb,
                fault_injection,
                max_docs,
                &mut handle,
            )
        };
        if status == FfiStatus::Ok.code() {
            if let Err(e) = set_long(&env, &out_handle, handle as i64, "outHandle") {
                engine_writer::ffi_engine_writer_close(handle);
                return Err(e);
            }
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerRegisterField<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    spec: JByteArray<'l>,
    out_number: JIntArray<'l>,
) -> jint {
    run(|| {
        let spec = bytes_of(&env, &spec, -1, "spec")?;
        let mut number = 0i32;
        // SAFETY: live buffer and out-pointer.
        let status = unsafe {
            engine_writer::ffi_engine_writer_register_field(
                handle as u64,
                spec.as_ptr(),
                spec.len(),
                &mut number,
            )
        };
        if status == FfiStatus::Ok.code() {
            env.set_int_array_region(&out_number, 0, &[number])
                .map_err(|e| jni_err(&env, "outNumber", e))?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerApply<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    op: JByteArray<'l>,
    len: jint,
) -> jint {
    run(|| {
        let op = bytes_of(&env, &op, len, "op")?;
        // SAFETY: live buffer.
        Ok(unsafe { engine_writer::ffi_engine_writer_apply(handle as u64, op.as_ptr(), op.len()) })
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerCommit<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    user_data: JByteArray<'l>,
    out_generation: JLongArray<'l>,
) -> jint {
    run(|| {
        let data = bytes_of(&env, &user_data, -1, "userData")?;
        let mut generation = 0i64;
        // SAFETY: live buffer and out-pointer.
        let status = unsafe {
            engine_writer::ffi_engine_writer_commit(
                handle as u64,
                data.as_ptr(),
                data.len(),
                &mut generation,
            )
        };
        if status == FfiStatus::Ok.code() {
            set_long(&env, &out_generation, generation, "outGeneration")?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerCommitGenerations<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    out: JLongArray<'l>,
    out_len: JIntArray<'l>,
) -> jint {
    run(|| {
        let cap = env
            .get_array_length(&out)
            .map_err(|e| jni_err(&env, "out", e))?;
        let mut buf: Vec<i64> = zeroed(usize::try_from(cap).unwrap_or(0))?;
        let mut n = 0usize;
        // SAFETY: `buf` holds `buf.len()` values.
        let status = unsafe {
            engine_writer::ffi_engine_writer_commit_generations(
                handle as u64,
                buf.as_mut_ptr(),
                buf.len(),
                &mut n,
            )
        };
        let n32 = jint::try_from(n).unwrap_or(jint::MAX);
        env.set_int_array_region(&out_len, 0, &[n32])
            .map_err(|e| jni_err(&env, "outLen", e))?;
        if status == FfiStatus::Ok.code() {
            env.set_long_array_region(&out, 0, &buf[..n])
                .map_err(|e| jni_err(&env, "out", e))?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerDeleteCommits<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    generations: JLongArray<'l>,
) -> jint {
    run(|| {
        let n = env
            .get_array_length(&generations)
            .map_err(|e| jni_err(&env, "generations", e))?;
        let mut gens: Vec<i64> = zeroed(usize::try_from(n).unwrap_or(0))?;
        env.get_long_array_region(&generations, 0, &mut gens)
            .map_err(|e| jni_err(&env, "generations", e))?;
        // SAFETY: live buffer.
        Ok(unsafe {
            engine_writer::ffi_engine_writer_delete_commits(
                handle as u64,
                gens.as_ptr(),
                gens.len(),
            )
        })
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerHoldCommit<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    generation: jlong,
    out_hold: JLongArray<'l>,
) -> jint {
    run(|| {
        let mut hold = 0u64;
        // SAFETY: live out-pointer.
        let status = unsafe {
            engine_writer::ffi_engine_writer_hold_commit(handle as u64, generation, &mut hold)
        };
        if status == FfiStatus::Ok.code() {
            if let Err(e) = set_long(&env, &out_hold, hold as i64, "outHold") {
                engine_writer::ffi_engine_writer_release_hold(handle as u64, hold);
                return Err(e);
            }
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerReleaseHold(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    hold: jlong,
) -> jint {
    engine_writer::ffi_engine_writer_release_hold(handle as u64, hold as u64)
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerSetRetention(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    enabled: jni::sys::jboolean,
    min_retained_seq_no: jlong,
) -> jint {
    engine_writer::ffi_engine_writer_set_retention(handle as u64, enabled, min_retained_seq_no)
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerForceMerge<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    max_segments: jint,
    only_deletes: jni::sys::jboolean,
    out_generation: JLongArray<'l>,
) -> jint {
    run(|| {
        let mut generation = 0i64;
        // SAFETY: live out-pointer.
        let status = unsafe {
            engine_writer::ffi_engine_writer_force_merge(
                handle as u64,
                max_segments,
                only_deletes,
                &mut generation,
            )
        };
        if status == FfiStatus::Ok.code() {
            set_long(&env, &out_generation, generation, "outGeneration")?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerStats<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    out: JLongArray<'l>,
) -> jint {
    run(|| {
        let mut stats = [0i64; engine_writer::STAT_COUNT];
        // SAFETY: `stats` holds `STAT_COUNT` values.
        let status = unsafe {
            engine_writer::ffi_engine_writer_stats(handle as u64, stats.as_mut_ptr(), stats.len())
        };
        if status == FfiStatus::Ok.code() {
            let cap = env
                .get_array_length(&out)
                .map_err(|e| jni_err(&env, "out", e))?;
            let k = stats.len().min(usize::try_from(cap).unwrap_or(0));
            env.set_long_array_region(&out, 0, &stats[..k])
                .map_err(|e| jni_err(&env, "out", e))?;
        }
        Ok(status)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_lucenerust_opensearch_NativeBridge_writerClose(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jint {
    engine_writer::ffi_engine_writer_close(handle as u64)
}
