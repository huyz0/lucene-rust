//! The Spark connector's entry points: [`crate::arrow_source`] over JNI, for
//! `org.accelerated.lucene.spark.Native`. A JVM starts a stream by passing
//! the address of an `ArrowArrayStream` it allocated with Arrow Java; the
//! library fills it, and Arrow Java's `Data.importArrayStream` reads it.
//!
//! A failure is thrown as `java.io.IOException` with the call's message.

use std::panic::AssertUnwindSafe;

use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use lucene_sql::arrow::ffi_stream::FFI_ArrowArrayStream;

use crate::arrow_source::{close_table, open_table, start_scan, table_info};
use crate::error::{guard, last_error, FfiStatus};

fn string(env: &mut JNIEnv, s: &JString) -> Result<String, FfiStatus> {
    env.get_string(s)
        .map(Into::into)
        .map_err(|_| FfiStatus::InvalidArgument)
}

/// Runs `f` as an exported entry point would, throwing its failure.
fn call<T: Default>(env: &mut JNIEnv, f: impl FnOnce(&mut JNIEnv) -> Result<T, FfiStatus>) -> T {
    let mut out = None;
    // The closure holds the JNI environment and `out`, which a panic leaves
    // unread: the status then reports it and `out` stays empty.
    let status = guard(AssertUnwindSafe(|| {
        out = Some(f(env)?);
        Ok(())
    }));
    match (status, out) {
        (0, Some(v)) => v,
        _ => {
            let message = last_error();
            let _ = env.throw_new(
                "java/io/IOException",
                if message.is_empty() {
                    format!("native call failed (status {status})")
                } else {
                    message
                },
            );
            T::default()
        }
    }
}

/// `long open(String path, String spec)`.
#[no_mangle]
pub extern "system" fn Java_org_accelerated_lucene_spark_Native_open(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
    spec: JString,
) -> jlong {
    call(&mut env, |env| {
        let path = string(env, &path)?;
        let spec = string(env, &spec)?;
        Ok(open_table(&path, &spec)? as jlong)
    })
}

/// `void close(long table)`.
#[no_mangle]
pub extern "system" fn Java_org_accelerated_lucene_spark_Native_close(
    mut env: JNIEnv,
    _class: JClass,
    table: jlong,
) {
    call(&mut env, |_| close_table(table as u64));
}

/// `String info(long table)`.
#[no_mangle]
pub extern "system" fn Java_org_accelerated_lucene_spark_Native_info(
    mut env: JNIEnv,
    _class: JClass,
    table: jlong,
) -> jstring {
    call(&mut env, |env| {
        let json = table_info(table as u64)?;
        env.new_string(json)
            .map(JString::into_raw)
            .map_err(|_| FfiStatus::InvalidArgument)
    })
}

/// `void scan(long table, String request, int partition, int partitions, long stream)`:
/// fills the `ArrowArrayStream` at address `stream`.
#[no_mangle]
pub extern "system" fn Java_org_accelerated_lucene_spark_Native_scan(
    mut env: JNIEnv,
    _class: JClass,
    table: jlong,
    request: JString,
    partition: jint,
    partitions: jint,
    stream: jlong,
) {
    call(&mut env, |env| {
        let request = string(env, &request)?;
        let s = start_scan(
            table as u64,
            &request,
            u32::try_from(partition).map_err(|_| FfiStatus::InvalidArgument)?,
            u32::try_from(partitions).map_err(|_| FfiStatus::InvalidArgument)?,
        )?;
        let out = stream as *mut FFI_ArrowArrayStream;
        if out.is_null() {
            return Err(FfiStatus::NullPointer);
        }
        // SAFETY: the JVM passes the address of an `ArrowArrayStream` it
        // allocated (Arrow Java `ArrowArrayStream.allocateNew`), which is
        // valid for one write and which the stream then owns.
        unsafe { std::ptr::write(out, s) };
        Ok(())
    })
}
