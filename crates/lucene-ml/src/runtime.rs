//! The process's ONNX Runtime: the shared library, loaded once, and one
//! environment whose intra-op thread pool every model session shares.
//!
//! One pool, not a pool per model: a node with several deployed models would
//! otherwise run several pools of `cores` threads each against the same
//! cores. Spinning is off, so an idle pool does not take CPU from searches.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::environment::{Environment, GlobalThreadPoolOptions};

use crate::{MlError, Result};

static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());

fn runtime_error(e: impl std::fmt::Display) -> MlError {
    MlError::Runtime(e.to_string())
}

/// Loads the ONNX Runtime library at `library` and builds the environment,
/// with `threads` intra-op threads (0: the runtime's default, one per
/// physical core). Later calls return without effect: the first library and
/// pool stay for the life of the process.
pub fn init(library: &Path, threads: usize) -> Result<()> {
    let _serial = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if ENVIRONMENT.get().is_some() {
        return Ok(());
    }
    if !library.is_file() {
        return Err(MlError::Runtime(format!(
            "no ONNX Runtime library at {}",
            library.display()
        )));
    }
    let mut pool = GlobalThreadPoolOptions::default()
        .with_spin_control(false)
        .map_err(runtime_error)?;
    if threads > 0 {
        pool = pool.with_intra_threads(threads).map_err(runtime_error)?;
    }
    let environment = ort::init_from(library)
        .map_err(runtime_error)?
        .with_name("lucene-rust-ml")
        .with_telemetry(false)
        .with_global_thread_pool(pool)
        .build()
        .map_err(runtime_error)?;
    // Only this function sets it, under `INIT`, after the check above.
    let _ = ENVIRONMENT.set(environment);
    Ok(())
}

/// Whether [`init`] has loaded the runtime.
pub fn is_loaded() -> bool {
    ENVIRONMENT.get().is_some()
}

/// The environment [`init`] built.
pub(crate) fn environment() -> Result<&'static Environment> {
    ENVIRONMENT
        .get()
        .ok_or_else(|| MlError::Runtime("the ONNX Runtime library is not loaded".into()))
}
