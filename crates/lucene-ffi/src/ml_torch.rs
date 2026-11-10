//! libtorch for the plugin's TorchScript models: the shared libraries,
//! loaded once per process the way DJL's PyTorch engine loads them, and the
//! C ABI shim (`torch/lucene_torch.cpp`) that runs a module.
//!
//! libtorch exports only C++ symbols, so the shim, built against its headers,
//! is what Rust calls. The shim comes in one build per C++ string ABI
//! (libtorch ships both); the first one whose symbols the loaded libtorch
//! resolves is used.

use std::ffi::{c_char, c_void, CStr};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LOCAL, RTLD_NOW};
use lucene_ml::torchscript::{Data, Input, TorchOutput, TorchScript};
use lucene_ml::{MlError, Result};

/// The shim's view of one output (`struct lucene_torch_output`).
#[repr(C)]
struct RawOutput {
    name: *const c_char,
    name_len: usize,
    group: u32,
    dtype: i32,
    rank: i32,
    shape: *const i64,
    data: *const c_void,
    numel: i64,
}

type LastError = unsafe extern "C" fn() -> *const c_char;
type SetThreads = unsafe extern "C" fn(i32) -> i32;
type Load = unsafe extern "C" fn(*const u8, usize) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Forward = unsafe extern "C" fn(
    *mut c_void,
    i32,
    *const *const u8,
    *const usize,
    *const *const i64,
    i64,
) -> *mut c_void;
type Outputs = unsafe extern "C" fn(*const c_void) -> i32;
type ReadOutput = unsafe extern "C" fn(*const c_void, i32, *mut RawOutput) -> i32;

/// The shim's entry points. The libraries stay loaded for the life of the
/// process: modules and results point into them.
struct Api {
    last_error: LastError,
    load: Load,
    free: Free,
    forward: Forward,
    outputs: Outputs,
    output: ReadOutput,
    result_free: Free,
    _libraries: Vec<Library>,
}

static API: OnceLock<Api> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());

fn runtime_error(e: impl std::fmt::Display) -> MlError {
    MlError::Runtime(e.to_string())
}

/// The libraries of the libtorch directory `dir`, in the order DJL's
/// `LibUtils.loadLibTorch` loads them: the helpers (OpenMP, libc10, ...),
/// then `libfbgemm`, `libtorch_cpu` and `libtorch`. GPU and Python
/// libraries are left out.
fn load_order(dir: &Path) -> Result<Vec<PathBuf>> {
    const DEFERRED: [&str; 3] = ["libfbgemm.so", "libtorch_cpu.so", "libtorch.so"];
    let mut helpers: Vec<(u8, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| MlError::Runtime(format!("libtorch directory {}: {e}", dir.display())))?;
    for entry in entries {
        let path = entry.map_err(runtime_error)?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let shared = name.ends_with(".so")
            || name
                .split_once(".so.")
                .is_some_and(|(_, v)| v.split('.').all(|d| d.parse::<u32>().is_ok()));
        if !shared || !path.is_file() {
            continue;
        }
        let rank = if name.starts_with("libarm_compute-") || name.starts_with("libopenblasp") {
            2
        } else if name.starts_with("libarm_compute_") {
            3
        } else if DEFERRED.contains(&name)
            || name.contains("djl_torch")
            || name.contains("torch_")
            || name.contains("caffe2_")
            || name.starts_with("cudnn")
            || name.contains("cuda")
        {
            continue;
        } else {
            1
        };
        helpers.push((rank, path));
    }
    helpers.sort();
    let mut order: Vec<PathBuf> = helpers.into_iter().map(|(_, p)| p).collect();
    for name in DEFERRED {
        let path = dir.join(name);
        if path.is_file() {
            order.push(path);
        }
    }
    Ok(order)
}

/// Refuses a directory without libtorch in it.
fn check_dir(dir: &Path) -> Result<()> {
    if dir.join("libtorch.so").is_file() {
        Ok(())
    } else {
        Err(MlError::Runtime(format!(
            "no libtorch.so in {}",
            dir.display()
        )))
    }
}

/// Loads libtorch from `dir` and the first of `shims` it can run, with
/// `threads` intra-op threads shared by every module (0: libtorch's
/// default, one per physical core). Later calls return without effect.
pub(crate) fn init(dir: &Path, shims: &[&Path], threads: u32) -> Result<()> {
    let _serial = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if API.get().is_some() {
        return Ok(());
    }
    check_dir(dir)?;
    // libtorch's OpenMP threads spin between parallel regions by default;
    // under concurrent calls that spinning takes the cores the other calls
    // need. Waiting passively keeps a node's throughput under load at
    // stock's while a call running alone still uses every core. libgomp reads
    // the variable when it loads, below; an operator's own choice stays.
    // Set once, under `INIT`, before any OpenMP runs. Other threads' native
    // code may read the environment meanwhile: glibc's setenv builds the new
    // array before it publishes it and does not free one it did not
    // allocate, and nothing else in the node sets variables, so a concurrent
    // getenv sees the old array or the new one.
    if std::env::var_os("OMP_WAIT_POLICY").is_none() {
        std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");
    }
    // libtorch's libraries register ops and exit handlers as they load;
    // unloading them is unsupported, so they stay loaded even when a later
    // step fails, and a retry finds them already in the process.
    let mut libraries = ManuallyDrop::new(Vec::new());
    // The system's C++ library first, when there is one: libtorch's
    // pre-C++11-ABI builds bundle a libstdc++ from GCC 4.8, which the shim,
    // built with a newer compiler, cannot run on, while the system's serves
    // both. Without one, the bundled copy is loaded with the rest.
    // SAFETY: the system C++ runtime; loading it runs its initializers.
    let system_cxx = unsafe { Library::open(Some("libstdc++.so.6"), RTLD_NOW | RTLD_GLOBAL) };
    let has_system_cxx = system_cxx.is_ok();
    libraries.extend(system_cxx.ok());
    for path in load_order(dir)? {
        let bundled_cxx = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("libstdc++"));
        if bundled_cxx && has_system_cxx {
            continue;
        }
        // Global, so the shim resolves libtorch's symbols.
        let flags = RTLD_NOW | RTLD_GLOBAL;
        // SAFETY: loading runs the library's initializers; these are
        // libtorch's own libraries from the directory the operator chose.
        let library = unsafe { Library::open(Some(&path), flags) }
            .map_err(|e| MlError::Runtime(format!("loading {}: {e}", path.display())))?;
        libraries.push(library);
    }
    let mut failures = Vec::new();
    let mut shim = None;
    for path in shims {
        // SAFETY: the shim is the plugin's own library; RTLD_NOW refuses it
        // here when the loaded libtorch lacks a symbol it needs (the other
        // C++ string ABI).
        match unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) } {
            Ok(library) => {
                shim = Some(library);
                break;
            }
            Err(e) => failures.push(e.to_string()),
        }
    }
    libraries.push(shim.ok_or_else(|| {
        MlError::Runtime(format!(
            "no TorchScript shim loads against {}: {}",
            dir.display(),
            failures.join("; ")
        ))
    })?);
    let shim = libraries.last().expect("the shim was just pushed");
    // SAFETY: each symbol is looked up under the name and C signature
    // torch/lucene_torch.cpp defines it with.
    let api = unsafe {
        let set_threads = *shim
            .get::<SetThreads>(b"lucene_torch_set_threads\0")
            .map_err(runtime_error)?;
        let api = Api {
            last_error: *shim
                .get::<LastError>(b"lucene_torch_last_error\0")
                .map_err(runtime_error)?,
            load: *shim
                .get::<Load>(b"lucene_torch_load\0")
                .map_err(runtime_error)?,
            free: *shim
                .get::<Free>(b"lucene_torch_free\0")
                .map_err(runtime_error)?,
            forward: *shim
                .get::<Forward>(b"lucene_torch_forward\0")
                .map_err(runtime_error)?,
            outputs: *shim
                .get::<Outputs>(b"lucene_torch_outputs\0")
                .map_err(runtime_error)?,
            output: *shim
                .get::<ReadOutput>(b"lucene_torch_output\0")
                .map_err(runtime_error)?,
            result_free: *shim
                .get::<Free>(b"lucene_torch_result_free\0")
                .map_err(runtime_error)?,
            _libraries: Vec::new(),
        };
        let threads = i32::try_from(threads).map_err(runtime_error)?;
        if set_threads(threads) != 0 {
            return Err(MlError::Runtime(shim_error(api.last_error)));
        }
        api
    };
    let _ = API.set(Api {
        _libraries: ManuallyDrop::into_inner(libraries),
        ..api
    });
    Ok(())
}

fn api() -> Result<&'static Api> {
    API.get()
        .ok_or_else(|| MlError::Runtime("libtorch is not loaded".into()))
}

/// The calling thread's last shim error.
fn shim_error(last_error: LastError) -> String {
    // SAFETY: the shim returns its thread-local message, NUL-terminated and
    // valid until its next failing call on this thread.
    unsafe { CStr::from_ptr(last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// A loaded TorchScript module.
pub(crate) struct Module {
    api: &'static Api,
    module: NonNull<c_void>,
}

// SAFETY: a TorchScript module runs `forward` from several threads at once
// (its interpreter keeps per-call state), and is freed once, on drop.
unsafe impl Send for Module {}
// SAFETY: as above; `forward` takes the module by shared reference.
unsafe impl Sync for Module {}

impl Module {
    /// Loads the TorchScript file at `path`.
    pub(crate) fn load(path: &Path) -> Result<Module> {
        let api = api()?;
        let path = path
            .to_str()
            .ok_or_else(|| MlError::Model(format!("{} is not UTF-8", path.display())))?;
        // SAFETY: the shim reads `path.len()` bytes from `path`.
        let module = unsafe { (api.load)(path.as_ptr(), path.len()) };
        let module =
            NonNull::new(module).ok_or_else(|| MlError::Model(shim_error(api.last_error)))?;
        Ok(Module { api, module })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: `module` came from `load` and is freed only here.
        unsafe { (self.api.free)(self.module.as_ptr()) };
    }
}

/// A forward call's result, freed on drop.
struct Outcome<'a> {
    api: &'a Api,
    result: NonNull<c_void>,
}

impl Drop for Outcome<'_> {
    fn drop(&mut self) {
        // SAFETY: `result` came from `forward` and is freed only here.
        unsafe { (self.api.result_free)(self.result.as_ptr()) };
    }
}

impl TorchScript for Module {
    fn forward(&self, inputs: &[Input<'_>]) -> Result<Vec<TorchOutput>> {
        let len = inputs.first().map_or(0, |i| i.values.len());
        if inputs.iter().any(|i| i.values.len() != len) {
            return Err(MlError::Model("the input arrays differ in length".into()));
        }
        let names: Vec<*const u8> = inputs.iter().map(|i| i.name.as_ptr()).collect();
        let name_lens: Vec<usize> = inputs.iter().map(|i| i.name.len()).collect();
        let arrays: Vec<*const i64> = inputs.iter().map(|i| i.values.as_ptr()).collect();
        let n = i32::try_from(inputs.len()).map_err(runtime_error)?;
        let len = i64::try_from(len).map_err(runtime_error)?;
        let api = self.api;
        // SAFETY: `n` names, lengths and arrays of `len` values each, all
        // alive for the call; the shim copies the arrays.
        let result = unsafe {
            (api.forward)(
                self.module.as_ptr(),
                n,
                names.as_ptr(),
                name_lens.as_ptr(),
                arrays.as_ptr(),
                len,
            )
        };
        let result =
            NonNull::new(result).ok_or_else(|| MlError::Model(shim_error(api.last_error)))?;
        let outcome = Outcome { api, result };
        // SAFETY: `result` is a live result.
        let count = unsafe { (api.outputs)(outcome.result.as_ptr()) };
        (0..count).map(|i| read_output(&outcome, i)).collect()
    }
}

fn read_output(outcome: &Outcome<'_>, i: i32) -> Result<TorchOutput> {
    let mut raw = RawOutput {
        name: std::ptr::null(),
        name_len: 0,
        group: 0,
        dtype: 0,
        rank: 0,
        shape: std::ptr::null(),
        data: std::ptr::null(),
        numel: 0,
    };
    // SAFETY: `i` is below the result's output count; `raw` is writable.
    if unsafe { (outcome.api.output)(outcome.result.as_ptr(), i, &mut raw) } != 0 {
        return Err(MlError::Model(format!("no model output {i}")));
    }
    let rank = usize::try_from(raw.rank).map_err(runtime_error)?;
    let numel = usize::try_from(raw.numel).map_err(runtime_error)?;
    // SAFETY: the shim's pointers stay valid until the result is freed,
    // which `outcome` outlives here: `name_len` bytes of name, `rank`
    // dimensions, and `numel` contiguous values of the type `dtype` names.
    let (name, shape, data) = unsafe {
        let name = slice(raw.name.cast::<u8>(), raw.name_len);
        let shape = slice(raw.shape, rank);
        let data = match raw.dtype {
            0 => Data::F32(slice(raw.data.cast::<f32>(), numel).to_vec()),
            1 => Data::I64(slice(raw.data.cast::<i64>(), numel).to_vec()),
            other => return Err(MlError::Model(format!("unknown output data type {other}"))),
        };
        (name, shape, data)
    };
    Ok(TorchOutput {
        name: String::from_utf8_lossy(name).into_owned(),
        group: raw.group,
        shape: shape
            .iter()
            .map(|&d| usize::try_from(d).map_err(runtime_error))
            .collect::<Result<_>>()?,
        data,
    })
}

/// `len` values at `ptr`; an empty slice when `len` is 0, whatever `ptr` is.
///
/// # Safety
/// When `len` is not 0, `ptr` must point to `len` initialized values that
/// outlive the returned slice.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    // SAFETY: caller contract.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_libraries_as_djl_loads_them() {
        let dir = lucene_util::test_support::TempDir::new("ml-torch-order");
        for name in [
            "libtorch.so",
            "libtorch_cpu.so",
            "libc10.so",
            "libgomp-98b21ff3.so.1",
            "libtorch_python.so",
            "0.31.1-libdjl_torch.so",
            "libopenblasp-r0.so",
            "libarm_compute_core.so",
            "notes.txt",
            "libfoo.so.x",
        ] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        let names: Vec<String> = load_order(dir.path())
            .unwrap()
            .into_iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "libc10.so",
                "libgomp-98b21ff3.so.1",
                "libopenblasp-r0.so",
                "libarm_compute_core.so",
                "libtorch_cpu.so",
                "libtorch.so"
            ]
        );
        assert!(load_order(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn refuses_a_directory_without_libtorch() {
        let dir = lucene_util::test_support::TempDir::new("ml-torch-empty");
        assert!(matches!(check_dir(dir.path()), Err(MlError::Runtime(_))));
        std::fs::write(dir.path().join("libtorch.so"), b"").unwrap();
        assert!(check_dir(dir.path()).is_ok());
    }

    /// The TorchScript fixtures carry the weights of the ONNX ones, so they
    /// give the vectors stock recorded for those (fixtures/ml/expected.json),
    /// when libtorch and the shims are available (`LUCENE_ML_LIBTORCH`,
    /// `LUCENE_ML_TORCH_SHIMS`).
    #[test]
    fn torchscript_models_give_stock_vectors() {
        use lucene_ml::sparse_encoding::SparseEncodingModel;
        use lucene_ml::text_embedding::{Pooling, TorchTextEmbeddingModel};
        use lucene_ml::text_similarity::TextSimilarityModel;
        use serde_json::Value;

        let (Some(dir), Some(shims)) = (
            std::env::var_os("LUCENE_ML_LIBTORCH"),
            std::env::var_os("LUCENE_ML_TORCH_SHIMS"),
        ) else {
            eprintln!("LUCENE_ML_LIBTORCH or LUCENE_ML_TORCH_SHIMS is not set: skipping");
            return;
        };
        let shims = shims.into_string().unwrap();
        let shims: Vec<&Path> = shims.split(':').map(Path::new).collect();
        init(Path::new(&dir), &shims, 2).unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ml");
        let tokenizer = std::fs::read(fixtures.join("tokenizer.json")).unwrap();
        let expected: Value =
            serde_json::from_str(&std::fs::read_to_string(fixtures.join("expected.json")).unwrap())
                .unwrap();
        let texts: Vec<&str> = expected["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        let module = |name: &str| Box::new(Module::load(&fixtures.join(name)).unwrap());
        let close = |g: f32, w: &Value| {
            let w = w.as_f64().unwrap() as f32;
            (g - w).abs() <= 1e-4 * w.abs().max(1.0)
        };
        let mut checked = 0;
        for case in expected["cases"].as_array().unwrap() {
            if case["model"] != "tiny-bert" {
                continue;
            }
            let pooling = Pooling::from_name(case["pooling"].as_str().unwrap()).unwrap();
            let normalize = case["normalize"].as_bool().unwrap();
            let model = TorchTextEmbeddingModel::hugging_face(
                module("tiny-hf.pt"),
                &tokenizer,
                pooling,
                normalize,
            )
            .unwrap();
            for (text, want) in texts.iter().zip(case["outputs"].as_array().unwrap()) {
                let got = model.embed(text);
                if want.get("error").is_some() {
                    assert!(got.is_err(), "{pooling:?} {text:?}: stock refused it");
                    continue;
                }
                let Data::F32(got) = &got.unwrap()[0].data else {
                    panic!("not FLOAT32");
                };
                let want = want["data"].as_array().unwrap();
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(want) {
                    assert!(close(*g, w), "{pooling:?} {text:?}: {g} vs {w}");
                }
                checked += 1;
            }
        }
        // The sentence-transformers trace shares tiny-bert's weights: its
        // sentence_embedding is stock's unnormalized mean pooling.
        let st = TorchTextEmbeddingModel::sentence_transformers(module("tiny-st.pt"), &tokenizer)
            .unwrap();
        let mean = expected["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["model"] == "tiny-bert" && c["pooling"] == "mean" && c["normalize"] == false
            })
            .unwrap();
        for (text, want) in texts.iter().zip(mean["outputs"].as_array().unwrap()) {
            let got = st.embed(text);
            if want.get("error").is_some() {
                assert!(got.is_err(), "{text:?}: stock refused it");
                continue;
            }
            let got = got.unwrap();
            let mut names: Vec<&str> = got.iter().map(|o| o.name.as_str()).collect();
            names.sort_unstable();
            assert_eq!(
                names,
                [
                    "attention_mask",
                    "input_ids",
                    "sentence_embedding",
                    "token_embeddings"
                ]
            );
            assert!(got.iter().all(|o| o.group == 1));
            let ids = got.iter().find(|o| o.name == "input_ids").unwrap();
            assert_eq!(ids.data.type_name(), "INT64");
            let embedding = got.iter().find(|o| o.name == "sentence_embedding").unwrap();
            let Data::F32(values) = &embedding.data else {
                panic!("not FLOAT32");
            };
            let want = want["data"].as_array().unwrap();
            assert_eq!(values.len(), want.len());
            for (g, w) in values.iter().zip(want) {
                assert!(close(*g, w), "sentence_transformers {text:?}: {g} vs {w}");
            }
            checked += 1;
        }
        let sparse =
            SparseEncodingModel::torchscript(module("tiny-sparse-ts.pt"), &tokenizer).unwrap();
        for (text, want) in texts.iter().zip(
            expected["sparse_encoding"]["outputs"]["WORD"]
                .as_array()
                .unwrap(),
        ) {
            let got = sparse.encode(text, false);
            if want.get("error").is_some() {
                assert!(got.is_err(), "{text:?}: stock refused it");
                continue;
            }
            let got = got.unwrap();
            assert_eq!(got[0].name, "output");
            let want = want[0]["dataAsMap"]["response"][0].as_object().unwrap();
            assert_eq!(got[0].weights.len(), want.len(), "{text:?}");
            for w in &got[0].weights {
                assert!(close(w.weight, &want[&w.token]), "{text:?} {}", w.token);
            }
            checked += 1;
        }
        let cross =
            TextSimilarityModel::torchscript(module("tiny-cross-ts.pt"), &tokenizer).unwrap();
        let case = &expected["text_similarity"];
        let query = case["query"].as_str().unwrap();
        for (text, want) in texts.iter().zip(case["outputs"].as_array().unwrap()) {
            let got = cross.score(query, text).unwrap();
            assert_eq!(got[0].shape, vec![1]);
            assert!(close(got[0].data[0], &want[0]["data"][0]), "{text:?}");
            checked += 1;
        }
        assert!(checked >= 50, "only {checked} vectors compared");
    }
}
