// Shared by the build scripts of the crates that embed fetched data
// (`include!("../../scripts/data-build.rs")`). Makes sure the files of one
// `scripts/fetch-data.py` pack exist before the crate compiles, running the
// script if they do not. The files are never tracked; see that script.

use std::path::Path;
use std::process::Command;

/// `pack` is a `fetch-data.py` pack name, `files` its files relative to the
/// crate directory.
fn ensure_data(pack: &str, files: &[&str]) {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).join("../..");
    let script = root.join("scripts/fetch-data.py");
    println!("cargo:rerun-if-changed={}", script.display());
    for f in files {
        println!("cargo:rerun-if-changed={}", Path::new(&manifest).join(f).display());
    }
    if files.iter().all(|f| Path::new(&manifest).join(f).exists()) {
        return;
    }
    let mut last = None;
    for python in ["python3", "python"] {
        match Command::new(python).arg(&script).args(["--pack", pack]).status() {
            Ok(status) if status.success() => return,
            Ok(status) => {
                last = Some(format!("{python} {} --pack {pack} exited with {status}", script.display()));
                break;
            }
            Err(e) => last = Some(format!("cannot run {python}: {e}")),
        }
    }
    panic!(
        "the {pack} data files are not in the tree and fetching them failed ({}). \
         Run `scripts/fetch-data.py --pack {pack}` (Python 3; the ICU pack also needs a JDK), \
         or set LUCENE_RUST_DATA_CACHE to a directory holding the pinned jars when offline.",
        last.unwrap_or_default()
    );
}
