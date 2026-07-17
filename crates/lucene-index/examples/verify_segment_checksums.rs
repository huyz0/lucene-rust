//! Standalone, fast checksum-only verifier CLI (task #217): a thin wrapper
//! around [`lucene_index::checksum_verify::verify_directory`], the lighter,
//! cheaper sibling of `check_index.rs`'s much deeper `CheckIndex`-equivalent
//! -- see that module's doc comment for exactly how the two differ in
//! scope (this tool only recomputes each file's CRC-32 footer checksum; it
//! does no structural cross-checks at all).
//!
//! Opens the directory given as the first CLI argument, finds its latest
//! commit, and checksum-verifies every file every segment declares.
//! Prints one pass/fail line per file plus an overall summary, and exits
//! with a non-zero status if any file fails (or if the commit itself can't
//! be read) -- a real CLI contract, not just a library call, so this is
//! usable directly in a pre-flight/CI script (e.g. `... && echo clean`).
//!
//! Run: `cargo run -p lucene-index --example verify_segment_checksums -- <dir>`

use lucene_index::checksum_verify::verify_directory;
use lucene_store::FsDirectory;

fn main() {
    let dir_arg = std::env::args()
        .nth(1)
        .expect("usage: verify_segment_checksums <directory>");
    let dir = FsDirectory::open(&dir_arg);

    let report = match verify_directory(&dir) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("error: could not read latest commit in {dir_arg}: {e}");
            std::process::exit(2);
        }
    };

    for file in &report.files {
        let status = if file.passed { "PASS" } else { "FAIL" };
        println!(
            "[{status}] {} ({}): {}",
            file.file_name, file.segment_name, file.message
        );
    }

    let total = report.total();
    let failed = report.failed_count();
    println!("---");
    println!("checked {total} file(s), {failed} failed");

    if failed > 0 {
        std::process::exit(1);
    }
}
