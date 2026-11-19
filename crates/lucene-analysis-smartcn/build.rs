//! Fetches the crate's dictionary files (`scripts/fetch-data.py --pack smartcn`)
//! when they are not in the tree.

include!("../../scripts/data-build.rs");

fn main() {
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_DATA").is_none() {
        return;
    }
    ensure_data(
        "smartcn",
        &[
            "src/resources/coredict.mem.z",
            "src/resources/bigramdict.mem.z",
        ],
    );
}
