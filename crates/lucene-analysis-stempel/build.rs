//! Fetches the crate's dictionary files (`scripts/fetch-data.py --pack stempel`)
//! when they are not in the tree.

include!("../../scripts/data-build.rs");

fn main() {
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_DATA").is_none() {
        return;
    }
    ensure_data("stempel", &["src/resources/stemmer_20000.tbl.z"]);
}
