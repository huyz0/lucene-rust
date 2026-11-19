//! Fetches the crate's dictionary files (`scripts/fetch-data.py --pack kuromoji`)
//! when they are not in the tree.

include!("../../scripts/data-build.rs");

fn main() {
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_DATA").is_none() {
        return;
    }
    ensure_data(
        "kuromoji",
        &[
            "src/resources/character_definition.dat.z",
            "src/resources/connection_costs.dat.z",
            "src/resources/token_info_buffer.dat.z",
            "src/resources/token_info_fst.dat.z",
            "src/resources/token_info_pos_dict.dat.z",
            "src/resources/token_info_target_map.dat.z",
            "src/resources/unknown_buffer.dat.z",
            "src/resources/unknown_pos_dict.dat.z",
            "src/resources/unknown_target_map.dat.z",
        ],
    );
}
