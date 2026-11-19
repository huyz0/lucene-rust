//! Fetches the crate's dictionary files (`scripts/fetch-data.py --pack icu`)
//! when they are not in the tree.

include!("../../scripts/data-build.rs");

fn main() {
    ensure_data(
        "icu",
        &[
            "src/resources/nfc.nrm",
            "src/resources/nfkc.nrm",
            "src/resources/nfkc_cf.nrm",
            "src/resources/nfkc_scf.nrm",
            "src/resources/uts46.nrm",
            "src/resources/ucase.icu",
            "src/resources/word.brk",
            "src/resources/thaidict.dict.z",
            "src/resources/laodict.dict.z",
            "src/resources/khmerdict.dict.z",
            "src/resources/burmesedict.dict.z",
            "src/resources/cjdict.dict.z",
            "src/resources/Default.brk",
            "src/resources/MyanmarSyllable.brk",
            "src/resources/utr30.nrm",
            "src/resources/coll.pack.z",
            "src/resources/uprops.bin.z",
            "src/resources/unames.bin.z",
        ],
    );
}
