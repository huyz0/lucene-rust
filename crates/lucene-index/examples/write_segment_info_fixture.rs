//! Writes `segment_info::write`-produced `.si` files plus per-segment
//! manifests to the directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes, and
//! `fixtures/src/VerifySegmentInfo.java` reads the result back through real
//! Lucene's own `Lucene99SegmentInfoFormat.read`.
//!
//! Lives in `lucene-index` (not `lucene-codecs`, unlike the field-infos/
//! stored-fields write-path examples) because `SegmentInfo` itself lives in
//! `lucene-index::segment_info` -- the architecture skill's downward-only
//! dependency graph (`codecs ← index`) means `lucene-codecs` cannot depend on
//! `lucene-index` to reuse its types.
//!
//! Run: `cargo run -p lucene-index --example write_segment_info_fixture -- <dir>`

use lucene_index::segment_info::{self, LuceneVersion, SegmentInfo};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_segment_info_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // _0: compound file, with minVersion, with files/diagnostics/attributes.
    gen(
        &out_dir,
        "_0",
        SegmentInfo {
            id: *b"rustwrittensi000",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: Some(LuceneVersion {
                major: 9,
                minor: 12,
                bugfix: 0,
            }),
            doc_count: 12345,
            is_compound_file: true,
            has_blocks: false,
            diagnostics: vec![
                ("source".to_string(), "flush".to_string()),
                ("lucene.version".to_string(), "10.0.0".to_string()),
                ("os".to_string(), "Linux".to_string()),
            ],
            files: vec![
                "_0.fdt".to_string(),
                "_0.fdx".to_string(),
                "_0_1.doc".to_string(),
            ],
            attributes: vec![(
                "Lucene90StoredFieldsFormat.mode".to_string(),
                "BEST_SPEED".to_string(),
            )],
            index_sort: None,
        },
    );

    // _1: not a compound file, no minVersion, no blocks/hasBlocks, empty
    // diagnostics/files/attributes -- exercises the "everything empty" path.
    gen(
        &out_dir,
        "_1",
        SegmentInfo {
            id: *b"rustwrittensi111",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 7,
            is_compound_file: false,
            has_blocks: true,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        },
    );

    println!("wrote segment-info fixtures to {out_dir}");
}

fn gen(out_dir: &str, segment_name: &str, si: SegmentInfo) {
    let bytes = segment_info::write(&si, "");
    let file_name = format!("{segment_name}.si");
    let dir = FsDirectory::open(out_dir);
    let mut out = dir.create_output(&file_name).unwrap();
    out.write_bytes(&bytes);
    out.close().unwrap();
    dir.sync(&[file_name]).unwrap();

    let mut manifest =
        std::fs::File::create(format!("{out_dir}/{segment_name}.manifest.properties")).unwrap();
    writeln!(manifest, "segment_name={segment_name}").unwrap();
    writeln!(manifest, "id_hex={}", hex(&si.id)).unwrap();
    writeln!(manifest, "version_major={}", si.version.major).unwrap();
    writeln!(manifest, "version_minor={}", si.version.minor).unwrap();
    writeln!(manifest, "version_bugfix={}", si.version.bugfix).unwrap();
    writeln!(
        manifest,
        "has_min_version={}",
        if si.min_version.is_some() { 1 } else { 0 }
    )
    .unwrap();
    if let Some(mv) = si.min_version {
        writeln!(manifest, "min_version_major={}", mv.major).unwrap();
        writeln!(manifest, "min_version_minor={}", mv.minor).unwrap();
        writeln!(manifest, "min_version_bugfix={}", mv.bugfix).unwrap();
    }
    writeln!(manifest, "doc_count={}", si.doc_count).unwrap();
    writeln!(
        manifest,
        "is_compound_file={}",
        if si.is_compound_file { 1 } else { 0 }
    )
    .unwrap();
    writeln!(manifest, "has_blocks={}", if si.has_blocks { 1 } else { 0 }).unwrap();
    writeln!(manifest, "diagnostics={}", join_map(&si.diagnostics)).unwrap();
    writeln!(manifest, "attributes={}", join_map(&si.attributes)).unwrap();
    writeln!(manifest, "files={}", si.files.join(",")).unwrap();

    println!(
        "wrote {segment_name}.si ({} bytes)",
        std::fs::metadata(format!("{out_dir}/{segment_name}.si"))
            .unwrap()
            .len()
    );
}

fn join_map(m: &[(String, String)]) -> String {
    m.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
