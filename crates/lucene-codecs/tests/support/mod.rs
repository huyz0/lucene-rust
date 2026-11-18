//! Shared by the doc-values fixture tests: a format-3 `.dvm` keeps sparse
//! fields' `IndexedDISI` in the segment's `.dvp`, which the tests open from
//! the fixture directory and hand to the parsed metadata.
#![allow(dead_code)]

use lucene_codecs::doc_values::{self, DisiBytes, DocValuesMeta};

/// Attaches the `.dvp` of the single segment in `dir` whose per-field suffix
/// is `suffix`, when the metadata's format `version` has one, and returns its
/// bytes.
pub fn attach_dvp(
    meta: &mut DocValuesMeta,
    version: i32,
    dir: &str,
    id: &[u8; 16],
    suffix: &str,
) -> Option<Vec<u8>> {
    if !doc_values::has_disi_file(version) {
        return None;
    }
    let ending = format!("_{suffix}.dvp");
    let mut found = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(&ending));
    let name = found.next().expect("a .dvp beside the .dvm");
    assert!(found.next().is_none(), "one segment's .dvp in {dir}");
    let bytes = std::fs::read(format!("{dir}{name}")).unwrap();
    doc_values::check_disi_header_footer(&bytes, id, suffix, version).unwrap();
    meta.attach_disi(DisiBytes::new(std::sync::Arc::new(bytes.clone())));
    Some(bytes)
}
