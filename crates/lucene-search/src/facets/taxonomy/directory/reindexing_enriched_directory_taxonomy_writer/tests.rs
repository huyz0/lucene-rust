//! Unit tests.

use super::*;
use lucene_store::ByteBuffersDirectory;

#[test]
fn the_writer_debugs_opaquely_and_derefs_to_the_inner_one() {
    let dir = ByteBuffersDirectory::new();
    let w = ReindexingEnrichedDirectoryTaxonomyWriter::new(&dir, None).unwrap();
    assert_eq!(
        format!("{w:?}"),
        "ReindexingEnrichedDirectoryTaxonomyWriter { .. }"
    );
    assert_eq!(w.writer().get_size().unwrap(), w.get_size().unwrap());
}
