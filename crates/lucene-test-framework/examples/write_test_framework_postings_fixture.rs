//! lucene-index's `write_per_field_formats_fixture` with the test
//! framework's terms-dictionary postings formats registered first, so its
//! format argument may name one (`LuceneFixedGap`, `UniformSplitRot13`,
//! `TestBloomFilteredLucenePostings`, ...), for `VerifyPerFieldFormats`.
//!
//! Usage: `write_test_framework_postings_fixture <output-dir> <format>`.

#[path = "../../lucene-index/examples/write_per_field_formats_fixture.rs"]
mod fixture;

fn main() {
    lucene_test_framework::codecs::register_postings_formats();
    fixture::main();
}
