//! lucene-index's `write_per_field_formats_fixture` with the test
//! framework's terms-dictionary postings formats registered first, so its
//! format argument may name one (`LuceneFixedGap`, `UniformSplitRot13`,
//! `TestBloomFilteredLucenePostings`, ...), for `VerifyPerFieldFormats`.
//! `MockRandom` writes through `MockRandomPostingsFormat(new Random(0))`, so
//! its flushes and merge draw varied dictionaries.
//!
//! Usage: `write_test_framework_postings_fixture <output-dir> <format>`.

#[path = "../../lucene-index/examples/write_per_field_formats_fixture.rs"]
mod fixture;

use lucene_test_framework::codecs::mockrandom::MockRandomPostingsFormat;

fn main() {
    lucene_test_framework::codecs::register_postings_formats();
    let mut random = lucene_util::java_random::JavaRandom::new(0);
    let mock_random = MockRandomPostingsFormat::new(&mut random).into_dictionary_format();
    lucene_codecs::terms_dictionary::register(mock_random);
    fixture::main();
}
