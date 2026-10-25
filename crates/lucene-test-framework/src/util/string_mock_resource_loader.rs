//! `org.apache.lucene.tests.util.StringMockResourceLoader`: every resource
//! name opens the same text as a new UTF-8 byte buffer.
//!
//! Rust strings are already UTF-8, so `openResource` copies their bytes rather
//! than transcoding Java's UTF-16. Each owned buffer replaces Java's independent
//! `ByteArrayInputStream`. Inputs are Unicode scalar values; isolated UTF-16
//! surrogates cannot be represented by a Rust `String`. Java's `findClass`
//! reflection follows the static SPI model described by
//! [`lucene_analysis::factory`], outside the resource-only trait.

use lucene_analysis::factory::{FactoryError, ResourceLoader};

/// A string-backed resource loader for analysis factory tests.
pub struct StringMockResourceLoader {
    text: String,
}

impl StringMockResourceLoader {
    /// Java: `StringMockResourceLoader(String)`.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl ResourceLoader for StringMockResourceLoader {
    /// Java: `StringMockResourceLoader.openResource` (the name is ignored).
    fn open_resource(&self, _resource: &str) -> Result<Vec<u8>, FactoryError> {
        Ok(self.text.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::factory::get_word_set;

    #[test]
    fn empty_text_and_arbitrary_names() {
        let loader = StringMockResourceLoader::new("");
        for name in ["", "missing", "../outside", "\0\u{1f600}"] {
            assert_eq!(loader.open_resource(name).unwrap(), b"");
        }
    }

    #[test]
    fn utf8_buffers_are_independent() {
        let text = "\u{feff}caf\u{e9} e\u{301}\0\r\n\u{1f600}\u{10ffff}";
        let loader = StringMockResourceLoader::new(text.to_string());
        let mut first = loader.open_resource("first").unwrap();
        let second = loader.open_resource("second").unwrap();
        assert_eq!(first, text.as_bytes());
        first.fill(0);
        assert_eq!(second, text.as_bytes());
        assert_eq!(loader.open_resource("").unwrap(), text.as_bytes());
    }

    #[test]
    fn works_through_analysis_factory_resource_contract() {
        let loader = StringMockResourceLoader::new("\u{feff}# comment\n  alpha  \r\n\n beta\n");
        let resource: &dyn ResourceLoader = &loader;
        let words = get_word_set(resource, "stopwords.txt", false)
            .unwrap()
            .unwrap();
        assert_eq!(words.len(), 2);
        assert!(words.contains("alpha"));
        assert!(words.contains("beta"));
    }
}
