//! `BaseTokenStreamFactoryTestCase`: fully initialized analysis factories
//! by SPI name and key/value arguments, for factory tests.
//!
//! Java's `ClasspathResourceLoader(getClass())` is the analysis crate's
//! [`ClasspathResourceLoader`] (the bundled resources); a test needing
//! others passes its own loader. Java's reflective constructor call
//! unwraps an `IllegalArgumentException` so a test can expect it; the
//! port's constructors return the [`FactoryError`] directly.

use lucene_analysis::factory::spi::{
    char_filter_for_name, token_filter_for_name, tokenizer_for_name,
};
use lucene_analysis::factory::{
    AnalysisFactory, CharFilterFactory, ClasspathResourceLoader, FactoryError, JavaArgs,
    ResourceLoader, TokenFilterFactory, TokenizerFactory,
};
use lucene_util::version::Version;

/// `analysisFactory`'s argument map: the pairs, plus `luceneMatchVersion`
/// when a version is given.
///
/// # Panics
/// On an odd-length `keys_and_values` (Java's `IllegalArgumentException`
/// "invalid keysAndValues map") or a duplicate key (Java's assertion).
fn args(match_version: Option<Version>, keys_and_values: &[&str]) -> JavaArgs {
    assert!(
        keys_and_values.len().is_multiple_of(2),
        "invalid keysAndValues map"
    );
    let mut args = JavaArgs::new();
    for kv in keys_and_values.chunks(2) {
        let previous = args.put(kv[0], kv[1]);
        assert!(previous.is_none(), "duplicate values for key: {}", kv[0]);
    }
    if let Some(v) = match_version {
        let previous = args.put("luceneMatchVersion", &v.to_string());
        assert!(
            previous.is_none(),
            "duplicate values for key: luceneMatchVersion"
        );
    }
    args
}

fn inform<F: AnalysisFactory + ?Sized>(
    factory: &mut F,
    loader: &dyn ResourceLoader,
) -> Result<(), FactoryError> {
    if factory.is_resource_loader_aware() {
        factory.inform(loader)?;
    }
    Ok(())
}

/// `tokenizerFactory(name, keysAndValues...)`: [`Version::LATEST`] and the
/// classpath loader.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn tokenizer_factory(
    name: &str,
    keys_and_values: &[&str],
) -> Result<Box<dyn TokenizerFactory>, FactoryError> {
    tokenizer_factory_with(
        name,
        Some(Version::LATEST),
        &ClasspathResourceLoader,
        keys_and_values,
    )
}

/// `tokenizerFactory(name, matchVersion, loader, keysAndValues...)`.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn tokenizer_factory_with(
    name: &str,
    match_version: Option<Version>,
    loader: &dyn ResourceLoader,
    keys_and_values: &[&str],
) -> Result<Box<dyn TokenizerFactory>, FactoryError> {
    let mut f = tokenizer_for_name(name, &mut args(match_version, keys_and_values))?;
    inform(&mut *f, loader)?;
    Ok(f)
}

/// `tokenFilterFactory(name, keysAndValues...)`.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn token_filter_factory(
    name: &str,
    keys_and_values: &[&str],
) -> Result<Box<dyn TokenFilterFactory>, FactoryError> {
    token_filter_factory_with(
        name,
        Some(Version::LATEST),
        &ClasspathResourceLoader,
        keys_and_values,
    )
}

/// `tokenFilterFactory(name, matchVersion, loader, keysAndValues...)`.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn token_filter_factory_with(
    name: &str,
    match_version: Option<Version>,
    loader: &dyn ResourceLoader,
    keys_and_values: &[&str],
) -> Result<Box<dyn TokenFilterFactory>, FactoryError> {
    let mut f = token_filter_for_name(name, &mut args(match_version, keys_and_values))?;
    inform(&mut *f, loader)?;
    Ok(f)
}

/// `charFilterFactory(name, keysAndValues...)`.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn char_filter_factory(
    name: &str,
    keys_and_values: &[&str],
) -> Result<Box<dyn CharFilterFactory>, FactoryError> {
    char_filter_factory_with(
        name,
        Some(Version::LATEST),
        &ClasspathResourceLoader,
        keys_and_values,
    )
}

/// `charFilterFactory(name, matchVersion, loader, keysAndValues...)`.
///
/// # Errors
/// What the factory's constructor or `inform` throws.
pub fn char_filter_factory_with(
    name: &str,
    match_version: Option<Version>,
    loader: &dyn ResourceLoader,
    keys_and_values: &[&str],
) -> Result<Box<dyn CharFilterFactory>, FactoryError> {
    let mut f = char_filter_for_name(name, &mut args(match_version, keys_and_values))?;
    inform(&mut *f, loader)?;
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::base_token_stream::{assert_token_stream_contents, Expected};
    use lucene_analysis::StrReader;

    #[test]
    fn the_javadoc_example_chain() {
        let reader = char_filter_factory("htmlStrip", &[])
            .unwrap()
            .create(Box::new(StrReader::new("<b>Some</b> Text to Analyze")));
        let mut stream = tokenizer_factory("standard", &[])
            .unwrap()
            .create()
            .unwrap();
        stream.as_tokenizer().unwrap().set_reader(reader).unwrap();
        let stream = token_filter_factory("lowercase", &[])
            .unwrap()
            .create(stream)
            .unwrap();
        let mut stream = token_filter_factory("asciiFolding", &[])
            .unwrap()
            .create(stream)
            .unwrap();
        assert_token_stream_contents(
            &mut stream,
            &Expected::terms(&["some", "text", "to", "analyze"]),
        )
        .unwrap();
    }

    #[test]
    fn resource_aware_factories_are_informed() {
        // a missing resource fails in inform, so inform ran
        assert!(token_filter_factory("stop", &["words", "no-such-file.txt"]).is_err());
        let f = token_filter_factory("stop", &[]).unwrap();
        assert!(f.is_resource_loader_aware());
    }

    #[test]
    fn bad_arguments() {
        assert!(tokenizer_factory("standard", &["bogusArg", "x"]).is_err());
        assert!(tokenizer_factory_with("standard", None, &ClasspathResourceLoader, &[]).is_ok());
    }

    #[test]
    #[should_panic(expected = "invalid keysAndValues map")]
    fn odd_arguments_panic() {
        let _ = tokenizer_factory("standard", &["a"]);
    }

    #[test]
    #[should_panic(expected = "duplicate values for key: luceneMatchVersion")]
    fn a_duplicate_version_panics() {
        let _ = tokenizer_factory("standard", &["luceneMatchVersion", "10.0.0"]);
    }

    #[test]
    #[should_panic(expected = "duplicate values for key: a")]
    fn a_duplicate_key_panics() {
        let _ = tokenizer_factory("standard", &["a", "1", "a", "2"]);
    }
}
