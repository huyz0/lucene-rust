//! Lucene analysis for the JVM plugin: an analyzer described by the Java side
//! (its char filters, tokenizer and token filters, by their SPI names and
//! factory arguments, recovered from the Lucene objects OpenSearch built) is
//! built here with `lucene_analysis::factory::CustomAnalyzer` and run over a
//! field's text.
//!
//! The Java side (`AnalyzerSpec`) writes the spec; this module builds each
//! distinct spec once and keeps it, so a spec is a cache key, not a handle
//! with a lifecycle. Java never sends a spec it could not describe exactly,
//! and checks every new spec against its own analyzer on a probe text before
//! trusting it.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use lucene_analysis::factory::{CustomAnalyzer, FactoryError, JavaArgs, ResourceLoader};
use lucene_analysis::TokenStream;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::Cursor;

/// The spec layout's version byte.
pub const SPEC_VERSION: u8 = 1;

/// At most this many distinct specs are kept; one more clears the cache
/// (an index with this many analyzers is not a steady state worth tuning).
const MAX_SPECS: usize = 4096;

static ANALYZERS: LazyLock<RwLock<HashMap<Vec<u8>, Arc<CustomAnalyzer>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// One component: its SPI name and factory arguments.
struct Component {
    name: String,
    args: JavaArgs,
}

/// The resources a spec carries inline (word lists, mapping rules), served
/// to the factories by name.
struct InlineLoader(HashMap<String, Vec<u8>>);

impl ResourceLoader for InlineLoader {
    fn open_resource(&self, resource: &str) -> Result<Vec<u8>, FactoryError> {
        self.0
            .get(resource)
            .cloned()
            .ok_or_else(|| FactoryError::io(format!("Resource not found: {resource}")))
    }
}

fn invalid(message: String) -> FfiStatus {
    set_last_error(message);
    FfiStatus::InvalidArgument
}

fn utf8<'a>(b: &'a [u8], what: &str) -> Result<&'a str, FfiStatus> {
    std::str::from_utf8(b).map_err(|_| invalid(format!("analysis spec: {what} is not UTF-8")))
}

fn component(c: &mut Cursor<'_>) -> Result<Component, FfiStatus> {
    let name = utf8(c.bytes()?, "a component name")?.to_string();
    let n = c.len()?;
    let mut args = JavaArgs::new();
    for _ in 0..n {
        let k = utf8(c.bytes()?, "an argument name")?;
        let v = utf8(c.bytes()?, "an argument value")?;
        args.put(k, v);
    }
    Ok(Component { name, args })
}

/// Builds the analyzer a spec describes.
///
/// | field | layout |
/// |---|---|
/// | version | `u8`, [`SPEC_VERSION`] |
/// | gaps | `position_increment_gap: i32`, `offset_gap: i32` |
/// | char filters | `n: i32`, then `n` components |
/// | tokenizer | one component |
/// | token filters | `n: i32`, then `n` components |
/// | resources | `n: i32`, then `n` of `name: bytes`, `content: bytes` |
///
/// A component is `name: bytes`, `n: i32`, then `n` of `key: bytes`,
/// `value: bytes`. Trailing bytes are an error.
fn build(spec: &[u8]) -> Result<CustomAnalyzer, FfiStatus> {
    let mut c = Cursor::new(spec);
    let version = c.u8()?;
    if version != SPEC_VERSION {
        return Err(invalid(format!(
            "analysis spec version {version}, expected {SPEC_VERSION}"
        )));
    }
    let pos_gap = c.i32()?;
    let offset_gap = c.i32()?;
    let mut char_filters = Vec::new();
    for _ in 0..c.len()? {
        char_filters.push(component(&mut c)?);
    }
    let tokenizer = component(&mut c)?;
    let mut filters = Vec::new();
    for _ in 0..c.len()? {
        filters.push(component(&mut c)?);
    }
    let mut resources = HashMap::new();
    for _ in 0..c.len()? {
        let name = utf8(c.bytes()?, "a resource name")?.to_string();
        resources.insert(name, c.bytes()?.to_vec());
    }
    if c.pos() != spec.len() {
        return Err(invalid(format!(
            "analysis spec: {} trailing bytes",
            spec.len() - c.pos()
        )));
    }
    let factory = |e: FactoryError| invalid(format!("analysis spec: {e}"));
    let mut b = CustomAnalyzer::builder_with_loader(Box::new(InlineLoader(resources)))
        .with_position_increment_gap(pos_gap)
        .map_err(factory)?
        .with_offset_gap(offset_gap)
        .map_err(factory)?;
    for cf in char_filters {
        b = b.add_char_filter_args(&cf.name, cf.args).map_err(factory)?;
    }
    b = b
        .with_tokenizer_args(&tokenizer.name, tokenizer.args)
        .map_err(factory)?;
    for f in filters {
        b = b.add_token_filter_args(&f.name, f.args).map_err(factory)?;
    }
    b.build().map_err(factory)
}

/// The analyzer for `spec`, built on its first use.
pub(crate) fn analyzer(spec: &[u8]) -> Result<Arc<CustomAnalyzer>, FfiStatus> {
    if let Some(a) = ANALYZERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(spec)
    {
        return Ok(Arc::clone(a));
    }
    let built = Arc::new(build(spec)?);
    let mut map = ANALYZERS.write().unwrap_or_else(|e| e.into_inner());
    if map.len() >= MAX_SPECS {
        map.clear();
    }
    Ok(Arc::clone(map.entry(spec.to_vec()).or_insert(built)))
}

/// How many distinct specs are built and kept.
#[cfg(test)]
pub(crate) fn cached_specs() -> usize {
    ANALYZERS.read().unwrap_or_else(|e| e.into_inner()).len()
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    // A term is at most 32766 bytes in an index, but analysis itself has no
    // such limit; a length past i32 cannot come from a Java string.
    put_i32(out, i32::try_from(b.len()).unwrap_or(i32::MAX));
    out.extend_from_slice(b);
}

/// Runs `spec`'s analyzer over `text` as field `field`, the way a consumer
/// drains a stream (`reset`, every `incrementToken`, `end`, `close`).
///
/// The result is `count: i32`, then per token: `term: bytes`,
/// `position_increment: i32`, `position_length: i32`, `start_offset: i32`,
/// `end_offset: i32`, `type: bytes`, `flags: i32`, `keyword: u8`,
/// `term_frequency: i32`, `payload: i32` length (-1 for none) and its bytes;
/// then what `end()` left: `end_offset: i32`, `position_increment: i32`.
/// Offsets count UTF-16 units, as Java's do.
pub(crate) fn analyze(spec: &[u8], field: &str, text: &str) -> Result<Vec<u8>, FfiStatus> {
    let analyzer = analyzer(spec)?;
    let failed = |e: lucene_analysis::AnalysisError| invalid(format!("analysis: {e}"));
    let mut stream = analyzer
        .analyzer()
        .token_stream(field, text)
        .map_err(failed)?;
    let mut out = Vec::with_capacity(16 + text.len() * 3);
    put_i32(&mut out, 0);
    let mut count: i32 = 0;
    stream.reset().map_err(failed)?;
    while stream.increment_token().map_err(failed)? {
        let a = stream.attributes();
        put_bytes(&mut out, a.term_bytes());
        put_i32(&mut out, a.position_increment());
        put_i32(&mut out, a.position_length());
        put_i32(&mut out, a.start_offset());
        put_i32(&mut out, a.end_offset());
        put_bytes(&mut out, a.token_type().as_bytes());
        put_i32(&mut out, a.flags());
        out.push(u8::from(a.is_keyword()));
        put_i32(&mut out, a.term_frequency());
        match a.payload() {
            Some(p) => put_bytes(&mut out, p),
            None => put_i32(&mut out, -1),
        }
        count = count.saturating_add(1);
    }
    stream.end().map_err(failed)?;
    let a = stream.attributes();
    put_i32(&mut out, a.end_offset());
    put_i32(&mut out, a.position_increment());
    stream.close().map_err(failed)?;
    out[..4].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A spec in [`build`]'s layout.
    pub(crate) fn spec(
        char_filters: &[(&str, &[(&str, &str)])],
        tokenizer: (&str, &[(&str, &str)]),
        filters: &[(&str, &[(&str, &str)])],
        resources: &[(&str, &str)],
    ) -> Vec<u8> {
        fn comp(out: &mut Vec<u8>, (name, args): (&str, &[(&str, &str)])) {
            put_bytes(out, name.as_bytes());
            put_i32(out, args.len() as i32);
            for (k, v) in args {
                put_bytes(out, k.as_bytes());
                put_bytes(out, v.as_bytes());
            }
        }
        let mut out = vec![SPEC_VERSION];
        put_i32(&mut out, 100);
        put_i32(&mut out, 1);
        put_i32(&mut out, char_filters.len() as i32);
        for &c in char_filters {
            comp(&mut out, c);
        }
        comp(&mut out, tokenizer);
        put_i32(&mut out, filters.len() as i32);
        for &f in filters {
            comp(&mut out, f);
        }
        put_i32(&mut out, resources.len() as i32);
        for (n, c) in resources {
            put_bytes(&mut out, n.as_bytes());
            put_bytes(&mut out, c.as_bytes());
        }
        out
    }

    /// The terms, increments and offsets of an [`analyze`] result.
    fn tokens(blob: &[u8]) -> (Vec<(String, i32, i32, i32)>, i32) {
        let mut c = Cursor::new(blob);
        let n = c.len().unwrap();
        let mut out = Vec::new();
        for _ in 0..n {
            let term = String::from_utf8(c.bytes().unwrap().to_vec()).unwrap();
            let inc = c.i32().unwrap();
            let _len = c.i32().unwrap();
            let start = c.i32().unwrap();
            let end = c.i32().unwrap();
            c.bytes().unwrap();
            c.i32().unwrap();
            c.u8().unwrap();
            c.i32().unwrap();
            if c.i32().unwrap() >= 0 {
                unreachable!("no payloads here");
            }
            out.push((term, inc, start, end));
        }
        let final_offset = c.i32().unwrap();
        c.i32().unwrap();
        assert_eq!(c.pos(), blob.len());
        (out, final_offset)
    }

    #[test]
    fn a_spec_builds_the_chain_it_names_and_counts_utf16_offsets() {
        let s = spec(
            &[("htmlStrip", &[])],
            ("standard", &[("maxTokenLength", "255")]),
            &[
                ("lowercase", &[]),
                ("stop", &[("words", "w.txt"), ("ignoreCase", "true")]),
            ],
            &[("w.txt", "the\nof\n")],
        );
        let (t, end) = tokens(&analyze(&s, "body", "<b>The</b> 😀 Rust of Lucene").unwrap());
        // "😀" is two UTF-16 units; "<b>" and "</b>" are stripped but count.
        assert_eq!(
            t,
            vec![
                ("😀".to_string(), 2, 11, 13),
                ("rust".to_string(), 1, 14, 18),
                ("lucene".to_string(), 2, 22, 28),
            ]
        );
        assert_eq!(end, 28);
        // The same spec is built once.
        let before = cached_specs();
        analyze(&s, "body", "again").unwrap();
        assert_eq!(cached_specs(), before);
    }

    #[test]
    fn a_bad_spec_is_an_invalid_argument() {
        let s = spec(&[], ("nope", &[]), &[], &[]);
        assert_eq!(
            analyze(&s, "f", "x").unwrap_err(),
            FfiStatus::InvalidArgument
        );
        let mut s = spec(&[], ("whitespace", &[]), &[], &[]);
        s.push(0);
        assert_eq!(
            analyze(&s, "f", "x").unwrap_err(),
            FfiStatus::InvalidArgument
        );
        let s = spec(
            &[],
            ("whitespace", &[]),
            &[("stop", &[("words", "missing.txt")])],
            &[],
        );
        assert_eq!(
            analyze(&s, "f", "x").unwrap_err(),
            FfiStatus::InvalidArgument
        );
    }
}
