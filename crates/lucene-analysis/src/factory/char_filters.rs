//! The char filter factories: `htmlStrip`, `mapping`, `cjkWidth`, `persian`,
//! `patternReplace`.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::args::{self, JavaArgs};
use super::loader::{get_lines, java_trim};
use super::{
    analysis_factory, CharFilterFactory, FactoryBase, FactoryClass, FactoryError, JavaException,
    ResourceLoader,
};
use crate::charfilter::{
    HTMLStripCharFilter, MappingCharFilter, NormalizeCharMap, NormalizeCharMapBuilder,
};
use crate::reader::CharReader;
use crate::util::java_regex::JavaMatcher;
use crate::util::JavaPattern;

/// `org.apache.lucene.analysis.charfilter.HTMLStripCharFilterFactory` (`htmlStrip`).
pub struct HTMLStripCharFilterFactory {
    base: FactoryBase,
    escaped_tags: Option<BTreeSet<String>>,
}
analysis_factory!(HTMLStripCharFilterFactory);

impl FactoryClass for HTMLStripCharFilterFactory {
    const NAME: &'static str = "htmlStrip";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.charfilter.HTMLStripCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let escaped_tags = args::get_set(args, "escapedTags");
        args::reject_unknown(args)?;
        Ok(HTMLStripCharFilterFactory { base, escaped_tags })
    }
}

impl CharFilterFactory for HTMLStripCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        match &self.escaped_tags {
            None => Box::new(HTMLStripCharFilter::new(input)),
            Some(tags) => Box::new(HTMLStripCharFilter::with_escaped_tags(input, tags)),
        }
    }
}

/// `MappingCharFilterFactory.p`.
const MAPPING_RULE: &str = "\"(.*)\"\\s*=>\\s*\"(.*)\"\\s*$";

/// The `parseString` of `MappingCharFilterFactory` and the word delimiter
/// factories: backslash escapes (`\\`, `\n`, `\t`, `\r`, `\b`, `\f`,
/// `\uXXXX`; any other escaped char, `\"` among them, stands for
/// itself), over UTF-16 units.
pub(crate) fn parse_escaped(s: &str) -> Result<Vec<u16>, FactoryError> {
    let units: Vec<u16> = s.encode_utf16().collect();
    let invalid = || FactoryError::illegal_argument(format!("Invalid escaped char in [{s}]"));
    let mut out = Vec::with_capacity(units.len());
    let mut read = 0;
    while read < units.len() {
        let mut c = units[read];
        read += 1;
        if c == u16::from(b'\\') {
            if read >= units.len() {
                return Err(invalid());
            }
            c = units[read];
            read += 1;
            c = match u8::try_from(c).map(char::from) {
                Ok('n') => u16::from(b'\n'),
                Ok('t') => u16::from(b'\t'),
                Ok('r') => u16::from(b'\r'),
                Ok('b') => 0x08,
                Ok('f') => 0x0C,
                Ok('u') => {
                    if read + 3 >= units.len() {
                        return Err(invalid());
                    }
                    let hex = String::from_utf16_lossy(&units[read..read + 4]);
                    read += 4;
                    parse_hex_unit(&hex)?
                }
                _ => c,
            };
        }
        out.push(c);
    }
    Ok(out)
}

/// `(char) Integer.parseInt(hex, 16)` of four units.
fn parse_hex_unit(hex: &str) -> Result<u16, FactoryError> {
    let digits = hex.strip_prefix(['+', '-']).unwrap_or(hex);
    let value = digits
        .chars()
        .try_fold(0u32, |acc, c| c.to_digit(16).map(|d| acc * 16 + d))
        .filter(|_| !digits.is_empty());
    match value {
        // A sign makes a four-unit string at most three digits: in range.
        Some(v) if hex.starts_with('-') => Ok((v as u16).wrapping_neg()),
        Some(v) => Ok(v as u16),
        None => Err(FactoryError::new(
            JavaException::NumberFormat,
            format!("For input string: \"{hex}\" under radix 16"),
        )),
    }
}

/// `org.apache.lucene.analysis.charfilter.MappingCharFilterFactory` (`mapping`).
pub struct MappingCharFilterFactory {
    base: FactoryBase,
    mapping: Option<String>,
    norm_map: Option<Arc<NormalizeCharMap>>,
}
analysis_factory!(MappingCharFilterFactory, aware);

impl FactoryClass for MappingCharFilterFactory {
    const NAME: &'static str = "mapping";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.charfilter.MappingCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let mapping = args::get(args, "mapping");
        args::reject_unknown(args)?;
        Ok(MappingCharFilterFactory {
            base,
            mapping,
            norm_map: None,
        })
    }
}

impl MappingCharFilterFactory {
    // Java: MappingCharFilterFactory.inform
    fn inform_impl(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        let Some(mapping) = &self.mapping else {
            return Ok(());
        };
        let mut rules = Vec::new();
        for file in args::split_file_names(Some(mapping)) {
            rules.extend(get_lines(loader, java_trim(&file))?);
        }
        let pattern = JavaPattern::compile(MAPPING_RULE)?;
        let mut builder = NormalizeCharMapBuilder::new();
        let mut count = 0;
        for rule in &rules {
            let mut m = JavaMatcher::new(&pattern, rule);
            if !m.find() {
                return Err(FactoryError::illegal_argument(format!(
                    "Invalid Mapping Rule : [{rule}], file = {mapping}"
                )));
            }
            let from = parse_escaped(&m.group(1).unwrap_or_default())?;
            let to = parse_escaped(&m.group(2).unwrap_or_default())?;
            builder.add(
                &String::from_utf16_lossy(&from),
                &String::from_utf16_lossy(&to),
            )?;
            count += 1;
        }
        // Java: a map with no rules has a null FST, and the factory returns
        // the reader unchanged.
        self.norm_map = (count > 0).then(|| Arc::new(builder.build()));
        Ok(())
    }
}

impl CharFilterFactory for MappingCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        match &self.norm_map {
            None => input,
            Some(map) => Box::new(MappingCharFilter::new(Arc::clone(map), input)),
        }
    }

    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

/// `org.apache.lucene.analysis.cjk.CJKWidthCharFilterFactory` (`cjkWidth`).
pub struct CJKWidthCharFilterFactory {
    base: FactoryBase,
}
analysis_factory!(CJKWidthCharFilterFactory);

impl FactoryClass for CJKWidthCharFilterFactory {
    const NAME: &'static str = "cjkWidth";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.cjk.CJKWidthCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(CJKWidthCharFilterFactory { base })
    }
}

impl CharFilterFactory for CJKWidthCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(crate::cjk::CJKWidthCharFilter::new(input))
    }

    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

/// `org.apache.lucene.analysis.fa.PersianCharFilterFactory` (`persian`).
pub struct PersianCharFilterFactory {
    base: FactoryBase,
}
analysis_factory!(PersianCharFilterFactory);

impl FactoryClass for PersianCharFilterFactory {
    const NAME: &'static str = "persian";
    const CLASS_NAME: &'static str = "org.apache.lucene.analysis.fa.PersianCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        args::reject_unknown(args)?;
        Ok(PersianCharFilterFactory { base })
    }
}

impl CharFilterFactory for PersianCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(crate::lang::fa::PersianCharFilter::new(input))
    }

    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

/// `org.apache.lucene.analysis.pattern.PatternReplaceCharFilterFactory` (`patternReplace`).
pub struct PatternReplaceCharFilterFactory {
    base: FactoryBase,
    pattern: JavaPattern,
    replacement: String,
}
analysis_factory!(PatternReplaceCharFilterFactory);

impl FactoryClass for PatternReplaceCharFilterFactory {
    const NAME: &'static str = "patternReplace";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.pattern.PatternReplaceCharFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let pattern = args::get_pattern(args, "pattern", "PatternReplaceCharFilterFactory")?;
        let replacement = args::get_or(args, "replacement", "");
        args::reject_unknown(args)?;
        Ok(PatternReplaceCharFilterFactory {
            base,
            pattern,
            replacement,
        })
    }
}

impl CharFilterFactory for PatternReplaceCharFilterFactory {
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(crate::pattern::PatternReplaceCharFilter::new(
            self.pattern.clone(),
            &self.replacement,
            input,
        ))
    }

    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        self.create(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::{AnalysisFactory, MapResourceLoader};
    use crate::reader::read_to_string;
    use crate::StrReader;

    fn run(f: &dyn CharFilterFactory, text: &str) -> String {
        let mut r = f.create(Box::new(StrReader::new(text)));
        read_to_string(&mut *r).unwrap()
    }

    #[test]
    fn char_filters_build_and_filter() {
        let mut a = JavaArgs::from_pairs(&[("escapedTags", "b")]);
        let html = HTMLStripCharFilterFactory::from_args(&mut a).unwrap();
        assert_eq!(run(&html, "<b>x</b><i>y</i>"), "<b>x</b>y");
        let plain = HTMLStripCharFilterFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert_eq!(run(&plain, "<i>y</i>"), "y");

        let cjk = CJKWidthCharFilterFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert_eq!(run(&cjk, "\u{FF21}"), "A");
        let mut r = cjk.normalize(Box::new(StrReader::new("\u{FF21}")));
        assert_eq!(read_to_string(&mut *r).unwrap(), "A");

        let fa = PersianCharFilterFactory::from_args(&mut JavaArgs::new()).unwrap();
        assert_eq!(run(&fa, "a\u{200C}b"), "a b");
        let mut r = fa.normalize(Box::new(StrReader::new("x")));
        assert_eq!(read_to_string(&mut *r).unwrap(), "x");

        let mut a = JavaArgs::from_pairs(&[("pattern", "a+"), ("replacement", "b")]);
        let p = PatternReplaceCharFilterFactory::from_args(&mut a).unwrap();
        assert_eq!(run(&p, "caat"), "cbt");
        let mut r = p.normalize(Box::new(StrReader::new("a")));
        assert_eq!(read_to_string(&mut *r).unwrap(), "b");
    }

    #[test]
    fn mapping_rules() {
        let loader = MapResourceLoader::new()
            .with(
                "m.txt",
                "\"a\" => \"b\"\n\"\\u00e9\" => \"e\"\n\"\\t\\\\\\n\\r\\b\\f\\\"\\x\" => \"\"\n",
            )
            .with("bad.txt", "a => b\n")
            .with("empty.txt", "# nothing\n")
            .with("dup.txt", "\"a\" => \"b\"\n\"a\" => \"c\"\n")
            .with("esc.txt", "\"\\\" => \"x\"\n")
            .with("hex.txt", "\"\\uzzzz\" => \"x\"\n")
            .with("short.txt", "\"\\u12\" => \"x\"\n");
        let build = |file: Option<&str>| {
            let mut a = match file {
                Some(f) => JavaArgs::from_pairs(&[("mapping", f)]),
                None => JavaArgs::new(),
            };
            let mut f = MappingCharFilterFactory::from_args(&mut a).unwrap();
            assert!(f.is_resource_loader_aware());
            f.inform(&loader).map(|_| f)
        };
        let f = build(Some("m.txt")).unwrap();
        assert_eq!(run(&f, "aé"), "be");
        let mut r = f.normalize(Box::new(StrReader::new("a")));
        assert_eq!(read_to_string(&mut *r).unwrap(), "b");
        assert_eq!(run(&build(None).unwrap(), "a"), "a");
        assert_eq!(run(&build(Some("empty.txt")).unwrap(), "a"), "a");
        assert_eq!(
            build(Some("bad.txt")).err().unwrap().message,
            "Invalid Mapping Rule : [a => b], file = bad.txt"
        );
        assert_eq!(
            build(Some("dup.txt")).err().unwrap().message,
            "match \"a\" was already added"
        );
        assert_eq!(
            build(Some("esc.txt")).err().unwrap().message,
            "Invalid escaped char in [\\]"
        );
        assert_eq!(
            build(Some("hex.txt")).err().unwrap().message,
            "For input string: \"zzzz\" under radix 16"
        );
        assert_eq!(
            build(Some("short.txt")).err().unwrap().message,
            "Invalid escaped char in [\\u12]"
        );
        assert_eq!(parse_escaped("\\u-001").unwrap(), vec![0xFFFF]);
        assert_eq!(parse_escaped("\\\"").unwrap(), vec![u16::from(b'"')]);
    }
}
