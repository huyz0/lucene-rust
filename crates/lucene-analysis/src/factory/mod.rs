//! The analysis factory SPI and `CustomAnalyzer`: build an analyzer from the
//! same configuration text as in Java -- a tokenizer, token filters and char
//! filters named by their SPI names (`"whitespace"`, `"lowercase"`,
//! `"htmlStrip"`) with string arguments.
//!
//! # Shape
//!
//! Java's `AbstractAnalysisFactory` is one base class holding the argument
//! parsing helpers and the `luceneMatchVersion`; `TokenizerFactory`,
//! `TokenFilterFactory` and `CharFilterFactory` add `create`, and
//! `AnalysisSPILoader` finds the classes by their `NAME` through
//! `ServiceLoader`. The port keeps the three kinds as traits
//! ([`TokenizerFactory`], [`TokenFilterFactory`], [`CharFilterFactory`])
//! over a shared [`AnalysisFactory`] (the base's accessors and
//! `ResourceLoaderAware.inform`), the base's state as [`FactoryBase`], its
//! helpers as the functions of [`args`] over [`JavaArgs`] (a `HashMap` that
//! iterates in Java's order, so `"Unknown parameters: " + args` prints what
//! Java prints), and `ServiceLoader` as a registry ([`spi`]) filled with
//! analysis-common's factories in `module-info.java`'s order. A factory type
//! is Java's `Class` object: [`FactoryClass`] carries its `NAME`, its class
//! name and its `Map` constructor.
//!
//! Every factory parses its arguments in Java's order with Java's messages,
//! and [`FactoryError`] names the exception class Java throws. A factory's
//! `create` builds the already-ported component.
//!
//! [`ResourceLoader`] is Java's interface minus class loading
//! ([`ClasspathResourceLoader`] serves the resource files this crate
//! vendors, [`FilesystemResourceLoader`] a directory); classes named in
//! configuration (a `SynonymFilterFactory`'s `tokenizerFactory`, a payload
//! encoder, a Snowball stemmer) are looked up in [`spi`]'s class table.
//!
//! Differs: Java prints a factory as `ClassName@identityHash`; the port
//! prints the class name alone (`CustomAnalyzer`'s `toString`).

pub mod args;
#[cfg(test)]
mod boundary_tests;
mod char_filters;
mod conditional;
mod custom_analyzer;
mod filters_lang;
mod filters_misc;
mod filters_resource;
mod loader;
pub mod spi;
mod tokenizers;
mod xml_source;

use lucene_util::version::Version;

pub use args::JavaArgs;
pub use char_filters::*;
pub use conditional::{
    ConditionalTokenFilterFactory, DynConditionalFilter, ProtectedTermFilterFactory,
    TermPredicateFactory,
};
pub use custom_analyzer::{ConditionBuilder, CustomAnalyzer, CustomAnalyzerBuilder};
pub use filters_lang::*;
pub use filters_misc::*;
pub use filters_resource::*;
pub use loader::{
    ClasspathResourceLoader, FilesystemResourceLoader, MapResourceLoader, ResourceLoader,
};
pub use tokenizers::*;

use crate::reader::CharReader;
use crate::token_stream::TokenStream;
use crate::AnalysisError;

/// The Java exception class a factory error stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JavaException {
    /// `IllegalArgumentException`.
    IllegalArgument,
    /// `NumberFormatException`.
    NumberFormat,
    /// `IllegalStateException`.
    IllegalState,
    /// `UnsupportedOperationException`.
    UnsupportedOperation,
    /// `IOException` (and `NoSuchFileException`, `MalformedInputException`,
    /// which Java reports as their own simple names: see
    /// [`FactoryError::java_class`]).
    Io,
    /// `RuntimeException`.
    Runtime,
    /// `NullPointerException`.
    NullPointer,
    /// `SetOnce.AlreadySetException`.
    AlreadySet,
    /// `AlreadyClosedException`.
    AlreadyClosed,
    /// `StringIndexOutOfBoundsException`.
    StringIndexOutOfBounds,
    /// `java.util.IllformedLocaleException`.
    IllformedLocale,
    /// `java.nio.charset.MalformedInputException`.
    MalformedInput,
    /// `TooComplexToDeterminizeException`.
    TooComplexToDeterminize,
    /// `java.util.NoSuchElementException`.
    NoSuchElement,
    /// `ArrayIndexOutOfBoundsException`.
    ArrayIndexOutOfBounds,
    /// `java.util.regex.PatternSyntaxException` (its message is the port's,
    /// not Java's multi-line one).
    PatternSyntax,
    /// `java.io.UnsupportedEncodingException`.
    UnsupportedEncoding,
}

impl JavaException {
    /// The class's fully qualified name (what `Throwable.toString()` prints).
    pub fn qualified_name(self) -> String {
        let package = match self {
            JavaException::Io | JavaException::UnsupportedEncoding => "java.io.",
            JavaException::AlreadySet => "org.apache.lucene.util.SetOnce$",
            JavaException::AlreadyClosed => "org.apache.lucene.store.",
            JavaException::TooComplexToDeterminize => "org.apache.lucene.util.automaton.",
            JavaException::IllformedLocale | JavaException::NoSuchElement => "java.util.",
            JavaException::MalformedInput => "java.nio.charset.",
            JavaException::PatternSyntax => "java.util.regex.",
            _ => "java.lang.",
        };
        format!("{package}{}", self.simple_name())
    }

    /// The class's simple name.
    pub fn simple_name(self) -> &'static str {
        match self {
            JavaException::IllegalArgument => "IllegalArgumentException",
            JavaException::NumberFormat => "NumberFormatException",
            JavaException::IllegalState => "IllegalStateException",
            JavaException::UnsupportedOperation => "UnsupportedOperationException",
            JavaException::Io => "IOException",
            JavaException::Runtime => "RuntimeException",
            JavaException::NullPointer => "NullPointerException",
            JavaException::AlreadySet => "AlreadySetException",
            JavaException::AlreadyClosed => "AlreadyClosedException",
            JavaException::StringIndexOutOfBounds => "StringIndexOutOfBoundsException",
            JavaException::IllformedLocale => "IllformedLocaleException",
            JavaException::MalformedInput => "MalformedInputException",
            JavaException::TooComplexToDeterminize => "TooComplexToDeterminizeException",
            JavaException::NoSuchElement => "NoSuchElementException",
            JavaException::ArrayIndexOutOfBounds => "ArrayIndexOutOfBoundsException",
            JavaException::PatternSyntax => "PatternSyntaxException",
            JavaException::UnsupportedEncoding => "UnsupportedEncodingException",
        }
    }
}

/// What building or informing a factory throws in Java: the exception class
/// and its `getMessage()`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}: {message}", kind.simple_name())]
pub struct FactoryError {
    /// The exception class.
    pub kind: JavaException,
    /// Java's message.
    pub message: String,
}

impl FactoryError {
    /// An error of `kind`.
    pub fn new(kind: JavaException, message: impl Into<String>) -> Self {
        FactoryError {
            kind,
            message: message.into(),
        }
    }

    /// An `IllegalArgumentException`.
    pub fn illegal_argument(message: impl Into<String>) -> Self {
        Self::new(JavaException::IllegalArgument, message)
    }

    /// An `IOException`.
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(JavaException::Io, message)
    }

    /// The simple name of the class Java throws.
    pub fn java_class(&self) -> &'static str {
        self.kind.simple_name()
    }
}

impl From<AnalysisError> for FactoryError {
    fn from(e: AnalysisError) -> Self {
        match e {
            AnalysisError::IllegalArgument(m) => match m.strip_prefix("NumberFormatException: ") {
                Some(rest) => FactoryError::new(JavaException::NumberFormat, rest),
                None => FactoryError::new(JavaException::IllegalArgument, m),
            },
            AnalysisError::IllegalState(m) => FactoryError::new(JavaException::IllegalState, m),
            AnalysisError::AlreadyClosed(m) => FactoryError::new(JavaException::AlreadyClosed, m),
            AnalysisError::Io(m) => FactoryError::new(JavaException::Io, m),
        }
    }
}

impl From<lucene_util::automaton::AutomatonError> for FactoryError {
    fn from(e: lucene_util::automaton::AutomatonError) -> Self {
        use lucene_util::automaton::AutomatonError as A;
        match e {
            A::TooComplex(t) => {
                FactoryError::new(JavaException::TooComplexToDeterminize, t.to_string())
            }
            A::IllegalArgument(m) => FactoryError::illegal_argument(m),
            A::IllegalState(m) => FactoryError::new(JavaException::IllegalState, m),
        }
    }
}

impl From<FactoryError> for AnalysisError {
    fn from(e: FactoryError) -> Self {
        match e.kind {
            JavaException::IllegalArgument | JavaException::IllformedLocale => {
                AnalysisError::IllegalArgument(e.message)
            }
            JavaException::NumberFormat => {
                AnalysisError::IllegalArgument(format!("NumberFormatException: {}", e.message))
            }
            JavaException::Io | JavaException::MalformedInput => AnalysisError::Io(e.message),
            JavaException::AlreadyClosed => AnalysisError::AlreadyClosed(e.message),
            _ => AnalysisError::IllegalState(e.message),
        }
    }
}

/// `AbstractAnalysisFactory.LUCENE_MATCH_VERSION_PARAM`.
pub const LUCENE_MATCH_VERSION_PARAM: &str = "luceneMatchVersion";
const CLASS_NAME: &str = "class";
const SPI_NAME: &str = "name";

/// `AbstractAnalysisFactory`'s state: the original arguments, the
/// `luceneMatchVersion`, the class name.
#[derive(Debug, Clone)]
pub struct FactoryBase {
    class_name: &'static str,
    original_args: JavaArgs,
    lucene_match_version: Version,
    explicit_lucene_match_version: bool,
}

impl FactoryBase {
    /// `AbstractAnalysisFactory(Map<String, String> args)`: keeps a copy of
    /// the arguments, consumes `luceneMatchVersion` (parsed leniently,
    /// [`Version::LATEST`] when absent) and the `class` and `name` keys.
    pub fn new(class_name: &'static str, args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let original_args = args.clone();
        let lucene_match_version = match args.remove(LUCENE_MATCH_VERSION_PARAM) {
            None => Version::LATEST,
            Some(v) => Version::parse_leniently(&v).map_err(|pe| {
                // new IllegalArgumentException(ParseException): the cause's toString.
                FactoryError::illegal_argument(format!("java.text.ParseException: {}", pe.0))
            })?,
        };
        args.remove(CLASS_NAME);
        args.remove(SPI_NAME);
        Ok(FactoryBase {
            class_name,
            original_args,
            lucene_match_version,
            explicit_lucene_match_version: false,
        })
    }

    /// `getOriginalArgs()`.
    pub fn original_args(&self) -> &JavaArgs {
        &self.original_args
    }

    /// `getLuceneMatchVersion()`.
    pub fn lucene_match_version(&self) -> Version {
        self.lucene_match_version
    }

    /// `getClassArg()`: the `class` argument, or the class name.
    pub fn class_arg(&self) -> &str {
        self.original_args
            .get(CLASS_NAME)
            .unwrap_or(self.class_name)
    }

    /// The factory's fully qualified Java class name.
    pub fn class_name(&self) -> &'static str {
        self.class_name
    }

    /// The class's simple name (`getClass().getSimpleName()`).
    pub fn simple_class_name(&self) -> &'static str {
        simple_name(self.class_name)
    }

    /// `isExplicitLuceneMatchVersion()`.
    pub fn is_explicit_lucene_match_version(&self) -> bool {
        self.explicit_lucene_match_version
    }

    /// `setExplicitLuceneMatchVersion(boolean)`.
    pub fn set_explicit_lucene_match_version(&mut self, explicit: bool) {
        self.explicit_lucene_match_version = explicit;
    }
}

/// The part of a class name after its last `.` and `$`.
pub(crate) fn simple_name(class_name: &str) -> &str {
    class_name.rsplit(['.', '$']).next().unwrap_or(class_name)
}

/// What every factory is: `AbstractAnalysisFactory`'s accessors and, for a
/// `ResourceLoaderAware` factory, `inform`.
pub trait AnalysisFactory: Send + Sync {
    /// The shared state.
    fn base(&self) -> &FactoryBase;

    /// The shared state, mutably.
    fn base_mut(&mut self) -> &mut FactoryBase;

    /// Whether the factory implements `ResourceLoaderAware`.
    fn is_resource_loader_aware(&self) -> bool {
        false
    }

    /// `ResourceLoaderAware.inform(ResourceLoader)`: load the resources the
    /// arguments name. A no-op for the others.
    fn inform(&mut self, _loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        Ok(())
    }
}

/// `org.apache.lucene.analysis.TokenizerFactory`.
pub trait TokenizerFactory: AnalysisFactory {
    /// `create()`: a new tokenizer.
    fn create(&self) -> Result<Box<dyn TokenStream>, AnalysisError>;
}

/// `org.apache.lucene.analysis.TokenFilterFactory`.
pub trait TokenFilterFactory: AnalysisFactory {
    /// `create(TokenStream)`: the filter over `input`.
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError>;

    /// `normalize(TokenStream)`: the part of the filter that applies to a
    /// query term; the identity unless the factory overrides it.
    fn normalize(&self, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        input
    }

    /// The factory as a `ConditionalTokenFilterFactory`, if it is one.
    fn as_conditional(&mut self) -> Option<&mut dyn ConditionalTokenFilterFactory> {
        None
    }
}

/// `org.apache.lucene.analysis.CharFilterFactory`.
pub trait CharFilterFactory: AnalysisFactory {
    /// `create(Reader)`: the char filter over `input`.
    fn create(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader>;

    /// `normalize(Reader)`: the identity unless the factory overrides it.
    fn normalize(&self, input: Box<dyn CharReader>) -> Box<dyn CharReader> {
        input
    }
}

/// A factory class: Java's `Class<? extends ...Factory>` -- its `NAME`, its
/// name and its `Map` constructor.
pub trait FactoryClass: Sized {
    /// `public static final String NAME`: the SPI name.
    const NAME: &'static str;
    /// The fully qualified Java class name.
    const CLASS_NAME: &'static str;
    /// `new XFactory(Map<String, String> args)`: consumes the arguments it
    /// reads; anything left is an error.
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError>;
}

/// Implements [`AnalysisFactory`] for a factory struct with a `base` field
/// (and, given `aware`, forwards `inform` to its `inform_impl`).
macro_rules! analysis_factory {
    ($t:ty) => {
        impl $crate::factory::AnalysisFactory for $t {
            fn base(&self) -> &$crate::factory::FactoryBase {
                &self.base
            }
            fn base_mut(&mut self) -> &mut $crate::factory::FactoryBase {
                &mut self.base
            }
        }
    };
    ($t:ty, aware) => {
        impl $crate::factory::AnalysisFactory for $t {
            fn base(&self) -> &$crate::factory::FactoryBase {
                &self.base
            }
            fn base_mut(&mut self) -> &mut $crate::factory::FactoryBase {
                &mut self.base
            }
            fn is_resource_loader_aware(&self) -> bool {
                true
            }
            fn inform(
                &mut self,
                loader: &dyn $crate::factory::ResourceLoader,
            ) -> Result<(), $crate::factory::FactoryError> {
                self.inform_impl(loader)
            }
        }
    };
}
pub(crate) use analysis_factory;

/// Declares a factory struct with a `base` and the given fields.
macro_rules! factory_struct {
    ($(#[$m:meta])* $name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        $(#[$m])*
        pub struct $name {
            base: FactoryBase,
            $($field: $ty,)*
        }
    };
}
pub(crate) use factory_struct;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_consumes_version_class_and_name() {
        let mut args = JavaArgs::from_pairs(&[
            ("luceneMatchVersion", "LUCENE_9_0_0"),
            ("class", "solr.X"),
            ("name", "x"),
            ("other", "1"),
        ]);
        let mut base =
            FactoryBase::new("org.apache.lucene.analysis.core.XFactory", &mut args).unwrap();
        assert_eq!(base.lucene_match_version(), Version::LUCENE_9_0_0);
        assert_eq!(args.keys(), vec!["other"]);
        assert_eq!(base.original_args().len(), 4);
        assert_eq!(base.class_arg(), "solr.X");
        assert_eq!(base.simple_class_name(), "XFactory");
        assert_eq!(
            base.class_name(),
            "org.apache.lucene.analysis.core.XFactory"
        );
        assert!(!base.is_explicit_lucene_match_version());
        base.set_explicit_lucene_match_version(true);
        assert!(base.is_explicit_lucene_match_version());

        let mut none = JavaArgs::new();
        let b = FactoryBase::new("a.B$1", &mut none).unwrap();
        assert_eq!(b.lucene_match_version(), Version::LATEST);
        assert_eq!(b.class_arg(), "a.B$1");
        assert_eq!(b.simple_class_name(), "1");

        let mut bad = JavaArgs::from_pairs(&[("luceneMatchVersion", "x.y")]);
        let e = FactoryBase::new("a.B", &mut bad).unwrap_err();
        assert_eq!(e.java_class(), "IllegalArgumentException");
        assert!(e.message.starts_with("java.text.ParseException: "));
    }

    #[test]
    fn errors_convert_both_ways() {
        let cases = [
            (
                AnalysisError::IllegalArgument("a".into()),
                JavaException::IllegalArgument,
            ),
            (
                AnalysisError::IllegalArgument("NumberFormatException: b".into()),
                JavaException::NumberFormat,
            ),
            (
                AnalysisError::IllegalState("c".into()),
                JavaException::IllegalState,
            ),
            (
                AnalysisError::AlreadyClosed("d".into()),
                JavaException::AlreadyClosed,
            ),
            (AnalysisError::Io("e".into()), JavaException::Io),
        ];
        for (e, kind) in cases {
            let f = FactoryError::from(e.clone());
            assert_eq!(f.kind, kind);
            assert_eq!(AnalysisError::from(f), e);
        }
        let f = FactoryError::new(JavaException::Runtime, "r");
        assert_eq!(f.to_string(), "RuntimeException: r");
        assert_eq!(
            AnalysisError::from(f),
            AnalysisError::IllegalState("r".into())
        );
        assert_eq!(
            AnalysisError::from(FactoryError::new(JavaException::MalformedInput, "m")),
            AnalysisError::Io("m".into())
        );
        assert_eq!(
            AnalysisError::from(FactoryError::new(JavaException::IllformedLocale, "l")),
            AnalysisError::IllegalArgument("l".into())
        );
        for k in [
            JavaException::UnsupportedOperation,
            JavaException::NullPointer,
            JavaException::AlreadySet,
            JavaException::StringIndexOutOfBounds,
        ] {
            assert!(k.simple_name().ends_with("Exception"));
        }
    }
}
