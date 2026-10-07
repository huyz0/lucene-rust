//! `ConditionalTokenFilterFactory`, `ProtectedTermFilterFactory`
//! (`protectedTerm`) and the anonymous factory behind
//! `CustomAnalyzer.Builder.whenTerm`.
//!
//! A conditional factory's inner filters are factories too, so its delegate
//! chain is built at run time over `Box<dyn TokenStream>`; the
//! [`ConditionalTokenFilter`] reaches the wrapper at the bottom of it through
//! [`TokenStream::conditional_root`] ([`DynDelegate`]).

use std::sync::Arc;

use super::args::{self, JavaArgs};
use super::loader::get_word_set;
use super::{
    spi, AnalysisFactory, FactoryBase, FactoryClass, FactoryError, ResourceLoader,
    TokenFilterFactory,
};
use crate::attributes::AttributeSource;
use crate::miscellaneous::{
    ConditionalRoot, ConditionalTokenFilter, EmptyTokenStream, OneTimeWrapper, ShouldFilter,
};
use crate::token_stream::{TokenStream, Tokenizer};
use crate::{AnalysisError, CharArraySet};

/// `shouldFilter()` of a factory-built conditional filter.
#[derive(Clone)]
pub enum FactoryCondition {
    /// `ProtectedTermFilter`: the term is not protected.
    NotProtected(Arc<CharArraySet>),
    /// `whenTerm(Predicate<CharSequence>)`.
    Term(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl ShouldFilter for FactoryCondition {
    fn should_filter(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(match self {
            FactoryCondition::NotProtected(set) => !set.contains(a.term()),
            FactoryCondition::Term(p) => p(a.term()),
        })
    }
}

type DynWrapper = OneTimeWrapper<Box<dyn TokenStream>, FactoryCondition>;

/// A delegate chain built by factories over a [`DynWrapper`].
pub struct DynDelegate(Box<dyn TokenStream>);

impl TokenStream for DynDelegate {
    fn attributes(&self) -> &AttributeSource {
        self.0.attributes()
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        self.0.attributes_mut()
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.0.increment_token()
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.0.reset()
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.0.end()
    }
    fn close(&mut self) -> Result<(), AnalysisError> {
        self.0.close()
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        self.0.as_tokenizer()
    }
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        self.0.conditional_root()
    }
}

impl DynDelegate {
    fn wrapper(&mut self) -> Option<&mut DynWrapper> {
        self.0.conditional_root()?.downcast_mut::<DynWrapper>()
    }
}

impl ConditionalRoot<Box<dyn TokenStream>, FactoryCondition> for DynDelegate {
    fn root(&mut self) -> &mut DynWrapper {
        // INVARIANT: `build_conditional` refuses a chain without the wrapper.
        self.wrapper()
            .expect("a factory-built delegate chain ends at its wrapper")
    }
}

/// A conditional filter whose delegate chain was built by factories.
pub type DynConditionalFilter =
    ConditionalTokenFilter<Box<dyn TokenStream>, FactoryCondition, DynDelegate>;

/// `ConditionalTokenFilterFactory.create(input)`: the input itself without
/// inner filters, else the conditional filter over them.
fn build_conditional(
    input: Box<dyn TokenStream>,
    cond: FactoryCondition,
    inner: &[Box<dyn TokenFilterFactory>],
) -> Result<Box<dyn TokenStream>, AnalysisError> {
    if inner.is_empty() {
        return Ok(input);
    }
    let mut error = None;
    let mut filter = ConditionalTokenFilter::new(input, cond, |wrapper| {
        let mut ts: Box<dyn TokenStream> = Box::new(wrapper);
        for factory in inner {
            match factory.create(ts) {
                Ok(next) => ts = next,
                Err(e) => {
                    error = Some(e);
                    return DynDelegate(Box::new(EmptyTokenStream::new()));
                }
            }
        }
        DynDelegate(ts)
    });
    if let Some(e) = error {
        return Err(e);
    }
    let delegate: &mut DynDelegate = crate::token_stream::TokenFilter::input_mut(&mut filter);
    if delegate.wrapper().is_none() {
        return Err(AnalysisError::IllegalState(
            "an inner filter of a conditional filter dropped its input".into(),
        ));
    }
    Ok(Box::new(filter))
}

/// `org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilterFactory`:
/// a token filter factory whose filter runs inner filters on the tokens a
/// condition selects.
pub trait ConditionalTokenFilterFactory: TokenFilterFactory {
    /// `setInnerFilters(List<TokenFilterFactory>)`.
    fn set_inner_filters(&mut self, inner_filters: Vec<Box<dyn TokenFilterFactory>>);
}

/// `ConditionalTokenFilterFactory.inform`: informs the inner factories, then
/// runs `doInform` -- neither when there are no inner filters.
fn inform_inner(
    inner: &mut Option<Vec<Box<dyn TokenFilterFactory>>>,
    loader: &dyn ResourceLoader,
) -> Result<bool, FactoryError> {
    let Some(inner) = inner else {
        return Ok(false);
    };
    for factory in inner.iter_mut() {
        if factory.is_resource_loader_aware() {
            factory.inform(loader)?;
        }
    }
    Ok(true)
}

/// `org.apache.lucene.analysis.miscellaneous.ProtectedTermFilterFactory`
/// (`protectedTerm`): `protected` word files whose terms skip the inner
/// filters; `ignoreCase`; `wrappedFilters` (`name[-id]`, comma-separated)
/// with their arguments as `name[-id].arg`.
pub struct ProtectedTermFilterFactory {
    base: FactoryBase,
    term_files: String,
    ignore_case: bool,
    inner: Option<Vec<Box<dyn TokenFilterFactory>>>,
    protected_terms: Option<Arc<CharArraySet>>,
}

impl ProtectedTermFilterFactory {
    /// `isIgnoreCase()`.
    pub fn is_ignore_case(&self) -> bool {
        self.ignore_case
    }

    /// `getProtectedTerms()`.
    pub fn protected_terms(&self) -> Option<&Arc<CharArraySet>> {
        self.protected_terms.as_ref()
    }

    // Java: ProtectedTermFilterFactory.handleWrappedFilterArgs
    fn handle_wrapped_filter_args(
        &mut self,
        wrapped_filters: &str,
        args: &mut JavaArgs,
    ) -> Result<(), FactoryError> {
        let mut wrapped: Vec<(String, JavaArgs)> = Vec::new();
        for name in args::split_at(',', Some(wrapped_filters)) {
            let name = java_lower(super::loader::java_trim(&name));
            if wrapped.iter().any(|(n, _)| *n == name) {
                return Err(FactoryError::illegal_argument(format!(
                    "wrappedFilters contains duplicate '{name}'. Add unique '-id' suffixes (stripped prior to SPI lookup)."
                )));
            }
            wrapped.push((name, JavaArgs::new()));
        }
        for key in args.keys() {
            let split = args::split_at('.', Some(&key));
            if split.len() != 2 {
                continue;
            }
            let filter = java_lower(&split[0]);
            if let Some((_, filter_args)) = wrapped.iter_mut().find(|(n, _)| *n == filter) {
                let value = args.remove(&key).unwrap_or_default();
                filter_args.put(&split[1], &value);
            }
        }
        if args.is_empty() {
            let mut inner = Vec::with_capacity(wrapped.len());
            for (name, mut filter_args) in wrapped {
                let spi_name = name.split('-').next().unwrap_or(&name);
                inner.push(spi::token_filter_for_name(spi_name, &mut filter_args)?);
            }
            self.inner = Some(inner);
        }
        Ok(())
    }
}

/// `String.toLowerCase(Locale.ROOT)`: the per-character mapping plus
/// Java's special cases (`İ` becomes `i̇`, a final `Σ` `ς`).
fn java_lower(s: &str) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    String::from_utf16_lossy(&crate::lang::java_string_to_lower_case(&units))
}

impl AnalysisFactory for ProtectedTermFilterFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    // Java: ConditionalTokenFilterFactory.inform, ProtectedTermFilterFactory.doInform
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        if inform_inner(&mut self.inner, loader)? {
            self.protected_terms =
                get_word_set(loader, &self.term_files, self.ignore_case)?.map(Arc::new);
        }
        Ok(())
    }
}

impl FactoryClass for ProtectedTermFilterFactory {
    const NAME: &'static str = "protectedTerm";
    const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.miscellaneous.ProtectedTermFilterFactory";
    fn from_args(args: &mut JavaArgs) -> Result<Self, FactoryError> {
        let base = FactoryBase::new(Self::CLASS_NAME, args)?;
        let term_files = args::require(args, "protected")?;
        let ignore_case = args::get_boolean(args, "ignoreCase", false);
        let mut f = ProtectedTermFilterFactory {
            base,
            term_files,
            ignore_case,
            inner: None,
            protected_terms: None,
        };
        if let Some(wrapped) = args::get(args, "wrappedFilters") {
            f.handle_wrapped_filter_args(&wrapped, args)?;
        }
        args::reject_unknown(args)?;
        Ok(f)
    }
}

impl TokenFilterFactory for ProtectedTermFilterFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        let Some(inner) = &self.inner else {
            return Ok(input);
        };
        if inner.is_empty() {
            return Ok(input);
        }
        let terms = self.protected_terms.clone().ok_or_else(|| {
            AnalysisError::IllegalState(
                "NullPointerException: ProtectedTermFilterFactory was not informed".into(),
            )
        })?;
        build_conditional(input, FactoryCondition::NotProtected(terms), inner)
    }

    fn as_conditional(&mut self) -> Option<&mut dyn ConditionalTokenFilterFactory> {
        Some(self)
    }
}

impl ConditionalTokenFilterFactory for ProtectedTermFilterFactory {
    fn set_inner_filters(&mut self, inner_filters: Vec<Box<dyn TokenFilterFactory>>) {
        self.inner = Some(inner_filters);
    }
}

/// The anonymous `ConditionalTokenFilterFactory` of
/// `CustomAnalyzer.Builder.whenTerm(Predicate<CharSequence>)`.
pub struct TermPredicateFactory {
    base: FactoryBase,
    predicate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    inner: Option<Vec<Box<dyn TokenFilterFactory>>>,
}

impl TermPredicateFactory {
    /// `CustomAnalyzer$Builder$1`'s class name.
    pub const CLASS_NAME: &'static str =
        "org.apache.lucene.analysis.custom.CustomAnalyzer$Builder$1";

    /// The factory over `predicate`, with no arguments.
    pub fn new(predicate: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        let base = FactoryBase::new(Self::CLASS_NAME, &mut JavaArgs::new())
            .expect("no arguments, no error");
        TermPredicateFactory {
            base,
            predicate: Arc::new(predicate),
            inner: None,
        }
    }
}

impl AnalysisFactory for TermPredicateFactory {
    fn base(&self) -> &FactoryBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.base
    }
    fn is_resource_loader_aware(&self) -> bool {
        true
    }
    fn inform(&mut self, loader: &dyn ResourceLoader) -> Result<(), FactoryError> {
        inform_inner(&mut self.inner, loader).map(|_| ())
    }
}

impl TokenFilterFactory for TermPredicateFactory {
    fn create(&self, input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        match &self.inner {
            None => Ok(input),
            Some(inner) => build_conditional(
                input,
                FactoryCondition::Term(Arc::clone(&self.predicate)),
                inner,
            ),
        }
    }

    fn as_conditional(&mut self) -> Option<&mut dyn ConditionalTokenFilterFactory> {
        Some(self)
    }
}

impl ConditionalTokenFilterFactory for TermPredicateFactory {
    fn set_inner_filters(&mut self, inner_filters: Vec<Box<dyn TokenFilterFactory>>) {
        self.inner = Some(inner_filters);
    }
}
