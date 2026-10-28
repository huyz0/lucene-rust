//! `lucene-expressions`: compiling JavaScript-like expressions over named
//! variables and using them as [`DoubleValuesSource`]s -- to sort by, to
//! rescore with, to score a [`crate::function::FunctionScoreQuery`] with.
//!
//! * [`js`] is the language: [`js::compile`] (`JavascriptCompiler`), the
//!   [`js::Expression`] it makes, the function table
//!   ([`js::default_functions`], `JavascriptCompiler.properties`) and
//!   [`variable_context`] (`VariableContext`).
//! * [`Bindings`] and [`SimpleBindings`] say what each variable name means;
//!   [`ExpressionValueSource`] (with its `Caching` variant) evaluates an
//!   expression per document over those meanings, and
//!   [`ExpressionRescorer`] rescores a first pass by one.
//!
//! # Deviations
//!
//! Java compiles an expression to a hidden JVM class; here it is lowered to a
//! tree evaluated with the same typing (see [`js`]). `hashCode` is not
//! ported; `equals` is [`DoubleValuesSource::equality_key`] (the class, the
//! expression's text and each variable's own key, as Java's `equals`
//! compares `variables`, `expression` and `needsScores`). A caching source
//! rewrites to a caching source (Java's inherited `rewrite` returns a plain
//! `ExpressionValueSource`: the values are the same, only the sharing of
//! nested values differs).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::explain::Explanation;
use crate::rescorer::{Rescorer, SortRescorer};
use crate::top_docs::TopDocs;
use crate::top_field::SortField;
use crate::values_source::{
    constant, double_sort_field, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};

pub mod js;
pub mod variable_context;

pub use js::{compile, compile_with, default_functions, ExprFunction, Expression, ParseException};
use js::{VarSlot, VarValues};

/// What compiling an expression can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `java.text.ParseException`.
    #[error("{0}")]
    Parse(ParseException),
    /// `NumberFormatException`: an integer literal past a `long`.
    #[error("{0}")]
    NumberFormat(String),
}

impl Expression {
    /// `getDoubleValuesSource(bindings)`.
    ///
    /// # Errors
    /// A variable `bindings` does not bind.
    pub fn double_values_source(
        self: &Arc<Self>,
        bindings: &dyn Bindings,
    ) -> crate::Result<Arc<dyn DoubleValuesSource>> {
        Ok(Arc::new(ExpressionValueSource::new(
            bindings,
            self.clone(),
        )?))
    }

    /// `getSortField(bindings, reverse)`.
    ///
    /// # Errors
    /// As [`Self::double_values_source`].
    pub fn sort_field(
        self: &Arc<Self>,
        bindings: &dyn Bindings,
        reverse: bool,
    ) -> crate::Result<SortField> {
        Ok(double_sort_field(
            self.double_values_source(bindings)?,
            reverse,
            0.0,
        ))
    }

    /// `getRescorer(bindings)`.
    ///
    /// # Errors
    /// As [`Self::double_values_source`].
    pub fn rescorer(
        self: &Arc<Self>,
        bindings: &dyn Bindings,
    ) -> crate::Result<ExpressionRescorer> {
        ExpressionRescorer::new(self.clone(), bindings)
    }
}

/// `Bindings`: what a variable name stands for.
pub trait Bindings {
    /// `getDoubleValuesSource(name)`.
    ///
    /// # Errors
    /// An unbound `name` (`IllegalArgumentException`).
    fn double_values_source(&self, name: &str) -> crate::Result<Arc<dyn DoubleValuesSource>>;
}

enum Binding {
    Source(Arc<dyn DoubleValuesSource>),
    Expression(Arc<Expression>),
}

/// `SimpleBindings`.
#[derive(Default)]
pub struct SimpleBindings {
    map: HashMap<String, Binding>,
}

impl SimpleBindings {
    /// `new SimpleBindings()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `add(name, source)`.
    pub fn add_source(&mut self, name: &str, source: Arc<dyn DoubleValuesSource>) {
        self.map.insert(name.to_string(), Binding::Source(source));
    }

    /// `add(name, expression)`: the expression is evaluated over these
    /// bindings.
    pub fn add_expression(&mut self, name: &str, expression: Arc<Expression>) {
        self.map
            .insert(name.to_string(), Binding::Expression(expression));
    }

    fn resolve(
        &self,
        name: &str,
        against: &dyn Bindings,
    ) -> crate::Result<Arc<dyn DoubleValuesSource>> {
        match self.map.get(name) {
            None => Err(crate::Error::IllegalArgument(format!(
                "Invalid reference '{name}'"
            ))),
            Some(Binding::Source(s)) => Ok(s.clone()),
            Some(Binding::Expression(e)) => e.double_values_source(against),
        }
    }

    /// `validate()`: every binding resolves, with no cycle among them.
    ///
    /// # Errors
    /// An unbound reference, or a cycle (`IllegalArgumentException`).
    pub fn validate(&self) -> crate::Result<()> {
        // in name order, so the error reported is the same from run to run
        let mut names: Vec<&String> = self.map.keys().collect();
        names.sort();
        for origin in names {
            let detect = CycleDetection {
                bindings: self,
                seen: vec![origin.clone()],
            };
            // the origin resolved against a detector that has seen only it
            self.resolve(origin, &detect)?;
        }
        Ok(())
    }
}

impl Bindings for SimpleBindings {
    /// Resolved through the cycle detector: a cycle among bindings never
    /// validated is `validate`'s error (Java's recursion ends in a
    /// `StackOverflowError`; Rust's would abort the process).
    fn double_values_source(&self, name: &str) -> crate::Result<Arc<dyn DoubleValuesSource>> {
        let detect = CycleDetection {
            bindings: self,
            seen: vec![name.to_string()],
        };
        self.resolve(name, &detect)
    }
}

/// `SimpleBindings.CycleDetectionBindings`.
struct CycleDetection<'a> {
    bindings: &'a SimpleBindings,
    seen: Vec<String>,
}

impl Bindings for CycleDetection<'_> {
    fn double_values_source(&self, name: &str) -> crate::Result<Arc<dyn DoubleValuesSource>> {
        if self.seen.iter().any(|s| s == name) {
            // `LinkedHashSet.toString()`
            return Err(crate::Error::IllegalArgument(format!(
                "Recursion error: Cycle detected [{}]->{name}",
                self.seen.join(", ")
            )));
        }
        if !self.bindings.map.contains_key(name) {
            return Err(crate::Error::IllegalArgument(format!(
                "Invalid reference '{name}'"
            )));
        }
        let mut seen = self.seen.clone();
        seen.push(name.to_string());
        let next = CycleDetection {
            bindings: self.bindings,
            seen,
        };
        self.bindings.resolve(name, &next)
    }
}

/// A [`DoubleValues`] shared by the variables that read the same source (and
/// the scores every variable reads).
struct Shared<'c>(Rc<RefCell<BoxDoubleValues<'c>>>);

impl DoubleValues for Shared<'_> {
    fn advance_exact(&mut self, doc: i32) -> crate::Result<bool> {
        self.0.borrow_mut().advance_exact(doc)
    }

    fn double_value(&mut self) -> crate::Result<f64> {
        self.0.borrow_mut().double_value()
    }
}

/// `ExpressionFunctionValues`: the expression's value for the current
/// document, evaluated once. Each variable is a [`VarSlot`]
/// (`zeroWhenUnpositioned`) stored inline and positioned as the evaluation
/// first reads it, so a variable read costs only the wrapped values' own
/// calls -- Java's per-document `advanceExact` over every wrapper is the
/// slot's document check.
struct ExpressionFunctionValues<'c> {
    expression: Arc<Expression>,
    function_values: Vec<VarSlot<'c>>,
    current_value: f64,
    current_doc: i32,
    computed: bool,
}

impl DoubleValues for ExpressionFunctionValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> crate::Result<bool> {
        if self.current_doc != doc {
            self.current_doc = doc;
            self.computed = false;
        }
        Ok(true)
    }

    fn double_value(&mut self) -> crate::Result<f64> {
        if !self.computed {
            self.current_value = self
                .expression
                .evaluate_at(&mut self.function_values, self.current_doc)?;
            self.computed = true;
        }
        Ok(self.current_value)
    }
}

type ValuesCache<'c> = HashMap<String, Rc<RefCell<BoxDoubleValues<'c>>>>;

/// `ExpressionValueSource` and its subclass `CachingExpressionValueSource`
/// (`caching`): the expression over the sources its variables are bound to.
/// The caching one shares the values of a variable that several nested
/// expressions name, rather than reading each again.
pub struct ExpressionValueSource {
    variables: Vec<Arc<dyn DoubleValuesSource>>,
    expression: Arc<Expression>,
    needs_scores: bool,
    caching: bool,
}

impl ExpressionValueSource {
    /// `new ExpressionValueSource(bindings, expression)`.
    ///
    /// # Errors
    /// A variable `bindings` does not bind.
    pub fn new(bindings: &dyn Bindings, expression: Arc<Expression>) -> crate::Result<Self> {
        Self::build(bindings, expression, false)
    }

    /// `new CachingExpressionValueSource(bindings, expression)`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn caching(bindings: &dyn Bindings, expression: Arc<Expression>) -> crate::Result<Self> {
        Self::build(bindings, expression, true)
    }

    fn build(
        bindings: &dyn Bindings,
        expression: Arc<Expression>,
        caching: bool,
    ) -> crate::Result<Self> {
        let mut variables = Vec::with_capacity(expression.variables.len());
        let mut needs_scores = false;
        for name in &expression.variables {
            let source = bindings.double_values_source(name)?;
            needs_scores |= source.needs_scores();
            variables.push(source);
        }
        Ok(Self {
            variables,
            expression,
            needs_scores,
            caching,
        })
    }

    fn values_with_cache<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: &Option<Rc<RefCell<BoxDoubleValues<'c>>>>,
        cache: &mut ValuesCache<'c>,
    ) -> crate::Result<BoxDoubleValues<'c>> {
        let mut external: Vec<VarSlot<'c>> = Vec::with_capacity(self.variables.len());
        for (name, source) in self.expression.variables.iter().zip(&self.variables) {
            let slot = match cache.get(name) {
                Some(v) => VarValues::Shared(v.clone()),
                // a plain source's cache is its own and its names are
                // distinct: nothing else reads a variable's values
                None if !self.caching => VarValues::Owned(
                    source.get_values(
                        ctx,
                        leaf,
                        scores
                            .as_ref()
                            .map(|s| Box::new(Shared(s.clone())) as BoxDoubleValues<'c>),
                    )?,
                ),
                None => {
                    let values: BoxDoubleValues<'c> = match source.as_expression() {
                        Some(nested) if self.caching && nested.caching => {
                            nested.values_with_cache(ctx, leaf, scores, cache)?
                        }
                        _ => source.get_values(
                            ctx,
                            leaf,
                            scores
                                .as_ref()
                                .map(|s| Box::new(Shared(s.clone())) as BoxDoubleValues<'c>),
                        )?,
                    };
                    let rc = Rc::new(RefCell::new(values));
                    cache.insert(name.clone(), rc.clone());
                    VarValues::Shared(rc)
                }
            };
            external.push(VarSlot::new(slot));
        }
        Ok(Box::new(ExpressionFunctionValues {
            expression: self.expression.clone(),
            function_values: external,
            current_value: 0.0,
            current_doc: -1,
            computed: false,
        }))
    }
}

impl DoubleValuesSource for ExpressionValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> crate::Result<BoxDoubleValues<'c>> {
        let scores = scores.map(|s| Rc::new(RefCell::new(s)));
        self.values_with_cache(ctx, leaf, &scores, &mut HashMap::new())
    }

    fn needs_scores(&self) -> bool {
        self.needs_scores
    }

    fn needs_searcher(&self) -> bool {
        self.variables.iter().any(|v| v.needs_searcher())
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        self.variables.iter().all(|v| v.is_cacheable(ctx, leaf))
    }

    fn describe(&self) -> String {
        format!("expr({})", self.expression.source_text)
    }

    fn as_expression(&self) -> Option<&ExpressionValueSource> {
        Some(self)
    }

    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        leaf: usize,
        doc: i32,
        score_explanation: &Explanation,
    ) -> crate::Result<Explanation> {
        let scores = constant(score_explanation.double_value()).get_values(ctx, leaf, None)?;
        let mut dv = self.get_values(ctx, leaf, Some(scores))?;
        if !dv.advance_exact(doc)? {
            return Ok(Explanation::no_match(self.expression.source_text.clone()));
        }
        let mut details = Vec::with_capacity(self.variables.len());
        for v in &self.variables {
            details.push(v.explain(ctx, leaf, doc, score_explanation)?);
        }
        Ok(Explanation::match_double(
            dv.double_value()?,
            format!("{}, computed from:", self.expression.source_text),
        )
        .with_details(details))
    }

    fn equality_key(&self) -> String {
        let variables: Vec<String> = self.variables.iter().map(|v| v.equality_key()).collect();
        format!(
            "{}({}; {})",
            if self.caching {
                "CachingExpressionValueSource"
            } else {
                "ExpressionValueSource"
            },
            self.expression.source_text,
            variables.join(", ")
        )
    }

    fn rewrite(
        &self,
        top: &crate::function::TopLevel<'_>,
    ) -> crate::Result<Option<Arc<dyn DoubleValuesSource>>> {
        let mut changed = false;
        let mut rewritten = Vec::with_capacity(self.variables.len());
        for v in &self.variables {
            match v.rewrite(top)? {
                Some(r) => {
                    changed = true;
                    rewritten.push(r);
                }
                None => rewritten.push(v.clone()),
            }
        }
        Ok(changed.then(|| {
            Arc::new(ExpressionValueSource {
                variables: rewritten,
                expression: self.expression.clone(),
                needs_scores: self.needs_scores,
                caching: self.caching,
            }) as Arc<dyn DoubleValuesSource>
        }))
    }
}

/// `ExpressionRescorer`: the first pass's hits sorted, descending, by the
/// expression.
pub struct ExpressionRescorer {
    sort: SortRescorer,
    expression: Arc<Expression>,
    source: Arc<dyn DoubleValuesSource>,
}

impl ExpressionRescorer {
    /// `new ExpressionRescorer(expression, bindings)`.
    ///
    /// # Errors
    /// A variable `bindings` does not bind.
    pub fn new(expression: Arc<Expression>, bindings: &dyn Bindings) -> crate::Result<Self> {
        let source = expression.double_values_source(bindings)?;
        let field = double_sort_field(source.clone(), true, 0.0);
        Ok(Self {
            sort: SortRescorer::new(vec![field]),
            expression,
            source,
        })
    }

    /// The expression rescored by.
    pub fn expression(&self) -> &Arc<Expression> {
        &self.expression
    }
}

impl Rescorer for ExpressionRescorer {
    fn rescore(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &TopDocs,
        top_n: usize,
    ) -> crate::Result<TopDocs> {
        self.sort.rescore(ctx, first_pass, top_n)
    }

    fn explain(
        &self,
        ctx: &ValuesContext<'_>,
        first_pass: &Explanation,
        doc: i32,
    ) -> crate::Result<Explanation> {
        let super_explanation = self.sort.explain(ctx, first_pass, doc)?;
        let searcher = ctx.searcher()?;
        let leaf = searcher.segment_of(doc).ok_or_else(|| {
            crate::Error::IllegalState(format!(
                "hit docId={doc} is not in any leaf of the searcher"
            ))
        })?;
        let in_segment = doc - searcher.segments()[leaf].doc_base;
        self.source.explain(
            &ValuesContext::new(searcher),
            leaf,
            in_segment,
            &super_explanation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

    fn expr(src: &str) -> Arc<Expression> {
        Arc::new(compile(src).unwrap())
    }

    fn message(r: crate::Result<()>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn simple_bindings_resolve_sources_and_expressions() {
        let mut b = SimpleBindings::new();
        b.add_source("a", constant(2.0));
        b.add_expression("b", expr("a * 3"));
        assert_eq!(
            b.double_values_source("a").unwrap().describe(),
            "constant(2.0)"
        );
        assert_eq!(
            b.double_values_source("b").unwrap().describe(),
            "expr(a * 3)"
        );
        assert_eq!(
            b.double_values_source("zzz").err().unwrap().to_string(),
            "Invalid reference 'zzz'"
        );
        b.validate().unwrap();
    }

    #[test]
    fn expression_over_an_unbound_variable_is_refused() {
        let b = SimpleBindings::new();
        let e = expr("missing + 1");
        assert_eq!(
            e.double_values_source(&b).err().unwrap().to_string(),
            "Invalid reference 'missing'"
        );
    }

    #[test]
    fn validate_reports_cycles_and_dangling_references() {
        let mut b = SimpleBindings::new();
        b.add_expression("x", expr("y + 1"));
        b.add_expression("y", expr("x + 1"));
        assert_eq!(
            message(b.validate()),
            "Recursion error: Cycle detected [x, y]->x"
        );
        // never validated: the cycle is still an error, not a stack overflow
        assert_eq!(
            message(b.double_values_source("y").map(|_| ())),
            "Recursion error: Cycle detected [y, x]->y"
        );
        let source = ExpressionValueSource::new(&b, expr("x * 2"));
        assert!(message(source.map(|_| ())).starts_with("Recursion error"));

        let mut b = SimpleBindings::new();
        b.add_expression("x", expr("nowhere"));
        assert_eq!(message(b.validate()), "Invalid reference 'nowhere'");

        let mut b = SimpleBindings::new();
        b.add_source("leaf", constant(1.0));
        b.add_expression("x", expr("leaf + leaf"));
        b.add_expression("y", expr("x + leaf"));
        b.validate().unwrap();
    }

    #[test]
    fn a_source_needs_scores_when_a_variable_does() {
        let mut b = SimpleBindings::new();
        b.add_source("s", crate::values_source::scores());
        b.add_source("c", constant(1.0));
        let with = ExpressionValueSource::new(&b, expr("s + c")).unwrap();
        let without = ExpressionValueSource::new(&b, expr("c * 2")).unwrap();
        assert!(with.needs_scores());
        assert!(!without.needs_scores());
        assert_eq!(with.describe(), "expr(s + c)");
        let caching = ExpressionValueSource::caching(&b, expr("c")).unwrap();
        assert!(caching.as_expression().is_some());
    }

    /// Even documents have the value `doc`, odd ones none; each read of a
    /// value is counted, and the first read at `fail_at` fails.
    struct Probe {
        reads: Arc<AtomicUsize>,
        fail_at: i32,
        failed: Arc<AtomicBool>,
    }

    struct ProbeValues {
        reads: Arc<AtomicUsize>,
        fail_at: i32,
        failed: Arc<AtomicBool>,
        doc: i32,
    }

    impl DoubleValues for ProbeValues {
        fn advance_exact(&mut self, doc: i32) -> crate::Result<bool> {
            self.doc = doc;
            Ok(doc % 2 == 0)
        }
        fn double_value(&mut self) -> crate::Result<f64> {
            if self.doc == self.fail_at && !self.failed.swap(true, Relaxed) {
                return Err(crate::Error::IllegalState("probe".into()));
            }
            self.reads.fetch_add(1, Relaxed);
            Ok(f64::from(self.doc))
        }
    }

    impl DoubleValuesSource for Probe {
        fn get_values<'c>(
            &self,
            _ctx: &ValuesContext<'c>,
            _leaf: usize,
            _scores: Option<BoxDoubleValues<'c>>,
        ) -> crate::Result<BoxDoubleValues<'c>> {
            Ok(Box::new(ProbeValues {
                reads: self.reads.clone(),
                fail_at: self.fail_at,
                failed: self.failed.clone(),
                doc: -1,
            }))
        }
        fn needs_scores(&self) -> bool {
            false
        }
        fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
            true
        }
        fn describe(&self) -> String {
            "probe".into()
        }
    }

    #[test]
    fn variables_read_once_per_doc_zero_when_missing_and_retried_after_an_error() {
        let reads = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let mut b = SimpleBindings::new();
        b.add_source(
            "p",
            Arc::new(Probe {
                reads: reads.clone(),
                fail_at: 4,
                failed: failed.clone(),
            }),
        );
        b.add_expression("twice", expr("p + p"));
        let searcher = crate::index_searcher::IndexSearcher::new(&[], &[]).unwrap();
        let ctx = ValuesContext::new(&searcher);
        for caching in [false, true] {
            reads.store(0, Relaxed);
            failed.store(false, Relaxed);
            let outer = expr("p * p + twice + p");
            let source = if caching {
                ExpressionValueSource::caching(&b, outer).unwrap()
            } else {
                ExpressionValueSource::new(&b, outer).unwrap()
            };
            let mut v = source.get_values(&ctx, 0, None).unwrap();
            let mut got = Vec::new();
            for doc in 0..6 {
                assert!(v.advance_exact(doc).unwrap());
                match v.double_value() {
                    Ok(x) => got.push(x),
                    Err(e) => {
                        assert_eq!(doc, 4);
                        assert!(e.to_string().contains("probe"), "{e}");
                        // the same document read again succeeds
                        got.push(v.double_value().unwrap());
                    }
                }
                // a second read of the document is the evaluated value
                assert_eq!(v.double_value().unwrap(), got[got.len() - 1]);
            }
            let want: Vec<f64> = (0..6)
                .map(|d| {
                    if d % 2 == 0 {
                        f64::from(d * d + 2 * d + d)
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_eq!(got, want, "caching={caching}");
            // per even document, `p` is read once for the outer expression
            // and once for the nested one (whose values a caching source
            // shares, but each variable keeps its own per-document value)
            assert_eq!(reads.load(Relaxed), 6, "caching={caching}");
        }
    }
}
