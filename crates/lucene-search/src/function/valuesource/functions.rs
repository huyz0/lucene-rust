//! The functions of other sources: `SingleFunction`/`SimpleFloatFunction`,
//! `DualFloatFunction` (`DivFloatFunction`, `PowFloatFunction`),
//! `MultiFloatFunction` (`SumFloatFunction`, `ProductFloatFunction`,
//! `MaxFloatFunction`, `MinFloatFunction`), `MultiFunction`
//! (`DefFunction`), `LinearFloatFunction`, `ReciprocalFloatFunction`,
//! `RangeMapFloatFunction`, `ScaleFloatFunction`, and the boolean ones:
//! `BoolFunction`, `SimpleBoolFunction`, `MultiBoolFunction`,
//! `ComparisonBoolFunction`, `IfFunction`.
//!
//! Every `float` is computed in `float`, in Java's order of operations
//! (`Math.max`/`Math.min` with their `NaN` and signed-zero rules,
//! `(float) Math.pow(double, double)`).

use std::marker::PhantomData;
use std::sync::Arc;

use super::constants::ConstValueSource;
use crate::function::docvalues::{Bool, BoolDocValues, Float, FloatDocValues};
use crate::function::{
    java_float, not_weighted, BoxValues, FunctionContext, FunctionValues, MutableValue, ObjectVal,
    TopLevel, ValueLeaf, ValueSource,
};
use crate::Result;

/// `Math.max(float, float)`: `NaN` wins, and `0.0 > -0.0`.
pub(crate) fn java_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0f32).to_bits() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
}

/// `Math.min(float, float)`: `NaN` wins, and `-0.0 < 0.0`.
pub(crate) fn java_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0f32).to_bits() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `MultiFunction.allExists(doc, values)`.
pub(crate) fn all_exists(doc: i32, values: &mut [BoxValues<'_>]) -> Result<bool> {
    for v in values {
        if !v.exists(doc)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `MultiFunction.anyExists(doc, values)`.
pub(crate) fn any_exists(doc: i32, values: &mut [BoxValues<'_>]) -> Result<bool> {
    for v in values {
        if v.exists(doc)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `MultiFunction.toString(name, valsArr, doc)`.
fn values_to_string(name: &str, values: &mut [BoxValues<'_>], doc: i32) -> Result<String> {
    let mut parts = Vec::with_capacity(values.len());
    for v in values {
        parts.push(v.to_string_doc(doc)?);
    }
    Ok(format!("{name}({})", parts.join(",")))
}

/// `MultiFunction.description(name, sources)` (each source's `toString()`,
/// its description).
fn sources_description(name: &str, sources: &[Arc<dyn ValueSource>]) -> String {
    let parts: Vec<String> = sources.iter().map(|s| s.description()).collect();
    format!("{name}({})", parts.join(","))
}

fn values_of<'a>(
    sources: &[Arc<dyn ValueSource>],
    fcx: &FunctionContext,
    leaf: &ValueLeaf<'a>,
) -> Result<Vec<BoxValues<'a>>> {
    sources.iter().map(|s| s.get_values(fcx, leaf)).collect()
}

fn as_refs(sources: &[Arc<dyn ValueSource>]) -> Vec<&dyn ValueSource> {
    sources.iter().map(|s| s.as_ref()).collect()
}

// ---------------------------------------------------------------------------
// SingleFunction, SimpleFloatFunction
// ---------------------------------------------------------------------------

/// `SingleFunction`: a function of one source, described `name(source)`.
pub trait SingleFunction: ValueSource {
    /// `name()`.
    fn name(&self) -> &str;
    /// The source.
    fn source(&self) -> &Arc<dyn ValueSource>;
}

/// `SimpleFloatFunction.func(doc, vals)`.
pub type FloatFn = dyn Fn(i32, &mut dyn FunctionValues) -> Result<f32> + Send + Sync;

/// `SimpleFloatFunction`: a `float` function of one source's values (the
/// subclass's `name()` and `func`).
#[derive(Clone)]
pub struct SimpleFloatFunction {
    name: String,
    source: Arc<dyn ValueSource>,
    func: Arc<FloatFn>,
}

impl SimpleFloatFunction {
    pub fn new(name: impl Into<String>, source: Arc<dyn ValueSource>, func: Arc<FloatFn>) -> Self {
        Self {
            name: name.into(),
            source,
            func,
        }
    }
}

struct SimpleFloatValues<'a> {
    vals: BoxValues<'a>,
    func: Arc<FloatFn>,
    name: String,
}

impl FloatDocValues for SimpleFloatValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        (self.func)(doc, self.vals.as_mut())
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!("{}({})", self.name, self.vals.to_string_doc(doc)?))
    }
}

impl ValueSource for SimpleFloatFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(SimpleFloatValues {
            vals: self.source.get_values(fcx, leaf)?,
            func: Arc::clone(&self.func),
            name: self.name.clone(),
        })))
    }
    fn description(&self) -> String {
        format!("{}({})", self.name, self.source.description())
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.source.as_ref()]
    }
}

impl SingleFunction for SimpleFloatFunction {
    fn name(&self) -> &str {
        &self.name
    }
    fn source(&self) -> &Arc<dyn ValueSource> {
        &self.source
    }
}

// ---------------------------------------------------------------------------
// DualFloatFunction
// ---------------------------------------------------------------------------

/// `DualFloatFunction`'s hooks: `name()` and `func(doc, aVals, bVals)` over
/// the two values' `floatVal`s.
pub trait DualFloatOp: Send + Sync + 'static {
    const NAME: &'static str;
    fn func(a: f32, b: f32) -> f32;
}

/// `DualFloatFunction`: a `float` function of two sources; a document has
/// a value when both do.
pub struct DualFloatFunction<F: DualFloatOp> {
    a: Arc<dyn ValueSource>,
    b: Arc<dyn ValueSource>,
    op: PhantomData<F>,
}

impl<F: DualFloatOp> DualFloatFunction<F> {
    pub fn new(a: Arc<dyn ValueSource>, b: Arc<dyn ValueSource>) -> Self {
        Self {
            a,
            b,
            op: PhantomData,
        }
    }
}

/// `DivFloatFunction`'s `func`: `a / b`.
pub struct Div;

impl DualFloatOp for Div {
    const NAME: &'static str = "div";
    fn func(a: f32, b: f32) -> f32 {
        a / b
    }
}

/// `PowFloatFunction`'s `func`: `(float) Math.pow(a, b)`.
pub struct Pow;

impl DualFloatOp for Pow {
    const NAME: &'static str = "pow";
    fn func(a: f32, b: f32) -> f32 {
        f64::from(a).powf(f64::from(b)) as f32
    }
}

/// `DivFloatFunction`: `a / b`.
pub type DivFloatFunction = DualFloatFunction<Div>;
/// `PowFloatFunction`: `(float) Math.pow(a, b)`.
pub type PowFloatFunction = DualFloatFunction<Pow>;

struct DualValues<'a, F> {
    a: BoxValues<'a>,
    b: BoxValues<'a>,
    op: PhantomData<F>,
}

impl<F: DualFloatOp> FloatDocValues for DualValues<'_, F> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        let a = self.a.float_val(doc)?;
        let b = self.b.float_val(doc)?;
        Ok(F::func(a, b))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        Ok(self.a.exists(doc)? && self.b.exists(doc)?)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "{}({},{})",
            F::NAME,
            self.a.to_string_doc(doc)?,
            self.b.to_string_doc(doc)?
        ))
    }
}

impl<F: DualFloatOp> ValueSource for DualFloatFunction<F> {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(DualValues::<F> {
            a: self.a.get_values(fcx, leaf)?,
            b: self.b.get_values(fcx, leaf)?,
            op: PhantomData,
        })))
    }
    fn description(&self) -> String {
        format!(
            "{}({},{})",
            F::NAME,
            self.a.description(),
            self.b.description()
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.a.as_ref(), self.b.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// MultiFloatFunction
// ---------------------------------------------------------------------------

/// `MultiFloatFunction`'s hooks: `name()`, `func(doc, valsArr)` and
/// `exists(doc, valsArr)` (by default every source's).
pub trait MultiFloatOp: Send + Sync + 'static {
    const NAME: &'static str;
    fn func(doc: i32, values: &mut [BoxValues<'_>]) -> Result<f32>;
    fn exists(doc: i32, values: &mut [BoxValues<'_>]) -> Result<bool> {
        all_exists(doc, values)
    }
}

/// `MultiFloatFunction`: a `float` function of several sources.
pub struct MultiFloatFunction<F: MultiFloatOp> {
    sources: Vec<Arc<dyn ValueSource>>,
    op: PhantomData<F>,
}

impl<F: MultiFloatOp> MultiFloatFunction<F> {
    pub fn new(sources: Vec<Arc<dyn ValueSource>>) -> Self {
        Self {
            sources,
            op: PhantomData,
        }
    }
}

/// `SumFloatFunction`'s `func`: the `float` sum, in order.
pub struct Sum;

impl MultiFloatOp for Sum {
    const NAME: &'static str = "sum";
    fn func(doc: i32, values: &mut [BoxValues<'_>]) -> Result<f32> {
        let mut val = 0.0f32;
        for v in values {
            val += v.float_val(doc)?;
        }
        Ok(val)
    }
}

/// `ProductFloatFunction`'s `func`: the `float` product, in order.
pub struct Product;

impl MultiFloatOp for Product {
    const NAME: &'static str = "product";
    fn func(doc: i32, values: &mut [BoxValues<'_>]) -> Result<f32> {
        let mut val = 1.0f32;
        for v in values {
            val *= v.float_val(doc)?;
        }
        Ok(val)
    }
}

/// `MaxFloatFunction`'s `func`: the greatest value of the sources that
/// have one (`0` when none does); a document has a value when any source
/// does.
pub struct Max;

impl MultiFloatOp for Max {
    const NAME: &'static str = "max";
    fn func(doc: i32, values: &mut [BoxValues<'_>]) -> Result<f32> {
        let mut none_found = true;
        let mut val = f32::NEG_INFINITY;
        for v in values {
            if v.exists(doc)? {
                none_found = false;
                val = java_max_f32(v.float_val(doc)?, val);
            }
        }
        Ok(if none_found { 0.0 } else { val })
    }
    fn exists(doc: i32, values: &mut [BoxValues<'_>]) -> Result<bool> {
        any_exists(doc, values)
    }
}

/// `MinFloatFunction`'s `func`.
pub struct Min;

impl MultiFloatOp for Min {
    const NAME: &'static str = "min";
    fn func(doc: i32, values: &mut [BoxValues<'_>]) -> Result<f32> {
        let mut none_found = true;
        let mut val = f32::INFINITY;
        for v in values {
            if v.exists(doc)? {
                none_found = false;
                val = java_min_f32(v.float_val(doc)?, val);
            }
        }
        Ok(if none_found { 0.0 } else { val })
    }
    fn exists(doc: i32, values: &mut [BoxValues<'_>]) -> Result<bool> {
        any_exists(doc, values)
    }
}

/// `SumFloatFunction`.
pub type SumFloatFunction = MultiFloatFunction<Sum>;
/// `ProductFloatFunction`.
pub type ProductFloatFunction = MultiFloatFunction<Product>;
/// `MaxFloatFunction`.
pub type MaxFloatFunction = MultiFloatFunction<Max>;
/// `MinFloatFunction`.
pub type MinFloatFunction = MultiFloatFunction<Min>;

struct MultiFloatValues<'a, F> {
    values: Vec<BoxValues<'a>>,
    op: PhantomData<F>,
}

impl<F: MultiFloatOp> FloatDocValues for MultiFloatValues<'_, F> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        F::func(doc, &mut self.values)
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        F::exists(doc, &mut self.values)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        values_to_string(F::NAME, &mut self.values, doc)
    }
}

impl<F: MultiFloatOp> ValueSource for MultiFloatFunction<F> {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(MultiFloatValues::<F> {
            values: values_of(&self.sources, fcx, leaf)?,
            op: PhantomData,
        })))
    }
    fn description(&self) -> String {
        sources_description(F::NAME, &self.sources)
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        as_refs(&self.sources)
    }
}

// ---------------------------------------------------------------------------
// MultiFunction, DefFunction
// ---------------------------------------------------------------------------

/// `MultiFunction`: a function of several sources (its `Values` described
/// `name(v1,v2,...)`).
pub trait MultiFunction: ValueSource {
    /// `name()`.
    fn name(&self) -> &str;
    /// `sources`.
    fn function_sources(&self) -> &[Arc<dyn ValueSource>];
}

/// `DefFunction`: the first source with a value for the document (the
/// last source when none has one).
#[derive(Clone)]
pub struct DefFunction {
    sources: Vec<Arc<dyn ValueSource>>,
}

impl DefFunction {
    pub fn new(sources: Vec<Arc<dyn ValueSource>>) -> Self {
        Self { sources }
    }
}

struct DefValues<'a> {
    values: Vec<BoxValues<'a>>,
}

impl DefValues<'_> {
    /// `get(doc)`: the index of the values answering for `doc`; with no
    /// sources, Java's `ArrayIndexOutOfBoundsException` (index `-1`).
    fn get(&mut self, doc: i32) -> Result<usize> {
        let Some(upto) = self.values.len().checked_sub(1) else {
            return Err(crate::Error::IllegalArgument(
                "def() has no sources: index -1 out of bounds for length 0".into(),
            ));
        };
        for i in 0..upto {
            if self.values[i].exists(doc)? {
                return Ok(i);
            }
        }
        Ok(upto)
    }
}

macro_rules! def_get {
    ($($m:ident -> $t:ty),*) => {
        $(
            fn $m(&mut self, doc: i32) -> Result<$t> {
                let i = self.get(doc)?;
                self.values[i].$m(doc)
            }
        )*
    };
}

impl FunctionValues for DefValues<'_> {
    def_get!(
        byte_val -> i8,
        short_val -> i16,
        float_val -> f32,
        int_val -> i32,
        long_val -> i64,
        double_val -> f64,
        str_val -> Option<String>,
        bool_val -> bool,
        object_val -> ObjectVal
    );
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let i = self.get(doc)?;
        self.values[i].bytes_val(doc, target)
    }
    /// Whether any source has a value.
    fn exists(&mut self, doc: i32) -> Result<bool> {
        any_exists(doc, &mut self.values)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        values_to_string("def", &mut self.values, doc)
    }
}

impl ValueSource for DefFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(DefValues {
            values: values_of(&self.sources, fcx, leaf)?,
        }))
    }
    fn description(&self) -> String {
        sources_description("def", &self.sources)
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        as_refs(&self.sources)
    }
}

impl MultiFunction for DefFunction {
    fn name(&self) -> &str {
        "def"
    }
    fn function_sources(&self) -> &[Arc<dyn ValueSource>] {
        &self.sources
    }
}

// ---------------------------------------------------------------------------
// LinearFloatFunction, ReciprocalFloatFunction, RangeMapFloatFunction
// ---------------------------------------------------------------------------

/// `LinearFloatFunction`: `slope * x + intercept`.
#[derive(Clone)]
pub struct LinearFloatFunction {
    source: Arc<dyn ValueSource>,
    slope: f32,
    intercept: f32,
}

impl LinearFloatFunction {
    pub fn new(source: Arc<dyn ValueSource>, slope: f32, intercept: f32) -> Self {
        Self {
            source,
            slope,
            intercept,
        }
    }
}

struct LinearValues<'a> {
    vals: BoxValues<'a>,
    slope: f32,
    intercept: f32,
}

impl FloatDocValues for LinearValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self.vals.float_val(doc)? * self.slope + self.intercept)
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.vals.exists(doc)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "{}*float({})+{}",
            java_float(self.slope),
            self.vals.to_string_doc(doc)?,
            java_float(self.intercept)
        ))
    }
}

impl ValueSource for LinearFloatFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(LinearValues {
            vals: self.source.get_values(fcx, leaf)?,
            slope: self.slope,
            intercept: self.intercept,
        })))
    }
    fn description(&self) -> String {
        format!(
            "{}*float({})+{}",
            java_float(self.slope),
            self.source.description(),
            java_float(self.intercept)
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.source.as_ref()]
    }
}

/// `ReciprocalFloatFunction`: `a / (m * x + b)`.
#[derive(Clone)]
pub struct ReciprocalFloatFunction {
    source: Arc<dyn ValueSource>,
    m: f32,
    a: f32,
    b: f32,
}

impl ReciprocalFloatFunction {
    pub fn new(source: Arc<dyn ValueSource>, m: f32, a: f32, b: f32) -> Self {
        Self { source, m, a, b }
    }
}

struct ReciprocalValues<'a> {
    vals: BoxValues<'a>,
    m: f32,
    a: f32,
    b: f32,
}

impl FloatDocValues for ReciprocalValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self.a / (self.m * self.vals.float_val(doc)? + self.b))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.vals.exists(doc)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "{}/({}*float({})+{})",
            java_float(self.a),
            java_float(self.m),
            self.vals.to_string_doc(doc)?,
            java_float(self.b)
        ))
    }
}

impl ValueSource for ReciprocalFloatFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(ReciprocalValues {
            vals: self.source.get_values(fcx, leaf)?,
            m: self.m,
            a: self.a,
            b: self.b,
        })))
    }
    fn description(&self) -> String {
        format!(
            "{}/({}*float({})+{})",
            java_float(self.a),
            java_float(self.m),
            self.source.description(),
            java_float(self.b)
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.source.as_ref()]
    }
}

/// `RangeMapFloatFunction`: a value in `[min, max]` maps to the target's
/// value, any other to the default's (or stays itself without one).
#[derive(Clone)]
pub struct RangeMapFloatFunction {
    source: Arc<dyn ValueSource>,
    min: f32,
    max: f32,
    target: Arc<dyn ValueSource>,
    default_val: Option<Arc<dyn ValueSource>>,
}

impl RangeMapFloatFunction {
    /// `new RangeMapFloatFunction(source, min, max, ValueSource target,
    /// ValueSource def)`.
    pub fn new(
        source: Arc<dyn ValueSource>,
        min: f32,
        max: f32,
        target: Arc<dyn ValueSource>,
        default_val: Option<Arc<dyn ValueSource>>,
    ) -> Self {
        Self {
            source,
            min,
            max,
            target,
            default_val,
        }
    }

    /// `new RangeMapFloatFunction(source, min, max, float target, Float
    /// def)`: constant target and default.
    pub fn with_constants(
        source: Arc<dyn ValueSource>,
        min: f32,
        max: f32,
        target: f32,
        default_val: Option<f32>,
    ) -> Self {
        Self::new(
            source,
            min,
            max,
            Arc::new(ConstValueSource::new(target)),
            default_val.map(|d| Arc::new(ConstValueSource::new(d)) as Arc<dyn ValueSource>),
        )
    }
}

struct RangeMapValues<'a> {
    vals: BoxValues<'a>,
    targets: BoxValues<'a>,
    defaults: Option<BoxValues<'a>>,
    min: f32,
    max: f32,
}

impl FloatDocValues for RangeMapValues<'_> {
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        let val = self.vals.float_val(doc)?;
        if val >= self.min && val <= self.max {
            return self.targets.float_val(doc);
        }
        match &mut self.defaults {
            None => Ok(val),
            Some(d) => d.float_val(doc),
        }
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let defaults = match &mut self.defaults {
            None => "null".to_string(),
            Some(d) => d.to_string_doc(doc)?,
        };
        Ok(format!(
            "map({},min={},max={},target={},defaultVal={defaults})",
            self.vals.to_string_doc(doc)?,
            java_float(self.min),
            java_float(self.max),
            self.targets.to_string_doc(doc)?
        ))
    }
}

impl ValueSource for RangeMapFloatFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Float(RangeMapValues {
            vals: self.source.get_values(fcx, leaf)?,
            targets: self.target.get_values(fcx, leaf)?,
            defaults: self
                .default_val
                .as_ref()
                .map(|d| d.get_values(fcx, leaf))
                .transpose()?,
            min: self.min,
            max: self.max,
        })))
    }
    fn description(&self) -> String {
        format!(
            "map({},{},{},{},{})",
            self.source.description(),
            java_float(self.min),
            java_float(self.max),
            self.target.description(),
            self.default_val
                .as_ref()
                .map_or_else(|| "null".to_string(), |d| d.description())
        )
    }
    /// Only the mapped source's (Java's `createWeight` skips the target and
    /// the default).
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        self.source.create_weight(fcx, top)
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        let mut v = vec![self.source.as_ref(), self.target.as_ref()];
        if let Some(d) = &self.default_val {
            v.push(d.as_ref());
        }
        v
    }
}

// ---------------------------------------------------------------------------
// ScaleFloatFunction
// ---------------------------------------------------------------------------

/// `ScaleFloatFunction`: values mapped linearly from the source's
/// reader-wide `[min, max]` (over every document of every leaf, deleted
/// ones included; infinities and `NaN` skipped) onto `[min, max]`.
#[derive(Clone)]
pub struct ScaleFloatFunction {
    source: Arc<dyn ValueSource>,
    min: f32,
    max: f32,
}

impl ScaleFloatFunction {
    pub fn new(source: Arc<dyn ValueSource>, min: f32, max: f32) -> Self {
        Self { source, min, max }
    }
}

/// `ScaleInfo`.
#[derive(Debug, Clone, Copy)]
struct ScaleInfo {
    min_val: f32,
    max_val: f32,
}

struct ScaleValues<'a> {
    vals: BoxValues<'a>,
    scale: f32,
    min_source: f32,
    max_source: f32,
    min: f32,
    max: f32,
}

impl FloatDocValues for ScaleValues<'_> {
    fn exists(&mut self, doc: i32) -> Result<bool> {
        self.vals.exists(doc)
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok((self.vals.float_val(doc)? - self.min_source) * self.scale + self.min)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "scale({},toMin={},toMax={},fromMin={},fromMax={})",
            self.vals.to_string_doc(doc)?,
            java_float(self.min),
            java_float(self.max),
            java_float(self.min_source),
            java_float(self.max_source)
        ))
    }
}

impl ValueSource for ScaleFloatFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        let info = *fcx
            .get::<ScaleInfo, _>(self)
            .ok_or_else(|| not_weighted(&self.description()))?;
        let scale = if info.max_val - info.min_val == 0.0 {
            0.0
        } else {
            (self.max - self.min) / (info.max_val - info.min_val)
        };
        Ok(Box::new(Float(ScaleValues {
            vals: self.source.get_values(fcx, leaf)?,
            scale,
            min_source: info.min_val,
            max_source: info.max_val,
            min: self.min,
            max: self.max,
        })))
    }
    fn description(&self) -> String {
        format!(
            "scale({},{},{})",
            self.source.description(),
            java_float(self.min),
            java_float(self.max)
        )
    }
    /// The source's, then `createScaleInfo` (Java's runs on the first
    /// `getValues`, over the same context).
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        self.source.create_weight(fcx, top)?;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for leaf in top.leaves() {
            let max_doc = leaf.max_doc()?;
            let mut vals = self.source.get_values(fcx, &leaf)?;
            for i in 0..max_doc {
                if !vals.exists(i)? {
                    continue;
                }
                let val = vals.float_val(i)?;
                // An exponent of all ones: an infinity or `NaN`.
                if (val.to_bits() & (0xff << 23)) == 0xff << 23 {
                    continue;
                }
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
        }
        if min_val == f32::INFINITY {
            min_val = 0.0;
            max_val = 0.0;
        }
        fcx.put(self, ScaleInfo { min_val, max_val });
        Ok(())
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.source.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// Boolean functions
// ---------------------------------------------------------------------------

/// `BoolFunction`: a source of boolean values (a marker, as in Java).
pub trait BoolFunction: ValueSource {}

/// `SimpleBoolFunction.func(doc, vals)`.
pub type BoolFn = dyn Fn(i32, &mut dyn FunctionValues) -> Result<bool> + Send + Sync;

/// `SimpleBoolFunction`: a boolean function of one source's values.
#[derive(Clone)]
pub struct SimpleBoolFunction {
    name: String,
    source: Arc<dyn ValueSource>,
    func: Arc<BoolFn>,
}

impl SimpleBoolFunction {
    pub fn new(name: impl Into<String>, source: Arc<dyn ValueSource>, func: Arc<BoolFn>) -> Self {
        Self {
            name: name.into(),
            source,
            func,
        }
    }
}

struct SimpleBoolValues<'a> {
    vals: BoxValues<'a>,
    func: Arc<BoolFn>,
    name: String,
}

impl BoolDocValues for SimpleBoolValues<'_> {
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        (self.func)(doc, self.vals.as_mut())
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!("{}({})", self.name, self.vals.to_string_doc(doc)?))
    }
}

impl ValueSource for SimpleBoolFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Bool(SimpleBoolValues {
            vals: self.source.get_values(fcx, leaf)?,
            func: Arc::clone(&self.func),
            name: self.name.clone(),
        })))
    }
    fn description(&self) -> String {
        format!("{}({})", self.name, self.source.description())
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.source.as_ref()]
    }
}

impl BoolFunction for SimpleBoolFunction {}

/// `MultiBoolFunction.func(doc, vals)`.
pub type MultiBoolFn = dyn Fn(i32, &mut [BoxValues<'_>]) -> Result<bool> + Send + Sync;

/// `MultiBoolFunction`: a boolean function of several sources' values.
#[derive(Clone)]
pub struct MultiBoolFunction {
    name: String,
    sources: Vec<Arc<dyn ValueSource>>,
    func: Arc<MultiBoolFn>,
}

impl MultiBoolFunction {
    pub fn new(
        name: impl Into<String>,
        sources: Vec<Arc<dyn ValueSource>>,
        func: Arc<MultiBoolFn>,
    ) -> Self {
        Self {
            name: name.into(),
            sources,
            func,
        }
    }
}

struct MultiBoolValues<'a> {
    values: Vec<BoxValues<'a>>,
    func: Arc<MultiBoolFn>,
    name: String,
}

impl BoolDocValues for MultiBoolValues<'_> {
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        (self.func)(doc, &mut self.values)
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        values_to_string(&self.name, &mut self.values, doc)
    }
}

impl ValueSource for MultiBoolFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Bool(MultiBoolValues {
            values: values_of(&self.sources, fcx, leaf)?,
            func: Arc::clone(&self.func),
            name: self.name.clone(),
        })))
    }
    fn description(&self) -> String {
        sources_description(&self.name, &self.sources)
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        as_refs(&self.sources)
    }
}

impl BoolFunction for MultiBoolFunction {}

/// `ComparisonBoolFunction.compare(doc, lhs, rhs)`.
pub type CompareFn =
    dyn Fn(i32, &mut dyn FunctionValues, &mut dyn FunctionValues) -> Result<bool> + Send + Sync;

/// `ComparisonBoolFunction`: a comparison of two sources' values; a
/// document has a value when both do.
#[derive(Clone)]
pub struct ComparisonBoolFunction {
    lhs: Arc<dyn ValueSource>,
    rhs: Arc<dyn ValueSource>,
    name: String,
    compare: Arc<CompareFn>,
}

impl ComparisonBoolFunction {
    pub fn new(
        lhs: Arc<dyn ValueSource>,
        rhs: Arc<dyn ValueSource>,
        name: impl Into<String>,
        compare: Arc<CompareFn>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            name: name.into(),
            compare,
        }
    }
}

struct ComparisonValues<'a> {
    lhs: BoxValues<'a>,
    rhs: BoxValues<'a>,
    compare: Arc<CompareFn>,
    name: String,
}

impl BoolDocValues for ComparisonValues<'_> {
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        (self.compare)(doc, self.lhs.as_mut(), self.rhs.as_mut())
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "{}({},{})",
            self.name,
            self.lhs.to_string_doc(doc)?,
            self.rhs.to_string_doc(doc)?
        ))
    }
    fn exists(&mut self, doc: i32) -> Result<bool> {
        Ok(self.lhs.exists(doc)? && self.rhs.exists(doc)?)
    }
}

impl ValueSource for ComparisonBoolFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(Bool(ComparisonValues {
            lhs: self.lhs.get_values(fcx, leaf)?,
            rhs: self.rhs.get_values(fcx, leaf)?,
            compare: Arc::clone(&self.compare),
            name: self.name.clone(),
        })))
    }
    fn description(&self) -> String {
        format!(
            "{}({},{})",
            self.name,
            self.lhs.description(),
            self.rhs.description()
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![self.lhs.as_ref(), self.rhs.as_ref()]
    }
}

impl BoolFunction for ComparisonBoolFunction {}

/// `IfFunction`: the true source's value where the condition's `boolVal`
/// holds, else the false source's.
#[derive(Clone)]
pub struct IfFunction {
    if_source: Arc<dyn ValueSource>,
    true_source: Arc<dyn ValueSource>,
    false_source: Arc<dyn ValueSource>,
}

impl IfFunction {
    pub fn new(
        if_source: Arc<dyn ValueSource>,
        true_source: Arc<dyn ValueSource>,
        false_source: Arc<dyn ValueSource>,
    ) -> Self {
        Self {
            if_source,
            true_source,
            false_source,
        }
    }
}

struct IfValues<'a> {
    ifv: BoxValues<'a>,
    t: BoxValues<'a>,
    f: BoxValues<'a>,
}

macro_rules! if_get {
    ($($m:ident -> $ty:ty),*) => {
        $(
            fn $m(&mut self, doc: i32) -> Result<$ty> {
                if self.ifv.bool_val(doc)? {
                    self.t.$m(doc)
                } else {
                    self.f.$m(doc)
                }
            }
        )*
    };
}

impl FunctionValues for IfValues<'_> {
    if_get!(
        byte_val -> i8,
        short_val -> i16,
        float_val -> f32,
        int_val -> i32,
        long_val -> i64,
        double_val -> f64,
        str_val -> Option<String>,
        bool_val -> bool,
        object_val -> ObjectVal,
        exists -> bool
    );
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        if self.ifv.bool_val(doc)? {
            self.t.bytes_val(doc, target)
        } else {
            self.f.bytes_val(doc, target)
        }
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        Ok(format!(
            "if({},{},{})",
            self.ifv.to_string_doc(doc)?,
            self.t.to_string_doc(doc)?,
            self.f.to_string_doc(doc)?
        ))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::float()
    }
}

impl ValueSource for IfFunction {
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>> {
        Ok(Box::new(IfValues {
            ifv: self.if_source.get_values(fcx, leaf)?,
            t: self.true_source.get_values(fcx, leaf)?,
            f: self.false_source.get_values(fcx, leaf)?,
        }))
    }
    fn description(&self) -> String {
        format!(
            "if({},{},{})",
            self.if_source.description(),
            self.true_source.description(),
            self.false_source.description()
        )
    }
    fn sources(&self) -> Vec<&dyn ValueSource> {
        vec![
            self.if_source.as_ref(),
            self.true_source.as_ref(),
            self.false_source.as_ref(),
        ]
    }
}

impl BoolFunction for IfFunction {}
