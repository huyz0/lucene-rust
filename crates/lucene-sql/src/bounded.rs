//! A bound on what one scalar function call may produce.
//!
//! DataFusion evaluates scalar functions over whole batches (and folds
//! constant calls while planning), and a function such as `repeat` or `lpad`
//! turns a few bytes of input into any number of bytes of output, outside the
//! statement's memory accounting. Every function the session registers is
//! wrapped in [`Bounded`]: a call whose output holds a value longer than
//! `max_value` bytes, or more than `max_call` bytes in all, fails with a plain
//! error instead of growing the node; nested calls stay bounded because each
//! call's input was an earlier call's bounded output. The functions that build
//! a length from an argument (`repeat`, `lpad`, `rpad`, `array_repeat`) are
//! also checked before they allocate, from their arguments.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion::logical_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion::logical_expr::{
    ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature,
};
use datafusion::prelude::SessionContext;

/// The limits a statement's function calls are held to.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The longest value (a string, a binary or a list) a call may produce.
    pub max_value: usize,
    /// The most bytes a call may produce for one batch.
    pub max_call: usize,
}

impl Limits {
    /// Limits derived from the engine's memory limit.
    pub fn from_memory_limit(memory_limit: usize) -> Self {
        Self {
            max_value: (memory_limit / 32).max(1 << 20),
            max_call: (memory_limit / 4).max(1 << 20),
        }
    }
}

#[derive(Debug)]
struct Bounded {
    inner: Arc<ScalarUDF>,
    limits: Limits,
}

impl PartialEq for Bounded {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Bounded {}

impl std::hash::Hash for Bounded {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

fn too_big(name: &str, what: &str, bytes: u128, limit: usize) -> DataFusionError {
    DataFusionError::ResourcesExhausted(format!(
        "{name}() would produce {what} of {bytes} bytes, over the statement's limit of {limit} \
         bytes (plugins.accelerated.sql.memory_limit)"
    ))
}

fn max_int(v: &ColumnarValue) -> Option<i128> {
    match v {
        ColumnarValue::Scalar(s) => scalar_int(s),
        ColumnarValue::Array(a) => {
            let a = datafusion::arrow::compute::cast(a, &DataType::Int64).ok()?;
            let a = a.as_primitive::<datafusion::arrow::datatypes::Int64Type>();
            datafusion::arrow::compute::max(a).map(i128::from)
        }
    }
}

fn scalar_int(s: &ScalarValue) -> Option<i128> {
    let s = s.cast_to(&DataType::Int64).ok()?;
    match s {
        ScalarValue::Int64(v) => v.map(i128::from),
        _ => None,
    }
}

fn longest(a: &ArrayRef) -> usize {
    match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().iter().flatten().map(str::len).max(),
        DataType::LargeUtf8 => a.as_string::<i64>().iter().flatten().map(str::len).max(),
        DataType::Utf8View => a.as_string_view().iter().flatten().map(str::len).max(),
        DataType::Binary => a.as_binary::<i32>().iter().flatten().map(<[u8]>::len).max(),
        DataType::LargeBinary => a.as_binary::<i64>().iter().flatten().map(<[u8]>::len).max(),
        DataType::List(_) => a
            .as_list::<i32>()
            .value_offsets()
            .windows(2)
            .map(|w| usize::try_from(w[1] - w[0]).unwrap_or(0))
            .max(),
        _ => None,
    }
    .unwrap_or(0)
}

fn longest_value(v: &ColumnarValue) -> usize {
    match v {
        ColumnarValue::Array(a) => longest(a),
        ColumnarValue::Scalar(s) => match s {
            ScalarValue::Utf8(Some(x))
            | ScalarValue::LargeUtf8(Some(x))
            | ScalarValue::Utf8View(Some(x)) => x.len(),
            ScalarValue::Binary(Some(x)) | ScalarValue::LargeBinary(Some(x)) => x.len(),
            _ => s.to_array().map_or(0, |a| longest(&a)),
        },
    }
}

fn bytes_of(v: &ColumnarValue) -> usize {
    match v {
        ColumnarValue::Array(a) => a.get_array_memory_size(),
        ColumnarValue::Scalar(s) => s.size(),
    }
}

impl Bounded {
    /// Refuses a call of a function that builds a length from an argument
    /// before it allocates.
    fn precheck(&self, args: &ScalarFunctionArgs) -> Result<()> {
        let name = self.inner.name();
        let rows = args.number_rows.max(1) as u128;
        let (len, per_row) = match (name, args.args.as_slice()) {
            ("repeat", [s, n]) => (
                longest_value(s) as u128,
                max_int(n).unwrap_or(0).max(0) as u128,
            ),
            ("lpad" | "rpad", [_, n, ..]) => (1, max_int(n).unwrap_or(0).max(0) as u128),
            ("array_repeat", [e, n]) => (
                longest_value(e).max(8) as u128,
                max_int(n).unwrap_or(0).max(0) as u128,
            ),
            _ => return Ok(()),
        };
        let value = len.saturating_mul(per_row);
        if value > self.limits.max_value as u128 {
            return Err(too_big(name, "a value", value, self.limits.max_value));
        }
        let call = value.saturating_mul(rows);
        if call > self.limits.max_call as u128 {
            return Err(too_big(name, "a result", call, self.limits.max_call));
        }
        Ok(())
    }

    fn check(&self, out: &ColumnarValue) -> Result<()> {
        let name = self.inner.name();
        let value = longest_value(out);
        if value > self.limits.max_value {
            return Err(too_big(
                name,
                "a value",
                value as u128,
                self.limits.max_value,
            ));
        }
        let call = bytes_of(out);
        if call > self.limits.max_call {
            return Err(too_big(
                name,
                "a result",
                call as u128,
                self.limits.max_call,
            ));
        }
        Ok(())
    }
}

impl ScalarUDFImpl for Bounded {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn aliases(&self) -> &[String] {
        self.inner.aliases()
    }
    fn signature(&self) -> &Signature {
        self.inner.signature()
    }
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(arg_types)
    }
    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        self.inner.inner().with_updated_config(config).map(|u| {
            ScalarUDF::new_from_impl(Bounded {
                inner: Arc::new(u),
                limits: self.limits,
            })
        })
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        self.inner.return_field_from_args(args)
    }
    fn is_strict(&self) -> bool {
        self.inner.inner().is_strict()
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        self.precheck(&args)?;
        let out = self.inner.invoke_with_args(args)?;
        self.check(&out)?;
        Ok(out)
    }
    fn should_evaluate_const(&self, args: &[&ScalarValue]) -> bool {
        self.inner.inner().should_evaluate_const(args)
    }
    fn simplify(&self, args: Vec<Expr>, info: &SimplifyContext) -> Result<ExprSimplifyResult> {
        self.inner.simplify(args, info)
    }
    fn short_circuits(&self) -> bool {
        self.inner.short_circuits()
    }
    fn conditional_arguments<'a>(
        &self,
        args: &'a [Expr],
    ) -> Option<(Vec<&'a Expr>, Vec<&'a Expr>)> {
        self.inner.inner().conditional_arguments(args)
    }
    fn output_ordering(&self, inputs: &[ExprProperties]) -> Result<SortProperties> {
        self.inner.output_ordering(inputs)
    }
    fn preserves_lex_ordering(&self, inputs: &[ExprProperties]) -> Result<bool> {
        self.inner.preserves_lex_ordering(inputs)
    }
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        self.inner.coerce_types(arg_types)
    }
    fn documentation(&self) -> Option<&Documentation> {
        self.inner.documentation()
    }
}

/// Wraps every scalar function `ctx` has in [`Bounded`] with `limits`.
pub(crate) fn bound_functions(ctx: &SessionContext, limits: Limits) {
    let state = ctx.state();
    let mut seen = std::collections::HashSet::new();
    for udf in state.scalar_functions().values() {
        if !seen.insert(udf.name().to_string()) {
            continue;
        }
        ctx.register_udf(ScalarUDF::new_from_impl(Bounded {
            inner: Arc::clone(udf),
            limits,
        }));
    }
}
