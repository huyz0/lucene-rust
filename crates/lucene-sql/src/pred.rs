//! Predicates a scan absorbs: `column = 'x'` on a term-indexed column,
//! `match_text(column, 'words')` and `knn(column, '0.1,0.2,...')`. They are
//! never evaluated by DataFusion: the table provider reports them as exactly
//! pushed down and the scan answers them with the native searches.

use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, Expr, Operator, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};

use crate::spec::{ColumnKind, TableSpec};

/// A pushed-down predicate. Texts are kept raw (not analysed or parsed) so a
/// cached plan can bind a new literal into it.
#[derive(Debug, Clone, PartialEq)]
pub enum Pred {
    /// `column = 'value'` on a key or keyword column.
    Term {
        /// Column index.
        column: usize,
        /// The term.
        value: String,
    },
    /// `match_text(column, 'text')`.
    Match {
        /// Column index of a text column.
        column: usize,
        /// The text, before analysis.
        text: String,
    },
    /// `knn(column, 'v1,v2,...')`.
    Knn {
        /// Column index of a vector column.
        column: usize,
        /// The vector as written.
        vector: String,
    },
}

impl Pred {
    /// The predicate with `lit` as its literal.
    pub fn with_literal(&self, lit: &str) -> Pred {
        match self {
            Pred::Term { column, .. } => Pred::Term {
                column: *column,
                value: lit.to_string(),
            },
            Pred::Match { column, .. } => Pred::Match {
                column: *column,
                text: lit.to_string(),
            },
            Pred::Knn { column, .. } => Pred::Knn {
                column: *column,
                vector: lit.to_string(),
            },
        }
    }

    /// The literal.
    pub fn literal(&self) -> &str {
        match self {
            Pred::Term { value, .. } => value,
            Pred::Match { text, .. } => text,
            Pred::Knn { vector, .. } => vector,
        }
    }

    /// Whether the predicate scores documents.
    pub fn is_scoring(&self) -> bool {
        !matches!(self, Pred::Term { .. })
    }
}

/// Parses a vector written `0.1, 0.2, ...`.
pub fn parse_vector(text: &str) -> Option<Vec<f32>> {
    text.split(',').map(|x| x.trim().parse().ok()).collect()
}

fn literal_str(e: &Expr) -> Option<String> {
    match e {
        Expr::Literal(v, _) => v.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

fn column_index(spec: &TableSpec, e: &Expr, ok: impl Fn(&ColumnKind) -> bool) -> Option<usize> {
    let Expr::Column(c) = e else { return None };
    let i = spec.columns().iter().position(|col| col.name == c.name)?;
    ok(&spec.columns()[i].kind).then_some(i)
}

/// The predicate `e` is, when the scan can absorb it.
pub fn parse(spec: &TableSpec, e: &Expr) -> Option<Pred> {
    match e {
        Expr::ScalarFunction(f) if f.name() == "match_text" && f.args.len() == 2 => {
            Some(Pred::Match {
                column: column_index(spec, &f.args[0], |k| *k == ColumnKind::Text)?,
                text: literal_str(&f.args[1])?,
            })
        }
        Expr::ScalarFunction(f) if f.name() == "knn" && f.args.len() == 2 => Some(Pred::Knn {
            column: column_index(spec, &f.args[0], |k| matches!(k, ColumnKind::Vector { .. }))?,
            vector: literal_str(&f.args[1])?,
        }),
        Expr::BinaryExpr(b) if b.op == Operator::Eq => {
            let (col, lit) = match (&*b.left, &*b.right) {
                (c @ Expr::Column(_), l) | (l, c @ Expr::Column(_)) => (c, l),
                _ => return None,
            };
            Some(Pred::Term {
                column: column_index(spec, col, ColumnKind::is_term_indexed)?,
                value: literal_str(lit)?,
            })
        }
        _ => None,
    }
}

/// A function that is only ever pushed down to a scan.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Pushdown {
    name: &'static str,
    signature: Signature,
}

impl ScalarUDFImpl for Pushdown {
    fn name(&self) -> &str {
        self.name
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Plan(format!(
            "{}() can only be used as a top-level AND-ed predicate with a literal argument, \
             where the scan can answer it",
            self.name
        )))
    }
}

/// `match_text(body, 'w1 w2')`: a full-text predicate that scores.
pub fn match_text_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Pushdown {
        name: "match_text",
        signature: Signature::any(2, Volatility::Immutable),
    })
}

/// `knn(vec, '0.1,0.2,...')`: a nearest-neighbour predicate that scores.
pub fn knn_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Pushdown {
        name: "knn",
        signature: Signature::any(2, Volatility::Immutable),
    })
}
