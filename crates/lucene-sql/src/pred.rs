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
    /// `column IN ('a', 'b', ...)` on a key or keyword column: documents with
    /// any of the values.
    Any {
        /// Column index.
        column: usize,
        /// The terms.
        values: Vec<String>,
    },
    /// A comparison of a long column with an integer, as the inclusive range
    /// it selects (documents without a value never match, as in SQL).
    Range {
        /// Column index of a long column.
        column: usize,
        /// The lowest value selected.
        lower: i64,
        /// The highest value selected.
        upper: i64,
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
            Pred::Any { .. } | Pred::Range { .. } => self.clone(),
        }
    }

    /// The literal.
    pub fn literal(&self) -> &str {
        match self {
            Pred::Term { value, .. } => value,
            Pred::Match { text, .. } => text,
            Pred::Knn { vector, .. } => vector,
            Pred::Any { .. } | Pred::Range { .. } => "",
        }
    }

    /// Whether the predicate scores documents.
    pub fn is_scoring(&self) -> bool {
        matches!(self, Pred::Match { .. } | Pred::Knn { .. })
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

fn literal_i64(e: &Expr) -> Option<i64> {
    use datafusion::scalar::ScalarValue as V;
    let Expr::Literal(v, _) = e else { return None };
    match v {
        V::Int8(Some(x)) => Some(i64::from(*x)),
        V::Int16(Some(x)) => Some(i64::from(*x)),
        V::Int32(Some(x)) => Some(i64::from(*x)),
        V::Int64(Some(x)) => Some(*x),
        V::UInt8(Some(x)) => Some(i64::from(*x)),
        V::UInt16(Some(x)) => Some(i64::from(*x)),
        V::UInt32(Some(x)) => Some(i64::from(*x)),
        V::UInt64(Some(x)) => i64::try_from(*x).ok(),
        _ => None,
    }
}

/// `column op literal` (or `literal op column`, mirrored) on a long column,
/// as the range it selects. An empty range is `lower > upper`.
fn parse_range(spec: &TableSpec, op: Operator, left: &Expr, right: &Expr) -> Option<Pred> {
    let (col, v, op) = match (left, right) {
        (c @ Expr::Column(_), l) => (c, literal_i64(l)?, op),
        (l, c @ Expr::Column(_)) => (c, literal_i64(l)?, op.swap()?),
        _ => return None,
    };
    let column = column_index(spec, col, |k| *k == ColumnKind::Long)?;
    let (lower, upper) = range_of(op, v)?;
    Some(Pred::Range {
        column,
        lower,
        upper,
    })
}

/// The inclusive range of the values `x` with `x op v`. An empty range is
/// `lower > upper`; `None` for an operator that is not a comparison.
pub fn range_of(op: Operator, v: i64) -> Option<(i64, i64)> {
    Some(match op {
        Operator::Eq => (v, v),
        Operator::Gt => (
            v.saturating_add(1),
            if v == i64::MAX { i64::MIN } else { i64::MAX },
        ),
        Operator::GtEq => (v, i64::MAX),
        Operator::Lt => (
            if v == i64::MIN { i64::MAX } else { i64::MIN },
            v.saturating_sub(1),
        ),
        Operator::LtEq => (i64::MIN, v),
        _ => return None,
    })
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
        Expr::BinaryExpr(b)
            if matches!(
                b.op,
                Operator::Eq | Operator::Gt | Operator::GtEq | Operator::Lt | Operator::LtEq
            ) && parse_range(spec, b.op, &b.left, &b.right).is_some() =>
        {
            parse_range(spec, b.op, &b.left, &b.right)
        }
        Expr::InList(l) if !l.negated && !l.list.is_empty() => Some(Pred::Any {
            column: column_index(spec, &l.expr, ColumnKind::is_term_indexed)?,
            values: l.list.iter().map(literal_str).collect::<Option<Vec<_>>>()?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};

    fn spec() -> TableSpec {
        TableSpec::new("t")
            .key("id")
            .long("price")
            .keyword("cat")
            .text("body")
            .vector("vec", 2)
    }

    #[test]
    fn predicates_are_parsed_from_expressions() {
        let s = spec();
        let m = match_text_udf().call(vec![col("body"), lit("a b")]);
        assert_eq!(
            parse(&s, &m),
            Some(Pred::Match {
                column: 3,
                text: "a b".into()
            })
        );
        let k = knn_udf().call(vec![col("vec"), lit("1,2")]);
        let p = parse(&s, &k).unwrap();
        assert_eq!(
            p,
            Pred::Knn {
                column: 4,
                vector: "1,2".into()
            }
        );
        assert!(p.is_scoring());
        let t = col("cat").eq(lit("x"));
        let p = parse(&s, &t).unwrap();
        assert!(!p.is_scoring());
        // The literal may come first.
        assert_eq!(parse(&s, &lit("x").eq(col("cat"))), Some(p));
        // Not predicates: wrong column kind, unknown column, non-literal, other operator.
        assert!(parse(&s, &match_text_udf().call(vec![col("cat"), lit("a")])).is_none());
        assert!(parse(&s, &knn_udf().call(vec![col("body"), lit("1")])).is_none());
        assert!(parse(&s, &col("price").eq(lit("5"))).is_none());
        assert!(parse(&s, &col("nope").eq(lit("x"))).is_none());
        assert!(parse(&s, &col("cat").eq(col("id"))).is_none());
        assert!(parse(&s, &lit("a").eq(lit("b"))).is_none());
        assert!(parse(&s, &col("cat").gt(lit("x"))).is_none());
        assert!(parse(&s, &match_text_udf().call(vec![col("body"), col("id")])).is_none());
        assert!(parse(&s, &col("cat")).is_none());
    }

    #[test]
    fn vectors_parse_or_not() {
        assert_eq!(parse_vector("1, 2.5,-3"), Some(vec![1.0, 2.5, -3.0]));
        assert_eq!(parse_vector("1,x"), None);
        assert_eq!(parse_vector(""), None);
    }

    #[test]
    fn the_functions_refuse_to_be_evaluated() {
        use datafusion::arrow::datatypes::Field;
        let udf = match_text_udf();
        let args = ScalarFunctionArgs {
            args: vec![],
            arg_fields: vec![],
            number_rows: 0,
            return_field: std::sync::Arc::new(Field::new("r", DataType::Boolean, true)),
            config_options: std::sync::Arc::new(Default::default()),
        };
        let err = udf.invoke_with_args(args).unwrap_err().to_string();
        assert!(err.contains("match_text"), "{err}");
        assert_eq!(udf.return_type(&[]).unwrap(), DataType::Boolean);
    }
}
