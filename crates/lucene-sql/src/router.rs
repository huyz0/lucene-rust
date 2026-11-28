//! The fast-path router: statements of four hot shapes are recognised from
//! their parse tree and answered by the native API directly; everything else
//! (and anything the router is not sure of) is left to DataFusion, which gives
//! the same rows more slowly.
//!
//! Recognition matches the tree's nodes, never the statement's text, so
//! spacing, case and redundant parentheses do not matter and a near miss (an
//! extra predicate, an alias, a different projection) is not mistaken for a
//! shape. The tree is built once per *template*: the statement with its quoted
//! literals cut out ([`templatize`]); a hit binds the statement's own literal
//! into the cached shape.
//!
//! | shape | statement |
//! |---|---|
//! | point read | `SELECT <columns> FROM t WHERE <key or keyword> = '<id>'` |
//! | text top-k | `SELECT <columns, _score> FROM t WHERE match_text(<text>, '<q>') ORDER BY _score DESC LIMIT k` |
//! | k-NN top-k | `SELECT <columns, _score> FROM t WHERE knn(<vector>, '<v>') ORDER BY _score DESC LIMIT k` |
//! | group-by | `SELECT <keyword>, count(*), sum/min/max/avg/count(<long>) ... FROM t GROUP BY <keyword>` |

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::sql::sqlparser::ast::{
    BinaryOperator, Expr, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
    LimitClause, ObjectNamePart, OrderByKind, OrderBySort, Query, Select, SelectItem, SetExpr,
    Statement, TableFactor, Value,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;

use crate::error::{Error, Result};
use crate::group::AggFn;
use crate::native::Hit;
use crate::pred::parse_vector;
use crate::snapshot::Snapshot;
use crate::spec::{ColumnKind, TableSpec};

/// The marker a template holds in place of a literal.
pub const PLACEHOLDER: &str = "?";

/// An output column of a group-by: the key or the i-th aggregate.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupItem {
    /// The group key.
    Key,
    /// The i-th aggregate.
    Agg(usize),
}

/// A recognised statement shape; the literal is bound by [`Shape::bind`].
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// `WHERE <key> = '<id>'`.
    Point {
        /// The key column.
        key: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// The id.
        value: String,
    },
    /// `WHERE match_text(...) ORDER BY _score DESC LIMIT k`.
    Match {
        /// The text column.
        column: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// Hits wanted.
        k: usize,
        /// The text.
        text: String,
    },
    /// `WHERE knn(...) ORDER BY _score DESC LIMIT k`.
    Knn {
        /// The vector column.
        column: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// Hits wanted.
        k: usize,
        /// The vector as written.
        vector: String,
    },
    /// `GROUP BY <keyword>`.
    Group {
        /// The keyword column.
        key: usize,
        /// The select list, in order.
        items: Vec<GroupItem>,
        /// The aggregates.
        aggs: Vec<AggFn>,
        /// The result's column names, as DataFusion names them.
        names: Vec<String>,
    },
}

impl Shape {
    fn bind(&self, lit: &str) -> Shape {
        let mut s = self.clone();
        match &mut s {
            Shape::Point { value, .. } => *value = lit.to_string(),
            Shape::Match { text, .. } => *text = lit.to_string(),
            Shape::Knn { vector, .. } => *vector = lit.to_string(),
            Shape::Group { .. } => {}
        }
        s
    }

    fn takes_literal(&self) -> bool {
        !matches!(self, Shape::Group { .. })
    }
}

/// The identifier's name as DataFusion resolves it: unquoted names fold to lower case.
fn name(id: &Ident) -> String {
    if id.quote_style.is_some() {
        id.value.clone()
    } else {
        id.value.to_lowercase()
    }
}

fn ident(e: &Expr) -> Option<String> {
    match e {
        Expr::Identifier(id) => Some(name(id)),
        _ => None,
    }
}

fn is_placeholder(e: &Expr) -> bool {
    matches!(e, Expr::Value(v) if matches!(&v.value, Value::SingleQuotedString(s) if s == PLACEHOLDER))
}

fn plain_select(s: &Select) -> bool {
    s.distinct.is_none()
        && s.select_modifiers.is_none()
        && s.top.is_none()
        && s.into.is_none()
        && s.lateral_views.is_empty()
        && s.prewhere.is_none()
        && s.connect_by.is_empty()
        && s.cluster_by.is_empty()
        && s.distribute_by.is_empty()
        && s.sort_by.is_empty()
        && s.having.is_none()
        && s.named_window.is_empty()
        && s.qualify.is_none()
        && s.exclude.is_none()
        && s.value_table_mode.is_none()
        && s.optimizer_hints.is_empty()
}

fn plain_query(q: &Query) -> bool {
    q.with.is_none()
        && q.fetch.is_none()
        && q.locks.is_empty()
        && q.for_clause.is_none()
        && q.settings.is_none()
        && q.format_clause.is_none()
        && q.pipe_operators.is_empty()
}

fn from_table(s: &Select, spec: &TableSpec) -> bool {
    let [t] = s.from.as_slice() else { return false };
    if !t.joins.is_empty() {
        return false;
    }
    match &t.relation {
        TableFactor::Table {
            name: n,
            alias: None,
            args: None,
            with_hints,
            version: None,
            partitions,
            json_path: None,
            sample: None,
            index_hints,
            ..
        } => {
            with_hints.is_empty()
                && partitions.is_empty()
                && index_hints.is_empty()
                && matches!(n.0.as_slice(), [ObjectNamePart::Identifier(id)] if name(id) == spec.name())
        }
        _ => false,
    }
}

/// The columns of a projection of plain column names (and `_score` when `score`).
fn projection(s: &Select, spec: &TableSpec, score: bool) -> Option<Vec<usize>> {
    if s.projection.is_empty() {
        return None;
    }
    s.projection
        .iter()
        .map(|item| {
            let SelectItem::UnnamedExpr(e) = item else {
                return None;
            };
            let i = spec.index_of(&ident(e)?)?;
            let readable = i == spec.score_index() || spec.columns()[i].kind.is_readable();
            (readable && (score || i != spec.score_index())).then_some(i)
        })
        .collect()
}

fn function_args(
    f: &datafusion::sql::sqlparser::ast::Function,
) -> Option<(String, Vec<FunctionArgExpr>)> {
    let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() else {
        return None;
    };
    if f.filter.is_some()
        || f.over.is_some()
        || !f.within_group.is_empty()
        || f.null_treatment.is_some()
        || !matches!(f.parameters, FunctionArguments::None)
    {
        return None;
    }
    let FunctionArguments::List(list) = &f.args else {
        return None;
    };
    if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
        return None;
    }
    let args = list
        .args
        .iter()
        .map(|a| match a {
            FunctionArg::Unnamed(e) => Some(e.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((name(id), args))
}

fn expr_arg(a: &FunctionArgExpr) -> Option<&Expr> {
    match a {
        FunctionArgExpr::Expr(e) => Some(e),
        _ => None,
    }
}

/// `match_text(col, '?')` or `knn(col, '?')`: the function's name and column.
fn search_call(e: &Expr, spec: &TableSpec) -> Option<(String, usize)> {
    let Expr::Function(f) = e else { return None };
    let (fname, args) = function_args(f)?;
    let [col, lit] = args.as_slice() else {
        return None;
    };
    if !is_placeholder(expr_arg(lit)?) {
        return None;
    }
    let col_name = ident(expr_arg(col)?)?;
    let c = spec.columns().iter().position(|c| c.name == col_name)?;
    Some((fname, c))
}

fn order_score_desc_limit(q: &Query, spec: &TableSpec) -> Option<usize> {
    let ob = q.order_by.as_ref()?;
    if ob.interpolate.is_some() {
        return None;
    }
    let OrderByKind::Expressions(exprs) = &ob.kind else {
        return None;
    };
    let [e] = exprs.as_slice() else { return None };
    if ident(&e.expr)? != crate::spec::SCORE_COLUMN
        || e.options.sort != Some(OrderBySort::Desc)
        || e.options.nulls_first.is_some()
        || e.with_fill.is_some()
    {
        return None;
    }
    let _ = spec;
    let LimitClause::LimitOffset {
        limit: Some(Expr::Value(v)),
        offset: None,
        limit_by,
    } = q.limit_clause.as_ref()?
    else {
        return None;
    };
    if !limit_by.is_empty() {
        return None;
    }
    match &v.value {
        Value::Number(n, false) => n.parse().ok(),
        _ => None,
    }
}

fn aggregate(e: &Expr, spec: &TableSpec) -> Option<(AggFn, String)> {
    let Expr::Function(f) = e else { return None };
    let (fname, args) = function_args(f)?;
    let [arg] = args.as_slice() else { return None };
    if fname == "count" && matches!(arg, FunctionArgExpr::Wildcard) {
        return Some((AggFn::CountStar, "count(*)".to_string()));
    }
    let col = ident(expr_arg(arg)?)?;
    let c = spec.columns().iter().position(|c| c.name == col)?;
    if spec.columns()[c].kind != ColumnKind::Long {
        return None;
    }
    let agg = match fname.as_str() {
        "count" => AggFn::Count(c),
        "sum" => AggFn::Sum(c),
        "min" => AggFn::Min(c),
        "max" => AggFn::Max(c),
        "avg" => AggFn::Avg(c),
        _ => return None,
    };
    Some((agg, format!("{fname}({}.{col})", spec.name())))
}

fn group_shape(s: &Select, q: &Query, spec: &TableSpec) -> Option<Shape> {
    if s.selection.is_some() || q.order_by.is_some() || q.limit_clause.is_some() {
        return None;
    }
    let GroupByExpr::Expressions(keys, mods) = &s.group_by else {
        return None;
    };
    let ([k], true) = (keys.as_slice(), mods.is_empty()) else {
        return None;
    };
    let key_name = ident(k)?;
    let key = spec.columns().iter().position(|c| c.name == key_name)?;
    if spec.columns()[key].kind != ColumnKind::Keyword {
        return None;
    }
    let mut items = Vec::new();
    let mut aggs = Vec::new();
    let mut names = Vec::new();
    for item in &s.projection {
        let SelectItem::UnnamedExpr(e) = item else {
            return None;
        };
        if ident(e).as_deref() == Some(&key_name) {
            items.push(GroupItem::Key);
            names.push(key_name.clone());
        } else {
            let (agg, n) = aggregate(e, spec)?;
            items.push(GroupItem::Agg(aggs.len()));
            aggs.push(agg);
            names.push(n);
        }
    }
    (!items.is_empty()).then_some(Shape::Group {
        key,
        items,
        aggs,
        names,
    })
}

/// The shape of `template`, a statement whose literals are all `'?'`.
pub fn recognise(template: &str, spec: &TableSpec) -> Option<Shape> {
    let stmts = Parser::parse_sql(&GenericDialect {}, template).ok()?;
    let [Statement::Query(q)] = stmts.as_slice() else {
        return None;
    };
    let SetExpr::Select(s) = &*q.body else {
        return None;
    };
    if !plain_query(q) || !plain_select(s) || !from_table(s, spec) {
        return None;
    }
    if matches!(&s.group_by, GroupByExpr::Expressions(g, m) if !g.is_empty() || !m.is_empty())
        || matches!(&s.group_by, GroupByExpr::All(_))
    {
        return group_shape(s, q, spec);
    }
    let sel = s.selection.as_ref()?;
    match sel {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } if q.order_by.is_none() && q.limit_clause.is_none() => {
            let col = ident(left)?;
            let key = spec.columns().iter().position(|c| c.name == col)?;
            (spec.columns()[key].kind.is_term_indexed() && is_placeholder(right)).then_some(())?;
            Some(Shape::Point {
                key,
                project: projection(s, spec, false)?,
                value: String::new(),
            })
        }
        e => {
            let (f, column) = search_call(e, spec)?;
            let k = order_score_desc_limit(q, spec)?;
            let project = projection(s, spec, true)?;
            match (f.as_str(), &spec.columns()[column].kind) {
                ("match_text", ColumnKind::Text) => Some(Shape::Match {
                    column,
                    project,
                    k,
                    text: String::new(),
                }),
                ("knn", ColumnKind::Vector { .. }) => Some(Shape::Knn {
                    column,
                    project,
                    k,
                    vector: String::new(),
                }),
                _ => None,
            }
        }
    }
}

/// Replaces every single-quoted literal with `'?'`; returns the template and
/// the literals in order. A scan of the text, no parsing: when it cannot tell
/// (an escaped quote makes two literals) the statement is simply not a shape;
/// an unterminated literal is `None`.
pub fn templatize(sql: &str) -> Option<(String, Vec<String>)> {
    templatize_with(sql, |_| PLACEHOLDER.to_string())
}

/// [`templatize`] with `marker(i)` as the text standing in for the i-th literal.
pub fn templatize_with(
    sql: &str,
    marker: impl Fn(usize) -> String,
) -> Option<(String, Vec<String>)> {
    let mut t = String::with_capacity(sql.len());
    let mut lits = Vec::new();
    let b = sql.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\'' {
            let start = i + 1;
            i = start;
            while i < b.len() && b[i] != b'\'' {
                i += 1;
            }
            if i >= b.len() {
                return None;
            }
            t.push('\'');
            t.push_str(&marker(lits.len()));
            t.push('\'');
            lits.push(sql[start..i].to_string());
            i += 1;
        } else {
            let start = i;
            while i < b.len() && b[i] != b'\'' {
                i += 1;
            }
            t.push_str(&sql[start..i]);
        }
    }
    Some((t, lits))
}

/// How many templates the shape cache holds before it starts over.
const MAX_TEMPLATES: usize = 4096;

/// The longest template the shape cache keeps: a longer statement is
/// recognised again each time, so the cache holds at most `MAX_TEMPLATES`
/// short strings, whatever the statements are.
pub(crate) const MAX_TEMPLATE_BYTES: usize = 2048;

/// The shape cache and the executor of recognised shapes over one snapshot.
pub struct Router {
    snapshot: Arc<Snapshot>,
    cache: Mutex<HashMap<String, Option<Shape>>>,
}

impl Router {
    /// A router over `snapshot`.
    pub fn new(snapshot: Arc<Snapshot>) -> Self {
        Self {
            snapshot,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The shape of `sql` with its literal bound, when it is one.
    pub fn shape(&self, sql: &str) -> Option<Shape> {
        let (template, lits) = templatize(sql)?;
        let cached = if template.len() > MAX_TEMPLATE_BYTES {
            recognise(&template, self.snapshot.spec())
        } else {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache.len() >= MAX_TEMPLATES && !cache.contains_key(&template) {
                cache.clear();
            }
            cache
                .entry(template)
                .or_insert_with_key(|t| recognise(t, self.snapshot.spec()))
                .clone()
        }?;
        match (cached.takes_literal(), lits.as_slice()) {
            (false, []) => Some(cached),
            (true, [one]) => Some(cached.bind(one)),
            _ => None,
        }
    }

    /// Runs `sql` when it is a recognised shape: `Ok(None)` means "not mine,
    /// run it the standard way" (also when the kernels could not answer a
    /// group-by exactly).
    ///
    /// # Errors
    /// The index could not be read, or a literal is not valid for its column
    /// (a vector of the wrong dimension, text that is not a list of numbers).
    pub fn run(&self, sql: &str) -> Result<Option<RecordBatch>> {
        match self.shape(sql) {
            Some(shape) => self.execute(&shape),
            None => Ok(None),
        }
    }

    /// Runs a recognised shape.
    ///
    /// # Errors
    /// As [`Self::run`].
    pub fn execute(&self, shape: &Shape) -> Result<Option<RecordBatch>> {
        let snap = &*self.snapshot;
        let spec = snap.spec();
        let name_of = |c: usize| spec.columns()[c].name.clone();
        Ok(Some(match shape {
            Shape::Point {
                key,
                project,
                value,
            } => {
                let hits: Vec<Hit> = snap
                    .lookup(&name_of(*key), value)?
                    .map(|d| (d, None))
                    .into_iter()
                    .collect();
                snap.read(project, &hits)?
            }
            Shape::Match {
                column,
                project,
                k,
                text,
            } => {
                let hits = snap.match_top_k(&name_of(*column), text, *k)?;
                snap.read(project, &hits)?
            }
            Shape::Knn {
                column,
                project,
                k,
                vector,
            } => {
                let v = parse_vector(vector).ok_or_else(|| {
                    Error::unsupported(format!("`{vector}` is not a list of numbers"))
                })?;
                let hits = snap.knn_top_k(&name_of(*column), &v, *k)?;
                snap.read(project, &hits)?
            }
            Shape::Group {
                key,
                items,
                aggs,
                names,
            } => {
                let Some(g) = snap.group_by(*key, aggs, &[])? else {
                    return Ok(None);
                };
                let arrays: Vec<ArrayRef> = items
                    .iter()
                    .map(|i| match i {
                        GroupItem::Key => g.keys.clone(),
                        GroupItem::Agg(a) => g.aggs[*a].clone(),
                    })
                    .collect();
                let fields: Vec<Field> = names
                    .iter()
                    .zip(&arrays)
                    .map(|(n, a): (&String, &ArrayRef)| Field::new(n, a.data_type().clone(), true))
                    .collect();
                let _ = DataType::Null;
                RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(Error::from)?
            }
        }))
    }
}
