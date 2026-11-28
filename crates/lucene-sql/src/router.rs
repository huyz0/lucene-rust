//! The fast-path router: statements of a few hot shapes are recognised from
//! their parse tree and answered by the native API directly; everything else
//! (and anything the router is not sure of) is left to DataFusion, which gives
//! the same rows more slowly.
//!
//! Recognition matches the tree's nodes, never the statement's text, so
//! spacing, case and redundant parentheses do not matter and a near miss (an
//! extra predicate, an alias on a column, a different projection) is not
//! mistaken for a shape. The tree is built once per *template*: the statement
//! with its quoted literals, and the integers a comparison holds, cut out
//! ([`templatize`]); a hit binds the statement's own literals into the cached
//! shape.
//!
//! | shape | statement |
//! |---|---|
//! | point read | `SELECT <columns> FROM t WHERE <key> = '<id>'` |
//! | text top-k | `SELECT <columns, _score> FROM t WHERE match_text(<text>, '<q>') [AND <filters>] ORDER BY _score DESC LIMIT k` |
//! | k-NN top-k | `SELECT <columns, _score> FROM t WHERE knn(<vector>, '<v>') [AND <filters>] ORDER BY _score DESC LIMIT k` |
//! | group-by | `SELECT <keyword>, count(*), sum/min/max/avg/count(<long>) ... FROM t [WHERE <filters>] GROUP BY <keyword>` |
//! | count | `SELECT count(*) FROM t [WHERE <filters>]` |
//!
//! `<filters>` are `AND`-ed `<key or keyword> = '<x>'`, `<key or keyword> IN
//! ('<x>', ...)`, and comparisons or `BETWEEN` of a long column with
//! integers. A select item may carry an alias.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::logical_expr::Operator;
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
use crate::pred::{parse_vector, range_of, Pred};
use crate::snapshot::Snapshot;
use crate::spec::{ColumnKind, TableSpec};

/// The marker a template holds in place of the `i`-th quoted literal.
fn quoted_marker(i: usize) -> String {
    format!("?{i}")
}

/// The marker a template holds in place of the `i`-th integer.
fn number_marker(i: usize) -> String {
    format!("#{i}")
}

/// What a marker in a template stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The `i`-th literal, written in quotes.
    Quoted(usize),
    /// The `i`-th literal, written as an integer.
    Number(usize),
}

impl Slot {
    fn index(self) -> usize {
        match self {
            Slot::Quoted(i) | Slot::Number(i) => i,
        }
    }
}

/// A literal of a shape: a slot of its template until bound, then its text.
#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    /// Not bound yet.
    Slot(usize),
    /// A quoted literal's text.
    Text(String),
    /// An integer literal's text.
    Number(String),
}

/// A filter of a shape: one `AND`-ed condition.
#[derive(Debug, Clone, PartialEq)]
pub enum Cond {
    /// `column = 'x'` on a key or keyword column.
    Term {
        /// The column.
        column: usize,
        /// The term.
        value: Lit,
    },
    /// `column IN ('x', ...)` on a key or keyword column.
    Any {
        /// The column.
        column: usize,
        /// The terms.
        values: Vec<Lit>,
    },
    /// `column <op> n` on a long column.
    Cmp {
        /// The column.
        column: usize,
        /// `=`, `>`, `>=`, `<` or `<=`.
        op: Operator,
        /// The integer.
        value: Lit,
    },
}

impl Cond {
    fn bind(&self, lits: &[String]) -> Cond {
        let one = |l: &Lit, number: bool| match l {
            Lit::Slot(i) => {
                let t = lits[*i].clone();
                if number {
                    Lit::Number(t)
                } else {
                    Lit::Text(t)
                }
            }
            other => other.clone(),
        };
        match self {
            Cond::Term { column, value } => Cond::Term {
                column: *column,
                value: one(value, false),
            },
            Cond::Any { column, values } => Cond::Any {
                column: *column,
                values: values.iter().map(|v| one(v, false)).collect(),
            },
            Cond::Cmp { column, op, value } => Cond::Cmp {
                column: *column,
                op: *op,
                value: one(value, true),
            },
        }
    }

    /// The predicate, when its literal is valid for the column (an integer
    /// that is one).
    fn pred(&self) -> Option<Pred> {
        let text = |l: &Lit| match l {
            Lit::Text(t) | Lit::Number(t) => Some(t.clone()),
            Lit::Slot(_) => None,
        };
        match self {
            Cond::Term { column, value } => Some(Pred::Term {
                column: *column,
                value: text(value)?,
            }),
            Cond::Any { column, values } => Some(Pred::Any {
                column: *column,
                values: values.iter().map(text).collect::<Option<Vec<_>>>()?,
            }),
            Cond::Cmp { column, op, value } => {
                let v: i64 = text(value)?.trim().parse().ok()?;
                let (lower, upper) = range_of(*op, v)?;
                Some(Pred::Range {
                    column: *column,
                    lower,
                    upper,
                })
            }
        }
    }
}

fn preds_of(conds: &[Cond]) -> Option<Vec<Pred>> {
    conds.iter().map(Cond::pred).collect()
}

/// An output column of a group-by: the key or the i-th aggregate.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupItem {
    /// The group key.
    Key,
    /// The i-th aggregate.
    Agg(usize),
}

/// A recognised statement shape; the literals are bound by [`Router::shape`].
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// `WHERE <key> = '<id>'`.
    Point {
        /// The key column.
        key: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// The id.
        value: Lit,
    },
    /// `WHERE match_text(...) [AND ...] ORDER BY _score DESC LIMIT k`.
    Match {
        /// The text column.
        column: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// Hits wanted.
        k: usize,
        /// The text.
        text: Lit,
        /// The other conditions.
        conds: Vec<Cond>,
    },
    /// `WHERE knn(...) [AND ...] ORDER BY _score DESC LIMIT k`.
    Knn {
        /// The vector column.
        column: usize,
        /// Columns to return.
        project: Vec<usize>,
        /// Hits wanted.
        k: usize,
        /// The vector as written.
        vector: Lit,
        /// The other conditions.
        conds: Vec<Cond>,
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
        /// The conditions.
        conds: Vec<Cond>,
    },
    /// `SELECT count(*)`.
    Count {
        /// The result's column name.
        name: String,
        /// The conditions.
        conds: Vec<Cond>,
    },
}

impl Shape {
    /// The shape with the statement's literals in place of its slots.
    fn bind(&self, lits: &[String]) -> Shape {
        let one = |l: &Lit| match l {
            Lit::Slot(i) => Lit::Text(lits[*i].clone()),
            other => other.clone(),
        };
        let conds = |c: &[Cond]| c.iter().map(|c| c.bind(lits)).collect::<Vec<_>>();
        match self {
            Shape::Point {
                key,
                project,
                value,
            } => Shape::Point {
                key: *key,
                project: project.clone(),
                value: one(value),
            },
            Shape::Match {
                column,
                project,
                k,
                text,
                conds: c,
            } => Shape::Match {
                column: *column,
                project: project.clone(),
                k: *k,
                text: one(text),
                conds: conds(c),
            },
            Shape::Knn {
                column,
                project,
                k,
                vector,
                conds: c,
            } => Shape::Knn {
                column: *column,
                project: project.clone(),
                k: *k,
                vector: one(vector),
                conds: conds(c),
            },
            Shape::Group {
                key,
                items,
                aggs,
                names,
                conds: c,
            } => Shape::Group {
                key: *key,
                items: items.clone(),
                aggs: aggs.clone(),
                names: names.clone(),
                conds: conds(c),
            },
            Shape::Count { name, conds: c } => Shape::Count {
                name: name.clone(),
                conds: conds(c),
            },
        }
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

/// The slot a marker literal in a template stands for.
fn slot_of(e: &Expr) -> Option<Slot> {
    let Expr::Value(v) = e else { return None };
    let Value::SingleQuotedString(s) = &v.value else {
        return None;
    };
    let (kind, rest) = s.split_at_checked(1)?;
    let i: usize = rest.parse().ok()?;
    match kind {
        "?" => Some(Slot::Quoted(i)),
        "#" => Some(Slot::Number(i)),
        _ => None,
    }
}

/// The marker slots a recognised template consumed.
#[derive(Default)]
struct Used(Vec<usize>);

impl Used {
    /// The quoted literal `e` is, noted as consumed.
    fn quoted(&mut self, e: &Expr) -> Option<Lit> {
        match slot_of(e)? {
            Slot::Quoted(i) => {
                self.0.push(i);
                Some(Lit::Slot(i))
            }
            Slot::Number(_) => None,
        }
    }

    /// The quoted or integer literal `e` is, noted as consumed.
    fn any(&mut self, e: &Expr) -> Option<Lit> {
        let s = slot_of(e)?;
        self.0.push(s.index());
        Some(Lit::Slot(s.index()))
    }

    /// How many slots there are, when they are 0..n each used once.
    fn count(mut self) -> Option<usize> {
        self.0.sort_unstable();
        self.0
            .iter()
            .copied()
            .eq(0..self.0.len())
            .then_some(self.0.len())
    }
}

/// A select item's expression and its alias, as DataFusion names it.
fn select_item(item: &SelectItem) -> Option<(&Expr, Option<String>)> {
    match item {
        SelectItem::UnnamedExpr(e) => Some((e, None)),
        SelectItem::ExprWithAlias { expr, alias } => Some((expr, Some(name(alias)))),
        _ => None,
    }
}

/// `match_text(col, '?0')` or `knn(col, '?0')`: the function's name, column
/// and literal.
fn search_call(e: &Expr, spec: &TableSpec, used: &mut Used) -> Option<(String, usize, Lit)> {
    let Expr::Function(f) = e else { return None };
    let (fname, args) = function_args(f)?;
    let [col, lit] = args.as_slice() else {
        return None;
    };
    let lit = used.quoted(expr_arg(lit)?)?;
    let col_name = ident(expr_arg(col)?)?;
    let c = spec.columns().iter().position(|c| c.name == col_name)?;
    Some((fname, c, lit))
}

/// The `AND`-ed terms of `e`.
fn conjuncts<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            conjuncts(left, out);
            conjuncts(right, out);
        }
        Expr::Nested(inner) => conjuncts(inner, out),
        other => out.push(other),
    }
}

fn comparison(op: &BinaryOperator) -> Option<Operator> {
    Some(match op {
        BinaryOperator::Eq => Operator::Eq,
        BinaryOperator::Gt => Operator::Gt,
        BinaryOperator::GtEq => Operator::GtEq,
        BinaryOperator::Lt => Operator::Lt,
        BinaryOperator::LtEq => Operator::LtEq,
        _ => return None,
    })
}

/// The conditions `e` is, when it is one of the filter forms; a `BETWEEN`
/// is two.
fn conds_of(e: &Expr, spec: &TableSpec, used: &mut Used, out: &mut Vec<Cond>) -> Option<()> {
    let column_of = |e: &Expr| {
        spec.columns()
            .iter()
            .position(|c| c.name == ident(e).unwrap_or_default())
    };
    match e {
        Expr::BinaryOp { left, op, right } => {
            let op = comparison(op)?;
            let column = column_of(left)?;
            match (&spec.columns()[column].kind, op) {
                (k, Operator::Eq) if k.is_term_indexed() => out.push(Cond::Term {
                    column,
                    value: used.quoted(right)?,
                }),
                (ColumnKind::Long, op) => out.push(Cond::Cmp {
                    column,
                    op,
                    value: used.any(right)?,
                }),
                _ => return None,
            }
        }
        Expr::Between {
            expr,
            negated: false,
            low,
            high,
        } => {
            let column = column_of(expr)?;
            if spec.columns()[column].kind != ColumnKind::Long {
                return None;
            }
            out.push(Cond::Cmp {
                column,
                op: Operator::GtEq,
                value: used.any(low)?,
            });
            out.push(Cond::Cmp {
                column,
                op: Operator::LtEq,
                value: used.any(high)?,
            });
        }
        Expr::InList {
            expr,
            list,
            negated: false,
        } if !list.is_empty() => {
            let column = column_of(expr)?;
            if !spec.columns()[column].kind.is_term_indexed() {
                return None;
            }
            let values = list
                .iter()
                .map(|v| used.quoted(v))
                .collect::<Option<Vec<_>>>()?;
            out.push(Cond::Any { column, values });
        }
        _ => return None,
    }
    Some(())
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

/// The conditions of a `WHERE` (none without one).
fn where_conds(s: &Select, spec: &TableSpec, used: &mut Used) -> Option<Vec<Cond>> {
    let mut conds = Vec::new();
    if let Some(sel) = &s.selection {
        let mut terms = Vec::new();
        conjuncts(sel, &mut terms);
        for t in terms {
            conds_of(t, spec, used, &mut conds)?;
        }
    }
    Some(conds)
}

fn group_shape(s: &Select, q: &Query, spec: &TableSpec, used: &mut Used) -> Option<Shape> {
    if q.order_by.is_some() || q.limit_clause.is_some() {
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
        let (e, alias) = select_item(item)?;
        if ident(e).as_deref() == Some(&key_name) {
            items.push(GroupItem::Key);
            names.push(alias.unwrap_or_else(|| key_name.clone()));
        } else {
            let (agg, n) = aggregate(e, spec)?;
            items.push(GroupItem::Agg(aggs.len()));
            aggs.push(agg);
            names.push(alias.unwrap_or(n));
        }
    }
    let conds = where_conds(s, spec, used)?;
    (!items.is_empty()).then_some(Shape::Group {
        key,
        items,
        aggs,
        names,
        conds,
    })
}

fn count_shape(s: &Select, q: &Query, spec: &TableSpec, used: &mut Used) -> Option<Shape> {
    if q.order_by.is_some() || q.limit_clause.is_some() {
        return None;
    }
    let [item] = s.projection.as_slice() else {
        return None;
    };
    let (e, alias) = select_item(item)?;
    let (AggFn::CountStar, n) = aggregate(e, spec)? else {
        return None;
    };
    Some(Shape::Count {
        name: alias.unwrap_or(n),
        conds: where_conds(s, spec, used)?,
    })
}

fn search_shape(s: &Select, q: &Query, spec: &TableSpec, used: &mut Used) -> Option<Shape> {
    let mut terms = Vec::new();
    conjuncts(s.selection.as_ref()?, &mut terms);
    let mut search = None;
    let mut conds = Vec::new();
    for t in terms {
        match search_call(t, spec, used) {
            Some(call) if search.is_none() => search = Some(call),
            Some(_) => return None,
            None => conds_of(t, spec, used, &mut conds)?,
        }
    }
    let (f, column, lit) = search?;
    let k = order_score_desc_limit(q, spec)?;
    let project = projection(s, spec, true)?;
    match (f.as_str(), &spec.columns()[column].kind) {
        ("match_text", ColumnKind::Text) => Some(Shape::Match {
            column,
            project,
            k,
            text: lit,
            conds,
        }),
        ("knn", ColumnKind::Vector { .. }) => Some(Shape::Knn {
            column,
            project,
            k,
            vector: lit,
            conds,
        }),
        _ => None,
    }
}

fn point_shape(s: &Select, q: &Query, spec: &TableSpec, used: &mut Used) -> Option<Shape> {
    if q.order_by.is_some() || q.limit_clause.is_some() {
        return None;
    }
    let Expr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = s.selection.as_ref()?
    else {
        return None;
    };
    let key = spec
        .columns()
        .iter()
        .position(|c| c.name == ident(left).unwrap_or_default())?;
    // A key names one document; a keyword can name many.
    (spec.columns()[key].kind == ColumnKind::Key).then_some(())?;
    Some(Shape::Point {
        key,
        project: projection(s, spec, false)?,
        value: used.quoted(right)?,
    })
}

/// The shape of `template` (a statement whose literals are markers) and how
/// many literals it takes.
pub fn recognise(template: &str, spec: &TableSpec) -> Option<(Shape, usize)> {
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
    let mut used = Used::default();
    let grouped = matches!(&s.group_by, GroupByExpr::Expressions(g, m) if !g.is_empty() || !m.is_empty())
        || matches!(&s.group_by, GroupByExpr::All(_));
    let shape = if grouped {
        group_shape(s, q, spec, &mut used)
    } else {
        point_shape(s, q, spec, &mut used)
            .or_else(|| {
                used = Used::default();
                search_shape(s, q, spec, &mut used)
            })
            .or_else(|| {
                used = Used::default();
                count_shape(s, q, spec, &mut used)
            })
    }?;
    Some((shape, used.count()?))
}

/// Replaces every single-quoted literal with `'?<i>'` and every integer a
/// comparison holds (after `=`, `<`, `>`, `BETWEEN` or `AND`) with `'#<i>'`;
/// returns the template and the literals in order. A scan of the text, no
/// parsing: when it cannot tell (an escaped quote makes two literals) the
/// statement is simply not a shape; an unterminated literal is `None`.
pub fn templatize(sql: &str) -> Option<(String, Vec<String>)> {
    scan_literals(sql, true, quoted_marker, number_marker)
}

/// [`templatize`] without integers, with `marker(i)` as the text standing in
/// for the i-th quoted literal.
pub fn templatize_with(
    sql: &str,
    marker: impl Fn(usize) -> String,
) -> Option<(String, Vec<String>)> {
    scan_literals(sql, false, &marker, &marker)
}

/// Whether the text before `at` ends where an integer may start a
/// comparison's operand: an operator, `BETWEEN` or `AND`.
fn operand_follows(before: &str) -> bool {
    let t = before.trim_end();
    match t.as_bytes().last() {
        Some(b'=' | b'<' | b'>') => true,
        Some(c) if c.is_ascii_alphabetic() => {
            let word: String = t
                .chars()
                .rev()
                .take_while(char::is_ascii_alphabetic)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let boundary = t.len() - word.len();
            let clear = boundary == 0
                || !(t.as_bytes()[boundary - 1].is_ascii_alphanumeric()
                    || t.as_bytes()[boundary - 1] == b'_');
            clear && (word.eq_ignore_ascii_case("between") || word.eq_ignore_ascii_case("and"))
        }
        _ => false,
    }
}

fn scan_literals(
    sql: &str,
    numbers: bool,
    quoted: impl Fn(usize) -> String,
    number: impl Fn(usize) -> String,
) -> Option<(String, Vec<String>)> {
    let mut t = String::with_capacity(sql.len());
    let mut lits = Vec::new();
    let b = sql.as_bytes();
    let mut i = 0;
    let mut plain = 0;
    while i < b.len() {
        if b[i] == b'\'' {
            t.push_str(&sql[plain..i]);
            let start = i + 1;
            i = start;
            while i < b.len() && b[i] != b'\'' {
                i += 1;
            }
            if i >= b.len() {
                return None;
            }
            t.push('\'');
            t.push_str(&quoted(lits.len()));
            t.push('\'');
            lits.push(sql[start..i].to_string());
            i += 1;
            plain = i;
        } else if numbers && (b[i].is_ascii_digit() || b[i] == b'-') && operand_follows(&sql[..i]) {
            let start = i;
            let mut j = i + usize::from(b[i] == b'-');
            let digits = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let ends =
                j >= b.len() || !(b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'.');
            if j > digits && ends {
                t.push_str(&sql[plain..start]);
                t.push('\'');
                t.push_str(&number(lits.len()));
                t.push('\'');
                lits.push(sql[start..j].to_string());
                i = j;
                plain = j;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    t.push_str(&sql[plain..]);
    Some((t, lits))
}

/// How many templates the shape cache holds before it starts over.
const MAX_TEMPLATES: usize = 4096;

/// The shape cache and the executor of recognised shapes over one snapshot.
pub struct Router {
    snapshot: Arc<Snapshot>,
    cache: Mutex<HashMap<String, Option<(Shape, usize)>>>,
}

impl Router {
    /// A router over `snapshot`.
    pub fn new(snapshot: Arc<Snapshot>) -> Self {
        Self {
            snapshot,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The shape of `sql` with its literals bound, when it is one.
    pub fn shape(&self, sql: &str) -> Option<Shape> {
        let (template, lits) = templatize(sql)?;
        let (cached, slots) = {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache.len() >= MAX_TEMPLATES && !cache.contains_key(&template) {
                cache.clear();
            }
            cache
                .entry(template)
                .or_insert_with_key(|t| recognise(t, self.snapshot.spec()))
                .clone()
        }?;
        (slots == lits.len()).then(|| cached.bind(&lits))
    }

    /// Runs `sql` when it is a recognised shape: `Ok(None)` means "not mine,
    /// run it the standard way" (also when the kernels could not answer a
    /// group-by exactly, or a literal is not valid for its column).
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
        let text = |l: &Lit| match l {
            Lit::Text(t) | Lit::Number(t) => t.clone(),
            Lit::Slot(_) => String::new(),
        };
        Ok(Some(match shape {
            Shape::Point {
                key,
                project,
                value,
            } => {
                let hits: Vec<Hit> = snap
                    .lookup(&name_of(*key), &text(value))?
                    .map(|d| (d, None))
                    .into_iter()
                    .collect();
                snap.read(project, &hits)?
            }
            Shape::Match {
                column,
                project,
                k,
                text: t,
                conds,
            } => {
                let Some(preds) = preds_of(conds) else {
                    return Ok(None);
                };
                let hits = snap.match_top_k_where(&name_of(*column), &text(t), *k, &preds)?;
                snap.read(project, &hits)?
            }
            Shape::Knn {
                column,
                project,
                k,
                vector,
                conds,
            } => {
                let vector = text(vector);
                let v = parse_vector(&vector).ok_or_else(|| {
                    Error::unsupported(format!("`{vector}` is not a list of numbers"))
                })?;
                let Some(preds) = preds_of(conds) else {
                    return Ok(None);
                };
                let hits = snap.knn_top_k_where(&name_of(*column), &v, *k, &preds)?;
                snap.read(project, &hits)?
            }
            Shape::Group {
                key,
                items,
                aggs,
                names,
                conds,
            } => {
                let Some(preds) = preds_of(conds) else {
                    return Ok(None);
                };
                let Some(g) = snap.group_by(*key, aggs, &preds)? else {
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
                    .zip(items)
                    .zip(&arrays)
                    .map(|((n, item), a): ((&String, &GroupItem), &ArrayRef)| {
                        let counts = matches!(item, GroupItem::Agg(i)
                            if matches!(aggs[*i], AggFn::CountStar | AggFn::Count(_)));
                        Field::new(n, a.data_type().clone(), !counts)
                    })
                    .collect();
                RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(Error::from)?
            }
            Shape::Count { name, conds } => {
                let Some(preds) = preds_of(conds) else {
                    return Ok(None);
                };
                let n = i64::try_from(snap.count(&preds)?).unwrap_or(i64::MAX);
                let array: ArrayRef = Arc::new(Int64Array::from(vec![n]));
                let schema = Schema::new(vec![Field::new(name, array.data_type().clone(), false)]);
                RecordBatch::try_new(Arc::new(schema), vec![array]).map_err(Error::from)?
            }
        }))
    }
}
