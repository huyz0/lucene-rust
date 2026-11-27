//! Fast-path router: recognises four SQL shapes with sqlparser and calls the
//! native API directly; everything else goes to DataFusion.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::sql::sqlparser::ast::{SetExpr, Statement};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;

use crate::direct;
use crate::index::SpikeIndex;

#[derive(Debug, Clone, PartialEq)]
pub enum Fast {
    Point { id: String },
    Match { words: Vec<String>, k: usize },
    Knn { vector: Vec<f32>, k: usize },
    GroupBy,
}

fn strip<'a>(s: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    s.strip_prefix(prefix)?.strip_suffix(suffix)
}

/// Parses `sql`; `None` when it is not one of the fast shapes.
pub fn recognise(sql: &str) -> Option<Fast> {
    let stmts = Parser::parse_sql(&GenericDialect {}, sql).ok()?;
    let [Statement::Query(q)] = stmts.as_slice() else {
        return None;
    };
    let SetExpr::Select(s) = &*q.body else {
        return None;
    };
    if s.from.len() != 1 || s.from[0].to_string() != "t" || s.having.is_some() {
        return None;
    }
    let proj: Vec<String> = s.projection.iter().map(|p| p.to_string()).collect();
    let sel = s.selection.as_ref().map(|e| e.to_string());
    let group = s.group_by.to_string();
    let order = q.order_by.as_ref().map(|o| o.to_string());
    let limit = q
        .limit_clause
        .as_ref()
        .map(|l| l.to_string().trim().to_string());
    let proj: Vec<&str> = proj.iter().map(String::as_str).collect();
    match (
        proj.as_slice(),
        sel.as_deref(),
        order.as_deref(),
        limit.as_deref(),
    ) {
        (["id", "price"], Some(sel), None, None) => Some(Fast::Point {
            id: strip(sel, "id = '", "'")?.to_string(),
        }),
        (["id", "_score"], Some(sel), Some("ORDER BY _score DESC"), Some(l)) => {
            let k: usize = l.strip_prefix("LIMIT ")?.parse().ok()?;
            if let Some(w) = strip(sel, "match_text(body, '", "')") {
                Some(Fast::Match {
                    words: w.split_whitespace().map(str::to_lowercase).collect(),
                    k,
                })
            } else {
                let v = strip(sel, "knn(vec, '", "')")?;
                // `?` is a template's placeholder, bound later.
                let vector: Result<Vec<f32>, _> = if v == "?" {
                    Ok(Vec::new())
                } else {
                    v.split(',').map(|x| x.trim().parse()).collect()
                };
                Some(Fast::Knn {
                    vector: vector.ok()?,
                    k,
                })
            }
        }
        (["cat", "count(*)", "sum(price)"], None, None, None) if group == "GROUP BY cat" => {
            Some(Fast::GroupBy)
        }
        _ => None,
    }
}

fn hits_batch(hits: Vec<(String, f32)>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("_score", DataType::Float32, true),
    ]));
    let ids: ArrayRef = Arc::new(StringArray::from_iter_values(
        hits.iter().map(|h| h.0.as_str()),
    ));
    let sc: ArrayRef = Arc::new(Float32Array::from_iter_values(hits.iter().map(|h| h.1)));
    RecordBatch::try_new(schema, vec![ids, sc]).unwrap()
}

impl Fast {
    pub fn execute(&self, ix: &SpikeIndex) -> RecordBatch {
        match self {
            Fast::Point { id } => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Utf8, true),
                    Field::new("price", DataType::Int64, true),
                ]));
                let row = direct::point_read(ix, id);
                let ids: ArrayRef = Arc::new(StringArray::from_iter(
                    row.iter().map(|r| Some(r.0.as_str())),
                ));
                let pr: ArrayRef = Arc::new(Int64Array::from_iter(row.iter().map(|r| r.1)));
                RecordBatch::try_new(schema, vec![ids, pr]).unwrap()
            }
            Fast::Match { words, k } => {
                let w: Vec<&str> = words.iter().map(String::as_str).collect();
                hits_batch(direct::match_topk(ix, &w, *k))
            }
            Fast::Knn { vector, k } => hits_batch(direct::knn_topk(ix, vector, *k)),
            Fast::GroupBy => {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("cat", DataType::Utf8, true),
                    Field::new("count(*)", DataType::Int64, true),
                    Field::new("sum(price)", DataType::Int64, true),
                ]));
                let rows = direct::group_by(ix);
                let a: ArrayRef = Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|r| r.0.as_str()),
                ));
                let b: ArrayRef = Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|r| r.1 as i64),
                ));
                let c: ArrayRef = Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|r| r.2 as i64),
                ));
                RecordBatch::try_new(schema, vec![a, b, c]).unwrap()
            }
        }
    }
}

/// Replaces every single-quoted literal with `'?'`; returns the template and
/// the literals in order. A cheap scan, no parsing.
pub fn templatize(sql: &str) -> (String, Vec<String>) {
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
            lits.push(sql[start..i].to_string());
            t.push_str("'?'");
            i += 1;
        } else {
            let start = i;
            while i < b.len() && b[i] != b'\'' {
                i += 1;
            }
            t.push_str(&sql[start..i]);
        }
    }
    (t, lits)
}

impl Fast {
    /// The shape with its one literal bound.
    pub fn bind(&self, lit: &str) -> Option<Fast> {
        Some(match self {
            Fast::Point { .. } => Fast::Point {
                id: lit.to_string(),
            },
            Fast::Match { k, .. } => Fast::Match {
                words: lit.split_whitespace().map(str::to_lowercase).collect(),
                k: *k,
            },
            Fast::Knn { k, .. } => {
                let v: Result<Vec<f32>, _> = lit.split(',').map(|x| x.trim().parse()).collect();
                Fast::Knn {
                    vector: v.ok()?,
                    k: *k,
                }
            }
            Fast::GroupBy => Fast::GroupBy,
        })
    }
}

/// Statement cache: the first time a text shape is seen it is parsed; after
/// that only the literals are cut out and bound.
#[derive(Default)]
pub struct Router {
    cache: std::sync::Mutex<std::collections::HashMap<String, Option<Fast>>>,
}

impl Router {
    pub fn fast_plan(&self, sql: &str) -> Option<Fast> {
        let (template, lits) = templatize(sql);
        let mut cache = self.cache.lock().unwrap();
        let entry = cache.entry(template).or_insert_with_key(|t| recognise(t));
        let shape = entry.as_ref()?;
        match lits.as_slice() {
            [] => Some(shape.clone()),
            [one] => shape.bind(one),
            _ => None,
        }
    }
}
