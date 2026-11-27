//! `ORDER BY _score DESC LIMIT k` pushed into the scan by a logical optimiser
//! rule: the rule only sets the `fetch` of the table scan under the sort, so
//! the provider returns each partition's `k` best hits instead of every
//! match. The sort stays in the plan and merges the partitions' hits.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::Result;
use datafusion::logical_expr::logical_plan::{LogicalPlan, Projection};
use datafusion::logical_expr::Expr;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

#[derive(Debug, Default)]
pub struct ScoreTopKRule;

fn is_score_desc(e: &datafusion::logical_expr::SortExpr) -> bool {
    !e.asc && matches!(&e.expr, Expr::Column(c) if c.name == "_score")
}

/// The plan with the scan's fetch set to `k`, only through nodes that keep one
/// row per row (projection, alias) down to a scan that has a `_score` column.
fn push(plan: &LogicalPlan, k: usize) -> Result<Option<LogicalPlan>> {
    Ok(match plan {
        LogicalPlan::TableScan(scan) => {
            let has_score = scan.source.schema().field_with_name("_score").is_ok();
            if has_score && scan.fetch.is_none_or(|f| f > k) {
                let mut s = scan.clone();
                s.fetch = Some(k);
                Some(LogicalPlan::TableScan(s))
            } else {
                None
            }
        }
        LogicalPlan::Projection(p) => match push(&p.input, k)? {
            Some(input) => Some(LogicalPlan::Projection(Projection::try_new(
                p.expr.clone(),
                Arc::new(input),
            )?)),
            None => None,
        },
        _ => None,
    })
}

impl OptimizerRule for ScoreTopKRule {
    fn name(&self) -> &str {
        "score_topk_pushdown"
    }
    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let LogicalPlan::Sort(sort) = &plan {
            if let (Some(k), Some(first)) = (sort.fetch, sort.expr.first()) {
                if is_score_desc(first) {
                    if let Some(input) = push(&sort.input, k)? {
                        let mut s = sort.clone();
                        s.input = Arc::new(input);
                        return Ok(Transformed::yes(LogicalPlan::Sort(s)));
                    }
                }
            }
        }
        Ok(Transformed::no(plan))
    }
}
