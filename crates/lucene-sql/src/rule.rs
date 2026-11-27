//! Logical optimiser rules that hand work to the scan.

use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::Result;
use datafusion::logical_expr::logical_plan::{LogicalPlan, Projection};
use datafusion::logical_expr::{Expr, SortExpr};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use crate::spec::SCORE_COLUMN;

/// Pushes `ORDER BY _score DESC LIMIT k` into the scan: the rule only sets the
/// `fetch` of the table scan under the sort, so the scan returns its `k` best
/// hits instead of every match. The sort stays in the plan (it orders what the
/// scan returns); the scan has no other way to learn `k`.
///
/// It fires only for a sort on `_score DESC` alone (a second key would make
/// the boundary among equal scores matter) over projections down to a scan
/// that has the `_score` column.
#[derive(Debug, Default)]
pub struct ScoreTopKRule;

fn is_score_desc(e: &SortExpr) -> bool {
    !e.asc && matches!(&e.expr, Expr::Column(c) if c.name == SCORE_COLUMN)
}

fn push(plan: &LogicalPlan, k: usize) -> Result<Option<LogicalPlan>> {
    Ok(match plan {
        LogicalPlan::TableScan(scan) => {
            let scored = scan.source.schema().field_with_name(SCORE_COLUMN).is_ok();
            if scored && scan.fetch.is_none_or(|f| f > k) {
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
            if let (Some(k), [first]) = (sort.fetch, sort.expr.as_slice()) {
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
