//! Parameterised physical-plan cache: the plan of a query shape (its literals
//! cut out) is built once; each execution binds the literals into the scan
//! nodes and runs it. This is what DataFusion's prepared statements do not
//! do (they reuse only the logical plan).

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::SessionContext;

use crate::provider::SegScanExec;
use crate::router::templatize;

pub struct PlanCache {
    ctx: SessionContext,
    plans: Mutex<HashMap<String, Arc<dyn ExecutionPlan>>>,
}

impl PlanCache {
    pub fn new(ctx: SessionContext) -> Self {
        Self {
            ctx,
            plans: Mutex::default(),
        }
    }

    pub async fn run(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let (template, lits) = templatize(sql);
        let cached = self.plans.lock().unwrap().get(&template).cloned();
        let plan = match cached {
            Some(p) => p,
            None => {
                let p = self
                    .ctx
                    .sql(&template)
                    .await?
                    .create_physical_plan()
                    .await?;
                self.plans.lock().unwrap().insert(template, p.clone());
                p
            }
        };
        let plan = if lits.is_empty() {
            plan
        } else {
            plan.transform_down(|node| {
                let any: &dyn Any = &*node;
                match any
                    .downcast_ref::<SegScanExec>()
                    .and_then(|s| s.bind(&lits))
                {
                    Some(bound) => Ok(Transformed::yes(Arc::new(bound) as Arc<dyn ExecutionPlan>)),
                    None => Ok(Transformed::no(node)),
                }
            })?
            .data
        };
        // Operators keep run state (TopK's dynamic filter holds the previous
        // query's score threshold): a reused plan must be reset before each run.
        let plan = datafusion::physical_plan::execution_plan::reset_plan_states(plan)?;
        if std::env::var_os("PLAN_DEBUG").is_some() {
            eprintln!(
                "{sql}\n{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
            );
        }
        collect(plan, self.ctx.task_ctx()).await
    }
}
