//! A cache of physical plans keyed by statement shape.
//!
//! DataFusion plans a statement from text every time (parse, logical plan,
//! optimise, physical plan: 0.1 to 0.7 ms) and reuses only logical plans for
//! prepared statements. Here the statement's quoted literals are replaced by
//! numbered markers (`'__p0__'`), the template is planned once, and each run
//! binds the statement's own literals into the scan nodes that hold the
//! markers and executes a fresh copy of the plan.
//!
//! A template is cached only when its plan provably does not depend on the
//! literals: every marker must end up exactly once in a scan's predicates and
//! nowhere else in the plan (a literal in a residual filter, an `IN` list, a
//! function argument or a folded constant fails that test). Such a template
//! is remembered as uncacheable and always planned from the statement itself.
//!
//! Operators keep state between runs (a top-k sort holds the previous run's
//! score threshold in a dynamic filter, and a stale one silently returns no
//! rows), so every run starts from `reset_plan_states`.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{displayable, execute_stream, ExecutionPlan};
use datafusion::prelude::{SQLOptions, SessionContext};

use crate::engine::collect_bounded;
use crate::error::Result;
use crate::provider::SegScanExec;
use crate::router::{templatize_with, MAX_TEMPLATE_BYTES};

/// Plans kept before the cache starts over.
const MAX_PLANS: usize = 1024;

const MARK: &str = "__p";

fn marker(i: usize) -> String {
    format!("{MARK}{i}__")
}

enum Entry {
    Plan(Arc<dyn ExecutionPlan>),
    /// The template's plan depends on its literals, or could not be planned.
    Uncacheable,
}

/// How the cache has been used.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanCacheStats {
    /// Runs that reused a cached plan.
    pub hits: u64,
    /// Runs that planned a template and cached it.
    pub planned: u64,
    /// Runs of statements that cannot be cached.
    pub uncacheable: u64,
}

/// The cache.
#[derive(Default)]
pub struct PlanCache {
    plans: Mutex<HashMap<String, Arc<Entry>>>,
    hits: AtomicU64,
    planned: AtomicU64,
    uncacheable: AtomicU64,
}

impl PlanCache {
    /// Use so far.
    pub fn stats(&self) -> PlanCacheStats {
        PlanCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            planned: self.planned.load(Ordering::Relaxed),
            uncacheable: self.uncacheable.load(Ordering::Relaxed),
        }
    }

    /// Runs `sql` from a cached plan when its template has one.
    ///
    /// Returns `None` when the statement cannot be run from a cached plan
    /// (the caller plans it itself): an unterminated literal, a template that
    /// does not plan or whose plan depends on its literals.
    ///
    /// # Errors
    /// The cached plan failed to run.
    pub async fn run(
        &self,
        ctx: &SessionContext,
        options: SQLOptions,
        sql: &str,
        max_rows: usize,
    ) -> Result<Option<Vec<RecordBatch>>> {
        let Some((template, lits)) = templatize_with(sql, marker) else {
            return Ok(None);
        };
        let cached = self
            .plans
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&template)
            .cloned();
        let entry = match cached {
            Some(e) => {
                if matches!(*e, Entry::Plan(_)) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                }
                e
            }
            None => {
                let e = Arc::new(plan_template(ctx, options, &template, lits.len()).await);
                let counter = if matches!(*e, Entry::Plan(_)) {
                    &self.planned
                } else {
                    &self.uncacheable
                };
                counter.fetch_add(1, Ordering::Relaxed);
                if template.len() <= MAX_TEMPLATE_BYTES {
                    let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
                    if plans.len() >= MAX_PLANS {
                        plans.clear();
                    }
                    plans.insert(template, e.clone());
                }
                e
            }
        };
        let Entry::Plan(plan) = &*entry else {
            return Ok(None);
        };
        let bound = plan
            .clone()
            .transform_down(|node| {
                let any: &dyn Any = &*node;
                match any.downcast_ref::<SegScanExec>() {
                    Some(scan) => {
                        let (bound, _) = scan.bind(&|m| bound_literal(m, &lits));
                        Ok(Transformed::yes(Arc::new(bound) as Arc<dyn ExecutionPlan>))
                    }
                    None => Ok(Transformed::no(node)),
                }
            })?
            .data;
        let fresh = reset_plan_states(bound)?;
        let stream = execute_stream(fresh, ctx.task_ctx())?;
        Ok(Some(collect_bounded(stream, max_rows).await?))
    }
}

fn bound_literal(m: &str, lits: &[String]) -> Option<String> {
    let i: usize = m.strip_prefix(MARK)?.strip_suffix("__")?.parse().ok()?;
    lits.get(i).cloned()
}

async fn plan_template(
    ctx: &SessionContext,
    options: SQLOptions,
    template: &str,
    literals: usize,
) -> Entry {
    let Ok(df) = ctx.sql_with_options(template, options).await else {
        return Entry::Uncacheable;
    };
    let Ok(plan) = df.create_physical_plan().await else {
        return Entry::Uncacheable;
    };
    if markers_only_in_scans(&plan, literals) {
        Entry::Plan(plan)
    } else {
        Entry::Uncacheable
    }
}

/// Every one of `literals` markers appears exactly once in the plan, in a
/// scan's predicates.
fn markers_only_in_scans(plan: &Arc<dyn ExecutionPlan>, literals: usize) -> bool {
    let mut in_scans = vec![0usize; literals];
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        let any: &dyn Any = &**node;
        if let Some(scan) = any.downcast_ref::<SegScanExec>() {
            for p in scan.preds() {
                if let Some(i) = marker_index(p.literal()) {
                    if i >= literals {
                        return false;
                    }
                    in_scans[i] += 1;
                }
            }
        }
        stack.extend(node.children());
    }
    let shown = displayable(plan.as_ref()).indent(false).to_string();
    let anywhere = shown.matches(MARK).count();
    in_scans.iter().all(|&n| n == 1) && anywhere == literals
}

fn marker_index(lit: &str) -> Option<usize> {
    lit.strip_prefix(MARK)?.strip_suffix("__")?.parse().ok()
}
