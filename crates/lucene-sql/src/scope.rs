//! The memory scope of the request a statement runs for
//! ([`lucene_util::memory`]), carried to the threads DataFusion runs scans on
//! and checked between the windows a scan reads.
//!
//! The host opens a scope on the thread that calls the engine; what that
//! thread allocates is charged to the request, and the host's cancel and
//! deadline reach it. DataFusion hands scans to its own worker threads, which
//! know nothing of the caller: the scope travels with the task context and a
//! scan enters it before it reads.

use std::sync::Arc;

use datafusion::execution::TaskContext;
use datafusion::prelude::SessionContext;
use lucene_util::memory::{self, Exceeded, Reason, Scope};

use crate::error::Result;

/// The scope in a task context's session configuration.
pub(crate) struct ScopeExt(pub Arc<Scope>);

/// A task context for `ctx` that carries this thread's scope, when it has one.
pub(crate) fn task_ctx(ctx: &SessionContext) -> Arc<TaskContext> {
    let task = ctx.task_ctx();
    let Some(scope) = memory::current() else {
        return task;
    };
    match Arc::try_unwrap(task) {
        Ok(task) => {
            let config = task
                .session_config()
                .clone()
                .with_extension(Arc::new(ScopeExt(scope)));
            Arc::new(task.with_session_config(config))
        }
        Err(shared) => shared,
    }
}

/// The scope a task context carries.
pub(crate) fn of(task: &TaskContext) -> Option<Arc<Scope>> {
    task.session_config()
        .get_extension::<ScopeExt>()
        .map(|e| Arc::clone(&e.0))
}

/// Whether the request may go on: an error once the host cancelled it,
/// refused it memory, or its deadline passed. Free outside a scope.
pub(crate) fn stop() -> Result<()> {
    memory::checkpoint()?;
    if memory::past_deadline() {
        return Err(Exceeded {
            reason: Reason::Cancelled,
            message: "the statement ran past its time limit".to_string(),
        }
        .into());
    }
    Ok(())
}
