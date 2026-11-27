//! The SQL engine over one snapshot: the router for the hot shapes, and
//! DataFusion for everything else.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SQLOptions, SessionConfig, SessionContext};

use crate::error::{Error, Result};
use crate::pred::{knn_udf, match_text_udf};
use crate::provider::LuceneTable;
use crate::router::Router;
use crate::rule::ScoreTopKRule;
use crate::snapshot::Snapshot;

/// Settings of a [`SqlEngine`].
#[derive(Debug, Clone)]
pub struct SqlConfig {
    /// Bytes DataFusion's operators (sorts, joins, aggregations) may hold in
    /// memory together; a statement that needs more fails instead of growing.
    pub memory_limit: usize,
    /// Threads [`SqlEngine::query_blocking`] runs scans on (`0`: the caller's
    /// thread only).
    pub worker_threads: usize,
    /// Scan partitions DataFusion plans for.
    pub target_partitions: usize,
}

impl Default for SqlConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, usize::from);
        Self {
            memory_limit: 512 << 20,
            worker_threads: cpus,
            target_partitions: cpus,
        }
    }
}

/// Runs read-only SQL over a [`Snapshot`].
pub struct SqlEngine {
    snapshot: Arc<Snapshot>,
    router: Router,
    ctx: SessionContext,
    runtime: tokio::runtime::Runtime,
}

fn read_only() -> SQLOptions {
    SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false)
}

impl SqlEngine {
    /// An engine over `snapshot`.
    ///
    /// # Errors
    /// DataFusion could not register the table or start its runtime.
    pub fn new(snapshot: Arc<Snapshot>, config: &SqlConfig) -> Result<Self> {
        let env = RuntimeEnvBuilder::new()
            .with_memory_limit(config.memory_limit, 1.0)
            .build_arc()?;
        let session = SessionConfig::new().with_target_partitions(config.target_partitions.max(1));
        let state = SessionStateBuilder::new()
            .with_config(session)
            .with_runtime_env(env)
            .with_default_features()
            .with_optimizer_rule(Arc::new(ScoreTopKRule))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(match_text_udf());
        ctx.register_udf(knn_udf());
        ctx.register_table(
            snapshot.spec().name(),
            Arc::new(LuceneTable::new(snapshot.clone())),
        )?;
        let runtime = if config.worker_threads == 0 {
            tokio::runtime::Builder::new_current_thread().build()
        } else {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(config.worker_threads)
                .build()
        }
        .map_err(|e| Error::unsupported(format!("starting the runtime: {e}")))?;
        Ok(Self {
            router: Router::new(snapshot.clone()),
            snapshot,
            ctx,
            runtime,
        })
    }

    /// The snapshot queried.
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        &self.snapshot
    }

    /// The router, to ask which shape a statement is.
    pub fn router(&self) -> &Router {
        &self.router
    }

    /// The DataFusion context, for tests and EXPLAIN.
    pub fn context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Runs `sql`: a recognised shape on the native API, anything else on
    /// DataFusion. Only queries run; DDL, DML and session statements are
    /// refused.
    ///
    /// # Errors
    /// A statement that does not parse or plan, or an unreadable index.
    pub async fn query(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        if let Some(batch) = self.router.run(sql)? {
            return Ok(vec![batch]);
        }
        self.standard(sql).await
    }

    /// Runs `sql` through DataFusion only, as the fallback path does.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub async fn standard(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql_with_options(sql, read_only()).await?;
        Ok(df.collect().await?)
    }

    /// [`Self::query`] from synchronous code; must not be called from inside a
    /// tokio runtime.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub fn query_blocking(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        if let Some(batch) = self.router.run(sql)? {
            return Ok(vec![batch]);
        }
        self.runtime.block_on(self.standard(sql))
    }

    /// [`Self::standard`] from synchronous code.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub fn standard_blocking(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        self.runtime.block_on(self.standard(sql))
    }
}
