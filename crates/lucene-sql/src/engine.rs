//! The SQL engine over one snapshot: the router for the hot shapes, and
//! DataFusion for everything else.

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::error::ArrowError;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::{SQLOptions, SessionConfig, SessionContext};

use crate::error::{Error, Result};
use crate::group_rule::{is_inexact, GroupByKernelRule};
use crate::plan_cache::PlanCache;
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
    /// Keep physical plans by statement shape so a repeated shape is not
    /// planned again.
    pub plan_cache: bool,
    /// Rows a statement may return; one more fails it, so a statement with no
    /// `LIMIT` over a big table does not collect the table.
    pub max_rows: usize,
}

impl Default for SqlConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, usize::from);
        Self {
            memory_limit: 512 << 20,
            worker_threads: cpus,
            target_partitions: cpus,
            plan_cache: true,
            max_rows: 10_000,
        }
    }
}

/// What the group-by kernels are assumed to hold per group (its buckets, their
/// metric states and the result arrays), to turn the memory limit into the most
/// groups a statement may make.
const GROUP_BYTES: usize = 256;

/// Which path answered a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// A hot shape, answered by the native API directly.
    Router,
    /// DataFusion.
    Standard,
}

/// The batches of a statement as DataFusion produces them.
struct SqlReader {
    engine: Arc<SqlEngine>,
    sql: String,
    schema: SchemaRef,
    stream: SendableRecordBatchStream,
    emitted: bool,
}

impl Iterator for SqlReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        use futures::StreamExt;
        loop {
            let item = self.engine.runtime.block_on(self.stream.next())?;
            match item {
                Ok(batch) => {
                    self.emitted = true;
                    return Some(Ok(batch));
                }
                // The kernels could not answer exactly and nothing was sent
                // yet: run the statement the standard way.
                Err(e) if !self.emitted && is_inexact(&e) => {
                    let opened = self
                        .engine
                        .runtime
                        .block_on(self.engine.open_stream(&self.engine.plain, &self.sql));
                    match opened {
                        Ok((_, stream)) => self.stream = stream,
                        Err(e) => return Some(Err(ArrowError::ExternalError(Box::new(e)))),
                    }
                }
                Err(e) => return Some(Err(ArrowError::ExternalError(Box::new(e)))),
            }
        }
    }
}

impl RecordBatchReader for SqlReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Runs read-only SQL over a [`Snapshot`].
pub struct SqlEngine {
    snapshot: Arc<Snapshot>,
    router: Router,
    ctx: SessionContext,
    /// The same tables without the aggregation-kernel rule, for statements
    /// the kernels cannot answer exactly.
    plain: SessionContext,
    runtime: Arc<tokio::runtime::Runtime>,
    plans: Option<PlanCache>,
    max_rows: usize,
}

/// Stack of the threads that run a statement with a long operator chain: the planner,
/// the expression evaluator and the tree's drop recurse over the chain, and the
/// calling thread (a host thread with a 1 MiB stack) cannot hold a chain of a
/// thousand terms. Address space only, committed as it is touched.
const DEEP_STACK: usize = 64 << 20;

/// Operators a statement may chain and still run on the caller's stack: release needs about
/// 2-8 KiB per operator and an unoptimised build several times that, so 64 stay under 1 MiB.
const SHALLOW_OPERATORS: usize = 64;

/// Runs `f` on the calling thread, or on a deep-stack thread carrying the same memory scope
/// when the statement chains more than [`SHALLOW_OPERATORS`] operators.
fn with_stack<T: Send>(operators: usize, f: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    if operators <= SHALLOW_OPERATORS {
        return f();
    }
    let scope = lucene_util::memory::current();
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn_scoped(s, move || lucene_util::memory::within(scope.as_ref(), f))
            .map_err(|e| Error::unsupported(format!("starting a statement thread: {e}")))?
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
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
        Self::with_runtime(snapshot, config, Arc::new(Self::runtime_for(config)?))
    }

    /// The runtime `config` asks for, to share between engines.
    ///
    /// # Errors
    /// The runtime could not start.
    pub fn runtime_for(config: &SqlConfig) -> Result<tokio::runtime::Runtime> {
        if config.worker_threads == 0 {
            tokio::runtime::Builder::new_current_thread().build()
        } else {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(config.worker_threads)
                .thread_stack_size(DEEP_STACK)
                .build()
        }
        .map_err(|e| Error::unsupported(format!("starting the runtime: {e}")))
    }

    /// An engine over `snapshot` running its scans on `runtime`, which many
    /// engines (one per snapshot) can share.
    ///
    /// # Errors
    /// DataFusion could not register the table.
    pub fn with_runtime(
        snapshot: Arc<Snapshot>,
        config: &SqlConfig,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Result<Self> {
        snapshot.set_group_limit((config.memory_limit / GROUP_BYTES).max(1));
        let ctx = Self::context_for(&snapshot, config, true)?;
        let plain = Self::context_for(&snapshot, config, false)?;
        Ok(Self {
            router: Router::new(snapshot.clone()),
            snapshot,
            ctx,
            plain,
            runtime,
            plans: config.plan_cache.then(PlanCache::default),
            max_rows: config.max_rows,
        })
    }

    fn context_for(
        snapshot: &Arc<Snapshot>,
        config: &SqlConfig,
        kernels: bool,
    ) -> Result<SessionContext> {
        let env = RuntimeEnvBuilder::new()
            .with_memory_limit(config.memory_limit, 1.0)
            .build_arc()?;
        let session = SessionConfig::new().with_target_partitions(config.target_partitions.max(1));
        let mut builder = SessionStateBuilder::new()
            .with_config(session)
            .with_runtime_env(env)
            .with_default_features()
            .with_optimizer_rule(Arc::new(ScoreTopKRule));
        if kernels {
            builder = builder.with_optimizer_rule(Arc::new(GroupByKernelRule));
        }
        let ctx = SessionContext::new_with_state(builder.build());
        crate::bounded::bound_functions(
            &ctx,
            crate::bounded::Limits::from_memory_limit(config.memory_limit),
        );
        ctx.register_udf(match_text_udf());
        ctx.register_udf(knn_udf());
        ctx.register_table(
            snapshot.spec().name(),
            Arc::new(LuceneTable::new(snapshot.clone())),
        )?;
        Ok(ctx)
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

    /// The context without the aggregation-kernel rule, for tests and
    /// benchmarks that compare against DataFusion's own aggregates.
    pub fn plain_context(&self) -> &SessionContext {
        &self.plain
    }

    /// Runs `sql`: a recognised shape on the native API, anything else on
    /// DataFusion. Only queries run; DDL, DML and session statements are
    /// refused.
    ///
    /// # Errors
    /// A statement that does not parse or plan, or an unreadable index.
    pub async fn query(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        crate::statement::check_operators(sql)?;
        if let Some(batch) = self.router.run(sql)? {
            return Ok(vec![batch]);
        }
        self.standard(sql).await
    }

    /// How the plan cache has been used (`None` when it is off).
    pub fn plan_cache_stats(&self) -> Option<crate::plan_cache::PlanCacheStats> {
        self.plans.as_ref().map(PlanCache::stats)
    }

    /// Runs `sql` through DataFusion only, as the fallback path does: from a
    /// cached plan when its shape has one, else planned from the text.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub async fn standard(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        crate::statement::check_operators(sql)?;
        if let Some(plans) = &self.plans {
            match plans.run(&self.ctx, read_only(), sql, self.max_rows).await {
                Ok(Some(batches)) => return Ok(batches),
                Ok(None) => {}
                Err(Error::DataFusion(e)) if is_inexact(&e) => return self.plain_run(sql).await,
                Err(e) => return Err(e),
            }
        }
        self.uncached(sql).await
    }

    /// Plans `sql` from its text and runs it, bypassing the plan cache.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub async fn uncached(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        crate::statement::check_operators(sql)?;
        match self.run_on(&self.ctx, sql).await {
            Err(Error::DataFusion(e)) if is_inexact(&e) => self.plain_run(sql).await,
            other => other,
        }
    }

    async fn plain_run(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        self.run_on(&self.plain, sql).await
    }

    async fn run_on(&self, ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
        let df = ctx.sql_with_options(sql, read_only()).await?;
        let plan = df.create_physical_plan().await?;
        let stream = execute_stream(plan, crate::scope::task_ctx(ctx))?;
        collect_bounded(stream, self.max_rows).await
    }

    /// [`Self::query`] from synchronous code; must not be called from inside a
    /// tokio runtime.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub fn query_blocking(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        self.query_traced(sql).map(|(batches, _)| batches)
    }

    /// [`Self::query_blocking`], also saying which path answered. The result
    /// always has a batch, so its schema is known when no row matched.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub fn query_traced(&self, sql: &str) -> Result<(Vec<RecordBatch>, Path)> {
        let operators = crate::statement::check_operators(sql)?;
        crate::scope::stop()?;
        if let Some(batch) = self.router.run(sql)? {
            return Ok((vec![batch], Path::Router));
        }
        let batches = with_stack(operators, || self.runtime.block_on(self.standard(sql)))?;
        Ok((batches, Path::Standard))
    }

    /// The rows of `sql` as a stream, for a consumer that reads Arrow batches
    /// as they are produced (the Arrow C stream interface): a recognised
    /// shape answers in one batch, anything else is executed on DataFusion
    /// and its batches are not collected first. Must not be called from
    /// inside a tokio runtime.
    ///
    /// # Errors
    /// A statement that does not parse or plan, or a router failure.
    pub fn stream_blocking(
        self: &Arc<Self>,
        sql: &str,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let operators = crate::statement::check_operators(sql)?;
        if let Some(batch) = self.router.run(sql)? {
            let schema = batch.schema();
            return Ok(Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema)));
        }
        let (schema, stream) = with_stack(operators, || {
            self.runtime.block_on(self.open_stream(&self.ctx, sql))
        })?;
        Ok(Box::new(SqlReader {
            engine: self.clone(),
            sql: sql.to_string(),
            schema,
            stream,
            emitted: false,
        }))
    }

    async fn open_stream(
        &self,
        ctx: &SessionContext,
        sql: &str,
    ) -> Result<(SchemaRef, SendableRecordBatchStream)> {
        let df = ctx.sql_with_options(sql, read_only()).await?;
        let schema: SchemaRef = Arc::new(df.schema().as_arrow().clone());
        let plan = df.create_physical_plan().await?;
        Ok((schema, execute_stream(plan, crate::scope::task_ctx(ctx))?))
    }

    /// [`Self::standard`] from synchronous code.
    ///
    /// # Errors
    /// As [`Self::query`].
    pub fn standard_blocking(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let operators = crate::statement::check_operators(sql)?;
        with_stack(operators, || self.runtime.block_on(self.standard(sql)))
    }
}

/// Collects `stream`, failing when it holds more than `max_rows` rows.
pub(crate) async fn collect_bounded(
    mut stream: datafusion::execution::SendableRecordBatchStream,
    max_rows: usize,
) -> Result<Vec<RecordBatch>> {
    use futures::StreamExt;
    let schema = stream.schema();
    let mut rows = 0usize;
    let mut out = Vec::new();
    while let Some(batch) = stream.next().await {
        crate::scope::stop()?;
        let batch = batch?;
        rows = rows.saturating_add(batch.num_rows());
        if rows > max_rows {
            return Err(Error::unsupported(format!(
                "the result has more than {max_rows} rows; add a LIMIT"
            )));
        }
        out.push(batch);
    }
    if out.is_empty() {
        out.push(RecordBatch::new_empty(schema));
    }
    Ok(out)
}
