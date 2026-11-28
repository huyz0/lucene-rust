//! The table as a stream of Arrow record batches, for tools that read Arrow
//! (pyarrow, Polars, DuckDB, Spark): column pruning, filters and a row limit
//! are applied inside the scan, so only the batches a consumer asked for are
//! decoded.
//!
//! A [`Scan`] is a declaration (columns, filters, limit, batch size) over a
//! [`Snapshot`]; [`Scan::open`] starts a reader over some of its segments on
//! a thread of its own, which decodes the next batches while the consumer
//! works on the current one. The reader is an Arrow
//! [`RecordBatchReader`], which `arrow::ffi_stream` exports as the C stream
//! interface.
//!
//! A batch holds up to [`ScanOptions::batch_rows`] consecutive documents of
//! one segment. A numeric column is decoded straight into the Arrow buffer
//! (`NumericReader::fill_window`); a keyword column is the segment's term
//! dictionary and one `Int32` key per row, or a `Utf8` array when
//! [`ScanOptions::dictionary`] is off. Filters run on the decoded doc values
//! of their window (a keyword's term comparison is a comparison of
//! ordinals), and a window nothing in survives is not read for the other
//! columns.

use std::collections::HashMap;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, RecordBatchReader,
    StringArray, StringBuilder,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use arrow::error::ArrowError;

use crate::columns::Columns;
use crate::error::{Error, Result};
use crate::native::matching_docs;
use crate::snapshot::Snapshot;
use crate::spec::{ColumnKind, SCORE_COLUMN};

/// Rows per batch unless [`ScanOptions::batch_rows`] says otherwise.
pub const DEFAULT_BATCH_ROWS: usize = 65_536;

/// Batches each decoding thread keeps ready ahead of the consumer.
const READY_BATCHES: usize = 2;

/// Most threads a reader starts when [`ScanOptions::threads`] is `0`.
pub const MAX_AUTO_THREADS: usize = 8;

/// Keyword columns whose segments hold more terms than this together keep a
/// dictionary per segment instead of one for the scan.
const MAX_GLOBAL_TERMS: usize = 1 << 20;

/// How a filter compares a column with its literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `=`
    Eq,
    /// `<>`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// A literal a filter compares a column with.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// For a `long` column.
    Long(i64),
    /// For a keyword or key column.
    Str(String),
}

/// A condition a row must meet. A row with no value in the column meets
/// none but [`Filter::IsNull`] (SQL's NULL comparisons).
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// `column <op> literal`. A key column (stored, not doc values) answers
    /// `=` only.
    Cmp {
        /// The column.
        column: String,
        /// The comparison.
        op: Op,
        /// What it compares with.
        value: Literal,
    },
    /// `column IN (values)`.
    In {
        /// The column.
        column: String,
        /// The accepted values.
        values: Vec<Literal>,
    },
    /// `column IS NULL`.
    IsNull {
        /// The column (not a key column).
        column: String,
    },
    /// `column IS NOT NULL`.
    IsNotNull {
        /// The column (not a key column).
        column: String,
    },
}

/// What a [`Scan`] reads.
#[derive(Clone)]
pub struct ScanOptions {
    /// The columns to return, in this order; `None` means every readable one
    /// and `Some(vec![])` none (batches that only count rows).
    pub columns: Option<Vec<String>>,
    /// Every filter must hold.
    pub filters: Vec<Filter>,
    /// Rows to return at most, per reader.
    pub limit: Option<usize>,
    /// Rows a batch holds at most (rounded up to a multiple of 64).
    pub batch_rows: usize,
    /// Return keyword columns as `Dictionary(Int32, Utf8)`: the term
    /// dictionary of the segment and a key per row, which is what the
    /// doc values hold and costs no per-row string work.
    pub dictionary: bool,
    /// Threads decoding windows side by side for one reader (batches still
    /// come out in document order); `0` picks one per available core, at
    /// most [`MAX_AUTO_THREADS`]. A consumer that reads several partitions
    /// at once sets `1`.
    pub threads: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            columns: None,
            filters: Vec::new(),
            limit: None,
            batch_rows: DEFAULT_BATCH_ROWS,
            dictionary: true,
            threads: 0,
        }
    }
}

/// A resolved filter: columns by index, keyword literals not yet turned into
/// ordinals (that is per segment).
#[derive(Debug, Clone)]
enum Cond {
    Doc {
        column: usize,
        test: Test,
    },
    /// Documents found by a term search (a key column's `=` / `IN`).
    Search {
        column: usize,
        values: Vec<String>,
    },
}

#[derive(Debug, Clone)]
enum Test {
    Long(Vec<(Op, i64)>),
    Terms(Vec<(Op, String)>),
    Null(bool),
}

/// The scan of one table.
#[derive(Clone)]
pub struct Scan {
    snapshot: Arc<Snapshot>,
    projection: Vec<usize>,
    conds: Vec<Cond>,
    schema: SchemaRef,
    limit: Option<usize>,
    batch_rows: usize,
    dictionary: bool,
    threads: usize,
}

impl Scan {
    /// Resolves `options` against the table.
    ///
    /// # Errors
    /// An unknown or unreadable column, or a literal of the wrong type.
    pub fn new(snapshot: Arc<Snapshot>, options: &ScanOptions) -> Result<Self> {
        let spec = snapshot.spec().clone();
        let projection: Vec<usize> = if let Some(names) = &options.columns {
            names
                .iter()
                .map(|name| {
                    let c = spec.index_of(name).filter(|&c| c != spec.score_index());
                    match c {
                        Some(c) if spec.columns()[c].kind.is_readable() => Ok(c),
                        Some(_) => Err(Error::unsupported(format!(
                            "column `{name}` is a search column and has no values to read"
                        ))),
                        None if name == SCORE_COLUMN => Err(Error::unsupported(
                            "`_score` needs a match or knn query; a scan has no score",
                        )),
                        None => Err(Error::unsupported(format!("no column `{name}`"))),
                    }
                })
                .collect::<Result<_>>()?
        } else {
            spec.readable_columns()
                .into_iter()
                .filter(|&c| c != spec.score_index())
                .collect()
        };
        let dictionary = options.dictionary;
        let fields: Vec<Field> = projection
            .iter()
            .map(|&c| {
                let col = &spec.columns()[c];
                let ty = match col.kind {
                    ColumnKind::Long => DataType::Int64,
                    ColumnKind::Keyword if dictionary => {
                        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
                    }
                    _ => DataType::Utf8,
                };
                Field::new(&col.name, ty, true)
            })
            .collect();
        let conds = options
            .filters
            .iter()
            .map(|f| resolve(&snapshot, f))
            .collect::<Result<_>>()?;
        Ok(Self {
            snapshot,
            projection,
            conds,
            schema: Arc::new(Schema::new(fields)),
            limit: options.limit,
            batch_rows: options.batch_rows.max(1).next_multiple_of(64),
            dictionary,
            threads: options.threads,
        })
    }

    /// The schema of the batches.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// The segments split into at most `n` groups of similar size, for `n`
    /// readers that run side by side.
    pub fn partitions(&self, n: usize) -> Vec<Vec<usize>> {
        let n = n.max(1);
        let mut groups: Vec<(i64, Vec<usize>)> = (0..n).map(|_| (0, Vec::new())).collect();
        let mut segments: Vec<(usize, i32)> = (0..self.snapshot.segment_count())
            .map(|s| (s, self.snapshot.leaf_max_doc(s)))
            .collect();
        segments.sort_by_key(|&(s, d)| (std::cmp::Reverse(d), s));
        for (s, docs) in segments {
            if let Some(g) = groups.iter_mut().min_by_key(|g| g.0) {
                g.0 += i64::from(docs);
                g.1.push(s);
            }
        }
        let mut out: Vec<Vec<usize>> = groups
            .into_iter()
            .map(|(_, mut v)| {
                v.sort_unstable();
                v
            })
            .filter(|v| !v.is_empty())
            .collect();
        out.sort();
        out
    }

    /// A reader over all segments.
    pub fn open(&self) -> ScanReader {
        self.open_segments((0..self.snapshot.segment_count()).collect())
    }

    /// A reader over `segments` (indices; see [`Self::partitions`]).
    pub fn open_segments(&self, segments: Vec<usize>) -> ScanReader {
        let job = match Job::new(self.clone(), &segments) {
            Ok(job) => Arc::new(job),
            Err(e) => return ScanReader::failed(self.schema.clone(), e),
        };
        let total = job.windows.len();
        let threads = if self.threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            self.threads
        };
        let auto_cap = if self.threads == 0 {
            MAX_AUTO_THREADS
        } else {
            usize::MAX
        };
        let threads = threads.min(auto_cap).min(total).max(1);
        let mut rxs = Vec::with_capacity(threads);
        let mut workers = Vec::with_capacity(threads);
        for k in 0..threads {
            let (tx, rx) = sync_channel(READY_BATCHES);
            let job = job.clone();
            let spawned = std::thread::Builder::new()
                .name("arrow-scan".into())
                .spawn(move || job.work(k, threads, &tx));
            match spawned {
                Ok(h) => {
                    rxs.push(rx);
                    workers.push(h);
                }
                Err(e) => {
                    drop(rx);
                    drop(rxs);
                    return ScanReader::failed(
                        self.schema.clone(),
                        Error::unsupported(format!("starting a scan thread: {e}")),
                    );
                }
            }
        }
        ScanReader {
            schema: self.schema.clone(),
            rxs,
            workers,
            seq: 0,
            total,
            left: self.limit.unwrap_or(usize::MAX),
            failure: None,
        }
    }
}

type Msg = std::result::Result<Option<RecordBatch>, ArrowError>;

/// The batches of a [`Scan`], decoded ahead of the consumer by worker
/// threads that each take every `threads`-th window; the reader takes them
/// back in window order.
pub struct ScanReader {
    schema: SchemaRef,
    rxs: Vec<Receiver<Msg>>,
    workers: Vec<JoinHandle<()>>,
    seq: usize,
    total: usize,
    left: usize,
    failure: Option<ArrowError>,
}

impl ScanReader {
    fn failed(schema: SchemaRef, e: Error) -> Self {
        Self {
            schema,
            rxs: Vec::new(),
            workers: Vec::new(),
            seq: 0,
            total: 0,
            left: 0,
            failure: Some(ArrowError::ExternalError(Box::new(e))),
        }
    }
}

impl Iterator for ScanReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.failure.take() {
            self.total = 0;
            return Some(Err(e));
        }
        while self.seq < self.total && self.left > 0 {
            let k = self.seq % self.rxs.len();
            match self.rxs[k].recv() {
                Ok(Ok(Some(batch))) => {
                    self.seq += 1;
                    let keep = batch.num_rows().min(self.left);
                    self.left -= keep;
                    return Some(Ok(if keep < batch.num_rows() {
                        batch.slice(0, keep)
                    } else {
                        batch
                    }));
                }
                Ok(Ok(None)) => self.seq += 1,
                Ok(Err(e)) => {
                    self.total = 0;
                    return Some(Err(e));
                }
                Err(_) => {
                    self.total = 0;
                    return Some(Err(ArrowError::ComputeError(
                        "a scan thread stopped before its windows were read".into(),
                    )));
                }
            }
        }
        None
    }
}

impl RecordBatchReader for ScanReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for ScanReader {
    fn drop(&mut self) {
        // Each worker stops at its next send once its receiver is gone.
        self.rxs.clear();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn resolve(snapshot: &Snapshot, f: &Filter) -> Result<Cond> {
    let spec = snapshot.spec();
    let find = |name: &str| -> Result<(usize, ColumnKind)> {
        spec.columns()
            .iter()
            .position(|c| c.name == name)
            .map(|i| (i, spec.columns()[i].kind.clone()))
            .ok_or_else(|| Error::unsupported(format!("no column `{name}`")))
    };
    let wrong = |name: &str, what: &str| Error::unsupported(format!("column `{name}`: {what}"));
    match f {
        Filter::Cmp { column, op, value } => {
            let (c, kind) = find(column)?;
            match (&kind, value) {
                (ColumnKind::Long, Literal::Long(v)) => Ok(Cond::Doc {
                    column: c,
                    test: Test::Long(vec![(*op, *v)]),
                }),
                (ColumnKind::Keyword, Literal::Str(v)) => Ok(Cond::Doc {
                    column: c,
                    test: Test::Terms(vec![(*op, v.clone())]),
                }),
                (ColumnKind::Key, Literal::Str(v)) if *op == Op::Eq => Ok(Cond::Search {
                    column: c,
                    values: vec![v.clone()],
                }),
                (ColumnKind::Key, _) => Err(wrong(column, "a key column answers = and IN")),
                (ColumnKind::Long, _) => Err(wrong(column, "compare a long column with a number")),
                (ColumnKind::Keyword, _) => Err(wrong(column, "compare a keyword with a string")),
                _ => Err(wrong(column, "a search column has no values to filter on")),
            }
        }
        Filter::In { column, values } => {
            let (c, kind) = find(column)?;
            match kind {
                ColumnKind::Long => {
                    let v = values
                        .iter()
                        .map(|l| match l {
                            Literal::Long(v) => Ok((Op::Eq, *v)),
                            Literal::Str(_) => Err(wrong(column, "IN a list of numbers")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Cond::Doc {
                        column: c,
                        test: Test::Long(v),
                    })
                }
                ColumnKind::Keyword | ColumnKind::Key => {
                    let v = values
                        .iter()
                        .map(|l| match l {
                            Literal::Str(s) => Ok(s.clone()),
                            Literal::Long(_) => Err(wrong(column, "IN a list of strings")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if kind == ColumnKind::Key {
                        Ok(Cond::Search {
                            column: c,
                            values: v,
                        })
                    } else {
                        Ok(Cond::Doc {
                            column: c,
                            test: Test::Terms(v.into_iter().map(|s| (Op::Eq, s)).collect()),
                        })
                    }
                }
                _ => Err(wrong(column, "a search column has no values to filter on")),
            }
        }
        Filter::IsNull { column } | Filter::IsNotNull { column } => {
            let (c, kind) = find(column)?;
            if !matches!(kind, ColumnKind::Long | ColumnKind::Keyword) {
                return Err(wrong(column, "IS NULL needs a long or keyword column"));
            }
            Ok(Cond::Doc {
                column: c,
                test: Test::Null(matches!(f, Filter::IsNull { .. })),
            })
        }
    }
}

/// What every worker of one reader shares: the windows to read and what is
/// computed once for them.
struct Job {
    scan: Scan,
    /// `(segment, first doc, docs)` of every window, in output order.
    windows: Vec<(usize, usize, usize)>,
    /// Documents the key-column filters select, ascending.
    searched: Option<Vec<i32>>,
    /// The doc-values tests of each segment, with keyword terms as ordinals.
    tests: HashMap<usize, Vec<(usize, SegTest)>>,
    /// One dictionary for the whole scan for each keyword column that is
    /// small enough, so every batch of a column shares it.
    globals: HashMap<usize, GlobalDict>,
}

struct GlobalDict {
    values: Arc<StringArray>,
    /// `remap[segment][ordinal]`: the ordinal in `values`.
    remap: HashMap<usize, Vec<i32>>,
}

/// A range test on doc values: `value - lo` (wrapping, as unsigned) `< width`,
/// optionally negated; the sets of the comparisons are all intervals.
#[derive(Clone, Copy)]
struct Interval {
    lo: i64,
    width: u64,
}

impl Interval {
    const EMPTY: Interval = Interval { lo: 0, width: 0 };

    fn new(lo: i64, hi: i64) -> Self {
        if lo > hi {
            Self::EMPTY
        } else {
            Self {
                lo,
                width: (hi as u64).wrapping_sub(lo as u64).wrapping_add(1),
            }
        }
    }

    /// Whether some value in `lo..=hi` meets the interval.
    fn overlaps(self, lo: i64, hi: i64) -> bool {
        if self.width == 0 {
            return false;
        }
        let end = i128::from(self.lo) + i128::from(self.width) - 1;
        i128::from(lo) <= end && i128::from(hi) >= i128::from(self.lo)
    }

    /// `width == 0` is also the interval of the whole range (2^64 values);
    /// every other whole-range comparison is built as two intervals avoided
    /// below, so the empty reading is the only one.
    #[inline(always)]
    fn holds(self, v: i64) -> bool {
        (v as u64).wrapping_sub(self.lo as u64) < self.width
    }
}

/// The intervals of `v <op> x` over `i64`, as at most two.
fn intervals(op: Op, x: i64) -> [Interval; 2] {
    let none = Interval::EMPTY;
    match op {
        Op::Eq => [Interval::new(x, x), none],
        Op::Lt => [
            if x == i64::MIN {
                none
            } else {
                Interval::new(i64::MIN, x - 1)
            },
            none,
        ],
        Op::Le if x == i64::MAX => [
            Interval {
                lo: i64::MIN,
                width: u64::MAX,
            },
            Interval::new(i64::MAX, i64::MAX),
        ],
        Op::Le => [Interval::new(i64::MIN, x), none],
        Op::Gt => [
            if x == i64::MAX {
                none
            } else {
                Interval::new(x + 1, i64::MAX)
            },
            none,
        ],
        Op::Ge if x == i64::MIN => [
            Interval {
                lo: i64::MIN,
                width: u64::MAX,
            },
            Interval::new(i64::MAX, i64::MAX),
        ],
        Op::Ge => [Interval::new(x, i64::MAX), none],
        Op::Ne => [
            if x == i64::MIN {
                none
            } else {
                Interval::new(i64::MIN, x - 1)
            },
            if x == i64::MAX {
                none
            } else {
                Interval::new(x + 1, i64::MAX)
            },
        ],
    }
}

/// Bits `i` of `out` set where `holds(vals[i])`, for all of `vals`.
fn mark<const K: usize>(vals: &[i64], set: [Interval; K], out: &mut [u64]) {
    for (w, chunk) in out.iter_mut().zip(vals.chunks(64)) {
        let mut bits = 0u64;
        for (j, &v) in chunk.iter().enumerate() {
            let mut hit = false;
            for iv in &set {
                hit |= iv.holds(v);
            }
            bits |= u64::from(hit) << j;
        }
        *w = bits;
    }
}

fn mark_any(vals: &[i64], set: &[Interval], out: &mut [u64]) {
    match set.len() {
        0 => out.fill(0),
        1 => mark::<1>(vals, [set[0]], out),
        2 => mark::<2>(vals, [set[0], set[1]], out),
        3 => mark::<3>(vals, [set[0], set[1], set[2]], out),
        4 => mark::<4>(vals, [set[0], set[1], set[2], set[3]], out),
        _ => {
            for (w, chunk) in out.iter_mut().zip(vals.chunks(64)) {
                let mut bits = 0u64;
                for (j, &v) in chunk.iter().enumerate() {
                    bits |= u64::from(set.iter().any(|iv| iv.holds(v))) << j;
                }
                *w = bits;
            }
        }
    }
}

/// Merges overlapping and touching intervals so a long `IN` list is a few
/// ranges, and a comparison chain stays small.
fn coalesce(mut set: Vec<Interval>) -> Vec<Interval> {
    set.retain(|iv| iv.width != 0);
    set.sort_by_key(|iv| iv.lo);
    let mut out: Vec<Interval> = Vec::with_capacity(set.len());
    for iv in set {
        if let Some(last) = out.last_mut() {
            let end = i128::from(last.lo) + i128::from(last.width);
            if i128::from(iv.lo) <= end {
                let new_end = (i128::from(iv.lo) + i128::from(iv.width)).max(end);
                last.width = u64::try_from(new_end - i128::from(last.lo)).unwrap_or(u64::MAX);
                continue;
            }
        }
        out.push(iv);
    }
    out
}

/// Per-window buffers of a decoded column.
struct Window {
    vals: Vec<i64>,
    present: Vec<u64>,
    /// The segment has no values for the column.
    absent: bool,
    /// Holds the current window's values (the buffers are kept from window
    /// to window, so only the first allocates).
    loaded: bool,
}

impl Window {
    fn new() -> Self {
        Self {
            vals: Vec::new(),
            present: Vec::new(),
            absent: true,
            loaded: false,
        }
    }
}

impl Job {
    fn new(scan: Scan, segments: &[usize]) -> Result<Self> {
        let mut job = Self {
            scan,
            windows: Vec::new(),
            searched: None,
            tests: HashMap::new(),
            globals: HashMap::new(),
        };
        job.searched = job.searched()?;
        let mut rows = job.scan.batch_rows;
        // Without filters the first `limit` rows are all a reader returns,
        // so it decodes a window of about that size and no more of them.
        let unfiltered = job.scan.conds.is_empty()
            && segments
                .iter()
                .all(|&s| job.scan.snapshot.leaf(s).live_docs().is_none());
        if let (Some(limit), true) = (job.scan.limit, unfiltered) {
            rows = rows.min(limit.max(1024).next_multiple_of(64));
        }
        for &seg in segments {
            let max = usize::try_from(job.scan.snapshot.leaf(seg).max_doc).unwrap_or(0);
            let mut start = 0;
            while start < max {
                let n = rows.min(max - start);
                job.windows.push((seg, start, n));
                start += n;
            }
            job.tests.insert(seg, job.segment_tests(seg));
        }
        if let (Some(limit), true) = (job.scan.limit, unfiltered) {
            let mut have = 0;
            job.windows.retain(|&(_, _, n)| {
                let keep = have < limit;
                have += n;
                keep
            });
        }
        if job.scan.dictionary {
            job.globals = job.global_dicts(segments);
        }
        Ok(job)
    }

    fn global_dicts(&self, segments: &[usize]) -> HashMap<usize, GlobalDict> {
        let snapshot = &*self.scan.snapshot;
        let spec = snapshot.spec();
        let mut out = HashMap::new();
        for &c in &self.scan.projection {
            if spec.columns()[c].kind != ColumnKind::Keyword {
                continue;
            }
            let total: usize = segments.iter().map(|&s| snapshot.terms(s, c).len()).sum();
            if total > MAX_GLOBAL_TERMS {
                continue;
            }
            let mut all: Vec<&str> = segments
                .iter()
                .flat_map(|&s| snapshot.terms(s, c).iter().map(String::as_str))
                .collect();
            all.sort_unstable();
            all.dedup();
            let remap = segments
                .iter()
                .map(|&s| {
                    let ords = snapshot
                        .terms(s, c)
                        .iter()
                        .map(|t| {
                            all.binary_search(&t.as_str())
                                .map_or(0, |i| i32::try_from(i).unwrap_or(0))
                        })
                        .collect();
                    (s, ords)
                })
                .collect();
            out.insert(
                c,
                GlobalDict {
                    values: Arc::new(StringArray::from_iter_values(all)),
                    remap,
                },
            );
        }
        out
    }

    /// Decodes windows `k`, `k + step`, ... and sends each batch (or `None`
    /// for a window nothing in survives) until the reader is gone.
    fn work(&self, k: usize, step: usize, tx: &std::sync::mpsc::SyncSender<Msg>) {
        let snapshot = self.scan.snapshot.clone();
        let mut cols = Columns::new(&snapshot);
        let mut dicts: HashMap<(usize, usize), Arc<StringArray>> = HashMap::new();
        let mut windows: HashMap<usize, Window> = HashMap::new();
        for idx in (k..self.windows.len()).step_by(step) {
            let msg = self
                .window(&mut cols, &mut dicts, &mut windows, idx)
                .map_err(|e| ArrowError::ExternalError(Box::new(e)));
            let failed = msg.is_err();
            if tx.send(msg).is_err() || failed {
                return;
            }
        }
    }

    fn window(
        &self,
        cols: &mut Columns<'_>,
        dicts: &mut HashMap<(usize, usize), Arc<StringArray>>,
        windows: &mut HashMap<usize, Window>,
        idx: usize,
    ) -> Result<Option<RecordBatch>> {
        let (seg, start, n) = self.windows[idx];
        let leaf = self.scan.snapshot.leaf(seg);
        let words = n.div_ceil(64);
        // Which rows of the window are still candidates.
        let mut mask: Option<Vec<u64>> = leaf.live_docs().map(|bits| {
            let from = start / 64;
            let mut m = bits.words()
                [from.min(bits.words().len())..(from + words).min(bits.words().len())]
                .to_vec();
            m.resize(words, 0);
            clear_tail(&mut m, n);
            m
        });
        if let Some(docs) = &self.searched {
            let m = mask.get_or_insert_with(|| full(n));
            intersect_docs(m, docs, leaf.doc_base, start, n);
        }
        let mut scratch = vec![0u64; words];
        for w in windows.values_mut() {
            w.loaded = false;
        }
        if let Some(tests) = self.tests.get(&seg) {
            // A window whose zones all lie outside a comparison's values is
            // not decoded at all.
            for (column, test) in tests {
                if let SegTest::Ranges(set) = test {
                    if let Some(zones) = cols.zones(seg, *column, start, n) {
                        if !zones
                            .iter()
                            .any(|&(lo, hi)| set.iter().any(|iv| iv.overlaps(lo, hi)))
                        {
                            return Ok(None);
                        }
                    }
                }
            }
            for (column, test) in tests {
                let w = windows.entry(*column).or_insert_with(Window::new);
                self.read(cols, seg, *column, start, n, w)?;
                test.apply(w, n, &mut scratch);
                let m = mask.get_or_insert_with(|| full(n));
                for (a, b) in m.iter_mut().zip(&scratch) {
                    *a &= *b;
                }
                if m.iter().all(|&w| w == 0) {
                    return Ok(None);
                }
            }
        }
        let rows = mask.as_ref().map_or(n, |m| popcount(m));
        if rows == 0 {
            return Ok(None);
        }
        self.build(cols, dicts, windows, seg, start, n, mask.as_deref(), rows)
            .map(Some)
    }

    /// The global doc ids the key-column filters select, ascending.
    fn searched(&self) -> Result<Option<Vec<i32>>> {
        let scan = &self.scan;
        let spec = scan.snapshot.spec();
        let mut result: Option<Vec<i32>> = None;
        for cond in &scan.conds {
            let Cond::Search { column, values } = cond else {
                continue;
            };
            let field = spec.columns()[*column].name.clone();
            let mut docs: Vec<i32> = Vec::new();
            for v in values {
                let f = vec![(field.clone(), v.clone())];
                docs.extend(scan.snapshot.with_searcher(|s| matching_docs(s, &f))?);
            }
            docs.sort_unstable();
            docs.dedup();
            result = Some(match result {
                None => docs,
                Some(prev) => prev
                    .into_iter()
                    .filter(|d| docs.binary_search(d).is_ok())
                    .collect(),
            });
        }
        Ok(result)
    }

    /// The doc-values tests of this segment: keyword terms turned into
    /// ordinal intervals.
    fn segment_tests(&self, seg: usize) -> Vec<(usize, SegTest)> {
        let snapshot = &*self.scan.snapshot;
        self.scan
            .conds
            .iter()
            .filter_map(|c| match c {
                Cond::Doc { column, test } => Some((*column, test)),
                Cond::Search { .. } => None,
            })
            .map(|(column, test)| {
                let t = match test {
                    Test::Long(parts) => {
                        let mut set = Vec::new();
                        // A list of `=` is a union; one comparison keeps its
                        // own intervals.
                        for (op, x) in parts {
                            set.extend(intervals(*op, *x));
                        }
                        SegTest::Ranges(coalesce(set))
                    }
                    Test::Terms(parts) => {
                        let terms = snapshot.terms(seg, column);
                        let mut set = Vec::new();
                        for (op, x) in parts {
                            let lo = terms.partition_point(|t| t.as_str() < x.as_str()) as i64;
                            let hi = terms.partition_point(|t| t.as_str() <= x.as_str()) as i64;
                            // Ordinals below `lo` are smaller, from `hi` larger.
                            let n = terms.len() as i64;
                            set.extend(match op {
                                Op::Eq => [Interval::new(lo, hi - 1), Interval::EMPTY],
                                Op::Lt => [Interval::new(0, lo - 1), Interval::EMPTY],
                                Op::Le => [Interval::new(0, hi - 1), Interval::EMPTY],
                                Op::Gt => [Interval::new(hi, n - 1), Interval::EMPTY],
                                Op::Ge => [Interval::new(lo, n - 1), Interval::EMPTY],
                                Op::Ne => [Interval::new(0, lo - 1), Interval::new(hi, n - 1)],
                            });
                        }
                        SegTest::Ranges(coalesce(set))
                    }
                    Test::Null(null) => SegTest::Null(*null),
                };
                (column, t)
            })
            .collect()
    }

    fn read(
        &self,
        cols: &mut Columns<'_>,
        seg: usize,
        column: usize,
        start: usize,
        n: usize,
        w: &mut Window,
    ) -> Result<()> {
        if w.loaded {
            return Ok(());
        }
        w.loaded = true;
        w.vals.resize(n, 0);
        w.present.resize(n.div_ceil(64), 0);
        let from = i32::try_from(start).unwrap_or(i32::MAX);
        w.absent = !cols.window_values(seg, column, from, &mut w.vals, &mut w.present)?;
        if w.absent {
            w.present.fill(0);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        cols: &mut Columns<'_>,
        dicts: &mut HashMap<(usize, usize), Arc<StringArray>>,
        windows: &mut HashMap<usize, Window>,
        seg: usize,
        start: usize,
        n: usize,
        mask: Option<&[u64]>,
        rows: usize,
    ) -> Result<RecordBatch> {
        let scan = &self.scan;
        let spec = scan.snapshot.spec();
        let kept: Option<Vec<usize>> = mask.map(|m| set_bits(m, rows));
        // Stored columns: one stored-fields visit per kept row serves all.
        let stored_cols: Vec<usize> = scan
            .projection
            .iter()
            .copied()
            .filter(|&c| spec.columns()[c].kind == ColumnKind::Key)
            .collect();
        let mut stored: Vec<StringBuilder> = stored_cols
            .iter()
            .map(|_| StringBuilder::with_capacity(rows, rows * 12))
            .collect();
        if !stored_cols.is_empty() {
            let each = |local: usize, stored: &mut Vec<StringBuilder>| -> Result<()> {
                let l = i32::try_from(start + local).unwrap_or(i32::MAX);
                let row = scan.snapshot.stored_strings(seg, l, &stored_cols)?;
                for (b, v) in stored.iter_mut().zip(row) {
                    b.append_option(v);
                }
                Ok(())
            };
            match &kept {
                Some(k) => {
                    for &i in k {
                        each(i, &mut stored)?;
                    }
                }
                None => {
                    for i in 0..n {
                        each(i, &mut stored)?;
                    }
                }
            }
        }
        let mut stored = stored.into_iter();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(scan.projection.len());
        for &c in &scan.projection {
            let kind = &spec.columns()[c].kind;
            if *kind == ColumnKind::Key {
                let mut b = stored.next().unwrap_or_default();
                arrays.push(Arc::new(b.finish()));
                continue;
            }
            let w = windows.entry(c).or_insert_with(Window::new);
            self.read(cols, seg, c, start, n, w)?;
            if *kind == ColumnKind::Long && kept.is_none() && !w.absent {
                // Every row is kept: the decoded values are the array.
                let nulls = validity(&w.present, false, n);
                let vals = std::mem::take(&mut w.vals);
                arrays.push(Arc::new(Int64Array::new(ScalarBuffer::from(vals), nulls)));
                continue;
            }
            if *kind == ColumnKind::Long {
                let (vals, nulls) = select(&w.vals, &w.present, w.absent, n, kept.as_deref());
                arrays.push(Arc::new(Int64Array::new(ScalarBuffer::from(vals), nulls)));
                continue;
            }
            // A keyword: one key per row into a dictionary shared by every
            // batch of the scan when it is small enough, else the segment's.
            let nulls = select_validity(&w.present, w.absent, n, kept.as_deref());
            let (dict, remap) = match self.globals.get(&c) {
                Some(g) => (
                    g.values.clone(),
                    g.remap.get(&seg).map_or(&[][..], Vec::as_slice),
                ),
                None => (
                    dicts
                        .entry((seg, c))
                        .or_insert_with(|| {
                            Arc::new(StringArray::from_iter_values(
                                scan.snapshot.terms(seg, c).iter(),
                            ))
                        })
                        .clone(),
                    &[][..],
                ),
            };
            let last = dict.len().saturating_sub(1);
            let keys: Vec<i32> = if remap.is_empty() {
                gather(&w.vals, n, kept.as_deref(), |o| {
                    i32::try_from(o.min(last)).unwrap_or(0)
                })
            } else {
                let top = remap.len() - 1;
                gather(&w.vals, n, kept.as_deref(), |o| remap[o.min(top)])
            };
            let keys = Int32Array::new(ScalarBuffer::from(keys), nulls);
            arrays.push(if scan.dictionary {
                Arc::new(DictionaryArray::<Int32Type>::try_new(keys, dict).map_err(Error::from)?)
            } else {
                arrow::compute::take(&*dict, &keys, None).map_err(Error::from)?
            });
        }
        let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(rows));
        RecordBatch::try_new_with_options(scan.schema.clone(), arrays, &options)
            .map_err(Error::from)
    }
}

enum SegTest {
    Ranges(Vec<Interval>),
    Null(bool),
}

impl SegTest {
    /// The rows of the window that meet the test, in `out`.
    fn apply(&self, w: &Window, n: usize, out: &mut [u64]) {
        match self {
            SegTest::Ranges(set) => {
                if w.absent {
                    out.fill(0);
                    return;
                }
                mark_any(&w.vals[..n], set, out);
                for (o, p) in out.iter_mut().zip(&w.present) {
                    *o &= *p;
                }
            }
            SegTest::Null(null) => {
                for (i, o) in out.iter_mut().enumerate() {
                    let p = if w.absent { 0 } else { w.present[i] };
                    *o = if *null { !p } else { p };
                }
                clear_tail(out, n);
            }
        }
    }
}

fn full(n: usize) -> Vec<u64> {
    let mut m = vec![u64::MAX; n.div_ceil(64)];
    clear_tail(&mut m, n);
    m
}

fn clear_tail(words: &mut [u64], n: usize) {
    if !n.is_multiple_of(64) {
        if let Some(last) = words.get_mut(n / 64) {
            *last &= (1u64 << (n % 64)) - 1;
        }
    }
    let used = n.div_ceil(64);
    if words.len() > used {
        words[used..].fill(0);
    }
}

fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

/// Indices of the set bits, `count` of them.
fn set_bits(words: &[u64], count: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(count);
    for (i, &w) in words.iter().enumerate() {
        let mut w = w;
        while w != 0 {
            out.push(i * 64 + w.trailing_zeros() as usize);
            w &= w - 1;
        }
    }
    out
}

/// Keeps in `mask` only the rows of the window (documents `start..start +
/// n` of a segment whose first document is `base`) that are in `docs`.
fn intersect_docs(mask: &mut [u64], docs: &[i32], base: i32, start: usize, n: usize) {
    let first = base.saturating_add(i32::try_from(start).unwrap_or(i32::MAX));
    let last = first.saturating_add(i32::try_from(n).unwrap_or(i32::MAX));
    let lo = docs.partition_point(|&d| d < first);
    let hi = docs.partition_point(|&d| d < last);
    let mut hit = vec![0u64; mask.len()];
    for &d in &docs[lo..hi] {
        let i = usize::try_from(d - first).unwrap_or(0);
        hit[i / 64] |= 1 << (i % 64);
    }
    for (m, h) in mask.iter_mut().zip(hit) {
        *m &= h;
    }
}

/// The values and validity of the `kept` rows (every row when `None`).
fn select(
    vals: &[i64],
    present: &[u64],
    absent: bool,
    n: usize,
    kept: Option<&[usize]>,
) -> (Vec<i64>, Option<NullBuffer>) {
    match kept {
        None => {
            let out = vals[..n].to_vec();
            (out, validity(present, absent, n))
        }
        Some(k) => {
            let out: Vec<i64> = k.iter().map(|&i| vals[i]).collect();
            let nulls = if absent {
                Some(NullBuffer::new_null(k.len()))
            } else if k.iter().all(|&i| present[i / 64] >> (i % 64) & 1 == 1) {
                None
            } else {
                Some(NullBuffer::from_iter(
                    k.iter().map(|&i| present[i / 64] >> (i % 64) & 1 == 1),
                ))
            };
            (out, nulls)
        }
    }
}

/// The key of each kept row (every row when `kept` is `None`): `f` of the
/// stored ordinal. An ordinal that is not a valid index (a row without a
/// value holds whatever the decoder left) is clamped by `f`.
fn gather(vals: &[i64], n: usize, kept: Option<&[usize]>, f: impl Fn(usize) -> i32) -> Vec<i32> {
    match kept {
        None => vals[..n].iter().map(|&v| f(v as usize)).collect(),
        Some(k) => k.iter().map(|&i| f(vals[i] as usize)).collect(),
    }
}

/// The validity of the kept rows, as [`select`] computes it for values.
fn select_validity(
    present: &[u64],
    absent: bool,
    n: usize,
    kept: Option<&[usize]>,
) -> Option<NullBuffer> {
    match kept {
        None => validity(present, absent, n),
        Some(k) if absent => Some(NullBuffer::new_null(k.len())),
        Some(k) => {
            let bit = |i: usize| present[i / 64] >> (i % 64) & 1 == 1;
            if k.iter().all(|&i| bit(i)) {
                None
            } else {
                Some(NullBuffer::from_iter(k.iter().map(|&i| bit(i))))
            }
        }
    }
}

fn validity(present: &[u64], absent: bool, n: usize) -> Option<NullBuffer> {
    if absent {
        return Some(NullBuffer::new_null(n));
    }
    let set: usize = present[..n / 64]
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>()
        + if n.is_multiple_of(64) {
            0
        } else {
            (present[n / 64] & ((1u64 << (n % 64)) - 1)).count_ones() as usize
        };
    if set == n {
        None
    } else {
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from_slice_ref(present),
            0,
            n,
        )))
    }
}
