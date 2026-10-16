//! `MonitorConfiguration`, `MonitorUpdateListener`, `QueryTimeListener` and
//! `SlowLog`: how a monitor is set up and observed.

use std::path::PathBuf;
use std::time::Duration;

use crate::monitor::analyzer::QueryDecomposer;
use crate::monitor::index::MonitorQuery;
use crate::monitor::serializer::MonitorQuerySerializer;

/// `MonitorConfiguration`: the settings of a monitor, with Lucene's defaults.
#[derive(Debug, Clone)]
pub struct MonitorConfiguration {
    /// `queryUpdateBufferSize`: queries registered per commit (5000).
    pub query_update_buffer_size: usize,
    /// `purgeFrequency` (5 minutes).
    pub purge_frequency: Duration,
    pub query_decomposer: QueryDecomposer,
    /// `serializer`: needed to persist and reload queries.
    pub serializer: Option<MonitorQuerySerializer>,
    /// `readOnly`: open a persisted index without writing to it.
    pub read_only: bool,
    /// Where the query index lives; `None` is in memory (Java's default
    /// `ByteBuffersDirectory`).
    pub index_path: Option<PathBuf>,
}

impl Default for MonitorConfiguration {
    fn default() -> Self {
        MonitorConfiguration {
            query_update_buffer_size: 5000,
            purge_frequency: Duration::from_secs(5 * 60),
            query_decomposer: QueryDecomposer,
            serializer: None,
            read_only: false,
            index_path: None,
        }
    }
}

impl MonitorConfiguration {
    /// `setIndexPath(path, serializer)`.
    pub fn set_index_path(
        mut self,
        path: impl Into<PathBuf>,
        serializer: MonitorQuerySerializer,
    ) -> Self {
        self.index_path = Some(path.into());
        self.serializer = Some(serializer);
        self
    }

    /// `setDirectoryProvider(provider, serializer, readOnly)`, for a path.
    pub fn set_directory(
        mut self,
        path: impl Into<PathBuf>,
        serializer: MonitorQuerySerializer,
        read_only: bool,
    ) -> Self {
        self.index_path = Some(path.into());
        self.serializer = Some(serializer);
        self.read_only = read_only;
        self
    }

    /// `setPurgeFrequency(frequency, units)`.
    pub fn set_purge_frequency(mut self, frequency: Duration) -> Self {
        self.purge_frequency = frequency;
        self
    }

    /// `setQueryUpdateBufferSize(size)`.
    pub fn set_query_update_buffer_size(mut self, size: usize) -> Self {
        self.query_update_buffer_size = size;
        self
    }
}

/// `MonitorUpdateListener`: told of changes to the query index. Every method
/// has Java's empty default.
pub trait MonitorUpdateListener: Send + Sync {
    /// `afterUpdate(updates)`.
    fn after_update(&self, _updates: &[MonitorQuery]) {}
    /// `afterDelete(queryIds)`.
    fn after_delete(&self, _query_ids: &[String]) {}
    /// `afterClear()`.
    fn after_clear(&self) {}
    /// `onPurge()`.
    fn on_purge(&self) {}
    /// `onPurgeError(t)`.
    fn on_purge_error(&self, _error: &str) {}
}

/// `QueryTimeListener`: told how long each query took to match.
pub trait QueryTimeListener {
    /// `logQueryTime(queryId, timeInNanos)`.
    fn log_query_time(&mut self, query_id: &str, nanos: u64);
}

/// `SlowLog`: the queries a timing listener found slow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SlowLog {
    entries: Vec<SlowLogEntry>,
}

/// `SlowLog.Entry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlowLogEntry {
    pub query_id: String,
    pub time: u64,
}

impl SlowLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// `addQuery(query, time)`.
    pub fn add_query(&mut self, query_id: impl Into<String>, time: u64) {
        self.entries.push(SlowLogEntry {
            query_id: query_id.into(),
            time,
        });
    }

    /// `addAll(queries)`.
    pub fn add_all(&mut self, entries: impl IntoIterator<Item = SlowLogEntry>) {
        self.entries.extend(entries);
    }

    /// `iterator()`.
    pub fn iter(&self) -> std::slice::Iter<'_, SlowLogEntry> {
        self.entries.iter()
    }
}

impl std::fmt::Display for SlowLog {
    /// `toString()`: `id [<time>ns]` per line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in &self.entries {
            writeln!(f, "{} [{}ns]", e.query_id, e.time)?;
        }
        Ok(())
    }
}

impl QueryTimeListener for SlowLog {
    /// A slow log is a listener that keeps everything it is told.
    fn log_query_time(&mut self, query_id: &str, nanos: u64) {
        self.add_query(query_id, nanos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::serializer::MonitorQuerySerializer;
    use std::sync::Arc;

    #[test]
    fn the_defaults_are_lucenes_and_the_setters_chain() {
        let c = MonitorConfiguration::default();
        assert_eq!(c.query_update_buffer_size, 5000);
        assert_eq!(c.purge_frequency, Duration::from_secs(300));
        assert!(!c.read_only && c.index_path.is_none() && c.serializer.is_none());
        let ser = MonitorQuerySerializer::from_parser(Arc::new(|_| Err("no".into())));
        let c = MonitorConfiguration::default()
            .set_index_path("/tmp/q", ser.clone())
            .set_purge_frequency(Duration::from_secs(1))
            .set_query_update_buffer_size(7);
        assert_eq!(
            c.index_path.as_deref(),
            Some(std::path::Path::new("/tmp/q"))
        );
        assert_eq!(
            (c.query_update_buffer_size, c.purge_frequency),
            (7, Duration::from_secs(1))
        );
        let c = c.set_directory("/tmp/r", ser, true);
        assert!(c.read_only && c.serializer.is_some());
    }

    #[test]
    fn a_slow_log_keeps_what_it_is_told_and_prints_it() {
        let mut log = SlowLog::new();
        log.add_query("a", 1500);
        log.log_query_time("b", 20);
        log.add_all(vec![SlowLogEntry {
            query_id: "c".into(),
            time: 3,
        }]);
        assert_eq!(log.iter().count(), 3);
        assert_eq!(log.to_string(), "a [1500ns]\nb [20ns]\nc [3ns]\n");
    }

    #[test]
    fn a_listener_has_empty_defaults() {
        struct Quiet;
        impl MonitorUpdateListener for Quiet {}
        let l = Quiet;
        l.after_update(&[]);
        l.after_delete(&["a".to_string()]);
        l.after_clear();
        l.on_purge();
        l.on_purge_error("x");
    }
}
