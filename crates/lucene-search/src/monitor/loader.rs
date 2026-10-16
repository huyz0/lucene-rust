//! `org.apache.lucene.monitor.ConcurrentQueryLoader`: registers queries into a
//! monitor from several threads. `add` puts a query on a bounded queue; each
//! worker drains up to its share of the queue (waiting briefly for more) and
//! registers the batch, so a large load commits in batches rather than one
//! query at a time.

use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::monitor::facade::Monitor;
use crate::monitor::index::MonitorQuery;
use crate::{Error, Result};

/// `ConcurrentQueryLoader.DEFAULT_QUEUE_SIZE`.
pub const DEFAULT_QUEUE_SIZE: usize = 2000;

/// How long a worker waits for a batch to fill (`drain`'s 100 ms).
const DRAIN_TIMEOUT: Duration = Duration::from_millis(100);

/// `ConcurrentQueryLoader`.
pub struct ConcurrentQueryLoader {
    sender: Option<SyncSender<MonitorQuery>>,
    workers: Vec<JoinHandle<()>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl ConcurrentQueryLoader {
    /// `new ConcurrentQueryLoader(monitor)`: a worker per available processor.
    pub fn new(monitor: Arc<Mutex<Monitor>>) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        Self::with_threads(monitor, threads, DEFAULT_QUEUE_SIZE)
    }

    /// `new ConcurrentQueryLoader(monitor, threads, queueSize)`.
    pub fn with_threads(monitor: Arc<Mutex<Monitor>>, threads: usize, queue_size: usize) -> Self {
        let threads = threads.max(1);
        let (sender, receiver) = sync_channel(queue_size.max(1));
        let receiver = Arc::new(Mutex::new(receiver));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let share = (queue_size / threads).max(1);
        let workers = (0..threads)
            .map(|_| {
                let (monitor, receiver, errors) = (
                    Arc::clone(&monitor),
                    Arc::clone(&receiver),
                    Arc::clone(&errors),
                );
                std::thread::spawn(move || worker(&monitor, &receiver, &errors, share))
            })
            .collect();
        ConcurrentQueryLoader {
            sender: Some(sender),
            workers,
            errors,
        }
    }

    /// `add(mq)`: blocks while the queue is full. After `close` it is refused.
    pub fn add(&self, query: MonitorQuery) -> Result<()> {
        let Some(sender) = &self.sender else {
            return Err(Error::IllegalState(
                "ConcurrentQueryLoader has been shutdown, cannot add new queries".into(),
            ));
        };
        sender.send(query).map_err(|_| {
            Error::IllegalState(
                "ConcurrentQueryLoader has been shutdown, cannot add new queries".into(),
            )
        })
    }

    /// `close()`: no more queries; waits for the workers to register what is
    /// queued. The registration errors, if any, are reported together.
    pub fn close(&mut self) -> Result<()> {
        self.sender = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        let errors = std::mem::take(&mut *self.errors.lock().unwrap_or_else(|e| e.into_inner()));
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::IllegalState(errors.join("; ")))
        }
    }
}

impl Drop for ConcurrentQueryLoader {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// One worker: drain a batch (up to `share` queries, waiting up to
/// [`DRAIN_TIMEOUT`] for it to fill) and register it, until the queue is
/// closed and empty.
fn worker(
    monitor: &Mutex<Monitor>,
    receiver: &Mutex<Receiver<MonitorQuery>>,
    errors: &Mutex<Vec<String>>,
    share: usize,
) {
    loop {
        let mut batch = Vec::with_capacity(share);
        let mut closed = false;
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        {
            let rx = receiver.lock().unwrap_or_else(|e| e.into_inner());
            while batch.len() < share {
                let left = deadline.saturating_duration_since(Instant::now());
                match rx.recv_timeout(left) {
                    Ok(q) => batch.push(q),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
        }
        if !batch.is_empty() {
            let result = monitor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .register(batch);
            if let Err(e) = result {
                errors
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(e.to_string());
            }
        }
        if closed {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Clause, TermQuery};
    use lucene_analysis::{Analyzer, StandardAnalyzer};
    use std::collections::BTreeMap;

    fn mq(i: usize) -> MonitorQuery {
        MonitorQuery::new(
            format!("q{i}"),
            Clause::Term(TermQuery::new("f", format!("t{i}").into_bytes())),
            Some(format!("t{i}")),
            BTreeMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn every_queued_query_is_registered_by_close() {
        let monitor = Arc::new(Mutex::new(Monitor::new(Arc::new(Analyzer::new(
            StandardAnalyzer::new(),
        )))));
        let mut loader = ConcurrentQueryLoader::with_threads(Arc::clone(&monitor), 4, 16);
        for i in 0..200 {
            loader.add(mq(i)).unwrap();
        }
        loader.close().unwrap();
        assert_eq!(monitor.lock().unwrap().query_count(), 200);
        // after close nothing more is taken
        let e = loader.add(mq(999)).unwrap_err().to_string();
        assert!(e.contains("has been shutdown"), "{e}");
        loader.close().unwrap();
    }

    #[test]
    fn registration_errors_come_back_from_close() {
        let tmp = lucene_util::test_support::TempDir::new("monitor-loader");
        let path: &std::path::Path = &tmp;
        let ser = crate::monitor::serializer::MonitorQuerySerializer::from_parser(Arc::new(|_| {
            Err("never".to_string())
        }));
        {
            // a read-only monitor refuses every registration
            let config = crate::monitor::config::MonitorConfiguration::default()
                .set_directory(path, ser, true);
            let monitor = Monitor::with_configuration(
                Arc::new(Analyzer::new(StandardAnalyzer::new())),
                crate::monitor::presearcher::TermFilteredPresearcher::default(),
                Vec::new(),
                &config,
            )
            .unwrap();
            let mut loader =
                ConcurrentQueryLoader::with_threads(Arc::new(Mutex::new(monitor)), 2, 8);
            for i in 0..5 {
                loader.add(mq(i)).unwrap();
            }
            let e = loader.close().unwrap_err().to_string();
            assert!(e.contains("readOnly"), "{e}");
        }
        // the default constructor starts a worker per processor and drops cleanly
        let monitor = Arc::new(Mutex::new(Monitor::new(Arc::new(Analyzer::new(
            StandardAnalyzer::new(),
        )))));
        let loader = ConcurrentQueryLoader::new(monitor);
        loader.add(mq(1)).unwrap();
    }
}
