//! `MemoryTracker` and `CollectorMemoryTracker`: accounting for the transient
//! objects a collector allocates, with a limit.

use std::sync::atomic::{AtomicI64, Ordering};

use crate::{Error, Result};

/// `org.apache.lucene.misc.util.MemoryTracker`: tracks dynamic allocations of
/// transient objects.
pub trait MemoryTracker {
    /// `updateBytes(bytes)`: adds `bytes` (negative to release) to the total.
    ///
    /// # Errors
    /// Whatever the tracker refuses, `CollectorMemoryTracker` its limit.
    fn update_bytes(&self, bytes: i64) -> Result<()>;

    /// `getBytes()`: the bytes currently tracked.
    fn bytes(&self) -> i64;
}

/// `CollectorMemoryTracker`: the default `MemoryTracker`, failing once the
/// total passes `memory_limit` or goes below zero.
#[derive(Debug)]
pub struct CollectorMemoryTracker {
    name: String,
    memory_usage: AtomicI64,
    memory_limit: i64,
}

impl CollectorMemoryTracker {
    /// `new CollectorMemoryTracker(name, memoryLimit)`.
    pub fn new(name: &str, memory_limit: i64) -> Self {
        Self {
            name: name.to_string(),
            memory_usage: AtomicI64::new(0),
            memory_limit,
        }
    }
}

impl MemoryTracker for CollectorMemoryTracker {
    /// The usage is updated before it is checked, as in Java: a refused update
    /// stays applied.
    fn update_bytes(&self, bytes: i64) -> Result<()> {
        // `AtomicLong.addAndGet` wraps.
        let current = self
            .memory_usage
            .fetch_add(bytes, Ordering::SeqCst)
            .wrapping_add(bytes);
        if current > self.memory_limit {
            return Err(Error::IllegalState(format!(
                "Memory limit exceeded for {}",
                self.name
            )));
        }
        if current < 0 {
            return Err(Error::IllegalState(format!(
                "Illegal Memory State for {}",
                self.name
            )));
        }
        Ok(())
    }

    fn bytes(&self) -> i64 {
        self.memory_usage.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_up_to_the_limit_and_releases() {
        let t = CollectorMemoryTracker::new("c", 100);
        t.update_bytes(60).unwrap();
        t.update_bytes(40).unwrap();
        assert_eq!(t.bytes(), 100);
        t.update_bytes(-100).unwrap();
        assert_eq!(t.bytes(), 0);
    }

    #[test]
    fn passing_the_limit_is_refused_but_stays_applied() {
        let t = CollectorMemoryTracker::new("big", 10);
        let err = t.update_bytes(11).unwrap_err();
        assert!(matches!(&err, Error::IllegalState(m) if m == "Memory limit exceeded for big"));
        assert_eq!(t.bytes(), 11);
    }

    #[test]
    fn going_below_zero_is_refused() {
        let t = CollectorMemoryTracker::new("neg", 10);
        let err = t.update_bytes(-1).unwrap_err();
        assert!(matches!(&err, Error::IllegalState(m) if m == "Illegal Memory State for neg"));
        assert_eq!(t.bytes(), -1);
    }
}
