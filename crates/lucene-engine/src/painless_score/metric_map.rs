//! A `scripted_metric`'s `map_script`: it runs once per matched document
//! and mutates the one `state` the shard shares, across every segment the
//! aggregation visits. It returns no values; the aggregation reads the
//! `state` afterwards ([`PainlessMetricMap::finish`]) and gives it to the
//! `combine_script`.
//!
//! The state lives between segments as a plain [`Constant`] and is a
//! script value only while a segment runs, so what a segment leaves in it
//! must be storable as one (`null`, booleans, numbers, strings, lists and
//! string-keyed maps), as `lucene_painless::ingest::from_value` reads it.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use lucene_aggs::bucket_aggs::{ScriptValues, ScriptValuesSource};
use lucene_painless::ingest::{from_value, to_value};
use lucene_painless::ir::{self, Constant};
use lucene_search::{Error, Result};

use super::{scripts, Leaf, ScriptScores, Spec};

/// The largest `state` a shard may hand its `combine_script`, as the bytes
/// of its encoding: the state is what a shard returns, so it is bounded
/// like any shard result.
pub const MAX_STATE_BYTES: usize = 16 << 20;

#[derive(Debug)]
struct Shared {
    state: Constant,
    /// A segment holds the state: another may not run until it is dropped.
    open: bool,
    error: Option<String>,
}

/// A `map_script` over the shard's `state`, which starts as what the
/// `init_script` left.
#[derive(Debug)]
pub struct PainlessMetricMap {
    spec: Arc<Spec>,
    shared: Arc<Mutex<Shared>>,
}

impl PainlessMetricMap {
    /// The map script `spec` (a `metric_map` context, checked) over the
    /// initial `state`.
    ///
    /// # Errors
    /// When `spec` is another context's, or the runtime does not run it as
    /// Painless would.
    pub fn new(spec: Spec, state: Constant) -> Result<Self> {
        if !spec.metric_map {
            return Err(Error::IllegalArgument(
                "scripted_metric map_script with another context's script".to_string(),
            ));
        }
        spec.check()
            .map_err(|r| Error::IllegalArgument(format!("painless script not checked: {r}")))?;
        Ok(Self {
            spec: Arc::new(spec),
            shared: Arc::new(Mutex::new(Shared {
                state,
                open: false,
                error: None,
            })),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The `state` the segments run so far left, once none is running.
    ///
    /// # Errors
    /// When a segment left a state that cannot be stored, or one over
    /// [`MAX_STATE_BYTES`].
    pub fn finish(&self) -> Result<Constant> {
        let shared = self.lock();
        if let Some(e) = &shared.error {
            return Err(Error::IllegalArgument(e.clone()));
        }
        let bytes = ir::encode_constant(&shared.state).len();
        if bytes > MAX_STATE_BYTES {
            return Err(Error::IllegalArgument(format!(
                "scripted_metric state is {bytes} bytes, over the limit of {MAX_STATE_BYTES}"
            )));
        }
        Ok(shared.state.clone())
    }
}

impl ScriptValuesSource for PainlessMetricMap {
    fn open<'a>(
        &self,
        reader: &'a lucene_search::directory_reader::SegmentReader,
    ) -> Result<Box<dyn ScriptValues + 'a>> {
        let mut shared = self.lock();
        if shared.open {
            return Err(Error::IllegalArgument(
                "scripted_metric map_script ran on two segments at once".to_string(),
            ));
        }
        let mut leaf = scripts(&self.spec, reader, Leaf::Base(reader.doc_base), None)?;
        leaf.env.state = to_value(&shared.state);
        shared.open = true;
        Ok(Box::new(MapLeaf {
            leaf,
            shared: Arc::clone(&self.shared),
        }))
    }
}

/// The map script on one segment.
struct MapLeaf<'c> {
    leaf: ScriptScores<'c>,
    shared: Arc<Mutex<Shared>>,
}

impl ScriptValues for MapLeaf<'_> {
    fn values(&mut self, doc: i32, _out: &mut Vec<i64>) -> Result<()> {
        self.leaf.env.doc = doc;
        self.leaf.run_object().map(drop)
    }

    fn mapped(&mut self, _doc: i32, _inputs: &[f64], _out: &mut Vec<i64>) -> Result<()> {
        Ok(())
    }
}

impl Drop for MapLeaf<'_> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        shared.open = false;
        match from_value(&self.leaf.env.state) {
            Ok(state) => shared.state = state,
            Err(e) => shared.error = Some(format!("scripted_metric state: {e}")),
        }
    }
}
