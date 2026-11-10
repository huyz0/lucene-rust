//! A multi-vector field's `MaxSim` scores as a values source, for the
//! search crate's rescorers and function queries.

use std::sync::Arc;

use lucene_codecs::doc_values::BinaryReader;
use lucene_codecs::field_infos::DocValuesType;
use lucene_search::rescorer::DoubleValuesSourceRescorer;
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, EmptyDoubleValues, ValuesContext,
};
use lucene_search::{Error, Result};

use crate::score::MaxSimQuery;

/// Each document's `MaxSim` against a query, read from a field's `BINARY`
/// doc values in any [`crate::TokenEncoding`]. A document without a value
/// has none, and a field without `BINARY` doc values has no values at all.
pub struct MaxSimValuesSource {
    field: String,
    query: Arc<MaxSimQuery>,
}

impl MaxSimValuesSource {
    /// `field`'s scores against `query`.
    pub fn new(field: &str, query: MaxSimQuery) -> Self {
        Self {
            field: field.to_string(),
            query: Arc::new(query),
        }
    }
}

struct MaxSimValues<'c> {
    column: BinaryReader<'c>,
    current: Option<&'c [u8]>,
    query: Arc<MaxSimQuery>,
    scratch: Vec<f32>,
}

impl DoubleValues for MaxSimValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.current = self.column.value(doc)?;
        Ok(self.current.is_some())
    }

    fn double_value(&mut self) -> Result<f64> {
        let value = self.current.unwrap_or_default();
        let score = self
            .query
            .score(value, &mut self.scratch)
            .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        Ok(f64::from(score))
    }
}

impl DoubleValuesSource for MaxSimValuesSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.reader(leaf)?;
        let column = reader
            .field_infos()
            .field_by_name(&self.field)
            .filter(|i| i.doc_values_type == DocValuesType::Binary)
            .and_then(|i| {
                let (meta, data) = reader.doc_values_for_field(i.number)?;
                meta.binary_entry(i.number)
                    .map(|e| BinaryReader::new(data, e))
            });
        Ok(match column {
            Some(column) => Box::new(MaxSimValues {
                column,
                current: None,
                query: Arc::clone(&self.query),
                scratch: Vec::new(),
            }),
            None => Box::new(EmptyDoubleValues),
        })
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!(
            "MaxSimValuesSource(field={} queryTokens={} dim={})",
            self.field,
            self.query.len(),
            self.query.dim()
        )
    }

    fn equality_key(&self) -> String {
        format!("{}{:?}", self.describe(), self.query)
    }
}

/// What a rescored hit without a multi-vector scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// `0`, as Lucene's `LateInteractionRescorer.create`. Lucene's scores are
    /// never negative, so a missing hit ranks last there. A raw `MaxSim` is
    /// negative when the query's tokens mostly point away from a document's,
    /// and such a hit ranks below a missing one here.
    Zero,
    /// Its first-pass score, as `LateInteractionRescorer.withFallbackToFirstPassScore`.
    FirstPassScore,
}

/// A rescorer that replaces each hit's score with its `MaxSim` against
/// `query` from `field`.
pub fn max_sim_rescorer(
    field: &str,
    query: MaxSimQuery,
    missing: Missing,
) -> DoubleValuesSourceRescorer {
    let source = Arc::new(MaxSimValuesSource::new(field, query));
    DoubleValuesSourceRescorer::new(
        source,
        Arc::new(move |first: f32, present: bool, value: f64| {
            if present {
                value as f32
            } else if missing == Missing::FirstPassScore {
                first
            } else {
                0.0
            }
        }),
    )
}
