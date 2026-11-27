//! Reading column values into Arrow arrays: for a list of matched documents,
//! or for a window of consecutive documents of one segment.

use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Int64Array, StringArray, StringBuilder,
};
use datafusion::arrow::datatypes::DataType;
use lucene_codecs::doc_values::NumericReader;
use lucene_search::global_ords::{keyword_column, Ords};

use crate::error::{Error, Result};
use crate::snapshot::Snapshot;
use crate::spec::ColumnKind;

/// Documents per window of a full scan.
pub const WINDOW: usize = 1024;

enum Slot<'s> {
    Unopened,
    Absent,
    Values(Box<NumericReader<'s>>),
}

/// Doc-values readers of a snapshot, opened the first time a segment's column
/// is read.
pub struct Columns<'s> {
    snapshot: &'s Snapshot,
    slots: Vec<Slot<'s>>,
    columns: usize,
}

impl<'s> Columns<'s> {
    /// Readers over `snapshot`; nothing is opened yet.
    pub fn new(snapshot: &'s Snapshot) -> Self {
        let columns = snapshot.spec().columns().len();
        let mut slots = Vec::new();
        slots.resize_with(snapshot.segment_count() * columns, || Slot::Unopened);
        Self {
            snapshot,
            slots,
            columns,
        }
    }

    fn reader(&mut self, segment: usize, column: usize) -> Result<Option<&mut NumericReader<'s>>> {
        let at = segment * self.columns + column;
        if matches!(self.slots[at], Slot::Unopened) {
            self.slots[at] = self.open(segment, column)?;
        }
        Ok(match &mut self.slots[at] {
            Slot::Values(r) => Some(r),
            _ => None,
        })
    }

    fn open(&self, segment: usize, column: usize) -> Result<Slot<'s>> {
        let spec = self.snapshot.spec();
        let col = &spec.columns()[column];
        let leaf = self.snapshot.leaf(segment);
        Ok(match col.kind {
            ColumnKind::Long => {
                let Some(number) = self.snapshot.field_number(segment, column) else {
                    return Ok(Slot::Absent);
                };
                let found = leaf.doc_values_for_field(number).and_then(|(meta, data)| {
                    meta.numeric_entry(number)
                        .map(|entry| NumericReader::new(data, entry))
                });
                found.map_or(Slot::Absent, |r| Slot::Values(Box::new(r)))
            }
            ColumnKind::Keyword => match keyword_column(leaf, &col.name)? {
                Some((Ords::Single(r), _, _)) => Slot::Values(r),
                Some((Ords::Multi(_), _, _)) => {
                    return Err(Error::unsupported(format!(
                        "keyword column `{}` is multi-valued",
                        col.name
                    )))
                }
                _ => Slot::Absent,
            },
            _ => Slot::Absent,
        })
    }

    /// The columns `projection` names, for the matched documents `docs`
    /// (global doc id, score).
    ///
    /// # Errors
    /// Unreadable doc values or stored fields, or a column that cannot be read.
    pub fn for_docs(
        &mut self,
        projection: &[usize],
        docs: &[(i32, Option<f32>)],
    ) -> Result<Vec<ArrayRef>> {
        let spec = self.snapshot.spec().clone();
        let located: Vec<(usize, i32)> = docs
            .iter()
            .map(|d| self.snapshot.locate(d.0))
            .collect::<Result<_>>()?;
        let stored: Vec<usize> = projection
            .iter()
            .copied()
            .filter(|&c| c < spec.columns().len() && spec.columns()[c].kind == ColumnKind::Key)
            .collect();
        // One stored-fields read per document serves every key column.
        let mut stored_rows: Vec<Vec<Option<String>>> = Vec::new();
        if !stored.is_empty() {
            for &(s, l) in &located {
                stored_rows.push(self.snapshot.stored_strings(s, l, &stored)?);
            }
        }
        let mut out = Vec::with_capacity(projection.len());
        for &c in projection {
            if c == spec.score_index() {
                out.push(Arc::new(Float32Array::from_iter(docs.iter().map(|d| d.1))) as ArrayRef);
                continue;
            }
            let col = spec
                .columns()
                .get(c)
                .ok_or_else(|| Error::unsupported(format!("column {c} is outside the table")))?;
            out.push(match col.kind {
                ColumnKind::Key => {
                    let at = stored.iter().position(|&s| s == c).unwrap_or(0);
                    let mut b = StringBuilder::with_capacity(docs.len(), docs.len() * 12);
                    for row in &stored_rows {
                        b.append_option(row[at].as_deref());
                    }
                    Arc::new(b.finish()) as ArrayRef
                }
                ColumnKind::Long => {
                    let mut vals = Vec::with_capacity(docs.len());
                    for &(s, l) in &located {
                        vals.push(match self.reader(s, c)? {
                            Some(r) => r.value(l).map_err(lucene_search::Error::from)?,
                            None => None,
                        });
                    }
                    Arc::new(Int64Array::from(vals)) as ArrayRef
                }
                ColumnKind::Keyword => {
                    let mut vals: Vec<Option<&str>> = Vec::with_capacity(docs.len());
                    for &(s, l) in &located {
                        let ord = match self.reader(s, c)? {
                            Some(r) => r.value(l).map_err(lucene_search::Error::from)?,
                            None => None,
                        };
                        vals.push(ord.and_then(|o| {
                            usize::try_from(o)
                                .ok()
                                .and_then(|o| self.snapshot.terms(s, c).get(o))
                                .map(String::as_str)
                        }));
                    }
                    Arc::new(StringArray::from(vals)) as ArrayRef
                }
                _ => return Err(unreadable(&col.name)),
            });
        }
        Ok(out)
    }

    /// The columns `projection` names for the `n` documents of `segment` from
    /// `start`, leaving out the deleted ones (`live`, when the segment has
    /// deletions). `_score` is NULL: nothing scored these documents.
    ///
    /// # Errors
    /// As [`Self::for_docs`].
    pub fn for_window(
        &mut self,
        projection: &[usize],
        segment: usize,
        start: i32,
        n: usize,
        live: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let spec = self.snapshot.spec().clone();
        let mut out: Vec<ArrayRef> = Vec::with_capacity(projection.len());
        for &c in projection {
            if c == spec.score_index() {
                out.push(Arc::new(Float32Array::from(vec![None; n])) as ArrayRef);
                continue;
            }
            let col = spec
                .columns()
                .get(c)
                .ok_or_else(|| Error::unsupported(format!("column {c} is outside the table")))?;
            let array: ArrayRef = match col.kind {
                ColumnKind::Key => {
                    let mut b = StringBuilder::with_capacity(n, n * 12);
                    for i in 0..n {
                        let local = start.saturating_add(i as i32);
                        let row = self.snapshot.stored_strings(segment, local, &[c])?;
                        b.append_option(row[0].as_deref());
                    }
                    Arc::new(b.finish())
                }
                ColumnKind::Long | ColumnKind::Keyword => {
                    let mut vals = vec![0i64; n];
                    let mut present = vec![0u64; n.div_ceil(64)];
                    let has = match self.reader(segment, c)? {
                        Some(r) => {
                            r.fill_window(start, &mut vals, &mut present)
                                .map_err(lucene_search::Error::from)?;
                            true
                        }
                        None => false,
                    };
                    let is_set = |i: usize| has && present[i / 64] >> (i % 64) & 1 == 1;
                    if col.kind == ColumnKind::Long {
                        Arc::new(Int64Array::from_iter(
                            (0..n).map(|i| is_set(i).then_some(vals[i])),
                        ))
                    } else {
                        let terms = self.snapshot.terms(segment, c);
                        Arc::new(StringArray::from_iter((0..n).map(|i| {
                            if is_set(i) {
                                usize::try_from(vals[i])
                                    .ok()
                                    .and_then(|o| terms.get(o))
                                    .map(String::as_str)
                            } else {
                                None
                            }
                        })))
                    }
                }
                _ => return Err(unreadable(&col.name)),
            };
            out.push(array);
        }
        if let Some(live) = live {
            out = out
                .iter()
                .map(|a| datafusion::arrow::compute::filter(a, live).map_err(Error::from_arrow))
                .collect::<Result<_>>()?;
        }
        Ok(out)
    }
}

fn unreadable(name: &str) -> Error {
    Error::unsupported(format!(
        "column `{name}` is a search column: use it in match_text or knn, it has no values to read"
    ))
}

/// The Arrow type of a column read by [`Columns`].
pub fn arrow_type(kind: &ColumnKind) -> Option<DataType> {
    match kind {
        ColumnKind::Key | ColumnKind::Keyword => Some(DataType::Utf8),
        ColumnKind::Long => Some(DataType::Int64),
        _ => None,
    }
}
