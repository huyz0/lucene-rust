//! What a chunk needs of a buffer's documents besides their doc values and
//! norms, kept as the documents arrive: which fields each run of documents
//! has, and their points packed per field.

use std::collections::HashMap;

use lucene_codecs::points::PackedPointsField;

use crate::index_writer::{encode_live_point, ExplicitFields, IndexingConfig};

/// The documents that have a field, as ascending runs `start..end`.
#[derive(Debug, Clone, Default)]
struct Runs(Vec<(i32, i32)>);

impl Runs {
    fn add(&mut self, doc: i32) {
        match self.0.last_mut() {
            Some((_, end)) if *end == doc => *end = doc.saturating_add(1),
            Some((_, end)) if *end > doc => {}
            _ => self.0.push((doc, doc.saturating_add(1))),
        }
    }

    /// Whether a document of `start..end` is in a run.
    fn any_in(&self, start: i32, end: i32) -> bool {
        let at = self.0.partition_point(|&(_, e)| e <= start);
        self.0.get(at).is_some_and(|&(s, _)| s < end)
    }

    fn ram_bytes(&self) -> usize {
        self.0.capacity().saturating_mul(8)
    }
}

#[derive(Debug, Clone, Default)]
pub(in crate::concurrent_writer) struct ChunkFields {
    present: HashMap<i32, Runs>,
    points: HashMap<i32, PackedPointsField>,
    /// By field: the documents that have a norm for it.
    normed: HashMap<i32, Runs>,
    /// A document stores term vectors, or gave a point the field does not
    /// take: chunks are written from the documents.
    unsupported: bool,
    /// By field: the least and greatest value of each indexed dimension
    /// of its points, kept even when the columns stop.
    bounds: HashMap<i32, PointBounds>,
}

/// A field's points' bounds, as [`ChunkFields::bounds`] gives them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PointBounds {
    /// Each indexed dimension's least value, packed.
    pub min: Vec<u8>,
    /// Each indexed dimension's greatest value, packed.
    pub max: Vec<u8>,
    /// How many points there are.
    pub count: u64,
}

impl PointBounds {
    /// Widens the bounds to take `point`'s first `dims` dimensions of
    /// `width` bytes each.
    fn take(&mut self, point: &[u8], dims: usize, width: usize) {
        let Some(point) = point.get(..dims.saturating_mul(width)) else {
            return;
        };
        if self.count == 0 {
            self.min = point.to_vec();
            self.max = point.to_vec();
        } else {
            for ((v, lo), hi) in point
                .chunks_exact(width)
                .zip(self.min.chunks_exact_mut(width))
                .zip(self.max.chunks_exact_mut(width))
            {
                if v < &*lo {
                    lo.copy_from_slice(v);
                }
                if v > &*hi {
                    hi.copy_from_slice(v);
                }
            }
        }
        self.count = self.count.saturating_add(1);
    }
}

impl ChunkFields {
    /// Adds document `doc`'s fields.
    pub(in crate::concurrent_writer) fn record(
        &mut self,
        doc: i32,
        fields: &ExplicitFields,
        cfg: &IndexingConfig,
    ) {
        if self.unsupported {
            // Without columns the points are encoded for their bounds alone.
            if !fields.points.is_empty() {
                self.bound(fields, cfg);
            }
            return;
        }
        if !fields.term_vectors.is_empty() {
            return self.give_up_with(fields, cfg);
        }
        let mut present = |n: i32| self.present.entry(n).or_default().add(doc);
        for inv in &fields.inverted {
            present(inv.field_number);
        }
        for v in &fields.doc_values {
            present(v.field_number);
        }
        for &n in &fields.registered {
            present(n);
        }
        for p in &fields.points {
            present(p.field_number);
        }
        for inv in &fields.inverted {
            let tv = cfg
                .field_by_number(inv.field_number)
                .is_none_or(|f| f.store_term_vectors);
            if tv {
                return self.give_up_with(fields, cfg);
            }
            if inv.norm.is_some() {
                self.normed.entry(inv.field_number).or_default().add(doc);
            }
        }
        for p in &fields.points {
            let Some(info) = cfg.field_by_number(p.field_number) else {
                return self.give_up_with(fields, cfg);
            };
            let column = self
                .points
                .entry(p.field_number)
                .or_insert_with(|| PackedPointsField {
                    field_number: info.number,
                    num_dims: info.point_dimension_count,
                    num_index_dims: info.point_index_dimension_count,
                    bytes_per_dim: info.point_num_bytes,
                    ..PackedPointsField::default()
                });
            let at = column.values.len();
            if !encode_live_point(info, &p.value, &mut column.values) {
                return self.give_up_with(fields, cfg);
            }
            column.docs.push(doc);
            if let (Ok(dims), Ok(width)) = (
                usize::try_from(info.point_index_dimension_count),
                usize::try_from(info.point_num_bytes),
            ) {
                self.bounds.entry(p.field_number).or_default().take(
                    column.values.get(at..).unwrap_or_default(),
                    dims,
                    width,
                );
            }
        }
    }

    /// [`Self::give_up`] on a document, its points still bounded.
    fn give_up_with(&mut self, fields: &ExplicitFields, cfg: &IndexingConfig) {
        self.give_up();
        self.bound(fields, cfg);
    }

    /// Widens the fields' bounds to take a document's points.
    fn bound(&mut self, fields: &ExplicitFields, cfg: &IndexingConfig) {
        let mut packed = Vec::new();
        for p in &fields.points {
            let Some(info) = cfg.field_by_number(p.field_number) else {
                continue;
            };
            packed.clear();
            if !encode_live_point(info, &p.value, &mut packed) {
                continue;
            }
            let (Ok(dims), Ok(width)) = (
                usize::try_from(info.point_index_dimension_count),
                usize::try_from(info.point_num_bytes),
            ) else {
                continue;
            };
            self.bounds
                .entry(p.field_number)
                .or_default()
                .take(&packed, dims, width);
        }
    }

    /// By field, its points' bounds over every document recorded, for the
    /// fields some document gave a point.
    pub(in crate::concurrent_writer) fn all_bounds(&self) -> &HashMap<i32, PointBounds> {
        &self.bounds
    }

    /// Frees the columns, keeping the bounds.
    pub(in crate::concurrent_writer) fn drop_columns(&mut self) {
        *self = ChunkFields {
            bounds: std::mem::take(&mut self.bounds),
            ..ChunkFields::default()
        };
    }

    /// Whether the columns stopped: the buffer's chunks are decoded.
    pub(in crate::concurrent_writer) fn gave_up(&self) -> bool {
        self.unsupported
    }

    /// Stops keeping columns: the buffer's chunks are decoded.
    pub(in crate::concurrent_writer) fn give_up(&mut self) {
        if self.unsupported {
            return;
        }
        *self = ChunkFields {
            unsupported: true,
            bounds: std::mem::take(&mut self.bounds),
            ..ChunkFields::default()
        };
    }

    /// The runs of documents of `start..end` with a norm for field
    /// `field`, clipped to it.
    pub(in crate::concurrent_writer) fn normed_in(
        &self,
        field: i32,
        start: i32,
        end: i32,
    ) -> impl Iterator<Item = (i32, i32)> + '_ {
        let runs = self.normed.get(&field).map_or(&[][..], |r| &r.0[..]);
        let at = runs.partition_point(|&(_, e)| e <= start);
        runs.get(at..)
            .unwrap_or_default()
            .iter()
            .take_while(move |&&(s, _)| s < end)
            .map(move |&(s, e)| (s.max(start), e.min(end)))
    }

    pub(in crate::concurrent_writer) fn ram_bytes(&self) -> usize {
        let present = self
            .present
            .values()
            .chain(self.normed.values())
            .map(Runs::ram_bytes);
        let points = self.points.values().map(|p| {
            p.docs
                .capacity()
                .saturating_mul(4)
                .saturating_add(p.values.capacity())
        });
        let bounds = self
            .bounds
            .values()
            .map(|b| b.min.capacity().saturating_add(b.max.capacity()));
        present
            .chain(points)
            .chain(bounds)
            .fold(0, usize::saturating_add)
    }

    /// The fields some document of `start..end` has, and with
    /// `with_points` their points numbered from `start`; `None` when the
    /// documents must be decoded.
    #[allow(clippy::type_complexity)]
    pub(in crate::concurrent_writer) fn range(
        &self,
        start: i32,
        end: i32,
        with_points: bool,
    ) -> Option<(Vec<i32>, Vec<PackedPointsField>)> {
        if self.unsupported {
            return None;
        }
        let present: Vec<i32> = self
            .present
            .iter()
            .filter(|(_, r)| r.any_in(start, end))
            .map(|(&n, _)| n)
            .collect();
        let mut points = Vec::new();
        for p in self.points.values().filter(|_| with_points) {
            let width = usize::try_from(p.num_dims.checked_mul(p.bytes_per_dim)?).ok()?;
            let hi = super::doc_values::before_from_end(&p.docs, end);
            let lo =
                super::doc_values::before_from_end(p.docs.get(..hi).unwrap_or_default(), start);
            if lo == hi {
                continue;
            }
            let values = p
                .values
                .get(lo.checked_mul(width)?..hi.checked_mul(width)?)?
                .to_vec();
            points.push(PackedPointsField {
                docs: p
                    .docs
                    .get(lo..hi)?
                    .iter()
                    .map(|&d| d.saturating_sub(start))
                    .collect(),
                values,
                ..header(p)
            });
        }
        Some((present, points))
    }
}

/// `p`'s field, without its points.
fn header(p: &PackedPointsField) -> PackedPointsField {
    PackedPointsField {
        field_number: p.field_number,
        num_dims: p.num_dims,
        num_index_dims: p.num_index_dims,
        bytes_per_dim: p.bytes_per_dim,
        ..PackedPointsField::default()
    }
}
