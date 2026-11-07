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
    /// A document stores term vectors, gave a point the field does not
    /// take, or a norm of zero: chunks are written from the documents.
    unsupported: bool,
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
            return;
        }
        if !fields.term_vectors.is_empty() {
            return self.give_up();
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
            if tv || inv.norm == Some(0) {
                return self.give_up();
            }
        }
        for p in &fields.points {
            let Some(info) = cfg.field_by_number(p.field_number) else {
                return self.give_up();
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
            if !encode_live_point(info, &p.value, &mut column.values) {
                return self.give_up();
            }
            column.docs.push(doc);
        }
    }

    /// Stops keeping columns: the buffer's chunks are decoded.
    fn give_up(&mut self) {
        *self = ChunkFields {
            unsupported: true,
            ..ChunkFields::default()
        };
    }

    pub(in crate::concurrent_writer) fn ram_bytes(&self) -> usize {
        let present = self.present.values().map(Runs::ram_bytes);
        let points = self.points.values().map(|p| {
            p.docs
                .capacity()
                .saturating_mul(4)
                .saturating_add(p.values.capacity())
        });
        present.chain(points).fold(0, usize::saturating_add)
    }

    /// The fields some document of `start..end` has, and their points
    /// numbered from `start`; `None` when the documents must be decoded.
    #[allow(clippy::type_complexity)]
    pub(in crate::concurrent_writer) fn range(
        &self,
        start: i32,
        end: i32,
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
        for p in self.points.values() {
            let width = usize::try_from(p.num_dims.checked_mul(p.bytes_per_dim)?).ok()?;
            let lo = p.docs.partition_point(|&d| d < start);
            let hi = p.docs.partition_point(|&d| d < end);
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
